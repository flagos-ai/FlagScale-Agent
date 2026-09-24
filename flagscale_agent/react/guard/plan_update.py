# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PlanUpdateGuard — flags a POSSIBLE stall and prompts a self-check.

Many iterations (or long wall-clock) on the SAME step without a plan_update is a
PROXY for a stall, not proof of one — an agent may be legitimately progressing
(many productive steps) or running a valid long op. So the guard does NOT assert
"you are stalled"; it fires a checkpoint that hands the verdict to the agent via a
falsifiable self-check: can you name, for each recent round, one concrete new fact
(three-part: assumption / how tested / result)? If yes → progressing, record it and
continue. If no (rounds ending "nothing new") → information gain is zero, the real
stall; escape DOWNWARD (smaller experiment) or UPWARD (different method-class),
never SIDEWAYS (another variant).

Firing signal: TIME only — wall-clock since last plan_update, every
TIME_REMIND_SECONDS. Catches a step that has run long without recorded progress;
the remedy is to convert deliberation into a real observation (or, if waiting on
in-flight background jobs, record that and continue).

Why TIME only (concurrency): a tool-call COUNT used to be a second trigger, but
once the agent can run jobs in parallel, one wall-clock window carries many more
tool calls — poll/wait on each in-flight job — so a count threshold fires on
legitimate parallel supervision rather than on a stall. Wall-clock is invariant to
concurrency, so it is the trustworthy proxy. The tool-call count is still SHOWN in
the reminder as context, but it never gates firing.

Escalation: every ESCALATE_AFTER-th reminder becomes a BLOCK
instead of inject, then the counter resets: inject, inject, block, repeat. A block
forces a substantive plan_update (or override) before continuing. Counters reset
on plan_update/plan_create. Meta tools (plan_status, evict, memory_read, ...) don't
count and don't tick.

Side-channel guard: on deliverable+threshold tasks, a fact only clears a block if
OBSERVED AT THE DELIVERABLE (edit + re-run + measure), not derived in scratch/prototype/
on paper. This wires PRINCIPLE 2 into the stall clearance bar.

Mechanism limit: guards run in check_post (AFTER a tool call). If the agent goes
silent with no tool calls, the guard never fires until the next call. In practice
thrash always emits tool calls, so the signal still fires.
"""

from __future__ import annotations

import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


def _extract_recent_activity(messages, limit: int = 8, lookback: int = 30) -> str:
    """Collect the agent's recent activity trace for the loop judge.

    Walks back through at most ``lookback`` messages, gathering, in chronological
    order, the agent's own narration (assistant text) and the tools it invoked
    (name + the single most identifying argument — a shell command or a file path).
    This is what the LLM judge reads to decide whether the agent is looping in one
    method-class or has actually escaped downward/upward.

    Intentionally content-agnostic: it does not parse or count anything itself
    (no signatures, no regex, no thresholds) — it only assembles the trace and
    hands the semantic judgment to the judge, per the design constraint.
    """
    if not messages:
        return ""
    entries: list[str] = []
    for msg in reversed(messages[-lookback:]):
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                text = " ".join(p for p in parts if p)
            # Tool calls invoked in this assistant turn (name + key arg).
            calls = msg.get("tool_calls") or []
            call_descs: list[str] = []
            for tc in calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {}) or {}
                key_arg = (
                    args.get("command")
                    or args.get("path")
                    or args.get("url")
                    or ""
                )
                key_arg = str(key_arg).strip().replace("\n", " ")
                if len(key_arg) > 160:
                    key_arg = key_arg[:160] + "…"
                call_descs.append(f"{name}({key_arg})" if key_arg else name)
            line = ""
            if text.strip():
                line = text.strip()
            if call_descs:
                line = (line + " " if line else "") + "→ " + ", ".join(call_descs)
            if line:
                entries.append(line)
        if len(entries) >= limit:
            break
    return "\n".join(reversed(entries))


class PlanUpdateGuard(Guard):
    """Detects being stuck on one plan step and prompts self-diagnosis.

    Tracks iterations since the last plan_update while a step is active. Many
    iterations on the same step without a plan_update is treated as a stall
    signal, and the guard nudges the agent to stop and diagnose (escape
    downward/upward, not sideways) rather than merely to bookkeep the plan.
    """

    name = "plan_update"
    priority = 50

    # Wall-clock stall signal: remind every this many seconds on the same step.
    # This is the ONLY firing trigger — tool-call count no longer gates (see module
    # docstring: concurrency inflates the count with legitimate parallel job
    # supervision, so only wall-clock cleanly proxies a stall).
    TIME_REMIND_SECONDS = 180
    # Every ESCALATE_AFTER-th stall reminder
    # escalates from advisory inject to a blocking verdict, then the counter
    # resets and the cycle repeats: inject, inject, block, inject, inject, block.
    # An advisory nudge can be read and ignored; a block cannot — it forces the
    # agent to actually stop and plan_update (or justify an override) before
    # continuing. We only escalate after ESCALATE_AFTER-1 ignored advisories,
    # so a block always means "gentle reminders demonstrably had no effect".
    ESCALATE_AFTER = 3

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "plan_status",  # Read-only
        "evict", "recall",
        "memory_read", "memory_list",
    ))

    # plan_update actions that are ALWAYS a substantive response: they mutate the
    # plan's real state (a step advanced/finished/abandoned, subtasks added,
    # acceptance refined, a batch of steps updated). Any of these earns a full
    # reset — including the escalation counter — because the agent genuinely moved
    # the plan forward, not merely pinged the guard.
    _PROGRESS_ACTIONS = frozenset((
        "step_done", "step_skip", "complete", "abandon",
        "add_steps", "update_acceptance", "batch",
    ))

    @staticmethod
    def _supplies_thinking(ctx) -> bool:
        """True iff THIS call supplies a fresh plan-level problem model.

        Detected from tool_args (not plan state) so it keys off what the agent
        actively did in this call: action='set_thinking', or any plan_update /
        plan_create carrying a non-empty ``thinking``. This is the "rebuilt the
        model" signal — the deliberate/deep-thinking move, as opposed to a
        retrospective note or a fast action.
        """
        args = getattr(ctx, "tool_args", None) or {}
        if args.get("action") == "set_thinking":
            return True
        t = args.get("thinking")
        return bool(t and str(t).strip())

    @classmethod
    def _clears_escalated_block(cls, ctx) -> bool:
        """Stricter bar than _is_substantive_update, for an ESCALATED block only.

        Once the guard has escalated to a block, the agent has already ignored
        multiple advisory nudges — the situation is a demonstrated loop. A bare
        retrospective note ("I confirmed X is better") is exactly the shallow
        rationalization that let the thrash resume, so notes ALONE no longer
        clear the block. Real cognition must have moved, shown by EITHER:
          - a genuine progress action (step done/skipped, plan complete/abandon,
            steps added, acceptance refined, batch) — plan state actually moved; or
          - a rebuilt plan-level problem MODEL (a fresh ``thinking`` with a
            falsifiable prediction) — deliberate thinking actually happened.
        plan_create (a whole new plan) also clears.
        """
        if getattr(ctx, "tool_name", "") == "plan_create":
            return True
        if getattr(ctx, "tool_name", "") != "plan_update":
            return False
        args = getattr(ctx, "tool_args", None) or {}
        if args.get("action", "") in cls._PROGRESS_ACTIONS:
            return True
        return cls._supplies_thinking(ctx)

    @classmethod
    def _is_substantive_update(cls, ctx) -> bool:
        """Distinguish a real plan response from an empty guard-clearing ping.

        The stall block is meant to be cleared by genuinely responding — marking
        a step done, or recording a concrete fact in notes. But the reset was
        unconditional, so a no-op ``plan_update(action='step_doing')`` with NO
        notes (same step, no info) also wiped the escalation counter, handing the
        agent a clean slate to resume the exact thrash the block was catching.

        A response is substantive iff it changes plan state or records info:
        - ``plan_create`` — a whole new plan.
        - a progress action (see ``_PROGRESS_ACTIONS``) — real state mutation.
        - any other action (step_doing/reactivate/deactivate) ONLY if it carries
          non-empty notes — i.e. it actually recorded something.

        This is domain-agnostic: it keys off the action type and notes presence,
        never off task content. It preserves the block's INTENDED clearance path
        (plan_update whose note names a concrete fact) while closing the empty-
        ping hole.
        """
        if ctx.tool_name == "plan_create":
            return True
        if ctx.tool_name != "plan_update":
            return False
        args = ctx.tool_args or {}
        action = args.get("action", "")
        if action in cls._PROGRESS_ACTIONS:
            return True
        notes = args.get("notes") or ""
        return bool(notes.strip())

    # Prepended to the block body when the LLM judge decides the agent is looping.
    _LOOP_ESCAPE_NOTE = (
        "\n\n⚠️ The recent trace looks like a LOOP: you keep editing the same "
        "target and re-running the whole program, without isolating a smaller unit "
        "or switching method-class. Re-running the entire thing is SIDEWAYS — it "
        "does not tell you WHICH assumption is wrong. Escape now:\n"
        "— DOWNWARD: write the smallest possible test that exercises ONE unit in "
        "isolation (a single function, one instruction, one syscall, one case). If "
        "it passes, the bug is in integration/scale; if it fails, the bug is in that "
        "unit — either way you learn something a full rerun cannot tell you.\n"
        "— UPWARD: reach outside the current tactic — consult documentation "
        "(web_fetch) or a reference implementation / library for the problem class, "
        "or adopt a different tool. If you have been hand-rolling something a "
        "standard component already does, switch to it.\n"
        "A minimal isolating test or a genuine method-class switch clears this — "
        "another edit-and-rerun of the whole program does not."
    )

    def _loop_diagnosis(self, ctx) -> str:
        """Return the loop-escape note iff the LLM judge sees a sideways loop.

        Delegates the semantic judgment to ``ctx.classify_fn`` (the LLM judge),
        feeding it the recent activity trace. Returns an empty string when no judge
        is available or the judge says the agent is NOT looping — so the block
        degrades gracefully to its prior behavior. No parsing/counting/regex here.
        """
        classify_fn = getattr(ctx, "classify_fn", None)
        if classify_fn is None:
            return ""
        activity = _extract_recent_activity(getattr(ctx, "messages", None))
        if not activity.strip():
            return ""
        try:
            looping = classify_fn(
                "agent_stuck_in_sideways_loop", {"activity": activity}, default=False
            )
        except Exception:
            return ""
        return self._LOOP_ESCAPE_NOTE if looping else ""

    def __init__(self, task_plan, time_fn=time.monotonic):
        self._task_plan = task_plan
        self._iters_since_update = 0
        # Injected for testability; defaults to a monotonic wall clock.
        self._time_fn = time_fn
        # PERIOD anchor: decides WHEN the time signal fires. Re-anchored on every
        # fire so reminders are periodic (once per TIME_REMIND_SECONDS window).
        # Lazily initialized on first counting tool call (None = not yet anchored).
        self._time_anchor = None
        # STALL-START anchor: the wall-clock moment the current stall began (first
        # counting tool call after a plan_update). Unlike _time_anchor it is NOT
        # re-anchored on fire, so it measures the TOTAL time stuck on the step —
        # the quantity the reminder should display, cumulative like the count n.
        # Reset only on plan_update/plan_create. None = not yet anchored.
        self._stall_start = None
        # How many stall reminders (all TIME-triggered) have fired since the last
        # plan_update — the agent perceives "I've been nudged N times". The
        # first ESCALATE_AFTER-1 fire as inject (advisory); the ESCALATE_AFTER-th
        # escalates to a block, then this resets to 0 (cycle repeats). A block is
        # only reached after the agent has IGNORED that many advisory nudges — so
        # it stays a backstop for "reminder had no effect", never the common case.
        self._stall_trigger_count = 0
        # Sticky-block latch. Set True the moment a stall reminder ESCALATES to a
        # block; cleared ONLY by a substantive plan response. While set, an empty
        # guard-clearing ping (plan_update with no state change and no notes)
        # cannot pass — it is re-blocked. This is what makes the block's clearance
        # bar real: the block says "name a concrete fact or mark the step done",
        # and this latch enforces that the NEXT plan_update actually does one of
        # those, instead of a no-op ping buying a clean slate.
        self._block_pending = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check if plan needs updating after tool execution."""
        active_plan = self._task_plan.get_active()
        if not active_plan:
            return None

        steps = active_plan.get("steps", [])
        if not steps:
            return None

        # Plan touched — decide whether it was a SUBSTANTIVE response or an empty
        # guard-clearing ping. Both reset the count/time anchors (so the guard
        # does not immediately re-fire on the very next tool call — that would be
        # noise). But ONLY a substantive response resets the escalation counter:
        # a bare plan_update(step_doing) with no notes must NOT buy back a clean
        # slate, or the block is trivially gamed by pinging the guard and resuming
        # the same thrash. Preserving _stall_trigger_count across empty pings means
        # the escalation tier survives, so the next fire can still BLOCK — forcing
        # a real response (a step marked done, or a concrete fact in notes).
        if ctx.tool_name in ("plan_update", "plan_create"):
            substantive = self._is_substantive_update(ctx)
            # Clearance bar depends on tier. An ESCALATED block (_block_pending)
            # means the agent already ignored multiple advisory nudges — a proven
            # loop — so the bar is the STRICTER _clears_escalated_block: a genuine
            # progress action OR a rebuilt problem model (fresh `thinking` with a
            # prediction). A bare retrospective note no longer clears it, because
            # "I confirmed X is better" is exactly the shallow rationalization that
            # let the thrash resume. Below the block tier, any substantive update
            # (including a concrete note) still resets the timers as before.
            clears = self._clears_escalated_block(ctx)
            if self._block_pending and not clears:
                return GuardVerdict.block(
                    message=(
                        "[PlanUpdate] This plan_update does not clear the ESCALATED "
                        "stall block. You have already ignored several advisory nudges "
                        "on this step — that is a demonstrated loop, so a bare note no "
                        "longer clears it (a retrospective \"I confirmed X is better\" is "
                        "the exact shallow rationalization that lets the thrash resume). "
                        "Real cognition must have MOVED. Clear it with EITHER:\n"
                        "  (a) a genuine progress action — mark the stalled step "
                        "done/skipped, or complete/abandon the plan (plan state actually "
                        "moves); OR\n"
                        "  (b) a REBUILT problem model — plan_update(action='set_thinking', "
                        "thinking=...) (or any action carrying `thinking=`) that names, for "
                        "the WHOLE task: the real bottleneck, the load-bearing hypothesis, "
                        "the evidence, and a FALSIFIABLE prediction for your next move "
                        "('if I change X, the metric should reach ~Y'). If you cannot state "
                        "a NEW prediction, that inability IS the proof you are looping on "
                        "one method-class — switch class (DOWNWARD: isolate one unit; "
                        "UPWARD: a different technique / web_fetch the standard method), "
                        "do not ping again.\n"
                        "If the task requires an output that does not exist on disk yet, "
                        "BUDGET ORDER: write the crudest complete-but-valid version to the "
                        "exact path now, then resume."
                    ),
                    reason="shallow_update_does_not_clear_escalated_block",
                    category="plan_update",
                )
            self._iters_since_update = 0
            self._time_anchor = self._time_fn()
            self._stall_start = self._time_anchor
            # Below the block tier, a substantive update resets the escalation
            # counter (as before). When a block IS pending, only a stricter
            # clearance (progress action or rebuilt model) lifts it — reached here
            # only when `clears` is True.
            if self._block_pending:
                if clears:
                    self._stall_trigger_count = 0
                    self._block_pending = False
            elif substantive:
                self._stall_trigger_count = 0
            return None

        # Meta tools don't count and don't tick the clock.
        if ctx.tool_name in self._META_TOOLS:
            return None

        self._iters_since_update += 1
        # Anchor the clock on the first counting tool call after a reset.
        now = self._time_fn()
        if self._time_anchor is None:
            self._time_anchor = now
        if self._stall_start is None:
            self._stall_start = now

        # n is retained only as a DISPLAY quantity ("N tool calls elapsed") — it no
        # longer GATES firing. Concurrency changed what a tool-call count means: once
        # the agent can run jobs in parallel, one wall-clock window carries many more
        # calls (poll/wait on each in-flight job), so a count threshold fires on
        # legitimate parallel supervision, not on a stall. Wall-clock is invariant to
        # concurrency, so TIME is the only trustworthy stall proxy — fire on it alone.
        n = self._iters_since_update
        # TIME signal: fire once per TIME_REMIND_SECONDS window. Re-anchor on fire
        # so the next window starts fresh (periodic wall-clock reminders).
        elapsed = now - self._time_anchor
        fire_time = elapsed >= self.TIME_REMIND_SECONDS
        if fire_time:
            self._time_anchor = now

        if fire_time:
            doing_steps = [s for s in steps if s.get("status") == "doing"]
            if doing_steps:
                step_id = doing_steps[0].get("id")
                # Describe whichever signal(s) fired, so the agent sees the actual
                # runtime evidence (many actions, or long wall-clock, or both).
                # Display the TOTAL time stuck on the step (cumulative from
                # _stall_start), NOT the periodic-window elapsed used to DECIDE
                # firing — the window resets on every fire, so it would under-report
                # the real stall duration and mismatch the cumulative count n.
                total_stuck = now - self._stall_start
                mins = int(total_stuck // 60)
                # Only the TIME signal gates firing now; the count is shown as
                # context, never as the trigger.
                sig = f"~{mins} min elapsed ({n} tool calls)"

                # Count this reminder. Every ESCALATE_AFTER-th one escalates to a
                # block; then reset so the cycle repeats (inject, inject, block).
                self._stall_trigger_count += 1
                escalate = self._stall_trigger_count % self.ESCALATE_AFTER == 0

                # TIME is the only firing signal. Wall-clock elapsed on one step
                # without recorded progress is the stall proxy; the remedy is to
                # convert deliberation into a real observation. (The tool-call count
                # is no longer a trigger — under concurrency it inflates with
                # legitimate parallel job supervision, so it cannot distinguish a
                # stall from healthy multi-job waiting.)
                signal_hint = (
                    "— TIME signal: wall-clock has elapsed on this step without "
                    "recorded progress. If you are waiting on in-flight background "
                    "jobs, that is legitimate — record it in a note and continue. "
                    "Otherwise, more deliberation in place will not move you; convert "
                    "the next thought into an OBSERVATION — run the smallest experiment "
                    "that returns a REAL output and read it. Every further action without a new "
                    "hypothesis has expected information gain ~= 0 — you are paying the same "
                    "price to re-learn what you already know."
                )

                body = (
                    f"[PlanUpdate] {sig} on step {step_id} with no plan update. Elapsed "
                    f"wall-clock is only a PROXY for a stall — it does NOT prove you are "
                    f"stuck; it is a checkpoint that says STOP and FIND OUT. Run this "
                    f"self-check now (you decide the verdict, not the clock):\n\n"
                    f"  Can you name, for EACH of the last few rounds, what did the last "
                    f"round tell you that you did not already know — one concrete new "
                    f"fact in THREE-PART form: (1) the assumption you tested, (2) HOW you "
                    f"tested it (command/probe/measurement), (3) the RESULT "
                    f"(value/output/conclusion)?\n\n"
                    f"  → YES, every round produced a new fact: you are PROGRESSING, not "
                    f"stalled. Record the latest fact in a one-line plan_update note and "
                    f"continue — this check is cheap by design and must not derail real "
                    f"progress.\n\n"
                    f"  → NO — recent rounds ended \"nothing new\" / \"same as expected\" "
                    f"/ \"still don't know why\": your information gain is zero. THAT is "
                    f"the real stall, and it is not a phrasing problem — \"I confirmed "
                    f"assumption X\" is the empty answer (it names no test, no result). "
                    f"You cannot fix this by thinking harder in the same place; only a "
                    f"new fact moves you.\n\n"
                    f"{signal_hint}\n\n"
                    f"Escape routes (pick one):\n"
                    f"— DOWNWARD: smallest experiment isolating which assumption is "
                    f"wrong.\n"
                    f"— UPWARD: switch to a different method-class.\n"
                    f"— NOT another variant of what failed (parallelizing, retuning, "
                    f"faster rewrite are all SIDEWAYS — same class).\n"
                    f"— THIRD escape: if rounds are oscillating — a metric wobbling "
                    f"around a plateau, each ending \"slightly worse, revert\" — you "
                    f"are burning budget on variance, not progressing. Stop tuning, "
                    f"deliver the best version you already measured, mark done.\n"
                    f"— FOURTH escape: if the task names a required output that does not "
                    f"exist on disk yet, STOP perfecting — BUDGET ORDER: write the "
                    f"crudest complete-but-valid version to the exact path now, confirm "
                    f"it exists, then resume improving. A rough answer that exists "
                    f"beats a perfect one that never got written.\n"
                    f"Record your shift in notes (\"tried X → failed because Y → now "
                    f"trying Z\") so you don't re-walk the dead path. If the step is "
                    f"genuinely finished or abandoned, mark it done/skipped."
                )

                if escalate:
                    # Advisory nudges were ignored this many times — force a stop.
                    self._block_pending = True
                    # LLM-judge loop diagnosis: does the recent activity show the
                    # agent repeating one method-class (edit-same-file + rerun-whole-
                    # program) instead of escaping downward/upward? If so, name the
                    # loop concretely in the block so the escape is unmistakable.
                    # Semantic judgment is delegated to the judge — no signatures,
                    # counting, or regex here (design constraint). Gracefully skips
                    # when no judge is wired.
                    loop_note = self._loop_diagnosis(ctx)
                    return GuardVerdict.block(
                        message=(
                            body
                            + loop_note
                            + f"\n\nThat is {self._stall_trigger_count} stall "
                            f"reminders with no plan_update — earlier advisories had no "
                            f"effect, so this one BLOCKS. Because you have now ignored "
                            f"several nudges, the clearance bar is RAISED: a bare "
                            f"retrospective note no longer clears this block (\"I confirmed "
                            f"X is better\" is the exact shallow rationalization that lets "
                            f"the thrash resume). Real cognition must have MOVED. Clear it "
                            f"with EITHER:\n"
                            f"  (a) a genuine PROGRESS action — mark the stalled step "
                            f"done/skipped, or complete/abandon (plan state actually moves; "
                            f"if oscillating around a plateau, ship the best version you "
                            f"already measured and mark done); OR\n"
                            f"  (b) a REBUILT problem model — plan_update(action='set_thinking', "
                            f"thinking=...) (or any action carrying `thinking=`) naming, for "
                            f"the WHOLE task: the real bottleneck, the load-bearing "
                            f"hypothesis, the evidence, and a FALSIFIABLE prediction for your "
                            f"next move ('if I change X, the metric should reach ~Y'). This "
                            f"is the HOME for deep thinking as opposed to another fast tweak. "
                            f"If you cannot state a NEW prediction, that inability IS the "
                            f"proof you are looping on one method-class — switch class "
                            f"(DOWNWARD: isolate one unit; UPWARD: a different technique / "
                            f"web_fetch the standard method), do not ping again; or (c) "
                            f"override with _override_reason.\n"
                            f"For deliverable+threshold tasks: your model/prediction only "
                            f"counts if it will be TESTED AT THE DELIVERABLE — edit it, "
                            f"re-run, measure score before/after. A prediction validated only "
                            f"in a scratch script or on paper is side-channel work — if the "
                            f"deliverable's score hasn't moved, you are optimizing the wrong "
                            f"medium. Land the next change in the deliverable and measure it."
                        ),
                        reason="repeated_stall_ignored",
                        category="plan_update",
                    )
                return GuardVerdict.inject(
                    message=body,
                    reason="possible_stall",
                    category="plan_update",
                )

        return None
