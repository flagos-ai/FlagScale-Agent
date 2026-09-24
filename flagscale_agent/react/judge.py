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

"""LLM Judge — cached LLM classification for shell safety and health monitoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ── Prompts ───────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPTS: dict[str, str] = {
    "is_error": """\
Is this stderr output a REAL training error that should stop monitoring?

Stderr content:
{text}

Context: {context}

Answer YES for:
- Python tracebacks (Traceback, RuntimeError, CUDA error, OOM)
- NCCL failures (NCCL error, timeout, connection refused)
- Segfaults, killed signals, abort
- Repeated assertion failures

Answer NO for:
- Deprecation/Future/UserWarnings
- torch.cuda.amp deprecation notices
- OMP_NUM_THREADS, thread settings
- wandb version notices
- Informational messages printed to stderr
- Single-line warnings without stack traces
Reply ONLY: {{"real": true/false}}""",

    "is_fatal": """\
Is this shell command FATALLY DESTRUCTIVE — causing IRREVERSIBLE catastrophic damage?

Command: {command}

Answer YES ONLY for commands that would:
- Destroy entire filesystems (rm -rf /, rm -rf /*, rm -rf ~)
- Format/overwrite disks (mkfs, dd if=/dev/zero of=/dev/sd*)
- Fork bombs (:(){ :|:& };:)
- Wipe all data on the system
- Brick the operating system

Answer NO for: targeted file deletion, package removal, process killing, git operations,
permission changes — these are risky but NOT catastrophic.
Reply ONLY: {{"real": true/false}}""",

    "is_dangerous": """\
Is this shell command DANGEROUS and should be BLOCKED?

Command: {command}

Answer YES for: rm -rf on system paths (/ or ~), chmod 777 on system dirs,
fork bombs, mkfs, dd without clear target, redirects to /dev/sd*.
Answer NO for: normal file operations, package management, regular shell commands.
Reply ONLY: {{"real": true/false}}""",

    "reason_lacks_observation": """\
An agent is about to mark a task COMPLETE. This is its stated reason.

Reason:
{reason}

Question: does this reason report an OBSERVATION — something the agent actually
RAN and READ (a command/test executed, an output or value read, a comparison to a
known/expected/ground-truth answer) — or is it only an ARGUMENT that explains why
the approach OUGHT to be right?

Answer YES (a real problem, block) when the reason ONLY argues the method —
"reasonable", "should generalize", "conservative", "I checked the logic", asserts
the output is correct — without stating what was run and what was read.
Answer NO (fine, allow) when it reports a concrete run/read/compare, OR when it
honestly states there is nothing runnable to check.
Reply ONLY: {{"real": true/false}}""",

    "reason_overfits_sample": """\
An agent is about to mark a task COMPLETE. This is its stated reason.

Reason:
{reason}

Question: the agent may have verified only on the ONE sample it could see —
tuning constants or assumptions until that visible case came out right — while the
real grade lands on inputs it never saw. Passing the visible sample is the FLOOR.

Answer YES (a real problem, block) when the reason reports work confined to the
single development sample, or admits fitting/tuning/hardcoding to it, with NO sign
the method was exercised on a different / hidden / held-out / perturbed input or
its value-universe explored.
Answer NO (fine, allow) when it shows generalization — a stress/perturbed input, a
held-out input, enumerated distinct values, an invariant checked — or when no
fitting is involved at all.
Reply ONLY: {{"real": true/false}}""",

    "reason_discloses_substitution": """\
An agent is about to mark a task COMPLETE. This is its stated reason.

Reason:
{reason}

Question: the task may name a GIVEN — an exact value it must hit (a version,
timestamp, size, filename, commit). Disclosing that a near-equivalent was used
does NOT legalize delivering it; the honest outcome is either the exact named
artifact or a BLOCKED report.

Answer YES (a real problem, block) when the reason discloses DELIVERING a
substitute for a GIVEN — near-equivalent / successor / different version / "close
enough" — without reporting inability (BLOCKED, did not deliver, abandoned).
Answer NO (fine, allow) when it reports the exact named thing, reports BLOCKED /
inability, or no substitution is involved.
Reply ONLY: {{"real": true/false}}""",

    "agent_stuck_in_sideways_loop": """\
An agent has been flagged as stalled on one task step — many tool calls with no
plan progress. Below is a trace of what it did most recently (its own narration of
each action and the tools it invoked).

Recent activity:
{activity}

Question: is the agent LOOPING — repeating the SAME method-class over and over —
rather than escaping the stall? The two genuine escapes are:
- DOWNWARD: shrink the problem to the smallest unit that isolates ONE assumption
  (test a single function/instruction/case in isolation), instead of re-running the
  whole program end-to-end each time.
- UPWARD: switch to a different method-class (reach for a library/tool/standard
  technique, consult documentation, change the approach itself).

A LOOP looks like: editing the same file then re-running the same whole program,
again and again; tweaking one value and re-running the full pipeline; trying
variant after variant of the same tactic (a faster rewrite, a different constant,
a reorder) — all SIDEWAYS moves within one method-class. The tell is that each
round re-runs the entire thing rather than isolating a single unit, and no new
method-class or smaller experiment appears.

Answer YES (a real problem, block) when the recent activity is dominated by
re-running the same whole program and editing the same target, with no smaller
isolating experiment and no switch of method-class.
Answer NO (fine, allow) when the agent has already gone DOWNWARD (built a minimal
isolating test) or UPWARD (switched method-class, consulted a reference, adopted a
different tool), or when the activity is varied enough that it is clearly not
looping.
Reply ONLY: {{"real": true/false}}""",
}


_HEALTH_PROMPT = """\
You are monitoring a running shell command. Analyze its status and decide
whether it should continue or be terminated.

Command: {command}
Total elapsed: {elapsed}
Output changed since last check: {output_changed}
Consecutive checks with no output change: {stall_count}
Recent output:
{output}
{expectation_block}{activity_block}
## Context-driven judgment

You have been given (when available) the container's resources (memory, CPU count),
the command's prior history in this session (how many times similar commands ran
and their outcomes), and the output pattern (continuous, intermittent, silent).

USE THIS CONTEXT to calibrate your judgment — do NOT apply fixed time thresholds.
A 4GB container importing a large library may run silent for minutes legitimately;
the same silence on a 64GB container running `ls` is immediately suspicious. A
command that was killed 3 times before with OOM on the same prefix needs a
fundamentally different approach, not another retry. A "silent" output pattern
for an import or model-loading command is normal; the same pattern for a command
that should be printing progress is a red flag.

Judge reasonableness from the command TYPE, the container RESOURCES, and the
command HISTORY — not from a stopwatch.

## Silence is not by itself a stall

A command producing no stdout/stderr is NOT evidence that it is stuck. Many
healthy workloads run silent for long stretches: progress printing turned off
or set to quiet, output fully buffered until completion, or a single
compute-bound library/native call that returns only when done. Treat a rising
"no output change" counter as a REASON TO LOOK CLOSER, never as a kill trigger
on its own. Before reading silence as a stall you must have a POSITIVE sign of
being stuck — negligible CPU with no live child work, a deadlock/error
signature in the output, or an explicit hard limit below being exceeded. If the
live resource signals (when provided) show sustained CPU or live children, the
process is computing, not hung — let it run. Do NOT invent a short "monitoring
window" or deadline that the command must emit output within; no such limit
exists.

## Kill criteria

Kill immediately if:
- Repeated error messages or crash signatures in output
- Network failures with no retry mechanism
- Deadlock indicators (process stuck after error, infinite retry loops)
- OOM indicators: process killed with Error 137 (SIGKILL from Linux OOM killer),
  repeated "Killed" messages, "Out of memory" errors, "MemoryError" exceptions,
  or "cannot allocate memory" messages. For parallel compilation (make -j N),
  if you see Killed/137 errors, the build needs lower parallelism (-j1 or -j2)
  to stay within memory bounds — kill and redirect to reduce -j value.
- Unbounded computation whose own progress counter is not advancing: the output
  prints a total work size (e.g. "0/1048576", "seed 0 of N", "processed 0 of ...")
  and that counter has NOT moved across several consecutive checks. This is not
  the patient case like COMPILING or INSTALLING (which show no counter but are
  making forward progress) — a stalled counter on a compute loop means the method
  is too slow to finish in any reasonable budget, typically a brute-force
  enumeration over a space the task's own numbers make intractable, where an
  analytic shortcut was intended. Do NOT wait it out. Kill so the agent is freed
  to re-derive a smarter approach from the task's constraints rather than blocking
  on a search that will never complete in time.
- Progress IS advancing, but the observed RATE or a printed ETA implies the work
  cannot finish within the remaining budget. This is distinct from the stalled
  counter above: here the counter moves and any metric may even improve, yet the
  rate is so low that the projected completion time exceeds the time the command
  will be allowed to run, so it will be killed by an external timeout before it
  produces its required output. Judge this from the output's own signals — a
  printed ETA longer than the plausible remaining budget, or a completion
  percentage advancing so little per check that linear extrapolation overshoots
  the budget. Before killing on this criterion you MUST rule out a genuine
  HARDWARE or NETWORK hard limit: a job that is legitimately I/O- or
  bandwidth-bound, waiting on a slow-but-progressing transfer, or already
  saturating the resources it was given is NOT a candidate, because it is already
  as fast as the machine allows and no change the agent makes would speed it up.
  Kill ONLY when the slowness is a CONFIGURABLE CHOICE under the agent's control —
  an oversized setting, an unnecessarily expensive method, under-used parallelism,
  or a workload scaled far beyond what the task requires — such that a cheaper
  configuration would finish in budget. When you cannot tell whether the cause is
  a hardware limit or a config choice, do NOT kill: prefer an advisory (kill=false
  with a reason) that prompts the agent to compare its own observed rate/ETA
  against the budget and reconsider its configuration.

When uncertain about install/compile: do NOT kill — a healthy-but-silent build
or install is normal; prefer an advisory (kill=false) and let the fixed-cadence
heartbeat keep watching.
When uncertain about network: KILL — network hangs don't self-resolve.

## Detaching to the background (a third option besides continue/kill)

Some commands are HEALTHY and making genuine forward progress, yet are simply
LONG — a model training run, a large-but-advancing build, a long download whose
bytes are steadily climbing, a compute job whose own counter is moving. Killing
these would throw away good work; but blocking the agent on them wastes the
agent's own working time while it just waits.

For exactly this case, recommend DETACHING the command to the background instead.
When detached, the command keeps running as a background job the agent can poll
or wait on later (via the shell_jobs tool), while the agent is freed NOW to do
other useful work in parallel. Nothing is lost and no time is wasted.

Recommend background (action="background") when ALL of these hold:
- the command is healthy — no crash/error/OOM/deadlock signatures, and (when
  activity is provided) resources show it is actively working, and
- it is making real forward progress — output or a progress counter is advancing,
  or it is a known long-running class (training, large build, big transfer), and
- it will PLAUSIBLY FINISH if left alone (this is the opposite of the "rate too
  slow to ever finish in budget" kill case) — it is just slow enough that the
  agent should not sit idle blocking on it.

Do NOT recommend background when the command is actually STUCK (that is a kill),
nor when it is short enough that waiting the few remaining checks is fine
(that is continue / kill=false). Background is specifically for "healthy, will
finish, but long enough that the agent's time is better spent elsewhere".

## Writing the kill reason (when kill=true)

A kill frees the agent, but the agent will simply relaunch a near-identical
command unless the reason redirects it. So on kill, the reason must be
actionable, not merely descriptive. State, in this order:
1. the failure category you observed (e.g. stalled compute loop, unretried
   network hang, repeated crash),
2. why continuing down this line cannot succeed within budget, and
3. what CLASS of alternative to pursue instead — derived from the failure
   category, not the specific command (e.g. "switch from exhaustive search to
   an analytic/constraint-based derivation", "fix the fault before retrying,
   don't rerun as-is", "reduce the problem size or precompute").
Do NOT prescribe the exact command; name the direction and let the agent
design it. Keep it to one or two sentences.

The monitor is a safety net, never the objective. NEVER write a reason that
tells the agent to shrink the work itself so it fits the monitor — do not
suggest reducing the input size, cutting the amount of work, lowering the
quality/accuracy target the task set, or emitting token output just to look
alive. Those trade away the task's actual requirements to satisfy a heuristic,
which is exactly backwards. When the true problem is that a CONFIGURATION is
needlessly expensive (an oversized setting, an unnecessarily costly method,
under-used parallelism), redirect to a genuinely cheaper or faster METHOD that
still meets every requirement the task stated — never to a weaker deliverable.
If you cannot name a faster method that preserves the task's stated targets,
prefer an advisory (kill=false) over killing.

## Reply format

Reply ONLY a JSON object with an "action" and a "reason":
{{"action": "continue"|"kill"|"background", "reason": "..."}}
- "continue": healthy, let it keep running under the monitor (advisory reason ok).
- "kill": terminate now (reason MUST be actionable, per the section above).
- "background": healthy but long — detach so the agent is freed while it finishes.
For backward compatibility you may instead reply {{"kill": true/false, "reason": "..."}};
kill=true is treated as action="kill", kill=false as action="continue"."""


# ── Judge ─────────────────────────────────────────────────────────────────────

class Judge:
    """Cached LLM classification for shell safety and health monitoring."""

    def __init__(self, provider):
        self.provider = provider
        self._cache: dict[str, Any] = {}

    def classify(self, category: str, context: dict, default: Any = False) -> Any:
        """Classify via LLM. Returns bool for safety categories."""
        cache_key = self._key(category, context)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_template = _CLASSIFY_PROMPTS.get(category)
        if not prompt_template:
            return default

        prompt = prompt_template
        for key, val in context.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(val))

        data = self._call_and_parse(prompt)
        result = self._extract_bool(data, default)
        self._cache[cache_key] = result
        return result

    def health(
        self, command: str, recent_output: str, elapsed: str,
        output_changed: bool = True, stall_count: int = 0,
        expectation: str = "", activity: str = "",
        command_history: str = "", container_resources: str = "",
        output_pattern: str = "", task_budget: str = "",
    ) -> dict:
        """Evaluate whether a long-running command is healthy.

        expectation: optional free-text anchor the agent declared BEFORE running
        this command — what outcome / completion time / result quality it expects.
        When present, the judge additionally checks whether the observed run is
        drifting away from that declared expectation (running far slower than
        expected, or a headline metric plateauing below the expected level) and,
        if so, prompts the agent to re-examine its theory rather than push another
        same-class variant. When empty, behavior is identical to no-anchor mode.

        activity: optional free-text summary of the process's live resource
        signals (CPU%, memory, live child count). When present, the judge can
        distinguish a genuinely HUNG process (no output AND no CPU/child work)
        from a HEALTHY SILENT one (no output but sustained CPU/child work — a
        compute-bound library call or a run with progress printing suppressed).
        When empty, behavior is identical to the no-activity mode.

        command_history: optional summary of prior runs of similar commands
        (same prefix), including how many times they were run, their durations,
        and outcomes (completed, killed, OOM). This lets the judge recognize
        repeated failures and escalate its recommendation rather than treating
        each run in isolation. When empty, behavior is identical to no-history mode.

        container_resources: optional summary of available system resources
        (memory GB, CPU count, swap). This helps the judge calibrate expectations
        — a 4GB container's import will be slower than a 64GB one's. When empty,
        behavior is identical to no-resource mode.

        output_pattern: optional classification of output trend — "continuous"
        (steady stream), "intermittent" (bursts with gaps), or "silent" (no
        output at all). This helps the judge distinguish a healthy silent
        compute from a truly stuck process. When empty, behavior is identical
        to no-pattern mode.

        task_budget: optional free-text summary of the WHOLE-TASK wall-clock
        budget — how long the entire task has been running so far (cumulative
        across every command, not just this one) versus the total time the task
        is allowed before the external harness terminates it. This is distinct
        from the per-command rate/ETA criterion, which only asks whether THIS
        command finishes inside ITS own timeout. The task-budget view lets the
        judge notice that the cumulative session is consuming most of its total
        allowance while the current method-class is unlikely to complete the
        remaining work in the remaining time, and — only when the slowness is a
        configurable choice, never a hardware/network hard limit — emit an
        advisory to switch to a faster method-class that still meets every task
        requirement. When empty, behavior is identical to no-budget mode.
        """
        prompt = _HEALTH_PROMPT.format(
            command=command, elapsed=elapsed,
            output=recent_output[-2000:],
            output_changed="yes" if output_changed else "no",
            stall_count=stall_count,
            expectation_block=self._build_expectation_block(expectation),
            activity_block=self._build_activity_block(activity),
        )
        # Append optional context blocks (byte-identical to old prompt when empty)
        if command_history:
            prompt += self._build_history_block(command_history)
        if container_resources:
            prompt += self._build_resources_block(container_resources)
        if output_pattern:
            prompt += self._build_pattern_block(output_pattern)
        if task_budget:
            prompt += self._build_task_budget_block(task_budget)
        decision = self._call_and_parse(prompt) or {}
        return self._normalize_health_decision(decision)

    @staticmethod
    def _normalize_health_decision(decision) -> dict:
        """Normalize a health decision into {action, kill, reason}.

        Accepts either the new schema ({"action": "continue"|"kill"|"background"})
        or the legacy schema ({"kill": true/false}). Always returns all three
        fields so downstream callers (the shell monitor) can branch on `action`
        while old code that reads `kill` still works. On an empty/garbled reply
        (LLM error, bad JSON), defaults to the safe no-op: continue/kill=False.
        """
        if not isinstance(decision, dict):
            return {"action": "continue", "kill": False, "reason": ""}
        reason = decision.get("reason", "") or ""
        action = decision.get("action")
        if isinstance(action, str):
            action = action.strip().lower()
        if action not in ("continue", "kill", "background"):
            # Fall back to the legacy kill flag.
            action = "kill" if decision.get("kill") else "continue"
        return {"action": action, "kill": action == "kill", "reason": reason}

    @staticmethod
    def _build_history_block(command_history: str) -> str:
        """Build the command-history section injected into the health prompt.

        Returns empty string when command_history is falsy, so the health
        prompt is byte-identical to the pre-history version (backward compat).
        """
        if not command_history or not command_history.strip():
            return ""
        return (
            "\n## Command history\n\n"
            "Similar commands (same prefix) have been run before in this session:\n\n"
            f"    {command_history}\n\n"
            "Use this to recognize patterns: if the same type of command was\n"
            "killed repeatedly before, the environment itself may be broken —\n"
            "recommend the agent stop retrying variants and try a fundamentally\n"
            "different approach or skip this dependency. If prior runs completed\n"
            "successfully, a current stall is more concerning than for a first run.\n"
        )

    @staticmethod
    def _build_resources_block(container_resources: str) -> str:
        """Build the container-resources section injected into the health prompt.

        Returns empty string when container_resources is falsy, so the health
        prompt is byte-identical to the pre-resources version (backward compat).
        """
        if not container_resources or not container_resources.strip():
            return ""
        return (
            "\n## Container resources\n\n"
            "The container's available system resources:\n\n"
            f"    {container_resources}\n\n"
            "Use this to calibrate expectations: a 4GB container will be slower\n"
            "for memory-intensive operations (large imports, parallel compiles)\n"
            "than a 64GB one. A command that seems slow on a small container may\n"
            "be perfectly normal — do not kill it just for being slow when the\n"
            "container is resource-constrained. Conversely, high parallelism on\n"
            "a small container risks OOM — advise reducing -j value.\n"
        )

    @staticmethod
    def _build_pattern_block(output_pattern: str) -> str:
        """Build the output-pattern section injected into the health prompt.

        Returns empty string when output_pattern is falsy, so the health
        prompt is byte-identical to the pre-pattern version (backward compat).
        """
        if not output_pattern or not output_pattern.strip():
            return ""
        return (
            "\n## Output pattern\n\n"
            f"    {output_pattern}\n\n"
            "Use this to judge silence: 'continuous' output that stopped is more\n"
            "concerning than 'silent' output that never started (some commands\n"
            "like imports produce no stdout by design). 'intermittent' output\n"
            "with long gaps is normal for batch processing.\n"
        )

    @staticmethod
    def _build_expectation_block(expectation: str) -> str:
        """Build the declared-anchor section injected into the health prompt.

        Returns "" when no anchor was declared, so the prompt is byte-identical
        to the pre-anchor version and no task-specific content leaks in.
        """
        anchor = (expectation or "").strip()
        if not anchor:
            return ""
        return (
            "\n## Declared expectation (anchor)\n\n"
            "Before starting, the operator declared what this command is expected\n"
            "to achieve — its intended outcome, a rough completion time, and/or the\n"
            "result quality it should reach:\n\n"
            f"    {anchor}\n\n"
            "Treat this anchor as a hypothesis to test against the live evidence,\n"
            "NOT as ground truth. Compare the observed run to it along two axes:\n"
            "- TIME: does the observed rate / printed ETA imply completion far\n"
            "  later than the declared expectation? (only meaningful if the anchor\n"
            "  states or implies a time budget)\n"
            "- QUALITY: does a headline metric in the output appear to plateau\n"
            "  below the declared target across several checks, while the attempts\n"
            "  driving it stay within one method-class (same approach, only knobs\n"
            "  turned)?\n\n"
            "If the run is drifting clearly away from the anchor on either axis, do\n"
            "NOT kill on that basis alone — instead emit an advisory (kill=false)\n"
            "whose reason names the specific gap between observed and expected and\n"
            "asks the operator to re-examine whether the current METHOD-CLASS can\n"
            "reach the anchor at all, versus turning another knob on the same one.\n"
            "Only escalate to kill if a separate hard kill criterion above is also\n"
            "met. If the observed run is consistent with the anchor, ignore this\n"
            "section. A mismatch may equally mean the anchor was wrong — say so\n"
            "rather than forcing the run to fit it.\n"
        )

    @staticmethod
    def _build_activity_block(activity: str) -> str:
        """Build the live-resource-signal section injected into the health prompt.

        Returns "" when no activity summary was supplied, so the prompt is
        byte-identical to the pre-activity version and no content leaks in.
        """
        signal = (activity or "").strip()
        if not signal:
            return ""
        return (
            "\n## Live resource signals\n\n"
            "Alongside the text output, the process's real resource usage right\n"
            "now is:\n\n"
            f"    {signal}\n\n"
            "Use this to tell apart two cases that look identical on stdout alone:\n"
            "- HUNG: no output AND negligible CPU AND no live child work — the\n"
            "  process is genuinely stuck/waiting. This is a real concern.\n"
            "- HEALTHY-SILENT: no output BUT sustained CPU and/or live children —\n"
            "  the process is actively computing with its progress printing simply\n"
            "  turned off or buffered (a compute-bound library call, a quiet\n"
            "  training/build phase). This is NORMAL, not a stall. Do NOT kill it,\n"
            "  and do NOT treat rising stall_count as evidence against it — silence\n"
            "  with live compute is expected. There is no hidden per-command\n"
            "  'monitoring window' or short deadline it must produce output within;\n"
            "  the only time-based hard limit is the explicit overall timeout named\n"
            "  in the kill criteria above.\n"
        )

    @staticmethod
    def _build_task_budget_block(task_budget: str) -> str:
        """Build the whole-task wall-clock budget section for the health prompt.

        Returns "" when no budget summary was supplied, so the prompt is
        byte-identical to the pre-budget version and no content leaks in.

        This block is a FUEL GAUGE, not a driver. It reports the objective
        cumulative wall-clock fact (total allowance vs. spent so far) and asks
        the agent to think carefully in light of it. It deliberately does NOT
        judge whether the current method-class can still finish, does NOT
        recommend switching methods, and does NOT steer the approach — those are
        the agent's own decisions. The health judge's only authority is the
        binary liveness/kill call; deciding HOW to drive with the remaining time
        belongs to the agent, which has the full task context the judge lacks.
        """
        summary = (task_budget or "").strip()
        if not summary:
            return ""
        return (
            "\n## Task-level time budget (cumulative wall-clock) — FUEL GAUGE ONLY\n\n"
            "Beyond this one command, the WHOLE TASK runs under a fixed overall\n"
            "wall-clock budget enforced by an external harness. The cumulative\n"
            "time spent so far across every command, versus that total budget, is:\n\n"
            f"    {summary}\n\n"
            "Treat this strictly as a FUEL-GAUGE READING you pass through to the\n"
            "agent — how much time is total, how much is already spent. It is NOT\n"
            "a kill criterion and NOT a reason to change your liveness verdict:\n"
            "never kill on the basis of the task budget alone (escalate to kill\n"
            "only if a separate hard kill criterion above is independently met).\n\n"
            "You are the fuel gauge, not the driver. Do NOT judge whether the\n"
            "current approach can still finish in the time left, do NOT recommend\n"
            "switching methods or method-classes, do NOT suggest shrinking the\n"
            "task, and do NOT otherwise steer HOW the agent should proceed —\n"
            "those decisions require the full task context you do not have, and\n"
            "they belong to the agent alone.\n\n"
            "When the budget is worth surfacing, emit an advisory (kill=false)\n"
            "whose reason does exactly two things and nothing more: (1) state the\n"
            "objective budget fact (total vs. spent), and (2) remind the agent to\n"
            "think carefully and deliberately in light of the remaining time\n"
            "before its next move. Phrase (2) as a prompt to reflect — e.g.\n"
            "'given the remaining budget, think carefully about how best to\n"
            "proceed' — never as a specific instruction on WHAT to do.\n"
        )

    def reset_turn(self):
        """Clear per-turn cache."""
        self._cache.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_bool(data: Any, default: Any) -> Any:
        """Extract boolean from LLM response dict."""
        if not isinstance(data, dict):
            return default
        real = data.get("real")
        if isinstance(real, bool):
            return real
        decision = data.get("decision")
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, str):
            return decision.lower() in ("yes", "true", "y")
        return default

    def _call_and_parse(self, prompt: str) -> dict | list:
        """Make LLM call and parse JSON response."""
        text = self._call(prompt)
        if not text:
            return {}
        return self._parse_json(text)

    def _call(self, prompt: str) -> str:
        """Dispatch LLM call through provider."""
        try:
            response = self.provider.chat(
                [{"role": "user", "content": prompt}], tools=[]
            )
            return (response.get("content") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """Extract JSON from LLM response text."""
        text = text.strip()
        if text.startswith("{"):
            end = text.rfind("}")
            if end > 0:
                try:
                    return json.loads(text[:end + 1])
                except json.JSONDecodeError:
                    pass
        if text.startswith("["):
            end = text.rfind("]")
            if end > 0:
                try:
                    return json.loads(text[:end + 1])
                except json.JSONDecodeError:
                    pass
        # Fallback: find JSON bounds
        for first, last in [("{", "}"), ("[", "]")]:
            start = text.find(first)
            end = text.rfind(last)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _key(category: str, context: dict) -> str:
        raw = category + json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:16]
