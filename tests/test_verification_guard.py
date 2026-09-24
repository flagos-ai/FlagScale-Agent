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

"""Tests for VerificationGuard — requires verification evidence before step_done."""

import pytest

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext


class TestVerificationGuard:
    """Test VerificationGuard blocks step_done without override_reason."""

    def test_blocks_step_done_without_override_reason(self):
        """step_done without _override_reason should be blocked (Mode 2)."""
        guard = VerificationGuard()
        # isolate the Mode 2 evidence check: consume the one-shot premise re-check
        # block that now fires ahead of it on the first step_done of the run.
        guard._step_done_recheck_reminded = True

        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        # reconstructed message poses a self-check question (what did you OBSERVE)
        # rather than the old "verification required" assertion-style phrasing.
        assert "observe" in verdict.message.lower()
        assert verdict.reason == "step_done_no_verification"

    def test_allows_step_done_with_override_reason(self):
        """step_done with _override_reason should pass."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "grep shows no conflicts, files parseable"
            },
            override_reason="grep shows no conflicts, files parseable"
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is None  # Should pass

    def test_allows_empty_override_reason_is_blocked(self):
        """Empty _override_reason should still be blocked."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "   "  # whitespace only
            }
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"

    def test_allows_other_plan_update_actions(self):
        """Other plan_update actions (step_doing, add_steps, etc.) should pass."""
        guard = VerificationGuard()
        
        # "complete" now has its own one-shot re-check block (see
        # TestTaskCompleteRecheck), so it is intentionally excluded here.
        actions = ["step_doing", "step_skip", "add_steps", "abandon"]
        
        for action in actions:
            ctx = GuardContext(
                tool_name="plan_update",
                tool_args={"action": action, "step_id": 3}
            )
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Action {action} should not be blocked"

    def test_allows_all_other_tools(self):
        """All other tools should pass through without blocking."""
        guard = VerificationGuard()
        
        tools = [
            "shell", "read_file", "write_file", "edit_file",
            "memory_read", "memory_write", "grep", "evict"
        ]
        
        for tool in tools:
            ctx = GuardContext(tool_name=tool, tool_args={})
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Tool {tool} should not be blocked"

    def test_first_step_done_blocks_for_premise_recheck(self):
        """The first step_done of a run is blocked once for a premise re-check,
        even when the agent already supplied verification. This is the finalize-time
        counterpart to the plan_create qualifier block: it forces the agent to
        re-examine the premises it operated under (a convenient reading of a term,
        a "close enough", a value taken as given) before any evidence is accepted —
        the exact blind spot that a plan-time block cannot reach, because a premise
        is re-interpreted during execution and only surfaces at completion."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "verification": ["did the thing"],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "step_done_premise_recheck"

    def test_premise_recheck_released_by_override_then_fires_evidence_check(self):
        """Override releases the premise re-check, but the step must STILL carry
        evidence — the re-check does not exempt the step from the Mode 1/2 checks.
        A second step_done no longer hits the premise block (fires once per run)."""
        guard = VerificationGuard()
        # first step_done: override releases the premise re-check, then falls through
        # to Mode 2 (no acceptance) which is satisfied by the same override_reason.
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "_override_reason": "re-read criteria, re-examined premises",
            },
            override_reason="re-read criteria, re-examined premises",
        )
        assert guard.check_pre(ctx1) is None
        assert guard._step_done_recheck_reminded is True

        # second step_done without override: premise block is spent, so this now
        # falls straight to the Mode 2 evidence check and is blocked for THAT reason.
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 2},
        )
        verdict2 = guard.check_pre(ctx2)
        assert verdict2 is not None
        assert verdict2.reason == "step_done_no_verification"

    def test_premise_recheck_reminder_is_task_agnostic_and_premise_focused(self):
        """The reminder must target premise re-examination in general terms, never
        naming a specific constraint kind (time/version/subset) — that would read as
        case-by-case gaming rather than a general finalize discipline."""
        from flagscale_agent.react.guard.verification import _STEP_DONE_RECHECK_REMINDER
        low = _STEP_DONE_RECHECK_REMINDER.lower()
        # names the core move: test belief against result, not restate it
        assert "premise" in low
        assert "restating" in low or "repeats the assumptions" in low
        # names convenience-premise as the trap
        assert "convenience" in low
        # points back at the user's actual ask — off-target competence still misses
        assert "user actually asked" in low or "the user actually asked" in low
        assert "off-target" in low
        assert "wording wins" in low
        # forcing checkpoint, not content check
        assert "not a content check" in low
        assert "_override_reason" in _STEP_DONE_RECHECK_REMINDER
        # no task specifics, no case-by-case constraint kinds
        for w in ("august", "mteb", "scandinavian", "leaderboard", "time boundary"):
            assert w not in low

    def test_premise_recheck_demands_concrete_anchor_not_summary(self):
        """The override_reason instruction must require a CONCRETE ANCHOR (value/command/
        path) or explicit "no evidence", not accept summary-of-diligence ("I re-checked").
        This defends against self-audit degenerating into empty affirmation."""
        from flagscale_agent.react.guard.verification import _STEP_DONE_RECHECK_REMINDER
        low = _STEP_DONE_RECHECK_REMINDER.lower()
        # demands concrete anchor in override_reason
        assert "concrete anchor" in low
        # explicitly names the empty answer this replaces
        assert "empty answer" in low
        assert "re-checked and it's correct" in low or "i re-checked" in low
        # offers specific anchor types: value observed vs expected, premise tested
        assert "value you observed" in low or "observed vs" in low
        assert "premise you re-tested" in low or "re-tested and what" in low
        # provides escape hatch for untestable premise — but must SAY SO explicitly
        assert "no evidence" in low
        # preserves override mechanism — any reason passes, but lack of anchor IS signal
        assert "not a content check" in low
        assert "any override_reason lets" in low or "any override_reason" in low
        assert "reason with no anchor" in low

    def test_post_recovery_inject_on_first_step_doing(self):
        """After notify_recovery(), should inject reminder on first step_doing."""
        guard = VerificationGuard()
        
        # Trigger recovery
        guard.notify_recovery()
        
        # First step_doing should inject reminder
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "inject"
        assert "recovered via hard_reset" in verdict.message.lower()
        assert verdict.reason == "post_recovery_reminder"

    def test_post_recovery_inject_only_once(self):
        """Post-recovery reminder should only fire once."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        
        # First step_doing - should inject
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict1 = guard.check_pre(ctx1)
        assert verdict1 is not None
        assert verdict1.action == "inject"
        
        # Second step_doing - should not inject
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 3}
        )
        verdict2 = guard.check_pre(ctx2)
        assert verdict2 is None

    def test_post_recovery_does_not_affect_step_done_blocking(self):
        """Post-recovery state should not interfere with step_done blocking."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        # isolate the Mode 2 evidence check from the one-shot premise re-check block
        guard._step_done_recheck_reminded = True

        # step_done without override_reason should still be blocked
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "step_done_no_verification"


class TestConstraintGuidanceBlockedComputation:
    """Regression: the six SITUATIONAL delivery traps (byte-immutability,
    reloaded-artifact, noisy-margin, best-so-far write-through, exact-contents,
    backup/rollback) were moved OUT of _ACCEPTANCE_GUIDANCE (plan-time) and into
    _TASK_COMPLETE_DELIVERY_HYGIENE (completion-time), where they are actionable.
    These tests assert each concept survives in the completion-gate text and no
    task-specific noun leaks in. The old mode-I 'no-progress run' bullet was
    deleted (compute-as-defined guidance retired), so no test asserts it."""

    def test_guidance_names_byte_immutability_undo_trap(self):
        # Regression: a task passed perf+correctness but failed a "do not modify X"
        # check — the agent added an internal structure to a graded resource then
        # removed it to "restore" it, but the file's bytes (hence checksum) changed.
        # MUST stay task-agnostic: the concrete DB/index/hash mechanism must NOT leak.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "unchanged" in low
        assert "hash" in low or "checksum" in low
        assert "reverting a change is not never changing it" in low
        assert "add-then-remove" in low
        assert "pre-touch backup" in low
        import re
        for w in ("sqlite", "wal", "sqlite_master", "sha256", "pixel", "ffmpeg"):
            assert not re.search(r"\b" + w + r"\b", low), f"leaked task-specific term: {w!r}"

    def test_guidance_names_reloaded_artifact_vs_inmemory_proxy(self):
        # Regression: tune-mjcf passed correctness but failed speed — the agent
        # measured a live in-session object, but the delivered file reloaded cold
        # behaved differently. MUST stay task-agnostic: no mujoco/model.xml leak.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "in-memory" in low or "in-session" in low
        assert "reload" in low or "reloaded" in low
        assert "fresh" in low
        assert "cold" in low
        assert "serialize" in low or "serialized" in low or "round-trip" in low
        import re
        for w in ("mujoco", "mjcf", "model.xml", "solver", "jacobian"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), (
                f"leaked task-specific term: {w!r}")

    def test_guidance_names_noisy_threshold_margin_over_own_measure(self):
        # Regression: tune-mjcf — own one-shot timing passed by a hair but the
        # verifier's rigorous measurement flipped the verdict. MUST stay task-agnostic.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "noisy" in low
        assert "hair" in low
        assert "margin" in low
        assert "repeat" in low
        assert "outlier" in low or "trim" in low or "central statistic" in low
        assert "variance" in low or "run-to-run" in low or "instrument" in low
        import re
        for w in ("mujoco", "mjcf", "model.xml", "solver", "jacobian", "0.60", "0.5979"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), (
                f"leaked task-specific term: {w!r}")

    def test_guidance_names_best_so_far_writethrough_to_delivery_path(self):
        # Regression: tune-mjcf — a better-and-valid candidate held only in memory
        # was lost on timeout; the on-disk intermediate shipped. MUST stay task-agnostic.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "only in memory" in low
        assert "banked" in low
        assert "delivery path" in low
        assert "write-through" in low or "written through" in low
        assert "best-so-far" in low
        assert "timeout" in low or "kill" in low or "disk" in low
        import re
        for w in ("mujoco", "mjcf", "model.xml", "solver", "jacobian", "0.60", "60.98", "900s"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), (
                f"leaked task-specific term: {w!r}")

    def test_guidance_names_exact_contents_and_shown_command_byproduct(self):
        # Regression: a single-file-delivery task scored 0 because a shown example
        # command deposited a build byproduct into the exact-contents delivery dir.
        # MUST stay task-agnostic: no polyglot/gcc nouns may leak.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "exact set" in low or "exact-contents" in low
        assert "nothing more" in low
        assert "stray sibling" in low
        assert "silently" in low
        assert "shows you" in low
        assert "byproduct" in low
        assert "scratch" in low or "outside the delivery" in low
        import re
        for w in ("polyglot", "cmain", "main.py.c", "fibonacci", ".py.c"):
            assert w not in low, f"leaked task-specific term: {w!r}"

    def test_guidance_names_measure_requires_write_backup_rollback(self):
        # Regression: tune-mjcf — overwrite-then-measure left a regression at the
        # delivery path when a candidate measured worse and was not rolled back.
        # MUST stay task-agnostic.
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        assert "overwrite-then-measure" in low
        assert "backup" in low
        assert "rollback" in low or "roll back" in low
        assert "restore" in low
        assert "delivery path" in low
        assert "strictly better" in low
        assert "regression" in low
        import re
        for w in ("mujoco", "mjcf", "model.xml", "solver", "jacobian", "0.60", "64.49", "900s"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), (
                f"leaked task-specific term: {w!r}")

    def test_qualifier_reminder_separates_subject_from_boundary(self):
        # Regression: a task asked for the best model "as of" a past date; the
        # agent did solid work on the subject (correct benchmark, correct metric)
        # but silently dropped the time qualifier and answered from the current
        # full snapshot — the run completed and returned a confident, out-of-bounds
        # answer. The qualifier guidance now lives in _QUALIFIER_EXTRACTION in
        # plan.py (delivered at plan-framing time via PlanGuard, not post) and must
        # name qualifiers (time point / version / subset / metric definition) as
        # first-class constraints and forbid substituting most-available/current data.
        from flagscale_agent.react.guard.plan import _QUALIFIER_EXTRACTION
        low = _QUALIFIER_EXTRACTION.lower()
        # separates the subject/thing-to-produce from its bounding qualifier
        assert "subject" in low and "qualifier" in low
        # names qualifier categories in the abstract, not any task-specific value
        assert "point in time" in low
        assert "version" in low
        assert "subset" in low
        # the danger: work completes and yields a confident yet out-of-bounds answer
        assert "confident answer" in low
        # forbids letting the most available / most current data stand in
        assert "most available" in low or "most up-to-date data" in low
        # must fold qualifiers into the plan while framing, not after
        assert "plan" in low
        # a qualifier is verified by an unfakeable external grader, not self-report:
        # declaring done does not move the scored verdict; unmet qualifier => FAIL
        assert "machine-verified" in low or "machine-checked" in low
        assert "byte-for-byte" in low
        assert "fail" in low
        # must not hardcode the originating task's specifics
        assert "august" not in low
        assert "mteb" not in low

    def test_qualifier_reminder_demands_correct_interpretation_not_just_extraction(self):
        # Regression: the block forced the agent to extract the time qualifier into
        # the plan, but the agent then read a past-point time bound as "data recent
        # enough" and used the latest snapshot — extracting a qualifier is not the
        # same as reading its boundary in the direction the task means. Guidance
        # must demand interpreting the boundary, name the "newest/most available is
        # best" bias as the trap, and flag the two-way ambiguity of a time bound
        # (state at a past point vs state now). Stays task-agnostic and must NOT
        # quote the originating task's phrasing (e.g. "as of") — that reads like
        # leaderboard-gaming rather than general guidance.
        from flagscale_agent.react.guard.plan import _QUALIFIER_EXTRACTION
        low = _QUALIFIER_EXTRACTION.lower()
        # extraction is only half — must read the boundary the task means
        assert "only half" in low
        assert "boundary" in low
        # state each qualifier's meaning before building on it
        assert "in your own words" in low
        # names the convenience/recency bias as the failure mode
        assert "convenient" in low
        assert "newest" in low or "most available is best" in low
        # a time bound is two-way: state at a past point vs state now
        assert "bound on time" in low
        assert "as it stood at a" in low
        assert "the state right now" in low
        assert "came later out of scope" in low
        # must not default to the freshest data
        assert "most up-to-date data" in low
        # still no task specifics AND no verbatim task phrasing that reads as gaming
        assert "as of" not in low
        assert "august" not in low
        assert "scandinavian" not in low

    def test_qualifier_content_moved_out_of_acceptance_guidance(self):
        # The qualifier-extraction guidance must NOT remain in the post-timing
        # _ACCEPTANCE_GUIDANCE — its useful moment is at plan framing (pre), and
        # duplicating it would fire it too late again. This asserts the move was
        # clean (no stale copy left behind).
        from flagscale_agent.react.guard.verification import _ACCEPTANCE_GUIDANCE
        low = _ACCEPTANCE_GUIDANCE.lower()
        assert "qualifier" not in low
        assert "scandinavian" not in low

    def test_delivery_hygiene_has_semantic_vs_syntactic_bullet(self):
        """方案B: SEMANTIC vs SYNTACTIC bullet 已添加到 Gate5"""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        
        msg = _TASK_COMPLETE_DELIVERY_HYGIENE
        low = msg.lower()
        
        # Bullet 7 present
        assert "semantic vs syntactic" in low, "Missing SEMANTIC vs SYNTACTIC bullet"
        
        # Key content: distinguishes form (syntactic) from content (semantic)
        assert "form" in low or "format" in low
        assert "content" in low
        assert "genuine result" in low or "true product" in low
        
        # Mentions the trap: hand-crafted artifact that looks right
        assert "hand-crafted" in low or "constructed" in low or "stub" in low or "placeholder" in low
        
        # Task-agnostic (no leaked specifics from make-mips or any other task)
        import re
        for w in ("mips", "doom", "frame", "bmp", "vm.js", "640", "400"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), f"leaked task term: {w!r}"

    def test_delivery_hygiene_demands_concrete_anchor_per_item(self):
        """Override_reason must name CONCRETE ANCHOR per checked item (path/command/value),
        not bare claim 'I checked reloaded-fresh'. Defends against checklist-of-ticks."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_DELIVERY_HYGIENE
        low = _TASK_COMPLETE_DELIVERY_HYGIENE.lower()
        # demands concrete anchor in override_reason
        assert "concrete anchor" in low
        # per-item anchor: path/command/value
        assert "path you read" in low or "command you ran" in low
        assert "value it returned" in low or "value" in low
        # explicitly names empty answer
        assert "empty answer" in low
        assert "i checked reloaded-fresh" in low or "i verified the delivered file" in low
        # the claim that names no specifics
        assert "names no path, no command, no value" in low or "no path, no command" in low
        # escape: explicit "none apply"
        assert "none apply" in low
        # preserves override mechanism
        assert "not a content check" not in low  # delivery_hygiene doesn't say this
        assert "_override_reason" in _TASK_COMPLETE_DELIVERY_HYGIENE

    def test_text_complete_hygiene_leads_with_final_answer(self):
        """The wrap-up hygiene prompt must FIRST demand the turn's real final output
        (not a checklist-only reply), because the user reads the end of the
        conversation, not the intermediate steps."""
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE
        low = _TEXT_COMPLETE_HYGIENE.lower()
        # leads with the final answer before the hygiene items
        assert "final answer" in low
        assert "deliver" in low
        # an addition, not a replacement for the real output
        assert "replacement" in low or "addition" in low
        # the framing: user does not read the middle, reads the end
        assert "end of the conversation" in low or "end" in low
        assert "intermediate steps" in low or "intermediate" in low
        # the final-answer directive must come BEFORE the first hygiene item
        assert low.index("final answer") < low.index("near vs far")

    def test_text_complete_hygiene_requires_user_language(self):
        """The wrap-up prompt must instruct responding in the USER'S OWN language,
        so a Chinese task does not get an English template answer."""
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE
        low = _TEXT_COMPLETE_HYGIENE.lower()
        assert "user's own language" in low or "user's language" in low
        assert "language" in low
        # the rationale: this is the message the user actually reads
        assert "actually read" in low or "they will actually" in low



class TestTaskCompleteRecheck:
    """The first plan_update(action="complete") of a run is blocked once for a
    task-completion re-check: classify the task (binary pass/fail vs open-ended
    optimization) and confirm the result meets the matching standard. Fires once,
    any override_reason releases it, content is not checked."""

    def test_first_complete_blocks_for_recheck(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete"},
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_premise_recheck"

    def test_complete_released_by_override_and_fires_once(self):
        guard = VerificationGuard()
        # first complete with override → released for good
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "complete",
                "_override_reason": "open-ended: compared two distinct methods, this is fastest measured",
            },
            override_reason="open-ended: compared two distinct methods, this is fastest measured",
            classify_fn=lambda q, data, default: False,  # Mock judge: no markers trigger
        )
        # gate1 releases (override present) and the reason reports a measurement,
        # so gates 2-4 pass too (all use classify_fn which returns False); the chain
        # falls through to the unconditional 5th gate (delivery hygiene), which fires once.
        v1 = guard.check_pre(ctx1)
        assert v1 is not None and v1.reason == "task_complete_delivery_hygiene"
        assert guard._complete_recheck_reminded is True
        # second complete without override → all gates spent, passes through
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete"},
        )
        assert guard.check_pre(ctx2) is None

    def test_complete_recheck_covers_both_task_kinds_generically(self):
        """The reminder must handle BOTH pass/fail and open-ended tasks, without
        hardcoding 'optimize more' (which would be noise on a binary task) and
        without naming any specific task domain (no case-by-case)."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        # names the binary pass/fail branch
        assert "pass/fail" in low
        # names the open-ended optimization branch
        assert "open-ended" in low
        # forces the agent to classify rather than assume one kind
        assert "which kind" in low
        # the open-ended standard: distinct methods compared, not first-improvement
        assert "distinct" in low
        assert "_override_reason" in _TASK_COMPLETE_RECHECK_REMINDER
        # no case-by-case task specifics leaked in
        assert "sql" not in low
        assert "compcert" not in low
        assert "chess" not in low

    def test_complete_recheck_closes_with_three_orthogonal_axes(self):
        """The closing self-check must ask THREE orthogonal questions — optimized,
        general, generalizes — against the task's REAL purpose, not against any
        imagined check/grader (naming the grade would invite overfitting). The axes
        must be declared independent so a strong yes on one cannot stand in for the
        others. Task-agnostic; no scoring vocabulary."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        full = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        # scope the scoring-vocab ban to the CLOSING three-axis block only — the
        # earlier pass/fail branch legitimately discusses the grader/clean-env gap.
        anchor = "run these orthogonal"
        assert anchor in full, "closing three-axis block missing"
        low = full[full.index(anchor):]
        # the three axes are each named as a distinct check
        assert "optimized" in low
        assert "general" in low
        assert "generalizes" in low
        # the axes are declared independent / orthogonal, not interchangeable
        assert "orthogonal" in low
        assert "independent" in low
        # checks are against the real purpose, NOT against a guessed check/grader
        assert "real purpose" in low
        # the closing self-check must NOT frame itself around scoring — naming the
        # grade would invite the agent to overfit to the check instead of the purpose
        for banned in ("grader", "verifier", "scored", "scoring", " grade "):
            assert banned not in low, f"scoring vocab leaked into closing: {banned!r}"
        # generalizes axis names the observe-vs-real-use gap and reproducing it
        assert "gap" in low
        assert "reproduce" in low or "reproduced" in low
        # still routes through the override checkpoint answering each axis
        assert "_override_reason" in _TASK_COMPLETE_RECHECK_REMINDER
        # strengthened: override must give each axis a CONCRETE ANCHOR, not a verdict
        assert "concrete anchor" in low
        assert "not a verdict" in low
        # no case-by-case task specifics
        for dom in ("sql", "sparql", "compcert", "chess", "fasttext"):
            assert dom not in low, f"task-specific leak: {dom!r}"

    def test_complete_recheck_demands_concrete_anchor_per_axis(self):
        """Each of the three axes must demand a CONCRETE ANCHOR (value/measurement/
        command/explicit no-evidence), not accept a verdict adjective ("optimized").
        Defends against three-adjective answers with no substance."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        full = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        anchor = "run these orthogonal"
        low = full[full.index(anchor):]
        # demands concrete anchor per axis in override_reason
        assert "concrete anchor" in low
        # each axis gets specific anchor requirements
        assert "measured number" in low  # OPTIMIZED
        assert "alternative you ruled out" in low  # OPTIMIZED
        assert "sibling input" in low or "property you special-cased" in low  # GENERAL
        assert "cite the command" in low or "probe you ran" in low  # GENERALIZES
        # explicitly names empty answer: three adjectives
        assert "empty answer" in low
        assert "optimized, general, and it generalizes" in low or "three adjectives" in low
        # provides explicit no-evidence escape per axis
        assert "no evidence" in low
        # each axis description now includes "or state/say explicitly"
        assert "state" in low or "say explicitly" in low
        # preserves override mechanism
        assert "not a content check" in low
        assert "any override_reason lets" in low

    def test_complete_recheck_reanchors_on_user_ask_before_kind(self):
        """Before classifying task kind, the reminder must force a re-anchor on the
        USER's stated ask — catching confident competence aimed at a nearby problem
        the agent's own framing substituted. Must name the upstream-premise drift and
        that plain wording beats the agent's interpretation. Task-agnostic."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        # re-anchor on the user's ask, framed as answering THAT not a nearby question
        assert "as the user stated it" in low or "user stated it" in low
        assert "nearby question" in low
        # names the mechanism: an early/upstream premise going unquestioned
        assert "upstream premise" in low or "early premise" in low
        # names that thorough downstream checking builds false confidence
        assert "false confidence" in low
        # plain need beats agent's self-serving reinterpretation
        assert "the need wins" in low
        # this re-anchor comes BEFORE the kind classification
        assert low.index("nearby question") < low.index("what kind of task")

    def test_complete_recheck_traces_answer_provenance(self):
        """The reminder must ask WHERE the answer came from — observed (read from a
        tool output / stdout / opened file) vs inferred from prior expectation. If the
        task handed an artifact to read from, the answer must trace to a tool call that
        transformed/inspected it; a value that never appeared in any output but is
        "known" from experience is a guess. Legitimately-computed answers count as
        observed. Task-agnostic — no decode/render/ocr specifics."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        # names the observe-vs-infer distinction for the answer's origin
        assert "where your answer came from" in low or "trace where your answer" in low
        assert "did i observe this" in low
        assert "infer" in low
        # prior knowledge is plausibility, not what is actually there
        assert "plausible" in low and "actually there" in low
        # computed answers are legitimately observed (no false positive on math tasks)
        assert "computed" in low
        # this trace comes BEFORE the kind classification (it's part of the opening)
        assert low.index("did i observe this") < low.index("what kind of task")
        # no task-specific leak
        for w in ("gcode", "g-code", "render", "ocr", "tesseract", "bitmap"):
            assert w not in low

    def test_complete_recheck_forbids_disguised_substitute_on_passfail(self):
        """The pass/fail branch must close the constraint-substitution loophole:
        when a named non-negotiable specific cannot be obtained, 'done' is NOT
        reachable by delivering a near-equivalent and dressing the surface to pass
        the check, and disclosing the swap in override_reason does not legalize it.
        The only honest closes are keep-searching or BLOCKED. Task-agnostic."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        # names the counterfeit-appearance trap generically
        assert "substitute" in low
        assert "proxy" in low
        # disclosing the swap is not permission
        assert "honesty is not permission" in low
        # the two legal closes
        assert "keep searching" in low
        assert "blocked" in low
        # empty-honest beats populated-fake
        assert "empty honest blocked" in low
        # still no case-by-case task specifics
        for w in ("povray", "pov-ray", "2.2", "wrapper script"):
            assert w not in low

    def test_complete_recheck_flags_self_contaminated_verify_environment(self):
        """The pass/fail branch must catch environmental near/far confusion: a
        green result produced in the agent's own working environment (loaded with
        locally-installed packages, env vars, helper files, services) does not
        prove the deliverable works in the clean consumer environment. The
        deliverable must carry deps inside or use only target-guaranteed tools;
        a shipped script must not import a locally pip-installed library.
        Task-agnostic."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        assert "clean environment that will actually consume it" in low
        assert "import a library you pip-installed only locally" in low
        assert "carry its dependencies inside itself" in low
        # no case-by-case specifics
        for w in ("cryptography", "openssl", "check_cert"):
            assert w not in low

    def test_complete_recheck_flags_deliverable_addressing_invocation(self):
        """The pass/fail branch must catch a DISTINCT flavor of the environmental
        near/far trap: not what the agent ADDED but HOW the deliverable gets
        addressed. The agent's interactive/login shell resolves the artifact via
        PATH/rc/alias/cwd; the consumer invokes it bare as a non-login
        non-interactive subprocess running the plain command by name. "Runs when I
        TYPE it" != "runs when a PROGRAM calls it"; a found-by-name artifact must
        live in the target's standard install location, checked by re-running the
        bare way the consumer will. Task-agnostic (no sqlite/.bashrc/symlink)."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        assert "how the deliverable gets" in low
        assert 'is not "it runs when a program calls it"' in low
        assert "standard install location" in low
        assert "primed to find it" in low
        for w in ("sqlite", ".bashrc", "profile.d", "symlink", "/usr/local/bin"):
            assert w not in low

    def test_complete_recheck_forbids_widening_exemption_and_unverified_attribution(self):
        """The pass/fail branch must catch two related self-deceptions: (1) a precise
        list of exceptions is a CLOSED constraint — widening the carve-out to swallow a
        failure you could not fix is rewriting the acceptance bar; (2) 'pre-existing /
        external library / unrelated to my change' is a claim needing evidence, not a
        default exit when stuck. Task-agnostic."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        assert "closed constraint" in low
        assert "does not earn a place on it" in low
        assert "is a claim that needs evidence" in low
        assert "outside the exemption list still fails" in low
        for w in ("pyknotid", "planarity", "cython"):
            assert w not in low

    def test_complete_recheck_flags_single_sample_overfit_magic_constants(self):
        """The pass/fail branch must catch single-sample overfitting: developing
        against one visible example but graded on hidden inputs — passing the sample
        is the floor, not generalization. Magic constants tuned to the sample are the
        tell; prefer relative/normalized/structure-derived judgments. Task-agnostic."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        assert "passing on that sample is the floor" in low
        assert "magic constants" in low
        assert "audit every hardcoded number" in low
        assert "cannot confirm generalization by testing" in low
        for w in ("jump", "hurdle", "takeoff", "jump_analyzer"):
            assert w not in low
        # magic-constant example must stay abstract — no specific tuned digits that
        # re-leak one task's concrete pipeline (threshold of 25 / cutoff of 2000 / etc.)
        for _lit in ("threshold of 25", "cutoff of 2000", "exceeds 40", "rises more than 40"):
            assert _lit not in low, f"leaked concrete tuned magic number: {_lit!r}"

    def test_complete_recheck_generalizes_axis_prefers_trace_over_rerun(self):
        """Axis 3 GENERALIZES must be retrospective-first: mine the existing trace for
        a command already run / output already read that exercises the gap, and CITE
        it as the observation. A fresh run is the FALLBACK, warranted only when the
        trace lacks evidence for a MATERIAL gap — and then a minimal targeted probe,
        never a re-run of the whole (long) pipeline. This encodes the user's intent:
        self-audit by reviewing prior reasoning/commands, do not re-run by default."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        # retrospective-first: mine the trace, cite what already ran
        assert "mine your own trace" in low
        assert "already ran" in low and "already read" in low
        assert "cite it" in low
        assert "retrospective evidence is an observation" in low
        # re-running an already-answered check is waste
        assert "no re-run needed" in low
        assert "wasted work" in low
        # fresh run is the gated fallback, and it is a minimal probe not a full re-run
        assert "only if the trace genuinely lacks evidence" in low
        assert "minimal targeted probe" in low
        assert "never re-run the whole task" in low

    def test_complete_recheck_env_portability_prefers_existing_bare_invocation(self):
        """The environment-portability check must also be retrospective-first: if some
        invocation in the trace already ran the bare/clean way the consumer will, cite
        it; a bare re-run is warranted only when EVERY traced invocation leaned on the
        primed shell / local setup."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = " ".join(_TASK_COMPLETE_RECHECK_REMINDER.lower().split())
        assert "how you already invoked it in your trace" in low
        assert "cite" in low
        assert "only if every invocation in your trace" in low

    def test_open_ended_depth_vs_breadth_discriminator(self):
        """Open-ended branch must teach how to judge whether 'distinct' attempts
        were actually distinct: a clustered/near-tie spread across attempts means
        depth limit of ONE method-family, and the response is to widen to a
        STRUCTURALLY different family, not to stop. Must NOT rely on knowing the
        target's absolute best (agent cannot see the grader/reference)."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_RECHECK_REMINDER
        low = _TASK_COMPLETE_RECHECK_REMINDER.lower()
        # reads the spread of measured results as the signal
        assert "spread" in low
        # a near-tie / same order of magnitude means same method-family
        assert "same order of magnitude" in low
        assert "method-family" in low
        # names the correct diagnosis: depth limit of one family, not global best
        assert "depth limit" in low
        # prescribes widening to a structurally different family
        assert "structurally different" in low
        # explicitly does not require knowing the absolute best target
        assert "do not need to know the target's absolute best" in low
        # guards against calling one family's limit the task's limit
        assert "one family's limit and called it the task's" in low


class TestTaskCompleteObservationGate:
    """Second task-complete gate: when the first gate's release reason only ARGUES
    the method is sound (method-defence vocabulary, no observation), bite once more
    and demand a concrete observation. Runtime form of the prompt's
    observation-vs-argument litmus. Task-agnostic."""

    def _complete(self, reason):
        # Mock judge: "ran...matched" means has observation, others lack observation
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_lacks_observation":
                # True = lacks observation (pure argument)
                return not ("ran" in r and "matched" in r)
            elif question == "reason_overfits_sample":
                return False
            elif question == "reason_discloses_substitution":
                return False
            return default
        
        return GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete", "_override_reason": reason},
            override_reason=reason,
            classify_fn=mock_judge,
        )

    def test_pure_argument_reason_blocked_by_second_gate(self):
        guard = VerificationGuard()
        # first gate releases (override present), second gate bites: pure argument
        # (no observation marker → inclusion gate demands one)
        verdict = guard.check_pre(
            self._complete("thresholds are conservative and should generalize to hidden inputs")
        )
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_observation_demand"
        assert guard._complete_recheck_reminded is True
        assert guard._complete_observation_demanded is True

    def test_observation_reason_releases_both_gates(self):
        guard = VerificationGuard()
        # reason reports an actual observation → second gate does not fire
        verdict = guard.check_pre(
            self._complete("I ran it on the sample and the output matched the known answer")
        )
        # gate2 does not fire (observation present); the chain falls through to the
        # terminal unconditional 5th gate (delivery hygiene).
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_recheck_reminded is True
        assert guard._complete_observation_demanded is False

    def test_neutral_assertion_now_blocked_by_inclusion_gate(self):
        # Inclusion flip: a confidently-wrong reason that merely ASSERTS the result
        # is correct — no argument marker, no observation marker — used to fall in
        # the gap between marker sets and pass the old EXCLUSION filter. Now the
        # default posture is positive: without a run+read signal, the gate bites.
        guard = VerificationGuard()
        verdict = guard.check_pre(self._complete("re-checked each acceptance criterion"))
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_observation_demand"
        assert guard._complete_observation_demanded is True

    def test_bare_correctness_claim_blocked_by_inclusion_gate(self):
        # The sparql pathology in one line: a clean plausible assertion with zero
        # observation vocabulary. Under exclusion it passed; under inclusion it is
        # forced to run+read.
        guard = VerificationGuard()
        verdict = guard.check_pre(
            self._complete("the query returns the correct set of professors")
        )
        assert verdict is not None
        assert verdict.reason == "task_complete_observation_demand"

    def test_second_gate_fires_at_most_once(self):
        guard = VerificationGuard()
        # first complete: no observation → second gate blocks
        v1 = guard.check_pre(self._complete("the approach is principled and should work"))
        assert v1 is not None and v1.reason == "task_complete_observation_demand"
        # agent re-issues, still no observation → gate is spent, passes through
        # (honest escape: "nothing runnable to check" said on the retry releases it)
        v2 = guard.check_pre(self._complete("still principled, nothing runnable to check"))
        assert v2 is None

    def test_second_gate_message_teaches_observation_not_argument(self):
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_OBSERVATION_DEMAND
        low = " ".join(_TASK_COMPLETE_OBSERVATION_DEMAND.lower().split())
        assert "observation" in low
        assert "argument" in low
        # prescribes concrete moves: run the sample / compare to known answer / perturb
        assert "compare" in low
        assert "known answer" in low
        assert "perturb" in low
        assert "_override_reason" in _TASK_COMPLETE_OBSERVATION_DEMAND
        # releases if genuinely nothing to run — not a hard block
        assert "fires once" in low
        # no case-by-case task specifics leaked in
        for w in ("jump", "takeoff", "chess", "sql", "povray", "video"):
            assert w not in low

    def test_has_observation_inclusion_predicate(self):
        from flagscale_agent.react.guard.verification import _has_observation
        
        # After marker removal: _has_observation requires classify_fn, returns False without it
        # Test with no classify_fn → returns False
        assert _has_observation("ran pytest, tests passed") is False
        assert _has_observation("the query returns the correct set") is False
        assert _has_observation("") is False
        
        # Test with mock classify_fn
        def mock_judge(question, data, default):
            if question == "reason_lacks_observation":
                r = data.get("reason", "").lower()
                # True = lacks observation
                return not any(w in r for w in ["ran", "compared", "diff", "result was"])
            return default
        
        # concrete run/read/compare signals → has observation (lacks=False → return True)
        assert _has_observation("ran pytest, tests passed", mock_judge) is True
        assert _has_observation("I compared output to the known answer", mock_judge) is True
        assert _has_observation("the result was 42, matches expected value", mock_judge) is True
        assert _has_observation("diff shows no changes, exit code 0", mock_judge) is True
        # bare assertion, no run+read → lacks observation (lacks=True → return False)
        assert _has_observation("the query returns the correct set", mock_judge) is False
        assert _has_observation("re-checked each acceptance criterion", mock_judge) is False
        assert _has_observation("this is reasonable and should generalize", mock_judge) is False

    def test_inclusion_gate_uses_has_observation(self):
        # Regression guard for the exclusion→inclusion flip: a reason that is
        # NEITHER argument NOR observation (the sparql-style bare assertion) must
        # now block. The inclusion predicate _has_observation correctly returns False → blocks.
        from flagscale_agent.react.guard.verification import _has_observation
        
        def mock_judge(question, data, default):
            if question == "reason_lacks_observation":
                return "ran" not in data.get("reason", "").lower()
            return default
        
        bare = "the output is correct for all cases"
        assert _has_observation(bare, mock_judge) is False    # new gate: WILL block


class TestTaskCompleteGeneralizationGate:
    """Third complete-gate: a TRUE observation confined to the one development
    sample (tuned to it, no generalization signal) is still overfit. It fires
    after the second gate, at most once, and any override releases it."""

    def _complete(self, reason):
        # Mock judge for generalization gate tests
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_lacks_observation":
                # Has observation if contains "ran" or "compared" or "result" or "tuned" (tuned implies ran)
                return not any(w in r for w in ["ran", "compared", "result", "diff", "tuned", "it got it right"])
            elif question == "reason_overfits_sample":
                # Overfits if contains "tuned" AND no generalization markers
                has_tuned = "tuned" in r or "adjusted until" in r
                has_generalization = any(w in r for w in ["perturbed", "rescaled", "stable", "other input", "variant"])
                return has_tuned and not has_generalization
            elif question == "reason_discloses_substitution":
                return False
            return default
        
        return GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete", "_override_reason": reason},
            override_reason=reason,
            classify_fn=mock_judge,
        )

    def test_sample_local_only_reason_blocked_by_third_gate(self):
        guard = VerificationGuard()
        # observation present (passes gate 2) but confined to the sample + tuned to it
        verdict = guard.check_pre(
            self._complete(
                "I ran it on the example and tuned the threshold until the output "
                "matched the expected value"
            )
        )
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_generalization_demand"
        assert guard._complete_observation_demanded is False
        assert guard._complete_generalization_demanded is True

    def test_generalization_reason_releases_all_gates(self):
        guard = VerificationGuard()
        # observation on a perturbed/other input → third gate does not fire
        verdict = guard.check_pre(
            self._complete(
                "I ran it on the sample, then wrote a rescaled perturbed variant to "
                "disk and ran that too; the output stayed stable"
            )
        )
        # gate3 does not fire (generalization signal present); chain falls through
        # to the terminal 5th gate (delivery hygiene).
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_generalization_demanded is False

    def test_pure_argument_routes_to_second_gate_not_third(self):
        guard = VerificationGuard()
        # pure argument → second gate claims it; third never sees it this call
        verdict = guard.check_pre(
            self._complete("the thresholds are relative and should generalize")
        )
        assert verdict is not None
        assert verdict.reason == "task_complete_observation_demand"
        assert guard._complete_generalization_demanded is False

    def test_neutral_reason_not_false_positived_by_third_gate(self):
        guard = VerificationGuard()
        # carries an observation (passes gate2) but no sample-local/tuning tell →
        # gate3 must not false-positive
        verdict = guard.check_pre(
            self._complete("I ran the full suite and every test passed")
        )
        # gate3 does not fire; chain falls through to the terminal 5th gate.
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_generalization_demanded is False

    def test_third_gate_fires_at_most_once(self):
        guard = VerificationGuard()
        v1 = guard.check_pre(
            self._complete("tuned it on the example until it got it right")
        )
        assert v1 is not None and v1.reason == "task_complete_generalization_demand"
        # re-issue, still sample-local-only → gate spent, passes through
        v2 = guard.check_pre(
            self._complete("still tuned to the sample, got it right")
        )
        assert v2 is None

    def test_third_gate_message_is_generic_and_prescribes_stress_input(self):
        from flagscale_agent.react.guard.verification import (
            _TASK_COMPLETE_GENERALIZATION_DEMAND,
        )
        low = " ".join(_TASK_COMPLETE_GENERALIZATION_DEMAND.lower().split())
        assert "stress input" in low
        assert "perturb" in low
        assert "generaliz" in low
        # names the develop-vs-grade gap and demands an OTHER input observation
        assert "gap" in low
        assert "_override_reason" in _TASK_COMPLETE_GENERALIZATION_DEMAND
        assert "fires once" in low
        # no case-by-case task specifics leaked in
        for w in ("jump", "takeoff", "chess", "sql", "povray", "video", "hurdle"):
            assert w not in low

    def test_is_sample_local_only_boundaries(self):
        from flagscale_agent.react.guard.verification import _is_sample_local_only
        
        # Without classify_fn → returns False (no marker fallback after removal)
        assert _is_sample_local_only("tuned the constants on the example until it matched") is False
        assert _is_sample_local_only("") is False
        
        # With mock judge
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_overfits_sample":
                has_tuned = "tuned" in r
                has_gen = any(w in r for w in ["perturbed", "hidden", "held-out", "variant"])
                return has_tuned and not has_gen
            return default
        
        # sample-local + tuned, no generalization → True
        assert _is_sample_local_only(
            "tuned the constants on the example until it matched", mock_judge
        ) is True
        # sample-local but generalization signal present → False
        assert _is_sample_local_only(
            "tuned on the sample then ran a perturbed variant", mock_judge
        ) is False
        # generalization only → False
        assert _is_sample_local_only("ran it on a hidden held-out input", mock_judge) is False
        # neither → False
        assert _is_sample_local_only("re-checked each criterion", mock_judge) is False


class TestTaskCompleteSubstitutionGate:
    """Fourth complete-gate: the prior gates check whether the result is verified
    and generalizes; this checks whether it is even the thing the task NAMED. A
    reason that discloses delivering a substitute for a GIVEN value (near-equivalent
    / successor / different version) without reporting BLOCKED is caught once.
    Runtime form of the prompt's CONSTRAINT LOYALTY / GIVEN-vs-RANGE rule.
    Task-agnostic."""

    def _complete(self, reason):
        # Mock judge for substitution gate tests
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_lacks_observation":
                return not any(w in r for w in ["ran", "compared", "result", "diff", "read the output"])
            elif question == "reason_overfits_sample":
                return False
            elif question == "reason_discloses_substitution":
                # Discloses substitution if names substitute vocab AND not BLOCKED
                has_sub = any(w in r for w in ["could not obtain", "successor", "instead", "backward-compatible", "substitute", "different version", "unavailable"])
                has_blocked = any(w in r for w in ["blocked", "cannot proceed", "did not deliver", "reporting inability"])
                return has_sub and not has_blocked
            return default
        
        return GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete", "_override_reason": reason},
            override_reason=reason,
            classify_fn=mock_judge,
        )

    def test_disclosed_substitution_blocked_by_fourth_gate(self):
        guard = VerificationGuard()
        # carries an observation (clears gate2), no tuning (clears gate3), reaches
        # gate4: discloses delivering a substitute for a GIVEN
        verdict = guard.check_pre(
            self._complete(
                "ran the build and it is in place, but I could not obtain 2.2 from "
                "any mirror; delivered the backward-compatible successor instead"
            )
        )
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_substitution_demand"
        assert guard._complete_substitution_demanded is True

    def test_blocked_report_releases_fourth_gate(self):
        guard = VerificationGuard()
        # observation present + discloses inability but frames it as BLOCKED
        verdict = guard.check_pre(
            self._complete(
                "ran the check; could not obtain the named version anywhere; reporting "
                "BLOCKED, delivered no artifact and not delivering a substitute"
            )
        )
        # gate4 does not fire (framed as BLOCKED); chain falls through to the
        # terminal 5th gate (delivery hygiene).
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_substitution_demanded is False

    def test_exact_artifact_reason_releases_fourth_gate(self):
        guard = VerificationGuard()
        # reports obtaining the exact named thing, observation present → no gate fires
        verdict = guard.check_pre(
            self._complete(
                "built the exact named version from source, ran it and the output "
                "matched the expected value"
            )
        )
        # gate4 does not fire (exact artifact); chain falls through to the 5th gate.
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_substitution_demanded is False

    def test_neutral_reason_not_false_positived_by_fourth_gate(self):
        guard = VerificationGuard()
        # observation present, no substitution vocabulary → gate4 must not fire
        verdict = guard.check_pre(
            self._complete("I ran the suite and every test passed")
        )
        # gate4 does not fire; chain falls through to the terminal 5th gate.
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_substitution_demanded is False

    def test_pure_argument_routes_to_second_gate_not_fourth(self):
        guard = VerificationGuard()
        # pure argument → second gate claims it; fourth never sees it this call
        verdict = guard.check_pre(
            self._complete("the successor is reasonable and should work just as well")
        )
        assert verdict is not None
        assert verdict.reason == "task_complete_observation_demand"
        assert guard._complete_substitution_demanded is False

    def test_fourth_gate_fires_at_most_once(self):
        guard = VerificationGuard()
        # observation present so gate2 releases; disclosed substitution → gate4 bites
        v1 = guard.check_pre(
            self._complete(
                "ran it and read the output; the required version was unavailable so "
                "I used a newer version"
            )
        )
        assert v1 is not None and v1.reason == "task_complete_substitution_demand"
        # re-issue, still a disclosed substitution → gate spent, passes through
        v2 = guard.check_pre(
            self._complete("ran it again, still used the newer version, it was unavailable")
        )
        assert v2 is None

    def test_fourth_gate_message_is_generic_and_teaches_given_vs_range(self):
        from flagscale_agent.react.guard.verification import (
            _TASK_COMPLETE_SUBSTITUTION_DEMAND,
        )
        low = " ".join(_TASK_COMPLETE_SUBSTITUTION_DEMAND.lower().split())
        assert "given" in low
        assert "zero tolerance" in low
        assert "blocked" in low
        assert "substitut" in low
        assert "_override_reason" in _TASK_COMPLETE_SUBSTITUTION_DEMAND
        assert "fires once" in low
        # no case-by-case task specifics leaked in
        for w in ("jump", "takeoff", "chess", "sql", "video", "hurdle"):
            assert w not in low

    def test_is_disclosed_substitution_boundaries(self):
        from flagscale_agent.react.guard.verification import _is_disclosed_substitution
        
        # Without classify_fn → returns False (no marker fallback after removal)
        assert _is_disclosed_substitution("used the backward-compatible successor instead") is False
        assert _is_disclosed_substitution("") is False
        
        # With mock judge
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_discloses_substitution":
                has_sub = any(w in r for w in ["successor", "instead", "unavailable", "substitute"])
                has_blocked = any(w in r for w in ["blocked", "reporting blocked"])
                return has_sub and not has_blocked
            return default
        
        # substitution disclosed, no BLOCKED → True
        # Updated to new signature: (reason, messages=None, classify_fn=None)
        assert _is_disclosed_substitution(
            "used the backward-compatible successor instead", None, mock_judge
        ) is True
        # substitution language but framed as BLOCKED → False
        assert _is_disclosed_substitution(
            "the successor was unavailable too; reporting BLOCKED", None, mock_judge
        ) is False
        # exact artifact, no substitution language → False
        assert _is_disclosed_substitution("built the exact version from source", None, mock_judge) is False
        # neutral → False
        assert _is_disclosed_substitution("re-checked each criterion", None, mock_judge) is False

    def test_substitution_disclosed_in_context_not_reason(self):
        """方案A: context披露替换，reason不披露 → Gate4应拦"""
        from flagscale_agent.react.guard.verification import _is_disclosed_substitution
        
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_discloses_substitution":
                # Judge sees reason + [Recent context] concatenated
                return "rather than deliver nothing" in r or "stub" in r
            return default
        
        reason = "Far-end verified, frame created successfully"
        messages = [
            {"role": "assistant", "content": "Rather than deliver nothing, I created a stub vm.js"},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "Generated frame.bmp with correct header"},
        ]
        result = _is_disclosed_substitution(reason, messages, mock_judge)
        assert result is True, "Should catch substitution disclosure in recent context"

    def test_substitution_in_reason_only(self):
        """方案A: reason单独披露 → Gate4应拦"""
        from flagscale_agent.react.guard.verification import _is_disclosed_substitution
        
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_discloses_substitution":
                return "successor" in r or "substitute" in r
            return default
        
        reason = "I delivered a near-equivalent successor version"
        messages = []
        result = _is_disclosed_substitution(reason, messages, mock_judge)
        assert result is True

    def test_no_substitution_in_reason_or_context(self):
        """方案A: reason和context都不披露 → 应放行"""
        from flagscale_agent.react.guard.verification import _is_disclosed_substitution
        
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_discloses_substitution":
                return "substitute" in r or "successor" in r or "stub" in r
            return default
        
        reason = "Verified the exact artifact version 3.0.10"
        messages = [
            {"role": "assistant", "content": "Implemented the MIPS interpreter"},
            {"role": "assistant", "content": "Ran DOOM, output captured"},
        ]
        result = _is_disclosed_substitution(reason, messages, mock_judge)
        assert result is False

    def test_empty_messages_backward_compatibility(self):
        """方案A: messages为空时向后兼容，仅用reason"""
        from flagscale_agent.react.guard.verification import _is_disclosed_substitution
        
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_discloses_substitution":
                return "substitute" in r
            return default
        
        reason = "Used a substitute implementation"
        result = _is_disclosed_substitution(reason, [], mock_judge)
        assert result is True
        
        reason2 = "Verified exact artifact"
        result2 = _is_disclosed_substitution(reason2, [], mock_judge)
        assert result2 is False


class TestMagicAssumptionGate:
    """Third complete-gate extension: the NUMBER-FREE overfit. A reason that keys a
    filter/match on the one concrete FORM a categorical value took in the sample
    (a fixed prefix, "always starts with X", an exact-form match) with NO sign the
    agent enumerated the field's real value universe is sample-local overfitting
    even though it contains no fitting/tuning word and no magic number. The escape
    is value-universe exploration (DISTINCT / GROUP BY / enumerated values), which
    is now a generalization marker."""

    def _complete(self, reason):
        # Mock judge for magic assumption gate tests
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_lacks_observation":
                return not any(w in r for w in ["ran", "got", "result", "rows"])
            elif question == "reason_overfits_sample":
                # Overfits if binds form (starts-with, always) AND no universe exploration
                has_form_bind = any(w in r for w in ["always starts", "starts with", "filter"])
                has_universe = any(w in r for w in ["distinct", "enumerated", "value universe", "group by"])
                return has_form_bind and not has_universe
            elif question == "reason_discloses_substitution":
                return False
            return default
        
        return GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete", "_override_reason": reason},
            override_reason=reason,
            classify_fn=mock_judge,
        )

    def test_magic_assumption_reason_blocked_by_third_gate(self):
        guard = VerificationGuard()
        # observation present (passes gate2), no tuning word, but binds concept to
        # sample's concrete form and never explored the value universe
        verdict = guard.check_pre(
            self._complete(
                "The role always starts with Professor, so I used a starts-with "
                "filter and ran it on the sample: got 3 rows"
            )
        )
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_generalization_demand"
        assert guard._complete_generalization_demanded is True

    def test_value_universe_exploration_releases_gate(self):
        guard = VerificationGuard()
        # same shape but the agent DID enumerate the real distinct values first →
        # calibrated to the concept boundary, not the sample's form → passes
        verdict = guard.check_pre(
            self._complete(
                "Ran SELECT DISTINCT role first to see the value universe; it had "
                "Professor and Full Professor, so I used set-membership; got 4 rows"
            )
        )
        # magic-assumption gate does not fire; chain falls through to the 5th gate.
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_generalization_demanded is False

    def test_plain_observation_not_false_positived(self):
        guard = VerificationGuard()
        # a genuine observation with neither a form-assumption tell nor a number →
        # must not trip the magic-assumption markers
        verdict = guard.check_pre(
            self._complete("Ran the query and compared output to the expected answer; matched")
        )
        # magic-assumption gate does not fire; chain falls through to the 5th gate.
        assert verdict is not None
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_generalization_demanded is False

    def test_is_sample_local_only_magic_assumption_boundaries(self):
        from flagscale_agent.react.guard.verification import _is_sample_local_only
        
        def mock_judge(question, data, default):
            if question != "reason_overfits_sample":
                return default
            r = data.get("reason", "").lower()
            # Overfits if binds form (assumes format/prefix/always) AND no universe exploration
            has_form_bind = any(w in r for w in ["assumed the format", "fixed prefix", "always starts", "hardcoded prefix"])
            has_universe = any(w in r for w in ["distinct", "enumerated", "value universe", "group by"])
            return has_form_bind and not has_universe
        
        # form-assumption, no universe exploration → True
        assert _is_sample_local_only(
            "assumed the format is a fixed prefix and matched on it",
            classify_fn=mock_judge
        ) is True
        assert _is_sample_local_only(
            "used a hardcoded prefix; it always starts with the same string",
            classify_fn=mock_judge
        ) is True
        # form-assumption BUT explored the value universe → False (escape)
        assert _is_sample_local_only(
            "it always starts with Professor in the sample, but I ran a distinct "
            "query over all the forms and used set-membership",
            classify_fn=mock_judge
        ) is False
        # generic mention of a string/filter with no assumption tell → False
        assert _is_sample_local_only("wrote a filter on the role column", classify_fn=mock_judge) is False

    def test_generalization_message_names_magic_assumption_and_universe(self):
        from flagscale_agent.react.guard.verification import (
            _TASK_COMPLETE_GENERALIZATION_DEMAND,
        )
        low = " ".join(_TASK_COMPLETE_GENERALIZATION_DEMAND.lower().split())
        assert "magic assumption" in low
        assert "value universe" in low
        assert "distinct" in low
        # still generic — no task specifics leaked
        for w in ("jump", "takeoff", "chess", "povray", "video", "hurdle"):
            assert w not in low


class TestCompletionDeliveryHygieneGate:
    """Fifth completion gate: after result-level gates (verified/generalizes/named
    thing) release, one delivery-hygiene checkpoint fires unconditionally, testing
    the DELIVERED artifact rather than the process. Honest 'none apply' releases it.
    Task-agnostic."""

    def _complete(self, reason):
        # Mock judge that releases all prior gates (observation, generalization, substitution)
        def mock_judge(question, data, default):
            r = data.get("reason", "").lower()
            if question == "reason_lacks_observation":
                # Has observation if mentions ran/read/outputs
                return not any(w in r for w in ["ran", "read", "outputs", "result"])
            elif question == "reason_overfits_sample":
                # Not sample-local if mentions perturbed/held-out/variant
                return "perturbed" not in r and "held-out" not in r and "variant" not in r
            elif question == "reason_discloses_substitution":
                return False
            return default
        
        return GuardContext(
            tool_name="plan_update",
            tool_args={"action": "complete", "_override_reason": reason},
            override_reason=reason,
            classify_fn=mock_judge,
        )

    def test_fifth_gate_fires_after_prior_gates_release(self):
        guard = VerificationGuard()
        # observation present (clears gate2), not sample-local (clears gate3), no
        # substitution (clears gate4) → gate5 bites once
        verdict = guard.check_pre(
            self._complete(
                "ran the full suite on every provided input and read the outputs; "
                "also ran a perturbed held-out variant and results stayed stable"
            )
        )
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "task_complete_delivery_hygiene"
        assert guard._complete_delivery_hygiene_demanded is True

    def test_fifth_gate_fires_at_most_once(self):
        guard = VerificationGuard()
        reason = (
            "ran it and read outputs on all inputs; ran a held-out variant, stable"
        )
        v1 = guard.check_pre(self._complete(reason))
        assert v1 is not None and v1.reason == "task_complete_delivery_hygiene"
        # re-issue → gate spent, passes through
        v2 = guard.check_pre(self._complete(reason + "; checked reloaded artifact fresh"))
        assert v2 is None

    def test_fifth_gate_message_names_delivery_traps_and_is_generic(self):
        from flagscale_agent.react.guard.verification import (
            _TASK_COMPLETE_DELIVERY_HYGIENE,
        )
        low = " ".join(_TASK_COMPLETE_DELIVERY_HYGIENE.lower().split())
        assert "reloaded" in low or "cold" in low
        assert "best-so-far" in low or "banked" in low
        assert "backup" in low
        assert "exact contents" in low
        assert "immutab" in low or "byte" in low
        assert "noisy" in low or "margin" in low
        assert "_override_reason" in _TASK_COMPLETE_DELIVERY_HYGIENE
        # generic — no task specifics leaked
        for w in ("jump", "takeoff", "chess", "povray", "video", "hurdle"):
            assert w not in low


class TestBatchStepDoneGate:
    """Timing 1b: a batch update marking any step 'done' commits the same claim as
    step_done and must clear the same bar. Without this, batch is a silent hole
    that skips every verification check. Non-done batches pass freely."""

    def test_batch_with_done_blocked_without_override(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [
                    {"step_id": 1, "status": "done"},
                    {"step_id": 2, "status": "doing"},
                ],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "batch_step_done_no_verification"

    def test_batch_with_done_allowed_with_override(self):
        guard = VerificationGuard()
        reason = "each done step re-checked against its acceptance, outputs read"
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [{"step_id": 1, "status": "done"}],
                "_override_reason": reason,
            },
            override_reason=reason,
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_batch_without_done_passes_freely(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [
                    {"step_id": 1, "status": "doing"},
                    {"step_id": 2, "status": "skipped"},
                ],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_batch_empty_override_still_blocked(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [{"step_id": 1, "status": "done"}],
                "_override_reason": "   ",
            },
            override_reason="   ",
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.reason == "batch_step_done_no_verification"


class TestTextCompleteHygieneGate:
    """Timing 0a: pure-text [TASK_COMPLETE] finish path (tool_name=="", no
    plan_update). Bites ONCE with a focused delivery-hygiene check; overridable
    via the kernel text override channel. [NEED_USER_INPUT] must not trigger it.
    Fires when there IS (or WAS) an active plan (status=active OR completed) —
    the "completed" case is the intra-guard gate-crosstalk backstop: if the
    Fifth gate was silently released by an override reason for an earlier gate,
    this catches it. No plan at all means casual conversation.
    Task-agnostic."""

    @staticmethod
    def _active_plan():
        from unittest.mock import MagicMock
        plan = MagicMock()
        plan.get_active.return_value = {"title": "test", "status": "active", "steps": []}
        return plan

    @staticmethod
    def _completed_plan():
        from unittest.mock import MagicMock
        plan = MagicMock()
        plan.get_active.return_value = {"title": "test", "status": "completed", "steps": []}
        return plan

    def _text_complete(self, override="", marker="[TASK_COMPLETE]"):
        args = {"_override_reason": override} if override else {}
        return GuardContext(
            tool_name="",
            tool_args=args,
            override_reason=override,
            assistant_text=f"Done with the work. {marker}",
            # These model a FRESH completion the LLM just emitted this iteration
            # (the completion-path consultation sets llm_responded=True). The
            # stale-history top-of-loop case is covered separately in
            # TestTextCompleteHygieneStaleCompletion.
            llm_responded=True,
        )

    def test_bare_text_complete_blocked_once(self):
        guard = VerificationGuard(plan=self._active_plan())
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"
        assert guard._text_complete_hygiene_demanded is True

    def test_text_complete_blocked_without_plan(self):
        # No plan at all — gate still fires. The hygiene gate is a final
        # delivery check, not plan-specific. Even single-step tasks (no plan)
        # need path/constraint/exact-contents/temp-cleanup verification.
        guard = VerificationGuard()
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"
        assert guard._text_complete_hygiene_demanded is True

    def test_text_complete_fires_when_plan_completed(self):
        # Plan already completed (status=completed) — gate MUST still fire.
        # This is the intra-guard gate-crosstalk fix: after plan_update(complete),
        # if the Fifth gate (delivery hygiene) was silently released by an
        # override reason written for an earlier gate, the pure-text
        # [TASK_COMPLETE] backstop catches it.
        guard = VerificationGuard(plan=self._completed_plan())
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"
        assert guard._text_complete_hygiene_demanded is True

    def test_second_text_complete_not_blocked(self):
        # Fires once — a re-emitted bare completion after the gate already bit
        # passes through (agent saw the message; the gate is a checkpoint).
        guard = VerificationGuard(plan=self._active_plan())
        guard.check_pre(self._text_complete())  # first: blocks
        verdict = guard.check_pre(self._text_complete())  # second: released
        assert verdict is None

    def test_override_reason_releases(self):
        guard = VerificationGuard(plan=self._active_plan())
        verdict = guard.check_pre(
            self._text_complete("checked: output.txt at /app, exact single file, no temp left")
        )
        assert verdict is None
        assert guard._text_complete_hygiene_demanded is True

    def test_need_user_input_not_triggered(self):
        # [NEED_USER_INPUT] is routed through the same kernel path but must NOT
        # fire the completion hygiene gate.
        guard = VerificationGuard(plan=self._active_plan())
        verdict = guard.check_pre(
            self._text_complete(marker="[NEED_USER_INPUT]")
        )
        assert verdict is None
        assert guard._text_complete_hygiene_demanded is False

    def test_non_empty_tool_name_not_triggered(self):
        # A real tool call that happens to have [TASK_COMPLETE] in prior text
        # (tool_name != "") must not fire this gate.
        guard = VerificationGuard(plan=self._active_plan())
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            assistant_text="almost there [TASK_COMPLETE]",
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None
        assert guard._text_complete_hygiene_demanded is False

    def test_empty_override_whitespace_still_blocked(self):
        guard = VerificationGuard(plan=self._active_plan())
        verdict = guard.check_pre(self._text_complete("   "))
        assert verdict is not None
        assert verdict.reason == "text_complete_hygiene"

    def test_wrap_up_message_is_five_light_items_in_order(self):
        # User decision: the text-complete gate is a LIGHT wrap-up for runs that
        # did NOT go through the plan_update(complete) cascade — near/far, temp/build
        # cleanup, re-read-task delivery confirm, memory review, harness-gap capture.
        # It must NOT re-run
        # the cascade's deep checks (three delivery-path checks, exact-command
        # listing, every-constraint re-read) — those belong to the cascade, and
        # re-asking them here is the duplication the user removed.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        msg = _TEXT_COMPLETE_HYGIENE
        low = msg.lower()
        # Five wrap-up items present.
        assert "near vs far" in low
        assert "far end" in low and "near end" in low
        assert "cleanup" in low and (".bak" in low or "temp" in low)
        assert "build" in low  # build/compile intermediates named in cleanup
        assert "confirm delivery" in low  # step 3 re-reads task & confirms delivery
        assert "memory review" in low and "memory_list()" in low
        # Order is load-bearing: near/far FIRST (may create files), then cleanup,
        # then delivery re-confirm (cleanup can over-reach), then memory, then the
        # harness-gap meta-question LAST (it reflects on everything above).
        # Anchor on the numbered section headers (**...**) so intro mentions of the
        # same words don't skew the positions.
        near_pos = low.index("**near vs far**")
        clean_pos = low.index("**temp & build cleanup**")
        confirm_pos = low.index("**re-read task & confirm delivery**")
        mem_pos = low.index("**memory review & update**")
        gap_pos = low.index("**harness gap & capture**")
        assert near_pos < clean_pos < confirm_pos < mem_pos < gap_pos
        # The order rationale is stated explicitly (verify before clean, re-confirm
        # after clean).
        assert "load-bearing" in low
        # Delivery re-confirm checks the EXACT path and observes rather than assumes.
        assert "exact" in low and ("re-read" in low or "read it again" in low)
        # Observation-vs-argument litmus carried in (the load-bearing near/far idea).
        assert "argument" in low and "observation" in low
        # Override channel intact.
        assert "_override_reason" in msg

    def test_wrap_up_near_far_prefers_existing_trace_over_rerun(self):
        """Step 1 NEAR vs FAR must be retrospective-first: answer from an existing
        command/output in the trace and cite it before manufacturing a fresh run.
        Re-running to re-confirm what was already observed is waste; a fresh probe is
        gated on the trace lacking evidence AND the gap being material. An argument —
        whether cited or freshly probed — still is not verification."""
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE
        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        assert "existing trace" in low
        assert "already ran" in low or "already read" in low
        assert "cite it" in low
        assert "no re-run needed" in low
        assert "re-running to re-confirm what you already observed is waste" in low
        # fresh run is the gated fallback
        assert "only when your trace has no such evidence" in low

    def test_wrap_up_harness_gap_proposes_for_human_approval(self):
        """Item-5 is proposal-mode: the agent routes each gap to a container
        (agent code / skill / knowledge) as an explicit one-line proposal for the
        human to approve and prioritize. Capture still happens (insight), but the
        agent must NOT edit harness files at wrap-up — control stays with the
        human."""
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE
        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        # routing taxonomy: three containers named
        assert "agent code" in low
        assert "a skill" in low and "multi-step procedure" in low
        assert "knowledge" in low and "promote-to-knowledge" in low
        # proposal-first ordering, then capture
        assert "propose first, implement never" in low
        assert "propose —" in low
        assert "capture —" in low
        assert "one line" in low
        # control stays with the human: no self-editing at wrap-up
        assert "control stays with the human" in low
        assert "an output, not a license" in low
        # capture channel intact (survives session even if message unseen)
        assert "memory_write()" in low
        # legacy escape hatch intact
        assert "none apply" in low

    def test_observation_demand_prefers_existing_trace(self):
        """The observation-demand gate must first point at the trace: the observation
        may already exist (a command run, an output read); only a genuine absence
        warrants a fresh run. Do not re-run a check the trace already settled."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_OBSERVATION_DEMAND
        low = " ".join(_TASK_COMPLETE_OBSERVATION_DEMAND.lower().split())
        assert "already exist in your trace" in low
        assert "cite that" in low
        assert "do not re-run a check the trace already settled" in low

    def test_generalization_demand_is_cheap_probe_not_full_rerun(self):
        """The generalization gate must frame its check as a CHEAP targeted probe
        (one perturbed run, or enumerate visible distinct values) rather than a re-run
        of the whole pipeline; and if the distinct values are already visible in the
        trace, cite them instead of re-running."""
        from flagscale_agent.react.guard.verification import _TASK_COMPLETE_GENERALIZATION_DEMAND
        low = " ".join(_TASK_COMPLETE_GENERALIZATION_DEMAND.lower().split())
        assert "cheap targeted probe" in low
        assert "not a re-run of the whole" in low
        assert "already visible in your trace" in low

    def test_memory_review_demands_updating_stale_entries(self):
        # The MEMORY REVIEW item is not write-only: it must push the agent to
        # RECONCILE existing entries this session touched, covering BOTH stale
        # triggers — (a) a value/command/path READ from memory that reality then
        # refuted, and (b) an insight/pitfall direction this session already
        # implemented/verified/disproved. A wrong memory is worse than none.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        # Reconcile pass is named and framed as the usually-skipped one.
        assert "reconcile" in low
        assert "worse than no memory" in low
        # Trigger (a): read-then-refuted — command errored / path wrong / config stale.
        assert "read" in low and ("refuted" in low or "contradicted" in low)
        assert "known-wrong" in low
        # Must fix EVERY copy, not just the one read (sibling sweep).
        assert "sibling" in low or "every copy" in low
        # Trigger (b): implemented/verified/disproved direction.
        assert "implemented" in low and "disproved" in low
        # Act now, not deferred.
        assert "supersede" in low

    def test_wrap_up_harness_gap_item_captures_not_just_reports(self):
        """Item 5 is the meta-question: did this session expose a gap in the agent
        harness itself (guards/gates/prompts/tools/memory) worth codifying? It must
        (a) name the harness machinery explicitly, (b) list concrete tell-tale
        shapes, (c) demand DURABLE capture via memory_write (insight/agent/), not
        prose-only reporting, and (d) permit an explicit 'none' — but push back on
        the reflex 'none' for sessions that edited configs/code or debugged tooling."""
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        # (a) the harness machinery is named, not just "a gap"
        assert "harness gap & capture" in low
        assert "guards, gates, prompts, tools, or memory" in low
        # (b) concrete tell-tale shapes are enumerated
        assert "twice with no guard" in low
        assert "post-check/inject" in low
        assert "not consulted before acting" in low
        # (c) durable capture: memory_write of an insight, not a note to the user
        assert "memory_write()" in low
        assert "insight/agent/" in low
        assert "digest direction" in low
        assert "target artifact" in low
        # (d) explicit none is allowed, but the escape-hatch needs a real look
        assert 'answer "none" explicitly' in low
        assert "deserves a real look" in low

    def test_wrap_up_does_not_duplicate_cascade_deep_checks(self):
        # The removed blocks: the numbered "three delivery checks" and the
        # exact-command listing demand. These are the cascade's job (gates 4/5);
        # the text gate must not repeat them verbatim, else planned tasks get
        # asked twice.
        #
        # NOTE: a LIGHT constraint/qualifier re-check IS intentionally present in
        # step 3 (see test_wrap_up_rechecks_constraints_not_just_form). The pure-
        # text finish path bypasses the plan_update(complete) cascade entirely, so
        # for single-shot tasks this gate is the ONLY place a dropped qualifier
        # (point-in-time, version, subset) can still be caught. What we forbid here
        # is duplicating the cascade's HEAVY blocks, not the concept of a
        # constraint check.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        assert "three delivery checks" not in low
        assert "path & constraint" not in low
        assert "list the exact commands you ran to verify" not in low

    def test_wrap_up_rechecks_constraints_not_just_form(self):
        # Regression: the mteb-leaderboard task finished via the pure-text
        # [TASK_COMPLETE] path (no cascade). Its finish routine confirmed only
        # delivery FORM (path/format/count) and shipped an answer that violated the
        # task's point-in-time qualifier — a newer entry instead of the leader at
        # the stated time. Step 3 must OBSERVE constraint CONTENT, not just form:
        # re-list every qualifier (time / version / subset / metric) as a
        # first-class item and confirm the answer literally honors each.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        # constraints/qualifiers named as a first-class re-check, not just files
        assert "constraint" in low and "qualifier" in low
        # the qualifier categories are enumerated in the abstract
        assert "time" in low
        assert "version" in low and "subset" in low and "metric" in low
        # framed as content-vs-form with zero tolerance (a GIVEN, not a range)
        assert "zero tolerance" in low
        # the specific failure mode: a right-form answer to a nearby question
        assert "nearby question" in low

    def test_wrap_up_names_the_no_cascade_reason(self):
        # The message should tell the agent WHY it is being asked now: this finish
        # did not go through the plan_update(complete) cascade, so this light gate
        # is the only completion check.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = " ".join(_TEXT_COMPLETE_HYGIENE.lower().split())
        assert "plan_update(complete)" in low
        assert "cascade" in low

    def test_wrap_up_fires_even_after_cascade(self):
        # The wrap-up is an ALWAYS-DO finish-line routine. Even if the
        # plan_update(complete) cascade already ran this turn (_complete_recheck_
        # reminded set), the wrap-up still fires — its content (near/far, cleanup,
        # memory review) is disjoint from the cascade's deep delivery checks, so
        # there is no re-run. It is not gated on whether a cascade happened.
        guard = VerificationGuard(plan=self._active_plan())
        guard._complete_recheck_reminded = True  # cascade ran this turn
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None and verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_wrap_up_fires_without_cascade(self):
        # Complement: no cascade this turn (single-step / no plan) → the wrap-up
        # still fires. Same behavior either way.
        guard = VerificationGuard(plan=self._active_plan())
        assert guard._complete_recheck_reminded is False
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None and verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_near_far_content_is_task_agnostic(self):
        # No leakage of the concrete tasks that motivated this (mips/doom/sqlite/
        # qemu/mjcf/chess); the block must stay a general principle.
        from flagscale_agent.react.guard.verification import _TEXT_COMPLETE_HYGIENE

        low = _TEXT_COMPLETE_HYGIENE.lower()
        for banned in (
            "mips", "doom", "sqlite", "qemu", "mjcf", "chess",
            "frame.bmp", "readelf", ".bashrc", "stdbuf",
        ):
            assert banned not in low, f"leaked task-specific token: {banned}"


class TestTextCompleteStalePlanGuard:
    """_TEXT_COMPLETE_HYGIENE now fires regardless of plan state — it is a
    final delivery check, not plan-specific. These tests verify the new
    behavior: [TASK_COMPLETE] always triggers the hygiene gate (once per turn)."""

    @staticmethod
    def _text_complete(override=""):
        args = {"_override_reason": override} if override else {}
        return GuardContext(
            tool_name="",
            tool_args=args,
            override_reason=override,
            assistant_text="Done with the work. [TASK_COMPLETE]",
            # Fresh completion this iteration (see companion note above).
            llm_responded=True,
        )

    def test_no_plan_still_blocks(self):
        """No plan at all → hygiene gate still fires (delivery check applies
        to all tasks, not just plan-based ones)."""
        guard = VerificationGuard(plan=None)
        guard.reset_turn()
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_stale_completed_plan_blocks_new_turn(self):
        """New turn with a stale completed plan from a prior turn → still blocks.
        The gate is a final delivery check, not plan-state-dependent."""
        from unittest.mock import MagicMock
        plan = MagicMock()
        plan.get_active.return_value = None
        plan.list_plans.return_value = [{"title": "old", "status": "completed"}]
        guard = VerificationGuard(plan=plan)
        guard.reset_turn()
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_active_plan_still_blocks(self):
        """Active plan + [TASK_COMPLETE] → blocks."""
        from unittest.mock import MagicMock
        plan = MagicMock()
        plan.get_active.return_value = {"title": "test", "status": "active", "steps": []}
        guard = VerificationGuard(plan=plan)
        verdict = guard.check_pre(self._text_complete())
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_hygiene_demanded_prevents_double_fire(self):
        """Second [TASK_COMPLETE] in the same turn → no block (demanded flag)."""
        guard = VerificationGuard(plan=None)
        guard.reset_turn()
        v1 = guard.check_pre(self._text_complete())
        assert v1 is not None and v1.action == "block"
        v2 = guard.check_pre(self._text_complete())
        assert v2 is None

    def test_override_releases_gate(self):
        """Override reason provided → gate releases, no block."""
        guard = VerificationGuard(plan=None)
        guard.reset_turn()
        verdict = guard.check_pre(self._text_complete(override="verified all outputs"))
        assert verdict is None


class TestTextCompleteHygieneStaleCompletion:
    """Timing 0a (text_complete_hygiene) must fire only on a FRESH completion
    signal the LLM produced this iteration — never on a prior turn's trailing
    [TASK_COMPLETE] scanned back out of history at a new turn's top-of-loop.

    Regression for: every new turn opened with a VerificationGuard block before
    the agent had done any work, because the top-of-loop check_pre built a ctx
    with tool_name="" and assistant_text = the previous turn's ending sentinel.
    """

    def test_stale_completion_at_new_turn_top_does_not_fire(self):
        """tool_name=="" + trailing [TASK_COMPLETE] but llm_responded=False
        (top-of-loop, stale history text) → must NOT block."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="",
            tool_args={},
            assistant_text="all done here.\n\n[TASK_COMPLETE]",
            llm_responded=False,
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_fresh_completion_this_iteration_fires(self):
        """tool_name=="" + trailing [TASK_COMPLETE] + llm_responded=True
        (completion-path, fresh text, no override) → must block once."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="",
            tool_args={},
            assistant_text="I finished the task.\n\n[TASK_COMPLETE]",
            llm_responded=True,
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "text_complete_hygiene"

    def test_fresh_completion_with_override_releases(self):
        """Fresh completion + inline _override_reason → released (no block)."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="",
            tool_args={"_override_reason": "info-only answer, nothing delivered"},
            assistant_text="answer above.\n\n[TASK_COMPLETE]",
            override_reason="info-only answer, nothing delivered",
            llm_responded=True,
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_stale_then_fresh_same_guard_instance(self):
        """The stale top-of-loop consultation must not consume the one-shot
        _text_complete_hygiene_demanded flag, so the later FRESH completion in
        the same turn still fires."""
        guard = VerificationGuard()
        # 1) top-of-loop with stale prior-turn sentinel → no block, flag untouched
        stale = GuardContext(
            tool_name="", tool_args={},
            assistant_text="prior turn.\n[TASK_COMPLETE]",
            llm_responded=False,
        )
        assert guard.check_pre(stale) is None
        # 2) LLM emits a real completion this iteration → must still block
        fresh = GuardContext(
            tool_name="", tool_args={},
            assistant_text="now really done.\n[TASK_COMPLETE]",
            llm_responded=True,
        )
        v = guard.check_pre(fresh)
        assert v is not None and v.reason == "text_complete_hygiene"

    def test_need_user_input_never_fires_regardless_of_flag(self):
        """[NEED_USER_INPUT] is not a completion signal — must not block even
        when llm_responded=True."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="", tool_args={},
            assistant_text="I need a decision.\n\n[NEED_USER_INPUT]",
            llm_responded=True,
        )
        assert guard.check_pre(ctx) is None
