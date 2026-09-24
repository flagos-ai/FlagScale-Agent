"""Unit tests for VerificationGuard acceptance freezing + soft guidance.

Two structural mechanisms (verification independent of implementation):
1. Timing 0 — acceptance is frozen once a step leaves "pending" (block, overridable)
2. Timing 3 — soft guidance (inject) when acceptance criteria are first defined
"""

import tempfile
import shutil
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def _plan_with_step(d, acceptance=("A1",)):
    plan = TaskPlan(d)
    plan.create("Test", [{"title": "Step A", "acceptance": list(acceptance)}])
    return plan


def test_update_acceptance_blocked_after_doing():
    """update_acceptance on a step that has started (doing) is blocked."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        plan.update_step(1, "doing")

        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "update_acceptance", "step_id": 1,
                       "acceptance": ["reshaped to match output"]},
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.category == "acceptance_frozen"
        assert "frozen" in verdict.message.lower()
    finally:
        shutil.rmtree(d)


def test_update_acceptance_blocked_after_done():
    """update_acceptance on a completed step is blocked."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        plan.update_step(1, "done")

        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "update_acceptance", "step_id": 1,
                       "acceptance": ["X"]},
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.category == "acceptance_frozen"
    finally:
        shutil.rmtree(d)


def test_update_acceptance_allowed_while_pending():
    """update_acceptance on a pending step passes freely."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        # step stays pending

        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "update_acceptance", "step_id": 1,
                       "acceptance": ["A1", "A2"]},
        )
        # First call also triggers the one-shot guidance inject — but the freeze
        # check returns None first (pending), so guidance is what we may see.
        verdict = guard.check_pre(ctx)
        # pending → not blocked; may be an inject (guidance) but never block
        assert verdict is None or verdict.action == "inject"
    finally:
        shutil.rmtree(d)


def test_update_acceptance_overridable_after_doing():
    """Frozen acceptance can be overridden with a reason (original was wrong)."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        plan.update_step(1, "doing")

        guard = VerificationGuard(plan=plan)
        verdict = guard.check_pre(GuardContext(
            tool_name="plan_update",
            tool_args={"action": "update_acceptance", "step_id": 1,
                       "acceptance": ["X"]},
        ))
        # guard emits block; override acceptance is validated by accept_override
        assert verdict.action == "block"
        assert guard.accept_override("original criterion referenced wrong artifact", None) is True
    finally:
        shutil.rmtree(d)


def test_no_block_when_status_unknown():
    """No plan / unknown step → permissive (no block)."""
    guard = VerificationGuard(plan=None)
    ctx = GuardContext(
        tool_name="plan_update",
        tool_args={"action": "update_acceptance", "step_id": 1,
                   "acceptance": ["X"]},
    )
    verdict = guard.check_pre(ctx)
    # None plan → status unknown → not blocked (may inject guidance once)
    assert verdict is None or verdict.action == "inject"


def test_first_plan_create_no_longer_blocks_for_qualifier():
    """Regression: the qualifier-extraction gate was MOVED out of VerificationGuard
    into PlanGuard (guard override_reason cross-talk let it be silently released).
    VerificationGuard must no longer block plan_create for qualifiers — a first
    plan_create carrying acceptance falls straight through to the Timing 3
    acceptance-quality inject, no override_reason required."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        guard = VerificationGuard(plan=plan)
        v = guard.check_pre(GuardContext(
            tool_name="plan_create",
            tool_args={"title": "T", "steps": [{"title": "S", "acceptance": ["A1"]}]},
        ))
        # Not blocked; acceptance guidance fires directly.
        assert v is not None
        assert v.action == "inject"
        assert v.category == "acceptance_guidance"
        assert "qualifier" not in v.message.lower()
    finally:
        shutil.rmtree(d)


def test_plan_create_acceptance_guidance_fires_once():
    """First plan_create carrying acceptance → Timing 3 acceptance-quality inject,
    no override_reason needed (the qualifier gate that used to require one is gone).
    Fires once; a later plan_create does not re-inject."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        guard = VerificationGuard(plan=plan)
        v1 = guard.check_pre(GuardContext(
            tool_name="plan_create",
            tool_args={"title": "T", "steps": [{"title": "S", "acceptance": ["A1"]}]},
        ))
        assert v1 is not None
        assert v1.action == "inject"
        assert v1.category == "acceptance_guidance"
        assert "observable" in v1.message.lower()
        # guidance does not fire again on later plan_create (flag set)
        v2 = guard.check_pre(GuardContext(
            tool_name="plan_create",
            tool_args={"title": "T2", "steps": ["a", "b"]},
        ))
        assert v2 is None
    finally:
        shutil.rmtree(d)


def test_plan_create_plain_steps_no_guidance():
    """A plain-string plan defines no acceptance, so Timing 3 has nothing to nudge
    and VerificationGuard stays silent (no qualifier gate anymore either)."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        guard = VerificationGuard(plan=plan)
        v = guard.check_pre(GuardContext(
            tool_name="plan_create",
            tool_args={"title": "T", "steps": ["a", "b"]},
        ))
        assert v is None
    finally:
        shutil.rmtree(d)


def test_verification_guard_silent_on_other_tools():
    """Non-plan tools never trigger any VerificationGuard check_pre verdict."""
    guard = VerificationGuard(plan=None)
    for tool in ("read_file", "shell", "edit_file"):
        v = guard.check_pre(GuardContext(tool_name=tool, tool_args={}))
        assert v is None


def test_acceptance_guidance_covers_all_inputs_and_generalization():
    """Guidance must carry the layer-1 coverage + blind-sample discipline,
    not just the independence note. Locks the design intent against regression."""
    d = tempfile.mkdtemp()
    try:
        plan = _plan_with_step(d)
        guard = VerificationGuard(plan=plan)
        # First plan_create carrying acceptance → acceptance-quality guidance
        # (Timing 3) fires directly; no override_reason needed anymore.
        ctx = GuardContext(
            tool_name="plan_create",
            tool_args={"title": "T", "steps": [{"title": "S", "acceptance": ["A1"]}]},
        )
        msg = guard.check_pre(ctx).message.lower()
        # Coverage: verify against every provided input, not just the debug one.
        assert "every input" in msg
        # Attribution-excuse: a failure inside your declared scope is yours,
        # regardless of origin.
        assert "declared scope" in msg
        assert "regardless of origin" in msg
        # Judge-measurement: verify against the judge's measurement, not a proxy.
        assert "the judge's measurement" in msg
        assert "not your proxy" in msg
        # Situational-traps pointer: delivery-time traps are deferred to
        # completion, not carried at plan time.
        assert "do not carry them into acceptance" in msg
        # Task contract adherence: re-read the task statement, map each
        # operation to a criterion; "I know how to do X" is not a criterion.
        assert "re-read the task statement" in msg
        assert "list every operation it explicitly asks for" in msg
        assert "know how to do x, so x is done" in msg  # quotes normalized to lowercase
        # Independence: the check must be independent of your implementation.
        assert "independent of your implementation" in msg
        # Externally observable end state.
        assert "externally observable" in msg
        # Removed modes (D definitional-grounding, E derived-quantity, F
        # fragile-state-snapshot, G close-the-gap) were retired from plan-time
        # guidance in Fix-3a — assert they no longer leak back in here.
        assert "authoritative source" not in msg
        assert "measures a different quantity" not in msg
        assert "recovering, forensically analyzing" not in msg
        assert "quantitative acceptance bound" not in msg
    finally:
        shutil.rmtree(d)


def test_plan_guard_qualifier_inject_fires_for_plain_steps():
    """The qualifier-extraction reminder now lives in PlanGuard and fires on the
    first plan_create regardless of whether steps carry acceptance — a subject can
    be qualified even without acceptance criteria. It is an inject (advisory), and
    carries the qualifier message, not the acceptance-quality guidance."""
    from flagscale_agent.react.guard.plan import PlanGuard

    guard = PlanGuard()
    ctx = GuardContext(
        tool_name="plan_create",
        tool_args={"title": "T", "steps": ["plain step", "another"]},
    )
    v = guard.check_pre(ctx)
    assert v is not None
    assert v.action == "inject"
    assert v.reason == "qualifier_extraction"
    low = v.message.lower()
    assert "qualifier" in low
    # acceptance-quality guidance must NOT be bundled in (that is VerificationGuard)
    assert "observable" not in low
