"""Tests for VerificationGuard step_done pre-mortem (check_post inject)."""

from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _STEP_DONE_PREMORTEM,
)
from flagscale_agent.react.guard import GuardContext


def _step_done_ctx(with_override=True):
    reason = "ran tests, all pass" if with_override else ""
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "_override_reason": reason},
        override_reason=reason,
    )


class TestStepDonePremortem:
    def test_premortem_fires_post_after_passing_step_done(self):
        """A step_done that passes the pre-check arms a check_post inject."""
        guard = VerificationGuard()
        # consume the one-shot premise re-check so the step_done passes Mode 2
        guard._step_done_recheck_reminded = True
        ctx = _step_done_ctx(with_override=True)
        # pre-side allows
        assert guard.check_pre(ctx) is None
        assert guard._premortem_pending is True
        # post-side fires the pre-mortem
        v = guard.check_post(ctx)
        assert v is not None and v.action == "inject"
        assert v.reason == "step_done_premortem"
        # flag cleared so it does not repeat spuriously
        assert guard._premortem_pending is False

    def test_no_premortem_when_step_done_blocked(self):
        """A blocked step_done (no evidence) must NOT arm the pre-mortem."""
        guard = VerificationGuard()
        guard._step_done_recheck_reminded = True
        ctx = _step_done_ctx(with_override=False)
        v = guard.check_pre(ctx)
        assert v is not None and v.action == "block"
        assert guard._premortem_pending is False
        assert guard.check_post(ctx) is None

    def test_premortem_does_not_fire_on_unrelated_post(self):
        """check_post is silent when no step_done passed."""
        guard = VerificationGuard()
        ctx = GuardContext(tool_name="shell", tool_args={})
        assert guard.check_post(ctx) is None

    def test_premortem_rearms_each_step_done(self):
        """The pre-mortem fires once per passing step_done, re-armed each time."""
        guard = VerificationGuard()
        guard._step_done_recheck_reminded = True
        # first step_done
        ctx1 = _step_done_ctx(with_override=True)
        guard.check_pre(ctx1)
        assert guard.check_post(ctx1).action == "inject"
        # a second post with nothing pending stays silent
        assert guard.check_post(ctx1) is None
        # second step_done re-arms
        ctx2 = _step_done_ctx(with_override=True)
        guard.check_pre(ctx2)
        assert guard.check_post(ctx2).action == "inject"

    def test_premortem_message_reverses_the_question(self):
        """Message must pose the dis-confirming 'assume you're wrong' framing."""
        msg = _STEP_DONE_PREMORTEM.lower()
        assert "wrong" in msg
        # asks for a concrete failure point + observable symptom + run it
        assert "failure point" in msg or "failure" in msg
        assert "symptom" in msg or "look like" in msg
        assert "run it" in msg or "read the result" in msg
        # explicitly calls out argument-words vs observation
        assert "argued" in msg or "principled" in msg
