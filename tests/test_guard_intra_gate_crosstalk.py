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

"""Regression tests for intra-guard gate-crosstalk.

Root cause: When a single Guard has cascading internal gates (e.g.
VerificationGuard's 5 gates on plan_update(complete)), an override reason
written for Gate 1 could silently release Gate 2-5 because:
  1. check_pre advances past Gate 1 (sets _gate1_done=True)
     and returns block(Gate 2) as the first verdict
  2. _resolve calls accept_override on block(Gate 2) — the default
     Guard.accept_override just checks reason.strip() > 5 chars, so the
     SAME override reason that released Gate 1 also releases Gate 2
  3. Re-call check_pre skips the entire if block (Gate 1 already released),
     returns None — no new block surfaces

Fix: Track _last_surfaced_reason alongside _last_surfaced. Only call
accept_override when the verdict's reason matches the last surfaced reason.
"""

from flagscale_agent.react.guard import (
    Guard, GuardVerdict, GuardContext, GuardRegistry,
)


class _CascadingGuard(Guard):
    """Mock guard with 3 cascading internal gates, mimicking VerificationGuard."""
    name = "cascade"
    priority = 50

    def __init__(self):
        super().__init__()
        self._gate1_done = False
        self._gate2_done = False
        self._gate3_done = False

    def check_pre(self, ctx):
        if not self._gate1_done:
            if not ctx.override_reason.strip():
                return GuardVerdict.block(
                    "[Gate 1] classify the task",
                    reason="gate1",
                    category="verification_required",
                )
            self._gate1_done = True
        if not self._gate2_done:
            self._gate2_done = True
            return GuardVerdict.block(
                "[Gate 2] provide an observation",
                reason="gate2",
                category="verification_required",
            )
        if not self._gate3_done:
            self._gate3_done = True
            return GuardVerdict.block(
                "[Gate 3] delivery hygiene",
                reason="gate3",
                category="verification_required",
            )
        return None

    def reset_turn(self):
        self._gate1_done = False
        self._gate2_done = False
        self._gate3_done = False


def _ctx(override="", tool_name="plan_update", tool_args=None):
    if tool_args is None:
        tool_args = {"action": "complete"}
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args,
        override_reason=override,
    )


class TestIntraGuardGateCrosstalk:
    """An override reason for Gate 1 must NOT release Gate 2."""

    def test_gate1_blocks_without_override(self):
        reg = GuardRegistry()
        reg.register(_CascadingGuard())
        v = reg.check_pre(_ctx(override=""))
        assert v is not None and v.action == "block"
        assert v.reason == "gate1"

    def test_gate1_released_gate2_surfaces(self):
        reg = GuardRegistry()
        reg.register(_CascadingGuard())
        v1 = reg.check_pre(_ctx(override=""))
        assert v1 is not None and v1.reason == "gate1"
        v2 = reg.check_pre(_ctx(override="classified the task as pass/fail"))
        assert v2 is not None and v2.action == "block"
        assert v2.reason == "gate2"

    def test_gate2_not_released_by_gate1_override(self):
        reg = GuardRegistry()
        reg.register(_CascadingGuard())
        reg.check_pre(_ctx(override=""))
        v = reg.check_pre(_ctx(override="classified the task as pass/fail"))
        assert v is not None and v.reason == "gate2"
        v2 = reg.check_pre(_ctx(override="ran the tests and verified output"))
        assert v2 is not None and v2.reason == "gate3"

    def test_full_cascade_requires_distinct_overrides(self):
        reg = GuardRegistry()
        reg.register(_CascadingGuard())
        v1 = reg.check_pre(_ctx(override=""))
        assert v1.reason == "gate1"
        v2 = reg.check_pre(_ctx(override="classified the task"))
        assert v2.reason == "gate2"
        v3 = reg.check_pre(_ctx(override="ran tests and verified"))
        assert v3.reason == "gate3"
        v4 = reg.check_pre(_ctx(override="checked delivery hygiene"))
        assert v4 is None, "All gates released, should pass"


class TestResetTurnClearsGateState:
    """reset_turn should clear all gate flags so a new task starts fresh."""

    def test_reset_turn_restores_gate1(self):
        """After completing all gates, reset_turn should make Gate 1 fire again."""
        reg = GuardRegistry()
        guard = _CascadingGuard()
        reg.register(guard)
        # Go through all 3 gates
        reg.check_pre(_ctx(override=""))  # Gate 1
        reg.check_pre(_ctx(override="classified"))  # Gate 2
        reg.check_pre(_ctx(override="verified"))  # Gate 3
        reg.check_pre(_ctx(override="hygiene checked"))  # Pass
        # All gates consumed — should return None
        assert reg.check_pre(_ctx(override="")) is None
        # reset_turn should restore all gates
        reg.reset_turn()
        v = reg.check_pre(_ctx(override=""))
        assert v is not None and v.reason == "gate1", \
            "Gate 1 should fire again after reset_turn"
