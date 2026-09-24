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

"""Regression test for intra-guard gate-crosstalk (Bug D).

Root cause: When GuardRegistry._resolve released a guard's block via
accept_override, it silently fell through to the next hardest block from
a DIFFERENT guard. But if the SAME guard had cascading internal gates
(e.g. VerificationGuard's 5 gates on plan_update(complete)), releasing
Gate 1's block silently released Gate 5 too — because check_pre was
never re-called. The override reason written for Gate 1 (premise recheck)
unintentionally released Gate 5 (delivery hygiene), bypassing the final
verification checkpoint.

Fix: After releasing a block, re-call check(guard). If it returns a NEW
block with a different reason, surface THAT instead of silently releasing.
"""

from flagscale_agent.react.guard import (
    Guard, GuardVerdict, GuardContext, GuardRegistry,
)


class _CascadingGateGuard(Guard):
    """A guard with cascading internal gates: after gate 1 is released,
    gate 2 surfaces with a different reason. Simulates VerificationGuard's
    5-gate cascade on plan_update(complete)."""

    name = "cascading"
    priority = 50

    def __init__(self):
        self._gate1_released = False

    def check_pre(self, ctx):
        if not self._gate1_released:
            return GuardVerdict.block(
                "[Gate 1] recheck your premises",
                reason="gate1_premise_recheck",
                category="verification_required",
            )
        # Gate 1 released → Gate 2 surfaces
        return GuardVerdict.block(
            "[Gate 5] delivery hygiene check",
            reason="gate5_delivery_hygiene",
            category="verification_required",
        )

    def accept_override(self, reason, ctx):
        # Gate 1 can only be released ONCE. After that, the guard's check_pre
        # returns Gate 5, and Gate 5 needs its OWN override reason.
        if not self._gate1_released and "for-gate-1" in reason:
            self._gate1_released = True
            return True
        if self._gate1_released and "for-gate-5" in reason:
            return True
        return False


def _ctx(override=None):
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "complete"},
        override_reason=override or "",
    )


class TestIntraGuardGateCrosstalk:
    """Verify that releasing Gate 1 surfaces Gate 5, not silence."""

    def test_gate1_override_surfaces_gate2(self):
        """Override written for Gate 1 should release Gate 1, but Gate 5
        (different reason) must surface — not silently pass through."""
        reg = GuardRegistry()
        guard = _CascadingGateGuard()
        reg.register(guard)

        # First turn: Gate 1 blocks, no override
        v1 = reg.check_pre(_ctx())
        assert v1 is not None and v1.action == "block"
        assert v1.reason == "gate1_premise_recheck"
        assert v1.guard_name == "cascading"

        # Second turn: override releases Gate 1, but Gate 5 surfaces
        v2 = reg.check_pre(_ctx("this is for-gate-1"))
        assert v2 is not None and v2.action == "block", \
            "Gate 5 must surface after Gate 1 released — not silently pass"
        assert v2.reason == "gate5_delivery_hygiene", \
            "Surfaced block must be Gate 5 (different reason), not silence"
        assert v2.guard_name == "cascading"

    def test_gate5_requires_own_override(self):
        """After Gate 5 surfaces, a reason for Gate 1 must NOT release it."""
        reg = GuardRegistry()
        guard = _CascadingGateGuard()
        reg.register(guard)

        # Turn 1: Gate 1 surfaces
        reg.check_pre(_ctx())
        # Turn 2: override Gate 1 → Gate 5 surfaces
        v2 = reg.check_pre(_ctx("this is for-gate-1"))
        assert v2 is not None and v2.reason == "gate5_delivery_hygiene"

        # Turn 3: try to override with Gate 1 reason again — must NOT release Gate 5
        v3 = reg.check_pre(_ctx("this is for-gate-1 again"))
        assert v3 is not None and v3.action == "block", \
            "Gate 5 must not be released by a Gate 1 reason"
        assert v3.reason == "gate5_delivery_hygiene"

        # Turn 4: override with Gate 5 reason — releases
        v4 = reg.check_pre(_ctx("this is for-gate-5"))
        assert v4 is None, "Gate 5 released with its own override reason"

    def test_no_crosstalk_with_other_guards(self):
        """Re-calling check_pre after release must not accidentally release
        blocks from OTHER guards — only the same guard is re-checked."""
        class _AlwaysBlock(Guard):
            name = "other"
            priority = 30  # higher severity position
            def check_pre(self, ctx):
                return GuardVerdict.block("[other]", reason="other_block",
                                          category="other")

        reg = GuardRegistry()
        cascading = _CascadingGateGuard()
        reg.register(_AlwaysBlock())
        reg.register(cascading)

        # Turn 1: surface the hardest block
        v1 = reg.check_pre(_ctx())
        assert v1 is not None

        # Turn 2: override whatever was surfaced
        v2 = reg.check_pre(_ctx("this is for-gate-1"))
        # Must still have a block — either Gate 5 or other_guard
        assert v2 is not None and v2.action == "block"
