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

"""Tests for guard override mechanisms — ensures no guard can cause dead loops.

Verifies:
1. All blocking guards have overridable=True
2. accept_override works with valid reasons
3. accept_override rejects trivial reasons
4. reset_turn prevents cross-session state persistence
5. End-to-end: block → override → unblock flow
"""

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.debug_discipline import DebugDisciplineGuard
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
from flagscale_agent.react.guard.comprehension_gate import ComprehensionGateGuard
from flagscale_agent.react.guard.circuit_breaker import CircuitBreakerGuard
from flagscale_agent.react.guard.training_attempt import TrainingAttemptGuard
from flagscale_agent.react.guard.output_quality import OutputQualityGuard
from flagscale_agent.react.state_machine import AgentState


def _ctx(tool_name="", tool_args=None, tool_result=None,
         assistant_text="", override_reason="", **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=AgentState.EXECUTING,
        assistant_text=assistant_text,
        override_reason=override_reason,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# All guards overridable
# ══════════════════════════════════════════════════════════════════════════════


class TestAllBlockingGuardsOverridable:
    """Every guard that can block/inject must be overridable."""

    def test_debug_discipline_overridable(self):
        g = DebugDisciplineGuard()
        assert g.overridable is True

    def test_memory_discipline_overridable(self):
        g = MemoryDisciplineGuard()
        assert g.overridable is True

    def test_comprehension_gate_overridable(self):
        g = ComprehensionGateGuard()
        assert g.overridable is True

    def test_circuit_breaker_overridable(self):
        g = CircuitBreakerGuard()
        assert g.overridable is True

    def test_training_attempt_overridable(self):
        g = TrainingAttemptGuard()
        assert g.overridable is True

    def test_output_quality_overridable(self):
        g = OutputQualityGuard()
        assert g.overridable is True


# ══════════════════════════════════════════════════════════════════════════════
# accept_override — valid reasons accepted
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptOverrideValid:
    """Guards accept override with substantive reasons."""

    def test_debug_discipline_accepts_reason(self):
        g = DebugDisciplineGuard()
        ctx = _ctx()
        assert g.accept_override("The failure is in unrelated test code, not training", ctx) is True
        # Also verifies side effect
        assert g._hypothesis_declared is True
        assert g._edits_since_failure == 0

    def test_memory_discipline_accepts_reason(self):
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 15
        ctx = _ctx()
        assert g.accept_override("Already checked memory, no relevant entries exist", ctx) is True
        assert g._calls_since_memory == 0

    def test_comprehension_gate_accepts_reason(self):
        g = ComprehensionGateGuard()
        ctx = _ctx()
        assert g.accept_override("I already read all relevant source files in this session", ctx) is True

    def test_circuit_breaker_accepts_reason(self):
        g = CircuitBreakerGuard()
        # Set up OPEN circuit
        g._circuit_state["general"] = g.OPEN
        g._open_block_count["general"] = 3
        ctx = _ctx()
        assert g.accept_override("Root cause identified: wrong env variable, now fixed", ctx) is True
        # Verify circuit transitions to HALF_OPEN
        assert g._circuit_state["general"] == g.HALF_OPEN
        assert g._open_block_count["general"] == 0

    def test_training_attempt_accepts_reason(self):
        g = TrainingAttemptGuard()
        ctx = _ctx()
        assert g.accept_override("Identified root cause: vocab_size mismatch in config", ctx) is True

    def test_output_quality_accepts_reason(self):
        g = OutputQualityGuard()
        g._consecutive_silent_failures = 5
        ctx = _ctx()
        assert g.accept_override("File changed externally, will re-read", ctx) is True
        assert g._consecutive_silent_failures == 0


# ══════════════════════════════════════════════════════════════════════════════
# accept_override — trivial reasons rejected
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptOverrideRejectsShort:
    """Guards reject empty or too-short override reasons."""

    def test_debug_discipline_rejects_empty(self):
        g = DebugDisciplineGuard()
        assert g.accept_override("", _ctx()) is False

    def test_debug_discipline_rejects_short(self):
        g = DebugDisciplineGuard()
        assert g.accept_override("ok", _ctx()) is False

    def test_memory_discipline_rejects_empty(self):
        g = MemoryDisciplineGuard()
        assert g.accept_override("", _ctx()) is False

    def test_memory_discipline_rejects_short(self):
        g = MemoryDisciplineGuard()
        assert g.accept_override("skip", _ctx()) is False

    def test_comprehension_gate_rejects_short(self):
        g = ComprehensionGateGuard()
        assert g.accept_override("just do it", _ctx()) is False

    def test_circuit_breaker_rejects_short(self):
        g = CircuitBreakerGuard()
        assert g.accept_override("retry please", _ctx()) is False

    def test_training_attempt_rejects_short(self):
        g = TrainingAttemptGuard()
        assert g.accept_override("try again", _ctx()) is False

    def test_output_quality_rejects_empty(self):
        g = OutputQualityGuard()
        assert g.accept_override("", _ctx()) is False


# ══════════════════════════════════════════════════════════════════════════════
# reset_turn — prevents cross-session persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestResetTurnPreventsDeadLoop:
    """reset_turn clears escalation state to prevent stale blocks."""

    def test_memory_discipline_reset(self):
        """reset_turn is a no-op — counter persists within a turn."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 7
        g.reset_turn()
        assert g._calls_since_memory == 7

    def test_memory_discipline_reset_state(self):
        """reset_state (full reset via decay/override) clears counter."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 20
        g.reset_state()
        assert g._calls_since_memory == 0

    def test_memory_discipline_preserves_knowledge(self):
        """reset_new_turn doesn't reset counter — memory gap persists across turns."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 8
        g.reset_new_turn()
        assert g._calls_since_memory == 8

    def test_comprehension_gate_reset(self):
        g = ComprehensionGateGuard()
        g._edits_to_complex_files = 5
        g.reset_turn()
        assert g._edits_to_complex_files == 0

    def test_debug_discipline_reset(self):
        g = DebugDisciplineGuard()
        g._failure_observed = True
        g._hypothesis_declared = False
        g._edits_since_failure = 3
        g.reset_new_turn()
        assert g._failure_observed is False
        assert g._hypothesis_declared is False
        assert g._edits_since_failure == 0

    def test_circuit_breaker_increments_iteration(self):
        """Circuit breaker uses iteration counter for cooldown — reset_turn increments it."""
        g = CircuitBreakerGuard()
        old_iter = g._current_iteration
        g.reset_turn()
        assert g._current_iteration == old_iter + 1


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: block → override → unblock
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndOverrideFlow:
    """Simulate the full cycle: trigger → block → override → continue."""

    def test_debug_discipline_full_cycle(self):
        """failure → 2 edits → blocked → override → allowed."""
        g = DebugDisciplineGuard()

        # Trigger failure via check_post
        post_ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: CUDA OOM")
        g.check_post(post_ctx)
        assert g._failure_observed is True

        # First edit passes
        ctx1 = _ctx("write_file", {"path": "/tmp/fix.py"}, assistant_text="trying fix")
        r1 = g.check_pre(ctx1)
        assert r1 is None

        # Second edit blocked
        ctx2 = _ctx("edit_file", {"path": "/tmp/fix2.py"}, assistant_text="another try")
        r2 = g.check_pre(ctx2)
        assert r2 is not None
        assert r2.action == "inject_msg"

        # Override with reason
        assert g.accept_override("Root cause: batch size exceeds GPU memory, reducing from 64 to 32", _ctx()) is True

        # Now edit passes
        ctx3 = _ctx("write_file", {"path": "/tmp/fix3.py"}, assistant_text="applying fix")
        r3 = g.check_pre(ctx3)
        assert r3 is None

    def test_circuit_breaker_full_cycle(self):
        """Multiple errors → circuit open → block → override → half-open."""
        g = CircuitBreakerGuard(trip_threshold=2, cooldown_iters=5)

        # Trigger errors to trip circuit
        for _ in range(2):
            ctx = _ctx("shell", {"command": "python train.py"},
                       tool_result="RuntimeError: CUDA out of memory")
            g.check_post(ctx)

        # Circuit should be open for 'resource' category
        assert g._circuit_state.get("resource") == g.OPEN

        # Override
        result = g.accept_override("Root cause found: leaked tensor in dataloader, fixed in commit abc123", _ctx())
        assert result is True
        assert g._circuit_state["resource"] == g.HALF_OPEN

    def test_memory_discipline_override_clears_block(self):
        """Simulate accumulated calls → override clears counter."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 15

        # Override
        result = g.accept_override("Already checked memory, working on unrelated code generation task", _ctx())
        assert result is True
        assert g._calls_since_memory == 0
