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

"""Tests for DebugDisciplineGuard — hypothesis-driven debugging and override mechanism."""

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.debug_discipline import DebugDisciplineGuard
from flagscale_agent.react.state_machine import AgentState


def _ctx(tool_name="", tool_args=None, tool_result=None,
         assistant_text="", **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=AgentState.EXECUTING,
        assistant_text=assistant_text,
        **kwargs,
    )


class TestHypothesisGate:
    """Test that the guard blocks edits after failure without hypothesis."""

    def test_no_failure_allows_edits(self):
        """When no failure observed, edits pass freely."""
        g = DebugDisciplineGuard()
        ctx = _ctx("write_file", {"path": "/tmp/fix.py"}, assistant_text="fixing bug")
        result = g.check_pre(ctx)
        assert result is None

    def test_failure_then_first_edit_allowed(self):
        """First edit after failure is allowed (threshold is 2)."""
        g = DebugDisciplineGuard()
        # Simulate failure detection via check_post
        post_ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: CUDA OOM")
        g.check_post(post_ctx)
        assert g._failure_observed is True

        # First .py edit — should pass (edits_since_failure becomes 1, threshold is >= 2)
        ctx = _ctx("write_file", {"path": "/tmp/fix.py"}, assistant_text="trying something")
        result = g.check_pre(ctx)
        assert result is None

    def test_failure_then_second_edit_blocked(self):
        """Second edit after failure without hypothesis is blocked."""
        g = DebugDisciplineGuard()
        # Simulate failure
        post_ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: boom")
        g.check_post(post_ctx)

        # First edit — passes
        ctx1 = _ctx("write_file", {"path": "/tmp/fix.py"}, assistant_text="trying")
        g.check_pre(ctx1)

        # Second edit — should be blocked
        ctx2 = _ctx("edit_file", {"path": "/tmp/fix2.py"}, assistant_text="another try")
        result = g.check_pre(ctx2)
        assert result is not None
        assert result.action == "inject_msg"
        assert "hypothesis" in result.message.lower()

    def test_non_py_files_not_blocked(self):
        """Edits to non-.py files are never blocked."""
        g = DebugDisciplineGuard()
        # Simulate failure
        post_ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: x")
        g.check_post(post_ctx)

        # .yaml edit — should pass regardless
        for _ in range(5):
            ctx = _ctx("write_file", {"path": "/tmp/config.yaml"}, assistant_text="editing yaml")
            result = g.check_pre(ctx)
            assert result is None


class TestHypothesisDetection:
    """Test that hypothesis in assistant_text clears the gate."""

    def test_hypothesis_in_text_clears_gate(self):
        """When LLM writes 'HYPOTHESIS:' in response, gate clears."""
        g = DebugDisciplineGuard()
        # Simulate failure
        post_ctx = _ctx("monitor", tool_result="RuntimeError: NCCL error")
        g.check_post(post_ctx)
        # First edit
        g.check_pre(_ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="try"))

        # Second edit WITH hypothesis in text — should pass
        ctx = _ctx("write_file", {"path": "/tmp/f.py"},
                   assistant_text="HYPOTHESIS: The TP config causes NCCL deadlock\nEVIDENCE: logs show rank 3 stuck")
        result = g.check_pre(ctx)
        assert result is None
        assert g._hypothesis_declared is True

    def test_hypothesis_bold_format(self):
        """Detects **hypothesis** markdown format."""
        g = DebugDisciplineGuard()
        post_ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: x")
        g.check_post(post_ctx)
        g.check_pre(_ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="x"))

        ctx = _ctx("write_file", {"path": "/tmp/f.py"},
                   assistant_text="**HYPOTHESIS**: the learning rate is wrong")
        result = g.check_pre(ctx)
        assert result is None

    def test_chinese_hypothesis(self):
        """Detects Chinese hypothesis patterns."""
        g = DebugDisciplineGuard()
        post_ctx = _ctx("monitor", tool_result="RuntimeError: OOM")
        g.check_post(post_ctx)
        g.check_pre(_ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="x"))

        ctx = _ctx("write_file", {"path": "/tmp/f.py"},
                   assistant_text="假设：batch size过大导致OOM")
        result = g.check_pre(ctx)
        assert result is None

    def test_root_cause_clears_gate(self):
        """'root cause:' also counts as hypothesis."""
        g = DebugDisciplineGuard()
        post_ctx = _ctx("monitor", tool_result="CUDA error: device-side assert")
        g.check_post(post_ctx)
        g.check_pre(_ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="x"))

        ctx = _ctx("write_file", {"path": "/tmp/f.py"},
                   assistant_text="root cause: the vocab size doesn't match embedding dim")
        result = g.check_pre(ctx)
        assert result is None

    def test_empty_assistant_text_skips_gate(self):
        """When assistant_text is empty, guard skips entirely (safety valve)."""
        g = DebugDisciplineGuard()
        post_ctx = _ctx("monitor", tool_result="RuntimeError: fail")
        g.check_post(post_ctx)
        g.check_pre(_ctx("write_file", {"path": "/tmp/f.py"}, assistant_text=""))

        # Empty text — should skip
        ctx = _ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="")
        result = g.check_pre(ctx)
        assert result is None


class TestOverrideMechanism:
    """Test the overridable + accept_override flow."""

    def test_overridable_is_true(self):
        """Guard is marked as overridable."""
        g = DebugDisciplineGuard()
        assert g.overridable is True

    def test_accept_override_with_valid_reason(self):
        """Override accepted with reason > 10 chars."""
        g = DebugDisciplineGuard()
        ctx = _ctx("write_file", {"path": "/tmp/f.py"})
        result = g.accept_override("This is not a training fix, it's a new feature implementation", ctx)
        assert result is True
        assert g._hypothesis_declared is True

    def test_accept_override_rejects_short_reason(self):
        """Override rejected when reason is too short."""
        g = DebugDisciplineGuard()
        ctx = _ctx("write_file", {"path": "/tmp/f.py"})
        result = g.accept_override("ok", ctx)
        assert result is False

    def test_accept_override_rejects_empty(self):
        """Override rejected when reason is empty."""
        g = DebugDisciplineGuard()
        ctx = _ctx("write_file", {"path": "/tmp/f.py"})
        result = g.accept_override("", ctx)
        assert result is False


class TestResetTurn:
    """Test that reset_turn clears state properly."""

    def test_reset_clears_failure(self):
        """reset_turn clears _failure_observed."""
        g = DebugDisciplineGuard()
        g._failure_observed = True
        g._hypothesis_declared = False
        g._edits_since_failure = 5
        g.reset_turn()
        assert g._failure_observed is False
        assert g._hypothesis_declared is False
        assert g._edits_since_failure == 0

    def test_fresh_turn_no_blocking(self):
        """After reset_turn, no blocking occurs even with .py edits."""
        g = DebugDisciplineGuard()
        # Simulate stale state from previous session
        g._failure_observed = True
        g._edits_since_failure = 10
        # New turn resets
        g.reset_turn()
        # Should not block
        ctx = _ctx("write_file", {"path": "/tmp/f.py"}, assistant_text="writing code")
        result = g.check_pre(ctx)
        assert result is None


class TestFailureDetection:
    """Test that only training monitor tools trigger failure state."""

    def test_shell_traceback_does_not_trigger(self):
        """Shell commands with tracebacks do NOT set _failure_observed."""
        g = DebugDisciplineGuard()
        ctx = _ctx("shell", tool_result="Traceback (most recent call last):\n  ValueError: bad")
        g.check_post(ctx)
        assert g._failure_observed is False

    def test_monitor_traceback_triggers(self):
        """monitor tool with traceback DOES set _failure_observed."""
        g = DebugDisciplineGuard()
        ctx = _ctx("monitor", tool_result="Traceback (most recent call last):\n  RuntimeError: NCCL")
        g.check_post(ctx)
        assert g._failure_observed is True

    def test_find_latest_log_triggers(self):
        """find_latest_log with error triggers failure."""
        g = DebugDisciplineGuard()
        ctx = _ctx("find_latest_log", tool_result="CUDA error: device-side assert triggered")
        g.check_post(ctx)
        assert g._failure_observed is True

    def test_parse_training_metrics_triggers(self):
        """parse_training_metrics with error triggers failure."""
        g = DebugDisciplineGuard()
        ctx = _ctx("parse_training_metrics", tool_result="Out of memory\nRuntimeError: CUDA OOM")
        g.check_post(ctx)
        assert g._failure_observed is True

    def test_monitor_clean_output_no_trigger(self):
        """monitor with clean output does NOT trigger failure."""
        g = DebugDisciplineGuard()
        ctx = _ctx("monitor", tool_result="iteration 100 | loss 2.34 | grad_norm 1.2")
        g.check_post(ctx)
        assert g._failure_observed is False


class TestDeclareHypothesis:
    """Test the explicit declare_hypothesis method."""

    def test_declare_resets_edits(self):
        g = DebugDisciplineGuard()
        g._edits_since_failure = 5
        g.declare_hypothesis("TP communication failure due to mismatched world size")
        assert g._hypothesis_declared is True
        assert g._edits_since_failure == 0
