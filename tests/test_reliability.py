"""Tests for P7 reliability guards: ErrorClassifier, CircuitBreaker, Budget, Checkpoint."""

import time
import tempfile
import os

import pytest

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.error_classifier import ErrorClassifierGuard
from flagscale_agent.react.guard.circuit_breaker import CircuitBreakerGuard
from flagscale_agent.react.guard.budget import BudgetGuard
from flagscale_agent.react.plan import TaskPlan, StepCheckpoint
from flagscale_agent.react.state_machine import AgentState


def _make_ctx(tool_result: str = "", tool_name: str = "shell") -> GuardContext:
    return GuardContext(
        tool_name=tool_name,
        tool_args={},
        tool_result=tool_result,
        current_state=AgentState.EXECUTING,
    )


# ── ErrorClassifierGuard Tests ──────────────────────────────────────────────


class TestErrorClassifier:
    def test_no_error_returns_none(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Success: file written to /tmp/out.txt")
        assert guard.check_post(ctx) is None

    def test_classifies_env_missing(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: ModuleNotFoundError: No module named 'torch'")
        result = guard.check_post(ctx)
        # First occurrence: no inject yet
        assert result is None
        assert guard._last_category == "env_missing"

    def test_classifies_permission(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: Permission denied: '/etc/shadow'")
        guard.check_post(ctx)
        assert guard._last_category == "permission"

    def test_classifies_resource(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
        guard.check_post(ctx)
        assert guard._last_category == "resource"

    def test_classifies_network(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: Connection refused to localhost:8080")
        guard.check_post(ctx)
        assert guard._last_category == "network"

    def test_classifies_config(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: KeyError: 'learning_rate' not found in config")
        guard.check_post(ctx)
        assert guard._last_category == "config"

    def test_suggest_after_2_consecutive(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: ModuleNotFoundError: No module named 'foo'")
        guard.check_post(ctx)
        result = guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "env_missing" in result.reason or "Environment" in result.message

    def test_escalate_after_3_consecutive(self):
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: Permission denied: '/root/secret'")
        guard.check_post(ctx)
        guard.check_post(ctx)
        result = guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "different approach" in result.message.lower() or "STOP" in result.message

    def test_success_resets_streak(self):
        guard = ErrorClassifierGuard()
        err_ctx = _make_ctx("Error: ModuleNotFoundError: No module named 'x'")
        guard.check_post(err_ctx)

        ok_ctx = _make_ctx("File written successfully.")
        guard.check_post(ok_ctx)

        # After reset, one more error should not trigger suggestion
        result = guard.check_post(err_ctx)
        assert result is None


# ── CircuitBreakerGuard Tests ───────────────────────────────────────────────


class TestCircuitBreaker:
    def test_closed_state_allows(self):
        guard = CircuitBreakerGuard(trip_threshold=4, cooldown_iters=3)
        ctx = _make_ctx()
        result = guard.check_pre(ctx)
        assert result is None

    def test_trips_after_threshold(self):
        guard = CircuitBreakerGuard(trip_threshold=3, cooldown_iters=2)
        err_ctx = _make_ctx("Error: Permission denied: '/etc/passwd'")

        # Feed errors
        for _ in range(3):
            guard.check_post(err_ctx)

        # The 3rd check_post should trip
        # Now check_pre should block
        result = guard.check_pre(_make_ctx())
        assert result is not None
        assert result.action == "block"
        assert "circuit" in result.reason.lower() or "CircuitBreaker" in result.message

    def test_cooldown_then_half_open(self):
        guard = CircuitBreakerGuard(trip_threshold=2, cooldown_iters=2)
        err_ctx = _make_ctx("Error: Permission denied")

        guard.check_post(err_ctx)
        guard.check_post(err_ctx)

        # Tripped — should block for cooldown_iters iterations
        result = guard.check_pre(_make_ctx())
        assert result is not None
        assert result.action == "block"

        result = guard.check_pre(_make_ctx())  # still in cooldown
        assert result is not None
        assert result.action == "block"

        # After cooldown (>2 iterations) → half_open, allowed
        result = guard.check_pre(_make_ctx())
        assert result is None  # half-open probe allowed

    def test_half_open_success_closes(self):
        guard = CircuitBreakerGuard(trip_threshold=2, cooldown_iters=1)
        err_ctx = _make_ctx("Error: Permission denied")

        guard.check_post(err_ctx)
        guard.check_post(err_ctx)

        # Trip — first check_pre blocks (cooldown)
        guard.check_pre(_make_ctx())  # blocked (iter 1, elapsed=1, not > 1)
        guard.check_pre(_make_ctx())  # half-open (iter 2, elapsed=2, > 1)

        # Success in half-open → close
        ok_ctx = _make_ctx("Success")
        guard.check_post(ok_ctx)

        # Should be closed now
        assert guard._circuit_state.get("permission") == CircuitBreakerGuard.CLOSED

    def test_half_open_failure_retrips(self):
        guard = CircuitBreakerGuard(trip_threshold=2, cooldown_iters=1)
        err_ctx = _make_ctx("Error: Permission denied")

        guard.check_post(err_ctx)
        guard.check_post(err_ctx)

        guard.check_pre(_make_ctx())  # blocked (cooldown)
        guard.check_pre(_make_ctx())  # half-open

        # Fail again in half-open
        result = guard.check_post(err_ctx)
        assert result is not None
        assert "re-tripped" in result.message.lower() or "retrip" in result.reason

    def test_skips_empty_tool_name(self):
        """CircuitBreaker should not fire on pre-iteration check (tool_name='')."""
        guard = CircuitBreakerGuard(trip_threshold=2, cooldown_iters=1)
        err_ctx = _make_ctx("Error: Permission denied")

        # Trip the circuit
        guard.check_post(err_ctx)
        guard.check_post(err_ctx)

        # Pre-iteration check with empty tool_name should NOT block
        empty_ctx = _make_ctx(tool_name="")
        result = guard.check_pre(empty_ctx)
        assert result is None

        # But with a real tool_name it should block
        result = guard.check_pre(_make_ctx())
        assert result is not None
        assert result.action == "block"

    def test_different_category_resets_count(self):
        """Switching error categories should reset consecutive count for the new one."""
        guard = CircuitBreakerGuard(trip_threshold=3, cooldown_iters=2)
        perm_ctx = _make_ctx("Error: Permission denied")
        net_ctx = _make_ctx("Error: Connection refused")

        # 2 consecutive permission errors
        guard.check_pre(_make_ctx())
        guard.check_post(perm_ctx)
        guard.check_pre(_make_ctx())
        guard.check_post(perm_ctx)
        assert guard._error_counts.get("permission") == 2

        # Switch to network error — should reset network count to 1
        guard.check_pre(_make_ctx())
        guard.check_post(net_ctx)
        assert guard._error_counts.get("network") == 1

        # One more network error should be 2, not trip (threshold=3)
        guard.check_pre(_make_ctx())
        result = guard.check_post(net_ctx)
        # Should not trip yet (only 2 consecutive)
        assert result is None or "TRIPPED" not in (result.message if result else "")


# ── BudgetGuard Tests ───────────────────────────────────────────────────────


class TestBudgetGuard:
    def test_no_warning_under_80(self):
        guard = BudgetGuard(max_tokens=1000, max_tool_calls=100)
        guard.report_tokens(300, 100)  # 40%
        ctx = _make_ctx()
        result = guard.check_pre(ctx)
        assert result is None

    def test_warning_at_80_percent(self):
        guard = BudgetGuard(max_tokens=1000, max_tool_calls=100)
        guard.report_tokens(400, 400)  # 80%
        ctx = _make_ctx()
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "80" in result.message or "Budget" in result.message

    def test_warning_at_95_percent(self):
        guard = BudgetGuard(max_tokens=1000, max_tool_calls=100)
        guard.report_tokens(500, 450)  # 95%
        ctx = _make_ctx()
        # First call triggers 80% (since both thresholds crossed)
        guard.check_pre(ctx)
        # Second call triggers 95%
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "95" in result.message or "Wrap up" in result.message

    def test_block_at_100_percent(self):
        guard = BudgetGuard(max_tokens=1000, max_tool_calls=100)
        guard.report_tokens(600, 500)  # 110% > 100%
        ctx = _make_ctx()
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "exhausted" in result.message.lower()

    def test_tool_call_counting(self):
        guard = BudgetGuard(max_tokens=10_000_000, max_tool_calls=3)
        ctx = _make_ctx(tool_name="shell")
        guard.check_post(ctx)
        guard.check_post(ctx)
        guard.check_post(ctx)
        # Now at 100% tool calls
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_tool_call_warning_independent_of_token_warning(self):
        """Tool call warnings fire even if token budget is low (separate flags)."""
        guard = BudgetGuard(max_tokens=10_000_000, max_tool_calls=10)
        # Report low token usage — no token warning triggered
        guard.report_tokens(100, 100)  # 0.002% tokens
        # Add 8 tool calls to reach 80% tool calls
        ctx = _make_ctx(tool_name="shell")
        for _ in range(8):
            guard.check_post(ctx)
        # Should warn about tool calls at 80%
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "tool" in result.message.lower() or "Tool" in result.message

    def test_usage_summary(self):
        guard = BudgetGuard(max_tokens=1000, max_tool_calls=10)
        guard.report_tokens(100, 50)
        summary = guard.usage_summary
        assert summary["total_tokens"] == 150
        assert summary["max_tokens"] == 1000
        assert summary["token_percent"] == 15.0


# ── StepCheckpoint Tests ────────────────────────────────────────────────────


class TestStepCheckpoint:
    def _make_plan(self):
        tmpdir = tempfile.mkdtemp()
        tp = TaskPlan(tmpdir)
        plan = tp.create("Test Plan", ["Step 1", "Step 2", "Step 3"])
        # Start step 1
        tp.update_step(1, "doing")
        return tp, plan

    def test_checkpoint_creation(self):
        tp, plan = self._make_plan()
        cp = tp.checkpoint(
            step_id=1,
            files=["src/main.py", "config.yaml"],
            memory_keys=["env_info"],
            summary="Completed environment setup",
        )
        assert cp is not None
        assert cp.step_id == 1
        assert cp.files_modified == ["src/main.py", "config.yaml"]
        assert cp.memory_keys == ["env_info"]
        assert cp.summary == "Completed environment setup"
        assert cp.timestamp > 0

    def test_get_checkpoint(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, files=["a.py"], summary="did step 1")
        retrieved = tp.get_checkpoint(1)
        assert retrieved is not None
        assert retrieved.summary == "did step 1"

    def test_get_checkpoint_missing(self):
        tp, plan = self._make_plan()
        assert tp.get_checkpoint(99) is None

    def test_rollback_info(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, files=["a.py"], memory_keys=["k1"], summary="step 1 done")
        tp.update_step(1, "done")
        tp.checkpoint(step_id=2, files=["b.py"], memory_keys=["k2"], summary="step 2 done")

        info = tp.get_rollback_info(1)
        assert "step 1 done" in info
        assert "a.py" in info
        assert "b.py" in info
        assert "k1" in info

    def test_rollback_info_no_plan(self):
        tmpdir = tempfile.mkdtemp()
        tp = TaskPlan(tmpdir)
        assert "No active plan" in tp.get_rollback_info(1)

    def test_list_checkpoints(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, summary="s1")
        tp.update_step(1, "done")
        tp.checkpoint(step_id=2, summary="s2")

        cps = tp.list_checkpoints()
        assert len(cps) == 2
        assert cps[0]["step_id"] == 1
        assert cps[1]["step_id"] == 2

    def test_checkpoint_to_dict(self):
        cp = StepCheckpoint(
            step_id=1,
            timestamp=1000.0,
            files_modified=["x.py"],
            memory_keys=["mem1"],
            summary="test",
        )
        d = cp.to_dict()
        assert d["step_id"] == 1
        assert d["timestamp"] == 1000.0
        assert d["files_modified"] == ["x.py"]

    def test_checkpoint_from_dict(self):
        data = {
            "step_id": 2,
            "timestamp": 2000.0,
            "files_modified": ["y.py"],
            "memory_keys": ["m2"],
            "summary": "restored",
        }
        cp = StepCheckpoint.from_dict(data)
        assert cp.step_id == 2
        assert cp.summary == "restored"
