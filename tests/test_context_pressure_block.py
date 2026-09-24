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

"""Tests for ContextPressureGuard infinite block fix.

Regression tests for the bug where:
- kernel pre-guard check passes tool_name="" (before LLM call)
- ContextPressureGuard blocked it because "" not in _SAVE_TOOLS
- LLM never got called, so it could never invoke evict/hard_reset
- Result: infinite block loop burning all iterations
"""

import pytest
from flagscale_agent.react.guard.context_pressure import (
    ContextPressureGuard,
    BLOCK_RATIO,
    EVICTABLE_THRESHOLD,
)
from flagscale_agent.react.guard import GuardContext


def _make_ctx(pressure: float, evictable_count: int = 80,
              tool_name: str = "", tool_args: dict = None) -> GuardContext:
    """Create a GuardContext with specified pressure and tool info."""
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        context_pressure=pressure,
        evictable_indexes=list(range(1, evictable_count + 1)),
    )


class TestEmptyToolNameNoBlock:
    """Verify that empty tool_name (pre-LLM-call check) is never blocked."""

    def test_high_pressure_empty_tool_not_blocked(self):
        """95% pressure with empty tool_name should NOT block (pre-LLM-call)."""
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.95, evictable_count=80, tool_name="")
        result = guard.check_pre(ctx)
        assert result is None

    def test_critical_pressure_empty_tool_not_blocked(self):
        """99% pressure, low evictable, empty tool_name → still no block."""
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.99, evictable_count=5, tool_name="")
        result = guard.check_pre(ctx)
        assert result is None

    def test_hard_reset_path_empty_tool_not_blocked(self):
        """Even when _need_hard_reset is set, empty tool_name passes through."""
        guard = ContextPressureGuard()
        # Trigger hard_reset path first
        ctx1 = _make_ctx(0.90, evictable_count=10, tool_name="shell")
        r = guard.check_pre(ctx1)
        assert r is not None  # blocked, _need_hard_reset now True
        assert guard._need_hard_reset is True

        # Now pre-LLM check with empty tool_name
        ctx2 = _make_ctx(0.90, evictable_count=10, tool_name="")
        result = guard.check_pre(ctx2)
        assert result is None  # NOT blocked

    def test_evict_path_empty_tool_not_blocked(self):
        """80%+ with enough evictable, empty tool_name → no block."""
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.85, evictable_count=100, tool_name="")
        result = guard.check_pre(ctx)
        assert result is None


class TestNonEmptyToolStillBlocked:
    """Verify that actual tool calls are still properly blocked."""

    def test_shell_blocked_at_high_pressure(self):
        """shell tool at 85% pressure with enough evictable → blocked."""
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.85, evictable_count=80, tool_name="shell")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert result.category == "context_pressure_evict"

    def test_read_file_blocked_at_high_pressure(self):
        """read_file at 90% → blocked."""
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.90, evictable_count=70, tool_name="read_file")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_save_tools_allowed_at_high_pressure(self):
        """Save tools pass through even at high pressure."""
        save_tools = ["memory_write", "memory_read", "memory_list",
                      "plan_update", "plan_status", "plan_create",
                      "evict", "recall", "hard_reset"]
        for tool in save_tools:
            guard = ContextPressureGuard()
            ctx = _make_ctx(0.95, evictable_count=80, tool_name=tool)
            result = guard.check_pre(ctx)
            assert result is None, f"{tool} should be allowed through"

    def test_hard_reset_path_blocks_non_save_tools(self):
        """When _need_hard_reset=True AND conditions still bad, non-save tools are blocked."""
        guard = ContextPressureGuard()
        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=10, tool_name="shell")
        r = guard.check_pre(ctx1)
        assert r is not None
        assert guard._need_hard_reset is True

        # Non-save tool still blocked (conditions still bad: pressure >= 80% AND evictable < 60)
        ctx2 = _make_ctx(0.85, evictable_count=10, tool_name="write_file")
        result = guard.check_pre(ctx2)
        assert result is not None
        assert result.action == "block"
        assert result.category == "context_pressure_hard_reset"

    def test_hard_reset_lock_releases_when_pressure_drops(self):
        """When pressure drops below BLOCK_RATIO, lock auto-releases."""
        guard = ContextPressureGuard()
        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=10, tool_name="shell")
        guard.check_pre(ctx1)
        assert guard._need_hard_reset is True

        # Pressure dropped below 80% → lock releases, no block
        ctx2 = _make_ctx(0.50, evictable_count=10, tool_name="write_file")
        result = guard.check_pre(ctx2)
        assert result is None
        assert guard._need_hard_reset is False

    def test_hard_reset_clears_flag(self):
        """After hard_reset executes, _need_hard_reset clears."""
        guard = ContextPressureGuard()
        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=10, tool_name="shell")
        guard.check_pre(ctx1)
        assert guard._need_hard_reset is True

        # Simulate hard_reset execution via check_post
        ctx_post = _make_ctx(0.30, tool_name="hard_reset")
        guard.check_post(ctx_post)
        assert guard._need_hard_reset is False

        # Normal tools now work
        ctx3 = _make_ctx(0.30, evictable_count=10, tool_name="shell")
        result = guard.check_pre(ctx3)
        assert result is None


class TestInfiniteBlockRegression:
    """Simulate the exact scenario that caused the infinite block."""

    def test_kernel_precheck_loop_does_not_block(self):
        """Simulate kernel's pre-guard loop: repeated checks with tool_name=''.
        
        Before the fix, this would block every iteration.
        After the fix, all iterations pass through.
        """
        guard = ContextPressureGuard()
        # Simulate 10 iterations of the kernel loop pre-check
        for i in range(10):
            ctx = _make_ctx(
                pressure=0.85 + i * 0.01,  # Pressure increasing (injected msgs)
                evictable_count=max(5, 60 - i * 5),
                tool_name="",  # This is what kernel passes for pre-LLM check
            )
            result = guard.check_pre(ctx)
            assert result is None, f"Iteration {i}: should not block pre-LLM check"

    def test_after_pre_check_passes_tool_still_blocked(self):
        """After pre-LLM check passes, actual tool call still gets blocked."""
        guard = ContextPressureGuard()
        
        # Pre-LLM check passes
        ctx_pre = _make_ctx(0.90, evictable_count=80, tool_name="")
        assert guard.check_pre(ctx_pre) is None
        
        # But actual tool call is blocked
        ctx_tool = _make_ctx(0.90, evictable_count=80, tool_name="shell")
        result = guard.check_pre(ctx_tool)
        assert result is not None
        assert result.action == "block"


class TestSaveGuidanceSurvivalRange:
    """Regression: evict + hard_reset blocks must carry the survival-range test
    (cross-session truth -> memory, this-session progress -> plan) so the agent
    knows WHAT to extract before context is destroyed."""

    def _evict_msg(self):
        guard = ContextPressureGuard()
        ctx = _make_ctx(0.85, evictable_count=100, tool_name="shell")
        v = guard.check_pre(ctx)
        assert v is not None and v.category == "context_pressure_evict"
        return v.message

    def _hard_reset_msg(self):
        guard = ContextPressureGuard()
        # trigger hard_reset path: high pressure + too few evictable
        ctx1 = _make_ctx(0.90, evictable_count=10, tool_name="shell")
        r = guard.check_pre(ctx1)
        assert r is not None and r.category == "context_pressure_hard_reset"
        return r.message

    def test_evict_block_has_survival_range(self):
        msg = self._evict_msg()
        assert "EXTRACT NOW" in msg
        assert "survival-range test" in msg
        assert "cross-session truth" in msg
        assert "this-session progress" in msg
        assert "WHAT NOT" in msg  # noise guard: don't re-dump cheaply re-derivable facts

    def test_hard_reset_block_has_survival_range(self):
        msg = self._hard_reset_msg()
        assert "EXTRACT NOW" in msg
        assert "survival-range test" in msg
        assert "continuation summary lives only in the next window" in msg

    def test_two_hard_reset_blocks_identical(self):
        """The two hard_reset emission sites (recovery-path + threshold-path)
        must carry the SAME guidance text; the evict site is distinct."""
        import inspect
        import re
        from flagscale_agent.react.guard import context_pressure as cp
        src = inspect.getsource(cp)
        # Three emission sites total (2 hard_reset + 1 evict), each ending
        # with "Allowed tools:".
        assert src.count("Allowed tools:") == 3
        # Both hard_reset sites carry the reset-specific clause.
        assert src.count("Only memory survives a reset") == 2

        # Extract the two hard_reset blocks and compare after indent normalizing.
        lines = src.split("\n")
        starts = [i for i, l in enumerate(lines)
                  if "Eviction cannot free enough space" in l]
        assert len(starts) == 2, f"expected 2 hard_reset blocks, got {len(starts)}"
        blocks = []
        for s in starts:
            e = s
            while "Allowed tools:" not in lines[e]:
                e += 1
            blocks.append([re.sub(r"^\s+", "", x).rstrip()
                           for x in lines[s:e + 1] if x.strip()])
        assert blocks[0] == blocks[1], "two hard_reset blocks differ"
