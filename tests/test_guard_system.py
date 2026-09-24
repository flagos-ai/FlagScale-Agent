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

"""Integration tests for the new guard system."""
import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard


def make_ctx(tool_name="", tool_args=None, tool_result=""):
    """Helper to create a GuardContext."""
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = tool_result
    ctx.classify_fn = None
    ctx.current_experiment_name = ""
    ctx.experiment_diff_fn = None
    return ctx



    def test_memory_discipline_reminder_threshold(self):
        """Memory discipline reminds every 10 non-memory tool calls."""
        guard = MemoryDisciplineGuard()

        # 9 calls — no reminder
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None, f"Unexpected reminder on call {i+1}: {result}"

        # 10th call — triggers reminder, counter resets
        ctx = make_ctx("shell", {"command": "echo 10"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"
        assert "10 tool calls" in result.message
        assert guard._calls_since_memory == 0  # Reset after firing

        # Next 9 calls — no reminder again
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None

        # 20th total call (10th since last reminder) — triggers again
        ctx = make_ctx("shell", {"command": "echo again"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"

        # memory_read resets counter
        ctx = make_ctx("memory_read", {"key": "test"}, tool_result="value")
        result = guard.check_pre(ctx)
        assert result is None
        assert guard._calls_since_memory == 0

    def test_memory_discipline_block_at_30_calls(self):
        """After 30 non-memory calls, guard blocks (overridable)."""
        guard = MemoryDisciplineGuard()

        # First 29 calls — should get injects at 10, 20 but no block
        for i in range(29):
            ctx = make_ctx("shell", {"command": "ls"})
            result = guard.check_pre(ctx)
            if result:
                assert result.action == "inject"

        # 30th call → block
        ctx = make_ctx("shell", {"command": "ls"})
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "30" in result.message

    def test_memory_discipline_block_overridable(self):
        """Block at 30 can be overridden with a reason."""
        guard = MemoryDisciplineGuard()

        ctx = make_ctx("shell", {"command": "ls"})
        assert guard.accept_override("No memory needed for this pure refactoring task", ctx)
        # Counter should reset
        assert guard._calls_since_memory == 0

    def test_memory_discipline_block_not_overridable_without_reason(self):
        """Block override rejected if reason is too short."""
        guard = MemoryDisciplineGuard()

        ctx = make_ctx("shell", {"command": "ls"})
        assert not guard.accept_override("ok", ctx)




class TestMemoryEvolution:
    """MemoryDisciplineGuard no longer fires at completion time — that was moved
    to VerificationGuard's _TEXT_COMPLETE_HYGIENE gate. These tests verify the
    new behavior: text-only (no tool_name) always returns None."""

    def test_text_complete_returns_none(self):
        """text-only [TASK_COMPLETE] → None (completion check moved to VerificationGuard)."""
        guard = MemoryDisciplineGuard()
        ctx = MagicMock(spec=GuardContext)
        ctx.tool_name = ""
        ctx.tool_args = {}
        ctx.tool_result = ""
        ctx.assistant_text = "Done. [TASK_COMPLETE]"
        ctx.classify_fn = None
        assert guard.check_pre(ctx) is None

    def test_no_evolution_reminded_attribute(self):
        """_evolution_reminded no longer exists on the guard."""
        guard = MemoryDisciplineGuard()
        assert not hasattr(guard, "_evolution_reminded")

    def test_no_has_memory_review_attribute(self):
        """_has_memory_review no longer exists on the guard."""
        guard = MemoryDisciplineGuard()
        assert not hasattr(guard, "_has_memory_review")


# ── Override Hint Tests ──

class TestOverrideHint:
    def test_override_hint_format(self):
        """Override hint should contain _override_reason instruction."""
        from flagscale_agent.react.guard import _OVERRIDE_HINT
        assert "_override_reason" in _OVERRIDE_HINT
        assert "OVERRIDE REQUIRED" in _OVERRIDE_HINT

    def test_hint_added_to_block(self):
        """Block verdicts get override hint appended by registry."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class BlockingGuard(Guard):
            name = "blocker"
            def check_pre(self, ctx):
                return GuardVerdict.block("[Blocked] reason", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(BlockingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="")
        result = reg.check_pre(ctx)
        assert "_override_reason" in result.message
        assert "OVERRIDE REQUIRED" in result.message

    def test_hint_not_re_added_after_rejected_override(self):
        """If override was attempted and rejected, no re-hint."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class BlockingGuard(Guard):
            name = "blocker"
            def check_pre(self, ctx):
                return GuardVerdict.block("[Blocked] still wrong", reason="test", category="test")
            def accept_override(self, reason, ctx):
                return False  # Always reject

        reg = GuardRegistry()
        reg.register(BlockingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="I already tried")
        result = reg.check_pre(ctx)
        # Override was attempted but rejected — hint should not be re-added
        assert "OVERRIDE REQUIRED" not in result.message

    def test_escalate_hint_added(self):
        """Escalate verdicts get escalate hint telling LLM not to retry."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class EscalatingGuard(Guard):
            name = "escalator"
            def check_pre(self, ctx):
                return GuardVerdict.escalate("[Escalated] forbidden", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(EscalatingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="")
        result = reg.check_pre(ctx)
        assert "ESCALATED" in result.message
        assert "DO NOT retry" in result.message
        assert "OVERRIDE REQUIRED" not in result.message

    def test_escalate_cannot_be_overridden(self):
        """Escalate ignores override_reason."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class EscalatingGuard(Guard):
            name = "escalator"
            def check_pre(self, ctx):
                return GuardVerdict.escalate("[Escalated] forbidden", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(EscalatingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="I have a good reason")
        result = reg.check_pre(ctx)
        # Still escalated — override_reason is ignored
        assert result is not None
        assert "ESCALATED" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
