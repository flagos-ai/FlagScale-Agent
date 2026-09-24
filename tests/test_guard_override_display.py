"""
Test guard override behavior and display.

Verifies that:
1. Override display shows every time an override is accepted
2. Override actually allows the tool to execute (not just display)
3. No duplicate guard checks cause duplicate displays
"""
import pytest
from flagscale_agent.react.guard import GuardRegistry, GuardContext, Guard, GuardVerdict


class MockGuard(Guard):
    """Test guard that blocks shell commands but accepts overrides."""
    
    def __init__(self, name="test_guard"):
        self.name = name
        self.priority = 50
        self.check_pre_calls = 0
        self.block_count = 0
    
    def check_pre(self, ctx: GuardContext):
        self.check_pre_calls += 1
        if ctx.tool_name == "shell":
            self.block_count += 1
            return GuardVerdict(action="block", message="Test block message", category="test_category", reason="test")
        return None
    
    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        return len(reason) > 10


class TestOverrideDisplay:
    """Test that override display always shows when override is accepted."""
    
    def test_override_displays_every_time(self, capsys):
        """Override should display every time it's used, not deduplicated."""
        reg = GuardRegistry()
        guard = MockGuard("display_test")
        reg.register(guard)
        
        # First, surface the block (no override yet) so registry remembers this guard
        ctx_no_override = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        verdict_surface = reg.check_pre(ctx_no_override)
        assert verdict_surface is not None  # Block surfaced
        
        # Now supply the override reason
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override reason that is long enough",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # First override call - should display
        verdict1 = reg.check_pre(ctx)
        assert verdict1 is None  # None means allowed
        
        # Second call in same turn - override should display again
        verdict2 = reg.check_pre(ctx)
        assert verdict2 is None
        
        # Third call in same turn - override should still display
        verdict3 = reg.check_pre(ctx)
        assert verdict3 is None
        
        # Check output - should show 3 times
        captured = capsys.readouterr()
        override_count = captured.out.count("Guard override")
        assert override_count == 3, f"Expected 3 override displays, got {override_count}"


class TestOverrideLogic:
    """Test override acceptance and rejection logic."""
    
    def test_override_accepted_passes_tool(self, capsys):
        """When override is accepted, tool should be allowed (verdict=None)."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        # Surface the block first
        ctx_surface = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        verdict_s = reg.check_pre(ctx_surface)
        assert verdict_s is not None  # Block surfaced
        
        # Now override
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="This is a valid override reason",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is None  # None = allowed
        
        # Verify override message was displayed
        captured = capsys.readouterr()
        assert "Guard override" in captured.out
        assert "valid override reason" in captured.out
    
    def test_override_rejected_blocks_tool(self):
        """When override reason is invalid, tool should be blocked."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="short",  # Too short, will be rejected
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
    
    def test_no_override_reason_blocks_tool(self):
        """Without override_reason, blocked tool should stay blocked."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",  # No override
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"


class TestMultipleGuards:
    """Test override behavior with multiple guards."""
    
    def test_different_guards_both_display_override(self, capsys):
        """With Bug C fix: only the last-surfaced guard releases; others still block."""
        reg = GuardRegistry()
        guard1 = MockGuard("guard1")
        guard1.priority = 10  # surfaces first
        guard2 = MockGuard("guard2")
        guard2.priority = 20
        reg.register(guard1)
        reg.register(guard2)
        
        # Surface guard1 first
        ctx_surface = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        verdict_s = reg.check_pre(ctx_surface)
        assert verdict_s is not None and verdict_s.guard_name == "guard1"
        
        # Now override: only guard1 (last surfaced) releases; guard2 still blocks
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override for guard1",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        # guard1 released, guard2 now surfaces
        assert verdict is not None and verdict.guard_name == "guard2"
        
        captured = capsys.readouterr()
        # Only ONE override display (guard1)
        override_count = captured.out.count("Guard override")
        assert override_count == 1, f"Expected 1 override display (guard1 only), got {override_count}"


class TestResetTurn:
    """Test that reset_turn doesn't affect override display."""
    
    def test_reset_turn_does_not_prevent_override_display(self, capsys):
        """reset_turn() should not affect override display behavior."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        # Surface first
        ctx_surface = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        verdict_s = reg.check_pre(ctx_surface)
        assert verdict_s is not None
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # First turn
        verdict1 = reg.check_pre(ctx)
        assert verdict1 is None
        
        # Reset turn
        reg.reset_turn()
        
        # Second turn - should still display override
        verdict2 = reg.check_pre(ctx)
        assert verdict2 is None
        
        # Both overrides should be visible
        captured = capsys.readouterr()
        override_count = captured.out.count("Guard override")
        assert override_count == 2, f"Expected 2 override displays, got {override_count}"
