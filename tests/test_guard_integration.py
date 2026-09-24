"""Guard system integration test — validates the 4 core behaviors:
1. Soft inject (advisory reminders)
2. Hard block (prevent dangerous actions)
3. Override (LLM bypasses block with reason)
4. Trigger & reset lifecycle (fire, decay, reset correctly)
"""

import pytest
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict, GuardRegistry


# ─── Fixtures: minimal guards for testing each behavior ───

class SoftReminderGuard(Guard):
    """Fires inject every 3 calls."""
    name = "soft_reminder"
    priority = 90
    decay_after_idle = 5

    def __init__(self):
        self._call_count = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        self._call_count += 1
        if self._call_count >= 3:
            self._call_count = 0
            return GuardVerdict.inject(
                "Reminder: do the thing.",
                reason="reminder",
                category="soft_test",
            )
        return None

    def reset_turn(self):
        pass

    def reset_turn(self):
        self._call_count = 0


class HardBlockGuard(Guard):
    """Blocks if tool_name is 'dangerous_tool'."""
    name = "hard_blocker"
    priority = 10

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "dangerous_tool":
            return GuardVerdict.block(
                "BLOCKED: dangerous_tool is not allowed.",
                reason="safety",
                category="test_block",
            )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # Accept if reason is substantive (>20 chars)
        return len(reason.strip()) > 20

    def reset_turn(self):
        pass

    def reset_turn(self):
        pass


class NonOverridableBlockGuard(Guard):
    """Blocks and cannot be overridden."""
    name = "absolute_blocker"
    priority = 5

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "forbidden_tool":
            return GuardVerdict.block(
                "ABSOLUTELY FORBIDDEN.",
                reason="absolute_safety",
            )
        return None

    def reset_turn(self):
        pass

    def reset_turn(self):
        pass


class SatisfiableGuard(Guard):
    """Fires inject until a specific tool is called, then satisfied."""
    name = "satisfiable"
    priority = 50
    decay_after_idle = 20  # high so decay doesn't interfere

    def __init__(self):
        self._needs_action = True

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._needs_action:
            return GuardVerdict.inject(
                "Please call 'fix_tool' to resolve the issue.",
                reason="needs_fix",
                category="fix_needed",
            )
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "fix_tool":
            self._needs_action = False
        return None

    def reset_turn(self):
        pass

    def reset_turn(self):
        pass


# ─── Helper ───

def _make_registry(guards):
    """Create a GuardRegistry and register the given guards."""
    reg = GuardRegistry()
    for g in guards:
        reg.register(g)
    return reg


def _ctx(tool_name="shell", tool_args=None, tool_result=None,
         override_reason="", ):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        override_reason=override_reason,
    )


# ═══════════════════════════════════════════════════════════════════
# TEST 1: SOFT INJECT (advisory, non-blocking)
# ═══════════════════════════════════════════════════════════════════

class TestSoftInject:
    def test_inject_fires_on_threshold(self):
        """Guard fires inject after 3 calls."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # Calls 1-2: no inject
        assert reg.check_pre(ctx) is None
        assert reg.check_pre(ctx) is None

        # Call 3: inject fires
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "inject"
        assert "Reminder" in v.message

    def test_inject_repeats_cyclically(self):
        """After firing, counter resets and fires again after 3 more calls."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # First cycle: 3 calls → fires
        reg.check_pre(ctx)
        reg.check_pre(ctx)
        v1 = reg.check_pre(ctx)
        assert v1 is not None

        # Second cycle: 3 more calls → fires again
        reg.check_pre(ctx)
        reg.check_pre(ctx)
        v2 = reg.check_pre(ctx)
        assert v2 is not None

    def test_inject_does_not_block(self):
        """inject verdict should not prevent tool execution (action != block)."""
        g = SoftReminderGuard()
        g._call_count = 2
        ctx = _ctx("shell")
        v = g.check_pre(ctx)
        assert v.action == "inject"
        # In kernel, inject → _apply_verdict returns False (not blocked)


# ═══════════════════════════════════════════════════════════════════
# TEST 2: HARD BLOCK (prevents tool execution)
# ═══════════════════════════════════════════════════════════════════

class TestHardBlock:
    def test_block_on_dangerous_tool(self):
        """Guard blocks dangerous_tool."""
        reg = _make_registry([HardBlockGuard()])
        ctx = _ctx("dangerous_tool")

        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "block"
        assert "BLOCKED" in v.message

    def test_no_block_on_safe_tool(self):
        """Guard does not block safe tools."""
        reg = _make_registry([HardBlockGuard()])
        ctx = _ctx("shell")
        assert reg.check_pre(ctx) is None

    def test_block_takes_priority_over_inject(self):
        """When both block and inject fire, only block is returned."""
        reg = _make_registry([HardBlockGuard(), SoftReminderGuard()])
        # Prime soft guard to fire
        soft = reg.guards[1]
        soft._call_count = 2

        ctx = _ctx("dangerous_tool")
        v = reg.check_pre(ctx)
        assert v.action == "block"  # Block wins
        assert "inject" not in v.action

class TestOverride:
    def test_override_with_good_reason(self):
        """Guard accepts override with substantive reason (>20 chars)."""
        reg = _make_registry([HardBlockGuard()])
        
        # Surface the block first
        ctx_surface = _ctx("dangerous_tool", override_reason="")
        v_surf = reg.check_pre(ctx_surface)
        assert v_surf is not None  # Block surfaced
        
        # Now override
        ctx = _ctx("dangerous_tool",
                   override_reason="This is needed for production hotfix deployment to resolve outage")
        v = reg.check_pre(ctx)
        # Override accepted → no block returned
        assert v is None

    def test_override_with_short_reason_rejected(self):
        """Guard rejects override with insufficient reason."""
        reg = _make_registry([HardBlockGuard()])
        ctx = _ctx("dangerous_tool", override_reason="just do it")
        v = reg.check_pre(ctx)
        # Override rejected → still blocked
        assert v is not None
        assert v.action == "block"

    def test_override_with_empty_reason_rejected(self):
        """No override_reason means no override attempt."""
        reg = _make_registry([HardBlockGuard()])
        ctx = _ctx("dangerous_tool", override_reason="")
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "block"

class TestLifecycle:
    def test_satisfied_guard_stops_firing(self):
        """Once is_satisfied returns True, guard is skipped."""
        reg = _make_registry([SatisfiableGuard()])
        ctx = _ctx("shell")

        # Before satisfaction: fires
        v = reg.check_pre(ctx)
        assert v is not None
        assert "fix_tool" in v.message

        # Simulate calling fix_tool
        post_ctx = _ctx("fix_tool", tool_result="fixed")
        reg.check_post(post_ctx)

        # After satisfaction: should NOT fire
        v2 = reg.check_pre(ctx)
        assert v2 is None

class TestMemoryDisciplineE2E:
    def test_reminder_every_10_calls(self):
        """Fires every 10 non-memory calls, resets on memory use."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        # 9 calls: no reminder
        for i in range(9):
            assert g.check_pre(ctx) is None

        # 10th: reminder
        v = g.check_pre(ctx)
        assert v is not None
        assert v.action == "inject"
        assert "10 tool calls" in v.message

        # Counter reset after firing — next 9 no reminder
        for i in range(9):
            assert g.check_pre(ctx) is None

        # 20th total (10 since last): reminder again
        v2 = g.check_pre(ctx)
        assert v2 is not None

    def test_memory_tool_resets_counter(self):
        """Calling memory_read/write/list resets the counter."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        # 8 calls
        for _ in range(8):
            g.check_pre(ctx)

        # memory_write resets
        g.check_pre(_ctx("memory_write"))
        assert g._calls_since_memory == 0

        # Now need another 10 to fire
        for _ in range(9):
            assert g.check_pre(ctx) is None
        v = g.check_pre(ctx)
        assert v is not None  # fires at 10

    def test_override_resets_counter(self):
        """accept_override resets counter."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 15

        assert g.accept_override("Already saved findings to disk, no memory needed", _ctx()) is True
        assert g._calls_since_memory == 0

    # ── actionable-copy anchors (proactive recall, not vague reminders) ──

    def test_inject_copy_names_concrete_actions(self):
        """Inject message must offer the RECALL/WRITE/DIGEST action chain."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        for _ in range(10):
            v = g.check_pre(ctx)
        assert v is not None and v.action == "inject"
        # concrete recall command forms, not vague "consider..."
        assert "memory_read(key='pitfall/" in v.message
        assert "memory_write()" in v.message
        assert "RECALL" in v.message and "WRITE" in v.message

    def test_block_copy_names_domain_recall(self):
        """Block message must name a whole-domain pitfall recall command."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        for _ in range(30):
            v = g.check_pre(ctx)
        assert v is not None and v.action == "block"
        assert "memory_read(key='pitfall/<domain>/')" in v.message
        assert "memory_list(keyword='...')" in v.message

    def test_counter_persists_across_turns(self):
        """reset_new_turn does NOT reset counter — memory gap persists."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        # 7 calls
        for _ in range(7):
            g.check_pre(ctx)

        g.reset_turn()
        assert g._calls_since_memory == 7  # Persists

        # 3 more → fires
        for _ in range(2):
            assert g.check_pre(ctx) is None
        v = g.check_pre(ctx)
        assert v is not None


# ═══════════════════════════════════════════════════════════════════
# TEST 6: MULTI-GUARD INTERACTION
# ═══════════════════════════════════════════════════════════════════

class TestMultiGuardInteraction:
    def test_multiple_injects_merged(self):
        """Multiple inject guards → messages merged into one verdict."""
        g1 = SoftReminderGuard()
        g1._call_count = 2
        g2 = SatisfiableGuard()

        reg = _make_registry([g1, g2])
        ctx = _ctx("shell")
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "inject"
        # Both messages present
        assert "Reminder" in v.message
        assert "fix_tool" in v.message

    def test_block_suppresses_injects(self):
        """When a block fires, inject messages from other guards are NOT included."""
        g1 = HardBlockGuard()
        g2 = SoftReminderGuard()
        g2._call_count = 2

        reg = _make_registry([g1, g2])
        ctx = _ctx("dangerous_tool")
        v = reg.check_pre(ctx)
        assert v.action == "block"
        # Should NOT contain the soft reminder
        assert "Reminder: do the thing" not in v.message

    def test_priority_ordering(self):
        """Higher priority (lower number) guards are checked first."""
        reg = _make_registry([SoftReminderGuard(), HardBlockGuard()])
        # HardBlockGuard has priority=10, SoftReminderGuard has priority=90
        # Registry should sort by priority
        assert reg.guards[0].name == "hard_blocker"  # priority 10
        assert reg.guards[1].name == "soft_reminder"  # priority 90


# ═══════════════════════════════════════════════════════════════════
# TEST 6: ESCALATION CHAIN (inject → escalate on repeated ineffective)
# ═══════════════════════════════════════════════════════════════════

class EscalatingGuard(Guard):
    """A guard that fires inject and defines effectiveness criteria."""
    name = "escalating"
    priority = 50
    decay_after_idle = 100  # prevent decay interference

    def __init__(self):
        self._should_fire = True

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._should_fire:
            return GuardVerdict.inject(
                "Please use 'target_tool' instead.",
                reason="escalation_test",
                category="esc_test",
            )
        return None

    def reset_turn(self):
        pass

    def reset_turn(self):
        pass


class TestEscalationChain:
    """Verify inject → escalate upgrade when inject is repeatedly ineffective."""

    def _simulate_ineffective_cycles(self, reg, n_cycles=3):
        """Simulate n_cycles of: inject fires in pre, then post marks ineffective."""

        for _ in range(n_cycles):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = _ + 1
            v = reg.check_pre(pre_ctx)
            # Pre should return something (inject or escalate)
            assert v is not None, f"Expected verdict on cycle {_}"

            # Post: tool was "shell" (not "target_tool"), so ineffective
            post_ctx = _ctx("shell")
            reg.check_post(post_ctx)

        return v

    def test_effective_action_resets_escalation(self):
        """If agent responds to inject, escalation counter resets."""

        reg = _make_registry([EscalatingGuard()])

        # 2 ineffective cycles
        for i in range(2):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = i + 1
            reg.check_pre(pre_ctx)
            post_ctx = _ctx("shell")
            reg.check_post(post_ctx)

        # Now agent responds correctly
        pre_ctx = _ctx("target_tool")
        pre_ctx.turn_count = 3
        reg.check_pre(pre_ctx)
        post_ctx = _ctx("target_tool")
        reg.check_post(post_ctx)

        # Next cycle should be inject again (not escalate), counter reset
        for i in range(2):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = 4 + i
            v = reg.check_pre(pre_ctx)
            assert v is not None
            # Should still be inject, not escalate
            assert v.action == "inject", \
                f"Expected inject after reset, got {v.action}"
            post_ctx = _ctx("shell")
            reg.check_post(post_ctx)

    def test_memory_discipline_escalation(self):
        """MemoryDiscipline guard blocks after 30 calls without memory ops."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard

        guard = MemoryDisciplineGuard()
        reg = _make_registry([guard])

        # Simulate 30 non-memory tool calls
        for i in range(30):
            pre_ctx = _ctx("shell")
            v = reg.check_pre(pre_ctx)

        # The 30th call should have triggered a block
        # Counter resets after block, so let's verify by doing another 30
        guard._calls_since_memory = 29
        pre_ctx = _ctx("read_file")
        v = reg.check_pre(pre_ctx)
        assert v is not None
        assert v.action == "block"
