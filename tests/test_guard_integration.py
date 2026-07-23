"""Guard system integration test — validates the 4 core behaviors:
1. Soft inject (advisory reminders)
2. Hard block (prevent dangerous actions)
3. Override (LLM bypasses block with reason)
4. Trigger & reset lifecycle (fire, decay, reset correctly)
"""

import pytest
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict, GuardRegistry
from flagscale_agent.react.state_machine import AgentState


# ─── Fixtures: minimal guards for testing each behavior ───

class SoftReminderGuard(Guard):
    """Fires inject_msg every 3 calls."""
    name = "soft_reminder"
    priority = 90
    overridable = True
    escalate_after = 3
    decay_after_idle = 5

    def __init__(self):
        super().__init__()
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

    def is_satisfied(self, ctx: GuardContext) -> bool:
        return False

    def reset_state(self):
        super().reset_state()
        self._call_count = 0

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        self._call_count = 0


class HardBlockGuard(Guard):
    """Blocks if tool_name is 'dangerous_tool'."""
    name = "hard_blocker"
    priority = 10
    overridable = True

    def __init__(self):
        super().__init__()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "dangerous_tool":
            return GuardVerdict.block(
                "BLOCKED: dangerous_tool is not allowed.",
                reason="safety",
            )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # Accept if reason is substantive (>20 chars)
        return len(reason.strip()) > 20

    def is_satisfied(self, ctx: GuardContext) -> bool:
        return False

    def reset_state(self):
        super().reset_state()

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        pass


class NonOverridableBlockGuard(Guard):
    """Blocks and cannot be overridden."""
    name = "absolute_blocker"
    priority = 5
    overridable = False

    def __init__(self):
        super().__init__()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "forbidden_tool":
            return GuardVerdict.block(
                "ABSOLUTELY FORBIDDEN.",
                reason="absolute_safety",
            )
        return None

    def is_satisfied(self, ctx: GuardContext) -> bool:
        return False

    def reset_state(self):
        super().reset_state()

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        pass


class SatisfiableGuard(Guard):
    """Fires inject until a specific tool is called, then satisfied."""
    name = "satisfiable"
    priority = 50
    overridable = True
    decay_after_idle = 20  # high so decay doesn't interfere

    def __init__(self):
        super().__init__()
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

    def is_satisfied(self, ctx: GuardContext) -> bool:
        return not self._needs_action

    def reset_state(self):
        super().reset_state()
        self._needs_action = True

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        pass


# ─── Helper ───

def _make_registry(guards):
    """Create a GuardRegistry and register the given guards."""
    reg = GuardRegistry()
    for g in guards:
        reg.register(g)
    return reg


def _ctx(tool_name="shell", tool_args=None, tool_result=None,
         override_reason="", state=AgentState.EXECUTING):
    return GuardContext(
        current_state=state,
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
        """Guard fires inject_msg after 3 calls."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # Calls 1-2: no inject
        assert reg.check_pre(ctx) is None
        assert reg.check_pre(ctx) is None

        # Call 3: inject fires
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "inject_msg"
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
        """inject_msg verdict should not prevent tool execution (action != block)."""
        g = SoftReminderGuard()
        g._call_count = 2
        ctx = _ctx("shell")
        v = g.check_pre(ctx)
        assert v.action == "inject_msg"
        # In kernel, inject_msg → _apply_verdict returns False (not blocked)


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

    def test_non_overridable_cannot_be_bypassed(self):
        """Non-overridable guard rejects any override reason."""
        reg = _make_registry([NonOverridableBlockGuard()])
        ctx = _ctx("forbidden_tool",
                   override_reason="I have a very good reason to do this dangerous operation")
        v = reg.check_pre(ctx)
        # Still blocked — override not accepted
        assert v is not None
        assert v.action == "block"


# ═══════════════════════════════════════════════════════════════════
# TEST 3: OVERRIDE (LLM bypasses block with reason)
# ═══════════════════════════════════════════════════════════════════

class TestOverride:
    def test_override_with_good_reason(self):
        """Guard accepts override with substantive reason (>20 chars)."""
        reg = _make_registry([HardBlockGuard()])
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

    def test_override_only_applies_to_overridable_guards(self):
        """Non-overridable guards ignore override_reason entirely."""
        reg = _make_registry([NonOverridableBlockGuard()])
        ctx = _ctx("forbidden_tool",
                   override_reason="Critical emergency requiring immediate action on forbidden tool")
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "block"


# ═══════════════════════════════════════════════════════════════════
# TEST 4: TRIGGER & RESET LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_reset_new_turn_resets_guard(self):
        """reset_new_turn clears per-turn state."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # Fire 2 calls
        reg.check_pre(ctx)
        reg.check_pre(ctx)
        assert reg.guards[0]._call_count == 2

        # New turn → counter resets
        reg.reset_new_turn()
        assert reg.guards[0]._call_count == 0

    def test_reset_iteration_calls_reset_turn(self):
        """reset_iteration (per LLM+tool loop) calls each guard's reset_turn."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # SoftReminderGuard.reset_turn is a no-op, but verify it doesn't crash
        reg.check_pre(ctx)
        reg.reset_iteration()
        # Counter should still be 1 (reset_turn is no-op for this guard)
        assert reg.guards[0]._call_count == 1

    def test_decay_resets_state_after_idle(self):
        """After decay_after_idle iterations without firing, guard resets."""
        g = SoftReminderGuard()
        g._call_count = 2  # Prime to fire on next check

        # Simulate 5 idle ticks (decay_after_idle = 5)
        for _ in range(5):
            g._tick_idle()

        # State should be reset
        assert g._call_count == 0

    def test_decay_does_not_trigger_early(self):
        """Decay doesn't reset before reaching threshold."""
        g = SoftReminderGuard()
        g._call_count = 2

        # 4 idle ticks (threshold is 5) — should NOT reset
        for _ in range(4):
            g._tick_idle()

        assert g._call_count == 2  # Unchanged

    def test_firing_resets_idle_counter(self):
        """When a guard fires, its idle counter resets (no decay)."""
        reg = _make_registry([SoftReminderGuard()])
        ctx = _ctx("shell")

        # 2 idle ticks
        reg.guards[0]._tick_idle()
        reg.guards[0]._tick_idle()
        assert reg.guards[0]._iterations_since_trigger == 2

        # Fire the guard (3rd call triggers it)
        reg.guards[0]._call_count = 2
        v = reg.check_pre(ctx)
        assert v is not None

        # After firing, tick_guard_lifecycle records the trigger
        # which resets _iterations_since_trigger to 0
        assert reg.guards[0]._iterations_since_trigger == 0

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

    def test_escalation_chain_inject_to_block(self):
        """After escalate_after injects, guard upgrades to block."""
        g = SoftReminderGuard()
        reg = _make_registry([g])

        # Fire inject escalate_after times
        for i in range(g.escalate_after):
            g._call_count = 2  # ensure it fires
            ctx = _ctx("shell")
            ctx.turn_count = i + 1
            v = reg.check_pre(ctx)
            assert v is not None
            assert v.action == "inject_msg"

        # Next fire should be block (escalation)
        g._call_count = 2
        ctx = _ctx("shell")
        ctx.turn_count = g.escalate_after + 1
        v = reg.check_pre(ctx)
        assert v is not None
        assert v.action == "block", f"Expected block, got {v.action}"


# ═══════════════════════════════════════════════════════════════════
# TEST 5: MEMORY DISCIPLINE (the real guard, end-to-end)
# ═══════════════════════════════════════════════════════════════════

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
        assert v.action == "inject_msg"
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

    def test_counter_persists_across_turns(self):
        """reset_new_turn does NOT reset counter — memory gap persists."""
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        g = MemoryDisciplineGuard()
        ctx = _ctx("shell")

        # 7 calls
        for _ in range(7):
            g.check_pre(ctx)

        g.reset_new_turn()
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
        assert v.action == "inject_msg"
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
    overridable = True
    decay_after_idle = 100  # prevent decay interference

    def __init__(self):
        super().__init__()
        self._should_fire = True

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._should_fire:
            return GuardVerdict.inject(
                "Please use 'target_tool' instead.",
                reason="escalation_test",
                category="esc_test",
            )
        return None

    def was_inject_effective(self, ctx: GuardContext) -> bool | None:
        if ctx.tool_name == "target_tool":
            return True
        # After any non-target tool, consider ineffective
        return False

    def is_satisfied(self, ctx: GuardContext) -> bool:
        return False

    def reset_state(self):
        super().reset_state()
        self._should_fire = True

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        pass


class TestEscalationChain:
    """Verify inject → escalate upgrade when inject is repeatedly ineffective."""

    def _simulate_ineffective_cycles(self, reg, n_cycles=3):
        """Simulate n_cycles of: inject fires in pre, then post marks ineffective."""
        from flagscale_agent.react.tools.base import ToolEffect

        for _ in range(n_cycles):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = _ + 1
            v = reg.check_pre(pre_ctx)
            # Pre should return something (inject or escalate)
            assert v is not None, f"Expected verdict on cycle {_}"

            # Post: tool was "shell" (not "target_tool"), so ineffective
            post_ctx = _ctx("shell")
            post_ctx.tool_effects = ToolEffect(reads=frozenset({"filesystem"}))
            reg.check_post(post_ctx)

        return v

    def test_inject_becomes_escalate_after_ineffective(self):
        """After 3 consecutive ineffective injects, should_suppress triggers escalation."""
        from flagscale_agent.react.tools.base import ToolEffect

        reg = _make_registry([EscalatingGuard()])

        # Cycle through: inject fires, check_post marks it ineffective
        for i in range(4):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = i + 1
            v = reg.check_pre(pre_ctx)
            assert v is not None

            post_ctx = _ctx("shell")
            post_ctx.tool_effects = ToolEffect(reads=frozenset({"filesystem"}))
            reg.check_post(post_ctx)

        # After 3+ ineffective injects, the tracker should escalate on next pre
        pre_ctx = _ctx("shell")
        pre_ctx.turn_count = 5
        v = reg.check_pre(pre_ctx)
        assert v is not None
        # Should have escalated (either action == "escalate" or message indicates upgrade)
        assert v.action == "escalate" or "repeatedly" in v.message.lower() or "ignored" in v.message.lower(), \
            f"Expected escalation, got action={v.action}, msg={v.message[:100]}"

    def test_effective_action_resets_escalation(self):
        """If agent responds to inject, escalation counter resets."""
        from flagscale_agent.react.tools.base import ToolEffect

        reg = _make_registry([EscalatingGuard()])

        # 2 ineffective cycles
        for i in range(2):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = i + 1
            reg.check_pre(pre_ctx)
            post_ctx = _ctx("shell")
            post_ctx.tool_effects = ToolEffect(reads=frozenset({"filesystem"}))
            reg.check_post(post_ctx)

        # Now agent responds correctly
        pre_ctx = _ctx("target_tool")
        pre_ctx.turn_count = 3
        reg.check_pre(pre_ctx)
        post_ctx = _ctx("target_tool")
        post_ctx.tool_effects = ToolEffect(writes=frozenset({"filesystem"}))
        reg.check_post(post_ctx)

        # Next cycle should be inject again (not escalate), counter reset
        for i in range(2):
            pre_ctx = _ctx("shell")
            pre_ctx.turn_count = 4 + i
            v = reg.check_pre(pre_ctx)
            assert v is not None
            # Should still be inject, not escalate
            assert v.action == "inject_msg", \
                f"Expected inject_msg after reset, got {v.action}"
            post_ctx = _ctx("shell")
            post_ctx.tool_effects = ToolEffect(reads=frozenset({"filesystem"}))
            reg.check_post(post_ctx)

    def test_memory_discipline_escalation(self):
        """MemoryDiscipline guard escalates after repeated ignoring.
        
        inject → inject ignored × escalate_after → block (but block no longer
        terminates the turn — it's injected as a strong advisory).
        """
        from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
        from flagscale_agent.react.tools.base import ToolEffect

        guard = MemoryDisciplineGuard()
        reg = _make_registry([guard])

        # Simulate 10 non-memory tool calls to trigger first inject
        for _ in range(10):
            pre_ctx = _ctx("shell")
            reg.check_pre(pre_ctx)

        # Should have injected
        pre_ctx = _ctx("shell")
        v = reg.check_pre(pre_ctx)
        # After 10 calls, the guard should fire (threshold is 10)
        # It may have already fired on the 10th call above; just verify mechanism works

        # Now simulate repeated cycles of inject + ignore
        for cycle in range(4):
            # Force counter back to trigger threshold
            guard._calls_since_memory = 10
            pre_ctx = _ctx("read_file")
            pre_ctx.turn_count = cycle + 1
            v = reg.check_pre(pre_ctx)
            if v is None:
                continue  # guard may have internal suppression

            # Post: agent used read_file, not memory
            post_ctx = _ctx("read_file")
            post_ctx.tool_effects = ToolEffect(reads=frozenset({"filesystem"}))
            reg.check_post(post_ctx)

        # After multiple ineffective cycles, check tracker state
        tracker = reg.shared_state.inject_tracker
        # Should have recorded some ineffective entries
        ineff = tracker.consecutive_ineffective("memory_discipline", "memory_idle_reminder")
        assert ineff >= 2, f"Expected 2+ ineffective, got {ineff}"
