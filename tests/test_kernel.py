"""Tests for state_machine, guard, and kernel modules (Phase 1)."""

import sys
import importlib
import pytest
from unittest.mock import MagicMock

# Import new modules directly to avoid agent.py's heavy dependencies
import importlib.util, pathlib

def _load(rel):
    base = pathlib.Path(__file__).parent.parent / "flagscale_agent" / "react"
    spec = importlib.util.spec_from_file_location(rel, base / (rel.replace(".", "/") + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[rel] = mod  # register before exec so dataclasses can find __module__
    spec.loader.exec_module(mod)
    return mod

_sm = _load("state_machine")
StateMachine = _sm.StateMachine
AgentState = _sm.AgentState
StateTransition = _sm.StateTransition

# guard/__init__.py
_guard_spec = importlib.util.spec_from_file_location(
    "guard", pathlib.Path(__file__).parent.parent / "flagscale_agent" / "react" / "guard" / "__init__.py"
)
_guard = importlib.util.module_from_spec(_guard_spec)
# inject state_machine into guard's namespace before exec
_guard.AgentState = AgentState
sys.modules["flagscale_agent.react.state_machine"] = _sm
sys.modules["guard"] = _guard  # register before exec so dataclasses can find __module__
_guard_spec.loader.exec_module(_guard)
Guard = _guard.Guard
GuardContext = _guard.GuardContext
GuardVerdict = _guard.GuardVerdict
GuardRegistry = _guard.GuardRegistry


# ── StateMachine tests ────────────────────────────────────────────────────────

class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.current_state == AgentState.IDLE

    def test_valid_transition(self):
        sm = StateMachine()
        ok = sm.transition(AgentState.EXECUTING, reason="test")
        assert ok
        assert sm.current_state == AgentState.EXECUTING

    def test_invalid_transition_rejected(self):
        sm = StateMachine()
        # IDLE → COMPLETED is not a valid transition
        ok = sm.transition(AgentState.COMPLETED)
        assert not ok
        assert sm.current_state == AgentState.IDLE  # unchanged

    def test_terminal_states_have_no_transitions(self):
        for terminal in [AgentState.COMPLETED, AgentState.FAILED, AgentState.INTERRUPTED]:
            sm = StateMachine(initial_state=terminal)
            assert sm.is_terminal()
            assert not sm.can_transition(AgentState.EXECUTING)

    def test_force_transition_bypasses_validation(self):
        sm = StateMachine()
        sm.force_transition(AgentState.COMPLETED, reason="forced")
        assert sm.current_state == AgentState.COMPLETED

    def test_history_recorded(self):
        sm = StateMachine()
        sm.transition(AgentState.EXECUTING)
        sm.transition(AgentState.REVIEWING)
        assert len(sm.history) == 2
        assert sm.history[0].from_state == AgentState.IDLE
        assert sm.history[0].to_state == AgentState.EXECUTING

    def test_phase_name_compat(self):
        sm = StateMachine(initial_state=AgentState.EXECUTING)
        assert sm.get_phase_name() == "executing"

    def test_from_phase_name(self):
        sm = StateMachine.from_phase_name("planning")
        assert sm.current_state == AgentState.PLANNING

    def test_from_unknown_phase_name_defaults_to_idle(self):
        sm = StateMachine.from_phase_name("nonexistent")
        assert sm.current_state == AgentState.IDLE

    def test_executing_can_loop_to_itself(self):
        sm = StateMachine(initial_state=AgentState.EXECUTING)
        ok = sm.transition(AgentState.EXECUTING)
        assert ok


# ── Guard tests ───────────────────────────────────────────────────────────────

class ConcreteGuard(Guard):
    name = "test_guard"
    priority = 10

    def __init__(self, verdict=None):
        self._verdict = verdict

    def check_pre(self, ctx):
        return self._verdict

    def check_post(self, ctx):
        return self._verdict


class TestGuardContext:
    def test_default_context(self):
        ctx = GuardContext()
        assert ctx.tool_name == ""
        assert ctx.current_state == AgentState.IDLE

    def test_context_with_values(self):
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            current_state=AgentState.EXECUTING,
        )
        assert ctx.tool_name == "shell"
        assert ctx.current_state == AgentState.EXECUTING


class TestGuardVerdict:
    def test_allow_factory(self):
        v = GuardVerdict.allow()
        assert v.action == "allow"

    def test_block_factory(self):
        v = GuardVerdict.block("stop!", reason="dangerous")
        assert v.action == "block"
        assert v.message == "stop!"
        assert v.reason == "dangerous"

    def test_inject_factory(self):
        v = GuardVerdict.inject("reminder")
        assert v.action == "inject_msg"

    def test_compact_factory(self):
        v = GuardVerdict.compact(reason="pressure")
        assert v.action == "force_compact"

    def test_escalate_factory(self):
        v = GuardVerdict.escalate("review needed")
        assert v.action == "escalate"

    def test_redirect_factory(self):
        v = GuardVerdict.redirect("re-plan", metadata={"key": "val"})
        assert v.action == "redirect"
        assert v.metadata == {"key": "val"}


class TestGuard:
    def test_should_activate_default(self):
        g = ConcreteGuard()
        ctx = GuardContext(current_state=AgentState.EXECUTING)
        assert g.should_activate(ctx)

    def test_should_not_activate_wrong_state(self):
        g = ConcreteGuard()
        ctx = GuardContext(current_state=AgentState.IDLE)
        assert not g.should_activate(ctx)

    def test_should_not_activate_wrong_tool(self):
        g = ConcreteGuard()
        g.activate_on_tools = {"shell"}
        ctx = GuardContext(current_state=AgentState.EXECUTING, tool_name="read_file")
        assert not g.should_activate(ctx)

    def test_should_activate_matching_tool(self):
        g = ConcreteGuard()
        g.activate_on_tools = {"shell"}
        ctx = GuardContext(current_state=AgentState.EXECUTING, tool_name="shell")
        assert g.should_activate(ctx)


class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = ConcreteGuard()
        g1.priority = 20
        g2 = ConcreteGuard()
        g2.priority = 5
        reg.register(g1)
        reg.register(g2)
        assert reg.guards[0].priority == 5
        assert reg.guards[1].priority == 20

    def test_check_pre_first_verdict_wins(self):
        reg = GuardRegistry()
        g1 = ConcreteGuard(verdict=GuardVerdict.block("blocked by g1"))
        g1.priority = 10
        g2 = ConcreteGuard(verdict=GuardVerdict.inject("injected by g2"))
        g2.priority = 20
        reg.register(g1)
        reg.register(g2)
        ctx = GuardContext(current_state=AgentState.EXECUTING)
        verdict = reg.check_pre(ctx)
        assert verdict.action == "block"
        assert verdict.message == "blocked by g1"

    def test_check_pre_returns_none_when_all_allow(self):
        reg = GuardRegistry()
        g = ConcreteGuard(verdict=None)
        reg.register(g)
        ctx = GuardContext(current_state=AgentState.EXECUTING)
        assert reg.check_pre(ctx) is None

    def test_reset_turn_called_on_all_guards(self):
        reg = GuardRegistry()
        g1 = MagicMock(spec=Guard)
        g1.priority = 10
        g2 = MagicMock(spec=Guard)
        g2.priority = 20
        reg._guards = [g1, g2]
        reg.reset_turn()
        g1.reset_turn.assert_called_once()
        g2.reset_turn.assert_called_once()


class TestGuardContextToolEffects:
    """Tests for tool_effects field in GuardContext."""

    def test_default_effects_empty(self):
        from flagscale_agent.react.tools.base import ToolEffect
        ctx = GuardContext()
        assert ctx.tool_effects == ToolEffect()
        assert ctx.tool_effects.is_read_only

    def test_effects_from_tool(self):
        from flagscale_agent.react.tools.base import ToolEffect, EFFECT_SHELL
        ctx = GuardContext(
            tool_name="shell",
            tool_effects=EFFECT_SHELL,
            current_state=AgentState.EXECUTING,
        )
        assert ctx.tool_effects.touches_process
        assert ctx.tool_effects.touches_filesystem
        assert not ctx.tool_effects.is_read_only

    def test_guard_can_use_effects_for_decision(self):
        """A guard can inspect tool_effects to make decisions."""
        from flagscale_agent.react.tools.base import ToolEffect, EFFECT_WRITE_FS

        class WriteBlockGuard(Guard):
            name = "write_blocker"
            priority = 1

            def check_pre(self, ctx):
                if ctx.tool_effects.is_write:
                    return GuardVerdict.block("writes blocked", reason="read-only mode")
                return None

        reg = GuardRegistry()
        reg.register(WriteBlockGuard())

        # Write tool → blocked
        ctx_write = GuardContext(
            tool_name="write_file",
            tool_effects=EFFECT_WRITE_FS,
            current_state=AgentState.EXECUTING,
        )
        verdict = reg.check_pre(ctx_write)
        assert verdict is not None
        assert verdict.action == "block"

        # Read tool → allowed
        from flagscale_agent.react.tools.base import EFFECT_READ_FS
        ctx_read = GuardContext(
            tool_name="read_file",
            tool_effects=EFFECT_READ_FS,
            current_state=AgentState.EXECUTING,
        )
        verdict = reg.check_pre(ctx_read)
        assert verdict is None
