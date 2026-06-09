"""Guard system — behavioral constraints with lifecycle hooks.

Guards fire at three points:
- pre: Before tool execution (can block)
- post: After tool execution (can inject messages)
- strategic: At review points (can redirect plan)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Any

from flagscale_agent.react.state_machine import AgentState
from flagscale_agent.react.tools.base import ToolEffect


@dataclass
class GuardContext:
    """Read-only snapshot passed to guards.

    Contains tool context, state machine info, and LLM classify function.
    """

    # Tool context
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None
    tool_effects: ToolEffect = field(default_factory=ToolEffect)
    turn_count: int = 0
    recent_tool_names: list[str] = field(default_factory=list)
    recent_tool_history: list[dict] = field(default_factory=list)  # [{tool, args_summary, result_summary}]
    context_pressure: float = 0.0

    # State machine context
    current_state: AgentState = AgentState.IDLE
    transitions_count: int = 0

    # LLM classify function
    classify_fn: Any = None  # (category: str, context: dict) -> Any

    # Experiment context
    experiment_compare_fn: Any = None
    experiment_diff_fn: Any = None
    current_experiment_name: str = ""

    @property
    def phase_name(self) -> str:
        """Derive phase name from current state for backward compatibility."""
        return self.current_state.name.lower()


@dataclass
class GuardVerdict:
    """What the guard wants the agent to do."""

    action: Literal["allow", "block", "inject_msg", "force_compact", "escalate", "redirect"]
    message: str = ""
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def allow(cls) -> GuardVerdict:
        return cls(action="allow")

    @classmethod
    def block(cls, message: str, reason: str = "") -> GuardVerdict:
        return cls(action="block", message=message, reason=reason)

    @classmethod
    def inject(cls, message: str, reason: str = "") -> GuardVerdict:
        return cls(action="inject_msg", message=message, reason=reason)

    @classmethod
    def compact(cls, reason: str = "") -> GuardVerdict:
        return cls(action="force_compact", reason=reason)

    @classmethod
    def escalate(cls, message: str, reason: str = "") -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason)

    @classmethod
    def redirect(cls, message: str, reason: str = "", metadata: dict | None = None) -> GuardVerdict:
        return cls(action="redirect", message=message, reason=reason, metadata=metadata or {})


class Guard(abc.ABC):
    """A behavioral constraint with lifecycle hooks.

    Each Guard OWNS its state — no agent._xxx scatter.
    """

    # Subclass must override
    name: str = "base"
    priority: int = 50  # Lower = earlier in check order

    # Activation conditions
    activate_on_states: set[AgentState] = {AgentState.EXECUTING}
    activate_on_tools: set[str] | None = None  # None = all tools

    def should_activate(self, ctx: GuardContext) -> bool:
        """Check if this guard should fire for the given context."""
        if ctx.current_state not in self.activate_on_states:
            return False
        if self.activate_on_tools is not None and ctx.tool_name not in self.activate_on_tools:
            return False
        return True

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Called BEFORE tool execution. Return verdict to act."""
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Called AFTER tool execution. Return verdict to act."""
        return None

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """Called at strategic review points (every N turns). Return verdict to redirect."""
        return None

    def reset_turn(self):
        """Called at the start of each turn. Override to reset per-turn state."""

    def notify_blocked(self, ctx: GuardContext):
        """Called when a tool call was blocked by another guard AFTER this guard's check_pre passed.

        Override to undo any state changes made in check_pre (e.g., remove from history).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"


class GuardRegistry:
    """Manages guard instances, sorted by priority."""

    def __init__(self):
        self._guards: list[Guard] = []

    def register(self, guard: Guard):
        """Register a guard and maintain priority order."""
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks. First non-None verdict wins.

        If a guard blocks/escalates, notify all earlier guards that passed
        so they can undo state changes (e.g., remove from history).
        """
        passed_guards: list[Guard] = []
        for guard in self._guards:
            if guard.should_activate(ctx):
                verdict = guard.check_pre(ctx)
                if verdict is not None:
                    # This guard blocked — notify all earlier guards that passed
                    for earlier in passed_guards:
                        earlier.notify_blocked(ctx)
                    return verdict
                passed_guards.append(guard)
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks.

        Unlike check_pre, ALL guards run (to update internal state).
        Collects all inject_msg verdicts and merges them.
        For block/escalate/force_compact, returns the first (highest-priority) one.
        """
        inject_messages: list[str] = []
        first_hard_verdict: GuardVerdict | None = None
        first_reason: str | None = None

        for guard in self._guards:
            if guard.should_activate(ctx):
                verdict = guard.check_post(ctx)
                if verdict is None:
                    continue
                if verdict.action == "inject_msg":
                    if verdict.message:
                        inject_messages.append(verdict.message)
                    if first_reason is None:
                        first_reason = verdict.reason
                elif first_hard_verdict is None:
                    first_hard_verdict = verdict

        # Hard verdicts (block/escalate/force_compact) take priority
        if first_hard_verdict is not None:
            # Prepend any inject messages to the hard verdict
            if inject_messages and first_hard_verdict.message:
                first_hard_verdict.message = "\n\n".join(inject_messages) + "\n\n" + first_hard_verdict.message
            return first_hard_verdict

        # Merge all inject messages into one verdict
        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject"
            )

        return None

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' strategic checks."""
        for guard in self._guards:
            if guard.should_activate(ctx):
                verdict = guard.check_strategic(ctx)
                if verdict is not None:
                    return verdict
        return None

    def reset_turn(self):
        """Reset all guards for a new turn."""
        for guard in self._guards:
            guard.reset_turn()

    def notify_all_blocked(self, ctx: GuardContext):
        """Notify all guards that a tool call was blocked externally (e.g., by user deny).

        Called by tool executor when a call is blocked after all guards passed check_pre.
        """
        for guard in self._guards:
            if guard.should_activate(ctx):
                guard.notify_blocked(ctx)

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)
