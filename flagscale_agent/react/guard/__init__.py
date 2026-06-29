"""Guard system — behavioral constraints with lifecycle hooks.

Guards fire at three points:
- pre: Before tool execution (can block)
- post: After tool execution (can inject messages)
- strategic: At review points (can redirect plan)

v2: Added SharedState for cross-guard communication and inject deduplication.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from flagscale_agent.react import display
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

    # LLM response text (for guards that need to scan assistant replies)
    assistant_text: str = ""

    # State machine context
    current_state: AgentState = AgentState.IDLE
    transitions_count: int = 0

    # LLM classify function
    classify_fn: Any = None  # (category: str, context: dict) -> Any

    # Experiment context
    experiment_compare_fn: Any = None
    experiment_diff_fn: Any = None
    current_experiment_name: str = ""

    # Override reason: LLM declares why a potentially-blocked call is justified
    override_reason: str = ""

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
    # v2: category tag for deduplication
    category: str = ""  # e.g. "read_stall", "loop", "plan_needed"

    @classmethod
    def block(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="block", message=message, reason=reason, category=category)

    @classmethod
    def inject(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="inject_msg", message=message, reason=reason, category=category)

    @classmethod
    def compact(cls, reason: str = "") -> GuardVerdict:
        return cls(action="force_compact", reason=reason)

    @classmethod
    def escalate(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason, category=category)

    @classmethod
    def redirect(cls, message: str, reason: str = "", metadata: dict = None) -> GuardVerdict:
        return cls(action="redirect", message=message, reason=reason, metadata=metadata or {})


class Guard(abc.ABC):
    """Base class for all guards.

    Lifecycle:
    - Guards accumulate state over time (across iterations/turns).
    - Guards that block/inject must define when they are SATISFIED (concern resolved).
    - Guards must support DECAY: if N iterations pass without re-triggering, state resets.
    - All guards are overridable by default (LLM can bypass with a reason).
    - Inject messages have a MAX_INJECT_REPEATS cooldown to avoid context pollution.
    """

    name: str = "unnamed"
    priority: int = 50  # lower = higher priority
    activate_on_states: set[AgentState] = set()
    activate_on_tools: set[str] | None = None  # None = all tools

    # Override mechanism: if True, LLM can bypass this guard's block by providing
    # a reason in tool_args["_override_reason"]. The guard's accept_override()
    # method decides whether the reason is sufficient.
    # v3: Default changed to True — all guards should be overridable to prevent
    # death spirals. Guards can set to False only for safety-critical blocks.
    overridable: bool = True

    # v3: Inject cooldown — max times the same inject message category fires
    # before going silent. Prevents context pollution from repetitive warnings.
    max_inject_repeats: int = 3

    # v3: Decay window — if this many iterations pass without the guard
    # re-triggering (returning a non-None verdict), persistent state is reset.
    # Set to 0 to disable decay (state persists indefinitely).
    decay_after_idle: int = 10

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # v3: Patch subclass __init__ to ensure lifecycle attrs exist even if
        # subclass doesn't call super().__init__()
        original_init = cls.__dict__.get('__init__')
        if original_init:
            def patched_init(self, *args, _original=original_init, **kw):
                # Ensure lifecycle attrs exist BEFORE subclass init
                if not hasattr(self, '_inject_counts'):
                    self._inject_counts = {}
                    self._iterations_since_trigger = 0
                    self._is_suppressed = False
                _original(self, *args, **kw)
                # Also ensure they exist AFTER (in case subclass clobbers)
                if not hasattr(self, '_inject_counts'):
                    self._inject_counts = {}
                    self._iterations_since_trigger = 0
                    self._is_suppressed = False
            cls.__init__ = patched_init

    def __init__(self):
        # v3: Lifecycle tracking (managed by GuardRegistry)
        self._inject_counts: dict[str, int] = {}  # category -> times fired
        self._iterations_since_trigger: int = 0
        self._is_suppressed: bool = False  # True = guard went silent after cooldown

    def should_activate(self, ctx: GuardContext) -> bool:
        """Check if this guard should run for the current context."""
        # v3: Suppressed guards skip entirely until reset
        if self._is_suppressed:
            return False
        # Empty activate_on_states means "all states" (no filter)
        if self.activate_on_states and ctx.current_state not in self.activate_on_states:
            return False
        if self.activate_on_tools and ctx.tool_name not in self.activate_on_tools:
            return False
        return True

    def is_satisfied(self, ctx: GuardContext) -> bool:
        """Return True if the guard's concern has been addressed.

        When satisfied, the guard resets its persistent state and stops firing.
        Subclasses SHOULD override this to define their satisfaction condition.
        Default: never satisfied (backward compat).
        """
        return False

    def reset_state(self):
        """Reset all persistent state to initial values.

        Called when:
        - is_satisfied() returns True
        - Decay window expires (no re-triggers for N iterations)
        - LLM successfully overrides the guard

        Subclasses MUST override this to reset their specific state.
        """
        self._inject_counts.clear()
        self._iterations_since_trigger = 0
        self._is_suppressed = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Pre-execution check. Return block/inject to prevent or warn."""
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Post-execution check. Return inject to add context."""
        return None

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """Strategic review check. Return redirect to change plan."""
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Evaluate whether the LLM's override reason is sufficient to bypass a block.

        Only called when overridable=True and the guard returned a block verdict.
        Default: accept any non-empty reason. Override for stricter validation.

        v3: When override is accepted, guard state is reset to prevent re-blocking
        on the same concern.
        """
        accepted = bool(reason and reason.strip())
        if accepted:
            self.reset_state()
        return accepted

    def notify_blocked(self, ctx: GuardContext):
        """Called when a tool call was blocked externally (e.g., by another guard)."""
        pass

    def reset_iteration(self):
        """Called at the start of each iteration (LLM+tool loop) within a turn.

        A "turn" is one user message → completion (may contain many iterations).
        An "iteration" is one LLM call + one tool execution within that turn.

        Most guards should NOT reset state here — they need to track patterns
        across iterations (e.g., consecutive errors, read streaks).
        Only reset per-iteration dedup caches or similar ephemeral state.
        """
        pass

    # Backward compat: subclasses may override either name
    reset_turn = reset_iteration

    def set_shared_state(self, shared_state):
        """Optional: receive SharedState from GuardRegistry. Override to use."""
        pass

    # --- v3: Lifecycle helpers (called by GuardRegistry) ---

    def _record_trigger(self, category: str = ""):
        """Record that this guard fired (returned non-None verdict)."""
        self._iterations_since_trigger = 0
        if category:
            self._inject_counts[category] = self._inject_counts.get(category, 0) + 1

    def _should_suppress_inject(self, category: str = "") -> bool:
        """Check if this inject category has exceeded its repeat limit."""
        if not category or self.max_inject_repeats <= 0:
            return False
        count = self._inject_counts.get(category, 0)
        return count >= self.max_inject_repeats

    def _tick_idle(self):
        """Called each iteration when guard did NOT fire. Manages decay."""
        self._iterations_since_trigger += 1
        if self.decay_after_idle > 0 and self._iterations_since_trigger >= self.decay_after_idle:
            self.reset_state()

    def dismiss_inject(self, category: str = ""):
        """Dismiss a specific inject category — it will no longer fire.

        Called when the LLM explicitly acknowledges/dismisses a guard's inject.
        The guard's inject for this category is permanently suppressed until
        reset_state() is called (via decay or override).
        """
        if category:
            # Set count to max+1 to ensure suppression
            self._inject_counts[category] = self.max_inject_repeats + 100
        else:
            # Dismiss ALL categories for this guard
            self._is_suppressed = True

    def check_dismiss_from_text(self, assistant_text: str) -> bool:
        """Check if LLM's response dismisses this guard's injects.

        Override in subclasses for guard-specific dismiss patterns.
        Base implementation checks for generic dismiss signals referencing
        this guard's name.

        Returns True if dismissed.
        """
        if not assistant_text:
            return False
        text_lower = assistant_text.lower()
        guard_name_lower = self.name.lower().replace("_", " ")

        # Generic dismiss patterns
        dismiss_signals = [
            f"dismiss {guard_name_lower}",
            f"ignore {guard_name_lower}",
            f"suppress {guard_name_lower}",
            f"stop {guard_name_lower}",
            f"{guard_name_lower} not needed",
            f"no need for {guard_name_lower}",
        ]
        if any(sig in text_lower for sig in dismiss_signals):
            self.dismiss_inject()
            return True
        return False


# Semantic categories for inject deduplication
# Injects with the same category in the same turn are merged, not duplicated.
_INJECT_CATEGORY_PATTERNS = {
    "read_stall": re.compile(r"read.only|re.reading|gathering information|not acting", re.IGNORECASE),
    "loop": re.compile(r"loop|repeated|same tool|same call", re.IGNORECASE),
    "plan_needed": re.compile(r"plan|plan_create|organize", re.IGNORECASE),
    "budget": re.compile(r"budget|token|exhausted", re.IGNORECASE),
}

# Category suppression: if a higher-priority category fires, suppress lower-priority ones.
# Key = category that suppresses, Value = set of categories it suppresses.
_CATEGORY_SUPPRESSION: dict[str, set[str]] = {
    "comprehension": {"memory_write_reminder", "memory_read_reminder"},
    "read_stall": {"memory_write_reminder"},
    "loop": {"read_stall", "memory_write_reminder"},
}


def _infer_category(verdict: GuardVerdict) -> str:
    """Infer the semantic category of an inject verdict for deduplication."""
    if verdict.category:
        return verdict.category
    # Try to infer from message content
    text = verdict.message + " " + verdict.reason
    for cat, pattern in _INJECT_CATEGORY_PATTERNS.items():
        if pattern.search(text):
            return cat
    return ""


_OVERRIDE_HINT = (
    "\n\n[To override: re-issue the same tool call with an added "
    "\"_override_reason\" field in tool_args explaining why this action is justified.]"
)


def _maybe_add_override_hint(
    verdict: GuardVerdict, blocking_guard: Guard | None, ctx: GuardContext
) -> str:
    """Append override instructions to a block message if the blocking guard is overridable.

    Only appends when:
    - The verdict is a "block"
    - The blocking guard has overridable=True
    - The LLM hasn't already provided an override_reason (avoids re-hinting on rejection)
    """
    if verdict.action != "block":
        return verdict.message
    if ctx.override_reason:
        # Override was attempted but rejected — don't re-hint
        return verdict.message
    if blocking_guard and blocking_guard.overridable:
        return verdict.message + _OVERRIDE_HINT
    return verdict.message


class GuardRegistry:
    """Manages all guards, runs them in priority order, deduplicates injects."""

    def __init__(self):
        self._guards: list[Guard] = []
        # v2: SharedState for cross-guard communication
        from flagscale_agent.react.guard.shared_state import SharedState
        self._shared_state = SharedState()

    def register(self, guard: Guard):
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)
        # Inject shared state into guards that support it
        guard.set_shared_state(self._shared_state)

    @property
    def shared_state(self):
        """Access the shared state for external use (e.g., agent setting TaskMode)."""
        return self._shared_state

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks with inject deduplication."""
        inject_messages = []
        inject_categories_seen: set = set()
        first_hard_verdict = None
        first_hard_guard = None
        first_reason = ""
        fired_guards: set[str] = set()

        # v3: Scan assistant text for dismiss signals BEFORE running guards
        assistant_text = getattr(ctx, 'assistant_text', "") or ""
        if assistant_text:
            for guard in self._guards:
                if guard._is_suppressed:
                    continue
                guard.check_dismiss_from_text(assistant_text)

        for guard in self._guards:
            if not guard.should_activate(ctx):
                continue

            # v3: Skip if guard is fully suppressed (dismissed)
            if guard._is_suppressed:
                continue

            # v3: Check satisfaction before running the guard
            # Note: do NOT call reset_state() here — that would clear the
            # satisfied condition and cause the guard to re-fire next iteration.
            # A satisfied guard simply stays silent until decay resets it.
            if hasattr(guard, 'is_satisfied') and guard.is_satisfied(ctx):
                continue

            verdict = guard.check_pre(ctx)
            if verdict is None:
                continue

            # v3: Track which guards fired
            fired_guards.add(guard.name)

            if verdict.action in ("block", "escalate", "force_compact", "redirect"):
                # Override mechanism: if guard is overridable and LLM provided a reason,
                # let the guard decide whether to accept
                if (
                    verdict.action == "block"
                    and guard.overridable
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    # Override accepted — skip this block, log it
                    self._shared_state.record_override(guard.name, ctx.override_reason)
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                if first_hard_verdict is None:
                    first_hard_verdict = verdict
                    first_hard_guard = guard
                continue

            if verdict.action == "inject_msg":
                # v2: Deduplicate by semantic category
                category = _infer_category(verdict)
                if category and category in inject_categories_seen:
                    # Skip duplicate — a similar warning was already queued
                    continue
                if category:
                    inject_categories_seen.add(category)
                    # v2.1: Suppress lower-priority categories
                    suppressed = _CATEGORY_SUPPRESSION.get(category)
                    if suppressed:
                        inject_categories_seen.update(suppressed)

                # v2: Check effectiveness — if this inject has been repeatedly
                # ineffective, escalate instead of repeating
                if category and self._shared_state.inject_tracker.should_suppress(
                    guard.name, category
                ):
                    escalation_msg = self._shared_state.inject_tracker.get_escalation_message(
                        guard.name, category
                    )
                    # Replace the inject with an escalation
                    if first_hard_verdict is None:
                        first_hard_verdict = GuardVerdict.escalate(
                            escalation_msg, reason=f"ineffective_inject_{guard.name}"
                        )
                    continue

                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

                # Track in SharedState
                self._shared_state.inject_tracker.record_inject(
                    guard.name, category or verdict.reason, ctx.turn_count
                )

        # If there's a hard verdict, prepend inject messages and add override hint
        if first_hard_verdict:
            # v3: Still tick lifecycle for idle guards
            self.tick_guard_lifecycle(fired_guards)
            if inject_messages and first_hard_verdict.message:
                first_hard_verdict.message = "\n\n".join(inject_messages) + "\n\n" + first_hard_verdict.message
            # Add override hint if the blocking guard is overridable
            first_hard_verdict.message = _maybe_add_override_hint(
                first_hard_verdict, first_hard_guard, ctx
            )
            return first_hard_verdict

        # Merge all inject messages into one verdict (deduplicated)
        if inject_messages:
            # v3: Tick lifecycle for idle guards
            self.tick_guard_lifecycle(fired_guards)
            combined = "\n\n".join(inject_messages)
            # v3: Add dismiss hint so LLM knows it can stop the inject
            # Only add hint if any guard has fired more than once (repeated inject)
            any_repeated = any(
                sum(g._inject_counts.values()) > 1
                for g in self._guards
                if g.name in fired_guards
            )
            if any_repeated:
                combined += (
                    "\n\n[To dismiss these reminders: include "
                    "\"_dismiss_guard\": \"<guard_name>\" in your next "
                    "tool_args, or simply proceed — they will stop after "
                    "2 repeats automatically.]"
                )
            return GuardVerdict.inject(
                combined,
                reason=first_reason or "multi_guard_inject"
            )

        # v3: Tick lifecycle for idle guards (no verdict case)
        self.tick_guard_lifecycle(fired_guards)
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks with inject deduplication."""
        inject_messages = []
        inject_categories_seen: set = set()
        first_hard_verdict = None
        first_hard_guard = None
        first_reason = ""

        # v2: Update shared state with tool call info
        self._shared_state.record_tool_call(
            ctx.tool_name, ctx.tool_args,
            is_read_only=ctx.tool_effects.is_read_only
        )

        for guard in self._guards:
            if not guard.should_activate(ctx):
                continue
            verdict = guard.check_post(ctx)
            if verdict is None:
                continue

            if verdict.action in ("block", "escalate", "force_compact", "redirect"):
                # Override mechanism (same as check_pre)
                if (
                    verdict.action == "block"
                    and guard.overridable
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    self._shared_state.record_override(guard.name, ctx.override_reason)
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                if first_hard_verdict is None:
                    first_hard_verdict = verdict
                    first_hard_guard = guard
                continue

            if verdict.action == "inject_msg":
                # v2: Deduplicate by semantic category
                category = _infer_category(verdict)
                if category and category in inject_categories_seen:
                    continue
                if category:
                    inject_categories_seen.add(category)
                    # v2.1: Suppress lower-priority categories
                    suppressed = _CATEGORY_SUPPRESSION.get(category)
                    if suppressed:
                        inject_categories_seen.update(suppressed)

                # v2: Check effectiveness — suppress repeatedly ineffective injects
                if category and self._shared_state.inject_tracker.should_suppress(
                    guard.name, category
                ):
                    escalation_msg = self._shared_state.inject_tracker.get_escalation_message(
                        guard.name, category
                    )
                    if first_hard_verdict is None:
                        first_hard_verdict = GuardVerdict.escalate(
                            escalation_msg, reason=f"ineffective_inject_{guard.name}"
                        )
                    continue

                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

                self._shared_state.inject_tracker.record_inject(
                    guard.name, category or verdict.reason, ctx.turn_count
                )

        if first_hard_verdict:
            if inject_messages and first_hard_verdict.message:
                first_hard_verdict.message = "\n\n".join(inject_messages) + "\n\n" + first_hard_verdict.message
            first_hard_verdict.message = _maybe_add_override_hint(
                first_hard_verdict, first_hard_guard, ctx
            )
            return first_hard_verdict

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

    def reset_iteration(self):
        """Reset per-iteration state for all guards.

        Called at the start of each iteration (LLM+tool loop) within a turn.
        v3: Satisfaction and decay are handled in tick_guard_lifecycle(), not here.
        """
        self._shared_state.new_iteration()
        for guard in self._guards:
            # Call reset_turn — subclasses override this name
            guard.reset_turn()

    # Backward compat alias
    reset_turn = reset_iteration

    def tick_guard_lifecycle(self, fired_guards: set[str]):
        """v3: Called after each check cycle to manage decay for idle guards.

        Args:
            fired_guards: set of guard names that returned non-None verdicts this cycle
        """
        for guard in self._guards:
            if guard.name in fired_guards:
                guard._record_trigger()
            else:
                guard._tick_idle()

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)
