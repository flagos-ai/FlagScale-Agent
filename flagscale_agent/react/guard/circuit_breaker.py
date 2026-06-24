"""CircuitBreakerGuard — prevents infinite retries by tripping on repeated errors.

Uses two-phase detection via ErrorClassifierGuard's shared utilities:
1. Cheap trigger: error keywords in output
2. Precise judgment: classify_fn("is_error") to confirm real errors
"""

from __future__ import annotations

from flagscale_agent.react import display
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState


class CircuitBreakerGuard(Guard):
    """Circuit breaker: trips (blocks) when same error category repeats N times.

    States: closed (normal) → open (tripped) → half_open (probe) → closed/open.
    Activates in EXECUTING state with highest priority.

    Two-phase detection:
    1. Cheap trigger: ErrorClassifierGuard._cheap_error_trigger()
    2. LLM confirm: classify_fn("is_error") eliminates false positives
    """

    name = "circuit_breaker"
    priority = 8  # high priority, before safety(10)
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override with diagnosis — this IS the 'step back and think' the guard wants."""
        if reason and len(reason.strip()) > 20:
            # Reset the circuit that was blocking
            for cat, state in list(self._circuit_state.items()):
                if state == self.OPEN:
                    self._circuit_state[cat] = self.HALF_OPEN
                    self._open_block_count[cat] = 0
            return True
        return False

    TRIP_THRESHOLD = 4       # consecutive same-category errors → trip
    COOLDOWN_ITERS = 3       # iterations to wait before half-open probe

    # Circuit states
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, trip_threshold: int = 4, cooldown_iters: int = 3):
        self._trip_threshold = trip_threshold
        self._cooldown_iters = cooldown_iters

        # Per-category state
        self._error_counts: dict[str, int] = {}  # category → consecutive count
        self._circuit_state: dict[str, str] = {}  # category → CLOSED/OPEN/HALF_OPEN
        self._trip_iteration: dict[str, int] = {}  # category → iteration when tripped
        self._current_iteration: int = 0
        self._last_error_category: str | None = None
        self._open_block_count: dict[str, int] = {}  # category → blocks while open

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Skip pre-iteration check (no specific tool being attempted)
        if not ctx.tool_name:
            return None

        # Check if any circuit is open and would block this tool
        for category, state in self._circuit_state.items():
            if state == self.OPEN:
                trip_iter = self._trip_iteration.get(category, 0)
                elapsed = self._current_iteration - trip_iter

                if elapsed > self._cooldown_iters:
                    # Transition to half-open: allow one probe
                    self._circuit_state[category] = self.HALF_OPEN
                    self._open_block_count[category] = 0  # reset block count on state change
                    return None
                else:
                    self._open_block_count[category] = self._open_block_count.get(category, 0) + 1
                    remaining = self._cooldown_iters - elapsed + 1
                    # Escalate if agent keeps hitting the open circuit
                    if self._open_block_count.get(category, 0) >= 3:
                        return GuardVerdict.escalate(
                            f"[CircuitBreaker] '{category}' circuit OPEN, hit {self._open_block_count[category]} times. "
                            f"Current approach is fundamentally wrong — diagnose and propose a different strategy.",
                            reason=f"circuit_open_persistent_{category}",
                        )
                    return GuardVerdict.block(
                        f"[CircuitBreaker] '{category}' circuit OPEN "
                        f"({self._error_counts.get(category, 0)} consecutive failures). "
                        f"Cooldown: {remaining} iteration(s). Rethink the root cause.",
                        reason=f"circuit_open_{category}",
                    )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_result:
            return None

        category = self._classify_error(ctx.tool_result, ctx.classify_fn)

        if category is None:
            # Success: close any half-open circuits
            for cat, state in list(self._circuit_state.items()):
                if state == self.HALF_OPEN:
                    self._circuit_state[cat] = self.CLOSED
                    self._error_counts[cat] = 0
            self._last_error_category = None
            return None

        # Error detected — track it
        if category == self._last_error_category:
            self._error_counts[category] = self._error_counts.get(category, 0) + 1
        else:
            # Different category from last error — reset this category's count
            self._error_counts[category] = 1

        self._last_error_category = category

        # Half-open probe failed → re-trip
        if self._circuit_state.get(category) == self.HALF_OPEN:
            self._circuit_state[category] = self.OPEN
            self._trip_iteration[category] = self._current_iteration
            return GuardVerdict.inject(
                f"[CircuitBreaker] Probe FAILED for '{category}' — circuit re-tripped. "
                f"The underlying issue persists. Try a fundamentally different approach.",
                reason=f"circuit_retrip_{category}",
            )

        # Check if threshold reached
        if self._error_counts.get(category, 0) >= self._trip_threshold:
            self._circuit_state[category] = self.OPEN
            self._trip_iteration[category] = self._current_iteration
            return GuardVerdict.inject(
                f"[CircuitBreaker] TRIPPED for '{category}' "
                f"({self._error_counts[category]} consecutive failures). "
                f"Blocking further attempts for {self._cooldown_iters} iterations. "
                f"Try a different approach.",
                reason=f"circuit_trip_{category}",
            )

        return None

    def reset_turn(self):
        # Increment iteration counter once per iteration (not per tool call)
        self._current_iteration += 1

    def _classify_error(self, result: str, classify_fn=None) -> str | None:
        """Detect error and return a category for circuit grouping.

        Two-phase:
        1. Cheap trigger: check for error keywords
        2. LLM confirm: classify_fn("is_error") to eliminate false positives
        
        Returns tool-based category string or None if not an error.
        """
        # Phase 1: Quick keyword gate
        text_lower = result.lower()
        if not any(ind in text_lower for ind in (
            "error", "traceback", "exception", "failed", "fatal",
            "denied", "not found", "cannot", "killed", "timeout",
            "refused", "oom", "cuda",
        )):
            return None

        # Phase 2: Use LLM if available to confirm
        if classify_fn:
            is_error, source = get_judge_result(
                classify_fn, "is_error",
                {"result": result[:2000]}, default=False
            )
            if is_trusted(source) and not is_error:
                return None  # LLM says not an error

        # Use a broad category from the output keywords for circuit grouping
        return self._infer_category(text_lower)

    @staticmethod
    def _infer_category(text_lower: str) -> str:
        """Infer a coarse category from keywords — for circuit grouping only."""
        if "permission" in text_lower or "denied" in text_lower:
            return "permission"
        if "oom" in text_lower or "out of memory" in text_lower:
            return "resource"
        if "timeout" in text_lower or "refused" in text_lower:
            return "network"
        if "no module" in text_lower or "not found" in text_lower:
            return "environment"
        return "general"
