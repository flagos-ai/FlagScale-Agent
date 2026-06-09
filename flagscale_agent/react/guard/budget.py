"""BudgetGuard — tracks token and tool-call budgets for the session."""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class BudgetGuard(Guard):
    """Tracks session-level token consumption and tool call counts.

    Warns at 80%, strongly warns at 95%, blocks at 100%.
    Activates in EXECUTING state with very high priority.
    """

    name = "budget"
    priority = 5  # very high priority
    activate_on_states = {AgentState.EXECUTING}

    def __init__(self, max_tokens: int = 2_000_000, max_tool_calls: int = 500):
        self._max_tokens = max_tokens
        self._max_tool_calls = max_tool_calls
        self._total_tokens: int = 0
        self._total_tool_calls: int = 0
        self._warned_token_80: bool = False
        self._warned_token_95: bool = False
        self._warned_tool_80: bool = False
        self._warned_tool_95: bool = False
        self._exhausted_block_count: int = 0  # track repeated blocks

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Check token budget
        token_pct = self._token_percent
        if token_pct >= 100:
            self._exhausted_block_count += 1
            if self._exhausted_block_count >= 3:
                return GuardVerdict.escalate(
                    "[Budget] Token budget exhausted and agent is not stopping. "
                    "Summarize your progress and STOP immediately. "
                    f"Used: {self._total_tokens:,} / {self._max_tokens:,} tokens.",
                    reason="budget_tokens_exhausted_persistent",
                )
            return GuardVerdict.block(
                "[Budget] Token budget exhausted. "
                "Summarize your progress and stop. "
                f"Used: {self._total_tokens:,} / {self._max_tokens:,} tokens.",
                reason="budget_tokens_exhausted",
            )

        # Check tool call budget
        tool_pct = self._tool_call_percent
        if tool_pct >= 100:
            self._exhausted_block_count += 1
            if self._exhausted_block_count >= 3:
                return GuardVerdict.escalate(
                    "[Budget] Tool call budget exhausted and agent is not stopping. "
                    "Summarize your progress and STOP immediately. "
                    f"Used: {self._total_tool_calls} / {self._max_tool_calls} calls.",
                    reason="budget_tool_calls_exhausted_persistent",
                )
            return GuardVerdict.block(
                "[Budget] Tool call budget exhausted. "
                "Summarize your progress and stop. "
                f"Used: {self._total_tool_calls} / {self._max_tool_calls} calls.",
                reason="budget_tool_calls_exhausted",
            )

        # Token warnings (inject once per threshold)
        if token_pct >= 95 and not self._warned_token_95:
            self._warned_token_95 = True
            return GuardVerdict.inject(
                f"[Budget] WARNING: Token budget nearly exhausted ({token_pct:.0f}%). "
                f"Wrap up NOW — complete the current step and summarize progress.",
                reason="budget_token_95_warning",
            )

        if token_pct >= 80 and not self._warned_token_80:
            self._warned_token_80 = True
            return GuardVerdict.inject(
                f"[Budget] Token budget at {token_pct:.0f}%. Prioritize completion.",
                reason="budget_token_80_warning",
            )

        # Tool call warnings (independent from token warnings)
        if tool_pct >= 95 and not self._warned_tool_95:
            self._warned_tool_95 = True
            return GuardVerdict.inject(
                f"[Budget] WARNING: Tool call budget nearly exhausted "
                f"({self._total_tool_calls}/{self._max_tool_calls}). Wrap up NOW.",
                reason="budget_tool_95_warning",
            )

        if tool_pct >= 80 and not self._warned_tool_80:
            self._warned_tool_80 = True
            return GuardVerdict.inject(
                f"[Budget] Tool call budget at {tool_pct:.0f}%. Prioritize completion.",
                reason="budget_tool_80_warning",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Count tool calls
        if ctx.tool_name:
            self._total_tool_calls += 1
        return None

    def report_tokens(self, input_tokens: int, output_tokens: int):
        """Called externally (by agent/kernel) to report token usage after each LLM call."""
        self._total_tokens += input_tokens + output_tokens

    def reset_turn(self):
        # Budget is session-level, never resets per turn
        pass

    @property
    def _token_percent(self) -> float:
        if self._max_tokens <= 0:
            return 0.0
        return (self._total_tokens / self._max_tokens) * 100

    @property
    def _tool_call_percent(self) -> float:
        if self._max_tool_calls <= 0:
            return 0.0
        return (self._total_tool_calls / self._max_tool_calls) * 100

    @property
    def usage_summary(self) -> dict:
        """Return current budget usage summary."""
        return {
            "total_tokens": self._total_tokens,
            "max_tokens": self._max_tokens,
            "token_percent": round(self._token_percent, 1),
            "total_tool_calls": self._total_tool_calls,
            "max_tool_calls": self._max_tool_calls,
            "tool_call_percent": round(self._tool_call_percent, 1),
        }
