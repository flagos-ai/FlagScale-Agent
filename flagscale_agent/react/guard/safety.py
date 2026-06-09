"""SafetyGuard — dangerous command detection, error escalation.

Uses LLM classify() for all judgments — no regex/keyword matching.

When Judge is unavailable (provider None or budget exhausted), takes
conservative action: block unverified shell commands, don't reset error
counters on uncertain success checks.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import (
    get_judge_result as _get_judge_result,
    SOURCE_LLM as _SOURCE_LLM,
    SOURCE_CACHE as _SOURCE_CACHE,
    SOURCE_DEFAULT as _SOURCE_DEFAULT,
    SOURCE_UNAVAILABLE as _SOURCE_UNAVAILABLE,
)
from flagscale_agent.react.state_machine import AgentState


class SafetyGuard(Guard):
    """Detects dangerous commands and escalating error patterns.

    Checked first (priority=10). Uses tool_effects to scope checks.
    """

    name = "safety"
    priority = 10
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING, AgentState.REVIEWING}

    # Escalation thresholds
    _ERROR_ESCALATE_WARN = 3
    _ERROR_ESCALATE_HARD = 5

    def __init__(self):
        self._consecutive_errors: int = 0
        self._root_cause_recorded_since_error: bool = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only check shell commands for danger
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        classify = ctx.classify_fn
        if not classify:
            return GuardVerdict.block(
                "[Safety] Safety classifier unavailable — blocking shell command. "
                "Re-run with a working LLM provider, or use /mode confirm to manually approve.",
                reason="classify_fn not available for safety pre-check",
            )

        is_dangerous, source = _get_judge_result(
            classify, "is_dangerous", {"command": cmd}, default=False,
        )

        if source in (_SOURCE_DEFAULT, _SOURCE_UNAVAILABLE):
            return GuardVerdict.block(
                "[Safety] Safety judge unavailable — blocking shell command. "
                f"Judge returned default value (source={source}). "
                "Re-run with a working LLM provider.",
                reason=f"safety classifier unavailable (source={source})",
            )

        if is_dangerous:
            return GuardVerdict.block(
                "[Safety] Dangerous command detected and blocked. "
                "If this is intentional, explain why and use a "
                "more targeted approach.",
                reason="dangerous command blocked by LLM judge",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        result = ctx.tool_result or ""
        classify = ctx.classify_fn

        # Use LLM to determine if this is a real error
        is_error = False
        error_source = ""
        if classify and ctx.tool_name in ("shell", "write_file", "edit_file"):
            is_error, error_source = _get_judge_result(
                classify, "is_error", {
                    "tool_name": ctx.tool_name,
                    "command": ctx.tool_args.get("command", ""),
                    "result": result,
                }, default=False,
            )

        error_trustworthy = error_source in (_SOURCE_LLM, _SOURCE_CACHE)

        # Track memory_write as root-cause documentation (regardless of error status)
        if ctx.tool_name == "memory_write" and self._consecutive_errors > 0:
            self._root_cause_recorded_since_error = True

        if is_error:
            self._consecutive_errors += 1

            if self._consecutive_errors >= self._ERROR_ESCALATE_HARD:
                return GuardVerdict.escalate(
                    f"[Safety] {self._consecutive_errors} consecutive tool errors. "
                    "The current approach is not working. Stop, diagnose the root "
                    "cause, and reformulate your strategy before continuing.",
                    reason=f"hard escalation: {self._consecutive_errors} errors",
                )

            if self._consecutive_errors >= self._ERROR_ESCALATE_WARN:
                if not self._root_cause_recorded_since_error:
                    return GuardVerdict.inject(
                        f"[Safety] {self._consecutive_errors} consecutive tool errors "
                        "without recording root cause. Use memory_write to document "
                        "what's failing and why before retrying.",
                        reason="error escalation warn: no root cause recorded",
                    )
        else:
            if error_trustworthy:
                if self._consecutive_errors > 0:
                    self._consecutive_errors = 0
                self._root_cause_recorded_since_error = False

        # Track recovery via LLM success check
        if ctx.tool_name == "shell" and classify:
            is_success, success_source = _get_judge_result(
                classify, "is_success", {
                    "command": ctx.tool_args.get("command", ""),
                    "result": result,
                }, default=False,
            )
            if is_success and success_source in (_SOURCE_LLM, _SOURCE_CACHE):
                self._consecutive_errors = 0

        return None

    def reset_turn(self):
        pass  # Safety/error state persists across turns
