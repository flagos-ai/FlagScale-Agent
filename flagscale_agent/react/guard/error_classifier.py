"""ErrorClassifierGuard — classifies tool errors and injects recovery suggestions.

Uses two-phase detection:
1. Cheap trigger: error-like keywords in output
2. Precise judgment: classify_fn("is_error") to confirm real errors
"""

from __future__ import annotations

from flagscale_agent.react import display
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState


class ErrorClassifierGuard(Guard):
    """Classifies errors from tool results and suggests recovery actions.

    Two-phase detection:
    1. Cheap trigger: _cheap_error_trigger() scans for error keywords
    2. LLM confirm: classify_fn("is_error") eliminates false positives
    """

    name = "error_classifier"
    priority = 25  # after loop_detect(20), before progress(30)
    activate_on_states = {AgentState.EXECUTING}

    # Error categories: pattern keywords → (category_key, human description)
    ERROR_CATEGORIES = {
        "env_missing": (
            ["ModuleNotFoundError", "No module named", "command not found",
             "ImportError", "not recognized as"],
            "Environment issue: package or command missing",
        ),
        "permission": (
            ["Permission denied", "EACCES", "Operation not permitted",
             "Access is denied"],
            "Permission issue",
        ),
        "resource": (
            ["No space left", "CUDA out of memory", "OOM", "Cannot allocate",
             "out of memory", "ResourceExhaustedError"],
            "Resource exhaustion",
        ),
        "network": (
            ["Connection refused", "timeout", "Name or service not known",
             "ConnectionError", "Network is unreachable", "ETIMEDOUT"],
            "Network issue",
        ),
        "config": (
            ["KeyError", "yaml.scanner", "Invalid value", "not found in config",
             "ValidationError", "missing required"],
            "Configuration error",
        ),
    }

    # Recovery suggestion templates per category
    RECOVERY_SUGGESTIONS = {
        "env_missing": (
            "Try: 1) Install the missing package/command, "
            "2) Check if the correct environment is activated, "
            "3) Verify PATH includes the required binary."
        ),
        "permission": (
            "Try: 1) Check file/directory ownership and permissions, "
            "2) Avoid writing to read-only locations, "
            "3) Use a different output path."
        ),
        "resource": (
            "Try: 1) Free up disk/GPU memory, "
            "2) Reduce batch size or model size, "
            "3) Check resource limits."
        ),
        "network": (
            "Try: 1) Verify network connectivity, "
            "2) Check proxy settings, "
            "3) Retry after a brief wait."
        ),
        "config": (
            "Try: 1) Validate the config file syntax, "
            "2) Check for typos in key names, "
            "3) Compare with a known-good config."
        ),
    }

    _SUGGEST_THRESHOLD = 2   # consecutive same-category errors → inject suggestion
    _ESCALATE_THRESHOLD = 3  # consecutive same-category errors → strong warning

    def __init__(self):
        self._error_history: list[str] = []  # recent error categories
        self._last_category: str | None = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_result:
            return None

        # Only classify errors from shell commands — other tools (load_skill,
        # read_file, memory_*) may contain "error" in their content without
        # indicating an execution failure.
        if ctx.tool_name != "shell":
            return None

        # Phase 1: Cheap trigger — has error-like keywords?
        if not self._cheap_error_trigger(ctx.tool_result):
            self._reset_on_success()
            return None

        # Phase 2: LLM precise classification
        category = self._classify_error(ctx)

        if category is None:
            self._reset_on_success()
            return None

        self._error_history.append(category)
        self._last_category = category

        # Count consecutive same-category errors
        consecutive = 0
        for cat in reversed(self._error_history):
            if cat == category:
                consecutive += 1
            else:
                break

        if consecutive >= self._ESCALATE_THRESHOLD:
            desc = self._get_category_description(category)
            return GuardVerdict.inject(
                f"[ErrorClassifier] {desc} occurred {consecutive} times consecutively. "
                f"Previous approach is not working. STOP retrying the same method and "
                f"try a fundamentally different approach or ask the user for help.",
                reason=f"repeated_{category}_x{consecutive}",
            )

        if consecutive >= self._SUGGEST_THRESHOLD:
            desc = self._get_category_description(category)
            suggestion = self.RECOVERY_SUGGESTIONS.get(category, "Try a different approach.")
            return GuardVerdict.inject(
                f"[ErrorClassifier] {desc} detected ({consecutive}x). {suggestion}",
                reason=f"{category}_recovery_hint",
            )

        return None

    def reset_turn(self):
        # Keep error history across turns (session-level tracking)
        pass

    def _reset_on_success(self):
        """Reset error streak on successful result."""
        self._error_history.clear()
        self._last_category = None

    def _classify_error(self, ctx: GuardContext) -> str | None:
        """Classify error using LLM when available, keyword fallback otherwise."""
        print(display.dim(f"  🔍 [error_classifier] triggered: error keywords in output"))

        if ctx.classify_fn:
            is_error, source = get_judge_result(
                ctx.classify_fn, "is_error",
                {"tool_name": ctx.tool_name,
                 "command": ctx.tool_args.get("command", ""),
                 "result": ctx.tool_result[:2000] if ctx.tool_result else ""},
                default=False
            )
            if is_trusted(source) and not is_error:
                print(display.dim(f"     ✓  [error_classifier] override: not a real error"))
                return None  # LLM says not an error
            if is_trusted(source) and is_error:
                print(display.yellow(f"     ⚠  [error_classifier] confirmed: real error"))

        # Fallback to keyword classification for category
        return self._classify_static(ctx.tool_result or "")

    @staticmethod
    def _cheap_error_trigger(result: str) -> bool:
        """Phase 1: Quick check if result might contain an error."""
        indicators = ("error", "Error", "ERROR", "failed", "Failed",
                      "Traceback", "Exception", "denied", "refused",
                      "错误", "失败", "异常", "拒绝", "超时")
        return any(ind in result for ind in indicators)

    @staticmethod
    def _classify_static(result: str) -> str | None:
        """Static keyword classifier — determines error category.

        Used as fallback when LLM confirms error but we need category name.
        Also used by CircuitBreakerGuard.
        """
        for category, (patterns, _desc) in ErrorClassifierGuard.ERROR_CATEGORIES.items():
            for pattern in patterns:
                if pattern in result:
                    return category
        return None

    def _get_category_description(self, category: str) -> str:
        """Get human-readable description for a category."""
        if category in self.ERROR_CATEGORIES:
            return self.ERROR_CATEGORIES[category][1]
        return f"Unknown error ({category})"

    @property
    def error_history(self) -> list[str]:
        """Expose error history for circuit breaker integration."""
        return list(self._error_history)

    @property
    def last_category(self) -> str | None:
        """Last classified error category."""
        return self._last_category
