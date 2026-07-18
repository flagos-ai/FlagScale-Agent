# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ErrorClassifierGuard — detects tool errors and injects LLM-generated recovery suggestions.

Design: no regex patterns or hardcoded suggestions. The LLM handles both
classification and suggestion generation in one call via classify_fn.

Flow:
1. Cheap trigger: keyword scan to skip non-error outputs (avoid LLM calls)
2. LLM classify: ask judge to classify error + generate suggestion
3. Escalation: track consecutive same-category errors, warn agent to step back
"""

from __future__ import annotations

from typing import Optional

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState

# Lightweight error indicators — intentionally broad, just a gate to avoid
# calling LLM on every successful tool result
_ERROR_INDICATORS = (
    "error", "traceback", "exception", "failed", "fatal",
    "denied", "not found", "no such", "cannot", "killed",
    "timeout", "refused", "oom", "cuda",
)


class ErrorClassifierGuard(Guard):
    """Detects errors in tool output and injects recovery guidance.

    Uses LLM (via classify_fn) for both classification and suggestion.
    No hardcoded patterns or advice — fully general across domains.
    """

    name = "error_classifier"
    priority = 25
    activate_on_states = {AgentState.EXECUTING}

    # Escalation thresholds
    SUGGEST_THRESHOLD = 2   # inject "consider a different approach" hint
    ESCALATE_THRESHOLD = 3  # inject strong "stop and diagnose" warning

    def __init__(self):
        self._last_category: str | None = None
        self._consecutive_same: int = 0

    def set_shared_state(self, shared_state):
        pass

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_result:
            return None

        text = ctx.tool_result
        text_lower = text.lower()

        # Gate: skip non-error outputs
        if not self._looks_like_error(text_lower):
            self._on_success()
            return None

        # Use LLM to classify and get suggestion
        category, suggestion, llm_said_no = self._classify_with_llm(ctx, text)

        if not category:
            if llm_said_no:
                # LLM actively confirmed this is NOT an error — reset streak
                self._on_success()
            # Otherwise (no classify_fn or untrusted source): don't reset, don't fire
            return None

        # Track consecutive same-category errors for escalation
        if category == self._last_category:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 1
        self._last_category = category

        # Build message with escalation prefix if needed
        msg = self._build_message(category, suggestion)

        return GuardVerdict.inject(
            msg,
            reason=f"error_classified_{category}",
            category=f"error_{category}",
        )

    def _classify_with_llm(
        self, ctx: GuardContext, text: str
    ) -> tuple[Optional[str], Optional[str], bool]:
        """Ask LLM to determine if this is a real error.

        Returns (category, suggestion, llm_said_no) where:
        - category is set if LLM confirms an error
        - llm_said_no is True only when LLM actively says "not an error"
        """
        if not ctx.classify_fn:
            return None, None, False

        # Truncate to avoid blowing up the classify prompt
        snippet = text[:1500] if len(text) > 1500 else text

        result, source = get_judge_result(
            ctx.classify_fn,
            "is_error",
            {"output": snippet, "tool": ctx.tool_name},
            default="no",
        )

        if not is_trusted(source):
            return None, None, False

        # is_error returns yes/no style — if "no", skip
        result_str = result if isinstance(result, str) else str(result)
        if result_str.lower().startswith("n"):
            return None, None, True  # LLM actively said "not an error"

        # Use tool_name as the category for escalation tracking
        category = f"{ctx.tool_name}_error"
        return category, None, False

    def _build_message(self, category: str, suggestion: Optional[str]) -> str:
        """Build the injection message with escalation if needed."""
        parts = []

        if self._consecutive_same >= self.ESCALATE_THRESHOLD:
            parts.append(
                f"⚠️ Same error type '{category}' hit {self._consecutive_same} times. "
                f"Stop and diagnose the root cause — don't retry the same approach."
            )
        elif self._consecutive_same >= self.SUGGEST_THRESHOLD:
            parts.append(
                f"Error '{category}' repeated {self._consecutive_same} times. "
                f"Consider a different approach."
            )

        if suggestion:
            parts.append(suggestion)

        return "\n\n".join(parts) if parts else f"[Error detected: {category}]"

    def _on_success(self):
        """Reset consecutive counter on non-error output."""
        if self._last_category:
            self._consecutive_same = 0
            self._last_category = None

    @staticmethod
    def _looks_like_error(text_lower: str) -> bool:
        """Quick keyword gate — does this look like an error?"""
        return any(ind in text_lower for ind in _ERROR_INDICATORS)

    def reset_turn(self):
        pass
