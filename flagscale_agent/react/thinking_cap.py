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

"""Runtime thinking-length cap.

GLM's Anthropic-compat endpoint ignores the request-side ``thinking_budget``
(served thinking runs to ``max_output_tokens``), so a single monolithic
thinking block can burn ~20 minutes producing zero visible output. The fix is
client-side: count thinking tokens as the stream is consumed, and abort the
stream once the accumulated reasoning reaches ``config.thinking_budget``.

The kernel treats an aborted ("capped") reasoning-only response like any other
reasoning-only response — it re-injects the full thinking text as a user
message — but with a CONVERGENCE nudge instead of a RESUME nudge: the model
must land its next concrete action, not keep reasoning.

Cap semantics:
    thinking_budget <= 0  → cap disabled (identical to pre-cap behavior)
    thinking_budget > 0   → abort when estimated thinking tokens >= budget

Token estimation mirrors history._estimate_tokens exactly (CJK-aware) so the
cap is measured on the same scale as the rest of the context accounting.
"""

from __future__ import annotations

from typing import Optional, Tuple


def estimate_thinking_tokens(text: str) -> int:
    """Estimate token count of a thinking snippet.

    Mirrors flagscale_agent.react.history._estimate_tokens:
    ~1.5 tokens per CJK char, ~0.25 tokens per ASCII char (4 chars/token).
    Kept as a local mirror instead of importing so the cap module stays
    dependency-free and unit-testable in isolation.
    """
    if not text:
        return 0
    cjk_count = sum(
        1 for c in text
        if '\u4e00' <= c <= '\u9fff'
        or '\u3040' <= c <= '\u30ff'
        or '\uac00' <= c <= '\ud7af'
    )
    ascii_count = len(text) - cjk_count
    return int(cjk_count * 1.5) + (ascii_count // 4)


def thinking_capped_state(
    accumulated_chars: int,
    cjk_chars: int,
    cap_budget: int,
) -> bool:
    """Pure threshold predicate: has thinking reached the cap?

    Args:
        accumulated_chars: total chars of thinking accumulated so far.
        cjk_chars: CJK chars inside those accumulated chars.
        cap_budget: configured thinking_budget (0 or negative = disabled).

    Returns:
        True iff cap is active (cap_budget > 0) and the estimated token
        count of the accumulated thinking is >= cap_budget.

    Mirrors the incremental arithmetic used by the agent-side counter:
        tokens = int(cjk * 1.5) + (ascii // 4)
    """
    if cap_budget <= 0:
        return False
    ascii_count = accumulated_chars - cjk_chars
    estimated = int(cjk_chars * 1.5) + (ascii_count // 4)
    return estimated >= cap_budget


class ThinkingCapCounter:
    """Incremental char/CJK tracker mirroring _estimate_tokens arithmetic.

    Feed every thinking delta once; query after each delta. The running
    estimate equals _estimate_tokens(whole_text) up to int() rounding:
    both compute int(cjk*1.5) + ascii//4 over the concatenated text.
    """

    __slots__ = ("_chars", "_cjk")

    def __init__(self) -> None:
        self._chars = 0
        self._cjk = 0

    def add(self, delta: str) -> Tuple[int, int]:
        """Absorb one thinking delta; returns (chars, cjk) after absorption."""
        if delta:
            self._chars += len(delta)
            self._cjk += sum(
                1 for c in delta
                if '\u4e00' <= c <= '\u9fff'
                or '\u3040' <= c <= '\u30ff'
                or '\uac00' <= c <= '\ud7af'
            )
        return self._chars, self._cjk

    @property
    def chars(self) -> int:
        return self._chars

    @property
    def cjk(self) -> int:
        return self._cjk

    def state(self, cap_budget: int) -> Tuple[bool, Optional[int]]:
        """Return (capped_now, capped_at_chars) given a budget.

        capped_now  : threshold predicate over the accumulated text
        capped_at   : char count at which the cap first tripped, or None
        """
        if thinking_capped_state(self._chars, self._cjk, cap_budget):
            return True, self._chars
        return False, None
