"""Shared utilities for guards using two-phase detection.

Two-phase pattern:
1. Cheap trigger: keyword/threshold/counter (zero LLM cost, may have false positives)
2. Precise judgment: classify_fn LLM call (eliminates false positives)
"""

from __future__ import annotations

# Source constants — must match judge.py
SOURCE_LLM = "llm"
SOURCE_FAST = "fast"
SOURCE_CACHE = "cache"
SOURCE_DEFAULT = "default"
SOURCE_UNAVAILABLE = "unavailable"

_TRUSTED_SOURCES = frozenset({SOURCE_LLM, SOURCE_FAST, SOURCE_CACHE})


def get_judge_result(classify_fn, category: str, context: dict, default=None):
    """Call classify_fn and return (value, source) tuple.

    Shared utility for all guards that use two-phase detection.
    Returns (default, SOURCE_UNAVAILABLE) on any failure.
    """
    try:
        judge = getattr(classify_fn, "__self__", None)
        if judge and hasattr(judge, "classify_traced"):
            return judge.classify_traced(category, context, default)

        result = classify_fn(category, context, default=default)
        if (isinstance(result, tuple) and len(result) == 2
                and isinstance(result[1], str)
                and result[1] in (SOURCE_FAST, SOURCE_LLM, SOURCE_CACHE,
                                  SOURCE_DEFAULT, SOURCE_UNAVAILABLE)):
            return result
        return (result, SOURCE_LLM)
    except Exception:
        return (default, SOURCE_UNAVAILABLE)


def is_trusted(source: str) -> bool:
    """Check if a classify source is trustworthy (not a fallback default)."""
    return source in _TRUSTED_SOURCES
