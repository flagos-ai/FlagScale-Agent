"""Constraint system — hard constraints from skills, compiled to Guards.

Design:
- Deterministic trigger: tool_name + keyword match (cheap, no LLM)
- Precise judgment: only when triggered, call Judge.classify() (LLM)
- Block behavior: violated constraints return block + correction
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Keyword normalization ────────────────────────────────────────────────

# Command aliases: keyword "pip" also matches "pip3", etc.
_COMMAND_ALIASES: dict[str, list[str]] = {
    "python": ["python3"],
    "python3": ["python"],
    "pip": ["pip3"],
    "pip3": ["pip"],
}


def _keyword_matches(keyword: str, search_text: str) -> bool:
    """Check if a keyword matches in search_text with normalization.

    Handles:
    1. Exact substring match (case-insensitive)
    2. Underscore/hyphen equivalence for package names (e.g. flash_attn ↔ flash-attn)
    3. Command aliases (e.g. pip ↔ pip3, python ↔ python3)
    """
    kw = keyword.lower()

    # Direct match
    if kw in search_text:
        return True

    # Underscore/hyphen equivalence
    if "_" in kw:
        if kw.replace("_", "-") in search_text:
            return True
    elif "-" in kw:
        if kw.replace("-", "_") in search_text:
            return True

    # Command alias expansion (only for command-like keywords)
    # Split keyword into parts, try replacing the command prefix
    for alias_from, alias_to_list in _COMMAND_ALIASES.items():
        if kw.startswith(alias_from + " ") or kw == alias_from:
            for alias_to in alias_to_list:
                variant = alias_to + kw[len(alias_from):]
                if variant in search_text:
                    return True

    return False


@dataclass
class ConstraintTrigger:
    """Deterministic trigger condition for a constraint.

    Checked before calling LLM for precise judgment.
    """
    tool_names: set[str] = field(default_factory=set)  # Empty = all tools
    keywords: list[str] = field(default_factory=list)  # Match in tool_args or tool_result

    def matches(self, tool_name: str, tool_args: dict, tool_result: str | None = None) -> bool:
        """Check if this trigger condition is satisfied."""
        # Never match on empty tool_name (pre-iteration guard check with no specific tool)
        if not tool_name:
            return False

        # Tool name filter
        if self.tool_names and tool_name not in self.tool_names:
            return False

        # Keyword filter (case-insensitive): at least 50% of keywords must match
        if self.keywords:
            # Combine all searchable text
            search_text = " ".join(str(v) for v in tool_args.values())
            if tool_result:
                search_text += " " + tool_result
            search_text = search_text.lower()

            # At least one keyword must match (with normalization)
            if not any(_keyword_matches(kw, search_text) for kw in self.keywords):
                return False

        # If both tool_names and keywords are empty, require at least tool_args or tool_result
        # to have content — prevents matching on vacuous pre-iteration checks
        if not self.tool_names and not self.keywords:
            if not tool_args and not tool_result:
                return False

        return True


@dataclass
class Constraint:
    """A hard constraint extracted from a skill.

    All constraints block on violation. No warning-only mode.
    """
    id: str
    description: str
    trigger: ConstraintTrigger

    # LLM judge prompt for precise violation detection
    prompt: str = ""

    # Correction message shown when violated
    correction: str = ""



@dataclass
class ConstraintViolation:
    """A detected constraint violation."""
    constraint_id: str
    reason: str
    correction: str
