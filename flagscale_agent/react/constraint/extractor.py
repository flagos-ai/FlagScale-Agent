"""Constraint extractor — LLM-based extraction from skill prose.

Reads a skill's SKILL.md content and extracts hard constraints that can be
enforced at the tool-call level. Uses Judge.classify("extract_constraints").
"""

from __future__ import annotations

from typing import Any, Callable

from flagscale_agent.react.constraint import Constraint, ConstraintTrigger


# Single-word keywords that are too generic and cause false triggers from file paths
_GENERIC_SINGLE_WORDS = frozenset({
    "pip", "pip3", "conda", "python", "python3", "install", "torch",
    "apex", "flagscale", "megatron", "flash", "attn", "cuda", "nvcc",
    "build", "source", "package", "train", "run", "setup",
})


def _validate_constraints(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate extracted constraints. Returns (valid_items, issues).

    Checks:
    1. tool_names must be non-empty
    2. keywords must not be single generic words that cause path false-triggers
    3. keywords must be >= 4 chars
    """
    valid = []
    issues = []

    for item in items:
        cid = item.get("id", "?")
        item_issues = []

        # Check tool_names
        tool_names = item.get("tool_names", [])
        if not tool_names:
            item_issues.append(
                f"[{cid}] tool_names is empty — every constraint must specify which tools it applies to"
            )

        # Check keywords
        keywords = item.get("keywords", [])
        for kw in keywords:
            kw_stripped = kw.strip().lower()
            if len(kw_stripped) < 4:
                item_issues.append(f"[{cid}] keyword '{kw}' is too short (< 4 chars)")
            elif " " not in kw_stripped and kw_stripped in _GENERIC_SINGLE_WORDS:
                item_issues.append(
                    f"[{cid}] keyword '{kw}' is a single generic word that will match file paths "
                    f"(e.g. '/path/to/{kw}/file'). Use a complete phrase like 'pip install {kw}' instead."
                )

        if item_issues:
            issues.extend(item_issues)
        else:
            valid.append(item)

    return valid, issues


def extract_constraints(
    skill_content: str,
    classify_fn: Callable[[str, dict], Any],
    skill_name: str = "unknown",
) -> list[Constraint]:
    """Extract hard constraints from skill prose via LLM.

    Validates the result and retries once if there are quality issues.

    Args:
        skill_content: Raw SKILL.md text (including frontmatter).
        classify_fn: Judge.classify or equivalent callable.
        skill_name: Skill identifier for constraint IDs.

    Returns:
        List of compiled Constraint objects. Empty list on failure.
    """
    if not skill_content.strip():
        return []

    # First attempt
    raw = _call_extract(classify_fn, skill_content, skill_name)
    if raw is None:
        return []

    valid, issues = _validate_constraints(raw)

    # Retry once if there are quality issues
    if issues:
        feedback = (
            "The previous extraction had quality issues that cause false triggers at runtime:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\n\nPlease re-extract, fixing these issues. Remember:\n"
            "- tool_names must be non-empty for every constraint\n"
            "- keywords must be complete phrases (e.g. 'pip install flagscale'), not single generic words\n"
            "- single words like 'flagscale', 'torch', 'apex' match file paths and cause false triggers"
        )
        from flagscale_agent.react import display
        print(display.yellow(
            f"  📋 [{skill_name}] {len(issues)} quality issue(s), retrying extraction..."
        ))

        retry_content = skill_content + f"\n\n<!-- FEEDBACK FROM PREVIOUS EXTRACTION:\n{feedback}\n-->"
        raw2 = _call_extract(classify_fn, retry_content, skill_name)
        if raw2 is not None:
            valid2, issues2 = _validate_constraints(raw2)
            if issues2:
                print(display.dim(
                    f"  📋 [{skill_name}] retry still has {len(issues2)} issue(s), using best result"
                ))
                # Use whichever pass gave more valid constraints
                valid = valid2 if len(valid2) > len(valid) else valid
            else:
                valid = valid2

    return _compile_all(valid, skill_name)


def _call_extract(
    classify_fn: Callable[[str, dict], Any],
    skill_content: str,
    skill_name: str,
) -> list[dict] | None:
    """Call LLM to extract constraints. Returns raw list or None on failure."""
    try:
        raw = classify_fn("extract_constraints", {"skill_content": skill_content})
    except Exception as e:
        return None

    if not isinstance(raw, list):
        return None

    return [item for item in raw if isinstance(item, dict)]


def _compile_all(items: list[dict], skill_name: str) -> list[Constraint]:
    """Compile validated raw dicts into Constraint objects."""
    constraints = []
    for i, item in enumerate(items):
        try:
            c = _compile_one(item, skill_name, i)
            if c is not None:
                constraints.append(c)
        except Exception as e:
            pass
    return constraints


def _compile_one(item: dict, skill_name: str, index: int) -> Constraint | None:
    """Compile a single raw dict from LLM into a Constraint object.

    Expected LLM output format per item:
    {
        "description": "Never delete experiment output directories",
        "tool_names": ["shell"],
        "keywords": ["rm -rf", "rmdir", "shutil.rmtree"],
        "prompt": "Does this command delete an experiment output directory?",
        "correction": "Do not delete experiment directories. Use archive instead."
    }

    Also supports legacy field names for backward compatibility:
    - "trigger_tools" -> "tool_names"
    - "trigger_keywords" -> "keywords"
    - "reminder" -> "correction"
    """
    description = item.get("description", "").strip()
    if not description:
        return None

    # Build trigger — support both new and legacy field names
    tool_names = set(item.get("tool_names", []) or item.get("trigger_tools", []) or [])
    keywords = list(item.get("keywords", []) or item.get("trigger_keywords", []) or [])
    trigger = ConstraintTrigger(tool_names=tool_names, keywords=keywords)

    # Build constraint — support both "correction" and legacy "reminder"
    constraint_id = item.get("id", f"{skill_name}_{index}")
    correction = item.get("correction", "") or item.get("reminder", "") or f"Constraint violated: {description}"
    return Constraint(
        id=constraint_id,
        description=description,
        trigger=trigger,
        prompt=item.get("prompt", f"Does this tool call violate: {description}"),
        correction=correction,
    )
