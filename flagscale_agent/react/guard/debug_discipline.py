"""DebugDisciplineGuard — enforces hypothesis-driven debugging and clean diffs.

Three responsibilities:
1. Hypothesis Gate: After a training failure, block code edits until the agent
   declares a hypothesis (what's wrong and why).
2. Diagnostic Maximization: When agent adds debug prints, inject reminder to
   add ALL needed prints in one pass (not one-at-a-time).
3. Clean Diff Gate: Before TASK_COMPLETE, scan modified files for debug residue.
"""

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Patterns that indicate debug code
_DEBUG_RESIDUE_PATTERNS = [
    re.compile(r"#\s*AGENT_DEBUG", re.I),
    re.compile(r"print\s*\(\s*f?\s*[\"'].*\[DBG", re.I),
    re.compile(r"print\s*\(\s*f?\s*[\"'].*DEBUG", re.I),
    re.compile(r"print\s*\(\s*f?\s*[\"'].*FIXME", re.I),
    re.compile(r"import\s+pdb", re.I),
    re.compile(r"pdb\.set_trace\(\)", re.I),
    re.compile(r"breakpoint\(\)", re.I),
    re.compile(r"# TODO.*remove.*debug", re.I),
    re.compile(r"# TEMP", re.I),
]

# Patterns indicating agent is adding debug prints
_ADDING_DEBUG_RE = re.compile(
    r"print\s*\(.*(?:debug|dbg|diag|trace|dump|log_)"
    r"|sys\.stdout\.write.*debug"
    r"|logging\.debug",
    re.IGNORECASE,
)

# Monitor failure indicators (reused from training_attempt)
_MONITOR_CRASH_RE = re.compile(
    r"TRAINING CRASHED|Traceback\s+\(most\s+recent"
    r"|RuntimeError:|NCCL\s+error|Out\s+of\s+memory"
    r"|CUDA\s+error|AttributeError:|ModuleNotFoundError:",
    re.IGNORECASE,
)


class DebugDisciplineGuard(Guard):
    """Enforce hypothesis-driven debugging and clean diffs."""

    name = "debug_discipline"
    priority = 22  # Between TrainingAttempt (15) and CircuitBreaker (25)
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override if LLM provides any non-trivial reason."""
        # Any explanation of why the edit is justified is sufficient
        if reason and len(reason.strip()) > 10:
            self._hypothesis_declared = True
            self._edits_since_failure = 0
            return True
        return False

    def __init__(self):
        # Hypothesis tracking
        self._failure_observed = False
        self._hypothesis_declared = False
        self._edits_since_failure = 0
        # Debug print tracking
        self._debug_prints_added = 0
        self._last_debug_file = ""
        # Modified files tracking
        self._modified_files: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Enforce hypothesis before edits after failure."""
        # Check assistant_text for hypothesis BEFORE evaluating the gate
        # This allows the LLM to declare hypothesis and write code in the same turn
        if ctx.assistant_text and not self._hypothesis_declared:
            text_lower = ctx.assistant_text.lower()
            if ("hypothesis:" in text_lower or "**hypothesis**" in text_lower
                    or "hypothesis**:" in text_lower
                    or "假设：" in text_lower or "根因" in text_lower
                    or "root cause:" in text_lower
                    or "hypothesis：" in text_lower):
                self._hypothesis_declared = True
                self._failure_observed = False
                self._edits_since_failure = 0

        # If assistant_text is empty (kernel doesn't provide it), skip the gate
        # to avoid false positives from stale _failure_observed state
        if not ctx.assistant_text:
            return None
        
        # If hypothesis was declared (either this turn or previously), allow edits
        if self._hypothesis_declared:
            return None

        if ctx.tool_name in ("edit_file", "write_file"):
            path = ctx.tool_args.get("path", "")
            if path and path.endswith(".py"):
                if self._failure_observed and not self._hypothesis_declared:
                    self._edits_since_failure += 1
                    if self._edits_since_failure >= 2:
                        return GuardVerdict.inject(
                            "[DebugDiscipline] You're editing code after a training "
                            "failure without declaring a hypothesis. Before making "
                            "more changes, state:\n"
                            "  HYPOTHESIS: [one sentence — what's actually wrong]\n"
                            "  EVIDENCE: [what you read/checked that supports this]\n"
                            "  VERIFICATION: [what you expect to see if correct]\n\n"
                            "This prevents blind trial-and-error. Training launches "
                            "are expensive — each attempt should test a specific theory.",
                            reason="hypothesis_required",
                        )
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track debug lifecycle events."""

        # Detect failure (from dedicated training monitor tools only)
        # NOTE: "shell" excluded because inline Python training (e.g., conv solvers)
        # produces Tracebacks as normal operation (architecture didn't converge).
        # Only dedicated training tools indicate a real infrastructure failure.
        if ctx.tool_name in ("monitor", "find_latest_log", "parse_training_metrics"):
            if ctx.tool_result and _MONITOR_CRASH_RE.search(ctx.tool_result):
                self._failure_observed = True
                self._hypothesis_declared = False
                self._edits_since_failure = 0

        # Detect hypothesis declaration via memory_write or direct memory_write calls
        if ctx.tool_name == "memory_write":
            content = ctx.tool_args.get("content", "").lower()
            key = ctx.tool_args.get("key", "").lower()
            if ("hypothesis" in content or "hypothesis" in key
                    or "root_cause" in key or "root cause" in content):
                self._hypothesis_declared = True
                self._edits_since_failure = 0

        # Also detect hypothesis in shell echo (legacy)
        if ctx.tool_name == "shell" and ctx.tool_args.get("command", "").startswith("echo HYPOTHESIS"):
            self._hypothesis_declared = True
            self._edits_since_failure = 0

        # Detect hypothesis declared inline in LLM assistant text
        if ctx.assistant_text:
            text_lower = ctx.assistant_text.lower()
            if ("hypothesis:" in text_lower or "**hypothesis**" in text_lower
                    or "假设：" in text_lower or "根因" in text_lower
                    or "root cause:" in text_lower
                    or "hypothesis：" in text_lower):
                self._hypothesis_declared = True
                self._edits_since_failure = 0

        # Track file modifications
        if ctx.tool_name in ("edit_file", "write_file"):
            path = ctx.tool_args.get("path", "")
            if path:
                self._modified_files.add(path)

            # Check if adding debug prints
            content = ctx.tool_args.get("content", "") or ctx.tool_args.get("new_string", "")
            if content and _ADDING_DEBUG_RE.search(content):
                self._debug_prints_added += 1
                self._last_debug_file = path

                if self._debug_prints_added == 1:
                    return GuardVerdict.inject(
                        "[DebugDiscipline] Adding debug prints. Remember: add ALL "
                        "diagnostic prints in ONE pass. Don't do one-print-per-launch. "
                        "Think about what OTHER intermediate values you need to see "
                        "and add them all now. Training startup cost is 2-5 minutes "
                        "— make each launch count.\n\n"
                        "Also: use the marker '# AGENT_DEBUG' on debug lines for "
                        "easy cleanup later.",
                        reason="debug_print_maximization",
                    )

        # Clean Diff Gate: when agent signals completion
        # (This hooks into the finalization flow — not a pre-check)

        return None

    def check_clean_diff(self, classify_fn=None) -> list[str]:
        """Check all modified files for debug residue. Call before TASK_COMPLETE.
        
        Tiered approach:
        1. Regex fast-path: catches explicit markers (AGENT_DEBUG, pdb, TEMP)
        2. LLM fallback: for ambiguous print statements that might be debug or production
        """
        residue_found = []
        ambiguous_lines = []  # Lines that regex didn't catch but might be debug
        import os
        for filepath in self._modified_files:
            if not os.path.isfile(filepath):
                continue
            if not filepath.endswith(".py"):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    # Tier 1: Regex fast-path — obvious debug markers
                    matched = False
                    for pat in _DEBUG_RESIDUE_PATTERNS:
                        if pat.search(line):
                            residue_found.append(f"  {filepath}:{i}: {line.strip()[:80]}")
                            matched = True
                            break
                    # Collect ambiguous cases for LLM tier
                    if not matched and "print(" in line and not line.strip().startswith("#"):
                        # Bare print statements in modified files — could be debug
                        context_start = max(0, i - 4)
                        context_end = min(len(lines), i + 3)
                        context = "".join(lines[context_start:context_end])
                        ambiguous_lines.append((filepath, i, line.strip(), context))
            except Exception:
                continue

        # Tier 2: LLM classify for ambiguous print statements
        if ambiguous_lines and classify_fn:
            # Batch check (max 5 to limit cost)
            for filepath, lineno, line_content, context in ambiguous_lines[:5]:
                try:
                    from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
                    result, source = get_judge_result(
                        classify_fn,
                        "is_debug_residue",
                        {
                            "filepath": filepath,
                            "lineno": lineno,
                            "line_content": line_content,
                            "context": context[:500],
                        },
                        default={"is_residue": False, "reason": ""},
                    )
                    if is_trusted(source) and isinstance(result, dict) and result.get("is_residue"):
                        reason = result.get("reason", "LLM detected debug code")
                        residue_found.append(
                            f"  {filepath}:{lineno}: {line_content[:60]} (LLM: {reason})"
                        )
                except Exception:
                    continue

        return residue_found

    def declare_hypothesis(self, hypothesis: str):
        """Called when the agent declares a hypothesis."""
        self._hypothesis_declared = True
        self._edits_since_failure = 0

    def reset_turn(self):
        """Reset failure state each turn to prevent stale cross-session persistence."""
        self._failure_observed = False
        self._hypothesis_declared = False
        self._edits_since_failure = 0
