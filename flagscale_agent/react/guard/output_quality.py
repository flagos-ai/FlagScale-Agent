"""OutputQualityGuard — detects silent failures and suspicious tool outputs.

Catches problems that other guards miss because the tool "succeeded" but the
result is wrong or empty:
1. edit_file with old_string that didn't match (no actual edit happened)
2. shell commands that return empty output when content is expected
3. write_file that silently truncated (file size much smaller than content)
4. shell commands that succeed (exit 0) but output contains error-like text
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState

# Patterns indicating a shell command "succeeded" but output looks like an error
_SUSPICIOUS_OUTPUT_PATTERNS = re.compile(
    r"warning:.*deprecated|"
    r"WARN|"
    r"fatal:|"
    r"Error:|"
    r"Traceback \(most recent|"
    r"SyntaxError|"
    r"NameError|"
    r"TypeError|"
    r"ValueError|"
    r"AssertionError|"
    r"IndentationError",
    re.IGNORECASE,
)

# Patterns that look like errors but are actually informational
_FALSE_POSITIVE_PATTERNS = re.compile(
    r"error_classifier|"  # Our own guard file
    r"ErrorClassifier|"
    r"error.*handler|"
    r"error.*log|"
    r"try.*except|"
    r"raise.*Error|"
    r"class.*Error|"
    r"def.*error|"
    r"#.*Error|"
    r"\".*Error.*\"|"
    r"'.*Error.*'|"
    r"SYNTAX OK|"
    r"grep.*Error",
    re.IGNORECASE,
)

# Shell commands where empty output is expected
_EMPTY_OK_COMMANDS = re.compile(
    r"^(mkdir|touch|rm|mv|cp|chmod|chown|ln|kill|pkill|export|cd|source|\.)\s",
    re.IGNORECASE,
)


class OutputQualityGuard(Guard):
    """Detects silent failures in tool execution.

    Activates in check_post since we need to examine the tool result.
    Does NOT use LLM judge — pure pattern matching for speed.
    """

    name = "output_quality"
    priority = 22  # Between LoopDetect (20) and ConstraintGuard (25)
    activate_on_states = {AgentState.EXECUTING}

    def __init__(self):
        self._consecutive_silent_failures: int = 0
        self._shared_state = None

    def set_shared_state(self, shared_state):
        self._shared_state = shared_state

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name or ctx.tool_result is None:
            return None

        result = ctx.tool_result

        # ── edit_file: detect "no match found" ──
        if ctx.tool_name == "edit_file":
            verdict = self._check_edit_file(ctx, result)
            if verdict:
                return verdict

        # ── write_file: detect truncation ──
        elif ctx.tool_name == "write_file":
            verdict = self._check_write_file(ctx, result)
            if verdict:
                return verdict

        # ── shell: detect suspicious output ──
        elif ctx.tool_name == "shell":
            verdict = self._check_shell(ctx, result)
            if verdict:
                return verdict

        # Reset consecutive failures on clean result
        self._consecutive_silent_failures = 0
        return None

    def _check_edit_file(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check edit_file for failed matches."""
        # Common indicators that edit_file didn't match
        if any(indicator in result.lower() for indicator in (
            "no match found",
            "old_string not found",
            "no changes made",
            "string not found",
        )):
            self._consecutive_silent_failures += 1
            old_str = ctx.tool_args.get("old_string", "")
            preview = old_str[:60] + "..." if len(old_str) > 60 else old_str
            return GuardVerdict.inject(
                f"[OutputQuality] edit_file failed — old_string not found in file. "
                f"The file was NOT modified. Preview: '{preview}'\n"
                f"Read the file first to find the exact string to replace.",
                reason="edit_file_no_match",
                category="output_quality",
            )

        # Check if result indicates the edit was applied to 0 locations
        if "0 replacements" in result.lower():
            self._consecutive_silent_failures += 1
            return GuardVerdict.inject(
                "[OutputQuality] edit_file made 0 replacements. The file was NOT modified.",
                reason="edit_file_zero_replacements",
                category="output_quality",
            )

        return None

    def _check_write_file(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check write_file for potential truncation."""
        content = ctx.tool_args.get("content", "")
        # If we wrote content but result mentions truncation
        if "truncated" in result.lower() and len(content) > 3000:
            self._consecutive_silent_failures += 1
            return GuardVerdict.inject(
                f"[OutputQuality] write_file content was truncated "
                f"({len(content)} chars intended). The file may be incomplete. "
                f"Use mode='append' for large files, or split into multiple calls.",
                reason="write_file_truncated",
                category="output_quality",
            )
        return None

    def _check_shell(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check shell output for hidden errors."""
        cmd = ctx.tool_args.get("command", "")

        # Skip commands where empty output is fine
        if _EMPTY_OK_COMMANDS.match(cmd):
            return None

        # Check for suspicious patterns in output
        if _SUSPICIOUS_OUTPUT_PATTERNS.search(result):
            # Filter false positives (e.g., grepping for errors, python source code)
            # Only flag if pattern appears in non-code context
            lines_with_errors = []
            for line in result.split("\n"):
                if _SUSPICIOUS_OUTPUT_PATTERNS.search(line):
                    if not _FALSE_POSITIVE_PATTERNS.search(line):
                        lines_with_errors.append(line.strip())

            if lines_with_errors and len(lines_with_errors) <= 5:
                # Only warn about genuine-looking errors, not code review output
                # Additional filter: if the command was grep/find/cat, don't warn
                if not re.match(r"^(grep|find|cat|head|tail|rg|ag)\s", cmd):
                    self._consecutive_silent_failures += 1
                    error_preview = "\n  ".join(lines_with_errors[:3])
                    return GuardVerdict.inject(
                        f"[OutputQuality] Shell command succeeded but output contains "
                        f"error-like text:\n  {error_preview}\n"
                        f"Verify the command did what you intended.",
                        reason="shell_suspicious_output",
                        category="output_quality",
                    )

        return None

    def reset_turn(self):
        # Don't reset consecutive failures — they accumulate within a turn
        pass
