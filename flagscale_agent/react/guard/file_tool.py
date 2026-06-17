"""FileToolGuard — prevents write_file truncation and helps with large file reading.

Two responsibilities:
1. Write Length Check: Before write_file with large content, inject a reminder
   to use append mode for content > 3000 chars.
2. Read Efficiency: After read_file on a large file (500 line limit hit),
   suggest using summarize patterns or targeted reads.

NOTE: The truncation issue is at the LLM output token level — the tool CALL
itself gets truncated because the content parameter is too long for a single
LLM response. The fix is behavioral: split writes proactively.
"""

import os

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class FileToolGuard(Guard):
    """Guard against file tool misuse patterns."""

    name = "file_tool"
    priority = 40  # Low priority — informational

    def __init__(self):
        self._large_file_warned: set[str] = set()
        self._truncation_count = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check write_file content length and warn about splitting."""
        if ctx.tool_name == "write_file":
            content = ctx.tool_args.get("content", "")
            mode = ctx.tool_args.get("mode", "write")

            # Only warn for initial writes (not appends) with large content
            if mode == "write" and len(content) > 4000:
                # Check if content looks truncated (ends mid-line or mid-string)
                if self._looks_truncated(content):
                    self._truncation_count += 1
                    return GuardVerdict.inject(
                        f"[FileTool] WARNING: Content appears truncated "
                        f"({len(content)} chars). This file write may be "
                        f"incomplete. Split large files: write first 3000 chars "
                        f"with mode='write', then append remaining with mode='append'.",
                        reason="possible_truncation",
                    )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """After read_file hits limit, suggest efficient alternatives."""
        if ctx.tool_name == "read_file" and ctx.tool_result:
            path = ctx.tool_args.get("path", "")
            # Detect "truncated at N lines" message in result
            if "truncated" in ctx.tool_result.lower() or "Use start_line=" in ctx.tool_result:
                if path not in self._large_file_warned:
                    self._large_file_warned.add(path)
                    return GuardVerdict.inject(
                        f"[FileTool] Large file hit read limit. For {path}:\n"
                        f"  - To find specific functions: grep -n 'def function_name' {path}\n"
                        f"  - To read a specific range: read_file(path, start_line=X, end_line=Y)\n"
                        f"  - Save key findings to memory_write() to avoid re-reading.",
                        reason="large_file_efficiency",
                    )

        # Check if write_file produced a file smaller than expected
        if ctx.tool_name == "write_file" and ctx.tool_result:
            content = ctx.tool_args.get("content", "")
            if "total file size:" in ctx.tool_result:
                # Extract reported size
                import re
                m = re.search(r"total file size:\s*(\d+)", ctx.tool_result)
                if m:
                    actual_size = int(m.group(1))
                    expected_size = len(content.encode("utf-8"))
                    if actual_size < expected_size * 0.9:
                        return GuardVerdict.inject(
                            f"[FileTool] Write may be incomplete: expected ~{expected_size} "
                            f"bytes but file is {actual_size} bytes. Use read_file to verify "
                            f"the file end, then append missing content with mode='append'.",
                            reason="write_possibly_incomplete",
                        )

        return None

    @staticmethod
    def _looks_truncated(content: str) -> bool:
        """Heuristic: does content look like it was cut off?"""
        if not content:
            return False
        # Ends with incomplete string (no closing quote/paren/bracket)
        last_line = content.rstrip().split("\n")[-1] if content.strip() else ""
        # Obvious truncation markers
        if content.rstrip().endswith(("...", "\\", ",")):
            return True
        # Unbalanced brackets suggest truncation
        opens = content.count("{") + content.count("[") + content.count("(")
        closes = content.count("}") + content.count("]") + content.count(")")
        if opens - closes > 3:
            return True
        # Unmatched triple-quotes
        triple_dq = content.count('"""')
        triple_sq = content.count("'''")
        if triple_dq % 2 != 0 or triple_sq % 2 != 0:
            return True
        return False

    def reset_turn(self):
        pass
