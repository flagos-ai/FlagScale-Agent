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

"""OutputQualityGuard — detects silent failures in tool outputs.

Catches problems that other guards miss because the tool "succeeded" but the
result is wrong or empty:
1. edit_file with old_string that didn't match (no actual edit happened)
2. shell commands that return empty output when content is expected
3. write_file that silently truncated (file size much smaller than content)

Note: shell error detection (error text in output) is handled by
ErrorClassifierGuard via LLM — no regex duplication here.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState

# Commands where empty output is normal
_EMPTY_OK_COMMANDS = (
    "mkdir", "rm", "cp", "mv", "chmod", "chown", "touch", "cd",
    "export", "source", "conda activate", "pip install",
    "git add", "git commit", "git push", "git checkout",
    "kill", "pkill", "nohup",
)


class OutputQualityGuard(Guard):
    """Detects silent tool failures where exit code is 0 but result is wrong."""

    name = "output_quality"
    priority = 30
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept any override — this guard is advisory."""
        if reason and len(reason.strip()) > 5:
            self._consecutive_silent_failures = 0
            return True
        return False

    def __init__(self):
        self._consecutive_silent_failures = 0

    def set_shared_state(self, shared_state):
        pass

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        result = ctx.tool_result or ""

        # ── edit_file: detect failed match ──
        if ctx.tool_name == "edit_file":
            verdict = self._check_edit_file(ctx, result)
            if verdict:
                return verdict

        # ── write_file: detect truncation ──
        elif ctx.tool_name == "write_file":
            verdict = self._check_write_file(ctx, result)
            if verdict:
                return verdict

        # ── shell: detect unexpected empty output ──
        elif ctx.tool_name == "shell":
            verdict = self._check_shell_empty(ctx, result)
            if verdict:
                return verdict

        # Reset consecutive failures on clean result
        self._consecutive_silent_failures = 0
        return None

    def _check_edit_file(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check edit_file for failed matches."""
        if any(indicator in result.lower() for indicator in (
            "no match found",
            "old_string not found",
            "no changes made",
            "string not found",
        )):
            self._consecutive_silent_failures += 1
            old_str = ctx.tool_args.get("old_string", "")
            preview = old_str[:80] + "..." if len(old_str) > 80 else old_str
            return GuardVerdict.inject(
                f"[OutputQuality] edit_file did not find the target string. "
                f"The file content may have changed. Re-read the file and retry.\n"
                f"  old_string preview: {preview!r}",
                reason="edit_no_match",
                category="output_quality",
            )
        return None

    def _check_write_file(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check write_file for truncation hints."""
        content = ctx.tool_args.get("content", "")
        if len(content) > 500 and "truncat" in result.lower():
            self._consecutive_silent_failures += 1
            return GuardVerdict.inject(
                "[OutputQuality] write_file may have truncated content. "
                "Check the file to confirm it was written completely.",
                reason="write_truncated",
                category="output_quality",
            )
        return None

    def _check_shell_empty(self, ctx: GuardContext, result: str) -> GuardVerdict | None:
        """Check shell for unexpected empty output."""
        if result.strip():
            return None

        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        # Many commands produce no output normally
        cmd_start = cmd.lstrip().split()[0] if cmd.strip() else ""
        if any(cmd.lstrip().startswith(ok) for ok in _EMPTY_OK_COMMANDS):
            return None
        if cmd_start in ("cd", "export", "set", "unset", "alias"):
            return None

        # Commands that are expected to produce output
        if cmd_start in ("ls", "cat", "echo", "python", "nvidia-smi", "ps", "df"):
            self._consecutive_silent_failures += 1
            if self._consecutive_silent_failures >= 2:
                return GuardVerdict.inject(
                    f"[OutputQuality] '{cmd[:60]}' produced no output "
                    f"({self._consecutive_silent_failures} times). "
                    f"Check if the command is correct or the target exists.",
                    reason="shell_empty_output",
                    category="output_quality",
                )
        return None

    def reset_turn(self):
        pass
