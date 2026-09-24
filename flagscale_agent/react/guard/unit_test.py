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

"""UnitTestGuard — reminds LLM to write unit tests when modifying agent source code.

Triggers on write_file/edit_file targeting flagscale_agent/ (excluding tests/ itself).
Post-check only: fires after a successful write/edit, not before.
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict


class UnitTestGuard(Guard):
    """Post-tool guard that reminds to write unit tests for agent source changes."""

    name = "unit_test_reminder"
    priority = 70  # Low priority — advisory, not blocking

    # Paths that should trigger the reminder
    SOURCE_MARKERS = ("flagscale_agent/",)
    # Paths that are themselves tests — don't remind on these
    TEST_MARKERS = ("tests/", "test_", "/test_")

    def __init__(self):
        self._pending_sources: set[str] = set()  # Source files modified without tests
        self._test_written_this_turn = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # No pre-check needed

    # Only these tools actually modify files
    WRITE_TOOLS = ("write_file", "edit_file")

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only trigger on file-writing operations
        if ctx.tool_name not in self.WRITE_TOOLS:
            return None

        path = ctx.tool_args.get("path", "") or ""

        # Track if a test file was written
        if self._is_test_file(path):
            self._test_written_this_turn = True
            # Clear pending sources — tests are being written
            self._pending_sources.clear()
            return None

        # Track source file modifications
        if self._is_agent_source(path):
            self._pending_sources.add(path)

            # Only fire reminder after accumulating changes, not on every edit
            if len(self._pending_sources) >= 2 and not self._test_written_this_turn:
                files_str = ", ".join(sorted(self._pending_sources)[-3:])
                return GuardVerdict.inject(
                    f"[UnitTest] You've modified agent source files ({files_str}) "
                    f"without corresponding test updates. Per project rules, write/update "
                    f"unit tests in tests/ before claiming the change is complete.",
                    reason="unit_test_needed",
                    category="unit_test_reminder",
                )

        return None

    def reset_turn(self):
        """Reset per-turn state but preserve pending sources across turns."""
        self._test_written_this_turn = False

    @staticmethod
    def _is_agent_source(path: str) -> bool:
        """Check if path is agent source code (not tests, not configs)."""
        if not path:
            return False
        # Must be in flagscale_agent/
        if not any(marker in path for marker in UnitTestGuard.SOURCE_MARKERS):
            return False
        # Must not be a test file itself
        if UnitTestGuard._is_test_file(path):
            return False
        # Must be a Python file
        if not path.endswith(".py"):
            return False
        return True

    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Check if path is a test file."""
        if not path:
            return False
        return any(marker in path for marker in UnitTestGuard.TEST_MARKERS)
