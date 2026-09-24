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

"""BackupGuard — upfront backup reminder at task start.

Runs FIRST (priority=5, before safety=10). One reminder:
  1. On the FIRST shell command: remind LLM to check for irreplaceable inputs and
     back them up before touching anything.

The guard does NOT detect, parse, or scan anything — it just reminds. The LLM
decides what needs backup. Completion-time .bak cleanup is handled by
VerificationGuard's _TEXT_COMPLETE_HYGIENE gate.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_UPFRONT_BACKUP_MESSAGE = """[BackupGuard] This is your first shell command — before you touch any files, check whether the task provides irreplaceable input resources.

Irreplaceable inputs include:
  • Databases (.db, .sqlite) and their WAL files
  • Binary data files that cannot be regenerated
  • Input files for forensic analysis or byte-level validation
  • Any resource where "opening it" may mutate it (even without explicit rm)

If such files exist in the task directory, back them up FIRST:
  cp <file> <file>.bak
  cp <file.wal> <file.wal>.bak

Many tools (sqlite3, recovery utilities) mutate files the moment they open them — a
logical undo does NOT restore original bytes. Protect irreplaceable data before the
first touch.

If no irreplaceable inputs exist (task generates data from scratch, or inputs are
regenerable), override this with "_override_reason" explaining why no backup is needed."""


class BackupGuard(Guard):
    """Upfront backup reminder at task start, cleanup reminder at completion."""

    name = "backup"
    priority = 5

    def __init__(self):
        self._first_shell_seen = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # First shell command: upfront backup reminder.
        # NOTE: do NOT set _first_shell_seen here. If we consumed the flag on the
        # first block, a BATCHED first turn (LLM emits several shell calls at once)
        # would only block the FIRST shell — the per-tool pre-check loop in
        # kernel.py runs check_pre for each call in order, and once the flag flips
        # every later shell in the same batch slips through unguarded and executes.
        # The flag is only consumed once the block is actually RELEASED via a valid
        # _override_reason (see accept_override), so until then every shell in the
        # turn is blocked.
        if ctx.tool_name == "shell" and not self._first_shell_seen:
            return GuardVerdict.block(
                message=_UPFRONT_BACKUP_MESSAGE,
                reason="upfront_backup_check",
                category="backup",
                overridable=True,
            )

        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Release the upfront backup block. Consume the one-shot flag only when
        the override is genuinely accepted, so batched first-turn shells are all
        blocked until the LLM acknowledges the backup concern once."""
        accepted = bool(reason and len(reason.strip()) > 5)
        if accepted:
            self._first_shell_seen = True
        return accepted

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        pass
