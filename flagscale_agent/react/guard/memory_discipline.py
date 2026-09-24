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

"""MemoryDisciplineGuard — reminds the agent to use memory proactively.

Logic:
- Track tool calls since last memory read/write
- Every 10 calls without memory operation → inject a reminder
- Every 30 calls without memory operation → block (overridable)
- If LLM reads/writes memory, reset counter

Note: The [TASK_COMPLETE] completion-time memory review was moved to
VerificationGuard's _TEXT_COMPLETE_HYGIENE gate (block, not inject) so it
actually stops the agent before completion.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class MemoryDisciplineGuard(Guard):
    """Remind agent to read/write memory if it hasn't done so recently."""

    name = "memory_discipline"
    priority = 90  # Low priority — advisory only

    INJECT_THRESHOLD = 10
    BLOCK_THRESHOLD = 30

    def __init__(self):
        self._calls_since_memory = 0

    _MEMORY_TOOLS = frozenset((
        "memory_write", "memory_read", "memory_list",
        "plan_status", "plan_create", "plan_update",
    ))

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # The [TASK_COMPLETE] completion-time memory review was moved to
            # VerificationGuard's _TEXT_COMPLETE_HYGIENE gate (block, not inject)
            # so it actually stops the agent before completion. An inject here
            # was useless — the agent had already emitted [TASK_COMPLETE] and
            # would not act on advisory text.
            return None

        if ctx.tool_name in self._MEMORY_TOOLS:
            self._calls_since_memory = 0
            return None

        self._calls_since_memory += 1

        if self._calls_since_memory >= self.BLOCK_THRESHOLD:
            # Do NOT reset counter here — only reset in accept_override if override succeeds
            return GuardVerdict.block(
                f"[MemoryDiscipline] {self.BLOCK_THRESHOLD} tool calls without any memory operation. "
                "Before continuing, run ONE concrete recall action: "
                "memory_read(key='pitfall/<domain>/') for the domain you are working in "
                "(whole-domain pitfall read), or memory_list(keyword='...') if you only have "
                "a symptom keyword. If you have findings worth saving, memory_write() them — "
                "a finding not written is a finding lost at the next eviction.",
                reason=f"no_memory_ops_{self.BLOCK_THRESHOLD}_calls",
                category="memory_discipline",
            )

        if self._calls_since_memory % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self._calls_since_memory} tool calls without "
                "reading or writing memory. Pick ONE: "
                "(1) RECALL — memory_read(key='pitfall/<current-domain>/') before your next "
                "risky action (launch/build/deploy/new host); "
                "(2) WRITE — memory_write() a finding from this stretch (path/config/error fix); "
                "(3) DIGEST — recurring pitfall → insight, evidenced insight → skill/knowledge.",
                reason="no_memory_ops_recently",
                category="memory_discipline",
            )

        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of block if LLM provides a reason."""
        if reason and len(reason.strip()) > 5:
            self._calls_since_memory = 0
            return True
        return False


