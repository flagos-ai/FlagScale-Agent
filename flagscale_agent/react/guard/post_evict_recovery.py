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

"""PostEvictRecoveryGuard — reminds agent to restore context after heavy eviction.

After evicting 10+ messages, the agent loses significant context. This guard
injects a reminder to run plan_status() and memory_read() before continuing
with other operations. Resets once recovery actions are taken.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Minimum evicted messages to trigger recovery reminder
EVICT_THRESHOLD = 10

# Tools considered "recovery actions" that satisfy the guard
_RECOVERY_TOOLS = frozenset((
    "plan_status", "plan_create", "plan_update",
    "memory_read", "memory_list",
    "recall",
))

# Tools that are eviction-related (don't trigger reminder during eviction itself)
_EVICT_TOOLS = frozenset(("evict",))


class PostEvictRecoveryGuard(Guard):
    """Remind agent to restore working state after heavy eviction.

    Tracks cumulative evictions within a turn. When threshold is crossed,
    injects a reminder before the next non-recovery tool call. Resets once
    the agent performs recovery actions (plan_status, memory_read, etc).
    """

    name = "post_evict_recovery"
    priority = 15  # High priority — context loss is critical

    def __init__(self):
        self._evicted_count = 0
        self._needs_recovery = False
        self._reminded = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track evictions and detect when recovery is needed."""
        if ctx.tool_name == "evict":
            # Count evicted messages from the result
            result = ctx.tool_result or ""
            # Parse "Evicted N message(s)" from result
            if "Evicted" in result:
                try:
                    count = int(result.split("Evicted")[1].split("message")[0].strip())
                    self._evicted_count += count
                except (ValueError, IndexError):
                    self._evicted_count += 5  # Conservative estimate

            if self._evicted_count >= EVICT_THRESHOLD:
                self._needs_recovery = True
                self._reminded = False
            return None

        # If agent performs recovery action, clear the flag
        if ctx.tool_name in _RECOVERY_TOOLS:
            if self._needs_recovery:
                self._needs_recovery = False
                self._reminded = False
                self._evicted_count = 0
            return None

        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Before non-recovery tool calls, remind to restore context."""
        if not ctx.tool_name:
            return None

        if not self._needs_recovery:
            return None

        if self._reminded:
            return None

        # Don't interrupt eviction flow
        if ctx.tool_name in _EVICT_TOOLS:
            return None

        # Don't interrupt recovery tools
        if ctx.tool_name in _RECOVERY_TOOLS:
            return None

        self._reminded = True
        return GuardVerdict.inject(
            f"[PostEvictRecovery] You just evicted {self._evicted_count}+ messages — "
            f"significant context was lost. Before continuing, restore your working state:\n"
            f"1. plan_status() — check current task progress and step notes\n"
            f"2. memory_list() — scan all memory entries to find relevant facts, "
            f"pitfalls, and insights for your current domain\n"
            f"3. If needed: recall(index=N) for specific evicted content\n"
            f"4. For deep recovery: grep/read_file on conversation_full.json in your "
            f"session directory to find past instructions, tool results, or code snippets "
            f"without re-executing commands\n\n"
            f"Do NOT proceed on stale assumptions. After recovery, verify these "
            f"commonly error-prone items against memory or plan notes:\n"
            f"- Exact commands and their syntax (flags, JSON formats, argument order)\n"
            f"- File paths and delivery paths (exact location the task requires output)\n"
            f"- Environment state (which user, which container, which working directory, "
            f"which branch)\n"
            f"- Task constraints (version numbers, exact formats, GIVEN values with "
            f"zero tolerance)\n"
            f"- Recently corrected memory entries (ensure you are using the updated "
            f"version, not the stale one)\n"
            f"- Previously failed approaches (avoid repeating the same mistake)",
            reason="heavy_eviction_detected",
            category="post_evict_recovery",
        )

    def reset_turn(self):
        """On new user message, allow re-reminding if still not recovered."""
        self._reminded = False
