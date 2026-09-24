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

"""MemoryPostCheckGuard — always-on post-write / post-read reconcile reminder.

Fires an advisory inject on EVERY successful memory_write / memory_read —
no cooldown, no per-key dedup, no per-session cap (removed by design: the
reconcile habit should be exercised on every memory op, not occasionally).
Failure paths stay silent: a failed write, an empty/missed read, and
non-memory tools never fire.

Two injects:
- after memory_write success ("Memorized [...]") → 4-point reconcile
  (contradiction/supersede · durability · generalization · key hygiene)
- after memory_read success (entry/entries found) → 3-point reconcile
  (staleness vs live env · conflict with other entries · did it answer)

Advisory only (inject, never block).
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class MemoryPostCheckGuard(Guard):
    """Reconcile memory the moment a read/write succeeds — every time."""

    name = "memory_post_check"
    priority = 88  # Advisory — below the hard gates, above nothing critical

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Post-only guard — never inspect a call before it runs.
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "memory_write" and ctx.tool_result:
            result = str(ctx.tool_result)
            if "Memorized [" not in result:
                return None  # write failed / errored — nothing to reconcile
            key = str(ctx.tool_args.get("key", ""))
            return GuardVerdict.inject(
                f"[MemoryPostCheck] You just wrote '{key or '?'}'. The write is not "
                "'done' until you have self-reconciled — check these in order and "
                "fix NOW if any applies:\n"
                "  (1) CONTRADICTION — does this entry contradict or supersede an "
                "existing entry on the same concept? If you did not pass "
                "supersedes=[...], a stale duplicate now coexists and is worse than "
                "no entry. Re-write with supersedes.\n"
                "  (2) DURABILITY — is this cross-session truth, or a one-off temp "
                "value? Temp belongs in plan notes, not global memory — bloat "
                "poisons everyone's later retrieval.\n"
                "  (3) GENERALIZATION — is this the 3rd+ time you recorded the same "
                "shape of finding? Digest it: recurring pitfall → insight/<domain>/, "
                "evidenced insight → skill/knowledge.\n"
                "  (4) KEY — would a session two weeks from now actually grep for "
                "this key? Run memory_list(keyword='<concept>') to check for a "
                "near-duplicate you should have reused instead.",
                reason="memory_write_reconcile",
                category="memory_post_check_write",
            )

        if ctx.tool_name == "memory_read" and ctx.tool_result:
            result = str(ctx.tool_result)
            # Only nudge when the read actually returned content.
            if "Content:" not in result and "entries:" not in result:
                return None
            key = str(ctx.tool_args.get("key", ""))
            return GuardVerdict.inject(
                f"[MemoryPostCheck] You just read '{key or '?'}'. Memory is a CLAIM, "
                "not ground truth — it records what was true when written, not now. "
                "Before you rely on it:\n"
                "  (1) STALENESS — if the entry describes environment state (a path, "
                "a version, a port, 'X is dead code', 'Y is the fix'), re-verify "
                "against the LIVE environment before acting. Files move, code "
                "changes, locks release.\n"
                "  (2) CONFLICT — do other entries on this concept disagree? "
                "memory_list(keyword='<concept>') and reconcile the survivor.\n"
                "  (3) GAIN — did this read actually answer your question, or do you "
                "still not know? If the memory itself is wrong, correct it at the "
                "source (memory_write with supersedes) rather than working around it.",
                reason="memory_read_staleness",
                category="memory_post_check_read",
            )

        return None
