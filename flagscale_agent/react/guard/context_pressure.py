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

"""Context pressure guard — blocks tool calls when context is too full.

Design:
- All detection in check_pre (before tool execution)
- Two paths: evict path (recoverable) and hard_reset path (unrecoverable)
- No inject — only block with clear instructions
- Minimal state: one bool flag (_need_hard_reset)
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Thresholds
BLOCK_RATIO = 0.80       # Start blocking at 80%
RELEASE_RATIO = 0.50     # Release block when evicted below 50%
EVICTABLE_THRESHOLD = 60 # If evictable < 60, need hard_reset instead of evict


class ContextPressureGuard(Guard):
    """Blocks tools when context pressure is too high.

    Two paths:
    - Evict path: pressure >= 80% AND evictable >= 60
      → block until pressure < 50%
    - Hard reset path: pressure >= 80% AND evictable < 60
      → block until hard_reset is called
    """

    name = "context_pressure"
    priority = 10

    # Tools allowed through during block
    _SAVE_TOOLS = frozenset({
        "memory_write", "memory_read", "memory_list",
        "plan_update", "plan_status", "plan_create",
        "evict", "recall",
        "hard_reset",
    })

    def __init__(self, working_window_tokens: int = 0):
        self._need_hard_reset = False
        self._working_window_tokens = working_window_tokens

    @property
    def working_window_tokens(self) -> int:
        return self._working_window_tokens or 120_000

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Skip the pre-LLM-call check (tool_name=""). We only gate actual tool
        # executions — blocking before the LLM call would prevent the LLM from
        # ever invoking evict/hard_reset, creating an infinite block loop.
        if not ctx.tool_name:
            return None

        pressure = ctx.context_pressure
        if pressure <= 0:
            return None

        evictable = ctx.evictable_indexes
        pct = int(pressure * 100)

        # Hard reset path — auto-release if conditions have recovered
        if self._need_hard_reset:
            # Recovery check: if evictable grew back above threshold OR
            # pressure dropped below block ratio, release the lock
            if len(evictable) >= EVICTABLE_THRESHOLD or pressure < BLOCK_RATIO:
                self._need_hard_reset = False
                # Fall through to normal threshold check below
            else:
                if ctx.tool_name in self._SAVE_TOOLS:
                    return None
                return GuardVerdict.block(
                    f"[Context pressure {pct}% with only "
                    f"{len(evictable)} evictable messages] "
                    f"Eviction cannot free enough space. Execute in order:\n"
                    f"1. memory_write() — EXTRACT NOW. You still hold the FULL message text\n"
                    f"   that is about to be discarded; after the reset it is gone and no\n"
                    f"   later session can ever see it. One entry now costs one call, while\n"
                    f"   re-discovering it later costs many turns.\n"
                    f"   WHERE — the survival-range test decides the container:\n"
                    f"     * cross-session truth (exact commands, paths, env state, pitfalls,\n"
                    f"       configs, decisions+why) -> memory_write() — GLOBAL.\n"
                    f"     * this-session progress (trial state, current hypothesis, temp paths)\n"
                    f"       -> plan_update(notes='...').\n"
                    f"   WHAT NOT — do not dump what one ls/grep/git status can cheaply re-derive;\n"
                    f"   bloat poisons later retrieval.\n"
                    f"2. plan_update(notes='...') — record current state\n"
                    f"3. hard_reset(reason='...') — reset context. Only memory survives a reset;\n"
                    f"   the continuation summary lives only in the next window, so a second\n"
                    f"   reset loses it — put durable facts in memory, not just continuation.\n"
                    f"Allowed tools: {', '.join(sorted(self._SAVE_TOOLS))}",
                    reason="hard_reset_required",
                    category="context_pressure_hard_reset",
                )

        # Below block threshold — pass
        if pressure < BLOCK_RATIO:
            return None

        # At or above 80% — decide which path
        if len(evictable) < EVICTABLE_THRESHOLD:
            # Not enough to evict — need hard_reset
            self._need_hard_reset = True
            if ctx.tool_name in self._SAVE_TOOLS:
                return None
            return GuardVerdict.block(
                f"[Context pressure {pct}% with only "
                f"{len(evictable)} evictable messages] "
                f"Eviction cannot free enough space. Execute in order:\n"
                f"1. memory_write() — EXTRACT NOW. You still hold the FULL message text\n"
                f"   that is about to be discarded; after the reset it is gone and no\n"
                f"   later session can ever see it. One entry now costs one call, while\n"
                f"   re-discovering it later costs many turns.\n"
                f"   WHERE — the survival-range test decides the container:\n"
                f"     * cross-session truth (exact commands, paths, env state, pitfalls,\n"
                f"       configs, decisions+why) -> memory_write() — GLOBAL.\n"
                f"     * this-session progress (trial state, current hypothesis, temp paths)\n"
                f"       -> plan_update(notes='...').\n"
                f"   WHAT NOT — do not dump what one ls/grep/git status can cheaply re-derive;\n"
                f"   bloat poisons later retrieval.\n"
                f"2. plan_update(notes='...') — record current state\n"
                f"3. hard_reset(reason='...') — reset context. Only memory survives a reset;\n"
                f"   the continuation summary lives only in the next window, so a second\n"
                f"   reset loses it — put durable facts in memory, not just continuation.\n"
                f"Allowed tools: {', '.join(sorted(self._SAVE_TOOLS))}",
                reason="hard_reset_required",
                category="context_pressure_hard_reset",
            )
        else:
            # Enough to evict — block until pressure < 50%
            if ctx.tool_name in self._SAVE_TOOLS:
                return None
            return GuardVerdict.block(
                f"[Context pressure {pct}% with "
                f"{len(evictable)} evictable messages] "
                f"Evict aggressively until pressure drops below 50%. "
                f"Call evict(indexes=[...]) with wide ranges.\n"
                f"BEFORE evicting, save progress — evicting destroys context:\n"
                f"1. memory_write() — EXTRACT NOW. You still hold the FULL message text that\n"
                f"   is about to become a placeholder; after eviction no later session can\n"
                f"   ever see it. This is your last and best moment to record it — one entry\n"
                f"   now costs one call, re-discovering it later costs many turns.\n"
                f"   WHERE — the survival-range test decides the container:\n"
                f"     * cross-session truth (exact commands, paths, env state, pitfalls,\n"
                f"       configs, decisions+why) -> memory_write() — GLOBAL, the next session\n"
                f"       greps it too. No plan? memory_write is your ONLY recovery mechanism\n"
                f"       after eviction.\n"
                f"     * this-session progress (trial state, current hypothesis, temp paths)\n"
                f"       -> plan_update(notes='...') — lives with THIS session.\n"
                f"   WHAT NOT — do not dump what one ls/grep/git status can cheaply re-derive;\n"
                f"   bloat poisons later retrieval. Record the non-reproducible.\n"
                f"2. plan_update(notes='...') — record current state\n"
                f"3. evict(indexes=[...]) — free context space\n"
                f"Evictable: {evictable}\n"
                f"Allowed tools: {', '.join(sorted(self._SAVE_TOOLS))}",
                reason="evict_required",
                category="context_pressure_evict",
            )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Reset _need_hard_reset after hard_reset or successful eviction."""
        if self._need_hard_reset:
            if ctx.tool_name == "hard_reset":
                self._need_hard_reset = False
            elif ctx.tool_name == "evict":
                # After eviction, re-check if conditions improved
                pressure = ctx.context_pressure
                evictable = ctx.evictable_indexes
                if len(evictable) >= EVICTABLE_THRESHOLD or pressure < BLOCK_RATIO:
                    self._need_hard_reset = False
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override and clear the block state.

        When an override is accepted, clear _need_hard_reset so subsequent
        tool calls aren't blocked again. Without this, the LLM would need
        to override every single call — the override doesn't "stick".
        """
        accepted = bool(reason and len(reason.strip()) > 5)
        if accepted:
            self._need_hard_reset = False
        return accepted

    def reset_turn(self):
        pass
