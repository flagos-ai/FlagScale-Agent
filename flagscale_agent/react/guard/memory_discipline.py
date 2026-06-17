"""MemoryDisciplineGuard — enforces timely memory reads and writes.

Problems this solves:
1. WRITE: Agent discovers critical info (log paths, rank assignments, error patterns)
   but doesn't memory_write it → lost on compaction → rediscovered from scratch
2. READ: Agent starts a new session/task without loading relevant memories →
   repeats previously-completed discovery work

Strategy:
- Post-check: After certain tool calls that reveal important information,
  inject a reminder to memory_write if the agent hasn't done so
- Pre-check: At session start (first tool call), inject reminder to memory_list
  for relevant context
- Track what's been written vs what should have been written
"""

import re
import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Patterns that indicate important discoverable information
_IMPORTANT_DISCOVERY_PATTERNS = [
    # Training output paths
    (re.compile(r"logs/details/host_\d+.*?/\d{8}_\d+", re.I), "log_timestamp_path"),
    # Rank that has metrics
    (re.compile(r"(?:rank|Rank)\s+(\d+).*(?:metrics|loss|iteration)", re.I), "metrics_rank"),
    # Active megatron path
    (re.compile(r"megatron/__file__.*?['\"]([^'\"]+)['\"]", re.I), "megatron_path"),
    # Checkpoint paths
    (re.compile(r"(?:checkpoint|ckpt).*?(?:saved|loaded).*?['\"]?(/\S+)", re.I), "checkpoint_path"),
    # NCCL/communication settings that work
    (re.compile(r"NCCL_\w+=\S+", re.I), "nccl_env"),
    # Working parallelism config
    (re.compile(r"(?:tp|pp|dp|ep)\s*=\s*\d+.*(?:works?|success)", re.I), "parallelism_config"),
]

# Patterns indicating the agent already wrote to memory
_MEMORY_WRITE_INDICATOR = re.compile(r"memory_write")

# Keys of memories that should be checked at session start
_SESSION_START_KEYWORDS = [
    "workspace_root",
    "megatron",
    "output_dir",
    "experiment",
    "env_path",
    "topology",
]


class MemoryDisciplineGuard(Guard):
    """Force timely memory reads at session start and writes after discoveries."""

    name = "memory_discipline"
    priority = 35  # Medium-low — informational but important

    def __init__(self):
        # Session start tracking
        self._session_started = False
        self._memory_loaded = False
        self._tool_call_count = 0

        # Write discipline tracking
        self._pending_discoveries: list[tuple[str, str]] = []  # (category, snippet)
        self._discoveries_since_last_write = 0
        self._last_memory_write_time = 0.0
        self._total_discoveries_unwritten = 0

        # Suppress repeated reminders
        self._write_reminder_count = 0
        self._read_reminder_given = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """At session start, remind to load memories."""
        # Only count real tool calls (per-tool), not turn-level empty ctx
        if not ctx.tool_name:
            return None

        self._tool_call_count += 1

        # Always track memory load status (regardless of reminder state)
        if not self._memory_loaded and ctx.tool_name in ("memory_list", "memory_read"):
            self._memory_loaded = True
            return None

        # Session start reminder (first 3 tool calls)
        if self._tool_call_count <= 3 and not self._memory_loaded and not self._read_reminder_given:
            # Don't interrupt if agent is already doing plan_status (also good start)
            if ctx.tool_name == "plan_status":
                return None

            self._read_reminder_given = True
            return GuardVerdict.inject(
                "[MemoryDiscipline] Session start: load relevant context before acting.\n"
                "  → memory_list() to see all stored findings\n"
                "  → plan_status() to check active plans\n"
                "This prevents re-discovering information from previous sessions.",
                reason="session_start_memory_load",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """After tool calls that reveal important info, remind to memory_write."""

        # Track memory writes
        if ctx.tool_name == "memory_write":
            self._last_memory_write_time = time.time()
            self._discoveries_since_last_write = 0
            self._pending_discoveries.clear()
            return None

        # Track memory reads
        if ctx.tool_name in ("memory_list", "memory_read"):
            self._memory_loaded = True
            return None

        # Scan tool results for important discoveries (Tiered: regex fast-path + LLM)
        if ctx.tool_result and len(ctx.tool_result) > 50:
            discovered = False
            # Tier 1: Regex fast-path
            for pattern, category in _IMPORTANT_DISCOVERY_PATTERNS:
                if pattern.search(ctx.tool_result):
                    self._pending_discoveries.append((category, ctx.tool_result[:100]))
                    self._discoveries_since_last_write += 1
                    self._total_discoveries_unwritten += 1
                    discovered = True
                    break

            # Tier 2: LLM classify for non-obvious discoveries
            # Only invoke on substantial tool outputs that regex missed
            if (not discovered
                    and ctx.classify_fn
                    and len(ctx.tool_result) > 200
                    and ctx.tool_name not in ("memory_write", "memory_read", "memory_list")
                    and self._tool_call_count % 5 == 0):  # Rate-limit: every 5th call
                try:
                    from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
                    result, source = get_judge_result(
                        ctx.classify_fn,
                        "is_important_discovery",
                        {"tool_name": ctx.tool_name, "tool_output": ctx.tool_result[:1500]},
                        default={"important": False, "key_info": None},
                    )
                    if is_trusted(source) and isinstance(result, dict) and result.get("important"):
                        key_info = result.get("key_info", "unspecified")
                        self._pending_discoveries.append(("llm_detected", key_info[:100]))
                        self._discoveries_since_last_write += 1
                        self._total_discoveries_unwritten += 1
                except Exception:
                    pass

        # After monitor/find_latest_log with results — these ALWAYS contain memorizable info
        if ctx.tool_name in ("monitor", "find_latest_log") and ctx.tool_result:
            if "stdout.log" in ctx.tool_result or "iteration" in ctx.tool_result.lower():
                self._discoveries_since_last_write += 1
                self._pending_discoveries.append(("training_state", ctx.tool_result[:80]))

        # Inject reminder when enough unwritten discoveries accumulate
        if self._discoveries_since_last_write >= 3 and self._write_reminder_count < 3:
            self._write_reminder_count += 1
            categories = list(set(cat for cat, _ in self._pending_discoveries[-5:]))
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self._discoveries_since_last_write} important "
                f"findings since last memory_write (categories: {', '.join(categories)}). "
                f"Write key discoveries to memory now — they'll be lost on compaction.\n"
                f"  → memory_write(key=..., type='finding', content=...)\n"
                f"Focus on: paths, working configs, rank assignments, error patterns.",
                reason="write_discoveries_to_memory",
            )

        # Periodic reminder: if >15 tool calls since last write and there are pending items
        if (self._tool_call_count % 15 == 0
            and self._pending_discoveries
            and time.time() - self._last_memory_write_time > 120
            and self._write_reminder_count < 5):
            self._write_reminder_count += 1
            return GuardVerdict.inject(
                f"[MemoryDiscipline] Periodic reminder: {len(self._pending_discoveries)} "
                f"unwritten discoveries. Consider batching them into memory_write calls "
                f"to preserve context across compaction/sessions.",
                reason="periodic_memory_write_reminder",
            )

        return None

    def reset_turn(self):
        """Memory state persists across turns."""
        pass
