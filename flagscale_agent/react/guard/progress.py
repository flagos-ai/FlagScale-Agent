"""ProgressGuard — detects read-only stalls and lack of productive output.

Uses tool_effects to determine read-only vs productive tools instead of hardcoded sets.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class ProgressGuard(Guard):
    """Detects read-only stalls and nudges agent toward productive action.

    Uses tool_effects.is_read_only to classify tools instead of hardcoded sets.
    """

    name = "progress"
    priority = 30
    activate_on_states = {AgentState.EXECUTING}

    # ── Thresholds ──
    _STALE_THRESHOLD_NORMAL = 25
    _STALE_THRESHOLD_PORTING = 40
    _STALE_THRESHOLD_DEBUG = 30
    _STALE_THRESHOLD_WORKER = 8  # Fix 5: Lower threshold for worker mode
    _STALE_EXTRA_FOR_BLOCK = 8
    _READS_HARD_CAP_NORMAL = 60
    _READS_HARD_CAP_PORTING = 80
    _READS_HARD_CAP_WORKER = 20  # Fix 5: Lower hard cap for worker mode

    def __init__(self):
        self._consecutive_reads: int = 0
        self._reads_since_last_new_file: int = 0
        self._rereads_without_save: int = 0
        self._read_files: set[str] = set()
        self._progress_triggers: int = 0
        self._progress_block_count: int = 0  # track repeated blocks for escalation
        # Mode flags (set externally)
        self.is_porting_mode: bool = False
        self.is_worker_mode: bool = False  # Fix 5: Worker mode flag
        self.consecutive_train_failures: int = 0

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Reset on productive action (tool that writes)
        is_productive = ctx.tool_effects.is_write or ctx.tool_name in (
            "write_file", "edit_file", "memory_write",
            "plan_create", "plan_update", "workspace_experiment",
        )
        # Shell: only productive if not read-only
        if ctx.tool_name == "shell" and not ctx.tool_effects.is_read_only:
            is_productive = True

        if is_productive:
            self._consecutive_reads = 0
            self._reads_since_last_new_file = 0
            self._rereads_without_save = 0
            self._progress_triggers = 0
            self._progress_block_count = 0
            return None

        # Track read-only calls
        if ctx.tool_effects.is_read_only:
            self._consecutive_reads += 1

            if ctx.tool_name == "read_file":
                path = ctx.tool_args.get("path", "") or ctx.tool_args.get("file_path", "")
                if path and path not in self._read_files:
                    self._read_files.add(path)
                    self._reads_since_last_new_file = 0
                elif path:
                    self._reads_since_last_new_file += 1
                    self._rereads_without_save += 1

        # Determine adaptive threshold
        stale_threshold = self._STALE_THRESHOLD_NORMAL
        if self.is_worker_mode:
            stale_threshold = self._STALE_THRESHOLD_WORKER
        elif self.is_porting_mode:
            stale_threshold = self._STALE_THRESHOLD_PORTING
        elif self.consecutive_train_failures >= 2:
            stale_threshold = self._STALE_THRESHOLD_DEBUG

        # Pattern 1: Re-reading without discovery
        if self._reads_since_last_new_file >= stale_threshold:
            self._progress_triggers += 1
            if self._reads_since_last_new_file >= stale_threshold + self._STALE_EXTRA_FOR_BLOCK:
                self._progress_block_count += 1
                if self._progress_block_count >= 3:
                    return GuardVerdict.escalate(
                        f"[PROGRESS] You've been busy but not productive — "
                        f"{self._reads_since_last_new_file} calls without new discoveries. "
                        "This means you're missing something fundamental. "
                        "State what you know, what you're looking for, and what's blocking you. "
                        "Then ask the user for direction.",
                        reason=f"progress_stall_persistent: {self._reads_since_last_new_file} reads",
                    )
                return GuardVerdict.block(
                    f"[PROGRESS] {self._reads_since_last_new_file} calls without "
                    f"new files or output. You're stuck — acknowledge it.\n"
                    "Ask yourself: am I missing information, or is my approach wrong? "
                    "Create a plan (plan_create) to structure what you know, "
                    "then continue with a specific hypothesis to test.",
                    reason=f"extended staleness: {self._reads_since_last_new_file} reads",
                )
            else:
                return GuardVerdict.inject(
                    "\n[PROGRESS] You're re-reading known files without learning anything new. "
                    "What specific question are you trying to answer? "
                    "If you've found what you need, move to action. "
                    "If not, a memory_write of current findings can clarify your next move.",
                    reason="re-reading known files",
                )

        # Pattern 2: Long exploration without checkpoint
        if self.is_worker_mode:
            reads_hard_cap = self._READS_HARD_CAP_WORKER
        elif self.is_porting_mode:
            reads_hard_cap = self._READS_HARD_CAP_PORTING
        else:
            reads_hard_cap = self._READS_HARD_CAP_NORMAL
        if self._consecutive_reads >= reads_hard_cap and self._progress_triggers <= 1:
            self._progress_triggers += 1
            return GuardVerdict.inject(
                "\n[CHECKPOINT SUGGESTION] You've done extensive exploration. "
                "Consider a memory_write to persist key findings — this protects "
                "against context compaction loss.",
                reason=f"extended exploration: {self._consecutive_reads} reads",
            )

        return None

    def reset_turn(self):
        # Do NOT reset _read_files or counters here — reset_turn is called per iteration.
        # Progress tracking needs to accumulate across iterations within a turn.
        # Counters are reset by productive tool calls in check_post.
        pass
