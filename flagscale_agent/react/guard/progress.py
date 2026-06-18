"""ProgressGuard — detects read-only stalls and lack of productive output.

v2: Uses SharedState for centralized read tracking and deduplication with LoopDetectGuard.
Only fires when LoopDetect hasn't already warned about the same stall.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class ProgressGuard(Guard):
    """Monitors read-only tool patterns and prompts action.

    v2 changes:
    - Uses SharedState.read_stats instead of private counters
    - Checks SharedState.read_warning_issued before firing (avoids duplication)
    - TaskMode-aware thresholds
    """

    name = "progress"
    priority = 30
    activate_on_states = {AgentState.EXECUTING}

    # Base thresholds (multiplied by TaskMode.read_tolerance)
    _READ_ONLY_STREAK_WARN_BASE = 8    # Consecutive reads before warn
    _READ_ONLY_STREAK_BLOCK_BASE = 14  # Consecutive reads before block
    _REREAD_WARN_BASE = 3              # Re-reads of same file before warn

    def __init__(self):
        self._read_files: set = set()
        self._reread_count: int = 0
        self._shared_state = None
        self._warned_this_session: bool = False

    def set_shared_state(self, shared_state):
        """Receive SharedState from GuardRegistry."""
        self._shared_state = shared_state

    @property
    def _tolerance_multiplier(self) -> float:
        """Get read tolerance from TaskMode."""
        if self._shared_state:
            return self._shared_state.task_mode.read_tolerance
        return 1.0

    @property
    def _warn_threshold(self) -> int:
        return max(5, int(self._READ_ONLY_STREAK_WARN_BASE * self._tolerance_multiplier))

    @property
    def _block_threshold(self) -> int:
        return max(8, int(self._READ_ONLY_STREAK_BLOCK_BASE * self._tolerance_multiplier))

    @property
    def _reread_threshold(self) -> int:
        return max(2, int(self._REREAD_WARN_BASE * self._tolerance_multiplier))

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # If this is a productive (write) tool, reset state
        if not ctx.tool_effects.is_read_only:
            self._read_files.clear()
            self._reread_count = 0
            self._warned_this_session = False
            return None

        # v2: Check if LoopDetect or another guard already warned about reads this turn
        if self._shared_state and self._shared_state.read_warning_issued_this_turn:
            # Another guard already injected a read-stall warning; suppress ours.
            return None

        # Track re-reads
        if ctx.tool_name == "read_file":
            path = ctx.tool_args.get("path", "")
            if path and path in self._read_files:
                self._reread_count += 1
                if self._reread_count >= self._reread_threshold:
                    if self._shared_state:
                        self._shared_state.issue_read_warning()
                    return GuardVerdict.inject(
                        f"[Progress] You've re-read '{path.split('/')[-1]}' "
                        f"{self._reread_count} times. Consider using grep/shell for "
                        f"targeted lookups, or save findings to memory.",
                        reason="re-read_same_file",
                        category="read_stall",
                    )
            else:
                self._read_files.add(path)

        # Use SharedState for centralized consecutive read count
        consecutive_reads = 0
        if self._shared_state:
            consecutive_reads = self._shared_state.read_stats.consecutive_reads
        else:
            # Fallback: count from recent_tool_names
            consecutive_reads = 0
            for name in reversed(ctx.recent_tool_names):
                if name in ("read_file", "shell"):
                    consecutive_reads += 1
                else:
                    break

        # Check thresholds
        if consecutive_reads >= self._block_threshold:
            if self._shared_state:
                self._shared_state.issue_read_warning()
            return GuardVerdict.inject(
                f"[Progress] {consecutive_reads}/{self._block_threshold} "
                f"consecutive read-only calls. You have enough information to act. "
                f"Ask yourself: do I have enough to move forward? "
                f"If yes — write, build, or fix something. "
                f"If no — what specific piece is missing?",
                reason="read_only_stall_block",
                category="read_stall",
            )

        if consecutive_reads >= self._warn_threshold and not self._warned_this_session:
            self._warned_this_session = True
            if self._shared_state:
                self._shared_state.issue_read_warning()
            return GuardVerdict.inject(
                f"[Progress] {consecutive_reads}/{self._warn_threshold} "
                f"consecutive read-only calls. Consider acting on what you've gathered.",
                reason="read_only_stall_warn",
                category="read_stall",
            )

        return None

    def reset_turn(self):
        # Progress tracking accumulates across iterations within a turn.
        # Counters are reset by productive tool calls in check_post.
        pass
