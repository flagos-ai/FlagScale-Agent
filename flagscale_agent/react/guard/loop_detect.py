"""LoopDetectGuard — detects repeated/looping tool calls.

Uses two-phase detection:
1. Cheap trigger: counters/ratios/patterns exceed thresholds
2. Precise judgment: classify_fn("is_stuck_in_loop") confirms before escalation

v2 improvements:
- SharedState integration for TaskMode-aware thresholds
- Continuation-read exemption (same file, different line ranges)
- Argument diversity check in same_tool_dominance
- Read-warning suppression when another guard already warned
"""

from __future__ import annotations

from collections import Counter

from flagscale_agent.react import display
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState

# Tools that only read state without modifying it
_READ_ONLY_TOOL_NAMES = frozenset({
    "read_file", "memory_read", "memory_list", "plan_status",
    "find_log", "parse_metrics", "monitor", "web_fetch",
    "validate_config", "inspect_checkpoint",
})

# Shell command prefixes that are read-only (don't modify state)
_READ_ONLY_SHELL_PREFIXES = (
    "ls ", "ls\n", "find ", "cat ", "head ", "tail ", "grep ",
    "which ", "echo ", "pwd", "env ", "printenv",
    "nvidia-smi", "nvcc ", "python --version", "python -c \"import",
    "python3 --version", "python3 -c \"import",
    "df ", "du ", "free ", "top ", "ps ", "uname ",
    "whoami", "hostname", "date", "wc ",
)

# Shell command patterns that indicate a retry loop (kill/restart cycles)
_RETRY_PATTERNS = (
    "kill", "pkill", "killall",
    "torchrun", "python -m torch.distributed", "deepspeed",
    "flagscale", "train.py", "pretrain",
)


class LoopDetectGuard(Guard):
    """Detects when the agent is looping on the same tool calls.

    Activates in EXECUTING state.
    Three detection modes:
    1. Exact match: same (tool_name, key_args) repeated N times
    2. Semantic: read-only tools dominate recent history with no writes
    3. Retry pattern: kill→launch cycles without diagnostic steps between them

    v2: Integrates with SharedState for:
    - TaskMode-aware thresholds (analysis mode is more tolerant)
    - Continuation-read exemption (reading same file at different offsets)
    - Read-warning deduplication (only one guard warns about reads per turn)
    """

    name = "loop_detect"
    priority = 20
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    _MAX_RECENT = 12

    # Detection 1: Exact match
    _LOOP_THRESHOLD = 3  # same (tool, args) N times in recent history

    # Detection 1B: Same tool dominance (base thresholds, multiplied by TaskMode)
    _SAME_TOOL_DOMINANCE_BASE = 8  # same tool_name N times in window
    _SAME_TOOL_WINDOW = 12

    # Detection 2: Semantic read-only ratio (base thresholds)
    _SEMANTIC_WINDOW = 12
    _SEMANTIC_READ_RATIO_BASE = 0.85  # adjusted by TaskMode
    _SEMANTIC_COOLDOWN = 4

    # Detection 3: Retry pattern
    _RETRY_WINDOW = 10

    def __init__(self):
        self._recent_tool_calls: list[tuple[str, str]] = []  # (tool_name, key_args)
        self._tool_name_history: list[str] = []
        self._shell_cmd_history: list[str] = []
        self._total_tool_calls: int = 0
        self._tool_call_cache: dict = {}  # for per-turn dedup

        # Detection 1 state
        self._exact_loop_inject_count: int = 0

        # Detection 2 state
        self._semantic_warned: bool = False
        self._semantic_warn_at: int = 0
        self._semantic_warn_count: int = 0

        # SharedState reference (set by GuardRegistry)
        self._shared_state = None

    def set_shared_state(self, shared_state):
        """Called by GuardRegistry to inject shared state."""
        self._shared_state = shared_state

    @property
    def _task_mode_multiplier(self) -> float:
        """Get loop sensitivity multiplier from TaskMode."""
        if self._shared_state:
            return self._shared_state.task_mode.loop_sensitivity
        return 1.0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        self._total_tool_calls += 1
        key_args = self._extract_key_args(ctx.tool_args)
        entry = (ctx.tool_name, key_args)

        # Track history
        self._recent_tool_calls.append(entry)
        if len(self._recent_tool_calls) > self._MAX_RECENT:
            self._recent_tool_calls = self._recent_tool_calls[-self._MAX_RECENT:]
        self._tool_name_history.append(ctx.tool_name)

        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "").lower()
            self._shell_cmd_history.append(cmd)
            if len(self._shell_cmd_history) > self._RETRY_WINDOW:
                self._shell_cmd_history = self._shell_cmd_history[-self._RETRY_WINDOW:]

        # ── Detection 1: Exact match loop ──
        recent_same = sum(
            1 for t in self._recent_tool_calls[-self._MAX_RECENT:]
            if t == entry
        )
        if recent_same >= self._LOOP_THRESHOLD:
            # Continuation-read exemption: reading same file at different offsets is not a loop
            if self._is_continuation_read(ctx.tool_name, ctx.tool_args):
                pass  # Exempt — this is sequential reading
            else:
                self._exact_loop_inject_count += 1
                if self._exact_loop_inject_count >= 3:
                    return GuardVerdict.escalate(
                        f"[LoopDetect] Same tool call repeated {recent_same} times across "
                        f"{self._exact_loop_inject_count} warnings. "
                        "You're in a loop. The approach isn't working — repeating it won't help. "
                        "Diagnose why it's failing and propose a different strategy.",
                        reason=f"exact_loop_persistent: {ctx.tool_name}",
                    )
                return GuardVerdict.inject(
                    f"[LoopDetect] Same tool call repeated {recent_same} times. "
                    "Each attempt gave the same result. "
                    "Why? What's different about what you need vs what you're getting? "
                    "Answer that before trying again.",
                    reason=f"looping on {ctx.tool_name}",
                )

        # ── Detection 1B: Same tool dominance (with argument diversity check) ──
        if len(self._tool_name_history) >= self._SAME_TOOL_WINDOW:
            window = self._tool_name_history[-self._SAME_TOOL_WINDOW:]
            same_tool_count = sum(1 for t in window if t == ctx.tool_name)
            # Apply TaskMode multiplier to threshold
            adjusted_threshold = int(self._SAME_TOOL_DOMINANCE_BASE * self._task_mode_multiplier)

            if same_tool_count >= adjusted_threshold:
                # NEW: Check argument diversity before triggering
                # If the agent is calling the same tool with DIFFERENT arguments,
                # it's exploring (not looping)
                recent_same_tool = [
                    args for name, args in self._recent_tool_calls[-self._SAME_TOOL_WINDOW:]
                    if name == ctx.tool_name
                ]
                arg_diversity = len(set(recent_same_tool)) / max(len(recent_same_tool), 1)

                # High diversity (>50% unique args) = legitimate exploration
                if arg_diversity > 0.50:
                    pass  # Not a loop — different targets each time
                else:
                    # Check SharedState: has another guard already warned about reads?
                    if self._shared_state and not self._shared_state.issue_read_warning():
                        pass  # Suppress — another guard already warned
                    else:
                        return GuardVerdict.inject(
                            f"[LoopDetect] '{ctx.tool_name}' called {same_tool_count}/{self._SAME_TOOL_WINDOW} "
                            f"times with low argument diversity ({arg_diversity:.0%}). "
                            f"You're tweaking arguments but the outcome isn't changing. "
                            f"Before calling it again, verify what the last attempt actually did.",
                            reason=f"same_tool_dominance: {ctx.tool_name}",
                        )

        # ── Detection 2: Semantic loop (read-only dominance) ──
        if len(self._tool_name_history) >= self._SEMANTIC_WINDOW:
            window = self._tool_name_history[-self._SEMANTIC_WINDOW:]
            read_count = sum(1 for t in window if t in _READ_ONLY_TOOL_NAMES)
            recent_entries = self._recent_tool_calls[-self._SEMANTIC_WINDOW:]

            # Count productive shells in the window
            productive_shells_in_window = sum(
                1 for name, args in recent_entries
                if name == "shell" and not self._is_read_only_shell_args(args)
            )
            # Count read-only shells
            read_only_shells = sum(
                1 for name, args in recent_entries
                if name == "shell" and self._is_read_only_shell_args(args)
            )
            effective_read_count = read_count + read_only_shells

            # Check for productive tools
            has_productive = any(
                t in ("write_file", "edit_file", "plan_create",
                      "plan_update", "workspace_experiment", "memory_write")
                for t in window
            )
            if productive_shells_in_window > 0:
                has_productive = True

            # Apply TaskMode to ratio threshold
            adjusted_ratio = min(0.95, self._SEMANTIC_READ_RATIO_BASE + (self._task_mode_multiplier - 1.0) * 0.1)

            ratio = effective_read_count / len(window)
            if ratio >= adjusted_ratio and not has_productive:
                # Diversity check using SharedState if available
                if self._shared_state:
                    diversity = self._shared_state.read_stats.diversity
                else:
                    unique_entries = set(recent_entries)
                    diversity = len(unique_entries) / len(recent_entries) if recent_entries else 1.0

                # High diversity = legitimate exploration
                # Also exempt if continuation-heavy (reading long files sequentially)
                is_continuation = (
                    self._shared_state and self._shared_state.read_stats.is_continuation_heavy
                )
                if diversity > 0.60 or is_continuation:
                    pass  # Not a loop
                else:
                    # Cooldown
                    calls_since_warn = self._total_tool_calls - self._semantic_warn_at
                    if not self._semantic_warned or calls_since_warn >= self._SEMANTIC_COOLDOWN:
                        # Check SharedState: suppress if another guard already warned
                        if self._shared_state and not self._shared_state.issue_read_warning():
                            pass  # Suppress
                        else:
                            self._semantic_warned = True
                            self._semantic_warn_at = self._total_tool_calls
                            self._semantic_warn_count += 1

                            if self._semantic_warn_count >= 2:
                                return GuardVerdict.escalate(
                                    f"[LoopDetect] You've been reading without acting for "
                                    f"{effective_read_count}/{len(window)} calls (diversity={diversity:.2f}). "
                                    "State your findings so far and what's blocking you from acting.",
                                    reason=f"semantic_read_persistent: {effective_read_count}/{len(window)}",
                                )

                            return GuardVerdict.inject(
                                f"[LoopDetect] {effective_read_count}/{len(window)} read-only calls "
                                f"(diversity={diversity:.2f}). "
                                "You're gathering information but not acting on it. "
                                "Ask yourself: do I have enough to move forward? "
                                "If yes — write, build, or fix something. "
                                "If no — what specific piece is missing?",
                                reason=f"semantic_read_only: {effective_read_count}/{len(window)}",
                            )

        # ── Detection 3: Retry pattern (kill→launch without diagnosis) ──
        verdict = self._check_retry_pattern(ctx)
        if verdict:
            return verdict

        return None

    def _is_continuation_read(self, tool_name: str, args: dict) -> bool:
        """Check if this is a continuation read (same file, different line range).

        Sequential reads of the same file (start_line > 1) are normal behavior
        when reading long files, not a loop.
        """
        if tool_name != "read_file":
            return False

        path = args.get("path", "")
        start_line = args.get("start_line")

        # Only counts as continuation if start_line > 1 (not re-reading from beginning)
        if not start_line or start_line <= 1:
            return False

        # Check if we recently read the same file at a different offset
        for prev_name, prev_args_str in reversed(self._recent_tool_calls[:-1]):
            if prev_name != "read_file":
                continue
            if f"path={path}" in prev_args_str:
                return True  # Same file was recently read — this is a continuation

        return False

    def _check_retry_pattern(self, ctx: GuardContext) -> GuardVerdict | None:
        """Detect kill→launch cycles without diagnostic steps."""
        if ctx.tool_name != "shell":
            return None

        cmd = ctx.tool_args.get("command", "").lower()
        recent = self._shell_cmd_history[-self._RETRY_WINDOW:]
        if len(recent) < 4:
            return None

        kill_count = sum(1 for c in recent if any(p in c for p in ("kill", "pkill", "killall")))
        launch_count = sum(
            1 for c in recent
            if any(p in c for p in ("torchrun", "python -m torch.distributed",
                                     "deepspeed", "flagscale", "train.py", "pretrain"))
        )
        diag_count = sum(
            1 for c in recent
            if any(p in c for p in ("grep", "cat", "tail", "nvidia-smi", "ps ",
                                     "dmesg", "journalctl"))
        )

        if kill_count >= 2 and launch_count >= 2 and diag_count == 0:
            return GuardVerdict.inject(
                "[LoopDetect] Kill→relaunch cycle detected without diagnostics. "
                "Before retrying, diagnose what went wrong: check logs, GPU state, "
                "or error output from the previous run.",
                reason="retry_without_diagnosis",
            )

        return None

    @staticmethod
    def _is_read_only_shell_args(args_str: str) -> bool:
        """Check if shell args string indicates a read-only command."""
        # args_str is the key_args string like "command=ls /some/dir"
        cmd_part = args_str.replace("command=", "", 1).strip().lower()
        return any(cmd_part.startswith(prefix.lower()) for prefix in _READ_ONLY_SHELL_PREFIXES)

    def notify_blocked(self, ctx: GuardContext):
        """Undo tracking when a tool call is externally blocked."""
        if not ctx.tool_name:
            return
        key_args = self._extract_key_args(ctx.tool_args)
        entry = (ctx.tool_name, key_args)
        if self._recent_tool_calls and self._recent_tool_calls[-1] == entry:
            self._recent_tool_calls.pop()
        if self._tool_name_history and self._tool_name_history[-1] == ctx.tool_name:
            self._tool_name_history.pop()
            self._total_tool_calls = max(0, self._total_tool_calls - 1)
        if ctx.tool_name == "shell" and self._shell_cmd_history:
            self._shell_cmd_history.pop()

    def reset_turn(self):
        # Clear per-iteration dedup cache, but keep history for cross-iteration detection
        self._tool_call_cache.clear()

    @staticmethod
    def _extract_key_args(args: dict) -> str:
        """Extract meaningful key arguments for dedup, skipping transient values."""
        skip_keys = {"timeout", "description", "run_in_background"}
        key_parts = []
        for k, v in sorted(args.items()):
            if k in skip_keys:
                continue
            val = str(v)[:80]
            key_parts.append(f"{k}={val}")
        return "|".join(key_parts)
