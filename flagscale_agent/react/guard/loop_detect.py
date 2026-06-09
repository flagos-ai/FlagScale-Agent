"""LoopDetectGuard — detects repeated/looping tool calls.

Uses two-phase detection:
1. Cheap trigger: counters/ratios/patterns exceed thresholds
2. Precise judgment: classify_fn("is_stuck_in_loop") confirms before escalation
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

    All detections use two-phase: cheap trigger → LLM confirmation before escalation.
    """

    name = "loop_detect"
    priority = 20
    activate_on_states = {AgentState.EXECUTING}

    _MAX_RECENT = 12
    _LOOP_THRESHOLD = 3

    # Semantic loop detection parameters
    _SEMANTIC_WINDOW = 12
    _SEMANTIC_READ_RATIO = 0.80  # 80% read-only triggers warning
    _SEMANTIC_COOLDOWN = 5  # turns before semantic warning can re-trigger

    # Retry pattern detection parameters
    _RETRY_WINDOW = 8
    _RETRY_KILL_THRESHOLD = 2  # 2 kill+launch cycles = retry loop

    def __init__(self):
        self._recent_tool_calls: list[tuple[str, str]] = []
        self._tool_call_cache: dict[tuple[str, str], str] = {}
        # Track full tool name history for semantic detection
        self._tool_name_history: list[str] = []
        # Track shell commands for retry pattern detection
        self._shell_cmd_history: list[str] = []
        self._retry_warned: bool = False
        # Semantic warning state: cooldown-based, not once-only
        self._semantic_warned: bool = False
        self._semantic_warn_count: int = 0  # total times warned
        # Monotonic counter for cooldown (not tied to capped list length)
        self._total_tool_calls: int = 0
        self._semantic_warn_at: int = 0  # _total_tool_calls value when last warned
        # Exact-match loop escalation
        self._exact_loop_inject_count: int = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        key_args = self._extract_key_args(ctx.tool_args)
        entry = (ctx.tool_name, key_args)

        self._recent_tool_calls.append(entry)
        if len(self._recent_tool_calls) > self._MAX_RECENT:
            self._recent_tool_calls = self._recent_tool_calls[-self._MAX_RECENT:]

        # Track tool names for semantic detection
        self._tool_name_history.append(ctx.tool_name)
        self._total_tool_calls += 1
        if len(self._tool_name_history) > self._SEMANTIC_WINDOW:
            self._tool_name_history = self._tool_name_history[-self._SEMANTIC_WINDOW:]

        # Track shell commands for retry pattern detection
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
            # Exact-match loops are unambiguous — skip LLM confirmation
            self._exact_loop_inject_count += 1
            # Escalate after 3 repeated warnings — abort the entire batch
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

        # ── Detection 1.5: Same-tool dominance (same tool, different args) ──
        # Catches: agent calling pip install with slightly different args each time
        if len(self._tool_name_history) >= self._SEMANTIC_WINDOW:
            window = self._tool_name_history[-self._SEMANTIC_WINDOW:]
            same_tool_count = sum(1 for t in window if t == ctx.tool_name)
            # If this tool dominates the window (>= 75%) and it's not a read-only tool
            # (read-only is handled by Detection 2), warn about repetitive behavior
            if (same_tool_count >= self._SEMANTIC_WINDOW * 0.75
                    and ctx.tool_name not in _READ_ONLY_TOOL_NAMES):
                # For shell: check arg diversity — different ls/find/cat/nvidia-smi
                # commands are legitimate exploration, not a loop
                if ctx.tool_name == "shell":
                    recent_shell_entries = [
                        args for name, args in self._recent_tool_calls[-self._SEMANTIC_WINDOW:]
                        if name == "shell"
                    ]
                    unique_shell = len(set(recent_shell_entries))
                    # High diversity (>50% unique commands) = exploration, not loop
                    if unique_shell > len(recent_shell_entries) * 0.5:
                        return None

                # Phase 2: LLM confirmation
                if not self._confirm_loop_with_llm(ctx, "same_tool_dominance",
                        f"'{ctx.tool_name}' called {same_tool_count}/{self._SEMANTIC_WINDOW} times"):
                    return None  # LLM says not a loop

                self._exact_loop_inject_count += 1
                if self._exact_loop_inject_count >= 3:
                    return GuardVerdict.escalate(
                        f"[LoopDetect] '{ctx.tool_name}' called {same_tool_count}/{self._SEMANTIC_WINDOW} "
                        f"times with varying args. You're trying variations but not getting different outcomes. "
                        "The tool itself isn't the problem — your expectation or approach is. "
                        "Step back: what are you actually trying to achieve?",
                        reason=f"same_tool_dominance_persistent: {ctx.tool_name}",
                    )
                return GuardVerdict.inject(
                    f"[LoopDetect] '{ctx.tool_name}' called {same_tool_count}/{self._SEMANTIC_WINDOW} times. "
                    f"You're tweaking arguments but the outcome isn't changing. "
                    f"Before calling it again, verify what the last attempt actually did.",
                    reason=f"same_tool_dominance: {ctx.tool_name}",
                )

        # ── Detection 2: Semantic loop (read-only dominance) ──
        if len(self._tool_name_history) >= self._SEMANTIC_WINDOW:
            window = self._tool_name_history[-self._SEMANTIC_WINDOW:]
            read_count = sum(1 for t in window if t in _READ_ONLY_TOOL_NAMES)
            # Count productive shells in the window by checking _recent_tool_calls
            recent_entries = self._recent_tool_calls[-self._SEMANTIC_WINDOW:]
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
            # Check for truly productive tools (shell excluded — handled separately)
            has_productive = any(
                t in ("write_file", "edit_file", "plan_create",
                      "plan_update", "workspace_experiment", "memory_write")
                for t in window
            )
            # Also count productive shells in window
            if productive_shells_in_window > 0:
                has_productive = True
            ratio = effective_read_count / len(window)
            if ratio >= self._SEMANTIC_READ_RATIO and not has_productive:
                # Diversity check: if the recent calls are all DIFFERENT targets,
                # the agent is exploring (not looping). Only trigger if low diversity.
                unique_entries = set(recent_entries)
                diversity = len(unique_entries) / len(recent_entries) if recent_entries else 1.0
                # High diversity (>60% unique) = legitimate exploration, don't trigger
                if diversity > 0.60:
                    pass  # Not a loop — agent is reading different things
                else:
                    # Cooldown: allow re-triggering after N calls since last warn
                    calls_since_warn = self._total_tool_calls - self._semantic_warn_at
                    if not self._semantic_warned or calls_since_warn >= self._SEMANTIC_COOLDOWN:
                        # Phase 2: LLM confirmation
                        if not self._confirm_loop_with_llm(ctx, "semantic_read_only",
                                f"{effective_read_count}/{len(window)} read-only, diversity={diversity:.2f}"):
                            pass  # LLM says not a loop — skip
                        else:
                            self._semantic_warned = True
                            self._semantic_warn_at = self._total_tool_calls
                            self._semantic_warn_count += 1

                            # Escalate on 2nd+ trigger — abort entire batch
                            if self._semantic_warn_count >= 2:
                                return GuardVerdict.escalate(
                                    f"[LoopDetect] You've been reading without acting for "
                                    f"{self._semantic_warn_count} consecutive windows. "
                                    "Forward progress isn't just gathering information — "
                                    "it's moving toward the goal. "
                                    "State what you've learned and what decision you need to make. "
                                    "Then either act or ask the user for direction.",
                                    reason="semantic_loop_persistent",
                                )

                            return GuardVerdict.inject(
                                f"[LoopDetect] {effective_read_count}/{len(window)} recent calls are read-only. "
                                "You're gathering information but not acting on it. "
                                "Ask yourself: do I have enough to move forward? "
                                "If yes — write, build, or fix something. "
                                "If no — what specific piece is missing?",
                                reason=f"semantic loop: {effective_read_count}/{len(window)} read-only",
                            )

        # ── Detection 3: Retry pattern (kill→launch cycles) ──
        if ctx.tool_name == "shell" and not self._retry_warned:
            verdict = self._check_retry_pattern(ctx)
            if verdict:
                return verdict

        return None

    def _confirm_loop_with_llm(self, ctx: GuardContext, detection_type: str,
                                evidence: str) -> bool:
        """Phase 2: Ask LLM if this is actually a loop.

        Returns True if confirmed as loop, False if overridden (not a loop).
        """
        print(display.dim(f"  🔍 [loop_detect] triggered: {detection_type}"))

        if not ctx.classify_fn:
            # Conservative: no LLM = assume loop
            print(display.yellow(f"     ⚠  [loop_detect] no judge — assuming loop"))
            return True

        # Build context from recent history
        recent_history = []
        for name, args in self._recent_tool_calls[-6:]:
            recent_history.append(f"{name}({args[:60]})")

        result, source = get_judge_result(
            ctx.classify_fn, "is_stuck_in_loop",
            {
                "detection_type": detection_type,
                "evidence": evidence,
                "recent_tool_history": "; ".join(recent_history),
            },
            default=True
        )
        if is_trusted(source) and not result:
            print(display.dim(f"     ✓  [loop_detect] override: not a loop (legitimate exploration)"))
            return False  # LLM says not a loop
        print(display.yellow(f"     ⚠  [loop_detect] confirmed: agent is looping"))
        return True

    def _is_productive_shell(self, cmd: str) -> bool:
        """Check if a shell command is productive (modifies state).

        Read-only commands (ls, find, cat, etc.) are not productive.
        Commands that install, build, write, or modify state are productive.
        """
        cmd_stripped = cmd.strip().lower()
        if not cmd_stripped:
            return False
        if any(cmd_stripped.startswith(p) for p in _READ_ONLY_SHELL_PREFIXES):
            return False
        return True

    def _is_read_only_shell_args(self, key_args: str) -> bool:
        """Check if a shell's key_args string represents a read-only command.

        key_args format from _extract_key_args: "command=ls /foo|other=bar"
        Extract the command value and check against read-only prefixes.
        """
        # Extract command value from key_args format
        cmd = ""
        for part in key_args.split("|"):
            if part.startswith("command="):
                cmd = part[len("command="):].strip().lower()
                break
        if not cmd:
            return True  # no command = treat as read-only
        return any(cmd.startswith(p) for p in _READ_ONLY_SHELL_PREFIXES)

    def _check_retry_pattern(self, ctx: GuardContext) -> GuardVerdict | None:
        """Detect kill→launch retry loops without diagnostic steps."""
        if len(self._shell_cmd_history) < 4:
            return None

        window = self._shell_cmd_history[-self._RETRY_WINDOW:]
        kill_count = sum(1 for cmd in window if any(k in cmd for k in ("kill", "pkill", "killall")))
        launch_count = sum(
            1 for cmd in window
            if any(k in cmd for k in ("torchrun", "deepspeed", "flagscale", "train.py", "pretrain"))
        )

        # Detect: 2+ kills AND 2+ launches in the window = retry loop
        if kill_count >= self._RETRY_KILL_THRESHOLD and launch_count >= self._RETRY_KILL_THRESHOLD:
            # Check if there were diagnostic steps (read_file, grep, etc.) between cycles
            recent_tools = self._tool_name_history[-self._RETRY_WINDOW:]
            diagnostic_count = sum(
                1 for t in recent_tools
                if t in ("read_file", "grep", "find_latest_log")
            )
            if diagnostic_count < 2:
                # Phase 2: LLM confirmation for retry pattern
                if not self._confirm_loop_with_llm(ctx, "retry_pattern",
                        f"{kill_count} kills + {launch_count} launches without diagnostics"):
                    return None

                self._retry_warned = True
                return GuardVerdict.inject(
                    f"[LoopDetect] RETRY LOOP DETECTED: {kill_count} kills + {launch_count} launches "
                    "in recent history without diagnostic steps between them.\n\n"
                    "STOP the kill→restart cycle. Instead:\n"
                    "1. Read the error log/output from the LAST failed attempt\n"
                    "2. Form a hypothesis about WHY it's failing\n"
                    "3. Verify the hypothesis (read source code, check versions, etc.)\n"
                    "4. Fix the root cause BEFORE relaunching\n\n"
                    "Common causes: wrong config path, missing env var, version mismatch, "
                    "port conflict, wandb/network issue.",
                    reason=f"retry_loop: {kill_count} kills + {launch_count} launches",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name:
            # Cache result for dedup detection
            key_args = self._extract_key_args(ctx.tool_args)
            if ctx.tool_result:
                self._tool_call_cache[(ctx.tool_name, key_args)] = ctx.tool_result
            # Reset semantic warning after a truly productive action
            is_productive = ctx.tool_name in (
                "write_file", "edit_file",
                "plan_create", "plan_update", "memory_write",
            )
            # Shell is only productive if it modifies state
            if ctx.tool_name == "shell":
                cmd = ctx.tool_args.get("command", "").lower()
                is_productive = self._is_productive_shell(cmd)
            if is_productive:
                self._semantic_warned = False
                self._semantic_warn_count = 0
                self._exact_loop_inject_count = 0
            # Reset retry warning after diagnostic action
            if ctx.tool_name in ("read_file", "grep", "find_latest_log"):
                self._retry_warned = False
        return None

    def notify_blocked(self, ctx: GuardContext):
        """Remove the last entry from history when a call is blocked by another guard.

        This prevents cascading failures where:
        1. LoopDetect passes and adds entry to history
        2. Another guard (e.g., ConstraintGuard) blocks the call
        3. Agent retries → LoopDetect sees "repeated" call → escalates
        """
        if not ctx.tool_name:
            return
        key_args = self._extract_key_args(ctx.tool_args)
        entry = (ctx.tool_name, key_args)
        # Remove from recent_tool_calls (last occurrence)
        if self._recent_tool_calls and self._recent_tool_calls[-1] == entry:
            self._recent_tool_calls.pop()
        # Remove from tool_name_history
        if self._tool_name_history and self._tool_name_history[-1] == ctx.tool_name:
            self._tool_name_history.pop()
            self._total_tool_calls = max(0, self._total_tool_calls - 1)
        # Remove from shell_cmd_history
        if ctx.tool_name == "shell" and self._shell_cmd_history:
            self._shell_cmd_history.pop()

    def reset_turn(self):
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
