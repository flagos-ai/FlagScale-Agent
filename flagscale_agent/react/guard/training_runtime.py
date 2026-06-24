"""TrainingRuntimeGuard — monitor enforcement, hang/kill-retry/zombie detection,
auto-restart strategy, and multi-node health check reminders.

Uses two-phase detection: cheap keyword trigger + LLM classify() for confirmation.
"""

from __future__ import annotations

import re
import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


# Cheap trigger patterns for training launch detection.
# Only commands matching these get sent to LLM for confirmation.
# NOTE: serve/inference patterns are NOT here — this guard only manages
# training lifecycle (monitor enforcement, failure tracking, restart strategy).
_LAUNCH_TRIGGER_PATTERNS = (
    r"\btorchrun\b",
    r"\bpython\s+-m\s+torch\.distributed\b",
    r"\bdeepspeed\b",
    r"\bflagscale\s+train\b",
    r"\bmegatron\b.*\btrain\b",
    r"\btrain\.py\b",
    r"\bpretrain\.py\b",
    r"\bpretrain\s",
    # FlagScale launcher patterns (run.py with Hydra config)
    r"\brun\.py\b.*\baction\s*=\s*run\b",
    r"\brun\.py\b.*--config",
    # conda run wrapping any training command
    r"\bconda\s+run\b.*\brun\.py\b",
    r"\bconda\s+run\b.*\btorchrun\b",
    r"\bconda\s+run\b.*\bflagscale\b",
)
_LAUNCH_TRIGGER_RE = re.compile("|".join(_LAUNCH_TRIGGER_PATTERNS), re.IGNORECASE)

# Patterns that indicate training failure in monitor() output
_MONITOR_FAILURE_PATTERNS = (
    r"TRAINING CRASHED",
    r"Process\s+died",
    r"Traceback\s+\(most\s+recent",
    r"RuntimeError:",
    r"NCCL\s+error",
    r"Out\s+of\s+memory",
    r"CUDA\s+error",
    r"AttributeError:",
    r"ModuleNotFoundError:",
    r"all\s+\d+\s+rank.*stderr.*error",
    r"exit\s+code\s+[1-9]",
)
_MONITOR_FAILURE_RE = re.compile("|".join(_MONITOR_FAILURE_PATTERNS), re.IGNORECASE)


# Auto-restart config templates
_AUTO_RESTART_STRATEGIES = {
    "oom": [
        ("global_batch_size", "reduce by 50%", "halve"),
        ("gradient_accumulation_steps", "increase to compensate", "double_gas"),
        ("recompute_activations", "true", "enable_recompute"),
    ],
    "nccl": [
        ("NCCL_IB_DISABLE", "1", "disable_ib"),
        ("NCCL_SOCKET_IFNAME", "eth0", "set_nic"),
        ("NCCL_DEBUG", "INFO", "enable_nccl_debug"),
    ],
    "cuda": [
        ("precision", "bf16 -> fp16", "downgrade_precision"),
        ("deterministic_mode", "false", "disable_deterministic"),
    ],
    "default": [
        ("global_batch_size", "reduce by 50%", "halve_batch"),
        ("tp", "reduce if >1", "reduce_tp"),
        ("gradient_checkpointing", "true", "enable_ckpt"),
    ],
}


class TrainingRuntimeGuard(Guard):
    """Training lifecycle management and monitoring enforcement.

    Only activates for training scenes (registered conditionally).
    """

    name = "training_runtime"
    priority = 50
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    # Thresholds
    _MONITOR_GATE_MAX_BLOCKS = 5
    _KILL_RETRY_WINDOW = 120  # seconds
    _KILL_RETRY_MAX = 3
    _MIN_SOURCE_READS_BEFORE_FIX = 2
    _HEARTBEAT_MONITOR_INTERVAL = 3
    _HEARTBEAT_GPU_CHECK_INTERVAL = 5

    def __init__(self):
        self._awaiting_monitor: bool = False
        self._monitor_gate_block_count: int = 0
        self._consecutive_train_failures: int = 0
        self._last_train_failure_reasons: list[str] = []
        self._kill_retry_timestamps: list[float] = []
        self._training_launch_timestamps: list[float] = []
        self._source_reads_since_last_failure: int = 0
        self._training_started: bool = False
        self._turns_since_last_monitor: int = 0
        self._turns_since_last_gpu_check: int = 0
        self._last_launch_output_dir: str = ""
        self._multi_node_warned: bool = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Monitor enforcement
        if self._awaiting_monitor:
            if ctx.tool_name == "monitor":
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None
            if ctx.tool_name in ("plan_update", "workspace_experiment", "read_file"):
                return None

            # Allow read-only diagnostic shell commands
            if ctx.tool_name == "shell":
                cmd = ctx.tool_args.get("command", "").strip()
                # Fast-path: common read-only prefixes don't need LLM
                if self._is_read_only_shell_fast(cmd):
                    return None
                classify = ctx.classify_fn
                if classify and classify("is_read_only_shell", {"command": cmd}, default=True):
                    return None

            self._monitor_gate_block_count += 1
            if self._monitor_gate_block_count >= self._MONITOR_GATE_MAX_BLOCKS:
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None

            return GuardVerdict.block(
                "[TrainingRuntime] BLOCKED: Monitor training before doing other work. "
                "Read-only commands (pgrep, ps, cat, ls) are allowed.",
                reason="monitor required after train launch",
            )

        # Source reading gate
        if self._consecutive_train_failures >= 2:
            if ctx.tool_name in ("write_file", "edit_file"):
                if self._source_reads_since_last_failure < self._MIN_SOURCE_READS_BEFORE_FIX:
                    target = ctx.tool_args.get("path", "") or ctx.tool_args.get("file_path", "")
                    if not target or any(ext in target for ext in (".yaml", ".yml", ".md", ".txt", ".json")):
                        return None
                    return GuardVerdict.inject(
                        f"[TrainingRuntime] {self._consecutive_train_failures} consecutive failures, "
                        f"only {self._source_reads_since_last_failure} source reads. "
                        f"Read the upstream framework code to understand what it expects before fixing.",
                        reason="source reading required before fix",
                    )

        # Heartbeat: periodic GPU check reminder
        if self._training_started and not self._awaiting_monitor:
            gpu_overdue = self._turns_since_last_gpu_check >= self._HEARTBEAT_GPU_CHECK_INTERVAL
            monitor_overdue = self._turns_since_last_monitor >= self._HEARTBEAT_MONITOR_INTERVAL

            if gpu_overdue and monitor_overdue:
                self._turns_since_last_gpu_check = 0
                self._turns_since_last_monitor = 0
                return GuardVerdict.inject(
                    "[HEARTBEAT] Training running but not monitored. Check GPU utilization and training progress.",
                    reason="periodic gpu health check + monitor overdue",
                )

            if gpu_overdue:
                self._turns_since_last_gpu_check = 0
                return GuardVerdict.inject(
                    "[HEARTBEAT] Check GPU utilization — if 0% with active process, training may be hung.",
                    reason="periodic gpu health check",
                )

            if monitor_overdue:
                self._turns_since_last_monitor = 0
                if self._last_launch_output_dir:
                    return GuardVerdict.inject(
                        "[HEARTBEAT] Monitor training progress — no observation in "
                        f"{self._HEARTBEAT_MONITOR_INTERVAL} turns.",
                        reason="monitor overdue",
                    )
                else:
                    return GuardVerdict.inject(
                        "[HEARTBEAT] Training running but unobserved for "
                        f"{self._HEARTBEAT_MONITOR_INTERVAL} turns. Check progress.",
                        reason="monitor overdue",
                    )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        classify = ctx.classify_fn

        # Track monitor calls for heartbeat
        if ctx.tool_name == "monitor":
            self._turns_since_last_monitor = 0
            self._turns_since_last_gpu_check = 0

        # Detect training launch (two-phase: cheap trigger + LLM confirm)
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            # Phase 1: cheap regex trigger — skip LLM call for non-launch commands
            if _LAUNCH_TRIGGER_RE.search(cmd):
                # Phase 2: LLM confirmation
                if classify and classify("is_training_command", {"command": cmd}, default=False):
                    self._training_launch_timestamps.append(time.time())
                    self._awaiting_monitor = True
                    self._training_started = True
                    self._turns_since_last_monitor = 0
                    self._turns_since_last_gpu_check = 0
                    # Extract output_dir
                    output_dir = ctx.tool_args.get("output_dir", "")
                    if not output_dir:
                        m = re.search(r'--output[_-]dir\s+(\S+)', cmd, re.IGNORECASE)
                        if m:
                            output_dir = m.group(1).strip('\'"')
                    self._last_launch_output_dir = output_dir

                    # Multi-node health check
                    multi_node_msg = self._check_multi_node_setup(ctx)
                    if multi_node_msg and not self._multi_node_warned:
                        self._multi_node_warned = True
                        return GuardVerdict.inject(multi_node_msg, reason="multi-node health check reminder")

            # Detect kill commands (only if training is active)
            if self._training_started and re.search(r'\b(kill|pkill|killall)\b', cmd):
                if classify and classify("is_kill_command", {"command": cmd}, default=False):
                    self._kill_retry_timestamps.append(time.time())
                    cutoff = time.time() - self._KILL_RETRY_WINDOW
                    self._kill_retry_timestamps = [t for t in self._kill_retry_timestamps if t > cutoff]
                    if len(self._kill_retry_timestamps) >= self._KILL_RETRY_MAX:
                        return GuardVerdict.inject(
                            "[TrainingRuntime] Kill-retry loop detected — "
                            f"{len(self._kill_retry_timestamps)} kill commands in "
                            f"{self._KILL_RETRY_WINDOW}s. "
                            "Diagnose the root cause before restarting.",
                            reason="kill-retry loop",
                        )

        # Track training failures (from shell OR monitor results)
        _failure_detected = False
        _failure_result = ""
        
        if self._training_started and ctx.tool_result:
            if ctx.tool_name == "shell":
                if classify and classify("is_training_failure", {
                    "command": ctx.tool_args.get("command", ""),
                    "result": ctx.tool_result,
                }, default=False):
                    _failure_detected = True
                    _failure_result = ctx.tool_result
            elif ctx.tool_name in ("monitor", "find_latest_log", "parse_training_metrics"):
                # Check monitor/log tool output for failure patterns
                if _MONITOR_FAILURE_RE.search(ctx.tool_result):
                    _failure_detected = True
                    _failure_result = ctx.tool_result

        if _failure_detected:
            self._consecutive_train_failures += 1
            self._last_train_failure_reasons.append(_failure_result[-300:])
            self._source_reads_since_last_failure = 0

            failure_lower = _failure_result.lower()
            strategy = _AUTO_RESTART_STRATEGIES["default"]
            if "oom" in failure_lower or "out of memory" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["oom"]
            elif "nccl" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["nccl"]
            elif "cuda" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["cuda"]

            strategy_lines = "\n".join(f"  - {k}: {desc}" for k, desc, _ in strategy)
            restart_msg = (
                f"\n[AUTO-RESTART STRATEGY] Detected failure category, "
                f"suggested config modifications before next attempt:\n"
                f"{strategy_lines}\n"
                "Apply fixes before next attempt."
            )

            compare_msg = ""
            if ctx.current_experiment_name and ctx.experiment_diff_fn:
                try:
                    diff_result = ctx.experiment_diff_fn(ctx.current_experiment_name)
                    if diff_result.get("diffs"):
                        compare_msg = (
                            f"\n\n[AUTO-COMPARE] Config diffs between last two attempts "
                            f"of '{ctx.current_experiment_name}':\n"
                            f"{diff_result['summary']}\n"
                            "Review which config change likely caused this failure."
                        )
                except Exception:
                    pass

            if self._consecutive_train_failures >= 3:
                return GuardVerdict.escalate(
                    f"[TrainingRuntime] {self._consecutive_train_failures} "
                    "consecutive training failures. The current configuration "
                    "will not succeed without changes. Diagnose root cause "
                    f"before retrying.{restart_msg}{compare_msg}",
                    reason="consecutive training failures",
                )
            elif compare_msg:
                return GuardVerdict.inject(
                    compare_msg.strip() + restart_msg,
                    reason="config diff and restart strategy after failure",
                )
        else:
            # Track source code reading (not a failure)
            if self._training_started and ctx.tool_name == "read_file" and self._consecutive_train_failures > 0:
                path = ctx.tool_args.get("path", "") or ctx.tool_args.get("file_path", "")
                if path and path.endswith(".py"):
                    self._source_reads_since_last_failure += 1

        # GPU zombie detection
        if ctx.tool_name == "shell" and ctx.tool_result:
            cmd = ctx.tool_args.get("command", "")
            if classify and classify("is_zombie_gpu", {
                "command": cmd,
                "result": ctx.tool_result,
            }, default=False):
                return GuardVerdict.inject(
                    "[TrainingRuntime] Possible zombie GPU processes detected. "
                    "Clean them up before launching new training.",
                    reason="gpu zombie process detected",
                )

        return None

    def reset_turn(self):
        """Increment heartbeat counters once per iteration (not per tool call)."""
        if self._training_started:
            self._turns_since_last_monitor += 1
            self._turns_since_last_gpu_check += 1

    @staticmethod
    def _is_read_only_shell_fast(cmd: str) -> bool:
        """Fast-path check for read-only shell commands (no LLM needed).

        Covers common diagnostic commands that should never be blocked by monitor gate.
        """
        cmd_lower = cmd.lower().strip()
        # Handle piped commands: check the first command in the pipe
        first_cmd = cmd_lower.split("|")[0].strip()
        # Handle command chains: check the first command
        for sep in ("&&", ";"):
            if sep in first_cmd:
                first_cmd = first_cmd.split(sep)[0].strip()

        _PREFIXES = (
            "ls", "find ", "cat ", "head ", "tail ", "grep ", "wc ",
            "which ", "echo ", "pwd", "env ", "printenv",
            "nvidia-smi", "nvcc ", "ps ", "pgrep ", "top ",
            "df ", "du ", "free ", "uname ", "whoami", "hostname", "date",
            "python --version", "python -c \"import", "python -c 'import",
            "python3 --version", "python3 -c \"import", "python3 -c 'import",
            "timeout ", "conda info", "conda list", "pip list", "pip show",
        )
        return any(first_cmd.startswith(p) for p in _PREFIXES)

    @staticmethod
    def _check_multi_node_setup(ctx: GuardContext) -> str | None:
        """Detect multi-node config and generate health check instructions."""
        cmd = ctx.tool_args.get("command", "")
        config = ctx.tool_args.get("config", {})
        args = ctx.tool_args.get("args", {})

        is_multi_node = False
        indicators = []

        if "nnodes" in cmd or "node_rank" in cmd or "hostfile" in cmd:
            is_multi_node = True
            indicators.append("command line contains multi-node args")

        nnodes = config.get("nnodes") or args.get("nnodes")
        if nnodes and int(nnodes) > 1:
            is_multi_node = True
            indicators.append(f"nnodes={nnodes}")

        tp = config.get("tp") or args.get("tp") or 1
        dp = config.get("dp") or args.get("dp") or 1
        if int(tp) * int(dp) > 8:
            is_multi_node = True
            indicators.append(f"tp={tp}, dp={dp} ({(int(tp) * int(dp))} GPUs)")

        if not is_multi_node:
            return None

        return (
            "[MULTI-NODE] Multi-node training detected "
            f"({' ; '.join(indicators)}). "
            "Verify inter-node connectivity (NCCL allreduce, SSH, shared storage) before launching."
        )
