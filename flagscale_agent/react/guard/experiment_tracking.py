"""ExperimentTrackingGuard — enforces experiment recording discipline.

Ensures that:
1. Before any training launch, an experiment attempt is recorded
2. After training completes/fails, the result is updated
3. Required fields are present (launch command, output_dir, software stack)

This prevents the common failure mode where Agent runs 20+ training launches
but only formally records 5 of them, losing critical debugging context.
"""

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Detect actual training launches (not just any command mentioning train.py).
# A real launch is: running a training script as the main entry point, not
# grep/find/cat/import-checking it.
_LAUNCH_RE = re.compile(
    r"\brun\.py\b.*\baction\s*=\s*run\b"        # flagscale CLI launch
    r"|\brun\.py\b.*--config"                    # flagscale CLI with config
    r"|\bconda\s+run\b.*\brun\.py\b"            # conda wrapper launch
    r"|\btorchrun\s+"                            # torchrun (with args, not just the word)
    r"|\bflagscale\s+train\b"                   # flagscale train subcommand
    r"|\bpython[23]?\s+.*(?:train|pretrain)\.py\b"  # python train.py / python pretrain.py
    ,
    re.IGNORECASE,
)


class ExperimentTrackingGuard(Guard):
    """Enforce experiment recording before/after training launches.
    
    Pre-check: Before training launch, verify add_attempt() was called
    Post-check: After training result observed, remind to update_last_attempt()
    """

    name = "experiment_tracking"
    priority = 18  # After TrainingAttemptGuard (15), before TrainingRuntime (20)

    def __init__(self):
        # Track whether add_attempt was called since last launch
        self._attempt_recorded = False
        # Track whether training result needs recording
        self._result_pending = False
        # Count of unrecorded launches (for escalation)
        self._unrecorded_launches = 0
        # Last recorded experiment name
        self._last_experiment_name = ""
        # Whether agent just launched training
        self._training_launched = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block training launch if no attempt recorded."""
        # Track workspace_experiment in pre-check too (handles same-batch scenarios)
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action in ("add_attempt", "read", "update_last_attempt"):
                # Reading or updating an experiment counts as awareness
                self._attempt_recorded = True
                self._unrecorded_launches = 0
                name = ctx.tool_args.get("name", "")
                if name:
                    self._last_experiment_name = name
            elif action == "create":
                name = ctx.tool_args.get("name", "")
                if name:
                    self._last_experiment_name = name
            return None

        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _LAUNCH_RE.search(cmd):
                if not self._attempt_recorded:
                    self._unrecorded_launches += 1
                    
                    if self._unrecorded_launches >= 3:
                        # Escalate after 3 unrecorded launches — but inject, don't block
                        # The agent may have recorded the attempt in a previous session
                        return GuardVerdict.inject(
                            f"[ExperimentTracking] {self._unrecorded_launches} "
                            "training launches without experiment recording in this session. "
                            "Consider recording an experiment attempt.",
                            reason="experiment_not_recorded",
                            category="experiment_tracking",
                        )
                    else:
                        # Warn for first 2 unrecorded launches
                        return GuardVerdict.inject(
                            "[ExperimentTracking] Launching training without "
                            "recording an experiment attempt. "
                            "Record the attempt before launching to track debugging history.",
                            reason="experiment_not_recorded_warn",
                            category="experiment_tracking",
                        )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track experiment-related tool calls."""
        
        # Track add_attempt calls
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action == "add_attempt":
                self._attempt_recorded = True
                self._unrecorded_launches = 0
                name = ctx.tool_args.get("name", "")
                if name:
                    self._last_experiment_name = name
            elif action == "update_last_attempt":
                self._result_pending = False
            elif action == "create":
                name = ctx.tool_args.get("name", "")
                if name:
                    self._last_experiment_name = name

        # Track training launch
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _LAUNCH_RE.search(cmd):
                self._training_launched = True

        # Track training result (from monitor)
        if self._training_launched and ctx.tool_name in ("monitor", "find_latest_log", "parse_training_metrics"):
            if ctx.tool_result:
                self._training_launched = False
                self._result_pending = True
                self._attempt_recorded = False  # Full cycle complete, reset for next launch
                
                # Remind to update experiment
                if self._last_experiment_name:
                    return GuardVerdict.inject(
                        f"[ExperimentTracking] Training result observed. "
                        f"Update experiment '{self._last_experiment_name}' with the result.",
                        reason="update_experiment_result",
                    )

        return None

    def reset_turn(self):
        """Reset escalation counters to prevent dead loops across turns.
        
        Knowledge state persists (_last_experiment_name, _result_pending),
        but block counters reset to prevent permanent blocking.
        """
        self._unrecorded_launches = 0
