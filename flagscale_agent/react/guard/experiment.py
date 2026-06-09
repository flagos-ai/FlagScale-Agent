"""ExperimentGuard — enforces experiment lifecycle before launching runs.

Uses two-phase detection:
1. Cheap trigger: keyword scan for launch-related terms
2. Precise judgment: classify_fn("is_training_command") to confirm
"""

import re

from flagscale_agent.react import display
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import get_judge_result, is_trusted
from flagscale_agent.react.state_machine import AgentState


# Shell commands that indicate a training/inference launch.
# These are actual launch patterns, not just substrings that appear in paths.
# Each entry is a regex pattern matched against the command.
_LAUNCH_PATTERNS = (
    r"\btorchrun\b",
    r"\bpython\s+-m\s+torch\.distributed\b",
    r"\bdeepspeed\b",
    r"\bflagscale\s+train\b",
    r"\bflagscale\s+serve\b",
    r"\bmegatron\b.*\btrain\b",
    r"\btrain\.py\b",
    r"\bpretrain\.py\b",
    r"\bpretrain\s",
    r"\bsglang\s+serve\b",
    r"\bvllm\s+serve\b",
    r"\binference\.py\b",
    r"\bpython\b.*\bserve\b",
)

# Precompile for performance
_LAUNCH_RE = re.compile("|".join(_LAUNCH_PATTERNS), re.IGNORECASE)


class ExperimentGuard(Guard):
    """Enforces workspace_experiment lifecycle before launching runs.

    Blocks shell commands that look like training/inference launches
    unless an experiment has been created and an attempt has been added.

    Two-phase detection:
    1. Cheap trigger: _LAUNCH_KEYWORDS in command
    2. LLM confirm: classify_fn("is_training_command") eliminates false positives
    """

    name = "experiment_lifecycle"
    priority = 45  # Run before PlanUpdateGuard
    activate_on_states = {AgentState.EXECUTING}

    def __init__(self, experiment_manager):
        self._experiment_manager = experiment_manager
        self._experiment_created = False
        self._attempt_added = False
        self._inject_count: int = 0  # track repeated injects for escalation

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block launch commands if experiment lifecycle not followed."""
        # Track workspace_experiment calls
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action == "create":
                self._experiment_created = True
                self._inject_count = 0  # reset: agent complied
            elif action == "add_attempt":
                self._attempt_added = True
                self._inject_count = 0  # reset: agent complied
            elif action == "update_last_attempt":
                # Reset attempt flag so next launch requires a new attempt
                self._attempt_added = False
            return None

        # Only check shell commands
        if ctx.tool_name != "shell":
            return None

        cmd = ctx.tool_args.get("command", "")
        cmd_lower = cmd.lower()

        # Phase 1: Cheap trigger — quick keyword scan
        if not self._cheap_trigger(cmd_lower):
            return None

        print(display.dim(f"  🔍 [experiment] triggered: launch keyword in command"))

        # Phase 2: LLM precise judgment
        if ctx.classify_fn:
            is_launch, source = get_judge_result(
                ctx.classify_fn, "is_training_command",
                {"command": cmd}, default=False
            )
            if is_trusted(source) and not is_launch:
                print(display.dim(f"     ✓  [experiment] override: not a training launch"))
                return None  # LLM says not a launch — allow
            if is_trusted(source) and is_launch:
                print(display.yellow(f"     ⚠  [experiment] confirmed: training launch detected"))
        # If classify_fn unavailable, fall through to enforcement (conservative)

        # Enforce experiment lifecycle
        if not self._experiment_created:
            self._inject_count += 1
            if self._inject_count >= 3:
                return GuardVerdict.escalate(
                    "⚠️ EXPERIMENT REQUIRED: Warned {0} times but no experiment created. "
                    "You MUST call workspace_experiment(action='create', ...) "
                    "BEFORE launching. STOP and comply.".format(self._inject_count),
                    reason="experiment_not_created_persistent"
                )
            return GuardVerdict.block(
                "⚠️ EXPERIMENT REQUIRED: You are about to launch a training/inference run, "
                "but no experiment has been created. Call:\n"
                "  workspace_experiment(action='create', name=..., purpose=..., hypothesis=...)\n"
                "THEN:\n"
                "  workspace_experiment(action='add_attempt', name=..., change=..., "
                "hardware={...}, config={...}, output_dir=...)\n"
                "BEFORE launching.",
                reason="experiment_not_created"
            )

        if not self._attempt_added:
            self._inject_count += 1
            if self._inject_count >= 3:
                return GuardVerdict.escalate(
                    "⚠️ ATTEMPT REQUIRED: Warned {0} times but no attempt added. "
                    "You MUST call workspace_experiment(action='add_attempt', ...) "
                    "BEFORE launching. STOP and comply.".format(self._inject_count),
                    reason="attempt_not_added_persistent"
                )
            return GuardVerdict.block(
                "⚠️ ATTEMPT REQUIRED: Experiment exists but no attempt has been added. Call:\n"
                "  workspace_experiment(action='add_attempt', name=..., change=..., "
                "hardware={...}, config={...}, output_dir=...)\n"
                "BEFORE launching. This records what you're about to try.",
                reason="attempt_not_added"
            )

        return None

    @staticmethod
    def _cheap_trigger(cmd_lower: str) -> bool:
        """Phase 1: regex check for launch patterns. May have false positives."""
        return bool(_LAUNCH_RE.search(cmd_lower))
