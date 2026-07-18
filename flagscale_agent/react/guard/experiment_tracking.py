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

"""ExperimentTrackingGuard — enforces experiment recording discipline.

Ensures that:
1. Before any training launch, an experiment attempt is recorded
2. After training completes/fails, the result is updated
3. Required fields are present (launch command, output_dir, software stack)

v3 Lifecycle:
- Satisfied when: add_attempt is called (concern resolved)
- Decay: after 10 idle iterations without a launch, state resets
- Override: always overridable (LLM may have valid reasons to skip recording)
- Inject suppression: after 2 inject warnings, stops injecting to avoid context pollution

This prevents the common failure mode where Agent runs 20+ training launches
but only formally records 5 of them, losing critical debugging context.
"""

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Launch patterns — STRICT: must look like an actual training launch command,
# not just a command that happens to contain "train.py" in a path or grep.
# The pattern checks for actual executable invocations at command boundaries.
_LAUNCH_RE = re.compile(
    r"(?:^|\s|&&|\|\||;)\s*(?:"
    # FlagScale run.py with action=run
    r"(?:python[23]?(?:\.\d+)?|conda\s+run\b.*?)\s+\S*\brun\.py\b.*\baction\s*=\s*run\b"
    # torchrun (always a launch)
    r"|torchrun\s+"
    # flagscale train command
    r"|flagscale\s+train\b"
    # Direct python invocation of train/pretrain scripts
    # Must be python <script>.py pattern, not python -m pytest etc.
    r"|(?:python[23]?(?:\.\d+)?)\s+(?!-m\s)(?:\S+/)?(?:train|pretrain)\.py\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Commands that merely REFERENCE training scripts (should NOT trigger)
_FALSE_POSITIVE_RE = re.compile(
    r"^\s*(?:"
    r"grep|find|cat|head|tail|less|wc|ls|file|stat|diff|vim|nano|echo"
    r"|python[23]?\s+-m\s+pytest"  # pytest runs are NOT training launches
    r")\s",
    re.IGNORECASE,
)


class ExperimentTrackingGuard(Guard):
    """Enforce experiment recording before/after training launches.

    Pre-check: Before training launch, verify add_attempt() was called
    Post-check: After training result observed, remind to update_last_attempt()

    v3: Implements full lifecycle (is_satisfied, reset_state, inject suppression).
    """

    name = "experiment_tracking"
    priority = 18  # After TrainingAttemptGuard (15), before TrainingRuntime (20)
    overridable = True  # LLM can always bypass with a reason

    # v3: Inject at most 2 times before going silent
    max_inject_repeats = 2
    # v3: Decay after 10 idle iterations (no launch detected)
    decay_after_idle = 10

    def __init__(self):
        super().__init__()
        self._reset_internal_state()

    def _reset_internal_state(self):
        """Reset all guard-specific state to initial values."""
        # Track whether add_attempt was called since last launch
        self._attempt_recorded = False
        # Track whether training result needs recording
        self._result_pending = False
        # Count of consecutive unrecorded launches (resets on add_attempt or decay)
        self._unrecorded_launches = 0
        # Last recorded experiment name
        self._last_experiment_name = ""
        # Whether agent just launched training
        self._training_launched = False

    def reset_state(self):
        """v3: Full state reset (called on satisfaction, decay, or override)."""
        super().reset_state()
        self._reset_internal_state()

    def is_satisfied(self, ctx: GuardContext) -> bool:
        """v3: Guard is satisfied when attempt has been recorded and no pending results."""
        return self._attempt_recorded and not self._result_pending

    def _is_actual_launch(self, cmd: str) -> bool:
        """Determine if a shell command is an actual training launch (not a reference)."""
        # First check: does it match the false positive pattern?
        if _FALSE_POSITIVE_RE.search(cmd):
            return False
        # Second check: does it match the actual launch pattern?
        return bool(_LAUNCH_RE.search(cmd))

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block training launch if no attempt recorded."""
        # Track workspace_experiment in pre-check too (handles same-batch scenarios)
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action == "add_attempt":
                self._attempt_recorded = True
                self._unrecorded_launches = 0  # Reset counter on recording
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
            if self._is_actual_launch(cmd):
                if not self._attempt_recorded:
                    self._unrecorded_launches += 1
                    self._record_trigger("launch_without_recording")

                    if self._unrecorded_launches >= 3:
                        # Block after 3 consecutive unrecorded launches
                        # But this is overridable — LLM can bypass with a reason
                        return GuardVerdict.block(
                            f"[ExperimentTracking] BLOCKED: {self._unrecorded_launches} "
                            "consecutive training launches without experiment recording. "
                            "Call workspace_experiment(action='add_attempt') first, "
                            "or provide an override reason if recording is not needed.",
                            reason="experiment_not_recorded",
                        )
                    else:
                        # v3: Check inject suppression before warning
                        if self._should_suppress_inject("launch_without_recording"):
                            return None  # Silently allow — we've warned enough
                        return GuardVerdict.inject(
                            "[ExperimentTracking] Launching training without "
                            "recording an experiment attempt. Consider calling "
                            "workspace_experiment(action='add_attempt') first.",
                            reason="experiment_not_recorded_warn",
                        )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track experiment-related tool calls."""

        # Track add_attempt calls
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action == "add_attempt":
                self._attempt_recorded = True
                self._unrecorded_launches = 0  # v3: Always reset counter
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
            if self._is_actual_launch(cmd):
                self._training_launched = True
                # Reset for next launch cycle
                self._attempt_recorded = False

        # Track training result (from monitor)
        if self._training_launched and ctx.tool_name in (
            "monitor", "find_latest_log", "parse_training_metrics"
        ):
            if ctx.tool_result:
                self._training_launched = False
                self._result_pending = True

                # v3: Check inject suppression before reminding
                if self._should_suppress_inject("update_result"):
                    return None

                # Remind to update experiment (only if we know which one)
                if self._last_experiment_name:
                    self._record_trigger("update_result")
                    return GuardVerdict.inject(
                        f"[ExperimentTracking] Training result observed. "
                        f"Update experiment '{self._last_experiment_name}' with the result.",
                        reason="update_experiment_result",
                    )

        return None

    def reset_turn(self):
        """v3: Check satisfaction on turn boundary.
        
        If guard is satisfied, reset state so it doesn't keep injecting.
        Persistent tracking (unrecorded_launches) still decays via _tick_idle.
        """
        # Don't fully reset — let the lifecycle system handle decay.
        # But if satisfied, clear inject state to prevent context pollution.
        if self._attempt_recorded and not self._result_pending:
            self._inject_counts.clear()
            self._is_suppressed = False
