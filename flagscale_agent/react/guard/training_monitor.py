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

"""TrainingMonitorGuard — block non-monitor calls after training launch.

Deterministic trigger: training detected AND next_call != monitor.
No escalation, no whitelist complexity — just block once until monitor is called.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command


# Error patterns that indicate the launch command failed
_FAILURE_PATTERNS = (
    "Cannot find primary config",
    "Error:",
    "Traceback (most recent call last)",
    "FileNotFoundError",
    "ModuleNotFoundError",
    "ImportError",
    "No such file or directory",
    "command not found",
    "Permission denied",
    "OmegaConf.errors",
    "hydra.errors",
)


def _command_failed(result: str) -> bool:
    """Heuristic: detect if a shell command output indicates failure."""
    if not result:
        return False
    # Check for common error indicators
    for pattern in _FAILURE_PATTERNS:
        if pattern in result:
            return True
    return False


class TrainingMonitorGuard(Guard):
    """Block non-monitor calls after training launch."""

    name = "training_monitor"
    priority = 50


    def __init__(self):
        self._launch_detected: bool = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Detect training launch."""
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if isinstance(cmd, str) and _is_flagscale_launch_command(cmd):
                # Only set launch_detected if command did not obviously fail
                result = ctx.tool_result or ""
                if isinstance(result, str) and _command_failed(result):
                    pass  # Don't flag failed launches
                else:
                    self._launch_detected = True
        elif ctx.tool_name == "flagscale_train_monitor":
            self._launch_detected = False  # Cleared
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block non-monitor calls after launch."""
        if not self._launch_detected:
            return None

        # Skip the pre-iteration synthetic check (tool_name="").
        # Only gate actual tool calls. The LLM must be allowed to run so it
        # can see the guard message and respond with flagscale_train_monitor.
        if not ctx.tool_name:
            return None

        if ctx.tool_name == "flagscale_train_monitor":
            self._launch_detected = False
            return None

        return GuardVerdict.block(
            "[TrainingMonitor] Training launched. Must call "
            "flagscale_train_monitor(output_dir='...') immediately to observe progress.",
            reason="must_monitor_after_launch",
            category="training_monitor",
        )

    def reset_turn(self):
        """State persists across turns."""
        pass
