"""EnvCompatGuard — gentle reminder to investigate before installing.

Fires once when the agent's first install command appears without any prior
investigative commands (reading docs, checking hardware, reading requirements).
Generic — works for any hardware platform and any task type.
"""

from __future__ import annotations


from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


_INSTALL_INDICATORS = (
    "pip install", "pip3 install",
    "conda install", "conda create", "mamba install", "mamba create",
)

# Broad set of "investigative" commands — anything that gathers info
_INVESTIGATION_INDICATORS = (
    "cat ", "head ", "tail ", "less ", "more ",
    "grep ", "find ", "ls ",
    "--version", "--help",
    "which ", "where ", "type ",
    "nvidia-smi", "npu-smi", "rocm-smi",
    "nvcc", "python -c",
)


class EnvCompatGuard(Guard):
    """Warns (once) if the agent tries to install without investigating first."""

    name = "env_compat"
    priority = 15
    activate_on_states = {AgentState.EXECUTING}
    activate_on_tools = {"shell"}

    def __init__(self):
        self._investigated: bool = False
        self._warned: bool = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None

        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        cmd_lower = cmd.lower()

        # Track any investigative command
        if not self._investigated:
            if any(ind in cmd_lower for ind in _INVESTIGATION_INDICATORS):
                self._investigated = True

        # Only care about install commands
        is_install = any(ind in cmd_lower for ind in _INSTALL_INDICATORS)
        if not is_install:
            return None

        # If already investigated or already warned, let it through
        if self._investigated or self._warned:
            return None

        # First install without any investigation — warn once, don't block
        self._warned = True
        return GuardVerdict.inject(
            "[EnvCompat] Installing without prior investigation. "
            "Consider checking hardware/requirements first.",
            reason="install_without_investigation"
        )

    def mark_analysis_done(self):
        """Manually mark investigation as complete."""
        self._investigated = True

    def reset_turn(self):
        pass
