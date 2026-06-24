"""MegatronPathGuard — prevents editing wrong megatron/ directory.

FlagScale workspace often has two copies of megatron code:
1. FlagScale/megatron/ — bundled copy (may be stale)
2. Megatron-LM-FL/megatron/ — editable install source (active in Python path)

Editing the wrong one wastes time and causes confusion. This guard:
- Detects when agent edits a file under any megatron/ path
- Checks which megatron path is actually active (via Python import)
- Warns if the edited path doesn't match the active one
"""

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class MegatronPathGuard(Guard):
    """Warn when editing megatron/ files that aren't on the active Python path."""

    name = "megatron_path"
    priority = 30

    def __init__(self):
        self._active_megatron_path: str = ""
        self._detection_attempted = False
        self._warned_paths: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name not in ("edit_file", "write_file"):
            return None

        path = ctx.tool_args.get("path", "")
        if not path:
            return None

        # Only care about megatron/ paths
        if "/megatron/" not in path and not path.startswith("megatron/"):
            return None

        # Detect active megatron path (once per session)
        if not self._detection_attempted:
            self._detect_active_path()

        if not self._active_megatron_path:
            return None

        # Check if the edited path is under the active megatron root
        edited_abs = os.path.abspath(path)
        active_root = self._active_megatron_path

        if not edited_abs.startswith(active_root):
            # Editing wrong copy!
            if edited_abs not in self._warned_paths:
                self._warned_paths.add(edited_abs)
                return GuardVerdict.inject(
                    f"[MegatronPath] Wrong megatron copy: editing {edited_abs} "
                    f"but runtime uses {active_root}. Edit the correct copy.",
                    reason="wrong_megatron_path",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def _detect_active_path(self):
        """Detect which megatron/ is active in the Python path."""
        self._detection_attempted = True
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c", "import megatron; import os; print(os.path.dirname(megatron.__file__))"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._active_megatron_path = result.stdout.strip()
        except Exception:
            pass

    def reset_turn(self):
        pass
