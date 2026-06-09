"""Edit file tool — exact string replacement."""

import os

from flagscale_agent.react.tools.base import Tool, EFFECT_WRITE_FS
from flagscale_agent.react.tools.read_file import get_file_cache

# -- Same protected paths as write_file.py --
_PROTECTED_PATHS = frozenset({
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.profile"),
    os.path.expanduser("~/.bash_profile"),
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.ssh/authorized_keys"),
})


def _is_protected_path(path: str) -> bool:
    resolved = os.path.abspath(os.path.realpath(path))
    if resolved in _PROTECTED_PATHS:
        return True
    if resolved.startswith("/etc/") and not resolved.startswith("/etc/apt/"):
        return True
    if resolved.startswith("/boot/"):
        return True
    return False


class EditFileTool(Tool):
    name = "edit_file"
    effects = EFFECT_WRITE_FS
    description = "Edit a file by replacing an exact string match. The old_string must match exactly (including whitespace)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact string to find and replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement string.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace all occurrences. Default: false (replace first only).",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]

        if _is_protected_path(path):
            return f"ERROR: Cannot edit protected system path: {path}"
        path = kwargs["path"]
        old_string = kwargs["old_string"]
        new_string = kwargs["new_string"]
        replace_all = kwargs.get("replace_all", False)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_string not in content:
                return f"ERROR: old_string not found in {path}"

            count = content.count(old_string)
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replaced = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replaced = 1

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            get_file_cache().invalidate(path)
            msg = f"Successfully edited {path}"
            if count > 1:
                msg += f" (replaced {replaced} of {count} occurrences)"
            return msg
        except FileNotFoundError:
            return f"ERROR: file not found: {path}"
        except Exception as e:
            return f"ERROR: {e}"
