"""Write file tool."""

import os

from flagscale_agent.react.tools.base import Tool, EFFECT_WRITE_FS
from flagscale_agent.react.tools.read_file import get_file_cache

# -- Paths that should never be written by the agent --
_PROTECTED_PATHS = frozenset({
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.profile"),
    os.path.expanduser("~/.bash_profile"),
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.ssh/authorized_keys"),
})


def _is_protected_path(path: str) -> bool:
    """Check if path is protected from agent writes."""
    resolved = os.path.abspath(os.path.realpath(path))
    if resolved in _PROTECTED_PATHS:
        return True
    if resolved.startswith("/etc/") and not resolved.startswith("/etc/apt/"):
        return True
    if resolved.startswith("/boot/"):
        return True
    return False


class WriteFileTool(Tool):
    name = "write_file"
    effects = EFFECT_WRITE_FS
    description = (
        "Create or overwrite a file at the given path with the provided content. "
        "For large files (>3000 chars), split into multiple calls using mode='append' "
        "after the first write. Example: first call with mode='write', then subsequent "
        "calls with mode='append' to add remaining content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to write.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append"],
                "description": "Write mode: 'write' (default) overwrites the file, 'append' adds to the end.",
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "write")

        if not path:
            return "ERROR: 'path' parameter is required but was empty or missing (possible output truncation)."
        if not content:
            return "ERROR: 'content' parameter is required but was empty or missing (possible output truncation)."

        if _is_protected_path(path):
            return f"ERROR: Cannot write to protected system path: {path}"

        file_mode = "a" if mode == "append" else "w"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, file_mode, encoding="utf-8") as f:
                f.write(content)
            get_file_cache().invalidate(os.path.abspath(path))
            get_file_cache().invalidate(path)
            action = "Appended" if mode == "append" else "Wrote"
            total = os.path.getsize(os.path.abspath(path))
            return f"{action} {len(content)} chars to {path} (total file size: {total} bytes)"
        except Exception as e:
            return f"ERROR: {e}"
