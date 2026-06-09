"""Read file tool with line range support and short-term caching."""

import os
import time

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_FS


class FileCache:
    """Short-term file content cache with TTL and invalidation on write.

    Design principles:
    - TTL-based expiry: cached content expires after _TTL seconds
    - mtime-based validation: if file mtime changes, cache is invalid
    - Explicit invalidation: write_file/edit_file calls invalidate the path
    - Memory-bounded: max _MAX_ENTRIES, LRU eviction
    """

    _TTL = 30  # seconds
    _MAX_ENTRIES = 50

    def __init__(self):
        self._store = {}  # path -> (content_lines, mtime, cached_at)

    def get(self, path: str):
        """Return cached lines or None if cache miss/stale."""
        entry = self._store.get(path)
        if entry is None:
            return None
        content_lines, cached_mtime, cached_at = entry
        # TTL check
        if time.time() - cached_at > self._TTL:
            del self._store[path]
            return None
        # mtime check — file may have been modified externally
        try:
            current_mtime = os.path.getmtime(path)
            if current_mtime != cached_mtime:
                del self._store[path]
                return None
        except OSError:
            del self._store[path]
            return None
        return content_lines

    def put(self, path: str, lines: list):
        """Cache file content."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        # LRU eviction: remove oldest entry if at capacity
        if len(self._store) >= self._MAX_ENTRIES and path not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][2])
            del self._store[oldest_key]
        self._store[path] = (lines, mtime, time.time())

    def invalidate(self, path: str):
        """Explicitly invalidate a path (called after write/edit)."""
        self._store.pop(path, None)

    def invalidate_all(self):
        """Clear entire cache."""
        self._store.clear()


# Module-level singleton so it's shared across tool instances
_file_cache = FileCache()


def get_file_cache() -> FileCache:
    """Access the shared file cache (for invalidation from write/edit tools)."""
    return _file_cache


class ReadFileTool(Tool):
    name = "read_file"
    effects = EFFECT_READ_FS
    description = (
        "Read the contents of a file. Supports line ranges for large files. "
        "Returns content with line numbers for easy reference. "
        "IMPORTANT: Prefer reading the whole file (omit start_line/end_line) over "
        "fragmented reads. If you need to understand a file, read it once in full. "
        "Re-reading the same file wastes tokens — save key findings to memory instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based). Default: 1",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive). Default: end of file. Max 500 lines per call.",
            },
            "numbered": {
                "type": "boolean",
                "description": "Include line numbers. Default: true",
            },
        },
        "required": ["path"],
    }

    MAX_LINES = 500

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        start = kwargs.get("start_line", 1)
        end = kwargs.get("end_line")
        numbered = kwargs.get("numbered", True)

        if not os.path.exists(path):
            return f"ERROR: File not found: {path}"
        if os.path.isdir(path):
            return f"ERROR: Path is a directory: {path}"

        # Try cache first (only for full-file reads without line range)
        is_full_read = (start <= 1 and end is None)
        lines = None
        if is_full_read:
            lines = _file_cache.get(path)

        if lines is None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as e:
                return f"ERROR: {e}"
            # Cache full file content
            _file_cache.put(path, lines)

        total = len(lines)
        start = max(1, start)
        if end is None:
            end = min(start + self.MAX_LINES - 1, total)
        end = min(end, total)

        if start > total:
            return f"File has {total} lines, requested start_line={start} is past end."

        selected = lines[start - 1:end]
        truncated = end < total and (end - start + 1) >= self.MAX_LINES

        if numbered:
            width = len(str(end))
            output_lines = [f"{i:{width}d}| {line}" for i, line in enumerate(selected, start=start)]
        else:
            output_lines = selected

        header = f"[{path}] lines {start}-{end} of {total}"
        result = header + "\n" + "".join(output_lines)

        if truncated:
            result += f"\n... truncated at {self.MAX_LINES} lines. Use start_line={end + 1} to continue."

        return result
