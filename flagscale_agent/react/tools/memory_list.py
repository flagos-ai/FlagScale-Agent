"""Memory list tool — browse and search memory entries."""

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_MEMORY


class MemoryListTool(Tool):
    name = "memory_list"
    effects = EFFECT_READ_MEMORY
    description = (
        "List and search memory entries. Use to browse what you've memorized, "
        "find entries by type or keyword, or check what's stored for a specific task. "
        "Returns entries sorted by relevance (type priority, then recency)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "type_filter": {
                "type": "string",
                "enum": ["finding", "decision", "todo", "context", ""],
                "description": "Filter by memory type. Empty string for all types.",
            },
            "keyword": {
                "type": "string",
                "description": "Search keyword (case-insensitive substring match on key and content).",
            },
            "task_filter": {
                "type": "string",
                "description": "Filter by task name.",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to return (default 20).",
            },
        },
        "required": [],
    }

    def __init__(self, memory):
        self._memory = memory

    def execute(self, **kwargs) -> str:
        type_filter = kwargs.get("type_filter", "")
        keyword = (kwargs.get("keyword") or "").lower()
        task_filter = kwargs.get("task_filter", "")
        limit = kwargs.get("limit", 20)

        entries = self._memory.list_entries()

        if type_filter:
            entries = [e for e in entries if e.get("type") == type_filter]

        if task_filter:
            entries = [e for e in entries if task_filter.lower() in (e.get("task") or "").lower()]

        if keyword:
            entries = [
                e for e in entries
                if keyword in (e.get("key") or "").lower()
                or keyword in (e.get("content") or "").lower()
            ]

        if not entries:
            parts = []
            if type_filter:
                parts.append(f"type={type_filter}")
            if keyword:
                parts.append(f"keyword='{keyword}'")
            if task_filter:
                parts.append(f"task='{task_filter}'")
            filter_desc = ", ".join(parts) if parts else "no filters"
            return f"(no memory entries found matching {filter_desc})"

        # Sort: type priority, then recency
        type_priority = {"finding": 0, "decision": 1, "todo": 2, "context": 3}
        entries.sort(key=lambda e: (
            type_priority.get(e.get("type", "context"), 9),
            -e.get("created", 0),
        ))

        entries = entries[:limit]

        lines = []
        for e in entries:
            key = e.get("key", "?")
            mem_type = e.get("type", "?")
            content = e.get("content", "")
            task = e.get("task", "")
            # Truncate content for listing
            if len(content) > 120:
                content = content[:117] + "..."
            task_tag = f" @{task}" if task else ""
            lines.append(f"[{mem_type}] {key}{task_tag}: {content}")

        total = len(self._memory.list_entries())
        shown = len(lines)
        header = f"Showing {shown}/{total} entries"
        if type_filter or keyword or task_filter:
            header += " (filtered)"
        return header + "\n" + "\n".join(lines)
