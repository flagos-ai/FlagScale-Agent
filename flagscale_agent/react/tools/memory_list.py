"""Memory list tool — browse and search memory entries."""

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_MEMORY

_TYPE_PRIORITY = {"finding": 0, "decision": 1, "todo": 2, "context": 3}


class MemoryListTool(Tool):
    name = "memory_list"
    effects = EFFECT_READ_MEMORY
    description = (
        "List and search memory entries. Use to browse what you've memorized, "
        "find entries by type or keyword, or check what's stored for a specific task. "
        "Returns entries sorted by relevance (type priority, then recency). "
        "Default scope='session' shows only current session memory. "
        "Use scope='global' for shared cross-session memory, or scope='all' for both."
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
            "scope": {
                "type": "string",
                "enum": ["session", "global", "all"],
                "description": (
                    "'session' (default): list only current session memory. "
                    "'global': list only global shared memory. "
                    "'all': list both, session entries shown first."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, global_memory, session_memory):
        self._global_memory = global_memory
        self._session_memory = session_memory

    def _filter_entries(self, entries, type_filter, keyword, task_filter):
        if type_filter:
            entries = [e for e in entries if e.get("type") == type_filter]
        if task_filter:
            entries = [e for e in entries if task_filter.lower() in (e.get("task") or "").lower()]
        if keyword:
            kw = keyword.lower()
            entries = [
                e for e in entries
                if kw in (e.get("key") or "").lower()
                or kw in (e.get("content") or "").lower()
            ]
        return entries

    def execute(self, **kwargs) -> str:
        type_filter = kwargs.get("type_filter", "")
        keyword = (kwargs.get("keyword") or "").lower()
        task_filter = kwargs.get("task_filter", "")
        limit = kwargs.get("limit", 20)
        scope = kwargs.get("scope", "session")

        session_entries = []
        global_entries = []

        if scope in ("session", "all"):
            raw = self._session_memory.list_entries()
            session_entries = self._filter_entries(raw, type_filter, keyword, task_filter)
            for e in session_entries:
                e["_scope_label"] = "session"

        if scope in ("global", "all"):
            raw = self._global_memory.list_entries()
            global_entries = self._filter_entries(raw, type_filter, keyword, task_filter)
            for e in global_entries:
                e["_scope_label"] = "global"

        # Session entries first, then global; each group sorted by type priority then recency
        def sort_key(e):
            return (_TYPE_PRIORITY.get(e.get("type", "context"), 9), -e.get("created", 0))

        session_entries.sort(key=sort_key)
        global_entries.sort(key=sort_key)
        entries = (session_entries + global_entries)[:limit]

        if not entries:
            parts = []
            if scope != "all":
                parts.append(f"scope={scope}")
            if type_filter:
                parts.append(f"type={type_filter}")
            if keyword:
                parts.append(f"keyword='{keyword}'")
            if task_filter:
                parts.append(f"task='{task_filter}'")
            filter_desc = ", ".join(parts) if parts else "no filters"
            return f"(no memory entries found matching {filter_desc})"

        lines = []
        for e in entries:
            key = e.get("key", "?")
            mem_type = e.get("type", "?")
            content = e.get("content", "")
            task = e.get("task", "")
            scope_label = e.get("_scope_label", "?")
            if len(content) > 120:
                content = content[:117] + "..."
            task_tag = f" @{task}" if task else ""
            lines.append(f"[{mem_type}] [{scope_label}] {key}{task_tag}: {content}")

        total_session = len(self._session_memory.list_entries())
        total_global = len(self._global_memory.list_entries())
        shown = len(lines)

        if scope == "session":
            header = f"Showing {shown}/{total_session} entries [session]"
        elif scope == "global":
            header = f"Showing {shown}/{total_global} entries [global]"
        else:
            header = f"Showing {shown}/{total_session + total_global} entries [session + global]"

        if type_filter or keyword or task_filter:
            header += " (filtered)"
        return header + "\n" + "\n".join(lines)
