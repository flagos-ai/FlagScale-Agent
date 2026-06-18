"""Memory write tool — save key findings, decisions, and todos."""

from flagscale_agent.react.tools.base import Tool, EFFECT_WRITE_MEMORY


class MemoryWriteTool(Tool):
    name = "memory_write"
    effects = EFFECT_WRITE_MEMORY
    description = (
        "Save a key finding, decision, or todo for future sessions. "
        "Writing the same key updates the existing entry. "
        "Use 'supersedes' to delete old entries that this new one replaces. "
        "Entries are automatically associated with the current task from workspace current.yaml. "
        "PROACTIVE RULE: after any unexpected failure that required a workaround, "
        "immediately memorize it if a future session could hit the same issue. "
        "SUPERSEDE RULE: when new information contradicts, completes, or replaces older memories, "
        "use 'supersedes' to list the old key(s) to delete.\n"
        "Do NOT use memory for: experiment records (use workspace_experiment), "
        "current session state (use workspace_current), or information easily re-read from files/configs.\n\n"
        "SCOPE: Default scope is determined by type — "
        "finding → global (objective facts worth sharing across sessions); "
        "decision/todo/context → session (current-session state, not reusable). "
        "Override with explicit scope='session' or scope='global' when the default is wrong."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Short identifier: <scope>_<topic>[_<detail>]. "
                    "Examples: 'qwen3_architecture_overview', 'env_apex_build_fix'. "
                    "Lowercase alphanumeric and underscores only, 2-80 chars."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["finding", "decision", "todo", "context"],
                "description": "Memory type.",
            },
            "content": {
                "type": "string",
                "description": "The memory content. Be specific — include exact errors, flags, or version numbers.",
            },
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of old memory keys that this entry replaces.",
            },
            "scope": {
                "type": "string",
                "enum": ["session", "global"],
                "description": (
                    "Override the default scope. Default is type-driven: "
                    "finding → global, decision/todo/context → session. "
                    "Only set this when the default is wrong for your use case."
                ),
            },
        },
        "required": ["key", "type", "content"],
    }

    def __init__(self, global_memory, session_memory, session_id: str = "", task_plan=None):
        # global_memory -> .flagscale/agent_memory/  (shared across sessions)
        # session_memory -> .flagscale/sessions/{id}/memory/  (this session only)
        self._global_memory = global_memory
        self._session_memory = session_memory
        self._session_id = session_id
        self._task_plan = task_plan

    def _get_current_task(self) -> str:
        if self._task_plan:
            active = self._task_plan.get_active()
            if active:
                return active.get("title", "")
        return ""

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        mem_type = kwargs["type"]
        content = kwargs["content"]
        supersedes = kwargs.get("supersedes", [])
        # Default scope by type: finding → global (reusable knowledge);
        # decision/todo/context → session (current-session state).
        # Can be overridden by explicit scope param.
        default_scope = "global" if mem_type == "finding" else "session"
        scope = kwargs.get("scope", default_scope)
        task = self._get_current_task()

        from flagscale_agent.react.memory import SessionMemory

        if not SessionMemory.is_valid_key(key):
            sanitized = SessionMemory.sanitize_key(key)
            if not sanitized or not SessionMemory.is_valid_key(sanitized):
                return (
                    f"ERROR: Invalid memory key '{key}'. "
                    "Key must be 2-80 chars, lowercase alphanumeric and underscores only."
                )
            return (
                f"ERROR: Invalid memory key '{key}'. "
                f"Suggested key: '{sanitized}'."
            )

        # Route to the correct store based on scope
        store = self._global_memory if scope == "global" else self._session_memory

        try:
            deleted = []
            for old_key in supersedes:
                # Try to delete from both stores (key might exist in either)
                deleted_global = self._global_memory.delete(old_key)
                deleted_session = self._session_memory.delete(old_key)
                if deleted_global or deleted_session:
                    deleted.append(old_key)

            store.put(key, mem_type, content, self._session_id, task=task)
            scope_label = "global" if scope == "global" else "session"
            task_info = f" [task: {task}]" if task else ""
            supersede_info = f" Superseded: {', '.join(deleted)}." if deleted else ""
            return f"Memorized [{mem_type}] '{key}' ({len(content)} chars) [{scope_label}].{task_info}{supersede_info}"
        except Exception as e:
            return f"ERROR: Failed to save memory: {e}"
