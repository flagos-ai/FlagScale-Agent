"""Memory read tool — retrieve a specific memory entry."""

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_MEMORY


class MemoryReadTool(Tool):
    name = "memory_read"
    effects = EFFECT_READ_MEMORY
    description = (
        "Read a specific memory entry by key. "
        "Use when you know a memory exists and want to retrieve its details."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key of the memory entry to read.",
            },
        },
        "required": ["key"],
    }

    def __init__(self, memory):
        self._memory = memory

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        entry = self._memory.get(key)
        if entry is None:
            return f"No memory found for '{key}'."
        return f"[{entry.get('type', '?')}] {entry.get('content', '')}"
