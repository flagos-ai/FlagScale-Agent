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

"""Memory read tool — retrieve a specific memory entry."""

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_MEMORY


class MemoryReadTool(Tool):
    name = "memory_read"
    effects = EFFECT_READ_MEMORY
    description = (
        "Read a specific memory entry by key. "
        "Use when you know a memory exists and want to retrieve its details. "
        "Searches session memory first, then global memory. "
        "If both session and global have the same key, the session entry takes precedence and the global entry is not returned. "
        "To explicitly read the global version, pass scope='global'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key of the memory entry to read.",
            },
            "scope": {
                "type": "string",
                "enum": ["session", "global", "all"],
                "description": (
                    "'all' (default): search session memory first, then global. "
                    "'session': only search current session memory. "
                    "'global': only search global shared memory."
                ),
            },
        },
        "required": ["key"],
    }

    def __init__(self, global_memory, session_memory):
        self._global_memory = global_memory
        self._session_memory = session_memory

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        scope = kwargs.get("scope", "all")

        if scope == "session":
            entry = self._session_memory.get(key)
            if entry is None:
                return f"No session memory found for '{key}'."
            return f"[{entry.get('type', '?')}] [session] {entry.get('content', '')}"

        if scope == "global":
            entry = self._global_memory.get(key)
            if entry is None:
                return f"No global memory found for '{key}'."
            return f"[{entry.get('type', '?')}] [global] {entry.get('content', '')}"

        # scope == "all": session first, then global
        entry = self._session_memory.get(key)
        if entry is not None:
            return f"[{entry.get('type', '?')}] [session] {entry.get('content', '')}"
        entry = self._global_memory.get(key)
        if entry is not None:
            return f"[{entry.get('type', '?')}] [global] {entry.get('content', '')}"
        return f"No memory found for '{key}'."
