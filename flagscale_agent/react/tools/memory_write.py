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

"""Memory write tool — save facts, pitfalls, and insights."""

import re

from flagscale_agent.react.tools.base import Tool


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Save a memory entry for cross-session continuity. "
        "Three types only:\n"
        "- fact: verifiable environment state (a value, path, config)\n"
        "- pitfall: failure experience (symptom → cause → fix)\n"
        "- insight: undigested pattern (discovery + digestion direction + target)\n\n"
        "Key format: type/domain/specific (e.g. fact/cluster/ssh_port, "
        "pitfall/nccl/nic_exclude_syntax, insight/agent/memory_redesign).\n\n"
        "Writing the same key updates the existing entry. "
        "Use 'supersedes' to delete old entries that this new one replaces.\n\n"
        "Semantic uniqueness: a NEW key that looks like a near-duplicate of an "
        "existing key in the same domain is BLOCKED — you must reuse/update that "
        "key, supersede it, or pass force_new=true. One key per concept.\n\n"
        "Write conditions:\n"
        "- fact: info obtained by probing, not obvious, likely needed in future sessions\n"
        "- pitfall: debugging took >2 rounds, cause was non-obvious, likely to recur\n"
        "- insight: reusable pattern found, cannot digest now, has clear target artifact\n\n"
        "Do NOT use memory for: "
        "session temp state (→ plan/context), easily re-read configs (→ read_file), "
        "complete procedures (→ skill), systematic knowledge (→ knowledge)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Three-level key: type/domain/specific. "
                    "Examples: 'fact/cluster/ssh_port', 'pitfall/nccl/nic_hang', "
                    "'insight/skill/nccl_debug_method'. "
                    "All segments lowercase, alphanumeric + underscore only."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["fact", "pitfall", "insight"],
                "description": "Memory type. Must match the first segment of the key.",
            },
            "content": {
                "type": "string",
                "description": (
                    "The memory content. Format by type:\n"
                    "- fact: 'value: X\\napplies: Y\\nverify cmd: Z'\n"
                    "- pitfall: 'symptom: X\\ncause: Y\\nfix: Z\\nenv: W'\n"
                    "- insight: 'finding: X\\ndigest direction: Y\\ntarget artifact: Z'"
                ),
            },
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of old memory keys to delete (this entry replaces them).",
            },
            "force_new": {
                "type": "boolean",
                "description": (
                    "Set true ONLY after reviewing the similar-entry list returned "
                    "by a prior blocked write, when you are certain this is a "
                    "genuinely new semantic entry (not a near-duplicate of an "
                    "existing key). Default false."
                ),
            },
        },
        "required": ["key", "type", "content"],
    }

    # Date/time-ish tokens carry no semantics for dedup (e.g. 0824, 1056, 2026).
    _NOISE_TOKEN_RE = re.compile(r"^\d{2,8}$|^v\d+$")

    @classmethod
    def _meaningful_tokens(cls, specific: str) -> set:
        """Split a key's specific segment into semantic tokens, dropping
        pure-digit date/time/version noise."""
        toks = set()
        for t in specific.split("_"):
            if not t or cls._NOISE_TOKEN_RE.match(t):
                continue
            toks.add(t)
        return toks

    def _find_similar(self, key: str) -> list:
        """Return existing keys under the same type/domain/ that share enough
        meaningful tokens with the new key's specific segment to be likely
        near-duplicates. Empty list means the write is semantically distinct."""
        parts = key.split("/")
        if len(parts) != 3:
            return []
        mem_type, domain, specific = parts
        prefix = f"{mem_type}/{domain}/"
        new_toks = self._meaningful_tokens(specific)
        if not new_toks:
            return []
        similar = []
        for entry in self._memory.list_by_prefix(prefix):
            ekey = entry.get("key", "")
            if ekey == key:
                continue  # exact-key update handled separately
            eparts = ekey.split("/")
            if len(eparts) != 3:
                continue
            etoks = self._meaningful_tokens(eparts[2])
            if not etoks:
                continue
            shared = new_toks & etoks
            # Flag when tokens strongly overlap: >=2 shared, OR one token-set
            # contains the other (prefix/superset relationship).
            if len(shared) >= 2 or new_toks <= etoks or etoks <= new_toks:
                similar.append(ekey)
        return similar

    def __init__(self, memory, session_id: str = "", task_plan=None):
        self._memory = memory
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
        force_new = kwargs.get("force_new", False)
        task = self._get_current_task()

        from flagscale_agent.react.memory import Memory, VALID_TYPES

        # Validate type
        if mem_type not in VALID_TYPES:
            return (
                f"ERROR: Invalid type '{mem_type}'. "
                f"Must be one of: {sorted(VALID_TYPES)}."
            )

        # Validate key format
        error = Memory.validate_key(key)
        if error:
            return f"ERROR: Invalid key '{key}'. {error}"

        # Validate type consistency: key prefix must match type
        key_type = key.split("/")[0]
        if key_type != mem_type:
            return (
                f"ERROR: Key prefix '{key_type}' does not match "
                f"type '{mem_type}'. They must be the same."
            )

        # Semantic-uniqueness gate: only for genuinely new keys.
        # Skipped when updating an existing key, superseding old ones, or
        # explicitly forced. Prevents near-duplicate keys for one concept.
        is_update = self._memory.get(key) is not None
        if not is_update and not supersedes and not force_new:
            similar = self._find_similar(key)
            if similar:
                listing = "\n".join(f"  - {k}" for k in sorted(similar))
                return (
                    f"BLOCKED: '{key}' looks semantically close to "
                    f"{len(similar)} existing entr"
                    f"{'y' if len(similar) == 1 else 'ies'} under the same "
                    f"domain:\n{listing}\n\n"
                    "Memory must keep one key per concept. Choose one:\n"
                    "  1. UPDATE an existing key above (re-call memory_write "
                    "with that exact key — it overwrites in place).\n"
                    "  2. REPLACE stale ones (re-call with supersedes=[...] "
                    "listing the keys this entry retires).\n"
                    "  3. If this really is a NEW concept (not a duplicate), "
                    "re-call with force_new=true.\n"
                    "Read the listed entries first (memory_read) before deciding."
                )

        try:
            # Delete superseded entries
            deleted = []
            for old_key in supersedes:
                if self._memory.delete(old_key):
                    deleted.append(old_key)

            # Write new entry
            self._memory.put(key, mem_type, content, self._session_id, task=task)

            supersede_info = f" Superseded: {', '.join(deleted)}." if deleted else ""
            return (
                f"Memorized [{mem_type}] '{key}' "
                f"({len(content)} chars).{supersede_info}"
            )
        except Exception as e:
            return f"ERROR: Failed to save memory: {e}"
