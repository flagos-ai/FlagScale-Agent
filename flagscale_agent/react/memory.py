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

"""Memory store — short-lived sticky notes for cross-session continuity.

Three categories only:
- fact: verifiable state (a value, path, config)
- pitfall: failure experience (symptom → cause → fix)
- insight: undigested pattern (discovery + digestion direction + target artifact)

Key format: type/domain/specific (e.g. fact/cluster/ssh_port)
"""

import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yaml


VALID_TYPES = frozenset(("fact", "pitfall", "insight"))

# Key must be type/domain/specific, each segment: lowercase alnum + underscore only
_KEY_RE = re.compile(
    r"^(fact|pitfall|insight)/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*$"
)


class Memory:
    """Flat-file memory store. Each entry is a YAML file keyed by sanitized path."""

    def __init__(self, memory_dir: str):
        self._dir = memory_dir

    @staticmethod
    def is_valid_key(key: str) -> bool:
        """Check if key matches type/domain/specific format."""
        return bool(_KEY_RE.match(key))

    @staticmethod
    def validate_key(key: str) -> Optional[str]:
        """Validate key format. Returns error message if invalid, None if valid."""
        if not key:
            return "Key cannot be empty."
        parts = key.split("/")
        if len(parts) != 3:
            return (
                f"Key must have exactly 3 segments: type/domain/specific. "
                f"Got {len(parts)} segment(s): '{key}'."
            )
        type_part, domain, specific = parts
        if type_part not in VALID_TYPES:
            return (
                f"First segment must be one of {sorted(VALID_TYPES)}. "
                f"Got '{type_part}'."
            )
        seg_re = re.compile(r"^[a-z][a-z0-9_]*$")
        if not seg_re.match(domain):
            return (
                f"Domain segment must be lowercase alphanumeric "
                f"(plus _), starting with a letter. Got '{domain}'."
            )
        if not seg_re.match(specific):
            return (
                f"Specific segment must be lowercase alphanumeric "
                f"(plus _), starting with a letter. Got '{specific}'."
            )
        return None

    def _entry_path(self, key: str) -> str:
        """Convert key to filesystem path. Slashes become double-underscores."""
        safe_name = key.replace("/", "__")
        return os.path.join(self._dir, f"{safe_name}.yaml")

    def get(self, key: str) -> Optional[dict]:
        """Get entry by exact key."""
        path = self._entry_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def put(self, key: str, mem_type: str, content: str,
            session_id: str = "", task: str = "") -> str:
        """Write or update a memory entry. Returns the file path."""
        os.makedirs(self._dir, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self._entry_path(key)

        # Check if updating existing entry
        existing = self.get(key)
        if existing:
            entry = {
                "key": key,
                "type": mem_type,
                "content": content,
                "created_session": existing.get("created_session", session_id),
                "created_at": existing.get("created_at", now),
                "updated_session": session_id,
                "updated_at": now,
                "task": task or existing.get("task", ""),
            }
        else:
            entry = {
                "key": key,
                "type": mem_type,
                "content": content,
                "created_session": session_id,
                "created_at": now,
                "updated_session": session_id,
                "updated_at": now,
                "task": task,
            }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
        return path

    def delete(self, key: str) -> bool:
        """Delete entry by key. Returns True if deleted."""
        path = self._entry_path(key)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def list_entries(self, type_filter: str = "",
                     domain_filter: str = "",
                     keyword: str = "") -> List[dict]:
        """List all entries with optional filters.

        Args:
            type_filter: filter by type (fact/pitfall/insight)
            domain_filter: filter by domain segment
            keyword: substring match on key + content
        """
        if not os.path.isdir(self._dir):
            return []

        entries = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if not entry or not isinstance(entry, dict):
                    continue
                # Apply filters
                if type_filter and entry.get("type") != type_filter:
                    continue
                if domain_filter:
                    key = entry.get("key", "")
                    parts = key.split("/")
                    if len(parts) < 2 or domain_filter not in parts[1]:
                        continue
                if keyword:
                    # Multi-token AND: "nccl timeout" → both must appear.
                    # Single token preserves the original substring semantics
                    # (backward compatible with existing callers/tests).
                    kws = keyword.lower().split()
                    text = (entry.get("key", "") + " " + entry.get("content", "")).lower()
                    if not all(kw in text for kw in kws):
                        continue
                entries.append(entry)
            except Exception:
                continue
        return entries

    def list_by_prefix(self, prefix: str) -> List[dict]:
        """List entries whose key starts with prefix.

        Examples:
            list_by_prefix("fact/") → all facts
            list_by_prefix("fact/cluster/") → all cluster facts
            list_by_prefix("pitfall/nccl/") → all nccl pitfalls
        """
        if not os.path.isdir(self._dir):
            return []

        entries = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if entry and isinstance(entry, dict):
                    if entry.get("key", "").startswith(prefix):
                        entries.append(entry)
            except Exception:
                continue
        return entries

    def clear(self) -> int:
        """Remove all entries. Returns count deleted."""
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if fname.endswith(".yaml"):
                os.remove(os.path.join(self._dir, fname))
                count += 1
        return count
