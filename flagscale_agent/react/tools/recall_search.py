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

"""recall_search tool — full-text search over the session conversation log.

Searches conversation_full.json (the complete, never-truncated session log)
so the agent can locate past context even after eviction / hard reset.
Returned hits carry the message's external index, which recall(index=N)
accepts directly.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from flagscale_agent.react.tools.base import Tool

# Flattened-content hard cap per hit (chars). Keeps each hit small enough
# that N hits stay inside max_result_size.
_HIT_SNIPPET_LIMIT = 600


def _flatten_content(content: Any) -> str:
    """Flatten message content (str or block list) to searchable text.

    Block dicts keep their type prefix so hits can be filtered by block kind;
    nested dict values are dumped as JSON for robustness on unknown shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "?")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype == "thinking":
                    parts.append(str(block.get("thinking", "")))
                elif btype == "tool_use":
                    parts.append("tool_use:%s(%s)" % (
                        block.get("name", "?"),
                        json.dumps(block.get("input", {}), ensure_ascii=False),
                    ))
                elif btype == "tool_result":
                    parts.append("tool_result:%s" % (
                        json.dumps(block.get("content", ""), ensure_ascii=False),
                    ))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _snippet(text: str, keywords: List[str], limit: int = _HIT_SNIPPET_LIMIT) -> str:
    """Build a hit snippet centered on the earliest keyword occurrence."""
    anchor = -1
    anchor_kw = ""
    lowered = text.lower()
    for kw in keywords:
        pos = lowered.find(kw.lower())
        if pos != -1 and (anchor == -1 or pos < anchor):
            anchor = pos
            anchor_kw = kw
    if anchor == -1:
        return text[:limit]
    start = max(0, anchor - limit // 3)
    end = start + limit
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


class RecallSearchTool(Tool):
    """Full-text search over conversation_full.json with recallable anchors."""

    name = "recall_search"
    description = (
        "Search the FULL session conversation log (conversation_full.json) by "
        "keywords. Unlike recall(index=N) which needs a known index, this finds "
        "the right index when you only remember a phrase. Multi-keyword AND "
        "matching, optional role filter. Returned hits include message index "
        "usable directly with recall(index=N).\n\n"
        "Examples:\n"
        "- recall_search(query='flock session lock') → hits containing BOTH terms\n"
        "- recall_search(query='NCCL timeout', role='user') → only user messages\n"
        "- recall_search(query='config', limit=5) → cap result count"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords to search, space-separated = AND semantics "
                    "(all must appear in the same message). Case-insensitive."
                ),
            },
            "role": {
                "type": "string",
                "description": (
                    "Optional role filter: user / assistant / system. "
                    "Omit to search all roles."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max hits to return (default 10, max 50).",
            },
            "order": {
                "type": "string",
                "description": (
                    "Hit ordering: 'recent' (default, newest first) or "
                    "'oldest' (chronological)."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, session_dir: str):
        # Path to the session's conversation_full.json. The tool re-resolves
        # the file on every call (cheap: existence check), so a long-lived
        # agent always searches the CURRENT log even after reload.
        self._session_dir = session_dir

    def _log_path(self) -> str:
        return os.path.join(self._session_dir, "conversation_full.json")

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        role_filter = kwargs.get("role") or None
        limit = kwargs.get("limit", 10)
        order = kwargs.get("order", "recent")

        if not query or not query.strip():
            return "ERROR: query is required (space-separated keywords)."
        keywords = [k for k in query.split() if k]
        limit = max(1, min(int(limit), 50))
        if order not in ("recent", "oldest"):
            order = "recent"

        path = self._log_path()
        if not os.path.exists(path):
            return ("ERROR: conversation_full.json not found at %s — "
                    "session log missing or session_dir wrong." % path)

        t0 = time.time()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return "ERROR: cannot read %s: %s" % (path, e)

        messages = data.get("messages", [])
        # External index = messages position + 1 (system prompt is not in
        # full_log; recall_from_full_log maps ext -> full_log_pos = ext - 1).
        hits: List[Dict[str, Any]] = []
        for pos, msg in enumerate(messages):
            if role_filter and msg.get("role") != role_filter:
                continue
            text = _flatten_content(msg.get("content", ""))
            if not text:
                continue
            if _match_all(text, keywords):
                hits.append({
                    "index": pos + 1,
                    "role": msg.get("role", "?"),
                    "text": text,
                })

        hits.sort(key=lambda h: h["index"], reverse=(order == "recent"))
        hits = hits[:limit]
        elapsed = time.time() - t0

        return self._format(keywords, hits, len(messages), role_filter,
                            elapsed, order)

    def _format(self, keywords: List[str], hits: List[Dict[str, Any]],
                total_messages: int, role_filter: Optional[str],
                elapsed: float, order: str) -> str:
        lines = []
        n_kw = " ".join(keywords)
        scope = "role=%s" % role_filter if role_filter else "all roles"
        lines.append(
            "recall_search: %d hit(s) for %r [%s] in %.2fs (scanned %d messages)"
            % (len(hits), n_kw, scope, elapsed, total_messages)
        )
        if not hits:
            lines.append("No message contains ALL keywords: %r" % keywords)
            lines.append("Tip: try fewer keywords, or memory_list(keyword=...) "
                         "for distilled knowledge.")
            return "\n".join(lines)
        lines.append("Use recall(index=N) to get a hit's full content.")
        for h in hits:
            snippet = _snippet(h["text"], keywords)
            snippet = snippet.replace("\n", " ")
            lines.append("")
            lines.append("[index=%d | role=%s] %s" % (h["index"], h["role"],
                                                      snippet))
        return "\n".join(lines)


def _match_all(text: str, keywords: List[str]) -> bool:
    """Case-insensitive AND match: every keyword must appear in text."""
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in keywords)
