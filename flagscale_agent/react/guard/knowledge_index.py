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

"""KnowledgeIndexGuard — reminds to rebuild the knowledge index after editing a doc.

Editing a knowledge doc under flagscale_agent/knowledge/docs/ shifts the line
numbers cached in indexes/<group>.idx (generate_index.py parses section headers
with their line numbers). Nothing else detects this, so a subsequent
load_knowledge(doc=..., start_line=L, end_line=L) silently reads the WRONG lines.
This guard fires a post-edit inject pointing at the rebuild command.

Mirrors UnitTestGuard: Guard subclass, advisory (never blocks), post-check only.
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict


class KnowledgeIndexGuard(Guard):
    """Post-tool guard that reminds to regenerate the knowledge index."""

    name = "knowledge_index_rebuild"
    priority = 70  # advisory, same tier as UnitTestGuard

    # Only these tools actually modify files
    WRITE_TOOLS = ("write_file", "edit_file")
    # A knowledge doc must live under this subtree and be markdown
    DOCS_MARKER = "knowledge/docs/"

    def __init__(self):
        # Knowledge docs modified this turn (for the reminder message)
        self._pending_docs: set[str] = set()
        # Fire at most once per turn — avoid spamming on many sequential edits
        self._injected_this_turn = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # No pre-check needed

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only trigger on file-writing operations
        if ctx.tool_name not in self.WRITE_TOOLS:
            return None

        path = ctx.tool_args.get("path", "") or ""
        if not self._is_knowledge_doc(path):
            return None

        self._pending_docs.add(path)

        # One reminder per turn is enough — the rebuild is per-group, so the
        # affected-group set does not change the remedy.
        if self._injected_this_turn:
            return None
        self._injected_this_turn = True

        docs = ", ".join(sorted(self._pending_docs)[-3:])
        return GuardVerdict.inject(
            f"[KnowledgeIndex] You edited a knowledge doc ({docs}). The knowledge "
            f"index caches each section's LINE NUMBER, so it is now stale — "
            f"load_knowledge would read the wrong lines. Regenerate it: "
            f"`cd flagscale_agent/knowledge && python3 generate_index.py` (or "
            f"regenerate just the affected group), then sync the derived stats "
            f"table in docs/analysis_standards/01_infrastructure_analysis.md.",
            reason="knowledge_index_stale",
            category="knowledge_index_rebuild",
        )

    def reset_turn(self):
        """Reset per-turn state at the start of each user message."""
        self._injected_this_turn = False
        self._pending_docs.clear()

    @staticmethod
    def _is_knowledge_doc(path: str) -> bool:
        """True for a markdown doc under the knowledge/docs/ subtree."""
        if not path:
            return False
        if KnowledgeIndexGuard.DOCS_MARKER not in path:
            return False
        return path.endswith(".md")
