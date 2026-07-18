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

"""CodeContextGuard — tracks code reads/writes to survive context compaction.

Problem: After context compaction, the agent loses:
1. Which files it already read and what it understood
2. Which files it modified and why
3. The "mental model" of code architecture built up over many reads

Solution: This guard maintains a code context map that:
- Records file reads with structural summaries (classes, functions, key patterns)
- Records file modifications with intent (what changed and why)
- Injects reminders when re-reading a file ("you already read this, key points: ...")
- Auto-dumps to memory when context pressure is high (before compaction erases everything)
- Auto-restores from memory on session start (reads memory_read result in check_post)

Lifecycle:
1. Agent reads/writes files → guard builds file_map in memory
2. Context pressure rises → guard suggests memory_write with key "code_context_map"
3. Context gets compacted → file_map lost (guard state is in-process only)
4. New session starts (or post-compaction) → file_map is empty
5. Guard detects empty map → injects "restore from memory" suggestion (once)
6. Agent does memory_read("code_context_map") → guard sees result in check_post
7. Guard parses the result → rebuilds file_map → continuity restored

Design principles:
- Advisory only (inject, never block) — never prevents work
- Summaries are compact (< 200 chars per file) — not full content
- Auto-dump threshold aligns with context_pressure SOFT_THRESHOLD
- Restore inject fires at most once per session to avoid loops
- Dump uses fixed key "code_context_map" — overwrites previous (always latest state)
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


# Max entries to keep (LRU eviction)
_MAX_FILE_ENTRIES = 50
# Max summary length per file
_MAX_SUMMARY_LEN = 200
# Pressure threshold to trigger auto-dump
_AUTO_DUMP_PRESSURE = 0.70
# Min reads before suggesting dump
_MIN_READS_FOR_DUMP = 8
# Fixed memory key for dump/restore
_MEMORY_KEY = "code_context_map"
# How many tool calls to wait before suggesting restore
_RESTORE_GRACE_PERIOD = 2


@dataclass
class FileContext:
    """Tracked context for a single file."""
    path: str
    last_read_time: float = 0.0
    read_count: int = 0
    summary: str = ""  # structural understanding
    modifications: list[str] = field(default_factory=list)  # list of change descriptions
    key_elements: list[str] = field(default_factory=list)  # classes, functions, patterns

    def compact_repr(self) -> str:
        """Compact string for inject messages."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        if self.modifications:
            parts.append(f"Modified: {'; '.join(self.modifications[-2:])}")
        if self.key_elements:
            parts.append(f"Contains: {', '.join(self.key_elements[:5])}")
        return " | ".join(parts)[:_MAX_SUMMARY_LEN]


class CodeContextGuard(Guard):
    """Track code reads/writes to provide continuity across context compaction."""

    name = "code_context"
    priority = 45  # Low priority — advisory, runs after most others
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING, AgentState.REVIEWING}
    overridable = True

    def __init__(self):
        # LRU-ordered map: path → FileContext
        self._file_map: OrderedDict[str, FileContext] = OrderedDict()
        # Track whether we already suggested a dump this session
        self._dump_suggested = False
        self._dump_done = False
        # Count of file operations since last dump
        self._ops_since_dump = 0
        # Restore state
        self._restore_suggested = False  # Only suggest restore once
        self._restored = False  # True after successfully parsing memory_read result
        self._tool_call_count = 0  # Track calls to know when to suggest restore

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Always accept — this guard is purely advisory."""
        return True

    def set_shared_state(self, shared_state):
        self._shared_state = shared_state

    # ── Pre-check: inject prior context when re-reading a file ──

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Suggest restore if file_map is empty and we haven't already
        if ctx.tool_name in ("read_file", "edit_file", "write_file"):
            restore_verdict = self._maybe_suggest_restore(ctx)
            if restore_verdict:
                return restore_verdict

        if ctx.tool_name != "read_file":
            return None

        path = ctx.tool_args.get("path", "")
        if not path:
            return None

        entry = self._file_map.get(path)
        if not entry or not entry.compact_repr():
            return None

        # Only inject if we've read it before AND it has meaningful context
        if entry.read_count < 1:
            return None

        # Don't inject for continuation reads (same file, different lines)
        # — agent is drilling deeper, not re-discovering
        if ctx.recent_tool_names and ctx.recent_tool_names[-1:] == ["read_file"]:
            last_args = ctx.recent_tool_history[-1].get("args_summary", "") if ctx.recent_tool_history else ""
            if path in last_args:
                return None

        context_str = entry.compact_repr()
        return GuardVerdict.inject(
            f"[CodeContext] You previously read {self._short_path(path)} "
            f"(×{entry.read_count}): {context_str}",
            reason="prior_read_context",
            category="code_context",
        )

    # ── Post-check: record reads and writes, detect restore ──

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        self._tool_call_count += 1

        # Detect memory_read("code_context_map") result → restore file_map
        if ctx.tool_name == "memory_read" and not self._restored:
            if self._is_context_map_read(ctx):
                self._restore_from_memory(ctx.tool_result or "")
                return None

        # Detect memory_write("code_context_map") → mark dump done
        if ctx.tool_name == "memory_write":
            key = ctx.tool_args.get("key", "")
            if key == _MEMORY_KEY:
                self.mark_dump_done()
                return None

        if ctx.tool_name == "read_file":
            self._record_read(ctx)
        elif ctx.tool_name in ("edit_file", "write_file"):
            self._record_write(ctx)

        # Check if we should suggest dumping to memory
        self._ops_since_dump += 1
        if self._should_suggest_dump(ctx):
            return self._suggest_dump(ctx)

        return None

    def _is_context_map_read(self, ctx: GuardContext) -> bool:
        """Check if this memory_read is for our context map key."""
        key = ctx.tool_args.get("key", "")
        return key == _MEMORY_KEY

    # ── Strategic: dump before compaction ──

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """At strategic review points, check if dump is needed."""
        if ctx.context_pressure >= _AUTO_DUMP_PRESSURE and not self._dump_done:
            if len(self._file_map) >= 3:
                return self._suggest_dump(ctx)
        return None

    # ── Recording logic ──

    def _record_read(self, ctx: GuardContext):
        """Record a file read and extract structural info."""
        path = ctx.tool_args.get("path", "")
        if not path:
            return

        result = ctx.tool_result or ""

        # Get or create entry
        entry = self._file_map.get(path)
        if entry is None:
            entry = FileContext(path=path)
            self._file_map[path] = entry
        else:
            # Move to end (LRU)
            self._file_map.move_to_end(path)

        entry.read_count += 1
        entry.last_read_time = time.time()

        # Extract structural elements from file content
        elements = self._extract_elements(result, path)
        if elements:
            entry.key_elements = elements

        # Extract summary from assistant_text if available
        # (The LLM often explains what it found after reading)
        if ctx.assistant_text:
            summary = self._extract_understanding(ctx.assistant_text, path)
            if summary:
                entry.summary = summary

        # Evict oldest if too many
        while len(self._file_map) > _MAX_FILE_ENTRIES:
            self._file_map.popitem(last=False)

    def _record_write(self, ctx: GuardContext):
        """Record a file modification."""
        path = ctx.tool_args.get("path", "")
        if not path:
            return

        entry = self._file_map.get(path)
        if entry is None:
            entry = FileContext(path=path)
            self._file_map[path] = entry
        else:
            self._file_map.move_to_end(path)

        # Build modification description
        if ctx.tool_name == "edit_file":
            old = ctx.tool_args.get("old_string", "")[:50]
            new = ctx.tool_args.get("new_string", "")[:50]
            desc = f"edited: '{old}...' → '{new}...'"
        else:
            content_preview = (ctx.tool_args.get("content", "") or "")[:60]
            mode = ctx.tool_args.get("mode", "write")
            desc = f"{mode}: {content_preview}..."

        entry.modifications.append(desc[:100])
        # Keep only last 5 modifications
        if len(entry.modifications) > 5:
            entry.modifications = entry.modifications[-5:]

    # ── Dump logic ──

    def _should_suggest_dump(self, ctx: GuardContext) -> bool:
        """Check if we should suggest dumping code context to memory."""
        if self._dump_done:
            return False
        if self._dump_suggested:
            return False
        if self._ops_since_dump < _MIN_READS_FOR_DUMP:
            return False
        if ctx.context_pressure >= _AUTO_DUMP_PRESSURE:
            return True
        return False

    def _suggest_dump(self, ctx: GuardContext) -> GuardVerdict:
        """Generate inject suggesting the agent dump code context to memory."""
        self._dump_suggested = True
        summary = self._build_dump_summary()
        return GuardVerdict.inject(
            f"[CodeContext] You have read/modified {len(self._file_map)} files. "
            f"Context pressure at {ctx.context_pressure:.0%}. "
            f"Suggest: memory_write(key='code_context_map', type='context', content=...)\n"
            f"Draft summary:\n{summary}",
            reason="pre_compaction_dump",
            category="code_context_dump",
        )

    def _build_dump_summary(self) -> str:
        """Build a compact summary of all tracked files for memory dump."""
        lines = []
        # Sort by modification (modified files first) then by read count
        entries = sorted(
            self._file_map.values(),
            key=lambda e: (bool(e.modifications), e.read_count),
            reverse=True,
        )
        for entry in entries[:20]:  # Top 20 most important
            short = self._short_path(entry.path)
            repr_str = entry.compact_repr()
            if repr_str:
                lines.append(f"  {short}: {repr_str}")
            else:
                lines.append(f"  {short}: (read ×{entry.read_count})")
        return "\n".join(lines)

    # ── Helpers ──

    def _maybe_suggest_restore(self, ctx: GuardContext) -> GuardVerdict | None:
        """Suggest restoring code context from memory if file_map is empty.

        Fires at most once. Won't fire if:
        - We already restored or suggested restore
        - File map already has entries (agent is building fresh context)
        - Too early in session (grace period to let MemoryDiscipline fire first)
        """
        if self._restored or self._restore_suggested:
            return None
        if len(self._file_map) > 0:
            return None
        if self._tool_call_count < _RESTORE_GRACE_PERIOD:
            return None

        self._restore_suggested = True
        return GuardVerdict.inject(
            f"[CodeContext] No code context loaded. If this is a continuation session, "
            f"restore with: memory_read(key='{_MEMORY_KEY}')",
            reason="suggest_restore",
            category="code_context_restore",
        )

    def _restore_from_memory(self, content: str):
        """Parse memory content and rebuild file_map.

        Expected format (one entry per line):
          path: summary_text
        or structured format:
          [FILES_READ]
          path | elements | summary
          [FILES_MODIFIED]
          path | modifications
        """
        if not content or _MEMORY_KEY not in content.split('\n')[0]:
            # Try direct parse — content IS the value
            self._parse_context_lines(content)
        else:
            self._parse_context_lines(content)

        if self._file_map:
            self._restored = True

    def _parse_context_lines(self, content: str):
        """Parse line-by-line context dump into file_map."""
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('[') or line.startswith('#'):
                continue

            # Format: "  path/to/file.py: description text"
            # or:     "path/to/file.py: description text"
            match = re.match(r'\s*(.+?\.\w+):\s*(.+)', line)
            if match:
                path = match.group(1).strip()
                desc = match.group(2).strip()

                entry = FileContext(path=path)
                entry.read_count = 1  # Mark as "seen before"

                # Parse description for known patterns
                if "Modified:" in desc:
                    mod_part = desc.split("Modified:")[1].strip()
                    entry.modifications = [m.strip() for m in mod_part.split(";")[:3]]
                    summary_part = desc.split("Modified:")[0].strip().rstrip("|").strip()
                    if summary_part:
                        entry.summary = summary_part
                elif "Contains:" in desc:
                    contains_part = desc.split("Contains:")[1].strip()
                    entry.key_elements = [e.strip() for e in contains_part.split(",")[:5]]
                    summary_part = desc.split("Contains:")[0].strip().rstrip("|").strip()
                    if summary_part:
                        entry.summary = summary_part
                else:
                    entry.summary = desc[:_MAX_SUMMARY_LEN]

                self._file_map[path] = entry

    def _extract_elements(self, content: str, path: str) -> list[str]:
        """Extract key structural elements from file content."""
        elements = []
        if not content:
            return elements

        # Python files: classes and top-level functions
        if path.endswith(".py"):
            classes = re.findall(r'^class (\w+)', content, re.MULTILINE)
            functions = re.findall(r'^def (\w+)', content, re.MULTILINE)
            elements.extend(f"class {c}" for c in classes[:5])
            elements.extend(f"def {f}" for f in functions[:5])

        # YAML files: top-level keys
        elif path.endswith((".yaml", ".yml")):
            keys = re.findall(r'^(\w[\w_-]*):', content, re.MULTILINE)
            elements.extend(keys[:8])

        return elements[:10]

    def _extract_understanding(self, text: str, path: str) -> str:
        """Try to extract the agent's understanding from its response text."""
        # Look for patterns where the agent explains what it found
        short = self._short_path(path)

        # Pattern: "X contains/has/implements Y"
        patterns = [
            rf'{re.escape(short)}[^.]*?(?:contains|implements|defines|has)\s+([^.]+)',
            rf'(?:this file|this module|it)\s+(?:contains|implements|defines|has)\s+([^.]+)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:_MAX_SUMMARY_LEN]

        return ""

    def mark_dump_done(self):
        """Called externally when agent successfully writes code context to memory."""
        self._dump_done = True
        self._ops_since_dump = 0

    @staticmethod
    def _short_path(path: str) -> str:
        """Shorten a path for display."""
        parts = path.split("/")
        if len(parts) > 3:
            return "/".join(parts[-3:])
        return path

    def reset_turn(self):
        """Keep file map across turns — it's the whole point.
        Only reset the dump suggestion flag so it can fire again."""
        self._dump_suggested = False

    def get_context_map(self) -> dict[str, str]:
        """Public API: get current code context map for external use."""
        return {
            path: entry.compact_repr()
            for path, entry in self._file_map.items()
            if entry.compact_repr()
        }
