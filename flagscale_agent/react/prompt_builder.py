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

"""System prompt builder for FlagScale Agent.

V2 redesign: builds a static prompt body (cache-friendly) with a tiny
dashboard appended at the end. Memory and plan are NOT injected into the
prompt — they are accessed on-demand via tools (memory_list, plan_status).
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Dict

from flagscale_agent.react.prompt import (
    SYSTEM_PROMPT_STATIC,
    DASHBOARD_TEMPLATE,
)
from flagscale_agent.react.memory import Memory
from flagscale_agent.react.paths import get_memory_dir

if TYPE_CHECKING:
    from flagscale_agent.react.skills import SkillManager


class PromptBuilder:
    """Assembles the system prompt from static template + optional sections + dashboard."""

    def __init__(self, skill_manager: "SkillManager"):
        self._skill_manager = skill_manager
        self._turn_count = 0
        # Live runtime gauges, populated by the agent before each refresh().
        # Keys (all optional): ctx_pressure (float 0..1+), evictable (int),
        # evict_count (int, cumulative evictions this session),
        # budget ({elapsed,budget,remaining,pct} or None). The dashboard only
        # renders a gauge when its key is present — absent means "no data",
        # never a fabricated value.
        self.runtime_stats: Dict[str, object] = {}

    def refresh(
        self,
        history,
        active_skill_content: dict[str, str],
        tool_names: list[str] | None = None,
        # Legacy params — accepted but ignored (removed from prompt injection)
        memory_context: str = "",
        plan_context: str = "",
        session_dir: str = "",
        # Deprecated — accepted but ignored
        shared_storage_paths: list[str] | None = None,
    ):
        """Build and set the system prompt on the history manager.

        Args:
            history: HistoryManager instance to set prompt on
            active_skill_content: IGNORED (kept for backward compat, skill content no longer injected)
            tool_names: List of available tool names
            memory_context: IGNORED (kept for backward compat, not injected)
            plan_context: IGNORED for prompt injection (used only for dashboard)
            session_dir: Session directory path, injected into dashboard
        """
        self._turn_count += 1

        # ── Tool names ──
        tools_str = (
            ", ".join(tool_names)
            if tool_names
            else "read_file, write_file, edit_file, shell, web_fetch, load_skill, "
            "memory_write, memory_read, memory_list, monitor, plan_create, "
            "plan_update, plan_status"
        )

        # ── Skills summary for header ──
        skills_summary = self._build_skills_summary()

        # ── Knowledge summary for header ──
        knowledge_summary = self._build_knowledge_summary()

        # ── Assemble static block ──
        core = SYSTEM_PROMPT_STATIC.format(
            cwd=os.getcwd(),
            tools=tools_str,
            skills=skills_summary,
            knowledge=knowledge_summary,
        )

        # ── Append dashboard at the very end ──
        dashboard = self._build_dashboard(plan_context, session_dir, history=history)
        if dashboard:
            core += DASHBOARD_TEMPLATE.format(dashboard_content=dashboard)

        history.set_system_prompt(core)

    def _build_skills_summary(self) -> str:
        """Build compact summary of all available skills for the header line."""
        try:
            available = self._skill_manager.list_skills()
            lines = []
            for s in available:
                name = s.get("name", "")
                desc = s.get("description", "")
                lines.append(f"- {name}: {desc}")
            return "\n".join(lines)
        except Exception:
            return "(skills not available)"

    def _build_knowledge_summary(self) -> str:
        """Build compact summary of available knowledge groups."""
        try:
            from flagscale_agent.knowledge import KnowledgeManager
            km = KnowledgeManager()
            groups = km.list_groups()
            if not groups:
                return "(no knowledge loaded)"
            lines = []
            for g in groups:
                lines.append(f"- {g['name']}: {g['description']}")
            return "\n".join(lines)
        except Exception:
            return "(knowledge not available)"

    def _build_dashboard(self, plan_context: str, session_dir: str = "",
                         history=None) -> str:
        """Build the dashboard line for the end of the prompt.

        Extracts plan title/step from plan_context if available.
        Format: "Task: <title> | Step: N/M | Turn: <n>"
        Appends session paths so the agent can access conversation logs directly.
        Renders runtime gauges (Ctx / Time / BG) from self.runtime_stats when
        the corresponding data exists — absent data renders nothing.
        """
        import re
        parts = []

        # Extract the plan-level hypothesis (if any) and REMOVE it from the
        # working copy before step parsing. It is free text and may contain
        # patterns like "[🔄] Step N:" that would otherwise corrupt step
        # counting. Rendered in full below — never truncated.
        hypothesis = ""
        if plan_context:
            hyp_match = re.search(
                r"<current-hypothesis>(.*?)</current-hypothesis>",
                plan_context, re.DOTALL,
            )
            if hyp_match:
                hypothesis = hyp_match.group(1).strip()
                plan_context = plan_context.replace(hyp_match.group(0), "")

        if plan_context:
            # Extract title from <active-plan title="...">
            title_match = re.search(r'title="([^"]*)"', plan_context)
            if title_match:
                title = title_match.group(1).strip()
                if title:
                    parts.append(f"Task: {title}")

            # Count total steps and find current step
            step_lines = re.findall(r'\[.\] Step (\d+):', plan_context)
            total = len(step_lines)
            # Current step is the one with 🔄 or the first ⬜
            doing_match = re.search(r'\[🔄\] Step (\d+):', plan_context)
            pending_match = re.search(r'\[⬜\] Step (\d+):', plan_context)
            if doing_match:
                current = int(doing_match.group(1))
                parts.append(f"Step: {current}/{total}")
            elif pending_match:
                current = int(pending_match.group(1))
                parts.append(f"Step: {current}/{total}")

        parts.append(f"Turn: {self._turn_count}")

        # Session paths — injected so agent can read logs without shell(find ...)
        if session_dir:
            parts.append(
                f"Session: {session_dir}"
                f" | conversation.json: {session_dir}/conversation.json"
                f" | conversation_full.json: {session_dir}/conversation_full.json"
            )

        # Memory domain summary — compact per-domain counts so the agent can see
        # which retrieval prefixes exist without dumping every key (13KB → <1KB).
        memory_keys = self._build_memory_keys_summary()
        if memory_keys:
            parts.append(f"Memory domains: {memory_keys}")

        # ── Runtime gauges ──
        # Ctx: context pressure + evictable headroom + session evictions.
        # Prefer the history's own pressure (max of char-estimate and actual
        # API-reported tokens); fall back to a snapshot injected via
        # runtime_stats when history is not available (e.g. unit tests).
        ctx_parts = []
        pressure = None
        if history is not None:
            try:
                pressure = history.get_context_pressure()
            except Exception:
                pressure = None
        if pressure is None:
            pressure = self.runtime_stats.get("ctx_pressure")
        if pressure is not None:
            try:
                ctx_parts.append(f"{float(pressure):.0%}")
            except (TypeError, ValueError):
                pass
        evictable = self.runtime_stats.get("evictable")
        if evictable is None and history is not None:
            try:
                evictable = len(history.get_evictable_indexes())
            except Exception:
                evictable = None
        if evictable is not None:
            ctx_parts.append(f"evictable={evictable}")
        evict_count = self.runtime_stats.get("evict_count")
        if evict_count:
            ctx_parts.append(f"evicted={evict_count}")
        if ctx_parts:
            parts.append("Ctx: " + " ".join(ctx_parts))

        # Time: whole-task wall-clock budget (only when a real budget exists —
        # _task_budget_stats returns None when no external deadline is enforced;
        # we must NOT fabricate one).
        budget = self.runtime_stats.get("budget")
        if isinstance(budget, dict) and budget.get("budget"):
            try:
                remaining_m = max(0.0, float(budget["remaining"])) / 60.0
                parts.append(
                    f"Time: {float(budget['pct']):.0f}% used, {remaining_m:.0f}m left"
                )
            except (TypeError, ValueError, KeyError):
                pass

        # BG: live background jobs (the registry drops jobs once polled to
        # completion, so anything listed here is still alive — the anchor the
        # agent needs after an eviction wiped the conversation side).
        try:
            from flagscale_agent.react.tools.shell import _JOB_REGISTRY
            jobs = _JOB_REGISTRY.all()
        except Exception:
            jobs = []
        if jobs:
            import time as _time
            bg_parts = []
            for job in jobs:
                try:
                    elapsed = max(0, int(_time.time() - job.start))
                except Exception:
                    elapsed = 0
                bg_parts.append(f"{job.job_id}:{job.status_str()}({elapsed}s)")
            parts.append("BG: " + ", ".join(bg_parts[:4]))
            if len(jobs) > 4:
                parts[-1] += f" (+{len(jobs) - 4} more)"

        # Hypothesized problem model, rendered as its own multi-line section BELOW
        # the metric line — a single "|"-joined line cannot hold it untruncated.
        # Permanently resident; falsifying/updating it is the agent's own action
        # (plan_update set_thinking), not a guard.
        if hypothesis:
            return " | ".join(parts) + f"\n\nHypothesis (current problem model):\n{hypothesis}"

        return " | ".join(parts)

    def _build_memory_keys_summary(self) -> str:
        """Return a domain-level summary of memory keys.

        Keys follow the three-level format ``type/domain/specific``. Instead of
        dumping every full key (which grows linearly with memory size and is
        mostly noise), summarize at the second level: ``type/domain(count)``,
        sorted alphabetically, no truncation. The labels double as retrieval
        prefixes for memory_list/memory_read (e.g. ``pitfall/baseline/``).
        """
        try:
            mem = Memory(get_memory_dir())
            entries = mem.list_entries()
            if not entries:
                return ""
            domains: Dict[str, int] = {}
            for e in entries:
                key = e.get("key", "")
                segments = key.split("/")
                if len(segments) >= 2:
                    group = f"{segments[0]}/{segments[1]}"
                else:
                    group = segments[0] if segments else "(malformed)"
                domains[group] = domains.get(group, 0) + 1
            summary = " ".join(f"{g}({n})" for g, n in sorted(domains.items()))
            return f"{summary} ({len(entries)} keys total; memory_list(keyword=...) to search)"
        except Exception:
            return ""


