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
    SYSTEM_PROMPT_OPTIONAL,
    DASHBOARD_TEMPLATE,
)

if TYPE_CHECKING:
    from flagscale_agent.react.skills import SkillManager
    from flagscale_agent.react.scene import ScenePreset


class PromptBuilder:
    """Assembles the system prompt from static template + optional sections + dashboard."""

    def __init__(self, skill_manager: "SkillManager", scene: "ScenePreset | None"):
        self._skill_manager = skill_manager
        self._scene = scene
        self._turn_count = 0

    @property
    def scene(self) -> "ScenePreset | None":
        return self._scene

    @scene.setter
    def scene(self, value: "ScenePreset | None"):
        self._scene = value

    def refresh(
        self,
        history,
        active_skill_content: dict[str, str],
        current_stage_id: str | None,
        shared_storage_paths: list[str],
        tool_names: list[str] | None = None,
        # Legacy params — accepted but ignored (removed from prompt injection)
        memory_context: str = "",
        plan_context: str = "",
    ):
        """Build and set the system prompt on the history manager.

        Args:
            history: HistoryManager instance to set prompt on
            active_skill_content: {skill_name: content} for loaded skills
            current_stage_id: Current workflow stage ID (for focused skill context)
            shared_storage_paths: Detected shared filesystem paths
            tool_names: List of available tool names
            memory_context: IGNORED (kept for backward compat, not injected)
            plan_context: IGNORED for prompt injection (used only for dashboard)
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

        # ── Optional sections based on scene constraints ──
        optional_sections = self._build_optional_sections(plan_context)

        # ── Skill context (full or abbreviated based on turn count) ──
        skill_context = self._build_skill_context(
            active_skill_content, current_stage_id
        )

        # ── Critical rules extracted from skills ──
        critical_rules = self._build_critical_rules(active_skill_content)

        # ── Shared storage note ──
        shared_storage_note = self._build_shared_storage_note(shared_storage_paths)

        # ── Assemble static block ──
        core = SYSTEM_PROMPT_STATIC.format(
            cwd=os.getcwd(),
            tools=tools_str,
            skills=skills_summary,
            critical_rules=critical_rules,
            optional_sections=optional_sections + shared_storage_note,
            skill_context=skill_context,
        )

        # ── Append dashboard at the very end ──
        dashboard = self._build_dashboard(plan_context)
        if dashboard:
            core += DASHBOARD_TEMPLATE.format(dashboard_content=dashboard)

        history.set_system_prompt(core)

    def _build_optional_sections(self, plan_context: str) -> str:
        """Select optional sections based on scene constraints."""
        optional_parts = []

        # Scene-driven sections
        scene_constraints = (
            (self._scene.constraints or set()) if self._scene else set()
        )

        CONSTRAINT_TO_SECTION = {
            "is_training": "experiment",
            "is_inference": "inference",
            "is_serving": "serving",
        }
        for constraint, section_key in CONSTRAINT_TO_SECTION.items():
            if constraint in scene_constraints:
                section = SYSTEM_PROMPT_OPTIONAL.get(section_key, "")
                if section:
                    optional_parts.append(section)

        # Experiment workflow for training/inference
        if "is_training" in scene_constraints or "is_inference" in scene_constraints:
            exp_section = SYSTEM_PROMPT_OPTIONAL.get("experiment", "")
            if exp_section and exp_section not in optional_parts:
                optional_parts.append(exp_section)

        # Planning section only when a plan exists
        if plan_context:
            planning = SYSTEM_PROMPT_OPTIONAL.get("planning", "")
            if planning:
                optional_parts.append(planning)

        # Always include these
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("memory_rules", ""))
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("decision", ""))

        # User commands only on first 3 turns
        if self._turn_count <= 3:
            optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("user_commands", ""))

        return "\n\n".join(p for p in optional_parts if p)

    def _build_skill_context(
        self, active_skill_content: Dict[str, str], current_stage_id: str | None
    ) -> str:
        """Build skill context block.

        Strategy:
        - Use focused context if skill has context_injection rules (stage-aware)
        - First 5 turns after loading: full skill content
        - After 5 turns: header only (critical rules extracted separately)
        """
        if not active_skill_content:
            return ""

        skill_bodies = []
        for name, content in active_skill_content.items():
            if self._turn_count <= 5:
                # Early turns: full content (focused or raw)
                focused = self._skill_manager.get_focused_context(
                    name, stage_id=current_stage_id, tool_name=None
                )
                # get_focused_context returns full body when no injection rules,
                # which is fine for early turns
                skill_bodies.append(focused if focused else content)
            else:
                # After turn 5: compact header only
                # Critical rules are already extracted separately
                lines = content.strip().split("\n")
                header = "\n".join(lines[:3])
                skill_bodies.append(
                    f"{header}\n[... use load_skill('{name}') for full content ...]"
                )

        return "\n\n".join(skill_bodies)

    def _build_critical_rules(self, active_skill_content: dict[str, str]) -> str:
        """Extract CRITICAL-level rules from loaded skills.

        Looks for content between `## CRITICAL` / `# CRITICAL` and the next heading.
        Always included regardless of turn count.
        """
        if not active_skill_content:
            return ""

        critical_parts = []
        for name, content in active_skill_content.items():
            lines = content.split("\n")
            capturing = False
            captured = []
            for line in lines:
                if line.strip().lower().startswith(("## critical", "# critical")):
                    capturing = True
                    continue
                elif capturing and line.strip().startswith("#"):
                    break
                elif capturing:
                    captured.append(line)

            if captured:
                text = "\n".join(captured).strip()
                if text:
                    critical_parts.append(f"[{name} critical rules]\n{text}")

        if not critical_parts:
            return ""
        return "\n\n".join(critical_parts) + "\n"

    def _build_skills_summary(self) -> str:
        """Build compact summary of all available skills for the header line."""
        try:
            available = self._skill_manager.list_skills()
            lines = []
            for s in available:
                name = s.get("name", "")
                desc = s.get("description", "")[:80]
                lines.append(f"- {name}: {desc}")
            return "\n".join(lines)
        except Exception:
            return "(skills not available)"

    def _build_dashboard(self, plan_context: str) -> str:
        """Build the dashboard line for the end of the prompt.

        Extracts plan title/step from plan_context if available.
        Format: "Task: <title> | Step: N/M | Turn: <n>"
        """
        import re
        parts = []

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
        return " | ".join(parts)

    def _build_shared_storage_note(self, shared_storage_paths: list[str]) -> str:
        """Build a note about shared storage paths for conda environments."""
        if not shared_storage_paths:
            return ""
        return (
            "\n\n## Shared Storage\n\nAvailable shared storage paths:\n"
            + "\n".join(f"- `{p}`" for p in shared_storage_paths)
            + "\n\nWhen creating conda environments, use `--prefix` targeting "
            "one of these paths instead of `-n <name>`.\n"
        )

    # Keep old name for backward compatibility
    build_shared_storage_note = _build_shared_storage_note
