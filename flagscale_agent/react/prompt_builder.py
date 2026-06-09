"""System prompt builder for FlagScale Agent.

Extracted from agent.py to reduce file size and improve modularity.
Handles assembly of the system prompt from skills, scene, and runtime context.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from flagscale_agent.react.prompt import (
    SYSTEM_PROMPT_CORE, SYSTEM_PROMPT_OPTIONAL,
)

if TYPE_CHECKING:
    from flagscale_agent.react.skills import SkillManager
    from flagscale_agent.react.scene import ScenePreset


class PromptBuilder:
    """Builds and refreshes the system prompt for WorkerAgent.

    Encapsulates all prompt assembly logic: skill summaries, critical rules,
    situational context, and focused context injection.
    """

    def __init__(self, skill_manager: "SkillManager", scene: "ScenePreset | None"):
        self._skill_manager = skill_manager
        self._scene = scene
        self._turn_count = 0

    def refresh(
        self,
        history,
        active_skill_content: dict[str, str],
        current_stage_id: str | None,
        shared_storage_paths: list[str],
        memory_context: str = "",
        plan_context: str = "",
        tool_names: list[str] | None = None,
    ):
        """Build and set the system prompt on the history manager.

        Args:
            history: HistoryManager instance to set prompt on
            active_skill_content: {skill_name: content} for loaded skills
            current_stage_id: Current workflow stage ID (for focused context)
            shared_storage_paths: Detected shared filesystem paths
            memory_context: Optional memory context string
            plan_context: Optional plan context string
        """
        self._turn_count += 1
        skills_summary = self._build_skills_summary()
        cwd = os.getcwd()
        tools_str = ", ".join(tool_names) if tool_names else "read_file, write_file, edit_file, shell, web_fetch, load_skill, memory_write, memory_read, memory_list, monitor, plan_create, plan_update, plan_status"

        # Build optional sections based on scene constraints (data-driven)
        optional_parts = []
        CONSTRAINT_TO_SECTION = {
            "is_training": "experiment",
            "is_inference": "inference",
            "is_serving": "serving",
        }
        scene_constraints = (self._scene.constraints or set()) if self._scene else set()
        for constraint, section_key in CONSTRAINT_TO_SECTION.items():
            if constraint in scene_constraints:
                section = SYSTEM_PROMPT_OPTIONAL.get(section_key, "")
                if section:
                    optional_parts.append(section)
        # Also inject experiment workflow for training/inference tasks
        if "is_training" in scene_constraints or "is_inference" in scene_constraints:
            exp_section = SYSTEM_PROMPT_OPTIONAL.get("experiment", "")
            if exp_section and exp_section not in optional_parts:
                optional_parts.append(exp_section)
        # Planning section only when a plan exists (saves ~192 tokens otherwise)
        if plan_context:
            optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("planning", ""))
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("memory_rules", ""))
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("decision", ""))
        # User commands only on first 3 turns (saves ~172 tokens on subsequent turns)
        if self._turn_count <= 3:
            optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("user_commands", ""))
        optional_sections = "\n\n".join(p for p in optional_parts if p)

        skill_context = ""
        if active_skill_content:
            skill_bodies = []
            for name, content in active_skill_content.items():
                # Use focused context if skill has context_injection rules
                focused = self._skill_manager.get_focused_context(
                    name,
                    stage_id=current_stage_id,
                    tool_name=None,
                )
                if focused:
                    skill_bodies.append(focused)
                elif self._turn_count <= 5:
                    # Full skill content only for first 5 turns
                    skill_bodies.append(content)
                else:
                    # After turn 5: compact summary only (critical rules already extracted separately)
                    # Include just the first 3 lines as a reminder of what the skill is
                    lines = content.strip().split("\n")
                    header = "\n".join(lines[:3])
                    skill_bodies.append(f"{header}\n[... full content omitted after turn 5 to save tokens ...]")
            skill_context = "\n\n".join(skill_bodies)

        critical_rules = self._build_critical_rules(active_skill_content)
        situational = self._build_situational_context(shared_storage_paths)

        core = SYSTEM_PROMPT_CORE.format(
            tools=tools_str,
            skills=skills_summary,
            cwd=cwd,
            plan_context=plan_context,
            memory_context=memory_context,
            situational_context=situational,
            optional_sections=optional_sections,
            skill_context=skill_context,
            critical_rules=critical_rules,
        )

        history.set_system_prompt(core)

    def _build_skills_summary(self) -> str:
        """Build a compact summary of all available skills.

        Keywords are omitted from the prompt (they're used by judge for matching,
        not needed in system prompt). Descriptions are capped at 80 chars.
        Saves ~780 tokens per turn compared to including keywords.
        """
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

    def _build_critical_rules(self, active_skill_content: dict[str, str]) -> str:
        """Extract CRITICAL-level rules from loaded skills.

        These appear in the system prompt BEFORE the main skill content,
        at a position of high attention weight.
        """
        blocks = []
        for name, content in active_skill_content.items():
            lines = content.split("\n")
            capturing = False
            captured = []
            for line in lines:
                if line.startswith("#") and "CRITICAL" in line.upper():
                    capturing = True
                    captured.append(line)
                elif capturing:
                    if line.startswith("#") and "CRITICAL" not in line.upper():
                        capturing = False
                        if captured:
                            blocks.append("\n".join(captured))
                            captured = []
                    else:
                        captured.append(line)
            if captured:
                blocks.append("\n".join(captured))
        return "\n\n".join(blocks)

    @staticmethod
    def _build_situational_context(shared_storage_paths: list[str]) -> str:
        """Build dynamic context about the runtime environment."""
        parts = []
        if shared_storage_paths:
            parts.append(
                "## Shared Storage\n\n"
                "The following shared/network filesystem paths are available. "
                "For multi-node training, conda environments and data should be "
                "placed on shared storage so all nodes can access them.\n\n"
                + "\n".join(f"- `{p}`" for p in shared_storage_paths)
                + "\n\nWhen creating conda environments, use `--prefix` targeting "
                "one of these paths instead of `-n <name>`.\n"
            )
        return "\n\n".join(parts)
