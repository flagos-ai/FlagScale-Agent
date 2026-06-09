"""Skill manager — load and parse SKILL.md files.

Enhanced for Phase 5.2 Skill-centric architecture:
- get_workflow(): extract workflow stages from frontmatter
- get_constraints(): extract hard constraints for ConstraintGuard
- get_focused_context(): return only relevant sections by stage/tool
"""

import os
import re

from typing import Dict, List, Optional, Tuple

import yaml

from flagscale_agent.react.constraint import Constraint, ConstraintTrigger



class SkillManager:
    """Manages skill loading from prioritized directories."""

    def __init__(self, dirs: List[str]):
        self._dirs = dirs
        self._scan_cache: Optional[Dict[str, str]] = None
        self._list_cache: Optional[List[Dict[str, str]]] = None
        self._meta_cache: Dict[str, Dict] = {}

    def invalidate_cache(self):
        """Invalidate cached scan/list results. Call after skill dirs change."""
        self._scan_cache = None
        self._list_cache = None
        self._meta_cache = {}

    def _scan(self) -> Dict[str, str]:
        """Build mapping: skill_name -> skill_file_path (later dirs override). Cached."""
        if self._scan_cache is not None:
            return self._scan_cache
        mapping = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        name = meta.get("name", entry)
                    except Exception:
                        name = entry
                    mapping[name] = skill_file
                    mapping[entry] = skill_file
        self._scan_cache = mapping
        return mapping

    def list_skills(self) -> List[Dict[str, str]]:
        """Scan all directories and return available skills (deduplicated). Cached."""
        if self._list_cache is not None:
            return self._list_cache
        seen_paths = {}
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for entry in os.listdir(d):
                skill_file = os.path.join(d, entry, "SKILL.md")
                if os.path.isfile(skill_file):
                    try:
                        meta, _ = self._parse_file(skill_file)
                        seen_paths[skill_file] = {
                            "name": meta.get("name", entry),
                            "description": meta.get("description", ""),
                            "keywords": meta.get("keywords", []),
                            "parameters": meta.get("parameters", []),
                            "requires": meta.get("requires", []) or [],
                            "suggests": meta.get("suggests", []) or [],
                        }
                    except Exception:
                        seen_paths[skill_file] = {"name": entry, "description": "", "keywords": [], "parameters": []}
        self._list_cache = list(seen_paths.values())
        return self._list_cache

    def load(self, name: str, _loading_stack: set | None = None, **params) -> str:
        """Load a skill by frontmatter name or directory name. Later directories take priority.

        Optional keyword arguments are substituted into {param_name} placeholders
        in the skill body. Parameters defined in frontmatter with defaults are
        used when not provided by the caller.

        Auto-loads dependency summaries declared in frontmatter 'requires' field.
        Appends 'suggests' hints for related skills.
        """
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            raise FileNotFoundError(f"Skill '{name}' not found in: {self._dirs}")
        meta, body = self._parse_file(skill_file)
        skill_name = meta.get("name", name)

        # Auto-load dependency summaries with circular dependency detection
        if _loading_stack is None:
            _loading_stack = set()
        _loading_stack.add(name)

        requires = meta.get("requires", [])
        if requires and isinstance(requires, list):
            dep_hints = []
            for dep_name in requires:
                if dep_name in _loading_stack:
                    continue
                # Prefer summary over full content for dependencies
                summary = self.load_summary(dep_name)
                if summary:
                    dep_hints.append(f"<dependency name=\"{dep_name}\" type=\"summary\">\n{summary}\n</dependency>")
                else:
                    try:
                        dep_content = self.load(dep_name, _loading_stack=_loading_stack, **params)
                        dep_hints.append(dep_content)
                    except FileNotFoundError:
                        pass
            if dep_hints:
                body = "\n\n".join(dep_hints) + "\n\n" + body

        # Append suggests hints (lightweight — just names and descriptions)
        suggests = meta.get("suggests", [])
        if suggests and isinstance(suggests, list):
            available = {s["name"] for s in self.list_skills()}
            valid_suggests = [s for s in suggests if s in available]
            if valid_suggests:
                hints = ", ".join(f"`{s}`" for s in valid_suggests)
                body += f"\n\n---\nRelated skills (load if needed): {hints}"

        _loading_stack.discard(name)

        param_defs = meta.get("parameters", [])
        if isinstance(param_defs, list):
            for pdef in param_defs:
                if isinstance(pdef, dict):
                    pname = pdef.get("name", "")
                    if pname and pname not in params and "default" in pdef:
                        params[pname] = pdef["default"]

        for k, v in params.items():
            body = body.replace(f"{{{k}}}", str(v))

        return f"<skill name=\"{skill_name}\">\n{body}\n</skill>"

    def get_effects(self, name: str) -> Dict:
        """Get the 'effects' declaration from skill frontmatter.

        Returns a dict that may contain:
          - mode: str — mode flag to set on the agent (e.g. "porting")
          - companion_skills: list[str] — skills to auto-load alongside
        Returns empty dict if no effects declared.
        """
        meta = self.get_meta(name)
        return meta.get("effects", {}) or {}

    def get_meta(self, name: str) -> Dict:
        """Get skill frontmatter metadata without loading full content. Cached."""
        if name in self._meta_cache:
            return self._meta_cache[name]
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return {}
        try:
            meta, _ = self._parse_file(skill_file)
            self._meta_cache[name] = meta
            return meta
        except Exception:
            return {}

    def get_dependency_closure(self, names: list, _seen: set | None = None) -> list:
        """Return a topologically sorted list of skill names to load, including
        all transitive `requires` and `suggests` of the given skills.

        `requires` are loaded first (strong deps), then the skill itself,
        then `suggests` (weak deps). Circular references are skipped.
        """
        if _seen is None:
            _seen = set()
        result = []
        for name in names:
            if name in _seen:
                continue
            _seen.add(name)
            meta = self.get_meta(name)
            if not meta:
                continue
            # Depth-first: load requires first
            reqs = meta.get("requires", []) or []
            for dep in reqs:
                if dep not in _seen:
                    sub_result = self.get_dependency_closure([dep], _seen)
                    result.extend(sub_result)
            # Then the skill itself (if not already added by depends chain)
            # We add the skill name separately so the caller loads it
            if name not in [r for r in result]:
                result.append(name)
            # Then suggests (weak deps, loaded after)
            suggs = meta.get("suggests", []) or []
            for sug in suggs:
                if sug not in _seen:
                    sub_result = self.get_dependency_closure([sug], _seen)
                    result.extend(sub_result)
        return result

    def load_summary(self, name: str) -> str | None:
        """Load SUMMARY.md for a skill if it exists. Returns None if no summary available."""
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return None
        skill_dir = os.path.dirname(skill_file)
        summary_file = os.path.join(skill_dir, "SUMMARY.md")
        if not os.path.isfile(summary_file):
            return None
        with open(summary_file, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_file(self, path: str) -> Tuple[dict, str]:
        """Read a SKILL.md and split YAML frontmatter from body."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._parse_frontmatter(content)

    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[dict, str]:
        """Split --- delimited YAML frontmatter from markdown body."""
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            meta = {}
        body = parts[2].strip()
        return meta, body

    # ── Phase 5.2: Skill-centric enhancements ─────────────────────────────

    def get_workflow(self, name: str) -> Optional[Dict]:
        """Get workflow definition from skill frontmatter.

        Returns the workflow dict (with 'trigger' and 'stages' keys) or None.
        """
        meta = self.get_meta(name)
        workflow = meta.get("workflow")
        if not workflow or not isinstance(workflow, dict):
            return None
        if "stages" not in workflow or not workflow["stages"]:
            return None
        return workflow

    def get_constraints(self, name: str) -> List[Constraint]:
        """Extract hard constraints from skill frontmatter.

        Returns compiled Constraint objects ready for ConstraintGuard.
        """
        meta = self.get_meta(name)
        raw_constraints = meta.get("constraints")
        if not raw_constraints or not isinstance(raw_constraints, list):
            return []

        result = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            try:
                constraint = self._compile_constraint(item)
                result.append(constraint)
            except Exception as e:
                pass
        return result

    def get_focused_context(
        self,
        name: str,
        stage_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> str:
        """Return focused context — only relevant sections of the skill body.

        Uses context_injection rules from frontmatter to determine which
        markdown sections to include. Falls back to full body when no rules defined.
        """
        mapping = self._scan()
        skill_file = mapping.get(name)
        if skill_file is None:
            return ""
        meta, body = self._parse_file(skill_file)

        injection_rules = meta.get("context_injection")
        if not injection_rules or not isinstance(injection_rules, dict):
            return body  # No rules → full body

        # Collect section titles to inject
        sections_to_inject: set = set()

        # Always-inject sections
        always = injection_rules.get("always", [])
        if isinstance(always, list):
            sections_to_inject.update(always)

        # By-stage sections
        if stage_id:
            by_stage = injection_rules.get("by_stage", {})
            if isinstance(by_stage, dict):
                stage_sections = by_stage.get(stage_id, [])
                if isinstance(stage_sections, list):
                    sections_to_inject.update(stage_sections)

        # By-tool sections
        if tool_name:
            by_tool = injection_rules.get("by_tool", {})
            if isinstance(by_tool, dict):
                tool_sections = by_tool.get(tool_name, [])
                if isinstance(tool_sections, list):
                    sections_to_inject.update(tool_sections)

        if not sections_to_inject:
            return body  # No sections specified → full body

        return self._extract_sections(body, sections_to_inject)

    @staticmethod
    def _extract_sections(body: str, section_titles: set) -> str:
        """Extract markdown sections by heading title.

        Matches ## or ### headings. Extracts content until the next heading
        of equal or higher level.
        """
        if not section_titles:
            return body

        # Normalize titles for case-insensitive matching
        normalized_titles = {t.lower().strip() for t in section_titles}

        # Split body into sections by headings
        # Pattern matches ## or ### headings
        heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

        sections = []
        matches = list(heading_pattern.finditer(body))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            if title.lower() not in normalized_titles:
                continue

            # Find the end of this section (next heading of same or higher level)
            start = match.start()
            end = len(body)
            for j in range(i + 1, len(matches)):
                next_level = len(matches[j].group(1))
                if next_level <= level:
                    end = matches[j].start()
                    break

            sections.append(body[start:end].strip())

        if not sections:
            # No matching sections found — return full body as fallback
            return body

        return "\n\n".join(sections)

    @staticmethod
    def _compile_constraint(item: dict) -> Constraint:
        """Compile a constraint dict from frontmatter into a Constraint object."""
        trigger_raw = item.get("trigger", item.get("trigger_on", {}))
        if not isinstance(trigger_raw, dict):
            trigger_raw = {}

        # Support both 'tools' and 'tool' keys
        tool_names_raw = trigger_raw.get("tools", [])
        if not tool_names_raw:
            tool_val = trigger_raw.get("tool", "")
            if tool_val:
                tool_names_raw = [tool_val]

        trigger = ConstraintTrigger(
            tool_names=set(tool_names_raw) if tool_names_raw else set(),
            keywords=trigger_raw.get("keywords", []),
        )

        return Constraint(
            id=item["id"],
            description=item.get("description", ""),
            trigger=trigger,
            prompt=item.get("prompt", ""),
            correction=item.get("correction", item.get("reminder", "")),
        )

