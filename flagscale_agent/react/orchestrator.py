"""Orchestrator — LLM-driven routing with declarative template config.

1. Single Worker: simple task → one WorkerAgent
2. SubtaskRunner: complex multi-stage task → serial pipeline with DAG
3. BatchRunner: independent experiments → parallel workers

Routing is LLM-first via Judge.route(). When Judge is unavailable,
uses keyword matching against the declarative config as a fallback.

Templates are loaded from:
1. Skill SKILL.md workflow definitions (primary, Phase 5.2)
2. subtask_config.yaml (fallback, legacy)
Skill workflows take priority over YAML templates with the same name.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import yaml

from .config import AgentConfig
from .profile import PROFILES, WorkerProfile
from .scene import PRESETS, ScenePreset
from .agent import WorkerAgent, WorkerResult
from .constraint.cache import ConstraintCache
from .judge import Judge

# Path to subtask config, relative to this file
_SUBTASK_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "subtask_config.yaml")


# ── Subtask config loading ─────────────────────────────────────────────────

def _load_subtask_config() -> dict:
    """Load subtask configuration from YAML. Returns empty dict on failure."""
    try:
        if not os.path.isfile(_SUBTASK_CONFIG_PATH):
            return {}
        with open(_SUBTASK_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return {}


# ── SubtaskDefinition ─────────────────────────────────────────────────────

@dataclass
class SubtaskDefinition:
    """A single subtask in a multi-stage pipeline.

    Supports DAG dependencies: some subtasks can run in parallel
    and others depend on multiple upstream subtasks.
    """
    id: str
    description: str
    profile_name: str
    upstream_keys: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SubtaskDefinition":
        return cls(
            id=d["id"],
            description=d.get("description", d["id"]),
            profile_name=d["profile"],
            upstream_keys=d.get("upstream_keys", []),
            depends_on=d.get("depends_on", []),
        )


# ── SubtaskTemplate ───────────────────────────────────────────────────────

@dataclass
class SubtaskTemplate:
    """A complete subtask pipeline definition."""
    name: str
    description: str
    subtasks: list[SubtaskDefinition]

    def to_routing_desc(self) -> str:
        return f"{self.name}: {self.description}"


# ── SubtaskRunner ─────────────────────────────────────────────────────────

class SubtaskRunner:
    """Executes a subtask DAG with isolated histories.

    NOT a Multi-Agent framework. Each Worker has independent HistoryManager.

    Templates are loaded from:
    1. Skill SKILL.md workflow definitions (primary)
    2. subtask_config.yaml (fallback)
    Skill workflows override YAML templates with the same trigger keywords.
    """

    def __init__(self, config: dict | None = None, skill_manager=None):
        """Initialize with optional config dict and skill_manager.

        If config is None, loads from YAML.
        If skill_manager is provided, also loads workflow templates from Skills.
        """
        cfg = config if config is not None else _load_subtask_config()
        self._cfg = cfg  # cache for use by _pick_template_keyword

        self._templates: dict[str, SubtaskTemplate] = {}
        self._build_templates(cfg)

        # Phase 5.2: Load workflow templates from Skills (override YAML)
        self._skill_manager = skill_manager
        if skill_manager is not None:
            self._build_templates_from_skills(skill_manager)

        self._profile_rules: list[dict] = cfg.get("profile_select", [])
        self._batch_keywords: list[str] = cfg.get("batch_keywords", [])

    def _build_templates(self, cfg: dict):
        """Parse template definitions from config."""
        raw_templates = cfg.get("templates", {})
        for name, tpl_data in raw_templates.items():
            subtasks = [SubtaskDefinition.from_dict(s) for s in tpl_data.get("subtasks", [])]
            self._templates[name] = SubtaskTemplate(
                name=name,
                description=tpl_data.get("description", name),
                subtasks=subtasks,
            )

    def _build_templates_from_skills(self, skill_manager):
        """Build SubtaskTemplates from Skill workflow definitions.

        Skill workflows override YAML templates. Each Skill with a workflow
        field generates a template named after the skill.
        """
        try:
            skills = skill_manager.list_skills()
        except Exception as e:
            return

        for skill_info in skills:
            skill_name = skill_info.get("name", "")
            if not skill_name:
                continue

            workflow = skill_manager.get_workflow(skill_name)
            if not workflow:
                continue

            stages = workflow.get("stages", [])
            if not stages:
                continue

            # Convert workflow stages to SubtaskDefinitions
            subtasks = []
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                subtask = SubtaskDefinition(
                    id=stage.get("id", ""),
                    description=stage.get("description", stage.get("name", "")),
                    profile_name=stage.get("profile", "training-reproduce"),
                    depends_on=stage.get("depends_on", []),
                    upstream_keys=stage.get("upstream_keys", stage.get("depends_on", [])),
                )
                subtasks.append(subtask)

            if not subtasks:
                continue

            # Build trigger config for keyword matching
            trigger = workflow.get("trigger", {})
            description = f"Skill workflow: {skill_name}"

            template = SubtaskTemplate(
                name=skill_name,
                description=description,
                subtasks=subtasks,
            )

            # Store template — overrides YAML if same name
            self._templates[skill_name] = template

            # Also store trigger info in _cfg for keyword matching
            if trigger and isinstance(trigger, dict):
                if "templates" not in self._cfg:
                    self._cfg["templates"] = {}
                self._cfg["templates"][skill_name] = {
                    "description": description,
                    "trigger_on": trigger,
                    "subtasks": stages,
                }

    # ── Template access ────────────────────────────────────────────────────

    def template_names(self) -> list[str]:
        return list(self._templates.keys())

    def template_descriptions(self) -> str:
        """Human-readable list of templates for Judge routing prompt."""
        lines = []
        for name, tpl in self._templates.items():
            lines.append(f"  {name}: {tpl.description}")
        return "\n".join(lines)

    def get_template(self, name: str) -> SubtaskTemplate | None:
        return self._templates.get(name)

    # ── Keyword-based fallback routing ─────────────────────────────────────

    def _pick_template_keyword(self, user_input: str) -> str | None:
        """Pick template by matching trigger_on keywords from config."""
        # Use cached config from it time, not disk reload
        raw_templates = self._cfg.get("templates", {}) if hasattr(self, "_cfg") else {}
        text_lower = user_input.lower()

        for name, tpl_data in raw_templates.items():
            trigger = tpl_data.get("trigger_on", {})

            # Check keyword pairs (both must be present in the same input)
            pairs = trigger.get("keywords_in_same_input", [])
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                if pair[0].lower() in text_lower and pair[1].lower() in text_lower:
                    return name

            # Check standalone keywords
            keywords = trigger.get("keywords", [])
            if keywords and any(k.lower() in text_lower for k in keywords):
                return name

        # No keywords matched — return None (caller should not assume a template)
        return None

    def _pick_profile_keyword(self, user_input: str) -> str:
        """Pick profile by matching ordered rules from config."""
        text_lower = user_input.lower()

        for rule in self._profile_rules:
            keywords = rule.get("keywords", [])

            # Check additional keywords (all must match to select this profile)
            extra = rule.get("additional_keywords", [])
            if extra and not any(k.lower() in text_lower for k in extra):
                continue

            # Check primary keywords
            if not keywords:
                # No keywords = default catch-all
                return rule.get("profile", "general")

            if any(k.lower() in text_lower for k in keywords):
                return rule.get("profile", "general")

        return "general"

    def _is_batch_keyword(self, user_input: str) -> bool:
        """Check if user input suggests batch comparison (keyword fallback)."""
        return any(k in user_input for k in self._batch_keywords)

    # ── DAG execution ──────────────────────────────────────────────────────

    @staticmethod
    def _topological_batches(
        subtasks: list[SubtaskDefinition],
    ) -> list[list[SubtaskDefinition]]:
        """Group subtasks into batches that can run in parallel.

        Each batch contains subtasks whose dependencies are all satisfied.
        Within a batch, subtasks are independent and can run concurrently.
        """
        remaining = {s.id: s for s in subtasks}
        completed: set[str] = set()
        batches: list[list[SubtaskDefinition]] = []

        while remaining:
            ready = [
                s for s in remaining.values()
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                unresolved = {s.id: s.depends_on for s in remaining.values()}
                raise ValueError(f"Unresolvable DAG dependencies: {unresolved}")
            batches.append(ready)
            for s in ready:
                del remaining[s.id]
                completed.add(s.id)

        return batches

    def run(
        self,
        template_name: str,
        user_input: str,
        orchestrator: "Orchestrator",
    ) -> WorkerResult:
        """Execute subtask DAG with topological batching.

        Propagates constraints across stages: skills loaded in earlier stages
        have their constraints registered in later stages.
        """
        template = self._templates.get(template_name)
        if template is None:
            return WorkerResult(status="failed",
                summary=f"Unknown template: {template_name}")

        subtasks = template.subtasks
        batches = self._topological_batches(subtasks)
        upstream: dict[str, str] = {}
        # Track skills loaded across all stages for constraint propagation
        accumulated_skills: set[str] = set()

        for batch in batches:
            if len(batch) == 1:
                sub = batch[0]
                context = self._build_upstream_summary(sub.upstream_keys, upstream)
                worker = orchestrator._create_worker(sub.profile_name)
                orchestrator._propagate_constraints(worker, accumulated_skills)
                task = self._build_task(sub.description, user_input, context)
                try:
                    result = worker.execute(task)
                except KeyboardInterrupt:
                    return WorkerResult(
                        status="interrupted",
                        summary=f"Stage '{sub.id}' interrupted by user (Ctrl+C)",
                    )
                accumulated_skills |= worker._loaded_skills
                if result.interrupted:
                    return WorkerResult(
                        status="interrupted",
                        summary=f"Stage '{sub.id}' interrupted: {result.summary}",
                        artifacts=upstream,
                    )
                if result.status == "failed":
                    return result
                upstream.update(result.artifacts)
                upstream[sub.id] = result.summary
            else:
                def _run_subtask(sub):
                    context = self._build_upstream_summary(sub.upstream_keys, upstream)
                    worker = orchestrator._create_worker(sub.profile_name)
                    orchestrator._propagate_constraints(worker, accumulated_skills)
                    task = self._build_task(sub.description, user_input, context)
                    return sub.id, worker.execute(task), worker._loaded_skills

                batch_results: dict[str, WorkerResult] = {}
                batch_skills: set[str] = set()
                try:
                    with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as pool:
                        futures = {pool.submit(_run_subtask, s): s for s in batch}
                        for future in as_completed(futures):
                            sub_id, result, skills = future.result()
                            batch_results[sub_id] = result
                            batch_skills |= skills
                except KeyboardInterrupt:
                    return WorkerResult(
                        status="interrupted",
                        summary="Parallel batch interrupted by user (Ctrl+C)",
                        artifacts=upstream,
                    )

                accumulated_skills |= batch_skills
                for sub in batch:
                    result = batch_results.get(sub.id)
                    if result is None:
                        return WorkerResult(status="failed",
                            summary=f"Subtask {sub.id} returned no result")
                    if result.interrupted:
                        return WorkerResult(
                            status="interrupted",
                            summary=f"Stage '{sub.id}' interrupted: {result.summary}",
                            artifacts=upstream,
                        )
                    if result.status == "failed":
                        return result
                    upstream.update(result.artifacts)
                    upstream[sub.id] = result.summary

        return WorkerResult(
            status="success",
            summary="All subtasks completed",
            artifacts=upstream,
        )

    @staticmethod
    def _build_upstream_summary(keys: list[str], upstream: dict) -> str:
        """Build concise summary from upstream results. NOT full history.

        keys can be stage IDs (e.g. "env_setup"), which map to stored summaries,
        or semantic artifact keys (e.g. "env_path") which map to WorkerResult.artifacts.
        """
        lines = ["Previous stage results:"]
        found_any = False
        for k in keys:
            if k in upstream:
                val = upstream[k]
                lines.append(f"  {k}: {str(val)[:300]}")
                found_any = True
        return "\n".join(lines) if found_any else ""

    @staticmethod
    def _build_task(description: str, user_input: str, context: str) -> str:
        """Build the task prompt for a subtask Worker."""
        parts = [description]
        if context:
            parts.append(f"\nContext from previous stages:\n{context}")
        parts.append(f"\nOriginal request: {user_input}")
        return "\n".join(parts)


# ── BatchRunner ────────────────────────────────────────────────────────────

class BatchRunner:
    """Execute same-type work with different parameters in parallel.

    NOT multi-agent — these are independent workers with isolated histories.
    Each uses the same WorkerProfile but different task descriptions.
    """

    def run(
        self,
        profile_name: str,
        tasks: list[str],
        orchestrator: "Orchestrator",
    ) -> list[WorkerResult]:
        """Run multiple independent workers in parallel."""

        def _run_one(task: str) -> WorkerResult:
            worker = orchestrator._create_worker(profile_name)
            return worker.execute(task)

        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
            ordered = list(pool.map(_run_one, tasks))
        return ordered

    @staticmethod
    def summarize(results: list[WorkerResult]) -> str:
        """Compare results across parallel runs."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"  Run {i}: {r.status} — {r.summary}")
        return "\n".join(lines)


# ── Orchestrator ───────────────────────────────────────────────────────────

class Orchestrator:
    """Entry point: routes user requests to the right execution mode.

    LLM-first routing via Judge.route(), falling back to keyword-based
    routing from subtask_config.yaml when Judge is unavailable.

    All regex-based routing has been removed.

    Infrastructure components are injected at construction time and
    shared across all workers.
    """

    def __init__(
        self,
        provider=None,
        tool_registry=None,
        skill_manager=None,
        session_memory=None,
        task_plan=None,
        experiment_manager=None,
        judge=None,
        config: AgentConfig | None = None,
    ):
        self.profiles: dict[str, WorkerProfile] = PROFILES
        self.presets: dict[str, ScenePreset] = PRESETS
        self.subtask_runner = SubtaskRunner(skill_manager=skill_manager)
        self.batch_runner = BatchRunner()
        self.scene: ScenePreset | None = None

        # Shared infrastructure
        self.provider = provider
        self.tool_registry = tool_registry
        self.skill_manager = skill_manager
        self.session_memory = session_memory
        self.task_plan = task_plan
        self.experiment_manager = experiment_manager
        self.judge = judge
        self.config = config or AgentConfig.auto_load()

        # Shared constraint cache for all workers
        from flagscale_agent.react.paths import get_sessions_root
        sessions_root = self.config.session_dir or get_sessions_root()
        self._constraint_cache = ConstraintCache(sessions_root)

    def handle(self, user_input: str) -> str:
        """Handle a user request. Route via LLM (primary) or keywords (fallback)."""
        # 1. Detect scene (env-based, no LLM needed)
        self.scene = self._refine_scene(user_input)

        # 2. Try LLM-based routing first
        route = self._route_via_llm(user_input)
        if route is not None:
            return self._dispatch_route(route, user_input)

        # 3. Fallback: keyword-based routing from config
        return self._route_via_keyword(user_input)

    # ── Public routing API for WorkerAgent interactive loop ─────────────────

    def route(self, user_input: str) -> dict:
        """Public routing method — returns route dict for external dispatch.

        Combines LLM-based routing (primary) and keyword-based fallback.
        Returns a dict with keys: mode, profile, template, batch_tasks.

        Used by WorkerAgent.run() to decide execution path in interactive mode.
        """
        # 1. Detect scene (env-based, no LLM needed)
        self.scene = self._refine_scene(user_input)

        # 2. Try LLM-based routing first
        route = self._route_via_llm(user_input)
        if route is not None:
            return route

        # 3. Fallback: keyword-based routing from config
        sr = self.subtask_runner

        # Check for batch
        if sr._is_batch_keyword(user_input):
            tasks = self._extract_batch_tasks_by_separator(user_input)
            if len(tasks) >= 2:
                return {
                    "mode": "batch",
                    "profile": sr._pick_profile_keyword(user_input),
                    "template": "",
                    "batch_tasks": tasks,
                }

        # Check for multi-subtask
        template = sr._pick_template_keyword(user_input)
        if template is not None and template in self.subtask_runner._templates:
            has_pair_match = self._has_template_keyword_pair(user_input, template)
            if has_pair_match:
                return {
                    "mode": "subtask",
                    "profile": "",
                    "template": template,
                    "batch_tasks": [],
                }

        # Single worker
        return {
            "mode": "single",
            "profile": sr._pick_profile_keyword(user_input),
            "template": "",
            "batch_tasks": [],
        }

    def run_single_worker(self, route: dict, user_input: str) -> WorkerResult:
        """Run a single WorkerAgent with the given route and return result.

        Creates a fresh worker for this specific task.
        Does NOT modify any shared state — caller owns the history.
        """
        profile = route["profile"] or self.subtask_runner._pick_profile_keyword(user_input)
        worker = self._create_worker(profile)
        return worker.execute(user_input)

    def run_subtask_interactive(
        self,
        route: dict,
        user_input: str,
    ) -> tuple[str, list[WorkerResult]]:
        """Execute subtask DAG in serial, interactive mode.

        Each stage runs one at a time with progress display.
        Returns (final_summary, list_of_per_stage_results).

        Supports both YAML templates (route["template"]) and LLM-generated
        dynamic stages (route["dynamic_stages"]).

        Constraint propagation: skills loaded in earlier stages propagate their
        constraints to later stages, ensuring safety rules aren't lost across
        stage boundaries.
        """
        subtasks = self._build_subtask_definitions(route, user_input)
        if not subtasks:
            return (f"No stages to execute for: {user_input}", [])

        batches = self.subtask_runner._topological_batches(subtasks)
        upstream: dict[str, str] = {}
        stage_results: list[WorkerResult] = []
        # Track skills loaded across all stages for constraint propagation
        accumulated_skills: set[str] = set()

        stage_idx = 0
        total_stages = len(subtasks)

        for batch in batches:
            for sub in batch:
                stage_idx += 1
                context = self.subtask_runner._build_upstream_summary(
                    sub.upstream_keys, upstream
                )
                worker = self._create_worker(sub.profile_name)

                # Propagate constraints from skills loaded in earlier stages
                self._propagate_constraints(worker, accumulated_skills)

                task = self.subtask_runner._build_task(
                    sub.description, user_input, context
                )

                try:
                    result = worker.execute(task)
                except KeyboardInterrupt:
                    return (
                        f"Stage {stage_idx}/{total_stages} ({sub.id}) interrupted by user.",
                        stage_results,
                    )
                stage_results.append(result)

                # Collect skills loaded during this stage for propagation
                accumulated_skills |= worker._loaded_skills

                if result.status == "failed":
                    upstream.update(result.artifacts)
                    upstream[sub.id] = result.summary
                    return (
                        f"Stage {stage_idx}/{total_stages} ({sub.id}) failed: {result.summary[:200]}",
                        stage_results,
                    )
                if result.interrupted:
                    upstream.update(result.artifacts)
                    upstream[sub.id] = result.summary
                    return (
                        f"Stage {stage_idx}/{total_stages} ({sub.id}) interrupted by user.",
                        stage_results,
                    )
                upstream.update(result.artifacts)
                upstream[sub.id] = result.summary

        return ("All subtasks completed", stage_results)

    def _propagate_constraints(self, worker: "WorkerAgent", skills: set[str]):
        """Propagate constraints from previously-loaded skills to a new worker.

        Only registers constraints (not full skill content) — the worker doesn't
        get the skill's system prompt, just its safety constraints. This ensures
        constraints like "must build from source" survive across stage boundaries.
        """
        for skill_name in skills:
            if skill_name in worker._loaded_skills:
                continue  # Already loaded via profile
            if skill_name in worker._skill_guards_registered:
                continue  # Already registered
            try:
                worker._register_skill_guards(skill_name)
            except Exception:
                pass

    def _build_subtask_definitions(
        self, route: dict, user_input: str
    ) -> list["SubtaskDefinition"]:
        """Build SubtaskDefinitions from YAML template or LLM dynamic stages."""
        template_name = route.get("template", "")
        dynamic_stages = route.get("dynamic_stages", [])

        # Prefer explicit template
        if template_name:
            template = self.subtask_runner._templates.get(template_name)
            if template:
                return template.subtasks

        # Fallback: keyword template matching
        kw_template = self.subtask_runner._pick_template_keyword(user_input)
        if kw_template and kw_template in self.subtask_runner._templates:
            return self.subtask_runner._templates[kw_template].subtasks

        # LLM-generated dynamic stages
        if dynamic_stages:
            return [
                SubtaskDefinition(
                    id=s["id"],
                    description=s.get("description", s["id"]),
                    profile_name=s.get("profile", "training-reproduce"),
                    depends_on=s.get("depends_on", []),
                    upstream_keys=s.get("upstream_keys", []),
                )
                for s in dynamic_stages
            ]

        return []

    def run_batch_interactive(
        self,
        route: dict,
        user_input: str,
    ) -> list[WorkerResult]:
        """Execute batch comparison in parallel, returning per-task results.

        Caller controls display and interactivity.
        """
        profile = route["profile"] or self.subtask_runner._pick_profile_keyword(user_input)
        return self.batch_runner.run(profile, route["batch_tasks"], self)

    # ── LLM-based routing (primary) ───────────────────────────────────────

    def _route_via_llm(self, user_input: str) -> dict | None:
        """Route via Judge. Returns route dict or None if Judge unavailable."""
        judge = self.judge
        if judge is None:
            if self.provider is None:
                return None
            judge = Judge(self.provider)

        profiles_str = json.dumps(list(self.profiles.keys()), ensure_ascii=False)
        templates_str = self.subtask_runner.template_descriptions()

        try:
            result, source = judge.route(user_input, profiles_str, templates_str)
        except Exception as e:
            return None

        if source in ("unavailable", "default"):
            return None

        mode = result.get("mode", "single")
        if mode not in ("single", "subtask", "batch"):
            return None

        # Validate profile name
        profile = result.get("profile", "")
        if profile and profile not in self.profiles:
            profile = ""

        # Validate template name or dynamic_stages for subtask mode
        template = result.get("template", "")
        dynamic_stages = result.get("dynamic_stages", [])

        if mode == "subtask":
            if template and template in self.subtask_runner._templates:
                # Reusing an existing YAML template
                pass
            elif isinstance(dynamic_stages, list) and len(dynamic_stages) >= 2:
                # LLM generated custom stages — validate each
                for s in dynamic_stages:
                    if not isinstance(s, dict):
                        return None
                    if not s.get("id") or not s.get("description"):
                        return None
                    stage_profile = s.get("profile", "")
                    if stage_profile and stage_profile not in self.profiles:
                        return None
                template = ""  # explicitly empty = dynamic
            else:
                return None

        # Validate batch tasks
        batch_tasks = result.get("batch_tasks", [])
        if mode == "batch" and not (isinstance(batch_tasks, list) and len(batch_tasks) >= 2):
            return None

        return {
            "mode": mode,
            "profile": profile,
            "template": template,
            "batch_tasks": batch_tasks,
            "dynamic_stages": dynamic_stages,
            "reason": result.get("reason", ""),
        }

    def _dispatch_route(self, route: dict, user_input: str) -> str:
        """Execute the route decision."""
        mode = route["mode"]

        if mode == "batch":
            profile = route["profile"] or self.subtask_runner._pick_profile_keyword(user_input)
            results = self.batch_runner.run(profile, route["batch_tasks"], self)
            return self._format_batch_response(results)

        if mode == "subtask":
            template = route["template"] or self.subtask_runner._pick_template_keyword(user_input)
            if template is None:
                return f"\u2717 [failed] No subtask template matched"
            result = self.subtask_runner.run(template, user_input, self)
            return self._format_response(result)

        # mode == "single"
        profile = route["profile"] or self.subtask_runner._pick_profile_keyword(user_input)
        worker = self._create_worker(profile)
        result = worker.execute(user_input)
        return self._format_response(result)

    # ── Keyword-based routing (fallback when Judge unavailable) ────────────

    def _route_via_keyword(self, user_input: str) -> str:
        """Fallback routing using keyword config from subtask_config.yaml.

        No regex — pure substring matching against declarative keyword lists.
        """
        sr = self.subtask_runner

        # Check for batch
        if sr._is_batch_keyword(user_input):
            tasks = self._extract_batch_tasks_by_separator(user_input)
            if len(tasks) >= 2:
                profile = sr._pick_profile_keyword(user_input)
                results = self.batch_runner.run(profile, tasks, self)
                return self._format_batch_response(results)

        # Check for multi-subtask
        template = sr._pick_template_keyword(user_input)
        if template is not None and template in self.subtask_runner._templates:
            has_pair_match = self._has_template_keyword_pair(user_input, template)
            if has_pair_match:
                result = self.subtask_runner.run(template, user_input, self)
                return self._format_response(result)

        # Single worker
        profile = sr._pick_profile_keyword(user_input)
        worker = self._create_worker(profile)
        result = worker.execute(user_input)
        return self._format_response(result)

    def _has_template_keyword_pair(self, user_input: str, template_name: str) -> bool:
        """Check if user_input matches a keyword_pair for the given template."""
        raw = self._cfg.get("templates", {}).get(template_name, {})
        trigger = raw.get("trigger_on", {})
        pairs = trigger.get("keywords_in_same_input", [])
        text_lower = user_input.lower()
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                if pair[0].lower() in text_lower and pair[1].lower() in text_lower:
                    return True
        return False

    @staticmethod
    def _extract_batch_tasks_by_separator(user_input: str) -> list[str]:
        """Extract individual task descriptions by splitting on separators."""
        # Split on common separators (these separators are language-specific
        # and would ideally come from config too)
        parts = []
        current = []
        for ch in user_input:
            if ch in ";；":
                if current:
                    parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())

        if len(parts) < 2:
            # Try splitting on conjunction words
            for sep in ["和", "跟", "对比", "分别"]:
                if sep in user_input:
                    parts = [p.strip() for p in user_input.split(sep) if p.strip()]
                    if len(parts) >= 2:
                        break

        return [p for p in parts if p]

    # ── Scene detection ───────────────────────────────────────────────────

    def _refine_scene(self, user_input: str) -> ScenePreset:
        """Auto-detect scene from environment (no regex)."""
        return ScenePreset.from_env_and_input(user_input=user_input)

    def _create_worker(self, profile_name: str) -> WorkerAgent:
        """Create a fresh WorkerAgent with shared infrastructure.

        Context isolation: each worker gets its OWN HistoryManager.
        Profile skills are preloaded (semantic routing already decided them).
        """
        profile = self.profiles[profile_name]

        constraints = set(profile.scene_constraints)
        if self.scene:
            constraints |= self.scene.constraints

        worker_scene = ScenePreset(
            name=profile.name,
            mode="training",
            chip_type=self.scene.chip_type if self.scene else "nvidia",
            chip_vendor_sdk=self.scene.chip_vendor_sdk if self.scene else "cuda",
            target_framework="megatron-core",
            source_framework="",
            default_precision="bf16",
            network_topology="single_node",
            constraints=constraints,
        )

        worker = WorkerAgent(
            config=self.config,
            scene=worker_scene,
            _provider=self.provider,
            _tool_registry=self.tool_registry,
            _skill_manager=self.skill_manager,
            _session_memory=self.session_memory,
            _task_plan=self.task_plan,
            _experiment_manager=self.experiment_manager,
            _constraint_cache=self._constraint_cache,
        )

        # Preload profile skills (already decided by Judge routing)
        for skill_name in profile.skills:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    worker._loaded_skills.add(skill_name)
                    worker._active_skill_content[skill_name] = content
                    worker._apply_skill_effects(skill_name)
                    worker._register_skill_guards(skill_name)
            except Exception:
                pass
        if profile.skills:
            skill_map = {n: worker._active_skill_content.get(n, "")
                         for n in profile.skills if n in worker._active_skill_content}
            if skill_map:
                worker._batch_extract_and_rebuild(skill_map)

        return worker

    @staticmethod
    def _format_response(result: WorkerResult) -> str:
        """Format WorkerResult for user display."""
        if result.status == "success":
            return f"\u2713 {result.summary}"
        return f"\u2717 [{result.status}] {result.summary}"

    @staticmethod
    def _format_batch_response(results: list[WorkerResult]) -> str:
        """Format batch results for user display."""
        summary = BatchRunner.summarize(results)
        return f"Batch comparison:\n{summary}"
