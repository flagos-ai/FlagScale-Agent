"""Workspace experiment tool — manage per-experiment YAML files."""

from flagscale_agent.react.tools.base import Tool, EFFECT_WRITE_FS


class WorkspaceExperimentTool(Tool):
    name = "workspace_experiment"
    effects = EFFECT_WRITE_FS
    description = (
        "Manage experiment records. Each experiment has its own YAML file.\n"
        "Experiment level: name, purpose, hypothesis, status, root_cause, learnings.\n"
        "Attempt level: change, hardware (gpus/gpu_type), config (model/tp/dp/batch/etc), "
        "output_dir, result.\n"
        "Flow: create → add_attempt (before EACH launch) → update_last_attempt (after result) → finalize."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "add_attempt", "update_last_attempt", "finalize",
                         "read", "list", "compare", "diff_last_attempts"],
                "description": "Action to perform.",
            },
            "name": {"type": "string", "description": "Experiment name (required for all actions except list). For compare: baseline experiment."},
            "name2": {"type": "string", "description": "Second experiment name (for compare action only — the experiment to compare against baseline)."},
            "purpose": {"type": "string", "description": "Why this experiment exists (for create)."},
            "hypothesis": {"type": "string", "description": "What you expect to happen and why (for create)."},
            "change": {
                "type": "string",
                "description": "What changed in this attempt vs previous (for add_attempt). First attempt: 'initial run'.",
            },
            "hardware": {
                "type": "object",
                "description": (
                    "Hardware for THIS attempt: {gpus: int, gpu_type: str, driver?: str, cuda?: str}. "
                    "Required for add_attempt."
                ),
            },
            "config": {
                "type": "object",
                "description": (
                    "Training config for THIS attempt. Structured training parameters only: "
                    "{model, tp, dp, pp?, global_batch_size, seq_length, train_iters, precision, ...}. "
                    "Do NOT put descriptions/reasons here — use 'change' for that."
                ),
            },
            "output_dir": {
                "type": "string",
                "description": "Unique output directory for THIS attempt's results/logs (for add_attempt).",
            },
            "result": {"type": "string", "description": "Attempt result (for update_last_attempt)."},
            "status": {
                "type": "string",
                "enum": ["running", "failed", "completed", "paused"],
                "description": "Final status (for finalize).",
            },
            "root_cause": {"type": "string", "description": "Root cause of failure (for finalize, if failed)."},
            "learnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key learnings from this experiment (for finalize).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, experiment_manager, task_plan=None):
        self._manager = experiment_manager
        self._task_plan = task_plan

    def _auto_link_to_plan(self, experiment_name: str):
        """Auto-link experiment to the current 'doing' step in the active plan."""
        if not self._task_plan:
            return
        plan = self._task_plan.get_active()
        if not plan:
            return
        for step in plan["steps"]:
            if step["status"] == "doing":
                self._task_plan.link_experiment(step["id"], experiment_name)
                break

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]

        if action == "create":
            name = kwargs.get("name")
            purpose = kwargs.get("purpose", "")
            hypothesis = kwargs.get("hypothesis", "")
            missing = []
            if not name:
                missing.append("name")
            if not purpose:
                missing.append("purpose")
            if missing:
                return f"ERROR: Required fields missing: {', '.join(missing)}"
            result = self._manager.create(
                name=name, purpose=purpose, hypothesis=hypothesis,
            )
            self._auto_link_to_plan(name)
            return result

        elif action == "add_attempt":
            name = kwargs.get("name")
            change = kwargs.get("change", "")
            hardware = kwargs.get("hardware", {})
            config = kwargs.get("config", {})
            output_dir = kwargs.get("output_dir", "")
            if not name:
                return "ERROR: name required for add_attempt."
            if not change:
                return "ERROR: change required — describe what's different in this attempt."
            result = self._manager.add_attempt(
                name, change, hardware=hardware, config=config, output_dir=output_dir)
            self._auto_link_to_plan(name)
            return result

        elif action == "update_last_attempt":
            name = kwargs.get("name")
            result = kwargs.get("result", "")
            if not name:
                return "ERROR: name required for update_last_attempt."
            if not result:
                return "ERROR: result required — what happened?"
            return self._manager.update_last_attempt(name, result)

        elif action == "finalize":
            name = kwargs.get("name")
            status = kwargs.get("status", "completed")
            root_cause = kwargs.get("root_cause", "")
            learnings = kwargs.get("learnings", [])
            if not name:
                return "ERROR: name required for finalize."
            if not learnings:
                return "ERROR: finalize requires non-empty 'learnings' list."
            if status == "failed" and not root_cause:
                return "ERROR: finalize with status='failed' requires 'root_cause'."
            result = self._manager.finalize(name, status, root_cause=root_cause, learnings=learnings)
            # Auto-update plan steps linked to this experiment
            if self._task_plan and status == "completed":
                plan = self._task_plan.get_active()
                if plan:
                    for step in plan.get("steps", []):
                        if name in step.get("experiments", []) and step.get("status") != "done":
                            self._task_plan.update_step(step["id"], "done",
                                notes=f"Auto-marked done: experiment '{name}' completed")
            return result

        elif action == "read":
            name = kwargs.get("name")
            if not name:
                return "ERROR: name required for read."
            exp = self._manager.read(name)
            if not exp:
                return f"Experiment '{name}' not found."
            import yaml
            return yaml.dump(exp, allow_unicode=True, default_flow_style=False, sort_keys=False)

        elif action == "list":
            experiments = self._manager.list()
            if not experiments:
                return "(no experiments yet)"
            lines = [f"- {e['name']} ({e['status']}, {e['attempts']} attempts)" for e in experiments]
            return "\n".join(lines)

        elif action == "compare":
            name1 = kwargs.get("name")  # First experiment name
            name2 = kwargs.get("name2")
            if not name1 or not name2:
                return "ERROR: Both 'name' (baseline) and 'name2' (experiment) required for compare."
            result = self._manager.compare(name1, name2)
            return result["summary"]

        elif action == "diff_last_attempts":
            name = kwargs.get("name")
            if not name:
                return "ERROR: name required for diff_last_attempts."
            result = self._manager.diff_last_attempts(name)
            return result["summary"]

        return f"ERROR: Unknown action '{action}'."
