"""Plan update tool — modify task plan steps and status."""

import re

from flagscale_agent.react.tools.base import Tool, ToolEffect

_EFFECT_PLAN_WRITE = ToolEffect(reads=frozenset({"plan"}), writes=frozenset({"plan"}))

# Pattern to extract integer from strings like "step_1", "step 2", "Step_3", "#4"
_STEP_ID_RE = re.compile(r'(?:step[_\s]?)?#?(\d+)', re.IGNORECASE)


def _parse_step_id(raw) -> int | None:
    """Parse step_id from various LLM formats: 1, "1", "step_1", "step 2", etc."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        # Try direct integer parse first
        try:
            return int(raw)
        except ValueError:
            pass
        # Try regex extraction
        m = _STEP_ID_RE.search(raw)
        if m:
            return int(m.group(1))
    return None


class PlanUpdateTool(Tool):
    name = "plan_update"
    effects = _EFFECT_PLAN_WRITE
    description = (
        "Update the active task plan: mark steps done/skipped, add new steps, "
        "replan, or complete/abandon the plan. Use to track progress as you work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["step_done", "step_doing", "step_skip", "add_steps", "complete", "abandon", "deactivate", "reactivate", "batch"],
                "description": "What to do: step_done/step_doing/step_skip (update a step), add_steps (insert new steps), complete/abandon (finish the plan), deactivate (pause current plan), reactivate (resume a paused plan by id), batch (update multiple steps at once).",
            },
            "step_id": {
                "type": "integer",
                "description": "Step number to update (for step_done/step_doing/step_skip).",
            },
            "notes": {
                "type": "string",
                "description": "Notes or reason for the update.",
            },
            "new_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New step descriptions (for add_steps).",
            },
            "after_step_id": {
                "type": "integer",
                "description": "Insert new steps after this step (for add_steps). Omit to append at end.",
            },
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "integer"},
                        "status": {"type": "string", "enum": ["done", "doing", "skipped"]},
                        "notes": {"type": "string"},
                    },
                    "required": ["step_id", "status"],
                },
                "description": "For batch action: list of step updates to apply at once.",
            },
            "plan_id": {
                "type": "string",
                "description": "Plan ID to reactivate (for reactivate action).",
            },
            "experiment": {
                "type": "string",
                "description": "Experiment name to link to this step (for step_done/step_doing). Automatically appended to the step's experiments list.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]
        experiment = kwargs.get("experiment", "")
        try:
            if action == "step_done":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_done (integer or 'step_N' format)."
                self._plan.update_step(step_id, "done", kwargs.get("notes", ""))
                if experiment:
                    self._plan.link_experiment(step_id, experiment)
            elif action == "step_doing":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_doing (integer or 'step_N' format)."
                self._plan.update_step(step_id, "doing", kwargs.get("notes", ""))
                if experiment:
                    self._plan.link_experiment(step_id, experiment)
            elif action == "step_skip":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_skip (integer or 'step_N' format)."
                self._plan.skip_step(step_id, kwargs.get("notes", ""))
            elif action == "add_steps":
                new_steps = kwargs.get("new_steps", [])
                if not new_steps:
                    return "ERROR: new_steps required for add_steps."
                after = _parse_step_id(kwargs.get("after_step_id"))
                self._plan.add_steps(new_steps, after)
            elif action == "complete":
                self._plan.complete()
            elif action == "abandon":
                self._plan.abandon(kwargs.get("notes", ""))
            elif action == "deactivate":
                plan = self._plan.deactivate()
                if not plan:
                    return "No active plan to deactivate."
                return f"Plan '{plan['title']}' paused."
            elif action == "reactivate":
                plan_id = kwargs.get("plan_id")
                if not plan_id:
                    return "ERROR: plan_id required for reactivate."
                plan = self._plan.reactivate(plan_id)
                if not plan:
                    return f"ERROR: Could not reactivate plan '{plan_id}'. Not found or not paused."
                return self._plan.summary()
            elif action == "batch":
                updates = kwargs.get("updates", [])
                if not updates:
                    return "ERROR: updates required for batch action."
                status_map = {"done": "done", "doing": "doing", "skipped": "skipped"}
                for u in updates:
                    sid = _parse_step_id(u.get("step_id"))
                    status = u.get("status", "")
                    if not sid or status not in status_map:
                        continue
                    if status == "skipped":
                        self._plan.skip_step(sid, u.get("notes", ""))
                    else:
                        self._plan.update_step(sid, status_map[status], u.get("notes", ""))
            else:
                return f"ERROR: Unknown action '{action}'."
            return self._plan.summary()
        except Exception as e:
            return f"ERROR: {e}"
