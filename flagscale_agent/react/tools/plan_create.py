"""Plan create tool — create a structured task plan."""

import json

from flagscale_agent.react.tools.base import Tool, ToolEffect

_EFFECT_PLAN_WRITE = ToolEffect(reads=frozenset({"plan"}), writes=frozenset({"plan"}))


class PlanCreateTool(Tool):
    name = "plan_create"
    effects = _EFFECT_PLAN_WRITE
    description = (
        "Create a task plan with ordered steps for complex multi-step work. "
        "Use when starting environment setup, model porting, training runs, "
        "or any task with 3+ sequential steps. Only one plan can be active at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short plan title, e.g. 'ESPnet LibriSpeech training reproduction'.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of step descriptions.",
            },
        },
        "required": ["title", "steps"],
    }

    def __init__(self, task_plan, session_id: str = ""):
        self._plan = task_plan
        self._session_id = session_id

    def execute(self, **kwargs) -> str:
        title = kwargs["title"]
        steps = kwargs["steps"]

        # Normalize: LLM sometimes returns steps as a JSON-encoded string instead of array
        if isinstance(steps, str):
            steps = steps.strip()
            if steps.startswith("["):
                try:
                    steps = json.loads(steps)
                except (json.JSONDecodeError, ValueError):
                    pass
            # If still a string (single step or unparseable), wrap in list
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]

        if not steps or not isinstance(steps, list):
            return "ERROR: At least one step is required."
        # Ensure all items are strings (not nested structures)
        steps = [str(s) for s in steps if s]
        if not steps:
            return "ERROR: At least one step is required."
        try:
            plan = self._plan.create(title, steps, self._session_id)
            return f"Plan created.\n\n{self._plan.summary()}"
        except Exception as e:
            return f"ERROR: {e}"
