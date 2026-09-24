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

"""Plan create tool — create a structured task plan."""

import json

from flagscale_agent.react.tools.base import Tool



class PlanCreateTool(Tool):
    name = "plan_create"
    description = (
        "Create a task plan with ordered steps for multi-step work. "
        "Use when about to produce a deliverable or act on a task. "
        "Only one plan can be active at a time."
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
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "acceptance": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["title"],
                        },
                    ]
                },
                "description": "Ordered list of step descriptions (strings) or structured steps (objects with title and optional acceptance).",
            },
            "thinking": {
                "type": "string",
                "description": (
                    "Optional but recommended for hard/open-ended tasks: your CURRENT "
                    "model of the problem — the bottleneck, the load-bearing hypothesis, "
                    "the evidence for it, and a FALSIFIABLE prediction for the next move "
                    "('if I change X, metric should move to ~Y'). This is the HOME for "
                    "deliberate/deep reasoning, distinct from fast action. Leave empty when "
                    "the task is mechanical and fast action is clearly right; fill it when "
                    "the task needs a theory before you act. You can update it later via "
                    "plan_update(thinking=...)."
                ),
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
        
        # Normalize steps: keep dicts as-is, convert strings
        normalized = []
        for s in steps:
            if not s:
                continue
            if isinstance(s, dict):
                # Structured step
                if "title" not in s:
                    return f"ERROR: Step dict must have 'title' field: {s}"
                normalized.append(s)
            else:
                # Plain string step
                normalized.append(str(s))
        
        if not normalized:
            return "ERROR: At least one step is required."
        
        thinking = kwargs.get("thinking", "") or ""

        try:
            plan = self._plan.create(title, normalized, self._session_id, thinking=thinking)
            return f"Plan created.\n\n{self._plan.summary()}"
        except Exception as e:
            return f"ERROR: {e}"
