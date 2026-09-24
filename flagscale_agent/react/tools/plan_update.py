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

"""Plan update tool — modify task plan steps and status."""

from __future__ import annotations

import re

from flagscale_agent.react.tools.base import Tool

_THINKING_ECHO = (
    "Model recorded — it is now your working theory. What does it PREDICT for "
    "your next move, and what is the cheapest observation that could kill it? "
    "Run that."
)


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
    description = (
        "Update the active task plan: mark steps done/skipped, add new steps, "
        "replan, or complete/abandon the plan. Use to track progress as you work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["step_done", "step_doing", "step_skip", "add_steps", "update_acceptance", "complete", "abandon", "deactivate", "reactivate", "batch", "set_thinking"],
                "description": "What to do: step_done/step_doing/step_skip (update a step), add_steps (insert new steps), update_acceptance (modify step acceptance criteria), complete/abandon (finish the plan), deactivate (pause current plan), reactivate (resume a paused plan by id), batch (update multiple steps at once), set_thinking (rewrite the plan-level problem model — pass the new model in the 'thinking' field).",
            },
            "step_id": {
                "type": "integer",
                "description": "Step number to update (for step_done/step_doing/step_skip/update_acceptance).",
            },
            "notes": {
                "type": "string",
                "description": "Append a note to this step (scratchpad). Use freely to record: attempts, failures, key decisions, things to remember. Each call appends a new line — previous notes are preserved.",
            },
            "verification": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verification evidence for step_done: what was checked, test results, output samples.",
            },
            "acceptance": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New acceptance criteria (for update_acceptance action).",
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
            "thinking": {
                "type": "string",
                "description": (
                    "Rewrite the plan-level problem MODEL — your current theory of the "
                    "task: the bottleneck, the load-bearing hypothesis, the evidence, and "
                    "a FALSIFIABLE prediction for the next move ('if I change X, metric "
                    "should reach ~Y'). Overwrites the previous model (it is the CURRENT "
                    "understanding, not a log). Can be passed with action='set_thinking' "
                    "on its own, or alongside any other action (e.g. step_done + an updated "
                    "model). This is the HOME for deep/deliberate reasoning — when you catch "
                    "yourself about to try yet another quick tweak, write the model here "
                    "first: what is actually limiting the result, and what does your next "
                    "move PREDICT? A shallow retry that cannot state a new prediction is the "
                    "signal to stop and rebuild the model, not act again."
                ),
            },
            "_override_reason": {
                "type": "string",
                "description": "Override a guard block with a justification (min 5 chars). Some guard checks (step_done premise re-check, batch marking a step done, task completion re-check) block until you re-issue the same call with this field explaining why proceeding is justified. The reason is recorded, not content-checked.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]
        thinking = kwargs.get("thinking")
        try:
            if action == "set_thinking":
                if not thinking or not str(thinking).strip():
                    return "ERROR: thinking text required for set_thinking."
                self._plan.set_thinking(str(thinking))
                return self._plan.summary() + "\n\n" + _THINKING_ECHO
            if action == "step_done":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_done (integer or 'step_N' format)."
                verification = kwargs.get("verification", [])
                self._plan.update_step(step_id, "done", kwargs.get("notes", ""), verification=verification)
            elif action == "step_doing":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_doing (integer or 'step_N' format)."
                self._plan.update_step(step_id, "doing", kwargs.get("notes", ""))
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
            elif action == "update_acceptance":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for update_acceptance."
                acceptance = kwargs.get("acceptance", [])
                if not acceptance:
                    return "ERROR: acceptance required for update_acceptance."
                self._plan.update_acceptance(step_id, acceptance)
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
            # thinking may accompany any action (e.g. step_done + updated model).
            # A recorded model earns its keep only by generating the next
            # falsifiable observation — echo the challenge when one is recorded.
            echo = ""
            if thinking and str(thinking).strip():
                self._plan.set_thinking(str(thinking))
                echo = "\n\n" + _THINKING_ECHO
            return self._plan.summary() + echo
        except Exception as e:
            return f"ERROR: {e}"
