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

"""Plan status tool — show current plan progress."""

from flagscale_agent.react.tools.base import Tool, ToolEffect

_EFFECT_PLAN_READ = ToolEffect(reads=frozenset({"plan"}))


class PlanStatusTool(Tool):
    name = "plan_status"
    effects = _EFFECT_PLAN_READ
    description = (
        "Show the current task plan and progress. "
        "Use at the start of a turn to check where you left off, "
        "or after completing steps to see what's next."
    )
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        return self._plan.summary()
