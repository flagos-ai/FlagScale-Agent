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

"""Load skill tool."""

from flagscale_agent.react.tools.base import Tool, ToolEffect


class LoadSkillTool(Tool):
    name = "load_skill"
    effects = ToolEffect(reads=frozenset({"filesystem"}), side_effects=frozenset({"skill_load"}))
    description = "Load a skill by name. Returns the skill content that provides specialized instructions. Extra arguments are passed as parameters to fill placeholders in the skill body."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the skill to load.",
            },
        },
        "required": ["name"],
        "additionalProperties": True,
    }

    def __init__(self, skill_manager):
        self._skill_manager = skill_manager

    def execute(self, **kwargs) -> str:
        name = kwargs.pop("name")
        try:
            content = self._skill_manager.load(name, **kwargs)
            return f"SUCCESS: Skill '{name}' loaded.\n\n{content}"
        except Exception as e:
            return f"ERROR: loading skill '{name}': {e}"
