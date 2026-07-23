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

"""Tool base class with effect declarations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolEffect:
    """Declares what a tool reads, writes, and what side effects it has.

    Used by Guards for permission checks and by the state machine
    for automatic phase transitions.

    Resource types: "filesystem", "memory", "network", "process", "config", "plan"
    Side effect types: "training_launch", "training_kill", "model_modify",
                       "data_delete", "config_modify", "skill_load"
    """

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    side_effects: frozenset[str] = frozenset()

    @property
    def is_read_only(self) -> bool:
        return not self.writes and not self.side_effects

    @property
    def is_write(self) -> bool:
        return bool(self.writes)

    @property
    def touches_filesystem(self) -> bool:
        return "filesystem" in self.reads or "filesystem" in self.writes

    @property
    def touches_network(self) -> bool:
        return "network" in self.reads or "network" in self.writes

    @property
    def touches_process(self) -> bool:
        return "process" in self.reads or "process" in self.writes


# Common effect presets for convenience
EFFECT_READ_FS = ToolEffect(reads=frozenset({"filesystem"}))
EFFECT_WRITE_FS = ToolEffect(reads=frozenset({"filesystem"}), writes=frozenset({"filesystem"}))
EFFECT_READ_MEMORY = ToolEffect(reads=frozenset({"memory"}))
EFFECT_WRITE_MEMORY = ToolEffect(reads=frozenset({"memory"}), writes=frozenset({"memory"}))
EFFECT_NETWORK = ToolEffect(reads=frozenset({"network"}))
EFFECT_SHELL = ToolEffect(
    reads=frozenset({"filesystem", "process", "network"}),
    writes=frozenset({"filesystem", "process"}),
    side_effects=frozenset({"training_launch", "training_kill"}),
)


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    max_result_size: int = 50000
    effects: ToolEffect = ToolEffect()  # Subclasses override

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def _inject_override_param(self, params: dict) -> dict:
        """Inject _override_reason as an optional parameter into schema.

        This allows LLM to bypass guard blocks by providing a reason.
        The field is stripped from tool_args before execute() is called.
        Only injected if the schema has a 'properties' dict.
        """
        if "properties" not in params:
            return params
        # Deep copy to avoid mutating the class-level parameters dict
        import copy
        params = copy.deepcopy(params)
        params["properties"]["_override_reason"] = {
            "description": (
                "If a previous tool call was blocked by a guard, provide a reason "
                "here to override the block and force execution."
            ),
            "type": "string",
        }
        return params

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._inject_override_param(self.parameters),
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._inject_override_param(self.parameters),
        }
