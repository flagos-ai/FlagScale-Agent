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

"""Guard: validate tool call argument types against JSON schema before execution.

If the LLM produces a malformed argument (e.g. start_line: "[130]" instead of 130),
this guard blocks the call and returns a clear error so the LLM can retry with
correct types. This prevents TypeError crashes in both tool execution and post-guards.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# JSON schema type -> acceptable Python types
_TYPE_MAP = {
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ArgTypeGuard(Guard):
    """Block tool calls with arguments that don't match the declared schema types."""

    name = "arg_type"
    overridable = False  # Type errors are never acceptable

    def __init__(self, tool_registry=None):
        super().__init__()
        self._tool_registry = tool_registry

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._tool_registry is None:
            return None

        try:
            tool = self._tool_registry.get(ctx.tool_name)
        except KeyError:
            return None

        props = tool.parameters.get("properties", {})
        if not props:
            return None

        errors = []
        for key, value in ctx.tool_args.items():
            if value is None:
                continue
            spec = props.get(key, {})
            expected_type = spec.get("type")
            if not expected_type:
                continue
            py_types = _TYPE_MAP.get(expected_type)
            if py_types and not isinstance(value, py_types):
                errors.append(
                    f"  - '{key}': expected {expected_type}, "
                    f"got {type(value).__name__} ({repr(value)!s:.80})"
                )

        if errors:
            msg = (
                f"Invalid argument types for '{ctx.tool_name}'. "
                f"Fix the types and retry:\n" + "\n".join(errors)
            )
            return GuardVerdict.block(msg, reason="arg_type_mismatch", category="arg_type")

        return None
