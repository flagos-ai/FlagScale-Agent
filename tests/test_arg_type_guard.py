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

"""Unit tests for ArgTypeGuard."""

import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.arg_type import ArgTypeGuard


def make_ctx(tool_name="", tool_args=None):
    """Helper to create a GuardContext for testing."""
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = ""
    ctx.classify_fn = None
    ctx.current_experiment_name = ""
    ctx.experiment_diff_fn = None
    return ctx


def make_registry(tools: dict):
    """Create a mock tool registry from a dict of {name: {properties: ...}}."""
    registry = MagicMock()

    def get_tool(name):
        if name not in tools:
            raise KeyError(name)
        tool = MagicMock()
        tool.parameters = tools[name]
        return tool

    registry.get = get_tool
    return registry


class TestArgTypeGuardBasics:
    """Test basic guard behavior and edge cases."""

    def test_no_registry_returns_none(self):
        guard = ArgTypeGuard(tool_registry=None)
        ctx = make_ctx("shell", {"command": "ls"})
        assert guard.check_pre(ctx) is None

    def test_unknown_tool_returns_none(self):
        registry = make_registry({})
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("nonexistent_tool", {"arg": "val"})
        assert guard.check_pre(ctx) is None

    def test_tool_with_no_properties_returns_none(self):
        registry = make_registry({"my_tool": {}})
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("my_tool", {"arg": "val"})
        assert guard.check_pre(ctx) is None

    def test_tool_with_empty_properties_returns_none(self):
        registry = make_registry({"my_tool": {"properties": {}}})
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("my_tool", {"arg": "val"})
        assert guard.check_pre(ctx) is None

    def test_none_value_skipped(self):
        registry = make_registry({
            "my_tool": {"properties": {"count": {"type": "integer"}}}
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("my_tool", {"count": None})
        assert guard.check_pre(ctx) is None

    def test_arg_not_in_schema_skipped(self):
        registry = make_registry({
            "my_tool": {"properties": {"name": {"type": "string"}}}
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("my_tool", {"unknown_arg": 123})
        assert guard.check_pre(ctx) is None

    def test_no_type_in_spec_skipped(self):
        registry = make_registry({
            "my_tool": {"properties": {"data": {"description": "some data"}}}
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("my_tool", {"data": [1, 2, 3]})
        assert guard.check_pre(ctx) is None


class TestArgTypeGuardTypeChecks:
    """Test type validation for each JSON schema type."""

    @pytest.fixture
    def guard(self):
        registry = make_registry({
            "test_tool": {
                "properties": {
                    "count": {"type": "integer"},
                    "rate": {"type": "number"},
                    "name": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "items": {"type": "array"},
                    "config": {"type": "object"},
                }
            }
        })
        return ArgTypeGuard(tool_registry=registry)

    # --- integer ---
    def test_integer_valid(self, guard):
        ctx = make_ctx("test_tool", {"count": 42})
        assert guard.check_pre(ctx) is None

    def test_integer_invalid_string(self, guard):
        ctx = make_ctx("test_tool", {"count": "42"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "count" in verdict.message
        assert "integer" in verdict.message

    def test_integer_invalid_float(self, guard):
        ctx = make_ctx("test_tool", {"count": 3.14})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_integer_invalid_list(self, guard):
        ctx = make_ctx("test_tool", {"count": [130]})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "count" in verdict.message

    # --- number ---
    def test_number_valid_int(self, guard):
        ctx = make_ctx("test_tool", {"rate": 10})
        assert guard.check_pre(ctx) is None

    def test_number_valid_float(self, guard):
        ctx = make_ctx("test_tool", {"rate": 3.14})
        assert guard.check_pre(ctx) is None

    def test_number_invalid_string(self, guard):
        ctx = make_ctx("test_tool", {"rate": "3.14"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "number" in verdict.message

    # --- string ---
    def test_string_valid(self, guard):
        ctx = make_ctx("test_tool", {"name": "hello"})
        assert guard.check_pre(ctx) is None

    def test_string_invalid_int(self, guard):
        ctx = make_ctx("test_tool", {"name": 123})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "string" in verdict.message

    # --- boolean ---
    def test_boolean_valid(self, guard):
        ctx = make_ctx("test_tool", {"enabled": True})
        assert guard.check_pre(ctx) is None

    def test_boolean_invalid_string(self, guard):
        ctx = make_ctx("test_tool", {"enabled": "true"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "boolean" in verdict.message

    def test_boolean_invalid_int(self, guard):
        # In Python bool is subclass of int, but int is not bool
        ctx = make_ctx("test_tool", {"enabled": 1})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    # --- array ---
    def test_array_valid(self, guard):
        ctx = make_ctx("test_tool", {"items": [1, 2, 3]})
        assert guard.check_pre(ctx) is None

    def test_array_invalid_string(self, guard):
        ctx = make_ctx("test_tool", {"items": "[1, 2, 3]"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "array" in verdict.message

    def test_array_invalid_tuple(self, guard):
        ctx = make_ctx("test_tool", {"items": (1, 2, 3)})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    # --- object ---
    def test_object_valid(self, guard):
        ctx = make_ctx("test_tool", {"config": {"key": "value"}})
        assert guard.check_pre(ctx) is None

    def test_object_invalid_string(self, guard):
        ctx = make_ctx("test_tool", {"config": '{"key": "value"}'})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "object" in verdict.message

    def test_object_invalid_list(self, guard):
        ctx = make_ctx("test_tool", {"config": [("key", "value")]})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"


class TestArgTypeGuardMultipleErrors:
    """Test behavior with multiple type errors."""

    def test_multiple_errors_reported(self):
        registry = make_registry({
            "test_tool": {
                "properties": {
                    "count": {"type": "integer"},
                    "name": {"type": "string"},
                }
            }
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("test_tool", {"count": "ten", "name": 42})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "count" in verdict.message
        assert "name" in verdict.message

    def test_mix_valid_and_invalid(self):
        registry = make_registry({
            "test_tool": {
                "properties": {
                    "count": {"type": "integer"},
                    "name": {"type": "string"},
                }
            }
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("test_tool", {"count": 5, "name": 42})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert "count" not in verdict.message  # count is valid
        assert "name" in verdict.message


class TestArgTypeGuardMetadata:
    """Test guard metadata and properties."""

    def test_name(self):
        guard = ArgTypeGuard()
        assert guard.name == "arg_type"

    def test_not_overridable(self):
        guard = ArgTypeGuard()
        assert guard.overridable is False

    def test_verdict_category(self):
        registry = make_registry({
            "test_tool": {"properties": {"x": {"type": "integer"}}}
        })
        guard = ArgTypeGuard(tool_registry=registry)
        ctx = make_ctx("test_tool", {"x": "bad"})
        verdict = guard.check_pre(ctx)
        assert verdict.category == "arg_type"
        assert verdict.reason == "arg_type_mismatch"
