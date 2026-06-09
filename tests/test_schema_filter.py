"""Tests for phase-based schema filtering (Layer 3)."""

import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.tools import ToolRegistry
from flagscale_agent.react.tools.base import Tool


class DummyTool(Tool):
    """Minimal tool for testing schema filtering."""

    def __init__(self, name, description="test tool"):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return "ok"


class TestToolRegistryFiltered:

    def test_to_schemas_filtered_returns_subset(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        reg.register(DummyTool("read_file"))
        reg.register(DummyTool("write_file"))
        reg.register(DummyTool("monitor"))
        reg.register(DummyTool("plan_create"))

        schemas = reg.to_schemas_filtered("openai", {"shell", "monitor"})
        names = {s["function"]["name"] for s in schemas}
        assert names == {"shell", "monitor"}

    def test_to_schemas_filtered_empty_set(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        schemas = reg.to_schemas_filtered("openai", set())
        assert schemas == []

    def test_to_schemas_filtered_unknown_names_ignored(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        schemas = reg.to_schemas_filtered("openai", {"shell", "nonexistent"})
        assert len(schemas) == 1

    def test_to_schemas_filtered_anthropic_format(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        reg.register(DummyTool("monitor"))
        schemas = reg.to_schemas_filtered("anthropic", {"shell"})
        assert len(schemas) == 1
        assert schemas[0]["name"] == "shell"


class TestPhaseDetection:
    """Test _detect_tool_phase logic in isolation."""

    def _make_agent_stub(self):
        """Create a minimal object with the filtered schemas method."""
        from flagscale_agent.react.agent import WorkerAgent
        from flagscale_agent.react.constants import PHASE_TOOL_SETS, CORE_TOOLS

        agent = MagicMock()
        agent._extra_tools_next_iter = set()
        agent._get_filtered_schemas = WorkerAgent._get_filtered_schemas.__get__(agent)
        return agent

    def test_filtered_schemas_monitoring(self):
        agent = self._make_agent_stub()
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file", "plan_create",
                     "parse_training_metrics", "workspace_experiment"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("monitoring")
        names = {s["function"]["name"] for s in schemas}
        # "monitoring" is not a defined phase, so only _CORE_TOOLS are available
        assert "monitor" in names
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names  # write_file is a core tool
        assert "plan_create" in names  # plan_create is a core tool
        # parse_training_metrics is NOT a core tool, so it's excluded
        assert "parse_training_metrics" not in names

    def test_filtered_schemas_default_returns_all(self):
        agent = self._make_agent_stub()
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file", "plan_create"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("default")
        assert len(schemas) == 5

    def test_extra_tools_next_iter_included(self):
        agent = self._make_agent_stub()
        agent._extra_tools_next_iter = {"write_file"}
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("monitoring")
        names = {s["function"]["name"] for s in schemas}
        assert "write_file" in names
