"""Tool registry."""


from typing import Dict, List

from flagscale_agent.react.tools.base import Tool



class ToolRegistry:
    """Registry for agent tools with unified result truncation."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def execute(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name, with unified result truncation."""
        tool = self.get(tool_name)
        result = tool.execute(**kwargs)
        if len(result) > tool.max_result_size:
            result = result[:tool.max_result_size] + f"\n... [truncated, total {len(result)} chars]"
        return result

    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_schemas(self, fmt: str = "openai") -> List[dict]:
        if fmt == "anthropic":
            return [t.to_anthropic_schema() for t in self._tools.values()]
        return [t.to_openai_schema() for t in self._tools.values()]

    def to_schemas_filtered(self, fmt: str, tool_names: set) -> List[dict]:
        """Return schemas only for the named tools (reduces token cost)."""
        tools = [t for t in self._tools.values() if t.name in tool_names]
        if fmt == "anthropic":
            return [t.to_anthropic_schema() for t in tools]
        return [t.to_openai_schema() for t in tools]
