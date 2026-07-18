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

"""LLM provider base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses handle all provider-specific message formatting so the agent
    loop stays provider-agnostic.
    """

    schema_format: str = "openai"

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Dict[str, Any]:
        """Send a chat request with tool definitions.

        Returns a unified dict:
            {
                "content": str | None,
                "tool_calls": list[{"id": str, "name": str, "arguments": dict}] | None,
            }
        """
        ...

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Iterator[Dict[str, Any]]:
        """Streaming variant of chat(). Yields events:
            {"type": "text", "content": str}
            {"type": "tool_start", "id": str, "name": str}
            {"type": "tool_delta", "id": str, "arguments_delta": str}
            {"type": "done"}
        The caller must accumulate tool arguments from deltas.
        """
        ...

    @abstractmethod
    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a unified response dict into a provider-specific assistant message."""
        ...

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        """Format a tool result into a provider-specific message."""
        ...
