"""Tests for LLM providers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from flagscale_agent.react.providers.base import LLMProvider


class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        with patch("flagscale_agent.react.providers.anthropic_provider.anthropic") as mock_mod:
            mock_client = MagicMock()
            mock_mod.Anthropic.return_value = mock_client
            from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(model="claude-test", api_key="test-key")
            p._mock_client = mock_client
            return p

    def test_split_system(self, provider):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, chat = provider._split_system(msgs)
        assert system == "You are helpful."
        assert len(chat) == 1
        assert chat[0]["role"] == "user"

    def test_split_system_no_system(self, provider):
        msgs = [{"role": "user", "content": "Hi"}]
        system, chat = provider._split_system(msgs)
        assert system is None
        assert len(chat) == 1

    def test_format_assistant_text_only(self, provider):
        response = {"content": "Hello!", "tool_calls": None}
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello!"

    def test_format_assistant_tool_calls_only(self, provider):
        response = {
            "content": None,
            "tool_calls": [{"id": "tc1", "name": "shell", "arguments": {"command": "ls"}}],
        }
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        blocks = msg["content"]
        assert any(b["type"] == "tool_use" for b in blocks)
        tool_block = [b for b in blocks if b["type"] == "tool_use"][0]
        assert tool_block["name"] == "shell"
        assert tool_block["input"] == {"command": "ls"}

    def test_format_assistant_both(self, provider):
        response = {
            "content": "Let me check.",
            "tool_calls": [{"id": "tc1", "name": "read_file", "arguments": {"path": "/tmp/x"}}],
        }
        msg = provider.format_assistant_message(response)
        types = [b["type"] for b in msg["content"]]
        assert "text" in types
        assert "tool_use" in types

    def test_format_assistant_empty(self, provider):
        response = {"content": None, "tool_calls": None}
        msg = provider.format_assistant_message(response)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == ""

    def test_format_tool_result(self, provider):
        msg = provider.format_tool_result("tc1", "file contents here")
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tc1"
        assert msg["content"][0]["content"] == "file contents here"

    def test_format_tool_result_empty(self, provider):
        msg = provider.format_tool_result("tc1", "")
        assert msg["content"][0]["content"] == "(empty)"

    def test_schema_format(self, provider):
        assert provider.schema_format == "anthropic"

    def test_chat(self, provider):
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Hello"
        mock_response = MagicMock()
        mock_response.content = [mock_text]
        provider._mock_client.messages.create.return_value = mock_response

        result = provider.chat(
            [{"role": "user", "content": "Hi"}],
            tools=[],
        )
        assert result["content"] == "Hello"
        assert result["tool_calls"] is None

    def test_chat_with_tool_call(self, provider):
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tc1"
        mock_tool.name = "shell"
        mock_tool.input = {"command": "ls"}
        mock_response = MagicMock()
        mock_response.content = [mock_tool]
        provider._mock_client.messages.create.return_value = mock_response

        result = provider.chat(
            [{"role": "user", "content": "list files"}],
            tools=[{"name": "shell"}],
        )
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "shell"


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        with patch("flagscale_agent.react.providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            from flagscale_agent.react.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(model="gpt-test", api_key="test-key")
            p._mock_client = mock_client
            return p

    def test_format_assistant_text_only(self, provider):
        response = {"content": "Hello!", "tool_calls": None}
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg

    def test_format_assistant_tool_calls(self, provider):
        response = {
            "content": None,
            "tool_calls": [{"id": "tc1", "name": "shell", "arguments": {"command": "ls"}}],
        }
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "shell"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    def test_format_assistant_both(self, provider):
        response = {
            "content": "Checking...",
            "tool_calls": [{"id": "tc1", "name": "read_file", "arguments": {"path": "/x"}}],
        }
        msg = provider.format_assistant_message(response)
        assert msg["content"] == "Checking..."
        assert len(msg["tool_calls"]) == 1

    def test_format_tool_result(self, provider):
        msg = provider.format_tool_result("tc1", "result text")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc1"
        assert msg["content"] == "result text"

    def test_schema_format(self, provider):
        assert provider.schema_format == "openai"

    def test_chat(self, provider):
        mock_message = MagicMock()
        mock_message.content = "Hi there"
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        provider._mock_client.chat.completions.create.return_value = mock_response

        result = provider.chat(
            [{"role": "user", "content": "Hi"}],
            tools=[],
        )
        assert result["content"] == "Hi there"
        assert result["tool_calls"] is None

    def test_chat_with_tool_call(self, provider):
        mock_tc = MagicMock()
        mock_tc.id = "tc1"
        mock_tc.function.name = "shell"
        mock_tc.function.arguments = '{"command": "ls"}'
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        provider._mock_client.chat.completions.create.return_value = mock_response

        result = provider.chat(
            [{"role": "user", "content": "list files"}],
            tools=[{"name": "shell"}],
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"] == {"command": "ls"}
