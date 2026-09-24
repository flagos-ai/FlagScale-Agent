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

    def test_thinking_budget_disabled_by_default(self):
        """Without thinking_budget, _build_kwargs should NOT include thinking key."""
        with patch("flagscale_agent.react.providers.anthropic_provider.anthropic") as mock_mod:
            mock_mod.Anthropic.return_value = MagicMock()
            from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(model="test", api_key="key", max_tokens=8192)
            kwargs = p._build_kwargs(
                [{"role": "user", "content": "hi"}], tools=[]
            )
            assert "thinking" not in kwargs

    def test_thinking_budget_enabled(self):
        """With thinking_budget > 0, _build_kwargs should include thinking config."""
        with patch("flagscale_agent.react.providers.anthropic_provider.anthropic") as mock_mod:
            mock_mod.Anthropic.return_value = MagicMock()
            from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(model="test", api_key="key", max_tokens=20480, thinking_budget=12288)
            kwargs = p._build_kwargs(
                [{"role": "user", "content": "hi"}], tools=[]
            )
            assert "thinking" in kwargs
            assert kwargs["thinking"]["type"] == "enabled"
            assert kwargs["thinking"]["budget_tokens"] == 12288

    def test_format_assistant_with_thinking(self, provider):
        """format_assistant_message should include thinking block with signature when present."""
        response = {
            "content": "Here's my answer.",
            "tool_calls": None,
            "thinking": "Let me think about this...",
            "signature": "sig123",
        }
        msg = provider.format_assistant_message(response)
        types = [b["type"] for b in msg["content"]]
        assert "thinking" in types
        thinking_block = [b for b in msg["content"] if b["type"] == "thinking"][0]
        assert thinking_block["thinking"] == "Let me think about this..."
        assert thinking_block["signature"] == "sig123"
        # Thinking block should come before text
        assert types.index("thinking") < types.index("text")

    def test_format_assistant_no_thinking_backward_compat(self, provider):
        """Without thinking data, format_assistant_message should not include thinking block."""
        response = {"content": "Hello!", "tool_calls": None}
        msg = provider.format_assistant_message(response)
        types = [b["type"] for b in msg["content"]]
        assert "thinking" not in types

    # ── cache_control tests ───────────────────────────────────────────────

    def test_system_prompt_has_cache_control(self, provider):
        """System prompt must be a list of blocks with cache_control on the first block."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = provider._build_kwargs(msgs, tools=[])
        assert "system" in kwargs
        system = kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_no_cache_control_when_absent(self, provider):
        """Without a system message, kwargs should not include system key."""
        msgs = [{"role": "user", "content": "Hi"}]
        kwargs = provider._build_kwargs(msgs, tools=[])
        assert "system" not in kwargs

    def test_system_prompt_split_at_dashboard(self, provider):
        """When system prompt contains dashboard separator, static body is cached, dashboard is not."""
        static_body = "You are FlagScale Agent.\nTools: shell, read_file\n"
        dashboard = "\n---\n[Turn: 1 | Task: Demo | Step: 1/2]"
        full_system = static_body + dashboard
        msgs = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = provider._build_kwargs(msgs, tools=[])
        system = kwargs["system"]
        # Should be two blocks: cached static + uncached dashboard
        assert len(system) == 2
        assert system[0]["text"] == static_body
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[1]["text"] == dashboard
        assert "cache_control" not in system[1]

    def test_last_message_str_content_gets_cache_control(self, provider):
        """Last message with string content gets cache_control wrapper."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello"},
        ]
        kwargs = provider._build_kwargs(msgs, tools=[])
        last = kwargs["messages"][-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_last_message_list_content_gets_cache_control(self, provider):
        """Last message with list content (tool_result) gets cache_control on last block."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc1", "content": "result"}]},
        ]
        kwargs = provider._build_kwargs(msgs, tools=[])
        last = kwargs["messages"][-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # Original message should not be mutated
        assert "cache_control" not in msgs[-1]["content"][-1]

    def test_original_messages_not_mutated(self, provider):
        """_build_kwargs must not mutate the original messages list."""
        original_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello"},
        ]
        original_content = original_msgs[-1]["content"]
        provider._build_kwargs(original_msgs, tools=[])
        # Original should still be a plain string
        assert original_msgs[-1]["content"] == original_content
        assert isinstance(original_msgs[-1]["content"], str)

    def test_no_messages_no_cache_control(self, provider):
        """When there are no chat messages, no cache_control is added to messages."""
        msgs = [{"role": "system", "content": "sys"}]
        kwargs = provider._build_kwargs(msgs, tools=[])
        assert kwargs["messages"] == []

    def test_chat_stream_thinking_events(self, provider):
        """chat_stream should emit thinking and signature events for thinking_delta."""
        # Mock Anthropic streaming events
        thinking_start = MagicMock()
        thinking_start.type = "content_block_start"
        thinking_start.content_block = MagicMock()
        thinking_start.content_block.type = "thinking"

        thinking_delta = MagicMock()
        thinking_delta.type = "content_block_delta"
        thinking_delta.delta = MagicMock()
        thinking_delta.delta.type = "thinking_delta"
        thinking_delta.delta.thinking = "Reasoning about the problem..."

        sig_delta = MagicMock()
        sig_delta.type = "content_block_delta"
        sig_delta.delta = MagicMock()
        sig_delta.delta.type = "signature_delta"
        sig_delta.delta.signature = "sig_abc123"

        block_stop = MagicMock()
        block_stop.type = "content_block_stop"

        text_delta = MagicMock()
        text_delta.type = "content_block_delta"
        text_delta.delta = MagicMock()
        text_delta.delta.type = "text_delta"
        text_delta.delta.text = "Here is my answer."

        text_start = MagicMock()
        text_start.type = "content_block_start"
        text_start.content_block = MagicMock()
        text_start.content_block.type = "text"

        events = [thinking_start, thinking_delta, sig_delta, block_stop, text_start, text_delta, block_stop]

        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter(events))
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_final = MagicMock()
        mock_final.usage = MagicMock()
        mock_final.usage.input_tokens = 100
        mock_final.usage.output_tokens = 200
        mock_final.usage.cache_read_input_tokens = 0
        mock_final.usage.cache_creation_input_tokens = 0
        mock_stream.get_final_message = MagicMock(return_value=mock_final)
        provider._mock_client.messages.stream = MagicMock(return_value=mock_stream)

        results = list(provider.chat_stream([{"role": "user", "content": "hi"}], tools=[]))
        types = [e["type"] for e in results]

        assert "thinking_start" in types
        assert "thinking" in types
        thinking_event = [e for e in results if e["type"] == "thinking"][0]
        assert thinking_event["content"] == "Reasoning about the problem..."

        assert "signature" in types
        sig_event = [e for e in results if e["type"] == "signature"][0]
        assert sig_event["content"] == "sig_abc123"

        assert "text" in types
        assert "done" in types


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        pytest.importorskip("openai")
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

    def test_chat_stream_reasoning_only(self, provider):
        """When stream has reasoning_content but no text/tool_calls, emit reasoning_only event."""
        # Build mock chunks: one with reasoning_content, one with usage
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        delta1 = MagicMock()
        delta1.content = None
        delta1.tool_calls = None
        delta1.reasoning_content = "thinking about the problem..."
        chunk1.choices[0].delta = delta1

        chunk2 = MagicMock()
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 100
        chunk2.usage.completion_tokens = 16000
        chunk2.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "solve circuit"}], tools=[]
        ))
        types = [e["type"] for e in events]
        assert "reasoning_only" in types
        assert "done" in types
        assert "text" not in types

    def test_chat_stream_emits_thinking_event(self, provider):
        """When reasoning_content is present, a thinking event is emitted with the content."""
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        delta1 = MagicMock()
        delta1.content = None
        delta1.tool_calls = None
        delta1.reasoning_content = "Let me analyze this circuit..."
        delta1.model_extra = {}
        chunk1.choices[0].delta = delta1

        chunk2 = MagicMock()
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 100
        chunk2.usage.completion_tokens = 200
        chunk2.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "solve circuit"}], tools=[]
        ))
        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0]["content"] == "Let me analyze this circuit..."

    def test_format_assistant_with_thinking(self, provider):
        """format_assistant_message includes thinking field when present in response."""
        response = {"content": "Answer", "tool_calls": None, "thinking": "My reasoning..."}
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Answer"
        assert msg["thinking"] == "My reasoning..."

    def test_format_assistant_no_thinking_backward_compat(self, provider):
        """format_assistant_message works without thinking field (backward compat)."""
        response = {"content": "Answer", "tool_calls": None}
        msg = provider.format_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Answer"
        assert "thinking" not in msg

    def test_chat_stream_reasoning_plus_text_no_reasoning_only(self, provider):
        """When stream has both reasoning and text, reasoning_only is emitted but text is also present."""
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        delta1 = MagicMock()
        delta1.content = None
        delta1.tool_calls = None
        delta1.reasoning_content = "thinking..."
        delta1.model_extra = {}
        chunk1.choices[0].delta = delta1

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [MagicMock()]
        delta2 = MagicMock()
        delta2.content = "Here is my answer."
        delta2.tool_calls = None
        delta2.reasoning_content = None
        chunk2.choices[0].delta = delta2

        chunk3 = MagicMock()
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 100
        chunk3.usage.completion_tokens = 200
        chunk3.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "solve"}], tools=[]
        ))
        types = [e["type"] for e in events]
        assert "reasoning_only" in types  # provider always emits it
        assert "text" in types  # but there IS text

    def test_chat_stream_no_reasoning_no_reasoning_only(self, provider):
        """When stream has only text (no reasoning), no reasoning_only event."""
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        delta1 = MagicMock()
        delta1.content = "Hello!"
        delta1.tool_calls = None
        # No reasoning in any form
        delta1.reasoning_content = None
        delta1.model_extra = {}
        chunk1.choices[0].delta = delta1

        chunk2 = MagicMock()
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 10
        chunk2.usage.completion_tokens = 5
        chunk2.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}], tools=[]
        ))
        types = [e["type"] for e in events]
        assert "reasoning_only" not in types
        assert "text" in types
        assert "done" in types

    # ── prompt caching observability tests ─────────────────────────────────

    def test_chat_stream_cached_tokens_in_usage(self, provider):
        """When usage has prompt_tokens_details.cached_tokens, it appears in usage event."""
        chunk1 = MagicMock()
        chunk1.usage = MagicMock()
        chunk1.usage.prompt_tokens = 5000
        chunk1.usage.completion_tokens = 100
        # Set up prompt_tokens_details with cached_tokens
        ptd = MagicMock()
        ptd.cached_tokens = 4000
        ptd.cache_write_tokens = 1000
        chunk1.usage.prompt_tokens_details = ptd
        chunk1.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}], tools=[]
        ))
        usage_events = [e for e in events if e["type"] == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["cache_read_input_tokens"] == 4000
        assert usage_events[0]["cache_creation_input_tokens"] == 1000

    def test_chat_stream_no_cached_tokens_when_absent(self, provider):
        """When usage has no prompt_tokens_details, cached_tokens is not in usage event."""
        chunk1 = MagicMock()
        chunk1.usage = MagicMock()
        chunk1.usage.prompt_tokens = 100
        chunk1.usage.completion_tokens = 50
        chunk1.usage.prompt_tokens_details = None
        chunk1.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}], tools=[]
        ))
        usage_events = [e for e in events if e["type"] == "usage"]
        assert len(usage_events) == 1
        assert "cache_read_input_tokens" not in usage_events[0]
        assert "cache_creation_input_tokens" not in usage_events[0]

    def test_chat_stream_partial_cache_fields(self, provider):
        """When only cached_tokens is present (no cache_write_tokens), only it is included."""
        chunk1 = MagicMock()
        chunk1.usage = MagicMock()
        chunk1.usage.prompt_tokens = 5000
        chunk1.usage.completion_tokens = 100
        ptd = MagicMock()
        ptd.cached_tokens = 4000
        ptd.cache_write_tokens = None  # No cache write
        chunk1.usage.prompt_tokens_details = ptd
        chunk1.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}], tools=[]
        ))
        usage_events = [e for e in events if e["type"] == "usage"]
        assert usage_events[0]["cache_read_input_tokens"] == 4000
        assert "cache_creation_input_tokens" not in usage_events[0]

    def test_build_kwargs_stream_mode(self, provider):
        """_build_kwargs with stream=True includes stream and stream_options."""
        kwargs = provider._build_kwargs(
            [{"role": "user", "content": "hi"}], tools=[], stream=True
        )
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_build_kwargs_non_stream_mode(self, provider):
        """_build_kwargs without stream flag does not include stream keys."""
        kwargs = provider._build_kwargs(
            [{"role": "user", "content": "hi"}], tools=[]
        )
        assert "stream" not in kwargs
        assert "stream_options" not in kwargs

    def test_build_kwargs_with_tools(self, provider):
        """_build_kwargs includes tools when provided."""
        kwargs = provider._build_kwargs(
            [{"role": "user", "content": "hi"}], tools=[{"name": "shell"}]
        )
        assert kwargs["tools"] == [{"name": "shell"}]

    def test_chat_stream_cached_tokens_dict_fallback(self, provider):
        """When prompt_tokens_details is a dict (third-party API), cached_tokens still read."""
        chunk1 = MagicMock()
        chunk1.usage = MagicMock()
        chunk1.usage.prompt_tokens = 5000
        chunk1.usage.completion_tokens = 100
        # Some third-party APIs may return dict instead of Pydantic model
        chunk1.usage.prompt_tokens_details = {"cached_tokens": 3000, "cache_write_tokens": 500}
        chunk1.choices = []

        provider._mock_client.chat.completions.create.return_value = iter([chunk1])

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}], tools=[]
        ))
        usage_events = [e for e in events if e["type"] == "usage"]
        assert usage_events[0]["cache_read_input_tokens"] == 3000
        assert usage_events[0]["cache_creation_input_tokens"] == 500
