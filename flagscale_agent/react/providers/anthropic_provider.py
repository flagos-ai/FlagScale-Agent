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

"""Anthropic provider implementation."""

import json

from typing import Any, Dict, Iterator, List

import anthropic

from flagscale_agent.react.providers.base import LLMProvider



class AnthropicProvider(LLMProvider):
    schema_format = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192, thinking_budget: int = 0):
        self._model = model
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget
        self._api_key = api_key
        self._base_url = base_url
        self._is_third_party = base_url and "anthropic.com" not in base_url
        self._auth_mode = None  # Will be auto-detected on first call
        self._timeout = 120.0  # 2-minute timeout for API calls + summarizer
        self._client = self._build_client()

    def _build_client(self):
        """Build Anthropic client with current auth mode."""
        kwargs = {"api_key": self._api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
            if self._is_third_party and self._auth_mode == "bearer":
                kwargs["api_key"] = "placeholder"
                kwargs["default_headers"] = {"Authorization": f"Bearer {self._api_key}"}
        return anthropic.Anthropic(**kwargs)

    def _switch_auth_and_retry(self):
        """Switch from x-api-key to Bearer auth after a 401."""
        if self._auth_mode == "bearer":
            return False  # Already tried Bearer, nothing more to do
        self._auth_mode = "bearer"
        self._client = self._build_client()
        return True

    def _split_system(self, messages):
        """Separate system message from chat messages (Anthropic requires this)."""
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        return system, chat_messages

    def _build_kwargs(self, messages, tools):
        system, chat_messages = self._split_system(messages)
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": chat_messages}
        if self._thinking_budget and self._thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
        if system:
            # Split static body from dynamic dashboard for prompt caching.
            # The dashboard (Turn counter, memory keys) changes every turn and
            # must be AFTER the cache_control breakpoint to avoid invalidating
            # the cached static body (~22K chars / ~5.5K tokens).
            sep = "\n---\n["
            idx = system.find(sep)
            if idx >= 0:
                kwargs["system"] = [
                    {"type": "text", "text": system[:idx], "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": system[idx:]},
                ]
            else:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
                ]
        if tools:
            kwargs["tools"] = tools
        # Add cache_control to the last message to cache conversation prefix.
        # Copy to avoid mutating shared message references from history.
        if chat_messages:
            last = chat_messages[-1]
            last_copy = dict(last)
            content = last.get("content")
            if isinstance(content, str):
                last_copy["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
                ]
            elif isinstance(content, list):
                content_copy = [dict(b) if isinstance(b, dict) else b for b in content]
                if content_copy and isinstance(content_copy[-1], dict):
                    content_copy[-1]["cache_control"] = {"type": "ephemeral"}
                last_copy["content"] = content_copy
            chat_messages[-1] = last_copy
        return kwargs

    def chat(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Dict[str, Any]:
        kwargs = self._build_kwargs(messages, tools)
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                response = self._client.messages.create(**kwargs)
            else:
                raise

        content = None
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})

        return {"content": content, "tool_calls": tool_calls}

    def chat_stream(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Iterator[Dict[str, Any]]:
        kwargs = self._build_kwargs(messages, tools)
        stream_ctx = None

        try:
            stream_ctx = self._client.messages.stream(**kwargs)
            stream = stream_ctx.__enter__()
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                # Close old context before creating new one
                if stream_ctx is not None:
                    try:
                        stream_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                stream_ctx = self._client.messages.stream(**kwargs)
                stream = stream_ctx.__enter__()
            else:
                raise

        stream_error = None
        final_message = None
        try:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "id": block.id, "name": block.name}
                    elif block.type == "thinking":
                        yield {"type": "thinking_start"}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "content": delta.text}
                    elif delta.type == "input_json_delta":
                        yield {"type": "tool_delta", "id": "", "arguments_delta": delta.partial_json}
                    elif delta.type == "thinking_delta":
                        yield {"type": "thinking", "content": delta.thinking}
                    elif delta.type == "signature_delta":
                        yield {"type": "signature", "content": delta.signature}
                elif event.type == "content_block_stop":
                    yield {"type": "block_stop"}
            
            # Get usage BEFORE closing the stream
            if stream_ctx is not None:
                try:
                    final_message = stream.get_final_message()
                except Exception:
                    pass
        except Exception as e:
            stream_error = e
        finally:
            if stream_ctx is not None:
                try:
                    stream_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        if stream_error:
            raise stream_error

        # Yield usage data if we got it
        if final_message and final_message.usage:
            usage_data = {
                "type": "usage",
                "input_tokens": final_message.usage.input_tokens,
                "output_tokens": final_message.usage.output_tokens,
            }
            # Include cache info when prompt caching is active
            cache_read = getattr(final_message.usage, "cache_read_input_tokens", None)
            cache_create = getattr(final_message.usage, "cache_creation_input_tokens", None)
            if cache_read:
                usage_data["cache_read_input_tokens"] = cache_read
            if cache_create:
                usage_data["cache_creation_input_tokens"] = cache_create
            yield usage_data

        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks = []
        if response.get("thinking"):
            content_blocks.append({
                "type": "thinking",
                "thinking": response["thinking"],
                "signature": response.get("signature", ""),
            })
        if response["content"]:
            content_blocks.append({"type": "text", "text": response["content"]})
        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": content_blocks}

    def format_tool_result(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content or "(empty)"}],
        }
