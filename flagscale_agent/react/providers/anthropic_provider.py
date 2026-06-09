"""Anthropic provider implementation."""

import json

from typing import Any, Dict, Iterator, List

import anthropic

from flagscale_agent.react.providers.base import LLMProvider



class AnthropicProvider(LLMProvider):
    schema_format = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192):
        self._model = model
        self._max_tokens = max_tokens
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
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
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
        try:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "id": block.id, "name": block.name}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "content": delta.text}
                    elif delta.type == "input_json_delta":
                        yield {"type": "tool_delta", "id": "", "arguments_delta": delta.partial_json}
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

        # Only try to get usage if stream completed normally
        if stream_ctx is not None:
            try:
                final = stream.get_final_message()
                if final and final.usage:
                    yield {
                        "type": "usage",
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                    }
            except Exception:
                pass

        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks = []
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
