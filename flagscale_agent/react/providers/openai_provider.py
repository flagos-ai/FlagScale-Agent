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

"""OpenAI provider implementation."""

import json

from typing import Any, Dict, Iterator, List

from openai import OpenAI

from flagscale_agent.react.providers.base import LLMProvider



class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192):
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url
        kwargs = {"api_key": api_key, "timeout": 120.0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _build_kwargs(self, messages, tools, stream=False):
        """Build API call kwargs, with prompt caching support."""
        kwargs = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def chat(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Dict[str, Any]:
        kwargs = self._build_kwargs(messages, tools)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            return {"content": f"[PROVIDER_ERROR] {type(e).__name__}: {e}", "tool_calls": None}

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})

        return {"content": message.content, "tool_calls": tool_calls}

    def chat_stream(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Iterator[Dict[str, Any]]:
        kwargs = self._build_kwargs(messages, tools, stream=True)

        stream = self._client.chat.completions.create(**kwargs)
        seen_tool_ids = set()
        had_reasoning = False
        for chunk in stream:
            if chunk.usage:
                usage_data = {
                    "type": "usage",
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
                # Include cache stats when available (OpenAI prompt caching)
                ptd = getattr(chunk.usage, "prompt_tokens_details", None)
                if ptd:
                    cached = getattr(ptd, "cached_tokens", None)
                    if cached is None and isinstance(ptd, dict):
                        cached = ptd.get("cached_tokens")
                    if cached:
                        usage_data["cache_read_input_tokens"] = cached
                    cache_write = getattr(ptd, "cache_write_tokens", None)
                    if cache_write is None and isinstance(ptd, dict):
                        cache_write = ptd.get("cache_write_tokens")
                    if cache_write:
                        usage_data["cache_creation_input_tokens"] = cache_write
                yield usage_data
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                yield {"type": "text", "content": delta.content}
            # Detect reasoning tokens: GLM uses "reasoning" field in raw SSE,
            # some providers use "reasoning_content". Check both attribute and
            # model_extra (Pydantic v2 strips unknown fields from typed models).
            rc = getattr(delta, "reasoning_content", None)
            if not rc:
                rc = (delta.model_extra or {}).get("reasoning_content") if hasattr(delta, "model_extra") else None
            if not rc:
                rc = (delta.model_extra or {}).get("reasoning") if hasattr(delta, "model_extra") else None
            if rc:
                had_reasoning = True
                yield {"type": "thinking", "content": rc}
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        tc_id = tc.id or ""
                        if tc_id and tc_id not in seen_tool_ids:
                            seen_tool_ids.add(tc_id)
                            yield {"type": "tool_start", "id": tc.id, "name": tc.function.name}
                    if tc.function and tc.function.arguments:
                        yield {"type": "tool_delta", "id": tc.id or "", "arguments_delta": tc.function.arguments}
        if had_reasoning:
            yield {"type": "reasoning_only"}
        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        msg = {"role": "assistant"}
        if response.get("thinking"):
            msg["thinking"] = response["thinking"]
        if response["content"]:
            msg["content"] = response["content"]
        if response["tool_calls"]:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in response["tool_calls"]
            ]
        return msg

    def format_tool_result(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
