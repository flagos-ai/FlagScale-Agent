"""OpenAI provider implementation."""

import json

from typing import Any, Dict, Iterator, List

from openai import OpenAI

from flagscale_agent.react.providers.base import LLMProvider



class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192):
        self._model = model
        self._max_tokens = max_tokens
        kwargs = {"api_key": api_key, "timeout": 120.0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Dict[str, Any]:
        kwargs = {"model": self._model, "messages": messages, "max_tokens": self._max_tokens}
        if tools:
            kwargs["tools"] = tools

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
        kwargs = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        stream = self._client.chat.completions.create(**kwargs)
        seen_tool_ids = set()  # Track tool_start already fired to avoid duplicates
        for chunk in stream:
            if chunk.usage:
                yield {
                    "type": "usage",
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                yield {"type": "text", "content": delta.content}
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        tc_id = tc.id or ""
                        if tc_id and tc_id not in seen_tool_ids:
                            seen_tool_ids.add(tc_id)
                            yield {"type": "tool_start", "id": tc.id, "name": tc.function.name}
                    if tc.function and tc.function.arguments:
                        yield {"type": "tool_delta", "id": tc.id or "", "arguments_delta": tc.function.arguments}
        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        msg = {"role": "assistant"}
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
