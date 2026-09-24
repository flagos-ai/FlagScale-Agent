# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for the empty-output retry fix (thinking-only outputs).

Two real bugs covered:
1. pop no-op: kernel empty-retry popped a get_messages() shallow copy, so the
   thinking-only assistant message stayed in the REAL history. Fixed via
   HistoryManager.pop_last_assistant().
2. Provider compatibility: reasoning detection relied on
   response["reasoning_only"], which only openai_provider sets. An
   anthropic-compat endpoint (e.g. GLM) with thinking-only output took the
   generic path. Detection now uses response["thinking"] (normalized by BOTH
   providers), and the nudge embeds the FULL thinking text — endpoints like
   GLM drop assistant thinking blocks from input, so the model could not see
   its own reasoning on retry.
"""

import types
from unittest.mock import MagicMock

import pytest

from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry


# ── HistoryManager.pop_last_assistant ────────────────────────────────────────

class TestPopLastAssistant:
    def _hm_with_history(self):
        hm = HistoryManager(max_context_tokens=64000)
        hm.append({"role": "user", "content": "task"})
        hm.append({"role": "assistant", "content": "step"})
        return hm

    def test_pop_actually_mutates_real_history(self):
        """THE shallow-copy regression: real _messages must shrink."""
        hm = self._hm_with_history()
        n_before = len(hm.messages)
        popped = hm.pop_last_assistant()
        assert popped is not None and popped["role"] == "assistant"
        assert len(hm.messages) == n_before - 1
        # get_messages() (validated copy) reflects the removal too
        assert all(m.get("role") != "assistant" for m in hm.get_messages())

    def test_pop_returns_none_when_last_is_user(self):
        hm = self._hm_with_history()
        hm.append({"role": "user", "content": "latest"})
        assert hm.pop_last_assistant() is None
        assert len(hm.messages) == 3  # unchanged

    def test_pop_returns_none_on_empty_history(self):
        hm = HistoryManager(max_context_tokens=64000)
        assert hm.pop_last_assistant() is None

    def test_pop_keeps_full_log_audit(self):
        """_full_log is append-only: popped message stays, recall still works."""
        hm = self._hm_with_history()
        ext_idx = hm.messages[-1]["_ext_idx"]
        popped = hm.pop_last_assistant()
        assert popped is not None
        assert len(hm._full_log) == 2  # audit log untouched
        assert hm.recall_from_full_log(ext_idx) is not None

    def test_get_messages_is_still_a_copy(self):
        """Documents the root cause: mutating get_messages() must NOT touch history."""
        hm = self._hm_with_history()
        msgs = hm.get_messages()
        if msgs and msgs[-1].get("role") == "assistant":
            msgs.pop()
        assert len(hm.messages) == 2  # real history unchanged


# ── Kernel empty-retry behavior ──────────────────────────────────────────────

class _RealHistoryHarness:
    """Real HistoryManager driven through AgentKernel.run_turn."""

    def __init__(self):
        self.history = HistoryManager(max_context_tokens=64000)
        self.history.append({"role": "user", "content": "do the task"})
        self.registry = GuardRegistry()
        self.config = types.SimpleNamespace(
            max_iterations=20, max_continuations=5, mode="auto", _turn_count=0,
        )
        self.call_count = {"n": 0}

    def build(self, llm_responses):
        def call_llm_fn(messages, schemas):
            idx = min(self.call_count["n"], len(llm_responses) - 1)
            self.call_count["n"] += 1
            return llm_responses[idx], {"input_tokens": 1, "output_tokens": 1}

        provider = MagicMock()
        provider.format_assistant_message.side_effect = lambda resp: {
            "role": "assistant", "content": resp.get("content") or "",
        }

        deps = KernelDeps(
            provider=provider,
            history=self.history,
            tool_registry=MagicMock(),
            judge=MagicMock(),
            guard_registry=self.registry,
            config=self.config,
            display=MagicMock(),
            get_schemas_fn=lambda: [],
            inject_message_fn=lambda msg: self.history.append(
                {"role": "user", "content": msg}),
            append_tool_results_fn=lambda results: None,
            format_tool_result_fn=lambda tid, r: {},
            execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
            is_context_limit_error_fn=lambda e: False,
            call_llm_fn=call_llm_fn,
        )
        kernel = AgentKernel(deps)
        kernel._call_count = self.call_count
        return kernel

    def user_nudges(self):
        return [m for m in self.history.messages
                if m["role"] == "user" and m["content"].startswith("[system:")]


@pytest.fixture
def harness():
    return _RealHistoryHarness()
class TestAnthropicStyleEmptyRetry:
    """Anthropic-compat endpoint (e.g. GLM): thinking blocks in response,
    NO reasoning_only flag (that is openai-only). Regression: kernel must
    take the reasoning path and embed the FULL thinking in the nudge."""

    def test_thinking_only_takes_reasoning_path_and_real_pop(self, harness):
        resp = {"content": None, "tool_calls": [],
                "thinking": "long reasoning " * 100, "signature": "sig==",
                "truncated": False}
        done = {"content": "[TASK_COMPLETE]", "tool_calls": []}
        # 2 thinking-only calls (2 retries), then success — within the
        # 3-retry budget (exhaustion covered by test_three_retries_then_gives_up)
        kernel = harness.build([resp, resp, done])
        result = kernel.run_turn()

        assert result.stop_reason == "explicit_signal", result.stop_reason
        assert harness.call_count["n"] == 3
        # Empty assistant messages REMOVED from real history (bug #1)
        assistants = [m for m in harness.history.messages
                      if m["role"] == "assistant" and not m["content"]]
        assert assistants == []
        # Nudges contain the FULL thinking text (bug #2, not a tail)
        nudges = harness.user_nudges()
        assert len(nudges) == 2
        for n in nudges:
            assert n["content"].count("long reasoning") == 100
            assert "<previous_reasoning>" in n["content"]

    def test_generic_nudge_when_no_thinking(self, harness):
        resp = {"content": None, "tool_calls": []}
        done = {"content": "[TASK_COMPLETE]", "tool_calls": []}
        kernel = harness.build([resp, done])
        result = kernel.run_turn()

        assert result.stop_reason == "explicit_signal", result.stop_reason
        nudges = harness.user_nudges()
        assert len(nudges) == 1
        assert "previous_reasoning" not in nudges[0]["content"]
        assert "empty response detected" in nudges[0]["content"]


class TestOpenAIStyleEmptyRetry:
    """openai_provider-shaped response: thinking + reasoning_only flag both
    present. Fix must be compatible — reasoning path + full thinking nudge."""

    def test_reasoning_only_flag_path_still_works(self, harness):
        resp = {"content": None, "tool_calls": [],
                "thinking": "chain of thought", "reasoning_only": True}
        done = {"content": "[TASK_COMPLETE]", "tool_calls": []}
        kernel = harness.build([resp, done])
        result = kernel.run_turn()

        assert result.stop_reason == "explicit_signal", result.stop_reason
        nudges = harness.user_nudges()
        assert len(nudges) == 1
        assert "chain of thought" in nudges[0]["content"]
        assert "<previous_reasoning>" in nudges[0]["content"]
        # real history has no leftover empty assistant message
        assert not [m for m in harness.history.messages
                    if m["role"] == "assistant" and not m["content"]]


class TestNoFalsePositive:
    def test_visible_text_skips_retry_entirely(self, harness):
        resp = {"content": "working on it", "tool_calls": []}
        done = {"content": "[TASK_COMPLETE]", "tool_calls": []}
        kernel = harness.build([resp, done])
        result = kernel.run_turn()

        assert result.stop_reason == "explicit_signal"
        assert harness.call_count["n"] == 2  # no extra retry calls
        assert harness.user_nudges() == []

    def test_three_retries_then_gives_up(self, harness):
        resp = {"content": None, "tool_calls": [], "thinking": "reasoning"}
        kernel = harness.build([resp])
        result = kernel.run_turn()

        assert result.stop_reason == "empty_output_max_retries"
        # 1 initial + 3 retries = 4 LLM calls
        assert harness.call_count["n"] == 4
        assert len(harness.user_nudges()) == 3
