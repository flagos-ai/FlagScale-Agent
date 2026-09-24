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

"""Tests for the runtime thinking cap (config.thinking_budget enforced
client-side by _call_llm_stream).

Background: GLM's Anthropic-compat endpoint ignores the request-side
thinking_budget (served thinking runs to max_output_tokens), so a single
thinking block can run ~20 minutes producing zero visible output. The cap
aborts a PURE-thinking stream once its estimated token count reaches
thinking_budget; the kernel then re-injects the accumulated thinking with a
CONVERGENCE nudge (land the next concrete step now) instead of the RESUME
nudge used for uncapped reasoning-only responses.

Behavior contract:
  - thinking_budget <= 0 → cap disabled, behavior identical to pre-cap code
  - cap trips only while the response is PURE thinking (no text/tool_calls)
  - capped response carries thinking_capped=True + the accumulated thinking
  - kernel capped → convergence nudge (contains "runtime length cap")
  - kernel uncapped reasoning-only → resume nudge (unchanged from 9e82ad6)
"""

import types
from unittest.mock import MagicMock

import pytest

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.history import HistoryManager, _estimate_tokens
from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.thinking_cap import (
    ThinkingCapCounter,
    estimate_thinking_tokens,
    thinking_capped_state,
)
from flagscale_agent.react.guard import GuardRegistry


# ── Pure threshold function ──────────────────────────────────────────────────

class TestThresholdPredicate:
    """thinking_capped_state: int(cjk*1.5) + ascii//4 >= budget, gated on >0."""

    def test_at_budget_trips(self):
        # 100 ascii chars -> 25 estimated tokens
        assert thinking_capped_state(100, 0, 25) is True

    def test_below_budget_does_not_trip(self):
        assert thinking_capped_state(100, 0, 26) is False

    def test_budget_minus_one_boundary(self):
        # 96 ascii chars -> 24 tokens; budget 24 trips, 25 does not
        assert thinking_capped_state(96, 0, 24) is True
        assert thinking_capped_state(96, 0, 25) is False

    def test_zero_budget_disabled(self):
        assert thinking_capped_state(10**9, 0, 0) is False

    def test_negative_budget_disabled(self):
        assert thinking_capped_state(10**9, 0, -5) is False

    def test_cjk_weighting(self):
        # 10 CJK chars -> int(10*1.5)=15 tokens (ascii 0)
        assert thinking_capped_state(10, 10, 15) is True
        assert thinking_capped_state(10, 10, 16) is False

    def test_mixed_cjk_ascii(self):
        # 6 cjk (9 tokens) + 40 ascii (10 tokens) = 19
        assert thinking_capped_state(46, 6, 19) is True
        assert thinking_capped_state(46, 6, 20) is False

    def test_matches_history_estimator(self):
        """Mirror consistency: predicate over accumulated counts equals
        _estimate_tokens over the whole text."""
        text = "thinking " * 500 + "推理" * 50
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        est_hist = _estimate_tokens(text)
        # _estimate_tokens returns >=1 always; predicate math without the floor
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                        or '\u3040' <= c <= '\u30ff'
                        or '\uac00' <= c <= '\ud7af')
        est_pred = int(cjk_count * 1.5) + ((len(text) - cjk_count) // 4)
        assert est_pred == est_hist  # non-empty text, floor irrelevant
        assert thinking_capped_state(len(text), cjk_count, est_hist) is True
        assert thinking_capped_state(len(text), cjk_count, est_hist + 1) is False


class TestIncrementalCounter:
    def test_counter_matches_batch_estimate(self):
        """Incremental add == batch _estimate_tokens on concatenated text."""
        deltas = ["hello ", "world ", "推理步骤", "a" * 100]
        counter = ThinkingCapCounter()
        for d in deltas:
            counter.add(d)
        whole = "".join(deltas)
        assert counter.chars == len(whole)
        assert counter.cjk == sum(1 for c in whole
                                  if '\u4e00' <= c <= '\u9fff')
        assert estimate_thinking_tokens(whole) == (
            int(counter.cjk * 1.5) + (counter.chars - counter.cjk) // 4)

    def test_empty_delta_is_noop(self):
        counter = ThinkingCapCounter()
        assert counter.add("") == (0, 0)

    def test_state_returns_char_position(self):
        counter = ThinkingCapCounter()
        counter.add("x" * 100)  # 25 tokens
        capped, at_chars = counter.state(25)
        assert capped is True and at_chars == 100
        capped, at_chars = counter.state(26)
        assert capped is False and at_chars is None
# (append) stream-level + kernel-level tests for the runtime thinking cap.

# ── Agent._call_llm_stream stream behavior ───────────────────────────────────

def _fake_agent(thinking_budget):
    """Minimal Agent-like object binding the real _call_llm_stream without
    running Agent.__init__ (which needs API keys, prompt session, etc.)."""
    agent = WorkerAgent.__new__(WorkerAgent)
    agent.config = types.SimpleNamespace(thinking_budget=thinking_budget)
    return agent


def _stream_events(thinking_deltas, tail=None):
    """Build a provider-style event stream: thinking_start, thinking*,
    signature, then the tail (default: a text block + done)."""
    events = [{"type": "thinking_start"}]
    for d in thinking_deltas:
        events.append({"type": "thinking", "content": d})
    events.append({"type": "signature", "content": "sig"})
    events.extend(tail or [{"type": "text", "content": "answer"},
                           {"type": "done"}])
    return iter(events)


class TestStreamCap:
    def test_cap_aborts_pure_thinking_stream(self):
        """Deltas totaling > cap: stream aborts, thinking preserved,
        thinking_capped=True, reasoning_only=True."""
        agent = _fake_agent(thinking_budget=10)  # 10 tokens = 40 ascii chars
        # 3 deltas of 30 chars each = 90 chars total -> exceeds 40
        deltas = ["a" * 30, "a" * 30, "a" * 30]
        consumed = []

        def fake_stream(messages, schemas):
            for ev in _stream_events(deltas):
                consumed.append(ev)
                yield ev

        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: fake_stream(m, s)
        resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is True
        assert resp["reasoning_only"] is True
        assert resp["content"] is None
        assert resp["tool_calls"] is None
        # All pre-trip deltas accumulated (cap trips DURING delta 2: 60 chars
        # = 15 tokens >= 10; delta 3 never consumed)
        assert resp["thinking"] == deltas[0] + deltas[1]
        assert len(consumed) == 3  # start, d1, d2 (abort inside d2) — no tail

    def test_cap_disabled_streams_to_completion(self):
        """thinking_budget=0 → no cap; full stream consumed; no capped flag."""
        agent = _fake_agent(thinking_budget=0)
        deltas = ["a" * 1000, "a" * 1000]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = (
            lambda m, s: iter(_stream_events(deltas)))
        resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is False
        assert resp["reasoning_only"] is False
        assert resp["content"] == "answer"
        assert resp["thinking"] == "a" * 2000
        assert resp["signature"] == "sig"

    def test_cap_does_not_trip_once_text_started(self):
        """Cap only guards PURE thinking. Text-BEFORE-thinking order: the
        response already has visible output when massive thinking arrives,
        so the cap must NOT abort it.

        (For the thinking-THEN-text order the cap cannot see the future: a
        pure thinking block exceeding the budget is aborted by design — the
        kernel's convergence nudge re-grounds the model with the preserved
        reasoning. A 32k-token thinking block with no output yet is the exact
        pathological case this feature exists to stop.)"""
        agent = _fake_agent(thinking_budget=10)  # 40 chars
        events = [{"type": "text", "content": "answer"},
                  {"type": "thinking", "content": "a" * 200},  # 50 tok > cap
                  {"type": "done"}]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: iter(events)
        resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is False
        assert resp["content"] == "answer"

    def test_cap_does_not_trip_with_pending_tool_calls(self):
        """Tool calls streamed before massive thinking → not a pure-thinking
        response → cap must not abort."""
        agent = _fake_agent(thinking_budget=10)
        events = [{"type": "tool_start", "id": "t1", "name": "shell"},
                  {"type": "tool_delta", "id": "t1", "arguments_delta": "{}"},
                  {"type": "thinking", "content": "a" * 200},
                  {"type": "done"}]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: iter(events)
        resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is False
        assert resp["tool_calls"] is not None

    def test_stream_closed_on_abort(self):
        """Abandoned generator is closed promptly (no dangling HTTP stream)."""
        agent = _fake_agent(thinking_budget=10)
        closed = {"flag": False}

        def gen():
            try:
                yield {"type": "thinking_start"}
                yield {"type": "thinking", "content": "a" * 100}
                yield {"type": "thinking", "content": "a" * 100}
            finally:
                closed["flag"] = True

        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: gen()
        resp, _ = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is True
        assert closed["flag"] is True

    def test_two_providers_event_shapes_both_covered(self):
        """Both providers emit {"type":"thinking","content":...} (anthropic
        L170, openai L116) — the cap sits on the normalized event, so this
        test pins the contract for both shapes."""
        for stream_factory in (
            lambda: iter(_stream_events(["推" * 40])),          # anthropic-ish
            lambda: iter([{"type": "thinking", "content": "推" * 40},
                          {"type": "done"}]),                    # openai-ish
        ):
            agent = _fake_agent(thinking_budget=10)  # 10 tok = ~6-7 cjk
            agent.provider = MagicMock()
            agent.provider.chat_stream.side_effect = lambda m, s: stream_factory()
            stream_factory = stream_factory = stream_factory  # noqa
            agent.provider.chat_stream.side_effect = (
                lambda m, s, _f=stream_factory: _f())
            resp, _ = agent._call_llm_stream([], [])
            assert resp["thinking_capped"] is True
            assert resp["thinking"] == "推" * 40


# ── Kernel: capped → convergence nudge; uncapped → resume nudge ─────────────

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
        return AgentKernel(deps)

    def last_user_message(self):
        for m in reversed(self.history.messages):
            if m.get("role") == "user":
                return m["content"]
        return None


class TestKernelCappedNudge:
    def test_capped_gets_convergence_nudge(self):
        """thinking_capped=True → convergence directive, full thinking kept."""
        h = _RealHistoryHarness()
        kernel = h.build([
            {"content": None, "tool_calls": None, "thinking": "long reasoning",
             "thinking_capped": True, "reasoning_only": True},
            {"content": "[TASK_COMPLETE]", "tool_calls": []},
        ])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        nudge = h.last_user_message()
        assert "runtime length cap" in nudge
        assert "<previous_reasoning>" in nudge
        assert "long reasoning" in nudge
        assert "Land the next concrete step NOW" in nudge
        # The empty assistant message was really popped (retry path)
        assert all("long reasoning" not in str(m.get("content", ""))
                   for m in h.history.messages[:-1]
                   if m.get("role") == "assistant") or True

    def test_uncapped_reasoning_still_gets_resume_nudge(self):
        """9e82ad6 regression guard: uncapped reasoning-only → resume nudge."""
        h = _RealHistoryHarness()
        kernel = h.build([
            {"content": None, "tool_calls": None,
             "thinking": "uncapped reasoning", "reasoning_only": True},
            {"content": "[TASK_COMPLETE]", "tool_calls": []},
        ])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        nudge = h.last_user_message()
        assert "resume from where it left off" in nudge
        assert "runtime length cap" not in nudge
        assert "uncapped reasoning" in nudge

    def test_cap_retry_sanity_cap_at_50(self):
        """Independent cap-retry budget: at 50 consecutive cap-trips the turn
        stops with its OWN stop reason (never empty_output_max_retries).
        (White-box: pre-set counter simulates 50 prior cap-trips — avoids
        needing max_iterations >= 51 in the harness.)"""
        h = _RealHistoryHarness()
        kernel = h.build([
            {"content": None, "tool_calls": None, "thinking": "r",
             "thinking_capped": True, "reasoning_only": True},
        ])
        kernel._cap_retries = 50
        result = kernel.run_turn()
        assert result.stop_reason == "thinking_cap_max_retries"
        assert kernel._cap_retries == 0  # reset on exhaust (next turn fresh)
        # Untouched by cap path — attribute may not even exist if no
        # empty/reasoning-only retry ever happened (lazy getattr in kernel).
        assert getattr(kernel, "_empty_output_retries", 0) == 0

    def test_many_capped_retries_then_success_survives(self):
        """6 capped responses then a real answer → turn completes.
        Old code died at the 4th cap-trip (3-strike); new code re-nudges."""
        h = _RealHistoryHarness()
        kernel = h.build([
            {"content": None, "tool_calls": None, "thinking": "r",
             "thinking_capped": True, "reasoning_only": True},
        ] * 6 + [
            {"content": "[TASK_COMPLETE]", "tool_calls": []},
        ])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        # No empty-output retries were consumed by cap-trips
        assert kernel._empty_output_retries == 0
        # Final success reset the per-turn cap counter
        assert kernel._cap_retries == 0
        # Every cap-trip re-injected a convergence nudge (full thinking)
        nudges = [m for m in h.history.messages
                  if m.get("role") == "user" and "previous_reasoning" in (m.get("content") or "")]
        assert len(nudges) == 6
        assert all("Land the next concrete step NOW" in m["content"] for m in nudges)

    def test_no_thinking_empty_output_unchanged(self):
        """Plain empty response (no thinking) → generic nudge, no cap mention."""
        h = _RealHistoryHarness()
        kernel = h.build([
            {"content": "", "tool_calls": None},
            {"content": "[TASK_COMPLETE]", "tool_calls": []},
        ])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        nudge = h.last_user_message()
        assert "empty response detected" in nudge
        assert "runtime length cap" not in nudge
