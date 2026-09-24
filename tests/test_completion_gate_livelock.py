# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the single-shot completion path (post completion-gate removal).

History: there used to be a NON-OVERRIDABLE PlanGuard completion gate that
blocked a bare [TASK_COMPLETE] when no plan was ever created, backed by a
kernel livelock breaker (stop_reason 'completion_gate_livelock'). That gate
punished weak models — they neither reliably override nor plan_create, so the
loop burned MAX_CONSECUTIVE_COMPLETION_BLOCKS and auto-killed the task (reward 0).

The gate is REMOVED. Completing without a plan is no longer a hard failure;
plan enforcement moved earlier to the write_file gate (overridable). These
tests assert the new reality:
  - single-shot [TASK_COMPLETE] with no plan → completes (explicit_signal),
  - the completion marker prints correctly (display/sentinel behavior is
    orthogonal to the removed gate and must still hold).
The kernel's livelock-breaker branch is kept as dead-but-harmless code (its
trigger reason 'single_shot_completion_without_plan' is no longer produced).
"""

from unittest.mock import MagicMock
import types

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry
from flagscale_agent.react.guard.plan import PlanGuard


class _History:
    """Minimal history that supports the methods run_turn touches."""

    def __init__(self):
        self.messages = [{"role": "user", "content": "do the task"}]

    def append(self, msg):
        self.messages.append(msg)

    def get_messages(self):
        return self.messages

    def get_context_pressure(self):
        return 0.1

    def get_evictable_indexes(self):
        return []

    def report_actual_tokens(self, n):
        pass


def _make_kernel(llm_responses, single_shot=True, max_iter=2000):
    """Build an AgentKernel driving a real PlanGuard through run_turn.

    llm_responses: list of dicts shaped like provider responses:
        {"content": "<text>", "tool_calls": [...]}  (tool_calls may be [])
    The last response is repeated once the list is exhausted.
    """
    history = _History()

    registry = GuardRegistry()
    registry.register(PlanGuard(task_plan=None, single_shot=single_shot))

    config = types.SimpleNamespace(
        max_iterations=max_iter, max_continuations=200, mode="auto",
        _turn_count=0,
    )

    call_count = {"n": 0}

    def call_llm_fn(messages, schemas):
        idx = min(call_count["n"], len(llm_responses) - 1)
        call_count["n"] += 1
        resp = llm_responses[idx]
        usage = {"input_tokens": 1, "output_tokens": 1}
        return resp, usage

    def format_assistant_message(resp):
        return {"role": "assistant", "content": resp.get("content", "")}

    provider = MagicMock()
    provider.format_assistant_message.side_effect = format_assistant_message

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=registry,
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda msg: history.append({"role": "user", "content": msg}),
        append_tool_results_fn=lambda results: None,
        format_tool_result_fn=lambda tid, r: {},
        execute_tools_fn=lambda tcs: [],
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
    )
    # execute_tools_fn must return one result per tool call so the kernel's
    # zip(tool_calls, results) pairing (and PlanGuard.check_post on plan_create)
    # runs. The default lambda above returns [] which drops single tool calls;
    # override with a length-matching stub.
    deps.execute_tools_fn = lambda tcs: ["ok"] * len(tcs)
    kernel = AgentKernel(deps)
    kernel._call_count = call_count
    return kernel


def test_bare_task_complete_single_shot_completes():
    """Single-shot [TASK_COMPLETE] with no plan now COMPLETES (gate removed).

    Previously this livelocked and auto-killed the task. It must now finish
    normally on the first iteration — no gate, no spin.
    """
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]", "tool_calls": []}],
        single_shot=True,
        max_iter=2000,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal", result.stop_reason
    assert kernel._call_count["n"] == 1
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_plan_create_then_complete_still_works():
    """A run that DID create a plan still completes normally."""
    kernel = _make_kernel(
        [
            {"content": "framing", "tool_calls": [
                {"id": "t1", "name": "plan_create",
                 "arguments": {"title": "T", "steps": ["a"]}}]},
            {"content": "done [TASK_COMPLETE]", "tool_calls": []},
        ],
        single_shot=True,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal", result.stop_reason
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_interactive_mode_completes_normally():
    """Interactive (non-single-shot) mode: completion always passes."""
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal", result.stop_reason
    assert kernel._call_count["n"] == 1


def test_accepted_completion_prints_marker_once():
    """A gate-approved completion (released by plan_create) prints the
    authoritative marker exactly once."""
    kernel = _make_kernel(
        [
            {"content": "framing", "tool_calls": [
                {"id": "t1", "name": "plan_create",
                 "arguments": {"title": "T", "steps": ["a"]}}]},
            {"content": "[TASK_COMPLETE]", "tool_calls": []},
        ],
        single_shot=True,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_interactive_completion_prints_marker():
    """Interactive-mode completion (gate never fires) still prints the marker."""
    kernel = _make_kernel(
        [{"content": "all done [TASK_COMPLETE]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_need_user_input_prints_that_marker():
    """A completion ending in [NEED_USER_INPUT] prints THAT marker, not the other."""
    kernel = _make_kernel(
        [{"content": "here is my question [NEED_USER_INPUT]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[NEED_USER_INPUT]")


def test_completion_gate_verdict_removed_no_block():
    """PlanGuard no longer blocks a text-only [TASK_COMPLETE] — the gate is gone.
    A completion signal with no tool_name returns no verdict regardless of plan."""
    from flagscale_agent.react.guard import GuardContext
    guard = PlanGuard(task_plan=None, single_shot=True)
    ctx = GuardContext(
        tool_name="", tool_args={}, tool_result=None,
        assistant_text="[TASK_COMPLETE]",
    )
    assert guard.check_pre(ctx) is None


def test_trailing_sentinel_wins_over_earlier_mention():
    """If the text QUOTES one sentinel earlier but ENDS with the other, the
    trailing (last-occurring) sentinel is the authoritative one displayed.

    Regression: the fixed tuple-order loop always printed [TASK_COMPLETE] first,
    so a response quoting "[TASK_COMPLETE]" while explaining code but actually
    ending with [NEED_USER_INPUT] showed the wrong marker on screen.
    """
    kernel = _make_kernel(
        [{"content": "the marker [TASK_COMPLETE] is printed here; "
                     "but I still need input [NEED_USER_INPUT]",
          "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[NEED_USER_INPUT]")
