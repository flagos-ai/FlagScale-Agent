# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression test for completion-path advisory / marker ordering.

Observed live (mteb-retrieve run, pre sentinel-stripper fix):
    [TASK_COMPLETE]                       <- streamed raw, BEFORE the gate
    🛡 [MemoryDiscipline] About to ...     <- advisory from the completion gate

The MemoryDiscipline "about to TASK_COMPLETE" reminder is an *inject* verdict
returned from the completion-path guard consultation. The bug was purely that
the raw sentinel streamed live ahead of the gate. After the sentinel-stripper
fix the marker is stripped from the stream and only printed by the kernel via
display.completion_signal() AFTER the gate consultation runs — so any inject
advisory (MemoryDiscipline, KnowledgeSkill, etc.) now displays BEFORE the
authoritative marker.

This test locks that ordering in.
"""

from unittest.mock import MagicMock
import types

import flagscale_agent.react.kernel as kernel_mod
from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict


class _History:
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


class _StubInjectGuard(Guard):
    """Fires an inject advisory on the completion consultation (tool_name empty)."""

    name = "stub_inject"
    priority = 90

    def check_pre(self, ctx):
        if not ctx.tool_name and "[TASK_COMPLETE]" in (ctx.assistant_text or ""):
            return GuardVerdict.inject(
                "[StubAdvisory] about to complete — advisory shown first",
                reason="stub_advisory",
                category="stub_advisory",
            )
        return None


def _make_kernel(shared_display):
    history = _History()
    registry = GuardRegistry()
    registry.register(_StubInjectGuard())

    config = types.SimpleNamespace(
        max_iterations=10, max_continuations=200, mode="auto", _turn_count=0,
    )

    def call_llm_fn(messages, schemas):
        return {"content": "[TASK_COMPLETE]", "tool_calls": []}, {
            "input_tokens": 1, "output_tokens": 1}

    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda resp: {"role": "assistant", "content": resp.get("content", "")})

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=registry,
        config=config,
        display=shared_display,
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda msg: history.append({"role": "user", "content": msg}),
        append_tool_results_fn=lambda results: None,
        format_tool_result_fn=lambda tid, r: {},
        execute_tools_fn=lambda tcs: [],
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
    )
    # append_advisory_fn is optional; leaving it None routes inject to
    # inject_message_fn + module-level display.guard_inject.
    return AgentKernel(deps)


def test_advisory_displays_before_completion_marker(monkeypatch):
    """A completion-path inject advisory must reach the screen BEFORE the marker.

    Both the advisory (display.guard_inject, module-level) and the marker
    (deps.display.completion_signal) are routed through one shared mock so we
    can assert their relative call order — the exact ordering the user saw
    inverted in the pre-fix log.
    """
    shared = MagicMock()
    # Route the module-level display (used by _apply_verdict -> guard_inject)
    # and deps.display (completion_signal) to the SAME mock.
    monkeypatch.setattr(kernel_mod, "display", shared)
    kernel = _make_kernel(shared)

    result = kernel.run_turn()

    assert result.stop_reason == "explicit_signal", result.stop_reason

    names = [c[0] for c in shared.mock_calls]
    assert "guard_inject" in names, names
    assert "completion_signal" in names, names
    # Ordering: advisory first, authoritative marker second.
    assert names.index("guard_inject") < names.index("completion_signal"), names
    shared.completion_signal.assert_called_once_with("[TASK_COMPLETE]")
