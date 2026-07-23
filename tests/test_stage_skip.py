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

"""Tests for routing downgrade when subtask stages are already completed in history.

When a subtask workflow (e.g., precision-alignment 6 stages) has already been executed
and all stages show OK in conversation history, routing should NOT re-execute the pipeline.
Instead, it should downgrade to single mode with the existing results as context.
"""

import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.orchestrator import (
    Orchestrator,
    SubtaskDefinition,
    SubtaskTemplate,
)


def _make_orchestrator_with_template(template_name, stage_ids):
    """Create an Orchestrator with a mock template for testing."""
    o = Orchestrator(provider=None, tool_registry=None)
    subtasks = []
    prev = []
    for sid in stage_ids:
        subtasks.append(SubtaskDefinition(
            id=sid,
            description=f"Check {sid}",
            profile_name="general",
            depends_on=prev.copy(),
            upstream_keys=prev.copy(),
        ))
        prev = [sid]
    o.subtask_runner._templates[template_name] = SubtaskTemplate(
        name=template_name,
        description=f"Test template: {template_name}",
        subtasks=subtasks,
    )
    return o


def _history_with_stages(stage_ids, statuses=None):
    """Build conversation history with stage result markers."""
    if statuses is None:
        statuses = ["OK"] * len(stage_ids)
    total = len(stage_ids)
    msgs = []
    for i, (sid, status) in enumerate(zip(stage_ids, statuses), 1):
        summary = f"{sid} verified" if status == "OK" else f"{sid} problem"
        msgs.append({
            "role": "user",
            "content": (
                f"[system: task stage result]\n"
                f"[Stage {i}/{total}] {sid}: {status} — {summary}"
            ),
        })
    return msgs


class TestCheckStagesCompletedInHistory:
    """Tests for Orchestrator.check_stages_completed_in_history."""

    def test_empty_history_returns_none(self):
        o = _make_orchestrator_with_template("test", ["a", "b", "c"])
        route = {"mode": "subtask", "template": "test"}
        result = o.check_stages_completed_in_history(route, "run test", [])
        assert result is None

    def test_partial_completion_returns_none(self):
        o = _make_orchestrator_with_template("test", ["a", "b", "c"])
        route = {"mode": "subtask", "template": "test"}
        history = _history_with_stages(["a", "b"])
        result = o.check_stages_completed_in_history(route, "run test", history)
        assert result is None

    def test_all_ok_returns_dict(self):
        o = _make_orchestrator_with_template("test", ["a", "b", "c"])
        route = {"mode": "subtask", "template": "test"}
        history = _history_with_stages(["a", "b", "c"])
        result = o.check_stages_completed_in_history(route, "run test", history)
        assert result is not None
        assert set(result.keys()) == {"a", "b", "c"}
        assert all("verified" in v for v in result.values())

    def test_interrupted_stage_returns_none(self):
        o = _make_orchestrator_with_template("test", ["a", "b"])
        route = {"mode": "subtask", "template": "test"}
        history = _history_with_stages(["a", "b"], ["OK", "INTERRUPTED"])
        result = o.check_stages_completed_in_history(route, "run test", history)
        assert result is None

    def test_failed_stage_returns_none(self):
        o = _make_orchestrator_with_template("test", ["a", "b"])
        route = {"mode": "subtask", "template": "test"}
        history = _history_with_stages(["a", "b"], ["OK", "FAILED"])
        result = o.check_stages_completed_in_history(route, "run test", history)
        assert result is None

    def test_precision_alignment_6_stages(self):
        """Real-world scenario: all 6 precision alignment stages completed."""
        stage_ids = [
            "structure", "hyperparams", "data",
            "init", "loss_curve", "forward_backward",
        ]
        o = _make_orchestrator_with_template("train-precision-alignment", stage_ids)
        route = {"mode": "subtask", "template": "train-precision-alignment"}
        history = _history_with_stages(stage_ids)
        result = o.check_stages_completed_in_history(route, "精度对齐", history)
        assert result is not None
        assert len(result) == 6
        assert "forward_backward" in result

    def test_mixed_messages_in_history(self):
        """Stage results mixed with normal user/assistant messages."""
        o = _make_orchestrator_with_template("test", ["a", "b"])
        route = {"mode": "subtask", "template": "test"}
        history = [
            {"role": "user", "content": "start precision alignment"},
            {"role": "assistant", "content": "OK, I'll begin."},
            {"role": "user", "content": "[system: task stage result]\n[Stage 1/2] a: OK — a done"},
            {"role": "assistant", "content": "Stage 1 complete."},
            {"role": "user", "content": "[system: task stage result]\n[Stage 2/2] b: OK — b done"},
            {"role": "assistant", "content": "All done."},
            {"role": "user", "content": "now check something else"},
        ]
        result = o.check_stages_completed_in_history(route, "test", history)
        assert result is not None
        assert result == {"a": "a done", "b": "b done"}

    def test_no_template_returns_none(self):
        """Route without a valid template should return None gracefully."""
        o = Orchestrator(provider=None, tool_registry=None)
        route = {"mode": "subtask", "template": "nonexistent"}
        result = o.check_stages_completed_in_history(route, "test", [])
        assert result is None


class TestBuildStageHistoryContext:
    """Tests for WorkerAgent._build_stage_history_context."""

    def _make_agent(self, messages):
        from flagscale_agent.react.agent import WorkerAgent
        agent = WorkerAgent.__new__(WorkerAgent)
        agent.history = MagicMock()
        agent.history.messages = messages
        return agent

    def test_no_stages_returns_empty(self):
        agent = self._make_agent([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        result = agent._build_stage_history_context()
        assert result == ""

    def test_all_stages_ok(self):
        agent = self._make_agent([
            {"role": "user", "content": "[system: task stage result]\n[Stage 1/2] foo: OK — done"},
            {"role": "user", "content": "[system: task stage result]\n[Stage 2/2] bar: OK — done"},
        ])
        result = agent._build_stage_history_context()
        assert "PRIOR COMPLETED STAGES" in result
        assert "foo:OK" in result
        assert "bar:OK" in result
        assert "Do NOT re-run" in result

    def test_mixed_status(self):
        agent = self._make_agent([
            {"role": "user", "content": "[system: task stage result]\n[Stage 1/3] a: OK — ok"},
            {"role": "user", "content": "[system: task stage result]\n[Stage 2/3] b: FAILED — err"},
            {"role": "user", "content": "[system: task stage result]\n[Stage 3/3] c: OK — ok"},
        ])
        result = agent._build_stage_history_context()
        assert "a:OK" in result
        assert "b:FAILED" in result
        assert "c:OK" in result
