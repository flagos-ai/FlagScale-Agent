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

"""Regression + behavior tests for WorkerAgent._health_judge.

Bug: the shell monitor loop calls _health_judge with keyword args
(command_history, container_resources, health_advisory) that the previous
signature did not accept, so every call raised TypeError. The bounded worker
swallows the exception and returns None, silently disabling the LLM health
judge. These tests exercise the exact call shape the loop uses and verify the
task-budget summary is computed from the env var.
"""

import inspect
from types import SimpleNamespace

from flagscale_agent.react.agent import WorkerAgent


def test_health_judge_accepts_shell_loop_kwargs_signature():
    # The signature must tolerate the loop's kwargs — either by declaring them
    # or via **kwargs — so the call never raises TypeError.
    sig = inspect.signature(WorkerAgent._health_judge)
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    for name in ("command_history", "container_resources", "health_advisory"):
        assert name in params or accepts_var_kw, (
            f"_health_judge cannot accept {name!r} — the shell loop's call "
            f"would raise TypeError and silently disable the LLM judge."
        )


class _FakeJudge:
    def __init__(self):
        self.last_kwargs = None

    def health(self, command, recent_output, elapsed,
               output_changed=True, stall_count=0, **kwargs):
        self.last_kwargs = kwargs
        return {"kill": False}


class _Stub:
    """Minimal stand-in exposing just what _health_judge touches."""

    _health_judge = WorkerAgent._health_judge
    _task_budget_summary = WorkerAgent._task_budget_summary
    _task_budget_stats = WorkerAgent._task_budget_stats
    _current_expectation_anchor = WorkerAgent._current_expectation_anchor

    def __init__(self, turn_start, judge):
        # Budget elapsed is measured from the CURRENT turn's start, not session
        # start (interactive sessions span many turns; only the active turn's
        # time counts against a per-task budget).
        self._turn_start = turn_start
        self.judge = judge
        # _task_budget_stats reads the config-level budget first (agent.py
        # getattr(self.config, "time_budget_sec", 0.0)). These tests exercise
        # the ENV-VAR fallback path, so the stub config carries 0.0 — same
        # semantics as a default AgentConfig with no --time-budget-sec flag.
        self.config = SimpleNamespace(time_budget_sec=0.0)

        class _NoPlan:
            def get_active(self_inner):
                return None

        self.task_plan = _NoPlan()


def test_health_judge_forwards_loop_kwargs_without_error(monkeypatch):
    import time
    monkeypatch.delenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", raising=False)
    judge = _FakeJudge()
    stub = _Stub(turn_start=time.time(), judge=judge)
    # Exact call shape from shell.py's monitor loop.
    result = stub._health_judge(
        "cmd", "recent out", "5m",
        output_changed=True, stall_count=0, activity="CPU 50%",
        command_history="ran 3x", container_resources="8GB",
        health_advisory="silent stall suspected",
    )
    assert result == {"kill": False}
    assert judge.last_kwargs.get("command_history") == "ran 3x"
    assert judge.last_kwargs.get("container_resources") == "8GB"


def test_task_budget_summary_disabled_when_env_unset(monkeypatch):
    # Unset env -> NO external wall enforced -> budget reporting disabled.
    # We deliberately do NOT fabricate a 24h code-uniformity default: reporting
    # one would tell the agent/judge it has day-scale time and invite leisurely
    # work. Byte-identical to the no-budget health prompt, and TimeBudgetGuard
    # stays silent.
    import time
    monkeypatch.delenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", raising=False)
    stub = _Stub(turn_start=time.time(), judge=_FakeJudge())
    assert stub._task_budget_summary() == ""
    assert stub._task_budget_stats() is None


def test_task_budget_summary_disabled_when_env_zero(monkeypatch):
    # Explicit 0 (or negative) disables budget reporting -> byte-identical
    # no-budget health prompt.
    import time
    monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "0")
    stub = _Stub(turn_start=time.time(), judge=_FakeJudge())
    assert stub._task_budget_summary() == ""
    monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "-5")
    assert stub._task_budget_summary() == ""


def test_task_budget_summary_reports_elapsed_and_budget(monkeypatch):
    import time
    monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "3600")
    # Current turn started 1800s ago -> ~50% used.
    stub = _Stub(turn_start=time.time() - 1800, judge=_FakeJudge())
    summary = stub._task_budget_summary()
    assert summary != ""
    assert "1h00m" in summary          # total budget 3600s
    assert "50% used" in summary or "49% used" in summary or "51% used" in summary


def test_task_budget_summary_invalid_env_disables(monkeypatch):
    # Unparseable value -> treated as no injected wall (disabled), NOT a 24h
    # fallback. We never fabricate a deadline from a garbage env value.
    import time
    monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "not-a-number")
    stub = _Stub(turn_start=time.time(), judge=_FakeJudge())
    assert stub._task_budget_summary() == ""
    assert stub._task_budget_stats() is None


def test_task_budget_measured_from_turn_not_session(monkeypatch):
    # Regression: an interactive session may be hours old, but budget elapsed
    # must reflect only the CURRENT turn. A stub whose session began long ago
    # but whose turn just started should read ~0% used, not near-expired.
    import time
    monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "3600")
    stub = _Stub(turn_start=time.time(), judge=_FakeJudge())
    # Simulate a long-lived session that started well before this turn.
    stub._session_start = time.time() - 100000
    summary = stub._task_budget_summary()
    # Elapsed is turn-based (~0s), so ~0% used — session age is irrelevant.
    assert "0% used" in summary


def test_react_loop_restamps_turn_start():
    # The loop must re-stamp _turn_start each turn so per-turn budget accounting
    # resets. Verify the source of _react_loop assigns _turn_start.
    import inspect
    src = inspect.getsource(WorkerAgent._react_loop)
    assert "self._turn_start" in src, (
        "_react_loop must re-stamp self._turn_start so task-budget elapsed "
        "resets per turn in long-lived interactive sessions."
    )
