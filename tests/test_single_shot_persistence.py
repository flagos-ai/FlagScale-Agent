"""Regression tests for single-shot (headless) trajectory persistence.

Background: headless single-shot runs (used by the Terminal-Bench / Harbor
adapter) previously only relied on the atexit hook to persist the trajectory.
A harness timeout (SIGKILL / unhandled SIGTERM) skips atexit, so the whole
conversation was lost — FLAGSCALE_HOME ended up with only an empty swap_store.

Fix (flagscale_agent/react/agent.py):
  1. _run_single_shot() saves in a finally: block after _react_loop(), so a
     normally-completed headless run is always durable.
  2. _install_signal_handlers() installs a SIGTERM/SIGINT handler that flushes
     the trajectory before the process dies. atexit does NOT run on SIGTERM
     (nor SIGKILL); Harbor timeouts send SIGTERM then SIGKILL, so catching
     SIGTERM covers the common timeout case (SIGKILL remains uncatchable).

These tests assert conversation.json is written on the corresponding events.
"""
import os
import signal
from unittest.mock import Mock

import pytest

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.config import AgentConfig
from flagscale_agent.react.memory import Memory
from flagscale_agent.react.plan import TaskPlan


def _make_agent(tmp_path, monkeypatch):
    """Construct a WorkerAgent with mocked provider/memory/task_plan.

    Mirrors tests/test_agent_init.py::test_agent_construction_smoke.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")

    mock_provider = Mock()
    mock_provider.count_tokens.return_value = 100
    mock_memory = Mock(spec=Memory)
    mock_task_plan = Mock(spec=TaskPlan)

    session_dir = str(tmp_path / "sess")
    config = AgentConfig(
        session_dir=session_dir,
        api_key="test-key-12345",
        provider="anthropic",
        max_context_tokens=50000,
    )
    agent = WorkerAgent(
        config,
        _provider=mock_provider,
        _memory=mock_memory,
        _task_plan=mock_task_plan,
    )
    # WorkerAgent nests the real session dir under config.session_dir/<id>.
    return agent, agent._session_dir


def _conv_path(session_dir):
    return os.path.join(session_dir, "conversation.json")


def test_single_shot_saves_on_completion(tmp_path, monkeypatch):
    """_run_single_shot() persists conversation.json even when _react_loop
    completes normally (no atexit dependency)."""
    agent, session_dir = _make_agent(tmp_path, monkeypatch)

    # Stub out the react loop and context injection so we exercise only the
    # persistence contract, not the LLM/kernel machinery.
    agent._inject_context = Mock()
    agent._react_loop = Mock()

    assert not os.path.exists(_conv_path(session_dir))

    agent._run_single_shot("do the task")

    assert os.path.exists(_conv_path(session_dir)), (
        "single-shot run must write conversation.json on completion"
    )
    # The user query must be in history and therefore persisted.
    assert any(
        m.get("role") == "user" and "do the task" in str(m.get("content", ""))
        for m in agent.history.messages
    )


def test_single_shot_saves_even_when_loop_raises(tmp_path, monkeypatch):
    """A crash inside _react_loop must not prevent the finally: save."""
    agent, session_dir = _make_agent(tmp_path, monkeypatch)

    agent._inject_context = Mock()
    agent._react_loop = Mock(side_effect=RuntimeError("boom"))

    # Should not propagate; save still happens.
    agent._run_single_shot("crashing task")

    assert os.path.exists(_conv_path(session_dir)), (
        "conversation.json must be saved in finally even if the loop raises"
    )


def test_sigterm_handler_persists_before_exit(tmp_path, monkeypatch):
    """The SIGTERM handler flushes conversation.json before the process dies.

    Harbor enforces timeouts by sending SIGTERM (then SIGKILL after a grace
    period). atexit does not run on SIGTERM, so a dedicated handler must save.
    We invoke the installed handler directly and stub the re-raise/exit so the
    test process survives, then assert the save happened.
    """
    agent, session_dir = _make_agent(tmp_path, monkeypatch)

    # Populate history so save_conversation has something to write.
    agent.history.append({"role": "user", "content": "hello"})
    agent.history.append({"role": "assistant", "content": "working"})

    # Grab the handler the agent installed for SIGTERM.
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), "agent must install a SIGTERM handler"

    assert not os.path.exists(_conv_path(session_dir))

    # Prevent the handler from actually terminating the test process.
    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(os, "kill", lambda *a, **k: None)

    handler(signal.SIGTERM, None)

    assert os.path.exists(_conv_path(session_dir)), (
        "SIGTERM handler must persist conversation.json before exit"
    )
