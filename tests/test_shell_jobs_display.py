"""Tests for shell_jobs terminal display: icon, arg summary, and the live
wait-spinner hint that surfaces job status/output/health instead of a frozen
'shell_jobs' line."""

import time

import pytest

from flagscale_agent.react import display
from flagscale_agent.react.display import _tool_icon
from flagscale_agent.react.tool_executor import tool_display_summary
from flagscale_agent.react.tools.shell import (
    ShellTool,
    ShellJobsTool,
    _JOB_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0
    display._active_spinner = None
    yield
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0
    display._active_spinner = None


# ── Icon ────────────────────────────────────────────────────────────────

def test_shell_jobs_has_dedicated_icon():
    icon = _tool_icon("shell_jobs")
    assert icon != _tool_icon("some_unknown_tool")  # not the generic fallback
    assert icon  # non-empty


# ── Arg summary ─────────────────────────────────────────────────────────

def test_summary_wait_shows_action_job_timeout():
    s = tool_display_summary("shell_jobs", {"action": "wait", "job_id": "job1", "timeout": 120})
    assert s.startswith("wait")
    assert "job1" in s
    assert "120" in s


def test_summary_poll_shows_action_and_job():
    s = tool_display_summary("shell_jobs", {"action": "poll", "job_id": "job2"})
    assert s == "poll job2"


def test_summary_list_shows_action_only():
    assert tool_display_summary("shell_jobs", {"action": "list"}) == "list"


def test_summary_kill_shows_action_and_job():
    assert tool_display_summary("shell_jobs", {"action": "kill", "job_id": "job3"}) == "kill job3"


def test_summary_missing_action_is_safe():
    # No crash on empty/partial args.
    assert tool_display_summary("shell_jobs", {}) == "?"


# ── tool_start spinner selection ─────────────────────────────────────────

def test_tool_start_spins_only_for_wait(monkeypatch):
    started = {}

    class _FakeSpinner:
        def __init__(self, prefix=""):
            started["created"] = True
        def start(self):
            started["started"] = True
        def stop(self):
            pass

    monkeypatch.setattr(display, "_Spinner", _FakeSpinner)
    monkeypatch.setattr(display, "_use_color", lambda: True)

    # wait → spins
    started.clear()
    display.tool_start("shell_jobs", "wait job1 (≤60s)")
    assert started.get("started") is True
    display._stop_all_spinners()

    # poll → does not spin
    started.clear()
    display.tool_start("shell_jobs", "poll job1")
    assert started.get("started") is None
    display._stop_all_spinners()


# ── Live wait hint ──────────────────────────────────────────────────────

def _launch(cmd):
    ShellTool().execute(command=cmd, background=True)
    return _JOB_REGISTRY.all()[0].job_id


def test_set_wait_hint_includes_status_output_activity():
    job_id = _launch("for i in 1 2 3; do echo line-$i; sleep 0.4; done")
    job = _JOB_REGISTRY.get(job_id)
    # Let some output accumulate.
    assert _wait_for_output(job, "line-")
    display._active_spinner = display._Spinner()
    sj = ShellJobsTool()
    sj._set_wait_hint(job, activity="cpu busy 42%", finished=False)
    with display._active_spinner._hint_lock:
        hint = display._active_spinner._hint
    assert job_id in hint
    assert "[running]" in hint
    assert "line-" in hint
    assert "cpu busy 42%" in hint


def test_set_wait_hint_no_spinner_is_noop():
    job_id = _launch("sleep 5")
    job = _JOB_REGISTRY.get(job_id)
    display._active_spinner = None
    # Should not raise when there is no active spinner.
    ShellJobsTool()._set_wait_hint(job, activity="x", finished=False)


def test_set_wait_hint_does_not_consume_output():
    """The hint peeks at output; the final wait/poll return must still see it."""
    job_id = _launch("echo important-line; sleep 3")
    job = _JOB_REGISTRY.get(job_id)
    assert _wait_for_output(job, "important-line")
    display._active_spinner = display._Spinner()
    ShellJobsTool()._set_wait_hint(job, activity="", finished=False)
    # new_output cursors must be untouched → poll still returns the line.
    out = ShellJobsTool().execute(action="poll", job_id=job_id)
    assert "important-line" in out


def _wait_for_output(job, needle, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if needle in job.full_output():
            return True
        time.sleep(0.05)
    return False
