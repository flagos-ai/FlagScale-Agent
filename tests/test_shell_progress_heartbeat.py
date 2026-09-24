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

"""Regression tests for shell long-command progress heartbeat + streaming.

Covers two fixes:
1. The [Xm Ys] heartbeat is emitted every check_interval EVEN WHEN the command
   produces no output yet (previously nested inside `if recent:` so a silent
   command showed nothing until exit).
2. Output streams incrementally via stdbuf line-buffering instead of
   full-buffering into the pipe and dumping only at exit.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flagscale_agent.react.tools.shell import ShellTool


def _run_capturing_stdout(tool, command, quiet=False):
    """Run the tool, capturing stdout progress prints separately from result."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = tool.execute(command=command, _quiet=quiet)
    finally:
        sys.stdout = old
    return buf.getvalue(), result


class TestShellProgressHeartbeat:
    def test_heartbeat_prints_with_no_output(self):
        """A silent command prints periodic heartbeats before it finishes."""
        tool = ShellTool(remind_interval=1)  # check_interval = min(30, 1) = 1s
        progress, result = _run_capturing_stdout(tool, "sleep 2; echo DONE_MARKER")
        # Heartbeat must appear during the silent window.
        assert "⏳ [" in progress
        assert "(running, no output yet)" in progress
        # Result is still captured correctly.
        assert "DONE_MARKER" in result

    def test_output_streams_incrementally(self):
        """Output appears in progress across heartbeats, not only at exit."""
        tool = ShellTool(remind_interval=1)
        prog = (
            "python3 -c \""
            "import time,sys\n"
            "for i in range(3):\n"
            "    sys.stdout.write('sline%d\\n'%i)\n"
            "    time.sleep(1)\n"
            "\""
        )
        progress, result = _run_capturing_stdout(tool, prog)
        # Recent output block shown while running.
        assert "Recent output" in progress
        # An early line shows up in progress (streamed), proving not dumped at exit.
        assert "sline0" in progress
        assert result == "sline0\nsline1\nsline2\n"

    def test_quiet_suppresses_progress_but_keeps_result(self):
        """_quiet=True prints no progress but returns full output."""
        tool = ShellTool(remind_interval=1)
        progress, result = _run_capturing_stdout(
            tool, "sleep 2; echo QUIET_MARKER", quiet=True
        )
        assert "⏳ [" not in progress
        assert "QUIET_MARKER" in result

    def test_short_command_correctness_unchanged(self):
        """Fast commands still return correct output (stdbuf/shell parity)."""
        tool = ShellTool()
        _, result = _run_capturing_stdout(tool, "echo a && echo b")
        assert result == "a\nb\n"

    def test_nonzero_exit_still_returns_output(self):
        """A failing command's output is still captured."""
        tool = ShellTool()
        _, result = _run_capturing_stdout(
            tool, "echo before_fail; false"
        )
        assert "before_fail" in result


class TestIdleHangKill:
    """A command that produces no output AND burns no CPU/IO (a hung network
    connect, simulated here by sleep) is killed within ~IDLE_KILL_THRESHOLD
    checks instead of waiting out the 6-minute frozen-output rule."""

    def test_idle_sleep_killed_fast(self):
        import time
        tool = ShellTool(remind_interval=1)  # check_interval = 1s
        t0 = time.time()
        # sleep 60 would hang for a full minute; idle-kill must stop it in ~5-6s
        # (first sample has no baseline, then 4 flat samples reach the threshold).
        progress, result = _run_capturing_stdout(tool, "sleep 60")
        elapsed = time.time() - t0
        assert result.startswith("TERMINATED"), result[:200]
        assert "no output" in result.lower()
        # Killed far sooner than the 6-min frozen rule or 60s sleep.
        assert elapsed < 30, f"idle-kill took {elapsed:.0f}s, expected <30s"

    def test_active_command_not_idle_killed(self):
        """A CPU-burning command with no stdout must NOT be idle-killed —
        cpu_time_delta stays positive so idle_count never accumulates."""
        import time
        tool = ShellTool(remind_interval=1)
        # Busy-loop ~6s, no output. Must complete normally, not TERMINATED.
        prog = (
            "python3 -c \""
            "import time\n"
            "e=time.time()+6\n"
            "x=0\n"
            "while time.time()<e:\n"
            "    x+=1\n"
            "print('BUSY_DONE')\""
        )
        _, result = _run_capturing_stdout(tool, prog)
        assert "BUSY_DONE" in result, result[:200]
        assert not result.startswith("TERMINATED"), result[:200]


class TestHealthJudgeBounded:
    """A stalled health-judge LLM call must not freeze the monitor loop.

    Regression for the compcert case: the synchronous no-timeout judge call ran
    inside the loop BEFORE the heartbeat display, so a wedged gateway suppressed
    the heartbeat for ~12 minutes. The fix wraps the judge in a bounded thread.
    """

    def test_stalled_judge_does_not_block_heartbeat(self):
        """When the judge hangs, heartbeats still appear on schedule."""
        import time

        def _hanging_judge(*args, **kwargs):
            time.sleep(600)  # simulate a wedged LLM gateway
            return {"kill": True, "reason": "should never be seen"}

        # Shorten the timeout so the test runs fast but still exercises the path.
        import flagscale_agent.react.tools.shell as shellmod
        orig_timeout = shellmod._HEALTH_JUDGE_TIMEOUT_SECS
        shellmod._HEALTH_JUDGE_TIMEOUT_SECS = 1
        try:
            tool = ShellTool(remind_interval=1, health_judge_fn=_hanging_judge)
            progress, result = _run_capturing_stdout(
                tool, "sleep 3; echo HB_DONE"
            )
        finally:
            shellmod._HEALTH_JUDGE_TIMEOUT_SECS = orig_timeout

        # Heartbeat fired despite the hanging judge; command completed normally.
        assert "⏳ [" in progress
        assert "HB_DONE" in result
        # The hanging judge's kill decision was abandoned (timed out -> None),
        # so the command was NOT terminated by the monitor.
        assert "TERMINATED" not in result

    def test_bounded_helper_returns_none_on_timeout(self):
        """_run_health_judge_bounded returns None when the fn exceeds timeout."""
        import time
        from flagscale_agent.react.tools.shell import _run_health_judge_bounded

        def _slow():
            time.sleep(5)
            return {"kill": True}

        out = _run_health_judge_bounded(lambda *a, **k: _slow(), (), {}, timeout=1)
        assert out is None

    def test_bounded_helper_passes_through_fast_decision(self):
        """A fast judge decision is returned unchanged."""
        from flagscale_agent.react.tools.shell import _run_health_judge_bounded

        def _fast(cmd, out, el, **kw):
            return {"kill": False, "reason": "healthy"}

        out = _run_health_judge_bounded(
            _fast, ("cmd", "output", "1s"),
            {"output_changed": True, "stall_count": 0, "activity": ""},
            timeout=5,
        )
        assert out == {"kill": False, "reason": "healthy"}

    def test_bounded_helper_swallows_exception(self):
        """A judge that raises yields None, not a propagated exception."""
        from flagscale_agent.react.tools.shell import _run_health_judge_bounded

        def _boom(*a, **k):
            raise RuntimeError("gateway error")

        out = _run_health_judge_bounded(_boom, (), {}, timeout=2)
        assert out is None


class TestProcessHealthBounded:
    """Regression: get_process_health must not be able to block the monitor loop.

    A make -j / opam-coq build explodes the child-process tree; the per-process
    /proc walk in get_process_health can take minutes under heavy IO, starving the
    heartbeat. The loop now samples it through _run_health_judge_bounded with a hard
    cap, falling back to a neutral 'alive & working' reading on timeout.
    """

    def test_slow_health_sample_times_out_not_blocks(self):
        """A get_process_health that hangs returns None within the cap (the stall)."""
        import time
        from flagscale_agent.react.tools.shell import _run_health_judge_bounded

        def _slow_health(pid):
            time.sleep(30)  # simulate a multi-minute /proc tree walk
            return {'alive': True}

        t0 = time.time()
        out = _run_health_judge_bounded(_slow_health, (1234,), {}, timeout=2)
        elapsed = time.time() - t0
        assert out is None
        assert elapsed < 5, f"bounded call took {elapsed:.1f}s, should abandon ~2s"

    def test_process_health_timeout_constant_below_interval(self):
        """The health-sample cap must be well under the ~30s check interval so a
        slow sample cannot swallow a whole heartbeat window."""
        from flagscale_agent.react.tools import shell as shellmod
        assert shellmod._PROCESS_HEALTH_TIMEOUT_SECS < 30
        assert shellmod._PROCESS_HEALTH_TIMEOUT_SECS > 0

    def test_fast_health_sample_passes_through(self):
        """A normal (fast) health sample returns its real dict unchanged."""
        from flagscale_agent.react.tools.shell import _run_health_judge_bounded

        real = {'alive': True, 'zombie': False, 'num_children': 7,
                'children_alive': 7, 'cpu_time': 3.0, 'io_bytes': 100,
                'cpu_percent': 55.0, 'memory_mb': 42.0}
        out = _run_health_judge_bounded(lambda pid: real, (99,), {}, timeout=5)
        assert out is real


import re
import threading
import time as _time


class TestHeartbeatNeverStarvedBySlowHealth:
    """Reproduce the compcert 8-minute heartbeat gap and prove it's fixed.

    Root cause (old design): the monitor loop called get_process_health on its
    OWN thread every check_interval. Under a giant build tree a single /proc
    walk blocked for minutes; the old per-check bounded threads piled up and
    the main heartbeat starved. Fix: one background _HealthSampler thread; the
    main loop only reads latest() (never blocks), so the heartbeat fires on
    schedule regardless of /proc latency.
    """

    def _run_with_slow_health(self, sleep_secs, cmd, remind_interval=1):
        import flagscale_agent.react.tools.process_health as ph
        from flagscale_agent.react.tools.shell import ShellTool

        orig = ph.get_process_health

        def _slow(pid):
            _time.sleep(sleep_secs)
            return orig(pid)

        ph.get_process_health = _slow
        try:
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                tool = ShellTool(remind_interval=remind_interval)
                result = tool.execute(command=cmd)
            finally:
                sys.stdout = old
            return buf.getvalue(), result
        finally:
            ph.get_process_health = orig

    def test_heartbeat_fires_on_schedule_despite_slow_health(self):
        progress, result = self._run_with_slow_health(20, "sleep 6; echo GAP_DONE")
        beats = [int(x) for x in re.findall(r'⏳ \[(\d+)s\]', progress)]
        assert len(beats) >= 4, (
            f"HEARTBEAT STARVED by slow health sample: only {len(beats)} beats "
            f"in ~6s (expected >=4 at 1s cadence). beats={beats}"
        )
        gaps = [b2 - b1 for b1, b2 in zip(beats, beats[1:])]
        assert all(g <= 3 for g in gaps), (
            f"heartbeat cadence dragged by slow health: gaps={gaps} (want ~1s)"
        )
        assert "GAP_DONE" in result

    def test_no_sampler_thread_leak_after_command(self):
        before = threading.active_count()
        self._run_with_slow_health(2, "echo QUICK", remind_interval=1)
        _time.sleep(0.3)
        after = threading.active_count()
        assert after <= before + 1, (
            f"sampler/reader threads leaked: before={before} after={after}"
        )
