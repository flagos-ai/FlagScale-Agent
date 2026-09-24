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

"""Tests for the interactive-prompt watchdog.

Scenario: a long interactive session's prompt_toolkit loop wedges — the
process is alive and rendering, but stdin has pending bytes it never reads, so
keystrokes are silently dropped (looks like a frozen terminal).  PromptWatchdog
detects "at prompt + stdin readable but unconsumed" and escalates
SIGWINCH -> SIGINT to force the loop to break and rebuild.

All effects are injected so the decision logic is verified deterministically.
"""

import signal

import pytest

from flagscale_agent.react.prompt_watchdog import PromptWatchdog


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Recorder:
    def __init__(self, pending=True):
        self.pending = pending
        self.kills = []

    def select(self, r, w, x, timeout):
        return ([r[0]], [], []) if self.pending else ([], [], [])

    def kill(self, pid, sig):
        self.kills.append(sig)


def make_watchdog(at_prompt=True, pending=True, clock=None, interval_effects=None):
    clock = clock or FakeClock()
    rec = Recorder(pending=pending)
    wd = PromptWatchdog(
        is_at_prompt=lambda: at_prompt,
        fd=0,
        threshold=60.0,
        winch_grace=15.0,
        select_fn=rec.select,
        monotonic=clock,
        kill_fn=rec.kill,
        getpid=lambda: 999,
        logger=lambda m: None,
    )
    return wd, rec, clock


class TestPromptWatchdog:
    def test_no_action_when_not_at_prompt(self):
        wd, rec, clock = make_watchdog(at_prompt=False)
        clock.advance(120)
        assert wd.tick() is None
        assert rec.kills == []

    def test_no_action_when_input_not_pending(self):
        # Healthy loop: prompt_toolkit consumed every keystroke, so nothing
        # is pending -> watchdog must stay silent forever.
        wd, rec, clock = make_watchdog(pending=False)
        for _ in range(100):
            clock.advance(10)
            assert wd.tick() is None
        assert rec.kills == []

    def test_no_action_before_threshold(self):
        wd, rec, clock = make_watchdog()
        clock.advance(30)  # < 60s threshold
        assert wd.tick() is None
        assert rec.kills == []

    def test_winch_after_threshold(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()  # arm pending_since
        clock.advance(60)
        assert wd.tick() == "winch"
        assert rec.kills == [signal.SIGWINCH]

    def test_winch_fires_only_once(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()
        clock.advance(5)
        assert wd.tick() is None
        assert rec.kills == [signal.SIGWINCH]

    def test_sigint_after_winch_grace(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()  # SIGWINCH
        clock.advance(15)  # winch_grace
        assert wd.tick() == "sigint"
        assert rec.kills == [signal.SIGWINCH, signal.SIGINT]

    def test_consumed_input_resets_and_never_fires(self):
        # The bug-avoidance case: input becomes consumed (loop healthy again)
        # before the threshold -> pending window resets, no signal.
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(30)
        rec.pending = False  # loop consumed it
        assert wd.tick() is None
        rec.pending = True
        clock.advance(30)  # was reset, only 30s since re-arm
        assert wd.tick() is None
        assert rec.kills == []

    def test_on_sigint_callback_invoked(self):
        fired = {"n": 0}

        def cb():
            fired["n"] += 1

        clock = FakeClock()
        rec = Recorder(pending=True)
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            on_sigint=cb,
            fd=0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=rec.select,
            monotonic=clock,
            kill_fn=rec.kill,
            getpid=lambda: 1,
        )
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()
        clock.advance(15)
        wd.tick()
        assert fired["n"] == 1

    def test_missing_fd_never_fires(self):
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=None,
            select_fn=lambda *a: ([], [], []),
            monotonic=FakeClock(),
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        wd._fd = None
        assert wd.tick() is None

    def test_backlog_probe_fires_when_fd_clean(self):
        # Second wedge class: the reader consumed the keystroke (fd no longer
        # readable -> select sees nothing) but the key never got dispatched, so
        # the userspace backlog is non-empty. The watchdog must still fire.
        dead = lambda *a: ([], [], [])  # fd permanently clean
        backlog = {"n": 1}
        kills = []
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=lambda: backlog["n"] > 0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=dead,
            monotonic=(clk := FakeClock()),
            kill_fn=lambda pid, sig: kills.append(sig),
            getpid=lambda: 1,
        )
        clk.advance(1)
        wd.tick()  # arm
        clk.advance(60)
        assert wd.tick() == "winch"
        clk.advance(15)
        assert wd.tick() == "sigint"
        assert kills == [signal.SIGWINCH, signal.SIGINT]

    def test_backlog_probe_cleared_resets(self):
        # If the backlog drains (loop recovered), the window resets — no fire.
        dead = lambda *a: ([], [], [])
        backlog = {"n": 1}
        kills = []
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=lambda: backlog["n"] > 0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=dead,
            monotonic=(clk := FakeClock()),
            kill_fn=lambda pid, sig: kills.append(sig),
            getpid=lambda: 1,
        )
        clk.advance(1)
        wd.tick()
        clk.advance(30)
        backlog["n"] = 0  # consumed -> healthy again
        assert wd.tick() is None
        clk.advance(60)
        assert wd.tick() is None
        assert kills == []

    def test_backlog_probe_exception_is_ignored(self):
        # A probe that raises must not crash the watchdog; it degrades to
        # fd-only detection.
        def boom():
            raise RuntimeError("unexpected")

        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=boom,
            select_fn=(lambda *a: ([], [], [])),
            monotonic=FakeClock(),
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        assert wd.tick() is None

    def test_start_runs_with_only_probe(self):
        # fd is None but a probe exists -> the watcher must still start.
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=None,
            pending_probe=lambda: False,
            interval=100.0,
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        wd._fd = None
        wd.start()
        assert wd._thread is not None and wd._thread.daemon
