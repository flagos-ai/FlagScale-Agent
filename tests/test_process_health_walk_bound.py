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

"""Tests for the bounded, GIL-yielding process-health walk.

Regression coverage for the caffe -j256 heartbeat-starvation bug: a giant
build process tree must NOT make a single health sample unbounded. The walk
caps the expensive per-process cpu_times()/io_counters() reads at
MAX_CHILDREN_WALK and periodically releases the GIL so the main heartbeat
thread stays scheduled.
"""

import threading
import time

import pytest

import flagscale_agent.react.tools.process_health as ph
from flagscale_agent.react.tools.process_health import get_process_health


class _FakeCpuTimes:
    def __init__(self, user, system):
        self.user = user
        self.system = system


class _FakeIO:
    def __init__(self, read_bytes, write_bytes):
        self.read_bytes = read_bytes
        self.write_bytes = write_bytes


class _FakeProc:
    """A fake psutil.Process whose per-proc reads count how often they run."""

    def __init__(self, pid, children, counter, read_delay=0.0):
        self._pid = pid
        self._children = children
        self._counter = counter  # dict shared across the tree
        self._read_delay = read_delay

    def status(self):
        return "running"

    def cpu_percent(self, interval=0.1):
        return 10.0

    def memory_info(self):
        class _M:
            rss = 100 * 1024 * 1024
        return _M()

    def children(self, recursive=True):
        return list(self._children)

    def is_running(self):
        return True

    def cpu_times(self):
        self._counter['reads'] += 1
        if self._read_delay:
            time.sleep(self._read_delay)
        return _FakeCpuTimes(1.0, 0.5)

    def io_counters(self):
        return _FakeIO(1000, 2000)


def _install_fake_tree(monkeypatch, n_children, read_delay=0.0):
    counter = {'reads': 0}
    children = [
        _FakeProc(1000 + i, [], counter, read_delay) for i in range(n_children)
    ]
    root = _FakeProc(999, children, counter)

    def _fake_process(pid):
        return root

    monkeypatch.setattr(ph.psutil, "Process", _fake_process)
    return counter


class TestWalkIsBounded:
    def test_num_children_reports_true_total(self, monkeypatch):
        # The reported count must be the REAL tree size, not the walk cap —
        # OOM/child-drop detection depends on the true number.
        _install_fake_tree(monkeypatch, n_children=300)
        health = get_process_health(999)
        assert health['num_children'] == 300
        assert health['children_alive'] == 300

    def test_per_proc_reads_capped(self, monkeypatch):
        # With 300 children, an uncapped walk would issue 301 cpu_times() reads.
        # The cap limits it to root + MAX_CHILDREN_WALK.
        counter = _install_fake_tree(monkeypatch, n_children=300)
        get_process_health(999)
        assert counter['reads'] == 1 + ph.MAX_CHILDREN_WALK

    def test_small_tree_walks_all(self, monkeypatch):
        # A tree under the cap is walked completely (no behavior change).
        counter = _install_fake_tree(monkeypatch, n_children=5)
        get_process_health(999)
        assert counter['reads'] == 1 + 5

    def test_cpu_time_accumulated_from_sample(self, monkeypatch):
        # Liveness signal still populated from the (capped) sample.
        _install_fake_tree(monkeypatch, n_children=300)
        health = get_process_health(999)
        # root + cap procs, each contributing user+system = 1.5
        assert health['cpu_time'] == pytest.approx((1 + ph.MAX_CHILDREN_WALK) * 1.5)
        assert health['io_bytes'] == (1 + ph.MAX_CHILDREN_WALK) * 3000


class TestHeartbeatNotStarved:
    def test_concurrent_thread_progresses_during_slow_walk(self, monkeypatch):
        # Simulate the caffe -j256 scenario: a slow per-proc read (disk-bound)
        # over a large tree. The GIL-yield in the walk must let a concurrent
        # "heartbeat" thread tick while the walk runs. Without the yield, a
        # tight C-bound loop would starve the other thread.
        # 40 children x 2ms read = ~80ms walk; heartbeat ticks every ~1ms.
        _install_fake_tree(monkeypatch, n_children=40, read_delay=0.002)

        ticks = {'n': 0}
        stop = threading.Event()

        def _heartbeat():
            while not stop.is_set():
                ticks['n'] += 1
                time.sleep(0.001)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        try:
            get_process_health(999)
        finally:
            stop.set()
            hb.join(timeout=1)

        # The walk takes ~80ms; a heartbeat that got scheduled should have
        # ticked many times. We assert a conservative lower bound to prove the
        # main thread was NOT starved for the whole walk.
        assert ticks['n'] >= 5


class TestConstants:
    def test_cap_and_yield_constants_sane(self):
        assert ph.MAX_CHILDREN_WALK >= 1
        assert ph._GIL_YIELD_EVERY >= 1
        assert ph._GIL_YIELD_SECS > 0
