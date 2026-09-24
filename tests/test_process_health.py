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

"""Tests for process health detection."""

import pytest

from flagscale_agent.react.tools.process_health import (
    detect_output_anomalies,
    should_kill_process,
    system_mem_pressure,
    _cgroup_mem_pressure,
)


class TestDetectOutputAnomalies:
    def test_oom_killed_detected(self):
        output = "make: *** [target] Error 137\nKilled\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is True
        assert result['killed_count'] == 1

    def test_multiple_killed(self):
        output = "Killed\ncompiling...\nKilled\nKilled\n"
        result = detect_output_anomalies(output)
        assert result['killed_count'] == 3

    def test_repeated_errors(self):
        output = "Error: something\nError: something\nError: something\n"
        result = detect_output_anomalies(output)
        assert result['repeated_errors'] is True

    def test_clean_output(self):
        output = "Building project...\ncompiling file.c\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is False
        assert result['killed_count'] == 0
        assert result['repeated_errors'] is False

    def test_cc1plus_oom_signature_detected(self):
        # REGRESSION (pitfall/flagscale_agent/shell_oom_cgroup_blindspot):
        # the kernel OOM-killer reaps cc1plus and g++ reports it as the line
        # below. The old oom_indicators list missed this, so an OOM build
        # looked healthy. This is the EXACT caffe-cifar-10 build.log signature.
        output = (
            "[ 45%] Building CXX object caffe.dir/layer_factory.cpp.o\n"
            "g++: fatal error: Killed signal terminated program cc1plus\n"
            "compilation terminated.\n"
            "g++: fatal error: Killed signal terminated program cc1plus\n"
            "compilation terminated.\n"
            "make: *** Waiting for unfinished jobs....\n"
        )
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is True
        assert result['killed_count'] >= 2

    def test_cc1_variant_signature_detected(self):
        output = "cc1: fatal error: Killed signal terminated program cc1\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is True

    def test_ice_killed_signature_detected(self):
        output = "internal compiler error: Killed\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is True

    def test_normal_build_no_false_oom(self):
        # A healthy build that merely contains the word "signal" or "program"
        # must NOT trip the OOM detector.
        output = (
            "Configuring signal handlers\n"
            "[ 50%] Building program main.o\n"
            "[100%] Linking CXX executable caffe\n"
        )
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is False


class TestCgroupMemPressure:
    """
    pitfall/flagscale_agent/shell_oom_cgroup_blindspot: in a container psutil
    reports the HOST RAM, so system_mem_pressure never fires while a build
    OOM-thrashes inside a small cgroup slice. The probe must prefer cgroup
    accounting and fall back to host only when there is no enforced limit.
    """

    def test_cgroup_v2_pressure_true(self, tmp_path, monkeypatch):
        (tmp_path / "memory.max").write_text("2147483648\n")       # 2GB limit
        (tmp_path / "memory.current").write_text("2040109465\n")   # ~95% used
        self._point_v2(monkeypatch, tmp_path)
        assert _cgroup_mem_pressure(0.10) is True

    def test_cgroup_v2_pressure_false(self, tmp_path, monkeypatch):
        (tmp_path / "memory.max").write_text("2147483648\n")
        (tmp_path / "memory.current").write_text("536870912\n")    # 25% used
        self._point_v2(monkeypatch, tmp_path)
        assert _cgroup_mem_pressure(0.10) is False

    def test_cgroup_v2_unlimited_returns_none(self, tmp_path, monkeypatch):
        (tmp_path / "memory.max").write_text("max\n")
        self._point_v2(monkeypatch, tmp_path)
        # no v1 files either → None (defer to host)
        assert _cgroup_mem_pressure(0.10) is None

    @staticmethod
    def _point_v2(monkeypatch, tmp_path):
        real_open = open

        def fake_open(path, *a, **k):
            p = str(path)
            if p == "/sys/fs/cgroup/memory.max":
                return real_open(tmp_path / "memory.max", *a, **k)
            if p == "/sys/fs/cgroup/memory.current":
                return real_open(tmp_path / "memory.current", *a, **k)
            # everything else (v1 paths) must miss
            if p.startswith("/sys/fs/cgroup/memory/"):
                raise FileNotFoundError(p)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)


class TestShouldKillProcess:
    def test_zombie_process(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': True, 'cpu_percent': 0, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
        )
        assert should_kill is True
        assert 'zombie' in reason.lower()

    def test_oom_killed_twice(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 2, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is True
        assert 'OOM' in reason

    def test_child_drop_alone_does_not_kill(self):
        # REDESIGN (support B): a child-count drop with NO OOM evidence
        # (killed_count=0, no memory pressure) is normal worker convergence,
        # NOT OOM. This is the caffe-cifar-10 false-kill case: 5→2 workers
        # exiting normally must not be convicted.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 30, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=5,  # 5 -> 2
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,   # no real memory pressure
        )
        assert should_kill is False

    def test_child_drop_with_oom_evidence_kills(self):
        # support B: drop >= half AND no live children AND real OOM evidence
        # (killed in output) → convict.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': True, 'killed_count': 1, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,  # 8 -> 2, dropped more than half
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert 'OOM likely' in reason

    def test_child_drop_with_mem_pressure_kills(self):
        # support B: same drop, evidence via system memory pressure instead of output
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=True,
        )
        assert should_kill is True
        assert 'OOM likely' in reason

    def test_liveness_veto_blocks_child_drop_kill(self):
        # support A: even with drop + no live children + memory pressure, if the
        # remaining process is burning CPU time, the liveness veto wins.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,
            progress_signals={'cpu_time_delta': 2.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=True,
        )
        assert should_kill is False

    def test_liveness_veto_does_not_block_oom_hard_evidence(self):
        # OOM hard evidence (Killed x2) fires even if cpu_time is rising.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 40, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 2, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
            progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert 'OOM' in reason

    def test_all_children_dead(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=150,
            output_changed=False,
            stall_count=5,
            proc_health={'zombie': False, 'cpu_percent': 0.1, 'num_children': 4, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is True
        assert 'child processes exited' in reason.lower()

    def test_cpu_zero_long_stall_is_advisory_not_hard_kill(self):
        # REDESIGN: rule #4 (silent stall >3min with 0% CPU) is now ADVISORY only.
        # The old rule killed healthy imports (torch, large libraries) that legitimately
        # run silent for minutes. The LLM judge with richer context makes this call.
        should_kill, reason = should_kill_process(
            elapsed_seconds=200,
            output_changed=False,
            stall_count=7,  # 7 * 30s = 210s > 3min
            proc_health={'zombie': False, 'cpu_percent': 0.0, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
        )
        assert should_kill is False  # advisory, not hard kill
        assert "0% CPU" in reason  # still returns a reason for LLM judge context

    def test_soft_timeout_20min_does_not_hard_kill(self):
        # support C: 20min soft cap no longer hard-kills — deferred to LLM judge.
        # A heavy compile making progress must survive past 20min.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1300,  # >20min <60min
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
            progress_signals={'cpu_time_delta': 3.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is False

    def test_soft_timeout_no_progress_still_defers(self):
        # 20min<elapsed<60min with no progress: still hand to judge, not hard-kill.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1500,
            output_changed=True,  # output moving, but no cpu/io progress
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.2, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is False

    def test_hard_timeout_60min_kills(self):
        # support C: hard cap 60min with no *progress* → kill. Output is moving
        # and children alive (so earlier heuristics #3/#4 skip), but cpu/io/rss
        # deltas are zero — output churns without real work. Only the 60min cap
        # catches this.
        should_kill, reason = should_kill_process(
            elapsed_seconds=3700,  # >60min
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.2, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert '60-minute' in reason

    def test_busy_loop_killed_after_6min_frozen(self):
        # A busy-loop burning CPU with NO output change for 6+ minutes is now
        # hard-killed. Previously this survived until 60min hard cap (liveness
        # veto), but that caused real hangs (e.g. stockfish subprocess holding
        # stdout while parent was dead). The frozen-output hard kill fires
        # regardless of liveness veto.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1800,  # 30min
            output_changed=False,
            stall_count=20,  # 20 * 30s = 10min frozen
            proc_health={'zombie': False, 'cpu_percent': 100, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
            progress_signals={'cpu_time_delta': 15.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True  # frozen output hard kill fires at stall_count >= 12

    def test_healthy_process(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
        )
        assert should_kill is False

    def test_empty_output_triggers_stall_advisory(self):
        # Regression test: Commands with no streaming output (e.g., `apt-get ... | tail -5`,
        # silent network waits) should accumulate stall_count and trigger the
        # frozen-output advisory at 3min.  This is ADVISORY (returns False) not
        # a hard kill.  The LLM judge gets the reason as context to make the
        # final call.
        should_kill, reason = should_kill_process(
            elapsed_seconds=190,  # >180s
            output_changed=False,  # empty output = not changed
            stall_count=7,  # >=6 (7*30s = 3m30s of stalling)
            proc_health={'zombie': False, 'cpu_percent': 0.0, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
        )
        assert should_kill is False  # advisory, not hard kill
        assert "frozen" in reason.lower()
        assert "210s" in reason  # stall_count * 30 = 7*30 = 210s

    def test_oom_killed_once_not_enough(self):
        # Need 2+ kills to trigger
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 1, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is False

    def test_children_dead_but_recent_output(self):
        # Output still changing, don't kill yet
        should_kill, reason = should_kill_process(
            elapsed_seconds=150,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.1, 'num_children': 4, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is False


_CLEAN = {'oom_killed': False, 'killed_count': 0, 'repeated_errors': False}
_HEALTH = {'zombie': False, 'cpu_percent': 0.0, 'num_children': 0, 'children_alive': 0}


class TestIdleKill:
    """Rule 1c: no output + flat CPU/IO/RSS for IDLE_KILL_THRESHOLD samples."""

    def test_idle_threshold_reached_kills(self):
        # 4 consecutive idle samples (~2 min): no output, all counters flat.
        should_kill, reason = should_kill_process(
            elapsed_seconds=130,
            output_changed=False,
            stall_count=4,
            proc_health=_HEALTH,
            output_anomalies=_CLEAN,
            had_children=False,
            prev_num_children=0,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            idle_count=4,
        )
        assert should_kill is True
        assert 'no output' in reason.lower()
        assert 'syscall' in reason.lower() or 'network' in reason.lower()

    def test_below_idle_threshold_does_not_kill(self):
        # Only 3 idle samples — under the 4-sample threshold.
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=3,
            proc_health=_HEALTH,
            output_anomalies=_CLEAN,
            had_children=False,
            prev_num_children=0,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            idle_count=3,
        )
        assert should_kill is False

    def test_cpu_progress_prevents_idle_kill(self):
        # A compile burns CPU: shell.py would never accumulate idle_count here,
        # but even if idle_count is stale, the liveness veto + zero idle_count
        # from the caller keeps it alive. Here idle_count=0 (caller reset it
        # because cpu_time_delta > 0.5) and progress vetoes heuristics.
        should_kill, reason = should_kill_process(
            elapsed_seconds=400,
            output_changed=False,
            stall_count=8,
            proc_health={'zombie': False, 'cpu_percent': 95, 'num_children': 20, 'children_alive': 20},
            output_anomalies=_CLEAN,
            had_children=True,
            prev_num_children=20,
            progress_signals={'cpu_time_delta': 30.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            idle_count=0,
        )
        assert should_kill is False

    def test_io_progress_prevents_idle_kill(self):
        # A download writing bytes: io_bytes_delta > 1MB means caller keeps
        # idle_count at 0 even with no stdout.
        should_kill, reason = should_kill_process(
            elapsed_seconds=200,
            output_changed=False,
            stall_count=6,
            proc_health={'zombie': False, 'cpu_percent': 2, 'num_children': 1, 'children_alive': 1},
            output_anomalies=_CLEAN,
            had_children=True,
            prev_num_children=1,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 5 << 20, 'rss_delta': 0},
            idle_count=0,
        )
        assert should_kill is False

    def test_idle_kill_fires_even_with_stale_progress_signal(self):
        # idle_count is the authority: if the caller says 4 idle samples, kill
        # regardless of a single-sample progress_signals blip (defense in depth
        # — caller only bumps idle_count when signals were flat anyway).
        should_kill, reason = should_kill_process(
            elapsed_seconds=140,
            output_changed=False,
            stall_count=4,
            proc_health=_HEALTH,
            output_anomalies=_CLEAN,
            had_children=False,
            prev_num_children=0,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            idle_count=5,
        )
        assert should_kill is True

    def test_default_idle_count_zero_is_backward_compatible(self):
        # Legacy callers omit idle_count → defaults to 0 → never triggers 1c.
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 4, 'children_alive': 4},
            output_anomalies=_CLEAN,
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is False
