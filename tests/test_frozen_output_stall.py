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

"""Tests for frozen-output stall detection.

Scenario: a command (e.g. python3 find_move.py) starts a child subprocess
(e.g. stockfish), then crashes with an error (e.g. KeyError).  The child
keeps running and burns CPU, but the parent process catches the error,
prints a traceback, and blocks forever waiting on the child.  The shell
monitor sees:
  - proc.poll() returns None (parent still alive)
  - output is identical for minutes (traceback printed once, no new output)
  - CPU is high (child subprocess working)
  - cpu_time_delta > 0.5 → making_progress = True

Previously, the liveness veto (making_progress=True) short-circuited at
line 218 and returned (False, "") — no advisory at all.  The LLM judge
only saw stall_count + output_changed + activity, with no hint that the
output was frozen.  After the fix, should_kill_process returns (False,
advisory_msg) even when making_progress is True, so the LLM judge gets
the frozen-output context and can decide to kill.
"""

import pytest

from flagscale_agent.react.tools.process_health import should_kill_process


class TestFrozenOutputAdvisory:
    """Frozen output with high CPU should return advisory, not be vetoed."""

    def test_frozen_output_with_cpu_returns_advisory(self):
        """Core bug fix: output frozen 3min + child burning CPU → advisory."""
        should_kill, reason = should_kill_process(
            elapsed_seconds=200,
            output_changed=False,
            stall_count=6,
            proc_health={
                'zombie': False,
                'cpu_percent': 85.0,
                'num_children': 1,
                'children_alive': 1,
                'memory_mb': 50,
            },
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={
                'cpu_time_delta': 5.0,  # stockfish burning CPU → making_progress=True
                'io_bytes_delta': 0,
                'rss_delta': 0,
            },
            mem_pressure=False,
        )
        assert should_kill is False
        assert reason != ""
        assert "frozen" in reason.lower()

    def test_frozen_output_vetoed_by_liveness_still_returns_advisory(self):
        """Even with strong liveness signals, frozen output gives advisory."""
        should_kill, reason = should_kill_process(
            elapsed_seconds=300,  # 5 minutes
            output_changed=False,
            stall_count=10,
            proc_health={
                'zombie': False,
                'cpu_percent': 99.0,
                'num_children': 3,
                'children_alive': 3,
                'memory_mb': 200,
            },
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=3,
            progress_signals={
                'cpu_time_delta': 30.0,  # very active
                'io_bytes_delta': 10 << 20,
                'rss_delta': 5 << 20,
            },
            mem_pressure=False,
        )
        assert should_kill is False
        assert reason != ""
        assert "frozen" in reason.lower()

    def test_changing_output_no_advisory(self):
        """When output is changing, no frozen-output advisory."""
        should_kill, reason = should_kill_process(
            elapsed_seconds=200,
            output_changed=True,
            stall_count=0,
            proc_health={
                'zombie': False,
                'cpu_percent': 50.0,
                'num_children': 2,
                'children_alive': 2,
                'memory_mb': 100,
            },
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
            progress_signals={
                'cpu_time_delta': 2.0,
                'io_bytes_delta': 0,
                'rss_delta': 0,
            },
            mem_pressure=False,
        )
        # No frozen advisory (output changed). May or may not have other
        # advisory — but should NOT contain "frozen".
        assert "frozen" not in (reason or "").lower()

    def test_frozen_output_short_duration_no_advisory(self):
        """Frozen for less than 2 minutes → no advisory yet."""
        should_kill, reason = should_kill_process(
            elapsed_seconds=90,  # < 120
            output_changed=False,
            stall_count=3,
            proc_health={
                'zombie': False,
                'cpu_percent': 50.0,
                'num_children': 1,
                'children_alive': 1,
                'memory_mb': 50,
            },
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={
                'cpu_time_delta': 1.0,
                'io_bytes_delta': 0,
                'rss_delta': 0,
            },
            mem_pressure=False,
        )
        assert "frozen" not in (reason or "").lower()

    def test_frozen_output_low_stall_count_no_advisory(self):
        """Only 2 stalls (1min) → not enough to trigger advisory."""
        should_kill, reason = should_kill_process(
            elapsed_seconds=150,
            output_changed=False,
            stall_count=2,  # < 6
            proc_health={
                'zombie': False,
                'cpu_percent': 50.0,
                'num_children': 1,
                'children_alive': 1,
                'memory_mb': 50,
            },
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={
                'cpu_time_delta': 1.0,
                'io_bytes_delta': 0,
                'rss_delta': 0,
            },
            mem_pressure=False,
        )
        assert "frozen" not in (reason or "").lower()
