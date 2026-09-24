"""
Regression test for the dead-code bug where the frozen-output advisory
(line 206) shadowed the hard kill (line 241) because both had the same
condition (stall_count >= 6, elapsed > 120).

After the fix:
- stall_count >= 6, elapsed > 120  → advisory (False, msg)
- stall_count >= 12, elapsed > 300 → hard kill (True, msg)
"""
from flagscale_agent.react.tools.process_health import should_kill_process


def _health(cpu=50.0, children=1, children_alive=1, zombie=False, mem=100.0):
    return {
        'alive': True, 'zombie': zombie,
        'cpu_percent': cpu, 'memory_mb': mem,
        'num_children': children, 'children_alive': children_alive,
        'cpu_time': 10.0, 'io_bytes': 0,
    }


def _anomalies():
    return {
        'oom_killed': False, 'killed_count': 0,
        'segfault': False, 'memory_error': False,
    }


def test_advisory_at_3min_not_killed():
    """At stall_count=6 (~3min), should return advisory (False), not kill."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=200, output_changed=False, stall_count=6,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, f"Should be advisory, not kill. Got should_kill={should_kill}"
    assert 'frozen' in reason.lower(), f"Reason should mention frozen: {reason}"


def test_hard_kill_at_6min():
    """At stall_count=12 (~6min), should return hard kill (True)."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is True, f"Should hard-kill after 6min frozen. Got should_kill={should_kill}"
    assert 'frozen' in reason.lower(), f"Reason should mention frozen: {reason}"


def test_hard_kill_fires_even_with_high_cpu():
    """The hard kill must fire even when CPU is high (stockfish burning CPU)."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=99.0), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 50.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is True, "Hard kill must fire even with high CPU when output frozen 6min"


def test_no_kill_when_output_changing():
    """When output IS changing, neither advisory nor kill should fire from frozen rules."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=True, stall_count=0,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, "Should not kill when output is changing"


def test_advisory_fires_before_liveness_veto():
    """Advisory must fire even when making_progress (liveness veto) is True."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=200, output_changed=False, stall_count=6,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 10.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, "Advisory should not hard-kill"
    assert reason, "Advisory should have a reason message"


def test_hard_kill_at_6min_with_no_progress():
    """Hard kill fires even without progress signals (None)."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=0.1), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals=None,
    )
    assert should_kill is True, "Should hard-kill after 6min frozen even without progress signals"
