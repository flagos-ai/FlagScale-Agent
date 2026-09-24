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

"""Tests for FlagScaleTrainMonitorTool (unified monitor)."""

import glob
import json
import os
import tempfile
import threading
import time

import pytest

from flagscale_agent.react.tools.monitor import (
    FlagScaleTrainMonitorTool,
    _parse_megatron_metrics,
    _health_check,
    _is_harmless,
    _last_sorted_subdir,
    _numeric_key,
    _tail,
)


# ─── Helper: create FlagScale log structure ───────────────────────────────────

def _make_flagscale_logs(base, num_hosts=1, num_ranks_per_host=8,
                         stdout_content=None, stderr_content=None):
    """Create a realistic FlagScale log directory structure.
    
    Returns output_dir path.
    """
    output_dir = os.path.join(base, "experiment_output")
    for host_idx in range(num_hosts):
        host_name = f"host_{host_idx}_gpu{host_idx:02d}"
        attempt_dir = os.path.join(
            output_dir, "logs", "details", host_name,
            "2024-08-04_12-00-00", "run_0", "attempt_0"
        )
        for rank in range(host_idx * num_ranks_per_host,
                         (host_idx + 1) * num_ranks_per_host):
            rank_dir = os.path.join(attempt_dir, str(rank))
            os.makedirs(rank_dir, exist_ok=True)
            # stdout
            content = stdout_content if stdout_content else ""
            if rank == num_hosts * num_ranks_per_host - 1 and stdout_content is None:
                # Last rank gets default metrics
                content = "iteration 10 | lm loss: 8.5432 | grad norm: 1.234\n"
            with open(os.path.join(rank_dir, "stdout.log"), "w") as f:
                f.write(content)
            # stderr
            err = stderr_content if stderr_content else ""
            with open(os.path.join(rank_dir, "stderr.log"), "w") as f:
                f.write(err)
    return output_dir


# ─── Unit tests: metrics parsing ──────────────────────────────────────────────

class TestMetricsParsing:
    def test_parse_basic_iteration(self):
        text = " iteration      42/ 1000 | lm loss: 7.8234E+00 | grad norm: 1.456"
        m = _parse_megatron_metrics(text)
        assert m["last_iter"] == 42
        assert abs(m["last_loss"]["lm_loss"] - 7.8234) < 0.001
        assert abs(m["last_loss"]["grad_norm"] - 1.456) < 0.001

    def test_parse_multiple_iterations(self):
        text = (
            "iteration 1 | loss: 10.0\n"
            "iteration 2 | loss: 9.5\n"
            "iteration 3 | loss: 9.0\n"
        )
        m = _parse_megatron_metrics(text)
        assert m["last_iter"] == 3
        assert len(m["iterations"]) == 3

    def test_parse_no_metrics(self):
        m = _parse_megatron_metrics("Loading model weights...\nDone.")
        assert m["last_iter"] is None
        assert m["iterations"] == []

    def test_health_check_random_output(self):
        metrics = {"last_iter": 5, "last_loss": {"lm_loss": 11.9}, "iterations": [5], "anomalies": []}
        # ln(151936) ≈ 11.93
        warnings = _health_check(metrics, vocab_size=151936)
        assert any("random" in w for w in warnings)

    def test_health_check_ok(self):
        metrics = {"last_iter": 100, "last_loss": {"lm_loss": 5.0}, "iterations": [100], "anomalies": []}
        warnings = _health_check(metrics, vocab_size=151936)
        assert warnings == []

    def test_health_check_zero_grad(self):
        metrics = {"last_iter": 5, "last_loss": {"grad_norm": 0}, "iterations": [5], "anomalies": []}
        warnings = _health_check(metrics)
        assert any("grad_norm=0" in w for w in warnings)


# ─── Unit tests: utilities ────────────────────────────────────────────────────

class TestUtilities:
    def test_last_sorted_subdir(self, tmp_path):
        os.makedirs(tmp_path / "a")
        os.makedirs(tmp_path / "b")
        os.makedirs(tmp_path / "c")
        assert _last_sorted_subdir(str(tmp_path)).endswith("c")

    def test_last_sorted_subdir_numeric(self, tmp_path):
        os.makedirs(tmp_path / "attempt_0")
        os.makedirs(tmp_path / "attempt_1")
        os.makedirs(tmp_path / "attempt_10")
        result = _last_sorted_subdir(str(tmp_path), key=_numeric_key)
        assert result.endswith("attempt_10")

    def test_last_sorted_subdir_empty(self, tmp_path):
        assert _last_sorted_subdir(str(tmp_path)) == ""

    def test_last_sorted_subdir_nonexist(self):
        assert _last_sorted_subdir("/nonexistent/path") == ""

    def test_is_harmless_deprecation(self):
        assert _is_harmless("DeprecationWarning: something old")

    def test_is_harmless_user_warning(self):
        assert _is_harmless("UserWarning: torch.cuda.amp is deprecated")

    def test_not_harmless_error(self):
        assert not _is_harmless("RuntimeError: CUDA out of memory")

    def test_not_harmless_traceback(self):
        assert not _is_harmless("Traceback (most recent call last):")


# ─── Integration tests: check mode ───────────────────────────────────────────

class TestCheckMode:
    def test_check_basic(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path))
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "FlagScale Training Status" in result
        assert "iteration" in result.lower() or "Loss Rank" in result

    def test_check_finds_loss_rank(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path), num_hosts=1, num_ranks_per_host=4)
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "rank 3" in result  # last rank has metrics

    def test_check_multi_host_shared_storage(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path), num_hosts=4, num_ranks_per_host=8)
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "shared" in result
        assert "4 host" in result
        assert "0-31" in result

    def test_check_with_errors(self, tmp_path):
        output_dir = _make_flagscale_logs(
            str(tmp_path),
            stderr_content="RuntimeError: CUDA out of memory\nTraceback..."
        )
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "Error" in result or "error" in result

    def test_check_no_logs(self, tmp_path):
        output_dir = str(tmp_path / "empty_experiment")
        os.makedirs(output_dir)
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "ERROR" in result

    def test_check_filter_progress(self, tmp_path):
        content = (
            "Loading model...\n"
            "iteration 1 | loss: 10.0\n"
            "Some debug info\n"
            "iteration 2 | loss: 9.5\n"
        )
        output_dir = _make_flagscale_logs(str(tmp_path), stdout_content=content)
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check", filter="progress")
        assert "iteration" in result
        # "Loading model" should be filtered out in progress mode from the filtered section
        # But it may appear in other parts of the output

    def test_check_json_summary(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path))
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="check")
        assert "JSON" in result
        # Extract JSON part
        json_start = result.index("{")
        json_end = result.rindex("}") + 1
        summary = json.loads(result[json_start:json_end])
        assert "loss_rank" in summary
        assert "training_started" in summary

    def test_check_nonexistent_dir(self):
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir="/nonexistent/path", mode="check")
        assert "ERROR" in result


# ─── Integration tests: watch mode ───────────────────────────────────────────

class TestWatchMode:
    def test_watch_immediate_error(self, tmp_path):
        """Watch should return immediately if stderr already has errors."""
        output_dir = _make_flagscale_logs(
            str(tmp_path),
            stderr_content="RuntimeError: CUDA error: device-side assert\n"
        )
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="watch", duration=10, interval=5)
        assert "CRASHED" in result or "ERROR" in result

    def test_watch_timeout_no_activity(self, tmp_path):
        """Watch with short timeout should return timeout."""
        output_dir = _make_flagscale_logs(str(tmp_path), stdout_content="init done\n")
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir=output_dir, mode="watch", duration=6, interval=5)
        # Should timeout or detect dead (no GPU process in test env)
        assert "timeout" in result.lower() or "DEAD" in result

    def test_watch_nonexistent_dir(self):
        tool = FlagScaleTrainMonitorTool()
        result = tool.execute(output_dir="/nonexistent", mode="watch", duration=5, interval=5)
        assert "ERROR" in result


# ─── Test liveness detection ──────────────────────────────────────────────────

class TestLiveness:
    def test_alive_recent_log_growth(self):
        tool = FlagScaleTrainMonitorTool()
        # Log grew 10s ago — should be alive
        assert tool._is_alive("training", time.time() - 10) is True

    def test_is_network_storage_real_local_tmp(self):
        """Un-mocked probe: /tmp (tmpfs/overlay on any Linux) must classify
        as LOCAL storage. Canary against whitelist inversion — the gate may
        only declare dead on proven-local paths."""
        tool = FlagScaleTrainMonitorTool()
        assert tool._is_network_storage("/tmp") is False

    def test_dead_no_signals(self, monkeypatch):
        """Local storage + both local probes negative → proven dead."""
        tool = FlagScaleTrainMonitorTool()
        monkeypatch.setattr(tool, "_gpu_has_compute_process", lambda: False)
        monkeypatch.setattr(tool, "_pgrep_alive", lambda: False)
        monkeypatch.setattr(tool, "_is_network_storage", lambda path=".": False)
        assert tool._is_alive("training", time.time() - 300) is False

    def test_uncertain_remote_shared_storage(self, monkeypatch):
        """Network storage + probes negative → 'uncertain', never False."""
        tool = FlagScaleTrainMonitorTool()
        monkeypatch.setattr(tool, "_gpu_has_compute_process", lambda: False)
        monkeypatch.setattr(tool, "_pgrep_alive", lambda: False)
        monkeypatch.setattr(tool, "_is_network_storage", lambda path=".": True)
        result = tool._is_alive("training", time.time() - 300)
        assert result == "uncertain"

    def test_watch_stderr_growth_keeps_alive(self, tmp_path, monkeypatch):
        """stderr bytes growing while stdout is locked on a quiet file
        must count as a growth signal (keeps watch alive, no false DEAD)."""
        output_dir = _make_flagscale_logs(
            str(tmp_path),
            stdout_content="init done, no metrics here\n",
        )
        # Fixture nests under experiment_output/ — find the stderr there
        stderr_files = sorted(
            glob.glob(os.path.join(
                output_dir, "logs", "details", "host_*", "*", "*", "*", "*", "stderr.log"
            ))
        )
        assert stderr_files, "fixture must create stderr.log"
        stderr_file = stderr_files[0]
        tool = FlagScaleTrainMonitorTool()
        # Disable GPU/pgrep probes: only the stderr-growth signal can prove life
        monkeypatch.setattr(tool, "_gpu_has_compute_process", lambda: False)
        monkeypatch.setattr(tool, "_pgrep_alive", lambda: False)
        monkeypatch.setattr(tool, "_is_network_storage", lambda path=".": True)

        def grow_stderr():
            for _ in range(6):
                time.sleep(1.5)
                with open(stderr_file, "a") as f:
                    f.write("progress heartbeat line\n")

        t = threading.Thread(target=grow_stderr, daemon=True)
        t.start()
        result = tool.execute(output_dir=output_dir, mode="watch", duration=8, interval=3)
        assert "DEAD" not in result
        assert "Timeout" in result or "running" in result

    def test_watch_shared_storage_stale_yields_uncertain_not_dead(self, tmp_path, monkeypatch):
        """End-to-end: logs fully stale + no probes on shared storage → watch
        must NOT declare DEAD; it reports UNCERTAIN and exits via timeout."""
        output_dir = _make_flagscale_logs(
            str(tmp_path),
            stdout_content="iteration 10 | lm loss: 8.5432 | grad norm: 1.234\n",
        )
        tool = FlagScaleTrainMonitorTool()
        monkeypatch.setattr(tool, "_gpu_has_compute_process", lambda: False)
        monkeypatch.setattr(tool, "_pgrep_alive", lambda: False)
        monkeypatch.setattr(tool, "_is_network_storage", lambda path=".": True)
        tool.LIVENESS_GRACE_SECONDS = {"training": 1, "startup": 1, "rendezvous": 1, "init": 1}
        result = tool.execute(output_dir=output_dir, mode="watch", duration=10, interval=2)
        assert "TRAINING DEAD" not in result
        assert "UNCERTAIN" in result
        assert "Timeout" in result

    def test_watch_local_storage_stale_yields_dead(self, tmp_path, monkeypatch):
        """End-to-end: same stale logs on LOCAL storage with no probes →
        watch correctly declares DEAD (gate only downgrades, never hides)."""
        output_dir = _make_flagscale_logs(
            str(tmp_path),
            stdout_content="iteration 10 | lm loss: 8.5432 | grad norm: 1.234\n",
        )
        tool = FlagScaleTrainMonitorTool()
        monkeypatch.setattr(tool, "_gpu_has_compute_process", lambda: False)
        monkeypatch.setattr(tool, "_pgrep_alive", lambda: False)
        monkeypatch.setattr(tool, "_is_network_storage", lambda path=".": False)
        tool.LIVENESS_GRACE_SECONDS = {"training": 1, "startup": 1, "rendezvous": 1, "init": 1}
        result = tool.execute(output_dir=output_dir, mode="watch", duration=10, interval=2)
        assert "TRAINING DEAD" in result

    def test_grace_period_rendezvous(self):
        tool = FlagScaleTrainMonitorTool()
        # In rendezvous phase, 90s without log growth is still OK (grace=120s)
        assert tool._is_alive("rendezvous", time.time() - 90) is True

    def test_grace_period_init(self):
        tool = FlagScaleTrainMonitorTool()
        # In init phase, 200s without log growth is still OK (grace=300s)
        assert tool._is_alive("init", time.time() - 200) is True


# ─── Test log discovery ───────────────────────────────────────────────────────

class TestLogDiscovery:
    def test_discover_single_host(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path), num_hosts=1, num_ranks_per_host=8)
        tool = FlagScaleTrainMonitorTool()
        logs = tool._discover_logs(output_dir)
        assert not logs.get("error")
        assert len(logs["rank_dirs"]) == 8
        assert len(logs["stderr_logs"]) == 8
        assert logs["stdout_log"]  # should find the one with metrics

    def test_discover_multi_host(self, tmp_path):
        output_dir = _make_flagscale_logs(str(tmp_path), num_hosts=4, num_ranks_per_host=8)
        tool = FlagScaleTrainMonitorTool()
        logs = tool._discover_logs(output_dir)
        assert not logs.get("error")
        assert len(logs["rank_dirs"]) == 32
        assert len(logs["host_dirs"]) == 4

    def test_discover_no_logs_dir(self, tmp_path):
        output_dir = str(tmp_path / "empty")
        os.makedirs(output_dir)
        tool = FlagScaleTrainMonitorTool()
        logs = tool._discover_logs(output_dir)
        assert "ERROR" in logs.get("error", "")



# ─── Tests for _tail and _read_tail unification ─────────────────────────────

class TestTailFunctions:
    def test_tail_basic(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = _tail(str(f), n=3)
        assert "line3" in result
        assert "line4" in result
        assert "line5" in result
        assert "line1" not in result

    def test_tail_missing_file(self):
        result = _tail("/nonexistent/path/log.txt", n=10)
        assert result == "(empty)"

    def test_tail_empty_file(self, tmp_path):
        f = tmp_path / "empty.log"
        f.write_text("")
        result = _tail(str(f), n=10)
        assert result == "(empty)"

    def test_read_tail_delegates_to_tail(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("aaa\nbbb\nccc\n")
        tool = FlagScaleTrainMonitorTool()
        lines = tool._read_tail(str(f), n=2)
        assert isinstance(lines, list)
        assert "bbb" in lines
        assert "ccc" in lines

    def test_read_tail_missing_file(self):
        tool = FlagScaleTrainMonitorTool()
        lines = tool._read_tail("/nonexistent/path", n=5)
        assert lines == ["(file not found)"]


# ─── Tests for _parse_megatron_metrics edge cases ────────────────────────────

class TestMetricsParsing:
    def test_parse_scientific_notation(self):
        text = " iteration     100/ 5000 | lm loss: 2.3456E+01 | grad norm: 1.5E-02"
        m = _parse_megatron_metrics(text)
        assert m["last_iter"] == 100
        assert abs(m["last_loss"]["lm_loss"] - 23.456) < 0.01
        assert abs(m["last_loss"]["grad_norm"] - 0.015) < 0.001

    def test_parse_ce_loss(self):
        text = " iteration     50/ 1000 | ce_loss: 8.1234 | grad norm: 0.5"
        m = _parse_megatron_metrics(text)
        assert abs(m["last_loss"]["ce_loss"] - 8.1234) < 0.001

    def test_parse_multiple_loss_types(self):
        # "loss:" regex also matches "lm loss:" — lm_loss captures 5.0, loss also gets 5.0
        text = " iteration     10/ 100 | lm loss: 5.0 | grad norm: 1.0"
        m = _parse_megatron_metrics(text)
        assert m["last_loss"]["lm_loss"] == 5.0
        assert m["last_loss"]["loss"] == 5.0  # "loss" regex matches "lm loss" too
        assert m["last_loss"]["grad_norm"] == 1.0

    def test_health_check_auto_detect_vocab(self):
        """When vocab_size=0 and loss ~ ln(vocab), warning should mention it."""
        import math
        # loss = ln(151936) ≈ 11.93, set loss exactly to trigger
        metrics = {"last_loss": {"lm_loss": 11.93}, "iterations": [1], "last_iter": 1, "anomalies": []}
        warnings = _health_check(metrics, vocab_size=0)
        assert any("151936" in w for w in warnings)
