"""Tests for the monitor tool."""

import os
import tempfile
import time
import threading

import pytest

from flagscale_agent.react.tools.monitor import MonitorTool


_LIVE_PATTERN = "pytest"  # always matches the test runner process


def _always_true_classify(category, text, context="", **kwargs):
    """Mock classify that always returns True (error detected)."""
    return True


class TestMonitorTool:

    def test_basic_file_watch_timeout(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        tool = MonitorTool()
        result = tool.execute(file=str(log_file), duration=3, interval=1,
                              process_pattern=_LIVE_PATTERN)

        assert "timeout" in result
        assert "polls" in result

    def test_file_not_found_waits(self, tmp_path):
        tool = MonitorTool()
        result = tool.execute(file=str(tmp_path / "nonexistent.log"), duration=3, interval=1,
                              process_pattern=_LIVE_PATTERN)
        assert "timeout" in result

    def test_target_step_reached(self, tmp_path):
        log_file = tmp_path / "log.txt"
        log_file.write_text("")

        def write_steps():
            time.sleep(1)
            with open(str(log_file), "a") as f:
                f.write("(step=0000005) loss: 0.5\n")
            time.sleep(1)
            with open(str(log_file), "a") as f:
                f.write("(step=0000010) loss: 0.3\n")

        t = threading.Thread(target=write_steps, daemon=True)
        t.start()

        tool = MonitorTool()
        result = tool.execute(file=str(log_file), target_step=10, duration=10, interval=1,
                              process_pattern=_LIVE_PATTERN)

        assert "target_reached" in result
        assert "step=0000010" in result
        t.join(timeout=3)

    def test_success_pattern(self, tmp_path):
        log_file = tmp_path / "log.txt"
        log_file.write_text("loading model...\n")

        def write_success():
            time.sleep(1)
            with open(str(log_file), "a") as f:
                f.write("training complete!\n")

        t = threading.Thread(target=write_success, daemon=True)
        t.start()

        tool = MonitorTool()
        result = tool.execute(
            file=str(log_file), success_pattern="training complete",
            duration=10, interval=1, process_pattern=_LIVE_PATTERN
        )

        assert "success" in result.lower() or "SUCCESS" in result
        t.join(timeout=3)

    def test_fail_pattern(self, tmp_path):
        log_file = tmp_path / "log.txt"
        log_file.write_text("starting...\n")

        def write_error():
            time.sleep(1)
            with open(str(log_file), "a") as f:
                f.write("CUDA error: out of memory\n")

        t = threading.Thread(target=write_error, daemon=True)
        t.start()

        tool = MonitorTool()
        result = tool.execute(
            file=str(log_file), fail_pattern="CUDA error",
            duration=10, interval=1, process_pattern=_LIVE_PATTERN
        )

        assert "error_detected" in result or "FAIL" in result
        t.join(timeout=3)

    def test_interesting_change_detected(self, tmp_path):
        log_file = tmp_path / "log.txt"
        log_file.write_text("normal output\n")

        def write_error():
            time.sleep(1)
            with open(str(log_file), "a") as f:
                f.write("RuntimeError: NCCL timeout\n")

        t = threading.Thread(target=write_error, daemon=True)
        t.start()

        tool = MonitorTool(classify_fn=_always_true_classify)
        result = tool.execute(file=str(log_file), duration=10, interval=1,
                              process_pattern=_LIVE_PATTERN)

        assert "interesting_change" in result
        assert "NCCL" in result
        t.join(timeout=3)

    def test_command_mode(self):
        tool = MonitorTool()
        result = tool.execute(command="echo hello", duration=3, interval=1,
                              process_pattern=_LIVE_PATTERN)
        # Should timeout since output doesn't change
        assert "timeout" in result

    def test_both_file_and_command_accepted(self):
        tool = MonitorTool()
        result = tool.execute(file="/tmp/nonexistent_xyz", command="echo y", duration=2, interval=1,
                              process_pattern=_LIVE_PATTERN)
        # No longer an error — tool accepts both; just check it runs
        assert "ERROR" not in result or "No such file" in result

    def test_error_neither_file_nor_command(self):
        tool = MonitorTool()
        result = tool.execute()
        assert "ERROR" in result

    def test_metrics_recorded_in_events(self, tmp_path):
        log_file = tmp_path / "log.txt"
        log_file.write_text("init\n")

        def write_metrics():
            time.sleep(0.5)
            with open(str(log_file), "a") as f:
                f.write("step=5 loss=0.5 MFU=30%\n")

        t = threading.Thread(target=write_metrics, daemon=True)
        t.start()

        tool = MonitorTool()
        result = tool.execute(file=str(log_file), duration=8, interval=1,
                              process_pattern=_LIVE_PATTERN)

        # Metrics are recorded but don't break the loop (timeout expected)
        assert "timeout" in result
        assert "metric" in result.lower()
        t.join(timeout=3)

    def test_schema(self):
        tool = MonitorTool()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "monitor"
        assert "file" in schema["function"]["parameters"]["properties"]
        assert "target_step" in schema["function"]["parameters"]["properties"]

    def test_anthropic_schema(self):
        tool = MonitorTool()
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "monitor"
        assert "file" in schema["input_schema"]["properties"]
