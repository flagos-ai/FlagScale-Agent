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

"""Tests for ShellTool command history tracking and container resource detection."""

import pytest

from flagscale_agent.react.tools.shell import ShellTool


class TestCommandHistory:
    def test_prefix_extraction(self):
        assert ShellTool._cmd_prefix("make -j4 2>&1 | tee build.log") == "make -j4"
        assert ShellTool._cmd_prefix("pip install torch") == "pip install"
        assert ShellTool._cmd_prefix("  ls  -la  ") == "ls -la"
        assert ShellTool._cmd_prefix("echo") == "echo"

    def test_record_and_build_history(self):
        tool = ShellTool()
        # No history yet
        assert tool._build_history_str("make -j4") == ""

        # Record some runs
        tool._record_history("make -j4", "completed", 120)
        tool._record_history("make -j4", "killed", 60)
        tool._record_history("make -j4", "completed", 90)

        history = tool._build_history_str("make -j4")
        assert "3 prior run" in history
        assert "completed in 120s" in history
        assert "killed in 60s" in history
        assert "completed in 90s" in history

    def test_history_grouped_by_prefix(self):
        tool = ShellTool()
        tool._record_history("pip install torch", "completed", 30)
        tool._record_history("make -j4", "killed", 60)

        # Different prefix → different history
        assert "pip install torch" not in tool._build_history_str("make -j4")
        assert "make -j4" not in tool._build_history_str("pip install torch")

    def test_history_capped_at_10(self):
        tool = ShellTool()
        for i in range(15):
            tool._record_history("make -j4", "completed", 100 + i)
        entries = tool._command_history["make -j4"]
        assert len(entries) == 10
        # Most recent 10 kept
        assert entries[0] == ("completed", 105.0)
        assert entries[-1] == ("completed", 114.0)

    def test_empty_history_for_unknown_command(self):
        tool = ShellTool()
        tool._record_history("make -j4", "completed", 120)
        # Different prefix → no history
        assert tool._build_history_str("cmake --build .") == ""


class TestContainerResources:
    def test_detect_resources_returns_string(self):
        tool = ShellTool()
        resources = tool._detect_resources()
        # On a Linux system, should detect CPU and maybe memory
        assert isinstance(resources, str)
        # CPU count should always be present on Linux
        if resources:
            assert "CPU" in resources or "cores" in resources

    def test_detect_resources_caches(self):
        tool = ShellTool()
        r1 = tool._detect_resources()
        r2 = tool._detect_resources()
        assert r1 == r2  # Same cached value
