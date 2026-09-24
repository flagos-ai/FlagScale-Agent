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

"""Tests for agent tools."""

import os
import tempfile

import pytest

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.react.tools.edit_file import EditFileTool
from flagscale_agent.react.tools.read_file import ReadFileTool
from flagscale_agent.react.tools.shell import (
    ShellTool, _strip_trailing_pipe,
)
from flagscale_agent.react.tools.write_file import WriteFileTool
from flagscale_agent.react.tools import ToolRegistry


class TestReadFileTool:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        tool = ReadFileTool()
        result = tool.execute(path=str(f))
        assert "hello world" in result
        assert "lines 1-1 of 1" in result

    def test_read_missing_file(self):
        tool = ReadFileTool()
        result = tool.execute(path="/nonexistent/path/file.txt")
        assert "ERROR" in result

    def test_schema_openai(self):
        tool = ReadFileTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"

    def test_schema_anthropic(self):
        tool = ReadFileTool()
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "read_file"
        assert "input_schema" in schema


class TestWriteFileTool:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "out.txt"
        tool = WriteFileTool()
        result = tool.execute(path=str(f), content="test content")
        assert "Wrote" in result or "Successfully" in result
        assert f.read_text() == "test content"

    def test_write_creates_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "out.txt"
        tool = WriteFileTool()
        tool.execute(path=str(f), content="nested")
        assert f.read_text() == "nested"


class TestEditFileTool:
    def test_edit_replaces(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo = 1\nbar = 2\n")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="foo = 1", new_string="foo = 42")
        assert "Successfully" in result
        assert "foo = 42" in f.read_text()

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="missing", new_string="x")
        assert result.startswith("ERROR:")

    def test_edit_missing_file(self):
        tool = EditFileTool()
        result = tool.execute(path="/nonexistent", old_string="a", new_string="b")
        assert result.startswith("ERROR:")


class TestShellTool:
    def test_basic_command(self):
        tool = ShellTool()
        result = tool.execute(command="echo hello")
        assert "hello" in result

    def test_health_judge_kills_command(self):
        tool = ShellTool(
            remind_interval=1,
            health_judge_fn=lambda cmd, out, t, **kw: {"kill": True, "reason": "test kill"},
        )
        result = tool.execute(command="bash -c 'echo running; sleep 10'")
        assert "TERMINATED" in result

    def test_kill_message_redirects_agent_away_from_variants(self):
        # Regression: after a health-monitor kill, the agent relaunched
        # near-identical variants. The TERMINATED message must tell the agent
        # this was a monitor decision and to change method class, not rerun.
        tool = ShellTool(
            remind_interval=1,
            health_judge_fn=lambda cmd, out, t, **kw: {"kill": True, "reason": "stalled loop"},
        )
        result = tool.execute(command="bash -c 'echo running; sleep 10'")
        assert "TERMINATED" in result
        low = result.lower()
        assert "health monitor" in low
        assert "do not relaunch" in low or "change your method" in low

    def test_empty_command_returns_error(self):
        tool = ShellTool()
        result = tool.execute(command="")
        assert "ERROR" in result

    def test_non_string_command_returns_error(self):
        tool = ShellTool()
        result = tool.execute(command=123)
        assert "ERROR" in result

    def test_self_kill_protection(self):
        tool = ShellTool()
        result = tool.execute(command="pkill -9 flagscale_agent")
        # Should rewrite command, not actually kill agent
        assert "flagscale" not in result or "xargs" in result or "(no output)" in result or "no process" in result.lower()


class TestHealthJudge:
    def test_health_judge_kills_on_first_interval(self):
        """health_judge_fn can kill even on the first check — no stall threshold needed."""
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0, **kw):
            return {"kill": True, "reason": "Repeated connection errors"}

        tool = ShellTool(
            remind_interval=1, 
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'Connection refused' && sleep 30")
        assert "TERMINATED" in result
        assert "Repeated connection errors" in result

    def test_health_judge_continue(self):
        """health_judge_fn says continue → command runs to completion."""
        judge_calls = []
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0, **kw):
            judge_calls.append(1)
            return {"kill": False, "reason": "Looks fine"}

        tool = ShellTool(
            remind_interval=1, 
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'working' && sleep 3 && echo 'done'")
        assert len(judge_calls) >= 1
        assert "TERMINATED" not in result
        assert "done" in result

    def test_health_judge_receives_stall_info(self):
        """health_judge_fn receives output_changed=False and stall_count when output stalls."""
        received = []
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0, **kw):
            received.append({"output_changed": output_changed, "stall_count": stall_count})
            if stall_count >= 2:
                return {"kill": True, "reason": "Stalled too long"}
            return {"kill": False}

        tool = ShellTool(
            remind_interval=1, 
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'stuck' && sleep 30")
        assert "TERMINATED" in result
        stall_counts = [r["stall_count"] for r in received]
        assert any(sc >= 2 for sc in stall_counts)
        assert any(not r["output_changed"] for r in received)

    def test_health_judge_overrides_legacy_stall(self):
        """When health_judge_fn is set, legacy stall detection is bypassed."""
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0, **kw):
            return {"kill": False, "reason": "Let it run"}

        tool = ShellTool(
            remind_interval=1, 
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'waiting' && sleep 4")
        assert "STALLED" not in result


class TestStripTrailingPipe:
    def test_no_pipe(self):
        cmd, fn = _strip_trailing_pipe("echo hello")
        assert cmd == "echo hello"
        assert fn is None

    def test_tail_n(self):
        cmd, fn = _strip_trailing_pipe("cat /var/log/syslog | tail -30")
        assert cmd == "cat /var/log/syslog"
        assert fn is not None
        lines = "\n".join(f"line{i}" for i in range(50)) + "\n"
        result = fn(lines)
        assert result.count("\n") == 30

    def test_head_n(self):
        cmd, fn = _strip_trailing_pipe("ls -la | head -5")
        assert cmd == "ls -la"
        lines = "\n".join(f"line{i}" for i in range(20)) + "\n"
        result = fn(lines)
        assert result.count("\n") == 5

    def test_tail_default(self):
        cmd, fn = _strip_trailing_pipe("dmesg | tail")
        assert cmd == "dmesg"
        lines = "\n".join(f"line{i}" for i in range(20)) + "\n"
        result = fn(lines)
        assert result.count("\n") == 10

    def test_tail_with_dash_n_space(self):
        cmd, fn = _strip_trailing_pipe("grep error log.txt | tail -n 20")
        assert cmd == "grep error log.txt"
        lines = "\n".join(f"line{i}" for i in range(50)) + "\n"
        result = fn(lines)
        assert result.count("\n") == 20

    def test_short_output_unchanged(self):
        _, fn = _strip_trailing_pipe("echo x | tail -30")
        result = fn("one\ntwo\nthree\n")
        assert result == "one\ntwo\nthree\n"

    def test_pipe_in_middle_not_stripped(self):
        cmd, fn = _strip_trailing_pipe("grep foo | sort | uniq")
        assert cmd == "grep foo | sort | uniq"
        assert fn is None

    def test_stderr_redirect_stripped_with_tail(self):
        cmd, fn = _strip_trailing_pipe('pip install -e ".[cuda-train]" 2>&1 | tail -20')
        assert cmd == 'pip install -e ".[cuda-train]"'
        assert fn is not None

    def test_stderr_redirect_stripped_with_head(self):
        cmd, fn = _strip_trailing_pipe("make -j4 2>&1 | head -10")
        assert cmd == "make -j4"
        assert fn is not None

    def test_integration_tail(self):
        tool = ShellTool()
        result = tool.execute(command="seq 100 | tail -5")
        lines = result.strip().splitlines()
        assert lines == ["96", "97", "98", "99", "100"]

    def test_integration_head(self):
        tool = ShellTool()
        result = tool.execute(command="seq 100 | head -3")
        lines = result.strip().splitlines()
        assert lines == ["1", "2", "3"]


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        tool = reg.get("read_file")
        assert tool.name == "read_file"

    def test_get_missing(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_execute_truncates(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100000)
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        result = reg.execute("read_file", path=str(f))
        assert len(result) < 100000
        assert "truncated" in result

    def test_to_schemas(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        reg.register(ShellTool())
        schemas = reg.to_schemas("openai")
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)


class TestRequiredArgPrevalidation:
    """Registry catches missing required args BEFORE dispatch (schema-driven)."""

    def test_missing_required_friendly_error_no_traceback(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        result = reg.execute("read_file")
        assert result.startswith("ERROR: required argument(s) missing")
        assert "path" in result
        assert "KeyError" not in result

    def test_missing_reports_all_names(self, tmp_path):
        reg = ToolRegistry()
        reg.register(EditFileTool())
        result = reg.execute("edit_file", path=str(tmp_path / "f.txt"))
        assert "old_string" in result and "new_string" in result

    def test_empty_value_counts_as_missing(self, tmp_path):
        reg = ToolRegistry()
        reg.register(WriteFileTool())
        result = reg.execute("write_file", path=str(tmp_path / "f.txt"), content="")
        assert "content" in result

    def test_memory_write_missing_key_blocked_before_dispatch(self):
        from flagscale_agent.react.tools.memory_write import MemoryWriteTool
        reg = ToolRegistry()
        reg.register(MemoryWriteTool(None))  # execute() never reached on this path
        result = reg.execute("memory_write", type="fact", content="x")
        assert "key" in result

    def test_plan_update_missing_action_blocked_before_dispatch(self):
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool
        reg = ToolRegistry()
        reg.register(PlanUpdateTool(None))  # execute() never reached on this path
        result = reg.execute("plan_update")
        assert "action" in result

    def test_valid_call_unaffected(self, tmp_path):
        f = tmp_path / "ok.txt"
        reg = ToolRegistry()
        reg.register(WriteFileTool())
        result = reg.execute("write_file", path=str(f), content="hello")
        assert "ERROR" not in result
        assert f.read_text() == "hello"


class TestEditFileReplaceAll:
    def test_replace_first_only(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\nx = 1\nx = 1\n")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="x = 1", new_string="x = 2", replace_all=False)
        assert "1 of 3" in result
        assert f.read_text().count("x = 2") == 1
        assert f.read_text().count("x = 1") == 2

    def test_replace_all(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\nx = 1\nx = 1\n")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="x = 1", new_string="x = 2", replace_all=True)
        assert "3 of 3" in result
        assert f.read_text().count("x = 2") == 3
        assert f.read_text().count("x = 1") == 0

    def test_replace_all_default_false(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("a\na\n")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="a", new_string="b")
        assert f.read_text().count("b") == 1


