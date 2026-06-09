"""Tests for agent tools."""

import os
import tempfile

import pytest

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.react.tools.edit_file import EditFileTool
from flagscale_agent.react.tools.read_file import ReadFileTool
from flagscale_agent.react.tools.shell import (
    ShellTool, _strip_trailing_pipe,
    _inject_proxy_exports, _ensure_wget_continue,
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
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command="echo hello")
        assert "hello" in result

    def test_health_judge_kills_command(self):
        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            health_judge_fn=lambda cmd, out, t, **kw: {"kill": True, "reason": "test kill"},
        )
        result = tool.execute(command="bash -c 'echo running; sleep 10'")
        assert "TERMINATED" in result

    def test_dangerous_command_blocked(self):
        tool = ShellTool(check_dangerous=True, require_confirm=True)
        result = tool.execute(command="rm -rf /")
        assert result.startswith("FATAL:")

    def test_dangerous_check_disabled(self):
        tool = ShellTool(check_dangerous=False, remind_interval=1, require_confirm=False)
        result = tool.execute(command="echo safe")
        assert "safe" in result

    def test_confirm_denied(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: False)
        result = tool.execute(command="rm /tmp/test_file")
        assert "DENIED" in result

    def test_confirm_approved(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: True)
        result = tool.execute(command="rm /tmp/nonexistent_flagscale_test_xyz")
        assert "DENIED" not in result

    def test_confirm_not_triggered_for_safe_commands(self):
        called = []
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: (called.append(1), False)[1])
        tool.execute(command="echo safe")
        assert len(called) == 0

    def test_confirm_allow_pattern_skips_subsequent(self):
        call_count = []
        def mock_confirm(cmd):
            call_count.append(1)
            return "allow_pattern"
        tool = ShellTool(require_confirm=True, confirm_fn=mock_confirm)
        tool.execute(command="pip install requests")
        assert len(call_count) == 1
        # Second pip install should be auto-approved
        tool.execute(command="pip install flask")
        assert len(call_count) == 1  # no additional confirm call

    def test_confirm_not_triggered_by_grep_pattern(self):
        called = []
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: (called.append(1), False)[1])
        tool.execute(command='ps aux | grep -E "pip install|conda" | grep -v grep')
        assert len(called) == 0  # grep pattern should not trigger confirm


class TestPreConfirm:
    """Test needs_confirm / pre_confirm / _skip_confirm for parallel execution."""

    def test_needs_confirm_true(self):
        tool = ShellTool(require_confirm=True)
        assert tool.needs_confirm("pip install numpy") is True

    def test_needs_confirm_false_safe_cmd(self):
        tool = ShellTool(require_confirm=True)
        assert tool.needs_confirm("echo hello") is False

    def test_needs_confirm_false_when_disabled(self):
        tool = ShellTool(require_confirm=False)
        assert tool.needs_confirm("pip install numpy") is False

    def test_pre_confirm_approved(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: True)
        assert tool.pre_confirm("pip install numpy") is True

    def test_pre_confirm_denied(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: False)
        assert tool.pre_confirm("pip install numpy") is False

    def test_pre_confirm_allow_pattern(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: "allow_pattern")
        assert tool.pre_confirm("pip install numpy") is True
        # After allow_pattern, same command should not need confirm
        assert tool.needs_confirm("pip install torch") is False

    def test_skip_confirm_bypasses_prompt(self):
        called = []
        tool = ShellTool(require_confirm=True,
                         confirm_fn=lambda cmd: (called.append(1), False)[1])
        result = tool.execute(command="echo ok", _skip_confirm=True)
        assert len(called) == 0
        assert "ok" in result


class TestStallDetection:
    def test_stall_kills_command_no_judge(self):
        """When output doesn't change across stall_threshold intervals, kill it."""
        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            stall_threshold=2,
        )
        # Command that prints once then hangs
        result = tool.execute(command="echo 'stuck here' && sleep 30")
        assert "STALLED" in result
        assert "stuck here" in result

    def test_stall_with_judge_kill(self):
        """stall_judge_fn returns kill=True → command terminated."""
        def judge(cmd, output, elapsed, stall_dur):
            return {"kill": True, "reason": "Download frozen"}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            stall_judge_fn=judge, stall_threshold=2,
        )
        result = tool.execute(command="echo 'downloading...' && sleep 30")
        assert "STALLED" in result
        assert "Download frozen" in result

    def test_stall_with_judge_continue(self):
        """stall_judge_fn returns kill=False → command keeps running."""
        judge_calls = []
        def judge(cmd, output, elapsed, stall_dur):
            judge_calls.append(1)
            return {"kill": False, "reason": "Compiling"}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            stall_judge_fn=judge, stall_threshold=1,
        )
        # Command prints once then sleeps briefly — judge says continue, command finishes
        result = tool.execute(command="echo 'compiling' && sleep 4")
        assert len(judge_calls) >= 1
        assert "STALLED" not in result

    def test_no_stall_when_output_changes(self):
        """Continuously changing output should never trigger stall."""
        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            stall_threshold=2,
        )
        result = tool.execute(command="for i in $(seq 1 10); do echo line$i; sleep 0.3; done")
        assert "STALLED" not in result
        assert "line10" in result

    def test_no_stall_on_empty_output(self):
        """Empty output (no chunks yet) should not count as stall."""
        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            stall_threshold=1,
        )
        # sleep produces no output — should NOT be killed as stalled
        result = tool.execute(command="sleep 4 && echo done")
        assert "STALLED" not in result
        assert "done" in result


class TestHealthJudge:
    def test_health_judge_kills_on_first_interval(self):
        """health_judge_fn can kill even on the first check — no stall threshold needed."""
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0):
            return {"kill": True, "reason": "Repeated connection errors"}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'Connection refused' && sleep 30")
        assert "TERMINATED" in result
        assert "Repeated connection errors" in result

    def test_health_judge_continue(self):
        """health_judge_fn says continue → command runs to completion."""
        judge_calls = []
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0):
            judge_calls.append(1)
            return {"kill": False, "reason": "Looks fine"}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'working' && sleep 3 && echo 'done'")
        assert len(judge_calls) >= 1
        assert "TERMINATED" not in result
        assert "done" in result

    def test_health_judge_receives_stall_info(self):
        """health_judge_fn receives output_changed=False and stall_count when output stalls."""
        received = []
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0):
            received.append({"output_changed": output_changed, "stall_count": stall_count})
            if stall_count >= 2:
                return {"kill": True, "reason": "Stalled too long"}
            return {"kill": False}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            health_judge_fn=judge,
        )
        result = tool.execute(command="echo 'stuck' && sleep 30")
        assert "TERMINATED" in result
        stall_counts = [r["stall_count"] for r in received]
        assert any(sc >= 2 for sc in stall_counts)
        assert any(not r["output_changed"] for r in received)

    def test_health_judge_overrides_legacy_stall(self):
        """When health_judge_fn is set, legacy stall detection is bypassed."""
        def judge(cmd, output, elapsed, output_changed=True, stall_count=0):
            return {"kill": False, "reason": "Let it run"}

        tool = ShellTool(
            remind_interval=1, require_confirm=False,
            health_judge_fn=judge,
            stall_threshold=1,
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
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command="seq 100 | tail -5")
        lines = result.strip().splitlines()
        assert lines == ["96", "97", "98", "99", "100"]

    def test_integration_head(self):
        tool = ShellTool(require_confirm=False)
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


class TestFindLatestLogTool:
    def test_missing_experiment(self, tmp_path):
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="nonexistent")
        assert "ERROR" in result
        assert "not found" in result

    def test_picks_latest_timestamp(self, tmp_path):
        """Should pick the latest timestamp dir, not an older one."""
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        old_dir = tmp_path / "myexp" / "logs" / "details" / "host_0" / "20260101_100000" / "default_a" / "attempt_0" / "0"
        new_dir = tmp_path / "myexp" / "logs" / "details" / "host_0" / "20260102_100000" / "default_b" / "attempt_0" / "0"
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)
        (old_dir / "stdout.log").write_text("old output\n")
        (new_dir / "stdout.log").write_text("new output\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="myexp", log_type="stdout", lines=10)
        assert "new output" in result
        assert "20260102" in result

    def test_picks_last_attempt(self, tmp_path):
        """Should pick attempt_1 over attempt_0."""
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        base = tmp_path / "exp" / "logs" / "details" / "host_0" / "20260101" / "default_x"
        (base / "attempt_0" / "0").mkdir(parents=True)
        (base / "attempt_1" / "0").mkdir(parents=True)
        (base / "attempt_0" / "0" / "stderr.log").write_text("old error\n")
        (base / "attempt_1" / "0" / "stderr.log").write_text("retry error\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp", log_type="stderr")
        assert "retry error" in result
        assert "attempt_1" in result

    def test_picks_last_rank(self, tmp_path):
        """Should pick rank 7 (last) over rank 0."""
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        base = tmp_path / "exp" / "logs" / "details" / "host_0" / "20260101" / "default_x" / "attempt_0"
        for r in range(8):
            (base / str(r)).mkdir(parents=True)
            (base / str(r) / "stdout.log").write_text(f"rank {r} output\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp", log_type="stdout")
        assert "rank 7 output" in result
        assert "/7/" in result

    def test_picks_last_node(self, tmp_path):
        """Multi-node: should pick host_1 over host_0."""
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        for host in ["host_0_node1", "host_1_node2"]:
            d = tmp_path / "exp" / "logs" / "details" / host / "20260101" / "default_x" / "attempt_0" / "0"
            d.mkdir(parents=True)
            (d / "stdout.log").write_text(f"{host} output\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp", log_type="stdout")
        assert "host_1_node2 output" in result

    def test_finds_both_logs(self, tmp_path):
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        log_dir = tmp_path / "exp1" / "logs" / "details" / "host_0" / "run1" / "default_x" / "attempt_0" / "0"
        log_dir.mkdir(parents=True)
        (log_dir / "stdout.log").write_text("training started\n")
        (log_dir / "stderr.log").write_text("ERROR: something went wrong\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp1", log_type="both")
        assert "stdout.log" in result
        assert "stderr" in result.lower()
        assert "training started" in result
        assert "something went wrong" in result

    def test_stderr_only(self, tmp_path):
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        log_dir = tmp_path / "exp2" / "logs" / "details" / "host_0" / "run1" / "default_x" / "attempt_0" / "0"
        log_dir.mkdir(parents=True)
        (log_dir / "stdout.log").write_text("good output\n")
        (log_dir / "stderr.log").write_text("ImportError: no module named apex\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp2", log_type="stderr")
        assert "stderr" in result.lower()
        assert "ImportError" in result
        assert "Loss rank" not in result

    def test_shows_path(self, tmp_path):
        from flagscale_agent.react.tools.find_log import FindLatestLogTool
        log_dir = tmp_path / "exp3" / "logs" / "details" / "host_0" / "run1" / "default_x" / "attempt_0" / "0"
        log_dir.mkdir(parents=True)
        (log_dir / "stdout.log").write_text("hello\n")

        tool = FindLatestLogTool(outputs_dir=str(tmp_path))
        result = tool.execute(experiment="exp3", log_type="stdout")
        assert "Path:" in result


class TestInjectProxyExports:
    def test_non_network_cmd_still_gets_proxy(self):
        result = _inject_proxy_exports("echo hello", {"HTTP_PROXY": "http://p:8080"})
        assert 'export HTTP_PROXY=http://p:8080' in result
        assert result.endswith("&& echo hello")

    def test_wget_with_proxy(self):
        env = {"HTTP_PROXY": "http://p:8080", "HTTPS_PROXY": "http://p:8080"}
        result = _inject_proxy_exports("wget http://example.com/file.tar", env)
        assert 'export HTTP_PROXY=http://p:8080' in result
        assert 'export HTTPS_PROXY=http://p:8080' in result
        assert result.endswith("&& wget http://example.com/file.tar")

    def test_pip_install_with_proxy(self):
        env = {"http_proxy": "http://p:8080"}
        result = _inject_proxy_exports("pip install numpy", env)
        assert 'export http_proxy=http://p:8080' in result

    def test_no_proxy_vars_set(self):
        result = _inject_proxy_exports("wget http://example.com", {})
        assert result == "wget http://example.com"

    def test_git_clone(self):
        env = {"HTTPS_PROXY": "http://p:8080"}
        result = _inject_proxy_exports("git clone https://github.com/repo.git", env)
        assert "export HTTPS_PROXY" in result


class TestEnsureWgetContinue:
    def test_plain_wget(self):
        assert _ensure_wget_continue("wget http://example.com/f.tar") == "wget -c http://example.com/f.tar"

    def test_wget_already_has_c(self):
        cmd = "wget -c http://example.com/f.tar"
        assert _ensure_wget_continue(cmd) == cmd

    def test_wget_already_has_continue(self):
        cmd = "wget --continue http://example.com/f.tar"
        assert _ensure_wget_continue(cmd) == cmd

    def test_no_wget(self):
        cmd = "curl -O http://example.com/f.tar"
        assert _ensure_wget_continue(cmd) == cmd

    def test_multiple_wget(self):
        cmd = "wget http://a.com/1.tar && wget http://b.com/2.tar"
        result = _ensure_wget_continue(cmd)
        assert result.count("wget -c") == 2



class TestToolEffect:
    """Tests for ToolEffect declarations on all tools."""

    def test_effect_dataclass_frozen(self):
        from flagscale_agent.react.tools.base import ToolEffect
        e = ToolEffect(reads=frozenset({"filesystem"}))
        with pytest.raises(Exception):
            e.reads = frozenset()

    def test_read_only_property(self):
        from flagscale_agent.react.tools.base import ToolEffect, EFFECT_READ_FS, EFFECT_WRITE_FS
        assert EFFECT_READ_FS.is_read_only
        assert not EFFECT_WRITE_FS.is_read_only

    def test_touches_filesystem(self):
        from flagscale_agent.react.tools.base import EFFECT_READ_FS, EFFECT_WRITE_FS, EFFECT_NETWORK
        assert EFFECT_READ_FS.touches_filesystem
        assert EFFECT_WRITE_FS.touches_filesystem
        assert not EFFECT_NETWORK.touches_filesystem

    def test_touches_network(self):
        from flagscale_agent.react.tools.base import EFFECT_NETWORK, EFFECT_READ_FS
        assert EFFECT_NETWORK.touches_network
        assert not EFFECT_READ_FS.touches_network

    def test_shell_effects(self):
        from flagscale_agent.react.tools.base import EFFECT_SHELL
        assert EFFECT_SHELL.touches_filesystem
        assert EFFECT_SHELL.touches_network
        assert EFFECT_SHELL.touches_process
        assert not EFFECT_SHELL.is_read_only

    def test_read_file_tool_has_effects(self):
        tool = ReadFileTool()
        assert tool.effects.is_read_only
        assert tool.effects.touches_filesystem

    def test_write_file_tool_has_effects(self):
        tool = WriteFileTool()
        assert tool.effects.is_write
        assert tool.effects.touches_filesystem

    def test_edit_file_tool_has_effects(self):
        tool = EditFileTool()
        assert tool.effects.is_write
        assert tool.effects.touches_filesystem

    def test_shell_tool_has_effects(self):
        tool = ShellTool()
        assert tool.effects.touches_process
        assert "training_launch" in tool.effects.side_effects

    def test_all_registered_tools_have_effects(self):
        """Every tool in the registry should have a non-default effects declaration."""
        from flagscale_agent.react.tools.base import ToolEffect
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        reg.register(WriteFileTool())
        reg.register(EditFileTool())
        reg.register(ShellTool())
        for tool in reg.all_tools():
            assert tool.effects != ToolEffect() or tool.name == "base", \
                f"Tool {tool.name} has no effects declared"


class TestDownloadExitCodeAnnotation:
    def test_nonzero_exit_wget(self):
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command="wget http://localhost:1/nonexistent 2>&1; exit 1")
        assert "non-zero" in result.lower() or "incomplete" in result.lower() or "ERROR" in result or "failed" in result.lower()
