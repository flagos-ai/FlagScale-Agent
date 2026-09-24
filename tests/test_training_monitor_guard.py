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

"""Regression tests for TrainingMonitorGuard latch behavior.

Covers three real-world failure modes observed in a live session:

1. Heredoc false positive: writing a launch script via
   `cat > x <<'EOF' ... flagscale train ... EOF` must NOT count as a launch —
   the guard latched on a file write and blocked every subsequent tool.
2. Over-broad blocking: after a REAL launch, read-only and agent-infra tools
   (read_file, memory_*, plan_*, evict/recall) must stay available so a failed
   launch can be diagnosed; only state-mutating calls are gated.
3. Override resets state: an accepted _override_reason must clear the latch
   (regression for "override没有把状态给置回来").
"""

from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command


def _ctx(tool_name="", tool_args=None, tool_result=None, override_reason=""):
    from flagscale_agent.react.guard import GuardContext
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        override_reason=override_reason,
    )


def _latch(g):
    """Simulate a successful launch detection."""
    launch_ctx = _ctx(
        "shell",
        {"command": "flagscale train qwen3 --config /path/to/config.yaml"},
        tool_result="Training started on 8 GPUs\n",
    )
    g.check_post(launch_ctx)
    assert g._launch_detected is True


# ── 1. Heredoc body must not be misread as a launch ─────────────────────────

class TestHeredocFalsePositive:

    def test_script_write_with_launch_body_not_launch(self):
        """The session-killer: writing a launcher script containing a
        `flagscale train` line inside a quoted heredoc."""
        cmd = (
            "cat > /tmp/launch_e13b.sh <<'EOF'\n"
            "#!/bin/bash\n"
            "cd /root/baseline_build/FlagScale\n"
            "nohup flagscale train qwen36 -c conf/e13b.yaml > /tmp/l.log 2>&1 &\n"
            "EOF\n"
        )
        assert _is_flagscale_launch_command(cmd) is False

    def test_unquoted_heredoc_with_launch_body_not_launch(self):
        cmd = (
            "cat > run.sh <<EOF\n"
            "flagscale train llama\n"
            "EOF\n"
        )
        assert _is_flagscale_launch_command(cmd) is False

    def test_dash_dash_heredoc_with_launch_body_not_launch(self):
        cmd = "cat <<-'SCRIPT'\n  flagscale train qwen36\nSCRIPT\n"
        assert _is_flagscale_launch_command(cmd) is False

    def test_real_launch_still_detected(self):
        """A genuine launch must still trigger detection after the heredoc fix."""
        assert _is_flagscale_launch_command(
            "cd /root/baseline_build/FlagScale && "
            "flagscale train qwen36 -c examples/qwen36/conf/e13b_launcher.yaml"
        ) is True

    def test_heredoc_then_real_launch_detected(self):
        """Heredoc body must not swallow a launch that FOLLOWS it."""
        cmd = (
            "cat > x.sh <<'EOF'\n"
            "echo hi\n"
            "EOF\n"
            "flagscale train qwen3 -c conf/x.yaml"
        )
        assert _is_flagscale_launch_command(cmd) is True

    def test_guard_does_not_latch_on_script_write(self):
        g = TrainingMonitorGuard()
        cmd = (
            "cat > /tmp/launch_e13b.sh <<'EOF'\n"
            "flagscale train qwen36 -c conf/e13b.yaml\n"
            "EOF\n"
        )
        g.check_post(_ctx("shell", {"command": cmd}, tool_result="PID=1"))
        assert g._launch_detected is False


# ── 2. After a real launch: read-only / infra tools stay available ──────────

class TestLaunchLatchedWhitelist:

    def test_read_only_tools_not_blocked(self):
        g = TrainingMonitorGuard()
        _latch(g)
        for tool, args in (
            ("read_file", {"path": "/tmp/run.log"}),
            ("memory_list", {"keyword": "nccl"}),
            ("memory_read", {"key": "fact/cluster/x"}),
            ("plan_status", {}),
            ("inspect_checkpoint", {"path": "/tmp/ckpt.pt"}),
            ("web_fetch", {"url": "https://example.com"}),
        ):
            assert g.check_pre(_ctx(tool, args)) is None, tool

    def test_agent_infra_tools_not_blocked(self):
        g = TrainingMonitorGuard()
        _latch(g)
        for tool, args in (
            ("plan_update", {"action": "step_done", "step_id": 1}),
            ("memory_write", {"key": "fact/x/y", "type": "fact", "content": "v"}),
            ("evict", {"indexes": [1, 2, 3]}),
            ("recall", {"index": 5}),
        ):
            assert g.check_pre(_ctx(tool, args)) is None, tool

    def test_monitor_clears_and_unblocks(self):
        g = TrainingMonitorGuard()
        _latch(g)
        assert g.check_pre(_ctx("flagscale_train_monitor", {"output_dir": "/tmp"})) is None
        assert g._launch_detected is False
        # After the monitor call, mutations are unblocked again.
        assert g.check_pre(_ctx("shell", {"command": "ls"})) is None

    def test_mutations_still_blocked(self):
        """The guard's real purpose survives: state-changing calls are gated."""
        g = TrainingMonitorGuard()
        _latch(g)
        for tool, args in (
            ("shell", {"command": "rm -rf /tmp/x"}),
            ("write_file", {"path": "/tmp/x", "content": "v"}),
            ("edit_file", {"path": "/tmp/x", "old_string": "a", "new_string": "b"}),
        ):
            verdict = g.check_pre(_ctx(tool, args))
            assert verdict is not None and verdict.action == "block", tool


# ── 3. Override must reset the latch ────────────────────────────────────────

class TestOverrideResetsLatch:

    def test_accept_override_clears_state(self):
        g = TrainingMonitorGuard()
        _latch(g)
        assert g.accept_override(
            "Launch actually failed (config path error), need to inspect /tmp/l.log",
            _ctx(),
        ) is True
        assert g._launch_detected is False
        assert g.check_pre(_ctx("shell", {"command": "ls"})) is None

    def test_trivial_override_rejected(self):
        g = TrainingMonitorGuard()
        _latch(g)
        assert g.accept_override("go", _ctx()) is False
        assert g._launch_detected is True


# ── 4. Shell-aware detection: ssh wrapper + quote false positives ───────────
#
# Regression for the production miss: the real launch shape is
#   ssh -i key -p 60023 host '... && $E/bin/flagscale train qwen36 -c x.yaml'
# The old predicate stripped single-quoted content, erasing the `flagscale
# train` token → launch never detected. Symmetrically, a bare `"flagscale
# train"` argument in a grep/echo could be misread when quote pairing broke.
# The shell-aware predicate tokenizes with shlex and DESCENDS into ssh/bash -c.


class TestSshWrappedLaunchDetected:

    def test_ssh_single_quoted_launch(self):
        assert _is_flagscale_launch_command(
            "ssh -i /k/id -p 60023 root@10.8.2.152 "
            "'cd /x/FlagScale && flagscale train qwen36 -c conf/x.yaml'"
        ) is True

    def test_ssh_quoted_launch_without_dryrun_flag(self):
        assert _is_flagscale_launch_command(
            "ssh host 'nohup flagscale train qwen3 -c c.yaml'"
        ) is True

    def test_ssh_launch_with_env_var_binary_path(self):
        assert _is_flagscale_launch_command(
            "ssh -p 60023 host 'timeout 240 $ENV/bin/flagscale train qwen36 "
            "-c examples/qwen36/conf/train_llm_flash_v2.yaml'"
        ) is True

    def test_bash_c_launch(self):
        assert _is_flagscale_launch_command(
            "bash -c 'flagscale train qwen3 -c c.yaml'"
        ) is True

    def test_ssh_dryrun_still_excluded(self):
        assert _is_flagscale_launch_command(
            "ssh host 'flagscale train qwen3 -c c.yaml --dryrun'"
        ) is False

    def test_ssh_stop_still_excluded(self):
        assert _is_flagscale_launch_command(
            "ssh host 'flagscale train qwen3 --stop'"
        ) is False

    def test_ssh_wrapped_guard_latches(self):
        g = TrainingMonitorGuard()
        cmd = (
            "ssh -i /k/id -p 60023 root@10.8.2.152 "
            "'cd /x/FlagScale && flagscale train qwen36 -c conf/x.yaml'"
        )
        g.check_post(_ctx("shell", {"command": cmd}, tool_result="started\n"))
        assert g._launch_detected is True


class TestQuoteFalsePositivesStaySilent:

    def test_read_only_grep_with_double_quoted_token(self):
        assert _is_flagscale_launch_command(
            'grep -n "flagscale train " log.txt'
        ) is False

    def test_read_only_grep_mixed_quotes(self):
        # The exact shape that used to false-positive: " and ' mixed, so the
        # old pairing was off-by-one and leaked the token.
        assert _is_flagscale_launch_command(
            "grep -rn \"flagscale train\" conf/ | grep -i 'qwen'"
        ) is False

    def test_echo_single_quoted_token(self):
        assert _is_flagscale_launch_command(
            "echo 'flagscale train qwen3'"
        ) is False

    def test_substring_word_not_a_launch(self):
        # `dev_flagscale` must not be mistaken for the `flagscale` program.
        assert _is_flagscale_launch_command("git push origin dev_flagscale") is False

    def test_grep_command_stays_silent_after_ssh(self):
        assert _is_flagscale_launch_command(
            "ssh host 'grep \"flagscale train\" /tmp/train.log'"
        ) is False
