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

"""Tests for native Guard implementations (safety, progress, training_runtime, etc.)."""

from types import SimpleNamespace

from flagscale_agent.react.guard import GuardContext, GuardVerdict, GuardRegistry
from flagscale_agent.react.guard.safety import ShellSafetyGuard

from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.judge import Judge


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:100])
        resp = self.responses.pop(0) if self.responses else "{}"
        return {"content": resp}


def _ctx(tool_name="", tool_args=None, tool_result=None,
         classify_fn=None, **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        classify_fn=classify_fn,
        **kwargs,
    )


# ── ShellSafetyGuard ──────────────────────────────────────────────────────────


class TestShellSafetyGuard:
    def test_blocks_dangerous_command(self):
        # First response: is_fatal=false, second: is_dangerous=true
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /etc"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_escalates_fatal_command(self):
        # is_fatal=true → escalate (cannot override)
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "escalate"
        assert "FATAL" in result.message

    def test_allows_safe_command(self):
        # is_fatal=false, is_dangerous=false
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": false, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "ls -la"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None

    def test_skips_non_shell_tools(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("read_file", {"path": "/tmp/test.py"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_check_post_returns_none(self):
        """After refactor, safety check_post does nothing."""
        g = ShellSafetyGuard()
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "python broken.py"},
                   "RuntimeError: something failed", classify_fn=judge.classify)
        result = g.check_post(ctx)
        assert result is None



# ── ContextPressureGuard ──────────────────────────────────────────────────


class TestContextPressureGuard:
    def test_no_action_below_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.5)
        result = g.check_pre(ctx)
        assert result is None

    def test_no_action_at_78_percent(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.78)
        result = g.check_pre(ctx)
        assert result is None  # below 80% threshold

    def test_block_at_80_percent_with_evictable(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.82,
                   evictable_indexes=list(range(80)))
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert result.category == "context_pressure_evict"

    def test_block_at_hard_reset_threshold(self):
        g = ContextPressureGuard()
        # pressure >= 85% AND evictable < 50 → block non-save tools
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.88,
                   evictable_indexes=[1, 2, 3, 4, 5])
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "hard_reset" in (result.category or "")

    def test_hard_reset_allows_save_tools(self):
        g = ContextPressureGuard()
        ctx = _ctx("memory_write", {"key": "x", "type": "fact", "content": "y"},
                   context_pressure=0.88, evictable_indexes=[1, 2, 3])
        result = g.check_pre(ctx)
        assert result is None


# ── PlanGuard ─────────────────────────────────────────────────────────────


class TestPlanGuard:
    def test_allows_plan_tools(self):
        g = PlanGuard()
        ctx = _ctx("plan_create", {})
        result = g.check_pre(ctx)
        assert result is None

    def test_reminds_at_threshold(self):
        g = PlanGuard(task_plan=None)
        for i in range(14):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"})
            result = g.check_pre(ctx)
            assert result is None
        # 15th call triggers
        ctx = _ctx("shell", {"command": "ls"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"

    def test_reminds_periodically(self):
        g = PlanGuard(task_plan=None)
        for i in range(30):
            ctx = _ctx("shell", {"command": f"cmd{i}"})
            g.check_pre(ctx)
        # 30th call should also trigger (second reminder)
        assert g._calls_without_plan == 30
        ctx = _ctx("shell", {"command": "extra"})
        result = g.check_pre(ctx)
        # 31st call, not multiple of 15 -> no inject
        assert result is None

    def test_resets_on_plan_create(self):
        g = PlanGuard(task_plan=None)
        g._calls_without_plan = 20
        ctx = _ctx("plan_create", {})
        g.check_post(ctx)
        assert g._calls_without_plan == 0

    def test_does_not_remind_when_plan_exists(self):
        from unittest.mock import MagicMock
        task_plan = MagicMock()
        task_plan.get_active.return_value = {"title": "test", "steps": []}
        g = PlanGuard(task_plan=task_plan)
        for i in range(30):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"})
            result = g.check_pre(ctx)
            assert result is None

    def test_reset_turn(self):
        g = PlanGuard(task_plan=None)
        g._calls_without_plan = 20
        g.reset_turn()
        assert g._calls_without_plan == 0



# ── GuardRegistry ─────────────────────────────────────────────────────────


class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = ShellSafetyGuard()  # priority 10
        g2 = ContextPressureGuard()  # priority 60
        reg.register(g2)
        reg.register(g1)
        assert reg.guards[0].priority <= reg.guards[1].priority

    def test_check_pre_first_verdict_wins(self):
        reg = GuardRegistry()
        g = ShellSafetyGuard()
        reg.register(g)
        # is_fatal=False, is_dangerous=True → blocks
        provider = MockProvider(responses=['{"decision": false}', '{"decision": true}'])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "rm -rf /"},
                   classify_fn=judge.classify)
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_reset_turn(self):
        reg = GuardRegistry()
        g = ContextPressureGuard()
        reg.register(g)
        reg.reset_turn()
        # Should not raise — guards can be reset without error


# ── GuardContext ──────────────────────────────────────────────────────────



# ── _is_flagscale_launch_command ──────────────────────────────────────────


class TestIsFlagscaleLaunchCommand:
    """Test precise FlagScale launch detection."""

    def test_flagscale_train_basic(self):
        assert _is_flagscale_launch_command("flagscale train qwen3_0_6b") is True

    def test_flagscale_train_with_config(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --config /path/to/config.yaml") is True

    def test_flagscale_run_with_config(self):
        assert _is_flagscale_launch_command(
            "flagscale run --config-path /workspace --config-name train_config"
        ) is True

    def test_python_run_py(self):
        assert _is_flagscale_launch_command(
            "python run.py --config-path=/workspace --config-name=train action=run"
        ) is True

    def test_python3_run_py(self):
        # v6: Pattern 3 requires action=run
        assert _is_flagscale_launch_command(
            "python3 run.py --config-path=/workspace --config-name=train action=run"
        ) is True
        # Without action=run → not a launch
        assert _is_flagscale_launch_command(
            "python3 run.py --config-path=/workspace --config-name=train"
        ) is False

    def test_flagscale_train_stop_not_launch(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --stop") is False

    def test_flagscale_train_dryrun_not_launch(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --dryrun") is False

    def test_flagscale_run_action_stop_not_launch(self):
        assert _is_flagscale_launch_command(
            "flagscale run --config-path /p --config-name c --action stop"
        ) is False

    def test_grep_flagscale_not_launch(self):
        """grep with flagscale keyword should NOT be detected as launch."""
        assert _is_flagscale_launch_command('grep "flagscale train" logs/') is False

    def test_echo_flagscale_not_launch(self):
        assert _is_flagscale_launch_command('echo "flagscale train qwen3"') is False

    def test_git_push_flagscale_not_launch(self):
        assert _is_flagscale_launch_command("git push origin dev_flagscale") is False

    def test_cat_run_py_not_launch(self):
        assert _is_flagscale_launch_command("cat run.py") is False

    def test_cd_flagscale_not_launch(self):
        assert _is_flagscale_launch_command("cd /workspace/FlagScale && ls") is False

    def test_compound_with_launch(self):
        """Compound command where one segment is a real launch."""
        assert _is_flagscale_launch_command(
            "cd /workspace/FlagScale && flagscale train qwen3_0_6b"
        ) is True

    def test_plain_torchrun_not_launch(self):
        """torchrun alone is NOT a FlagScale launch."""
        assert _is_flagscale_launch_command("torchrun --nproc_per_node=8 train.py") is False





# ── TrainingMonitorGuard ──────────────────────────────────────────────────

class TestTrainingMonitorGuard:

    def test_no_launch_no_block(self):
        g = TrainingMonitorGuard()
        ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(ctx) is None

    def test_launch_detected_blocks_non_monitor(self):
        g = TrainingMonitorGuard()
        launch_ctx = _ctx("shell", {"command": "python3 run.py --config-path=conf --config-name=config action=run"})
        g.check_post(launch_ctx)
        next_ctx = _ctx("shell", {"command": "ls"})
        verdict = g.check_pre(next_ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_launch_then_monitor_clears(self):
        g = TrainingMonitorGuard()
        launch_ctx = _ctx("shell", {"command": "python3 run.py --config-path=conf --config-name=config action=run"})
        g.check_post(launch_ctx)
        monitor_ctx = _ctx("flagscale_train_monitor", {"output_dir": "/tmp"})
        verdict = g.check_pre(monitor_ctx)
        assert verdict is None
        next_ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(next_ctx) is None

    def test_empty_tool_name_not_blocked(self):
        """Pre-iteration synthetic check (tool_name='') must not be blocked.
        This prevents the infinite loop where kernel.py continues without
        calling the LLM."""
        g = TrainingMonitorGuard()
        # Simulate successful launch
        launch_ctx = _ctx("shell", {"command": "python3 run.py --config-path=conf --config-name=config action=run"})
        g.check_post(launch_ctx)
        # Pre-iteration check with empty tool_name should pass
        pre_iter_ctx = _ctx("", {})
        assert g.check_pre(pre_iter_ctx) is None
        # But actual tool calls should still be blocked
        shell_ctx = _ctx("shell", {"command": "ls"})
        verdict = g.check_pre(shell_ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_failed_launch_not_detected(self):
        """If the launch command fails (error in output), don't set _launch_detected."""
        g = TrainingMonitorGuard()
        launch_ctx = _ctx(
            "shell",
            {"command": "python3 run.py --config-path=conf --config-name=config_bad action=run"},
            tool_result="Cannot find primary config 'config_bad'. Check that it's in your config search path.\n"
        )
        g.check_post(launch_ctx)
        # Should NOT be blocked since launch failed
        next_ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(next_ctx) is None

    def test_failed_launch_traceback(self):
        """Traceback in output indicates failure."""
        g = TrainingMonitorGuard()
        launch_ctx = _ctx(
            "shell",
            {"command": "python3 run.py --config-path=conf --config-name=config action=run"},
            tool_result="Traceback (most recent call last):\n  File \"run.py\", line 1\nImportError: No module named 'megatron'\n"
        )
        g.check_post(launch_ctx)
        next_ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(next_ctx) is None

    def test_successful_launch_still_detected(self):
        """Normal output (no error patterns) should still trigger detection."""
        g = TrainingMonitorGuard()
        launch_ctx = _ctx(
            "shell",
            {"command": "python3 run.py --config-path=conf --config-name=config action=run"},
            tool_result="Training started on 8 GPUs\nOutput dir: /workspace/outputs\n"
        )
        g.check_post(launch_ctx)
        next_ctx = _ctx("shell", {"command": "ls"})
        verdict = g.check_pre(next_ctx)
        assert verdict is not None
        assert verdict.action == "block"
