"""Tests for native Guard implementations (safety, progress, training_runtime, etc.)."""

from types import SimpleNamespace

from flagscale_agent.react.guard import GuardContext, GuardVerdict, GuardRegistry
from flagscale_agent.react.guard.safety import SafetyGuard
from flagscale_agent.react.guard.progress import ProgressGuard
from flagscale_agent.react.guard.loop_detect import LoopDetectGuard
from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.training_runtime import TrainingRuntimeGuard
from flagscale_agent.react.state_machine import AgentState
from flagscale_agent.react.tools.base import ToolEffect
from flagscale_agent.react.judge import Judge, JudgeBudget


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
         classify_fn=None, state=AgentState.EXECUTING, **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=state,
        classify_fn=classify_fn,
        **kwargs,
    )


# ── SafetyGuard ──────────────────────────────────────────────────────────


class TestSafetyGuard:
    def test_blocks_dangerous_command(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /etc"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_allows_safe_command(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "ls -la"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None

    def test_skips_non_shell_tools(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("read_file", {"path": "/tmp/test.py"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_blocks_when_no_classify(self):
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_error_increments_counter(self):
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',   # is_error
            '{"real": false, "need_more": null}',  # is_success
        ])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "python broken.py"},
                   "RuntimeError: something failed", classify_fn=judge.classify)
        g.check_post(ctx)
        assert g._consecutive_errors == 1

    def test_escalates_at_hard_threshold(self):
        g = SafetyGuard()
        g._consecutive_errors = 4
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',   # is_error
            '{"real": false, "need_more": null}',  # is_success
        ])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "fail"}, "RuntimeError",
                   classify_fn=judge.classify)
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "escalate"
        assert g._consecutive_errors == 5


# ── ProgressGuard ─────────────────────────────────────────────────────────


class TestProgressGuard:
    def test_tracks_reads(self):
        g = ProgressGuard()
        for i in range(5):
            ctx = _ctx("read_file", {"path": f"/tmp/file_{i}.py"}, "content",
                       tool_effects=ToolEffect(reads=frozenset({"filesystem"})))
            g.check_post(ctx)
        assert g._consecutive_reads == 5

    def test_resets_on_productive_tool(self):
        g = ProgressGuard()
        g._consecutive_reads = 10
        ctx = _ctx("write_file", {"path": "/tmp/test.py", "content": "x=1"},
                   "File written",
                   tool_effects=ToolEffect(writes=frozenset({"filesystem"})))
        g.check_post(ctx)
        assert g._consecutive_reads == 0

    def test_stale_threshold_triggers_inject(self):
        g = ProgressGuard()
        # Pre-populate: file already seen, so re-reads count
        g._read_files.add("/tmp/same.py")
        g._reads_since_last_new_file = 25
        g._progress_triggers = 0
        g._consecutive_reads = 25
        ctx = _ctx("read_file", {"path": "/tmp/same.py"}, "content",
                   tool_effects=ToolEffect(reads=frozenset({"filesystem"})))
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"


# ── LoopDetectGuard ───────────────────────────────────────────────────────


class TestLoopDetectGuard:
    def test_detects_repeated_calls(self):
        g = LoopDetectGuard()
        for _ in range(3):
            ctx = _ctx("read_file", {"path": "/tmp/same.py"})
            g.check_pre(ctx)
        # After 3 identical calls, should detect loop
        ctx = _ctx("read_file", {"path": "/tmp/same.py"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_no_loop_with_different_calls(self):
        g = LoopDetectGuard()
        for i in range(5):
            ctx = _ctx("read_file", {"path": f"/tmp/file_{i}.py"})
            result = g.check_pre(ctx)
        assert result is None


# ── ContextPressureGuard ──────────────────────────────────────────────────


class TestContextPressureGuard:
    def test_no_action_below_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.5)
        result = g.check_post(ctx)
        assert result is None

    def test_inject_at_soft_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.78)
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_compact_at_force_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.96)
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "force_compact"


# ── PlanGuard ─────────────────────────────────────────────────────────────


class TestPlanGuard:
    def test_allows_plan_tools(self):
        g = PlanGuard()
        g._complex_task_no_plan = True
        ctx = _ctx("plan_create", {})
        result = g.check_pre(ctx)
        assert result is None

    def test_blocks_after_threshold_when_complex(self):
        g = PlanGuard()
        g.mark_complex_task()
        for i in range(7):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"},
                       tool_effects=ToolEffect(reads=frozenset({"filesystem"})))
            g.check_pre(ctx)
        ctx = _ctx("read_file", {"path": "/tmp/extra.py"},
                   tool_effects=ToolEffect(reads=frozenset({"filesystem"})))
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_resets_on_plan_create(self):
        g = PlanGuard()
        g.mark_complex_task()
        g._pre_plan_tool_calls = 5
        ctx = _ctx("plan_create", {})
        g.check_post(ctx)
        assert g._complex_task_no_plan is False
        assert g._pre_plan_tool_calls == 0


# ── TrainingRuntimeGuard ──────────────────────────────────────────────────


class TestTrainingRuntimeGuard:
    def test_detects_training_launch(self):
        # Fast path handles is_training_command(torchrun)=True and is_kill_command(torchrun)=False
        # No LLM calls needed
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = TrainingRuntimeGuard()
        ctx = _ctx("shell",
                   {"command": "torchrun --nproc_per_node=8 train.py"},
                   classify_fn=judge.classify)
        g.check_post(ctx)
        assert g._awaiting_monitor is True
        assert g._training_started is True

    def test_monitor_gate_blocks_after_launch(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("shell", {"command": "pip install pkg"},
                   classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_monitor_clears_gate(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("monitor", {"output_dir": "/tmp/train"})
        result = g.check_pre(ctx)
        assert result is None
        assert g._awaiting_monitor is False

    def test_escalates_after_3_failures(self):
        g = TrainingRuntimeGuard()
        g._consecutive_train_failures = 2
        g._training_started = True
        # With cheap trigger: torchrun matches _LAUNCH_TRIGGER_RE, so
        # is_training_command is called. Kill detection skipped (no kill keyword).
        # Then is_training_failure and is_zombie_gpu are called.
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',   # is_training_command (torchrun matches trigger)
            '{"real": true, "need_more": null}',   # is_training_failure
            '{"real": false, "need_more": null}',  # is_zombie_gpu
        ])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "torchrun train.py"},
                   "RuntimeError: OOM", classify_fn=judge.classify)
        result = g.check_post(ctx)
        assert g._consecutive_train_failures == 3
        assert result is not None
        assert result.action == "escalate"

    def test_read_only_diagnostic_allowed(self):
        # nvidia-smi is fast-pathed as read-only — no LLM call needed
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("shell", {"command": "nvidia-smi"},
                   classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None or result.action != "block"


# ── GuardRegistry ─────────────────────────────────────────────────────────


class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = SafetyGuard()  # priority 10
        g2 = ProgressGuard()  # priority 30
        reg.register(g2)
        reg.register(g1)
        assert reg.guards[0].priority <= reg.guards[1].priority

    def test_check_pre_first_verdict_wins(self):
        reg = GuardRegistry()
        g = SafetyGuard()
        reg.register(g)
        # No classify_fn → blocks
        ctx = _ctx("shell", {"command": "rm -rf /"})
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_reset_turn(self):
        reg = GuardRegistry()
        g = LoopDetectGuard()
        reg.register(g)
        g._tool_call_cache[("read_file", "path=/tmp/x")] = "content"
        reg.reset_turn()
        assert len(g._tool_call_cache) == 0


# ── GuardContext ──────────────────────────────────────────────────────────


class TestGuardContextPhaseName:
    def test_phase_name_from_executing(self):
        ctx = GuardContext(current_state=AgentState.EXECUTING)
        assert ctx.phase_name == "executing"

    def test_phase_name_from_idle(self):
        ctx = GuardContext(current_state=AgentState.IDLE)
        assert ctx.phase_name == "idle"

    def test_phase_name_from_planning(self):
        ctx = GuardContext(current_state=AgentState.PLANNING)
        assert ctx.phase_name == "planning"
