"""Tests for TaskPlan."""

import os
import shutil
import tempfile

import pytest

from flagscale_agent.react.plan import TaskPlan


@pytest.fixture
def plan_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tp(plan_dir):
    return TaskPlan(plan_dir)


class TestCreate:
    def test_basic(self, tp):
        plan = tp.create("Test plan", ["Step 1", "Step 2", "Step 3"])
        assert plan["title"] == "Test plan"
        assert plan["status"] == "active"
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["status"] == "pending"
        assert plan["steps"][1]["depends_on"] == [1]

    def test_replaces_active(self, tp):
        p1 = tp.create("Plan 1", ["A"])
        p2 = tp.create("Plan 2", ["B"])
        assert tp.get_active()["id"] == p2["id"]
        old = tp._load(p1["id"])
        assert old["status"] == "paused"


class TestUpdateStep:
    def test_mark_done(self, tp):
        tp.create("Test", ["A", "B", "C"])
        plan = tp.update_step(1, "done", "finished A")
        assert plan["steps"][0]["status"] == "done"
        assert plan["steps"][0]["notes"] == "finished A"
        # Step 2 should auto-advance to doing
        assert plan["steps"][1]["status"] == "doing"

    def test_mark_doing(self, tp):
        tp.create("Test", ["A", "B"])
        plan = tp.update_step(1, "doing")
        assert plan["steps"][0]["status"] == "doing"

    def test_invalid_status(self, tp):
        tp.create("Test", ["A"])
        with pytest.raises(ValueError, match="Invalid status"):
            tp.update_step(1, "invalid")

    def test_no_active_plan(self, tp):
        with pytest.raises(ValueError, match="No active plan"):
            tp.update_step(1, "done")

    def test_step_not_found(self, tp):
        tp.create("Test", ["A"])
        with pytest.raises(ValueError, match="Step 99 not found"):
            tp.update_step(99, "done")


class TestAddSteps:
    def test_append(self, tp):
        tp.create("Test", ["A", "B"])
        plan = tp.add_steps(["C", "D"])
        assert len(plan["steps"]) == 4
        assert plan["steps"][2]["title"] == "C"
        assert plan["steps"][3]["title"] == "D"

    def test_insert_after(self, tp):
        tp.create("Test", ["A", "B"])
        plan = tp.add_steps(["X"], after_step_id=1)
        assert len(plan["steps"]) == 3
        assert plan["steps"][1]["title"] == "X"
        assert plan["steps"][1]["depends_on"] == [1]

    def test_insert_after_invalid(self, tp):
        tp.create("Test", ["A"])
        with pytest.raises(ValueError, match="Step 99 not found"):
            tp.add_steps(["X"], after_step_id=99)


class TestSkip:
    def test_skip_step(self, tp):
        tp.create("Test", ["A", "B"])
        plan = tp.skip_step(1, "not needed")
        assert plan["steps"][0]["status"] == "skipped"
        assert plan["steps"][0]["notes"] == "not needed"
        # Step 2 should auto-advance
        assert plan["steps"][1]["status"] == "doing"


class TestComplete:
    def test_complete(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "done")
        plan = tp.complete()
        assert plan["status"] == "completed"
        assert tp.get_active() is None


class TestAbandon:
    def test_abandon(self, tp):
        tp.create("Test", ["A"])
        plan = tp.abandon("changed approach")
        assert plan["status"] == "abandoned"
        assert tp.get_active() is None

    def test_abandon_no_plan(self, tp):
        with pytest.raises(ValueError):
            tp.abandon()


class TestSummary:
    def test_no_plan(self, tp):
        assert tp.summary() == "No active plan."

    def test_with_plan(self, tp):
        tp.create("Test", ["A", "B"])
        tp.update_step(1, "done")
        text = tp.summary()
        assert "Test" in text
        assert "[✓]" in text
        assert "1/2" in text


class TestContextForPrompt:
    def test_no_plan(self, tp):
        assert tp.context_for_prompt() == ""

    def test_with_plan(self, tp):
        tp.create("Test", ["A", "B"])
        ctx = tp.context_for_prompt()
        assert "<active-plan" in ctx
        assert "1. [ ] A" in ctx
        assert "</active-plan>" in ctx


class TestListPlans:
    def test_empty(self, tp):
        assert tp.list_plans() == []

    def test_multiple(self, tp):
        tp.create("Plan 1", ["A"])
        tp.abandon()
        tp.create("Plan 2", ["B", "C"])
        plans = tp.list_plans()
        assert len(plans) == 2


class TestClearCompleted:
    def test_clear(self, tp):
        tp.create("Plan 1", ["A"])
        tp.abandon()
        tp.create("Plan 2", ["B"])
        count = tp.clear_completed()
        assert count == 1
        plans = tp.list_plans()
        assert len(plans) == 1
        assert plans[0]["status"] == "active"


class TestPersistence:
    def test_reload(self, plan_dir):
        tp1 = TaskPlan(plan_dir)
        tp1.create("Persistent", ["A", "B"])
        tp1.update_step(1, "done", "completed")

        tp2 = TaskPlan(plan_dir)
        plan = tp2.get_active()
        assert plan is not None
        assert plan["title"] == "Persistent"
        assert plan["steps"][0]["status"] == "done"
        assert plan["steps"][1]["status"] == "doing"


class TestAutoSync:
    def test_auto_sync_marks_done_on_write(self, tp):
        tp.create("Test", ["Write config", "Run training"])
        tp.update_step(1, "doing")
        tp.auto_sync_step("write_file", success=True, result_summary="config.yaml", turn=3)
        plan = tp.get_active()
        assert plan["steps"][0]["status"] == "done"
        assert "config.yaml" in plan["steps"][0]["notes"]
        # Step 2 should auto-advance
        assert plan["steps"][1]["status"] == "doing"

    def test_auto_sync_records_failure(self, tp):
        tp.create("Test", ["Run training"])
        tp.update_step(1, "doing")
        tp.auto_sync_step("shell", success=False, result_summary="OOM error", turn=2)
        plan = tp.get_active()
        assert plan["steps"][0]["status"] == "doing"  # Not marked done
        assert "fail #1" in plan["steps"][0]["notes"]
        assert plan["steps"][0]["_failure_count"] == 1

    def test_auto_sync_no_plan(self, tp):
        # Should not raise
        tp.auto_sync_step("write_file", success=True, result_summary="test", turn=1)

    def test_auto_sync_no_doing_step(self, tp):
        tp.create("Test", ["A", "B"])
        # All steps are pending, no doing step
        tp.auto_sync_step("write_file", success=True, result_summary="test", turn=1)
        plan = tp.get_active()
        assert plan["steps"][0]["status"] == "pending"

    def test_auto_sync_non_write_productive_tool(self, tp):
        tp.create("Test", ["Document findings"])
        tp.update_step(1, "doing")
        # memory_write is productive but not write_file/edit_file — should not auto-complete
        tp.auto_sync_step("memory_write", success=True, result_summary="saved", turn=2)
        plan = tp.get_active()
        assert plan["steps"][0]["status"] == "doing"


class TestConsistencyCheck:
    def test_no_issues(self, tp):
        tp.create("Test", ["A", "B"])
        tp.update_step(1, "doing")
        tp.record_turn_activity(5)
        result = tp.check_consistency(6)
        assert result is None

    def test_stale_step(self, tp):
        tp.create("Test", ["A", "B"])
        tp.update_step(1, "doing")
        tp.record_turn_activity(1)
        result = tp.check_consistency(15)  # 14 turns stale > threshold 10
        assert result is not None
        assert "Step 1" in result
        assert "without progress" in result

    def test_repeated_failures(self, tp):
        tp.create("Test", ["Run training"])
        tp.update_step(1, "doing")
        for i in range(3):
            tp.auto_sync_step("shell", success=False, result_summary=f"error {i}", turn=i + 1)
        result = tp.check_consistency(5)
        assert result is not None
        assert "3 failures" in result

    def test_no_plan(self, tp):
        result = tp.check_consistency(10)
        assert result is None


class TestShouldRebuild:
    def test_below_threshold(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing")
        assert tp.should_rebuild(2) is False

    def test_above_threshold_with_failures(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing")
        for i in range(3):
            tp.auto_sync_step("shell", success=False, result_summary="err", turn=i)
        assert tp.should_rebuild(3) is True

    def test_no_plan(self, tp):
        assert tp.should_rebuild(5) is False


class TestRecordTurnActivity:
    def test_records_turn(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing")
        tp.record_turn_activity(7)
        plan = tp.get_active()
        assert plan["steps"][0]["_last_activity_turn"] == 7


class TestPlanCreateToolStringSteps:
    """Regression: LLM sometimes returns steps as a JSON string instead of array."""

    def test_steps_as_json_string(self, tp):
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        tool = PlanCreateTool(tp, session_id="test")
        result = tool.execute(
            title="Test plan",
            steps='["Create conda env", "Install deps", "Run training"]',
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["title"] == "Create conda env"
        assert plan["steps"][2]["title"] == "Run training"

    def test_steps_as_plain_string(self, tp):
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        tool = PlanCreateTool(tp, session_id="test")
        result = tool.execute(
            title="Test plan",
            steps="Create conda env\nInstall deps\nRun training",
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert len(plan["steps"]) == 3

    def test_steps_as_normal_list(self, tp):
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        tool = PlanCreateTool(tp, session_id="test")
        result = tool.execute(
            title="Test plan",
            steps=["Step A", "Step B"],
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert len(plan["steps"]) == 2
        assert plan["steps"][0]["title"] == "Step A"


class TestPlanUpdateToolStepIdParsing:
    """Regression: LLM sometimes passes step_id as 'step_1' instead of 1."""

    def test_integer_step_id(self, tp):
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool, _parse_step_id
        tp.create("Test", ["A", "B", "C"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="step_done", step_id=1)
        assert "ERROR" not in result

    def test_string_integer_step_id(self, tp):
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool
        tp.create("Test", ["A", "B", "C"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="step_done", step_id="1")
        assert "ERROR" not in result

    def test_step_underscore_format(self, tp):
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool
        tp.create("Test", ["A", "B", "C"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="step_done", step_id="step_1")
        assert "ERROR" not in result

    def test_step_space_format(self, tp):
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool
        tp.create("Test", ["A", "B", "C"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="step_doing", step_id="step 2")
        assert "ERROR" not in result

    def test_hash_format(self, tp):
        from flagscale_agent.react.tools.plan_update import _parse_step_id
        assert _parse_step_id("#3") == 3

    def test_none_returns_none(self, tp):
        from flagscale_agent.react.tools.plan_update import _parse_step_id
        assert _parse_step_id(None) is None

    def test_garbage_returns_none(self, tp):
        from flagscale_agent.react.tools.plan_update import _parse_step_id
        assert _parse_step_id("hello") is None
