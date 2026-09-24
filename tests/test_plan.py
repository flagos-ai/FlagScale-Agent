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


class TestNotesAppendMode:
    """Notes are now append-only: each update adds a line, doesn't overwrite."""

    def test_single_note(self, tp):
        tp.create("Test", ["A"])
        plan = tp.update_step(1, "doing", "started working")
        assert plan["steps"][0]["notes"] == "started working"

    def test_append_multiple(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing", "attempt 1: failed OOM")
        plan = tp.update_step(1, "doing", "attempt 2: reduced batch size")
        notes = plan["steps"][0]["notes"]
        assert "attempt 1: failed OOM" in notes
        assert "attempt 2: reduced batch size" in notes
        assert notes == "attempt 1: failed OOM\nattempt 2: reduced batch size"

    def test_append_on_done(self, tp):
        tp.create("Test", ["A", "B"])
        tp.update_step(1, "doing", "trying approach A")
        plan = tp.update_step(1, "done", "approach A worked")
        notes = plan["steps"][0]["notes"]
        assert "trying approach A" in notes
        assert "approach A worked" in notes

    def test_empty_notes_dont_append(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing", "first note")
        plan = tp.update_step(1, "doing", "")
        # Empty notes should not add blank line
        assert plan["steps"][0]["notes"] == "first note"

    def test_notes_visible_in_summary(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing", "key finding: port 2222")
        summary = tp.summary()
        assert "key finding: port 2222" in summary

    def test_notes_visible_in_context_for_prompt(self, tp):
        tp.create("Test", ["A"])
        tp.update_step(1, "doing", "remember: use snake_case")
        ctx = tp.context_for_prompt()
        assert "remember: use snake_case" in ctx


class TestThinking:
    """Plan-level `thinking` — the single-slot 'home for deep thinking'."""

    def test_create_without_thinking_defaults_empty(self, tp):
        plan = tp.create("Test", ["A"])
        assert plan["thinking"] == ""
        assert plan["thinking_updated"] == 0.0

    def test_create_with_thinking_stamps_time(self, tp):
        plan = tp.create("Test", ["A"], thinking="bottleneck: data too small")
        assert plan["thinking"] == "bottleneck: data too small"
        assert plan["thinking_updated"] > 0.0

    def test_set_thinking_overwrites(self, tp):
        tp.create("Test", ["A"], thinking="first model")
        plan = tp.set_thinking("second model, predict acc 0.61->0.64")
        # single-slot overwrite semantics, not append
        assert plan["thinking"] == "second model, predict acc 0.61->0.64"
        assert "first model" not in plan["thinking"]
        assert plan["thinking_updated"] > 0.0

    def test_set_thinking_strips(self, tp):
        tp.create("Test", ["A"])
        plan = tp.set_thinking("  padded model  ")
        assert plan["thinking"] == "padded model"

    def test_thinking_visible_in_summary(self, tp):
        tp.create("Test", ["A"], thinking="hypothesis: tokenizer offset bug")
        assert "hypothesis: tokenizer offset bug" in tp.summary()

    def test_thinking_visible_in_context_for_prompt(self, tp):
        tp.create("Test", ["A"], thinking="predict: 5x data lifts acc")
        ctx = tp.context_for_prompt()
        assert "predict: 5x data lifts acc" in ctx

    def test_empty_thinking_not_shown(self, tp):
        tp.create("Test", ["A"])
        # No thinking-model header when thinking slot is empty.
        assert "当前问题模型" not in tp.context_for_prompt()
        assert "当前问题模型" not in tp.summary()
