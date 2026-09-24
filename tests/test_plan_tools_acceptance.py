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

"""Test plan tools with acceptance and verification."""

import pytest
import tempfile
import shutil

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


@pytest.fixture
def plan_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestSetThinkingEcho:
    """set_thinking records a model; the tool echoes the falsifiability challenge."""

    def test_set_thinking_action_echoes_challenge(self, plan_dir):
        tp = TaskPlan(plan_dir)
        PlanCreateTool(tp).execute(title="T", steps=["A"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="set_thinking", thinking="bottleneck is IO")
        assert "Model recorded" in result
        assert "cheapest observation" in result

    def test_step_done_with_thinking_echoes(self, plan_dir):
        tp = TaskPlan(plan_dir)
        PlanCreateTool(tp).execute(title="T", steps=["A"])
        tool = PlanUpdateTool(tp)
        sid = tp.get_active()["steps"][0]["id"]
        result = tool.execute(action="step_done", step_id=sid, thinking="new model")
        assert "Model recorded" in result

    def test_step_done_without_thinking_no_echo(self, plan_dir):
        tp = TaskPlan(plan_dir)
        PlanCreateTool(tp).execute(title="T", steps=["A"])
        tool = PlanUpdateTool(tp)
        sid = tp.get_active()["steps"][0]["id"]
        result = tool.execute(action="step_done", step_id=sid, notes="progress")
        assert "Model recorded" not in result


class TestPlanCreateToolStructuredSteps:
    def test_dict_steps_with_acceptance(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[
                {"title": "A", "acceptance": ["A1", "A2"]},
                {"title": "B", "acceptance": ["B1"]},
            ]
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1", "A2"]
        assert plan["steps"][1]["acceptance"] == ["B1"]

    def test_mixed_steps(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[
                {"title": "A", "acceptance": ["A1"]},
                "B",
            ]
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1"]
        assert plan["steps"][1]["acceptance"] == []

    def test_backward_compatible_string_steps(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(title="Test", steps=["A", "B"])
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["title"] == "A"
        assert plan["steps"][0]["acceptance"] == []

    def test_dict_without_title_fails(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[{"acceptance": ["A1"]}]
        )
        assert "ERROR" in result
        assert "title" in result.lower()


class TestPlanUpdateToolVerification:
    def test_step_done_with_verification(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A", "B"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(
            action="step_done",
            step_id=1,
            verification=["V1", "V2"]
        )
        assert "✓ V1" in result
        assert "✓ V2" in result
        
        plan = tp.get_active()
        assert plan["steps"][0]["verification"] == ["V1", "V2"]

    def test_step_done_without_verification(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(action="step_done", step_id=1)
        assert "✓" in result or "done" in result.lower()
        
        plan = tp.get_active()
        assert plan["steps"][0]["verification"] == []


class TestPlanUpdateToolUpdateAcceptance:
    def test_update_acceptance(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(
            action="update_acceptance",
            step_id=1,
            acceptance=["A1", "A2"]
        )
        assert "A1" in result
        assert "A2" in result
        
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1", "A2"]

    def test_update_acceptance_without_list_fails(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(action="update_acceptance", step_id=1)
        assert "ERROR" in result


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])


class TestPlanToolsThinking:
    """plan_create/plan_update expose the plan-level `thinking` slot."""

    def test_create_with_thinking(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test", steps=["A"],
            thinking="bottleneck: data too small; predict 5x lifts acc",
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["thinking"] == "bottleneck: data too small; predict 5x lifts acc"
        assert plan["thinking_updated"] > 0.0

    def test_create_without_thinking_ok(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(title="Test", steps=["A"])
        assert "Plan created" in result
        assert tp.get_active()["thinking"] == ""

    def test_set_thinking_action(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"], thinking="first model")
        tool = PlanUpdateTool(tp)
        result = tool.execute(
            action="set_thinking",
            thinking="rebuilt: real bottleneck is LR; predict 0.61->0.64",
        )
        assert "ERROR" not in result
        plan = tp.get_active()
        assert plan["thinking"] == "rebuilt: real bottleneck is LR; predict 0.61->0.64"
        assert "first model" not in plan["thinking"]

    def test_set_thinking_empty_fails(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(action="set_thinking", thinking="")
        assert "ERROR" in result

    def test_thinking_alongside_other_action(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A", "B"])
        tool = PlanUpdateTool(tp)
        result = tool.execute(
            action="step_done", step_id=1,
            thinking="model: step A confirmed tokenizer fix; predict B trivial",
        )
        assert "ERROR" not in result
        plan = tp.get_active()
        # both the progress action AND the model update took effect
        assert plan["steps"][0]["status"] == "done"
        assert "tokenizer fix" in plan["thinking"]
