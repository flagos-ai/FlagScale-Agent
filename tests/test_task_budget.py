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

"""Tests for the task-level time-budget health block and the _health_judge
kwarg-forwarding fix (regression: the LLM judge was silently dead because
shell.py passed kwargs _health_judge did not accept)."""

from flagscale_agent.react.judge import Judge


class CapturingProvider:
    """Records the FULL prompt of each call, returns a fixed response."""

    def __init__(self, response='{"kill": false, "reason": ""}'):
        self.response = response
        self.prompts = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return {"content": self.response}


class TestTaskBudgetBlock:
    def test_no_budget_block_is_empty(self):
        # No budget summary -> byte-identical to pre-budget prompt (backward compat).
        assert Judge._build_task_budget_block("") == ""
        assert Judge._build_task_budget_block(None) == ""
        assert Judge._build_task_budget_block("   \n ") == ""

    def test_budget_block_contains_summary_text(self):
        block = Judge._build_task_budget_block(
            "cumulative task time elapsed 50m of total budget 1h00m (83% used)"
        )
        assert "50m" in block and "83% used" in block
        assert "budget" in block.lower()

    def test_budget_block_is_cumulative_wall_clock(self):
        # Still frames a TASK-LEVEL cumulative whole-task fact.
        low = Judge._build_task_budget_block("x").lower()
        assert "cumulative" in low
        assert "task-level" in low or "task level" in low

    def test_budget_block_is_fuel_gauge_only(self):
        # The block must present itself as a FUEL GAUGE (report the fact), not a
        # driver: no kill on budget alone, and it stays an advisory (kill=false).
        low = Judge._build_task_budget_block("x").lower()
        assert "fuel gauge" in low or "fuel-gauge" in low
        assert "advisory" in low or "kill=false" in low
        # never kill on budget alone
        assert "not a kill criterion" in low or "never kill" in low

    def test_budget_block_does_not_steer_the_approach(self):
        # Regression (insight/tbench/fasttext_fzron4y_monitor_claim_refuted): the
        # judge must NOT recommend switching methods/method-classes or otherwise
        # decide HOW the agent drives — that is the agent's call. The fuel gauge
        # reports fuel; it does not grab the wheel.
        low = Judge._build_task_budget_block("x").lower()
        assert "not the driver" in low or "you are the fuel gauge" in low
        # explicitly forbids method-switch steering
        assert "switching methods" in low or "method-class" in low or "method class" in low
        assert "do not recommend" in low or "do not" in low
        # forbids shrinking the task too (still an anti-downgrade guard)
        assert "shrink" in low or "shrinking" in low

    def test_budget_block_reminds_to_think_not_what_to_do(self):
        # The one permitted nudge: state the fact + ask the agent to think
        # carefully given the remaining budget. It must NOT prescribe a specific
        # action.
        low = Judge._build_task_budget_block("x").lower()
        assert "think carefully" in low
        assert "remaining" in low or "time left" in low
        # must NOT dictate the specific move
        assert "never as a specific instruction" in low or "not as a specific" in low

    def test_budget_block_is_generic_no_task_specifics(self):
        block = Judge._build_task_budget_block("x").lower()
        for leaked in ("fasttext", "caffe", "cifar", "yelp", "epoch"):
            assert leaked not in block


class TestHealthBudgetInjection:
    def test_health_injects_budget_into_prompt(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health(
            "run something", "progress 5%", "3m", True, 0,
            task_budget="cumulative task time elapsed 50m of total budget 1h",
        )
        prompt = provider.prompts[0]
        assert "cumulative task time elapsed 50m" in prompt
        assert "Task-level time budget" in prompt

    def test_health_without_budget_omits_block(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("run something", "progress 5%", "3m", True, 0)
        prompt = provider.prompts[0]
        assert "Task-level time budget" not in prompt

    def test_health_accepts_shell_loop_kwargs(self):
        # Regression: shell.py forwards command_history + container_resources.
        # judge.health must accept them without TypeError.
        provider = CapturingProvider()
        judge = Judge(provider)
        result = judge.health(
            "cmd", "out", "5m", True, 0,
            command_history="ran 3x, killed 2x",
            container_resources="8GB RAM, 4 CPU",
            task_budget="50m of 1h",
        )
        assert isinstance(result, dict)
