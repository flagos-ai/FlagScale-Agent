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

"""PlanGuard — complex task without plan detection.

Two activation modes:
1. Complexity judge fired → hard block at _PLAN_GATE_MAX_EXPLORATORY
2. Independent: warn at dynamic threshold, hard block at dynamic threshold

v2: TaskMode-aware thresholds via SharedState. Analysis mode allows more
exploratory calls before requiring a plan.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class PlanGuard(Guard):
    """Detects complex tasks without a plan and prompts plan creation.

    Uses tool_effects.is_read_only to identify exploratory calls.
    v2: Integrates with SharedState for TaskMode-aware thresholds.
    """

    name = "plan"
    priority = 35
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING, AgentState.REVIEWING}
    overridable = True

    # Base thresholds (multiplied by TaskMode.plan_required_threshold ratio)
    _PLAN_GATE_MAX_EXPLORATORY_BASE = 6
    _PLAN_GATE_INDEPENDENT_WARN_BASE = 8
    _PLAN_GATE_INDEPENDENT_BLOCK_BASE = 12

    def __init__(self, task_plan=None):
        self._task_plan = task_plan
        self._complex_task_no_plan: bool = False
        self._pre_plan_tool_calls: int = 0
        self._consecutive_reads: int = 0
        self._block_count: int = 0  # track repeated blocks for escalation
        self._shared_state = None

    def set_shared_state(self, shared_state):
        """Receive SharedState from GuardRegistry."""
        self._shared_state = shared_state

    @property
    def _threshold_multiplier(self) -> float:
        """Get threshold multiplier from TaskMode. Higher = more tolerant."""
        if self._shared_state:
            # Normalize: implementation=1.0, analysis=2.08, porting=1.67, etc.
            return self._shared_state.task_mode.plan_required_threshold / 12.0
        return 1.0

    @property
    def _plan_gate_max_exploratory(self) -> int:
        return max(4, int(self._PLAN_GATE_MAX_EXPLORATORY_BASE * self._threshold_multiplier))

    @property
    def _plan_gate_independent_warn(self) -> int:
        return max(6, int(self._PLAN_GATE_INDEPENDENT_WARN_BASE * self._threshold_multiplier))

    @property
    def _plan_gate_independent_block(self) -> int:
        return max(8, int(self._PLAN_GATE_INDEPENDENT_BLOCK_BASE * self._threshold_multiplier))

    def _has_active_plan(self) -> bool:
        """Check if a plan already exists (active or paused)."""
        if self._task_plan is None:
            return False
        try:
            return self._task_plan.get_active() is not None
        except Exception:
            return False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Plan-related tools are always allowed
        if ctx.tool_name in ("plan_create", "memory_write", "workspace_experiment"):
            return None

        # If a plan already exists, skip all plan-gate logic — the agent is
        # executing under a plan and should not be blocked for reading files.
        if self._has_active_plan():
            return None

        # Use tool_effects to classify: read-only = exploratory
        if ctx.tool_effects.is_read_only:
            self._consecutive_reads += 1
        else:
            self._consecutive_reads = 0

        self._pre_plan_tool_calls += 1

        # Mode 1: complexity judge fired → hard block at threshold
        if self._complex_task_no_plan:
            if self._pre_plan_tool_calls > self._plan_gate_max_exploratory:
                self._block_count += 1
                if self._block_count >= 3:
                    return GuardVerdict.escalate(
                        f"[PLAN GATE] Complex task blocked {self._block_count} times "
                        f"without plan creation. Create a plan or ask the user for guidance.",
                        reason="complex task no plan persistent",
                        category="plan_needed",
                    )
                return GuardVerdict.block(
                    f"[PLAN GATE] BLOCKED: {self._pre_plan_tool_calls} exploratory calls without a plan. "
                    f"Create a plan based on what you've gathered so far.",
                    reason="complex task no plan exceeded",
                    category="plan_needed",
                )

        # Mode 2: independent — soft warn, then hard block
        if self._consecutive_reads >= self._plan_gate_independent_block:
            self._block_count += 1
            if self._block_count >= 3:
                return GuardVerdict.escalate(
                    f"[PLAN GATE] Blocked {self._block_count} times without plan creation. "
                    f"Create a plan or ask the user for guidance.",
                    reason="independent plan threshold persistent",
                    category="plan_needed",
                )
            return GuardVerdict.block(
                f"[PLAN GATE] BLOCKED: {self._consecutive_reads} consecutive exploratory calls "
                f"without a plan. Create a plan to organize your approach.",
                reason="independent plan threshold exceeded",
                category="plan_needed",
            )

        if self._consecutive_reads >= self._plan_gate_independent_warn:
            # v2: Use SharedState to suppress if another guard already warned about reads
            if self._shared_state and not self._shared_state.issue_read_warning():
                return None  # Another guard already warned this turn
            return GuardVerdict.inject(
                f"\n[PLAN REMINDER] You've made {self._consecutive_reads} "
                f"exploratory calls without a plan. Consider calling plan_create "
                f"soon to organize your findings. "
                f"You will be BLOCKED at {self._plan_gate_independent_block} calls.",
                reason="plan independent warn threshold",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name in ("plan_create",):
            self._complex_task_no_plan = False
            self._pre_plan_tool_calls = 0
            self._consecutive_reads = 0
            self._block_count = 0
        return None

    def reset_turn(self):
        # Per-iteration reset: only reset consecutive reads dedup within iteration
        # _consecutive_reads tracks patterns across iterations within a turn.
        # It is reset by productive tool calls in check_pre, not here.
        pass

    def reset_new_turn(self):
        """Reset all counters at the start of a new user turn.

        This prevents state leaking between user messages — a fresh question
        should start with a clean slate for plan-gate detection.
        """
        self._pre_plan_tool_calls = 0
        self._consecutive_reads = 0
        self._block_count = 0
        self._complex_task_no_plan = False

    def reset_state(self):
        """v3: Full state reset — called on decay or override acceptance.

        Must reset all PlanGuard-specific counters so that an accepted override
        actually allows the LLM to proceed without being immediately re-blocked.
        """
        super().reset_state()
        self._pre_plan_tool_calls = 0
        self._consecutive_reads = 0
        self._block_count = 0
        self._complex_task_no_plan = False
