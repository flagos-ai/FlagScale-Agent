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

    def mark_complex_task(self):
        """Called externally (by ComplexityJudge) when a task needs a plan."""
        self._complex_task_no_plan = True

    def reset_plan_state(self):
        """Called externally when a plan is created."""
        self._complex_task_no_plan = False
        self._pre_plan_tool_calls = 0
        self._consecutive_reads = 0
        self._block_count = 0

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
                        f"without plan creation. You MUST call plan_create NOW or "
                        f"ask the user for guidance.",
                        reason="complex task no plan persistent",
                        category="plan_needed",
                    )
                return GuardVerdict.block(
                    f"[PLAN GATE — TOOL NOT EXECUTED] This task was flagged "
                    f"as complex. You've used {self._pre_plan_tool_calls} exploratory "
                    f"calls (limit: {self._plan_gate_max_exploratory}) without creating "
                    f"a plan.\n"
                    f"This tool call was BLOCKED. You MUST call plan_create NOW.\n"
                    f"Use what you've gathered so far to create a concrete step-by-step plan.",
                    reason="complex task no plan exceeded",
                    category="plan_needed",
                )

        # Mode 2: independent — soft warn, then hard block
        if self._consecutive_reads >= self._plan_gate_independent_block:
            self._block_count += 1
            if self._block_count >= 3:
                return GuardVerdict.escalate(
                    f"[PLAN GATE] Blocked {self._block_count} times without plan creation. "
                    f"You MUST call plan_create NOW or ask the user for guidance.",
                    reason="independent plan threshold persistent",
                    category="plan_needed",
                )
            return GuardVerdict.block(
                f"[PLAN GATE — TOOL NOT EXECUTED] You've made "
                f"{self._consecutive_reads} consecutive exploratory calls "
                f"without creating a plan.\n"
                f"This tool call was BLOCKED. You MUST call plan_create NOW "
                f"to organize your approach.",
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

    def check_plan_staleness(self, task_plan, turn_count: int) -> GuardVerdict | None:
        """Check if plan's 'doing' step is stale (>8 turns without update)."""
        plan = task_plan.get_active() if task_plan else None
        if not plan:
            return None

        doing_steps = [s for s in plan.get("steps", []) if s.get("status") == "doing"]
        if not doing_steps:
            return None

        step = doing_steps[0]
        last_activity = step.get("_last_activity_turn", 0)
        turns_stale = turn_count - last_activity if last_activity else 0

        if turns_stale >= 8:
            return GuardVerdict.inject(
                f"\n[PLAN MAINTENANCE] Step {step['id']} "
                f"('{step.get('title', '')[:40]}') has had no plan_update "
                f"for {turns_stale} turns. "
                f"If it's done, call plan_update(action='step_done'). "
                f"If blocked, call plan_update(action='step_skip') and move on.",
                reason=f"plan step stale: {turns_stale} turns",
                category="plan_needed",
            )
        return None

    def reset_turn(self):
        # Do NOT reset _consecutive_reads here — reset_turn is called per iteration,
        # and we need to track consecutive reads across iterations within a turn.
        # _consecutive_reads is reset by productive tool calls in check_pre.
        pass
