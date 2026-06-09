"""PlanUpdateGuard — enforces plan updates after step completion."""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class PlanUpdateGuard(Guard):
    """Enforces plan_update after completing plan steps.

    Tracks when a plan exists and whether the agent has updated it recently.
    If the agent completes work without updating the plan, injects a reminder.
    """

    name = "plan_update"
    priority = 50  # Run after ConstraintGuard but before LoopDetectGuard
    activate_on_states = {AgentState.EXECUTING}

    def __init__(self, task_plan):
        self._task_plan = task_plan
        self._last_update_turn = -1
        self._turns_since_update = 0

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check if plan needs updating after tool execution."""
        # Only enforce if a plan exists with active steps
        if not self._task_plan:
            return None

        active_plan = self._task_plan.get_active()
        if not active_plan:
            return None

        steps = active_plan.get("steps", [])
        if not steps:
            return None

        # Track plan_update calls
        if ctx.tool_name == "plan_update":
            self._last_update_turn = ctx.turn_count
            self._turns_since_update = 0
            return None

        # Count actual turns (not tool calls) since last update
        if ctx.turn_count > self._last_update_turn:
            turns_elapsed = ctx.turn_count - self._last_update_turn
        else:
            turns_elapsed = self._turns_since_update

        # Inject reminder if plan hasn't been updated in 3+ actual turns
        if turns_elapsed >= 3 and ctx.tool_name not in ("plan_status", "plan_create"):
            doing_steps = [s for s in steps if s.get("status") == "doing"]
            if doing_steps:
                step_id = doing_steps[0].get("id")
                return GuardVerdict.inject(
                    message=(
                        f"⚠️ PLAN UPDATE REQUIRED: You have a plan with active step {step_id}, "
                        f"but haven't updated it in {turns_elapsed} turns. "
                        f"Call plan_update(action='step_done', step_id={step_id}) if you completed it, "
                        f"or plan_update(action='step_skip', step_id={step_id}, reason=...) if blocked."
                    ),
                    reason="plan_not_updated"
                )

        return None
