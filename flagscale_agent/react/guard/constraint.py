"""ConstraintGuard — enforces compiled Constraints via pre-exec blocking.

Design:
1. Deterministic trigger: tool_name + keyword match (cheap, no LLM)
2. Precise judgment: only when triggered, call classify_fn (LLM)
3. Block behavior: violated constraints return block + correction
"""

from __future__ import annotations


from flagscale_agent.react import display
from flagscale_agent.react.constraint import Constraint
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState



class ConstraintGuard(Guard):
    """Enforces a set of compiled Constraints (pre-exec only).

    Trigger strategy:
    - First: deterministic trigger check (ConstraintTrigger.matches)
    - Then: LLM precise judgment via classify_fn
    - Finally: block + correction on violation
    """

    name = "constraint"
    priority = 25  # After safety (10), before progress (30)
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING}

    def __init__(self, constraints: list[Constraint] | None = None):
        self._constraints: list[Constraint] = constraints or []
        self._violations: dict[str, int] = {}  # constraint_id -> violation count
        self._ESCALATE_THRESHOLD = 3  # escalate after 3 blocks on same constraint

    def add_constraints(self, constraints: list[Constraint]):
        """Add constraints (e.g., after skill load)."""
        self._constraints.extend(constraints)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check constraints before tool execution."""
        for constraint in self._constraints:
            # Step 1: Deterministic trigger (cheap)
            if not constraint.trigger.matches(ctx.tool_name, ctx.tool_args, ctx.tool_result):
                continue

            # Display: constraint triggered
            print(display.dim(
                f"  🔍 Constraint triggered: [{constraint.id}]"
            ))

            # Step 2: Precise judgment via LLM
            violated, reason = self._judge_violation(ctx, constraint)

            # Display: LLM judgment result with reason (indented under trigger)
            if violated:
                print(display.yellow(
                    f"     🚫 Constraint VIOLATED: [{constraint.id}] — {reason}"
                ))
            else:
                print(display.dim(
                    f"     ✓  Constraint passed: [{constraint.id}] — {reason}"
                ))

            if not violated:
                # LLM says not violated — reset violation count for this constraint
                if constraint.id in self._violations:
                    self._violations[constraint.id] = 0
                continue

            # Step 3: Record and block (or escalate if persistent)
            count = self._violations.get(constraint.id, 0) + 1
            self._violations[constraint.id] = count


            if count >= self._ESCALATE_THRESHOLD:
                return GuardVerdict.escalate(
                    f"[Constraint] PERSISTENT VIOLATION of [{constraint.id}] "
                    f"({count} times). {constraint.correction}\n"
                    "You keep violating this constraint. STOP and ask the user for guidance.",
                    reason=f"Constraint [{constraint.id}] persistent: {reason}",
                )

            return GuardVerdict.block(
                message=constraint.correction,
                reason=f"Constraint [{constraint.id}]: {reason}",
            )

        return None

    def _judge_violation(self, ctx: GuardContext, constraint: Constraint) -> tuple[bool, str]:
        """Use LLM to precisely judge if a constraint is violated.

        Returns (violated: bool, reason: str).
        Falls back to (True, "") if no classify_fn available.
        """
        if not ctx.classify_fn:
            return (True, "no judge available")

        judge_context = {
            "constraint": constraint.description,
            "prompt": constraint.prompt,
            "tool_name": ctx.tool_name,
            "tool_args": str(ctx.tool_args),
            "tool_result": "(not yet executed — this is a pre-execution check)",
            "recent_tool_history": "(none)",
        }
        if ctx.tool_result:
            judge_context["tool_result"] = ctx.tool_result[:2000]
        if ctx.recent_tool_history:
            history_lines = [
                f"  {i+1}. [{e['tool']}] {e['args_summary']} → {e['result_summary'][:100]}"
                for i, e in enumerate(ctx.recent_tool_history)
            ]
            judge_context["recent_tool_history"] = "\n".join(history_lines)

        try:
            result = ctx.classify_fn("is_constraint_violated", judge_context)
            if isinstance(result, dict):
                return (bool(result.get("violated", True)), result.get("reason", ""))
            # Fallback for legacy bool return
            return (bool(result), "")
        except Exception as e:
            return (True, f"judge error: {e}")

    @property
    def violations(self) -> dict[str, int]:
        """Current violation counts per constraint."""
        return dict(self._violations)

    @property
    def constraints(self) -> list[Constraint]:
        """Currently loaded constraints."""
        return list(self._constraints)

    def reset_turn(self):
        """No per-turn reset needed — violations accumulate across turns."""
