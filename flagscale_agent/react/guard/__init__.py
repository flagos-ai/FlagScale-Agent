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

"""Guard system — behavioral constraints for the agent.

Guards fire at two points:
- pre: Before tool execution (can block or inject)
- post: After tool execution (can inject context)

Three action levels:
- inject: advisory message appended to context (does not block)
- block: prevents tool execution, LLM can override with _override_reason
- escalate: prevents tool execution, cannot be overridden
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from flagscale_agent.react import display
from typing import Literal, Any


@dataclass
class GuardContext:
    """Read-only snapshot passed to guards."""

    # Tool context
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None
    turn_count: int = 0
    recent_tool_names: list[str] = field(default_factory=list)
    recent_tool_history: list[dict] = field(default_factory=list)
    context_pressure: float = 0.0
    evictable_indexes: list[int] = field(default_factory=list)

    # Full message history
    messages: list[dict] = field(default_factory=list)

    # LLM response text
    assistant_text: str = ""

    # True only when assistant_text is the text the LLM produced THIS turn's
    # current iteration (completion-path consultation), not stale text scanned
    # back out of history at the top of a fresh iteration. Completion gates that
    # match on assistant_text.endswith("[TASK_COMPLETE]") MUST require this, or a
    # prior turn's trailing sentinel re-triggers them at every new turn's top.
    llm_responded: bool = False

    # LLM classify function
    classify_fn: Any = None

    # Override reason from LLM
    override_reason: str = ""


@dataclass
class GuardVerdict:
    """What the guard wants the agent to do."""

    action: Literal["allow", "block", "inject", "escalate"]
    message: str
    reason: str
    category: str  # For inject deduplication
    # Owner guard name, stamped by the registry when a block/escalate is selected
    # for presentation. Used both to name the owner in the message and to scope
    # override: an _override_reason only releases the guard whose name matches the
    # currently presented block, so a reason written for guard A can never release
    # guard B.
    guard_name: str = ""
    # A block is overridable by default (_override_reason releases it). Set
    # overridable=False for a block that must be released by a concrete corrective
    # ACTION (a specific tool call), not by a text/tool-arg override. Unlike
    # escalate (a dead-end that tells the agent to stop and ask the user), a
    # non-overridable block RE-PROMPTS: the agent can recover on its own by taking
    # the required action. Used by the single-shot completion gate, whose only exit
    # is calling plan_create — a text-inline override the model cannot reliably
    # emit was letting bare [TASK_COMPLETE] livelock the loop.
    overridable: bool = True

    @classmethod
    def block(cls, message: str, reason: str, category: str,
              overridable: bool = True) -> GuardVerdict:
        return cls(action="block", message=message, reason=reason,
                   category=category, overridable=overridable)

    @classmethod
    def inject(cls, message: str, reason: str, category: str) -> GuardVerdict:
        return cls(action="inject", message=message, reason=reason, category=category)

    @classmethod
    def escalate(cls, message: str, reason: str, category: str) -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason, category=category)


_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: Add "_override_reason" to your next tool call to proceed.\n'
    "DO: re-call the same tool with an extra parameter:\n"
    '  {"command": "...", "_override_reason": "reason why this is safe/justified"}\n'
    "DON'T: explain in text. Only _override_reason in tool_args works."
)

_TEXT_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: To override this gate and complete, add an inline '
    '_override_reason to your [TASK_COMPLETE] message.\n'
    "DO: re-emit with a reason on the same line or next line:\n"
    '  [TASK_COMPLETE]\n  _override_reason: <reason why no plan was warranted>\n'
    "DON'T: re-emit a bare [TASK_COMPLETE] — it will be blocked again."
)

_ESCALATE_HINT = (
    "\n\n🚫 ESCALATED: This tool call is blocked and cannot be overridden.\n"
    "DO NOT retry the same tool call — it will be blocked again.\n"
    "Choose a different approach. If you must proceed this way, stop and ask the user."
)


class Guard(abc.ABC):
    """Base class for all guards.

    Three action levels:
    - inject: advisory, does not block tool execution
    - block: blocks execution, LLM can override with _override_reason
    - escalate: blocks execution, cannot be overridden
    """

    name: str = "unnamed"
    priority: int = 50  # lower = higher priority

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Pre-execution check. Return block/escalate to prevent, inject to warn."""
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Post-execution check. Return inject to add context."""
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Validate LLM's override reason. Only called for block verdicts.

        Default: accept any reason longer than 5 chars.
        Override for stricter validation.
        """
        return bool(reason and len(reason.strip()) > 5)

    def reset_turn(self):
        """Called at the start of each new user message. Clear per-turn state."""
        pass


class GuardRegistry:
    """Manages all guards, runs them in priority order, deduplicates injects."""

    def __init__(self):
        self._guards: list[Guard] = []
        # Name of the guard whose block was surfaced to the agent on the most
        # recent resolve. An override reason may release ONLY this guard's block
        # (you can only override a block you were actually shown). Prevents a
        # reason written for guard A from silently releasing guard B that fired
        # for the first time this turn — the phantom-override crosstalk (Bug C).
        self._last_surfaced: str | None = None
        self._last_surfaced_reason: str | None = None

    def register(self, guard: Guard):
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)

    @staticmethod
    def _severity_rank(verdict: GuardVerdict) -> int:
        """Hardest-first ordering key (lower = surfaced first).

        escalate (0) > non-overridable block (1) > overridable block (2).
        A non-overridable block MUST be surfaced before an overridable one at the
        same decision point: otherwise the agent overrides the soft block, retries,
        and only then hits the hard wall it could have addressed first — the
        crosstalk/deadlock (Bug B).
        """
        if verdict.action == "escalate":
            return 0
        if verdict.action == "block" and not verdict.overridable:
            return 1
        return 2  # overridable block

    def _resolve(self, ctx: GuardContext, phase: str) -> GuardVerdict | None:
        """Shared block/escalate resolution for pre/post checks.

        collect-all + severity-rank + owner-scoped override:
        - Collect ALL block/escalate verdicts paired with their owning guard
          (injects still merge, deduplicated by category).
        - Rank hardest-first (escalate > non-overridable > overridable); ties
          broken by guard priority (guards are pre-sorted, so stable sort keeps
          priority order within a severity tier).
        - Present the single top-ranked block. Offer _override_reason ONLY to that
          block's owning guard via guard.accept_override — a reason written for one
          guard can never release another (Bug B root cause: global override_reason
          fed to whichever guard happened to win).
        - Stamp guard_name so the message names the owner.
        """
        blocks: list[tuple[Guard, GuardVerdict]] = []
        inject_messages: list[str] = []
        inject_categories_seen: set[str] = set()
        first_reason = ""

        check = (lambda g: g.check_pre(ctx)) if phase == "pre" else (lambda g: g.check_post(ctx))

        for guard in self._guards:
            verdict = check(guard)
            if verdict is None:
                continue

            if verdict.action in ("block", "escalate"):
                verdict.guard_name = guard.name
                blocks.append((guard, verdict))
            elif verdict.action == "inject":
                cat = verdict.category
                if cat and cat in inject_categories_seen:
                    continue
                if cat:
                    inject_categories_seen.add(cat)
                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

        if blocks:
            # Stable sort by severity; priority ties already ordered by _guards sort.
            blocks.sort(key=lambda gv: self._severity_rank(gv[1]))

            # Walk hardest-first. The reason (if any) is offered ONLY to the guard
            # currently surfaced — never carried to another guard. If the surfaced
            # guard accepts, it is released and we drop to the next hardest block;
            # any other block requires its OWN reason next turn. accept_override is
            # called at most once per guard here, so no double-eval side effects.
            surfaced: GuardVerdict | None = None
            surfaced_guard: Guard | None = None
            for guard, verdict in blocks:
                # An override reason may release a block ONLY if that block is the
                # one this registry surfaced to the agent last turn. You cannot
                # override a block you were never shown. A guard that fires for the
                # FIRST time this turn (e.g. BackupGuard on the turn's first shell)
                # was never presented, so a reason authored for a DIFFERENT guard
                # must not fall through and release it — that is the phantom
                # override crosstalk (Bug C: printed "override: backup" though no
                # backup block was ever displayed).
                if (
                    verdict.action == "block"
                    and verdict.overridable
                    and ctx.override_reason
                    and guard.name == self._last_surfaced
                    and verdict.reason == self._last_surfaced_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    display.guard_overridden(guard.name, ctx.override_reason)
                    # Intra-guard gate-crosstalk prevention: a single guard may
                    # have cascading internal gates (e.g. VerificationGuard's 5
                    # gates on plan_update(complete)). When the first gate's block
                    # is released by an override, the guard's check_pre may now
                    # return a DIFFERENT block (the next gate). Re-call check_pre
                    # and if it returns a new block with a different reason, surface
                    # THAT instead of silently releasing. This prevents an
                    # override reason written for Gate 1 from releasing Gate 5.
                    new_verdict = check(guard)
                    if (
                        new_verdict is not None
                        and new_verdict.action in ("block", "escalate")
                        and new_verdict.reason != verdict.reason
                    ):
                        new_verdict.guard_name = guard.name
                        surfaced, surfaced_guard = new_verdict, guard
                        break
                    continue  # released; fall through to next hardest block
                surfaced, surfaced_guard = verdict, guard
                break

            if surfaced is not None:
                # Name the owner guard so the agent knows WHICH guard is blocking
                # and whose criteria an override reason must satisfy — a reason for
                # another guard will not release this one.
                owner = surfaced.guard_name or surfaced_guard.name if surfaced_guard else surfaced.guard_name
                owner_tag = f"\n\n[blocked by guard: {owner}]" if owner else ""
                # Attach the appropriate hint to the surfaced (still-blocking) block.
                if surfaced.action == "block" and surfaced.overridable:
                    surfaced.message += owner_tag
                    # Add the override hint ONLY when no reason was supplied this
                    # turn. If a reason WAS given but did not release this block
                    # (rejected by accept_override, or written for a different
                    # guard), do not re-nag with the hint — the guard's own message
                    # explains what it needs, and the owner_tag names it.
                    if not ctx.override_reason:
                        if not ctx.tool_name:
                            surfaced.message += _TEXT_OVERRIDE_HINT
                        else:
                            surfaced.message += _OVERRIDE_HINT
                elif surfaced.action == "escalate":
                    surfaced.message += owner_tag + _ESCALATE_HINT
                else:
                    # Non-overridable block: no override hint, but still name the
                    # owner so the agent addresses the right guard's required action.
                    surfaced.message += owner_tag
                # Remember which guard (and which internal gate/reason) we surfaced.
                # Next turn, only THIS guard's block with THIS exact reason may be
                # released by an override reason (see loop above). Tracking the
                # reason prevents intra-guard gate-crosstalk: an override written
                # for Gate 1 (e.g. task_complete_premise_recheck) must not release
                # Gate 2 (e.g. task_complete_observation_demand) even though both
                # belong to the same VerificationGuard.
                self._last_surfaced = owner or (surfaced_guard.name if surfaced_guard else None)
                self._last_surfaced_reason = surfaced.reason
                return surfaced

        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject",
                category="merged"
            )
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks. Hardest block surfaces first (severity-ranked,
        owner-scoped override); injects merge. See _resolve."""
        return self._resolve(ctx, "pre")

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks. Same resolution as check_pre. See _resolve."""
        return self._resolve(ctx, "post")

    def reset_turn(self):
        """Reset per-turn state for all guards.

        Note: _last_surfaced is intentionally NOT cleared here — it must persist
        across turns so an override reason on turn N+1 can release the block that
        was surfaced on turn N (the block and its override live in different turns).
        """
        for guard in self._guards:
            guard.reset_turn()

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)
