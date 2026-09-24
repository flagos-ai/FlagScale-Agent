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

"""PlanGuard — reminds agent to create a plan for long tasks."""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Folded into the plan-framing gate on purpose: qualifier extraction is a
# plan-framing concern, and PlanGuard forces the plan into existence. One gate,
# one override, no laundering — a separate block caused cross-talk where the
# override_reason for the plan also released the qualifier block.
_QUALIFIER_EXTRACTION = """

While framing this plan — extract the task's qualifiers, not just its subject.

A task names a subject and usually qualifies it: a point in time, a version, a
subset, a specific metric definition. The subject is obvious; the qualifier is
easy to drop — work completes with a confident answer whether or not you honored
it. A dropped qualifier silently answers a nearby question the task did not ask.

Treat every qualifier as machine-verified by a grader you cannot reach or fool: it
re-parses the time boundary, diffs the exact version string, compares the output file
byte-for-byte against a reference, re-counts the entries. Reporting the task done does
not feed that check — if a qualifier is not literally satisfied, the scored result is
FAIL no matter how finished the work looks or how you word the completion. So a
qualifier is not a nicety to mention; it is a hard pass/fail condition to build the
plan around.

Re-read the task statement; list every qualifier as a first-class item. Fold each
into the plan as a step or acceptance criterion. A qualifier not in the plan is one
execution silently skips. But extraction is only half the job — you must also read
what boundary each qualifier draws, in the direction the task means. State each
qualifier's meaning in your own words. The common failure is bending a fixed
boundary toward what is convenient: widening to the larger, fresher, or more
available thing. A "newest / most available is best" instinct overwrites a bounded
qualifier. When the task fixes a boundary, the answer inside it is right even if a
better candidate exists outside. A bound on time is especially easy to invert: it
can mean the state as it stood at a given point (with anything that came later out of scope), or the state right now; these select different answers — let the task's words decide, not defaulting to the most up-to-date data.

Before committing to shape, apply P1's framework: name the problem CLASS, identify the STANDARD METHOD, and ensure the plan includes a SMALL-SAMPLE step that validates the method on the smallest meaningful input before scaling to the full task — the small run also gives a time estimate, and each debugging iteration at full scale costs 10-100x more. Generic steps ("analyze", "implement", "test") mean the class was never identified — a brute-force sweep is the tell.

Some qualifiers pin you to a concrete tool instance (specific version, revision, named model). The instance may have usage requirements that differ from the generic API in ANY form (preprocessing, defaults, precision, prefix). When pinned, add a step to consult that instance's own documentation before writing the call — per P1's TOOL INSTANCE rule, the task's VERB selects which documented usage applies, and the documented way is the DEFAULT. Consulting is the near half; APPLYING what the doc says is the far half — a literal-minimal reading silently swaps the task for an easier one."""

_WRITE_FILE_NO_PLAN = (
    "[Plan] You are about to write or edit a file — the concrete signal that you have "
    "started producing a deliverable. A plan must exist before you produce "
    "output.\n"
    "\n"
    "Call plan_create() now: frame the task as ordered steps with acceptance "
    "criteria naming what THIS task requires. A budget-order first step ('land a "
    "crude but complete, scorable deliverable at the required path, then refine') "
    "keeps an unsupervised run from spending its whole budget with nothing on "
    "disk.\n"
    "\n"
    "If this write is a genuinely throwaway scratch/draft file (not the "
    "deliverable), override with a one-line reason and proceed."
)


class PlanGuard(Guard):
    """Nudges (interactive) or requires (single-shot) an active plan.

    Interactive: after REMIND_THRESHOLD tool calls without a plan, injects a
    periodic reminder. Never blocks.

    Single-shot: the plan stands in as structural supervisor. Enforcement is
    front-loaded to the "start producing" moment: the first write_file with no
    plan blocks (overridable) to force plan_create before deliverable output.
    After SINGLE_SHOT_BLOCK_THRESHOLD calls without a plan, blocks further
    non-plan tools until plan_create. Overridable for genuinely-trivial tasks.
    There is NO completion-time block: completing without a plan is never a hard
    failure (that punished weak models with a livelock kill); the plan is forced
    earlier, at write_file, where it is actionable rather than punitive.
    """

    name = "plan"
    priority = 35

    # Interactive: periodic nudge cadence. Single-shot uses the tighter
    # SINGLE_SHOT_REMIND cadence — an unsupervised run should be reminded sooner,
    # since the write_file gate (not a completion block) is the real enforcement.
    REMIND_THRESHOLD = 15
    SINGLE_SHOT_REMIND = 5
    # Single-shot: allow this many observation/exploration calls before requiring
    # a plan. Set generously — an unsupervised run legitimately needs to probe the
    # environment (paths, configs, GPU state, prior findings) before it can frame a
    # plan whose steps land on real checkpoints. Blocking too early forces a plan
    # written before understanding, which is worse than no plan.
    SINGLE_SHOT_BLOCK_THRESHOLD = 20

    def __init__(self, task_plan=None, single_shot: bool = False):
        self._task_plan = task_plan
        self._calls_without_plan = 0
        self._single_shot = single_shot
        # Whether plan_create was ever called — completion gate checks this
        # (distinct from get_active(): a plan may be created then deactivated).
        self._plan_ever_created = False
        # Qualifier extraction delivered exactly once: rides the single-shot
        # block or injects on first plan_create, whichever fires first.
        self._qualifier_reminded = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot enforcement at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # No completion-time gate. Completing without a plan is intentionally
            # NOT blocked: the old NON-OVERRIDABLE completion block livelocked weak
            # models (they neither reliably override nor plan_create, so the loop
            # burned MAX_CONSECUTIVE_COMPLETION_BLOCKS and auto-killed the task →
            # reward 0). Enforcement moved earlier to write_file, where forcing a
            # plan is actionable instead of punitive.
            return None

        # write_file gate (single-shot): the first attempt to write a file with
        # no plan is the "start producing a deliverable" moment. Block once
        # (overridable) to force plan_create before deliverable output. Only
        # write_file triggers — read_file/shell exploration never does, so the
        # investigation phase is not disturbed. Once a plan exists this never
        # fires.
        if ((ctx.tool_name == "write_file" or ctx.tool_name == "edit_file")
                and self._single_shot
                and not self._plan_ever_created
                and not (self._task_plan and self._task_plan.get_active())):
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=_WRITE_FILE_NO_PLAN + _QUALIFIER_EXTRACTION,
                reason="write_file_without_plan",
                category="plan_required",
                overridable=True,
            )

        # Plan-related tools don't count toward the no-plan budget. The FIRST
        # plan_create is the plan-framing moment — inject qualifier-extraction
        # demand if not yet delivered.
        if ctx.tool_name == "plan_create":
            if not self._qualifier_reminded:
                self._qualifier_reminded = True
                return GuardVerdict.inject(
                    message="[Plan] Framing the plan." + _QUALIFIER_EXTRACTION,
                    reason="qualifier_extraction",
                    category="plan_required",
                )
            return None
        if ctx.tool_name in ("plan_update", "plan_status"):
            return None

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        self._calls_without_plan += 1

        # Single-shot: after observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
            # Block carries the qualifier demand; mark delivered to prevent
            # double-injection on the subsequent plan_create.
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=(
                    f"[Plan] Pause. {self._calls_without_plan} tool calls in this "
                    f"unsupervised run without a plan. Do you understand the problem's "
                    f"structure well enough to plan it?\n"
                    f"— If yes: call plan_create() now. Steps must land on real "
                    f"checkpoints, acceptance criteria naming what THIS task requires, "
                    f"not generic \"works correctly\".\n"
                    f"— If no: override and keep investigating. A plan written before "
                    f"understanding is worse than no plan — it locks in a shape you'll "
                    f"fight later."
                    + _QUALIFIER_EXTRACTION
                ),
                reason="single_shot_plan_required",
                category="plan_required",
            )

        # Periodic reminder. Single-shot uses the tighter cadence (5) since the
        # write_file gate is the real enforcement and an unsupervised run should
        # be nudged toward a plan sooner.
        cadence = self.SINGLE_SHOT_REMIND if self._single_shot else self.REMIND_THRESHOLD
        if self._calls_without_plan % cadence == 0:
            return GuardVerdict.inject(
                message=(
                    f"[Plan] {self._calls_without_plan} tool calls without a plan. "
                    f"If environment exploration is done and you've begun producing "
                    f"output, you understand the structure — call plan_create() to "
                    f"freeze it into steps with acceptance criteria."
                ),
                reason="plan_reminder",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
            self._plan_ever_created = True
        return None

    def reset_turn(self):
        """New user message resets counter.

        Note: _plan_ever_created is intentionally NOT reset here. In single-shot
        mode there is only one turn, so it never matters; in interactive mode the
        completion gate never fires anyway. Keeping it sticky avoids a spurious
        block if reset_turn is ever called mid-single-shot-run.
        """
        self._calls_without_plan = 0
