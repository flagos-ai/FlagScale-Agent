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

"""KnowledgeSkillGuard — MIDDLE-phase research discipline.

Part of the three-phase guard framework:
  • START  — StartupGuard: setup pipeline (backup, network probe, first research).
  • MIDDLE — this guard: CONTINUOUS research nudges throughout the task.
  • END    — VerificationGuard: completion-time verification.

Logic (MIDDLE phase only):
- Track tool calls since last knowledge/skill/web_fetch load.
- Every 15 calls without a research load → inject a reminder.
- Every 40 calls without a research load → block (overridable).
- After any research load (web_fetch/load_knowledge/load_skill), inject an
  information-gain self-check (three-state: ACQUIRED / MISSING / SELF-FILLABLE).
- Meta tools (evict, plan, memory) don't count toward the threshold.

The START-phase duties this guard USED to carry — the single-shot early
research gate and the network probe gate — now live in StartupGuard
(NetworkProbePhase + ResearchPhase). This guard no longer has any single-shot
mode; it is always-on and advisory/periodic.

Design parallel to MemoryDisciplineGuard but with looser thresholds, because
not every task requires domain knowledge.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class KnowledgeSkillGuard(Guard):
    """Remind agent to load knowledge/skills if it hasn't done so recently."""

    name = "knowledge_skill"
    priority = 85  # Low priority — advisory

    INJECT_THRESHOLD = 15
    BLOCK_THRESHOLD = 40

    _KNOWLEDGE_TOOLS = frozenset((
        "load_knowledge", "load_skill", "web_fetch",
    ))

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "evict", "recall",
        "plan_status", "plan_create", "plan_update",
        "memory_read", "memory_list", "memory_write",
    ))

    def __init__(self):
        self._calls_since_knowledge = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Meta tools don't count (and never trigger a reminder). Checked FIRST so
        # plan/memory bookkeeping is never blocked.
        if ctx.tool_name in self._META_TOOLS:
            return None

        # Knowledge/skill loaded — reset counter.
        if ctx.tool_name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            return None

        # Read-only projection for the block/inject DECISION. The persistent
        # increment happens in check_post AFTER the tool executed, so a blocked or
        # retried call does not inflate the count on every check_pre pass.
        calls_since = self._calls_since_knowledge + 1

        if calls_since >= self.BLOCK_THRESHOLD:
            # Do NOT reset counter here — only reset in accept_override if override succeeds
            return GuardVerdict.block(
                f"[KnowledgeSkill] {self.BLOCK_THRESHOLD} tool calls without loading "
                "domain knowledge, skills, or external references. Consider whether "
                "load_knowledge()/load_skill() (INTERNAL domains), web_fetch() "
                "(EXTERNAL domains, for any field where your prior knowledge may not "
                "reflect standard methods), or networked shell operations (git clone, "
                "pip/apt install, wget, curl downloads — substantive external dependency "
                "acquisition) would help. If you are about to conclude "
                "'no better/other method exists within this library or tool budget', that "
                "is an unverified knowledge gap — web_fetch the standard technique for the "
                "problem class before you commit to it; you cannot prove 'no other method "
                "exists' with 'I do not know another method'. "
                "To override, do NOT just assert 'I know this domain' or 'I have prior "
                "experience' — that bare claim is exactly the self-exemption this gate "
                "targets, and it releases on any string. Instead make the exemption "
                "FALSIFIABLE: name the PROBLEM CLASS in one phrase AND the STANDARD "
                "METHOD you are applying to it (the concrete algorithm / API / config "
                "name, not 'my experience'). If you cannot name BOTH concretely — if it "
                "comes out as 'some heuristic' or 'I'll tune it' — that vagueness IS the "
                "knowledge gap, and you should research before overriding.",
                reason=f"no_knowledge_load_{self.BLOCK_THRESHOLD}_calls",
                category="knowledge_skill",
            )

        if calls_since % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {calls_since} tool calls without "
                "loading domain knowledge, skills, or external references. Two channels: "
                "(1) INTERNAL domain → load_knowledge()/load_skill() (e.g. parallelism, "
                "training config, NCCL, data pipeline, model porting). "
                "(2) EXTERNAL domain → web_fetch() for any field where your prior knowledge "
                "may not reflect standard methods. The trigger is a "
                "KNOWLEDGE GAP, not a syntax error — and note you often will NOT feel the "
                "gap. If you are about to reach for a hand-tuned threshold, a static "
                "assumption, or conclude 'no other method is available within this "
                "library/tool', that conclusion is itself an unverified knowledge gap: "
                "web_fetch the standard technique for the problem class BEFORE committing. "
                "Before you proceed without research, run one falsifiable self-check: can "
                "you name this task's PROBLEM CLASS and the STANDARD METHOD (concrete "
                "algorithm / API / config) you are applying? If yes, proceed. If the "
                "answer is a vague 'I'll figure it out' or 'some heuristic', that is the "
                "gap — look it up first.",
                reason="no_knowledge_load_recently",
                category="knowledge_skill",
            )

        return None


    def _persist_call_count(self, ctx: GuardContext) -> None:
        """Advance the persistent counter for an ACTUALLY-EXECUTED tool call.

        Mirrors the skip/reset rules of check_pre, but runs post-execution so a
        blocked or not-yet-run call never advances the count. Rules:
          • No tool name → nothing happened, skip.
          • Result carries the [BLOCKED BY GUARD] marker → the call was prevented,
            do NOT count it (this is the exact over-count source).
          • Knowledge tool executed → reset counter.
          • Meta tool → does not count.
          • Anything else that ran → advance the counter by one.
        """
        name = ctx.tool_name
        if not name:
            return
        result = ctx.tool_result
        if isinstance(result, str) and "[BLOCKED BY GUARD]" in result:
            return
        if name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            return
        if name in self._META_TOOLS:
            return
        # A real, executed tool call.
        self._calls_since_knowledge += 1

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Persist the call-count increment HERE — after the tool has actually
        # executed. check_pre only READS a projected count for its block/inject
        # decision; doing the increment here means a tool call that was ultimately
        # blocked (its result carries the [BLOCKED BY GUARD] marker) or retried
        # does NOT inflate the counter on every check_pre pass.
        self._persist_call_count(ctx)

        # Information-gain inject for knowledge tools (inject, not block).
        # Aligns with the system prompt's Information Gain section: after any
        # knowledge acquisition (web_fetch / load_knowledge / load_skill), inject a
        # gentle nudge to self-check whether the retrieval produced NEW information
        # the agent did not already know. Advisory (inject), not a block.
        if ctx.tool_name in self._KNOWLEDGE_TOOLS and ctx.tool_result:
            result_str = str(ctx.tool_result)
            is_error = "[WEB_FETCH_NETWORK_ERROR]" in result_str.upper()
            if is_error:
                return GuardVerdict.inject(
                    "[KnowledgeSkill] Your last web_fetch hit a network error. "
                    "Did this attempt teach you something you didn't already know? "
                    "If not, you are stalled — try a different fallback: proxy unset, "
                    "mirror source, case-flipped URL, alternative endpoint, or offline "
                    "cache. Information gain is the exit, not a retry of the same path.",
                    reason="web_fetch_network_info_gain",
                    category="knowledge_skill",
                )
            # Success — classify the info-gain into one of three states so the
            # agent knows its NEXT move, not just whether it "learned something".
            return GuardVerdict.inject(
                "[KnowledgeSkill] You just fetched external knowledge. Classify what "
                "you now have into ONE of three states, then take that state's exit:\n"
                "  • ACQUIRED — the fetch answered the gap with something you did NOT "
                "already know. State the concrete gain (the fact/API/method) and "
                "PROCEED to apply it.\n"
                "  • MISSING — the gap is still open and it is NOT self-fillable "
                "(needs an external fact you don't have). Do NOT reason from memory — "
                "fetch AGAIN with a narrower query, a different source, or a mirror.\n"
                "  • SELF-FILLABLE — the fetch gave you enough that the remaining gap "
                "can be closed by REASONING from what you already have (inference, "
                "connecting known facts). Don't over-fetch — reason it out, then proceed.\n"
                "If the fetch merely confirmed prior knowledge and added nothing (not "
                "any of the three), you are stalled — narrow the query or switch source.",
                reason="knowledge_info_gain_three_state",
                category="knowledge_skill",
            )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of the call-count block on a substantive reason (>5 chars).

        The START-phase gates (single-shot early research, network probe/recovery)
        that used stricter, evidence-scoped overrides now live in StartupGuard.
        Only the periodic call-count block remains here, released by any
        substantive reason.
        """
        if not (reason and len(reason.strip()) > 5):
            return False
        self._calls_since_knowledge = 0
        return True

    def reset_turn(self):
        """Don't reset per-turn — knowledge need persists across turns."""
        pass
