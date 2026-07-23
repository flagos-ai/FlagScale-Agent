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

"""ComprehensionGateGuard — enforces "understand before modify" for complex infra code.

Core principle: If you can't draw the timeline, you don't understand it.
Don't start writing code until you have a mental model you can reason about.

Three enforcement mechanisms:
1. Model Requirement: Before editing pipeline/parallelism code, agent must have
   written a comprehension model (timeline + invariants) to memory.
2. Modification Impact Check: Before editing, agent must predict what the change
   will affect (which ranks, which steps, which invariants).
3. Post-failure Comprehension Audit: After a failure in complex code, verify
   the mental model covers the failure point. If not, require model update.

Trigger: Activated when agent edits files in "complex infra" paths (pipeline
schedules, parallelism code, communication patterns).
"""

import re
from typing import Optional

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Paths that indicate complex infra code requiring comprehension
_COMPLEX_INFRA_PATTERNS = [
    re.compile(r"pipeline.parallel", re.I),
    re.compile(r"dualpipe", re.I),
    re.compile(r"schedules?\.py$", re.I),
    re.compile(r"schedule_plan", re.I),
    re.compile(r"token_dispatcher", re.I),
    re.compile(r"moe_layer", re.I),
    re.compile(r"expert.parallel", re.I),
    re.compile(r"context.parallel", re.I),
    re.compile(r"tensor.parallel", re.I),
    re.compile(r"p2p_communication", re.I),
    re.compile(r"dist_signal_handler", re.I),
    re.compile(r"overlap", re.I),
]

# Keywords in tool results that indicate the agent read complex infra code
_INFRA_READ_INDICATORS = [
    re.compile(r"def\s+(?:forward|backward)_step", re.I),
    re.compile(r"send_forward|recv_forward|send_backward|recv_backward", re.I),
    re.compile(r"pipeline_model_parallel", re.I),
    re.compile(r"num_microbatches|num_chunks", re.I),
    re.compile(r"alltoall|all_gather|reduce_scatter", re.I),
    re.compile(r"ScheduleNode|schedule_plan", re.I),
]

# Keywords that indicate a comprehension model has been built
_MODEL_KEYWORDS = [
    "timeline",
    "invariant",
    "rank 0",
    "rank 1",
    "step ",
    "fwd",
    "bwd",
    "send",
    "recv",
    "p2p",
    "constraint",
    "must be",
    "before",
    "after",
    "每个 rank",
    "每个 step",
    "不变量",
]


class ComprehensionGateGuard(Guard):
    """Enforce comprehension before modification of complex infra code.

    Philosophy: Speed comes from understanding, not from iteration speed.
    Three blind trials > one informed modification in wall-clock time.
    """

    name = "comprehension_gate"
    priority = 20  # Between training_attempt(15) and debug_discipline(22)
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override with any substantive reason."""
        if reason and len(reason.strip()) > 10:
            return True
        return False

    # Thresholds
    MIN_SOURCE_READS = 2  # Must read ≥2 complex files before editing
    MIN_MODEL_KEYWORDS = 3  # Memory/response must contain ≥3 model keywords
    EDIT_GRACE_PERIOD = 1  # Allow 1 edit before enforcing (for trivial fixes)

    def __init__(self):
        self._in_complex_context = False  # Are we working on complex infra?
        self._complex_files_read: set[str] = set()  # Complex files agent has read
        self._comprehension_model_written = False  # Has agent written a model?
        self._model_memory_key: str = ""  # Key where model is stored
        self._edits_to_complex_files = 0  # Edits without comprehension model
        self._failure_in_complex_code = False  # Did a failure occur?
        self._impact_declared = False  # Did agent declare expected impact?
        self._current_complex_path: str = ""  # What complex file is being edited

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Gate code edits to complex infra behind comprehension requirement."""
        if not ctx.tool_name:
            return None

        if ctx.tool_name in ("edit_file", "write_file"):
            path = ctx.tool_args.get("path", "")
            if not self._is_complex_infra(path):
                return None

            self._in_complex_context = True
            self._current_complex_path = path
            self._edits_to_complex_files += 1

            # Grace period: allow first edit (might be a trivial fix)
            if self._edits_to_complex_files <= self.EDIT_GRACE_PERIOD:
                return None

            # Check 1: Has agent read enough source code?
            if len(self._complex_files_read) < self.MIN_SOURCE_READS:
                return GuardVerdict.inject(
                    f"[ComprehensionGate] You're editing complex infra code "
                    f"({self._short_path(path)}) but have only read "
                    f"{len(self._complex_files_read)}/{self.MIN_SOURCE_READS} "
                    f"related source files. "
                    f"Read more of the execution flow before modifying — you need to understand "
                    f"what happens at each timestep on each rank and which P2P operations pair together.\n"
                    f"Files read so far: {list(self._complex_files_read)[:5]}",
                    reason="insufficient_source_reading",
                    category="comprehension",
                )

            # Check 2: Has agent built a comprehension model?
            if not self._comprehension_model_written:
                return GuardVerdict.inject(
                    f"[ComprehensionGate] You're editing complex infra code "
                    f"({self._short_path(path)}) without a documented comprehension model. "
                    f"Write a model to memory BEFORE modifying, containing: "
                    f"(1) TIMELINE: per-rank, per-step sequence of operations, "
                    f"(2) INVARIANTS: rules that must always hold, "
                    f"(3) IMPACT PREDICTION: what your edit changes in the timeline. "
                    f"Use memory_write(key='comprehension_<feature>', type='finding', "
                    f"content='TIMELINE: ...\\nINVARIANTS: ...').",
                    reason="no_comprehension_model",
                    category="comprehension",
                )

            # Check 3: After failure, require impact prediction
            if self._failure_in_complex_code and not self._impact_declared:
                return GuardVerdict.inject(
                    f"[ComprehensionGate] Previous attempt failed in complex code. "
                    f"Before editing again, state in your response: "
                    f"which ranks are affected, which timeline steps change, "
                    f"and which invariants need re-verification.",
                    reason="impact_not_declared",
                    category="comprehension",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track reading of complex files and comprehension model creation."""
        if not ctx.tool_name:
            return None

        # Track reads of complex infra files
        if ctx.tool_name == "read_file" and ctx.tool_result:
            path = ctx.tool_args.get("path", "")
            if self._is_complex_infra(path):
                self._complex_files_read.add(path)
                self._in_complex_context = True
            # Also count if result contains infra indicators
            elif any(p.search(ctx.tool_result) for p in _INFRA_READ_INDICATORS):
                self._complex_files_read.add(path)
                self._in_complex_context = True

        # Track memory writes that contain comprehension models
        if ctx.tool_name == "memory_write" and ctx.tool_args:
            content = ctx.tool_args.get("content", "")
            key = ctx.tool_args.get("key", "")
            if self._looks_like_comprehension_model(content):
                self._comprehension_model_written = True
                self._model_memory_key = key

        # Track failures in complex code context
        # Only detect failures in execution results (shell), NOT in source code being read
        if (
            self._in_complex_context
            and ctx.tool_result
            and ctx.tool_name in ("shell", "monitor", "find_latest_log")
        ):
            if self._is_complex_failure(ctx.tool_result):
                self._failure_in_complex_code = True
                self._impact_declared = False
                # Check if comprehension model covers the failure
                return self._check_model_coverage(ctx)

        # Track impact declarations (agent states what will change)
        if ctx.tool_name == "memory_write" and ctx.tool_args:
            content = ctx.tool_args.get("content", "").lower()
            if "impact" in content or "影响" in content:
                self._impact_declared = True

        return None

    def declare_comprehension(self, model_key: str = ""):
        """Called when agent explicitly declares comprehension model is ready."""
        self._comprehension_model_written = True
        self._model_memory_key = model_key

    def declare_impact(self, impact: str = ""):
        """Called when agent declares expected impact of a change."""
        self._impact_declared = True

    def reset_complex_context(self):
        """Reset when moving to a different task."""
        self._in_complex_context = False
        self._failure_in_complex_code = False
        self._impact_declared = False
        self._edits_to_complex_files = 0

    def reset_turn(self):
        """Reset escalation counters but keep comprehension model."""
        self._edits_to_complex_files = 0

    # ── Private helpers ──

    def _is_complex_infra(self, path: str) -> bool:
        """Check if a file path is complex infra code."""
        if not path:
            return False
        return any(p.search(path) for p in _COMPLEX_INFRA_PATTERNS)

    def _looks_like_comprehension_model(self, content: str) -> bool:
        """Check if content looks like a comprehension model (timeline + invariants)."""
        content_lower = content.lower()
        keyword_count = sum(1 for kw in _MODEL_KEYWORDS if kw in content_lower)
        return keyword_count >= self.MIN_MODEL_KEYWORDS

    def _is_complex_failure(self, text: str) -> bool:
        """Detect training failure in complex code context."""
        failure_patterns = [
            r"NCCL\s+(?:error|timeout)",
            r"TRAINING\s+CRASHED",
            r"RuntimeError:",
            r"hang|deadlock",
            r"P2P.*(?:mismatch|timeout)",
        ]
        return any(re.search(p, text, re.I) for p in failure_patterns)

    def _check_model_coverage(self, ctx: GuardContext) -> Optional[GuardVerdict]:
        """After failure, check if comprehension model needs updating."""
        if not self._comprehension_model_written:
            return GuardVerdict.inject(
                "[ComprehensionGate] Failure detected in complex infra code, "
                "but you have no comprehension model. "
                "Based on what you've already read, write a comprehension model to memory "
                "containing: (1) per-rank timeline of operations, "
                "(2) invariants that must hold, (3) which invariant this failure likely violates. "
                "Use memory_write with key='comprehension_<feature>'.",
                reason="failure_without_model",
                category="comprehension",
            )
        # Has model, suggest checking against it
        return GuardVerdict.inject(
            f"[ComprehensionGate] Failure in complex infra code. "
            f"Check your comprehension model (memory key: '{self._model_memory_key}') — "
            f"does the timeline predict this failure? Which invariant was violated? "
            f"Update the model if needed, then fix based on the updated model.",
            reason="failure_check_model",
            category="comprehension",
        )

    @staticmethod
    def _short_path(path: str) -> str:
        """Shorten a path for display."""
        parts = path.split("/")
        if len(parts) > 3:
            return "/".join(parts[-3:])
        return path
