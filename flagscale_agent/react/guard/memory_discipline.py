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

"""MemoryDisciplineGuard — enforces proactive memory reads and writes.

Core principle: The agent should REMEMBER what it has done and CHECK before repeating.

1. PROACTIVE READING:
   - Session start: must load relevant memories before doing real work
   - Before experiments: must check if similar attempts were tried before

2. PROACTIVE WRITING:
   - After shell produces measurable results -> remind to write
   - After errors/timeouts -> remind to record the failure
   - After multiple shells without any memory_write -> periodic reminder

v3 Lifecycle:
- Satisfied when: memory_list or plan_status has been called (read concern resolved)
- Decay: after 10 idle iterations, escalation counters reset
- Inject suppression: max 2 inject messages per category before going silent
- Override: LLM can dismiss with any reason > 10 chars
"""

import re
import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_RESULT_PATTERNS = [
    (re.compile(r"(?:score|accuracy|loss|error|f1|auc|bleu)\s*[=:]\s*[\d.]+", re.I), "metric"),
    (re.compile(r"(?:solved|passed|failed)\s*[=:]?\s*\d+\s*/\s*\d+", re.I), "solve_rate"),
    (re.compile(r"(?:time|elapsed|duration)\s*[=:]\s*[\d.]+\s*(?:s|sec|min|ms)", re.I), "timing"),
    (re.compile(r"(?:params|parameters)\s*[=:]\s*[\d,]+", re.I), "params"),
    # Only match actual crashes, not any output containing the word "Error"
    (re.compile(r"(?:Traceback \(most recent|TERMINATED|TIMEOUT|FATAL ERROR)", re.I), "error"),
    (re.compile(r"(?:iteration|step|epoch)\s*[=:]\s*\d+", re.I), "training"),
]

# Experiment launch patterns — stricter than just "python" or "train" in a command
_EXPERIMENT_LAUNCH_RE = re.compile(
    r"(?:^|\s)(?:"
    r"torchrun\s+"
    r"|flagscale\s+train\b"
    r"|(?:python[23]?(?:\.\d+)?)\s+(?!-m\s)(?:\S+/)?(?:train|pretrain|run)\.py\b"
    r")",
    re.IGNORECASE,
)


class MemoryDisciplineGuard(Guard):
    """Enforce proactive memory discipline for all projects."""

    name = "memory_discipline"
    priority = 30
    overridable = True

    # v3: Inject at most 2 times per category before going silent
    max_inject_repeats = 2
    # v3: Decay after 10 idle iterations
    decay_after_idle = 10

    def __init__(self):
        super().__init__()
        self._tool_call_count = 0
        self._memory_list_done = False
        self._plan_status_done = False
        self._last_write_time = 0.0
        self._pending_discoveries: list[str] = []
        self._shells_since_write = 0
        self._write_reminders = 0
        self._read_reminders = 0
        # v3: Track if LLM has dismissed this guard's concerns
        self._dismissed = False

    def reset_state(self):
        """v3: Full state reset on decay or override."""
        super().reset_state()
        self._tool_call_count = 0
        self._memory_list_done = False
        self._plan_status_done = False
        self._last_write_time = 0.0
        self._pending_discoveries.clear()
        self._shells_since_write = 0
        self._write_reminders = 0
        self._read_reminders = 0
        self._dismissed = False

    def is_satisfied(self, ctx: GuardContext) -> bool:
        """v3: Satisfied when memory has been checked or LLM dismissed."""
        if self._dismissed:
            return True
        # Read concern satisfied if memory_list or plan_status was called
        return self._memory_list_done or self._plan_status_done

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override if LLM provides any non-trivial reason."""
        if reason and len(reason.strip()) > 10:
            # v3: Mark as dismissed so it stops firing entirely
            self._dismissed = True
            self._read_reminders = 0
            self._write_reminders = 0
            self._shells_since_write = 0
            return True
        return False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # v3: If dismissed or satisfied, stop firing
        if self._dismissed:
            return None

        # Turn-level check (tool_name=""): session start reminder
        if not ctx.tool_name:
            if not self._memory_list_done and not self._plan_status_done:
                if self._tool_call_count == 0 and self._read_reminders < 1:
                    # v3: Check inject suppression
                    if self._should_suppress_inject("memory_read_reminder"):
                        return None
                    self._read_reminders += 1
                    self._record_trigger("memory_read_reminder")
                    return GuardVerdict.inject(
                        "[MemoryDiscipline] Session start — check memory and plan status "
                        "before acting to avoid repeating past efforts.",
                        reason="session_start_read",
                        category="memory_read_reminder",
                    )
            return None
        self._tool_call_count += 1

        if ctx.tool_name == "memory_list":
            self._memory_list_done = True
            self._read_reminders = 0  # Concern resolved
        elif ctx.tool_name == "plan_status":
            self._plan_status_done = True
            self._read_reminders = 0  # Concern resolved
        elif ctx.tool_name == "memory_write":
            self._last_write_time = time.time()
            self._pending_discoveries.clear()
            self._shells_since_write = 0
            self._write_reminders = 0
        elif ctx.tool_name == "memory_read":
            # memory_read also counts as "checked memory"
            self._memory_list_done = True
            self._read_reminders = 0

        # Session start: first 3 calls should include memory check
        if self._tool_call_count <= 3 and self._read_reminders < 2:
            if not self._memory_list_done and not self._plan_status_done:
                if ctx.tool_name not in ("memory_list", "memory_read", "plan_status"):
                    # v3: Check inject suppression before firing
                    if self._should_suppress_inject("memory_read_reminder"):
                        return None
                    self._read_reminders += 1
                    self._record_trigger("memory_read_reminder")
                    return GuardVerdict.inject(
                        "[MemoryDiscipline] Session start — check memory and plan status "
                        "before acting to avoid repeating past efforts.",
                        reason="session_start_read",
                        category="memory_read_reminder",
                    )

        # Before experiment: check memory if haven't yet
        # v3: Use strict regex instead of broad keyword matching
        if ctx.tool_name == "shell" and not self._memory_list_done:
            cmd = ctx.tool_args.get("command", "")
            if _EXPERIMENT_LAUNCH_RE.search(cmd):
                self._read_reminders += 1
                # v3: Check inject suppression
                if self._should_suppress_inject("pre_experiment_check"):
                    return None
                self._record_trigger("pre_experiment_check")
                if self._read_reminders >= 4:
                    # Escalate: block after repeated ignoring
                    return GuardVerdict.block(
                        "[MemoryDiscipline] BLOCKED: Check memory before running experiments.",
                        reason="pre_experiment_block",
                    )
                return GuardVerdict.inject(
                    "[MemoryDiscipline] About to run experiment without checking memory. "
                    "Check for prior attempts/failures before proceeding.",
                    reason="pre_experiment_check",
                    category="pre_experiment_check",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name or not ctx.tool_result:
            return None

        # v3: If dismissed, stop firing
        if self._dismissed:
            return None

        # Check if LLM acknowledged/dismissed the reminder in its response
        if ctx.assistant_text:
            self._check_dismissal(ctx.assistant_text)
            if self._dismissed:
                return None

        if ctx.tool_name == "shell":
            self._shells_since_write += 1

        if ctx.tool_name == "memory_write":
            self._last_write_time = time.time()
            self._pending_discoveries.clear()
            self._shells_since_write = 0
            self._write_reminders = 0
            return None

        # Detect discoveries worth recording
        if ctx.tool_name == "shell" and ctx.tool_result:
            result_text = ctx.tool_result[:2000]
            for pattern, category in _RESULT_PATTERNS:
                if pattern.search(result_text):
                    self._pending_discoveries.append(category)
                    break

        # Error detection — remind to record
        if ctx.tool_name == "shell" and ctx.tool_result:
            result_text = ctx.tool_result[:2000]
            if re.search(r"(?:Traceback \(most recent|TERMINATED|TIMEOUT|FATAL ERROR)", result_text, re.I):
                if self._should_suppress_inject("error_record"):
                    return None
                self._record_trigger("error_record")
                return GuardVerdict.inject(
                    "[MemoryDiscipline] Error/failure detected. Record what failed and "
                    "why to memory so future sessions don't repeat this mistake.",
                    reason="error_record",
                    category="error_record",
                )

        # Periodic reminder: many shells without any memory write
        if self._shells_since_write >= 8 and self._pending_discoveries:
            if self._should_suppress_inject("periodic_write"):
                return None
            self._write_reminders += 1
            self._record_trigger("periodic_write")
            return GuardVerdict.inject(
                "[MemoryDiscipline] Multiple tool calls with discoveries but no memory_write. "
                "Consider recording key findings for future sessions.",
                reason="periodic_write",
                category="periodic_write",
            )

        return None

    def _check_dismissal(self, assistant_text: str):
        """v3: Detect if LLM has dismissed/acknowledged this guard's concerns.

        If LLM explicitly says it doesn't need to check memory or explains why,
        mark as dismissed to stop all future inject messages.
        """
        if not assistant_text:
            return
        text_lower = assistant_text.lower()
        dismiss_patterns = [
            "这不是error", "这不是错误", "not an error", "not a failure",
            "normal output", "正常输出", "不是error", "isn't an error",
            "这是正常", "this is expected", "already checked", "已经检查",
            "don't need to check", "不需要检查", "skip memory",
            "i know", "我知道了",
        ]
        if any(p in text_lower for p in dismiss_patterns):
            self._dismissed = True
            self._pending_discoveries.clear()
            self._write_reminders = 0

    # Backward compat alias
    acknowledge_from_text = _check_dismissal

    def reset_turn(self):
        """Reset per-turn ephemeral state only.

        v3: Do NOT reset _read_reminders or _write_reminders — these track
        cross-iteration escalation. Only reset per-turn dedup state.
        Knowledge state (_memory_list_done, _plan_status_done) persists.
        """
        # Only reset things that are truly per-turn (not escalation counters)
        self._pending_discoveries.clear()
        # v3: Do NOT reset _dismissed — if LLM said "I know", it persists
        # v3: Do NOT reset _read_reminders/_write_reminders — they prevent re-firing
