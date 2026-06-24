"""MemoryDisciplineGuard — enforces proactive memory reads and writes.

Core principle: The agent should REMEMBER what it has done and CHECK before repeating.

1. PROACTIVE READING:
   - Session start: must load relevant memories before doing real work
   - Before experiments: must check if similar attempts were tried before

2. PROACTIVE WRITING:
   - After shell produces measurable results -> remind to write
   - After errors/timeouts -> remind to record the failure
   - After multiple shells without any memory_write -> periodic reminder
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


class MemoryDisciplineGuard(Guard):
    """Enforce proactive memory discipline for all projects."""

    name = "memory_discipline"
    priority = 30
    overridable = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept override if LLM provides any non-trivial reason."""
        if reason and len(reason.strip()) > 10:
            self._read_reminders = 0
            self._write_reminders = 0
            self._shells_since_write = 0
            return True
        return False

    def __init__(self):
        self._tool_call_count = 0
        self._memory_list_done = False
        self._plan_status_done = False
        self._last_write_time = 0.0
        self._pending_discoveries: list[str] = []
        self._shells_since_write = 0
        self._write_reminders = 0
        self._read_reminders = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Turn-level check (tool_name=""): session start reminder
        if not ctx.tool_name:
            if not self._memory_list_done and not self._plan_status_done:
                if self._tool_call_count == 0 and self._read_reminders < 1:
                    self._read_reminders += 1
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
        elif ctx.tool_name == "plan_status":
            self._plan_status_done = True
        elif ctx.tool_name == "memory_write":
            self._last_write_time = time.time()
            self._pending_discoveries.clear()
            self._shells_since_write = 0

        # Session start: first 3 calls should include memory check
        if self._tool_call_count <= 3 and self._read_reminders < 2:
            if not self._memory_list_done and not self._plan_status_done:
                if ctx.tool_name not in ("memory_list", "memory_read", "plan_status"):
                    self._read_reminders += 1
                    return GuardVerdict.inject(
                        "[MemoryDiscipline] Session start — check memory and plan status "
                        "before acting to avoid repeating past efforts.",
                        reason="session_start_read",
                        category="memory_read_reminder",
                    )

        # Before experiment: check memory if haven't yet
        if ctx.tool_name == "shell" and not self._memory_list_done:
            cmd = ctx.tool_args.get("command", "")
            if any(k in cmd for k in ["python", "torchrun", "train", "timeout"]):
                self._read_reminders += 1
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
                    category="memory_read_reminder",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name or not ctx.tool_result:
            return None

        # Check if LLM acknowledged/dismissed the reminder in its response
        if ctx.assistant_text:
            self.acknowledge_from_text(ctx.assistant_text)

        if ctx.tool_name == "shell":
            self._shells_since_write += 1

        if ctx.tool_name == "memory_write":
            self._last_write_time = time.time()
            self._pending_discoveries.clear()
            self._shells_since_write = 0
            self._write_reminders = 0
            return None

        # Detect discoveries in output
        result_text = (ctx.tool_result or "")[:2000]
        found_categories = set()
        for pattern, category in _RESULT_PATTERNS:
            if pattern.search(result_text):
                found_categories.add(category)

        for cat in found_categories:
            self._pending_discoveries.append(cat)

        # Error detected -> immediate reminder (highest priority)
        # But max 2 consecutive error reminders before backing off
        if "error" in found_categories:
            if self._write_reminders < 2:
                self._write_reminders += 1
                return GuardVerdict.inject(
                    "[MemoryDiscipline] Error/failure detected. "
                    "Record what failed and why to memory so future sessions don't repeat this mistake.",
                    reason="record_failure",
                    category="memory_write_reminder",
                )

        # Accumulation: 3+ discoveries without write
        if len(self._pending_discoveries) >= 3 and self._write_reminders < 4:
            cats = list(set(self._pending_discoveries[-4:]))
            self._write_reminders += 1
            msg = (
                f"[MemoryDiscipline] {len(self._pending_discoveries)} findings since last "
                f"memory_write (types: {', '.join(cats)}). "
                f"Record key results now — include exact numbers, what worked/failed, config details."
            )
            if self._write_reminders >= 4:
                # Final reminder, then back off — don't block forever
                self._pending_discoveries.clear()
                self._write_reminders = 0
                return GuardVerdict.inject(
                    msg + "\n  FINAL REMINDER: Will stop asking after this.",
                    reason="final_accumulate_write",
                    category="memory_write_reminder",
                )
            return GuardVerdict.inject(msg, reason="accumulate_write", category="memory_write_reminder")

        # Periodic: many shells without any write
        if self._shells_since_write >= 5 and self._write_reminders < 4:
            if self._pending_discoveries:
                self._write_reminders += 1
                return GuardVerdict.inject(
                    f"[MemoryDiscipline] {self._shells_since_write} shell calls since last "
                    f"memory_write. Summarize findings before they're lost to compaction.",
                    reason="periodic_write",
                    category="memory_write_reminder",
                )

        return None

    def acknowledge_from_text(self, assistant_text: str):
        """Detect if LLM acknowledged the reminder in its response text.
        
        If LLM explains why it's not writing memory (e.g. "this is not an error",
        "normal output"), treat as acknowledged and reduce pressure.
        """
        if not assistant_text:
            return
        text_lower = assistant_text.lower()
        dismiss_patterns = [
            "这不是error", "这不是错误", "not an error", "not a failure",
            "normal output", "正常输出", "不是error", "isn't an error",
            "这是正常", "this is expected",
        ]
        if any(p in text_lower for p in dismiss_patterns):
            # LLM explicitly says it's not a real error — clear pending
            self._pending_discoveries.clear()
            self._write_reminders = 0

    def reset_turn(self):
        """Reset escalation counters to prevent cross-session dead loops.

        Knowledge state (_memory_list_done, _plan_status_done) persists,
        but reminder/block counters reset each turn.
        """
        self._read_reminders = 0
        self._write_reminders = 0
        self._shells_since_write = 0
        self._pending_discoveries.clear()
