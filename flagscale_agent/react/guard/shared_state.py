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

"""SharedState — global state shared across all guards.

Provides:
1. TaskMode: analysis/implementation/debugging/porting — affects all guard thresholds
2. Centralized read/write counters to prevent duplicate counting across guards
3. Inject deduplication tracking
4. Guard effectiveness tracking (how often injects change agent behavior)

Usage:
  - GuardRegistry owns the SharedState instance
  - Each guard reads from SharedState instead of maintaining private counters
  - Agent sets TaskMode based on user intent (detected via LLM or explicit)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskMode(Enum):
    """Task type that controls guard thresholds globally."""
    IMPLEMENTATION = "implementation"  # Writing code, configs, editing files
    ANALYSIS = "analysis"             # Reading, understanding, reporting
    DEBUGGING = "debugging"           # Diagnosing failures, fixing bugs
    PORTING = "porting"               # Model porting (deep reading expected)
    WORKER = "worker"                 # Subtask execution (tight budget)

    @property
    def read_tolerance(self) -> float:
        """Multiplier for read-related thresholds. Higher = more reads allowed."""
        return {
            TaskMode.IMPLEMENTATION: 1.0,
            TaskMode.ANALYSIS: 2.5,
            TaskMode.DEBUGGING: 1.5,
            TaskMode.PORTING: 2.0,
            TaskMode.WORKER: 0.5,
        }[self]

    @property
    def plan_required_threshold(self) -> int:
        """Number of exploratory calls before plan is required."""
        return {
            TaskMode.IMPLEMENTATION: 12,
            TaskMode.ANALYSIS: 25,
            TaskMode.DEBUGGING: 15,
            TaskMode.PORTING: 20,
            TaskMode.WORKER: 6,
        }[self]

    @property
    def loop_sensitivity(self) -> float:
        """Multiplier for loop detection thresholds. Higher = less sensitive."""
        return {
            TaskMode.IMPLEMENTATION: 1.0,
            TaskMode.ANALYSIS: 2.0,
            TaskMode.DEBUGGING: 1.2,
            TaskMode.PORTING: 1.8,
            TaskMode.WORKER: 0.8,
        }[self]


@dataclass
class ReadStats:
    """Centralized read tracking — single source of truth for all guards."""
    consecutive_reads: int = 0
    total_reads_session: int = 0
    unique_files_read: set = field(default_factory=set)
    reads_since_new_file: int = 0
    continuation_reads: int = 0  # Same file, different line range

    # Recent read targets for diversity calculation
    recent_read_targets: deque = field(default_factory=lambda: deque(maxlen=20))

    def record_read(self, tool_name: str, args: dict):
        """Record a read-only tool call."""
        self.consecutive_reads += 1
        self.total_reads_session += 1

        if tool_name == "read_file":
            path = args.get("path", "") or args.get("file_path", "")
            start_line = args.get("start_line")
            if path:
                target = f"{path}:{start_line or 0}"
                self.recent_read_targets.append(target)

                if path not in self.unique_files_read:
                    self.unique_files_read.add(path)
                    self.reads_since_new_file = 0
                else:
                    self.reads_since_new_file += 1
                    # Check if this is a continuation read (same file, different offset)
                    if start_line and start_line > 1:
                        self.continuation_reads += 1
        elif tool_name == "shell":
            cmd = args.get("command", "")[:80]
            self.recent_read_targets.append(f"shell:{cmd}")
        else:
            self.recent_read_targets.append(f"{tool_name}")

    def record_productive(self):
        """Reset counters on productive action."""
        self.consecutive_reads = 0
        self.reads_since_new_file = 0
        self.continuation_reads = 0

    @property
    def diversity(self) -> float:
        """How diverse are recent read targets. 1.0 = all unique, 0.0 = all same."""
        if not self.recent_read_targets:
            return 1.0
        unique = len(set(self.recent_read_targets))
        return unique / len(self.recent_read_targets)

    @property
    def is_continuation_heavy(self) -> bool:
        """Are most recent reads continuations of the same file?"""
        if self.consecutive_reads == 0:
            return False
        return self.continuation_reads / max(self.consecutive_reads, 1) >= 0.5


@dataclass
class InjectTracker:
    """Track inject effectiveness to avoid repeating ineffective warnings."""

    # (guard_name, category) -> list of (turn_injected, was_effective)
    _history: dict = field(default_factory=dict)
    # Recent inject categories in this turn (for deduplication)
    _current_turn_categories: set = field(default_factory=set)
    _turn_count: int = 0

    def record_inject(self, guard_name: str, category: str, turn: int):
        """Record that a guard injected a message."""
        key = (guard_name, category)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append({"turn": turn, "effective": None})
        self._current_turn_categories.add(category)

    def effectiveness_rate(self, guard_name: str, category: str) -> float:
        """Get effectiveness rate for a guard+category. Returns 1.0 if no data."""
        key = (guard_name, category)
        history = self._history.get(key, [])
        rated = [h for h in history if h["effective"] is not None]
        if not rated:
            return 1.0  # assume effective if no data
        return sum(1 for h in rated if h["effective"]) / len(rated)

    def consecutive_ineffective(self, guard_name: str, category: str) -> int:
        """Count consecutive ineffective injects (most recent)."""
        key = (guard_name, category)
        history = self._history.get(key, [])
        count = 0
        for entry in reversed(history):
            if entry["effective"] is False:
                count += 1
            else:
                break
        return count

    def should_suppress(self, guard_name: str, category: str) -> bool:
        """Check if this guard's inject should be suppressed due to repeated ineffectiveness.

        If a guard has injected 3+ times in a row without changing behavior,
        it should escalate rather than repeat the same message.
        """
        return self.consecutive_ineffective(guard_name, category) >= 3

    def get_escalation_message(self, guard_name: str, category: str) -> str:
        """Get escalation message when suppress threshold is reached."""
        rate = self.effectiveness_rate(guard_name, category)
        count = self.consecutive_ineffective(guard_name, category)
        return (
            f"[Guard Escalation] {guard_name}/{category} has been ineffective "
            f"{count} times in a row (overall rate: {rate:.0%}). "
            f"The current approach is not working. Fundamentally change strategy or ask the user."
        )

    def new_turn(self):
        """Reset per-turn tracking."""
        self._current_turn_categories.clear()
        self._turn_count += 1


class SharedState:
    """Singleton-ish shared state owned by GuardRegistry, accessed by all guards."""

    def __init__(self):
        self.task_mode: TaskMode = TaskMode.IMPLEMENTATION
        self.read_stats: ReadStats = ReadStats()
        self.inject_tracker: InjectTracker = InjectTracker()

        # Guard suppression: if one guard already warned about reads this iteration,
        # suppress similar warnings from other guards in the same iteration.
        # (Reset in new_iteration(), so each iteration allows one new warning.)
        self._read_warning_issued_this_turn: bool = False

        # Override audit log: tracks all accepted overrides for transparency
        self._override_log: list[dict] = []

    def set_task_mode(self, mode: TaskMode):
        """Set the current task mode. Affects all guard thresholds."""
        self.task_mode = mode

    @property
    def read_warning_issued_this_turn(self) -> bool:
        return self._read_warning_issued_this_turn

    def record_tool_call(self, tool_name: str, args: dict, is_read_only: bool):
        """Called by GuardRegistry after each tool execution.

        Also evaluates effectiveness of recent injects:
        - If a read-stall inject was followed by a productive action → effective
        - If a loop inject was followed by a different tool call → effective
        """
        if is_read_only:
            self.read_stats.record_read(tool_name, args)
            # Mark any recent "read_stall" / "loop" injects as ineffective
            # (agent is still reading after being warned)
            self._mark_recent_injects_ineffective(["read_stall", "loop"])
        else:
            self.read_stats.record_productive()
            # Agent changed behavior — mark recent read-related injects as effective
            self._mark_recent_injects_effective(["read_stall", "loop", "plan_needed"])

    def _mark_recent_injects_effective(self, categories: list[str]):
        """Mark pending injects in given categories as effective."""
        for key, history in self.inject_tracker._history.items():
            _, cat = key
            if cat in categories and history:
                last = history[-1]
                if last["effective"] is None:
                    last["effective"] = True

    def _mark_recent_injects_ineffective(self, categories: list[str]):
        """Mark pending injects in given categories as ineffective (behavior didn't change)."""
        for key, history in self.inject_tracker._history.items():
            _, cat = key
            if cat in categories and history:
                last = history[-1]
                # Only mark ineffective after 2 consecutive reads post-inject
                if last["effective"] is None and self.read_stats.consecutive_reads >= 2:
                    last["effective"] = False

    def issue_read_warning(self) -> bool:
        """Try to issue a read warning. Returns False if one was already issued this turn."""
        if self._read_warning_issued_this_turn:
            return False
        self._read_warning_issued_this_turn = True
        return True

    def new_iteration(self):
        """Reset per-iteration state.

        Called at the start of each iteration (LLM+tool loop) within a turn.
        Only resets flags that should allow re-firing within the same turn.
        """
        self._read_warning_issued_this_turn = False
        self.inject_tracker.new_turn()

    # Backward compat alias
    new_turn = new_iteration

    def record_override(self, guard_name: str, reason: str):
        """Record an accepted override for audit purposes."""
        self._override_log.append({
            "guard": guard_name,
            "reason": reason,
            "turn": self.read_stats.total_reads_session,
        })

    @property
    def override_log(self) -> list[dict]:
        """Read-only access to override history."""
        return self._override_log
