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

"""Tests for PlanUpdateGuard LLM-judge loop diagnosis.

When the stall guard escalates to a BLOCK, it consults an LLM judge to decide
whether the recent activity is a SIDEWAYS loop (edit-same-file + rerun-whole-
program) and, if so, prepends a DOWNWARD/UPWARD escape note to the block. The
judgment is delegated entirely to the judge — no signatures, counting, or regex.
"""

import tempfile

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.guard.plan_update import (
    PlanUpdateGuard,
    _extract_recent_activity,
)
from flagscale_agent.react.guard import GuardContext


class _FakeClock:
    """Controllable monotonic clock — firing is time-gated, not count-gated."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _guard_with_doing_step(tmpdir):
    tp = TaskPlan(tmpdir)
    tp.create("Test", ["Step 1"])
    tp.update_step(1, "doing")
    clock = _FakeClock()
    return PlanUpdateGuard(tp, time_fn=clock), clock


def _fire_to_block(guard, clock, messages=None, classify_fn=None):
    """Advance the clock across windows until the guard escalates to a block.

    Firing is now gated SOLELY by wall-clock (count no longer triggers), so each
    reminder needs one time window to elapse; the 3rd reminder escalates.
    """
    window = PlanUpdateGuard.TIME_REMIND_SECONDS
    v = None
    for i in range(600):
        ctx = GuardContext(
            tool_name="shell",
            turn_count=i,
            messages=messages or [],
            classify_fn=classify_fn,
        )
        v = guard.check_post(ctx)
        if v is not None and v.action == "block":
            return v
        clock.advance(window + 1)  # cross one window before the next call
    raise AssertionError("no block fired")


# ── _extract_recent_activity ────────────────────────────────────────────────

class TestExtractRecentActivity:
    def test_empty_messages_returns_empty(self):
        assert _extract_recent_activity([]) == ""
        assert _extract_recent_activity(None) == ""

    def test_collects_assistant_text_and_tool_calls_chronologically(self):
        messages = [
            {"role": "assistant", "content": "editing the interpreter",
             "tool_calls": [{"name": "edit_file", "arguments": {"path": "vm.js"}}]},
            {"role": "user", "content": "tool result"},
            {"role": "assistant", "content": "rerun",
             "tool_calls": [{"name": "shell", "arguments": {"command": "node vm.js"}}]},
        ]
        out = _extract_recent_activity(messages)
        # Chronological order preserved.
        assert out.index("editing the interpreter") < out.index("rerun")
        # Tool name + key arg surfaced.
        assert "edit_file(vm.js)" in out
        assert "shell(node vm.js)" in out

    def test_key_arg_truncated(self):
        long_cmd = "node " + "x" * 300
        messages = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "shell", "arguments": {"command": long_cmd}}]},
        ]
        out = _extract_recent_activity(messages)
        assert "…" in out
        assert len(out) < len(long_cmd) + 50

    def test_structured_content_list(self):
        messages = [
            {"role": "assistant",
             "content": [{"type": "text", "text": "hello"},
                         {"type": "text", "text": "world"}],
             "tool_calls": []},
        ]
        out = _extract_recent_activity(messages)
        assert "hello" in out and "world" in out


# ── loop diagnosis wired into the block ──────────────────────────────────────

_LOOP_MESSAGES = [
    {"role": "assistant", "content": "fix JAL offset",
     "tool_calls": [{"name": "edit_file", "arguments": {"path": "vm.js"}}]},
    {"role": "assistant", "content": "run",
     "tool_calls": [{"name": "shell", "arguments": {"command": "node vm.js"}}]},
]


class TestLoopDiagnosisInBlock:
    def test_block_appends_escape_note_when_judge_says_looping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            calls = {}

            def classify_fn(category, context, default=False):
                calls["category"] = category
                calls["activity"] = context.get("activity", "")
                return True  # judge: yes, looping

            v = _fire_to_block(guard, clock, messages=_LOOP_MESSAGES, classify_fn=classify_fn)
            # The judge was consulted with the right category + activity trace.
            assert calls["category"] == "agent_stuck_in_sideways_loop"
            assert "vm.js" in calls["activity"]
            # The block now carries the loop escape note.
            msg = v.message
            assert "LOOP" in msg
            assert "DOWNWARD" in msg and "UPWARD" in msg
            assert "smallest possible test" in msg

    def test_block_has_no_escape_note_when_judge_says_not_looping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)

            def classify_fn(category, context, default=False):
                return False  # judge: not looping

            v = _fire_to_block(guard, clock, messages=_LOOP_MESSAGES, classify_fn=classify_fn)
            # Base block body still present, but no loop escape note prepended.
            assert "BLOCKS" in v.message
            assert "The recent trace looks like a LOOP" not in v.message

    def test_block_degrades_gracefully_without_judge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            # No classify_fn (None) → behaves exactly as before, no crash.
            v = _fire_to_block(guard, clock, messages=_LOOP_MESSAGES, classify_fn=None)
            assert v.action == "block"
            assert "The recent trace looks like a LOOP" not in v.message

    def test_judge_exception_is_swallowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)

            def classify_fn(category, context, default=False):
                raise RuntimeError("judge unavailable")

            v = _fire_to_block(guard, clock, messages=_LOOP_MESSAGES, classify_fn=classify_fn)
            assert v.action == "block"
            assert "The recent trace looks like a LOOP" not in v.message

    def test_no_activity_skips_judge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            called = {"n": 0}

            def classify_fn(category, context, default=False):
                called["n"] += 1
                return True

            # Empty messages → no activity trace → judge not consulted.
            v = _fire_to_block(guard, clock, messages=[], classify_fn=classify_fn)
            assert called["n"] == 0
            assert "The recent trace looks like a LOOP" not in v.message
