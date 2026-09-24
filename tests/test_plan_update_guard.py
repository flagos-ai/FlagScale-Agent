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

"""Tests for PlanUpdateGuard — TIME-only stall reminder logic.

Firing is gated SOLELY by wall-clock (TIME_REMIND_SECONDS). Tool-call count is
NO LONGER a trigger: under concurrency a single wall-clock window carries many
more tool calls (poll/wait on in-flight jobs), so a count threshold fires on
legitimate parallel supervision rather than on a stall. The count is still shown
in the reminder as context. These tests lock that contract in.
"""

import re
import tempfile

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.guard.plan_update import PlanUpdateGuard
from flagscale_agent.react.guard import GuardContext


class _FakeClock:
    """Controllable monotonic clock for time-signal tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


_WINDOW = PlanUpdateGuard.TIME_REMIND_SECONDS


def _doing_guard(tmpdir, clock):
    tp = TaskPlan(tmpdir)
    tp.create("Test", ["Step 1"])
    tp.update_step(1, "doing")
    return PlanUpdateGuard(tp, time_fn=clock), tp


def _anchor(guard):
    """First counting call anchors the clock without firing."""
    return guard.check_post(GuardContext(tool_name="shell"))


def _fire(guard, clock, tool="shell"):
    """Advance past one time window and make one counting call → one reminder."""
    if guard._time_anchor is None:
        _anchor(guard)
    clock.advance(_WINDOW + 1)
    return guard.check_post(GuardContext(tool_name=tool))


class TestEscalation:
    """Every ESCALATE_AFTER-th reminder blocks; then resets and repeats."""

    def test_third_reminder_escalates_to_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            v1 = _fire(guard, clock)
            assert v1.action == "inject"
            v2 = _fire(guard, clock)
            assert v2.action == "inject"
            v3 = _fire(guard, clock)
            assert v3.action == "block"
            assert v3.reason == "repeated_stall_ignored"
            msg = v3.message
            assert "BLOCKS" in msg
            assert "_override_reason" in msg
            assert "raised" in msg.lower()
            assert "set_thinking" in msg
            assert "FALSIFIABLE prediction" in msg
            assert "oscillating around a plateau" in msg
            assert "I confirmed X is better" in msg
            low = msg.lower()
            assert "deliverable" in low
            assert "side-channel" in low or "side channel" in low
            assert "before/after" in low or "re-run" in low or "re-measure" in low
            assert "paper" in low or "scratch" in low

    def test_cycle_repeats_after_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            actions = [_fire(guard, clock).action for _ in range(6)]
            assert actions == [
                "inject", "inject", "block",
                "inject", "inject", "block",
            ]

    def test_substantive_plan_update_resets_escalation_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            assert _fire(guard, clock).action == "inject"
            assert _fire(guard, clock).action == "inject"
            # Substantive response (concrete note) → reset.
            guard.check_post(GuardContext(
                tool_name="plan_update",
                tool_args={"action": "step_doing",
                           "notes": "measured X=3, ruled out linear approx"},
            ))
            # Next reminder is the FIRST again → inject, not block.
            assert _fire(guard, clock).action == "inject"

    def test_empty_ping_does_not_reset_escalation_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            assert _fire(guard, clock).action == "inject"
            assert _fire(guard, clock).action == "inject"
            # Empty ping (no notes, non-progress action) must NOT reset.
            guard.check_post(GuardContext(
                tool_name="plan_update",
                tool_args={"action": "step_doing"},
            ))
            # Escalation preserved → next reminder still blocks.
            assert _fire(guard, clock).action == "block"


class TestEscalatedBlockClearance:
    """Once escalated, a bare note does not clear; only progress or a rebuilt
    model (thinking) does."""

    def _to_block(self, guard, clock):
        v = None
        for _ in range(3):
            v = _fire(guard, clock)
        assert v.action == "block"
        return v

    def test_bare_note_does_not_clear_escalated_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            self._to_block(guard, clock)
            # A bare retrospective note is rejected.
            v = guard.check_post(GuardContext(
                tool_name="plan_update",
                tool_args={"action": "step_doing", "notes": "I confirmed X is better"},
            ))
            assert v is not None
            assert v.action == "block"
            assert v.reason == "shallow_update_does_not_clear_escalated_block"

    def test_progress_action_clears_escalated_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, tp = _doing_guard(tmpdir, clock)
            self._to_block(guard, clock)
            v = guard.check_post(GuardContext(
                tool_name="plan_update",
                tool_args={"action": "step_done"},
            ))
            assert v is None  # cleared
            assert guard._block_pending is False

    def test_rebuilt_model_clears_escalated_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            self._to_block(guard, clock)
            v = guard.check_post(GuardContext(
                tool_name="plan_update",
                tool_args={"action": "set_thinking",
                           "thinking": "bottleneck is IO; if I batch reads the "
                                       "throughput should reach ~2x"},
            ))
            assert v is None  # cleared
            assert guard._block_pending is False


class TestReminderMessageContent:
    """Lock in the self-check reframe and escape-route guidance in the body."""

    def _first_fire(self, tmpdir):
        clock = _FakeClock()
        guard, _ = _doing_guard(tmpdir, clock)
        return _fire(guard, clock)

    def test_reminder_carries_gain_pricing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msg = self._first_fire(tmpdir).message
            assert "expected information gain" in msg
            assert "re-learn what you already know" in msg

    def test_opens_with_answer_first_selfcheck(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msg = self._first_fire(tmpdir).message.lower()
            assert "what did the last round tell you" in msg
            assert "nothing new" in msg
            # the question comes before the escape prescription
            assert msg.index("what did the last round tell you") < msg.index("downward")

    def test_demands_three_part_fact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msg = self._first_fire(tmpdir).message.lower()
            assert "three-part" in msg
            assert "(1)" in msg and "(2)" in msg and "(3)" in msg
            assert "assumption" in msg
            assert "how you tested" in msg
            assert "result" in msg or "value/output" in msg
            assert "empty answer" in msg
            assert "confirmed assumption" in msg

    def test_has_progressing_and_zero_gain_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msg = self._first_fire(tmpdir).message.lower()
            assert "progressing" in msg
            assert "information gain is zero" in msg
            assert "the real stall" in msg

    def test_lists_all_four_escape_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msg = self._first_fire(tmpdir).message.lower()
            assert "downward" in msg
            assert "upward" in msg
            assert "sideways" in msg
            assert "third escape" in msg and "oscillating" in msg
            assert "fourth escape" in msg and "budget order" in msg
            assert "never got written" in msg
            assert self._first_fire(tmpdir).reason == "possible_stall"


class TestTimeOnlySignalHint:
    """Only the TIME signal fires now, so the hint is always the thinking-layer
    remedy — the old COUNT (action-layer 'thrashing') hint must be gone."""

    def test_hint_is_time_and_mentions_concurrency_legitimacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            msg = _fire(guard, clock).message.lower()
            assert "time signal" in msg
            assert "observation" in msg
            # concurrency-aware: waiting on in-flight jobs is legitimate
            assert "in-flight" in msg or "background job" in msg
            # the old action-layer 'thrashing' remedy is gone
            assert "thrashing" not in msg
            # count appears only as displayed CONTEXT, never as a firing signal
            assert "count signal" not in msg


class TestCountNoLongerGatesFiring:
    """The core design change: tool-call count never triggers a reminder."""

    def test_many_calls_zero_time_never_fires(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            v = None
            for _ in range(200):
                v = guard.check_post(GuardContext(tool_name="shell"))
                assert v is None
            assert guard._iters_since_update == 200

    def test_no_reminder_without_active_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            clock = _FakeClock()
            guard = PlanUpdateGuard(tp, time_fn=clock)
            clock.advance(_WINDOW * 5)
            assert guard.check_post(GuardContext(tool_name="shell")) is None

    def test_time_window_fires_regardless_of_call_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            assert _anchor(guard) is None
            clock.advance(_WINDOW + 1)
            v = guard.check_post(GuardContext(tool_name="shell"))
            assert v is not None
            assert v.action == "inject"
            assert guard._iters_since_update == 2  # tiny count, time still fired
            assert "min elapsed" in v.message
            assert v.reason == "possible_stall"

    def test_removed_count_constants_are_gone(self):
        assert not hasattr(PlanUpdateGuard, "FIRST_REMIND")
        assert not hasattr(PlanUpdateGuard, "REMIND_INTERVAL")


class TestTimeSignal:
    """Wall-clock is the sole firing trigger."""

    def test_meta_tools_dont_tick_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            _anchor(guard)
            clock.advance(_WINDOW * 3)
            assert guard.check_post(GuardContext(tool_name="memory_read")) is None

    def test_time_reminder_is_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            fires = 0
            _anchor(guard)
            for _ in range(6):
                clock.advance(_WINDOW * 0.55)
                v = guard.check_post(GuardContext(tool_name="shell"))
                if v is not None:
                    fires += 1
            assert fires == 3

    def test_plan_update_resets_time_anchor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            _anchor(guard)
            clock.advance(_WINDOW - 10)
            guard.check_post(GuardContext(tool_name="plan_update"))
            clock.advance(_WINDOW - 10)
            assert guard.check_post(GuardContext(tool_name="shell")) is None
            clock.advance(20)
            assert guard.check_post(GuardContext(tool_name="shell")) is not None

    def test_displayed_minutes_are_cumulative_not_per_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            _anchor(guard)
            mins_seen = []
            for _ in range(3):
                clock.advance(_WINDOW + 10)
                v = guard.check_post(GuardContext(tool_name="shell"))
                assert v is not None
                m = re.search(r"~(\d+) min elapsed", v.message)
                assert m is not None, v.message
                mins_seen.append(int(m.group(1)))
            assert mins_seen[0] < mins_seen[1] < mins_seen[2]

    def test_stall_start_resets_on_plan_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = _doing_guard(tmpdir, clock)
            _anchor(guard)
            clock.advance(600)
            guard.check_post(GuardContext(tool_name="plan_update"))
            clock.advance(_WINDOW + 10)
            v = guard.check_post(GuardContext(tool_name="shell"))
            assert v is not None
            m = re.search(r"~(\d+) min elapsed", v.message)
            assert m is not None, v.message
            assert int(m.group(1)) < 6
