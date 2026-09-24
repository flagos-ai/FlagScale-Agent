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

"""Unit tests for TimeBudgetGuard — the agent-side wall-clock awareness guard,
and the prompt-side urgency wording that pairs with it."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard, _fmt


def make_ctx(tool_name="shell"):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


class _Stats:
    """Mutable stats source: set .pct=None to simulate no injected wall."""

    def __init__(self):
        self.pct = 0.0

    def __call__(self):
        if self.pct is None:
            return None
        budget = 1200.0
        elapsed = budget * self.pct / 100.0
        return {
            "elapsed": elapsed,
            "budget": budget,
            "remaining": budget - elapsed,
            "pct": self.pct,
        }


class TestTimeBudgetGuardSilence:
    def test_none_stats_is_silent(self):
        s = _Stats()
        s.pct = None
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx()) is None

    def test_stats_fn_none_callable_is_silent(self):
        g = TimeBudgetGuard(stats_fn=None)
        assert g.check_post(make_ctx()) is None

    def test_stats_fn_exception_is_silent(self):
        def boom():
            raise RuntimeError("stats failed")

        g = TimeBudgetGuard(stats_fn=boom)
        # Must swallow the error, never break tool execution.
        assert g.check_post(make_ctx()) is None

    def test_no_tool_name_is_silent(self):
        s = _Stats()
        s.pct = 95.0
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx(tool_name="")) is None

    def test_below_first_threshold_is_silent(self):
        # First rung is now 25% — below it the guard stays silent.
        s = _Stats()
        s.pct = 20.0
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx()) is None


class TestTimeBudgetGuardThresholds:
    def test_fires_each_threshold_once(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)

        # 25% is the new earliest rung: pacing / front-load guidance.
        s.pct = 30.0
        v = g.check_post(make_ctx())
        assert v is not None and v.action == "inject"
        assert v.reason == "time_budget_25pct"
        # front-load / pacing language, fired while the decision window is open
        assert "front-load" in v.message or "front load" in v.message
        assert "background=true" in v.message
        # Same threshold must not refire.
        assert g.check_post(make_ctx()) is None

        s.pct = 55.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_50pct"
        # 50% is now a HEALTH CHECK, not a repeat of the pacing tip.
        assert "HEALTH CHECK" in v.message or "health check" in v.message.lower()
        # Same threshold must not refire.
        assert g.check_post(make_ctx()) is None

        s.pct = 78.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_75pct"
        assert g.check_post(make_ctx()) is None

        s.pct = 92.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_90pct"
        assert "CRITICAL" in v.message
        assert g.check_post(make_ctx()) is None

    def test_jump_past_multiple_fires_most_severe_only(self):
        # Jumping 30 -> 95 in one step should emit the 90% CRITICAL message,
        # not stack three separate advisories.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 95.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_90pct"
        # Lower thresholds are now considered spent for this turn.
        s.pct = 96.0
        assert g.check_post(make_ctx()) is None

    def test_over_100_fires_wrap_up(self):
        # At/over 100% the wall is fully spent: the most severe rung is the 100%
        # WRAP-UP inject, not the 90% advisory.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 130.0
        v = g.check_post(make_ctx())
        assert v is not None and v.action == "inject"
        assert v.reason == "time_budget_100pct"

    def test_reset_turn_rearms(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 60.0
        assert g.check_post(make_ctx()) is not None
        assert g.check_post(make_ctx()) is None
        g.reset_turn()
        assert g.check_post(make_ctx()) is not None

    def test_inject_only_never_blocks(self):
        # 25/50/75 remain inject-only in check_post
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        for p in (30.0, 55.0, 78.0):
            s.pct = p
            v = g.check_post(make_ctx())
            assert v is None or v.action == "inject"


class TestTimeBudgetGuard90Block:
    """90% threshold now blocks in check_pre rather than injecting in check_post."""

    def test_90_blocks_in_check_pre(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 92.0
        v = g.check_pre(make_ctx())
        assert v is not None
        assert v.action == "block"
        assert v.reason == "time_budget_90pct_block"
        assert v.overridable is True
        assert "CRITICAL CHECKPOINT" in v.message
        assert "deliverable" in v.message.lower()
        # Regression: 90% block must also guide pre-timeout memory extraction
        assert "memory_write()" in v.message
        assert "survival-range test" in v.message
        assert "cross-session truth" in v.message

    def test_90_block_fires_once_per_turn(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 92.0
        # First call blocks
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"
        # Second call in same turn does not re-block
        assert g.check_pre(make_ctx()) is None
        # After reset_turn, it blocks again
        g.reset_turn()
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"

    def test_90_block_suppresses_check_post_inject(self):
        # When check_pre fires and marks 90 as spent, check_post should not
        # emit a duplicate inject for the same threshold.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 92.0
        # check_pre blocks and marks 90 as fired
        v_pre = g.check_pre(make_ctx())
        assert v_pre is not None and v_pre.action == "block"
        # check_post should now be silent (90 already fired)
        v_post = g.check_post(make_ctx())
        assert v_post is None

    def test_below_90_does_not_block(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 89.0
        assert g.check_pre(make_ctx()) is None

    def test_jump_past_90_blocks_once(self):
        # Jumping 30 -> 95 should block at 90, marking all thresholds as spent.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 95.0
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"
        # check_post should not inject any lower thresholds
        assert g.check_post(make_ctx()) is None

    def test_no_tool_name_does_not_block(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 95.0
        assert g.check_pre(make_ctx(tool_name="")) is None


class TestTimeBudgetGuard100WrapUp:
    """100% threshold: the per-turn wall is spent. Inject a WRAP-UP nudge and,
    crucially, do NOT block — the agent must be free to emit its closing response
    (and NEED_USER_INPUT) without a gate in front of it."""

    def test_100_injects_wrap_up_not_block(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 100.0
        v = g.check_post(make_ctx())
        assert v is not None
        assert v.action == "inject"
        assert v.reason == "time_budget_100pct"

    def test_wrap_up_message_content(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 105.0
        v = g.check_post(make_ctx())
        assert v is not None
        # Tells the agent time is up and to hand back with NEED_USER_INPUT.
        assert "TIME IS UP" in v.message
        assert "NEED_USER_INPUT" in v.message
        # Still steers toward banking a complete deliverable first.
        assert "delivery path" in v.message

    def test_check_pre_does_not_block_at_100(self):
        # The [90, 100) block window steps aside once the wall is fully spent, so
        # the wrap-up response is never gated.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 100.0
        assert g.check_pre(make_ctx()) is None
        s.pct = 130.0
        assert g.check_pre(make_ctx()) is None

    def test_check_pre_still_blocks_just_below_100(self):
        # Guard the boundary: 99% is still inside the block window.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 99.0
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"

    def test_100_fires_once_per_turn(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 100.0
        assert g.check_post(make_ctx()) is not None
        assert g.check_post(make_ctx()) is None
        g.reset_turn()
        assert g.check_post(make_ctx()) is not None

    def test_90_block_then_100_wrap_up_sequence(self):
        # Realistic sequence within one turn: cross 90 (block in check_pre), then
        # time fully runs out (100 wrap-up inject in check_post). The 90 block
        # having fired must NOT suppress the 100 wrap-up.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 92.0
        v_pre = g.check_pre(make_ctx())
        assert v_pre is not None and v_pre.action == "block"
        # Now the wall is fully spent.
        s.pct = 101.0
        v_post = g.check_post(make_ctx())
        assert v_post is not None and v_post.reason == "time_budget_100pct"


class TestFmt:
    def test_negative_clamps_to_zero(self):
        assert _fmt(-5) == "0m00s"

    def test_hours_format(self):
        assert _fmt(3720) == "1h02m"

    def test_minutes_format(self):
        assert _fmt(65) == "1m05s"
