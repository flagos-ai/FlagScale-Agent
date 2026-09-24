"""Tests for PostEvictRecoveryGuard and KnowledgeSkillGuard."""

import pytest
from unittest.mock import MagicMock
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard, EVICT_THRESHOLD
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


def _make_ctx(tool_name=None, tool_result=None, assistant_text=None, tool_args=None):
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_result = tool_result
    ctx.tool_args = tool_args or {}
    ctx.assistant_text = assistant_text
    ctx.context_pressure = 0.5
    ctx.evictable_indexes = []
    return ctx


def _advance(guard, ctx):
    """Simulate one real tool cycle: check_pre decision, then (if the call was
    not blocked) check_post persists the counter. KnowledgeSkillGuard moved the
    count increment into check_post so a blocked/not-yet-run call never inflates
    the counters — tests that accumulate calls must therefore drive BOTH phases.
    Returns the check_pre verdict so callers can assert on it.
    """
    verdict = guard.check_pre(ctx)
    # Mirror the kernel: a blocked call carries the marker in its result and does
    # NOT advance the count. Here the ctx has no such marker, so a real executed
    # call advances via check_post.
    guard.check_post(ctx)
    return verdict


# ── PostEvictRecoveryGuard ──

class TestPostEvictRecoveryGuard:
    def test_no_reminder_without_eviction(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_no_reminder_below_threshold(self):
        guard = PostEvictRecoveryGuard()
        # Evict 5 messages (below threshold of 10)
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 5 message(s), freed ~2000 tokens.")
        guard.check_post(ctx)
        
        # Next tool call should not trigger reminder
        ctx2 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx2) is None

    def test_reminder_after_heavy_eviction(self):
        guard = PostEvictRecoveryGuard()
        # Evict 15 messages (above threshold)
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 15 message(s), freed ~5000 tokens.")
        guard.check_post(ctx)
        
        # Next non-recovery tool should trigger reminder
        ctx2 = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx2)
        assert verdict is not None
        assert "evicted" in verdict.message.lower()
        assert "plan_status" in verdict.message
        assert "conversation_full.json" in verdict.message

    def test_no_reminder_for_recovery_tools(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 20 message(s), freed ~8000 tokens.")
        guard.check_post(ctx)
        
        # Recovery tools should not trigger reminder
        for tool in ["plan_status", "memory_read", "memory_list", "recall"]:
            ctx2 = _make_ctx(tool_name=tool)
            assert guard.check_pre(ctx2) is None

    def test_recovery_clears_state(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 12 message(s), freed ~4000 tokens.")
        guard.check_post(ctx)
        
        # Do recovery
        ctx2 = _make_ctx(tool_name="plan_status", tool_result="Plan: ...")
        guard.check_post(ctx2)
        
        # Next tool should NOT trigger reminder (state cleared)
        ctx3 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx3) is None

    def test_cumulative_eviction(self):
        guard = PostEvictRecoveryGuard()
        # Two small evictions that sum above threshold
        ctx1 = _make_ctx(tool_name="evict", tool_result="Evicted 6 message(s), freed ~2000 tokens.")
        guard.check_post(ctx1)
        ctx2 = _make_ctx(tool_name="evict", tool_result="Evicted 6 message(s), freed ~2000 tokens.")
        guard.check_post(ctx2)
        
        # Total = 12, above threshold
        ctx3 = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx3)
        assert verdict is not None

    def test_only_reminds_once(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 15 message(s), freed ~5000 tokens.")
        guard.check_post(ctx)
        
        # First non-recovery tool: reminder
        ctx2 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx2) is not None
        
        # Second non-recovery tool: no reminder (already reminded)
        ctx3 = _make_ctx(tool_name="read_file")
        assert guard.check_pre(ctx3) is None

    def test_empty_tool_name_safe(self):
        guard = PostEvictRecoveryGuard()
        # Trigger recovery state
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 20 message(s), freed ~8000 tokens.")
        guard.check_post(ctx)
        
        # Pre-LLM check with empty tool_name should not trigger
        ctx2 = _make_ctx(tool_name="")
        assert guard.check_pre(ctx2) is None
        
        # None tool_name should also be safe
        ctx3 = _make_ctx(tool_name=None)
        assert guard.check_pre(ctx3) is None


# ── KnowledgeSkillGuard ──

class TestKnowledgeSkillGuard:
    def test_no_reminder_initially(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_inject_after_threshold(self):
        guard = KnowledgeSkillGuard()
        # Make INJECT_THRESHOLD calls without knowledge loading
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        
        # The Nth call should trigger inject
        ctx = _make_ctx(tool_name="read_file")
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert "knowledge" in verdict.message.lower()
        # Strengthened: the inject must demand a FALSIFIABLE self-check
        # (name PROBLEM CLASS + STANDARD METHOD), not a bare "I know this".
        low = verdict.message.lower()
        assert "problem class" in low
        assert "standard method" in low
        assert "falsifiable" in low

    def test_block_after_threshold(self):
        guard = KnowledgeSkillGuard()
        # Make BLOCK_THRESHOLD calls
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        
        # The BLOCK_THRESHOLD-th call should block
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "override" in verdict.message.lower()
        # Strengthened: overriding must be FALSIFIABLE — name PROBLEM CLASS +
        # STANDARD METHOD, not a bare "solid prior experience" self-exemption.
        low = verdict.message.lower()
        assert "problem class" in low
        assert "standard method" in low
        assert "falsifiable" in low

    def test_load_knowledge_resets(self):
        guard = KnowledgeSkillGuard()
        # Accumulate some calls
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        # Load knowledge resets counter
        ctx = _make_ctx(tool_name="load_knowledge")
        guard.check_pre(ctx)
        
        # Now need another full threshold before reminder
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            assert guard.check_pre(ctx) is None

    def test_load_skill_resets(self):
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="load_skill")
        guard.check_pre(ctx)
        
        # Counter reset
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_meta_tools_dont_count(self):
        guard = KnowledgeSkillGuard()
        # Only meta tools — should never trigger
        for i in range(50):
            ctx = _make_ctx(tool_name="memory_read")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="plan_status")
        assert guard.check_pre(ctx) is None

    def test_accept_override_with_reason(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        # Should accept override with sufficient reason
        assert guard.accept_override("This task is simple file editing, no domain knowledge needed", ctx)
        # Counter should be reset after override
        assert guard._calls_since_knowledge == 0

    def test_reject_override_without_reason(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        assert not guard.accept_override("", ctx)
        assert not guard.accept_override("ok", ctx)

    def test_reset_turn_does_nothing(self):
        guard = KnowledgeSkillGuard()
        # Accumulate calls
        for i in range(5):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        assert guard._calls_since_knowledge == 5
        
        # reset_turn should NOT reset the counter
        guard.reset_turn()
        assert guard._calls_since_knowledge == 5

    def test_web_fetch_resets(self):
        """web_fetch fills an external knowledge gap → should reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)

        # web_fetch resets counter
        ctx = _make_ctx(tool_name="web_fetch")
        assert guard.check_pre(ctx) is None
        assert guard._calls_since_knowledge == 0

        # Fresh threshold needed before next reminder
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            assert guard.check_pre(ctx) is None

    def test_inject_mentions_web_fetch_and_external(self):
        """Inject text must point at BOTH internal and external channels."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "inject"
        msg = verdict.message.lower()
        assert "web_fetch" in msg
        assert "external" in msg
        # no-alternative claim is flagged as a knowledge gap
        assert "no other method" in msg or "knowledge gap" in msg
        # INTERNAL/EXTERNAL labels are product-neutral (no brand qualifier)
        assert "internal domain" in msg
        assert "flagscale" not in msg

    def test_block_mentions_web_fetch_and_no_alternative(self):
        """Block text must cover web_fetch and the no-alternative-claim trap."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "block"
        msg = verdict.message.lower()
        assert "web_fetch" in msg
        assert "no better" in msg or "no other method" in msg

    def test_block_internal_external_wording_symmetric(self):
        """INTERNAL/EXTERNAL domain labels must stay product-neutral and symmetric.

        The mechanism text describes the knowledge channel, not its contents, so
        neither label should carry a product/brand qualifier (e.g. "FlagScale").
        """
        guard = KnowledgeSkillGuard()
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        verdict = guard.check_pre(_make_ctx(tool_name="shell"))
        assert verdict is not None and verdict.action == "block"
        assert "INTERNAL domains" in verdict.message
        assert "EXTERNAL domains" in verdict.message
        assert "FlagScale" not in verdict.message



# ── KnowledgeSkillGuard information-gain inject (check_post) ──

class TestKnowledgeSkillGuardInfoGainInject:
    """check_post now returns inject (not block) for knowledge tools — aligns with
    system prompt's Information Gain section. Inject is advisory, not blocking."""

    def test_inject_when_web_fetch_succeeds(self):
        """Successful web_fetch gets info-gain inject (was None when gate disabled)."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="Here is the documentation page content..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "knowledge_skill"

    def test_inject_on_web_fetch_network_error(self):
        """Network error gets info-gain inject with fallback guidance."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_NETWORK_ERROR] Could not retrieve https://example.com: ConnectionError"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "knowledge_skill"

    def test_inject_case_insensitive_marker(self):
        """Network error marker triggers inject regardless of case."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[web_fetch_network_error] timeout"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_no_inject_for_non_network_web_fetch_errors(self):
        """Non-network errors (blocked, size exceeded) still get info-gain inject
        (they are knowledge tools returning a result)."""
        guard = KnowledgeSkillGuard()
        ctx1 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_BLOCKED] Blocked host: localhost"
        )
        v1 = guard.check_post(ctx1)
        assert v1 is not None
        assert v1.action == "inject"

        ctx2 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_SIZE_EXCEEDED] Response exceeded 5 MB limit..."
        )
        v2 = guard.check_post(ctx2)
        assert v2 is not None
        assert v2.action == "inject"

        ctx3 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_LOW_CONTENT] returned only 12 chars"
        )
        v3 = guard.check_post(ctx3)
        assert v3 is not None
        assert v3.action == "inject"

    def test_no_inject_for_non_knowledge_tools(self):
        """Info-gain inject only fires for knowledge tools, not shell/etc."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="shell",
            tool_result="curl: (7) Failed to connect"
        )
        assert guard.check_post(ctx) is None

    def test_no_inject_when_tool_result_none(self):
        """Guard handles missing tool_result gracefully."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="web_fetch", tool_result=None)
        assert guard.check_post(ctx) is None

    def test_inject_for_load_knowledge(self):
        """load_knowledge also gets info-gain inject."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="load_knowledge",
            tool_result="NCCL topology detection algorithm..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_inject_for_load_skill(self):
        """load_skill also gets info-gain inject."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="load_skill",
            tool_result="Skill content: train-run workflow..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_success_inject_presents_three_state_loop(self):
        """Successful fetch inject presents the acquired/missing/self-fillable
        three-state loop so the agent knows its next move."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="Here is the documentation page content..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        msg = verdict.message.upper()
        # All three states must be named
        assert "ACQUIRED" in msg
        assert "MISSING" in msg
        assert "SELF-FILLABLE" in msg
        assert verdict.reason == "knowledge_info_gain_three_state"
