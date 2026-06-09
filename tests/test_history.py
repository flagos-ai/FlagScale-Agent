"""Tests for HistoryManager."""

from flagscale_agent.react.history import (
    HistoryManager, _estimate_tokens, _message_tokens,
    _is_tool_result, _has_tool_use, _collect_droppable,
    _smart_truncate, _age_message, _age_tool_results,
)


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_short(self):
        assert _estimate_tokens("hello") >= 1

    def test_proportional(self):
        short = _estimate_tokens("a" * 100)
        long = _estimate_tokens("a" * 1000)
        assert long > short

    def test_cjk_higher_than_ascii(self):
        ascii_text = "a" * 100
        cjk_text = "你" * 100
        assert _estimate_tokens(cjk_text) > _estimate_tokens(ascii_text)

    def test_cjk_chars_counted_as_1_5_tokens(self):
        cjk_text = "你好世界"
        tokens = _estimate_tokens(cjk_text)
        assert tokens >= 6  # 4 chars * 1.5 = 6

    def test_mixed_cjk_ascii(self):
        text = "Hello 你好 World 世界"
        tokens = _estimate_tokens(text)
        ascii_only = "Hello  World "
        cjk_only = "你好世界"
        assert tokens >= int(len(cjk_only) * 1.5) + len(ascii_only) // 4

    def test_japanese_counted(self):
        text = "こんにちは"
        tokens = _estimate_tokens(text)
        assert tokens >= 7  # 5 * 1.5 = 7.5

    def test_korean_counted(self):
        text = "안녕하세요"
        tokens = _estimate_tokens(text)
        assert tokens >= 7  # 5 * 1.5 = 7.5


class TestHelpers:
    def test_is_tool_result_openai(self):
        assert _is_tool_result({"role": "tool", "content": "result"})

    def test_is_tool_result_anthropic(self):
        msg = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]}
        assert _is_tool_result(msg)

    def test_is_tool_result_normal_user(self):
        assert not _is_tool_result({"role": "user", "content": "hello"})

    def test_has_tool_use_openai(self):
        msg = {"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}]}
        assert _has_tool_use(msg)

    def test_has_tool_use_anthropic(self):
        msg = {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "shell", "input": {}}]}
        assert _has_tool_use(msg)

    def test_has_tool_use_text_only(self):
        msg = {"role": "assistant", "content": "just text"}
        assert not _has_tool_use(msg)


class TestHistoryManager:
    def test_append_and_get(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "You are helpful."})
        hm.append({"role": "user", "content": "Hi"})
        msgs = hm.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_full_log_preserved(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        assert len(hm.full_log) == 2

    def test_truncation_on_budget(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert any(m["role"] == "user" and m["content"] == "recent" for m in msgs)

    def test_compaction_flag(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 5000})
        hm.get_messages()
        assert hm.compaction_happened

    def test_no_compaction_under_budget(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        hm.get_messages()
        assert not hm.compaction_happened

    def test_summarizer_called_on_compaction(self):
        called = []
        def fake_summarizer(text):
            called.append(text)
            return "Summary: stuff happened"

        hm = HistoryManager(max_context_tokens=100)
        hm.set_summarizer(fake_summarizer)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert len(called) > 0
        # Summary should be injected (may be merged with adjacent user message)
        def _has_summary(m):
            c = m.get("content", "")
            if isinstance(c, str):
                return "<context-summary>" in c
            if isinstance(c, list):
                return any("<context-summary>" in (b.get("text", "") if isinstance(b, dict) else "") for b in c)
            return False
        summary_msgs = [m for m in msgs if _has_summary(m)]
        assert len(summary_msgs) == 1

    def test_orphaned_tool_result_removed(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "orphan"})
        hm.append({"role": "user", "content": "hi"})
        msgs = hm.get_messages()
        assert not any(m.get("role") == "tool" for m in msgs)

    def test_anthropic_tool_pair_preserved(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "shell", "input": {}}]})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        # Tool pair preserved; consecutive user messages merged into one
        assert len(msgs) == 3
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        # Merged user message contains both tool_result and text
        content = msgs[2]["content"]
        assert isinstance(content, list)
        assert any(b.get("type") == "tool_result" for b in content)
        assert any(b.get("type") == "text" and "recent" in b.get("text", "") for b in content)

    def test_anthropic_orphan_removed(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "orphan"}]})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert len(msgs) == 2


class TestMergeConsecutiveUserMessages:
    def test_two_string_users_merged(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": "hi"})
        hm.append({"role": "user", "content": "first"})
        hm.append({"role": "user", "content": "second"})
        msgs = hm.get_messages()
        assert len(msgs) == 3
        content = msgs[2]["content"]
        assert isinstance(content, list)
        assert any("first" in b.get("text", "") for b in content)
        assert any("second" in b.get("text", "") for b in content)

    def test_three_consecutive_users_merged(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": "hi"})
        hm.append({"role": "user", "content": "a"})
        hm.append({"role": "user", "content": "b"})
        hm.append({"role": "user", "content": "c"})
        msgs = hm.get_messages()
        assert len(msgs) == 3
        content = msgs[2]["content"]
        assert isinstance(content, list)
        assert len(content) == 3

    def test_no_merge_when_alternating(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "q1"})
        hm.append({"role": "assistant", "content": "a1"})
        hm.append({"role": "user", "content": "q2"})
        msgs = hm.get_messages()
        assert len(msgs) == 4
        assert msgs[1]["content"] == "q1"
        assert msgs[3]["content"] == "q2"

    def test_list_and_string_merged(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "sh", "input": {}}]})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]})
        hm.append({"role": "user", "content": "follow up"})
        msgs = hm.get_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert isinstance(content, list)
        assert any(b.get("type") == "tool_result" for b in content)
        assert any(b.get("type") == "text" and "follow up" in b.get("text", "") for b in content)


class TestCollectDroppable:
    def test_preserves_system(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 10000},
        ]
        _, kept = _collect_droppable(messages, budget=10)
        assert kept[0]["role"] == "system"

    def test_under_budget_no_change(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        dropped, kept = _collect_droppable(messages, budget=100000)
        assert len(kept) == 2
        assert len(dropped) == 0

    def test_drops_openai_tool_pair(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 5000},
            {"role": "user", "content": "recent"},
        ]
        dropped, kept = _collect_droppable(messages, budget=10)
        assert kept[-1]["content"] == "recent"
        assert not any(m.get("role") == "tool" for m in kept)

    def test_fallback_drops_when_still_over(self):
        """Even after truncation, if still over budget, drop old pairs."""
        hm = HistoryManager(max_context_tokens=50)
        hm.append({"role": "system", "content": "s"})
        # Old pair
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        # Recent
        hm.append({"role": "user", "content": "hi"})
        msgs = hm.get_messages()
        # Should not contain the old tool result
        assert not any(m.get("role") == "tool" and "xxxxx" in m.get("content", "") for m in msgs)


class TestSmartTruncate:
    def test_short_content_unchanged(self):
        assert _smart_truncate("hello world") == "hello world"

    def test_long_content_truncated(self):
        big = "\n".join(f"line {i}" for i in range(200))
        result = _smart_truncate(big, max_chars=300)
        assert len(result) < len(big)
        assert "omitted" in result or "truncated" in result

    def test_preserves_head_and_tail(self):
        lines = [f"line {i}" for i in range(50)]
        big = "\n".join(lines)
        result = _smart_truncate(big, max_chars=200)
        assert "line 0" in result
        assert "line 49" in result

    def test_error_tail_preserved(self):
        content = "line 1\nline 2\nline 3\n" + "x\n" * 100 + "Traceback (most recent call last):\n  File test.py\nValueError: bad"
        result = _smart_truncate(content, max_chars=300)
        assert "Traceback" in result or "ValueError" in result


class TestAgeMessage:
    def test_small_tool_result_unchanged(self):
        msg = {"role": "tool", "tool_call_id": "1", "content": "short"}
        assert _age_message(msg) is msg

    def test_large_tool_result_truncated(self):
        msg = {"role": "tool", "tool_call_id": "1", "content": "x\n" * 1000}
        result = _age_message(msg)
        assert len(result["content"]) < len(msg["content"])

    def test_user_message_unchanged(self):
        msg = {"role": "user", "content": "x" * 2000}
        assert _age_message(msg) is msg

    def test_anthropic_tool_result_truncated(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "y\n" * 1000}
        ]}
        result = _age_message(msg)
        assert len(result["content"][0]["content"]) < len(msg["content"][0]["content"])

    def test_anthropic_non_tool_block_unchanged(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "x" * 2000}
        ]}
        assert _age_message(msg) is msg


class TestAgeToolResults:
    def test_short_history_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _age_tool_results(msgs, keep_recent=10) is msgs

    def test_recent_messages_preserved(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(15):
            msgs.append({"role": "tool", "tool_call_id": str(i), "content": "x\n" * 500})
        result = _age_tool_results(msgs, keep_recent=5)
        # Last 5 should be full size
        assert result[-1]["content"] == msgs[-1]["content"]
        # Old ones should be truncated
        assert len(result[1]["content"]) < len(msgs[1]["content"])

    def test_system_messages_never_aged(self):
        msgs = [
            {"role": "system", "content": "x" * 2000},
            {"role": "tool", "tool_call_id": "1", "content": "y\n" * 500},
        ] + [{"role": "user", "content": "recent"}] * 15
        result = _age_tool_results(msgs, keep_recent=5)
        assert result[0]["content"] == msgs[0]["content"]


class TestContextPressure:
    def test_zero_when_no_limit(self):
        hm = HistoryManager(max_context_tokens=0)
        assert hm.get_context_pressure() == 0.0

    def test_low_pressure(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        pressure = hm.get_context_pressure()
        assert 0.0 < pressure < 0.1

    def test_high_pressure(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 2000})
        pressure = hm.get_context_pressure()
        assert pressure > 1.0

    def test_actual_tokens_used_when_higher(self):
        hm = HistoryManager(max_context_tokens=1000)
        hm.append({"role": "user", "content": "hi"})
        hm.report_actual_tokens(800)
        pressure = hm.get_context_pressure()
        assert pressure >= 0.8


class TestForceCompact:
    def test_no_compaction_when_under_target(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        assert hm.force_compact(target_ratio=0.50) is False

    def test_compaction_reduces_tokens(self):
        hm = HistoryManager(max_context_tokens=500)
        hm.append({"role": "system", "content": "sys"})
        for i in range(10):
            hm.append({"role": "assistant", "tool_calls": [{"id": str(i), "name": "shell"}], "content": ""})
            hm.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 2000})
        hm.append({"role": "user", "content": "recent"})
        before = sum(_message_tokens(m) for m in hm._messages)
        result = hm.force_compact(target_ratio=0.50)
        assert result is True
        after = sum(_message_tokens(m) for m in hm._messages)
        assert after < before

    def test_force_compact_preserves_recent(self):
        hm = HistoryManager(max_context_tokens=500)
        hm.append({"role": "system", "content": "sys"})
        for i in range(10):
            hm.append({"role": "assistant", "tool_calls": [{"id": str(i), "name": "shell"}], "content": ""})
            hm.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 2000})
        hm.append({"role": "user", "content": "keep this"})
        hm.force_compact(target_ratio=0.50)
        msgs = hm._messages
        assert any(m.get("content") == "keep this" for m in msgs)
