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

"""Tests for HistoryManager."""

from flagscale_agent.react.history import (
    HistoryManager, _estimate_tokens, _message_tokens,
    _is_tool_result, _has_tool_use,
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

    def test_truncation_on_budget(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert any(m["role"] == "user" and m["content"] == "recent" for m in msgs)

    def test_compaction_flag(self):
        """V3: compaction no longer happens — get_messages returns messages unchanged."""
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 5000})
        msgs = hm.get_messages()
        # V3: no compaction, messages returned as-is
        assert not hm.compaction_happened
        assert any("x" * 100 in m.get("content", "") for m in msgs)

    def test_no_compaction_under_budget(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        hm.get_messages()
        assert not hm.compaction_happened

    def test_no_automatic_compaction_in_v3(self):
        """V3: no automatic compaction — messages are never modified on append."""
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        # V3: content unchanged, no summarization or truncation
        assert any("x" * 100 in m.get("content", "") for m in msgs)

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



class TestContextPressureV3:
    """V3: pressure uses 64K working window."""

    def test_low_pressure(self):
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "short message"})
        pressure = hm.get_context_pressure()
        assert 0.0 < pressure < 0.1

    def test_high_pressure(self):
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        # working_window = 200000 * 0.75 = 150000 tokens
        # Need > 0.7 * 150000 = 105000 estimated tokens
        # ASCII estimate: len / 4, so need > 420000 chars
        hm.append({"role": "user", "content": "x" * 500000})
        pressure = hm.get_context_pressure()
        assert pressure > 0.7

    def test_actual_tokens_used_when_higher(self):
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "user", "content": "hi"})
        # working_window = 150000 tokens
        # Need >= 0.7 * 150000 = 105000 actual tokens
        hm.report_actual_tokens(120000)
        pressure = hm.get_context_pressure()
        assert pressure >= 0.7


class TestGetMessagesV3:
    """V3: get_messages returns messages unchanged (no aging/compaction)."""

    def test_no_aging(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 5000})
        msgs = hm.get_messages()
        assert not hm.compaction_happened
        assert any("x" * 100 in m.get("content", "") for m in msgs)

    def test_preserves_all_content(self):
        hm = HistoryManager(max_context_tokens=50)
        hm.append({"role": "system", "content": "s"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "hi"})
        msgs = hm.get_messages()
        assert any(m.get("role") == "tool" and "xxxxx" in m.get("content", "") for m in msgs)
