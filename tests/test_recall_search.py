# Copyright 2026 FlagOS Contributors
# Tests for recall_search tool — full-text search over conversation_full.json.

import json
import os
import tempfile

import pytest

from flagscale_agent.react.tools.recall_search import (
    RecallSearchTool,
    _flatten_content,
    _match_all,
)


def _write_log(session_dir, messages):
    """Write a minimal conversation_full.json with the given messages."""
    path = os.path.join(session_dir, "conversation_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "index_offset": 0,
                   "reset_count": 0, "turn_count": len(messages)}, f)
    return path


@pytest.fixture
def tool_and_dir():
    with tempfile.TemporaryDirectory() as d:
        messages = [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "user asks identity"},
                {"type": "text", "text": "I am an infrastructure agent."},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "content": "NCCL timeout on rank 3"},
            ]},
            {"role": "assistant", "content": "The NCCL timeout came from a bad NIC."},
            {"role": "user", "content": "flock and session and lock keywords here"},
        ]
        _write_log(d, messages)
        yield RecallSearchTool(d), d


# ── pure helpers ────────────────────────────────────────────────────────────

class TestFlatten:
    def test_string_passthrough(self):
        assert _flatten_content("hello") == "hello"

    def test_block_list_text_thinking(self):
        text = _flatten_content([
            {"type": "text", "text": "aaa"},
            {"type": "thinking", "thinking": "bbb"},
        ])
        assert "aaa" in text and "bbb" in text

    def test_tool_result_block_searchable(self):
        text = _flatten_content([
            {"type": "tool_result", "content": "NCCL timeout"},
        ])
        assert "NCCL timeout" in text

    def test_tool_use_block_includes_name(self):
        text = _flatten_content([
            {"type": "tool_use", "name": "shell", "input": {"command": "ls"}},
        ])
        assert "shell" in text and "ls" in text

    def test_none_and_unknown(self):
        assert _flatten_content(None) == ""
        assert "123" in _flatten_content(123)


class TestMatchAll:
    def test_and_semantics(self):
        assert _match_all("flock and session and lock", ["flock", "session"]) is True
        assert _match_all("flock only", ["flock", "session"]) is False

    def test_case_insensitive(self):
        assert _match_all("NCCL Timeout", ["nccl", "timeout"]) is True

    def test_order_independent(self):
        assert _match_all("alpha beta", ["beta", "alpha"]) is True


# ── tool behavior ───────────────────────────────────────────────────────────

class TestRecallSearchExecute:
    def test_single_keyword(self, tool_and_dir):
        tool, _ = tool_and_dir
        out = tool.execute(query="NCCL")
        assert "hit(s)" in out
        assert "[index=" in out

    def test_multi_keyword_and(self, tool_and_dir):
        tool, _ = tool_and_dir
        # "NCCL timeout" — messages 3 and 4 contain both
        out = tool.execute(query="NCCL timeout")
        assert "2 hit(s)" in out or "hit(s)" in out
        # "NCCL flock" — no message has both
        out2 = tool.execute(query="NCCL flock")
        assert "0 hit(s)" in out2
        assert "No message contains ALL keywords" in out2

    def test_index_is_position_plus_one(self, tool_and_dir):
        """message[0] → external index 1 (system prompt not in full_log)."""
        tool, _ = tool_and_dir
        out = tool.execute(query="who are you")
        assert "[index=1 " in out

    def test_role_filter(self, tool_and_dir):
        tool, _ = tool_and_dir
        out = tool.execute(query="NCCL", role="assistant")
        # only the assistant message mentioning NCCL should match
        assert "[index=4 " in out
        assert "role=user" not in out

    def test_limit(self, tool_and_dir):
        tool, _ = tool_and_dir
        out = tool.execute(query="the", limit=1)
        # exactly one hit block
        assert out.count("[index=") == 1

    def test_order_recent_vs_oldest(self, tool_and_dir):
        tool, _ = tool_and_dir
        recent = tool.execute(query="the", limit=5, order="recent")
        oldest = tool.execute(query="the", limit=5, order="oldest")
        # indices descending vs ascending
        import re
        r_idx = [int(x) for x in re.findall(r"index=(\d+)", recent)]
        o_idx = [int(x) for x in re.findall(r"index=(\d+)", oldest)]
        assert r_idx == sorted(r_idx, reverse=True)
        assert o_idx == sorted(o_idx)

    def test_empty_query_errors(self, tool_and_dir):
        tool, _ = tool_and_dir
        assert "ERROR" in tool.execute(query="")
        assert "ERROR" in tool.execute(query="   ")

    def test_missing_file(self):
        tool = RecallSearchTool("/nonexistent/path/xyz")
        out = tool.execute(query="anything")
        assert "ERROR" in out and "not found" in out

    def test_thinking_block_searchable(self, tool_and_dir):
        tool, _ = tool_and_dir
        out = tool.execute(query="identity")
        assert "[index=2 " in out

    def test_tool_result_searchable(self, tool_and_dir):
        tool, _ = tool_and_dir
        out = tool.execute(query="rank 3")
        assert "[index=3 " in out


class TestSchema:
    def test_openai_schema(self):
        tool = RecallSearchTool("/tmp/x")
        s = tool.to_openai_schema()
        assert s["function"]["name"] == "recall_search"
        props = s["function"]["parameters"]["properties"]
        assert set(["query", "role", "limit", "order"]).issubset(props.keys())

    def test_required_is_query(self):
        tool = RecallSearchTool("/tmp/x")
        assert tool.parameters["required"] == ["query"]
