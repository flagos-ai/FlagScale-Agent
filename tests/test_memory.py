"""Tests for the session memory system."""

import os
import time

import pytest

from flagscale_agent.react.memory import SessionMemory
from flagscale_agent.react.tools.memory_write import MemoryWriteTool
from flagscale_agent.react.tools.memory_read import MemoryReadTool


@pytest.fixture
def memory_dir(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def memory(memory_dir):
    return SessionMemory(memory_dir, ttl_days=7)


class TestSessionMemory:
    def test_put_and_get(self, memory):
        memory.put("k1", "finding", "TP=8 causes OOM", "sess1")
        entry = memory.get("k1")
        assert entry is not None
        assert entry["key"] == "k1"
        assert entry["type"] == "finding"
        assert entry["content"] == "TP=8 causes OOM"
        assert entry["session_id"] == "sess1"

    def test_get_missing(self, memory):
        assert memory.get("nonexistent") is None

    def test_put_overwrites(self, memory):
        memory.put("k1", "finding", "old content", "sess1")
        memory.put("k1", "decision", "new content", "sess2")
        entry = memory.get("k1")
        assert entry["type"] == "decision"
        assert entry["content"] == "new content"
        assert entry["session_id"] == "sess2"

    def test_delete(self, memory):
        memory.put("k1", "finding", "content", "sess1")
        assert memory.delete("k1") is True
        assert memory.get("k1") is None
        assert memory.delete("k1") is False

    def test_list_entries(self, memory):
        memory.put("a", "finding", "fact a", "s1")
        memory.put("b", "decision", "choice b", "s1")
        entries = memory.list_entries()
        assert len(entries) == 2
        keys = {e["key"] for e in entries}
        assert keys == {"a", "b"}

    def test_list_entries_empty(self, memory):
        assert memory.list_entries() == []

    def test_clear(self, memory):
        memory.put("a", "finding", "x", "s1")
        memory.put("b", "todo", "y", "s1")
        count = memory.clear()
        assert count == 2
        assert memory.list_entries() == []

    def test_clear_by_type(self, memory):
        memory.put("alpha_env", "finding", "python version is 3.12 with cuda 12.4", "s1")
        memory.put("beta_ctx", "context", "user prefers verbose output", "s1")
        memory.put("gamma_perf", "finding", "transformer engine requires flash attention", "s1")
        memory.put("delta_task", "todo", "implement checkpoint conversion", "s1")
        count = memory.clear_by_type("finding")
        assert count == 2
        remaining = memory.list_entries()
        assert len(remaining) == 2
        remaining_types = {e["type"] for e in remaining}
        assert "finding" not in remaining_types

    def test_clear_by_type_returns_zero_for_unknown(self, memory):
        memory.put("a", "finding", "fact", "s1")
        count = memory.clear_by_type("nonexistent")
        assert count == 0
        assert len(memory.list_entries()) == 1

    def test_ttl_expiry(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=0)
        memory.put("k1", "finding", "content", "s1")
        time.sleep(0.1)
        assert memory.get("k1") is None

    def test_recent_returns_newest_first(self, memory):
        memory.put("env_python_version", "finding", "python 3.12 installed with cuda 12.4", "s1")
        time.sleep(0.05)
        memory.put("megatron_tp_config", "finding", "tensor parallel size must be divisible by attention heads", "s1")
        entries = memory.recent(max_tokens=1000)
        assert len(entries) == 2
        assert entries[0]["key"] == "megatron_tp_config"
        assert entries[1]["key"] == "env_python_version"

    def test_recent_respects_budget(self, memory):
        memory.put("large_entry", "finding", "x" * 2000, "s1")
        time.sleep(0.05)
        memory.put("small_entry", "finding", "short", "s1")
        entries = memory.recent(max_tokens=100)
        assert len(entries) == 1
        assert entries[0]["key"] == "small_entry"

    def test_key_with_special_chars(self, memory):
        memory.put("my/key with spaces", "context", "content", "s1")
        entry = memory.get("my/key with spaces")
        assert entry is not None
        assert entry["content"] == "content"


class TestMemoryTools:
    def test_memory_write_tool(self, memory):
        tool = MemoryWriteTool(memory, "sess1")
        result = tool.execute(key="test", type="finding", content="test content")
        assert "Memorized" in result
        assert "[finding]" in result
        entry = memory.get("test")
        assert entry is not None
        assert entry["content"] == "test content"

    def test_memory_read_tool_hit(self, memory):
        memory.put("test", "decision", "use TP=4", "s1")
        tool = MemoryReadTool(memory)
        result = tool.execute(key="test")
        assert "[decision]" in result
        assert "use TP=4" in result

    def test_memory_read_tool_miss(self, memory):
        tool = MemoryReadTool(memory)
        result = tool.execute(key="nonexistent")
        assert "No memory found" in result


class TestAccessTracking:
    """Tests for access frequency tracking and auto-promotion."""

    def test_access_count_increments(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("k1", "finding", "some finding", "s1")
        memory.get("k1")
        memory.get("k1")
        entry = memory.get("k1")
        assert entry["access_count"] == 3

    def test_auto_promotion_to_high(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("k1", "finding", "important finding", "s1")
        # Access 3 times to trigger promotion
        memory.get("k1")
        memory.get("k1")
        entry = memory.get("k1")
        assert entry["priority"] == "high"

    def test_no_promotion_for_already_high(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("k1", "finding", "content", "s1", priority="high")
        memory.get("k1")
        memory.get("k1")
        entry = memory.get("k1")
        assert entry["priority"] == "high"

    def test_query_relevant_tracks_access(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("oom_fix", "finding", "OOM fixed by reducing batch", "s1")
        memory.query_relevant(["oom"])
        memory.query_relevant(["oom"])
        memory.query_relevant(["oom"])
        entry = memory.get("oom_fix")
        # 3 from query_relevant + 1 from get
        assert entry["access_count"] >= 3


class TestDedup:
    """Tests for write-time deduplication."""

    def test_no_dedup_without_llm(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("k1", "finding", "OOM on TP=2", "s1")
        memory.put("k2", "finding", "Out of memory on TP=2", "s1")
        # Without LLM, both entries should exist
        assert memory.get("k1") is not None
        assert memory.get("k2") is not None

    def test_dedup_merges_with_llm(self, memory_dir):
        def mock_llm(prompt):
            if "Which existing entries should be merged" in prompt:
                return "[0]"
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("k1", "finding", "OOM on TP=2, fixed by batch_size=4", "s1")
        path = memory.put("k2", "finding", "OOM on TP=2, reduced batch to 4", "s1")
        # k1 should be merged into k2 (old absorbed into new)
        assert "k2" in path
        assert memory.get("k1") is None
        entry = memory.get("k2")
        assert "fixed by batch_size=4" in entry["content"]

    def test_dedup_no_merge_when_different(self, memory_dir):
        def mock_llm(prompt):
            if "Which existing entries should be merged" in prompt:
                return "[]"
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("k1", "finding", "OOM on TP=2", "s1")
        memory.put("k2", "finding", "NCCL timeout on PP=4", "s1")
        assert memory.get("k1") is not None
        assert memory.get("k2") is not None

    def test_dedup_low_confidence_no_merge(self, memory_dir):
        def mock_llm(prompt):
            if "confidence score" in prompt:
                return "0.5"  # Below threshold of 0.7
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("k1", "finding", "OOM on TP=2", "s1")
        memory.put("k2", "finding", "OOM on TP=4 with different config", "s1")
        # Both should exist since confidence is below threshold
        assert memory.get("k1") is not None
        assert memory.get("k2") is not None


class TestKeywordExpansion:
    """Tests for semantic keyword expansion."""

    def test_no_expansion_without_llm(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=7)
        memory.put("oom_fix", "finding", "out of memory fixed by reducing batch", "s1")
        # Without LLM, "cuda_malloc" won't match "out of memory"
        results = memory.query_relevant(["cuda_malloc"])
        assert len(results) == 0

    def test_expansion_with_llm(self, memory_dir):
        import json

        def mock_llm(prompt):
            if "Expand these" in prompt:
                return json.dumps({"OOM": ["oom", "out of memory", "memory exhaustion"]})
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("fix1", "finding", "out of memory fixed by reducing batch", "s1")
        results = memory.query_relevant(["OOM"])
        assert len(results) == 1
        assert results[0]["key"] == "fix1"

    def test_expansion_fallback_on_error(self, memory_dir):
        def mock_llm(prompt):
            raise RuntimeError("LLM unavailable")

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("oom_fix", "finding", "oom fixed", "s1")
        # Should fall back to original keywords
        results = memory.query_relevant(["oom"])
        assert len(results) == 1

    def test_expansion_cache_avoids_repeated_calls(self, memory_dir):
        import json
        call_count = [0]

        def mock_llm(prompt):
            if "Expand these" in prompt:
                call_count[0] += 1
                return json.dumps({"oom": ["oom", "out of memory"]})
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("fix1", "finding", "out of memory error", "s1")
        # First call — hits LLM
        memory.query_relevant(["oom"])
        assert call_count[0] == 1
        # Second call — should use cache
        memory.query_relevant(["oom"])
        assert call_count[0] == 1

    def test_expansion_cache_order_independent(self, memory_dir):
        import json
        call_count = [0]

        def mock_llm(prompt):
            if "Expand these" in prompt:
                call_count[0] += 1
                return json.dumps({"oom": ["oom", "out of memory"], "nccl": ["nccl", "nccl timeout"]})
            return ""

        memory = SessionMemory(memory_dir, ttl_days=7, llm_fn=mock_llm)
        memory.put("fix1", "finding", "out of memory and nccl timeout", "s1")
        memory.query_relevant(["oom", "nccl"])
        assert call_count[0] == 1
        # Same keywords in different order — should hit cache
        memory.query_relevant(["nccl", "oom"])
        assert call_count[0] == 1
