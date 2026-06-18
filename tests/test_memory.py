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
    def test_memory_write_tool_session_scope(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        tool = MemoryWriteTool(global_mem, session_mem, "sess1")
        # Default scope = session
        result = tool.execute(key="test_key", type="finding", content="test content")
        assert "Memorized" in result
        assert "[finding]" in result
        assert "[session]" in result
        # Should be in session_mem, not global_mem
        assert session_mem.get("test_key") is not None
        assert global_mem.get("test_key") is None

    def test_memory_write_tool_global_scope(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        tool = MemoryWriteTool(global_mem, session_mem, "sess1")
        result = tool.execute(key="global_key", type="finding", content="global content", scope="global")
        assert "[global]" in result
        assert global_mem.get("global_key") is not None
        assert session_mem.get("global_key") is None

    def test_memory_read_tool_session_first(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        global_mem.put("shared_key", "decision", "global version", "s1")
        session_mem.put("shared_key", "decision", "session version", "s1")
        tool = MemoryReadTool(global_mem, session_mem)
        result = tool.execute(key="shared_key")
        # session takes priority
        assert "session version" in result
        assert "[session]" in result

    def test_memory_read_tool_falls_back_to_global(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        global_mem.put("global_only", "decision", "use TP=4", "s1")
        tool = MemoryReadTool(global_mem, session_mem)
        result = tool.execute(key="global_only")
        assert "[decision]" in result
        assert "use TP=4" in result
        assert "[global]" in result

    def test_memory_read_tool_explicit_scope(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        global_mem.put("k", "finding", "global fact", "s1")
        session_mem.put("k", "finding", "session fact", "s1")
        tool = MemoryReadTool(global_mem, session_mem)
        assert "global fact" in tool.execute(key="k", scope="global")
        assert "session fact" in tool.execute(key="k", scope="session")

    def test_memory_read_tool_miss(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        tool = MemoryReadTool(global_mem, session_mem)
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


# ── New tests for per-session scope feature ────────────────────────────────

from flagscale_agent.react.tools.memory_list import MemoryListTool
from flagscale_agent.react.paths import get_session_memory_dir, get_memory_dir


class TestMemoryListToolScope:
    """MemoryListTool scope filtering and output format."""

    def _make_tools(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        tool = MemoryListTool(global_mem, session_mem)
        return global_mem, session_mem, tool

    def test_default_scope_is_session(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        session_mem.put("s1_key", "finding", "session entry", "s1")
        result = tool.execute()
        assert "session entry" in result
        assert "global entry" not in result

    def test_scope_global_shows_only_global(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        session_mem.put("s1_key", "finding", "session entry", "s1")
        result = tool.execute(scope="global")
        assert "global entry" in result
        assert "session entry" not in result

    def test_scope_all_shows_both(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        session_mem.put("s1_key", "finding", "session entry", "s1")
        result = tool.execute(scope="all")
        assert "global entry" in result
        assert "session entry" in result

    def test_scope_labels_in_output(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        session_mem.put("s1_key", "finding", "session entry", "s1")
        result = tool.execute(scope="all")
        assert "[global]" in result
        assert "[session]" in result

    def test_session_entries_before_global_in_all(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        session_mem.put("s1_key", "finding", "session entry", "s1")
        result = tool.execute(scope="all")
        session_pos = result.index("session entry")
        global_pos = result.index("global entry")
        assert session_pos < global_pos, "session entries should appear before global entries"

    def test_header_shows_correct_count_session(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "g", "s1")
        global_mem.put("g2", "finding", "g", "s1")
        session_mem.put("s1_key", "finding", "s", "s1")
        result = tool.execute(scope="session")
        assert "1/1" in result
        assert "[session]" in result

    def test_header_shows_correct_count_global(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "g", "s1")
        global_mem.put("g2", "finding", "g", "s1")
        session_mem.put("s1_key", "finding", "s", "s1")
        result = tool.execute(scope="global")
        assert "2/2" in result
        assert "[global]" in result

    def test_header_shows_combined_count_all(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "g", "s1")
        session_mem.put("s1_key", "finding", "s", "s1")
        result = tool.execute(scope="all")
        assert "2/2" in result

    def test_keyword_filter_applied_within_scope(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        session_mem.put("nccl_timeout", "finding", "nccl timeout fix", "s1")
        session_mem.put("oom_fix", "finding", "out of memory fix", "s1")
        result = tool.execute(scope="session", keyword="nccl")
        assert "nccl timeout fix" in result
        assert "out of memory fix" not in result

    def test_empty_session_returns_no_entries_message(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        global_mem.put("g1", "finding", "global entry", "s1")
        result = tool.execute(scope="session")
        assert "no memory entries found" in result

    def test_type_filter_works_with_scope(self, tmp_path):
        global_mem, session_mem, tool = self._make_tools(tmp_path)
        session_mem.put("d1", "decision", "use TP=4", "s1")
        session_mem.put("f1", "finding", "nccl issue", "s1")
        result = tool.execute(scope="session", type_filter="decision")
        assert "use TP=4" in result
        assert "nccl issue" not in result


class TestMemoryWriteSupersedeCrossScope:
    """supersedes should remove entries from either scope store."""

    def test_supersede_removes_session_entry(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        session_mem.put("old_key", "finding", "old info", "s1")
        # Verify file exists before supersede
        assert os.path.isfile(os.path.join(str(tmp_path / "session"), "old_key.yaml"))
        tool = MemoryWriteTool(global_mem, session_mem, "s1")
        result = tool.execute(
            key="new_key", type="finding", content="new info",
            supersedes=["old_key"]
        )
        assert "Superseded" in result
        assert "old_key" in result
        # File must be gone — use direct file check to avoid get() semantic fallback
        assert not os.path.isfile(os.path.join(str(tmp_path / "session"), "old_key.yaml"))

    def test_supersede_removes_global_entry(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        global_mem.put("old_global", "finding", "old global info", "s1")
        tool = MemoryWriteTool(global_mem, session_mem, "s1")
        result = tool.execute(
            key="new_key", type="finding", content="new info",
            supersedes=["old_global"], scope="session"
        )
        assert "Superseded" in result
        assert global_mem.get("old_global") is None

    def test_supersede_nonexistent_key_is_silent(self, tmp_path):
        global_mem = SessionMemory(str(tmp_path / "global"), ttl_days=7)
        session_mem = SessionMemory(str(tmp_path / "session"), ttl_days=365)
        tool = MemoryWriteTool(global_mem, session_mem, "s1")
        # Should not raise, and "Superseded" should not appear since nothing deleted
        result = tool.execute(
            key="new_key", type="finding", content="content",
            supersedes=["ghost_key"]
        )
        assert "ERROR" not in result
        assert "Superseded" not in result


class TestPaths:
    """get_session_memory_dir returns correct path structure."""

    def test_session_memory_dir_structure(self, tmp_path, monkeypatch):
        # Monkeypatch get_sessions_root to use tmp_path
        import flagscale_agent.react.paths as paths_mod
        monkeypatch.setattr(paths_mod, "get_sessions_root", lambda: str(tmp_path / "sessions"))
        result = get_session_memory_dir("abc123")
        expected = str(tmp_path / "sessions" / "abc123" / "memory")
        assert result == expected

    def test_session_memory_dir_unique_per_session(self, tmp_path, monkeypatch):
        import flagscale_agent.react.paths as paths_mod
        monkeypatch.setattr(paths_mod, "get_sessions_root", lambda: str(tmp_path / "sessions"))
        dir_a = get_session_memory_dir("session_a")
        dir_b = get_session_memory_dir("session_b")
        assert dir_a != dir_b

    def test_session_memory_dir_separate_from_global(self, tmp_path, monkeypatch):
        import flagscale_agent.react.paths as paths_mod
        monkeypatch.setattr(paths_mod, "get_dot_flagscale_root", lambda: str(tmp_path / ".flagscale"))
        monkeypatch.setattr(paths_mod, "get_sessions_root", lambda: str(tmp_path / ".flagscale" / "sessions"))
        global_dir = get_memory_dir()
        session_dir = get_session_memory_dir("abc123")
        assert not session_dir.startswith(global_dir)
