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

"""Tests for the redesigned memory system (three-category: fact/pitfall/insight)."""

import os

import pytest

from flagscale_agent.react.memory import Memory, VALID_TYPES


@pytest.fixture
def memory_dir(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def memory(memory_dir):
    return Memory(memory_dir)


class TestKeyValidation:
    """Key format must be type/domain/specific."""

    def test_valid_keys(self):
        assert Memory.is_valid_key("fact/cluster/ssh_port")
        assert Memory.is_valid_key("pitfall/nccl/nic_exclude_syntax")
        assert Memory.is_valid_key("insight/agent/memory_redesign")
        assert Memory.is_valid_key("fact/env/cuda_version")

    def test_invalid_keys(self):
        # Missing segments
        assert not Memory.is_valid_key("fact")
        assert not Memory.is_valid_key("fact/cluster")
        # Too many segments
        assert not Memory.is_valid_key("fact/cluster/ssh/port")
        # Invalid type prefix
        assert not Memory.is_valid_key("finding/cluster/ssh_port")
        assert not Memory.is_valid_key("context/tmp/state")
        # Uppercase
        assert not Memory.is_valid_key("fact/Cluster/ssh_port")
        # Starting with number
        assert not Memory.is_valid_key("fact/1cluster/ssh_port")

    def test_validate_key_returns_none_for_valid(self):
        assert Memory.validate_key("fact/cluster/ssh_port") is None
        assert Memory.validate_key("pitfall/nccl/nic_hang") is None

    def test_validate_key_rejects_hyphens(self):
        """Hyphens are not allowed — underscore only."""
        err = Memory.validate_key("fact/env/cuda-version")
        assert err is not None
        assert "lowercase" in err or "_" in err
        assert not Memory.is_valid_key("pitfall/nccl/nic-hang")

    def test_validate_key_returns_error_for_invalid(self):
        err = Memory.validate_key("fact")
        assert "3 segments" in err

        err = Memory.validate_key("finding/cluster/port")
        assert "fact" in err or "pitfall" in err or "insight" in err

        err = Memory.validate_key("fact/Cluster/port")
        assert "lowercase" in err


class TestMemoryPutGet:
    """Core put/get operations."""

    def test_put_and_get(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "值: 2222", "sess1")
        entry = memory.get("fact/cluster/ssh_port")
        assert entry is not None
        assert entry["key"] == "fact/cluster/ssh_port"
        assert entry["type"] == "fact"
        assert entry["content"] == "值: 2222"
        assert entry["created_session"] == "sess1"
        assert entry["updated_session"] == "sess1"

    def test_get_missing(self, memory):
        assert memory.get("fact/cluster/nonexistent") is None

    def test_put_overwrites_content(self, memory):
        memory.put("fact/env/cuda_version", "fact", "12.4", "sess1")
        memory.put("fact/env/cuda_version", "fact", "13.0", "sess2")
        entry = memory.get("fact/env/cuda_version")
        assert entry["content"] == "13.0"
        assert entry["updated_session"] == "sess2"
        # created_session preserved from original
        assert entry["created_session"] == "sess1"

    def test_put_preserves_created_fields(self, memory):
        memory.put("pitfall/nccl/nic_hang", "pitfall", "original", "sess1", task="debug")
        memory.put("pitfall/nccl/nic_hang", "pitfall", "updated", "sess2", task="")
        entry = memory.get("pitfall/nccl/nic_hang")
        assert entry["created_session"] == "sess1"
        assert entry["updated_session"] == "sess2"
        # Task from original preserved when new task is empty
        assert entry["task"] == "debug"

    def test_delete(self, memory):
        memory.put("fact/env/path", "fact", "/usr/local", "s1")
        assert memory.delete("fact/env/path") is True
        assert memory.get("fact/env/path") is None
        assert memory.delete("fact/env/path") is False

    def test_slash_in_key_stored_safely(self, memory):
        """Slashes in keys are converted to __ in filenames."""
        memory.put("fact/cluster/node_ips", "fact", "10.0.0.1", "s1")
        # Check file exists with double-underscore encoding
        expected_file = os.path.join(memory._dir, "fact__cluster__node_ips.yaml")
        assert os.path.isfile(expected_file)


class TestMemoryList:
    """Listing and filtering."""

    def test_list_all(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "2222", "s1")
        memory.put("pitfall/nccl/nic_hang", "pitfall", "hang issue", "s1")
        memory.put("insight/agent/refactor", "insight", "need refactor", "s1")
        entries = memory.list_entries()
        assert len(entries) == 3

    def test_list_empty(self, memory):
        assert memory.list_entries() == []

    def test_filter_by_type(self, memory):
        memory.put("fact/env/cuda", "fact", "12.4", "s1")
        memory.put("pitfall/nccl/hang", "pitfall", "issue", "s1")
        entries = memory.list_entries(type_filter="fact")
        assert len(entries) == 1
        assert entries[0]["type"] == "fact"

    def test_filter_by_domain(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "2222", "s1")
        memory.put("fact/env/cuda", "fact", "12.4", "s1")
        memory.put("pitfall/cluster/timeout", "pitfall", "timeout", "s1")
        entries = memory.list_entries(domain_filter="cluster")
        assert len(entries) == 2

    def test_filter_by_keyword(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "值: 2222", "s1")
        memory.put("fact/env/cuda", "fact", "CUDA 13.0", "s1")
        entries = memory.list_entries(keyword="cuda")
        assert len(entries) == 1
        assert "13.0" in entries[0]["content"]

    def test_combined_filters(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "2222", "s1")
        memory.put("pitfall/cluster/nccl_hang", "pitfall", "hang with 2222", "s1")
        entries = memory.list_entries(type_filter="fact", keyword="2222")
        assert len(entries) == 1
        assert entries[0]["key"] == "fact/cluster/ssh_port"

    def test_keyword_multi_token_and(self, memory):
        """Multi-token keyword = AND semantics across key+content."""
        memory.put("fact/cluster/ssh_port", "fact", "port 2222", "s1")
        memory.put("fact/cluster/timeout", "fact", "ssh port issue with 2222", "s1")
        memory.put("fact/env/cuda", "fact", "portable CUDA", "s1")
        # Both tokens must appear (in key or content, either order)
        entries = memory.list_entries(keyword="ssh 2222")
        assert len(entries) == 2
        keys = {e["key"] for e in entries}
        assert keys == {"fact/cluster/ssh_port", "fact/cluster/timeout"}
        # Order-insensitive: "2222 ssh" matches the same set
        assert len(memory.list_entries(keyword="2222 ssh")) == 2
        # AND is strict: a token present in only one entry excludes it
        assert len(memory.list_entries(keyword="cuda 2222")) == 0

    def test_keyword_multi_token_case_insensitive(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "Port 2222", "s1")
        entries = memory.list_entries(keyword="SSH 2222")
        assert len(entries) == 1

    def test_keyword_multi_token_extra_whitespace(self, memory):
        memory.put("fact/cluster/ssh_port", "fact", "port 2222", "s1")
        # Leading/trailing/double spaces collapse to the same token set
        assert len(memory.list_entries(keyword="  ssh   2222  ")) == 1


class TestMemoryPrefix:
    """Prefix-based listing."""

    def test_list_by_type_prefix(self, memory):
        memory.put("fact/cluster/a", "fact", "a", "s1")
        memory.put("fact/env/b", "fact", "b", "s1")
        memory.put("pitfall/nccl/c", "pitfall", "c", "s1")
        entries = memory.list_by_prefix("fact/")
        assert len(entries) == 2

    def test_list_by_domain_prefix(self, memory):
        memory.put("fact/cluster/ssh", "fact", "22", "s1")
        memory.put("fact/cluster/nodes", "fact", "4", "s1")
        memory.put("fact/env/cuda", "fact", "13", "s1")
        entries = memory.list_by_prefix("fact/cluster/")
        assert len(entries) == 2

    def test_empty_prefix_result(self, memory):
        memory.put("fact/env/cuda", "fact", "13", "s1")
        entries = memory.list_by_prefix("pitfall/")
        assert entries == []


class TestMemoryClear:
    """Clear operations."""

    def test_clear_all(self, memory):
        memory.put("fact/a/b", "fact", "x", "s1")
        memory.put("pitfall/c/d", "pitfall", "y", "s1")
        count = memory.clear()
        assert count == 2
        assert memory.list_entries() == []

    def test_clear_empty(self, memory):
        assert memory.clear() == 0


# ── Tool tests ────────────────────────────────────────────────────────────────

from flagscale_agent.react.tools.memory_write import MemoryWriteTool
from flagscale_agent.react.tools.memory_read import MemoryReadTool
from flagscale_agent.react.tools.memory_list import MemoryListTool


class TestMemoryWriteTool:
    """MemoryWriteTool validates key/type and writes entries."""

    def _make_tool(self, tmp_path):
        mem = Memory(str(tmp_path / "mem"))
        return mem, MemoryWriteTool(mem, "sess1")

    def test_write_fact(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="fact/cluster/ssh_port", type="fact", content="值: 2222"
        )
        assert "[fact]" in result
        assert "fact/cluster/ssh_port" in result
        assert mem.get("fact/cluster/ssh_port") is not None

    def test_write_pitfall(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="pitfall/nccl/nic_hang", type="pitfall",
            content="现象: hang\n原因: 9 NICs\n解决: whitelist"
        )
        assert "[pitfall]" in result
        assert mem.get("pitfall/nccl/nic_hang") is not None

    def test_write_insight(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="insight/agent/memory_redesign", type="insight",
            content="发现: need refactor\n消化方向: agent\n目标产物: new memory.py"
        )
        assert "[insight]" in result

    def test_rejects_invalid_type(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="fact/env/test", type="finding", content="test"
        )
        assert "ERROR" in result
        assert "finding" in result

    def test_rejects_invalid_key_format(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="my_old_style_key", type="fact", content="test"
        )
        assert "ERROR" in result
        assert "3 segments" in result

    def test_rejects_type_key_mismatch(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="fact/env/test", type="pitfall", content="test"
        )
        assert "ERROR" in result
        assert "does not match" in result

    def test_supersedes_deletes_old_keys(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/env/old_version", "fact", "old", "s1")
        result = tool.execute(
            key="fact/env/new_version", type="fact", content="new",
            supersedes=["fact/env/old_version"]
        )
        assert "Superseded" in result
        assert mem.get("fact/env/old_version") is None
        assert mem.get("fact/env/new_version") is not None

    def test_supersede_nonexistent_is_silent(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        result = tool.execute(
            key="fact/env/test", type="fact", content="val",
            supersedes=["fact/env/ghost"]
        )
        assert "ERROR" not in result
        assert "Superseded" not in result

    # --- semantic-uniqueness gate (list-before-write) ---

    def test_near_duplicate_key_blocked(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_0824_1056", "fact", "first", "s1")
        result = tool.execute(
            key="fact/tbench/mjcf_relaunch_0824_1139", type="fact",
            content="second"
        )
        assert result.startswith("BLOCKED")
        assert "mjcf_relaunch_0824_1056" in result
        # nothing written
        assert mem.get("fact/tbench/mjcf_relaunch_0824_1139") is None

    def test_force_new_bypasses_gate(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_0824_1056", "fact", "first", "s1")
        result = tool.execute(
            key="fact/tbench/mjcf_relaunch_0824_1139", type="fact",
            content="second", force_new=True
        )
        assert "Memorized" in result
        assert mem.get("fact/tbench/mjcf_relaunch_0824_1139") is not None

    def test_exact_key_update_not_blocked(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_0824_1056", "fact", "first", "s1")
        result = tool.execute(
            key="fact/tbench/mjcf_relaunch_0824_1056", type="fact",
            content="updated"
        )
        assert "Memorized" in result
        assert mem.get("fact/tbench/mjcf_relaunch_0824_1056")["content"] == "updated"

    def test_distinct_concept_not_blocked(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_0824_1056", "fact", "x", "s1")
        # different domain, no shared meaningful tokens
        result = tool.execute(
            key="fact/nccl/nic_exclude_syntax", type="fact", content="y"
        )
        assert "Memorized" in result

    def test_supersedes_bypasses_gate(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_0824_1056", "fact", "x", "s1")
        result = tool.execute(
            key="fact/tbench/mjcf_relaunch_0824_1307", type="fact", content="z",
            supersedes=["fact/tbench/mjcf_relaunch_0824_1056"]
        )
        assert "Memorized" in result
        assert "Superseded" in result

    def test_single_meaningful_token_not_blocked(self, tmp_path):
        # only one shared meaningful token and neither is a subset -> distinct
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/mjcf_relaunch_cmd", "fact", "x", "s1")
        result = tool.execute(
            key="fact/tbench/mjcf_task_contract", type="fact", content="y"
        )
        assert "Memorized" in result

    def test_subset_token_relationship_blocked(self, tmp_path):
        # new key's tokens are a superset of existing -> containment -> block
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/tbench/fasttext_relaunch", "fact", "x", "s1")
        result = tool.execute(
            key="fact/tbench/fasttext_relaunch_cmd", type="fact", content="y"
        )
        assert result.startswith("BLOCKED")


class TestMemoryReadTool:
    """MemoryReadTool reads by exact key or prefix."""

    def _make_tool(self, tmp_path):
        mem = Memory(str(tmp_path / "mem"))
        return mem, MemoryReadTool(mem)

    def test_read_exact_key(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/cluster/ssh_port", "fact", "值: 2222", "s1")
        result = tool.execute(key="fact/cluster/ssh_port")
        assert "2222" in result
        assert "[fact]" in result

    def test_read_missing_key(self, tmp_path):
        _, tool = self._make_tool(tmp_path)
        result = tool.execute(key="fact/cluster/nonexistent")
        assert "No memory found" in result

    def test_read_prefix(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/cluster/ssh_port", "fact", "2222", "s1")
        mem.put("fact/cluster/node_ips", "fact", "10.0.0.1", "s1")
        mem.put("fact/env/cuda", "fact", "13.0", "s1")
        result = tool.execute(key="fact/cluster/")
        assert "2 entries" in result
        assert "ssh_port" in result
        assert "node_ips" in result
        assert "cuda" not in result

    def test_read_prefix_empty(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/env/cuda", "fact", "13.0", "s1")
        result = tool.execute(key="pitfall/")
        assert "No entries found" in result


class TestMemoryListTool:
    """MemoryListTool displays entries grouped by type."""

    def _make_tool(self, tmp_path):
        mem = Memory(str(tmp_path / "mem"))
        return mem, MemoryListTool(mem)

    def test_list_all_grouped(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/env/cuda", "fact", "13.0", "s1")
        mem.put("pitfall/nccl/hang", "pitfall", "issue", "s1")
        mem.put("insight/agent/refactor", "insight", "todo", "s1")
        result = tool.execute()
        assert "3/3" in result
        assert "── fact" in result
        assert "── pitfall" in result
        assert "── insight" in result
        # fact should appear before pitfall, pitfall before insight
        assert result.index("fact") < result.index("pitfall")
        assert result.index("pitfall") < result.index("insight")

    def test_list_with_type_filter(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/env/cuda", "fact", "13.0", "s1")
        mem.put("pitfall/nccl/hang", "pitfall", "issue", "s1")
        result = tool.execute(type_filter="fact")
        assert "1/1" in result
        assert "cuda" in result
        assert "hang" not in result

    def test_list_with_domain_filter(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/cluster/ssh", "fact", "22", "s1")
        mem.put("fact/env/cuda", "fact", "13", "s1")
        result = tool.execute(domain_filter="cluster")
        assert "ssh" in result
        assert "cuda" not in result

    def test_list_with_keyword(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        mem.put("fact/env/cuda", "fact", "CUDA version 13.0", "s1")
        mem.put("fact/env/python", "fact", "Python 3.12", "s1")
        result = tool.execute(keyword="cuda")
        assert "cuda" in result
        assert "python" not in result

    def test_list_empty(self, tmp_path):
        _, tool = self._make_tool(tmp_path)
        result = tool.execute()
        assert "no memory entries found" in result

    def test_list_respects_limit(self, tmp_path):
        mem, tool = self._make_tool(tmp_path)
        for i in range(10):
            mem.put(f"fact/env/item{i}", "fact", f"value {i}", "s1")
        result = tool.execute(limit=3)
        assert "3/10" in result
