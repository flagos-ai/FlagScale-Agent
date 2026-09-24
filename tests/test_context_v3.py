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

"""Tests for Context Management V3: evict/recall system.

Covers:
- SwapStore: save, load, has, delete, list, size
- HistoryManager: evict_message, recall_message, get_evictable_indexes
- ContextPressureGuard: soft/hard limits, 30% requirement, reset
- Agent integration: _handle_evict, _handle_recall
"""

import json
import os
import tempfile
import shutil
import pytest

from flagscale_agent.react.swap_store import SwapStore


# ════════════════════════════════════════════════════════════════════════════════
# SwapStore Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestSwapStore:
    """Test SwapStore local file operations."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """Basic save and load cycle."""
        content = "This is a large tool result with many lines...\n" * 100
        self.store.save(5, content, {"tool_name": "read_file", "tokens": 500})

        loaded = self.store.load(5)
        assert loaded == content

    def test_load_nonexistent(self):
        """Loading a non-existent index returns None."""
        assert self.store.load(999) is None

    def test_has(self):
        """Check if index exists."""
        assert not self.store.has(3)
        self.store.save(3, "content")
        assert self.store.has(3)

    def test_delete(self):
        """Delete removes stored content."""
        self.store.save(7, "content")
        assert self.store.has(7)
        result = self.store.delete(7)
        assert result is True
        assert not self.store.has(7)
        assert self.store.load(7) is None

    def test_delete_nonexistent(self):
        """Deleting a non-existent index returns False."""
        assert self.store.delete(999) is False

    def test_list_indexes(self):
        """List all stored indexes in sorted order."""
        self.store.save(3, "a")
        self.store.save(10, "b")
        self.store.save(1, "c")
        assert self.store.list_indexes() == [1, 3, 10]

    def test_list_indexes_empty(self):
        """Empty store returns empty list."""
        assert self.store.list_indexes() == []

    def test_size_bytes(self):
        """size_bytes returns total bytes of stored files."""
        self.store.save(1, "hello")
        self.store.save(2, "world" * 1000)
        size = self.store.size_bytes()
        assert size > 0
        # Second file should be much larger
        assert size > 5000

    def test_save_with_metadata(self):
        """Metadata is stored alongside content."""
        self.store.save(5, "content", {"tool_name": "shell", "tokens": 200})
        # Verify file structure
        path = os.path.join(self.tmpdir, "5.json")
        with open(path) as f:
            data = json.load(f)
        assert data["content"] == "content"
        assert data["metadata"]["tool_name"] == "shell"
        assert data["metadata"]["tokens"] == 200

    def test_overwrite_existing(self):
        """Saving to the same index overwrites."""
        self.store.save(5, "original")
        self.store.save(5, "updated")
        assert self.store.load(5) == "updated"

    def test_unicode_content(self):
        """Handle unicode content correctly."""
        content = "中文内容 日本語 한국어 🎉"
        self.store.save(1, content)
        assert self.store.load(1) == content

    def test_large_content(self):
        """Handle large content (simulating big tool results)."""
        content = "x" * 100_000  # 100K chars
        self.store.save(1, content)
        assert self.store.load(1) == content
        assert len(self.store.load(1)) == 100_000

    def test_store_dir_creation(self):
        """SwapStore creates its directory if it doesn't exist."""
        new_dir = os.path.join(self.tmpdir, "nested", "deep", "swap")
        store = SwapStore(new_dir)
        assert os.path.isdir(new_dir)
        store.save(1, "test")
        assert store.load(1) == "test"


# ════════════════════════════════════════════════════════════════════════════════
# HistoryManager evict/recall Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestHistoryEvictRecall:
    """Test HistoryManager evict and recall methods."""

    def setup_method(self):
        from flagscale_agent.react.history import HistoryManager
        self.hm = HistoryManager(max_context_tokens=64000)

    def _build_messages(self):
        """Build a realistic message sequence."""
        self.hm._messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Read the file for me."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/src/main.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "def main():\n    print('hello')\n" * 100,
            },
            {"role": "assistant", "content": "I can see the main.py file..."},
            {"role": "user", "content": "Now run shell command."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_002",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "ls -la /workspace"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": "total 128\ndrwxr-xr-x 5 root root 4096 ...\n" * 50,
            },
            {"role": "assistant", "content": "Here's the directory listing."},
        ]

    def test_evict_tool_result(self):
        """Evict a tool_result message."""
        self._build_messages()
        result = self.hm.evict_message(3)  # read_file result

        assert result is not None
        assert "def main():" in result["content"]
        assert result["metadata"]["tool_name"] == "read_file"
        assert "/src/main.py" in result["metadata"]["tool_input"]

        # Message should now have placeholder
        msg = self.hm._messages[3]
        assert msg["_evicted"] is True
        assert "[evicted | index=3" in msg["content"]
        assert "read_file" in msg["content"]

    def test_evict_non_tool_result_fails(self):
        """Cannot evict system messages. User/assistant are evictable."""
        self._build_messages()
        # User message — now evictable in V3
        assert self.hm.evict_message(1) is not None
        # Assistant message — now evictable in V3
        assert self.hm.evict_message(4) is not None
        # System message — still not evictable
        assert self.hm.evict_message(0) is None

    def test_evict_already_evicted_fails(self):
        """Cannot evict the same message twice."""
        self._build_messages()
        result1 = self.hm.evict_message(3)
        assert result1 is not None
        result2 = self.hm.evict_message(3)
        assert result2 is None

    def test_evict_out_of_range(self):
        """Out-of-range index returns None."""
        self._build_messages()
        assert self.hm.evict_message(-1) is None
        assert self.hm.evict_message(100) is None

    def test_recall_message(self):
        """Recall restores evicted content in-place."""
        self._build_messages()
        original_content = self.hm._messages[3]["content"]
        self.hm.evict_message(3)
        assert self.hm._messages[3]["_evicted"] is True

        success = self.hm.recall_message(3, original_content)
        assert success is True
        assert self.hm._messages[3]["content"] == original_content
        assert "_evicted" not in self.hm._messages[3]

    def test_recall_non_evicted_fails(self):
        """Cannot recall a message that wasn't evicted."""
        self._build_messages()
        assert self.hm.recall_message(3, "something") is False

    def test_get_evictable_indexes(self):
        """All non-system, non-tail messages are evictable."""
        self._build_messages()
        evictable = self.hm.get_evictable_indexes()
        assert 3 in evictable  # read_file result
        assert 0 not in evictable  # system
        # User and assistant messages are evictable in V3
        assert 1 in evictable  # user
        # Last 4 messages are protected (total 9, so 5,6,7,8 protected)
        assert 5 not in evictable
        assert 7 not in evictable

    def test_get_evictable_after_evict(self):
        """Evicted messages no longer appear as evictable."""
        self._build_messages()
        self.hm.evict_message(3)
        evictable = self.hm.get_evictable_indexes()
        assert 3 not in evictable
        # Index 1 and 2 should still be evictable (OpenAI format doesn't trigger paired eviction)
        assert 1 in evictable
        assert 2 in evictable

    def test_evict_shell_tool(self):
        """Evict shell command result with correct metadata."""
        self._build_messages()
        # Index 7 is in protected tail (9 msgs, protected from idx 5+)
        # Use index 3 (read_file) which is evictable
        result = self.hm.evict_message(3)
        assert result is not None
        assert result["metadata"]["tool_name"] == "read_file"
        assert "/src/main.py" in result["metadata"]["tool_input"]

    def test_placeholder_format(self):
        """Placeholder contains all required info."""
        self._build_messages()
        self.hm.evict_message(3)
        placeholder = self.hm._messages[3]["content"]
        assert "index=3" in placeholder
        assert "read_file" in placeholder
        assert "tokens" in placeholder


class TestEvictHighExtIdxAfterReset:
    """Regression: evicting a tool_result whose external index is far larger than
    len(_messages) must NOT raise IndexError.

    Reproduces the post-hard_reset state where _ext_idx grows unboundedly (e.g.
    17480) while _messages is short. Previously evict_message passed the external
    index into _extract_tool_info_for_index, which indexed self._messages[index]
    directly -> 'list index out of range'.
    """

    def setup_method(self):
        from flagscale_agent.react.history import HistoryManager
        self.hm = HistoryManager(max_context_tokens=64000)

    def _build_post_reset_messages(self):
        # Short _messages list, but each carries a large _ext_idx as if many
        # messages were appended then discarded by hard_reset.
        base = 17476
        self.hm._messages = [
            {"role": "system", "content": "You are an assistant.", "_ext_idx": 0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_hi",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "make -j"}),
                        },
                    }
                ],
                "_ext_idx": base + 1,
            },
            {
                "role": "tool",
                "tool_call_id": "call_hi",
                "content": "build output\n" * 200,
                "_ext_idx": base + 2,
            },
            {"role": "assistant", "content": "done", "_ext_idx": base + 3},
            {"role": "user", "content": "tail1", "_ext_idx": base + 4},
            {"role": "assistant", "content": "tail2", "_ext_idx": base + 5},
            {"role": "user", "content": "tail3", "_ext_idx": base + 6},
            {"role": "assistant", "content": "tail4", "_ext_idx": base + 7},
        ]
        # Simulate reset offset so ext->internal resolution goes through _ext_idx.
        self.hm._index_offset = base

    def test_evict_high_ext_idx_tool_result_no_crash(self):
        """Evicting the high-ext-idx tool_result pair must succeed, not raise."""
        self._build_post_reset_messages()
        base = 17476
        # The assistant tool_use at base+1 pairs with the tool_result at base+2.
        result = self.hm.evict_message(base + 1)
        assert result is not None
        # tool_name should be resolved from the paired assistant tool_use.
        # Metadata tool_name may be 'shell' or 'unknown'; the key assertion is no crash
        # and that the message is now evicted.
        assert self.hm._messages[1]["_evicted"] is True

    def test_evict_high_ext_idx_out_of_range_returns_none(self):
        """An ext_idx not present in _messages resolves to None (skipped), no crash."""
        self._build_post_reset_messages()
        # 999999 is not any message's _ext_idx and > len(_messages) -> None
        assert self.hm.evict_message(999999) is None

    def test_extract_tool_info_out_of_range_guard(self):
        """_extract_tool_info_for_index tolerates out-of-range internal positions."""
        self._build_post_reset_messages()
        assert self.hm._extract_tool_info_for_index(999999) == ("unknown", "")
        assert self.hm._extract_tool_info_for_index(-5) == ("unknown", "")

    def test_evict_high_ext_idx_tool_result_directly(self):
        """Evicting the tool_result directly exercises the L470 placeholder path.

        This is the path that passed the external index (base+2 == 17478) into
        _extract_tool_info_for_index, which then did self._messages[17478] -> crash.
        """
        self._build_post_reset_messages()
        base = 17476
        result = self.hm.evict_message(base + 2)  # the tool_result
        assert result is not None
        assert self.hm._messages[2]["_evicted"] is True
        # Placeholder is built via _extract_tool_info_for_index; tool_name resolved
        # from the preceding assistant tool_use.
        assert result["metadata"]["tool_name"] == "shell"


# ════════════════════════════════════════════════════════════════════════════════
# ContextPressureGuard Tests
# ════════════════════════════════════════════════════════════════════════════════


class MockGuardContext:
    """Minimal mock of GuardContext for testing."""

    def __init__(self, pressure: float, evictable_count: int = 60):
        self.context_pressure = pressure
        self.evictable_indexes = list(range(1, evictable_count + 1))
        self.tool_name = ""
        self.tool_args = {}


class TestContextPressureGuard:
    """Test the ContextPressureGuard with simplified block-only design."""

    def setup_method(self):
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        self.guard = ContextPressureGuard()

    def _fresh_guard(self):
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        return ContextPressureGuard()

    def test_no_action_below_80(self):
        """Below 80%, no action."""
        ctx = MockGuardContext(0.5)
        result = self.guard.check_pre(ctx)
        assert result is None

        ctx = MockGuardContext(0.79)
        result = self.guard.check_pre(ctx)
        assert result is None

    def test_evict_path_blocks(self):
        """>=80% with evictable >= 60 → block non-save tools."""
        ctx = MockGuardContext(0.82, evictable_count=80)
        ctx.tool_name = "shell"
        ctx.tool_args = {"command": "ls"}
        result = self.guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "evict" in result.message.lower()
        assert result.category == "context_pressure_evict"

    def test_evict_path_allows_save_tools(self):
        """Evict path allows memory/plan/evict tools through."""
        for tool in ["memory_write", "plan_update", "evict", "recall"]:
            guard = self._fresh_guard()
            ctx = MockGuardContext(0.85, evictable_count=70)
            ctx.tool_name = tool
            ctx.tool_args = {}
            result = guard.check_pre(ctx)
            assert result is None, f"{tool} should be allowed"

    def test_evict_path_releases_below_50(self):
        """After evicting below 50%, normal tools work again."""
        # First block
        ctx = MockGuardContext(0.82, evictable_count=80)
        ctx.tool_name = "shell"
        ctx.tool_args = {}
        result = self.guard.check_pre(ctx)
        assert result is not None and result.action == "block"

        # After evict, pressure drops to 45%
        ctx2 = MockGuardContext(0.45, evictable_count=40)
        ctx2.tool_name = "shell"
        ctx2.tool_args = {}
        result2 = self.guard.check_pre(ctx2)
        assert result2 is None

# ════════════════════════════════════════════════════════════════════════════════
# Integration: Agent evict/recall handling
# ════════════════════════════════════════════════════════════════════════════════


class TestAgentEvictRecallIntegration:
    """Test the agent-level evict/recall handling with swap store."""

    def setup_method(self):
        """Set up a minimal agent-like environment for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(os.path.join(self.tmpdir, "swap_store"))

        # Build a simple history
        from flagscale_agent.react.history import HistoryManager
        self.history = HistoryManager(max_context_tokens=64000)
        self.history._messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "read my file"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "function": {"name": "read_file", "arguments": '{"path": "/a.py"}'},
                }],
            },
            {
                "role": "tool", "tool_call_id": "tc_1",
                "content": "line1\nline2\nline3\n" * 200,
            },
            {"role": "assistant", "content": "The file has 600 lines."},
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_2",
                    "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                }],
            },
            {
                "role": "tool", "tool_call_id": "tc_2",
                "content": "file1.py\nfile2.py\n" * 50,
            },
            {"role": "assistant", "content": "Found 100 files."},
        ]

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _handle_evict(self, arguments: dict) -> str:
        """Replicate agent's _handle_evict logic."""
        indexes = arguments.get("indexes", [])
        if not indexes:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        if not isinstance(indexes, list):
            return "ERROR: 'indexes' must be a list of integers."

        evicted_count = 0
        freed_tokens = 0
        errors = []

        for idx in indexes:
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable")
                continue
            self.store.save(idx, result["content"], result.get("metadata"))
            evicted_count += 1
            freed_tokens += result.get("metadata", {}).get("tokens", 0)

        parts = [f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."]
        if errors:
            parts.append(f"Skipped: {'; '.join(errors)}")
        return " ".join(parts)

    def _handle_recall(self, arguments: dict) -> str:
        """Replicate agent's _handle_recall logic."""
        index = arguments.get("index")
        if index is None:
            return "ERROR: 'index' parameter is required."
        if not isinstance(index, int):
            return "ERROR: 'index' must be an integer."
        content = self.store.load(index)
        if content is None:
            return f"ERROR: No evicted content found at index {index}."
        return content

    def test_evict_and_recall_flow(self):
        """Full evict then recall cycle."""
        original = self.history._messages[3]["content"]

        # Evict
        result = self._handle_evict({"indexes": [3]})
        assert "Evicted 1" in result
        assert self.history._messages[3]["_evicted"] is True
        assert self.store.has(3)

        # Recall
        recalled = self._handle_recall({"index": 3})
        assert recalled == original

    def test_evict_multiple(self):
        """Evict multiple indexes at once."""
        # Add extra messages to push 3 and 7 out of protected tail
        self.history._messages.extend([
            {"role": "user", "content": "extra1"},
            {"role": "assistant", "content": "extra2"},
            {"role": "user", "content": "extra3"},
            {"role": "assistant", "content": "extra4"},
        ])
        result = self._handle_evict({"indexes": [3, 7]})
        assert "Evicted" in result
        assert self.store.has(3)
        assert self.store.has(7)

    def test_evict_mixed_valid_invalid(self):
        """Some valid, some invalid indexes."""
        # Add padding to allow more evictions
        self.history._messages.extend([
            {"role": "user", "content": "extra1"},
            {"role": "assistant", "content": "extra2"},
            {"role": "user", "content": "extra3"},
            {"role": "assistant", "content": "extra4"},
        ])
        result = self._handle_evict({"indexes": [3, 0, 99]})
        # index 0 = system (invalid), index 99 = out of range (invalid), index 3 = valid
        assert "Evicted 1" in result
        assert "Skipped" in result

    def test_evict_empty_list(self):
        """Empty indexes list returns error."""
        result = self._handle_evict({"indexes": []})
        assert "ERROR" in result

    def test_recall_nonexistent(self):
        """Recall without prior evict returns error."""
        result = self._handle_recall({"index": 99})
        assert "ERROR" in result

    def test_recall_missing_param(self):
        """Missing index parameter returns error."""
        result = self._handle_recall({})
        assert "ERROR" in result

    def test_evict_preserves_placeholder_info(self):
        """Placeholder contains tool name and input."""
        self._handle_evict({"indexes": [3]})
        placeholder = self.history._messages[3]["content"]
        assert "read_file" in placeholder
        assert "/a.py" in placeholder
        assert "index=3" in placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ════════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases that could occur in real usage."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(os.path.join(self.tmpdir, "swap"))
        from flagscale_agent.react.history import HistoryManager
        self.history = HistoryManager(max_context_tokens=64000)
        self.history._messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "/a.py"})},
                }],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "file content " * 500},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "more stuff"},
            {"role": "assistant", "content": "sure thing"},
            {"role": "user", "content": "and more"},
            {"role": "assistant", "content": "ok"},
        ]

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _handle_evict(self, arguments):
        """Replicate agent logic."""
        if not arguments:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        indexes = arguments.get("indexes", [])
        if not indexes:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        if not isinstance(indexes, list):
            if isinstance(indexes, (int, float)):
                indexes = [int(indexes)]
            else:
                return "ERROR: 'indexes' must be a list of integers."
        evicted_count = 0
        freed_tokens = 0
        errors = []
        for idx in indexes:
            if isinstance(idx, float) and idx == int(idx):
                idx = int(idx)
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable")
                continue
            content = result["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            self.store.save(idx, content, result.get("metadata"))
            evicted_count += 1
            freed_tokens += result.get("metadata", {}).get("tokens", 0)
        parts = [f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."]
        if errors:
            parts.append(f"Skipped: {'; '.join(errors)}")
        return " ".join(parts)

    def _handle_recall(self, arguments):
        if not arguments:
            return "ERROR: 'index' parameter is required."
        index = arguments.get("index")
        if index is None:
            return "ERROR: 'index' parameter is required."
        if isinstance(index, float) and index == int(index):
            index = int(index)
        if not isinstance(index, int):
            return "ERROR: 'index' must be an integer."
        content = self.store.load(index)
        if content is None:
            return f"ERROR: No evicted content found at index {index}."
        return content

    # ── Float index handling ──

    def test_evict_float_index(self):
        """Model sends 3.0 instead of 3 — should work."""
        result = self._handle_evict({"indexes": [3.0]})
        assert "Evicted 1" in result
        assert self.store.has(3)

    def test_recall_float_index(self):
        """Model sends 3.0 for recall — should work."""
        self._handle_evict({"indexes": [3]})
        result = self._handle_recall({"index": 3.0})
        assert "file content" in result

    # ── None/empty arguments ──

    def test_evict_none_arguments(self):
        """None arguments should give clear error."""
        result = self._handle_evict(None)
        assert "ERROR" in result

    def test_recall_none_arguments(self):
        """None arguments should give clear error."""
        result = self._handle_recall(None)
        assert "ERROR" in result

    def test_evict_empty_dict(self):
        """Empty dict should give clear error."""
        result = self._handle_evict({})
        assert "ERROR" in result

    # ── Single int instead of list ──

    def test_evict_single_int_not_list(self):
        """Model passes indexes=3 instead of indexes=[3]."""
        result = self._handle_evict({"indexes": 3})
        assert "Evicted 1" in result

    # ── Invalid types ──

    def test_evict_string_index(self):
        """String in index list should be skipped."""
        result = self._handle_evict({"indexes": ["three"]})
        assert "Skipped" in result
        assert "not an integer" in result

    def test_recall_string_index(self):
        """String index should error."""
        result = self._handle_recall({"index": "three"})
        assert "ERROR" in result

    def test_evict_mixed_valid_invalid(self):
        """Mix of valid and invalid indexes."""
        result = self._handle_evict({"indexes": [3, "bad", 99]})
        assert "Evicted 1" in result  # only 3 works
        assert "Skipped" in result

    # ── Double evict same index ──

    def test_double_evict_same_index(self):
        """Evicting same index twice — second time skipped."""
        r1 = self._handle_evict({"indexes": [3]})
        assert "Evicted 1" in r1
        r2 = self._handle_evict({"indexes": [3]})
        assert "Evicted 0" in r2
        assert "not evictable" in r2

    # ── Recall without prior evict ──

    def test_recall_never_evicted_index(self):
        """Recall an index that was never evicted."""
        result = self._handle_recall({"index": 3})
        assert "ERROR" in result
        assert "No evicted content" in result

    # ── Non-string content (list-type tool results) ──

    def test_save_list_content(self):
        """Some tool results might be lists (multi-part)."""
        self.store.save(99, ["part1", "part2"])
        loaded = self.store.load(99)
        assert loaded is not None
        assert "part1" in loaded
        assert "part2" in loaded

    # ── Large index numbers ──

    def test_large_index(self):
        """Very large index number (long conversation)."""
        self.store.save(99999, "content at 99999")
        assert self.store.has(99999)
        assert self.store.load(99999) == "content at 99999"

    # ── Concurrent-like save/load ──

    def test_overwrite_then_load(self):
        """Save twice to same index, load gets latest."""
        self.store.save(3, "first")
        self.store.save(3, "second")
        assert self.store.load(3) == "second"

    # ── Guard edge: exact boundary values ──

    def test_guard_at_exactly_80_percent(self):
        """Exactly 80% is the threshold boundary."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.80, evictable_count=80)
        ctx.tool_name = "shell"
        ctx.tool_args = {}
        # At exactly threshold, should trigger (>=)
        result = guard.check_pre(ctx)
        assert result is not None

    def test_guard_just_below_80(self):
        """Just below 80% should not trigger."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.799)
        result = guard.check_pre(ctx)
        assert result is None

    def test_guard_at_exactly_90(self):
        """Exactly 90% with enough evictable should block immediately."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.90, evictable_count=80)
        ctx.tool_name = "shell"
        ctx.tool_args = {}
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert result.category == "context_pressure_evict"


class TestFullLog:
    """Test that _full_log preserves original content after eviction."""

    def setup_method(self):
        from flagscale_agent.react.history import HistoryManager
        self.hm = HistoryManager(max_context_tokens=64000)

    def test_full_log_preserves_content_after_evict(self):
        """After evicting a message, _full_log still has original content."""
        self.hm.append({"role": "system", "content": "system prompt"})
        self.hm.append({"role": "user", "content": "hello world"})
        self.hm.append({"role": "assistant", "content": "hi there"})
        self.hm.append({"role": "user", "content": "more"})
        self.hm.append({"role": "assistant", "content": "response"})
        self.hm.append({"role": "user", "content": "latest 1"})
        self.hm.append({"role": "assistant", "content": "latest 2"})
        self.hm.append({"role": "user", "content": "latest 3"})
        self.hm.append({"role": "assistant", "content": "latest 4"})

        # Evict message at ext_idx 2 (the "hello world" user message)
        # Note: ext_idx starts at 1 for first appended msg; system is ext_idx=1
        result = self.hm.evict_message(2)
        assert result is not None

        # _messages[1] should now be placeholder (internal pos 1 = ext_idx 2)
        assert self.hm._messages[1].get("_evicted") is True
        assert "evicted" in self.hm._messages[1]["content"]

        # _full_log[1] should still have original content (full_log is 0-based)
        assert self.hm._full_log[1]["content"] == "hello world"
        assert self.hm._full_log[1].get("_evicted") is None

    def test_full_log_grows_unbounded(self):
        """_full_log no longer has a cap (saves to disk instead)."""
        for i in range(100):
            self.hm.append({"role": "user", "content": f"msg {i}"})
        assert len(self.hm._full_log) == 100


class TestPromptIntegrity:
    """Test that prompt modules load correctly after refactoring."""

    def test_static_prompt_format_no_keyerror(self):
        """SYSTEM_PROMPT_STATIC.format() must not raise KeyError.

        All braces in example JSON/code blocks must be escaped as {{ }}.
        Regression test for: unescaped {"command": ...} caused KeyError.
        """
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        # These are the actual placeholders used by prompt_builder.py
        try:
            result = SYSTEM_PROMPT_STATIC.format(
                cwd="/workspace/test",
                tools="shell, read_file, write_file",
                skills="train-run, train-config",
                knowledge="know-megatron-training",
                critical_rules="",
                optional_sections="",
                skill_context="",
            )
        except KeyError as e:
            raise AssertionError(
                f"SYSTEM_PROMPT_STATIC has unescaped braces causing KeyError: {e}. "
                f"Escape literal braces as {{{{ }}}} in prompt.py."
            )
        assert len(result) > 1000  # sanity: prompt is non-trivial

    def test_static_prompt_is_english(self):
        import re
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        # No Chinese characters in static prompt
        assert not re.search(r'[\u4e00-\u9fff]', SYSTEM_PROMPT_STATIC)

    def test_static_prompt_contains_planning_section(self):
        """Static prompt should contain planning guidance."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "## Plan" in SYSTEM_PROMPT_STATIC
        assert "plan_create" in SYSTEM_PROMPT_STATIC
        assert "Step Notes" in SYSTEM_PROMPT_STATIC

    def test_static_prompt_contains_memory_section(self):
        """Static prompt should contain memory rules."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "## Memory" in SYSTEM_PROMPT_STATIC
        assert "memory_read" in SYSTEM_PROMPT_STATIC
        assert "cross-session knowledge" in SYSTEM_PROMPT_STATIC

    def test_plan_gated_by_acting_not_difficulty(self):
        """Plan guidance must state plan is gated by whether you ACT, not by
        task difficulty, so 'simple' tasks are never a license to skip the plan."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "not gated by task difficulty" in SYSTEM_PROMPT_STATIC
        assert "no such thing as a task too simple to plan" in SYSTEM_PROMPT_STATIC

    def test_plan_guards_wired_to_lifecycle_rationale(self):
        """Prompt must explain WHY skipping a plan is dangerous: runtime guards
        (stall/verification) are wired to the plan lifecycle and cannot fire
        without an active plan."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "wired to the plan lifecycle" in SYSTEM_PROMPT_STATIC
        assert "With no active plan, none of them can fire" in SYSTEM_PROMPT_STATIC

    def test_method_switch_cannot_retarget_easier_goal(self):
        """Method-switch must not be usable to disguise lowering the bar /
        substituting a reachable target for the constrained one (pov-ray class)."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "change HOW you solve the problem the task set — never WHAT problem that is" in SYSTEM_PROMPT_STATIC
        assert "wearing the vocabulary of Principle 3 as a disguise" in SYSTEM_PROMPT_STATIC
        assert "never a license to retarget an easier goal" in SYSTEM_PROMPT_STATIC

    def test_verify_output_not_method_rationale(self):
        """Verification must reject arguing the method is sound in place of
        checking the output against ground truth (video-processing class)."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "arguing that your METHOD is sound in place of checking that your OUTPUT is correct" in SYSTEM_PROMPT_STATIC
        assert "A confident rationale for a wrong answer is still a wrong answer" in SYSTEM_PROMPT_STATIC

    def test_established_constraint_must_be_durable(self):
        """A load-bearing constraint must be written to memory immediately so it
        survives context eviction (build-pov-ray class: agent held the 2.2
        constraint 19min, lost it to eviction, reverted to the rejected 3.7)."""
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        assert "This list must be DURABLE, not held in working context" in SYSTEM_PROMPT_STATIC
        assert "re-inferring \"what was I even asked to do?\" from environmental scraps" in SYSTEM_PROMPT_STATIC
        assert "recover the constraint from memory before acting" in SYSTEM_PROMPT_STATIC


class TestNoUnnecessaryTruncation:
    """Regression test: ensure no truncation patterns sneak back in key display paths."""

    def test_memory_list_no_content_truncation(self):
        """memory_list tool should not truncate content."""
        import inspect
        from flagscale_agent.react.tools.memory_list import MemoryListTool
        source = inspect.getsource(MemoryListTool.execute)
        assert "[:97]" not in source
        assert "[:100]" not in source

    def test_plan_context_no_title_truncation(self):
        """Plan context rendering should not truncate titles."""
        import inspect
        from flagscale_agent.react.plan import TaskPlan
        source = inspect.getsource(TaskPlan.context_for_prompt)
        assert "[:120]" not in source
        assert "[:80]" not in source
        assert "[:40]" not in source

    def test_prompt_builder_no_desc_truncation(self):
        """Prompt builder should not truncate skill descriptions."""
        import inspect
        from flagscale_agent.react.prompt_builder import PromptBuilder
        source = inspect.getsource(PromptBuilder)
        assert "[:80]" not in source

    def test_tool_executor_no_summary_truncation(self):
        """Tool executor display summaries should not truncate."""
        import inspect
        from flagscale_agent.react.tool_executor import tool_display_summary
        source = inspect.getsource(tool_display_summary)
        assert "[:50]" not in source
        assert "[:60]" not in source
        assert "[:120]" not in source
