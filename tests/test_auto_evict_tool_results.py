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

"""Tests for auto-eviction of oversized tool results."""

import json

import pytest

from flagscale_agent.react.context_manager import (
    AUTO_EVICT_CHARS_THRESHOLD,
    AUTO_EVICT_PREVIEW_CHARS,
    ContextManager,
)
from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.swap_store import SwapStore


@pytest.fixture
def setup(tmp_path):
    """Create a ContextManager with real HistoryManager and SwapStore."""
    history = HistoryManager(max_context_tokens=200000)
    history.set_system_prompt("You are a test assistant.")
    swap_store = SwapStore(store_dir=str(tmp_path / "swap"))
    cm = ContextManager(history=history, swap_store=swap_store)
    return cm, history, swap_store


class TestAutoEvictSmallResults:
    """Small tool results should NOT be auto-evicted."""

    def test_small_anthropic_format(self, setup):
        cm, history, swap_store = setup
        # Append a small tool_result (Anthropic format)
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_123", "content": "small result"}
            ]
        })
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0
        # Content unchanged
        last_msg = history._messages[-1]
        assert last_msg["content"][0]["content"] == "small result"

    def test_small_openai_format(self, setup):
        cm, history, swap_store = setup
        # Append a small tool_result (OpenAI format)
        history.append({"role": "tool", "content": "small output", "tool_call_id": "tc_1"})
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0
        last_msg = history._messages[-1]
        assert last_msg["content"] == "small output"

    def test_exactly_at_threshold_not_evicted(self, setup):
        cm, history, swap_store = setup
        content = "x" * AUTO_EVICT_CHARS_THRESHOLD  # exactly at threshold
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": content}
            ]
        })
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0


class TestAutoEvictLargeResults:
    """Large tool results should be auto-evicted with preview."""

    def test_large_anthropic_format(self, setup):
        cm, history, swap_store = setup
        large_content = "A" * 50000  # well over threshold
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_big", "content": large_content}
            ]
        })
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        # Check the content was replaced with a trimmed version
        last_msg = history._messages[-1]
        new_content = last_msg["content"][0]["content"]
        assert "AUTO-TRIMMED" in new_content
        assert "50,000 chars" in new_content
        assert "recall(" in new_content
        # Preview should contain first N chars
        assert "AAA" in new_content
        # Total size should be much smaller
        assert len(new_content) < 5000

    def test_large_openai_format(self, setup):
        cm, history, swap_store = setup
        large_content = "B" * 40000
        history.append({"role": "tool", "content": large_content, "tool_call_id": "tc_big"})
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        last_msg = history._messages[-1]
        assert "AUTO-TRIMMED" in last_msg["content"]
        assert "40,000 chars" in last_msg["content"]
        assert len(last_msg["content"]) < 5000

    def test_multiple_blocks_mixed(self, setup):
        """When one block is large and another is small, only the large one is trimmed."""
        cm, history, swap_store = setup
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "small"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "C" * 50000},
            ]
        })
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        last_msg = history._messages[-1]
        # First block unchanged
        assert last_msg["content"][0]["content"] == "small"
        # Second block trimmed
        assert "AUTO-TRIMMED" in last_msg["content"][1]["content"]

    def test_preview_has_line_boundary(self, setup):
        """Preview should cut at a line boundary when possible."""
        cm, history, swap_store = setup
        # Create content with clear line structure
        lines = ["line " + str(i) for i in range(5000)]
        large_content = "\n".join(lines)
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_lines", "content": large_content}
            ]
        })
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        last_msg = history._messages[-1]
        new_content = last_msg["content"][0]["content"]
        # Preview part (after the header) should end cleanly
        assert "line 0" in new_content


class TestAutoEvictRecall:
    """Full content should be recallable after auto-eviction."""

    def test_recall_after_auto_evict(self, setup):
        cm, history, swap_store = setup
        large_content = "RECALL_ME_" * 5000  # 50K chars
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_recall", "content": large_content}
            ]
        })
        ext_idx = history._messages[-1].get("_ext_idx")
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        # The swap key is ext_idx * 1000 + block_index
        swap_key = ext_idx * 1000 + 0
        recalled = swap_store.load(swap_key)
        assert recalled == large_content

    def test_recall_openai_format(self, setup):
        cm, history, swap_store = setup
        large_content = "OAI_RECALL_" * 4000
        history.append({"role": "tool", "content": large_content, "tool_call_id": "tc_r"})
        ext_idx = history._messages[-1].get("_ext_idx")
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 1

        swap_key = ext_idx * 1000
        recalled = swap_store.load(swap_key)
        assert recalled == large_content

    def test_metadata_saved(self, setup):
        cm, history, swap_store = setup
        large_content = "M" * 35000
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_meta", "content": large_content}
            ]
        })
        ext_idx = history._messages[-1].get("_ext_idx")
        cm.auto_evict_oversized_results()

        swap_key = ext_idx * 1000 + 0
        meta = swap_store.load_metadata(swap_key)
        assert meta is not None
        assert meta["original_chars"] == 35000
        assert meta["tool_use_id"] == "tool_meta"


class TestAutoEvictEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_history(self, setup):
        cm, history, swap_store = setup
        # Only system prompt, no tool results
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0

    def test_evicted_message_skipped(self, setup):
        cm, history, swap_store = setup
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_ev", "content": "D" * 50000}
            ]
        })
        # Mark as already evicted
        history._messages[-1]["_evicted"] = True
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0

    def test_non_tool_result_not_affected(self, setup):
        cm, history, swap_store = setup
        # A regular user message with large content
        history.append({"role": "user", "content": "E" * 50000})
        slimmed = cm.auto_evict_oversized_results()
        assert slimmed == 0

    def test_custom_threshold(self, setup):
        cm, history, swap_store = setup
        content = "F" * 5000  # Below default but above custom threshold
        history.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_custom", "content": content}
            ]
        })
        # With default threshold, not evicted
        slimmed = cm.auto_evict_oversized_results(threshold=AUTO_EVICT_CHARS_THRESHOLD)
        assert slimmed == 0

        # With custom low threshold, evicted
        # Reset content first
        history._messages[-1]["content"][0]["content"] = content
        slimmed = cm.auto_evict_oversized_results(threshold=1000)
        assert slimmed == 1
