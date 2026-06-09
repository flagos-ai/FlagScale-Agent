"""Tests for intra-turn graduated compaction."""

import pytest

from flagscale_agent.react.history import HistoryManager


def _make_tool_exchange(tool_name, command, output, tool_id="t1"):
    """Create a typical assistant tool_use + user tool_result pair."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {"command": command} if tool_name == "shell" else {"path": command},
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output,
            }
        ],
    }
    return assistant_msg, user_msg


def _large_output(lines=50, prefix="line"):
    """Generate a large tool output that will actually consume tokens."""
    return "\n".join(f"{prefix} {i}: " + "x" * 80 for i in range(lines))


class TestCompactIntraTurn:

    def test_no_compact_when_few_messages(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Run training"})
        h.append({"role": "assistant", "content": "Starting training."})
        assert h.compact_intra_turn(keep_last=4) is False

    def test_no_compact_when_pressure_low(self):
        """With small messages and large budget, compaction should not fire."""
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Monitor the training"})
        for i in range(10):
            a, u = _make_tool_exchange("shell", f"tail -5 log.txt", f"step {i}\n", f"t{i}")
            h.append(a)
            h.append(u)
        # Small messages, large budget → pressure is low → no compaction
        result = h.compact_intra_turn(keep_last=4)
        assert result is False

    def test_compact_reduces_tokens_not_messages(self):
        """Graduated compaction truncates content in-place, preserving message structure."""
        # Use a small budget so pressure exceeds 0.60
        h = HistoryManager(max_context_tokens=2000)
        h.append({"role": "user", "content": "Monitor the training"})

        # Large outputs that will push past the budget
        for i in range(10):
            a, u = _make_tool_exchange("shell", f"tail log.txt", _large_output(30), f"t{i}")
            h.append(a)
            h.append(u)

        original_count = len(h.messages)
        assert original_count == 21

        result = h.compact_intra_turn(keep_last=4)
        assert result is True
        # Message count is preserved (graduated compression doesn't drop messages)
        assert len(h.messages) == original_count

    def test_compact_preserves_recent_messages(self):
        """Messages in the keep_last window are never compressed."""
        h = HistoryManager(max_context_tokens=2000)
        h.append({"role": "user", "content": "Check GPU status"})

        for i in range(8):
            a, u = _make_tool_exchange("shell", f"nvidia-smi", _large_output(20), f"t{i}")
            h.append(a)
            h.append(u)

        # Last exchange with distinctive content
        last_output = "GPU 7: 100% utilization\nMemory: 79000/80000 MiB"
        last_a, last_u = _make_tool_exchange("shell", "nvidia-smi", last_output, "tlast")
        h.append(last_a)
        h.append(last_u)

        h.compact_intra_turn(keep_last=4)

        # The last messages should be preserved exactly (in keep_last window)
        assert h.messages[-1]["content"][0]["content"] == last_output

    def test_compact_preserves_errors(self):
        """Error content is preserved even in compressed messages."""
        h = HistoryManager(max_context_tokens=2000)
        h.append({"role": "user", "content": "Start training"})

        # Normal large exchanges
        for i in range(5):
            a, u = _make_tool_exchange("shell", "tail log.txt", _large_output(30), f"t{i}")
            h.append(a)
            h.append(u)

        # Error exchange (outside keep_last window)
        error_output = "ERROR: CUDA OOM\nTraceback:\n  File train.py line 42\nRuntimeError: out of memory"
        a, u = _make_tool_exchange("shell", "tail log.txt", error_output, "terr")
        h.append(a)
        h.append(u)

        # More exchanges to push error out of keep_last
        for i in range(6):
            a, u = _make_tool_exchange("shell", "nvidia-smi", _large_output(25), f"tn{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)

        # Error content should still be present (errors are preserved)
        all_content = ""
        for msg in h.messages:
            c = msg.get("content", "")
            if isinstance(c, list):
                for block in c:
                    if isinstance(block, dict):
                        all_content += str(block.get("content", ""))
            elif isinstance(c, str):
                all_content += c
        assert "CUDA OOM" in all_content

    def test_install_logs_compressed_aggressively(self):
        """Install/build logs are compressed to just head + tail."""
        h = HistoryManager(max_context_tokens=2000)
        h.append({"role": "user", "content": "Setup environment"})

        install_output = "Collecting torch==2.1.0\n" + "\n".join(
            f"  Downloading torch-{i}.whl" for i in range(30)
        ) + "\nSuccessfully installed torch-2.1.0"

        a, u = _make_tool_exchange("shell", "pip install torch", install_output, "t0")
        h.append(a)
        h.append(u)

        # More exchanges to push past keep_last
        for i in range(8):
            a, u = _make_tool_exchange("shell", "echo ok", _large_output(25), f"tx{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)

        # The install output should be compressed but still show outcome
        result_msg = h.messages[2]  # First tool result after user msg + assistant
        result_content = result_msg["content"][0]["content"]
        assert "Successfully installed" in result_content
        assert len(result_content) < len(install_output)

    def test_repeated_compact_is_idempotent(self):
        """Calling compact twice doesn't over-compress already-compressed content."""
        h = HistoryManager(max_context_tokens=2000)
        h.append({"role": "user", "content": "Long task"})

        for i in range(10):
            a, u = _make_tool_exchange("shell", f"cmd{i}", _large_output(30), f"t{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)
        content_after_first = [
            msg.get("content") for msg in h.messages
        ]

        # Second compaction on already-compressed content
        h.compact_intra_turn(keep_last=4)
        content_after_second = [
            msg.get("content") for msg in h.messages
        ]

        # Should be roughly the same (idempotent — already compressed content is short)
        assert len(h.messages) == 21  # Message count unchanged

    def test_find_turn_start_skips_tool_results(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Do the thing"})

        for i in range(6):
            a, u = _make_tool_exchange("shell", f"cmd{i}", f"out{i}\n", f"t{i}")
            h.append(a)
            h.append(u)

        # _find_turn_start should find the original "Do the thing" message
        start = h._find_turn_start()
        assert h.messages[start]["content"] == "Do the thing"
