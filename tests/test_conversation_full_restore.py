# Copyright 2026 FlagOS Contributors
# Tests for conversation_full.json persistence across /reload and hard_reset.

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def session_dir():
    """Create a temporary session directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_agent_mock(session_dir):
    """Create a minimal agent-like object with real history."""
    from flagscale_agent.react.history import HistoryManager

    agent = MagicMock()
    agent.history = HistoryManager(max_context_tokens=200000)
    agent._session_id = "test123"
    agent._session_dir = session_dir
    agent._session_lock_fd = None  # no real lock held by this fake agent
    agent._loaded_skills = set()
    agent._session_input_tokens = 0
    agent._session_output_tokens = 0
    agent.turn_count = 0
    agent._session_input_history = []
    agent.task_plan = MagicMock()
    agent.skill_manager = MagicMock()
    return agent


class TestConversationFullRestore:
    """Test that _full_log survives /reload via conversation_full.json."""

    def test_save_includes_reset_metadata(self, session_dir):
        """_save_conversation_full should save index_offset and reset_count."""
        from flagscale_agent.react.agent import WorkerAgent

        agent = _make_agent_mock(session_dir)
        agent.history._full_log = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent.history._index_offset = 42
        agent.history._reset_count = 2

        # Call the real _save_conversation_full
        WorkerAgent._save_conversation_full(agent)

        path = os.path.join(session_dir, "conversation_full.json")
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["index_offset"] == 42
        assert data["reset_count"] == 2
        assert len(data["messages"]) == 2

    def test_save_includes_turn_count_and_input_history(self, session_dir):
        """_save_conversation_full should save turn_count and session_input_history."""
        from flagscale_agent.react.agent import WorkerAgent

        agent = _make_agent_mock(session_dir)
        agent.history._full_log = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent.turn_count = 3
        agent._session_input_history = ["hello", "world", "test"]

        WorkerAgent._save_conversation_full(agent)

        path = os.path.join(session_dir, "conversation_full.json")
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["turn_count"] == 3
        assert data["session_input_history"] == ["hello", "world", "test"]

    def test_restore_seeds_full_log_from_file(self, session_dir):
        """_restore_session should seed _full_log from conversation_full.json."""
        from flagscale_agent.react.agent import WorkerAgent

        # Pre-populate conversation_full.json with 5 messages
        full_messages = [
            {"role": "user", "content": f"msg{i}", "_ext_idx": i + 1}
            for i in range(5)
        ]
        full_data = {
            "session_id": "test123",
            "messages": full_messages,
            "index_offset": 3,
            "reset_count": 1,
        }
        full_path = os.path.join(session_dir, "conversation_full.json")
        with open(full_path, "w") as f:
            json.dump(full_data, f)

        # conversation.json has only the last 2 messages (post-reset window)
        conv_data = {
            "session_id": "test123",
            "messages": [
                {"role": "user", "content": "msg3", "_ext_idx": 4},
                {"role": "user", "content": "msg4", "_ext_idx": 5},
            ],
            "loaded_skills": [],
        }

        agent = _make_agent_mock(session_dir)
        # Patch required attributes
        agent.task_plan._dir = os.path.join(session_dir, "plans")

        # Call real _restore_session
        WorkerAgent._restore_session(agent, conv_data, session_dir)

        # _full_log should have all 5 messages from conversation_full.json
        assert len(agent.history._full_log) == 5
        # _messages should have the 2 from conversation.json (the active window)
        assert len(agent.history._messages) == 2
        # Reset state should be restored
        assert agent.history._index_offset == 3
        assert agent.history._reset_count == 1

    def test_restore_without_full_file_falls_back(self, session_dir):
        """Without conversation_full.json, _restore_session appends normally."""
        from flagscale_agent.react.agent import WorkerAgent

        conv_data = {
            "session_id": "test123",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "loaded_skills": [],
        }

        agent = _make_agent_mock(session_dir)
        agent.task_plan._dir = os.path.join(session_dir, "plans")

        WorkerAgent._restore_session(agent, conv_data, session_dir)

        # Both _messages and _full_log should have 2 entries
        assert len(agent.history._messages) == 2
        assert len(agent.history._full_log) == 2
        # Reset state should be default
        assert agent.history._index_offset == 0
        assert agent.history._reset_count == 0

    def test_post_restore_append_adds_to_full_log(self, session_dir):
        """After restore with full_log seeded, new appends still grow _full_log."""
        from flagscale_agent.react.agent import WorkerAgent

        full_messages = [
            {"role": "user", "content": "old1", "_ext_idx": 1},
            {"role": "assistant", "content": "old2", "_ext_idx": 2},
        ]
        full_data = {
            "session_id": "test123",
            "messages": full_messages,
            "index_offset": 0,
            "reset_count": 0,
        }
        with open(os.path.join(session_dir, "conversation_full.json"), "w") as f:
            json.dump(full_data, f)

        conv_data = {
            "session_id": "test123",
            "messages": [
                {"role": "user", "content": "old1", "_ext_idx": 1},
            ],
            "loaded_skills": [],
        }

        agent = _make_agent_mock(session_dir)
        agent.task_plan._dir = os.path.join(session_dir, "plans")

        WorkerAgent._restore_session(agent, conv_data, session_dir)

        # Now simulate a new message arriving after restore
        agent.history.append({"role": "user", "content": "new_msg"})

        # _full_log should now have 3 entries (2 seeded + 1 new)
        assert len(agent.history._full_log) == 3
        assert agent.history._full_log[-1]["content"] == "new_msg"

    def test_restore_turn_count_from_saved_value(self, session_dir):
        """turn_count should be restored from saved turn_count, not computed."""
        from flagscale_agent.react.agent import WorkerAgent

        conv_data = {
            "session_id": "test123",
            "messages": [
                # All evicted — would produce turn_count=0 with old heuristic
                {"role": "user", "content": "[evicted | index=1 | ...]"},
                {"role": "user", "content": "[evicted | index=2 | ...]"},
            ],
            "loaded_skills": [],
            "turn_count": 42,
            "session_input_history": ["hello", "fix the bug"] + [f"input {i}" for i in range(40)],
        }

        agent = _make_agent_mock(session_dir)
        agent.task_plan._dir = os.path.join(session_dir, "plans")

        WorkerAgent._restore_session(agent, conv_data, session_dir)

        assert agent.turn_count == 42
        assert len(agent._session_input_history) == 42

    def test_restore_turn_count_fallback_to_input_history_len(self, session_dir):
        """If turn_count not saved, fall back to len(session_input_history)."""
        from flagscale_agent.react.agent import WorkerAgent

        conv_data = {
            "session_id": "test123",
            "messages": [
                {"role": "user", "content": "[evicted | index=1 | ...]"},
            ],
            "loaded_skills": [],
            # No turn_count field — old format
            "session_input_history": ["hello", "world", "test"],
        }

        agent = _make_agent_mock(session_dir)
        agent.task_plan._dir = os.path.join(session_dir, "plans")

        WorkerAgent._restore_session(agent, conv_data, session_dir)

        assert agent.turn_count == 3
        assert agent._session_input_history == ["hello", "world", "test"]
