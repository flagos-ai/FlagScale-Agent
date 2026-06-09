"""Tests for session persistence."""

import json
import os

import pytest

from flagscale_agent.react.session import (
    save_conversation, load_conversation, list_sessions,
    find_resumable_sessions, get_session_dir, mark_completed,
)


class TestSession:
    def test_save_and_load(self, tmp_path):
        session_dir = str(tmp_path / "session_test1")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        path = save_conversation(session_dir, "test1", msgs)
        assert os.path.isfile(path)
        assert path.endswith("conversation.json")

        data = load_conversation(session_dir)
        assert data["session_id"] == "test1"
        assert len(data["messages"]) == 3
        assert data["messages"][1]["content"] == "Hello"

    def test_save_with_metadata(self, tmp_path):
        session_dir = str(tmp_path / "session_meta")
        msgs = [{"role": "user", "content": "test"}]
        save_conversation(session_dir, "meta1", msgs, metadata={"model": "gpt-4o"})
        data = load_conversation(session_dir)
        assert data["metadata"]["model"] == "gpt-4o"

    def test_list_sessions(self, tmp_path):
        s1_dir = str(tmp_path / "s1")
        s2_dir = str(tmp_path / "s2")
        save_conversation(s1_dir, "s1", [{"role": "user", "content": "a"}])
        save_conversation(s2_dir, "s2", [{"role": "user", "content": "b"}])
        sessions = list_sessions(sessions_root=str(tmp_path))
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert "s1" in ids
        assert "s2" in ids

    def test_list_sessions_empty(self, tmp_path):
        sessions = list_sessions(sessions_root=str(tmp_path))
        assert sessions == []

    def test_list_sessions_nonexistent_dir(self):
        sessions = list_sessions(sessions_root="/nonexistent/path/xyz")
        assert sessions == []

    def test_list_sessions_counts_user_turns(self, tmp_path):
        session_dir = str(tmp_path / "multi")
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        save_conversation(session_dir, "multi", msgs)
        sessions = list_sessions(sessions_root=str(tmp_path))
        assert sessions[0]["turns"] == 2

    def test_find_resumable_sessions(self, tmp_path):
        session_dir = str(tmp_path / "abc123")
        msgs = [{"role": "user", "content": "hi"}]
        save_conversation(session_dir, "abc123", msgs, completed=False)
        results = find_resumable_sessions(sessions_root=str(tmp_path))
        assert len(results) == 1
        assert results[0]["session_id"] == "abc123"

    def test_find_resumable_skips_completed(self, tmp_path):
        session_dir = str(tmp_path / "done1")
        msgs = [{"role": "user", "content": "hi"}]
        save_conversation(session_dir, "done1", msgs, completed=True)
        results = find_resumable_sessions(sessions_root=str(tmp_path))
        assert len(results) == 0

    def test_mark_completed(self, tmp_path):
        session_dir = str(tmp_path / "to_complete")
        msgs = [{"role": "user", "content": "hi"}]
        save_conversation(session_dir, "to_complete", msgs, completed=False)
        mark_completed(session_dir)
        data = load_conversation(session_dir)
        assert data["completed"] is True

    def test_get_session_dir(self, monkeypatch):
        monkeypatch.setattr("flagscale_agent.react.session._sessions_root", lambda: "/tmp/test_sessions")
        assert get_session_dir("abc") == "/tmp/test_sessions/abc"
