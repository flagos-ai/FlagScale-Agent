# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for the resume-preview stale-summary bug.

Bug: normal per-turn saves called _save_conversation() without a summary;
session.py preserves any existing session_summary, and _check_resume only
regenerates summaries for sessions MISSING one. Result: the resume list
preview stayed frozen at the last crash-time summary while turn_count kept
advancing (user-visible as "(242 turns)" but preview showing turns [91]/[92]).

Fix: _save_conversation auto-generates the summary from the current input
history whenever none is explicitly provided.
"""

from unittest.mock import MagicMock

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.session import (
    save_conversation,
    load_conversation,
    find_resumable_sessions,
)


def _make_agent(sessions_root, input_history):
    """Minimal agent stub exposing the real _save_conversation logic.

    Uses a plain class (not MagicMock(spec=...)) — spec enforcement rejects
    instance attributes like history/turn_count that the real code reads.
    """

    class _StubAgent:
        pass

    agent = _StubAgent()
    agent.history = MagicMock()
    agent.history.messages = [{"role": "user", "content": input_history[-1]}] if input_history else []
    agent._session_dir = str(sessions_root)
    agent._session_id = "regress-1234"
    agent._loaded_skills = set()
    agent._session_input_tokens = 0
    agent._session_output_tokens = 0
    agent._session_input_history = list(input_history)
    agent.turn_count = len(input_history)
    # Bind the REAL methods under test
    agent._save_conversation = lambda completed=False, session_summary=None: (
        WorkerAgent._save_conversation(agent, completed=completed, session_summary=session_summary)
    )
    agent._generate_session_summary = lambda: WorkerAgent._generate_session_summary(agent)
    agent._save_conversation_full = MagicMock()
    return agent


class TestSaveConversationRefreshesSummary:
    """Normal saves must refresh session_summary so the resume preview stays current."""

    def test_stale_summary_refreshed_on_normal_save(self, tmp_path):
        # First save with a summary frozen at turn 92 (the bug state)
        stale = "[1] 确定是60022端口吗\n...\n[91] 旧预览\n[92] 旧预览"
        save_conversation(
            tmp_path, "regress-1234",
            [{"role": "user", "content": "旧输入"}],
            completed=False,
            session_summary=stale,
            session_input_history=["旧输入"],
            turn_count=92,
        )

        # Conversation grows: now 244 turns with new inputs
        new_inputs = ["旧输入"] + [f"turn {i}" for i in range(2, 244)] + ["中文总结一下怎么回事"]
        agent = _make_agent(tmp_path, new_inputs)

        # Normal save (completed=False, no explicit summary) — the per-turn path
        agent._save_conversation(completed=False)

        data = load_conversation(tmp_path)
        # The frozen stale summary must be replaced by a fresh one
        assert data["session_summary"] != stale
        # Tail lines must reflect the LATEST inputs, not the frozen [91]/[92]
        tail = data["session_summary"].strip().splitlines()[-2:]
        assert tail[0] == "[243] turn 243"
        assert tail[1] == "[244] 中文总结一下怎么回事"
        # turn_count stays in sync
        assert data["turn_count"] == 244

    def test_explicit_summary_still_wins(self, tmp_path):
        """Crash/exit path (L892) passes an explicit summary — must not be overridden."""
        agent = _make_agent(tmp_path, ["你好", "继续"])
        explicit = "显式摘要：异常退出时生成"
        agent._save_conversation(completed=False, session_summary=explicit)

        data = load_conversation(tmp_path)
        assert data["session_summary"] == explicit

    def test_preview_matches_history_after_fix(self, tmp_path):
        """End-to-end: find_resumable_sessions preview tail == last 2 input history entries."""
        # Sessions root expects <root>/<session_id>/conversation.json
        root = tmp_path / "root"
        sess_dir = root / "regress-1234"
        sess_dir.mkdir(parents=True)
        inputs = ["第一条", "第二条", "第三条", "最后一条"]
        agent = _make_agent(sess_dir, inputs)
        agent._save_conversation(completed=False)

        sessions = find_resumable_sessions(str(root))
        assert len(sessions) == 1
        summary_lines = sessions[0]["session_summary"].strip().splitlines()
        assert summary_lines[-2] == "[3] 第三条"
        assert summary_lines[-1] == "[4] 最后一条"
