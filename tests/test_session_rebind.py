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

"""Regression tests: _restore_session must re-point tools that captured
constructor-bound state at registration time.

Bug: RecallSearchTool captured self._session_dir, MemoryWriteTool and
PlanCreateTool captured self._session_id, all at registration. _restore_session
replaced _session_id/_session_dir but re-registered none of them, so after a
resume/reload the tools still pointed at the ABANDONED fresh dir/id:
  - recall_search searched the empty new dir (never the restored log),
  - memory entries / plans were tagged with the wrong session id.
"""
import json
import os

import pytest


def _make_real_agent(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from flagscale_agent.react.agent import WorkerAgent
    from flagscale_agent.react.config import AgentConfig
    from flagscale_agent.react.memory import Memory
    from flagscale_agent.react.plan import TaskPlan

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
    mock_provider = Mock()
    mock_provider.count_tokens.return_value = 100
    config = AgentConfig(
        session_dir=str(tmp_path / "sessions"),
        api_key="test-key-12345",
        provider="anthropic",
        max_context_tokens=50000,
    )
    return WorkerAgent(
        config,
        _provider=mock_provider,
        _memory=Mock(spec=Memory),
        _task_plan=Mock(spec=TaskPlan),
    )


class TestRestoreRebindsTools:
    def test_restore_repoints_recall_search_and_session_id_tools(self, tmp_path, monkeypatch):
        agent = _make_real_agent(tmp_path, monkeypatch)
        fresh_id = agent._session_id
        fresh_dir = agent._session_dir

        # Precondition: tools captured the fresh dir/id.
        assert agent.tool_registry.get("recall_search")._session_dir == fresh_dir
        assert agent.tool_registry.get("memory_write")._session_id == fresh_id
        assert agent.tool_registry.get("plan_create")._session_id == fresh_id

        # A restored session on disk with a real conversation_full.json.
        restored_dir = os.path.join(agent._sessions_root, "restored1")
        os.makedirs(restored_dir, exist_ok=True)
        data = {
            "session_id": "restored1",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        with open(os.path.join(restored_dir, "conversation_full.json"), "w") as f:
            json.dump(data, f)

        agent._restore_session(data, restored_dir)

        # Postcondition: every captured-state tool re-pointed to the restored target.
        assert agent._session_id == "restored1"
        assert agent.tool_registry.get("recall_search")._session_dir == restored_dir
        assert agent.tool_registry.get("memory_write")._session_id == "restored1"
        assert agent.tool_registry.get("plan_create")._session_id == "restored1"

    def test_registry_overwrites_by_name_not_accumulates(self, tmp_path, monkeypatch):
        # register() must replace (same name), else re-registration would
        # leave duplicate/stale instances. Count recall_search entries after
        # a restore: still exactly one.
        agent = _make_real_agent(tmp_path, monkeypatch)
        restored_dir = os.path.join(agent._sessions_root, "restored2")
        os.makedirs(restored_dir, exist_ok=True)
        data = {"session_id": "restored2", "messages": [{"role": "user", "content": "x"}]}
        with open(os.path.join(restored_dir, "conversation_full.json"), "w") as f:
            json.dump(data, f)
        agent._restore_session(data, restored_dir)
        names = [t.name for t in agent.tool_registry.all_tools() if t.name == "recall_search"]
        assert names == ["recall_search"]
