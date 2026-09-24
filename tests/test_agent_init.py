"""Smoke test: agent can be fully imported and instantiated without errors.

This catches import-time failures (missing modules, circular imports, 
removed guards still referenced) that would crash on reload.
"""
import pytest


def test_agent_import():
    """All imports in agent.py resolve without errors."""
    from flagscale_agent.react.agent import WorkerAgent
    assert WorkerAgent is not None


def test_guard_registry_complete():
    """All guards imported in agent.py can be instantiated."""
    from flagscale_agent.react.guard.safety import ShellSafetyGuard
    from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
    from flagscale_agent.react.guard.plan import PlanGuard
    from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
    from flagscale_agent.react.guard.package_search import PackageSearchGuard
    from flagscale_agent.react.guard.unit_test import UnitTestGuard
    from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
    from flagscale_agent.react.guard.memory_post_check import MemoryPostCheckGuard
    from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard
    from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
    from flagscale_agent.react.guard.arg_type import ArgTypeGuard

    # All must instantiate without error (no missing deps)
    ShellSafetyGuard()
    ContextPressureGuard()
    PlanGuard()
    TrainingMonitorGuard()
    PackageSearchGuard()
    UnitTestGuard()
    MemoryDisciplineGuard()
    MemoryPostCheckGuard()
    PostEvictRecoveryGuard()
    KnowledgeSkillGuard()


def test_tool_registry_complete():
    """All tools imported in agent.py can be instantiated."""
    from flagscale_agent.react.tools.shell import ShellTool
    from flagscale_agent.react.tools.read_file import ReadFileTool
    from flagscale_agent.react.tools.write_file import WriteFileTool
    from flagscale_agent.react.tools.edit_file import EditFileTool
    from flagscale_agent.react.tools.web_fetch import WebFetchTool
    from flagscale_agent.react.tools.load_skill import LoadSkillTool
    from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool
    from flagscale_agent.react.tools.monitor import FlagScaleTrainMonitorTool
    from flagscale_agent.react.tools.inspect_checkpoint import InspectCheckpointTool
    from flagscale_agent.react.tools.evict import EvictTool
    from flagscale_agent.react.tools.recall import RecallTool
    from flagscale_agent.react.tools.memory_write import MemoryWriteTool
    from flagscale_agent.react.tools.memory_read import MemoryReadTool
    from flagscale_agent.react.tools.memory_list import MemoryListTool
    from flagscale_agent.react.tools.plan_create import PlanCreateTool
    from flagscale_agent.react.tools.plan_status import PlanStatusTool
    from flagscale_agent.react.tools.recall_search import RecallSearchTool

    # Verify imports resolve (some tools need constructor args, so just check class exists)
    assert ShellTool is not None
    assert ReadFileTool is not None
    assert WriteFileTool is not None
    assert EditFileTool is not None
    assert WebFetchTool is not None
    assert LoadSkillTool is not None
    assert LoadKnowledgeTool is not None
    assert FlagScaleTrainMonitorTool is not None
    assert InspectCheckpointTool is not None
    assert EvictTool is not None
    assert RecallTool is not None
    assert MemoryWriteTool is not None
    assert MemoryReadTool is not None
    assert MemoryListTool is not None
    assert PlanCreateTool is not None
    assert PlanStatusTool is not None
    assert RecallSearchTool is not None


def test_guard_registry_no_shared_state():
    """GuardRegistry initializes without shared_state attribute."""
    from flagscale_agent.react.guard import GuardRegistry
    reg = GuardRegistry.__new__(GuardRegistry)
    assert not hasattr(reg, 'shared_state')


def test_plan_guard_no_shared_state():
    """PlanGuard has no set_shared_state method."""
    from flagscale_agent.react.guard.plan import PlanGuard
    guard = PlanGuard()
    assert not hasattr(guard, 'set_shared_state')


def test_agent_construction_smoke(tmp_path, monkeypatch):
    """Agent can be constructed with minimal config - catches __init__ order bugs.
    
    This test actually instantiates WorkerAgent, exercising the full __init__ path.
    Without this, initialization order bugs (e.g., using self.X before it's created)
    would only be caught at runtime, not in tests.
    """
    from unittest.mock import Mock
    from flagscale_agent.react.agent import WorkerAgent
    from flagscale_agent.react.config import AgentConfig
    from flagscale_agent.react.memory import Memory
    from flagscale_agent.react.plan import TaskPlan
    
    # Mock provider to avoid real API calls
    mock_provider = Mock()
    mock_provider.count_tokens.return_value = 100
    
    # Mock memory and task_plan to avoid filesystem setup
    mock_memory = Mock(spec=Memory)
    mock_task_plan = Mock(spec=TaskPlan)
    
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
    
    config = AgentConfig(
        session_dir=str(tmp_path / "test_session"),
        api_key="test-key-12345",
        provider="anthropic",
        max_context_tokens=50000,
    )
    
    # Actually instantiate the agent - this runs __init__ completely
    agent = WorkerAgent(
        config,
        _provider=mock_provider,
        _memory=mock_memory,
        _task_plan=mock_task_plan,
    )
    
    # Verify critical components are initialized in correct order
    assert agent.provider is not None, "provider should be created"
    assert agent.history is not None, "history should be created"
    assert agent.context_manager is not None, "context_manager should be created"
    assert agent.memory is not None, "memory should be injected"
    assert agent.task_plan is not None, "task_plan should be injected"
    
    # Verify context_manager has access to its dependencies
    assert agent.context_manager.history is agent.history


def test_startup_hints_surface_open_proposals(tmp_path, monkeypatch):
    """Startup hints must surface unreviewed proposals (not in the dashboard)."""
    from unittest.mock import Mock
    from flagscale_agent.react.agent import WorkerAgent
    from flagscale_agent.react.config import AgentConfig
    from flagscale_agent.react.memory import Memory
    from flagscale_agent.react.plan import TaskPlan
    from flagscale_agent.react.proposals import ProposalRegistry

    mock_provider = Mock()
    mock_provider.count_tokens.return_value = 100
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
    config = AgentConfig(
        session_dir=str(tmp_path / "s"),
        api_key="test-key-12345",
        provider="anthropic",
        max_context_tokens=50000,
    )
    agent = WorkerAgent(
        config, _provider=mock_provider,
        _memory=Mock(spec=Memory), _task_plan=Mock(spec=TaskPlan),
    )
    # Point the agent at a temp registry with one open proposal.
    reg = ProposalRegistry(str(tmp_path / "props"))
    reg.add("Widen a guard", container="agent-code", session_id="prev")
    agent.proposals = reg
    hints = agent._startup_hints()
    assert any("open improvement proposal" in h for h in hints), hints



# ── Expectation anchor assembly (agent side) ──────────────────────────────


class _FakePlan:
    def __init__(self, plan):
        self._plan = plan

    def get_active(self):
        return self._plan


class _Stub:
    """Minimal object exposing task_plan, to exercise the unbound method."""
    def __init__(self, plan):
        self.task_plan = _FakePlan(plan)


def _anchor(plan):
    from flagscale_agent.react.agent import WorkerAgent
    return WorkerAgent._current_expectation_anchor(_Stub(plan))


def test_anchor_empty_when_no_active_plan():
    assert _anchor(None) == ""


def test_anchor_empty_when_no_doing_or_pending_step():
    plan = {"steps": [{"status": "done", "title": "old", "notes": "", "acceptance": []}]}
    assert _anchor(plan) == ""


def test_anchor_prefers_doing_step():
    plan = {"steps": [
        {"status": "done", "title": "s1", "notes": "n1", "acceptance": []},
        {"status": "doing", "title": "train model", "notes": "expect acc>=0.62 in 10min",
         "acceptance": ["acc>=0.62"]},
        {"status": "pending", "title": "s3", "notes": "", "acceptance": []},
    ]}
    a = _anchor(plan)
    assert "train model" in a
    assert "expect acc>=0.62 in 10min" in a
    assert "acc>=0.62" in a
    assert "s3" not in a  # did not pick the pending one


def test_anchor_falls_back_to_first_pending():
    plan = {"steps": [
        {"status": "done", "title": "s1", "notes": "", "acceptance": []},
        {"status": "pending", "title": "next thing", "notes": "plan notes", "acceptance": []},
    ]}
    a = _anchor(plan)
    assert "next thing" in a
    assert "plan notes" in a


def test_anchor_survives_broken_plan_manager():
    class _Boom:
        def get_active(self):
            raise RuntimeError("boom")

    class _S:
        task_plan = _Boom()

    from flagscale_agent.react.agent import WorkerAgent
    assert WorkerAgent._current_expectation_anchor(_S()) == ""
