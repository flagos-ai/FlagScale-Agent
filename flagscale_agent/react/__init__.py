"""FlagScale Agent. Single WorkerAgent with composable Guard + Judge architecture.

Entry points:
- run_agent(provider, model, mode) — high-level CLI launcher
"""

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.config import AgentConfig
from flagscale_agent.react.orchestrator import Orchestrator
from flagscale_agent.react.scene import ScenePreset, PRESETS
from flagscale_agent.react.profile import WorkerProfile, PROFILES


def run_agent(provider: str = "anthropic", model: str = None, mode: str = None):
    """Entry point: create agent and orchestrator, then run the agent.

    The Orchestrator owns routing decisions (single/subtask/batch).
    WorkerAgent.run() calls Orchestrator.route() on each user input
    to dispatch to the right execution mode.
    """
    config = AgentConfig.auto_load(provider=provider, model=model, mode=mode)
    agent = WorkerAgent(config)

    # ── Wire Orchestrator with shared infrastructure ──
    orchestrator = Orchestrator(
        provider=agent.provider,
        tool_registry=agent.tool_registry,
        skill_manager=agent.skill_manager,
        session_memory=agent.session_memory,
        task_plan=agent.task_plan,
        experiment_manager=agent._experiment_manager,
        config=config,
    )
    agent._orchestrator = orchestrator

    agent.run()
