"""FlagScale Agent CLI entry point.

Usage:
    flagscale-agent                          # interactive mode
    flagscale-agent --provider openai        # use OpenAI provider
    flagscale-agent --model gpt-4o           # specify model
    flagscale-agent "train qwen3 0.6b"       # single-shot query
"""

from pathlib import Path

import typer

from flagscale_agent import __version__

app = typer.Typer(
    name="flagscale-agent",
    help="FlagScale Agent - AI infrastructure agent for large-scale model training and inference.",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider (anthropic, openai)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name (default: provider's default)"),
    base_url: str | None = typer.Option(None, "--base-url", "-b", help="API base URL (for proxies/gateways)"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Agent config YAML path"),
    auto_resume: str | None = typer.Option(None, "--auto-resume", help="Auto-resume session ID (internal use by /reload)"),
    query: str | None = typer.Argument(None, help="Single-shot query (non-interactive mode)"),
    version: bool = typer.Option(None, "--version", "-v", is_eager=True, help="Show version"),
):
    """Start the FlagScale Agent, or run a single query."""
    if version:
        typer.echo(f"flagscale-agent {__version__}")
        raise typer.Exit()

    # Skip if a subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    from flagscale_agent.react.config import AgentConfig
    from flagscale_agent.react.agent import WorkerAgent
    from flagscale_agent.react.orchestrator import Orchestrator

    if config:
        cfg = AgentConfig.from_yaml(str(config))
        if provider != "anthropic":
            cfg.provider = provider
        if model:
            cfg.model = model
        if base_url:
            cfg.base_url = base_url
    else:
        cfg = AgentConfig.auto_load(provider=provider, model=model, base_url=base_url)

    if query:
        cfg.confirm_commands = False

    agent_instance = WorkerAgent(cfg)

    # Wire Orchestrator with shared infrastructure
    orchestrator = Orchestrator(
        provider=agent_instance.provider,
        tool_registry=agent_instance.tool_registry,
        skill_manager=agent_instance.skill_manager,
        session_memory=agent_instance.session_memory,
        task_plan=agent_instance.task_plan,
        experiment_manager=agent_instance._experiment_manager,
        config=cfg,
    )
    agent_instance._orchestrator = orchestrator

    agent_instance.run(single_shot_query=query)


def app_entry():
    """Setuptools console_scripts entry point."""
    app()


if __name__ == "__main__":
    app_entry()
