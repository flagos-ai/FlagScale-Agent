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

"""FlagScale Agent CLI entry point.

Usage:
    flagscale-agent                          # interactive mode
    flagscale-agent --provider openai        # use OpenAI provider
    flagscale-agent --model gpt-4o           # specify model
    flagscale-agent "train qwen3 0.6b"       # single-shot query

Environment variables:
    ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN  # Anthropic API key
    ANTHROPIC_MODEL                           # Default Anthropic model (e.g., claude-opus-4-20250514)
    OPENAI_API_KEY                            # OpenAI API key
    OPENAI_DEFAULT_MODEL                      # Default OpenAI model (e.g., gpt-4o-mini)
    OPENAI_BASE_URL / ANTHROPIC_BASE_URL      # Custom API endpoints
"""

from __future__ import annotations

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
    max_output_tokens: int = typer.Option(8192, "--max-output-tokens", help="Max output tokens per response"),
    thinking_budget: int = typer.Option(0, "--thinking-budget", help="Thinking budget tokens (0=disabled, >0=enable thinking)"),
    time_budget_sec: float = typer.Option(0.0, "--time-budget-sec", help="Per-turn wall-clock budget in seconds (0=unset). Drives time-remaining warnings and a wrap-up reminder at 100%; NOT a hard kill. Equivalent to env FLAGSCALE_AGENT_TIME_BUDGET_SEC."),
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
    if config:
        cfg = AgentConfig.from_yaml(str(config))
        # Always apply CLI overrides
        cfg.provider = provider
        if model:
            cfg.model = model
        if base_url:
            cfg.base_url = base_url
    else:
        cfg = AgentConfig.auto_load(provider=provider, model=model, base_url=base_url)
    if thinking_budget:
        cfg.thinking_budget = thinking_budget
    if max_output_tokens != 8192:
        cfg.max_output_tokens = max_output_tokens
    if time_budget_sec > 0:
        cfg.time_budget_sec = time_budget_sec

    agent_instance = WorkerAgent(cfg)

    agent_instance.run(single_shot_query=query)


def app_entry():
    """Setuptools console_scripts entry point."""
    app()


if __name__ == "__main__":
    app_entry()
