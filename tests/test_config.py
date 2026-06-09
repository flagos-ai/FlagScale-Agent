"""Tests for AgentConfig."""

import os

import pytest

from flagscale_agent.react.config import AgentConfig

# Env vars that AgentConfig reads — clear them so tests are deterministic
_AGENT_ENV_VARS = [
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "FLAGSCALE_AGENT_CONFIG",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all agent-related env vars before each test."""
    for var in _AGENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.provider == "anthropic"
        assert "claude" in cfg.model
        assert cfg.max_iterations == 200
        assert cfg.max_context_tokens == 200000
        assert cfg.shell_remind_interval == 60
        assert cfg.dangerous_commands_check is True

    def test_anthropic_default_model(self):
        cfg = AgentConfig(provider="anthropic")
        assert "claude" in cfg.model

    def test_openai_model(self):
        cfg = AgentConfig(provider="openai")
        assert cfg.model == "gpt-4o"
        cfg = AgentConfig(provider="openai", model="gpt-3.5-turbo")
        assert cfg.model == "gpt-3.5-turbo"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        cfg = AgentConfig(provider="anthropic")
        assert cfg.api_key == "test-key-123"

    def test_api_key_auth_token_priority(self, monkeypatch):
        """ANTHROPIC_AUTH_TOKEN takes priority over ANTHROPIC_API_KEY."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-abc")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-xyz")
        cfg = AgentConfig(provider="anthropic")
        assert cfg.api_key == "token-abc"

    def test_api_key_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        cfg = AgentConfig(provider="openai")
        assert cfg.api_key == "openai-key"

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
        cfg = AgentConfig(provider="anthropic")
        assert cfg.base_url == "https://proxy.example.com"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-custom-model")
        cfg = AgentConfig(provider="anthropic")
        assert cfg.model == "claude-custom-model"

    def test_from_yaml(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "provider: anthropic\nmax_iterations: 10\nshell_remind_interval: 60\n"
        )
        cfg = AgentConfig.from_yaml(str(f))
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 10
        assert cfg.shell_remind_interval == 60

    def test_from_yaml_ignores_unknown(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("provider: openai\nunknown_field: value\n")
        cfg = AgentConfig.from_yaml(str(f))
        assert cfg.provider == "openai"
        assert not hasattr(cfg, "unknown_field")

    def test_auto_load_with_file(self, tmp_path, monkeypatch):
        f = tmp_path / "agent.yaml"
        f.write_text("provider: anthropic\nmax_iterations: 5\n")
        monkeypatch.setenv("FLAGSCALE_AGENT_CONFIG", str(f))
        cfg = AgentConfig.auto_load()
        assert cfg.provider == "anthropic"
        assert cfg.max_iterations == 5

    def test_auto_load_overrides(self, monkeypatch):
        monkeypatch.delenv("FLAGSCALE_AGENT_CONFIG", raising=False)
        cfg = AgentConfig.auto_load(provider="anthropic", model="claude-test")
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-test"

    def test_skill_dirs_default(self):
        cfg = AgentConfig()
        assert len(cfg.skill_dirs) == 3
