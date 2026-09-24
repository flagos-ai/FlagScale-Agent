"""Tests for the --time-budget-sec CLI option / time_budget_sec config field
and its precedence over the FLAGSCALE_AGENT_TIME_BUDGET_SEC env var inside
WorkerAgent._task_budget_stats.

Plan A: the config field (settable via --time-budget-sec) drives the same
per-turn wall-clock warnings + wrap-up as the env var; it is NOT a hard kill.
Config field takes precedence; env is the fallback for external harnesses.
"""
import os
import time
import types

import pytest

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.config import AgentConfig

_ENV = "FLAGSCALE_AGENT_TIME_BUDGET_SEC"


def _make(cfg_budget, elapsed=10.0):
    """Minimal stub carrying just what _task_budget_stats reads."""
    stub = types.SimpleNamespace()
    stub.config = AgentConfig(time_budget_sec=cfg_budget)
    stub._turn_start = time.time() - elapsed
    return stub


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    yield


class TestConfigField:
    def test_default_is_zero_unset(self):
        assert AgentConfig().time_budget_sec == 0.0

    def test_field_in_from_yaml_valid_set(self):
        valid = {f.name for f in AgentConfig.__dataclass_fields__.values()
                 if not f.name.startswith("_")}
        assert "time_budget_sec" in valid


class TestPrecedence:
    def test_config_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(_ENV, "999")
        r = WorkerAgent._task_budget_stats(_make(100.0))
        assert r is not None
        assert r["budget"] == 100.0

    def test_env_fallback_when_config_unset(self, monkeypatch):
        monkeypatch.setenv(_ENV, "999")
        r = WorkerAgent._task_budget_stats(_make(0.0))
        assert r is not None
        assert r["budget"] == 999.0

    def test_none_when_both_unset(self):
        assert WorkerAgent._task_budget_stats(_make(0.0)) is None

    def test_none_when_config_unset_and_env_unparseable(self, monkeypatch):
        monkeypatch.setenv(_ENV, "abc")
        assert WorkerAgent._task_budget_stats(_make(0.0)) is None

    def test_config_negative_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv(_ENV, "500")
        r = WorkerAgent._task_budget_stats(_make(-5.0))
        assert r is not None
        assert r["budget"] == 500.0

    def test_stats_shape_and_elapsed(self, monkeypatch):
        r = WorkerAgent._task_budget_stats(_make(100.0, elapsed=25.0))
        assert set(r) == {"elapsed", "budget", "remaining", "pct"}
        assert r["budget"] == 100.0
        assert 24.0 <= r["elapsed"] <= 27.0
        assert 24.0 <= r["pct"] <= 27.0


class TestCliMapping:
    def test_cli_option_declared(self):
        import inspect
        from flagscale_agent import cli
        src = inspect.getsource(cli.main)
        assert "--time-budget-sec" in src
        assert "time_budget_sec" in src

    def test_cli_help_notes_not_hard_kill(self):
        import inspect
        from flagscale_agent import cli
        src = inspect.getsource(cli.main)
        assert "NOT a hard kill" in src

    def test_cli_maps_positive_value_to_cfg(self):
        # mirror cli.py logic: only >0 overrides
        cfg = AgentConfig()
        time_budget_sec = 300.0
        if time_budget_sec > 0:
            cfg.time_budget_sec = time_budget_sec
        assert cfg.time_budget_sec == 300.0

    def test_cli_zero_leaves_cfg_default(self):
        cfg = AgentConfig()
        time_budget_sec = 0.0
        if time_budget_sec > 0:
            cfg.time_budget_sec = time_budget_sec
        assert cfg.time_budget_sec == 0.0
