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

"""Tests for StartupGuard — the START phase three-phase pipeline.

Covers: ordered phase progression (backup -> network_probe -> research),
the core anti-self-satisfaction rule (a heavy network op does NOT satisfy the
probe phase), single-shot gating, and override routing to the surfaced phase.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.startup import (
    StartupGuard, BackupPhase, NetworkProbePhase, ResearchPhase,
    _is_heavy_network_cmd, _is_network_probe,
)


def _shell(cmd, override=""):
    args = {"command": cmd}
    if override:
        args["_override_reason"] = override
    return GuardContext(tool_name="shell", tool_args=args, override_reason=override)


def _tool(name, override=""):
    return GuardContext(tool_name=name, tool_args={}, override_reason=override)


def _drive(guard, ctx, result="ok"):
    """Emulate registry: pre-check, then (if not blocked) post-observe."""
    v = guard.check_pre(ctx)
    if v is None or v.action != "block":
        ctx.tool_result = result
        guard.check_post(ctx)
    return v


class TestCommandClassification:
    def test_heavy_ops_detected(self):
        for cmd in ["git clone https://x", "pip install torch",
                    "sudo apt install gcc", "wget http://x/f.tar",
                    "curl -L http://x -o f"]:
            assert _is_heavy_network_cmd(cmd), cmd

    def test_probe_not_heavy(self):
        for cmd in ["curl -sI https://x", "wget --spider http://x",
                    "command -v curl", "ping -c 1 host", "dig x.com"]:
            assert _is_network_probe(cmd), cmd

    def test_echo_of_heavy_op_not_heavy(self):
        # Only matched at command START (after prefixes) — an echo is not heavy.
        assert not _is_heavy_network_cmd("echo git clone http://x")

    def test_compound_command_heavy_op_detected(self):
        # Regression: the old startswith-only check went blind on any heavy op
        # that was not the first token of the line. These are the common real
        # forms that previously slipped through with the guard seeing nothing.
        for cmd in [
            "cd /repo && git clone https://github.com/x/y",
            "source venv/bin/activate && pip install -r req.txt",
            "mkdir -p build; cd build && pip install torch",
            "for i in 1 2; do wget http://x/$i; done",
        ]:
            assert _is_heavy_network_cmd(cmd), cmd

    def test_package_manager_variants_detected(self):
        # Regression: only bare 'pip install' / 'apt install' were caught before.
        # The full package-manager universe must trigger the probe requirement.
        for cmd in [
            "python -m pip install torch", "python3 -m pip install torch",
            "uv pip install torch", "uv sync", "uv add numpy",
            "poetry install", "conda install -y numpy", "mamba install x",
            "apt-get update", "apt update", "yum install gcc", "dnf install gcc",
            "apk add curl", "brew install wget",
            "npm install", "npm ci", "yarn add react", "pnpm install",
            "cargo build", "cargo install ripgrep", "go mod download", "go get x",
            "gem install rails", "bundle install",
            "huggingface-cli download meta-llama/x", "hf download x",
            "docker pull ubuntu", "pipx install black",
        ]:
            assert _is_heavy_network_cmd(cmd), cmd

    def test_benign_and_probe_not_heavy(self):
        # These must NOT be flagged heavy — compound non-net, echoes, and probes.
        for cmd in [
            "cd /repo && make", "ls -la && cat req.txt", "python train.py",
            "curl -sI https://pypi.org", "time curl -sI https://x",
            "env -u HTTP_PROXY curl -sI https://x", "command -v curl",
        ]:
            assert not _is_heavy_network_cmd(cmd), cmd

    def test_probe_message_demands_fastest_path_not_just_reachability(self):
        # Concern 1: the probe block must instruct picking the FASTEST path via
        # mirror/proxy switching — connectivity alone is insufficient.
        from flagscale_agent.react.guard.startup import _PROBE_BLOCK_MESSAGE
        low = _PROBE_BLOCK_MESSAGE.lower()
        assert "fastest" in low
        assert "mirror" in low
        assert "proxy" in low
        # must frame it as more than a yes/no reachability check
        assert "reachability" in low or "not a yes/no" in low


class TestBackupPhase:
    def test_first_shell_blocks_overridable(self):
        g = StartupGuard()
        v = g.check_pre(_shell("ls -la"))
        assert v is not None and v.action == "block"
        assert v.reason == "upfront_backup_check" and v.overridable is True

    def test_override_routes_to_backup_and_releases(self):
        g = StartupGuard()
        assert g.check_pre(_shell("ls")) is not None
        # registry calls accept_override on the guard that surfaced the block
        assert g.accept_override("regenerable input, no backup needed", _shell("ls")) is True
        # backup satisfied; non-shell / next shell passes (no other phase active)
        assert g.check_pre(_shell("rm x")) is None

    def test_batch_all_blocked_before_override(self):
        g = StartupGuard()
        batch = [_shell("file a"), _shell("readelf -h a"), _shell("cat a")]
        assert all(g.check_pre(c) is not None for c in batch)


class TestPhaseOrdering:
    def test_backup_surfaces_before_probe(self):
        # single-shot on so probe/research phases are active
        g = StartupGuard(single_shot=True)
        v = g.check_pre(_shell("git clone https://x"))
        # backup (order 0) must surface first, not the network probe
        assert v is not None and v.reason == "upfront_backup_check"

    def test_probe_surfaces_after_backup_released(self):
        g = StartupGuard(single_shot=True)
        g.check_pre(_shell("git clone https://x"))
        g.accept_override("no irreplaceable inputs here", _shell("git clone https://x"))
        v = g.check_pre(_shell("git clone https://x"))
        assert v is not None and v.reason == "startup_network_probe"
        assert v.overridable is False


class TestNetworkProbePhase:
    def _released_backup(self, g):
        g.check_pre(_shell("ls"))
        g.accept_override("no irreplaceable inputs here", _shell("ls"))

    def test_heavy_op_does_not_satisfy_probe(self):
        # CORE FIX: a heavy op must NOT self-satisfy the probe. Even after a
        # (blocked) heavy op, the next heavy op is still blocked.
        g = StartupGuard(single_shot=True)
        self._released_backup(g)
        v1 = _drive(g, _shell("git clone https://x"))
        assert v1 is not None and v1.reason == "startup_network_probe"
        # blocked op did not execute; probe still unmet
        v2 = _drive(g, _shell("pip install torch"))
        assert v2 is not None and v2.reason == "startup_network_probe"

    def test_probe_passes_and_satisfies_phase(self):
        g = StartupGuard(single_shot=True)
        self._released_backup(g)
        # a lightweight probe passes and marks the phase satisfied in post
        v = _drive(g, _shell("curl -sI --connect-timeout 3 https://x"))
        assert v is None
        # probe phase now satisfied; a non-heavy real-work call is gated by the
        # research phase (order 2), not the probe phase again. (A heavy op would
        # itself count as research and pass.)
        v2 = g.check_pre(_tool("read_file"))
        assert v2 is not None and v2.reason == "startup_research_gate"

    def test_probe_non_overridable(self):
        g = StartupGuard(single_shot=True)
        self._released_backup(g)
        v = g.check_pre(_shell("git clone https://x"))
        assert v.overridable is False
        # an override does not release a non-overridable probe block
        assert g.accept_override("host is reachable trust me", _shell("git clone https://x")) is False

    def test_probe_inactive_when_not_single_shot(self):
        # supervised mode: probe/research phases skipped entirely
        g = StartupGuard(single_shot=False)
        g.check_pre(_shell("ls"))
        g.accept_override("no irreplaceable inputs here", _shell("ls"))
        assert g.check_pre(_shell("git clone https://x")) is None


class TestResearchPhase:
    def _ready(self, g):
        # release backup + satisfy probe so research is the active gate
        g.check_pre(_shell("ls"))
        g.accept_override("no irreplaceable inputs here", _shell("ls"))
        _drive(g, _shell("curl -sI https://x"))

    def test_first_real_work_blocked_without_research(self):
        g = StartupGuard(single_shot=True)
        self._ready(g)
        v = g.check_pre(_tool("read_file"))
        assert v is not None and v.reason == "startup_research_gate"
        assert v.overridable is False

    def test_research_tool_clears_gate(self):
        g = StartupGuard(single_shot=True)
        self._ready(g)
        _drive(g, _tool("load_knowledge"))
        assert g.check_pre(_tool("read_file")) is None

    def test_heavy_op_counts_as_research(self):
        g = StartupGuard(single_shot=True)
        self._ready(g)
        # a heavy network op is an external info-gain act ≡ research
        _drive(g, _shell("pip install numpy"))
        assert g.check_pre(_tool("read_file")) is None

    def test_meta_tools_never_trip_gate(self):
        g = StartupGuard(single_shot=True)
        self._ready(g)
        for m in ["plan_update", "memory_write", "evict"]:
            assert _drive(g, _tool(m)) is None

    def test_research_override_rejected(self):
        g = StartupGuard(single_shot=True)
        self._ready(g)
        g.check_pre(_tool("read_file"))
        assert g.accept_override("I already know this problem class", _tool("read_file")) is False


class TestFullPipeline:
    def test_full_ordered_progression(self):
        g = StartupGuard(single_shot=True)
        # 1. backup gate
        assert g.check_pre(_shell("ls")).reason == "upfront_backup_check"
        g.accept_override("no irreplaceable inputs here", _shell("ls"))
        # 2. probe gate on first heavy op
        assert g.check_pre(_shell("git clone https://x")).reason == "startup_network_probe"
        _drive(g, _shell("curl -sI https://x"))
        # 3. research gate on first real work
        assert g.check_pre(_tool("read_file")).reason == "startup_research_gate"
        _drive(g, _tool("web_fetch"))
        # all phases satisfied → work flows
        assert g.check_pre(_tool("read_file")) is None
        assert g.check_pre(_shell("gcc x.c")) is None

    def test_reset_turn_preserves_state(self):
        g = StartupGuard(single_shot=True)
        g.check_pre(_shell("ls"))
        g.accept_override("no irreplaceable inputs here", _shell("ls"))
        g.reset_turn()
        # backup stays satisfied across turns (setup is once-per-task)
        assert g.check_pre(_shell("rm x")) is None or \
            g.check_pre(_shell("rm x")).reason != "upfront_backup_check"
