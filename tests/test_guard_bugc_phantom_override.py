"""
Bug C regression test: phantom override crosstalk.

Reproduces the circuit-fibsqrt failure mode — agent writes an override reason for
guard A (e.g. NetworkResilience/knowledge_skill), but when the tool runs it's the
agent's first shell command, BackupGuard fires for the first time, sorts ahead of A
by priority, and silently consumes A's reason (BackupGuard.accept_override accepts
any string >5 chars). Display prints "Guard override: backup" even though the agent
was never shown a backup block. The tool executes (both guards released by one
reason), hiding the bug from the agent, but the override attribution is wrong.

Fix: an override reason may release ONLY the guard that was last surfaced to the
agent. `GuardRegistry._last_surfaced` tracks which guard's block was presented on
the prior turn; the fall-through loop (L233-251 in guard/__init__.py) checks
`guard.name == self._last_surfaced` before calling `accept_override`.
"""
import pytest
from flagscale_agent.react.guard import GuardRegistry, GuardContext, Guard, GuardVerdict


def _ctx(tool="shell", args=None, override=""):
    return GuardContext(
        tool_name=tool,
        tool_args=args or {"command": "ls"},
        override_reason=override,
        turn_count=1,
        recent_tool_history=[],
        context_pressure=0.0,
        classify_fn=lambda x, y, default=False: default
    )


class BackupMockGuard(Guard):
    """Mimics BackupGuard: fires on first shell, accepts any reason >5 chars."""
    name = "backup_mock"
    priority = 5  # BackupGuard sorts very early

    def check_pre(self, ctx):
        if ctx.tool_name == "shell":
            return GuardVerdict.block("[backup]", reason="backup", category="backup")
        return None

    def accept_override(self, reason, ctx):
        return len(reason.strip()) > 5


class NetworkMockGuard(Guard):
    """Mimics NetworkResilience: blocks web_fetch errors, accepts network-recovery reasons."""
    name = "network_mock"
    priority = 85  # NetworkResilience is part of knowledge_skill, high priority

    def check_pre(self, ctx):
        if ctx.tool_name == "web_fetch":
            return GuardVerdict.block("[network]", reason="network", category="network")
        return None

    def accept_override(self, reason, ctx):
        return "network" in reason.lower() or "proxy" in reason.lower()


class TestBugCPhantomOverride:
    """Reproduce and verify fix for Bug C: phantom override crosstalk."""

    def test_backup_does_not_consume_network_reason(self, capsys):
        """
        Bug C scenario: agent hits a network error (web_fetch fails), the network
        guard blocks and is surfaced. Agent writes a network-recovery reason.
        On the retry, the tool is now a shell command (curl), BackupGuard fires
        for the FIRST time, sorts ahead of network by priority. In the old
        fall-through model, BackupGuard's loose accept_override consumed the
        network reason, printed "override: backup", and both guards released.

        Fix: only the last-surfaced guard (network) may be released by the reason.
        BackupGuard was never shown, so it still blocks.
        """
        reg = GuardRegistry()
        reg.register(BackupMockGuard())
        reg.register(NetworkMockGuard())

        # Turn 1: web_fetch fails, network guard surfaces
        ctx_web = _ctx(tool="web_fetch", args={"url": "https://example.com"}, override="")
        v1 = reg.check_pre(ctx_web)
        assert v1 is not None and v1.guard_name == "network_mock"

        # Turn 2: agent retries with shell + network reason
        # BackupGuard fires for first time (prio 5 < 85), but was never surfaced.
        ctx_shell = _ctx(tool="shell",
                          args={"command": "curl https://example.com"},
                          override="attempting network fetch without proxy")
        v2 = reg.check_pre(ctx_shell)
        
        # Bug C would have released backup via network's reason and returned None.
        # Fix: backup was never surfaced (last-surfaced is network_mock), so its
        # loose accept_override is never even consulted — backup blocks.
        assert v2 is not None, "BackupGuard must block (was never shown to agent)"
        assert v2.guard_name == "backup_mock"

        # Verify no phantom "override: backup" was printed.
        captured = capsys.readouterr()
        assert "override: backup" not in captured.out.lower(), \
            "BackupGuard must not display override (never surfaced) — the Bug C symptom"

    def test_sequential_override_each_guard_once_surfaced(self, capsys):
        """
        Positive test: after surfacing each guard, its reason does release it.
        """
        reg = GuardRegistry()
        reg.register(BackupMockGuard())
        reg.register(NetworkMockGuard())

        # Turn 1: shell → backup surfaces
        v1 = reg.check_pre(_ctx(tool="shell", override=""))
        assert v1 is not None and v1.guard_name == "backup_mock"

        # Turn 2: backup reason releases backup
        v2 = reg.check_pre(_ctx(tool="shell", override="backup acknowledged"))
        assert v2 is None, "backup released when it was last-surfaced"

        # Turn 3: web_fetch → network surfaces
        v3 = reg.check_pre(_ctx(tool="web_fetch", override=""))
        assert v3 is not None and v3.guard_name == "network_mock"

        # Turn 4: network reason releases network
        v4 = reg.check_pre(_ctx(tool="web_fetch", override="network recovery attempt"))
        assert v4 is None, "network released when it was last-surfaced"

        captured = capsys.readouterr()
        assert captured.out.count("Guard override") == 2, "2 overrides (one per guard)"
