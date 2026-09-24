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

"""IpPortGuard — post-check advisory on IP/port-bearing shell commands.

IPs and ports in this cluster are a proven hallucination hotspot: hosts live in
parallel segments with near-identical suffixes, three distinct SSH channels
carry different keys and port roles (agent→host, agent→container
denied-by-design, container→container), and stale memory entries have
repeatedly produced false "node down" verdicts and wrong-node probes.

This guard never blocks. After a shell command that touches an IP literal or an
SSH-family port flag, it injects a reminder: authoritative values
come from memory facts and on-disk hostfiles, not from recall; and port role
must be distinguished before diagnosing. Every matching command fires —
a batch of ssh fan-outs gets one reminder per command, not one per turn.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# IPv4 literal anywhere in the command (cluster addresses, but also
# also localhost/127.x probes — the reminder is role-aware, not segment-blind).
_IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# SSH-family commands that take a port flag: ssh/scp/rsync/nc/ncat/netcat.
# Matches the command word, so `-P`/`-p` belonging to it is meaningful.
_SSH_FAMILY_RE = re.compile(
    r"(?:^|[|;&]|&&|\|\|)\s*(?:ssh|scp|rsync|nc|ncat|netcat)\b", re.IGNORECASE
)

# A port flag on ssh-family commands: -p <num>, -P <num>, --port=<num>.
_PORT_FLAG_RE = re.compile(r"(?:\s-[pP]\s*\d+)|(?:\s--port=\d+)")

_IP_PORT_MESSAGE = """[IpPortGuard] This command involves an IP or port — both are proven hallucination hotspots in this cluster (parallel host segments with near-identical suffixes, multiple SSH channels with different port roles, false "node down" verdicts from misread ports).

Before drawing any conclusion from this command's result:
1. IP/port values must come from FACTS, not recall: look up the authoritative IP/port mapping saved in memory (a fact-type entry), and cat the on-disk hostfile the launcher actually uses, before any node-health claim or fan-out. Never type an IP from memory recall without verifying it against these.
2. Port role first: each SSH channel (agent→host, agent→container, container→container) has its own port, key set, and failure semantics. A "connection refused/denied" on one channel says nothing about the others — misreading a channel's port as an outage signal is the classic false verdict.

This is an advisory only — no action is blocked. Override/retry freely; the reminder exists to break recall-based probing."""


class IpPortGuard(Guard):
    """Inject-only advisory after shell commands touching IPs/ports.

    Fires on EVERY matching command (no per-turn latch): each new ssh/scp/nc
    command that touches an IP or port gets its own reminder, since each
    command is an independent hallucination risk.
    """

    name = "ip_port"
    priority = 12  # early among post-checks; advisory only, order barely matters

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None

        command = str(ctx.tool_args.get("command", ""))
        if not command:
            return None

        triggered = bool(_IPV4_RE.search(command)) or (
            _SSH_FAMILY_RE.search(command) and _PORT_FLAG_RE.search(command)
        )
        if not triggered:
            return None

        return GuardVerdict.inject(
            message=_IP_PORT_MESSAGE,
            reason="ip_or_port_in_command",
            category="ip_port",
        )
