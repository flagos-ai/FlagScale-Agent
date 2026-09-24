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

"""Tests for IpPortGuard — post-check advisory on IP/port-bearing commands.

All IPs in this file are RFC 5737 documentation-range addresses
(TEST-NET-1: 192.0.2.0/24). No real cluster addresses may appear here.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.ip_port import IpPortGuard


def _shell(cmd):
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


TRIGGER = [
    "ssh -p 22 root@192.0.2.10 hostname",                       # IP + port flag
    "ssh root@192.0.2.20 hostname",                             # bare IP, no flag
    "timeout 8 ssh -o BatchMode=yes root@192.0.2.30 hostname",  # prefix + IP
    "scp -P 2222 file host:~",                                  # scp port flag
    "rsync -avz -e 'ssh -p 60222' data node:/tmp",              # rsync port via -e
    "nc -z 192.0.2.40 60222",                                   # bare IP with nc
    "cat x && ssh -p 60222 root@192.0.2.50 echo OK",            # compound command
    "ping -c1 127.0.0.1",                                       # localhost literal
]

NO_TRIGGER = [
    "ls -la",
    "cat <shared_fs_root>/baseline/hostfile",                   # reads IPs from disk, is the fix
    "grep -rn 'pattern' flagscale_agent/",
    "md5sum /tmp/patch.py",
    "docker exec caozhou nvidia-smi",                           # no IP, no port flag
    "python3 -c 'print(1)'",
    "ssh user@host hostname",                                   # ssh family, NO port flag
    "pytest tests/test_ip_port_guard.py",                       # -p absent entirely
]


class TestTrigger:
    def test_all_trigger_forms_inject(self):
        for cmd in TRIGGER:
            g = IpPortGuard()  # per-command semantics: a fresh guard behaves identically
            v = g.check_post(_shell(cmd))
            assert v is not None, f"should inject: {cmd}"
            assert v.action == "inject", f"should be inject (not block): {cmd}"

    def test_never_blocks(self):
        for cmd in TRIGGER:
            g = IpPortGuard()
            v = g.check_post(_shell(cmd))
            assert v is None or v.action == "inject"
            assert v is None or v.action != "block"

    def test_check_pre_always_none(self):
        for cmd in TRIGGER:
            g = IpPortGuard()
            assert g.check_pre(_shell(cmd)) is None


class TestNoTrigger:
    def test_non_ip_commands_pass(self):
        g = IpPortGuard()
        for cmd in NO_TRIGGER:
            assert g.check_post(_shell(cmd)) is None, f"should not inject: {cmd}"

    def test_4_digit_tail_does_not_match(self):
        # Observed behavior: the IPv4 regex cannot match a 4-digit tail — the
        # boundary/quantifier alignment fails, so no false positive on
        # version-like numbers (e.g. 192.0.2.9999).
        g = IpPortGuard()
        assert g.check_post(_shell("echo 192.0.2.9999")) is None

    def test_non_shell_tools_ignored(self):
        g = IpPortGuard()
        assert g.check_post(GuardContext(tool_name="read_file",
                                         tool_args={"path": "x"})) is None


class TestPerCommand:
    def test_fires_on_every_matching_command(self):
        # Design: no per-turn latch — each matching command is an independent
        # hallucination risk and gets its own reminder (user decision 20260914).
        g = IpPortGuard()
        for _ in range(3):
            v = g.check_post(_shell("ssh -p 22 root@192.0.2.10 hostname"))
            assert v is not None, "every matching command must inject"
            assert v.action == "inject"

    def test_same_command_repeats_within_turn(self):
        g = IpPortGuard()
        assert g.check_post(_shell("ssh root@192.0.2.10 hostname")) is not None
        assert g.check_post(_shell("ssh root@192.0.2.10 hostname")) is not None

    def test_reset_turn_is_base_default(self):
        # The turn-latch override was removed; reset_turn falls back to the
        # base-class no-op — calling it must be safe and change nothing.
        g = IpPortGuard()
        g.reset_turn()  # must not raise
        v = g.check_post(_shell("ssh root@192.0.2.10 hostname"))
        assert v is not None


class TestMessageContent:
    def test_message_carries_both_lessons(self):
        g = IpPortGuard()
        v = g.check_post(_shell("ssh root@192.0.2.20 hostname"))
        # Lesson 1: values from facts, not recall — describes the CHANNELS
        # (memory fact entry + on-disk hostfile), never hardcodes a key name
        assert "FACTS, not recall" in v.message
        assert "hostfile" in v.message
        assert "fact/cluster/" not in v.message  # no session-specific key names
        # Lesson 2: port roles distinguished (abstract form, no port literals)
        assert "port role" in v.message

    def test_no_real_addresses_or_key_names_anywhere(self):
        # The guard module must not leak real cluster segments or specific
        # memory keys into the repo (documentation-range IPs only in tests).
        import flagscale_agent.react.guard.ip_port as m
        src = open(m.__file__).read()
        assert "10.8." not in src
        assert "authoritative_ip_port_map" not in src

    def test_verdict_metadata(self):
        g = IpPortGuard()
        v = g.check_post(_shell("ssh root@192.0.2.60 hostname"))
        assert v.reason == "ip_or_port_in_command"
        assert v.category == "ip_port"
