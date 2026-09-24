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

"""Tests for LongTimeShellGuard — foreground sleep/timeout >30s blocking (M1)."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.longtimeshell import (
    LongTimeShellGuard, _parse_duration, _find_excessive,
)


def _shell(cmd, background=False):
    return GuardContext(tool_name="shell",
                        tool_args={"command": cmd, "background": background})


# ── Duration parsing ─────────────────────────────────────────────────────────

class TestParseDuration:
    def test_bare_numbers_and_decimals(self):
        assert _parse_duration("31") == 31.0
        assert _parse_duration("0.5") == 0.5
        assert _parse_duration("30.0") == 30.0

    def test_suffixes_case_insensitive(self):
        assert _parse_duration("1m") == 60.0
        assert _parse_duration("45S") == 45.0
        assert _parse_duration("1.5h") == 5400.0
        assert _parse_duration("2d") == 172800.0

    def test_non_durations_return_none(self):
        for tok in ("file", "abc", "1x", "-5", "", "1.2.3", "30%"):
            assert _parse_duration(tok) is None, tok


# ── Blocking table ───────────────────────────────────────────────────────────

BLOCK = [
    "sleep 31",
    "sleep 180",
    "sleep 60 && echo done",
    "sleep 1m",
    "sleep 1.5h",
    "timeout 45 ./run.sh",
    "timeout -k 5 45 ./run.sh",          # -k 5 is kill-after, 45 is duration
    "timeout --signal=KILL 60 ./run.sh",
    "timeout --kill-after=10 45 ./run.sh",
    "FOO=1 timeout 60 ./run.sh",         # env prefix
    "train && sleep 120",                # later segment
    "sleep 2; sleep 40",                 # second sleep is the offense
]

PASS = [
    "sleep 30",                          # exactly at threshold: legal
    "sleep 30.0",
    "sleep 0.5s",
    "sleep 2 && curl -s localhost:8000",
    "timeout 30 ./probe.sh",             # exactly at threshold
    "timeout 5 ./quick.sh",
    "grep timeout file",                 # not in command position
    "grep 'timeout 99' log.txt",
    "echo sleep 999",
    "echo timeout 500 x",
    "man sleep",
    "ls && grep sleep x",                # pipe keeps grep in command position
    "sleep $DUR",                        # variable: not a static duration
    "git log --grep timeout --oneline",
    'grep "x && sleep 999" file',        # quoted span: data, not command syntax
    "grep 'a; sleep 500' f",
    'echo "run && timeout 120 cmd"',     # quoted && must not fake a segment head
    "sleep infinity",                    # static token: not parseable, conservatively pass
]


class TestBlocked:
    def test_all_block_forms_block(self):
        g = LongTimeShellGuard()
        for cmd in BLOCK:
            v = g.check_pre(_shell(cmd))
            assert v is not None and v.action == "block", cmd
            assert v.overridable is True, cmd
            assert v.category == "longtimeshell", cmd

    def test_message_names_kind_duration_and_recipe(self):
        g = LongTimeShellGuard()
        v = g.check_pre(_shell("sleep 180"))
        assert "180" in v.message and "30" in v.message
        assert "background=true" in v.message
        assert "shell_jobs" in v.message
        v = g.check_pre(_shell("timeout -k 5 45 ./run.sh"))
        assert "timeout" in v.message

    def test_reason_scopes_kind_and_duration(self):
        g = LongTimeShellGuard()
        v = g.check_pre(_shell("sleep 180"))
        assert v.reason == "long_foreground_sleep_180s"
        v = g.check_pre(_shell("timeout 45 x"))
        assert v.reason == "long_foreground_timeout_45s"


class TestPasses:
    def test_all_pass_forms_pass(self):
        g = LongTimeShellGuard()
        for cmd in PASS:
            assert g.check_pre(_shell(cmd)) is None, cmd

    def test_background_true_always_passes(self):
        g = LongTimeShellGuard()
        for cmd in ("sleep 180", "timeout 9999 ./train.sh", "sleep 1h"):
            assert g.check_pre(_shell(cmd, background=True)) is None, cmd

    def test_non_shell_tools_ignored(self):
        g = LongTimeShellGuard()
        assert g.check_pre(GuardContext(tool_name="read_file",
                                        tool_args={"path": "x"})) is None
        assert g.check_pre(GuardContext(tool_name="shell",
                                        tool_args={})) is None


# ── Threshold boundary (strictly greater) ────────────────────────────────────

class TestThresholdBoundary:
    def test_exactly_30_passes(self):
        g = LongTimeShellGuard()
        assert g.check_pre(_shell("sleep 30")) is None
        assert g.check_pre(_shell("timeout 30 x")) is None
        assert _find_excessive("sleep 30") is None

    def test_just_over_30_blocks(self):
        g = LongTimeShellGuard()
        assert g.check_pre(_shell("sleep 30.1")) is not None
        assert g.check_pre(_shell("sleep 31")) is not None
        assert g.check_pre(_shell("timeout 30.1 x")) is not None

    def test_custom_threshold_respected(self):
        g = LongTimeShellGuard(threshold=5.0)
        assert g.check_pre(_shell("sleep 5")) is None
        assert g.check_pre(_shell("sleep 6")) is not None


# ── Override semantics ───────────────────────────────────────────────────────

class TestOverride:
    def test_justified_reason_releases_same_offense(self):
        g = LongTimeShellGuard()
        ctx = _shell("sleep 180")
        assert g.check_pre(ctx) is not None
        assert g.accept_override(
            "foreground wait justified: rate-limit cooldown before retry", ctx
        ) is True
        assert g.check_pre(_shell("sleep 180")) is None

    def test_any_nontrivial_reason_releases(self):
        # Trust the LLM after the block — no keyword validation (base-class
        # default semantics). The block itself was the value; the reason's
        # content is not inspected.
        g = LongTimeShellGuard()
        ctx = _shell("sleep 180")
        g.check_pre(ctx)
        assert g.accept_override("just do it please now", ctx) is True
        assert g.check_pre(_shell("sleep 180")) is None

    def test_too_short_reason_rejected(self):
        g = LongTimeShellGuard()
        ctx = _shell("sleep 180")
        g.check_pre(ctx)
        assert g.accept_override("ok", ctx) is False

    def test_different_command_reblocks_after_ack(self):
        g = LongTimeShellGuard()
        ctx = _shell("sleep 180")
        g.check_pre(ctx)
        assert g.accept_override("foreground wait justified: settle window", ctx) is True
        assert g.check_pre(_shell("timeout 90 x")) is not None

    def test_reset_turn_clears_ack(self):
        g = LongTimeShellGuard()
        ctx = _shell("sleep 180")
        g.check_pre(ctx)
        g.accept_override("foreground wait justified: settle window", ctx)
        g.reset_turn()
        assert g.check_pre(_shell("sleep 180")) is not None
