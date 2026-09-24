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

"""Tests for ShellJobsWaitGuard — block long blocking waits on background jobs."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.shell_jobs_wait import ShellJobsWaitGuard


def _sj(action=None, timeout=None):
    args = {}
    if action is not None:
        args["action"] = action
    if timeout is not None:
        args["timeout"] = timeout
    return GuardContext(tool_name="shell_jobs", tool_args=args)


class TestLongWaitBlocked:
    """action=wait with timeout > 60 is blocked (overridable)."""

    def test_wait_over_60_blocks(self):
        g = ShellJobsWaitGuard()
        v = g.check_pre(_sj("wait", 300))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "long_blocking_wait"
        assert v.overridable is True

    def test_wait_61_blocks(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", 61)).action == "block"

    def test_wait_string_timeout_blocks(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", "120")).action == "block"
        assert g.check_pre(_sj("wait", 120.5)).action == "block"

    def test_message_points_at_poll_and_short_wait(self):
        g = ShellJobsWaitGuard()
        v = g.check_pre(_sj("wait", 300))
        assert "poll" in v.message
        assert "60" in v.message
        assert "_override_reason" in v.message


class TestWaitAllowed:
    """Short/absent waits and non-wait actions pass through."""

    def test_wait_exactly_60_allows(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", 60)) is None

    def test_wait_under_60_allows(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", 30)) is None

    def test_wait_no_timeout_allows(self):
        # Missing timeout defaults to the tool's 60s — allowed.
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait")) is None

    def test_wait_unparseable_timeout_allows(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", "bad")) is None

    def test_poll_list_kill_never_block(self):
        g = ShellJobsWaitGuard()
        for action in ("poll", "list", "kill"):
            assert g.check_pre(_sj(action, 999)) is None, action

    def test_non_shell_jobs_tool_passes(self):
        g = ShellJobsWaitGuard()
        ctx = GuardContext(tool_name="shell",
                           tool_args={"action": "wait", "timeout": 300})
        assert g.check_pre(ctx) is None

    def test_no_action_passes(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(timeout=300)) is None


class TestEveryIterBlocks:
    """Block fires on EVERY long wait, not once-per-turn."""

    def test_repeated_long_wait_all_block(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj("wait", 300)).action == "block"
        assert g.check_pre(_sj("wait", 900)).action == "block"

    def test_override_reason_accepted(self):
        g = ShellJobsWaitGuard()
        assert g.accept_override(
            "all downstream steps staged, only join remains", _sj("wait", 300)
        ) is True
        assert g.accept_override("", _sj("wait", 300)) is False
