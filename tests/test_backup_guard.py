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

"""Tests for BackupGuard — upfront backup reminder at task start."""

import os

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.backup import BackupGuard


def _shell(cmd, override=""):
    args = {"command": cmd}
    if override:
        args["_override_reason"] = override
    return GuardContext(tool_name="shell", tool_args=args, override_reason=override)


class TestUpfrontBackupGate:
    """First shell command is blocked unconditionally with a backup reminder."""

    def test_first_shell_blocks(self):
        g = BackupGuard()
        v = g.check_pre(_shell("ls -la"))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "upfront_backup_check"
        assert v.overridable is True

    def test_first_shell_blocks_even_readonly(self):
        # Even a harmless read-only command triggers the gate — the point is to
        # make the LLM think about backup BEFORE any command runs.
        g = BackupGuard()
        v = g.check_pre(_shell("cat task.md"))
        assert v is not None
        assert v.action == "block"

    def test_message_mentions_backup(self):
        g = BackupGuard()
        v = g.check_pre(_shell("ls"))
        assert "cp" in v.message
        assert ".bak" in v.message

    def test_second_shell_still_blocked_until_override(self):
        # The one-shot flag is consumed only when the block is RELEASED via a valid
        # override — NOT merely because a first shell was seen. This closes the
        # batched-first-turn leak: without this, a batch of shells in one turn had
        # only its first call blocked while the rest executed unguarded.
        g = BackupGuard()
        v1 = g.check_pre(_shell("ls"))  # first — blocked
        assert v1 is not None and v1.action == "block"
        # A second shell in the SAME turn, with no override, is STILL blocked.
        v2 = g.check_pre(_shell("sqlite3 main.db 'SELECT 1'"))
        assert v2 is not None and v2.action == "block"

    def test_override_releases_gate_permanently(self):
        # Once a valid override is accepted, the flag flips and later shells pass.
        g = BackupGuard()
        assert g.check_pre(_shell("ls")) is not None
        # Accept a valid override (mirrors GuardRegistry.check_pre release path).
        assert g.accept_override("backups not needed, regenerable input", _shell("ls")) is True
        assert g.check_pre(_shell("rm x")) is None
        assert g.check_pre(_shell("sqlite3 main.db")) is None

    def test_short_override_reason_rejected(self):
        # A trivial (<=5 char) reason does not release the gate.
        g = BackupGuard()
        g.check_pre(_shell("ls"))
        assert g.accept_override("ok", _shell("ls")) is False
        # Still blocked because override was rejected.
        assert g.check_pre(_shell("rm x")) is not None

    def test_non_shell_never_blocks(self):
        g = BackupGuard()
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "x"})
        assert g.check_pre(ctx) is None
        # a read_file before any shell does NOT consume the gate
        v = g.check_pre(_shell("ls"))
        assert v is not None and v.action == "block"

    def test_no_command_parsing(self):
        # The guard must NOT inspect the command content — first shell always
        # blocks regardless of whether it looks destructive.
        g = BackupGuard()
        v = g.check_pre(_shell("echo hello"))
        assert v is not None and v.action == "block"


class TestBatchedFirstTurnLeak:
    """Regression: a BATCH of shell calls in one turn must NOT let later calls
    slip past the backup gate once the first is blocked.

    Root cause (extract-elf task, 2026-08-27): the LLM emitted two shells in one
    assistant turn (`file a.out`, `readelf -h a.out`). kernel.py runs check_pre
    per tool_call in order. The old guard set _first_shell_seen=True on the first
    block, so the SECOND shell got a None verdict and executed unguarded — the
    readelf ran and mutation-capable commands could too. The fix consumes the flag
    only in accept_override, so every shell in the turn stays blocked until the LLM
    acknowledges the backup concern once.
    """

    def test_batch_of_shells_all_blocked_before_override(self):
        g = BackupGuard()
        # Simulate kernel.py's per-tool pre-check loop over a single batch.
        batch = [_shell("file a.out"), _shell("readelf -h a.out"), _shell("cat a.out")]
        verdicts = [g.check_pre(ctx) for ctx in batch]
        # Every shell in the batch must be blocked — none may leak through.
        assert all(v is not None and v.action == "block" for v in verdicts), (
            "batched first-turn shells must all be blocked; a None verdict means "
            "a shell would execute unguarded"
        )

    def test_batch_after_override_all_pass(self):
        g = BackupGuard()
        g.check_pre(_shell("file a.out"))
        assert g.accept_override("read-only inspection, input regenerable", _shell("file a.out")) is True
        batch = [_shell("readelf -h a.out"), _shell("cat a.out"), _shell("rm tmp")]
        verdicts = [g.check_pre(ctx) for ctx in batch]
        assert all(v is None for v in verdicts)
