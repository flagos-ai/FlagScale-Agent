# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for VcsBackupGuard — targeted interception of destructive git ops (M1)."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.vcs_backup import VcsBackupGuard


def _shell(cmd):
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


DESTRUCTIVE = [
    "git checkout -- flagscale_agent/react/agent.py",          # discard path
    "git checkout .",                                           # discard all (dot form)
    "git reset --hard HEAD~1",                                  # hard reset
    "git reset --hard origin/main",
    "git clean -fd",                                            # force clean w/ dirs
    "git clean -f",                                             # force clean
    "git clean -xdf",
    "git restore flagscale_agent/react/agent.py",               # discard via restore
    "git stash drop",                                           # destroy a stash
    "git stash clear",                                          # destroy all stashes
    "git branch -D feature-x",                                  # force-delete branch
    "git branch -d feature-x",
    "git push --force origin main",                             # force push
    "git push -f origin main",
    "git push --force-with-lease=abc origin main",
    "git rebase main",                                          # rebase can lose commits
    "git rebase -i HEAD~3",
    "git filter-branch --tree-filter 'rm -rf x'",
]

SAFE = [
    "git checkout -b new-branch",                               # create branch
    "git checkout main",                                        # switch branch
    "git stash push -u -m backup",                              # the backup ritual
    "git stash push -u -m 'backup-before-destructive' && git reset --hard HEAD",
    "git stash list",
    "git stash apply",
    "git stash pop",                                            # pops but does NOT destroy
    "git clean -n",                                             # dry run
    "git clean -nd",
    "git restore --staged foo.py",                              # unstage only
    "git reset HEAD",                                           # unstage (no --hard)
    "git reset --soft HEAD~1",
    "git status",
    "git log --oneline -5",
    "git diff",
    "git branch -a",
    "git push origin main",                                     # normal push
    "ls | grep git",                                            # not a git command
    "cat README.md && git commit -m 'wip'",                     # safe git alongside
]


class TestDestructiveBlocked:
    def test_all_destructive_forms_block(self):
        g = VcsBackupGuard()
        for cmd in DESTRUCTIVE:
            v = g.check_pre(_shell(cmd))
            assert v is not None, f"should block: {cmd}"
            assert v.action == "block", f"should block: {cmd}"
            assert v.overridable is True

    def test_message_has_recovery_recipe(self):
        g = VcsBackupGuard()
        v = g.check_pre(_shell("git reset --hard HEAD~1"))
        assert "stash push" in v.message
        assert "stash apply" in v.message

    def test_reason_is_pattern_scoped(self):
        g = VcsBackupGuard()
        v = g.check_pre(_shell("git reset --hard HEAD"))
        assert v.reason.startswith("destructive_git_pattern_")


class TestSafePasses:
    def test_all_safe_forms_pass(self):
        g = VcsBackupGuard()
        for cmd in SAFE:
            v = g.check_pre(_shell(cmd))
            assert v is None, f"should pass: {cmd}"

    def test_non_shell_tools_ignored(self):
        g = VcsBackupGuard()
        assert g.check_pre(GuardContext(tool_name="read_file",
                                        tool_args={"path": "x"})) is None


class TestOverride:
    def test_valid_backup_reason_releases(self):
        g = VcsBackupGuard()
        ctx = _shell("git reset --hard HEAD~1")
        v = g.check_pre(ctx)
        assert v is not None
        assert g.accept_override("backup exists: stash@{0}", ctx) is True
        assert g.check_pre(_shell("git reset --hard HEAD~1")) is None

    def test_non_backup_reason_rejected(self):
        g = VcsBackupGuard()
        ctx = _shell("git reset --hard HEAD~1")
        g.check_pre(ctx)
        assert g.accept_override("just do it please now", ctx) is False
        assert g.check_pre(_shell("git reset --hard HEAD~1")) is not None

    def test_too_short_reason_rejected(self):
        g = VcsBackupGuard()
        ctx = _shell("git checkout -- x.py")
        g.check_pre(ctx)
        assert g.accept_override("ok", ctx) is False

    def test_different_command_reblocks_after_ack(self):
        # Acknowledging reset --hard does not release checkout --.
        g = VcsBackupGuard()
        ctx = _shell("git reset --hard HEAD~1")
        g.check_pre(ctx)
        assert g.accept_override("backup exists: stash@{0}", ctx) is True
        v = g.check_pre(_shell("git checkout -- x.py"))
        assert v is not None

    def test_reset_turn_clears_acks(self):
        g = VcsBackupGuard()
        ctx = _shell("git reset --hard HEAD~1")
        g.check_pre(ctx)
        g.accept_override("backup exists: stash@{0}", ctx)
        g.reset_turn()
        assert g.check_pre(_shell("git reset --hard HEAD~1")) is not None


class TestCompoundCommandHole:
    """stash-push whitelist must not shield stash drop/clear on the same line."""

    def test_stash_push_plus_clear_blocks(self):
        g = VcsBackupGuard()
        v = g.check_pre(_shell("git stash push -m x && git stash clear"))
        assert v is not None and v.action == "block"

    def test_stash_push_plus_drop_blocks(self):
        g = VcsBackupGuard()
        v = g.check_pre(_shell("git stash push -u -m b && git stash drop"))
        assert v is not None and v.action == "block"

    def test_pure_stash_push_still_passes(self):
        g = VcsBackupGuard()
        assert g.check_pre(_shell("git stash push -u -m backup")) is None

    def test_stash_push_plus_reset_still_passes(self):
        # The taught recipe: snapshot then hard-reset is safe (stash survives).
        g = VcsBackupGuard()
        assert g.check_pre(_shell("git stash push -u -m b && git reset --hard HEAD")) is None
