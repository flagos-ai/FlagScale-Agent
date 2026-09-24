# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org restore:session_locks/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for _restore_session concurrency guard + empty-dir cleanup predicate.

These tests exercise the EXACT lock-prefix sequence used by
agent.py::_restore_session (guard -> acquire new -> rebind -> release old ->
cleanup), replicated via a minimal fake agent so no API keys are needed.
The conversation-restore part is covered by test_conversation_full_restore.py.
"""

import json
import os
import shutil
import time

import pytest

from flagscale_agent.react.session import (
    SESSION_LOCK_FILE, acquire_session_lock, release_session_lock,
    dir_empty_except_lock, get_session_lock_holder, SessionLockedError,
    clean_stale_empty_sessions, CLEAN_MIN_AGE_SEC,
)


class TestDirEmptyExceptLock:
    def test_fresh_empty_dir(self, tmp_path):
        d = str(tmp_path / "empty")
        os.makedirs(d)
        assert dir_empty_except_lock(d) is True

    def test_dir_with_only_lock_file(self, tmp_path):
        d = str(tmp_path / "lockonly")
        os.makedirs(d)
        fd = acquire_session_lock(d)
        release_session_lock(fd)
        # lock file remains on disk; dir still counts as empty
        assert os.path.isfile(os.path.join(d, SESSION_LOCK_FILE))
        assert dir_empty_except_lock(d) is True

    def test_dir_with_conversation_not_empty(self, tmp_path):
        d = str(tmp_path / "hasconv")
        os.makedirs(d)
        with open(os.path.join(d, "conversation.json"), "w") as f:
            json.dump({}, f)
        assert dir_empty_except_lock(d) is False

    def test_dir_with_empty_swap_store_subdir_is_empty(self, tmp_path):
        # Semantics flip (2026-09): SwapStore/TaskPlan __init__ always
        # makedirs their subdirs, so an abandoned reload dir holds an empty
        # swap_store/ and nothing else. The old dotfile-only predicate kept
        # those shells forever (the accumulation bug). An empty subdir
        # carries no data -> still counts as empty.
        d = str(tmp_path / "emptysub")
        os.makedirs(os.path.join(d, "swap_store"))
        assert dir_empty_except_lock(d) is True

    def test_dir_with_nonempty_subdir_not_empty(self, tmp_path):
        # The flip must not open a hole: real content at ANY depth protects
        # the directory. A swap shard inside swap_store/ makes it non-empty.
        d = str(tmp_path / "deepdata")
        os.makedirs(os.path.join(d, "swap_store"))
        with open(os.path.join(d, "swap_store", "3.json"), "w") as f:
            json.dump({"idx": 3}, f)
        assert dir_empty_except_lock(d) is False

    def test_missing_dir_is_not_empty(self, tmp_path):
        assert dir_empty_except_lock(str(tmp_path / "nope")) is False


class _FakePlan:
    def __init__(self):
        self._dir = None


class _FakeAgent:
    """Minimal agent stub exposing exactly the attrs _restore_session touches."""

    def __init__(self, session_dir, sessions_root):
        self._session_id = "aaaa1111"
        self._session_dir = session_dir
        self._sessions_root = sessions_root
        self._session_lock_fd = acquire_session_lock(session_dir)
        self.task_plan = _FakePlan()
        self.turn_count = 0


def _restore_lock_prefix(agent, data, session_dir):
    """Replicate the lock-guard prefix of agent.py::_restore_session.

    Kept in sync with agent.py lines ~1039-1082: guard -> acquire new ->
    rebind -> release old -> plan dir -> cleanup empty old dir.
    """
    from flagscale_agent.react.agent import (
        get_session_lock_holder, acquire_session_lock as _a, release_session_lock as _r,
    )
    holder = get_session_lock_holder(session_dir)
    if holder:
        raise SessionLockedError(session_dir, holder)
    old_session_dir = agent._session_dir
    old_lock_fd = getattr(agent, "_session_lock_fd", None)
    new_lock_fd = _a(session_dir)
    agent._session_id = data.get("session_id", agent._session_id)
    agent._session_dir = session_dir
    if new_lock_fd is not old_lock_fd:
        if old_lock_fd is not None:
            _r(old_lock_fd)
        agent._session_lock_fd = new_lock_fd
    agent.task_plan._dir = os.path.join(session_dir, "plans")
    if old_session_dir != session_dir and dir_empty_except_lock(old_session_dir):
        shutil.rmtree(old_session_dir, ignore_errors=True)
    # agent.py also sweeps stale empty shells across the root on every resume.
    try:
        clean_stale_empty_sessions(agent._sessions_root)
    except Exception:
        pass


class TestRestoreSessionGuard:
    def test_rebind_releases_old_lock_holds_new(self, fake_agent):
        new_dir = str(fake_agent._sessions_root + "/bbbb2222")
        os.makedirs(new_dir)
        _restore_lock_prefix(fake_agent, {"session_id": "bbbb2222"}, new_dir)
        # old dir's lock released -> acquirable by others
        fd = acquire_session_lock(str(fake_agent._sessions_root + "/aaaa1111"))
        release_session_lock(fd)
        # new dir held by us
        holder = get_session_lock_holder(new_dir)
        assert holder is not None and holder["pid"] == os.getpid()

    def test_refused_when_held_by_other_process(self, fake_agent):
        new_dir = str(fake_agent._sessions_root + "/cccc3333")
        os.makedirs(new_dir)
        other_fd = acquire_session_lock(new_dir)  # simulate another live process
        try:
            with pytest.raises(SessionLockedError):
                _restore_lock_prefix(fake_agent, {"session_id": "cccc3333"}, new_dir)
        finally:
            release_session_lock(other_fd)
        # agent still holds its ORIGINAL lock (state unchanged on refusal)
        assert get_session_lock_holder(fake_agent._session_dir) is not None

    def test_lock_only_old_dir_gets_cleaned(self, fake_agent):
        # Old dir contains ONLY the lock file: old predicate kept it forever
        # (the empty-shell bug); new predicate cleans it.
        new_dir = str(fake_agent._sessions_root + "/dddd4444")
        os.makedirs(new_dir)
        _restore_lock_prefix(fake_agent, {"session_id": "dddd4444"}, new_dir)
        assert not os.path.isdir(str(fake_agent._sessions_root + "/aaaa1111"))

    def test_nonempty_old_dir_not_cleaned(self, fake_agent):
        old = str(fake_agent._sessions_root + "/aaaa1111")
        with open(os.path.join(old, "conversation.json"), "w") as f:
            json.dump({"session_id": "aaaa1111"}, f)
        new_dir = str(fake_agent._sessions_root + "/eeee5555")
        os.makedirs(new_dir)
        _restore_lock_prefix(fake_agent, {"session_id": "eeee5555"}, new_dir)
        assert os.path.isdir(old)


@pytest.fixture
def fake_agent(tmp_path):
    old_dir = str(tmp_path / "aaaa1111")
    os.makedirs(old_dir)
    return _FakeAgent(old_dir, str(tmp_path))

class TestCleanStaleEmptySessions:
    """The sweeper that reaps accumulated empty shell dirs.

    Three-condition guard: empty-except-lock AND unlocked AND stale. Each
    test holds two conditions fixed and varies the third, so a failure
    localizes to one condition.
    """

    def _age(self, d, seconds):
        old = time.time() - seconds
        os.utime(d, (old, old))

    def test_stale_empty_unlocked_is_removed(self, tmp_path):
        d = str(tmp_path / "stale_shell")
        os.makedirs(os.path.join(d, "swap_store"))
        self._age(d, CLEAN_MIN_AGE_SEC + 60)
        removed = clean_stale_empty_sessions(str(tmp_path))
        assert d in removed
        assert not os.path.isdir(d)

    def test_locked_empty_is_kept(self, tmp_path):
        # Even a stale+empty dir is kept while a live process holds its lock
        # (that process may still be booting/writing). Condition isolated:
        # only the lock differs from test_stale_empty_unlocked_is_removed.
        d = str(tmp_path / "locked_shell")
        os.makedirs(os.path.join(d, "swap_store"))
        self._age(d, CLEAN_MIN_AGE_SEC + 60)
        fd = acquire_session_lock(d)
        try:
            removed = clean_stale_empty_sessions(str(tmp_path))
            assert d not in removed
            assert os.path.isdir(d)
        finally:
            release_session_lock(fd)

    def test_fresh_empty_is_kept(self, tmp_path):
        # Only staleness differs here: a fresh empty dir may be a process
        # between makedirs() and acquire_session_lock().
        d = str(tmp_path / "fresh_shell")
        os.makedirs(os.path.join(d, "swap_store"))  # mtime == now
        removed = clean_stale_empty_sessions(str(tmp_path))
        assert d not in removed
        assert os.path.isdir(d)

    def test_stale_nonempty_is_kept(self, tmp_path):
        # Only emptiness differs: recoverable content pins the dir forever.
        d = str(tmp_path / "stale_data")
        os.makedirs(d)
        with open(os.path.join(d, "conversation.json"), "w") as f:
            json.dump({"session_id": "x"}, f)
        self._age(d, CLEAN_MIN_AGE_SEC + 3600)
        removed = clean_stale_empty_sessions(str(tmp_path))
        assert d not in removed
        assert os.path.isdir(d)

    def test_missing_root_is_noop(self, tmp_path):
        removed = clean_stale_empty_sessions(str(tmp_path / "does_not_exist"))
        assert removed == []
