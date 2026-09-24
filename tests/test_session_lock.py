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

"""Tests for session directory concurrency lock."""

import json
import os
import subprocess
import sys

import pytest

from flagscale_agent.react.session import (
    SESSION_LOCK_FILE,
    SessionLockedError,
    acquire_session_lock,
    release_session_lock,
    get_session_lock_holder,
    dir_empty_except_lock,
)

# Child snippet run in a SEPARATE process to test cross-process exclusion.
_CHILD_TRY = r"""
import fcntl, json, os, sys
lp = sys.argv[1]
fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps({"acquired": True}))
except OSError as e:
    print(json.dumps({"acquired": False, "errno": e.errno}))
"""


def _child_try(lock_path):
    out = subprocess.run([sys.executable, "-c", _CHILD_TRY, lock_path],
                         capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestSessionLock:
    def test_acquire_and_release(self, tmp_path):
        d = str(tmp_path / "s1")
        fd = acquire_session_lock(d)
        assert os.path.isfile(os.path.join(d, SESSION_LOCK_FILE))
        release_session_lock(fd)
        # After release, reacquire must succeed
        fd2 = acquire_session_lock(d)
        release_session_lock(fd2)

    def test_cross_process_exclusion(self, tmp_path):
        d = str(tmp_path / "s2")
        fd = acquire_session_lock(d)
        try:
            r = _child_try(os.path.join(d, SESSION_LOCK_FILE))
            assert r["acquired"] is False, f"child acquired lock while held: {r}"
        finally:
            release_session_lock(fd)
        # After release, child must get it
        r2 = _child_try(os.path.join(d, SESSION_LOCK_FILE))
        assert r2["acquired"] is True

    def test_lock_owner_diagnostics_written(self, tmp_path):
        d = str(tmp_path / "s3")
        fd = acquire_session_lock(d)
        try:
            with open(os.path.join(d, SESSION_LOCK_FILE)) as f:
                info = json.load(f)
            assert info["pid"] == os.getpid()
            assert "cmd" in info and "start" in info
        finally:
            release_session_lock(fd)

    def test_get_session_lock_holder_live(self, tmp_path):
        d = str(tmp_path / "s4")
        fd = acquire_session_lock(d)
        try:
            holder = get_session_lock_holder(d)
            assert holder is not None
            assert holder["pid"] == os.getpid()
        finally:
            release_session_lock(fd)
        # After release: no live holder
        assert get_session_lock_holder(d) is None

    def test_get_session_lock_holder_stale_file_is_none(self, tmp_path):
        # A leftover lock file with NO live holder must NOT block
        d = str(tmp_path / "s4")
        os.makedirs(d)
        with open(os.path.join(d, SESSION_LOCK_FILE), "w") as f:
            json.dump({"pid": 999999, "start": "2000-01-01", "cmd": "dead"}, f)
        assert get_session_lock_holder(d) is None

    def test_acquire_raises_session_locked_error(self, tmp_path):
        d = str(tmp_path / "s5")
        fd = acquire_session_lock(d)
        try:
            with pytest.raises(SessionLockedError):
                acquire_session_lock(d)
        finally:
            release_session_lock(fd)
