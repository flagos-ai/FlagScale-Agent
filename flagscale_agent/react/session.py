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

"""Session persistence — save/load conversation history.

Layout:
  <project_root>/.flagscale/sessions/{session_id}/conversation.json
"""

import fcntl
import json
import os
import shutil
import sys
import tempfile
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_sessions_root


def _sessions_root() -> str:
    return get_sessions_root()


def get_session_dir(session_id: str) -> str:
    return os.path.join(_sessions_root(), session_id)


# ── Session directory lock (concurrency guard) ──────────────────────────────
# A session directory is shared mutable state: conversation.json and
# conversation_full.json are fully rewritten by every save (last-writer-wins),
# swap_store/{index}.json is keyed by external message index (two processes
# evicting the same index silently overwrite each other's stored content),
# and plans/active.yaml is a single pointer. Two live agent processes bound
# to the same directory therefore corrupt each other's history and recall
# garbage across timelines. The lock makes that state single-owner.
#
# Mechanism: flock(LOCK_EX|LOCK_NB) on <session_dir>/.session.lock.
# Verified on the session NFS store: cross-process acquisition is denied
# (BlockingIOError), and the kernel releases the lock automatically when the
# holder process dies (SIGKILL included) — no stale-lock cleanup needed.

SESSION_LOCK_FILE = ".session.lock"


class SessionLockedError(Exception):
    """Another live agent process already holds the session directory."""

    def __init__(self, session_dir: str, holder: Optional[dict] = None):
        self.session_dir = session_dir
        self.holder = holder or {}
        desc = ""
        if self.holder:
            desc = (f" (holder PID {self.holder.get('pid')}, "
                    f"started {self.holder.get('start', '?')}: "
                    f"{self.holder.get('cmd', '?')})")
        super().__init__(f"session directory is locked by another agent process{desc}: {session_dir}")


def _read_lock_owner(session_dir: str) -> Optional[dict]:
    """Best-effort read of holder diagnostics written by the lock owner."""
    try:
        with open(os.path.join(session_dir, SESSION_LOCK_FILE), "r",
                  encoding="utf-8") as f:
            info = json.load(f)
        return info if isinstance(info, dict) else None
    except Exception:
        return None


def acquire_session_lock(session_dir: str) -> int:
    """Take an exclusive advisory lock on session_dir.

    Returns the lock fd — the caller MUST keep it open for the lifetime of
    the binding and pass it to release_session_lock(). flock is released
    automatically by the kernel if the process dies.

    Raises SessionLockedError if another live process holds the lock.
    """
    os.makedirs(session_dir, exist_ok=True)
    lock_path = os.path.join(session_dir, SESSION_LOCK_FILE)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        raise SessionLockedError(session_dir, _read_lock_owner(session_dir)) from e
    try:
        # Diagnostics only — the flock itself is the source of truth.
        os.truncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmd": " ".join(os.sys.argv[1:])[:200],
        }).encode())
    except Exception:
        pass
    return fd


def release_session_lock(fd: int) -> None:
    """Release a lock acquired by acquire_session_lock (idempotent-ish)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def dir_empty_except_lock(session_dir: str) -> bool:
    """True if session_dir holds nothing but the lock file, dotfiles, and
    EMPTY subdirectories.

    Empty subdirs (a freshly-created swap_store/ or plans/ that never
    received data) carry no information and must not pin an abandoned
    session dir on disk forever — that was the accumulation bug: __init__
    always materializes swap_store/, so a reload's old dir could never
    satisfy the old dotfile-only predicate. Any real content at any depth
    (a conversation.json, a swap shard, a plan yaml) still counts as
    non-empty and protects the directory.
    """
    try:
        entries = os.listdir(session_dir)
    except OSError:
        return False
    for e in entries:
        if e.startswith("."):
            continue
        p = os.path.join(session_dir, e)
        if os.path.isdir(p) and not os.listdir(p):
            continue  # empty subdir — ignorable
        return False
    return True


# A candidate must be at least this old before clean_stale_empty_sessions
# will remove it: guards against racing a concurrently booting process in
# the window between its makedirs(session_dir) and acquire_session_lock.
CLEAN_MIN_AGE_SEC = 600


def clean_stale_empty_sessions(sessions_root: str, min_age_sec: int = CLEAN_MIN_AGE_SEC) -> list:
    """Remove session dirs that are empty-except-lock, unlocked, and stale.

    Reload/resume used to leave behind empty shell dirs (dotfiles + an empty
    swap_store/). This is the one-off sweeper for the accumulated backlog and
    the standing hook that keeps the root clean. A directory is removed only
    when ALL hold:
      - dir_empty_except_lock() is True (no recoverable content),
      - no live process holds its lock (get_session_lock_holder is None),
      - its mtime is at least min_age_sec old (not a boot-in-progress).
    Returns the list of removed dir paths. Best-effort: failures are skipped.
    """
    removed: list = []
    try:
        entries = os.listdir(sessions_root)
    except OSError:
        return removed
    now = time.time()
    for e in entries:
        d = os.path.join(sessions_root, e)
        if not os.path.isdir(d):
            continue
        if not dir_empty_except_lock(d):
            continue
        if get_session_lock_holder(d) is not None:
            continue
        try:
            if now - os.stat(d).st_mtime < min_age_sec:
                continue
        except OSError:
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
        except Exception:
            continue
    return removed


def get_session_lock_holder(session_dir: str) -> Optional[dict]:
    """Return live holder info if session_dir is locked by another process, else None.

    Cheap gate on file presence, then a non-blocking probe. A leftover lock
    file with no live holder returns None (lock is auto-released on death).
    """
    lock_path = os.path.join(session_dir, SESSION_LOCK_FILE)
    if not os.path.isfile(lock_path):
        return None
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return _read_lock_owner(session_dir)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def save_conversation(session_dir: str, session_id: str, messages: List[Dict[str, Any]],
                      loaded_skills: List[str] = None, metadata: Dict = None,
                      completed: bool = False, session_summary: str = None,
                      session_input_tokens: int = 0, session_output_tokens: int = 0,
                      turn_count: int = 0, session_input_history: List[str] = None) -> str:
    """Save conversation to session_dir/conversation.json. Overwrites on each call.

    Uses atomic write (tmp file + rename) to prevent corruption on crash.
    """
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, "conversation.json")
    data = {
        "session_id": session_id,
        "timestamp": time.time(),
        "completed": completed,
        "messages": messages,
        "loaded_skills": loaded_skills or [],
        "metadata": metadata or {},
        "session_input_tokens": session_input_tokens,
        "session_output_tokens": session_output_tokens,
        "turn_count": turn_count,
        "session_input_history": session_input_history or [],
    }
    if session_summary:
        data["session_summary"] = session_summary
    # Preserve existing summary if not overwritten
    elif os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("session_summary"):
                data["session_summary"] = existing["session_summary"]
        except Exception:
            pass
    # Atomic write: write to tmp then rename
    fd, tmp = tempfile.mkstemp(dir=session_dir, prefix=".tmp_conversation_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_conversation(session_dir: str) -> Optional[Dict[str, Any]]:
    """Load conversation.json from a session directory. Returns None if not found."""
    path = os.path.join(session_dir, "conversation.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_completed(session_dir: str):
    """Mark a session as completed (normal exit). Uses atomic write."""
    path = os.path.join(session_dir, "conversation.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["completed"] = True
    data["timestamp"] = time.time()
    fd, tmp = tempfile.mkstemp(dir=session_dir, prefix=".tmp_conversation_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def find_resumable_sessions(sessions_root: str = None) -> List[Dict[str, Any]]:
    """Find sessions with completed=false, sorted by timestamp desc."""
    root = sessions_root or _sessions_root()
    if not os.path.isdir(root):
        return []
    results = []
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue
        conv_path = os.path.join(entry_path, "conversation.json")
        if not os.path.isfile(conv_path):
            continue
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("completed", True):
                continue
            messages = data.get("messages", [])
            user_msgs = [m for m in messages if m.get("role") == "user"]
            last_user = ""
            if user_msgs:
                content = user_msgs[-1].get("content", "")
                if isinstance(content, str):
                    last_user = content[:100]
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_user = block.get("text", "")[:100]
                            break
            results.append({
                "session_id": data.get("session_id", entry),
                "session_dir": entry_path,
                "timestamp": data.get("timestamp", 0),
                "loaded_skills": data.get("loaded_skills", []),
                "user_turns": data.get("turn_count") or len(user_msgs),
                "last_user_msg": last_user,
                "session_summary": data.get("session_summary", ""),
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


def list_sessions(sessions_root: str = None) -> List[Dict[str, Any]]:
    """List all sessions by scanning sessions/*/conversation.json."""
    root = sessions_root or _sessions_root()
    if not os.path.isdir(root):
        return []
    sessions = []
    for entry in sorted(os.listdir(root), reverse=True):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue
        conv_path = os.path.join(entry_path, "conversation.json")
        if not os.path.isfile(conv_path):
            continue
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            sessions.append({
                "session_id": data.get("session_id", entry),
                "session_dir": entry_path,
                "timestamp": data.get("timestamp", 0),
                "completed": data.get("completed", True),
                "turns": len([m for m in messages if m.get("role") == "user"]),
            })
        except Exception:
            continue
    return sessions

