"""Session persistence — save/load conversation history.

Layout:
  <project_root>/.flagscale/sessions/{session_id}/conversation.json
  <project_root>/.flagscale/sessions/index.yaml
"""

import json
import os
import tempfile
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from flagscale_agent.react.paths import get_sessions_root


def _sessions_root() -> str:
    return get_sessions_root()


def get_session_dir(session_id: str) -> str:
    return os.path.join(_sessions_root(), session_id)


def save_conversation(session_dir: str, session_id: str, messages: List[Dict[str, Any]],
                      loaded_skills: List[str] = None, metadata: Dict = None,
                      completed: bool = False) -> str:
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
    }
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
                "user_turns": len(user_msgs),
                "last_user_msg": last_user,
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


def append_session_index(session_id: str, task: str, summary: str, metadata: str = ""):
    """Append a session summary to the global index.yaml. Keeps last 10."""
    index_path = os.path.join(_sessions_root(), "index.yaml")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    entries = []
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                entries = yaml.safe_load(f) or []
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []

    entry = {
        "session_id": session_id,
        "task": task[:200],
        "summary": summary[:500],
        "metadata": metadata,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    entries.append(entry)
    entries = entries[-10:]

    with open(index_path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, default_flow_style=False)


def get_recent_sessions(n: int = 3) -> List[Dict]:
    """Read recent sessions from global index.yaml."""
    index_path = os.path.join(_sessions_root(), "index.yaml")
    if not os.path.isfile(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            entries = yaml.safe_load(f) or []
        if not isinstance(entries, list):
            return []
        return entries[-n:]
    except Exception:
        return []
