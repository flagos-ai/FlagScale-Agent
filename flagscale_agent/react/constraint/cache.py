"""Constraint extraction cache for FlagScale Agent.

Manages disk-cached constraint extraction results so that LLM-based
constraint extraction only runs once per skill content version.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from flagscale_agent.react import display


# Bump this when constraint extraction format changes
_CACHE_VERSION = 4


class ConstraintCache:
    """Disk-backed cache for LLM-extracted skill constraints.

    Thread-safe: uses a lock to serialize extraction calls.
    """

    def __init__(self, cache_dir: str):
        """Initialize constraint cache.

        Args:
            cache_dir: Directory to store constraint_cache.json
        """
        self._cache_dir = cache_dir
        self._lock = threading.Lock()
        self._memory: dict[str, list[dict]] = {}

    @property
    def items(self) -> dict[str, list[dict]]:
        """In-memory constraint items by skill name."""
        return self._memory

    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, "constraint_cache.json")

    def _load_disk_cache(self) -> dict:
        path = self._cache_path()
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if data.get("_version") != _CACHE_VERSION:
                    return {}
                return data.get("entries", {})
        except Exception:
            pass
        return {}

    def _save_disk_cache(self, cache: dict):
        path = self._cache_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "_version": _CACHE_VERSION,
                    "entries": cache,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_or_extract(self, skill_name: str, skill_content: str,
                       extract_fn) -> list[dict]:
        """Get constraints from cache or extract via LLM.

        Args:
            skill_name: Name of the skill
            skill_content: Full skill content text
            extract_fn: Callable that takes skill_content and returns list[dict]

        Returns:
            List of constraint dicts
        """
        if skill_name in self._memory:
            return self._memory[skill_name]

        # Check disk cache (read under lock to avoid race with concurrent writers)
        with self._lock:
            disk_cache = self._load_disk_cache()
            content_hash = hashlib.md5(skill_content.encode()).hexdigest()[:16]
            if skill_name in disk_cache:
                entry = disk_cache[skill_name]
                if entry.get("content_hash") == content_hash and entry.get("items"):
                    self._memory[skill_name] = entry["items"]
                    print(display.dim(
                        f"  📋 [{skill_name}] {len(entry['items'])} constraints (from cache)"
                    ))
                    return entry["items"]

        # Extract outside lock (LLM call is slow), then save under lock
        return self._extract_and_save(skill_name, skill_content, content_hash, extract_fn)

    def _extract_and_save(self, skill_name: str, skill_content: str,
                          content_hash: str, extract_fn) -> list[dict]:
        """Extract constraints via LLM, then save under lock.

        Re-reads disk cache under lock before writing to avoid overwriting
        concurrent writes from other threads.
        """
        # Double-check memory (another thread may have finished first)
        if skill_name in self._memory:
            return self._memory[skill_name]

        try:
            print(display.dim(f"  📋 [{skill_name}] analyzing..."), end="\r")
            raw = extract_fn(skill_content)
            items = []
            if isinstance(raw, list) and raw:
                for c in raw:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id", "")
                    if not cid or not c.get("prompt"):
                        continue
                    items.append(c)
            if items:
                self._memory[skill_name] = items
                # Re-read disk under lock to merge with concurrent writes
                with self._lock:
                    disk_cache = self._load_disk_cache()
                    disk_cache[skill_name] = {"content_hash": content_hash, "items": items}
                    self._save_disk_cache(disk_cache)
                constraint_list = ", ".join(c["id"] for c in items)
                print(display.green(
                    f"  📋 [{skill_name}] {len(items)} constraints: {constraint_list}"
                ))
            else:
                print(display.dim(
                    f"  📋 [{skill_name}] 0 constraints extracted"
                ))
            return items
        except Exception:
            import traceback
            print(display.dim(
                f"  📋 [{skill_name}] constraint extraction skipped"
            ))
            return []

    def batch_extract(self, skill_map: dict[str, str], extract_fn) -> None:
        """Extract constraints from multiple skills concurrently.

        Args:
            skill_map: {skill_name: skill_content, ...}
            extract_fn: Callable that takes skill_content and returns list[dict]
        """
        pending = {
            name: content
            for name, content in skill_map.items()
            if name not in self._memory
        }
        if not pending:
            return

        print(display.dim(f"  📋 Extracting constraints from {len(pending)} skill(s)..."))
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {
                pool.submit(self.get_or_extract, name, content, extract_fn): name
                for name, content in pending.items()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
