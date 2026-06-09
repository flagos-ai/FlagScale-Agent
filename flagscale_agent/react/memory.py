"""Session memory — stores key findings, decisions, and todos across conversations."""

import json
import os
import re
import time

from typing import Callable, Dict, List, Optional

import yaml


_TYPE_PRIORITY = {"finding": 0, "decision": 1, "todo": 2, "context": 3}

_PRIORITY_TTL = {
    "high": None,       # Never expires
    "critical": 7,      # 7 days — compaction checkpoints, auto-cleanup
    "normal": 30,       # 30 days (default)
    "low": 7,           # 7 days — ephemeral context
}

_PROMOTION_THRESHOLD = 3  # Access count to auto-promote normal → high
_DEDUP_CONFIDENCE_THRESHOLD = 0.7  # Confidence score to consider duplicate (0-1)


class SessionMemory:
    """Incremental memory for cross-session continuity with TTL expiration."""

    def __init__(self, memory_dir: str, ttl_days: int = 30, llm_fn: Optional[Callable[[str], str]] = None):
        self._dir = memory_dir
        self._default_ttl = ttl_days * 86400
        self._ttl = ttl_days * 86400
        self._llm_fn = llm_fn
        self._expansion_cache = {}  # keyword → expanded list, avoids repeated LLM calls
        self._expansion_cache_ttl = 300  # 5 minutes
        self._expansion_cache_ts = {}  # keyword → timestamp
        self._cleanup_expired()  # Clean up expired entries on init

    def _cleanup_expired(self):
        """Remove expired memory entries from disk."""
        if not os.path.isdir(self._dir):
            return
        removed = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if not self._is_valid(entry):
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
        if removed > 0:
            pass

    _KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,78}[a-z0-9]$")

    @staticmethod
    def sanitize_key(raw: str) -> str:
        """Normalize a raw key to lowercase alphanumeric + underscores, max 80 chars."""
        k = raw.lower().strip()
        k = re.sub(r"[^a-z0-9]+", "_", k)
        k = k.strip("_")
        if len(k) > 80:
            k = k[:80].rstrip("_")
        return k

    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        return bool(cls._KEY_RE.match(key))

    def _entry_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.yaml")

    def get(self, key: str) -> Optional[dict]:
        """Get entry by key. Falls back to semantic search if exact match not found."""
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key
        path = self._entry_path(safe)

        # Try exact match first
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if self._is_valid(entry):
                    self._record_access(entry, path)
                    return entry
                else:
                    self.delete(safe)
            except Exception:
                pass

        # Fallback: semantic search by key words
        keywords = re.findall(r'[a-z0-9]+', key.lower())
        keywords = [w for w in keywords if len(w) > 2]
        if not keywords:
            return None

        relevant = self.query_relevant(keywords, max_tokens=500)
        if relevant:
            return relevant[0]
        return None

    def put(self, key: str, mem_type: str, content: str, session_id: str = "", task: str = "", priority: str = "normal", scope: str = "persistent"):
        os.makedirs(self._dir, exist_ok=True)
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key

        # Find and merge related entries (keyword overlap >= 50%)
        merged_content = content
        merged_from = self._find_and_merge_related(safe, content)
        if merged_from:
            merged_content = content
            for old_entry in merged_from:
                old_content = old_entry.get("content", "")
                if old_content and old_content not in merged_content:
                    merged_content = f"{merged_content}\n\n{old_content}"

        entry = {
            "key": safe,
            "type": mem_type,
            "content": merged_content,
            "session_id": session_id,
            "task": task,
            "priority": priority,
            "scope": scope,
            "created": time.time(),
            "access_count": 0,
        }
        path = self._entry_path(safe)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
        self._invalidate_cache()
        return path

    def _find_and_merge_related(self, new_key: str, new_content: str) -> List[dict]:
        """Find entries related to new content and merge them.

        Uses a two-phase approach to minimize LLM calls:
        1. Fast keyword overlap check (no LLM)
           - overlap >= 60%: auto-merge (clearly same topic)
           - overlap 35-60%: ask LLM if available (ambiguous)
           - overlap < 35%: skip (clearly different)
        2. LLM confirmation only for ambiguous cases
        """
        if not os.path.isdir(self._dir):
            return []

        entries = self.list_entries()
        candidates = [e for e in entries if e.get("key") != new_key]
        if not candidates:
            return []

        new_words = set(re.findall(r'[a-z0-9]+', (new_key + " " + new_content).lower()))
        new_words = {w for w in new_words if len(w) > 2}  # min 3 chars
        if not new_words:
            return []

        auto_merge = []
        ambiguous = []

        for entry in candidates:
            existing_key = entry.get("key", "")
            old_words = set(re.findall(r'[a-z0-9]+', (existing_key + " " + entry.get("content", "")).lower()))
            old_words = {w for w in old_words if len(w) > 2}
            if not old_words:
                continue
            overlap = len(new_words & old_words)
            smaller = min(len(new_words), len(old_words))
            if smaller == 0:
                continue
            ratio = overlap / smaller

            # Require at least 3 overlapping words for auto-merge to avoid
            # false positives when one entry has very few words
            if ratio >= 0.60 and overlap >= 3:
                auto_merge.append(entry)
            elif ratio >= 0.35 and self._llm_fn:
                ambiguous.append(entry)

        # For ambiguous cases, ask LLM (batch them in one call)
        llm_merge = []
        if ambiguous and self._llm_fn:
            llm_merge = self._llm_find_related(new_key, new_content, ambiguous)

        # Combine and remove merged entries
        merged = auto_merge + llm_merge
        for entry in auto_merge:
            path = self._entry_path(entry.get("key", ""))
            if os.path.isfile(path):
                os.remove(path)

        return merged

    def _llm_find_related(self, new_key: str, new_content: str, candidates: List[dict]) -> List[dict]:
        """Use LLM to identify which existing entries should be merged with the new one."""
        # Build a concise summary of candidates for LLM
        candidate_summaries = []
        for i, e in enumerate(candidates[:15]):  # Cap to avoid huge prompts
            summary = f"{i}: [{e.get('type','?')}] {e.get('key','')} — {e.get('content','')[:150]}"
            candidate_summaries.append(summary)

        if not candidate_summaries:
            return []

        prompt = (
            "You are a memory management system. A new memory entry is being stored. "
            "Determine which existing entries are about the SAME topic and should be merged.\n\n"
            f"NEW ENTRY:\nKey: {new_key}\nContent: {new_content[:300]}\n\n"
            f"EXISTING ENTRIES:\n" + "\n".join(candidate_summaries) + "\n\n"
            "Which existing entries should be merged into the new one? "
            "Reply with a JSON list of indices (e.g. [0, 3, 7]). "
            "Only include entries that are clearly about the same topic. "
            "If none are related, reply with []. Output ONLY the JSON list:"
        )

        try:
            response = self._llm_fn(prompt).strip()
            # Extract JSON list
            json_match = re.search(r'\[[\d,\s]*\]', response)
            if not json_match:
                return []
            indices = json.loads(json_match.group(0))
            merged = []
            for idx in indices:
                if 0 <= idx < len(candidates):
                    entry = candidates[idx]
                    merged.append(entry)
                    path = self._entry_path(entry.get("key", ""))
                    if os.path.isfile(path):
                        os.remove(path)
            return merged
        except Exception as e:
            return []

    def delete(self, key: str) -> bool:
        safe = self.sanitize_key(key) if not self.is_valid_key(key) else key
        path = self._entry_path(safe)
        if os.path.isfile(path):
            os.remove(path)
            self._invalidate_cache()
            return True
        return False

    def cleanup_session(self, session_id: str) -> int:
        """Remove all session-scoped entries for a given session. Called at session end."""
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if entry.get("scope") == "session" and entry.get("session_id") == session_id:
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count

    def list_entries(self, scope_filter: str = "") -> List[dict]:
        if not os.path.isdir(self._dir):
            return []
        
        # Cache: only re-scan disk if directory mtime changed
        try:
            dir_mtime = os.path.getmtime(self._dir)
        except OSError:
            dir_mtime = 0
        
        cache_key = (scope_filter, dir_mtime)
        if hasattr(self, '_list_cache') and self._list_cache_key == cache_key:
            return list(self._list_cache)  # return copy
        
        entries = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if self._is_valid(entry):
                    if scope_filter and entry.get("scope", "persistent") != scope_filter:
                        continue
                    entries.append(entry)
            except Exception:
                continue
        
        self._list_cache = entries
        self._list_cache_key = cache_key
        return list(entries)

    def _invalidate_cache(self):
        """Invalidate the list_entries cache after writes/deletes."""
        self._list_cache = None
        self._list_cache_key = None

    def query_relevant(self, keywords: list, max_tokens: int = 2000, current_session_id: str = "") -> List[dict]:
        """Return entries relevant to given keywords, within token budget.

        Scores each entry by keyword hits in key+content. Returns top matches
        sorted by score desc, then type priority, then recency.
        Uses LLM keyword expansion when available for semantic matching.
        """
        entries = self.list_entries()
        if current_session_id:
            entries = [e for e in entries
                       if e.get("scope", "persistent") != "session"
                       or e.get("session_id") == current_session_id]

        # Semantic keyword expansion via LLM
        expanded = self._expand_keywords(keywords)
        kws = [k.lower() for k in expanded if k]
        if not kws:
            return []

        scored = []
        for e in entries:
            text = (e.get("key", "") + " " + e.get("content", "")).lower()
            # Use word-boundary matching to avoid false positives
            # e.g., "train" should not match "constraint"
            score = sum(1 for k in kws if re.search(rf'(?<![a-z]){re.escape(k)}(?![a-z])', text))
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: (
            -x[0],
            _TYPE_PRIORITY.get(x[1].get("type", "context"), 9),
            -x[1].get("created", 0),
        ))

        result, used = [], 0
        for _, e in scored:
            content = e.get("content", "")
            cjk = sum(1 for c in content if '一' <= c <= '鿿')
            cost = (len(content) - cjk) // 4 + int(cjk * 1.5) + 10
            if used + cost > max_tokens:
                break
            result.append(e)
            used += cost

        # Record access for returned entries
        for e in result:
            path = self._entry_path(e.get("key", ""))
            if os.path.isfile(path):
                self._record_access(e, path)

        return result

    def recent(self, max_tokens: int = 4000, task_filter: str = "", current_session_id: str = "") -> List[dict]:
        """Return entries within a token budget, prioritized by task relevance, type, then recency.

        If task_filter is provided, entries matching that task come first, then other entries.
        Priority within each group: finding > decision > todo > context.
        Within the same type, newest entries come first.
        Session-scoped entries from other sessions are excluded.
        """
        entries = self.list_entries()
        # Exclude session-scoped entries from other sessions
        if current_session_id:
            entries = [e for e in entries
                       if e.get("scope", "persistent") != "session"
                       or e.get("session_id") == current_session_id]

        if task_filter:
            # Split into task-matching and other entries
            task_entries = [e for e in entries if e.get("task", "") == task_filter]
            other_entries = [e for e in entries if e.get("task", "") != task_filter]

            # Sort each group by type priority, then recency
            task_entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))
            other_entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))

            # Task entries come first
            entries = task_entries + other_entries
        else:
            # No filter: sort by type priority, then recency
            entries.sort(key=lambda e: (
                _TYPE_PRIORITY.get(e.get("type", "context"), 9),
                -e.get("created", 0),
            ))

        result = []
        used = 0
        for e in entries:
            content = e.get("content", "")
            cjk = sum(1 for c in content if '一' <= c <= '鿿' or '　' <= c <= '〿' or '가' <= c <= '힯' or '぀' <= c <= 'ヿ')
            ascii_chars = len(content) - cjk
            cost = ascii_chars // 4 + int(cjk * 1.5) + 10
            if used + cost > max_tokens:
                break
            result.append(e)
            used += cost
        return result

    def clear(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if fname.endswith(".yaml"):
                os.remove(os.path.join(self._dir, fname))
                count += 1
        return count

    def clear_by_type(self, mem_type: str) -> int:
        """Delete all entries of a specific type. Returns count deleted."""
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f)
                if entry.get("type") == mem_type:
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count

    def _is_valid(self, entry: dict) -> bool:
        priority = entry.get("priority", "normal")
        ttl_days = _PRIORITY_TTL.get(priority)
        if ttl_days is None:
            return True  # "high" priority never expires
        # Use the smaller of priority-based TTL and constructor-specified TTL
        ttl_seconds = min(ttl_days * 86400, self._ttl) if self._ttl > 0 else ttl_days * 86400
        if self._ttl == 0:
            return False
        created = entry.get("created", 0)
        return time.time() - created <= ttl_seconds

    def _record_access(self, entry: dict, path: str):
        """Increment access count and promote priority if threshold reached."""
        try:
            access_count = entry.get("access_count", 0) + 1
            entry["access_count"] = access_count

            # Auto-promote normal → high after threshold
            if access_count >= _PROMOTION_THRESHOLD and entry.get("priority") == "normal":
                entry["priority"] = "high"

            # Write back
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            pass

    def _try_dedup(self, key: str, content: str, mem_type: str) -> Optional[str]:
        """Check if semantically duplicate entry exists. If yes, merge and return existing key."""
        if not self._llm_fn:
            return None

        candidates = self._find_dedup_candidates(key, content)
        for candidate in candidates:
            if self._llm_judge_duplicate(candidate, content):
                return self._merge_entries(candidate["key"], content)
        return None

    def _find_dedup_candidates(self, key: str, content: str) -> List[dict]:
        """Find existing entries that might be duplicates of the new entry."""
        entries = self.list_entries()
        # Extract keywords from key and content
        words = re.findall(r'\w+', (key + " " + content).lower())
        keywords = [w for w in words if len(w) > 3][:8]

        if not keywords:
            return []

        # Score by keyword overlap
        scored = []
        for e in entries:
            text = (e.get("key", "") + " " + e.get("content", "")).lower()
            score = sum(1 for k in keywords if k in text)
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:5]]

    def _llm_judge_duplicate(self, existing: dict, new_content: str) -> bool:
        """Ask LLM whether new content is semantically duplicate of existing entry.

        Uses confidence scoring — only merges when confidence >= threshold.
        """
        if not self._llm_fn:
            return False

        try:
            prompt = (
                "Are these two memory entries semantically duplicate (same core information)?\n\n"
                f"Existing [{existing.get('type')}] {existing.get('key')}: {existing.get('content', '')[:300]}\n\n"
                f"New: {new_content[:300]}\n\n"
                "Reply with a confidence score 0.0-1.0 (1.0 = definitely duplicate, 0.0 = completely different).\n"
                "Output ONLY the number, e.g. 0.85"
            )
            answer = self._llm_fn(prompt).strip()
            # Extract float from response
            score_match = re.search(r'(\d+\.?\d*)', answer)
            if score_match:
                confidence = float(score_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
                return confidence >= _DEDUP_CONFIDENCE_THRESHOLD
            # Fallback: treat "yes" as 1.0, "no" as 0.0
            return answer.lower().startswith("yes")
        except Exception as e:
            return False

    def _merge_entries(self, existing_key: str, new_content: str) -> str:
        """Merge new content into existing entry, return merged key."""
        path = self._entry_path(existing_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = yaml.safe_load(f)

            # Append new content if different
            old_content = entry.get("content", "")
            if new_content not in old_content:
                entry["content"] = f"{old_content}\n\n[Updated {time.strftime('%Y-%m-%d')}] {new_content}"
            entry["created"] = time.time()  # Update timestamp

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)

            return existing_key
        except Exception as e:
            return existing_key

    def _expand_keywords(self, keywords: list) -> list:
        """Use LLM to expand keywords with synonyms/aliases for semantic matching.

        Results are cached for 5 minutes to avoid repeated LLM calls for the same keywords.
        """
        if not self._llm_fn or not keywords:
            return keywords

        # Check cache — use sorted tuple as cache key for order-independence
        cache_key = tuple(sorted(k.lower() for k in keywords[:5]))
        now = time.time()
        if cache_key in self._expansion_cache:
            if now - self._expansion_cache_ts.get(cache_key, 0) < self._expansion_cache_ttl:
                return self._expansion_cache[cache_key]
            else:
                # Expired
                del self._expansion_cache[cache_key]
                del self._expansion_cache_ts[cache_key]

        try:
            kw_str = ", ".join(f'"{k}"' for k in keywords[:5])
            prompt = (
                "Expand these technical/error keywords with common synonyms and aliases. "
                "Return JSON mapping each keyword to a list of variants.\n\n"
                f"Input: [{kw_str}]\n\n"
                "Example output: "
                '{"OOM": ["oom", "out of memory", "memory exhaustion", "cuda malloc failed"], '
                '"batch_size": ["batch_size", "micro_batch", "global_batch"]}\n\n'
                "Output JSON only, no explanation:"
            )
            response = self._llm_fn(prompt).strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            expansion = json.loads(response)
            expanded = []
            for kw in keywords:
                expanded.append(kw)
                variants = expansion.get(kw, [])
                if isinstance(variants, list):
                    expanded.extend(variants[:5])

            # Store in cache
            self._expansion_cache[cache_key] = expanded
            self._expansion_cache_ts[cache_key] = now
            # Evict old entries if cache grows too large
            if len(self._expansion_cache) > 50:
                oldest_key = min(self._expansion_cache_ts, key=self._expansion_cache_ts.get)
                del self._expansion_cache[oldest_key]
                del self._expansion_cache_ts[oldest_key]

            return expanded
        except Exception as e:
            return keywords
