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

"""Conversation history management — V3 (evict/recall based).

Context management is handled by the model via evict/recall tools.
No automatic aging, truncation, or compaction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Working window ratio: 75% of max_context_tokens
WORKING_WINDOW_RATIO = 0.75
# Fallback if not dynamically set
WORKING_WINDOW_TOKENS = 120_000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~1.5 tokens per CJK char."""
    if not text:
        return 1  # Every message has at least structural overhead
    # Count CJK characters (they typically become 2-3 tokens each in BPE)
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u30ff'
                    or '\uac00' <= c <= '\ud7af')
    ascii_count = len(text) - cjk_count
    # CJK: ~1.5 tokens per char; ASCII: ~0.25 tokens per char (4 chars/token)
    tokens = int(cjk_count * 1.5) + (ascii_count // 4)
    return max(1, tokens)


def _message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens in a single message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += _estimate_tokens(json.dumps(block, ensure_ascii=False))
            else:
                total += _estimate_tokens(str(block))
        return total
    return _estimate_tokens(json.dumps(msg, ensure_ascii=False))


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    """Check if a message is a tool result (OpenAI role=tool or Anthropic tool_result block)."""
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _has_tool_use(msg: Dict[str, Any]) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


def _validate_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure valid conversation structure:
    1. Every tool_result has a preceding tool_use (remove orphaned ones)
    2. Merge consecutive user messages (required by Anthropic API)
    """
    # Step 1: Remove orphaned tool results
    result = []
    for i, msg in enumerate(messages):
        if _is_tool_result(msg):
            # Check if there's a matching tool_call_id in history
            tool_call_id = msg.get("tool_call_id", "")
            tool_use_id = ""
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        break

            has_match = False
            search_id = tool_call_id or tool_use_id
            if search_id:
                for prev in result:
                    if prev.get("role") == "assistant":
                        # OpenAI format
                        for tc in prev.get("tool_calls", []):
                            if tc.get("id") == search_id:
                                has_match = True
                                break
                        # Anthropic format
                        prev_content = prev.get("content")
                        if isinstance(prev_content, list):
                            for block in prev_content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    if block.get("id") == search_id:
                                        has_match = True
                                        break
                    if has_match:
                        break
            else:
                # No ID — check if previous message is assistant with tool_use
                if result and result[-1].get("role") == "assistant" and _has_tool_use(result[-1]):
                    has_match = True

            if has_match:
                result.append(msg)
            # else: drop orphaned tool_result
        else:
            result.append(msg)

    # Step 2: Merge consecutive user messages (Anthropic requires alternating roles)
    merged = []
    for msg in result:
        if msg.get("role") == "user" and merged and merged[-1].get("role") == "user":
            merged[-1] = _merge_user_messages(merged[-1], msg)
        else:
            merged.append(msg)

    return merged


def _merge_user_messages(msg1: Dict[str, Any], msg2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two consecutive user messages into one with list content."""
    def _to_blocks(msg):
        content = msg.get("content", "")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return content
        return [{"type": "text", "text": str(content)}]

    blocks = _to_blocks(msg1) + _to_blocks(msg2)
    return {"role": "user", "content": blocks}


class HistoryManager:
    """Manages conversation message history.

    V3 design: no automatic compaction/aging/truncation.
    Context management is fully handled by the model via evict/recall tools.
    Guard provides pressure awareness, model decides what to evict.
    
    Hard Reset: when eviction is exhausted, hard_reset() clears all messages
    (except system prompt + last N) and injects a continuation message.
    Supports multiple resets per session via _index_offset tracking.
    """

    # Threshold: trigger hard reset when evictable messages fall below this
    HARD_RESET_EVICTABLE_THRESHOLD = 20
    # Minimum messages between resets (cooldown)
    HARD_RESET_COOLDOWN_MESSAGES = 100

    def __init__(self, max_context_tokens: int = 200000):
        self.max_context_tokens = max_context_tokens
        self.working_window = int(max_context_tokens * WORKING_WINDOW_RATIO)
        self._messages: List[Dict[str, Any]] = []
        self._full_log: List[Dict[str, Any]] = []
        self._actual_input_tokens: int = 0
        # Hard reset state
        self._index_offset: int = 0
        self._reset_count: int = 0
        # Legacy properties (kept for compatibility, no longer used for compaction)
        self._compaction_count: int = 0
        self._compaction_happened: bool = False

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages

    @property
    def compaction_happened(self) -> bool:
        """V3: always False — no automatic compaction."""
        return False

    def append(self, message: Dict[str, Any]):
        import copy
        self._messages.append(message)
        # Store a deep copy so eviction (which modifies in-place) doesn't affect the full log
        self._full_log.append(copy.deepcopy(message))
        # Tag message with its external index (full_log position + 1; 0 = system prompt)
        message["_ext_idx"] = len(self._full_log)  # 1-based: full_log[-1] is at position len-1, ext = len

    def set_system_prompt(self, content: str):
        """Replace or prepend the system message."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def report_actual_tokens(self, input_tokens: int):
        """Feed back the actual input_tokens from the API response."""
        self._actual_input_tokens = input_tokens

    def get_context_pressure(self) -> float:
        """Return current context usage as ratio against dynamic working window (60% of max_context_tokens).
        
        Uses max(estimated, actual) where:
        - estimated: character-based estimation of all current messages
        - actual: last API-reported input tokens (reset on eviction since it becomes stale)
        """
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        total = max(estimated, actual)
        return total / self.working_window

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages list. No aging or compaction — evict/recall handles context."""
        return _validate_tool_pairs(list(self._messages))

    def pop_last_assistant(self) -> Optional[Dict[str, Any]]:
        """Remove and return the trailing assistant message from the REAL history.

        get_messages() returns a validated SHALLOW COPY — mutating that copy
        never touches self._messages. Callers that need to actually remove the
        last assistant message (e.g. empty-output retry) must use this method.

        Scope: operates ONLY on _messages (the LLM prompt). _full_log is the
        append-only audit log — the popped message stays there, so session
        persistence, recall_from_full_log, and _ext_idx bookkeeping (the popped
        message holds the highest ext index; new appends only go higher) are
        all unaffected. Not protected by ContextPressureGuard because it never
        touches _full_log.

        Returns None (and does nothing) if history is empty or the last
        message is not an assistant message.
        """
        if self._messages and self._messages[-1].get("role") == "assistant":
            return self._messages.pop()
        return None

    def get_message_at(self, index: int) -> Optional[Dict[str, Any]]:
        """Return message at given index (external), or None if not found/evicted.

        Resolves external index to internal position via _ext_idx scan.
        """
        pos = self.internal_pos_for_ext(index)
        if pos is None:
            # Fallback: try direct internal position (backward compat, no reset case)
            if index >= 0 and index < len(self._messages):
                msg = self._messages[index]
                if msg.get("_evicted"):
                    return None
                return msg
            return None
        msg = self._messages[pos]
        if msg.get("_evicted"):
            return None
        return msg

    # Legacy stubs (no-op, kept for backward compatibility with kernel/commands)
    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self._full_log.clear()
        self._actual_input_tokens = 0
        self._index_offset = 0
        self._reset_count = 0

    # ── Hard Reset ─────────────────────────────────────────────────────────────

    def should_hard_reset(self) -> bool:
        """Determine if a hard reset should be triggered.

        Conditions (any triggers reset):
        1. Evictable < threshold AND context pressure > 0.80
        2. Total messages > 200 AND evictable/total < 5%

        Cooldown: at least HARD_RESET_COOLDOWN_MESSAGES new messages since last reset.
        """
        evictable = self.get_evictable_indexes()
        pressure = self.get_context_pressure()
        total_messages = len(self._messages)

        # Cooldown check: don't reset too soon after previous reset
        if self._reset_count > 0:
            messages_since_reset = len(self._full_log) - self._index_offset
            if messages_since_reset < self.HARD_RESET_COOLDOWN_MESSAGES:
                return False

        # Condition 1: eviction nearly exhausted + high pressure
        if (len(evictable) < self.HARD_RESET_EVICTABLE_THRESHOLD
                and pressure > 0.80):
            return True

        # Condition 2: mostly placeholders
        if (total_messages > 200
                and len(evictable) / max(total_messages, 1) < 0.05):
            return True

        return False

    def hard_reset(self, continuation_message: str, preserve_last_n: int = 4) -> dict:
        """Execute hard reset: discard all messages except last N, inject continuation.

        Can be called multiple times in a single session. Each call:
        - Appends continuation + ack to _full_log (time order)
        - Clears _messages, rebuilds with system + continuation + ack + last N
        - Updates _index_offset to current len(_full_log)
        - _messages order (presentation) differs from _full_log order (chronological)

        Design note:
            _full_log records events in time order: ...last4...continuation...ack...
            _messages presents in comprehension order: system, continuation, ack, last4
            This is intentional — _full_log is an audit log, _messages is an LLM prompt.

        Args:
            continuation_message: The summary/continuation text (role=user).
            preserve_last_n: Number of recent messages to keep (default 4).

        Returns:
            dict with stats: {cleared_count, preserved_count, new_offset, reset_count}
        """
        import copy

        total = len(self._messages)
        self._reset_count += 1

        # Identify system prompt
        system_msg = None
        if self._messages and self._messages[0].get("role") == "system":
            system_msg = self._messages[0]

        # Preserve last N messages (these already exist in _full_log)
        preserved = self._messages[-preserve_last_n:] if preserve_last_n > 0 else []

        # Build continuation + ack messages
        cont_msg = {"role": "user", "content": continuation_message}
        ack_msg = {
            "role": "assistant",
            "content": (
                "Understood. I've read the continuation summary and will resume work. "
                "If I need more context, I'll check conversation_full.json, "
                "plan_status(), or memory_list()."
            ),
        }

        # Append continuation + ack to full_log (chronological order)
        self._full_log.append(copy.deepcopy(cont_msg))
        cont_ext_idx = len(self._full_log)  # 1-based
        cont_msg["_ext_idx"] = cont_ext_idx

        self._full_log.append(copy.deepcopy(ack_msg))
        ack_ext_idx = len(self._full_log)  # 1-based
        ack_msg["_ext_idx"] = ack_ext_idx

        # Set new offset = current full_log length (after cont+ack)
        # Post-reset appends will start at this offset in full_log
        new_offset = len(self._full_log)
        self._index_offset = new_offset

        # Clear and rebuild _messages (presentation order)
        cleared_count = total - (1 if system_msg else 0) - preserve_last_n
        self._messages.clear()

        if system_msg:
            self._messages.append(system_msg)

        # Continuation + ack first (macro context)
        self._messages.append(cont_msg)
        self._messages.append(ack_msg)

        # Then preserved last N (micro context — LLM continues from here)
        for msg in preserved:
            self._messages.append(msg)

        # Reset actual tokens (will be re-reported on next API call)
        self._actual_input_tokens = 0

        return {
            "cleared_count": cleared_count,
            "preserved_count": preserve_last_n,
            "new_offset": new_offset,
            "reset_count": self._reset_count,
        }

    # ── Index Mapping (for hard reset support) ─────────────────────────────────

    def external_index(self, internal_pos: int) -> int:
        """Convert internal _messages position to external index.

        Uses _ext_idx tag on the message if available.
        _messages[0] = system prompt → always external index 0.
        For newly appended messages, _ext_idx is set by append().
        For preserved messages after reset, _ext_idx is their original index.
        """
        if internal_pos == 0:
            return 0  # system prompt
        if internal_pos < 0 or internal_pos >= len(self._messages):
            return -1
        msg = self._messages[internal_pos]
        ext = msg.get("_ext_idx")
        if ext is not None:
            return ext
        # Fallback for messages without tag (shouldn't happen after this change)
        return self._index_offset + (internal_pos - 1)

    def internal_pos_for_ext(self, external_index: int) -> Optional[int]:
        """Convert external index to internal _messages position.

        Returns None if the index is not in current _messages
        (use recall_from_full_log instead).
        """
        if external_index == 0:
            return 0  # system prompt
        for i, msg in enumerate(self._messages):
            if msg.get("_ext_idx") == external_index:
                return i
        return None

    def recall_from_full_log(self, external_index: int) -> Optional[str]:
        """Recall content for any external index from _full_log.

        _full_log[0] corresponds to external index 1
        (first message after system prompt).

        Returns:
            Content string, or None if index is invalid.
        """
        if external_index == 0:
            return None  # system prompt, not in full_log
        full_log_pos = external_index - 1  # -1: ext_idx is 1-based, full_log is 0-based
        if full_log_pos < 0 or full_log_pos >= len(self._full_log):
            return None
        msg = self._full_log[full_log_pos]
        content = msg.get("content", "")
        if isinstance(content, list):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    # ── Evict/Recall (V3 Context Management) ──────────────────────────────────

    def evict_message(self, index: int) -> Dict[str, Any] | None:
        """Evict a message at the given external index.

        Can evict any message except:
        - System prompt (index 0 if role=system)
        - Already evicted messages
        - The last 4 messages (to keep recent context intact)

        Resolves external index via _ext_idx. Falls back to direct position
        for backward compatibility (no reset scenario).

        Returns the original message data (for storage), or None if invalid.
        """
        # Resolve external index to internal position
        internal = self.internal_pos_for_ext(index)
        if internal is None:
            # Fallback: direct internal position (pre-reset compat)
            if index >= 0 and index < len(self._messages):
                internal = index
            else:
                return None

        msg = self._messages[internal]
        # Never evict system prompt
        if msg.get("role") == "system":
            return None
        if msg.get("_evicted"):
            return None
        # Protect the last 4 messages (recent context)
        if internal >= len(self._messages) - 4:
            return None

        content = msg.get("content", "")
        tokens = _message_tokens(msg)

        # Build placeholder based on message type
        # Use _ext_idx for the displayed index so recall works across resets
        display_index = msg.get("_ext_idx", index)
        role = msg.get("role", "unknown")
        if _is_tool_result(msg):
            # Use the resolved INTERNAL position, not the external _ext_idx.
            # _extract_tool_info_for_index indexes self._messages directly, so it
            # must receive an internal position. Passing the external index (which
            # grows unboundedly across hard_resets) caused IndexError on high indexes.
            tool_name, tool_input = self._extract_tool_info_for_index(internal)
            placeholder = (
                f"[evicted | index={display_index} | {tool_name}({tool_input}) | {tokens} tokens]"
            )
        else:
            # For assistant/user messages, show a brief summary
            text_preview = ""
            if isinstance(content, str):
                text_preview = content[:80].replace("\n", " ")
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_preview = b.get("text", "")[:80].replace("\n", " ")
                        break
            placeholder = (
                f"[evicted | index={display_index} | role={role} | {text_preview}... | {tokens} tokens]"
            )

        original_content = content

        # If this assistant message contains tool_use blocks, we must also evict
        # the next user message (which contains the paired tool_result blocks).
        # Otherwise the API will see orphaned tool_result without matching tool_use.
        paired_evict = None
        if role == "assistant" and _has_tool_use(msg):
            next_internal = internal + 1
            if next_internal < len(self._messages) - 4:
                next_msg = self._messages[next_internal]
                if next_msg.get("role") == "user" and not next_msg.get("_evicted"):
                    next_content = next_msg.get("content", "")
                    next_tokens = _message_tokens(next_msg)
                    next_display_idx = next_msg.get("_ext_idx", next_internal)
                    next_placeholder = (
                        f"[evicted | index={next_display_idx} | role=user | tool_results | {next_tokens} tokens]"
                    )
                    paired_evict = {
                        "internal": next_internal,
                        "ext_idx": next_display_idx,
                        "content": next_content,
                        "tokens": next_tokens,
                        "placeholder": next_placeholder,
                    }
        # Reverse: if evicting a user message with tool_result, also evict preceding assistant
        elif role == "user" and _is_tool_result(msg):
            prev_internal = internal - 1
            if prev_internal > 0:
                prev_msg = self._messages[prev_internal]
                if prev_msg.get("role") == "assistant" and _has_tool_use(prev_msg) and not prev_msg.get("_evicted"):
                    prev_content = prev_msg.get("content", "")
                    prev_tokens = _message_tokens(prev_msg)
                    prev_display_idx = prev_msg.get("_ext_idx", prev_internal)
                    prev_placeholder = (
                        f"[evicted | index={prev_display_idx} | role=assistant | tool_use | {prev_tokens} tokens]"
                    )
                    paired_evict = {
                        "internal": prev_internal,
                        "ext_idx": prev_display_idx,
                        "content": prev_content,
                        "tokens": prev_tokens,
                        "placeholder": prev_placeholder,
                    }

        msg["content"] = placeholder
        msg["_evicted"] = True
        msg["_evicted_tokens"] = tokens

        # Invalidate actual tokens — stale after eviction
        self._actual_input_tokens = 0

        # Apply paired eviction
        if paired_evict:
            paired_msg = self._messages[paired_evict["internal"]]
            paired_msg["content"] = paired_evict["placeholder"]
            paired_msg["_evicted"] = True
            paired_msg["_evicted_tokens"] = paired_evict["tokens"]

        metadata = {
            "role": role,
            "tokens": tokens,
        }
        # Include tool info in metadata for tool_result messages
        if _is_tool_result(msg) or (role == "tool"):
            tool_name_meta, tool_input_meta = self._extract_tool_info_for_index(internal)
            metadata["tool_name"] = tool_name_meta
            metadata["tool_input"] = tool_input_meta

        # Include paired evict info with external index for the caller
        paired_info = None
        if paired_evict:
            paired_info = {
                "index": paired_evict["ext_idx"],
                "content": paired_evict["content"],
                "tokens": paired_evict["tokens"],
            }

        return {
            "content": original_content,
            "metadata": metadata,
            "paired_evict": paired_info,
        }

    def recall_message(self, index: int, content: str) -> bool:
        """Restore evicted content at a given index (external index, in-place).
        
        Clears the _evicted flag to allow re-eviction if needed.
        This enables evict → recall → re-evict cycles for dynamic context management.

        Returns True if restored, False if index invalid or not evicted.
        """
        # Resolve external index to internal position
        internal = self.internal_pos_for_ext(index)
        if internal is None:
            # Fallback: direct position (backward compat)
            if index >= 0 and index < len(self._messages):
                internal = index
            else:
                return False
        msg = self._messages[internal]
        if not msg.get("_evicted"):
            return False
        # Deserialize JSON content back to original structure if possible
        # (swap_store serializes lists/dicts to JSON strings; restore them for
        # proper _has_tool_use / _is_tool_result detection on re-eviction)
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith(("[", "{")):
                try:
                    import json as _json
                    content = _json.loads(stripped)
                except (ValueError, TypeError):
                    pass
        msg["content"] = content
        # Clear _evicted flag to allow this message to be evicted again
        del msg["_evicted"]
        if "_evicted_tokens" in msg:
            del msg["_evicted_tokens"]
        return True


    def get_evictable_indexes(self) -> List[int]:
        """Return external indexes of messages that can be evicted, in order.

        All messages except system prompt, already-evicted, and last 4 are evictable.
        Returns external indexes (_ext_idx) for use by the LLM.
        """
        protected_tail = max(0, len(self._messages) - 4)
        result = []
        for i, msg in enumerate(self._messages):
            if msg.get("_evicted"):
                continue
            if msg.get("role") == "system":
                continue
            if i >= protected_tail:
                continue
            ext = msg.get("_ext_idx", i)
            result.append(ext)
        return result

    def _extract_tool_info_for_index(self, index: int) -> tuple:
        """Extract tool_name and key input for a tool_result at internal position `index`.

        `index` is an INTERNAL position into self._messages. Guard against
        out-of-range values so callers never trigger IndexError.
        """
        tool_name = "unknown"
        tool_input = ""

        if index < 0 or index >= len(self._messages):
            return (tool_name, tool_input)

        for i in range(index - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") == "assistant":
                # Anthropic format
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_result_id = self._messages[index].get("tool_use_id", "")
                            if block.get("id", "") == tool_result_id or not tool_result_id:
                                tool_name = block.get("name", "unknown")
                                tool_input = self._summarize_tool_input(
                                    tool_name, block.get("input", {})
                                )
                                return (tool_name, tool_input)
                # OpenAI format
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_result_id = self._messages[index].get("tool_call_id", "")
                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        if tc_id == tool_result_id or not tool_result_id:
                            fn = tc.get("function", {})
                            tool_name = fn.get("name", "unknown")
                            try:
                                args = json.loads(fn.get("arguments", "{}"))
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            tool_input = self._summarize_tool_input(tool_name, args)
                            return (tool_name, tool_input)
                break
        return (tool_name, tool_input)

    @staticmethod
    def _summarize_tool_input(tool_name: str, args: dict) -> str:
        """Create a short summary of tool input for the placeholder."""
        if tool_name == "read_file":
            return args.get("path", "")[:80]
        if tool_name == "shell":
            cmd = args.get("command", "")
            return cmd[:60] + ("..." if len(cmd) > 60 else "")
        if tool_name == "write_file":
            return args.get("path", "")[:80]
        if tool_name == "edit_file":
            return args.get("path", "")[:80]
        if tool_name == "web_fetch":
            return args.get("url", "")[:80]
        if tool_name in ("flagscale_train_monitor",):
            return args.get("output_dir", args.get("log_path", args.get("experiment", "")))[:60]
        for v in args.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""
