"""Conversation history management with context window protection."""

import json

from typing import Any, Callable, Dict, List, Optional

from flagscale_agent.react import display

SUMMARIZE_PROMPT = (
    "Summarize this conversation segment for an AI agent that will continue working on the same task. "
    "Think about RE-READ COST: if a piece of information is lost, will the agent need to re-execute "
    "a command or re-read a file to recover it? If yes, that information MUST be in the summary.\n\n"
    "PRESERVE with high fidelity:\n"
    "- File paths read AND their key content (structure, config values, version numbers)\n"
    "- Error messages, root causes, and what fixed them\n"
    "- Environment state: what's installed, what versions, what paths\n"
    "- Decisions made and their rationale (especially version choices, architecture choices)\n"
    "- What was tried and failed (so the agent doesn't retry)\n"
    "- Current approach/strategy and what phase we're in\n\n"
    "DO NOT preserve:\n"
    "- Verbose install/build logs (just the outcome)\n"
    "- Repetitive monitoring output (just the conclusion)\n"
    "- Directory listings (just the key paths found)\n\n"
    "Be specific: include exact version numbers, exact paths, exact error messages. "
    "Keep the summary under 1500 tokens."
)

COMPACTION_NOTICE = (
    "<context-compacted>\n"
    "Previous context was compacted. A summary of dropped content is available in "
    "<context-summary> above.\n"
    "If you need details that aren't in the summary, re-read the relevant files "
    "rather than assuming you remember.\n"
    "</context-compacted>"
)

MAX_SUMMARY_TOKENS = 4000  # default, overridden dynamically as max_context_tokens * 0.05
TRUNCATE_THRESHOLD = 2000
KEEP_RECENT = 16
AGING_WINDOW = 10
AGING_THRESHOLD = 800


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 ASCII chars per token, ~1.5 CJK chars per token."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿' or '가' <= c <= '힯' or '぀' <= c <= 'ヿ')
    ascii_chars = len(text) - cjk
    return ascii_chars // 4 + int(cjk * 1.5) + 1


def _smart_truncate(content: str, max_chars: int = 600) -> str:
    """Truncate preserving structure: first lines + error tail + summary."""
    if len(content) <= max_chars:
        return content
    lines = content.splitlines()

    error_tail = _extract_error_tail(content, max_chars=400)
    if error_tail:
        head = "\n".join(lines[:3])
        return f"{head}\n[... {len(lines)} lines, {len(content)} chars ...]\n{error_tail}"

    if len(lines) > 15:
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-5:])
        return f"{head}\n[... {len(lines) - 10} lines omitted, {len(content)} chars total ...]\n{tail}"

    return content[:max_chars] + f"\n[... truncated, {len(content)} chars total]"


def _classify_content_value(content: str) -> str:
    """Classify tool result content by its re-read cost / information density.

    Returns: 'high', 'medium', or 'low'
    - high: file contents, configs, errors — expensive to re-obtain, dense info
    - medium: directory listings, grep results — moderate density
    - low: install logs, build output, repetitive monitoring — low density
    """
    lower = content[:600].lower()

    # Errors are always high value
    if any(kw in content for kw in ("Error", "ERROR", "Traceback", "FAILED", "Exception")):
        return "high"

    # Install/build logs are low value
    if any(kw in lower for kw in ("installing", "collecting", "downloading",
                                   "successfully installed", "requirement already",
                                   "building wheel", "running setup")):
        return "low"

    # File content (read_file results, cat output) — high value
    # Heuristic: contains code-like patterns or structured data
    if any(kw in lower for kw in ("import ", "def ", "class ", "from ", "---\n",
                                   "\"type\":", "{", "}", "export ", "#include")):
        return "high"

    # Directory listings — medium value
    lines = content.splitlines()
    if len(lines) > 5 and sum(1 for l in lines[:10] if l.strip().startswith(("/", "./"))) > 5:
        return "medium"

    # Default: medium
    return "medium"


def _age_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate a single message's tool results based on content value.

    High-value content (file contents, errors) gets a generous limit.
    Low-value content (install logs) gets aggressively truncated.
    """
    content = msg.get("content", "")

    if isinstance(content, str) and len(content) > AGING_THRESHOLD:
        if msg.get("role") == "tool":
            return {**msg, "content": _value_aware_truncate(content)}

    if isinstance(content, list):
        new_blocks = []
        changed = False
        for block in content:
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and isinstance(block.get("content", ""), str)
                    and len(block["content"]) > AGING_THRESHOLD):
                new_blocks.append({**block, "content": _value_aware_truncate(block["content"])})
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            return {**msg, "content": new_blocks}

    return msg


def _value_aware_truncate(content: str) -> str:
    """Truncate based on content value classification."""
    value = _classify_content_value(content)

    if value == "high":
        # Generous limit: keep structure visible
        return _smart_truncate(content, max_chars=1500)
    elif value == "low":
        # Aggressive: just outcome
        lines = content.splitlines()
        if len(lines) > 6:
            head = "\n".join(lines[:2])
            tail = "\n".join(lines[-3:])
            return f"{head}\n[... {len(lines) - 5} lines of output ...]\n{tail}"
        return _smart_truncate(content, max_chars=300)
    else:
        # Medium: standard truncation
        return _smart_truncate(content, max_chars=800)


def _age_tool_results(messages: List[Dict[str, Any]], keep_recent: int = AGING_WINDOW) -> List[Dict[str, Any]]:
    """Proactively truncate old tool results to save context budget.

    Skill-injection messages (identified by tool_call_id starting with 'auto_' or 'skill_')
    are always truncated aggressively regardless of recency, since the LLM has already
    absorbed their content in earlier iterations.
    """
    if len(messages) <= keep_recent:
        return messages
    cutoff = len(messages) - keep_recent
    result = []
    for i, msg in enumerate(messages):
        if i >= cutoff:
            # Even in the recent window, aggressively truncate skill messages
            if _is_skill_injection(msg):
                result.append(_truncate_skill_message(msg))
            else:
                result.append(msg)
            continue
        if msg.get("role") == "system":
            result.append(msg)
            continue
        result.append(_age_message(msg))
    return result


def _is_skill_injection(msg: Dict[str, Any]) -> bool:
    """Check if a message is a skill-injection tool_result."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id", "")
            if tid.startswith("auto_") or tid.startswith("skill_"):
                return True
    return False


def _truncate_skill_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Replace skill content with a minimal reference."""
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    new_blocks = []
    for block in content:
        if (isinstance(block, dict) and block.get("type") == "tool_result"
                and (block.get("tool_use_id", "").startswith("auto_")
                     or block.get("tool_use_id", "").startswith("skill_"))):
            text = block.get("content", "")
            # Extract just the skill name from the content
            skill_ref = "[skill content already loaded — use read_file on SKILL.md if needed]"
            if '<skill name="' in text:
                import re as _re
                m = _re.search(r'<skill name="([^"]+)"', text)
                if m:
                    skill_ref = f"[skill '{m.group(1)}' already loaded]"
            new_blocks.append({**block, "content": skill_ref})
        else:
            new_blocks.append(block)
    return {**msg, "content": new_blocks}


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


def _extract_text(msg: Dict[str, Any]) -> str:
    """Extract readable text from a message for summarization."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(inner[:500])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    parts.append(f"[tool_use: {name}({json.dumps(inp, ensure_ascii=False)[:200]})]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class HistoryManager:
    """Manages conversation history to stay within context limits.

    Strategy: when total tokens exceed max_context_tokens:
    1. Truncate tool results in older messages (threshold: 2000 chars)
    2. If still over, collect messages to drop, generate LLM summary, then drop
    3. Insert summary as <context-summary> that survives future compactions
    4. Notify agent that compaction happened
    """

    _COMPACTION_RATIOS = [0.60, 0.50, 0.40, 0.35]

    def __init__(self, max_context_tokens: int = 100000):
        self.max_context_tokens = max_context_tokens
        self._messages: List[Dict[str, Any]] = []
        self._full_log: List[Dict[str, Any]] = []
        self._last_compacted_from = None
        self._last_compacted_to = None
        self._actual_input_tokens = None
        self._last_inflation_ratio = 1.0  # Preserve inflation ratio across compactions
        self._summarizer: Optional[Callable[[str], str]] = None
        self._scorer: Optional[Callable[[List[Dict[str, Any]]], List[int]]] = None
        self._accumulated_summary: str = ""
        self._compaction_anchors: List[str] = []
        self._compaction_happened = False
        self._compaction_count = 0
        self._plan_summary_fn: Optional[Callable[[], str]] = None
        self._pre_compaction_hook: Optional[Callable[[List[Dict[str, Any]]], None]] = None

    def set_summarizer(self, callback: Callable[[str], str]):
        """Inject LLM summarization callback. Signature: (text) -> summary_string."""
        self._summarizer = callback

    def set_scorer(self, callback: Callable[[List[Dict[str, Any]]], List[int]]):
        """Inject LLM scoring callback for drop priority.

        Signature: (messages) -> list of scores (0-10, higher = more valuable to keep).
        Called during full compaction to decide which messages to drop first.
        """
        self._scorer = callback

    def set_plan_summary_fn(self, callback: Callable[[], str]):
        """Inject a callback that returns the current plan state as a string.

        Called during compaction to include plan context in the summary so the
        agent can recover its working state after context is dropped.
        Signature: () -> str
        """
        self._plan_summary_fn = callback

    def set_compaction_anchors(self, anchors: List[str]):
        """Set anchors that MUST be preserved in the next compaction summary."""
        self._compaction_anchors = anchors[:10]

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages

    @property
    def full_log(self) -> List[Dict[str, Any]]:
        """Complete uncompacted message history for export/archival."""
        return self._full_log

    @property
    def compaction_happened(self) -> bool:
        """True if the last get_messages() call triggered compaction."""
        return self._compaction_happened

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def last_compaction_ratio(self) -> Optional[float]:
        """The ratio used in the most recent compaction, or None."""
        if self._compaction_count == 0:
            return None
        idx = min(self._compaction_count - 1, len(self._COMPACTION_RATIOS) - 1)
        return self._COMPACTION_RATIOS[idx]

    def get_context_pressure(self) -> float:
        """Return current context usage as a ratio (0.0 to 1.0+)."""
        if self.max_context_tokens <= 0:
            return 0.0
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        total = max(estimated, actual)
        return total / self.max_context_tokens

    def _get_inflation_ratio(self) -> float:
        """Return EMA-smoothed ratio of actual API tokens to local estimate.

        Uses exponential moving average to avoid single-point spikes (e.g. from
        skill load/unload changing system prompt size). Outliers (ratio > 3.0)
        are discarded to prevent corruption.
        """
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        if actual > 0 and estimated > 0:
            current_ratio = max(actual / estimated, 1.0)
            if current_ratio > 3.0:
                return self._last_inflation_ratio
            self._last_inflation_ratio = 0.7 * self._last_inflation_ratio + 0.3 * current_ratio
            return self._last_inflation_ratio
        return self._last_inflation_ratio

    def force_compact(self, target_ratio: float = 0.50, base_limit: int = None) -> bool:
        """Force compaction to a target ratio. Returns True if compaction occurred.

        Args:
            target_ratio: Target ratio (0.0-1.0) of the base limit
            base_limit: Base limit to calculate target from. If None, uses self.max_context_tokens.
                        When recovering from overflow, pass the actual token count that triggered the error.
        """
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        current = max(estimated, actual)
        inflation = self._get_inflation_ratio()

        # Use actual limit if provided (for overflow recovery), otherwise use configured limit
        effective_limit = base_limit if base_limit is not None else self.max_context_tokens

        # Target in *real* tokens, then deflate to local-estimate space
        real_target = int(effective_limit * target_ratio)
        local_target = int(real_target / inflation)

        if current <= real_target:
            return False


        # Scale keep_recent with target_ratio — more aggressive ratio = fewer kept
        if target_ratio <= 0.25:
            keep_recent = min(4, max(len(self._messages) - 2, 1))
        elif target_ratio <= 0.35:
            keep_recent = min(8, max(len(self._messages) - 2, 1))
        else:
            keep_recent = min(KEEP_RECENT, max(len(self._messages) - 2, 1))

        result = []
        for i, msg in enumerate(self._messages):
            is_recent = (i >= len(self._messages) - keep_recent)
            if msg.get("role") == "system":
                result.append(msg)
            elif is_recent:
                # For very aggressive compaction, truncate even recent messages
                if target_ratio <= 0.25:
                    result.append(_truncate_message(msg, max_chars=800))
                else:
                    result.append(msg)
            else:
                result.append(_truncate_message(msg))

        new_estimated = sum(_message_tokens(m) for m in result)
        if new_estimated > local_target:
            to_drop, to_keep = _collect_droppable(result, local_target, scorer=self._scorer)
            if to_drop and self._summarizer:
                summary_text = self._build_summary_input(to_drop, self._compaction_anchors)
                # Append plan state so the agent can recover after compaction
                if self._plan_summary_fn:
                    try:
                        plan_ctx = self._plan_summary_fn()
                        if plan_ctx:
                            summary_text += (
                                f"\n\n## PLAN STATE (MUST PRESERVE IN SUMMARY):\n{plan_ctx}"
                            )
                    except Exception:
                        pass
                self._compaction_anchors = []
                try:
                    new_summary = self._summarizer(summary_text)
                    self._merge_summary(new_summary)
                except Exception:
                    pass  # Drop without summary if summarizer fails
            result = to_keep

        result = self._inject_summary(result)
        self._messages = result
        self._compaction_count += 1
        final_estimated = sum(_message_tokens(m) for m in self._messages)
        self._last_compacted_from = estimated
        self._last_compacted_to = final_estimated
        # Reset actual tokens — stale value would mislead next compaction attempt
        self._actual_input_tokens = None
        display.context_compacted(
            estimated, final_estimated,
            compaction_num=self.compaction_count,
            ratio=target_ratio,
        )
        return True

    def append(self, message: Dict[str, Any]):
        self._messages.append(message)
        self._full_log.append(message)
        # Cap _full_log to prevent unbounded memory growth in long sessions
        _FULL_LOG_MAX = 2000
        if len(self._full_log) > _FULL_LOG_MAX:
            # Keep the most recent messages; drop oldest
            self._full_log = self._full_log[-_FULL_LOG_MAX:]

    def set_system_prompt(self, content: str):
        """Replace or prepend the system message."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def compact_intra_turn(self, keep_last: int = 6):
        """Graduated in-place compression of older tool results.

        Instead of replacing all old messages with a lossy summary, this method
        truncates individual tool results based on their content type and value.
        Messages are preserved (the agent still sees WHAT it did and WHAT it found),
        but verbose output is trimmed to key lines.

        This avoids the re-read problem: the agent retains enough context to know
        what files contain and what commands produced, without needing to re-execute.

        Only called when context pressure > 0.70.
        """
        turn_start = self._find_turn_start()
        turn_messages = self._messages[turn_start:]

        if len(turn_messages) <= keep_last + 2:
            return False

        # Target: reduce pressure to ~0.50 of budget
        current_tokens = sum(_message_tokens(m) for m in self._messages)
        target_tokens = int(self.max_context_tokens * 0.50)
        tokens_to_free = current_tokens - target_tokens
        if tokens_to_free <= 0:
            return False

        # Only compress messages BEFORE the keep_last window
        compress_end = len(self._messages) - keep_last
        compress_start = turn_start
        if compress_end <= compress_start:
            return False

        freed = 0
        # Pass 1: Compress low-value results (install logs, ls, find, repetitive output)
        for i in range(compress_start, compress_end):
            if freed >= tokens_to_free:
                break
            freed += self._compress_message_graduated(i, level=1)

        # Pass 2: If still need more, compress medium-value results (file contents, shell output)
        if freed < tokens_to_free:
            for i in range(compress_start, compress_end):
                if freed >= tokens_to_free:
                    break
                freed += self._compress_message_graduated(i, level=2)

        if freed > 0:
            display._print(display.dim(
                f"📦 Intra-turn compact: freed ~{freed // 1000}k tokens"
            ))
        return freed > 0

    def _compress_message_graduated(self, idx: int, level: int) -> int:
        """Compress a single message in-place at the given aggressiveness level.

        Level 1: Only compress clearly low-value content (install logs, directory listings)
        Level 2: Also compress file contents and shell output to key lines

        Returns estimated tokens freed.
        """
        msg = self._messages[idx]
        old_tokens = _message_tokens(msg)
        content = msg.get("content", "")

        if msg.get("role") == "assistant":
            # Assistant reasoning is cheap and valuable — never compress at level 1
            if level >= 2 and isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if len(text) > 800:
                            # Keep first and last paragraphs of long reasoning
                            lines = text.splitlines()
                            if len(lines) > 10:
                                kept = lines[:4] + ["[...]"] + lines[-4:]
                                new_blocks.append({**block, "text": "\n".join(kept)})
                            else:
                                new_blocks.append(block)
                        else:
                            new_blocks.append(block)
                    else:
                        new_blocks.append(block)
                self._messages[idx] = {**msg, "content": new_blocks}
            return old_tokens - _message_tokens(self._messages[idx])

        # Tool result messages
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str) and len(inner) > 300:
                        compressed = self._compress_tool_result(inner, level)
                        new_blocks.append({**block, "content": compressed})
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            self._messages[idx] = {**msg, "content": new_blocks}
        elif isinstance(content, str) and len(content) > 300 and msg.get("role") == "tool":
            self._messages[idx] = {**msg, "content": self._compress_tool_result(content, level)}

        return old_tokens - _message_tokens(self._messages[idx])

    @staticmethod
    def _compress_tool_result(content: str, level: int) -> str:
        """Compress a tool result string based on content type and level.

        Preserves enough information to avoid re-reads:
        - File paths and structure are always kept
        - Error messages are always kept in full
        - Verbose output (install logs, build output) is aggressively trimmed
        """
        lines = content.splitlines()
        num_lines = len(lines)

        # Never compress short content
        if num_lines <= 8 or len(content) <= 400:
            return content

        # Always preserve errors fully
        has_error = any(kw in content for kw in
                       ("Error", "ERROR", "Traceback", "FAILED", "fatal", "Exception"))
        if has_error:
            if num_lines <= 30:
                return content
            # Keep error context: first 5 + last 15 lines (error usually at end)
            head = "\n".join(lines[:5])
            tail = "\n".join(lines[-15:])
            return f"{head}\n[... {num_lines - 20} lines omitted ...]\n{tail}"

        # Detect content type and compress accordingly
        content_lower = content[:500].lower()

        # Install/build logs — very low value, keep only outcome
        if level >= 1 and any(kw in content_lower for kw in
                              ("installing", "collecting", "downloading",
                               "successfully installed", "requirement already",
                               "building wheel", "running setup", "compiling")):
            # Keep first 2 lines (command) + last 3 lines (result)
            head = "\n".join(lines[:2])
            tail = "\n".join(lines[-3:])
            return f"{head}\n[... {num_lines - 5} lines of install/build output ...]\n{tail}"

        # Directory listings (find/ls output) — keep paths but trim
        if level >= 1 and num_lines > 20 and all(
            l.strip().startswith(("/", "./", "total ")) or not l.strip()
            for l in lines[:10] if l.strip()
        ):
            # Keep first 15 and last 5 paths
            head = "\n".join(lines[:15])
            tail = "\n".join(lines[-5:])
            return f"{head}\n[... {num_lines - 20} more entries ...]\n{tail}"

        # Level 2: General long output — keep head + tail
        if level >= 2:
            if num_lines > 20:
                head = "\n".join(lines[:8])
                tail = "\n".join(lines[-8:])
                return f"{head}\n[... {num_lines - 16} lines omitted ...]\n{tail}"

        return content

    def _find_turn_start(self) -> int:
        """Find the index of the last real user message (not a tool_result)."""
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            # Real user message = string content (not tool_result blocks)
            if isinstance(content, str) and "<turn-progress>" not in content:
                # Check it's not a system injection
                if not content.startswith("[") and not content.startswith("<"):
                    return i
            # Also check for list content that's NOT tool_result
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if not has_tool_result:
                    return i
        return 0

    @staticmethod
    def _extract_turn_progress(messages: List[Dict[str, Any]]) -> str:
        """Extract structured progress summary from messages without LLM."""
        actions = []
        findings = []
        errors = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                # Extract key decisions/statements from assistant text
                if isinstance(content, str) and content.strip():
                    lines = content.strip().splitlines()
                    # Keep last meaningful line as the conclusion
                    for line in reversed(lines):
                        line = line.strip()
                        if line and len(line) > 10 and not line.startswith("```"):
                            findings.append(line[:150])
                            break
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})
                            if tool_name == "shell":
                                cmd = tool_input.get("command", "")[:100]
                                actions.append(f"shell: {cmd}")
                            elif tool_name == "read_file":
                                actions.append(f"read: {tool_input.get('path', '')}")
                            elif tool_name == "monitor":
                                f = tool_input.get("file", tool_input.get("command", ""))
                                actions.append(f"monitor: {f}")
                            else:
                                actions.append(f"{tool_name}")

            elif role == "user" and isinstance(content, list):
                # Extract tool results
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_text = block.get("content", "")
                        if isinstance(result_text, str):
                            # Check for errors
                            if any(kw in result_text for kw in
                                   ("ERROR", "Error", "Traceback", "FAILED")):
                                err_lines = result_text.strip().splitlines()
                                errors.append(err_lines[-1][:150] if err_lines else "error")
                            # Extract last few lines as context
                            lines = result_text.strip().splitlines()
                            if lines:
                                last = lines[-1].strip()[:100]
                                if last and "..." not in last:
                                    findings.append(last)

        # Build compact summary
        parts = []
        if actions:
            # Deduplicate consecutive identical actions
            deduped = []
            for a in actions:
                if not deduped or deduped[-1] != a:
                    deduped.append(a)
            parts.append("Actions: " + " → ".join(deduped[-8:]))
        if errors:
            parts.append("Errors: " + "; ".join(errors[-3:]))
        if findings:
            # Keep only last 3 unique findings
            seen = set()
            unique = []
            for f in reversed(findings):
                if f not in seen:
                    seen.add(f)
                    unique.append(f)
                if len(unique) >= 3:
                    break
            parts.append("State: " + " | ".join(reversed(unique)))

        return "\n".join(parts) if parts else "No significant progress recorded."

    def report_actual_tokens(self, input_tokens: int):
        """Feed back the actual input_tokens from the API response."""
        self._actual_input_tokens = input_tokens

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages, compacting with LLM summary if over budget."""
        # Only age old tool results when context pressure is meaningful.
        # With a 200K budget, aggressively truncating at 10% usage causes
        # the agent to re-read files it already read — a net token loss.
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        inflation = self._get_inflation_ratio()
        # Use inflation-adjusted estimate to predict real API cost
        predicted = max(int(estimated * inflation), actual)
        total = predicted
        self._last_compacted_from = None
        self._last_compacted_to = None
        self._compaction_happened = False

        # Pressure-gated aging: truncate old tool results earlier to prevent token bloat
        pressure = total / self.max_context_tokens if self.max_context_tokens > 0 else 0
        if pressure > 0.35:
            before_est = estimated
            self._messages = _age_tool_results(self._messages, keep_recent=AGING_WINDOW)
            estimated = sum(_message_tokens(m) for m in self._messages)
            freed = before_est - estimated
            if freed > 1000:
                display._print(display.dim(
                    f"📦 Aging: freed ~{freed // 1000}k tokens (pressure {int(pressure * 100)}%)"
                ))
            predicted = max(int(estimated * inflation), actual)
            total = predicted

        if total <= self.max_context_tokens:
            return _validate_tool_pairs(list(self._messages))

        original_total = total

        # Dynamic target: compress harder each successive time
        ratio_idx = min(self._compaction_count, len(self._COMPACTION_RATIOS) - 1)
        ratio = self._COMPACTION_RATIOS[ratio_idx]
        real_target = int(self.max_context_tokens * ratio)
        local_target = int(real_target / inflation)

        keep_recent = min(KEEP_RECENT, max(len(self._messages) - 2, 1))

        # Step 1: truncate old tool results (threshold raised to 2000 chars)
        result = []
        for i, msg in enumerate(self._messages):
            is_recent = (i >= len(self._messages) - keep_recent)
            if msg.get("role") == "system":
                result.append(msg)
            elif is_recent:
                result.append(msg)
            else:
                result.append(_truncate_message(msg))

        new_estimated = sum(_message_tokens(m) for m in result)

        # Step 2: if still over target, collect messages to drop and summarize them
        if new_estimated > local_target:
            to_drop, to_keep = _collect_droppable(result, local_target, scorer=self._scorer)

            # Pre-compaction hook: extract key info from to_drop before losing them
            if to_drop and self._pre_compaction_hook:
                try:
                    self._pre_compaction_hook(to_drop)
                except Exception:
                    pass

            if to_drop and self._summarizer:
                summary_text = self._build_summary_input(to_drop, self._compaction_anchors)
                self._compaction_anchors = []
                try:
                    new_summary = self._summarizer(summary_text)
                    self._merge_summary(new_summary)
                except Exception:
                    pass  # Drop without summary if summarizer fails

            result = to_keep

        # Step 3: inject accumulated summary after system message
        result = self._inject_summary(result)

        # Step 4: hard ceiling — if still over budget, aggressively truncate recent messages
        new_estimated = sum(_message_tokens(m) for m in result)
        if new_estimated > local_target:
            result = self._hard_ceiling_truncate(result, local_target)

        new_estimated = sum(_message_tokens(m) for m in result)
        self._messages = result
        # Keep _actual_input_tokens to preserve inflation ratio memory
        self._last_compacted_from = original_total
        self._last_compacted_to = new_estimated
        self._compaction_happened = True
        self._compaction_count += 1
        display.context_compacted(
            original_total, new_estimated,
            compaction_num=self._compaction_count,
            ratio=ratio,
        )
        return _validate_tool_pairs(list(self._messages))

    def _build_summary_input(self, messages: List[Dict[str, Any]], anchors: Optional[List[str]] = None) -> str:
        """Build text input for the summarizer from messages about to be dropped."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = _extract_text(msg)
            if text.strip():
                parts.append(f"[{role}] {text}")
        combined = "\n---\n".join(parts)
        if len(combined) > 32000:
            combined = combined[:32000] + "\n[... truncated for summarization ...]"
        anchor_section = ""
        if anchors:
            anchor_section = (
                "\n\nMANDATORY ANCHORS — these MUST appear verbatim in your summary:\n"
                + "\n".join(f"- {a}" for a in anchors[:10])
                + "\n"
            )
        return f"{SUMMARIZE_PROMPT}{anchor_section}\n\n---\nConversation segment:\n{combined}"

    def _merge_summary(self, new_summary: str):
        """Merge new summary into accumulated summary, keeping total under limit.

        Uses dynamic limit: 5% of max_context_tokens (default 10K for 200K window).
        When over limit, fuses the two oldest sections via summarizer rather than
        dropping outright, preserving key decisions from early context.
        """
        dynamic_limit = max(MAX_SUMMARY_TOKENS, int(self.max_context_tokens * 0.05))
        if self._accumulated_summary:
            merged = f"{self._accumulated_summary}\n\n---\n\n{new_summary}"
        else:
            merged = new_summary
        _max_merge_iters = 20  # Safety cap to prevent infinite loop
        _merge_iter = 0
        while _estimate_tokens(merged) > dynamic_limit and "\n\n---\n\n" in merged:
            _merge_iter += 1
            if _merge_iter > _max_merge_iters:
                # Keep only the most recent section
                sections = merged.split("\n\n---\n\n")
                merged = sections[-1]
                break
            # Fuse the two oldest sections instead of dropping
            sections = merged.split("\n\n---\n\n")
            if len(sections) <= 2:
                # Only 2 sections left — drop the oldest as last resort
                merged = sections[-1]
                break
            oldest_two = sections[0] + "\n" + sections[1]
            if self._summarizer and _estimate_tokens(oldest_two) > 500:
                try:
                    fused = self._summarizer(
                        f"Fuse these two summaries into one concise summary:\n\n{oldest_two}"
                    )
                    sections = [fused] + sections[2:]
                except Exception:
                    sections = sections[1:]
            else:
                sections = sections[1:]
            merged = "\n\n---\n\n".join(sections)
        self._accumulated_summary = merged

    def _inject_summary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert or update the <context-summary> message after the system message."""
        if not self._accumulated_summary:
            return messages

        summary_msg = {
            "role": "user",
            "content": f"<context-summary>\n{self._accumulated_summary}\n</context-summary>"
        }

        result = []
        inserted = False
        for msg in messages:
            # Remove any existing context-summary message
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.startswith("<context-summary>"):
                    continue
            result.append(msg)
            # Insert after system message
            if not inserted and msg.get("role") == "system":
                result.append(summary_msg)
                inserted = True

        if not inserted:
            result.insert(0, summary_msg)

        return result

    def _hard_ceiling_truncate(self, messages: List[Dict[str, Any]], local_target: int) -> List[Dict[str, Any]]:
        """Emergency truncation when normal compaction fails to reach target.

        Keeps system/summary messages at the front, then fills from the most
        recent messages backward. Truncation budget per message is based on
        heuristic value score: high-value messages get more space.
        """
        head = []
        body = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            is_summary = isinstance(content, str) and content.startswith("<context-summary>")
            if role == "system" or is_summary:
                head.append(msg)
            else:
                body.append(msg)

        head_tokens = sum(_message_tokens(m) for m in head)
        budget = local_target - head_tokens

        kept = []
        used = 0
        for msg in reversed(body):
            score = _heuristic_score(msg)
            if score >= 7:
                char_limit = 1200
            elif score >= 4:
                char_limit = 600
            else:
                char_limit = 200
            truncated = _truncate_message(msg, max_chars=char_limit)
            t = _message_tokens(truncated)
            if used + t > budget:
                break
            kept.append(truncated)
            used += t

        kept.reverse()
        result = head + kept
        return result

    def clear(self):
        self._messages.clear()
        self._full_log.clear()
        self._accumulated_summary = ""


def _extract_error_tail(content: str, max_chars: int = 1500) -> str:
    """Extract error/traceback portion from tool output for preservation during truncation."""
    lines = content.splitlines()
    error_start = -1
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in ('traceback', 'error:', 'exception:', 'fatal:', 'failed')):
            if error_start < 0:
                error_start = i
    if error_start >= 0:
        error_text = "\n".join(lines[error_start:])
        if len(error_text) > max_chars:
            error_text = error_text[-max_chars:]
        return error_text
    return ""


def _truncate_message(msg: Dict[str, Any], max_chars: int = TRUNCATE_THRESHOLD) -> Dict[str, Any]:
    """Replace long content in tool results with a summary placeholder.
    Preserves error/traceback content to avoid losing diagnostic information."""
    content = msg.get("content", "")

    if isinstance(content, str) and len(content) > max_chars:
        role = msg.get("role", "")
        if role == "tool":
            error_tail = _extract_error_tail(content)
            if error_tail:
                return {**msg, "content": f"[truncated tool result, {len(content)} chars. Error preserved:]\n{error_tail}"}
            return {**msg, "content": f"[truncated tool result, {len(content)} chars]"}

    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > max_chars:
                    error_tail = _extract_error_tail(inner)
                    if error_tail:
                        new_blocks.append({**block, "content": f"[truncated tool result, {len(inner)} chars. Error preserved:]\n{error_tail}"})
                    else:
                        new_blocks.append({**block, "content": f"[truncated tool result, {len(inner)} chars]"})
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}

    return msg


def _merge_user_messages(msg1: Dict[str, Any], msg2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two user messages into one, handling both string and list content."""
    c1 = msg1.get("content", "")
    c2 = msg2.get("content", "")
    blocks1 = c1 if isinstance(c1, list) else [{"type": "text", "text": c1}]
    blocks2 = c2 if isinstance(c2, list) else [{"type": "text", "text": c2}]
    return {"role": "user", "content": blocks1 + blocks2}


def _validate_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove orphaned tool_use or tool_result blocks to prevent API 400 errors.

    Steps:
    1. Merge consecutive tool_result user messages (Anthropic format splits them
       into separate messages, but they belong to the same assistant turn).
    2. Validate that each assistant+tool_use has a following tool_result,
       and each tool_result has a preceding assistant+tool_use.
    3. Merge remaining consecutive user messages for role alternation.
    """
    result = list(messages)

    # Step 1: Merge consecutive tool_result user messages only.
    # This handles the case where multiple tool_results from one assistant turn
    # are stored as separate user messages (Anthropic format).
    i = 0
    while i < len(result) - 1:
        if (_is_tool_result(result[i]) and _is_tool_result(result[i + 1])
                and result[i].get("role") == "user" and result[i + 1].get("role") == "user"):
            result[i] = _merge_user_messages(result[i], result[i + 1])
            result.pop(i + 1)
        else:
            i += 1

    # Step 2: Remove orphaned tool_use or tool_result
    i = 0
    while i < len(result):
        msg = result[i]
        if msg.get("role") == "assistant" and _has_tool_use(msg):
            if i + 1 >= len(result) or not _is_tool_result(result[i + 1]):
                result.pop(i)
                continue
        elif _is_tool_result(msg):
            if i == 0 or not (result[i - 1].get("role") == "assistant" and _has_tool_use(result[i - 1])):
                result.pop(i)
                continue
        i += 1

    # Step 3: Merge remaining consecutive user messages (role alternation)
    i = 0
    while i < len(result) - 1:
        if result[i].get("role") == "user" and result[i + 1].get("role") == "user":
            result[i] = _merge_user_messages(result[i], result[i + 1])
            result.pop(i + 1)
        else:
            i += 1

    return result


def _collect_droppable(messages: List[Dict[str, Any]], budget: int,
                       scorer: Optional[Callable] = None):
    """Separate messages into (to_drop, to_keep) lists to fit within budget.

    If a scorer callback is available, uses LLM-based value scoring to drop
    low-value messages first (regardless of age). Falls back to FIFO if no scorer.

    Keeps system messages and context-summary messages unconditionally.
    """
    total = sum(_message_tokens(m) for m in messages)
    if total <= budget:
        return [], messages

    # Identify droppable candidates (not system, not context-summary)
    candidates = []  # (index, msg, tokens)
    protected = []   # (index, msg) — always kept
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        is_system = msg.get("role") == "system"
        is_summary = isinstance(content, str) and content.startswith("<context-summary>")
        if is_system or is_summary:
            protected.append((i, msg))
        else:
            candidates.append((i, msg, _message_tokens(msg)))

    if not candidates:
        return [], messages

    # Score candidates: LLM if available, else heuristic
    if scorer and len(candidates) > 4:
        try:
            candidate_msgs = [c[1] for c in candidates]
            scores = scorer(candidate_msgs)
            if len(scores) == len(candidates):
                # Attach scores
                scored = [(candidates[j][0], candidates[j][1], candidates[j][2], scores[j])
                          for j in range(len(candidates))]
            else:
                scored = [(c[0], c[1], c[2], _heuristic_score(c[1])) for c in candidates]
        except Exception:
                scored = [(c[0], c[1], c[2], _heuristic_score(c[1])) for c in candidates]
    else:
        scored = [(c[0], c[1], c[2], _heuristic_score(c[1])) for c in candidates]

    # Sort by score ascending (drop lowest-value first)
    scored.sort(key=lambda x: (x[3], x[0]))

    # Drop until we're under budget, respecting tool_use/tool_result pairs
    to_drop_indices = set()
    tokens_freed = 0
    tokens_to_free = total - budget

    for idx, msg, tokens, score in scored:
        if tokens_freed >= tokens_to_free:
            break
        # If this is an assistant with tool_use, also drop the paired tool_result
        if msg.get("role") == "assistant" and _has_tool_use(msg):
            pair_idx = idx + 1
            pair_tokens = 0
            if pair_idx < len(messages) and _is_tool_result(messages[pair_idx]):
                pair_tokens = _message_tokens(messages[pair_idx])
                to_drop_indices.add(pair_idx)
            to_drop_indices.add(idx)
            tokens_freed += tokens + pair_tokens
        elif _is_tool_result(msg):
            # Check if the preceding assistant is already being dropped
            if idx - 1 >= 0 and messages[idx - 1].get("role") == "assistant":
                to_drop_indices.add(idx - 1)
                tokens_freed += _message_tokens(messages[idx - 1])
            to_drop_indices.add(idx)
            tokens_freed += tokens
        else:
            to_drop_indices.add(idx)
            tokens_freed += tokens

    to_drop = [messages[i] for i in sorted(to_drop_indices)]
    to_keep = [messages[i] for i in range(len(messages)) if i not in to_drop_indices]

    return to_drop, to_keep


def _heuristic_score(msg: Dict[str, Any]) -> int:
    """Score a message's value heuristically (0-10, higher = more valuable).

    Used as fallback when LLM scorer is unavailable.
    Higher score = more likely to be kept during compaction.
    """
    content = _extract_text(msg)
    if not content:
        return 1

    # Errors are high value (agent needs to remember what failed)
    if any(kw in content for kw in ("Error", "ERROR", "Traceback", "FAILED", "Exception")):
        return 9

    # Decisions and reasoning from assistant are high value
    if msg.get("role") == "assistant":
        if any(kw in content for kw in ("because", "decision", "approach", "strategy",
                                         "found that", "discovered", "the issue is")):
            return 8
        # Successful write/edit operations — keep (shows what was changed)
        if any(kw in content for kw in ("Successfully edited", "Wrote ", "write_file", "edit_file")):
            return 8
        return 5

    # User messages — always high value
    if msg.get("role") == "user":
        return 9

    # Tool results — score by content type
    lower = content[:500].lower()

    # Empty or near-empty shell output — very low value
    if len(content.strip()) < 10:
        return 1

    # Install/build logs — low value
    if any(kw in lower for kw in ("installing", "collecting", "downloading",
                                   "successfully installed", "building wheel")):
        return 2

    # plan_status, memory_list verbose output — low value (easily re-fetched)
    if any(kw in lower for kw in ("plan:", "progress:", "showing", "entries")):
        if "plan:" in lower and "step" in lower:
            return 3

    # Successful edit/write results — high value (shows what changed)
    if any(kw in lower for kw in ("successfully edited", "wrote ", "created ")):
        return 8

    # File content — medium-high (expensive to re-read but can be re-read)
    if any(kw in lower for kw in ("import ", "def ", "class ", "from ", "---\n",
                                   "export ", "#include", "\"type\":")):
        return 6

    # Directory listings — low-medium
    lines = content.splitlines()
    if len(lines) > 5 and sum(1 for l in lines[:10] if l.strip().startswith(("/", "./"))) > 5:
        return 3

    # Short results (likely conclusions) — medium-high
    if len(content) < 200:
        return 6

    return 4
