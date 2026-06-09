"""CompactContext tool — allows the agent to proactively trigger context compaction."""

from flagscale_agent.react.tools.base import Tool


class CompactContextTool(Tool):
    name = "compact_context"
    description = (
        "Proactively compact the conversation context to free up space. "
        "Use when context is getting long and you need room for more work. "
        "Specify target_ratio (0.3-0.7) to control how aggressively to compact. "
        "Lower ratio = more aggressive compaction. Default: 0.5 (keep ~50%)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_ratio": {
                "type": "number",
                "description": "Target ratio of context to keep (0.3-0.7). Default: 0.5",
            },
            "reason": {
                "type": "string",
                "description": "Why you're compacting (logged for debugging).",
            },
        },
        "required": [],
    }

    def __init__(self, history_manager):
        self._history = history_manager

    def execute(self, **kwargs) -> str:
        target_ratio = kwargs.get("target_ratio", 0.5)
        reason = kwargs.get("reason", "proactive compaction")

        # Clamp ratio to safe range
        target_ratio = max(0.3, min(0.7, target_ratio))

        # Check if compaction is actually needed
        from flagscale_agent.react.history import _message_tokens
        estimated = sum(_message_tokens(m) for m in self._history._messages)
        if estimated < 5000:
            return "Context is already small (< 5000 tokens). No compaction needed."

        # Execute compaction
        compacted = self._history.force_compact(target_ratio=target_ratio)

        if compacted:
            new_estimated = sum(_message_tokens(m) for m in self._history._messages)
            return (
                f"Context compacted successfully. Reason: {reason}\n"
                f"Before: ~{estimated} tokens → After: ~{new_estimated} tokens "
                f"(ratio: {target_ratio}, saved ~{estimated - new_estimated} tokens)"
            )
        else:
            return (
                f"Compaction skipped — context ({estimated} tokens) is already "
                f"below target ({int(self._history.max_context_tokens * target_ratio)} tokens)."
            )
