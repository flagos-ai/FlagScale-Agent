"""ContextPressureGuard — monitors context window pressure and triggers compaction."""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


class ContextPressureGuard(Guard):
    """Monitors context pressure and triggers warnings / forced compaction.

    Activates in all states — context pressure is always relevant.
    Enhanced: at soft threshold, scans recent context for memory candidates.
    """

    name = "context_pressure"
    priority = 40
    activate_on_states = {AgentState.EXECUTING, AgentState.PLANNING, AgentState.REVIEWING}

    # ── Thresholds ──
    _SOFT_THRESHOLD = 0.75
    _HARD_THRESHOLD = 0.85
    _FORCE_THRESHOLD = 0.95

    def __init__(self):
        self._soft_warned: bool = False
        self._hard_warned: bool = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        pressure = ctx.context_pressure

        if pressure >= self._FORCE_THRESHOLD:
            return GuardVerdict.compact(
                reason=f"pressure at {pressure:.0%}",
            )

        if pressure >= self._HARD_THRESHOLD and not self._hard_warned:
            self._hard_warned = True
            candidates = self._suggest_memory_candidates(ctx)
            suggestion = ""
            if candidates:
                suggestion = " Suggested to memorize: " + "; ".join(candidates[:3])
            return GuardVerdict.inject(
                f"[ContextPressure] Context at {pressure:.0%}. "
                f"Write key findings to memory NOW — compaction is imminent.{suggestion}",
                reason=f"hard threshold reached: {pressure:.0%}",
            )

        if pressure >= self._SOFT_THRESHOLD and not self._soft_warned:
            self._soft_warned = True
            candidates = self._suggest_memory_candidates(ctx)
            suggestion = ""
            if candidates:
                suggestion = " Candidates: " + "; ".join(candidates[:3])
            return GuardVerdict.inject(
                f"[ContextPressure] Context at {pressure:.0%}. "
                f"Consider writing intermediate results to memory.{suggestion}",
                reason=f"soft threshold reached: {pressure:.0%}",
            )

        return None

    def _suggest_memory_candidates(self, ctx: GuardContext) -> list[str]:
        """Scan recent tool results for memory-worthy content (heuristic, no LLM)."""
        candidates = []
        # Access recent messages from context if available
        recent_text = ctx.tool_result or ""
        if not recent_text and hasattr(ctx, "recent_outputs"):
            recent_text = str(ctx.recent_outputs)

        # Pattern 1: Error resolutions
        if re.search(r'(?:fixed|solved|resolved|workaround)', recent_text, re.IGNORECASE):
            candidates.append("error workaround found in recent output")

        # Pattern 2: Path discoveries
        paths = re.findall(r'/[\w/.-]{15,}(?:\.py|\.yaml|\.json|\.pt|\.safetensors)', recent_text)
        if paths:
            candidates.append(f"path: {paths[0][:80]}")

        # Pattern 3: Configuration decisions
        if re.search(r'(?:changed|set|using)\s+(?:tp|dp|pp|batch|lr|seq.len)', recent_text, re.IGNORECASE):
            candidates.append("configuration decision")

        # Pattern 4: Numerical results
        if re.search(r'(?:loss|throughput|tokens.per.sec)\s*[:=]\s*[\d.]+', recent_text, re.IGNORECASE):
            candidates.append("training metrics")

        return candidates

    def reset_turn(self):
        # Do NOT reset warned flags here — reset_turn is called per iteration.
        # Context pressure warnings should fire at most once per session threshold crossing.
        pass
