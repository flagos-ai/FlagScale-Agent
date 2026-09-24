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

"""AgentKernel — minimal event loop core.

Replaces the monolithic _react_loop() in agent.py.

Responsibilities:
- LLM call + retry on context overflow
- Guard pre/post checks
- Tool execution dispatch
- State machine transitions
- Token accounting

Everything else (session, history, tools, prompts) is injected via dependencies.
"""

from __future__ import annotations

import re
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from flagscale_agent.react.guard import GuardContext, GuardRegistry, GuardVerdict
from flagscale_agent.react import display


@dataclass
class KernelDeps:
    """All external dependencies the Kernel needs — injected, not imported."""

    provider: Any                          # LLM provider
    history: Any                           # HistoryManager
    tool_registry: Any                     # ToolRegistry
    judge: Any                             # Judge
    guard_registry: GuardRegistry
    config: Any                            # AgentConfig
    display: Any                           # display module
    get_schemas_fn: Callable               # () -> list[dict]
    inject_message_fn: Callable            # (msg: str) -> None  [block/escalate only]
    append_tool_results_fn: Callable       # (results: list) -> None
    format_tool_result_fn: Callable        # (id, result) -> dict
    execute_tools_fn: Callable             # (tool_calls) -> list[str]
    is_context_limit_error_fn: Callable    # (exc) -> bool
    append_advisory_fn: Callable | None = None  # (msg: str) -> None [soft inject → tool_result]
    call_llm_fn: Callable | None = None    # (messages, schemas) -> (response, usage)
    task_plan: Any = None                  # TaskPlan (optional)
    on_response_fn: Callable | None = None  # (response) -> None, called after LLM response appended
    on_tool_results_fn: Callable | None = None  # (tool_calls, results) -> None, called after tool exec


@dataclass
class KernelResult:
    """Result of one kernel run (one user turn)."""

    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed: float = 0.0
    interrupted: bool = False
    stop_reason: str = ""


class AgentKernel:
    """Minimal event loop. < 200 lines of logic.

    One instance per agent. Call run_turn() for each user message.
    """

    # Max consecutive completion-gate blocks with zero intervening progress
    # before the loop gives up (livelock breaker). Small: a genuinely diligent
    # agent creates a plan or supplies an override on the FIRST block, so any
    # value >1 gives it room to react while still catching a stuck re-emit loop
    # long before max_iter.
    MAX_CONSECUTIVE_COMPLETION_BLOCKS = 5

    def __init__(self, deps: KernelDeps):
        self.deps = deps
        self._interrupted = False
        self._continuation_count = 0
        self._consecutive_completion_blocks = 0

    def run_turn(self) -> KernelResult:
        """Run one ReAct turn (one user message → completion).

        Returns KernelResult with token stats and stop reason.
        """
        result = KernelResult()
        d = self.deps
        max_iter = d.config.max_iterations
        turn_start = time.time()

        self._interrupted = False
        self._continuation_count = 0  # Reset per turn
        # Circuit breaker for the completion gate: counts consecutive
        # completion-gate blocks with no intervening progress (no plan created,
        # no normal tool executed). A single-shot agent that keeps re-emitting a
        # bare [TASK_COMPLETE] with neither a plan nor an _override_reason would
        # otherwise livelock — the gate correctly blocks every time and the loop
        # `continue`s without incrementing _continuation_count, so it spins all
        # the way to max_iter (thousands of identical blocks). This breaks out
        # once the agent has clearly stalled on the gate.
        self._consecutive_completion_blocks = 0
        d.judge.reset_turn()
        d.guard_registry.reset_turn()

        _prev_handler = signal.getsignal(signal.SIGINT)

        def _sigint(signum, frame):
            if self._interrupted:
                signal.signal(signal.SIGINT, _prev_handler)
                raise KeyboardInterrupt
            self._interrupted = True
            d.display.interrupted()

        signal.signal(signal.SIGINT, _sigint)

        try:
            for iteration in range(max_iter):
                if self._interrupted:
                    break

                # Guard reset happens per-turn (at line 101), not per-iteration
                d.judge.reset_turn()

                schemas = d.get_schemas_fn()

                # ── Pre-guard checks ──
                ctx = self._build_ctx(tool_name="", tool_args={}, tool_result=None)
                verdict = d.guard_registry.check_pre(ctx)
                if verdict is not None:
                    blocked = self._apply_verdict(verdict, pre=True)
                    if blocked:
                        # Livelock breaker: the completion gate reads the LAST
                        # assistant message via _get_last_assistant_text(). Once a
                        # bare [TASK_COMPLETE] (no plan / no override) lands in
                        # history, this top-of-loop pre-guard re-fires the gate
                        # EVERY iteration on that stale text — and `continue`s
                        # before the LLM is ever called again, so the agent can
                        # never react. Left unchecked it spins to max_iter
                        # (~2000 identical blocks). Count consecutive completion-
                        # gate blocks and give up once clearly stalled.
                        if verdict.reason == "single_shot_completion_without_plan":
                            self._consecutive_completion_blocks += 1
                            if (self._consecutive_completion_blocks
                                    >= self.MAX_CONSECUTIVE_COMPLETION_BLOCKS):
                                result.stop_reason = (
                                    f"completion_gate_livelock: {verdict.reason}")
                                break
                        # Don't break — re-prompt LLM with the guard message in history
                        # Guard message was injected by _apply_verdict, now give LLM a chance to respond
                        if iteration >= max_iter - 1:
                            # Safety: if we're at max_iter, stop to avoid infinite loop
                            result.stop_reason = f"blocked_by_guard_at_max_iter: {verdict.reason}"
                            break
                        # Continue to next iteration — LLM will see the guard message
                        continue

                # ── LLM call ──
                d.display.thinking()
                messages = d.history.get_messages()
                self._t0 = time.time()

                try:
                    _call = d.call_llm_fn or (lambda m, s: d.provider.chat_stream(m, s))
                    response, usage = _call(messages, schemas)
                except KeyboardInterrupt:
                    d.display.interrupted()
                    self._interrupted = True
                    break
                except Exception as e:
                    # Context overflow should be prevented by ContextPressureGuard
                    # which prompts LLM to call evict/hard_reset before reaching this point
                    d.display.thinking_clear()
                    display.warn(f"LLM call failed: {e}")
                    result.stop_reason = f"llm_error: {e}"
                    break

                elapsed = time.time() - getattr(self, "_t0", time.time())
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                cache_read = usage.get("cache_read_input_tokens") or 0
                cache_create = usage.get("cache_creation_input_tokens") or 0
                total_in_tok = in_tok + cache_read + cache_create
                result.input_tokens += total_in_tok
                result.output_tokens += out_tok
                if total_in_tok:
                    d.history.report_actual_tokens(total_in_tok)

                d.display.llm_done(elapsed, total_in_tok, out_tok,
                                   cache_read_tokens=cache_read or None,
                                   cache_creation_tokens=cache_create or None)

                if self._interrupted:
                    break

                d.history.append(d.provider.format_assistant_message(response))

                if d.on_response_fn:
                    d.on_response_fn(response)

                # ── No tool calls → done ──
                if not response.get("tool_calls"):
                    result.iterations = iteration + 1
                    # Check for explicit stop signals in assistant response
                    assistant_text = ""
                    if isinstance(response.get("content"), str):
                        assistant_text = response["content"]
                    elif isinstance(response.get("content"), list):
                        assistant_text = "".join(
                            b.get("text", "") for b in response["content"]
                            if isinstance(b, dict) and b.get("type") == "text"
                        )

                    # ── Empty output defense: auto-retry up to 3 times ──
                    if not assistant_text.strip():
                        # Provider-agnostic reasoning detection: BOTH providers
                        # normalize reasoning into response["thinking"]
                        # (anthropic: thinking blocks; openai: reasoning_content).
                        # Do NOT rely on response["reasoning_only"] — that flag is
                        # only ever set by openai_provider, so an anthropic-compat
                        # endpoint (e.g. GLM) would never take the reasoning path.
                        thinking_text = response.get("thinking") or ""
                        was_reasoning_only = bool(thinking_text.strip())
                        was_thinking_capped = bool(response.get("thinking_capped"))
                        # SEPARATE BUDGETS: a cap-trip is a PACING event, not a
                        # failure — the model still produced (thinking) output
                        # and each retry is already bounded by the cap itself
                        # (~cap_budget thinking tokens per call). Budgeting it
                        # like a genuine empty output killed real runs: e.g.
                        # fibsqrt 2026-09-08 09:36 — cap fired 4x in one turn,
                        # retries 1/3..3/3 exhausted the shared budget, and the
                        # 4th cap-trip stopped the trial (reward 0) with the
                        # full solution design sitting in the discarded
                        # reasoning. Cap-retries get their own generous counter
                        # (per turn) instead of the 3-strike empty budget.
                        cap_retries = getattr(self, "_cap_retries", 0)
                        empty_retries = getattr(self, "_empty_output_retries", 0)
                        if was_thinking_capped:
                            if cap_retries >= 50:
                                self._cap_retries = 0
                                result.stop_reason = "thinking_cap_max_retries"
                                break
                            self._cap_retries = cap_retries + 1
                            d.display.warn(f"Thinking reached runtime cap (cap-retry {self._cap_retries}), forcing convergence...")
                        elif empty_retries < 3:
                            self._empty_output_retries = empty_retries + 1
                            if was_reasoning_only:
                                d.display.warn(f"Reasoning produced content but no visible output (retry {empty_retries + 1}/3), continuing...")
                            else:
                                d.display.warn(f"Empty LLM output (retry {empty_retries + 1}/3), auto-continuing...")
                        else:
                            # Exhausted empty-output budget (genuine empty /
                            # reasoning-only response, nothing to re-inject
                            # into a working retry loop).
                            self._empty_output_retries = 0
                            result.stop_reason = "empty_output_max_retries"
                            break

                        # ── Retry budget passed: remove the dead assistant
                        # message, re-inject reasoning, continue the turn ──
                        # NOTE: get_messages() returns a shallow copy — popping
                        # that copy is a no-op on the real history (the
                        # thinking block would stay in the LLM prompt). Use
                        # pop_last_assistant() which mutates _messages for real.
                        d.history.pop_last_assistant()
                        # Inject a targeted nudge
                        if was_thinking_capped:
                            # The runtime cap aborted a monolithic thinking
                            # stream. Same re-injection as the resume path
                            # (assistant thinking blocks are dropped by the
                            # GLM endpoint and the aborted stream has no
                            # valid signature), but the directive is
                            # CONVERGENCE, not resume: the model must land
                            # its next concrete step now — segmented
                            # reasoning beats one 20-minute thought.
                            nudge = (
                                "[system: your reasoning reached its runtime length cap and was stopped. "
                                "Your reasoning so far is reproduced below — do NOT continue it. "
                                "Land the next concrete step NOW: either call a tool to test your "
                                "next hypothesis, or output your answer. Deep problems are solved "
                                "stepwise; experiments beat more thinking.]\n\n"
                                "<previous_reasoning>\n"
                                f"{thinking_text}\n"
                                "</previous_reasoning>"
                            )
                        elif was_reasoning_only:
                            # Many providers (e.g. GLM Anthropic-compat endpoint)
                            # DROP assistant thinking blocks from the input, so
                            # the model cannot see its own reasoning on retry.
                            # Re-inject the FULL thinking content as a user
                            # message so the model resumes from its prior
                            # reasoning instead of starting over.
                            nudge = (
                                "[system: your previous response generated reasoning but no visible output, "
                                "and the reasoning was not preserved in the conversation. Your full reasoning "
                                "is reproduced below — resume from where it left off and output your answer "
                                "or next action directly. Do NOT repeat the reasoning.]\n\n"
                                "<previous_reasoning>\n"
                                f"{thinking_text}\n"
                                "</previous_reasoning>"
                            )
                        else:
                            nudge = "[system: empty response detected, please continue your work]"
                        d.history.append({"role": "user", "content": nudge})
                        continue
                    else:
                        self._empty_output_retries = 0
                        self._cap_retries = 0

                    # Detect completion signals. Use end-anchored matching to
                    # distinguish a real sentinel (trailing the text) from a
                    # mention (e.g. the agent quoting "[TASK_COMPLETE]" while
                    # explaining guard logic). A sentinel is a line that is
                    # (almost) just the marker, optionally followed by an
                    # _override_reason on the same or next line.
                    _text_stripped = assistant_text.rstrip()
                    _is_completion = (
                        _text_stripped.endswith("[TASK_COMPLETE]")
                        or re.search(
                            r'\[TASK_COMPLETE\]\s*\n?_override_reason\s*[:=]',
                            _text_stripped,
                        ) is not None
                    )
                    _is_need_input = _text_stripped.endswith("[NEED_USER_INPUT]")
                    if _is_completion or _is_need_input:
                        # ── Completion-path guard consultation ──
                        # The break below is otherwise unconditional, so guards that
                        # gate completion (e.g. PlanGuard single-shot completion gate)
                        # get no say on a text-only [TASK_COMPLETE]. Consult them here:
                        # a block re-prompts the LLM (continue) instead of breaking, so
                        # the agent must address the gate (or override) before completing.
                        # Text-only [TASK_COMPLETE] has no tool_args to carry an
                        # _override_reason, so a completion-gate block would be
                        # unoverridable unless we give it a text channel. Parse an
                        # override reason from the assistant text itself: the agent
                        # declares it inline as  _override_reason: <reason>  (same
                        # keyword as the tool-arg form). This lets a genuinely-trivial
                        # task be released deliberately, while a bare re-emit of
                        # [TASK_COMPLETE] (no reason) stays blocked.
                        comp_override = self._extract_text_override(assistant_text)
                        comp_ctx = self._build_ctx(
                            tool_name="",
                            tool_args=({"_override_reason": comp_override}
                                       if comp_override else {}),
                            tool_result=None,
                            # This consultation runs right after the LLM produced
                            # the sentinel THIS iteration — the completion text is
                            # fresh, not stale history. The top-of-loop check_pre
                            # (llm_responded defaults False) must NOT be treated as
                            # a completion signal even when it scans a prior turn's
                            # trailing [TASK_COMPLETE] out of history.
                            llm_responded=True,
                        )
                        comp_verdict = d.guard_registry.check_pre(comp_ctx)
                        if comp_verdict is not None:
                            comp_blocked = self._apply_verdict(comp_verdict, pre=True)
                            if comp_blocked:
                                # Livelock breaker: a blocked completion `continue`s
                                # without touching _continuation_count, so a bare
                                # [TASK_COMPLETE] re-emitted with no plan / no override
                                # would spin to max_iter. Count consecutive blocks and
                                # give up once the agent has clearly stalled on the gate.
                                self._consecutive_completion_blocks += 1
                                if (self._consecutive_completion_blocks
                                        >= self.MAX_CONSECUTIVE_COMPLETION_BLOCKS):
                                    result.stop_reason = (
                                        "completion_gate_livelock: "
                                        f"{comp_verdict.reason}"
                                    )
                                    break
                                if iteration < max_iter - 1:
                                    # Re-prompt: LLM sees the guard message and must act
                                    continue
                        # Not blocked (released via plan/override) — reset the breaker.
                        self._consecutive_completion_blocks = 0
                        # Print the authoritative completion marker HERE — only now
                        # that the gate has passed. The raw sentinel was stripped
                        # from the live stream (SentinelStripper), so this is the
                        # single, gate-approved place it reaches the screen.
                        # By convention the terminating sentinel is the LAST one in
                        # the text; an earlier mention (e.g. the agent quoting
                        # "[TASK_COMPLETE]" while explaining code) must NOT override
                        # the actual trailing signal. Pick by last occurrence, not
                        # by a fixed tuple order.
                        _last_sentinel = max(
                            ("[TASK_COMPLETE]", "[NEED_USER_INPUT]"),
                            key=lambda s: assistant_text.rfind(s),
                        )
                        if _last_sentinel in assistant_text:
                            d.display.completion_signal(_last_sentinel)
                        result.stop_reason = "explicit_signal"
                        break

                    # ── Continuation: LLM didn't stop, keep going ──
                    # Short output retry (truncated/confused)
                    if (len(assistant_text.strip()) < 10 and iteration > 0
                            and getattr(self, "_last_turn_had_tools", False)):
                        short_retries = getattr(self, "_short_output_retries", 0)
                        if short_retries < 2:
                            self._short_output_retries = short_retries + 1
                            d.display.warn(f"Short output without tools after active turn, auto-continuing...")
                            d.history.append({"role": "user", "content": "[system: please continue your work]"})
                            continue
                        else:
                            self._short_output_retries = 0

                    # Context pressure check
                    pressure = d.history.get_context_pressure() if hasattr(d.history, 'get_context_pressure') else 0
                    if pressure >= 0.85:
                        result.stop_reason = "context_pressure"
                        break

                    # Continuation count limit (hard ceiling)
                    self._continuation_count += 1
                    if self._continuation_count > d.config.max_continuations:
                        result.stop_reason = "max_continuations"
                        break

                    # Track consecutive text-only responses (no tools, no stop signal)
                    self._consecutive_text_only = getattr(self, "_consecutive_text_only", 0) + 1

                    continuation = self._generate_continuation(text_only=True)
                    d.history.append({"role": "user", "content": continuation})
                    continue

                self._continuation_count = 0
                self._consecutive_text_only = 0  # Reset: tools were executed
                self._consecutive_completion_blocks = 0  # progress made; reset breaker

                # ── Execute tools ──
                _pre_guard_verdicts = []
                try:
                    tool_calls = response["tool_calls"]

                    # ── Per-tool pre-guard checks ──
                    # Give guards a chance to block individual tool calls before execution.
                    # IMPORTANT: Do NOT inject messages here (before tool_results are appended)
                    # because that would break tool_use/tool_result pairing required by the API.
                    # Collect verdicts and apply them AFTER tool_results are in history.
                    blocked_indices = set()
                    _pre_guard_verdicts = []  # (verdict, blocked) pairs to apply after tool_results
                    _seen_injects = set()  # Deduplicate inject across multiple tool_calls
                    for i, tc in enumerate(tool_calls):
                        ctx = self._build_ctx(
                            tool_name=tc["name"],
                            tool_args=tc.get("arguments", {}),
                            tool_result=None,
                        )
                        verdict = d.guard_registry.check_pre(ctx)
                        if verdict is not None:
                            if verdict.action in ("block", "escalate"):
                                blocked_indices.add(i)
                                # Deduplicate block/escalate across tool_calls
                                msg_key = verdict.message[:120]
                                if msg_key not in _seen_injects:
                                    _seen_injects.add(msg_key)
                                    _pre_guard_verdicts.append(verdict)
                                # Display is handled later in _apply_verdict — don't display here
                            elif verdict.action == "inject":
                                # Soft advisory — defer until after tool_results are appended
                                # Deduplicate: same message from same guard across tool_calls
                                msg_key = verdict.message[:120]
                                if msg_key not in _seen_injects:
                                    _seen_injects.add(msg_key)
                                    _pre_guard_verdicts.append(verdict)

                    # Execute tools (skip blocked ones)
                    if blocked_indices:
                        blocked_msg = (
                            "[BLOCKED BY GUARD] This tool call was prevented. "
                            "Read the guard message below carefully and retry with a corrected tool call. "
                            "Do NOT respond with plain text only — you MUST make a tool call in your next response."
                        )
                        if len(blocked_indices) == len(tool_calls):
                            # All blocked
                            results = [blocked_msg] * len(tool_calls)
                        else:
                            # Partial block: execute non-blocked, merge results
                            exec_calls = [tc for i, tc in enumerate(tool_calls) if i not in blocked_indices]
                            exec_results = d.execute_tools_fn(exec_calls)
                            results = []
                            exec_idx = 0
                            for i in range(len(tool_calls)):
                                if i in blocked_indices:
                                    results.append(blocked_msg)
                                else:
                                    results.append(exec_results[exec_idx])
                                    exec_idx += 1
                    else:
                        results = d.execute_tools_fn(tool_calls)
                except KeyboardInterrupt:
                    d.display.interrupted()
                    self._interrupted = True
                    break
                except Exception as e:
                    display.warn(f"Tool execution failed: {e}")
                    # Create error results for all tool calls so the LLM can see what happened
                    tool_calls = response["tool_calls"]
                    results = [f"Error executing tool: {e}"] * len(tool_calls)

                # ── Post-guard checks (per tool) ──
                post_verdicts = []
                _seen_post_injects = set()  # Deduplicate inject across tool_calls
                for tc, tool_result in zip(tool_calls, results):
                    ctx = self._build_ctx(
                        tool_name=tc["name"],
                        tool_args=tc.get("arguments", {}),
                        tool_result=tool_result,
                    )
                    verdict = d.guard_registry.check_post(ctx)
                    if verdict is not None:
                        # Deduplicate all verdict types across multiple tool_calls
                        msg_key = verdict.message[:120]
                        if msg_key in _seen_post_injects:
                            continue
                        _seen_post_injects.add(msg_key)
                        post_verdicts.append(verdict)

                tool_results = [
                    d.format_tool_result_fn(tc["id"], r)
                    for tc, r in zip(tool_calls, results)
                ]
                d.append_tool_results_fn(tool_results)

                # Apply deferred pre-guard verdicts AFTER tool_results are in history,
                # so inject/block messages don't break tool_use → tool_result pairing.
                for verdict in _pre_guard_verdicts:
                    self._apply_verdict(verdict, pre=True)

                # Apply post-guard verdicts AFTER tool results are appended,
                # so inject messages don't break tool_call → tool_result pairing.
                # v4: post verdicts are advisory/escalate only (never block),
                # so they don't stop the turn.
                for verdict in post_verdicts:
                    self._apply_verdict(verdict, pre=False)

                if d.on_tool_results_fn:
                    d.on_tool_results_fn(tool_calls, results)

                self._last_turn_had_tools = True
                result.iterations = iteration + 1

        finally:
            signal.signal(signal.SIGINT, _prev_handler)

        result.interrupted = self._interrupted
        result.elapsed = time.time() - turn_start
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_text_override(text: str) -> str:
        """Parse an inline override reason from assistant text.

        A text-only completion signal ([TASK_COMPLETE]) carries no tool_args, so
        an overridable completion-gate block needs a text channel. The agent
        declares an override inline using the same keyword as the tool-arg form:

            _override_reason: <reason text to end of line>

        Returns the reason (stripped) or "" if absent. Only a reason longer than
        the guard's minimum (checked downstream by accept_override) actually
        releases a block, so a bare keyword with no substance still gets blocked.
        """
        if not text:
            return ""
        # Match  _override_reason: ...  or  _override_reason = ...  (quotes optional)
        m = re.search(
            r'_override_reason\s*[:=]\s*["\']?(.+?)["\']?\s*$',
            text,
            re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    def _get_last_assistant_text(self) -> str:
        """Extract text from last assistant message in history."""
        d = self.deps
        if not d.history:
            return ""
        for msg in reversed(d.history.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content
                             if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    def _build_ctx(self, tool_name: str, tool_args: dict, tool_result: str | None,
                   llm_responded: bool = False) -> GuardContext:
        d = self.deps
        history = d.history
        # Resolve tool effects from registry
        try:
            tool = d.tool_registry.get(tool_name)
        except (KeyError, AttributeError):
            pass
        # Extract override_reason from tool_args (LLM declares why a blocked call is justified)
        # Use .get() + conditional del to avoid mutating the original dict unexpectedly
        override_reason = ""
        if tool_args and "_override_reason" in tool_args:
            override_reason = tool_args["_override_reason"]
            tool_args = {k: v for k, v in tool_args.items() if k != "_override_reason"}
        # Get last assistant text for guards that need to scan LLM responses
        assistant_text = self._get_last_assistant_text()
        return GuardContext(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            
            turn_count=getattr(d.config, "_turn_count", 0),
            context_pressure=history.get_context_pressure() if history else 0.0,
            evictable_indexes=history.get_evictable_indexes() if history else [],
            messages=history.get_messages() if history else [],
            classify_fn=d.judge.classify,
            override_reason=override_reason,
            assistant_text=assistant_text,
            llm_responded=llm_responded,
        )

    def _apply_verdict(self, verdict: GuardVerdict, pre: bool) -> bool:
        """Apply a guard verdict. Returns True if this tool call should be blocked.

        v6 semantics (escalation chain: inject → block → escalate):
        - inject: soft advisory appended to tool_result, turn continues
        - block: prevent tool execution, LLM can override with reason
        - escalate: prevent tool execution + independent message, NOT overridable
        """
        d = self.deps
        if verdict.action == "block":
            d.inject_message_fn(verdict.message)
            display.guard_block(verdict.message)
            return True
        elif verdict.action == "escalate":
            # Ultimate enforcement — block tool + inject message, no override path.
            d.inject_message_fn(verdict.message)
            display.guard_escalate(verdict.message)
            return True
        elif verdict.action == "inject":
            # v4: Soft advisory — append to last tool_result instead of
            # creating a new user message. This avoids conversation pollution.
            if d.append_advisory_fn:
                d.append_advisory_fn(verdict.message)
            else:
                d.inject_message_fn(verdict.message)
            display.guard_inject(verdict.message)
        return False



    def _get_last_assistant_text(self) -> str:
        """Get the text content of the last assistant message."""
        d = self.deps
        for msg in reversed(d.history.get_messages()):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content 
                             if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    def _generate_continuation(self, text_only: bool = False) -> str:
        """Generate continuation prompt.

        Args:
            text_only: If True, the previous response had no tool calls and no
                stop signal. Use a more explicit prompt to help LLM self-correct.
        """
        consecutive = getattr(self, "_consecutive_text_only", 0)

        # If LLM has been outputting text without tools repeatedly, escalate clarity
        if text_only and consecutive >= 3:
            return (
                "You have responded 3+ times with text only — no tool calls and no "
                "stop signal ([TASK_COMPLETE] or [NEED_USER_INPUT]).\n\n"
                "You MUST do one of:\n"
                "1. Emit a tool call (shell, readFile, editFile, etc.) to take action\n"
                "2. End with [TASK_COMPLETE] if the task is done\n"
                "3. End with [NEED_USER_INPUT] if you need user input\n\n"
                "Do NOT output only narration or intentions. Act or stop."
            )

        if text_only:
            plan_hint = ""
            task_plan = getattr(self.deps, "task_plan", None)
            if task_plan:
                active = task_plan.get_active()
                if active:
                    pending = [
                        s for s in active.get("steps", [])
                        if s.get("status") not in ("done", "skipped")
                    ]
                    if pending:
                        step = pending[0]
                        plan_hint = f" Current step: {step.get('title', step.get('description', ''))}"
            return (
                "Your previous response contained text but no tool calls and no stop signal. "
                "The ONLY valid stop signals are [TASK_COMPLETE] and [NEED_USER_INPUT] "
                "— they must appear as the last line of your response. "
                "Simply stopping output is NOT a stop signal.\n\n"
                "You MUST choose exactly one:\n"
                "1. If the task is fully done: end with [TASK_COMPLETE] as the last line, "
                "no tool calls in the same response.\n"
                "2. If you need user input: end with [NEED_USER_INPUT] as the last line, "
                "no tool calls in the same response.\n"
                "3. If you need to take action: emit ONLY tool call(s), no stop signal.\n\n"
                "Do NOT mix tool calls and stop signals in the same response."
                f"{plan_hint}"
            )

        # Normal continuation (after tool execution, LLM just didn't stop)
        task_plan = getattr(self.deps, "task_plan", None)
        if task_plan:
            active = task_plan.get_active()
            if active:
                pending = [
                    s for s in active.get("steps", [])
                    if s.get("status") not in ("done", "skipped")
                ]
                if pending:
                    step = pending[0]
                    return f"Continue with step: {step.get('title', step.get('description', ''))}"
        return "Continue with the next step."
