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

import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from flagscale_agent.react.state_machine import AgentState, StateMachine
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
    inject_message_fn: Callable            # (msg: str) -> None
    append_tool_results_fn: Callable       # (results: list) -> None
    format_tool_result_fn: Callable        # (id, result) -> dict
    execute_tools_fn: Callable             # (tool_calls) -> list[str]
    is_context_limit_error_fn: Callable    # (exc) -> bool
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
    final_state: AgentState = AgentState.COMPLETED
    stop_reason: str = ""


class AgentKernel:
    """Minimal event loop. < 200 lines of logic.

    One instance per agent. Call run_turn() for each user message.
    """

    def __init__(self, deps: KernelDeps):
        self.deps = deps
        self.fsm = StateMachine(AgentState.IDLE)
        self._interrupted = False
        self._plan_auto_continue_count = 0

    def run_turn(self) -> KernelResult:
        """Run one ReAct turn (one user message → completion).

        Returns KernelResult with token stats and stop reason.
        """
        result = KernelResult()
        d = self.deps
        max_iter = d.config.max_iterations
        turn_start = time.time()

        self._interrupted = False
        self._plan_auto_continue_count = 0  # Reset per turn to avoid poisoning
        self.fsm.transition(AgentState.EXECUTING, reason="new turn")
        d.judge.reset_turn()

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

                # Reset guards for this iteration (called once per LLM+tool loop)
                d.guard_registry.reset_iteration()
                d.judge.reset_turn()

                schemas = d.get_schemas_fn()

                # ── Pre-guard checks ──
                ctx = self._build_ctx(tool_name="", tool_args={}, tool_result=None)
                verdict = d.guard_registry.check_pre(ctx)
                if verdict is not None:
                    blocked = self._apply_verdict(verdict, pre=True)
                    if blocked:
                        result.stop_reason = f"blocked_by_guard: {verdict.reason}"
                        break

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
                    if d.is_context_limit_error_fn(e):
                        response, usage = self._recover_context_overflow(e, schemas)
                        if response is None:
                            result.stop_reason = "context_overflow_unrecoverable"
                            break
                    else:
                        d.display.thinking_clear()
                        display.warn(f"LLM call failed: {e}")
                        result.stop_reason = f"llm_error: {e}"
                        break

                elapsed = time.time() - getattr(self, "_t0", time.time())
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                result.input_tokens += in_tok
                result.output_tokens += out_tok
                if in_tok:
                    d.history.report_actual_tokens(in_tok)

                d.display.llm_done(elapsed, in_tok, out_tok)

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
                        empty_retries = getattr(self, "_empty_output_retries", 0)
                        if empty_retries < 3:
                            self._empty_output_retries = empty_retries + 1
                            d.display.warn(f"Empty LLM output (retry {empty_retries + 1}/3), auto-continuing...")
                            # Remove the empty assistant message we just appended
                            msgs = d.history.get_messages()
                            if msgs and msgs[-1].get("role") == "assistant":
                                msgs.pop()
                            # Inject a nudge
                            d.history.append({"role": "user", "content": "[system: empty response detected, please continue your work]"})
                            continue
                        else:
                            self._empty_output_retries = 0
                            result.stop_reason = "empty_output_max_retries"
                            break
                    else:
                        self._empty_output_retries = 0

                    if "[TASK_COMPLETE]" in assistant_text or "[NEED_USER_INPUT]" in assistant_text:
                        result.stop_reason = "explicit_signal"
                        break
                    if not self._should_auto_continue_plan():
                        # Defense: if last turn had tool calls (still working) and this turn
                        # is trivially short (<10 chars), auto-continue instead of stopping
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
                        result.stop_reason = "no_tool_calls"
                        break
                    # Plan auto-continue — check token budget first
                    pressure = d.history.get_context_pressure() if hasattr(d.history, 'get_context_pressure') else 0
                    if pressure >= 0.85:
                        result.stop_reason = "context_pressure"
                        break
                    self._plan_auto_continue_count += 1
                    if self._plan_auto_continue_count > 10:
                        result.stop_reason = "plan_auto_continue_limit"
                        break
                    continuation = self._generate_continuation()
                    d.history.append({"role": "user", "content": continuation})
                    continue

                self._plan_auto_continue_count = 0

                # ── Execute tools ──
                try:
                    tool_calls = response["tool_calls"]

                    # ── Per-tool pre-guard checks ──
                    # Give guards a chance to block individual tool calls before execution
                    blocked_indices = set()
                    for i, tc in enumerate(tool_calls):
                        ctx = self._build_ctx(
                            tool_name=tc["name"],
                            tool_args=tc.get("arguments", {}),
                            tool_result=None,
                        )
                        verdict = d.guard_registry.check_pre(ctx)
                        if verdict is not None:
                            blocked = self._apply_verdict(verdict, pre=True)
                            if blocked:
                                blocked_indices.add(i)

                    # Execute tools (skip blocked ones)
                    if blocked_indices:
                        blocked_msg = (
                            "[BLOCKED BY GUARD] This tool call was prevented. "
                            "See the injected message above for instructions."
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
                for tc, tool_result in zip(tool_calls, results):
                    ctx = self._build_ctx(
                        tool_name=tc["name"],
                        tool_args=tc.get("arguments", {}),
                        tool_result=tool_result,
                    )
                    verdict = d.guard_registry.check_post(ctx)
                    if verdict is not None:
                        post_verdicts.append(verdict)

                tool_results = [
                    d.format_tool_result_fn(tc["id"], r)
                    for tc, r in zip(tool_calls, results)
                ]
                d.append_tool_results_fn(tool_results)

                # Apply post-guard verdicts AFTER tool results are appended,
                # so inject messages don't break tool_call → tool_result pairing.
                # v3: Inject messages are applied normally but with ADVISORY prefix
                # (set in _inject_message) + max_inject_repeats prevents flooding.
                for verdict in post_verdicts:
                    self._apply_verdict(verdict, pre=False)

                if d.on_tool_results_fn:
                    d.on_tool_results_fn(tool_calls, results)

                self._last_turn_had_tools = True
                result.iterations = iteration + 1

                # ── Self-modification detection ──
                # If any file tool modified flagscale_agent/ source, stop and ask for /reload
                if self._detect_self_modification(tool_calls):
                    # Inject a notice to the assistant so it knows to stop
                    d.history.append({"role": "user", "content": (
                        "[system: You just modified FlagScale Agent's own source code "
                        "(flagscale_agent/). These changes require /reload to take effect. "
                        "STOP here and tell the user to run /reload. Do NOT continue other work.]"
                    )})
                    # Do one more LLM call to let it produce the stop message
                    response, usage = d.stream_fn(d.history.get_messages())
                    if d.append_response_fn:
                        d.append_response_fn(response)
                    result.stop_reason = "self_modification_reload_needed"
                    break

        finally:
            signal.signal(signal.SIGINT, _prev_handler)

        result.interrupted = self._interrupted
        result.final_state = self.fsm.current_state
        result.elapsed = time.time() - turn_start
        if self._interrupted:
            self.fsm.force_transition(AgentState.INTERRUPTED, reason="user interrupt")
        else:
            self.fsm.transition(AgentState.COMPLETED, reason=result.stop_reason or "done")
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

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

    def _build_ctx(self, tool_name: str, tool_args: dict, tool_result: str | None) -> GuardContext:
        d = self.deps
        history = d.history
        # Resolve tool effects from registry
        from flagscale_agent.react.tools.base import ToolEffect
        tool_effects = ToolEffect()
        try:
            tool = d.tool_registry.get(tool_name)
            tool_effects = tool.effects
        except (KeyError, AttributeError):
            pass
        # Extract override_reason from tool_args (LLM declares why a blocked call is justified)
        # Use .get() + conditional del to avoid mutating the original dict unexpectedly
        override_reason = ""
        if tool_args and "_override_reason" in tool_args:
            override_reason = tool_args["_override_reason"]
            tool_args = {k: v for k, v in tool_args.items() if k != "_override_reason"}
        # v3: Extract _dismiss_guard (LLM explicitly dismisses a guard's inject)
        if tool_args and "_dismiss_guard" in tool_args:
            dismiss_name = tool_args["_dismiss_guard"]
            tool_args = {k: v for k, v in tool_args.items() if k != "_dismiss_guard"}
            for guard in d.guard_registry.guards:
                if guard.name == dismiss_name:
                    guard.dismiss_inject()
                    display.guard_inject(f"[{dismiss_name}] dismissed by LLM")
                    break
        # Get last assistant text for guards that need to scan LLM responses
        assistant_text = self._get_last_assistant_text()
        return GuardContext(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            tool_effects=tool_effects,
            turn_count=getattr(d.config, "_turn_count", 0),
            context_pressure=history.get_context_pressure() if history else 0.0,
            current_state=self.fsm.current_state,
            transitions_count=len(self.fsm.history),
            classify_fn=d.judge.classify,
            override_reason=override_reason,
            assistant_text=assistant_text,
        )

    def _apply_verdict(self, verdict: GuardVerdict, pre: bool) -> bool:
        """Apply a guard verdict. Returns True if the verdict is a 'block' action."""
        d = self.deps
        if verdict.action == "block":
            d.inject_message_fn(verdict.message)
            display.guard_block(verdict.message)
            return True
        elif verdict.action == "inject_msg":
            d.inject_message_fn(verdict.message)
            display.guard_inject(verdict.message)
        elif verdict.action == "force_compact":
            d.history.force_compact()
        elif verdict.action == "escalate":
            d.inject_message_fn(verdict.message)
            display.guard_block(verdict.message)
            self.fsm.transition(AgentState.REVIEWING, reason=verdict.reason)
        return False

    def _recover_context_overflow(self, exc, schemas):
        """Try progressively aggressive compaction on context overflow."""
        d = self.deps

        # Save recovery state to plan before compaction
        self._save_recovery_state()

        d.display.thinking_clear()
        display.warn("Context overflow, compacting...")
        _call = d.call_llm_fn or (lambda m, s: d.provider.chat_stream(m, s))
        for ratio in [0.50, 0.35, 0.25]:
            overflow_limit = d.history._actual_input_tokens or d.config.max_context_tokens
            d.history.force_compact(target_ratio=ratio, base_limit=int(overflow_limit * 0.80))
            messages = d.history.get_messages()
            try:
                d.display.thinking()
                return _call(messages, schemas)
            except Exception as e2:
                d.display.thinking_clear()
                if not d.is_context_limit_error_fn(e2):
                    display.warn(f"LLM error after compact: {e2}")
                    return None, {}
        return None, {}

    def _detect_self_modification(self, tool_calls: list) -> bool:
        """Check if any tool call modified flagscale_agent/ source files.
        
        Detects write_file/edit_file operations targeting the agent's own code,
        which require a /reload to take effect.
        """
        SELF_PATHS = ("flagscale_agent/", "flagscale_agent\\")
        FILE_TOOLS = ("write_file", "edit_file")
        
        for tc in tool_calls:
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name not in FILE_TOOLS:
                continue
            # Extract path from tool call input
            inp = tc.get("input", {}) if isinstance(tc, dict) else getattr(tc, "input", {})
            if isinstance(inp, str):
                try:
                    import json
                    inp = json.loads(inp)
                except (json.JSONDecodeError, TypeError):
                    continue
            path = inp.get("path", "") if isinstance(inp, dict) else ""
            # Check if path touches agent source
            if any(seg in path for seg in SELF_PATHS):
                return True
        return False

    def _save_recovery_state(self):
        """Save current progress to plan notes before compaction.

        This ensures the agent can recover its working state after context
        is compacted, preventing the post-compaction death loop.
        """
        d = self.deps
        task_plan = getattr(d, "task_plan", None)
        if not task_plan:
            return

        active = task_plan.get_active()
        if not active:
            return

        # Find current "doing" step
        steps = active.get("steps", [])
        doing_steps = [s for s in steps if s.get("status") == "doing"]
        if not doing_steps:
            return

        step = doing_steps[0]
        step_id = step.get("id")

        # Build recovery context from recent history
        recent_msgs = d.history.get_messages()[-6:]  # Last 3 exchanges
        recovery_lines = []
        for msg in recent_msgs:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    for line in content.split("\n"):
                        if line.strip() and not line.startswith("["):
                            recovery_lines.append(line.strip()[:200])
                            break

        if recovery_lines:
            recovery_note = "RECOVERY: " + " | ".join(recovery_lines[-3:])
            try:
                task_plan.update_step(step_id, "doing", recovery_note)
            except Exception:
                pass

    def _should_auto_continue_plan(self) -> bool:
        """Check if there's an active plan with pending steps.
        
        Also checks if the assistant's last response is asking the user a question
        (waiting for user input). If so, do NOT auto-continue — let the user respond.
        """
        task_plan = getattr(self.deps, "task_plan", None)
        if task_plan is None:
            return False
        active = task_plan.get_active()
        if not active:
            return False
        has_pending = any(
            s.get("status") not in ("done", "skipped")
            for s in active.get("steps", [])
        )
        if not has_pending:
            return False
        
        # Check if assistant is waiting for user input (asking a question)
        last_text = self._get_last_assistant_text()
        if last_text and self._is_asking_user(last_text):
            return False
        
        return True

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

    def _is_asking_user(self, text: str) -> bool:
        """Detect if the assistant text is asking the user a question / waiting for input.
        
        Heuristic: check if the last meaningful line ends with a question mark or
        contains explicit "waiting for user" patterns.
        """
        # Get the last few non-empty lines
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return False
        
        last_line = lines[-1]
        
        # Explicit signal
        if "[NEED_USER_INPUT]" in text:
            return True
        
        # Ends with question mark (supports Chinese and English)
        if last_line.endswith("?") or last_line.endswith("？"):
            return True
        
        # Common patterns for asking user (Chinese + English)
        asking_patterns = [
            "你选", "你觉得", "你希望", "你要", "要我",
            "which do you", "what do you", "do you want", "shall i",
            "should i", "would you", "let me know", "your choice",
            "选哪个", "怎么处理", "如何处理",
        ]
        last_lower = last_line.lower()
        for pattern in asking_patterns:
            if pattern in last_lower:
                return True
        
        return False

    def _generate_continuation(self) -> str:
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
