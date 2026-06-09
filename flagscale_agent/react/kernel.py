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

                # Reset guards for this iteration
                d.guard_registry.reset_turn()
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
                    if "[TASK_COMPLETE]" in assistant_text or "[NEED_USER_INPUT]" in assistant_text:
                        result.stop_reason = "explicit_signal"
                        break
                    if not self._should_auto_continue_plan():
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
                # so inject messages don't break tool_call → tool_result pairing
                for verdict in post_verdicts:
                    self._apply_verdict(verdict, pre=False)

                if d.on_tool_results_fn:
                    d.on_tool_results_fn(tool_calls, results)

                result.iterations = iteration + 1

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
        )

    def _apply_verdict(self, verdict: GuardVerdict, pre: bool) -> bool:
        """Apply a guard verdict. Returns True if the verdict is a 'block' action."""
        d = self.deps
        if verdict.action == "block":
            d.inject_message_fn(verdict.message)
            return True
        elif verdict.action == "inject_msg":
            d.inject_message_fn(verdict.message)
        elif verdict.action == "force_compact":
            d.history.force_compact()
        elif verdict.action == "escalate":
            d.inject_message_fn(verdict.message)
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
        """Check if there's an active plan with pending steps."""
        task_plan = getattr(self.deps, "task_plan", None)
        if task_plan is None:
            return False
        active = task_plan.get_active()
        if not active:
            return False
        return any(
            s.get("status") not in ("done", "skipped")
            for s in active.get("steps", [])
        )

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
