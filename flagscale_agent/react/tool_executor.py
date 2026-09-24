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

"""Tool execution engine for FlagScale Agent.

Handles single and batch tool execution with:
- Deduplication of identical calls
- Batch size capping
- Guard pre-checks (block/inject)
- Shell command confirmation
- Parallel execution with display
- Tool result caching within a turn
- File read/write tracking
- Skill auto-loading on load_skill
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from flagscale_agent.react import display

if TYPE_CHECKING:
    from flagscale_agent.react.agent import WorkerAgent



def _short_path(path: str, max_len: int = 60) -> str:
    """Show a short but distinguishing path (not just basename).

    Shows the relative path from cwd if short enough, otherwise the last
    few path components with a …/ prefix.
    """
    if not path:
        return ""
    # Try relative path from cwd
    try:
        rel = os.path.relpath(path)
        if not rel.startswith("../../") and len(rel) <= max_len:
            return rel
    except ValueError:
        pass
    # Fallback: last N components that fit within max_len
    parts = path.replace("\\", "/").split("/")
    # Always include at least basename
    result = parts[-1]
    for i in range(len(parts) - 2, -1, -1):
        candidate = "/".join(parts[i:])
        if len(candidate) + 2 > max_len:  # +2 for "…/"
            break
        result = candidate
    if result != path:
        result = "…/" + result
    return result


def tool_display_summary(tool_name: str, arguments: dict) -> str:
    """Short human-readable summary for a tool call display."""
    if tool_name == "shell":
        cmd = arguments.get("command", "")
        if not isinstance(cmd, str):
            cmd = str(cmd) if cmd else ""
        s = cmd.replace("\n", " ").replace("\r", "").strip()
        # Show full command — most shell commands are short enough to display completely
        return s
    if tool_name == "read_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        summary = _short_path(path)
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if start or end:
            summary += f":{start or 1}-{end or 'EOF'}"
        return summary
    if tool_name == "edit_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        # Just show path, no content comparison
        return _short_path(path)
    if tool_name == "write_file":
        path = arguments.get("path", "") or arguments.get("file_path", "")
        return _short_path(path)
    if tool_name == "web_fetch":
        url = arguments.get("url", "")
        return url
    if tool_name == "load_skill":
        return arguments.get("name", "")
    if tool_name == "memory_write":
        return arguments.get("key", "")
    if tool_name == "plan_create":
        return arguments.get("title", "")
    if tool_name == "plan_update":
        action = arguments.get("action", "")
        step_id = arguments.get("step_id", "")
        return f"{action} step_{step_id}" if step_id else action
    if tool_name == "plan_status":
        return ""
    if tool_name == "flagscale_train_monitor":
        # Show what's being monitored
        output_dir = arguments.get("output_dir", "")
        mode = arguments.get("mode", "watch")
        duration = arguments.get("duration", 300)
        target = arguments.get("target_step")
        summary = _short_path(output_dir, 40) if output_dir else "unknown"
        if target:
            summary += f" →step {target}"
        summary += f" ({duration}s)"
        return summary
    if tool_name == "memory_read":
        return arguments.get("key", "")
    if tool_name == "evict":
        indexes = arguments.get("indexes", [])
        if isinstance(indexes, list) and len(indexes) > 5:
            return f"[{', '.join(str(i) for i in indexes)}] ({len(indexes)} total)"
        return str(indexes)
    if tool_name == "recall":
        return f"index={arguments.get('index', '?')}"
    if tool_name == "shell_jobs":
        action = arguments.get("action", "")
        job_id = arguments.get("job_id", "")
        summary = action or "?"
        if job_id:
            summary += f" {job_id}"
        if action == "wait":
            timeout = arguments.get("timeout", 60)
            summary += f" (≤{timeout}s)"
        return summary
    return ""


class ToolExecutor:
    """Executes tools with batching, dedup, guards, and parallel dispatch.

    This class encapsulates the tool execution lifecycle:
    1. Pre-checks (guards, confirmation)
    2. Deduplication and batch capping
    3. Parallel execution with display
    4. Post-execution tracking (skills, cache)
    """

    def __init__(self, agent: "WorkerAgent"):
        self._agent = agent

    def execute_batch(self, tool_calls: list[dict]) -> list[str]:
        """Execute a batch of tool calls.

        Single-call batches execute directly.
        Multi-call batches get dedup, batch capping, and parallel execution.
        """
        if len(tool_calls) == 1:
            return [self.execute_single(tool_calls[0])]

        return self._execute_parallel(tool_calls)

    def execute_single(self, tool_call: dict) -> str:
        """Execute a single tool call with display, caching, and tracking."""
        agent = self._agent
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # Check turn cache
        cached_key = (tool_name, json.dumps(arguments, sort_keys=True))
        if cached_key in agent._tool_call_cache:
            return agent._tool_call_cache[cached_key] + "\n[Cached result from earlier in this turn]"

        detail = tool_display_summary(tool_name, arguments)
        display.tool_start(tool_name, detail)
        t0 = time.time()
        try:
            result = agent.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            result = f"Error executing tool: {e}\n\n[Tool: {tool_name}]\n[Args: {arguments}]\n[Traceback]\n{tb}"
        elapsed = time.time() - t0

        error = False
        err_detail = ""
        if result:
            first_line = result.split('\n')[0] if '\n' in result else result
            if (first_line.startswith(("ERROR:", "FATAL:", "STALLED:", "TERMINATED:", "DENIED:"))
                    or "Traceback (most recent call last)" in result):
                error = True
                err_detail = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
        display.tool_done(tool_name, elapsed, detail=err_detail, error=error)

        # Post-execution tracking
        result = self._post_execute(tool_name, arguments, result, error)

        # Cache for this turn
        agent._tool_call_cache[cached_key] = result
        return result

    def _post_execute(self, tool_name: str, arguments: dict, result: str, error: bool) -> str:
        """Handle post-execution side effects (tracking, skill loading, etc.)."""
        agent = self._agent



        # Track load_skill side effects
        if tool_name == "load_skill" and not error:
            skill_name = arguments.get("name", "")
            if skill_name not in agent._loaded_skills:
                agent._loaded_skills.add(skill_name)

        return result

    def _execute_parallel(self, tool_calls: list[dict]) -> list[str]:
        """Execute multiple tool calls with dedup, guards, and parallel dispatch."""
        agent = self._agent

        # Dedup
        seen_calls = {}
        dedup_indices = set()
        for i, tc in enumerate(tool_calls):
            key = (tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True))
            if key in seen_calls:
                dedup_indices.add(i)
            else:
                seen_calls[key] = i

        _MAX_BATCH = 20
        capped_indices = set()
        if len(tool_calls) > _MAX_BATCH:
            for i in range(_MAX_BATCH, len(tool_calls)):
                capped_indices.add(i)

        # Note: Guard checks are handled by kernel.py BEFORE tools reach here.
        # kernel.py filters out blocked tools, so we don't re-check guards here.
        # This prevents duplicate guard checks and double override displays.
        skip_indices: set[int] = set()
        results = [None] * len(tool_calls)

        skip_indices |= dedup_indices | capped_indices

        for i in dedup_indices:
            orig = seen_calls[(tool_calls[i]["name"], json.dumps(tool_calls[i].get("arguments", {}), sort_keys=True))]
            results[i] = f"[DEDUP: identical to call #{orig + 1} in this batch, skipped]"
        for i in capped_indices:
            results[i] = f"[BATCH CAPPED — TOOL NOT EXECUTED] Only {_MAX_BATCH} tool calls allowed per response."

        to_run = [(i, tc) for i, tc in enumerate(tool_calls) if i not in skip_indices]

        # Start parallel display
        summaries = [(tc["name"], tool_display_summary(tc["name"], tc.get("arguments", {})))
                     for _, tc in to_run]
        display.parallel_tools_start(summaries)

        idx_to_line = {orig_i: line_i for line_i, (orig_i, _) in enumerate(to_run)}

        def _run_quiet(idx, tc):
            tool_name = tc["name"]
            arguments = tc["arguments"]
            t0 = time.time()
            try:
                if tool_name == "shell":
                    result = agent.tool_registry.execute(
                        tool_name, _quiet=True, **arguments)
                else:
                    result = agent.tool_registry.execute(tool_name, **arguments)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                result = f"Error executing tool: {e}\n\n[Tool: {tool_name}]\n[Args: {arguments}]\n[Traceback]\n{tb}"
            elapsed = time.time() - t0
            error = "ERROR" in result or "Error executing tool" in result if result else False
            detail = ""
            if error and result:
                raw = result.split('\n')[0].replace("ERROR:", "").strip()
                detail = raw
            display.parallel_tool_update(idx_to_line[idx], elapsed, error, detail)
            return result

        if not to_run:
            display.parallel_tools_finish()
            return results

        with ThreadPoolExecutor(max_workers=min(len(to_run), 4)) as pool:
            futures = {pool.submit(_run_quiet, i, tc): i for i, tc in to_run}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        display.parallel_tools_finish()

        return results
