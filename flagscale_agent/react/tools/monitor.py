"""Monitor tool — long-running local polling without LLM calls.

The agent calls this tool to declare "I want to watch this file/command
until something interesting happens." The system then polls locally,
returning to the LLM only when a meaningful change is detected or timeout.
"""

import glob
import os
import re
import subprocess
import time

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_FS
from flagscale_agent.react.tools.find_log import _last_sorted_subdir, _numeric_key



# Display-only heuristic: detects training metric patterns in log output.
# Not safety-critical — used for log discovery and display summaries where
# classify_fn is not available (polling loops, _discover_logs, _format_result).
_METRIC_RE = re.compile(
    r'step[=:\s]|iteration[=:\s]|loss[=:\s]|grad.norm|throughput|MFU',
    re.IGNORECASE,
)


class MonitorTool(Tool):
    name = "monitor"
    effects = EFFECT_READ_FS
    description = (
        "Watch a file or command output locally WITHOUT calling the LLM. "
        "Use this when you need to wait for training progress, model loading, "
        "or any long-running process. The tool polls locally and only returns "
        "when: (1) an error/completion pattern is detected, (2) new training "
        "metrics appear, (3) the timeout is reached, (4) the target step is hit, "
        "or (5) the monitored process has died. "
        "IMPORTANT: For FlagScale training, use 'output_dir' to auto-scan all "
        "rank stderr.log files for errors — this catches crashes that don't "
        "appear in stdout. "
        "This saves tokens by avoiding repeated LLM calls during waiting."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Path to the log file to watch (e.g., results/log.txt or train.log).",
            },
            "command": {
                "type": "string",
                "description": (
                    "Shell command to poll instead of watching a file. "
                    "Use either 'file' or 'command', not both."
                ),
            },
            "output_dir": {
                "type": "string",
                "description": (
                    "FlagScale output directory (e.g., outputs/qwen3_0_6b_dp_tp). "
                    "When set, the monitor auto-discovers the latest run's log directory "
                    "and scans ALL rank stderr.log files for errors each poll cycle. "
                    "This is the recommended way to monitor FlagScale training."
                ),
            },
            "duration": {
                "type": "integer",
                "description": "Maximum monitoring duration in seconds. Default: 300 (5 min). Max: 1800 (30 min).",
            },
            "interval": {
                "type": "integer",
                "description": "Polling interval in seconds. Default: 30.",
            },
            "target_step": {
                "type": "integer",
                "description": "Stop and return when training reaches this step number.",
            },
            "success_pattern": {
                "type": "string",
                "description": "Regex pattern — return immediately when matched (e.g., 'step=0000100').",
            },
            "fail_pattern": {
                "type": "string",
                "description": "Regex pattern — return immediately on match, flagged as error.",
            },
            "process_pattern": {
                "type": "string",
                "description": (
                    "Regex to check process liveness via pgrep -f. "
                    "If the process dies and no new output appears, monitoring stops early. "
                    "Default: auto-detect from 'torchrun|python.*train|flagscale'."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, display_fn=None, classify_fn=None):
        self._display_fn = display_fn
        self._classify_fn = classify_fn

    def _is_real_error(self, lines: list, context: str = "") -> list:
        """Filter error lines through LLM classify. Returns confirmed error lines."""
        if not lines:
            return []
        if not self._classify_fn:
            return []
        # Send all lines (up to 10) to classify("is_error") in one call
        matched_text = "\n".join(lines[:10])
        context_text = context or matched_text[:500]
        if self._classify_fn("is_error", matched_text, context_text):
            return lines[:10]
        return []

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file", "")
        command = kwargs.get("command", "")
        output_dir = kwargs.get("output_dir", "")
        duration = min(kwargs.get("duration", 300), 1800)
        interval = max(kwargs.get("interval", 30), 5)
        target_step = kwargs.get("target_step")
        success_pattern = kwargs.get("success_pattern", "")
        fail_pattern = kwargs.get("fail_pattern", "")
        process_pattern = kwargs.get("process_pattern", "")

        # If output_dir is given, auto-discover the log file to watch
        if output_dir and not file_path and not command:
            # Wait up to 30s for logs to appear (handles nohup race condition)
            discovered = None
            for _wait in range(6):
                discovered = self._discover_logs(output_dir)
                if not discovered.get("error"):
                    break
                time.sleep(5)
            if discovered.get("error"):
                return discovered["error"]
            file_path = discovered.get("stdout_log", "")
            if not file_path:
                return f"ERROR: No log file found in {output_dir}. Check if training has started."

        if not file_path and not command:
            return "ERROR: Provide 'file', 'command', or 'output_dir' to monitor."

        success_re = re.compile(success_pattern) if success_pattern else None
        fail_re = re.compile(fail_pattern) if fail_pattern else None

        start = time.time()
        poll_count = 0
        last_content = ""
        last_line_count = 0
        events = []
        no_change_cycles = 0
        stderr_checked = {}  # track stderr sizes to detect new errors
        baseline_captured = False  # skip pattern matching on first poll (existing content)

        # Discover stderr logs for output_dir (works for FlagScale and generic layouts)
        stderr_logs = []
        if output_dir:
            discovered = self._discover_logs(output_dir)
            stderr_logs = discovered.get("stderr_logs", [])
            # Immediate stderr check: if stderr already has errors, return immediately
            # (handles case where training crashed before monitor started)
            for sp in stderr_logs:
                try:
                    size = os.path.getsize(sp)
                    if size > 0:
                        with open(sp, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read(16384)
                        error_lines = self._is_real_error(
                            content.splitlines(), content[:500])
                        if error_lines:
                            rank = self._extract_rank_from_path(sp)
                            return self._format_result(
                                "stderr_error", 0, 0,
                                [f"[STDERR ERROR rank {rank} — already present at monitor start: {error_lines[0][:80]}]"],
                                content.strip().splitlines()[-30:], ""
                            )
                except OSError:
                    pass
            # Capture initial stderr sizes so we only detect NEW errors going forward
            for sp in stderr_logs:
                try:
                    stderr_checked[sp] = os.path.getsize(sp)
                except OSError:
                    pass

        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                events.append(f"[timeout after {int(elapsed)}s, {poll_count} polls]")
                break

            # Get current output
            if file_path:
                current = self._read_file(file_path)
            else:
                current = self._run_command(command)

            poll_count += 1

            # First poll: capture baseline content without pattern matching
            # This prevents returning immediately due to pre-existing log content
            if not baseline_captured:
                baseline_captured = True
                last_content = current
                last_line_count = len(current.strip().splitlines())
                # Immediate liveness check — if process is already dead, don't wait
                if not self._is_process_alive(process_pattern):
                    if stderr_logs:
                        stderr_error = self._scan_stderr_logs(stderr_logs, stderr_checked, elapsed)
                        if stderr_error:
                            events.append(stderr_error["event"])
                            return self._format_result(
                                "stderr_error", poll_count, elapsed,
                                events, stderr_error["lines"], current
                            )
                    events.append(f"[process DEAD at start, no training running]")
                    return self._format_result(
                        "process_dead", poll_count, elapsed,
                        events, self._tail_lines(current, 20), current
                    )
                if self._display_fn:
                    self._display_fn(poll_count, elapsed, last_line_count)
                time.sleep(interval)
                continue

            if current != last_content:
                no_change_cycles = 0
                new_lines = self._get_new_lines(last_content, current)

                # Check fail pattern
                if fail_re:
                    for line in new_lines:
                        if fail_re.search(line):
                            events.append(f"[FAIL pattern matched at {int(elapsed)}s]")
                            return self._format_result(
                                "error_detected", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check success pattern
                if success_re:
                    for line in new_lines:
                        if success_re.search(line):
                            events.append(f"[SUCCESS pattern matched at {int(elapsed)}s]")
                            return self._format_result(
                                "success", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check target step
                if target_step is not None:
                    for line in new_lines:
                        step_match = re.search(r'step[=:\s]*0*(\d+)', line, re.IGNORECASE)
                        if step_match and int(step_match.group(1)) >= target_step:
                            events.append(f"[target step {target_step} reached at {int(elapsed)}s]")
                            return self._format_result(
                                "target_reached", poll_count, elapsed,
                                events, new_lines[-20:], current
                            )

                # Check for errors
                error_lines = self._is_real_error(
                    new_lines, "\n".join(new_lines[:10]))
                if error_lines:
                    events.append(f"[interesting change at {int(elapsed)}s: {len(error_lines)} lines]")
                    return self._format_result(
                        "interesting_change", poll_count, elapsed,
                        events, new_lines[-20:], current
                    )

                # Check for new metrics (record but don't break)
                metric_lines = [l for l in new_lines if _METRIC_RE.search(l)]
                if metric_lines:
                    events.append(f"[+{len(metric_lines)} metric lines at {int(elapsed)}s]")

                last_content = current
                current_lines = current.strip().splitlines()
                last_line_count = len(current_lines)
            else:
                no_change_cycles += 1

            # FlagScale stderr scan — every cycle
            if stderr_logs:
                stderr_error = self._scan_stderr_logs(stderr_logs, stderr_checked, elapsed)
                if stderr_error:
                    events.append(stderr_error["event"])
                    return self._format_result(
                        "stderr_error", poll_count, elapsed,
                        events, stderr_error["lines"], current
                    )

            # Process liveness check — every cycle, unconditionally
            if not self._is_process_alive(process_pattern):
                # Process gone — do one final stderr scan for crash reason
                if stderr_logs:
                    stderr_error = self._scan_stderr_logs(stderr_logs, stderr_checked, elapsed)
                    if stderr_error:
                        events.append(stderr_error["event"])
                        return self._format_result(
                            "stderr_error", poll_count, elapsed,
                            events, stderr_error["lines"], current
                        )
                events.append(f"[process DEAD at {int(elapsed)}s, no new output for {no_change_cycles} cycles]")
                return self._format_result(
                    "process_dead", poll_count, elapsed,
                    events, self._tail_lines(current, 20), current
                )

            # Display progress
            if self._display_fn:
                self._display_fn(poll_count, elapsed, last_line_count)

            time.sleep(interval)

        # Timeout — return final state with summary
        return self._format_result(
            "timeout", poll_count, time.time() - start,
            events, self._tail_lines(current, 20), current
        )

    def _discover_logs(self, output_dir):
        """Discover log files — tries FlagScale structure first, then generic scan."""
        # Try FlagScale layout first
        logs_dir = os.path.join(output_dir, "logs", "details")
        if os.path.isdir(logs_dir):
            return self._discover_flagscale_logs(output_dir)
        # Generic log discovery: scan for common log file patterns
        return self._discover_generic_logs(output_dir)

    def _discover_generic_logs(self, output_dir):
        """Discover logs in a generic directory layout.

        Looks for common patterns:
        - *.log files (stdout.log, train.log, output.log, etc.)
        - stderr*, error* files
        - nohup.out
        - Recursively up to 3 levels deep
        """
        result = {"stdout_log": "", "stderr_logs": [], "error": ""}
        if not os.path.isdir(output_dir):
            result["error"] = f"ERROR: Directory not found: {output_dir}"
            return result

        stdout_candidates = []
        stderr_candidates = []

        # Walk up to 3 levels deep
        for root, dirs, files in os.walk(output_dir):
            depth = root.replace(output_dir, "").count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            for f in files:
                fpath = os.path.join(root, f)
                flow = f.lower()
                # stderr / error files
                if "stderr" in flow or "error" in flow:
                    stderr_candidates.append(fpath)
                # stdout / main log files
                elif flow in ("stdout.log", "train.log", "output.log", "nohup.out", "main.log"):
                    stdout_candidates.append(fpath)
                elif flow.endswith(".log"):
                    stdout_candidates.append(fpath)

        if not stdout_candidates and not stderr_candidates:
            result["error"] = f"ERROR: No log files found in {output_dir} (searched 3 levels deep)."
            return result

        # Pick the best stdout log: prefer the most recently modified with content
        if stdout_candidates:
            stdout_candidates.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
            # Prefer files with training metrics
            for candidate in stdout_candidates:
                try:
                    size = os.path.getsize(candidate)
                    if size == 0:
                        continue
                    with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(max(0, size - 4096))
                        tail = fh.read()
                    if _METRIC_RE.search(tail):
                        result["stdout_log"] = candidate
                        break
                except OSError:
                    continue
            # Fallback: most recently modified non-empty file
            if not result["stdout_log"]:
                for candidate in stdout_candidates:
                    try:
                        if os.path.getsize(candidate) > 0:
                            result["stdout_log"] = candidate
                            break
                    except OSError:
                        continue

        result["stderr_logs"] = stderr_candidates
        return result

    def _discover_flagscale_logs(self, output_dir):
        """Discover the latest FlagScale run's log files.

        Reuses find_log utilities for directory traversal.
        FlagScale log structure (multi-node):
          outputs/<exp>/logs/details/host_<N>_<hostname>/<timestamp>/<run_name>/attempt_<N>/<rank>/
            - stdout.log
            - stderr.log

        Scans ALL hosts to collect all rank logs, then picks the rank with
        training metrics (last pipeline rank, which may be on any host).
        """
        result = {"stdout_log": "", "stderr_logs": [], "error": ""}
        logs_dir = os.path.join(output_dir, "logs", "details")
        if not os.path.isdir(logs_dir):
            result["error"] = f"ERROR: No logs directory at {logs_dir}. Training may not have started."
            return result

        host_dirs = sorted(glob.glob(os.path.join(logs_dir, "host_*")))
        if not host_dirs:
            result["error"] = f"ERROR: No host directories in {logs_dir}."
            return result

        # Collect rank dirs from ALL hosts (multi-node support)
        all_rank_dirs = []
        stderr_logs = []
        for host_dir in host_dirs:
            ts_dir = _last_sorted_subdir(host_dir)
            if not ts_dir:
                continue
            run_dir = _last_sorted_subdir(ts_dir)
            if not run_dir:
                continue
            attempt_dir = _last_sorted_subdir(run_dir, key=_numeric_key)
            if not attempt_dir:
                continue
            for entry in sorted(os.listdir(attempt_dir)):
                rank_dir = os.path.join(attempt_dir, entry)
                if not os.path.isdir(rank_dir):
                    continue
                all_rank_dirs.append(rank_dir)
                stderr_path = os.path.join(rank_dir, "stderr.log")
                if os.path.isfile(stderr_path):
                    stderr_logs.append(stderr_path)

        if not all_rank_dirs:
            result["error"] = f"ERROR: No rank directories found under {logs_dir}."
            return result

        # Sort rank dirs by rank number (basename) to find the last rank
        all_rank_dirs.sort(key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0)

        # Pick stdout_log: scan from last rank backwards to find the one with metrics
        stdout_log = ""
        for rank_dir in reversed(all_rank_dirs):
            candidate = os.path.join(rank_dir, "stdout.log")
            if not os.path.isfile(candidate):
                continue
            if not stdout_log:
                stdout_log = candidate  # fallback: last rank's stdout
            try:
                size = os.path.getsize(candidate)
                if size == 0:
                    continue
                with open(candidate, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(max(0, size - 4096))
                    tail = f.read()
                if _METRIC_RE.search(tail):
                    stdout_log = candidate
                    break
            except OSError:
                continue

        # If no rank has metrics yet, fall back to first rank with any content
        if not stdout_log:
            for rank_dir in all_rank_dirs:
                candidate = os.path.join(rank_dir, "stdout.log")
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    stdout_log = candidate
                    break

        result["stdout_log"] = stdout_log
        result["stderr_logs"] = stderr_logs
        return result

    def _scan_stderr_logs(self, stderr_logs, checked_sizes, elapsed):
        """Scan all stderr.log files for new error content.

        Returns dict with 'event' and 'lines' if error found, else None.
        """
        for log_path in stderr_logs:
            try:
                size = os.path.getsize(log_path)
            except OSError:
                continue

            prev_size = checked_sizes.get(log_path, 0)
            if size <= prev_size:
                continue

            # New content in this stderr.log
            checked_sizes[log_path] = size
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(prev_size)
                    new_content = f.read(8192)  # Read up to 8KB of new content
            except Exception:
                continue

            if not new_content.strip():
                continue

            # Check for error patterns
            error_lines = self._is_real_error(
                new_content.splitlines(), new_content[:500])
            if error_lines:
                rank = self._extract_rank_from_path(log_path)
                return {
                    "event": f"[STDERR ERROR rank {rank} at {int(elapsed)}s: {error_lines[0][:80]}]",
                    "lines": new_content.strip().splitlines()[-30:],
                }

            # Even without regex match, non-trivial stderr content is suspicious
            lines = new_content.strip().splitlines()
            if len(lines) > 3:
                rank = self._extract_rank_from_path(log_path)
                return {
                    "event": f"[STDERR activity rank {rank} at {int(elapsed)}s: {len(lines)} lines, possible error]",
                    "lines": lines[-30:],
                }

        return None

    @staticmethod
    def _extract_rank_from_path(path):
        """Extract rank number from FlagScale log path like .../attempt_0/6/stderr.log"""
        parts = path.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p == "stderr.log" and i > 0:
                return parts[i - 1]
        return "?"

    def _is_process_alive(self, process_pattern):
        """Check if the training process is still running.

        Default pattern matches common training launchers while excluding
        the agent's own process (which contains 'flagscale' in its path).
        """
        pattern = process_pattern or r'torchrun|deepspeed|python.*train\.py|python.*finetune|accelerate\s+launch'
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False
            # Filter out our own process and pgrep itself
            my_pid = os.getpid()
            pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
            alive_pids = [p for p in pids if p != my_pid]
            return len(alive_pids) > 0
        except Exception:
            return True

    def _read_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception as e:
            return f"[read error: {e}]"

    def _run_command(self, cmd):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "[command timed out]"
        except Exception as e:
            return f"[command error: {e}]"

    @staticmethod
    def _get_new_lines(old, new):
        old_lines = old.strip().splitlines()
        new_lines = new.strip().splitlines()
        if len(new_lines) > len(old_lines):
            return new_lines[len(old_lines):]
        elif new_lines != old_lines:
            return new_lines[-10:]
        return []

    @staticmethod
    def _tail_lines(content, n=20):
        lines = content.strip().splitlines()
        return lines[-n:] if len(lines) > n else lines

    @staticmethod
    def _format_result(reason, poll_count, elapsed, events, recent_lines, full_content):
        # Make crash reasons unmistakable to the agent
        if reason == "stderr_error":
            header = "TRAINING CRASHED — fatal error detected in stderr"
        elif reason == "process_dead":
            header = "TRAINING DEAD — all processes exited"
        else:
            header = f"Monitor result: {reason}"
        parts = [f"{header} ({poll_count} polls, {int(elapsed)}s)"]

        if events:
            parts.append("Events:")
            for e in events[-10:]:
                parts.append(f"  {e}")

        if reason in ("stderr_error", "process_dead") and recent_lines:
            parts.append("Error output (stderr):" if reason == "stderr_error" else "Last output before death:")
            for line in recent_lines:
                parts.append(f"  {line}")
            if reason == "stderr_error":
                parts.append("ACTION REQUIRED: Training has failed. Do NOT re-monitor — diagnose the error above.")
        elif recent_lines:
            parts.append("Recent output:")
            for line in recent_lines:
                parts.append(f"  {line}")

        # Extract latest metrics for quick reference (only for non-crash results)
        if reason not in ("stderr_error", "process_dead"):
            metric_lines = [l for l in (full_content or "").splitlines() if _METRIC_RE.search(l)]
            if metric_lines:
                parts.append("Latest metrics:")
                for line in metric_lines[-3:]:
                    parts.append(f"  {line.strip()}")

        return "\n".join(parts)
