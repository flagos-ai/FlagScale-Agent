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

"""Shell command tool — pure executor with long-command monitoring."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time

from flagscale_agent.react.tools.base import Tool


# Max seconds to wait for the health-judge LLM call before giving up on it for
# this check. The judge is a best-effort advisory; if the gateway stalls we must
# NOT let it freeze the monitor loop (which would suppress the heartbeat and the
# hard-indicator checks). On timeout we treat the judge as "no opinion" and let
# the loop continue emitting its heartbeat on schedule.
_HEALTH_JUDGE_TIMEOUT_SECS = 20

# get_process_health walks the entire child-process tree with per-process /proc
# reads. Under a huge parallel build the tree explodes and a single sample can
# block for minutes on slow /proc IO. Cap it well under the ~30s check interval so
# the heartbeat never starves; on timeout we fall back to a neutral reading.
_PROCESS_HEALTH_TIMEOUT_SECS = 5

# Max characters of command output returned to the caller. A chatty build
# (compile, make -j, pip) can emit hundreds of MB in seconds; returning it all
# as ONE observation floods the LLM context (measured: 200k trivial lines =
# 7.7 MB ≈ 1.9M tokens) and grows memory unbounded. We keep the HEAD (command
# echo, config, early errors) and the TAIL (final errors, exit status, result
# summary) — the two regions that actually carry signal — and drop the middle,
# which is almost always repetitive progress. ~200KB ≈ 50k tokens: large enough
# to preserve real diagnostics, small enough to never single-handedly blow a
# context window.
_MAX_OUTPUT_CHARS = 200_000
# When truncating, how to split the budget between head and tail. The tail is
# where failures and exit status land, so it gets the larger share.
_OUTPUT_HEAD_CHARS = 60_000
_OUTPUT_TAIL_CHARS = 140_000


def _truncate_output(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Bound command output by keeping the head and tail, dropping the middle.

    Build logs put the signal at the ends: the head has the command/config and
    first errors, the tail has the final errors and exit status. The middle is
    repetitive progress. If text fits, it is returned unchanged.
    """
    if len(text) <= max_chars:
        return text
    head = text[:_OUTPUT_HEAD_CHARS]
    tail = text[-_OUTPUT_TAIL_CHARS:]
    # Snap the cut points to line boundaries so we never emit a half-line that
    # could split a key error message. Head: trim back to the last newline it
    # contains. Tail: trim forward to the first newline. A pathological line
    # longer than the budget (no newline to snap to) keeps the raw slice, so the
    # size bound is always honored.
    head_nl = head.rfind("\n")
    if head_nl != -1:
        head = head[: head_nl + 1]
    tail_nl = tail.find("\n")
    if tail_nl != -1:
        tail = tail[tail_nl + 1 :]
    omitted = len(text) - len(head) - len(tail)
    marker = (
        f"\n\n... [output truncated: {omitted:,} chars omitted from the middle; "
        f"showing first {len(head):,} and last {len(tail):,} of {len(text):,} "
        f"total. Re-run with a narrower filter (grep/tail/head) to see more] ...\n\n"
    )
    return head + marker + tail


class _HealthSampler:
    """Single long-lived background thread that samples process health.

    WHY THIS EXISTS: the monitor loop must never call get_process_health on
    its own (heartbeat) thread. Under a huge parallel build (opam/coq, make
    -j) the child-process tree explodes and one get_process_health call —
    children(recursive=True) + per-proc cpu_times()/io_counters() /proc reads
    — can block for MINUTES on saturated disk IO. The previous design spawned
    a fresh bounded daemon thread every check_interval; each blocked for
    minutes, they PILED UP (dozens of abandoned threads all hammering /proc,
    contending for the GIL), and the main heartbeat starved for minutes
    (observed: an 8-minute heartbeat gap during a compcert build).

    This sampler runs exactly ONE thread for the whole command. It samples,
    stores the latest result + timestamp under a lock, then sleeps. The main
    loop reads latest() — an O(1) dict copy that NEVER blocks — so the
    heartbeat fires on schedule no matter how slow /proc is. A slow sample
    just means the main loop reuses the last-known reading (marked stale via
    its age), never that the heartbeat freezes.
    """

    def __init__(self, fn, pid, interval):
        self._fn = fn
        self._pid = pid
        self._interval = interval
        self._lock = threading.Lock()
        self._latest = None
        self._ts = 0.0
        self._seq = 0  # increments on every fresh sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                value = self._fn(self._pid)
            except Exception:
                value = None
            if value is not None:
                with self._lock:
                    self._latest = value
                    self._ts = time.time()
                    self._seq += 1
            # Wait interval, but wake early if asked to stop.
            self._stop.wait(self._interval)

    def latest(self):
        """Return (sample_or_None, age_seconds, seq). Never blocks on /proc.

        seq lets the caller tell a FRESH sample from a re-read of the same one.
        Liveness deltas (cpu_time/io/rss) are only meaningful between two
        DISTINCT samples — computing a delta against the same snapshot always
        yields 0 and would false-flag a busy process as idle. The main loop
        must only run its idle/kill delta logic when seq advanced.
        """
        with self._lock:
            if self._latest is None:
                return None, None, self._seq
            return dict(self._latest), time.time() - self._ts, self._seq

    def stop(self):
        self._stop.set()


def _run_health_judge_bounded(fn, args, kwargs, timeout=_HEALTH_JUDGE_TIMEOUT_SECS):
    """Call the health-judge fn in a worker thread with a hard timeout.

    Returns the judge's decision dict, or None if it timed out / raised. The
    worker thread is daemonized, so a genuinely wedged LLM call cannot block
    process exit — it is abandoned, not joined indefinitely.
    """
    result = {}

    def _worker():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception:
            result["value"] = None

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None  # judge stalled — abandon it, keep the loop alive
    return result.get("value")


class _HealthEvaluator:
    """One tick of the health/liveness decision, extracted so BOTH the
    synchronous monitor loop and the ShellJobsTool background poller can run
    the exact same logic against a _HealthSampler.

    It holds the incremental state the decision needs across ticks (stall /
    idle counters, previous cumulative CPU/IO/RSS counters, the last acted-on
    sample seq). Each tick:
      • compute output_changed from the latest output snapshot,
      • read the sampler (O(1), never blocks on /proc),
      • handle the stale / fresh / non-fresh sample cases exactly as the sync
        loop does (neutral fallback for a stuck sampler; skip when no fresh
        sample so counters are never fabricated),
      • derive liveness deltas + idle_count,
      • call should_kill_process,
      • build the human-readable activity string for the LLM judge.

    Returns a dict; status='skip' means "no fresh health info this tick, do
    nothing" (heartbeat already handled by the caller).
    """

    def __init__(self):
        self.stall_count = 0
        self.idle_count = 0
        self._had_children = False
        self._prev_num_children = 0
        self._prev_cpu_time = None
        self._prev_io_bytes = None
        self._prev_rss_mb = None
        self._prev_health_time = None
        self._tree_cpu_pct = 0.0
        self._last_health_seq = -1
        self._last_output_snapshot = ""

    def evaluate(self, sampler, stdout_chunks, stderr_chunks, elapsed):
        """Run one health tick. Returns a decision dict:
            {'status': 'skip'}  — no fresh sample, do nothing this tick, or
            {'status': 'ok', 'should_kill': bool, 'kill_reason': str,
             'activity': str, 'output_changed': bool, 'stall_count': int,
             'recent_text': str}
        """
        from flagscale_agent.react.tools.process_health import (
            detect_output_anomalies,
            should_kill_process,
        )

        recent_text = "".join(stdout_chunks[-20:] + stderr_chunks[-20:])
        current_snapshot = "".join(stdout_chunks[-10:] + stderr_chunks[-10:])

        # Empty output counts as a STALL, not "changed" (see sync loop).
        output_changed = (
            bool(current_snapshot) and current_snapshot != self._last_output_snapshot
        )
        if not output_changed:
            self.stall_count += 1
        else:
            self.stall_count = 0
        self._last_output_snapshot = current_snapshot

        proc_health, _sample_age, _sample_seq = sampler.latest()
        _fresh = proc_health is not None and _sample_seq != self._last_health_seq
        _stale = (
            _sample_age is not None
            and _sample_age > 3 * _PROCESS_HEALTH_TIMEOUT_SECS
        )
        if _stale:
            _base_cpu = self._prev_cpu_time if self._prev_cpu_time is not None else 0.0
            _base_io = self._prev_io_bytes if self._prev_io_bytes is not None else 0.0
            proc_health = {
                'alive': True, 'zombie': False,
                'cpu_percent': 1.0, 'memory_mb': self._prev_rss_mb or 0.0,
                'num_children': self._prev_num_children,
                'children_alive': max(1, self._prev_num_children),
                'cpu_time': _base_cpu + 1.0, 'io_bytes': _base_io,
            }
        elif not _fresh:
            return {'status': 'skip', 'output_changed': output_changed,
                    'stall_count': self.stall_count, 'recent_text': recent_text}
        else:
            self._last_health_seq = _sample_seq

        output_anomalies = detect_output_anomalies(recent_text)
        if proc_health['num_children'] > 0:
            self._had_children = True

        _now_health = time.time()
        if self._prev_cpu_time is None:
            progress_signals = None
        else:
            progress_signals = {
                'cpu_time_delta': proc_health.get('cpu_time', 0.0) - self._prev_cpu_time,
                'io_bytes_delta': proc_health.get('io_bytes', 0.0) - self._prev_io_bytes,
                'rss_delta': (proc_health.get('memory_mb', 0.0) - self._prev_rss_mb) * 1024 * 1024,
            }
            _wall_dt = _now_health - self._prev_health_time if self._prev_health_time else 0.0
            if _wall_dt > 0:
                self._tree_cpu_pct = max(
                    0.0, progress_signals['cpu_time_delta'] / _wall_dt * 100.0,
                )
        self._prev_cpu_time = proc_health.get('cpu_time', 0.0)
        self._prev_io_bytes = proc_health.get('io_bytes', 0.0)
        self._prev_rss_mb = proc_health.get('memory_mb', 0.0)
        self._prev_health_time = _now_health

        if progress_signals is None:
            is_idle = False
        else:
            counters_flat = (
                progress_signals.get('cpu_time_delta', 0.0) <= 0.5
                and progress_signals.get('io_bytes_delta', 0.0) <= (1 << 20)
                and abs(progress_signals.get('rss_delta', 0.0)) <= (1 << 20)
            )
            is_idle = (not output_changed) and counters_flat
        if is_idle:
            self.idle_count += 1
        else:
            self.idle_count = 0

        should_kill, kill_reason = should_kill_process(
            elapsed, output_changed, self.stall_count,
            proc_health, output_anomalies, self._had_children,
            self._prev_num_children,
            progress_signals=progress_signals,
            idle_count=self.idle_count,
        )
        self._prev_num_children = proc_health['num_children']

        _cpu_display = (
            self._tree_cpu_pct if progress_signals is not None
            else proc_health['cpu_percent']
        )
        activity = (
            f"CPU {_cpu_display:.0f}% (whole process tree), "
            f"memory {proc_health['memory_mb']:.0f} MB (whole tree), "
            f"live child processes {proc_health['children_alive']}"
            f" of {proc_health['num_children']}"
        )

        return {
            'status': 'ok',
            'should_kill': should_kill,
            'kill_reason': kill_reason,
            'activity': activity,
            'output_changed': output_changed,
            'stall_count': self.stall_count,
            'recent_text': recent_text,
        }


# --- Self-kill protection ---

_SELF_KILL_RE = re.compile(
    r"\bkill\b.*\b(flagscale|agent\.py|react/agent)\b"
    r"|\bgrep\b.*\b(flagscale|agent\.py)\b.*\bkill\b"
    r"|\bpkill\b.*\b(flagscale|agent)\b"
    r"|\bkillall\b.*\b(flagscale|agent)\b",
)


def _get_agent_pids():
    """Get PIDs of the agent process tree that must not be killed."""
    agent_pid = os.getpid()
    ppid = os.getppid()
    exclude = {agent_pid, ppid}
    try:
        with open(f"/proc/{ppid}/stat") as f:
            pppid = int(f.read().split()[3])
            exclude.add(pppid)
    except (OSError, ValueError, IndexError):
        pass
    try:
        result = subprocess.run(
            f"pgrep -P {agent_pid}", shell=True,
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            try:
                exclude.add(int(line.strip()))
            except ValueError:
                pass
    except Exception:
        pass
    return exclude


def _protect_self_kill(command: str) -> str:
    """Rewrite kill pipelines to exclude the agent's own process tree."""
    exclude = _get_agent_pids()
    pids_str = "|".join(str(p) for p in sorted(exclude))

    pkill_re = re.compile(r"\b(pkill|killall)\s+(-\S+\s+)*(flagscale\S*|agent\S*)")
    m = pkill_re.search(command)
    if m:
        signal_flag = m.group(2) or ""
        pattern = m.group(3)
        kill_sig = "-9" if "-9" in signal_flag else ""
        replacement = (
            f"ps aux | grep '{pattern}' | grep -v grep"
            f" | awk '{{print $2}}'"
            f" | grep -Ev '\\b({pids_str})\\b'"
            f" | xargs -r kill {kill_sig}"
        )
        command = command[:m.start()] + replacement + command[m.end():]
        return command

    if "xargs" in command and "kill" in command:
        pid_filter = f"grep -Ev '\\b({pids_str})\\b' | "
        command = re.sub(
            r'\|\s*xargs\s+(-r\s+)?kill',
            lambda m: f"| {pid_filter}xargs {m.group(1) or ''}kill",
            command,
        )

    return command


# --- Trailing pipe optimization ---

_TRAILING_PIPE_RE = re.compile(
    r"\|\s*(tail|head)\s+-n\s*(\d+)\s*$"
    r"|\|\s*(tail|head)\s+-(\d+)\s*$"
    r"|\|\s*(tail|head)\s*$"
)


def _strip_trailing_pipe(command: str):
    """Strip trailing | tail -N / | head -N and return (new_cmd, post_fn).

    Applies the equivalent truncation in Python so we get real-time output
    from the main command instead of buffering in tail/head.
    """
    m = _TRAILING_PIPE_RE.search(command)
    if not m:
        return command, None

    cmd_name = m.group(1) or m.group(3) or m.group(5)
    count_str = m.group(2) or m.group(4)
    count = int(count_str) if count_str else 10

    stripped = command[:m.start()].rstrip()
    stripped = re.sub(r'\s*2>&1\s*$', '', stripped)

    if cmd_name == "tail":
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[-count:]) if len(lines) > count else output
    else:
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[:count]) if len(lines) > count else output

    return stripped, post_fn


# --- Background job registry ---
#
# A long-running command can be launched with background=true, or a healthy but
# long command can be DETACHED from the synchronous monitor loop by the health
# judge (kill→background). Both paths hand the running process to this registry
# so the agent is freed to do other work instead of blocking. The registry is a
# process-wide singleton because both ShellTool (which detaches) and
# ShellJobsTool (which polls/waits/kills) must see the same jobs.

class _BackgroundJob:
    """A detached/background shell command still running under reader threads.

    The reader threads keep appending to stdout_chunks/stderr_chunks after
    detach — they are NOT joined or killed. poll() reports incremental output
    since the last poll via a per-job read cursor over the chunk lists.
    """

    def __init__(self, job_id, command, proc, stdout_chunks, stderr_chunks,
                 t_out, t_err, start, health_sampler=None):
        self.job_id = job_id
        self.command = command
        self.proc = proc
        self.stdout_chunks = stdout_chunks
        self.stderr_chunks = stderr_chunks
        self.t_out = t_out
        self.t_err = t_err
        self.start = start
        self.health_sampler = health_sampler
        # Per-job health evaluator: when a live sampler is attached (the job was
        # detached/backgrounded WITHOUT stopping monitoring), shell_jobs uses
        # this to run the same liveness/kill logic the synchronous loop runs, so
        # a backgrounded job that later hangs still gets caught. None when the
        # job has no live sampler (nothing to evaluate).
        self.evaluator = _HealthEvaluator() if health_sampler is not None else None
        # Last human-readable health note (activity / advisory), surfaced by poll.
        self.health_note = ""
        # Read cursors: how many chunks the caller has already seen.
        self._out_cursor = 0
        self._err_cursor = 0
        self.finished_at = None  # wall time the proc was first observed exited

    @property
    def running(self):
        return self.proc.poll() is None

    @property
    def returncode(self):
        return self.proc.poll()

    def status_str(self):
        if self.running:
            return "running"
        rc = self.returncode
        return "done" if rc == 0 else f"exited({rc})"

    def new_output(self):
        """Return output appended since the last poll, advancing the cursors."""
        out = "".join(self.stdout_chunks[self._out_cursor:])
        err = "".join(self.stderr_chunks[self._err_cursor:])
        self._out_cursor = len(self.stdout_chunks)
        self._err_cursor = len(self.stderr_chunks)
        return out + err

    def full_output(self):
        return "".join(self.stdout_chunks) + "".join(self.stderr_chunks)


class _JobRegistry:
    """Process-wide registry of background jobs, keyed by job_id."""

    def __init__(self):
        self._jobs: dict[str, _BackgroundJob] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def add(self, command, proc, stdout_chunks, stderr_chunks,
            t_out, t_err, start, health_sampler=None):
        with self._lock:
            self._counter += 1
            job_id = f"job{self._counter}"
            self._jobs[job_id] = _BackgroundJob(
                job_id, command, proc, stdout_chunks, stderr_chunks,
                t_out, t_err, start, health_sampler,
            )
            return job_id

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def all(self):
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id):
        with self._lock:
            return self._jobs.pop(job_id, None)

    def cleanup_all(self):
        """Kill every still-running job. Called at agent shutdown to avoid
        leaving orphaned/zombie processes behind."""
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            try:
                if job.health_sampler is not None:
                    job.health_sampler.stop()
            except Exception:
                pass
            try:
                if job.proc.poll() is None:
                    job.proc.kill()
                    job.proc.wait(timeout=3)
            except Exception:
                pass


# Process-wide singleton.
_JOB_REGISTRY = _JobRegistry()


# --- ShellTool ---

class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return its output (stdout + stderr)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Launch as a long-running background job and return a job "
                    "handle IMMEDIATELY instead of blocking. Use this for any "
                    "command you expect to take more than a minute or two "
                    "(compiling a large project, training, big downloads) so you "
                    "are freed to make progress on other steps. Poll it later "
                    "with the shell_jobs tool. Default: false (run synchronously)."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(self, remind_interval: int = 120, env: dict = None,
                 health_judge_fn=None):
        self._remind_interval = remind_interval
        self._env = env or {}
        self._health_judge_fn = health_judge_fn
        # Per-prefix command history for the LLM health judge. Maps a command
        # prefix (first 2 tokens) to a list of (outcome, duration_secs) tuples.
        # outcome is "completed", "killed", or "error".
        self._command_history: dict[str, list[tuple[str, float]]] = {}
        # Container resources cached once (memory, cpu count).
        self._cached_resources: str | None = None

    @staticmethod
    def _cmd_prefix(command: str) -> str:
        """Extract a 2-token prefix from a command for history grouping."""
        toks = command.strip().split()[:2]
        return " ".join(toks)

    def _record_history(self, command: str, outcome: str, duration: float):
        """Record the outcome of a command execution for the LLM judge."""
        prefix = self._cmd_prefix(command)
        if prefix not in self._command_history:
            self._command_history[prefix] = []
        self._command_history[prefix].append((outcome, duration))
        # Keep only the last 10 entries per prefix to bound memory.
        if len(self._command_history[prefix]) > 10:
            self._command_history[prefix] = self._command_history[prefix][-10:]

    def _build_history_str(self, command: str) -> str:
        """Build a human-readable history summary for the LLM judge."""
        prefix = self._cmd_prefix(command)
        entries = self._command_history.get(prefix, [])
        if not entries:
            return ""
        parts = []
        for outcome, dur in entries:
            parts.append(f"{outcome} in {dur:.0f}s")
        return f"{prefix}: {len(entries)} prior run(s) — {', '.join(parts)}"

    def _detect_resources(self) -> str:
        """Detect container resources (memory, CPU count) for the LLM judge."""
        if self._cached_resources is not None:
            return self._cached_resources
        try:
            import multiprocessing
            nproc = multiprocessing.cpu_count()
            # Read memory from /proc/meminfo (Linux containers)
            mem_kb = None
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = int(line.split()[1])
                            break
            except Exception:
                pass
            if mem_kb:
                mem_gb = mem_kb / (1024 * 1024)
                self._cached_resources = f"Memory: {mem_gb:.0f}GB, CPU: {nproc} cores"
            else:
                self._cached_resources = f"CPU: {nproc} cores (memory unknown)"
        except Exception:
            self._cached_resources = ""
        return self._cached_resources

    def _detach_to_background(self, command, proc, stdout_chunks,
                              stderr_chunks, t_out, t_err, start,
                              health_sampler, reason=""):
        """Detach a healthy-but-long running command to the background.

        Called from the monitor loop when the health judge decides the command
        is healthy and making progress but is long enough that the agent should
        not keep blocking on it. Unlike a kill, the process KEEPS RUNNING and
        stays MONITORED: we hand the still-running health sampler to the
        registry job (we do NOT stop it) so shell_jobs can keep evaluating the
        job's liveness and catch a later hang. We do NOT kill the process and
        do NOT join the reader threads — they keep draining the pipes into the
        same chunk lists the registry job holds, so later poll/wait sees the
        full output. Returns the DETACHED handle string the agent gets in place
        of the (would-be) blocking result.
        """
        # Keep the sampler ALIVE and hand it to the registry: a backgrounded job
        # remains monitored so shell_jobs(wait/poll) can still detect a hang.
        job_id = _JOB_REGISTRY.add(
            command, proc, stdout_chunks, stderr_chunks,
            t_out, t_err, start, health_sampler=health_sampler,
        )
        elapsed = int(time.time() - start)
        self._record_history(command, "backgrounded", time.time() - start)
        reason_line = f"Reason: {reason}\n" if reason else ""
        return (
            f"DETACHED as {job_id} (after {elapsed}s): {command}\n"
            f"{reason_line}"
            f"The health monitor judged this command healthy and still making "
            f"progress, but long enough that blocking on it wastes your time. "
            f"It is now running in the background; you are NOT blocked and "
            f"NOTHING was killed. Continue with other useful work now. Check "
            f"progress later with the shell_jobs tool: "
            f"shell_jobs(action=\"poll\", job_id=\"{job_id}\") for incremental "
            f"output, or shell_jobs(action=\"wait\", job_id=\"{job_id}\") to "
            f"block until it finishes. Do NOT relaunch it — it is already "
            f"running."
        )

    def execute(self, **kwargs) -> str:
        command = kwargs.get("command", "")
        if not command:
            return "ERROR: 'command' parameter is required but was empty or missing (possible output truncation)."
        if not isinstance(command, str):
            return f"ERROR: shell command must be a string, got {type(command).__name__}: {repr(command)[:200]}"

        quiet = kwargs.pop("_quiet", False)
        background = bool(kwargs.get("background", False))

        # Self-kill protection
        if _SELF_KILL_RE.search(command):
            command = _protect_self_kill(command)

        # Trailing pipe optimization
        command, post_fn = _strip_trailing_pipe(command)

        try:
            run_env = {**os.environ, **self._env} if self._env else dict(os.environ)
            # Encourage line-buffered output from children so long-running
            # commands (make/gcc/etc.) stream progress incrementally instead of
            # full-buffering into the pipe and dumping everything only at exit.
            run_env.setdefault("PYTHONUNBUFFERED", "1")

            # Prefer `stdbuf -oL -eL` when available: it sets _STDBUF_* / LD_PRELOAD
            # env vars that propagate to ALL descendants, forcing line-buffered
            # stdout/stderr transitively (make, gcc, ...). Fall back to a plain
            # shell=True invocation when stdbuf is missing.
            stdbuf = shutil.which("stdbuf")
            if stdbuf:
                popen_args = [stdbuf, "-oL", "-eL", "/bin/sh", "-c", command]
                popen_kwargs = {"shell": False}
            else:
                popen_args = command
                popen_kwargs = {"shell": True}

            proc = subprocess.Popen(
                popen_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=run_env,
                **popen_kwargs,
            )

            stdout_chunks: list = []
            stderr_chunks: list = []

            def _read_stream(stream, buf):
                for line in stream:
                    buf.append(line)

            t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True)
            t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True)
            t_out.start()
            t_err.start()

            # ── Background launch: register and return a handle immediately ──
            # The reader threads keep draining the pipes; the process runs on
            # under the registry. We do NOT run the synchronous monitor loop —
            # the whole point is to not block. But we DO start the health
            # sampler and hand it to the registry job, so shell_jobs(wait/poll)
            # can still evaluate the job's liveness and catch a later hang. The
            # sampler is the same O(1)-to-read background thread the sync loop
            # uses; it costs one thread and does not block the agent.
            if background:
                start = time.time()
                from flagscale_agent.react.tools.process_health import (
                    get_process_health,
                )
                _sampler_interval = min(_PROCESS_HEALTH_TIMEOUT_SECS,
                                        min(30, self._remind_interval))
                _bg_sampler = _HealthSampler(
                    get_process_health, proc.pid, interval=_sampler_interval,
                ).start()
                job_id = _JOB_REGISTRY.add(
                    command, proc, stdout_chunks, stderr_chunks,
                    t_out, t_err, start, health_sampler=_bg_sampler,
                )
                self._record_history(command, "backgrounded", 0.0)
                return (
                    f"BACKGROUNDED as {job_id}: {command}\n"
                    f"The command is running in the background; you are NOT "
                    f"blocked. Continue with other steps now. Check progress "
                    f"later with the shell_jobs tool: "
                    f"shell_jobs(action=\"poll\", job_id=\"{job_id}\") for "
                    f"incremental output, or shell_jobs(action=\"wait\", "
                    f"job_id=\"{job_id}\") to block until it finishes. "
                    f"Do NOT relaunch it — it is already running."
                )

            # Background health sampler — see _HealthSampler docstring. The
            # main loop below NEVER calls get_process_health directly; it only
            # reads sampler.latest(), which cannot block. This is what keeps
            # the heartbeat firing on schedule even when a single /proc sample
            # takes minutes under a giant build tree.
            from flagscale_agent.react.tools.process_health import (
                get_process_health,
                detect_output_anomalies,
                should_kill_process,
            )
            # Sampler cadence tracks the check cadence but is capped at
            # _PROCESS_HEALTH_TIMEOUT_SECS. In production the check interval is
            # 30s so the sampler runs every 5s (plenty fresh per check). When a
            # caller/test uses a tighter remind_interval (e.g. 1s) the sampler
            # matches it so every check gets a fresh sample and idle-kill timing
            # stays responsive. Under a giant build tree a sample naturally
            # takes longer than this interval — the sampler just runs as fast as
            # /proc allows; the interval is only the MINIMUM sleep between samples.
            _sampler_interval = min(_PROCESS_HEALTH_TIMEOUT_SECS,
                                    min(30, self._remind_interval))
            _health_sampler = _HealthSampler(
                get_process_health, proc.pid,
                interval=_sampler_interval,
            ).start()

            # --- Long-command monitoring loop ---
            start = time.time()
            # Default cadence is 30s; honor a smaller remind_interval so callers
            # (and tests) can request tighter checks. With the default
            # remind_interval (120) this is 30s.
            check_interval = min(30, self._remind_interval)
            next_check = check_interval
            last_output_snapshot = ""
            stall_count = 0
            health_reason = ""
            _streams_done_at = None
            _STREAM_EOF_GRACE_SECS = 3
            _had_children = False  # Track if process ever had children
            _prev_num_children = 0  # Track child count drops (OOM signal)
            _prev_cpu_time = None   # Cumulative CPU secs (liveness signal A)
            _prev_io_bytes = None   # Cumulative IO bytes (liveness signal A)
            _prev_rss_mb = None     # RSS MB (liveness signal A)
            _prev_health_time = None  # Wall clock of the last fresh health sample
            _tree_cpu_pct = 0.0     # Tree-wide CPU% derived from cpu_time delta
            idle_count = 0          # Consecutive no-output + flat-counter samples
            _last_health_seq = -1   # Seq of the last health sample we acted on

            while proc.poll() is None:
                elapsed = time.time() - start

                if elapsed > next_check:
                    next_check = elapsed + check_interval
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}m{secs}s" if mins > 0 else f"{secs}s"
                    recent_text = "".join(stdout_chunks[-20:] + stderr_chunks[-20:])
                    current_snapshot = "".join(stdout_chunks[-10:] + stderr_chunks[-10:])

                    # ── Heartbeat FIRST (before any health checks) ──
                    # The heartbeat must fire unconditionally every check_interval,
                    # independent of whether the health checks below block or stall.
                    # Previously the heartbeat was at the END of the health block;
                    # if get_process_health or the LLM judge blocked (even with
                    # timeouts, GIL contention from psutil C calls or API latency
                    # could delay the main thread), the heartbeat was suppressed
                    # for minutes — making a healthy long build look hung AND
                    # hiding the fact that the monitor itself was alive.
                    if not quiet:
                        from flagscale_agent.react import display
                        recent_hb = stdout_chunks[-5:] + stderr_chunks[-5:]
                        if hasattr(display, '_active_spinner') and display._active_spinner:
                            display._active_spinner.stop()
                        health_note_hb = f"\n   🩺 {health_reason}\n" if health_reason else ""
                        if recent_hb:
                            header_hb = f"\033[2m   ⏳ [{time_str}]{health_note_hb}   Recent output:\033[0m"
                            lines_out_hb = [header_hb]
                            for line_hb in recent_hb[-5:]:
                                lines_out_hb.append(f"\033[2m   │ {line_hb.rstrip()}\033[0m")
                        else:
                            lines_out_hb = [
                                f"\033[2m   ⏳ [{time_str}]{health_note_hb}   "
                                f"(running, no output yet)\033[0m"
                            ]
                        if hasattr(display, '_stdout_lock'):
                            with display._stdout_lock:
                                sys.stdout.write("\n".join(lines_out_hb) + "\n")
                                sys.stdout.flush()
                        else:
                            sys.stdout.write("\n".join(lines_out_hb) + "\n")
                            sys.stdout.flush()
                        if hasattr(display, '_active_spinner') and display._active_spinner:
                            display._active_spinner.start()

                    # Track output changes. Empty output counts as a STALL,
                    # not as "changed" — otherwise commands that never stream
                    # output (piped through tail/grep, silent network waits)
                    # look permanently healthy and stall indicators never fire.
                    output_changed = bool(current_snapshot) and current_snapshot != last_output_snapshot
                    if not output_changed:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_output_snapshot = current_snapshot

                    # Read the latest health sample from the background sampler.
                    # This is an O(1) lock-guarded dict copy — it NEVER blocks on
                    # /proc, so the heartbeat above always fires on schedule no
                    # matter how slow a real sample is under a giant build tree.
                    proc_health, _sample_age, _sample_seq = _health_sampler.latest()
                    # Liveness deltas (cpu_time/io/rss) are only meaningful
                    # between two DISTINCT samples. The heartbeat cadence
                    # (remind_interval, as low as 1s) can be much faster than
                    # the sampler interval (5s), so several loop iterations read
                    # the SAME snapshot. Computing a delta against an unchanged
                    # snapshot always yields 0 → a busy CPU-loop with no output
                    # would look idle and get false-killed. So: only act on a
                    # FRESH sample (seq advanced). A missing sample (seq==-1
                    # never advanced) or a STALE one (sampler stuck minutes on a
                    # giant /proc walk) also falls back to the neutral reading.
                    _fresh = proc_health is not None and _sample_seq != _last_health_seq
                    _stale = (
                        _sample_age is not None
                        and _sample_age > 3 * _PROCESS_HEALTH_TIMEOUT_SECS
                    )
                    # THREE cases:
                    #  (a) STALE: a real sample exists but the sampler is stuck
                    #      minutes on a giant /proc walk. Use a neutral "working"
                    #      reading (cpu_time bumped) so we never false-kill a
                    #      healthy build just because /proc is slow. Evaluate.
                    #  (b) FRESH: a new distinct sample — evaluate deltas normally.
                    #  (c) NON-FRESH & not stale: heartbeat cadence outran the
                    #      sampler (e.g. 1s checks vs a sample still in flight).
                    #      There is NO new health info, so we must NOT evaluate —
                    #      fabricating a reading here would either reset idle_count
                    #      (masking a real hang) or accumulate it (false-kill).
                    #      Skip the whole health/kill block; idle_count is left
                    #      untouched until the next fresh sample arrives.
                    if _stale:
                        _base_cpu = _prev_cpu_time if _prev_cpu_time is not None else 0.0
                        _base_io = _prev_io_bytes if _prev_io_bytes is not None else 0.0
                        proc_health = {
                            'alive': True, 'zombie': False,
                            'cpu_percent': 1.0, 'memory_mb': _prev_rss_mb or 0.0,
                            'num_children': _prev_num_children,
                            'children_alive': max(1, _prev_num_children),
                            'cpu_time': _base_cpu + 1.0, 'io_bytes': _base_io,
                        }
                    elif not _fresh:
                        # No fresh health data this tick — heartbeat already fired
                        # above; just wait for the next sample. Do not touch
                        # idle_count or _prev_* counters.
                        time.sleep(0.2)
                        continue
                    else:
                        _last_health_seq = _sample_seq
                    output_anomalies = detect_output_anomalies(recent_text)
                    
                    # Track if we ever had children
                    if proc_health['num_children'] > 0:
                        _had_children = True

                    # Liveness deltas across sampling points (support A):
                    # cumulative CPU time / IO bytes are monotonic counters, so
                    # a positive delta proves the process is actually working —
                    # this vetoes heuristic kills. None on the first sample.
                    _now_health = time.time()
                    if _prev_cpu_time is None:
                        progress_signals = None  # no baseline yet
                    else:
                        progress_signals = {
                            'cpu_time_delta': proc_health.get('cpu_time', 0.0) - _prev_cpu_time,
                            'io_bytes_delta': proc_health.get('io_bytes', 0.0) - _prev_io_bytes,
                            'rss_delta': (proc_health.get('memory_mb', 0.0) - _prev_rss_mb) * 1024 * 1024,
                        }
                        # Tree-wide CPU% for the human-readable activity string
                        # shown to the LLM judge. The parent-only cpu_percent
                        # from psutil reads ~0% for a launcher/wrapper parent
                        # even while its children saturate cores. Deriving it
                        # from the tree cpu_time delta over the wall interval
                        # (cpu_secs consumed / wall_secs elapsed * 100) reflects
                        # the WHOLE tree and costs zero extra syscalls — the
                        # cpu_time was already summed during the health walk.
                        # >100% is expected and correct for multi-core work.
                        _wall_dt = _now_health - _prev_health_time if _prev_health_time else 0.0
                        if _wall_dt > 0:
                            _tree_cpu_pct = max(
                                0.0,
                                progress_signals['cpu_time_delta'] / _wall_dt * 100.0,
                            )
                    _prev_cpu_time = proc_health.get('cpu_time', 0.0)
                    _prev_io_bytes = proc_health.get('io_bytes', 0.0)
                    _prev_rss_mb = proc_health.get('memory_mb', 0.0)
                    _prev_health_time = _now_health

                    # Idle tracking: a sample is idle when there is NO new output
                    # AND all three cumulative counters are flat (same test the
                    # liveness veto uses, inverted). idle_count accumulates only
                    # across consecutive idle samples and resets the moment the
                    # process produces output or moves any counter. When the
                    # health sample times out, progress_signals stays None (no
                    # baseline yet) OR the neutral fallback bumped cpu_time by
                    # +1.0 so cpu_time_delta > 0.5 → not idle → we never
                    # false-kill on a slow /proc read.
                    if progress_signals is None:
                        is_idle = False  # no baseline delta yet — cannot judge
                    else:
                        counters_flat = (
                            progress_signals.get('cpu_time_delta', 0.0) <= 0.5
                            and progress_signals.get('io_bytes_delta', 0.0) <= (1 << 20)
                            and abs(progress_signals.get('rss_delta', 0.0)) <= (1 << 20)
                        )
                        is_idle = (not output_changed) and counters_flat
                    if is_idle:
                        idle_count += 1
                    else:
                        idle_count = 0

                    should_kill, kill_reason = should_kill_process(
                        elapsed, output_changed, stall_count,
                        proc_health, output_anomalies, _had_children,
                        _prev_num_children,
                        progress_signals=progress_signals,
                        idle_count=idle_count,
                    )
                    
                    # Update child count for next iteration
                    _prev_num_children = proc_health['num_children']

                    if should_kill:
                        _health_sampler.stop()
                        proc.kill()
                        t_out.join(timeout=2)
                        t_err.join(timeout=2)
                        partial = _truncate_output("".join(stdout_chunks) + "".join(stderr_chunks))
                        self._record_history(command, "killed", time.time() - start)
                        return (
                            f"TERMINATED: {kill_reason} (after {time_str}).\n"
                            f"This was stopped by the health monitor, not by "
                            f"the command itself. Do not relaunch a variant of "
                            f"the same approach — treat the reason above as a "
                            f"redirection and change your method class before "
                            f"running anything similar again.\n"
                            f"Output:\n{partial}"
                        )

                    # Advisory from hard indicators (e.g. silent stall) —
                    # pass as context to the LLM judge, which has richer
                    # information (command history, container resources,
                    # output pattern) to make a better-informed decision.
                    if kill_reason:
                        health_reason = kill_reason

                    # Fallback: LLM judge (if hard indicators passed).
                    # Pass the live resource signals so the judge can tell a
                    # genuinely hung process (no output + no CPU/child work)
                    # apart from a healthy silent one (no output but actively
                    # computing) instead of guessing from silence alone.
                    if self._health_judge_fn:
                        # Report TREE-wide CPU% and memory, not the parent-only
                        # psutil snapshot. A launcher/wrapper parent (bash sleep,
                        # a training driver that forks workers) reads ~0% CPU and
                        # ~2 MB RSS on its own even while children saturate cores
                        # and use gigabytes — which made the judge repeatedly
                        # (mis)report "CPU is 0% with only 2 MB memory" and
                        # false-flag healthy jobs as hung. _tree_cpu_pct is
                        # derived from the cpu_time delta over the wall interval;
                        # memory_mb is now the summed tree RSS. On the very first
                        # sample (no delta baseline yet) _tree_cpu_pct is 0.0, so
                        # fall back to the parent snapshot to avoid a misleading
                        # hard 0%.
                        _cpu_display = (
                            _tree_cpu_pct if progress_signals is not None
                            else proc_health['cpu_percent']
                        )
                        activity = (
                            f"CPU {_cpu_display:.0f}% (whole process tree), "
                            f"memory {proc_health['memory_mb']:.0f} MB (whole tree), "
                            f"live child processes {proc_health['children_alive']}"
                            f" of {proc_health['num_children']}"
                        )
                        # Bounded call: a stalled LLM gateway must never freeze
                        # the monitor loop. On timeout we get None and skip the
                        # judge this round, so the heartbeat below still fires on
                        # schedule and hard-indicator checks keep running.
                        decision = _run_health_judge_bounded(
                            self._health_judge_fn,
                            (command, recent_text, time_str),
                            {
                                "output_changed": output_changed,
                                "stall_count": stall_count,
                                "activity": activity,
                                "command_history": self._build_history_str(command),
                                "container_resources": self._detect_resources(),
                                "health_advisory": health_reason,
                            },
                        ) or {"action": "continue", "kill": False}
                        _action = decision.get("action")
                        if _action is None:
                            # Legacy shape: derive action from the kill flag.
                            _action = "kill" if decision.get("kill") else "continue"
                        if _action == "background":
                            # Healthy but long: detach and free the agent instead
                            # of killing. The process keeps running as a job.
                            return self._detach_to_background(
                                command, proc, stdout_chunks, stderr_chunks,
                                t_out, t_err, start, _health_sampler,
                                reason=decision.get("reason", ""),
                            )
                        if _action == "kill":
                            _health_sampler.stop()
                            proc.kill()
                            t_out.join(timeout=2)
                            t_err.join(timeout=2)
                            partial = _truncate_output("".join(stdout_chunks) + "".join(stderr_chunks))
                            reason = decision.get("reason", "Unhealthy command")
                            self._record_history(command, "killed", time.time() - start)
                            return (
                                f"TERMINATED: {reason} (after {time_str}).\n"
                                f"This was stopped by the health monitor, not by "
                                f"the command itself. Do not relaunch a variant of "
                                f"the same approach — treat the reason above as a "
                                f"redirection and change your method class before "
                                f"running anything similar again.\n"
                                f"Output:\n{partial}"
                            )
                        else:
                            reason = decision.get("reason", "")
                            health_reason = reason
                            if reason and not quiet:
                                from flagscale_agent.react import display
                                if hasattr(display, '_active_spinner') and display._active_spinner:
                                    display._active_spinner.set_hint(f"🩺 {reason}")

                # Ctrl-C handling
                try:
                    pass  # KeyboardInterrupt is caught in outer try
                except KeyboardInterrupt:
                    proc.kill()
                    proc.wait(timeout=3)
                    return f"TERMINATED by user after {int(elapsed)}s."

                # EOF grace kill — prevent zombie processes
                if not t_out.is_alive() and not t_err.is_alive():
                    if _streams_done_at is None:
                        _streams_done_at = time.time()
                    elif time.time() - _streams_done_at > _STREAM_EOF_GRACE_SECS:
                        proc.kill()
                        proc.wait(timeout=3)
                        break
                else:
                    _streams_done_at = None

                time.sleep(0.2)

            # Drain the reader threads. When the pipe closes normally these
            # return in milliseconds. But if the command backgrounded a child
            # that INHERITED the stdout/stderr write end (e.g. `cmd; daemon &`),
            # the pipe never sees EOF even though proc.poll() already returned —
            # the read threads' `for line in stream` blocks on the orphaned
            # write end. The process is DONE; its own output was already flushed
            # the moment it exited, so we only need a brief grace to drain the
            # buffer, not the full 5s per stream. A short bounded join keeps the
            # tool responsive (returns in <1s instead of 10s) while still
            # collecting any last buffered lines. The threads are daemon, so a
            # still-blocked reader is abandoned cleanly at process exit.
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
            _health_sampler.stop()

            # --- Post-execution: assemble result ---
            elapsed_total = time.time() - start
            output = ""
            if stdout_chunks:
                output += "".join(stdout_chunks)
            if stderr_chunks:
                output += "".join(stderr_chunks)
            if not output:
                output = "(no output)"
            if post_fn and output != "(no output)":
                output = post_fn(output)
            # Bound the returned output so a chatty build can never flood the
            # context or blow memory (see _truncate_output). Applied AFTER
            # post_fn so a trailing head/tail pipe still sees full output first.
            output = _truncate_output(output)

            # Surface the RAW process exit code when the command did not exit 0.
            # We report the fact, not an interpretation — no mapping of specific
            # codes to causes (e.g. 137 → OOM) lives here or in the prompt. The
            # LLM sees the actual code and judges what it means, because the same
            # code means different things across programs and platforms. A clean
            # exit (0) adds no line, to keep normal output uncluttered.
            rc = proc.poll() if proc is not None else None
            if rc is not None and rc != 0 and not output.startswith(("TERMINATED", "ERROR")):
                output = f"{output}\n[exit code: {rc}]"

            # Record command outcome for future health-judge context.
            outcome = "completed"
            if output.startswith("TERMINATED"):
                outcome = "killed"
            elif output.startswith("ERROR"):
                outcome = "error"
            self._record_history(command, outcome, elapsed_total)

            return output

        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
            partial = "".join(stdout_chunks) + "".join(stderr_chunks) if (
                "stdout_chunks" in locals() and "stderr_chunks" in locals()
            ) else ""
            raise
        except Exception as e:
            return f"ERROR: {e}"


# --- ShellJobsTool ---

class ShellJobsTool(Tool):
    """Manage background shell jobs launched via shell(background=true) or
    detached from the synchronous monitor by the health judge (kill→background).

    This is the polling/control counterpart to ShellTool's launch side. It never
    blocks longer than the caller-supplied timeout on `wait`, so a runaway
    background job cannot silently burn the whole task budget.
    """

    name = "shell_jobs"
    description = (
        "List, poll, wait on, or kill background shell jobs (started with "
        "shell(background=true) or auto-detached from a long healthy command). "
        "Use 'poll' to read incremental output without blocking, 'wait' to block "
        "until a job finishes (bounded by timeout), 'list' to see all jobs, and "
        "'kill' to terminate one."
    )
    def __init__(self, health_judge_fn=None, history_fn=None, resources_fn=None):
        # Optional LLM health judge (same one ShellTool uses). When provided,
        # wait ticks run it as a soft fallback after the hard indicators, so a
        # backgrounded job that later hangs is caught even if the hard rules
        # miss it. history_fn/resources_fn give the judge the same context the
        # sync loop passes; both optional (judge tolerates empty strings).
        self._health_judge_fn = health_judge_fn
        self._history_fn = history_fn
        self._resources_fn = resources_fn

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "wait", "kill"],
                "description": (
                    "list: show all background jobs and their status. "
                    "poll: return output appended since the last poll (does NOT "
                    "block). wait: block until the job exits or `timeout` seconds "
                    "elapse. kill: terminate the job."
                ),
            },
            "job_id": {
                "type": "string",
                "description": (
                    "Job handle (e.g. 'job1') from a BACKGROUNDED/DETACHED "
                    "message. Required for poll/wait/kill; ignored for list."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "For action='wait': max seconds to block before returning "
                    "control (the job keeps running if still unfinished). "
                    "Default 60. Keep this small and poll in a loop so you never "
                    "block on a single call for long."
                ),
            },
        },
        "required": ["action"],
    }

    def _fmt_job_line(self, job):
        elapsed = time.time() - job.start
        return f"{job.job_id}: [{job.status_str()}] {elapsed:.0f}s  {job.command[:80]}"

    def _set_wait_hint(self, job, activity="", finished=False):
        """Push a live one-line progress hint onto the active spinner during a
        'wait', so the terminal shows what the job is doing instead of a frozen
        '🧵 shell_jobs wait' line. No-op when there is no color spinner."""
        try:
            from flagscale_agent.react import display
        except Exception:
            return
        spinner = getattr(display, "_active_spinner", None)
        if spinner is None:
            return
        parts = [f"{job.job_id} [{job.status_str()}]"]
        # Latest non-empty output line, peeked WITHOUT advancing the read
        # cursors (new_output would consume what the final return needs). Scan
        # only the last few chunks to stay O(1) on long-running jobs.
        try:
            recent = (job.stdout_chunks[-5:] + job.stderr_chunks[-5:])
        except Exception:
            recent = []
        last_line = ""
        for chunk in reversed(recent):
            for ln in reversed(chunk.splitlines()):
                if ln.strip():
                    last_line = ln.strip()
                    break
            if last_line:
                break
        if last_line:
            parts.append(f"│ {last_line[:80]}")
        if activity:
            parts.append(f"🩺 {activity[:60]}")
        if finished:
            parts.append("✓ finished")
        try:
            spinner.set_hint("  ".join(parts))
        except Exception:
            pass

    def _health_tick(self, job):
        """Run one health/liveness tick against a backgrounded job's live
        sampler and return {'kill': bool, 'reason': str, 'activity': str}.

        Mirrors the synchronous monitor loop: first the hard indicators
        (should_kill_process, inside _HealthEvaluator.evaluate), then — only if
        the hard rules passed — the optional LLM judge as a soft fallback. A
        backgrounded job is already detached, so the only actionable verdict
        here is 'kill'; 'background' is a no-op (already backgrounded) and
        'continue' just refreshes the advisory note.
        """
        noop = {'kill': False, 'reason': '', 'activity': job.health_note}
        if job.evaluator is None or job.health_sampler is None:
            return noop
        elapsed = time.time() - job.start
        res = job.evaluator.evaluate(
            job.health_sampler, job.stdout_chunks, job.stderr_chunks, elapsed,
        )
        if res.get('status') != 'ok':
            # No fresh sample this tick — keep the last note, do nothing.
            return {'kill': False, 'reason': '', 'activity': job.health_note}
        job.health_note = res['activity']
        if res['should_kill']:
            return {'kill': True, 'reason': res['kill_reason'],
                    'activity': res['activity']}
        # Hard indicators passed. Soft fallback: LLM judge (if wired), same
        # bounded call the sync loop uses, so a stalled gateway can't freeze us.
        if self._health_judge_fn:
            time_str = f"{int(elapsed)}s"
            decision = _run_health_judge_bounded(
                self._health_judge_fn,
                (job.command, res['recent_text'], time_str),
                {
                    "output_changed": res['output_changed'],
                    "stall_count": res['stall_count'],
                    "activity": res['activity'],
                    "command_history": (
                        self._history_fn(job.command) if self._history_fn else ""
                    ),
                    "container_resources": (
                        self._resources_fn() if self._resources_fn else ""
                    ),
                    "health_advisory": res['kill_reason'],
                },
            ) or {"action": "continue", "kill": False}
            _action = decision.get("action")
            if _action is None:
                _action = "kill" if decision.get("kill") else "continue"
            if _action == "kill":
                return {'kill': True,
                        'reason': decision.get("reason", "Unhealthy command"),
                        'activity': res['activity']}
            # "background" is a no-op here (already backgrounded); "continue"
            # just carries the reason forward as the surfaced note.
            _reason = decision.get("reason", "")
            if _reason:
                job.health_note = f"{res['activity']} — {_reason}"
        return {'kill': False, 'reason': '', 'activity': job.health_note}

    def _kill_unhealthy(self, job, reason):
        """Stop the sampler, kill the process, remove the job, and return the
        same TERMINATED redirection message the synchronous monitor emits."""
        elapsed = time.time() - job.start
        if job.health_sampler is not None:
            try:
                job.health_sampler.stop()
            except Exception:
                pass
        try:
            job.proc.kill()
            job.proc.wait(timeout=3)
        except Exception:
            pass
        job.t_out.join(timeout=0.5)
        job.t_err.join(timeout=0.5)
        partial = _truncate_output(job.new_output())
        _JOB_REGISTRY.remove(job.job_id)
        return (
            f"TERMINATED {job.job_id}: {reason} (after {int(elapsed)}s).\n"
            f"This was stopped by the health monitor, not by the command "
            f"itself. Do not relaunch a variant of the same approach — treat "
            f"the reason above as a redirection and change your method class "
            f"before running anything similar again.\n"
            f"Output:\n{partial}"
        )

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        job_id = kwargs.get("job_id", "")
        timeout = kwargs.get("timeout", 60)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 60

        if action == "list":
            jobs = _JOB_REGISTRY.all()
            if not jobs:
                return "No background jobs."
            return "Background jobs:\n" + "\n".join(
                self._fmt_job_line(j) for j in jobs
            )

        if action in ("poll", "wait", "kill"):
            if not job_id:
                return f"ERROR: action='{action}' requires a job_id."
            job = _JOB_REGISTRY.get(job_id)
            if job is None:
                known = ", ".join(j.job_id for j in _JOB_REGISTRY.all()) or "(none)"
                return f"ERROR: no such job '{job_id}'. Known jobs: {known}"

            if action == "poll":
                # Run a health tick first so a backgrounded job that has since
                # hung is caught even on a non-blocking poll, and so the note we
                # surface is fresh.
                tick = self._health_tick(job)
                if tick['kill'] and job.running:
                    return self._kill_unhealthy(job, tick['reason'])
                new_out = job.new_output()
                header = self._fmt_job_line(job)
                if tick['activity']:
                    header += f"  🩺 {tick['activity']}"
                body = new_out if new_out.strip() else "(no new output since last poll)"
                body = _truncate_output(body)
                if not job.running:
                    if job.health_sampler is not None:
                        try:
                            job.health_sampler.stop()
                        except Exception:
                            pass
                    _JOB_REGISTRY.remove(job_id)
                    header += "  [finished — removed from registry]"
                return f"{header}\n{body}"

            if action == "wait":
                deadline = time.time() + timeout
                # Cadence for the in-wait health tick: at most once per sampler
                # interval, but no less often than every second so a fast hang
                # is still caught within the bounded wait.
                _tick_every = 1.0
                _next_tick = time.time() + _tick_every
                while time.time() < deadline:
                    if not job.running:
                        break
                    if time.time() >= _next_tick:
                        tick = self._health_tick(job)
                        if tick['kill'] and job.running:
                            self._set_wait_hint(job, tick.get('activity', ''),
                                                 finished=False)
                            return self._kill_unhealthy(job, tick['reason'])
                        # Surface live progress on the spinner so the wait line
                        # is not a frozen "🧵 shell_jobs wait" — the user sees
                        # the job's status, latest output, and health activity.
                        self._set_wait_hint(job, tick.get('activity', ''),
                                            finished=False)
                        _next_tick = time.time() + _tick_every
                    time.sleep(min(1.0, max(0.1, deadline - time.time())))
                job.t_out.join(timeout=0.5)
                job.t_err.join(timeout=0.5)
                header = self._fmt_job_line(job)
                if job.running:
                    if job.health_note:
                        header += f"  🩺 {job.health_note}"
                    body = _truncate_output(job.new_output()) or "(still running)"
                    return (
                        f"{header}  [still running after {timeout}s wait]\n{body}\n"
                        f"Not finished yet. Poll again or wait more."
                    )
                if job.health_sampler is not None:
                    try:
                        job.health_sampler.stop()
                    except Exception:
                        pass
                new_out = _truncate_output(job.new_output())
                _JOB_REGISTRY.remove(job_id)
                return f"{header}  [finished — removed from registry]\n{new_out}"

            if action == "kill":
                if job.health_sampler is not None:
                    try:
                        job.health_sampler.stop()
                    except Exception:
                        pass
                if job.running:
                    try:
                        job.proc.kill()
                        job.proc.wait(timeout=3)
                    except Exception:
                        pass
                _JOB_REGISTRY.remove(job_id)
                return f"Killed {job_id}: {job.command[:80]}"

        return (
            f"ERROR: unknown action '{action}'. "
            f"Use one of: list, poll, wait, kill."
        )
