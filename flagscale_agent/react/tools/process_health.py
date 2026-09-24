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

"""Process health detection for shell command monitoring."""

from __future__ import annotations

import time

import psutil


# Hard cap on how many children a single health sample will walk. A giant
# parallel build (make -j256) spawns hundreds of compiler processes; walking the
# WHOLE tree means children(recursive=True) + 2 /proc reads per process = 500+
# syscalls, and every one runs Python bytecode that holds the GIL between the
# blocking read()s. On a disk saturated by the build, one full walk can take
# minutes, and while the sampler thread grinds through it the main heartbeat
# thread cannot get a GIL time-slice — so the heartbeat starves (observed: a
# 644s gap during a caffe -j256 build even though the sampler was already a
# single background thread). Capping the walk bounds per-sample cost regardless
# of tree size: the liveness signal (tree-wide cpu_time/io deltas) stays
# representative from a large sample, and num_children still reports the true
# total via len() before truncation.
MAX_CHILDREN_WALK = 64

# Yield the GIL every this-many processes during the walk so the main heartbeat
# thread gets scheduled even mid-sample. time.sleep(0) is not reliable for a GIL
# handoff in CPython; a tiny non-zero sleep forces the interpreter to release
# the GIL and reschedule waiting threads.
_GIL_YIELD_EVERY = 8
_GIL_YIELD_SECS = 0.001


# Consecutive idle samples (no output AND flat CPU/IO/RSS) before a hard kill.
# At the 30s check cadence this is ~2 minutes. A genuinely computing process
# (compile, torch import, buffered work) always moves at least one of
# cpu_time / io_bytes / rss between samples, so it never accumulates idle
# samples. A process that is byte-for-byte silent AND burns no CPU/IO for two
# straight minutes is blocked on a syscall — a dead network connect, a DNS
# black hole, a lock it will never get — not working. Killing it fast reclaims
# the minutes an agent would otherwise burn waiting on a hung curl/wget/apt.
IDLE_KILL_THRESHOLD = 4


def get_process_health(proc_pid: int) -> dict:
    """
    Get process health status using psutil.
    
    Returns:
        {
            'alive': bool,
            'zombie': bool,
            'cpu_percent': float,
            'memory_mb': float,
            'num_children': int,
            'children_alive': int,
        }
    """
    try:
        p = psutil.Process(proc_pid)
        
        # Check if zombie
        status = p.status()
        is_zombie = status == psutil.STATUS_ZOMBIE
        
        # CPU and memory (skip if zombie)
        cpu = 0.0
        mem = 0.0
        # Cumulative CPU time (user+system) and IO bytes across the whole
        # process tree. Unlike the 0.1s cpu_percent snapshot, these are
        # monotonic counters — comparing them across sampling points gives a
        # noise-free "is it actually doing work" signal (see support A of the
        # health-monitor design). Used by shell.py to compute deltas.
        cpu_time = 0.0
        io_bytes = 0.0
        # Parent's OWN instantaneous CPU% (parent-only snapshot). Kept for
        # backward compat, but NOTE it is misleading for a launcher/wrapper
        # parent (bash sleep, a training driver that forks workers) which sits
        # at ~0% while its children burn CPU. Callers should prefer the
        # tree-wide cpu_time delta for a truthful "is it working" signal and
        # the tree-wide memory_mb below for a truthful footprint. See the
        # tree-RSS accumulation in the walk loop.
        if not is_zombie:
            try:
                cpu = p.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            try:
                mem = p.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Child processes (+ accumulate tree-wide cpu_time / io_bytes).
        # num_children/alive_children report the TRUE total (cheap: one
        # children() call + is_running() checks), but the expensive per-process
        # cpu_times()/io_counters() walk below is capped at MAX_CHILDREN_WALK to
        # keep a single sample bounded on a giant build tree (see module docstring).
        try:
            children = p.children(recursive=True)
            num_children = len(children)
            alive_children = sum(1 for c in children if c.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []
            num_children = 0
            alive_children = 0

        # Tree-wide RSS. The parent-only `mem` above is ~2MB for a bash/launcher
        # wrapper even while its workers use gigabytes; summing RSS over the
        # walked tree gives the footprint a human/LLM judge expects to see.
        # Accumulated in the SAME walk as cpu_time/io_bytes — no extra syscalls.
        # Capped at MAX_CHILDREN_WALK like the counters, so it is a large-sample
        # lower bound on a giant tree, never an unbounded cost.
        tree_mem = mem  # start from the parent's own RSS
        walk_list = [p] + list(children)[:MAX_CHILDREN_WALK]
        for _i, proc_obj in enumerate(walk_list):
            # Periodically release the GIL so the main heartbeat thread is
            # scheduled even while this (potentially slow, disk-bound) walk runs.
            if _i and _i % _GIL_YIELD_EVERY == 0:
                time.sleep(_GIL_YIELD_SECS)
            try:
                t = proc_obj.cpu_times()
                cpu_time += t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
            try:
                io = proc_obj.io_counters()
                io_bytes += io.read_bytes + io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError,
                    NotImplementedError):
                pass
            # Skip the parent (index 0): its RSS is already in tree_mem seed.
            if _i:
                try:
                    tree_mem += proc_obj.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    pass
        
        return {
            'alive': p.is_running(),
            'zombie': is_zombie,
            'cpu_percent': cpu,
            # memory_mb reports the TREE footprint (parent + walked children),
            # not the parent-only RSS. This is what shell.py surfaces to the
            # LLM judge, so a launcher wrapper no longer looks like "2 MB".
            'memory_mb': tree_mem,
            'memory_mb_parent': mem,
            'num_children': num_children,
            'children_alive': alive_children,
            'cpu_time': cpu_time,
            'io_bytes': io_bytes,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            'alive': False,
            'zombie': False,
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'memory_mb_parent': 0.0,
            'num_children': 0,
            'children_alive': 0,
            'cpu_time': 0.0,
            'io_bytes': 0.0,
        }


def _cgroup_mem_pressure(threshold_ratio: float) -> "bool | None":
    """
    Return True/False from the CONTAINER's cgroup memory accounting, or None
    when there is no enforced cgroup limit (bare metal / unlimited).

    In a container psutil.virtual_memory() reports the HOST (e.g. 909GB total)
    even when the cgroup caps the process at 2GB. A build that OOM-kills inside
    the 2GB slice leaves host `available` almost untouched, so the host-based
    check never fires. Reading the cgroup limit+usage is the only way to see
    the real pressure the process actually experiences.
    """
    # cgroup v2: unified files
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            raw = f.read().strip()
        if raw != 'max':
            limit = int(raw)
            with open('/sys/fs/cgroup/memory.current') as f:
                usage = int(f.read().strip())
            if limit > 0:
                return (limit - usage) < limit * threshold_ratio
    except (OSError, ValueError):
        pass
    # cgroup v1: per-controller files
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
            limit = int(f.read().strip())
        # v1 "unlimited" is a huge sentinel (~ PAGE_COUNTER_MAX); treat as no cap
        if 0 < limit < (1 << 62):
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes') as f:
                usage = int(f.read().strip())
            return (limit - usage) < limit * threshold_ratio
    except (OSError, ValueError):
        pass
    return None


def system_mem_pressure(threshold_ratio: float = 0.10) -> bool:
    """
    Return True if the process is under real memory pressure — available memory
    has dropped below `threshold_ratio` of total. This is the ONLY evidence
    (besides an explicit "Killed" in output) that turns a child-count drop into
    an OOM verdict. Without it, a child drop is treated as normal convergence.

    Prefers the CONTAINER's cgroup limit over the host total: in a container
    psutil sees the host RAM (e.g. 909GB) and would never report pressure even
    while the process OOM-thrashes inside a 2GB cgroup slice. Falls back to the
    host figure only when no cgroup limit is enforced.
    """
    cg = _cgroup_mem_pressure(threshold_ratio)
    if cg is not None:
        return cg
    try:
        vm = psutil.virtual_memory()
        return vm.available < vm.total * threshold_ratio
    except Exception:
        return False


def detect_output_anomalies(recent_output: str) -> dict:
    """
    Detect anomalies in command output.
    
    Returns:
        {
            'oom_killed': bool,
            'killed_count': int,
            'repeated_errors': bool,
        }
    """
    lines = recent_output.split('\n')
    
    # Count "Killed" messages
    killed_count = sum(1 for line in lines if 'Killed' in line)
    
    # OOM indicators. Note the compiler-driver signatures: when the kernel OOM
    # killer reaps a compiler subprocess (cc1plus/cc1/lto1), gcc/g++ reports it
    # as "Killed signal terminated program cc1plus" (or "... program cc1") and
    # make follows with Error 137 / signal 9. Matching only "Out of memory" or
    # "Error 137" misses the first, clearest line and lets an OOM build look
    # like a normal in-progress compile.
    oom_indicators = [
        'Error 137',
        'Out of memory',
        'MemoryError',
        'cannot allocate memory',
        'OOM killer',
        'Killed signal terminated program',   # gcc/g++ driver: OOM-killed cc1plus
        'signal terminated program cc1',       # cc1 / cc1plus variants
        'internal compiler error: Killed',     # ICE wording when cc1plus is reaped
    ]
    oom_killed = any(indicator in recent_output for indicator in oom_indicators)
    
    # Repeated errors (same error 3+ times)
    error_lines = [l for l in lines if 'error' in l.lower() or 'Error' in l]
    repeated_errors = len(error_lines) >= 3 and len(set(error_lines)) <= 2
    
    return {
        'oom_killed': oom_killed,
        'killed_count': killed_count,
        'repeated_errors': repeated_errors,
    }


def should_kill_process(
    elapsed_seconds: float,
    output_changed: bool,
    stall_count: int,
    proc_health: dict,
    output_anomalies: dict,
    had_children: bool,
    prev_num_children: int = 0,
    progress_signals: dict | None = None,
    mem_pressure: bool | None = None,
    idle_count: int = 0,
) -> tuple[bool, str]:
    """
    Unified hard-indicator check. Returns (should_kill, reason).

    Redesigned around three principles (see health_monitor_improvement.md):
    a KILL must be driven by MULTIPLE independent signals converging on death;
    any single signal — especially a child-count drop — must NOT convict alone;
    and POSITIVE liveness evidence (the process is burning CPU / IO / memory)
    vetoes every heuristic kill.

    progress_signals: {'cpu_time_delta', 'io_bytes_delta', 'rss_delta'} computed
        by shell.py across two sampling points. When None (legacy callers), the
        liveness veto is skipped and only deterministic-death rules apply.
    mem_pressure: real system memory pressure (available < ~10% total). When
        None it is probed live via system_mem_pressure().
    """
    if mem_pressure is None:
        mem_pressure = system_mem_pressure()

    # --- Liveness veto (support A): is the process actually working? ---
    making_progress = False
    if progress_signals is not None:
        making_progress = (
            progress_signals.get('cpu_time_delta', 0.0) > 0.5      # cumulative CPU secs rising
            or progress_signals.get('io_bytes_delta', 0.0) > (1 << 20)   # >1MB IO
            or abs(progress_signals.get('rss_delta', 0.0)) > (1 << 20)   # RSS shifting
        )

    # Frozen-output advisory — fires EVEN WHEN the process is making progress.
    # A child subprocess (e.g. stockfish) burning CPU in the background does
    # NOT mean the parent is healthy: if stdout has been byte-for-byte
    # identical for stall_count consecutive 30s checks, the parent likely
    # caught an error (e.g. KeyError printed a traceback) and is now
    # blocking forever on the child.  We don't hard-kill yet — we return an
    # advisory so the LLM judge (which has command history and richer
    # context) can make the final call.
    # Capped at stall_count < 12: after 6 minutes the hard kill below fires.
    if not output_changed and 6 <= stall_count < 12 and elapsed_seconds > 120:
        return False, (
            f"Process output has been frozen (identical) for {stall_count * 30}s "
            f"despite {proc_health['cpu_percent']:.0f}% CPU activity. "
            f"The command may have caught an error and is blocking on a subprocess. "
            f"Consider killing if this pattern persists."
        )

    # === Deterministic death (fires regardless of liveness) ===

    # 1. Zombie process
    if proc_health['zombie']:
        return True, (
            "Process became zombie (all children died, parent stuck). "
            "This usually means child processes crashed but parent didn't detect it."
        )

    # 2a. OOM HARD evidence: output shows Killed x2 / Error 137. This is real,
    #     independent of liveness.
    if output_anomalies['oom_killed'] and output_anomalies['killed_count'] >= 2:
        return True, (
            f"OOM killer terminated processes {output_anomalies['killed_count']} times. "
            "Reduce parallelism (e.g., make -j1 instead of make -j) or increase memory."
        )

    # 1b. FROZEN OUTPUT — hard kill. The process is alive (poll() is None),
    #     children may be alive and burning CPU, but the output has been
    #     byte-for-byte identical for stall_count consecutive 30s checks.
    #     This means the command caught an error and is blocking forever
    #     (e.g. a try/except that swallows the exception then waits on a
    #     subprocess that never responds). A child burning CPU in the
    #     background (e.g. stockfish thinking) is NOT real progress when the
    #     parent process is stuck in a deadlock — so this fires REGARDLESS of
    #     the liveness veto.
    #     Threshold: 12 consecutive stalls (≈6 minutes at 30s interval).
    #     The advisory at line 206 fires first at stall_count >= 6 (≈3 min),
    #     giving the LLM judge a chance to kill. This hard kill is the safety
    #     net for when the judge doesn't act (no judge_fn, API timeout, or
    #     judge returns kill=False).
    if not output_changed and stall_count >= 12 and elapsed_seconds > 300:
        return True, (
            f"Process output has been frozen (identical) for {stall_count * 30}s "
            f"with {proc_health['cpu_percent']:.0f}% CPU. "
            f"The command likely caught an error and is blocking on a subprocess. "
            f"Stop and fix the root cause instead of waiting."
        )

    # 1c. IDLE HANG — hard kill, fires regardless of the liveness veto because
    #     idle_count is ITSELF the negation of liveness: shell.py only bumps it
    #     when output is unchanged AND all three progress deltas (cpu_time, io,
    #     rss) are flat. IDLE_KILL_THRESHOLD consecutive such samples (~2 min at
    #     the 30s cadence) means the process produced no output and did no
    #     measurable CPU/IO/memory work for two straight minutes — it is blocked
    #     on a syscall (dead network connect, DNS black hole, unobtainable lock),
    #     not computing. A real compile / import / buffered job moves at least
    #     one counter every sample and never reaches this count. This kills a
    #     hung curl/wget/apt/git in ~2 min instead of waiting out the 6-min
    #     frozen-output rule (1b), which never fires when there is simply no
    #     output at all.
    if idle_count >= IDLE_KILL_THRESHOLD:
        return True, (
            f"No output and no CPU/IO/memory activity for ~{idle_count * 30}s "
            f"(idle for {idle_count} consecutive checks). The process is not "
            f"computing — it is blocked on a syscall (a dead network endpoint, "
            f"DNS black hole, or a lock it will never acquire). Retrying the "
            f"same command will hang the same way. Change your approach: unset "
            f"the proxy, try a different mirror/endpoint, add an explicit "
            f"--max-time/timeout, or use a local cache."
        )

    # --- Liveness veto: below here every rule is a heuristic; if the process
    #     is demonstrably working, never kill. ---
    if making_progress:
        return False, ""

    # === Heuristics (only reached when NOT making progress) ===

    # 2b. Child-count drop — DOUBLE-evidence gate (support B). Relative drop
    #     (>= half, min 3) AND no live children AND real OOM evidence
    #     (system memory pressure OR at least one Killed in output). A drop
    #     alone with killed_count==0 and no memory pressure is treated as
    #     normal worker convergence, NOT OOM.
    child_drop = max(0, prev_num_children - proc_health['num_children'])
    drop_threshold = max(3, prev_num_children // 2)
    if (child_drop >= drop_threshold
            and proc_health['children_alive'] == 0
            and (mem_pressure or output_anomalies['killed_count'] >= 1)):
        return True, (
            f"OOM likely: children dropped {prev_num_children}→{proc_health['num_children']} "
            f"AND no live children AND memory pressure/Killed evidence. "
            "Reduce parallelism or increase memory."
        )

    # 3. All children dead + no output + >2min (already past liveness veto)
    if (had_children and
        proc_health['children_alive'] == 0 and
        not output_changed and
        elapsed_seconds > 120):
        return True, (
            f"All {proc_health['num_children']} child processes exited "
            f"but parent still running with no output for {int(elapsed_seconds)}s. "
            "Build likely failed silently."
        )

    # 4. SILENT + SUSTAINED STALL — now ADVISORY ONLY (handled by LLM judge).
    #    The old rule killed processes with 0% CPU + no output after 3 minutes.
    #    But this killed healthy imports (torch, large libraries) that legitimately
    #    run silent for minutes. The LLM judge with command history + container
    #    resources + output pattern has far better context to make this call.
    #    We keep the signal but downgrade to advisory — return (False, advisory_msg)
    #    so the LLM judge sees it as context, not a hard kill.
    if (proc_health['cpu_percent'] < 0.5 and
        not output_changed and
        stall_count >= 6 and
        elapsed_seconds > 180):
        return False, (
            f"Process using 0% CPU with no output for {stall_count * 30}s. "
            "This MAY be stuck, but could also be a legitimate silent operation "
            "(library import, buffered computation). Deferring to LLM judge "
            "for contextual assessment."
        )

    # 5. Graduated timeout (support C). Hard cap 60min with no activity → kill.
    #    Soft cap 20min → do NOT kill in the hard layer; hand to the LLM judge
    #    with elapsed context (heavy compiles/training legitimately run long).
    if elapsed_seconds > 3600:
        return True, (
            f"Command exceeded 60-minute hard timeout (ran {int(elapsed_seconds)}s) "
            "with no detectable activity. Almost certainly stuck."
        )
    # 20min < elapsed < 60min with no progress: defer to judge, don't hard-kill.

    return False, ""
