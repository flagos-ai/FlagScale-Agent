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

"""FlagScale Training Monitor — specialized for FlagScale + torchrun distributed training.

Two modes:
  - check: one-shot inspection of current log state (replaces find_latest_log)
  - watch: continuous polling until event/timeout (replaces old monitor)

Designed for multi-node container environments with shared storage.
"""

import glob
import json
import math
import os
import re
import subprocess
import time

from flagscale_agent.react.tools.base import Tool


# ─── Metrics parsing ───────────────────────────────────────────────────────────

_METRIC_RE = re.compile(
    r'iteration\s+\d+|step[=:\s]\d+|loss[=:\s]|grad.norm|throughput|MFU',
    re.IGNORECASE,
)

_COMMON_VOCAB_SIZES = [32000, 50257, 65536, 100000, 128256, 151936, 256000]


def _parse_megatron_metrics(text: str) -> dict:
    """Extract training metrics from Megatron log output."""
    metrics = {"iterations": [], "last_iter": None, "last_loss": {}, "anomalies": []}
    for line in text.splitlines():
        m = re.search(r'iteration\s+(\d+)', line, re.IGNORECASE)
        if not m:
            continue
        iteration = int(m.group(1))
        metrics["iterations"].append(iteration)
        metrics["last_iter"] = iteration
        for pattern, field in [
            (r'lm loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'lm_loss'),
            (r'ce[_ ]?loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'ce_loss'),
            (r'loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'loss'),
            (r'grad[ _]norm[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'grad_norm'),
            (r'num[_ ]zeros[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'num_zeros'),
        ]:
            fm = re.search(pattern, line, re.IGNORECASE)
            if fm:
                try:
                    metrics["last_loss"][field] = float(fm.group(1))
                except ValueError:
                    pass
    return metrics


def _health_check(metrics: dict, vocab_size: int = 0) -> list:
    """Run training health checks on parsed metrics."""
    warnings = []
    loss_val = (
        metrics["last_loss"].get("ce_loss")
        or metrics["last_loss"].get("lm_loss")
        or metrics["last_loss"].get("loss")
    )
    if loss_val is not None and vocab_size > 0:
        random_loss = math.log(vocab_size)
        if loss_val > random_loss * 0.8:
            warnings.append(
                f"WARNING: loss={loss_val:.4f} ~ ln({vocab_size})={random_loss:.2f} "
                f"-> model may be outputting random."
            )
    elif loss_val is not None and vocab_size == 0:
        best_v, best_diff = None, float("inf")
        for v in _COMMON_VOCAB_SIZES:
            diff = abs(loss_val - math.log(v))
            if diff < best_diff:
                best_v, best_diff = v, diff
        if best_v and best_diff / math.log(best_v) < 0.10:
            warnings.append(
                f"WARNING: loss={loss_val:.4f} ~ ln({best_v})={math.log(best_v):.2f} "
                f"-> model may be outputting random."
            )
    grad_norm = metrics["last_loss"].get("grad_norm")
    if grad_norm is not None and grad_norm == 0:
        warnings.append("WARNING: grad_norm=0 -> gradients not flowing.")
    num_zeros = metrics["last_loss"].get("num_zeros")
    if num_zeros is not None and num_zeros > 1e9:
        warnings.append(f"WARNING: num_zeros={num_zeros:.2e} -> most gradients are zero.")
    return warnings


# ─── Filesystem utilities ──────────────────────────────────────────────────────

def _last_sorted_subdir(parent: str, key=None):
    """Return the last subdirectory under parent when sorted by key."""
    if not os.path.isdir(parent):
        return ""
    entries = [e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e))]
    if not entries:
        return ""
    entries.sort(key=key)
    return os.path.join(parent, entries[-1])


def _numeric_key(name: str):
    """Extract trailing number for sorting: 'attempt_2' -> 2."""
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 0


def _tail(path: str, n: int = 50) -> str:
    """Return the last n lines of a file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n:]) or "(empty)"
    except (FileNotFoundError, OSError):
        return "(empty)"


# ─── Harmless warning filter ──────────────────────────────────────────────────

_HARMLESS_PATTERNS = [
    re.compile(r"DeprecationWarning", re.I),
    re.compile(r"FutureWarning", re.I),
    re.compile(r"UserWarning", re.I),
    re.compile(r"PendingDeprecationWarning", re.I),
    re.compile(r"torch\.cuda\.amp.*deprecated", re.I),
    re.compile(r"Setting\s+.*\s+threads", re.I),
    re.compile(r"OMP_NUM_THREADS", re.I),
    re.compile(r"wandb.*version.*available", re.I),
    re.compile(r"NOTE:\s+Redirects are currently not supported", re.I),
    re.compile(r"warnings\.warn\(", re.I),
    re.compile(r"^\s*$"),
]


def _is_harmless(line: str) -> bool:
    for pat in _HARMLESS_PATTERNS:
        if pat.search(line):
            return True
    return False


# ─── Main Tool Class ──────────────────────────────────────────────────────────

class FlagScaleTrainMonitorTool(Tool):
    name = "flagscale_train_monitor"
    description = (
        "Monitor FlagScale distributed training launched via torchrun. "
        "Two modes: 'check' (one-shot log inspection) and 'watch' (continuous polling). "
        "Automatically discovers logs from output_dir, scans all ranks for errors, "
        "finds the loss rank (last pipeline stage), and detects training phases. "
        "Designed for multi-node container environments with shared storage."
    )
    parameters = {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": "FlagScale experiment output directory. Required.",
            },
            "mode": {
                "type": "string",
                "enum": ["check", "watch"],
                "description": (
                    "check: one-shot inspection of current log state. "
                    "watch: continuous polling until event/timeout. Default: watch."
                ),
            },
            "duration": {
                "type": "integer",
                "description": "Max watch duration in seconds. Default: 300. Max: 1800.",
            },
            "interval": {
                "type": "integer",
                "description": "Polling interval in seconds. Default: 30.",
            },
            "target_step": {
                "type": "integer",
                "description": "Stop when training reaches this iteration/step number.",
            },
            "filter": {
                "type": "string",
                "enum": ["all", "errors", "progress"],
                "description": "Filter for check mode output. Default: all.",
            },
            "vocab_size": {
                "type": "integer",
                "description": "Model vocab size for health check (e.g. 151936).",
            },
            "lines": {
                "type": "integer",
                "description": "Tail lines per log file (check mode). Default: 50.",
            },
        },
        "required": ["output_dir"],
    }

    def __init__(self, classify_fn=None):
        self._classify_fn = classify_fn

    def execute(self, **kwargs) -> str:
        output_dir = kwargs.pop("output_dir")
        mode = kwargs.pop("mode", "watch")

        if not os.path.isdir(output_dir):
            return f"ERROR: output_dir does not exist: {output_dir}"

        if mode == "check":
            return self._check_mode(output_dir, **kwargs)
        else:
            return self._watch_mode(output_dir, **kwargs)

    # ─── Check Mode (one-shot, replaces find_latest_log) ──────────────────────

    def _check_mode(self, output_dir, **kwargs):
        """One-shot inspection of current training log state."""
        lines_count = kwargs.get("lines", 50)
        vocab_size = kwargs.get("vocab_size", 0)
        filter_mode = kwargs.get("filter", "all")

        logs = self._discover_logs(output_dir)
        if logs.get("error"):
            return logs["error"]

        rank_dirs = logs.get("rank_dirs", [])
        if not rank_dirs:
            return f"ERROR: No rank directories found in {output_dir}"

        # Find loss rank (last pipeline stage prints metrics)
        loss_rank, loss_content, loss_metrics = self._find_loss_rank(rank_dirs, lines_count)
        error_ranks = self._find_error_ranks(rank_dirs)
        shared = len(logs.get("host_dirs", [])) > 1

        parts = [
            "=== FlagScale Training Status ===",
            f"Output dir: {output_dir}",
            f"Storage: {'shared' if shared else 'local'} ({len(logs.get('host_dirs', []))} host(s))",
            f"Phase: {self._detect_phase_from_logs(logs, loss_metrics)}",
            f"Ranks: 0-{len(rank_dirs)-1}",
        ]

        # Loss rank output
        if loss_rank is not None:
            rank_num = os.path.basename(loss_rank)
            parts.append(f"\n=== Loss Rank (rank {rank_num}, last pipeline stage) ===")
            parts.append(f"Path: {os.path.join(loss_rank, 'stdout.log')}")
            filtered = self._apply_filter(loss_content, filter_mode)
            parts.append(filtered)
            if loss_metrics["last_iter"] is not None:
                summary_parts = [f"Latest iteration: {loss_metrics['last_iter']}"]
                for k, v in loss_metrics["last_loss"].items():
                    summary_parts.append(f"{k}: {v}")
                parts.append("\n--- Metrics ---")
                parts.append(", ".join(summary_parts))
            health_warnings = _health_check(loss_metrics, vocab_size)
            if health_warnings:
                parts.append("\n--- Health Warnings ---")
                parts.extend(health_warnings)
            else:
                parts.append("\n--- Health Check ---")
                parts.append("✅ No anomalies detected")
        else:
            parts.append("\n=== No rank with training metrics found ===")
            if rank_dirs:
                last_rank = rank_dirs[-1]
                stdout_path = os.path.join(last_rank, "stdout.log")
                if os.path.isfile(stdout_path):
                    parts.append(f"Fallback (last rank {os.path.basename(last_rank)}):")
                    parts.append(_tail(stdout_path, lines_count))

        # Error ranks
        if error_ranks:
            parts.append(f"\n=== Errors ({len(error_ranks)} rank(s)) ===")
            for rank_dir, stderr_content in error_ranks[:5]:
                rank_num = os.path.basename(rank_dir)
                parts.append(f"\n--- rank {rank_num} stderr ---")
                parts.append(stderr_content)
        else:
            parts.append("\n=== No stderr errors ===")

        # Structured summary
        summary = {
            "loss_rank": int(os.path.basename(loss_rank)) if loss_rank else None,
            "last_iteration": loss_metrics["last_iter"] if loss_metrics else None,
            "last_loss": loss_metrics.get("last_loss", {}) if loss_metrics else {},
            "error_ranks": [int(os.path.basename(rd)) for rd, _ in error_ranks],
            "training_started": loss_metrics.get("last_iter") is not None if loss_metrics else False,
            "health_ok": not bool(_health_check(loss_metrics, vocab_size)) if loss_metrics else False,
        }
        parts.append(f"\n=== JSON ===\n{json.dumps(summary, indent=2)}")

        return "\n".join(parts)

    # ─── Watch Mode (continuous polling) ──────────────────────────────────────

    def _watch_mode(self, output_dir, **kwargs):
        """Continuous polling until event/timeout."""
        duration = min(kwargs.get("duration", 300), 1800)
        interval = max(kwargs.get("interval", 30), 5)
        target_step = kwargs.get("target_step")
        vocab_size = kwargs.get("vocab_size", 0)

        start = time.time()
        poll_count = 0
        events = []
        phase = "startup"
        last_log_growth = time.time()
        last_step_time = None
        last_step_number = 0
        last_stdout_size = 0
        last_stderr_total = 0
        stderr_checked = {}
        last_rediscover = time.time()
        ever_uncertain = False
        hang_timeout = 600  # 10 min without step advance = hang

        # Wait for logs to appear
        logs = None
        for _wait in range(6):
            logs = self._discover_logs(output_dir)
            if not logs.get("error"):
                break
            time.sleep(5)
        if logs and logs.get("error"):
            return logs["error"]

        stdout_log = logs.get("stdout_log", "")
        stderr_logs = logs.get("stderr_logs", [])

        # Immediate error check
        for sp in stderr_logs:
            try:
                if os.path.getsize(sp) > 0:
                    with open(sp, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(16384)
                    error_lines = self._filter_real_errors(content.splitlines())
                    if error_lines:
                        rank = self._rank_from_path(sp)
                        return self._format_watch_result(
                            "stderr_error", 0, 0, events,
                            [f"[rank {rank} ERROR at start]: {error_lines[0][:120]}"] + error_lines[:10]
                        )
            except OSError:
                pass

        # Capture initial stderr sizes
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

            poll_count += 1

            # Re-discover logs (handles delayed creation + stdout lock-in).
            # Periodic: every 60s re-run discovery so that a stdout selector
            # stuck on a non-growing file can switch to the real metric rank.
            now_ts = time.time()
            if not stdout_log or (now_ts - last_rediscover >= 60):
                logs = self._discover_logs(output_dir)
                last_rediscover = now_ts
                if not logs.get("error"):
                    new_stdout = logs.get("stdout_log", "")
                    stderr_logs = logs.get("stderr_logs", [])
                    if new_stdout != stdout_log:
                        stdout_log = new_stdout
                        last_stdout_size = 0
                        events.append(
                            f"[stdout switched to {stdout_log} at {int(elapsed)}s]"
                        )

            # Read stdout
            current_size = 0
            new_lines = []
            if stdout_log and os.path.isfile(stdout_log):
                try:
                    current_size = os.path.getsize(stdout_log)
                except OSError:
                    current_size = 0
                if current_size > last_stdout_size:
                    last_log_growth = time.time()
                    new_content = self._read_tail_from(stdout_log, last_stdout_size, 8192)
                    new_lines = new_content.splitlines()
                    last_stdout_size = current_size

            # Stderr growth counts as liveness too: in remote/shared-storage
            # setups the metric rank's stdout may stall while 64 stderr logs
            # keep growing — stdout-only growth misses that heartbeat.
            stderr_total = 0
            for sp in stderr_logs:
                try:
                    stderr_total += os.path.getsize(sp)
                except OSError:
                    pass
            if stderr_total > last_stderr_total:
                last_log_growth = time.time()
                last_stderr_total = stderr_total

            # Phase detection
            if not stdout_log or current_size == 0:
                phase = "rendezvous" if logs and not logs.get("error") else "startup"
            elif new_lines and any(_METRIC_RE.search(l) for l in new_lines):
                phase = "training"
            elif phase != "training":
                phase = "init"

            # Check for training metrics
            if phase == "training" and new_lines:
                metric_lines = [l for l in new_lines if _METRIC_RE.search(l)]
                if metric_lines:
                    events.append(f"[+{len(metric_lines)} metrics at {int(elapsed)}s]")
                # Update step tracking
                for line in new_lines:
                    m = re.search(r'iteration\s+(\d+)', line, re.IGNORECASE)
                    if m:
                        step = int(m.group(1))
                        if step > last_step_number:
                            last_step_number = step
                            last_step_time = time.time()
                # Check target step
                if target_step and last_step_number >= target_step:
                    events.append(f"[target step {target_step} reached at {int(elapsed)}s]")
                    tail = self._read_tail(stdout_log, 20)
                    return self._format_watch_result(
                        "target_reached", poll_count, elapsed, events, tail
                    )

            # Stderr scan
            stderr_error = self._scan_stderr(stderr_logs, stderr_checked, elapsed)
            if stderr_error:
                events.append(stderr_error["event"])
                return self._format_watch_result(
                    "stderr_error", poll_count, elapsed, events, stderr_error["lines"]
                )

            # Liveness check (multi-signal, three-state)
            alive = self._is_alive(phase, last_log_growth)
            if alive == "uncertain":
                if not ever_uncertain:
                    ever_uncertain = True
                    events.append(
                        "[UNCERTAIN: no local liveness signal (remote/shared-storage run) "
                        "and no log growth — NOT declaring DEAD]"
                    )
            elif alive is False and phase not in ("startup",):
                # Final stderr scan
                stderr_error = self._scan_stderr(stderr_logs, stderr_checked, elapsed)
                if stderr_error:
                    events.append(stderr_error["event"])
                    return self._format_watch_result(
                        "stderr_error", poll_count, elapsed, events, stderr_error["lines"]
                    )
                events.append(f"[process DEAD at {int(elapsed)}s]")
                events.append(
                    "[if this is a remote/shared-storage run, verify directly: "
                    "(1) grep 'elapsed time per iteration' <rank>/stdout.log "
                    "(2) stat -c %Y <rank>/stdout.log "
                    "(3) nvidia-smi --query-compute-apps=pid --format=csv,noheader]"
                )
                tail = self._read_tail(stdout_log, 20) if stdout_log else ["(no stdout)"]
                return self._format_watch_result(
                    "process_dead", poll_count, elapsed, events, tail
                )

            # Hang detection (training phase only)
            if phase == "training" and last_step_time:
                stall = time.time() - last_step_time
                if stall > hang_timeout:
                    events.append(f"[HANG: no step advance for {int(stall)}s, last step={last_step_number}]")
                    tail = self._read_tail(stdout_log, 20) if stdout_log else []
                    return self._format_watch_result(
                        "hang_detected", poll_count, elapsed, events, tail
                    )

            time.sleep(interval)

        # Timeout — return final state
        if ever_uncertain and not any("metrics" in e for e in events):
            events.append(
                "[UNCERTAIN: watch ended with no confirmed liveness signal — "
                "if this is a remote/shared-storage run, verify directly on the training node]"
            )
        tail = self._read_tail(stdout_log, 20) if stdout_log else ["(no output)"]
        metrics = _parse_megatron_metrics("\n".join(tail))
        health = _health_check(metrics, vocab_size)
        if health:
            events.extend(health)
        return self._format_watch_result("timeout", poll_count, time.time() - start, events, tail)

    # ─── Log Discovery ────────────────────────────────────────────────────────

    def _discover_logs(self, output_dir):
        """Discover FlagScale log files from output_dir.

        FlagScale structure:
          output_dir/logs/details/host_N_<hostname>/<timestamp>/<run>/attempt_N/<rank>/
            - stdout.log
            - stderr.log

        Returns dict with: stdout_log, stderr_logs, rank_dirs, host_dirs, error
        """
        result = {"stdout_log": "", "stderr_logs": [], "rank_dirs": [], "host_dirs": [], "error": ""}
        logs_dir = os.path.join(output_dir, "logs", "details")
        if not os.path.isdir(logs_dir):
            result["error"] = f"ERROR: No logs directory at {logs_dir}. Training may not have started."
            return result

        host_dirs = sorted(glob.glob(os.path.join(logs_dir, "host_*")))
        if not host_dirs:
            result["error"] = f"ERROR: No host directories in {logs_dir}."
            return result
        result["host_dirs"] = host_dirs

        # Collect rank dirs from ALL hosts (multi-node shared storage)
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
            # List rank subdirs
            for entry in os.listdir(attempt_dir):
                rank_path = os.path.join(attempt_dir, entry)
                if os.path.isdir(rank_path) and entry.isdigit():
                    all_rank_dirs.append(rank_path)
                    stderr_path = os.path.join(rank_path, "stderr.log")
                    if os.path.isfile(stderr_path):
                        stderr_logs.append(stderr_path)

        all_rank_dirs.sort(key=lambda p: int(os.path.basename(p)))
        result["rank_dirs"] = all_rank_dirs
        result["stderr_logs"] = stderr_logs

        # Find stdout with training metrics (last pipeline rank)
        stdout_log = ""
        for rank_dir in reversed(all_rank_dirs):
            candidate = os.path.join(rank_dir, "stdout.log")
            if not os.path.isfile(candidate):
                continue
            try:
                size = os.path.getsize(candidate)
                if size == 0:
                    continue
                with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(max(0, size - 4096))
                    tail = fh.read()
                if _METRIC_RE.search(tail):
                    stdout_log = candidate
                    break
            except OSError:
                continue

        # Fallback: first non-empty stdout
        if not stdout_log:
            for rank_dir in all_rank_dirs:
                candidate = os.path.join(rank_dir, "stdout.log")
                try:
                    if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                        stdout_log = candidate
                        break
                except OSError:
                    continue

        result["stdout_log"] = stdout_log
        return result

    # ─── Loss Rank & Error Detection ─────────────────────────────────────────

    def _find_loss_rank(self, rank_dirs, lines_count=50):
        """Find the rank printing training metrics (last pipeline stage)."""
        for rank_dir in reversed(rank_dirs):
            stdout_path = os.path.join(rank_dir, "stdout.log")
            if not os.path.isfile(stdout_path):
                continue
            content = _tail(stdout_path, lines_count)
            if re.search(r'iteration\s+\d+', content, re.IGNORECASE):
                metrics = _parse_megatron_metrics(content)
                return rank_dir, content, metrics
        return None, "", {"iterations": [], "last_iter": None, "last_loss": {}, "anomalies": []}

    def _find_error_ranks(self, rank_dirs):
        """Find ranks with stderr errors."""
        error_ranks = []
        for rank_dir in rank_dirs:
            stderr_path = os.path.join(rank_dir, "stderr.log")
            if not os.path.isfile(stderr_path):
                continue
            try:
                size = os.path.getsize(stderr_path)
            except OSError:
                continue
            if size == 0:
                continue
            content = _tail(stderr_path, 10)
            if any(kw in content.lower() for kw in ["error", "exception", "traceback", "fault", "killed", "oom"]):
                error_ranks.append((rank_dir, content))
        return error_ranks

    # ─── Liveness Detection ───────────────────────────────────────────────────

    def _is_alive(self, phase, last_log_growth_time):
        """Multi-signal liveness check for container environments.

        Priority: log growth > GPU process > pgrep.
        Grace periods vary by training phase.

        Returns tri-state: True (alive), False (all signals absent AND
        at least one local signal available to prove absence), or
        "uncertain" (all signals absent AND no local signal could be
        probed — e.g. remote/shared-storage runs where nvidia-smi and
        pgrep see only this agent's container). Callers must treat
        "uncertain" as NOT proven dead.
        """
        now = time.time()
        since_growth = now - last_log_growth_time

        # Signal 1: log file recently grew (most reliable, no container issues)
        grace = {"startup": 60, "rendezvous": 120, "init": 300, "training": 120}
        if since_growth < self.LIVENESS_GRACE_SECONDS.get(phase, 120):
            return True

        # Signal 2: GPU has compute processes (works if nvidia-smi accessible)
        gpu_signal = self._gpu_has_compute_process()

        # Signal 3: pgrep (least reliable in containers, but useful as backup)
        pgrep_signal = self._pgrep_alive()

        if gpu_signal or pgrep_signal:
            return True

        # Signals 2/3 found nothing. Only conclude death when this host
        # could actually host the job — i.e. the watched output lives on
        # LOCAL (non-network) storage. On shared/network storage the
        # training usually runs on OTHER hosts, so local probes prove
        # nothing (E62/E29 false-DEAD class).
        if not self._is_network_storage():
            return False

        return "uncertain"

    # Known-LOCAL filesystem types (stat -f -c %T strings). Only when the
    # watched path PROVABLY lives on local storage can negative GPU/pgrep
    # probes prove the training process dead. Anything else — nfs, lustre,
    # or an UNRECOGNIZED type (many distributed FSes report
    # "UNKNOWN (0x...)", e.g. 0x797083 on shared NVMe mounts) — is treated
    # as network/shared storage where local probes prove nothing.
    _LOCAL_FS_TYPES = {
        "ext2/ext3", "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay",
        "ramfs", "f2fs", "jfs", "vfat", "exfat", "ntfs",
    }

    # Grace period (seconds without log growth) per phase before falling
    # back to process probes. Class attribute so callers/tests can tighten
    # it for fast end-to-end runs instead of sleeping through real waits.
    LIVENESS_GRACE_SECONDS = {"startup": 60, "rendezvous": 120, "init": 300, "training": 120}

    def _is_network_storage(self, path="."):
        """True unless path is PROVEN to live on local (non-network) storage.

        On shared/remote storage, stdout/stderr are written by training
        hosts other than (possibly) the agent host, so GPU/pgrep probes on
        the agent host cannot prove liveness either way. Unknown fs types
        are treated as network storage: declaring a process dead requires
        positive evidence, and an unclassifiable filesystem provides none
        (a false "network" verdict only costs an UNCERTAIN report with
        direct-check hints — real crashes still surface via stderr scan).
        """
        try:
            result = subprocess.run(
                ["stat", "-f", "-c", "%T", path],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().lower() not in self._LOCAL_FS_TYPES
        except Exception:
            pass
        # Could not determine → cannot prove local → conservative.
        return True

    def _signals_available(self):
        """True if at least one local liveness probe can actually run here.

        In remote-SSH / shared-storage workflows the agent often runs on a
        different host than the training job: nvidia-smi finds no GPUs and
        pgrep finds no training processes — but that proves nothing about
        the remote run. Returns False in that situation.
        """
        try:
            result = subprocess.run(
                "nvidia-smi -L", shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except Exception:
            pass
        # No local GPUs: if we are also running where torchrun would run, the
        # pgrep probe is meaningful; if even pgrep itself fails (missing
        # binary, restricted procfs), nothing local can prove liveness.
        try:
            result = subprocess.run(
                ["pgrep", "-f", "torchrun|python.*train"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode in (0, 1):
                return True
        except Exception:
            pass
        return False

    def _gpu_has_compute_process(self):
        """Check if any GPU has compute processes via nvidia-smi."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                shell=True, capture_output=True, text=True, timeout=5
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _pgrep_alive(self):
        """Check if torchrun/training process is alive via pgrep."""
        pattern = r'torchrun|python.*train'
        my_pid = os.getpid()
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False
            pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip()]
            return any(p != my_pid for p in pids)
        except Exception:
            return False

    # ─── Stderr Scanning ─────────────────────────────────────────────────────

    def _scan_stderr(self, stderr_logs, checked_sizes, elapsed):
        """Scan stderr logs for new errors since last check."""
        all_errors = {}

        for log_path in stderr_logs:
            try:
                size = os.path.getsize(log_path)
            except OSError:
                continue
            prev_size = checked_sizes.get(log_path, 0)
            if size <= prev_size:
                continue
            # Read new content
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(prev_size)
                    new_content = f.read(16384)
            except OSError:
                continue
            checked_sizes[log_path] = size

            error_lines = self._filter_real_errors(new_content.splitlines())
            if error_lines:
                rank = self._rank_from_path(log_path)
                all_errors[rank] = error_lines

        if not all_errors:
            return None

        # Format multi-rank error report
        ranks = sorted(all_errors.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        report = [f"[STDERR ERROR at {int(elapsed)}s — {len(ranks)} rank(s): {', '.join(ranks[:5])}]"]
        detail = []
        for rank in ranks[:5]:
            detail.extend([f"  rank {rank}: {line}" for line in all_errors[rank][:3]])

        return {"event": report[0], "lines": report + detail}

    def _filter_real_errors(self, lines):
        """Filter lines to only real errors (not harmless warnings)."""
        filtered = [l for l in lines if l.strip() and not _is_harmless(l)]
        if not filtered:
            return []
        # Check for actual error keywords
        error_kw = re.compile(
            r'(error|exception|traceback|fault|killed|oom|out of memory|abort|segfault)',
            re.IGNORECASE
        )
        real_errors = [l for l in filtered if error_kw.search(l)]
        if real_errors:
            return real_errors[:10]
        # If classify_fn available, ask LLM
        if self._classify_fn:
            text = "\n".join(filtered[:10])
            if self._classify_fn("is_error", text, text[:500]):
                return filtered[:10]
        return []

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _detect_phase_from_logs(self, logs, metrics):
        """Detect phase from check-mode logs."""
        if not logs.get("rank_dirs"):
            return "startup"
        if not logs.get("stdout_log"):
            return "rendezvous"
        if metrics and metrics.get("last_iter"):
            return "training"
        return "init"

    def _apply_filter(self, text, mode):
        """Apply filter mode to log text."""
        if mode == "all":
            return text
        lines = text.splitlines()
        if mode == "errors":
            error_re = re.compile(
                r'(error|fatal|traceback|exception|killed|oom|cuda error|nccl error|out of memory)',
                re.IGNORECASE,
            )
            result = []
            for i, line in enumerate(lines):
                if error_re.search(line):
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    for j in range(start, end):
                        if lines[j] not in result:
                            result.append(lines[j])
            return "\n".join(result) if result else "(no error lines found)"
        if mode == "progress":
            prog_re = re.compile(r'(iteration\s+\d+|training\s+step|loss[:\s]|elapsed)', re.IGNORECASE)
            result = [l for l in lines if prog_re.search(l)]
            return "\n".join(result) if result else "(no progress lines found)"
        return text

    @staticmethod
    def _rank_from_path(path):
        """Extract rank number from path like .../attempt_0/6/stderr.log"""
        parts = path.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p in ("stderr.log", "stdout.log") and i > 0:
                return parts[i - 1]
        return "?"

    @staticmethod
    def _read_tail(path, n=20):
        """Read last n lines of a file, return as list."""
        content = _tail(path, n)
        if content == "(empty)":
            return ["(file not found)"]
        return [l.rstrip() for l in content.splitlines()]

    @staticmethod
    def _read_tail_from(path, offset, max_bytes=8192):
        """Read new content from offset."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                return f.read(max_bytes)
        except (FileNotFoundError, OSError):
            return ""

    @staticmethod
    def _format_watch_result(reason, poll_count, elapsed, events, lines):
        """Format watch mode result."""
        status_map = {
            "stderr_error": "TRAINING CRASHED — fatal error in stderr",
            "process_dead": "TRAINING DEAD — all processes exited",
            "hang_detected": "TRAINING HANG — step not advancing",
            "target_reached": "✓ Target step reached",
            "timeout": "⏱ Timeout (training still running)",
        }
        header = status_map.get(reason, f"Monitor: {reason}")
        parts = [f"{header} ({poll_count} polls, {int(elapsed)}s)"]

        if events:
            parts.append("Events:")
            for e in events[-10:]:
                parts.append(f"  {e}")

        if reason in ("stderr_error", "process_dead", "hang_detected"):
            parts.append("Error/last output:")
            for line in (lines or [])[-20:]:
                parts.append(f"  {line}")
            parts.append("ACTION REQUIRED: Training has failed. Diagnose the error above.")
        elif lines:
            parts.append("Recent output:")
            for line in (lines or [])[-10:]:
                parts.append(f"  {line}")

        return "\n".join(parts)
