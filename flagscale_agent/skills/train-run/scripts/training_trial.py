# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run one bounded, single-host FlagScale trial and preserve its evidence.

Requires a FlagScale version whose ``train --test`` waits for torchrun. Some
versions mask worker exit codes; CLI exit zero is never sufficient evidence.
This helper does not authorize device use or verify individual rank exits.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from .analyze_training_results import analyze_results


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _validate(request):
    config = Path(request["config_path"])
    cwd = Path(request["cwd"])
    if not config.is_absolute() or not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError("config_path and cwd must be absolute; cwd must exist")
    raw = config.read_bytes()
    recipe = yaml.safe_load(raw)
    if not isinstance(recipe, dict) or recipe.get("defaults", ["_self_"]) != ["_self_"]:
        raise ValueError("Use a complete recipe; composed Hydra defaults are not supported")
    if not all(isinstance(recipe.get(key), dict) for key in ("experiment", "train")):
        raise ValueError("experiment and train must be mappings in the complete recipe")
    experiment, train = recipe["experiment"], recipe["train"]
    if not all(isinstance(parent.get(key), dict) for parent, key in (
        (experiment, "runner"), (experiment, "task"), (train, "system"), (train, "model")
    )):
        raise ValueError("runner, task, system and model must be mappings")
    exp_dir = experiment["exp_dir"]
    if not isinstance(exp_dir, str) or "${" in exp_dir or not Path(exp_dir).is_absolute():
        raise ValueError("experiment.exp_dir must be a literal absolute path without interpolation")
    exp_dir = Path(exp_dir)
    if exp_dir.exists() or exp_dir.is_symlink():
        raise ValueError("experiment.exp_dir must be a new directory for this attempt")
    runner = experiment["runner"]
    if (runner.get("nnodes") != 1 or runner.get("type") != "ssh"
            or runner.get("backend") != "torchrun" or runner.get("hostfile") is not None
            or runner.get("no_shared_fs", False) or train["system"].get("no_shared_fs", False)):
        raise ValueError("Only one-host ssh/torchrun with no hostfile and shared filesystem is supported")
    if experiment.get("task", {}).get("type") != "train" or experiment["task"].get("backend") != "megatron":
        raise ValueError("Only the Megatron training backend is supported")
    logging = train["system"].get("logging", {})
    if not isinstance(logging, dict) or logging.get("log_interval") != 1:
        raise ValueError("train.system.logging.log_interval must be 1")
    if any("dir" in key or "path" in key for key in logging):
        raise ValueError("Custom logging paths are not supported")
    for name in ("expected_ranks", "end_iteration", "global_batch_size", "sequence_length"):
        if type(request.get(name)) is not int or request[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if runner.get("nproc_per_node") != request["expected_ranks"]:
        raise ValueError("expected_ranks disagrees with runner.nproc_per_node")
    for field, key in (("global_batch_size", "global_batch_size"),
                       ("sequence_length", "seq_length"), ("end_iteration", "train_iters")):
        if train["model"].get(key) != request[field]:
            raise ValueError(f"{field} disagrees with train.model.{key}")
    first, warmup = request.get("first_iteration", 1), request.get("warmup_steps", 10)
    if (type(first) is not int or first < 0 or type(warmup) is not int or warmup < 0
            or not first + warmup <= request["end_iteration"] <= first + 100000):
        raise ValueError("Invalid or empty measurement window")
    timeout = request["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(request.get("run_id"), str) or not request["run_id"] or request.get("role") not in ("baseline", "candidate"):
        raise ValueError("run_id and role baseline/candidate are required")
    model = request.get("model")
    if model is not None and (not isinstance(model, str) or not model or model.startswith("-")):
        raise ValueError("model must be a non-option model name")
    argv = ["flagscale", "train"] + ([model] if model else []) + ["-c", str(config), "--test"]
    return config, cwd, exp_dir, raw, argv


def _loss_log(exp_dir, expected_ranks):
    details = exp_dir / "logs" / "details"
    attempts = [p for p in details.rglob("attempt_*") if p.is_dir()]
    if len(attempts) != 1:
        raise ValueError(f"Expected exactly one torchrun attempt, found {len(attempts)}")
    attempt = attempts[0]
    rank_dirs = {p.name: p for p in attempt.iterdir() if p.is_dir() and p.name.isdigit()}
    if set(rank_dirs) != {str(i) for i in range(expected_ranks)}:
        raise ValueError("Rank log directories do not match expected_ranks")
    logs = [rank_dirs[str(i)] / "stdout.log" for i in range(expected_ranks)]
    candidates = []
    for path in logs:
        if not path.resolve().is_relative_to(exp_dir.resolve()) or not path.is_file():
            raise ValueError(f"Missing or foreign rank log: {path}")
        if re.search(r"\biteration\s+\d+[^\n]*\blm[ _]loss\s*:", path.read_text(errors="replace"), re.I):
            candidates.append(path)
    for path in logs + list(attempt.glob("*/stderr.log")) + [exp_dir / "agent_run" / "launcher.log"]:
        if not path.resolve().is_relative_to(exp_dir.resolve()):
            raise ValueError(f"Foreign log: {path}")
        if re.search(r"Traceback \(most recent call last\)|ChildFailedError|\b(?:ERROR|FATAL)\b",
                     path.read_text(errors="replace")):
            raise ValueError(f"Fatal error marker found in {path}; inspect this log")
    if len(candidates) != 1:
        raise ValueError(f"Expected one loss-reporting rank, found {len(candidates)}")
    return candidates[0]


def _stop_group(process):
    """Signal only the session created by this invocation, never name-based matches."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        if signum == signal.SIGTERM:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    try:
        process.wait(timeout=5)
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return "owned_process_group_gone"
    except subprocess.TimeoutExpired:
        pass
    return "cleanup_unknown"


def run_training(request):
    """Return a manifest; only measured status permits considering another trial."""
    config, cwd, exp_dir, raw, argv = _validate(request)
    # This serializes helper invocations in the same checkout, not unrelated jobs.
    lock_name = f"flagscale-training-{os.getuid()}-{hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:16]}.lock"
    with (Path(tempfile.gettempdir()) / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another training_run is active in this cwd") from exc
        exp_dir.mkdir(parents=True, exist_ok=False)
        evidence = exp_dir / "agent_run"
        evidence.mkdir()
        (evidence / "config.yaml").write_bytes(raw)
        manifest_path = evidence / "run.json"
        manifest = {
            "schema_version": 1, "run_id": request["run_id"], "role": request["role"],
            "status": "running", "argv": argv, "cwd": str(cwd), "config_path": str(config),
            "config_sha256": hashlib.sha256(raw).hexdigest(), "started_at": time.time(),
            "timeout_seconds": request["timeout_seconds"], "manifest_path": str(manifest_path),
            "launcher_log": str(evidence / "launcher.log"), "launcher_exit_code": None,
            "all_rank_completion": "not_verified",
            "warning": "CLI exit code is not an independently captured worker exit code",
        }
        _atomic_json(manifest_path, manifest)
        process = None
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise InterruptedError("Training helper interrupted by SIGTERM")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (evidence / "launcher.log").open("wb") as output:
                process = subprocess.Popen(argv, cwd=cwd, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                manifest["launcher_pid"] = process.pid
                _atomic_json(manifest_path, manifest)
                code = process.wait(timeout=request["timeout_seconds"])
            manifest["launcher_exit_code"] = code
            exit_path = evidence / "launcher.exit_code"
            exit_path.write_text(f"{code}\n")
            manifest["exit_code_path"] = str(exit_path)
            if code != 0:
                raise RuntimeError(f"FlagScale foreground launcher exited with code {code}")
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Foreground launcher exited with processes still in its owned group")
            log = _loss_log(exp_dir, request["expected_ranks"])
            manifest["log_path"] = str(log)
            analysis = {key: request[key] for key in (
                "end_iteration", "global_batch_size", "sequence_length", "first_iteration", "warmup_steps"
            ) if key in request}
            result = analyze_results(runs=[{"run_id": request["run_id"], "role": request["role"],
                                           "log_path": str(log), "exit_code_path": str(exit_path)}],
                                     output_path=str(evidence / "measurement.json"), **analysis)
            manifest["measurement_path"] = result["output_path"]
            measurement = result["runs"][0]["measurement"]
            if result["status"] != "ok":
                raise ValueError("Invalid measurement: " + ", ".join(measurement["errors"]))
            manifest.update(status="measured", mean_step_ms=measurement["step_time_ms"]["mean"],
                            tokens_per_second=measurement["tokens_per_second"])
        except (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt) as exc:
            manifest.update(status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted",
                            error=str(exc) or "KeyboardInterrupt")
            if process is not None:
                manifest["cleanup_status"] = _stop_group(process)
                manifest["launcher_exit_code"] = process.returncode
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            manifest.update(status="failed", error=str(exc))
            if process is not None:
                manifest["cleanup_status"] = _stop_group(process)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            if process is not None and process.returncode is not None:
                manifest["launcher_exit_code"] = process.returncode
                exit_path = evidence / "launcher.exit_code"
                exit_path.write_text(f"{process.returncode}\n")
                manifest["exit_code_path"] = str(exit_path)
            manifest["finished_at"] = time.time()
            _atomic_json(manifest_path, manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_training(json.loads(args.request.read_text()))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        result = {"status": "error", "error": str(exc)}
    summary = {key: value for key, value in result.items() if key not in (
        "argv", "cwd", "config_path", "config_sha256", "started_at", "finished_at", "launcher_pid"
    )}
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
