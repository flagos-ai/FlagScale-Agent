# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Analyze Megatron training logs for the train-run skill.

Use one loss-reporting log per *attempt*, with log_interval=1. Do not concatenate
rank logs or resumed attempts. Run ``python analyze_training_results.py
--request request.json`` with the arguments of ``analyze_results`` in JSON.
Stdout contains a summary; use ``--detail full`` for the complete report.
This measures logged training steps, not end-to-end job time or convergence.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)"
_ITERATION = re.compile(r"\biteration\s+(\d+)(?:\s*/\s*(\d+))?", re.I)
_FIELDS = {
    "step_ms": r"elapsed time per iteration\s*\(ms\)",
    "loss": r"lm[ _]loss",
    "grad_norm": r"grad[ _]norm",
    "global_batch_size": r"global batch size",
    "skipped_iterations": r"number of skipped iterations",
    "nan_iterations": r"number of nan iterations",
}
_PATTERNS = {
    name: re.compile(r"\b" + label + r"\s*:\s*(" + _NUMBER + r")(?=\s|\||$)", re.I)
    for name, label in _FIELDS.items()
}
_LABELS = {
    name: re.compile(r"\b" + label + r"(?=\s|:|$)", re.I)
    for name, label in _FIELDS.items()
}


def _read_evidence(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Evidence is not a file: {path}")
    with path.open("rb") as handle:
        identity = os.fstat(handle.fileno())
        raw = handle.read()
    return raw.decode("utf-8", errors="replace"), {
        "path": str(path), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "device": identity.st_dev, "inode": identity.st_ino,
    }


def _stats(values):
    if not values:
        return None
    mean = statistics.mean(values)
    median = statistics.median(values)
    return {
        "count": len(values), "mean": mean, "median": median,
        "min": min(values), "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "range_over_median_pct": 100 * (max(values) - min(values)) / median if median else None,
    }


def _parse_log(text, first, end, warmup, batch_size):
    rows = {}
    duplicates, nonfinite, malformed_fields, errors = [], [], [], []
    logged_order = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _ITERATION.search(line)
        if not match:
            continue
        # Exclude prose such as "resuming iteration 20": require a metric field.
        labels = {key: pattern.search(line) for key, pattern in _LABELS.items()}
        if not any(labels.values()):
            continue
        iteration = int(match.group(1))
        if not first <= iteration <= end:
            continue
        logged_order.append(iteration)
        row = {"iteration": iteration, "line": line_number}
        for key, label in labels.items():
            field = _PATTERNS[key].match(line, label.start()) if label else None
            if label and not field:
                malformed_fields.append({"iteration": iteration, "field": key, "line": line_number})
            value = float(field.group(1)) if field else None
            if value is not None and not math.isfinite(value):
                nonfinite.append({"iteration": iteration, "field": key, "line": line_number})
                value = None
            row[key] = value
        if iteration in rows:
            duplicates.append(iteration)
        else:
            rows[iteration] = row
    expected = set(range(first, end + 1))
    missing = sorted(expected - rows.keys())
    if missing:
        errors.append("missing_iterations")
    if duplicates:
        errors.append("duplicate_iterations: select one rank log and one attempt")
    if logged_order != sorted(logged_order):
        errors.append("out_of_order_iterations: split resumed or concatenated attempts")
    if nonfinite:
        errors.append("nonfinite_metrics")
    if malformed_fields:
        errors.append("malformed_metrics")
    mismatches = sorted(i for i, r in rows.items()
                        if r["global_batch_size"] is not None and r["global_batch_size"] != batch_size)
    if mismatches:
        errors.append("global_batch_size_disagrees_with_log")
    invalid_times = sorted(i for i, r in rows.items() if r["step_ms"] is None or r["step_ms"] <= 0)
    if invalid_times:
        errors.append("missing_or_nonpositive_step_time")
    selected = [r["step_ms"] for i, r in sorted(rows.items())
                if i >= first + warmup and r["step_ms"] is not None and r["step_ms"] > 0]
    counters = {}
    for field in ("skipped_iterations", "nan_iterations"):
        values = [r[field] for r in rows.values() if r[field] is not None]
        malformed = (any(v < 0 or v != int(v) for v in values)
                     or any(item["field"] == field for item in malformed_fields + nonfinite))
        if malformed:
            errors.append(f"invalid_{field}_counter")
        counters[field] = {
            "status": "invalid" if malformed else "known" if len(values) == len(expected) else "unknown",
            # Megatron prints these counters for the logging interval, not as a cumulative total.
            "total": int(sum(values)) if not malformed and len(values) == len(expected) else None,
            "observed_total": sum(values), "logged_count": len(values),
        }
    return rows, {
        "observed_iteration_count": len(rows), "expected_iteration_count": len(expected),
        "missing_iterations": missing, "duplicate_iterations": sorted(set(duplicates)),
        "nonfinite_metrics": nonfinite, "malformed_metrics": malformed_fields,
        "invalid_time_iterations": invalid_times,
        "batch_size_mismatch_iterations": mismatches, "counters": counters,
        "step_time_ms": _stats(selected), "errors": errors,
    }


def _completion(run, coverage_complete):
    result = {
        "iteration_coverage": "complete" if coverage_complete else "incomplete",
        "launcher_status": "unknown", "launcher_exit_code": None,
        "all_rank_completion": "not_verified", "exit_evidence": None,
    }
    if run.get("exit_code_path"):
        text, evidence = _read_evidence(run["exit_code_path"])
        if not re.fullmatch(r"\s*-?\d+\s*", text):
            raise ValueError(f"Exit evidence must contain one integer: {evidence['path']}")
        code = int(text)
        result.update(launcher_status="success" if code == 0 else "failed",
                      launcher_exit_code=code, exit_evidence=evidence)
    return result


def _loss_compare(baseline, candidate, first, end, atol, rtol):
    aligned = []
    missing = []
    for iteration in range(first, end + 1):
        left = baseline["_rows"].get(iteration, {}).get("loss")
        right = candidate["_rows"].get(iteration, {}).get("loss")
        if left is None or right is None:
            missing.append(iteration)
        else:
            aligned.append((iteration, abs(left - right), left, right))
    worst = max(aligned, key=lambda row: row[1]) if aligned else None
    tolerance_set = atol is not None or rtol is not None
    failures = [i for i, diff, left, _ in aligned
                if tolerance_set and diff > (atol or 0) + (rtol or 0) * abs(left)]
    invalid = bool(missing or baseline["measurement"]["errors"] or candidate["measurement"]["errors"])
    status = ("invalid_evidence" if invalid else "tolerance_not_set" if not tolerance_set
              else "outside_tolerance" if failures else "within_tolerance")
    return {
        "baseline_run_id": baseline["run_id"], "candidate_run_id": candidate["run_id"],
        "metric": "logged_lm_loss", "compared_iterations": len(aligned),
        "missing_loss_iterations": missing, "max_absolute_difference": worst[1] if worst else None,
        "max_difference_iteration": worst[0] if worst else None,
        "baseline_loss_at_max_difference": worst[2] if worst else None,
        "candidate_loss_at_max_difference": worst[3] if worst else None,
        "atol": atol, "rtol": rtol, "outside_tolerance_iterations": failures,
        "status": status, "parameter_equivalence": "not_verified",
        "long_term_convergence": "not_verified",
    }


def _validate(runs, first_iteration, end_iteration, warmup_steps,
              global_batch_size, sequence_length, loss_atol, loss_rtol, max_run_variation_pct):
    for name, value, minimum in (
        ("first_iteration", first_iteration, 0), ("end_iteration", end_iteration, 0),
        ("warmup_steps", warmup_steps, 0), ("global_batch_size", global_batch_size, 1),
        ("sequence_length", sequence_length, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if end_iteration < first_iteration + warmup_steps:
        raise ValueError("Measurement window is empty after warmup")
    if end_iteration - first_iteration > 100000:
        raise ValueError("Analysis window exceeds 100001 iterations; select a bounded window")
    for name, value in (("loss_atol", loss_atol), ("loss_rtol", loss_rtol),
                        ("max_run_variation_pct", max_run_variation_pct)):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(runs, list) or not 1 <= len(runs) <= 32:
        raise ValueError("runs must contain 1 to 32 attempts")
    ids, paths = set(), set()
    for run in runs:
        if not isinstance(run, dict) or run.get("role") not in ("baseline", "candidate"):
            raise ValueError("Each run requires role baseline or candidate")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in ids:
            raise ValueError("run_id must be nonempty and unique")
        ids.add(run_id)
        if not isinstance(run.get("log_path"), str) or not run["log_path"]:
            raise ValueError("Each run requires an exact log_path")
        path = str(Path(run["log_path"]).expanduser().resolve())
        if path in paths:
            raise ValueError("Repeated log_path is not an independent run")
        paths.add(path)


def analyze_results(*, runs, end_iteration, global_batch_size, sequence_length,
                    first_iteration=1, warmup_steps=10, loss_atol=None, loss_rtol=None,
                    max_run_variation_pct=None, output_path=None):
    """Analyze a single fixed candidate against baseline attempts in chronological order.

    Assumptions are explicit: log_interval=1, same workload/seed/data and batch-token
    definition across runs. The analyzer does not infer config equivalence from timing.
    ``exit_code_path`` is a file containing the real launcher exit status, captured
    by the launcher; its absence never implies successful process completion.
    """
    _validate(runs, first_iteration, end_iteration, warmup_steps, global_batch_size,
              sequence_length, loss_atol, loss_rtol, max_run_variation_pct)
    result = {
        "schema_version": 1, "status": "ok",
        "window": {"first_iteration": first_iteration, "end_iteration": end_iteration,
                   "warmup_steps": warmup_steps, "measured_first_iteration": first_iteration + warmup_steps,
                   "log_interval": 1, "loss_comparison_includes_warmup": True},
        "workload": {"global_batch_size": global_batch_size, "sequence_length": sequence_length,
                     "tokens_per_iteration": global_batch_size * sequence_length,
                     "config_data_seed_equivalence": "caller_must_verify"},
        "run_order_basis": "caller_supplied_chronological_order",
        "runs": [], "comparison": None,
        "limitations": [
            "Throughput assumes every sample has sequence_length tokens; padding/packing and real-data throughput require separate accounting.",
            "Log intervals must be 1. Use one loss-reporting rank and one attempt per log.",
            "Timing covers logged training steps only; profiling overhead, job setup and data representativeness are not assessed.",
            "Logged loss tolerance does not establish parameter equality or long-term convergence.",
            "Run spread is descriptive, not a confidence interval or statistical significance test.",
            "Launcher exit evidence does not independently verify all-rank completion.",
            "Each attempt must have distinct original log evidence. Shared inodes and byte-identical logs are rejected; distinct paths or run_ids alone do not prove independent runs.",
        ],
    }
    evidence_paths = set()
    log_identities, log_hashes = {}, {}
    for run in runs:
        text, evidence = _read_evidence(run["log_path"])
        identity = (evidence["device"], evidence["inode"])
        if identity in log_identities:
            raise ValueError(
                f"Runs {log_identities[identity]!r} and {run['run_id']!r} use the same physical "
                "log file (device/inode); hard links are not independent runs"
            )
        if evidence["sha256"] in log_hashes:
            raise ValueError(
                f"Runs {log_hashes[evidence['sha256']]!r} and {run['run_id']!r} have byte-identical "
                "log evidence; independent attempts cannot be established. Supply original logs "
                "with captured attempt identity/timestamps, or analyze each run separately; do not alter copies"
            )
        log_identities[identity] = run["run_id"]
        log_hashes[evidence["sha256"]] = run["run_id"]
        evidence_paths.add(evidence["path"])
        rows, measurement = _parse_log(text, first_iteration, end_iteration,
                                       warmup_steps, global_batch_size)
        completion = _completion(run, not measurement["missing_iterations"])
        if completion["exit_evidence"]:
            evidence_paths.add(completion["exit_evidence"]["path"])
        if completion["launcher_status"] == "failed":
            measurement["errors"].append("launcher_failed")
        if measurement["errors"]:
            result["status"] = "invalid_evidence"
        times = measurement["step_time_ms"]
        measurement["tokens_per_second"] = (
            global_batch_size * sequence_length * 1000 / times["mean"] if times else None
        )
        measurement["valid"] = not measurement["errors"]
        result["runs"].append({"run_id": run["run_id"], "role": run["role"],
                               "log_evidence": evidence, "measurement": measurement,
                               "completion": completion, "_rows": rows})
    baseline = [r for r in result["runs"] if r["role"] == "baseline"]
    candidate = [r for r in result["runs"] if r["role"] == "candidate"]
    if baseline and candidate:
        role_stats = {
            role: _stats([r["measurement"]["step_time_ms"]["mean"] for r in group
                          if r["measurement"]["valid"] and r["measurement"]["step_time_ms"]])
            for role, group in (("baseline", baseline), ("candidate", candidate))
        }
        # Never silently drop invalid attempts and declare the remaining runs successful.
        all_valid = result["status"] == "ok"
        speedup = (role_stats["baseline"]["median"] / role_stats["candidate"]["median"]
                   if all_valid and all(role_stats.values()) else None)
        pair_order = (len(runs) % 2 == 0 and all(
            {runs[i]["role"], runs[i + 1]["role"]} == {"baseline", "candidate"}
            for i in range(0, len(runs), 2)))
        enough_repeats = min(len(baseline), len(candidate)) >= 2
        paired = []
        if pair_order:
            for i in range(0, len(result["runs"]), 2):
                pair = result["runs"][i:i + 2]
                b = next(r for r in pair if r["role"] == "baseline")
                c = next(r for r in pair if r["role"] == "candidate")
                paired.append((b, c))
        else:
            paired = [(baseline[0], c) for c in candidate]
        loss_checks = [_loss_compare(b, c, first_iteration, end_iteration, loss_atol, loss_rtol)
                       for b, c in paired]
        if not all_valid:
            stability = "invalid_evidence"
        elif not enough_repeats:
            stability = "insufficient_baseline_or_candidate_repeats"
        elif not pair_order:
            stability = "not_interleaved_pairs"
        elif max_run_variation_pct is None:
            stability = "variation_limit_not_set"
        elif any(s["range_over_median_pct"] > max_run_variation_pct for s in role_stats.values()):
            stability = "outside_declared_variation_limit"
        else:
            stability = "within_declared_variation_limit"
        counters = [r["measurement"]["counters"] for r in result["runs"]]
        counters_clean = all(v["status"] == "known" and v["total"] == 0
                             for counter in counters for v in counter.values())
        completion_known = all(r["completion"]["launcher_status"] == "success" for r in result["runs"])
        result["comparison"] = {
            "performance": {
                "aggregation": "ratio of median per-run mean step times",
                "run_mean_step_time_ms": role_stats,
                "tokens_per_second": {
                    role: (global_batch_size * sequence_length * 1000 / stats["median"]
                           if all_valid and stats else None)
                    for role, stats in role_stats.items()
                },
                "observed_speedup": speedup,
                "time_reduction_pct": 100 * (1 - 1 / speedup) if speedup else None,
                "throughput_improvement_pct": 100 * (speedup - 1) if speedup else None,
                "observed_improvement": speedup > 1 if speedup is not None else None,
            },
            "repeatability": {"baseline_runs": len(baseline), "candidate_runs": len(candidate),
                              "interleaved_pairs": pair_order, "status": stability,
                              "max_run_variation_pct": max_run_variation_pct},
            "loss_checks": loss_checks,
            "loss_pairing": "adjacent_chronological_pairs" if pair_order else "first_baseline_to_each_candidate",
            "acceptance_checks": {
                "all_measurements_valid": all_valid,
                "all_launcher_exits_successful": completion_known,
                "all_logged_skip_nan_counters_known_and_zero": counters_clean,
                "all_loss_checks_within_declared_tolerance": all(
                    check["status"] == "within_tolerance" for check in loss_checks),
                "repeatability_within_declared_limit": stability == "within_declared_variation_limit",
                "all_rank_completion": "not_verified",
                "workload_equivalence": "caller_must_verify",
            },
        }
    for run in result["runs"]:
        del run["_rows"]
    if output_path:
        target = Path(output_path).expanduser().resolve()
        if str(target) in evidence_paths:
            raise ValueError("output_path must not overwrite an evidence file")
        if not target.parent.is_dir():
            raise ValueError(f"Output parent directory does not exist: {target.parent}")
        result["output_path"] = str(target)
        # Write atomically, so consumers never read an incomplete JSON report.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                             prefix=".training-results-", delete=False) as handle:
                temp_name = handle.name
                json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    return result


def summarize_results(result):
    """Keep decision metrics in the summary; detailed evidence stays in the report.

    This is a projection of the full result, not another acceptance decision.
    Missing launcher/counter evidence and unverified checks remain explicit.
    """
    summary = {
        "schema_version": result["schema_version"], "status": result["status"],
        "detail": "summary", "output_path": result.get("output_path"),
        "window": result["window"], "workload": result["workload"],
        "runs": [], "comparison": None,
        "unverified": ["all_rank_completion", "parameter_equivalence", "long_term_convergence"],
    }
    for run in result["runs"]:
        measurement = run["measurement"]
        times = measurement["step_time_ms"]
        item = {
            "run_id": run["run_id"], "role": run["role"],
            "mean_step_time_ms": times["mean"] if measurement["valid"] and times else None,
            "tokens_per_second": measurement["tokens_per_second"] if measurement["valid"] else None,
            "iteration_coverage": run["completion"]["iteration_coverage"],
            "launcher_status": run["completion"]["launcher_status"],
            "skip_nan_counters": {
                key: value["total"] if value["status"] == "known" else value["status"]
                for key, value in measurement["counters"].items()
            },
        }
        if measurement["errors"]:
            item["errors"] = measurement["errors"]
        summary["runs"].append(item)
    comparison = result["comparison"]
    if comparison:
        performance = comparison["performance"]
        summary["comparison"] = {
            "performance": {
                key: value for key, value in performance.items() if key != "run_mean_step_time_ms"
            },
            "repeatability": comparison["repeatability"],
            "loss_checks": [
                {key: check[key] for key in (
                    "baseline_run_id", "candidate_run_id", "max_absolute_difference",
                    "max_difference_iteration", "atol", "rtol", "status",
                )}
                for check in comparison["loss_checks"]
            ],
            "acceptance_checks": comparison["acceptance_checks"],
        }
        summary["comparison"]["performance"]["median_run_mean_step_time_ms"] = {
            role: stats["median"] if result["status"] == "ok" and stats else None
            for role, stats in performance["run_mean_step_time_ms"].items()
        }
        summary["comparison"]["performance"]["run_variation_pct"] = {
            role: stats["range_over_median_pct"] if result["status"] == "ok" and stats else None
            for role, stats in performance["run_mean_step_time_ms"].items()
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="JSON request using analyze_results arguments")
    parser.add_argument("--detail", choices=("summary", "full"), default="summary",
                        help="Stdout detail; output_path always saves the full report")
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = analyze_results(**request)
        if args.detail == "summary":
            result = summarize_results(result)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
