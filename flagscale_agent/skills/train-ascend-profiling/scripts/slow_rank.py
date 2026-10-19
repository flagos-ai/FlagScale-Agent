#!/usr/bin/env python3
"""Compare per-rank training metrics and flag slow-card candidates.

Two inputs are supported: a directory of per-rank Megatron logs, or a CSV where
each row is one rank. A difference is reported together with its size and its
evidence; it is never turned into a fixed-threshold verdict.
"""
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Independently implemented; method notes in references/slow-rank-analysis.md

import argparse
import csv
import io
import json
import math
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA_VERSION = 1

STEP_MS = re.compile(r"elapsed time per iteration\s*\(ms\)\s*:\s*([0-9.eE+-]+)", re.I)
ITERATION = re.compile(r"\biteration\s+(\d+)")
HCCL = re.compile(r"\b(hccl|allreduce|allgather|reducescatter|alltoall)\w*", re.I)
RANK_FIELD = re.compile(r"rank[^0-9]{0,3}(\d+)", re.I)


def _dec(value):
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError):
        return None
    return result if result.is_finite() else None


def _robust_flags(values):
    """Modified z-scores: median/MAD, so one outlier cannot hide the others."""
    if len(values) < 3:
        return None
    centre = statistics.median(values)
    deviations = [abs(value - centre) for value in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return [0.0 if value == centre else float("inf") for value in values]
    return [0.6745 * (value - centre) / mad for value in values]


def from_logs(directory, max_files=256):
    """Collect the mean logged step time and communication-line count per rank."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("logs mode requires a directory")
    ranks, seen = {}, 0
    for path in sorted(root.rglob("*.log")):
        if seen >= max_files:
            break
        seen += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        steps = [m.group(1) for m in STEP_MS.finditer(text)]
        if not steps:
            continue
        durations = [d for d in (_dec(value) for value in steps) if d is not None]
        if not durations:
            continue
        # Prefer an explicit rank number, else use the containing directory name.
        match = RANK_FIELD.search(path.name) or RANK_FIELD.search(path.parent.name)
        rank = match.group(1) if match else path.parent.name
        entry = ranks.setdefault(rank, {"rank": rank, "step_ms": [], "comm_lines": 0,
                                        "files": []})
        entry["step_ms"].extend(float(value) for value in durations)
        entry["comm_lines"] += len(HCCL.findall(text))
        entry["files"].append(str(path))
    return ranks


def from_csv(path, rank_column, step_column, comm_column=None):
    """Collect per-rank metrics from a CSV export."""
    raw = Path(path).read_bytes()
    with io.StringIO(raw.decode("utf-8-sig", errors="replace"), newline="") as text:
        reader = csv.DictReader(text)
        headers = reader.fieldnames or []
        for needed in (rank_column, step_column):
            if needed not in headers:
                raise ValueError(f"missing CSV column: {needed}")
        ranks = {}
        for row in reader:
            rank = (row.get(rank_column) or "").strip()
            step = _dec(row.get(step_column))
            if not rank or step is None:
                continue
            entry = ranks.setdefault(rank, {"rank": rank, "step_ms": [], "comm_lines": 0,
                                            "files": [str(path)]})
            entry["step_ms"].append(float(step))
            if comm_column and row.get(comm_column):
                count = _dec(row.get(comm_column))
                if count is not None:
                    entry["comm_lines"] += int(count)
    return ranks


def compare(ranks, metric="step_ms", min_delta_pct=0.0, z_threshold=3.5):
    """Summarise per-rank metrics and list differences as candidates."""
    rows = []
    for entry in ranks.values():
        values = entry[metric]
        if not values:
            continue
        rows.append({"rank": entry["rank"], "samples": len(values),
                     "mean_ms": statistics.fmean(values),
                     "median_ms": statistics.median(values),
                     "min_ms": min(values), "max_ms": max(values),
                     "comm_lines": entry["comm_lines"],
                     "files": entry["files"][:5]})
    result = {"schema_version": SCHEMA_VERSION, "command": "compare", "metric": metric,
              "status": "ok", "ranks_compared": len(rows), "ranks": rows,
              "spread": None, "candidates": [], "limitations": [
                  "Ranks may legitimately differ: pipeline stages, expert routing and "
                  "data shards are not symmetric. A difference is a candidate, not a fault.",
                  "Means over logged steps mix warmup unless the window was trimmed before "
                  "collecting.",
                  "Communication line counts are a proxy for activity, not communication time.",
              ]}
    if len(rows) < 2:
        result["status"] = "insufficient_ranks"
        result["limitations"].append("Need at least two ranks with usable metrics.")
        return result

    values = [row["mean_ms"] for row in rows]
    low, high = min(values), max(values)
    median = statistics.median(values)
    result["spread"] = {"min_ms": low, "max_ms": high, "median_ms": median,
                        "max_over_min": high / low if low else None,
                        "spread_pct_of_median": (high - low) / median * 100 if median else None}

    zscores = _robust_flags(values)
    for row, z in zip(rows, zscores or [0.0] * len(rows)):
        delta_pct = (row["mean_ms"] - median) / median * 100 if median else 0.0
        if delta_pct <= min_delta_pct or abs(z) < z_threshold:
            continue
        result["candidates"].append({
            "rank": row["rank"], "mean_ms": row["mean_ms"],
            "delta_vs_median_pct": delta_pct, "modified_z": z,
            "evidence": {"samples": row["samples"], "files": row["files"]},
        })
    result["candidates"].sort(key=lambda item: item["delta_vs_median_pct"], reverse=True)
    if not result["candidates"]:
        result["note"] = ("No rank clears both the size and outlier bars. That is not "
                          "proof of balance; a uniform slowdown across all ranks is "
                          "invisible to a comparison like this.")
    result["method"] = {"centre": "median", "outlier": "modified z-score on MAD",
                        "z_threshold": z_threshold, "min_delta_pct": min_delta_pct}
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    logs = commands.add_parser("logs", help="per-rank metrics from a log directory")
    logs.add_argument("root")
    logs.add_argument("--metric", default="step_ms", choices=("step_ms",))
    logs.add_argument("--min-delta-pct", type=float, default=0.0)
    logs.add_argument("--z-threshold", type=float, default=3.5)
    logs.add_argument("--max-files", type=int, default=256)

    table = commands.add_parser("csv", help="per-rank metrics from a CSV")
    table.add_argument("path")
    table.add_argument("--rank-column", required=True)
    table.add_argument("--step-column", required=True)
    table.add_argument("--comm-column")
    table.add_argument("--min-delta-pct", type=float, default=0.0)
    table.add_argument("--z-threshold", type=float, default=3.5)

    try:
        args = vars(parser.parse_args())
        command = args.pop("command")
        if command == "logs":
            ranks = from_logs(args["root"], max_files=args["max_files"])
            metric = args["metric"]
        else:
            ranks = from_csv(args["path"], args["rank_column"], args["step_column"],
                             args.get("comm_column"))
            metric = "step_ms"
        output = compare(ranks, metric=metric, min_delta_pct=args["min_delta_pct"],
                         z_threshold=args["z_threshold"])
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if output["status"] == "ok" else 1
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                          "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
