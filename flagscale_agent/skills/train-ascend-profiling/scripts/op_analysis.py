#!/usr/bin/env python3
"""Operator-level profiling analysis: hot-op ranking, instruction-ratio pivot,
shape parsing, and an optional Roofline sanity check.

Reads Ascend profiler CSV exports. Everything is reported as an observation plus
its evidence boundary; nothing here is a verdict about what to optimize.
"""
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Independently implemented; method notes in references/operator-analysis.md

import argparse
import csv
import io
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA_VERSION = 1

# Instruction-pipeline ratio columns seen in Ascend kernel exports.
RATIO_COLUMNS = ("aic_mac_ratio", "aic_scalar_ratio", "aic_saclar_ratio",
                 "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
                 "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_saclar_ratio",
                 "aiv_mte2_ratio", "aiv_mte3_ratio")

DURATION = ("Duration(us)", "Task Duration(us)", "Total Duration(us)")


def _num(value):
    try:
        result = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    return result if result.is_finite() else None


def _read_csv(path, max_rows):
    """Return (headers, rows, truncated) for one CSV, tolerating short rows."""
    with path.open("rb") as stream:
        raw = stream.read()
    with io.StringIO(raw.decode("utf-8-sig", errors="replace"), newline="") as text:
        reader = csv.reader(text, strict=False)
        headers = [h.strip() for h in next(reader, [])]
        rows, truncated = [], False
        for values in reader:
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append({headers[i]: value for i, value in enumerate(values)
                         if i < len(headers)})
    return headers, rows, truncated


def _find(directory, names, max_depth=6):
    """Bounded recursive search for the named CSV files."""
    found, root = [], Path(directory)
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with __import__("os").scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if depth < max_depth:
                            stack.append((path, depth + 1))
                    elif entry.is_file(follow_symlinks=False) and entry.name in names:
                        found.append(path)
        except OSError:
            continue
    return sorted(found)


def _group_key(row):
    op = (row.get("OP Type") or row.get("Type") or row.get("Name") or "?").strip()
    shape = (row.get("Input Shapes") or "").strip()
    dtype = (row.get("Input Data Types") or "").strip()
    return op, shape, dtype


def _op_column(headers):
    return next((c for c in ("OP Type", "Type", "Op Name", "Name") if c in headers), None)


def _duration_column(headers):
    return next((c for c in DURATION if c in headers), None)


def hot_ops(directory, top=5, max_rows=200000):
    """Rank operator groups by cumulative recorded duration.

    Per-invocation files are preferred. Aggregate tables (op_statistic) are only
    used when no per-invocation file exists, because mixing the two double-counts
    the same work and splits a group by whether a shape column was present.
    """
    per_invocation = _find(directory, {"op_summary.csv", "kernel_details.csv"})
    aggregate = _find(directory, {"op_statistic.csv"})
    files = per_invocation or aggregate
    result = {"schema_version": SCHEMA_VERSION, "command": "hot-ops",
              "source": str(Path(directory).absolute()), "status": "ok",
              "files_used": [], "top": [], "limitations": [
                  "Cumulative operator duration is not critical-path contribution; a hot "
                  "operator may be fully overlapped or may be intrinsic to the workload.",
                  "Rows outside the scanned files/rows are not represented.",
              ]}
    if not files:
        result["status"] = "no_supported_csv"
        result["limitations"].append(
            "Expected one of op_statistic.csv / op_summary.csv / kernel_details.csv.")
        return result

    totals, truncated_any = {}, False
    for path in files:
        headers, rows, truncated = _read_csv(path, max_rows)
        truncated_any = truncated_any or truncated
        duration = _duration_column(headers)
        op = _op_column(headers)
        if not duration or not op:
            continue
        result["files_used"].append(str(path))
        for row in rows:
            value = _num(row.get(duration))
            if value is None or value < 0:
                continue
            # Group by operator name: the question is which operator dominates.
            # Grouping by (operator, shape) would split one operator across many
            # entries and let a less important operator outrank the real hotspot.
            name = (row.get(op) or "").strip()
            key = name or "<unnamed>"
            entry = totals.setdefault(key, {"calls": 0, "duration_us": Decimal(0),
                                            "shapes": set(), "dtypes": set()})
            entry["calls"] += 1
            entry["duration_us"] += value
            shape = (row.get("Input Shapes") or "").strip()
            if shape:
                entry["shapes"].add(shape)
            dtype = (row.get("Input Data Types") or "").strip()
            if dtype:
                entry["dtypes"].add(dtype)

    if not totals:
        result["status"] = "no_usable_rows"
        result["limitations"].append("Files were found but none carried both an operator "
                                    "column and a duration column.")
        return result

    grand = sum((entry["duration_us"] for entry in totals.values()), Decimal(0))
    ranked = sorted(totals.items(), key=lambda item: item[1]["duration_us"], reverse=True)
    for op, entry in ranked[:top]:
        share = float(entry["duration_us"] / grand) if grand else None
        result["top"].append({"op": op,
                              "input_shapes": sorted(entry["shapes"])[:10],
                              "distinct_shapes": len(entry["shapes"]),
                              "dtypes": sorted(entry["dtypes"]),
                              "calls": entry["calls"],
                              "duration_sum_us": float(entry["duration_us"]),
                              "share_of_scanned_duration": share})
    result["scanned_operator_groups"] = len(totals)
    result["scanned_duration_us"] = float(grand)
    if truncated_any:
        result["status"] = "partial"
        result["limitations"].append("Row limit reached; ranking covers the scanned rows only.")
    result["note"] = ("Shares describe the scanned rows, not a window. To attribute a "
                      "bottleneck, confirm the window and the call counts against the "
                      "training step being analysed.")
    return result


def pivot(directory, ops=None, top=5, max_rows=200000):
    """Per-shape average instruction-pipeline ratios for the hottest operators."""
    hot = hot_ops(directory, top=top, max_rows=max_rows)
    result = {"schema_version": SCHEMA_VERSION, "command": "pivot",
              "source": hot["source"], "status": "ok", "rows": [],
              "limitations": list(hot["limitations"])}
    if hot["status"] not in ("ok", "partial"):
        result["status"] = hot["status"]
        return result
    wanted = set(ops) if ops else {item["op"] for item in hot["top"]}

    files = _find(directory, {"op_summary.csv", "kernel_details.csv"})
    groups = {}
    for path in files:
        headers, rows, truncated = _read_csv(path, max_rows)
        duration, op = _duration_column(headers), _op_column(headers)
        ratios = [c for c in RATIO_COLUMNS if c in headers]
        if not duration or not op or not ratios:
            continue
        if truncated:
            result["status"] = "partial"
        for row in rows:
            name = (row.get(op) or "").strip()
            if name not in wanted:
                continue
            value = _num(row.get(duration))
            if value is None or value < 0:
                continue
            key = (name, (row.get("Input Shapes") or "").strip())
            entry = groups.setdefault(key, {"calls": 0, "duration_us": Decimal(0),
                                            "_ratios": {c: [] for c in ratios}})
            entry["calls"] += 1
            entry["duration_us"] += value
            for column_name in ratios:
                ratio = _num(row.get(column_name))
                if ratio is not None:
                    entry["_ratios"][column_name].append(ratio)

    if not groups:
        result["status"] = "no_ratio_columns"
        result["limitations"].append(
            "No rows matched the selected operators with instruction-ratio columns "
            "present. Aggregate exports may omit them.")
        return result

    for (name, shape), entry in sorted(groups.items(),
                                       key=lambda item: item[1]["duration_us"],
                                       reverse=True):
        row = {"op": name, "input_shapes": shape, "calls": entry["calls"],
               "duration_sum_us": float(entry["duration_us"]),
               "mean_ratios": {}}
        for column_name, values in entry["_ratios"].items():
            if values:
                row["mean_ratios"][column_name] = float(sum(values) / len(values))
        result["rows"].append(row)
    result["note"] = ("Ratios are per-row means over the scanned rows. A high mac ratio "
                      "means the cube pipeline was busy, not that the kernel is at peak.")
    return result


def parse_shapes(directory, op="matmul", max_rows=200000, limit=50):
    """Split 'Input Shapes' into named dimensions for simple matrix operators."""
    shapes, unparsed = {}, 0
    files = _find(directory, {"kernel_details.csv", "op_summary.csv"})
    lower = op.lower()
    for path in files:
        headers, rows, _ = _read_csv(path, max_rows)
        op_col = _op_column(headers)
        if not op_col or "Input Shapes" not in headers:
            continue
        for row in rows:
            if lower not in (row.get(op_col) or "").lower():
                continue
            raw = (row.get("Input Shapes") or "").strip()
            dims = []
            for part in raw.split(";"):
                numbers = [piece for piece in part.replace("x", ",").split(",") if piece.strip()]
                try:
                    dims.append([int(float(piece)) for piece in numbers if piece.strip()])
                except ValueError:
                    dims = []
                    break
            if len(dims) == 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                m, k = dims[0]
                k2, n = dims[1]
                entry = shapes.setdefault(raw, {"input_shapes": raw, "M": m, "K": k,
                                                "N": n, "k_consistent": k == k2,
                                                "calls": 0})
                entry["calls"] += 1
            else:
                unparsed += 1
    ordered = sorted(shapes.values(), key=lambda item: item["calls"], reverse=True)
    return {"schema_version": SCHEMA_VERSION, "command": "shapes", "op": op,
            "status": "ok" if shapes else "no_parsed_shapes",
            "parsed": ordered[:limit], "distinct_shapes": len(shapes),
            "unparsed_rows": unparsed,
            "limitations": [
                "Only the two-2D-operand matrix form is named; batched or fused "
                "variants (BatchMatMul, GroupedMatmul) need their own interpretation.",
                "A K mismatch is reported rather than silently accepted.",
            ]}


def roofline(path, peak_flops, bandwidth_gbs, mac_threshold=0.80, top=10,
             max_rows=200000, int8_scale=2):
    """Compare a lower-bound theoretical time against the observed kernel time.

    Only operators whose mac ratio clears the threshold are checked, because the
    bound assumes the compute pipeline is the limit. FLOPs are derived from the
    named MatMul shape; unmodelled operators are counted, never guessed.
    """
    path = Path(path)
    headers, rows, truncated = _read_csv(path, max_rows)
    duration = _duration_column(headers)
    op_col = _op_column(headers)
    result = {"schema_version": SCHEMA_VERSION, "command": "roofline",
              "source": str(path.absolute()), "status": "ok", "peak_flops": peak_flops,
              "bandwidth_gbs": bandwidth_gbs, "mac_threshold": mac_threshold,
              "checked": [], "unmodelled": [], "skipped_no_ratio": 0,
              "limitations": [
                  "Theoretical time is max(compute, memory) under ideal overlap; real "
                  "kernels rarely reach it, so a low efficiency is a prompt to look, "
                  "not proof of a fixable stall.",
                  "Efficiency above 1.0 means the model undercounts traffic or FLOPs.",
                  "FLOPs come from the named shape and dtype only.",
              ]}
    if not duration or not op_col or "aic_mac_ratio" not in headers:
        result["status"] = "unsupported_schema"
        result["limitations"].append(
            "Needs a duration column, an operator column and aic_mac_ratio.")
        return result
    if truncated:
        result["status"] = "partial"

    band = bandwidth_gbs * 1e9
    for row in rows:
        ratio = _num(row.get("aic_mac_ratio"))
        if ratio is None:
            result["skipped_no_ratio"] += 1
            continue
        if float(ratio) < mac_threshold:
            continue
        observed = _num(row.get(duration))
        raw = (row.get("Input Shapes") or "").strip()
        dims = []
        for part in raw.split(";"):
            numbers = [p for p in part.replace("x", ",").split(",") if p.strip()]
            try:
                dims.append([int(float(p)) for p in numbers if p.strip()])
            except ValueError:
                dims = []
                break
        name = (row.get(op_col) or "").strip()
        if len(dims) != 2 or len(dims[0]) != 2 or len(dims[1]) != 2 or observed is None:
            result["unmodelled"].append({"op": name, "input_shapes": raw,
                                         "reason": "shape not a two-operand matrix"})
            continue
        m, k = dims[0]
        _, n = dims[1]
        flops = 2.0 * m * k * n
        # Lower bound on bytes moved: read both operands, write the output.
        dtype_bytes = _dtype_bytes(row.get("Input Data Types"), int8_scale)
        moved = (m * k + k * n + m * n) * dtype_bytes
        theoretical_us = max(flops / (peak_flops * 1e12), moved / band) * 1e6
        if theoretical_us <= 0:
            continue
        efficiency = theoretical_us / float(observed)
        result["checked"].append({"op": name, "input_shapes": raw, "M": m, "K": k, "N": n,
                                  "flops": flops, "bytes_moved_lower_bound": moved,
                                  "theoretical_us": theoretical_us,
                                  "observed_us": float(observed),
                                  "efficiency": efficiency,
                                  "modelled_bound": "compute" if flops / (peak_flops * 1e12) >= moved / band else "memory"})
    result["checked"] = sorted(result["checked"], key=lambda item: item["observed_us"],
                               reverse=True)[:top]
    if not result["checked"]:
        result["status"] = "no_checkable_kernels"
    return result


def _dtype_bytes(label, int8_scale):
    text = (label or "").lower()
    if "int8" in text or "fp8" in text:
        return 1
    if "int4" in text:
        return 0.5
    if "fp32" in text or "float32" in text or "bf16" in text or "fp16" in text or "float16" in text:
        return 2
    return 2


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    hot = commands.add_parser("hot-ops", help="rank operator groups by cumulative duration")
    hot.add_argument("root")
    hot.add_argument("--top", type=int, default=5)
    hot.add_argument("--max-rows", type=int, default=200000)

    piv = commands.add_parser("pivot", help="mean instruction ratios per operator shape")
    piv.add_argument("root")
    piv.add_argument("--op", action="append", help="restrict to this operator (repeatable)")
    piv.add_argument("--top", type=int, default=5)
    piv.add_argument("--max-rows", type=int, default=200000)

    shp = commands.add_parser("shapes", help="name matrix dimensions of one operator")
    shp.add_argument("root")
    shp.add_argument("--op", default="matmul")
    shp.add_argument("--max-rows", type=int, default=200000)
    shp.add_argument("--limit", type=int, default=50)

    roo = commands.add_parser("roofline", help="theoretical vs observed for hot kernels")
    roo.add_argument("csv")
    roo.add_argument("--peak-flops", type=float, required=True,
                     help="FP16/BF16 cube peak TFLOPS for one device")
    roo.add_argument("--bandwidth", type=float, required=True,
                     help="HBM bandwidth, in GB/s unless --bw-unit says otherwise")
    roo.add_argument("--bw-unit", default="GB/s", choices=("GB/s", "TB/s"))
    roo.add_argument("--mac-threshold", type=float, default=0.80)
    roo.add_argument("--int8-scale", type=float, default=2)
    roo.add_argument("--top", type=int, default=10)
    roo.add_argument("--max-rows", type=int, default=200000)

    try:
        args = vars(parser.parse_args())
        command = args.pop("command")
        if command == "hot-ops":
            output = hot_ops(args["root"], top=args["top"], max_rows=args["max_rows"])
        elif command == "pivot":
            output = pivot(args["root"], ops=args["op"], top=args["top"],
                           max_rows=args["max_rows"])
        elif command == "shapes":
            output = parse_shapes(args["root"], op=args["op"], max_rows=args["max_rows"],
                                  limit=args["limit"])
        else:
            bandwidth = args["bandwidth"]
            if args.get("bw_unit") == "TB/s":
                bandwidth *= 1000.0
            output = roofline(args["csv"], args["peak_flops"], bandwidth,
                              mac_threshold=args["mac_threshold"], top=args["top"],
                              max_rows=args["max_rows"], int8_scale=args["int8_scale"])
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if output.get("status") in ("ok", "partial", "no_parsed_shapes",
                                             "no_checkable_kernels") else 1
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                          "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
