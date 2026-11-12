#!/usr/bin/env python3
"""Bounded, offline inspection of Ascend profiler exports (Python standard library)."""
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Independently implemented; reference methods and upstream licenses: ../../../knowledge/docs/ascend_profiling/collection-and-analysis.md

import argparse
import csv
import io
import json
import math
import os
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def checked_path(value):
    path = Path(value).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink paths are not followed")
    return path


def decimal(value):
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError("invalid numeric value") from exc
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ValueError("numeric value must be finite")
    return result


def positive(value):
    result = int(value)
    if result <= 0:
        raise ValueError("limits must be positive")
    return result


def inventory(root, max_files=200, max_depth=3, max_read_bytes=65536, max_total_read_bytes=1048576):
    root = checked_path(root)
    if not root.is_dir():
        raise ValueError("inventory requires a directory")
    result = {"schema_version": 1, "command": "inventory", "source": str(root), "files": [],
              "limits": dict(max_files=max_files, max_depth=max_depth, max_read_bytes=max_read_bytes,
                             max_total_read_bytes=max_total_read_bytes),
              "entries_seen": 0, "bytes_read": 0, "skipped_symlinks": 0, "limitations": []}
    found, stack = set(), [(root, 0)]
    while stack and result["entries_seen"] < max_files:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if result["entries_seen"] >= max_files:
                        result["limitations"].append("entry_limit")
                        break
                    result["entries_seen"] += 1
                    if entry.is_symlink():
                        result["skipped_symlinks"] += 1
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.startswith("PROF_"):
                            found.add("raw_prof_dir")
                        if depth < max_depth:
                            stack.append((path, depth + 1))
                        else:
                            result["limitations"].append("depth_limit")
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    name = entry.name.lower()
                    kind = "other"
                    if name.endswith(".csv"):
                        kind = "csv"
                        found.add("kernel_csv" if name == "kernel_details.csv" else "other_csv")
                    elif name.endswith(".db"):
                        kind = "database"
                    elif (name == "trace_view.json" or name.endswith(".pt.trace.json")
                          or name.endswith(".pt.trace.json.gz")
                          or name.startswith("rank-") and name.endswith((".json", ".json.gz"))
                          or name.startswith("msprof_") and name.endswith(".json")):
                        kind = "trace"
                    elif name in ("communication.json", "communication_matrix.json"):
                        kind = "communication"
                    found.add(kind)
                    item = {"path": str(path), "bytes": entry.stat(follow_symlinks=False).st_size, "kind": kind}
                    if kind == "csv":
                        allowance = min(max_read_bytes, max_total_read_bytes - result["bytes_read"])
                        if allowance <= 0:
                            item["header_status"] = "read_budget_exhausted"
                        else:
                            with path.open("rb") as stream:
                                raw = stream.readline(allowance)
                            result["bytes_read"] += len(raw)
                            try:
                                if len(raw) == allowance and not raw.endswith(b"\n"):
                                    raise ValueError("header_read_limit")
                                item["headers"] = next(csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True), [])
                                item["header_status"] = "read"
                            except (UnicodeError, csv.Error, ValueError) as exc:
                                item["header_status"] = str(exc)
                    result["files"].append(item)
        except OSError as exc:
            result["limitations"].append(f"unreadable:{directory}:{exc.strerror}")
    if stack:
        result["limitations"].append("entry_limit")
    result["files"].sort(key=lambda item: item["path"])
    expected = {"kernel_csv", "trace", "communication", "database", "raw_prof_dir"}
    result["present"] = sorted(found & expected)
    result["not_observed"] = sorted(expected - found)
    result["limitations"] = sorted(set(result["limitations"]))
    result["note"] = "Presence and CSV headers only; not_observed is not absence if the inventory was bounded. No conversion performed."
    return result


def column(headers, *names):
    return next((name for name in names if name in headers), None)


def window(source, start_us, end_us, device_id=None, max_bytes=8388608, max_rows=100000, top=10):
    source = checked_path(source)
    start, end = decimal(start_us), decimal(end_us)
    if end <= start:
        raise ValueError("end-us must be greater than start-us")
    if not math.isfinite(float(end - start)):
        raise ValueError("window duration exceeds finite range")
    size = source.stat().st_size
    if not source.is_file() or size > max_bytes:
        raise ValueError("input is not a regular file or exceeds max-bytes; inspect inventory or export a smaller file")
    result = {"schema_version": 1, "command": "window", "source": str(source), "source_bytes": size,
              "window": {"start_us": str(start), "end_us": str(end), "duration_us": float(end - start)},
              "limits": {"max_bytes": max_bytes, "max_rows": max_rows, "top": top},
              "rows_read": 0, "invalid_rows": [], "invalid_row_count": 0, "truncated": False,
              "metrics": None, "uncovered_windows": [], "top_ops": [], "limitations": []}
    events, devices, unknown_device = [], set(), False
    with source.open("rb") as raw_stream:
        raw = raw_stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("input grew beyond max-bytes; export a smaller stable file")
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.reader(stream, strict=True)
        headers = [h.strip() for h in next(reader, [])]
        result["headers"] = headers
        if len(headers) != len(set(headers)):
            raise ValueError("duplicate CSV columns")
        pair = next(((s, d) for s, d in [("Start Time(us)", "Duration(us)"),
                                        ("Task Start Time(us)", "Task Duration(us)")]
                     if s in headers and d in headers), None)
        if not pair:
            result.update(status="unsupported_schema", missing_fields=["per-invocation start and duration columns"])
            result["limitations"].append("Aggregate statistics cannot establish event intervals or gaps.")
            return result
        dev = column(headers, "Device_id", "Device ID", "Device Id", "device_id")
        name = column(headers, "Name", "Op Name")
        op_type = column(headers, "Type", "OP Type", "Task Type", "Accelerator Core")
        task_type = column(headers, "Task Type", "Accelerator Core")
        shape, dtype = column(headers, "Input Shapes"), column(headers, "Input Data Types")
        stream_col = column(headers, "Stream ID")
        result["missing_fields"] = [label for label, value in [("device", dev), ("name", name),
                                    ("type", op_type), ("task_type", task_type), ("shape", shape),
                                    ("dtype", dtype), ("stream", stream_col)] if value is None]
        while True:
            line = reader.line_num + 1
            try:
                values = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                result["invalid_row_count"] += 1
                result["invalid_rows"].append({"line": line, "reason": str(exc)})
                result["truncated"] = True
                break
            if result["rows_read"] >= max_rows:
                result["truncated"] = True
                break
            result["rows_read"] += 1
            row = dict(zip(headers, values))
            row_device = row.get(dev, "").strip()
            if row_device:
                devices.add(row_device)
            else:
                unknown_device = True
            try:
                if len(values) != len(headers):
                    raise ValueError("column count differs from header")
                s, duration = decimal(row[pair[0]]), decimal(row[pair[1]])
                if duration < 0:
                    raise ValueError("negative duration")
                e = s + duration
                if not math.isfinite(float(e)):
                    raise ValueError("event end exceeds finite range")
                if device_id is not None and row_device != str(device_id):
                    continue
                if s < end and e > start and duration > 0:
                    events.append({"start": max(s, start), "end": min(e, end), "line": line,
                                   "line_end": reader.line_num, "name": row.get(name, ""),
                                   "type": row.get(op_type, ""), "task_type": row.get(task_type, ""),
                                   "shape": row.get(shape, ""), "dtype": row.get(dtype, ""),
                                   "stream": row.get(stream_col, ""), "device_id": row_device})
            except ValueError as exc:
                result["invalid_row_count"] += 1
                if len(result["invalid_rows"]) < 20:
                    result["invalid_rows"].append({"line": line, "reason": str(exc)})
    result["devices_observed"] = sorted(devices)
    result["scope"] = {"device_id": str(device_id) if device_id is not None else next(iter(devices), None),
                       "clock_domain": "single CSV device; external clock alignment not verified"}
    if not dev or unknown_device or (len(devices) > 1 and device_id is None):
        result.update(status="unknown_scope")
        result["scope"]["device_id"] = None
        result["limitations"].append("Cannot establish one device clock domain; multi-device input requires --device-id and populated device columns.")
        return result
    if not events:
        result.update(status="no_window_events")
        result["limitations"].append("No positive-duration tasks overlap this window; capture coverage is unknown, not 100% idle.")
        return result
    complete = not result["truncated"] and not result["invalid_row_count"]
    result["status"] = "ok" if complete else "partial"
    merged, buckets = [], defaultdict(list)
    for event in sorted(events, key=lambda item: (item["start"], item["end"])):
        buckets[(event["name"], event["type"], event["shape"], event["dtype"])].append(event)
        if merged and event["start"] <= merged[-1][1]:
            if event["end"] > merged[-1][1]:
                merged[-1][1], merged[-1][3] = event["end"], event
        else:
            merged.append([event["start"], event["end"], event, event])
    busy = sum((b - a for a, b, _, _ in merged), Decimal(0))
    result["metrics"] = {"observed_task_union_us": float(busy), "uncovered_us": float(end - start - busy),
                         "uncovered_ratio": float((end - start - busy) / (end - start)),
                         "intersecting_tasks": len(events), "complete_csv_scan": complete,
                         "interpretation": "exact_for_valid_recorded_rows" if complete else "busy_lower_bound_uncovered_upper_bound"}
    def evidence(event):
        return {key: value for key, value in event.items() if key not in ("start", "end")} if event else None
    gaps, cursor, before = [], start, None
    for s, e, first, last in merged + [[end, end, None, None]]:
        if s > cursor:
            gaps.append({"start_us": str(cursor), "end_us": str(s), "duration_us": float(s - cursor),
                         "kind": "pre_window" if before is None else "tail_window" if first is None else "internal",
                         "before": evidence(before), "after": evidence(first)})
        cursor, before = e, last
    result["uncovered_window_count"] = len(gaps)
    result["uncovered_windows"] = sorted(gaps, key=lambda gap: gap["duration_us"], reverse=True)[:top]
    result["uncovered_windows_omitted"] = max(0, len(gaps) - top)
    for (name, kind, shape, dtype), group in buckets.items():
        total = sum((event["end"] - event["start"] for event in group), Decimal(0))
        result["top_ops"].append({"name": name, "type": kind, "shape": shape, "dtype": dtype,
                                  "calls": len(group), "clipped_duration_sum_us": float(total),
                                  "example_lines": [event["line"] for event in group[:5]]})
    result["top_ops"] = sorted(result["top_ops"], key=lambda op: op["clipped_duration_sum_us"], reverse=True)[:top]
    result["limitations"].extend(["Uncovered means no recorded task; capture completeness and host/device alignment are not verified.",
                                  "Window edges are not step boundaries. Operator duration sums are not critical-path contributions."])
    if not complete:
        result["limitations"].append("Missing or rejected rows can fill reported uncovered windows; do not claim exact device idle.")
    return result


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("root")
    for name, default in [("max-files", 200), ("max-depth", 3), ("max-read-bytes", 65536),
                          ("max-total-read-bytes", 1048576)]:
        inv.add_argument("--" + name, type=positive, default=default)
    win = commands.add_parser("window")
    win.add_argument("source")
    win.add_argument("--start-us", required=True)
    win.add_argument("--end-us", required=True)
    win.add_argument("--device-id")
    for name, default in [("max-bytes", 8388608), ("max-rows", 100000), ("top", 10)]:
        win.add_argument("--" + name, type=positive, default=default)
    try:
        args = vars(parser.parse_args())
        command = args.pop("command")
        output = (inventory if command == "inventory" else window)(**args)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
