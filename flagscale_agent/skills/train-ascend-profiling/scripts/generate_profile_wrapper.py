#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate a standalone NPU profiler entrypoint; never starts training."""

import argparse
import ast
import json
from pathlib import Path


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def ranks(value):
    if value.lower() == "all":
        return None
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use all or comma-separated global rank IDs") from exc
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("rank IDs must be unique nonnegative integers")
    return values


def generate(args):
    entrypoint = Path(args.entrypoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not entrypoint.is_file():
        raise ValueError("--entrypoint must name the original existing training file on this machine")
    if output == entrypoint:
        raise ValueError("Wrapper output must not replace the original entrypoint")
    if args.active < 1:
        raise ValueError("--active must be positive")
    config = {
        "entrypoint": str(entrypoint),
        "profile_output": str(Path(args.profile_output).expanduser().resolve()),
        "training_module": args.training_module,
        "expected_training_file": str(Path(args.training_file).expanduser().resolve()) if args.training_file else None,
        "python_paths": [str(Path(p).expanduser().resolve()) for p in args.python_path],
        "wait": args.wait, "warmup": args.warmup, "active": args.active,
        "ranks": args.ranks, "level": args.level,
        "record_shapes": args.record_shapes, "with_stack": args.with_stack,
        "profile_memory": args.profile_memory,
    }
    template = Path(__file__).resolve().parents[1] / "assets/npu_profile_wrapper.py"
    marker = "CONFIG = None  # populated by generator"
    source = template.read_text(encoding="utf-8")
    if source.count(marker) != 1:
        raise ValueError("Wrapper template config marker is missing or ambiguous")
    # repr is a Python literal, not shell text or executable user interpolation.
    source = source.replace(marker, "CONFIG = " + repr(config))
    ast.parse(source, filename=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(source)
    return {"wrapper": str(output), "config": config, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--training-module", default="megatron.training.training")
    parser.add_argument("--training-file", help="Expected actual training.py; fail if another version is imported")
    parser.add_argument("--python-path", action="append", default=[], help="Prepend a verified module root; repeat as needed")
    parser.add_argument("--wait", type=nonnegative, default=3)
    parser.add_argument("--warmup", type=nonnegative, default=1)
    parser.add_argument("--active", type=nonnegative, default=1)
    parser.add_argument("--ranks", type=ranks, default=[0])
    parser.add_argument("--level", choices=["Level0", "Level1", "Level2"], default="Level1")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(generate(args), ensure_ascii=False))
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
