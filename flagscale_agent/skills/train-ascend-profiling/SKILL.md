---
name: train-ascend-profiling
description: >-
  Generate a torch_npu profiler wrapper, collect a profile from FlagScale
  training on Ascend, or analyze existing NPU performance data. Deliver
  a bottleneck report backed by raw evidence and propose the next experiment.
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Collect and Analyze Ascend Training Profiles

Reuse the current recipe, run records, and resource allocation. First identify the task: generate a wrapper, collect a profile, or analyze existing artifacts. Run only the corresponding steps. When invoked during tuning, use the main plan; do not create a separate tuning loop.

## 1. Generate the wrapper

Follow the [wrapper instructions](references/wrapper-generation.md) using the original training entrypoint from the target environment. `SKILL_DIR` is this skill's directory.
For a generation-only task, deliver the wrapper, recipe change, and command, then stop. If the original entrypoint is inaccessible, deliver the command to run in the target environment and state that the file has not been generated.

## 2. Collect

Prepare a separate collection recipe using the wrapper instructions; preserve the workload and distributed strategy.
Load `train-run` if not already loaded and launch through it. For a supported single-node short run, follow its [bounded execution instructions](../train-run/references/single-run.md). Record `stage=profile`, the configuration, global ranks, window, logs, and exit evidence in the same experiment record.

Training continues after the wrapper completes the collection window; the recipe's iteration count and launcher time limit must end the run. Confirm that all workers from this run have exited. On failure, return the original error and artifacts so the caller can decide whether to retry.

## 3. Inventory and validate

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" inventory "$PROFILE_DIR" > "$RUN_DIR/profile-inventory.json"
```

Check the summary and read limits. If inventory is truncated, inspect only the relevant subdirectories next. For a newly collected profile, verify `wrapper-status.json` for each selected rank, the final entrypoint status, and real NPU events covering the target update with valid timestamps and device mapping. A directory or returned callback alone does not prove a complete collection. For existing artifacts, use their actual metadata; do not invent wrapper status.

`inventory` identifies files and CSV headers; it does not convert a DB, raw PROF directory, or trace. Direct window analysis accepts per-task CSV columns `Start Time(us)`/`Duration(us)` or `Task Start Time(us)`/`Task Duration(us)`, plus a nonempty device column. For other formats, reuse an export method already validated in the environment. If none exists, report the format gap instead of improvising a command.

For a collection-only task, deliver the raw artifacts, commands, and validated scope, then stop.

## 4. Analyze the target window

Get the window on the same device from actual step or time markers. Replace these example values:

```bash
python "$SKILL_DIR/scripts/profile_inspect.py" window "$KERNEL_CSV" \
  --start-us 1000000 --end-us 1100000 --device-id 0 > "$RUN_DIR/profile-window.json"
```

By default the tool reads at most 8 MiB or 100,000 rows. For larger inputs, explicitly set `--max-bytes` and `--max-rows` within the resource budget, or use an existing export of the complete window. Do not read only the start of a CSV and claim to cover the entire target window.
Check the JSON `status`, device range, malformed rows, and truncation; then compare the key differences against the raw tasks. `uncovered` means no recorded task covers that time, and cumulative operator duration is not a measure of critical-path contribution.

For each finding, record only the observation, possible cause, missing evidence, and smallest validation experiment. If statistics or clock relationships are unclear, read the relevant part of `ascend_profiling/collection-and-analysis.md` in `know-ascend-profiling`. For training metric definitions, read `ascend_training/measurement-and-records.md` in `know-ascend-training`.

Deliver raw paths, the window JSON, observations, and hypotheses to verify. Return to the main tuning workflow to select candidates. Do not include profiled runs in profiler-free performance rankings. Keep partial artifacts and unknown causes within their actual evidence scope.
