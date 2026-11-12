---
name: train-ascend-profiling
description: >-
  Generate a torch_npu profiler wrapper, collect a profile from FlagScale
  training on Ascend, or analyze existing NPU performance data. Also computes
  modelled FLOPs/throughput/MFU, ranks operators and checks a Roofline bound,
  and compares per-rank step times. Deliver a bottleneck report backed by raw
  evidence and propose the next experiment.
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

## 5. Metrics, operator ranking, and slow-rank comparison

Run only the branch that matches the question; each writes JSON with its own
limitations and none of them decides what to optimise.

**How much of peak did the step reach?** Follow [performance metrics](references/performance-metrics.md). FLOPs are modelled and MFU needs a peak for the exact device and precision; the tool withholds MFU without one and flags an implausible result rather than reporting it.

```bash
python "$SKILL_DIR/scripts/mfu_calculator.py" mfu --config "$MODEL_CONFIG_JSON" \
  --seq-length 8192 --global-batch-size 32 --num-devices 16 --step-time 4.4 \
  --peak-flops <device FP16/BF16 peak FLOPS> --peak-source "<vendor spec, SKU, dtype>"
python "$SKILL_DIR/scripts/mfu_calculator.py" self-test   # cross-check the FLOPs model
```

**Which operators dominate?** Read [operator and rank analysis](references/operator-and-rank-analysis.md) first. Cumulative duration ranks but does not attribute; confirm any candidate inside a window from section 4.

```bash
python "$SKILL_DIR/scripts/op_analysis.py" hot-ops "$PROFILE_DIR" --top 5
python "$SKILL_DIR/scripts/op_analysis.py" pivot   "$PROFILE_DIR" --op MatMul
python "$SKILL_DIR/scripts/op_analysis.py" shapes  "$PROFILE_DIR" --op matmul
python "$SKILL_DIR/scripts/op_analysis.py" roofline "$KERNEL_CSV" \
  --peak-flops <TFLOPS> --bandwidth <GB/s>
```

Roofline needs the device's FP16/BF16 cube peak and HBM bandwidth as inputs; do not substitute a remembered number for a different SKU or precision. A low `efficiency` is a prompt to investigate, and above 1.0 means the model is wrong — operators whose shape is not a two-operand matrix are reported as `unmodelled`, never estimated.

**Is one rank slower?** Use the same reference. A peer comparison cannot see a slowdown that is uniform across all ranks, so `no candidates` is not proven balance.

```bash
python "$SKILL_DIR/scripts/slow_rank.py" logs "$RANK_LOG_DIR" --min-delta-pct 10
```

Report the metric definition, the inputs, the peak/bandwidth source, and every "limitations" entry that applies. Do not turn a ranked observation or a flagged rank into a bottleneck verdict without the window evidence from section 4.
