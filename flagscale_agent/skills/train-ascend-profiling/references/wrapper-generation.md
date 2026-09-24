<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Generate a torch_npu Profiler Training Entrypoint

Use this workflow when the original training entrypoint already runs and an independent NPU profile is needed. It applies to FlagScale Megatron entrypoints that run through `megatron.training.training.train/train_step`, including Qwen3.5. The implementation consists of the [generator](../scripts/generate_profile_wrapper.py) and a [standalone template](../assets/npu_profile_wrapper.py). The generator needs only the Python standard library and does not launch training. The generated wrapper runs with the target environment's existing torch, torch_npu, and three-repository dependencies.

## Generate

In the target container, reuse the original job's verified cwd, Python, and module paths. The FlagScale launcher may add its own `flagscale/train` to `PYTHONPATH`, so `megatron.training.training` may not come from the Megatron-LM-FL repository. Check the actual module file only if its binding is unverified or an error occurs; do not repeat an investigation already captured in the run record. `--training-file` pins a verified path as a runtime guard. Do not guess that path from the repository name.

Set `TRAIN_ENTRYPOINT` to the existing original training file on the target machine and `RUN_DIR` to a separate attempt directory. `SKILL_DIR` is this skill's directory; the generator needs the adjacent assets template.

```bash
python "$SKILL_DIR/scripts/generate_profile_wrapper.py" \
  --entrypoint "$TRAIN_ENTRYPOINT" --output "$RUN_DIR/train_npu_profile.py" \
  --profile-output "$RUN_DIR/npu-profile" \
  --wait 3 --warmup 1 --active 1 --ranks 0 --level Level1
```

Add `--training-file "$VERIFIED_TRAINING_FILE"` only to guard a verified module binding. If the launcher does not supply required import roots, pass repeatable `--python-path` options in the original job's search order.

The generator returns JSON with the wrapper path, full configuration, and `training_started: false`. It refuses to overwrite an existing wrapper or the original entrypoint. The generated `.py` is self-contained and does not require installing this skill. Its original entrypoint, dependencies, and output paths refer to absolute paths on the target machine. On multiple nodes, ensure every worker can access the same wrapper and corresponding paths. You may move the generated wrapper, but if directories differ across machines, regenerate its configuration in the target environment.

`--ranks 0,7` and `--ranks all` are supported. These are **global ranks**, not physical NPU IDs. The default is rank 0 only. Select representative ranks based on evidence when diagnosing PP, EP, or a slow rank. The original launcher still manages device assignment, visible devices, and world size; the generator neither selects nor reserves devices. Enable `--record-shapes`, `--with-stack`, or `--profile-memory` only to address an evidence gap; all are off by default. The default profiler level is Level1. Check the installed torch_npu version's supported levels, export formats, and overhead without automatically upgrading dependencies.

## Configure the collection recipe

Copy the original recipe for this profiling attempt and replace its entrypoint. The current FlagScale runner reads:

```yaml
experiment:
  task:
    entrypoint: /absolute/run-dir/train_npu_profile.py
```

This is a configuration fragment. Merge it into the existing recipe and retain the other task fields. Disable the built-in profiler under the current schema: verify that the final `args.profile` mapped from `use_nsys_profiler` is false, `use_pytorch_profiler` is false, and `pytorch_profiler_collect_chakra` is false. YAML nesting may vary by version, so inspect the resolved configuration and final argv; do not append `--profile`. The wrapper rejects built-in `args.profile`, Chakra requests, and `skip_train` rather than silently changing them.

The wrapper executes the original entrypoint once through `runpy.run_path(..., run_name="__main__")`, retaining its arguments, providers, and callbacks without copying `pretrain(...)`. Missing bindings, replaced hooks, or training that never starts cause an error. Other training loops, custom `train_step` lookup, or multiple `train` calls per process require adaptation first.

## Window and return count

The profiler starts after training initialization, on entry into `train`, with `repeat=1`. After each **normal return** from the original `train_step`, it calls `prof.step()` once. It preserves the original return value and does not alter rerun or skip logic. With the default wait=3, warmup=1, active=1, returns 1–3 wait, return 4 warms up, and return 5 is active for this `train` call. At least five normal returns are required to complete the default window. Choose `wait` based on compilation, initialization, and training stability before actual collection; the default does not guarantee a stable interval.

`train_step` typically includes all microbatches, backward passes, optimizer work, and learning-rate updates. However, overflow, reruns, or exit conditions can make a normal return different from a successful optimizer update. The count is relative to this `train` call, not the absolute iteration after resume. The `iteration_argument` status field records only an integer `iteration` keyword actually passed to the call; it is null if absent, and no iteration number is inferred. Verify complete update boundaries against the training implementation, logs, and trace.

The profiler wraps all of `train`, so intervals between adjacent return boundaries may include outer logging, evaluation, checkpoint saves, or scheduling. The template labels the function body during collection as `ascend_train_step/N`; N is the relative call number above. Do not interpret a whole profiler step as pure compute time, or use profiled timings to rank training performance.

The profiler stops immediately after wait+warmup+active while training continues. Exception paths also clean up and remove the wrapper's hooks. `SystemExit(0/None)` keeps normal exit semantics; an unfinished window is still marked incomplete. Nonzero exit, KeyboardInterrupt, or a training error is marked failed. A save or evaluation failure after `train` also updates entrypoint status. A cleanup failure does not hide the original training error. SIGKILL, process crashes, and power loss cannot guarantee `finally` or flush; a residual running/pending state must be treated as unverified.

## Output and status meanings

Each selected rank gets a new directory such as `$RUN_DIR/npu-profile/rank-00000/`. The wrapper refuses to reuse an existing one. `trace/` is passed to `tensorboard_trace_handler`; the installed version determines the subdirectories, CSV, JSON, or DB files. `wrapper-status.json` records the actual module and function files, rank/world size, schedule, return count, callback, and entrypoint status.

| Status or evidence | Meaning |
| --- | --- |
| `collecting` / `window_complete_pending_training_end` | Still running or not yet cleanly finished; do not report collection success |
| `incomplete` | Too few returns or no export callback; even exit code 0 does not complete the collection |
| `failed` | Training, entrypoint, or collection cleanup failed; partial files may help diagnosis |
| `window_complete_needs_artifact_validation` | Scheduled window finished and callback returned; real NPU events still need validation |
| `entrypoint_status: completed` | The full original entrypoint returned normally or exited with code 0; this alone does not prove a complete collection |

Return to the [profiling workflow](../SKILL.md) for launch, artifact validation, and analysis. Empty files, aggregate-only summaries, or a missed target window cannot establish complete task-level coverage.
