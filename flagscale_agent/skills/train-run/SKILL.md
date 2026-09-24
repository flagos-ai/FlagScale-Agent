---
description: Launch, monitor, stop, and verify FlagScale training from a YAML recipe.
  Use a shared execution workflow with device-specific references for hardware
  checks, runtime dependencies, and diagnostics.
name: train-run
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# FlagScale Training Launch

Launch, observe, stop and verify a training job from the supplied recipe. Reuse the caller's environment, workload, plan and experiment record; return execution evidence without starting a separate tuning or profiling workflow.

## 1. Reuse the setup and select hardware tools

- Work in the confirmed server/container, Python environment and training cwd. If the environment is already set up, do not install packages; investigate only demonstrated missing dependencies.
- Read the [device index](references/devices/index.md) and only the matching reference. Reuse verified device mapping and runtime facts across candidates; recheck assigned-device availability before each launch and every assigned host for multi-node work.
- Availability combines authorization, utilization, memory baseline and process visibility. No visible container PID does not prove a device is free; a failed probe means unknown. Do not reset devices or terminate unrelated jobs.
- For a new setup, or changed environment/data/checkpoint facts, read the relevant [preflight checks](references/preflight.md). An unchanged validated recipe needs only its delta, batch/layout constraints, output path and remaining budget checked. Its scheduled baseline/candidate run supplies validation; no separate smoke run is required.

`load_skill` loads only this file. Read references relative to the active `train-run/SKILL.md`, including a user override if present, not relative to the training cwd. Use documented commands; inspect CLI/launcher/helper source only for an unresolved error or contradictory behavior.

## 2. Choose one launch path

**Direct FlagScale launch** — preserve a full YAML containing `experiment` and `train`; its filename is arbitrary:

```bash
PYTHONUNBUFFERED=1 flagscale train -c /absolute/path/recipe.yaml --test
```

| Parameter or setting | Purpose |
| --- | --- |
| `-c <yaml>` / `--config <yaml>` | Select the full FlagScale recipe |
| `--test` | Run real training in the foreground; it does not reduce iterations |
| `train.model.train_iters` | Set the agreed iteration count in the recipe |
| `experiment.runner.nproc_per_node` | Match workers per node to the assigned devices |
| `experiment.exp_dir` | Select this attempt's output directory |

If the installed CLI requires MODEL, add its confirmed name after `train`. For rejected arguments, check `flagscale train --help`. Use `--dryrun` instead of `--test` only when generated scripts need inspection; it does not run training. The bounded path requires a fresh output directory, so use a separate directory for any dryrun.

**Bounded single-host Megatron measurements** — read [single-run execution](references/single-run.md) once, use its request/launch/wait procedure, and reuse it for subsequent trials. It calls the FlagScale CLI, enforces a timeout and returns compact evidence. Do not also run the direct-launch monitoring loop for the same job; inspect extra logs only for a missing fact or anomaly.

Use a unique output directory for each attempt. Preserve config-relative dependencies, checkpoint identity and the caller's fixed workload. Reuse one record containing the config delta, command/cwd, actual output directory, assigned devices/backend, time limit and job identity.

## 3. Launch, wait and verify

1. Confirm the preceding owned job has ended and assigned devices remain available.
2. Use `shell(background=True)` and retain its job id. For a direct CLI launch, call `flagscale_train_monitor(output_dir="<exp_dir>", mode="check")` before waiting, then `shell_jobs(action="wait", job_id=..., timeout=60)`. For the bounded path, follow its returned result and wait procedure.
3. A monitor response with no logs yet is not failure. Wait within the run's budget; inspect again for progress, completion or an error. Use `mode="watch"` only when the selected device reference confirms its probes are compatible. Log locations and metric interpretation are in [monitoring](references/monitoring.md), read when needed.
4. Verify expected iterations/progress, loss and gradient health, current-run errors and owned-worker completion. CLI exit 0, loss from one rank, or absence of a crash alone is not success; some launchers return before workers or mask their failures.
5. Append the terminal result once, then return command, output directory, job/exit evidence, exact loss-rank log and any unresolved error to the caller. Missing evidence remains unverified; do not rerun training just to retrieve known results.

For measurement or comparison of completed Megatron runs, follow [result analysis](references/result-analysis.md) with the recorded logs, exit evidence, and the caller's measurement contract.

## 4. Stop or handle a failure

Use `flagscale train -c <recipe.yaml> --stop` only after verifying that its PID/job belongs to this experiment. A missing PID file is not permission for a broad fallback kill. Retain all run artifacts, and confirm owned workers have exited before reusing devices; wait timeout alone does not stop training.

For an actual failure, read [troubleshooting](references/troubleshooting.md), diagnose the first relevant error and verify the fix before a new attempt. OOM and quality failures return to the tuning caller with evidence; this execution skill does not change its search policy or workload.

For third-party reproduction explicitly using a native training script, preserve that launcher and apply the same hardware, job tracking and evidence rules. For FlagScale recipes, use the CLI above.
