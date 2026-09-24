<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Current-run logs and health

For direct CLI runs, prefer `flagscale_train_monitor(output_dir="<exp_dir>", mode="check", vocab_size=<actual_vocab_size>)`. Omit `vocab_size` when unknown. The bounded single-run path already returns compact measurements and exit evidence; read extra logs only for a specific missing fact or anomaly.

## Locate this run

`exp_dir` is the effective `experiment.exp_dir`. Typical FlagScale layout:

```text
<exp_dir>/logs/
  host_0_<hostname>.output                 launcher output
  pids/host_0_<hostname>.pid               launcher PID
  scripts/host_0_<hostname>_run.sh         generated launch script
  details/host_0_<hostname>/<timestamp>/
    default_<hash>/attempt_0/<rank>/
      stdout.log
      stderr.log
```

Each launch can add a timestamp directory. Match job identity/start time to the recorded attempt; “latest” alone does not prove ownership. Prefer a unique directory per attempt, and locate the actual loss-reporting rank rather than assuming rank 0, especially with PP.

If the monitor tool is unavailable or omits needed evidence, inspect only the recorded run's launcher and rank logs. Scan stderr on affected ranks, including nonzero ranks; avoid workspace-wide log searches or logs from an older attempt.

The monitor reads paths in the Agent's environment and has no SSH/container arguments. For remote logs, use a shared path or a synchronized snapshot that preserves the directory structure and records its timestamp. If that is unavailable, record the access gap and inspect original logs through the established remote connection; inaccessible logs do not establish training failure.

## Interpret progress and metrics

- No logs on the first monitor call can be normal startup. Retain the job id and use bounded `shell_jobs` waits; inspect launcher output/rank stderr for a real startup error.
- Loss near `ln(vocab_size)` can be expected for scratch or mock-data training. When pretrained weights were requested, inspect actual load evidence and data before concluding that loading failed.
- Zero gradient norm or unexpectedly all-zero gradients warrants checking frozen parameters, loss/backward and updates against the baseline; it is a diagnostic clue, not an automatic model-independent verdict.
- Judge loss/gradient health and skipped updates against the caller's workload. A finite short-run loss does not prove update equivalence or convergence.
- Check `error_ranks` even when the monitor reports `health_ok`; numerical health does not rule out rank errors.
- Use the hardware reference's occupancy probes. `mode="watch"` is appropriate only when those probes are compatible with the selected chip.

## Completion and interrupted observation

Expected progress, current-run error checks and owned-worker exit evidence jointly determine completion. One rank's loss or outer CLI exit cannot prove all workers succeeded. Preserve the exact logs and unresolved checks in the existing record.

After interrupted observation or an Agent restart, reconcile the recorded host/container, owned PID and current-run logs before waiting again. Native shell job IDs are process-local; an old ID may refer to a different job after restart. A timed-out wait does not stop training; use the main skill's stop procedure when needed.
