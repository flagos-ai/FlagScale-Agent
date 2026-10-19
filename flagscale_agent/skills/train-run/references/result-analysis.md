<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Analyze Training Results

Use the [analysis script](../scripts/analyze_training_results.py) for Megatron logs with training `log_interval=1`. Supply the exact loss-reporting rank log from each attempt, not a directory, concatenated rank logs, or a guessed timestamp. Keep one fixed workload and one fixed candidate in each comparison request. For other log formats, state the unsupported format and use a compatible analysis script with explicit measurement rules.

Save a JSON request using the actual paths, iteration window, and agreed tolerances. For example:

```json
{
  "runs": [
    {"run_id": "baseline-01", "role": "baseline", "log_path": "/path/to/baseline/stdout.log", "exit_code_path": "/path/to/baseline/launcher.exit_code"},
    {"run_id": "candidate-01", "role": "candidate", "log_path": "/path/to/candidate/stdout.log", "exit_code_path": "/path/to/candidate/launcher.exit_code"}
  ],
  "first_iteration": 1,
  "end_iteration": 40,
  "warmup_steps": 10,
  "global_batch_size": 64,
  "sequence_length": 512,
  "loss_atol": 0.00001,
  "output_path": "/path/to/results.json"
}
```

`TRAIN_RUN_SKILL_DIR` is the directory containing the loaded `train-run/SKILL.md`. Run:

```bash
python "$TRAIN_RUN_SKILL_DIR/scripts/analyze_training_results.py" \
  --request /absolute/path/analysis-request.json
```

The example values are illustrative; use the current workload and predeclared tolerances. `runs`, `end_iteration`, `global_batch_size`, and `sequence_length` are required. `first_iteration` defaults to 1 and `warmup_steps` to 10; warmup is excluded from timing but included in loss checks. Optional `loss_rtol` combines with `loss_atol` as `abs(candidate - baseline) <= atol + rtol * abs(baseline)`. Omit `exit_code_path` when genuine exit evidence is unavailable.

`output_path` must have an existing parent directory. The script saves the complete JSON, including evidence paths and hashes, and prints a concise summary. Add `--detail full` only for a diagnostic that needs the complete stdout report. Exit codes are `0` for parsed valid evidence, `1` for invalid evidence, and `2` for a request or file error; zero does not mean acceptance. Do not relaunch training just to repair an analysis request.

For repeated comparisons, supply attempts in actual chronological order as adjacent B/C or C/B pairs, and pass the agreed `max_run_variation_pct`. Each candidate attempt uses the same candidate configuration. One old baseline plus repeated candidates does not establish repeatability. The analyzer requires at least two pairs to assess a declared spread limit; use the calling workflow's agreed repeat count. It does not conduct a statistical significance test.

The default summary includes per-run timing, launcher status, skip/NaN counters, evidence errors, comparison metrics and unchanged acceptance checks. In `comparison.performance`, `median_run_mean_step_time_ms`, `tokens_per_second`, and `observed_speedup` use the same aggregation; copy these values directly into the final report. Do not substitute averages of individual throughputs or recalculate throughput by hand. `run_variation_pct` is the measured range divided by the median of per-run mean times. Missing evidence stays unknown and `status="ok"` does not mean acceptance.

The complete report has these additional evidence boundaries:

| Field | What It Establishes |
| --- | --- |
| Top-level `status` | Parsing/evidence validity, not acceptance |
| `runs[].measurement` | Logged iteration coverage, timing window, counters and fixed-length token throughput |
| `runs[].completion` | Iteration coverage and optional launcher exit evidence; all-rank completion remains separate |
| `comparison.performance` | Observed speedup, `time_reduction_pct` and `throughput_improvement_pct`, with different meanings |
| `comparison.loss_checks` | Actual aligned loss differences over all requested iterations, including warmup; tolerance status if declared |
| `comparison.repeatability` | This comparison's run counts, pairing and measured spread against the declared limit |
| `comparison.acceptance_checks` | Separate checks to combine with workload equivalence, all-rank status and task-specific requirements |

Missing skip/NaN counters stay unknown. Omitted loss tolerances do not pass quality. A finite, close loss trajectory does not establish parameter equivalence or long-term convergence. No input to this script verifies configuration/seed/data equivalence; confirm those from the saved effective configurations and workload records. Use profiler evidence before making a bottleneck attribution; timing alone supports a performance observation and a hypothesis.

If the launcher captured its exit status, provide `exit_code_path` per attempt, pointing to the actual file containing its integer exit code. Do not create a zero exit file afterward based on step count or process absence. Launcher success still does not independently verify every remote worker.

When reporting mock-data measurements, state the model, fixed sequence length and synthetic-data scope. Real-data input throughput and long-term convergence require their own evidence.
