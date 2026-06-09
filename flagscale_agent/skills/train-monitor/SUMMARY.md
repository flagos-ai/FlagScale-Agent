# Train-Monitor — Summary

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies, and parse metrics.

**Load when**: monitoring a running training job, diagnosing training anomalies (NaN loss, OOM, hangs), or needing to find/parse training logs.

Key rule: always use `monitor(output_dir=...)` as primary method — it auto-discovers latest logs and scans stderr. Never use raw find commands (they find old logs). Check stderr first, not stdout.
