<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Launch and training failure diagnosis

Read the first relevant error and enough surrounding launcher/rank context. Preserve the failed attempt's files and job identity. Inspect implementation only when documented commands, CLI help and the observed error cannot resolve the issue.

## Choose the relevant check

| Symptom | Next check |
| --- | --- |
| CLI rejects an argument | `flagscale train --help`; correct that specific installed-version mismatch |
| `ModuleNotFoundError` or wrong module | Active parent/worker environment, actual module paths and dependencies; fix a demonstrated missing dependency rather than reinstalling a working stack |
| Runtime or collective error | First error on affected ranks, device mapping/connectivity and the selected hardware reference |
| OOM / allocation failure | Failed rank/stage, memory and owned processes; return evidence to the tuning caller without changing its fixed workload |
| Missing data or loader failure | Config → registry → metadata → actual files; test the real loader with a few samples |
| Address already in use | Port owner and exact job/PID; stop only an owned job or choose an allowed free port |
| Hydra/config error | Selected full recipe, value types and generated arguments; dryrun can isolate generation failures |
| Early or apparently silent exit | Launcher output and rank stderr, including nonzero ranks; outer exit code may mask worker failure |
| Unexpected pretrained-model loss | Actual checkpoint load/missing-key messages, requested load scope and input data; loss alone is not load proof |

## Verify and retry

Use the smallest check that reaches the failed component: runtime import for path errors, actual dataset samples for loader errors, dryrun for generated config, or the supported model construction path for shape issues. Distributed or stateful failures may require a bounded training run. Passing an isolated check narrows the fault; it does not replace eventual training validation.

After fixing the diagnosed issue, use a new output directory, verify the intended YAML/arguments and confirm prior owned workers have exited. Do not repeat the same failing launch without a new diagnosis or changed condition. Never remove experiment history, reset devices or use broad process-name kills as recovery.

For a CLI with legacy `run.py` only, preserve an already confirmed working command or inspect its help for the observed incompatibility. Do not switch away from `flagscale train -c ...` merely to revalidate a normal launch.
