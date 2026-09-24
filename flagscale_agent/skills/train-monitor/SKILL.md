---
description: Monitor FlagScale distributed training jobs. Check training health,
  detect anomalies (NaN loss, OOM, NCCL timeout, hangs), parse training metrics
  (loss, grad norm, throughput), and provide periodic status reports. Supports
  single-node and multi-node monitoring.
name: train-monitor
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# FlagScale Training Monitor

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies, and report metrics.

## Critical Rules

1. **Use `flagscale_train_monitor(output_dir=...)` as the primary monitoring method.**
   It auto-discovers the latest logs across all hosts, scans stderr for errors, finds
   the loss rank, and reports metrics — all in one call. NEVER use raw `find` to locate
   logs (they find old runs from previous launches).
   - `mode="check"`: one-shot inspection of current log state (use for spot checks).
   - `mode="watch"`: continuous polling until event or `duration` (max 1800s) elapses
     (use for supervised waits).
   - `filter="errors"|"progress"` narrows check-mode output; `lines` sets tail depth.
2. **Check stderr FIRST, not stdout.** Crash information is in stderr. A process showing
   "wandb initialized" or "loading model" in stdout may already be dead. The monitor
   tool scans stderr automatically.
3. **Old log trap**: multiple runs accumulate under the same experiment dir — always
   resolve the LATEST timestamp dir. The monitor tool handles this; if you must go
   manual, sort timestamp dirs and take the last.

## Locating the Experiment Directory

The experiment output dir is whatever you passed as `output_dir` at launch (recorded in
the experiment ledger memory entry per the ops-discipline HARD GATE). If not recorded,
read it from the task config: `<exp_dir>` is the value of the `output_dir` field in
`train.yaml`, or `<exp_dir>/outputs` for FlagScale-launched runs.

## Log Directory Structure (reference only — prefer the monitor tool)

```
<exp_dir>/
├── logs/
│   ├── host_<N>_<hostname>.output              # torchrun launcher output
│   ├── pids/host_<N>_<hostname>.pid            # launcher PID
│   └── details/host_<N>_<hostname>/
│       └── <timestamp>/<run_id>/attempt_<N>/<local_rank>/
│           ├── stdout.log                      # training metrics, progress
│           └── stderr.log                      # errors, warnings, stack traces
├── checkpoints/
└── tensorboard/
```

Key facts:
- Each training launch creates a NEW timestamp directory — find the LATEST one
- Training metrics (loss, iteration) are printed by the **last pipeline rank**
- Errors can appear on **any rank**'s stderr

## Step 1: Locate the Latest Logs

Preferred — one call:

```
flagscale_train_monitor(output_dir=<exp_dir>, mode="check")
```

Manual fallback (when the tool is unavailable):

```bash
EXP_DIR=<exp_dir>
# Latest timestamp dir under host_0 (all hosts share the timestamp)
LATEST=$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ | sort | tail -1)
# Latest attempt inside it
ATTEMPT=$(ls -d ${LATEST}*/attempt_[0-9]* 2>/dev/null | sort | tail -1)
LAST_RANK=$(( $(nproc) - 1 ))   # loss rank = last local rank
tail -1 ${ATTEMPT}/${LAST_RANK}/stdout.log
```

## Step 2: Health Check

Quick check — one stdout line + GPU state:

```bash
tail -1 ${ATTEMPT}/${LAST_RANK}/stdout.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Process liveness (launcher PID per host):

```bash
for f in ${EXP_DIR}/logs/pids/*.pid; do
  PID=$(cat $f); ps -p ${PID} > /dev/null && echo "ALIVE ${PID}" || echo "DEAD ${PID}"
done
```

Loss sanity — the four gates (checked in order):

| Gate | Reading | Meaning |
|------|---------|---------|
| 1 | Initial loss ≈ ln(vocab_size): 32000→10.37, 38016→10.55, 128256→11.76 | Still within ~10% after >10 iterations = fundamentally wrong (data or masking bug). Dummy data trains like random tokens; real data must dive quickly. |
| 2 | `number of zeros` in gradient histogram > 90% | Gradient flow broken (stale/detached params) — inspect grads. |
| 3 | `params norm` identical across iterations | Params frozen — check optimizer wiring. |
| 4 | Loss trend flat or rising after warmup | Investigate immediately (LR, data order, NaN). |

A clean run: initial loss near ln(vocab), then smooth monotone descent, grad norm
decaying. If iter-1 loss looks like a checkpoint-continued value instead of the
cold-start value, verify checkpoint-load evidence (timers + resume lines) before
judging the conversion.

## Step 3: Anomaly Detection

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| `CUDA out of memory` in any stderr | TP/PP/DP memory budget too tight | Check activation checkpointing / micro-batch size. |
| `NCCL timeout` / `watchdog` in stderr | A rank stalled (straggler or deadlock) | Find which rank printed last; check its node. |
| Loss becomes NaN | Numerical instability (fp16 overflow, bad data) | Check grad-norm history; inspect first NaN iter's inputs. |
| stderr silent + stdout stalls | Hung collectives (mismatched world size) | Compare `nproc` vs launched ranks. |
| Process alive, GPU util 0% | Deadlocked training loop | py-spy dump or kill + relaunch. |
| Repeated restarts in multiple attempts | flaky node or OOM loop | Read the earliest stderr in each attempt. |
| `TorchDynamo` / `TE` errors in stderr | Plugin/version mismatch | Compare against env-setup records. |

Scan ALL ranks' stderr (errors rarely land on rank 0):

```bash
for d in ${EXP_DIR}/logs/details/host_*/${TS}/${RUN}/attempt_0; do
  grep -l -iE "error|nan|out of memory|timeout" ${d}/*/stderr.log
done
```

Multi-node (SSH loop, stdout and stderr both):

```bash
for h in host_1 host_2; do
  ssh ${h} "grep -iE 'error|nan|oom|timeout' ${EXP_DIR}/logs/details/${h}_*/${TS}/${RUN}/attempt_0/*/stderr.log | tail -5"
done
```

## Step 4: Parse Metrics

Preferred: `flagscale_train_monitor(output_dir=<exp_dir>, mode="check", filter="progress")`.

Manual (stdout line layout — Megatron-style logs):

```bash
grep -oE "iteration +[0-9]+/" ${ATTEMPT}/${LAST_RANK}/stdout.log | tail -1   # progress
grep "lm loss" ${ATTEMPT}/${LAST_RANK}/stdout.log | tail -3                  # loss trend
grep -oE "grad norm: [0-9.]+" ${ATTEMPT}/${LAST_RANK}/stdout.log | tail -3   # stability
grep "elapsed time per iteration" ${ATTEMPT}/${LAST_RANK}/stdout.log | tail -1  # throughput
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv              # hardware
```

Multi-node metrics (SSH per host, last rank only):

```bash
ssh host_2 "tail -1 ${EXP_DIR}/logs/details/host_2_*/${TS}/${RUN}/attempt_0/7/stdout.log"
```

Alert conditions (report immediately, do not wait for the next poll):
- NaN in loss, or grad norm > 10x its recent median
- throughput drop > 30% vs the run's median
- any stderr error line appearing in the last N minutes
- GPU memory within 2 GB of capacity (OOM incoming)

## Checkpoints and TensorBoard

- Checkpoint cadence and content: see the train-config skill (`checkpoint` section).
- TensorBoard event files live under `<exp_dir>/tensorboard/` — if the tool surface
  lacks a reader, report the path and let the user open it.

## Common Issues

| Issue | Fix |
|-------|-----|
| Monitoring an old run | Resolve the LATEST timestamp dir (sort + tail -1). |
| Empty stdout for a rank | Rank may not have started; check that rank's stderr first. |
| Different loss across ranks | Expected under PP (only last stage prints) — compare same stage across attempts. |
| Monitor tool finds nothing | `output_dir` typo, or logs not yet flushed; fall back to manual tail. |

## Related Skills

- `train-run`: launch/stop jobs and preflight validation (pairs with this skill).
- `ops-discipline`: experiment ledger HARD GATE and root-cause discipline.
- `train-config`: parallelism/precision knobs that the anomalies above point at.
