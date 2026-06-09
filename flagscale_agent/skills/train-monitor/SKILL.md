---
name: train-monitor
description: Monitor FlagScale distributed training jobs. Locate logs, check training health, detect anomalies (NaN loss,
  OOM, NCCL timeout, hangs), parse training metrics (loss, grad norm, throughput), and provide periodic status reports. Supports
  single-node and multi-node monitoring.
keywords:
- monitor
- monitoring
- loss
- log
- logs
- status
- check
- anomaly
- OOM
- nan
- hang
- grad norm
- throughput
- 监控
- 日志
- 训练状态
- 训练监控
- 查看loss
parameters:
- name: exp_dir
  description: Experiment output directory (from train.yaml experiment.exp_dir)
- name: nproc_per_node
  description: Number of GPUs per node
  default: '8'
requires:
- train-run
suggests: []
constraints:
- id: use_monitor_tool_not_find
  description: Always use monitor(output_dir=...) instead of raw find commands to locate logs
  trigger:
    tools:
    - shell
    keywords:
    - find
    - locate
    - stdout.log
    - stderr.log
  prompt: Check if the agent is using raw find/locate commands to find training logs instead of the monitor tool
  correction: Use monitor(output_dir=<exp_dir>) which auto-discovers latest logs and scans stderr.
- id: check_stderr_first
  description: Always check stderr before stdout — crashes are in stderr
  trigger:
    tools:
    - shell
    keywords:
    - stdout.log
    - stdout_worker
    - .output
  prompt: Check if the agent is reading stdout.log without first checking stderr.log for errors
  correction: Check stderr.log FIRST. A process may look running in stdout while already crashed.
- id: old_log_trap
  description: Warn about finding old logs from previous runs
  trigger:
    keywords:
    - find -name
    - stdout.log
    - stderr.log
    - logs/details
  prompt: Check if log file discovery might return old logs from previous runs
  correction: 'Old log trap: find without timestamp filtering returns ALL previous runs. Use monitor tool or filter by latest
    timestamp dir.'
context_injection:
  always:
  - Critical Rules
  - Log Directory Structure
  by_tool:
    shell:
    - Locate Latest Logs
    - 'Multi-Node: Finding the Loss Log'
    monitor:
    - Critical Rules
---
# FlagScale Training Monitor

Monitor running FlagScale training jobs: locate logs, check health, detect anomalies, and report metrics.

## Critical Rules

1. **Always use `monitor(output_dir=...)` as the primary monitoring method.** It auto-discovers the latest logs, scans stderr for errors, and reports metrics — all in one call. NEVER use raw `find` commands to locate logs (they find old runs from previous launches).

2. **Check stderr FIRST, not stdout.** Crash information is in stderr. A process showing "wandb initialized" or "loading model" in stdout may already be dead. The monitor tool checks stderr automatically.

3. **Old log trap**: `find ... -name "stdout.log"` without timestamp filtering will return logs from ALL previous runs. Always filter by the current run's timestamp or use the monitor tool which handles this automatically.

## Finding the Experiment Directory

Before searching the filesystem, check workspace_state's "Experiments" section — it contains a registry of all launched experiments with their directories. Use `workspace_state(action="read")` and look up the experiment dir from the table. This is the fastest and most reliable way to find logs.

If the registry is empty or the experiment isn't listed, fall back to the log directory structure below.

## Log Directory Structure

FlagScale training logs follow this structure:

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   └── details/host_0_<hostname>/
│       ├── 20260424_153816.588538/           # timestamp dir
│       │   └── default_<hash>/attempt_0/
│       │       ├── 0/stdout.log  stderr.log  # rank 0
│       │       ├── 1/stdout.log  stderr.log  # rank 1
│       │       └── .../
│       └── 20260424_162209.763893/           # newer run
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key facts:
- Each training launch creates a NEW timestamp directory
- Multiple runs accumulate — always find the LATEST timestamp dir
- Training metrics (loss, iteration) are printed by the **last rank**
- Errors can appear on **any rank**'s stderr

---

## Step 1: Locate Latest Logs

**Preferred: use the `find_latest_log` tool.** It auto-scans all ranks, finds the loss rank, and checks for errors.

```
find_latest_log(experiment="<exp_name>", vocab_size=<vocab_size>)
```

For deeper metric analysis, use `parse_training_metrics`:
```
parse_training_metrics(log_path="<stdout.log path>", vocab_size=<vocab_size>)
```

**Manual fallback** (if tools unavailable):

```bash
EXP_DIR=<exp_dir from train.yaml>
LATEST=$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" 2>/dev/null | head -1)
NPROC={nproc_per_node}
LAST_RANK=$((NPROC - 1))
```

Verify the path is valid:
```bash
echo "Latest log dir: $LATEST"
echo "Attempt dir: $ATTEMPT"
ls "$ATTEMPT/" 2>/dev/null | head -5
```

If `LATEST` or `ATTEMPT` is empty, training either hasn't started or logs are in an unexpected location. Check `EXP_DIR` value in `train.yaml`.

### Multi-Node: Finding the Loss Log

For multi-node training, the loss log is on the **last node's last rank**:

```bash
LAST_HOST_DIR=$(ls -d ${EXP_DIR}/logs/details/host_*/ 2>/dev/null | sort | tail -1)
LATEST=$(ls -d ${LAST_HOST_DIR}[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" 2>/dev/null | head -1)
tail -30 ${ATTEMPT}/${LAST_RANK}/stdout.log
```

---

## Step 2: Health Check

Training is normal when BOTH conditions are true:
1. Loss is being printed (new lines in stdout.log)
2. GPUs are active (utilization > 0%, memory occupied)

### Quick Health Check

```bash
# Latest loss line + GPU status
tail -1 ${ATTEMPT}/${LAST_RANK}/stdout.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

### Process Check

```bash
cat ${EXP_DIR}/logs/pids/*.pid 2>/dev/null | xargs -I{} ps -p {} -o pid,stat,etime --no-headers
```

If no processes found, training has stopped. Check stderr for the reason.

### Loss Sanity Check — CRITICAL

**This is the most important check. Do it EVERY time you look at training output.**

The `parse_training_metrics` and `find_latest_log` tools do this automatically. If checking manually:

1. **Random output check**: `ce_loss ≈ ln(vocab_size)` means the model is outputting random probabilities.
   - For vocab_size=32000: random loss ≈ 10.37
   - For vocab_size=38016: random loss ≈ 10.55
   - For vocab_size=128256: random loss ≈ 11.76
   - If loss is within 10% of ln(vocab_size) after >10 iterations, something is fundamentally wrong (weights not loaded, forward pass broken, attention mask not applied).

2. **Zero gradient check**: if `num_zeros` / total_params > 90%, gradients are not flowing. The model is not learning.

3. **Params norm check**: if `params_norm` doesn't change across iterations, weights are frozen or not being updated.

4. **Loss trend check**: loss should decrease over the first 50-100 iterations. If it's flat or increasing, investigate immediately.

**When any of these checks fail, STOP and diagnose before continuing.** Do NOT celebrate a "successful" training run that has random-output loss. Do NOT proceed to the next task. The training is broken.

### Dummy data vs real data

- Dummy/mock data: loss near ln(vocab_size) is EXPECTED. The data is random.
- Real data: loss near ln(vocab_size) is a CRITICAL BUG. Investigate immediately.
- Always know which data you're using. Check the config's `data_path` field.

### Training Progress

```bash
# Last 30 lines of training output (last rank)
tail -30 ${ATTEMPT}/${LAST_RANK}/stdout.log

# Errors on rank 0 (most common error location)
tail -30 ${ATTEMPT}/0/stderr.log

# Errors on last rank
tail -30 ${ATTEMPT}/${LAST_RANK}/stderr.log
```

---

## Step 3: Anomaly Detection

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| GPU memory occupied but 0% util, no new loss | Hanging / NCCL deadlock | Check stderr on all ranks, check network |
| GPU memory 0, no new loss | Training crashed or was killed | Check stderr for error, check PID |
| `nan` in loss or grad norm | Numerical instability | Reduce LR, check data for corruption |
| `skipped iterations` increasing | Loss scale too high | Will auto-recover, monitor |
| Loss suddenly spikes | Data issue or LR too high | Check data batch, check LR schedule |
| NCCL timeout errors in stderr | Network issue between nodes | Check IB/RoCE, check firewall |
| OOM (Out of Memory) | Model too large for GPU memory | Reduce micro_batch_size, increase TP/PP, enable recomputation |

### Check All Ranks for Errors

When anomaly detected, check stderr on ALL ranks (not just rank 0):

```bash
for r in $(ls ${ATTEMPT}/ 2>/dev/null); do
  [ -f "${ATTEMPT}/$r/stderr.log" ] || continue
  errors=$(grep -c -iE "error|exception|traceback|oom|killed" "${ATTEMPT}/$r/stderr.log" 2>/dev/null)
  if [ "$errors" -gt 0 ]; then
    echo "=== rank $r ($errors errors) ==="
    tail -10 ${ATTEMPT}/$r/stderr.log
  fi
done
```

### Multi-Node Error Check

```bash
for host_dir in ${EXP_DIR}/logs/details/host_*/; do
  host=$(basename "$host_dir")
  latest=$(ls -d ${host_dir}[0-9]*/ 2>/dev/null | sort | tail -1)
  attempt=$(find "$latest" -type d -name "attempt_*" 2>/dev/null | head -1)
  [ -z "$attempt" ] && continue
  echo "=== $host ==="
  for r in $(ls "$attempt/" 2>/dev/null); do
    [ -f "${attempt}/$r/stderr.log" ] || continue
    errors=$(grep -c -iE "error|exception|traceback" "${attempt}/$r/stderr.log" 2>/dev/null)
    [ "$errors" -gt 0 ] && echo "  rank $r: $errors errors" && tail -3 "${attempt}/$r/stderr.log"
  done
done
```

---

## Step 4: Periodic Monitoring Report

Report the following metrics at user-requested intervals:

| Metric | Source | Command |
|--------|--------|---------|
| Iteration / total | stdout.log last rank | `grep "iteration" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -1` |
| LM loss | stdout.log last rank | `grep "lm loss" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -5` |
| Elapsed time per iteration | stdout.log last rank | `grep "elapsed time per iteration" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -1` |
| Grad norm | stdout.log last rank | `grep "grad norm" ${ATTEMPT}/${LAST_RANK}/stdout.log \| tail -5` |
| GPU utilization + memory | nvidia-smi | `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader` |
| Node health (multi-node) | SSH | `ssh <node> nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` |

### Loss Trend Analysis

```bash
# Extract loss values for trend analysis
grep "lm loss" ${ATTEMPT}/${LAST_RANK}/stdout.log | awk '{for(i=1;i<=NF;i++) if($i=="loss:") print $(i+1)}' | tail -50
```

A healthy training run shows:
- Loss decreasing over time (not necessarily monotonically)
- Grad norm stable (not exploding)
- Elapsed time per iteration consistent (no sudden slowdowns)
- All GPUs at high utilization (>90%)

### Alert Conditions

Alert immediately if ANY of these occur:
- No new loss output for > 5 minutes (training may have stopped)
- Loss becomes `nan` or `inf`
- Grad norm exceeds 100x its typical value
- GPU utilization drops to 0% on any GPU
- Any node becomes unreachable (multi-node)
- stderr shows new errors

---

## Checkpoint Monitoring

```bash
# Check latest checkpoint
ls -lt ${EXP_DIR}/checkpoints/ 2>/dev/null | head -5

# Check if checkpoint is being written (file size changing)
ls -lh ${EXP_DIR}/checkpoints/iter_*/mp_rank_00/ 2>/dev/null | tail -3
```

---

## TensorBoard / WandB

If TensorBoard logs exist:
```bash
ls ${EXP_DIR}/tensorboard/ 2>/dev/null
```

If WandB logs exist:
```bash
ls ${EXP_DIR}/wandb/ 2>/dev/null
```

These provide richer visualization but require a browser. For CLI-based monitoring, use the log parsing commands above.

---

## Common Issues During Monitoring

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Log file exists but empty | Training crashed during startup | Check stderr.log for import errors or CUDA issues |
| Loss stuck at same value | Learning rate too low or gradient issue | Check LR schedule, verify grad norm is non-zero |
| Loss oscillating wildly | LR too high or batch size too small | Reduce LR or increase effective batch size |
| `Killed` in stderr | OOM killed by system | Check `dmesg \| tail`, reduce memory usage |
| Log timestamps stop updating | Process hung (NCCL deadlock, data loader stuck) | Check GPU util — if 0%, likely NCCL hang; if >0%, likely data loader |

---

## Related Skills

- `train-run` — launch, stop, and manage training jobs
- `train-config` — generate and validate training configuration
- `train-precision-alignment` — verify numerical alignment between implementations
- `topo-detect` — detect hardware topology for diagnosing performance issues
