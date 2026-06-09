---
name: train-run
description: Launch, stop, and manage FlagScale distributed training jobs. Covers server connection, environment checks, GPU
  availability, preflight validation, training launch (CLI and legacy), stop commands, log directory structure, and quick
  verification paths.
keywords:
- train
- training
- launch
- start
- stop
- run
- preflight
- dry run
- dryrun
- 训练
- 启动训练
- 开始训练
- 停止训练
- 分布式训练
- GPU
- finetune
- pretrain
parameters:
- name: model_name
  description: Model name (e.g., qwen3, llama3)
- name: exp_dir
  description: Experiment output directory
requires:
- workspace-layout
- train-config
suggests:
- train-env-setup
- train-data-prep
constraints:
- id: no_kill_without_check
  description: Never kill training processes without checking logs first
  trigger:
    tools:
    - shell
    keywords:
    - pkill
    - killall
    - kill -9
  prompt: Check if the agent is about to kill a training process without first reading logs to understand why
  correction: Read training logs first to diagnose the issue before killing the process.
- id: monitor_reminder
  description: Remind to monitor training after launch
  trigger:
    tools:
    - shell
    keywords:
    - flagscale.train
    - torchrun
    - python -m
  prompt: Check if this is a training launch command (not dryrun)
  correction: After launching training, immediately call monitor() to observe the process.
context_injection:
  always:
  - Critical Rules
  - Post-Launch Protocol
  by_tool:
    shell:
    - 'Step 3: Preflight Check'
    - 'Step 4: Start / Stop Training'
    - Error Handling
---
# FlagScale Training Launch

Launch, stop, and manage FlagScale distributed training jobs on GPU servers.

## Critical Rules

1. **If the user says the environment/conda is already set up, DO NOT install packages.** Go straight to preflight verification (Step 3). Only install if preflight imports fail.
2. **After launching training (not dryrun — dryrun only generates scripts), you MUST immediately call monitor() to observe the process.** Do not proceed to other tasks without monitoring.
3. **Never delete experiment output directories.**

## Prerequisites

- SSH access to training server
- Docker container with FlagScale environment (or bare metal with conda)
- FlagScale cloned and installed in the conda environment (see `train-env-setup` skill)
- Training config files ready (see `train-config` skill)

---

## Step 1: Connect to Server

SSH into training server, enter Docker container, activate conda env, cd to FlagScale project directory.

```bash
sudo docker exec -it <container_name> bash
# In non-interactive shells (agent), use: conda run --prefix <env_path> <command>
# In interactive shells (user), use: conda activate <env_name>
cd <workspace_root>/code/FlagScale
```

---

## Step 2: Check Environment and GPU Availability

### Determine Environment Type

Do this once per server, remember the result:

```bash
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
  echo "CONTAINER environment"
else
  echo "BARE METAL environment"
fi
```

### Check GPU Status

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

**Interpreting GPU status — depends on environment**:

- **Container**: `nvidia-smi` shows memory/utilization from ALL containers sharing the GPU, but only shows PIDs from the current container. If GPUs show memory occupied but no processes visible, this means OTHER containers are using those GPUs — NOT leaked memory. Report to user: "GPUs X-Y are in use by other containers, GPUs Z are available."
- **Bare metal**: all processes are visible. If GPUs show memory occupied with no PID, that is genuinely abnormal (zombie GPU memory). Can try `nvidia-smi --gpu-reset` or report to user.

**Go/no-go**: Target GPUs must show near-zero memory used. If occupied, alert user with the correct explanation based on environment type.

### Multi-Node GPU Check

```bash
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
  host=$(echo "$line" | awk '{print $1}')
  echo "=== $host ==="
  ssh $host "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
done < <hostfile_path>
```

---

## Step 3: Preflight Check

**ALWAYS run this before starting training.** Environment may have changed since last session.

### 3a. Core Dependencies

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')
from megatron.plugin.platform import get_platform
print(f'Megatron platform: {get_platform()}')
import transformer_engine
print(f'TransformerEngine: {transformer_engine.__version__}')
import apex; print('Apex: OK')
import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')
print('All dependencies OK')
"
```

If ANY import fails, stop and tell the user which dependency is broken. Suggest running `/skill train-env-setup` to fix.

### 3b. GPU Availability

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

All target GPUs must show near-zero memory usage.

### 3c. Data Path Validation

Validate ALL data paths referenced in the training config. This is not just "do the files exist" — it's "will the data pipeline actually load them at runtime."

**For Megatron binary format (FlagScale native):**
```bash
DATA_PATH="<data_path from config>"
ls -lh ${DATA_PATH}.bin ${DATA_PATH}.idx
```

**For third-party frameworks (parquet, JSONL, custom loaders):**
1. Check that data directories exist and contain expected files:
   ```bash
   ls <data_dir>/*.parquet | wc -l   # or *.jsonl, *.json, etc.
   ```
2. If the config references a metadata/index file (e.g., parquet_info JSON, dataset_info), open it and verify that the paths INSIDE the file match the actual data locations. Placeholder paths like `your_data_path/`, `/path/to/`, or paths from a different machine are the #1 cause of silent data loading failures.
   ```bash
   # Example: check if parquet_info keys match actual data_dir
   python -c "
   import json, os
   info = json.load(open('<parquet_info_path>'))
   data_dir = '<actual_data_dir>'
   actual_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
   matched = [f for f in actual_files if f in info]
   print(f'Matched: {len(matched)}/{len(actual_files)} files')
   if len(matched) == 0:
       print(f'WARNING: Zero matches! Info keys sample: {list(info.keys())[:2]}')
       print(f'Actual files sample: {actual_files[:2]}')
   "
   ```
3. If paths don't match, fix the metadata file (replace placeholder prefix with actual path) BEFORE launching.

**General rule**: any config file, JSON, or Python dict that maps dataset names to file paths is a potential source of path mismatch. After modifying data paths, always verify the FULL chain: config → dataset registry → metadata files → actual files on disk.

If files are missing or paths don't match, stop and tell the user. Suggest running `/skill train-data-prep`.

### 3d. Topology Freshness (Optional)

If memory contains `topo_compute` from a previous topo-detect run, do a quick sanity check:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
```

If GPU count or model differs from what's in memory, warn the user that topology data is stale.

### 3e. Dryrun (Script Generation) — HARD GATE

**CRITICAL DISTINCTION:**
- `flagscale train <model> --dryrun` = generates launch scripts ONLY. No training is launched, no GPU is used, no process starts. It validates config syntax and produces shell scripts.
- Validation run (`--train-iters 20`) = actual short training that launches processes, uses GPUs, loads model, and runs 20 iterations. This is what validates the pipeline.

**Step 1: Generate scripts with dryrun**
```bash
flagscale train <model> --dryrun
```

If dryrun fails, it means config has syntax errors or missing fields. Fix and retry.

After dryrun succeeds, inspect the generated launch script:
```bash
cat {exp_dir}/logs/scripts/host_*_run.sh
```
Verify: correct GPU count (`--nproc_per_node`), correct entrypoint, all expected CLI flags, no placeholder paths (`/path/to/`, `FIXME`, `TODO`).

**If you modify the config after dryrun, you MUST re-run dryrun.** Cached scripts contain hardcoded values from the previous config.

**Step 2: Run a short validation training**
After dryrun scripts look correct, run actual training with minimal iterations to validate the full pipeline (model loading, data loading, forward/backward pass):
```bash
flagscale train <model> --train-iters 20
```
Only proceed to full training after this validation passes.

For third-party reproduction tasks (no FlagScale launcher): construct the full launch command, print it, and verify it manually before executing. Check: correct `--nproc_per_node`, correct `PYTHONPATH`, correct entrypoint script, all required CLI args present, no placeholder paths in any referenced config files.

### 3f. Launch Script Validation (MANDATORY)

Before launching, validate the generated launch script against the actual FlagScale source code:

1. **Read the argument parser** — find how each CLI flag is parsed in the entrypoint (e.g., `megatron/training/arguments.py` or the model's custom argparse). Verify your config values match the expected types and formats defined there.
2. **Read the launcher code** — understand how FlagScale generates and executes launch scripts. Check that your config produces the expected command structure.
3. **Read existing examples** — look at `examples/<similar_model>/conf/` for reference configs. Compare your config structure, field names, and value formats against working examples.
4. **Trace the data path** — follow how `--data-path` or equivalent is consumed by the dataloader code. Verify your paths match what the code expects (prefix format, file extensions, index files).

The source code defines what's valid — not a static checklist. If you're unsure about a config value, read the code that consumes it.

### 3g. Config Arithmetic Verification

Before launching, verify ALL of the following. Do not skip any item:
- `global_batch_size % (micro_batch_size × data_parallel_size) == 0`
- `num_attention_heads % tensor_model_parallel_size == 0`
- `num_key_value_heads % tensor_model_parallel_size == 0` (for GQA models)
- `num_layers % pipeline_model_parallel_size == 0` (if PP > 1)

Also verify config value types match expectations. Read argparse definitions for non-obvious types (e.g., `--rotary-base` expects int, not float).

### 3g. Checkpoint Compatibility Verification

If loading a checkpoint (`--load`):
1. Verify checkpoint exists and contains `latest_checkpointed_iteration.txt`
2. Verify checkpoint TP/PP matches config TP/PP — a TP=1 checkpoint cannot be loaded with TP=4 without resharding
3. Verify `vocab_size` matches between checkpoint and config
4. Verify checkpoint format (`torch` vs `dist`) matches `ckpt_format` in config

```bash
ls <checkpoint_path>/latest_checkpointed_iteration.txt
cat <checkpoint_path>/latest_checkpointed_iteration.txt
```

### 3h. Memory Budget Estimation

Before launching, estimate per-GPU memory:
```
per_gpu_memory = model_params × 2 (bf16) + gradients × 2 + optimizer_states × (8 / DP)
```
If this exceeds GPU memory, do NOT launch — fix parallelism or enable activation checkpointing first.

### 3i. Data Pipeline Standalone Test — HARD GATE

**Do NOT launch training without verifying the data pipeline independently.**

This catches: path mismatches, format errors, infinite loops, missing files, wrong tokenization — all in seconds instead of the 10+ minutes of model loading.

```bash
python -c "
import sys, time
sys.path.insert(0, '.')
# Import the dataset class used in training config
from <dataset_module> import <DatasetClass>

# Instantiate with the SAME args as training config
ds = <DatasetClass>(<args_from_config>)
print(f'Dataset length: {len(ds)}')

# Fetch 3 batches, with timeout
for i in range(3):
    t0 = time.time()
    batch = ds[i]
    elapsed = time.time() - t0
    if isinstance(batch, dict):
        shapes = {k: v.shape if hasattr(v, 'shape') else type(v).__name__ for k, v in batch.items()}
    else:
        shapes = batch.shape if hasattr(batch, 'shape') else type(batch).__name__
    print(f'Batch {i}: {shapes} ({elapsed:.2f}s)')
    if elapsed > 10:
        print(f'WARNING: batch {i} took {elapsed:.1f}s — possible infinite loop or I/O issue')
        break
print('Data pipeline OK')
"
```

If this script hangs, crashes, or shows unexpected shapes, fix the data pipeline BEFORE launching training.

### 3j. Checkpoint Loading Verification

If loading a pretrained checkpoint (not training from scratch), verify it actually loaded by checking step-0 loss.

After the validation run (step 2 of 3e) completes 2 iterations:
1. Check the loss at iteration 0
2. Compare with `ln(vocab_size)` — the expected loss for random initialization

```bash
# Extract step-0 loss from training log
grep -m1 "lm loss" <log_file>
python -c "import math; print(f'Random init baseline: {math.log(<vocab_size>):.2f}')"
```

**Decision rule**:
- Loss ≈ ln(vocab_size) → checkpoint did NOT load. Model is randomly initialized. STOP and debug.
- Loss << ln(vocab_size) → checkpoint loaded successfully. Proceed.

Common causes of checkpoint not loading:
- Conversion code exists but isn't called in the training script
- `--load` path is wrong or points to empty directory
- TP/PP mismatch between checkpoint and config (silent fallback to random init)
- `--finetune` flag missing (Megatron skips optimizer state but still needs the flag to load weights in some modes)

**Only after ALL checks pass (3a through 3j), proceed to start training.**

---

## Step 4: Start / Stop Training

**ALWAYS use FlagScale Launcher** — never bypass it with raw `torchrun` or hand-written launch scripts. The launcher provides per-rank log separation, experiment directory structure, config validation, and clean shutdown (`--stop`). Without it, all ranks write to one stream (debug prints get lost or interleaved), there's no experiment directory structure, and you can't use `--stop`. If the launcher fails, fix the root cause — do not work around it.

**Exception — third-party reproduction tasks**: When reproducing a third-party model's training (e.g., LLaVA-OneVision, Qwen-VL) using their own training scripts before migrating to FlagScale, use their native launch method (typically `torchrun` + their training script). The FlagScale launcher doesn't apply here. However, ALL other rules in this skill still apply: experiment registry, preflight checks, data validation, post-launch monitoring, and health checks. The only difference is the launch command itself.

**IMPORTANT**: Always set `PYTHONUNBUFFERED=1` before launching training. Without it, Python buffers stdout and training logs appear delayed or empty, making health monitoring unreliable.

```bash
# Start (CLI)
PYTHONUNBUFFERED=1 flagscale train <model>
# Start (legacy)
PYTHONUNBUFFERED=1 python run.py --config-path ./examples/<model>/conf --config-name train action=run

# Stop (CLI)
flagscale train <model> --stop
# Stop (legacy)
python run.py --config-path ./examples/<model>/conf --config-name train action=stop

# Dry run (generate scripts only — NO training launched, NO GPU used)
flagscale train <model> --dryrun
```

### Post-Launch Protocol (MANDATORY)

After launching training, follow this sequence EVERY time. Do NOT skip steps.

**Before launch — HARD GATE: register the experiment and create its directory:**

0a. Create a DEDICATED experiment directory for this run. Never reuse a directory from a previous experiment. Naming convention: `<model>_<config>_<purpose>_v<N>` (e.g., `qwen3_tp4_pp1_pretrain_v1`).

**Version bumping rule — what counts as a new experiment:**
- Produced at least 1 step of metrics → real experiment. Next meaningful change → bump version.
- Changed a meaningful parameter (LR, TP/PP, batch size, data, model code, freeze strategy) → new version.
- Launch failed before any metrics (import error, path error, config typo, port conflict) → NOT a new experiment. Record as "launch attempt N failed: reason" in the current entry's **Launch notes** field. Fix and retry under the same version and directory.
- Training crashed after producing metrics, restarting with same config → still the same experiment. Note the crash in Result.

0b. Write the experiment entry in workspace_state (section "Experiments") BEFORE launching. **If you haven't written this entry, you are NOT allowed to launch.**

   ```
   workspace_state(action="write", section="Experiments", content="""
   ### <model>_<config>_<purpose> (running)
   - **Purpose**: <what you are verifying and why>
   - **Hypothesis**: <expected outcome — e.g., loss starts near ln(vocab_size) and decreases>
   - **Config**: TP=<N> PP=<N> DP=<N>, micro_bs=<N>, seq_len=<N>, bf16, <N> steps
   - **Dir**: <experiment_directory>
   - **Launch notes**: (record failed launch attempts here, not as separate experiments)
   - **Result**: (pending)
   - **Reflection**: (pending)
   - **Next**: (pending)
   """)
   ```

   Purpose and Hypothesis are the most important fields — they force you to think about WHY before acting. If you can't articulate the purpose, you're not ready to launch.

0b-retry. **If a launch fails before producing metrics**, update the SAME entry's Launch notes:
   ```
   workspace_state(action="write", section="Experiments", content="""
   ### <model>_<config>_<purpose> (running)
   ...same fields...
   - **Launch notes**: attempt 1 failed: <reason>. Fixed: <what you changed>.
   """)
   ```
   Then fix and retry. Do NOT create a new experiment entry or bump the version.

0c. **Verify no old training processes are running.** Before every launch, check for leftover processes from previous runs and clean them up:

   ```bash
   # Check for existing training processes
   pgrep -fa "torchrun|train_|flagscale" | grep -v grep
   # If any found, kill them and wait for GPU memory to be released
   pkill -9 -f "torchrun|train_|flagscale" 2>/dev/null; sleep 5
   # Verify GPUs are free
   nvidia-smi | grep -E "MiB|%"
   ```

   Launching a new training run while an old one is still alive causes port conflicts, GPU OOM, and log corruption. This check takes seconds; debugging the resulting failures takes much longer.

**Within 30 seconds of launch:**
1. **Wait 10-15 seconds** before checking logs — the log directory may not exist yet (race condition with nohup/background launch)
2. Use `monitor(output_dir="<exp_dir>", duration=60)` to auto-discover logs AND scan stderr — NEVER use raw `find` commands (they may find old logs from previous runs)
3. If stderr has errors → training failed at startup. Fix and retry.
4. **Check stderr FIRST, not stdout** — crash info is in stderr. A process showing "wandb initialized" in stdout may already be dead.

**After first metrics appear (usually 1-3 minutes):**
4. **Auto-trigger `parse_training_metrics`** — do NOT use `tail -f` or `grep` to manually scan logs. The tool parses structured metrics and runs health checks automatically. Call it with `vocab_size` for the random-output check:
   ```
   parse_training_metrics(log_path="<log_path>", vocab_size=<vocab_size>)
   ```
5. Interpret the health check results:
   - `loss ≈ ln(vocab_size)` → model outputs are random. Stop. Check: weights loaded? forward pass correct?
   - `grad_norm = 0` or `num_zeros ≈ total_params` → gradients not flowing. Check loss computation, frozen params.
   - `loss not decreasing after 10+ steps` → learning rate, optimizer, or data issue.
6. Report the first metrics to the user with health assessment. Include: initial loss, loss trend, grad norm, throughput (tokens/sec or samples/sec).

**After training completes or fails — close the experiment:**

8. Update the experiment entry with Result, Reflection, and Next:

   ```
   workspace_state(action="write", section="Experiments", content="""
   ### <model>_<config>_<purpose> (completed)
   - **Purpose**: <same as above>
   - **Hypothesis**: <same as above>
   - **Config**: <same as above>
   - **Dir**: <experiment_directory>
   - **Result**: <actual outcome — steps completed, loss trajectory, key metrics, peak memory>
   - **Reflection**: <lessons learned — what worked, what was tight, what surprised you>
   - **Next**: <what to try next based on this result>
   """)
   ```

   The Reflection field is critical — it captures lessons that prevent future experiments from repeating mistakes. A failed experiment with good reflection is more valuable than a successful one with no reflection.

**If health judge killed a long-running command:**
When the agent's health judge kills a `sleep` or `tail -f` command, do NOT blindly retry with another sleep. Instead:
1. Check if the training process is still alive: `kill -0 <pid>` or check PID file
2. Check GPU utilization: `nvidia-smi`
3. Check the latest log lines directly (no sleep)
4. Then decide: wait more, or investigate a problem

**Never declare training "successful" based only on "it didn't crash".** A training run that produces random output is worse than a crash — it wastes GPU hours silently.

---

## Log Directory Structure

FlagScale training logs are organized as follows. Understanding this structure is CRITICAL — you MUST use the correct commands to find logs, never guess paths.

```
<exp_dir>/
├── logs/
│   ├── host_0_<hostname>.output              # torchrun launcher output
│   ├── pids/host_0_<hostname>.pid            # launcher PID
│   ├── scripts/host_0_<hostname>_run.sh      # actual launch script
│   ├── scripts/host_0_<hostname>_stop.sh     # stop script
│   └── details/host_0_<hostname>/
│       ├── 20260424_153816.588538/           # timestamp dir (YYYYMMDD_HHMMSS.us)
│       │   └── default_<hash>/attempt_0/
│       │       ├── 0/stdout.log  stderr.log  # rank 0
│       │       ├── 1/stdout.log  stderr.log  # rank 1
│       │       └── .../                      # one dir per rank
│       └── 20260424_162209.763893/           # another run (newer!)
│           └── ...
├── checkpoints/
├── tensorboard/
└── wandb/
```

Key points:
- `exp_dir` comes from `experiment.exp_dir` in `train.yaml`
- Each training launch creates a NEW timestamp directory under `details/host_X_<hostname>/`
- Multiple runs accumulate — you MUST find the LATEST timestamp dir, not the first one
- Each rank (GPU process) has its own `stdout.log` and `stderr.log`
- Rank 0's stdout.log contains the main training output (loss, iteration, etc.)
- stderr.log contains errors, warnings, and import failures

### Finding the Latest Logs

**Always use the dedicated tool first** — it handles the full directory traversal, rank scanning, and health checks in one call:

```
find_latest_log(experiment="<exp_dir>", vocab_size=<vocab_size>)
```

If the experiment dir is recorded in workspace_state's "Experiments" section, use that path directly. NEVER use `find`, `ls -R`, or shell globbing to search for log files.

**Manual fallback** (only if the tool is unavailable):
EXP_DIR=$(grep 'exp_dir:' examples/<model>/conf/train.yaml | awk '{print $2}')
LATEST=$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ 2>/dev/null | sort | tail -1)
ATTEMPT=$(find "$LATEST" -type d -name "attempt_*" | head -1)
tail -30 ${ATTEMPT}/0/stdout.log
tail -30 ${ATTEMPT}/0/stderr.log
```

One-liners:
```bash
tail -30 "$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ | sort | tail -1)"/*/attempt_0/0/stdout.log
tail -30 "$(ls -d ${EXP_DIR}/logs/details/host_0_*/[0-9]*/ | sort | tail -1)"/*/attempt_0/0/stderr.log
```

**NEVER do this:**
- Don't hardcode timestamp dirs like `20260424_153816.588538`
- Don't use `find -name stdout.log` without sorting — it may return old runs
- Don't use `sleep N && tail` — check directly

---

## Quick Verification Paths

When the user wants to quickly verify a training setup works:

1. **Minimal config**: `train_iters: 3-5`, `micro_batch_size: 1`, `global_batch_size: DP × 1`
2. **Single GPU first**: Start with 1 GPU (TP=1, PP=1, DP=1) before scaling
3. **Smallest dataset**: Use the smallest available split or demo data
4. **Dry run**: Use `flagscale train <model> --dryrun` to validate config without launching
5. **Stage-by-stage**: If the recipe has stages, run one stage at a time to isolate failures

### Common Pitfalls

- `micro_batch_size` must divide `global_batch_size / (TP * PP * DP)`
- Megatron checkpoint format: `--load` path must contain `latest_checkpointed_iteration.txt`
- Multi-node: verify NCCL connectivity before launching full training
- OOM on first iteration: reduce `micro_batch_size` or enable activation checkpointing before reducing parallelism

---

## Error Handling

### Launch Failures

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `ModuleNotFoundError: megatron.*` | Megatron-LM-FL not installed or wrong PYTHONPATH | Check `pip list \| grep megatron`, reinstall if needed |
| `NCCL error: unhandled system error` | Network issue between nodes or wrong NCCL config | Check `NCCL_SOCKET_IFNAME`, verify SSH connectivity |
| `RuntimeError: CUDA out of memory` | Model too large for GPU memory | Reduce `micro_batch_size`, enable activation checkpointing, or increase TP/PP |
| `FileNotFoundError: data path` | Data files missing or wrong path in config | Verify data path with `ls`, check train.yaml data section |
| `Address already in use` | Previous training process still running | Kill old processes: `pkill -f torchrun`, wait, retry |
| `Hydra config error` | YAML syntax error or missing required field | Run `flagscale train <model> --dryrun` to check config syntax (generates scripts only) |
| Process starts but exits silently | Import error or early crash | Check stderr.log of rank 0 immediately |

### Recovery Steps

1. Read FULL stderr.log (not just tail) — multiple errors may exist
2. Fix ALL identified issues before relaunching
3. **HYDRA CACHE**: If you edited any config YAML, you MUST clean the cache before relaunching: `rm -rf outputs/<exp>/hydra/ outputs/<exp>/logs/scripts/`. FlagScale may use cached config from a previous run, making your edits appear to have no effect.
4. Never retry more than once without a clear diagnosis

### Fast Isolated Verification (before relaunching)

A full training launch can take 10+ minutes just to load the model before reaching the code you're debugging. Before relaunching, ask: "can I verify this fix without a full launch?"

**Data pipeline bugs** (the most common category):
```python
# Write a quick standalone script — runs in seconds, no model loading
import sys; sys.path.insert(0, '<project_root>')
from <dataset_module> import <DatasetClass>
ds = <DatasetClass>(<args_from_config>)
batch = next(iter(ds))
print(f"Batch keys: {batch.keys()}, shapes: {[(k, v.shape) for k, v in batch.items() if hasattr(v, 'shape')]}")
```

**Import / path errors**: `python -c "import <module>; print('OK')"` — instant.

**Config errors**: run `flagscale train <model> --dryrun` to check syntax (script generation only, no GPU).

**Shape / architecture errors**: instantiate model on meta device, no checkpoint needed.

Only relaunch the full training when the fix is in a component that can't be tested in isolation (e.g., distributed communication, optimizer state, checkpoint loading itself).

---

## Related Skills

- `train-config` — generate and validate training configuration YAML files
- `train-monitor` — monitor running training jobs, check health, detect anomalies
- `train-env-setup` — install FlagScale and all dependencies
- `topo-detect` — detect hardware topology for parallelism planning
- `train-data-prep` — prepare training data in Megatron binary format
