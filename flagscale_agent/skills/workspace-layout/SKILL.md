---
name: workspace-layout
description: Standardized workspace directory layout and storage management for FlagScale projects. Covers shared storage
  detection, fixed paths for models/datasets/experiments/checkpoints/logs/conda envs, experiment isolation (never overwrite),
  disk space pre-checks, and artifact deduplication via memory.
keywords:
- 下载
- 模型
- 权重
- download
- model
- weights
- checkpoint
- 存储
- storage
- 磁盘
- disk
- 路径
- path
- 目录
- directory
- workspace
- 实验
- experiment
- exp_dir
- 数据集
- dataset
- conda
- 环境
- huggingface
- snapshot_download
requires: []
suggests: []
constraints:
- id: workspace_layout_install_to_local_not_shared
  description: Conda envs, pip packages, models, datasets, and experiments MUST be on shared storage, NOT local paths (/tmp,
    ~/)
  trigger:
    tools:
    - shell
    keywords:
    - conda create
    - conda install
    - pip install
    - snapshot_download
    - wget http
    - mkdir -p /tmp
  prompt: Check if artifacts are being created on local storage instead of shared storage
  correction: Use shared storage path for all persistent artifacts.
- id: no_experiment_overwrite
  description: Never overwrite or reuse a previous experiment directory
  trigger:
    tools:
    - shell
    - write_file
    keywords:
    - rm -rf
    - rmdir
    - exp_dir
  prompt: Check if the agent is about to delete or overwrite an existing experiment directory
  correction: Create a new experiment directory with a unique name. Never reuse or delete experiment dirs.
- id: disk_space_precheck
  description: Check disk space before large downloads or training launches
  trigger:
    keywords:
    - snapshot_download
    - wget http
    - curl http
    - huggingface download
    - launch training
  prompt: Check if a disk space check was done before this large operation
  correction: Run `df -h <target_dir>` before large downloads. Warn if free space < 1.5x estimated size.
- id: artifact_dedup_check
  description: Check memory for existing artifacts before downloading
  trigger:
    keywords:
    - snapshot_download
    - wget http
    - curl http
    - git clone http
    - huggingface download
  prompt: Check if memory was consulted for existing artifact paths before downloading
  correction: Check memory_read for existing paths before downloading. Avoid duplicate downloads.
- id: workspace_layout_experiment_must_be_training
  description: workspace_experiment create should only be used for actual training/inference experiments, not for environment setup,
    data download, or other infrastructure tasks. If created, the experiment MUST be updated (workspace_experiment update) before
    the task ends.
  trigger:
    tools:
    - workspace_experiment
    keywords:
    - create
  prompt: "SCOPE: workspace_experiment create. CHECK: Is this experiment being created for a non-training purpose (environment setup,
    conda install, data download, dependency build)? Experiments should only track training/inference runs that produce metrics,
    not infrastructure preparation. Also: if an experiment was already created earlier in this session, was it properly updated
    with results before creating a new one?"
  correction: "Do NOT create workspace_experiment entries for infrastructure tasks (env setup, data prep, dependency install).
    Only create experiments for actual training/inference runs. If you already created one, you MUST call
    workspace_experiment update with results before the task ends."
context_injection:
  always:
  - Standard Directory Layout
  by_tool:
    shell:
    - Detect Storage Root
    - Disk Space Pre-check
    - Artifact Discovery
    write_file:
    - Rules for Each Artifact Type
---
# Workspace Layout & Storage Management

This skill defines the standard directory layout and storage management rules for all FlagScale projects. Follow these rules whenever creating, downloading, or referencing artifacts (models, datasets, checkpoints, logs, conda envs).

---

## Step 1: Detect Storage Root

Run once per session. If memory key `workspace_root` already exists and the path is still valid, skip detection.

**User override**: If the user specifies a custom root or custom paths, always respect their choice. Present the auto-detected recommendation first, then confirm with the user before proceeding. Record the user's choice in memory.

### 1a. Identify shared storage

```bash
df -hT 2>/dev/null
mount | grep -iE 'type (nfs|lustre|gpfs|ceph|fuse\.ceph|beegfs|panfs|cifs)' 2>/dev/null
```

Shared storage = NFS, Lustre, GPFS, Ceph, BeeGFS, CIFS, or FUSE-based network mounts.

If `topo_storage` exists in memory (from topo-detect skill), use the shared mount recorded there directly — no need to re-detect.

### 1b. Choose root

Priority order:
1. **Shared storage mount** — required for multi-node training. All nodes must see the same path.
2. **Largest persistent volume** — if no shared storage, pick the mount with the most free space (excluding tmpfs/overlay).
3. **`/workspace`** — last resort fallback.

### 1c. Confirm with user

Before proceeding, present the detected root and layout to the user:
- "Detected shared storage at `/mnt/shared` (NFS, 2TB free). Will use it as workspace root. Artifacts will go under `/mnt/shared/models/`, `/mnt/shared/experiments/`, etc. OK or do you prefer a different path?"
- If the user specifies a different path, use that instead.

### 1d. Record

Save the chosen root to workspace state:
- `workspace_state(action="write", section="Workspace", content="root: <chosen_path>")`

---

## Step 2: Standard Directory Layout

All artifacts go under `<root>` (the detected storage root):

```
<root>/
├── models/<org>/<model_name>/            # pretrained weights (read-only after download)
│   ├── config.json
│   ├── tokenizer.json / tokenizer.model
│   └── model*.safetensors / pytorch_model*.bin
├── datasets/<dataset_name>/              # processed training data
│   ├── *.bin / *.idx                     # Megatron format
│   └── *.tar                             # webdataset / multimodal tars
├── code/<project_name>/                  # cloned repos, custom code
├── experiments/<model>/<exp_name>/       # one dir per experiment — NEVER reuse
│   ├── config/                           # frozen copy of all configs used
│   ├── logs/                             # training logs (FlagScale launcher manages subdirs)
│   ├── checkpoints/                      # model checkpoints
│   ├── tensorboard/                      # TB event files
│   └── metrics/                          # extracted metrics for comparison
└── envs/<env_name>/                      # conda environments (--prefix)
```

### Path examples

| Artifact | Path |
|----------|------|
| FlagScale source code | `<root>/code/FlagScale/` |
| Qwen2.5-7B-Instruct weights | `<root>/models/Qwen/Qwen2.5-7B-Instruct/` |
| SigLIP vision encoder | `<root>/models/google/siglip-so400m-patch14-384/` |
| Processed Megatron data | `<root>/datasets/pile-10k/` |
| Training experiment | `<root>/experiments/qwen3_0.6b/tp2_pp1_dp4_bs8/` |
| Conda environment | `<root>/envs/flagscale/` |

---

## Step 3: Rules for Each Artifact Type

### 3a. Model weights

- **Before downloading**: check standard path, `~/.cache/huggingface/hub/`, and memory for existing copies. List what's found and what's missing.
- **Confirm with user**: show a table — model name, estimated size, target path. Wait for approval before downloading.
- **Download method**: `snapshot_download(repo_id, local_dir=<root>/models/<org>/<model>)` for consistent paths.
- **After downloading**: use `workspace_state(action="append", content="model: <path>")` so future sessions find it without re-downloading.
- **Read-only**: never modify downloaded weights in place. Checkpoint conversion outputs go to a separate path.

### 3b. Datasets

- Same confirm-before-download rule as model weights for files > 1GB.
- Use `<root>/datasets/<name>/` as `data_path` prefix in training configs.
- Record path in memory after creation.

### 3c. Experiments

- **Isolation is non-negotiable**: each experiment gets its own directory. NEVER overwrite or reuse a previous experiment directory.
- Naming: use descriptive names reflecting the config (e.g., `qwen3_0.6b_tp2_pp1_bs8`). For reruns of the same config, append a timestamp (e.g., `qwen3_0.6b_tp2_pp1_bs8_20260429`).
- When generating FlagScale `train.yaml`: set `experiment.exp_dir` to `<root>/experiments/<model>/<exp_name>/`.
- Checkpoints, logs, and tensorboard dirs are subdirectories of the experiment — don't scatter them elsewhere.
- **Experiment registry**: Every experiment MUST be recorded in workspace_state (section "Experiments") with structured fields: Purpose (目的), Hypothesis (假设), Config (配置), Dir (目录), Result (结果), Reflection (反思), Next (下一步). See train-run skill for the full format and lifecycle. This registry serves two purposes:
  1. **Log discovery**: find any experiment's directory instantly without filesystem search
  2. **Knowledge accumulation**: each experiment's Reflection feeds into the next experiment's design, creating a chain of reasoning that accelerates iteration and prevents repeating mistakes

### 3d. Conda environments

- Create with fixed prefix: `conda create --prefix <root>/envs/<env_name> python=<version>`
- Execute with: `conda run --prefix <root>/envs/<env_name> <command>`
- This makes environments discoverable and consistent across sessions. Never use auto-generated env names.
- Record env path in memory after creation.

### 3e. Code repositories

- Clone to `<root>/code/<project>/` for repos that need to persist across sessions.
- Working directory code (already in `/workspace/...`) stays where it is — don't move it.

---

## Step 4: Disk Space Pre-check

Before any large operation, verify sufficient space on the target path.

### 4a. Before downloading

```bash
df -h <target_directory>
```

Estimate total download size. Warn if free space < 1.5× estimated size. If insufficient, suggest:
1. A different mount with more space
2. Cleaning up old artifacts
3. Let user decide

### 4b. Before training

Estimate storage needs:
- **Checkpoint size** ≈ `param_count × 2 bytes` (BF16) per checkpoint
- **Total checkpoint storage** ≈ `ckpt_size × (total_steps / save_interval)`
- **Logs + TensorBoard** ≈ 1-5 GB depending on duration
- **Optimizer states** (if saved) ≈ `param_count × 8 bytes` per checkpoint

Warn if free space < estimated total. For long training runs, also warn about checkpoint accumulation.

---

## Step 5: Artifact Discovery

Before creating or downloading anything:

1. Check memory for previously recorded paths (`workspace_root`, model paths, env paths, etc.)
2. Check standard paths under `<root>/`
3. Check common alternatives (`~/.cache/huggingface/hub/`, `/tmp/`, working directory)
4. List what was found and what's missing
5. Only proceed to download/create what's actually missing

After creating or downloading:
- Record the path in memory with a descriptive key (e.g., `model_qwen2.5_7b_path`, `env_flagscale_path`)
