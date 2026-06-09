---
name: train-config
description: Generate and manage FlagScale training configuration files. Covers the two-level Hydra YAML structure (experiment
  config + task config), parallelism strategy (TP/PP/DP/EP/CP/VPP), mixed precision (BF16/FP16/FP8), TransformerEngine integration,
  checkpoint resume, multi-node setup, and topology-aware defaults.
keywords:
- config
- configuration
- yaml
- parallel
- parallelism
- TP
- PP
- DP
- tensor parallel
- pipeline parallel
- mixed precision
- bf16
- fp8
- transformer engine
- hostfile
- multi-node
- 配置
- 并行
- 并行策略
- 训练配置
parameters:
- name: model_name
  description: Model name for config directory (e.g., qwen3, llama3)
- name: model_size
  description: Model size variant (e.g., 0_6b, 7b, 70b)
requires:
- workspace-layout
suggests:
- topo-detect
constraints:
- id: train_config_gbs_too_large_for_smoke_test
  description: Global batch size must be minimal for smoke tests and initial validation runs. GBS > DP*mbs*8 is wasteful.
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - global_batch_size
  prompt: "For smoke test / environment validation / initial '跑通' runs, global_batch_size should equal
    DP * micro_batch_size (or at most 8x that). If train_iters <= 50 or train_samples is small or this is
    clearly an initial validation (not full pretraining), check if GBS is far larger than DP*mbs.
    With DP=4 mbs=1, GBS should be 4 (not 2048). The GBS=2048 in getting-started.md examples is for
    real pretraining — never copy it for validation runs."
  correction: "GBS too large for smoke test — reduce to DP × micro_batch_size (e.g., GBS=4 for DP=4 mbs=1).
    Also set train_iters=20 and REMOVE train_samples entirely for smoke tests."
- id: train_config_train_samples_too_large_for_smoke_test
  description: train_samples should not be set for smoke tests. Use train_iters instead.
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - train_samples
  prompt: "If this is a smoke test / environment validation / initial '跑通' run (train_iters <= 50, or
    the user said '跑通'/'验证'/'test'), train_samples should NOT be set. Use train_iters=20 instead.
    train_samples=244141056 (from getting-started.md) means the model will train for thousands of iterations
    which defeats the purpose of a quick validation. Check: is train_samples being set alongside a small
    train_iters? If train_iters is not set at all and train_samples is very large (>10000), this is likely
    a smoke test that forgot to limit iterations."
  correction: "Remove train_samples for smoke tests. Use train_iters=20 (or at most 50) instead.
    train_samples controls total training duration — for validation, you only need 10-20 iterations."
- id: train_config_exp_dir_not_shared_storage
  description: exp_dir must use shared storage path (not ./outputs/) when shared storage is available
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - exp_dir
    - output_dir
    - ./outputs
  prompt: If shared_storage is available and exp_dir starts with './outputs/' or './' (local path), flag it. exp_dir should
    use the workspace_root from workspace-layout.
  correction: exp_dir should use shared storage path, not ./outputs/
- id: smoke_test_reminder
  description: Remind to run a smoke test before full training
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - train_iters
    - global_batch_size
    - num_layers
  prompt: Check if this is a new config that hasn't been smoke-tested yet
  correction: Run a smoke test (train_iters=20) before launching full training.
context_injection:
  always:
  - Common Configuration Pitfalls
  - Config Validation Before Launch
  by_tool:
    write_file:
    - Two-Level YAML Structure
    - Config Generation Template
    edit_file:
    - Config Verification Checklist
    - Parallelism Strategy
    shell:
    - Quick Test vs Real Training
---
# FlagScale Training Configuration

Generate and manage FlagScale training configuration files for distributed training.

## Two-Level YAML Structure

FlagScale uses Hydra for configuration management with two levels. All paths below are relative to the FlagScale project root (e.g., `<workspace_root>/code/FlagScale/`).

### Level 1: Experiment Config

`examples/<model>/conf/train.yaml` — controls the experiment runner, environment, and which task config to load:

```yaml
defaults:
  - _self_
  - train: 0_6b          # references train/0_6b.yaml

experiment:
  exp_name: qwen3_0_6b_train
  seed: 42
  save_steps: 999999
  load: null              # checkpoint path to resume from
  exp_dir: <workspace_root>/experiments/qwen3_0_6b_train
  ckpt_format: torch      # torch or dist (distributed checkpoint)
  task:
    type: train
    backend: megatron
    entrypoint: flagscale/train/megatron/train_gpt.py
  runner:
    per_node_task: false
    no_shared_fs: false
    rdzv_backend: static
    hostfile: null         # null for single-node, path for multi-node
  cmds:
    before_start: ulimit -n 1048576
  envs:
    LOGLEVEL: "INFO"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    CUDA_DEVICE_MAX_CONNECTIONS: 1

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

Key points:
- `defaults.train` value must match a filename in `train/` subdirectory (without `.yaml`)
- `cmds.before_start` runs before training — typically activates conda env
- `experiment.exp_dir` is where all outputs go — MUST be meaningful (e.g., `./outputs/qwen3_0_6b_train`), NEVER generic names like `xxx` or `test`
- `experiment.exp_name` should match model and purpose (e.g., `Qwen3-0.6B-Train`)
- `action: run` starts training; use `action: stop` to stop

### Level 2: Task Config

`examples/<model>/conf/train/<size>.yaml` — contains three major sections:

```yaml
system:    # parallelism, precision, logging, checkpoint
model:     # architecture, training hyperparameters, optimizer
data:      # data path, tokenizer, data loading
```

---

## YAML-to-Megatron Argument Mapping

All parameters in the task-level YAML correspond to Megatron-LM command-line arguments, with hyphens replaced by underscores:
- Megatron CLI: `--tensor-model-parallel-size 4`
- FlagScale YAML: `system.tensor_model_parallel_size: 4`

This means you can look up any Megatron-LM argument documentation to understand what a YAML parameter does.

---

## Hydra Caching

Hydra generates resolved config scripts in `${experiment.exp_dir}/hydra/`. If you modify YAML configs and re-run, Hydra may use cached configs. When config changes don't take effect:

```bash
rm -rf ${experiment.exp_dir}/hydra/
```

Also check for Python cache:
```bash
find . -name "__pycache__" -path "*/conf/*" -exec rm -rf {} +
```

---

## Common Configuration Pitfalls

1. `data_path` with suffix: `data_path: ./data/file.bin` is WRONG. Use `data_path: ./data/file` (no suffix)
2. `before_start` conda: if `cmds.before_start` activates a different env than your current shell, training runs in that env — verify it has all dependencies
3. `global_batch_size` not divisible: must be divisible by `DP × micro_batch_size`. DP = total_GPUs / (TP × PP × CP × EP)
4. `transformer_impl` mismatch: if `transformer_impl: transformer_engine` but TransformerEngine-FL is not installed, training crashes immediately. Fall back to `transformer_impl: local`
5. `hostfile` null vs missing: for single-node, explicitly set `hostfile: null`. Omitting it may cause Hydra to use a default
6. Modifying the wrong YAML: changes to `train.yaml` don't affect model/data params — those are in `train/<size>.yaml`
7. `system.checkpoint.load` structure: this is a NESTED config, not a flat path. Read an existing working example before writing it. Getting the structure wrong causes silent failures (weights not loaded, loss starts at random).
8. `vocab_size` mismatch: the training config's vocab_size MUST match the tokenizer's vocabulary. Mismatch causes shape errors or silent incorrect training.

## Config Validation Before Launch

Before EVERY training launch, verify these programmatically (don't eyeball):

```bash
# 1. Data path exists (without suffix)
ls ${data_path}.bin ${data_path}.idx

# 2. Model weights exist (if loading checkpoint)
ls ${checkpoint_load_path}/

# 3. GPU count matches parallelism
# total_GPUs = TP * PP * DP * EP (DP is implicit)
# nproc_per_node * nnodes must equal total_GPUs

# 4. global_batch_size divisibility
python3 -c "
tp, pp, ep, cp = TP, PP, EP, CP
total_gpus = NPROC * NNODES
dp = total_gpus // (tp * pp * ep * cp)
gbs = GLOBAL_BATCH_SIZE
mbs = MICRO_BATCH_SIZE
assert gbs % (dp * mbs) == 0, f'GBS {gbs} not divisible by DP*MBS={dp*mbs}'
print(f'OK: DP={dp}, accumulation_steps={gbs//(dp*mbs)}')
"
```

Do NOT skip this check. Config errors waste GPU hours.

---

## Parallelism Strategy

### Parallelism Dimensions

| Dimension | YAML Key | What It Splits |
|-----------|----------|---------------|
| TP (Tensor Parallel) | `system.tensor_model_parallel_size` | Splits weight matrices across GPUs within a node |
| PP (Pipeline Parallel) | `system.pipeline_model_parallel_size` | Splits layers across GPU groups |
| DP (Data Parallel) | Implicit: total_GPUs / (TP × PP × CP × EP) | Replicates model, splits data |
| EP (Expert Parallel) | `system.expert_model_parallel_size` | Splits MoE experts across GPUs |
| CP (Context Parallel) | `system.context_parallel_size` | Splits sequence length across GPUs |
| VPP (Virtual Pipeline) | `system.num_layers_per_virtual_pipeline_stage` | Reduces PP bubble when PP ≥ 4 |

### Guidelines — Use as Context, Not Rules

The following are general considerations for parallelism strategy. They are NOT rigid rules — the right strategy depends on the specific model, hardware, workload, and constraints. Use your judgment based on the actual situation.

**General considerations:**
- TP communication is intensive — NVLink/NVSwitch interconnects handle it well, slower interconnects may not
- PP introduces pipeline bubbles that reduce efficiency, but enables training models that don't fit in GPU memory
- DP scales linearly and is the simplest form of parallelism
- EP is specific to MoE architectures
- CP is for very long sequences and requires compatible attention implementations
- VPP can reduce pipeline bubbles when PP is used

**Things to verify, not assume:**
- Whether the interconnect actually supports efficient TP at the desired scale — check topology data if available
- Whether the model actually needs PP — estimate memory requirements first
- Whether the reference config or paper specifies a parallelism strategy — prefer following proven configs over theoretical optimization
- Whether the specific model architecture has constraints (e.g., num_layers must be divisible by PP)

Don't hardcode parallelism choices based on generic rules. The optimal strategy depends on factors that vary per deployment: GPU memory, interconnect bandwidth, model size, sequence length, batch size requirements, and more. When in doubt, start with the simplest config (TP=1, PP=1, maximize DP) and scale up parallelism only as needed.

### Constraint Validation

```
total_GPUs = nnodes × nproc_per_node
TP × PP × CP × EP must divide total_GPUs evenly
DP = total_GPUs / (TP × PP × CP × EP)
global_batch_size must be divisible by (DP × micro_batch_size)
num_layers must be divisible by PP
If VPP: num_layers / PP must be divisible by num_layers_per_virtual_pipeline_stage
```

### Topology-Aware Defaults

Before generating a training config, check memory for topology data (written by topo-detect skill). Read keys: `topo_compute`, `topo_comm`, `topo_storage`. If any exist, use them as context for making parallelism decisions — but treat them as inputs to your reasoning, not as deterministic rules.

**Compute topology context:**
- `gpu_count`, `mem_gb`, `interconnect` inform what's feasible, not what's optimal
- High-bandwidth interconnect (NVSwitch/NVLink) makes larger TP feasible but doesn't mean you must use it
- Memory capacity helps estimate whether PP is needed: rough guide — each billion parameters needs ~2GB in bf16, ~4GB with optimizer states

**Communication topology context:**
- RDMA/GDR availability affects inter-node communication efficiency
- NIC count affects multi-rail NCCL performance for large-scale DP
- These are factors to consider, not automatic configuration switches

**Storage topology context:**
- Shared storage availability affects where to place data and checkpoints
- Sequential write speed affects checkpoint IO — slow storage may need async checkpointing

**If no topology data in memory:** Use the general considerations above and suggest running `/skill topo-detect` for better context.

---

## Mixed Precision

| Mode | Config Keys | When to Use |
|------|------------|-------------|
| BF16 | `model.bf16: true` | Default for A100/H100/A800. Best training stability |
| FP16 | `model.fp16: true` | For older GPUs (V100) that lack BF16 support. Requires loss scaling |
| FP8 | `system.fp8: true` | H100/H800 only. Requires TransformerEngine. Fastest but may need tuning |

Always check GPU compute capability first: BF16 requires compute capability >= 8.0 (A100+).

---

## TransformerEngine Integration

FlagScale supports NVIDIA TransformerEngine-FL for optimized transformer layers and FP8 training.

### transformer_impl Setting

In the task-level YAML under `model`:

```yaml
model:
  transformer_impl: transformer_engine   # use TE (default)
  # transformer_impl: local              # use Megatron's native implementation
```

- `transformer_engine`: uses TE's fused kernels for attention, LayerNorm, Linear — faster and supports FP8
- `local`: uses Megatron's pure PyTorch implementation — no TE dependency required

**When to use `local`**:
- TransformerEngine-FL is not installed or build failed
- Debugging numerical issues (TE fused ops can mask precision problems)
- Model architecture not supported by TE

**When to use `transformer_engine`**:
- Production training (better performance)
- FP8 training on Hopper/Blackwell GPUs (H100, B200)

### FP8 Configuration

FP8 is only available with `transformer_impl: transformer_engine` on Hopper+ GPUs:

```yaml
system:
  fp8: true                    # enable FP8 compute
  fp8_margin: 0                # scaling margin
  fp8_amax_history_len: 1024   # history length for dynamic scaling
  fp8_amax_compute_algo: max   # how to compute amax (max or most_recent)
```

FP8 reduces memory usage and increases throughput but may affect convergence — monitor loss carefully when enabling.

---

## Checkpoint Resume

To resume training from a checkpoint:

1. Set `model.load` to the checkpoint directory (the parent of `iter_NNNNNNN/`)
2. The directory must contain `latest_checkpointed_iteration.txt`
3. Parallelism (TP/PP/EP) must match the checkpoint's parallelism — changing parallelism requires checkpoint conversion
4. If `model.save` is set to the same directory as `model.load`, training will auto-resume from the latest iteration

Common checkpoint issues:
- "Could not find latest iteration" → `latest_checkpointed_iteration.txt` is missing or empty
- "Checkpoint shape mismatch" → parallelism changed since checkpoint was saved
- Checkpoint directory structure: `<save_dir>/iter_NNNNNNN/mp_rank_XX/model_optim_rng.pt`

---

## Multi-Node Configuration

### Hostfile

`examples/<model>/conf/hostfile.txt`:
```
# Format: ip slots=<num_gpus> type=<gpu_type>[optional]
# First entry is master node
10.0.0.1 slots=8
10.0.0.2 slots=8
10.0.0.3 slots=8
10.0.0.4 slots=8
```

For single-node training, set `hostfile: null` and `nnodes: 1` (or omit both).

### SSH and Network

For multi-node training, verify before launching:

1. Passwordless SSH between all nodes (both directions):
   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 <node> hostname
   ```
2. Firewall allows NCCL ports (default: 29500 for rendezvous, plus dynamic ports for NCCL):
   ```bash
   nc -zv <master_node> 29500
   ```
3. All nodes can resolve each other's hostnames
4. NCCL environment variables are consistent across all nodes

---

## Quick Test vs Real Training

Before configuring, determine the user's intent:

**Quick test / environment validation** — goal is to run 1-20 steps as fast as possible. **CRITICAL: always minimize global_batch_size for smoke tests.** A GBS of 2048 for a 0.6B model on 8 GPUs is severely wasteful — with TP=2 DP=4, set GBS = DP × micro_batch_size = 4 (mbs=1). The GBS = 2048 rule from getting-started.md is for real pretraining, not for environment validation.

**DO NOT copy train_samples from getting-started.md examples for smoke tests.** `train_samples: 244141056` means training for thousands of iterations. For validation, use `train_iters` only.

**LR MUST be scaled with GBS.** The linear scaling rule: `lr = base_lr × (your_gbs / reference_gbs)`. For smoke tests with minimal GBS:
- `model.train_iters`: 10-20 (NEVER use train_samples for smoke tests)
- `model.micro_batch_size`: 1
- `model.global_batch_size`: smallest valid value (= DP × micro_batch_size), NOT 2048
- `model.lr`: scale down proportionally: base_lr × (smoke_gbs / reference_gbs). E.g., if reference is lr=1.5e-4 at GBS=2048, and smoke GBS=4, then lr = 1.5e-4 × (4/2048) ≈ 3e-7
- `model.min_lr`: proportional to lr (typically lr/10)
- `model.lr_warmup_iters`: 0 (no warmup needed for a few steps)
- `model.eval_iters`: 0
- `model.eval_interval`: 999999
- `system.save_interval`: 999999
- `model.log_interval`: 1

**Do NOT use the reference config's lr directly with a reduced GBS — this will cause loss spikes or NaN.** Always scale lr with GBS. If unsure, use an even smaller lr (1e-7 range) — it's better to lose slowly than to explode.

**Real training** — use values from the model's reference config or paper:
- `model.train_iters`: as specified
- `model.micro_batch_size`: maximize within GPU memory
- `model.global_batch_size`: as specified (affects learning dynamics)
- Enable checkpointing, evaluation, and logging at appropriate intervals

---

## Config Generation Template

When generating a new training config, use this structure for the task-level YAML:

### System Section

```yaml
system:
  tensor_model_parallel_size: <TP>
  pipeline_model_parallel_size: <PP>
  context_parallel_size: 1
  use_distributed_optimizer: true
  precision:
    bf16: true
  logging:
    log_interval: 1
    tensorboard_log_interval: 1
    wandb_project: null
    wandb_exp_name: null
  checkpoint:
    save_interval: <save_interval>
    load: ${experiment.load}
    ckpt_format: ${experiment.ckpt_format}
```

### Model Section

```yaml
model:
  num_layers: <from source>
  hidden_size: <from source>
  num_attention_heads: <from source>
  num_query_groups: <from source>
  ffn_hidden_size: <from source>
  seq_length: <from source>
  max_position_embeddings: <from source>
  group_query_attention: true
  swiglu: true
  normalization: RMSNorm
  position_embedding_type: rope
  rotary_base: <from source>
  untie_embeddings_and_output_weights: <from source>
  train_iters: <from recipe>
  micro_batch_size: <from memory or default>
  global_batch_size: <from recipe>
  transformer_impl: transformer_engine
  # Optimizer
  lr: <from recipe>
  min_lr: <from recipe>
  lr_decay_style: cosine
  weight_decay: <from recipe>
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0
  lr_warmup_iters: <from recipe>
```

### Data Section

```yaml
data:
  data_path: <path_to_preprocessed_data>
  split: 1
  no_mmap_bin_files: true
  reset_position_ids: true
  reset_attention_mask: true
  tokenizer:
    tokenizer_type: <type>
    tokenizer_path: <path>
    vocab_size: <from config.json>
    make_vocab_size_divisible_by: 64
```

### Tokenizer Type Mapping

- Qwen models → `QwenTokenizerFS`
- LLaMA 3 models → `Llama3Tokenizer`
- LLaMA 2 / SentencePiece models → `SentencePieceTokenizer`
- Other → check the model's `tokenizer_config.json`

---

## Config Verification Checklist

After generating or modifying a config, verify ALL of the following before handing off to `train-run`:

### Arithmetic Constraints
```python
assert global_batch_size % (micro_batch_size * data_parallel_size) == 0, "batch size not divisible"
assert num_attention_heads % tensor_model_parallel_size == 0, "heads not divisible by TP"
assert num_key_value_heads % tensor_model_parallel_size == 0, "KV heads not divisible by TP"  # GQA
if pipeline_model_parallel_size > 1:
    assert num_layers % pipeline_model_parallel_size == 0, "layers not divisible by PP"
```

### Path Validation
- All paths in config (`data_path`, `vocab_file`, `merge_file`, `tokenizer_path`, `load`) must point to existing files/directories
- Check for placeholder values: `/path/to/`, `FIXME`, `TODO`, `/data/dataset`
- For checkpoint paths (`load`), verify `latest_checkpointed_iteration.txt` exists

### Type Validation
- Read the argparse definitions for non-obvious types before setting values
- Common traps: `--rotary-base` expects int not float, boolean flags vs YAML booleans, string lists vs comma-separated strings
- YAML `1e5` is a float — if the parser expects int, use `100000`

### Cross-Config Consistency
- `vocab_size` in task config must match the tokenizer's actual vocab size
- `seq_length` must match what the data was preprocessed with
- `ckpt_format` in experiment config must match the checkpoint's actual format
- `num_layers`, `hidden_size`, `num_attention_heads` must match the model weights being loaded

---

## Related Skills

- `train-run` — launch training with generated configuration
- `topo-detect` — detect hardware topology for parallelism planning
- `train-model-porter` — port model architecture before configuring training
- `train-data-prep` — prepare training data referenced in configuration
