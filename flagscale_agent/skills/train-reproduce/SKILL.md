---
name: train-reproduce
description: Reproduce training results from open-source implementations, papers, or reference codebases. Establishes a verified
  baseline before migrating to FlagScale. Covers the IMMUTABLE vs ADAPTABLE parameter framework, original artifact reuse,
  quick baseline validation, per-step logging, and experiment log isolation.
keywords:
- reproduce
- reproduction
- baseline
- reference
- original
- paper
- 复现
- 基线
- 原始实现
- 对齐基线
parameters:
- name: source_dir
  description: Directory containing the original implementation source code
- name: output_dir
  description: Directory for reproduction experiment outputs. Use shared storage for multi-node scenarios.
requires:
- train-precision-alignment
suggests:
- train-env-setup
- train-data-prep
constraints:
- id: no_modify_immutable_params
  description: Never change IMMUTABLE parameters (architecture, optimizer type, LR schedule, loss function) during reproduction
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - hidden_size
    - num_layers
    - num_heads
    - learning_rate
    - lr_schedule
    - loss_function
    - optimizer_type
  prompt: Check if the agent is modifying an IMMUTABLE parameter that defines the experiment identity
  correction: Only ADAPTABLE parameters (num_nodes, batch distribution, logging) may be changed. If an immutable param conflicts,
    STOP and ask.
- id: use_original_artifacts
  description: Always use original tokenizer/config/weights from the official release, never regenerate
  trigger:
    tools:
    - shell
    - write_file
    keywords:
    - train_tokenizer
    - train tokenizer
    - sentencepiece_trainer
    - build vocab
  prompt: Check if the agent is regenerating artifacts (tokenizer, vocab) instead of using the official release
  correction: Download the exact tokenizer/config from the original model release. Never regenerate.
- id: experiment_tracking_required
  description: Every training launch must be tracked via workspace_experiment (create before launch, update after completion)
  trigger:
    tools:
    - shell
    - write_file
    - edit_file
    keywords:
    - torchrun
    - python train
    - python -m torch
    - deepspeed
    - accelerate launch
  prompt: Check if the agent is launching a training run without first creating a workspace_experiment entry to track it
  correction: Before launching training, call workspace_experiment create with purpose/hypothesis/config. After training completes,
    call workspace_experiment update with results/reflection/next steps.
context_injection:
  always:
  - 'Core Principle: IMMUTABLE vs ADAPTABLE'
  - Why Reproduction Matters
  by_tool:
    shell:
    - Quick Baseline Validation
    edit_file:
    - Original Artifact Reuse
---
# Reproduce Training Results

Reproduce training results from open-source implementations to establish a verified baseline. This baseline is the foundation for precision alignment when migrating to FlagScale.

## Why Reproduction Matters

Reproduction serves as the BASELINE for migrating to FlagScale. If the baseline is wrong, everything built on top of it is meaningless. Treat reproduction with the highest rigor.

---

## Core Principle: IMMUTABLE vs ADAPTABLE

Before touching any parameter, classify it:

### IMMUTABLE Parameters

These define the experiment — changing any of them means it is no longer the same experiment:

- Model architecture (layers, hidden size, attention heads, FFN size)
- Tokenizer and vocabulary
- Optimizer type and learning rate schedule
- Loss function
- Data processing pipeline and preprocessing logic
- Special tokens and their handling
- Evaluation protocol and metrics
- Weight initialization scheme
- Dropout rates and regularization

### ADAPTABLE Parameters

These are hardware mapping — changing them preserves the experiment on different hardware:

- `num_nodes`, `num_gpus`
- `batch_size` + `gradient_accumulation_steps` (must maintain same effective batch size)
- Data parallelism strategy (DP, DDP, FSDP)
- `num_workers` for data loading
- Logging intervals, checkpoint intervals
- Mixed precision settings (if the original also uses mixed precision)

**Rule**: If unsure whether a parameter is immutable or adaptable, treat it as immutable and ask the user.

**Conflict handling**: If an immutable parameter conflicts with the current setup (e.g., data too small for the original vocab size), STOP and explain the conflict. Let the user decide — never silently adjust.

---

## Original Artifact Reuse

Tokenizers, vocab files, pretrained weights, and config files should be extracted from the original release (official repo, model hub, checkpoints), NOT regenerated. Regenerating on different data produces different artifacts even with the same settings.

- Download the exact tokenizer from the original model release
- Use the exact config file from the original repo
- If pretrained weights are needed, use the official checkpoint
- If the original uses a specific data processing script, use that script

---

## Quick Baseline Validation

When the source model comes with runnable training code, run a quick baseline BEFORE porting. This catches data format issues, missing dependencies, and model bugs early.

**Goal**: Complete a minimal training run in under 1 hour. Not for real training — just to prove the code works end-to-end.

### Step 1: Analyze the Training Script

Read the training entry point and identify all arguments:

```bash
find {source_dir} -maxdepth 2 -name "*.py" | xargs grep -l "if __name__" | head -10
find {source_dir} -maxdepth 2 -name "*.sh" | xargs grep -l "python\|torchrun" | head -10
grep -A 2 "add_argument\|argparse\|@click\|typer" <train_script> | head -80
```

Categorize each argument:

| Category | Examples | Action |
|----------|---------|--------|
| Essential for training | `--data_path`, `--model_config`, `--lr` | Keep, adjust values for quick run |
| Scale controls | `--epochs`, `--max_steps`, `--max_seq_length` | Minimize (epochs=1, max_steps=100-500) |
| Non-essential IO | `--wandb`, `--tensorboard`, `--save_interval` | Disable or set to large interval |
| Visualization | `--plot`, `--visualize`, `--generate_samples` | Disable entirely |
| Evaluation | `--eval`, `--eval_interval` | Disable or run once at end |
| Distributed | `--nproc_per_node`, `--nnodes` | Single-node all-GPU |
| Checkpointing | `--save_steps`, `--save_total_limit` | Save once at end or disable |

### Step 2: Prepare Minimal Data

Check what data format the training script expects:

```bash
grep -rn "Dataset\|DataLoader\|load_dataset\|read_csv\|jsonl\|parquet" <train_script> | head -20
```

Present data options to the user with concrete sizes and tradeoffs:

```
Data options for quick verification:
  A. [repo demo data] — 50MB, ready to use, may not cover all code paths
  B. [smallest real subset] — 350MB download, real data, covers full pipeline
  C. [synthetic data] — 0 download, instant, but only validates code runs
  Recommend: B (smallest real subset that exercises the full pipeline)
  Which do you prefer?
```

Do NOT start downloading multi-GB datasets without user confirmation.

For synthetic text data (if user chooses):
```python
import json
with open("mini_train.jsonl", "w") as f:
    for i in range(100):
        f.write(json.dumps({"text": f"This is sample document {i}. " * 20}) + "\n")
```

### Step 3: Construct the Quick-Run Command

```bash
NPROC=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node=$NPROC --nnodes=1 <train_script> \
    --data_path <mini_data> \
    --output_dir {output_dir} \
    --epochs 1 \
    --max_steps 100 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_steps 999999 \
    --eval_steps 999999 \
    --logging_steps 1 \
    --no_wandb \
    --num_workers 2
```

Key principles:
- Single-node all-GPU distributed training — validates distributed code paths
- If the script has its own launcher (e.g., `deepspeed`, `accelerate launch`), use that instead
- **logging_steps=1 (every step)** — critical for precision alignment later
- 100-500 steps max
- Disable ALL non-training operations

### Step 4: Ensure Per-Step Logging

Many training scripts only log per-epoch by default. This is NOT acceptable — we need per-step loss for precision alignment.

Check the training loop:
```bash
grep -n "log\|print\|logger\|wandb.log\|writer.add" <train_script> | grep -i "loss\|step\|train"
```

If logging is per-epoch only, patch it to log every step:
```python
# Add to training loop:
print(f"step {global_step} | loss {loss.item():.6f}")
```

The output format must include at minimum: **step number** and **loss value** per step. Grad norm is also useful if available.

### Step 4.5: Add Diagnostic Prints for Migration Comparison

Reproduction isn't just about "does it run" — it's the baseline you'll compare against during migration. Add prints that capture information you'll need later for alignment:

- **Intermediate tensor shapes and dtypes** at component boundaries (e.g., encoder output, connector output, decoder input) — so you can verify the migrated model produces the same shapes
- **Key tensor statistics** (mean, std, min, max) at a few critical points in the forward pass — so you can spot numerical divergence early during migration
- **Checkpoint key names and shapes** — dump the state_dict structure so you have a reference for writing the checkpoint converter
- **Config values that affect computation** — log the resolved values of hidden_size, num_heads, ffn_hidden_size, vocab_size, etc. as the model sees them

These prints only need to run for the first few steps. Gate them with `if step < 5` or similar. The goal is to produce a reference snapshot that makes migration comparison straightforward — instead of re-running the reproduction later when you realize you need a specific piece of information.

### Step 5: Run and Verify

```bash
torchrun --nproc_per_node=$NPROC --nnodes=1 <train_script> <flags> 2>&1 | tee {output_dir}/baseline.log
```

Check:
1. **Starts without error** — all imports resolve, data loads, model initializes
2. **Per-step loss is printed** — verify every step has a loss line
3. **Loss decreases** — even slightly over 100 steps confirms forward/backward pass works
4. **No NaN/Inf** — gradient flow is healthy
5. **Memory usage** — note peak GPU memory for later parallelism planning

```bash
grep -i "loss" {output_dir}/baseline.log | head -50
grep -i "nan\|inf\|error" {output_dir}/baseline.log
```

### Step 6: Record Baseline

Save the baseline results for precision alignment after porting:

```bash
grep -i "loss" {output_dir}/baseline.log > {output_dir}/baseline_loss_curve.txt
```

Record to memory:
- Per-step loss values for the first 50-100 steps (alignment target)
- Peak GPU memory usage per GPU
- Time per step
- Exact command and data used (for reproducibility)
- Any patches made to the training script

**If the baseline fails**: Fix the issue in the original code first. Do NOT proceed to porting a broken implementation.

**If no training code exists** (paper-only or HuggingFace weights-only): Skip baseline validation and proceed directly to model porting.

---

## Experiment Log Isolation

Every reproduction experiment MUST have its own isolated directory. This is non-negotiable — mixing logs from different experiments makes results unverifiable.

### Directory Structure

```
{output_dir}/
├── README.md              # Purpose, config, key results
├── config/                # Copy of all config files used
├── logs/                  # Training logs
│   ├── baseline.log       # Full training output
│   └── loss_curve.txt     # Extracted per-step loss
├── checkpoints/           # Saved checkpoints (if any)
└── metrics/               # Extracted metrics for comparison
```

### README Template

Every experiment directory must contain a README with:
- Date and purpose of the experiment
- Exact command used to run
- Hardware description (GPU model, count)
- Key results (final loss, steps completed, time taken)
- Any modifications made to the original code
- Comparison with expected results (if available)

### Isolation Rules

- Reproduction experiments, FlagScale migration verification, production training, and precision alignment experiments MUST use separate directories
- Never reuse an experiment directory for a different purpose
- If re-running an experiment, create a new directory (append timestamp or version)
- Keep all experiment directories until the user explicitly says to clean up

---

## Hardware Scaling

When reproducing on different hardware than the original:

### Batch Size Scaling

Maintain the same effective batch size:
```
effective_batch_size = micro_batch_size × gradient_accumulation_steps × data_parallel_size
```

If the original uses 8 GPUs with batch_size=32 and grad_accum=4:
- effective_batch_size = 32 × 4 × 8 = 1024
- On 4 GPUs: batch_size=32, grad_accum=8 (or batch_size=16, grad_accum=16)

### Learning Rate Scaling

If changing effective batch size (not recommended for reproduction):
- Linear scaling rule: `new_lr = original_lr × (new_batch_size / original_batch_size)`
- This is an approximation — for strict reproduction, keep the same effective batch size

---

## Common Frameworks

### HuggingFace Trainer
```bash
python <script> --output_dir {output_dir} --max_steps 100 --logging_steps 1 --save_steps 999999 --report_to none
```

### DeepSpeed
```bash
deepspeed <script> --deepspeed <ds_config.json> --max_steps 100 --logging_steps 1
```

### Custom PyTorch
Look for the training loop and add per-step logging if not present.

### Fairseq
```bash
fairseq-train <data_dir> --max-update 100 --log-interval 1 --save-interval-updates 999999
```

---

## Next Steps

After successful reproduction:
1. Compare reproduction results with published results (if available)
2. Use the baseline loss curve for precision alignment when migrating to FlagScale
3. Load the `train-precision-alignment` skill for detailed alignment methodology

---

## Related Skills

- `train-model-porter` — port the model to FlagScale after establishing baseline
- `train-precision-alignment` — align FlagScale implementation against reproduction baseline
- `train-env-setup` — set up environment for running the original implementation
- `train-data-prep` — prepare data for reproduction experiments
