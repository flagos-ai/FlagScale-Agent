---
name: train-precision-alignment
description: Systematically align training precision across three scenarios — model migration (native→FlagScale), internal
  iteration (self-regression), and hardware migration (NVIDIA→new hardware). Progressive 6-level elimination from structure
  to forward/backward.
keywords:
- alignment
- precision
- accuracy
- loss
- divergence
- 精度对齐
- 对齐
- loss对比
- 前向对齐
- 反向对齐
- 复现
- reproduce
- cross-framework
- cross-hardware
- deterministic
- spike
- gradient
parameters:
- name: source_framework
  description: The reference framework/hardware (ground truth side)
- name: target_framework
  description: The framework/hardware being aligned
- name: model_name
  description: Model name being aligned
- name: work_dir
  description: Working directory for alignment artifacts
requires: []
suggests:
- train-run
- train-monitor
workflow:
  trigger:
    keywords:
    - precision alignment
    - loss alignment
    - 精度对齐
    - 前向对齐
    - 反向对齐
    keywords_in_same_input:
    - - align
      - loss
    - - precision
      - compare
    - - 对齐
      - 精度
  stages:
  - id: structure
    name: 'Level 1: Structure Check'
    description: Compare model structure — param names, shapes, layer count
    depends_on: []
  - id: hyperparams
    name: 'Level 2: Hyperparameter Alignment'
    description: Verify all hyperparams match between reference and target
    depends_on:
    - structure
  - id: data
    name: 'Level 3: Data Pipeline'
    description: Verify identical data feeding — same tokens, same order
    depends_on:
    - hyperparams
  - id: init
    name: 'Level 4: Initialization'
    description: Load identical weights, verify tensor equality
    depends_on:
    - data
  - id: loss_curve
    name: 'Level 5: Loss Curve Comparison'
    description: Run both, compare loss curves step-by-step
    depends_on:
    - init
  - id: forward_backward
    name: 'Level 6: Forward/Backward Debugging'
    description: If loss diverges, instrument forward/backward to find divergence point
    depends_on:
    - loss_curve
constraints:
- id: no_skip_levels
  description: Never skip alignment levels — each level eliminates one category of variables
  trigger:
    tools:
    - shell
    - write_file
    keywords:
    - loss_curve
    - loss comparison
    - forward_backward
    - Level 5
    - Level 6
  prompt: Check if the agent is skipping alignment levels (e.g. jumping to loss comparison without verifying structure/hyperparams/data/init)
  correction: Complete each level in order. Skipping levels means uncontrolled variables.
- id: one_variable_at_a_time
  description: Each alignment experiment changes exactly one thing
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - alignment experiment
    - align config
    - alignment config
    - exp00
  prompt: Check if multiple variables are being changed simultaneously in an alignment experiment
  correction: Change exactly one variable per experiment. Multiple changes make it impossible to identify the cause.
- id: align_against_reproduced
  description: Align against reproduced baseline, not paper values
  trigger:
    tools:
    - shell
    - write_file
    - edit_file
    keywords:
    - paper reported
    - paper values
    - expected loss
    - reported accuracy
    - claimed performance
  prompt: Check if alignment is being done against paper-reported values instead of a reproduced baseline
  correction: Align against REPRODUCED baseline, never against 'expected' values from papers.
context_injection:
  always:
  - Three Alignment Scenarios
  - Core Principles
  by_tool:
    shell:
    - Experiment Structure
    edit_file:
    - Core Principles
---
# Training Precision Alignment

Systematically verify and align training precision. Progressive, level-by-level — each level eliminates one category of variables. Never skip levels.

## Three Alignment Scenarios

### Scenario A: Model Migration (Native → FlagScale on NVIDIA)
- **Reference**: Native implementation on NVIDIA
- **Target**: FlagScale on NVIDIA
- Same hardware, different framework → strict numerical alignment possible
- One-time. Once aligned, all further work is Scenario B.
- Levels 1-6 apply.

### Scenario B: FlagScale Internal Iteration
- **Reference**: FlagScale's own aligned baseline
- **Target**: FlagScale with new changes (parallelism, TE-FL, FP8)
- Same hardware, same framework, different config → loss curve must not regress
- Ongoing. Primarily Level 5 (loss comparison). Level 6 if regression found.

### Scenario C: Hardware Migration (NVIDIA → new hardware)
- **Reference**: FlagScale on NVIDIA
- **Target**: FlagScale on new hardware (DCU, TPU)
- Same framework, different hardware → strict match may be impossible
- All levels apply. Level 6 focuses on operator-level differences.

## Core Principles

1. **Align against reproduced baseline** — never against "expected" values from papers
2. **Tensor-level verification** — compare intermediate tensors, not just final loss
3. **DEBUG-first** — add diagnostic prints BEFORE launching, not after failure
4. **Shared storage** — all paths must be on shared storage for multi-node
5. **Instrument existing code** — don't write standalone alignment scripts; hook into training loops
6. **Constraint elimination** — final accuracy = structure × hyperparams × data × init × computation. Isolate each.
7. **One variable at a time** — each experiment changes exactly one thing

## Experiment Structure

```
{work_dir}/experiments/
├── exp001_structure_check/    # Level 1
│   ├── README.md              # what, why, config, conclusion
│   ├── ref_params.txt
│   └── tgt_params.txt
├── exp002_hyperparam_align/   # Level 2
└── ...
```

Every experiment: isolated directory, documented config, explicit conclusion.

---

## Level 1: Model Structure Alignment

Verify parameter-level equivalence between reference and target.

**Method**: Print `named_parameters()` on both sides. Build fusion mapping table (e.g., separate q/k/v → fused qkv). Verify total element count matches.

**Common issues**: Padded vocab size, tied weights, bias presence flags, fused vs unfused layers.

**Pass criteria**: All parameters have 1:1 mapping, total element count identical.

---

## Level 2: Hyperparameter Alignment

Unify ALL hyperparameters that affect training dynamics.

**Categories to check**: Optimizer (type, betas, epsilon, weight_decay), LR schedule (type, peak, min, warmup, decay), Batch size (micro, gradient_accum, global), Regularization (grad clip, dropout), Parallelism (DP, TP, PP, ZeRO), Precision (dtype, loss scaling, grad accum dtype), Duration (steps, seed), Implementation (FA, TE, fused kernels).

**Critical formula**: `global_batch_size = micro_batch_size × DP × gradient_accumulation_steps`

**Difference classification**:
- Mathematically equivalent (ZeRO-1 vs AllReduce) → document only
- Numerically different (FA vs native attention) → acceptable, document
- Semantically different (epsilon 1e-8 vs 1e-18) → must fix

**Pass criteria**: All semantically different items resolved, batch size equation verified.

---

## Level 3: Data Pipeline Alignment

Ensure each DP rank at each step receives exactly the same input data in both systems.

**Method**: Save `input_ids` for first N steps on both sides, compare tensor equality.

**Key checks**: Same tokenizer, same shuffle seed, same data order, same padding/truncation, same DP rank assignment.

**Pass criteria**: Bit-exact match on input tensors for N steps.

---

## Level 4: Weight Initialization Alignment

Ensure both systems start from identical weights.

**Method**: Save initial state_dict from both, compare with `torch.allclose(atol=0, rtol=0)`.

**For checkpoint loading**: Convert checkpoint, load in both systems, compare loaded weights.

**Pass criteria**: Bit-exact match on all parameters after initialization.

---

## Level 5: Loss/Evaluation Alignment

Compare training dynamics over N steps.

**Alignment modes**:
- **Strict**: Per-step loss matches within tolerance (for same-hardware, deterministic)
- **Relaxed**: Statistical bounds on relative error (for cross-hardware or non-deterministic ops)
- **Trend**: Same convergence shape and final value (for production configs with dropout)

**Verification strategy** (cheapest first):
1. Small-scale deterministic (same init, same data, dropout=0, small cluster)
2. Large-scale deterministic (full cluster, dropout=0)
3. End-to-end with non-determinism (production config)
4. Downstream task evaluation (generate/classify from checkpoint)

**Divergence pattern diagnosis**:

| Pattern | Likely Cause | Action |
|---------|-------------|--------|
| From step 0 | Init mismatch | Re-verify Level 4 |
| Linear growth | Small numerical diff | Acceptable if within tolerance |
| Exponential growth | Bug in computation | Level 6 |
| One flat, other descends | Broken gradient/operator | Level 6 |
| Sudden spike then recovery | Numerical instability | Check loss scaling, grad clip |
| Periodic large errors | Data or LR schedule mismatch | Re-verify Levels 2-3 |

---

## Level 6: Forward/Backward Alignment

Locate the exact layer and operator where computation diverges.

**When to use**: Loss curves diverge beyond tolerance, or one system fails to learn.

**Method 1: Controlled experiments** (change one variable at a time):
- Switch attention kernel on reference → isolates attention implementation
- Load reference weights in target → isolates init
- Switch to high-precision variant of suspected op → isolates operator precision

**Method 2: Layer-by-layer hooks** — register forward hooks on both sides, compare per-layer statistics (sum, mean, std, max, norm). Find first layer where divergence appears.

**Divergence metrics** (use multiple):
- Max absolute diff — catches outliers
- Mean absolute diff — average error level
- Cosine similarity — direction alignment
- Gradient ratio — backward bugs (2x = likely bug)

**Binary search**: Once divergence layer found, add hooks to its submodules to narrow further.

---

## Determinism Controls

For strict alignment, enable full determinism:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Disable all stochastic ops: `attention_dropout=0`, `hidden_dropout=0`.

**Cross-hardware limits**: Different hardware has inherent floating-point differences (compiler, operator fusion, accumulation order). When strict match is impossible, convergence trend + downstream quality is the acceptance standard.

---

## Known Sources of Numerical Divergence

- **Flash Attention**: ~10x more numeric deviation than baseline at BF16 (tile-based recomputation)
- **Compiler differences**: NVCC vs HIP generate different instruction sequences → different rounding
- **AllReduce non-determinism**: Sum order varies → floating-point non-associativity
- **Loss spikes**: Check grad norm, loss scaling, data anomalies before blaming alignment

---

## Rules

1. Never skip levels. Levels 1-4 are cheap and eliminate most issues.
2. Do not modify reference code. Reference is ground truth.
3. One variable at a time when diagnosing.
4. Classify differences — not every difference is a bug.
5. Downstream task is the ultimate judge.
6. Save all artifacts to `{work_dir}`.
7. Cross-hardware: check vendor high-precision variants before concluding it's a bug.

## References

- [Is Flash Attention Stable?](https://arxiv.org/abs/2405.02803)
- [Finding Numerical Differences Between NVIDIA and AMD GPUs](https://arxiv.org/abs/2410.09172)
- [Joint Training on AMD and NVIDIA GPUs](https://arxiv.org/abs/2602.18007)
- [NVIDIA Framework Reproducibility](https://github.com/NVIDIA/framework-reproducibility)

## Related Skills

- `train-reproduce` — establish verified baseline before alignment
- `train-model-porter` — port model architecture (alignment verifies the port)
- `train-run` — launch training runs for comparison
- `train-monitor` — monitor metrics during alignment experiments
