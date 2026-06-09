---
name: train-parallel-strategy
description: Guide for selecting and configuring parallelism strategies (TP/PP/DP/EP/CP/SP) in Megatron-LM-FL and TransformerEngine-FL.
  Covers data pipeline handling under parallelism, attention strategy selection, and memory estimation. Use when porting models
  to FlagScale or debugging parallelism-related issues.
keywords:
- parallel
- parallelism
- tensor parallel
- pipeline parallel
- data parallel
- expert parallel
- context parallel
- sequence parallel
- TP
- PP
- DP
- EP
- CP
- SP
- OOM
- memory
- get_batch
- attention
- transformer_engine
- MoE
- 并行策略
- 张量并行
- 流水线并行
- 数据并行
- 专家并行
- 显存
parameters:
- name: model_size
  description: Approximate model size (e.g., 7B, 70B, 235B-MoE)
  default: ''
- name: gpu_count
  description: Number of GPUs available
  default: '8'
requires: []
suggests:
- topo-detect
constraints:
- id: tp_pp_divisibility
  description: TP x PP x EP x CP must divide world_size evenly
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - tensor_model_parallel
    - pipeline_model_parallel
    - expert_model_parallel
    - context_parallel
  prompt: Check if the parallelism config satisfies TP*PP*EP*CP divides world_size evenly
  correction: Adjust parallelism dimensions so TP*PP*EP*CP divides total GPU count evenly.
- id: sp_requires_tp
  description: sequence_parallel requires tensor_model_parallel_size > 1
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - sequence_parallel
  prompt: Check if sequence_parallel is enabled without TP > 1
  correction: sequence_parallel only works with TP > 1. Either enable TP or disable SP.
- id: heads_divisible_by_tp
  description: num_attention_heads must be divisible by TP
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - num_attention_heads
    - tensor_model_parallel
  prompt: Check if num_attention_heads is divisible by tensor_model_parallel_size
  correction: num_attention_heads must be evenly divisible by TP. Adjust TP or model config.
- id: memory_estimation_before_launch
  description: Estimate memory before launching training. Only triggers on actual training launch commands (torchrun, flagscale train),
    not on load_skill, workspace_experiment, ps/grep, or other non-launch operations.
  trigger:
    tools:
    - shell
    keywords:
    - torchrun
    - flagscale train
    - python -m torch.distributed
    - deepspeed
    - train.py
    - pretrain.py
  prompt: "SCOPE: shell command that LAUNCHES a training process (torchrun, flagscale train, python train.py, deepspeed).
    NOT in scope: ps/grep/pgrep commands checking process status, load_skill, workspace_experiment, read_file, or any
    non-shell tool. CHECK: Was memory estimation (params*bytes + optimizer states + activations) performed earlier in
    the conversation before this launch command?"
  correction: 'Estimate GPU memory: params*2(bf16) + grads*2 + optimizer*(8/DP). If exceeds GPU memory, increase TP/PP.'
- id: enable_sp_with_tp
  description: Always enable sequence_parallel when TP > 1
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - tensor_model_parallel
    - tensor_model_parallel_size
  prompt: "SCOPE: write_file or edit_file that sets tensor_model_parallel_size > 1 in a YAML config.
    NOT in scope: shell commands (even if they mention TP for validation/calculation purposes).
    CHECK: Is TP > 1 configured WITHOUT sequence_parallel: true in the same config file?"
  correction: 'Always enable sequence_parallel: true with TP > 1. It reduces activation memory at no extra GPU cost.'
context_injection:
  always:
  - Core Principle
  - Parallelism Dimensions — What Each One Does
  by_tool:
    edit_file:
    - Strategy Selection — Decision Tree
    - Memory Estimation
    write_file:
    - Strategy Selection — Decision Tree
    - Memory Estimation
    shell:
    - Memory Estimation
---
# Parallel Strategy for FlagScale / Megatron-LM-FL

Goal: select the right parallelism dimensions, configure them correctly, and handle data/attention so training runs without errors or hangs.

## Core Principle

FlagScale's value is parallelism-powered speedup. When porting a model, the goal is to USE the parallel infrastructure, not bypass it. If something doesn't work with TP/PP/EP, fix the integration — don't fall back to single-GPU or wrapper hacks.

## 1. Parallelism Dimensions — What Each One Does

| Dimension | Config Key | Splits | When to Use |
|-----------|-----------|--------|-------------|
| **TP** (Tensor Parallel) | `tensor_model_parallel_size` | Weight matrices column/row-wise across GPUs | Model too large for one GPU. Always try first. |
| **PP** (Pipeline Parallel) | `pipeline_model_parallel_size` | Layers across GPU groups | Model still OOM after max TP, or very deep models (>60 layers). |
| **DP** (Data Parallel) | Implicit: `world_size / (TP * PP * EP)` | Batch across GPU groups | Always present. More DP = higher throughput. |
| **EP** (Expert Parallel) | `expert_model_parallel_size` | MoE experts across GPUs | MoE models only. EP ≤ num_experts. |
| **CP** (Context Parallel) | `context_parallel_size` | Sequence length across GPUs | Very long sequences (>8K). Rarely needed for standard training. |
| **SP** (Sequence Parallel) | `sequence_parallel: true` | Activations along sequence dim during LayerNorm/Dropout | Always enable with TP. Reduces activation memory. No extra GPUs needed. |

**Constraint**: `TP × PP × EP × CP` must divide `world_size` evenly. Remaining GPUs become DP replicas.

## 2. Strategy Selection — Decision Tree

```
START: Estimate model memory (Section 5)
  │
  ├─ Fits on 1 GPU? → TP=1, PP=1, maximize DP
  │
  ├─ Fits with TP? → Set TP to minimum that fits (2, 4, or 8)
  │   └─ Enable sequence_parallel: true (always with TP>1)
  │
  ├─ Still OOM with TP=8? → Add PP
  │   └─ PP = ceil(model_layers / layers_per_stage)
  │   └─ Use decoder_first_pipeline_num_layers for uneven splits
  │
  ├─ MoE model? → Add EP
  │   └─ EP ≤ num_experts, EP should divide num_experts evenly
  │   └─ EP reduces per-GPU expert count: local_experts = num_experts / EP
  │
  └─ Long sequences (>8K)? → Consider CP=2
      └─ CP splits sequence, requires reset_position_ids + reset_attention_mask
```

### Quick Reference from Real Configs

| Model | Size | TP | PP | EP | CP | SP | Notes |
|-------|------|----|----|----|----|-----|-------|
| Qwen3 | 0.6B | 1 | 1 | - | 1 | yes | Small model, DP only |
| Qwen3 | 32B | 8 | 1 | - | 1 | yes | Full TP on 8 GPUs |
| Qwen3 | 235B-A22B (MoE) | 2 | 2 | 2 | 1 | yes | TP+PP+EP for MoE |
| DeepSeek-V3 | 16B-A3B (MoE) | 1 | 2 | 4 | 1 | yes | PP+EP, MLA attention |

## 3. Data Pipeline Under Parallelism

This is where most porting failures happen. Megatron's data pipeline is tightly coupled to parallelism.

### 3.1 How `get_batch` Works in Megatron

Megatron's `get_batch_on_this_tp_rank()` (in `megatron/training/utils.py`) handles the TP/PP data distribution:

```
DataLoader (DP rank) → get_batch_on_this_tp_rank() → model forward
                              │
                              ├─ PP first stage: loads from DataLoader
                              ├─ PP other stages: receives from previous stage (no data needed)
                              └─ TP: rank 0 broadcasts to other TP ranks
```

**Key rules for custom data pipelines:**

1. **DP**: Each DP rank gets a different data shard. Megatron handles this via `MegatronPretrainingSampler` which shards by DP rank. If you write a custom dataset, it must be shardable — no global shuffling that differs across ranks.

2. **TP**: Only TP rank 0 loads data, then broadcasts to other TP ranks. Your `get_batch` must NOT do different things on different TP ranks. Return the same tensor shapes on all TP ranks — the broadcast handles the rest.

3. **PP**: Only the first pipeline stage needs input data (tokens, position_ids, attention_mask). Other stages receive activations from the previous stage. Your `get_batch` should return `None` for non-first stages, or Megatron handles this automatically if you use the standard pipeline.

4. **EP**: Data pipeline is unaffected. EP only splits expert weights, not data.

5. **CP**: Sequence is split across CP ranks. Requires `reset_position_ids: true` and `reset_attention_mask: true` in config. The data pipeline must provide full sequences — Megatron splits them internally.

### 3.2 Custom Dataset Integration Checklist

When porting a model with a non-standard dataset (e.g., PackedDataset, interleaved multimodal):

- [ ] Dataset `__len__` returns a consistent value across all ranks
- [ ] Dataset `__getitem__` returns tensors with shapes independent of rank
- [ ] If using packing: `reset_position_ids: true`, `reset_attention_mask: true`
- [ ] If using custom collation: output dict keys match what `get_batch_on_this_tp_rank` expects
- [ ] Test with `--train-iters 20` at target parallelism BEFORE long runs
- [ ] Watch for infinite loops: if your dataset has a custom `repeat` or cycling mechanism, verify it terminates correctly with Megatron's `MegatronPretrainingSampler`

### 3.3 Common Data Pipeline Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hang at first iteration | TP ranks have different data shapes | Ensure uniform shapes, check broadcast |
| NCCL timeout | PP stages waiting for data that never comes | Check `get_batch` returns None for non-first stages |
| Loss is NaN from step 1 | Attention mask wrong under packing | Enable `reset_position_ids` + `reset_attention_mask` |
| Infinite loop, no progress | Custom dataset repeat logic conflicts with Megatron sampler | Use Megatron's built-in data cycling, remove custom repeat |
| Different loss across DP ranks | Non-deterministic data loading | Set seed, ensure sampler shards deterministically |

## 4. Attention Strategy

### 4.1 Decision Order (NEVER skip to custom implementation)

```
Step 1: Use TransformerEngine-FL built-in attention
        ├─ Set transformer_impl: transformer_engine in config
        ├─ TE handles FlashAttention, fused kernels, TP-aware QKV projection
        └─ Works for: standard MHA, GQA (num_query_groups), RoPE

Step 2: If model has non-standard attention (MLA, sliding window, cross-attention)
        ├─ Check if Megatron-LM-FL already supports it:
        │   grep -r "multi_latent_attention\|sliding_window\|cross_attention" megatron/
        ├─ If supported: use the existing config flags
        └─ If not: adapt the existing attention module (Step 3)

Step 3: Adapt existing attention — DO NOT write from scratch
        ├─ Subclass or modify the existing SelfAttention/CrossAttention
        ├─ Keep TP-aware linear layers (ColumnParallelLinear, RowParallelLinear)
        ├─ Keep the existing forward() signature — add parameters, don't remove
        └─ Test: verify output matches HF model at TP=1 first, then test TP>1

Step 4: ONLY if Step 3 is impossible (fundamentally different compute)
        ├─ Write custom attention with full TP support
        ├─ Use ColumnParallelLinear for Q/K/V projections
        ├─ Use RowParallelLinear for output projection
        └─ Handle attention mask correctly under CP if used
```

### 4.2 TransformerEngine-FL Config Flags

```yaml
model:
  transformer_impl: transformer_engine    # REQUIRED — enables TE backend
  # TE-FL extensions (optional, for FlagOS integration):
  te_fl_prefer: flagos                    # prefer flagos:triton backend
  te_fl_per_op: "rmsnorm_fwd=vendor:acme|flagos;rope_fwd=flagos|reference"
```

### 4.3 Attention Variants in FlagScale

| Variant | Config Flag | Example Model |
|---------|------------|---------------|
| Standard MHA | (default) | LLaMA 2 |
| GQA | `num_query_groups: N` | Qwen3, LLaMA 3 |
| MLA (Multi-Latent) | `multi_latent_attention: true` + `kv_lora_rank`, `qk_head_dim`, `v_head_dim` | DeepSeek-V3 |
| QK LayerNorm | `qk_layernorm: true` | Qwen3 (large), DeepSeek-V3 |

### 4.4 Common Attention Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `TypeError: __init__() got unexpected keyword argument` | Megatron base class API changed | Read the CURRENT `__init__` signature of the class you're subclassing |
| Double projection (loss explodes) | Calling `get_query_key_value_tensors()` which internally calls `linear_qkv()` again | Read the method body — it may already project |
| Shape mismatch at TP>1 | QKV split doesn't account for `num_query_groups` under TP | `local_kv_heads = num_query_groups // TP`, not `num_attention_heads // TP` |
| `torch.compile` + `flex_attention` crash | Inductor lowering fails in Megatron context | Use TE's built-in attention instead of custom flex_attention |

## 5. Memory Estimation

Quick formula for transformer models (bf16):

```
Per-GPU model memory ≈ (2 × params_B × 1e9 / TP / PP) bytes  [bf16 = 2 bytes/param]
Per-GPU optimizer memory ≈ (12 × params_B × 1e9 / TP / PP / DP) bytes  [AdamW fp32 states]
Per-GPU activation memory ≈ f(seq_len, hidden_size, batch_size, num_layers/PP)
```

**OOM debugging order** (root cause first, not parallelism first):

1. Check if gradients are in fp32 unnecessarily (`accumulate_allreduce_grads_in_fp32`)
2. Check activation checkpointing (`recompute_granularity: selective` or `full`)
3. Check `use_distributed_optimizer: true` (shards optimizer states across DP)
4. THEN increase TP/PP if still OOM

## 6. MoE-Specific Parallelism

MoE models add Expert Parallelism (EP) on top of TP/PP/DP.

### 6.1 Key Config Fields

```yaml
system:
  expert_model_parallel_size: 4    # split experts across 4 GPUs
model:
  num_experts: 128                 # total experts
  moe_router_topk: 8              # experts activated per token
  moe_ffn_hidden_size: 1536       # per-expert FFN size (smaller than dense ffn_hidden_size)
  moe_grouped_gemm: true          # fuse expert computation
  moe_token_dispatcher_type: "alltoall"  # communication pattern
  moe_router_load_balancing_type: "aux_loss"  # or "seq_aux_loss"
  moe_aux_loss_coeff: 0.001
  # Shared experts (DeepSeek-V3 style):
  moe_shared_expert_intermediate_size: 2816
  # Sparse layers pattern:
  moe_layer_freq: "[0]+[1]*26"    # first layer dense, rest MoE
```

### 6.2 EP Sizing

- `local_experts = num_experts / EP` — must be integer
- EP GPUs form a separate communication group for all-to-all
- EP is orthogonal to TP: you can have TP=2, EP=4 on 8 GPUs (DP=1)
- Memory per GPU: only `local_experts` expert weights, but all-to-all communication increases

## 7. Verification Checklist

Before committing to a long training run, verify parallelism works:

```bash
# 1. Dry run: 20 iterations at target parallelism
python -m flagscale.train --config ... --train-iters 20

# 2. Check GPU memory is balanced (no single GPU much higher)
nvidia-smi  # during the 2-iteration run

# 3. Verify loss is finite and decreasing
grep "lm loss" <log_file> | head -5

# 4. If porting: compare loss at step 0 with random init baseline
#    Expected: loss ≈ ln(vocab_size) for random init
#    If loading pretrained: loss should be much lower

# 5. Check throughput (tokens/sec/GPU) is reasonable
#    Compare with reference configs in examples/
```

## 8. Troubleshooting Quick Reference

| Problem | First Check | Second Check |
|---------|------------|--------------|
| OOM | `nvidia-smi` — which memory type? Model or activation? | Try `recompute_granularity: selective` before adding TP |
| NCCL timeout | Are all ranks reaching the same collective? | Check PP stage assignment, data pipeline |
| Loss NaN | Attention mask correct? | Gradient clipping enabled? (`clip_grad: 1.0`) |
| Throughput too low | TP communication overhead? | Try reducing TP, increasing DP |
| Hang after N steps | Gradient sync deadlock? | Check `overlap_grad_reduce` compatibility |
