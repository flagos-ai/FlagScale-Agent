# Example: Train Qwen2.5 7B with FlagScale-Agent

This example shows a typical conversation where FlagScale-Agent sets up and runs a Qwen2.5 7B training job.

## Prerequisites
- 8x NVIDIA A100/H100 GPUs
- FlagScale installed
- HuggingFace model weights downloaded

## Conversation

### Step 1: Setup and Configuration

```
User: 在8卡A100上训练Qwen2.5 7B，TP=4 DP=2，batch size 4M tokens，
      数据用 /data/pile_text_document，训练10000步。
```

Agent will:
1. Detect hardware topology (NVLink, NUMA)
2. Validate parallelism strategy fits GPU memory
3. Generate FlagScale Hydra config

Generated config (simplified):
```yaml
experiment:
  task_type: train
  runner:
    backend: megatron

train:
  model:
    name: qwen2_7b
    transformer_impl: transformer_engine
    num_layers: 32
    hidden_size: 3584
    num_attention_heads: 28
    seq_length: 4096
    bf16: true

  parallel:
    tensor_model_parallel_size: 4
    data_parallel_size: 2
    sequence_parallel: true

  training:
    global_batch_size: 256    # 256 * 4096 * 4 ≈ 4M tokens
    micro_batch_size: 2
    train_iters: 10000
    lr: 1.5e-4
    min_lr: 1.5e-5
    lr_warmup_iters: 500

  data:
    data_path: /data/pile_text_document
    split: "990,9,1"
```

### Step 2: Launch Training

```
User: 配置看着不错，启动训练吧
```

Agent will:
1. Validate config with `validate_config`
2. Launch training via FlagScale CLI
3. Monitor initial iterations for healthy loss

### Step 3: Monitor Progress

```
User: 训练跑的怎么样了？
```

Agent will:
1. Use `find_latest_log` to locate the loss-printing rank
2. Use `parse_training_metrics` to extract loss trend
3. Report: iteration progress, loss value, throughput (tokens/sec/GPU)

Example output:
```
Training is progressing well:
- Current iteration: 1250/10000 (12.5%)
- Loss: 2.34 (decreasing normally from initial ~10.5 ≈ ln(151936))
- Throughput: 3,850 tokens/sec/GPU
- Estimated remaining: ~5.7 hours
- No anomalies detected (grad norm stable, no NaN)
```

### Step 4: Handle Issues

```
User: loss 突然跳到 NaN 了
```

Agent will:
1. Check stderr logs across all ranks for OOM or NCCL errors
2. Inspect gradient norms before the NaN
3. Check for data corruption at the failing iteration
4. Suggest fixes (reduce LR, enable grad clipping, check data)

---

## Key Takeaways
- Agent auto-detects hardware and suggests optimal parallelism
- Training monitoring is multi-rank aware (checks ALL ranks)
- Health checks compare initial loss against ln(vocab_size) to catch random-output issues
- Memory system remembers findings for future sessions
