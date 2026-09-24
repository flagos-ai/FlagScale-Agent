# 第19章：v0.18.2 结构变化与 FlagScale 训练代码归属 (Megatron-LM-FL e15cb6928)

> 依据：Megatron-LM-FL `c024887..e15cb6928` diff 实测（357 files, +42806/−18771），
> 并经 FlagScale 侧源码交叉验证。

## 1. 两仓职责分工（最常犯错的点）

- **训练代码（training/legacy）不在 pip 包里**。`pyproject.toml` 的 packages.find.include
  恰好是 `megatron.core*` + `megatron.plugin*`；`megatron/training`、`megatron/legacy`、
  `megatron/rl`、`post_training` 通过 PYTHONPATH 下游解析。
- **FlagScale 训练侧 = `flagscale/train/megatron/`**（从 upstream training 派生 + FlagScale 定制，
  带 `FlagScale Begin/End` 特性块）。运行时经 PYTHONPATH 与 MC-FL 配合。
- 因此：**改训练循环/入口 → FlagScale 仓；改 core/plugin → Megatron-LM-FL 仓**，不要在
  FlagScale 里 shim core 的问题，也不要把 core 弯成 FlagScale 特有形状。

## 2. FlagScale 侧训练代码结构 (实测)

```
flagscale/train/megatron/
├── train_gpt.py                    # GPT 训练入口 (含 train_step 数据面)
├── model_specific_batch.py         # get_batch / BATCH_KEYS (模型特化 batch 组装)
└── ...                             # training.py 之外的参数/模型/provider 文件
```

`get_batch()` 返回 **10 键**（megatron 训练数据契约，实测）：
`tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, cu_seqlens_padded,
max_seqlen, local_cp_size, hybrid_cp_group`

- `BATCH_KEYS` 与 `train_gpt.py` 现在都在 **FlagScale 侧**（`flagscale/train/megatron/`）；
  MC-FL 只保留 `megatron/training/training.py`。旧文档若引用 `megatron/training/train_gpt.py`
  需改指 FlagScale 侧。

## 3. megatron/training/ 目录变化

- `utils.py` 拆分为 **`utils/` 包**（`log_utils.py`、`common_utils.py` 等，实测 291 行搬家）。
- 新增 `vocab_utils.py`、`argument_utils.py`、`dgrad_logging.py`、`ft_integration.py`、
  `inprocess_restart.py`、`one_logger_utils.py`、`wandb_utils.py` 等。
- `yaml_arguments.py` 仍在（+22/-）。

## 4. megatron/core/ 新增模块 (v0.18.2)

| 模块 | 说明 |
|------|------|
| `transformer/cuda_graph_config.py` | CUDA Graph 归一化/迁移 API（详见第18章） |
| `models/hybrid/` | hybrid model/block/layer_allocation/layer_specs（dense+Mamba 混合） |
| `models/mimo/` | 多模态 MIMO（comm/colocated_communicator） |
| `datasets/data_schedule.py` | 数据调度 |
| `optimizer/emerging_optimizers.py`、`optimizer/param_layout.py` | 新优化器/参数布局 |
| `fault_injector.py` | 故障注入 |
| `distributed/fsdp/` (mcore_fsdp) | mcore FSDP adapter（distributed_data_parallel_config、fully_shard、param_and_grad_buffer 等） |
| `energy_monitor.py` | 能耗监控 |
| `fp4_utils.py` | FP4 相关工具 |
| `post_training/modelopt/hybrid/` | hybrid 模型 post-training |
| `extensions/transformer_engine_spec_provider.py` | TE spec 提供者 |

## 5. 其他实测 API 变化

- `broadcast_data`: **v0.18.2 位于 `megatron/core/tensor_parallel/data.py`**（megatron/training 下无）。
- `set_streams` (pipeline_parallel/utils.py L365): `set_streams(comm_stream=None, high_priority=False)`。
- chunked cross-entropy（#126）: 配置 `chunked_cross_entropy: bool = False` +
  `cross_entropy_chunk_size: int = 4096`（`model_parallel_config.py` L245/L255）；逐 chunk fp32
  化降峰值显存，backward 按 chunk 重算 softmax（activation-recompute 风格）。
- MoE: `moe_pad_experts_for_cuda_graph_inference`（推理 padding + drop token）。

## 6. 何时读本章

- 找训练入口/batch 组装代码却按旧路径找不到时（先想 FlagScale 侧）。
- 升级 FlagScale 训练代码适配 v0.18.2 时（结构对照 + get_batch 契约）。
- 评估 v0.18.2 新能力（hybrid/mimo/fsdp adapter/chunked CE）能否用于任务时。
