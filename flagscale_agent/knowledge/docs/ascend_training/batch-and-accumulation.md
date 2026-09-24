<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend micro-batch、梯度累积与调度

## 批量与更新单位

MBS 是一次 microbatch 前反向的样本数，GBS 是一次 optimizer 更新的全局样本数。常见同构 Megatron 布局下，累积 microbatch 数 `G = GBS / (MBS × DP)`，必须为正整数；DP 指独立数据副本数，不是总卡数，TP/PP/CP rank 不各自再计一份样本。批量定义见 [Megatron batching][batching]，分组推导见 [并行整数约束](parallelism.md)。

固定 GBS/DP 时，增大 MBS 会减少 G。梯度累积把多个 microbatch 的梯度汇入一次更新，不能把“每个 microbatch 更快”与“每次更新更快”混为一谈；框架日志中的 step 也可能指不同单位。

## 计算、显存与通信的权衡

| 变化 | 潜在收益 | 主要代价或边界 |
| --- | --- | --- |
| 增大 MBS | 更大的局部矩阵可能提高算子效率，减少每次更新的 Python/算子下发与重复调用 | 激活和部分 workspace 增大；shape 变化可能改变算子实现或触发重新编译，性能不一定单调。 |
| 减小 MBS | 降低部分激活峰值；更多 microbatch 可改善流水线填充 | 增加调用和调度次数，较小矩阵可能降低计算效率；不直接减少权重或 optimizer 状态常驻量。 |
| 与重计算、布局组合 | 可在算子粒度、激活存活量与重放成本间折中 | 其他布局或重计算配置下的最优 MBS 与 OOM 边界不一定相同。 |

在固定 GBS 的普通累积路径中，DP 梯度同步通常围绕一次更新组织，减少 microbatch 数并不意味着 DP 通信次数等比例减少；TP、PP 或 EP 的逐 microbatch 通信则可能改变消息粒度和次数。具体收益由实际同步位置及 overlap 决定，见 [通信机制](communication.md)。

PP 的启动/排空开销由有效 microbatch 工作摊薄，G 变小时空泡占比可能上升；激活峰值还与 stage、在途 microbatch 和重计算有关。不能仅按 MBS 推断完整更新峰值，也没有跨 NPU 型号、模型与后端通用的最佳 MBS 或固定对齐倍数。

[MindSpeed PP 自动并行][pipeline]提出在启动/冷却阶段使用较小 microbatch、稳态使用更高效粒度的动态方案，说明 microbatch 划分与调度可以共同优化。这需要专门的调度实现；固定整数 `micro_batch_size` 不表达动态序列，社区功能也不自动等于 FlagScale 已接入。

## 当前实现的批量与流水线约束

以下为 FlagScale `a3088a5`、Megatron-LM-FL `1366b6bc1` 对应路径的静态语义，不代表所有 NPU 模型或调度都支持：

- FlagScale `flagscale/train/megatron/training/arguments.py` 在未指定 GBS 时默认使用 `MBS × DP`；`step_batch_size_schedule` 与显式 GBS 互斥。
- Megatron `megatron/core/num_microbatches_calculator.py` 区分配置 GBS 与实际运行 GBS；`decrease_batch_size_if_needed` 可向下取整以适配 MBS×DP。只看请求 GBS 不足以证明批量不变。
- 同一 calculator 的分阶段 batch schedule 按训练进度更新当前 GBS；每阶段仍有整除约束。单阶段的合法 MBS 不一定适合整个 schedule。
- `megatron/core/pipeline_parallel/schedules.py` 的 interleaved 路径以 `q = microbatch_group_size_per_vp_stage` 分组，要求 `PP ≤ q ≤ G`，且 `G % q` 为 0 或不小于 PP；`model_parallel_config.py` 将未指定的 q 默认设为 PP。因此不能统一规定所有 VPP 都只检查 `G % PP == 0`；这些条件也不替代其他调度及模型约束。

packing、动态 CP、自定义 schedule 或按 microbatch 索引设置的重计算可能附带其他条件；标量批量公式不覆盖这些路径的全部语义。

## 固定 GBS 的数值等价边界

同一逻辑 batch 拆分后，只有样本/token 权重、累积缩放与更新时机保持一致，才具有相同的目标梯度语义。[MindSpeed-MM 的 loss 归一化说明][loss]区分 microbatch 均值、样本均值与全局有效 token 均值：不同 microbatch 的有效 token 数不相等时，“先分别求均值再平均”通常不等于全局 token 均值。固定 GBS 本身不能消除这种差异。

同样需要区分两类差异：浮点累加顺序或随机数分配变化可能引起数值偏差；而按 microbatch 计算的批内统计、MoE 辅助损失或容量/丢 token 行为，则可能随样本分组改变目标或有效数据。是否保持语义取决于实际模型，不能通过自动改 loss 归一化、关闭 dropout 或调整路由规则来制造等价。

变长和多模态数据中，相同样本数也不保证相同有效 token 数或 padding 开销；packing 后的样本单位还可能是打包序列。性能口径与质量判断见 [训练测量与正确性](measurement-and-records.md)。

[batching]: https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/nemo_megatron/batching.html
[pipeline]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/automated-pipeline.md
[loss]: https://github.com/Ascend/MindSpeed-MM/blob/master/docs/zh/features/vlm_model_loss_calculate_type.md
