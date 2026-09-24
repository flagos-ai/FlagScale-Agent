<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 训练状态恢复与数值可比性

性能、模型更新等价与完整续训等价是不同结论。本文解释加载或比较状态时的边界；一般计时与质量证据见 [训练测量](measurement-and-records.md)。

## 初态、更新与完整恢复

| 证据 | 能说明什么 |
| --- | --- |
| 冷启动或 checkpoint 加载成功 | 对应初始化/加载路径可运行，不证明全部状态已恢复 |
| `model_only` 对齐 | 同初态、同逻辑 batch，模型参数及更新按真实分片对齐；不包含未验证的 optimizer、scheduler、RNG 或数据进度 |
| 完整恢复 | “连续训练到 N”与“训练到 K、保存退出、恢复到 N”在同一逻辑边界的必要状态和进度一致 |

完整恢复涉及模型、FP32 主参数、optimizer 动量/步数、scheduler、consumed samples 和实际使用的 RNG。绝对更新进度相同与两段运行的计数恰好相等不同；没有采集到的状态不能视为相等。

## 跨布局与 optimizer schema

TP/PP/EP 变化可能改变参数归属、子优化器组织和保存格式。模型权重能重分片不保证 optimizer 兼容；比较同号 rank 的局部张量也不等于比较同一参数。

某些跨 TP/PP loader 路径会跳过 checkpoint RNG，即使没有设置 `no_load_rng`。请求选项、加载日志和最终状态各自反映不同层次；模型对齐通过不能补足未恢复的组件。

在 `convert_to_ep` 多子优化器加载分支要求 `param_state` 的版本中，源保存 schema 必须提供该字段。Float16 与 distributed optimizer 状态格式不能互换假设；缺字段是恢复格式问题，不足以证明 EP/dispatcher 不支持。新 seed 或冷启动只能验证新配置功能，未经状态对齐不能作为同条件续训对照。

`ChainedOptimizer` 等组合优化器的状态来自全部子优化器，不一定能由面向单个 optimizer 的便利属性表示。只有明确的空参数组且确实没有状态时，组计数不变才有解释；缺失状态不等于空组。非 tensor 的 step、计数和超参数也属于状态。

## LR 与 weight decay 调度

部分 Megatron 版本即使采用 constant weight decay，仍由 `train_iters × GBS` 派生 `wd_incr_steps`。缩短运行只固定 `lr_decay_iters`，仍可能改变恢复时的调度约束。

`use_checkpoint_opt_param_scheduler` 在提供该选项的版本中表示沿用保存的调度；其实际效果由 loader 决定。LR、weight decay 和调度进度均影响更新，override 后重新起算与保持原续训语义不同。

## RNG 与数据迭代器

Python、NumPy、torch CPU、NPU 及并行 RNG tracker 可能分别持有状态。要求精确恢复的随机状态、离散计数不适用浮点权重容差；dropout=0 或短跑权重相同也不能证明必需 RNG 一致。

部分 DataLoader 实现在创建 iterator 时抽取 `base_seed`；没有专用 generator 时可能消耗全局 CPU RNG。因此“加载后、首个更新前”的偏移可能来自 iterator 初始化，其因果关系还取决于实际数量、顺序和 generator 绑定。专用 generator 本身也需要保存/恢复，并保持数据顺序和随机增强语义。

## 对齐证据与观测边界

权重绝对容差若大于一次更新量，只比较更新后参数可能漏掉 optimizer 错误。初态、实际更新差量、主参数/动量、步数及 scheduler 应在同一逻辑边界比较；合理非确定性的统计界限与精确状态相等不能混用，也不能按结果事后放宽门槛。

观测器异常与训练异常是不同来源，快照 schema 不同也可能改变字段含义。用于离线检查的状态应以受限 CPU 反序列化读取；未知重建类型不能自动转为无限制加载。带状态快照/复制的诊断运行不属于无扰动性能样本。
