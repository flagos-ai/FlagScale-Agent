<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 激活重计算的边界与代价

## 保存张量与峰值内存

激活重计算以反向阶段的重放换取更少的前向保存张量，主要作用于 activation，不直接减少参数、梯度或 optimizer 状态。边界输入通常仍需保留，反向重放也会产生临时张量和 workspace，因此节省量不等于该模块前向生成的全部张量大小。[checkpoint 原理][checkpoint]

峰值由同一时刻仍存活的张量决定。释放大量张量若没有覆盖真正的峰值，峰值显存可能变化很小；重放还可能把峰值移到 backward 或与通信缓冲重叠的阶段。外层 checkpoint、融合算子和不同保存策略会改变内层重计算的边际收益，各模块的节省量不能直接相加。其他来源见 [显存、分片与卸载](memory-and-sharding.md)。

## 重计算边界的取舍

以下按保存对象解释边界，不将上游模块标签当作已支持的 Ascend 配置；具体实现取决于模型及 Megatron-LM-FL/TE-FL 路径。[Megatron 模块定义][megatron]

| 边界 | 潜在节省 | 主要代价或限制 |
| --- | --- | --- |
| Attention 核心 | 显式保存的 attention score/probability 等中间量 | 融合 attention 可能已经避免保存这些大张量，额外重计算收益因而较小；CP 路径还可能重放通信。 |
| 投影、归一化或激活函数等局部区域 | 展开后的表示或局部输出 | 只丢弃输出再恢复与重跑整个模块的代价不同；收益取决于实际保存张量和边界输入是否仍驻留。 |
| Dense/Expert MLP | FFN 中间激活 | 整个 MLP 重放通常包含 GEMM；仅恢复激活函数输出的实现不能按整层重放估算。 |
| 整层或整个 MoE 区域 | 更大范围的保存激活 | 可能重跑 router、dispatch/combine 和专家计算，也延长重放时间；更宽边界不保证更低的完整更新峰值。 |

较有价值的边界通常覆盖峰值时仍存活、恢复代价相对较低的张量；只按模块大小或固定 attention→MLP 顺序推导优先级，可能错过真实瓶颈。减少重计算可降低重放成本，增加重计算也可能为更大的 MBS 腾出空间，两者都存在有效的时间/容量权衡。

## 层数、分组与当前配置语义

Megatron-LM-FL `1366b6bc1` 的 `megatron/plugin/Ascend/transformer/transformer_config.py` 区分以下配置；实际覆盖受 PP/VPP 本地层划分、模型实现及分阶段覆盖影响：

| 配置 | 含义与边界 |
| --- | --- |
| `selective` + `recompute_modules` | 选择模块边界；未指定模块时默认为 `core_attn`。该模式不设置 `recompute_num_layers`，FlagScale `a3088a5` 的 `training/arguments.py` 也要求不设置 `recompute_method`。 |
| `full` + `block` + `recompute_num_layers: n` | 在本地层块中选择部分层重算，`n` 表示重算层数；其余层保存激活。 |
| `full` + `uniform` + `recompute_num_layers: n` | 对层分组 checkpoint，`n` 表示每组层数，而非只重算 `n` 层；更大分组可能减少边界输入，但改变重放时的临时占用。 |
| `distribute_saved_activations` | 分片保存 checkpoint 输入、重放时恢复，增加通信以减少这部分保存量。上述 FlagScale 入口要求 TP > 1、full 与有效 method；上述 Ascend 配置禁止它与 SP 同用。 |

同一 Ascend 配置对 `moe_act` 要求 grouped GEMM，对 `mla_up_proj` 要求 MLA，并限制 `shared_experts` 重算与 shared-expert overlap 的组合。`*_per_stage_micro_batch` 可覆盖全局重计算设置。标签通过校验不等于目标模型实际走到该边界，这些规则也不描述所有混合架构的保存策略。

## 输出丢弃与重放时机

普通 checkpoint 省去边界内部保存量，但下游反向所需的边界输出可能仍被保留。[MindSpeed 激活函数重计算][activation]通过输出丢弃与恢复，在下游前向使用后释放输出，在下游反向需要前重建；相较重跑整个 MLP，可避开部分 GEMM。其有效性依赖真实保存关系与恢复时机，不能把所有模块都当作可随意释放输出。

[MindSpeed 重计算独立调度][schedule]把重放从反向中分离，利用流水线空泡提前执行，或调整哪些 microbatch 保存激活。由此可见，重放时机也构成计算与容量的权衡：提前恢复可减少等待，也会改变恢复张量的存活期。这类机制依赖调度器接入，不能把 MindSpeed 的开关直接移植到 FlagScale。

## 通信与状态重放

边界内的 collective 可能在反向重放时再次执行；通信次数、顺序和 buffer 存活期都会影响收益。通信重叠或 dispatcher 重排后，原先可重放的 forward 区域不一定仍保持相同的异步依赖。消除冗余通信、缓存 gather 结果等机制见 [通信优化知识](communication.md)。

重放需恢复原前向所需的状态，避免重复执行有副作用的更新。含 dropout 的区域需要适当保存与恢复 RNG 状态，使同次前向及其重放对应的随机结果一致；这不要求不同训练 step 使用相同随机结果。具体 RNG tracker 和设备支持由实际 checkpoint 实现决定。[PyTorch 状态语义][checkpoint]

PyTorch 的 reentrant 与 non-reentrant checkpoint 是不同实现策略，正常的 reentrant backward 不等于异常无限递归；也不能把 PyTorch 的配置项直接套到框架自定义 checkpoint。[实现区别][checkpoint]

重计算与图执行的组合还取决于 checkpoint 边界、状态恢复和实际被捕获的工作，见 [图执行机制](graph-execution.md)。这类交互不存在脱离版本与实现的统一互斥表。

[checkpoint]: https://docs.pytorch.org/docs/2.7/checkpoint.html
[megatron]: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/apidocs/core/core.transformer.transformer_config.html
[activation]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/activation-function-recompute.md
[schedule]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/recompute_independent_pipelining.md
