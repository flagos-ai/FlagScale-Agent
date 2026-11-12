<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 并行策略的机制与权衡

并行布局同时改变单 rank 的状态、计算粒度、通信组和执行时序。更大的并行度可能解决容量问题，也可能增加等待或降低算子效率；收益取决于完整更新的关键路径。

## 各策略分摊什么

| 策略 | 分摊对象与可能收益 | 主要代价与边界 |
| --- | --- | --- |
| TP：张量并行 | 切分层内权重和计算，减少被切分部分的单 rank 状态与计算量 | 需要层内 collective；局部矩阵变小可能降低计算效率，未分片部分不会按 TP 等比例缩小。 |
| SP：序列并行 | 在 TP 组内沿序列维切分部分原本重复的激活与计算，例如部分 norm/dropout 区域 | 改变 TP 的张量布局及 gather/scatter 路径；它不切分所有层的完整上下文，不等同于 CP，也不增加独立设备维度。 |
| PP：流水线并行 | 将不同层分配给不同 stage，分摊层参数及相应梯度、优化器状态 | 引入 stage 间 P2P 和流水线空泡；保存激活数量随调度变化，均匀层数不保证均匀显存或耗时。 |
| VPP：虚拟流水线 | 每个物理 rank 持有多个模型 chunk，交错调度以减少空泡 | 不增加物理卡数；更多 chunk 会改变通信次数、调度开销和激活存活期，收益受 microbatch 数与 stage 负载限制。 |
| CP：上下文并行 | 沿上下文维分摊各层序列激活与计算，缓解长序列容量压力 | attention 仍须覆盖原有上下文，需要跨分片交换数据；并行度越大，本地块越小，通信越可能暴露。 |
| EP / ETP：专家并行 | EP 将不同专家放在不同 rank；ETP 切分单个专家内部的张量与计算 | EP 影响 token dispatch/combine，ETP 增加专家内部通信并改变矩阵形状；专家数量均分不意味着 token 或计算负载均衡。 |

TP/SP/PP/VPP/CP 的通用机制见 [Megatron 并行说明][megatron]；这些原理不等同于某一 Ascend 后端的组合支持清单。

## 布局、批量与负载的联动

固定总进程数时，常见 dense 布局中 TP/PP/CP 的变化会改变派生 DP；固定 GBS/MBS 时，每次更新的 microbatch 数随之改变。因此布局比较的收益可能同时来自通信、计算粒度和流水线填充程度。增大 MBS 可能提高算子效率，却减少 microbatch 数并加重空泡。批量定义与调度约束见 [批量与累积](batch-and-accumulation.md)。

VPP 复用物理 rank；MoE 的 dense 与 expert mesh 也在同一组资源上组织不同通信组，不能把 TP、CP、EP、ETP、VPP 全部相乘。ETP 若默认继承 TP，改变 dense TP 还可能同时改变专家内部布局。

stage 负载取决于层类型、embedding/loss、专家 token 分布及保存激活的生命周期。MindSpeed 的 PP 自动并行资料指出，1F1B 下前部 stage 可因更多尚未完成 backward 的 microbatch 而持有更多激活；不均匀层分配和重计算可以联动平衡容量。层数均衡、计算时间均衡与峰值显存均衡是不同目标。[PP 负载与内存机制][pipeline]

MoE 中热点专家可能使少数 rank 决定更新耗时；增大 EP 本身不会使路由负载均匀。MindSpeed 的动态负载均衡通过调整参数副本和 token 计算位置缓解偏斜，也引入参数同步与调度依赖；该机制不同于改变 routing、top-k 或丢 token 规则。[专家负载机制][balanced-moe]

## 分组与整数约束

常见同构 decoder-only 布局中，`DP = W / (TP × PP × CP)`，`G = GBS / (MBS × DP)`；并行度、MBS、GBS 为正整数，两次除法须整除。SP 不增加设备维度，EP 不再乘入 dense DP 的分母。异构并行、动态 CP、encoder-decoder 和自定义 mapping 不由此简式完整描述。

在 Megatron-LM-FL 使用独立 expert rank generator、其 CP=1 且 ETP 默认继承 TP 的实现中，`EDP = W / (ETP × EP × PP)`。dense/expert mesh 复用进程；还受 PP groups 对齐、rank ordering 和专家数分片约束，不能通用断言 EP 必须整除 dense DP。

上述分组公式的算术规模下界为 `PP × lcm(TP×CP, EP×ETP)`；只有两边均整除时才能简化为 max。这个下界不证明任意两种 mesh 都被模型、拓扑和后端支持。

| 对象 | 额外约束 |
| --- | --- |
| TP / SP | hidden、FFN、head/group 与序列切分受实现限制；GQA/MQA 可能复制 KV，不能统一要求 KV heads 被 TP 整除。SP 通常依赖 TP>1。 |
| PP / VPP | 简单均匀布局受层数整除约束；自定义 layout、首尾层和 embedding/loss/MTP 按实际解析器计数。microbatch 数另受 schedule 约束。 |
| CP | 通信模式、序列切分、mask、packing/THD、GQA/MLA 和 dropout backward 须兼容；部分算法要求长度整除 2×CP，不能推广至所有后端。 |
| EP / ETP | 专家切分、dense/expert groups、dispatcher、grouped GEMM 及 TP/SP 组合共同决定合法性。 |

例如 W=8、TP=2、PP=2、CP=1 得 DP=2；GBS=64、MBS=4 得 G=8，MBS=3 不合法。整数合法只是运行条件之一。

## 词表形状与跨布局语义

配置 `vocab_size`、tokenizer 含 special token 的大小和 `padded_vocab_size` 可能不同。对“NullTokenizer 额外增加 EOD、再按 `make_vocab_size_divisible_by × TP` 向上填充”的实现，TP 变化可能改变 embedding/output 和 logits 的实际形状；新增 logits 不能自动视为不影响 loss 的空位。这不是所有 tokenizer 的通用公式。

保持实际词表形状的 divisor 调整还受 parser/checkpoint 条件限制；改变原始词表则改变工作负载。跨布局比较需要按真实分片对齐，状态恢复的证据范围见 [恢复与可比性](state-and-resume.md)。

CP 的分片顺序、全局位置和 attention mask 必须一致。zigzag 切分不能直接使用连续分片的位置公式；fallback 可运行也不证明数值等价。公共接口的位置语义见 [算子知识](../ascend_operators/operator-optimization.md)。

## 昇腾社区的 CP 算法与拓扑取舍

MindSpeed 提供了以下长序列并行实现，适合解释 CP 的不同成本。其开关、支持矩阵和经验阈值不能直接套用到 FlagScale/Megatron-LM-FL/TE-FL。

| 算法 | 数据组织与通信 | 性能与形状边界 |
| --- | --- | --- |
| Ulysses | 通过 All-to-All 在序列分片与 attention head 分片间转换，使每个 rank 对部分 head 处理完整上下文 | 受 head 切分及 All-to-All 路径影响；所引 MindSpeed 实现要求 attention heads 可被 TP×CP 整除，GQA/MQA 还涉及具体 KV 分片支持，不能推广为所有 CP 算法的约束。[Ulysses][ulysses] |
| Ring Attention | 本地保留 Q 块，环传 KV 块并累计分块 attention 结果 | 不依赖同样的 CP head 切分，但仍有后端形状约束。本地块过小时，计算不足以隐藏通信；增加并发还可能争用片上内存带宽。[Ring][ring] |
| 混合 CP | 将 CP 分解为 Ulysses 与 Ring 两个维度 | 可在 head 切分限制与环传粒度之间折中，收益同时依赖组内/组间链路和分组映射。[混合 CP][hybrid] |

这些机制说明 CP 大小与通信算法、拓扑需要共同考虑；不存在对所有模型、芯片和链路都成立的固定序列长度门槛。通信路径与关键路径等待的区别见 [通信知识](communication.md)。

## 当前 FlagScale / Ascend 实现约束

以下规则对应 FlagScale `a3088a5`、Megatron-LM-FL `1366b6b`。配置校验通过不证明所有模型、shape 和 NPU 算子组合可运行；配置与实际派发的区别见 [三仓配置消费](stack-capabilities.md)。

- **GQA 与 TP**：Ascend `TransformerConfig` 允许 `num_query_groups` 是 TP 的整数倍或整数约数；实际 attention 仍须支持对应分片，不能仅凭一种整除方向排除配置。
- **SP 与重计算**：在该实现的重计算配置校验分支中，`distribute_saved_activations` 与 `sequence_parallel` 同时启用会报错。前者分摊重计算保存的激活，不是启用 SP 的前置条件。
- **PP/VPP 与 P2P**：interleaved 调度开启 `overlap_p2p_comm` 时要求 PP>1，关闭时要求 PP>2；非 interleaved 路径会关闭该 overlap。这里是版本特定的调度限制，不是 PP 的通用数学约束。
- **层分配**：普通均匀布局受 stage/chunk 划分约束；显式 layout 或 hybrid pattern 定义自己的层分布与虚拟 stage，不能同时叠加多个相互排斥的定义入口。

上述校验来源分别是 Megatron-LM-FL 的 `megatron/plugin/Ascend/transformer/transformer_config.py`，以及 FlagScale 的 `flagscale/train/megatron/training/arguments.py`。跨布局比较涉及真实分片与状态对齐，不能直接比较同号 rank 的局部张量；证据口径见 [训练测量与正确性](measurement-and-records.md)。

[megatron]: https://docs.nvidia.com/nemo/megatron-bridge/nightly/parallelisms.html
[pipeline]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/automated-pipeline.md
[balanced-moe]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/balanced_moe.md
[ulysses]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md
[ring]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/ring-attention-context-parallel.md
[hybrid]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/hybrid-context-parallel.md
