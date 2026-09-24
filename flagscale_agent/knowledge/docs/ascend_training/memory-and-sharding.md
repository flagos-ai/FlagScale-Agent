<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 显存来源、分片与卸载

## 容量由哪些对象决定

峰值取决于同一时刻存活的对象，不能把不同阶段的最大占用简单相加。首个 optimizer 更新可能才分配状态，反向、参数 gather、图捕获或保存时也可能出现独立峰值。

| 来源 | 可改变的机制与边界 |
| --- | --- |
| 参数、主参数、梯度和 optimizer 状态 | 张量精度、模型切分与状态分片影响常驻量；仅改激活重计算不会消除这部分。 |
| 保存的激活 | MBS、序列长度、流水线中在途 microbatch、保存边界及重计算决定存活量；模型总参数量不是激活大小的充分指标。 |
| 临时张量与 workspace | 算子实现、融合、重排和重放决定临时开销；局部释放未覆盖峰值时，容量收益可能很小。 |
| 通信、预取与图资源 | 并发和提前预取可能延长 buffer 生命周期；HCCL 等框架分配器之外的占用也影响设备容量。 |

`allocated` 描述分配器中的活跃张量占用，`reserved` 包含缓存池保留空间；二者之差不能直接认定为可回收碎片。设备总占用还包含运行时及其他分配，不能与框架统计互换。重计算见 [激活边界](recompute.md)，布局见 [并行机制](parallelism.md)，通信缓冲见 [HCCL 与 overlap](communication.md)。

## OOM 阶段与容量边界

| 峰值位置 | 可能主导的对象 |
| --- | --- |
| 模型加载或首次 optimizer 更新 | 层/专家分片、主参数和 optimizer 状态；激活重计算不减少这部分常驻量 |
| forward 保存激活 | 实际 token 数、层类型、保存边界与在途 microbatch |
| backward / 重放 | 临时 workspace、嵌套重计算、重复通信及峰值迁移 |
| 梯度同步 / 参数 gather | bucket、预取数量、异步 buffer 的存活期 |
| compile / capture / replay | shape 集、编译缓存及图的私有/常驻资源 |
| 个别 rank 或后期样本 | token/expert 偏斜、stage 差异、长样本或 checkpoint 峰值 |

失败 rank、阶段、allocated/reserved 与失败申请量共同限定归因；通信超时本身不是 OOM 证据。OOM 从 forward 转移到首次 optimizer 更新仍未达到完整更新容量要求，平均 rank 占用也不能代表最重 rank。

理论内存模型受 dtype、分片与活跃期假设限制；未覆盖的 HCCL/dispatcher buffer、workspace、图缓存、碎片和 token skew 属于估计缺口。目标 NPU 上某个布局的校准不能直接外推到其他布局，也不能仅凭未校准估计断言候选必然 OOM。

## 分布式优化器分摊什么

Megatron 分布式优化器在实际分片组内分摊 optimizer 状态与用于更新的主参数；梯度 ReduceScatter 后，各 rank 更新自己的片段，再 AllGather 训练所用参数。[Ascend MindSpeed 说明][dist-ascend]与 [Megatron 实现说明][dist-megatron]描述了这一机制。

这不等于把全部参数、梯度和激活常驻量都除以 DP 数：Megatron 路径仍有完整的训练参数和通信缓冲，节省量取决于 dtype、optimizer 状态及实现。分片组大小为 1 时没有跨 rank 的状态分摊；MoE 的普通参数与专家参数可能使用不同分组，world size 不能直接代替分片规模。

状态分片可以减少冗余更新，但引入参数收集并改变梯度通信方式，是否更快取决于通信窗口和分桶。它与 activation 重计算作用于不同来源，可以形成组合；保存格式和恢复能力仍由 optimizer/checkpoint 实现决定，不由开关名称保证。

## 激活卸载与预取

激活卸载把暂时不用的激活放到 Host，反向消费前再搬回，提供“保存、重算、搬运”之间的第三种取舍。[MindSpeed swap-attention][swap]用预取减少反向等待，可在部分场景替代全重计算；这是机制参考，其参数不自动适用于 FlagScale。

由这一机制可推导，收益取决于 D2H/H2D 搬运能否被计算隐藏，以及 Host 容量、NUMA 和带宽条件。过早预取或多份 staging buffer 会增加同时存活量；减少 NPU 保存量也可能增加主机内存与传输负担，不能默认零开销。卸载对象和恢复边界必须匹配实际反向需求。

## 缓存分配器与碎片

[TorchNPU 分配器配置][allocator]控制内存复用，不减少活跃张量本身。`expandable_segments` 让内存段可扩展，适合频繁变化的分配大小；`max_split_size_mb` 限制大块切分，主要针对 OOM 且存在大量非活动切分块的情况，未必改善计算吞吐。

参数支持与组合限制受版本和芯片影响。所引文档要求非默认 `max_split_size_mb` 或 `garbage_collection_threshold` 与 `expandable_segments:False` 配合；不能把多项“省显存参数”机械叠加。新版还存在 `PYTORCH_ALLOC_CONF` 别名及同时设置的限制，应以安装版本为准。

## 当前实现的适用边界

Megatron-LM-FL `1366b6bc1` 的 `megatron/plugin/Ascend/transformer/transformer_config.py` 对 `cpu_offloading` 禁止 PP > 1 及同时启用重计算。这是该路径的校验条件，不是所有 Ascend 卸载实现的共同限制；参数存在也不证明当前模型与 TE-FL 已执行卸载。

[dist-ascend]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/distributed-optimizer.md
[dist-megatron]: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/dist_optimizer.html
[swap]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/swap_attention.md
[allocator]: https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md
