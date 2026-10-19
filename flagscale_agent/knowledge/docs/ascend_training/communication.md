<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 通信代价与重叠机制

## 通信耗时来自哪里

HCCL 算子的持续时间可能包含传输、等待与同步，不能全部解释为链路传输。多个通信任务还可能并发，累计时长不能直接当作训练停顿；端到端影响取决于关键路径上未被其他工作隐藏的依赖。区间统计见 [profile 数据解释](../ascend_profiling/collection-and-analysis.md)。

| 成因 | 机制与优化权衡 |
| --- | --- |
| 消息过碎 | 每次通信都有下发和调度成本；合并消息减少调用次数，但会推迟数据就绪后的发起时刻。 |
| 传输量大或链路受限 | 代价受消息大小、collective 算法、rank 映射及实际链路影响；相近消息与并发条件下的同类链路才适合比较带宽，不能仅凭“节点内/节点间”判断实际路径。 |
| rank 到达不齐 | 较早进入 collective 的 rank 可能等得更久；上游计算、token 负载或 Host 下发不均，也会表现为通信耗时长。 |
| Host 下发跟不上 | 设备可能等待通信任务到达；此时提高链路带宽未必有效，CPU 调度和下发粒度也影响空泡。 |
| 通信未被计算隐藏 | overlap 需要数据已就绪、结果尚未被消费，且存在可并发的独立计算；关键路径上的串行依赖限制可隐藏范围。 |

[msprof-analyze][cluster] 提供通信域、传输/等待时间及 rank 间通信矩阵；[MindStudio 的快慢卡案例][slow-rank]显示，较晚到达的慢卡反而可能具有较短的通信算子时长。因此，单卡通信时长排名不能直接当作慢卡排名。[HCCL 下发分析][launch]也区分 Host 下发瓶颈与设备执行。

## 分桶、预取与重叠

梯度 bucket 聚合已产生的梯度，按实现条件触发归约。较小的 bucket 可能更早就绪，给后续反向计算留下掩盖窗口，但增加调用次数；较大的 bucket 则减少调用次数，可能把更多通信推迟到反向尾部。bucket 参数可能按元素数或字节计量，需按具体字段解释。[MindSpeed 异步 DDP][async-ddp]给出了这一流水机制。

参数 gather 预取把参数收集提前到消费之前；提前不足会阻塞计算，提前过多可能延长参数/缓冲存活时间。梯度就绪顺序、参数消费顺序和分桶顺序决定重叠窗口，开关打开不代表窗口一定存在。

DP 归约/参数 gather、PP P2P、MoE dispatch/combine 的生产和消费位置不同，不能共用一个固定的最优粒度。通信与计算也会竞争设备资源；时间线上重叠更多，不保证完整更新更快。重放交互见 [重计算知识](recompute.md)，捕获相关约束见 [图执行知识](graph-execution.md)。

## 可形成优化假设的机制

以下机制提供不同的优化假设，不限定探索顺序。

| 机制 | 值得考虑的线索 | 原理与主要代价 |
| --- | --- | --- |
| 消除冗余通信 | 重计算中再次执行了前向 collective | 若重放边界末端的通信结果不被激活恢复或后续反向消费，调整边界可能省去该通信；必须保持分布式张量与梯度语义，不能任意删除 AllReduce。[TP 重计算通信优化][recomp-comm] |
| 缓存通信结果 | TP 线性层在前反向重复 gather 相同输入，显存有余量 | 缓存前向 AllGather 后的输入张量，供同一 microbatch 的反向使用，以更长的激活存活时间换取更少通信。更一般的结果复用要求数据仍有效，收益受完整更新的峰值内存约束。[CoC 的结果复用][coc] |
| 分块流水或通算融合 | 相邻 MatMul 与 collective 因数据依赖串行 | 按数据块建立流水，使一块计算与另一块通信重叠；切分过细可能降低 GEMM 效率、增加下发与临时缓冲成本，融合实现也受 shape、dtype 和后端支持限制。[CoC][coc] |
| 跨 microbatch 或阶段调度 | 层内缺乏足够独立计算来隐藏 MoE dispatch/combine | 利用其他 microbatch 的计算，或分离输入梯度与权重梯度计算，形成重叠窗口；会改变调度和张量生命周期，延后的权重梯度仍须在归约与更新前完成。[MoE 跨 microbatch 通信掩盖][fb-overlap] |
| MoE 分层 token 分发 | 同一 token 发往远端多个专家，跨慢链路重复传输 | 利用分层通信和专家布局减少重复的跨层级传输，再在目标层级分发；收益取决于 routing 分布及额外本地通信、重排和缓冲成本，不保证固定倍数。[分层 AlltoAllv][hierarchical] |

所引 MindSpeed 跨 microbatch 实现注明主要在 DeepSeek V3 验证，其他模型仍需适配。上述 MoE 通信优化以保持 routing、top-k 和丢 token 语义为前提。

## HCCL 缓冲区与显存

梯度 bucket 决定框架如何组织通信，HCCL buffer 则是通信后端的数据缓冲，两者不是同一个参数。HCCL 缓冲按通信域管理，多个通信域的占用会叠加；减小缓冲可能节省显存，也可能降低通信效率，不能把增大缓冲视为无代价的加速。[官方内存调优说明][buffer-memory]

`HCCL_BUFFSIZE` 提供全局缓冲配置；支持该能力的 TorchNPU 可通过通信域 `pg_options.hccl_config` 中的 `hccl_buffer_size` 单独设置，显式通信域配置优先于环境变量。接口含义见 [TorchNPU 通信域参数文档][pg-options]。MindSpeed 的[按通信域设置缓冲实现][group-buffer]体现了不同通信组需求不同的思路。

## HCCL 算法选择的边界

HCCL 默认按硬件、数据量和节点规模自适应选算法，手工指定会覆盖相应自动选择。CANN 8.5 的 `HCCL_ALGO` 控制 Server 间和超节点间算法，Server 内 `level0` 仅支持 `NA`，因此该版本的跨机算法配置不构成普通单机通信的搜索维度。算法对照的意义取决于目标通信所在层级、算子和实际支持范围，不能从“通信慢”直接推出某种算法更优。[官方配置说明][hccl-algo]

## 已知配置依赖与适用范围

FlagScale `a3088a5` 的两个参数校验入口均要求 `overlap_param_gather` 同时启用 `overlap_grad_reduce`，但 optimizer 条件不同：

- `flagscale/train/megatron/training/arguments.py`：接受 `use_distributed_optimizer`、`use_megatron_fsdp` 或 `optimizer == 'dist_muon'`。
- `flagscale/train/megatron/training/yaml_arguments.py`：对应校验要求 `use_distributed_optimizer`。

这些是对应入口的静态条件，不能只凭配置文件后缀判断实际校验路径，也不能据此证明 NPU 后端已执行重叠。PP P2P 的调度条件见 [并行配置约束](parallelism.md)。

社区实现用于参考机制；后端接口和 MindSpeed 开关不等于 FlagScale 配置项。具体支持及组合兼容性取决于芯片、CANN/TorchNPU 与框架版本，原实现的限制不直接推广到其他路径。CUDA/NCCL 的环境变量和 Userbuffers 参数不构成 NPU 路径的配置依据。

[cluster]: https://github.com/Ascend/msprof-analyze/blob/master/docs/en/user_guide/cluster_analyse_instruct.md
[slow-rank]: https://www.hiascend.com/document/detail/zh/mindstudio/830/practicalcases/GeneralPerformanceIssue/toolsample6_034.html
[launch]: https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000017.html
[async-ddp]: https://gitee.com/ascend/MindSpeed/blob/cf7f0092061caef91e17f016b3864639dad5a008/docs/features/async-ddp.md
[buffer-memory]: https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0048.html
[pg-options]: https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/distributed/parameter_setting/setting_HCCL_communicator_parameter.md
[group-buffer]: https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hccl-group-buffer-set.md
[recomp-comm]: https://gitee.com/ascend/MindSpeed/blob/8710f5452cebc79899e1dd3bf18f438a0f9f6ce1/docs/features/recomputation-communication.md
[coc]: https://gitee.com/ascend/MindSpeed/blob/4ac3c68db3e8f4fcb1a50eb1b011c03455ae5623/docs/features/communication-over-computation.md
[fb-overlap]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/megatron_moe/megatron-moe-fb-overlap.md
[hierarchical]: https://gitee.com/ascend/MindSpeed/blob/c68e201a5733065b3aa59a3d44dc5e2ec700d7c9/docs/features/hierarchical-alltoallv.md
[hccl-algo]: https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000075.html
