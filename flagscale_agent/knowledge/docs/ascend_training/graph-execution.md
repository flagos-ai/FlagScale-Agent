<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 图编译与捕获重放

## 优化机制与实现路径

| 机制 | 可能的收益 | 主要代价 |
| --- | --- | --- |
| 图编译与优化 | 改变算子组织、融合及执行方式，减少中间访存或调度开销 | 编译成本、算子覆盖范围、图断裂及重编译。 |
| 图捕获与重放 | 复用设备执行序列，减少重复 Host 下发开销 | 捕获成本、图资源与静态缓冲，以及形状、地址和执行依赖的约束。 |

两种机制可以组合。TorchAir 的 Ascend IR 路径将 FX 图转换为 GE 图；`npugraph_ex` 提供 FX 优化与 ACLGraph 捕获路径；TorchNPU 26.1 文档还列有名为 `npugraphs` 的捕获后端。这些名称对应不同入口，不能互换配置或支持范围。[TorchAir][torchair]、[npugraph_ex][ex-guide]、[npugraphs][npugraphs]

训练能力必须区分路径：TorchNPU 的 `make_graphed_callables` 支持 callable 的前向和反向子图捕获，并接入 autograd；这不等于捕获 optimizer 或完整分布式更新。[NPUGraph][npugraph] 所引 `npugraph_ex` 快速指南则明确主要面向推理，暂不支持反向流程和随机数算子 capture；其优化机制可供参考，不能直接作为当前训练栈的使能配方。[适用范围][ex-guide]

## 接口参数与训练边界

| 接口/参数 | 语义与开销 |
| --- | --- |
| `make_graphed_callables(module, sample_args, ...)` | 样例参数是张量 tuple，shape、dtype 和 `requires_grad` 对应实际调用；返回 callable 将前向与反向子图接入 autograd，输入复制仍有成本。它不自动捕获外围 optimizer、数据读取或整个更新。[TorchNPU 接口][graph-source] |
| `num_warmup_iters` 与 `pool` | 前者控制该接口的准备迭代，增加准备成本，不是测量窗口；后者提供图池共享提示，共享是否正确仍受调用顺序和张量生命周期约束。[参数定义][graph-source] |
| `torch.compile` 的 `backend`、`dynamic` 与 `fullgraph` | backend 决定实际编译实现；后两者控制追踪形状策略和 FX 图断裂要求，不是训练支持开关。后端专有 options 不能原样移给另一后端。[编译入口][ex-guide] |

预热和捕获可能实际执行前反向，因而改变 RNG、梯度或有状态模块的缓冲；这些准备工作不等于一次正常 optimizer 更新，也不应改变对照起点。所引 [NPUGraph 说明][npugraph-guide]限定了入图算子与执行条件；模型前向可捕获不证明训练组合已支持。

## 可形成优化假设的机制

| 线索 | 机制 | 代价与适用边界 |
| --- | --- | --- |
| 高频稳定模块包含许多小算子，Host 下发形成空泡 | 局部前反向捕获，以子图接入 eager 训练循环 | 外围动态逻辑可以留在图外，但分段过细会留下图间调度和输入更新成本。[前反向子图接口][npugraph] |
| 图中出现重复 copy、临时张量或可融合算子链 | FX pass 消除冗余搬运、恢复合法的 inplace 或匹配融合算子 | vLLM Ascend 展示了该机制；迁移到训练需保持别名关系及反向保存值，推理的 inplace 合法性不能直接沿用。[FX 优化实现][fx-passes] |
| shape 种类多，捕获数量或图内存持续增长 | 缩小捕获范围或 shape 覆盖集，在支持的实现中复用图池 | 图池复用受输出存活期和执行顺序约束；分段越多、shape 覆盖越广，也可能增加 stream 等运行时资源需求。[图池机制][graph-memory]、[ACLGraph 资源案例][acl-design] |
| shape 稳定，但算子仍有可观的运行时形状处理成本 | 静态 Kernel 编译，将部分 shape/标量处理提前，并生成对应二进制 | 增加编译时间和产物，缓存依赖硬件、CANN、算子属性及编译选项；属于支持该能力的后端机制。[静态 Kernel][static-kernel] |
| 多个 Kernel 之间的调度间隙明显 | SuperKernel 将多个已编译 Kernel 的调用组合，减少调度成本 | 二进制调度融合不等于消除所有中间访存；所引 GE 实现要求静态图，并有可融合算子和同步约束。[SuperKernel][super-kernel] |

这些机制针对不同成本，不构成固定组合。捕获主要减少下发，FX 融合还可能减少设备访存；计算或通信已经占据关键路径时，仅增加捕获范围未必有收益。社区推理实例提供机制依据，不证明当前 FlagScale/Megatron-LM-FL/TE-FL 的训练接入已完成。

## 形状、地址与资源

动态 FX 追踪与运行时图复用是两层问题：所引 `npugraph_ex` 实现中，`dynamic=True` 可让不同 shape 共用 FX 图，但仍可能为不同 shape 捕获不同 ACLGraph。`fullgraph=True` 要求被编译 callable 不发生 FX 图断裂，不代表一次完整训练更新被捕获成一张运行时图。[图池文档][graph-memory]、[接口含义][ex-guide]

NPUGraph 重放使用捕获时的内存地址；更新数据依赖写入静态缓冲或后端支持的地址/参数处理机制，单纯重新绑定 Python 变量不会更新原捕获地址。输入复制、输出克隆及参数更新本身也有成本。[地址语义][npugraph]

图输出可能在后续 replay 时被覆盖。训练中跨 microbatch 保留激活、延后 backward 或共享图池时，输出与保存张量的生命周期会影响复用是否合法；减少 clone 不能以覆盖仍需使用的数据为代价。[输出与图池生命周期][graph-memory]

动态序列长度、MoE 专家 token 数和数据相关控制流都可能改变执行图。局部静态区域可以保留外围动态逻辑；padding 可以规整 shape，但增加计算和内存。只有保持有效 token、mask、routing 和损失语义时，才仍是同一训练问题。图数量还受设备运行时资源影响，不能只按剩余显存估算容量。[分段捕获案例（推理）][acl-design]

## 训练状态与组合依赖

前向、反向子图和完整更新是不同能力范围。样例输入的 `requires_grad`、参数状态和实际调用顺序影响训练子图的正确性；具体限制以对应版本实现为准。[TorchNPU 实现][graph-source]

随机数状态需要正确推进：不同训练 step 的 dropout 不应因 replay 而固定；同次前向与 checkpoint 重放需维持相应 RNG 关系，见 [重计算知识](recompute.md)。某个后端不支持随机数 capture，不构成关闭训练 dropout 的理由。

重计算、通信重叠和图执行都可能改变保存张量、stream 依赖或执行边界。单模块可捕获不证明 collective、checkpoint 和梯度累积的组合受支持。

## 成本解释

首次编译/捕获、稳定执行、持续重编译/重复捕获是不同成本。一次性准备可由后续迭代摊销，反复发生的成本影响实际吞吐。编译缓存主要缩短准备过程，不能据此推断稳态 step 加速。

FX 图断裂、guard 失配后的重编译、ACLGraph 重捕获及捕获失败分别发生在不同层。图数量、输入复制、图池占用和实际 replay 共同解释收益；完整更新的性能与质量口径见 [训练测量与正确性](measurement-and-records.md)。

[torchair]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/graph/graphguide/atlasgraphug_24_0002.html
[ex-guide]: https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md
[npugraphs]: https://www.hiascend.com/document/detail/en/Pytorch/2610/userguide/torchcompile/docs/en/torch_compile/pytorch_compile_npugraph_desc.md
[npugraph]: https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/pytorch_npugraph_desc.md
[fx-passes]: https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/npugraph_ex.html
[graph-memory]: https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/memory_reuse.md
[acl-design]: https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/developer_guide/Design_Documents/ACL_Graph.html
[static-kernel]: https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md
[super-kernel]: https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md
[graph-source]: https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/graphs.py
[npugraph-guide]: https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/npugraph.md
