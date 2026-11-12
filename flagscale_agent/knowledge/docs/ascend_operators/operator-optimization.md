<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 算子调用成本与接口语义

算子优化的对象是训练实际使用的公共调用。设备 kernel、公共调用和完整训练更新有不同的成本边界；某个私有 kernel 更快，不代表模型已经调用它或训练已经加速。

## 热点耗时来自哪里

同一算子名可能覆盖不同 shape、dtype、stride、前后向与 rank/stage。MoE 的矩阵形状还受本 rank 实际 token/专家分布影响，不能只从模型配置推导。调用次数和累计 device 时间可用于筛选，但重叠区间之和不等于 step 耗时；关键路径解释见 [profile 知识](../ascend_profiling/collection-and-analysis.md)。

| 成因 | 机制与判断边界 |
| --- | --- |
| 高频小调用 | 多次下发、调度及中间结果写回可能超过有效计算成本；次数多本身不证明 Host 下发受限。 |
| 数据转换与重排 | 非连续输入、dtype/layout 转换、索引构造和输出恢复可能引入额外搬运或分配；`reshape` 是否复制取决于原 stride。 |
| Host 往返与同步 | 从设备取 Python 标量/列表、复制到 CPU 或显式同步可能让 Host 等待；仅依赖元数据的 shape 查询与读取设备数据不是同一成本。 |
| 设备执行低效 | 搬运、计算、Scalar 指令或核内依赖可能限制 kernel；设备流水指标的分母及输入桶决定其含义，见 [kernel 知识](kernel-experiments.md)。 |
| 负载与依赖等待 | 专家 token 偏斜、跨 rank 到达不齐或前置依赖可使局部热点变慢；单纯替换算子未必解决根因。 |

出现 AICPU 或某个 `argsort` 名称不足以证明回退，实际公共调用、输入与后端绑定共同决定实现归属。

## 可形成优化假设的机制

| 机制 | 值得考虑的线索 | 收益与主要代价 |
| --- | --- | --- |
| 复用已有融合实现 | 多个逐元素/归约算子反复读写同一张量 | 融合可能将中间量留在片上并减少下发；更大的活跃集、尾块 padding 和输入整理可能抵消收益。Triton-Ascend Softmax 示例体现了减少重复读写的机制，不代表可直接替换训练前后向。[融合示例][softmax] |
| MoE 重排融合 | permutation/unpermutation 中排序、索引和搬运占比较高 | MindSpeed 将排列、反排列分别融合，以减少重排开销；支持范围依赖 dispatcher、容量/padding 与版本。其配置限制不能直接推广到 TE-FL 的另一实现。[MoE 重排][permute] |
| 减少转换或复用中间量 | 同一输入在相邻调用中反复整理 | 可减少 copy 和临时分配，但必须保留 stride、别名、反向保存值及更新时序；缓存有效期越长，常驻内存和失效管理代价越大。 |
| 调整 Host/设备边界 | 设备计算结果往返 CPU 后又参与设备计算 | 设备侧表达或减少观测频率可降低往返；业务确需的标量决策和数据依赖不会因删除同步而消失。 |
| 改变算法或按输入选择实现 | 相同功能的排序、归约或 gather/scatter 在不同规模下代价不同 | 算法与实现选择可能减少工作量；分派应基于真实输入契约，未覆盖输入沿用原有行为。硬件标签不保证某实现总是更快。 |

融合与图捕获针对不同成本：前者可改变设备计算和访存，后者主要复用下发序列；机制见 [图执行知识](../ascend_training/graph-execution.md)。

## 公共接口、后端与仓库边界

实际调用可能经过 FlagScale、Megatron 模型/override、TE-FL 或 FLA 公共 API、vendor adapter，再进入 `torch_npu` 或其他 NPU 实现，也可能跳过其中若干层。公共 Python API、扩展模块和私有 Triton 函数可能使用不同入口；文件存在、import 成功、注册成功与训练实际绑定是不同证据。

| 接入层 | 典型职责 |
| --- | --- |
| FlagScale / Megatron-LM-FL | 训练与模型集成、模型级 override、并行上下文和调用参数 |
| TransformerEngine-FL backend / adapter | 公共接口映射、后端注册及选择、训练前后向衔接 |
| TorchNPU 或具体 kernel 库 | 算子 API、设备实现及各自支持的输入范围 |

本仓 TE-FL 的 NPU RMSNorm 适配是一个例子：`rmsnorm_fwd/rmsnorm_bwd` 映射到 `torch_npu.npu_rms_norm/npu_rms_norm_backward`，还包含输入形状整理、统计量和返回值映射。对应实现位于 `transformer_engine/plugin/core/backends/vendor/npu/npu.py`，注册位于同目录 `register_ops.py`；这说明适配不仅是更换函数名，不证明所有可选参数均已覆盖。

局部 adapter/override 可以限制修改范围；统一分派与 fallback 的职责由当前框架决定。逐次捕获任意异常再静默尝试另一实现，会隐藏实现错误和额外成本，不能作为候选已正确接入的证据。

## 训练接口中容易遗漏的语义

| 对象 | 必须保持的含义 |
| --- | --- |
| norm / 归约 | epsilon、统计轴、累加精度、权重变换及反向需要的统计量；输出相近不证明权重梯度正确。 |
| MoE 排列与反排列 | token 到专家的展开、row/index 映射、概率加权和梯度；只在配套下游及反向同步保持时，内部排列变化才可能等价。 |
| FLA / GDN 等带状态接口 | 可选状态、输出顺序、可微输入及状态更新的契约；底层前向实现可调用不等于公共训练接口完成适配。 |
| attention / CP | Q/K/V 的实际全局位置、分片与 gather 顺序、mask 的行列含义；连续位置规则不能直接套到 zigzag 分片。 |
| inplace / buffer 复用 | 输出别名及后续写入是否覆盖仍供 backward、其他 microbatch 或异步消费者使用的数据。 |

CP 的简单位置参照中，causal mask 由全局 `k_pos <= q_pos` 定义，并叠加原窗口/packing 等约束；`rank × local_seq_len` 只适用于已确认的连续切分。局部 mask 反例不能单独证明真实后端绑定错误或 K/V 梯度通信缺失。

浮点输出及所有可微输入梯度按项目约定容差比较；离散索引与映射按语义判定。空输入、非连续布局、尾块和极端数值只有在公共接口支持或本次变化涉及时才构成对应边界，不存在所有算子通用的一组尺寸或容差。功能修复改变了原错误行为时，原错误运行不能充当正确训练的性能基线。

## 三层性能口径

| 层次 | 包含的成本 | 结论范围 |
| --- | --- | --- |
| Device kernel | 设备上的算子执行 | 当前输入与版本下的 kernel 性能 |
| 公共调用 | 输入整理、分配、转换、辅助张量、下发、输出恢复及必要完成等待 | 模型实际调用边界的性能 |
| 完整训练更新 | 前后向、梯度累积、通信和 optimizer 更新 | 相同工作负载与质量条件下的训练收益 |

异步提交耗时不等于设备完成时间；微基准的完成等待属于测量边界，不应变成生产路径中每个内部算子的同步。编译、预热、profile 与稳态成本不同，测量契约见 [训练测量与正确性](../ascend_training/measurement-and-records.md)。kernel 加速被转换、分配或依赖等待抵消时，公共调用或训练可以没有收益。

[softmax]: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/examples/02_fused_softmax_example.md
[permute]: https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/moe-token-permute-and-unpermute.md
