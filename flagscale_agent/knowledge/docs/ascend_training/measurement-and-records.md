# 训练测量与正确性证据

训练性能比较涉及吞吐口径、显存统计、分布式计时和数值正确性。相同的性能数字，只有在工作负载、测量边界和状态语义一致时才具有可比性。

## 目标与受控差异

吞吐、容量和可靠性是不同的优化目标。吞吐改善意味着在可比条件下，收益超过预设最小值和测量噪声；容量改善意味着原本无法容纳的工作负载能够运行，并满足允许的吞吐代价；可靠性改善体现为长尾、故障或恢复行为的改善。容量优化不一定提高 token/s，吞吐下降也不自动构成容量改善。

A/B 的可比性取决于受控变量：除被研究的配置、环境或源码差异及必要派生值外，其余条件应一致。有效配置、实际 argv 和环境变量共同决定运行行为，源码版本和补丁也属于比较条件。TP/MBS 改变时，DP 或累积数可能需要联动以保持 GBS；这与更改训练数学定义或数据起点是不同的变化。

比较指纹使用运行时实际形状，包括 tokenizer/special token 与填充后的词表，不能只比较输入 YAML 的 `vocab_size`。TP 可能改变词表填充与输出分片；即使 token 输入相同，也须先证明 logits、初始状态和更新可比。词表派生规则与布局约束见[并行约束](parallelism.md)。

## 计时与吞吐口径

一个计时样本对应完整 optimizer step，包含该 step 的全部梯度累积、必要 collective 和 optimizer update。先检查日志计时器是否等待设备工作完成、是否只测 host enqueue、是否按多个 step 平均。不能把 microbatch 的 forward/backward 时间当作完整 step。

使用框架已有的低扰动计时；需要独立计时时，核对目标 NPU event/synchronize API，只在测量边界等待完成，不往每个 op 内加入同步以破坏 overlap。相同测量方法用于所有候选。

设有效窗口包含 N 个完整 step，每步耗时 `t_i` 秒、实际训练 token 数 `k_i`：

```text
window_tokens_per_second = sum(k_i) / sum(t_i)
tokens_per_second_per_npu = window_tokens_per_second / W
step_p50 = median(t_i)
step_p90 = p90(t_i)，固定分位数算法并记录
throughput_ratio = candidate_tokens_per_second / baseline_tokens_per_second
throughput_gain = throughput_ratio - 1
step_time_reduction = 1 - candidate_step_p50 / baseline_step_p50
```

区分吞吐倍数、吞吐增幅与耗时下降比例。不要先取每步 token/s 再做算术平均作为窗口吞吐，也不要混用 seconds/ms。

- 固定长度且每个 token 都计入训练工作时，可用 `k_i = GBS × seq_length`。GBS 已是全局批量，不能再乘 DP/TP/PP/EP/CP。
- packing、padding、变长文本、多模态或 loss mask 存在时，同时报告实际有效训练 token 和名义容量口径；从数据/训练日志获得计数并避免跨模型并行 rank 重复累计。
- 存在 padding/drop-token 变化的候选，先判断是否改变训练语义；不能只靠更高的名义 token/s 入选。
- 只知 p50 时可给出 `GBS×seq_length / p50` 的吞吐估计，但明确标注估计，不替代窗口真实吞吐。
- W 变化的实验另外列明资源规模和每 NPU 吞吐；没有同工作负载小规模参照时，不编造扩展效率。MFU 只有算量模型和本精度硬件峰值均可靠时才报告，MoE 需区分激活参数和总参数。

## 稳定窗口与 rank

固定采样规则，剔除编译、数据/优化器初始化、显式 warmup、checkpoint、evaluation 和 profiler step。记录每条剔除原因，不因某条计时慢而事后删掉它。重复编译、数据卡顿或通信长尾属于实际问题，应另报告或在生产窗口中保留。

启用了功能调用计数、执行 hook 或状态快照复制的诊断作业不参与稳定性能排名；在退出时移除 hook 不会消除运行期间的扰动。保留这类作业作为执行或质量证据，性能复测使用相同口径且关闭这些观测的独立作业。

普通训练计时排除周期性 I/O 以比较计算性能；最终生产验证另报包含真实 I/O 的 wall-clock/有效 token 吞吐。所有候选的窗口与筛选规则必须一致。

框架只由一个指定 rank 打印全局 step 时间时，用该权威日志并记录口径；不要拼接多个 rank 的重复行。各 rank 独立计时且覆盖同一 step 时，按 step id 取最大完整 step 耗时表示同步关键路径，再算统计量。不能对不同 rank 时钟的绝对时间戳直接求差。

核对 step id 单调、没有混入重启前日志、有效样本数达标、每个 worker 的完成信息可追溯。只收到 rank 0 的 loss 不是分布式任务完成证明。

普通 PP 的非末物理 stage 可合法返回空 loss；由实际 PP group、stage 位置和训练入口的 loss provider 语义确定哪些 rank 应有有限 loss，VPP 或特殊 schedule 不能只按全局 rank 猜测。仍须核对全部 rank 的成功更新、有限梯度证据、绝对 iteration/consumed progress 和最终退出；不能因非末 stage 无 loss 判整个作业失败，也不能因末 stage 有 loss 补判其他 rank 完成。

显存需明确 MiB/GiB 或 MB/GB，记录跨 rank 的 peak allocated、peak reserved 及可用容量来源。`npu-smi` 是设备级占用观察，不能直接当作某个训练进程的 PyTorch 分配峰值。精确 API 名称按安装版本确认；缺失不能填 0。显存峰值至少覆盖首次 optimizer step 与代表性训练阶段，不只测稳态计时窗口。

## 独立 A/B 与正确性

候选短跑阶段检查正常结束、有限 loss/grad、没有意外 skipped updates 或 backend 回退；这些是可行性检查，不是数值等价证明。

固定 checkpoint、数据起点与种子可以减少初始化差异，交错运行基线与候选可以降低随时间漂移的偏差。独立作业的窗口吞吐是重复比较的统计单位；同一作业内相邻 step 通常相关，不能视为等量的独立重复。
OOM 基线没有有效吞吐，容量方案的吞吐代价只能相对于明确的可运行参照计算。可行候选的独立复测用于确认容量与性能的稳定性，重复确定性 OOM 本身不会增加吞吐证据。

成对吞吐比的中位数与两组吞吐中位数之比是不同的统计量，报告须沿用实际聚合方法，并结合各次运行范围与重复基线波动解释。收益接近噪声时，证据不足以区分真实改善和环境波动。少量重复不自动具有统计显著性；显著性取决于检验方法、样本量和独立性。

对应的正确性证据：

| 变化 | 必要检查 |
| --- | --- |
| 仅配置、保持数学定义 | 相同逻辑 batch/初始状态的 loss、grad norm 和更新结果，按项目容差比较；布局改变需正确重分片对齐 |
| 通信 overlap/调度 | 多个真实更新的完整性、梯度归约/累积结果，无新增 hang/丢更新 |
| TE/Ascend backend 或融合算子 | 实际公共接口的前向和全部可微输入梯度；dtype/shape/layout/mask/空 tensor 等受影响边界 |
| 重计算 | RNG/dropout 语义、梯度一致性、优化器更新和额外数值误差 |
| 跨布局 `model_only` 质量 | 初始与更新后模型参数按真实分片对齐，同逻辑 batch 的 loss、grad norm 和更新差量符合预设门槛；不代表 optimizer/RNG 恢复 |
| 完整续训或重分片恢复 | 模型/optimizer/scheduler 状态、RNG、consumed samples 与绝对更新进度的加载、保存和恢复 |

容差与检查方法需要预先确定；合理的浮点非确定性不等于语义变化。若权重绝对容差大于一次更新量，仅比较更新后权重可能漏掉错误，因此初态和更新差量也是质量证据。

训练性能、模型更新等价和完整续训等价分别回答不同问题。涉及跨布局初态、optimizer/scheduler、RNG 或数据进度时，细节见 [状态恢复与数值可比性](state-and-resume.md)。采集扰动与 rank 覆盖见 [Profiling 知识](../ascend_profiling/collection-and-analysis.md)。
