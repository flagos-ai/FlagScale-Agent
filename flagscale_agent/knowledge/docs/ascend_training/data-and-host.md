<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend 数据供给与 Host 下发

## 等待发生在哪一段

读取、解码/预处理、collate、Host 到设备搬运、算子下发与设备执行构成相互依赖的链路。只有暴露在关键路径上的等待缩短，才改善完整更新；NPU 空闲还可能来自通信依赖、流水线空泡或同步，不能仅凭低利用率认定数据慢或 Host Bound。时间线解释见 [采集与分析](../ascend_profiling/collection-and-analysis.md)。

## worker 与预取

`num_workers` 增加数据准备进程；并行解码可能受益，已驻留内存且处理很轻的数据则可能由 IPC、调度和复制主导。worker 数应结合实际读取数据的 rank 数考虑，不能把每个进程的设置当作整机总量。[PyTorch DataLoader][loader]

普通 DataLoader 的 `prefetch_factor` 是每个 worker 预取的 batch 数；更多预取可吸收供给抖动，却不能长期弥补平均供给速率不足。队列扩大增加 Host 内存与共享内存压力。`persistent_workers` 主要减少迭代器/epoch 之间的重建成本，不保证稳态更快；`pin_memory` 提供锁页缓冲，也不等于 H2D 已与计算重叠。[参数语义][loader]

增加 worker 或改变持久化方式可能改变随机预处理状态；IterableDataset 的分片、sampler 顺序及 loader 保存状态也与实现有关。`in_order=False` 允许乱序返回，不能作为保持样本顺序的等价优化。有效数据与进度相同是比较条件，详见 [批量与累积](batch-and-accumulation.md)。

## CPU 亲和性与任务队列

| 机制 | 含义与权衡 |
| --- | --- |
| `CPU_AFFINITY_CONF` | TorchNPU 的模式 1 在核区间内粗粒度绑定，模式 2 进一步分隔热点线程。可减少迁移和争抢，也可能挤压 loader 等线程；默认核区间划分不等于当前容器允许核或真实 NUMA 映射。具体选项受版本/芯片限制。[绑核文档][affinity] |
| `TASK_QUEUE_ENABLE` | 所引 TorchNPU 文档的 Level 1 将部分下发工作分到二级流水，Level 2 进一步移动 workspace 相关工作，仅在二进制场景生效。它优化下发而非数据读取；更多并发可能提高 NPU 峰值显存，多线程跨流依赖还需保持正确下发顺序。[任务队列][queue] |

`ASCEND_LAUNCH_BLOCKING=1` 强制同步执行，帮助错误栈定位，并使任务队列配置失效；同步诊断的时间不能代表正常异步性能。[阻塞调试][blocking] 环境变量未显式设置不等于功能关闭，实际默认值与支持范围取决于运行时版本。

## 搬运与同步

锁页内存、异步提交和实际搬运计算重叠是不同条件。`non_blocking=True` 不能单独证明重叠；还需要实现支持、独立工作及正确的 stream/event 依赖。源缓冲过早复用或在数据到达前消费会破坏正确性，更多 staging 缓冲也有内存成本。

`.item()`、`.cpu()` 或显式同步可能暴露设备等待，但同步处耗时也可能包含此前排队的设备工作。必要的训练决策与数据依赖不能随意删除；设备下发、同步、实际计算的耗时不能简单累加。融合与图执行的机制见 [算子优化](../ascend_operators/operator-optimization.md) 和 [图执行](graph-execution.md)。

## 当前 FlagScale 数据入口

FlagScale `a3088a5` 的两条路径不同：

- `flagscale/train/megatron/training/datasets/data_samplers.py` 的普通 loader 使用 `args.num_workers`，固定 `pin_memory=True`，在 workers > 0 时启用 `persistent_workers`；构造处未显式传入 `prefetch_factor`，沿用安装版本的 DataLoader 默认值。`external` 分支直接返回外部 loader。
- `flagscale/train/megatron/train_qwen35.py` 将 `args.num_workers` 传入 Energon `WorkerConfig`，训练使用 `get_savable_loader`。普通 DataLoader 的参数和恢复语义不能据此直接套用到 Energon。

这些是数据构造路径，不证明所有 rank 都读取数据或所有参数均已暴露为 YAML；已启用的功能不构成新增优化。

[loader]: https://docs.pytorch.org/docs/stable/data.html
[affinity]: https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/performance_tuning/CPU_AFFINITY_CONF.md
[queue]: https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/TASK_QUEUE_ENABLE.md
[blocking]: https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/ASCEND_LAUNCH_BLOCKING.md
