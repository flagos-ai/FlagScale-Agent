<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# FlagScale native tuner 的生成与结果语义

本文仅解释仓库原生 auto_tuner，和 Agent 自行生成候选、逐次运行训练不是同一执行机制。模块与字段以所用 FlagScale 版本为准；配置消费的一般机制见 [三仓配置](stack-capabilities.md)。

## 候选如何生成与筛选

searcher 决定搜索维度和枚举；generator 将策略映射到训练配置；prune/memory_model 估计可行性；recorder 解析并排序性能；tuner/runner 控制作业与退出。构造完整 tuner 可能创建输出或删除旧日志，因此不等同于纯配置解析。

以下仅描述具有相应源码分支时的行为，不代表所有版本：

| 源码条件 | 对配置与测量的影响 |
| --- | --- |
| `TrainAutoTuner.tune()` 在 `control.run_best=true` 时自动启动最优项 | 搜索完成后可能继续占用设备；false 表示关闭此额外启动行为 |
| `Generator.gen()` 删除 checkpoint load、train_samples、部分 warmup/decay/rampup 字段，并设置短跑步数与 auto_tune | 生成配置可能改变训练起点或调度；原配方等价性取决于生成后的有效差异 |
| `Recorder.grep_performance()` 找到一个有指标的日志，删第一条后取平均 | 该数字不必然对应稳定窗口、最慢 rank 或统一更新口径 |
| 停止或退出异常的策略仍保留 performance，排序只依赖 performance | 数值排序不能证明所有 worker 完成或质量通过 |
| `max_time` 仅在任务之间检查，首个任务单次时限另有倍增逻辑 | 总时限可能被正在运行的任务和退出延迟突破；严格上限需要独立的作业范围控制 |
| resume 初始化删除 breakpoint task 日志 | 恢复操作具有破坏既有运行证据的副作用，不等同于纯读取进度 |
| memory model 使用 `gpu_memory` / `gpu_utilization` 并按估计剪枝 | 字段名不证明 NPU 适配；未经目标设备校准的估计不能证明配置必然 OOM |
| 搜索维度和 `args_mapping` 是固定集合 | 未映射的 ETP、overlap、TE 后端等字段不因加入 space 就自动成为有效搜索维度 |

## 控制字段的含义

这些字段属于 `experiment.auto_tuner`，不是普通训练参数：

| 字段 | 含义与边界 |
| --- | --- |
| `control.run_best` | 搜索后是否额外启动最优配置；关闭时省去这次后续作业 |
| `control.max_time` / `max_time_per_task` | 搜索与单任务的时间设置；严格墙钟上限仍取决于检查时机、首任务逻辑和停止范围 |
| `control.train_iters` | 候选短跑长度，可能联动 LR/weight decay 和数据进度语义 |
| `performance.name` / `order` | 指标匹配及排序；`order: ascend` 表示升序，与昇腾硬件无关，耗时与吞吐需要相反排序方向 |
| `space` | 显式值、固定项、映射及自动扩张规则共同决定有效候选，不是任意字段的透传空间 |

生成配置的实际差异可能包含 checkpoint、RNG、consumed samples、LR/warmup/rampup、GBS 和 routing。普通训练 dryrun 也不自动覆盖 tuner 的生成变更。调度与跨布局恢复细节见 [状态恢复](state-and-resume.md)，内存模型边界见 [显存知识](memory-and-sharding.md)。

## 输出能证明什么

history 中的 performance 受日志选择、窗口和退出判定影响，单独一个排名数字不证明全部 worker 完成、质量通过或稳态收益。生成配置、实际 argv/env、原始日志和作业退出各自提供不同证据；比较定义见 [训练测量](measurement-and-records.md)。

配置解析、实际 NPU 路径可运行、数值通过和性能收益不能互相替代。消费者、默认值或依赖版本变化也可能改变行为，跨版本单次数字不足以归因到某个开关。

源码参考：[FlagScale auto_tuner](https://github.com/flagos-ai/FlagScale/tree/main/flagscale/runner/auto_tuner)。
