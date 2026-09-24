<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend kernel 的资源与性能机制

本文解释 Triton-Ascend / Ascend C 的分块、任务划分、片上资源和指令流水。公共接口、训练梯度及三层性能口径见 [算子调用知识](operator-optimization.md)。

## 执行资源与瓶颈

Cube 主要承担矩阵计算，Vector 承担向量计算，Scalar 处理标量控制及部分地址/索引工作，MTE 承担数据搬运。流水可以部分并发，也可能因依赖等待；占用比不是相互排斥的耗时分解，不能用固定百分比直接判定瓶颈。[Triton-Ascend 性能分析][profiling]

UB、L1、L0 等存储层用途不同，容量不能简单相加。可用资源、对齐、核数与编译接口由 SoC、CANN 和编译器版本决定；示例中的 UB 容量、固定 grid 或 CUDA 的 warp/stage 经验不构成所有 Ascend kernel 的约束。

## 可形成优化假设的机制

| 线索 | 机制 | 收益与代价 |
| --- | --- | --- |
| 大量短任务，下发/初始化占比高 | 合并逻辑任务，每个 program 在核内跨步处理多个 tile | 减少多轮调度，但可能增加串行工作或尾核不均；逻辑 grid 与物理核数不是同一概念。[任务划分][programming] |
| 搬运指令多或重复读取 | 连续分块、合并搬运、复用已载入数据 | 较大 tile 提高有效搬运粒度，却增加活跃资源；整体搬入再 gather 是否值得取决于额外读取量与复用率。[编程指南][programming] |
| CopyIn、Compute、CopyOut 串行 | 多 tile 与双缓冲形成搬运/计算流水 | 独立 tile 才有重叠空间；增加缓冲占用，过小工作量或纯计算主导时收益有限。[Ascend C 流水][pipeline] |
| 尾块填充造成 Vector→MTE 依赖 | 在被屏蔽位置不影响任何有效结果时省去不必要填充 | Triton-Ascend 的 `care_padding=False` 是条件化实现示例；它不代替越界 mask，不能让未定义值进入有效归约。[迁移性能指南][migration] |
| Scalar 指令或索引处理突出 | 简化重复地址计算，使用支持的表达与数据类型 | 某些版本/芯片的整数运算可能被标量化；改窄索引或改用浮点比较需要证明范围、溢出与精确表示，不可直接照搬示例 cast。[迁移性能指南][migration] |
| 融合后反而变慢或编译溢出 | 缩小活跃 tile、缩短中间量生命周期，必要时拆分计算 | 融合节省写回，但同时活跃量可能迫使更小分块或阻碍多缓冲；单 pass 与多 pass 是访存、资源和数值之间的权衡。[资源说明][programming] |

这些机制不限定优化次序。矩阵计算的 tile 形状还影响 Cube 利用与边界浪费；不同核内流水之间的事件依赖不能因“减少同步”而省略。[Ascend C 流水与同步][pipeline]

## Tiling 与片上资源活跃量

资源峰值由同时存活的对象决定：输入/输出、累加器、升精度中间量、offset/index/mask、padding 和多缓冲副本都可能参与占用。源码的逻辑元素数不等于编译后的分配；非连续访问和对齐扩张可能放大实际占用。[迁移中的资源扩张][migration-memory]

tile 变大可能减少循环和搬运指令，却提高资源压力；变小可能恢复流水能力，也可能增加循环、Scalar 与同步开销。编译器报告和真实输入下的测量用于判断这类取舍，不能只按“剩余 UB 越少越好”选择配置。

双缓冲不是把所有可用内存机械减半：其额外占用取决于哪些队列/张量有多个副本及各自活跃期。Ascend C 的 `TPipe.InitBuffer` 用缓冲块数量和每块字节数描述队列存储；收益来自不同 tile 的并发，而非缓冲数本身。[双缓冲机制][pipeline]

## 自动调优的作用与边界

`triton.autotune` 比较候选配置并按 key 缓存选择；key 用于区分会影响配置选择的输入条件，不能用单一 shape 的最优值推断其他 stride 或规模同样最优。tile 影响任务划分时，grid 必须随候选 meta 变化。[Autotune][autotune]

所引 Triton-Ascend 指南中，导入 `triton.backends.ascend.runtime` 后的扩展支持 `configs=[]` 自动生成 Tiling 候选；它不等于自动搜索所有编译选项，也不保证全局最优。手写 `triton.Config` 是另一条路径。此处描述社区能力，不证明当前安装版本已提供相同接口。[能力边界][autotune]

调优会反复执行 kernel：inplace、原子累加或状态更新若没有恢复初态，会污染后续测量及正确性。首次编译/搜索成本与缓存命中后的执行成本不同，选择范围越大，准备成本通常越高；自动生成失败也不证明该算子无法实现。[副作用与成本][autotune]

## 正确性与测量边界

尾块 mask 同时涉及合法访存和计算语义。归约的填充值必须与运算匹配，例如 sum 的加法零元与 max 的负无穷；仅在最终 store 屏蔽尾部，不能补救其先前污染有效归约。

降低精度或改变归约顺序可能影响误差；更高精度累加又会增加资源代价。索引、offset 和总展开 token 数需要覆盖真实取值范围，不能用一个固定整数边界代表全部算子。

公共 API 允许别名，不代表某个 kernel 编译器支持所有指针别名。所引 Triton-Ascend FAQ 对多个输入指针指向同一存储列有限制；原地改写或减少 clone 必须同时满足接口与编译器假设。[指针别名约束][faq]

`msprof op` 的设备分析与 simulator 指令流水用于解释算子瓶颈；仿真时序不等于真实训练耗时。不同候选、输入桶与编译版本需要可区分，同名 kernel 的聚合可能掩盖差异。[分析口径][profiling] 公共调用的输入整理、辅助分配和同步仍属于候选成本，kernel 局部加速不能单独证明训练收益。

[profiling]: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/profiling.md
[programming]: https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/programming_guide/index.md
[pipeline]: https://www.hiascend.com/developer/techArticles/20240819-1
[migration]: https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/migration_guide/performance_guidelines.md
[migration-memory]: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md
[autotune]: https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/autotune_guide.md
[faq]: https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/FAQ.md
