<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# NPU profiling 的采集机制与分析依据

Profile 描述采样窗口内记录到的活动；它能支持哪些结论，取决于事件覆盖、字段含义和统计口径。完整 optimizer step 与 A/B 比较的指标定义见 [训练测量依据](../ascend_training/measurement-and-records.md)。

## 采集机制

`torch_npu.profiler` 的可用接口、采集层级和事件字段取决于安装版本及设备。调用栈、shape 等采集选项会改变开销与可分析范围；带 profiler 的步时不能直接代替无 profiler 的性能测量。接口兼容涉及构造、生命周期和导出，成功创建对象不等于已经取得 NPU 事件。

采样 schedule 按 `prof.step()` 调用推进，与训练 iteration 的对应关系由调用位置决定。CPU 的 `ProfilerStep#` 或训练调用标注反映主机时间窗；NPU 异步任务可能跨越边界，因此主机标注结束不代表该次更新的设备工作已经结束。

不同 rank 可能承担不同 PP stage 或负载，局部采样只反映被采集的范围。原始 PROF、数据库和 trace 的解析依赖对应格式的工具；已有逐任务 CSV 的离线区间统计不需要 NPU 执行环境。采集接口参考 [Ascend PyTorch Profiler 文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0033.html)，具体能力以安装版本为准。

### 采集参数与 timeline 的影响

下表区分**增加哪些可见信息**与**采集怎样扰动执行**。开销随模型、事件密度、软件版本和存储条件变化，没有通用百分比；“更多事件”也不意味着时间测量更接近无采集运行。参数语义参考 [Ascend Profiler 接口说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0033.html)，开销方向参考 [官方最小膨胀采集说明](https://www.hiascend.com/document/caselibrary/detail/profilingcase_007)。

#### `profile(...)`：框架与设备事件

| 参数 | 含义及 timeline 可见信息 | 开销与判断边界 |
| --- | --- | --- |
| `activities` | `CPU` 记录 PyTorch 框架事件；`NPU` 记录 CANN 与设备事件。两者同时开启才能结合框架下发与设备执行分析 | 仅采 NPU 可减少框架侧采集，但缺少 PyTorch 调用上下文；仅采 CPU 无法判断 NPU 实际执行与空洞 |
| `record_shapes` | 为框架算子记录输入 shape/type，帮助区分同名算子；依赖 CPU 采集 | 增加逐算子元数据记录与数据量，不会让设备时间戳更精确 |
| `with_stack` | 记录框架与 CPU 算子调用栈，便于追到调用位置；依赖 CPU 采集 | 官方列为主要高开销项；栈采集可能拖慢 host 下发，放大设备等待，影响原有重叠 |
| `with_modules` | 记录 module 层级的 Python 调用信息；依赖 CPU 采集 | 需要模块归属时通常比完整 `with_stack` 开销小，但仍会引入开销；二者都关闭才是不采这两类调用信息 |
| `profile_memory` | 记录内存分配、释放及占用，用于内存分析 | 增加内存事件记录开销，不是观察计算/通信 timeline 的必要项；独立 `memory_timeline` 导出还需要 shape、调用信息等配套数据 |
| `with_flops` | 请求算子浮点操作信息，依赖 CPU 采集 | 不是硬件 FLOPS 实测；上述版本文档标注暂不支持解析，不能期待它增加有效 timeline 轨道或据此计算 MFU |

上表除 `activities` 外的布尔开关默认均为 `False`。只有需要对应归属或内存证据时，额外信息才有助于当前分析。

#### `schedule(...)`：采集范围

顺序为一次性的 `skip_first`，随后循环 `wait → warmup → active`：

- `skip_first` 只在最开始跳过指定步数；`wait` 每个周期跳过指定步数，这些阶段不进行活动数据采集。
- `warmup` 是 profiler 的准备阶段，不作为正式采样窗口；它不能代替模型编译、缓存等训练预热。
- `active` 决定连续记录多少步；`repeat` 决定周期数，`0` 表示持续循环直到停止。增大两者会增加总采集量和解析成本；缩小窗口并不会消除单步采集开销。

例如从第 1 次训练更新前启动，并在每次完整更新后调用一次 `prof.step()`，`wait=3, warmup=1, active=2, repeat=1, skip_first=0` 对应跳过更新 1–3、准备更新 4、记录更新 5–6。`prof.step()` 若放在 microbatch 后，计数也会变为 microbatch。未配置 schedule 时默认持续记录，不会自动跳过训练前期。

#### `experimental_config`：层级与硬件指标

以下参数传给 `torch_npu.profiler._ExperimentalConfig(...)`，再作为 `profile` 的 `experimental_config`。

| 参数/取值 | 增加的信息 | 开销及 timeline 影响 |
| --- | --- | --- |
| `profiler_level=Level0` | 基础框架、设备任务与算子信息 | 层级中开销较低，适合先看整体时序；部分详细事件和通信分析产物不在此层级 |
| `profiler_level=Level1` | 在 Level0 上增加 AscendCL、通信分析数据，并支持 AI Core 指标 | 比 Level0 采集更多数据；需要通信明细时有用，指标采集还受 `aic_metrics` 控制 |
| `profiler_level=Level2` | 在 Level1 上增加 Runtime、AI CPU 等详细数据 | 数据量和扰动进一步增加，适合定位低层调用，不宜作为每次采集的默认值 |
| `aic_metrics` | `PipeUtilization` 等选项提供 AI Core 计数器指标，`AiCoreNone` 关闭指标采集 | 增加硬件指标采集成本，主要补充算子详情，不是让时间线更细；Level0 不采这类指标，提高到 Level1/2 时还需留意指标默认值 |
| `l2_cache` | 采集 L2 Cache 指标，生成相应统计 | 额外的硬件指标采集；不需要缓存证据时保持关闭，不把开启前后的算子时长直接视为同口径 |
| `mstx` | 采集代码中已有的 MSTX 自定义标记，用于识别阶段/范围；旧接口名为 `msprof_tx` | 开销取决于打点密度；开关本身不会自动生成层、FWD/BWD 等模型语义 |
| `export_type` / `data_simplification` | 前者选择 Text/Db 等解析产物；后者控制导出后的文件精简 | 主要影响解析、存储和后续可用数据，不会降低已发生的事件采集开销；精简保留哪些文件取决于版本 |

#### 回调、落盘与时序扰动

`on_trace_ready` 在采集窗口完成时调用处理函数；`tensorboard_trace_handler(dir_name, worker_name, analyse_flag, async_mode)` 中，前两项设置目录与工作进程标识，**`worker_name` 不负责选择采集 rank**。`analyse_flag=True` 默认自动解析，`False` 留待离线解析；`async_mode=False` 默认同步解析，`True` 让解析异步进行。异步解析减少主流程阻塞，但仍占用 CPU、内存和 I/O，也不消除采集本身的开销。参见 [导出回调参数说明](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/torch_npu-profiler-tensorboard_trace_handler.md)。

由这些机制可知：timeline 中的下发间隔、设备空洞和计算通信重叠，可能同时受到模型与 profiler 扰动；启停、flush、同步解析附近的墙钟耗时不能全算作训练计算。多 rank 同时写共享存储还可能放大 I/O 等待；只在部分 rank 开高开销采集，也可能通过通信等待影响其他 rank。

用于观察整体时序的低开销起点是 CPU+NPU、Level0、短 active 窗口，关闭 shape、stack、modules、memory 和硬件指标等附加采集；通信证据不足再升 Level1，缺调用归属再补 modules/stack，缺 shape 或内存证据再单独开启。采集扰动可用同配置稳态步时的 `T_profile / T_no_profile - 1` 估算，并把启停/解析耗时单列；优化收益仍以无 profiler 运行确认。

## 数据含义

逐任务表包含每次任务的起点和时长，可以重建记录到的时间区间；只有 Count、总时长或平均值的汇总表不能定位任务间空洞。数据能力由实际列决定，不能仅根据 `op_summary`、`operator_details` 等文件名判断。

| 信息 | 常见字段与含义 |
| --- | --- |
| 起点与时长 | `Start Time(us)` / `Duration(us)`，或 `Task Start Time(us)` / `Task Duration(us)`；单位为微秒 |
| 设备 | `Device ID`、`device_id` 等；确定任务所属设备 |
| 名称与类型 | `Name` / `Op Name`、`Type` / `OP Type`；任务类型或加速核类型与算子类型不是同一概念 |
| 流 | `Stream ID`；区分流中的任务与并发关系 |
| 输入 | `Input Shapes`、`Input Data Types`；区分同名算子的形状和精度 |

通信类型由导出工具定义，可能以 `COMMUNICATION`、`hcom_` 等表示，名称不含 `HCCL` 不代表没有通信。CANN 8.1.RC1 的 `Task Wait Time(us)` 表示前后任务间隔，不是 host enqueue 延迟；字段语义参考对应版本的 [op_summary 说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0067.html)。

任务区间合并以同一设备、同一时钟域为前提。跨 rank 时间戳差值需要可靠的时钟对齐；host 与设备任务的对应关系需要关联 ID、flow 或其他可靠的任务匹配证据，单靠时间包围关系不能确定。

## 统计口径

设窗口为 `W=[start,end)`，同一设备内的任务区间为 `I_i`：

- **窗口时长**：`|W|`，即分析范围的 wall-clock 时长。
- **任务累计时长**：`Σ|I_i ∩ W|`，并发任务会重复计时，可能大于窗口时长。
- **记录覆盖时长**：`|⋃(I_i ∩ W)|`，并发区间只计一次。
- **未覆盖时长**：`|W| - |⋃(I_i ∩ W)|`，表示没有被所选任务记录覆盖的时间，不直接等于设备空闲。

窗口统计包含所有与窗口相交的任务，并裁剪到窗口边界。若输入遗漏、读取截断或字段无效，覆盖率和空洞结论只能用于已知记录；空文件不能证明设备完全空闲。

计算与通信各自的区间并集，其交集表示时间重叠。通信未与计算重叠的部分是否延迟了训练，还取决于生产、等待和消费关系。算子累计时长、区间并集与窗口时长是不同口径，不能相加为 step 延迟；记录覆盖率也不等于算力利用率。

## 归因依据

| 观察 | 支持归因所需的证据 | 能得出的结论范围 |
| --- | --- | --- |
| 同类算子累计耗时高 | 名称、shape/dtype、调用次数，以及层、FWD/BWD、重计算标记 | 可筛选热点；累计时长不等于优化后能节省的 step 时间 |
| 反复出现未覆盖窗口 | 同一时钟域内的 host 下发、数据准备、copy/sync 和前后任务 | 可区分数据等待、下发或同步等候选原因；空洞本身不能确定根因 |
| 通信耗时长或重叠少 | 通信报告、计算区间和生产/等待/消费依赖 | 能定位等待时才能判断关键路径上的通信代价 |
| 某个 rank 的 step 较慢 | 完整 step、PP stage 职责、负载与拓扑 | 差异可能来自负载或通信，不能只凭固定百分比认定慢卡 |

缺少模型标记时，重复 kernel 模板只能提供结构线索；融合、重计算和 MoE 动态路由都会影响次数。Host 嵌套事件的累计时长也会重复计时；低 host 覆盖不能单独证明 host-bound，设备任务覆盖高也不能在缺少硬件计数器等证据时证明算力饱和。
