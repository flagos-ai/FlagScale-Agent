# 三仓配置消费与平台能力

FlagScale 负责配置组织与启动，Megatron-LM-FL 负责训练、并行和模型集成，TransformerEngine-FL 通过公共接口与平台实现提供算子能力。配置是否生效取决于三仓的实际消费链路；磁盘上的 checkout 不一定是训练进程加载的源码。

配置可比性、计时与状态语义见 [训练测量与正确性](measurement-and-records.md)，跨维度代价见 [搜索与权衡](search-space.md)。

## 环境指纹与实际路径

实际加载路径由训练 Python 环境、模块 `__file__`、editable 安装和搜索路径决定；commit 相同但工作树修改不同，运行行为仍可能不同。多节点的包版本、源码和环境差异也属于配置可比性的条件。

| 层 | 影响运行行为的组成 |
| --- | --- |
| 硬件与系统 | 昇腾型号、逻辑设备映射、显存、CPU/NUMA、驱动/固件、节点和网络拓扑 |
| 运行时 | CANN、Python、PyTorch、torch_npu、HCCL，以及实际 collective backend；使用 FlagCX 等中间层时一并记录 |
| FlagScale | Hydra defaults、runner/backend、训练入口、所有 `experiment.envs` 中与本任务有关的覆盖 |
| Megatron-LM-FL | 平台选择、Ascend overrides、并行组、训练计时器、优化器和 checkpoint 格式 |
| TE-FL | public facade、manager/registry、NPU vendor 实现、实际绑定的 attention/norm/GEMM/permutation 等 callable |
| 扩展包 | 实际使用的 `transformer_engine_npu`、`fla_npu`、自定义算子版本和构建信息 |

FlagScale 的 `auto_tuner`、`runner_train`、`train_config` 与 CLI 模块决定配置生成和启动；Megatron 的 `parallel_state.py`、参数解析、Transformer 配置及 Ascend 插件决定分组和平台覆盖；TE 的 manager、注册表、NPU vendor 与 attention 模块决定算子分派。文件路径和 backend id 可能随版本变化，注册项存在不等于运行时选中了该实现。

导入模块或构造探测对象也可能初始化设备，因此它们与纯源码读取具有不同的副作用。

## 能力描述与证据层次

功能能力由用户配置路径、最终 argv/对象、单位、默认值、消费者、前置条件和实际运行路径共同描述。`supported / unsupported / unverified` 区分已证实支持、明确不支持与未知；配置运行失败与能力不支持并不等价。

例如 `sequence_parallel` 受 TP、MoE 组合和 Ascend override 约束；`moe_permute_fusion` 依赖公共接口实际分派到兼容 NPU 实现。TE 安装成功不证明所有 op 已注册；只完成配置解析也不证明未知能力已获得运行兼容性。

训练公共调用路径应可追溯：

```text
FlagScale resolved config / generated argv
  → Megatron training / model / Ascend override
  → TransformerEngine-FL public API / dispatch manager
  → 当前注册的 NPU adapter
  → torch_npu / transformer_engine_npu / 其他实际设备实现
```

backend 选择日志、绑定对象的模块/类/源码位置、代表性前后向调用和 profiler 中的设备算子分别说明配置选择与实际执行。`vendor.npu` 文件存在或其他运行的日志不能证明当前绑定。reference 回退会改变执行路径，只有回退范围和数学语义已明确时，结果才具有可解释性；意外回退会破坏性能可比性。

NVIDIA FP8/FP4、Userbuffers、NCCL、CUDA Graph 或 `NVTE_*` 的存在不证明 NPU 支持，也不能反向推断全部不可用。CUDA 与 HCCL/NPU 参数不能按名称映射；相似接口的默认值、同步、精度和 fallback 语义可能不同。MindSpeed 的同名参数也不自动适用于这个三仓栈。

### 有效配置与跨特性证据

“用户请求值、生成值、运行时最终值”可能不同。Hydra/defaults、generator、recipe helper、Ascend override 和初始化 gating 都可能隐式覆写、删除、回退或联动字段。因此一个 helper 的影响是完整有效配置的差异集合；YAML 中的 true 或启动未报错不证明功能已执行。

可用 CLI 由实际入口的 parser、dataclass 生成器及排除列表共同决定，不由另一个仓库的同名字段保证。在将 `batch_p2p_comm` 排除出自动参数生成、并由 `overlap_p2p_comm` 取反派生的 FlagScale 版本中，它不是独立 CLI 维度；关闭 overlap 的入口是已注册的 `--no-overlap-p2p-communication`。不要从字段名猜测 no/disable-batch 开关；其他版本须重新核对注册和派生逻辑。省略负开关可依赖当前 parser 默认值，但结果审计仍须检查最终对象，不能把未识别参数导致的启动失败记为 VPP 算法不支持。

重计算、overlap 和图模式的兼容性取决于模块边界、rank groups、schedule、buffer 生命周期、梯度累积/RNG 及 NPU 实现条件，详见 [重计算交互](recompute.md) 与 [图执行边界](graph-execution.md)。配置开关只能选择已有路径；缺少所需公共接口或后端实现时，仅修改配置不能补足能力。

训练 parser 的环境前置条件与设备算子支持是两层约束。例如某个 TP/CP 校验分支要求 `CUDA_DEVICE_MAX_CONNECTIONS=1`，不代表它是所有 NPU 任务的默认要求；其前置断言失败也不证明 TP/CP 算子不支持。未选中 backend 中的限制不能代表实际路径。

使用仓库自带搜索器时，其生成、计时与停止机制见 [FlagScale native tuner](native-tuner.md)。

## 源码参考

- [FlagScale](https://github.com/flagos-ai/FlagScale)：配置生成与启动。
- [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL)：并行、参数与 Ascend override。
- [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL)：注册与公共接口。
