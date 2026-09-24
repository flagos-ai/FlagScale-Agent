# Chapter 09: Megatron-LM 集成接口 — 源码深度分析

## 源码位置

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `megatron/core/extensions/transformer_engine.py` | 2837 | Megatron↔TE 全量适配层 |
| `transformer_engine/pytorch/linear.py` | ~1670 | TE Linear 基础实现 |
| `transformer_engine/pytorch/distributed.py` | ~1950 | TE checkpoint/comm原语 |
| `tests/plugin/plugin/test_cuda_comm_overlap_contract.py` | — | vendor overlap factory 契约测试 (PR#115 新增) |

---

## 0. Vendor 通信重叠 factory 契约 (PR#115, 2026-09-08)

> 适用于 MetaX/MUSA/Hygon/Iluvatar/Enflame 等 vendor 后端接入 TE-FL 时，
> 实现共享的 comm overlap factory 接口必须满足的契约（PR#115 实测）：

1. **共享 factory 接口新增参数**：`use_cublasmp` 与 `comm_type`。
2. **CUDA 路径的 overlap 选项一律按 keyword 转发**，并归一化通信类型
   （避免位置参数顺序漂移导致 vendor 侧错绑）。
3. **vendor legacy 构造参数保留**：MUSA/MetaX/Hygon/Iluvatar/Enflame 的旧构造签名不破坏。
4. **不支持的 cuBLASMp 组合显式 reject**（fail-fast），不允许静默降级。

回归防护：`tests/plugin/plugin/test_cuda_comm_overlap_contract.py` +
`test_vendor_comm_overlap_contract.py` —— 改 factory 签名或新增 vendor 后端时先跑这两个契约测试。

---

## 1. 架构总览

```
Megatron 模型层
    │
    ├── TELinear (L675)                  ← 基类: te.pytorch.Linear + Megatron参数适配
    │   ├── TEColumnParallelLinear (L1145) ← 列切分: output_size/tp, forward AllGather
    │   ├── TERowParallelLinear (L1255)    ← 行切分: input_size/tp, forward ReduceScatter
    │   └── TELayerNormColumnParallelLinear (L918) ← 融合LN+Linear一体
    │
    ├── TEDotProductAttention (L1359)    ← DotProductAttention + CP/GQA/SWA/FP8支持
    ├── TENorm (L616)                    ← LayerNorm/RMSNorm 封装
    │
    ├── TEDelayedScaling (L2447)         ← FP8 DelayedScaling recipe 适配
    ├── TECudaRNGStatesTracker (L2483)   ← RNG tracker Megatron兼容接口
    ├── te_checkpoint (L2518)            ← TE激活检查点, 兼容distribute_saved_activations
    └── get_cpu_offload_context (L2558)  ← CPU offload跨版本兼容封装
```

## 2. TELinear：基类设计 (L675-916)

### 2.1 初始化参数映射

```python
class TELinear(te.pytorch.Linear):
    def __init__(
        self,
        input_size, output_size,
        parallel_mode: Optional[str],  # "column" / "row" / "duplicated"
        config: ModelParallelConfig,
        ...
    ):
        # Expert TP通信禁用逻辑
        explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)
        if explicit_expert_comm:
            # MoE场景: TP通信由token_dispatcher处理，这里禁用TE内置通信
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)  # 预切分尺寸
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)
            te_parallel_mode = None    # 告知TE不做通信
            tp_size = 1
            tp_group_for_te = None

        super().__init__(
            fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
            tp_group=tp_group_for_te,
            sequence_parallel=config.sequence_parallel,
            get_rng_state_tracker=get_cuda_rng_tracker,
            **extra_kwargs,
        )
```

### 2.2 is_first_microbatch 优化 (L720, L878-884)

```python
self.is_first_microbatch = True  # 初始化时设置

def forward(self, x):
    _is_first_microbatch = (
        None if self.disable_parameter_transpose_cache
        else self.is_first_microbatch
    )
    out = super().forward(x, is_first_microbatch=_is_first_microbatch)
    self.is_first_microbatch = False  # 之后的microbatch不缓存
```

**机制**：TE Linear在首个microbatch时会缓存权重的转置版本（用于FP8 GEMM的列式访问），后续microbatch复用缓存。`disable_parameter_transpose_cache=True`时传`None`，每次都计算。

### 2.3 delay_wgrad_compute支持 (L730-734, L912-915)

```python
# 初始化时透传给TE (需要TE >= 2.3.0)
if self.config.delay_wgrad_compute:
    extra_kwargs["delay_wgrad_compute"] = True

# 提供backward_dw方法供外部显式调用
def backward_dw(self):
    if self.config.delay_wgrad_compute:
        super().backward_dw()  # 触发延迟的wgrad计算
```

**用途**：在PP interleaved schedule中，可将wgrad计算延迟到通信完成后，与下一层forward重叠。

### 2.4 Expert并行参数标记 (L846-858)

```python
for param in self.parameters():
    if is_expert:
        # Expert参数: 梯度在expert_data_parallel组上reduce
        setattr(param, "allreduce", not self.expert_parallel)
    else:
        # 普通参数: 梯度在DP组上reduce
        setattr(param, "allreduce", True)
        if parallel_mode == "duplicated":
            # duplicated模式: 权重在TP各rank复制，需额外TP组reduce
            setattr(param, "sequence_parallel", config.sequence_parallel)
            setattr(param, "tensor_model_parallel", False)  # 标记为非TP分片
```

## 3. TELayerNormColumnParallelLinear：融合LN+Linear (L918-1143)

### 3.1 设计动机

```
分离方式:
  RMSNorm kernel → 写activations → ColumnParallelLinear kernel
  代价: 2次kernel launch + 1次中间tensor内存读写

融合方式 (TELayerNormColumnParallelLinear):
  单kernel完成RMSNorm + Linear
  代价: 1次kernel launch, 无中间tensor
```

### 3.2 Userbuffers配置映射 (L988-1019)

```python
if self.config.tp_comm_overlap:
    # TE >= 1.5.0: 使用新版overlap标志
    extra_kwargs["ub_overlap_ag"] = (
        self.config.tp_comm_overlap_ag    # 显式配置优先
        if hasattr(self.config, "tp_comm_overlap_ag")
        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag  # fallback
    )
    extra_kwargs["ub_overlap_rs_dgrad"] = self.config.tp_comm_overlap_rs_dgrad
    
    # 按buffer name禁用特定层的overlap
    if tp_comm_buffer_name == "qkv" and self.config.tp_comm_overlap_disable_qkv:
        extra_kwargs["ub_overlap_ag"] = False
    if tp_comm_buffer_name == "fc1" and self.config.tp_comm_overlap_disable_fc1:
        extra_kwargs["ub_overlap_ag"] = False
    
    # buffer name必须是预定义值之一
    extra_kwargs["ub_name"] = tp_comm_buffer_name  # "qkv", "proj", "fc1", "fc2"
```

**buffer name的意义**：TE内部为每个named buffer预分配固定大小的userbuffer内存。名称对应模型中的固定位置：
- `qkv`: Q/K/V投影的输入AllGather
- `proj`: Attention输出投影的ReduceScatter
- `fc1`: MLP第一层（Gate/Up）的AllGather
- `fc2`: MLP第二层（Down）的ReduceScatter

### 3.3 symmetric_ar_type参数 (L1021-1026)

```python
# FlagScale扩展: NVLink对称内存AllReduce
if self.config.symmetric_ar_type is not None:
    extra_kwargs["symmetric_ar_type"] = self.config.symmetric_ar_type
    # 要求: torch >= 2.7 + TE >= 2.3
```

### 3.4 split_te_layernorm_column_parallel_linear (L325-403)

提供将融合层拆分为独立层的工具函数（用于模型结构变换/checkpoint转换）：

```python
def split_te_layernorm_column_parallel_linear(fused_layer, config, ...):
    # 1. 创建独立TENorm
    norm_layer = TENorm(config=config, hidden_size=fused_layer.in_features)
    norm_layer.weight.copy_(fused_layer.layer_norm_weight)  # 权重迁移
    
    # 2. 创建独立TEColumnParallelLinear
    linear_layer = TEColumnParallelLinear(...)
    linear_layer.weight.copy_(fused_layer.weight)
    
    # 3. 迁移FP8元数据
    for key in fused_layer.fp8_meta:
        linear_layer.fp8_meta[key] = fused_layer.fp8_meta[key]
    
    return norm_layer, linear_layer
```

## 4. TEColumnParallelLinear & TERowParallelLinear (L1145-1357)

### 4.1 列并行 vs 行并行参数差异

| 参数 | TEColumnParallelLinear | TERowParallelLinear |
|-----|----------------------|-------------------|
| parallel_mode | "column" | "row" |
| output_size | output_size / tp_size | output_size (全) |
| input_size | input_size (全) | input_size / tp_size |
| Forward comm | AllGather (SP时) | ReduceScatter (SP时) |
| Backward comm | ReduceScatter | AllGather |
| ub_overlap | ub_overlap_ag=True | ub_overlap_rs=True |

### 4.2 sharded_state_dict (分布式检查点)

```python
# TEColumnParallelLinear (列切分, 沿axis 0)
def sharded_state_dict(self, prefix, ...):
    return make_sharded_tensors_for_checkpoint(
        state_dict, prefix,
        {"weight": 0, "bias": 0},  # weight/bias沿axis 0切分
        sharded_offsets,
        tp_group=self._tp_group,
        dp_cp_group=metadata["dp_cp_group"],
    )

# TERowParallelLinear (行切分, 沿axis 1)
def sharded_state_dict(self, prefix, ...):
    return make_sharded_tensors_for_checkpoint(
        state_dict, prefix,
        {"weight": 1},   # weight沿axis 1切分; bias不切分
        ...
    )
```

## 5. TEDotProductAttention：全特性注意力适配 (L1359-2445)

### 5.1 初始化核心流程

```python
# 步骤1: 构建ProcessGroupCollection
pg_collection = ProcessGroupCollection(
    tp=get_tensor_model_parallel_group(),
    cp=get_context_parallel_group(),
    hcp=get_hierarchical_context_parallel_groups(),  # 仅a2a+p2p模式
)

# 步骤2: CP配置注入
if config.context_parallel_size > 1:
    # 共享cp_stream (类变量，所有层共用)
    if TEDotProductAttention.cp_stream is None:
        TEDotProductAttention.cp_stream = Stream()  # FlagScale Add
    
    extra_kwargs["cp_group"] = pg_collection.cp
    extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
    
    # cp_comm_type决定Ring Attention通信模式
    if cp_comm_type == "a2a+p2p":
        # 层次化: 节点内A2A + 节点间P2P
        extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups()
        extra_kwargs["cp_comm_type"] = "a2a+p2p"
    else:
        extra_kwargs["cp_comm_type"] = cp_comm_type  # "p2p" or "a2a"

# 步骤3: 版本条件特性
extra_kwargs["num_gqa_groups"] = config.num_query_groups    # GQA (TE >= 0.11)
extra_kwargs["window_size"] = config.window_size             # SWA (TE >= 1.2)
extra_kwargs["softmax_scale"] = softmax_scale                # 自定义scale (TE >= 1.10)
extra_kwargs["softmax_type"] = config.softmax_type           # (TE >= 2.8)
extra_kwargs["return_max_logit"] = True                      # qk-clip (TE >= 2.9)
```

### 5.2 cp_stream共享设计

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    cp_stream: Stream = None  # 类变量：所有层共用同一个CP通信stream
```

**设计意图**：Ring Attention的P2P通信需要与计算重叠，所有层使用同一stream确保：
1. 通信按序执行（防止乱序）
2. 不需要每层创建新stream（节省资源）
3. 类变量保证全局唯一

### 5.3 PackedSeqParams版本兼容 (L1509-1529)

```python
self.kept_packed_seq_params = set(
    field.name for field in dataclasses.fields(PackedSeqParams)
)
# 根据TE版本裁剪不支持的参数
if get_te_version() < PkgVersion("1.3.0"):
    self.kept_packed_seq_params.discard("max_seqlen_q")
    self.kept_packed_seq_params.discard("max_seqlen_kv")
if get_te_version() < PkgVersion("1.10.0"):
    self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
    self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")
# 非注意力字段过滤
self.kept_packed_seq_params.discard("total_tokens")  # Mamba专用
self.kept_packed_seq_params.discard("seq_idx")
```

## 6. 量化配置系统 (L86-307)

### 6.1 三层配置结构

```
TransformerConfig
    └── quantization_config: QuantizationConfig (per-layer override)
            └── config dict: {
                    "_config_type": "TEQuantizationParams",
                    "training_recipe": {
                        "fp8_quantization_recipe": "current_scaling",
                        "fp8_format": "e4m3",
                        "override_quantized_autocast": true,
                        "tp_only_amax_red": false
                    },
                    "evaluation_recipe": null
                }
```

### 6.2 TEQuantizationRecipe (L93-160)

```python
@dataclasses.dataclass
class TEQuantizationRecipe:
    fp8_quantization_recipe: Optional[Fp8Recipe] = None
    fp4_quantization_recipe: Optional[Fp4Recipe] = None
    custom_recipe_factory: Optional[str] = None
    fp8_format: str = "e4m3"            # e4m3 or e5m2
    override_quantized_autocast: bool = True   # 是否覆盖全局FP8 context
    override_nonquantized_autocast: bool = False  # 是否在非FP8 context启用FP8
    tp_only_amax_red: bool = False       # amax reduction只在TP组内(节省通信)
```

**约束**：
- fp8 和 fp4 互斥（不能同时设置）
- fp8_quantization_recipe不能是`delayed`（delayed只能全局配置）
- custom需提供custom_recipe_factory路径

### 6.3 运行时量化决策 (_get_should_context_be_quantized_params, L285-305)

```python
def _get_should_context_be_quantized_params(
    qparams: TEQuantizationParams | None,
    training: bool,
    is_context_quantized: bool,
) -> bool:
    if qparams is None:
        return is_context_quantized  # 无per-layer override，遵循全局context
    
    recipe = qparams.training_recipe if training else (
        qparams.evaluation_recipe or qparams.training_recipe)
    
    if is_context_quantized:
        return recipe.override_quantized_autocast  # 全局开FP8时，是否覆盖
    else:
        return recipe.override_nonquantized_autocast  # 全局关FP8时，是否启用
```

**典型用例**：
```python
# 场景1: 只让某些层用FP8，其余BF16
# 全局不开fp8_autocast，对特定层设置 override_nonquantized_autocast=True

# 场景2: 全局FP8，但某层回退到BF16
# 全局开fp8_autocast，对特定层设置 override_quantized_autocast=False
```

### 6.4 forward中的量化context注入 (L876-892)

```python
def forward(self, x):
    quant_context = _get_fp8_autocast_for_quant_params(
        self.te_quant_params, self.training)
    
    with quant_context:   # 可能是nullcontext(BF16)或fp8_autocast(FP8)
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
    
    return out, None  # 始终返回(output, bias)二元组
```

## 7. TEDelayedScaling：FP8 Recipe Megatron适配 (L2447-2480)

### 7.1 参数映射

```python
class TEDelayedScaling(te.common.recipe.DelayedScaling):
    def __init__(self, config, fp8_format, override_linear_precision=(False,False,False)):
        extra_kwargs = {}
        
        # FP8 Attention支持 (TE >= 1.6.0)
        if is_te_min_version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        
        # fp8_interval在TE 1.8.0后废弃(现在每步都更新)
        if get_te_version() < PkgVersion("1.8.0"):
            extra_kwargs["interval"] = config.fp8_interval
        
        super().__init__(
            margin=config.fp8_margin,          # 防溢出余量(bits)
            fp8_format=fp8_format,             # HYBRID/E4M3/E5M2
            amax_compute_algo=config.fp8_amax_compute_algo,  # "max" or "most_recent"
            amax_history_len=config.fp8_amax_history_len,    # history窗口长度
            override_linear_precision=override_linear_precision,  # (fwd,bwd_dgrad,bwd_wgrad)
        )
```

### 7.2 在Megatron中的使用

```python
# training.py中的典型调用
if config.fp8 == "hybrid":
    fp8_recipe = TEDelayedScaling(
        config,
        fp8_format=te.common.recipe.Format.HYBRID,  # fwd E4M3, bwd E5M2
        override_linear_precision=(False, False, True)  # wgrad用BF16
    )
elif config.fp8 == "e4m3":
    fp8_recipe = TEDelayedScaling(
        config,
        fp8_format=te.common.recipe.Format.E4M3,
    )
```

## 8. TECudaRNGStatesTracker：RNG兼容适配 (L2483-2515)

### 8.1 Megatron与TE的RNG差异

| 特性 | Megatron原生 | TE CudaRNGStatesTracker | TECudaRNGStatesTracker(适配层) |
|------|------------|------------------------|-------------------------------|
| 状态管理 | 全局dict | generator对象 | 继承TE，加is_initialized() |
| CUDA Graph | 不支持 | register_generator_state | 继承TE支持 |
| is_initialized() | 无 | 无 | 新增，Megatron接口 |
| reset() | 有 | 有 | 调用super+重置_is_initialized |

### 8.2 is_initialized状态追踪

```python
def reset(self):
    super().reset()
    self._is_initialized = False

def set_states(self, states):
    super().set_states(states)
    self._is_initialized = True  # 设置状态后标记为已初始化

def add(self, name, seed):
    super().add(name, seed)
    self._is_initialized = True  # 添加tracker后标记为已初始化
```

**用途**：Megatron代码中需要检查`get_cuda_rng_tracker().is_initialized()`决定是否传给TE层：
```python
get_rng_state_tracker=(
    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
)
```

## 9. te_checkpoint：跨版本激活检查点 (L2518-2542)

```python
def te_checkpoint(
    forward_func,
    distribute_saved_activations,  # 激活是否分散到TP各rank
    get_rng_state_tracker,         # RNG状态tracker
    tp_group,                      # TP group
    *args, **kwargs
):
    from transformer_engine.pytorch.distributed import checkpoint
    
    if is_te_min_version("1.5.0"):
        # 新接口: 用keyword args传递
        return checkpoint(
            forward_func, *args,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            **kwargs,
        )
    else:
        # 旧接口: 位置参数
        return checkpoint(
            forward_func, distribute_saved_activations,
            get_rng_state_tracker, tp_group, *args
        )
```

**关键参数**：
- `distribute_saved_activations`: True时激活沿TP维度分散保存，节省tp_size倍内存
- TE的checkpoint比PyTorch原生多了FP8 amax状态保存/恢复（见Chapter 08 §9）

## 10. get_cpu_offload_context：跨版本CPU Offload封装 (L2558-2637)

### 10.1 版本分支逻辑

```python
def get_cpu_offload_context(enabled, num_layers, model_layers,
                             activation_offloading, weight_offloading,
                             double_buffering, retain_pinned_cpu_buffers):
    
    if is_te_min_version("2.5.0"):
        # FlagScale额外兼容检查: 防止自定义TE版本号声明>=2.5但参数不同
        _sig = inspect.signature(_get_cpu_offload_context)
        if "retain_pinned_cpu_buffers" in _sig.parameters:
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, model_layers,
                activation_offloading, weight_offloading,
                double_buffering,
                retain_pinned_cpu_buffers=retain_pinned_cpu_buffers,  # V2接口
            )
        else:
            # 降级: 不传retain_pinned_cpu_buffers
            context, sync_func = _get_cpu_offload_context(...)
    
    elif is_te_min_version("1.10.0.dev0"):
        # 中间版本: 有activation_offloading/weight_offloading, 无retain
        context, sync_func = _get_cpu_offload_context(
            enabled, num_layers, model_layers,
            activation_offloading, weight_offloading)
    
    else:
        # 旧版本
        context, sync_func = _get_cpu_offload_context(
            enabled, num_layers, model_layers)
    
    return context, sync_func
```

**FlagScale特殊处理**：通过`inspect.signature`动态检测实际参数，解决自定义版本号与实际API不匹配问题。

## 11. _get_extra_te_kwargs：通用参数注入 (L307-317)

```python
def _get_extra_te_kwargs(config: TransformerConfig):
    extra_kwargs = {"params_dtype": config.params_dtype}  # BF16/FP32
    
    if is_te_min_version("0.12.0"):
        if config.use_cpu_initialization:
            extra_kwargs["device"] = "cpu"     # CPU初始化
        elif config.init_model_with_meta_device:
            extra_kwargs["device"] = "meta"    # Meta device延迟初始化
        else:
            extra_kwargs["device"] = cur_platform.current_device()  # FlagScale: 跨硬件
    
    return extra_kwargs
```

`cur_platform.current_device()`是FlagScale扩展点，支持非CUDA硬件（如NPU）。

## 12. FlagScale特定扩展

### 12.1 tp_size动态获取 (L1037, L1547)

```python
# FlagScale修改: 支持FlexDistributed (get_parallel_context() != None时)
tp_size=self.config.tensor_model_parallel_size
    if get_parallel_context() is None
    else get_tensor_model_parallel_world_size()
```

FlagScale引入`ParallelContext`抽象层，支持动态切换并行配置，此处在ParallelContext存在时通过运行时查询获取tp_size。

### 12.2 cp_stream FlagScale Add (L1368, L1452)

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    cp_stream: cur_platform.Stream = None  # FlagScale Add: 使用平台抽象Stream

    if getattr(TEDotProductAttention, "cp_stream") is None:
        TEDotProductAttention.cp_stream = cur_platform.Stream()  # FlagScale Add
```

使用`cur_platform.Stream()`而非`torch.cuda.Stream()`，支持非CUDA硬件。

### 12.3 device抽象 (L316)

```python
extra_kwargs["device"] = cur_platform.current_device()  # FlagScale Add
```

## 13. 端到端配置传递链路

### 13.1 FP8训练完整链路

```
megatron_config.yaml:
  fp8: "hybrid"
  fp8_amax_history_len: 1024
  fp8_amax_compute_algo: "max"
  tp_comm_overlap: true
  sequence_parallel: true
  
  ↓ TransformerConfig解析

TransformerConfig:
  fp8 = "hybrid"
  tensor_model_parallel_size = 4
  
  ↓ 模型构建

TELayerNormColumnParallelLinear.__init__():
  extra_kwargs["ub_overlap_ag"] = True      # from tp_comm_overlap
  extra_kwargs["ub_name"] = "qkv"           # buffer名
  → super(te.pytorch.LayerNormLinear).__init__(
      sequence_parallel=True,
      tp_size=4, tp_group=tp_group,
      ...)

  ↓ forward时

TELayerNormColumnParallelLinear.forward():
  quant_context = fp8_autocast(recipe=recipe)
  with quant_context:
      super().forward(x, is_first_microbatch=True)
      → UserbuffersForwardLinear (AllGather + FP8 GEMM overlap)
```

### 13.2 模块替换关系

```
Megatron原生            TE替换                       核心增益
ColumnParallelLinear → TEColumnParallelLinear    FP8/UB/wgrad-delay
RowParallelLinear    → TERowParallelLinear       FP8/UB
LayerNorm+Column     → TELayerNormColumnParallel 融合kernel
DotProductAttention  → TEDotProductAttention     FlashAttn/CP/FP8-attn
LayerNorm            → TENorm                   RMSNorm支持
checkpoint()         → te_checkpoint             FP8-safe recompute
RNGTracker           → TECudaRNGStatesTracker    CUDA Graph兼容
```

## 14. 设计决策与取舍

| 决策 | 方案 | 理由 |
|------|------|------|
| is_first_microbatch | 模块内状态标志 | 避免外部传参复杂度 |
| Expert TP禁用 | 尺寸预切分+te_parallel_mode=None | token_dispatcher已处理通信 |
| per-layer量化配置 | TEQuantizationParams + forward注入context | 细粒度控制，无需改模型结构 |
| cp_stream类变量 | 共享单stream | 确保P2P通信顺序+节省资源 |
| FlagScale platform抽象 | cur_platform.* 替换 torch.cuda.* | 跨NPU/GPU硬件 |
| 版本条件特性 | is_te_min_version() 守卫 | 向下兼容，不强制TE升级 |
| split_fused_layer工具 | 独立函数，非in-place | 便于checkpoint转换/模型变换 |
| inspect.signature检查 | 运行时参数检测 | 应对自定义版本号与API不匹配 |

## 15. 调试与验证

```python
# 检查模块替换是否生效
for name, module in model.named_modules():
    if "linear" in name.lower():
        print(f"{name}: {type(module).__name__}")
        # 期望: TEColumnParallelLinear / TERowParallelLinear

# 验证UB overlap配置
for name, module in model.named_modules():
    if isinstance(module, TELinear):
        print(f"{name}: ub_overlap_ag={getattr(module, 'ub_overlap_ag', False)}")

# 检查FP8量化状态
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
print(f"FP8 enabled: {FP8GlobalStateManager.is_fp8_enabled()}")
print(f"FP8 recipe: {FP8GlobalStateManager.get_fp8_recipe()}")

# 验证CP stream共享
print(f"All layers share same cp_stream: "
      f"{TEDotProductAttention.cp_stream is not None}")
```
