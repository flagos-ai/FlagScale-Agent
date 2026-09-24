# 第18章：CUDA Graph 配置面重构 (Megatron-LM v0.18.2)

> 依据：Megatron-LM-FL e15cb6928（v0.18.2, PR#109）实测源码 —
> `megatron/core/transformer/enums.py`、`cuda_graph_config.py`、`transformer_config.py`。

## 1. 重构概览

v0.18.2 将训练/推理 CUDA Graph 的配置拆分为三个正交字段：

| 关注点 | 旧 API (重构前) | 新 API (v0.18.2) |
|--------|----------------|------------------|
| 训练捕获覆盖 | `cuda_graph_scope` + `CudaGraphScope` | `cuda_graph_modules` + `CudaGraphModule` |
| 捕获实现 | `enable_cuda_graph` / `external_cuda_graph` (bool) | `cuda_graph_impl` (Literal) |
| 推理图所有权 | `CudaGraphScope.full_iteration_inference` | `inference_cuda_graph_scope` (InferenceCudaGraphScope) |

**CudaGraphScope 没有被删除**：它保留为 deprecated 独立类（`enums.py`），仅用于旧
checkpoint 反序列化。注意它的序数与 CudaGraphModule 不同（full_iteration=1, attn=2, ...），
因此不能做简单别名——别名会静默重建出错误身份的枚举成员。

## 2. 新枚举定义 (enums.py)

```python
class CudaGraphModule(enum.Enum):
    """per-layer CUDA graph 的命名捕获区域; 整层捕获用空 scope 表示"""
    attn = 1; mlp = 2; moe = 3; moe_router = 4; moe_preprocess = 5; mamba = 6

class InferenceCudaGraphScope(enum.Enum):
    """推理 CUDA graph 的所有权边界"""
    none = 1      # eager, 无图
    layer = 2     # 图归 module/layer 所有 (TransformerLayer / MambaLayer)
    block = 3     # 图归外层 block 所有 (TransformerBlock / HybridBlock)

class CudaGraphScope(enum.Enum):   # DEPRECATED: 仅 checkpoint 向后兼容
    full_iteration = 1; attn = 2; mlp = 3; moe = 4; moe_router = 5
    moe_preprocess = 6; mamba = 7; full_iteration_inference = 8
```

## 3. 配置字段 (TransformerConfig)

- `cuda_graph_impl: Literal['none','local','transformer_engine','full_iteration'] = "none"`
  - `local`: `make_graphed_callables()` 按 `cuda_graph_modules` 捕获 per-layer 图；推理不支持
  - `transformer_engine`: TE 层图捕获；推理不支持
  - `full_iteration`: 整迭代图（不含 optimizer）；此时 `cuda_graph_modules` 必须为空
- `cuda_graph_modules: Union[str, CudaGraphModule, List[...]] = "full"`
  - 成员: `"attn"`(_forward_attention) / `"mlp"`(dense _forward_mlp) / `"moe"`(MoE _forward_mlp,
    drop-and-pad) / `"moe_router"`(router 前向, 含未与 EP 通信 overlap 的共享专家) /
    `"moe_preprocess"`(MoELayer.preprocess, 必须与 moe_router 同用) / `"mamba"`
  - 空列表 = 捕获整层；`"full"` 为 deprecated 值，`__post_init__` 中转为空列表
- `inference_cuda_graph_scope: Optional[InferenceCudaGraphScope] = None`
  - 未设置时由 impl 推导: `local→layer`，其他→`none`
  - 合法组合: 仅 `local` 允许 `layer|block`，其余 impl 必须 `none`
- `cuda_graph_scope`: deprecated 字段，`__post_init__` 迁移到 `cuda_graph_modules`，将移除
- 相关: `cuda_graph_use_single_mempool`(full_iteration 下训练/eval/optimizer 共池)、
  `cuda_graph_retain_backward_graph`、`cuda_graph_warmup_steps=3`、
  `external_cuda_graph`(DEPRECATED→cuda_graph_impl)、`moe_pad_experts_for_cuda_graph_inference`

## 4. 归一化 / 迁移 API (cuda_graph_config.py)

```python
normalize_cuda_graph_modules(...)                        # modules 字段归一化
normalize_inference_cuda_graph_scope(...)                # 推理 scope 归一化
validate_deprecated_cuda_graph_modules_migration_inputs(...)
get_deprecated_cuda_graph_modules_migration(...)         # 旧值→新值迁移
ALLOWED_INFERENCE_SCOPES                                 # 合法推理 scope 集合
```
`TransformerConfig.__post_init__` 调用上述函数完成旧→新迁移；旧 checkpoint 中的
CudaGraphScope 实例先转字符串名再走归一化（`CUDA_GRAPH_MODULES_DEPRECATIONS`）。

## 5. 迁移对照表

| 旧写法 | v0.18.2 新写法 |
|--------|----------------|
| `cuda_graph_scope=[CudaGraphScope.full_iteration]` | `cuda_graph_impl="full_iteration"` |
| `cuda_graph_scope=[CudaGraphScope.full_iteration_inference]` | `inference_cuda_graph_scope=InferenceCudaGraphScope.block` |
| `cuda_graph_scope=[CudaGraphScope.attn]` | `cuda_graph_modules=["attn"]` |
| `enable_cuda_graph=True` | `cuda_graph_impl` 置为非 `"none"` |
| `external_cuda_graph=True` | DEPRECATED，由 `cuda_graph_impl` 取代 |

## 6. 运行时消费点

- `transformer_layer.py`: per-layer 图捕获（`_forward_attention`/`_forward_mlp` 分区）
- `transformer_block.py` (~L534): `inference_cuda_graph_scope == InferenceCudaGraphScope.block`
  时按 decode-only / `using_cuda_graph_this_step()` 决定是否走图
- `transformer_config.py` (`__post_init__` ~L2630+): moe_preprocess 必须伴随 moe_router；
  `full_iteration` impl 下 modules 必须为空
- `cuda_graphs.py`: `CudaGraphManager` 实现

## 7. 何时读本章

排查 CUDA Graph 配置报错（"must be empty" / scope 组合非法 / 旧 checkpoint 加载）、
把 FlagScale yaml 从旧 API 迁到 v0.18.2、或给新硬件平台适配层写图捕获支持时。
