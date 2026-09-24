# 第14章：Transformer 层抽象与 ModuleSpec 系统 深度源码分析

## 1. 概述与设计动机

### 1.1 核心问题

大模型 Transformer 层需要同时支持多种变体：
- **后端差异**: 本地 PyTorch 实现 vs TransformerEngine (FP8) vs Kitchen
- **架构差异**: 标准 Attention vs MLA, Dense MLP vs MoE
- **并行差异**: TP 切分模式不同、CP 通信策略不同
- **优化差异**: CUDA Graph scope (attn/mlp/moe_router)

传统继承树在组合爆炸时不可维护。Megatron 采用**声明式 Spec 系统**解决此问题。

### 1.2 WHY: 为什么不用简单的 if-else？

```python
# 反面示例 — 不可维护:
if use_te and use_mla and use_moe:
    layer = TEMLAMoELayer(...)
elif use_te and use_mla:
    layer = TEMLALayer(...)
# ... 2^N 个分支
```

Spec 系统的解法：**将"组件声明"与"组件实现"分离**，用数据驱动替代逻辑分支。

## 2. 源码定位

| 文件 | 路径 | 行数 | 核心内容 |
|------|------|------|----------|
| spec_utils.py | `megatron/core/transformer/spec_utils.py` | 142 | ModuleSpec + build_module |
| transformer_layer.py | `megatron/core/transformer/transformer_layer.py` | 2160 | TransformerLayer 类族 |
| transformer_block.py | `megatron/core/transformer/transformer_block.py` | 1152 | 层堆叠 + PP 分配 |
| gpt_layer_specs.py | `megatron/core/models/gpt/gpt_layer_specs.py` | 824 | GPT Spec 定义 |

## 3. ModuleSpec 数据结构 (spec_utils.py L1-142)

### 3.1 定义

```python
# spec_utils.py L23-30
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]   # 模块类 或 (module_path, class_name) 元组
    params: dict = field(default_factory=dict)  # 构造参数
    submodules: object = None    # 嵌套的 Submodules dataclass
    metainfo: dict = field(default_factory=dict)  # 元信息
```

### 3.2 build_module 工厂 (spec_utils.py L55-90)

```python
def build_module(spec_or_module, *args, **kwargs):
    """从 ModuleSpec 实例化模块"""
    # Case 1: 函数类型 → 直接返回（如 bias_dropout_add）
    if isinstance(spec_or_module, FunctionType):
        return spec_or_module
    
    # Case 2: 获取模块类（支持 Tuple 延迟导入）
    module = get_module(spec_or_module)
    
    # Case 3: 注入 submodules
    if spec_or_module.submodules is not None:
        kwargs["submodules"] = spec_or_module.submodules
    
    # Case 4: 合并参数并实例化
    return module(*args, **spec_or_module.params, **kwargs)
```

**WHY Tuple 延迟导入？**
- 避免 TransformerEngine 与 Megatron 之间的循环依赖
- 仅在实际构建时才 `importlib.import_module(path)`
- 允许条件性安装（无 TE 时也不报错）

### 3.3 递归构建模式

```
build_module(layer_spec)
  ├── layer_spec.submodules = TransformerLayerSubmodules(...)
  │     ├── self_attention = ModuleSpec(SelfAttention, submodules=...)
  │     │     └── submodules.linear_qkv = ModuleSpec(TEColumnParallelLinear)
  │     ├── mlp = ModuleSpec(MoELayer, submodules=...)
  │     └── input_layernorm = RMSNorm (callable, 非 ModuleSpec)
  └── TransformerLayer.__init__(config, submodules)
        ├── build_module(submodules.self_attention) → SelfAttention
        ├── build_module(submodules.mlp) → MoELayer
        └── submodules.input_layernorm(config) → RMSNorm instance
```

## 4. TransformerLayerSubmodules 声明 (L279-325)

### 4.1 完整字段

```python
# transformer_layer.py L279-325
@dataclass
class TransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityOp
    
    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityOp
    
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityOp
```

**设计要点**:
- 默认 `IdentityOp`：不声明的组件自动透传，无需 `None` 检查
- 支持 `ModuleSpec` 或裸类型：简单组件直接传类即可
- 9 个插槽覆盖 Pre-LN / Post-LN / 交叉注意力 所有变体

### 4.2 GPT Spec 实例 (gpt_layer_specs.py L356-420)

```python
# gpt_layer_specs.py L356+
def get_gpt_layer_with_transformer_engine_spec(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False, ...
) -> ModuleSpec:
    """构建 TransformerEngine 后端的 GPT 层 Spec"""
    
    mlp = _get_mlp_module_spec(use_te=True, num_experts=num_experts, ...)
    
    if multi_latent_attention:
        # DeepSeek MLA 模式
        attention_submodules = MLASelfAttentionSubmodules(
            linear_q_proj=TEColumnParallelLinear,
            linear_q_down_proj=TEColumnParallelLinear,
            linear_kv_down_proj=TEColumnParallelLinear,
            ...)
    else:
        attention_submodules = SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=TERowParallelLinear, ...)
    
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(module=SelfAttention, submodules=attention_submodules),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
```

## 5. TransformerLayer 前向传播 (L610-850)

### 5.1 _forward_attention (L610-780)

```
_forward_attention 数据流:
────────────────────────────────────────
hidden_states [s, b, h]
    │
    ├─ input_layernorm → input_layernorm_output
    │   └─ (如果 recompute_input_layernorm: CheckpointWithoutOutput)
    │   └─ (如果 fine_grained_offloading: CPU 卸载)
    │
    ├─ residual = hidden_states (FP32 if fp32_residual_connection)
    │
    ├─ self_attention(input_layernorm_output, mask, rotary_pos_emb, ...)
    │   └─ 返回 (attention_output, bias)
    │
    ├─ self_attn_bda(attention_output, bias, residual, dropout_rate)
    │   └─ hidden_states = dropout(attention_output + bias) + residual
    │
    └─ 返回 hidden_states [s, b, h]
────────────────────────────────────────
```

### 5.2 _forward_mlp (L780-850)

```
_forward_mlp 数据流:
────────────────────────────────────────
hidden_states [s, b, h]
    │
    ├─ pre_mlp_layernorm → mlp_layernorm_output
    │   └─ (同样支持 recompute / offload)
    │
    ├─ residual = hidden_states
    │
    ├─ mlp(mlp_layernorm_output)
    │   ├─ Dense: linear_fc1 → activation → linear_fc2
    │   └─ MoE: router → expert dispatch → combine
    │
    ├─ mlp_bda(mlp_output, bias, residual, dropout_rate)
    │   └─ hidden_states = dropout(mlp_output + bias) + residual
    │
    └─ 返回 hidden_states [s, b, h]
────────────────────────────────────────
```

### 5.3 WHY: 为什么 BiasDropoutAdd 是独立组件？

三个原因：
1. **融合优化**: TE 提供 fused bias_dropout_add CUDA kernel（单次内存访问）
2. **精度控制**: 残差加法在 FP32 执行（即使主计算是 BF16/FP8）
3. **可替换**: 不同后端的 BDA 实现不同

```python
# 本地 PyTorch 版本:
def bias_dropout_add_unfused(x, bias, residual, prob):
    return torch.dropout(x + bias, p=prob) + residual

# TE 融合版本:
# 单个 CUDA kernel 完成 bias + dropout + residual add
```

## 6. TransformerBlock 层堆叠 (transformer_block.py L1-500)

### 6.1 层分配给 PP stages

```python
# transformer_block.py L200-280
def _build_layers(self):
    # 计算当前 PP rank 负责哪些层
    offset = get_transformer_layer_offset(self.config)
    num_layers_per_pipeline_rank = self.config.num_layers // pp_size
    
    # Virtual Pipeline (VP) 多 chunk:
    if self.config.virtual_pipeline_model_parallel_size:
        # 每个 VP chunk 负责 num_layers / (pp * vp) 层
        chunk_offset = vp_rank * num_layers_per_chunk
    
    self.layers = torch.nn.ModuleList()
    for i in range(num_layers_per_pipeline_rank):
        layer_spec = self._get_layer_spec(i + offset)
        layer = build_module(layer_spec, config=self.config, layer_number=i+1)
        self.layers.append(layer)
```

### 6.2 前向循环

```python
# transformer_block.py forward():
def forward(self, hidden_states, attention_mask, ...):
    for layer in self.layers:
        hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb, ...)
    
    # 最后一层后的 LayerNorm (Pre-LN 架构需要)
    if self.post_process and self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)
    
    return hidden_states
```

## 7. 层变体继承体系 (L1419-2160)

### 7.1 类层次

```
BaseTransformerLayer (ABC, L325)
    │
    └── TransformerLayer (L341)                 标准 Dense 层
            │
            ├── HyperConnectionTransformerLayer (L1419)  超连接层
            │     └─ 多残差路径（α权重混合）
            │
            └── MoETransformerLayer (L1914)      MoE 层
                  └─ 重写 _forward_mlp 支持 expert 路由
                  └─ CUDA Graph 支持 moe_router (v0.18.2: CudaGraphModule, 见第18章)
```

### 7.2 MoETransformerLayer 扩展 (L1914+)

```python
class MoETransformerLayer(TransformerLayer):
    """支持 CUDA Graph 的 MoE 层"""
    
    def _forward_mlp(self, ...):
        # 如果 CUDA Graph scope 包含 moe_router:
        #   先 graph-capture router 前向
        #   再执行 expert 计算（无法 graph capture 因为动态路由）
        if CudaGraphModule.moe_router in self.config.cuda_graph_modules:
            # v0.18.2: 原 CudaGraphScope.moe_router in self.config.cuda_graph_scope
            # 配置面重构: scope→modules + cuda_graph_impl + inference_cuda_graph_scope, 详见第18章
            router_output = self._cudagraph_router(...)
            mlp_output = self.mlp.forward_after_router(router_output, ...)
        else:
            mlp_output = self.mlp(hidden_states)
```

**WHY 单独 MoETransformerLayer？**
MoE 的动态路由（top-K dispatch）无法完全 CUDA Graph 化，
需要将 router (静态计算) 与 expert (动态计算) 分离。

## 8. Selective Recomputation (L486-555)

### 8.1 粒度控制

```python
# 通过 config.recompute_modules 精确选择重计算的子模块:
if self.config.recompute_granularity == 'selective':
    if "layernorm" in self.config.recompute_modules:
        self.recompute_input_layernorm = True   # 重计算 LN 节省内存
    if "mlp" in self.config.recompute_modules:
        self.recompute_mlp = True               # 重计算 MLP 节省更多
```

### 8.2 WHY: 为什么 selective 比 full 更好？

| 策略 | 内存节省 | 计算开销 | 适用场景 |
|------|----------|----------|----------|
| full recompute | ~60% | +33% 计算 | 极端内存受限 |
| selective (LN only) | ~5% | +1% 计算 | 一般训练 |
| selective (LN + MLP) | ~30% | +15% 计算 | 中等内存压力 |

LN 激活占比小但持久（整个 layer 生命周期），重计算代价极低。

## 9. Fine-Grained Activation Offloading (L556-565)

```python
# 按模块粒度卸载激活到 CPU:
self.offload_attn_norm = (
    self.config.fine_grained_activation_offloading
    and "attn_norm" in self.config.offload_modules
)
self.offload_mlp_norm = (
    self.config.fine_grained_activation_offloading
    and "mlp_norm" in self.config.offload_modules
)
```

**WHY fine-grained offloading？**
全量 offload 会 PCIe 带宽瓶颈。
按模块卸载（只卸 LN activations）PCIe 数据量小，
且与 GPU 计算 overlap，几乎零开销。

## 10. CUDA Graph 集成 (L575-594)

```python
def create_mcore_cudagraph_manager(self, config):
    """按捕获覆盖粒度注册 CUDA Graph (v0.18.2 API)"""
    # 训练捕获覆盖 = cuda_graph_modules (成员: attn/mlp/moe/moe_router/moe_preprocess/mamba)
    # 空列表 = 整层捕获; 实现选择 = cuda_graph_impl (none/local/transformer_engine/full_iteration)
    # 旧 API (cuda_graph_scope + CudaGraphScope) 已 deprecated, 详见第18章
    if not self.config.cuda_graph_modules:
        # 覆盖为空 → Graph 整个 layer
        self.cudagraph_manager = CudaGraphManager(config)
    elif CudaGraphModule.attn in self.config.cuda_graph_modules:
        # 只 Graph attention 部分
        self.cudagraph_manager = CudaGraphManager(config)
    elif CudaGraphModule.mlp in self.config.cuda_graph_modules:
        # 只 Graph MLP 部分（MoE 层用 moe/moe_router 覆盖，不走这里）
        assert not self.is_moe_layer
        self.cudagraph_manager = CudaGraphManager(config)
```

## 11. 设计决策对比

| 维度 | Megatron ModuleSpec | PyTorch Module 继承 | 选择理由 |
|------|--------------------|--------------------|----------|
| 可组合性 | 数据声明 + 工厂 | 类继承 | 避免组合爆炸 |
| 后端切换 | 换 Spec 即可 | 需新子类 | 解耦 |
| 运行时开销 | build_module 一次性 | 无 | 可忽略 |
| 调试可见性 | Spec 可序列化打印 | 需遍历 modules | 运维友好 |
| 动态修改 | 修改 Spec fields | 需 monkey-patch | 灵活 |

## 12. 与其他章节的关联

- **→ 第2章 Tensor Parallel**: `TEColumnParallelLinear` 在 Spec 中声明
- **→ 第5章 Expert Parallel**: `MoELayer` 作为 mlp slot 注入
- **→ 第6章 Mixed Precision**: TE 后端的 Spec 自动启用 FP8
- **→ 第15章 parallel_state**: `pg_collection` 传入 layer
- **→ 第16章 MLA**: `MLASelfAttentionSubmodules` 定义 MLA 特有的 linear 组合
- **→ TE-FL 第4章 Attention**: 实际注意力计算实现

## 13. 源码版本信息

- `transformer_layer.py`: 2160 行 (含 MoETransformerLayer, HyperConnectionTransformerLayer)
- `transformer_block.py`: 1152 行 (含 PP 层分配, 异构层模式)
- `spec_utils.py`: 142 行 (ModuleSpec + build_module)
- `gpt_layer_specs.py`: 824 行 (具体 Spec 声明)
- FlagScale 扩展: HyperConnection, DualPipeV, fine-grained offloading
