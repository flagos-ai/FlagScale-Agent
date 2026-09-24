# Flash-Linear-Attention 源码分析：线性注意力的 chunk 并行与状态递推机制

> **知识组**: `know-linear-attention` | **文档**: `docs/linear_attention/01_fla_source_analysis.md`
> **源码**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) @ main（2026-09 快照，515 个 py 文件 / 147,451 行；版本号经 `pyproject.toml` 动态取 `fla.__version__`）
> **分析范围**: `fla/ops/common`（chunk 通用内核）+ `fla/ops/gated_delta_rule`（GDN，Qwen3-Next 采用）+ `fla/ops/gla` + `fla/ops/delta_rule` + `fla/ops/cp` + `fla/layers` + `fla/modules` + 基础设施（backends/utils）。其余 40+ 算子目录与通用链路同构，不展开。
> **行号说明**: 所有 `L<n>` 锚点均在分析时对源码实测（read_file/sed 抽查核对）；FLA 迭代快，引用时建议 `git grep` 复核。

---

## 1. 库定位与总体架构

### 1.1 FLA 是什么

FLA（flash-linear-attention）是线性注意力（Linear Attention）家族的 Triton 算子库，覆盖 GLA、DeltaNet、GatedDeltaNet（GDN）、RetNet、RWKV-6/7、KDA、HGRN-2、Titans、TTT、NSA 等几十种"递推状态 + 门外积"结构的注意力变体。与 softmax 注意力 `o_t = Σ softmax(q_t k_sᵀ) v_s` 不同，线性注意力的共性是一个 **一阶递推状态**：

```
S_t = S_{t-1} · diag(α_t) + k_t ⊗ (β_t v_t)      # [dk, dv]，α_t 是遗忘门，β_t 是写入强度
o_t = q_t S_t                                     # 读出
```

GLA 中 α 是逐维向量门（`logsigmoid`），DeltaNet 中 β 是标量写入强度、且状态更新带 delta 修正（`v_new = v − w·S`，即"用新 k 覆盖旧记忆"）。**线性性带来 O(T) 递推，但串行依赖阻止并行**——FLA 的全部核心工程就是把这条递推改造成"chunk 内矩阵化 + chunk 间并行传递"的分块算法。

### 1.2 分块化思想（为什么是 chunk=64）

把 T 个 token 切成 NT = T/BT 个 chunk（BT 默认 64，必须 2 的幂——`cumsum.py L267/311` 有断言）。递推在 chunk 边界处解耦：

```
S(chunk i+1) = S(chunk i) · A_gate(chunk i) + intra(chunk i) 的写入项
```

- **chunk 内**：把递推展开成三角线性方程组，`A = strictly_lower(β · K Kᵀ · gate)`，解 `(I+A)⁻¹` 得到 WY 表示（第 4 章）；
- **chunk 间**：状态 `h[i] ∈ [BT, K, V]` 是一条可以**并行计算读出、串行递推更新**的链（第 6 章）。

效果（README benchmark，GB200，B=1/T=8192/H=96/D=128 前向）：

| kernel | 时间 |
|---|---|
| `chunk_gla` (FLA) | 1.765 ms |
| flash_attn (softmax 注意力同规模) | 3.753 ms |

即分块后的 GLA 反而比 FlashAttention 快约 2.1×——大 T 下线性注意力的 FLOPs 优势（O(T·d²) vs O(T²·d)）在分块后得以兑现。

### 1.3 仓库布局与数据流总图

```
fla/
├── layers/            # 45 个 nn.Module 层（gla.py, gated_deltanet.py, rwkv7.py…）
├── models/            # 40 个 HF 集成模型
├── modules/           # 共享组件：l2norm, layernorm_gated, conv(短卷积), rotary, parallel…
└── ops/               # 45 个算子目录（核心）
    ├── common/        # 跨算子共享的 chunk 内核（chunk_delta_h/chunk_o/wy 链路）
    │   └── backends/  #   平台后端：intracard.py / tilelang/ / triton_ascend/
    ├── gated_delta_rule/  # GDN：chunk.py(管线) chunk_fwd.py(融合intra) wy_fast.py gate.py
    ├── gla/ delta_rule/   # GLA 与纯 DeltaNet（GDN 的无门控退化版）
    ├── cp/            # Context Parallelism（comm.py + context.py）
    ├── backends/__init__.py  # 通用 dispatch 注册机制（第 9 章）
    └── utils/         # index(块索引) cache(autotune持久化) op(exp2/safe_dot) cumsum solve_tril
```

一条前向数据的完整旅程（GDN 为例，第 7 章展开）：

```
x → [layer: q/k/v/g/beta 投影 + 短卷积 + l2norm]
  → gdn_gate_chunk_cumsum      # 门控日志 → chunk 内前缀和（gdn gate.py, scale=RCP_LN2）
  → chunk_gated_delta_rule_fwd_intra   # WY 表示: kkt → solve_tril → w,u（chunk_fwd.py 融合版）
  → chunk_fwd_h                # 状态递推：寄存器 tile，输出每 chunk 状态 h + final_state
  → chunk_fwd_o                # 读出: o = q·h + intra(A@v)，双 tl.dot 重用
  → 输出 o + recurrent_state（写回 layer cache）
```

反向是一条 9 步管线（第 7.2 节），核心是重算 `h`/`w,u`（省显存）+ 逆序 `dh` 递推 + 三角求逆的三明治梯度。


---

## 2. Layer 集成路径（`fla/layers`）

### 2.1 GatedDeltaNet（`layers/gated_deltanet.py`，368 行）

GDN 是 Qwen3-Next 采用的层结构，参数预算刻意对齐 Mamba2（L36-44 docstring）：

```
q_proj/k_proj: 0.75·H² 各；v_proj/g_proj/o_proj: 1.5·H² 各
总计 0.75×2 + 1.5×3 = 6·H²   （num_heads·head_dim = 0.75·H，需配平）
```

**投影与门控派生**（L145-152）：`q/k/v` 是常规投影；`a_proj`/`b_proj` 把 hidden 压到 `num_v_heads` 维——`a` 经 sigmoid 得遗忘门原料 `g`，`b` 经 sigmoid 得写入强度 `beta`。Delta 的"可学习负特征值"来自 L151-168：

```python
A = torch.empty(num_v_heads).uniform_(0, 16)
self.A_log = nn.Parameter(torch.log(A))         # 遗忘速率 log 参数
self.dt_bias = nn.Parameter(inv_softplus(dt))   # 保证 dt∈(0,1)
```

正向 `g = −exp(A_log) · softplus(g_raw + dt_bias)`（负值、单调），这就是 GDN 相对 GLA 的"数据相关可学习遗忘谱"。

**GVA（Gated Value Attention）分组**：L117-121 `num_v_heads` 可大于 `num_heads`——K/V 头按组共享（`HV // H` 组），内核里所有 K 侧索引都带 `i_h // (HV // H)`（见 `chunk_o.py L103-104`），输出侧 `dq/dk` 按 HV 展开后 `sum(3)` 归并（`chunk_o.py L786-788`）。

**训练/推理路径切换**（L236-243）：

```python
if torch.is_grad_enabled():   mode = 'chunk'
elif q_len <= 64 and not self.training: mode = 'fused_recurrent'
else:                          mode = self.mode
if self.training: assert mode == 'chunk'   # 训练只允许 chunk
```

即**训练必走 chunk 分块；短序列（T≤64）推理走单 kernel 递推**（`fused_recurrent.py`：fwd kernel L30 / wrapper L184 / autograd `FusedRecurrentFunction` L253（forward L257 / backward L301）/ 顶层入口 L309——寄存器内逐 token 递推，无 chunk 切分开销）。

**短卷积融合**（L202-217）：`_use_fused_qkv_conv` 判定无 cache、无 varlen、同 backend 时把 q/k/v 三个 causal depthwise conv 合成一个 kernel。

### 2.2 GLA（`layers/gla.py`，305 行）

与 GDN 的三点差异：

1. **门控低秩**（L156-159）：`gk_proj = Linear(H→16) → Linear(16→key_dim_per_group, bias=True)`——16 维低秩瓶颈生成逐维门，省参数；
2. **门归一化**（L240-242）：`gk = logsigmoid(gk) / gate_logit_normalizer`（默认 16），防大负值下溢；
3. **输出门**（L283-303）：`o = g_norm_swish_gate(o, g)` 或 `g_norm(o) * silu(g)`——GLA 的值门在**输出端**（GDN 的门在**递推写入端**），这是两者语义的核心区别。

同样的 T≤64 切换在 L198：`mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode`。

### 2.3 共享缓存协议

两层都用 `get_layer_cache` / `update_layer_cache`（gla.py L200/L270-279）读写 `recurrent_state`（跨 chunk/跨请求状态 `[N, HV, K, V]`，fp32）与 `conv_state`（短卷积尾部）——生成式推理 KV-cache 的线性注意力等价物：状态只有 K·V 而非 T·K，与上下文长度无关。

### 2.4 modules/ 共享组件（两个 layer 都依赖）

| 组件 | 作用 | 备注 |
|---|---|---|
| `modules/l2norm` | q/k 的 L2 归一化（DeltaNet 系必需，稳定 K·Kᵀ 谱） | 保存 `q_rstd/k_rstd` 供反向（§3.3） |
| `modules/conv` | causal depthwise 短卷积（window=4 常见） | 可被 §2.1 的 fused_qkv_conv 合并 |
| `modules/layernorm_gated` | GLA 输出门的门控归一化（g_norm / g_norm_swish_gate） | gla.py L283-303 调用 |
| `modules/rotary` | 部分 layer（如 NSA）的 RoPE | 线性注意力本身无位置编码需求 |
| `fla/ops/utils/cache.py` | autotune 配置持久缓存（§9.2 FlaCacheMode） | graph 模式下 zeros_like 分配语义见 §5 |

---

## 3. GDN 前向管线（`ops/gated_delta_rule/chunk.py`）

### 3.1 管线总览

`chunk_gated_delta_rule_fwd`（L33 起）五步：

```
① gdn_gate_chunk_cumsum (gate.py L163)  g_raw → g = −exp(A_log)·softplus(g_raw+dt_bias)
                          → chunk 内 cumsum × RCP_LN2（转 log2 域）
  → chunk_gated_delta_rule_fwd_intra (chunk_fwd.py L332，融合版)  kkt → solve_tril → recompute_w_u → (w, u, A)
③ [CP] pre_process / compress_h0     多卡上下文并行钩子（第 10 章）
④ chunk_fwd_h (chunk_delta_h.py L688)   状态递推 → (h, v_new, final_state)
⑤ chunk_fwd_o (chunk_o.py L64 wrapper)  o = q·h·exp2(g) + (A@u)·exp2(g)·scale
```

`RCP_LN2 = 1.4426950216`（`ops/utils/constant.py L10`）——全库统一把自然对数门控转成 **log2 域**，之后所有指数走 `exp2`（`op.py L16-37`：`FLA_USE_FAST_OPS=1` 时走 libdevice fast 硬件路径），一次乘法换掉 `exp`。

### 3.2 反向管线（L126-251）

```
recompute_w_u_fwd        重算 w,u（省显存，只存 A）
expand_h0 [CP]           展开初始状态
chunk_fwd_h (重算)       反向需要全部 chunk 状态
chunk_bwd_dv_local       dv（delta 专用局部版，复用 A 免重算 kkt）
[CP] pre_process
chunk_bwd_dhu            逆序 dh 递推 → (dh, dh0, dv2)
chunk_bwd_dqkwg          dq, dk, dw, dg
prepare_wy_repr_bwd      dk2, dv, db, dg2 —— 三角逆的三明治梯度
dk += dk2; dg += dg2; dg revcumsum   段内反向前缀和
gdn_gate_bwd             dA_log, ddt_bias（门参数梯度）
```

### 3.3 autograd 依赖图

`ChunkGatedDeltaRuleFunction`（L253 起）保存 15 个张量：

```
q, q_rstd, k, k_rstd, v, g, beta_raw, beta, A, initial_state,
cu_seqlens, chunk_indices, g_input, A_log, dt_bias
```

取向与 GLA（`gla/chunk.py L1395-1400`）不同：GLA 只存 8 个、按"g 是否 fp32"二选一（丢 g_cumsum 反向重算，或丢 g 重算 cumsum）省显存；GDN 因为反向要**精确重放**门控派生（A_log/dt_bias 梯度链），把门控中间量全保存，把**状态 h / WY 的 w,u** 留给重算——大状态量（NT×K×V×fp32）不进 ctx。

入口 L395-396 `@torch.compiler.disable @dispatch('gated_delta_rule')`：前者防 torch.compile 捕获运行时分发（同 `ops/backends/__init__.py L217` wrapper 策略），后者进后端注册表（第 9 章）。

---

## 4. WY 表示：chunk 内三角方程组的解（`ops/*/wy_fast.py` + `chunk_scaled_dot_kkt` + `solve_tril`）

线性注意力分块化的数学核心。chunk 内递推 `S_t = S_{t-1}·diag(α) + k_t ⊗ (β_t v_t)` 展开后是下三角系统，解出 "WY 表示"：

- `u = A·v`（写入方向的等价形式，形状 [NT, K, V]）
- `w = A·k·diag(β)`（读出修正项）
- `A = strictly_lower_tril(β·K·Kᵀ·gate) − diag` 的逆作用结果

WY 的意义：把 chunk 内 T 步串行递推压缩成 **3 次 GEMM + 1 次三角求逆**，全部可用 tensor core。

### 4.1 kkt：构造严格下三角系统（`common/chunk_scaled_dot_kkt.py`，kernel L33 / wrapper L84）

```python
b_A = tl.zeros([BT, BT], dtype=tl.float32)          # L63
b_A = tl.dot(b_k, tl.trans(b_k), b_A)               # L68 K·Kᵀ 累加进 b_A
b_A *= exp2(b_g_diff)                               # L74 门控衰减差（GDN/GLA 变体）
b_A *= b_b[:, None]                                 # L75 行乘 β（写入强度）
b_A = tl.where(m_A, b_A, 0)                         # L78 严格下三角置零
tl.store(p_A, ...)                                  # L80 存出 A（反向/复用原料）
```

### 4.2 solve_tril：BT×BT 下三角块求逆（`ops/utils/solve_tril.py`）

不调用任何库函数，纯 Triton 分治，三个 kernel：

- `solve_tril_16x16_kernel`（L38）：16×16 子块前向代换逐行求逆；
- `merge_16x16_to_32x32_inverse_kernel`（L108）与 `merge_16x16_to_64x64_inverse_kernel`（L198）：块合并公式 `Ai21 = −Ai22·A21·Ai11` 逐级放大；
- wrapper `solve_tril`（L355）按 BT 选择层级，BT=64 走两级合并。

复杂度 O(BT·logBT) 次小 GEMM。反向梯度用**三明治**：`dA⁻¹ = −A⁻¹·dA·A⁻¹`，即 `prepare_wy_repr_bwd` 里对 A 的两个 tl.dot。

### 4.3 recompute_w_u（`delta_rule/wy_fast.py`，kernel L38）

solve_tril 出的 A⁻¹ 直接在 kernel 内做两次 GEMM：

```python
b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False)   # L81 u = A⁻¹·v 侧
b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)   # L91 w = A⁻¹·k·β 侧
```

（b_vb 与 b_kb 分别是 value 侧与 key·β 侧的复合加载，u/w 各自一次 GEMM；`allow_tf32=False` 保精度。）GDN 版同构（`gated_delta_rule/wy_fast.py`），另含门控衰减差项。

### 4.4 融合版：kkt+solve_tril+recompute_w_u 一个 kernel（`gated_delta_rule/chunk_fwd.py`）

GDN 生产路径不走三步独立 kernel，而是 `chunk_gated_delta_rule_fwd_intra`（L40-328）：单个 kernel 内**寄存器持有 10 个 [16,16] 块**（b_Ai00..b_Ai33 + A21 等中间块），前代换 + 块合并在 SM 内完成，A 矩阵零 HBM 往返。代价是寄存器压力极大——

**Intel 后端直接放弃融合版**（L378-381 注释）：寄存器溢出，回退到三步 kernel（实测慢 2.3–3×）。

GDN 版 wy 反向（`gated_delta_rule/wy_fast.py`）在三明治基础上多一项：`dA = row_sum(dAᵢⱼ·A) − col_sum(dAᵢⱼ·A)` 的分离累计（L63-80），因严格下三角系统的梯度要同时往"行写入者"与"列依赖者"两方向流。

---

## 5. 门控 chunk cumsum（`ops/utils/cumsum.py`）

chunk 内门控前缀和 `g_cumsum[i] = Σ_{j≤i} g[j]`，是所有衰减 `exp2(g_i − g_j)` 的原料。kernel 按两个正交维度分四类（L33 local_scalar / L89 local_vector / L149 global_scalar / L207 global_vector）：

- **local vs global**：chunk 内前缀和（衰减差 `g_i − g_j` 的原料，绝大多数用法）vs 全序列前缀和（KDA 等需要跨 chunk 衰减的变体）；
- **scalar vs vector**：门是每头标量（GDN 系）还是逐维向量（GLA 系）。

**跨 chunk 进位**（L170 / L230）：kernel 输入 `b_g_last`（上一 chunk 的累计和）作为加性进位，使 NT 个 chunk 的 cumsum 完全并行——这是"chunk 间解耦"在门控上的体现。反向用 revcumsum（反向前缀和）+ `dg = dgcumsum − dgcumsum_shift`。

**CUDA graph 路径**（L259 参数 / L272-273）：`use_graph=True` 时分配用 `torch.zeros_like`（否则 `empty_like`）——L272 原文注释说明原因：*graph 模式下未覆盖行须为 0，kda_gate_bwd 对输出做全量归约，脏行会污染 dA/dbias*。即 graph 安全不靠特殊分配器，而靠"必须零初始化"的分配语义。

**与反向的对称性**：cumsum 的进位是"加性"（`b_g_last` 直接加到 chunk 首元素），反向的 revcumsum 把 chunk 边界梯度还原回逐 token 梯度——两方向的进位路径互为镜像，读任何一个 kernel 时都应对称地想到另一个。


---

## 6. 状态递推：`chunk_fwd_h` / `chunk_bwd_dhu`（`ops/common/chunk_delta_h.py`，全库心脏）

把 NT 个 chunk 的状态链 `h[i] = h[i−1]·D_i + W_i` 做成"读出并行、更新串行"的两个 kernel：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（L59）/ `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`（L370），autograd wrapper 分别在 L688 / L751。

### 6.1 fwd kernel（L59-349）：寄存器内 tile 递推

每 SM 持 **4 个 [BV=64, BK] 寄存器 tile**（b_h1..b_h4，对应 K≤256 的 4 个 64 宽 key 子块），逐 chunk：

```
① w 段读出：b_v = tl.dot(b_w, b_h)（STATE_V_FIRST 时带 trans）  L226-244
② v_new 残差：b_v = tl.load(p_v) - b_v（delta 修正：擦除旧记忆） L248
③ SAVE_NEW_VALUE 时把 b_v 存为 v_new                              L254-256
④ 门控：USE_G 时 b_v *= exp2(b_g_last - b_g)、b_h *= exp2(g_last) L258-270
⑤ 状态更新：b_h1 = tl.dot(b_k, b_v, b_h1)（或 += trans(dot)）    L293-316
   USE_GK 时 b_h *= exp2(b_gk_last) 逐维列门                      L272-291 段
```

关键点：**v_new 残差在状态更新里**（L248），使 `kᵀ@v_new` 成为"写入增量"而非"覆盖值"——这是 delta rule 与普通线性注意力（直接 `kᵀ@v`）在 kernel 层面的唯一区别。K>64/128/192 时 b_h2..4 依次加入（L295-316 的分支展开）。exp2 基-2 门控配合 RCP_LN2 预转换，全 kernel 无一次 `exp`。

### 6.2 bwd kernel（L370-684）：逆序 dh 递推

```
dh[NT] 初始化为 0（或 dh0 传入）
for i in reversed(range(NT)):
    读出侧梯度 dv += q·dh
    dh_prev = dh·exp2(g_last) + qᵀ@do（L420-500 主体）
    dk/w 梯度从 dh·v_newᵀ 得到
dh0 输出（初始状态梯度，供 CP/长上下文分段训练）
```

### 6.3 平台差异与 autotune（真实注释，L27-30）

```python
# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# Intel needs more warps than NVIDIA here: 8 warps is ~1.5x faster than the best
```

即：**Blackwell 因 tl.dot 递推竞态锁 2 warp**（Triton 3.8 前），**Intel 因 warp 调度特性要 8 warp**（快 1.5×）——同一 kernel 在两家平台上最优 warp 数相反。autotune 空间（L51-52）：`num_warps` 按 `GATED_DELTA_RULE_FWD_H_NUM_WARPS` 常量、`num_stages` 按 `check_shared_mem('ampere')` 分档；bwd（L361-363）固定 [2,4] warp × {2,3,4} stages（有 SMEM 才开）。

`h`/`v_new` buffer 在 graph 模式下由上游保证零初始化语义（同 §5 的 zeros_like 分配机制）。

---

## 7. 读出与反向：`chunk_o` 家族（`ops/common/chunk_o.py`）

### 7.1 fwd kernel（L64-161）

```
b_o = tl.dot(b_q, b_h[i])            # 读 chunk 外状态（跨 chunk 信息）
b_o = b_o * exp2(b_g)                # 门控衰减 L140-150
b_o += tl.dot(b_A_local, b_v)        # chunk 内 (A@v) 项
b_o *= scale                          # 输出缩放
store o（TMA）
```

**双 tl.dot 重用**：b_q 加载一次同时用于两个 dot（L128-137），省一次寄存器往返。门控在**读出前**乘（写入门已在 fwd_h 内），GVA 时 k 侧索引用 `i_h // (HV//H)`（L103-104）。

### 7.2 bwd_dqkwg（L180-350）

一个 kernel 同时出 dq/dk/dw/dg 四个梯度（共享 q/k/h 加载）。门控梯度特殊：**dg 只在 chunk 最后一个 token 产生完整值**（`dg_last`，L279 累加逻辑），其余 token 的 dg 由 chunk 边界值经 **revcumsum 反向传播**（L325 注释 `dg = dg_last + revcumsum(中间项)`）——与 cumsum 的进位对称。

### 7.3 dv_local（delta 专用）

`chunk_bwd_dv_local`：dv 梯度复用已保存的 A 矩阵直接 `dv = Aᵀ·do_local`，避免重算 kkt——GDN 反向管线里独立于全局 dv2 的一支。

### 7.4 GVA 梯度归并

HV > H 时 `dq/dk` 以 [B, T, HV, K] 算完后在头组维 `sum(dim=3)`（L786-788），符合"组内共享 K"的链式法则。

---

## 8. Context Parallelism（`ops/cp/`）

线性注意力的 CP 比注意力难：递推状态是全序列依赖，不能像 attention 一样按 KV 分片独立算。

### 8.1 通信层（`ops/cp/comm.py`，169 行）

- `all_gather_into_tensor`（L19）：把各 rank 的 KV/g/beta 段 all-gather 成全局序列（varlen 由上层 cu_seqlens 管理）；
- `send_recv_fwd`（L64）/ `send_recv_bwd`（L102）：流水线式 P2P——rank i 把自己 chunk 的状态算完立即发给 i+1、同时接收 i−1 的输入，形成"状态接力"；另有 conv 尾部专用的 `conv_cp_send_recv_fwd/bwd`（L142/L156）。

### 8.2 序列切分与负载均衡（`ops/cp/context.py`，240 行）

每个 rank 只需要**自己段前的所有 token 产生的状态**。核心类型与函数：

- `FLACPContext`（L23）：CP 元数据容器，`copy_for_backward`（L51）给反向做快照；
- `_interval_cp_meta`（L87）：段表元数据生成；`get_cp_cu_seqlens`（L129）：切分后的 cu_seqlens；
- `build_cp_context`（L210）：入口组装。两种切分模式：

- **contiguous**：连续切分，通信量 O((N−1)/N · T)，最坏最后一个 rank 空转；
- **zigzag**：交错切分（rank i 拿第 i、N−1−i 段），让"依赖前缀长度"在 rank 间对称，消除 straggler——与 Megatron CP 的 zigzag 思想同源。

h0 的压缩/展开在 `cp/chunk_delta_h.py`：`compress_h0`（L1142）/ `expand_h0`（L1155），进 kernel 前由 context 统一变换。

---

## 9. 基础设施三件套：dispatch / cache / 数学工具

### 9.1 后端 dispatch（`ops/backends/__init__.py`）

```python
class BaseBackend:              # L33：backend_type/priority/can_use() 能力判断
class BackendRegistry:          # L96：per-operation 注册表
    # _registries 全局字典 + _init_lock 懒加载；register() 按 priority 排序
    # _update_active_backend()：首个 can_use()==True 的后端设为 active
    # ensure_initialized()：首次使用时 import fla.ops.<op>.backends 触发注册
def dispatch(operation):        # L160：装饰器，运行时取 active backend 的实现
```

两个关键工程决策（L185 原文注释）：

> *"Avoid be.can_use(): its @cache wrapper breaks torch.compile tracing"* —— dispatch 每次调用都绕过 `can_use()` 的 `@cache`，直接查 registry，牺牲微小 CPU 开销换取 torch.compile 图可捕获性；

L217 入口统一 `@torch.compiler.disable`——双保险。算子侧用法：`@dispatch('gated_delta_rule')`，各后端（tilelang / triton_ascend / intracard）在各自 `backends/` 子包里注册同名实现，`can_use()` 做架构/TMA/驱动检查，priority 决定尝试顺序。

### 9.2 autotune 持久缓存（`ops/utils/cache.py`）

Triton autotune 每次 launch 要 benchmark 数十种配置，训练重启后重复耗时。FLA 的 `FlaCacheMode`（L27 起）六档（docstring L29-46 原文语义）：

| mode | 行为（docstring 原文） |
|---|---|
| DISABLED | 跳过全部缓存查找，总是重跑 Triton autotune（**env 未设置时的默认值**，L53） |
| STRICT | 仅精确 key 匹配；无匹配回退 autotune |
| FUZZY | exact → fuzzy 匹配；无匹配回退 autotune |
| FULL | exact → fuzzy → **default_config 兜底** |
| DEFAULT | 只用顶层 default_config 字段，跳过 key 查找 |
| ALWAYS | 同 DEFAULT 但每次 kernel 调用都重读配置文件（调试热重载） |

`CachedAutotuner`（L337）+ `load_cached_config`（L271）把 per-(kernel, backend) 的最佳 config 存盘，env `FLA_CACHE_MODE` 控制。生产建议：**训练任务用 STRICT**（防不同 shape 串配置），调参调试用 ALWAYS（改 JSON 即时生效），benchmark 用默认 DISABLED。

### 9.3 数学与发行版差异（`ops/utils/op.py`）

- `exp/exp2/log/log2/tanh`（L18-26）：`FLA_USE_FAST_OPS=1` 时绑定为 `tldevice.fast_*` 硬件快速路径，否则标准 `tl.*`；
- `safe_dot`（L40-67）：**Blackwell（SM100/SM120）专用**——用 `tl.inline_asm_elementwise`（`mov.f32` 恒等汇编）包住 `tl.dot` 的结果，**防止 TritonGPUHoistTMEMAlloc pass 错误地融合 add 与 dot**（fla#638 / triton#8695，上游修复前保留）；非 Blackwell 退化为普通 `tl.dot`；
- TMA 描述符兼容层（L81-90）：依次探测 `tl._experimental_make_tensor_descriptor` → `tl.make_tensor_descriptor` → 返回 None 的 fallback，抹平 Triton 3.2/3.3/3.4 的 API 变化。


---

## 10. 平台适配与已知坑（逐条带源码锚点）

| # | 平台/条件 | 坑 | 处置 | 锚点 |
|---|---|---|---|---|
| 1 | Hopper + Triton ≥3.4.0 且 <3.7.1 | gated `chunk_bwd_dqkwg` 产生错误结果（#640） | runtime `raise RuntimeError` 拦截，要求升级 Triton 或装 tilelang | `chunk_o.py L729-734`（原文注释） |
| 2 | Blackwell + fwd_h 状态递推 | tl.dot 递推竞态（Triton mainline 待修） | 锁 `num_warps=2` 直到 Triton 3.8 | `chunk_delta_h.py L27-28` |
| 3 | Blackwell + chunk_o | 8-warp（BK=BV=128）配置出问题 | Blackwell 禁用该档 | `chunk_o.py L33-34` |
| 4 | Blackwell + WY 反向 autotune | autotuner 选出不稳定配置（#913） | 锁定已验证配置：[2]warp × [4]stages | `gated_delta_rule/wy_fast.py L18-22` |
| 5 | Blackwell + `tl.dot` | HoistTMEMAlloc pass 错误融合 add+dot（#638/#8695） | `safe_dot` 内联汇编恒等包装 | `op.py L40-67` |
| 6 | Intel + fwd_h | warp 调度特性相反 | 要 8 warp（比 NVIDIA 最优快 ~1.5×） | `chunk_delta_h.py L30` |
| 7 | Intel + chunk_o | BK=BV=64 配对差一档 | 8 warp 最快 | `chunk_o.py L30` |
| 8 | Intel + recompute_w_u | 同向（要更多 warp） | 16 warp 快 ~1.3× | `gated_delta_rule/wy_fast.py L24-25` |
| 9 | Intel + 融合 intra | 10 块寄存器超 Intel 寄存器文件 | 弃融合版回退 unfused（`chunk_fwd.py L379` 注释） | `chunk_fwd.py L379` |
| 10 | 全平台 | `exp` 慢 | log2 域 + exp2（RCP_LN2 预转换） | `constant.py L10`、`op.py L18-26` |
| 11 | torch.compile | can_use 的 @cache 破图追踪 | dispatch 绕过缓存 + compiler.disable | `backends/__init__.py L185/L217` |
| 12 | Intel | Intel-xpu-backend Triton 多处已知 bug | 注释挂 intel-xpu-backend issue #3449 | `dplr/wy_fast_bwd.py L15`、`simple_gla/parallel.py L26` |

三条系统性教训：(1) **寄存器压力是 Triton 融合的第一约束**——融合版 10 个 [16,16] 块恰在 NVIDIA 寄存器文件内、Intel 溢出（chunk_fwd.py L379）；(2) **平台差异主要表现为最优 warp 数相反**（Intel 要更多 warp，Blackwell 反而锁 2）——跨平台调优不可共享 autotune 缓存；(3) 数学等价变换（exp→exp2、log 域）是跨平台可移植性的第一杠杆。

---

## 11. 与 FlashAttention / DeepEP 的机制对照

| 维度 | FlashAttention | FLA 线性注意力 | DeepEP（对照记忆） |
|---|---|---|---|
| 核心状态 | KV cache（随 T 增长） | 递推状态 S=[K,V] 固定 | 两级 buffer（NVL/RDMA） |
| 并行单元 | KV tile online softmax | chunk（BT=64）+ WY 三角解 | warp 角色分工 |
| 跨块传递 | online softmax m/l 重标定 | h 状态 + 门控 exp2 衰减 | notify_dispatch 元数据 |
| 数值技巧 | log2 域 softmax | 全库 log2 域 exp2 | FP8 量化 + LZ4 |
| 反向 | 重算 tile | 重算 h/w/u + 三明治梯度 | 重算 routing |
| 多卡 | TP/CP | CP zigzag 状态接力 | EP a2a |

深挖两点：

1. **online softmax vs WY**：FA 的 online 重标定（m 新旧最大值缩放）与 FLA 的门控衰减 `exp2(g_i − g_j)` 在"增量修正旧累计量"上是同构思想——只是 FA 修正的是 softmax 分母，FLA 修正的是线性状态。两者的反向都选择"存少量、重算大头"。
2. **KV cache vs recurrent_state**：softmax 注意力推理时每 token 新增 K·V（缓存 O(T)），线性注意力只维护 O(1) 的 S 矩阵——这是长上下文推理的根因优势，也是 FLA 把 cache 协议（§2.3）做成 layer 级一等公民的原因。

---

## 12. 调优顺序与工程建议

### 12.1 出问题时的排查优先级

```
① 精度异常 → 查门控域：g 必须 ≤0 且 log2 域一致（RCP_LN2 有没有漏乘）
② Blackwell NaN/竞态 → fwd_h 是否锁 2 warp；tl.dot 是否走 safe_dot；Triton 是否 ≥3.8
③ Hopper + Triton 3.4~3.7 → dqkwg 会直接 RuntimeError（#640），升级或装 tilelang
④ Intel 慢 → 部分属已知（融合版寄存器溢出回退），另注意 Intel 需要更多 warp（§10 #6-8）
⑤ 训练重启慢 → FLA_CACHE_MODE=STRICT 开 autotune 缓存
⑥ OOM → 反向 ctx 15 张量是否可减（GDN 减不了，GLA 可换 g dtype 触发重算路径）
```

### 12.2 chunk_size 选择

- BT=64（默认）是寄存器 tile（4×[64,BK]）与三角求逆开销的平衡点；
- 解码/短序列 T≤64 直接 fused_recurrent（无 chunk 开销）；
- 训练超长序列慎调 BT：solve_tril 的块合并层级按 64 硬编码了 16→64 两级。

### 12.3 生产配置建议

| 场景 | 配置 |
|---|---|
| 训练 | mode='chunk'（强制）、FLA_CACHE_MODE=STRICT、FLA_USE_FAST_OPS=1 |
| 长上下文训练 | 开 CP（zigzag）+ dh0 传递 |
| 推理 decode | fused_recurrent + CUDA graph（use_graph 零初始化语义，§5） |
| benchmark | FLA_CACHE_MODE=DISABLED（避免缓存噪声） |

### 12.4 与 FlagScale 的结合点

- **算子替换层**：Qwen3-Next 类模型的 GDN 层可直接映射 FLA layer（6·H² 预算与 Mamba2 对齐，Megatron 侧可复用 MoE 混合专家外的"线性注意力混合层"思路）；
- **CP 集成**：FLA 的 zigzag CP 与 Megatron CP 的 zigzag 分段同构，理论上可共享 seq 切分元数据；
- **Triton 版本管理**：FLA 对 Triton 3.2/3.3/3.4/3.8 的 TMA/Blackwell 适配路径（§9.3）与"按平台锁 warp/stages"的坑表（§10）是 FlagScale 做多版本依赖兼容时的参照样板。

---

## 附：本次分析的方法论备注

- 行号锚点全部在分析当时对 `/tmp/FLA` 快照实测；FLA 迭代快（README 自述 40+ 算子持续演进），复用本文时建议 `git grep -n "<关键串>"` 复核；
- 与既有知识组 `know-moe-training`（DeepEP 深度分析 03_deepep_source_analysis.md）共享"先抓数据流总图、再逐 kernel 锚点、最后平台坑表"的三段式结构。
