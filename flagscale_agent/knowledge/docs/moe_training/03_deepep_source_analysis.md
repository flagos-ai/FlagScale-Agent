# DeepEP 源码深度分析：MoE Expert Parallelism 通信引擎（V1 legacy 双模式 + V2 架构演进）

> 分析对象：DeepSeek 开源的 Expert Parallelism 通信库 DeepEP（GitHub main 分支 2026-09 tarball，全库 23390 行）。
> 深潜主线：`csrc/kernels/legacy/*`（V1 双模式 CUDA kernel）+ `deep_ep/buffers/legacy.py`（Python API）+ `csrc/legacy/*`（C++ 编排层）。V2（ElasticBuffer / NCCL Gin / 全 JIT）作为架构演进对比放在 §9——生产训练（DeepSeek-V3/R1、FlagScale/Megatron 集成）使用的是 V1 API（`Buffer + Config + EventOverlap`）。
> 所有行号均在本文写作时对照源码逐条核实；性能数字引自仓库 README 与 docs/legacy.md 实测表。

## 1. 概述与设计动机

### 1.1 解决什么问题

MoE 层前向要把每个 token 送到 top-k 路由命中的专家所在 GPU：EP 组内做两次集合通信——**dispatch**（token 出发）与 **combine**（专家输出按路由权重归并返回）。朴素实现（NCCL `all_to_all_single`）有三大结构性问题：

1. **变长消息与同步**：NCCL 集合通信把 all-to-all 当作黑盒集合操作，sizes 需在 host 侧准备，GPU 与 CPU 之间来回同步，kernel 无法自主处理"每目标变长"。
2. **额外复制**：数据经 staging 中转，多一次 HBM 读写；且无法与本地计算重叠（通信是黑盒）。
3. **延迟敏感**：decode/训练小 microbatch 下，每层 2 次 A2A 串行（V3 规模 61 层即每 microbatch 122 个串行通信窗口），每次 20-50 µs 的启动/同步开销都直接乘到端到端。

DeepEP 的解法：**GPU 端点对端通信内核 + 去中心化流控**。dispatch/combine 各是一个（或一对）kernel：SM 里的 warp 直接发起 NVLink P2P 写或 RDMA put（IBGDA），用"channel 化 ring buffer + head/tail 索引 + 系统级内存序"完成流控，CPU 完全不参与数据面。由此得到两种工作模式（`docs/legacy.md`）：

- **normal 模式（高吞吐）**：训练场景。NVLink 域内用 IPC 直接 P2P 搬运，跨机流量经"RDMA→NVLink 转发"只走一次 RDMA，带宽逼近链路上限——实测 8 rank NVLink 域 153 GB/s、64 EP 跨机 RDMA 51 GB/s（docs/legacy.md 性能表）。
- **low-latency 模式**：推理/训练 decode 场景。放弃 NVLink 中转、每对 rank 纯 RDMA 单跳（**节点内也走 RDMA**，`deep_ep/buffers/legacy.py:L561-563` 明确要求 IBGDA），128 token dispatch 77 µs（≈98 GB/s）、combine 114 µs（≈127 GB/s），并内置 FP8 in-kernel 量化（`internode_ll.cu:L219-244`）。

### 1.2 为什么是双模式（WHY）

两类工作负载的瓶颈不同，一套数据结构无法同时最优（详见 §11 对比表）：

| 约束 | 训练（prefill/大 batch） | 推理 decode/训练 LL |
|---|---|---|
| 关键指标 | 聚合带宽 GB/s | 单次 A2A 延迟 µs |
| token 数 | 数千~数万 | 几十~128 |
| 消息形状 | 大块连续（可 TMA/coalescing） | 小消息、逐 expert 定点投递 |
| 通信拓扑偏好 | NVLink 优先（转发换带宽） | RDMA 单跳（转发换延迟） |
| 与计算的关系 | 可与专家计算重叠（hook/双流） | 必须极低延迟，CUDA Graph 友好 |

（WHY 不用一套 kernel 自适应？）normal 模式的收益来自"NVLink 转发摊薄 RDMA 流量"，这要求 token 够多以填满 chunked pipeline；LL 模式的收益来自"去掉中转 warp、直接 per-expert 定点投递"，这要求接收端按 `[expert][rank][slot]` 静态预留空间——大 batch 下空间爆炸，两模式互斥（`docs/legacy.md` 明确两套 API 不混用）。

### 1.3 关键设计决策一览

1. **去中心化流控**：每 (channel, src-dst rank 对) 一条独立 ring buffer，生产者/消费者用 head/tail 单调计数器 + acquire/release 同步，无全局锁（§6.3、§7.3）。
2. **两级信号**：数据面走 NVLink/RDMA 写，控制面走原子计数器（`nvshmemi_ibgda_amo_nonfetch_add`，`internode.cu:L840`）；计数先行、数据后到，消费者用 `-value-1` 编码区分"0 token"与"信号未到"（§6.2）。
3. **通信-计算重叠**：LL 模式把 kernel 拆成 SEND/RECV 两相位（`compiled.cuh:L11-12`），RECV 相位可延迟执行（hook），等待窗口 0 SM 占用（§8.4）。
4. **CPU 只做布局**：host 侧仅用 mapped counter 忙等一次总 token 数来分配输出 tensor（`buffer.hpp:L581-583`），随后全程 GPU 自治（§4.2）。

## 2. 源码定位

| 文件 | 行数 | 职责 |
|---|---|---|
| `deep_ep/buffers/legacy.py` | 713 | Python `Buffer` 类（L14）：NVSHMEM 环境初始化（L106-126）、Config 查表（L233/L263）、dispatch/combine/LL 全部公开 API（L322/L408/L553/L598） |
| `csrc/legacy/buffer.hpp` | 1794 | C++ 编排：Buffer 生命周期（L227 sync / L292 destroy）、host 侧同步与输出分配、kernel 启动、pybind 绑定（L1791） |
| `csrc/legacy/config.hpp` | 190 | `Config` 5 字段（L25-29）、约束断言（L41-50）、NVL/RDMA buffer size hint（L53-98）、`LowLatencyLayout`（L119-158） |
| `csrc/kernels/legacy/intranode.cu` | 1117 | 单机 NVLink 域 dispatch/combine kernel（≤8 rank） |
| `csrc/kernels/legacy/internode.cu` | 2384 | 跨机 dispatch/combine kernel：NVLink+RDMA 两级转发（>8 rank） |
| `csrc/kernels/legacy/internode_ll.cu` | 1289 | 低延迟 kernel：纯 RDMA/IBGDA，FP8/UE8M0/LogFMT |
| `csrc/kernels/legacy/layout.cu` | 154 | `get_dispatch_layout` 统计 kernel（4 张 layout 表） |
| `csrc/kernels/legacy/utils.cuh` | ~590 | `SourceMeta`/`SymBuffer`/内存序原语/TMA 封装/系统 barrier（L526-527） |
| `csrc/kernels/legacy/compiled.cuh` | ~30 | 编译期常量：`LEGACY_NUM_MAX_NVL_PEERS=8`、`LEGACY_FINISHED_SUM_TAG=1024`（L14）、SEND/RECV_PHASE（L11-12）、`NUM_MAX_LOCAL_EXPERTS=1024`（L8） |
| `deep_ep/utils/gate.py` | 180 | 测试专用非均衡路由生成器（L4-137），非运行时组件 |

V2 入口（`deep_ep/buffer.py`、`deep_ep/cpp/`、`csrc/gin/`）不在本文深潜范围，§9 只做架构层对比。

### 2.1 仓库目录速览（含本文未深潜的支撑件）

```
DeepEP-main/
├── deep_ep/
│   ├── buffers/legacy.py      # 本文 §4 主线: V1 Python API (713 行)
│   ├── buffer.py              # V2 ElasticBuffer (§9)
│   ├── utils/gate.py          # 测试负载生成器 (L4-137, §9.1)
│   └── cpp/                   # V2 pybind/extension 构建 (全 JIT)
├── csrc/
│   ├── legacy/                # buffer.hpp(编排)/config.hpp(布局) — §4-§5
│   ├── kernels/legacy/        # intranode/internode/internode_ll/layout + utils.cuh
│   └── gin/                   # V2 NCCL Gin backend (§9)
└── docs/legacy.md             # V1 设计文档(双模式/hook/性能表)
```

**阅读顺序建议**：`docs/legacy.md`（设计意图）→ `legacy.py`（API 面）→ `config.hpp`（数据结构）→ `buffer.hpp`（host 编排）→ `intranode.cu`（最简流控）→ `internode.cu`（最复杂转发）→ `internode_ll.cu`（最精炼性能路径）。

## 3. 架构总览

### 3.1 四层结构

```
┌──────────────────────────────────────────────────────────────────────┐
│ L1 Python API  deep_ep/buffers/legacy.py: Buffer (L14)               │
│   __init__: NVSHMEM env + IPC all_gather (L74-126)                   │
│   dispatch/combine (L322/L408): 按 num_ranks 走 intranode/internode  │
│   low_latency_dispatch/combine (L553/L598): 纯 RDMA + FP8 + hook     │
├──────────────────────────────────────────────────────────────────────┤
│ L2 C++ 编排  csrc/legacy/buffer.hpp: Buffer (L70-71 mapped counter)  │
│   notify kernel → CPU 忙等 moe_recv_counter (L581-583, GIL release)  │
│   → 分配 recv tensor → 启动 data kernel；LL: launcher(phases)(L1577) │
├──────────────────────────────────────────────────────────────────────┤
│ L3 CUDA kernel  intranode.cu / internode.cu / internode_ll.cu        │
│   warp 角色分工 × channel 化 ring buffer × head/tail × 系统级内存序  │
├──────────────────────────────────────────────────────────────────────┤
│ L4 传输层  NVLink/IPC: st_*/ld_*_sys 原子    IBGDA: put/amo/quiet     │
│   (单机 ≤8 rank)                            (跨机, NVSHMEM RC QP)    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 一次 dispatch→combine 的完整数据流（normal 模式）

```
 Python 层                                kernel/host 层                        combine 消费端
 dispatch(x, topk_idx) (legacy.py L322)
  ├─ get_dispatch_layout (layout.cu L11)
  │   → num_tokens_per_rank / _per_rdma_rank / _per_expert / is_token_in_rank
  ├─ notify_dispatch kernel (intranode.cu L26)
  │   → 写 moe_recv_counter_mapped；host 忙等 (buffer.hpp L581-583)
  │   → 得到 recv token 总数 → 分配 recv_x/recv_topk_idx/recv_src_meta
  ├─ data kernel (intranode.cu L212 或 internode.cu L446)
  │   → 按 channel ring buffer 搬运 + 写 4 张前缀表
  └─ handle = (rank_prefix_matrix, channel_prefix_matrix,
               recv_channel_prefix_matrix, recv_src_idx,
               is_token_in_rank, send_head)        ← internode 时含 rdma/nvl 两级
        │
        ▼  MoE 专家计算（permute → grouped GEMM → unpermute）
        │
 combine(x, handle) (legacy.py L408)
  └─ 反向搬运（combine kernel 复用 send_head 找回源槽位）+ FP32 归约 → combined_x
```

### 3.3 LL 模式两相位与 hook 时序

```
 普通调用:  [ kernel: SEND phase ‖ RECV phase 一次完成 ]        ← 等待期占满 SM
 hook 调用: [ kernel: SEND phase ] → 立即返回 (return_recv_hook=True, L584)
              (SEND 发完 RDMA 请求即退出, 不等数据)
            │ ………… 0-SM 等待窗口: GPU 完全让给专家计算 prep …………
            └→ recv_hook() (L1592/1711): 再启动 [ RECV phase kernel ] 收尾
 缓冲: buffer_idx ^= 1 奇偶双缓冲 (config.hpp L119-158) → 最多 2 个未消费结果共存
```

## 4. Python API 层（`deep_ep/buffers/legacy.py`）

### 4.1 构造与 NVSHMEM 环境（L14-136）

`Buffer.__init__`（L33）做三件事：

1. **IPC all_gather**（L74-83）：同机 rank 互开 IPC 句柄，得到 `buffer_ptrs[]`——这是 intranode 模式的 P2P 通道底座（kernel 里所有 NVLink 写都落在对端 mapped 显存上）。
2. **NVSHMEM 环境变量**（L106-126），逐条 WHY：
   - `NVSHMEM_IB_ENABLE_IBGDA=1`（L109）：启用 In-Band GPU-Direct Async，让 kernel 直接下发 RDMA put/amo 而不回 CPU。
   - `NVSHMEM_IBGDA_NUM_RC_PER_PE=num_qps_per_rank`（L110）：每 PE 的 RC QP 数。normal 模式按通道配，LL 模式 = 本地专家数（见 §8.2）；QP 数直接决定可并发的 RDMA 流数。
   - `NVSHMEM_QP_DEPTH ≥ 2×(num_max_dispatch_tokens_per_rank+1)`（L113-114）：QP depth 下限。**WHY**：一次 dispatch 最多向同一对端排队 `max_tokens` 个消息 + 计数原子操作，depth 不足会挂起 WQE 队列造成死锁。
   - `NVSHMEM_MAX_TEAMS=7`（L118）、`NVSHMEM_DISABLE_NVLS=1`（L120）：NVLS 多播与 LL kernel 的逐 slot 写模式冲突，显式关闭。
   - `NVSHMEM_CUMEM_GRANULARITY=2^29`（L122）：512 MiB 粒度，降低大 buffer 的注册表项数量。
3. **RDMA buffer 注册**：对称分配（所有 rank 同 size），供 NVSHMEM/IBGDA 远端寻址。

### 4.2 Config 查表与启动参数（L233-292）

`get_dispatch_config/get_combine_config`（L233/L263）按 `num_ranks` 静态查表返回 5 元组 Config（TODO 注释承认后续自动调优）。注意 **dispatch 与 combine 的表不同**：例如 64 rank 时 dispatch=(32, 288, 8, 128)、combine=(1, 288, 8, 128)——combine 的 NVL send tokens 只要 1，因为 combine 方向每源 rank 的回包更碎（每 token 按 top-k 分散后又归并），大批量会破坏流控假设（§5.2 的 send<recv 断言链）。

### 4.3 dispatch/combine 返回值与 handle（L322-456）

- `dispatch(x, topk_idx, topk_weights, ...)`：按 `num_ranks ≤ 8 → intranode_dispatch`（L458），否则 `internode_dispatch`。返回 `(recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event)`。
- **handle 是跨 kernel 的"寄存器"**：intranode 6 元组 `(rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)`；internode 扩展为含 rdma/nvl 两级前缀与 `send_rdma_head/send_nvl_head` 的 10 元组。combine 用它免掉第二次路由计算（`cached_notify` 系列 kernel 直接消费前缀表，`intranode.cu:L176/L627`）。
- **cached 模式**：第二次调用同 shape 的 dispatch 时跳过 notify kernel 与 layout 统计（`buffer.hpp` 走 `cached_notify_dispatch`，`intranode.cu:L176`），省一次全 barrier——**前置条件：路由分布与上次完全一致**（两调用间 topk_idx 不变）。
- `get_dispatch_layout`（L293）单独暴露：把 topk_idx 统计成 4 张表（`layout.cu:L11-122`：expert 统计每 SM 管 4 个 expert（L134 `kNumExpertsPerSM=4`），rank 统计每 SM 管 8 个 rank，RDMA rank 按 `LEGACY_NUM_MAX_NVL_PEERS=8` 归并（L59））。**该 kernel 是 normal 模式的路由前置**：dispatch 的 channel 任务划分（`get_channel_task_range`）与 RDMA/nvl 两级前缀全部来自这 4 张表。

### 4.4 EventOverlap 与流（L166-199）

`capture()` 返回 `EventOverlap`（L166）——normal 模式的计算重叠句柄：dispatch 在独立 comm stream 上 launch（`get_comm_stream` L191），返回的 event 供 Megatron 侧 `EventOverlap` 让 MoE 计算 kernel 与 A2A 并发；`set_num_sms`（L154）可在运行中调整数据 kernel 的 SM 预算（与 Config 的 num_sms 同源，`buffer.hpp` 侧透传给 launch grid）。**边界**：重叠窗口只在 dispatch 的 SEND 侧有效，combine 的归并必须等数据（V1 无 combine 侧 hook——这是 LL 相位拆分存在的理由之一，§8.4）。

### 4.5 LL API 与容错（L538-622）

`low_latency_dispatch(x, topk_idx, topk_weights, num_max_dispatch_tokens_per_rank, ..., use_fp8=True, round_scale=False, use_ue8m0=False, ..., return_recv_hook=False)`（L553-558）：

- 返回 **静态 shape 的 packed 输出**：`packed_recv_x = [num_local_experts, num_ranks × num_max_dispatch_tokens_per_rank, hidden]`（L590-597 docstring）——这就是"LL 模式接收端按 `[expert][rank][slot]` 静态预留"的 API 体现；`packed_recv_x_scales` 列主序打包（UE8M0 时为 uint32 pack，`internode_ll.cu:L168/L443-458`）。
- `return_recv_hook=True` 时 kernel 只做 SEND 相位，返回的 `recv_hook` 延迟执行 RECV 相位（L584-605 docstring；§8.4）。**约束：同一 Buffer 上最多同时存活 2 个未消费的 LL 结果**（双缓冲，L605 附近 docstring 警告）。
- **容错**：`mask_buffer_ptr` 支持把超时/故障 rank 屏蔽出集合（`internode_ll.cu:L10-19 is_rank_masked`，接收超时自动 `atomicExch(mask_buffer_ptr+src_rank,1)` L397-405）；配套 `clean_low_latency_buffer`（L538）清下一轮的计数/标志 buffer。
- `use_ue8m0=True`（仅 SM90+，需 `round_scale=True`）：scale 以 2 的幂打包，是为 DeepGEMM 的 UE8M0 排布（L582 docstring）。

## 5. Config 与 buffer 布局（`csrc/legacy/config.hpp`）

### 5.1 Config 五字段与约束（L25-50)

```cpp
struct Config {
  int num_sms;                            // 数据 kernel 占用 SM 数 (L25)
  int num_max_nvl_chunked_send_tokens;    // NVL 一次连续发送量 (L26)
  int num_max_nvl_chunked_recv_tokens;    // NVL ring buffer 深度 (L27)
  int num_max_rdma_chunked_send_tokens;   // RDMA 一次连续发送量 (L28)
  int num_max_rdma_chunked_recv_tokens;   // RDMA ring buffer 深度 (L29)
  ...
  EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);      // L43
  this->num_max_rdma_chunked_recv_tokens =
      align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);  // L47
  EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);    // L48
  // RDMA lazy head 更新相关: 必须保证 sender 始终有空间推数据 (L49 注释)
  EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2); // L50
};
```

**WHY send < recv 且 RDMA send ≤ recv/2**：head 的更新是**懒惰/批量**的——接收方要攒够一段（≥ send_tokens）才通过原子加归还 head（`internode.cu:L1048-1056` 的 `min_head ≥ last_head + num_max_rdma_chunked_send_tokens` 条件）。若 send 深度超过 recv 的一半，接收方"还没来得及释放"而发送方"已写满整个 buffer"会同时成立，双方互相等待 → 死锁。NVL 侧同理但阈值宽松（combine 表里 send=1 是极端保守取值）。`align_up`（L47）保证 recv 深度是 send 的整数倍，使 coordinator 的 `num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0` 断言（`internode.cu:L760`）恒成立。

### 5.2 buffer size hint 公式（L53-98）

**NVL 域**（`get_nvl_buffer_size_hint`，L53）：每 rank 的 IPC 总量 ≈ 每 channel × 每 peer 一条 ring：

```
num_channels = num_sms / 2                                  (L62)
size ≈ Σ_channels Σ_peers [
  num_max_nvl_chunked_recv_tokens × (hidden_bytes            // x 数据 (L66)
  + source_meta_bytes                                        // SourceMeta (L67)
  + num_topk × sizeof(topk_idx) × 2) ]                       // topk idx+weights (L68)
```

**RDMA 域**（`get_rdma_buffer_size_hint`，L78 区）：同理按 `num_rdma_ranks` 展开，外加 2×int4 对齐余量与 count/signaling 区（L95-98）。

### 5.3 LowLatencyLayout：奇偶双缓冲（L119-158）

```
LowLatencyBuffer {  // (L105-117) 每个 buffer 三段
  dispatch_rdma_send_buffer, dispatch_rdma_recv_data_buffer, dispatch_rdma_recv_count_buffer,
  combine_rdma_send_buffer,  combine_rdma_recv_data_buffer,  combine_rdma_recv_flag_buffer,
}
LowLatencyLayout:
  num_bytes_per_dispatch_msg = sizeof(int4) + max(BF16 hidden×2, FP8 hidden + num_scales×4)   (L135)
  num_bytes_per_combine_msg  = num_scales×4(BF16x2 min/max) + hidden×2                        (L136)
  total = send×2 + recv×2 + signaling×2   // send/recv/signaling 各两份, odd/even 交替 (L153/160/166)
```

**WHY 双缓冲**：CUDA Graph capture 要求地址固定；hook 模式下"上一轮的 RECV 尚未消费、下一轮的 SEND 已要开跑"是合法时序，奇偶交替让两代数据共存，代价是 RDMA buffer 翻倍。`clean_meta()`（L114-118）返回下一轮要清零的 count buffer 指针对（LL kernel 的 `next_clean` 参数，`internode_ll.cu:L287-289`）。

## 6. intranode kernel：NVLink 域 dispatch/combine（`intranode.cu`）

> 适用：EP 组全部 rank 落在同一 NVLink 域（≤ `LEGACY_NUM_MAX_NVL_PEERS=8`）。所有 P2P 写经 IPC mapped 显存，用系统级原子原语同步。

### 6.1 notify_dispatch：设备端通信 + mapped counter 回传（L26-168）

单 block（sm_id==0）先做全 rank barrier（L47 `barrier_block<kNumRanks,true>`），随后**所有 rank 的 SM0 在设备端交换 `num_tokens_per_rank` 并做前缀和**（L77 写 `moe_recv_counter_mapped` = 全组收到的 token 总数，L90/L118 两次 `__syncthreads` 分阶段）。其他 block（≥1）并行计算 channel 级前缀 `channel_prefix_matrix` 写入共享 IPC 区。host 侧（`buffer.hpp:L549-583`）置 `*moe_recv_counter=-1` → 启动 notify kernel → **CPU 忙轮询 mapped counter 直到非负**（L581-583 `while(true)`，期间 GIL release、1s 超时告警）→ 按返回值分配 recv tensor。

**WHY 让 GPU 算、CPU 等而不是反过来**：路由统计天然是设备数据（topk_idx 在显存），若回 CPU 做 all-to-all，需 3 次 D2H/H2H/H2D 往返 + 集合通信同步；设备端做只需一次 4 字节回传。CPU 忙等的唯一目的是分配输出 tensor（PyTorch 要求 host 侧知道 shape），窗口仅覆盖 notify kernel 生命周期。

### 6.2 dispatch kernel（L212-546，kNumRanks 模板）

**线程组织**（L211-265 区）：`kNumThreads=768`（24 warp），`num_channels = num_sms/2`，**每个 channel 占 2 个 SM，SM0 偶/发 SM1 奇收**（对应 `config.hpp:L59` 的 `num_sms%2==0` 断言）。每 rank 一组线程（`num_threads_per_rank`），用**具名 barrier `bar.sync responsible_rank`**（L409/L470/L526）把同一 rank 的线程分成同步域——**WHY 具名 barrier**：不同 rank 的进度天然不齐，全局 `__syncthreads` 会把整块 SM 的 24 个 warp 绑死在最慢 rank 上；具名 barrier 让每 rank 独立流水。

**ring buffer 与负值编码**（L322-324）：发送方把"本 channel 起止偏移"以 `-value-1` 编码写进 `channel_start_offset/channel_end_offset`——0 是合法值（该 channel 0 token），接收方用"负值=信号已到，解码后为真实值"区分"还没到"（L430-432 忙等 `==0`）。这是全文反复出现的**信号编码惯例**：LL 计数 `-num_tokens_sent-1`（`internode_ll.cu:L336`）、meta 计数（`internode.cu:L599-611`）同理。

**发送端**（L330-411）：每发送 warp 以 `num_max_nvl_chunked_send_tokens` 为 chunk 填 ring，`UNROLLED_WARP_COPY(5,...)`（L374）直写对端 IPC 显存，chunk 完成 `st_release_sys_global(channel_tail_idx)`（L411）。

**接收端**（L416-536）：`ld_acquire_sys_global(channel_tail_idx)` 观察新数据（L451），TMA 路径 `tma_load_1d` + mbarrier 双 half（L484，per-warp `kNumTMABytesPerWarp` 共享内存）或 `UNROLLED_WARP_COPY` 回退；逐 chunk 消费后 `st_relaxed_sys_global(channel_head_idx)` 归还槽位（L528）。**relaxed vs release 的分工**：tail 释放必须 release（数据先行原则），head 归还只管"槽位空了"，与数据无依赖，用 relaxed 省一次 fence。

### 6.3 流控与内存序伪代码

```
Sender(warp):                                    Receiver(warp):
while tokens remain:                             head = ld_acquire_sys(tail[peer])   # 看见数据才算数
  # 尾部深度 = ring 深度 - in-flight, 保证不追尾   chunk = pop(ring, head)
  wait(tail[peer] - my_head + chunk <= DEPTH)     copy chunk → recv_x (TMA/UNROLL)
  write chunk → peer IPC memory (st_na)           head += chunk
  st_release_sys(tail[me], my_tail+=chunk)        st_relaxed_sys(head[peer], head)
```

（内存序原理：`st_release` 保证"此前的普通写对 observes tail 的读者可见"；对端 `ld_acquire` 与之配对。跨 GPU 的可见性靠系统域——`st_release_sys_global/ld_acquire_sys_global`（utils.cuh 提供），域选小了（如 cta）NVLink 对端看不见，选大了白白降吞吐；普通数据写用 `st_na_global`（非原子直写）+ 对方 release-tail 兜底可见性，避免逐字原子的开销。）

### 6.4 combine kernel 与 min-head retire（L694 起）

combine 反向：每 token 按 `send_head`（记录"我的第 j 个 top-k 落在目标 rank ring 的第几槽"）从各源 rank ring 读回，**FP32 精度累加** top-k 份输出再写回（标准是 BF16 输入 FP32 归约，避免 top-k 次加法的精度损失）。槽位回收沿用**min-head retire**：协调者收集各消费 warp 的 head，取最小值一次性释放——所有源 rank 中"最慢的消费者"决定槽位寿命（与 internode 转发协调者 `internode.cu:L1036-1056` 同构）。cached_notify_combine（L627-690）是 cached 模式变体：sm0 只清 buffer + barrier，跳过前缀重算。

### 6.5 边界与约束

- 仅 kNumRanks ≤ 8；rank 数 > 8 必须走 internode（`legacy.py` 按 num_ranks 分支）。
- `hidden_int4 × 2 ≤ per-warp TMA buffer`，超了走 UNROLLED 拷贝路径（L484 vs L492）。
- dispatch 的 `num_worst_tokens` 路径（L538-546）：预分配时用最坏 token 数清尾，防上轮残留 -1 干扰。
- CPU 忙等窗口内若 kernel 卡死，host 侧 1s 超时抛错（`buffer.hpp:L581-583` 区，1s 常量在启动参数）——诊断时先看是不是某个 rank 掉队导致 notify 不齐。

## 7. internode kernel：NVLink→RDMA 两级转发（`internode.cu`）

> 适用：EP 组 >8 rank（跨机）。核心思想：**跨机流量只走一次 RDMA**——同一 RDMA rank（= 一台机器的 8 个 NVLink rank 组）内的流量经 NVLink 分发，只有真正跨机的那部分才上 RDMA，然后由转发 warp 分发到机内各 NVLink rank。这正是 DeepSeek-V3 group-limited gating（专家分组约束路由到邻近节点）能省广域带宽的结构基础。

### 7.1 dispatch kernel：5 种 warp 角色（L446-1213）

kernel 入口先定角色（L487 `WarpRole` 枚举，L499-513 角色分配）；`num_channels = num_sms/2`，**偶数 SM 是 forwarder SM**（L493-494 `channel_id=sm_id/2; is_forwarder=sm_id%2==0`），每 SM warp 数 = `kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS`（launch_bounds L452）：

| 角色 | warp 数 | 目标 | 职责 |
|---|---|---|---|
| kRDMASender | 配置（如 18） | — | 把本机 token 广播拷入对称 send buffer，攒窗口 |
| kRDMASenderCoordinator | 1 | — | 窗口 coalescing 后批量 IBGDA put + 原子更新远端 tail |
| kRDMAAndNVLForwarder | 8（每 warp 一个目标 nvl_rank） | target_rank | 从 RDMA ring 收，向目标 NVLink rank 的 ring 发 |
| kForwarderCoordinator | 1（多余退出 L1021-1022） | — | 汇总转发进度，批量释放 RDMA head |
| kNVLReceivers | 8 | target_rank | 从 NVLink ring 收，写入最终 recv tensor |

**对称/非对称 buffer 视图**（L526-560）：RDMA 侧用 `SymBuffer`（所有 rank 同布局，对称地址 = 对端可直接寻址）；NVL 侧用 `AsymBuffer` + `rs_wr/ws_rr` 双指针——"发送者写/接收者读"与"发送者读/接收者写"分别落在不同 rank 的 IPC buffer 上（L535-542 选指针逻辑），头/尾计数器由**槽位所有者**持有（L557-560），天然避免写冲突。

### 7.2 RDMA sender：32-slot 位图窗口（L562-757）

发送 warp 的核心状态是三个 shared 数组（L565-567）：`rdma_send_channel_lock/tail/window[kNumRDMASenderWarps 目标数]`。流程（L730-755）：

```
每个 token（每 warp 轮流负责 token_idx % kNumDispatchRDMASenderWarps）:
  lane 保存各 RDMA rank 的 slot_idx（-1 = 不去该 rank）        (L645)
  等待远端 head 释放: tail - cached_head >= DEPTH 时自旋       (L649-663, clock64 超时 trap)
  拷贝 x/scales/SourceMeta/topk 到各目标 send buffer           (L688-727, st_broadcast 广播写)
  acquire_lock(lock[lane]) → offset = tail - latest_tail
    while offset >= 32: 释放锁重读 tail                        (L736-741)
    window |= 1 << offset          # 标记"该事务数据已就绪"
    if offset == 0:  # 我是窗口头 → 数连续就绪位 n=__ffs(~window)-1
        st_release_cta(tail, latest_tail + n)                  (L748)
        window >>= n                # 滑动窗口
    release_lock
```

**WHY 要窗口**：IBGDA put 是异步的，若每个 token 都触发一次远端 tail 原子更新，QPN/原子流量爆炸；窗口把"连续就绪"的事务打包成一次 tail 释放，coordinator 只在窗口头推进时才需要被唤醒。**为什么 lock 是必须的**：同 lane 位（同一目标 rank）的多个 warp 竞争窗口位图，`acquire_lock/release_lock`（utils.cuh 提供）串行化位图修改。

### 7.3 coordinator：coalescing 批量 put（L758-848）

coordinator warp 轮询每个目标 rank 的窗口进度（L806 `ld_acquire_cta` 读 shared tail），满足 `processed==total 或 processed ≥ chunked_send_tokens`（L809）才发起 `nvshmemi_ibgda_put_nbi_warp`（L823）——把 send buffer 里攒好的整块（`num_tokens_to_issue × num_bytes_per_token`）一次 put 到对端 recv buffer 对应 slot，随后 `nvshmemi_ibgda_amo_nonfetch_add` 远端 tail += issue 量（L840-845）。本机 rank 目标只做 `memory_fence()`（L832，本地写已可见，put 省略）。**incast 缓解**：遍历目标顺序按 `(i + channel_id + rdma_rank) % kNumRDMARanks` 打散（L798），避免全组同时轰同一目标。

### 7.4 forwarder：round-robin 双 ring 中转（L849-1018）

每个 forwarder warp 绑定一个目标 NVLink rank（L851），三步循环：

1. **等 meta**（L853-899）：`ld_volatile_global` 轮询 4 个 meta 计数（来自所有源 RDMA rank 的 `-value-1` 编码，L599-611 发出），全负即到齐；解码后把本 channel 的 NVL 起止前缀再以负编码写给 NVL 接收方（L867-868），并记下 `send_nvl_head` 的全局偏移（L902）。
2. **转发**（L907-1013）：`src_rdma_rank` 从 `(sm_id % kNumRDMARanks)` 起步轮询（L909，**WHY 交错起点**：8 个 forwarder warp 各自偏移，避免同时挤同一个源 rank 的 ring）；对每个到达 token 读 `SourceMeta`（L969）判断是否属于本目标 nvl rank，是则 TMA load→smem→TMA store 搬到目标 ring（L986-1001），并把 `send_nvl_head[i×8]=槽位或-1`（L975-976）记录给 combine；攒够 `num_max_nvl_chunked_send_tokens` 提前收尾本批（L996-998）。
3. **释放**（L1019-1060）：kForwarderCoordinator 汇总 `forward_channel_head[dst_nvl][src_rdma]`（L583 shared 表），**min-head ≥ last_head + chunked_send_tokens 才批量 amo 归还**（L1048-1056，翻译到 `channel+num_channels` 通道——RDMA buffer 的 head/tail 各用一套通道索引），`__nanosleep` 让路（L1059）。

### 7.5 NVLReceiver 与本地化（L1061-1213）

接收 warp 按 lane 读 8 个源 RDMA rank 的 NVL 前缀（L1069-1102），逐 token：TMA 拷 hidden+aligned scales（L1141-1151，**scale 16B 对齐才走 TMA**，L1137-1138）、非对齐回退 UNROLLED（L1157-1164）、写 `recv_src_meta`（L1169）、**topk_idx 本地化**（L1174-1184：全局 expert id → 本地 id，不属于本机的置 -1 且权重清 0——combine 端据此跳过）。最后若声明了 `num_worst_tokens`，非 forwarder SM 分块清尾（L1199-1212）。

### 7.6 combine kernel（L1711-2282）

角色镜像 dispatch（L1746：kNVLSender/kNVLAndRDMAForwarder/kRDMAReceiver/kCoordinator；**奇数 SM 是 forwarder**，L1752，与 dispatch 相反——两 kernel 各自把"需要更多共享内存/寄存器的一方"放在偶数 SM）。数据流：源 token 的 MoE 输出先按 `combined_nvl_head/combined_rdma_head`（dispatch 时记录的槽位）逆向收集，在归并侧做 **FP32 累加**（top-k 权重加权）后写回 `combined_x`；跨机方向同样有"RDMA put + 原子计数 + NVLink 转发"三级，槽位经由 dispatch 的 handle 元组免重算。

### 7.7 时序图：一个 token 的跨机旅程（rank0→rank9 的专家，rank0/9 同机）

```
rank0 SM(偶)                      RDMA rank0(本机)            rank9(远端机)
 RDMASender: x,scales,meta ──┐   (对称 send buffer)
   slot=global_tail++        │
   window |= 1<<offset ──────┼→ Coordinator: 攒满 chunk
                             │    ibgda_put(→远端 recv buf) ══╗ IBGDA RC QP (QP id=channel_id)
                             │    amo tail+=n ═══════════════╣
                             │                                ▼
                             │              Forwarder warp(绑nvl_rank=1): 读 meta(-编码)
                             │              TMA: RDMA ring → NVL ring(rank9 显存)
                             │              send_nvl_head[slot]=i
                             │              Coordinator: min-head 批量归 head
                             │                                ▼
                             │              NVLReceiver(rank9): ring → recv_x
                             │                                → 专家计算 → combine 逆向
```

## 8. low-latency kernel：纯 RDMA + IBGDA（`internode_ll.cu`）

> 适用：小批量 decode/训练 LL 场景。**节点内也走 RDMA**（除非 `allow_nvlink_for_low_latency_mode`，此时同机对端 `nvshmemi_get_p2p_ptr` 非 0 走 P2P 拷贝直写，L263-272）。模板参数 `<kUseFP8, kUseUE8M0, kHidden>`（L128）——hidden 是编译期常量，1024 线程/SM（L129 `__launch_bounds__(1024,1)`）。

### 8.1 消息布局与两相位骨架（L177-190, L352-359）

```
num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? hidden + num_scales×4 : hidden×2)   (L180)
                     └ src token idx + 3 reserved    └ 数据 + scales
phases & SEND_PHASE(1) == 0 → goto RECV;  phases & RECV_PHASE(2) == 0 → return   (L189/L354)
两相位同 kernel: 发送后 cg::this_grid().sync() (L358-359) 才能开始收（保证计数可见）
```

### 8.2 SEND 相位：FP8 量化 + per-expert 定点投递（L188-349）

- **warp 分工**（L195）：前 `num_warps-1` 个 warp 逐 token 做 cast+发送，最后一个 warp 读 `topk_idx` 统计 per-expert 计数。
- **FP8 cast in-kernel**（L212-245）：每 lane 读一个 int4（8×BF16），算局部 amax → `warp_reduce_max<16>`（L232，128 通道一组）→ `calculate_fp8_scales(amax, scale, scale_inv, round_scale)`（L233）→ lane0/16 写 scale_inv（L234-235）→ `__nv_cvt_float2_to_fp8x2` 成对转换写 send buffer（L239-245）。**WHY 在通信 kernel 里做 cast**：省一次全量量化的独立 kernel（1024 线程×逐 128 通道 amax），LL 场景每 µs 都值钱；副作用是 amax 用 `kFP8Margin=1e-4` 下限防全零（`utils.cuh:L477`）。
- **定点投递**（L254-273）：`dst_expert_idx = topk_idx[token][warp_id]`（L209，每 warp 恰好负责该 token 的一个 top-k 专家）→ 目标 slot 由 `atomicAdd(atomic_counter_per_expert)` 现场分配（L255-256）→ 目标地址 = `rdma_recv_x + expert×(ranks×max_tokens×msg) + rank×(max_tokens×msg) + slot×msg`（L260-262）→ **QP 选择：`nvshmemi_ibgda_put_nbi_warp(..., dst_rank, dst_expert_local_idx, lane_id, slot_idx)` 的第 5 参 qp_id = 目标本地专家号**（L266）——即**每对 (src,dst) rank 有 num_local_experts 条 QP 并发**，把"多 expert 小消息"摊到多 QP 上并行下发；`num_rc_per_pe ≥ num_local_experts` 断言（L284）。若对端 P2P 指针非 0（同机）改走 `UNROLLED_WARP_COPY(8,...)` 直写（L265-272）。
- **完成计数**（L277, L291-345）：每个专家两个计数器——`atomic_counter_per_expert`（动态 slot 分配）与 `atomic_finish_counter_per_expert`（完成量）。尾 warp 发出 `LEGACY_FINISHED_SUM_TAG(=1024, compiled.cuh:L14)` 基准（L295），计数器达到 `TAG×2`（L330）即"发送完成量+统计量"双到位，才发 `amo_nonfetch_add(rdma_recv_count, -num_tokens_sent-1)`（L336，负编码）或同机 `st_release_sys_global`（L338）。

### 8.3 RECV 相位：静态 packed 布局与容错（L352-461）

- 每 warp_group 负责一个本地 expert（`responsible_expert_idx = sm_id×num_warp_groups + warp_group_id`，L731 区角色表），sub-warp0 忙等 `ld_acquire_sys_global(rdma_recv_count + expert×num_ranks + src_rank)`（L387-391），`clock64` 超过 `LEGACY_NUM_TIMEOUT_CYCLES` 时**把源 rank 写进 mask buffer**（L397-405，若未启用 mask buffer 直接 `trap()`）。
- 拷贝：`recv_src_info`（源 token 号）、数据 `UNROLLED_WARP_COPY(7,...)`（L436）、scales 按列主序 pack 写入（L439-458，CuTe 等价布局注释 L440-441：`(token,(pack,elem)):(pack, tokens×pack,1)`——UE8M0 时 4 个 scale 打进一个 uint32，`packed_t=uint32_t` L168）。
- 输出布局就是 packed 静态 shape：`[local_expert][src_rank][slot]` → 训练侧按 `packed_recv_layout_range`（L411 pack2）做零拷贝 permute。

### 8.4 hook 机制与 0-SM 等待窗口（L188-190, buffer.hpp:L1577/L1592/L1696/L1711）

```
launcher = lambda phases: 启动 kernel(phases=SEND|RECV)
普通模式: launcher(SEND | RECV)                    # 一次 launch, 等待期占 SM
hook 模式: launcher(SEND)   → 返回 recv_hook
          recv_hook = lambda: launcher(RECV)       # 二次 launch, 等待期 0 SM
```

**WHY 拆相位**：LL 场景 kernel 大部分时间在等远端数据（µs 级延迟 >> 拷贝时间）。SEND 相位只发请求（不等回包）几 µs 就退场，把等待期让给上层做专家权重预取/共享专家计算；等到要消费时再起 RECV 相位。代价：多一次 launch（~2µs）+ `buffer_idx^=1` 双缓冲支持两代数据共存。`zero_copy` combine（L733 参数区）更进一步：直接从 MoE 输出缓冲读，省一次显式拷贝（需 `get_next_low_latency_combine_buffer`，`buffer.hpp:L1717`）。

### 8.5 combine kernel：LogFMT 9-bit 与 FP32 归约（L715-1139）

模板 `<kUseLogFMT, kHidden, kNumMaxTopk, kNumMaxUnrolls>`（L700 区）。LogFMT 是自定义 10-bit 浮点（1 符号+9 位，含 3 位共享指数打包进 int 的 concat 位段）——`decode(concat[k]>>18 & 0x1ff, sign)`（L702-711 区）把 32bit 还原成 2×BF16 精度的近似值，**用 10bit/元素替代 16bit BF16 传输，combine 通信量省 ~35%**；开关在 Python `use_logfmt`（`legacy.py` L557 参数区）。归并循环：每 lane 维护 `combined_values[kNumRecvUnrolls×4]` FP32 累加器（L1081），乘 `topk_weight` 后 `tma_st_buffers` 落 shared，最后 `st_global` 写回 combined_x（L1125-1131）。同一 kernel 的 send 相位负责把本地 expert 输出 put 给所有相关源 rank（对称于 dispatch 的 RECV 布局）。

### 8.6 边界与约束

- `num_topk ≤ 8`（per-warp 一个 top-k 的设计上限，L209 的 `warp_id < num_topk`）；hidden 必须被 `32×8` 整除（L197）。
- **CUDA Graph 兼容**：输出静态 shape + `next_clean` 参数化清零（L287-289）+ 无 host 同步，这是 LL 模式能被 capture 的全部前提；normal 模式因 CPU 忙等分配 tensor 不可 graph 化。
- 容错与 mask buffer 见 §4.4；**rank 被 mask 后 count 永远 0，接收侧以 -1 token 跳过**（L394-395）——训练框架需在 mask 生效后重置集合。

## 9. V2 架构演进（main 分支新形态）与 V1 的关系

V2 在仓库 main 分支上以 `deep_ep/buffer.py`（ElasticBuffer）+ `csrc/gin/`（NCCL Gin backend）+ 全 JIT 编译呈现，README 与 V1 legacy 并存。关键变化及 WHY：

| 维度 | V1 legacy（本文 §3-§8 主线） | V2（README/当前 main） |
|---|---|---|
| API | `Buffer` + `Config` + 显式 Config 查表 | `ElasticBuffer` 统一 high-throughput/low-latency 两套 API |
| 传输后端 | NVSHMEM + IBGDA（强依赖 NVSHMEM 安装/env） | NCCL Gin backend（复用 NCCL 传输栈，去掉 NVSHMEM 依赖） |
| 编译 | 预编译扩展（需匹配 CUDA/toolkit） | **全 JIT**（运行时 nvcc/nvrtc，随驱动/CUDA 自适应） |
| SM/QP 配置 | rank 数查表（§4.2）+ 自动调优 TODO | **解析法直接算出 SM 数/QP 数**（无需离线自动调优） |
| 规模上限 | 实测到 EP64（docs/legacy.md） | 宣称 EP2048 |
| 性能 | normal 51 GB/s@64EP RDMA；LL 77µs@128tok | 90 GB/s RDMA@12SM（SM90 CX7 EP8×2，**省 4× SM**）；SM100 NVLink 726 GB/s@64SM；对 V1 最高 1.3× 带宽 |

**与 FlagScale/Megatron 的关系**：当前生产集成路径（Megatron `MoEAllGatherTPExpandK`/`use_deep_ep` 分支）消费的是 V1 API（`Buffer + Config + EventOverlap` + hook），因此本文以 legacy 为主线；V2 的解析式配置思想（§12 调优可直接借用其公式）与 Gin 后端是后续升级方向。`deep_ep/utils/gate.py`（180 行）属于 V2 测试体系：构造**精确非均衡路由**（`generate_rank_count` 按 ratio 控制热 rank 流量倍率，L32-113；`get_precise_unbalanced_scores` 明确注明"非真实分布"L116-137）来压测两模式在 skewed 负载下的正确性与带宽——这回答了"负载不均时 DeepEP 是否仍稳"的验证问题。

### 9.1 V2 三个变化的机制含义

- **全 JIT**：V1 预编译扩展要求 `torch extension` 与集群 CUDA/toolkit 严格匹配（集群升级驱动即要重编）；V2 把 `csrc/` 编译移到首次 import 时按本机 CUDA 版本现场编译（JIT cache 落用户目录）。**收益**：跨集群可移植；**代价**：冷启动多几十秒编译、CI 无法捕获编译期问题（问题推迟到部署现场）。
- **NCCL Gin backend**：V1 的 NVSHMEM 栈需要独立安装、独立 env（§4.1 的 8 个变量）、独立 HCA 绑定；Gin 复用 NCCL 已注册的传输资源（bootstrap 连接、拓扑检测、网络注册），把"对称 buffer + 设备端通信原语"嫁接到 NCCL 传输栈上。**WHY 值得做**：NVSHMEM 的部署摩擦（版本、IB 驱动、env 调优）是 V1 落地的主要运维成本；代价是 V2 初期版本对 IBGDA 级别的细粒度控制（per-expert QP、amo 语义）要重新映射。
- **解析式资源配置**：V1 的 `num_sms`/chunked tokens 来自人工查表（`legacy.py:L233-292` 的 TODO: automatically tune）；V2 按"每 channel 需要 2 SM + 每 SM 寄存器/共享内存占用 + RDMA QP 数与通道数关系"直接解出（README：无需自动调优），把 §12.3 的调优顺序第 1-3 步收敛为一次公式计算。
- **规模外推**：V2 宣称 EP2048——两级转发架构（§7）天然外推到多级 RDMA 域（机内 NVLink → 机间 RDMA → 跨 fabric），但要防 meta 计数（L599-611 的 `-value-1` 表）随 rank 数线性放大占用 lane 位宽（当前 `LEGACY_NUM_MAX_NVL_PEERS×2+2 ≤ 32` 断言，L593）。

### 9.2 迁移判据

| 信号 | 动作 |
|---|---|
| 集群 CUDA/驱动频繁升级 | 优先 V2（JIT 免重编） |
| 需要 CUDA Graph + FP8 + UE8M0 组合 | 暂留 V1 LL（V2 Gin 的 fp8 语义需核对其 kernel 覆盖） |
| EP > 256 或跨 fabric | 评估 V2（解析式配置 + 规模验证） |
| FlagScale 现网训练任务 | V1 API 不变，勿在训练进程混初始化两代 |

## 10. 量化分析：通信量、带宽与流水线

### 10.1 通信量公式

设 hidden=H（BF16, 2B）、topk=K、num_ranks=R、num_experts=E、batch token 数 T、每 rank 本地专家 E/R：

```
dispatch 网络总流量(忽略 meta/scales):
  理想 A2A 流量 = T × K × H            # 每个 (token, expert) 对一份
  normal 模式实际: 跨机链路流量 = Σ_rdma_pairs T×K×H×(跨机占比)   # NVLink 转发不占广域
                机内 NVLink 流量 = 每机 (本机命中的 K 份 × H) 转发 + RDMA 落地再分发
  LL 模式实际: 每 (token,expert) 一次 put = T×K×(H_fp8 + H/128×4 + 16)  # FP8 时 H + 4B/128ch
combine 网络总流量: normal = T×H×K 与 dispatch 对称；LL 用 LogFMT ≈ 0.65×T×H×K (10bit/16bit)
```

**数值示例**（DSv3-lite：H=7168, K=8, FP8, T=4096, 64 EP = 8 机×8 卡）：
- dispatch 理想流量 = 4096×8×7168 = 235 MB/层；FP8 后 ≈ 235×(1024/2048) ≈ 118 MB + scales 9.2 MB。
- 64 EP 均匀分布时跨机占比 = 7/8，跨机链路 ≈ 206 MB/层；51 GB/s 实测带宽下单层 dispatch 耗时 ≈ 4.0 ms（大 batch 下与专家 GEMM 同量级，这就是 normal 模式要 coalescing/TMA 的原因）。
- LL decode（T=128, K=8）：dispatch 消息 = 128×8×(7168+56×4+16) ≈ 7.5 MB，实测 77 µs → 98 GB/s 聚合带宽，即 docs/legacy.md 数字。

### 10.2 buffer 与 QP 的量约束

```
RDMA buffer/接口卡 ≈ num_channels × num_rdma_ranks × rdma_chunked_recv_tokens × bytes_per_token   (config.hpp L78-98)
LL RDMA buffer    = LowLatencyLayout: max(dispatch, combine)×2(双缓冲)                            (L153-166)
  dispatch_recv = E × max_tokens × (16 + H_fp8 + H/128×4)
  combine_recv  = E × max_tokens × (H/128×4 + H_bf16)
QP depth ≥ 2×(max_tokens+1)   (legacy.py L113)
LL QP 数/对端 = num_local_experts  (internode_ll.cu L266 qp_id=expert → L284 断言)
```

**数值示例**（LL, DSv3-lite, max_tokens=128, E=256, R=64）：dispatch_recv = 256×128×(16+7168+224) ≈ 245 MB；combine_recv = 256×128×(224+14336) ≈ 482 MB；双缓冲 ×2 → ~1.5 GB/卡——这就是 LL 模式"用显存换延迟"的定价，也是它不能直接用于大 batch 训练的原因（normal 模式同样流量只需 chunked 深度 ≪ max_tokens）。

### 10.3 SM 预算

normal 模式 kernel 占 `num_sms` 个 SM（Config L25；查表 32-36 个），其余留给 MoE 计算；LL 模式固定 1024 线程全 SM 短时占用（hook 模式把等待期卸载后，占空比大幅下降——V2 宣称 12 SM 达 90 GB/s 即沿此思路压缩常驻 SM）。

## 11. 设计决策对比表

### 11.1 normal vs low-latency（同一仓库内两条路径）

| 维度 | normal（intranode/internode） | low-latency（IBGDA） |
|---|---|---|
| 目标场景 | 训练大 batch prefill | decode 小 batch / 训练 LL |
| 拓扑策略 | NVLink 转发摊薄 RDMA（跨机只走一次） | 纯 RDMA 单跳（节点内也走 RDMA/P2P） |
| 接收布局 | 动态：ring buffer + 前缀表，紧凑 | 静态：`[expert][rank][slot]` 预留，shape 恒定 |
| 流控单元 | channel×rank ring buffer，head/tail 批量归还 | per-expert 原子计数器 + FINISHED_SUM_TAG 双计数 |
| CPU 参与 | notify 后忙等分配 recv tensor（1 次回传） | 无（完全 GPU 自治，CUDA Graph 可捕获） |
| 量化 | 无（BF16 直传） | FP8 in-kernel（UE8M0 可选）、combine LogFMT 10bit |
| 重叠手段 | EventOverlap 双流（hook 式 0-SM 有限） | SEND/RECV 相位拆分，等待窗口 0 SM |
| 实测带宽 | 153 GB/s(8R NVL) / 51 GB/s(64EP RDMA) | 77µs dispatch / 114µs combine @128tok |
| 显存代价 | chunked ring（MB 级×channel） | 静态预留（~GB 级，§10.2） |
| 容错 | 无 mask（kernel trap on timeout） | mask buffer 屏蔽故障 rank |

### 11.2 信号与同步原语选型

| 机制 | 用在 | WHY |
|---|---|---|
| `st_release_sys` + `ld_acquire_sys` | 数据到达通知（tail/count） | release/acquire 配对建立 happens-before，sys 域跨 GPU/主机可见 |
| `st_relaxed_sys` | head 归还（槽位回收） | 与数据无依赖，省 fence；单调计数无撕裂风险 |
| `-value-1` 编码 | 一切"计数先行"信号 | 区分"信号未到"与"合法的 0"，免额外 ready 标志 |
| mapped host counter | host 拿 recv 总数 | 唯一必须 CPU 参与的点（分配 tensor），设备直写 host 内存免 cudaMemcpy |
| `bar.sync` 具名 barrier | 按 rank/channel 分组同步 | 避免 `__syncthreads` 把不齐进度的 warp 绑死 |
| IBGDA amo vs quiet+poll | 远端计数 vs 完成确认 | amo 单次往返携带增量；quiet 仅在批量边界（notify 阶段）保证请求落定 |

### 11.3 DeepEP vs 替代方案

| 维度 | DeepEP normal | DeepEP LL | NCCL all_to_all_single | 基于 NVSHMEM 自研 |
|---|---|---|---|---|
| 变长 A2A | 原生支持 | 原生支持 | 需 host 预收集 sizes | 需自建 |
| 跨机带宽 | 51 GB/s@64EP（转发架构收益） | 98 GB/s 聚合@128tok（FP8 后） | 通常 <30 GB/s（ staging+同步） | 取决实现 |
| 小批延迟 | 不适用 | 77 µs | 100 µs+（集合语义+同步） | 可逼近但工程量大 |
| 与计算重叠 | EventOverlap/双流 | SEND/RECV hook | 难（黑盒） | 可 |
| 依赖 | NVSHMEM+IBGDA（V1） | 同左 | 仅 NCCL | NVSHMEM |

## 12. 配置建议与调优指南（FlagScale/Megatron 视角）

### 12.1 推荐配置

- **训练 prefill（normal 模式）**：Config 用官方查表（`legacy.py:L233-292`）；hidden 尽量 128 对齐走 TMA；EP≤8 且单机 8 卡时用 intranode 路径（无需 NVSHMEM 注册压力）；跨机保证 `num_sms%2==0`（forwarder SM 对，`config.hpp:L59`）。`EventOverlap`：dispatch 与上一层 MoE 计算双流重叠。
- **训练/推理 decode（LL 模式）**：`use_fp8=True`（吞吐敏感再叠加 `use_ue8m0=True`+`round_scale=True` 对齐 DeepGEMM）；小 batch 用 `return_recv_hook=True` 把等待窗口让给计算；CUDA Graph 场景保持输出静态 shape + `clean_low_latency_buffer` 纳入 graph。
- **NUMA/网卡**：NVSHMEM 绑定到与 GPU 同 NUMA 的 HCA（V1 env 可用 `NVSHMEM_HCA_LIST`），否则 51 GB/s 的跨机数字会腰斩。

### 12.2 常见陷阱

1. **QP depth 不足**：`max_tokens` 调大后忘记调 `NVSHMEM_QP_DEPTH`（要求 ≥2×(max_tokens+1)），表现为 LL dispatch 偶发挂死（WQE 队列满）——先查 env 再查网络。
2. **RDMA send/recv 违例**：手工构造 Config 时打破 `send ≤ recv/2`（L50），lazy head 更新下死锁，症状是 combine 阶段全组卡住、kernel timeout 打印 `RDMA sender timeout`。
3. **cached 模式滥用**：两次 dispatch 之间 topk_idx 已变但走了 cached 路径 → 静默错位。确认条件（同 shape 同路由）后再开。
4. **combine 精度**：自定义改动若把 FP32 归约改成 BF16 直加，收敛精度下降；top-k 权重乘法在归并侧（LL）与源侧（normal）位置不同，改代码时注意归约点。
5. **超时语义**：`LEGACY_NUM_TIMEOUT_CYCLES` 是时钟周期不是秒，低功耗/降频卡上会误报；LL 收到超时先看是否真的网络故障还是 rank 掉队。
6. **V1/V2 混用**：同一进程里初始化 legacy Buffer 与 V2 ElasticBuffer 会争 NVSHMEM/NCCL Gin 资源，选一条路径。

### 12.3 配置→kernel 行为的映射表（调优时按图索骥）

| 你改的参数 | 影响的 kernel 路径 | 观测点 |
|---|---|---|
| `num_sms` (Config L25) | channel 数 = num_sms/2（intranode L211 区 / internode L493）；RDMA sender warp 数 | nsys 里 dispatch kernel 占用 SM 与时长 |
| `num_max_nvl_chunked_send_tokens` | forwarder 提前收尾阈值（internode L996-998）、NVL 批量大小 | NVL ring 等待时间（timeout 打印里的 head/tail 差） |
| `num_max_rdma_chunked_send_tokens` | coordinator 触发批量 put 的阈值（internode L809）+ head 归还阈值（L1048） | RDMA put 次数 vs 单次大小（IB 计数器） |
| `num_max_*_recv_tokens` | ring 深度 = 反压窗口（L649 等待循环） | sender 自旋占比（clock64 统计/dispatch_wait_recv_cost_stats） |
| `NVSHMEM_QP_DEPTH` | WQE 队列深度（legacy.py L113-114） | LL 挂死且无 timeout 打印 → 先查它 |
| `use_fp8/use_ue8m0` | in-kernel cast 路径（internode_ll L212-245）+ 消息尺寸（L180） | loss 曲线 + dispatch 带宽 |
| `return_recv_hook` | SEND/RECV 相位拆分（buffer.hpp L1577/L1592） | 0-SM 窗口长度（nsys 时间线上的空档） |
| `use_logfmt` | combine LogFMT decode（internode_ll L700-711 区） | combine 带宽 ↑ 精度曲线 |
| `num_worst_tokens` | 尾部清理量（intranode L538-546 / internode L1199-1212） | 预分配显存峰值 |

### 12.4 调优顺序（从cheap到expensive）

1. 查表 Config 直接用（已覆盖 2-160 rank）；2. 打开 FP8/LogFMT 看端到端 loss 曲线（LL）；3. 微调 `num_max_*_chunked_send_tokens`（受 §5.1 约束链保护，改小安全、改大先想 lazy head）；4. hook/EventOverlap 与计算重叠参数（看 nsys 里 A2A 窗口是否被计算盖住）；5. 最后才是 QP 数/NVSHMEM env 级调优。
