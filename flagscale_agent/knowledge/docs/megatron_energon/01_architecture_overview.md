# 第1章：Megatron-Energon 架构总览与数据管道设计

## 1. 概述与设计动机

Megatron-Energon 是 Megatron-LM 的多模态数据加载库，解决大规模分布式训练中的数据加载问题：
- **大规模数据分片**：将数据按 rank 和 worker 精确切分，避免重复或遗漏
- **断点恢复**：训练中断后从精确位置恢复，不丢失/重复样本
- **多数据集混合**：灵活地将不同来源、不同模态的数据按权重混合
- **可组合管道**：通过 wrapper 链式组合实现 shuffle → blend → batch → pack 等操作

核心设计思想：**可保存的迭代器模式**（Savable Iterator Pattern）。每个数据集节点实现 `save_state()` / `restore_state()` 接口，使得整条管道可以在任意时刻被序列化/恢复。

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| WorkerConfig | `worker.py` | 288 | 分布式 worker 配置与生命周期管理 |
| SavableDataLoader | `savable_loader.py` | 1411 | 可保存状态的 DataLoader（训练用） |
| BasicDataLoader | `savable_loader.py:L1204` | ~170 | 不保存状态的 DataLoader（验证用） |
| loader.py | `loader.py` | 119 | 工厂函数 get_loader / get_savable_loader |
| SavableDataset | `flavors/base_dataset.py` | 485 | 所有数据集的抽象基类 |
| Sample | `flavors/base_dataset.py:L114` | ~90 | 样本基类（含 __key__, __restore_key__） |
| WorkerRng | `rng.py` | 90 | Worker 级确定性随机数生成器 |
| SystemRng | `rng.py:L129` | 44 | 全局 RNG 状态管理（torch/numpy/random） |

## 3. 架构总览

### 3.1 整体分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop (Megatron-LM)                    │
│  注：Megatron侧通过 broadcast_data() 将batch广播到TP组,           │
│  并用 pre_process/post_process 控制PP首/末stage的数据获取          │
├─────────────────────────────────────────────────────────────────┤
│  SavableDataLoader / BasicDataLoader                             │
│    ├── checkpoint管理 (定时保存 worker 状态)                      │
│    ├── worker通信 (cmd_queue / result_queue)                      │
│    └── epoch迭代 + sample计数                                    │
├─────────────────────────────────────────────────────────────────┤
│  Wrappers Pipeline (可组合的数据变换链)                           │
│    BatchDataset → PackingDataset → ShuffleBuffer → BlendDataset  │
│    → MapDataset → FilterDataset → ...                            │
├─────────────────────────────────────────────────────────────────┤
│  TaskEncoder (用户自定义的编码逻辑)                               │
│    get_train_dataset() → 构建完整管道                             │
├─────────────────────────────────────────────────────────────────┤
│  Metadataset (多数据集混合与路由)                                 │
│    MetadatasetV2 → BlendDataset(weights=[...])                   │
├─────────────────────────────────────────────────────────────────┤
│  Flavors / Data Sources (底层数据读取)                            │
│    WebDataset (.tar shards) / JSONL / Joined                     │
│    注：数据按 DP rank 分片, 每个 DP rank 只读自己的 shard 子集     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心数据流（含并行策略集成）

```
User Dataset (.tar shards / .jsonl)
    │
    ▼
StandardWebdatasetFactory.build()  ──→  构造 SavableDataset
    │                                    (按 DP world_size 分 shard)
    ▼
TaskEncoder.build_train_datasets()  ──→  包装为训练管道
    │                                    (shuffle → encode → batch)
    ▼
get_train_dataset()  ──→  BlendDataset(多源混合) + BatchDataset
    │
    ▼
get_savable_loader(dataset)
    │
    ▼
SavableDataLoader(DataLoader)  ──→  fork workers, 每个 worker 独立迭代
    │
    ▼
Megatron get_batch():
    │  ── PP: 仅 pre_process stage 从 loader 取数据
    │  ── TP: broadcast_data() 将 batch 从 tp_rank=0 广播到 TP 组
    │  ── DP: 各 DP rank 的 loader 独立产出不重叠数据（shard级切分）
    ▼
Training Loop: for batch in loader: ...
```

**并行集成说明**：
- **DP (Data Parallel)**：Energon 的 `WorkerConfig(rank, world_size)` 直接对应 DP rank/size。底层 WebDataset 的 shard 分配（sharder.py）确保各 DP rank 读取不重叠的 shard 子集。
- **TP (Tensor Parallel)**：Energon 不感知 TP。Megatron 侧通过 `broadcast_data()` (megatron/core/tensor_parallel/data.py) 将 TP rank=0 获取的 batch 广播到整个 TP 组。
- **PP (Pipeline Parallel)**：Energon 不感知 PP。Megatron 侧仅在 `pre_process=True` 的 PP stage（第一个 stage）调用 data loader，其余 stage 通过流水线接收激活值。


## 4. WorkerConfig — 分布式 Worker 配置

### 4.1 设计动机（WHY）

分布式训练中，每个 rank 可能有多个 data worker 进程。WorkerConfig 是全局单例配置，确保：
1. 每个 worker 知道自己的全局 ID → 决定读取哪些 shard
2. Seed 确定性 → 保证 shuffle/blend 结果可复现
3. Sample index 追踪 → 支持断点恢复时跳过已处理样本

### 4.2 核心字段与语义（worker.py:L23-84）

```python
@dataclass(slots=True, kw_only=True, eq=False)
class WorkerConfig:
    rank: int           # DP rank (0 ~ world_size-1)
    world_size: int     # DP world_size
    num_workers: int    # 每个 rank 的 worker 数量（0=主进程加载）
    
    data_parallel_group: Optional[ProcessGroup] = None  # 非全量 DP 时的通信组
    seed_offset: int = 0  # 用于：1) 数据集 shuffle seed  2) worker RNG seed
    
    # ClassVar: 跨 worker 共享的运行时状态
    worker_id_offset: ClassVar[int] = 0  # 恢复 checkpoint 时的 worker 轮转偏移
    _sample_index_stack: ClassVar[Optional[List[int]]] = None  # 当前迭代位置
    active_worker_config: ClassVar[Optional["WorkerConfig"]] = None  # 激活标记
```

### 4.3 Worker ID 计算逻辑（worker.py:L175-196）

```python
def rank_worker_id(self) -> int:
    """当前 worker 在 rank 内的逻辑 ID"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return self.worker_id_offset  # 主进程模式
    # 关键：左旋转逻辑 worker id，确保恢复后第一个物理 worker
    # 对应应该发射下一个 sample 的逻辑 worker
    return (worker_info.id + self.worker_id_offset) % worker_info.num_workers

def global_worker_id(self) -> int:
    """全局唯一 worker ID = rank * num_workers + local_id"""
    return self.rank * max(self.num_workers, 1) + self.rank_worker_id()
```

**设计要点**：`worker_id_offset` 实现了 worker 轮转。恢复 checkpoint 时，如果上次最后一个 sample 来自 worker_id=2，那么恢复后应从 worker_id=3 开始。通过设置 `worker_id_offset=3`，物理 worker 0 被映射到逻辑 worker 3。

### 4.4 Worker 生命周期管理（worker.py:L86-121）

```
worker_activate(sample_index)    ←─ 每次取下一个 sample 前调用
    │  设置 _sample_index_stack, active_worker_config
    ▼
  dataset.__next__()             ←─ 迭代产出一个 sample
    │
    ▼
worker_deactivate()              ←─ sample 产出后调用
    │  清空 _sample_index_stack, active_worker_config
    ▼
  yield sample to DataLoader
```

这种 activate/deactivate 模式确保 dataset 内部可以通过 `WorkerConfig.active_worker_config` 获取当前 worker 上下文（如 sample_index），而不需要显式传参。

## 5. SavableDataLoader — 可保存的 DataLoader

### 5.1 设计动机（WHY）

PyTorch DataLoader 不支持保存/恢复迭代状态。在大规模训练中（千步以上），如果训练中断，需要从精确的数据位置恢复，否则：
- 重复已见数据 → 影响收敛
- 跳过未见数据 → 浪费数据

SavableDataLoader 通过「周期性内部 checkpoint + 快进」实现精确恢复。

### 5.2 类层次结构

```
torch.utils.data.DataLoader
    └── SavableDataLoader  (训练用, 支持 save/restore)
    └── BasicDataLoader    (验证用, 无 save 能力)
```

### 5.3 初始化流程（savable_loader.py:L660-798）

```python
SavableDataLoader.__init__(dataset, checkpoint_every_sec=60, ...):
    # 1. 包装 Watchdog（检测 worker 卡死）
    dataset = WatchdogDataset(dataset, timeout_seconds=60)
    
    # 2. 包装 GC（定期垃圾回收，避免大 tensor OOM）
    dataset = GcDataset(dataset, every_n_iter=gc_collect_every_n_steps)
    
    # 3. 创建 worker 通信队列
    cmd_queues = [Queue() for _ in range(num_workers)]
    result_queues = [Queue() for _ in range(num_workers)]
    
    # 4. 包装为 SavableDatasetWrapper（多 worker）或 SimpleSavableDatasetWrapper
    if num_workers > 0:
        dataset = SavableDatasetWrapper(dataset, checkpoint_every_sec=60, ...)
    else:
        dataset = SimpleSavableDatasetWrapper(dataset, ...)
    
    # 5. 调用父类 DataLoader.__init__
    super().__init__(dataset, batch_size=None, shuffle=False,
                     num_workers=num_workers, pin_memory=True,
                     multiprocessing_context="fork", persistent_workers=True)
```

### 5.4 Checkpoint 机制时序图

```
Worker Process                    Main Process (SavableDataLoader)
    │                                     │
    │  ── produce sample ──────────────→  │  _epoch_iter(): yield sample
    │  ── produce sample ──────────────→  │
    │                                     │
    │  [每 checkpoint_every_sec 秒]        │
    │  _store_checkpoint():               │
    │    save dataset_state + rng_state   │
    │    保留最近 n_checkpoints 个         │
    │                                     │
    │                                     │  save_state_rank():
    │  ←── cmd: "get_checkpoint" ──────   │    通过 cmd_queue 发命令
    │  ──→ result: checkpoint data ─────→ │    从 result_queue 收结果
    │                                     │    返回 SavableDataLoaderState
    │                                     │
    │         [训练中断, 重启后]            │
    │                                     │  restore_state_rank(state):
    │  ←── restore dataset_state ──────   │    恢复 dataset 到 checkpoint 位置
    │  skip N samples (offset) ─────────→ │    快进跳过 checkpoint 后的样本
    │  ── resume normal production ─────→ │
```

### 5.5 Save/Restore 状态结构（savable_loader.py:L156-195）

```python
@edataclass
class SavableDatasetState(State):
    rng: SystemRngState       # torch/numpy/random 全局 RNG 状态
    dataset_state: FlexState  # 递归保存整条 wrapper 链的状态
    sample_index: int         # 下一个要产出的 sample index

@edataclass
class SavableDatasetCheckpoint:
    state: Optional[SavableDatasetState]  # checkpoint 时刻的状态
    offset: int                            # 从 checkpoint 到当前位置的偏移
```

恢复逻辑：恢复到最近的 checkpoint state，然后快进（skip）offset 个样本。

## 6. 与 Megatron 并行策略的集成接口

Energon 本身只负责 DP 维度的数据切分。TP/PP 的数据分发由 Megatron 训练循环负责：

### 6.1 DP — 数据并行（Energon 直接处理）

WorkerConfig 的 `rank` 和 `world_size` 对应 DP rank/size。WebDataset sharder（sharder.py）按照 `global_worker_id` 将 shard 分配给不同 worker，确保各 DP rank 读取不重叠的数据。

### 6.2 TP — 张量并行（Megatron 侧 broadcast_data）

Energon 不感知 TP。在 Megatron 的 get_batch() 中，仅 TP rank=0 的进程从 DataLoader 获取数据，然后通过 `broadcast_data()` (megatron/core/tensor_parallel/data.py) 广播到整个 TP 组的其他 rank。

### 6.3 PP — 流水线并行（Megatron 侧 pre/post_process）

Energon 不感知 PP。Megatron 仅在 PP 第一个 stage（`pre_process=True`）调用 DataLoader 获取输入数据，最后一个 stage（`post_process=True`）计算 loss。中间 stage 通过 P2P 通信接收/发送激活值，不直接与数据加载交互。


## 7. RNG 确定性系统

### 7.1 两级 RNG 设计（rng.py）

| 级别 | 类 | 作用域 | 用途 |
|------|----|----|------|
| Worker级 | `WorkerRng` | 每个 dataset 实例 | shuffle buffer, blend 选择 |
| 全局级 | `SystemRng` | 整个 worker 进程 | 用户代码 (TaskEncoder 中的增强) |

**WHY 分两级？** Worker RNG 基于 `worker_seed()`（由 rank + worker_id + seed_offset 确定性计算），保证数据管道的 shuffle/blend 在恢复后完全一致。而 SystemRng 影响用户代码中的随机增强，需要独立保存/恢复。

### 7.2 Seed 计算（worker.py:L220-258）

```python
def worker_seed(self, worker_id: Optional[int] = None) -> int:
    """基于 rank, worker_id, seed_offset 确定性生成 seed"""
    return SystemRng.get_seed_from_args(
        "energon",
        self.rank,
        worker_id if worker_id is not None else self.rank_worker_id(),
        self.seed_offset,
    )
    # 内部用 SHA1 哈希 → 取前 4 字节 → 大端整数
```

### 7.3 WorkerRng 自定义 multinomial（rng.py:L54-66）

```python
def choice_idx(self, probs: torch.Tensor) -> int:
    # 不用 torch.multinomial（2.7.0 改变了实现，破坏了跨版本确定性）
    # 改用 CDF + searchsorted，保证任何 PyTorch 版本结果一致
    cdf = torch.cumsum(probs, dim=0)
    val = torch.rand(1, generator=self.rng) * cdf[-1]
    return torch.searchsorted(cdf, val).item()
```

## 8. Sample 基类与 Restore Key

### 8.1 Sample 数据类（base_dataset.py:L114-155）

```python
@edataclass
class Sample(ABC, PinMemoryMixin, ExtendableDataclassMixin):
    __key__: str          # 样本唯一 ID（如 "shard-00001/000042"）
    __restore_key__: Tuple[Union[str, int, tuple], ...]  # 恢复路径
    __subflavors__: Optional[Dict[str, Any]] = None      # 子类型标记
    __sources__: Optional[tuple[SourceInfo, ...]] = None  # 来源追踪
```

`__restore_key__` 是一个嵌套 tuple，记录从 DataLoader → Wrapper 链 → 底层 Dataset 的完整路径，用于在训练出错时定位并重现问题样本。

### 8.2 PinMemoryMixin（base_dataset.py:L41-69）

递归地对嵌套结构（dict/dataclass/namedtuple/list/tuple）中的所有 Tensor 调用 `pin_memory()`。这是 DataLoader 的 `pin_memory=True` 所需的接口。

## 9. 完整 get_batch 流程（Energon + Megatron 并行集成）

Energon 数据管道产出 batch 后，Megatron 训练循环中的 `get_batch()` 负责将数据正确分发到所有并行 rank：

```python
# 伪代码：Megatron get_batch() 的并行感知逻辑
def get_batch(data_iterator):
    # PP 保护：仅第一个 pipeline stage 从 loader 取数据
    if mpu.is_pipeline_first_stage():  # pre_process=True
        # TP 广播：仅 TP rank=0 实际调用 data_iterator
        if mpu.get_tensor_model_parallel_rank() == 0:
            batch = next(data_iterator)  # ← Energon SavableDataLoader 产出
        else:
            batch = None
        # broadcast_data() 将 batch 从 tp_rank=0 广播到 TP 组所有 rank
        batch = broadcast_data(keys, batch, datatype=torch.int64)
    else:
        # PP 非首 stage：不从 loader 取数据，通过流水线接收激活值
        batch = None
    return batch
```

### 9.1 DP 数据切分（Energon 直接处理）

WorkerConfig 的 `rank`/`world_size` 对应 DP rank/size。Sharder（sharder.py）按 `global_worker_id` 分配 shard，各 DP rank 读取不重叠数据。

### 9.2 TP broadcast_data（Megatron megatron/core/tensor_parallel/data.py）

TP 组内仅 rank=0 持有数据，`broadcast_data()` 广播 tensor dict 到组内所有 rank，避免重复 IO。

### 9.3 PP pre_process/post_process 守卫（Megatron pipeline schedule）

PP 第一 stage（`pre_process=True`）执行 embedding + 从 loader 取数据；最后一 stage（`post_process=True`）执行 output layer + loss 计算。中间 stage 仅做 transformer 层计算，不与数据加载交互。

## 10. 设计决策对比

| 维度 | SavableDataLoader | 标准 DataLoader | 选择理由 |
|------|-------------------|-----------------|----------|
| 状态保存 | 周期性 checkpoint + offset 快进 | 不支持 | 大规模训练必须支持恢复 |
| Worker 通信 | cmd_queue/result_queue 双向通信 | 仅单向产出 | 需要在运行时获取 worker 状态 |
| GC 策略 | 显式定期 gc.collect + gc.freeze | Python 默认 GC | 大 tensor 场景避免 OOM |
| Watchdog | 超时检测 + 打印 stack trace | 无 | 生产环境必须检测 worker 卡死 |
| 多进程模式 | fork + persistent_workers | spawn/fork | fork 避免重新加载模型；persistent 避免重建 |

| 维度 | Energon | WebDataset (原生) | 选择理由 |
|------|---------|-------------------|----------|
| Shard 分配 | 确定性 hash-based per-worker | 随机 | 恢复后需保证相同分配 |
| 数据混合 | BlendDataset(可保存权重状态) | 无内置 | 多数据集训练是核心场景 |
| 格式支持 | WebDataset + JSONL + Joined | 仅 WebDataset | 多模态需要多种格式 |

## 11. 配置建议与调优指南

- `num_workers`: 推荐 4-8。过多会增加 fork 内存开销和 checkpoint 时间
- `checkpoint_every_sec`: 默认 60s。频率越高恢复越快，但运行时 overhead 越大
- `prefetch_factor`: 默认 2。增大可隐藏 IO 延迟，但增加内存和 checkpoint 数量
- `n_checkpoints`: 默认 `prefetch_factor * num_workers + 1`。必须 ≥ 此值否则恢复可能失败
- `seed_offset`: 每次训练运行时改变此值可获得不同的数据顺序
- Micro batch size 变更：恢复时允许缩小（必须整除旧值），不允许增大
