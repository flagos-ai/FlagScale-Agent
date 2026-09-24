# Kimi K3 / K2.5 训练基础设施与配方走读（技术报告一手提取）

> 来源：Kimi K3 技术报告 arXiv:2607.24653（MoonshotAI/Kimi-K3 仓库内 k3_tech_report.pdf，同文）§2.3/§3/§5；
> Kimi K2.5 技术报告（MoonshotAI/Kimi-K2.5 仓库内 tech_report.pdf，无 arXiv）。
> 提取时间 2026-09-13，数据为报告原文数值。机制综述见 `01_moe_pretrain_perf_survey.md`。

## 1. K3 vs K2 架构对照（Table 1 原文数据）

| 项 | K2 | K3 | 变化 |
|---|---|---|---|
| 层数 | 61 | 93 | +52% |
| 总参/激活 | 1.04T / 32.6B | 2.78T / 104.2B | +167% / +220% |
| hidden dim | 7168 | 7168 | 持平 |
| Latent MoE 维度 | – | 3584 (0.5×) | 新 |
| 每专家 MoE hidden | 2048 | 3072 | +50% |
| 路由专家/激活数 | 384 / 8 | 896 / 16 | +133% / ×2 |
| 共享专家 | 1 | 2 | ×2 |
| 注意力头 | 64 | 96 | +50% |
| 训练上下文 | 128K | 1M | 8× |
| 注意力 | MLA | KDA 69 层 + MLA 24 层混合 | 新 |
| 激活函数 | SwiGLU | SiTU-GLU | 新 |
| ViT | – | 401M / 27 层 / patch 14 | 原生多模态 |

## 2. 训练配方（§3.2–§3.4）

- **Scaling law 重标定**：架构+数据+训练改进合计带来 **2.5× scaling 效率**（vs K2，OOD 验证集拟合）。
  batch size / lr / TPP / 模型形状全部重新搜索。
- **cosine vs WSD 的方法论结论**：两调度各自最优超参差异巨大；共享超参对比会系统性偏袒其一。
  各自独立搜索后 cosine 稳定更低 final loss → 采用 cosine（1% linear warmup，weight decay 0.1）。
- **优化器**：Per-Head Muon（§2.5）+ K2 式 weight clipping；MoE 负载均衡用 **Quantile Balancing**。
- **上下文课程**：预训练 8K→64K，cooldown 阶段 256K→1M（四阶段）；长文档/视频上采样 +
  人工拼接跨 1M 上下文的合成任务（防止注意力退化为局部模式）。
- **NoPE**：无显式位置编码，位置信息由 KDA 门控/衰减隐式携带，直接外推 1M（无 RoPE 缩放/插值）。
- **原生多模态**：视觉-文本从头联合训练（非后挂对齐），单一 next-token 目标内交错。

## 3. MoonEP：完美均衡的专家并行（§5.2.1，开源 github.com/MoonshotAI/MoonEP）

问题：常规 EP 下 token 负载跨 rank 不均 → 算力浪费 + 动态 shape 造成显存碎片。

- **完美均衡定义**：每 rank 恰好收到 S×K tokens（S 序列长、K 每token专家数），所有 rank 计算量相同。
- **冗余专家上界（可证明）**：E 专家 / R EP度时，平衡方案总存在且 **每 rank ≤ E/R 个冗余专家**（证明在附录E，
  上界 essentially tight）。每 rank 预留 E/R 冗余槽 → 规划永远可行，训练永不中断。
  对比 ECHO/UltraEP：预设冗余数或 per-rank token cap，无可行解时被迫停训且需手调。
- **在线规划**：每 micro-batch 每层从 router 输出规划冗余专家并预取；离线 ILP 求精确解作参照，
  线上用 GPU 规划 kernel（近优、开销可忽略、恒满足 E/R 上界）。backward 梯度先暂存本地 reduce buffer，
  完成后归约回 home rank。
- **零拷贝通信**：fused permute/unpermute，规划 kernel 预计算每个 token 目的地，直接发送到远端
  expert-grouped 位置，通信 buffer 视图直接返回计算。最坏不均衡下 DeepEP 需 S×K×R buffer，MoonEP 仅需固定 S×K。
- **静态 shape 无同步**：每 rank 恰好 S×K → 所有层计算 shape 编译期已知，消除逐层 host-device 同步
  与 kernel launch 间隙。
- **Expert-GEMM 感知调度**：rank 内每专家 token 数仍有偏斜 → workload-aware 调度器在 launch 前
  依解析成本模型自适应参数（系数离线 autotune）；共享专家 GEMM 放独立 stream 重叠。

## 4. 显存效率（§5.2.2）

- **统一激活管理器**：backward 所需张量绑定可插拔存储后端；重计算/量化/offload/remote-offload 只是
  存储策略，张量粒度自由组合，注解声明与模型代码解耦。全部 GPU 内存在主计算流单池管理（避免多流碎片）。
  K3 实际配置：多数激活 block-wise FP8 量化 + (remote-)offload；逐元素算子用重计算。
- **Memory-efficient MoE**（受 SonicMoE 启发）：permuted probs 的梯度改写为只依赖中间激活与上游梯度
  （消除对 forward output 的反向依赖）；group GEMM 只存 dispatch 输入，backward 重算 dispatch，
  其通信与 group-GEMM backward 部分重叠。
- **AttnRes 分块共享**：块表示在边界层生成一次、后续层共享驻留 GPU；整个 AttnRes 计算包 checkpoint，
  每层 backward 存活激活与标准残差架构相同。PP 通信用 cache-based 增量传输（只传新块），
  达到显存理论下界。
- **PP rank 间激活均衡**：interleaved 1F1B warmup 导致激活分布不均 → Mooncake Transfer Engine
  远程 offload 到其他 PP rank 的显存。
- **Pipeline ZeRO-2**：梯度按 DP rank 分片并驻留 CPU（GPU 保留 double grad buffer）。
- **P2P Muon 正交化**：Newton–Schulz 需要完整参数矩阵；不做全参数 all-gather，而是每 rank 通过 P2P
  只取本地拥有的参数分片（消除全参数 buffer），通信与计算按 model-chunk 粒度流水。

## 5. 多模态编码器（§5.2.3）

- **动态 CP**：大图/长视频沿 patch 维跨设备分区，gather-KV 计算注意力；CP 组内再分 sub-CP 组
  负载均衡多张大图 → 编码时延与跨设备不均衡同降，剩余编码计算藏进 pipeline 气泡。
- **ViT 藏进 PP 气泡**（K2.5 引入 DEP，K3 进一步分解）：1F1B 下文本 forward 集中在头、backward
  集中在尾；ViT forward 同步执行头几个 micro-batch，其余排进气泡，backward 对称处理
  → 视觉编码器有效开销基本消除。

## 6. 对 FlagScale/Megatron 用户的映射提示

- MoonEP 的 E/R 冗余上界与零拷贝思路可对照 Megatron `expert_model_parallel_size` 下的容量规划；
  FlagScale 暂无等价配置，模型移植时 `moe_grouped_gemm` 静态 shape 假设与 MoonEP 的"完美均衡"
  是同族问题的不同解（一个容忍偏斜+融合 kernel，一个消灭偏斜+静态 shape）。
- QB/MoonEP/SiTU-GLU 均为 2026 论文级新机制， Megatron-LM-FL 尚未内置；移植 K3 类模型时
  需评估 router 替换（quantile 直解 bias）与激活函数替换（SiTU-GLU）的实现成本。
- K3 上下文课程（8K→64K→256K→1M）可作为 FlagScale 长上下文训练 phase 划分的参照模板。
