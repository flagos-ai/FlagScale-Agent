---
name: train-moe-perf
description: Select and tune MoE pretraining performance optimizations in FlagScale
  (Megatron backend). Covers architecture choices (fine-grained/shared experts, SCMoE,
  layer re-configuration), routing and load balancing (aux loss vs auxiliary-loss-free
  expert bias, group-limited routing), expert parallelism sizing, all-to-all
  communication overlap (fb_overlap/shared_expert_overlap/EP comm overlap, DeepEP,
  FP8 comm), kernels (grouped GEMM, permute fusion, FP8), stability (fp32 routing,
  z-loss), and evaluation (MFU, max/mean load, A2A share). Use when configuring,
  debugging, or accelerating MoE training (throughput, load imbalance, all-to-all
  bottleneck, expert stragglers, loss spikes in MoE runs).
---

# MoE 预训练性能优化指南 (FlagScale / Megatron-LM-FL)

> 定位：**性能调优工作流**。机制/源码细节见 knowledge：`know-moe-training`（调研全表 +
> K3/K2.5 训练基础设施走读 `moe_training/02_kimi_k3_k25_training_infra.md`）、
> `know-megatron-parallel`（EP 进程组与通信）、`megatron_lm_fl/05_expert_parallelism.md`（源码走读）。

## 0. 快速诊断路径

吞吐低于预期时，按顺序查四件事（对应四大瓶颈）：

1. **A2A 时间占比**：profiler 看 dispatch/combine all-to-all 占比。>15% → 通信章(§3)。
2. **max/mean 专家负载比**：router stats。>1.5 → 均衡章(§2)。
3. **专家 GEMM 效率**：kernel 分析。细长小 GEMM、低 MFU → 内核章(§4)。
4. **显存溢出/OOM**：EP 度不足或激活过高 → 并行章(§1)。

## 1. 并行拓扑先行

- EP 度 = 专家权重显存 ÷ 单卡可用显存（向上取整），再对齐网络拓扑：**专家放置让高频共现
  的专家组落在 NVLink 域内**（group-limited routing 与之配套）。
- 专家跨节点是 A2A 变慢的第一原因：优先增大 `expert_model_parallel_size` 使每节点专家
  完整，而不是加大跨节点 EP。
- DP attention + EP MoE 是标准混合形态；EP 度超单层卡数时考虑分层 EP（X-MoE 类）。
- 用资源模型预筛（Piper 类：记忆/算力/通信解析模型），别全量试错。

## 2. 路由与负载均衡

| 场景 | 推荐 | 配置 |
|---|---|---|
| 默认/大规模 | aux-free expert bias（sigmoid 打分） | `moe_router_score_function: sigmoid` + `moe_router_enable_expert_bias: true` + `moe_router_bias_update_rate: 0.001` |
| 小模型/快速实验 | aux loss（有梯度） | `moe_router_load_balancing_type: aux_loss` + `moe_aux_loss_coeff: 0.01~0.05` |
| DSv3 式兜底 | bias 主力 + 极小 seq aux | 同时开，coeff 0.02 |
| 数值不稳 | z-loss | `moe_z_loss_coeff: 0.001` |
| 通信受限 | group-limited routing | `moe_router_num_groups: 8, moe_router_group_topk: 4` |

- **打分全程 fp32**（`moe_router_dtype: fp32`）——bf16 sigmoid/softmax 会下溢，这是
  MoE loss spike 的常见根因。
- 偏置只改“选谁”不进前向数值：排查均衡问题先看 `moe_stats`/router replay 的专家分布，
  不要先怀疑训练超参。
- aux loss 过强会损害质量：看验证损失 vs 均衡指标（max/mean 比）的联合曲线，不是单看
  aux loss 值。

## 3. 通信优化（最大杠杆）

按性价比顺序开（DSv3 配置全部启用）：

1. `moe_fb_overlap: true` —— dispatch 与上层计算重叠
2. `moe_shared_expert_overlap: true` —— 共享专家旁路与主路 A2A 重叠
3. `overlap_moe_expert_parallel_comm: true` —— EP 通信计算重叠
4. `moe_token_dispatcher_type: alltoall` —— 变长 A2A dispatcher
5. FP8 通信（激活 dispatch 压 FP8，DeepEP/DSv3 同款）：通信量减半
6. 两跳 dispatch 仅当层级带宽比 > breakeven（TerraceMoE 成本模型）才开

通信库选型：单用 Megatron dispatcher 不足时上 DeepEP（NVSHMEM/IBGDA）或 NCCL EP。

## 4. 内核与显存

- `moe_grouped_gemm: true`（所有专家 FC1/FC2 一次 kernel）；
- permute/prob 加权融合（`moe_permute_fusion`，如可用）；
- FP8 GEMM（TE）+ bf16 主权重 + fp32 优化器状态；
- 激活重计算 + SP 省激活；671B 级可试 MXFP4 激活+通信（+12.5% 吞吐，671B 验证不伤收敛）。

## 5. 验收指标

| 指标 | 目标 | 工具 |
|---|---|---|
| tokens/s/GPU、MFU | 与同规模公开实现相当 | 训练日志 |
| A2A 时间占比 | <15%（越低越好） | nsys/profiler |
| max/mean 专家负载比 | <1.5 | router stats |
| 验证损失/下游 | 与优化前持平 | eval pipeline |
| loss spike | 无不可恢复 spike | 训练监控 |

先在小规模（单机）验证三项指标齐全，再扩规模；每次只改一类优化，A/B 对照。

## 6. 常见坑

1. **忘了 fp32 路由打分** → 周期性 loss spike / 均衡失效。
2. **expert_bias 配了但 score_function 不是 sigmoid** → bias 分支根本不走。
3. **overlap 三项开了但 EP 组太小** → 收益被小通信量稀释，先扩 EP。
4. **容量因子>1 丢 token** → dropless 为默认；除非复现老系统，不要开 token dropping。
5. **跨节点专家放置随机** → A2A 走慢速 fabric；放置必须对齐拓扑（topo-detect）。
6. **默认 group-limited routing** → 2026 旗舰并非都用：GLM-5 744B 用 `moe_router_num_groups=1`（不分组）；是否分组取决于集群网络层级。
7. **忽略架构对并行的反向约束** → GLM-5 DSA indexer 缓存要求 PP stage 首层为 full 层（`decoder_first_pipeline_num_layers=2`）；线性注意力/混合注意力层频会改变层规格（`experimental_attention_variant`）。
8. **MTP 误当发布必需** → GLM-5 发布 checkpoint 无 MTP 层（`mtp_num_layers=0`），训练期辅助信号与发布形态解耦。

## 7. 2026 最新系列参照模板（一手配置）

新模型选型/调参直接对照官方 examples 配置：

| 系列 | 配置路径 | 关键特征 |
|---|---|---|
| GLM-5 744B-A40B | `examples/glm5/conf/train/744b_a40b.yaml` | MLA+DSA（indexer topk 2048）、sigmoid+expert-bias、无分组、TP4×PP4×EP8。**技术报告 arXiv:2602.15763**；层规则 `2*full+repeat(full,shared,shared,shared)`=IndexCache [2603.12201]（Full 层跑 indexer、Shared 层复用其 top-k 索引）；系列已至 GLM-5.3/5.3-Flash(320B-A18B) |
| Qwen3.5 35B-A3B | `examples/qwen35/conf/train/35b_a3b.yaml` | Gated DeltaNet 混合（freq=4）、global_aux_loss@1e-3、shared_expert_gate、permute_fusion。**无 arXiv TR**，官方博客 qwen.ai/blog?id=qwen3.5；系列已至 Qwen3.6-35B-A3B / Qwen3.8 |
| Kimi K3 2.78T-A104B | **技术报告 arXiv:2607.24653**（MoonshotAI/Kimi-K3 仓库内同文 PDF；训练基础设施细节见 knowledge `moe_training/02_kimi_k3_k25_training_infra.md`） | 896 专家×16 激活 latent 3584+2 共享、KDA 69 层+MLA 24 层混合、Quantile Balancing（分位数直解 bias）、MoonEP（E/R 冗余上界+零拷贝+静态 shape）、SiTU-GLU、NoPE 1M ctx |
| Kimi K2.5 | **技术报告** MoonshotAI/Kimi-K2.5 仓库内 tech_report.pdf（无 arXiv） | 基座复用 K2（1.04T/32B 激活/384×8）；多模态三阶段 1T ViT→15T joint→500B 长上下文；DEP 编码器-流水线解耦 |
| DeepSeek-V4 1.6T-A49B | 内部架构走读笔记（**无公开报告**，org 无 V4 仓库，非公开文献） | CSA+HCA 压缩分层、mHC 残差、1M 上下文 |
| DeepSeek-V3（基线） | `examples/deepseek_v3/` | group-limited(8/4)、sigmoid+bias+小系数 aux、overlap 三件套 |

机制细节与完整调研见 knowledge `know-moe-training`（`moe_training/01_moe_pretrain_perf_survey.md`）。
