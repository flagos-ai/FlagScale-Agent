# MoE 预训练性能优化调研

> 调研范围：Mixture-of-Experts (MoE) 大模型预训练的系统与算法协同性能优化。
> 整理时间：2026-09（引用核验 2026-09-13）。文献来源：arXiv（GShard/Switch/Mixtral/DeepSeek 系列/Tutel/Megatron-MoE 及 2025-2026 最新系统论文）、Megatron-LM / FlagScale 配置实践、各官方技术报告（GLM-5 TR arXiv:2602.15763；Kimi K3/K2.5 官方仓库 PDF；Qwen3.5/3.6/3.8 官方博客；DeepSeek-V4 暂无公开报告）。
> 配套实现级细节见内部知识库：`know-megatron-parallel`（EP 进程组/通信模式）、`megatron_lm_fl/05_expert_parallelism.md`（MoELayer/TopKRouter 源码走读 + token 级全流程数据流）。

## 1. 背景与动机

MoE 用稀疏激活换模型容量：总参数 N_total 远大于每 token 激活参数 N_act（典型激活比 1/16 ~ 1/25），使预训练能在固定算力预算下逼近更密集模型的损失（Switch Transformer 报告同算力下预训练速度提升最高 7 倍；DeepSeek-V3 671B-A37B 全程 2.788M H800 GPU 时，仅相当于一个 70B 级密集模型的开销）。2026 年最新旗舰全部采用 MoE：DeepSeek-V4（1.6T 总参/49B 激活，暂无公开报告）、GLM-5（744B-A40B，arXiv:2602.15763）、Qwen3.5-35B-A3B（官方博客，系列已至 Qwen3.8）、Kimi K3（2.78T-A104B，896 路由专家激活 16，官方 TR）——MoE 已是前沿预训练的默认架构，其性能优化方法也随之快速演化（详见 §2.3、§2.4）。

但 MoE 的性能优势不是免费的——预训练吞吐受四大瓶颈制约：

1. **All-to-All 通信**：专家并行 (EP) 下 dispatch/combine 两个 all-to-all 占端到端训练时间的显著比例（CE-MoE 论文与 UBEP 均以此为主要动机）。
2. **负载不均**：动态路由使部分专家/设备成为 straggler，同步训练下整卡等待最重负载（"木桶效应"）。
3. **显存压力**：专家权重总参数巨大（即使激活稀疏也要全驻留），激活在 permute 时复制 topk 倍。
4. **内核效率**：专家级小 GEMM 呈"细长"形状、算术强度低；token 按 expert 重排 (permute/unpermute) 引入额外访存与 kernel 启动开销。

MoE 预训练性能优化就是围绕这四点的系统+算法协同设计。本文按"架构 → 路由与均衡 → 并行 → 通信 → 内核与显存 → 稳定性 → 评测与成本 → 配置实践"组织。

## 2. 架构层优化：让专家"该细的细、该共享的共享"

### 2.1 细粒度专家 + 共享专家隔离 (DeepSeekMoE)

- **细粒度**：把传统粗粒度专家的 FFN 中间维切细、数量加倍（如 256 个细专家），组合自由度指数级提升，同一 topk 预算下知识组合更灵活。
- **共享专家**：设少量所有 token 必经的共享专家，承担通用知识，让路由专家只学专属知识，降低冗余；共享专家旁路可与主路 all-to-all 重叠执行（见 §5.3）。
- DeepSeek-V3 配置：256 路由专家 + 1 共享专家，topk=8（对应 Megatron 配置 `moe_router_num_experts=256, moe_router_topk=8, moe_shared_expert_intermediate_size=2048`）。

### 2.2 SCMoE：每层共享专家化

Mixtral 之后出现的折中：保留原 FFN 作为共享路径，路由专家只取一部分中间维，激活显存更低、结构改动最小（GLM-4.5/Qwen3-Next 采用 SCMoE 变体）。

### 2.3 DeepSeek-V4 系列：混合压缩注意力 + mHC 残差（2026 新旗舰，暂无公开技术报告）

DeepSeek-V4（1.6T 总参数 / 49B 激活，90 层）把 V3 的"均匀 MLA"升级为**层间异构注意力**，是"注意力预算的 MoE 化"：

- **CSA + HCA 混合**：奇数层用 CSA（Compressed Sparse Attention，近似 V3 的 MLA 压缩率，KV 压缩维 1152），偶数层用激进压缩的 HCA（压缩维 288，1/4 于 CSA），平均压缩率 62.5%——细粒度信息与显存/带宽节省在层间交替换取。
- **mHC（Manifold-Constrained Hyper-Connections）**：替代标准残差连接——残差流与变换流各自做 RMSNorm + 独立线性投影后再相加。动机：90 层深度下标准残差的梯度爆炸；mHC 相当于给残差通路加了可学习的"流形约束"。
- **1M 上下文**：Lookahead Sparse Attention（滑动窗口 8192 + 步长 128 稀疏采样），注意力复杂度从 O(s²) 降到近线性；1M 上下文 KV cache 4.2GB（V3 为 6.4GB）。
- MTP 模块与 mHC 配对：MTP 层内部用双归一化（norm_e/norm_h）+ 双投影的 mHC 式融合，再残差回主干。
- 训练系统的直接影响：注意力侧不再均匀，PP/VP 切分需感知层异构性；KV 压缩率分层使 CP/重计算策略也要分层配置（来源：内部架构走读笔记，2031 行全量数据流解析，非公开文献。**截至 2026-09-13，deepseek-ai 官方 GitHub 组织无 V4 仓库（最新公开模型仓库为 V3.2-Exp），也未检索到 arXiv 技术报告——本节是全文唯一非一手文献来源，引用时须注明**）。

### 2.4 层配置与专家拓扑的新进展（2026）

- **GLM-5（744B-A40B，技术报告 arXiv:2602.15763；官方 FlagScale 训练配置）**：78 层，前三层 dense + 后 75 层 MoE；注意力 = MLA + **DSA 稀疏注意力**（`dsa_indexer_topk=2048`、indexer 32 头、`loss_coeff=0.01` 训练 indexer 的辅助损失）；层规则 `2*full + repeat(full, shared, shared, shared)`（注释标明 GLM-5.2 启用此规则、GLM-5/GLM-5.1 为 None）——**共享注意力层**（shared attention）是与共享专家对偶的新自由度：部分层完全复用上层的 KV 压缩表示，省注意力计算与 KV 显存。并行 TP4×PP4×EP8，且 `decoder_first_pipeline_num_layers=2`：DSA indexer 有缓存约束——PP stage 首层必须是 full 层，这是"架构特性反向约束并行切分"的实例。GLM-5 TR 摘要确认 DSA 旨在"显著降低训练与推理成本、保持长上下文保真"，并引入异步 RL 基础设施解耦生成与训练。
- **GLM-5.3 系列与 IndexCache（arXiv:2603.12201，官方 README 称 IndexShare）**：GLM-5 系列已迭代至 5.3（744B-A40B）并新增 5.3-Flash（320B-A18B）；层规则 `2*full+repeat(full,shared,shared,shared)` 正是 IndexCache 机制的生产实例——把层分为少数 Full 层（独立跑 indexer）与多数 Shared 层（复用最近 Full 层的 top-k 索引，indexer 本身的 O(L²) 开销被摊销），官方称 1M 上下文每 token FLOPs 降 2.9×，并从启发式复用演进到贪心搜索自动选层。
- **Qwen3.5-35B-A3B（官方博客 qwen.ai/blog?id=qwen3.5；官方 FlagScale 训练配置）**：Gated DeltaNet 线性注意力与全注意力按 `linear_attention_freq=4` 混合（每 4 层 1 层线性注意力），线性层 conv_kernel=4、16 key 头/32 value 头、带输出门控；256 专家 topk=8 + `moe_shared_expert_gate`（共享专家带门控）；`moe_permute_fusion` 开启；多模态（27 层 ViT、mrope 三段 [11,11,10]、262144 位置扩展）。系列最新已至 Qwen3.8 与 Qwen3.6-35B-A3B（官方博客 qwen.ai/blog?id=qwen3.8、qwen.ai/blog?id=qwen3.6-35b-a3b，Qwen 系不发 arXiv TR）。
- **CE-MoE（层重配置，arXiv:2608.28511）**：解耦 token-mixing 深度与 channel-mixing 深度——把专家容量集中在少数层、其余层用 dense FFN 补深度。2B→31.5B 扩展阶梯上，同参数预算下训练 GPU 时省 33.3%，下游分数持平。GLM-5 的"3 dense + 75 MoE"与 Qwen3.5 的线性注意力混合层频正是同一思想的生产实例。
- **Kimi K3 Stable LatentMoE（官方技术报告 arXiv:2607.24653 "Kimi K3: Open Frontier Intelligence"，标题经 curl 实测；仓库内 k3_tech_report.pdf 为同一报告的 PDF 版）**：2.78T 总参/104.2B 激活（激活比 1/26.7）、93 层、69 KDA + 24 MLA 混合注意力、NoPE 免位置编码直接外推 1M 上下文。896 路由专家中每 token 激活 16 个（稀疏度 56）+2 全宽共享专家，路由专家工作在 3584 维潜空间（LatentMoE：共享专家保留全宽度、路由专家压潜空间，通信与专家权重流量与路由倍数解耦）。整体扩展效率较 K2 提升约 2.5 倍（独立调参的 scaling-law 曲线）。
- 系统综述（arXiv:2608.08650）将演进归纳为五个耦合维度：专家粒度、专家拓扑、路由自由度、均衡范围、执行结构，主线是"语义路由、计算预算、物理执行三者解耦"。2026 旗舰配置印证该框架：粒度（256~896 细专家）、层分布（dense/MoE/线性注意力混合）、注意力异构（CSA/HCA、DSA、GatedDeltaNet）成为并列的设计轴。

## 3. 路由与负载均衡：从有梯度辅助损失到无梯度偏置

### 3.1 负载均衡三条主线

| 手段 | 代表 | 原理 | 代价 |
|---|---|---|---|
| aux loss（有梯度） | GShard/Switch/Mixtral | 在训练损失上叠加「最小化各专家负载方差」项 | 与主任务梯度耦合，系数需调；过大损害质量 |
| aux loss（有梯度） | Qwen3.5-35B-A3B（global_aux_loss, coeff=1e-3） | 跨 CP/EP 聚合的全局 aux loss | 同上；但 Qwen 系沿用并配 shared_expert_gate |
| expert bias（无梯度） | DeepSeek-V3；Kimi K3 Quantile Balancing | bias 只影响"选谁"不影响权重值；K3 用路由分位数一步把每个专家 bias 设到目标负载（直方图跨 rank 汇聚，仅几百 bin/专家 all-reduce），偏差 b 不进入混合权重 | 需维护 bias 状态；K3 需 Top-(k+1) 截断值 α 做一次前向；不引入梯度污染 |
| 系统级再布局 | TAOT/LAER-MoE 等 | 训练中动态迁移/复制专家参数 | 权重搬运开销，需拓扑感知 |

Megatron 系 `moe_router_load_balancing_type` 支持 `aux_loss / seq_aux_loss / sinkhorn / none`；aux loss 有 local（本卡批内）、seq（序列内）、global（跨卡全局，需跨 CP/EP 聚合）三档（`moe_aux_loss_coeff` 控制系数）。z-loss（`moe_z_loss_coeff`）单独做数值稳定。

### 3.2 DeepSeek-V3 auxiliary-loss-free 路由（实践主力）

- Sigmoid 打分 + 每专家偏置 `b_i`：路由排序用 `sigmoid(logits)+b`，选中后**取回未加偏置的原始分数**作为加权权重——偏置只改变选择、不进入前向数值，因此不污染主任务梯度。
- 偏置更新率 `moe_router_bias_update_rate=0.001`（0.001/步，按全局负载统计调整）。
- 实测：DSv3 训练全程无不可恢复 loss spike、无回滚，是 aux-free 路由在超大规模下稳定性的直接证据。
- FlagScale/DSv3 配置中仍保留 `moe_aux_loss_coeff=0.02 + seq_aux_loss` 作极小系数兜底，与 bias 并行使用。

### 3.3 通信友好的路由约束：group-limited routing

DeepSeek 系把 256 专家分 8 组、每 token 先选 4 组再在组内选专家（`moe_router_num_groups=8, moe_router_group_topk=4`），保证同 token 的专家落点集中，显著降低跨节点 all-to-all 量；Kimi K3、LB 侧研究（如分层 dispatch 的 TerraceMoE 成本模型）延续该方向。**反例（同样一手）**：GLM-5 744B 用 `moe_router_num_groups=1`（完全不做 group-limited）+ sigmoid/expert-bias 路由——当拓扑、通信库（DeepEP 类）与专家放置已把 A2A 压到低占比时，group-limited 的路由自由度代价可能大于通信收益；是否启用应取决于集群网络层级，而非默认。

### 3.4 2026 前沿：从"逼均衡"到"调度均衡"

- **动态副本/迁移**：TAOT（arXiv:2608.03676）把热专家副本放置建模为带通信代价矩阵的最优传输问题（Sinkhorn-Knopp 求解），1.43x 端到端加速；LAER-MoE（ASPLOS'26）用全分片专家参数 (FSEP) + 训练中重布局，1.69x 加速。
- **跨层/预测式均衡**（推理为主，思想可迁移训练）：EasyBalance 跨层联合调度、FreeBalance 用残差表征预测路由提前迁移。
- **Router Replay（Megatron 系）**：记录/回放路由决策，保证 LB 实验可复现，是做均衡对比实验的必备工具。
- **K3 QB 的因果约束**：bias 更新只在下一步生效（批永不使用由自身导出的偏置），推理时冻结——与 DSv3 EMA 偏置同属 aux-free 谱系，K3 把"何时到目标负载"从调 γ 变成解一个分位数。

## 4. 并行策略：EP 与混合并行的设计空间

### 4.1 EP 基本形态

专家并行把专家切到不同设备，token 需跨设备搬运（dispatch/combine 两个 all-to-all）。Megatron 中 EP 组与 TP/DP/CP 组合：`expert_model_parallel_size` 独立于 `tensor_model_parallel_size`，另有专家张量并行 ETP（`expert_tensor_parallel_size`）允许专家矩阵的 TP 度与注意力层不同。token 顺序：gating 后 permute 聚拢 → all-to-all → 专家计算 → all-to-all 回 → unpermute 加权求和。

### 4.2 关键设计决策

- **EP 度 vs 单机内专家数**：专家跨节点越多，通信越走慢速 fabric；group-limited routing（§3.3）与"专家放置对齐网络拓扑"（TAOT/UBEP）是对策。EP 扩展的极限是 all-to-all 延迟占比——Piper（OLCF，arXiv:2605.05049）用资源模型量化各并行方案的记忆/算力/通信需求，指出大规模下 EP all-to-all 延迟是首要瓶颈，其新 all-to-all 算法带宽达厂商实现的 1.2-9 倍。
- **DP attention + EP MoE**：注意力用 DP（每副本独立 batch），MoE 用 EP 汇聚全局 token——现代 MoE 训练的标准混合形态（DSv3 实际部署形态），减少注意力侧冗余通信。
- **容量因子与 token 丢弃**：早期系统用 capacity factor + drop-token 控制不均；dropless（MegaBlocks/Megatron 现代 dispatcher）无丢弃，是当前默认。
- **层级扩展**：X-MoE 等分层方案解决 EP 度超过单层卡数的问题；Piper 的流水线混合并行在 HPC 平台达 2-3.5 倍 MFU 提升。

### 4.3 2026 前沿：解耦执行

DisagMoE（arXiv:2605.11005）把 attention 与 FFN(MoE) 拆到不相交 GPU 组，多级流水 + 单向多对多通信，roofline 模型平衡算力/带宽分配，16 节点 8×H800 上最高 1.8 倍加速——代表"打破层内锁步"的新方向。

## 5. 通信优化：All-to-All 的每一毫秒

### 5.1 通信结构

- **dispatch/combine 两阶段**：token（含 prob）按专家去向聚合发送；结果加权回收。变量长 all-to-all：每 rank 的 input/output splits 依赖路由结果，需 GPU→CPU 同步读取 split（Megatron `token_dispatcher` 的 cuda sync point）。
- **TP×EP 两条通信世界**：EP 组跑 all-to-all；TP 组在 dispatch 入口 all-gather（补回 SP 切开的序列并跨 EP 汇集）、combine 出口 reduce-scatter（合并部分和）。

### 5.2 DeepEP 式双模式 + 低精度通信

- **高吞吐模式（训练/prefill）**：NVLink 域内聚合 → 跨节点 RDMA；**低延迟模式（decode）**：纯 RDMA + 双 buffer。
- **FP8 通信**：dispatch 时激活压到 FP8（DSv3 全程启用，SGLang/DeepEP 同款），通信量减半；Hopper 无原生 FP4 下的 MXFP4 通信压缩（arXiv:2603.02731）在 671B 上再省 14.8% 激活显存、吞吐 +12.5%。

### 5.3 通信-计算 overlap（最大收益点）

| overlap 手段 | 覆盖 | 备注 |
|---|---|---|
| shared expert overlap | 共享专家计算 vs 主路 dispatch | `moe_shared_expert_overlap` |
| 双流 all-to-all（DeepEP） | dispatch 前半 vs attention 计算 | 借异步通信把 A2A 藏进计算 |
| **moe fb overlap（feedback/延迟合并）** | dispatch vs 上一层的 combine/计算 | `moe_fb_overlap: true`（DSv3/ernie45 配置启用） |
| UniEP megakernel | dispatch+GEMM+combine 全融合 | 确定性 token 排序保证数值一致（arXiv:2604.19241） |
| **MoonEP 动态冗余专家（Kimi K3 TR）** | dispatch/combine + 专家执行 | 每 rank 恰收 S×K token 的完美均衡：冗余专家数有 E/R 紧上界（ILP 离线参考+GPU 在线规划核），零拷贝 permute/unpermute、全层静态 shape 免除 host-device 同步 | 已开源 github.com/MoonshotAI/MoonEP |
| HyperParallel-MoE | AIC/AIV 异构任务流 | NPU 上 tile 级调度，1.58x（arXiv:2605.23764） |

Megatron/FlagScale 侧的对应开关：`moe_fb_overlap`、`overlap_moe_expert_parallel_comm`、`moe_shared_expert_overlap`（见 §9）。

### 5.4 层级两跳与新通信库

- **两跳 dispatch**：先节点内聚合再跨节点转发（TerraceMoE 给出 breakeven 成本模型：仅当层级带宽比超过阈值才划算）。
- **通信库**：DeepEP（NVSHMEM/IBGDA）、NCCL EP（Device API 原生 dispatch/combine，LL/HT 双模式，SIGCOMM'26）、UBEP（统一总线超节点，A2A 延迟 -52%）——生产级 EP 通信栈已从"框架内嵌"走向"专用库"。

## 6. 内核与显存：GroupedGEMM、稀疏计算与 FP8

- **GroupedGEMM**（Megatron `moe_grouped_gemm: true`）：一次 kernel 算完本卡所有专家的 FC1/FC2，替代逐专家小 GEMM；与 CUTLASS grouped GEMM / Triton 实现协同。
- **permute/unpermute 融合**：token 重排与 prob 加权融合，减少访存；topk 加权可直接融进 FC2 输出（Megatron `moe_permute_fusion: true`）。
- **MTP（多 token 预测）**：DSv3 的 MTP 模块让同一前向多学一个预测步，等效提升每 FLOP 的学习信号（`mtp_loss` 系列）；与 MoE 主干协同做 spec decode 推理加速。2026 系列实践：DeepSeek-V4 的 MTP 层用 mHC 式双归一化+双投影融合；GLM-5 配置保留 `mtp_loss_scaling_factor=0.3` 但注明发布 checkpoint 无 MTP 层（`mtp_num_layers=0`）——MTP 在"训练期辅助信号"与"发布形态解耦"上是可选件（Kimi 系则两代均保留 1 层 MTP）。（K3 TR 另录：Qwen3.5 式多模态联合预训练中，视觉编码器**从头训练**反而比 SigLIP 初始化更稳——后者梯度范数持续偏高且频繁尖峰；27 层 0.4B ViT、RMSNorm、去全部 bias，视觉 token 2×2 像素混洗后 3584² 输入仍可承受。）
- **FP8 GEMM**：核心 MoE 计算走 FP8（TE），路由打分保持 FP32（`moe_router_dtype: fp32`）；显存侧激活重计算 + SP 已是标配。
- **⚠️ 低精度（FP8）的显存收益是"净额"，可能为负——量化工作区债务**：低精度省下的是**权重/激活的静态存储**（每参数 2B→1B），但量化路径自身新增**临时工作区**（rowwise+columnwise 双份 scale、permute/unpermute 对齐填充、按 token 数一次性分配的重排缓冲），且这些缓冲在激活重计算下**翻倍并驻留**至下一 forward。因此：
  **FP8 净显存 = 权重/激活节省 − 量化工作区债务**，其符号由模型形态与前沿性质决定：
  - **稠密大权重、compute-bound**：节省占主导 → FP8 净省显存并提速（常规收益场景）。
  - **MoE（单专家小、激活/工作区占比高）、memory-bound**：工作区债务可超过节省 → FP8 在同 MBS 下反而 OOM（净负）。
  **决策规则**：投入低精度杠杆前，先判定前沿是 compute-bound 还是 memory-bound；memory-bound 下低精度很少有用，除非**激活工作区本身能缩小**。切勿假定"开 FP8 必省显存"——是否净省取决于"权重/激活节省"与"量化工作区债务"两者的权衡，需结合模型形态（专家粒度、激活占比）与并行配置（MBS/PP/seqlen）综合评估。
- **低比特前沿**：FP8 之外，MXFP4 激活+通信（671B 规模验证不伤收敛）与"activation memory 压缩"是当前活跃方向。

## 7. 训练稳定性：MoE 特有的问题与对策

1. **路由打分 fp32 化**：softmax/sigmoid 在 bf16 下易下溢，打分全程 fp32（Megatron 强制 `moe_router_dtype=fp32`）。
2. **z-loss**：惩罚路由 logits 的对数范数，防 softmax 饱和（`moe_z_loss_coeff`）。
3. **aux-free bias 的稳定性优势**：偏置不进梯度 → 不与主损失竞争，超长训练无辅助损失过强导致的质量损失。
4. **loss spike 治理**：DSv3 报告全程无不可恢复 spike/无回滚，归功于 aux-free 路由 + 精细的混合精度策略（FP8 计算 + BF16 主权重 + fp32 优化器状态）；训练中监控每专家 token 分布（`moe_stats`/router replay）是标配。2026 新增四类稳定性抓手：DSA indexer 的辅助损失（GLM-5 `dsa_indexer_loss_coeff=0.01`，防稀疏索引塌缩）、mHC 残差（DSv4，投影+归一化约束超深栈梯度）、K3 的 LatentMoE 双稳定件（聚合后上投影前插 RMSNorm——原 LatentMoE 的 W↓/门控FFN/W↑ 近四连乘在 2.8T 规模下路由支路内部激活爆炸；SiTU-GLU 软封顶 β₁tanh(x/β₁)⊙σ(x)⊙β₂tanh(x/β₂)，β₁=4/β₂=25 保 SwiGLU 近线性区同时限幅，降低低精度溢出风险）、QK 类权重截断（K2 引入的 MuonClip/weight-clipping 沿用至 K3）。

## 8. 评测与成本：怎么衡量"优化得好"

- **吞吐指标**：tokens/s/GPU、MFU（MoE 的有效 FLOPs 定义要注意：稀疏模型的 FLOPs 计数与激活参数挂钩，跨论文比较需统一口径）。
- **端到端成本**：GPU-hours（DSv3 2.788M H800；CE-MoE 相同验证损失省 33% GPU 时；TAOT 1.43x；DisagMoE 1.8x）。
- **均衡质量**：max/mean 专家负载比、跨 EP rank token 方差；aux loss 值本身不是好指标（可被 bias 操纵）。
- **质量对齐**：验证损失 + 下游 benchmark 持平是所有性能优化的底线（CE-MoE/Piper 等均以此设 gate）。
- **成本模型**：Piper（记忆/算力/通信解析模型 + 微基准校准）、TerraceMoE（通信调用级成本模型）——用模型预筛并行策略，避免全量试错。

## 9. FlagScale / Megatron-LM-FL 配置实践（映射速查）

架构特性会反向约束训练系统：DSA indexer 缓存约束 PP 切分（GLM-5）、mHC 残差改变重计算边界（DSv4）、线性注意力混合层频改变层规格（Qwen3.5 的 `get_gpt_decoder_layer_specs` 需按 `experimental_attention_variant` 生成异构层规格，FlagScale 已在 `gpt_builders.py` 支持 DSA 进入 MTP 块 spec）。配置映射速查如下：

以 FlagScale examples/deepseek_v3 真实训练配置为基准；2026 最新系列的取值对照（GLM-5 / Qwen3.5，均出自官方 examples 配置）附后。

| 配置项 | DSv3 取值 | GLM-5 744B | Qwen3.5 35B-A3B | 性能维度 |
|---|---|---|---|---|
| `moe_router_score_function` | sigmoid | sigmoid | softmax（默认，pre_softmax=False） | 均衡/质量 |
| `moe_router_enable_expert_bias` + `moe_router_bias_update_rate` | true / 0.001 | true / 0.001 | — | 均衡 |
| `moe_router_load_balancing_type` | seq_aux_loss | none | global_aux_loss | 均衡 |
| `moe_aux_loss_coeff` | 0.02 | 0.0 | 1.0e-3 | 均衡 |
| `moe_router_num_groups` / `moe_router_group_topk` | 8 / 4（路由级） | 1 / 1（不分组） | —（未启用） | 通信 |
| `moe_router_topk` / `moe_router_topk_scaling_factor` | 6~8 / 2.446 | 8 / 2.5 | 8 / — | 质量/算力 |
| `num_experts` / `moe_ffn_hidden_size` | 256 / 2048 | 256 / 2048 | 256 / 512 | 架构 |
| `moe_shared_expert_*` | inter 2048~2816, overlap | inter 2048（无 gate） | inter 512 + `moe_shared_expert_gate: true` | 架构 |
| `moe_token_dispatcher_type` | alltoall | alltoall | alltoall | 通信 |
| `moe_grouped_gemm` / `moe_permute_fusion` | true / true | true / — | true / true | 内核 |
| `moe_layer_freq` | "[0]+[1]*N" | "[0]*3+[1]*75" | linear_attention_freq=4 混合 | 架构 |
| `moe_router_dtype` | fp32 | —（默认 fp32） | fp32 | 稳定性 |
| 注意力 | MLA（均匀） | MLA + DSA（indexer topk 2048） | GatedDeltaNet(freq=4) + 全注意力 | 架构/显存 |
| 共享注意力层 | 无 | `2*full+repeat(full,shared,shared,shared)`（GLM-5.2，IndexCache 实例） | 无 | 架构 |
| **Kimi K3**（官方 TR） | **2.78T-A104B** | 69 KDA + 24 MLA（NoPE，1M ctx 外推） | 896 专家×16 激活（latent 3584）+2 共享、SiTU-GLU、Quantile Balancing、MoonEP | 均衡/通信/稳定性 |

> 对照读法：DSv3 与 GLM-5 同走 sigmoid+expert-bias 路线（GLM-5 更纯粹：aux loss 完全为 0、不分组）；Qwen3.5 走 global aux loss 路线但系数极小（1e-3）并给共享专家加门控；Kimi K3 亦 aux-free（sigmoid+bias），但把偏置更新从固定步长换成 Quantile Balancing 分位数直解，并率先给出 896 专家下偏置调节的规模化方案。四家在"均衡手段"上收敛到同一谱系，在"注意力稀疏化"上各选了一条路（DSA 索引+跨层复用 / 线性注意力混合 / 压缩率分层 / KDA 递归门控+MLA 混合）。

**调参起点**（新模型/新集群）：
1. 先定 EP×TP×DP 拓扑（用 Piper 类资源模型或 topo-detect 结果），专家放置对齐网络层级（节点内 NVLink 一组）。
2. 均衡策略默认 aux-free bias（sigmoid 打分）+ 小系数 seq_aux_loss 兜底；z-loss 视稳定性加。
3. 通信三项 overlap 全开（fb_overlap / shared_expert_overlap / EP comm overlap），dispatcher 用 alltoall。
4. 内核用 grouped_gemm + permute fusion；路由打分保持 fp32。
5. 跑小规模 step 验证：MFU、max/mean 负载比、A2A 时间占比（nsys/profiler）三项指标齐全再扩规模。

> Kimi K3 TR 的 infra 章节与本文各节一一对应：MoonEP（完美均衡 EP，§3.4/§5.3）、统一激活管理器（重计算/量化/offload 按张量粒度可组合，§6）、激活跨 PP rank 远程 offload（Mooncake 引擎）、Pipeline ZeRO-2 梯度分片入 CPU、P2P Muon 正交化（免全参数 all-gather）、ViT 计算藏入 PP bubble（§4.3 的生产实例）。

> 最新系列配置模板直接可抄：`examples/glm5/conf/train/744b_a40b.yaml`（DSA+MLA、共享注意力层）、`examples/qwen35/conf/train/35b_a3b.yaml`（线性注意力混合+多模态）、`examples/deepseek_v3/`（group-limited 基线）。

## 10. 开放问题与研究前沿

1. **EP 规模天花板**：千卡以上 EP 的 all-to-all 延迟占比仍高；megakernel 化（UniEP）、统一总线通信（UBEP）、两跳分层是三条路线，尚无统一胜者。
2. **均衡的"质量-效率"帕累托**：aux-free bias 已成主流，但训练中动态副本/迁移（TAOT/LAER-MoE）的权重搬运成本在什么网络条件下回收，缺少统一结论（TerraceMoE 的 breakeven 思路值得推广到所有系统级均衡手段）。
3. **架构-系统协同搜索**：CE-MoE 的层重配置表明"专家放哪几层"本身是性能变量，联合搜索空间（粒度×拓扑×层分布×并行映射）尚无系统化方法。2026 旗舰把搜索空间进一步扩大：注意力压缩率分层（DSv4 CSA/HCA）、共享注意力层（GLM-5.2）、线性/全注意力混合频率（Qwen3.5）都是新轴——且已出现架构特性反向约束并行切分的硬约束（GLM-5 的 DSA 缓存与 PP 首层规则）。
4. **低比特 MoE 训练**：FP8 已实用，FP4 激活+通信在 671B 验证初步可行；专家权重的 MX/NVFP4 与路由数值稳定性的相互作用是开放问题——K3 的 SiTU-GLU（软限幅激活）与 LatentMoE 的 RMSNorm 表明"架构级限幅"已与"数值格式降比特"同台竞争稳定性预算。
5. ** MoE 的 RL 阶段**：rollout 与训练负载特征迥异（RoutePack），预训练优化经验如何迁移到 post-training 是新战场。


## 主要参考文献（arXiv ID）

**经典与架构**：GShard [2006.16668]、Switch Transformer [2101.03961]、ST-MoE [2202.08906]、Mixtral [2401.04088]、DeepSeekMoE [2401.06066]、DeepSeek-V2 [2405.04434]、DeepSeek-V3 [2412.19437]、Qwen3 [2505.09388]、Expert Choice Routing [2202.09368]、Kimi K2.5（官方 TR，仓库内 tech_report.pdf，基座复用 K2：1.04T/32B 激活/384×8，新增多模态联合训练 1T+15T+500B 三阶段）、Kimi K3（官方 TR [2607.24653]，仓库内 k3_tech_report.pdf 为同一报告）、**GLM-5: from Vibe Coding to Agentic Engineering [2602.15763]**、IndexCache [2603.12201]
**最新模型系列（2026-09-13 逐系列核验）**：DeepSeek-V4 1.6T-A49B（**无公开技术报告**——deepseek-ai GitHub 组织无 V4 仓库、最新模型仓库为 V3.2-Exp；数据取内部架构走读笔记（非公开文献）：CSA+HCA/mHC/1M ctx）、GLM-5 744B-A40B（**技术报告 arXiv:2602.15763**，标题经 curl 实测；FlagScale `examples/glm5/conf/train/744b_a40b.yaml`：MLA+DSA、`2*full+repeat(full,shared,shared,shared)` 层规则=IndexCache [2603.12201] 实例；系列含 GLM-5.1/5.2/5.3 与 5.3-Flash 320B-A18B，官方博客 z.ai/blog/glm-5.3）、Qwen3.5-35B-A3B（FlagScale `examples/qwen35/conf/train/35b_a3b.yaml`：Gated DeltaNet 混合、256 专家、多模态；**无 arXiv TR**，官方博客 qwen.ai/blog?id=qwen3.5，系列已至 Qwen3.6-35B-A3B/Qwen3.8）、Kimi K3 2.78T-A104B（**技术报告 arXiv:2607.24653** + MoonshotAI/Kimi-K3 仓库内同文 PDF + 博客 kimi.com/blog/kimi-k3：Stable LatentMoE/Quantile Balancing/MoonEP/SiTU-GLU）
**系统与并行**：MegaBlocks [2211.15841]、Tutel [2206.03382]、MegaScale [2402.15627]、Megatron-LM MoE（NVIDIA Megatron-LM 实现与文档，无独立 arXiv 号）、DeepEP（GitHub DeepSeek）、Piper [2605.05049]、DisagMoE [2605.11005]、UniEP [2604.19241]、HyperParallel-MoE [2605.23764]
**通信**：NCCL EP [2603.13606]、UBEP [2607.06202]、TerraceMoE [2608.27874]、DySHARP (in-switch) [2605.05607]
**均衡**：TAOT [2608.03676]、LAER-MoE [2602.11686]、EasyBalance [2608.07964]、RoutePack [2608.12146]
**架构-性能联合**：CE-MoE [2608.28511]、MoE 演进综述 [2608.08650]
**低精度训练**：Practical FP4 Training for MoE [2603.02731]

---
*注：arXiv ID 均经核验（2026-09：多数经 export.arxiv.org API 元数据核验；GLM-5 [2602.15763] 与 IndexCache [2603.12201] 因 API 限流改为 arXiv abs 页 curl 实测标题核验；ST-MoE [2202.08906] 为经典补充未单独核验）。技术报告检索方法（2026-09-13）：官方 GitHub 组织仓库列表 API + 仓库内 PDF（K3/K2.5）+ README 报告链接（GLM-5）+ 官方博客（Qwen 系不发 arXiv）；Kimi K3/K2.5 报告为仓库内 PDF，引用时以"官方技术报告（仓库内 PDF）"标注。DeepSeek-V4 经 org 全量仓库核验确认无公开报告，为本文唯一非一手来源。*
