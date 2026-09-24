# Train-MoE-Perf — Summary

MoE 预训练性能优化调优指南（FlagScale / Megatron-LM-FL 后端）。

**Load when**: 配置或加速 MoE 预训练（吞吐低、all-to-all 占比高、专家负载不均/straggler、
MoE loss spike、EP 拓扑选择），或需要 MoE 相关配置项（router/dispatcher/overlap/grouped
gemm）的推荐值与调参路径。

覆盖：快速诊断路径（A2A 占比→负载比→内核效率→显存）、并行拓扑选择、路由与负载均衡
（aux loss vs aux-free expert bias、group-limited routing）、通信 overlap 三件套与低比特
通信、内核（grouped GEMM/permute fusion/FP8）、验收指标（MFU/A2A 占比/max-mean 负载比/
验证损失持平）、常见坑，以及 **2026 最新系列参照模板**（GLM-5 744B DSA/MLA（arXiv:2602.15763）、Qwen3.5
Gated DeltaNet 混合、DeepSeek-V4 CSA+HCA/mHC——一手官方配置路径直接可抄）。机制细节见
knowledge `know-moe-training`。
