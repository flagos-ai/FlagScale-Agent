# FlagScale-Agent

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

**面向大规模训练、推理和服务的自主 AI Agent**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/flagos-ai/FlagScale-Agent)

</div>

---

## 🌟 概述

FlagScale-Agent 是一个专注于大规模分布式训练、推理和服务基础设施的自主 AI Agent。基于 **ReAct (推理 + 行动)** 范式，它将 LLM 推理能力与领域专用工具、知识和安全约束相结合，自动化复杂工作流 — 从环境配置、数据准备到训练启动、监控、调试和模型移植。

**为什么选择 FlagScale-Agent？**

- **领域专业化** — 内置 18 个技能和 13 个知识域，涵盖 Megatron-LM、TransformerEngine、NCCL、FlashAttention 等
- **自主且安全** — 多层 Guard 系统（inject/block/escalate 机制）防止失控执行
- **持久化智能** — 跨会话记忆系统（fact/pitfall/insight）积累发现和经验教训
- **结构化执行** — 带验收标准和验证关卡的计划系统确保质量
- **上下文感知** — 自动上下文压缩（evict/recall）保持长会话流畅运行

---

## 📋 快速开始

### 前置要求

- Python 3.10 或更高版本
- Anthropic Claude 或 OpenAI GPT 的 API 密钥

### 安装

```bash
git clone https://github.com/flagos-ai/FlagScale-Agent.git
cd FlagScale-Agent
pip install -e .
```

### 配置

设置 API 密钥：
```bash
# Anthropic Claude（推荐）
export ANTHROPIC_API_KEY="your_api_key_here"

# OpenAI GPT
export OPENAI_API_KEY="your_api_key_here"
```

可选：在 `~/.flagscale/agent.yaml` 创建配置文件：
```yaml
# LLM 提供商
provider: anthropic                        # anthropic | openai
model: claude-sonnet-4-20250514            # 模型名称
# api_key: sk-xxx                          # 可选：默认从环境变量读取
# base_url: https://api.custom.com/v1      # 可选：自定义端点

# 执行限制
max_iterations: 200                        # 每个 turn 的最大迭代次数
max_continuations: 200                     # 最大连续空响应次数
max_context_tokens: 0                      # 上下文窗口大小（0 = 自动检测）
max_output_tokens: 8192                    # 单次响应最大 token 数

# 完整配置选项参见 examples/agent_config.yaml
```

### 第一个命令

```bash
# 交互模式
flagscale-agent

# 单次查询
flagscale-agent "检查本服务器的 GPU 可用性和 CUDA 版本"
```

---

## 📚 核心概念

### 架构

FlagScale-Agent 遵循 **ReAct 循环**模式：

1. **用户请求** → Agent 接收任务
2. **推理** → LLM 分析任务、加载相关技能/知识、规划方法
3. **工具执行** → Agent 调用工具（读文件、执行命令、检查日志等）
4. **Guard 检查** → 安全 Guard 在执行前后验证工具调用
5. **观察** → 工具结果反馈到 LLM 上下文
6. **迭代** → 循环直到任务完成或达到最大迭代次数

Guard 系统有三种工作模式：
- **inject** — 建议性提醒（非阻塞）
- **block** — 操作被阻止，可通过说明理由覆盖
- **escalate** — 硬阻止，安全关键场景，无法覆盖

### 技能（Skills）

技能是领域专用的工作流指南，教会 Agent 如何处理特定任务。每个技能包含任务描述、工具推荐、安全约束和示例。

**训练技能：**
- `train-env-setup` — 安装 FlagScale、依赖和 conda 环境
- `train-data-prep` — 准备训练数据（文本分词、多模态 WebDataset）
- `train-config` — 生成带并行策略的 Hydra 配置
- `train-run` — 启动、停止和管理分布式训练任务
- `train-monitor` — 监控日志、检测异常（NaN、OOM、NCCL 超时）
- `train-parallel-strategy` — 设计 TP/PP/DP/EP/CP/SP 策略
- `train-precision-alignment` — 调试跨迁移的精度不匹配
- `train-model-porter` — 从 HuggingFace 移植模型到 Megatron-LM
- `train-reproduce` — 从论文/代码库复现训练结果

**推理技能：**
- `infer-env-setup` — 配置 vllm-plugin-FL 推理环境
- `infer-model-adapt` — 适配新模型到 vllm-plugin-FL
- `infer-hw-adapt` — 移植 vllm-plugin-FL 到新硬件后端
- `infer-plugin-upgrade` — 升级 vllm-plugin-FL 到新 vLLM 版本
- `infer-precision-check` — 验证推理输出正确性

**基础设施技能：**
- `topo-detect` — 检测硬件拓扑（NVLink、NUMA、RDMA）
- `workspace-layout` — 标准化工作空间目录管理
- `debug-strategy` — 系统化调试方法论
- `ops-discipline` — 通用运维最佳实践

技能会根据任务自动加载。使用 `load_skill(name)` 手动加载。

### 知识（Knowledge）

知识模块为基础设施领域提供深度技术文档。Agent 在行动**之前**加载相关知识，避免试错式错误。

**可用知识域：**
- `know-megatron-parallel` — TP/PP/DP/CP/EP 进程组和通信模式
- `know-megatron-training` — 训练循环、前向/反向传播、优化器步骤、检查点
- `know-megatron-model` — Transformer 层、MLA/MTP、RoPE、混合精度、MoE
- `know-te-fp8` — TransformerEngine FP8 量化系统
- `know-te-attention` — TE 注意力后端（DotProductAttention、Context Parallel）
- `know-te-comm` — TE 通信优化（Userbuffers、comm-gemm overlap）
- `know-nccl-core` — NCCL 拓扑检测和 channel/ring/tree 算法
- `know-nccl-runtime` — NCCL 集合通信算法、传输层、调优
- `know-flash-attn` — FlashAttention tiling、TMA/WGMMA kernel、KV cache
- `know-torch-distributed` — PyTorch ProcessGroup、DDP、FSDP、DeviceMesh
- `know-cuda-kernel` — CUDA 算子开发（CUTLASS、CuTe、TMA）
- `know-profiling` — Nsys、NCU、PyTorch Profiler 集成
- `know-flagscale` — FlagScale 代码结构、Hydra 配置、Runner 执行

使用 `load_knowledge(name)` 访问文档。

### 工具（Tools）

Agent 可以使用以下工具：

**文件操作：**
- `read_file` — 读取文件内容（带行号）
- `write_file` — 创建或覆盖文件（支持追加模式）
- `edit_file` — 通过精确字符串替换编辑文件

**Shell：**
- `shell` — 执行 shell 命令（支持超时和后台执行）

**训练基础设施：**
- `flagscale_train_monitor` — 监控 FlagScale 训练（check/watch 模式）
- `inspect_checkpoint` — 深度检查 PyTorch 检查点

**记忆系统：**
- `memory_write` — 保存 fact/pitfall/insight 用于跨会话
- `memory_read` — 读取特定记忆条目
- `memory_list` — 列出和搜索记忆条目

**计划系统：**
- `plan_create` — 创建带验收标准的结构化任务计划
- `plan_update` — 更新计划步骤（doing/done/skip）、添加笔记、添加步骤
- `plan_status` — 显示当前计划和进度

**上下文管理：**
- `evict` — 交换消息以释放上下文空间
- `recall` — 检索之前被驱逐的消息

**知识与技能：**
- `load_skill` — 加载领域专用工作流指南
- `load_knowledge` — 加载技术文档

**Web：**
- `web_fetch` — 获取并提取 URL 文本内容
- `web_search` — 搜索当前信息

### Guards（守卫）

Guards 通过生命周期钩子强制执行安全和质量。当 Guard 触发时：

- **inject** — 建议性提醒（正常继续，只是提醒）
- **block** — 操作被阻止，添加 `_override_reason` 继续
- **escalate** — 硬阻止，无法覆盖（罕见，安全关键）

**活跃的 Guards：**
- `VerificationGuard` — 标记计划步骤完成时强制验证
- `SafetyGuard` — 阻止破坏性操作（数据删除、基础设施变更）
- `KnowledgeSkillGuard` — 提醒为专业任务加载知识/技能
- `MemoryDisciplineGuard` — 提示在发现工作后保存发现
- `PlanGuard` — 建议为多步骤任务创建计划
- `ContextPressureGuard` — 上下文接近限制时强制驱逐
- `TrainingMonitorGuard` — 提醒为训练任务使用 flagscale_train_monitor
- `PackageSearchGuard` — 防止盲目的包位置搜索
- `UnitTestGuard` — 修改 agent 源代码时要求测试
- `PlanUpdateGuard` — 验证 plan_update 调用的正确使用
- `ArgTypeGuard` — 验证工具参数类型

**覆盖机制：**

当 Guard 阻止时，用 `_override_reason` 重新发起相同的工具调用：
```python
# 第一次尝试（被 SafetyGuard 阻止）
shell(command="rm -rf /data/experiments")

# 覆盖（解释为什么安全）
shell(command="rm -rf /data/experiments", _override_reason="用户确认删除，实验已归档")
```

### 记忆系统（Memory System）

Agent 使用三种记忆类型跨会话持久化发现：

**fact** — 可验证的环境状态（路径、配置、值）
```python
memory_write(
    key="fact/cluster/gpu_topology",
    type="fact",
    content="值: 4 节点 × 8 GPU，节点内 NVLink，跨节点 IB\n适用: 多节点训练\n验证命令: nvidia-smi topo -m"
)
```

**pitfall** — 调试教训（现象 → 原因 → 解决）
```python
memory_write(
    key="pitfall/nccl/nic_hang",
    type="pitfall",
    content="现象: NCCL 初始化时挂起，无进展\n原因: 使用 mlx5_1 网卡，已知驱动 bug\n解决: export NCCL_IB_HCA=^mlx5_1\n环境: 混合代际网卡的 IB fabric"
)
```

**insight** — 待消化的可复用模式
```python
memory_write(
    key="insight/megatron/checkpoint_sharding",
    type="insight",
    content="发现: Megatron 检查点分片随 TP/PP 配置变化\n消化方向: 记录分片模式，编写转换工具\n目标产物: Skill 或 knowledge 文档"
)
```

使用 `memory_list()`、`memory_read(key)` 或 `memory_list(keyword='nccl')` 查询。

### 计划系统（Plan System）

计划为多步骤任务提供带验收标准和验证关卡的结构化跟踪。

**基础计划：**
```python
plan_create(
    title="配置 FlagScale 训练环境",
    steps=[
        "检查 CUDA 和 GPU 可用性",
        "从 GitHub 安装 FlagScale",
        "准备 tokenizer 和数据",
        "生成训练配置",
        "启动训练并验证"
    ]
)
```

**结构化计划（推荐用于复杂任务）：**
```python
plan_create(
    title="移植 Qwen2.5 到 Megatron-LM",
    steps=[
        {
            "title": "分析 Qwen2.5 架构",
            "acceptance": [
                "记录所有层类型和维度",
                "识别与标准 Transformer 的差异",
                "列出所需的 Megatron 模块"
            ]
        },
        {
            "title": "在 Megatron 中实现模型",
            "acceptance": [
                "所有层编译无错误",
                "前向传播形状匹配参考",
                "单元测试通过"
            ]
        }
    ]
)
```

**带验证完成步骤：**
```python
# 对于有验收标准的步骤
plan_update(step_done, step_id=1, verification=[
    "创建了 docs/qwen25_architecture.md，包含完整层分解",
    "与 LLaMA 对比：使用 RoPE，QKV 无 bias",
    "列出模块：GPTModel、TransformerLayer、Attention、MLP"
])

# 对于没有验收标准的简单步骤
plan_update(step_done, step_id=2, _override_reason="运行安装脚本，import flagscale 成功")
```

**跟踪进度：**
```python
plan_update(step_doing, step_id=3)
plan_update(notes="尝试 batch=64，遇到 OOM，降至 32")
plan_update(step_done, step_id=3, verification=["训练启动，loss 下降"])
```

### 上下文管理（Context Management）

长会话通过驱逐和召回自动管理上下文：

**驱逐（Eviction）** — 交换旧消息以释放空间：
```python
evict(indexes=[1, 2, 3, ..., 100])  # 驱逐消息 1-100
```

**召回（Recall）** — 检索被驱逐的内容：
```python
recall(index=42)  # 取回消息 42
```

上下文压力自动监控。当上下文达到 80% 时，`ContextPressureGuard` 在允许进一步工具调用前强制驱逐。

---

## 🎯 使用场景

### 1. 环境配置
```bash
flagscale-agent "配置 FlagScale 训练环境，使用 CUDA 12.1"
```
Agent 将：
- 检测硬件（GPU 数量、类型、CUDA 版本）
- 创建带正确 PyTorch 版本的 conda 环境
- 从源码克隆并安装 FlagScale
- 构建 Megatron-LM-FL、TransformerEngine-FL、Apex、Flash-Attention
- 验证安装

### 2. 训练配置
```bash
flagscale-agent "为 Qwen2.5-7B 生成 Megatron 配置，8 GPU，TP=4 DP=2，batch size 1M tokens"
```
Agent 生成验证过的 Hydra YAML，包含：
- 正确的并行设置（TP=4, PP=1, DP=2）
- 根据 1M token 全局批次计算的 micro-batch size
- 模型架构参数（层数、隐藏层大小、注意力头数）
- 混合精度配置（BF16 + TransformerEngine）

### 3. 训练启动与监控
```bash
flagscale-agent "启动 Qwen2.5-7B 训练并监控问题"
```
Agent 将：
- 验证配置并检查 GPU 可用性
- 使用正确的环境变量启动 torchrun
- 监控所有 rank 的日志错误
- 解析 loss/grad_norm/throughput 指标
- 自动诊断问题（OOM、NaN loss、NCCL 超时、挂起检测）
- 提供可操作的修复方案

### 4. 调试训练失败
```bash
flagscale-agent "上次训练因 OOM 崩溃。调查并修复。"
```
Agent 将：
- 通过实验跟踪定位最新训练日志
- 在 stderr 中识别 OOM 错误
- 计算模型内存需求（权重 + 优化器状态 + 激活）
- 与可用 GPU 内存对比
- 建议修复：增加 TP、启用激活检查点、减少 micro-batch size

### 5. 多节点训练
```bash
flagscale-agent "在 4 个节点（node1-4）上运行 Qwen2.5-7B，每节点 8 GPU，TP=8 PP=4"
```
Agent 将：
- 验证所有节点已挂载共享存储
- 检测 RDMA 网络接口
- 生成带正确 NCCL 设置的多节点启动脚本
- 设置 MASTER_ADDR、MASTER_PORT、NODE_RANK
- 并行监控所有节点日志
- 检测跨节点通信问题

### 6. 模型移植
```bash
flagscale-agent "将 HuggingFace LLaMA-3-8B 权重转换为 Megatron 格式，TP=4"
```
Agent 将：
- 分析模型架构和层映射
- 编写带形状验证的转换脚本
- 处理 TP 分片（拆分 QKV、列并行、行并行）
- 执行带进度跟踪的转换
- 使用 inspect_checkpoint 验证输出检查点完整性

---

## 🛠️ 高级用法

### 斜杠命令

在 agent 内部：
- `/quit` — 退出 agent
- `/reload` — 热重载（重启进程，恢复会话并加载新代码）
- `/reload config` — 仅重载配置（无进程重启）
- `/resume` — 列出可恢复的会话
- `/resume <number|session_id>` — 恢复特定会话
- `/session` — 显示当前会话信息（ID、目录、turn 计数）

### 自定义技能

在 `~/.flagscale/skills/my-skill/SKILL.md` 创建自己的技能：

```markdown
---
name: my-skill
description: XYZ 框架的自定义训练流程
keywords: [xyz, training]
---

# XYZ 训练流程

## 概述
自动化 XYZ 框架的训练工作流。

## 前置条件
- 已安装 XYZ 框架
- 带共享存储的 GPU 集群

## 步骤
1. 验证环境变量（XYZ_HOME、XYZ_DATA_PATH）
2. 使用 XYZ 预处理器准备数据
3. 从模板生成配置
4. 使用 XYZ 启动器启动训练
5. 监控日志收敛

## 注意事项
- A100 GPU 始终设置 XYZ_PRECISION=fp16
- 模型 >10B 参数使用 XYZ_STRATEGY=zero2
```

在对话中使用 `load_skill('my-skill')` 加载。

### 配置文件选项

`~/.flagscale/agent.yaml` 的完整配置选项：

```yaml
# LLM 提供商
provider: anthropic                        # anthropic | openai
model: claude-sonnet-4-20250514            # 模型名称
api_key: sk-xxx                            # 可选：默认从环境变量读取
base_url: https://api.custom.com/v1        # 可选：自定义端点

# 执行限制
max_iterations: 200                        # 每个 turn 的最大迭代次数（1 次迭代 = 推理 + 工具调用 + 观察）
max_continuations: 200                     # 最大连续空响应次数（防止无限循环）
max_context_tokens: 0                      # 上下文窗口大小（0 = 从模型自动检测）
max_output_tokens: 8192                    # 每次 LLM 响应的最大 token 数

# Shell
shell_remind_interval: 60                  # 长时间运行的 shell 命令提醒间隔（秒）

# 会话与技能
session_dir: ~/.flagscale/sessions         # 会话存储目录
skill_dirs:                                # 额外的技能目录
  - /path/to/custom/skills

# 环境
shell_env:                                 # shell 命令的环境变量
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
  NCCL_DEBUG: INFO
```

完整的 `AgentConfig` 数据类参见 `flagscale_agent/react/config.py`。

### 提供商特定模型

**Anthropic：**
- `claude-sonnet-4-20250514`（200K 上下文，推荐）
- `claude-opus-4-20250514`（200K 上下文，最强能力）
- `claude-3-7-sonnet-20250219`（200K 上下文）

**OpenAI：**
- `gpt-4o`（128K 上下文）
- `o1`（200K 上下文，推理专注）
- `o3-mini`（200K 上下文）

**DeepSeek：**
- `deepseek-chat`（200K 上下文）
- `deepseek-reasoner`（200K 上下文，R1 推理）

---

## 🧪 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_config.py -v

# 带覆盖率运行
pytest tests/ --cov=flagscale_agent --cov-report=html
open htmlcov/index.html
```

### 代码质量

```bash
# 格式化代码
ruff format flagscale_agent/ tests/

# Lint
ruff check flagscale_agent/ tests/

# 类型检查
mypy flagscale_agent/
```

### 测试 Agent 变更

修改 agent 源代码（`flagscale_agent/**`）时：
1. 为新函数/方法编写单元测试
2. 为 bug 修复添加回归测试
3. 为行为变更更新现有测试
4. 运行 `pytest tests/` 验证 0 失败

在交互模式下使用 `/reload` 测试变更而无需重启：
```bash
# 在 agent 内部
/reload          # 完全重载（重启进程，恢复会话）
/reload config   # 仅配置重载（无重启）
```

### 项目结构

```
FlagScale-Agent/
├── flagscale_agent/
│   ├── react/
│   │   ├── agent.py              # 主 ReAct 循环
│   │   ├── config.py             # AgentConfig
│   │   ├── prompt.py             # 系统提示构建器
│   │   ├── tool_executor.py      # 工具执行
│   │   ├── judge.py              # 基于 LLM 的推理判断
│   │   ├── guard/                # Guard 实现
│   │   │   ├── verification.py
│   │   │   ├── safety.py
│   │   │   ├── context_pressure.py
│   │   │   └── ...
│   │   ├── skills/               # 技能管理
│   │   └── commands.py           # 斜杠命令处理器
│   ├── skills/                   # 内置技能
│   │   ├── train-env-setup/
│   │   ├── train-run/
│   │   └── ...
│   ├── knowledge/                # 知识域
│   │   └── docs/
│   │       ├── megatron_lm_fl/
│   │       ├── nccl/
│   │       └── ...
│   └── cli.py                    # CLI 入口点
├── tests/                        # 单元测试
├── examples/                     # 示例配置和工作流
└── docs/                         # 文档
```

---

## 🤝 贡献

我们欢迎贡献！以下是入门方法：

1. **Fork 并克隆**仓库
2. **创建分支**用于您的功能：`git checkout -b feature/my-feature`
3. **进行变更**，包含测试和文档
4. **运行测试**：`pytest tests/ -v`
5. **格式化代码**：`ruff format .`
6. **提交 PR**，清晰描述变更

**贡献指南：**
- 遵循 [ruff](https://github.com/astral-sh/ruff) 风格指南
- 为新功能添加单元测试
- 为面向用户的变更更新文档
- 编写清晰的提交消息
- 保持 PR 专注（每个 PR 一个功能/修复）

**添加新技能：**
- 创建 `skills/<skill-name>/SKILL.md`，包含前置和内容
- 用真实用例测试技能
- 记录前置条件和常见陷阱

**添加新知识：**
- 创建 `knowledge/docs/<domain>/` 目录
- 编写具有技术深度的 markdown 文件
- 包含代码示例和参考
- 更新 `knowledge/knowledge_config.yaml`

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可。

---

## 🙏 致谢

基于以下项目构建：
- [FlagScale](https://github.com/FlagOpen/FlagScale) — 大规模训练框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 分布式 transformer 训练
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) — FP8 训练加速
- [vLLM](https://github.com/vllm-project/vllm) — 高性能推理引擎
- [Anthropic Claude](https://www.anthropic.com/) — LLM 推理引擎
- [OpenAI](https://openai.com/) — LLM API

---

## 📬 联系

- **GitHub Issues：** [https://github.com/flagos-ai/FlagScale-Agent/issues](https://github.com/flagos-ai/FlagScale-Agent/issues)
- **讨论区：** [https://github.com/flagos-ai/FlagScale-Agent/discussions](https://github.com/flagos-ai/FlagScale-Agent/discussions)

---

<div align="center">

**为 AI 基础设施社区用 ❤️ 构建**

</div>
