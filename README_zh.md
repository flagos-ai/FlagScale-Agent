# FlagScale-Agent

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

**面向大规模模型训练、推理与部署的 AI 基础设施 Agent**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/flagos-ai/FlagScale-Agent)

</div>

---

## 🌟 简介

FlagScale-Agent 是一个面向大规模分布式训练、推理和部署场景的自主 AI Agent。基于 **ReAct（推理 + 行动）** 范式，结合领域专用工具和约束系统，自动化完成复杂的基础设施任务——从环境搭建、数据处理、模型训练到问题诊断。

**核心特点：**
- 🎯 **领域专精** — 内置 FlagScale 训推专用工具：训练监控、配置校验、Checkpoint 检查、日志分析。推理与部署支持即将推出。
- 🤖 **自主执行** — Auto 模式下完全自主多轮执行，Plan 驱动长期任务
- 🛡️ **安全约束** — 多层 Guard 机制（循环检测、熔断器、预算限制）防止失控执行
- 💾 **会话记忆** — 持久化记忆系统跨对话保存发现、决策和上下文
- 📊 **可观测性** — 实时训练监控、结构化实验追踪、自动错误分类

---

## 📋 快速开始

### 环境要求

- Python 3.10 或更高版本
- LLM Provider API Key（Anthropic Claude 或 OpenAI GPT）

### 安装

```bash
git clone https://github.com/flagos-ai/FlagScale-Agent.git
cd FlagScale-Agent
pip install -e .
```

### 配置

设置 API Key：
```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your_api_key_here"

# OpenAI GPT
export OPENAI_API_KEY="your_api_key_here"
```

可选：创建配置文件 `~/.flagscale/agent.yaml`：
```yaml
provider: anthropic
model: claude-sonnet-4-20250514
mode: auto
max_iterations: 200
auto_skill: true
auto_plan: true
```

### 基本使用

#### 交互模式
```bash
flagscale-agent
```

#### 指定 Provider 和模型
```bash
flagscale-agent --provider openai --model gpt-4o
flagscale-agent --provider anthropic --model claude-sonnet-4-20250514
```

#### 单次查询
```bash
flagscale-agent "检查这台服务器上的 CUDA 版本和 GPU 信息"
flagscale-agent "生成 Qwen2.5 7B 的 FlagScale 训练配置，TP=4, DP=2"
```

---

## 📚 核心概念

### Skills（技能）
技能是领域知识模块，教 Agent 如何处理特定任务。内置技能包括：
- `train-env-setup` — 安装 FlagScale、配置 conda 环境
- `train-data-prep` — 准备和分词训练数据
- `train-config` — 生成 Hydra 训练配置
- `train-run` — 启动、监控、停止分布式训练
- `train-monitor` — 分析日志、检测训练问题
- `train-parallel-strategy` — 设计并行策略（TP/PP/DP/EP/SP）
- `train-precision-alignment` — 调试精度对齐
- `train-model-porter` — 从 HuggingFace 移植模型到 Megatron
- `train-reproduce` — 复现参考实现的训练结果
- `debug-strategy` — 系统化调试训练故障
- `topo-detect` — 检测硬件拓扑（NVLink, NUMA, RDMA）
- `workspace-layout` — 标准化工作区目录结构
- `ops-discipline` — 通用运维规范

### Tools（工具）
Agent 内置 19 个专用工具：
- **文件操作**: `read_file`, `write_file`, `edit_file`
- **Shell**: `shell`（支持超时、后台执行）
- **训练**: `find_latest_log`, `parse_training_metrics`, `monitor`, `validate_config`, `inspect_checkpoint`
- **记忆**: `memory_write`, `memory_read`, `memory_list`
- **规划**: `plan_create`, `plan_update`, `plan_status`
- **实验**: `workspace_experiment`（追踪训练尝试）
- **技能**: `load_skill`（加载领域知识）
- **网络**: `web_fetch`（查阅文档、GitHub Issues）
- **上下文**: `compact_context`（手动压缩上下文）

### Guards（守卫）
Guard 是行为约束机制，保证执行安全与质量：
- **LoopDetectGuard** — 检测循环调用，通过 LLM 二次确认避免误报
- **CircuitBreakerGuard** — 重复错误自动熔断
- **BudgetGuard** — Token/工具调用次数限制
- **ProgressGuard** — 监控 Agent 是否在推进任务
- **SafetyGuard** — 阻止危险操作
- **ConstraintGuard** — 技能相关约束

### 会话记忆
Agent 跨会话持久化关键发现、决策和待办事项：
```python
# Agent 内部自动调用
memory_write(
    key="flagscale_native_backend_pattern",
    type="finding",
    content="FlagScale native 后端需要在配置中设置 train.runner.backend=native"
)
# 后续会话自动加载相关记忆
```

### 任务计划
复杂多步骤任务通过 Plan 驱动：
```python
plan_create(
    title="搭建 FlagScale 训练环境",
    steps=[
        "检查 CUDA 和 GPU 可用性",
        "安装 FlagScale",
        "准备 tokenizer 和数据",
        "生成训练配置",
        "启动训练并监控"
    ]
)
# Agent 完成每步后自动推进到下一步
```

---

## 🎯 使用场景

### 环境搭建
```
> 在这台服务器上搭建 FlagScale 训练环境
```
Agent 将自动检测硬件 → 安装依赖 → 创建 conda 环境 → 验证安装。

### 训练启动与监控
```
> 用 8 卡训练 Qwen2.5 7B，TP=4 DP=2，监控 loss 收敛
```
Agent 将生成配置 → 启动训练 → 实时监控 → 检测异常 → 报告结果。

### 问题诊断
```
> 训练 loss 不收敛，帮我排查
```
Agent 将分析日志 → 检查配置 → 检查 checkpoint → 定位根因 → 给出修复方案。

### 模型迁移
```
> 把 HuggingFace 的 LLaMA-3 权重转换为 Megatron 格式
```
Agent 将分析模型结构 → 编写转换脚本 → 执行转换 → 验证正确性。

### 多节点训练
```
> 在 4 个节点（每节点 8 卡）上训练 Qwen2.5-7B，TP=8 PP=4
```
Agent 将验证共享存储 → 生成多节点启动配置 → 设置 NCCL 环境变量 → 并行监控所有节点日志。

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────┐
│              FlagScale Agent                 │
├─────────────────────────────────────────────┤
│  AgentKernel (ReAct Event Loop)             │
│  ┌───────────────────────────────────────┐  │
│  │  LLM → Think → Act → Observe → ...   │  │
│  └───────────────────────────────────────┘  │
├──────────────┬──────────────┬───────────────┤
│   Guards     │    Tools     │   Skills      │
│  ──────────  │  ──────────  │  ──────────   │
│  Loop Detect │  shell       │  train-run    │
│  Budget      │  read_file   │  train-config │
│  Safety      │  monitor     │  debug        │
│  Progress    │  validate    │  topo-detect  │
│  Circuit     │  checkpoint  │  env-setup    │
├──────────────┴──────────────┴───────────────┤
│  Memory  │  Plan  │  Experiment Tracking    │
├─────────────────────────────────────────────┤
│  Providers: Anthropic / OpenAI / Custom     │
└─────────────────────────────────────────────┘
```

---

## 🛠️ 进阶用法

### 自定义技能
在 `~/.flagscale/skills/my-skill/` 添加 `SKILL.md`：

```markdown
---
name: my-skill
description: 自动化 XYZ 训练流水线
keywords: [xyz, training, pipeline]
constraints:
  - id: xyz_env_check
    trigger: {tools: [shell]}
    prompt: "运行 XYZ_SCRIPT 前必须设置 XYZ_ENV=production"
    correction: "在命令前添加 export XYZ_ENV=production"
---
# 自定义工作流

## 步骤
1. 检查环境变量
2. 启动训练脚本
3. 监控输出
```

### 配置选项
详见 `flagscale_agent/react/config.py`：
- `max_iterations` — 每轮最大迭代次数（默认 200）
- `max_context_tokens` — 上下文窗口大小（默认 200k）
- `budget_max_tokens` — Token 总预算（默认 2M）
- `circuit_breaker_threshold` — 熔断器阈值（默认 4 次连续错误）
- `memory_ttl_days` — 记忆过期天数（默认 30 天）

### 命令

| 命令 | 说明 |
|------|------|
| `/mode auto\|confirm` | 切换执行模式 |
| `/skill <name>` | 手动加载技能 |
| `/plan` | 查看当前计划 |
| `/memory list` | 列出记忆条目 |
| `/save` | 保存当前会话 |
| `/resume` | 恢复之前的会话 |
| `/compact` | 手动压缩上下文 |
| `/reload` | 重新加载配置 |
| `/quit` | 退出 |

---

## 📖 文档

- [架构设计](docs/architecture.md) — ReAct 循环、Guard 系统、Judge 等内部机制深入解析
- [技能参考](flagscale_agent/skills/) — 浏览内置技能

---

## 🧪 测试

```bash
pytest tests/ -x -q
```

---

## 🤝 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

贡献检查清单：
- 代码遵循 [ruff](https://github.com/astral-sh/ruff) 风格
- 新功能包含测试
- 公共 API 添加 docstring
- 新增技能/工具时更新文档

---

## 📄 License

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

---

## 🙏 致谢

基于以下项目构建：
- [FlagScale](https://github.com/FlagOpen/FlagScale) — 大规模训练框架
- [Anthropic Claude](https://www.anthropic.com/) — LLM Provider
- [OpenAI GPT](https://openai.com/) — LLM Provider

---

## 📬 联系方式

- GitHub Issues: [https://github.com/flagos-ai/FlagScale-Agent/issues](https://github.com/flagos-ai/FlagScale-Agent/issues)

---

<div align="center">

**Built with ❤️ for the AI infrastructure community**

</div>
