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
- `debug-strategy` — 系统化调试训练故障
- `topo-detect` — 检测硬件拓扑（NVLink, NUMA, RDMA）

### Tools（工具）
Agent 内置 20+ 专用工具：
- **文件操作**: `read_file`, `write_file`, `edit_file`
- **Shell**: `shell`（支持超时、后台执行）
- **训练**: `find_latest_log`, `monitor`, `validate_config`, `inspect_checkpoint`
- **记忆**: `memory_write`, `memory_read`, `memory_list`
- **规划**: `plan_create`, `plan_update`, `plan_status`
- **实验**: `workspace_experiment`（追踪训练尝试）
- **网络**: `web_fetch`（查阅文档、GitHub Issues）

### Guards（守卫）
Guard 是行为约束机制，保证执行安全与质量：
- **LoopDetectGuard** — 检测循环调用，通过 LLM 二次确认避免误报
- **CircuitBreakerGuard** — 重复错误自动熔断
- **BudgetGuard** — Token/工具调用次数限制
- **ProgressGuard** — 监控 Agent 是否在推进任务
- **SafetyGuard** — 阻止危险操作
- **ConstraintGuard** — 技能相关约束

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

详细架构设计参见 [docs/architecture.md](docs/architecture.md)。

---

## 🔧 命令

| 命令 | 说明 |
|------|------|
| `/skill <name>` | 手动加载技能 |
| `/file <path>` | 读取文件内容 |
| `/plan` | 查看当前计划 |
| `/memory list` | 列出记忆条目 |
| `/mode auto\|confirm` | 切换执行模式 |
| `/reload` | 重新加载配置 |
| `/quit` | 退出 |

---

## 🤝 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

---

## 📄 License

本项目采用 [Apache License 2.0](LICENSE) 开源协议。
