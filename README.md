# FlagScale-Agent

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

**AI Infrastructure Agent for Large-Scale Model Training, Inference, and Serving**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/flagos-ai/FlagScale-Agent)

</div>

---

## 🌟 Overview

FlagScale-Agent is an autonomous AI agent built for large-scale distributed training, inference, and serving workflows. It combines **ReAct (Reasoning + Acting)** paradigm with domain-specific tools and constraints to automate complex infrastructure tasks — from environment setup and data preparation to model training, monitoring, and debugging.

**Key Features:**
- 🎯 **Domain-Specialized** — Built-in tools for FlagScale training: monitoring, config validation, checkpoint inspection, log analysis. Inference & serving support coming soon.
- 🤖 **Autonomous Execution** — Auto mode for fully hands-off multi-turn execution with Plan-driven long-running tasks
- 🛡️ **Safety-First** — Multi-layer Guard system (loop detection, circuit breaker, budget limits) prevents runaway execution
- 💾 **Session Memory** — Persistent memory system stores findings, decisions, and context across conversations
- 📊 **Rich Observability** — Real-time training monitoring, structured experiment tracking, automatic error classification

---

## 📋 Quick Start

### Prerequisites

- Python 3.10 or higher
- API key for an LLM provider (Anthropic Claude or OpenAI GPT)

### Installation

```bash
git clone https://github.com/flagos-ai/FlagScale-Agent.git
cd FlagScale-Agent
pip install -e .
```

### Configuration

Set your API key:
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your_api_key_here"

# For OpenAI GPT
export OPENAI_API_KEY="your_api_key_here"
```

Optionally create a config file at `~/.flagscale/agent.yaml`:
```yaml
provider: anthropic
model: claude-sonnet-4-20250514
mode: auto
max_iterations: 200
auto_skill: true
auto_plan: true
```

### Basic Usage

#### Interactive Mode
```bash
flagscale-agent
```

#### Specify Provider/Model
```bash
flagscale-agent --provider openai --model gpt-4o
flagscale-agent --provider anthropic --model claude-sonnet-4-20250514
```

#### Single-Shot Query
```bash
flagscale-agent "Check if CUDA 12.1 is available on this server"
flagscale-agent "Generate a FlagScale config for Qwen2.5 7B with TP=4, DP=2"
```

---

## 📚 Core Concepts

### Skills
Skills are domain knowledge modules that teach Agents how to handle specific tasks. FlagScale-Agent incorporates multiple skills to guide the Agent in completing tasks such as model training, inference and troubleshooting. Each skill consists of the following contents:
- **Task description** — What the skill solves
- **Tools** — Which tools to use
- **Constraints** — Safety rules and best practices
- **Examples** — Reference workflows

Introduction to built-in skills is as follows:
| Name | Description |
|:---|:---|
| [train-env-setup](flagscale_agent/skills/train-env-setup/SKILL.md) | Set up the FlagScale training environment, including GPU environment checks, FlagScale installation, Conda environment configuration, and dependency installation. |
| [train-data-prep](flagscale_agent/skills/train-data-prep/SKILL.md) | Prepare training data, supporting both plain-text and multimodal data processing. |
| [train-config](flagscale_agent/skills/train-config/SKILL.md) | Generate and manage training configurations, including Hydra YAML files, TP/DP/PP parallel strategies, mixed precision, and checkpoint settings. |
| [train-run](flagscale_agent/skills/train-run/SKILL.md) | Launch, monitor, and manage distributed training jobs, including GPU availability checks, CLI and legacy launch modes, and log directory management. |
| [train-monitor](flagscale_agent/skills/train-monitor/SKILL.md) | Monitor distributed training by analyzing logs, checking training status, detecting anomalies such as OOM errors and NCCL timeouts, and parsing training metrics. |
| [train-parallel-strategy](flagscale_agent/skills/train-parallel-strategy/SKILL.md) | Select and configure TP, PP, DP, EP, and SP parallel strategies for model migration and parallelism issue debugging. |
| [train-precision-alignment](flagscale_agent/skills/train-precision-alignment/SKILL.md) | Debug precision alignment. Precision alignment for three scenarios: model migration (Native → FlagScale), internal iteration (autoregressive), hardware migration (NVIDIA → new hardware) |
| [train-model-porter](flagscale_agent/skills/train-model-porter/SKILL.md) | Port models from Hugging Face and other platforms to Megatron-LM-FL, including architecture analysis and checkpoint conversion and so on. |
| [train-reproduce](flagscale_agent/skills/train-reproduce/SKILL.md) | Reproduce training results, including reuse of original artifacts and baseline verification,etc. |
| [infer-env-setup](flagscale_agent/skills/infer-env-setup/SKILL.md) | Set up the inference environment, including vLLM-Plugin-FL installation, Docker container creation, and FlagGems deployment. |
| [infer-hw-adapt](flagscale_agent/skills/infer-hw-adapt/SKILL.md) | Adapt and fix vLLM-Plugin-FL support for specific hardware backends, such as MetaX and Ascend, through a complete test–patch–validation cycle. |
| [infer-model-adapt](flagscale_agent/skills/infer-model-adapt/SKILL.md) | Model Adaptation. Adapt new models to vllm-plugin-FL, including source coverage discovery, copy-then-patch workflow, import conversion and other steps. |
| [infer-precision-check](flagscale_agent/skills/infer-precision-check/SKILL.md) | Verify the inference output accuracy of vLLM-Plugin-FL across different hardware backends. |
| [infer-plugin-upgrade](flagscale_agent/skills/infer-plugin-upgrade/SKILL.md) |Upgrade vllm-plugin-FL to the new version, including version detection, API difference analysis, etc. |
| [workspace-layout](flagscale_agent/skills/workspace-layout/SKILL.md) | Standardize the workspace directory structure, including storage capacity checks, private disk space prechecks, and dataset path management. |
| [debug-strategy](flagscale_agent\skills\debug-strategy\SKILL.md) | A systematic troubleshooting method for training infrastructure, including error classification, problem localization, and root cause analysis. |
| [topo-detect](flagscale_agent/skills/topo-detect/SKILL.md) | Hardware topology detection, which checks NVLink, NUMA, RDMA and disk topologies. |
| [ops-discipline](flagscale_agent/skills/ops-discipline/SKILL.md) | Define general operational practices, including documentation review strategies, terminal safety, and environment awareness. |

Skills are automatically loaded based on task context. Use `/skill <name>` to manually load.

### Tools
The agent has 19 built-in tools:
- **File ops**: `read_file`, `write_file`, `edit_file`
- **Shell**: `shell` (execute commands with timeout/background support)
- **Training**: `find_latest_log`, `parse_training_metrics`, `monitor`, `validate_config`, `inspect_checkpoint`
- **Memory**: `memory_write`, `memory_read`, `memory_list`
- **Planning**: `plan_create`, `plan_update`, `plan_status`
- **Experiments**: `workspace_experiment` (track training attempts)
- **Skills**: `load_skill` (load domain knowledge)
- **Web**: `web_fetch` (read documentation, GitHub issues)
- **Context**: `compact_context` (manual context compaction)

### Guards
Guards are behavioral constraints with lifecycle hooks that enforce safety and quality:
- **LoopDetectGuard** — Detects repeated/looping tool calls with LLM verification
- **CircuitBreakerGuard** — Trips on repeated errors to prevent infinite retries
- **BudgetGuard** — Enforces token/tool-call limits
- **ProgressGuard** — Monitors whether the agent is making forward progress
- **SafetyGuard** — Blocks dangerous operations (data deletion, infrastructure changes)
- **ConstraintGuard** — Enforces skill-specific constraints (e.g., "always check GPU memory before training")

### Session Memory
The agent persists key findings, decisions, and todos across conversations:
```bash
# Inside the agent
memory_write(
    key="flagscale_native_backend_pattern",
    type="finding",
    content="FlagScale native backend requires train.runner.backend=native in config"
)

# Later sessions automatically retrieve relevant memories
```

### Plans
Multi-step tasks can be tracked with plans:
```bash
# Agent creates a plan
plan_create(
    title="Setup FlagScale training environment",
    steps=[
        "Check CUDA and GPU availability",
        "Install FlagScale from GitHub",
        "Prepare LLaMA tokenizer and data",
        "Generate training config",
        "Launch training and monitor"
    ]
)
# Agent auto-continues to next step after each completion
```

---

## 🎯 Use Cases

### 1. Environment Setup
```bash
flagscale-agent "Set up FlagScale training environment on this server with CUDA 12.1"
```
The agent will:
- Detect hardware (GPU count, type, CUDA version)
- Create conda env with correct dependencies
- Clone and install FlagScale
- Verify installation

### 2. Training Config Generation
```bash
flagscale-agent "Generate a Megatron config for Qwen2.5-7B training with 8 GPUs, TP=4 DP=2, batch size 1M tokens"
```
The agent generates a validated Hydra YAML config with proper parallelism settings.

### 3. Training Launch & Monitor
```bash
flagscale-agent "Launch Qwen2.5-7B training and monitor logs for issues"
```
The agent:
- Validates config
- Launches torchrun command
- Monitors all ranks' stderr for errors
- Checks loss trajectory for divergence
- Auto-diagnoses issues (OOM, NaN, communication timeouts)

### 4. Debug Training Failure
```bash
flagscale-agent "Last training run crashed with OOM. Investigate and suggest fix."
```
The agent:
- Locates latest training logs
- Identifies OOM error in stderr
- Calculates model memory requirement
- Suggests increasing TP or reducing micro-batch size

### 5. Multi-Node Training
```bash
flagscale-agent "Run Qwen2.5-7B training on 4 nodes (node1-4), 8 GPUs each, with TP=8 PP=4"
```
The agent:
- Verifies shared storage (/share/project)
- Generates multi-node launch script
- Sets up NCCL environment variables
- Monitors all nodes' logs in parallel

### 6. Model Porting
```bash
flagscale-agent "Convert HuggingFace LLaMA-3 weights to Megatron format"
```
The agent:
- Analyzes model architecture and layer mapping
- Writes conversion script with shape validation
- Executes conversion with progress tracking
- Verifies output checkpoint integrity

---

## 🛠️ Advanced Usage

### Custom Skills
Create your own skill by adding a `SKILL.md` file to `~/.flagscale/skills/my-skill/`:

```markdown
---
name: my-skill
description: Automate XYZ training pipeline
keywords: [xyz, training, pipeline]
constraints:
  - id: xyz_env_check
    trigger: {tools: [shell]}
    prompt: "Always set XYZ_ENV=production before running XYZ_SCRIPT"
    correction: "Add export XYZ_ENV=production before the command"
---
# My Custom Training Workflow

## Steps
1. Check environment variables
2. Launch XYZ training script
3. Monitor output

## Notes
- Always verify GPU availability first
```

### Config Options
See `flagscale_agent/react/config.py` for all options:
- `max_iterations` — Max turns per session (default: 200)
- `max_context_tokens` — Context window size (default: 200k)
- `budget_max_tokens` — Total token budget (default: 2M)
- `circuit_breaker_threshold` — Error count before circuit trips (default: 4)
- `memory_ttl_days` — Memory expiration (default: 30 days)

### Commands
Inside the agent:
- `/mode auto|confirm` — Switch execution mode
- `/skill <name>` — Load a skill
- `/plan` — Show current plan
- `/memory list` — List memories
- `/save` — Save session
- `/resume` — Resume previous session
- `/compact` — Compact context manually
- `/reload` — Reload config
- `/quit` — Exit

---

## 📖 Documentation

- [Architecture Design](docs/architecture.md) — Deep dive into ReAct loop, Guard system, Judge, and internals
- [Skills Reference](flagscale_agent/skills/) — Browse built-in skills

---

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

Test coverage:
```bash
pytest tests/ --cov=flagscale_agent --cov-report=html
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick checklist:
- Code follows [ruff](https://github.com/astral-sh/ruff) style
- New features include tests
- Docstrings for public APIs
- Update docs if adding new skills/tools

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgments

Built on top of:
- [FlagScale](https://github.com/FlagOpen/FlagScale) — Large-scale training framework
- [Anthropic Claude](https://www.anthropic.com/) — LLM provider
- [OpenAI GPT](https://openai.com/) — LLM provider

---

## 📬 Contact

- GitHub Issues: [https://github.com/flagos-ai/FlagScale-Agent/issues](https://github.com/flagos-ai/FlagScale-Agent/issues)

---

<div align="center">

**Built with ❤️ for the AI infrastructure community**

</div>
