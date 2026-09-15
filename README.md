# FlagScale-Agent

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

**Autonomous AI Agent for Large-Scale Training, Inference, and Serving**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/flagos-ai/FlagScale-Agent)

</div>

---

## 🌟 Overview

FlagScale-Agent is an autonomous AI agent specialized in large-scale distributed training, inference, and serving infrastructure. Built on the **ReAct (Reasoning + Acting)** paradigm, it combines LLM reasoning with domain-specific tools, knowledge, and safety constraints to automate complex workflows — from environment setup and data preparation to training launch, monitoring, debugging, and model porting.

**Why FlagScale-Agent?**

- **Domain-Specialized** — 18 built-in skills and 13 knowledge domains covering Megatron-LM, TransformerEngine, NCCL, FlashAttention, and more
- **Autonomous & Safe** — Multi-layer guard system with inject/block/escalate mechanisms prevents runaway execution
- **Persistent Intelligence** — Cross-session memory system (fact/pitfall/insight) accumulates findings and lessons learned
- **Structured Execution** — Plan system with acceptance criteria and verification gates ensures quality
- **Context-Aware** — Automatic context compaction with evict/recall keeps long sessions running smoothly

---

## 📋 Quick Start

### Prerequisites

- Python 3.10 or higher
- API key for Anthropic Claude or OpenAI GPT

### Installation

```bash
git clone https://github.com/flagos-ai/FlagScale-Agent.git
cd FlagScale-Agent
pip install -e .
```

### Configuration

Set your API key:
```bash
# For Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="your_api_key_here"

# For OpenAI GPT
export OPENAI_API_KEY="your_api_key_here"
```

Optionally create a config file at `~/.flagscale/agent.yaml`:
```yaml
# LLM Provider
provider: anthropic                        # anthropic | openai
model: claude-sonnet-4-20250514            # Model name
# api_key: sk-xxx                          # Optional: defaults to env var
# base_url: https://api.custom.com/v1      # Optional: custom endpoint

# Execution Limits
max_iterations: 200                        # Max iterations per turn
max_continuations: 200                     # Max consecutive empty responses
max_context_tokens: 0                      # Context window (0 = auto-detect)
max_output_tokens: 8192                    # Max tokens per response

# See examples/agent_config.yaml for full configuration options
```

### Your First Command

```bash
# Interactive mode
flagscale-agent

# Single-shot query
flagscale-agent "Check GPU availability and CUDA version on this server"
```

---

## 📚 Core Concepts

### Architecture

FlagScale-Agent follows the **ReAct loop** pattern:

1. **User Request** → Agent receives task
2. **Reasoning** → LLM analyzes task, loads relevant skills/knowledge, plans approach
3. **Tool Execution** → Agent calls tools (read files, run commands, check logs, etc.)
4. **Guard Check** → Safety guards validate tool calls before/after execution
5. **Observation** → Tool results feed back into LLM context
6. **Iteration** → Loop continues until task complete or max iterations reached

The guard system operates in three modes:
- **inject** — Advisory reminder (non-blocking)
- **block** — Operation blocked, override available with justification
- **escalate** — Hard block for safety-critical violations

### Skills

Skills are domain-specific workflow guides that teach the agent how to handle specific tasks. Each skill includes task descriptions, tool recommendations, safety constraints, and examples.

**Training Skills:**
- `train-env-setup` — Install FlagScale, dependencies, and conda environments
- `train-data-prep` — Prepare training data (text tokenization, multimodal WebDataset)
- `train-config` — Generate Hydra configs with parallelism strategies
- `train-run` — Launch, stop, and manage distributed training jobs
- `train-monitor` — Monitor logs, detect anomalies (NaN, OOM, NCCL timeout)
- `train-parallel-strategy` — Design TP/PP/DP/EP/CP/SP strategies
- `train-precision-alignment` — Debug precision mismatches across migrations
- `train-model-porter` — Port models from HuggingFace to Megatron-LM
- `train-reproduce` — Reproduce training results from papers/repos

**Inference Skills:**
- `infer-env-setup` — Set up vllm-plugin-FL inference environment
- `infer-model-adapt` — Adapt new models to vllm-plugin-FL
- `infer-hw-adapt` — Port vllm-plugin-FL to new hardware backends
- `infer-plugin-upgrade` — Upgrade vllm-plugin-FL to new vLLM versions
- `infer-precision-check` — Verify inference output correctness

**Infrastructure Skills:**
- `topo-detect` — Detect hardware topology (NVLink, NUMA, RDMA)
- `workspace-layout` — Standardized workspace directory management
- `debug-strategy` — Systematic debugging methodology
- `ops-discipline` — General operational best practices

Skills are automatically loaded based on task context. Use `load_skill(name)` to manually load.

### Knowledge

Knowledge modules provide deep technical documentation for infrastructure domains. The agent loads relevant knowledge **before** acting to avoid trial-and-error mistakes.

**Available Knowledge Domains:**
- `know-megatron-parallel` — TP/PP/DP/CP/EP process groups and communication patterns
- `know-megatron-training` — Training loop, forward/backward, optimizer step, checkpointing
- `know-megatron-model` — Transformer layers, MLA/MTP, RoPE, mixed precision, MoE
- `know-te-fp8` — TransformerEngine FP8 quantization system
- `know-te-attention` — TE attention backends (DotProductAttention, Context Parallel)
- `know-te-comm` — TE communication optimizations (Userbuffers, comm-gemm overlap)
- `know-nccl-core` — NCCL topology detection and channel/ring/tree algorithms
- `know-nccl-runtime` — NCCL collective algorithms, transport layers, tuning
- `know-flash-attn` — FlashAttention tiling, TMA/WGMMA kernels, KV cache
- `know-torch-distributed` — PyTorch ProcessGroup, DDP, FSDP, DeviceMesh
- `know-cuda-kernel` — CUDA operator development (CUTLASS, CuTe, TMA)
- `know-profiling` — Nsys, NCU, PyTorch Profiler integration
- `know-flagscale` — FlagScale repo structure, Hydra configs, Runner execution

Use `load_knowledge(name)` to access documentation.

### Tools

The agent has access to these tools:

**File Operations:**
- `read_file` — Read file contents with line numbers
- `write_file` — Create or overwrite files (supports append mode)
- `edit_file` — Edit files by exact string replacement

**Shell:**
- `shell` — Execute shell commands with timeout and background support

**Training Infrastructure:**
- `flagscale_train_monitor` — Monitor FlagScale training (check/watch modes)
- `inspect_checkpoint` — Deep inspection of PyTorch checkpoints

**Memory System:**
- `memory_write` — Save fact/pitfall/insight for cross-session use
- `memory_read` — Read specific memory entries
- `memory_list` — List and search memory entries

**Planning:**
- `plan_create` — Create structured task plan with acceptance criteria
- `plan_update` — Update plan steps (doing/done/skip), add notes, add steps
- `plan_status` — Show current plan and progress

**Context Management:**
- `evict` — Swap out messages to free context space
- `recall` — Retrieve previously evicted messages

**Knowledge & Skills:**
- `load_skill` — Load domain-specific workflow guide
- `load_knowledge` — Load technical documentation

**Web:**
- `web_fetch` — Fetch and extract text from URLs
- `web_search` — Search for current information

### Guards

Guards enforce safety and quality through lifecycle hooks. When a guard fires:

- **inject** — Advisory reminder (proceed normally, just a heads-up)
- **block** — Operation prevented, add `_override_reason` to proceed
- **escalate** — Hard block, no override (rare, safety-critical only)

**Active Guards:**
- `VerificationGuard` — Enforces verification when marking plan steps done
- `SafetyGuard` — Blocks destructive operations (data deletion, infrastructure changes)
- `KnowledgeSkillGuard` — Reminds to load knowledge/skills for specialized tasks
- `MemoryDisciplineGuard` — Prompts to save findings after discovery work
- `PlanGuard` — Suggests creating plans for multi-step tasks
- `ContextPressureGuard` — Forces eviction when context nears limit
- `TrainingMonitorGuard` — Reminds to use flagscale_train_monitor for training jobs
- `PackageSearchGuard` — Prevents blind package location searches
- `UnitTestGuard` — Requires tests when modifying agent source code
- `PlanUpdateGuard` — Validates plan_update calls for correct usage
- `ArgTypeGuard` — Validates tool parameter types

**Override Mechanism:**

When a guard blocks, re-issue the same tool call with `_override_reason`:
```python
# First attempt (blocked by SafetyGuard)
shell(command="rm -rf /data/experiments")

# Override (explain why it's safe)
shell(command="rm -rf /data/experiments", _override_reason="User confirmed deletion, experiments are archived")
```

### Memory System

The agent persists findings across sessions using three memory types:

**fact** — Verifiable environment state (paths, configs, values)
```python
memory_write(
    key="fact/cluster/gpu_topology",
    type="fact",
    content="值: 4 nodes × 8 GPUs, NVLink within node, IB across nodes\n适用: Multi-node training\n验证命令: nvidia-smi topo -m"
)
```

**pitfall** — Debugging lessons (symptom → cause → fix)
```python
memory_write(
    key="pitfall/nccl/nic_hang",
    type="pitfall",
    content="现象: NCCL hangs at init, no progress\n原因: Using mlx5_1 NIC with known driver bug\n解决: export NCCL_IB_HCA=^mlx5_1\n环境: IB fabric with mixed NIC generations"
)
```

**insight** — Reusable patterns pending digestion
```python
memory_write(
    key="insight/megatron/checkpoint_sharding",
    type="insight",
    content="发现: Megatron checkpoint sharding varies by TP/PP config\n消化方向: Document sharding patterns, write conversion tool\n目标产物: Skill or knowledge doc"
)
```

Query with `memory_list()`, `memory_read(key)`, or `memory_list(keyword='nccl')`.

### Plan System

Plans provide structured tracking for multi-step tasks with acceptance criteria and verification gates.

**Basic Plan:**
```python
plan_create(
    title="Setup FlagScale training environment",
    steps=[
        "Check CUDA and GPU availability",
        "Install FlagScale from GitHub",
        "Prepare tokenizer and data",
        "Generate training config",
        "Launch training and verify"
    ]
)
```

**Structured Plan (recommended for complex tasks):**
```python
plan_create(
    title="Port Qwen2.5 to Megatron-LM",
    steps=[
        {
            "title": "Analyze Qwen2.5 architecture",
            "acceptance": [
                "Document all layer types and dimensions",
                "Identify differences from standard Transformer",
                "List required Megatron modules"
            ]
        },
        {
            "title": "Implement model in Megatron",
            "acceptance": [
                "All layers compile without errors",
                "Forward pass shape matches reference",
                "Unit tests pass"
            ]
        }
    ]
)
```

**Complete Steps with Verification:**
```python
# For steps with acceptance criteria
plan_update(step_done, step_id=1, verification=[
    "Created docs/qwen25_architecture.md with full layer breakdown",
    "Compared with LLaMA: uses RoPE, no bias in QKV",
    "Listed modules: GPTModel, TransformerLayer, Attention, MLP"
])

# For simple steps without acceptance
plan_update(step_done, step_id=2, _override_reason="Ran install script, import flagscale works")
```

**Track Progress:**
```python
plan_update(step_doing, step_id=3)
plan_update(notes="Tried batch=64, got OOM, reducing to 32")
plan_update(step_done, step_id=3, verification=["Training launched, loss decreasing"])
```

### Context Management

Long sessions automatically manage context through eviction and recall:

**Eviction** — Swap out old messages to free space:
```python
evict(indexes=[1, 2, 3, ..., 100])  # Evict messages 1-100
```

**Recall** — Retrieve evicted content:
```python
recall(index=42)  # Get message 42 back
```

Context pressure is monitored automatically. When context reaches 80%, `ContextPressureGuard` forces eviction before allowing further tool calls.

---

## 🎯 Use Cases

### 1. Environment Setup
```bash
flagscale-agent "Set up FlagScale training environment with CUDA 12.1"
```
The agent will:
- Detect hardware (GPU count, type, CUDA version)
- Create conda environment with correct PyTorch version
- Clone and install FlagScale from source
- Build Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention
- Verify installation

### 2. Training Configuration
```bash
flagscale-agent "Generate Megatron config for Qwen2.5-7B with 8 GPUs, TP=4 DP=2, batch size 1M tokens"
```
The agent generates a validated Hydra YAML with:
- Correct parallelism settings (TP=4, PP=1, DP=2)
- Micro-batch size calculated for 1M token global batch
- Model architecture params (layers, hidden size, attention heads)
- Mixed precision config (BF16 + TransformerEngine)

### 3. Training Launch & Monitoring
```bash
flagscale-agent "Launch Qwen2.5-7B training and monitor for issues"
```
The agent:
- Validates config and checks GPU availability
- Launches torchrun with correct environment variables
- Monitors all ranks' logs for errors
- Parses loss/grad_norm/throughput metrics
- Auto-diagnoses issues (OOM, NaN loss, NCCL timeout, hang detection)
- Provides actionable fixes

### 4. Debugging Training Failures
```bash
flagscale-agent "Last training run crashed with OOM. Investigate and fix."
```
The agent:
- Locates latest training logs via experiment tracking
- Identifies OOM error in stderr
- Calculates model memory requirement (weights + optimizer states + activations)
- Compares with available GPU memory
- Suggests fixes: increase TP, enable activation checkpointing, reduce micro-batch size

### 5. Multi-Node Training
```bash
flagscale-agent "Run Qwen2.5-7B on 4 nodes (node1-4), 8 GPUs each, TP=8 PP=4"
```
The agent:
- Verifies shared storage mounted on all nodes
- Detects RDMA network interfaces
- Generates multi-node launch script with proper NCCL settings
- Sets up MASTER_ADDR, MASTER_PORT, NODE_RANK
- Monitors all nodes' logs in parallel
- Detects cross-node communication issues

### 6. Model Porting
```bash
flagscale-agent "Convert HuggingFace LLaMA-3-8B weights to Megatron format with TP=4"
```
The agent:
- Analyzes model architecture and layer mapping
- Writes conversion script with shape validation
- Handles TP sharding (split QKV, column-parallel, row-parallel)
- Executes conversion with progress tracking
- Verifies output checkpoint integrity with inspect_checkpoint

---

## 🛠️ Advanced Usage

### Slash Commands

Inside the agent:
- `/quit` — Exit the agent
- `/reload` — Hot reload (restart process, resume session with new code)
- `/reload config` — Reload config only (no process restart)
- `/resume` — List resumable sessions
- `/resume <number|session_id>` — Resume a specific session
- `/session` — Show current session info (ID, directory, turn count)

### Custom Skills

Create your own skill in `~/.flagscale/skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: Custom training pipeline for XYZ framework
keywords: [xyz, training]
---

# XYZ Training Pipeline

## Overview
Automates training workflow for XYZ framework.

## Prerequisites
- XYZ framework installed
- GPU cluster with shared storage

## Steps
1. Validate environment variables (XYZ_HOME, XYZ_DATA_PATH)
2. Prepare data using XYZ preprocessor
3. Generate config from template
4. Launch training with XYZ launcher
5. Monitor logs for convergence

## Notes
- Always set XYZ_PRECISION=fp16 for A100 GPUs
- Use XYZ_STRATEGY=zero2 for models >10B params
```

Load with `load_skill('my-skill')` in conversation.

### Configuration File Options

Full config options in `~/.flagscale/agent.yaml`:

```yaml
# LLM Provider
provider: anthropic                        # anthropic | openai
model: claude-sonnet-4-20250514            # Model name
api_key: sk-xxx                            # Optional: defaults to env var
base_url: https://api.custom.com/v1        # Optional: custom endpoint

# Execution Limits
max_iterations: 200                        # Max iterations per turn (1 iter = reasoning + tool calls + observation)
max_continuations: 200                     # Max consecutive empty responses (prevents infinite loops)
max_context_tokens: 0                      # Context window size (0 = auto-detect from model)
max_output_tokens: 8192                    # Max tokens per LLM response

# Shell
shell_remind_interval: 60                  # Seconds between long-running shell command reminders

# Session & Skills
session_dir: ~/.flagscale/sessions         # Session storage directory
skill_dirs:                                # Additional skill directories
  - /path/to/custom/skills

# Environment
shell_env:                                 # Environment variables for shell commands
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
  NCCL_DEBUG: INFO
```

See `flagscale_agent/react/config.py` for full `AgentConfig` dataclass.

### Provider-Specific Models

**Anthropic:**
- `claude-sonnet-4-20250514` (200K context, recommended)
- `claude-opus-4-20250514` (200K context, most capable)
- `claude-3-7-sonnet-20250219` (200K context)

**OpenAI:**
- `gpt-4o` (128K context)
- `o1` (200K context, reasoning-focused)
- `o3-mini` (200K context)

**DeepSeek:**
- `deepseek-chat` (200K context)
- `deepseek-reasoner` (200K context, R1 reasoning)

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=flagscale_agent --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
ruff format flagscale_agent/ tests/

# Lint
ruff check flagscale_agent/ tests/

# Type check
mypy flagscale_agent/
```

### Testing Agent Changes

When modifying agent source code (`flagscale_agent/**`):
1. Write unit tests for new functions/methods
2. Add regression tests for bug fixes
3. Update existing tests for behavior changes
4. Run `pytest tests/` to verify 0 failures

Use `/reload` in interactive mode to test changes without restarting:
```bash
# Inside agent
/reload          # Full reload (restart process, resume session)
/reload config   # Config-only reload (no restart)
```

### Project Structure

```
FlagScale-Agent/
├── flagscale_agent/
│   ├── react/
│   │   ├── agent.py              # Main ReAct loop
│   │   ├── config.py             # AgentConfig
│   │   ├── prompt.py             # System prompt builder
│   │   ├── tool_executor.py      # Tool execution
│   │   ├── judge.py              # LLM-based reasoning judge
│   │   ├── guard/                # Guard implementations
│   │   │   ├── verification.py
│   │   │   ├── safety.py
│   │   │   ├── context_pressure.py
│   │   │   └── ...
│   │   ├── skills/               # Skill management
│   │   └── commands.py           # Slash command handlers
│   ├── skills/                   # Built-in skills
│   │   ├── train-env-setup/
│   │   ├── train-run/
│   │   └── ...
│   ├── knowledge/                # Knowledge domains
│   │   └── docs/
│   │       ├── megatron_lm_fl/
│   │       ├── nccl/
│   │       └── ...
│   └── cli.py                    # CLI entry point
├── tests/                        # Unit tests
├── examples/                     # Example configs and workflows
└── docs/                         # Documentation
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork and clone** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make changes** with tests and documentation
4. **Run tests**: `pytest tests/ -v`
5. **Format code**: `ruff format .`
6. **Submit PR** with clear description of changes

**Contribution Guidelines:**
- Follow [ruff](https://github.com/astral-sh/ruff) style guide
- Add unit tests for new features
- Update documentation for user-facing changes
- Write clear commit messages
- Keep PRs focused (one feature/fix per PR)

**Adding New Skills:**
- Create `skills/<skill-name>/SKILL.md` with frontmatter and content
- Test the skill with real use cases
- Document prerequisites and common pitfalls

**Adding New Knowledge:**
- Create `knowledge/docs/<domain>/` directory
- Write markdown files with technical depth
- Include code examples and references
- Update `knowledge/knowledge_config.yaml`

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgments

Built on top of:
- [FlagScale](https://github.com/FlagOpen/FlagScale) — Large-scale training framework
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — Distributed transformer training
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) — FP8 training acceleration
- [vLLM](https://github.com/vllm-project/vllm) — High-performance inference engine
- [Anthropic Claude](https://www.anthropic.com/) — LLM reasoning engine
- [OpenAI](https://openai.com/) — LLM APIs

---

## 📬 Contact

- **GitHub Issues:** [https://github.com/flagos-ai/FlagScale-Agent/issues](https://github.com/flagos-ai/FlagScale-Agent/issues)
- **Discussions:** [https://github.com/flagos-ai/FlagScale-Agent/discussions](https://github.com/flagos-ai/FlagScale-Agent/discussions)

---

<div align="center">

**Built with ❤️ for the AI infrastructure community**

</div>
