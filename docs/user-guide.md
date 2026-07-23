<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagScale Agent — User Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/flagos-ai/FlagScale-Agent.git
cd FlagScale-Agent

# Install (editable mode recommended for development)
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Start interactive mode
flagscale-agent

# Single-shot query
flagscale-agent "set up training environment for qwen3 0.6b"
```

## CLI Options

```
flagscale-agent [OPTIONS] [QUERY]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `anthropic` | LLM provider (`anthropic`, `openai`) |
| `--model` | `-m` | per-provider | Model name (e.g. `claude-sonnet-4-20250514`, `gpt-4o`) |
| `--base-url` | `-b` | None | API base URL for proxies/gateways |
| `--config` | `-c` | auto-detect | Path to agent config YAML |
| `--version` | `-v` | — | Show version and exit |
| `QUERY` | — | — | Single-shot query (non-interactive) |

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | API key for Anthropic provider |
| `OPENAI_API_KEY` | API key for OpenAI provider |

### Optional

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_MODEL` | Override default Anthropic model |
| `ANTHROPIC_BASE_URL` | Override Anthropic API endpoint |
| `OPENAI_BASE_URL` | Override OpenAI API endpoint |
| `HTTP_PROXY` / `HTTPS_PROXY` | Proxy settings (auto-propagated to shell) |

## Configuration File

The agent looks for config in this order:
1. `FLAGSCALE_AGENT_CONFIG` env var
2. `.flagscale/agent.yaml` in current directory
3. `~/.flagscale/agent.yaml`

### Example Config

```yaml
provider: anthropic
model: claude-sonnet-4-20250514
mode: auto                    # "confirm" (default) or "auto"
max_iterations: 200

# Additional skill/tool directories
skill_dirs:
  - /path/to/custom/skills
plugin_tool_dirs:
  - /path/to/custom/tools

# Shell environment variables
shell_env:
  CUDA_HOME: /usr/local/cuda
```

### Config Fields Reference

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"anthropic"` | LLM provider (`anthropic`, `openai`) |
| `model` | auto | Model name |
| `mode` | `"confirm"` | `confirm` = ask before shell; `auto` = run freely |
| `max_iterations` | 200 | Max ReAct iterations per turn |
| `skill_dirs` | built-in | Additional skill search directories |
| `plugin_tool_dirs` | `[]` | Directories to load custom tool plugins from |
| `shell_env` | `{}` | Extra environment variables passed to shell commands |

## Interactive Mode

### Input

- Type your message and press **Enter** to send
- **Alt+Enter** or **Esc → Enter**: Insert newline (multiline input)
- **Ctrl+C**: Interrupt current operation
- **Ctrl+D**: Exit agent

### Slash Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit the agent |
| `/reload` | Reload config and restart session |
| `/skill` | List available skills |
| `/skill <name>` | Load a specific skill |
| `/save` | Save conversation to disk |
| `/resume` | List resumable sessions |
| `/resume <n>` | Resume session by number |
| `/resume <id>` | Resume session by ID prefix |
| `/memory list` | Show all memory entries |
| `/memory clear [type]` | Clear memory entries (optionally by type) |
| `/memory delete <key>` | Delete a specific memory entry |
| `/mode confirm` | Switch to confirmation mode |
| `/mode auto` | Switch to auto mode (no confirmations) |
| `/plan` | Show current plan status |
| `/compact` | Force context compaction (50% target) |

## Tools

Built-in tools are located in `flagscale_agent/react/tools/`.

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (supports line ranges) |
| `write_file` | Create or overwrite files (supports append mode) |
| `edit_file` | Replace exact string matches in files |
| `shell` | Execute shell commands |
| `web_fetch` | Fetch and extract text from URLs |
| `load_skill` | Load a domain-specific skill into context |
| `memory_write` | Save a finding/decision/todo for future sessions |
| `memory_read` | Read a specific memory entry |
| `memory_list` | Search and list memory entries |
| `plan_create` | Create a step-by-step task plan |
| `plan_update` | Update plan progress (step_done, add_steps, etc.) |
| `plan_status` | Show current plan and progress |
| `find_latest_log` | Find and display FlagScale training logs across all ranks |
| `parse_training_metrics` | Parse loss, grad norm, throughput from logs |
| `monitor` | Watch log files for progress/errors without LLM calls |
| `validate_config` | Validate FlagScale YAML config structure |
| `inspect_checkpoint` | Inspect PyTorch/Megatron checkpoint contents |
| `workspace_experiment` | Manage experiment records (create, compare, finalize) |
| `compact_context` | Manually trigger context compaction |

To add a custom tool, simply tell the agent what you need (e.g. "add a tool that checks GPU memory usage") and it will handle the implementation and registration automatically.

## Skills

Skills are domain-specific knowledge modules loaded on demand. They inject expert instructions into the system prompt.

### Built-in Skills

| Skill | Domain |
|-------|--------|
| `train-env-setup` | Set up FlagScale training environment |
| `train-config` | Generate FlagScale training configs |
| `train-run` | Launch and manage distributed training |
| `train-monitor` | Monitor training jobs and health |
| `train-data-prep` | Prepare training data (Megatron format) |
| `train-model-porter` | Port models to Megatron-LM-FL |
| `train-parallel-strategy` | Select parallelism strategies (TP/PP/DP/EP/CP) |
| `train-precision-alignment` | Align training precision across migrations |
| `train-reproduce` | Reproduce training results from references |
| `debug-strategy` | Systematic debugging methodology |
| `topo-detect` | Detect hardware topology (NVLink, NUMA, RDMA) |
| `infer-env-setup` | Set up inference environment |
| `infer-hw-adapt` | Adapt vllm-plugin-FL for hardware backends |
| `ops-discipline` | General operational discipline |
| `workspace-layout` | Standardized workspace directory layout |

### Loading Skills

Skills are loaded automatically based on task context, or manually:

```
# Via slash command
/skill train-run

# The LLM will also call load_skill when needed
```

### Custom Skills

Add custom skill directories via config:

```yaml
skill_dirs:
  - /path/to/my/skills
```

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: my-skill
description: What this skill does
keywords: [keyword1, keyword2]
parameters:
  - name: param1
    description: Optional parameter
requires: []        # Skills that must be loaded first
suggests: []        # Skills that are helpful alongside
constraints:        # Hard rules enforced by ConstraintGuard
  - id: my_rule
    description: Always do X before Y
    trigger:
      tools: [shell]
      keywords: [deploy]
    prompt: Check if X was done
    correction: You must do X first
---

# Skill Content

Instructions injected into the system prompt...
```

## Memory System

The agent maintains persistent memory across sessions for key findings, decisions, and todos.

### Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| `finding` | Discovered facts | "Apex build requires CUDA 12.1+" |
| `decision` | Choices made | "Using TP=4 for this model" |
| `todo` | Pending work | "Need to fix checkpoint conversion" |
| `context` | Background info | "Server has 8xA100 80GB" |

### Memory Lifecycle

- Entries have TTL (default 30 days, configurable)
- Frequently accessed entries auto-promote to permanent
- `supersedes` field allows replacing outdated entries
- Semantic dedup prevents duplicate entries (LLM-judged)

### Storage Location

```
.flagscale/memory/<key>.yaml
```

## Plan System

For complex multi-step tasks, the agent creates and tracks plans.

### Plan Lifecycle

```
plan_create → plan_update (step_doing/step_done) → plan_update (complete)
```

### Plan Features

- Ordered steps with status tracking (pending, doing, done, skipped)
- Auto-continuation: agent proceeds to next step without user prompt
- Plan gate: blocks exploratory behavior if plan is needed but missing
- One active plan at a time (can deactivate/reactivate)

## Experiment Tracking

For training experiments, the agent tracks structured records:

```
workspace_experiment create → add_attempt → update_last_attempt → finalize
```

Each experiment has:
- Name, purpose, hypothesis
- Multiple attempts with: change description, hardware, config, output_dir, result
- Final status and learnings

## Session Management

### Auto-save

Conversations are automatically saved on exit. Incomplete sessions can be resumed.

### Resume

```bash
# Via slash command
/resume <id>
```

### Storage

```
.flagscale/sessions/
  <session-id>/conversation.json
  index.yaml
```

## Modes

### Confirm Mode (default)

- Shell commands require user confirmation (y/n)
- Safer for production environments
- Set via: `mode: confirm` in config

### Auto Mode

- Shell commands run without confirmation
- Suitable for trusted automation pipelines
- Set via: `mode: auto` in config, or `/mode auto`
- Also activated by single-shot queries

## Safety Features

### Guard System

The agent enforces behavioral constraints automatically:

- **Dangerous command blocking**: `rm -rf /`, fork bombs, etc. are blocked
- **Circuit breaker**: Stops after 4 consecutive failures (configurable)
- **Loop detection**: Detects repeated failed patterns
- **Context pressure**: Auto-compacts history before hitting token limits
- **Plan enforcement**: Requires structured planning for complex tasks
- **Budget limits**: Caps total tokens and tool calls per session

### What Gets Blocked

- Destructive system commands without confirmation
- Infinite retry loops on the same error
- Running past context window limits
- Excessive exploratory behavior without a plan

## Troubleshooting

### API Key Not Found

```
API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY.
```

Set the appropriate environment variable for your provider.

### Context Limit Errors

The agent auto-compacts when context pressure exceeds ~80%. If you see repeated compactions:
- Use `/compact` to manually reduce history
- Consider breaking complex tasks into smaller sessions
- Use `plan_create` to maintain continuity across compactions

### Circuit Breaker Triggered

If the agent stops with "circuit breaker":
- The same type of error occurred 4+ times consecutively
- Review the error pattern and fix the underlying issue
- The agent will resume after a cooldown period

### Plan Gate Blocking

If you see "[PLAN GATE] Blocked":
- The agent has been exploring without a plan
- This is expected for complex tasks — the agent will create a plan
- If a plan already exists, this is a bug (fixed in recent versions)

## Project Layout

```
.flagscale/                    # Agent workspace (auto-created)
├── sessions/                  # Saved conversations
│   ├── <id>/conversation.json
│   └── index.yaml
├── memory/                    # Persistent memory entries
│   └── <key>.yaml
├── plans/                     # Task plans
│   └── <id>.yaml
├── experiments/               # Experiment records
│   └── <name>.yaml
└── agent.yaml                 # Local config (optional)
```

## Examples

### Training Workflow

```
> set up environment for training qwen3 0.6b on 8 GPUs

# Agent will:
# 1. Load train-env-setup skill
# 2. Detect hardware topology
# 3. Create conda environment
# 4. Install FlagScale and dependencies
# 5. Load train-config skill
# 6. Generate training YAML config
# 7. Launch training with train-run skill
# 8. Monitor with train-monitor skill
```

### Debugging

```
> my training is getting NCCL timeout errors

# Agent will:
# 1. Load debug-strategy skill
# 2. Check recent training logs
# 3. Analyze NCCL error patterns
# 4. Check network topology
# 5. Suggest fixes (env vars, topology, etc.)
```

### Model Porting

```
> port llama3 8b from huggingface to megatron format

# Agent will:
# 1. Load train-model-porter skill
# 2. Download HF checkpoint
# 3. Analyze model architecture
# 4. Convert weights to Megatron format
# 5. Validate converted checkpoint
# 6. Generate training config
```
