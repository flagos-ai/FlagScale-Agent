# FlagScale Agent — System Architecture

## Overview

FlagScale Agent is an autonomous AI agent specialized for large-scale model training and inference infrastructure. It uses a **ReAct (Reasoning + Acting) loop** with a composable **Guard/Judge** architecture to provide safe, skill-driven automation.

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (typer)                          │
│              flagscale-agent [--provider] [query]            │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Orchestrator   │  Route: single worker / subtask pipeline
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   WorkerAgent    │  Owns tools, skills, memory, guards
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   AgentKernel    │  ReAct event loop (iteration engine)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──┐   ┌──────▼──────┐  ┌───▼────────┐
     │  Guards    │   │  ToolExec   │  │  Provider   │
     │ (pre/post) │   │  (parallel) │  │ (LLM call)  │
     └────────────┘   └─────────────┘  └─────────────┘
```

## Core Components

### 1. CLI (`cli.py`)

Entry point via `flagscale-agent` command. Uses Typer for argument parsing.

- `--provider` / `-p`: LLM provider (anthropic, openai)
- `--model` / `-m`: Model name override
- `--base-url` / `-b`: API base URL for proxies
- `--config` / `-c`: Agent config YAML path
- Positional `query`: Single-shot mode (non-interactive)

Creates `AgentConfig`, instantiates `WorkerAgent` and `Orchestrator`, then calls `agent.run()`.

### 2. Orchestrator (`orchestrator.py`)

Routes user input to execution strategies:

| Strategy | Use Case |
|----------|----------|
| **Single Worker** | Simple task → one WorkerAgent turn |
| **SubtaskRunner** | Complex multi-stage task → serial DAG pipeline |
| **BatchRunner** | Parallel comparison tasks |

Uses LLM-based routing to classify task complexity. Falls through to single worker for most tasks.

### 3. WorkerAgent (`agent.py`)

The central coordinator. Owns all subsystems:

- **ToolRegistry**: Registered tool instances
- **SkillManager**: Loads and indexes SKILL.md files
- **SessionMemory**: Cross-session persistent memory
- **TaskPlan**: Step-based plan tracking
- **Judge**: Tiered classification engine
- **GuardRegistry**: Behavioral constraint system
- **PromptBuilder**: Dynamic system prompt assembly
- **ToolExecutor**: Parallel/serial tool dispatch
- **ExperimentManager**: Training experiment records

Key responsibilities:
- Interactive REPL with prompt_toolkit (multiline, completions, key bindings)
- Session save/load/resume
- Slash command dispatch (`/skill`, `/plan`, `/memory`, `/reload`, etc.)
- Kernel construction and lifecycle

### 4. AgentKernel (`kernel.py`)

Minimal ReAct event loop. One turn = one user message → completion.

```
for iteration in range(max_iter):
    1. Pre-guard check (can block entire iteration)
    2. LLM call (streaming with tool_use)
    3. If no tool_calls → check auto-continue or stop
    4. Execute tools via ToolExecutor
    5. Post-guard check (per tool, can inject messages)
    6. Append tool results to history
    7. Loop
```

Features:
- Context pressure tracking (token budget awareness)
- Auto-continuation when plan steps remain
- Graceful handling of context limit errors (compact + retry)
- Interrupt support (Ctrl+C)

### 5. ToolExecutor (`tool_executor.py`)

Dispatches tool calls with:
- **Parallel execution** (ThreadPoolExecutor) for independent calls
- **Deduplication** (identical calls in same batch → skip)
- **Guard pre-check** per tool in parallel mode
- **Animated display** with per-tool progress indicators
- **Error isolation** (one tool failure doesn't block others)

## Guard System (`guard/`)

Guards are behavioral constraints that fire at lifecycle points. They compose via `GuardRegistry`.

### Lifecycle Hooks

| Hook | When | Can Do |
|------|------|--------|
| `check_pre` | Before tool execution | Block, inject, escalate |
| `check_post` | After tool execution | Inject messages, force compact |
| `check_strategic` | At review points | Redirect plan |

### Guard Inventory

| Guard | Purpose |
|-------|---------|
| **SafetyGuard** | Block dangerous shell commands |
| **PlanGuard** | Enforce plan creation for complex tasks |
| **LoopDetectGuard** | Detect repeated failed tool patterns |
| **ContextPressureGuard** | Auto-compact when context is full |
| **TrainingRuntimeGuard** | Monitor running training jobs |
| **ConstraintGuard** | Enforce skill-defined constraints |
| **ErrorClassifierGuard** | Classify tool errors via Judge |
| **CircuitBreakerGuard** | Stop after N consecutive failures |
| **BudgetGuard** | Enforce per-turn resource limits |
| **EnvCompatGuard** | Environment compatibility checks |
| **ProgressGuard** | Detect stalled progress |
| **PlanUpdateGuard** | Remind agent to update plan steps |
| **ExperimentGuard** | Enforce experiment tracking for training runs |

### GuardVerdict Actions

- `allow` — proceed normally
- `block` — tool call NOT executed, message injected to LLM
- `inject_msg` — tool executes, but extra message appended
- `force_compact` — trigger context compaction
- `escalate` — hard block with user-facing alert
- `redirect` — change plan direction

## Judge System (`judge.py`)

Three-tier classification engine:

```
Request → FastClassifier (heuristics, 0 cost)
       → Cache (MD5-keyed, 0 cost)
       → LLM call (deep classification)
```

Categories include: `is_error`, `is_success`, `is_dangerous`, `is_read_only_shell`, `is_training_command`, `skill_suggest_by_context`, `is_constraint_violated`, etc.

**JudgeBudget**: max 64 LLM classify calls per turn to prevent runaway costs.

## Skill System (`skills/`)

Skills are domain-specific knowledge packs stored as `SKILL.md` files with YAML frontmatter.

```
skills/
  train-run/
    SKILL.md          # frontmatter + body
  train-config/
    SKILL.md
  ...
```

### Frontmatter Schema

```yaml
---
name: train-run
description: Launch and manage FlagScale training jobs
keywords: [training, launch, flagscale, torchrun]
parameters: []
requires: []          # Must-load dependencies
suggests: []          # Optional dependencies
workflow:             # Stage-based workflow
  - id: setup
    tools: [shell, read_file]
  - id: launch
    tools: [shell, monitor]
constraints:          # Hard rules for ConstraintGuard
  - id: no_main_push
    trigger: {tools: [shell]}
    prompt: "..."
    correction: "..."
---
# Body (markdown content injected into system prompt)
```

### Loading Flow

1. `load_skill` tool triggered by LLM or auto-suggested by Judge
2. SkillManager resolves name → file path (later dirs override)
3. Frontmatter parsed for workflow, constraints, keywords
4. Body content injected into system prompt via PromptBuilder
5. Constraints registered with ConstraintGuard

## Chip System (`chip/`)

Hardware-aware capability declarations:

- **ChipCapability**: Dataclass describing operator support, precision, communication backends, known issues
- **detect_chip()**: Probes environment (nvidia-smi, rocminfo, etc.)
- **CHIP_REGISTRY**: Maps vendor → ChipCapability instance
- **MigrationDiff**: Computes source→target differences for cross-chip porting

Currently supports: NVIDIA (full), with extensibility for other backends.

## Memory System (`memory.py`)

Cross-session persistent storage with TTL expiration.

```
.flagscale/memory/
  <key>.yaml          # One file per memory entry
```

### Entry Schema

```yaml
key: env_apex_build_fix
type: finding         # finding | decision | todo | context
content: "..."
task: qwen3_train
created: 1719000000
accessed: 1719100000
access_count: 3
priority: normal      # high | critical | normal | low
ttl: 2592000         # seconds (30 days default)
```

### Features

- **TTL expiration**: Entries auto-cleanup based on priority
- **Auto-promotion**: Frequently accessed entries promote normal → high
- **Semantic search**: LLM-powered fuzzy key matching when exact lookup fails
- **Deduplication**: LLM judges if new entry duplicates existing one
- **Keyword expansion**: LLM expands search keywords for better recall

## Session System (`session.py`)

Conversation persistence for resume/reload:

```
.flagscale/sessions/
  index.yaml            # Last 10 sessions summary
  <session_id>/
    conversation.json   # Full message history (atomic writes)
```

- Atomic write via tmp file + rename (crash-safe)
- Auto-finds resumable sessions (incomplete ones)
- `/resume` command to continue previous session
- Session metadata: loaded skills, timestamps, completion status

## Plan System (`plan.py`)

Step-based task tracking with YAML persistence:

```
.flagscale/plans/
  active.yaml           # Currently active plan
  archive/              # Completed/abandoned plans
```

### Plan Structure

```yaml
title: "Train Qwen3 0.6B on 8xA100"
steps:
  - id: 1
    description: "Set up environment"
    status: done        # pending | doing | done | skipped
    notes: ""
    experiments: []
  - id: 2
    description: "Prepare data"
    status: doing
```

## Prompt Builder (`prompt_builder.py`)

Assembles the system prompt dynamically per turn:

1. **Core prompt** (identity, capabilities, rules)
2. **Optional sections** (training, inference, planning — only when relevant)
3. **Skill summaries** (available skills list)
4. **Active skill content** (loaded skill bodies)
5. **Critical rules** (extracted `## CRITICAL` sections from skills)
6. **Situational context** (shared storage paths, hardware info)
7. **Memory context** (relevant memory entries)
8. **Plan context** (active plan status)

Optimized to minimize token waste — sections only included when relevant constraints are active.

## Display System (`display.py`)

Terminal UI with:
- Thread-safe output (lock-protected stdout)
- 256-color support with `NO_COLOR` env var respect
- Animated parallel tool execution display
- Spinner for LLM thinking state
- Progress indicators per tool with elapsed time
- Collapsible output for large tool batches

## Provider System (`providers/`)

Abstraction over LLM APIs:

- **Anthropic**: Claude models with tool_use, streaming, extended thinking
- **OpenAI**: GPT-4o and compatible APIs

Common interface: `call(messages, tools, stream=True) → response`

Handles: token counting, context limit detection, tool schema formatting.

## Configuration (`config.py`)

`AgentConfig` dataclass with:

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | "anthropic" | LLM provider |
| `model` | (per-provider) | Model name |
| `base_url` | None | API endpoint override |
| `context_window` | 200000 | Token budget |
| `confirm_commands` | True | Require user confirmation for shell |
| `shell_remind_interval` | 5 | Batch efficiency reminder interval |
| `dangerous_commands_check` | True | Enable safety guard for shell |

Supports: `AgentConfig.from_yaml()` and `AgentConfig.auto_load()`.

## End-to-End Flow: Interactive Session

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SESSION LIFECYCLE                                    │
└──────────────────────────────────────────────────────────────────────────────┘

 ┌─────────┐
 │  Start  │
 └────┬────┘
      │
      ▼
 ┌─────────────────────┐     ┌────────────────────────────┐
 │  Load Config        │────▶│ FLAGSCALE_AGENT_CONFIG      │
 │  (auto_load)        │     │ .flagscale/agent.yaml       │
 └─────────┬───────────┘     └────────────────────────────┘
           │
           ▼
 ┌─────────────────────┐     ┌────────────────────────────┐
 │  Init WorkerAgent   │────▶│ Register tools, guards,     │
 │                     │     │ skills, memory, judge       │
 └─────────┬───────────┘     └────────────────────────────┘
           │
           ▼
 ┌─────────────────────┐  /resume  ┌──────────────────────┐
 │  REPL Prompt        │─────────▶│ Restore session state │
 │  (prompt_toolkit)   │◀─────────┘                       │
 └─────────┬───────────┘                                   
           │ user input
           ▼
 ┌─────────────────────┐  /slash?  ┌──────────────────────┐
 │  Dispatch           │─────────▶│ CommandHandler         │
 │                     │◀─────────│ (local, no LLM call)  │
 └─────────┬───────────┘          └──────────────────────┘
           │ normal message
           ▼
 ┌─────────────────────┐
 │  Scene Detection    │  training / inference / migration?
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐  yes  ┌───────────────────────────┐
 │  Auto-Skill Suggest?│──────▶│ Judge: skill_suggest_by_  │
 │                     │       │ context → load_skill      │
 └─────────┬───────────┘       └───────────────────────────┘
           │
           ▼
 ┌─────────────────────────────────────────────────────────┐
 │                  AgentKernel ReAct Loop                  │
 │                                                         │
 │  ┌───────────────┐                                      │
 │  │ Build system  │  skills + memory + plan + scene      │
 │  │ prompt        │                                      │
 │  └──────┬────────┘                                      │
 │         │                                               │
 │         ▼                                               │
 │  ┌───────────────┐                                      │
 │  │  LLM Call     │  streaming, tool_use enabled         │
 │  └──────┬────────┘                                      │
 │         │                                               │
 │    ┌────┴────┐                                          │
 │    │ tools?  │                                          │
 │    └─┬────┬──┘                                          │
 │  no  │    │ yes                                         │
 │      │    ▼                                             │
 │      │  ┌──────────────────────────────────────┐        │
 │      │  │  Per tool_call:                      │        │
 │      │  │                                      │        │
 │      │  │  ┌─────────┐  block  ┌───────────┐  │        │
 │      │  │  │Guard PRE│───────▶│Skip + Msg  │  │        │
 │      │  │  └────┬────┘        └───────────┘  │        │
 │      │  │       │ allow                       │        │
 │      │  │       ▼                             │        │
 │      │  │  ┌─────────┐                        │        │
 │      │  │  │ Execute │  (parallel batch)      │        │
 │      │  │  └────┬────┘                        │        │
 │      │  │       │                             │        │
 │      │  │       ▼                             │        │
 │      │  │  ┌──────────┐ inject ┌───────────┐ │        │
 │      │  │  │Guard POST│──────▶│Advisory Msg│ │        │
 │      │  │  └────┬─────┘       └───────────┘ │        │
 │      │  │       │                            │        │
 │      │  │       ▼                            │        │
 │      │  │  Append result to history          │        │
 │      │  └──────────────────────────────────────┘        │
 │      │    │                                             │
 │      │    ▼                                             │
 │      │  ┌──────────────────┐                            │
 │      │  │ Context Pressure │  >80%? → auto compact     │
 │      │  └────────┬─────────┘                            │
 │      │           │                                      │
 │      │           ▼                                      │
 │      │  ┌──────────────────┐                            │
 │      │  │ Plan auto-cont?  │  next step? → loop        │
 │      │  └────────┬─────────┘                            │
 │      │           │                                      │
 │      │    ◀──────┘  (back to LLM Call)                  │
 │      │                                                  │
 │      ▼                                                  │
 │  ┌───────────────┐                                      │
 │  │ Final text    │  display to user                     │
 │  └───────────────┘                                      │
 └─────────────────────────────────────────────────────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Auto-save session  │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  REPL Prompt        │  wait for next input...
 └─────────────────────┘
```

## Design Principles

1. **Composition over inheritance**: Guards, tools, skills are independent components. No mixin trees.
2. **Data-driven behavior**: Scene presets and skill frontmatter parameterize agent behavior, not code branches.
3. **Safety by default**: Guards block dangerous actions; user confirmation required for mutations.
4. **Token efficiency**: Prompt sections conditional on relevance; context compaction when pressure is high.
5. **Crash resilience**: Atomic writes for sessions/plans; TTL cleanup for memory; circuit breakers for loops.
6. **Extensibility**: New tools, guards, skills, and chip definitions are additive — no existing code changes required.
