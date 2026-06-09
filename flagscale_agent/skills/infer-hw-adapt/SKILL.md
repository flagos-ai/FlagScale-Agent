---
name: infer-hw-adapt
description: Adapt and fix vllm-plugin-FL for specific hardware backends after plugin version
  upgrades. Covers the full test-patch-verify cycle from unit tests through serving,
  plus PR submission with squashed commits. Requires infer-env-setup to be completed first.
keywords:
- inference
- vllm
- hardware
- adaptation
- patch
- testing
- metax
- ascend
- moore-threads
- plugin
- pr
requires:
- infer-env-setup
suggests:
- infer-model-adapt
- debug-strategy
- ops-discipline
constraints:
- id: test_order
  description: Always run tests in order — unit → functional → offline → serving
  trigger:
    keywords: [functional test, offline inference, serving test]
  prompt: Check if the previous stage was completed and passed before this one
  correction: Fix all failures at the current stage before proceeding to the next.
- id: no_vllm_source_modification
  description: Never modify vLLM source code — all patches go through plugin
  trigger:
    tools: [edit_file, write_file]
    keywords: [site-packages/vllm, vllm/worker, vllm/model_runner]
  prompt: Check if the agent is editing vLLM source files
  correction: Create or modify plugin patch files in vllm_fl/ instead.
- id: persist_logs
  description: All test commands must tee output to /workspace/adapt-logs/
  trigger:
    tools: [shell]
    keywords: [pytest, python examples, vllm serve]
  prompt: Check if the test command includes 2>&1 | tee /workspace/adapt-logs/
  correction: Add `2>&1 | tee /workspace/adapt-logs/<stage>_$(date +%Y%m%d_%H%M%S).log` to the command.
- id: one_patch_per_failure
  description: Fix one failure at a time — patch, re-test, then move to next failure
  trigger:
    keywords: [also fix, fix all, patch multiple]
  prompt: Check if the agent is applying multiple unrelated patches at once
  correction: Fix one failure, verify it passes, then address the next failure.
- id: platform_gate
  description: All hardware-specific patches must be gated by platform check
  trigger:
    tools: [edit_file, write_file]
    keywords: [def forward, def __init__, ops.dispatch]
  prompt: Check if new hardware-specific code is wrapped in a platform check
  correction: Wrap with `if current_platform.is_<backend>():` or equivalent.
- id: todo_on_workaround
  description: Every temporary workaround must have a TODO comment
  trigger:
    tools: [edit_file, write_file]
    keywords: [workaround, temporary, hack, TODO]
  prompt: Check if the workaround has a TODO stating when it can be removed
  correction: Add `# TODO: Remove when <condition>` above the workaround.
- id: squash_before_pr
  description: All adaptation commits must be squashed into one before PR
  trigger:
    keywords: [git push, create pr, open pr]
  prompt: Check if commits have been squashed
  correction: Run `git rebase -i HEAD~N` to squash all adaptation commits into one.
context_injection:
  always:
  - Critical Rules
  - Test Progression
  by_tool:
    shell:
    - Stage 0 Workspace Orientation
    edit_file:
    - Version-Adaptive Patching
    - Copy-then-Patch Discipline
---
# Hardware Adaptation after Plugin Upgrade

Adapt and fix vllm-plugin-FL for specific hardware backends after each plugin version upgrade.

## When to Use This Skill

Every time vllm-plugin-FL upgrades its base vLLM version (e.g., 0.19 → 0.20), hardware-specific code paths may break because:
- vLLM internal APIs change (worker, model_runner, ops dispatch)
- New Triton kernels are introduced that the hardware's Triton backend doesn't support
- FlagGems op coverage may lag behind new vLLM requirements
- Plugin patch points may shift or become invalid

This skill covers the **adaptation and testing cycle** for one hardware backend per invocation. Environment setup (SSH, container, installation) is handled by `infer-env-setup`.

## Prerequisites

Before starting adaptation, ensure the environment is ready (via `infer-env-setup`):
- SSH connection confirmed
- Docker container running with correct image, device mounts, and workspace volume
- vLLM (CPU-only), vllm-plugin-FL (editable), and FlagGems installed
- All imports verified (`import vllm`, `import vllm_fl`, `import flag_gems`)

If any of these are not ready, run `infer-env-setup` first.

## Critical Rules

1. **Test in order**: unit → functional → offline inference → serving. Fix each stage before proceeding.
2. **Never modify vLLM source** — all hardware adaptations go through plugin patches.
3. **Stream and persist logs** — use `2>&1 | tee /workspace/adapt-logs/<stage>.log`; diagnose from log files, don't re-run commands.
4. **After tests pass, review all changes** — remove anything unnecessary before PR.
5. **One patch per failure** — fix one issue, re-test, then move to the next.
6. **Patches are hardware-gated** — use `if current_platform.is_<backend>()` or equivalent.
7. **Every workaround has a TODO** — state when it can be removed.
8. **Sync code before testing** — if editing locally, push changes to container before running tests.
9. **Check device occupancy before tests** — use the backend's monitoring tool to confirm devices are free.
10. **Use tmux for long-running commands** — SSH sessions will timeout otherwise.
11. **Workspace orientation first** — run Stage 0 probe before any test. Record plugin path, branch, vLLM version, and model path in memory. Never guess paths.
12. **Read once, memorize** — after reading any source file, record key findings with `memory_write`. Check `memory_read` before re-opening a file. Never read the same file twice unless it was modified.
13. **Squash before PR** — squash all adaptation commits into one clean commit with a comprehensive message. Remove debug tools before squashing.
14. **Batch independent tool calls** — when multiple shell commands, file reads, or memory operations are independent, execute them in one response.

---

## Test Progression

Run tests in strict order. Fix all failures at each stage before proceeding to the next.

### Stage 0: Workspace Orientation (MANDATORY)

Before any test, record all paths to memory to avoid path confusion:

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  echo \"=== plugin workspace ===\" &&
  find /workspace -name \"vllm_fl\" -type d 2>/dev/null | head -5 &&
  echo \"=== plugin git info ===\" &&
  for dir in \$(find /workspace -name \".git\" -type d 2>/dev/null | grep vllm-plugin-FL); do
    repo=\$(dirname \$dir)
    echo \"Repo: \$repo\"
    git -C \$repo branch --show-current
    git -C \$repo log -1 --oneline
  done &&
  echo \"=== vllm version ===\" &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  echo \"=== plugin installed ===\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  echo \"=== adapt-logs ===\" &&
  ls -lh /workspace/adapt-logs/ 2>/dev/null | tail -10 &&
  echo \"=== models ===\" &&
  ls -lh /workspace/models/ 2>/dev/null | head -10
'"
```

**Immediately record to memory:**
```
memory_write('<backend>_plugin_workspace', '<discovered_path>')
memory_write('<backend>_plugin_branch', '<branch_name>')
memory_write('<backend>_vllm_version', '<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
memory_write('<backend>_log_dir', '/workspace/adapt-logs')
```

### Stage 1: Unit Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL &&
  VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v \
  2>&1 | tee /workspace/adapt-logs/unit_$(date +%Y%m%d_%H%M%S).log
'"
```

Monitor with `duration=120`, `process_pattern="pytest"`. Unit tests complete in under 60s on most backends; if pytest dies the monitor returns immediately.

Purpose: verify import compatibility, API surface, basic plugin registration.

### Stage 2: Functional Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL &&
  VLLM_PLUGINS=fl pytest tests/functional_tests/ -x -v \
  2>&1 | tee /workspace/adapt-logs/functional_$(date +%Y%m%d_%H%M%S).log
'"
```

Monitor with `duration=300`, `process_pattern="pytest"`, `fail_pattern="FAILED|ERROR|hang|timeout"`. Functional tests can hang on graph capture failures — if a test hangs beyond 5 min, kill it and diagnose with `-k <test_name>`.

Purpose: verify operator correctness, kernel dispatch, dtype handling.

### Stage 3: Offline Inference

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL &&
  VLLM_PLUGINS=fl MODEL_PATH=/workspace/models/<model> TP_SIZE=2 \
  python examples/<model>_offline_inference.py \
  2>&1 | tee /workspace/adapt-logs/offline_$(date +%Y%m%d_%H%M%S).log
'"
```

Monitor with `duration=600`, `process_pattern="python"`, `success_pattern="Prompt.*Output:|Generated text:"`. Model loading takes 2–4 min on first run.

Purpose: full model execution without serving overhead. Validates model loading, forward pass, sampling.

### Stage 4: Serving Test

```bash
# Terminal 1 — start server
ssh <ssh_host> "docker exec <container> bash -c '
  VLLM_PLUGINS=fl vllm serve /workspace/models/<model> \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --trust-remote-code \
  2>&1 | tee /workspace/adapt-logs/serving_$(date +%Y%m%d_%H%M%S).log
'"

# Terminal 2 — send request after server is ready
ssh <ssh_host> "docker exec <container> bash -c '
  curl -s http://localhost:8000/v1/completions \
  -H \"Content-Type: application/json\" \
  -d \"{\\\"model\\\": \\\"/workspace/models/<model>\\\",
       \\\"prompt\\\": \\\"Hello, world\\\",
       \\\"max_tokens\\\": 20}\"
'"
```

Monitor serving with `success_pattern="Application startup complete"`, `duration=300`.

---

## Version-Adaptive Patching

The key principle: **detect first, patch only what's actually missing in the installed version.**

```bash
# Check if an API exists before patching
ssh <ssh_host> "docker exec <container> python3 -c \
  'from vllm.worker import Worker; print(dir(Worker))'"
```

**Common breakage patterns and fixes:**

| Breakage | Symptom | Fix |
|---|---|---|
| Worker API change | `AttributeError` on worker init | Update `vllm_fl/worker/<backend>_worker.py` |
| New Triton kernel | `RuntimeError: Triton not supported` | Gate with `if not current_platform.is_<backend>()` |
| FlagGems op missing | `NotImplementedError` for op | Add fallback or report to FlagGems upstream |
| Attention backend change | Wrong attention output | Update `vllm_fl/attention/<backend>_attn.py` |
| Model runner API shift | `TypeError` on runner call | Update `vllm_fl/model_runner/<backend>_runner.py` |

---

## Copy-then-Patch Discipline

1. **Copy first**: `cp <vllm_source>/<file>.py <plugin>/vllm_fl/<file>.py` — verbatim, no edits
2. **Patch via targeted edits**: use `edit_file` for specific lines, not `write_file` for the whole file
3. **One category at a time**: group related changes, verify import, then continue
4. **Import check after each batch**: `python3 -c "from vllm_fl.X import Y; print('OK')"`
5. **Never rewrite from scratch**: if tempted to rewrite, stop and read more upstream code first

---

## Stage 5: Clean-Up Before PR

Review all changes before squashing:

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL &&
  git diff main
'"
```

### Clean-Up Checklist

- [ ] Every patch has a `# TODO: Remove when ...` comment
- [ ] No debug prints, temporary hacks, or commented-out code
- [ ] Patches are gated by `if current_platform.is_<backend>()`
- [ ] `git diff main` reviewed — only necessary changes remain
- [ ] No passwords, tokens, or API keys in code or comments
- [ ] No hardcoded IP addresses or internal hostnames

---

## Stage 6: PR Submission

### Squash commits

```bash
# Count commits since branch point
git log --oneline main..HEAD | wc -l

# Squash all into one
git rebase -i HEAD~<N>
# In editor: keep first as 'pick', change rest to 'squash'
```

### Commit message template

```
adapt(<backend>): vllm-plugin-FL compatibility for vLLM <version>

Backend: <backend name> (<hardware model>)
vLLM version: <version>

Changes:
- <file>: <what changed and why>
- <file>: <what changed and why>

Test results:
- Unit tests: PASS (<N> tests)
- Functional tests: PASS (<N> tests)
- Offline inference: PASS (<model>, TP=<N>)
- Serving: PASS (throughput: <X> tok/s)

FlagGems missing ops (need native implementation):
- <op_name>: currently falls back to PyTorch

Generated with FlagScale-Agent
```

### PR description sections

1. **Target backend and vLLM version**
2. **Reason for each patch** (what broke and why)
3. **Test results summary** (which stages pass, key metrics)
4. **FlagGems needed-ops list** (ops that currently fall back to PyTorch)
5. **Agent attribution footer**

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-model-adapt` — port a new model into vllm-plugin-FL
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts

---
Related skills (load if needed): `debug-strategy`, `ops-discipline`
### Stage 0: Workspace Orientation (MANDATORY)

Before any test, record all paths to memory to avoid path confusion:

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  echo \"=== plugin workspace ===\" &&
  find /workspace -name \"vllm_fl\" -type d 2>/dev/null | head -5 &&
  echo \"=== plugin git info ===\" &&
  for dir in \$(find /workspace -name \".git\" -type d 2>/dev/null | grep vllm-plugin-FL); do
    repo=\$(dirname \$dir)
    echo \"Repo: \$repo\"
    git -C \$repo branch --show-current
    git -C \$repo log -1 --oneline
  done &&
  echo \"=== vllm version ===\" &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  echo \"=== plugin installed ===\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  echo \"=== adapt-logs ===\" &&
  ls -lh /workspace/adapt-logs/ 2>/dev/null | tail -10 &&
  echo \"=== models ===\" &&
  ls -lh /workspace/models/ 2>/dev/null | head -10
'"
```

Immediately record to memory:
```
memory_write('<backend>_plugin_workspace', '<discovered_path>')
memory_write('<backend>_plugin_branch', '<branch_name>')
memory_write('<backend>_vllm_version', '<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
memory_write('<backend>_log_dir', '/workspace/adapt-logs')
```

**Never guess paths. Always read from memory or re-probe.**

### Stage 1: Unit Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_path> && \
   VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v \
   2>&1 | tee /workspace/adapt-logs/unit_$(date +%Y%m%d_%H%M%S).log'"
```

Monitor with `duration=120`, `process_pattern="pytest"`. Unit tests complete in under 60s.
If pytest dies the monitor returns immediately rather than waiting out the full timeout.

**Purpose**: verify import compatibility, API surface, basic plugin registration.

### Stage 2: Functional Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_path> && \
   VLLM_PLUGINS=fl pytest tests/functional_tests/ -x -v \
   2>&1 | tee /workspace/adapt-logs/functional_$(date +%Y%m%d_%H%M%S).log'"
```

Monitor with `duration=300`, `process_pattern="pytest"`, `fail_pattern="FAILED|ERROR|hang|timeout"`.
If a test hangs beyond 5 min, kill it and diagnose the specific test with `-k <test_name>`.

**Purpose**: verify operator correctness, kernel dispatch, dtype handling.

### Stage 3: Offline Inference

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_path> && \
   VLLM_PLUGINS=fl MODEL_PATH=<model_path> TP_SIZE=2 \
   python examples/<model>_offline_inference.py \
   2>&1 | tee /workspace/adapt-logs/offline_$(date +%Y%m%d_%H%M%S).log'"
```

Monitor with `duration=600`, `process_pattern="python"`, `success_pattern="Prompt.*Output:|Generated text:"`.
Model loading takes 2–4 min on first run.

**Purpose**: full model execution without serving overhead. Validates model loading, forward pass, sampling.

### Stage 4: Serving Test

```bash
# Terminal 1: start server
ssh <ssh_host> "docker exec <container> bash -c \
  'VLLM_PLUGINS=fl vllm serve <model_path> \
   --tensor-parallel-size 2 \
   --enforce-eager \
   --trust-remote-code \
   2>&1 | tee /workspace/adapt-logs/serving_$(date +%Y%m%d_%H%M%S).log'"

# Terminal 2: test request (after server is ready)
ssh <ssh_host> "docker exec <container> curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"Hello, world\", \"max_tokens\": 20}'"
```

Monitor server log with `success_pattern="Uvicorn running on"`, `fail_pattern="Error|Traceback"`.

**Purpose**: validate the full serving stack under real HTTP load.

---

## Version-Adaptive Patching

The key principle: **detect first, patch only what's actually missing in the installed version.**

```bash
# Check what API exists before patching
ssh <ssh_host> "docker exec <container> python3 -c \
  'import vllm.worker.worker as w; print(dir(w.Worker))'"
```

Patch categories by typical vLLM upgrade impact:

| Category | What breaks | Plugin patch location |
|---|---|---|
| Worker API | `determine_num_available_blocks` signature | `vllm_fl/worker/` |
| Model runner | `capture_model` / graph capture hooks | `vllm_fl/model_runner/` |
| Ops dispatch | New Triton ops without backend support | `vllm_fl/ops/` |
| Attention backend | KV cache layout changes | `vllm_fl/attention/` |
| Sampling | New sampler params | `vllm_fl/sampling/` |

---

## Copy-then-Patch Discipline

1. **Copy first**: `cp <vllm_source>/<file>.py <plugin>/vllm_fl/<path>/<file>.py` — verbatim, no edits
2. **Patch via targeted edits**: use `edit_file` for specific lines, not `write_file` for the whole file
3. **One category at a time**: group related changes, verify import, then continue
4. **Import check after each batch**: `python3 -c "from vllm_fl.models.X import Y; print('OK')"`
5. **Never rewrite from scratch**: if tempted to rewrite, stop and read more upstream code first

---

## Stage 5: Clean-Up Checklist

Before squashing commits and opening PR:

### Code Quality
- [ ] No debug `print()` statements left in any file
- [ ] No commented-out code blocks
- [ ] No temporary `import pdb; pdb.set_trace()` or similar
- [ ] Every patch has a `# TODO: Remove when ...` comment
- [ ] Patches are gated by platform check (`if current_platform.is_<backend>()`)
- [ ] `git diff main` reviewed — only necessary changes remain

### Sensitive Content
- [ ] No passwords, tokens, or API keys in code or comments
- [ ] No SSH config, private keys, or `.pem` file paths committed
- [ ] No hardcoded IP addresses or internal hostnames
- [ ] Run `git diff main | grep -iE '(password|token|secret|pem|private_key|proxy)'` — should return nothing

### Commits & PR
- [ ] All commits squashed into one clean commit
- [ ] Commit message describes all changes with *what* and *why*
- [ ] Commit message includes test results summary
- [ ] Branch name follows convention: `adapt/<backend>-vllm-<version>`

---

## Stage 6: PR Submission

### Squash commits

```bash
# Count commits since branching from main
git log --oneline main..HEAD | wc -l

# Squash all into one
git rebase -i HEAD~<N>
# In the editor: keep first as 'pick', change rest to 'squash'
```

### Commit message template

```
adapt(<backend>): vllm-plugin-FL v<version> hardware adaptation

Backend: <MetaX C550 | Ascend 910B | Moore Threads S4000>
vLLM version: <X.Y.Z>

Changes:
- <file>: <what changed and why>
- <file>: <what changed and why>

Test results:
- Unit tests: PASS (<N> tests)
- Functional tests: PASS (<N> tests)
- Offline inference: PASS (<model>, TP=2)
- Serving: PASS (throughput: X tok/s)

FlagGems ops needing native impl (currently falling back to PyTorch):
- <op_name>: needed for <use case>

🤖 Generated with FlagScale-Agent
```

### Push and open PR

```bash
git push -u origin adapt/<backend>-vllm-<version>
```

Then open PR targeting `main` branch of `flagos-ai/vllm-plugin-FL`.

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-model-adapt` — port a new model into vllm-plugin-FL
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts

---
Related skills (load if needed): `debug-strategy`, `ops-discipline`
## Stage 6: PR Submission

### Squash commits

```bash
# Count your adaptation commits
git -C <plugin_path> log --oneline main..HEAD | wc -l

# Squash into one
git -C <plugin_path> rebase -i HEAD~<N>
# In editor: mark first commit as 'pick', rest as 'squash'
```

### Commit message template

```
adapt(<backend>-vllm-<version>): hardware adaptation for <Backend> on vLLM <version>

## What changed

- <file1>: <what and why>
- <file2>: <what and why>

## Why

<Root cause: what broke and why after vLLM upgrade>

## Test results

- Unit tests: PASS (N/N)
- Functional tests: PASS (N/N)
- Offline inference: PASS — <model>, TP=<N>, throughput=<X> tok/s
- Serving: PASS — <model>, first-token latency=<X>ms

## FlagGems missing ops (for upstream)

- op_name_1: falls back to PyTorch, needs native <backend> implementation
- op_name_2: ...

---
Co-authored-by: FlagScale-Agent <agent@flagos.ai>
```

### Push and open PR

```bash
git -C <plugin_path> push origin adapt/<backend>-vllm-<version> -u

# Then open PR via GitHub UI or gh CLI:
gh pr create \
  --title "adapt(<backend>): vLLM <version> hardware adaptation" \
  --body-file /tmp/pr_body.md \
  --base main
```

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-model-adapt` — port a new model into vllm-plugin-FL
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts

---
Related skills (load if needed): `debug-strategy`, `ops-discipline`
