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
- musa
- iluvatar
- sunrise
- hygon
- ppu
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
  correction: "Add `2>&1 | tee /workspace/adapt-logs/<stage>_$(date +%Y%m%d_%H%M%S).log` to the command."
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
  correction: "Wrap with `if current_platform.is_<backend>():` or equivalent."
- id: todo_on_workaround
  description: Every temporary workaround must have a TODO comment
  trigger:
    tools: [edit_file, write_file]
    keywords: [workaround, temporary, hack]
  prompt: Check if the workaround has a TODO stating when it can be removed
  correction: "Add `# TODO: Remove when <condition>` above the workaround."
- id: squash_before_pr
  description: All adaptation commits must be squashed into one before PR
  trigger:
    keywords: [git push, create pr, open pr]
  prompt: Check if commits have been squashed
  correction: "Run `git rebase -i HEAD~N` to squash all adaptation commits into one."
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

# Hardware Adaptation after Plugin Upgrade

Adapt and fix vllm-plugin-FL for specific hardware backends after each plugin version upgrade. When vLLM bumps its base version (e.g., 0.19 -> 0.20), hardware-specific code paths break because internal APIs shift, new Triton kernels are introduced, or FlagGems op coverage lags.

## Prerequisites

Before starting adaptation, ensure the environment is ready (via `infer-env-setup`):
- SSH connection confirmed
- Docker container running with correct image, device mounts, and workspace volume
- vLLM (CPU-only), vllm-plugin-FL (editable), and FlagGems installed
- All imports verified (`import vllm`, `import vllm_fl`, `import flag_gems`)

## Critical Rules

1. **Test in order**: unit -> functional -> offline inference -> serving. Fix each stage before proceeding.
2. **Never modify vLLM source** -- all patches go through plugin files in `vllm_fl/`.
3. **Stream and persist logs** -- use `2>&1 | tee /workspace/adapt-logs/<stage>.log`.
4. **One patch per failure** -- fix, re-test, then move to the next.
5. **Patches are hardware-gated** -- use `if current_platform.is_<backend>()` or vendor_name check.
6. **Every workaround has a TODO** -- state when it can be removed.
7. **Squash before PR** -- squash all adaptation commits into one clean commit.

---

## Test Progression

Run tests in strict order. Fix all failures at each stage before proceeding to the next.

### Stage 0: Workspace Orientation (MANDATORY)

Before any test, record all paths to memory to avoid path confusion:

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  find /workspace -name vllm_fl -type d 2>/dev/null | head -5 &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  ls -lh /workspace/adapt-logs/ 2>/dev/null | tail -10 &&
  ls -lh /workspace/models/ 2>/dev/null | head -10
'"
```

Immediately record to memory:
```
memory_write('<backend>_plugin_workspace', '<discovered_path>')
memory_write('<backend>_vllm_version', '<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
```

**Never guess paths. Always read from memory or re-probe.**

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

Monitor with `duration=300`, `process_pattern="pytest"`, `fail_pattern="FAILED|ERROR|hang|timeout"`. If a test hangs beyond 5 min, kill it and diagnose with `-k <test_name>`.

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

Monitor with `duration=600`, `process_pattern="python"`, `success_pattern="Prompt.*Output:|Generated text:"`. Model loading takes 2-4 min on first run.

Purpose: full model execution without serving overhead. Validates model loading, forward pass, sampling.

### Stage 4: Serving Test

```bash
# Terminal 1: start server
ssh <ssh_host> "docker exec <container> bash -c '
  VLLM_PLUGINS=fl vllm serve /workspace/models/<model> \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  2>&1 | tee /workspace/adapt-logs/serving_$(date +%Y%m%d_%H%M%S).log
'"

# Terminal 2: test request (after "Uvicorn running on" appears)
ssh <ssh_host> "docker exec <container> curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"/workspace/models/<model>\", \"prompt\": \"Hello\", \"max_tokens\": 20}'"
```

Monitor server log with `success_pattern="Uvicorn running on"`, `fail_pattern="Error|Traceback"`.

Purpose: validate the full serving stack under real HTTP load.

---

## Version-Adaptive Patching

The key principle: **detect first, patch only what's actually missing in the installed version.**

```bash
# Check what API exists before patching
ssh <ssh_host> "docker exec <container> python3 -c \
  'import vllm.worker.worker as w; print(dir(w.Worker))'"
```

**Common breakage patterns and fixes:**

| Breakage | Symptom | Fix |
|---|---|---|
| Worker API change | `AttributeError` on worker init | Update `vllm_fl/worker/<backend>_worker.py` |
| New Triton kernel | `RuntimeError: Triton not supported` | Gate with `if not current_platform.is_<backend>()` |
| FlagGems op missing | `NotImplementedError` for op | Add fallback or report to FlagGems upstream |
| Attention backend change | Wrong attention output | Update `vllm_fl/attention/<backend>_attn.py` |
| Model runner API shift | `TypeError` on runner call | Update `vllm_fl/model_runner/<backend>_runner.py` |
| Platform detection wrong | Device routed to wrong code path | Set `is_cuda_alike()`, `use_custom_allreduce()` correctly in `vllm_fl/platform.py` |
| Patch applied twice | Double-patched function crashes or loops | Add idempotency guard (`_patches_applied` flag) at top of `apply_patches()` |
| `torch.accelerator` API missing | `AttributeError` on `torch.accelerator.*` | Stub missing attrs in `patch.py` with backend-specific equivalents |
| Ops registry conflict | `c10::Error: duplicate op registration` | Pre-load vendor C extension before registering schemas in `_C_ops_registry.py` |
| CUDAGraph capture timeout | Server killed during graph capture | Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` to a large value (e.g., 7200) |
| FlagGems fast-path on wrong backend | Wrong output on non-CUDA hardware | Restrict FlagGems-routed ops to `current_platform.is_cuda()` only |
| New MoE dispatch point | `NotImplementedError: invoke_fused_moe_triton_kernel` | Register vendor MoE kernel in `vllm_fl/dispatch/backends/vendor/<backend>/register_ops.py` |

---

## Copy-then-Patch Discipline

1. **Copy first**: `cp <vllm_source>/<file>.py <plugin>/vllm_fl/<path>/<file>.py` -- verbatim, no edits
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
- [ ] `git diff main` reviewed -- only necessary changes remain

### Sensitive Content
- [ ] No passwords, tokens, or API keys in code or comments
- [ ] No SSH config, private keys, or `.pem` file paths committed
- [ ] No hardcoded IP addresses or internal hostnames
- [ ] Run `git diff main | grep -iE '(password|token|secret|pem|private_key|proxy)'` -- should return nothing

### Commits & PR
- [ ] All commits squashed into one clean commit
- [ ] Commit message describes all changes with *what* and *why*
- [ ] Commit message includes test results summary
- [ ] Branch name follows convention: `adapt/<backend>-vllm-<version>`

---

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
- Offline inference: PASS -- <model>, TP=<N>
- Serving: PASS -- <model>, TP=<N>, throughput=<X> tok/s

## FlagGems missing ops (for upstream)

- op_name_1: falls back to PyTorch, needs native <backend> implementation

---
Co-authored-by: FlagScale-Agent <agent@flagos.ai>
```

### Push and open PR

```bash
git -C <plugin_path> push origin adapt/<backend>-vllm-<version> -u

gh pr create \
  --title "adapt(<backend>): vLLM <version> hardware adaptation" \
  --body-file /tmp/pr_body.md \
  --base main
```

---

## Related Skills

- `infer-env-setup` -- environment setup (SSH, container, installation)
- `infer-model-adapt` -- port a new model into vllm-plugin-FL
- `debug-strategy` -- systematic debugging when tests fail repeatedly
- `ops-discipline` -- shell safety and environment awareness
- `workspace-layout` -- shared storage paths for models and artifacts

---
Related skills (load if needed): `debug-strategy`, `ops-discipline`
