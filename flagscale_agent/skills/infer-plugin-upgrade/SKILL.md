---
name: infer-plugin-upgrade
description: Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version
  detection, API diff analysis, targeted fixes, and validation across unit tests, offline
  inference, and serving. Applies to any vLLM minor version bump (e.g., 0.20.x → 0.24.x).
triggers:
  - vllm-plugin-FL upgrade
  - plugin version bump
  - vllm version upgrade
  - plugin API breakage after vllm update
---

## Critical Rules

1. **Auto-detect versions first** — never assume plugin or vLLM version. Always read from installed packages.
2. **Never modify vLLM source** — all fixes go through `vllm_fl/` plugin files only.
3. **One patch per failure** — fix, re-test, then move to the next error. Never batch unverified fixes.
4. **Fix order matters**: imports → class/factory API → signature kwargs → op schemas → model-specific.
5. **NVIDIA A800 is ground truth** — validate every fix on A800 before declaring done.
6. **Stream and persist logs** — use `2>&1 | tee /nfs/wlx/tmp/<stage>_<timestamp>.log`.
7. **Squash before PR** — all upgrade commits squashed into one clean commit.

---

## Step 0: Version Detection & Workspace Orientation

Run before ANY work. Never skip.

```bash
ssh <host> "docker exec <container> bash -c '
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  cat /nfs/wlx/adapt/nvidia-vllm-*/vllm-plugin-FL/pyproject.toml | grep -E \"^version|vllm\" | head -10
'"
```

Record to memory immediately:
```
memory_write('nvidia_vllm_version', 'X.Y.Z')          # installed vllm
memory_write('nvidia_plugin_version', 'A.B.C')         # plugin version
memory_write('nvidia_plugin_root', '/nfs/wlx/adapt/nvidia-vllm-X.Y.Z/vllm-plugin-FL')
memory_write('nvidia_vllm_root', '/nfs/wlx/code/vllm-X.Y.Z')
```

### Detect version gap

```bash
ssh <host> "docker exec <container> bash -c '
  # what vllm version does the plugin declare it supports?
  grep -r \"vllm\" /nfs/wlx/adapt/nvidia-vllm-*/vllm-plugin-FL/pyproject.toml | grep -i \"requires\|version\"
  # what is actually installed?
  python3 -c \"import vllm; print(vllm.__version__)\"
'"
```

If installed vllm version ≠ plugin's declared compatible version → version gap confirmed, proceed with upgrade.

---

## Step 1: API Diff Analysis

Before touching any code, enumerate what changed between the old and new vLLM versions.

### 1a. Check which plugin files import from vllm directly

```bash
ssh <host> "docker exec <container> grep -r 'from vllm\|import vllm' \
  /nfs/wlx/adapt/nvidia-vllm-X.Y.Z/vllm-plugin-FL/vllm_fl/ \
  --include='*.py' -l"
```

### 1b. Run unit tests to get the baseline error list

```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  /nfs/wlx/envs/vllm-fl-0.24.0/bin/python -m pytest \
  <plugin_root>/tests/unit_tests/ -x --tb=short \
  2>&1 | tee /nfs/wlx/tmp/unit_baseline_$(date +%Y%m%d_%H%M%S).log"
```

Collect all `ImportError`, `AttributeError`, `TypeError` from unit tests — these are the API breakages to fix.

### 1c. Check _C_cache_ops op availability

Write a probe script to find which plugin-declared ops are missing from the installed vLLM:

```python
# /nfs/wlx/tmp/check_ops.py
import torch, re, sys
sys.path.insert(0, '<plugin_root>')
from vllm_fl.ops._C_ops_schemas import SCHEMAS

native_ops = set(dir(torch.ops._C_cache_ops)) | set(dir(torch.ops._C))
schema_names = set(re.split(r'[\(.]', sc)[0].strip() for sc in SCHEMAS)

missing = schema_names - native_ops
present = schema_names & native_ops
print(f'Plugin schemas missing from native vllm ({len(missing)}):')
for n in sorted(missing): print(' ', n)
print(f'\nPlugin schemas present in native vllm ({len(present)}):')
for n in sorted(present): print(' ', n)
```

**Missing ops = ops that need NVIDIA backend implementation in the plugin.**
Note: FlagGems covers compute kernels (matmul, attention, elementwise). It does NOT cover `_C_cache_ops` — those are vLLM's paged KV cache management ops and must be implemented in the plugin.

### 1d. Known API change patterns by area

| Area | What to check | Common breakage |
|------|--------------|-----------------|
| `FusedMoE` | Is it a class or factory in new vllm? | Factory returns existing instance → recursive call in plugin's `__init__` |
| `InputBatch` | `__init__` signature | New kwargs added that plugin passes but vllm doesn't accept |
| `use_uniform_kv_cache` | Signature | Stale kwargs like `cache_dtype` removed |
| `_C_cache_ops` | Op list vs plugin schemas | Model-specific ops (MLA, DSA) may not be in base vllm |
| `WorkerProc` | `worker_main` vs `worker_busy_loop` | Entry point renamed between versions |
| `parallel_state` | `world_size`, `rank` init API | Signature changes in distributed init |

---

## Step 2: Fix API Breakages

Fix in this order. After each fix, run the relevant test before moving on.

### 2a. Import errors

If `from vllm.X import Y` fails → check new vllm for where `Y` moved:

```bash
ssh <host> "docker exec <container> python3 -c \"
import subprocess
r = subprocess.run(['grep', '-r', 'class Y\|def Y', '/nfs/wlx/code/vllm-X.Y.Z/vllm/'],
    capture_output=True, text=True)
print(r.stdout[:3000])
\""
```

Update the import in the plugin file. Never add a `sys.path` hack.

### 2b. FusedMoE factory/class change

**Symptom**: `RecursionError` or `TypeError: FusedTopKRouter.__init__() got unexpected keyword argument 'moe_config'`

**Root cause**: vLLM changed `FusedMoE` from a class to a factory function (or changed its `__init__` signature). The plugin's `FusedMoEFL` subclass or factory wrapper calls the parent in a way that no longer works.

**Fix pattern** (vllm 0.24.0 approach):
1. Save a reference to the original class before any monkey-patching:
   ```python
   _OrigFusedMoE = _fused_moe_pkg.FusedMoE  # capture before patching
   ```
2. Use `replace_router_with_fl()` to monkey-patch the router on existing instances rather than subclassing:
   ```python
   def replace_router_with_fl(module):
       # replace module.router with FusedTopKRouterFL instance
       ...
   ```
3. Apply the replacement in `FusedMoEFL.__init__` after calling `_OrigFusedMoE.__init__`.

File: `vllm_fl/ops/fused_moe/layer.py`

### 2c. InputBatch signature mismatch

**Symptom**: `TypeError: InputBatch.__init__() got an unexpected keyword argument 'pin_memory'`

**Root cause**: Plugin's `model_runner.py` was ported from a newer vLLM version that has `pin_memory`/`is_spec_decode` in `InputBatch.__init__`, but the installed vLLM doesn't.

**Fix**: Add a shim that filters kwargs to only what the installed vLLM accepts:

```python
# In vllm_fl/worker/model_runner.py, near InputBatch construction:
import inspect as _inspect
_InputBatch_params = set(_inspect.signature(InputBatch.__init__).parameters)

def _make_input_batch(*args, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _InputBatch_params}
    return InputBatch(*args, **filtered)
```

### 2d. use_uniform_kv_cache stale kwargs

**Symptom**: `TypeError: use_uniform_kv_cache() got an unexpected keyword argument 'cache_dtype'`

**Fix**: Remove the stale kwarg from the call site in `vllm_fl/worker/model_runner.py`. Check the new signature:

```bash
ssh <host> "docker exec <container> grep -n 'def use_uniform_kv_cache' \
  /nfs/wlx/code/vllm-X.Y.Z/vllm/v1/worker/gpu_model_runner.py"
```

### 2e. Missing _C_cache_ops (model-specific)

**Symptom**: `AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute 'concat_and_cache_mla'`

**Root cause**: Model-specific KV cache ops (e.g., GLM-5.2 DSA/MLA) are registered in `vllm_fl/ops/_C_ops_schemas.py` but have no NVIDIA backend implementation.

**Fix**: Implement the missing op in plugin's CUDA extension or provide a pure-PyTorch fallback:
- Schema location: `vllm_fl/ops/_C_ops_schemas.py`
- Reference: upstream vllm's `_custom_ops.py` for the function wrapper pattern
- For MLA-specific ops: reference the existing MetaX implementation as algorithmic guide

---

## Step 3: Unit Test Verification

After all API fixes, run unit tests and establish the new baseline:

```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  /nfs/wlx/envs/vllm-fl-0.24.0/bin/python -m pytest \
  <plugin_root>/tests/unit_tests/ -v --tb=short \
  2>&1 | tee /nfs/wlx/tmp/unit_after_fix_$(date +%Y%m%d_%H%M%S).log"
```

**Acceptable**: pre-existing failures that were also present before the upgrade (document these).
**Not acceptable**: new failures introduced by the upgrade fixes.

---

## Step 4: Offline Inference Validation (NVIDIA A800)

Test each model in the validation matrix. Run models roughly in order of architectural complexity: dense → MoE → Mamba hybrid → multi-modal → multi-node.

### Launch pattern (single node)

```bash
ssh <host> "docker exec -d \
  -e CUDA_HOME=/usr/local/cuda \
  -e PATH=/nfs/wlx/envs/vllm-fl-0.24.0/bin:/usr/local/cuda/bin:/usr/bin:/usr/local/bin \
  -e PYTHONPATH=<plugin_root> \
  -e CC=/usr/bin/gcc \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PLUGINS=fl \
  <container> bash /nfs/wlx/tmp/run_<model>.sh"
```

### Launch pattern (multi-node, e.g. TP=16 across 2 nodes)

**Critical**: launch both nodes simultaneously — do NOT wait for rank0 before launching rank1. Gloo init has a short timeout; if rank1 is late, rank0 workers time out with `Connection refused`.

```bash
# Launch rank0 and rank1 in the same shell response (not sequentially)
ssh <host0> "docker exec -d -e ... <container> bash /nfs/wlx/tmp/run_rank0.sh && echo rank0_ok"
ssh <host1> "docker exec -d -e ... <container> bash /nfs/wlx/tmp/run_rank1.sh && echo rank1_ok"
```

### Monitor

After launch, wait 3-5 minutes then check logs:

```bash
ssh <host> "tail -20 /nfs/wlx/tmp/<model>_rank0.log"
```

**Success pattern**: `Uvicorn running on` or `Generated text:` or `Application startup complete`
**Failure pattern**: `EXIT:1`, `WorkerProc failed`, `AttributeError`, `RuntimeError`

### Common runtime failures and fixes

| Error | Root cause | Fix |
|-------|-----------|-----|
| `float8_e4m3fn` triton error on A800 | sm_80 doesn't support fp8e4nv in triton | Override dtype to `float8_e5m2` when `sm_major < 9` in `fp8_utils.py` |
| `broadcast_to` CPU→CUDA copy during CUDA graph | FlagGems `broadcast_to.py` creates CPU tensors | Use `pin_memory()` before `.to(device)` in broadcast_to (see FlagGems PR #4472) |
| `mm_kernel_general` out of resource: shared memory | FlagGems autotune config exceeds A800 shmem limit (166912 bytes) | Remove configs where `BLOCK_M*BLOCK_K*num_stages > 166912 / element_size` from `tune_configs.yaml` |
| `concat_and_cache_mla` AttributeError | MLA KV cache op missing in NVIDIA backend | Implement in plugin CUDA extension or pure-PyTorch fallback |
| `Connection refused` on Gloo | rank1 not started before rank0 Gloo timeout | Launch both nodes simultaneously |
| `EADDRINUSE` port busy | Previous process still running | `pkill -9 -f 'vllm\|python'` on both nodes, wait 8s, check `/proc/net/tcp` |

### Validation matrix

Build this table as models are tested. Target: all models in the matrix must reach ✅ PASS.

| Model | Type | TP | CUDA Graph | Result |
|-------|------|----|-----------|--------|
| (dense 1-7B) | Dense LLM | 1 | ✅ | |
| (MoE) | MoE LLM | 4 | ✅ | |
| (Mamba hybrid) | Mamba Dense | 4 | ✅ | |
| (VLM) | VLM | 1 | ✅ | |
| (large MoE, 2-node) | Dense MoE | 16 | — | |

---

## Step 5: FlagGems Integration Checks

FlagGems replaces compute kernels (matmul, attention, elementwise). It does NOT replace `_C_cache_ops`. After offline inference passes, verify FlagGems is actually being used:

```bash
ssh <host> "docker exec <container> grep -a 'FlagGems\|flag_gems' /nfs/wlx/tmp/<model>.log | head -5"
```

### FlagGems-specific issues to watch for

**broadcast_to CUDA graph bug** (models with `attention_bias=true` or `mlp_bias=true`):
- Symptom: `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`
- Fix: In `FlagGems/src/flag_gems/ops/broadcast_to.py`, replace bare `torch.tensor(..., device=x.device)` with `torch.tensor(...).pin_memory().to(x.device)` for the three constant tensors (~lines 147-155)
- Gate with `if x.device.type == flag_gems.device:`

**mm shared memory overflow on A800**:
- Symptom: `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 196608, Hardware limit: 166912`
- Fix: In FlagGems `tune_configs.yaml`, remove autotune configs where `BLOCK_M × BLOCK_K × num_stages × dtype_bytes > 166912`

---

## Step 6: Serving Validation

Test the full serving stack (APIServer + EngineCore + Workers) for at least one model:

```bash
ssh <host> "docker exec -d -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> \
  -e CUDA_HOME=/usr/local/cuda -e PATH=<env_bin>:/usr/local/cuda/bin:/usr/bin:/usr/local/bin \
  <container> /nfs/wlx/envs/vllm-fl-0.24.0/bin/python -m vllm.entrypoints.openai.api_server \
  --model <model_path> --tensor-parallel-size <tp> --port 8000 \
  2>&1 | tee /nfs/wlx/tmp/serve_<model>_$(date +%Y%m%d_%H%M%S).log &"

# Wait for startup (~2-3 min), then test
ssh <host> "docker exec <container> curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20}'"
```

Success: JSON response with `choices[0].text` containing a coherent completion.

---

## Step 7: PR

### Commit discipline

```bash
cd <plugin_root>
git add -p  # stage specific hunks, not git add .
git commit -m "feat: upgrade to vllm X.Y.Z compatibility"
# if multiple fix commits, squash before PR:
git rebase -i HEAD~N  # squash N commits
```

### PR description template

```markdown
## PR Category
Core

## PR Type
Improvements

## Description
Upgrades vllm-plugin-FL compatibility from vllm A.B.C to vllm X.Y.Z.

## Changes
- `vllm_fl/worker/model_runner.py`: [describe change]
- `vllm_fl/ops/fused_moe/layer.py`: [describe change]
- [other files]

## Testing
Hardware: A800 80GB × N, CUDA 12.x, vllm X.Y.Z, FlagGems (NVIDIA backend)

| Model | Type | TP | CUDA Graph | Result |
|-------|------|----|-----------|--------|
| ...   | ...  | .. | ..        | ✅ PASS |

## Upstream follow-ups
- [any bugs found in FlagGems or vllm during testing]
```

---

## Environment Variables (NVIDIA)

Always set these when launching inference in the container:

```bash
CUDA_HOME=/usr/local/cuda
PATH=/nfs/wlx/envs/vllm-fl-0.24.0/bin:/usr/local/cuda/bin:/usr/bin:/usr/local/bin
PYTHONPATH=<plugin_root>
CC=/usr/bin/gcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_PLUGINS=fl
# For multi-node:
GLOO_SOCKET_IFNAME=<nic_name>   # e.g. ens201
NCCL_SOCKET_IFNAME=<nic_name>
NCCL_DEBUG=INFO
```

---

## Diagnostic Commands

```bash
# Check GPU free before launch
ssh <host> "docker exec <container> nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader"

# Check port availability
ssh <host> "cat /proc/net/tcp | grep -c '7335'"  # 0x7335 = port 29501

# Kill stale processes
ssh <host> "docker exec <container> pkill -9 -f 'vllm|python'"

# Find errors in log (skip NCCL noise)
ssh <host> "docker exec <container> grep -a 'ERROR' /nfs/wlx/tmp/<model>.log | grep -v 'NCCL\|nccl\|fa_utils' | head -20"

# Check which ops are missing
ssh <host> "docker exec <container> python3 /nfs/wlx/tmp/check_ops.py"
```

---

Related skills (load if needed): `infer-hw-adapt`, `infer-model-adapt`, `debug-strategy`, `ops-discipline`
