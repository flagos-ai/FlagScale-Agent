---
name: infer-plugin-upgrade
description: Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version
  detection, API diff analysis, targeted fixes, and validation across unit tests, offline
  inference, and serving. Applies to any vLLM minor version bump (e.g., 0.20.x to 0.24.x).
triggers:
  - vllm-plugin-FL upgrade
  - plugin version bump
  - vllm version upgrade
  - plugin API breakage after vllm update
---

## Critical Rules

1. **Auto-detect versions first** -- never assume plugin or vLLM version. Always read from installed packages and pyproject.toml.
2. **Never modify vLLM source** -- all fixes go through `vllm_fl/` plugin files only.
3. **One patch per failure** -- fix, re-test, then move to the next error. Never batch unverified fixes.
4. **Fix order matters**: imports then class/factory API then signature kwargs then op schemas then model-specific.
5. **NVIDIA GPU is ground truth** -- validate every fix on real hardware before declaring done.
6. **Stream and persist logs** -- use `2>&1 | tee <log_dir>/<stage>_<timestamp>.log`.
7. **Squash before PR** -- all upgrade commits squashed into one clean commit.

---

## Step 0: Workspace Orientation and Version Detection

Run before ANY work. Never skip. All paths must be probed -- never assumed.

### 0a. SSH connection and container check

```bash
ssh <host> "hostname && docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"
```

Identify the running container for vllm-plugin-FL work. If no container exists, set one up following `infer-env-setup`.

### 0b. Locate plugin and vllm roots

```bash
ssh <host> "docker exec <container> bash -c '
  echo === vllm version === &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  echo === plugin location === &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  echo === plugin pyproject === &&
  find / -path \"*/vllm-plugin-FL/pyproject.toml\" 2>/dev/null | head -3 &&
  echo === vllm source root === &&
  python3 -c \"import vllm, os; print(os.path.dirname(vllm.__file__))\"
'"
```

Record to memory immediately:
```
memory_write('nvidia_vllm_version', 'X.Y.Z')
memory_write('nvidia_plugin_root', '<discovered_plugin_root>')
memory_write('nvidia_vllm_root', '<discovered_vllm_root>')
memory_write('nvidia_container', '<container_name>')
memory_write('nvidia_log_dir', '<log_dir>')
```

### 0c. Detect version gap

```bash
ssh <host> "docker exec <container> bash -c '
  cat <plugin_root>/pyproject.toml | grep -E \"vllm|version\" &&
  python3 -c \"import vllm; print(vllm.__version__)\"
'"
```

If installed vllm version != plugin declared compatible version, version gap is confirmed -- proceed with upgrade.

---

## Step 1: API Diff Analysis

Before touching any code, enumerate what changed between old and new vLLM versions.

### 1a. Find plugin files that import from vllm directly

```bash
ssh <host> "docker exec <container> grep -r 'from vllm\|import vllm' \
  <plugin_root>/vllm_fl/ --include='*.py' -l"
```

### 1b. Run unit tests to get the baseline error list

```bash
ssh <host> "docker exec \
  -e VLLM_PLUGINS=fl \
  -e PYTHONPATH=<plugin_root> \
  <container> \
  python3 -m pytest <plugin_root>/tests/unit_tests/ -x --tb=short \
  2>&1 | tee <log_dir>/unit_baseline_$(date +%Y%m%d_%H%M%S).log"
```

Collect all ImportError, AttributeError, TypeError -- these are the API breakages to fix.

### 1c. Check _C_cache_ops op availability

Write a probe script to find which plugin-declared ops are missing from installed vLLM:

```python
# check_ops.py -- run inside container with VLLM_PLUGINS=fl and plugin on PYTHONPATH
import torch, re, sys

native_ops = set(dir(torch.ops._C_cache_ops)) | set(dir(torch.ops._C))

from vllm_fl.ops._C_ops_schemas import SCHEMAS
schema_names = set(re.split(r'[\(.]', sc)[0].strip() for sc in SCHEMAS)

missing = schema_names - native_ops
present = schema_names & native_ops
print(f'Plugin schemas missing from native vllm ({len(missing)}):')
for n in sorted(missing): print(' ', n)
print(f'\nPlugin schemas present in native vllm ({len(present)}):')
for n in sorted(present): print(' ', n)
```

Key distinction: FlagGems covers compute kernels (matmul, attention, elementwise). It does NOT cover
`_C_cache_ops` -- those are vLLM's paged KV cache management ops and must be implemented in the plugin.

Missing ops fall into two categories:
- **Generic cache ops** (e.g., `reshape_and_cache`, `copy_blocks`): must be present for any model to run.
  If missing, the plugin is broken for this vllm version.
- **Model-specific ops** (e.g., `concat_and_cache_mla` for a specific model's attention variant): only
  blocks that model. Other models run fine without them.

### 1d. Breakage-prone areas to audit

Every vLLM minor bump changes something. Before writing any fix, audit these areas by reading both the
plugin code and the new vllm source side by side:

| Plugin file | What to audit in new vllm | Why it commonly breaks |
|---|---|---|
| `vllm_fl/ops/fused_moe/layer.py` | Is `FusedMoE` still a class or now a factory? What is `FusedTopKRouter.__init__` signature? | vllm toggles between class and factory; subclassing or kwarg forwarding breaks silently |
| `vllm_fl/worker/model_runner.py` | `InputBatch.__init__` params, `use_uniform_kv_cache` signature, `WorkerProc` entry point name | Plugin often lags behind by 1-2 vllm versions; new params appear or old ones are removed |
| `vllm_fl/ops/_C_ops_schemas.py` | Run check_ops.py (Step 1c) to diff registered schemas vs installed ops | vllm reorganizes C extensions; model-specific ops may never be in base vllm |
| Any file with `from vllm.X import Y` | Does that import path still exist in new vllm? | vllm moves symbols between modules frequently |
| `vllm_fl/worker/` distributed code | `parallel_state` init API, `world_size`/`rank` call signatures | Distributed init API evolves across versions |

The specific breakages you encounter depend entirely on the version gap. Do not assume the same bugs will
appear across upgrades -- read the actual error, trace it to the changed vllm code, then fix.

---

## Step 2: Fix API Breakages

The workflow for every breakage is the same:
1. Read the error traceback -- identify which plugin file and line calls into vllm
2. Read the new vllm source at that call site to understand what changed
3. Apply the minimal fix
4. Verify with an import check or targeted test before moving to the next error

**Fix strategies by error type**:

### TypeError: unexpected keyword argument

The plugin passes a kwarg that no longer exists in the new vllm signature.

```bash
# Find the new signature
ssh <host> "docker exec <container> grep -rn 'def <function_name>' <vllm_root>/vllm/"
```

Options:
- Remove the stale kwarg if it was genuinely dropped upstream
- Use `inspect.signature()` to filter kwargs dynamically if the plugin must support multiple vllm versions

### ImportError or AttributeError on import

A symbol moved between vllm modules.

```bash
# Find where it moved
ssh <host> "docker exec <container> grep -r 'class <Name>\|def <name>' \
  <vllm_root>/vllm/ --include='*.py' -l"
```

Update the import path in the plugin file. Never add `sys.path` hacks or guessing `try/except` imports.

### RecursionError or infinite loop in __init__

The plugin subclasses or wraps a vllm class, but vllm changed that class to a factory or changed its MRO.
The plugin's `__init__` ends up calling back into itself.

Fix pattern: capture the original class reference before any monkey-patching, and call it explicitly:
```python
_Orig = _pkg.SomeClass   # capture before any patching occurs
class SomeClassFL(_Orig):
    def __init__(self, ...):
        _Orig.__init__(self, ...)   # explicit call, not through patched name
```

### AttributeError: _C_cache_ops has no attribute X

A KV cache op is declared in `vllm_fl/ops/_C_ops_schemas.py` but has no NVIDIA backend implementation.

- Generic cache op (needed by all models): must implement it -- the plugin is broken without it
- Model-specific op: only blocks that model, implement when adding support for that model

For implementation: reference upstream vllm's `_custom_ops.py` for the C extension wrapper pattern.
For model-specific ops, check other hardware backends in the plugin for algorithmic reference.

---

## Step 3: Unit Test Verification

After all API fixes, run unit tests to confirm no regressions:

```bash
ssh <host> "docker exec \
  -e VLLM_PLUGINS=fl \
  -e PYTHONPATH=<plugin_root> \
  <container> \
  python3 -m pytest <plugin_root>/tests/unit_tests/ -v --tb=short \
  2>&1 | tee <log_dir>/unit_after_fix_$(date +%Y%m%d_%H%M%S).log"
```

Compare pass/fail counts against the baseline from Step 1b.
- Acceptable: pre-existing failures that were present before the upgrade (document these)
- Not acceptable: new failures introduced by the upgrade fixes

---

## Step 4: Offline Inference Validation (NVIDIA)

Validate on real GPU hardware. Run at minimum one model per architecture type.

### 4a. Probe environment before running

```bash
ssh <host> "docker exec <container> bash -c '
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader &&
  python3 -c \"import torch; print(torch.cuda.device_count(), torch.version.cuda)\" &&
  find /usr/local -name nvcc 2>/dev/null | head -1
'"
```

Required env vars for the container (probe all values, never hardcode):
```
VLLM_PLUGINS=fl
PYTHONPATH=<plugin_root>
CUDA_HOME=<cuda_home>    # probe: find /usr/local -name nvcc | xargs dirname | xargs dirname
CC=<gcc_path>            # probe: which gcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4b. Model coverage matrix

Run at minimum one model per architecture type, in order of increasing complexity:

| Architecture | Why |
|---|---|
| Dense LLM (e.g., Qwen, LLaMA) | Base case, no MoE or special ops |
| MoE LLM (e.g., Qwen-MoE, Mixtral) | Exercises FusedMoE plugin path |
| Mamba/Hybrid | Exercises CUDA graph with non-attention layers |
| VLM (e.g., Gemma, SmolVLM) | Exercises multimodal pipeline |

For each model:
```bash
ssh <host> "docker exec \
  -e VLLM_PLUGINS=fl \
  -e PYTHONPATH=<plugin_root> \
  <container> \
  python3 <plugin_root>/examples/<model>_offline_inference.py \
  2>&1 | tee <log_dir>/offline_<model>_$(date +%Y%m%d_%H%M%S).log"
```

Monitor: `monitor(file=<log_file>, success_pattern='Generated text:|Output:', fail_pattern='ERROR|Traceback', duration=600)`

### 4c. Hardware-specific failure patterns on NVIDIA A800/A100

These are documented from past upgrades as reference. New upgrades may encounter different issues.

**fp8e4nv on sm<89 (A800, A100)**

Triton rejects `torch.float8_e4m3fn` on GPUs with `sm_major < 9`.

Symptom: `triton.runtime.errors.OutOfResources` or `TypeError` in fp8 quantization code.

Fix location: `vllm/model_executor/layers/quantization/utils/fp8_utils.py` (note: vllm source, prefer
upstreaming to vllm rather than keeping as a local patch):
```python
# TODO: remove when triton supports fp8e4nv on sm<89
if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
    dtype = torch.float8_e5m2
```

**FlagGems mm shmem overflow**

Symptom: `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 196608, Hardware limit: 166912`

Root cause: FlagGems mm autotune configs exceed A800's shared memory limit (166912 bytes).

Fix: in FlagGems `tune_configs.yaml`, remove autotune entries where the product of BLOCK sizes
exceeds the hardware limit.

**FlagGems broadcast_to CUDA graph bug**

Symptom: `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless
the CPU tensor is pinned`

Fix: wrap `torch.tensor(...)` calls in `broadcast_to.py` with `pin_memory=True` when device is CUDA.
Reference: https://github.com/FlagOpen/FlagGems/pull/4472

---

## Step 5: FlagGems Integration Checks

After basic inference passes, verify FlagGems kernels are actually dispatched and not silently falling back:

```bash
ssh <host> "docker exec \
  -e VLLM_PLUGINS=fl \
  -e PYTHONPATH=<plugin_root> \
  -e FLAG_GEMS_LOG_LEVEL=DEBUG \
  <container> \
  python3 <plugin_root>/examples/<model>_offline_inference.py \
  2>&1 | grep -i 'flag_gems\|triton' | head -20"
```

Check:
1. FlagGems ops are being called (not falling back to torch)
2. No silent errors swallowed by FlagGems error handling
3. Output tokens are correct (compare against a non-FlagGems run if suspicious)

---

## Step 6: Serving Validation

```bash
ssh <host> "docker exec -d \
  -e VLLM_PLUGINS=fl \
  -e PYTHONPATH=<plugin_root> \
  <container> \
  python3 -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --port 8000 \
  2>&1 | tee <log_dir>/serve_<model>_$(date +%Y%m%d_%H%M%S).log &"

# Wait for server ready, then test
sleep 30
ssh <host> "curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"Hello\", \"max_tokens\": 20}' | python3 -m json.tool"
```

Check: response contains `choices[0].text` with actual tokens (not empty, not error).

---

## Step 7: PR Discipline

Before opening a PR:

1. Review all changes: `git diff HEAD~<n> --stat` -- remove any debug prints, temporary patches, or commented-out code
2. Verify no vllm source files were modified: `git diff HEAD~<n> -- <vllm_root>/` should be empty
3. Squash all commits: `git rebase -i HEAD~<n>`
4. Run unit tests one final time on the squashed commit

PR commit message format:
```
feat(plugin): upgrade vllm-plugin-FL compatibility to vllm X.Y.Z

- <one line per fix, e.g. "fix FusedMoE recursion by capturing _OrigFusedMoE before patching">
- <fix InputBatch kwargs mismatch with inspect-based shim>
- <remove stale cache_dtype kwarg from use_uniform_kv_cache call>

Tested: unit tests (N passed, M pre-existing failures), offline inference on NVIDIA A800
Models validated: <list>
```

---

## Diagnostic Commands

```bash
# Check all remaining errors after a fix attempt
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 -m pytest <plugin_root>/tests/unit_tests/ --tb=short -q 2>&1 | tail -30"

# Check which ops are missing
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 check_ops.py"

# Check CUDA graph capture failures
ssh <host> "docker exec <container> grep -a 'graph capture\|cudaGraphCapture\|CUDA graph' <log> | tail -10"

# Check GPU processes before relaunch
ssh <host> "docker exec <container> nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader"
```

---

## Related skills

- `infer-env-setup` -- set up the container and conda env from scratch
- `infer-hw-adapt` -- hardware-specific backend adaptation (non-NVIDIA)
- `infer-model-adapt` -- port a new model into the plugin
- `debug-strategy` -- systematic debugging when stuck
- `ops-discipline` -- shell safety and environment awareness
appear across upgrades -- read the actual error, trace it to the changed vllm code, then fix.

---

