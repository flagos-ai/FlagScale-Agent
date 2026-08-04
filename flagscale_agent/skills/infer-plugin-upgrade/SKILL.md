---
name: infer-plugin-upgrade
description: Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version
  detection, API diff analysis, targeted fixes, and validation across unit tests, offline
  inference, and serving. Applies to any vLLM minor version bump (e.g., 0.20.x to 0.24.x).
keywords:
  - vllm plugin upgrade
  - vllm version bump
  - plugin API breakage
  - vllm-plugin-FL
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

Before starting, create a rollback point:
```bash
ssh <host> "docker exec <container> bash -c 'cd <plugin_root> && git stash'"
```
If the upgrade fails at any point, restore with `git stash pop`.

### 0a. SSH connection and container check

```bash
ssh <host> "hostname && docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"
```

Identify the running container for vllm-plugin-FL work. If no container exists, set one up following `infer-env-setup`.

### 0b. Locate plugin and vllm roots, detect version gap

```bash
ssh <host> "docker exec <container> bash -c '
  echo === vllm version === &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  echo === plugin location === &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  echo === plugin pyproject (declared compatible version) === &&
  find / -path \"*/vllm-plugin-FL/pyproject.toml\" 2>/dev/null | head -3 | xargs grep -E \"vllm|version\" &&
  echo === vllm source root === &&
  python3 -c \"import vllm, os; print(os.path.dirname(vllm.__file__))\"
'"
```

If installed vllm version != plugin declared compatible version, version gap is confirmed -- proceed with upgrade.

Record all discovered paths to memory immediately:
```python
memory_write(key='fact/infer/nvidia_vllm_version', type='fact',
  content='值: X.Y.Z\n验证命令: python3 -c "import vllm; print(vllm.__version__)"')
memory_write(key='fact/infer/nvidia_plugin_root', type='fact',
  content='值: <discovered_plugin_root>')
memory_write(key='fact/infer/nvidia_vllm_root', type='fact',
  content='值: <discovered_vllm_root>')
memory_write(key='fact/infer/nvidia_container', type='fact',
  content='值: <container_name>')
memory_write(key='fact/infer/nvidia_log_dir', type='fact',
  content='值: <log_dir>')
```

---

## Step 1: Change Analysis

A vLLM minor bump brings two kinds of changes. Both require attention.
Missing either one leaves the plugin silently incomplete or broken.

**Breakages** surface as errors (ImportError, TypeError, AttributeError).
**New features** are silent -- the plugin simply won't support them.

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

Collect all ImportError, AttributeError, TypeError -- these are the API breakages to fix in Step 2.
Record the pass/fail counts as baseline; any new failure introduced by your fixes is a regression.

Pre-existing failures (present before the upgrade) are acceptable -- document them in
`<log_dir>/known_failures.txt` and reference this file in the PR description.

### 1c. Check _C_cache_ops op availability

Write a probe script, copy it into the container, then run it:

```bash
# Write the probe script locally
cat > /tmp/check_ops.py << 'EOF'
# check_ops.py -- diff plugin-declared ops vs installed vLLM ops
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
EOF

# Copy into container and run
ssh <host> "docker cp /tmp/check_ops.py <container>:/tmp/check_ops.py && \
  docker exec \
    -e VLLM_PLUGINS=fl \
    -e PYTHONPATH=<plugin_root> \
    <container> \
    python3 /tmp/check_ops.py \
  2>&1 | tee <log_dir>/check_ops_$(date +%Y%m%d_%H%M%S).log"
```

Key distinction: FlagGems covers compute kernels (matmul, attention, elementwise). It does NOT cover
`_C_cache_ops` -- those are vLLM's paged KV cache management ops and must be implemented in the plugin.

Missing ops fall into two categories:
- **Generic cache ops** (needed by all models to manage paged KV cache): must be present for any model to run.
  If missing, the plugin is broken for this vllm version.
- **Model-specific ops** (needed only by a particular attention variant or architecture): only
  blocks that model. Other models run fine without them.

### 1d. Scan for new features that require plugin glue

Breakages surface as errors; new vLLM features are silent. The plugin simply won't support them
until you add the glue code. For each minor bump, actively scan for the following categories
by reading both the vLLM changelog and the actual source diff:

**New execution paths the plugin must route through:**
Check whether vLLM added new wrapper classes around `nn.Module` (such as graph wrappers or
quantization wrappers). The plugin's `get_model()` and `load_model()` must handle any new wrapper
type -- if they don't, callers receive the wrapper instead of the raw model, causing subtle failures
that don't produce errors at import time.

```bash
# Find new wrapper classes that wrap nn.Module
ssh <host> "docker exec <container> grep -rn 'class.*Wrapper\|class.*Graph' \
  <vllm_root>/vllm/compilation/ <vllm_root>/vllm/worker/ --include='*.py' -l"
```

**New env vars that gate new execution modes:**
vLLM adds env vars to enable new features. If the plugin's worker or compilation code doesn't
check these vars, the feature is silently unavailable even when the user sets them.

```bash
# Find new env vars in vllm that affect execution
ssh <host> "docker exec <container> grep -rn 'os.environ.get\|envs\.' \
  <vllm_root>/vllm/compilation/ <vllm_root>/vllm/worker/ --include='*.py' | \
  grep -v '\.pyc'"
```

**New speculative decoding proposer types:**
The plugin's drafter dispatch must include every proposer class vLLM introduces. A missing branch
typically doesn't raise an error immediately -- it either falls through to the wrong code path or
silently disables spec decode.

```bash
# Find proposer classes in new vllm
ssh <host> "docker exec <container> grep -rn 'class.*Proposer' \
  <vllm_root>/vllm/v1/spec_decode/ --include='*.py'"
# Compare against what the plugin's model_runner.py handles
ssh <host> "docker exec <container> grep -n 'Proposer\|proposer' \
  <plugin_root>/vllm_fl/worker/model_runner.py"
```

**New fields in core data structures:**
Structures like `InputBatch`, `CommonAttentionMetadata`, `ModelRunnerOutput` gain new fields in
every minor release. Check whether the plugin initializes or passes these correctly.

```bash
# Diff InputBatch fields between plugin and vllm
ssh <host> "docker exec <container> bash -c '
  grep -n \"def __init__\|self\\.\" <vllm_root>/vllm/v1/worker/gpu_input_batch.py | head -40
  echo ---
  grep -n \"def __init__\|self\\.\" <plugin_root>/vllm_fl/worker/model_runner.py | head -40
'"
```

### 1e. Breakage-prone areas to audit

Every vLLM minor bump changes something in these areas. Read both the plugin code and the
new vllm source side by side before writing any fix:

| Plugin file | What to audit in new vllm | Why it commonly breaks |
|---|---|---|
| MoE layer implementation | Is the MoE class still directly subclassable or is it now a factory? What did the router's `__init__` signature change to? | vllm toggles between class and factory; subclassing or kwarg forwarding breaks silently |
| `vllm_fl/worker/model_runner.py` | Core data structure constructor params (input batch, attention metadata), deprecated method signatures, new model wrapper types that `get_model()` must unwrap, new execution dispatch branches | Plugin often lags behind by 1-2 vllm versions; new params appear or old ones are removed |
| `vllm_fl/ops/_C_ops_schemas.py` | Run check_ops.py (Step 1c) to diff registered schemas vs installed ops | vllm reorganizes C extensions; model-specific ops may never be in base vllm |
| Any file with `from vllm.X import Y` | Does that import path still exist in new vllm? | vllm moves symbols between modules frequently |
| `vllm_fl/worker/` distributed code | Distributed init API, process group call signatures | Distributed init API evolves across versions |

The specific breakages depend entirely on the version gap. Do not assume the same bugs will
appear across upgrades -- read the actual error, trace it to the changed vllm code, then fix.

---

## Step 2: Fix API Breakages

The workflow for every breakage is the same:
1. Read the error traceback -- identify which plugin file and line calls into vllm
2. Read the new vllm source at that call site to understand what changed
3. Apply the minimal fix
4. **Verify the fix immediately** before moving to the next error (see verification commands below)

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

Verify fix:
```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 -c 'from vllm_fl.<module> import <ClassName>; print(\"OK\")'  "
```

### ImportError or AttributeError on import

A symbol moved between vllm modules.

```bash
# Find where it moved
ssh <host> "docker exec <container> grep -r 'class <Name>\|def <name>' \
  <vllm_root>/vllm/ --include='*.py' -l"
```

Update the import path in the plugin file. Never add `sys.path` hacks or guessing `try/except` imports.

Verify fix:
```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 -c 'from vllm_fl.<module> import <Name>; print(\"OK\")'  "
```

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

Verify fix:
```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 -c 'from vllm_fl.<module> import <ClassName>; obj = <ClassName>(...); print(\"OK\")'  "
```

### AttributeError: _C_cache_ops has no attribute X

A KV cache op is declared in `vllm_fl/ops/_C_ops_schemas.py` but has no NVIDIA backend implementation.

- Generic cache op (needed by all models): must implement it -- the plugin is broken without it
- Model-specific op: only blocks that model, implement when adding support for that model

For implementation: reference upstream vllm's C extension wrapper conventions for the correct pattern.
For model-specific ops, check other hardware backends in the plugin for algorithmic reference.

Verify fix:
```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 /tmp/check_ops.py"
```

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

Compare pass/fail counts against the baseline from Step 1b:
- **Acceptable**: pre-existing failures documented in `<log_dir>/known_failures.txt`
- **Not acceptable**: new failures introduced by your fixes

If new failures appear, do not proceed -- diagnose and fix before moving to Step 4.

---

## Step 4: Offline Inference Validation (NVIDIA)

Validate on real GPU hardware. Run at minimum one model per coverage category.

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

Run at minimum one model per category, in order of increasing complexity:

| Category | Why |
|---|---|
| Dense LLM | Base case, no MoE or special ops |
| MoE LLM | Exercises the MoE layer plugin path |
| Mamba/Hybrid | Exercises CUDA graph with non-attention layers |
| VLM (multimodal) | Exercises multimodal pipeline and encoder cache |
| Dense LLM + speculative decoding | Exercises drafter dispatch, spec decode input batch paths, and any new proposer routing added in the new vllm version |

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

### 4c. Hardware-specific failures

These failures won't surface until you run on real hardware. When a model run fails here but unit
tests passed, the root cause is typically one of three categories:

**Quantization kernel / dtype incompatibility**

Symptom: kernel errors or type errors inside quantization code, often from Triton or CUDA ops.

Diagnosis: check whether the dtype or quantization format is supported on this specific GPU generation.
Some dtypes require a minimum compute capability. When the hardware is below that threshold, the
code must fall back to a compatible dtype.

Fix approach: add a hardware capability check before the unsupported dtype is used. Mark the fix
with a TODO if the underlying library is expected to add support later, and prefer upstreaming
such fixes to the dependency rather than keeping them as local patches.

**FlagGems kernel resource overflow**

Symptom: Triton out-of-resource errors (shared memory or register file) during FlagGems kernel
execution.

Diagnosis: the autotune config for a FlagGems op contains block size combinations that exceed the
hardware resource limits of this GPU. Limits vary by GPU model.

Fix approach: remove or constrain autotune configurations that exceed the hardware resource budget.
The fix lives in FlagGems config, not in the plugin.

**CUDA graph capture incompatibility**

Symptom: errors during CUDA graph capture involving CPU-GPU tensor interactions, wrong device
placement, or operations that are illegal inside a graph capture context.

Diagnosis: an op inside the capture window performs an illegal action (e.g., CPU tensor allocation,
synchronous device-to-host copy, or pinned memory violation).

Fix approach: ensure tensors created during capture-time paths use pinned memory or are pre-allocated
outside the capture. When the bug is in a dependency (FlagGems, vLLM), file upstream; add a local
workaround with a TODO referencing the upstream issue.

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
1. FlagGems ops are being dispatched (not falling back to torch)
2. No silent errors swallowed by FlagGems error handling

**If FlagGems ops are falling back to torch**: this is a known risk on every upgrade. Decide by impact:
- If the fallback is for a non-critical op (e.g. elementwise) and output is correct: document in PR,
  file a follow-up issue, do not block the upgrade.
- If the fallback is for a performance-critical op (e.g. matmul, attention): investigate root cause
  before declaring the upgrade done. Throughput regression may be significant.

**Verify output correctness when fallback is suspected**:
Run the same prompt with and without `VLLM_PLUGINS=fl` and compare the first-token outputs.
Divergence indicates a correctness bug, not just a performance issue.

---

## Step 6: Serving Validation

Start the server inside the container (not with `docker exec -d`, which swallows stdout):

```bash
# Launch server inside container, writing logs to a file
ssh <host> "docker exec <container> bash -c '
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model <model_path> --port 8000 \
    > <log_dir>/serve_<model>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  echo \$! > /tmp/vllm_server.pid && echo Server PID: \$!
'"

# Wait for server ready with health polling (handles slow model loading)
ssh <host> "docker exec <container> bash -c '
  for i in \$(seq 1 60); do
    curl -sf http://localhost:8000/health && echo && break
    echo \"Waiting... (\$i/60)\" && sleep 5
  done
'"

# Test inference
ssh <host> "curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"Hello\", \"max_tokens\": 20}' \
  | python3 -m json.tool"
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

- <one line per breakage fix, e.g. "fix MoE layer recursion by capturing original class before patching">
- <fix data structure constructor kwargs mismatch with inspect-based shim>
- <remove stale kwarg from KV cache initialization call>
- <add glue for <new feature> introduced in vllm X.Y.Z>

Tested: unit tests (N passed, M pre-existing failures documented in known_failures.txt)
Offline inference: <hardware> -- Dense, MoE, Mamba, VLM, speculative decoding
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
  python3 /tmp/check_ops.py"

# Check CUDA graph capture failures
ssh <host> "docker exec <container> grep -a 'graph capture\|cudaGraphCapture\|CUDA graph' <log> | tail -10"

# Check GPU processes before relaunch
ssh <host> "docker exec <container> nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader"

# Quick import check after any fix
ssh <host> "docker exec -e VLLM_PLUGINS=fl -e PYTHONPATH=<plugin_root> <container> \
  python3 -c 'import vllm_fl; print(\"plugin import OK\")'"
```

---

## Related skills

- `infer-env-setup` -- set up the container and conda env from scratch
- `infer-hw-adapt` -- hardware-specific backend adaptation (non-NVIDIA)
- `infer-model-adapt` -- port a new model into the plugin
- `debug-strategy` -- systematic debugging when stuck
- `ops-discipline` -- shell safety and environment awareness
