---
name: infer-model-adapt
description: Adapt a new model to vllm-plugin-FL by migrating model code from the latest vLLM
  upstream. Covers source discovery, copy-then-patch workflow, import conversion, model
  registration, and correctness verification against upstream GPU ground truth.
keywords:
- inference
- vllm
- model
- porting
- adaptation
- migration
- plugin
- model-runner
- registration
- correctness
requires:
- infer-env-setup
suggests:
- infer-hw-adapt
- debug-strategy
- ops-discipline
constraints:
- id: no_vllm_source_modification
  description: Never modify vLLM source code — all changes go through plugin
  trigger:
    tools: [edit_file, write_file]
    keywords: [site-packages/vllm, vllm/models]
  prompt: Check if the agent is editing vLLM source files
  correction: Copy the file to vllm_fl/models/ first, then patch the copy.
- id: copy_before_patch
  description: Always copy upstream file verbatim before making any edits
  trigger:
    keywords: [write_file, create model file]
  prompt: Check if a clean upstream copy exists before editing
  correction: First `cp <vllm_upstream>/<model>.py <plugin>/vllm_fl/models/<model>.py`, then edit.
- id: import_check_per_batch
  description: Run an import check after each batch of patches
  trigger:
    tools: [edit_file]
    keywords: [from vllm, import vllm]
  prompt: Check if the agent will verify the import after this patch
  correction: Run `python3 -c "from vllm_fl.models.<model> import <Class>; print('OK')"` before next batch.
- id: absolute_imports
  description: All relative imports in copied files must be converted to absolute plugin-rooted imports
  trigger:
    tools: [edit_file, write_file]
    keywords: [from .., from .]
  prompt: Check if relative imports have been converted to absolute plugin imports
  correction: Change `from ..utils import X` to `from vllm_fl.utils import X` (or `from vllm.utils import X` if not patched).
- id: version_adaptive
  description: Detect installed vLLM version before deciding what to patch
  trigger:
    keywords: [patch, adapt, copy upstream]
  prompt: Check if the agent verified the installed vLLM version first
  correction: Run `python3 -c "import vllm; print(vllm.__version__)"` and select patches based on actual version.
context_injection:
  always:
  - Critical Rules
  - Porting Pipeline
  by_tool:
    edit_file:
    - Copy-then-Patch Discipline
    - Import Conversion Rules
    shell:
    - Stage 0 Orientation
    - Correctness Verification
---
# Model Adaptation for vllm-plugin-FL

Port a new model from the latest vLLM upstream into vllm-plugin-FL.

## When to Use This Skill

Use this skill when:
- Adding a model to vllm-plugin-FL that vLLM supports but the plugin does not
- Re-porting a model after a major vLLM version bump changed the model implementation
- Migrating a community model that uses vLLM's model API into the plugin

This skill covers everything from source discovery through E2E correctness testing.
Environment setup (SSH, container, installation) is handled by `infer-env-setup`.

## Critical Rules

1. **Detect before patching** — always check installed vLLM version and plugin version before deciding what to port.
2. **Never modify vLLM source** — copy the upstream file to plugin first, then patch the copy.
3. **Copy verbatim, then patch** — `cp` the upstream file unchanged, then apply targeted edits. Never write a model file from scratch.
4. **One batch at a time** — group related changes (e.g., all import conversions), verify with an import check, then move to next batch.
5. **Import check after every batch** — `python3 -c "from vllm_fl.models.X import Y; print('OK')"`.
6. **Absolute imports only** — convert all relative imports to absolute plugin-rooted or vllm-rooted imports.
7. **Workspace orientation first** — run Stage 0 before any porting work.
8. **Record key findings** — use `memory_write` after reading any source file; check `memory_read` before re-opening.
9. **Platform-gate hardware patches** — wrap backend-specific code with `if current_platform.is_<backend>():`.
10. **Squash before PR** — all porting commits squashed into one clean commit before opening PR.

---

## Porting Pipeline

### Step 0: Workspace Orientation

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  find /workspace -name \"vllm-plugin-FL\" -type d 2>/dev/null | head -3 &&
  ls /workspace/models/ 2>/dev/null
'"
```

Record:
```
memory_write('<backend>_vllm_version', 'X.Y.Z')
memory_write('<backend>_plugin_root', '/workspace/adapt/<backend>-vllm-X.Y.Z/vllm-plugin-FL')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
```

### Step 1: Baseline Unit Tests (must pass before porting)

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v \
   2>&1 | tee /workspace/adapt-logs/unit_baseline.log'"
```

Do NOT start porting if unit tests fail — that indicates an environment problem.

### Step 2: Locate Upstream Model Source

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'python3 -c \"import vllm; import os; print(os.path.dirname(vllm.__file__))\"'"

# Then find the model file
ssh <ssh_host> "docker exec <container> bash -c \
  'find \$(python3 -c \"import vllm, os; print(os.path.dirname(vllm.__file__))\") \
   -name \"*.py\" | xargs grep -l \"class <ModelClass>\" 2>/dev/null'"
```

Record: `memory_write('<model>_upstream_file', '<full_path_to_upstream_model.py>')`

### Step 3: Study Plugin Patterns

Read two existing ported models from the plugin to understand the expected patterns before writing any code:

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'ls <plugin_root>/vllm_fl/models/'"
```

Read at minimum: a simple dense model and a MoE model (if porting MoE). Record key patterns with `memory_write`.

### Step 4: Identify Model Identity

```bash
# Check HF config.json for model_type
ssh <ssh_host> "docker exec <container> bash -c \
  'cat <model_path>/config.json | python3 -c \
   \"import json,sys; c=json.load(sys.stdin); print(c.get(\\\"model_type\\\"), c.get(\\\"architectures\\\"))\"'"
```

The `model_type` value is the registration key. Record it:
```
memory_write('<model>_model_type', '<model_type_from_config>')
memory_write('<model>_arch_class', '<ArchClass from architectures list>')
```

### Step 5: Check Config Bridge

If the model has a custom config class in vLLM, check if the plugin already has it:

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'find <plugin_root> -name \"*.py\" | xargs grep -l \"<ModelConfig>\" 2>/dev/null'"
```

If missing, copy `vllm/config/<model>_config.py` → `vllm_fl/config/<model>_config.py` and patch imports.

### Step 6: Copy-then-Patch the Model File

```bash
# Step 6a: Copy verbatim
ssh <ssh_host> "docker exec <container> bash -c \
  'cp <upstream_model_file> <plugin_root>/vllm_fl/models/<model>.py'"

# Step 6b: Check imports in copied file
ssh <ssh_host> "docker exec <container> bash -c \
  'head -50 <plugin_root>/vllm_fl/models/<model>.py'"
```

Then apply patches in batches:

**Batch A: Import conversion** (convert relative vLLM internals to absolute)
```python
# Before (in copied file)
from vllm.attention import Attention
from .utils import make_layers

# After (in plugin file)
from vllm.attention import Attention          # vLLM public API — keep as-is
from vllm_fl.utils import make_layers         # plugin-patched utility — update
```

Run import check: `python3 -c "from vllm_fl.models.<model> import <Class>; print('OK')"`

**Batch B: Operator substitutions** (swap vLLM ops for plugin-provided equivalents)
```python
# Before
from vllm.ops.fused_moe import fused_moe
# After
from vllm_fl.ops.fused_moe import fused_moe
```

Run import check again.

**Batch C: Hardware-specific patches**
```python
# Platform-gated adaptation example
from vllm.platforms import current_platform

class MyModelAttention(nn.Module):
    def forward(self, ...):
        if current_platform.is_metax():
            # TODO: Remove when MetaX supports flash-attn 2.6+
            return self._eager_attn(...)
        return self._flash_attn(...)
```

### Step 7: Register the Model

Add entry to `<plugin_root>/vllm_fl/__init__.py`:

```python
# In the model registry dict
_MODEL_REGISTRY = {
    ...
    "<model_type_from_config>": "vllm_fl.models.<model>.<ArchClass>",
    ...
}
```

Verify registration:
```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'VLLM_PLUGINS=fl python3 -c \
   \"from vllm import LLM; print(LLM._supported_models())\" | grep <model_type>'"
```

### Step 8: Code Review Before Testing

Before running any tests, review the diff:

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && git diff HEAD'"
```

Checklist:
- [ ] No relative imports left in ported file
- [ ] All platform-specific patches are gated
- [ ] Every workaround has a `# TODO` comment
- [ ] No debug print statements
- [ ] No hardcoded paths or credentials

### Step 9: Regression Unit Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v \
   2>&1 | tee /workspace/adapt-logs/unit_post_port.log'"
```

All previously-passing unit tests must still pass. If any new failure appears, bisect: it is caused by the registration or import change, not the model itself.

### Step 10: Model Functional Tests

If the plugin has model-specific functional tests:
```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl pytest tests/functional_tests/ -x -v \
   -k <model_name> 2>&1 | tee /workspace/adapt-logs/functional_<model>.log'"
```

### Step 11: Throughput Benchmark

Measure throughput to establish a baseline for the PR:

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl \
   python benchmarks/benchmark_throughput.py \
   --model <model_path> \
   --tensor-parallel-size 2 \
   --input-len 512 --output-len 128 --num-prompts 50 \
   2>&1 | tee /workspace/adapt-logs/bench_<model>.log'"
```

Record: `memory_write('<model>_throughput_baseline', '<X> tok/s, TP=2, in=512, out=128')`

### Step 12: Serving + Sample Request

```bash
# Start server
ssh <ssh_host> "docker exec -d <container> bash -c \
  'VLLM_PLUGINS=fl vllm serve <model_path> \
   --tensor-parallel-size 2 --enforce-eager --trust-remote-code \
   2>&1 | tee /workspace/adapt-logs/serve_<model>.log'"

# Wait for ready (monitor for "Uvicorn running on")
# Then send test request
ssh <ssh_host> "docker exec <container> curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20}'"
```

### Step 13: E2E Correctness vs Upstream

Compare token-level output against a reference (NVIDIA GPU or vLLM CPU):

```bash
# On reference server (upstream vLLM on NVIDIA)
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM('<model_path>', tensor_parallel_size=2)
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=50, temperature=0))
print('GT:', out[0].outputs[0].token_ids)
"

# On plugin (hardware backend)
ssh <ssh_host> "docker exec <container> bash -c '
VLLM_PLUGINS=fl python3 -c \"
from vllm import LLM, SamplingParams
llm = LLM(\\\"<model_path>\\\", tensor_parallel_size=2)
out = llm.generate([\\\"The capital of France is\\\"], SamplingParams(max_tokens=50, temperature=0))
print(\\\"Plugin:\\\", out[0].outputs[0].token_ids)
\"'"
```

**Acceptance criteria**:
- First 15 tokens must be identical to the reference
- Divergence after token 15 is acceptable (numerical noise from different FP16/BF16 accumulation)
- Early divergence (before token 5) is a bug — investigate op dispatch or weight loading

### Step 14: Final Report

Before opening PR, record in memory:
```
memory_write('<model>_adapt_result', 'unit: PASS, functional: PASS, throughput: X tok/s, E2E: first 15 tokens match')
```

---

## Import Conversion Rules

When patching a copied model file, apply these import rules:

| Original import | Action |
|---|---|
| `from vllm.attention import Attention` | Keep as-is (vLLM public API) |
| `from vllm.model_executor.layers.X import Y` | Keep if plugin doesn't override, change to `vllm_fl.model_executor.layers.X` if it does |
| `from .utils import X` | Change to `from vllm_fl.utils import X` or `from vllm.utils import X` |
| `from ..ops.fused_moe import X` | Change to `from vllm_fl.ops.fused_moe import X` |
| `from vllm._custom_ops import X` | Keep as-is (C extension, not overridden by plugin) |

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, vLLM + plugin install)
- `infer-hw-adapt` — hardware backend adaptation after plugin version upgrades
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety, environment awareness, and command discipline
- `workspace-layout` — shared storage paths for models and adaptation artifacts

---
Related skills (load if needed): `infer-hw-adapt`, `debug-strategy`, `ops-discipline`
### Step 4: Identify Model Identity

Get the exact model_type and class name from the model's HF config:

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cat <model_path>/config.json | python3 -m json.tool | grep -E \"model_type|architectures\"'"
```

Record:
```
memory_write('<model>_model_type', '<e.g. qwen3>')
memory_write('<model>_arch_class', '<e.g. Qwen3ForCausalLM>')
```

### Step 5: Config Bridge

If the model uses a custom config class, create `vllm_fl/configs/<model>_config.py`:

```python
# vllm_fl/configs/<model>_config.py
from vllm.config import ModelConfig  # re-export or extend as needed

class <Model>PluginConfig:
    """Bridge between HF config fields and vLLM plugin expectations."""
    def __init__(self, hf_config):
        self.hidden_size = hf_config.hidden_size
        # ... map fields
```

If the upstream vLLM model uses the config directly (most do), skip this step.

### Step 6: Copy-then-Patch

#### 6a. Copy verbatim

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cp <upstream_model_file> <plugin_root>/vllm_fl/models/<model>.py'"
```

#### 6b. Batch 1 — Convert imports

Change relative imports to absolute:

```python
# Before (relative vLLM internal imports):
from ..attention import Attention
from ..layers.linear import ColumnParallelLinear
from ...utils import is_hip

# After (absolute, plugin-rooted or vllm-rooted):
from vllm.attention import Attention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.utils import is_hip
```

Import check:
```bash
ssh <ssh_host> "docker exec <container> python3 -c \
  'from vllm_fl.models.<model> import <ModelClass>; print(\"import OK\")'"
```

#### 6c. Batch 2 — Hardware-specific op patches

Identify ops that the target hardware does not support natively. Common issues:

| Op | Issue | Fix |
|---|---|---|
| Triton Flash Attention | Not supported on hardware | Replace with `xformers`/hardware FA implementation |
| `torch.compile` graph capture | Segfault or hang | Add `enforce_eager=True` guard |
| `awq`, `gptq` quantized ops | Not implemented | Skip / raise NotImplementedError with TODO |
| `scaled_dot_product_attention` | Wrong dtype | Cast to float32 before, cast back after |

Apply patches with platform gates:

```python
from vllm.platforms import current_platform

def my_attention(q, k, v, ...):
    if current_platform.is_metax():
        # TODO: Remove when MetaX triton FA supports bf16
        return metax_attention_fallback(q, k, v, ...)
    return flash_attn_func(q, k, v, ...)
```

Import check after each patch.

#### 6d. Batch 3 — Plugin hook points

Some plugin versions require specific hook methods. Check existing models for:

```python
# Does the plugin expect this method?
def get_input_embeddings(self) -> nn.Module: ...
def set_input_embeddings(self, embeddings: nn.Module): ...

# LoRA support (if required by plugin)
def get_lora_manager(self): ...
```

Compare with an existing plugin model to see what hooks are expected.

### Step 7: Register the Model

Add to `<plugin_root>/vllm_fl/__init__.py` (or wherever the model registry is):

```python
from vllm_fl.models.<model> import <ModelClass>

# Register using the exact model_type from config.json
ModelRegistry.register_model("<model_type>", <ModelClass>)
```

Verify registration:
```bash
ssh <ssh_host> "docker exec <container> python3 -c \
  'import vllm_fl; from vllm.model_executor.model_loader.utils import get_model_architecture
   from vllm.config import ModelConfig
   print(\"registry OK\")'"
```

### Step 8: Code Review Checklist

Before running any tests:
- [ ] No relative imports remain in the ported file
- [ ] All hardware patches are platform-gated
- [ ] All workarounds have `# TODO: Remove when ...` comments
- [ ] No debug `print()` statements
- [ ] `git diff` reviewed — only the ported model + registration changes

### Step 9: Regression Unit Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v \
   2>&1 | tee /workspace/adapt-logs/unit_post_port.log'"
```

Must pass. If new failures appeared, the porting introduced a regression.

### Step 10: Functional Tests

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && VLLM_PLUGINS=fl pytest tests/functional_tests/ -x -v \
   -k \"<model_name>\" \
   2>&1 | tee /workspace/adapt-logs/functional_<model>.log'"
```

If no model-specific functional tests exist, run the full suite to check for regressions.

### Step 11: Benchmark (offline inference)

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd <plugin_root> && \
   VLLM_PLUGINS=fl MODEL_PATH=<model_path> TP_SIZE=2 \
   python examples/<model>_offline_inference.py \
   2>&1 | tee /workspace/adapt-logs/offline_<model>.log'"
```

Record:
- Time to first token
- Throughput (tokens/sec)
- GPU/NPU memory utilization

### Step 12: Serving Test

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'VLLM_PLUGINS=fl vllm serve <model_path> \
   --tensor-parallel-size 2 --enforce-eager --trust-remote-code \
   2>&1 | tee /workspace/adapt-logs/serving_<model>.log &

   sleep 30 && \
   curl -s http://localhost:8000/v1/completions \
   -H \"Content-Type: application/json\" \
   -d \"{\\\"model\\\": \\\"<model_path>\\\", \\\"prompt\\\": \\\"Hello\\\", \\\"max_tokens\\\": 20}\"'"
```

### Step 13: E2E Correctness Verification

Compare token-level output against upstream vLLM on NVIDIA GPU (ground truth):

```bash
# On GT server (NVIDIA GPU):
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM('<model_path>', tensor_parallel_size=1)
out = llm.generate(['Paris is the capital of'], SamplingParams(max_tokens=20, temperature=0))
print([o.outputs[0].token_ids for o in out])
"

# On target hardware:
ssh <ssh_host> "docker exec <container> python3 -c \"
import os; os.environ['VLLM_PLUGINS'] = 'fl'
from vllm import LLM, SamplingParams
llm = LLM('<model_path>', tensor_parallel_size=2, enforce_eager=True)
out = llm.generate(['Paris is the capital of'], SamplingParams(max_tokens=20, temperature=0))
print([o.outputs[0].token_ids for o in out])
\""
```

**Acceptance criteria:**
- First 5 tokens must match GT exactly (greedy decoding, temperature=0)
- Minor divergence allowed after token 15 due to floating-point accumulation
- Any divergence before token 5 is a bug — investigate attention or sampling

### Step 14: Final Report

Document in commit message and PR description:
- Model name and vLLM version
- What was patched and why (per batch)
- Test results (unit/functional/offline/serving)
- E2E correctness result (token match count)
- Known limitations or TODOs
- Throughput numbers vs NVIDIA GT

---

## Import Conversion Rules

| Before (relative) | After (absolute) |
|---|---|
| `from ..attention import X` | `from vllm.attention import X` |
| `from ..model_executor.layers.linear import X` | `from vllm.model_executor.layers.linear import X` |
| `from ...utils import X` | `from vllm.utils import X` |
| `from .utils import X` (plugin util) | `from vllm_fl.utils import X` |
| `from .base_model import X` (plugin base) | `from vllm_fl.models.base_model import X` |

When in doubt: check if the module exists in `vllm_fl/` first; if yes use `vllm_fl.`; if no use `vllm.`.

---

## Copy-then-Patch Discipline

1. **Copy first**: `cp <upstream> <plugin>` — verbatim, no edits
2. **Patch via targeted edits**: use `edit_file` for specific lines, not `write_file` for the whole file
3. **One category at a time**: group related changes, verify import, then continue
4. **Import check after each batch**: `python3 -c "from vllm_fl.models.X import Y; print('OK')"`
5. **Never rewrite from scratch**: if tempted to rewrite, stop and read more upstream code first

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-hw-adapt` — hardware backend adaptation after plugin version upgrades
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety, environment awareness, and command discipline
- `workspace-layout` — shared storage paths for models and adaptation artifacts

---
Related skills (load if needed): `infer-hw-adapt`, `debug-strategy`, `ops-discipline`
Acceptance criteria:
- First 15 tokens must match GT exactly (greedy decode, temperature=0)
- Divergence at token 16+ is acceptable (numerical noise from different hardware)
- Divergence at token 5 or earlier is a bug — investigate attention or sampling

### Step 14: Final Report

Collect metrics into PR description:

```
Model: <model_name> (<param_count>B)
Backend: <backend> (vLLM <version>)

Unit tests: PASS (N/N)
Functional tests: PASS (N/N)
Offline inference: PASS
  - TTFT: Xms
  - Throughput: X tok/s (TP=2)
Serving: PASS
E2E correctness: PASS (first 15 tokens match GT)

Patches applied:
- <patch1>: <why>
- <patch2>: <why>

FlagGems missing ops (for upstream):
- <op>: falls back to PyTorch
```

---

## Import Conversion Rules

| Before (relative/vLLM-internal) | After (absolute) |
|---|---|
| `from ..attention import X` | `from vllm.attention import X` |
| `from ..layers.linear import X` | `from vllm.model_executor.layers.linear import X` |
| `from ...utils import X` | `from vllm.utils import X` |
| `from .utils import X` (plugin file) | `from vllm_fl.utils import X` |
| `from ..._custom_op import X` | Check if plugin has override; if not, `from vllm._custom_op import X` |

---

## Common Porting Failures and Fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `ImportError: cannot import name X from vllm_fl.models` | Relative import not converted | Convert all `from .` and `from ..` to absolute |
| `AttributeError: 'NoneType' object has no attribute 'hidden_size'` | Config bridge missing | Add config field mapping in Step 5 |
| `RuntimeError: Expected device maca, got cpu` | Model not registered | Verify registration in `__init__.py` with exact model_type |
| Segfault during graph capture | Hardware doesn't support CUDA graphs | Add `enforce_eager=True` in hw-gated path |
| Token mismatch at position < 5 | Attention op gives wrong values | Verify FA implementation or fall back to eager |
| `KeyError: <model_type> not found` | model_type mismatch | Compare `config.json` value with registry key exactly |
| `TypeError: forward() got unexpected argument` | vLLM API changed in new version | Read upstream forward signature, update plugin copy |

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-hw-adapt` — hardware backend adaptation after plugin version upgrades
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety, environment awareness, and command discipline
- `workspace-layout` — shared storage paths for models and adaptation artifacts

---
Related skills (load if needed): `infer-hw-adapt`, `debug-strategy`, `ops-discipline`
