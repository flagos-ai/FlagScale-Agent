---
description: Upgrade sglang-plugin-FL to a newer SGLang release on NVIDIA hardware.
  Covers upstream and dependency analysis, OOT plugin compatibility fixes, FlagTree
  and FlagGems alignment, and validation with unit tests, model serving, streaming,
  concurrency, and CUDA Graph. Use when bumping the SGLang compatibility version or
  fixing sglang-plugin-FL after an upstream SGLang upgrade; do not use for a
  model-only port or a non-NVIDIA backend adaptation.
name: infer-sglang-plugin-upgrade
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

# Upgrade sglang-plugin-FL to a New SGLang Version

Upgrade the NVIDIA path of sglang-plugin-FL without patching the installed
SGLang package. Treat SGLang, PyTorch, sglang-kernel, the Triton provider,
FlagTree, FlagGems, and flashinfer as one compatibility tuple.

## When to Use

Use this skill when the plugin's SGLang pin must move, or when an upstream
SGLang upgrade breaks imports, registration, fused ops, platform APIs, CUDA
Graph, FLA, disaggregation, or serving.

Use `infer-model-adapt` for a model-only port and `infer-hw-adapt` for a
non-NVIDIA backend. MUSA and Ascend upgrades are separate passes unless the user
explicitly includes them.

## Required Inputs

Discover these values before changing code:

- plugin repository and active branch;
- installed, declared, and target SGLang versions;
- NVIDIA host, container, GPU type, and device occupancy;
- matching SGLang source checkout or installed package root;
- model paths and intended tensor-parallel sizes;
- packaging scope: code only, or also docs/container images.

Ask only for values that cannot be probed safely.

## Critical Rules

1. **Detect versions and paths first.** Never infer them from an image tag alone.
2. **Upgrade NVIDIA first by default.** Preserve other backend pins until their
   own upgrade and validation passes.
3. **Never modify SGLang source.** Compatibility belongs in `sglang_fl/`,
   configuration, and plugin-owned tests.
4. **Resolve the full dependency tuple.** A passing `import sglang` does not
   prove that the kernel/compiler packages agree.
5. **When FlagTree supplies Triton, install PyTorch first, remove every
   standalone Triton installation, then install the pinned FlagTree build.**
6. **Pin the tested FlagGems revision.** Do not report “master” without its SHA.
7. **Do not add FlagCX by default on NVIDIA.** NCCL remains the normal path
   unless the requested feature explicitly requires FlagCX.
8. **Fix one compatibility theme at a time and re-run its focused test.**
9. **Validate with and without the plugin.** Vanilla SGLang at the same version
   is the behavioral baseline.
10. **Final tests must use default plugin configuration.** Remove diagnostic
    overrides or turn them into intentional, reviewed defaults before claiming
    success.
11. **Use tmux for long remote commands and persist logs.**
12. **Treat docs and containerfiles as independent scope.** Do not modify them
    merely because the runtime was assembled manually.

---

## Stage 0: Orient and Establish a Baseline

Preserve unrelated work and record repository state:

```bash
git status --short --branch
git remote -v
grep -nE 'sglang|torch|triton|flagtree|flag.?gems' pyproject.toml
```

Probe the actual runtime inside the target container:

```bash
python3 - <<'PY'
import importlib
import inspect

for name in ("torch", "sglang", "sglang_kernel", "triton", "flag_gems"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "<no __version__>"),
              inspect.getfile(module))
    except Exception as exc:
        print(name, "ERROR", repr(exc))
PY

python3 -m pip show torch sglang sglang-kernel triton flagtree flag-gems flashinfer-python
nvidia-smi
```

Verify plugin discovery explicitly:

```bash
SGLANG_PLUGINS=sglang_fl python3 -c 'import sglang, sglang_fl; print(sglang.__version__, sglang_fl.__file__)'
```

Run the available unit tests before changing the dependency pin. Save the exact
pass/fail/skip counts and full failure names. If the old environment cannot be
reproduced, state that instead of inventing a baseline.

---

## Stage 1: Analyze the Upstream Delta

Use official SGLang tags and source. Compare the current compatible tag with the
target tag, including intermediate releases when the gap spans several minors.

```bash
git -C <sglang_source> fetch --tags origin
git -C <sglang_source> diff --stat <old_tag>..<new_tag>
git -C <sglang_source> diff --name-only <old_tag>..<new_tag> -- python/sglang/srt python/sglang/multimodal_gen
rg -n 'from sglang|import sglang' sglang_fl tests
```

Audit high-risk surfaces even when baseline tests do not fail:

| Surface | What to compare |
|---|---|
| Plugin registration | entry points, plugin manager, `BaseFusedOp` OOT registry |
| Fused ops | call signatures, schemas, platform keys, fallback order |
| Platform | device/memory APIs, graph hooks, worker selection |
| CUDA Graph | runner split, capture modes, batch sizes, CLI flags |
| FLA/hybrid attention | public function locations and monkey-patch targets |
| KV/disaggregation | pool classes, staging metadata, registration signatures |
| Models/MoE | model-local subclasses, expert routing, top-k API |
| Dependencies | Dockerfile, package metadata, kernel and flashinfer pins |

Prefer public SGLang APIs. When a symbol moved, find its definition and callers
instead of guessing an import path. Create a short matrix:

```text
area | old API | target API | plugin dependency | planned test
```

---

## Stage 2: Build the Target NVIDIA Environment

Prefer a known SGLang runtime image or an isolated container. Registry timeout,
TLS, and DNS failures do not imply `docker login` is required; login is relevant
only to authentication errors. Follow workspace network policy and never
persist credentials in the repository.

When FlagTree supplies the target Triton module:

```bash
# Install the selected PyTorch build first.
python3 -m pip install <pinned-pytorch-command>

# Repeat until no standalone triton distribution remains.
python3 -m pip uninstall -y triton
python3 -m pip uninstall -y triton

python3.12 -m pip install "flagtree===<version>" --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

python3 - <<'PY'
import inspect
import triton
print("triton version:", triton.__version__)
print("triton module:", inspect.getfile(triton))
PY
```

Install FlagGems from the requested tag or commit. Record the resolved SHA and
package version. Keep any packaging-only wheel correction minimal and separate
from runtime changes.

```bash
python3 -m pip check
python3 -m pip freeze | grep -Ei 'torch|sglang|kernel|flashinfer|triton|flagtree|flag.?gems'
```

Do not update repository containerfiles unless container integration is an
explicit deliverable.

---

## Stage 3: Reproduce and Fix Compatibility Failures

Activate the plugin explicitly for plugin tests:

```bash
SGLANG_PLUGINS=sglang_fl python3 -c 'import sglang_fl; print("plugin import OK")'
SGLANG_PLUGINS=sglang_fl python3 -m pytest <focused_test> -x -vv
```

Fix in this order:

1. **Imports and module moves** — locate the target symbol in matching source.
2. **Plugin and fused-op registration** — prefer the target public OOT registry;
   keep a capability-gated legacy path only for still-pinned backends.
3. **Function and constructor signatures** — compare with
   `inspect.signature` and preserve optional hints without changing semantics.
4. **Platform and worker APIs** — update only plugin-owned behavior.
5. **FLA/hybrid-attention hooks** — capture original public functions before
   monkey-patching so plugin dispatch cannot recurse into itself.
6. **CUDA Graph integration** — handle changed runners and capture modes; eager
   success is not graph success.
7. **Disaggregation/KV metadata** — capability-gate staging and pool interfaces.
8. **FlagGems/operator behavior** — isolate failures per op. Prefer an explicit
   vendor CUDA fallback over broadly disabling FlagGems.
9. **Model-specific behavior** — address only after generic paths pass.

Avoid broad `except Exception` compatibility. Gate on an inspected capability,
signature, import, or version and add focused coverage for each branch.

After each fix:

```bash
python3 -m compileall -q sglang_fl
SGLANG_PLUGINS=sglang_fl python3 -m pytest <focused_test> -q
```

---

## Stage 4: Repository Verification

```bash
python3 -m ruff check sglang_fl tests
python3 -m ruff format --check sglang_fl tests
SGLANG_PLUGINS=sglang_fl python3 -m pytest tests -q
git diff --check
```

Separate real failures from unavailable-hardware or unavailable-SGLang skips.
Use `pytest.importorskip("sglang")` only when SGLang is genuinely optional for
local test discovery; never skip a regression to make the suite green. Record
exact counts and the environment that produced them.

---

## Stage 5: Real-Model NVIDIA Validation

Check GPU occupancy before every run. Start eager to isolate correctness, then
enable the intended CUDA Graph mode.

Choose models based on affected paths:

- a small dense TP=1 model for a fast smoke when available;
- one representative dense or hybrid-attention model;
- one MoE model when routing, fused MoE, or top-k changed;
- deployment TP when sharding or communication changed.

For every representative model, verify:

1. weights and KV cache load;
2. health endpoint reaches HTTP 200;
3. deterministic non-streaming generation matches the baseline;
4. streaming SSE completes and reconstructs the same content;
5. a forced long decode, for example 128 tokens, completes;
6. at least four concurrent requests succeed;
7. decode CUDA Graph captures and replays for configured batch sizes;
8. logs prove FlagGems ATen replacement and fused-op dispatch are enabled;
9. fallbacks are explicit, bounded, and non-blocking.

Compare against vanilla SGLang at the same version with identical weights,
tokenizer, prompt, dtype, sampling parameters, and TP. Record exact token IDs
when available; otherwise retain the complete greedy response and comparison
method.

Prefill/piecewise and decode CUDA Graph are separate features. Test and report
them separately with target-version CLI flags.

Use named tmux sessions. Save server logs, request payloads, responses, GPU,
dependency versions, graph mode, model, TP, and throughput. Shared-machine
throughput is a functional observation, not a benchmark claim.

---

## Stage 6: Final Review and PR

```bash
git status --short
git diff --name-status <base>...HEAD
git diff --check <base>...HEAD
rg -n 'print\(|pdb|breakpoint\(' sglang_fl tests
```

Confirm:

- no installed SGLang source changed;
- non-NVIDIA pins are untouched unless explicitly in scope;
- no temporary diagnostic override is required;
- docs/containerfiles appear only when requested;
- PR dependency versions and FlagGems SHA match the tested runtime;
- compatibility branches have focused coverage;
- model claims include model, TP, graph mode, and checks performed;
- limitations and untested follow-ups are separate from passing claims.

The PR description should contain:

```text
from/to SGLang versions
scope and excluded backends/artifacts
tested dependency tuple and GPU
compatibility changes by subsystem
unit/static test counts
model validation matrix
known fallbacks and remaining follow-ups
```

Squash only when repository policy or the user requests it. Never rewrite shared
history without explicit authorization.

## Success Criteria

The upgrade is complete only when:

- plugin import and discovery succeed on target SGLang;
- no new unit regression remains;
- vanilla and plugin-enabled deterministic behavior was compared;
- affected dense/hybrid and MoE paths pass;
- serving, streaming, long decode, concurrency, and intended CUDA Graph pass on
  real NVIDIA hardware;
- the final run uses normal plugin defaults;
- the diff and PR description match requested scope.

## Diagnosis Guide

| Symptom | Likely cause | First check |
|---|---|---|
| `ImportError` | upstream symbol moved | search matching tag and installed root |
| unexpected keyword | signature changed | compare with `inspect.signature` |
| FLA/fused-op recursion | wrapper calls itself | call captured original function |
| op not registered | OOT lifecycle changed | inspect timing and late subclasses |
| eager passes, graph fails | runner/capture drift | split decode and prefill tests |
| server loads, output wrong | silent op replacement error | compare vanilla, bisect FlagGems |
| MoE hangs/errors | top-k or routing drift | inspect fallback and MoE config |
| image pull timeout | registry/network failure | test DNS/TLS; do not assume login |
| HTTP 200, empty stream | serving/SSE regression | retain raw events |

## Related Skills

- `infer-env-setup` — reuse SSH, container, occupancy, tmux, and logging
  discipline, not its vLLM-specific install commands.
- `infer-vllm-plugin-upgrade` — analogous workflow, not SGLang API truth.
- `infer-hw-adapt` — follow after NVIDIA for another backend.
- `infer-model-adapt` — use when a model needs a separate port.
- `debug-strategy` — use after repeated failed hypotheses.
- `ops-discipline` — shell safety, state checks, and persistent logs.
