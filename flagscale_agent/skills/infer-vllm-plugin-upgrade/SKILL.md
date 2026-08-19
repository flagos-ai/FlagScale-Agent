---
description: Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers
  version detection, API diff analysis, targeted fixes, and validation across unit
  tests, offline inference, and serving. Applies to any vLLM minor version bump (e.g.,
  0.20.x to 0.24.x).
name: infer-vllm-plugin-upgrade
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

# Plugin Upgrade to New vLLM Version

Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. A minor bump brings two kinds of changes: breakages (surface as import/type/attribute errors) and silent behavioral shifts (require proactive audit of high-risk areas).

## Prerequisites

Before starting the upgrade, ensure the environment is ready (via `infer-env-setup`):
- SSH connection to NVIDIA GPU machine confirmed
- Docker container running with correct image and GPU mounts
- vLLM installed, vllm-plugin-FL in editable mode
- All imports verified (`import vllm`, `import vllm_fl`)

## Critical Rules

1. **Auto-detect versions first** -- never assume plugin or vLLM version. Always read from installed packages and pyproject.toml.
2. **Never modify vLLM source** -- all fixes go through `vllm_fl/` plugin files only.
3. **One patch per failure** -- fix, re-test, then move to the next error. Never batch unverified fixes.
4. **Fix order matters**: imports -> class/factory API -> signature kwargs -> op schemas -> model-specific.
5. **NVIDIA GPU is ground truth** -- validate every fix on real hardware before declaring done.
6. **Stream and persist logs** -- use `2>&1 | tee <log_dir>/<stage>_<timestamp>.log`.
7. **Squash before PR** -- all upgrade commits squashed into one clean commit.

---

## Stage 0: Workspace Orientation and Version Detection (MANDATORY)

Run before ANY work. Never skip. All paths must be probed -- never assumed.

```bash
ssh <host> "docker exec <container> bash -c '
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  find / -path \"*/vllm-plugin-FL/pyproject.toml\" 2>/dev/null | head -3 | xargs grep -E \"vllm|version\" &&
  python3 -c \"import vllm, os; print(os.path.dirname(vllm.__file__))\"
'"
```

Immediately record to memory:
```
memory_write('<prefix>_vllm_version', '<version>')
memory_write('<prefix>_plugin_root', '<discovered_plugin_root>')
memory_write('<prefix>_vllm_root', '<discovered_vllm_root>')
```

If installed vLLM version != plugin declared compatible version, version gap is confirmed -- proceed.

Create a rollback point before any changes:
```bash
ssh <host> "docker exec <container> bash -c 'cd <plugin_root> && git stash'"
```

**Never guess paths. Always read from memory or re-probe.**

---

## Stage 1: Change Analysis

Before touching any code, enumerate what broke and what's new.

### 1a. Unit test baseline -- find all current failures

```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> bash -c '
  cd <plugin_root> &&
  python3 -m pytest tests/unit_tests/ --tb=short -q \
  2>&1 | tee <log_dir>/unit_baseline.log
'"
```

Capture the full failure list. Categorize each by error type:
- `ImportError` / `ModuleNotFoundError` -- moved symbols
- `TypeError` -- stale kwargs or signature change
- `RecursionError` -- class vs factory confusion in patching
- `AttributeError` -- removed attributes or missing op registrations

### 1b. Identify plugin files that import from vLLM

```bash
ssh <host> "docker exec <container> grep -rn 'from vllm\|import vllm' \
  <plugin_root>/vllm_fl/ --include='*.py' -l"
```

Each of these files is a potential breakage point. Cross-reference with the error list from 1a.

### 1c. Audit high-risk areas

Even if tests pass, proactively check areas that frequently change across vLLM bumps:
- Plugin override points (worker, model_runner, scheduler)
- Op registration/schema files
- Any class the plugin subclasses or monkey-patches

Compare the plugin's assumptions against the new vLLM code:
```bash
ssh <host> "docker exec <container> bash -c '
  diff <(grep -n \"def <function>\" <vllm_root>/vllm/<module>.py) \
       <(grep -n \"def <function>\" <plugin_root>/vllm_fl/<module>.py)
'"
```

---

## Stage 2: Fix API Breakages

Fix in strict order. Each fix gets its own verify cycle.

### Fix methodology

1. **Imports** -- trace the moved symbol with `grep -rn "class X\|def X" <vllm_root>/vllm/`, update the import path in `vllm_fl/`.

2. **Class/factory API** -- if plugin patches a class that vLLM replaced with a factory (or vice versa), capture the original before patching. Use `type(obj)` and `inspect.getmro()` to understand the new structure.

3. **Signature kwargs** -- use `inspect.signature()` to compare old vs new function signatures. Add/remove kwargs in plugin's override accordingly.

4. **Op schemas** -- compare plugin's registered ops against actual `torch.ops` namespace. Update schema definitions to match.

5. **Model-specific** -- only after all generic fixes. Test with the specific model that fails.

### Per-fix verification pattern

After EVERY single fix:
```bash
# Quick import check
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> \
  python3 -c 'import vllm_fl; print(\"plugin import OK\")'"

# Targeted test (if the fix addresses a specific test)
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> \
  python3 -m pytest <plugin_root>/tests/unit_tests/<specific_test>.py -x -v \
  2>&1 | tail -20"
```

Never proceed to the next fix without confirming the current one passes.

---

## Stage 3: Unit Test Verification

After all fixes from Stage 2, run the full unit test suite:

```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> bash -c '
  cd <plugin_root> &&
  python3 -m pytest tests/unit_tests/ --tb=short -q \
  2>&1 | tee <log_dir>/unit_final.log
'"
```

Compare against the baseline from Stage 1a:
- All NEW failures from the upgrade must be fixed (zero regression)
- Pre-existing failures (present before upgrade) are acceptable -- document them
- If a test was passing before and fails now, it MUST be fixed before proceeding

---

## Stage 4: Offline Inference Validation

Test with at least one representative model to verify full forward pass works.

```bash
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> bash -c '
  python3 -c \"
from vllm import LLM, SamplingParams
llm = LLM(model=\\\"<model_path>\\\", tensor_parallel_size=<tp>)
outputs = llm.generate([\\\"Hello, my name is\\\"], SamplingParams(temperature=0, max_tokens=50))
for o in outputs:
    print(f\\\"Output: {o.outputs[0].text}\\\")
\" 2>&1 | tee <log_dir>/offline.log'"
```

Check: output contains coherent text (not empty, not garbled, not repeated tokens).

If the plugin supports multiple model architectures, test each category that has dedicated plugin code paths.

---

## Stage 5: Serving Validation

Final validation -- the plugin must work in the full serving stack.

```bash
# Launch server
ssh <host> "docker exec -e VLLM_PLUGINS=fl <container> bash -c '
  python3 -m vllm.entrypoints.openai.api_server \
    --model <model_path> --tensor-parallel-size <tp> \
    --host 0.0.0.0 --port 8000 \
    > <log_dir>/serve.log 2>&1 &
  echo \$!
'"

# Wait for ready
ssh <host> "docker exec <container> bash -c '
  for i in \$(seq 1 60); do
    curl -sf http://localhost:8000/health && break
    sleep 5
  done
'"

# Test completion
ssh <host> "curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"<model_path>\", \"prompt\": \"Hello\", \"max_tokens\": 20}'"

# Cleanup
ssh <host> "docker exec <container> pkill -f api_server"
```

Check: response contains non-empty generated text.

---

## Stage 6: PR Submission

Before opening a PR:

1. **Review all changes**: ensure no debug prints, temporary patches, or commented-out code remain.
2. **Verify no vLLM source files were modified**: only `vllm_fl/` files should appear in the diff.
3. **Squash all commits** into one clean commit.
4. **Run unit tests one final time** on the squashed commit.

Commit message format:
```
feat(plugin): upgrade vllm-plugin-FL compatibility to vLLM <version>

- <one line per fix, describing what broke and how it was resolved>

Tested: unit tests, offline inference, serving on NVIDIA
Models validated: <list>
```

Push and open PR:
```bash
git -C <plugin_root> push origin <branch> -u
gh pr create --title "feat(plugin): upgrade to vLLM <version>" --body-file <body_file> --base main
```

---

## Error Diagnosis Guide

| Error Type | Likely Cause | Diagnosis Approach |
|-----------|-------------|-------------------|
| `ImportError` | Symbol moved to different module | `grep -rn "class X\|def X" <vllm_root>/vllm/` |
| `TypeError: unexpected keyword` | Function signature changed | `python3 -c "import inspect, vllm.<mod>; print(inspect.signature(...))"` |
| `RecursionError` | Plugin patches class that became factory | Check `type()` of the object before patching |
| `AttributeError` | Attribute removed or renamed | Read the new vLLM source for the class |
| Tests pass but inference garbled | Silent behavioral change | Compare output with vanilla vLLM (no plugin) |
| Server hangs on request | Async/scheduling API change | Check server logs, compare with vanilla vLLM |

### Recovery discipline

1. Read FULL error output -- multiple issues may coexist
2. If stuck after 2 attempts on same error -> step back, read vLLM changelog or git log for the breaking commit
3. If the upgrade breaks too many things -> consider incremental approach (one minor version at a time)
4. Restore rollback point if needed: `git stash pop`

---

## Related Skills

- `infer-env-setup` -- set up the container and environment from scratch
- `infer-hw-adapt` -- hardware-specific backend adaptation (non-NVIDIA)
- `infer-model-adapt` -- port a new model into the plugin
- `infer-precision-check` -- verify inference output correctness
- `debug-strategy` -- systematic debugging when stuck
- `ops-discipline` -- shell safety and environment awareness
