---
description: Upgrade vllm-plugin-FL to a newer vLLM release by reconciling upstream
  API and architecture changes while preserving FL dispatch and hardware backends.
  Use for version bumps, post-upgrade compatibility failures, or upgrade planning;
  use infer-model-adapt when only a new model is being ported.
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

# Upgrade vllm-plugin-FL

Treat a vLLM bump as an upstream reconciliation, not a sequence of import fixes.
The result must preserve plugin behavior, remove compatibility code that upstream
has made obsolete, and provide evidence on every hardware path claimed by the PR.

## Invariants

- Resolve the current plugin base, target vLLM tag or commit, repository roots,
  runtime environment, and exact dependency tuple before editing.
- Never edit the upstream or installed vLLM package. Change only plugin-owned
  runtime code, tests, dependency metadata, documentation, and CI when required.
- Preserve FL dispatch, platform abstraction, vendor backends, custom ops,
  FlagGems integration, graph capture, I/O dumping, and FL environment controls.
- Keep device calls behind vLLM/FL platform abstractions. Do not replace them with
  unconditional `torch.cuda` calls on code shared by non-CUDA backends.
- Preserve unrelated work. Do not stash, reset, clean, rewrite shared history, or
  squash unless repository policy or the user explicitly requires it.
- A passing import or unit suite is not completion. Runtime claims require real
  inference on the named hardware and the exact submitted commit.

## 1. Establish the upgrade contract

Inspect the repository and runtime instead of assuming paths or versions:

```bash
git status --short --branch
git log -1 --oneline
grep -n "vllm" pyproject.toml
python3 -c "import vllm; print(vllm.__version__, vllm.__file__)"
python3 -c "import vllm_fl; print(vllm_fl.__file__)"
```

Record:

- current plugin base and target vLLM tag/commit;
- plugin, upstream vLLM, model, log, and test paths;
- target hardware and whether execution is local, SSH, containerized, or CI;
- vLLM, Python, PyTorch, accelerator runtime, compiler, FlagTree/Triton,
  FlagGems, flashinfer, and plugin revisions;
- requested scope: plan only, NVIDIA reference upgrade, vendor adaptation, or
  all supported hardware.

Use a tested dependency tuple. A nominal vLLM version is insufficient because
compiler and operator behavior can change with PyTorch, Triton/FlagTree,
FlagGems, flashinfer, or the accelerator runtime. Detect duplicate or shadowed
packages, especially a standalone Triton installed alongside FlagTree.

For NVIDIA, prefer the official vLLM image or matching wheel as the reference
environment. For backends that cannot install vLLM's CUDA extension, build the
same target vLLM source with `VLLM_TARGET_DEVICE=empty`, then install the plugin
without dependencies. Keep each vendor runtime in a separate environment or
branch and validate it independently; NVIDIA success is not evidence for Ascend,
MetaX, MUSA, Hygon, Sunrise, or another backend.

If the request is planning-only, complete Sections 1-3 and return a concrete
file-by-file plan, risk register, environment matrix, and verification matrix.
Do not change code.

## 2. Read the history and measure the delta

Before patching, read both codebases and the closest earlier upgrade PRs. Find
the commits between the old and target vLLM versions that affect plugin-owned
integration points. Measure large diffs before reading them and inspect them in
bounded chunks.

Prioritize these surfaces:

1. platform registration and device capability APIs;
2. compilation and CUDA Graph wrappers;
3. worker initialization, memory accounting, and KV-cache sizing;
4. model-runner selection and execution paths;
5. custom-op registration, schemas, FLA, attention, and fused MoE;
6. dispatch configuration, vendor patches, and I/O dumping;
7. plugin model/config/patch registrations;
8. tests, packaging, environment setup, and CI.

For each surface, capture:

- upstream files/classes that changed, moved, split, or disappeared;
- plugin overrides and hooks that must survive;
- new upstream behavior that the plugin should inherit;
- compatibility code that may now be redundant;
- targeted tests and runtime paths needed to prove the adaptation.

Do not infer compatibility from matching names. Inspect signatures, factories,
abstract methods, op schemas, enum registration, output types, and initialization
order. Major transitions can include worker/model-runner splits, V1/V2 runner
selection, class-to-factory changes, and fused-MoE routing changes.

## 3. Baseline and plan before editing

Run the best available baseline against the current plugin and target runtime.
Persist full logs outside the repository or in an ignored task directory. At a
minimum, collect import results and the full unit-test failure list. When a
known-good old environment exists, keep its results separate from target-runtime
failures.

Build an ordered implementation plan with one checkpoint per component. The
usual dependency order is:

```text
platform -> compilation -> worker -> model runner -> ops/dispatch
         -> obsolete compatibility cleanup -> registration -> packaging/CI
```

The plan must also identify independent hardware passes. Start with a reference
backend when it helps isolate framework API changes, but do not silently broaden
the requested scope to every vendor.

## 4. Reconcile upstream and reapply FL behavior

For files substantially derived from upstream, use copy-then-patch:

1. save or diff the current plugin file and catalogue every FL-specific block;
2. start from the target upstream implementation or structure;
3. add a source/tag header when the repository uses that convention;
4. reapply each required FL customization deliberately;
5. compare again to confirm that no customization disappeared accidentally.

For smaller wrappers, make focused changes. Diagnose from target upstream code
first:

- moved symbol: locate its target definition and update the import;
- class/factory transition: inspect the returned object and preserve the
  unpatched original before registering a replacement;
- signature or output change: compare definitions and all call sites;
- custom op failure: compare schemas and the live `torch.ops` namespace;
- graph failure: compare capture modes, wrapper lifecycle, and memory-pool APIs;
- worker failure: compare device initialization, memory statistics, cache config,
  distributed state, and scheduler/model-runner output contracts.

After each component, run imports and focused tests before moving downstream.
Batch mechanically related edits when appropriate; the meaningful requirement is
a small, attributable verification cycle, not one commit per traceback.

### Runner and operator compatibility

If the target vLLM has multiple runner generations, define explicit selection
rules. Validate the upstream-default runner and retain an older-compatible path
only for architectures that still require it. Test selection itself so a model
cannot silently take the wrong path.

Verify native and FL-dispatched paths separately where both are supported.
Explicit operator or attention selection must be honored rather than overwritten
by a default. Derive FlagGems allow/deny entries from reproducible failures on the
target dependency tuple; minimize the list and keep a native/vendor fallback.

### Remove code that upstream now owns

For every plugin model, config, or patch considered for deletion:

1. prove the target vLLM or Transformers version implements and registers it;
2. search the plugin for imports, registration, tests, and documentation;
3. delete the obsolete implementation and its obsolete tests together;
4. keep or add coverage proving the upstream path is selected;
5. run a dangling-import search after cleanup.

Do not retain dead compatibility code merely because it imports successfully, and
do not delete vendor patches that have only superficially similar upstream code.

## 5. Validate the exact compatibility claims

Use a matrix selected from changed code paths. A substantial upgrade normally
needs:

- plugin import and component import checks;
- full unit suite, compared with the recorded baseline;
- vanilla vLLM versus plugin output on a small deterministic prompt;
- dense and MoE models when fused-MoE or dispatch changed;
- eager and compiled/graph execution when compilation changed;
- every runner generation or architecture-specific path retained by the plugin;
- explicit FlagGems/operator routing and native fallback when dispatch changed;
- I/O-dump smoke coverage when runner hooks changed;
- offline generation that finishes with a non-empty, coherent result;
- OpenAI-compatible serving health plus at least one completed request;
- longer decode, concurrency, tensor parallelism, or multimodal coverage when the
  corresponding scheduler, cache, distributed, or input path changed.

Use model paths and tensor parallel sizes that fit the actual machine. Never call
a run successful merely because weights loaded or the server opened a port. For a
generation gate, require completion and a simple deterministic content assertion.

On non-NVIDIA hardware, first prove the `empty` vLLM and plugin imports, then the
vendor platform/worker/model runner, and finally real device inference. If a
container cannot see devices, distinguish a host/container permission problem
from a plugin defect. Privileged containers or broader device mounts require the
user's authorization; without hardware access, report the runtime gate as blocked
rather than passed.

## 6. Integrate CI and finalize

When CI is in scope, verify the whole path rather than editing only a version pin:

- setup scripts assert the intended vLLM and dependency tuple;
- the selected runner labels and model paths exist;
- referenced images are pullable and their tags/digests match the validated image;
- automatic and manual vendor workflows reflect scarce-hardware policy;
- generated wheels install with `--no-build-isolation --no-deps` where required;
- runtime E2E jobs exercise the plugin rather than only importing it.

Rebase or merge the latest target base before final verification when repository
policy requires it. Resolve conflicts semantically: re-check upstream-derived
files and rerun affected tests. Do not present pre-rebase results as current-HEAD
merge gates.

Before pushing:

```bash
git diff --check
git status --short --branch
git diff --name-status <target-base>...HEAD
```

Remove task scripts, credentials, model outputs, and standalone diagnostics from
the diff unless explicitly requested. Push only task-owned commits.

The PR report must separate:

- target base and exact plugin commit;
- runtime/dependency tuple and hardware;
- code and compatibility cleanup;
- current-HEAD passing evidence;
- older supporting evidence;
- untested backends and remaining blockers;
- intentional out-of-scope work.

Never claim a backend was validated when it only received import-path changes.

## Recovery

- Reproduce a failure with vanilla vLLM before blaming the plugin.
- After two unsuccessful fixes to the same symptom, return to the relevant target
  upstream implementation and the introducing commit instead of stacking guesses.
- If the version gap is too large to reason about safely, use intermediate tags
  for analysis or implementation checkpoints, but still validate the final target.
- Preserve logs and commit identifiers so work can resume without rerunning proven
  stages. Revert only task-owned changes when backing out.

## Related Skills

- `infer-env-setup` for provisioning the runtime.
- `infer-vllm-hw-adapt` for deeper vendor-backend implementation after the shared vLLM
  upgrade is established.
- `infer-model-adapt` when only a new model or architecture is being added.
- `infer-precision-check` for detailed numerical comparison.
- `debug-strategy` and `ops-discipline` for systematic diagnosis and safe remote
  execution.
