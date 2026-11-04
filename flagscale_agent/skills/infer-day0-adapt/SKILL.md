---
description: Deliver Day 0 support for a newly released model in vllm-plugin-FL
  or sglang-plugin-FL, including models not yet supported upstream. Covers model
  and checkpoint analysis, operator gap resolution, plugin integration, and
  verified cross-repository delivery. Use dedicated upgrade or hardware skills
  when the primary goal is a framework version bump or a new device backend.
name: infer-day0-adapt
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

# Day 0 Model Adaptation

Take a newly released model from authoritative architecture and checkpoint
artifacts to an executable, verified FL plugin integration. Adapt the frameworks,
devices, and features requested by the user; validate each claimed combination.

## When to Use This Skill

Use this skill for first-day model support requiring architecture/checkpoint
analysis, missing operators, or integration across framework and operator
repositories. It also covers models without an upstream implementation.

Use `infer-model-adapt` for an existing upstream vLLM model's copy-and-patch
port. Use framework upgrade or hardware adaptation skills when those changes
are the primary task; do not expand a model adaptation into either by default.

## Critical Rules

These rules apply the Inference Framework Plugin Development Specification v1
(plugin clauses A-H) and the New Model Adaptation Operator-Layer Development
Specification (operator clauses A-F). MUST preserves a source requirement;
SHOULD preserves a recommendation whose exception needs justification and
validation. Follow the target branch's actual layout, interfaces, and checks.

1. **Establish exact revisions.** Record model, checkpoint, framework, plugin,
   operator libraries, and runtime before choosing an implementation.
2. **Reuse by contract.** Inspect mainline and dependencies; matching function
   names do not prove compatible parameters, layouts, state, or error behavior.
3. **Keep ownership explicit.** Put orchestration and scheduling metadata in the
   plugin, numerical computation in the appropriate operator library, and
   persistent resources under framework ownership.
4. **Use public APIs and existing dispatch.** Do not copy kernels into model
   code, load private operator files, modify installed third-party source, or
   introduce hidden PyTorch fallback in production.
5. **Select before execution.** Probe real APIs/device capabilities and choose
   the backend before side effects. Once a launch, mutation, state update, or
   collective starts, propagate errors without retrying another backend.
6. **Require independent evidence.** Assert actual backend hits and compare
   numerics with an independent implementation. Imports, stubs, plausible text,
   or an arbitrary number of matching tokens do not establish correctness.
7. **Deliver dependencies in order.** Operator public API and tests first,
   plugin integration second, then a publicly accessible fixed dependency.
8. **Report support honestly.** Missing hardware, weights, or references leave
   explicit verification gaps; static checks are not target-device validation.

## Adaptation Pipeline

### Step 0: Establish the Baseline

Inspect the workspace and repository guidance before asking for missing inputs.
Record:

- model/checkpoint identifier and immutable revision, config, tokenizer,
  processor, weight index, `architectures`, and `model_type`;
- plugin checkout/commit, installed and declared framework versions, operator
  libraries, compiler/runtime, dependency pins, and resolved import paths;
- target device/vendor, available memory/devices, execution environment,
  dtype, TP/EP and other parallel layouts, and eager/graph scope;
- requested dense/MoE, multimodal, attention/cache, quantization, offline,
  serving, concurrency, and workload coverage;
- independent model/operator references, fixed input data, generation settings,
  and repository-defined numerical, quality, and performance criteria.

Establish relevant pre-change tests. Check current upstream model support at a
recorded primary-source revision; do not assume latest source works with the
installed framework. If neither network access nor a versioned upstream snapshot
is available, mark upstream support unverified and continue from authoritative
model artifacts. Inspect checkpoint metadata before downloading full weights.
Follow the environment's storage and long-running job conventions.

### Step 1: Analyze Architecture and Operator Gaps

Trace config -> preprocessing -> loading -> forward -> cache/state -> outputs.
Compare with the closest supported model: attention, RoPE, norm/activation,
residual order, router/shared experts, weight tying, parallel shards, and
multimodal token/feature alignment. Check names and shapes against the checkpoint.

Record a working gap matrix in task notes or the PR:

| Model stage / public call | Contract | Existing API and revision | Gap / owner | Verification |
| --- | --- | --- | --- | --- |
| Each affected stage | dtype, layout, mask, mutation, state, errors | Exact symbol/backend | reuse / thin API / extend / new | Independent reference and route assertion |

Choose ownership by actual reads, writes, and lifecycle (plugin A1-A4; operator A1-A4):

| Responsibility | Owner |
| --- | --- |
| Model/config, processor, weight mapping, runtime orchestration | Target framework plugin |
| Cross-framework tensor and fused numerical computation | FlagGems |
| Numerical computation tied to vLLM layout/state/call contracts | FlagGems-vllm |
| Numerical computation tied to SGLang layout/state/call contracts | FlagGems-sglang |
| Request/token/slot/block mappings, copy plans, persistent resources | Framework, even when metadata is prepared on GPU |

Score-based top-k, numerical KV reads/writes, quantization, and format conversion
are computation. Split mixed functions into framework metadata preparation and
operator consumption. Leave unrelated historical kernels alone; preserve
behavior and migrate tests when migration is in scope. SHOULD inspect vendor
copies of changed generic operators for shared semantic fixes without erasing
hardware specialization.

### Step 2: Choose the Model Integration Path

- **Compatible upstream model exists:** reuse public layers and interfaces. If
  copying model code is necessary, preserve its source revision/license and
  make focused compatibility changes in plugin-owned files.
- **Only newer upstream supports the model:** assess a model backport separately
  from infrastructure changes. Use the relevant upgrade skill only when an
  infrastructure upgrade is necessary and within scope.
- **No upstream implementation exists:** derive the plugin model from official
  model code, config, checkpoint, and architecture description. Reuse compatible
  layers and current extension points; do not wait for an upstream port or
  invent APIs. Keep the official implementation independent for comparison.

Inspect the actual model/config/processor registry contracts: `architectures`
and `model_type` are distinct and neither is automatically the registry key.
Check fused QKV/gate-up mappings, tied weights, missing/unexpected parameters,
and parallel shards. Reject incompatible required parameters before computation.
For SGLang, trace a supported model through its pinned registry, loader, layer
construction, and dispatch entrypoints rather than transplanting vLLM APIs.

### Step 3: Deliver Missing Operators

Apply operator B1-F6 when numerical implementations change. Use existing
compiler, launcher, autotune, reference, test, and benchmark infrastructure.
Generic ops belong in the existing `ops/` directories; FlagGems fused APIs use
`fused/`. Genuine vendor specializations and tuning belong under the existing
`runtime/backend/_<vendor>/` layout; framework-specific vendor ops use its
`ops/` subdirectory. SHOULD avoid vendor copies that only forward generic code.
Do not register a reference implementation as vendor support.

Complete each repository's public call chain:

| Repository | Required integration |
| --- | --- |
| FlagGems | Public exports, actual ATen schema/overload, `conf/operators.yaml`, `_FULL_CONFIG`, test selectors; fused APIs use their separate path |
| FlagGems-vllm | Public exports, `conf/operators.yaml`, backend registration, test markers |
| FlagGems-sglang | Real operators in module `__all__`, registrar/configuration; independent references under `src/flaggems_sglang/reference/` via `get_reference()` |

For FlagGems, retain exact registration keys, unique/ordered metadata, and each
declared default, in-place, `out=`, scalar, and backward variant. Fused non-ATen
ops must not enter `_FULL_CONFIG`. Direct `flag_gems.<op>` precision tests and
registration-dispatch tests establish different requirements; run both. Do not
alias `reference` or `solution` to production SGLang operators.

Define the full contract before coding and implement it faithfully:

- parameters/defaults, return structure, supported variants and input domain;
- each tensor's own shape, dtype, device, stride, storage offset, broadcasting,
  and output layout; assume contiguity only when required by the API;
- mutation, aliasing, caller-provided buffers, view storage sharing, cache/state
  write timing, and untouched regions; preserve schema-required resize/overlap
  semantics, and return the original buffer for `out=`;
- masks, causal/scale/bias/window/index parameters, promotion, accumulation,
  rounding, intermediate order, packed weights, and applicable NaN/Inf/extremes;
- empty inputs, invalid combinations, exception types/timing, index width,
  masked-load fill values, and masked stores for tails and large address ranges.

Reject unsupported input before writes; never silently clamp invalid indices,
ignore semantic arguments, or use removable `assert` for user-input validation.
Use the correct device context. Autotune must restore affected inputs/outputs/
state after every candidate. Optional acceleration checks must not pollute
compilation/cache keys or skip the baseline tests. SHOULD avoid unnecessary hot
path synchronization, device-to-host reads, launches, and Python element loops.

Use project loggers, English source/comments/errors, and required license/source
headers. FlagGems public entries log `GEMS <OP>` with variant suffixes and vendor
entries use `GEMS_<VENDOR>`. In FlagGems-vllm, a public vendor implementation's
first executable statement after its docstring is
`logger.debug("GEMS_<VENDOR> <OP>")` with repository-correct names.

### Step 4: Wire Plugin Dispatch and Runtime State

Apply plugin A2, B1-B4, C1-C4, D1-D5, and E1-E4. Put model-specific code in the
existing model entrypoint, capability/default/attention/graph selection in the
platform layer, and public op contracts/registration/policy/bridges in existing
`dispatch/`. Use generic, vendor, and reference backends for their actual roles.
New or replaced numerical backends must use this chain; retain normal upstream
layer reuse. Do not branch directly to a generic/vendor op from model `forward`.

Check actual APIs, signatures, and device capabilities before optional imports
and backend selection; importability or version alone is insufficient. Scope
patches to the intended model/platform/conditions, install centrally and
idempotently, and validate dependent patches before atomically mounting them.
On installation failure restore all objects/wrappers and do not cache success.
SHOULD use capability-based compatibility and real upstream state.

Reuse existing configuration before adding switches. A new switch needs owner,
model/platform scope, default, full propagation/precedence, user documentation,
and enabled/disabled/unset tests. Copy shared configuration before mutation.
Keep platform defaults in platform hooks; define merge/replacement semantics for
lists and overrides and test the effective value. Debug switches must not write
files by default. Clean removed switches through code, tests, docs, builds, CI,
and deployment. SHOULD give temporary blacklists a tracking issue and exit condition.

Declare mutation/aliasing accurately in APIs and custom-op schemas. The framework
owns persistent cache/state, memory budgets, graph workspace, stable addresses,
concurrent isolation, reuse, and release; wrappers may allocate ordinary outputs
and call-local temporaries. Establish dependencies on the actual current/capture
stream before consuming asynchronous work. Prepare graph workspace before
capture and keep addresses valid through replay; avoid capture-illegal
synchronization and blocking CPU-device transfers. Reuse costly streams/communicators
where appropriate and roll back partial initialization, with failure-path tests.
A launch log cannot claim completed computation or numerical validation.

### Step 5: Check Conditional Model Features

Run only the branches implicated by the requested support and changed contracts.

**Quantization (plugin H1-H10):**

- Record checkpoint format, quantization scheme, and execution backend separately;
  specify weight/activation bits and dtype, static/dynamic scales, granularity,
  supported layers, packing, scale/inverse-scale, zero-points, rounding,
  accumulation/output dtype, and alignment/padding. Equal bit width does not
  make FP8/FP4/INT4 formats interchangeable.
- Keep config/layer selection/loading lifecycle in the plugin and numerical
  quantize/dequantize/repack/GEMM/MoE/KV conversions in operator libraries,
  including initialization-time transforms. Use the pinned framework's public
  registration and layer-method extension points; discover SGLang equivalents
  through existing implementations.
- Preserve explicit-user/checkpoint/per-layer precedence, excluded/ignored layers,
  prefix and packed-module mappings, fused-member scales, mixed precision, and
  tied weights. Distinguish inapplicable, intentionally unquantized, and unsupported
  layers. Reject config conflicts and missing required fields before execution.
- Declare parameter shapes/dtypes/shard and packing axes/loaders; load all required
  scales, zero-points, and indices. Complete repack/scale/layout preparation after
  loading and before inference/capture; prevent repeated conversion and per-token
  whole-weight processing. Requantization needs independent source dequantization.
- Distinguish serialized quantized checkpoints, online weight quantization, and
  dynamic activation quantization. Test each claimed entry separately; budget
  peak loading memory. Record calibration data/tools when needed and keep
  calibration data separate from acceptance data.
- Treat KV quantization separately: validate storage versus attention compute dtype,
  K/V and required Q/probability scales, static/dynamic scale mapping and consumption,
  invalid slots, and cache reuse/lifetime. Follow the format's missing-scale
  semantics; neither unconditional scale=1 nor forcing dynamic scales to static
  is valid. Cover affected prefill/decode, chunked/prefix/paged cache, and context lengths.
- Verify TP/EP sharding of weights/scales/zero-points/indices, group/block boundaries,
  padding, and expert reordering. Preserve router/top-k/shared experts, routing
  weight timing, and collective order/count. Dense Linear success does not prove MoE.
- Separate implementation error against the same quantized semantics from quality
  degradation against high precision. Use fixed data/settings and justified
  numerical/quality thresholds. Test affected formats, layers, nonaligned sizes,
  extremes, cache updates, and missing-scale/config/device failures. Reusing an
  unchanged quantization path needs affected-path validation, not reimplementation.

**MoE and multimodal:** preserve routing, residual/shared-expert order, intermediate
dtypes, communication, processor/input shapes, and token/feature alignment. Verify
actual checkpoint loading and representative inputs for each claimed path.

**A new vendor backend (plugin B5):** use its dedicated hardware adaptation skill
when in scope. Device identity alone is insufficient: validate registration,
per-op policy, defaults, actual communication, graph capabilities, dependencies,
platform tests, CI routing, and container configuration.

### Step 6: Validate the Claimed Support

Record pass, fail, not run, or not applicable with a reason for each affected item:

1. **Static/build:** formatter, linter, pre-commit, imports, model/config/operator
   registration, unit tests, and compilation. If packaging changes, inspect and
   install the actual artifact and verify public dependency entrypoints.
2. **Real dispatch:** start at the public business entry and assert the target
   backend/kernel and exact parameters. Cover nonmatching routes, missing optional
   capability, unaffected platforms, and execution-time errors without retry.
3. **Independent numerics:** compare operators/layers/logits or applicable model
   outputs against a reference that does not call the tested implementation or
   share potentially faulty production calculations. Cover declared dtypes,
   shapes, non-contiguous/offset layouts, masks, boundaries, empty/invalid inputs,
   variants, mutation/aliasing, and untouched state. Restore global config/seeds/
   device state in teardown; justify skip/xfail with scope and a tracking issue.
4. **Real model:** run a usable offline example and actual serving requests when
   serving is claimed. Check output content and affected prefill/decode, cache,
   parallel, MoE, multimodal, and concurrency paths. Do not weaken tolerances or
   assertions to hide defects.
5. **Graph, when affected or claimed:** perform real capture/replay; update inputs
   before replay, sentinel-fill independent outputs, and reset mutated/aliased
   inputs/state so reference and replay start identically. Compare final outputs/
   state across affected batch, concurrent, distributed, and cache scenarios.
   Eager execution, empty capture, and stale output do not establish graph support.
6. **Performance:** for operator implementations/optimizations use the repository
   benchmark with verified baseline/target backends on the same device, workload,
   dtype, streams, warmup, graph mode, iteration count, and rank synchronization.
   Report baseline/target latency, speedup, regressions, and uncovered cases under
   project thresholds. SHOULD assess model performance and quantization loading
   cost with fixed lengths, batch/concurrency, and parallelism: TTFT, TPOT/ITL,
   throughput, peak/steady memory, and load time, with preprocessing/compile/warmup
   accounting. A favorable kernel shape or theoretical bit ratio is insufficient.

When hardware, model assets, or reference execution are unavailable, continue the
independent work possible and preserve runnable tests/examples. Record exact
blockers and commands still needed; do not claim the unverified support.

### Step 7: Deliver Reproducible Support

Deliver operator public APIs/tests, then plugin integration, then a released
version or publicly accessible immutable commit pin, consistently in installation
and build configuration (plugin D3, F4-F7, G1-G2). Identify dependent PRs and open
gates; private/floating branches and unmerged dependencies are not completed
delivery. While waiting, continue model analysis and test/example preparation.

Update the plugin README's existing model-support table/list with evidenced
status and a link to a real committed or verified existing example. Include
required arguments, dependencies, hardware, launch commands, and limits without
credentials or personal paths. Keep transient logs, benchmark data, and experiment
reports in the PR/CI artifacts; maintain usage and compatibility in user docs.

Include in the PR description:

- model/checkpoint/source revisions, framework/plugin/operator commits, dependency
  pins, target device/runtime, dtype, parallel and graph modes;
- architecture/weight changes, operator gaps/ownership/public APIs, actual backend
  hits, cross-repository dependencies, and runnable examples;
- exact checks/commands, independent references and justified thresholds,
  numerical/quality/performance results, failures, and unverified scope;
- support per device/format/feature: **implemented** (code/static checks),
  **loadable** (real target checkpoint loading), **numerically verified**
  (independent comparison), **end-to-end verified** (real workload/quality gates).

Required CI jobs must actually run; an all-skipped hardware matrix is not a pass.
Re-run affected checks on the final code after conflict resolution. SHOULD keep
PRs focused and justify shared infrastructure changes with caller regressions.
Remove temporary probes, dead paths, and unrelated changes before delivery.

## Related Skills

- `infer-env-setup` - prepare a missing inference environment.
- `infer-model-adapt` - existing vLLM upstream-copy porting techniques; this skill's
  contract and evidence requirements govern Day 0 delivery.
- `infer-vllm-hw-adapt`, `infer-sglang-hw-adapt` - new hardware backend support.
- `infer-vllm-plugin-upgrade`, `infer-sglang-plugin-upgrade` - framework upgrades.
- `infer-precision-check`, `debug-strategy` - deeper numerical diagnosis using
  independent references and repository-defined acceptance criteria.
