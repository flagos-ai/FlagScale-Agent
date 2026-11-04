---
name: infer-day0-adapt
description: >-
  Deliver Day 0 (首日适配) support for a newly released model in vllm-plugin-FL
  or sglang-plugin-FL, including models not yet supported upstream. Use for
  model architecture and checkpoint analysis, operator gap analysis, framework
  integration, and verified cross-repository delivery. Covers conditional MoE,
  multimodal, quantization, KV cache, and graph paths; use framework upgrade or
  hardware adaptation skills for tasks whose primary goal is upgrading the
  framework or adding a device backend.
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

Take a newly released model from its authoritative architecture/checkpoint to
an executable, verified FL plugin integration. Day 0 describes the timing of
support, not an exemption from correctness or delivery requirements. Adapt the
frameworks and features requested by the user; support in one framework or on
one device does not establish support elsewhere.

## References and scope

This workflow is derived from **推理框架插件开发规范v1** (plugin clauses A–H)
and **新模型适配算子层开发规范** (operator clauses A–F). The following
references preserve their task-specific requirements and clause identifiers:

- Read [framework-contracts.md](references/framework-contracts.md) before
  changing plugin code, dispatch, platform behavior, or runtime state.
- Read [operator-contracts.md](references/operator-contracts.md) when adding,
  migrating, registering, or modifying an operator or backend implementation.
- Read [quantization.md](references/quantization.md) when loading a quantized
  checkpoint or changing quantization configuration, layer methods, numerical
  backends, or weight/activation/KV formats. Reusing an unchanged quantization
  path requires validating the affected integration, not reimplementing it.

Paths are relative to this skill directory. `load_skill` returns the entrypoint
only; read the relevant reference files with `read_file`. For a bundled install,
locate the directory with:

```bash
python -c "from importlib.resources import files; print(files('flagscale_agent').joinpath('skills/infer-day0-adapt'))"
```

Use the target repository's current development instructions, directory layout,
registration mechanisms, and checks. Paths in the references describe existing
responsibilities; do not create a parallel architecture to match an example.
**Must** denotes a source requirement; **should** denotes a default recommendation
whose justified exceptions need an explanation and validation.

## 1. Establish the model and execution baseline

Discover existing workspace, repository, model, and environment information
before asking for missing inputs. Record:

- Model/checkpoint identifier and immutable revision; config, tokenizer,
  processor, weight index, architecture class names, and `model_type`.
- Target plugin(s), checked-out commits, installed and declared framework
  versions, operator library commits, compiler/runtime, and dependency pins.
- Target device/vendor, available memory/devices, execution environment,
  requested dtype, TP/EP and other parallel layouts, and graph/eager scope.
- Required features: dense/MoE, multimodal inputs, attention/cache variants,
  quantization format, offline inference, serving, and intended workloads.
- Independent reference implementation, fixed input data, generation settings,
  and repository-defined numerical/quality/performance acceptance criteria.

Read local development guidance and establish relevant pre-change tests.
Check current upstream model support against primary sources at a recorded
revision; do not infer it from a release name or assume latest source works
with the installed framework. Inspect checkpoint metadata before downloading
full weights. If neither network access nor a versioned upstream snapshot is
available, record upstream support as unverified and continue work based on the
available authoritative model artifacts; do not guess upstream support.
Use available model assets and hardware within the task scope;
follow the environment's long-running job and storage conventions.

## 2. Analyze architecture and build an operator gap map

Trace config → preprocessing → weight loading → forward → cache/state updates
→ outputs. Compare with the closest supported model, including attention type,
RoPE/norm/activation, residual order, routing/shared experts, weight tying,
parallel shards, and multimodal token/feature alignment where applicable.
Validate shape and name mappings against actual checkpoint metadata.

Build a working matrix in the task notes or PR, not a new user-facing report:

| Model stage / public call | Contract and supported inputs | Existing API / backend | Gap and owning repo | Verification |
|---|---|---|---|---|
| Each affected stage | dtype, layout, mask, mutation, state, errors | Exact symbol and revision | reuse / thin API / extend / new | Independent reference and route assertion |

Inspect the current mainline and installed dependencies before writing kernels.
Same-named functions are not evidence of compatible contracts. Choose ownership
by semantics:

| Responsibility | Owner |
|---|---|
| Architecture, config, processor, weight mapping, runtime orchestration | Target framework plugin |
| Cross-framework tensor or fused numerical computation | FlagGems |
| Numerical computation tied to vLLM layout/state/call contracts | FlagGems-vllm |
| Numerical computation tied to SGLang layout/state/call contracts | FlagGems-sglang |
| Request/token/slot/block mapping and persistent resource ownership | Framework, even if metadata is computed on GPU |

Score-based top-k, numerical KV reads/writes, quantization, and format conversion
are computation. Split mixed functions into framework metadata preparation and
operator consumption. Avoid unrelated migration of historical kernels.

## 3. Choose and implement the integration path

- **Compatible upstream implementation exists:** reuse existing public model
  layers when they satisfy the contract. If a model copy is needed, preserve
  its source revision/license and apply focused compatibility changes in the
  plugin; do not edit installed third-party source.
- **Upstream model exists only on a newer revision:** separate model-specific
  backports from framework infrastructure changes. Use the corresponding
  `infer-vllm-plugin-upgrade` or `infer-sglang-plugin-upgrade` skill only if an
  infrastructure upgrade is actually needed and within the requested scope.
- **No upstream implementation yet:** derive the plugin model from the official
  model code, config, checkpoint, and architecture description. Reuse compatible
  layers and implement the missing orchestration through current extension
  points. Do not require an upstream port as a prerequisite or invent APIs.
  Keep the official implementation independent for numerical comparison.

Follow the framework reference for registration, dispatch, patches, and state.
Inspect the actual registry key contract: `architectures` and `model_type`
serve different roles; never assume either is the registration key. Implement
config/processor registration and weight loaders where required. Check fused
QKV/gate-up mappings, tied weights, missing/unexpected weights, and parallel
shards; reject incompatible required parameters before computation.

For operator gaps, first deliver the operator library's public API, exports,
registration, independent tests, and benchmark. Then integrate the plugin
through its existing call bridge and unified dispatch. Keep kernels out of
model modules and vendor glue; do not load private implementation files.
Vendor backends must represent real vendor implementations. Probe capabilities
and select a compatible backend before execution; errors after any launch,
mutation, state update, or collective must propagate without backend retry.

Use existing platform/configuration mechanisms. Scope and atomically install
necessary patches, with idempotence and rollback tests. Keep persistent cache,
workspace, address stability, memory budgets, and stream dependencies under
framework ownership. Validate each meaningful batch of changes before moving on.

## 4. Validate the claimed support

Derive the matrix from the requested features and actual changed contracts.
Record pass, fail, not run, or not applicable with a reason for each item;
unavailable hardware or weights leave a verification gap, not a passing result.

1. **Static and build checks:** run applicable formatter/linter/pre-commit,
   imports, registration and configuration checks, unit tests, and compilation.
   If packaging changes, inspect and install the actual build artifact.
2. **Real dispatch:** start at the public model/operator entry and assert the
   actual backend and parameter forwarding. Exercise nonmatching routes,
   missing capabilities, non-target platforms, and execution errors with no
   retry. Mock/stub results establish wiring only.
3. **Independent numerics:** compare affected operators/layers and logits or
   appropriate model outputs against the independent reference. Cover declared
   shapes/dtypes/layouts, masks, empty and invalid inputs, boundary values,
   mutation/aliasing, and untouched state. Follow the repository's justified
   tolerances; do not weaken assertions or use an arbitrary matching-token
   count as the acceptance standard.
4. **Real model:** run a usable offline example and, when serving is claimed,
   a real request with output-content validation. Cover prefill/decode and the
   affected cache, parallel, MoE, multimodal, and concurrency paths. A started
   process, successful import, or plausible generated text alone is insufficient.
5. **Graph, if affected or claimed:** perform real capture and replay on new
   input, sentinel-fill independent outputs, and reset mutations/state so the
   reference and replay share the same initial conditions. Check affected batch,
   concurrent, distributed, and cache scenarios; eager success proves no graph
   support. Use the reference's resource and synchronization requirements.
6. **Performance:** for operator implementations/optimizations, run the existing
   benchmark against a verified baseline on the same device, shapes, dtype,
   streams, warmup, graph mode, and rank synchronization. Report baseline and
   target latency, speedup, regressions, and uncovered cases. For model-level
   performance claims, record the workload, TTFT, TPOT/ITL, throughput, and memory
   as applicable; do not extrapolate from one favorable operator shape.

If reference execution or target hardware is unavailable, continue the
independent work possible, preserve runnable tests/examples, and report exact
blockers and commands still needed. Do not label the model fully supported.

## 5. Deliver reproducible support

Cross-repository order is **operator public API and tests → plugin integration
→ fixed dependency**. Use a released version or publicly accessible immutable
commit, consistently in installation and build configuration. A local checkout
or unmerged dependency branch is not a completed dependency delivery; identify
dependent PRs and unresolved gates explicitly.
This is the delivery gate order: while an operator dependency awaits hardware
validation or publication, continue model analysis and prepare tests/examples
without claiming the dependent integration is complete.

Update the target plugin README's supported-model entry with the evidenced
status and a link to a real, runnable example submitted or verified in the
repository. Include required arguments, dependencies, hardware, and supported
limits without secrets or personal paths. Keep transient logs, benchmark data,
and experiment reports in the PR/CI artifacts; user documentation contains
maintained usage and limitations.

Deliver reviewable changes and a PR description containing:

- Model/checkpoint and source revisions; framework/operator commits, dependency
  pins, hardware/runtime, dtype, parallel and graph modes.
- Architecture/weight mapping changes, operator gap decisions and ownership,
  public APIs and backend-hit evidence, linked cross-repository dependencies.
- Exact checks and reproducible commands, independent reference and thresholds,
  numerical/quality/performance results, failures, and unverified scope.
- A support matrix distinguishing implemented, loadable, numerically verified,
  and end-to-end verified features; required gates still open and follow-ups.

Use these evidence levels per device/format/feature: **implemented** means code
and applicable static checks exist; **loadable** requires actual checkpoint
loading on the target; **numerically verified** requires independent numerical
comparison; **end-to-end verified** requires real model execution and applicable
quality/workload gates. Lower levels do not imply the higher ones.

Required CI jobs must actually run; an all-skipped hardware matrix is not a
pass. Re-run affected checks on the final code after resolving conflicts.
Remove debug artifacts and unrelated changes. Publish PRs within the user's
requested scope and describe incomplete validation honestly.

## Related skills

- `infer-env-setup`: prepare a missing inference environment.
- `infer-model-adapt`: existing vLLM upstream-copy porting workflow; this skill's
  contract and evidence requirements govern the broader Day 0 delivery.
- `infer-vllm-hw-adapt`, `infer-sglang-hw-adapt`: a genuinely new device backend.
- `infer-vllm-plugin-upgrade`, `infer-sglang-plugin-upgrade`: framework upgrades.
- `infer-precision-check`, `debug-strategy`: deeper numerical diagnosis; retain
  independent references and repository-defined acceptance criteria.
