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

# Framework contracts for day 0 adaptation

Condensed from **推理框架插件开发规范v1.md**, clauses A1–G2. Clause IDs below
refer to that source. MUST preserves a hard requirement; SHOULD preserves a
recommended practice whose exception needs a reason and validation in the change
description. Apply project-specific requirements only to the named project.
Use the target branch's current structure and interfaces; paths are responsibility
examples, not instructions to create parallel infrastructure.

Read A, B, F and G for each model integration. Read C for patches, D for
configuration/dependency changes, and E for state, cache, streams or graphs.
For quantization, also read [quantization.md](quantization.md).

## A. Ownership and reuse — always

- **A1 MUST:** Put model definitions, configuration, weight loading, processors,
  multimodal wiring and runtime orchestration in `vllm-plugin-FL` or
  `sglang-plugin-FL`. Put framework-independent computation in `FlagGems`, and
  computation tied to a framework's layout/state contract in `FlagGems-vllm` or
  `FlagGems-sglang`. New normalization, activation, RoPE, attention, MoE,
  quantization, FP8 conversion and numerical KV access belong in operator
  libraries. Do not copy kernels into the plugin or load private operator files.
- **A2 MUST:** Keep model-specific work in the existing model entrypoint
  (`vllm_fl/models/<model>/` for the usual vLLM layout). Keep device detection,
  defaults and attention/graph selection in the platform layer. Keep public op
  contracts, registration, policy, bridges and configuration in the existing
  `dispatch/` mechanism. Vendor adapters belong in
  `dispatch/backends/vendor/<vendor>/`; references belong in
  `dispatch/backends/reference/`. Use SGLang's actual equivalent entrypoints.
- **A3 MUST:** Classify mixed modules by each symbol's reads and writes. Request,
  token, slot and block mappings, valid-index/copy plans and long-lived resources
  belong to the framework, even when metadata is prepared on a GPU/Triton.
  Numerical KV reads/writes and format conversion belong to operators. Scoring,
  sorting and Top-k on model scores are numerical computation even if they return
  indices. Split mixed functions into metadata preparation and numerical consumers.
  Do not migrate unrelated legacy kernels; preserve behavior and migrate tests
  when migration is in scope.
- **A4 MUST:** Inspect current mainline and dependencies before implementing.
  Compare parameters, defaults, dtype, stride, masks, mutations and error semantics;
  names alone do not prove compatibility. Add a thin public operator API for an
  existing compatible kernel, or extend/add a semantically explicit variant.

## B. Dispatch and capability — always; B5 only for a new vendor

- **B1 MUST:** Route new/replacement compute backends through existing op contracts,
  registration, backend adapters, call bridges and configuration. Preserve normal
  upstream layer/API reuse. Do not create model/vendor selection chains or call
  the generic operator library directly from `forward` to bypass dispatch policy.
- **B2 MUST:** Register generic implementations in the generic backend and
  independent references in the reference backend. Vendor backends need genuine
  vendor implementations, may call vendor-native APIs, and must use names matching
  detection, directories and configuration. A copied reference is not vendor support.
- **B3 MUST:** Use existing platform detection and select capabilities/backends
  before execution. Check actual APIs/signatures, device properties or a probe
  without state effects; importability/version alone is insufficient. Load optional
  vendor dependencies only after establishing the platform and selected path.
- **B4 MUST:** Before execution, an existing policy may choose another implementation
  with the same contract or fail explicitly. Once computation, mutation, state
  updates, asynchronous launches or collectives start, propagate errors; never
  catch an execution failure and retry another backend.
- **B5 MUST, new vendor:** Check device identity, registration, per-op policy,
  defaults, communication, capabilities and graph modes. Exercise actual
  collectives when supporting multiple devices. Cover installation dependencies,
  platform tests, CI routing and container configuration, explaining inapplicable
  items. Registering a device name alone does not complete hardware support.

## C. Interfaces and patches — when replacing or patching upstream behavior

- **C1 MUST:** Match parameter/default and return structure, dtype/layout, masks,
  scales, rounding, mutations, state changes, exceptions and support boundaries.
  Reject unsupported parameter combinations at configuration or call entry;
  never accept and silently ignore them.
- **C2 MUST:** Scope patches to the intended model/platform/conditions, install
  centrally through its patch entrypoint, and make repeated initialization
  idempotent. Importing a generic module must not install vendor patches globally.
- **C3 MUST:** For dependent patch groups, validate every required API/signature,
  construct replacements, then install atomically. On failure restore all original
  objects and wrappers; do not retain a partial install or cache failure as success.
- **C4 SHOULD:** Use capability-based compatibility conditions, explicit arguments
  and real upstream state rather than duplicate state. If probing is necessary,
  avoid changing model state and define probe-cache initialization, reuse and
  invalidation.

## D. Configuration and dependencies — when changed

- **D1 MUST:** Before adding a switch, establish why existing configuration,
  shapes or capability detection cannot express it; who sets it and for which
  models/platforms; its default; the end-to-end propagation and precedence; and
  documentation plus enabled/disabled/unset tests. Do not duplicate inferable
  internal policy. Copy shared configuration before modifying it.
- **D2 MUST:** For changed/removed switches, inspect actual readers, defaults and
  environment injection in code, tests, documentation, builds, CI and deployment.
  Remove obsolete call chains; renaming a switch or hardcoding false leaves dead
  code. Preserve normal inference and multimodal behavior.
- **D3 MUST:** Deliver operator public APIs and tests first, integrate the framework
  second, then lock dependencies to a release or team-accessible public commit SHA.
  Keep installation documentation and build configuration consistent. A private
  personal branch or moving branch is not a deliverable dependency.
- **D4 MUST:** Keep platform defaults in platform configuration/hooks with one source
  of truth, not global shared model settings. For lists and environment overrides,
  define source/default, replacement versus merge semantics, final value and
  precedence, and test overrides. Diagnostic switches must not write files by default.
- **D5 SHOULD:** Give temporary blacklists platform/trigger/failure details, a
  tracking link and removal conditions. Remove or narrow them after the fix;
  do not hide defects by widening permanent global exclusions.

## E. State, memory and graph execution — when affected

- **E1 MUST:** Declare mutation, aliasing and shared storage accurately in APIs and
  custom-op schemas. Match cache write timing, padding, invalid slots and returns.
  Preserve MoE router weights, residual/shared-expert operations, intermediate
  dtypes and collective order/count.
- **E2 MUST:** The framework owns long-lived cache, cross-call state and graph
  workspace allocation, memory budgeting, reuse, release and concurrent isolation.
  Prepare capture resources before capture and keep addresses valid/stable for
  replay. Wrappers may allocate ordinary outputs and call-local temporaries;
  fixed-address, reused or cross-stream storage needs caller-owned synchronization.
- **E3 MUST:** Establish explicit dependencies on the actual current/capture stream
  before consumers use asynchronous compute/communication outputs; reuse existing
  event/scheduling mechanisms. Initialization or launch logs cannot claim completed
  computation or successful numerical validation.
- **E4 SHOULD:** Cache costly streams/communicators at their device/process ownership
  scope and release them at lifecycle end. If later stateful initialization fails,
  rollback/cleanup of earlier components and tests of that failure path are required.

## F. Verification and delivery — scale to the change

- **F1 MUST:** Test the real business entrypoint and assert the backend selected and
  exact arguments forwarded. Cover non-matching dispatch, missing capabilities,
  execution failure without retry and unaffected platforms. Mocks prove wiring,
  not numerical correctness.
- **F2 MUST:** Compare against an independent reference across affected shapes,
  dtypes, strides, masks, edge values and state updates. Expected-value computation
  must not call the tested implementation or share potentially faulty intermediate
  calculations. Fix invalid tests when justified; do not weaken tolerances,
  assertions or supported-case coverage to conceal defects.
- **F3 MUST, graphs:** Perform real capture/replay, change inputs before replay and
  initialize independent output buffers with sentinels. For mutation, aliasing or
  state, reset inputs/state so reference and replay start identically, then compare
  final values/state. Cover affected batches, concurrency, distributed operation
  and cache access. Empty graphs and stale inputs/outputs are not valid evidence.
- **F4 MUST:** Run applicable format, unit, build and target-hardware checks; verify
  generated packages and dependency entrypoints when packaging changes. Record
  tested commits, environment, actual commands/results and unverified scope in the
  PR. CPU/static/stub checks and historical runs are not current hardware validation.
- **F5 SHOULD:** Keep the PR focused; remove dead code, duplicate registration,
  probes, one-off logs, obsolete switches and unrelated formatting. Explain shared
  infrastructure changes and regress affected callers.
- **F6 MUST, hardware/CI changes:** Validate discovery, task routing, checkout and
  prerequisite results. Configuration errors/required failures must fail the gate;
  accidentally skipping every required hardware job cannot pass. End-to-end tests
  must check generated content as well as exit status. Re-run affected checks on
  the latest code after resolving merge conflicts.
- **F7 MUST:** Use English source comments, logs and errors, the project's logger
  rather than `print`, and required copyright/license/source attribution headers.
  Run configured formatters, linters and pre-commit hooks without bypassing gates.

## G. User-facing support — for a completed new model or vendor

- **G1 MUST:** Update the existing README support table/list with the actual model
  or vendor status and a link to a committed runnable example; create an equivalent
  navigation entry if none exists. Use new or verified existing example scripts
  with dependencies, required arguments, launch commands and applicable hardware.
  No placeholders, broken links, credentials, personal paths or private configuration.
  Validate links and entrypoints, and claim only the support actually demonstrated.
- **G2 MUST:** Keep README/docs focused on maintained user information. Beyond new
  models/vendors, update them for material changes in functionality, installation,
  configuration, interfaces or compatibility, using existing entrypoints. Store
  one-off benchmarks, test logs and intermediate conclusions in the PR/CI artifacts,
  not README or process reports under `docs/`.
