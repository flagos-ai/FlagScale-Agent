---
description: Adapt sglang-plugin-FL to a non-NVIDIA hardware backend after the
  shared SGLang upgrade is established. Covers srt_empty/vendor environments,
  operator and scheduling fixes, multimodal and distributed validation, and
  backend CI. Do not use for the framework-wide SGLang bump or vLLM.
name: infer-sglang-hw-adapt
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

# Adapt sglang-plugin-FL to a Hardware Backend

Use this skill after `infer-sglang-plugin-upgrade` establishes the shared SGLang
API baseline. Adapt and validate one vendor backend at a time. Keep vendor-only
runtime choices out of the framework-wide upgrade unless shared code requires
them.

## Establish the contract

Record before editing:

- backend, accelerator model, allocated devices, driver/toolkit, and container;
- plugin base commit and exact SGLang tag/commit;
- Python, vendor PyTorch, SGLang kernel packages, attention/MoE libraries,
  FlagTree/Triton, FlagGems, FlagCX, flashinfer, and communication-library
  revisions;
- current checkout/import paths, model and image assets, logs, and CI scope;
- required coverage: dense, MoE, hybrid attention, multimodal, MTP, graph,
  serving, concurrency, tensor/pipeline parallel, or benchmark smoke.

Use a vendor branch based on the completed shared SGLang upgrade. Preserve
unrelated work and keep other backend pins unchanged unless they are explicitly
in scope.

Prefer the target release's supported non-CUDA installation mode. When SGLang
provides `srt_empty`, use it instead of carrying old empty-install patches. Prove
that every deleted patch is now upstream-owned. Install the plugin without
dependencies, and verify `sglang_fl`, SGLang, FlagGems, and Triton/FlagTree import
from the intended locations; vendor images often contain older packages that can
shadow an editable checkout.

## Invariants

- Never modify installed SGLang source.
- Treat SGLang, vendor PyTorch, kernel/attention/MoE packages, compiler,
  FlagGems, FlagCX, flashinfer, communication library, and plugin commit as one
  tuple.
- Gate vendor behavior by platform or inspected capability. Keep registration
  idempotent and avoid broad exception-based compatibility.
- Preserve original examples, prompts, baselines, and assertions. Add diagnostics
  after the unchanged test body rather than weakening it.
- Prefer explicit per-op vendor/reference fallback over globally disabling
  FlagGems. An override must not erase required platform defaults accidentally.
- Start eager, then enable only graph modes proven on the backend. Decode and
  prefill graphs are separate claims.
- Squash or rewrite history only when policy or the user requests it.

## Analyze and adapt the backend

Compare the vendor's last known-good plugin/SGLang pair with the target shared
upgrade. Audit:

| Surface | Typical risks |
| --- | --- |
| Platform/plugin lifecycle | discovery order, device APIs, worker selection, late subclasses |
| Fused ops | OOT registry changes, schema/signature drift, vendor extension load order |
| Attention/FLA | backend selection, public patch points, recursion, hybrid/MTP paths |
| Scheduling/cache | overlap scheduling, radix cache, page size, KV/disaggregation metadata |
| MoE/sampling | top-k, routing, vendor kernels, reference fallback, numerical drift |
| Graphs | decode/prefill separation, batch sizes, warmup, vendor compiler limits |
| Distributed | TP/PP topology, host networking, interface selection, communication library |
| Packaging/CI | `srt_empty`, native wheels, platform registry, images, timeouts, model mounts |

Implement from platform and registration outward. Capture original public
functions before monkey-patching so a wrapper cannot recurse into itself. Keep
vendor fallbacks explicit and observable. When asynchronous or overlap scheduling
is unsafe, a capability-gated synchronous path is acceptable, but document the
lost behavior and test that decode graphs or other retained features still work.

Do not generalize a numerical workaround from one symptom. Run native-op,
reference, precision, and scheduling controls before changing a platform default.
If a strict output comparison still warns while all controls diverge differently,
report the limitation rather than inventing a causal claim.

After each compatibility theme, run imports, focused tests, and source-path
checks before proceeding.

## Validate on real hardware

Use the complete **FlagGems + FlagTree + FlagCX** stack simultaneously for final
hardware acceptance. Record exact revisions and resolved paths; prove FlagTree
owns the active Triton compiler/runtime, FlagGems dispatches the intended ops,
and FlagCX executes a real collective or TP/PP path without silently falling
back to NCCL or a vendor communication backend. Imports, isolated component
tests, or partial-stack results are insufficient. If any component cannot be
enabled and exercised, report acceptance blocked unless the user explicitly
changes the contract.

Choose coverage from the affected paths. A broad backend adaptation normally
includes:

- plugin discovery, platform, fused-op registration, and dependency imports;
- targeted compatibility tests plus full unit and functional suites;
- original offline and serving examples without removed assertions;
- dense/hybrid and MoE models;
- text and multimodal requests when the backend claims both;
- deterministic non-streaming, streaming, long decode, and concurrent requests;
- eager and supported decode/prefill graph modes;
- MTP against its unchanged baseline when MTP/scheduling changed;
- deployment-scale TP and cross-node TP/PP when distributed code changed;
- cleanup checks for server, worker, and communication processes;
- benchmark entrypoint smoke, clearly separated from controlled performance.

Hardware acceptance additionally requires all runnable examples declared
compatible with the backend and every enabled concurrent E2E case. Inventory
documented scripts under `examples/` and run each unchanged; helper modules are
not cases.
Offline, concurrent, MTP, and multinode examples are mandatory when the backend
claims those capabilities. Missing models, image assets, hosts, or devices block
acceptance rather than justifying a pass or silent skip. Every example and
concurrent-test process must inherit the same verified full-stack
environment; do not mix results from partial-stack environments.

Run the target platform/device's complete concurrent matrix, including every
configured model/case and advertised text, VL, and mixed mode, through:

```bash
python tests/run.py --platform <platform> --device <device> --scope e2e --task concurrent
```

The final report must list discovered, executed, passed, failed, skipped, and
blocked examples/concurrent cases by name, plus model, TP/PP, graph mode,
concurrency, request totals, failures, and available latency/throughput fields.
Every applicable example and concurrent case must pass without request errors,
timeouts, OOMs, hangs, empty/corrupt responses, or unhealthy workers.

Inspect logs in addition to exit codes: assertion failures, repeated/corrupt
output, cleanup/resource-tracker tracebacks, or zero served requests are not
passes. Require the configured request count and every requested text/VL/mixed
phase to execute; an early phase failure leaves later phases blocked. Any custom
pressure wrapper must propagate child exit codes and verify configured
input/output token lengths rather than reporting only process liveness.

For multimodal tests, verify the referenced assets exist before starting a long
run. For multi-node tests, use matching code, dependencies, model paths, network
interfaces, and reachable ports on every host; require success from every role.

An exit code alone is not sufficient when a suite has warning thresholds. Report
pass/fail/warning counts separately. A server reaching health readiness, a model
loading, or a graph compiling is not generation success. Retain complete request
and response evidence with commit, model, TP/PP, modes, and dependency tuple.

## CI and evidence discipline

Integrate the backend through the repository's platform registry and reusable
test chain rather than creating an unrelated workflow. Preserve the normal order:

```text
discover/setup -> unit -> functional -> inference/serving/concurrent
               -> benchmark smoke -> notification
```

CI updates are a mandatory deliverable. Update `examples/**` path triggers,
platform YAML concurrent matrices, runner/image/model mounts, full-stack setup
assertions, reusable E2E inputs, result artifacts, and aggregate status. Required
example and concurrent jobs must not pass through skips or missing assets. If
scarce hardware requires a manual gate, encode it explicitly and retain a
required current-head result in the PR evidence.

Adapt resource policy to the hardware:

- select idle devices only within the runner allocation;
- serialize large E2E groups when concurrent jobs would contend for the same
  accelerator;
- set timeouts that account for model/image initialization without changing
  unaffected platforms;
- keep manual or disabled platform entries structurally intact when CI/CD owners
  need to resolve infrastructure separately.

Separate container initialization from test execution. A timeout while pulling a
large image is infrastructure evidence, not a failed or passed plugin test. Check
DNS/TLS/authentication from the error instead of assuming credentials are needed.
A cached-image pass does not prove cold-pull reliability.

Retry only after the failed run has finished and the blocking condition changed.
Preserve completed jobs when the CI system supports targeted reruns. Repeated
force-pushes with an unchanged tree do not repair a runner network/cache problem
and invalidate current-HEAD evidence.

After rebasing or changing CI, rerun the affected merge gates on the exact new
head. Keep historical passes as supporting evidence only. The PR must list:

- exact source/base/head commits and dependency tuple, including FlagGems,
  FlagTree, and FlagCX provenance and activation evidence;
- backend hardware and device allocation;
- code paths, fallbacks, and scheduling/graph limitations;
- current-HEAD unit, functional, example, serving, distributed, and CI results;
- historical evidence clearly labeled;
- untested features and infrastructure blockers.

Do not commit credentials, internal host details, temporary transfer scripts,
raw model outputs, or local validation records unless explicitly requested.

## Related skills

- `infer-sglang-plugin-upgrade` establishes the shared SGLang baseline.
- `infer-env-setup` supplies general remote/container setup discipline; do not
  reuse vLLM-specific install commands.
- `infer-model-adapt` handles model-only work.
- `infer-precision-check` supports detailed output comparison.
- `debug-strategy` and `ops-discipline` support safe diagnosis.
