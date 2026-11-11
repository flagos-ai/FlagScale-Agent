---
description: Adapt sglang-plugin-FL to a non-NVIDIA hardware backend after the
  shared SGLang upgrade is established. Covers srt_empty/vendor environments,
  operator and scheduling fixes, FlagCX build detection, multi-node validation,
  example and concurrent E2E gates, and backend CI. Do not use for the
  framework-wide SGLang bump or vLLM.
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
- base image and CI image, including the tag and digest where available;
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
and FlagCX handles the required cross-node communication. If any component
cannot be enabled and exercised, report hardware acceptance as Blocked.

### Detect the active FlagCX build on each node

Probe automatically before launch in the same container and loader environment
used by each node's SGLang processes. Inspect the target checkout's communicator
and FlagCX wrapper first. For example,
`sglang_fl/distributed/device_communicators/flagcx.py` resolves the wrapper and
`libflagcx.so` using `FLAGCX_PATH`. After launch, confirm the resolved wrapper path
and the actual mapped library path in each rank process using loader diagnostics
or the process's library mappings.

Prefer a documented version API on the loaded library or binding when it reports
the actual FlagCX release. Otherwise, use the wheel, binding, or source build
metadata that corresponds to the mapped binary. Match it by build ID, checksum,
or equivalent provenance. Record any native version that reports only the vendor
adaptor version separately. Log the FlagCX release/build, version source,
resolved paths, and available API capabilities for each node. Stop the affected
run on an unknown, conflicting, or incompatible build and report the reason.

Select the launch arguments and communicator initialization from the detected
version and verified API signatures and capabilities. In the current integration,
`SGLANG_FL_DIST_BACKEND=flagcx` requests FlagCX and `FLAGCX_PATH` supplies its build.
Verify both against the target branch. Confirm the communicator is active and the
exercised data-plane operations reach FlagCX on every participating rank.
`torch.distributed` can provide bootstrap or control-plane communication, and
FlagCX can use its configured vendor CCL adaptor. A FlagCX initialization failure
or implicit framework fallback for tested data-plane operations fails the gate.

### Prove cross-node communication

Use at least two actual nodes with matching code and verified software stacks,
including compatible model assets, reachable rendezvous ports, and the intended
network interfaces. Record node, rank, and device placement, TP/PP group layout,
and the communication peers that cross hosts.

Run a cross-node FlagCX collective with known inputs and validate the expected
result on every participating rank. If the checkout lacks a real cross-node
collective test, use a minimal probe through the same plugin communicator. Also
run real model inference with a TP/PP layout that exercises the required
communication across hosts. The Qwen3.6 `*_multinode.py` examples expose
`--role`, `--master-addr`, `--nnodes`, `--node-rank`, `--tp`, and `--pp` in the
current checkout. Keep their generation and assertion bodies and choose
supported arguments for the detected stack.

Capture per-rank collective results, real model output checks, actual FlagCX
operation and library evidence, and exit status from every rank and launcher
role. Require the expected outputs, successful exits, and clean teardown. Retain
raw child exit statuses and evidence for any existing version-specific teardown
normalization. If hosts, devices, model assets, or network access are unavailable,
mark the affected multi-node case as Blocked.

### Run the examples and concurrent E2E gates

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
- deployment-scale TP and the required cross-node FlagCX validation;
- cleanup checks for server, worker, and communication processes;
- benchmark entrypoint smoke, clearly separated from controlled performance.

Hardware acceptance requires both independent gates to pass.

1. **Examples.** Inventory the documented runnable scripts under `examples/`
   and run every example compatible with the target backend. Preserve the
   original prompts, baselines, and assertions. Offline, concurrent, MTP, and
   multi-node examples are required when their capabilities are claimed. Exclude
   helper modules from the inventory.
2. **Concurrent E2E.** Resolve the target platform/device's complete matrix from
   `tests/platforms/` and the referenced cases in `tests/models/`. Run all
   configured model/case and text, VL, or mixed modes through the existing runner.

   ```bash
   python tests/run.py --platform <platform> --device <device> --scope e2e --task concurrent
   ```

Check the resolved inventory against executed cases and their pytest results.
Avoid `--model`, `--case`, `--cases`, or other filters that omit required cases.
Check any `FL_CONCURRENT_MODES` override against the configured modes.
`tests/run.py` can exit successfully with zero selected cases, and a case can
have skipped tests, so check that every required case and mode actually ran and
passed. Keep the original test assertions and configured coverage.

Both gates must use the same verified full-stack environment. Mark missing
models, image assets, hosts, or devices as Blocked. Record discovered, executed,
passed, failed, skipped, and blocked examples and concurrent cases by name in the
PR's test table, with model, TP/PP, graph mode, concurrency, request totals,
failures, and available latency/throughput. Each required case must pass without
request errors, timeouts, OOMs, hangs, empty/corrupt responses, or unhealthy workers.

Inspect logs in addition to exit codes: assertion failures, repeated/corrupt
output, cleanup/resource-tracker tracebacks, or zero served requests are not
passes. Require the configured request count and every requested text/VL/mixed
phase to execute; an early phase failure leaves later phases blocked. Any custom
pressure wrapper must propagate child exit codes and verify configured
input/output token lengths rather than reporting only process liveness.

For multimodal tests, verify the referenced assets exist before starting a long
run.

An exit code alone is not sufficient when a suite has warning thresholds. Report
pass/fail/warning counts separately. A server reaching health readiness, a model
loading, or a graph compiling is not generation success. Retain complete request
and response evidence with commit, model, TP/PP, modes, and dependency tuple.

## CI and evidence discipline

Integrate the backend through the repository's platform registry and reusable
test chain. Keep the repository's usual job order.

```text
discover/setup -> unit -> functional -> inference/serving/concurrent
               -> benchmark smoke -> notification
```

Deliver basic CI for the target plugin and backend. In the current layout,
reuse `.github/configs/platforms.yml`, the target platform config, and
`.github/workflows/_platform_test.yml`. Configure a matching runner, CI image,
device access, and model mounts. Use the existing build/install path and assert
the resolved software stack, including FlagGems, FlagTree, and FlagCX. Run focused
unit tests and a functional inference smoke test that checks model output.

Run the workflow for the final submitted commit on available resources. Check
that required jobs selected the target backend and completed successfully, and
link their logs and artifacts. Make the aggregate status reflect each required
job's result. Mark missing runner, image, or hardware resources and zero-case
required jobs as Blocked. Ensure the existing triggers cover `examples/**` and
the platform configuration includes required concurrent cases. Keep basic CI
smoke, examples, concurrent E2E, and multi-node FlagCX results separate. If scarce
hardware requires an explicit manual gate, retain its required result on the
final commit in the PR evidence.

Adapt resource policy to the hardware:

- select idle devices only within the runner allocation;
- use named tmux sessions for long-running remote validation;
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
head. Label historical results separately in the test table.

### Write the PR description

Use four concise English tables. Replace each example row with the actual
environment, changes, results, and dependencies. Link CI runs and artifacts in
the test table and label historical evidence with its tested commit.

**A. Images and software stack**

| Item | Tag, digest, or version |
| --- | --- |
| Base image | Image tag and digest |
| CI image | Image tag and digest |
| Software stack | Framework and plugin commits, Python, vendor PyTorch, driver/toolkit, SGLang kernels, FlagGems, FlagTree, and FlagCX |

**B. Changes and reasons**

| Change | Reason |
| --- | --- |
| Component and relevant code path | Compatibility issue and reason for the fix |

**C. Tests and results**

| Test | Device, topology, and scope | Result and evidence |
| --- | --- | --- |
| Examples | Script, model, hardware, TP/PP, graph mode, and request totals | Status, tested commit, artifacts, and any unverified examples |
| Concurrent E2E | Platform/device, model/case/mode, concurrency, and request totals | Status, tested commit, artifacts, and any unverified cases |
| Multi-node FlagCX | Hosts/ranks, TP/PP, resolved version, library paths, and API | Status, per-rank collective values and exits, inference evidence, and unverified scope |
| Basic CI | Runner, required jobs, and tested commit | Status, run link, and any blocked jobs |

**D. Additional FlagGems and FlagTree dependencies**

Use `None` for a component with no additional dependency.

| Component | Additional requirement | Issue, PR, or exact extra change |
| --- | --- | --- |
| FlagGems / FlagGems-sglang | Required revision or change, or None | Link or change needed |
| FlagTree | Required revision or change, or None | Link or change needed |

Do not commit credentials, internal host details, temporary transfer scripts,
raw model outputs, or local validation records unless explicitly requested.

## Related skills

- `infer-sglang-plugin-upgrade` establishes the shared SGLang baseline.
- `infer-env-setup` supplies general remote/container setup discipline; do not
  reuse vLLM-specific install commands.
- `infer-model-adapt` handles model-only work.
- `infer-precision-check` supports detailed output comparison.
- `debug-strategy` and `ops-discipline` support safe diagnosis.
