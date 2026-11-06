---
description: Adapt vllm-plugin-FL to a non-NVIDIA hardware backend after the
  shared vLLM upgrade is established. Covers empty-vLLM environments, vendor
  platform and operator fixes, tools-based acceptance, multi-node FlagCX,
  and backend CI. Do not use for the framework-wide vLLM version bump or an
  SGLang backend.
name: infer-vllm-hw-adapt
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

# Adapt vllm-plugin-FL to a Hardware Backend

Use this skill after `infer-vllm-plugin-upgrade` has established the shared vLLM
API baseline. Adapt one named vendor backend at a time and make only the runtime
claims proven on that backend.

## Establish the contract

Before editing, record:

- backend, accelerator model, host/container, allocated devices, and driver;
- base image and CI image tags or digests, and the CI runner allocation;
- base plugin branch/commit and target vLLM tag/commit;
- Python, PyTorch/vendor fork, accelerator toolkit, compiler, FlagTree/Triton,
  FlagGems, FlagGems-vllm, FlagCX, communication-library, and plugin revisions;
- upstream vLLM and plugin roots, model paths, tests, logs, and CI scope;
- validation modes requested: imports, unit/functional, offline, serving,
  tensor parallel, graph, multimodal, MoE, or benchmark.

Inspect repository status and device visibility before changing code. Preserve
unrelated work and use a vendor-specific branch based on the completed shared
upgrade. Do not fold unverified vendor changes back into the shared upgrade.

For a backend that cannot build vLLM's CUDA extension, install the exact target
vLLM source with `VLLM_TARGET_DEVICE=empty`. Install vllm-plugin-FL with
`--no-build-isolation --no-deps`; do not let it replace the validated vLLM or
vendor packages. Verify imports resolve to the current checkout rather than an
older package in the base image.

## Invariants

- Never modify upstream or installed vLLM source.
- Keep shared device operations behind `current_platform.torch_device_fn` or the
  relevant vLLM/FL platform abstraction; do not hardcode `torch.cuda`.
- Gate vendor workarounds by a concrete platform or inspected capability. Make
  patch registration idempotent.
- Prefer a vendor implementation, then a verified FlagGems implementation, then
  a bounded reference fallback. Never hide unsupported behavior in a broad
  exception handler.
- A workaround needs a removal condition only when that condition is known; do
  not add speculative TODOs.
- Do not claim runtime success when the container cannot see the device. Broader
  device mounts or privileged containers require user authorization.
- Squash or rewrite history only when repository policy or the user requests it.

## Analyze the backend delta

Compare the shared-upgrade branch with the vendor backend and its last known-good
version. Audit:

| Surface | Questions |
| --- | --- |
| Platform | Is detection correct? Are device, stream, memory, capability, and communication APIs complete? |
| Worker | Do initialization, memory snapshots, cache sizing, distributed state, and teardown use vendor-compatible APIs? |
| Attention | Does backend selection use the target vLLM registry/enum contract? Are MLA, sparse, and multimodal paths separated? |
| Ops | Which activation, norm, rotary, sampling, attention, and fused-MoE operations are vendor, FlagGems, or reference? |
| Compilation | Are custom-op registration, graph capture, memory pools, and compiler imports valid for this backend? |
| Models | Which dense, MoE, hybrid, multimodal, or quantized paths touch vendor code? |
| Packaging/CI | Are empty-vLLM, native extensions, images, runner labels, model mounts, and setup assertions correct? |

Inspect the target vLLM definition and call sites before patching. Recurring
backend failures include changed attention registration, missing CUDA-shaped
memory APIs, new KV block-size contracts, class/factory transitions, duplicate
custom-op registration, moved FLA kernels, and graph wrappers that assume CUDA.

## Implement in dependency order

Work from foundations outward:

```text
platform/device shims -> worker/cache -> attention -> ops/dispatch
                      -> compilation/graph -> models -> packaging/CI
```

Keep shims narrow:

- synthesize a CUDA-shaped API such as `memory_stats()` only when vLLM consumes
  that contract and the vendor exposes equivalent underlying values;
- declare vendor block sizes and capabilities from real kernel constraints;
- register attention with the target vLLM public mechanism, using custom slots
  only when required by that version;
- preserve explicit backend selection and make fallback order observable;
- preload vendor extensions only when needed to avoid schema collisions;
- use eager mode to isolate correctness before enabling graph capture.

After each component, run its import and focused regression tests. Keep negative
controls when they distinguish a plugin defect from a compiler, driver, model, or
upstream vLLM failure.

## Validate on the named hardware

Run acceptance with **FlagGems, FlagGems-vllm, FlagTree, and FlagCX** active
together. Pin all four from the target checkout's dependency and build settings.
Install the pinned FlagGems-vllm dependency explicitly when using `--no-deps`.
Record resolved paths and versions in server and worker processes. Verify the
`flaggems_vllm` import and trace an applicable plugin op through dispatch to
that package, alongside evidence for generic FlagGems and the active FlagTree
compiler/runtime. Read `docs/flaggems-vllm.md` when present and enable the fused
path using the target checkout's switch, such as `VLLM_FL_USE_FLAGGEMS_VLLM=1`.
Set the stack and switches in the serving environment before workers start.
A focused plugin test can exercise a required operator absent from the model
matrix. Use the same pinned stack for that test and the gate runs.
Mark acceptance blocked if a required component cannot be exercised.

Select a matrix from the affected paths. A substantial adaptation normally needs:

- vLLM, plugin, platform, worker, model-runner, and vendor-extension imports;
- focused backend tests and the full unit/functional suites available there;
- deterministic vanilla-vs-plugin comparison where vanilla can run;
- a small dense smoke, plus MoE, hybrid, multimodal, or quantized models when
  those paths changed;
- eager first, then the supported compiled/graph modes;
- tensor parallel at deployment scale when communication or sharding changed;
- offline generation and serving with a completed, non-empty response;
- explicit verification of dispatch choices and bounded fallbacks;
- teardown checks so successful runs do not leave task-owned processes.

### Run the adaptation gate in `tools/`

Use `tools/adaptation-gate-cases/README.md` and the scripts and tests in the
submitted vllm-plugin-FL checkout to determine the required model/mode groups,
fixtures, scenarios, and request totals. The documented matrix covers
Qwen3.6-27B and Qwen3.6-35B-A3B in both eager and graph modes. Run single text
and image cases and every concurrent text, image, and mixed text-image case
in each group. Count pytest scenarios separately from requests within them.

From `tools/adaptation-gate-cases`, start the corresponding server in a named
session and run the test entrypoint against it in another session. For example,
use these commands for one eager group, substituting the actual model path
and an available port. Set the same `SERVED_MODEL_NAME` on both sides if used.

```bash
MODEL_PATH=/path/to/model PORT=8000 ./run_serve_eager.sh
MODEL_PATH=/path/to/model PORT=8000 RESULTS_DIR=/path/to/run-results ./run_test.sh
```

Repeat with `run_serve_graph.sh` for graph mode. Verify the serving PID, model,
and mode before each test run and confirm graph capture and replay in the graph
run. Stop the previous task-owned service and confirm the port is free before
reusing it. Use distinct ports and `SERVER_PID_FILE` paths for concurrent
groups. Use fresh result directories to preserve every group's JSON files.
Keep the committed text, image, and mixed tests and response assertions intact.
Confirm all test files ran and each scenario produced fresh JSON with the
expected number of passing responses.
Bind artifacts to the commit, device, model revision, mode, and stack in a run
record because the JSON schema may omit these fields.

Reconcile discovered, executed, passed, failed, skipped, and blocked counts
for each model/mode. Missing fixtures or weights, skips, filtered files,
unstarted groups, and timeouts block acceptance. Keep the server/test logs and
JSON artifacts. Use focused tests and `tests/run.py` as supporting checks and
require the complete tools gate for acceptance.

### Validate FlagCX across nodes

Detect the FlagCX build automatically on every allocated node before launch.
Inspect the plugin's resolved binding and native library selection. For the
wheel, use the launch environment's `python -m pip show flagcx` or distribution
metadata and verify the selected `flagcx.api` and `libflagcx.so` paths. For a
source build, inspect the selected `FLAGCX_PATH` wrapper and record its release
or source commit and build metadata. Check the workers' mapped library, for
example via `/proc/<pid>/maps` on Linux, against the selected build. Record the
native version separately when the loaded library exposes a supported version API.

Select initialization and launcher settings from the detected build and actual
binding capabilities. Inspect the target plugin's communicator and the matching
FlagCX docs or examples. Check exported symbols, signatures, and unique-ID
handling. Legacy source wrappers and newer wheel APIs can pass communicator IDs
differently. Resolve incompatible or unknown builds before running the gate.

Use at least two distinct hosts with a matching software stack, reachable
rendezvous, and a verified rank-to-device map. Run a cross-node collective with
rank-specific inputs through the plugin communicator and a real distributed
inference request on a supported TP/PP topology. Check expected collective
values, model responses, and successful exit from every rank. Use traces or logs to
prove that the data collective crossed hosts through FlagCX. Match the group
members to their host/rank map and capture call evidence during the request.
A separate control-plane backend for rendezvous and a vendor adaptor called by
FlagCX are allowed. An implicit framework collective fallback fails this gate.
Record topology, per-node versions, paths, and result logs. Mark the multi-node
gate blocked when the allocation or runtime evidence is unavailable.

Model loading, worker initialization, or HTTP readiness alone is not a pass.
Require completed generation and a simple deterministic assertion. Record exact
model, TP, dtype/quantization, graph mode, dependency tuple, commit, exit code,
and log path.

If a device or container problem blocks runtime validation, retain successful
import/unit evidence but mark offline and serving gates blocked. Do not translate
"reached device initialization" into inference success.

## CI and delivery

Deliver a basic CI lane through the target plugin's existing platform registry
and reusable workflow. Start with `.github/workflows/ci.yml`, its platform
workflow, and `.github/configs/` where present. Require a passing run on the
submitted commit and rerun affected gates after rebasing or changing CI.
At minimum, cover build/install, resolved stack assertions, focused unit tests,
and a functional inference smoke test on the target device. Keep the full tools
gate and multi-node FlagCX validation
as separate required results. Use the hardware CI jobs where supported or an
explicit manual gate for scarce resources.

- keep the platform registry, runner label, image, device selector, model mounts,
  and setup script consistent;
- assert dependency versions and checkout import/library paths before tests,
  including the active FlagGems, FlagGems-vllm, FlagTree, and FlagCX builds;
- use named tmux sessions for long-running remote validation;
- distinguish image-pull/setup time from test time and preserve completed jobs;
- keep scarce or manual hardware out of automatic PR CI unless repository policy
  enables it;
- verify an image is pullable and authorized before referencing it in CI;
- update path triggers (including `tools/adaptation-gate-cases/**`), platform
  YAML, matrices, reusable jobs, and aggregate status so the backend's applicable
  adaptation-gate coverage runs and cannot be silently skipped;
- upload per-case logs/results and fail required jobs on missing cases/assets;
- verify the CI run and required job conclusions for the final PR commit.
  A failed, cancelled, skipped, missing, or zero-case required job leaves the
  CI gate incomplete;
- do not include credentials, internal host configuration, temporary scripts, or
  standalone diagnostics in the PR.

### Write the PR description

Use four concise English tables. Replace each example row with the actual
environment, changes, results, and dependencies. Link CI runs and artifacts in
the test table and label historical evidence with its tested commit.

**A. Images and software stack**

| Item | Tag, digest, or version |
| --- | --- |
| Base image | Image tag and digest |
| CI image | Image tag and digest |
| Software stack | Framework and plugin commits, Python, PyTorch, driver/toolkit, FlagGems, FlagGems-vllm, FlagTree, and FlagCX |

**B. Changes and reasons**

| Change | Reason |
| --- | --- |
| Component and relevant code path | Compatibility issue and reason for the fix |

**C. Tests and results**

| Test | Device, topology, and scope | Result and evidence |
| --- | --- | --- |
| Tools gate | Model/mode, scenario and request counts | Status, tested commit, artifacts, and any unverified cases |
| Multi-node FlagCX | Hosts/ranks, TP/PP, resolved version and API | Status, collective values and inference evidence |
| Basic CI | Runner, required jobs, and tested commit | Status and run link |

**D. Additional FlagGems and FlagTree dependencies**

| Component | Additional requirement | Issue, PR, or exact extra change |
| --- | --- | --- |
| FlagGems | Required revision or change, or None | Link or change needed |
| FlagGems-vllm | Required revision or change, or None | Link or change needed |
| FlagTree | Required revision or change, or None | Link or change needed |

## Related skills

- `infer-vllm-plugin-upgrade` establishes the shared framework baseline.
- `infer-env-setup` provisions vLLM/plugin environments.
- `infer-model-adapt` handles model-only additions.
- `infer-precision-check` performs deeper output comparison.
- `debug-strategy` and `ops-discipline` support systematic, safe diagnosis.
