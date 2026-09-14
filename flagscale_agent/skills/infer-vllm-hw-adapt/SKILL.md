---
description: Adapt vllm-plugin-FL to a non-NVIDIA hardware backend after the
  shared vLLM upgrade is established. Covers empty-vLLM environments, vendor
  platform and operator fixes, real-device validation, and backend CI. Do not use
  for the framework-wide vLLM version bump or an SGLang backend.
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
- base plugin branch/commit and target vLLM tag/commit;
- Python, PyTorch/vendor fork, accelerator toolkit, communication library,
  compiler, FlagTree/Triton, FlagGems, and plugin revisions;
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

Model loading, worker initialization, or HTTP readiness alone is not a pass.
Require completed generation and a simple deterministic assertion. Record exact
model, TP, dtype/quantization, graph mode, dependency tuple, commit, exit code,
and log path.

If a device or container problem blocks runtime validation, retain successful
import/unit evidence but mark offline and serving gates blocked. Do not translate
"reached device initialization" into inference success.

## CI and delivery

When backend CI is in scope:

- keep the platform registry, runner label, image, device selector, model mounts,
  and setup script consistent;
- assert dependency versions and checkout import paths before tests;
- use named long-running sessions for manual validation;
- distinguish image-pull/setup time from test time and preserve completed jobs;
- keep scarce or manual hardware out of automatic PR CI unless repository policy
  enables it;
- verify an image is pullable and authorized before referencing it in CI;
- do not include credentials, internal host configuration, temporary scripts, or
  standalone diagnostics in the PR.

Run final tests on the exact submitted commit after integrating the latest base.
The PR must separate current-HEAD passes, older supporting evidence, untested
paths, infrastructure blockers, and intentional follow-ups. Runtime-compatible
import changes on another vendor are not evidence that vendor was validated.

## Related skills

- `infer-vllm-plugin-upgrade` establishes the shared framework baseline.
- `infer-env-setup` provisions vLLM/plugin environments.
- `infer-model-adapt` handles model-only additions.
- `infer-precision-check` performs deeper output comparison.
- `debug-strategy` and `ops-discipline` support systematic, safe diagnosis.
