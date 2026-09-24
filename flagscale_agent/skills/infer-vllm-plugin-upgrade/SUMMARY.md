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

# Infer-vLLM-Plugin-Upgrade — Summary

Upgrade vllm-plugin-FL to a newer vLLM release by reconciling upstream code and
reapplying FL-specific behavior across platform, compilation, worker, model
runner, ops, dispatch, registration, packaging, and CI surfaces.

**Load when**: planning or implementing a vLLM version bump, repairing the plugin
after a vLLM upgrade, or validating a completed upgrade. Use `infer-model-adapt`
when the task is only to add a model.

**Workflow**: establish exact versions and dependency tuple -> read upstream and
earlier upgrade history -> baseline and plan -> reconcile components in dependency
order -> remove verified-obsolete compatibility code -> validate a change-driven
runtime matrix -> rebase, rerun merge gates, and report auditable evidence.

**Key principles**:

- Treat vLLM, PyTorch, accelerator runtime, FlagTree/Triton, FlagGems, FlagCX,
  flashinfer, and plugin revision as one tested compatibility tuple.
- Preserve FL dispatch, vendor abstraction, graph capture, custom ops, I/O dumping,
  and explicit backend selection while inheriting new upstream behavior.
- Use the official NVIDIA runtime as a reference where useful; use
  `VLLM_TARGET_DEVICE=empty` for vendor backends that cannot build CUDA vLLM, and
  validate each claimed backend separately.
- Test runner selection, native and FL-dispatched paths, dense/MoE,
  eager/compiled/graph execution, offline generation, and serving as applicable to
  the changed code.
- Delete model/config/patch code only after proving the target upstream owns and
  registers it, then verify no dangling references remain.
- Separate current-HEAD evidence from pre-rebase supporting evidence and list
  untested hardware honestly.
- Final acceptance requires every case under `tools/adaptation-gate-cases`: both
  Qwen3.6 models, eager and graph, and all text/image/mixed single and concurrent
  scenarios must pass with FlagGems, FlagTree, and FlagCX enabled together.
- CI must be updated for the adapted platform, full-stack setup, required gate
  coverage, result artifacts, and non-skippable aggregate status.

**Constraints**: never modify upstream vLLM, never hardcode CUDA device calls in
shared paths, preserve unrelated work, install the plugin without changing the
vLLM dependency, and do not claim completion without real inference on the named
hardware.
