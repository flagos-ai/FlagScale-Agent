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

# Infer-SGLang-Plugin-Upgrade — Summary

Upgrade sglang-plugin-FL to a newer SGLang release on NVIDIA hardware.

**Load when**: the plugin's SGLang compatibility version is being bumped, or an
upstream SGLang upgrade breaks registration, fused ops, platform APIs, FLA,
CUDA Graph, disaggregation, or serving. Use model or hardware adaptation when
the base SGLang version is not changing.

**Full pipeline**: Stage 0 orientation/baseline → Stage 1 upstream delta → Stage
2 isolated NVIDIA dependency environment → Stage 3 ordered compatibility fixes
→ Stage 4 repository verification → Stage 5 real-model serving and CUDA Graph
validation → Stage 6 final review and PR.

**Key principles**:

- detect installed and declared versions before changing code;
- treat SGLang, PyTorch, sglang-kernel, Triton/FlagTree, FlagGems, FlagCX, and
  flashinfer as one tested tuple;
- upgrade NVIDIA first and preserve other backend pins;
- never patch installed SGLang source;
- install PyTorch before FlagTree and remove standalone Triton when FlagTree
  provides the Triton module;
- pin FlagGems, FlagTree, and FlagCX and prove all three are active together;
- compare plugin inference with vanilla SGLang at the same version;
- validate eager, serving, streaming, long decode, concurrency, and decode CUDA
  Graph on representative dense/hybrid and MoE paths;
- keep docs and containerfiles outside the diff unless explicitly requested.

**Completion gate**: no new unit regressions or temporary diagnostic overrides;
every target-platform example and every enabled concurrent E2E model/case and
text/VL/mixed mode passes on NVIDIA; CI is updated for example/concurrent
coverage, result artifacts, and full-stack setup; the PR reports exact
dependencies, model/TP/graph/concurrency coverage, fallbacks, and untested
follow-ups. A silent NCCL fallback or separately tested components do not
satisfy the full-stack gate.
