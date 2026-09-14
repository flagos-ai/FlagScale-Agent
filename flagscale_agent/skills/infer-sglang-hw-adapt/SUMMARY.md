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

# Infer-SGLang-HW-Adapt — Summary

Adapt one non-NVIDIA sglang-plugin-FL backend after the shared SGLang upgrade.

**Load when**: a target SGLang version is established and a vendor backend needs
`srt_empty`, platform, fused-op, attention/FLA, scheduling, graph, distributed,
packaging, or CI adaptation. Do not use for the framework-wide bump or vLLM.

**Workflow**: lock the vendor dependency tuple -> verify empty/vendor runtime and
source provenance -> adapt platform and operators -> run focused/full tests ->
validate dense/MoE, multimodal, MTP, graph, serving/concurrency, and TP/PP as
applicable -> integrate resource-aware CI -> report exact-head evidence.

**Key constraints**: preserve original examples and assertions, keep fallbacks
explicit, distinguish decode from prefill graph claims, separate image-pull
failures from test results, and never use historical or cached-image evidence as
a substitute for current-head real-device validation.
