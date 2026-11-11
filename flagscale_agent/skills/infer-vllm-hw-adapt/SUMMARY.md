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

# Infer-vLLM-HW-Adapt — Summary

Adapt one non-NVIDIA vllm-plugin-FL backend after the shared vLLM upgrade.

**Load when**: a target vLLM version is already established and a vendor backend
needs platform, worker, attention, operator, graph, packaging, or CI adaptation.
Do not use for the framework-wide version bump or an SGLang backend.

**Workflow** — record the images and software stack, adapt the backend, run the
complete tools gate and cross-node FlagCX tests, then pass basic CI on the
submitted commit and record the evidence in the PR.

**Key constraints** — keep vendor shims scoped and verify the active operator,
compiler, and communicator paths. Acceptance requires all applicable
`tools/adaptation-gate-cases` scenarios with FlagGems, FlagGems-vllm, FlagTree,
and FlagCX active together. Automatically detect each node's FlagCX build and
binding, select compatible settings, and prove cross-host communication on at
least two nodes. Require a green basic CI lane and retain the separate hardware
gate artifacts.

Use four English PR tables for images and software versions, changes and reasons,
test results, and additional FlagGems/FlagTree dependencies.
