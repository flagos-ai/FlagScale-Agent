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

**Workflow** — record the images and software stack, adapt the backend, run the
examples, configured `tests/run.py` concurrent E2E, and cross-node FlagCX gates,
then pass basic CI on the submitted commit and record the results in the PR.

**Key constraints** — preserve original examples and assertions. Require every
compatible example and configured concurrent case/mode with FlagGems, FlagTree,
and FlagCX active together. Detect each node's loaded FlagCX build and binding,
select compatible settings, and prove collective and model TP/PP communication
on at least two nodes. Verify per-rank results and exits. Keep the hardware gates
separate from basic CI and mark unavailable resources as Blocked.

Use four English PR tables for images and software versions, changes and reasons,
test results, and additional FlagGems/FlagTree dependencies.
