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

**Workflow**: record the backend dependency tuple -> build exact or empty vLLM
environment -> audit vendor contracts -> patch from platform outward -> run
focused and full tests -> verify real-device inference/serving -> integrate
backend CI -> report current-HEAD evidence and blockers.

**Key constraints**: never patch vLLM source, never hardcode CUDA in shared paths,
gate vendor shims narrowly, preserve observable fallbacks, install without
replacing dependencies, and never claim a hardware pass without completed
generation on that hardware. Full acceptance also requires every case discovered
by the target platform/device's unfiltered `tests/run.py` invocation to pass.
