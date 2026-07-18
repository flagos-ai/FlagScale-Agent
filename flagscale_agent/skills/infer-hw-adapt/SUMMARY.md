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

# Infer-HW-Adapt — Summary

Adapt and fix vllm-plugin-FL for specific hardware backends after plugin version upgrades.

**Load when**: a vllm-plugin-FL version upgrade breaks hardware-specific code paths (worker, model_runner, ops dispatch, platform detection), or when adding hardware support to an existing plugin version. Run after `infer-env-setup` confirms the environment is ready.

**Full cycle**: Stage 0 orientation -> Stage 1 unit tests -> Stage 2 functional tests -> Stage 3 offline inference -> Stage 4 serving -> Stage 5 clean-up -> Stage 6 PR.

**Key principles**:
- Test in strict order -- fix all failures at each stage before proceeding
- Never modify vLLM source -- all patches go through plugin
- One patch per failure -- fix, re-test, then move to next
- Patches are hardware-gated with `if current_platform.is_<backend>()`
- Every workaround has a TODO comment stating when it can be removed
- Stream and persist all logs to `/workspace/adapt-logs/`
- Squash all commits before PR

**Constraints**: 7 hard rules covering test order, source isolation, log persistence, patch discipline, platform gating, TODO requirements, and PR hygiene.
