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

Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware.

**Load when**: vllm-plugin-FL needs to track a new vLLM release, plugin breaks after vLLM dependency update, or unit tests fail with TypeError/ImportError/RecursionError after updating vLLM.

**Full pipeline**: Stage 0 orientation + version detection -> Stage 1 change analysis (unit test baseline + import audit + high-risk area check) -> Stage 2 fix API breakages (imports -> class/factory -> kwargs -> op schemas -> model-specific) -> Stage 3 unit test verification -> Stage 4 offline inference -> Stage 5 serving -> Stage 6 PR.

**Key principles**:
- Auto-detect both plugin version and installed vLLM version before any changes
- A minor bump brings breakages (errors) and silent behavioral shifts (require audit)
- Fix by error type in strict order: ImportError -> RecursionError -> TypeError -> AttributeError -> model-specific
- One patch per failure with per-fix verification before moving to next
- Never modify vLLM source -- all patches go through `vllm_fl/` plugin files only
- Validate on real NVIDIA GPU hardware before declaring done
- Squash all commits into one clean commit before PR

**Constraints**: no vLLM source modification, one-patch-at-a-time discipline, test stage order enforcement, squash before PR.
