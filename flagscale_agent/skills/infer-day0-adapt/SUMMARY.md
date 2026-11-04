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

# Infer-Day0-Adapt — Summary

Deliver newly released model support in vllm-plugin-FL or sglang-plugin-FL,
including when no upstream model implementation exists yet.

**Load when**: a new model needs architecture/checkpoint analysis, operator gap
resolution, framework integration, and reproducible support evidence. Pure
framework upgrades and new hardware backends use their dedicated skills.

**Workflow**: establish exact model/environment revisions → trace architecture
and checkpoint mappings → classify reusable APIs and operator gaps by contract
→ deliver operator APIs/tests → integrate through existing plugin extension and
dispatch points → lock public dependencies → verify actual backend hits,
independent numerics, real inference, applicable graph and performance paths
→ update supported-model status and a runnable example.

**Key requirements**:

- Keep model orchestration and request metadata in the framework; place
  numerical computation in FlagGems or the framework-specific operator library.
- Select implementations before execution; never retry on another backend after
  a launch, mutation, state update, or collective has started.
- Keep persistent cache/workspace ownership and stream ordering in the framework.
- Read framework, operator, and quantization references according to the change.
- Validate graph replay using new inputs, sentinel outputs, and equal initial
  state; apply independent numerical references and justified tolerances.
- Distinguish implemented/loadable/numerically verified/end-to-end verified
  support; unavailable hardware is an explicit verification gap.
- Keep transient evidence in PR/CI artifacts and maintain real user examples.

Based on **推理框架插件开发规范v1** and **新模型适配算子层开发规范**;
the references retain clause identifiers and mandatory/recommended distinctions.
