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

# Model Porter — Summary

Port models from HuggingFace, papers, or other frameworks to Megatron-LM-FL for distributed training on FlagScale.

**Load when**: starting a model migration, doing checkpoint conversion, or analyzing model architecture for porting.

Three modes: Config-driven (YAML only, most LLMs), Megatron Native (full parallelism, custom architectures), HuggingFace Wrapper (FSDP2 fast path). Process: source analysis → whole-model implementation → checkpoint conversion → real-data verification → training. Key principle: analysis is per-component, but implementation is always whole-model-first with real data.
