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

# Data-Prep — Summary

Prepare training data for FlagScale in Megatron binary format (.bin/.idx) and Megatron-Energon multimodal format.

**Load when**: preparing training data, converting datasets to Megatron format, setting up tokenization, or debugging data pipeline issues.

Three pipelines: Pipeline A (GPT-style pretraining with document-level tokenization to .bin/.idx), Pipeline B (Megatron-Energon multimodal with webdataset .tar shards), and instruction-style (SFT with chat templates). Covers tokenizer selection, data conversion, blending ratios, multimodal packer configuration, and validation.
