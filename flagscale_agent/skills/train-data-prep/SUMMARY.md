# Data-Prep — Summary

Prepare training data for FlagScale in Megatron binary format (.bin/.idx) and Megatron-Energon multimodal format.

**Load when**: preparing training data, converting datasets to Megatron format, setting up tokenization, or debugging data pipeline issues.

Three pipelines: Pipeline A (GPT-style pretraining with document-level tokenization to .bin/.idx), Pipeline B (Megatron-Energon multimodal with webdataset .tar shards), and instruction-style (SFT with chat templates). Covers tokenizer selection, data conversion, blending ratios, multimodal packer configuration, and validation.
