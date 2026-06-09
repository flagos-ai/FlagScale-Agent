# Model Porter — Summary

Port models from HuggingFace, papers, or other frameworks to Megatron-LM-FL for distributed training on FlagScale.

**Load when**: starting a model migration, doing checkpoint conversion, or analyzing model architecture for porting.

Three modes: Config-driven (YAML only, most LLMs), Megatron Native (full parallelism, custom architectures), HuggingFace Wrapper (FSDP2 fast path). Process: source analysis → whole-model implementation → checkpoint conversion → real-data verification → training. Key principle: analysis is per-component, but implementation is always whole-model-first with real data.
