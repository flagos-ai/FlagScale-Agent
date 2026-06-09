# Infer-Model-Adapt — Summary

Port a new model from the latest vLLM upstream into vllm-plugin-FL by migrating model code and adapting it for the plugin's architecture.

**Load when**: adding a model that vllm-plugin-FL does not yet support, or when an existing model needs to be re-ported after a major vLLM version bump that changed the model implementation.

**Full pipeline**: Step 0 orientation → Step 1 baseline unit tests → Step 2 upstream source → Step 3 plugin patterns → Step 4 model identity → Step 5 config bridge → Step 6 copy-then-patch → Step 7 register → Step 8 code review → Step 9 regression unit tests → Step 10 functional tests → Step 11 benchmark → Step 12 serve + request → Step 13 E2E correctness → Step 14 final report.

**Key principles**:
- Auto-detect installed vLLM version at Step 0; select patches based on actual version gap
- Copy upstream file verbatim first, then apply targeted edits — never rewrite from scratch
- One patch category at a time with import check after each batch
- All relative imports must be converted to absolute plugin-rooted imports
- Register model in vllm_fl/__init__.py using the exact model_type from HF config.json
- E2E correctness: compare token output against upstream GT server, allow late minor divergence (>token 15) from numerical noise but flag early divergence (<token 5) as a bug

**Constraints**: no vLLM source modification, copy-then-patch discipline, import verification after each patch batch, platform-gated hardware patches.
