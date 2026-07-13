# Infer-Plugin-Upgrade — Summary

Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version detection, API diff analysis, targeted fixes, and full test validation from unit tests through serving.

**Load when**: vllm-plugin-FL needs to be upgraded to a new vLLM version (e.g., 0.20.2 → 0.24.0), or when the plugin breaks after a vLLM dependency update.

**Full pipeline**: Step 0 orientation + version detection → Step 1 API diff analysis → Step 2 unit tests (establish baseline) → Step 3 fix API breakages → Step 4 offline inference validation → Step 5 multi-model validation → Step 6 serving validation → Step 7 PR.

**Key principles**:
- Auto-detect both the current plugin version and the target vLLM version before any changes
- Fix API breakages in order: imports → class/factory changes → signature changes → schema changes
- One patch per failure — fix, verify, then move to next
- Never modify vLLM source — all patches go through plugin files in `vllm_fl/`
- NVIDIA A800 is the primary validation target; all models must pass offline inference

**Known patterns across vLLM minor versions**:
- `FusedMoE` may change between class and factory function → OOT registration breaks, use monkey-patch
- `InputBatch.__init__()` gains/loses kwargs between versions → use `**kwargs` shim
- `_C_cache_ops` op set changes → check for missing ops before running model tests
- `use_uniform_kv_cache()` signature changes → remove stale kwargs

**Constraints**: no vLLM source modification, one-patch-at-a-time discipline, all fixes platform-gated if hardware-specific, TODO comment on every workaround.
