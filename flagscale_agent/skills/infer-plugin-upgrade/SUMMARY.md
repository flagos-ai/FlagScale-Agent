# Infer-Plugin-Upgrade -- Summary

Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version detection, API diff
analysis, targeted fixes, and full test validation from unit tests through serving.

**Load when**: vllm-plugin-FL needs to be upgraded to a new vLLM version, or when the plugin breaks
after a vLLM dependency update.

**Full pipeline**: Step 0 orientation + version detection -- Step 1 API diff analysis (unit test
baseline + _C_cache_ops probe + high-risk area audit) -- Step 2 fix API breakages -- Step 3 unit test
verification -- Step 4 offline inference validation (dense, MoE, Mamba, VLM) -- Step 5 FlagGems checks
-- Step 6 serving validation -- Step 7 PR.

**Key principles**:
- Auto-detect both plugin version and installed vLLM version before any changes -- never assume
- Every upgrade has different breakages; read the actual error and trace it to changed vllm code first
- Fix by error type: TypeError (stale kwargs) / ImportError (moved symbols) / RecursionError (class vs factory) / AttributeError (_C_cache_ops missing)
- One patch per failure -- fix, verify import or unit test, then move to next
- Never modify vLLM source -- all patches go through `vllm_fl/` plugin files only
- Validate on real NVIDIA GPU hardware before declaring done

**High-risk areas to audit each upgrade**:
- `vllm_fl/ops/fused_moe/layer.py` -- FusedMoE class vs factory, FusedTopKRouter signature
- `vllm_fl/worker/model_runner.py` -- InputBatch params, use_uniform_kv_cache signature, WorkerProc entry point
- `vllm_fl/ops/_C_ops_schemas.py` -- diff registered schemas vs installed ops with check_ops.py
- Any `from vllm.X import Y` -- symbol may have moved to a different module

**Constraints**: no vLLM source modification, one-patch-at-a-time discipline, TODO comment on every
workaround, squash all commits before PR.
