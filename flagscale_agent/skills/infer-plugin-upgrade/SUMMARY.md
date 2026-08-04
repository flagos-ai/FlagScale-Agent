# Infer-Plugin-Upgrade -- Summary

Upgrade vllm-plugin-FL to a new vLLM version on NVIDIA hardware. Covers version detection, change
analysis (both API breakages and new features requiring plugin glue), targeted fixes, and full
validation from unit tests through serving.

**Load when**: vllm-plugin-FL needs to be upgraded to a new vLLM version, or when the plugin breaks
after a vLLM dependency update.

**Full pipeline**: Step 0 orientation + version detection + rollback anchor -- Step 1 change analysis
(unit test baseline + _C_cache_ops probe + new-feature scan + high-risk area audit) -- Step 2 fix API
breakages (with per-fix verification) -- Step 3 unit test verification -- Step 4 offline inference
validation (dense, MoE, Mamba, VLM, speculative decoding) -- Step 5 FlagGems checks -- Step 6 serving
validation -- Step 7 PR.

**Key principles**:
- Auto-detect both plugin version and installed vLLM version before any changes -- never assume
- A minor bump brings two kinds of changes: breakages (surface as errors) and new features (silent)
- For breakages: read the actual error and trace it to changed vllm code first
- For new features: actively scan for new wrappers, env vars, proposer types, and data structure fields
- Fix by error type: TypeError (stale kwargs) / ImportError (moved symbols) / RecursionError (class vs factory) / AttributeError (_C_cache_ops missing)
- One patch per failure -- fix, verify with a targeted import check or unit test, then move to next
- Never modify vLLM source -- all patches go through `vllm_fl/` plugin files only
- Validate on real NVIDIA GPU hardware before declaring done

**High-risk areas to audit each upgrade**:
- `vllm_fl/ops/fused_moe/layer.py` -- FusedMoE class vs factory, FusedTopKRouter signature
- `vllm_fl/worker/model_runner.py` -- InputBatch params, get_model() wrapper handling, new proposer dispatch branches
- `vllm_fl/ops/_C_ops_schemas.py` -- diff registered schemas vs installed ops with check_ops.py
- Any `from vllm.X import Y` -- symbol may have moved to a different module

**Constraints**: no vLLM source modification, one-patch-at-a-time discipline, TODO comment on every
workaround, pre-existing test failures documented in known_failures.txt, squash all commits before PR.
