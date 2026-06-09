# Infer-HW-Adapt — Summary

Adapt and fix vllm-plugin-FL for specific hardware backends after plugin version upgrades.

**Load when**: a vllm-plugin-FL version upgrade breaks hardware-specific code paths (worker, model_runner, ops dispatch), or when adding hardware support to an existing plugin version. Run after `infer-env-setup` confirms the environment is ready.

**Full cycle**: Stage 0 orientation → Stage 1 unit tests → Stage 2 functional tests → Stage 3 offline inference → Stage 4 serving → Stage 5 clean-up → Stage 6 PR.

**Key principles**:
- Test in strict order — fix all failures at each stage before proceeding
- Never modify vLLM source — all patches go through plugin
- One patch per failure — fix, re-test, then move to next
- Patches are hardware-gated with `if current_platform.is_<backend>()`
- Every workaround has a TODO comment stating when it can be removed
- Stream and persist all logs to `/workspace/adapt-logs/`
- Squash all commits before PR

**Constraints**: 14 hard rules covering test order, source isolation, log persistence, patch discipline, platform gating, and PR hygiene.
