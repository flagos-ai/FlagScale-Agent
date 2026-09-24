<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# First setup and changed-input checks

Read only the sections whose facts are missing or changed. Reuse a working environment and recipe; do not repeat these investigations on each tuning candidate. Hardware commands and runtime checks come from the selected device reference.

## Environment and launch process

Connect to the assigned server/container and activate the established environment. In non-interactive shells, use the existing environment's `conda run --prefix <env_path> ...` when needed. Preserve working cwd, environment variables and module paths; optional packages matter only when the recipe consumes them.

Verify runtime imports and actual module paths in the worker environment. Separately confirm the parent launch process can find the CLI or bounded helper: YAML worker settings do not configure its parent. Use command-scoped path additions only for a demonstrated missing path; do not edit shell startup files as routine preparation. Failed imports are a reason to diagnose the active environment before installing anything.

## Data and metadata

- For `mock_data: true`, check the synthetic dataset configuration; do not search for unused `.bin/.idx` files.
- For Megatron binary data, check the configured prefixes' `.bin` and `.idx` files.
- For parquet/JSONL or custom loaders, check actual referenced files and paths inside dataset registries/metadata, not just the top-level directory. Placeholder prefixes and metadata from another machine can point to the wrong data.
- For a new or changed custom pipeline, fetch a few samples through its actual dataset/loader with the training arguments before costly model loading. Do not rebuild a pipeline already validated by the baseline. Match its iterable/indexed interface and use a bounded job; measuring elapsed time after a fetch is not a timeout.

Correct a confirmed path mismatch in the task's config/metadata without changing dataset identity. If the required data is unavailable, report the missing input; `train-data-prep` is relevant only when data preparation is actually needed.

## Configuration and generated arguments

Check `GBS % (MBS × DP) == 0` and the selected implementation's model/layout constraints. Reuse a validated unchanged layout; GQA/KV replication, custom stage layouts and dynamic schedules cannot be reduced to universal equal-layer/KV-head divisibility rules. Preserve the recipe's working value types.

A new launch schema or changed generated arguments may need:

```bash
flagscale train -c /absolute/path/recipe.yaml --dryrun
```

Inspect that run's generated scripts under `<exp_dir>/logs/scripts/` for worker count, device mapping/backend, entrypoint and expected arguments. Dryrun generates scripts; it does not run forward/backward or prove the data path works. An MBS-only candidate with valid prior launch evidence needs its delta checked, not another source survey. The bounded helper requires a fresh `exp_dir`; use a separate validation directory if dryrun is needed first.

Set short-run iterations in a copied full YAML, not an assumed CLI `--train-iters` argument. In a tuning loop, the planned baseline/candidate run is the validation run. Reduced device count, dataset or GBS belongs only to an explicitly separate setup smoke test; it cannot replace fixed-workload measurement.

## Checkpoint and output capacity

Preserve the requested checkpoint identity, format and model/optimizer/RNG load scope. A layout change needs a supported restore/reshard path; a tracker filename or matching TP/PP alone does not prove compatibility. Confirm actual load/missing-key messages after startup; first loss alone cannot establish whether pretrained weights loaded.

For Megatron performance-only short runs, increasing `save_interval` does not disable the final save. Use an already verified no-save configuration if saving is unnecessary, otherwise budget its time and storage. Configuration semantics: `know-flagscale`, `flagscale/04_train_config.md` §3.5.

Reuse measured capacity where available. State dtype, sharding, activations and temporary buffers affect capacity; initialization success does not prove the first optimizer update fits. Return an OOM stage and evidence to the caller rather than silently shrinking its workload.
