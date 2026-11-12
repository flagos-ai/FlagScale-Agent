<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Micro-Batch Tuning at Fixed GBS

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about compute granularity, dispatch overhead, pipeline bubbles, or activation capacity.
Obtain effective MBS, GBS, DP, and schedule from the parent recipe and existing logs. If batch, accumulation, or comparability is unclear, read `know-ascend-training`: `ascend_training/batch-and-accumulation.md` as needed.

## Generate candidates

Keep GBS and the current parallel layout fixed. Select a valid MBS from the positive integer factors of `GBS / DP`, then check model and pipeline-schedule constraints. Do not restrict choices to powers of two or assume the largest runnable value is best.
The paths below refer to the composed configuration. Edit existing fields in place to avoid duplicate definitions. Omit the `train` prefix when editing a child YAML mounted under `train`.

| Direction | Concrete operation |
| --- | --- |
| Larger compute units, fewer microbatches | Set `train.model.micro_batch_size` to a larger valid value while keeping `train.model.global_batch_size` fixed. Compare full-update time and test whether the benefit outweighs increased activations and the changed microbatch count. |
| Lower activation peak or better pipeline fill | Set `train.model.micro_batch_size` to a smaller valid value while keeping GBS fixed. This is also worth trying for pipeline bubbles, not just OOM. First identify whether activations cause the pressure; if resident weights/optimizer state dominate, use [parallelism and state sharding](parallelism.md). |
| Joint MBS and recomputation | When a combination is justified, adjust MBS together with `train.model.recompute.recompute_modules` or `recompute_num_layers`, matching the current mode through the [recomputation method](recompute.md). For example, saved activations may permit less replay, or recomputation may permit a larger MBS. Record both differences; neither component must improve performance alone. |
| Joint MBS and pipeline schedule | Keep GBS fixed. For an ordinary uniform VPP layout, adjust MBS together with `train.system.num_virtual_stages_per_pipeline_rank`; use the [parallelism method](parallelism.md) for other layouts. Recompute microbatch count and chunk constraints; do not make a candidate valid by increasing GBS. |

If a constant-batch configuration omits GBS, first write the parent configuration's actual GBS explicitly as `train.model.global_batch_size`, then change MBS so the default does not shift. Megatron normally derives accumulation count from GBS/MBS/DP; do not add unsupported `gradient_accumulation_steps` to this entrypoint.
Preserve the semantics of any dynamic batch schedule and compare at the same progress and batch stage. Do not add a fixed GBS that conflicts with the schedule. Do not let `decrease_batch_size_if_needed` silently round away a GBS change.

For example, with PP=1, GBS=64, DP=16, and MBS=1, each update has 4 microbatches; MBS=2/4 yields 2/1, while MBS=3 does not divide evenly. This illustrates batch arithmetic only; it proves neither model support nor better performance.
For deeper MBS candidates, investigate actual GEMM/attention shapes, dispatch and communication calls per update, in-flight activations, and schedule bubbles to form a broader set of valid MBS or joint candidates. Confirm that extensions such as dynamic microbatching are integrated before using them.

## Additional checks

- **Ordinary comparison:** Confirm actual MBS, accumulation count, and runtime GBS. Keep logical samples per update, sequence/packing, precision, and optimizer unchanged. Measure a complete update and peak memory. With graph execution, complete necessary warmup for the new shape before measuring steady state.
- **Dynamic batch or staged configuration:** Confirm the comparison window stays within one batch stage. If the parent recipe indexes recomputation/schedules by microbatch, check coverage; do not change MBS and retain a mismatched table. A short run covering one stage does not validate the entire schedule.
- **Quality:** Variable-length masks, packing, or within-batch MoE statistics may break equivalence even at fixed GBS. When results differ, check loss weights, accumulation scaling, gradients, and updates as needed. Finite loss alone does not establish equivalence.

No gain from a larger MBS does not invalidate this direction; compute granularity, pipeline bubbles, and memory can favor different values. Results with a changed effective GBS cannot enter the original workload ranking. Reassess MBS limits under a different recomputation, graph, or layout choice.
