<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Activation Saving, Recomputation, and Offload

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about activation saving, replay, or offload. This may lower the activation peak or trade more saved memory for less replay.
For recomputation boundaries, read `know-ascend-training`: `ascend_training/recompute.md` as needed; for offload mechanisms, read `ascend_training/memory-and-sharding.md`. If resident weights/optimizer state dominate, use [parallelism and state sharding](parallelism.md).

## Generate candidates

Start from the parent recipe's effective configuration. State which memory source or replay cost should fall, then choose an operation. The recomputation fields below belong under `train.model.recompute`:

| Direction | Concrete operation |
| --- | --- |
| Selective recomputation | Set `recompute_granularity: selective` and choose boundaries with labels the current model actually supports, such as `recompute_modules: [mlp]`. If already selective, add or remove one target module. Remove `recompute_method` and `recompute_num_layers`. Do not assume recomputing `core_attn` is preferred with fused attention. |
| Partial-layer recomputation | Set `recompute_granularity: full`, `recompute_method: block`, and `recompute_num_layers: 1`, or select a nearby value from the current effective layer count. The count must fit actual PP/VPP local chunks. Remove stale `recompute_modules`. |
| Recomputation grouping | Under full recomputation, compare `recompute_method: uniform` and set `recompute_num_layers` to a group size allowed by the current implementation, for example `1`. This controls layers per group, not the total number of layers recomputed. Remove stale `recompute_modules`. |
| Reduce existing recomputation | With memory headroom, reduce selective modules or block layers; compare uniform with a partial-layer block configuration where useful. To disable entirely, remove the recomputation configuration and any old aliases or `*_per_stage_micro_batch` overrides in the parent recipe that would re-enable it. Do not use an empty module list or YAML `null` as a substitute for disabling. |
| Shard checkpoint inputs | If TP>1, full recomputation is active, SP is off, and the implementation supports it, set `train.model.recompute.distribute_saved_activations: true`. Compare saved-input memory reduction against restore-communication cost. Do not disable existing SP by default just to enable this option. |

For deeper activation-memory candidates, investigate peak rank/stage and tensor lifetimes. Explore better save boundaries, output-discard recomputation, replay scheduling, or activation offload.
For offload or other community features, confirm integration in the current model and NPU path before forming a configuration or code candidate. Do not transplant MindSpeed switches directly.
For MBS changes, use the [micro-batch method](micro-batch.md); for TP/PP/CP/EP, the [parallelism method](parallelism.md); for communication buffers and overlap, the [communication method](communication.md); for fusion and temporary tensors, the [operator method](operator.md).

When switching modes, check whether existing `distribute_saved_activations` and staged overrides in the parent configuration remain valid.

## Additional checks

- **Ordinary configuration comparison:** Confirm effective parameters and target modules. Include first optimizer-state allocation and a complete update. Record available peak rank, stage, and memory measure.
- **New recomputation boundary or unexpected result:** As needed, distinguish original forward from backward replay and check RNG/dropout, gradients, and updates. Calling a checkpoint entrypoint does not prove replay occurred. With a new offload path, confirm transfer completion before tensor consumption, host memory use, and transfer cost.

An OOM shifted to another stage still means capacity is insufficient. If there is no gain, distinguish replay already covered by an outer boundary, a peak from another source, or transfer costs offsetting savings. Investigate the implementation when reentry is abnormal.
