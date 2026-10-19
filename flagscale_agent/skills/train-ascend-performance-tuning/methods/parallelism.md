<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Parallel Layout, Scheduling, and State Sharding

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about capacity, communication, pipeline bubbles, or stage/expert imbalance, or when the user explicitly requests a layout comparison.
Obtain candidate-relevant layout, MBS/GBS, layer distribution, and group information from the parent recipe and existing logs. If constraints are unclear, read the relevant section of `know-ascend-training`: `ascend_training/parallelism.md` as needed. For state sharding, read `ascend_training/memory-and-sharding.md` as needed.

## Generate candidates

Unless fully qualified, fields in the table belong under `train.system` in the composed configuration. Edit existing fields in place rather than defining them twice under system/model. Omit the `train` prefix when editing a child YAML mounted under `train`. Each row is a separate direction: choose nearby valid values rather than enabling the entire table at once.

| Direction | Concrete operation |
| --- | --- |
| Optimizer state sharding | If the current optimizer supports it and the actual sharding group has more than one rank, set `train.system.use_distributed_optimizer: true`. Keep the optimizer family, precision, and initial state. Do not also add communication overlap or change checkpoint format unless actual compatibility requires it. |
| TP: capacity / communication and compute granularity | Change `tensor_model_parallel_size`. Compare a smaller value when communication dominates and memory has headroom; compare a larger value when shard capacity is limiting. Keep PP/CP fixed. When reducing TP to 1, set `sequence_parallel` to `false`. In MoE, if ETP previously inherited TP by default, explicitly preserve the effective `expert_tensor_parallel_size` or record the ETP change as part of a joint candidate. |
| SP: activation capacity | At a fixed, validated TP>1, compare `sequence_parallel: false/true`. If the parent recipe enables `distribute_saved_activations`, disable it when enabling SP. Do not construct an invalid SP-off comparison when the current MoE combination requires SP. |
| PP: layer/state capacity and pipeline bubbles | Change `pipeline_model_parallel_size` while keeping TP/CP fixed, comparing a larger or smaller valid value according to the capacity or bubble hypothesis. If VPP or explicit layer allocation exists, update stage/chunk partitioning together. Check derived DP and microbatch count; adjust `train.model.micro_batch_size` jointly if needed, keeping `train.model.global_batch_size` fixed. |
| VPP: pipeline schedule | For an ordinary uniform layout with PP>1 fixed, compare chunk counts using `num_virtual_stages_per_pipeline_rank`. Remove the old `num_layers_per_virtual_pipeline_stage`; do not set the derived `virtual_pipeline_model_parallel_size` directly. Remove VPP definition fields for a non-interleaved comparison. If the recipe uses an explicit layout or hybrid pattern, adjust it through the next row instead of adding this field. |
| Stage layer allocation: load imbalance | With PP and the model's layer sequence fixed, adjust `decoder_first_pipeline_num_layers` / `decoder_last_pipeline_num_layers` when the first or last stage is slower, moving layers to stages with spare capacity. If the parent recipe uses `pipeline_model_parallel_layout`, move stage boundaries there while preserving layer types, total count, and order. If it uses `hybrid_layer_pattern`, adjust segments through that field without adding another layout/VPP definition. |
| CP: long-sequence capacity | Compare a larger `context_parallel_size` within the supported range of the current attention path; compare a smaller value when communication is costly and memory permits. Keep sequence length, TP/PP, and current `cp_comm_type` fixed. Check derived DP/accumulation and sequence-sharding requirements; do not shorten the sequence to make a candidate pass. |
| EP / ETP: expert capacity and communication | With dense TP/PP/CP fixed, change `expert_model_parallel_size` or `expert_tensor_parallel_size` separately. Explicitly fix effective ETP when comparing EP, and fix EP when comparing ETP. Preserve expert count, routing, top-k, capacity/token-dropping semantics, and dispatcher. Check expert groups and sharding. Generate a dispatcher comparison through the [communication method](communication.md). |

When coupling VPP with P2P overlap, use the configuration entry in the [communication method](communication.md) and check the actual schedule; include the coupled field in the full difference.
Keep MBS fixed by default; change it jointly only for an explicit capacity, compute, or schedule tradeoff. After a layout change, recheck derived DP, accumulation count, and effective GBS.

For deeper parallelism candidates, investigate how topology mapping, layer allocation, scheduling, and dense/expert layouts interact.
For directions without a fixed field listed, such as topology mapping, locate an entrypoint supported by the current stack. If source changes are necessary, identify the edit point and intended groups.

## Additional checks

- **New sharding path or resume:** Check actual sharding and updates of master parameters/optimizer state. Verify save/load when the task requires resume. Results with a different optimizer family, precision, or initial state are not equivalent comparisons under the original conditions.
- **Before launch:** Recheck only groups, shapes, and schedules affected by this change. For TP or vocab sharding, keep actual vocabulary and embedding/output shapes comparable.
- **Short-run activation:** Verify the changed layout, stage/chunk arrangement, or actual rank groups. For CP, also check attention, sequence splits, real positions, and mask/packing. For MoE, check expert mesh and dispatcher. Establish compatibility evidence for previously unverified combinations in the candidate's short run. Record silent rewrites or fallback; successful initialization is not an activation check.
- **Quality comparison:** Align loss, gradients, and parameter updates by actual shards under the same initial state and logical batch; do not compare local tensors merely because rank numbers match. Check the state-loading path actually used. If the task includes resume or requires complete restoration, also check optimizer, scheduler, RNG, and data progress. Read `ascend_training/state-and-resume.md` as needed for state alignment or resume.

Record the new peak memory after a layout change. Explore capacity limits only when the capacity goal or next candidate requires it. Explain quality differences in terms of actual shards, positions, or restored state. A successful cold start does not cover a failed resume.
