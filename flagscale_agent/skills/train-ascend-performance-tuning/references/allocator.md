<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# NPU Allocator Settings

Read only when there is evidence of variable-size allocations or fragmentation. For principles, consult `ascend_training/memory-and-sharding.md` in `know-ascend-training` as needed.

Given that evidence, confirm support in the installed TorchNPU version and chip. Then compare a run with `expandable_segments:True` merged into `PYTORCH_NPU_ALLOC_CONF` under `experiment.envs`. Preserve unrelated settings and handle conflicts according to this version's constraints. Do not create another candidate if an equivalent setting is already active.

Keep the workload and other settings fixed, and run complete updates. Compare complete-update time, allocated/reserved peaks on the most loaded rank, and the original OOM stage. `reserved > allocated` alone does not prove fragmentation.
