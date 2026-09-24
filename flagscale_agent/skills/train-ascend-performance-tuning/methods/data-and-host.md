<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Data Supply and Host Scheduling

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about data waits, CPU contention, or host dispatch overhead.
Reuse the known data path and wait evidence. For principles, read `know-ascend-training`: `ascend_training/data-and-host.md` as needed. Read the relevant `know-energon` section only when Energon reader or resume semantics are unclear.
For graph capture or compilation boundaries, use the [graph and compilation method](graph-execution.md); for host round trips and synchronization inside an operator, use the [operator method](operator.md).

## Generate candidates

Start from the parent recipe's effective configuration. Choose an operation and state whether it should reduce read waits, CPU contention, or dispatch overhead. Paths below refer to the composed configuration. Omit the `train` prefix when editing a child YAML mounted under `train`, and edit existing fields in place.

| Direction | Concrete operation |
| --- | --- |
| Data read/preprocessing concurrency | Change `train.system.num_workers` to a nearby larger or smaller value. Increase it when decoding is slow and CPU capacity is available; decrease it when CPU contention or IPC overhead dominates; compare `0` if useful. Preserve sampler, data, and processing semantics. Keep Energon on its existing `WorkerConfig`/loader path; do not replace it with a standard DataLoader. |
| Data prefetch depth | Adjust the effective prefetch value through an existing configuration entry. If a standard PyTorch loader exposes no configuration field and the task permits source edits, pass `prefetch_factor` explicitly at the actual `DataLoader(...)` construction only for `num_workers > 0`, for example comparing 2 and 4. Do not add a nonexistent YAML field or apply this parameter to Energon. |
| CPU affinity | When CPU migration, preemption, or NUMA access warrants tuning, compare the current mode with the supported `"1"`/`"2"` values of `experiment.envs.CPU_AFFINITY_CONF`. Preserve any custom core-range mapping. Choose ranges from the container's allowed CPU cores and NPU topology; do not copy core numbers from another machine or override existing binding by default. |
| NPU task dispatch queue | If the current binary execution path supports Level 2 and dispatch overhead matters, compare `"1"`/`"2"` for `experiment.envs.TASK_QUEUE_ENABLE` and observe memory. If `ASCEND_LAUNCH_BLOCKING=1` remains set, distinguish diagnostic from performance runs. After diagnosis, restore the asynchronous `"0"` baseline; do not rank a blocking diagnostic run with normal runs. |

For deeper data/host candidates, inspect dependencies across read, preprocessing, H2D, and execution; CPU hotspots; and implicit synchronization. Explore batched preprocessing or a transfer pipeline.
Asynchronous H2D must preserve source-buffer lifetime and consumer dependencies. Adding `non_blocking=True` alone does not establish overlap.
Do not reconvert valid data or replace it with mock data.

## Additional checks

- **Data/host configuration:** Confirm actual workers, prefetch, or thread affinity. Check sample order, valid tokens, update progress, host/shared memory, and peak NPU memory. Verify resume only when data-state recovery is involved. A briefly prefilled queue is not evidence of sustained supply gains.

Supply gains require evidence from sustained execution. Less local waiting does not automatically mean faster training, and mock data cannot establish gains for the real data path.
