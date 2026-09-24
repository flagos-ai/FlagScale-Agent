<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Communication Scheduling and Overlap

## When to use

Use this method when configuration, logs, or a profile support a hypothesis about communication waits, extra buffers, or degraded overlap.
For parameter dependencies, read `know-ascend-training`: `ascend_training/communication.md` as needed. For recomputation, read `ascend_training/recompute.md`; for graph execution, read `ascend_training/graph-execution.md`.

## Generate candidates

Start from the parent recipe's effective configuration. Choose an operation supported by evidence and state which wait or memory cost it should reduce:

| Direction | Concrete operation |
| --- | --- |
| DP gradient synchronization / parameter gather | Where this optimizer version's dependencies are met, compare enabled and disabled `overlap_grad_reduce` or `overlap_param_gather`. Keep gradient synchronization overlap enabled when isolating gather. When disabling gradient synchronization overlap, also disable gather overlap if it depends on it, and record the coupled change. |
| Gradient buckets | Identify the effective field, value, and unit. Generate one nearby smaller or larger value based on the hypothesis. Keep other communication settings fixed; compare full-update time and peak memory. |
| PP P2P | If the current PP/VPP schedule permits it, disable with `train.system.no_overlap_p2p_communication: true`. For an enabled comparison, remove this disabling field or set it to `false`. The internal effective field is `overlap_p2p_comm`; do not add it directly as a YAML switch. If either state is invalid, do not flip it in isolation. Handle layout dependencies through the [parallelism method](parallelism.md). |
| MoE dispatcher / overlap | Choose one dispatcher replacement or dispatch/combine or shared-expert overlap comparison among implementations already integrated in this stack. Preserve routing, top-k, capacity, and token-dropping semantics; record required coupled changes. |

For deeper communication candidates, examine message sizes and call counts, rank load and topology, compute/communication dependencies, and buffer lifetimes along the relevant communication path. Extend analysis across modules and ranks if needed.
Use the findings to explore communication coalescing/chunking, TP/CP/EP layouts and mapping, dispatch/combine data flow, or deeper scheduling and backend changes.
For layout changes, use the [parallelism method](parallelism.md); for operator implementation changes, use the [operator method](operator.md).

## Additional checks

- **Ordinary configuration changes:** Confirm the target parameter takes effect, the bucket unit, and dependencies required by this change. Reuse validated group, optimizer, and schedule conditions.
- **Group, dispatcher, or schedule changes:** Check communication order, gradient reduction/accumulation, and buffer lifetimes on affected ranks. With delayed wgrad, confirm gradients complete before reduction and optimizer consumption. A bucket-only adjustment does not require rechecking the entire schedule and graph mode.
- **Unexpected results or unclear mechanism:** Collect a timeline only as needed to distinguish the current hypotheses. A configuration flag set to `true` or a called entrypoint does not prove actual overlap. Keep diagnostic runs separate from profiler-free performance measurements.

Without timeline evidence, report end-to-end gains without claiming reduced communication wait was proven. Use evidence about layout, load, or implementation for the next candidate.
