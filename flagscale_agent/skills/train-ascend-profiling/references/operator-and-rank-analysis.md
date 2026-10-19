<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Operator-Level Analysis and Slow-Rank Comparison

## Two tools, two questions

| Tool | Command | Answers |
| --- | --- | --- |
| `scripts/op_analysis.py` | `hot-ops`, `pivot`, `shapes`, `roofline` | Which operators dominate, how their pipelines were used, and whether a kernel's time is close to its modelled bound |
| `scripts/slow_rank.py` | `logs`, `csv` | Whether one rank's logged step time differs enough from its peers to be worth investigating |

Neither tool decides what to optimise. They produce ranked observations with the
evidence boundary attached.

## Which CSV to use

| File | Contains | Use for |
| --- | --- | --- |
| `op_statistic.csv` | per-operator totals | first-pass ranking |
| `op_summary.csv` | per-invocation rows with shapes | pivot, shape parsing |
| `kernel_details.csv` | per-kernel rows, usually with `aic_mac_ratio` | Roofline, pipeline ratios |

Only per-invocation files can establish intervals and gaps; aggregate tables can
rank but cannot show when anything ran.

## Cumulative duration is not critical-path cost

`hot-ops` sums recorded durations. That number:

- double-counts concurrent work,
- says nothing about whether the work was on the step's critical path,
- changes with MoE routing and recomputation.

So a top-ranked operator is a *candidate to look at*. Confirm it inside a window
before attributing a bottleneck.

## Instruction-ratio pivot

`pivot` averages the pipeline-occupancy columns (`aic_mac_ratio`, `aiv_vec_ratio`,
`aic_mte1/2_ratio`, …) per operator and input shape. A high `aic_mac_ratio` means
the cube pipeline was busy — not that the kernel is at peak, and not that a faster
kernel exists.

## Roofline: a bound, not a verdict

For the two-operand MatMul form the tool computes

```
theoretical_us = max(FLOPs / peak, lower_bound_bytes / bandwidth)
efficiency     = theoretical_us / observed_us
```

Read it as: *if this kernel were limited only by the named resource, it would take
`theoretical_us`*. Consequences worth keeping in mind:

- Low efficiency is a **prompt to investigate**, not proof of a fixable stall. Real
  kernels lose time to tiling tails, bank conflicts, launch overhead and small
  problem sizes that the bound does not model.
- Efficiency above 1.0 means the model undercounts work or traffic — the shape,
  dtype or peak is wrong. Do not report it as a speedup.
- Only operators above `--mac-threshold` are checked, because the bound assumes the
  compute pipeline is the limit.
- Operators whose shape is not a two-operand matrix are listed as `unmodelled` with
  a reason. They are never estimated with an invented FLOPs figure. Batched,
  grouped and fused variants need their own accounting.

The peak and bandwidth are **required inputs**. Do not fall back to a remembered
number for a different SKU or precision.

## Slow-rank comparison

`slow_rank.py` centres on the **median** and flags outliers by a modified
z-score on the median absolute deviation, so one extreme rank cannot mask the
others. A rank is reported only if it clears *both* a minimum delta and the
outlier bar.

This is deliberately weaker than a threshold: pipeline stages, expert routing and
data shards are legitimately asymmetric, so a difference is a candidate. The tool
states the limit out loud — a **uniform** slowdown across all ranks is invisible to
a peer comparison, so "no candidates" is not "balanced". For a uniform slowdown,
compare against the previous configuration instead, and use the profiler.

After a candidate appears, the next question is *why*: check the PP stage and layer
assignment, token distribution, data shard, and communication waits. A slow rank is
a symptom, and the cause is usually upstream of the rank itself.
