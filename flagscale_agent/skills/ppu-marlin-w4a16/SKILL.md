---
description: Optimize or audit W4A16 Marlin GEMM / fused MoE on PPU, covering ACU
  per-PC attribution, packed INT4 decoding, MMA thread layout, shared bank conflict,
  prefetch and register lifetimes, and verifying Triton/FlagGems source landing. Applies
  to Up+SiLU, Down, decode/prefill performance analysis and experiment design; not
  for generic CUDA tuning.
name: ppu-marlin-w4a16
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# PPU Marlin W4A16 Optimization

Determine whether work is necessary, who produces the data, and who consumes it before optimizing bank mapping and scheduling. The goal is a correct, reproducible net gain over the full path — not zeroing out a single counter.

## Getting Started

1. Read-only: confirm the actual entry point, Up/Down binary SHAs, source and toolchain identity, routing samples, and the allowed modification scope. Distinguish retained AOT, source-generated, custom-pass, and research candidates; never infer version from filename alone.
2. Clarify the workload: A is activations, B is expert weights. Identify the quantization group, packed format, N/K, top-k, routing density, and the actual number of rows per expert/tile. PPU Marlin K32 is not the same as generic CUDA Marlin.
3. Keep the baseline; work on candidates in an isolated directory/cache. Verify device availability first. Do not automatically restore revoked entry points, upgrade the global toolchain, modify vendor vLLM, or restart services. Historical compiler authorization does not automatically extend to the current task scope.
4. Each round: write a falsifiable hypothesis, the target ISA delta, the conditions that must hold, and the expected cost. Audit in this order: math/layout → ISA data path → resources/conflicts → lifetimes/pipeline → micro-scheduling. Choose the next item from current evidence; do not mechanically replay historical optimizations.

## Core Decisions

- Prioritize eliminating unnecessary relay: read packed W on the MMA consumer threads, decode into full BF16×2 fragments, and avoid B shared round-trips. This requires preserving coalescing, CTA reuse, and scale mapping; bypass is not inherently faster.
- Attribute bank conflicts first by actual instruction, lane address, and transaction width; then design a swizzle that is symmetric for reads and writes. Zeroing conflicts while adding conversions, scalarization, or synchronization is not a success.
- Load packed W/scale in wide batches, decode per-K32 fragment, and consume early to avoid keeping an entire large-K BF16 fragment live. Verify the actual in-flight window, not just where loads appear in source.
- Validate both the A and W prefetch chains, scale consumption, full MMA input coverage chain, barrier/wait, spill, and occupancy together. Reducing a stall at one PC may only move the wait to the first read or the next iteration.
- Prefer expressing maintainable mechanisms in FlagGems/plugin source. When source cannot express them, clearly identify the compiler gap; isolated ISA/AOT experiments are for validating mechanisms, not for claiming pure Triton landing.
- Audit Up and Down separately. Analyze decode/prefill using real tile density and resources; do not restore the `M == 64/4096` specialized dispatch. Smaller-M BF16 MMA had precision issues in historical environments and must not be adopted without full current validation.

## Delivery and Stopping

Each round report: the change and its rationale, whether the target mechanism compiled, correctness, resources and memory access, Up/Down/joint-chain timing in the same session with spread, ACU attribution, code diff, generality, and promotion decision. Distinguish latency reduction from throughput increase.

If the first version shows no gain, verify whether the implementation actually realized the theoretical mechanism before concluding. Do not use "failed to compile", wrong results, or confounded variables to dismiss a direction. When the mechanism is confirmed, neighboring scheduling, reasonable resource configurations, and representative inputs consistently show no stable net gain with an explicable cost — close that local direction. Do not promise guaranteed theoretical gains, unbounded parameter sweeps, or claim the device is globally optimal. When resources or evidence are insufficient, explicitly label the result as incomplete.

---

## Optimization Mechanisms and Failure Lessons

### 1. Direct Decode and Consumption Layout

Derive the expert, packed word, nibble, and scale group from the logical coordinates of the final MMA tile, then read and decode; do not first produce multiple large BF16 tensors and then concat/transpose. INT32 is a storage container (typically eight INT4 values), not a compute precision. The exact format must be verified from the packer/unpacker — "Marlin" naming is not interchangeable.

Historical large gains came from direct tile decoding, layout, and the 2-warps/1-stage combination together, not from removing a single concat. BF16×2 PRMT/SHR/LOP3 and ADD/MUL can reduce conversion cost, but element-wise BF16 rounding is not equivalent to FP32 accumulation/FMA.

**B Register Bypass**

Build the mapping `(expert, n, k) → packed word/nibble → producer lane/register → consumer lane/register` for each MMA's B input. Have the read, unpack, and scale broadcast directly form that fragment, eliminating the shared store/read, exchange, and corresponding synchronization.

First check whether B is already bypassed in the current implementation; do not optimize a path that does not exist. Early bypass attempts increased LSU requests, L2 traffic, and redundant loads; matching the consumer layout, wide reads, and reuse led to eventual success. After compilation, check for extra shuffles, masked moves, and shared exchanges. A source-level reshape or native MMA call does not by itself guarantee zero conversions.

When comparing to CUDA/Humming, distinguish compatibility compilation from PPU-native optimization. Conversions between MMA fragments can recur throughout the accumulator chain; static instruction counts are for locating issues only and cannot directly explain timing proportions.

### 2. A Shared Layout and Slot Release

A is the hidden states for Up, or SiLU(gate)×up for Down. The typical path is global → async shared → register → MMA; after B bypass, A synchronization is still required.

Attribute actual conflict instructions for A writes, A reads, scale, and packed W separately. Use the device's current bank/transaction model to enumerate each lane's byte address, bank, and vector sub-transactions; read broadcasts, write conflicts, and cross-transaction counts must not be conflated.

A swizzle must simultaneously satisfy: write and read addresses are mutually inverse/consistent, slots do not overlap, effective addresses are within the allocation, vectorization is preserved, and the consumer fragment order is correct. Do not add padding mechanically. The mask `p ^ ((p & 0x100) >> 2)` can vary per lane and must not be unconditionally hoisted to a uniform scalar constant without proof.

Specify each slot's lifetime: async write complete → released to consumer → shared read complete → last accessor releases → next-round overwrite. If the MMA already reads from registers only, whether a slot can be released early depends on all shared consumers and synchronization contracts — it need not wait for the entire MMA iteration; but cross-warp producer/consumer synchronization cannot be omitted.

Double-buffering should maintain statically/optimizably expressible address computation; dynamic shared indexing has previously broken vectorization or caused spill. Before increasing slot count, check actual occupancy rather than only the total shared budget. Experiments must observe refill, first-read, read-complete, and tail waits — zeroing the refill sync alone is not a success.

### 3. W/Scale Wide Load and Pipeline

Loading packed data wide, decoding in fragments, and consuming early is generally better than keeping large BF16 blocks live simultaneously. W and scale must jointly match the consumption layout; increasing W batch size alone has previously made things slower. Examine actual load width, transactions, repeated requests, and the in-flight window.

Distinguish between source `num_stages`, actual shared slot count, W/scale prefetch distance, and register double-buffering — these are not the same parameter. When K32 is the quantization/packing unit, changing the compute BK does not mean the weight format can be changed arbitrarily.

Cross-iteration carry-forward requires auditing prologue, steady state, and epilogue: whether a new value is defined before the last consumption of the old target, whether a back-edge COPY is introduced, and whether register reuse forces `vldcnt(0)` at the loop tail. Moving the entire block to the last consumer may eliminate the COPY but lose in-flight time; prefer interleaving by fragment.

More VGPRs can expand the in-flight window, but actual register allocation and CTA occupancy steps must be checked. The measured values 96/100/124/126 are not configuration ceilings; `maxnreg` or tied asm may be ignored/eliminated — compare the produced artifacts. Recomputation is beneficial only when it genuinely shortens the live range and the extra instructions can be hidden.

Scale early load, shared spill, broadcast, and last consumption are distinct events. Isolate each one while preserving W-next and A issue width. Encoding/pseudo-dependencies can add instructions, copy-move overhead, and even reduce occupancy — that is not a clean mechanism experiment.

### 4. Release and MMA Input Coverage

Audit reaching definitions, last reads, next writes, and address/SCC chains for the entire fragment, not just moving the first PRMT. Physical register coverage may have WAR constraints; it cannot be asserted directly equal to the full MMA compute latency.

Placing independent scalars between load and wait may shorten the wait while canceling the MMA → next-input coverage gap. Perform a valid spacing comparison and check hot spots on both sides and total time; a two-halves unrolled structure may differ and cannot be mechanically copied.

For `vldcnt(n)`, first verify the completion semantics and all producers in the queue for that ISA. Under historical semantics, 6→5 is a stricter wait, not an earlier release. Merging loads changes the number of outstanding requests and does not authorize directly removing a wait; a scale wait may also protect W.

Adjusting B store cadence applies only to old paths that still have B shared. Audit together with the last multiply and store producer to avoid equivalent transfer of fence and multiply→store dependency; after bypass this optimization no longer applies.

### 5. Padding, Folding, and Scenarios

Padding 384 real routed rows to 2944 does not mean 87% of CTAs are entirely empty. Many BM16 tiles aligned to an expert contain only a few real rows; masking does not cause the MMA to automatically compute less. Skipping applies only to genuinely empty tiles/experts; check sort boundaries and launch.

Different M/N tiles for the same expert can repeatedly read/decode the same weights. Evaluating reuse must be done per CTA workload, not by claiming each expert decodes only once.

N→M block-diagonal folding uses sparse rows to carry two N halves, potentially shrinking the accumulator and reducing redundant execution, but `16×2K×N/2 = 16×K×N` — this does not automatically reduce MACs. Historically, Down selected folding based on the validity of the ninth route per tile; the optimal folding for Up has never stably outperformed non-folding. Never decide by the expert's total token count — the tail tile must be checked.

Large M/prefill requires considering higher tile density, weight reuse, memory, and dynamic routing. Test non-aligned M and maximum M; do not treat the historical M4096 experiments as the only dispatch. Adding pre-packed copies requires reporting persistent memory, peak temporary footprint, and lifetime.

### 6. Source Code Landing

Distinguish in sequence: mathematical expression → packed BF16×2 lowering → distributed/shared layout → release/consumption scheduling → backend allocation. Plain Triton, Gluon explicit layout, PPU inline asm, custom pass, and offline ISA patch must each be labeled accurately.

For each migrated item, close the corresponding hook, recompile from a fresh cache, and compare actual ISA, numerical correctness, resources, and performance. Using the system FlagTree does not imply no custom passes. `is_pure=False` is not a universal scheduling barrier; bound strings may be lost after lowering. Modifying only shared metadata does not fix wrong addresses.

A historical PPU wait_group lowering already included a CTA barrier, so deduplicating the release synchronization was effective; this is a verified contract for that specific toolchain only and cannot be generalized to arbitrary versions, read-complete barriers, or reuse barriers.

---

## Experiment and Acceptance Protocol

### Per-Round Records

Save candidate ID, parent/baseline, status (uncompiled / mechanism not realized / wrong / correct but regressed / partial gain / retained / revoked), hypothesis, single variable, expected ISA, source diff, compilation flags and version/hash, binary and disassembly SHAs separately, input identity and construction method, raw timing, ACU, correctness, resources, conclusion, and next step.

Recompile using an isolated cache and confirm the actual artifact loaded; the same symbol name does not imply the same binary. Post-processed ISA PCs may differ from old DWARF/source line numbers. A single report mixing same-name variants may mismatch disassembly; prefer one tested candidate per report. Historical PCs must be re-mapped.

### Static Acceptance Before Performance Attribution

- Global load addresses, lane mappings, widths, counts, and steady-state order; W/A prefetch must not be inadvertently shortened.
- Shared actual effective addresses, vectorization, slot boundaries, and read/write correspondence; correct allocation metadata alone is not sufficient.
- Unpack/scale/full BF16 fragment/accumulator definitions, last uses, overwrites, SCC, and back-edge relationships.
- MMA count and accumulation order; whether compiler CSE removes computations in synthetic references.
- All dependencies protected by barrier/wait; sources of spill, actual register count, shared, and occupancy limits.

When the target compilation delta is absent, mark "mechanism not realized" and do not use that performance result to dismiss the theoretical mechanism.

### Numerical and Graph

First check nibble order, signed/zero-point, group-32 scale, and BF16 decoded values; then check each MMA fragment and FP32 accumulation; then verify Up+SiLU, Down route weight multiply, and moe_sum full output.

Maintain pre-established numerical contracts; layout/scheduling experiments that preserve semantics should use bit-exact comparison first. When accumulation order changes, audit the error separately with an independent high-precision reference — it cannot be automatically tolerated as "just rounding." The historical three over-tolerance elements were later fixed via a compatible path; that cannot be used as a permanent exemption.

When real samples are available, cover layer0/21/42 × decode1/128/512 (nine groups), and add sparse/dense expert, tail tile, empty expert, M1/7/9/32/65, prefill, and maximum M (historically up to 32768). When samples are missing, note the coverage gap; do not label synthetic or duplicated inputs as real captures.

After NaN contamination, run at least three Graph replays, compare against the same implementation in eager mode and baseline, and re-verify after timing. Check the full output, not just a few sampled points. Numerical pass should also include auditing race conditions and slot errors — there is a historical revoked case where a wrong shared address passed on samples by coincidence.

### Performance and ACU Separated

First measure: same device, inputs, layout, routing, cache conditions, and correct output — a warmed-up Graph without profiler. Exclude compilation and packaging unless the task measures those costs. Alternate candidate/baseline ordering; retain all raw samples, medians, and spread across multiple batches. Differences smaller than noise do not advance; use the same code with different labels (SHAM test) when needed.

Measure separately: Up+SiLU, Down (state whether route weight is fused), joint chain (state whether align and sum are included), and service end-to-end. The joint chain cannot be approximated by adding two independent kernel medians. W8 comparison must account for the actual activation quantization chain; Humming comparison must state the extra epilogue kernel and BF16 rounding.

Sample ACU in a warmed interval bound to the same SHA. Choose parameters from the current version's `acu --help` and locally verified driver; do not blindly run historical commands. Do not modify monitoring/GPM or other services. Export raw metrics and per-PC; trace hot spots to producers, waits, consumers, and loop back-edges.

Distinguish:

- ACU replay duration from no-profiler latency.
- Tensor active, compute throughput, and effective FLOPS/peak.
- DRAM active from DRAM bandwidth saturation; DRAM traffic is not only W.
- Static instruction count from dynamic execution count.
- Stall samples/`per_issue_active.ratio` from percentage, cycles, and microseconds.
- Theoretical occupancy, achieved occupancy, and the ceiling imposed by register configuration.

A hot spot decreasing may run in parallel with another wait, or may be a wait migration. Summing or subtracting ACU samples does not account for time; use same-session timing and controlled ablations to verify causality.

### Achievable Lower Bound and Termination

Model separately: real padded MMA work, effective MACs, logical bytes, and measured traffic. `max(compute lower bound, transfer lower bound)` is only a resource-ideal lower bound; it excludes decoding, synchronization, and issue overhead. Effective peak must come from matching data types and instructions. Do not declare peak achieved from active metrics alone.

Prefer real loop isolation, preserving allocation, consumption, and MMA count. Synthetic references must be checked for CSE, extra instructions, different lifetimes, and cache behavior. Switching to hot weights or copied data is not a free production optimization; it only measures the recoverable portion under that experimental condition. Historical per-round ablations of a few microseconds do not prove no global headroom.

After a first-version failure, proceed: "is the target generated → correctness → new cost added → next isolated item." When the mechanism is confirmed and adjacent scheduling, reasonable resource configurations, and multiple inputs consistently show no stable net gain — record a conditional local boundary. When device is unavailable, permissions are insufficient, or evidence is missing, stop and state the conditions required. Do not use unlimited retries to fulfill a "until theoretical gain" commitment.

---

## Historical Evidence and Artifact Navigation

This skill was compiled on 2026-11-11 from locally available reports without re-running PPU experiments. The numbers below are historical evidence, not current environment commitments; results from different dates/inputs cannot be spliced into an acceleration curve. When files are not packaged with the skill, they may be unavailable — ask the user or rebuild evidence from the current project instead of assuming you have read them.

### Source Materials

Local historical root: `/Users/lvmengsi/Documents/Codex/2026-09-03/d-n/`.

Primary references: `handoff_w4a16_marlin_20261003/W4A16_Marlin_MoE_优化交接文档.md`, in the same directory `全部实验记录索引.md`, `source/source_lock.json`, `manifest.json`, and `reports/`. The handoff summary is current as of 10-02 and explicitly records revocations; it takes precedence over "current best" in earlier reports.

Directional reports:

| Direction | Report under root |
|---|---|
| Initial full pipeline and tile resources | `marlin_full_optimization_0913.md` |
| A/scale conflict isolation | `marlin_layout_optimization_0914.md` |
| B bypass from regression to success | `marlin_register_bypass_iterations_0918.md` |
| Back-edge binding / fragment pipeline | `up_binding_backedge_audit_0920.md`, `up_fragment_pipeline_0920.md` |
| Early per-PC and window | `claude_marlin_optimization_handoff_0920.md` |
| Wrong candidate revocation | handoff package `reports/up_independent_bf16_fragment_0922.md` |
| Source migration / late performance | handoff package `reports/side_up_routes_0924.md` and `evidence/source_migration_1002_results.json` |

Subsequent local experiments: results/role_release JSONs in `/Users/lvmengsi/Documents/Codex/experiments/a_producer_1016/`; manifest/results/isa_diagnosis_1106 JSONs in `/Users/lvmengsi/Documents/Codex/experiments/humming_224_1101/`. Verify the corresponding raw artifacts before citing specific conclusions.

### Key Historical Identities

Retained Up+SiLU AOT SHA as of 10-02: `1e93638ec04bac607753e98dc4b2d2541d2e44eb314dca419595a677985a8396`.

Down dynamic-M mixed-folding AOT SHA: `8c0b5f62b46f91733b4b9fb75991fdd9f9ab08468f020fd971bd5002d2508414`.

Full no-custom-pass Up source candidate SHA: `f9888dc1dd1ae7c62608864ea0a1ce15fe68051c4ba47dc336a2cd549d65e5d9`. It uses Gluon explicit layout and PPU inline asm — not generic tl auto-layout; it did not match AOT performance at the time. Down did not complete an equivalent source migration.

`tail_pub` and Down `pub+release` were revoked from deployment and testing by user request and must not be automatically restored. `r4frontprespread` has a wrong shared address; the ~254.5 μs Up / 412.9 μs chain results are revoked. Adding only shared allocation metadata does not fix it.

### Representative Gains — Not Cumulative Across Rows

| Mechanism | Historical same-round result | Limitation |
|---|---|---|
| Final tile direct nibble decode + config | Up 6.911335→0.485993 ms, Down 3.234002→0.263444 ms | Combined change; not the gain of a single expression |
| B consumer-thread register bypass | Up 345.775→298.224 μs, Down 183.787→157.134 μs | First version ~766 μs; layout fix was required for success |
| Wide packed W/scale + fragmented consumption | Up ~278.6→258.36→257.37 μs | Check per-round comparability; do not treat wide load as the sole cause |
| Down tile-density mixed folding | 154.910→141.390 μs | ~8.73% latency reduction; not all tiles folded |
| Scale high-half PRMT broadcast | Up 212.649→208.626 μs, chain 335.985→331.039 μs | M64-specific capture; chain excludes align |
| No-pass source migration | Same-batch M64 Up 232.383 vs AOT 208.264 μs | Source is reproducible but performance still trails; do not report as a new speedup |

The decode step512 across nine groups is not M512; layer/expert distributions differ, so even M64 timings are not the same across groups. The historical layer0/decode1 value of ~279 μs cannot be directly compared to the layer42/decode512 value of ~209 μs.

### Boundaries for Follow-up External Implementation Comparisons

The Humming data from 11-01 uses upstream v0.1.7 compatibility compilation and config filtering — it is not a PPU deep-optimization build. The Marlin baseline is the optimized AOT described above. Same-shape synthetic data is not equivalent to historical real captures. The 11-06 static ISA audit shows that CUDA fragment→PPU MMA conversions introduce large numbers of shuffles/masked moves; this cannot be used to quantify the total timing gap or judge the format itself.

This skill does not embed SSH credentials, device idle assumptions, or auto-execution commands from historical scripts; connections, dependencies, and device permissions are handled per the current task.
