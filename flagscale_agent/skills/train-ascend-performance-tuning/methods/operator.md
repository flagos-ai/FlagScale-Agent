<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Operator and Backend Selection, Integration, and Implementation

## When to use

Use this method when configuration, logs, a profile, or a minimal reproduction supports a hypothesis about a specific operator or backend. Switching an integrated implementation may be enough.
Reuse evidence about the actual call, shape/dtype/stride, relevant rank/stage, and backend. Trace only call boundaries still unclear; do not search every repository.
For interfaces and full-call costs, read `know-ascend-operators`: `ascend_operators/operator-optimization.md` as needed. Read `ascend_operators/kernel-experiments.md` when changing a device kernel.
Organize candidates by target operator. Both configuration selection among integrated implementations and new integration belong here; for graph capture or compilation boundaries, use the [graph and compilation method](graph-execution.md).

## Generate candidates

Start from the public call actually used in training. Choose an operation supported by evidence and identify the configuration field or file/function to change, expected gain, and main cost:

| Direction | Concrete operation |
| --- | --- |
| Integrated backend/fusion | Use a per-operator selection field or fusion switch supported by the current entrypoint. Replace only the target call and required forward/backward coupling. For TE-FL, adjust `train.model.te_fl_per_op` according to a confirmed mapping; name the exact op and currently registered implementation, and leave other entries unchanged. Treat an unintegrated implementation as the next row, not as a reason to alter model definitions such as `swiglu` or norm type for a "fusion gain." |
| Integrate an existing NPU operator | Replace the corresponding implementation in the current adapter/override and map parameters, outputs, and training backward behavior. Integrate TE-FL operators through the corresponding NPU backend registration. Connecting existing RMSNorm `rmsnorm_fwd/rmsnorm_bwd` to `torch_npu.npu_rms_norm/npu_rms_norm_backward` is an adaptation example; first confirm the hot path's current binding and do not integrate a path already active. |
| Remove repeated conversions and temporary tensors | Locate repeated `.contiguous()`, `.to()`, `clone()`, or copy operations inside the public call. Remove, combine, or defer them only when original stride/dtype, aliasing, and backward-saved values permit it. Reuse intermediates only within their valid input lifetime; do not cache changing weights or activations across updates. |
| Reduce host round trips and synchronization | Locate hot-path `.item()`, `.cpu()`, `.tolist()`, explicit synchronization, or per-call logging. Move observation-only work out of the hot path and keep computation on device tensors where supported. Remove synchronization or `empty_cache()` only when dependency evidence shows it is redundant. |
| Fusion and algorithm changes | Replace an identified elementwise/reduction chain, or reorder chain in MoE permutation/unpermutation, with an available fused public interface. Consider a local kernel only when no compatible implementation exists. Preserve stable reductions, indexing/probability semantics, and backward behavior. Include new input preparation, auxiliary tensors, and output restoration in the candidate cost. |
| Triton tiling and task assignment | Change actual target-kernel parameters such as `BLOCK_SIZE` or `BLOCK_M/N/K` and the associated grid/loops. When dispatch overhead dominates small tasks, compare processing multiple tiles with in-kernel strides. If resources exceed limits, shrink tiles or shorten intermediate lifetimes. If needed and supported by the current version, compare a small set of valid `triton.Config` choices through `triton.autotune`, updating the grid for each candidate. |
| Ascend C transfer/compute pipeline | In the target implementation's tiled `CopyIn/Compute/CopyOut` loop, compare single and double buffering. When using `TPipe.InitBuffer(queue, num, len)`, update buffer count, tile length, and loop coverage together. Preserve queue dependencies and tail-tile handling; changing only the buffer count to 2 is insufficient. |

For deeper candidates supported by critical-path evidence, investigate algorithms, cross-operator intermediates, layouts, memory lifetimes, or instruction pipelines. Turn findings into concrete candidates without expanding into unrelated global dispatch rewrites or dependency upgrades.

## Additional checks

- **Switching an integrated implementation:** Confirm that the target public call binds to the candidate and supports forward/backward and required shapes/masks. A configuration flag set to `true` does not prove the candidate took effect.
- **Interface correctness after an implementation change:** Prefer existing public API tests and reference implementations; cover production shapes and boundaries affected by this change. Compare forward and backward for every differentiable input, including affected dtype/device/shape, stride, indices, returns, and fallback behavior. Use project tolerances. A correctness failure excludes performance comparison.
- **Integration activation:** Confirm that the training public call actually binds to the candidate. Successfully calling a vendor/private kernel directly is not equivalent evidence. Reuse the existing integration conclusion when changing only the kernel.
- **Local performance:** Reuse microbenchmarks as needed. Explain kernel and full public-call costs separately; timing must include necessary device completion. Add operator analysis such as `msprof op` only if the reason is unclear. Use the current tool version and keep diagnostic collection separate from performance measurement.
- **Benchmarks with side effects:** Autotuning or repeated microbenchmarks may execute inplace, atomic accumulation, or state updates multiple times. Restore modified buffers to the original state for each candidate/repetition so accumulated inputs do not become the next reference.

If interfaces pass but training shows no gain, explain it using critical-path evidence and added conversion costs. After a functional fix, rebuild the correct baseline through the main workflow. An operator-only task may deliver public-interface and local-performance results while leaving training gain unverified.
