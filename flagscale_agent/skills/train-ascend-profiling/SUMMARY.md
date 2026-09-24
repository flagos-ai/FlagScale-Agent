<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Train-Ascend-Profiling — Summary

Generate `torch_npu.profiler` wrappers and collect or analyze FlagScale training profiles on Ascend.

**Load when**: generating a profiler wrapper, collecting an NPU trace, or analyzing existing profiles to investigate compute, communication, or Host stalls.

Includes wrapper generation, artifact inventory, and device-window analysis. Reuse `train-run` for collection and return raw evidence, bottleneck hypotheses, and suggested experiments to the tuning workflow. Analysis of existing profiles does not launch training; profiled timings are excluded from performance comparisons.
