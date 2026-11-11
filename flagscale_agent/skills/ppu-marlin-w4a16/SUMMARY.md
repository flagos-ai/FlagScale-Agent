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

# PPU Marlin W4A16 Optimization — Summary

Optimize or audit W4A16 Marlin GEMM / fused MoE kernels on PPU, covering Up+SiLU, Down, and joint-chain decode/prefill performance analysis and experiment design.

**Load when**: You need to analyze or improve PPU W4A16 Marlin MoE performance — including B register bypass, shared bank conflict swizzle, packed W/scale prefetch, MMA thread layout, register lifetimes, or migrating mechanisms to FlagGems/Triton source. Not for generic CUDA tuning or non-PPU devices.

**Workflow** — Read-only confirm entry/SHA/toolchain identity → clarify A/B workload and quantization format → run candidates in an isolated cache (keep baseline) → audit in order math/layout → ISA → resources → lifetimes → report each round whether the mechanism compiled, correctness, same-session timing, and ACU attribution → close the direction when the mechanism holds but stable net gain is zero.

**Key constraints** — Historical AOT SHAs (Up `1e93638e`, Down `8c0b5f62`) and revoked candidates (`tail_pub`, `pub+release`, `r4frontprespread`) must not be automatically restored. Performance data is only valid for same-round same-caliper comparisons and cannot be accumulated across rows. Each round must state a falsifiable hypothesis and pass static acceptance before looking at performance; after numerical pass, also audit race conditions and slot errors.
