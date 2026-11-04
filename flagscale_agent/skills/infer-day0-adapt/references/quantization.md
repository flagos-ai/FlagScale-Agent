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

# Quantization integration contracts

Source: **推理框架插件开发规范v1**, H1–H10, together with its applicable
A–G clauses and the [operator contracts](operator-contracts.md). Read when
using quantized checkpoints or changing weight, activation, or KV quantization.
Unchanged reuse requires affected-path validation, not a second implementation.
Requirements below are mandatory unless explicitly marked **should**.

## Define format and layer coverage (H1, H5)

Record checkpoint source/revision, algorithm, storage encoding, and execution
backend separately. AWQ/GPTQ support or a Marlin registration does not establish
that every checkpoint or FL device works. Describe weight/activation bits and
dtype, weight-only versus joint quantization, static/dynamic activation scales,
per-tensor/channel/token/group/block granularity, and actual Linear/MoE/Attention/
Embedding/LM Head coverage. Different INT4/FP4/FP8 encodings are not interchangeable.

Specify scale versus inverse scale, zero-point semantics, group/block axes and
sizes, packing order, rounding/saturation, accumulation/output dtype, bias
placement, shape/stride/alignment/padding constraints. Select the backend before
execution using real device, format, shape, and parallel-layout capabilities.
An upstream NVIDIA capability threshold is insufficient for other devices.
Fallback must preserve the same format/precision contract and happen before
any computation or side effect; execution failures propagate without retry.

## Keep configuration and computation separate (H1–H3)

The plugin owns quantization parsing/registration, per-layer methods, parameter
declarations/loading, and lifecycle orchestration. Quantize/dequantize, repack,
GEMM, MoE, and KV encoding/decoding numerics belong in operator libraries,
including numerical preprocessing during initialization. New/replaced numerical
backends use the existing call bridge and unified dispatch; a model `forward`
must not introduce private quantization/vendor selection branches.

For vLLM, inspect the pinned version's `QuantizationConfig`,
`get_quantization_config`, `register_quantization_config`, and
`get_quant_method(layer, prefix)` extension points and signatures. SGLang uses
its own current equivalents; do not transplant vLLM APIs. Limit necessary
registration overrides and test repeated initialization.

For SGLang discovery, trace a supported model's quantized layer construction
back to the configuration registry/parser, layer-method selection, weight
loader/postprocessing, and dispatch bridge in the pinned checkout. Record the
actual symbols and signatures at those extension points before implementing
the new format or layer; do not infer API equivalence from similar names.

Preserve precedence and consistency among explicit user settings, checkpoint
`quantization_config`/separate files, per-layer rules, and platform capabilities.
Reject conflicts and unknown required fields before loading/execution; do not
rename a method or drop semantic fields to disguise incompatible encoding.

Apply full layer prefixes, ignored/excluded layers, dynamic overrides, and
packed-module mappings to both weights and selection rules. Distinguish an
inapplicable method, an explicitly unquantized layer, and an unsupported layer
that must fail. Preserve each fused QKV/gate-up member's parameters; handle rule
conflicts according to upstream semantics or reject them. Verify shared weights
and tied Embedding/LM Head only for methods that actually support them.

## Prepare and consume parameters exactly once (H4, H6)

For vLLM, align each layer base class's `create_weights`,
`process_weights_after_loading`, and `apply` interfaces with the pinned version.
Declare shape, dtype, shard/packing dimensions, and weight-loader metadata.
Load packed weights, scales, zero-points, and required ordering/index metadata.
Respect device/meta-device and preprocessed-weight capabilities. Complete
repack/layout/scale conversion after all required inputs arrive and before
first inference or graph capture; prevent repeated conversion on reinitialization.
`apply` consumes prepared parameters without repeated whole-weight conversion
per token. Missing required parameters fail explicitly; format-specific defaults
or ignored redundant fields need tests. Requantization requires independent
source dequantization and error validation, not reinterpreting packed storage.

Distinguish serialized quantized checkpoints, online quantization of floating
weights at load time, and dynamic activation quantization at inference time.
Specify input formats, timing, scale sources, and parameter lifetimes and test
each claimed entry separately. Budget peak loading memory, floating copies,
temporaries, and persistent parameters. If calibration is needed, record data
revision, preprocessing/sampling, and tool versions; keep calibration and
acceptance datasets separate. Mark calibration-free methods explicitly.

## KV cache is an independent capability (H7)

Specify cache storage dtype, attention compute dtype, granularity, and read/write
backend independently of weight/activation quantization. Trace creation, name
mapping, loading, processing, and consumption of K/V and any Q/probability scales.
Validate static scale shape/value/sharding; store and consume dynamic scales
with the correct token/head/block. Follow the format's missing-scale contract:
neither unconditional scale=1 nor forcing dynamic formats to static is valid.

Validate affected, claimed prefill/decode, chunked prefill, prefix reuse, paged
cache, and long-context paths. Check slot/block mappings, invalid regions,
concurrent isolation, and the shared lifetime of scales/cache. The framework
owns metadata, budgeting, workspace, and stream ordering; numerical cache
conversion belongs in operators. Graph claims require real capture/replay.

## Parallelism and MoE (H8)

Check weights, scales, zero-points, and index sharding/replication for TP/EP,
row/column parallelism, group/block boundaries, padding, and reordered local
matrices. Reject unsupported layouts at initialization. Verify gate/up/down and
per-expert parameters, routing/top-k/shared experts, routing-weight application
order, and collectives. Expert reordering/load balancing must move quantization
parameters with weights. Dense Linear success does not prove FusedMoE support.

## Evidence and delivery (H9, H10)

For new methods/numerical changes, test configuration/layer selection, parameter
loading/postprocessing, independent operator numerics, and a real quantized model.
For wiring-only changes, test the affected chain and actual method/backend hits.
Separate implementation error against the **same quantized semantics** from
model quality degradation against a **high-precision baseline**. Use fixed data
and inference settings with justified tolerances/quality thresholds (such as
perplexity or task metrics); identical generated text is not the sole criterion.

Cover affected bits/dtypes/granularities, static/dynamic modes, nonaligned
dimensions, fused/mixed layers, zeros/extremes, shards, and cache/state updates.
Test missing scales, illegal groups/blocks, conflicting config, unsupported
devices, and execution-time failure at their correct stages. In graph tests,
update inputs/scales, reset state, sentinel-fill independent output buffers, and
compare from equal initial conditions so old scale/output/cache cannot pass.

Report checkpoint/tool/config revisions, framework/operator commits, device,
parallel and graph modes, actual backend, failures, and unverified scope. The
support matrix separates loadable, numerically verified, and end-to-end quality
verified states for each format, layer family, weight/activation scheme, and KV
scheme. Deliver operator APIs/tests before plugin integration and a public fixed
dependency. Update maintained user docs/examples for new support or substantive
usage changes; transient measurements belong in PR/CI artifacts.

For performance optimization, **should** assess both end-to-end benefit and
loading cost with fixed model, input/output lengths, batch/concurrency,
parallelism, and service constraints. Report TTFT, TPOT/ITL, throughput, peak and
steady memory, and load time, stating whether quantization/repack/compilation/
warmup are counted. Bit-width ratios and isolated kernel speedups are not
measured model-level benefits.
