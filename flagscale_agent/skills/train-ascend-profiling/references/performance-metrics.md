<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Performance Metrics: FLOPs, Throughput and MFU

## When this applies

You have a step time from a log (or a profiler) and want to know how much of the
device's theoretical compute it actually used. `scripts/mfu_calculator.py` turns an
architecture description plus that step time into FLOPs, throughput and, if you
supply a peak, MFU.

## What the number means

```
MFU = modelled training FLOPs per device per second / device peak FLOPS
```

Both the numerator and the denominator are assumptions:

- **Numerator** comes from a FLOPs model, not a measurement. `with_flops` in the
  profiler does not produce a measured hardware FLOPS figure, so it cannot be used
  to compute MFU.
- **Denominator** must be the peak for the exact device *and* the exact precision
  actually used. A BF16 peak does not apply to an FP8 run.

The tool refuses to emit MFU without a peak, and marks the result `implausible`
when modelled FLOPs exceed the supplied peak (a sign the model, the peak, the
device count or the step time disagree).

## The FLOPs model in this tool

Per token, per layer, forward:

| Term | Count |
| --- | --- |
| Q projection | `2*hidden*hidden` |
| K, V projections | `2*hidden*kv_dim` each, where `kv_dim = num_key_value_heads * head_dim` |
| Output projection | `2*hidden*hidden` |
| Attention scores + weights×V | `2*seq*hidden + 2*seq*kv_dim` |
| Dense FFN | `gates * 2*hidden*intermediate`, `gates=3` for SwiGLU/GeGLU, else 2 |
| MoE | `2*hidden*num_experts` (router) `+ topk * gates * 2*hidden*expert_intermediate` |

Then `training_flops = 3 * (per_token * num_layers + 2*hidden*vocab) * seq * gbs`,
where the factor 3 is forward plus a backward pass costing twice the forward.

Known omissions, all deliberate: bias/activation/normalisation elementwise work,
communication, embedding-table lookups, and the optimiser's own arithmetic.

## Cross-check against `6*N*D`

The usual rule of thumb is `6*N*D` with `N` the non-embedding parameter count and
`D` the tokens trained. On a canonical dense configuration the explicit sum above
lands ~3.4% above `6*N*D`; the excess is the attention term that scales with
sequence length plus the embedding/logits term, which `6*N*D` omits on purpose.
Agreement in the low single digits is the expected result — a large gap means the
architecture fields or the FFN gate count are wrong, not that `6*N*D` is.

Run the check with:

```bash
python "$SKILL_DIR/scripts/mfu_calculator.py" self-test
```

## Reporting rules

- Quote the peak figure with its source (vendor spec, exact device SKU, precision).
- State that FLOPs are modelled, not measured, whenever MFU is reported.
- Never compare MFU across precisions or device SKUs without saying so.
- For MoE, distinguish activated parameters from total parameters; a total-parameter
  MFU flatters the run.
- A low MFU is a starting question, not a diagnosis: gather a profile before
  naming a cause.
