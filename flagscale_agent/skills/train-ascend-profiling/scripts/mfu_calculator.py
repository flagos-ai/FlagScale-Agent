#!/usr/bin/env python3
"""Compute training FLOPs, throughput and MFU from a model/training description.

Standard library only; this tool does no measurement of its own. It converts an
explicit architecture description plus an observed step time into FLOPs and, only
when a hardware peak is supplied, an MFU figure.
"""
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Independently implemented; no upstream code copied. Method notes:
# references/performance-metrics.md

import argparse
import json
import math

SCHEMA_VERSION = 1


def _pos(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _layered(spec, name, default=None):
    value = spec.get(name, default)
    if value is None:
        raise ValueError(f"missing required architecture field: {name}")
    return value


def layer_flops_per_token(spec, seq_len):
    """Forward FLOPs for one transformer layer, for a single token.

    Terms are counted explicitly so the reported number is auditable:
    QKV/output projections, FFN (dense or routed experts), and the attention
    score/AV products that scale with sequence length.
    """
    hidden = _pos(_layered(spec, "hidden_size"), "hidden_size")
    heads = _pos(_layered(spec, "num_attention_heads"), "num_attention_heads")
    head_dim = spec.get("head_dim")
    head_dim = _pos(head_dim, "head_dim") if head_dim else hidden / heads
    kv_heads = _pos(spec.get("num_key_value_heads", heads), "num_key_value_heads")
    kv_dim = kv_heads * head_dim

    if kv_heads > heads:
        raise ValueError("num_key_value_heads cannot exceed num_attention_heads")

    # Q, K, V and output projections (2 * in * out multiply-adds each).
    proj = 2 * hidden * hidden + 2 * 2 * hidden * kv_dim + 2 * hidden * hidden

    # Attention products: scores (S x S) and weights x V, both per token.
    attention = 2 * seq_len * hidden + 2 * seq_len * kv_dim

    moe = bool(spec.get("is_moe", False))
    if moe:
        experts = _pos(spec.get("num_experts"), "num_experts")
        topk = _pos(spec.get("num_experts_per_tok", 1), "num_experts_per_tok")
        if topk > experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        expert_hidden = _pos(spec.get("expert_intermediate_size")
                             or spec.get("moe_intermediate_size"), "expert_intermediate_size")
        # Router over all experts, then the selected experts only.
        router = 2 * hidden * experts
        ffn = router + 3 * 2 * hidden * expert_hidden * topk
        shared = spec.get("shared_expert_intermediate_size")
        if shared:
            ffn += 3 * 2 * hidden * _pos(shared, "shared_expert_intermediate_size")
    else:
        intermediate = _pos(_layered(spec, "intermediate_size"), "intermediate_size")
        gates = 3 if str(spec.get("ffn_type", "swiglu")).lower() in ("swiglu", "geglu") else 2
        ffn = gates * 2 * hidden * intermediate

    return proj + attention + ffn


def step_flops(spec, seq_len, global_batch_size):
    """Training FLOPs for one optimizer step (forward plus backward)."""
    layers = _pos(_layered(spec, "num_layers"), "num_layers")
    hidden = _pos(_layered(spec, "hidden_size"), "hidden_size")
    vocab = _pos(_layered(spec, "vocab_size"), "vocab_size")

    per_token = layers * layer_flops_per_token(spec, seq_len)
    logits = 2 * hidden * vocab
    forward = (per_token + logits) * seq_len * global_batch_size
    # Backward costs roughly twice the forward pass.
    return {"forward_flops": forward, "training_flops": 3 * forward}


def _self_test():
    """Cross-check the explicit layer sum against the usual 6*N*D estimate.

    N is the parameter count implied by the same architecture description; the
    two routes are computed independently, so a mismatch means the FLOPs terms
    or the parameter terms are wrong.
    """
    spec = {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32,
            "head_dim": 128, "num_key_value_heads": 32, "intermediate_size": 11008,
            "vocab_size": 32000, "ffn_type": "swiglu"}
    seq_len, gbs = 2048, 512
    hidden = spec["hidden_size"]
    kv = spec["num_key_value_heads"] * spec["head_dim"]
    params_per_layer = (2 * hidden * hidden + 2 * hidden * kv
                        + 3 * hidden * spec["intermediate_size"])
    params = params_per_layer * spec["num_layers"] + 2 * hidden * spec["vocab_size"]
    tokens = seq_len * gbs

    explicit = step_flops(spec, seq_len, gbs)
    approx = 6 * params * tokens
    # Attention's S-dependent term is not part of 6*N*D, so exclude it here.
    attention = 2 * spec["num_layers"] * (2 * seq_len * hidden) * tokens
    explicit_linear = explicit["training_flops"] - attention

    deviation = abs(explicit_linear - approx) / approx
    print(json.dumps({
        "schema_version": SCHEMA_VERSION, "command": "self-test",
        "parameters": params, "tokens_per_step": tokens,
        "training_flops_explicit": explicit["training_flops"],
        "training_flops_6nd": approx,
        "attention_quadratic_flops": attention,
        "relative_deviation_linear_terms": deviation,
        "status": "ok" if deviation < 0.05 else "mismatch",
    }, indent=2))
    return 0 if deviation < 0.05 else 1


def summarize(spec, seq_len, global_batch_size, num_devices, step_time_s,
              peak_flops=None, peak_source=None, phase="training"):
    if phase not in ("training", "inference"):
        raise ValueError("phase must be 'training' or 'inference'")
    flops = step_flops(spec, seq_len, global_batch_size)
    tokens = seq_len * global_batch_size
    if phase == "inference":
        # No backward pass.
        flops["training_flops"] = flops["forward_flops"]

    result = {"schema_version": SCHEMA_VERSION, "command": "mfu", "phase": phase,
              "architecture": {"hidden_size": spec.get("hidden_size"),
                               "num_layers": spec.get("num_layers"),
                               "num_attention_heads": spec.get("num_attention_heads"),
                               "num_key_value_heads": spec.get("num_key_value_heads"),
                               "is_moe": bool(spec.get("is_moe", False))},
              "workload": {"seq_length": seq_len, "global_batch_size": global_batch_size,
                           "num_devices": num_devices, "tokens_per_step": tokens,
                           "step_time_s": step_time_s},
              "flops": flops, "mfu": None,
              "assumptions": [
                  "Backward pass costs 2x the forward pass.",
                  "Attention score/AV terms scale with sequence length and are included.",
                  "No padding, masking or dropped-token adjustment is applied.",
                  "MoE FLOPs count router plus the selected experts only.",
              ],
              "limitations": []}

    achieved = flops["training_flops"] / (step_time_s * num_devices)
    result["achieved_flops_per_device"] = achieved
    result["tokens_per_s_per_device"] = tokens / (step_time_s * num_devices)

    if peak_flops is None:
        result["status"] = "flops_only_no_peak"
        result["limitations"].append(
            "No hardware peak supplied, so MFU is not computed. Supply the FP16/BF16 "
            "peak for the exact device and precision actually used.")
        return result

    peak = _pos(peak_flops, "peak_flops")
    result["peak_flops_per_device"] = peak
    result["peak_source"] = peak_source
    result["mfu"] = achieved / peak
    result["status"] = "ok"
    result["interpretation"] = (
        "MFU compares the modelled FLOPs against a theoretical peak. It is only as "
        "good as the FLOPs model and the peak figure; both must match the actual "
        "model and dtype.")
    if achieved > peak:
        result["status"] = "implausible"
        result["limitations"].append(
            "Achieved FLOPs exceed the supplied peak: the FLOPs model, the peak "
            "figure, the device count or the step time is inconsistent. Do not "
            "report this as a utilization result.")
    if result["peak_source"] is None:
        result["limitations"].append(
            "Peak figure was not attributed to a source; cite the vendor spec for "
            "the exact device and precision before reporting.")
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="cross-check the FLOPs model against 6*N*D")

    run = commands.add_parser("mfu", help="compute FLOPs, throughput and MFU")
    run.add_argument("--config", help="JSON file with architecture fields")
    for name in ("hidden-size", "num-layers", "num-attention-heads", "intermediate-size",
                 "vocab-size"):
        run.add_argument("--" + name, type=float)
    run.add_argument("--num-key-value-heads", type=float)
    run.add_argument("--head-dim", type=float)
    run.add_argument("--ffn-type", default="swiglu")
    run.add_argument("--is-moe", action="store_true")
    run.add_argument("--num-experts", type=float)
    run.add_argument("--num-experts-per-tok", type=float)
    run.add_argument("--expert-intermediate-size", type=float)
    run.add_argument("--shared-expert-intermediate-size", type=float)
    run.add_argument("--seq-length", type=float, required=True)
    run.add_argument("--global-batch-size", type=float, required=True)
    run.add_argument("--num-devices", type=float, required=True)
    run.add_argument("--step-time", type=float, required=True,
                     help="observed seconds per optimizer step")
    run.add_argument("--peak-flops", type=float,
                     help="FP16/BF16 peak FLOPS for one device; omit to skip MFU")
    run.add_argument("--peak-source", help="where the peak figure came from")
    run.add_argument("--phase", default="training", choices=("training", "inference"))

    try:
        args = vars(parser.parse_args())
        command = args.pop("command")
        if command == "self-test":
            return _self_test()
        config = {}
        if args.get("config"):
            with open(args["config"], "r", encoding="utf-8") as stream:
                config = json.load(stream)
            if not isinstance(config, dict):
                raise ValueError("--config must contain a JSON object")
        mapping = {"hidden_size": "hidden_size", "num_layers": "num_layers",
                   "num_attention_heads": "num_attention_heads",
                   "num_key_value_heads": "num_key_value_heads", "head_dim": "head_dim",
                   "intermediate_size": "intermediate_size", "vocab_size": "vocab_size",
                   "ffn_type": "ffn_type", "num_experts": "num_experts",
                   "num_experts_per_tok": "num_experts_per_tok",
                   "expert_intermediate_size": "expert_intermediate_size",
                   "shared_expert_intermediate_size": "shared_expert_intermediate_size"}
        for source, target in mapping.items():
            value = args.get(source)
            if value is not None:
                config[target] = value
        config["is_moe"] = bool(args.get("is_moe")) or bool(config.get("is_moe"))
        result = summarize(config, _pos(args["seq_length"], "seq-length"),
                           _pos(args["global_batch_size"], "global-batch-size"),
                           _pos(args["num_devices"], "num-devices"),
                           _pos(args["step_time"], "step-time"),
                           args.get("peak_flops"), args.get("peak_source"),
                           args.get("phase") or "training")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if result["status"] in ("ok", "flops_only_no_peak") else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error",
                          "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
