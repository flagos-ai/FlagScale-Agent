"""inspect_checkpoint tool — deep checkpoint inspection with shape/dtype/numerical verification."""

import os
import re

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_FS


def _safe_torch_load(path: str):
    """Load a PyTorch checkpoint safely — try weights_only first, fall back."""
    import logging as _logging
    _log = _logging.getLogger(__name__)
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        _log.warning("weights_only=True failed for %s, falling back to weights_only=False. "
                     "This is needed for checkpoints with custom objects.", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_state_dict(path: str):
    """Load state_dict from a checkpoint file."""
    import torch
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        sd = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    data = _safe_torch_load(path)
    if isinstance(data, dict):
        for k in ("model", "state_dict", "module"):
            if k in data:
                return data[k]
        if all(isinstance(v, torch.Tensor) for v in list(data.values())[:5]):
            return data
    return None


def inspect_checkpoint(path, reference_path="", expected_keys="", sample_count=20):
    """Core inspection logic. Returns formatted report string."""
    import random
    import torch

    if not os.path.isfile(path):
        return f"ERROR: File not found: {path}"

    try:
        state_dict = _load_state_dict(path)
    except Exception as e:
        return f"ERROR: Failed to load checkpoint: {type(e).__name__}: {e}"

    if state_dict is None:
        data = _safe_torch_load(path)
        keys = list(data.keys())[:20] if isinstance(data, dict) else type(data).__name__
        return f"ERROR: Cannot find state_dict. Top-level keys: {keys}"

    out = []
    out.append(f"## Checkpoint: {os.path.basename(path)}")
    out.append(f"Keys: {len(state_dict)} | Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")

    # Dtype distribution
    dtype_counts = {}
    anomalies = []
    for key, tensor in state_dict.items():
        dt = str(tensor.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        if tensor.numel() == 0:
            anomalies.append(f"  EMPTY: {key} shape={tuple(tensor.shape)}")
        elif tensor.is_floating_point():
            if torch.isnan(tensor).any():
                anomalies.append(f"  NaN: {key} shape={tuple(tensor.shape)}")
            elif torch.isinf(tensor).any():
                anomalies.append(f"  Inf: {key} shape={tuple(tensor.shape)}")
            elif tensor.abs().max().item() == 0.0:
                anomalies.append(f"  ALL-ZERO: {key} shape={tuple(tensor.shape)}")

    out.append("\nDtypes: " + ", ".join(f"{dt}={c}" for dt, c in sorted(dtype_counts.items(), key=lambda x: -x[1])))

    if anomalies:
        out.append(f"\nANOMALIES ({len(anomalies)}):")
        for a in anomalies[:15]:
            out.append(a)
        if len(anomalies) > 15:
            out.append(f"  ... +{len(anomalies) - 15} more")
    else:
        out.append("\nAnomalies: None")

    # Sample statistics
    keys_list = list(state_dict.keys())
    n = min(sample_count, len(keys_list))
    sample_keys = keys_list[:n] if len(keys_list) <= sample_count else random.sample(keys_list, n)
    out.append(f"\nSample ({n} tensors):")
    for key in sorted(sample_keys):
        t = state_dict[key]
        if t.is_floating_point() and t.numel() > 0:
            tf = t.float()
            out.append(
                f"  {key}: {tuple(t.shape)} {t.dtype} "
                f"mean={tf.mean().item():.6f} std={tf.std().item():.6f}"
            )
        else:
            out.append(f"  {key}: {tuple(t.shape)} {t.dtype}")

    # Expected keys check
    if expected_keys:
        out.append("\nExpected keys:")
        for pat in (p.strip() for p in expected_keys.split(",")):
            matches = [k for k in state_dict if re.search(pat, k)]
            out.append(f"  {'OK' if matches else 'MISSING'}: '{pat}' → {len(matches)} keys")

    # Reference comparison
    if reference_path and os.path.isfile(reference_path):
        out.append(f"\nReference comparison vs {os.path.basename(reference_path)}:")
        try:
            ref_dict = _load_state_dict(reference_path)
            if ref_dict is not None:
                ref_keys = set(ref_dict)
                cur_keys = set(state_dict)
                only_ref = ref_keys - cur_keys
                only_cur = cur_keys - ref_keys
                common = ref_keys & cur_keys
                out.append(f"  Ref={len(ref_dict)} Cur={len(state_dict)} Common={len(common)}")
                if only_ref:
                    out.append(f"  Only in ref ({len(only_ref)}):")
                    for k in sorted(only_ref)[:8]:
                        out.append(f"    - {k}")
                if only_cur:
                    out.append(f"  Only in cur ({len(only_cur)}):")
                    for k in sorted(only_cur)[:8]:
                        out.append(f"    + {k}")
                mismatches = []
                for k in sorted(common):
                    r, c = ref_dict[k], state_dict[k]
                    if r.shape != c.shape:
                        mismatches.append(f"  SHAPE: {k} ref={tuple(r.shape)} cur={tuple(c.shape)}")
                    elif r.dtype != c.dtype:
                        mismatches.append(f"  DTYPE: {k} ref={r.dtype} cur={c.dtype}")
                if mismatches:
                    out.append(f"  Mismatches ({len(mismatches)}):")
                    for m in mismatches[:10]:
                        out.append(m)
                else:
                    out.append(f"  All {len(common)} common keys match shape and dtype")
        except Exception as e:
            out.append(f"  ERROR: {e}")

    # Verdict
    issues = len(anomalies)
    if expected_keys:
        issues += sum(1 for p in expected_keys.split(",")
                      if not any(re.search(p.strip(), k) for k in state_dict))
    out.append(f"\nVerdict: {'ISSUES FOUND — fix before training' if issues else '[OK] checkpoint valid'}")

    return "\n".join(out)


class InspectCheckpointTool(Tool):
    name = "inspect_checkpoint"
    effects = EFFECT_READ_FS
    description = (
        "Deep inspection of a PyTorch checkpoint (.pt/.bin/.safetensors). "
        "Reports key count, shape/dtype summary, detects anomalies (all-zero, NaN/Inf), "
        "samples tensor statistics, and optionally compares against a reference checkpoint. "
        "Use after checkpoint conversion to catch mapping errors before training."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to checkpoint file to inspect.",
            },
            "reference_path": {
                "type": "string",
                "description": "Optional: reference checkpoint to compare shapes/dtypes against.",
            },
            "expected_keys": {
                "type": "string",
                "description": "Optional: comma-separated key patterns (regex) to verify exist.",
            },
            "sample_count": {
                "type": "integer",
                "description": "Number of tensors to sample for statistics. Default: 20.",
            },
        },
        "required": ["path"],
    }

    def execute(self, **kwargs) -> str:
        return inspect_checkpoint(
            path=kwargs["path"],
            reference_path=kwargs.get("reference_path", ""),
            expected_keys=kwargs.get("expected_keys", ""),
            sample_count=kwargs.get("sample_count", 20),
        )
