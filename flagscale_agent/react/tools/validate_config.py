"""Validate FlagScale YAML config structure."""

import os

import yaml

from flagscale_agent.react.tools.base import Tool, EFFECT_READ_FS

# --- Schema definitions ---

_TOP_LEVEL_KEYS = {"defaults", "experiment", "action", "hydra"}

_EXPERIMENT_KEYS = {
    "exp_name", "seed", "save_steps", "load", "exp_dir", "ckpt_format",
    "task", "runner", "cmds", "envs",
}
_EXPERIMENT_TASK_KEYS = {"type", "backend", "entrypoint"}
_EXPERIMENT_RUNNER_KEYS = {
    "per_node_task", "no_shared_fs", "rdzv_backend", "hostfile", "nnodes",
    "nproc_per_node",
}

_MODEL_LEVEL_TOP_KEYS = {"system", "model", "data"}

_SYSTEM_KEYS = {
    "no_shared_fs", "num_workers", "tensor_model_parallel_size",
    "pipeline_model_parallel_size", "context_parallel_size",
    "expert_model_parallel_size", "disable_bias_linear",
    "reset_position_ids", "reset_attention_mask", "qk_layernorm",
    "sequence_parallel", "use_distributed_optimizer",
    "overlap_grad_reduce", "overlap_param_gather", "finetune",
    "precision", "logging", "checkpoint",
    "virtual_pipeline_model_parallel_size",
    "distributed_timeout_minutes",
}
_SYSTEM_PRECISION_KEYS = {
    "bf16", "fp16", "attention_softmax_in_fp32",
    "accumulate_allreduce_grads_in_fp32", "fp32_residual_connection",
}
_SYSTEM_LOGGING_KEYS = {
    "log_interval", "tensorboard_log_interval", "wandb_project",
    "wandb_exp_name", "log_timers_to_tensorboard",
    "log_validation_ppl_to_tensorboard", "log_throughput",
    "log_params_norm", "log_num_zeros_in_grad",
    "log_memory_to_tensorboard",
}
_SYSTEM_CHECKPOINT_KEYS = {"save_interval", "load", "ckpt_format"}

_MODEL_KEYS = {
    "transformer_impl", "num_layers", "hidden_size", "ffn_hidden_size",
    "kv_channels", "num_attention_heads", "group_query_attention",
    "num_query_groups", "seq_length", "max_position_embeddings",
    "norm_epsilon", "use_rotary_position_embeddings", "rotary_base",
    "swiglu", "normalization", "init_method_std", "attention_dropout",
    "hidden_dropout", "clip_grad", "position_embedding_type",
    "untie_embeddings_and_output_weights", "no_position_embedding",
    "no_rope_fusion", "seed", "micro_batch_size", "global_batch_size",
    "eval_iters", "train_iters", "optimizer", "num_experts",
    "moe_router_topk", "moe_grouped_gemm", "moe_aux_loss_coeff",
    "moe_token_dispatcher_type",
}
_MODEL_OPTIMIZER_KEYS = {
    "weight_decay", "adam_beta1", "adam_beta2", "lr_scheduler",
}
_MODEL_LR_SCHEDULER_KEYS = {
    "lr", "min_lr", "lr_warmup_iters", "lr_decay_style",
    "lr_warmup_fraction",
}

_DATA_KEYS = {
    "data_path", "split", "no_mmap_bin_files", "tokenizer",
    "mock", "data_cache_path",
}
_DATA_TOKENIZER_KEYS = {
    "legacy_tokenizer", "tokenizer_type", "tokenizer_path",
    "vocab_size", "make_vocab_size_divisible_by", "tokenizer_model",
}

# Keys that are commonly misplaced
_MISPLACEMENT_RULES = [
    {
        "keys": {"bf16", "fp16"},
        "wrong_parent": "model",
        "correct_location": "system.precision",
    },
    {
        "keys": {"save_interval"},
        "wrong_parent": "model",
        "correct_location": "system.checkpoint",
    },
    {
        "keys": {"tensor_model_parallel_size", "pipeline_model_parallel_size",
                 "context_parallel_size", "expert_model_parallel_size",
                 "sequence_parallel", "use_distributed_optimizer"},
        "wrong_parent": "model",
        "correct_location": "system",
    },
    {
        "keys": {"log_interval", "wandb_project", "wandb_exp_name",
                 "tensorboard_log_interval"},
        "wrong_parent": "model",
        "correct_location": "system.logging",
    },
    {
        "keys": {"data_path", "split", "tokenizer"},
        "wrong_parent": "",
        "correct_location": "data",
    },
]


def _detect_config_type(data: dict) -> str:
    """Detect whether this is a top-level or model-level config."""
    if "experiment" in data:
        return "top_level"
    if "system" in data or "model" in data or "data" in data:
        return "model_level"
    return "unknown"


def _check_misplacement(data: dict, config_type: str) -> list:
    """Check for commonly misplaced keys."""
    errors = []
    if config_type != "model_level":
        return errors

    for rule in _MISPLACEMENT_RULES:
        parent = rule["wrong_parent"]
        section = data.get(parent, {}) if parent else data
        if not isinstance(section, dict):
            continue
        for key in rule["keys"]:
            if key in section:
                errors.append(
                    f"ERROR: '{key}' found under '{parent or 'top-level'}' "
                    f"— should be under '{rule['correct_location']}'"
                )
    return errors


def _validate_top_level(data: dict) -> tuple:
    """Validate a top-level config. Returns (errors, warnings)."""
    errors = []
    warnings = []

    unknown_top = set(data.keys()) - _TOP_LEVEL_KEYS
    if unknown_top:
        warnings.append(f"Unknown top-level keys: {sorted(unknown_top)}")

    if "experiment" not in data:
        errors.append("Missing required 'experiment' section")
        return errors, warnings

    exp = data["experiment"]
    if not isinstance(exp, dict):
        errors.append("'experiment' must be a mapping")
        return errors, warnings

    unknown_exp = set(exp.keys()) - _EXPERIMENT_KEYS
    if unknown_exp:
        warnings.append(f"Unknown keys in 'experiment': {sorted(unknown_exp)}")

    task = exp.get("task")
    if not task or not isinstance(task, dict):
        errors.append("Missing required 'experiment.task' section")
    else:
        if "type" not in task:
            errors.append("Missing required 'experiment.task.type'")
        if "backend" not in task:
            errors.append("Missing required 'experiment.task.backend'")

    if "action" not in data:
        errors.append("Missing required 'action' key")

    return errors, warnings


def _validate_model_level(data: dict) -> tuple:
    """Validate a model-level config. Returns (errors, warnings)."""
    errors = []
    warnings = []

    unknown_top = set(data.keys()) - _MODEL_LEVEL_TOP_KEYS
    if unknown_top:
        warnings.append(f"Unknown top-level keys: {sorted(unknown_top)}")

    # System section
    system = data.get("system")
    if system and isinstance(system, dict):
        unknown = set(system.keys()) - _SYSTEM_KEYS
        if unknown:
            warnings.append(f"Unknown keys in 'system': {sorted(unknown)}")

        precision = system.get("precision")
        if precision and isinstance(precision, dict):
            unknown_p = set(precision.keys()) - _SYSTEM_PRECISION_KEYS
            if unknown_p:
                warnings.append(f"Unknown keys in 'system.precision': {sorted(unknown_p)}")

        logging_sec = system.get("logging")
        if logging_sec and isinstance(logging_sec, dict):
            unknown_l = set(logging_sec.keys()) - _SYSTEM_LOGGING_KEYS
            if unknown_l:
                warnings.append(f"Unknown keys in 'system.logging': {sorted(unknown_l)}")

        ckpt = system.get("checkpoint")
        if ckpt and isinstance(ckpt, dict):
            unknown_c = set(ckpt.keys()) - _SYSTEM_CHECKPOINT_KEYS
            if unknown_c:
                warnings.append(f"Unknown keys in 'system.checkpoint': {sorted(unknown_c)}")

    # Model section
    model = data.get("model")
    if model and isinstance(model, dict):
        unknown = set(model.keys()) - _MODEL_KEYS
        if unknown:
            warnings.append(f"Unknown keys in 'model': {sorted(unknown)}")

        if "num_layers" not in model:
            errors.append("Missing required 'model.num_layers'")
        if "hidden_size" not in model:
            errors.append("Missing required 'model.hidden_size'")

        optimizer = model.get("optimizer")
        if optimizer and isinstance(optimizer, dict):
            unknown_o = set(optimizer.keys()) - _MODEL_OPTIMIZER_KEYS
            if unknown_o:
                warnings.append(f"Unknown keys in 'model.optimizer': {sorted(unknown_o)}")

            lr_sched = optimizer.get("lr_scheduler")
            if lr_sched and isinstance(lr_sched, dict):
                unknown_lr = set(lr_sched.keys()) - _MODEL_LR_SCHEDULER_KEYS
                if unknown_lr:
                    warnings.append(
                        f"Unknown keys in 'model.optimizer.lr_scheduler': {sorted(unknown_lr)}"
                    )
    elif "model" not in data:
        errors.append("Missing required 'model' section")

    # Data section
    data_sec = data.get("data")
    if data_sec and isinstance(data_sec, dict):
        unknown = set(data_sec.keys()) - _DATA_KEYS
        if unknown:
            warnings.append(f"Unknown keys in 'data': {sorted(unknown)}")

        tokenizer = data_sec.get("tokenizer")
        if tokenizer and isinstance(tokenizer, dict):
            unknown_t = set(tokenizer.keys()) - _DATA_TOKENIZER_KEYS
            if unknown_t:
                warnings.append(f"Unknown keys in 'data.tokenizer': {sorted(unknown_t)}")

    return errors, warnings


def validate_config(path: str) -> str:
    """Validate a FlagScale YAML config file. Returns formatted result string."""
    if not os.path.isfile(path):
        return f"ERROR: file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return f"YAML SYNTAX ERROR: {e}"

    if not isinstance(data, dict):
        return "ERROR: config must be a YAML mapping (dict), got " + type(data).__name__

    config_type = _detect_config_type(data)

    if config_type == "top_level":
        errors, warnings = _validate_top_level(data)
    elif config_type == "model_level":
        errors, warnings = _validate_model_level(data)
        errors.extend(_check_misplacement(data, config_type))
    else:
        return (
            "WARNING: Could not determine config type (no 'experiment' or 'system'/'model' key). "
            f"Top-level keys found: {sorted(data.keys())}"
        )

    lines = [f"Config type: {config_type} ({os.path.basename(path)})"]
    if errors:
        lines.append(f"\nERRORS ({len(errors)}):")
        for e in errors:
            lines.append(f"  ✗ {e}")
    if warnings:
        lines.append(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    if not errors and not warnings:
        lines.append("✓ No issues found.")

    status = "ERRORS" if errors else ("WARNINGS" if warnings else "OK")
    lines.insert(0, f"[{status}]")
    return "\n".join(lines)


class ValidateConfigTool(Tool):
    name = "validate_config"
    effects = EFFECT_READ_FS
    description = (
        "Validate a FlagScale YAML config file for structural correctness. "
        "Detects wrong nesting, misplaced keys, and missing required fields. "
        "Use BEFORE launching training to catch config errors early."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the YAML config file to validate.",
            },
        },
        "required": ["path"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        return validate_config(path)
