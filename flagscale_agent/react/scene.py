"""Scene preset — parameterizes agent behavior by scenario.

ScenePreset replaces free-form SceneContext dataclass construction.
Users select a preset (or auto-detect), then optionally override fields.

Key design: constraints set is machine-consumable tags:
- WorkerProfile.scene_constraints declares "I activate under these constraints"
- Interrupt.activate_on declares "I activate under these constraints"
- Checklist uses constraints to decide which checks to activate
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ScenePreset:
    """A named preset that bundles typical scene parameters."""

    name: str  # "megatron-training-nvidia"
    mode: str  # "training" | "inference_serving" | "inference_engine"

    # Hardware
    chip_type: str  # "nvidia"
    chip_vendor_sdk: str  # "cuda"

    # Framework
    target_framework: str  # "megatron-core" | "flagscale+vllm" | "flagscale+sglang"
    source_framework: str  # "" = not migrating

    # Precision
    default_precision: str  # "bf16" | "fp16" | "fp8"

    # Network
    network_topology: str  # "single_node" | "multi_node_ib" | "multi_node_roce"

    # Constraints — machine-consumable tags parameterizing behavior
    constraints: set[str] = field(default_factory=set)

    @classmethod
    def from_env_and_input(cls, user_input: str = "") -> "ScenePreset":
        """Detect scene from environment and user input.

        Uses keyword-based matching on user_input for mode/hints.
        Full intent classification (migration vs training vs inference,
        multi-node) should be done by Judge; this is a lightweight fallback.
        """
        # Chip type (env only)
        chip_type = "nvidia"
        chip_vendor_sdk = "cuda"

        # Mode hints (keyword-based)
        constraints: set[str] = set()
        mode = "training"
        text_lower = user_input.lower()

        inference_keywords = ["inference", "serving", "vllm", "sglang"]
        migration_keywords = ["migrate", "port", "porting", "from "]
        multi_node_keywords = ["multi-node", "multi_node", "cluster", "slurm"]
        rl_keywords = ["rl", "reinforcement", "ppo", "grpo", "reward"]

        if any(k in text_lower for k in inference_keywords):
            mode = "inference_serving"
            constraints.add("is_inference")
        else:
            constraints.add("is_training")

        if any(k in text_lower for k in migration_keywords):
            constraints.add("is_migration")

        if any(k in text_lower for k in multi_node_keywords):
            constraints.add("requires_multi_node")
            network_topology = "multi_node_ib"
        else:
            network_topology = "single_node"

        if any(k in text_lower for k in rl_keywords):
            constraints.add("is_rl")

        # Source framework hints
        source = ""
        if "megatron" in text_lower and any(k in text_lower for k in ["from ", "migrate"]):
            source = "megatron"
        elif "deepspeed" in text_lower:
            source = "deepspeed"
        elif "fsdp" in text_lower:
            source = "fsdp"
        elif any(k in text_lower for k in ["vllm", "vLLM"]):
            source = "vllm"

        # Target
        target = "megatron-core"
        if mode == "inference_serving":
            target = "flagscale+vllm"

        # Precision
        precision = "bf16"

        # Name
        name = f"{target.split('+')[0]}-{mode}-{chip_type}"
        if source:
            name += f"-from-{source}"

        return cls(
            name=name,
            mode=mode,
            chip_type=chip_type,
            chip_vendor_sdk=chip_vendor_sdk,
            target_framework=target,
            source_framework=source,
            default_precision=precision,
            network_topology=network_topology,
            constraints=constraints,
        )

    @classmethod
    def auto_detect(cls, cwd: str | None = None, user_input: str = "") -> "ScenePreset":
        """Backward-compatible alias for from_env_and_input."""
        return cls.from_env_and_input(user_input=user_input)


# Preset library

PRESETS: dict[str, ScenePreset] = {
    "megatron-training-nvidia": ScenePreset(
        name="megatron-training-nvidia",
        mode="training",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="megatron-core",
        source_framework="",
        default_precision="bf16",
        network_topology="single_node",
        constraints={"is_training"},
    ),
    "vllm-inference-nvidia": ScenePreset(
        name="vllm-inference-nvidia",
        mode="inference_serving",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="flagscale+vllm",
        source_framework="",
        default_precision="fp16",
        network_topology="single_node",
        constraints={"is_inference"},
    ),
    "megatron-migration-deepspeed-nvidia": ScenePreset(
        name="megatron-migration-deepspeed-nvidia",
        mode="training",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="megatron-core",
        source_framework="deepspeed",
        default_precision="bf16",
        network_topology="single_node",
        constraints={"is_training", "is_migration"},
    ),
}

