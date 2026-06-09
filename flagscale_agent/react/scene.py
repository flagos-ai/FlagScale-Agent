"""Scene preset — parameterizes WorkerAgent behavior by scenario.

ScenePreset replaces free-form SceneContext dataclass construction.
Users select a preset (or auto-detect), then optionally override fields.

Key design: constraints set 是机器可消费的标记:
- WorkerProfile.scene_constraints 声明"我在这些 constraint 下才激活"
- Interrupt.activate_on 声明"我在这些 constraint 下才生效"
- Checklist 根据 constraints 决定激活哪些检查项
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
    chip_type: str  # "nvidia" | "ascend" | "kunlun" | "dcu" | "mthreads"
    chip_vendor_sdk: str  # "cuda" | "ascend" | "kunlunxin" | "rocm"

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
        """Detect scene from environment only (no regex).

        Uses keyword-based matching on user_input for mode/hints.
        Full intent classification (migration vs training vs inference,
        multi-node) should be done by Judge; this is a lightweight fallback.
        """
        # ── Chip type (env only) ──
        chip_type = "nvidia"
        chip_vendor_sdk = "cuda"
        if os.environ.get("ASCEND_HOME"):
            chip_type = "ascend"
            chip_vendor_sdk = "ascend"
        elif os.environ.get("ROCM_PATH"):
            chip_type = "dcu"
            chip_vendor_sdk = "rocm"

        # ── Mode hints (keyword, not regex) ──
        constraints: set[str] = set()
        mode = "training"
        text_lower = user_input.lower()

        inference_keywords = ["inference", "serving", "vllm", "sglang", "推理", "部署"]
        migration_keywords = ["migrate", "port", "porting", "from ", "迁移"]
        multi_node_keywords = ["multi-node", "multi_node", "集群", "cluster", "slurm"]
        rl_keywords = ["rl", "reinforcement", "ppo", "grpo", "reward", "强化学习"]

        if any(k in text_lower for k in inference_keywords):
            mode = "inference_serving"
            constraints.add("is_inference")
        else:
            constraints.add("is_training")

        if any(k in text_lower for k in migration_keywords):
            constraints.add("is_migration")
            if chip_type != "nvidia":
                constraints.add("is_chip_migration")

        if any(k in text_lower for k in multi_node_keywords):
            constraints.add("requires_multi_node")
            network_topology = "multi_node_ib"
        else:
            network_topology = "single_node"

        if any(k in text_lower for k in rl_keywords):
            constraints.add("is_rl")

        # ── Source framework hints ──
        source = ""
        if "megatron" in text_lower and any(k in text_lower for k in ["from ", "迁移", "migrate", "原来是"]):
            source = "megatron"
        elif "deepspeed" in text_lower:
            source = "deepspeed"
        elif "fsdp" in text_lower:
            source = "fsdp"
        elif any(k in text_lower for k in ["vllm", "vLLM"]):
            source = "vllm"

        # ── Target ──
        target = "megatron-core"
        if mode == "inference_serving":
            target = "flagscale+vllm"

        # ── Precision ──
        precision = "bf16" if chip_type == "nvidia" else "fp16"

        # ── Name ──
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


# ── Preset library ────────────────────────────────────────────────────────

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
    "megatron-training-ascend": ScenePreset(
        name="megatron-training-ascend",
        mode="training",
        chip_type="ascend",
        chip_vendor_sdk="ascend",
        target_framework="megatron-core",
        source_framework="",
        default_precision="fp16",
        network_topology="single_node",
        constraints={"is_training", "is_chip_migration", "flash_attn_no_ascend"},
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
