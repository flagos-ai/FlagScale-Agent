"""NVIDIA chip capability declarations.

Provides baseline reference capabilities for NVIDIA GPUs.
"""

from flagscale_agent.react.chip.base import (
    ChipCapability,
    OperatorSupport,
    PrecisionSupport,
    CommunicationBackend,
    KnownIssue,
)


NVIDIA_A100 = ChipCapability(
    vendor="nvidia",
    chip_type="A100",
    sdk_name="cuda",
    sdk_version="12.1+",
    operators=OperatorSupport(
        flash_attention=True,
        fused_softmax=True,
        fused_layernorm=True,
        fused_adam=True,
        operator_library="native",
        coverage_percentage=100.0,
        missing_ops=[],
        workarounds={},
    ),
    precision=PrecisionSupport(
        bf16=True,
        fp16=True,
        fp8=False,  # Requires H100+
        tf32=True,
        recommended_precision="bf16",
        precision_notes={
            "bf16": "Native support, recommended for training",
            "fp16": "Native support, may require loss scaling",
            "tf32": "Enabled by default for matmul on Ampere+",
        },
    ),
    communication=CommunicationBackend(
        name="nccl",
        version="2.18+",
        all_reduce=True,
        all_gather=True,
        reduce_scatter=True,
        broadcast=True,
        notes="NCCL is the standard for NVIDIA multi-GPU communication",
    ),
    known_issues=[],
    auto_constraints=set(),
    compatible_flagscale_versions=["0.4.0+"],
    compatible_megatron_versions=["4.0.0+"],
)


NVIDIA_H100 = ChipCapability(
    vendor="nvidia",
    chip_type="H100",
    sdk_name="cuda",
    sdk_version="12.1+",
    operators=OperatorSupport(
        flash_attention=True,
        fused_softmax=True,
        fused_layernorm=True,
        fused_adam=True,
        operator_library="native",
        coverage_percentage=100.0,
        missing_ops=[],
        workarounds={},
    ),
    precision=PrecisionSupport(
        bf16=True,
        fp16=True,
        fp8=True,  # H100 supports FP8
        tf32=True,
        recommended_precision="bf16",
        precision_notes={
            "bf16": "Native support, recommended for training",
            "fp16": "Native support, may require loss scaling",
            "fp8": "Transformer Engine support for H100",
            "tf32": "Enabled by default for matmul on Hopper",
        },
    ),
    communication=CommunicationBackend(
        name="nccl",
        version="2.18+",
        all_reduce=True,
        all_gather=True,
        reduce_scatter=True,
        broadcast=True,
        notes="NCCL with NVLink 4.0 for high-bandwidth communication",
    ),
    known_issues=[],
    auto_constraints=set(),
    compatible_flagscale_versions=["0.4.0+"],
    compatible_megatron_versions=["4.0.0+"],
)
