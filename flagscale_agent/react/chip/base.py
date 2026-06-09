"""Base data models for chip capabilities.

ChipCapability is a declarative data structure describing:
- What operators are supported (and which have workarounds)
- Precision support (bf16, fp16, fp8, etc.)
- Communication backends (NCCL, HCCL, etc.)
- Known issues and workarounds
- SDK version requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OperatorSupport:
    """Operator availability and coverage."""

    # Core operators (attention, linear, etc.)
    flash_attention: bool = True
    fused_softmax: bool = True
    fused_layernorm: bool = True
    fused_adam: bool = True

    # Operator library (e.g., FlagGems)
    operator_library: str = ""  # "flaggems", "torch_npu", etc.
    coverage_percentage: float = 100.0  # % of PyTorch ops covered

    # Known missing operators
    missing_ops: list[str] = field(default_factory=list)

    # Workarounds for missing ops
    workarounds: dict[str, str] = field(default_factory=dict)


@dataclass
class PrecisionSupport:
    """Precision and mixed-precision support."""

    bf16: bool = True
    fp16: bool = True
    fp8: bool = False
    tf32: bool = False

    # Default precision for this chip
    recommended_precision: Literal["bf16", "fp16", "fp32"] = "bf16"

    # Precision-specific issues
    precision_notes: dict[str, str] = field(default_factory=dict)


@dataclass
class CommunicationBackend:
    """Collective communication backend info."""

    name: str  # "nccl", "hccl", "rccl", etc.
    version: str = ""

    # Supported collective ops
    all_reduce: bool = True
    all_gather: bool = True
    reduce_scatter: bool = True
    broadcast: bool = True

    # Performance notes
    notes: str = ""


@dataclass
class KnownIssue:
    """A known issue with workaround."""

    id: str
    description: str
    severity: Literal["critical", "major", "minor"] = "major"
    workaround: str = ""
    affects: list[str] = field(default_factory=list)  # ["flash_attention", "fp8"]


@dataclass
class ChipCapability:
    """Complete capability declaration for a chip vendor.

    This is a data-driven model — no behavior logic.
    Used by Guards to inject chip-specific constraints.
    """

    # Identity
    vendor: str  # "nvidia", "tianshu", "ascend", etc.
    chip_type: str  # "A100", "T20", "910B", etc.
    sdk_name: str  # "cuda", "tianshu-sdk", "ascend", etc.
    sdk_version: str = ""

    # Capabilities
    operators: OperatorSupport = field(default_factory=OperatorSupport)
    precision: PrecisionSupport = field(default_factory=PrecisionSupport)
    communication: CommunicationBackend = field(default_factory=CommunicationBackend)

    # Issues and constraints
    known_issues: list[KnownIssue] = field(default_factory=list)

    # Constraints to inject into ScenePreset
    auto_constraints: set[str] = field(default_factory=set)

    # FlagOS stack compatibility (optional, for P4-5)
    compatible_flagscale_versions: list[str] = field(default_factory=list)
    compatible_megatron_versions: list[str] = field(default_factory=list)

    def get_issue(self, issue_id: str) -> KnownIssue | None:
        """Lookup a known issue by ID."""
        for issue in self.known_issues:
            if issue.id == issue_id:
                return issue
        return None

    def has_operator(self, op_name: str) -> bool:
        """Check if an operator is supported."""
        return op_name not in self.operators.missing_ops

    def get_workaround(self, op_name: str) -> str | None:
        """Get workaround for a missing operator."""
        return self.operators.workarounds.get(op_name)
