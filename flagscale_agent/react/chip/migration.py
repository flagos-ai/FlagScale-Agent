"""Cross-chip migration diff — computes differences between source and target chips.

Used by the agent to provide migration guidance when porting models
from one chip vendor to another (e.g., NVIDIA → Tianshu).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flagscale_agent.react.chip.base import ChipCapability


@dataclass
class MigrationItem:
    """A single migration action item."""

    category: str  # "operator", "precision", "communication", "config"
    description: str
    severity: str  # "critical", "major", "minor"
    action: str  # What to do about it
    source_value: str = ""
    target_value: str = ""


@dataclass
class MigrationDiff:
    """Complete migration diff from source chip to target chip."""

    source_vendor: str
    source_chip: str
    target_vendor: str
    target_chip: str
    items: list[MigrationItem] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.items if i.severity == "critical")

    @property
    def major_count(self) -> int:
        return sum(1 for i in self.items if i.severity == "major")

    def summary(self) -> str:
        """Human-readable summary of migration requirements."""
        lines = [
            f"Migration: {self.source_vendor}/{self.source_chip}"
            f" → {self.target_vendor}/{self.target_chip}",
            f"  Items: {len(self.items)} "
            f"(critical={self.critical_count}, major={self.major_count})",
        ]
        for item in self.items:
            lines.append(f"  [{item.severity}] {item.category}: {item.description}")
            lines.append(f"    Action: {item.action}")
        return "\n".join(lines)


def compute_migration_diff(
    source: ChipCapability, target: ChipCapability
) -> MigrationDiff:
    """Compute migration diff from source chip to target chip.

    Identifies operators, precision, and communication differences
    that require action when porting a model.
    """
    diff = MigrationDiff(
        source_vendor=source.vendor,
        source_chip=source.chip_type,
        target_vendor=target.vendor,
        target_chip=target.chip_type,
    )

    # Operator differences
    _diff_operators(source, target, diff)
    # Precision differences
    _diff_precision(source, target, diff)
    # Communication differences
    _diff_communication(source, target, diff)

    return diff


def _diff_operators(
    source: ChipCapability, target: ChipCapability, diff: MigrationDiff
) -> None:
    """Find operator gaps between source and target."""
    # Check operators available on source but missing on target
    source_ops = {
        "flash_attention": source.operators.flash_attention,
        "fused_softmax": source.operators.fused_softmax,
        "fused_layernorm": source.operators.fused_layernorm,
        "fused_adam": source.operators.fused_adam,
    }
    target_ops = {
        "flash_attention": target.operators.flash_attention,
        "fused_softmax": target.operators.fused_softmax,
        "fused_layernorm": target.operators.fused_layernorm,
        "fused_adam": target.operators.fused_adam,
    }

    for op_name, available_on_source in source_ops.items():
        if available_on_source and not target_ops.get(op_name, False):
            workaround = target.operators.workarounds.get(op_name, "")
            severity = "critical" if not workaround else "major"
            diff.items.append(MigrationItem(
                category="operator",
                description=f"{op_name} available on source but missing on target",
                severity=severity,
                action=workaround or f"Find alternative for {op_name}",
                source_value="supported",
                target_value="missing",
            ))

    # Operator library coverage gap
    if target.operators.coverage_percentage < source.operators.coverage_percentage:
        gap = source.operators.coverage_percentage - target.operators.coverage_percentage
        diff.items.append(MigrationItem(
            category="operator",
            description=(
                f"Operator coverage gap: {target.operators.coverage_percentage}%"
                f" vs source {source.operators.coverage_percentage}%"
            ),
            severity="major" if gap > 10 else "minor",
            action=(
                f"Install {target.operators.operator_library} for best coverage; "
                f"remaining ops fall back to eager mode"
            ),
            source_value=f"{source.operators.coverage_percentage}%",
            target_value=f"{target.operators.coverage_percentage}%",
        ))


def _diff_precision(
    source: ChipCapability, target: ChipCapability, diff: MigrationDiff
) -> None:
    """Find precision support differences."""
    precisions = ["bf16", "fp16", "fp8", "tf32"]
    for prec in precisions:
        src_val = getattr(source.precision, prec)
        tgt_val = getattr(target.precision, prec)
        if src_val and not tgt_val:
            diff.items.append(MigrationItem(
                category="precision",
                description=f"{prec} supported on source but not on target",
                severity="major",
                action=(
                    f"Switch to {target.precision.recommended_precision}; "
                    f"{target.precision.precision_notes.get(prec, '')}"
                ),
                source_value="supported",
                target_value="not supported",
            ))

    # Recommended precision change
    if source.precision.recommended_precision != target.precision.recommended_precision:
        diff.items.append(MigrationItem(
            category="precision",
            description=(
                f"Recommended precision differs: "
                f"{source.precision.recommended_precision}"
                f" → {target.precision.recommended_precision}"
            ),
            severity="minor",
            action=(
                f"Update --precision flag to"
                f" {target.precision.recommended_precision}"
            ),
            source_value=source.precision.recommended_precision,
            target_value=target.precision.recommended_precision,
        ))


def _diff_communication(
    source: ChipCapability, target: ChipCapability, diff: MigrationDiff
) -> None:
    """Find communication backend differences."""
    if source.communication.name != target.communication.name:
        diff.items.append(MigrationItem(
            category="communication",
            description=(
                f"Communication backend change: "
                f"{source.communication.name} → {target.communication.name}"
            ),
            severity="major",
            action=(
                f"Ensure {target.communication.name} is installed and configured; "
                f"{target.communication.notes}"
            ),
            source_value=source.communication.name,
            target_value=target.communication.name,
        ))
