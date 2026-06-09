"""Chip capability system — hardware-aware constraints and migration support.

Phase 4: Provides structured chip capability data for:
- Constraint injection (auto-activate guards based on detected chip)
- Cross-chip migration diff (operator/precision/communication differences)
- FlagOS stack compatibility queries

Design:
- ChipCapability is a data-driven model (no behavior logic)
- Each chip vendor provides a YAML-like declaration of capabilities
- detect_chip() probes the environment and returns the matching capability
- MigrationDiff computes source→target differences for agent guidance
"""

from flagscale_agent.react.chip.base import (
    ChipCapability,
    OperatorSupport,
    PrecisionSupport,
    CommunicationBackend,
    KnownIssue,
)
from flagscale_agent.react.chip.detect import detect_chip
from flagscale_agent.react.chip.registry import CHIP_REGISTRY, get_chip
from flagscale_agent.react.chip.migration import (
    MigrationDiff,
    MigrationItem,
    compute_migration_diff,
)

__all__ = [
    "ChipCapability",
    "OperatorSupport",
    "PrecisionSupport",
    "CommunicationBackend",
    "KnownIssue",
    "detect_chip",
    "CHIP_REGISTRY",
    "get_chip",
    "MigrationDiff",
    "MigrationItem",
    "compute_migration_diff",
]
