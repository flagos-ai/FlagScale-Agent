# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
