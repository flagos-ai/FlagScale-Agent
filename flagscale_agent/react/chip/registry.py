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

"""Chip registry — maps vendor/chip_type to ChipCapability instances.

To add a new domestic chip, create a file like tianshu.py with a
ChipCapability instance and register it here.
"""

from __future__ import annotations

from flagscale_agent.react.chip.base import ChipCapability
from flagscale_agent.react.chip.nvidia import NVIDIA_A100, NVIDIA_H100


# Registry keyed by (vendor, chip_type)
CHIP_REGISTRY: dict[tuple[str, str], ChipCapability] = {
    ("nvidia", "A100"): NVIDIA_A100,
    ("nvidia", "H100"): NVIDIA_H100,
}

# Vendor default: used when chip_type is unknown but vendor is detected
_VENDOR_DEFAULTS: dict[str, ChipCapability] = {
    "nvidia": NVIDIA_A100,
}


def get_chip(vendor: str, chip_type: str = "") -> ChipCapability | None:
    """Lookup a chip capability by vendor and optional chip_type.

    Falls back to vendor default if chip_type not found.
    Returns None if vendor is unknown.
    """
    key = (vendor.lower(), chip_type.upper()) if chip_type else None
    if key and key in CHIP_REGISTRY:
        return CHIP_REGISTRY[key]
    return _VENDOR_DEFAULTS.get(vendor.lower())
