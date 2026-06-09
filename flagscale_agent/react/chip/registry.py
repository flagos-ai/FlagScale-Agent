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
