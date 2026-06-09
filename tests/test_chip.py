"""Tests for Phase 4 chip capability system."""

import os
import pytest
from unittest.mock import patch

from flagscale_agent.react.chip.base import (
    ChipCapability,
    OperatorSupport,
    PrecisionSupport,
    CommunicationBackend,
    KnownIssue,
)
from flagscale_agent.react.chip.registry import CHIP_REGISTRY, get_chip
from flagscale_agent.react.chip.detect import detect_chip
from flagscale_agent.react.chip.nvidia import NVIDIA_A100, NVIDIA_H100
from flagscale_agent.react.chip.migration import (
    MigrationDiff,
    MigrationItem,
    compute_migration_diff,
)


# ── ChipCapability data model ─────────────────────────────────────────


class TestChipCapability:
    def test_nvidia_a100_identity(self):
        assert NVIDIA_A100.vendor == "nvidia"
        assert NVIDIA_A100.chip_type == "A100"
        assert NVIDIA_A100.sdk_name == "cuda"

    def test_nvidia_h100_has_fp8(self):
        assert NVIDIA_H100.precision.fp8 is True
        assert NVIDIA_A100.precision.fp8 is False

    def test_has_operator(self):
        assert NVIDIA_A100.has_operator("flash_attention_v2") is True

    def test_get_issue_none_for_nvidia(self):
        assert NVIDIA_A100.get_issue("nonexistent") is None

    def test_auto_constraints_empty_for_nvidia(self):
        assert len(NVIDIA_A100.auto_constraints) == 0


# ── Registry ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_chip_exact_match(self):
        chip = get_chip("nvidia", "A100")
        assert chip is NVIDIA_A100

    def test_get_chip_case_insensitive_vendor(self):
        chip = get_chip("NVIDIA", "H100")
        assert chip is NVIDIA_H100

    def test_get_chip_vendor_default(self):
        chip = get_chip("nvidia")
        assert chip is NVIDIA_A100

    def test_get_chip_unknown_vendor(self):
        chip = get_chip("unknown_vendor", "X1")
        assert chip is None

    def test_registry_has_nvidia_chips(self):
        assert len(CHIP_REGISTRY) == 2
        assert ("nvidia", "A100") in CHIP_REGISTRY
        assert ("nvidia", "H100") in CHIP_REGISTRY


# ── Detection ─────────────────────────────────────────────────────────


class TestDetection:
    def test_env_override(self):
        with patch.dict(os.environ, {
            "FLAGSCALE_CHIP_VENDOR": "nvidia",
            "FLAGSCALE_CHIP_TYPE": "H100",
        }):
            chip = detect_chip()
            assert chip is not None
            assert chip.vendor == "nvidia"
            assert chip.chip_type == "H100"

    def test_env_override_vendor_only(self):
        with patch.dict(os.environ, {
            "FLAGSCALE_CHIP_VENDOR": "nvidia",
        }, clear=False):
            env = os.environ.copy()
            env.pop("FLAGSCALE_CHIP_TYPE", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"FLAGSCALE_CHIP_VENDOR": "nvidia"}):
                    chip = detect_chip()
                    assert chip is not None
                    assert chip.vendor == "nvidia"

    def test_env_override_unknown_vendor(self):
        with patch.dict(os.environ, {
            "FLAGSCALE_CHIP_VENDOR": "unknown",
        }):
            with patch(
                "flagscale_agent.react.chip.detect._detect_nvidia", return_value=None
            ):
                chip = detect_chip()
                assert chip is None


# ── Migration Diff ────────────────────────────────────────────────────


class TestMigrationDiff:
    def test_same_chip_no_diff(self):
        diff = compute_migration_diff(NVIDIA_A100, NVIDIA_A100)
        assert len(diff.items) == 0

    def test_h100_to_a100_fp8_gap(self):
        diff = compute_migration_diff(NVIDIA_H100, NVIDIA_A100)
        prec_items = [i for i in diff.items if i.category == "precision"]
        descriptions = " ".join(i.description for i in prec_items)
        assert "fp8" in descriptions

    def test_summary_format(self):
        diff = compute_migration_diff(NVIDIA_H100, NVIDIA_A100)
        summary = diff.summary()
        assert "nvidia/H100" in summary
        assert "nvidia/A100" in summary

    def test_migration_item_fields(self):
        item = MigrationItem(
            category="operator",
            description="test",
            severity="critical",
            action="fix it",
            source_value="yes",
            target_value="no",
        )
        assert item.category == "operator"
        assert item.severity == "critical"

    def test_critical_count(self):
        diff = MigrationDiff(
            source_vendor="a", source_chip="X",
            target_vendor="b", target_chip="Y",
            items=[
                MigrationItem("op", "d1", "critical", "a1"),
                MigrationItem("op", "d2", "major", "a2"),
                MigrationItem("op", "d3", "critical", "a3"),
            ],
        )
        assert diff.critical_count == 2
        assert diff.major_count == 1
