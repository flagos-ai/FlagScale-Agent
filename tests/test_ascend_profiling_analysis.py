# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Ascend profiling analysis helpers: MFU model, op analysis,
and slow-rank comparison."""

import csv
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / "flagscale_agent" / "skills" / "train-ascend-profiling" / "scripts"
mfu = import_module("flagscale_agent.skills.train-ascend-profiling.scripts.mfu_calculator")
op_analysis = import_module("flagscale_agent.skills.train-ascend-profiling.scripts.op_analysis")
slow_rank = import_module("flagscale_agent.skills.train-ascend-profiling.scripts.slow_rank")


class TestMfuCalculator:
    def test_self_test_agrees_with_6nd(self):
        assert mfu._self_test() == 0

    def test_dense_flops_forward_backward_ratio(self):
        spec = {"hidden_size": 512, "num_layers": 4, "num_attention_heads": 8,
                "intermediate_size": 1024, "vocab_size": 1000, "ffn_type": "swiglu"}
        result = mfu.step_flops(spec, seq_len=128, global_batch_size=8)
        assert result["training_flops"] == pytest.approx(3 * result["forward_flops"])

    def test_gqa_reduces_flops(self):
        base = {"hidden_size": 512, "num_layers": 2, "num_attention_heads": 8,
                "intermediate_size": 1024, "vocab_size": 1000}
        mha = mfu.step_flops({**base, "num_key_value_heads": 8}, 128, 1)
        gqa = mfu.step_flops({**base, "num_key_value_heads": 2}, 128, 1)
        assert gqa["forward_flops"] < mha["forward_flops"]

    def test_moe_counts_only_active_experts(self):
        base = {"hidden_size": 512, "num_layers": 2, "num_attention_heads": 8,
                "vocab_size": 1000, "is_moe": True, "num_experts": 8,
                "num_experts_per_tok": 2, "expert_intermediate_size": 512}
        active2 = mfu.step_flops(base, 128, 1)
        active4 = mfu.step_flops({**base, "num_experts_per_tok": 4}, 128, 1)
        assert active4["forward_flops"] > active2["forward_flops"]
        # 8 experts of full size would be far larger than top-2 routing.
        dense_equivalent = mfu.step_flops(
            {**base, "is_moe": False, "intermediate_size": 8 * 512}, 128, 1)
        assert active2["forward_flops"] < dense_equivalent["forward_flops"]

    def test_mfu_withheld_without_peak(self):
        spec = {"hidden_size": 512, "num_layers": 2, "num_attention_heads": 8,
                "intermediate_size": 1024, "vocab_size": 1000}
        result = mfu.summarize(spec, 128, 4, 8, 1.0)
        assert result["status"] == "flops_only_no_peak"
        assert result["mfu"] is None
        assert result["achieved_flops_per_device"] > 0

    def test_implausible_result_flagged(self):
        spec = {"hidden_size": 512, "num_layers": 4, "num_attention_heads": 8,
                "intermediate_size": 1024, "vocab_size": 1000}
        result = mfu.summarize(spec, 128, 4, 8, 1e-9, peak_flops=1e12)
        assert result["status"] == "implausible"
        assert any("Do not report" in note for note in result["limitations"])

    def test_mfu_plausible(self):
        spec = {"hidden_size": 512, "num_layers": 4, "num_attention_heads": 8,
                "intermediate_size": 1024, "vocab_size": 1000}
        result = mfu.summarize(spec, 128, 4, 8, 1.0, peak_flops=1e14,
                               peak_source="vendor spec")
        assert result["status"] == "ok"
        assert 0 < result["mfu"] < 1

    def test_malformed_spec_rejected(self):
        with pytest.raises(ValueError):
            mfu.step_flops({"hidden_size": 512}, 128, 1)
        with pytest.raises(ValueError):
            mfu.step_flops({"hidden_size": 512, "num_layers": 2,
                            "num_attention_heads": 8, "num_key_value_heads": 16,
                            "intermediate_size": 64, "vocab_size": 10}, 128, 1)


@pytest.fixture()
def prof_dir(tmp_path):
    """A synthetic MindStudio-style output tree with quoted CSV fields."""
    prof = tmp_path / "PROF_0001" / "mindstudio_profiler_output"
    prof.mkdir(parents=True)
    header = ["Type", "OP Name", "Duration(us)", "Input Shapes", "Input Data Types",
              "aic_mac_ratio", "aiv_vec_ratio"]
    rows = [
        ["MatMul", "mm_fast", "500", "4096,4096;4096,4096", "bf16,bf16,bf16", "0.92", "0.03"],
        ["MatMul", "mm_slow", "1200", "4096,4096;4096,4096", "bf16,bf16,bf16", "0.88", "0.06"],
        ["MatMul", "mm_kmis", "900", "1024,3072;2048,1024", "fp16,fp16,fp16", "0.85", "0.05"],
        ["GroupedMatmul", "gm", "2000", "8x1024,3072;3072,1024", "bf16", "0.91", "0.02"],
        ["RmsNorm", "norm", "300", "4096,4096", "bf16", "0.40", "0.50"],
        ["Add", "add", "50", "4096,4096", "bf16", "", ""],
    ]
    with (prof / "kernel_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "OP Name", "Duration(us)", "Input Shapes",
                         "Input Data Types", "aic_mac_ratio", "aiv_vec_ratio"])
        writer.writerows(rows)
    with (prof / "op_statistic.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["OP Type", "Total Duration(us)", "Count"],
                                 ["MatMul", "2600", "3"], ["RmsNorm", "300", "1"]])
    return prof


class TestOpAnalysis:
    def test_hot_ops_ranks_and_shares(self, prof_dir):
        result = op_analysis.hot_ops(prof_dir, top=2)
        assert result["status"] == "ok"
        # Grouped by operator: MatMul 500+1200+900=2600 is the top entry.
        assert result["top"][0]["op"] == "MatMul"
        assert result["top"][0]["duration_sum_us"] == 2600.0
        assert result["top"][0]["calls"] == 3
        assert result["top"][0]["distinct_shapes"] == 2
        assert 0 < result["top"][0]["share_of_scanned_duration"] < 1
        assert result["top"][1]["op"] == "GroupedMatmul"

    def test_shapes_parses_and_flags_k_mismatch(self, prof_dir):
        result = op_analysis.parse_shapes(prof_dir, op="matmul")
        by_shape = {item["input_shapes"]: item for item in result["parsed"]}
        assert by_shape["4096,4096;4096,4096"]["k_consistent"] is True
        assert by_shape["1024,3072;2048,1024"]["k_consistent"] is False
        assert result["unparsed_rows"] == 1  # GroupedMatmul 3D form

    def test_roofline_flags_slow_kernel(self, prof_dir):
        path = prof_dir / "kernel_details.csv"
        result = op_analysis.roofline(path, peak_flops=300, bandwidth_gbs=1000)
        same_shape = [item for item in result["checked"]
                      if item["input_shapes"] == "4096,4096;4096,4096"]
        assert len(same_shape) == 2
        fast = min(same_shape, key=lambda item: item["observed_us"])
        slow = max(same_shape, key=lambda item: item["observed_us"])
        # 2*4096**3 / 300e12 = 458us theoretical; the 500us kernel is near it.
        assert fast["theoretical_us"] == pytest.approx(458.1, abs=1.0)
        assert fast["efficiency"] == pytest.approx(0.916, abs=0.01)
        assert slow["efficiency"] < 0.5

    def test_roofline_reports_unmodelled_never_estimates(self, prof_dir):
        result = op_analysis.roofline(prof_dir / "kernel_details.csv", 300, 1000)
        reasons = {item["op"] for item in result["unmodelled"]}
        assert "GroupedMatmul" in reasons
        assert all(item["reason"] for item in result["unmodelled"])

    def test_roofline_skips_below_mac_threshold(self, prof_dir):
        result = op_analysis.roofline(prof_dir / "kernel_details.csv", 300, 1000,
                                      mac_threshold=0.80)
        assert all(item["efficiency"] for item in result["checked"])
        assert not any(item["op"] == "RmsNorm" for item in result["checked"])

    def test_missing_ratio_columns_status(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("Type,Duration(us)\nMatMul,10\n", encoding="utf-8")
        result = op_analysis.roofline(empty, 300, 1000)
        assert result["status"] == "unsupported_schema"


def slow_efficiencies(rows):
    return [row["efficiency"] for row in rows]


class TestSlowRank:
    def _write_logs(self, root, layout):
        for rank, values in layout.items():
            directory = root / f"rank_{rank}"
            directory.mkdir(parents=True)
            lines = []
            for index, ms in enumerate(values, 1):
                lines.append(f"iteration     {index}/  40 | elapsed time per iteration (ms): {ms}")
            (directory / "stdout.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_outlier_rank_is_candidate(self, tmp_path):
        root = tmp_path / "host0"
        self._write_logs(root, {0: [100] * 10, 1: [101] * 10, 2: [99] * 10,
                                7: [4000] + [800] * 9})
        result = slow_rank.compare(slow_rank.from_logs(root), min_delta_pct=10)
        assert result["ranks_compared"] == 4
        assert [c["rank"] for c in result["candidates"]] == ["7"]
        assert result["candidates"][0]["delta_vs_median_pct"] > 100

    def test_uniform_slowdown_yields_no_candidates(self):
        root = Path("/nonexistent-for-test")  # never touched in csv mode
        ranks = {"0": {"rank": "0", "step_ms": [200.0] * 10, "comm_lines": 0, "files": ["x"]},
                 "1": {"rank": "1", "step_ms": [201.0] * 10, "comm_lines": 0, "files": ["y"]},
                 "2": {"rank": "2", "step_ms": [199.0] * 10, "comm_lines": 0, "files": ["z"]}}
        result = slow_rank.compare(ranks, min_delta_pct=10)
        assert result["candidates"] == []
        assert "uniform slowdown" in result["note"]

    def test_balance_is_not_proven(self):
        ranks = {"0": {"rank": "0", "step_ms": [100.0] * 5, "comm_lines": 0, "files": []},
                 "1": {"rank": "1", "step_ms": [100.0] * 5, "comm_lines": 0, "files": []}}
        result = slow_rank.compare(ranks)
        assert result["candidates"] == []
        assert result["spread"]["max_over_min"] == pytest.approx(1.0)

    def test_single_rank_is_insufficient(self):
        ranks = {"0": {"rank": "0", "step_ms": [100.0] * 5, "comm_lines": 0, "files": []}}
        assert slow_rank.compare(ranks)["status"] == "insufficient_ranks"

    def test_csv_mode(self, tmp_path):
        path = tmp_path / "ranks.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "step_ms"])
            for rank, ms in [(0, 100), (1, 100), (2, 100), (9, 500)]:
                writer.writerow([rank, ms])
        result = slow_rank.compare(slow_rank.from_csv(path, "rank", "step_ms"))
        assert result["status"] == "ok"
        assert result["candidates"][0]["rank"] == "9"
