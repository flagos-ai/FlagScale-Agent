"""Tests for inspect_checkpoint tool."""

import os
import tempfile

import pytest
import torch

from flagscale_agent.react.tools.inspect_checkpoint import (
    InspectCheckpointTool,
    inspect_checkpoint,
)


@pytest.fixture
def tool():
    return InspectCheckpointTool()


@pytest.fixture
def good_ckpt(tmp_path):
    """Create a valid checkpoint with normal tensors."""
    sd = {
        "model.layer1.weight": torch.randn(128, 64),
        "model.layer1.bias": torch.randn(128),
        "model.layer2.weight": torch.randn(256, 128),
        "model.layer2.bias": torch.randn(256),
        "model.embed.weight": torch.randn(1000, 64),
    }
    path = str(tmp_path / "good.pt")
    torch.save({"model": sd}, path)
    return path


@pytest.fixture
def bad_ckpt(tmp_path):
    """Create a checkpoint with anomalies."""
    sd = {
        "layer1.weight": torch.randn(128, 64),
        "layer2.weight": torch.zeros(256, 128),  # all-zero
        "layer3.weight": torch.tensor([float('nan'), 1.0, 2.0]),  # NaN
        "layer4.weight": torch.tensor([float('inf'), 1.0]),  # Inf
    }
    path = str(tmp_path / "bad.pt")
    torch.save(sd, path)
    return path


class TestBasicInspection:
    def test_good_checkpoint(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt)
        assert "Keys: 5" in result
        assert "Anomalies: None" in result
        assert "[OK]" in result

    def test_detects_all_zero(self, bad_ckpt):
        result = inspect_checkpoint(bad_ckpt)
        assert "ALL-ZERO" in result
        assert "layer2.weight" in result

    def test_detects_nan(self, bad_ckpt):
        result = inspect_checkpoint(bad_ckpt)
        assert "NaN" in result
        assert "layer3.weight" in result

    def test_detects_inf(self, bad_ckpt):
        result = inspect_checkpoint(bad_ckpt)
        assert "Inf" in result
        assert "layer4.weight" in result

    def test_reports_dtype(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt)
        assert "torch.float32" in result

    def test_file_not_found(self):
        result = inspect_checkpoint("/nonexistent/path.pt")
        assert "ERROR" in result
        assert "not found" in result

    def test_sample_statistics(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt, sample_count=3)
        assert "Sample (3 tensors)" in result
        assert "mean=" in result
        assert "std=" in result


class TestExpectedKeys:
    def test_expected_keys_found(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt, expected_keys="layer1,layer2")
        assert "OK: 'layer1'" in result
        assert "OK: 'layer2'" in result

    def test_expected_keys_missing(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt, expected_keys="layer1,nonexistent")
        assert "OK: 'layer1'" in result
        assert "MISSING: 'nonexistent'" in result

    def test_regex_patterns(self, good_ckpt):
        result = inspect_checkpoint(good_ckpt, expected_keys="layer.*weight,embed")
        assert "OK: 'layer.*weight'" in result
        assert "OK: 'embed'" in result


class TestReferenceComparison:
    def test_matching_reference(self, tmp_path):
        sd = {"w1": torch.randn(10, 10), "w2": torch.randn(5)}
        p1 = str(tmp_path / "a.pt")
        p2 = str(tmp_path / "b.pt")
        torch.save(sd, p1)
        torch.save(sd, p2)
        result = inspect_checkpoint(p1, reference_path=p2)
        assert "match shape and dtype" in result

    def test_shape_mismatch(self, tmp_path):
        p1 = str(tmp_path / "a.pt")
        p2 = str(tmp_path / "b.pt")
        torch.save({"w": torch.randn(10, 10)}, p1)
        torch.save({"w": torch.randn(10, 20)}, p2)
        result = inspect_checkpoint(p1, reference_path=p2)
        assert "SHAPE" in result

    def test_missing_keys_in_reference(self, tmp_path):
        p1 = str(tmp_path / "a.pt")
        p2 = str(tmp_path / "b.pt")
        torch.save({"w1": torch.randn(10)}, p1)
        torch.save({"w1": torch.randn(10), "w2": torch.randn(5)}, p2)
        result = inspect_checkpoint(p1, reference_path=p2)
        assert "Only in ref" in result
        assert "w2" in result


class TestNestedCheckpoint:
    def test_model_key(self, tmp_path):
        path = str(tmp_path / "nested.pt")
        torch.save({"model": {"a": torch.randn(5)}, "optimizer": {}}, path)
        result = inspect_checkpoint(path)
        assert "Keys: 1" in result

    def test_state_dict_key(self, tmp_path):
        path = str(tmp_path / "nested2.pt")
        torch.save({"state_dict": {"a": torch.randn(5), "b": torch.randn(3)}}, path)
        result = inspect_checkpoint(path)
        assert "Keys: 2" in result


class TestToolInterface:
    def test_execute(self, tool, good_ckpt):
        result = tool.execute(path=good_ckpt)
        assert "Keys: 5" in result
        assert "[OK]" in result

    def test_schema(self, tool):
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "inspect_checkpoint"
        assert "path" in schema["function"]["parameters"]["properties"]
