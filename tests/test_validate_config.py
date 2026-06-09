"""Tests for validate_config tool."""

import os
import tempfile

import pytest

from flagscale_agent.react.tools.validate_config import (
    ValidateConfigTool,
    validate_config,
    _detect_config_type,
    _check_misplacement,
)


@pytest.fixture
def tool():
    return ValidateConfigTool()


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write a temp YAML file and return its path."""
    def _write(content: str) -> str:
        p = tmp_path / "test_config.yaml"
        p.write_text(content)
        return str(p)
    return _write


class TestDetectConfigType:
    def test_top_level(self):
        assert _detect_config_type({"experiment": {}, "action": "run"}) == "top_level"

    def test_model_level(self):
        assert _detect_config_type({"system": {}, "model": {}, "data": {}}) == "model_level"

    def test_unknown(self):
        assert _detect_config_type({"foo": "bar"}) == "unknown"


class TestValidateTopLevel:
    def test_valid_config(self, tmp_yaml):
        content = """
defaults:
  - train: 0_6b
  - _self_
experiment:
  exp_name: test
  task:
    type: train
    backend: megatron
    entrypoint: train.py
  runner:
    nnodes: 1
action: run
hydra:
  run:
    dir: ./hydra
"""
        result = validate_config(tmp_yaml(content))
        assert "[OK]" in result
        assert "No issues found" in result

    def test_missing_experiment(self, tmp_yaml):
        content = "action: run\nhydra: {}\n"
        result = validate_config(tmp_yaml(content))
        assert "WARNING" in result
        assert "Could not determine" in result

    def test_missing_task(self, tmp_yaml):
        content = """
experiment:
  exp_name: test
action: run
"""
        result = validate_config(tmp_yaml(content))
        assert "Missing required 'experiment.task'" in result

    def test_missing_action(self, tmp_yaml):
        content = """
experiment:
  exp_name: test
  task:
    type: train
    backend: megatron
"""
        result = validate_config(tmp_yaml(content))
        assert "Missing required 'action'" in result

    def test_unknown_top_keys(self, tmp_yaml):
        content = """
experiment:
  exp_name: test
  task:
    type: train
    backend: megatron
action: run
extra_key: value
"""
        result = validate_config(tmp_yaml(content))
        assert "WARNINGS" in result
        assert "extra_key" in result


class TestValidateModelLevel:
    def test_valid_config(self, tmp_yaml):
        content = """
system:
  tensor_model_parallel_size: 2
  precision:
    bf16: true
  logging:
    log_interval: 1
  checkpoint:
    save_interval: 1000
model:
  num_layers: 28
  hidden_size: 1024
  num_attention_heads: 16
  seq_length: 4096
  micro_batch_size: 1
  global_batch_size: 8
  optimizer:
    weight_decay: 0.1
    lr_scheduler:
      lr: 3.0e-3
      min_lr: 3.0e-4
      lr_decay_style: cosine
data:
  data_path: ./data/test
  tokenizer:
    tokenizer_type: GPT2
    tokenizer_path: ./tokenizer
    vocab_size: 50257
"""
        result = validate_config(tmp_yaml(content))
        assert "[OK]" in result

    def test_missing_model_section(self, tmp_yaml):
        content = """
system:
  tensor_model_parallel_size: 2
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "Missing required 'model' section" in result

    def test_missing_required_model_keys(self, tmp_yaml):
        content = """
system:
  tensor_model_parallel_size: 2
model:
  seq_length: 4096
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "Missing required 'model.num_layers'" in result
        assert "Missing required 'model.hidden_size'" in result

    def test_unknown_keys_warning(self, tmp_yaml):
        content = """
system:
  tensor_model_parallel_size: 2
  nonexistent_key: true
model:
  num_layers: 28
  hidden_size: 1024
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "WARNINGS" in result
        assert "nonexistent_key" in result


class TestMisplacement:
    def test_bf16_under_model(self, tmp_yaml):
        content = """
system:
  tensor_model_parallel_size: 2
model:
  num_layers: 28
  hidden_size: 1024
  bf16: true
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "ERRORS" in result
        assert "bf16" in result
        assert "system.precision" in result

    def test_tp_under_model(self, tmp_yaml):
        content = """
system: {}
model:
  num_layers: 28
  hidden_size: 1024
  tensor_model_parallel_size: 4
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "tensor_model_parallel_size" in result
        assert "system" in result

    def test_save_interval_under_model(self, tmp_yaml):
        content = """
system: {}
model:
  num_layers: 28
  hidden_size: 1024
  save_interval: 1000
data:
  data_path: ./data
"""
        result = validate_config(tmp_yaml(content))
        assert "save_interval" in result
        assert "system.checkpoint" in result

    def test_data_path_at_top(self, tmp_yaml):
        content = """
system: {}
model:
  num_layers: 28
  hidden_size: 1024
data_path: ./data/test
"""
        result = validate_config(tmp_yaml(content))
        assert "data_path" in result
        assert "under 'data'" in result or "data" in result


class TestEdgeCases:
    def test_file_not_found(self, tool):
        result = tool.execute(path="/nonexistent/path.yaml")
        assert "ERROR" in result
        assert "not found" in result

    def test_invalid_yaml(self, tmp_yaml):
        content = "key: [unclosed bracket"
        result = validate_config(tmp_yaml(content))
        assert "YAML SYNTAX ERROR" in result

    def test_non_dict_yaml(self, tmp_yaml):
        content = "- item1\n- item2\n"
        result = validate_config(tmp_yaml(content))
        assert "ERROR" in result
        assert "mapping" in result

    def test_unknown_config_type(self, tmp_yaml):
        content = "foo: bar\nbaz: 123\n"
        result = validate_config(tmp_yaml(content))
        assert "WARNING" in result
        assert "Could not determine" in result


class TestRealConfigs:
    """Test against actual FlagScale example configs."""

    @pytest.fixture
    def examples_dir(self):
        base = os.path.dirname(os.path.abspath(__file__))
        examples = os.path.join(base, "../../../../examples")
        if not os.path.isdir(examples):
            pytest.skip("examples directory not found")
        return examples

    def test_qwen3_top_level(self, examples_dir):
        path = os.path.join(examples_dir, "qwen3/conf/train.yaml")
        if not os.path.isfile(path):
            pytest.skip("qwen3 train.yaml not found")
        result = validate_config(path)
        assert "ERRORS" not in result

    def test_qwen3_model_level(self, examples_dir):
        path = os.path.join(examples_dir, "qwen3/conf/train/0_6b.yaml")
        if not os.path.isfile(path):
            pytest.skip("qwen3 0_6b.yaml not found")
        result = validate_config(path)
        assert "ERRORS" not in result
