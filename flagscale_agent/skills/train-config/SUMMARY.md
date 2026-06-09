# Train-Config — Summary

Generate and validate FlagScale training configuration YAML files.

**Load when**: creating a new training config, modifying parallelism/optimizer/data settings, or debugging config validation errors.

Two-level Hydra YAML structure: `config.yaml` (top-level) references `conf/train/<size>.yaml` (model-specific). Covers all config sections: system, model, data, optimizer, distributed, logging. Includes config arithmetic validation and existing example reference.
