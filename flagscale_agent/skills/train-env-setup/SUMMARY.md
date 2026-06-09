# Train-Env-Setup — Summary

Set up FlagScale training environment on GPU servers with all FL-customized dependencies.

**Load when**: creating a new conda environment for training, installing FlagScale dependencies, resolving CUDA/PyTorch version conflicts, or debugging import errors.

Strategy: collect ALL constraints first (driver, framework, recipe), solve for compatible versions, then install. PyTorch installs via official whl (excellent CUDA version coverage). Megatron-LM-FL, TransformerEngine-FL, and Apex MUST be built from source (no guaranteed whl for arbitrary CUDA versions). Always use `--no-deps` for packages that pull PyTorch.
