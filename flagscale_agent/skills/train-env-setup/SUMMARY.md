# Train-Env-Setup — Summary

Set up FlagScale training environment on GPU servers with all FL-customized dependencies.

**Load when**: creating a new conda environment for training, installing FlagScale dependencies, resolving CUDA/PyTorch version conflicts, or debugging import errors.

Strategy: collect ALL constraints first (driver, framework, recipe), solve for compatible versions, then install. PyTorch version selection: try FlagScale's version + driver's CUDA tag first (`pip install --dry-run` to verify); if that combination doesn't exist on PyPI (e.g., torch 2.9.0 has no cu124 wheel), fall back to the latest version that does. PyTorch CUDA tag MUST match system nvcc for source-build compatibility. Megatron-LM-FL, TransformerEngine-FL, Apex, and Flash-Attention MUST ALL be built from source — pre-built whls are not acceptable. Always use `--no-deps` for packages that could pull a different PyTorch.
