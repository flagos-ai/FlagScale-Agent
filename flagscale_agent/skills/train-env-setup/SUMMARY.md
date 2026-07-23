<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Train-Env-Setup — Summary

Set up FlagScale training environment on GPU servers with all FL-customized dependencies.

**Load when**: creating a new conda environment for training, installing FlagScale dependencies, resolving CUDA/PyTorch version conflicts, debugging import errors, or encountering "undefined symbol" errors after building CUDA extensions.

Strategy: collect ALL constraints first (driver, framework, recipe), solve for compatible versions, then install. PyTorch version selection: try FlagScale's version + driver's CUDA tag first (`pip install --dry-run` to verify); if that combination doesn't exist on PyPI (e.g., torch 2.9.0 has no cu124 wheel), fall back to the latest version that does. PyTorch CUDA tag MUST be compatible with system nvcc for source-build compatibility — if nvcc > torch CUDA, create an nvcc version shim (CUDA forward-compatibility). Megatron-LM-FL, TransformerEngine-FL, Apex, and Flash-Attention MUST ALL be built from source — pre-built whls are not acceptable. ALL source builds must use the SAME `CUDA_HOME` to prevent ABI mismatches. Always use `--no-deps` for packages that could pull a different PyTorch. Build-time deps (pybind11, cmake, ninja, packaging) must be pre-installed before source builds.
