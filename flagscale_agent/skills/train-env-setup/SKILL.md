---
name: train-env-setup
description: Set up FlagScale training environment on GPU servers. Install conda env, FlagScale, and all FL-customized dependencies.
  PyTorch installs via official whl matching the driver's max CUDA version. Megatron-LM-FL, TransformerEngine-FL, Apex, and
  Flash-Attention MUST ALL be built from source — pre-built whls are NOT acceptable because they may not match the system CUDA.
  Source builds guarantee binary compatibility with the actual hardware. Handles CUDA compatibility detection, multi-node
  deployment, and Docker image setup.
keywords:
- 安装
- 环境
- setup
- install
- env
- 环境搭建
- 训练环境
- conda
- megatron
- transformer-engine
- apex
- flash-attention
- 依赖
- 编译
- build
- cuda
- driver
- 驱动
- 多机
- multi-node
parameters:
- name: env_name
  description: Conda environment name
  default: flagscale-train
- name: python_version
  description: Python version
  default: '3.12'
- name: deps_dir
  description: Directory to clone source dependencies. If shared storage is detected, use <workspace_root>/code/deps/ so all
    nodes can access the same builds. Only fall back to a local path if no shared storage exists.
  default: <workspace_root>/code/deps/
requires:
- workspace-layout
suggests: []
constraints:
- id: train_env_setup_conda_prefix_not_shared_storage
  description: Conda environment must be created with --prefix on shared storage (not local /tmp). Local paths prevent multi-node
    access.
  trigger:
    tools:
    - shell
    keywords:
    - conda create
    - conda env create
  prompt: Check if conda env is being created on shared storage (--prefix) rather than local path
  correction: Use --prefix on shared storage for multi-node access.
- id: train_env_setup_pip_install_flagscale_without_no_deps
  description: pip install flagscale must use --no-deps to prevent PyTorch silent upgrade.
  trigger:
    tools:
    - shell
    keywords:
    - pip install flagscale
    - pip install -e .
    - pip3 install flagscale
  prompt: Check if pip install flagscale uses --no-deps flag
  correction: 'Use: pip install --no-deps -e . (or pip install --no-deps flagscale)'
- id: train_env_setup_pip_install_flash_attn_missing_no_deps
  description: pip install flash-attn must use --no-deps to prevent PyTorch from being overridden.
  trigger:
    tools:
    - shell
    keywords:
    - pip install flash-attn
    - pip install flash_attn
    - pip3 install flash-attn
    - pip3 install flash_attn
  prompt: Check if pip install flash-attn uses --no-deps flag
  correction: 'Use: pip install --no-deps flash-attn'
- id: train_env_setup_no_modify_dep_source
  description: Never modify dependency source code (apex, Megatron-LM-FL, TransformerEngine-FL, flash-attention) to work around
    build errors.
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - /apex/
    - /Megatron-LM-FL/
    - /TransformerEngine-FL/
    - /flash-attention/
    - deps/apex
    - deps/Megatron-LM-FL
    - deps/TransformerEngine-FL
    - deps/flash-attention
  prompt: Check if the file being edited is inside a dependency source tree (apex/, Megatron-LM-FL/, TransformerEngine-FL/,
    flash-attention/). Editing dependency source code is forbidden.
  correction: Do NOT modify dependency source code. Report the build error to the user and ask for guidance.
- id: train_env_setup_no_pip_install_apex_from_pypi
  description: Never run 'pip install apex' from PyPI — the PyPI 'apex' package is a Pyramid web framework, NOT NVIDIA Apex.
    NVIDIA Apex must always be built from source.
  trigger:
    tools:
    - shell
    keywords:
    - pip install apex
    - pip3 install apex
  prompt: Check if the command is 'pip install apex' from PyPI (without a local path or git URL). This installs the WRONG
    package (Pyramid web framework). NVIDIA Apex must be built from source via 'pip install --no-build-isolation .' inside
    the cloned apex directory.
  correction: 'Do NOT ''pip install apex'' from PyPI — that installs the wrong package. Build NVIDIA Apex from source: git
    clone https://github.com/NVIDIA/apex.git && cd apex && APEX_CUDA_EXT=1 pip install --no-build-isolation . -v'
- id: train_env_setup_megatron_must_use_fl_source
  description: megatron_core must be built from source — never install from any PyPI (generic or FlagScale).
  trigger:
    tools:
    - shell
    keywords:
    - pip install megatron_core
    - pip install megatron-core
    - pip3 install megatron_core
  prompt: "SCOPE: pip install megatron_core. CHECK: Is it installing from ANY PyPI (generic or FlagScale) or from a whl URL,
    rather than a local source build (pip install --no-build-isolation . inside a cloned Megatron-LM-FL directory)?
    Only local source builds are allowed."
  correction: "Build from source: git clone https://github.com/flagos-ai/Megatron-LM-FL.git && cd Megatron-LM-FL &&
    pip install --no-build-isolation . Pre-built whls are NOT acceptable."
- id: train_env_setup_te_must_use_fl_source
  description: transformer_engine must be built from source — never install from any PyPI (generic or FlagScale).
  trigger:
    tools:
    - shell
    keywords:
    - pip install transformer_engine
    - pip install transformer-engine
    - pip3 install transformer_engine
  prompt: "SCOPE: pip install transformer_engine. CHECK: Is it installing from ANY PyPI (generic or FlagScale) or from a whl URL,
    rather than a local source build (pip install --no-build-isolation . inside a cloned TransformerEngine-FL directory)?
    Only local source builds are allowed."
  correction: "Build from source: git clone --recursive https://github.com/flagos-ai/TransformerEngine-FL.git && cd TransformerEngine-FL &&
    NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . Pre-built whls are NOT acceptable."
- id: train_env_setup_pytorch_must_match_driver
  description: PyTorch CUDA tag must match the driver's max supported CUDA version, not FlagScale's default requirement.
  trigger:
    tools:
    - shell
    keywords:
    - pip install torch
    - pip3 install torch
  prompt: "SCOPE: pip install torch/pytorch. CHECK: Is the CUDA tag (cu118/cu121/cu124/cu126/cu128) compatible with the
    system driver's max supported CUDA? E.g., if driver is 535.x (max CUDA 12.4), installing torch+cu128 is WRONG.
    The agent should have already determined the driver's max CUDA in Step 1a."
  correction: "PyTorch CUDA tag must match driver's max supported CUDA. Check nvidia-smi for driver version, then choose
    the appropriate cuXXX tag. E.g., Driver 535.x → max cu124, Driver 560.x → max cu126, Driver 570.x → max cu128."
- id: train_env_setup_torch_version_from_pypi_not_requirements
  description: PyTorch version must come from PyPI availability for the driver's CUDA tag, NOT from FlagScale's requirements files.
    E.g., if driver supports cu124, install the latest torch that has a cu124 wheel (e.g., 2.6.0), even if train.txt says 2.9.0.
  trigger:
    tools:
    - shell
    keywords:
    - pip install torch
    - pip3 install torch
  prompt: "SCOPE: pip install torch. CHECK: Is the torch version being installed actually available as a wheel for the
    driver's max CUDA tag? Common mistake: agent reads torch==2.9.0 from FlagScale's train.txt and tries torch==2.9.0+cu124,
    but 2.9.0 only ships cu126/cu128 wheels. The correct approach is to query PyPI for the latest torch version that HAS
    a wheel for the driver's CUDA tag. If the version+cu_tag combination doesn't exist on PyPI, this is a violation."
  correction: "Do NOT use FlagScale's torch version. Query PyPI for available versions with your CUDA tag:
    pip install torch==<version>+<cu_tag> --dry-run. If it fails, the version doesn't exist for that CUDA.
    Use the latest torch that actually has a wheel for your driver's max CUDA."
- id: train_env_setup_no_whl_for_fl_deps
  description: Megatron-LM-FL, TransformerEngine-FL, Apex, and Flash-Attention must ALL be source-built. No whl installs.
  trigger:
    tools:
    - shell
    keywords:
    - .whl
    - --extra-index-url
  prompt: "SCOPE: pip install with .whl URL or --extra-index-url. CHECK: Is the command installing megatron_core,
    transformer_engine, apex, or flash_attn from a pre-built whl or from any PyPI index? These four packages
    MUST be built from source — pre-built whls may not match the system CUDA."
  correction: "Do NOT install FL dependencies from pre-built whls or PyPI. Build from source instead:
    Megatron-LM-FL: git clone + pip install --no-build-isolation .
    TransformerEngine-FL: git clone --recursive + NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .
    Apex: git clone + APEX_CUDA_EXT=1 pip install --no-build-isolation .
    Flash-Attention: git clone + pip install --no-build-isolation --no-deps ."
- id: train_env_setup_cuda_version_check
  description: Check CUDA/driver version before installing GPU packages
  trigger:
    tools:
    - shell
    keywords:
    - pip install torch
    - pip install apex
    - pip install flash-attn
    - pip install transformer_engine
    - pip3 install torch
  prompt: Check if CUDA/driver version was verified before installing GPU-dependent packages
  correction: Run nvidia-smi and check CUDA version before installing torch/TE/apex/flash-attn.
- id: train_env_setup_compat_analysis_before_install
  description: Must complete compatibility analysis (nvidia-smi, driver version, max CUDA version, available PyTorch wheels)
    before running any pip install for GPU packages. Without this analysis, wrong CUDA tags and version mismatches are inevitable.
  trigger:
    tools:
    - shell
    keywords:
    - pip install torch
    - pip install apex
    - pip install flash-attn
    - pip install transformer_engine
    - pip install megatron_core
    - conda run pip install
  prompt: "SCOPE: Any pip/conda install command for GPU-related packages (torch, apex, flash-attn, transformer_engine, megatron_core).
    CHECK: Has the agent already run nvidia-smi (or equivalent) and determined the driver's max supported CUDA version
    in THIS session? If the install command contains a CUDA tag (cu118/cu121/cu124/cu126/cu128) or installs a GPU package,
    but no prior nvidia-smi output exists in the conversation, this is a violation."
  correction: "Before installing GPU packages, complete compatibility analysis:
    1. Run nvidia-smi to get driver version and max CUDA version
    2. Determine the correct cuXXX tag for PyTorch
    3. Verify the target torch version has a wheel for that CUDA tag on PyPI
    Only then proceed with installation."
- id: train_env_setup_torch_version_must_exist_on_pypi
  description: Before installing a specific torch version+CUDA tag, must verify the exact wheel exists on PyPI. For example,
    torch==2.9.0+cu124 does NOT exist — only cu128 wheels exist for 2.9.0. Installing a non-existent version wastes time
    and fails silently or installs CPU-only torch.
  trigger:
    tools:
    - shell
    keywords:
    - pip install torch
    - pip3 install torch
  prompt: "SCOPE: pip install command that specifies an exact torch version with CUDA tag (e.g., torch==X.Y.Z+cuXXX or
    --extra-index-url .../whl/cuXXX).
    CHECK: Has the agent previously queried PyPI (pip index versions torch, pip install torch==<ver> --dry-run, or
    checked the PyTorch download page) to confirm this EXACT version+CUDA combination exists?
    Common non-existent combinations: torch==2.9.0+cu124, torch==2.8.0+cu124 (these only have cu128).
    If no prior verification exists in the conversation, this is a violation."
  correction: "Before installing torch, verify the wheel exists:
    1. pip index versions torch --index-url https://download.pytorch.org/whl/cuXXX
    2. Or: pip install torch==X.Y.Z+cuXXX --dry-run 2>&1 | head -5
    Common mapping: cu124 → torch<=2.6.0, cu126 → torch<=2.7.0, cu128 → torch>=2.8.0.
    Choose the latest version that actually has a wheel for your CUDA tag."
- id: train_env_setup_source_build_must_limit_arch
  description: Source builds (flash-attn, apex, TransformerEngine) must set TORCH_CUDA_ARCH_LIST to only the current GPU's SM
    architecture. Without this, builds compile for ALL architectures (sm_80, sm_86, sm_89, sm_90...) which wastes 20-50 minutes
    of compile time and may get killed by timeouts.
  trigger:
    tools:
    - shell
    keywords:
    - pip install --no-build-isolation
    - FLASH_ATTENTION_FORCE_BUILD
    - APEX_CUDA_EXT
    - NVTE_FRAMEWORK
  prompt: "SCOPE: Source build commands for flash-attn, apex, or TransformerEngine.
    CHECK: Does the command set TORCH_CUDA_ARCH_LIST to limit compilation to the current GPU's compute capability?
    Without it, the build compiles for all SM architectures which is extremely slow (30-60 min for flash-attn).
    The correct pattern is: TORCH_CUDA_ARCH_LIST=\"8.0\" (or whatever the GPU's SM is) before the pip install command.
    Detect with: python -c \"import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')\"."
  correction: "Add TORCH_CUDA_ARCH_LIST to limit compilation to current GPU only:
    SM_ARCH=$(python -c \"import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')\")
    TORCH_CUDA_ARCH_LIST=\"$SM_ARCH\" <rest of build command>"
context_injection:
  always:
  - Strategy
  - 'CRITICAL: Source-of-truth principle'
  by_tool:
    shell:
    - General rules
    - Step 1
    - Step 2
    - Step 3
---
# FlagScale Training Environment Setup

Set up a complete FlagScale training environment on a GPU server. All dependencies use FL-customized versions.

## Strategy

Environment setup is a constraint satisfaction problem. Collect ALL constraints first, solve for compatible versions, then install once.

### CRITICAL: Source-of-truth principle

**NEVER reference or inspect existing environments when determining what to install.** Existing environments (even `flagscale-train`, even on the same machine) may have different hardware, editable installs pointing to other workspaces, patched packages, or stale versions. They tell you NOTHING useful about what the CURRENT environment needs.

The ONLY valid sources of truth for dependency versions are:
1. FlagScale's own `requirements/*.txt`, `setup.py`, `setup.cfg`, `pyproject.toml`
2. The upstream repos of FL-customized dependencies: Megatron-LM-FL, TransformerEngine-FL
3. The actual hardware (driver version, GPU type) — queried fresh with nvidia-smi

**Do NOT run `pip list`, `conda list`, `pip show` in any existing environment.** Do NOT look at what another environment has installed. These are irrelevant and misleading.

### General rules

1. ALL installs go into the target conda environment — NEVER install into base or current environment. Use `conda run -n <env> pip install ...` for every pip command. To check dependency versions without installing, read setup.cfg/pyproject.toml from the source repo or use `pip index versions <pkg>`.
2. PyTorch installs via official whl — choose the CUDA tag that matches the driver's max supported CUDA version (NOT necessarily what FlagScale's train.txt specifies)
3. Megatron-LM-FL, TransformerEngine-FL, Apex, and Flash-Attention MUST ALL be built from source. Pre-built whls (including from FlagScale PyPI) are NOT acceptable — they are compiled against a specific CUDA version that may not match the system. Source builds are the ONLY way to guarantee binary compatibility with the actual hardware. Never install from generic PyPI (pypi.org) either — those packages are either wrong (apex) or missing FL customizations
4. Never modify dependency source code to work around errors — report to user
5. **After EVERY pip install, VERIFY the import works.** DO NOT assume a successful pip exit code means the package is usable. Immediately test: `python -c "import <package>; print(<package>.__version__)"`. For large packages (torch, flash-attn, apex), if `import` hangs >10s, the install is corrupt and must be redone. On NFS/shared storage, use `timeout 15 python -c "import <package>"` to catch hangs quickly without blocking the session.
6. **Auto-fetch FL dependencies**: When Megatron-LM-FL or TransformerEngine-FL source code is needed (for analysis, compilation, or debugging) and is not available locally, pull the latest automatically — don't ask the user. Repos: `https://github.com/flagos-ai/Megatron-LM-FL.git`, `https://github.com/flagos-ai/TransformerEngine-FL.git` (use `--recursive` for TE-FL)
7. **ALL FL-customized dependencies are MANDATORY.** Do NOT skip Megatron-LM-FL, TransformerEngine-FL, Apex, or Flash-Attention. These are not optional — FlagScale training will fail or produce incorrect results without them. If one is difficult to install, try the source build fallback. Only skip a dependency if the user explicitly requests it after being warned of the consequences.
8. **If the user asks to create a new environment, create a new environment.** Do not reuse an existing one, even if it appears to have the right packages. Existing environments may have editable installs pointing to other workspaces, patched packages, or stale versions. A fresh environment is the only way to guarantee a clean, reproducible baseline. If you believe reusing is genuinely better, explain why and ask — but do not silently substitute.
9. **NEVER copy packages between environments using `cp -r` from site-packages.** This bypasses pip's metadata tracking — pip won't know the package exists, so dependency resolution, upgrades, and uninstalls all break silently. Always install via `pip install` (from wheel, PyPI, or source build). If a prebuilt wheel isn't available, build from source — it takes longer but produces a properly registered package.
10. **Prefer shared storage for conda environments.** If the working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment with `--prefix <shared_path>/envs/<name>` instead of `-n <name>`. This ensures all nodes can access the same environment in multi-node training without duplication. Use `--prefix` for ALL subsequent `conda run` commands targeting this environment. Only use `-n` if no shared storage is available.
11. **Conda envs and pip packages MUST go on shared storage, not local paths.** Even if `/tmp` or local disk has more space or is faster, the conda environment prefix and pip install target MUST be on shared storage (e.g., `/share/.../envs/<name>`). The only exception is `TMPDIR` for pip's temporary build cache — that can point to local storage to speed up compilation, but the final installed packages must land in the shared prefix.

## Step 0: Determine Dependency Source Directory

**Before anything else, determine `deps_dir` — the directory for cloning and building source dependencies.**

1. If workspace-layout skill has been loaded and `workspace_root` is known (from memory or detection), set `deps_dir = <workspace_root>/code/deps/`. This ensures all nodes in multi-node training can access the same builds.
2. If shared storage is available but workspace_root is not yet set, detect it now (see workspace-layout Step 1) and use it.
3. Only if NO shared storage is available, fall back to a local path.

**Summary**: `deps_dir` is always on shared storage when available. Never hardcode `/opt/flagscale/deps` — this path is local to one node and invisible to others.

Record `deps_dir` in memory after determining it.

## Step 1: Constraint Collection (NO installs in this step)

Collect ALL version constraints before installing anything. Do NOT look at existing environments.

### 1a. Hardware constraint — driver → max CUDA

```bash
nvidia-smi --query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader | head -1 && echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
nvcc --version 2>/dev/null || echo "nvcc not found"
```

The `GPU_COUNT=` line gives the exact GPU count. Use that number in all subsequent references — never count nvidia-smi output lines manually.

Driver → max CUDA version (for PyTorch wheel selection):
- Driver 570.x → CUDA ≤ 12.8 → wheels: cu118, cu121, cu124, cu126, cu128
- Driver 560.x → CUDA ≤ 12.6 → wheels: cu118, cu121, cu124, cu126
- Driver 550.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 535.x → CUDA ≤ 12.4 → wheels: cu118, cu121, cu124
- Driver 530.x → CUDA ≤ 12.1 → wheels: cu118, cu121
- Driver 520.x → CUDA ≤ 11.8 → wheels: cu118

### CRITICAL: CUDA version alignment for source builds

PyTorch whl bundles its own CUDA runtime, so `torch+cu128` can run on a system with driver 535.x (CUDA 12.4 compatible). **However**, source-building Apex/TE-FL/Flash-Attention uses the system `nvcc` compiler. If system nvcc version ≠ PyTorch's CUDA version, builds will fail or produce incompatible binaries.

**MANDATORY resolution strategy:**

1. **PyTorch MUST match the driver's max supported CUDA version.** Choose the PyTorch whl whose CUDA tag matches what the driver supports. E.g., Driver 535.x → max CUDA 12.4 → use `torch+cu124`. This guarantees all source builds are compatible with the system nvcc.

2. **If system nvcc is missing or wrong version**: Install the CUDA toolkit that matches the chosen PyTorch CUDA tag. E.g., torch+cu124 → install CUDA 12.4 toolkit, then set `CUDA_HOME=/usr/local/cuda-12.4` for all source builds.

3. **ALL four FL dependencies (Megatron-LM-FL, TransformerEngine-FL, Apex, Flash-Attention) MUST be built from source.** Pre-built whls from FlagScale PyPI are NOT acceptable when driver/CUDA versions don't match FlagScale's default requirements — source builds guarantee binary compatibility with the actual hardware.

**Decision rule**: If `nvcc --version` reports a different major.minor than `torch.version.cuda`, you MUST resolve this BEFORE attempting any source build. Do NOT bypass version checks by modifying dependency source code.

**When driver doesn't match FlagScale's default CUDA requirement** (e.g., FlagScale's train.txt specifies torch+cu128 but driver only supports cu124):
- Install PyTorch matching the DRIVER (e.g., torch+cu124), NOT what train.txt says
- Then source-build ALL four FL dependencies against that PyTorch version
- This is the ONLY reliable path — never force a higher CUDA version than the driver supports

### 1b. FlagScale framework constraint — read from source (REFERENCE ONLY for non-PyTorch deps)

Read FlagScale's own dependency declarations (NOT from any installed environment):

```bash
cat requirements.txt
cat requirements/cuda/train.txt
cat requirements/cuda/base.txt
cat setup.py
```

**CRITICAL: FlagScale's train.txt torch version is REFERENCE ONLY, NOT authoritative.**
FlagScale's `requirements/cuda/train.txt` may specify e.g. `torch==2.9.0+cu128`. **IGNORE this version for PyTorch installation.** The actual PyTorch version is determined SOLELY by the driver's max CUDA tag — query PyPI for the latest torch available with that tag (see Step 1d). Do NOT combine FlagScale's torch version number with the driver's CUDA tag (e.g., `torch==2.9.0+cu124` will likely NOT exist).

Use FlagScale requirements ONLY to determine:
- Python version requirement
- Non-PyTorch dependency versions (pydantic, hydra, etc.)
- Which FL-customized dependencies are needed (Megatron-LM-FL, TE-FL, etc.)

**IMPORTANT**: `requirements/cuda/train.txt` may contain `megatron_core @ https://...whl` or `transformer_engine @ https://...whl` URLs. These whl URLs point to the official FlagScale PyPI. **However, do NOT install from these whls** — they are compiled against a specific CUDA version that may not match your system. Instead, use the version information to identify the correct source branch/tag, then build from source. Do NOT install megatron_core or transformer_engine from generic PyPI (pypi.org) either.

Also fetch the setup configs of the two FL forks to check their torch/python requirements:

```bash
# Megatron-LM-FL: check setup.py for torch/python_requires
web_fetch https://raw.githubusercontent.com/flagos-ai/Megatron-LM-FL/main/setup.py
# TransformerEngine-FL: check setup.py for torch/python/minor version requirements
web_fetch https://raw.githubusercontent.com/flagos-ai/TransformerEngine-FL/main/setup.py
```

### 1c. FL-customized dependency analysis (as important as PyTorch itself)

FlagScale requires four FL-customized / special packages. ALL four are MANDATORY and ALL MUST be built from source:

| Package | Source | Install method |
|---------|--------|----------------|
| Megatron-LM-FL | flagos-ai GitHub | source build ONLY (`git clone` + `pip install --no-build-isolation .`) |
| TransformerEngine-FL | flagos-ai GitHub | source build ONLY (`git clone --recursive` + `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .`) |
| Apex | NVIDIA GitHub | source build ONLY (`APEX_CUDA_EXT=1 pip install --no-build-isolation .`) |
| Flash-Attention | Dao-AILab GitHub | source build ONLY (`--no-deps --no-build-isolation .`) |

**Why source build is mandatory**: Pre-built whls are compiled against a specific CUDA version. When the system driver/CUDA doesn't match FlagScale's default (which is common), pre-built whls produce silent runtime errors or segfaults. Source builds compile against the ACTUAL system CUDA toolkit, guaranteeing binary compatibility.

For each, analyze:
- **Megatron-LM-FL**: MUST build from source. Clone from GitHub and build with `pip install --no-build-isolation .`. Never use pre-built whls from FlagScale PyPI — they may not match the system CUDA.
- **TransformerEngine-FL**: MUST build from source. Requires `--recursive` clone for submodules. Build with `NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .`
- **Apex**: MUST build from source. Must compile with `APEX_CUDA_EXT=1` matching PyTorch's CUDA version. Check that the nvcc toolkit version matches torch.version.cuda (not just driver CUDA version).
- **Flash-Attention**: MUST build from source. The version must match the installed PyTorch version. Use `--no-deps` to prevent pip from upgrading PyTorch. Check: GPU compute capability ≥ 8.0 required for flash-attn v2.x.

### 1d. Solve — write the FULL compatibility table

Write a COMPLETE compatibility table covering ALL components. Do NOT skip to Step 2 until this table is written and verified.

**CRITICAL: Determine PyTorch version BEFORE writing the table**

Do NOT write "torch_ver+cuXXX" as a placeholder. You MUST determine the EXACT PyTorch version that exists for the driver's max CUDA tag:

1. From Step 1a, you know the driver's max CUDA (e.g., Driver 535.x → max cu124)
2. **IGNORE FlagScale's torch version** — it's for a different CUDA. Do NOT combine FlagScale's version number with your CUDA tag.
3. Query PyPI to find the LATEST PyTorch version available for YOUR CUDA tag:
   ```bash
   pip install torch== 2>&1 | grep -oP '\d+\.\d+\.\d+' | sort -V | tail -10
   # Then check which versions have your CUDA tag:
   pip install torch==<latest_version>+cu124 --dry-run 2>&1 | head -5
   ```
   Or use the PyTorch download page logic:
   - cu124: latest is typically 2.6.0
   - cu126: latest is typically 2.7.0+
   - cu128: latest is typically 2.9.0+
4. Choose the LATEST stable version that has a wheel for your CUDA tag
5. Write the EXACT version in the table (e.g., `torch==2.6.0+cu124`)

**Example decision flow**:
- Driver 535.x → max CUDA 12.4 → need cu124 wheels
- FlagScale train.txt says `torch==2.9.0+cu128` → **IGNORE this** (2.9.0 has no cu124 wheel)
- Query PyPI: latest torch with cu124 is 2.6.0
- Write in table: `torch==2.6.0+cu124`

This eliminates trial-and-error in Step 3a — you install exactly what you determined here.

```
COMPATIBILITY ANALYSIS TABLE
============================
Hardware: N×GPU_TYPE, Driver DRI_VER → max CUDA CUDA_MAX
FlagScale requirements:
  Python: py_req
  PyTorch: torch_req (from requirements, may differ from what we'll install)
  CUDA toolkit required: cuda_toolkit_needed

| # | Component | Required Version | Install Method | Notes |
|---|-----------|-----------------|---------------|-------|
| 1 | Conda env | python=py_ver | conda create --prefix | path: <shared>/envs/env_name (or -n if no shared storage) |
| 2 | PyTorch | torch==X.Y.Z+cuXXX | pip (whl) | EXACT version from PyPI query; cuXXX matches driver's max CUDA; --extra-index-url https://download.pytorch.org/whl/cuXXX |
| 3 | FlagScale | editable | pip -e ".[cuda-train]" | from project root, --no-deps |
| 4 | Megatron-LM-FL | latest | SOURCE BUILD | git clone + pip install --no-build-isolation . |
| 5 | TransformerEngine-FL | latest | SOURCE BUILD | git clone --recursive + NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . |
| 6 | Apex | master | SOURCE BUILD | git clone NVIDIA/apex + APEX_CUDA_EXT=1 |
| 7 | Flash-Attention | fa_ver | SOURCE BUILD | --no-deps --no-build-isolation to protect PyTorch |
```

CRITICAL CHECKLIST before proceeding:
- [ ] PyTorch version is EXACT (e.g., `torch==2.6.0+cu124`), NOT a placeholder
- [ ] PyTorch CUDA tag matches driver's max supported CUDA (NOT FlagScale's default if they differ)
- [ ] PyTorch version was verified to exist on PyPI (via `pip index versions torch`)
- [ ] All versions in the table are derived from FlagScale source files (NOT existing envs)
- [ ] Shared storage checked — conda env path uses --prefix on shared FS if available
- [ ] CUDA toolkit version matches PyTorch's CUDA (not driver's)
- [ ] GPU compute capability ≥ required by flash-attn
- [ ] Megatron-LM-FL will be built from source (NO whl, NO PyPI)
- [ ] TransformerEngine-FL will be built from source (NO whl, NO PyPI)
- [ ] Apex build flags include APEX_CUDA_EXT=1
- [ ] Flash-attn install uses --no-deps

Present the table and ASK FOR CONFIRMATION. Do NOT proceed to Step 2 until the user confirms.
After confirmation, annotate your response with [ENV_COMPAT_ANALYZED].

## Step 2: Conda Environment

### 2a. Check shared storage FIRST

**CRITICAL**: If the current working directory is under a shared filesystem (e.g., `/share/`, `/mnt/share/`, `/mnt/cfs/`), create the conda environment on the shared storage — NOT on the local node. This ensures all nodes in multi-node training can access the same environment without duplication.

```bash
# Check if we're on shared storage
df -h . | grep -E '^[^/]' | head -5

# Check available shared mount points
ls -d /share /mnt/share /mnt/cfs /mnt/dfs 2>/dev/null
```

If shared storage is found (e.g., `/share/project/...`), use `--prefix` instead of `--name`:

```bash
# Create env in shared storage — use --prefix with full path
conda create --prefix /share/project/<path>/envs/{env_name} python={python_version} -y

# For all subsequent commands, use --prefix (not -n):
conda run --prefix /share/project/<path>/envs/{env_name} <command>
```

If NO shared storage is found, fall back to `-n`:

```bash
conda create -n {env_name} python={python_version} -y
# In non-interactive shells (agent), use: conda run -n {env_name} <command>
# In interactive shells (user), use: conda activate {env_name}
```

### 2b. Verify

```bash
python --version
```

## Step 3: Install FlagScale

### 3a. Pin PyTorch FIRST (before installing FlagScale)

**PyTorch installs via official whl** — PyTorch has excellent coverage of CUDA versions (cu118, cu121, cu124, cu126, cu128), so a pre-built wheel is always available. No source build needed.

**CRITICAL**: Use the EXACT version determined in Step 1d. Do NOT re-derive or guess the version here. The version was already verified to exist on PyPI during Step 1d.

**CRITICAL**: The PyTorch CUDA tag is determined by the DRIVER, not by FlagScale. If FlagScale's train.txt says `torch==2.9.0+cu128` but your driver only supports cu124, install the latest torch available for cu124 (e.g., `torch==2.6.0+cu124`). NEVER try to install a torch version that doesn't have a wheel for your CUDA tag.

**CRITICAL**: `pip install -e ".[cuda-train]"` will pull in ALL requirements, including PyTorch from the requirements files. If those requirements specify a different CUDA version than what your driver supports, pip will silently upgrade PyTorch and all CUDA libraries. This is the #1 cause of wasted time in environment setup.

**Always pin PyTorch before FlagScale install:**

```bash
# Install exact PyTorch version determined in Step 1d — cu_tag matches driver's max CUDA
pip install torch=={exact_version_from_step_1d} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/{cu_tag}
# Verify CUDA version is correct
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 3b. Clone and Install FlagScale

FlagScale-Agent is an independent repository. FlagScale itself must be cloned separately:

```bash
# Clone FlagScale into the workspace code directory
mkdir -p {workspace_root}/code
git clone --depth 1 https://github.com/FlagOpen/FlagScale.git {workspace_root}/code/FlagScale
cd {workspace_root}/code/FlagScale
```

Then install in editable mode:

```bash
pip install -e ".[cuda-train]"
```

**If pip tries to upgrade PyTorch during this step**, abort and use the two-phase approach:
```bash
# Phase 1: install FlagScale without deps
pip install --no-deps -e .
# Phase 2: install remaining deps from requirements (PyTorch already pinned, won't change)
pip install -r requirements/cuda/train.txt
```

This ensures PyTorch stays at the pinned version.

Verify:
```bash
flagscale --help
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# CRITICAL: confirm torch version did NOT change from what was installed in 3a
```

**Important**: `requirements/cuda/train.txt` includes a megatron-core whl from the FlagScale PyPI. Do NOT use this whl — it is compiled against a specific CUDA version that may not match your system. Always build Megatron-LM-FL from source in Step 4. If `pip install -e ".[cuda-train]"` installs the whl version, it will be overwritten by the source build in Step 4.

## Step 4: FL-Customized Dependencies (ALL SOURCE BUILDS)

These are FlagScale's customized forks. ALL MUST be built from source — pre-built whls are NOT acceptable. Install order matters — Megatron-LM-FL first, then the rest.

### 4a. Megatron-LM-FL (MANDATORY source build)

**Always build from source** — pre-built whls are NOT acceptable regardless of source.

```bash
mkdir -p {deps_dir}
git clone https://github.com/flagos-ai/Megatron-LM-FL.git {deps_dir}/Megatron-LM-FL
cd {deps_dir}/Megatron-LM-FL
pip install --no-build-isolation . -v
```

Verify:
```bash
python -c "from megatron.plugin.platform import get_platform; print('OK:', get_platform())"
```

### 4b. TransformerEngine-FL (MANDATORY source build)

**Always build from source** — pre-built whls are NOT acceptable regardless of source.

```bash
pip install nvidia-mathdx --extra-index-url https://pypi.nvidia.com
git clone --recursive https://github.com/flagos-ai/TransformerEngine-FL.git {deps_dir}/TransformerEngine-FL
cd {deps_dir}/TransformerEngine-FL
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . -v
```

Note: Source build takes 10-30 minutes. Do NOT interrupt or ask for confirmation during compilation — just wait for it to finish.

Verify:
```bash
python -c "import transformer_engine; print('TE version:', transformer_engine.__version__)"
```

### 4c. NVIDIA Apex (source build)

**WARNING: The PyPI package named `apex` is a Pyramid web framework — NOT NVIDIA Apex.** Never run `pip install apex` from PyPI. NVIDIA Apex must always be built from source.

```bash
git clone --depth 1 https://github.com/NVIDIA/apex.git {deps_dir}/apex
cd {deps_dir}/apex

# Detect current GPU compute capability — only compile for this architecture
SM_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")

TORCH_CUDA_ARCH_LIST="$SM_ARCH" NVCC_APPEND_FLAGS='--threads 4' APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    pip install --no-build-isolation . -v
```

Verify:
```bash
python -c "import apex; print('Apex OK')"
```

**Common issue**: CUDA version mismatch between system nvcc and PyTorch's CUDA. If Apex build fails with version check error, go back to Step 1a "CUDA version alignment for source builds" and resolve the mismatch. Do NOT modify Apex source code to bypass the check.

**IMPORTANT: Pure-Python vs CUDA Extensions**

Apex has two install modes:
- **Full install** (with `APEX_CUDA_EXT=1`): Compiles CUDA extensions for fused kernels. Required for `gradient_accumulation_fusion`, fused Adam, fused layer norm, etc.
- **Pure-Python install** (without CUDA flags or `pip install apex`): Only Python wrappers, NO fused kernels. Many Megatron features silently fall back to slower paths or fail with `RuntimeError: ... requires APEX CUDA extensions`.

**If you see `gradient_accumulation_fusion requires APEX CUDA extensions`**: Apex was installed in pure-Python mode. You must either:
1. Reinstall with CUDA extensions (recommended): use the build command above with `APEX_CUDA_EXT=1`
2. OR disable ALL fusion flags at once: `gradient_accumulation_fusion: false`, `bias_gelu_fusion: false`, `bias_swiglu_fusion: false` — and note the performance impact

Never disable just one fusion flag — if APEX CUDA extensions are missing, ALL fused kernels are unavailable.

### 4d. Flash-Attention 2

**CRITICAL**: Always use `--no-deps` when installing flash-attn. Without it, pip may upgrade PyTorch to an incompatible version, causing cascading failures (triton mismatch, CUDA version conflicts). The PyTorch version was already pinned in Step 3 — do not let flash-attn override it.

**CRITICAL**: Only compile for the current GPU's SM architecture. Flash-attn defaults to compiling ALL supported architectures (sm_80, sm_86, sm_89, sm_90, ...), which takes 30-60 minutes and is completely unnecessary — you only need the architecture of the GPUs on this machine. Set `TORCH_CUDA_ARCH_LIST` to the detected compute capability. This reduces compile time to 5-10 minutes.

```bash
git clone --branch v2.8.1 --depth 1 https://github.com/Dao-AILab/flash-attention.git {deps_dir}/flash-attention
cd {deps_dir}/flash-attention

# Detect current GPU compute capability — ONLY compile for this architecture
SM_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')")
echo "Building flash-attn for SM $SM_ARCH only (skipping other architectures)"

TORCH_CUDA_ARCH_LIST="$SM_ARCH" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
    pip install --no-build-isolation --no-deps . -v
```

**CUDA toolkit vs driver version**: Flash-attn compilation requires the CUDA **toolkit** version (nvcc) to match PyTorch's CUDA version, NOT the driver version. Check with `nvcc --version` (toolkit) vs `nvidia-smi` (driver). If nvcc is missing or wrong version, install the matching CUDA toolkit or set `CUDA_HOME` to the correct path.

After installing, verify PyTorch was NOT changed:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
If the version differs from what was installed in Step 3, flash-attn broke the environment. Uninstall flash-attn, reinstall the correct PyTorch, and retry with `--no-deps`.

Verify:
```bash
python -c "import flash_attn; print('Flash-Attention version:', flash_attn.__version__)"
```

## Step 5: Final Verification

Run a comprehensive check:

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')

from megatron.plugin.platform import get_platform
print(f'Megatron platform: {get_platform()}')

import transformer_engine
print(f'TransformerEngine: {transformer_engine.__version__}')

import apex
print('Apex: OK')

import flash_attn
print(f'Flash-Attention: {flash_attn.__version__}')

print('All dependencies ready!')
"
```

**Post-install verification gate — do NOT proceed to training or model porting until ALL checks pass:**

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| PyTorch CUDA | `python -c "import torch; assert torch.cuda.is_available()"` | No error |
| PyTorch version unchanged | Compare against version from Step 3 | Exact match |
| Megatron-LM-FL | `python -c "from megatron.plugin.platform import get_platform"` | No ImportError |
| TransformerEngine-FL | `python -c "import transformer_engine"` | No ImportError |
| Apex | `python -c "import apex"` | No ImportError |
| Flash-Attention | `python -c "import flash_attn"` | No ImportError |

If ANY check fails, fix it before moving on. Do not proceed with "we'll fix it later" — dependency issues compound during training and are much harder to debug.

### 5b. Package provenance check

Verify that each FL dependency is installed from the correct source — not from a different workspace or stale editable install:

```bash
pip show megatron-core transformer-engine apex flash-attn 2>/dev/null | grep -E "^(Name|Location|Editable)"
```

For each package:
- If `Editable project location` is shown, verify it points to a directory within the CURRENT workspace (not a different `/workspace/X/` directory)
- If the editable path points to a different workspace, the installed code won't match the code you'll read for debugging — reinstall from the correct source tree within your workspace
- For non-editable installs, verify the `Location` is inside the target conda environment's `site-packages/`

**Cross-workspace editable installs are NEVER acceptable.** Even if two directories are at the same git commit today, they can diverge silently. If the dependency source doesn't exist in your workspace, clone it locally first (`git clone <repo> /workspace/<your_workspace>/<dep>/`), then editable-install from the local clone.

This check prevents the most insidious debugging trap: reading source code from one directory while the runtime uses code from a completely different directory.

## Step 6: Multi-Node Deployment

When setting up multiple nodes for distributed training:

1. Ensure the same conda environment and dependencies are installed on ALL nodes
2. Verify passwordless SSH between nodes:
   ```bash
   ssh -o BatchMode=yes <other_node> hostname
   ```
3. Verify NCCL connectivity between nodes:
   ```bash
   # On each node, check IB/RoCE NICs are up
   ibstat 2>/dev/null || rdma link show 2>/dev/null || echo "No RDMA detected"
   ```
4. Set consistent NCCL environment variables across all nodes:
   ```bash
   export NCCL_IB_DISABLE=0        # Enable IB if available
   export NCCL_NET_GDR_LEVEL=5     # GPUDirect RDMA level
   export NCCL_SOCKET_IFNAME=eth0  # Fallback interface (adjust to actual)
   ```
5. Verify shared filesystem is mounted at the same path on all nodes (for checkpoints and data)

## Error Handling Rules

1. **Network errors** (git clone fails, pip timeout): Tell user to configure proxy. Do NOT try alternative URLs or workarounds.
2. **Build errors** (compilation fails): Report the exact error to user. Do NOT modify dependency source code.
3. **Version mismatch**: Report versions found and let user decide. Do NOT skip version checks by patching code.
4. **Successful builds**: Proceed to next step automatically. Do NOT ask user to confirm after each successful install.

## Alternative: Docker Image

If source builds are too complex, recommend the official training Docker image:

```bash
docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
docker run -itd --gpus all --shm-size=500g --name <name> harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856 /bin/bash
docker exec -it <name> /bin/bash
# In non-interactive shells (agent), use: conda run -n flagscale-train <command>
```

This image has all dependencies pre-installed.

## Download Best Practices

- Always use `wget -c` (resume) instead of plain `wget` for large files.
- For files > 1GB, verify size after download: `ls -lh <file>`.
- Use proxy when available: check `echo $HTTP_PROXY` before downloading.
- For git clone on large repos, use `--depth 1` to avoid fetching full history.
- If a download fails, resume instead of deleting and re-downloading.
- Run large downloads as separate commands, not chained with `&&` or `&`, so failures are isolated.

---

## Related Skills

- `topo-detect` — detect hardware topology after environment setup
- `train-config` — generate training configuration files
- `train-run` — launch training after environment is ready
