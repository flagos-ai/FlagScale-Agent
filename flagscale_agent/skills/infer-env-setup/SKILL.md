---
description: Set up inference environment for vllm-plugin-FL on hardware backends.
  Covers SSH connection, Docker container creation, CPU-only vLLM install, plugin
  editable install, FlagGems install, and import verification. Use before infer-hw-adapt
  or infer-model-adapt.
name: infer-env-setup
---

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

# Inference Environment Setup

Set up the inference environment for vllm-plugin-FL on hardware backends.

## When to Use This Skill

Use this skill when:
- Setting up a new inference environment for hardware adaptation
- Creating Docker containers for vllm-plugin-FL testing
- Installing vLLM, plugin, and FlagGems on a new machine
- Reconnecting to an existing environment after a break

## Critical Rules

1. **Confirm SSH connection first** — ask user for SSH host/alias if not provided, verify with `ssh <host> hostname` before any work.
2. **All work happens inside Docker containers** — never install inference packages on the host.
3. **Fresh workspace isolation** — every adaptation task starts with a fresh clone (local and remote). Do NOT reuse existing directories or mix with other projects. Use a dedicated directory per task (e.g., `adapt/<backend>-vllm-<version>/`).
4. **Local edit → sync → remote test** — edit code locally (or on host), sync to container workspace, then run tests inside container. Don't edit files inside the container directly.
5. **Check device occupancy before tests** — use the backend's monitoring tool (see Container Setup) to confirm compute devices are free.
6. **vLLM installs as CPU-only** (`VLLM_TARGET_DEVICE=empty`) — the plugin provides hardware-specific backends.
7. **Pin vLLM version** — check plugin's `pyproject.toml` for the required version, never `pip install vllm` without `==X.Y.Z`.
8. **Check container existence** before creating — reuse running containers, start stopped ones.
9. **Use `--network host`** for Docker containers on GPU machines.
10. **Use tmux for long-running commands** — SSH sessions will timeout otherwise.
11. **Record paths to memory** — after Step 0 probe, immediately save all paths (ssh_host, container_name, workspace_root, model_path, log_dir) to memory. Never guess paths.
12. **Batch independent tool calls** — when multiple shell commands, file reads, or memory operations are independent, execute them in one response.

---

## Remote Access

All operations run on remote GPU machines via SSH. The agent does NOT have direct access to GPUs.

### Step 0: Confirm Connection & Gather Environment Info

If `ssh_host` is not provided, **ask the user**:
> "What is the SSH alias or connection string for the target hardware? (e.g., `metax_c550`, `ssh user@host -p port`)"

Once obtained, run the following **environment probe** in one shot:

```bash
ssh <ssh_host> "echo '=== hostname ===' && hostname && \
  echo '=== date ===' && date && \
  echo '=== device info ===' && \
  (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
   || mx-smi 2>/dev/null || npu-smi info 2>/dev/null \
   || echo 'no device tool found') && \
  echo '=== device processes ===' && \
  (nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null \
   || echo 'N/A — check with backend tool') && \
  echo '=== disk space ===' && df -h /workspace /home 2>/dev/null | head -5 && \
  echo '=== docker containers ===' && \
  docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' && \
  echo '=== docker images (vllm) ===' && \
  docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -i vllm && \
  echo '=== existing adapt dirs ===' && \
  find /workspace -maxdepth 3 -name 'vllm-plugin-FL' -type d 2>/dev/null && \
  echo '=== workspace layout ===' && ls -la /workspace/ 2>/dev/null | head -20"
```

**After the probe, immediately record to memory:**

```
memory_write('<backend>_ssh_host', '<ssh_host>')
memory_write('<backend>_container_name', '<container_name>')
memory_write('<backend>_workspace_root', '/workspace/adapt/<backend>-vllm-<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
memory_write('<backend>_log_dir', '/workspace/adapt-logs')
```

---

## Container Setup

### MetaX C550

```bash
# 1. Check existing containers
ssh <ssh_host> "docker ps -a | grep vllm"

# 2. If stopped container exists, start and reuse it
ssh <ssh_host> "docker start <container_name>"

# 3. If no container exists, create one
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt \
  --network host \
  --device /dev/mxcd0 --device /dev/mxcd1 \
  --device /dev/mxcm0 \
  -v /workspace:/workspace \
  -v /home:/home \
  --shm-size 64g \
  <metax_vllm_image> \
  sleep infinity"

# 4. Verify devices inside container
ssh <ssh_host> "docker exec vllm_fl_adapt mx-smi"
```

Device occupancy check for MetaX:
```bash
ssh <ssh_host> "docker exec <container> mx-smi | grep -E 'Used|Proc'"
```

### Ascend 910B

```bash
# Create container with NPU device mounts
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt_ascend \
  --network host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v /workspace:/workspace \
  --shm-size 64g \
  <ascend_vllm_image> \
  sleep infinity"

# Verify NPU devices
ssh <ssh_host> "docker exec <container> npu-smi info"
```

### Moore Threads S4000/S5000 (MUSA)

```bash
# 1. Check existing containers
ssh <ssh_host> "docker ps -a | grep vllm"

# 2. If stopped container exists, start and reuse it
ssh <ssh_host> "docker start <container_name>"

# 3. If no container exists, create one
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt_mt \
  --network host \
  --env MUSA_VISIBLE_DEVICES=all \
  -v /workspace:/workspace \
  --shm-size 64g \
  <mt_vllm_image> \
  sleep infinity"

# 4. Verify MUSA devices
ssh <ssh_host> "docker exec <container> mthreads-gmi"
```

Device occupancy check for Moore Threads:
```bash
ssh <ssh_host> "docker exec <container> mthreads-gmi | grep -E 'Used|Proc'"
```

File transfer — `scp` to host path that is bind-mounted into the container:
```bash
scp localfile <ssh_host>:/data/wlx/filename
# Immediately visible inside container at the corresponding mount path
```

### Iluvatar 天数 (CoreX / tianshu)

```bash
# 1. Check existing containers
ssh tianshu "docker ps -a | grep vllm"

# 2. If stopped container exists, start and reuse it
ssh tianshu "docker start <container_name>"

# 3. If no container exists, create one
#    --init is required: tini as PID 1 reaps zombie worker processes
#    Mount only the three runtime .so files, not the entire corex directory
ssh tianshu "docker run -d \
  --name vllm_fl_adapt \
  --network host \
  --init \
  --device /dev/iluvatar0 --device /dev/iluvatar1 \
  --device /dev/iluvatar2 --device /dev/iluvatar3 \
  --device /dev/iluvatar4 --device /dev/iluvatar5 \
  --device /dev/iluvatar6 --device /dev/iluvatar7 \
  --device /dev/ixdnn \
  -v /mnt/share/wlx:/workspace \
  -v /usr/local/corex-4.5.0/lib64/libcuda.so.1:/usr/local/corex-4.5.0/lib64/libcuda.so.1 \
  -v /usr/local/corex-4.5.0/lib64/libcuda.so:/usr/local/corex-4.5.0/lib64/libcuda.so \
  -v /usr/local/corex-4.5.0/lib64/libixml.so:/usr/local/corex-4.5.0/lib64/libixml.so \
  --shm-size 64g \
  dev-community-acr-registry.cn-shanghai.cr.aliyuncs.com/dev-community/dev-community:vllm-py3.12-corex.4.5.0-ubuntu24.04 \
  sleep infinity"

# 4. Verify devices inside container
ssh tianshu "docker exec <container> ixsmi"
```

Device occupancy check for Iluvatar:
```bash
ssh tianshu "docker exec <container> ixsmi | grep -E 'Used|Proc'"
```

**JumpServer exec pattern** — tianshu's JumpServer blocks interactive TTY and tmux. Use background exec + log polling for all long-running commands:
```bash
# Launch in background, redirect output to log file
ssh tianshu "docker exec -d <container> sh -c \
  'cmd > /workspace/adapt-logs/out.log 2>&1'"

# Poll progress with short-output commands (safe for JumpServer)
ssh tianshu "docker exec <container> tail -20 /workspace/adapt-logs/out.log"
ssh tianshu "docker exec <container> wc -l /workspace/adapt-logs/out.log"
```

File transfer — JumpServer blocks heredoc and scp piping. Use git pull via NFS:
```bash
# On local: commit and push changes
git push origin <branch>

# On tianshu: pull inside container
ssh tianshu "docker exec <container> bash -c \
  'cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL && git pull'"
```

### Hygon DCU

```bash
# 1. Check existing containers
ssh <ssh_host> "docker ps -a | grep vllm"

# 2. If stopped container exists, start and reuse it
ssh <ssh_host> "docker start <container_name>"

# 3. If no container exists, create one
ssh <ssh_host> "docker run -d \
  --name hygon-vllm-<version> \
  --network host \
  --device /dev/dri \
  --device /dev/kfd \
  -v /data:/data \
  --shm-size 64g \
  --group-add video \
  <hygon_vllm_image> \
  sleep infinity"

# 4. Verify devices inside container
ssh <ssh_host> "docker exec <container> hy-smi"
```

Device occupancy check for Hygon DCU:
```bash
ssh <ssh_host> "docker exec <container> hy-smi | grep -E 'Used|Proc'"
```

### Adding a New Backend

Copy the MetaX template above and replace:
- `--device` flags with the backend's device node paths
- Volume mounts for any vendor-specific SDK paths
- The device occupancy check command (`mx-smi` → backend equivalent)

---

## Installation Steps

### Step 1: Check pyproject.toml for pinned vLLM version

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cat /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/pyproject.toml \
   | grep -A3 vllm'"
```

Record the version: `memory_write('<backend>_vllm_pinned_version', 'X.Y.Z')`

### Step 2: Install vLLM CPU-only (from source)

Install from source to ensure `VLLM_TARGET_DEVICE=empty` takes effect correctly.
Pre-built PyPI wheels may bundle device-specific compiled extensions that conflict
with the plugin's hardware backend.

```bash
# Clone the pinned vLLM version
ssh <ssh_host> "docker exec <container> bash -c \
  'cd /workspace/adapt/<backend>-vllm-<version> && \
   git clone https://github.com/vllm-project/vllm.git vllm-src && \
   cd vllm-src && git checkout v<pinned_version> && \
   git log -1 --oneline'"

# Install from source with empty device target
ssh <ssh_host> "docker exec <container> bash -c \
  'cd /workspace/adapt/<backend>-vllm-<version>/vllm-src && \
   VLLM_TARGET_DEVICE=empty pip install -e . \
   --extra-index-url https://download.pytorch.org/whl/cpu \
   2>&1 | tee /workspace/adapt-logs/install_vllm.log'"
```

> **Why source install?** `VLLM_TARGET_DEVICE=empty` must be set at *build time* when
> installing from source so that no CUDA/hardware C extensions are compiled. Installing
> a pre-built wheel from PyPI may silently include compiled ops that break on non-NVIDIA
> backends. Source install with this flag produces a pure-Python, device-agnostic vLLM
> that the plugin can override completely.

### Step 3: Clone vllm-plugin-FL (fresh workspace)

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'mkdir -p /workspace/adapt/<backend>-vllm-<version> && \
   cd /workspace/adapt/<backend>-vllm-<version> && \
   git clone https://github.com/flagos-ai/vllm-plugin-FL.git && \
   cd vllm-plugin-FL && git log -1 --oneline'"
```

### Step 4: Install plugin in editable mode

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL && \
   pip install -e . 2>&1 | tee /workspace/adapt-logs/install_plugin.log'"
```

### Step 5: Install FlagGems

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'cd /workspace/adapt/<backend>-vllm-<version> && \
   git clone https://github.com/FlagOpen/FlagGems.git && \
   cd FlagGems && pip install -e . \
   2>&1 | tee /workspace/adapt-logs/install_flaggems.log'"
```

> **MetaX note**: FlagGems requires `GEMS_VENDOR=metax` at runtime. The C extension
> (`cmake`) is not supported on MetaX — skip cmake build errors, they are non-fatal.

### Step 6: Sync local edits to container (development workflow)

When editing plugin source locally, sync before testing:

```bash
# Sync a single file
scp ./vllm_fl/models/my_model.py <ssh_host>:/workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/vllm_fl/models/

# Sync entire plugin directory
rsync -avz --exclude='.git' \
  ./vllm-plugin-FL/ \
  <ssh_host>:/workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL/
```

### Step 7: Verify installation

```bash
ssh <ssh_host> "docker exec <container> python3 -c \
  \"import vllm; print(f'vLLM {vllm.__version__}')\" && \
  docker exec <container> python3 -c \
  \"import vllm_fl; print('Plugin loaded:', vllm_fl.__file__)\" && \
  docker exec <container> python3 -c \
  \"import flag_gems; print('FlagGems loaded')\" && \
  docker exec <container> python3 -c \
  \"import torch; print(f'torch {torch.__version__}, devices: {torch.cuda.device_count()}')\""
```

All four imports must succeed before proceeding to `infer-hw-adapt` or `infer-model-adapt`.

### Step 8: Create adapt-logs directory

```bash
ssh <ssh_host> "docker exec <container> mkdir -p /workspace/adapt-logs"
```

This directory is used by `infer-hw-adapt` to store test and inference logs.

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `VLLM_PLUGINS=fl` | Activate the FL plugin | Required for all tests |
| `VLLM_TARGET_DEVICE=empty` | CPU-only vLLM install | Only during pip install |
| `MODEL_PATH` | Model weights location | `/workspace/models/Qwen3-8B` |
| `TP_SIZE` | Tensor parallel size | `2` |
| `PP_SIZE` | Pipeline parallel size | `1` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | Timeout for model execute call; increase for long graph capture | `7200` (Moore Threads graph mode) |
| `GEMS_VENDOR` | FlagGems hardware vendor | `metax` (MetaX), `hygon` (Hygon DCU), `ascend` (Ascend) |
| `PYTORCH_ROCM_ARCH` | DCU GPU architecture for kernel compilation | `gfx936` (Hygon DCU only) |
| `ROCM_PATH` | triton hcu compiler search path | `/opt/dtk` (Hygon DCU only) |
| `AMDGCN_USE_BUFFER_OPS` | Disable buffer ops to work around clang 17 limitation | `0` (Hygon DCU only) |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | Extend model execution timeout — needed during graph capture on MUSA | `7200` (Moore Threads) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow models with very long max sequence length | `1` |

---

## Related Skills

- `infer-hw-adapt` — hardware adaptation testing, patching, and PR submission (use after environment is set up)
- `infer-model-adapt` — port a new model into vllm-plugin-FL (use after environment is set up)
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts
