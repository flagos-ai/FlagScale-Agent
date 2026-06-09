---
name: infer-env-setup
description: Set up inference environment for vllm-plugin-FL on hardware backends. Covers SSH
  connection, Docker container creation, CPU-only vLLM install, plugin editable install,
  FlagGems install, and import verification. Use before infer-hw-adapt or infer-model-adapt.
keywords:
- inference
- vllm
- docker
- container
- install
- environment
- setup
- ssh
- flaggems
- plugin
requires:
- workspace-layout
suggests:
- infer-hw-adapt
- infer-model-adapt
- ops-discipline
constraints:
- id: no_host_install
  description: Never install inference packages (vLLM, plugin, FlagGems) directly on the host machine
  trigger:
    tools: [shell]
    keywords: [pip install vllm, pip install -e, pip install flag]
  prompt: Check if the agent is installing packages outside of a Docker container
  correction: All inference package installs must happen inside the Docker container via `docker exec`.
- id: cpu_only_vllm
  description: vLLM must be installed with VLLM_TARGET_DEVICE=empty (CPU-only)
  trigger:
    tools: [shell]
    keywords: [pip install vllm]
  prompt: Check if VLLM_TARGET_DEVICE=empty is set during vLLM install
  correction: The plugin provides the hardware backend. Install with VLLM_TARGET_DEVICE=empty.
- id: pin_vllm_version
  description: Always pin the exact vLLM version from plugin's pyproject.toml
  trigger:
    tools: [shell]
    keywords: [pip install vllm]
  prompt: Check if a specific vLLM version is pinned
  correction: Read pyproject.toml first, then install with `pip install vllm==X.Y.Z`.
- id: fresh_workspace
  description: Each adaptation task uses a fresh clone in its own directory
  trigger:
    keywords: [reuse, existing directory, same workspace]
  prompt: Check if the agent is reusing an existing workspace directory
  correction: Create a new adapt/<backend>-vllm-<version>/ directory for each task.
- id: network_host
  description: Docker containers on GPU machines must use --network host
  trigger:
    tools: [shell]
    keywords: [docker run, docker create]
  prompt: Check if --network host is included in docker run/create command
  correction: Add --network host to the docker command.
- id: check_container_first
  description: Check for existing containers before creating new ones
  trigger:
    keywords: [docker run, docker create]
  prompt: Check if the agent verified no suitable container already exists
  correction: Run `docker ps -a` first. Reuse running containers; start stopped ones.
context_injection:
  always:
  - Critical Rules
  - Remote Access
  by_tool:
    shell:
    - Container Setup
    - Environment Variables
---
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
# 1. Pull image (if not present)
ssh <ssh_host> "docker pull <metax-vllm-image>:<tag>"

# 2. Create container
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt \
  --network host \
  --device /dev/mxcd0 --device /dev/mxcd1 \
  -v /workspace:/workspace \
  <metax-vllm-image>:<tag> sleep infinity"

# 3. Verify devices inside container
ssh <ssh_host> "docker exec vllm_fl_adapt mx-smi"
```

### Ascend 910B

```bash
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt \
  --network host \
  --device /dev/davinci0 \
  -v /workspace:/workspace \
  <ascend-vllm-image>:<tag> sleep infinity"
```

### Moore Threads S4000

```bash
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt \
  --network host \
  --device /dev/musa0 \
  -v /workspace:/workspace \
  <mthreads-vllm-image>:<tag> sleep infinity"
```

### Adding a New Backend

Copy the MetaX template above; replace:
- `--device` with the backend's device nodes
- image name/tag with the backend's official vLLM image
- monitoring command (`mx-smi` → backend equivalent)

---
  echo '=== workspace layout ===' && ls -la /workspace/ 2>/dev/null | head -20"
```

This gives you: connectivity, hardware type/count/memory, device occupancy, disk space, existing containers, available images, and existing adapt dirs.

**After the probe, immediately record to memory:**

```
memory_write('<backend>_ssh_host', '<ssh_host>')
memory_write('<backend>_container_name', '<container_name>')
memory_write('<backend>_workspace_root', '/workspace/adapt/<backend>-vllm-<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
memory_write('<backend>_log_dir', '/workspace/adapt-logs')
```

**Never guess paths. Always read from memory or re-probe.**

If SSH fails, ask the user to check their `~/.ssh/config` or provide full connection details (host, port, user, key file).

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

### Moore Threads S4000

```bash
# Create container with MUSA device mounts
ssh <ssh_host> "docker run -d \
  --name vllm_fl_adapt_mt \
  --network host \
  --env MUSA_VISIBLE_DEVICES=all \
  -v /workspace:/workspace \
  --shm-size 64g \
  <mt_vllm_image> \
  sleep infinity"

# Verify MUSA devices
ssh <ssh_host> "docker exec <container> mthreads-gmi"
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

### Step 2: Install vLLM CPU-only

```bash
ssh <ssh_host> "docker exec <container> bash -c \
  'VLLM_TARGET_DEVICE=empty pip install vllm==<pinned_version> \
   --extra-index-url https://download.pytorch.org/whl/cpu \
   2>&1 | tee /workspace/adapt-logs/install_vllm.log'"
```

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
| `GEMS_VENDOR` | FlagGems hardware vendor | `metax` (MetaX only) |

---

## Related Skills

- `infer-hw-adapt` — hardware adaptation testing, patching, and PR submission
- `infer-model-adapt` — migrate a new model into vllm-plugin-FL
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts

---
Related skills (load if needed): `ops-discipline`
| `GEMS_VENDOR` | FlagGems hardware vendor | `metax` (MetaX), `ascend` (Ascend) |

---

## Related Skills

- `infer-hw-adapt` — hardware adaptation testing, patching, and PR submission (use after environment is set up)
- `infer-model-adapt` — port a new model into vllm-plugin-FL (use after environment is set up)
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts

---
Related skills (load if needed): `ops-discipline`
