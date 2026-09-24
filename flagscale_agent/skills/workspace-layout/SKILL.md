---
description: Standardized workspace directory layout and storage management for FlagScale
  projects. Covers environment setup paradigm (shared-storage mounts, one-build-many-nodes),
  shared storage detection, fixed paths for models/datasets/experiments/checkpoints/logs/conda
  envs, experiment isolation (never overwrite), disk space pre-checks, path-consistency
  (symlinks, per-host trees), and artifact deduplication via memory.
name: workspace-layout
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

# Workspace Layout & Storage Management

This skill defines the standard directory layout, environment setup paradigm, and
storage management rules for all FlagScale projects. Follow these rules whenever
creating, downloading, or referencing artifacts (models, datasets, checkpoints,
logs, conda envs), and whenever setting up a new environment or syncing one across
hosts.

All persistent state discovered or decided under this skill is recorded with the
memory tools (`memory_write` / `memory_read` / `memory_list`) using the standard
key format `type/domain/specific` (e.g. `fact/env/workspace_root`). There is no
separate workspace-state tool.

---

## Step 0: Environment Setup Paradigm (multi-node)

**Goal: build ONCE on shared storage so every node sees the same environment —
never sync node-by-node.** Node-by-node copying (e.g. `docker cp` to each host)
is the root cause of three recurring failure classes: files missed on some hosts,
extra files on others, and silent version drift between hosts.

### 0a. Before creating containers: mount shared storage

When you are asked to create containers / set up an environment on a machine
(especially multi-node):

1. **Ask which directory to mount** if not specified — the shared-storage mount
   (NFS/Lustre/GPFS/Ceph) that all nodes can reach.
2. **Mount it into every container identically** (same source, same target path
   on every node). A path that exists on only one host is a per-host convenience,
   not an environment.
3. **Everything that must be identical across nodes lives under that mount**:
   conda envs (`--prefix <root>/envs/...`), source trees, pip installs done to
   shared prefixes, confpacks, datasets, models, checkpoints.
4. **Host-local paths are for nothing environment-critical.** A container-private
   path (visible to one container only) can never be referenced by another host;
   cross-host references to it require an explicit copy step with verification
   (checksum after transfer).

### 0b. Verify the paradigm held: two-tree check

After any container/env setup, verify ONCE per node that the shared path inside
the container is the same live tree as the one on the host (a stale copy that
*looks* like the shared path is a classic silent failure):

```bash
# inside container — the path the job will use
md5sum <shared_root>/code/<project>/<sentinel_file>
# on host — the path you will edit/build in
md5sum <host_path>/<sentinel_file>   # must match
```

If they differ, the container is not seeing the shared mount — fix the mount
(docker `-v` source must be the shared storage or a host-visible path), do not
patch files to match.

### 0c. Installed source code: never edit in place

- Framework code that came from git → manage with git (clone → edit → build →
  install). Loop through the repo, not through installed copies.
- Installed site-packages / runtime code → **never edit directly**. In-place
  edits to installed packages are silently ignored by some loaders and drift
  from every other node. If an installed copy truly must change: back it up,
  change it, record the exact file+change in memory immediately
  (`memory_write(key="pitfall/env/...", type="pitfall", content="...")`), and treat the change as temporary
  until it is upstreamed to the source repo.
- After ANY code change that should affect training: verify the running process
  actually picks it up (e.g. `python -c "import <pkg>; print(<pkg>.__file__)"`,
  `pip show <pkg> | grep Location`) before trusting a launch.

### 0d. Symlink and mount-point discipline

- If a path turns out to be a **symlink**, immediately `readlink -f` it to the
  real target and decide which layer created it (host convenience / container
  writable layer / image). Cross-host and cross-container scripts must use the
  **real absolute target path**, never the symlink — the symlink exists only on
  the host (or container) that created it.
- Same-shaped paths on different hosts may be **different filesystems**: before
  reusing "the same" path on another node, compare identity (device/UUID suffix
  of the mount), not just the path string.
- A `docker -v` mount source must live on shared storage or a host-visible path.
  Paths private to the agent container cannot serve as mount sources; move the
  payload to shared storage first.

### 0e. Record the environment

After setup, write one memory entry per host-class (`fact/env/...`) containing:
container name/ID, mounted shared root, conda env prefix, key package versions.
This is what makes the next session's Step 1 cheap and prevents re-derivation.

---

## Step 1: Detect Storage Root

Run once per session. If a memory key for the workspace root already exists and
the path is still valid (`memory_read` it), skip detection.

**User override**: If the user specifies a custom root or custom paths, always
respect their choice. Present the auto-detected recommendation first, then confirm
with the user before proceeding. Record the user's choice in memory
(`memory_write(key="fact/env/workspace_root", type="fact", ...)`).

### 1a. Identify shared storage

```bash
df -hT 2>/dev/null
mount | grep -iE 'type (nfs|lustre|gpfs|ceph|fuse\.ceph|beegfs|panfs|cifs)' 2>/dev/null
```

Shared storage = NFS, Lustre, GPFS, Ceph, BeeGFS, CIFS, or FUSE-based network mounts.

If `topo_storage` exists in memory (written by the topo-detect skill), use the
shared mount recorded there directly — no need to re-detect.

### 1b. Choose root

Priority order:
1. **Shared storage mount** — required for multi-node training. All nodes must see the same path.
2. **Largest persistent volume** — if no shared storage, pick the mount with the most free space (excluding tmpfs/overlay).
3. **`/workspace`** — last resort fallback.

### 1c. Confirm with user

Before proceeding, present the detected root and layout to the user:
- "Detected shared storage at `/mnt/shared` (NFS, 2TB free). Will use it as workspace root. Artifacts will go under `/mnt/shared/models/`, `/mnt/shared/experiments/`, etc. OK or do you prefer a different path?"
- If the user specifies a different path, use that instead.

### 1d. Record

Save the chosen root to memory:
- `memory_write(key="fact/env/workspace_root", type="fact", content="value: <chosen_path>\napplies: this project\nverify cmd: df <chosen_path>")`

---

## Step 2: Standard Directory Layout

All artifacts go under `<root>` (the detected storage root):

```
<root>/
├── models/            # pretrained weights — <org>/<model_name>/, read-only after download
├── datasets/          # processed training data — <dataset_name>/ (Megatron .bin/.idx or webdataset .tar)
├── code/              # cloned repos & custom code — <project_name>/, git-managed
├── experiments/       # <model>/<exp_name>/ — one dir per experiment, NEVER reuse
└── envs/              # conda environments — <env_name>/, created with --prefix
```

### Path examples

| Artifact | Path |
|----------|------|
| FlagScale source code | `<root>/code/FlagScale/` |
| Qwen2.5-7B-Instruct weights | `<root>/models/Qwen/Qwen2.5-7B-Instruct/` |
| SigLIP vision encoder | `<root>/models/google/siglip-so400m-patch14-384/` |
| Processed Megatron data | `<root>/datasets/pile-10k/` |
| Training experiment | `<root>/experiments/qwen3_0.6b/tp2_pp1_dp4_bs8/` |
| Conda environment | `<root>/envs/flagscale/` |

---

## Step 3: Rules for Each Artifact Type

### 3a. Model weights

- **Before downloading**: check standard path, `~/.cache/huggingface/hub/`, and memory (`memory_list(keyword=...)`) for existing copies. List what's found and what's missing.
- **Confirm with user**: show a table — model name, estimated size, target path. Wait for approval before downloading.
- **Download method**: `snapshot_download(repo_id, local_dir=<root>/models/<org>/<model>)` for consistent paths.
- **After downloading**: `memory_write(key="fact/env/model_<name>_path", type="fact", content="value: <path>\napplies: weights for <name>\nverify cmd: ls <path>")` so future sessions find it without re-downloading.
- **Read-only**: never modify downloaded weights in place. Checkpoint conversion outputs go to a separate path.

### 3b. Datasets

- Same confirm-before-download rule as model weights for files > 1GB.
- Use `<root>/datasets/<name>/` as `data_path` prefix in training configs.
- Record path in memory after creation (`fact/env/dataset_<name>_path`).

### 3c. Experiments

- **Isolation is non-negotiable**: each experiment gets its own directory. NEVER overwrite or reuse a previous experiment directory.
- Naming: use descriptive names reflecting the config (e.g., `qwen3_0.6b_tp2_pp1_bs8`). For reruns of the same config, append a timestamp (e.g., `qwen3_0.6b_tp2_pp1_bs8_20260429`).
- When generating FlagScale `train.yaml`: set `experiment.exp_dir` to `<root>/experiments/<model>/<exp_name>/`.
- Checkpoints, logs, and tensorboard dirs are subdirectories of the experiment — don't scatter them elsewhere.
- **Experiment registry via memory**: record every experiment as a memory entry
  (e.g. `fact/env/exp_<name>` with purpose, hypothesis, config summary, dir, and
  on completion, result + reflection). This is the log-discovery channel: find any
  experiment's directory instantly via `memory_read` instead of filesystem search.
  For the full launch-and-track lifecycle see the train-run skill.

### 3d. Conda environments

- Create with fixed prefix: `conda create --prefix <root>/envs/<env_name> python=<version>`
- Execute with: `conda run --prefix <root>/envs/<env_name> <command>`
- This makes environments discoverable and consistent across sessions. Never use auto-generated env names.
- Record env path in memory after creation (`fact/env/env_<name>_prefix`).

### 3e. Code repositories

- Clone to `<root>/code/<project>/` for repos that need to persist across sessions (shared storage, so all nodes compile against the same tree — see Step 0).
- Working directory code that already exists elsewhere stays where it is — don't move it.

---

## Step 4: Disk Space Pre-check

Before any large operation, verify sufficient space on the target path.

### 4a. Before downloading

```bash
df -h <target_directory>
```

Estimate total download size. Warn if free space < 1.5× estimated size. If insufficient, suggest:
1. A different mount with more space
2. Cleaning up old artifacts
3. Let user decide

### 4b. Before training

Estimate storage needs:
- **Checkpoint size** ≈ `param_count × 2 bytes` (BF16) per checkpoint
- **Total checkpoint storage** ≈ `ckpt_size × (total_steps / save_interval)`
- **Logs + TensorBoard** ≈ 1-5 GB depending on duration
- **Optimizer states** (if saved) ≈ `param_count × 8 bytes` per checkpoint

Warn if free space < estimated total. For long training runs, also warn about checkpoint accumulation.

---

## Step 5: Artifact Discovery

### CRITICAL: Workspace Isolation Principle

**NEVER search, reference, or use artifacts from other users' directories.** Even if you can see other projects at sibling paths (e.g., `/share/project/other_user/FlagScale`), treat them as off-limits. Reasons:
- Using another user's repo risks breaking their environment (editable installs, dirty state)
- Their code may be at a different version, with local patches or uncommitted changes
- It creates invisible dependencies — if they delete or move their directory, your setup breaks
- Reading their configs/code to "learn" the structure still leaks assumptions that may be wrong for your version

**The only valid search scope is YOUR workspace root** (i.e., `<root>/` as determined in Step 1). If an artifact doesn't exist under your root, the correct action is to **download/clone it fresh**, not to search the filesystem for someone else's copy.

### Discovery procedure

Before creating or downloading anything:

1. Check memory for previously recorded paths (`memory_list(keyword=...)`, `memory_read(key="fact/env/...")`)
2. Check standard paths under YOUR `<root>/` only (e.g., `<root>/code/FlagScale`, `<root>/models/...`)
3. Check `~/.cache/huggingface/hub/` for model weights (this is user-local, safe to reuse)
4. List what was found and what's missing
5. Only proceed to download/create what's actually missing

**DO NOT**:
- Run `find /share/project -name "FlagScale"` or similar broad searches
- Look at sibling directories under `/share/project/`
- Reference paths you saw in other users' directories
- Use `ls /share/project/` to discover what's available

After creating or downloading:
- Record the path in memory with a descriptive key (e.g., `fact/env/model_qwen25_7b_path`, `fact/env/env_flagscale_prefix`)
