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

# Workspace-Layout — Summary

Standardized directory layout, environment setup paradigm, and storage management for FlagScale projects.

**Load when**: setting up new containers/environments (especially multi-node), downloading models/data, creating conda environments, organizing experiment outputs, or before any operation that creates large files.

**Step 0 — Environment paradigm**: build ONCE on shared storage, never sync node-by-node. Mount the shared storage into every container identically; keep conda envs, source trees, confpacks on it; verify the in-container path is the same live tree as the host (two-tree checksum); never edit installed source in place (git loop instead, or backup+memory-record for unavoidable cases); use real absolute targets, never symlinks, in cross-host scripts; docker `-v` sources must be shared/host-visible.

**Layout**: detects shared storage (NFS/Lustre/GPFS/etc.) and uses it as workspace root. Fixed subdirectories: `models/`, `datasets/`, `experiments/`, `envs/`, `code/`. Conda envs use `--prefix <root>/envs/<name>`. All state is recorded via memory (`memory_write`, key format `type/domain/specific`, e.g. `fact/env/workspace_root`) — including the experiment registry. Includes disk space pre-checks, path-consistency discipline (symlinks, per-host trees), and experiment isolation rules (never overwrite existing experiment dirs).
