# Workspace-Layout — Summary

Standardized directory layout and storage management for FlagScale projects.

**Load when**: downloading models/data, creating conda environments, organizing experiment outputs, or before any operation that creates large files.

Detects shared storage (NFS/Lustre/GPFS/etc.) and uses it as workspace root. Fixed subdirectories: `models/`, `datasets/`, `experiments/`, `envs/`, `code/`. Conda envs use `--prefix <root>/envs/<name>`. Includes disk space pre-checks and experiment isolation rules (never overwrite existing experiment dirs).
