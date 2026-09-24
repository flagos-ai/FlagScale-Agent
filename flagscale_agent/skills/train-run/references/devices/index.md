<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Device references

Select from the actual assigned hardware and active training backend, then read only that file. These are instructions read with `read_file`, not separate skills or executable adapters. Paths below are relative to this index.

| Device family | Selection evidence | Reference |
| --- | --- | --- |
| NVIDIA GPU | Assigned NVIDIA hardware and the configured CUDA training backend | [NVIDIA](nvidia.md) |
| Ascend NPU | Assigned Ascend hardware and the configured CANN / torch_npu training backend | [Ascend](ascend.md) |

An API name, compatibility layer, installed CLI, or failed probe alone is insufficient to select a family. Reuse previously confirmed identity/mapping. If evidence conflicts, resolve that conflict before device-dependent execution; do not run every vendor's probes as a discovery sweep.

For an unlisted family, inspect only the actual environment's read-only inventory, framework backend and installed tool documentation. Keep unsupported probes and unverified support explicit. Configuration/log analysis may continue; launching requires a reliable authorized device mapping and supported runtime/launcher. Do not install a different vendor stack or silently fall back to CPU.

To add a device family, add one row and a matching reference with device mapping, runtime, monitoring, and diagnostic commands; keep the common execution workflow unchanged.
