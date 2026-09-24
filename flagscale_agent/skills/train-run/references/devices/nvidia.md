<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA device branch

## Applicability

Use for assigned NVIDIA GPUs with the configured CUDA training backend. A CUDA-compatible API on another vendor's device is not sufficient evidence. Follow the common `train-run` workflow; read this file once per unchanged environment.

## Devices and mapping

Use `shell` for read-only device probes:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

Preserve existing `CUDA_VISIBLE_DEVICES` and launcher mapping. Framework logical indices may be reordered or restricted relative to `nvidia-smi`; correlate index/UUID with the assigned devices, and inspect the configured partition mapping if applicable. Query each assigned host for multi-node runs. Reuse the model/count/mapping for topology freshness checks.

Combine utilization, memory baseline and process information with allocation evidence. Container-visible PIDs may be incomplete. For unsupported query fields, check installed `nvidia-smi --help-query-gpu`; do not reset devices or require memory to be exactly zero.

## Runtime

Run in the existing training environment on first use or after a relevant change:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__, torch.__file__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Visible GPUs:', torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count():
    print('First visible GPU:', torch.cuda.get_device_name(0))
import megatron, transformer_engine
print('Megatron:', megatron.__file__)
print('TransformerEngine:', transformer_engine.__file__)
"
```

Verify the actual framework backend and existing CUDA/NCCL environment. Check `apex` or `flash_attn` only when this recipe's CUDA path consumes them. A version string alone does not verify kernel or distributed execution; reuse actual launch evidence where available.

## Monitoring

Use shared `flagscale_train_monitor(mode="check")` for logs and `nvidia-smi` for device state. The current monitor's `watch` device probe uses `nvidia-smi`, so bounded watch is applicable here. It also uses broad process matching: this cannot establish task ownership or completion. Keep the returned job/PID and actual worker-exit evidence.

## Failure diagnostics

- **CUDA/import/device error:** inspect the first exception, active module paths, assigned mapping and driver/runtime compatibility before repairing dependencies.
- **NCCL error:** inspect the first rank error, rank connectivity and existing network settings such as `NCCL_SOCKET_IFNAME`; preserve the workload's configured backend.
- **CUDA out of memory:** inspect assigned-device memory and owned processes. In fixed-workload tuning, return the failed candidate and memory evidence to the caller; do not change parallelism or recomputation here.

## Optional profiling

Only for an explicit profiling task, use installed `torch.profiler` with CUDA activities or Nsight tools. Routine launch and MBS trials do not trigger profiling.
