<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend device branch

## Applicability

Use for assigned Ascend NPUs with the configured CANN / torch_npu training backend. For FlagScale + Megatron-LM-FL + TransformerEngine-FL, preserve the actual three-repository environment. Follow the common `train-run` workflow; read this file once per unchanged environment.

## Devices and mapping

Use `shell` for read-only device probes:

```bash
npu-smi info
npu-smi info -m
# Use the mapped card/chip pair, not a torch.npu logical index:
npu-smi info -t usages -i <card_id> -c <chip_id>
# Only when memory/health needs investigation:
npu-smi info -t memory -i <card_id> -c <chip_id>
npu-smi info -t health -i <card_id> -c <chip_id>
```

Check installed `npu-smi info -h` if a query is unsupported. Card ID, chip ID, physical ID and framework-visible logical ID are different namespaces; map the authorized devices before querying them or choosing workers. Card count need not equal worker count. Preserve existing `ASCEND_RT_VISIBLE_DEVICES` and launcher mapping; reuse inventory plus framework count for topology freshness.

Combine utilization, HBM baseline and process information with allocation evidence. Idle devices may retain runtime HBM; do not require zero memory. Container-visible PIDs may be incomplete, so no visible PID does not prove availability. A failed probe remains unknown; do not reset devices.

## Runtime

Run in the existing training environment on first use or after a relevant change:

```bash
python -c "
import torch, torch_npu
print('PyTorch:', torch.__version__, torch.__file__)
print('torch_npu:', torch_npu.__version__, torch_npu.__file__)
print('NPU available:', torch.npu.is_available())
print('Visible NPUs:', torch.npu.device_count())
import megatron, transformer_engine
print('Megatron:', megatron.__file__)
print('TransformerEngine:', transformer_engine.__file__)
"
```

Verify existing CANN/torch_npu/HCCL compatibility and the actual TE-FL backend selection. Do not replace these with CUDA/NCCL settings or require CUDA-only Apex/FlashAttention packages. Missing `nvidia-smi` or `torch.version.cuda is None` is not an Ascend failure. Imports/count checks do not prove distributed execution; reuse actual launch evidence where available.

## Monitoring

Use shared `flagscale_train_monitor(mode="check")` for logs, `npu-smi` for assigned-device state and the actual job/PID for waiting and completion. The current monitor's `watch` device probe still calls `nvidia-smi` and uses broad process matching; do not use it as an NPU device monitor or owned-worker completion check. Do not pass an invented `device_type` parameter to the tool.

## Failure diagnostics

- **Import/device error:** inspect the first exception, configured CANN environment, actual torch_npu/TE-FL paths and assigned mapping before repairing dependencies.
- **HCCL error:** inspect the first rank error, mapped NPU IDs, rank connectivity and existing network configuration; check CANN/torch_npu/HCCL compatibility rather than applying NCCL-only flags.
- **NPU out of memory / HBM allocation failure:** use the mapped `usages` query and owned-process evidence. Return a failed candidate to the calling tuning workflow without changing its fixed workload or layout.

## Optional profiling

Only for an explicit profiling task, use `torch_npu.profiler` or installed `msprof`; load `train-ascend-profiling` when that workflow is needed. Routine launch and MBS trials do not trigger profiling.
