# Parallel-Strategy — Summary

Select and configure parallelism strategies (TP/PP/DP/EP/CP/SP) for Megatron-LM-FL training.

**Load when**: choosing parallelism dimensions for a model, estimating memory requirements, or debugging parallelism-related errors (NCCL timeout, OOM, shape mismatch).

Decision flow: memory budget calculation → TP (fits in NVLink domain) → PP (if still OOM) → DP (remaining GPUs) → EP (MoE models) → CP (long sequences). Includes memory estimation formulas and hardware-aware placement rules.
