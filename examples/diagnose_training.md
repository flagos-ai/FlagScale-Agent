# Example: Diagnosing a Training Failure

This example shows how FlagScale-Agent systematically diagnoses a training issue.

## Scenario
Training was running fine for 5000 iterations, then suddenly the loss spiked and training became unstable.

## Conversation

### Initial Report

```
User: 训练跑到5000步的时候loss突然从2.1飙到15，现在一直震荡不收敛。
      实验目录是 outputs/qwen2_7b_pretrain/
```

### Agent Diagnosis Process

Agent will execute the following diagnostic chain:

**1. Check training metrics:**
```
→ parse_training_metrics(log_path="outputs/qwen2_7b_pretrain/", last_n=500)

Results:
  iter 4990: loss=2.12, grad_norm=0.45
  iter 4995: loss=2.11, grad_norm=0.43
  iter 5000: loss=2.13, grad_norm=0.44
  iter 5005: loss=8.72, grad_norm=125.3   ← spike!
  iter 5010: loss=14.8, grad_norm=892.1
  iter 5015: loss=12.3, grad_norm=45.6
```

**2. Check all rank stderr for errors:**
```
→ find_latest_log(experiment="outputs/qwen2_7b_pretrain/", filter="errors")

Results:
  No CUDA OOM, no NCCL errors, no exceptions
```

**3. Analyze gradient norm pattern:**
```
The grad norm spike from 0.44 → 125.3 at iter 5005 suggests a data anomaly
or learning rate schedule issue.
```

**4. Check learning rate at the spike:**
```
→ shell("grep 'iteration 500[0-9]' outputs/qwen2_7b_pretrain/.../stdout.log | grep 'lr'")

Results:
  iter 5000: lr=1.5e-4 (warmup complete, peak reached, decay begins)
  iter 5005: lr=1.49e-4
```

**5. Check data at the failing iteration:**
```
→ shell("python -c 'from megatron.data import ... ; check_sample(5000)'")

Results:
  Sample at iter 5000-5010: normal text data, no corruption detected
```

**6. Root cause identified:**
```
The spike coincides exactly with the warmup→decay transition at iter 5000.
The peak learning rate 1.5e-4 is too high for a 7B model — the model was
barely stable during warmup, and once the scheduler hit peak LR at iter 5000,
gradient explosion occurred.
```

### Agent Recommendation

```
Root cause: Peak learning rate too high (1.5e-4) for 7B model scale.
The model was marginally stable during warmup and exploded at peak LR.

Recommendations:
1. Reduce peak learning rate from 1.5e-4 to 6.0e-5
2. Enable gradient clipping (clip_grad=1.0) as a safety net
3. Increase lr_warmup_iters from 5000 to 8000 for smoother ramp
4. Resume from checkpoint at iter 4900 (before the spike)

Shall I update the config and restart from the checkpoint?
```

---

## Diagnostic Chain Summary

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `parse_training_metrics` | Identify exact iteration of failure |
| 2 | `find_latest_log` (errors) | Rule out infrastructure issues |
| 3 | Analysis | Correlate grad norm with loss spike |
| 4 | `shell` (grep) | Check LR schedule transition |
| 5 | `shell` (data check) | Rule out data corruption |
| 6 | Reasoning | Connect evidence to root cause |

## Key Takeaways
- Agent follows systematic elimination: infrastructure → data → hyperparameters
- Multi-rank log scanning catches errors that don't appear on rank 0
- Gradient norm is the earliest indicator of instability (precedes loss spike)
- Agent proposes actionable fixes, not just diagnosis
