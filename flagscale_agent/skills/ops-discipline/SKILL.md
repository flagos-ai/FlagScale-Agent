---
name: ops-discipline
description: General operational discipline for FlagScale infrastructure work. Covers reading strategy, shell safety, environment
  awareness, and root cause diagnosis patterns. For training-specific operations, use train-run skill.
keywords:
- shell
- install
- dependency
- environment
- setup
- debug
- diagnosis
- safety
requires: []
suggests:
- debug-strategy
constraints:
- id: no_modify_third_party
  description: Never modify third-party or framework source code to work around build errors
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - site-packages
    - megatron-lm
    - transformer-engine
    - apex
    - flash-attn
  prompt: Check if the agent is modifying third-party framework source code instead of fixing the integration
  correction: Fix your own code or configuration. If the framework has a bug, report it.
- id: no_repeat_command
  description: Never run the same shell command twice in a row expecting different results
  trigger:
    tools:
    - shell
  prompt: Check if this shell command is identical to the immediately preceding shell command
  correction: Try a different diagnostic command or change the approach.
- id: read_before_write
  description: Read docs/code before implementing
  trigger:
    keywords:
    - implement new
    - write new
    - create new
    - build from scratch
  prompt: Check if the agent is writing code without having read relevant documentation or source first
  correction: 'Reading strategy: understand before implementing. Read docs, example configs, and source code BEFORE writing.'
- id: disk_space_check
  description: Check disk space before large operations
  trigger:
    keywords:
    - pip install
    - conda install
    - git clone
    - wget http
    - curl http
    - huggingface download
    - snapshot_download
  prompt: Check if a large download/install is about to happen without prior disk space check
  correction: Check disk space with `df -h` before large downloads or builds.
context_injection:
  always:
  - Reading strategy — depth over speed
  by_tool:
    shell:
    - Shell command rules
    - Environment awareness
    - Fail-fast preflight
    edit_file:
    - Root cause diagnosis
    write_file:
    - Root cause diagnosis
---
# Operational Discipline

General operational rules for infrastructure work. The system prompt covers principles; this skill covers execution details.

---

## Reading strategy — depth over speed

- **Understand before implementing.** For complex tasks, read docs, example configs, and source code BEFORE writing anything.
- **Read complete files, not fragments.** One complete read beats ten partial reads.
- **First read: full file.** Note key line numbers. Subsequent reads: targeted ranges.
- **Record key findings in memory_write** so they survive context compaction.
- **Never re-read a file you read in the last 5 turns** unless it was modified.
- **Breadth matters:** for a training config, read at least: the getting-started doc, an existing example config, and the model's source code.

---

## Shell command rules

- Prefer `grep -rn "pattern" . --include="*.py"` for code search.
- Use `head`/`tail` ONLY for quick previewing. Never truncate error logs you need to diagnose.
- NEVER run the same command twice in a row. If results are unclear, try a DIFFERENT diagnostic.
- NEVER modify third-party source code to work around build errors.
- For large downloads: `wget -c` or `curl -C -`. Execute as SEPARATE commands, not combined with `&&`.
- After any download, verify with `ls -lh <file>`.
- Download speed < 500 KB/s for multi-GB file → check proxy, then STOP and ask user.

---

## Environment awareness

- FIRST thing on any new server: `nvidia-smi`, `cat /etc/os-release`, `which conda`, `echo $CUDA_HOME`. Save to memory.
- Check disk space (`df -h`) before large downloads or builds.
- Check GPU memory (`nvidia-smi`) before launching training.

---

## FlagScale log structure

```
outputs/<exp>/logs/details/host_<N>_<hostname>/<timestamp>/<run_id>/attempt_<N>/<rank>/
  ├── stdout.log   (training metrics, progress)
  └── stderr.log   (errors, warnings, stack traces)
```

**Critical: ALWAYS check stderr.log after launch.** Training can appear "running" while crashing on rank > 0. Use `monitor(output_dir=...)` to auto-scan all stderr files.

---

## Root cause diagnosis

- dtype mismatches (fp32 in bf16 pipelines) are architecture-level. Trace dtype from source rather than adding `.to(dtype)` at error site.
- Cascading TypeError/AttributeError on module init → read the COMPLETE base class API, fix ALL mismatches at once.
- Before calling any base class method, read its IMPLEMENTATION, not just signature.

---

## Fail-fast preflight

Before operations >30 seconds:
- **Model loading**: verify state_dict keys/shapes match BEFORE loading to GPU
- **Checkpoint conversion**: compare key counts/shapes between source and target
- **Training launch**: validate config arithmetic, verify ALL dependencies importable
- **Memory budget**: `params × 2 (bf16) + grads × 2 + optimizer × (8/DP)` — if exceeds GPU memory, don't launch
- **Config arithmetic**: `global_batch_size % (micro_batch_size × DP) == 0`, `num_heads % TP == 0`
