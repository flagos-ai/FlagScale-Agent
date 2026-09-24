---
description: General operational discipline for FlagScale infrastructure work. Covers
  reading strategy, shell safety, environment awareness, remote execution posture,
  pitfall recall, and root cause diagnosis patterns. For training-specific operations,
  use train-run skill.
name: ops-discipline
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

# Operational Discipline

General operational rules for infrastructure work. The system prompt covers principles; this skill covers execution details.

---

## Pitfall recall — before the error, not after

A non-zero pitfall count in a memory domain means this exact failure class already
happened and cost real debugging rounds. Do not re-pay that cost.

- **Before touching a fresh domain** (new server, docker, launch, build, deploy):
  `memory_read(key='pitfall/<domain>/')` — one call, whole domain.
- **Before executing a known-risky operation**: pull 2-3 DISTINCTIVE words from the
  command you are about to run (tool name, flag, file type) and
  `memory_list(keyword='<word>')`. Distinctive beats broad: `docker` alone is noise;
  `tar stdin` or `heredoc docker exec` is signal.
- **On a hit, read the original entry** (`memory_read` the exact key) — do not act
  from the title alone. Adjust the action FIRST; if you proceed anyway, state why
  the recalled pitfall does not apply.
- **After debugging that took >2 rounds**: write the pitfall immediately
  (symptom / cause / fix / env). The next session depends on it.

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
- **Never issue multiple tool calls that read-and-write the SAME object in one parallel
  batch.** Each call works from the same pre-batch snapshot and the last writer silently
  overwrites the others — all calls report success, only one edit survives. Same-file
  edits MUST be sequential (one edit_file per response) or merged into one atomic call.
  (Reads are safe to batch.)
- **Overridable guards are advisors, not obstacles.** When a guard blocks, read the
  message and address the concern; if it genuinely does not apply, override with a
  reason naming the CONCRETE anchor (observed value, re-tested premise) — never a bare
  "it's fine". A risky action re-issued to bypass a guard is the pitfall recall you skipped.

---

## Remote execution posture (ssh → docker)

The standard target is a container reachable only through a host. Rules learned from
real failures:

1. **docker exec has no stdin by default.** `ssh host "docker exec ctr bash -s" < script`
   silently delivers NOTHING (exec without `-i` closes stdin) — the command "runs" and
   does nothing. Always pass `-i`.
2. **Nested `bash -c` eats quoting.** A command nested inside `ssh "docker exec ctr bash
   -c '...'"` can lose its argument list entirely. Prefer direct form:
   `ssh host "docker exec -i ctr tar -xzf - -C /dst" < pack.tar`.
3. **Three checkpoints before trusting a remote change:**
   (a) file landed — `md5sum`/`ls -l` on host vs container;
   (b) content is the intended version — `grep` a known marker line in-container;
   (c) change took effect — re-run a cheap probe AFTER the change, not before.

Multi-node fan-out: loop with per-host timeout + per-host result log, then summarize ALL
host results before proceeding — one green host proves nothing about eight.

---

## Environment awareness

- FIRST thing on any new server: `nvidia-smi`, `cat /etc/os-release`, `which conda`, `echo $CUDA_HOME`. Save as `fact/env/*` memory.
- Check disk space (`df -h`) before large downloads or builds.
- Check GPU memory (`nvidia-smi`) before launching training.

---

## Root cause diagnosis

- dtype mismatches (fp32 in bf16 pipelines) are architecture-level. Trace dtype from source rather than adding `.to(dtype)` at error site.
- Cascading TypeError/AttributeError on module init → read the COMPLETE base class API, fix ALL mismatches at once.
- Before calling any base class method, read its IMPLEMENTATION, not just signature.
- Silent success is the worst failure class: a command that "runs" but changes nothing
  (empty stdin to exec, a no-op patch, a wrong output dir). Verify EFFECT, not exit code.

---

## Fail-fast preflight

Before operations >30 seconds:
- **Model loading**: verify state_dict keys/shapes match BEFORE loading to GPU
- **Checkpoint conversion**: compare key counts/shapes between source and target
- **Training launch**: validate config arithmetic, verify ALL dependencies importable
- **Memory budget**: `params × 2 (bf16) + grads × 2 + optimizer × (8/DP)` — if exceeds GPU memory, don't launch
- **Config arithmetic**: `global_batch_size % (micro_batch_size × DP) == 0`, `num_heads % TP == 0`

---

## Experiment ledger — HARD GATE

**Every training launch MUST be recorded. No exceptions. Not even "quick retries."**

The ledger lives in memory (fact/pitfall/insight) — not a special tool. Sequence:
open ledger → record attempt → launch → record result → repeat or close.

| When | Action | How |
|------|--------|-----|
| First time on a model/task line | Open the ledger | `memory_write(key='fact/agent/exp_<name>', type='fact')` with purpose / hypothesis / config dir |
| BEFORE every `flagscale train` | Record the attempt | Append to the same entry: what changes, config diff, hardware, output_dir. If you cannot articulate what is DIFFERENT from the last attempt, do not launch. |
| AFTER every result (success or crash) | Record result | Update the entry: result, key metrics, failure signature. A crash without a written signature will be re-paid. |
| Line of work concluded | Close it out | Fold durable lessons into `pitfall/*` and `insight/*` entries; mark the exp entry final. |

**Why this matters:**
- Rapid debug-fix-retry cycles are the HARDEST to reconstruct after the fact
- Without tracking, you repeat failed approaches because you forgot what you tried
- The change field forces you to articulate what's different — if you can't, you shouldn't launch
- Context compaction WILL eat working memory mid-campaign; the ledger is the only survivor

**Self-check:** about to call `flagscale train` and the ledger has no uncommitted attempt
record for this change → STOP and write it first.

**Irreversible git operations** (`reset --hard`, `checkout .`, `clean -fd`, branch
deletion): back up first — `git stash push -u -m <label>` or `git branch backup/<name>`.
The VcsBackupGuard enforces this at the moment; perform the ritual without being asked.
