---
name: train-ascend-performance-tuning
description: Use when optimizing FlagScale Ascend training for capacity or throughput. Choose or combine candidates across memory, compute efficiency, and communication; experiment, compare, and retest within the budget.
---

<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# Ascend Training Performance Tuning

Core loop: `confirm the goal → establish a baseline → choose optimization hypotheses → compare experiments → update the assessment → retest and deliver`.
Consider opportunities in memory, compute, and communication each round. They interact; there is no fixed order or requirement to exhaust each category.
Read only this file by default. Load `train-run` once at the start, and read a method only after selecting it. For principles, constraints, or trade-offs, query `ascend_training/search-space.md` in `know-ascend-training` as needed; do not preload the entire Knowledge group or every method.

## 1. Confirm the goal and comparison conditions

Read the user-provided configuration, command, and existing logs. Reuse the confirmed devices, budget, and goal (capacity, throughput, or an agreed trade-off).
Keep the model, data and data order, sequence length, GBS, precision, optimizer math, and training starting point fixed. Use resources within the authorized scope. Parallel layout, MBS, recomputation, and implementation choices may be searched unless the user has fixed them.
Agree on warmup, timing window, numerical tolerance, minimum gain, and maximum retest variation. The last two are distinct: a small gain is not automatically invalid just because it falls below the variation limit.
Reuse one plan and experiment record. Preserve the original configuration; the actual paths and versions of all three repositories; the launch cwd and command; and each run's configuration, logs, and result location. Preserve existing modifications.
Keep full logs and configurations in files; serialize updates to the shared experiment record. When adding step verification through `plan_update`, retain earlier valid entries because the supplied verification array replaces the previous one.
When resuming a session, first inspect the plan, record, and owned jobs. Do not start training or recreate a completed plan when only checking progress or analyzing existing results.

## 2. Establish a baseline

For the first training run, call `load_skill(name="train-run")` and select its Ascend path. Always launch with `flagscale train -c /absolute/path/to/config.yaml`; the file may have any name.
For an eligible bounded single-node run, use that skill's [bounded single run](../train-run/references/single-run.md) instructions to launch and wait. Otherwise follow its CLI path.
Reuse a confirmed environment and a valid baseline under the same conditions. The baseline short run serves as validation; do not add a separate single-device smoke test.
Check complete updates, loss/gradient health, skipped steps or NaNs, stable timing, and evidence that workers exited. Record peak memory if directly available; leave unavailable metrics unknown.
If the baseline OOMs, use the failure stage to pursue memory capacity and first find a feasible configuration. Fix other launch or measurement failures before tuning. Use the original OOM configuration only as a capacity reference; do not calculate a speedup against it.
If a repair changes data processing, the numerical path, or framework implementation, record the difference and establish a new comparable baseline. Calculate gains against that baseline and explicitly state when training-semantic equivalence has not been verified.
The launcher enforces the per-run deadline. A wait timeout does not mean training stopped; confirm that all owned workers exited before the next run.

## 3. Choose directions and generate candidates

When selecting candidates for the first time, use evidence across all three directions to propose and retain, where possible, three worthwhile hypotheses with different mechanisms in the existing experiment record. Different MBS values alone do not count as different mechanisms. Three is a default starting point: use fewer if evidence or feasible methods are lacking, or more if the budget allows. There is no need to choose one per category or run every candidate.
For each hypothesis, briefly record `id / hypothesis / supporting and opposing evidence / next experiment and expected observation / status (pending, exploring, deferred, closed)`. Link candidates and results to that ID. Update affected entries only when selecting an experiment or receiving a key result. Keep untried alternatives without creating another record system or duplicate analysis reports.
The three directions describe optimization goals; methods are organized by the concrete adjustment. A method can serve several goals. Read only relevant methods after selecting them. This table suggests candidates; it is not an allowlist or a fixed order.

| Direction | What to observe | Known options and procedure |
| --- | --- | --- |
| Memory | OOM stage, peak memory on the most loaded rank, persistent state and activations, available headroom | [MBS](methods/micro-batch.md), [activation storage/recomputation/offload](methods/recompute.md), [parallel layout and state sharding](methods/parallelism.md); read [allocator settings](references/allocator.md) only with evidence of fragmentation |
| Compute efficiency | Complete-update time, repeated compute, small operations/dispatch and data stalls, pipeline bubbles, load imbalance | [MBS](methods/micro-batch.md), [reduce recomputation](methods/recompute.md), [data and Host scheduling](methods/data-and-host.md), [graph capture and compilation](methods/graph-execution.md), [PP/VPP and layer assignment](methods/parallelism.md), [operator and backend selection/implementation](methods/operator.md) |
| Communication overhead | Communication waits on the critical path, rank skew, communication groups and intra-/inter-node topology | [gradient synchronization/parameter gather/P2P overlap and buckets](methods/communication.md), [parallel degrees/group mapping/scheduling](methods/parallelism.md) |

When choosing the next experiment, state which hypotheses it should test or distinguish and what observations would support or weaken them. Rank options by potential gain, ability to change the current assessment, evidence strength, and experiment cost. If the evidence is sufficient, test the configuration directly instead of repeatedly gathering confirmation for the first guess.
When the cause is unclear, a result is unexpected, or the next experiment is costly, load `train-ascend-profiling` as needed and collect the smallest trace that answers the question. Measure performance in a separate run without a profiler.
With a tight budget, favor cheap experiments that distinguish hypotheses. With more time, broaden layout and parameter ranges, test other directions and combinations, and leave room for plausible but uncertain options. Reserve time for the final retest; if time runs out, deliver the verified result.
You may propose options outside the table. State their mechanism, expected benefit and cost, current implementation support, and verification approach. The user's goal, resources, and allowed modification scope still bound the search.
Start from the current best configuration or another useful parent configuration. Every method adds only candidate actions and special checks; the following rules are shared:

- Record a candidate as `parent / hypothesis / change / checks`. Express `change` as a configuration diff or proposed file/function edit; list reusable evidence and necessary follow-up checks under `checks`; link it to an existing hypothesis.
- After selection, copy a separate YAML and select it with `-c`, keeping dependencies and relative paths valid. For authorized code changes, make the smallest patch, preserve the parent version and full diff, and record the actual configuration/patch used. Preserve user changes.
- Use sections 4–5 for launch, wait, comparison, and adoption. A method only adds proof that it took effect, special metrics, and explanations for anomalies; it does not maintain a separate run or rollback process.

Single-change experiments make attribution easier. Combine changes when there is a clear dependency or cross-direction trade-off, such as "recomputation frees memory + larger MBS"; each switch need not improve speed by itself. Preserve the full diff and do not attribute the combined result to one switch.
Before launch, check the selected method's integer and scheduling constraints and the effective configuration. For changes to layout or state sharding, add their restore and numerical checks. Reuse confirmed mappings and capabilities; consult Knowledge or source for remaining gaps. If a field is rejected, rewritten, or lacks evidence of taking effect, inspect its consumer.

## 4. Experiment, compare, and update the search

Run one candidate at a time under the same comparison conditions using `train-run`, including the method's effective-path and quality checks. If a target setting did not take effect or a combination is invalid, mark it as an invalid experiment and fix the configuration first. Do not include it in the benefit comparison or treat it as a rejection of the direction.
For supported Megatron logs, follow [train-run's result analysis procedure](../train-run/references/result-analysis.md). Supply the correct loss-rank log and existing `exit_code_path`; set training `log_interval=1` and specify the iteration range, warmup_steps, GBS, sequence_length, loss tolerance, and JSON `output_path`.
Each comparison uses one valid baseline and one fixed candidate. Use the script summary and saved JSON; inspect detailed logs only for anomalies. `status="ok"` means only that parsing succeeded.
Compare complete-update time/throughput, peak memory, and quality. Record the configuration diff, evidence paths, and reason to keep or revert. Missing metrics are not passes.
Compare actual results with expected observations. Update the relevant hypotheses' supporting/opposing evidence, applicable configuration range, and next experiment. Leave an unclear result unresolved; multiple hypotheses can hold at once.
Keep the best verified configuration for the goal. You may also retain useful configurations such as one that is slower but saves memory for later combinations. Compare the current branch with unexplored hypotheses, then decide whether to continue, combine, or switch.
If repeated attempts produce neither gain nor new evidence, a key prediction is refuted, or the next step is not worth its cost, record why and defer that branch. Select again from the remaining hypotheses; do not fix a number of failures. When switching, derive the candidate from an explicit parent configuration without carrying over failed changes. Reopen a branch if new evidence or combination conditions warrant it.
OOM or lack of gain narrows only the tested conditions. You may revisit MBS after changing layout, recomputation, or buffers. Reject the current candidate on quality failure; record unknown capability separately from run failure. Do not generalize one failure to an entire direction.
Stop when the budget cannot cover another experiment plus required retests, the goal is met, or remaining hypotheses have insufficient expected value. Record unexplored directions; do not claim a global optimum.

## 5. Retest and deliver

For a throughput goal with no better candidate, deliver the original baseline and tested results. For a capacity goal, apply the agreed feasibility, peak-memory, and acceptable-speed-cost criteria. If the original baseline OOMed, retest only feasible candidates; do not rerun the failed baseline or calculate a speedup against it.
For the final throughput candidate, default to three pairs of independent retests: six fresh requests, each with a separate YAML, run_id, and exp_dir. The initial trials do not count. On the bounded single-node path, run the loop below with `shell(background=True)` and wait according to `train-run`. Adjust the list if a different count was agreed:

```bash
RETEST_REQUEST_DIR=/absolute/path/to/retest-requests
for run in pair1-baseline pair1-candidate pair2-candidate pair2-baseline pair3-baseline pair3-candidate; do
  PYTHONUNBUFFERED=1 python -m flagscale_agent.skills.train-run.scripts.training_trial \
    --request "${RETEST_REQUEST_DIR}/${run}.json" || exit $?
done
```

Compare only these fresh runs, passing logs and exit codes in their actual order. For other launch backends, run the same number of trials sequentially. Accept according to the agreed quality, gain, and variation criteria; mark the result pending verification if the budget or evidence is insufficient.
Deliver the best verified configuration, a reproducible `flagscale train -c ...` command, the experiment record, and a concise conclusion: adopted changes, gains and costs across the three directions, quality evidence, search scope, and unverified items.
Quote performance figures from the same final analysis summary without mixing measurement conventions. For mock data, state the model, sequence length, and synthetic-data scope. Healthy loss in a short run does not establish update equivalence or long-term convergence.
Keep the original configuration and reasons for rejecting candidates. Complete the current plan, attach evidence, review the report once, and finish.
