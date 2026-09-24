---
description: Sync Megatron-LM-FL fork with an upstream NVIDIA/Megatron-LM release, plus
  the co-equal FlagScale training-code sync. Version-agnostic (base/target refs),
  evidence-gated. Covers patch-based fork integration (not git merge), plugin integrity
  (cur_platform platform mechanism, @overridable/@override override mechanism,
  dualpipev/hetero features, override-body drift), stale-ref fixes, assertion-based
  build/import/unit-test gates, tree-replacement merge to main, three-way training sync
  (diff=main-base), and precision alignment. Distilled from flagos-ai PR#42 SOP fused
  with Megatron-LM-FL core/parallel/training knowledge.
name: mg-fl-upstream-sync
parameters:
  - name: base
    description: upstream ref the fork is currently based on (e.g. core_v0.16.1 tip or SHA)
    default: <BASE_REF>
  - name: target
    description: upstream ref to sync to (e.g. core_v0.17.0 tip or SHA)
    default: <TARGET_REF>
  - name: mg_fl_dir
    description: path to Megatron-LM-FL checkout
    default: <MG_FL_DIR>
  - name: flagscale_dir
    description: path to FlagScale checkout (for Track B; empty until needed)
    default: <FLAGSCALE_DIR>
---

## What this skill does

Bring the FlagScale fork `flagos-ai/Megatron-LM-FL` up to a new upstream
`NVIDIA/Megatron-LM` release, then sync FlagScale's own training/legacy code to match.
A full upgrade has two co-equal tracks:

- **Track A — library** (`megatron.core` + `megatron.plugin`): the fork's value. Preserve
  the plugin system while merging upstream. Only these two packages are pip-installed.
- **Track B — training** (`FlagScale/flagscale/train/megatron/`): derived from upstream,
  carries FlagScale customizations. Resolved at runtime via PYTHONPATH, not pip.

Three refs drive everything — nothing hardcoded to a version:

- `base` — upstream commit the fork last synced from (common ancestor).
- `target` — upstream commit/tag to sync up to.
- fork delta to preserve = `base..main`; upstream delta = `base..target`.

The job: replay the fork delta onto `target` without dropping the plugin system, the
`FlagScale Begin/End` feature blocks, or fork features (dualpipev/hetero/engram), and
without regressing upstream fixes.

## Fork architecture (what makes Megatron-LM-FL special)

Megatron-LM-FL = upstream Megatron-LM + modified core + modified plugin. Enhancements,
all at the Python level, live in three mechanisms you must protect:

1. **Platform mechanism** — `megatron/plugin/platform/`. Every `torch.cuda.*` call in
   `megatron/core/` and `megatron/plugin/` is replaced with `cur_platform.*` so one
   codebase runs on NVIDIA/MetaX/Hygon/etc. Import pattern:
   `from megatron.plugin.platform import get_platform; cur_platform = get_platform()`.
2. **Override mechanism** — `megatron/plugin/decorators/`. Selected core functions are
   decorated `@overridable`; alternative impls live in `megatron/plugin/` with `@override`.
   Runtime swap without editing core.
3. **New features** — `megatron/plugin/dualpipev/`, `megatron/plugin/hetero/`, engram, etc.

Plus `FlagScale Begin/End` blocks: standalone fork additions (config dataclass fields like
`qk_layernorm_hidden_dim`, `use_partial_reduce_for_shared_embedding`; conditional branches;
imports; test-skip markers) scattered inside `megatron/core/` files. NOT cur_platform, NOT
@overridable — easy to silently drop.

**Install scope**: pyproject.toml packages `find.include` must contain exactly
`megatron.core`, `megatron.core.*`, `megatron.plugin`, `megatron.plugin.*` (match the
`target` version's scope, not upstream main). `megatron.training/legacy/rl/post_training`
exist in the repo but are NOT pip-installed — they resolve via PYTHONPATH downstream.

## Critical Rules

1. **Version-agnostic** — derive everything from `base`/`target`; never hardcode a release
   number or date in commands or decisions.
2. **Probe, never assume paths** — confirm a file/dir exists (`git ls-tree <ref> -- <path>`
   or `ls`) before referencing it. Grep for the real symbol before claiming it moved.
3. **Patch-based integration, NOT `git merge`** — a direct merge of divergent histories
   produces spurious conflicts. Apply the fork delta as categorized per-file patches onto a
   clean `dev` branch cut from `target`. Each conflict then means something.
4. **Plugin files are sacred** — never auto-resolve a P0 conflict toward upstream. Keep the
   fork version, manually integrate upstream changes.
5. **cur_platform must survive** — if upstream added new `torch.cuda` calls in core, convert
   them to `cur_platform`. Post-merge scan must be zero (Gate G2).
6. **@overridable/@override must survive AND stay in sync** — preserve decorators; then run
   the override-body drift check (Stage 4d). This is the most dangerous regression class:
   passes syntax + decorator checks, fails only at runtime under specific conditions.
7. **FlagScale Begin/End blocks are never P2** — before accepting an upstream file, run
   `git show main:"$f" | grep -c "FlagScale Begin"`; if >0 you must re-apply the blocks.
8. **Evidence gate** — every stage writes artifacts to a persistent dir
   `$ART` (NOT `/tmp`; `/tmp` is wiped and un-auditable). A stage is done only when its
   artifact exists and a pass/fail assertion was inspected.
9. **Gates are assertions, not prose** — "success criteria" must return pass/fail
   (decorator counts, torch.cuda residue count, import smoke, parsed pytest summary), never
   "looks fine".
10. **Two tracks, fix in the right repo** — core/plugin ImportError/AttributeError → fix in
    Megatron-LM-FL, reinstall, retest. Training/config/loop bugs → fix in FlagScale. Never
    shim a core issue inside FlagScale; never bend core to a FlagScale-specific pattern.
11. **Branch pairing is mandatory** — baseline = main+main; comparison = dev+dev-train.
    Verify both repos' branches before every run; a mismatch masquerades as a real bug.
12. **Rollback always ready** — `git revert -m 1 <merge-commit>`.

---

## Stage 0: Orientation and ref resolution

Resolve refs, set the artifact dir, add upstream remote. Never proceed until this is clean.

```bash
MG_FL_DIR="{mg_fl_dir}"; cd "$MG_FL_DIR"
git remote -v | grep -q upstream || git remote add upstream https://github.com/NVIDIA/Megatron-LM.git
git fetch upstream --tags --prune
# Resolve to concrete SHAs so nothing shifts under us
BASE=$(git rev-parse "{base}");   echo "base=$BASE"
TARGET=$(git rev-parse "{target}"); echo "target=$TARGET"
ART="$MG_FL_DIR/../mg-fl-sync-{target}"; mkdir -p "$ART"; echo "artifacts -> $ART"
git rev-parse --abbrev-ref HEAD   # confirm fork main is checked out
```

Write `$ART/00_refs.txt` with base/target SHAs, date, `git log --oneline -1` of each.
Gate G0: both SHAs resolve, upstream fetched, main is the fork tip.

## Stage 1: Classify the fork delta

Cut a clean `dev` branch from `target` and catalogue what the fork adds, by category. This
is the map for the whole merge.

```bash
DEV=dev-{target}; BASEB=base-{target}
git checkout -b "$BASEB" "$BASE"      # upstream base, no fork changes
git checkout -b "$DEV" "$TARGET"      # upstream target, no fork changes
git checkout main

D="$BASEB..main"   # fork delta = what fork changed vs its upstream base
git diff --name-only --diff-filter=A $D | grep '^megatron/plugin/'  > "$ART/A_new_plugin.txt"
git diff --name-only --diff-filter=A $D | grep -v '^megatron/plugin/'> "$ART/A2_new_other.txt"
git diff --name-only --diff-filter=M $D | grep '^megatron/core/'    > "$ART/B_mod_core.txt"
git diff --name-only --diff-filter=M $D | grep '^megatron/plugin/'  > "$ART/C_mod_plugin.txt"
git diff --name-only --diff-filter=D $D                             > "$ART/D_deleted.txt"
git diff --name-only --diff-filter=M $D | grep -Ev '^megatron/(core|plugin)/' > "$ART/E_mod_other.txt"
grep -rln "FlagScale Begin" megatron/core/ --include='*.py' | sort -u > "$ART/F_flagscale_blocks.txt"
grep -rln "@overridable" megatron/core/ --include='*.py' | sort -u > "$ART/overridable_files.txt"
wc -l "$ART"/[A-F]*.txt "$ART/overridable_files.txt"
```

Write `$ART/01_plugin_changes.md`: per-category counts + the full `base..main` diff saved to
`$ART/fork_full.patch`. Gate G1: every category file exists, counts reviewed, nothing in an
unexpected bucket (e.g. a core file that is neither B nor F when you expected a fork change).

## Stage 2: Patch-based integration onto dev

Apply fork changes in layers onto `$DEV`. Resolve one category at a time; commit per category.

**Category A/A2 (new files — zero conflict):** copy from main.
```bash
git checkout "$DEV"
while read -r f; do [ -z "$f" ] && continue; mkdir -p "$(dirname "$f")"
  git show main:"$f" > "$f"; git add "$f"; done < "$ART/A_new_plugin.txt"
while read -r f; do [ -z "$f" ] && continue
  git show "$DEV:$f" >/dev/null 2>&1 || { mkdir -p "$(dirname "$f")"; git show main:"$f" > "$f"; git add "$f"; }
done < "$ART/A2_new_other.txt"
git commit -m "sync: add new plugin + fork-specific files" --allow-empty
```

**Category B/C/E (modified — per-file 3-way patch):**
```bash
CONFLICTS=""
for LIST in B_mod_core C_mod_plugin E_mod_other; do
  while read -r f; do [ -z "$f" ] && continue
    P="$ART/patch_$(echo "$f" | tr '/' '_').patch"
    git diff "$BASEB..main" -- "$f" > "$P"; [ -s "$P" ] || continue
    if git apply --3way "$P" 2>/dev/null; then git add "$f"; echo "OK  $f"
    else echo "CONFLICT $f"; CONFLICTS="$CONFLICTS $f"; fi
  done < "$ART/${LIST}.txt"
done
echo "$CONFLICTS" | tr ' ' '\n' | grep -v '^$' > "$ART/conflicts.txt"
```

Resolve each conflicted file by priority (write decisions to `$ART/02_decisions.md`):

| Prio | Identify | Strategy |
|------|----------|----------|
| P0   | has `@overridable` or listed decorated in 01_plugin_changes | Start from `$DEV` (upstream). Re-apply in order: (1) `from megatron.plugin` imports, (2) `@overridable` decorators, (3) `cur_platform` replacements, (4) any FlagScale Begin/End blocks. Verify the `@override` in plugin still matches the new signature. |
| P0-F | has FlagScale Begin/End (in F list), no @overridable | Start from `$DEV`. Re-apply each Begin/End block onto the new upstream code; adapt to changed context (renamed vars, new signatures). Do NOT copy main's whole file. |
| P1   | cur_platform-modified only | Start from `$DEV`. Add cur_platform import; mechanically replace each `torch.cuda.*` → `cur_platform.*`. |
| P2   | not in 01_plugin_changes AND `grep -c "FlagScale Begin"` on main == 0 | Accept upstream: `git checkout "$DEV" -- "$f"`. |

**Never P2 a file with FlagScale Begin blocks.** Check first, every time.
Commit: `git commit -m "sync: apply fork core/plugin/other patches"`.
Gate G2 (partial): `git grep -n '<<<<<<<' -- megatron/` returns nothing.

## Stage 3: Plugin integrity verification (assertion gates)

All four are pass/fail. Write results to `$ART/03_plugin_integrity.md`.

```bash
git checkout "$DEV"
# G3a platform: zero raw torch.cuda in core (excluding cur_platform lines)
RES=$(grep -rn "torch\.cuda" megatron/core/ --include='*.py' | grep -v __pycache__ | grep -v cur_platform)
[ -z "$RES" ] && echo "G3a PASS" || { echo "G3a FAIL"; echo "$RES"; }
# G3b override: every @overridable has a matching @override, counts vs pre-merge
grep -rn "@overridable" megatron/core/ --include='*.py' | grep -v __pycache__ | wc -l
grep -rn "@override"    megatron/plugin/ --include='*.py' | grep -v __pycache__ | wc -l
# G3c features: fork modules parse
python3 -c "import ast; ast.parse(open('megatron/plugin/dualpipev/__init__.py').read())" && echo "dualpipev OK"
python3 -c "import ast; ast.parse(open('megatron/plugin/hetero/__init__.py').read())"    && echo "hetero OK"
```

**G3d override-body drift** (the dangerous one). `@override` impls are usually full copies of
the base body + fork logic. When upstream changes the base function (new branch, new param,
bug fix), the override silently misses it. For each `@overridable` function:
1. read the `$DEV` (new upstream) body; read the `@override` body in plugin.
2. `git diff "$BASEB..$DEV" -- <core_file>` to see exactly what upstream changed.
3. verify the override incorporates it. Drift → fix per this table, preserving fork parts:

| Drift | Fix |
|-------|-----|
| new feature branch in base | add same branch to override, adapt for hetero/multi-group |
| bug fix in base | apply same fix to override's copy |
| new parameter in base | add to override signature; forward/handle it |
| refactored base logic | re-copy base logic, then re-apply fork mods |

Example danger: upstream adds MTP support to `_allreduce_embedding_grad`; a stale override
silently drops MTP in hetero mode. Passes every static check.
Gate G3: G3a empty, G3b counts consistent with 01_plugin_changes, G3c both OK, G3d every
override reconciled (list each in 03 report).

## Stage 4: Stale references

Upstream may rename/move/remove symbols the fork depends on.
```bash
git diff "$BASEB..$DEV" -- '*.py' | grep -E '^-(def |class )' \
  | sed -E 's/^-(def|class) ([a-zA-Z_]+).*/\2/' | sort -u > "$ART/removed_symbols.txt"
while read -r s; do [ -z "$s" ] && continue
  M=$(grep -rn "\b$s\b" megatron/plugin/ --include='*.py' 2>/dev/null)
  [ -n "$M" ] && { echo "STALE $s"; echo "$M"; }
done < "$ART/removed_symbols.txt" | tee "$ART/04_stale.txt"
```
For each: find the new name in upstream, update fork code. Gate G4: `04_stale.txt` shows no
unresolved STALE lines. Commit if fixes were made.

## Stage 5: Build & import + native unit tests

```bash
# G5a syntax: all core+plugin parse
find megatron/core megatron/plugin -name '*.py' ! -path '*__pycache__*' \
  -exec python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" {} \; 2>"$ART/05_syntax_err.txt"
[ -s "$ART/05_syntax_err.txt" ] && echo "G5a FAIL" || echo "G5a PASS"
# G5b pyproject scope
python3 -c "import tomllib; c=tomllib.load(open('pyproject.toml','rb'));\
 inc=c['tool']['setuptools']['packages']['find']['include'];\
 need={'megatron.core','megatron.core.*','megatron.plugin','megatron.plugin.*'};\
 assert need<=set(inc), need-set(inc); print('G5b PASS', inc)"
# G5c install + import
pip install -e . --no-build-isolation 2>&1 | tail -5
python3 -c "import megatron.core; import megatron.plugin; \
 from megatron.plugin.platform import get_platform; \
 print('G5c PASS', getattr(megatron.core,'__version__','N/A'), get_platform())"
# G5d native unit tests (subset; parse the summary line)
pytest tests/unit_tests/ -q --timeout=120 -k "not distributed" 2>&1 | tee "$ART/05_pytest.txt" | tail -20
```
Gate G5: G5a no syntax errors, G5b scope exact, G5c imports pass, G5d pytest summary parsed
(record passed/failed/skipped in `$ART/05_build.md`). Failures on non-plugin code → a file
wasn't applied cleanly, revisit Stage 2. Failures on plugin code → Stage 3.

## Stage 6: Tree-replacement merge to main

`$DEV` is a superset of main (all fork features re-integrated). Do NOT use `-X theirs` — it
keeps both sides' non-conflicting additions and produces duplicate imports / duplicate
FlagScale blocks (PR#42 reports this bug hit 72+ files). Replace the tree instead:

```bash
git checkout main && git checkout -b merge-upstream-{target}
git merge -s ours "$DEV" --no-edit      # record both parents, keep main's tree
git read-tree -m -u "$DEV"              # replace tree with dev's content
git commit --amend --no-edit
git diff "$DEV" HEAD                     # MUST be empty (G6a)
git grep -n '<<<<<<<' -- megatron/ tests/ .github/   # MUST be empty (G6b)
git log --oneline --graph -5             # both parents present (G6c)
pip install -e . 2>&1 | tail -3; python3 -c "import megatron.core; import megatron.plugin; print('G6d PASS')"
```
Gate G6: G6a empty diff, G6b no markers, G6c two parents, G6d imports. Push branch, open PR.
Write `$ART/06_sync_report.md`: base→target, per-stage pass/fail, plugin status, merge SHA,
rollback command `git revert -m 1 <sha>`.

---

## Track B: FlagScale training sync (Stages 7–10)

Only start after Track A merged and `import megatron.core`/`megatron.plugin` work. FlagScale
keeps its own copy of training/legacy under `flagscale/train/megatron/` with FlagScale-only
changes. Same three-way idea, different repo.

**Three-way, not direct compare.** `diff = FS_main - base` isolates FlagScale's
customizations; apply that diff onto `dev` (= upstream training at `target`). Direct
FlagScale-vs-target comparison is WRONG: it cannot tell "upstream removed this" from
"FlagScale added this". Canonical case: `--config-logger-dir` came from upstream, not
FlagScale; when upstream removes it in dev it must stay removed (it is not in the fork diff).

### Stage 7: FlagScale branch setup
```bash
FS="{flagscale_dir}"; cd "$FS"
# base_tr = upstream training at BASE version; dev_tr = upstream training at TARGET version
# FS main carries FlagScale customizations. Cut FS dev-train from dev_tr, then apply diff.
git checkout -b dev-train-{target}   # seed from upstream target training tree
```
Gate G7: FS repo present, `flagscale/train/megatron/` exists, branch created. Record in
`$ART/07_fs_setup.md`.

### Stage 8: Apply FlagScale customizations
Scope is ALL of `flagscale/train/megatron/**` that has an upstream counterpart — not just
`training/` and `legacy/`. Top-level files too: `train_gpt.py`↔`pretrain_gpt.py`,
`model_provider.py`, `gpt_builders.py`, etc. Preserve FlagScale-only additions (wrapped in
`######### FlagScale Begin/End ########` or clearly FlagScale features: `get_parallel_context()`,
`extra_valid`, `spiky_loss`, hetero args, engram builders). When upstream changed a signature,
added args, or reworked logic, update FlagScale's copy accordingly.

Common runtime symptom if skipped: `AttributeError: 'Namespace' object has no attribute 'xxx'`
(e.g. stale `args.moe_grouped_gemm` after upstream renamed/removed it), or `TypeError` on a
changed signature. Also audit `FlagScale/flagscale/models/**` and `train_*.py` for references
to changed core symbols. Gate G8: `git grep -n '<<<<<<<'` empty; FlagScale features enumerated
present in `$ART/08_fs_apply.md`.

### Stage 9: Verify — import + real training smoke
Static import checks miss runtime API drift; a short run is definitive.
```bash
# branch pairing check BEFORE every run
echo "FS=$(cd "$FS" && git branch --show-current)  MG=$(cd "$MG_FL_DIR" && git branch --show-current)"
# comparison run must be: FS=dev-train-*, MG=dev-*  (baseline: both main)
cd "$FS" && python run.py --config-path <yaml_dir> --config-name <yaml_name>  # train_iters small
```
Watch for **TE-FL signature drift**: upstream adding a param to a fused op TE-FL implements
(e.g. `fused_rope_backward` gaining a 9th arg) raises `TypeError: takes N positional args but
N+1 given`, only in backward, only on affected ranks — looks like a hang. Fix: update TE-FL
(preferred) or disable the fusion in the test YAML (`no_rope_fusion: true`) as a stopgap.
Gate G9: import clean, smoke run reaches training iterations without crash.

### Stage 10: Precision alignment + feature verification
Compare baseline (main+main) vs comparison (dev+dev-train) over a short run (e.g. 10 steps),
per validated config. Extract per-iter `lm loss`, `grad norm`, throughput, mem. Thresholds:
lm loss within ±5% relative, grad norm same order of magnitude, throughput ±10%. Divergence
beyond thresholds → STOP, investigate (changed default, reordered reduction, dropped scale,
missing FlagScale optimization) before declaring done.

Also verify FlagScale-specific features that standard GPT training doesn't exercise, reusing
CI configs + gold values: **hetero-train** (`hetero_pipeline_layer_split`,
`hetero_process_meshes`, `hetero_device_types`) and **engram** (`use_engram`,
`engram_vocab_size`, `engram_layer_ids`, `engram_embedding_parallel_size`). Engram note: its
embedding params have `allreduce=False` and need their own `engram_dp_group` + separate grad
buffers — without the DDP patch they land in `expert_parallel_buffers` and cause a KeyError in
`DistributedOptimizer`. Gate G10: alignment table within thresholds, hetero+engram gold
values matched. Write `$ART/10_alignment.md`.

---

## Artifact layout (`$ART = <workspace>/mg-fl-sync-<target>/`)
```
00_refs.txt              base/target SHAs, date
01_plugin_changes.md     per-category counts; fork_full.patch
02_decisions.md          per-conflict priority + resolution ledger
03_plugin_integrity.md   G3a-d results, override-drift reconciliation
04_stale.txt             stale-ref scan + fixes
05_build.md 05_pytest.txt pyproject scope, import, parsed pytest summary
06_sync_report.md         Track A per-stage table, merge SHA, rollback cmd
07..10_*.md              FlagScale track: setup, apply, verify, alignment
```

## Gate summary (all pass/fail, never prose)
G0 refs resolve · G1 categories complete · G2 no conflict markers · G3a no raw torch.cuda /
G3b decorator counts / G3c features parse / G3d overrides reconciled · G4 no stale refs ·
G5 syntax+scope+import+tests · G6 tree==dev, no dup, two parents, imports · G7 FS setup ·
G8 FS customizations applied · G9 import+smoke · G10 precision+feature alignment.

## Anti-patterns (why the plain SOP isn't enough)
- Using `git merge` for the fork delta → spurious conflicts. Use categorized per-file patches.
- Using `-X theirs` at Stage 6 → duplicate imports/blocks (72+ buggy files). Use tree replace.
- Accepting an upstream file that has FlagScale Begin blocks (silent feature loss).
- Trusting decorator-count/syntax checks alone → misses override-body drift (Stage 3d).
- Writing artifacts to `/tmp` → wiped, un-auditable. Use `$ART`.
- Declaring a gate "looks fine" → gates must be assertions with recorded output.
- Direct FlagScale-vs-target compare → cannot distinguish upstream deletion from FS addition.
