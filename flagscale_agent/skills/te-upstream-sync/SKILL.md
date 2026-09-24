---
description: Sync TransformerEngine-FL fork with an upstream NVIDIA/TransformerEngine
  release. Version-agnostic (base/fork/target refs), evidence-gated workflow covering
  merge conflict resolution, plugin API audit, runtime patch preservation, build/submodule
  integration, CI/CD audit, multi-backend test matrix, and finalize-to-main. Fuses
  V2 (PR#103) decomposed skeleton with V1 (PR#67) hard-won vendor field experience.
name: te-upstream-sync
parameters:
  - name: base
    description: upstream ref the fork last synced from (e.g. release_v2.x)
    default: <BASE_REF>
  - name: fork
    description: current FlagScale fork HEAD (e.g. origin/main)
    default: <FORK_REF>
  - name: target
    description: upstream ref to sync to (e.g. release_v2.y or a tag)
    default: <TARGET_REF>
---

## What this skill does

Sync the FlagScale fork of TransformerEngine (TransformerEngine-FL) up to a new
upstream NVIDIA release while preserving all FlagScale multi-backend plugin work.
Three refs drive everything -- nothing is hardcoded to a specific version:

- `base` -- the upstream commit the fork last merged from (the common ancestor).
- `fork` -- current fork HEAD carrying FlagScale customizations.
- `target` -- the upstream commit/tag to sync up to.

The delta the fork must preserve = `base..fork`. The delta upstream added =
`base..target`. The job is to replay the fork delta onto target without dropping
FlagScale plugin backends or regressing upstream fixes.

## Critical Rules

1. **Version-agnostic** -- never hardcode a release number or date in commands or
   decisions. Everything is derived from `base`/`fork`/`target` refs.
2. **Probe, never assume paths** -- before referencing any file/dir in a command,
   confirm it exists with `git ls-tree <ref> -- <path>` or `ls`. (V1 shipped a
   cicd doc + validate script referencing `common/plugin/*.h`, `common/cuda_patches/`,
   `tests/unit/` that never existed. See `pitfall/skill/doc_vs_reality_drift`.)
3. **Real layout** -- FlagScale plugin work lives in `transformer_engine/plugin/`:
   - `core/ops.py` -- plugin base op dispatch (Python).
   - `core/backends/vendor/<name>/` -- per-vendor backend implementations.
   - `core/backends/flagos/` and `core/backends/reference/` -- FlagOS + reference impls.
   C++ bindings live in `transformer_engine/pytorch/csrc/`.
4. **Dynamic backend discovery** -- enumerate backends from
   `transformer_engine/plugin/core/backends/vendor/` at runtime. Never paste a
   fixed vendor list into logic; the set changes over releases.
5. **Evidence gate** -- every stage emits artifacts to a persistent dir under
   `/share/.../temp/te-upstream-<target>/` (NOT `/tmp`). A stage is "done" only
   when its artifact exists and is inspected.
6. **P0 needs approval** -- dropping/altering any FlagScale backend op, or any
   change that could change numerics, is P0: record in decisions ledger and get
   sign-off before merging.
7. **Blocked != pass** -- a backend test that cannot run (no hardware) is BLOCKED,
   never counted as passing. Only NVIDIA CUDA is available as local ground truth.
8. **One conflict class at a time** -- resolve, build, verify, then next. Never
   batch unverified conflict resolutions.
9. **Squash before PR** -- final sync is one clean commit onto a new branch.
10. **Re-parent onto fork HEAD** -- the squash commit's parent MUST be `{fork}` HEAD,
    not the upstream target. Use `git commit-tree -p {fork}` (Stage 8). Skipping this
    causes GitHub to find an ancient merge-base and report hundreds of false conflicts
    on the PR. See `pitfall/te_upstream/pr_conflict_reparent_fix`.
11. **Test set is diff-driven, never handed-in** -- the tests that must run are
    derived from `{base}..{target}` (Stage 7b), independent of any list someone
    provides. Treat any handed-in list as a *candidate*, then reconcile it against
    the diff-derived expected set. "No regression" may NOT be concluded while the
    (expected − collected) difference set is non-empty and unwaived. This rule exists
    because a handed-in 4-file list once hid an 891-failure `test_fused_router.py`
    that the sync had directly broken. See `pitfall/te_fl/v217_routing_map_format_plugin_sync`.

---

## Stage 0: Orientation and ref resolution

Resolve all three refs to commits and set the artifact dir. Never proceed until
`base`, `fork`, `target` all resolve.

```bash
REPO=$(git -C . rev-parse --show-toplevel)
for r in {base} {fork} {target}; do
  git -C "$REPO" rev-parse --verify "$r^{commit}" || { echo "unresolved ref: $r"; exit 1; }
done
ART=/share/$(whoami)/flagos/temp/te-upstream-{target}
mkdir -p "$ART"
```

Record resolved commits to memory (`fact/te_upstream/refs`). Confirm the working
tree is clean before touching anything.

## Stage 1: Classify the fork delta

Compute what the fork changed on top of `base` and bucket every touched path into
preserve / adapt / drop. This is the decision ledger that drives the whole sync.

```bash
git -C "$REPO" diff --name-status {base} {fork} > "$ART/fork-delta.txt"
# dynamic backend discovery -- do NOT hardcode vendor names
git -C "$REPO" ls-tree -r --name-only {fork} \
  -- transformer_engine/plugin/core/backends/vendor/ \
  | awk -F/ 'NF>5 && $6!="__init__.py"{print $6}' | sort -u > "$ART/backends.txt"
```

For each changed path emit a row to `$ART/decisions.tsv` with columns:
`path  bucket(preserve|adapt|drop)  reason  owner  test  status  evidence`.
Plugin dir changes are almost always **preserve**; upstream files the fork edited
are **adapt**; anything the fork added that upstream now provides natively is a
**drop** candidate (P0 -- needs approval).

## Stage 2: Integrate upstream, resolve conflicts

Merge (or rebase-replay) `target` and resolve conflicts one class at a time.

```bash
git -C "$REPO" checkout -b sync/upstream-{target} {fork}
git -C "$REPO" merge --no-commit --no-ff {target} 2>&1 | tee "$ART/merge.log"
git -C "$REPO" diff --name-only --diff-filter=U > "$ART/conflicts.txt"
```

Resolve conflict files grouped by area (build, csrc, pytorch modules, plugin). For
each resolution note the class + rationale in `$ART/conflict-notes.md`. Keep the
plugin dir intact; when upstream refactors a base class the plugin subclasses,
adapt the subclass rather than reverting upstream.

## Stage 3: Audit the plugin API surface

Verify the FlagScale plugin ops still line up with upstream C++ bindings after the
merge. Three sets must stay consistent:

- **bindings** -- `.def("...")` names in `transformer_engine/pytorch/csrc/`.
- **plugin base** -- `def <name>` in `transformer_engine/plugin/core/ops.py`.
- **registered ops** -- `op_name = "..."` under `core/backends/`.

```bash
git -C "$REPO" grep -hE '\.def\(\s*"[A-Za-z0-9_]+' {target} \
  -- transformer_engine/pytorch/csrc/ | grep -oE '"[A-Za-z0-9_]+' | tr -d '"' | sort -u > "$ART/bindings.txt"
git -C "$REPO" grep -hE '^\s*def\s+\w+' {fork} \
  -- transformer_engine/plugin/core/ops.py | grep -oE 'def\s+\w+' | awk '{print $2}' | sort -u > "$ART/plugin-base.txt"
```

Build a matrix (`$ART/api-matrix.tsv`): `symbol  binding  plugin_base  registered  cuda_impl  disposition`.
A binding present upstream but missing in plugin base after refactor = adapt task.
A registered op whose binding vanished upstream = P0 investigation.

**Also run the reverse check** -- new bindings added in `target` that are absent from
`vendor/cuda/cuda.py` are implementation gaps. Stage 2 conflict resolution preserves
`cuda.py` from `{fork}` (correct), but does not automatically add methods for new
`target` bindings. These gaps cause silent `AttributeError` at dispatch time, not at import.

```bash
# New bindings in target not present in cuda.py
comm -23 \
  <(grep -hoE '\.def\("([A-Za-z0-9_]+)"' \
      "$REPO"/transformer_engine/pytorch/csrc/extensions/pybind.cpp | \
      grep -oE '"[A-Za-z0-9_]+"' | tr -d '"' | sort -u) \
  <(grep -oE '^\s*def ([A-Za-z0-9_]+)' \
      "$REPO"/transformer_engine/plugin/core/backends/vendor/cuda/cuda.py | \
      awk '{print $2}' | sort -u) \
  > "$ART/cuda-impl-gaps.txt"
cat "$ART/cuda-impl-gaps.txt"   # every line here = missing method in cuda.py
```

**Two false-positive classes to filter before treating a gap as real** (both
observed in the v2.17 sync -- the raw `comm -23` reported them but neither needs a
`cuda.py` method):

1. **C++ class-member `.def`s.** `.def("copy_into_buffer", &CommOverlap::copy_into_buffer)`
   binds a *method on a bound C++ class* (`CommOverlap`, `CommOverlapP2P`, ...), not a
   module-level op. These are reached as `obj.copy_into_buffer(...)` on the class
   instance the plugin already exposes (`TEFLModule.__init__` binds `self.CommOverlap =
   CommOverlap`), so they never route through op dispatch and need no wrapper. Detect
   them: a `.def(` whose second arg is `&Namespace::method` (pointer-to-member) rather
   than a lambda/free function. Grep the *surrounding* `.def` line, not just the name.
2. **Pure C++ query/util helpers** re-exported by the
   `NVTE_DECLARE_COMMON_PYBIND11_HANDLES` macro (`pybind_helper.h`) --
   e.g. `ubuf_built_with_mpi`, `device_supports_multicast`, `get_stream_priority_range`.
   Upstream code (`module/base.py`) calls these as `tex.<fn>()`, but the plugin's
   `TEFLModule.__getattr__` only resolves names registered as OpManager ops, so these
   raise `AttributeError: Operator '<fn>' not found` at *collection* time (breaks
   `tests/pytorch/distributed/test_comm_gemm_overlap.py` and
   `test_fusible_ops_with_userbuffers.py`). This is a pre-existing plugin gap, not a
   sync regression -- fix by explicitly passing these query fns through in
   `TEFLModule` (bind them in `__init__` or special-case in `__getattr__`), NOT by
   adding a `cuda.py` op. See `pitfall/te_fl/plugin_unregistered_util_funcs_comm_gemm`.

For each *remaining* real gap: implement the method in `cuda.py` following the existing
pattern (`tex = self._get_tex(); return tex.<binding>(...)`).

**Signature drift -- both added AND removed params.** Diff `pybind.cpp`
`{base}..{target}` for every `.def()` whose `py::arg` list changed. Upstream commonly
does BOTH in one release: append a new arg (e.g. `routing_map_format`) to some ops and
**remove** leading args from others (e.g. drop `num_tokens, num_experts` from `*_bwd`).
Reconcile each `cuda.py` method signature AND its call-through positional order against
the `py::arg` list -- a removed param left in the plugin signature raises
`TypeError: takes N positional args but M given` only when the op is *called*, never at
import, so unit tests for that op are the only thing that catches it (see Stage 7).

```bash
git -C "$REPO" diff {base} {target} -- \
  transformer_engine/pytorch/csrc/extensions/pybind.cpp \
  | grep -E '^[+-].*(m\.def|py::arg)' > "$ART/binding-arg-drift.txt"
```

**Enum / symbol re-exports (import-time crashers).** A Python module may re-export a
C++ enum via the `tex` proxy, e.g. `router.py`: `RoutingMapFormat = tex.NVTERoutingMapFormat`.
If upstream `{target}` added a new enum in a header and pybind, the FlagScale plugin's
`tex` proxy (`plugin/core/ops.py`) must (a) define a mirroring `IntEnum`, (b) bind it in
the proxy `__init__`, and (c) list it in `__dir__`. A missing enum makes the proxy's
`__getattr__` treat it as an *operator* lookup and raise `AttributeError` **at import of
the re-exporting module** -- so any test file that does not import that module will never
see it. Detect new enums and confirm each is mirrored:

```bash
# New enums pybind REGISTERS in target. Note two traps this command handles:
#   (1) the repo mixes `py::enum_<>` and `pybind11::enum_<>` spellings -- match both;
#   (2) enum types may carry a namespace (e.g. transformer_engine::pytorch::FP8FwdTensors)
#       -- strip it and keep only the class name.
# Scope to the registration file (pybind.cpp), NOT the whole csrc/ dir, otherwise
# every static_cast<Enum> use site becomes noise.
grep -hoE '(py|pybind11)::enum_<\s*[A-Za-z0-9_:]+' \
  "$REPO"/transformer_engine/pytorch/csrc/extensions/pybind.cpp \
  | sed -E 's/.*[<:]([A-Za-z0-9_]+)$/\1/' | sort -u > "$ART/pybind-enums.txt"
# Enums the plugin tex proxy mirrors. NOTE: `grep -oE 'class X(IntEnum)'` prints the
# whole match, so `awk '{print $2}'` would yield `X(IntEnum)` (with the paren suffix)
# and every enum would falsely show as a gap. Use sed to capture just the class name.
grep -oE 'class [A-Za-z0-9_]+\(IntEnum\)' \
  "$REPO"/transformer_engine/plugin/core/ops.py \
  | sed -E 's/class ([A-Za-z0-9_]+)\(IntEnum\)/\1/' | sort -u > "$ART/plugin-enums.txt"
comm -23 "$ART/pybind-enums.txt" "$ART/plugin-enums.txt" > "$ART/enum-gaps.txt"
cat "$ART/enum-gaps.txt"   # each line = enum to mirror in plugin/core/ops.py
```

## Stage 4: Preserve runtime patches

FlagScale carries runtime shims that adapt upstream code for non-NVIDIA backends
(e.g. lazy `transformer_engine_torch` alias resolution, staged imports so a missing
vendor extension doesn't break import). After merge, re-verify these still apply.

- Diff the fork's runtime-patch touchpoints (`git diff {base} {fork} --
  transformer_engine/pytorch/`) and confirm each still lands cleanly on target.
- Confirm `import transformer_engine` succeeds with no vendor extension built
  (CPU-only), proving the staged-import guard survived the merge.
- Any patch that no longer applies because upstream changed the target lines is an
  **adapt** row -- re-derive the shim against the new upstream code, do not force
  the old diff.

## Stage 5: Integrate build and submodules

Reconcile build system and the three submodules with target.

```bash
git -C "$REPO" diff {base} {target} -- .gitmodules setup.py build_tools/ > "$ART/build-delta.txt"
git -C "$REPO" submodule status > "$ART/submodule-status.txt"
```

Discover submodules dynamically -- do not hardcode the list (v2.17 added
`3rdparty/nccl` for EP support, earlier versions did not have it):

```bash
git -C "$REPO" submodule status | awk '{print $2}' | sort > "$ART/submodule-list.txt"
```

Update each to the commit target expects; record old->new SHAs in the ledger. A
cudnn-frontend or cutlass bump can change kernel availability -- flag for the test
matrix.

## Stage 6: Audit CI/CD (added in V2 -- V1 predates CI/CD)

Verify GitHub Actions still cover every discovered backend and contain no broken
local references or silent failure masks.

```bash
ls "$REPO/.github/workflows"/all_tests_*.yml
ls "$REPO/.github/configs"/*.yml
# broken local action references
git -C "$REPO" grep -hoE 'uses:\s*\./[^ #]+' HEAD -- .github/workflows/ | sort -u > "$ART/local-uses.txt"
# silent failure masks
git -C "$REPO" grep -nE '\|\|\s*true|continue-on-error:\s*true' HEAD -- .github/workflows/ > "$ART/failure-masks.txt"
```

Cross-check each backend in `$ART/backends.txt` against a matching workflow
(`all_tests_<backend>.yml`) and config (`.github/configs/<backend>.yml`). A backend
with a plugin dir but no workflow = coverage gap (record it). Any `|| true` /
`continue-on-error: true` that hides a real test = flag. This stage exists because
V1's helper docs drifted from reality; here we detect drift automatically.

## Stage 7: Run the multi-backend test matrix

Only NVIDIA CUDA runs locally as ground truth. Everything else is BLOCKED unless
the corresponding hardware/CI is reachable.

### 7a. Import smoke test (gate 0 -- runs before any functional test)

Import-time crashers (a Python module re-exporting an enum the plugin proxy never
bound -- see Stage 3) never surface in functional tests that don't import that
module. Catch them in seconds by importing every submodule (`{PY}` = the interpreter
of the env TE was built into, e.g. the build conda env's `python`; the bare host may
have only `python3` or none):

```bash
{PY} - <<'EOF'
import pkgutil, importlib, transformer_engine.pytorch as tep
# TWO traps, both verified against a real broken sync:
# (1) Collect ALL module names FIRST, THEN import in a separate loop. If you import
#     inside the walk_packages() loop, walk_packages itself imports each package to
#     recurse into it and SWALLOWS the exception (default onerror ignores), leaving
#     the module half-initialized so your own import_module() no longer re-raises ->
#     a broken module reports 0 failures. Two-phase is mandatory.
# (2) Exclude packaging/build-only modules (e.g. `setup`, anything importing
#     `build_tools`) -- they legitimately fail at runtime and are false positives.
SKIP = ("setup",)  # extend if the tree adds more build-only modules
names = [m.name for m in pkgutil.walk_packages(tep.__path__, tep.__name__ + ".",
                                               onerror=lambda n: None)]
bad = []
for n in names:
    if n.rsplit(".", 1)[-1] in SKIP:
        continue
    try:
        importlib.import_module(n)
    except Exception as e:
        bad.append((n, repr(e)))
for n, e in bad:
    print("IMPORT-FAIL", n, e[:100])
print("SMOKE total_fail", len(bad))
raise SystemExit(1 if bad else 0)
EOF
```

Any `IMPORT-FAIL` is a hard stop -- fix (usually a missing enum/op binding in
`plugin/core/ops.py`) before running the matrix. Verified: with the pre-fix
`ops.py` this prints `IMPORT-FAIL transformer_engine.pytorch.router ... AttributeError:
Operator 'NVTERoutingMapFormat' not found`; with the fix it prints `SMOKE total_fail 0`.

### 7b. Build the expected-test set from the diff (NOT from a handed-in list)

**The single most important discipline in this stage.** The set of tests to run is
derived from the sync diff, independently of any list someone hands you. A
subsystem upstream touched in `{base}..{target}` MUST have its test file run.

```bash
# subsystems upstream changed -> map to test files
git -C "$REPO" diff --name-only {base} {target} \
  -- transformer_engine/ > "$ART/upstream-touched.txt"
```

Map touched paths to test files (maintain this table in the ledger; extend as the
tree grows):

| touched path signal              | required test file            |
|----------------------------------|-------------------------------|
| `*fused_router*`, `router.py`    | `tests/pytorch/test_fused_router.py` |
| `*comm_gemm_overlap*`, `*comm_overlap*`, overlap factory | `tests/plugin/plugin/test_cuda_comm_overlap_contract.py`, `tests/plugin/plugin/test_vendor_comm_overlap_contract.py` (PR#115) |
| `*fused_attn*`, `attention.py`   | `tests/pytorch/test_fused_attn.py`   |
| `*gemm*`, `*quantize*`, `cublaslt*` | `test_fusible_ops.py`, `test_float8_blockwise_gemm_exact.py` |
| any transformer layer / norm     | `tests/pytorch/test_numerics.py`     |
| `*jit*`, `*onnx*`                 | `tests/pytorch/test_onnx_export.py`  |

Write the resulting required set to `$ART/expected-tests.txt`.

### 7c. Run, then reconcile expected vs actually-collected (mandatory)

```bash
{PY} -m pytest tests/pytorch/ --collect-only -q 2>&1 | tee "$ART/collected.txt"
{PY} -m pytest tests/pytorch/ -q 2>&1 | tee "$ART/test-cuda.log"
```

Reconcile `expected-tests.txt` against the files actually in `collected.txt`:

- **The difference set (expected − collected) MUST be empty, or each missing item
  MUST carry an explicit waiver reason** (e.g. "diff did not touch attention").
  A non-empty difference set with no waiver = you may NOT conclude "no regression".
  This is exactly the gap that hid `test_fused_router.py` when the test list was
  handed in rather than derived from the diff.
- Watch for **collection errors and files with 0 collected** -- an import crash
  shows up here (or is truncated away by `tail`), not as a normal failure line.
- Never accept a handed-in test list as complete. Treat it as a *candidate*
  actual-set and reconcile it against `expected-tests.txt` before drawing any
  conclusion.

### 7d. Classify and record

- For each non-CUDA backend, mark BLOCKED with the reason in the matrix; do not
  guess pass. If CI for that backend can be triggered, capture the run URL as
  evidence instead.
- Classify every failure by root cause: FL regression / toolchain limit (e.g.
  cuBLAS/CUDA version) / upstream-not-yet-adapted. Only FL regressions block the sync.
- Update `decisions.tsv` status column: pass / blocked / fail + evidence path.

The stage conclusion must be stated in a refutable form:
> "Impact set N subsystems; N test files collected and executed; difference set 0
> (or: M waived, reasons ...); X failures classified, 0 FL regressions."
An empty "difference set" field means the conclusion cannot be signed off.

## Stage 8: Finalize to main and open PR

Only when the ledger has no open P0 and CUDA is green.

**Step 1 — squash via commit-tree, parent MUST be `{fork}`**

Do NOT use `git add -A && git commit` directly. That produces a commit whose
parent is the upstream target (or a merge node), so GitHub's 3-way merge finds
an ancient merge-base and explodes into hundreds of false conflicts.
Instead, capture the final tree and re-parent explicitly onto `{fork}`:

```bash
git -C "$REPO" add -A
# Commit working state as a scratch commit, then re-parent
TREE=$(git -C "$REPO" rev-parse HEAD^{tree})
NEW=$(git -C "$REPO" commit-tree "$TREE" -p {fork} \
        -m "sync: TransformerEngine-FL {base}..{fork} onto upstream {target}")
git -C "$REPO" reset --hard "$NEW"
```

Verify the parent is correct before pushing:
```bash
git -C "$REPO" log --oneline -2   # HEAD parent should be {fork} HEAD
git -C "$REPO" merge-base HEAD {fork}   # must equal $(git rev-parse {fork})
```

**Step 2 — simulate GitHub's 3-way merge before pushing**

```bash
MERGE_BASE=$(git -C "$REPO" merge-base HEAD {fork})
# Expect zero conflict lines in the output:
git -C "$REPO" merge-tree "$MERGE_BASE" {fork} HEAD | grep -c '^<<<<<<<' \
  && echo "CONFLICTS FOUND -- fix before pushing" || echo "clean"
```

Only proceed if the conflict count is 0.

**Step 3 — push to personal fork, open PR to official repo**

The branch is pushed to the **personal fork** (e.g. `origin` = Caozhou1995/TransformerEngine-FL).
The PR target is the **official fork** (e.g. flagos-ai/TransformerEngine-FL:main).
Never push the sync branch directly to the official repo.

Because re-parenting rewrites history, a force push is required. Use
`--force-with-lease` (fetch first to update the tracking ref):

```bash
git -C "$REPO" fetch origin sync/upstream-{target}   # update tracking ref
git -C "$REPO" push origin sync/upstream-{target} --force-with-lease
```

Open the PR with: summary of upstream delta, list of preserved plugin
backends, adapt/drop decisions with rationale, test matrix (pass/blocked/fail), and
submodule SHA bumps. Attach the artifact dir contents as evidence.

---

## Appendix A: Vendor field traps (distilled from V1 / PR#67)

These are real breakages seen in past syncs. The backend set is discovered
dynamically (Stage 1); the traps below are what to look for per backend, not a
fixed roster. Confirm each backend still exists under
`transformer_engine/plugin/core/backends/vendor/` before applying its lore.

- **Op-count gap (seen on hygon)** -- a vendor backend may register fewer ops than
  the plugin base defines. After merge, diff registered ops vs `core/ops.py` base
  methods per backend; a shrinking count usually means an upstream op was renamed
  and the vendor register map wasn't updated. Treat as adapt, not drop.
- **Lazy tex module (seen on enflame)** -- some backends resolve
  `transformer_engine_torch` (the `tex` extension) lazily so import works without
  the compiled extension. Upstream refactors that eagerly import `tex` at module
  top-level break this. Preserve the lazy/staged import guard; re-derive it if
  upstream moved the import site.
- **AttentionParams field drift** -- upstream periodically adds/removes fields on
  attention param structs. A vendor attention impl passing positional args, or
  reading a removed field, silently breaks. After merge, diff the attention param
  definition `base..target` and reconcile every vendor `attention/` impl field by
  field.
- **No `*args`/`**kwargs` passthrough** -- FlagScale forbids swallowing args with
  `*args`/`**kwargs` in plugin op signatures, because it hides upstream signature
  changes. Keep explicit signatures so a mismatch surfaces as a TypeError at the
  boundary instead of silently mis-dispatching.

## Appendix B: Anti-patterns to avoid (why V1 needed a rewrite)

- **Do not hardcode a release number or date** anywhere in the skill or in
  generated commands. Derive from refs.
- **Do not reference paths you have not confirmed.** V1 shipped a `cicd-pipeline.md`
  and a `validate_plugin.sh` referencing `transformer_engine/common/plugin/*.h`,
  `common/cuda_patches/*.patch`, and `tests/unit/` -- none of which exist. Always
  `git ls-tree`/`ls` first (Rule 2). Stage 6 automates this drift check.
- **Do not count a blocked backend as a pass.** No hardware = BLOCKED with reason.
- **Do not squash without re-parenting.** `git commit-tree -p {fork}` is mandatory.
  If you use plain `git commit` after a merge, the squash commit's parent becomes
  the upstream target (or a merge node). GitHub then diffs from an ancient merge-base
  (~v0.1) and reports hundreds of false conflicts on the PR. The fix is to re-parent
  after the fact: `TREE=$(git rev-parse HEAD^{tree}); NEW=$(git commit-tree $TREE -p {fork} -m "..."); git reset --hard $NEW`.
  See `pitfall/te_upstream/pr_conflict_reparent_fix`.

## Artifact layout

All under `$ART = /share/.../temp/te-upstream-<target>/`:

```
fork-delta.txt         backends.txt          decisions.tsv
merge.log conflicts.txt         conflict-notes.md
bindings.txt           plugin-base.txt       api-matrix.tsv
cuda-impl-gaps.txt     binding-arg-drift.txt pybind-enums.txt
plugin-enums.txt       enum-gaps.txt
build-delta.txt        submodule-status.txt
local-uses.txt         failure-masks.txt     test-cuda.log
upstream-touched.txt   expected-tests.txt    collected.txt
```

The `decisions.tsv` ledger is the source of truth: no open P0 rows -> eligible to
finalize (Stage 8).