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

# Operator contracts for day 0 adaptation

Condensed from **新模型适配算子层开发规范.md**, clauses A1–F6. Clause IDs below
refer to that source. MUST preserves a hard requirement; SHOULD preserves a
recommended practice whose exception needs a reason and validation in the change
description. Repository-specific rules apply only to the named repository; verify
paths, registration and commands against its current branch.

Read this reference when adding, migrating or changing numerical operators. A/B
cover placement and integration, C covers semantics, D covers runtime/graphs/tuning,
E covers correctness, and F covers performance and delivery. Pure documentation,
logging or registration-metadata changes use checks appropriate to their impact
and repository gates; they do not require unrelated kernel benchmarks.

## A. Ownership and placement

- **A1 MUST:** Use `FlagGems` for framework-independent ATen and fused computation;
  `FlagGems-vllm`/`FlagGems-sglang` for their framework's specific layouts, state or
  fused contracts. Keep models, weight loading, request/token/slot scheduling and
  long-lived resource ownership in the framework. Classification follows actual
  reads/writes: GPU-generated scheduling metadata stays in the framework, while
  score-based Top-k and numerical gather/scatter are operator work. Independent
  tensor view/reshape operations can also belong in operator libraries. Operators
  consume explicit buffers/indices/parameters.
- **A2 MUST:** Use the existing layout below. Keep shared semantics in generic
  code, genuine specialization/tuning in vendor code and independent references
  separate. Vendor names must agree with detection/configuration; never disguise
  a reference as a vendor-optimized implementation. Production must not depend
  on test code.

| Repository | Generic implementation | Vendor implementation | Reference/tests |
|---|---|---|---|
| FlagGems | `src/flag_gems/ops/`; fused APIs in `src/flag_gems/fused/` | `src/flag_gems/runtime/backend/_<vendor>/` | Existing independent reference entrypoints; `tests/`, `benchmark/` |
| FlagGems-vllm | `src/flaggems_vllm/ops/` | `src/flaggems_vllm/runtime/backend/_<vendor>/ops/` | Existing independent reference entrypoints; `tests/`, `benchmark/` |
| FlagGems-sglang | `src/flaggems_sglang/ops/` | `src/flaggems_sglang/runtime/backend/_<vendor>/ops/` | `src/flaggems_sglang/reference/`, `tests/` |

- **A3 MUST:** Inventory generic, framework-specific and target-backend APIs first;
  compare dtype, layout, masks, quantization parameters, state updates and errors.
  Expose a thin public API when a compatible kernel already exists. Do not copy it
  into the framework. Leave unrelated legacy implementations alone; preserve
  numerical semantics, order and contracts when migration is required.
- **A4 SHOULD:** Find vendor copies of a changed generic operator and assess their
  shared semantic fixes while preserving necessary specialization. Do not create
  a vendor copy that only forwards to generic code.

## B. Public APIs, registration and infrastructure

- **B1 MUST:** Deliver the complete chain: signature, public exports, configuration,
  registration, backend selection and test selectors. Exercise the target kernel
  through the repository's public entrypoint; an isolated implementation is incomplete.
- **B2 MUST:** Follow each repository's existing registration mechanism. FlagGems
  updates the supported ATen schemas/overloads in `conf/operators.yaml`,
  `_FULL_CONFIG` and exports; fused APIs use their separate export/configuration
  path. FlagGems-vllm updates exports, `conf/operators.yaml`, backend registration
  and test markers. FlagGems-sglang exports real ops in the module `__all__` and
  uses its registrar/configuration and `get_reference()` mechanism. Match public
  names, variants, configuration and selectors under that repository's conventions.
- **B3 MUST:** Reuse the compiler wrappers, launcher and tuning infrastructure.
  Do not edit installed third-party sources, substitute PyTorch for a target-device
  kernel or add implicit production PyTorch fallback. Keep host wrappers to
  validation, metadata, allocations and launch; follow existing conventions for
  pure metadata operations and independent references.
- **B4 SHOULD:** Avoid changing shared test/benchmark/build infrastructure for one
  operator. If needed, explain why and its scope, and regress all affected callers.
- **B5 MUST, FlagGems:** Verify each declared variant/overload's real ATen schema,
  YAML id/attributes, `_FULL_CONFIG` key, exports and selector. Preserve leading
  underscores and dotted overload suffixes in registration keys. Keep YAML ids
  unique/ordered under current schema and change only relevant entries. API names,
  YAML ids and registration keys need not be identical. Never put fused non-ATen
  ops in `_FULL_CONFIG`. Separately prove registration dispatch, not just direct calls.
- **B6 MUST, FlagGems-sglang:** Put references only in
  `src/flaggems_sglang/reference/` and obtain them through `get_reference()`.
  Do not define/export `reference` or `solution` aliases in production op modules
  or `__all__`, or compare a kernel against itself.

## C. Numerical and data contracts — define before coding

- **C1 MUST:** Use the actual ATen schema/overload or pinned framework calling
  contract. Define every parameter/default, return structure, valid domain,
  variant and unsupported combination. Validation must not reject valid upstream inputs.
- **C2 MUST:** Respect each tensor's own shape, dtype, device, stride, storage offset,
  broadcasting and output layout. Assume contiguity only when the API requires it;
  use the correct device context/guard for multi-device launches.
- **C3 MUST:** Honor mutation, storage aliasing, cache/state read-write timing and
  protected regions. Implement caller-provided `out=` buffers with their dtype,
  layout and aliasing semantics; views must preserve shared storage.
- **C4 MUST:** Actually consume masks, causal flags, scales, bias, windows, indices,
  packed weights and zero-points. Match promotion, accumulation, integer range,
  rounding and intermediate materialization order. Handle NaN, Inf, negative zero
  and extremes according to the reference contract.
- **C5 MUST:** Define empty batches/sequences and zero-dimensional inputs, including
  state effects. `has_initial_state=False` does not imply empty input. Reject
  unsupported inputs or missing capabilities before any write or state side effect.
- **C6 MUST:** Size index types for the maximum supported problem. Supply suitable
  `other` values for masked loads; mask potentially out-of-bounds stores. Check
  large shapes, tail tiles and every buffer's independent address range.
- **C7 MUST:** Match exception types/timing for invalid indices, dimensions, parameter
  combinations and device/dtype mismatches. Do not silently clamp/truncate/modulo
  illegal inputs. Use explicit exceptions rather than removable Python `assert`
  for user-input validation.
- **C8 MUST, FlagGems:** Implement/test each declared default, in-place, `out=`, scalar
  overload and backward variant separately. `out=` must honor resize, dtype/device,
  stride/offset and return the original buffer reference; in-place variants must
  honor aliasing/overlap. A normal output function cannot replace different side effects.

## D. Runtime, graphs and tuning — when applicable

- **D1 MUST:** Probe optional capabilities before computation. Before any effects,
  existing policy may select another implementation of the same contract or reject
  the call. After compute, mutation, state updates or asynchronous launch starts,
  propagate failures without cross-backend retry.
- **D2 MUST:** Declare buffer/state/workspace aliasing, ownership, lifetime and
  concurrency requirements. Framework/caller owns persistent cache, reused state,
  graph workspace and cross-stream synchronization; wrappers may allocate ordinary
  outputs and call-local temporaries. Avoid capture-illegal synchronization or
  blocking CPU-device transfers; callers manage stable addresses and stream dependencies.
- **D3 MUST:** Use standard autotune/state restoration. Restore affected inputs,
  outputs and state after every candidate run. Match candidates, tuning keys and
  shape coverage to real hardware; use explicit fixed heuristics when autotune is unsuitable.
- **D4 SHOULD:** Avoid hot-path synchronization, device-to-host reads, redundant
  launches and element-wise Python loops. Preserve numerical order, output dtype
  and state effects when optimizing.
- **D5 MUST:** Validate optional acceleration environment/APIs before use and clear
  invalid symbols that could pollute compilation, dependency scanning or cache keys.
  Test optional paths separately; their skip conditions must not skip baseline
  tests. Honor actual backend/compiler compatibility.
- **D6 SHOULD, FlagGems:** Place vendor tuning in its backend configuration and load
  it through the existing tuning entrypoint. Explain fixed code heuristics and
  validate affected shapes; do not hardcode one device's policy in generic code.

## E. Correctness and dispatch evidence

- **E1 MUST:** Derive expected results from native upstream or an independent
  reference; do not directly/indirectly call the tested operator or share potentially
  defective production calculations. Tests may import the tested entrypoint;
  reference/solution must not resolve to it.
- **E2 MUST:** Cover representative shapes, every claimed dtype, non-contiguous
  inputs, nonzero storage offsets and relevant devices. Exercise substantive
  branches, empty inputs, boundaries, invalid/unsupported combinations, variants,
  output buffers, mutation and untouched state regions.
- **E3 MUST:** Follow test layout/markers and standard selectors. Assert that public
  entrypoint/registration dispatch actually executes the target backend kernel;
  a returned wrapper value or successful service startup is insufficient.
- **E4 MUST, graphs:** Capture and replay real work. Update inputs and fill independent
  outputs with sentinels before replay. Reset mutated/aliased inputs and state so
  reference/replay start identically; compare final outputs/state and cover affected
  state evolution/concurrency. An empty graph or stale output cannot count as passing.
- **E5 MUST:** Fix genuine defects. Correct broken tests/references when justified,
  but do not weaken tolerances, critical assertions or supported-case coverage.
  Base tolerances on dtype, arithmetic order and repository standards; retain
  necessary numerical boundary checks.
- **E6 MUST:** Sample the real allowed domain, including negative/zero/boundary and
  special floating values where valid. Restore global configuration, seeds and
  device state in teardown. Explain skip/xfail reasons, scope and tracking issue
  in code. Keep unit-test workload within CI limits.
- **E7 MUST, FlagGems:** Precision tests directly call `flag_gems.<op>` for each
  registered variant; `use_gems()` alone is insufficient. Separately test that
  PyTorch registration dispatch actually routes correctly.

## F. Benchmarks and complete delivery

- **F1 MUST:** Use the repository benchmark framework and real supported workloads,
  dtypes and shapes. Confirm baseline/target implementations; compare on the same
  device with matching synchronization, warmup and graph capture/replay settings.
  Match rank synchronization and iteration counts for multi-device work.
- **F2 MUST:** Report baseline latency, target latency and speedup, with device,
  dependency versions, commits, shape/dtype, failures and missing coverage. Apply
  current project thresholds. Investigate core regressions or follow its review
  process; the best single shape does not justify a universal performance claim.
- **F3 MUST:** Deliver implementation, public exports, registration/configuration,
  unit tests and necessary benchmarks. Check the downstream public API is accessible
  and dependencies pinned. Run applicable formatting, correctness, build and target
  hardware checks; report failed/unverified scope in the PR. Keep debug scripts,
  intermediate artifacts and unrelated changes out of commits.
- **F4 MUST:** Use the project logger instead of `print`; write source, comments and
  logs in English. Include required license/source headers and generated-code
  attribution. Pass configured formatter, linter and pre-commit checks.
- **F5 MUST, FlagGems:** Log public entries with `logger.debug("GEMS <OP>")`, including
  variant suffixes. Vendor entries use `GEMS_<VENDOR>` under existing logger naming.
- **F6 MUST, FlagGems-vllm:** The first executable statement of a public vendor
  implementation, after any docstring, is
  `logger.debug("GEMS_<VENDOR> <OP>")` with repository-correct vendor/op names.
