# Mg-Fl-Upstream-Sync -- Summary

Bring the FlagScale fork `flagos-ai/Megatron-LM-FL` up to a new upstream
`NVIDIA/Megatron-LM` release, then sync FlagScale's own training/legacy code to match.
Version-agnostic and evidence-gated. Distilled from flagos-ai PR#42's human SOP, fused
with Megatron-LM-FL core/parallel/training knowledge and hardened into assertion gates.

**Load when**: Megatron-LM-FL needs to merge a newer upstream NVIDIA/Megatron-LM release
(core version bump, e.g. core_v0.16.1 -> core_v0.17.0), or a full upgrade also requires
syncing FlagScale training code afterward.

**Two co-equal tracks**:
- Track A -- library (`megatron.core` + `megatron.plugin`, the only pip-installed packages).
- Track B -- training (`FlagScale/flagscale/train/megatron/`, resolved via PYTHONPATH).

**Two refs drive everything** (never hardcode a version):
- `base` -- upstream commit the fork last synced from (common ancestor)
- `target` -- upstream commit/tag to sync up to
- fork delta to preserve = `base..main`; upstream delta = `base..target`

**Fork mechanisms to protect**:
- Platform mechanism -- `torch.cuda.*` -> `cur_platform.*` in core/plugin (multi-chip)
- Override mechanism -- `@overridable` in core, `@override` impls in plugin
- New features -- dualpipev, hetero, engram
- `FlagScale Begin/End` blocks -- standalone fork additions inside core files

**Pipeline**: Stage 0 orientation+refs -- Stage 1 classify fork delta (categories A/A2/B/C/
D/E/F + @overridable list) -- Stage 2 patch-based integration onto a clean dev branch (per-file
3-way apply, priority matrix P0/P0-F/P1/P2) -- Stage 3 plugin integrity (torch.cuda residue,
decorator counts, feature parse, override-body drift) -- Stage 4 stale refs -- Stage 5 build+
import+native unit tests -- Stage 6 tree-replacement merge to main -- Stage 7-10 FlagScale
three-way training sync + precision alignment + hetero/engram feature verification.

**Key principles**:
- Version-agnostic: derive from base/target refs; never hardcode a release or date
- Probe before referencing any path (`git ls-tree`/`ls`/grep the real symbol)
- Patch-based integration, NOT `git merge` (divergent histories -> spurious conflicts)
- Plugin files sacred: never auto-resolve P0 toward upstream
- cur_platform + @overridable/@override must survive AND stay in sync (drift is the most
  dangerous regression -- passes static checks, fails only at runtime)
- FlagScale Begin/End blocks are never P2 (check `grep -c` on main first)
- Evidence gate: artifacts to `$ART = <workspace>/mg-fl-sync-<target>/`, never `/tmp`
- Gates are assertions (counts, residue, parsed pytest), never "looks fine"
- Tree-replacement merge (`merge -s ours` + `read-tree`), never `-X theirs` (72+ dup-bug files)
- Two tracks, fix in the right repo: core/plugin issue -> Megatron-LM-FL; training -> FlagScale
- Branch pairing mandatory: main+main baseline, dev+dev-train comparison
- Three-way for FlagScale (`diff = main - base`), respect upstream deletions
- Precision alignment gate: lm loss +-5%, grad norm same order, throughput +-10%

**Traps baked in**: `-X theirs` duplicate blocks; override-body drift (e.g. MTP silently
dropped in hetero mode); FlagScale-block loss on P2; TE-FL fused-op signature drift
(`fused_rope_backward` 9th arg -> backward-only TypeError that looks like a hang); engram
`allreduce=False` params needing `engram_dp_group` + separate grad buffers.
