# Infer-Precision-Check — Summary

Verify inference output precision for vllm-plugin-FL on any hardware backend, comparing token-level outputs against NVIDIA + upstream vLLM ground truth.

**Load when**: any task reaches a correctness gate — model porting (after offline inference passes), hardware adaptation (after functional tests pass), plugin version upgrades (after unit + functional tests pass), or any suspected output divergence. Also load before PR submission as the final correctness check.

**Full cycle**: Step 0 env probe → Step 1 prepare prompts → Step 2 collect GT (NVIDIA) → Step 3 collect target outputs → Step 4 compare → Step 5 multimodal check (if applicable) → Step 6 TP scaling check → Step 7 serving mode check.

**Key principles**:
- GT = NVIDIA + upstream vLLM — never use target hardware as ground truth
- Greedy decoding only (temperature=0) — sampling randomness invalidates comparison
- Same weights, same tokenizer — verify with md5sum or shared NFS path
- Save token IDs to JSON, compare offline — do not eyeball terminal output
- TP=1 first — eliminate sharding as a variable before scaling up
- Report first mismatch position — "token 3" is actionable; "outputs differ" is not
- Log both sides to `/workspace/adapt-logs/` with timestamps

**Tiers**: Tier 1 (exact token ID match) → Tier 2 (correct token in top-5 logits) → Tier 3 (semantic equivalence). Document tier achieved in PR description.

**Constraints**: 5 hard rules covering greedy decoding, GT-first ordering, weight identity, log persistence, and TP parity.
