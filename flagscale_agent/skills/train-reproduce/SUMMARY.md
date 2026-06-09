# Reproduce — Summary

Reproduce training results from open-source implementations to establish a verified baseline before migrating to FlagScale.

**Load when**: reproducing a paper's training results, establishing a reference baseline for precision alignment, or validating that a source implementation works before porting.

Key concept: IMMUTABLE parameters (model arch, tokenizer, optimizer, loss, data) vs ADAPTABLE parameters (parallelism, hardware, batch schedule). "Reproduce" = strict immutable params. "Verify" = quick pipeline check.
