# Precision Alignment — Summary

Systematically align training precision between reference and target systems using progressive 6-level elimination.

**Load when**: verifying numerical alignment after model porting, comparing loss curves between framework versions, or diagnosing training divergence across hardware.

Three scenarios: Model Migration (native→FlagScale), Internal Iteration (self-regression), Hardware Migration (NVIDIA→new hardware). Six levels: structure → hyperparams → data → init → loss/eval → forward/backward. Each level eliminates one category of variables.
