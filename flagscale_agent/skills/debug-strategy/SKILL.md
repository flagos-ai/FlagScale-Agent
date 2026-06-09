---
name: debug-strategy
description: Systematic debugging methodology for training infrastructure. Covers error classification, root cause analysis,
  the 2-strike rule, and when to escalate vs keep trying.
keywords:
- debug
- error
- fix
- stuck
- loop
- failure
- diagnose
- root cause
- 调试
- 排错
- 报错
- 失败
requires: []
suggests:
- ops-discipline
- train-run
constraints:
- id: no_blind_retry
  description: Never retry a failed command a 3rd time without a fundamentally different approach
  trigger:
    tools:
    - shell
  prompt: Check if the agent is retrying a command that failed twice with similar errors without changing approach
  correction: Stop and perform root cause analysis. Fill the RCA template before trying again.
- id: no_symptom_fix
  description: Never fix at crash site (.to(dtype), .reshape) without tracing root cause
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - .to(
    - .reshape(
    - .view(
    - strict=False
  prompt: Check if this edit patches a symptom rather than fixing root cause upstream
  correction: Trace the data flow upstream to find where the wrong dtype/shape originates.
- id: two_strike_reminder
  description: Remind about 2-strike rule when same error category appears twice
  trigger:
    tools:
    - shell
    keywords:
    - Traceback (most recent
    - RuntimeError:
    - CUDA error
    - OOM
  prompt: Check if the same category of error has appeared twice consecutively
  correction: '2-Strike Rule: Same error category hit twice. STOP fixing forward — root cause audit needed.'
context_injection:
  always:
  - The 2-Strike Rule
  - Root Cause Analysis Template
  by_tool:
    shell:
    - Error → Action Mapping
    - Diagnosis Methodology
    edit_file:
    - Anti-Patterns to Avoid
    write_file:
    - Anti-Patterns to Avoid
---
# Debug Strategy

Systematic debugging methodology for training infrastructure failures.

---

## The 2-Strike Rule

If the same category of error occurs twice consecutively, STOP fixing forward. The repeated failure means a wrong assumption upstream, not a local bug.

**Error categories** (same category = same strike counter):
- Shape/dimension errors (tensor size mismatch, broadcast failure)
- Import/module errors (ModuleNotFoundError, AttributeError)
- Parallelism errors (NCCL timeout, rank mismatch, hang)
- Data pipeline errors (wrong batch format, missing keys)
- Config errors (unknown argument, invalid value)

**After 2 strikes**:
1. Pause execution — do not attempt a 3rd fix of the same type
2. Root cause audit — re-read relevant source code end-to-end
3. Identify the systemic gap — what assumption is wrong?
4. Report to user — explain findings and propose different approach
5. Only proceed after confirming new approach

---

## Root Cause Analysis Template

When stuck, fill this before trying another fix:

```
SYMPTOM: [what error message / behavior]
HYPOTHESIS: [one sentence — what's actually wrong]
EVIDENCE: [what I read/checked that supports this]
PREVIOUS ATTEMPTS: [what I tried and why it didn't work]
NEW APPROACH: [fundamentally different strategy]
```

If you can't fill HYPOTHESIS with confidence, you need to read more code — not try more fixes.

---

## Error → Action Mapping

| Error Pattern | First Action | NOT This |
|---------------|-------------|----------|
| ImportError / ModuleNotFoundError | Check what's installed: `pip show <pkg>`, verify conda env | Don't pip install blindly |
| Shape mismatch | Print actual shapes at the boundary, trace data flow | Don't reshape at error site |
| NCCL timeout / hang | Check if all ranks reach the same collective, check network | Don't increase timeout |
| OOM | Calculate memory budget, identify the largest tensor | Don't immediately add TP |
| Config error | Read the argument parser source code | Don't guess valid values |
| RuntimeError (CUDA extension) | Check if extension is compiled for current CUDA version | Don't disable the feature without noting perf impact |
| KeyError in state_dict | Print both sides' keys, find the mapping gap | Don't use strict=False and move on |

---

## Diagnosis Methodology

### Isolate → Hypothesize → Verify

1. **Isolate**: Reduce to minimal reproduction. Remove parallelism, use tiny model, use synthetic data. Find the smallest config that still fails.

2. **Hypothesize**: Based on the error and your reading of the code, state ONE hypothesis about root cause. Not "maybe X or Y" — commit to one.

3. **Verify**: Design a test that PROVES or DISPROVES your hypothesis. Not "try the fix and see if it works" — that's guessing, not diagnosing.

### When to Read More vs Try More

**Read more when**:
- You don't understand what the error message means
- You haven't read the function that's failing
- Your last 2 fixes were wrong
- The error is in framework code you didn't write

**Try more when**:
- You understand the root cause and have a specific fix
- The fix is cheap to test (< 30 seconds)
- You're iterating on YOUR code, not framework code

---

## When to Ask the User

Ask when:
- Same error after 3 attempts with different approaches
- The fix requires a design decision (e.g., disable feature vs install dependency)
- You need information not available in the codebase (credentials, hardware details, project constraints)
- The root cause is in code you can't modify (third-party, framework)

Don't ask when:
- You haven't tried reading the relevant source code yet
- The error message is clear and the fix is obvious
- You're just uncertain — try the most likely fix first

---

## Anti-Patterns to Avoid

1. **"Let me try one more small fix"** after 2 failures → The problem is upstream
2. **Disabling features to work around errors** without noting the impact → Document what's lost
3. **Reading 30-line snippets** instead of full functions → You'll miss context
4. **Fixing the symptom** (adding .to(dtype) at crash site) instead of the cause (wrong dtype upstream)
5. **Running the same command again** hoping for different results → Change something first
6. **Monitoring stdout for 300s** when stderr already has the crash → Check stderr first
