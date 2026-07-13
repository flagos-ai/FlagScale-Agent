---
name: infer-precision-check
description: >
  Verify inference output precision for vllm-plugin-FL on any hardware backend.
  Compares token-level outputs against NVIDIA + upstream vLLM ground truth.
  Use this skill whenever a task requires a correctness gate: model porting,
  hardware adaptation, plugin version upgrades, or regression detection after
  any code change. Trigger when the user says "check precision", "verify
  correctness", "compare outputs", "run E2E precision test", "precision
  alignment", or "does the output match NVIDIA?". Works for text-only and
  multimodal models; greedy decoding (temperature=0) is the standard mode.
keywords:
- inference
- precision
- correctness
- e2e
- ground-truth
- token-comparison
- regression
- vllm
- hardware
requires:
- infer-env-setup
suggests:
- infer-hw-adapt
- infer-model-adapt
- debug-strategy
constraints:
- id: greedy_only
  description: Always use temperature=0 (greedy) for precision comparisons
  trigger:
    keywords: [temperature, sampling, top_p, top_k]
  prompt: Check if the comparison uses greedy decoding
  correction: >
    Set temperature=0, top_p=1.0, top_k=-1 on both GT and target sides.
    Any sampling randomness makes token-level comparison meaningless.
- id: gt_first
  description: Always generate GT outputs before running target hardware
  trigger:
    keywords: [target hardware, target backend, run on]
  prompt: Check if GT outputs have already been collected and saved
  correction: >
    Run the GT script on NVIDIA first, save outputs to a JSON file,
    then run the target hardware script and compare.
- id: same_model_weights
  description: GT and target must load identical model weights
  trigger:
    keywords: [model path, model_path, load model]
  prompt: Check if both sides point to the same model weights
  correction: >
    Verify with md5sum on representative weight files, or confirm both sides
    mount the same NFS path. Different weights make comparison meaningless.
- id: log_all_outputs
  description: All inference outputs must be saved to files, never just printed
  trigger:
    tools: [shell]
    keywords: [generate, llm.generate, vllm serve, curl]
  prompt: Check if the command saves output to a log file
  correction: >
    Pipe to 2>&1 | tee /workspace/adapt-logs/precision_<side>_<timestamp>.log
    and save token IDs to JSON for offline comparison.
- id: tp_parity
  description: TP size must be identical on GT and target unless testing TP scaling
  trigger:
    keywords: [tensor_parallel, tp_size, --tensor-parallel-size]
  prompt: Check if TP size matches between GT and target runs
  correction: >
    Use TP=1 for the initial precision gate. Only increase TP after TP=1 passes.
context_injection:
  always:
  - Critical Rules
  - Comparison Protocol
  by_tool:
    shell:
    - Environment Probe
    - GT Collection Commands
    - Target Collection Commands
---

# Inference Precision Check

Standardized correctness gate for vllm-plugin-FL on any hardware backend.

## When to Use This Skill

| Checkpoint | Trigger |
|---|---|
| Model porting (Step 13 in infer-model-adapt) | After offline inference passes |
| Hardware adaptation (Stage 4 in infer-hw-adapt) | After functional tests pass |
| Plugin version upgrade | After unit + functional tests pass |
| Any suspicious output divergence | Immediately on user report |
| Before PR submission | Always — final correctness gate |

---

## Critical Rules

1. **GT = NVIDIA + upstream vLLM** — never use the target hardware as ground truth.
2. **Greedy decoding only** — temperature=0. One mismatch position tells you exactly where the bug is.
3. **Same weights, same tokenizer** — verify with file hash or shared NFS path.
4. **Save to JSON, compare offline** — save token IDs and decoded text to files; don't eyeball terminals.
5. **TP=1 first** — eliminate TP sharding as a variable. Scale up only after TP=1 passes.
6. **Report divergence position** — "first mismatch at token 3" is actionable; "outputs differ" is not.
7. **Log both sides** — persist GT and target logs to `/workspace/adapt-logs/` with timestamps.

---

## Comparison Protocol

### Tier 1 — Token ID Match (strict)

Exact token ID match at every position.

Required for:
- First 10 tokens of any prompt
- Models with fully deterministic attention (no Flash Attention variants)
- Baseline text-only prompts

Pass criterion: **all token IDs identical**.

### Tier 2 — Top-K Token Set Match (relaxed)

The correct token appears in top-K logits even if greedy selection differs.

Acceptable for:
- Tokens 11–30 when hardware FP16/BF16 accumulation order differs
- Models using Flash Attention (small numerical noise expected)

Pass criterion: **correct token in top-5 candidates at every position**.

### Tier 3 — Semantic Match (fallback)

Human-readable output conveys the same meaning.

Use only when:
- Tier 2 cannot be achieved due to hardware numerical limits
- Divergence is in filler tokens (punctuation, articles), not content tokens
- The model uses hardware-specific fused kernels with documented precision loss

Pass criterion: **output judged semantically equivalent on all test prompts**.
Document tier level achieved in the PR description.

---

## Step 0: Environment Probe

Before any test, verify both sides are ready.

```bash
# NVIDIA GT machine:
ssh <gt_ssh_host> "python3 -c 'import vllm; print(vllm.__version__)' && \
  nvidia-smi --query-gpu=name,memory.free --format=csv,noheader"

# Target machine:
ssh <ssh_host> "docker exec <container> bash -c '
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  ls -lh <model_path>
'"
```

Record to memory if not already present:
```
memory_write('<backend>_gt_ssh_host',  '<gt_ssh_host>')
memory_write('<backend>_model_path',   '/workspace/models/<model_name>')
memory_write('<backend>_log_dir',      '/workspace/adapt-logs')
```

Verify weight identity (if GT and target do not share NFS):
```bash
# On GT:
ssh <gt_ssh_host> "md5sum <model_path>/config.json <model_path>/tokenizer.json"
# On target:
ssh <ssh_host> "docker exec <container> md5sum <model_path>/config.json <model_path>/tokenizer.json"
# Hashes must match.
```

---

## Step 1: Prepare Test Prompts

Use the standard 5-prompt set. For multimodal models, add 3 image prompts (Step 3).

Save to `/workspace/adapt-logs/precision_prompts.json` on the target machine:

```json
[
  "Paris is the capital of",
  "The quick brown fox jumps over the",
  "In machine learning, a transformer model",
  "Once upon a time in a land far away",
  "The speed of light in vacuum is approximately"
]
```

For chat models, wrap each prompt in the model's chat template before generating.
Use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.

---

## Step 2: Collect Ground Truth (NVIDIA)

Run on the NVIDIA GT machine. Save token IDs and decoded text to JSON.

```python
# gt_collect.py  — run on NVIDIA GT machine
import json, os
from vllm import LLM, SamplingParams

MODEL_PATH = "<model_path>"
PROMPTS_FILE = "/workspace/adapt-logs/precision_prompts.json"
OUTPUT_FILE  = "/workspace/adapt-logs/precision_gt.json"
TP_SIZE = 1   # always start with TP=1

prompts = json.load(open(PROMPTS_FILE))
llm = LLM(MODEL_PATH, tensor_parallel_size=TP_SIZE, enforce_eager=True)
params = SamplingParams(temperature=0, max_tokens=32, top_p=1.0)

outputs = llm.generate(prompts, params)
results = []
for req_out in outputs:
    results.append({
        "prompt":    req_out.prompt,
        "token_ids": req_out.outputs[0].token_ids,
        "text":      req_out.outputs[0].text,
    })

json.dump(results, open(OUTPUT_FILE, "w"), indent=2, ensure_ascii=False)
print(f"GT saved to {OUTPUT_FILE}")
```

```bash
# On GT machine:
ssh <gt_ssh_host> "cd /workspace && \
  python3 gt_collect.py \
  2>&1 | tee /workspace/adapt-logs/precision_gt_$(date +%Y%m%d_%H%M%S).log"
```

Copy GT results to target machine if not on shared NFS:
```bash
scp <gt_ssh_host>:/workspace/adapt-logs/precision_gt.json \
    /tmp/precision_gt.json
ssh <ssh_host> "docker cp /tmp/precision_gt.json \
    <container>:/workspace/adapt-logs/precision_gt.json"
```

---

## Step 3: Collect Target Hardware Outputs

Run on target machine inside the Docker container.

```python
# target_collect.py  — run inside container on target machine
import json, os
os.environ["VLLM_PLUGINS"] = "fl"

from vllm import LLM, SamplingParams

MODEL_PATH    = "<model_path>"
PROMPTS_FILE  = "/workspace/adapt-logs/precision_prompts.json"
OUTPUT_FILE   = "/workspace/adapt-logs/precision_target.json"
TP_SIZE = 1   # match GT TP size

prompts = json.load(open(PROMPTS_FILE))

if __name__ == "__main__":
    llm = LLM(MODEL_PATH, tensor_parallel_size=TP_SIZE, enforce_eager=True,
              trust_remote_code=True)
    params = SamplingParams(temperature=0, max_tokens=32, top_p=1.0)

    outputs = llm.generate(prompts, params)
    results = []
    for req_out in outputs:
        results.append({
            "prompt":    req_out.prompt,
            "token_ids": req_out.outputs[0].token_ids,
            "text":      req_out.outputs[0].text,
        })

    json.dump(results, open(OUTPUT_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"Target saved to {OUTPUT_FILE}")
```

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  cd /workspace &&
  python3 target_collect.py \
  2>&1 | tee /workspace/adapt-logs/precision_target_$(date +%Y%m%d_%H%M%S).log
'"
```

---

## Step 4: Compare Outputs

Run the comparison script. This is the pass/fail gate.

```python
# compare.py  — run anywhere with both JSON files accessible
import json, sys

gt_file     = "/workspace/adapt-logs/precision_gt.json"
target_file = "/workspace/adapt-logs/precision_target.json"

gt      = json.load(open(gt_file))
target  = json.load(open(target_file))

assert len(gt) == len(target), "Prompt count mismatch — check both files"

all_pass = True
for i, (g, t) in enumerate(zip(gt, target)):
    g_ids = g["token_ids"]
    t_ids = t["token_ids"]
    min_len = min(len(g_ids), len(t_ids))

    first_mismatch = None
    for pos in range(min_len):
        if g_ids[pos] != t_ids[pos]:
            first_mismatch = pos
            break

    status = "PASS" if first_mismatch is None else f"MISMATCH@token{first_mismatch}"
    if first_mismatch is not None:
        all_pass = False

    print(f"[{status}] Prompt {i}: {g['prompt'][:40]!r}")
    print(f"  GT:     {g_ids[:16]}")
    print(f"  Target: {t_ids[:16]}")
    if first_mismatch is not None:
        print(f"  GT text:     {g['text'][:80]!r}")
        print(f"  Target text: {t['text'][:80]!r}")

print()
print("=== RESULT:", "ALL PASS" if all_pass else "FAILED — see mismatches above")
sys.exit(0 if all_pass else 1)
```

```bash
ssh <ssh_host> "docker exec <container> python3 /workspace/compare.py \
  2>&1 | tee /workspace/adapt-logs/precision_compare_$(date +%Y%m%d_%H%M%S).log"
```

---

## Step 5: Multimodal Precision Check (if applicable)

Skip this step for text-only models.

### 5a. Prepare image test inputs

Use 3 fixed, publicly available images with unambiguous content.
Save image URLs or local paths to `/workspace/adapt-logs/precision_mm_prompts.json`:

```json
[
  {
    "prompt": "What is shown in this image?",
    "image":  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
  },
  {
    "prompt": "Describe the scene.",
    "image":  "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
  },
  {
    "prompt": "What text appears in the image?",
    "image":  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Culinary_fruits_front_view.jpg/320px-Culinary_fruits_front_view.jpg"
  }
]
```

### 5b. Collect GT and target for multimodal

Use the same gt_collect / target_collect pattern, replacing `prompts` with
`vllm.inputs.TextPrompt` + image data. The comparison script (Step 4) applies
unchanged — compare token IDs for the generated portion only.

---

## Step 6: TP Scaling Check (optional but recommended)

After TP=1 passes Tier 1, verify TP>1 does not introduce sharding errors.

```bash
# Re-run Steps 2–4 with TP_SIZE=<target_tp> on both GT and target.
# Accepted outcome: Tier 1 or Tier 2 pass.
# TP sharding commonly causes token mismatch at position 1–3 due to
# attention output accumulation order differences across GPUs/NPUs.
```

Record the highest TP size tested and the tier achieved at each size.

---

## Step 7: Serving Mode Check

Verify precision holds when model is served via OpenAI-compatible API.

```bash
# Start server on target:
ssh <ssh_host> "docker exec -d <container> bash -c '
  VLLM_PLUGINS=fl vllm serve <model_path> \
    --tensor-parallel-size 1 --enforce-eager \
    --trust-remote-code --port 8122 \
  2>&1 | tee /workspace/adapt-logs/precision_serve_$(date +%Y%m%d_%H%M%S).log
'"

# Wait for server to be ready:
sleep 30

# Query with greedy params:
ssh <ssh_host> "docker exec <container> bash -c '
  curl -s http://localhost:8122/v1/completions \
    -H \"Content-Type: application/json\" \
    -d \"{
      \\\"model\\\": \\\"<model_path>\\\",
      \\\"prompt\\\": \\\"Paris is the capital of\\\",
      \\\"max_tokens\\\": 32,
      \\\"temperature\\\": 0,
      \\\"top_p\\\": 1.0
    }\" | python3 -m json.tool
'"
```

Compare the `text` field in the response against GT text from Step 2.
Token-level comparison is not possible via the API; use Tier 3 (semantic match)
for serving mode.

---

## Failure Diagnosis Guide

When a mismatch is found, narrow the root cause systematically.

### Mismatch at token 0–2 (embedding or first attention block)

Most likely causes, in order:
1. **Weight loading error** — wrong tensor mapped to wrong layer. Check `load_weights` in the model file.
2. **Tokenizer mismatch** — different tokenizer version or chat template applied on only one side.
   Verify: `tokenizer.encode("Paris is the capital of")` gives same IDs on GT and target.
3. **Attention op numerical error** — backend Flash Attention gives wrong values.
   Fix: add `enforce_eager=True` and re-test. If eager passes, the kernel is at fault.

### Mismatch at token 3–10 (residual stream accumulates error)

Most likely causes:
1. **MLP kernel precision** — GEMM accumulation order differs on target hardware.
   Fix: compare with `torch.float32` forced via `dtype=torch.float32` in both runs.
2. **Layernorm implementation** — different epsilon or fused vs. unfused.
   Fix: check `RMSNorm` / `LayerNorm` dispatch in plugin; try forcing PyTorch fallback.
3. **Activation function** — `SiLU`, `GELU` variants differ across hardware libs.

### Mismatch at token 10+ (late drift)

Likely Tier 2 acceptable. Verify top-K membership:

```python
# Add to target_collect.py to capture logprobs for comparison
params = SamplingParams(temperature=0, max_tokens=32, top_p=1.0,
                        logprobs=5)   # capture top-5 logprobs
# token_ids from GT should appear in logprobs on target side
```

### All prompts mismatch (systematic failure)

Check:
- `VLLM_PLUGINS=fl` set on target but not GT (or vice versa)
- Different model weights or tokenizer versions
- Different dtype (fp16 vs bf16) on the two sides

---

## Pass/Fail Checklist

Before recording the result:

- [ ] GT outputs collected with temperature=0, TP=1 on NVIDIA
- [ ] Target outputs collected with temperature=0, TP=1 and `enforce_eager=True`
- [ ] compare.py ran to completion with no Python exceptions
- [ ] Tier achieved documented (Tier 1 / 2 / 3)
- [ ] First mismatch position recorded (or "no mismatch")
- [ ] Log files saved to `/workspace/adapt-logs/` with timestamps
- [ ] TP>1 check completed (if model will be deployed with TP>1)
- [ ] Serving mode check completed (if model will be served via API)

---

## Result Recording Template

Copy into `workspace_experiment.update_last_attempt(result=...)` or PR description:

```
## Precision Check Result

Backend: <backend>
Model:   <model_name>
vLLM:    <version>
TP:      <tp_size>

### Tier achieved: Tier <1|2|3>

| Prompt | First mismatch token | GT text (first 20 tok) | Target text (first 20 tok) |
|---|---|---|---|
| "Paris is the capital of" | none / @token N | ... | ... |
| "The quick brown fox..." | none / @token N | ... | ... |
| "In machine learning..." | none / @token N | ... | ... |
| "Once upon a time..."   | none / @token N | ... | ... |
| "The speed of light..."  | none / @token N | ... | ... |

### Notes
<Any known deviations, accepted numerical noise, workarounds applied>

### Log files
- GT:      /workspace/adapt-logs/precision_gt_<timestamp>.log
- Target:  /workspace/adapt-logs/precision_target_<timestamp>.log
- Compare: /workspace/adapt-logs/precision_compare_<timestamp>.log
```

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `infer-model-adapt` — port a new model into vllm-plugin-FL (calls this skill at Step 13)
- `infer-hw-adapt` — hardware backend adaptation (calls this skill at Stage 4)
- `debug-strategy` — systematic debugging when precision failures repeat
- `ops-discipline` — shell safety, log persistence, environment awareness

---
Related skills (load if needed): `debug-strategy`, `infer-hw-adapt`, `infer-model-adapt`
