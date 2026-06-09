---
name: train-model-porter
description: Port models from papers, HuggingFace, or other frameworks to Megatron-LM-FL for distributed training on FlagScale.
  Covers architecture analysis, whole-model implementation, checkpoint conversion, and real-data verification.
keywords:
- model
- port
- porting
- migrate
- migration
- convert
- checkpoint
- huggingface
- megatron
- architecture
- alignment
- 模型迁移
- 模型转换
- 权重转换
- 模型适配
- 检查点
parameters:
- name: source
  description: 'Source type: paper, huggingface, or code'
  default: huggingface
- name: model_name
  description: Name for the ported model (used in directory names)
requires:
- train-env-setup
- train-config
- train-data-prep
- train-run
suggests:
- topo-detect
- train-parallel-strategy
- train-precision-alignment
effects:
  mode: porting
  initial_phase: analysis
  companion_skills:
  - train-parallel-strategy
  - train-precision-alignment
  - train-config
  - train-run
  - train-env-setup
  - workspace-layout
  - train-data-prep
workflow:
  trigger:
    keywords:
    - migrate to
    - port to
    - 迁移到
    - porting to
    - convert to megatron
    keywords_in_same_input:
    - - migrate
      - train
    - - 迁移
      - 训练
    - - port
      - megatron
  stages:
  - id: analyze
    name: Analyze source model
    description: Read source code completely, extract architecture config, identify all submodules
    profile: model-migration
    depends_on: []
  - id: implement
    name: Implement Megatron-native model
    description: Write model Module, checkpoint converter, get_batch with real data
    profile: model-migration
    depends_on:
    - analyze
  - id: verify
    name: Verify correctness
    description: Load checkpoint, run forward pass with real data, check loss convergence
    profile: model-migration
    depends_on:
    - implement
constraints:
- id: baseline_before_migration
  description: Establish reproduction baseline before migrating to FlagScale
  trigger:
    keywords:
    - migrate to
    - port to
    - flagscale train
    - megatron format
  prompt: Check if migration work is starting without a verified reproduction baseline
  correction: Reproduction baseline must be verified (loss matches reference) before starting migration.
- id: no_dummy_data
  description: NEVER use torch.rand/zeros/ones as model input for verification
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - torch.rand
    - torch.zeros
    - torch.ones
    - dummy_input
    - fake_data
  prompt: Check if code uses dummy/random data as model input instead of real data from get_batch
  correction: Use real data from get_batch. Dummy data hides tokenizer, format, and special token bugs.
- id: read_before_write
  description: Must read source model code completely before writing any implementation
  trigger:
    tools:
    - write_file
    keywords:
    - class MegatronModule
    - class FlagScaleModel
    - def forward(self
    - MegatronModule):
    - ColumnParallelLinear
    - RowParallelLinear
  prompt: Check if the agent has read the source model code before writing implementation
  correction: Read source code first. Use 'Porting Discipline' — read COMPLETELY before writing.
- id: frozen_native_impl
  description: Frozen components still need Megatron-native implementation
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - requires_grad=False
    - requires_grad_(False)
    - .freeze()
    - feature_extractor
  prompt: Check if agent is skipping native implementation for a frozen component
  correction: Frozen ≠ skip native. ALL components must use Megatron primitives regardless of training status.
- id: parallelism_assessment
  description: Assess TP benefit per component before implementing
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - ColumnParallelLinear
    - RowParallelLinear
    - tensor_model_parallel
  prompt: Check if parallelism feasibility was assessed for this component
  correction: Assess each component independently for TP benefit. Small MLPs may not need TP.
context_injection:
  always:
  - Important Notes
  by_stage:
    analyze:
    - Porting Discipline
    - Pre-coding analysis
    - Porting Modes
    implement:
    - Implementation Flow
    - Failure Pivot Discipline
    - Multimodal Module Nesting
    - get_batch Under Parallelism
    verify:
    - Verification Standard
  by_tool:
    write_file:
    - Implementation Flow
    - Failure Pivot Discipline
    shell:
    - Verification Standard
---
# Model Porter

Port models to Megatron-LM-FL / FlagScale for distributed pre-training.

**Scope**: All model types — decoder-only LLM, VL, robotics, multimodal generation, etc.

**Outputs**:
1. Complete model Module (`flagscale/models/megatron/<model>/`)
2. Checkpoint conversion code (`tools/checkpoint/<model>/`)
3. Training script with real-data `get_batch` (`flagscale/train/megatron/train_<model>.py`)

## Important Notes

- **Environment isolation**: Create a dedicated conda env for FlagScale porting. Use `train-env-setup` skill for correct Python/CUDA/dependency versions.
- **Shared storage for multi-node**: All paths (data, checkpoints, logs) must be on shared storage. Avoid `/tmp/` or `./` for multi-node.
- **Quantized models (GPTQ, AWQ, GGUF)**: NOT suitable for training. Need original full-precision weights.
- **Tokenizer handling**: Always copy tokenizer from source. Verify `vocab_size` matches. Check `added_tokens.json`.
- **Source code provenance**: Verify you're reading ACTUALLY INSTALLED code: `conda run -n <env> python -c "import megatron; print(megatron.__file__)"`. Editable installs from other workspaces are a trap.
- **Auto-fetch FL deps**: Pull Megatron-LM-FL / TransformerEngine-FL from github.com/flagos-ai/ when needed — don't ask user.
- **Model size selection**: If multiple sizes available and user didn't specify, list options and recommend smallest for initial porting.
- **Architectural completeness**: The ported model Module must own ALL submodules from the source. If the source has `self.vit_model`, the target must have `self.vit_model`. Freeze/unfreeze is a training config decision (`requires_grad`, optimizer param groups), never an architecture decision. A submodule excluded from the Module cannot receive gradients — even if the user later wants to train it. After checkpoint conversion, sanity-check: source tensor count vs converted tensor count. A large gap (>10%) means you dropped a submodule.
- **Frozen ≠ Skip Native**: A component being frozen (no gradient, feature extractor, inference-only) is NEVER a valid reason to skip Megatron-native implementation. ALL components must use Megatron primitives regardless of training status. Reasons: (1) unified checkpoint conversion — one converter for the whole model, not separate logic for frozen vs trainable parts; (2) future unfreezing — if the user later wants to train the backbone, the architecture must support it without a rewrite; (3) TP memory distribution — even frozen parameters can be sharded across TP ranks to reduce per-GPU memory; (4) architectural consistency — one top-level MegatronModule owns everything.
- **TP is per-component**: Assess each component independently for TP benefit. A small MLP (e.g., 256→256) may not need TP. A large attention layer (2048 hidden, 32 heads) benefits from TP. But even without TP, use Megatron's ColumnParallelLinear (with `gather_output=True`) instead of `nn.Linear` — this preserves the option to enable TP later without code changes.
- **Unified checkpoint conversion**: There must be ONE conversion script/function that converts the entire source checkpoint to Megatron format. Do NOT have separate converters for "frozen backbone" and "trainable head". The converter maps ALL source weights to their Megatron-native equivalents in a single pass.
- **get_batch is a porting deliverable**: `get_batch` must be implemented with real data as part of the model porting output — not a separate follow-up task. Dummy data hides every bug that matters (tokenizer, format, special tokens, sequence length).
- **Dataset logic must be self-contained**: Never `import` or `sys.path.insert` an external project's dataset/dataloader code directly. External datasets don't support TP broadcast, PP stage guards, or Megatron's data contract. Instead: read the source dataset to understand the format, then implement your own `get_batch` + dataset class inside FlagScale that loads the same data files with proper parallelism support. The data *files* are shared; the data *loading code* is ours.

---

## Porting Discipline

### Read COMPLETELY before writing ANY code

1. Read COMPLETE source model code (modeling_*.py, config.json, tokenizer_config.json)
2. Read COMPLETE target Megatron model code (model_provider, builder, spec)
3. Read FULL `__init__` and `forward` signatures of every Megatron base class you'll subclass
4. Read IMPLEMENTATION of every base class method you plan to call (not just signature)
5. **Read TransformerEngine-FL attention stack** — this is a critical porting surface:
   - `TEDotProductAttention` in `megatron/core/extensions/transformer_engine.py` — Megatron's TE wrapper
   - `DotProductAttention` in `transformer_engine/pytorch/attention/` — TE's core attention
   - `backends.py` — FlashAttention, FusedAttention, UnfusedDotProductAttention backends
   - Understand: what `attn_mask_type` options exist, how `qkv_format` maps to memory layout, how CP integrates with attention
   - If source model uses a non-standard attention (flex_attention, custom masks, sliding window, sparse), map it to TE's equivalent backend and mask type
6. Search FlagScale ecosystem for similar implementations — reuse when possible
7. Build complete mapping table: source layer → target layer with shape transforms
8. Extract ALL config parameters from source config.json (not just obvious ones)
9. Save analysis to workspace before proceeding

### Pre-coding analysis (MANDATORY for models >10B or multimodal)

**Analysis 0: Model Structure Enumeration (MANDATORY — must complete BEFORE any implementation)**

Before writing ANY porting code, enumerate the COMPLETE source model structure:

1. **List ALL top-level modules** from the source model's `__init__`:
   ```python
   # Example for a multimodal model:
   self.vision_encoder = ...
   self.vision_projection = ...
   self.language_model = ...
   self.lm_head = ...
   ```

2. **Count total parameters and submodules**:
   - Run: `sum(p.numel() for p in model.parameters())` on source model
   - Run: `len(list(model.named_modules()))` to count all nested modules
   - Record these numbers — they are your completeness verification baseline

3. **Create a porting checklist** with ALL components:
   ```markdown
   - [ ] vision_encoder (ViT, 86M params)
   - [ ] vision_projection (MLP, 2M params)
   - [ ] language_model (Transformer, 7B params)
   - [ ] lm_head (Linear, 50M params)
   ```

4. **Persist the checklist** to workspace memory BEFORE starting implementation

**Analysis 1: Component diff table**

| Source Component | HF Implementation | Megatron-LM-FL Equivalent | Existing Reference | Gap / Action |
|------------------|-------------------|---------------------------|-------------------|--------------|

Every row must have an explicit action. "TBD" is not acceptable.

**Analysis 1b: Attention mechanism diff (MANDATORY)**

| Aspect | Source Model | TE-FL / Megatron | Mapping / Gap |
|--------|-------------|------------------|---------------|
| Backend | (e.g., flex_attention, sdpa, custom) | TEDotProductAttention → FlashAttention / FusedAttention | |
| Mask type | (e.g., causal, bidirectional, block-sparse, sliding window) | AttnMaskType: no_mask / causal / padding / arbitrary | |
| QKV format | (e.g., separate Q/K/V, fused QKV) | qkv_format: sbhd / bshd / thd | |
| GQA/MQA | (num_kv_heads vs num_heads) | num_gqa_groups in TEDotProductAttention | |
| Position encoding | (RoPE variant, ALiBi, absolute) | Applied before/after TE attention? | |
| Sliding window | (window_size if any) | window_size param in DotProductAttention | |
| Special masking | (cross-modal masks, prefix masks) | How to express via attn_mask_type + attention_mask tensor | |

If source uses flex_attention or custom attention kernels: identify what mask/score_mod functions they apply, then determine the equivalent TE configuration (attn_mask_type + window_size + custom mask tensor). This is a common porting gap — TE supports arbitrary masks via `attn_mask_type="arbitrary"` but performance differs from specialized kernels.

**Fallback: use source model's native attention implementation.** If the source attention cannot be cleanly mapped to TE (e.g., complex score_mod in flex_attention, custom sparse patterns, or novel attention variants), keep the original attention code and integrate it as `core_attention` in the Megatron layer spec instead of `TEDotProductAttention`. This trades TE's fused kernels for correctness and faster porting. The rest of the model (linear layers, norms, embeddings) can still use TE modules — only the attention needs to fall back.

**Analysis 2: Memory budget**

Calculate: params × bytes_per_param (weights + optimizer + gradients + activations) → total per-GPU → choose parallelism.

**Analysis 3: Parallelism strategy** (derived from memory budget)

| Model Size | TP | PP | DP | CP | Min GPUs |
|-----------|----|----|----|----|----------|

**Analysis 5: Parallelism Strategy Feasibility Assessment (MANDATORY — must complete BEFORE writing training config)**

For EACH parallelism strategy, assess feasibility based on actual model dimensions. This prevents wasted effort on strategies that don't apply and ensures the chosen strategy is documented before training config is written.

Fill in this table using the model's actual numbers:

| Strategy | Feasibility | Key Dimensions | Adaptation Work | Recommendation |
|----------|-------------|----------------|-----------------|----------------|
| **DP** | Always feasible | N/A | None | ✅/❌ + reason |
| **FSDP/ZeRO** | Feasible if model fits in memory with sharding | params × 18 bytes (bf16 + Adam) vs GPU memory | None | ✅/❌ + reason |
| **TP** | Feasible if num_heads % TP == 0 AND num_kv_heads % TP == 0 | hidden_size=?, num_heads=?, num_kv_heads=?, intermediate_size=? | Replace nn.Linear → ColumnParallel/RowParallel | ✅/❌ + reason |
| **PP** | Feasible if model has ≥PP natural stages or layers | num_layers=?, natural stages=? (e.g., vision+LLM+head) | Implement set_input_tensor(), define stage splits | ✅/❌ + reason |
| **SP** | Feasible only with TP, useful if seq_len × hidden × num_layers is large | seq_len=?, hidden=?, num_layers=? | Comes free with TP in Megatron-Core | ✅/❌ + reason |
| **CP** | Feasible only if seq_len > 4K | seq_len=? | Ring attention integration | ✅/❌ + reason |
| **EP** | Feasible only if model has MoE layers | num_experts=?, expert_size=? | Expert routing + token dispatch | ✅/❌ + reason |

Assessment criteria:
- **TP divisibility**: num_heads must be divisible by TP degree. num_kv_heads (for GQA) must also be divisible. intermediate_size must be divisible.
- **PP balance**: Calculate params per stage. If max_stage / min_stage > 2x, PP will have severe bubble overhead.
- **SP threshold**: Only beneficial when activation memory per layer (seq_len × hidden × 2 bytes) > 100MB.
- **CP threshold**: Only beneficial when seq_len > 4096. Below this, ring attention latency dominates.
- **Model size threshold for TP/PP**: If model fits on single GPU in bf16 with optimizer (params × 18 < GPU_memory × 0.8), TP/PP add communication overhead without memory benefit.

Final recommendation format:
```
Recommended strategy for [N] GPUs: DP=[x], TP=[y], PP=[z], FSDP=[stage]
Rationale: [one sentence explaining why this combination]
Not recommended: [list strategies that don't apply and why]
```

Save this assessment to memory (`memory_write key='parallelism_assessment'`) BEFORE writing any training YAML config.

**Analysis 4: Data→Model Interface Contract (MANDATORY — must complete BEFORE writing model forward())**

This is the #1 cause of porting rework: the model's forward() is designed in isolation, then when real data is connected, everything needs rewriting because the model expects different inputs than what the data pipeline produces.

Document the COMPLETE data→model interface:

| Aspect | Details |
|--------|---------|
| **Data pipeline output** | Exact dict keys from get_batch (e.g., `input_ids`, `attention_mask`, `pixel_values`, `labels`) |
| **Tensor shapes** | Shape for each key (e.g., `input_ids: [B, seq_len]`, `pixel_values: [B, C, H, W]`) |
| **Tensor dtypes** | dtype for each key (e.g., `int64` for tokens, `bfloat16` for images) |
| **Model forward() signature** | Exact parameter names the model expects |
| **Key mapping** | How data keys map to forward() params (e.g., `pixel_values` → `images`) |
| **Preprocessing between get_batch and forward** | Any transforms, normalization, padding, masking |
| **Parallelism contract** | How data is distributed across TP/PP/DP/EP/CP/SP ranks (see below) |

**⚠️ PARALLELISM IS NOT OPTIONAL — IT IS THE CORE OF MEGATRON DATA INTEGRATION:**

A data pipeline without parallelism awareness is a FAILED Megatron integration. There is no "add parallelism later" — it must be designed from the start:

| Parallelism | Data Contract | Implementation |
|-------------|--------------|----------------|
| **TP** | All TP ranks receive IDENTICAL input | `broadcast_data()` from `megatron.training.utils` |
| **PP** | Only first stage needs tokens, only last needs labels | Guard with `pre_process`/`post_process` |
| **DP** | Different micro-batch per rank | Sampler handles this — don't break with global indexing |
| **EP** | Token routing consistent across EP ranks | Dispatch/combine tensors align with expert sharding |
| **CP** | Sequence split across ranks | Position IDs, attention masks, loss masks per rank |
| **SP** | Activations distributed along sequence dim | Automatic when enabled with TP |

If your get_batch does not call `broadcast_data` and does not guard inputs with `pre_process`/`post_process`, it WILL deadlock or produce wrong results at runtime. This is not a "nice to have" — it is the fundamental contract.

How to determine this:
1. Read the source training script's data loading (get_batch / dataloader)
2. Read the source model's forward() signature
3. Read an existing Megatron train_*.py (e.g., `train_gpt.py`) to see how parallelism is handled in get_batch
4. Trace the data flow: raw data → preprocessing → get_batch output → broadcast → model.forward() input
5. Save to memory: `memory_write(key='data_model_interface', content='...')`

**This contract is your SINGLE SOURCE OF TRUTH for both model and data implementation.** Design them together. If you write model.forward() without knowing what get_batch produces, you WILL rewrite it later.

🚫 **NEVER use dummy data to bypass this step.** `torch.rand`/`torch.zeros`/`torch.ones` as model input is STRICTLY FORBIDDEN — not for "quick shape checks", not for "testing forward pass", not as a placeholder. ALL verification flows through real data.

---

## Porting Modes

### Mode 1: Config-driven (YAML only)

For standard architectures already in Megatron-LM-FL (GPT, LLaMA, Mistral, Qwen).

- No new model code needed — only YAML config + checkpoint conversion
- Verify: architecture params match source exactly (hidden_size, num_layers, num_heads, intermediate_size, vocab_size, norm_eps)
- Checkpoint conversion: weight name mapping + transpose where needed

### Mode 2: Megatron Native (full parallelism)

For custom architectures needing TP/PP/CP support.

- Implement as MegatronModule with TransformerLayer specs
- Use `ColumnParallelLinear`, `RowParallelLinear` for TP
- Implement pipeline stage splits via `pre_process`/`post_process`
- Reference: `flagscale/models/megatron/qwen2_5_vl/`, `flagscale/models/megatron/qwen3_vl/`

⚠️ **CRITICAL: "Megatron Native" means using Megatron's parallelism primitives throughout the ENTIRE model — NOT wrapping HuggingFace models inside a MegatronModule shell.** If you choose Mode 2, you MUST:
- Replace ALL `nn.Linear` in the model's compute path with `ColumnParallelLinear`/`RowParallelLinear`
- Replace ALL attention implementations with `TEDotProductAttention` or Megatron's attention modules
- Use `TransformerLayer` with proper `layer_spec` (ModuleSpec + TransformerLayerSubmodules)
- Implement `set_input_tensor()` for PP inter-stage tensor passing
- **ALL components — including frozen ones — must be implemented with Megatron primitives.** A frozen vision encoder is still part of the top-level model; implement it natively and set `requires_grad=False`. Do NOT load frozen components from HF pretrained checkpoints at runtime.
- **TP support is per-component**: Assess each component independently. A vision encoder with 16 attention heads can use TP=2/4. A small projection MLP may not need TP. But even without TP, use Megatron primitives (ColumnParallelLinear with gather_output=True) — this preserves the option to enable TP later.

**There is only ONE top-level model.** Every submodule (vision encoder, LLM backbone, projection layers, action heads, etc.) lives inside this single MegatronModule. Freeze/unfreeze is a training config decision (param groups, `requires_grad`), never an excuse to skip native implementation.

**The "frozen" anti-pattern**: The most common mistake is reasoning "this component is frozen → no gradient → no TP benefit → use HF model directly." This is WRONG because:
1. **Unified checkpoint conversion**: One converter for the whole model. Mixing HF-format frozen weights with Megatron-format trainable weights creates two conversion paths.
2. **Future unfreezing**: If the user later wants to fine-tune the backbone, the architecture must support it without a rewrite.
3. **TP memory distribution**: Even frozen 3B-parameter backbones benefit from TP sharding — 3B params × 2 bytes = 6GB per GPU without TP, vs 1.5GB with TP=4.
4. **Architectural consistency**: A model with half-HF half-Megatron internals is fragile and hard to maintain.

**Common MISTAKE that defeats the entire purpose**: Importing existing model classes (from HuggingFace, from the project's own modules, or from any external source) and wrapping them in a MegatronModule. This gives you ZERO TP/PP support — external `nn.Linear` doesn't know about Megatron's parallel groups, external attention doesn't call `broadcast_data`, and external forward doesn't implement `set_input_tensor()`. The result is functionally identical to Mode 3 (FSDP wrapper) but with more complexity. If you're going to use existing models as-is, use Mode 3 honestly.

**Decision checkpoint**: Before implementing, explicitly decide and document which mode you're using:
- Mode 2 → Rewrite model internals using Megatron primitives (more work, full parallelism)
- Mode 3 → Keep HF models, use FSDP2 (less work, DP-only scalability)
There is NO middle ground. "HF model inside MegatronModule" is Mode 3 pretending to be Mode 2.

**If you choose Mode 2, you MUST write a Migration Blueprint BEFORE writing any code.** Save it to memory with `key='migration_blueprint'`. The blueprint process:

**Step 0 (FIRST): Survey Megatron-Core available components**

Before deciding how to implement anything, check what Megatron already provides:
- `megatron/core/models/` — GPTModel, CLIPViTModel, LLaVAModel, multimodal models
- `megatron/core/transformer/` — TransformerLayer, layer specs, attention, MLP
- `megatron/core/tensor_parallel/` — ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
- `flagscale/models/megatron/` — existing ported models (qwen2_5_vl, qwen3_vl) as reference

The goal is to MAXIMIZE reuse of Megatron's high-performance implementations (fused kernels, TE attention, TP/PP support). Only write custom torch code for components that genuinely have no Megatron equivalent.

**Then document the blueprint covering:**

1. **Forward Logic Mapping** — for each source component, specify the Megatron target:
   ```
   source: VisionEncoder(24 layers)     → target: CLIPViTModel (if compatible) or TransformerLayer stack
   source: LLM Decoder(12 layers, GQA)  → target: GPTModel with TransformerConfig(num_layers=12, ...)
   source: nn.Linear(in, out)           → target: ColumnParallelLinear / RowParallelLinear
   source: SelfAttention(heads=N)       → target: TEDotProductAttention (via layer_spec)
   source: VisionProjection(MLP)        → target: compose from ColumnParallelLinear layers
   source: custom op (no equivalent)    → torch implementation: [describe algorithm]
   ```
   Priority: Megatron high-level model > TE layer (TEDotProductAttention, TE Linear, TE Norm) > Megatron primitive > compose from primitives > torch
   Use TransformerEngine wherever possible — it enables FP8, fused kernels, and seamless TP/CP integration.
   Only fall back to vanilla torch when no TE/Megatron equivalent exists.
   Do NOT write "reuse source class" or "import from source" — that is not a valid target.

2. **Data Pipeline Mapping** — for each preprocessing step in the source training script:
   ```
   source: tokenizer(text) → target: same tokenizer, called in get_batch
   source: normalize(action) → target: same normalization in get_batch
   source: image_processor(img) → target: same processor in get_batch
   parallelism: broadcast_data() for TP, pre_process/post_process guards for PP
   get_batch output: {key: shape, dtype} for every tensor
   ```

3. **Optimizer/Scheduler Mapping** — map source training config to Megatron args:
   ```
   source: lr=1e-4, warmup=500, cosine → target: --lr 1e-4 --lr-warmup-iters 500 --lr-decay-style cosine
   source: Adam(betas=[0.9,0.999])     → target: --adam-beta1 0.9 --adam-beta2 0.999
   source: grad_clip=1.0               → target: --clip-grad 1.0
   ```

Only after this blueprint is saved to memory should you write any model or training code.

### Mode 3: HuggingFace Wrapper (FSDP2 fast path)

For rapid prototyping or models that don't need Megatron parallelism.

- Wrap HF model with FSDP2 sharding
- Limited to DP + FSDP (no TP/PP/CP)
- Fastest path to training but limited scalability

---

## Implementation Flow — Whole Model First

**Core principle**: Analysis is per-component. Implementation is whole-model.

Do NOT verify components in isolation. Do NOT use dummy/synthetic data for verification. Build the complete model as a single nested Module and verify with real data.

**Data pipeline is EQUALLY important as model adaptation.** A ported model without real data integration is incomplete — it WILL require rework when data is connected because forward() signatures, tensor shapes, and preprocessing assumptions are all guesswork without real data. Implement model and data pipeline together, not sequentially.

### Step 1: Build complete Module structure

Create ONE top-level Module that nests all components:

```
class MyMultimodalModel(MegatronModule):
    def __init__(self, ...):
        self.vision_encoder = VisionTransformer(...)   # ViT
        self.vision_projection = MLP(...)              # bridge
        self.language_model = TransformerDecoder(...)   # LLM
        self.generation_head = DiffusionHead(...)      # VAE/gen (if applicable)

    def forward(self, ...):
        # Wire all components in one forward pass
        vision_features = self.vision_encoder(images)
        projected = self.vision_projection(vision_features)
        output = self.language_model(tokens, visual_embeds=projected)
        return output
```

⚠️ **COMPLETENESS CHECK**: Before writing this class, cross-reference your Analysis 0 checklist. Every component in the checklist MUST appear as `self.xxx = ...` in `__init__`. If your source model has 6 submodules but your implementation only has 4, you are dropping components. Common mistakes:
- Dropping action heads / generation heads because "they're separate"
- Dropping projection layers because "they're small"
- Dropping state encoders / auxiliary modules because "they're not the main model"
- Implementing a "simplified" version that skips components — this ALWAYS requires rework later

The top-level model must be a 1:1 structural mirror of the source. Freeze/unfreeze is a training config decision, not an architecture decision.

Reference implementations:
- `flagscale/models/megatron/qwen2_5_vl/qwen2_5_vl_model.py` — ViT + projection + language
- `flagscale/models/megatron/llava_onevision/` — multimodal with vision encoder

### Step 2: Checkpoint conversion (all weights at once)

Convert the ENTIRE checkpoint into the nested structure in one pass:
- Map source weight names → target weight names for ALL components
- Handle shape transforms (transpose, reshape, split/merge heads)
- Verify: `model.load_state_dict()` with `strict=True` — zero missing, zero unexpected keys
- **Completeness sanity check**: Compare source vs converted tensor counts. If you loaded 1200 tensors but only converted 500, you dropped a submodule. Go back and include it. Common mistake: excluding vision encoders or VAE because "they'll be frozen" — wrong, they must still be in the model.

### Step 3: Real data adaptation

Implement `get_batch` with actual dataset as part of the porting deliverable — not a separate follow-up task. This is the primary verification mechanism:
- Tokenizer mismatches → vocab index errors (caught instantly)
- Preprocessing differences → shape mismatches in forward pass
- Sequence length issues → OOM or padding bugs
- Missing special tokens → silent training degradation (caught by loss comparison)

Use the real dataset the model will train on. If the full dataset isn't ready, use a representative subset.

**No dummy data.** `get_batch` must NEVER use `torch.rand`/`torch.zeros`/`torch.randn` or any synthetic tensors — not during development, not for "shape debugging", not as a placeholder. Always load real data from the start. If the data pipeline isn't working, fix it before proceeding. Dummy data hides every bug that matters (tokenizer, format, special tokens, sequence length).

**Own your dataset code.** Do NOT import the source project's dataset/dataloader classes (e.g., `from data.dataset_base import PackedDataset`, `sys.path.insert(0, BAGEL_DIR)`). External dataset code is unaware of Megatron's parallelism contract — it won't call `broadcast_data` for TP, won't guard inputs by PP stage, and may break under multi-node. Instead: read the source dataset code to understand the data format and preprocessing, then write your own dataset + `get_batch` inside FlagScale that reads the same data files with correct parallelism handling. Reuse data *files*, never data *code*.

**Parallelism-aware design.** When the target is distributed training, `get_batch` must handle:
- **TP**: All TP ranks receive identical input — use `broadcast_data` from `megatron.training.utils`
- **PP**: Only first stage needs tokens, only last needs labels — guard with `pre_process`/`post_process`
- **DP**: Different micro-batch per rank — handled by sampler, don't break it with global indexing

Read an existing `get_batch` (e.g., `train_gpt.py`, `train_qwen2_5_vl.py`) before writing yours. Copy the broadcast pattern.

### Step 4: First forward pass = verification

The first successful forward pass with real data that produces a finite loss IS the structural verification. If loss is produced:
- All weight shapes are correct
- All component connections work
- Data pipeline feeds correct formats
- Tokenizer is compatible

---

## Verification Standard

| Level | Criterion | What it proves |
|-------|-----------|----------------|
| 1 | `load_state_dict(strict=True)` passes | All weights mapped correctly |
| 2 | Forward pass with real data → finite loss | Model structure is correct, data pipeline works |
| 3 | Loss decreases over 50-100 steps | Model is learning, gradients flow through all components |
| 4 | Loss curve matches reference within tolerance | Numerical equivalence with source implementation |

Level 2 is the minimum bar before declaring "porting works". Levels 3-4 confirm correctness.

---

## Multimodal Module Nesting

For models with multiple modalities (vision + language + generation):

**Architecture pattern**:
- Top-level Module owns ALL sub-modules
- Pipeline parallelism splits at the top level via `pre_process`/`post_process`/`add_encoder`/`add_decoder`
- Each sub-module is a standard MegatronModule — no standalone verification needed
- Cross-component connections (vision→projection→language) are wired in the top-level `forward()`

**Pipeline stage assignment** (typical VL model):
- Stage 0: vision encoder + projection (when `add_encoder=True`)
- Stages 1..N-1: language model transformer layers
- Stage N: output head (when `post_process=True`)

**Critical wiring points**:
- Vision encoder output → projection layer → language model input embeddings
- Position IDs must account for visual tokens (image token positions)
- Attention mask must handle mixed visual/text sequences
- Loss computation must mask visual token positions appropriately

**Common pitfalls**:
- Vision encoder produces non-zero embeddings but wrong dtype (fp32 vs bf16)
- Projection output shape doesn't match language model hidden_size
- Rotary embeddings applied to visual token positions incorrectly
- Generation components (VAE) not receiving gradients due to detach

---

## Failure Pivot Discipline

**2-strike rule**: Same error category twice → STOP. Don't attempt 3rd fix.

1. Pause execution
2. Root cause audit — re-read relevant source end-to-end
3. Identify systemic gap (wrong assumption upstream, not local bug)
4. Report to user with new approach
5. Only proceed after confirmation

Error categories: shape/dimension, import/module, parallelism, data pipeline, config.

---

## get_batch Under Parallelism

`get_batch` is a critical porting surface:
- **TP**: All TP ranks must receive identical input (use `broadcast_data`)
- **PP**: Only first stage needs tokens, only last needs labels
- **CP**: Sequence split across ranks — correct position IDs and masks
- **DP**: Different micro-batch per rank (handled by sampler)

Verify: print shapes on rank 0 and rank 1, confirm TP ranks match, DP ranks differ.

---

## Related Skills

- `train-reproduce` — establish verified baseline before porting
- `train-config` — generate FlagScale training configuration
- `train-run` — launch training with ported model
- `train-precision-alignment` — verify numerical alignment
- `train-data-prep` — prepare real dataset for verification
