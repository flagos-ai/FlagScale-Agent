---
name: train-data-prep
description: Prepare training data for FlagScale (Megatron backend). Covers two pipelines — (A) text-only via preprocess_data.py
  (.bin+.idx), and (B) multimodal via Megatron-Energon (WebDataset .tar + TaskEncoder). Includes tokenizer setup, data validation,
  Energon dataset config, custom TaskEncoder patterns, and multi-dataset blending.
keywords:
- data
- dataset
- preprocess
- tokenizer
- tokenize
- bin
- idx
- jsonl
- data preparation
- data format
- energon
- megatron-energon
- webdataset
- wds
- multimodal
- image
- video
- vision
- vqa
- caption
- sft
- 数据
- 数据准备
- 数据预处理
- 数据格式
- 分词器
- 多模态
parameters:
- name: data_dir
  description: Directory to store processed data. Use shared storage for multi-node training.
- name: tokenizer_dir
  description: Directory containing tokenizer files. Use shared storage for multi-node training.
requires:
- workspace-layout
suggests:
- train-env-setup
constraints:
- id: no_dummy_data
  description: NEVER use dummy/synthetic data (torch.rand/zeros/ones) for model verification
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - torch.rand
    - torch.zeros
    - torch.ones
    - dummy_input
    - synthetic_data
    - fake_data
  prompt: Check if the agent is creating dummy/synthetic data instead of using real data pipeline
  correction: ALL verification must flow through the real data pipeline. No exceptions.
- id: parallelism_aware_data
  description: Data pipeline MUST handle parallelism (broadcast_data for TP, pre_process/post_process for PP)
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - get_batch
    - DataLoader
    - train_valid_test_datasets_provider
  prompt: Check if the data pipeline implementation handles parallelism (TP broadcast, PP guards, DP sampling)
  correction: Every get_batch must call broadcast_data() for TP, guard with pre_process/post_process for PP.
- id: know_parallelism_first
  description: Must know parallelism strategy before implementing data pipeline
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - get_batch
    - DataLoader
    - Dataset
    - data_pipeline
  prompt: Check if parallelism strategy (TP/PP/DP/EP/CP) is determined before implementing data pipeline
  correction: Determine parallelism strategy first (use train-parallel-strategy skill). Data pipeline depends on it.
- id: validate_after_preprocess
  description: Validate preprocessed data before training
  trigger:
    keywords:
    - preprocess data
    - .bin file
    - .idx file
    - tokenize dataset
  prompt: Check if data validation is planned after preprocessing
  correction: 'After preprocessing, validate: check file sizes, sample a few records, verify token counts match expectations.'
- id: data_model_interface_first
  description: Document data→model interface contract before implementing
  trigger:
    tools:
    - write_file
    - edit_file
    keywords:
    - data_pipeline
    - get_batch
    - implement data
    - create data
  prompt: Check if the data→model interface contract is documented before implementation
  correction: Document the data→model interface contract BEFORE writing code. Know what get_batch produces and what model.forward
    expects.
context_injection:
  always:
  - 'CRITICAL: Data Pipeline is EQUALLY Important as Model Adaptation'
  - 'ABSOLUTE RULE: No Parallelism = Failed Megatron Integration'
  by_tool:
    write_file:
    - 'CRITICAL: Parallelism Strategy is a Prerequisite'
    - 'Pipeline A: Text-Only'
    - 'Pipeline B: Megatron-Energon'
    edit_file:
    - 'CRITICAL: Parallelism Strategy is a Prerequisite'
    shell:
    - Data Format
    - Pre-processed Demo Data
---
# Data Preparation for FlagScale

## CRITICAL: Data Pipeline is EQUALLY Important as Model Adaptation

Data pipeline is NOT a follow-up task after model porting. It is a co-equal deliverable that must be designed alongside the model. If you implement model.forward() without knowing what get_batch produces, you WILL rewrite the model later.

**Before writing ANY data pipeline or model code:**
1. Document the data→model interface contract (see `train-model-porter` skill, Analysis 4)
2. Know the parallelism strategy
3. Understand the source data format AND the model's expected input format

🚫 **NEVER use dummy/synthetic data** (torch.rand/zeros/ones) for model verification. ALL verification must flow through the real data pipeline. No exceptions — not for "quick checks", not for "shape testing", not as a placeholder.

## ABSOLUTE RULE: No Parallelism = Failed Megatron Integration

**A data pipeline without parallelism awareness is a FAILED Megatron integration.** This is not optional, not "add later", not "nice to have". It is the fundamental contract of distributed training in Megatron.

If your get_batch does NOT:
- Call `broadcast_data()` for TP rank consistency
- Guard inputs with `pre_process`/`post_process` for PP stages
- Respect DP micro-batch distribution via sampler

...then it WILL deadlock or produce wrong results at runtime. Every single get_batch in Megatron handles parallelism. There are zero exceptions.

## CRITICAL: Parallelism Strategy is a Prerequisite

**Before implementing ANY data pipeline code, you MUST know the parallelism strategy.**

Data loading in FlagScale is NOT independent of parallelism. Every `get_batch` implementation must handle ALL applicable parallelism dimensions:

| Parallelism | Data Requirement | Implementation |
|-------------|-----------------|----------------|
| **TP** (Tensor Parallel) | All TP ranks receive IDENTICAL input | Use `broadcast_data()` from `megatron.training.utils` |
| **PP** (Pipeline Parallel) | Only first stage needs tokens, only last needs labels | Guard with `pre_process`/`post_process` flags |
| **DP** (Data Parallel) | Different micro-batch per rank | Handled by sampler — don't use global indexing |
| **EP** (Expert Parallel) | Token-to-expert routing must be consistent across EP ranks | Ensure dispatch/combine tensors align with expert sharding |
| **CP** (Context Parallel) | Sequence split across ranks | Correct position IDs, attention masks, and loss masks per rank |
| **SP** (Sequence Parallel) | Activations distributed along sequence dim | Automatically handled by framework when enabled with TP |

**If you don't know the full parallelism configuration (TP/PP/DP/EP/CP/SP), STOP and determine it first** (use `train-parallel-strategy` skill).

Consider special cases:
- MoE models: EP affects how tokens are routed — data pipeline must produce consistent routing inputs
- Long sequences with CP: position IDs and loss masks must be correctly split per rank
- Packed samples: multiple sequences in one sample require careful attention mask construction under parallelism

---

FlagScale supports two data pipelines depending on modality:

| Pipeline | Modality | Format | Tool |
|----------|----------|--------|------|
| **A: preprocess_data.py** | Text-only (pretrain/SFT) | `.bin` + `.idx` | `Megatron-LM-FL/tools/preprocess_data.py` |
| **B: Megatron-Energon** | Multimodal (image/video + text) | WebDataset `.tar` + Energon config | Convert scripts + `megatron.energon` |

---

## Pipeline A: Text-Only (bin + idx)

### Data Format

FlagScale (Megatron backend) requires training data as paired binary files:
- `<prefix>.bin` — tokenized content in binary format
- `<prefix>.idx` — index file mapping document boundaries

Both files must exist with the same prefix. The `data_path` in training config references the prefix WITHOUT file extension.

### Pre-processed Demo Data (Quick Start)

```bash
mkdir -p {data_dir} && cd {data_dir}
wget -c https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx
wget -c https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin
cd ..
```

### Preprocessing Command

Input: JSONL file, one JSON object per line with a `text` field.

```bash
cd <Megatron-LM-FL-path>
python tools/preprocess_data.py \
    --input <input.jsonl> \
    --output-prefix {data_dir}/<output_name> \
    --tokenizer-type <tokenizer_type> \
    --tokenizer-model <tokenizer_path> \
    --workers <num_workers> \
    --append-eod
```

### Tokenizer Types

| Tokenizer Type | Models | Key Files |
|---------------|--------|-----------|
| `QwenTokenizerFS` | Qwen, Qwen2.5, Qwen3 | `qwen.tiktoken`, `tokenizer_config.json` |
| `Llama3Tokenizer` | LLaMA 3 | `tokenizer.model` |
| `GPT2BPETokenizer` | GPT-2 style | `vocab.json`, `merges.txt` |
| `SentencePieceTokenizer` | LLaMA 2, many others | `tokenizer.model` |
| `HuggingFaceTokenizer` | Generic HF tokenizer | HF tokenizer directory |

### Preprocessing Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `--input` | Input JSONL file path | Required |
| `--output-prefix` | Output path prefix (no extension) | Required |
| `--tokenizer-type` | Tokenizer type from table above | Required |
| `--tokenizer-model` | Path to tokenizer files | Required for most types |
| `--workers` | Number of parallel workers | Default: 1 |
| `--append-eod` | Append end-of-document token | Recommended for pretraining |
| `--json-keys` | JSON field(s) to tokenize | Default: `text` |
| `--split-sentences` | Split into sentences | For sentence-level tasks |

### Verify Preprocessed Data

```bash
ls -lh {data_dir}/<output_name>.bin {data_dir}/<output_name>.idx
python -c "
import numpy as np
idx = np.fromfile('{data_dir}/<output_name>.idx', dtype=np.int64)
print(f'Documents: {len(idx) - 1}')
"
```

---

## Pipeline B: Megatron-Energon (Multimodal)

### Architecture Overview

The Energon pipeline has three layers:

```
Raw Data (JSON + images/videos)
    │
    ▼  [Convert Script — offline, one-time]
WebDataset .tar files + .nv-meta/
    │
    ▼  [ChatMLWebdataset — runtime decode]
ChatMLSample (paths + conversation string)
    │
    ▼  [TaskEncoder.encode_sample — runtime]
Training tensors (tokenized text + processed images/videos)
```

Key design: images/videos are stored as FILE PATHS in the tar, not pixel data. Actual loading happens at training time in the TaskEncoder. This makes data preparation fast and storage efficient.

### Step 1: Prepare Raw Data

Input format: a JSON file (`dataset.json`) with entries like:

```json
[
  {
    "id": "sample_001",
    "conversations": [
      {"from": "user", "value": "<image>\nDescribe this image."},
      {"from": "assistant", "value": "The image shows a cat sitting on a windowsill."}
    ],
    "images": ["images/cat_001.jpg"]
  },
  {
    "id": "sample_002",
    "conversations": [
      {"from": "user", "value": "<video>\nWhat happens in this video?"},
      {"from": "assistant", "value": "A person is walking through a park."}
    ],
    "videos": [{"video_path": "videos/park.mp4", "fps": 2.0}]
  }
]
```

- `conversations`: multi-turn dialogue in GPT format (`from` + `value`)
- `images`: list of image file paths (relative to vision_root)
- `videos`: list of video objects with `video_path` and optional `fps`
- Use `<image>` and `<video>` placeholders in conversation text to mark where visual content appears
- Additional fields (state, action, metadata) are automatically captured as `.metadata` in the tar

### Step 2: Convert to WebDataset

FlagScale provides convert scripts under `tools/datasets/`:

**For VLM (QwenVL, Qwen2.5-VL, Qwen3-VL):**

```bash
cd <FlagScale-root>
python tools/datasets/qwenvl/convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root /path/to/raw/data \
    --output-root /path/to/output/wds \
    --vision-root /path/to/images_and_videos \
    --json dataset.json \
    --images-key images \
    --videos-key videos \
    --max-samples-per-tar 10000 \
    --dp-size <num_data_parallel_ranks> \
    --shuffle-tars
```

**For VLA (robotics with state/action):**

```bash
python tools/datasets/vla/convert.py \
    --dataset-root /path/to/raw/data \
    --output-root /path/to/output/wds \
    --vision-root /path/to/images \
    --json dataset.json \
    --max-samples-per-tar 10000 \
    --dp-size <num_data_parallel_ranks>
```

**For NVIDIA-style VQA (LLaVA, MIMO):**

Use the Megatron-LM-FL built-in converter + `energon prepare`:

```bash
# 1. Convert to WebDataset
python examples/multimodal/convert_llava_pretrain_to_wds.py

# 2. Generate Energon metadata
cd /path/to/wds/output
energon prepare ./
# Select: VQAWebdataset, field_map: image→jpg, context→json[0][value], answers→json[1][value]
```

### What the Convert Script Produces

```
output_root/
├── wds-<dp_size>/
│   ├── tar_0000000.tar      # WebDataset shards
│   ├── tar_0000001.tar
│   ├── ...
│   └── .nv-meta/
│       ├── dataset.yaml      # Energon dataset type declaration
│       └── split.yaml        # Train/val/test split
```

Each tar contains samples with fields:
- `__key__`: sample ID (e.g., "000001")
- `.conversation`: JSON string of the dialogue
- `.jpgs`: pickle-encoded list of image paths
- `.videos`: pickle-encoded list of video frame paths
- `.metadata`: JSON-encoded extra fields (VLA only)

### Step 3: Dataset Config YAML

**Top-level Metadataset config** (referenced by training script):

```yaml
# dataset_config.yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  train:
    datasets:
      - weight: 1.0
        path: /path/to/output/wds/wds-8
        subflavors:
          augmentation: false
  val:
    datasets:
      - weight: 1.0
        path: /path/to/output/wds/wds-8
        subflavors:
          augmentation: false
```

- `weight`: blending weight when mixing multiple datasets (normalized automatically)
- `path`: points to the directory containing `.nv-meta/`
- `subflavors`: key-value metadata passed to TaskEncoder (e.g., augmentation flags)

**Auto-generated dataset.yaml** (inside `.nv-meta/`, created by convert script):

For QwenVL/VLA (ChatML format):
```yaml
__module__: tools.datasets.qwenvl.data.energon.chatml
__class__: ChatMLWebdataset
field_map:
  conversation: conversation
  imgs: jpgs
  videos: videos
```

For NVIDIA VQA format:
```yaml
__module__: megatron.energon
__class__: VQAWebdataset
field_map:
  image: jpg
  context: json[0][value]
  answers: json[1][value]
```

### Step 4: TaskEncoder (Data Processing at Training Time)

The TaskEncoder transforms raw samples into training tensors. FlagScale provides ready-made encoders:

| Model Family | TaskEncoder Location | Sample Type |
|-------------|---------------------|-------------|
| QwenVL / Qwen2.5-VL / Qwen3-VL | `tools/datasets/qwenvl/data/dataset_helpers.py` | `ChatMLSample` |
| VLA (robotics) | `tools/datasets/vla/data/dataset_helpers.py` | `ChatMLSample` |
| LLaVA-OneVision | `flagscale/models/megatron/llava_onevision/dataset_helpers.py` | `InterleavedSample` |
| MIMO | `Megatron-LM-FL/examples/mimo/data/energon_vlm_task_encoder.py` | `VQASample` |

**What TaskEncoder.encode_sample does:**
1. Parse conversation JSON
2. Load images from disk paths (`PIL.Image.open`)
3. Apply visual transforms (resize, normalize)
4. Tokenize text with model-specific tokenizer
5. Insert visual placeholder tokens (`<|image_pad|>`, `<|video_pad|>`) at `<image>`/`<video>` positions
6. Generate labels with `IGNORE_IDX = -100` masking non-assistant tokens

### Writing a Custom TaskEncoder

For new model architectures or custom data formats, implement a TaskEncoder subclass:

```python
from dataclasses import dataclass
from megatron.energon import Batch, DefaultTaskEncoder
from tools.datasets.qwenvl.data.energon.chatml import ChatMLSample
# Or use built-in: from megatron.energon import VQASample, InterleavedSample

@dataclass
class MyTaskSample:
    __key__: str
    __subflavors__: dict
    images: list  # processed image tensors
    text: np.ndarray  # tokenized input
    target: np.ndarray  # labels

@dataclass
class MyBatch(Batch):
    __keys__: list[str]
    images: torch.Tensor
    text: torch.Tensor
    target: torch.Tensor

class MyTaskEncoder(DefaultTaskEncoder[ChatMLSample, MyTaskSample, MyBatch, dict]):
    def __init__(self):
        super().__init__()
        # Initialize tokenizer, image processor, etc.

    def encode_sample(self, sample: ChatMLSample) -> MyTaskSample:
        # 1. Parse sample.conversation (JSON string)
        # 2. Load images from sample.imgs (list of paths)
        # 3. Tokenize and create labels
        return MyTaskSample(...)

    def batch(self, samples: list[MyTaskSample]) -> MyBatch:
        # Pad sequences, stack tensors
        return MyBatch(...)

    def encode_batch(self, batch: MyBatch) -> dict:
        # Convert to dict for Megatron forward pass
        return dataclasses.asdict(batch)
```

### Writing a Custom Sample Type + WebDataset Factory

For data formats beyond ChatML (e.g., custom binary fields):

```python
from dataclasses import dataclass
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory

@dataclass
class MySample(Sample):
    images: list[str]
    text: str
    custom_field: dict  # any additional data

class MyWebdataset(DefaultDecoderWebdatasetFactory[MySample]):
    __sample_type__ = MySample

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        # Register custom decoders if needed
        # self._decoder = Decoder([MyCustomHandler()])
```

Then in `.nv-meta/dataset.yaml`:
```yaml
__module__: my_module.my_dataset
__class__: MyWebdataset
field_map:
  images: jpgs
  text: txt
  custom_field: metadata
```

### Step 5: Integrate with Training Script

The training script creates an Energon DataLoader:

```python
from megatron.energon import WorkerConfig, get_train_dataset, get_savable_loader

worker_config = WorkerConfig(
    rank=parallel_state.get_data_parallel_rank(),
    world_size=parallel_state.get_data_parallel_world_size(),
    num_workers=args.num_workers,
    data_parallel_group=parallel_state.get_data_parallel_group(),
)

train_ds = get_train_dataset(
    dataset_config_path,  # path to top-level Metadataset YAML
    batch_size=args.micro_batch_size,
    task_encoder=MyTaskEncoder(),
    worker_config=worker_config,
)

train_loader = get_savable_loader(train_ds, worker_config=worker_config)
```

`get_savable_loader` returns a loader that supports `save_state()` / `restore_state()` for checkpoint resumption.

---

## Multi-Dataset Blending

Both pipelines support weighted blending of multiple datasets. The concept is the same — weighted sampling — but the implementation layer differs.

### Text-only (Pipeline A) — BlendedDataset

Configured via `data_path` with interleaved weight/path format. Weights are automatically normalized to sum to 1.0.

```yaml
data:
  data_path:
    - 0.7
    - /path/to/dataset_a    # .bin + .idx prefix
    - 0.3
    - /path/to/dataset_b
```

Implementation: `BlendedDataset` in Megatron Core builds a global index array at startup, mapping each sample index to `(dataset_idx, sample_offset)`. Uses C-level index arrays for efficient random access on pre-tokenized binary data.

Key behaviors:
- Weights are proportional — `[0.7, 0.3]` and `[7, 3]` produce the same blend
- Each dataset must have matching `.bin` + `.idx` files
- Blending happens at the sample level (not shard level)
- Supports deterministic resumption via index checkpointing

### Multimodal (Pipeline B) — Energon Metadataset

Configured via Metadataset YAML. Each dataset entry points to a WebDataset directory with `.nv-meta/`.

```yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  train:
    datasets:
      - weight: 0.7
        path: /path/to/wds_a/wds-8
        subflavors:
          augmentation: false
      - weight: 0.3
        path: /path/to/wds_b/wds-8
        subflavors:
          augmentation: true
  val:
    datasets:
      - weight: 1.0
        path: /path/to/wds_val/wds-8
```

Key behaviors:
- Weights are normalized per split (train/val/test independently)
- Each dataset can use a different WebDataset factory class (e.g., `ChatMLWebdataset` vs `VQAWebdataset`)
- `subflavors` are passed to TaskEncoder as `sample.__subflavors__` — use for per-dataset flags (augmentation, prompt template, etc.)
- Blending happens at the DataLoader level (shard-based sampling)
- TaskEncoder handles heterogeneous sample types polymorphically
- Supports `save_state()` / `restore_state()` for checkpoint resumption

### Mixing Different Modalities

You can blend text-only and multimodal datasets in the same Metadataset — as long as the TaskEncoder handles both sample types:

```yaml
splits:
  train:
    datasets:
      - weight: 0.6
        path: /path/to/multimodal_wds    # images + text
        subflavors: { modality: vl }
      - weight: 0.4
        path: /path/to/text_only_wds     # text only
        subflavors: { modality: text }
```

The TaskEncoder checks `sample.__subflavors__["modality"]` to decide whether to load images or skip visual processing.

---

## Data Validation Checklist

### Text-only
1. Both `.bin` and `.idx` files exist at the configured `data_path` prefix
2. Files have non-zero size
3. `vocab_size` in training config matches the tokenizer

### Multimodal
1. Tar files exist and are non-empty: `ls -lh /path/to/wds/*.tar`
2. `.nv-meta/dataset.yaml` exists and points to correct WebDataset factory class
3. `.nv-meta/split.yaml` lists all tar files
4. Image/video paths in tar are accessible from training nodes
5. Quick validation — decode one sample:

```bash
python -c "
import webdataset as wds
import pickle, json
ds = wds.WebDataset('/path/to/wds/tar_0000000.tar').decode()
sample = next(iter(ds))
print('Keys:', list(sample.keys()))
if 'jpgs' in sample:
    paths = pickle.loads(sample['jpgs']) if isinstance(sample['jpgs'], bytes) else sample['jpgs']
    print('Image paths:', paths)
if 'conversation' in sample:
    conv = json.loads(sample['conversation']) if isinstance(sample['conversation'], str) else sample['conversation']
    print('Conversation turns:', len(conv))
"
```

---

## Download Best Practices

- Always use `wget -c` (resume) for large files
- For files > 1GB, verify size after download
- Use proxy when available: check `echo $HTTP_PROXY`
- For git clone on large repos, use `--depth 1`
- If download speed < 500KB/s for a large file, stop and ask user
- Run large downloads as separate commands, not chained with `&&`

---

## Data Pipeline Comprehension

Before writing any data processing code (TaskEncoder, preprocessing scripts, dataset classes, data config), you must trace the full data pipeline for the target model. This is not a checklist — it is a thinking framework. The goal is to understand the chain from raw data to model input so that your code is correct on the first attempt.

### The Three-Link Chain

Every training data pipeline has three links. You must understand all three before writing code:

```
Source Format                Processing Operations           Model Input Interface
─────────────                ────────────────────           ─────────────────────
What does the raw data       What transformations           How does the training
look like? File format,      happen? Tokenization,          loop consume data?
schema, fields, modalities.  image processing, padding,     get_batch signature,
                             label masking, special          tensor shapes, dtypes,
                             tokens, sequence packing.       batch collation.
```

### How to Trace the Chain

1. **Source Format** — Read the raw data files or their documentation. Identify:
   - File format (JSONL, WebDataset tar, parquet, custom binary)
   - Schema: what fields exist, what types, what modalities (text, image paths, video, state/action)
   - Sample count and size characteristics
   - Any special conventions (placeholder tokens like `<image>`, conversation format like ChatML)

2. **Processing Operations** — Read the preprocessing/encoding code. Identify:
   - Tokenizer: which one, special tokens, chat template
   - Visual processing: resize, normalize, frame sampling for video
   - Label construction: which tokens are masked (IGNORE_IDX = -100), loss mask logic
   - Sequence construction: how multi-turn conversations become a single sequence
   - Packing: whether multiple samples are packed into one sequence

3. **Model Input Interface** — Read the training script's `get_batch` or equivalent. Identify:
   - What keys the model's `forward()` expects
   - Tensor shapes and dtypes for each input
   - How visual tokens are interleaved with text tokens
   - Batch collation: padding strategy, attention mask construction

### Persist Your Understanding

After tracing the chain, write a concise summary to memory covering:
- Source format and key fields
- Critical transformations (especially non-obvious ones like label masking rules or special token insertion)
- Model input tensor shapes and the mapping from data fields to model inputs

This persistence serves two purposes: (1) the engineering gate clears, allowing you to write code, and (2) future sessions can pick up without re-reading everything.

### When This Applies

This framework applies whenever you are about to write or modify:
- TaskEncoder or encode_sample implementations
- Data preprocessing scripts
- Dataset class definitions
- Data configuration files that affect how data is loaded or processed
- Training script data loading (get_batch, data_provider)

It does NOT apply to:
- Downloading or copying data files
- Simple config changes (paths, weights)
- Running existing preprocessing commands with known parameters

---

## Related Skills

- `train-config` — configure data paths in training YAML after data preparation
- `train-run` — launch training with prepared data
- `train-reproduce` — prepare data for baseline reproduction experiments
