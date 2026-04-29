# Training Workflow

## Phase 0: Smoke

```bash
bash scripts/run_all_smoke_multimodal.sh
```

Goals:

```text
- import works
- forward/backward works
- text-only path works
- multimodal path works with random image features
- loss is finite
- checkpoint saves
```

## Phase 1: 4090 prototype

Train small standalone models before loading heavy donors.

```text
38M → 160M
seq_len 512~1024
workspace 64~128
outer steps 2~4
```

## Phase 2: MemoryOS

Build basic memory indexes.

```bash
bash scripts/04_build_text_memory.sh data/docs memory/text
bash scripts/05_build_visual_memory.sh data/images memory/visual
```

## Phase 3: Qwen3.5-2B donor adapter

```text
Load donor.
Freeze vision encoder, embeddings, and most backbone weights.
Train:
  - multimodal projector
  - QTRM recursive core
  - controller heads
  - JEPA predictor
  - LoRA/adapters if added
```

## Phase 4: Teacher trace distillation

Teacher candidates:

```text
Qwen3.6-27B
Qwen3.5-4B
Qwen2.5-VL-72B / 7B for visual tasks
```

Trace labels:

```text
THINK
LOOK
READ_OCR
RETRIEVE_TEXT
RETRIEVE_IMAGE
TOOL
VERIFY
REVISE
ANSWER
```

## Phase 5: Healing tune

After donor insertion / franken-init:

```text
70% general high-quality text/code
20% reasoning/math
10% instruction/RAG/tool/multimodal traces
```

## Phase 6: Production backend

Switch from reference backends:

```yaml
attention_backend: flash_attn
delta_backend: fla_gated_delta
strict_backends: true
```

Keep reference backend only for debugging.
