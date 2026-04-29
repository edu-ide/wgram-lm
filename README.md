# QTRM Multimodal MemoryOS

This repository is a research-oriented implementation scaffold for a multimodal recursive cognitive core:

```text
Qwen3.5 donor / teacher route
+ QTRM recursive latent workspace
+ Parcae-style stable recurrence
+ JEPA-style latent-state prediction
+ MemoryOS with LLM Wiki, Harrier text embeddings, and visual embeddings
```

The package is designed to support two modes:

1. **Standalone smoke mode**: train and test a small randomly initialized QTRM model without downloading Qwen weights.
2. **Donor adapter mode**: load a Qwen3.5-style multimodal donor through Hugging Face Transformers, freeze most donor weights, and train QTRM adapters/core/head modules.

The code intentionally separates:

- core architecture,
- backend kernels,
- Qwen donor adapters,
- multimodal workspace projection,
- MemoryOS indexing,
- training and inference workflows.

## Important status

This is a runnable research scaffold, not a production-ready model. It includes trainable PyTorch reference backends so smoke tests and small training runs work without official KDA/FlashAttention kernels. For production-scale runs, use the strict backend mode and install official backends.

## Quick start

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
export PYTHONPATH=$PWD/src

bash scripts/run_all_smoke_multimodal.sh
```

## Main scripts

```text
scripts/00_setup_env.sh              Create venv and install dependencies
scripts/01_smoke_forward.sh          Random tensor forward/backward test
scripts/02_train_smoke_text.sh       Tiny text-only training
scripts/03_train_smoke_multimodal.sh Tiny image+text training with random image features
scripts/04_build_text_memory.sh      Build Harrier/FAISS text memory index
scripts/05_build_visual_memory.sh    Build visual memory index from image paths
scripts/06_infer_multimodal.sh       Run multimodal inference scaffold
scripts/run_all_smoke_multimodal.sh  End-to-end smoke workflow
```

## Recommended real route

```text
1. Run standalone smoke tests.
2. Train 38M / 160M QTRM standalone versions.
3. Build MemoryOS indexes with Harrier text embedding.
4. Load Qwen3.5-2B-Base donor through qtrm_mm.qwen_donor.
5. Freeze donor; train QTRM projector/core/controller adapters.
6. Use Qwen3.6-27B or Qwen3.5-4B as teacher to generate multimodal RAG/tool/verifier traces.
7. Healing tune with general text/code + reasoning + multimodal RAG traces.
```

## Repository layout

```text
configs/
  smoke_multimodal.yaml
  qwen35_2b_adapter.yaml
  qwen35_4b_adapter.yaml

docs/
  MULTIMODAL_ARCHITECTURE.md
  TRAINING_WORKFLOW.md
  BACKENDS.md

src/qtrm_mm/
  config.py
  norm.py
  rotary.py
  attention.py
  ffn.py
  mixers.py
  stability.py
  workspace.py
  core.py
  heads.py
  multimodal_projector.py
  qtrm_model.py
  qwen_donor.py
  losses.py
  infer.py

src/qtrm_mm/backends/
  registry.py
  delta_backend.py
  attention_backend.py

src/qtrm_mm/memoryos/
  chunk.py
  text_index.py
  visual_index.py
  retrieve.py
  wiki_compile.py

src/qtrm_mm/training/
  synthetic_data.py
  train.py
```

## Design invariants

```text
1. The recurrent QTRM core only runs over fixed latent workspace tokens.
2. The causal decoder produces final text; the latent workspace is not a replacement for causal decoding.
3. Qwen3.5 donor weights are not blindly merged into QTRM; they are frozen or adapted through projectors/LoRA-style adapters.
4. Harrier is used for text semantic memory, not visual embeddings.
5. Visual memory uses donor vision embeddings, SigLIP-style embeddings, or reference CLIP-like embeddings.
6. Dense embeddings are candidate generators; source-backed verifier remains required for truth.
7. Strict backend mode must be used for production runs so fallback kernels are not accidentally used.
```
