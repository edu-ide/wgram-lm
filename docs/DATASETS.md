# QTRM-MemoryOS Dataset Pipeline

This package downloads and normalizes real datasets before training.

## Default sources

Text / pretraining:
- `HuggingFaceTB/smollm-corpus`, configs: `cosmopedia-v2`, `fineweb-edu-dedup`

Math / reasoning:
- `AI-MO/NuminaMath-CoT`
- `open-r1/OpenR1-Math-220k`, config: `default`

Multimodal SFT:
- `HuggingFaceM4/the_cauldron`, default configs: `scienceqa,ai2d,chartqa,docvqa,textvqa`
- fallbacks: `lmms-lab/ScienceQA-IMG`, `lmms-lab/ChartQA`

## Smoke run

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
export PYTHONPATH=$PWD/src
bash scripts/run_all_smoke_multimodal.sh
```

## 4090 run

```bash
PROFILE=4090 TEXT_SAMPLES=20000 MATH_SAMPLES=4000 MM_SAMPLES_PER_CONFIG=1000 \
  bash scripts/run_all_4090_multimodal.sh
```

## DGX Spark run

```bash
PROFILE=dgx TEXT_SAMPLES=100000 MATH_SAMPLES=20000 MM_SAMPLES_PER_CONFIG=5000 \
  bash scripts/run_all_dgx_multimodal.sh
```

## Output files

```text
data/raw/text_train.jsonl
data/raw/math_train.jsonl
data/raw/mm_train.jsonl
data/raw/images/...
data/docs/downloaded_dataset_seed_corpus.md
```

## Notes

- The Cauldron has per-subdataset licenses; review licenses before full training or redistribution.
- The visual feature path in this scaffold uses a deterministic image patch featurizer for smoke training. Production should replace it with Qwen3.5 vision encoder features or SigLIP/Qwen visual embeddings.
- The hash tokenizer is only for debug/scaffold training. Production should use the donor tokenizer or a trained tokenizer.
