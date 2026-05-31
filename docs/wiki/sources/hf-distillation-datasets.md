# Hugging Face Distillation Dataset Intake

Date: 2026-04-30.

Purpose: use current public datasets before spending teacher budget on new
offline generation.

## Decision

Do not generate all offline data from scratch. Start with public HF datasets,
normalize them into the QTRM teacher-record schema, then use GPT-5.5 xhigh only
for small gold seed/verification and Qwen3.6-27B for online/token-compatible
distillation.

```text
HF datasets
-> cheap baseline warmup and evidence/routing records

GPT-5.5 xhigh
-> small high-quality gold seed and verifier/judge corrections

local llama-server Qwen3.6
-> cheap offline expansion after schema is stable

DGX Qwen3.6-27B
-> online top-k KL, on-policy correction, MSA routing supervision
```

## First Wave Manifest

Local manifest:
`configs/hf_distill_datasets.yaml`

| Name | HF dataset | Role | Why selected |
| --- | --- | --- | --- |
| `yana_reasoning_dpo` | `Yana/ft-llm-2026-reasoning-dpo` | QTRM preference + CoT-to-latent warmup | 2026 reasoning DPO data with chosen/rejected answers. |
| `noesis_50k_reasoning_sft` | `AMAImedia/NOESIS-50K-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54` | QTRM multilingual reasoning warmup | 2026 multilingual reasoning SFT; includes Korean among many languages. |
| `ragognize_evidence` | `F4biian/RAGognize` | MSA routing + evidence gate | 2026 RAG dataset with documents and hallucination annotations. |
| `halluclaim_76k` | `lrsbrgrn/HalluClaim-76k` | logical/causal evidence bottleneck | 2026 RAG hallucination/evidence detection data. |

## Converter

Implementation:

- `src/wgram_lm/distill/hf_dataset_convert.py`;
- `src/wgram_lm/distill/training_mix.py`;
- `scripts/131_convert_hf_distill_dataset.py`;
- `scripts/132_convert_first_wave_hf_distill_smoke.sh`;
- `scripts/133_build_hf_distill_training_mix.py`;
- `scripts/134_run_hf_first_wave_warmup.sh`;
- `tests/test_hf_distill_manifest.py`;
- `tests/test_hf_distill_converters.py`;
- `tests/test_hf_distill_convert_script.py`.
- `tests/test_hf_distill_training_mix.py`.

All converted rows use:
`src/wgram_lm/distill/teacher_schema.py`.

Example local smoke:

```bash
PYTHONPATH=src python3 scripts/131_convert_hf_distill_dataset.py \
  --adapter yana_reasoning_dpo \
  --local-jsonl /tmp/sample.jsonl \
  --out runs/distill/hf_yana_smoke.jsonl \
  --max-rows 10
```

Example HF smoke:

```bash
PYTHONPATH=src uv run --with datasets --with pyyaml \
  python scripts/131_convert_hf_distill_dataset.py \
  --adapter ragognize \
  --hf-id F4biian/RAGognize \
  --split train \
  --out data/filtered/hf_ragognize_teacher_records_s100.jsonl \
  --max-rows 100
```

Use `--streaming` only for datasets that are too large to load normally. Some
HF streaming iterators can produce the requested output and still crash during
cleanup, so the first-wave smoke path defaults to non-streaming.

First-wave batch smoke:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  uv run --with datasets --with pyyaml \
  bash scripts/132_convert_first_wave_hf_distill_smoke.sh
```

Set `MAX_ROWS=3` for a fast schema check or `MAX_ROWS=100` for the first
training-preview set.

First-wave QTRM training mix:

```bash
PYTHONPATH=src python3 scripts/133_build_hf_distill_training_mix.py \
  --input data/filtered/hf_distill_smoke/yana_reasoning_dpo_s100.jsonl \
  --input data/filtered/hf_distill_smoke/noesis_50k_reasoning_sft_s100.jsonl \
  --input data/filtered/hf_distill_smoke/ragognize_evidence_s100.jsonl \
  --input data/filtered/hf_distill_smoke/halluclaim_76k_s100.jsonl \
  --out data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl \
  --max-rows-per-source 100
```

Current local output:
`data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl`

Mix counts:

- 400 total rows;
- 100 rows each from Yana, NOESIS, RAGognize, and HalluClaim;
- 171 preference rows with `chosen`/`rejected`;
- 200 workspace-evidence rows with `memory_docs`;
- 166 rows with positive `target_doc_ids`;
- 34 unsupported RAGognize rows converted to `chosen=NEEDS_SEARCH` and
  `rejected=<unsupported answer>` so hallucinated answers are not SFT targets.

Warmup runner:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/134_run_hf_first_wave_warmup.sh
```

The runner sets:

- `DATA_JSONL=data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl`;
- `MULTIMODAL=0`;
- config `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml`.

## Use Order

1. Convert `Yana/ft-llm-2026-reasoning-dpo` smoke rows.
2. Convert `NOESIS-50K` smoke rows.
3. Convert `RAGognize` smoke rows and inspect `memory_docs`/`target_doc_ids`.
4. Convert `HalluClaim-76k` smoke rows and inspect support/refute mapping.
5. Build `qtrm_hf_first_wave_mix_s400.jsonl`.
6. Run a 400-step QTRM warmup smoke on the mixed JSONL.
7. Run workspace/core/evidence ablations before expanding to full dataset.
8. Use GPT-5.5 xhigh only for rows where public data is missing:
   Korean religious/value synthesis, local MemoryOS documents, and high-quality
   counterfactual evidence.

## Claim Boundary

Recent upload date is not proof of quality. Every dataset must pass:

- 20-100 row preview inspection;
- schema conversion success rate;
- duplicate/empty answer filtering;
- small train/eval smoke;
- ablation gate if used to support architecture claims.
