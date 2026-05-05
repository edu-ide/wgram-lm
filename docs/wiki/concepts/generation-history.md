# Generation History

QTRM generation history is a research artifact, not a convenience log.

The goal is to preserve every useful output path:

```text
raw generation -> eval history -> labeled failure rows -> curated training data
```

Do not train directly on raw history. Raw model outputs can contain repetition,
decoy-copy, hallucination, or format leakage. They must be filtered by an
evaluator, verifier, or human label before becoming preference, hard-negative,
or SFT data.

## Paths

Default interactive generation history:

```text
runs/history/generations/YYYY-MM-DD.jsonl
```

Default eval generation history:

```text
runs/history/evals/YYYY-MM-DD.jsonl
```

Curated future training rows should live separately:

```text
data/curated/failures/*.jsonl
data/curated/preferences/*.jsonl
data/curated/span_reader/*.jsonl
```

## Row Contract

Each row should preserve:

- timestamp;
- source script;
- checkpoint and config;
- mode or answer channel;
- prompt/question;
- full output and extracted completion;
- hit or unlabeled status;
- failure type;
- retrieved evidence summary;
- answer-channel metadata;
- run metadata such as token limits and guards.

## Current Implementation

Code:

```text
src/qtrm_mm/history.py
```

Interactive inference:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  HISTORY_JSONL=auto \
  bash scripts/90_infer_with_donor.sh "질문"
```

Interactive inference now defaults to language-safe donor-preserving mode:

```text
LANGUAGE_SAFE=1
DONOR_LOGITS_SCALE=1.0
QTRM_LOGITS_SCALE=0.0
QTRM_RESIDUAL_CLAMP=0.0
STOP_AFTER_SENTENCE=1
```

History metadata records these guard values so fluent donor-preserving samples
are not mistaken for donor-free QTRM language-policy evidence. The default
guard records `mode=language_safe_donor`; raw residual experiments should use
`LANGUAGE_SAFE=0` or explicit logit-scale overrides.

Disable history:

```bash
HISTORY_JSONL=none bash scripts/90_infer_with_donor.sh "질문"
```

Memory eval history:

```bash
PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --history-jsonl-out auto
```

Disable eval history:

```bash
PYTHONPATH=src uv run python scripts/95_eval_memory_retrieval.py \
  --history-jsonl-out none
```

## Research Rule

History is useful only if failures are kept. A good history file should contain:

- successes for regression targets;
- misses for architecture failure ledgers;
- repetition and format failures for verifier training;
- decoy-copy failures for hard-negative evidence training;
- UNKNOWN behavior for abstention calibration.

The next step is a curator script that converts history rows into explicit
labels:

```text
good | miss | repetition | decoy_copy | hallucination | format_leak | needs_search
```
