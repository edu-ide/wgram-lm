# Generation Verifier

Status: probe-only, 2026-05-01.

Primary source notes:
[Generation Verifier And Reranking](../sources/generation-verifier-reranking.md).

The generation verifier is a small set of heads that scores QTRM's own
generated candidate text for output failures:

- `repeat`: the completion is falling into a repeated n-gram loop;
- `stop`: a stop-sensitive prompt hit the token cap instead of stopping;
- `quality`: the candidate is acceptable under the current narrow labels.

This is not a replacement for the donor, the residual head, or the recursive
core. It is an output-failure probe that can later become a reranker or decoding
gate only after held-out calibration passes.

## Why It Was Added

The Qwen-Scope SAE feature probe found useful repeat-related signals, but those
signals did not generalize cleanly on the first 50-sample split. The sparse
feature detector fit the discovery half and failed on holdout. Therefore
Qwen-Scope should remain diagnostic, and QTRM needs a model-visible output
failure objective trained on its own generated failures.

## Mechanism

The dataset is built from QTRM eval JSONL:

```text
prompt + generated completion -> text
repeated_2gram_rate            -> repeat target
token cap on stop prompt       -> stop target
not repeat and not stop        -> quality target
```

The verifier heads are enabled by:

```yaml
model:
  generation_verifier_enabled: true

train:
  loss_generation_verifier_weight: 1.0
  trainable_param_policy: generation_verifier_only
```

Critical architecture decision: the heads read the coda/post-norm last valid
text hidden state, not the recursive `z_h` state alone.

The first `z_h`-pooled version learned mostly label priors because `z_h` did not
directly contain the candidate completion text. Moving the heads to the final
text-side hidden state gives the verifier access to the actual generated output
while still staying inside the QTRM forward pass.

## Artifacts

Dataset builder:

```text
scripts/141_build_generation_verifier_dataset.py
```

Smoke train runner:

```text
scripts/142_run_generation_verifier_s020.sh
```

Evaluation:

```text
scripts/143_eval_generation_verifier.py
```

Generated dataset:

```text
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier.jsonl
```

Checkpoint:

```text
runs/qwen35_2b_4090_generation_verifier_s020/last.pt
```

Evaluation JSON:

```text
runs/qwen35_2b_4090_generation_verifier_s020/eval_generation_verifier_s50.json
```

## First Smoke Result

The 50-row dataset contains:

| Target | Count |
| --- | ---: |
| repeat failures | 18 |
| stop failures | 6 |
| quality passes | 28 |

After the coda-text pooling fix and a 20-step verifier-only smoke train:

| Head | Fixed 0.5 F1 | Best in-sample F1 | Note |
| --- | ---: | ---: | --- |
| repeat | 0.100 | 0.757 | Signal exists, threshold not calibrated. |
| stop | 0.000 | 0.250 | Too few positives and weak label definition. |
| quality | 0.000 | 0.800 | Signal exists, threshold not calibrated. |

This is in-sample only. It proves that the coda-text verifier can see useful
output-failure signal on the smoke set. It does not prove a deployable decoding
gate yet.

## Research-Driven Failure Ledger

Failure:
the first verifier smoke had in-sample best-threshold signal, but no held-out
calibration proof. Fixed `0.5` thresholds were poor, and the stop head was weak.

Evidence:
the all-50 in-sample smoke reached best-threshold F1 `0.757` for repeat and
`0.800` for quality, but fixed-threshold F1 was `0.100` and `0.000`
respectively. Stop best-threshold F1 was only `0.250`.

Likely component:
the coda-text verifier has some output-failure signal, but calibration and stop
labels are underpowered.

Alternative explanations:

- the 50-row set is too small;
- the verifier overfits prompt/category artifacts;
- stop failure is not the same phenomenon as repetition;
- the current 20-step smoke is too short for stable calibration.

Prior work checked:

- unlikelihood training for repetition penalties;
- SimCTG/contrastive search for degeneration;
- FUDGE-style future discriminators for guided decoding;
- generator-reranker systems for candidate selection.

Architecture candidates:

| Rank | Candidate | Prototype |
| ---: | --- | --- |
| 1 | Full-candidate verifier reranker | Split train/calibration/holdout and use calibration thresholds on holdout. |
| 2 | FUDGE-style prefix verifier | Train partial-prefix labels and adjust next-token scores. |
| 3 | SimCTG/contrastive decoding path | Add representation contrast or contrastive search around residual logits. |

Recommended candidate:
candidate 1, because it is the smallest falsifiable step and does not disturb
the donor-backed residual architecture.

Smallest next experiment:
train verifier heads on train split only, choose thresholds on calibration
split, and report holdout metrics.

Acceptance gate:
repeat and quality should retain useful F1 on holdout with calibration-selected
thresholds. Stop should not be accepted until precision improves.

## Split Calibration Smoke

New tooling:

```text
scripts/144_split_generation_verifier_dataset.py
scripts/145_calibrate_generation_verifier_eval.py
configs/qwen35_2b_4090_generation_verifier_split_s020.yaml
```

Split artifacts:

```text
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_train.jsonl
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_calibration.jsonl
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_holdout.jsonl
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_split_summary.json
```

Split distribution:

| Split | Rows | Repeat Failures | Stop Failures | Quality Pass |
| --- | ---: | ---: | ---: | ---: |
| train | 29 | 11 | 3 | 16 |
| calibration | 10 | 3 | 1 | 6 |
| holdout | 11 | 4 | 2 | 6 |

Train command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
PYTHONPATH=src \
DATA=data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_train.jsonl \
bash scripts/142_run_generation_verifier_s020.sh \
  configs/qwen35_2b_4090_generation_verifier_split_s020.yaml
```

Checkpoint:

```text
runs/qwen35_2b_4090_generation_verifier_split_s020/last.pt
```

Calibration-selected holdout report:

```text
runs/qwen35_2b_4090_generation_verifier_split_s020/calibrated_holdout_report.json
```

| Head | Calibration Threshold | Calibration F1 | Holdout F1 | Holdout Precision | Holdout Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat | 0.4948 | 0.667 | 0.571 | 0.667 | 0.500 |
| stop | 0.3495 | 0.286 | 0.222 | 0.143 | 0.500 |
| quality | 0.4918 | 0.800 | 0.750 | 0.600 | 1.000 |

Decision:
repeat and quality have weak but real held-out signal in this tiny smoke.
Stop remains rejected. The verifier should next be tested as a reranker over
multiple candidate generations, not as a hard decoding gate.

## Candidate Rerank Probe

Added candidate generation and rerank tooling:

```text
scripts/92_eval_qtrm_logits.py --num-candidates --do-sample
scripts/146_eval_generation_verifier_rerank.py
scripts/147_summarize_generation_format.py
```

First candidate-rerank smoke:

- prompts: 8;
- candidates per prompt: 3;
- total candidates: 24;
- generator checkpoint:
  `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`;
- verifier checkpoint:
  `runs/qwen35_2b_4090_generation_verifier_split_s020/last.pt`.

Original repeat/stop-only labels:

| Metric | Value |
| --- | ---: |
| baseline quality rate | 1.000 |
| reranked quality rate | 0.750 |
| oracle quality rate | 1.000 |
| reranked repeat failure rate | 0.250 |

After format-aware labels, where visible `<think>`/meta-reasoning is a quality
failure:

| Metric | Value |
| --- | ---: |
| baseline quality rate | 0.625 |
| reranked quality rate | 0.250 |
| oracle quality rate | 0.750 |
| selected changed rate | 0.750 |

Decision:
the current post-hoc verifier reranker is rejected. It does not reliably select
the best candidate and can make selection worse. This triggered the
[Answer Channel Contract](answer-channel-contract.md) root-architecture probe.

## Claim Boundary

Accepted claims:

- QTRM now has optional repeat/stop/quality verifier heads.
- The verifier can be trained without updating the rest of the model.
- Coda-text pooling is materially better than `z_h`-only pooling for judging
  generated text on the 50-row smoke set.

Rejected claims until further evidence:

- Do not call this a solved repetition fix.
- Do not use a hard `0.5` threshold as a production gate.
- Do not claim generalization without a held-out split.
- Do not claim stop control from the current six-positive stop subset.

## Next Gates

1. Build a larger train/calibration/holdout verifier dataset from QTRM
   generations.
2. Train the verifier for longer with class balancing, especially for stop
   positives.
3. Add a rerank-only evaluation before any hard decoding gate.
4. If reranking helps, test whether verifier-guided generation improves
   held-out answer quality without reducing donor fluency.
