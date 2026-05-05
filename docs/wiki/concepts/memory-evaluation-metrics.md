# Memory Evaluation Metrics

Status: implementation snapshot as of 2026-04-30.

QTRM MemoryOS evaluation now reports layered answer metrics instead of a single
permissive hit flag.

## Why

The old score counted a case as correct if any normalized answer alias appeared
anywhere in the generated completion. That is useful for fast regression tests,
but it can hide important failures:

- answer is present but surrounded by extra copied evidence;
- answer repeats many times;
- `UNKNOWN` is correct but emitted as `UNKNOWN UNKNOWN ...`;
- completion contains both correct and conflicting values;
- prompt/evidence leakage looks like an answer.

## Deterministic Metrics

Code:

- `src/qtrm_mm/eval/memory_retrieval.py`
- `scripts/95_eval_memory_retrieval.py`
- `scripts/116_rescore_memory_eval.py`

Per record:

| Field | Meaning |
| --- | --- |
| `hit` | Backward-compatible permissive score. True if normalized alias appears, or expected-UNKNOWN case emits UNKNOWN. |
| `exact_match` | Canonical short answer exactly equals an alias after stripping `Answer:` and terminal punctuation. |
| `normalized_exact` | Canonical short answer equals an alias after case/punctuation/spacing normalization. |
| `normalized_contains` | Any normalized alias appears anywhere in the completion. This is the old positive-match behavior. |
| `unknown_correct` | Expected-UNKNOWN case emitted UNKNOWN. |
| `match_type` | `exact`, `normalized_exact`, `normalized_contains`, `unknown_exact`, `unknown_contains`, or `none`. |
| `needs_human_audit` | Record is a miss or only a loose/extra-text match. |
| `audit_reasons` | Reasons such as `answer_miss`, `loose_contains_match`, or `unknown_with_extra_text`. |
| `judge_status` | Currently `not_run`; audit JSONL contains prompts for later LLM-as-judge or human review. |

Summary reports all of these by overall, mode, category, and task family:

```text
accuracy
exact_match_rate
normalized_exact_rate
normalized_contains_rate
unknown_correct_rate
human_audit_rate
retrieved_target_rate
all_targets_retrieved_rate
target_recall_mean
```

## Audit Queue

`scripts/95_eval_memory_retrieval.py` can write an audit file during generation:

```bash
PYTHONPATH=src python scripts/95_eval_memory_retrieval.py \
  ... \
  --jsonl-out runs/eval/run.jsonl \
  --audit-jsonl-out runs/eval/run_audit.jsonl
```

Existing eval files can be rescored without rerunning the model:

```bash
PYTHONPATH=src python scripts/116_rescore_memory_eval.py \
  runs/eval/run.jsonl \
  runs/eval/run_scored.jsonl \
  --audit-jsonl-out runs/eval/run_audit.jsonl
```

The audit JSONL is deliberately model-agnostic. Each item includes question,
aliases, completion, deterministic match type, audit reasons, and a compact
`judge_prompt` for later LLM-as-judge or manual review.

## Current Gated Workspace Re-Score

Rescored outputs:

- `runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_gated_workspace_s050_scored.jsonl`
- `runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_gated_workspace_s050_scored.jsonl`
- `runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_gated_workspace_s050_scored.jsonl`

Key result for the main donor-vs-residual gate:

| Mode | Permissive hits | Exact / normalized exact | Human-audit items |
| --- | ---: | ---: | ---: |
| `donor_only_with_evidence` | 26/72 | 9/72 | 63/72 |
| `qtrm_residual_with_evidence` | 49/72 | 24/72 | 48/72 |

Strict causality gate remains unchanged at the permissive level:

| Mode | Permissive hits | Exact / normalized exact | Human-audit items |
| --- | ---: | ---: | ---: |
| `qtrm_residual_head_off_with_evidence` | 26/72 | 9/72 | 63/72 |
| `qtrm_workspace_only_with_evidence` | 49/72 | 24/72 | 48/72 |
| `qtrm_workspace_gate_off_with_evidence` | 49/72 | 24/72 | 48/72 |

Interpretation:

- The residual adapter still improves over donor-only.
- The exact short-answer quality is much lower than the permissive hit score.
- Many "correct" outputs still require audit because they include repetition,
  extra explanation, or copied context.
- Gated-workspace causality is still not proven: gate-off has the same
  permissive and strict counts as the full/workspace-only path.

## Claiming Rule

Use `hit` for fast regression tracking, but use `normalized_exact_rate`,
`human_audit_rate`, and task-family breakdown before making architecture claims.

A strong result should improve all three:

```text
permissive hit rate up
normalized exact rate up
human audit rate down
```
