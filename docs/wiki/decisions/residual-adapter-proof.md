# Residual Adapter Proof

Claim: QTRM is currently a donor-backed residual adapter, not a standalone donor-free language model.

This package fixes the current proof target: with donor logits intact, QTRM residual generation should improve evidence-sensitive MemoryOS tasks over donor-only generation while preserving the Qwen base language policy.

This is not a donor-free standalone-LM claim.

## Summary

| Eval | Source | Donor-only | QTRM residual | Delta hits | Delta accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| hard memory probe | runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl | 5/9 | 9/9 | +4 | +0.444 |
| held-out memory probe | runs/eval/memory_reasoning_heldout_qwen3_rerank_32tok_synth_generalization_s050.jsonl | 6/12 | 9/12 | +3 | +0.250 |
| expanded held-out memory probe | runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl | 26/72 | 49/72 | +23 | +0.319 |
| **aggregate** | all listed evals | 37/93 | 67/93 | +30 | +0.323 |

## Task-Family Delta

| Task family | Donor-only | QTRM residual | Delta hits | Delta accuracy |
| --- | ---: | ---: | ---: | ---: |
| abstention | 1/31 | 25/31 | +24 | +0.774 |
| conflict | 25/32 | 26/32 | +1 | +0.031 |
| multi_hop | 11/30 | 16/30 | +5 | +0.167 |

## Limitations

- These evals include a 72-case expanded synthetic held-out gate, but they still prove MemoryOS residual-adapter usefulness rather than broad language-model replacement.
- Donor-free `donor_logits_scale=0.0` generation remains a later standalone-student gate.
- The expanded gate is template-generated; the next gate should add larger hand-authored and web-like held-out cases.
- Retrieval success is tracked separately from answer accuracy; evidence recall alone is not enough.

## Next Gates

1. Add hand-authored/web-like held-out cases across conflict, multi-hop, abstention, and Korean authority conflict.
2. Keep donor-only and QTRM residual modes paired in every eval.
3. Add workspace/core ablations when claiming latent-memory or recursive-core causality.
4. Resume donor-free work only after OPD/GKD/DistiLLM-style rollout training is implemented.
