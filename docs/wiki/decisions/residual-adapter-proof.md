# Residual Adapter Proof

Claim: QTRM is currently a donor-backed residual adapter, not a standalone donor-free language model.

This package fixes the current proof target: with donor logits intact, QTRM residual generation should improve evidence-sensitive MemoryOS tasks over donor-only generation while preserving the Qwen base language policy.

This is not a donor-free standalone-LM claim.

## Summary

| Eval | Source | Donor-only | QTRM residual | Delta hits | Delta accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| hard memory probe | runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl | 5/9 | 9/9 | +4 | +0.444 |
| held-out memory probe | runs/eval/memory_reasoning_heldout_qwen3_rerank_32tok_synth_generalization_s050.jsonl | 6/12 | 9/12 | +3 | +0.250 |
| **aggregate** | all listed evals | 11/21 | 18/21 | +7 | +0.333 |

## Task-Family Delta

| Task family | Donor-only | QTRM residual | Delta hits | Delta accuracy |
| --- | ---: | ---: | ---: | ---: |
| abstention | 0/7 | 7/7 | +7 | +1.000 |
| conflict | 6/8 | 6/8 | +0 | +0.000 |
| multi_hop | 5/6 | 5/6 | +0 | +0.000 |

## Limitations

- These evals prove residual-adapter usefulness on small MemoryOS probes, not broad language-model replacement.
- Donor-free `donor_logits_scale=0.0` generation remains a later standalone-student gate.
- The held-out set is still small; the next gate should expand to 50-100 balanced cases.
- Retrieval success is tracked separately from answer accuracy; evidence recall alone is not enough.

## Next Gates

1. Expand held-out MemoryOS cases across conflict, multi-hop, abstention, and Korean authority conflict.
2. Keep donor-only and QTRM residual modes paired in every eval.
3. Add workspace/core ablations when claiming latent-memory or recursive-core causality.
4. Resume donor-free work only after OPD/GKD/DistiLLM-style rollout training is implemented.
