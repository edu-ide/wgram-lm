# Expanded Workspace/Core Ablation Proof

Claim: the expanded 72-case MemoryOS gate should separate full QTRM residual behavior from workspace-off and core-off ablations.

Positive drop means the ablated mode is worse than full `qtrm_residual_with_evidence` on the same expanded gate.

## Sources

- expanded donor/residual gate: `runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl`
- expanded workspace/core ablation gate: `runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_synth_generalization_s050.jsonl`
- expanded strict causality ablation gate: `runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_synth_generalization_s050.jsonl`

## Overall

| Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | ---: | ---: | ---: |
| qtrm_residual_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_workspace_off_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_core_off_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_coda_off_with_evidence | 39/72 | +10 | +0.139 |
| qtrm_residual_head_off_with_evidence | 26/72 | +23 | +0.319 |
| qtrm_donor_hidden_off_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_workspace_only_with_evidence | 49/72 | +0 | +0.000 |

## Completion Identity

| Mode | Same completions vs residual | Same rate |
| --- | ---: | ---: |
| qtrm_workspace_off_with_evidence | 72/72 | 1.000 |
| qtrm_core_off_with_evidence | 72/72 | 1.000 |
| qtrm_coda_off_with_evidence | 19/72 | 0.264 |
| qtrm_residual_head_off_with_evidence | 1/72 | 0.014 |
| qtrm_donor_hidden_off_with_evidence | 72/72 | 1.000 |
| qtrm_workspace_only_with_evidence | 72/72 | 1.000 |

## Current Interpretation

- Turning off QTRM residual logits causes a large drop, so the measured gain is genuinely in the residual head rather than donor-only generation.
- Turning off coda causes a smaller but real drop, so coda contributes to the residual behavior.
- Removing projected donor hidden states does not change this gate, so the current gain is not caused by direct donor-hidden prefix tokens.
- Workspace-only context matches full residual, but workspace-off also matches full residual; this gate still does not prove latent-workspace causality.

## Task-Family Drops

| Task family | Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | --- | ---: | ---: | ---: |
| abstention | qtrm_residual_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_workspace_off_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_core_off_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_coda_off_with_evidence | 20/24 | -2 | -0.083 |
| abstention | qtrm_residual_head_off_with_evidence | 1/24 | +17 | +0.708 |
| abstention | qtrm_donor_hidden_off_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_workspace_only_with_evidence | 18/24 | +0 | +0.000 |
| conflict | qtrm_residual_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_workspace_off_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_core_off_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_coda_off_with_evidence | 14/24 | +6 | +0.250 |
| conflict | qtrm_residual_head_off_with_evidence | 19/24 | +1 | +0.042 |
| conflict | qtrm_donor_hidden_off_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_workspace_only_with_evidence | 20/24 | +0 | +0.000 |
| multi_hop | qtrm_residual_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_workspace_off_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_core_off_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_coda_off_with_evidence | 5/24 | +6 | +0.250 |
| multi_hop | qtrm_residual_head_off_with_evidence | 6/24 | +5 | +0.208 |
| multi_hop | qtrm_donor_hidden_off_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_workspace_only_with_evidence | 11/24 | +0 | +0.000 |

## Interpretation Rule

- If workspace-off or core-off matches full residual, the current gain is not yet localized to that component.
- If an ablation drops below full residual, that component is contributing to the measured behavior.
- This is still a MemoryOS evidence-task proof, not a broad standalone-LM proof.
