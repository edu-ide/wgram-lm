# Expanded Workspace/Core Ablation Proof

Claim: the expanded 72-case MemoryOS gate should separate full QTRM residual behavior from workspace-off and core-off ablations.

Positive drop means the ablated mode is worse than full `qtrm_residual_with_evidence` on the same expanded gate.

## Sources

- expanded donor/residual gate: `runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl`
- expanded workspace/core ablation gate: `runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_synth_generalization_s050.jsonl`

## Overall

| Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | ---: | ---: | ---: |
| qtrm_residual_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_workspace_off_with_evidence | 49/72 | +0 | +0.000 |
| qtrm_core_off_with_evidence | 49/72 | +0 | +0.000 |

Current result: workspace-off and core-off match the full residual score, so this run does not localize the residual gain to the latent workspace or recursive core.

## Completion Identity

| Mode | Same completions vs residual | Same rate |
| --- | ---: | ---: |
| qtrm_workspace_off_with_evidence | 72/72 | 1.000 |
| qtrm_core_off_with_evidence | 72/72 | 1.000 |

## Task-Family Drops

| Task family | Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | --- | ---: | ---: | ---: |
| abstention | qtrm_residual_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_workspace_off_with_evidence | 18/24 | +0 | +0.000 |
| abstention | qtrm_core_off_with_evidence | 18/24 | +0 | +0.000 |
| conflict | qtrm_residual_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_workspace_off_with_evidence | 20/24 | +0 | +0.000 |
| conflict | qtrm_core_off_with_evidence | 20/24 | +0 | +0.000 |
| multi_hop | qtrm_residual_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_workspace_off_with_evidence | 11/24 | +0 | +0.000 |
| multi_hop | qtrm_core_off_with_evidence | 11/24 | +0 | +0.000 |

## Interpretation Rule

- If workspace-off or core-off matches full residual, the current gain is not yet localized to that component.
- If an ablation drops below full residual, that component is contributing to the measured behavior.
- This is still a MemoryOS evidence-task proof, not a broad standalone-LM proof.
