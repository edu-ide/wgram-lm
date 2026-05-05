# Expanded Workspace/Core Ablation Proof

Claim: this expanded 72-case MemoryOS gate measures whether residual behavior is localized to workspace, core, coda, residual-head, donor-hidden, or workspace-gate paths.

Positive drop means the ablated mode is worse than full `qtrm_residual_with_evidence` on the same expanded gate.

## Sources

- workspace evidence path: `runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_trained_s050.jsonl`

## Overall

| Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | ---: | ---: | ---: |
| qtrm_residual_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_workspace_off_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_core_off_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_coda_off_with_evidence | 2/4 | +1 | +0.250 |
| qtrm_residual_head_off_with_evidence | 0/4 | +3 | +0.750 |
| qtrm_donor_hidden_off_with_evidence | 0/0 | +3 | +0.750 |
| qtrm_workspace_only_with_evidence | 0/0 | +3 | +0.750 |
| qtrm_workspace_gate_off_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_workspace_memory_off_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_core_context_off_with_evidence | 3/4 | +0 | +0.000 |
| qtrm_evidence_bottleneck_off_with_evidence | 3/4 | +0 | +0.000 |

## Completion Identity

| Mode | Same completions vs residual | Same rate |
| --- | ---: | ---: |
| qtrm_workspace_off_with_evidence | 4/4 | 1.000 |
| qtrm_core_off_with_evidence | 4/4 | 1.000 |
| qtrm_coda_off_with_evidence | 0/4 | 0.000 |
| qtrm_residual_head_off_with_evidence | 0/4 | 0.000 |
| qtrm_donor_hidden_off_with_evidence | 0/0 | 0.000 |
| qtrm_workspace_only_with_evidence | 0/0 | 0.000 |
| qtrm_workspace_gate_off_with_evidence | 4/4 | 1.000 |
| qtrm_workspace_memory_off_with_evidence | 4/4 | 1.000 |
| qtrm_core_context_off_with_evidence | 4/4 | 1.000 |
| qtrm_evidence_bottleneck_off_with_evidence | 4/4 | 1.000 |

## Current Interpretation

- Turning off QTRM residual logits causes a large drop, so the measured gain is genuinely in the residual head rather than donor-only generation.
- Turning off coda causes a smaller but real drop, so coda contributes to the residual behavior.
- Turning off the workspace memory gate does not change score or completions, so this run does not prove gated latent-memory causality.
- Removing workspace-side evidence memory does not change score or completions, so this run does not prove workspace-memory evidence causality.
- Turning off gated core context injection does not change score or completions, so this run does not prove direct core-context causality.

## Task-Family Drops

| Task family | Mode | Hits | Hit drop vs residual | Accuracy drop vs residual |
| --- | --- | ---: | ---: | ---: |
| abstention | qtrm_residual_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_workspace_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_core_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_coda_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_residual_head_off_with_evidence | 0/2 | +2 | +1.000 |
| abstention | qtrm_donor_hidden_off_with_evidence | 0/0 | +2 | +1.000 |
| abstention | qtrm_workspace_only_with_evidence | 0/0 | +2 | +1.000 |
| abstention | qtrm_workspace_gate_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_workspace_memory_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_core_context_off_with_evidence | 2/2 | +0 | +0.000 |
| abstention | qtrm_evidence_bottleneck_off_with_evidence | 2/2 | +0 | +0.000 |
| multi_hop | qtrm_residual_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_workspace_off_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_core_off_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_coda_off_with_evidence | 0/2 | +1 | +0.500 |
| multi_hop | qtrm_residual_head_off_with_evidence | 0/2 | +1 | +0.500 |
| multi_hop | qtrm_donor_hidden_off_with_evidence | 0/0 | +1 | +0.500 |
| multi_hop | qtrm_workspace_only_with_evidence | 0/0 | +1 | +0.500 |
| multi_hop | qtrm_workspace_gate_off_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_workspace_memory_off_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_core_context_off_with_evidence | 1/2 | +0 | +0.000 |
| multi_hop | qtrm_evidence_bottleneck_off_with_evidence | 1/2 | +0 | +0.000 |

## Interpretation Rule

- If workspace-off or core-off matches full residual, the current gain is not yet localized to that component.
- If an ablation drops below full residual, that component is contributing to the measured behavior.
- This is still a MemoryOS evidence-task proof, not a broad standalone-LM proof.
