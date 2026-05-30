# LoRA Rank-8 Depth-Sweep Clarification

Status: active diagnostic clarification, 2026-05-30.

## Purpose

Record the exact interpretation of the `fc8c9133` M1/M2 LoRA work so future
agents do not confuse three separate claims:

1. the loop-wise Mythos LoRA rank-8 implementation,
2. the local causal forced-choice depth-sweep signal, and
3. the still-rejected autoregressive renderer / promoted-checkpoint claim.

## Commit And Config

Source commit:

```text
fc8c9133 feat: restore stochastic breadth LoRA steering & elastic depth policy (M1/M2)
tag: v2026.05.29-qtrm-lora-m1-m2
```

The S040 answer-loop config used by that commit sets:

```yaml
answer_state_loop_mythos_update_enabled: true
answer_state_loop_mythos_loop_index_enabled: true
answer_state_loop_mythos_loop_dim: 64
answer_state_loop_mythos_lora_rank: 8
answer_state_loop_mythos_act_enabled: true
```

The runner writes the local candidate to:

```text
/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt
```

Mirror path:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt
```

Checkpoint tensor audit confirms rank 8:

```text
answer_state_loop_mythos_lora_down.weight  (8, 512)
answer_state_loop_mythos_lora_up.weight    (512, 8)
answer_state_loop_mythos_lora_scale.weight (16, 8)
```

## Historical 8/20 Result

The remembered `8/20` result is from the same rank-8 S040 checkpoint, but it is
not a free-generation score. It is the aggregate hit count in:

```text
/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/causal_forced_choice_smoke4.jsonl
```

The 20 rows are `5 eval modes x 4 cases`:

| Mode | Hits |
| --- | ---: |
| `donor_only_no_evidence` | 0/4 |
| `qtrm_core_off_no_evidence` | 0/4 |
| `qtrm_core_steps_8_no_evidence` | 2/4 |
| `qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence` | 4/4 |
| `qtrm_core_steps_8_answer_halt_gate_off_no_evidence` | 2/4 |
| **Aggregate** | **8/20** |

Therefore `8/20` should be read as a causal forced-choice/ablation aggregate
for the same checkpoint, not as "20 free-generation prompts solved."

## 2026-05-30 Free-Generation Depth Sweep

After the `8/20` clarification, the same checkpoint was re-tested with an
explicit free-generation depth sweep.

Artifacts:

```text
local_eval/necessary_condition_smoke/local_lora_mythos_free_generation_depth_sweep_smoke8.jsonl
local_eval/necessary_condition_smoke/local_lora_mythos_cfc_depth_sweep_smoke8.jsonl
```

Free generation contract:

```text
scoring: generation
max_cases: 8
modes: donor, core_off, depth1, depth2, depth4, depth8
max_new_tokens: 8
```

Free generation results:

| Mode | Hits | Typical completion |
| --- | ---: | --- |
| `donor_only_no_evidence` | 0/8 | prompt-local number such as `50004` |
| `qtrm_core_off_no_evidence` | 0/8 | `!!!!!!!!` |
| `qtrm_core_steps_1_no_evidence` | 0/8 | `Answer:Answer:Answer:Answer:` |
| `qtrm_core_steps_2_no_evidence` | 0/8 | `Answer:Answer:Answer:Answer:` |
| `qtrm_core_steps_4_no_evidence` | 0/8 | `1600000` |
| `qtrm_core_steps_8_no_evidence` | 0/8 | `1600000` |
| **Aggregate** | **0/48** | - |

Matched causal forced-choice depth sweep:

| Mode | Hits |
| --- | ---: |
| `donor_only_no_evidence` | 0/8 |
| `qtrm_core_off_no_evidence` | 0/8 |
| `qtrm_core_steps_1_no_evidence` | 2/8 |
| `qtrm_core_steps_2_no_evidence` | 4/8 |
| `qtrm_core_steps_4_no_evidence` | 3/8 |
| `qtrm_core_steps_8_no_evidence` | 3/8 |
| **Aggregate** | **12/48** |

Interpretation: this checkpoint has a real forced-choice answer-basin signal,
but free generation remains closed across the tested depth ladder. Depth helps
candidate discrimination, but the renderer/mouth path still collapses into
format loops or numeric attractors.

## Local Necessity Smoke

Eval contract:

```text
scoring: causal_forced_choice
heldout size: 2 rows
evidence/retrieval/MemoryOS: off
gold answer: 300015
```

Canonical no-donor-logit-blend result:

| Mode | Hits | Completion |
| --- | ---: | --- |
| `donor_only_no_evidence` | 0/2 | `100004,100008,100012` |
| `qtrm_core_off_no_evidence` | 0/2 | `__FORCED_CHOICE_TIE__` |
| `qtrm_core_steps_1_no_evidence` | 0/2 | `100004,100008,100012` |
| `qtrm_core_steps_4_no_evidence` | 2/2 | `300015` |
| `qtrm_core_steps_8_no_evidence` | 2/2 | `300015` |

Donor-logit scale-1.0 blend result:

| Mode | Hits | Completion |
| --- | ---: | --- |
| `qtrm_core_steps_1_donor_scale_1p0_no_evidence` | 0/2 | `100004,100008,100012` |
| `qtrm_core_steps_4_donor_scale_1p0_no_evidence` | 0/2 | `100004,100008,100012` |
| `qtrm_core_steps_8_donor_scale_1p0_no_evidence` | 0/2 | `100004,100008,100012` |

Interpretation: the donor backbone can be present, but full donor-logit blending
at scale 1.0 masks the QTRM answer-path signal on this smoke. The observed gain
is the canonical QTRM recurrent answer path at depth 4/8, not a generic donor
logit mix.

## Boundary

This clarification does not overturn
`ouro-answer-loop-joint-decoder-s040-reject.md`.

The S040 answer-loop checkpoint remains rejected as:

- a promoted autoregressive renderer,
- a general generation-ready checkpoint, and
- a second canonical reasoning core.

Why: historical held-out gates record generation `0/8`, the 2026-05-30 explicit
free-generation depth sweep records `0/48`, and the original forced-choice smoke
had decoder-off beating full. The rank-8 LoRA result is therefore a useful local
forced-choice depth-sweep diagnostic, not a broad model quality claim.

## Current Use

Use the rank-8 S040 checkpoint as a local necessity-test baseline for:

- donor/core-off/depth-ladder causality,
- depth 4/8 recurrence opening the target answer basin,
- checking whether later SFT or renderer changes preserve the causal signal.

Do not use it alone to claim free-generation, broad benchmark, or final raw
intelligence promotion. Promotion requires larger heldout depth sweeps and a
renderer/generation gate that does not depend on private answer-path patches.

## 2026-05-30 S041 Donor-Preserving Free-Generation Smoke

Follow-up decision:

```text
docs/wiki/decisions/2026-05-30-s041-donor-preserving-freegen-smoke.md
```

The S041 inference-only donor-preserving alpha sweep used the same S040
checkpoint and tested free generation on 8 held-out rows:

```text
donor-only: 2/8 exact
QTRM-only depth2/depth4/depth8: 0/8 exact
donor_scale=1.0 + qtrm_scale in {0.25, 0.5, 1.0}: 2/8 exact at depths 2/4/8
```

Interpretation: donor preservation helps avoid the worst private-renderer
collapse, but inference-time mixing does not beat donor-only.  This strengthens
the conclusion that the next UltraData run must train the donor-preserving
free-running renderer contract instead of relying on dataset scale alone.
