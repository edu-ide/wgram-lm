# Ouro Answer-Loop Joint Decoder S040 Reject

Status: rejected renderer/reasoning-path probe, 2026-05-07.

## Purpose

Test whether an OpenMythos/Parcae-style stable recurrent answer loop can repair
the QTRM renderer bottleneck when trained jointly with the in-loop decoder and
answer halt gate.

## Artifacts

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040.yaml
scripts/256_run_qtrm_ouro_answer_loop_joint_decoder_s040.sh
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/
```

Implementation additions are guarded by default-disabled config fields:

```text
answer_state_loop_mythos_update_enabled
answer_state_loop_mythos_loop_index_enabled
answer_state_loop_mythos_lora_rank
answer_state_loop_mythos_act_enabled
```

## Training Signal

The train slice became easy:

```text
final_path_ce: 2.4569
final_path_acc: 1.0000
self_rollout mismatch rate at step 40: 0.0000
```

This is not sufficient for promotion because the held-out runtime gates still
fail.

## Held-Out Gates

Gold-token rank probe:

```text
donor_only:    first_unique@1 0/4, all<=10 3/4
core_off:      first_unique@1 0/4, all<=10 4/4
core8 full:    first_unique@1 4/4, all<=10 0/4, max_rank_mean 13.00
decoder_off:   first_unique@1 4/4, all<=10 0/4, max_rank_mean 37.75
halt_gate_off: first_unique@1 0/4, all<=10 0/4
```

Generation smoke8:

```text
donor_only: 0/8
core_off: 0/8
core8 full: 0/8
decoder_off: 0/8
halt_gate_off: 0/8
```

Causal forced-choice smoke4:

```text
donor_only: 0/4
core_off: 0/4
core8 full: 2/4
decoder_off: 4/4
halt_gate_off: 2/4
```

## Decision

Reject as a promoted renderer or canonical reasoning checkpoint.

The Mythos-style stable answer loop improves some local/train diagnostics, but
it does not repair generation and does not beat the decoder-off ablation on the
forced-choice smoke. Therefore it must not become a second canonical reasoning
core.

Canonical interpretation:

```text
TRM/QTRM recursive core = main reasoning core
answer-state loop = readout/renderer/control
Mythos/OpenMythos ideas = stability references only
```

The next architecture work should strengthen the TRM core itself and then solve
renderer alignment without creating a parallel hidden answer path.

## 2026-05-30 Clarification

The `fc8c9133` M1/M2 work did use loop-wise Mythos LoRA rank 8:

```text
answer_state_loop_mythos_lora_rank: 8
```

The good local checkpoint path for the rank-8 diagnostic is:

```text
/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt
```

Tensor audit confirms the rank:

```text
answer_state_loop_mythos_lora_down.weight  (8, 512)
answer_state_loop_mythos_lora_up.weight    (512, 8)
answer_state_loop_mythos_lora_scale.weight (16, 8)
```

A later local causal forced-choice smoke on two rows reproduced the narrow
necessity signal:

```text
donor_only_no_evidence:      0/2
qtrm_core_off_no_evidence:   0/2
qtrm_core_steps_1_no_evidence: 0/2
qtrm_core_steps_4_no_evidence: 2/2
qtrm_core_steps_8_no_evidence: 2/2
```

But this is only a forced-choice depth-sweep diagnostic. With donor-logit
scale-1.0 blending, depth 1/4/8 all return the intermediate doubled list and
score `0/2`. Generation remains governed by the historical rejection above.

The earlier remembered `8/20` number is this same checkpoint's
`causal_forced_choice_smoke4.jsonl` aggregate, not a free-generation result:

```text
donor_only:              0/4
core_off:                0/4
core8 full:              2/4
decoder_off:             4/4
halt_gate_off:           2/4
aggregate:               8/20
generation_smoke8 full:  0/8
```

The 2026-05-30 explicit free-generation depth sweep confirms the same boundary:

```text
free generation depth sweep:
  donor/core-off/depth1/depth2/depth4/depth8 = 0/8 each
  aggregate = 0/48

matched causal forced-choice depth sweep:
  donor/core-off = 0/8
  depth1 = 2/8
  depth2 = 4/8
  depth4 = 3/8
  depth8 = 3/8
```

So the checkpoint can improve candidate discrimination under recurrence, but it
is still not generation-ready.
