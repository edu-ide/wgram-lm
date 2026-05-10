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
