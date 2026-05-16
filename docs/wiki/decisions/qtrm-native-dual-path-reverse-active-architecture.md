# QTRM-Native Dual-Path Reverse Active Architecture

Status: active hypothesis, not proof of final canonical superiority.

Date: 2026-05-14

## Decision

Freeze QTRM-native architecture shopping around one active core until the
length/depth gate says otherwise:

```text
prompt/chat-template tokens
-> tokenizer
-> native token embeddings
-> native encoder
-> mandatory dual-path reverse TRM core
   -> z_L trajectory/state-flow path
   -> z_H correction/high-level update path
   -> reverse-primed z_L <-> z_H interaction
-> native decoder/readout
-> LM head
-> autoregressive text
```

Implementation label:

```text
trm_dual_z_reversed_hybrid_3to1
```

Operational alias:

```text
dual_path_reverse
```

Baseline:

```text
official_trm_think
```

## Why This Is Fixed Now

The previous failure pattern was architecture shopping: official TRM, coupled
TRM, Mamba3, GatedDeltaNet, diffusion-style variants, and route/order wrappers
were compared before the active hypothesis had a stable length/depth ladder.

The fixed rule is:

```text
Do not switch the main QTRM-native core again until dual_path_reverse fails a
specific gate and the failure axis is named.
```

Allowed fixes are training efficiency, supervision, curriculum, stabilization,
and ablation repair for this same architecture. A new mixer/backbone is only a
new candidate after this gate explains why the current path fails.

## Gate

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/352_run_qtrm_native_dual_path_reverse_gate.py \
  --out-dir local_eval/qtrm_native_dual_path_reverse_gate \
  --profile short \
  --lengths 4,6,8 \
  --candidates official,dual_path_reverse
```

Research runner:

```bash
bash scripts/300_run_research_gate.sh
```

Equivalent explicit form:

```bash
PYTHONPATH=src .venv/bin/python scripts/300_research_gate_runner.py \
  --gate qtrm_native_dual_path_reverse_length_gate \
  --profile standard
```

The gate writes:

```text
summary.json  # raw length/candidate rows from the shared length runner
report.json   # accepted/rejected dual_path_reverse decision wrapper
```

## Promotion Criteria

Promote only if all requested active rows pass:

```text
full_generation_exact > 0
full_minus_think0 > 0
full_minus_worst_ablation > 0
min_family_generation_exact > 0
dual_path_reverse >= official_trm_think at the same length
```

This is still a small-gate claim. It does not prove broad language ability.

## Donor Boundary

QTRM-integrated donor training may use LoRA or partial donor unfreezing, but it
is not QTRM-native. The donor bridge can reuse the active core idea only after
the native dual-path reverse core has a clean length/depth result.

Naming:

```text
donor disabled + dual_path_reverse core -> QTRM-native active core
trainable Qwen donor + mandatory core   -> QTRM-integrated donor bridge
frozen Qwen donor + sidecar QTRM         -> donor-backed adapter
```

## Next Work

1. Run the smoke gate to verify wiring.
2. Run the short/standard len4, len6, len8 gate.
3. If rejected, classify the failure as exactness, depth gain, ablation margin,
   family coverage, or official-baseline loss.
4. Repair only that axis in the same `dual_path_reverse` architecture.
5. After acceptance, run native language non-regression before donor-integrated
   healing.

## Wiring Smoke 2026-05-14

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/352_run_qtrm_native_dual_path_reverse_gate.py \
  --out-dir local_eval/qtrm_native_dual_path_reverse_gate_smoke_20260514 \
  --profile smoke \
  --lengths 4 \
  --candidates official,dual_path_reverse \
  --device cpu \
  --steps 2 \
  --train-cases 16 \
  --eval-cases 8 \
  --batch-size 4 \
  --d-model 16 \
  --d-ff 32 \
  --n-heads 4 \
  --n-kv-heads 2 \
  --log-every 0
```

Result:

```text
report: local_eval/qtrm_native_dual_path_reverse_gate_smoke_20260514/report.json
decision: rejected_dual_path_reverse_length_gate
reason: 2-step smoke produced zero exact/depth/ablation metrics
meaning: wiring verified, not a performance promotion
```

## Runner Result 2026-05-14T09:55:22

```text
gate: qtrm_native_dual_path_reverse_length_gate
target_level: L5R fixed dual-path reverse length gate
profile: standard
decision: command_failed
accepted: False
next_action: do not resume architecture shopping; diagnose the failed dual-path reverse length/depth/ablation axis and repair it
```

Decisive metrics:

```json
{}
```

Report: `local_eval/research_gate_runner/qtrm_native_dual_path_reverse_length_gate_standard/report.json`

## Runner Result 2026-05-14T10:50:03

```text
gate: qtrm_native_dual_path_reverse_length_gate
target_level: L5R fixed dual-path reverse length gate
profile: standard
decision: rejected_dual_path_reverse_length_gate
accepted: False
next_action: do not resume architecture shopping; diagnose the failed dual-path reverse length/depth/ablation axis and repair it
```

Decisive metrics:

```json
{}
```

Report: `local_eval/research_gate_runner/qtrm_native_dual_path_reverse_length_gate_standard/report.json`

## Repair Result 2026-05-14

The standard fixed gate rejected at len4 because the active architecture beat
the official baseline on average exactness but failed causal ablation:

```text
report:
  local_eval/research_gate_runner/qtrm_native_dual_path_reverse_length_gate_standard/report.json

full_generation_exact:                 0.052083333333333336
full_minus_think0:                     0.026041666666666668
full_minus_worst_ablation:            -0.005208333333333329
min_active_minus_official:             0.015625
reject axis:                           full_minus_worst_ablation<=0
```

Two narrow repair probes showed the axis more precisely:

```text
op-counterfactual mixed active lengths:
  local_eval/qtrm_native_dual_path_reverse_opcf_repair_len4_20260514_105510
  full_minus_worst_ablation:  0.026041666666666668
  target_active_len4_exact:   0.0
  verdict: op-zero repaired, but true len4 coverage failed

op-counterfactual active_len4 focus:
  local_eval/qtrm_native_dual_path_reverse_opcf_repair_len4_focus_20260514_111510
  full_minus_worst_ablation:  0.0
  target_active_len4_exact:   0.041666666666666664
  verdict: len4 coverage repaired, but z_L ablation matched full
```

Accepted len4 repair:

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len4_20260514_115012

fixed architecture:
  trm_dual_z_reversed_hybrid_3to1

recipe:
  H steps: 3
  L cycles: 6
  d_model: 128
  d_ff: 256
  lr_schedule: linear_warmup_cosine
  depth_intermediate_loss_weight: 0.20
  answer_space_ranking_loss_weight: 0.02
  active_len_replay_loss_weight: 0.02

metrics:
  full_generation_exact:        0.059895833333333336
  full_minus_think0:            0.0390625
  full_minus_worst_ablation:    0.015625
  min_family_generation_exact:  0.03125
  target_active_len4_exact:     0.03968253968253968
  z_h_zero_generation_exact:    0.0
```

Decision:

```text
Keep as the current len4 causal repair baseline. The fix did not change the
architecture; it changed only schedule/capacity/loss curriculum inside
dual_path_reverse. Next promotion step is len6 with the same H=3/L=6 recipe.
```

## Len6 Promotion 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len6_20260514_125601

fixed architecture:
  trm_dual_z_reversed_hybrid_3to1

recipe:
  H steps: 3
  L cycles: 6
  d_model: 128
  d_ff: 256
  lr_schedule: linear_warmup_cosine
  active lengths: 3..6
  depth_intermediate_loss_weight: 0.20
  answer_space_ranking_loss_weight: 0.02
  active_len_replay_loss_weight: 0.02

metrics:
  full_generation_exact:        0.044270833333333336
  full_minus_think0:            0.015625000000000003
  full_minus_worst_ablation:    0.0078125
  min_family_generation_exact:  0.0234375
  target_active_len6_exact:     0.052083333333333336
  z_h_zero_generation_exact:    0.0
```

Decision:

```text
Keep as len6 promotion signal. This is lower than len4 in average exactness but
still passes the strict causal gate: deeper core helps, destructive ablations
hurt, every family has nonzero exactness, and true active_len6 is nonzero.
Next promotion step is len8 with the same H=3/L=6 recipe.
```

## Len8 Attempt 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len8_20260514_140218

fixed architecture:
  trm_dual_z_reversed_hybrid_3to1

recipe:
  H steps: 3
  L cycles: 6
  d_model: 128
  d_ff: 256
  lr_schedule: linear_warmup_cosine
  active lengths: 3..8
  depth_intermediate_loss_weight: 0.20
  answer_space_ranking_loss_weight: 0.02
  active_len_replay_loss_weight: 0.02

metrics:
  full_generation_exact:        0.03125
  full_minus_think0:            0.005208333333333332
  full_minus_worst_ablation:    0.002604166666666668
  min_family_generation_exact:  0.0
  target_active_len8_exact:     0.015873015873015872

by_family:
  checksum: 0.0390625
  modchain: 0.0
  revchain: 0.0546875
```

Decision:

```text
Reject len8 only on the family floor axis. Depth gain, ablation gap, and true
active_len8 are nonzero, but modchain collapsed to zero. The next repair must
stay inside dual_path_reverse and adjust family replay / family-balanced loss,
not switch architecture.
```

## Len8 Family Replay Repair 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len8_modchain_replay_20260514_151625

recipe change:
  keep trm_dual_z_reversed_hybrid_3to1
  keep H=3 / L=6
  increase modchain replay
  active_len_replay_loss_weight: 0.03
  operation_counterfactual_loss_weight: 0.04

metrics:
  full_generation_exact:        0.033854166666666664
  full_minus_think0:            0.005208333333333332
  full_minus_worst_ablation:   -0.0078125
  min_family_generation_exact:  0.015625
  target_active_len8_exact:     0.031746031746031744

by_family:
  checksum: 0.015625
  modchain: 0.0390625
  revchain: 0.046875

worst ablation:
  z_l_zero_generation_exact: 0.041666666666666664
```

Decision:

```text
Reject on ablation causality only. The family-floor repair worked, including
modchain, and true active_len8 became nonzero. The remaining failure is z_L:
zeroing z_L improves exactness over full. Next repair should keep the same
architecture and add a weak z_L trajectory/codec pressure rather than more
family replay.
```

## Len8 Replay Plus z_L Codec Attempt 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len8_replay_zlcodec_20260514_163318

recipe change:
  keep trm_dual_z_reversed_hybrid_3to1
  keep H=3 / L=6
  keep modchain replay
  reduce operation_counterfactual_loss_weight: 0.02
  add core_step_codec_loss_weight: 0.02
  core_step_codec_state_source: l
  add state_trace_anti_collapse_loss_weight: 0.005

metrics:
  full_generation_exact:        0.036458333333333336
  think0_generation_exact:      0.036458333333333336
  full_minus_think0:            0.0
  full_minus_worst_ablation:   -0.005208333333333329
  min_family_generation_exact:  0.0234375
  target_active_len8_exact:     0.047619047619047616

by_family:
  checksum: 0.0234375
  modchain: 0.0390625
  revchain: 0.046875

worst ablation:
  z_l_zero_generation_exact: 0.041666666666666664
```

Decision:

```text
Reject on depth/ablation causality. This repaired the family floor and improved
true active_len8, but it collapsed the depth gain and still allowed z_L-zero to
beat the full model. Do not keep the z_L codec/anti-collapse recipe as a
canonical fix. The next repair must preserve the len8 family signal while
restoring full > think0 and full > z_L_zero.
```

## Len8 Base-Resume Family-DRO Repair 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_stage_recipe_len8_base_resume_familydro_20260514_175305

recipe change:
  resume from:
    local_eval/qtrm_native_dual_path_reverse_stage_recipe_len8_20260514_140218/len8_dual_path_reverse/last.pt
  keep trm_dual_z_reversed_hybrid_3to1
  keep H=3 / L=6
  add family_dro_loss_weight: 0.04
  family_dro_temperature: 1.0
  train only 600 continuation steps at lr 2e-5

metrics:
  full_generation_exact:        0.0390625
  think0_generation_exact:      0.026041666666666668
  full_minus_think0:            0.013020833333333332
  full_minus_worst_ablation:    0.0
  min_family_generation_exact:  0.03125
  target_active_len8_exact:     0.031746031746031744

by_family:
  checksum: 0.0390625
  modchain: 0.03125
  revchain: 0.046875

ablations:
  state_reset_generation_exact: 0.0390625
  op_zero_generation_exact:    0.036458333333333336
  z_l_zero_generation_exact:   0.03125
  z_h_zero_generation_exact:   0.0
```

Decision:

```text
Reject on a single strict-causality tie: state_reset == full. This is the
closest len8 result so far. The family floor is repaired, true active_len8 is
nonzero, deeper thinking beats think0, and z_L-zero no longer beats full.
Next repair should continue from this checkpoint and add only weak
state_reset counterfactual pressure. Do not add another architecture.
```

## Slow-Loop Reset 2026-05-14

```text
problem:
  Full len8 continuation runs are too slow for local one-axis repairs because
  each run performs training plus full/think0/state_reset/op_zero/z_L/z_H
  generation eval.

method update:
  Add PROFILE=triage to the length-gate wrapper.

triage role:
  Use 5-15 minute runs to decide keep/discard/probe only.
  Do not promote from triage. Promote only after a standard run confirms the
  same metric direction.

current triage target:
  Resume from the best len8 family-DRO checkpoint and test only whether weak
  state_reset counterfactual pressure can break the state_reset == full tie
  without losing:
    full > think0
    min_family > 0
    target_active_len8 > 0
    full >= z_L_zero/op_zero
```

## Len8 State-Reset Counterfactual Triage 2026-05-14

```text
run:
  local_eval/qtrm_native_dual_path_reverse_len8_state_reset_cf_triage_20260514_185042

recipe:
  resume from best len8 family-DRO checkpoint
  PROFILE=triage
  steps: 120
  eval_cases: 96
  state_reset_counterfactual_loss_weight: 0.03
  state_reset_counterfactual_margin: 0.2

metrics:
  full_generation_exact:        0.041666666666666664
  full_minus_think0:            0.03125
  full_minus_worst_ablation:   -0.010416666666666671
  min_family_generation_exact:  0.0
  target_active_len8_exact:     0.06666666666666667

by_family:
  checksum: 0.03125
  modchain: 0.09375
  revchain: 0.0

ablations:
  state_reset_generation_exact: 0.052083333333333336
  z_l_zero_generation_exact:   0.052083333333333336
```

Decision:

```text
Discard. The direct state-reset counterfactual made depth gain larger, but it
damaged the family floor and let state_reset/z_L-zero beat the full model.
Next triage should preserve the base checkpoint with retention KL and use only
weaker state-reset pressure.
```

## Len8 Retention/Depth-CF Triage Results 2026-05-14

```text
runs:
  local_eval/qtrm_native_dual_path_reverse_len8_state_reset_cf_retention_triage_20260514_190239
  local_eval/qtrm_native_dual_path_reverse_len8_depth1_cf_triage_20260514_191426

recipes:
  resume from best len8 family-DRO checkpoint
  PROFILE=triage
  steps: 120
  eval_cases: 96
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.08

state_reset_cf_retention:
  state_reset_counterfactual_loss_weight: 0.008
  state_reset_counterfactual_margin: 0.1

depth1_cf:
  depth_counterfactual_loss_weight: 0.012
  depth_counterfactual_think_steps: 1
  depth_counterfactual_margin: 0.1

shared metrics:
  full_generation_exact:        0.041666666666666664
  full_minus_think0:            0.03125
  full_minus_worst_ablation:   -0.010416666666666671
  min_family_generation_exact:  0.0
  target_active_len8_exact:     0.06666666666666667

by_family:
  checksum: 0.03125
  modchain: 0.09375
  revchain: 0.0

ablations:
  state_reset_generation_exact: 0.052083333333333336
  z_l_zero_generation_exact:   0.052083333333333336
```

Decision:

```text
Discard both. Weak retention, weak state-reset counterfactual, and weak
think3-vs-think1 counterfactual all converge to the same failure: state_reset
and z_L-zero beat full, and revchain collapses to zero in triage. The local
loss-repair budget for this axis is spent.

Next hypothesis:
  The fixed 3-step output is overthinking or not using persistent z_L state.
  The next repair should not add more loss weights. It should test a loop/halting
  structural change within dual_path_reverse: either choose the best causal
  depth with an orthodox halt/readout gate, or redesign the recurrent state
  carry so state_reset is no longer equivalent to the useful shallow path.
```

## Attractor Loop Paper Note 2026-05-14

```text
reference:
  arXiv 2605.12466v1, "Solve the Loop: Attractor Models for Language and Reasoning"
  repo: https://github.com/jacobfa/Attractor

relevance:
  This is not a replacement for dual_path_reverse during the current gate.
  It is a next-candidate prior for the same loop-stability bottleneck:
  fixed recurrence depth, train/test loop-count mismatch, BPTT memory growth,
  and unstable longer-loop refinement.

QTRM mapping:
  current:
    prompt tokens -> native encoder -> fixed H=3/L=6 dual_path_reverse core
    -> decoder -> LM logits

  possible future candidate:
    prompt tokens -> native encoder/backbone proposal
    -> attractor-style recurrent refinement to an approximate fixed point
    -> decoder/tied LM logits

kill condition:
  If an attractor-style candidate improves surface exactness but core-off,
  solver-off, or state-zero ablations do not drop the same metric, it remains
  a diagnostic probe and cannot replace dual_path_reverse.
```

## Runner Result 2026-05-14T10:54:45

```text
gate: qtrm_native_dual_path_reverse_length_gate
target_level: L5R fixed dual-path reverse length gate
profile: standard
decision: rejected_dual_path_reverse_length_gate
accepted: False
next_action: do not resume architecture shopping; diagnose the failed dual-path reverse length/depth/ablation axis and repair it
```

Decisive metrics:

```json
{
  "decisive_metrics.active_rows": 1,
  "decisive_metrics.min_active_full_generation_exact": 0.052083333333333336,
  "decisive_metrics.min_active_full_minus_think0": 0.026041666666666668,
  "decisive_metrics.min_active_full_minus_worst_ablation": -0.005208333333333329,
  "decisive_metrics.min_active_minus_official": 0.015625
}
```

Report: `local_eval/research_gate_runner/qtrm_native_dual_path_reverse_length_gate_standard/report.json`
