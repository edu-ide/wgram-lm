# Donor-Preserving Controller Next Method

Date: 2026-05-06

## Decision

After web search over 2025-2026 reasoning-control prior, the next QTRM
architecture candidate should be:

```text
Donor-preserving QTRM logit guider
```

not another private-QTRM renderer.

## Why

The rejected Ouro renderer probes show:

```text
private QTRM hidden/head can score forced-choice narrowly
private QTRM hidden/head cannot generate correct text
small head/bridge patches either keep generation at 0 or damage forced-choice
```

The strongest matching prior now points to preserving the base model path:

```text
ThinkLogit / Proxy-Tuning:
  guide donor logits with another model's logit delta

ReFT / steering / GLoRE / BREP:
  intervene in frozen model representations with bounded, small edits

BuPO:
  optimize internal policies rather than only final output tokens

Dead Weights:
  frozen models can communicate through learned projections and residual hooks
```

## Proposed Architecture

First candidate:

```text
prompt tokens
-> Qwen donor hidden/logits
-> QTRM workspace + recursive core reads donor hidden states
-> QTRM predicts bounded logit delta and intervention gate
-> final logits = donor_logits + alpha * gate * clamp(delta)
-> autoregressive generation
```

This uses Qwen as the language renderer and QTRM as a reasoning controller.

## Training Objective

Start with a small supervised/preference objective:

```text
CE on final answer tokens through final_logits
donor-preservation KL on donor-correct rows
delta L2 / clamp / alpha schedule
gate sparsity so QTRM intervenes only when useful
core-off contrast so improvement must depend on recursive core
```

Do not train a private QTRM `lm_head` for this gate.

## Minimal Eval Gate

Run on the same mixed-composition heldout rows:

```text
modes:
  donor_only
  qtrm_donor_guided
  qtrm_delta_off
  qtrm_gate_off
  qtrm_core_off

metrics:
  generation hit
  forced-choice hit
  donor-correct preservation
  repetition
  entropy / donor-KL
```

Accept only if:

```text
qtrm_donor_guided generation > donor_only
delta_off returns donor behavior
core_off loses the gain
donor-correct rows are preserved
no repetition/entropy collapse
```

## Implementation Status

Implemented on 2026-05-06:

```text
model forward:
  disable_qtrm_residual keeps donor logits as the base policy
  disable_qtrm_residual_gate bypasses the learned QTRM intervention gate

raw-intelligence eval:
  qtrm_core_steps_8_delta_off_no_evidence
  qtrm_core_steps_8_residual_gate_off_no_evidence
  both are now included in the default no-retrieval depth sweep

config:
  configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml
```

Canonical first run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
PYTHONPATH=src \
bash scripts/08_train_donor_adapter.sh \
  configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml
```

Post-train gate:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
PYTHONPATH=src \
.venv/bin/python scripts/192_eval_raw_intelligence.py \
  --config configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml \
  --checkpoint runs/qwen35_2b_4090_donor_preserving_logit_guider_s120/last.pt \
  --scoring causal_forced_choice \
  --out runs/eval/donor_preserving_logit_guider_s120_causal_fc.jsonl
```

## If This Fails

Move to the second candidate:

```text
QTRM residual-stream intervention into Qwen donor layers
```

That should follow ReFT/GLoRE/steering/Dead-Weights style:

```text
donor_hidden[layer, token] += alpha * gate * P(qtrm_core_state)
```

This requires hook-based donor forward instrumentation and layer/token sweeps,
so it should not be attempted before the simpler donor-logit guider falsifier.

## Follow-Up Results

2026-05-06 gate results:

```text
v1 gated generic:
  reject, all full/core-off/delta-off modes stayed at donor 9/24

v2 no-gate generic:
  reject, no raw-reasoning gain

v3 pure-recursive preference:
  partial signal
  donor_only 9/24, delta_off 9/24, core_steps_1 14/24
  reject as causal-core proof because core_off also rose to 11/24

v4 core-forced readout:
  accept as causal architecture improvement
  donor_only 9/24, core_off 9/24, delta_off 9/24
  core_steps_1 14/24, core_steps_4 13/24, core_steps_8 10/24

v5 outer4 continuation:
  reject
  core_steps_4 fell to 11/24 and core_steps_8 fell to 8/24

v6 adaptive halt teacher-depth probe:
  reject as a depth selector
  halt head trained successfully but selected step 8 on the eval path
  donor_only 9/24, core_off 9/24, delta_off 9/24
  core_steps_1 14/24, core_steps_4 13/24, core_steps_8 10/24
  core_halt_steps_8 10/24 with core_steps_mean 8.0
```

Updated decision:

```text
Keep the donor-preserving bounded-delta path.
Make the QTRM delta core-forced by construction.
Do not solve depth instability by fixed longer loops.
Do not solve depth selection with final-depth stability targets alone.
Next falsifier: supervised depth router / verifier over the core-forced readout,
using fixed-depth outcome labels and preserving donor/core_off/delta_off
ablations.
```

First router-label artifact:

```text
script:
  scripts/depth_router_labels.py

input:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_halt_teacher_depth_s080_causal_fc_24_v2.jsonl

output:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_depth_router_labels_24.jsonl

summary:
  donor route: 9
  core_steps_1 route: 5
  core_steps_4 route: 1
  core_steps_8 route: 1
  unknown route: 8
  fixed-depth oracle: 16/24
```

## 2026-05-07 Router-Head Follow-Up Results

The supervised route labels were useful as a falsifier, but simple route heads
did not recover the fixed-depth oracle.

```text
oracle from fixed-depth sweep:
  16/24

linear softmax head, prompt mismatch:
  best 12/24
  reject

linear softmax head, prompt-exact:
  best 11/24
  reject

MLP head over final core pooled state, outer_steps=1:
  best 11/24
  reject

MLP head over final core pooled state, outer_steps=8:
  best 12/24
  core_steps_* targets: 0/7
  reject

trajectory MLP over core_depth_states[1..8]:
  best 12/24
  core_steps_1: up to 4/5
  core_steps_4/core_steps_8: 0/2
  donor route collapses at later checkpoints
  reject as a solved depth selector
```

Artifacts:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_depth_router_trajectory_mlp_s480.yaml

scan:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_depth_router_trajectory_mlp_s480_checkpoint_scan.jsonl

best predictions:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_depth_router_trajectory_mlp_s480_best_predictions_24.jsonl
```

Decision:

```text
Do not keep scaling route-head-only experiments on the 24-case label set.
The fixed-depth oracle proves useful depth choices exist, but frozen core
states do not expose a reliable route signal to a small classifier.
```

Updated next architecture candidate:

```text
LoopFormer-style variable trajectory training:
  train the recurrent core on long and short loop schedules
  add time/step-size conditioning or reuse existing step conditioning
  align short-route states to long-route states with shortcut consistency
  evaluate depth-1/2/4/8 answer accuracy and core-off/delta-off ablations
```

This replaces post-hoc routing with trajectory-level causal pressure. The
route head remains a diagnostic probe, not the canonical architecture.

## 2026-05-07 Shortcut-Consistency Follow-Up

Tested the first LoopFormer-inspired trajectory loss on the accepted
core-forced readout:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120.yaml

init:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_shortcut_outer4_s120_causal_fc_24.jsonl
```

Result:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:           10/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off at step 8:     9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject shortcut-consistency v1 as a canonical improvement.
It preserves the known step-1 gain but does not improve it, lowers step-4
from the accepted 13/24 to 12/24, and collapses step-8 to donor/core_off.
```

Interpretation:

```text
The simple same-run early-state-to-final-state consistency loss is not the
full LoopFormer variable-trajectory recipe. It regularizes the existing
trajectory but does not teach a reliable short/long recurrent policy.
```

Next candidate:

```text
true variable-trajectory training:
  sample short and long loop schedules during training
  run the same prompt through both schedules
  detach the long schedule as the trajectory teacher
  train short schedule logits/states to match only when the long schedule is
  correct or verifier-preferred
  keep donor/core_off/delta_off/depth sweep ablations
```

## 2026-05-07 Variable-Trajectory V1 Result

Implemented a true two-pass short/long trajectory loss:

```text
long path:
  normal forward with outer_steps=4

short path:
  temporary second forward with outer_steps=1

loss:
  short final core state aligns to detached long final core state
  short path keeps an LM loss
  optional long-over-short preference margin
```

Artifacts:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080.yaml

init:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_variable_traj_s080_causal_fc_24.jsonl
```

Training signal moved in the intended direction:

```text
core_variable_trajectory_state_cosine:
  about 0.22 -> 0.43

core_variable_trajectory_logp_margin:
  about -0.03 -> +0.27
```

But held-out forced-choice regressed:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           13/24
core_steps_2:            9/24
core_steps_4:           11/24
core_steps_8:            8/24
delta_off step8:         9/24
residual_gate_off step8: 8/24
```

Decision:

```text
Reject variable-trajectory v1.
The objective optimized its internal alignment metrics but did not improve raw
reasoning. It is another example where hidden-state/logp proxy progress does
not transfer to the 24-case causal raw-intelligence gate.
```

Updated root-architecture doubt:

```text
The accepted gain is concentrated at depth 1 and mostly boolean logic.
Trying to make deeper fixed loops imitate or outperform shallow loops has now
failed twice. The next candidate should not be another depth regularizer.
It should change the recurrent core's information update rule or training
target so each loop step performs a verifiable state transition.
```

## 2026-05-07 Depth-Text Process CE Result

Implemented per-depth process credit over `core_depth_text_logits`:

```text
main forward:
  return_core_depth_text_logits=True

loss:
  CE over every requested recurrent depth text-logit slice

memory fix:
  auxiliary return flags are stripped from preference/counterfactual/ablation
  forwards so rejected/core_off paths do not materialize huge depth logits
```

Artifacts:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_depth_text_ce_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_depth_text_ce_s080/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_depth_text_ce_s080_causal_fc_24.jsonl
```

Training signal:

```text
core_depth_text_ce:
  about 11.44 -> 11.13
```

Held-out forced-choice:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:            9/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off step8:         9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject depth-text process CE as canonical.
It preserves the shallow depth-1 signal but does not beat the accepted
core-forced baseline and does not create deeper-loop gains.
```

Branch-level kill criterion:

```text
Depth router heads, same-run shortcut consistency, true variable trajectory,
and per-depth text CE all failed to improve the accepted baseline.
Stop spending local effort on depth selection/alignment for this branch.
Next architecture work must make the loop perform explicit verifiable state
updates, not merely expose more depth logits.
```
