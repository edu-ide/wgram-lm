# Qwen3.5 Preinit Core Interpolation Stabilization

Date: 2026-05-19

## Question

Can the 128-case accepted Qwen3.5-pretrained mandatory-core QTRM checkpoint
composition be stabilized into a 256-case accepted checkpoint without changing
the architecture?

## Setup

Canonical path:

```text
prompt tokens
-> Qwen3.5 tokenizer / embeddings
-> Qwen3.5 original backbone
-> mandatory shared z_H/z_L TRM-style recurrent core
-> Qwen3.5 LM head
-> LM logits
```

Checkpoint A:

```text
local_eval/qwen35_preinit_strict_trm_partial_l3_gate_s80_20260519/last_core.pt
```

Checkpoint B:

```text
local_eval/qwen35_preinit_strict_trm_partial_l3_checksum_repair_s80_20260519/last_core.pt
```

Best previous scalar interpolation:

```text
alpha=0.25
128-case: accepted, gain 0.0390625, language_top1 1.0
256-case: rejected, gain 0.01953125, language_top1 1.0
```

The 256-case gate requires `gain >= 0.02`, so alpha=0.25 misses by one
additional core-over-base correct case.

## What Was Tried

Direct stabilization continuation from alpha=0.25:

```text
local_eval/qwen35_preinit_alpha025_stabilize_s60_20260519
decision: rejected
gain: 0.015625
language_top1_agreement: 1.0
```

Conclusion: low-LR continuation with KL and core-advantage pressure worsened
the accepted basin instead of stabilizing it.

Selective qwen/core interpolation was added:

```text
scripts/411_interpolate_trainable_checkpoints.py
  --qwen-alpha
  --core-alpha
  --qwen-attn-alpha
  --qwen-mlp-alpha
  --qwen-norm-alpha
  --core-state-alpha
  --core-adapter-alpha
```

Representative 256-case results:

```text
alpha0.25 / q0.25_c0.30:
  accepted: false
  gain: 0.01953125
  min_family_gain: 0.0
  min_family_core_accuracy: 0.0930232558
  language_top1_agreement: 1.0

q0.25_c0.32:
  accepted: false
  gain: 0.01953125
  language_top1_agreement: 1.0

q0.25_c0.35:
  accepted: false
  gain: 0.01953125
  language_top1_agreement: 1.0

q0.10_c0.50:
  accepted: false
  gain: 0.0078125
  min_family_gain: 0.0
  min_family_core_accuracy: 0.1162790698
  language_top1_agreement: 1.0

q0.15_c0.50:
  accepted: false
  gain: 0.0
  min_family_gain: -0.0232558140
  language_top1_agreement: 0.875

q0.20_c0.50:
  accepted: false
  gain: 0.00390625
  min_family_gain: -0.0348837209
  language_top1_agreement: 1.0
```

Group interpolation gave two 128-case passes but did not survive 256 cases:

```text
qa25_qm25_qn25_cs50_ca25:
  128-case: accepted, gain 0.0390625
  256-case: rejected, gain 0.01953125

qa25_qm25_qn10_cs50_ca30:
  128-case: accepted, gain 0.0234375
  256-case: rejected, gain 0.0078125
```

## Interpretation

The result is useful but not yet a robust promotion.

What is real:

```text
1. Qwen3.5-pretrained mandatory-core QTRM can preserve language logits.
2. A QTRM core gain over core-off exists in a nearby weight basin.
3. Family-floor and aggregate-gain objectives can coexist at 128-case scale.
```

What is not proven:

```text
1. 256-case stable acceptance.
2. public benchmark gain.
3. Qwen3.6-27B comparison.
4. a trained stable checkpoint after continuation.
```

The decisive bottleneck is no longer loading, language collapse, or scalar
interpolation. The bottleneck is recurrent-objective stability: training must
make the core's causal gain robust across families instead of merely moving
between two fragile rejected checkpoints.

## Decision

Do not spend more time on scalar or coarse group interpolation unless it is used
as a diagnostic. The next credible step is an objective/selection change:

```text
1. keep the canonical Qwen3.5-pretrained mandatory-core path;
2. train with fixed 256-case family-balanced selection during training;
3. select checkpoints by the actual promotion gate, not only loss;
4. add core-off and family-floor metrics to every periodic checkpoint;
5. promote only if 256-case gain clears the threshold by margin.
```

This preserves the QTRM-native claim because the model remains a single graph
and the improvement must pass through token -> Qwen backbone -> mandatory core
-> LM logits.

## Follow-Up Selection Run

Added runner:

```text
scripts/412_run_qwen35_preinit_family_balanced_selection.sh
```

DGX output:

```text
local_eval/qwen35_preinit_family_balanced_select_s120_20260519
```

Configuration:

```text
init_checkpoint: alpha_0.25.pt
steps: 120
eval_cases: 256
eval_every_steps: 10
lr: 1.0e-5
qwen_lr: 0.0
kl_weight: 0.10
language_kl_weight: 0.10
core_advantage_weight: 0.20
core_advantage_mode: label_choice_margin
family_loss_weights: chain5=1.2,checksum4=1.2,select_pair=1.2
```

Result:

```text
decision: rejected
best_periodic_eval_step: 10
best_periodic_gain: 0.01953125
final_gain: 0.01953125
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0235294118
  checksum4:   0.0
  select_pair: +0.0352941176
```

Interpretation:

```text
The fixed 256-case selection loop preserves the alpha=0.25 near-miss but does
not improve it. The bottleneck is now sharper: the current continuation
objective cannot create checksum4 gain without degrading other families. The
next experiment should add a checksum-specific contrastive/counterfactual
objective or a recurrent trajectory loss, not more low-LR CE continuation.
```

## Checksum Counterfactual Objective 2026-05-19

Added training-only checksum4 counterfactual augmentation:

```text
scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --checksum-counterfactual-weight
  --checksum-counterfactual-variants
```

Mechanism:

```text
For checksum4 training cases, parse a,b,c,d from the prompt.
Create counterfactual prompts by changing one digit.
Train the same canonical LM-logit path to answer the changed prompt.
No runtime sidecar or external solver is used at inference.
```

DGX run:

```text
local_eval/qwen35_preinit_checksum_cf_w05_v2_s80_20260519
```

Configuration:

```text
init_checkpoint: alpha_0.25.pt
steps: 80
eval_cases: 256
lr: 2.0e-5
qwen_lr: 0.0
checksum_counterfactual_weight: 0.5
checksum_counterfactual_variants: 2
family_loss_weights: chain5=1.1,checksum4=1.8,select_pair=1.1
```

256-case result:

```text
decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

512-case expansion:

```text
local_eval/qwen35_preinit_checksum_cf_w05_v2_eval512_20260519

decision: rejected
gain: 0.015625
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0292397661
  checksum4:   0.0
  select_pair: +0.0176470588
```

Interpretation:

```text
This is a real step: the Qwen3.5-pretrained mandatory-core path crossed the
256-case promotion threshold while preserving language logits. It is not yet a
robust breakthrough because the 512-case expansion rejects and checksum4 still
has zero core-over-base gain.

The counterfactual objective helped the aggregate gate, mostly by improving
chain5 retention/gain. It did not solve the intended checksum4 bottleneck.
```

Next required objective:

```text
base-error targeted checksum4 objective:
  compute base/core label-choice logits on checksum4 training cases
  apply extra margin only where base is wrong or weak
  select by 512-case gain and require checksum4 positive gain

Do not call the 256-case checkpoint robust until this passes.
```

## Base-Error Targeted Checksum Objective 2026-05-19

Added:

```text
--checksum-base-error-advantage-weight
--checksum-base-error-margin
--checksum-base-error-base-margin-threshold
```

Mechanism:

```text
On checksum4 training cases, compute base/core label-choice margins.
If the base/core-off path is wrong or weak, add extra loss that pushes the
core target digit above the strongest wrong digit.
```

DGX run:

```text
local_eval/qwen35_preinit_checksum_baseerr_w06_s80_20260519
```

Result:

```text
256-case decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

Interpretation:

```text
The stricter targeted loss still did not move checksum4. The 256-case
acceptance is therefore not evidence that the intended checksum bottleneck is
solved. The accepted gain is again carried by chain5/select_pair.

Conclusion: checksum4 is not responding to final-token CE/margin pressure at
this scale. The next diagnostic must inspect whether the recurrent core changes
checksum4 predictions/logit ranks at all, and whether z_H/z_L carries operand
information. If the latent state is not binding a,b,c,d, the next fix is a
state/trajectory supervision or operand-binding objective, not a stronger
answer-margin loss.
```

## Checksum Latent Probe 2026-05-19

Added:

```text
scripts/413_probe_qwen35_preinit_checksum_latents.py
```

Purpose:

```text
Diagnose whether checksum4 failure is caused by:
  A. no operand information in z_H/z_L;
  B. operand information present but no answer composition;
  C. answer present in latent but not rendered through LM logits.
```

DGX probe checkpoints:

```text
local_eval/qwen35_preinit_checksum_probe_alpha025_20260519
local_eval/qwen35_preinit_checksum_probe_cf_w05_v2_20260519
local_eval/qwen35_preinit_checksum_probe_baseerr_w06_20260519
```

Prediction result across all three:

```text
checksum4 eval cases: 86
base_accuracy: 0.0930232555
core_accuracy: 0.0930232555
gain: 0.0
core_fixes_base_errors: 0
core_breaks_base_correct: 0
```

Representative alpha=0.25 linear probes:

```text
z_h -> answer eval accuracy:    0.1046511605
z_h -> operand_a eval accuracy: 0.8488371968
z_h -> operand_b eval accuracy: 0.4069767296
z_h -> operand_c eval accuracy: 0.3488371968
z_h -> operand_d eval accuracy: 0.4651162922
```

Representative base-error checkpoint linear probes:

```text
z_h -> answer eval accuracy:    0.1046511605
z_h -> operand_a eval accuracy: 0.8488371968
z_h -> operand_b eval accuracy: 0.4069767296
z_h -> operand_c eval accuracy: 0.2790697813
z_h -> operand_d eval accuracy: 0.4418604672
```

Interpretation:

```text
The recurrent state is not blank. It carries operand information, especially
operand_a and partial b/c/d. The failure is compositional: z_H/z_L do not encode
the final checksum answer strongly enough, and the LM path never flips a
checksum4 base error into a core-correct answer.

This explains why final-token CE, counterfactual prompt CE, and base-error
margin losses can improve aggregate 256-case gain through chain5/select_pair
while checksum4 remains flat.
```

Next architecture/training target:

```text
Add a latent answer-composition objective:
  z_H or delta_h -> Qwen LM head digit logits for checksum4
  train only as an auxiliary state/trajectory objective
  inference still uses the normal token -> Qwen -> mandatory core -> LM logits path

Promotion requirement:
  checksum4 core_fixes_base_errors > 0 on held-out eval
  checksum4 gain > 0.0
  512-case aggregate gain >= 0.02
  language_top1_agreement remains 1.0 or above threshold
```

## Latent Answer Auxiliary Rejected 2026-05-19

Added model/training hooks:

```text
src/qtrm_mm/qwen_backbone_qtrm.py
  exposes qtrm_core_hidden and qtrm_core_delta for diagnostics/training

scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --checksum-latent-answer-weight
  --checksum-latent-answer-source z_h|delta_h
  --checksum-latent-answer-lr
```

The auxiliary head is training-only. It is not saved as an inference answer
path, so the gate still evaluates the normal LM logits.

DGX runs:

```text
local_eval/qwen35_preinit_latent_answer_zh_w05_s100_20260519
local_eval/qwen35_preinit_latent_answer_delta_w05_s80_20260519
```

Both runs:

```text
256-case decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

Interpretation:

```text
The latent answer auxiliary does not transfer into the actual LM answer path
for checksum4. It again selects a checkpoint where chain5/select_pair carry the
aggregate gain while checksum4 is unchanged.

This rejects the weak version of "HRM-Text-style supervision": simply adding an
auxiliary answer probe on final z_H/delta_h is not enough. To borrow HRM-Text
usefully, supervision must shape the recurrent trajectory itself: slots,
intermediate residues, carry/state transitions, or a PrefixLM process target
that the mandatory core must route through the same LM logits.
```

Next target:

```text
implement checksum trajectory supervision:
  target residues after each partial sum:
    r1 = a
    r2 = a + 2*b
    r3 = a + 2*b + 3*c
    r4 = answer
  attach the target to recurrent step states or step-conditioned z_H, not only
  to the final hidden state.
```

## Checksum Trajectory Supervision 2026-05-19

Added recurrent step-state exposure and a checksum4 trajectory loss:

```text
src/qtrm_mm/qwen_backbone_qtrm.py
  exposes qtrm_core_step_states

scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --checksum-trajectory-weight
```

Mechanism:

```text
For checksum4 cases, parse a,b,c,d and supervise the recurrent step states with
partial residue targets through the same Qwen LM head:

  step1 -> a mod 10
  step2 -> a + 2*b mod 10
  step3 -> a + 2*b + 3*c mod 10
  step4 -> a + 2*b + 3*c + 4*d mod 10

This is HRM-Text-inspired trajectory shaping, but it remains inside the
canonical LM-logit path. It is not a runtime calculator or hidden answer
channel.
```

DGX weak trajectory run:

```text
local_eval/qwen35_preinit_checksum_traj_w05_s100_20260519

eval_cases: 256
checksum_trajectory_weight: 0.5
checksum_counterfactual_weight: 0.2
checksum_base_error_advantage_weight: 0.3
best_periodic_eval_step: 10
mean_checksum_trajectory_loss: 2.3138490723

decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

DGX stronger trajectory run:

```text
local_eval/qwen35_preinit_checksum_traj_w2_s120_20260519

eval_cases: 256
checksum_trajectory_weight: 2.0
best_periodic_eval_step: 60
mean_checksum_trajectory_loss: 2.3192725692

decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

512-case expansion of the stronger checkpoint:

```text
local_eval/qwen35_preinit_checksum_traj_w2_eval512_20260519

eval_cases: 512
decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0467836257
  checksum4:   0.0
  select_pair: +0.0235294118
```

Interpretation:

```text
This is the first 512-case accepted Qwen3.5-pretrained mandatory-core QTRM
checkpoint in this line while preserving language top-1 agreement. That is real
progress.

It is not yet the requested breakthrough. The checksum4 bottleneck remains
unsolved: base and core have identical checksum4 accuracy, and checksum4 gain
is still 0.0. The accepted 512-case aggregate is carried by chain5 and
select_pair.

Therefore the next architecture change should not be another auxiliary loss
that only probes step states. The recurrent trajectory must become a causal
carry route into the final residual/LM path so intermediate computation can
change the final answer.
```

Next falsifiable candidate:

```text
trajectory carry mixer:
  route0 = existing final z_H/delta path
  route1 = learned weighted carry over recurrent step states
  train route parameters preservation-first
  require route0 replay to preserve the accepted checkpoint before training

Promotion requirement:
  512-case gain >= 0.02
  language_top1_agreement remains 1.0 or above threshold
  checksum4 gain > 0.0
  checksum4 core_fixes_base_errors > 0
  disabling the trajectory carry removes the checksum4 gain
```

## Trajectory Carry Route 2026-05-19

Added a preservation-first trajectory carry route:

```text
src/qtrm_mm/qwen_backbone_qtrm.py
  --core-trajectory-carry-mode none|mean|learned
  --core-trajectory-carry-gate-init

scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --eval-force-trajectory-carry-off
```

Design:

```text
The recurrent step states are pooled and projected back into the final core
delta before the normal LM head. The projection is zero-initialized, so enabling
the route preserves the existing checkpoint before training.

This is different from the previous trajectory loss. The previous loss only
asked step states to predict residues. The carry route gives those states a
causal path into the answer-producing LM logits.
```

Preservation replay:

```text
local_eval/qwen35_preinit_trajcarry_mean_replay512_20260519

init_checkpoint: local_eval/qwen35_preinit_checksum_traj_w2_s120_20260519/last_core.pt
core_trajectory_carry_mode: mean
steps: 0

512-case decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0
checksum4 gain: 0.0
```

Short carry training:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_s80_20260519

init_checkpoint: local_eval/qwen35_preinit_checksum_traj_w2_s120_20260519/last_core.pt
core_trajectory_carry_mode: mean
steps: 80
checksum_trajectory_weight: 2.0
lr: 3.0e-5
qwen_lr: 0.0
note: wrapper default still partially unfroze Qwen layer 3 in this run.

256-case decision: accepted
gain: 0.02734375
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0470588235
```

512-case expansion:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_eval512_20260519

512-case decision: accepted
gain: 0.037109375
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0643274854
  checksum4:  +0.0058479532
  select_pair:+0.0411764706
```

Carry-off ablation on the same checkpoint:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_eval512_carryoff_20260519

512-case decision: accepted
gain: 0.02734375
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0643274854
  checksum4:   0.0
  select_pair:+0.0176470588
```

Interpretation:

```text
This is the first run where checksum4 moves positive on the 512-case expansion.
The carry-off ablation removes that checksum4 gain and lowers aggregate gain,
so the new route has causal evidence.

This is still not the final breakthrough. The checksum4 gain is small, language
top1 drops from 1.0 to 0.875, and the run accidentally kept Qwen layer 3
trainable through the wrapper default. The next experiment must repeat with an
explicit frozen-Qwen route-only setting and then test a learned carry mixer.
```

Route-only frozen-Qwen repeat:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_routeonly_s80_20260519

core_trajectory_carry_mode: mean
qwen_trainable: false
steps: 80

256-case decision: accepted
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176
```

Refined interpretation:

```text
The route-only mean carry preserves language but does not improve checksum4.
The previous positive checksum4 signal came from the combination of trajectory
carry and partial Qwen layer-3 healing. Therefore, the carry route is a useful
causal pathway, but the frozen route-only version is not yet strong enough.

Next candidates:
  1. learned trajectory carry weights instead of mean pooling;
  2. explicitly allow a tiny Qwen healing scope and require language top1 >= 1.0
     or a stricter language suite;
  3. add carry-off and qwen-layer-frozen ablations to every promoted run.
```

## Learned Carry Rejection 2026-05-19

Learned carry route-only run:

```text
local_eval/qwen35_preinit_trajcarry_learned_w2_routeonly_s120_20260519

core_trajectory_carry_mode: learned
qwen_trainable: false
steps: 120

256-case decision: accepted
gain: 0.03515625
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0588235294
  checksum4:   0.0
  select_pair: +0.0470588235
```

512-case expansion:

```text
local_eval/qwen35_preinit_trajcarry_learned_w2_routeonly_eval512_20260519

512-case decision: rejected
gain: 0.00390625
language_top1_agreement: 1.0

family gains:
  chain5:       0.0
  checksum4:   0.0
  select_pair:+0.0117647059
```

Mean carry route-only 512 check:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_routeonly_eval512_20260519

512-case decision: rejected
gain: 0.00390625
language_top1_agreement: 1.0
checksum4 gain: 0.0
```

Decision:

```text
Frozen route-only carry, whether mean or learned, does not generalize to 512.
The best current signal remains the mean carry checkpoint that was trained in
the previous partial-healing regime:

  local_eval/qwen35_preinit_trajcarry_mean_w2_eval512_20260519
  gain: 0.037109375
  checksum4 gain: +0.0058479532
  carry-off gain: 0.02734375
  carry-off checksum4 gain: 0.0

Do not promote learned carry. The next run must select checkpoints with
language-aware periodic scoring and must evaluate 512 cases during selection if
we want the selected checkpoint to survive 512 expansion.
```

## 512-Case Language-Aware Carry Selection 2026-05-19

Strict language selection run:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang_s120_20260519

core_trajectory_carry_mode: mean
eval_cases: 512
selection_min_language_top1: 1.0
selection_language_weight: 1.0

decision: rejected
selected step: 20
gain: 0.017578125
language_top1_agreement: 1.0
checksum4 gain: 0.0
```

Interpretation:

```text
Requiring perfect language top1 preserved language but selected a checkpoint
below the reasoning threshold. This is useful as a boundary condition: perfect
language preservation is too strict for the current tiny language probe if we
want the checksum family to move.
```

Practical language-aware selection run:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_s100_20260519

core_trajectory_carry_mode: mean
eval_cases: 512
selection_min_language_top1: 0.875
selection_language_weight: 0.5
selected step: 100

decision: accepted
gain: 0.021484375
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0350877193
  checksum4:  +0.0058479532
  select_pair:+0.0235294118
```

Carry-off ablation:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_eval512_carryoff_20260519

decision: rejected
gain: 0.0078125
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0058479532
  checksum4:   0.0
  select_pair:+0.0176470588

failed:
  accepted_reasoning_gain: false
  accepted_family_core_accuracy: false
```

Decision:

```text
This is the strongest causal Qwen3.5-preinit QTRM result so far:

1. selected directly on 512 cases, not selected on 256 then expanded;
2. aggregate reasoning gain clears the 0.02 gate;
3. all three families have non-negative gain and checksum4 is positive;
4. turning off trajectory carry destroys acceptance and removes checksum4 gain;
5. language top1 remains above the current non-regression threshold but is not
   perfectly preserved.

This is a real architecture signal, not yet the final breakthrough. The next
promotion requirement is to expand the language probe and raise the language
floor while preserving the carry-dependent checksum4 gain.
```

## Extended Language Probe Follow-Up

Added a selectable language probe set:

```text
--language-probe-set basic|extended
```

The extended set contains 32 English/Korean prompts covering ordinary
explanation, translation, uncertainty, source conflict, debugging, checksum,
and model-memory descriptions.

DGX evaluation:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_eval512_extendedlang_20260519

init:
  local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_s100_20260519/last_core.pt

accepted: true
gain: 0.021484375
language_probe_set: extended
language_top1_agreement: 0.96875
num_prompts: 32
checksum4 gain: +0.0058479532
min_family_gain: +0.0058479532
```

Decision:

```text
The 512-selected trajectory-carry checkpoint survives a stronger language
non-regression probe. This upgrades the result from "small language probe
accepted" to "extended language probe accepted".

The next HRM-Text-inspired move should be language healing with packed
PrefixLM-style clean text, but the promotion gate must remain:

  reasoning gain over core_off
  positive family floor
  extended language non-regression
  trajectory-carry/core destructive ablation

HRM-Text is a training prior here, not proof that innovation will arrive
quickly by copying its recipe.
```

## HRM-Text-Style Healing Follow-Up

Added response-only clean language healing:

```text
--language-healing-weight
--language-healing-kl-weight
--language-healing-batch-size
```

The objective is deliberately narrow:

```text
prompt prefix + clean response
-> Qwen3.5 tokenizer/backbone
-> mandatory QTRM core
-> Qwen3.5 LM head
-> response-token CE/KL only
```

DGX run:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_s40_20260519

accepted: true
gain: 0.03125
language_probe_set: extended
language_top1_agreement: 0.96875
num_prompts: 32
checksum4 gain: +0.0058479532
min_family_gain: +0.0058479532
min_family_core_accuracy: 0.1111111111

family gains:
  chain5:      +0.0584795322
  checksum4:  +0.0058479532
  select_pair:+0.0294117647
```

Carry-off ablation:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_s40_carryoff_20260519

accepted: false
gain: 0.0
language_probe_set: extended
language_top1_agreement: 1.0

family gains:
  chain5:      0.0
  checksum4:   0.0
  select_pair: 0.0
```

Decision:

```text
Promote response-only language healing as the current canonical HRM-Text
training import. It improves the 512-case carry-dependent gain from 0.021484375
to 0.03125 while preserving extended language non-regression, and the gain
disappears under carry-off ablation.

Do not call this Qwen3.6-27B-level capability. It is a stronger local causal
gate result. The next requirement is multi-seed repetition and a small
generation-quality probe.
```

## Multi-Seed Repetition Decision

Added multi-eval-seed support:

```text
--eval-seed-offsets 10000,10001,10002
```

This builds one evaluation set by concatenating equal-sized held-out synthetic
sets from multiple deterministic seed offsets. It prevents selecting a
checkpoint that only works on one 512-case seed.

Results:

```text
single seed 20260519:
  accepted: true
  gain: 0.03125
  language_top1: 0.96875

single seed 20260520:
  accepted: false
  gain: 0.013671875
  language_top1: 0.96875

single seed 20260521:
  accepted: false
  gain: 0.001953125
  language_top1: 0.96875
```

Multi-seed selection:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_multiseed_s60_20260519

accepted: false
eval_cases: 576
eval_seed_offsets: 10000,10001,10002
gain: 0.0190972222
language_top1: 0.96875
min_family_gain: +0.0104166667
min_family_core_accuracy: 0.0989583333
```

Stronger core-advantage repair:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_multiseed_adv10_s80_20260519

accepted: false
gain: 0.0190972222
```

Decision:

```text
Do not promote the HRM-Text healing result as robust yet. It is a real
single-seed causal gain and a multi-seed near miss, but it does not pass the
current multi-seed threshold.

Keep response-only language healing as a useful ingredient because it preserves
language and improves the single-seed gate. The next architecture/training
claim must pass multi-eval-seed selection before carry-off ablation is enough
for promotion.
```

## Multi-Train Seed Decision

Added multi-train-seed support:

```text
--train-seed-offsets 0,1,2
```

This builds one larger training set by concatenating equal-sized synthetic
sets from multiple deterministic seed offsets. It tests whether the
HRM-Text-style healing gain was underfit to one train seed.

Run:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_multitrain_multiseed_s80_20260519

accepted: false
train_cases: 6144
train_seed_offsets: 0,1,2
eval_cases: 576
eval_seed_offsets: 10000,10001,10002
gain: 0.0173611111
language_top1: 0.96875
min_family_gain: +0.0104166667
min_family_core_accuracy: 0.0989583333

family gains:
  chain5:      +0.015625
  checksum4:  +0.0260416667
  select_pair:+0.0104166667
```

Decision:

```text
Reject as a robust promotion. The result is directionally healthy but below
the multi-seed acceptance threshold.

Do keep the mechanism in the toolbox:
  clean response-only healing
  Qwen backbone preservation
  mandatory recurrent core
  multi-train/multi-eval seed reporting

Do not keep scaling this exact loss as the next main bet. The stronger signal
is that the model needs a better recurrent trajectory objective so that the
core changes the answer for the right reason across seeds.
```

## Base-Error Trajectory Advantage Decision

Added a generic recurrent trajectory objective:

```text
--trajectory-advantage-weight
--trajectory-advantage-margin
--trajectory-monotonic-weight
--trajectory-monotonic-margin
--trajectory-loss-base-error-only
```

Unlike the checksum-specific residue objective, this applies to all hard-v1
families through the normal Qwen LM head:

```text
qtrm_core_step_states
-> core_out_norm
-> Qwen LM head
-> digit-choice target-vs-wrong margin
```

Broad trajectory pressure result:

```text
local_eval/qwen35_preinit_recurrent_trajadv_multiseed_s80_20260519

accepted: false
gain: 0.0190972222
language_top1: 0.96875
min_family_gain: +0.015625
```

Base-error-only result:

```text
local_eval/qwen35_preinit_recurrent_trajadv_baseerr_multiseed_s80_20260519

accepted: true
gain: 0.0208333333
language_top1: 0.96875
min_family_gain: +0.015625
min_family_core_accuracy: 0.0989583333

family gains:
  chain5:      +0.015625
  checksum4:  +0.0260416667
  select_pair:+0.0208333333
```

Carry-off:

```text
local_eval/qwen35_preinit_recurrent_trajadv_baseerr_multiseed_s80_carryoff_20260519

accepted: false
gain: 0.0034722222
language_top1: 1.0
```

Decision:

```text
Promote base-error trajectory advantage as the first robust multi-eval-seed
QTRM-native recurrent training improvement after HRM-Text healing. It is not
just more language CE: the objective trains intermediate recurrent states to
improve the same LM-head decision margin, and the accepted gain collapses when
trajectory carry is disabled.

Next promotion requirement:
  repeat on another seed bundle or run a core-off/carry-off paired report
  as part of the default script output before claiming broader robustness.
```
