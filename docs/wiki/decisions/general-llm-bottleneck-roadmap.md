# General LLM Bottleneck Roadmap

Date: 2026-05-07

Status: active SSoT roadmap.

This document answers what remains after the current reverse-composition gate
is accepted. A reverse accept is important, but it is only one raw-intelligence
milestone. It proves prompt-conditioned latent operation order, not a general
LLM.

Operating discipline:

- accept levels and failure budgets are defined in
  [Research Major-Gate Discipline](research-major-gate-discipline.md);
- only `L3` or `L4` results count as major bottleneck progress;
- small accepted probes, scaffolds, and local gates must remain separated from
  the `0/10` major-bottleneck count.

Current major-bottleneck count:

```text
accepted: 2 narrow milestones / 10 broad bottlenecks
active: bottleneck 4 -> non-copy latent-state-to-text synthesis
```

2026-05-11 orthodox status:

```text
roadmap target:
  L4/general-LM promotion

active gate:
  L2/L3 prerequisite repair for non-copy latent-state-to-autoregressive text
  synthesis.

why this is not L4 yet:
  strict generation still rejects on the mixed non-copy scalar target even
  after core-state-only answer-loop constraints, ablation-contrast loss,
  greedy-margin training, self-rollout loss, and beam search diagnostics.

current blocker:
  the recurrent core can perturb the output surface, but the LM path still
  collapses to repeated/partial digits and does not reliably emit the final
  scalar tail plus EOS.

orthodox next condition:
  full generation must beat donor-only/core-off/core-state-zero/path-off on the
  same held-out runner. Teacher-forced CE or forced-choice movement is useful
  evidence, but not promotion.
```

The accepted milestones are deliberately narrow:

- source-position recurrent state reached strict L3 on hard list-transform
  perturbations;
- source-copy lexicalization reached L4 standard128 through the canonical LM
  generation path.

They do not yet prove broad/general LLM promotion. The current blocking result
is that the same source-copy L4 checkpoint scores `0/48` on mixed-family
non-copy generation across donor-only, core-off, and full QTRM modes.
The follow-up causal forced-choice diagnostic also scores `0/12`; every mode
ranks the intermediate doubled list above the final scalar answer. This makes
the next blocker explicit: scalar reduction/accumulator/final-answer state,
not another source-copy renderer patch.

An older preserved Ouro recurrent-answer L2 checkpoint was also rechecked on
the same harder len11/13 mixed non-copy split:

```text
checkpoint:
  local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0_s40_eval4/accepted.pt

forced-choice max_cases=8:
  donor_only:        0/8
  core_off:          0/8
  recurrent_off:     0/8
  full recurrent:    0/8

full tail classes:
  doubled_list:      6/8
  pre_subtract_sum:  2/8
```

This demotes the earlier len7 recurrent-answer smoke to a narrow L2 local
result. It does not scale to longer non-copy reductions. The active blocker is
therefore length-stable scalar reduction and final subtract retention inside
the recurrent answer state.

The direct orthodox retry also failed:

```text
run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_len1113_s020_eval4_from_jointonly

init:
  accepted len11/13 transition joint controller

action/finality after training:
  trace exact:    32/32
  finality exact: 32/32
  halted exact:   32/32

forced-choice eval4:
  donor_only:     0/4
  core_off:       0/4
  recurrent_off:  0/4
  full recurrent: 0/4

full tail class:
  doubled_list:   4/4
```

So the next architecture candidate should not be "more answer CE" on the same
answer loop. The core must expose a causal value accumulator state, or the
answer loop must receive process-supervised value-state updates rather than
only final token CE.

## Typed Value State To Answer Loop Contract

Status: active L2/L3 repair candidate, not L4 promotion.

Prior principle:

- recurrent latent reasoning and state-to-answer residual adaptation;
- process-supervised intermediate state is allowed only if it flows back into
  the canonical LM logits path;
- closest local prior is the existing transition-state and role-value
  answer-bridge mechanisms, but the scalar value state must remain a learned
  latent distribution, not a rule solver.

QTRM tensor path:

```text
prompt/chat-template tokens
-> token embeddings / frozen donor hidden states
-> recurrent QTRM core depth states
-> typed_algorithmic scalar/final residual logits
-> typed value-state answer tokens
-> answer_state_loop cross-attention hidden state
-> LM head logits
-> autoregressive text
```

Causal ablation:

- `disable_typed_algorithmic_value_state_answer_bridge=True` must reduce the
  same final-answer metric that the full model improves;
- `disable_core=True` and `disable_answer_state_loop_recurrent=True` must also
  remove the advantage;
- donor-only/core-off must not match the full path on held-out mixed non-copy
  cases.

Shortcut risk:

- the bridge must not render text or compute numeric answers outside the model;
- it may expose value-state probabilities as latent tokens, but the answer
  still has to be selected by the answer loop and LM head;
- if the bridge wins only because a typed scalar class is copied directly into
  a task-specific renderer, the result remains a diagnostic probe.

Kill criterion:

- reject if full forced-choice/generation ties donor-only, core-off,
  recurrent-off, or typed-bridge-off;
- reject if the dominant failure remains `doubled_list` or `pre_subtract_sum`;
- reject if the run improves only on the training surface without held-out
  length/order perturbation.

Smoke result, 2026-05-10:

```text
implementation commit:
  db596b6 feat(qtrm): add typed value answer bridge

train smoke:
  config:
    configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_answer_bridge_s040.yaml
  init:
    local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_jointonly_s080_from_s720/last.pt
  out:
    local_eval/smoke_typed_value_answer_bridge_s1/last.pt
  result:
    one step completed; answer_state_loop_logit_ce and
    typed_algorithmic_value_state_ce both contributed to the loss.

eval smoke:
  out:
    local_eval/smoke_typed_value_answer_bridge_s1/eval_bridge_ablation.jsonl
  modes:
    qtrm_core_steps_4_no_evidence
    qtrm_core_steps_4_typed_value_answer_bridge_off_no_evidence
  result:
    0/2 hits; both modes still prefer the doubled-list candidate after one
    step. This is wiring validation only, not an accepted reasoning result.
```

Short gate result, 2026-05-10:

```text
run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_answer_bridge_s040_from_jointonly

checkpoint:
  last.pt

forced-choice max_cases=4:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        0/4
  full bridge/recurrent:   0/4

dominant full failure:
  doubled_list:            4/4

decision:
  rejected as an L2/L3 performance gate. The bridge is committed as a causal
  wiring scaffold, but it does not yet convert typed value state into the final
  scalar LM answer. The next candidate needs stronger process supervision for
  accumulator/final-subtract state or a different scalar-to-LM alignment loss,
  not merely more steps on the same shallow bridge.
```

Final LM choice-margin retry, 2026-05-10:

```text
implementation:
  add final_choice_sequence_margin_loss to the training script.

principle:
  apply preference pressure only on the canonical final LM logits path:
    chosen final answer > rejected row choices
  compatible with --final-path-only-supervision.
  no depth head, renderer, external solver, or candidate-time shortcut.

smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_typed_value_answer_bridge_final_choice_s1

short run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_answer_bridge_final_choice_s020_from_s040

training signal:
  final_choice_sequence_margin_final_path fell from 0.2256 to 0.0524 in the
  observed log window; final_path_acc rose from 0.1429 to 0.2857.

forced-choice max_cases=4:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        0/4
  full bridge/recurrent:   0/4

score-gap diagnostic:
  donor gold-minus-pred mean:          -1.5316
  recurrent_off gold-minus-pred mean:  -1.5696
  full gold-minus-pred mean:           -0.8886
  typed_bridge_off gold-minus-pred:    -0.8885

decision:
  rejected as an L2/L3 answer-change gate. The final LM choice-margin loss is
  a useful canonical-path pressure because it improves the scalar answer score
  gap, but the dominant output remains the doubled list and typed-bridge-off
  matches full. The causal effect is currently in the answer recurrent path,
  not in the typed value-state bridge, and it is not strong enough to flip the
  selected answer.
```

Typed bridge final-contrast retry, 2026-05-10:

```text
implementation:
  e9fc05e feat(training): add typed bridge final contrast

principle:
  run a no-grad ablated forward with
  disable_typed_algorithmic_value_state_answer_bridge=True, then apply a
  final LM-path target-logp contrast:
    full target logp > typed-bridge-off target logp + margin

smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_typed_value_bridge_final_contrast_s1
  metric observed:
    typed_value_answer_bridge_final_contrast=0.0463
    typed_value_answer_bridge_final_target_logp_delta=0.0037

short run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_bridge_final_contrast_s020_from_final_choice_s020

forced-choice max_cases=4:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        0/4
  full bridge/recurrent:   0/4

score-gap diagnostic:
  donor gold-minus-pred mean:          -1.5316
  recurrent_off gold-minus-pred mean:  -1.5538
  full gold-minus-pred mean:           -0.7651
  typed_bridge_off gold-minus-pred:    -0.7664

strong final-choice continuation:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_bridge_final_choice_strong_s020_from_contrast_s020

strong forced-choice max_cases=4:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        0/4
  full bridge/recurrent:   0/4

strong score-gap diagnostic:
  donor gold-minus-pred mean:          -1.5316
  recurrent_off gold-minus-pred mean:  -1.5657
  full gold-minus-pred mean:           -0.6753
  typed_bridge_off gold-minus-pred:    -0.6794

dominant full failures:
  pre_subtract_sum:        2/4
  doubled_list:            2/4

decision:
  rejected as a solved answer-change gate. The final LM path now clearly moves
  away from donor-only doubled-list behavior and the answer recurrent path is
  causal by score-gap ablation, but the typed value-state bridge still has only
  a tiny causal delta and turning it off nearly matches full. The remaining
  blocker is final subtract/value-generalization, not text rendering alone.
```

Final subtract-tail LM-margin retry, 2026-05-10:

```text
implementation:
  2698945 feat(training): add final subtract-tail margin

principle:
  apply a final LM-path-only margin against subtract-tail counterfactuals:
    chosen final scalar > pre-subtract sum and final +/- 1 variants
  This is compatible with --final-path-only-supervision and does not use depth
  logits, a renderer, or an external solver.

smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_final_subtract_tail_margin_s1
  metric observed:
    final_subtract_tail_counterfactual_margin_final_path=0.1619

short run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  final_subtract_tail_margin_s020_from_strong_s020

forced-choice max_cases=4:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        0/4
  full bridge/recurrent:   0/4

score-gap diagnostic:
  donor gold-minus-pred mean:          -1.5316
  recurrent_off gold-minus-pred mean:  -1.5363
  full gold-minus-pred mean:           -0.5628
  typed_bridge_off gold-minus-pred:    -0.5705

dominant full failures:
  pre_subtract_sum:        2/4
  doubled_list:            2/4

decision:
  rejected as a solved answer-change gate, but keep the result as a useful
  causal-path diagnostic. The recurrent answer path now moves the final LM
  choice distribution substantially away from donor/recurrent-off, while
  typed-bridge-off remains very close to full. The next candidate should target
  scalar value extrapolation or offset binding inside the latent state, because
  answer-path margins alone are improving the score gap without crossing the
  final scalar decision boundary.
```

Full token/core/answer-loop typed scalar codec, 2026-05-10:

```text
implementation:
  b9c4bc1 feat(training): add full token answer loop typed policy
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_s060.yaml

principle:
  previous candidates split the trainable path:
    core+answer_loop+typed, but token/prompt input path frozen
    token+core+typed, but answer loop frozen
  The new policy opens the canonical path together:
    prompt tokens -> token/prelude/workspace/core -> typed scalar codec
    -> answer recurrent loop -> LM logits
  The scalar codec remains auxiliary. It must not be treated as a hidden
  answer channel.

smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_typed_value_fullpath_scalar_codec_s1
  observed in one step:
    final_path_acc=0.7500
    final_choice_sequence_margin_final_path=0.0667
    final_subtract_tail_counterfactual_margin_final_path=0.0500
    typed_algorithmic_value_state_ce=28.6436
    typed_algorithmic_scalar_regression_mae=46.7143

short path:
  s020:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    typed_value_fullpath_scalar_codec_s020_from_subtract_tail
    full forced-choice max_cases=4: 0/4
    donor: 0/4, core_off: 0/4, recurrent_off: 0/4
    full gold-minus-pred mean: -0.4107

  s060:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    typed_value_fullpath_scalar_codec_s060_from_s020
    full forced-choice max_cases=4: 0/4
    donor: 0/4, core_off: 0/4, recurrent_off: 0/4
    full gold-minus-pred mean: -0.3750

  subtract-heavy continuation:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    typed_value_fullpath_subtract_heavy_s020_from_s060
    final-subtract margin weight=2.0, margin=0.20

forced-choice max_cases=4 after subtract-heavy:
  donor_only:              0/4
  core_off:                0/4
  recurrent_off:           0/4
  typed_bridge_off:        2/4
  full bridge/recurrent:   2/4

forced-choice max_cases=8 after subtract-heavy:
  donor_only:              0/8
  core_off:                0/8
  core_steps_1:            2/8
  core_steps_2:            4/8
  recurrent_off:           2/8
  full core_steps_4:       4/8

score-gap diagnostic after subtract-heavy:
  donor gold-minus-pred mean:          -1.5316
  recurrent_off gold-minus-pred mean:  -0.7360
  full gold-minus-pred mean:           -0.4290
  typed_bridge_off gold-minus-pred:    -0.4292

depth-sweep score-gap diagnostic max_cases=8:
  donor gold-minus-pred mean:          -1.4534
  core_steps_1 gold-minus-pred:        -0.3065
  core_steps_2 gold-minus-pred:        -0.2582
  recurrent_off gold-minus-pred:       -0.5035
  full core_steps_4 gold-minus-pred:   -0.2319

failed follow-up:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_fullpath_len1113_surface_s020_from_subtract_heavy
  trained on same length/surface family but held-out 40000-series values.
  max_cases=8 regressed:
    full core_steps_4:       4/8 -> 2/8
    core_steps_2:            4/8 -> 2/8
  This rejects the simple "just add len11/13 surface training" hypothesis for
  this loss mix.

shuffle follow-up:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_fullpath_subtract_heavy_shuffle_s020_from_s060
  retrained the same subtract-heavy continuation from s060 with shuffled rows.
  max_cases=8:
    donor_only:              0/8
    core_off:                0/8
    core_steps_1:            2/8
    core_steps_2:            4/8
    recurrent_off:           2/8
    full core_steps_4:       4/8
  score-gap diagnostic max_cases=8:
    donor gold-minus-pred mean:          -1.4534
    core_steps_1 gold-minus-pred:        -0.2564
    core_steps_2 gold-minus-pred:        -0.2062
    recurrent_off gold-minus-pred:       -0.4244
    full core_steps_4 gold-minus-pred:   -0.2028
  This preserves the narrow answer-change result and slightly improves score
  gaps, but it does not improve hit count over 4/8. Row order was not the
  decisive bottleneck.

checkpoint-selection follow-up:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_fullpath_subtract_heavy_s020_save_every5_from_s060
  reran the original subtract-heavy continuation with --save-every 5.
  qtrm_core_steps_4_no_evidence max_cases=8:
    step_000005: 0/8, gold-minus-pred mean -0.2924
    step_000010: 2/8, gold-minus-pred mean -0.2691
    step_000015: 4/8, gold-minus-pred mean -0.2457
    step_000020: 4/8, gold-minus-pred mean -0.2316
  This rejects the hypothesis that an earlier checkpoint clearly beats the
  final checkpoint. The narrow answer-change gate appears late and then
  plateaus at 4/8 for this loss mix.

length-coverage follow-up:
  4498ae8 data(qtrm): add mixed length coverage curriculum
  data/filtered/
  pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len5791113_train40000_v0to5_mixed_only.jsonl
  adds train lengths 5/7/9/11/13 with train surface variants 0-5.

  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_fullpath_len5791113_coverage_s040_from_subtract_heavy
  continued from the 4/8 subtract-heavy checkpoint with shuffled rows, lr=1e-5,
  and the same final LM-path subtract-heavy loss.
  qtrm_core_steps_4_no_evidence max_cases=8:
    baseline step_000020: 4/8, gold-minus-pred mean -0.2316
    length coverage s040: 2/8, gold-minus-pred mean -0.2443
  This rejects the simple "add 11/13 train lengths" hypothesis for this loss
  mix. The remaining blocker is more likely final subtract/offset binding in
  the causal latent-to-LM path, not length coverage alone.

typed value-state diagnostic:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_fullpath_subtract_heavy_s020_save_every5_from_s060/
  eval_typed_value_state_max8.json
  on the 4/8 baseline checkpoint, with typed scalar regression values:
    trace_exact_accuracy:       0/8
    step_exact_accuracy:        0/32
    field_accuracy:             32/200 = 0.1600
    content_field_accuracy:     32/168 = 0.1905
  Scalar diagnostics show offset/final-residual fields are wrong on held-out
  len11/13 cases. This means the current typed scalar bridge is not merely
  failing to route a correct latent value to LM logits; the typed value state
  itself is not a reliable held-out value representation.

  next design implication:
    do not promote the absolute typed scalar codec. Treat it as diagnostic.
    The next canonical candidate should bind prompt source values and offset
    variables from the token stream into recurrent state, then make that state
    causal for the normal LM answer path. Source/offset binding is the bottleneck
    to test before more final-answer margin tuning.

depth-8 diagnostic:
  same baseline checkpoint, max_cases=8:
    core_steps_4: 4/8, gold-minus-pred mean -0.2316
    core_steps_8: 0/8, gold-minus-pred mean -0.2451
  The current narrow gate is not a clean "deeper recurrence improves reasoning"
  result. It is a depth-4-local answer-change effect. This blocks promotion to
  a stronger recursive-reasoning claim until recurrent state remains stable, or
  halting reliably chooses the useful depth, under deeper core rollout.

prompt source-position binding probes:
  supporting fixes/data:
    0ac2e9c fix(qtrm): infer mixed source lists for binders
    9a452a4 data(qtrm): add multibase mixed curriculum
    b140d85 data(qtrm): add holdout surface mixed curriculum
    1e78132 feat(qtrm): add relative source parity binder probe

  single-base token source slots:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    prompt_source_position_mixed_len1113_token_slots_d1_s300
    train exact: 1.0000
    eval exact:  0.0000
    eval slot/content acc: 0.4375 / 0.2500
    Absolute numeric source-slot ids overfit train values and do not generalize
    from 40000-series train values to 60000-series eval values.

  single-base donor hidden:
    prompt_source_position_mixed_len1113_donor_hidden_d1_s300
    best eval exact: 0.0000
    final eval slot/content acc: 0.4844 / 0.3125

  multibase donor hidden:
    prompt_source_position_mixed_multibase_donor_hidden_d1_s500
    final eval exact: 0.0000
    final eval slot/content acc: 0.6094 / 0.4792

  multibase + surface-diverse donor hidden, variants 6/7 still held out:
    prompt_source_position_mixed_multibase_holdout67_donor_hidden_d1_s600
    final eval exact: 0.0000
    final eval slot/content acc: 0.6797 / 0.5729

  relative source-slot parity, same multibase/surface-diverse train split:
    prompt_source_position_mixed_relative_parity_holdout67_d1_s200
    input: source slot sequence with absolute value removed:
      0=pad, 1=odd source value, 2=even source value
    held-out base 60000 and surface variants 6/7:
      step 50 eval exact:   1.0000
      step 100 eval exact:  1.0000
      step 200 eval exact:  1.0000
    This accepts the L1 source-binding scaffold as soon as absolute numeric
    identity is replaced by relative source slots plus a learned/derived
    numeric predicate.

  conclusion:
    even before final subtract and LM rendering, prompt-to-source binding is not
    exact under held-out value/surface shift. More base/surface diversity helps
    slot accuracy, but does not close exact binding when the representation is
    absolute or donor-hidden only. Relative source slots plus parity/predicate
    state closes the source-selection part of the bottleneck. The next QTRM
    candidate should move this representation into the token-derived recurrent
    state and add offset variables, then test whether that state causally changes
    the normal LM answer path.

decision:
  partial acceptance for the narrow answer-change prerequisite:
  the recurrent answer/core path can now change causal forced-choice answers
  over donor-only, core-off, shallower depth, and recurrent-off on this mixed
  arithmetic gate. This is not L3/L4/general LLM acceptance because the
  strongest held-out score is only 4/8, typed-bridge-off tied full on the
  4-case check, and odd-start len11 cases still choose the pre-subtract sum.
  The scalar typed bridge is diagnostic, not yet causal.

next hypothesis:
  keep the canonical token/core/answer-loop policy, but stop relying on the
  typed scalar bridge as the decisive mechanism. The next gate should stabilize
  final-answer selection under subtract-tail pressure without catastrophic
  surface-continuation regression, then expand beyond 8 cases.
```

Core-state-only strict retry, 2026-05-11:

```text
implementation:
  add answer_state_loop_core_state_only_enabled
  add final LM-path contrast against:
    zero_core_trajectory=True
    disable_answer_state_loop_recurrent=True

principle:
  visible prompt/donor states may query core trajectory state, but the answer
  residual/value side must not append raw text context when making a
  core-causality claim. The same destructive ablations used by the gate can now
  be trained directly on target log-prob.

20-step CE recovery from:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  l4_sufficient_onecase_overfit/train_eos_s020/last.pt

out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_causal_prefix_s020_from_eos

observed training:
  final_path_ce: 6.9826 -> 4.8107 in the logged window
  final_path_acc: 0.1429

strict 1-case gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_s020_gate_1case

decision:
  rejected_noncopy_lm_gate
  target=600054, full=00000000, donor=10000, core_off=!!!!!!!!,
  core_state_zero=55555555

gate-contrast smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_core_state_only_gate_contrast_s001
  final_path_ce=2.8354, final_path_acc=0.4286
  core_state_zero_final_target_logp_delta=9.2229
  answer_state_recurrent_final_target_logp_delta=1.9025

greedy-margin retry:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_gate_contrast_greedy_s010
  final_path_acc remained 0.4286

strict 1-case gate after greedy-margin:
  rejected_noncopy_lm_gate
  full=00000000
  typed_value_answer_bridge_off=60060060
  core_state_zero=Vega5555555

decision:
  rejected as L4 capability. The strict core-state-only path is wired,
  trainable, and surface-sensitive to core_state_zero, but it still cannot
  synthesize the final scalar answer through greedy LM generation. The typed
  bridge remains diagnostic because disabling it produced a closer, still
  incorrect output in the latest gate. The active blocker remains bottleneck 4:
  latent-state-to-autoregressive text synthesis.
```

## Dual-Process Ordering

QTRM may eventually become a dual-process architecture:

```text
System 1 path: fast donor/latent prior, LeWM-style prediction, surprise signal
System 2 path: mandatory TRM/QTRM recurrent core, depth sweep, halt, verification
Output path: one canonical LM logits path
```

This is a roadmap, not the current canonical claim.

Near-term priority is System 2 first:

- prove that the mandatory recurrent core improves raw reasoning over
  donor-only/simple baselines;
- prove that deeper recurrence helps harder held-out cases and that halt
  prevents overthinking;
- prove that the improved latent state causally affects the normal LM answer
  path.

System 1 extensions such as LeWM-style world-model prediction, fast latent
priors, or surprise signals remain experimental until the System 2 path has a
stable causal gate. They may be trained as probes or auxiliary losses, but they
must not become a second answer model, hidden router, or side-channel output.

## Current Gate

Current active gate:

```text
prompt -> donor/prelude hidden states -> recurrent QTRM core
-> transition joint action/finality policy -> held-out composition trace
```

Acceptance means:

- `mixed_list_arithmetic`: old list-to-arithmetic order remains correct.
- `mixed_arithmetic_list`: reverse arithmetic-to-list order becomes correct.
- transition-off/code-shuffle ablations drop.
- no MemoryOS, retrieval, external solver, or hidden answer channel is involved.

If accepted, the model has crossed bottleneck 1 below.

## Remaining Major Bottlenecks

### 1. Prompt-Conditioned Latent Operation Order

Question:

```text
Can the recurrent core choose a different internal operation sequence when the
same operation family is described in a different order?
```

Current status:

- active reverse-composition gate.
- clean split audit found hidden surface overlap in the old reverse gate; old
  best `123/128` is demoted to historical diagnostic only.
- clean primitive operation head reaches `93/128`, with reverse `61/64` but
  old-order regression to `32/64`.
- clean canonical joint path remains stuck at `64/128`, reverse `0/64`.
- clean operation-residual and joint+operation-residual retries are rejected.
- best clean partial is phase+joint at `99/128`, but residual scale behaves
  like a brittle old/reverse switch rather than a stable composition policy.
- centered/gated/global-query phase variants and factorized action+halt are
  rejected.
- full-core reopen with factorized action+halt is rejected at `0/128`; it has
  too much blast radius.
- narrow core-context adapter with factorized action+halt is also rejected at
  `49/128`.
- next candidate: transition/hint feedback must enter the next recurrent step,
  not merely be decoded after the core. The minimal root change should test
  teacher-forced or predicted transition-state feedback into recurrence, with
  feedback-off ablation. Do not reopen all core weights as the next step.

Accept only if:

- held-out reverse exact reaches `64/64`;
- old order remains exact;
- transition-off/code-shuffle ablations drop strongly.

### 2. Recursive Depth Scaling And Halting

Question:

```text
Does more latent recursion improve harder cases without overthinking easier
ones?
```

Required gates:

- `core_steps=1/2/4/8/16` sweep;
- difficulty/length sweep;
- adaptive halt on/off;
- overthinking detection where too many steps degrade accuracy.

Accept only if deeper recurrence causally improves harder held-out cases and
halting prevents easy-case drift.

### 3. Latent State/Value Binding

Question:

```text
Can the model carry exact intermediate values, entities, and role bindings in
latent state, not only action codes?
```

Current problem:

- action traces can be correct while value traces remain wrong.
- typed/register/value-state probes have improved partial value accuracy but
  have not produced accepted exact value traces.

Accept only if:

- action code remains correct;
- intermediate value/state exactness improves on held-out lengths/ranges;
- state/value path off causes a real drop.

### 4. Latent-State To Autoregressive Text Renderer

Question:

```text
Can the learned latent reasoning state reliably become normal LM text?
```

Current problem:

- forced-choice/action-code success does not guarantee greedy generation.
- answer renderer/LM-head probes often preserve internal scores but fail
  multi-token generation.

Accept only if:

- normal greedy/autoregressive answers improve, not only forced-choice scores;
- renderer-off or core-off ablation drops;
- language fluency and donor-correct behavior are preserved.

### 5. Donor Preservation And Override Policy

Question:

```text
When should QTRM override the donor, and when should it stay silent?
```

Required gates:

- donor-only vs QTRM vs core-off;
- donor-correct preservation;
- donor-wrong correction;
- residual scale/override ablation.

Accept only if QTRM improves donor-wrong cases without damaging donor-correct
language behavior.

### 6. Trainable Memory Intelligence

Question:

```text
Can MSA/LM2-style trainable memory retain and retrieve useful information
better than prompt-only context or external MemoryOS?
```

Required gates:

- memory-on/off;
- length sweep;
- distractor sweep;
- delayed recall and update tests.
- MSA/LM2-style memory module ablation, not only external MemoryOS retrieval.

MemoryOS/RAG may be a runtime system, but it does not prove model memory. The
canonical claim requires a trainable memory path with causal ablations.

### 7. Reasoning And Memory Composition

Question:

```text
Can the model use retained facts and then compose them through recursive
reasoning?
```

This is stricter than retrieval. The model must not only find a fact; it must
use it in a multi-step latent computation.

Accept only if:

- memory-off loses recall/use;
- core-off loses composition;
- full model beats both on held-out distractor and multi-hop cases.

### 8. Metacognition, Uncertainty, And Contradiction Handling

Question:

```text
Does the model know when it is underdetermined, contradictory, temporally stale,
or out-of-distribution?
```

Required gates:

- known/unknown accuracy;
- confidence/ECE/Brier;
- fake evidence and contradiction cases;
- temporal/source conflict cases;
- answer/search/abstain routing.

Accept only if calibration improves because of the model's internal core or
memory state, not only because of post-hoc thresholds.

### 9. Scalable Context Routing

Question:

```text
Can the model focus on the relevant prompt/context tokens without a separate
side-channel architecture?
```

Required gates:

- token-attention context reader on/off;
- long prompt and distractor prompt sweeps;
- same-mean prompt counterfactuals;
- selective context routing without leaving the universal LLM causal path.

This supports long context, but it is not the same as raw reasoning or memory.

### 10. Agentic Closed-Loop And Multimodal Grounding

Question:

```text
Can the model plan, observe, revise, use tools, and ground multimodal inputs
through a causal learned controller?
```

Required gates:

- action choice improves task success;
- failed action records change future decisions;
- vision/audio/text grounding ablations drop;
- runtime tools remain outside the model boundary unless their effects are
  compiled into the canonical prompt/token stream or learned memory path.

## Promotion Rule

Do not promote a capability because a sidecar, retrieval system, typed
executor, or prompt rule solved the task.

Promote only if:

```text
full model > donor-only/simple baseline
full model > core-off/memory-off/path-off ablation
held-out exactness improves
normal LM generation is not damaged
failure ledger and regression gate are preserved
```

## Practical Priority Order

1. Finish bottleneck 1: reverse composition order.
2. Immediately test bottleneck 2: recurrence depth and halting on harder
   composition cases.
3. Re-open bottleneck 4: make the accepted latent trace affect normal text
   generation.
4. Re-open bottleneck 3: exact value/state binding.
5. Then move to bottlenecks 6 and 7: trainable memory and reasoning-memory
   composition.
6. Run metacognition gates in parallel only when raw reasoning does not regress.

MSA is therefore included, but it is not the current first blocker. It becomes
canonical only after the pure recursive reasoning path has a stable causal
gate; otherwise memory improvements can hide a weak core.

Latest bottleneck 1 update:

- predicted core-transition feedback was implemented and rejected:
  `64/128` clean reverse exact, with `64/64` on one family and `0/64` on the
  opposite family;
- teacher-forced core-transition feedback improved family balance but was still
  rejected: `81/128`, below the current `99/128` partial baseline;
- teacher-forced -> self-feedback finetune regressed to `61/128`;
- prompt-derived core order bottleneck was implemented and rejected:
  strict frozen-core variant scored `0/128`; core-open variant improved to
  `74/128` but remained below the `99/128` partial baseline and over-selected
  one operation-order sequence (`58/64` vs `16/64` by family);
- order-conditioned recurrent step conditioning was implemented and rejected:
  base run improved the weak family to `32/64` and total exact to `83/128`,
  but still remained below `99/128`; hard-family finetune raised step accuracy
  to `0.9092` but exact fell to `73/128`, showing curriculum weighting trades
  family balance rather than solving the shared algorithm;
- checkpoint storage was corrected to support frozen-donor delta checkpoints
  (`qtrm_trainable_delta_v1`) so future narrow adapter experiments do not fill
  the root filesystem;
- next candidate: factorized per-step transition state with independent
  operation, order, and finality targets. The order signal must be represented
  as a stable recurrent state variable, not only as a prompt classifier or
  curriculum bias.

Latest source-binding / L4 audit:

- source-position recurrent state is no longer the immediate blocker. After
  switching integrated QTRM training from single-row updates to batched paired
  hard-negative updates, the paired source-position gate accepted at L2, and
  the L3 hard split accepted with:
  `full_trace_exact_accuracy=0.8203125`,
  `full_value_accuracy=0.95849609375`,
  `primitive_value_drop=0.95849609375`,
  `source_binder_value_drop=0.55224609375`.
- source-copy lexicalization then passed the canonical LM generation path at
  standard128:
  `full=100/128`, `donor=44/128`, `core_off=44/128`,
  `primitive_off=37/128`, `source_slot_off=41/128`,
  `source_binder_off=41/128`, `vocab_renderer_off=44/128`.
- This is accepted only as source-copy L4. It proves that a narrow latent
  source/copy state can causally improve autoregressive generation.
- The next diagnostic failed: the same architecture scores `0/48` on
  mixed-family non-copy generation (`donor_only`, `core_off`, `full` all
  `0/16`). This shows the remaining bottleneck is not source-copy
  lexicalization; it is non-copy answer synthesis from latent state.
- A 4-case causal forced-choice diagnostic confirms the same failure in a
  sharper form: donor-only, core-off, and full QTRM all select the intermediate
  doubled list instead of the final scalar answer. This means the failure is
  upstream of greedy text generation.
- Therefore the shortest next path is a scalar reduction/accumulator gate that
  reuses the accepted source-binding discipline but requires `sum/subtract`
  final-answer state before LM generation. LeWM, MemoryOS, MSA, larger donors,
  and agent loops remain downstream extensions; they must not mask failure of
  the canonical latent-state-to-text path.

2026-05-10 relative source-slot integration update:

- Implemented prompt-derived relative parity source-slot ids in the shared
  QTRM train/eval path:
  `b39f72f feat(qtrm): add relative parity source slot mode`.
- Propagated the id mode through the L3/L4 gate runners:
  `afc8694 feat(qtrm): propagate source slot id mode through gates`.
- Preserved visible-prompt token/span copy while using relative ids:
  `48ea7b2 fix(qtrm): preserve source copy spans for relative slots`.
- Smoke run:
  `/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_relative_parity_smoke_s001`
  with `--token-numeric-source-slot-id-mode relative_parity`,
  `--token-numeric-source-slot-vocab-size 3`, `steps=1`, `max_eval_cases=4`.
- Result: executable but rejected. All measured modes scored `1/4 = 0.25`;
  `full_minus_donor=0.0`, `full_minus_core_off=0.0`,
  `full_minus_source_slot_off=0.0`, `full_minus_source_binder_off=0.0`.
- Important nuance: full generation did change one completion vs donor/core-off
  (`surface_paraphrase`: donor/core-off `2,`, full `56,`), so the canonical
  LM path can be perturbed by the QTRM path. This is not promotion because the
  change is not accuracy-positive and does not drop under primitive/source-slot
  or source-binder ablations.
- The 20-step follow-up smoke also rejected. Before tightening scoring it
  reported `full=1/8`, `donor=1/8`, `core_off=1/8`, but audit showed those
  hits were loose contains matches such as target `52` inside output `52,54`.
  `list_transform` and `sequential_list_transform` generation scoring now use
  strict exact/normalized-exact matching. Offline strict rescore of that JSONL
  is `0/8` for donor, core-off, full, and all listed ablations.
- The same audit found a training-policy issue for relative mode: changing the
  source-slot id space can freshly initialize `token_numeric_source_slot_*` and
  `core_source_position_binder_*` parameters, while the prior L4 policy trained
  mostly answer bridge / answer loop / vocab renderer parameters. Added
  `source_slot_binder_answer_bridge_loop_vocab_renderer` so the prompt-derived
  source-binding path and LM answer path can be trained together in the next
  run.
- Next bottleneck: train the relative source-slot path long enough, or more
  likely add a scalar reduction/accumulator state, then require a causal drop
  under `primitive_off`, `source_slot_off`, `source_binder_off`, and
  `answer_bridge_off`. Mere completion perturbation is not sufficient.

2026-05-10 direct L4 sufficient-condition update:

- The mixed non-copy gate has been tightened into a direct sufficient-condition
  gate instead of another loose diagnostic:
  strict exact generation only, no loose contains matches.
- Required baselines and ablations are now:
  donor-only, core-off, full QTRM, primitive-off, source-slot-off,
  source-binder-off, answer-bridge-off, typed-value-answer-bridge-off,
  vocab-renderer-off, and answer-recurrent-off.
- Acceptance requires full QTRM to beat donor-only and core-off, and to lose
  its advantage when each required causal path is disabled.
- The existing source-copy default checkpoint cannot currently be used for
  this direct gate because its trainable-delta base chain refers to a missing
  `primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt`.
- A self-contained typed-value checkpoint was used for a 1-case runner smoke:
  `/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  l4_sufficient_noncopy_gate_typed_value_s020_smoke_1case`.
  It executed successfully but rejected with strict exact `0/1` for donor,
  core-off, full, and all ablations.
- Conclusion: the direct sufficient-condition path is now available, but the
  architecture still lacks a reliable scalar/list accumulator-to-LM-logits
  path. This remains the active bottleneck for broad L4 promotion.

2026-05-11 orthodox core-state-only update:

- Implemented `answer_state_loop_core_state_only_enabled`.
- In this mode, visible text/donor states may query the core trajectory, but
  raw text states are not appended as answer-loop cross-attention values.
- Added a unit test that catches the old shortcut: the answer loop previously
  cross-attended over workspace/core state plus text state (`10` tokens in the
  smoke unit); the strict mode keeps the value/context side to core/workspace
  only (`4` tokens in the smoke unit).
- Added strict candidate configs:
  `qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_core_state_only_s060.yaml`
  and
  `qwen35_2b_4090_source_copy_pointer_renderer_core_state_only_scaffold.yaml`.
- Updated `scripts/330_run_mixed_noncopy_lm_gate.py` default config/checkpoint
  to use a self-contained typed-value core-state-only candidate instead of the
  previous source-copy delta checkpoint with a missing base chain.
- Ran a 1-case smoke:
  `/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/mixed_noncopy_typed_core_state_only_smoke_1case/report.json`.
- Result: rejected. Full, donor-only, core-off, core-state-zero, and
  answer-recurrent-off all scored `0/1`.
- Surface outputs changed under destructive ablations (`full=66666666`,
  `core_state_zero=55555555`, `answer_recurrent_off=00000000`), so the strict
  path is executable and perturbable, but it still does not synthesize the
  correct non-copy scalar answer `600054`.
- A 1-step causal-prefix training smoke on the same strict config completed and
  saved
  `/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/smoke_core_state_only_causal_prefix_s1/last.pt`.
  It produced `final_path_ce=6.9826`, `answer_state_loop_logit_ce=6.9826`,
  `final_path_acc=0.1429`, and `causal_prefix_examples=7`. This proves the
  strict path is trainable, not that it has recovered the answer.
- Next blocker remains scalar/list accumulator state to LM-logit synthesis,
  now under the stricter core-state-only answer-loop path.

2026-05-11 tail-weight/KISS update:

```text
tail-weight continuation:
  train:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    core_state_only_tail_weight_s005
  gate:
    /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
    mixed_noncopy_core_state_only_tail_weight_s005_gate_1case
  result:
    rejected_noncopy_lm_gate
    full=40000000
    typed_value_answer_bridge_off=44444444
    core_state_zero=Vega  555555
    answer_recurrent_off=00000000
  rank:
    target tokens 6 0 0 0 5 4 <eos>
    full ranks    8 1 1 1 3 2 5

KISS no-typed-bridge config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_
  core_state_only_kiss_answer_loop_s040.yaml

KISS no-train gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_no_train_gate_1case
  rejected_noncopy_lm_gate
  full=60060060
  core_state_zero=!!!!!!!!
  answer_recurrent_off=00000000

KISS 5-step gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_s005_gate_1case
  rejected_noncopy_lm_gate
  full=00000000
```

Roadmap decision:

```text
Do not continue local tail weighting, typed-bridge tuning, or short CE
continuations as if they were close to L4. The closest surface is the KISS
no-train readout, but its 5-step continuation regresses. This points to a
root recurrent-core-to-next-token synthesis objective, not another auxiliary
typed field or renderer.

Next accepted-gate candidate should first prove a minimal recurrent decoder or
latent-state-to-next-token readout on a small prior-backed reproduction, then
port that mechanism into the QTRM universal LLM path with the same strict
generation and destructive ablations.
```

L1 latent-readout reproduction, 2026-05-11:

```text
script:
  scripts/331_train_latent_readout_reproduction.py

decision doc:
  docs/wiki/decisions/latent-readout-reproduction.md

teacher-forcing-only out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_teacher_forcing_s200/report.json

scheduled-sampling out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_scheduled_s200/report.json

both profiles:
  teacher_forced_token_acc=1.0
  teacher_forced_exact=1.0
  greedy_token_acc=1.0
  greedy_exact=1.0
  decision=accepted
```

Interpretation:

```text
This accepts only the minimal readout reproduction. It proves that a small
recurrent readout can emit greedy exact digit/EOS tokens from sufficient,
token-aligned latent states. It does not prove QTRM has those states.

The next bottleneck test must port this contract into QTRM:
  core trajectory state -> per-output-step latent readout state ->
  recurrent next-token readout -> LM logits -> greedy text.

If that still fails, the upstream core state is not aligned to the answer token
sequence or does not contain the final scalar answer.
```

2026-05-16 Qwen-integrated native MCQ transfer update:

```text
Current bottleneck:
  donor/base preservation is now easier than donor/base correction.

Evidence:
  cloned Qwen core layer can preserve language and tie validation MCQ, but it
  has not produced reliable held-out base_wrong_core_correct flips.

best scaffold:
  local_eval/qwen35_integrated_language_knowledge_healing_clonecore_coreonly_basewrong_margin_preserve_valctrl_s120_20260516/report.json
  corrected interpretation: layer23 partial-unfreeze, not true core-only
  base 191/256, core 191/256, gain 0.0

true core-only:
  local_eval/qwen35_integrated_language_knowledge_healing_clonecore_true_coreonly_basewrong_margin_preserve_valctrl_s120_20260516/report.json
  base 191/256, core 190/256, gain -0.00390625

positive-gain attempts:
  residual_scale 0.08:
    rejected, best gain 0.0
  stronger base-correct KL:
    rejected, gain 0.0, zero harmful flips and zero corrective flips
  base_wrong retry4:
    rejected, gain -0.00390625, correction pressure caused preservation loss
  dual-stream retry3:
    rejected, gain -0.00390625, separate preservation stream still did not
    create base_wrong_core_correct flips

Decision:
  Do not run public 1K promotion from this stage. The next LLM-transfer
  bottleneck is a dual-stream preservation/correction objective or a stronger
  core placement/verifier target that creates base_wrong_core_correct flips
  without damaging base-correct rows. The first dual-stream attempt is not
  sufficient.
```
