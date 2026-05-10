# QTRM LLM Wiki Log

## [2026-05-07] experiment | Mixed composition length 11/13 accepted

Extended the accepted dynamic-halt mixed list-to-arithmetic Stage 1 gate from
held-out list lengths 7/9 to 11/13.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-len1113-s080.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_jointonly_s080_from_s720/last.pt
```

Result:

```text
S720 baseline on len11/13:
  trace exact:    32/32
  halted exact:  30/32

joint-only S080 recovery:
  len11/13 full:             32/32
  len11/13 transition-off:    0/32
  len11/13 code-shuffle:      0/32
  len11/13 code-dropout:      0/32
  canonical len7/9 full:     32/32
```

Decision:

```text
Accept as a Stage 1 length-transfer extension. The fix used only
transition-state joint CE; answer-token CE was disabled.
```

## [2026-05-07] experiment | Depth-text process CE rejected

Goal: test whether direct process credit on every recurrent depth's answer
logits can make deeper QTRM loops useful.

Implementation:

```text
new loss:
  core_depth_text_ce_loss

main forward:
  return_core_depth_text_logits=True when the loss is enabled

memory fix:
  drop return_core_depth_text_logits from preference rejected, counterfactual,
  short-trajectory, and canonical ablation forwards

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
core_depth_text_ce: about 11.44 -> 11.13
```

Held-out 24-case causal forced-choice:

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
Reject. The accepted baseline remains core_steps_1 14/24 and core_steps_4
13/24. Depth-text CE did not create deeper-loop gains.
```

Branch decision:

```text
Depth router heads, shortcut consistency, variable trajectory, and depth-text
process CE are all rejected for this donor-preserving controller branch.
The next architecture must train explicit verifiable recurrent state updates.
```

## [2026-05-07] experiment | Variable-trajectory v1 rejected

Goal: test the stronger version of the LoopFormer-inspired idea after
same-run shortcut consistency failed.

Implementation:

```text
long path:
  normal forward with outer_steps=4

short path:
  second forward on the same batch with outer_steps=1

new loss:
  core_variable_trajectory_consistency_loss

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_variable_traj_s080_causal_fc_24.jsonl
```

Training proxy moved:

```text
state cosine: about 0.22 -> 0.43
long-short logp margin: about -0.03 -> +0.27
```

Held-out 24-case causal forced-choice:

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
Reject. The accepted baseline remains core_steps_1 14/24 and core_steps_4
13/24. Variable-trajectory v1 optimized internal proxy metrics but did not
improve raw intelligence.
```

Next:

```text
Stop adding depth alignment/router heads for this branch.
Move to a recurrent transition objective: each loop step must perform a
verifiable state update, and the final answer path must lose the gain when
that update path is ablated.
```

## [2026-05-07] experiment | Shortcut-consistency v1 rejected

Goal: test whether a LoopFormer-inspired trajectory regularizer stabilizes the
accepted donor-preserving core-forced readout across recursive depths.

Implementation:

```text
new loss:
  core_trajectory_shortcut_consistency_loss

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120.yaml

init checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

trained checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_shortcut_outer4_s120_causal_fc_24.jsonl
```

Training signal was active:

```text
core_trajectory_shortcut cosine reached about 0.56 near the final log
```

24-case causal forced-choice result:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:           10/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off step8:         9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject shortcut-consistency v1 as a canonical improvement.
It preserves the known step-1 gain but does not improve it, lowers step-4
from 13/24 to 12/24, and collapses step-8 to donor/core_off.
```

Next:

```text
Do not add another post-hoc route head.
Implement a true variable-trajectory short/long schedule experiment with
correctness or verifier gating before claiming LoopFormer-style training.
```

## [2026-05-07] experiment | Adaptive halt probe on core-forced readout

Goal: test whether the accepted core-forced donor-preserving readout can choose
its own recursive depth instead of relying on unstable fixed-depth sweeps.

Implementation and eval hygiene:

```text
raw eval:
  fixed-depth modes now pass enable_core_halt=false explicitly
  qtrm_core_halt_steps_N_no_evidence is the only halt-enabled mode
  donor_only now disables the core instead of merely setting qtrm scale to 0
  choice telemetry records actual core_steps_mean

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_halt_teacher_depth_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_halt_teacher_depth_s080/last.pt

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_halt_teacher_depth_s080_causal_fc_24_v2.jsonl
```

The halt probe initialized from the accepted core-forced checkpoint and trained
only `core.halt_head.*` with teacher-depth stability targets. Training signal
was real but mostly told the model to continue to step 8:

```text
core_halt loss: 1.23 -> 0.69
teacher_depth_earliest_step_mean: 8.0
teacher_depth_step1_stable_rate: 0.0
```

24-case causal forced-choice result:

```text
donor_only:          9/24, core_steps_mean=0
core_off:            9/24, core_steps_mean=0
core_steps_1:       14/24, core_steps_mean=1
core_steps_2:        9/24, core_steps_mean=2
core_steps_4:       13/24, core_steps_mean=4
core_steps_8:       10/24, core_steps_mean=8
core_halt_steps_8:  10/24, core_steps_mean=8
delta_off:           9/24, core_steps_mean=8
```

Decision: reject teacher-depth stability halting as the next depth solution.
It learned a valid no-early-exit policy and reproduced fixed depth 8, but it
did not select the useful depths 1 or 4. The next depth selector must optimize
answer correctness / verifier preference directly, not only final-depth logit
stability.

Next candidate:

```text
supervised depth router:
  run fixed-depth candidates offline
  label each case with the shallowest correct depth, or UNKNOWN if none
  train a small router/verifier to select depth before answer scoring
  keep donor/core_off/delta_off ablations
```

Implemented the first label builder:

```text
script:
  scripts/depth_router_labels.py

labels:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_depth_router_labels_24.jsonl

summary:
  cases: 24
  donor_hits: 9
  oracle fixed-depth hits: 16
  causal_core_gains: 7
  unknown_routes: 8

target_route distribution:
  donor: 9
  core_steps_1: 5
  core_steps_4: 1
  core_steps_8: 1
  unknown: 8
```

Interpretation: the fixed-depth oracle is already 16/24, so there is routeable
signal above donor-only 9/24. The immediate bottleneck is learning/predicting
the route, not inventing another core path.

## [2026-05-06] experiment | Core-forced donor-preserving readout

Tested the third donor-preserving controller candidate after v1/v2 failed:

```text
donor logits remain the base policy
QTRM residual is bounded
n_coda_layers = 0
core_loop_readout_enabled = true
core_loop_readout_requires_core = true
```

This makes the QTRM delta collapse to zero when `core_off`; the model can no
longer get a raw-reasoning gain from a non-recursive coda side path.

Pure-recursive preference run:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_pure_recursive_pref_s160/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:           11/24
  core_steps_1:       14/24
  core_steps_2:       10/24
  core_steps_4:       11/24
  core_steps_8:       11/24
  delta_off:           9/24
  residual_gate_off:  11/24
```

Decision: partial signal only. The delta learned something because
`delta_off` returns to donor, but `core_off` also beats donor. The recursive
core is not yet the causal source.

Core-forced run:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:       14/24
  core_steps_2:        9/24
  core_steps_4:       13/24
  core_steps_8:       10/24
  delta_off:           9/24
  residual_gate_off:  10/24
```

Decision: accept as a causal architecture improvement, not as a final raw
recursive intelligence result. The gain now disappears under `core_off` and
`delta_off`, so the improved modes depend on the core-forced QTRM delta.
However, fixed deeper recursion is unstable: depth 1 and 4 help, depth 8
degrades.

Outer4 continuation:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_outer4_s120.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_outer4_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:       14/24
  core_steps_2:        9/24
  core_steps_4:       11/24
  core_steps_8:        8/24
  delta_off:           9/24
  residual_gate_off:   8/24
```

Decision: reject. Fixed deeper training did not improve deeper inference; it
regressed depth 4 and 8. Next candidate should be adaptive early-exit / depth
selection over the core-forced donor-preserving readout, not longer fixed
recursion.

## [2026-05-06] implementation | Donor-preserving logit guider gate wired

Implemented the first falsifier for the donor-preserving controller path:

```text
final logits = donor logits + bounded/gated QTRM residual delta
```

Added:

```text
forward ablation:
  disable_qtrm_residual_gate

raw eval modes:
  qtrm_core_steps_8_delta_off_no_evidence
  qtrm_core_steps_8_residual_gate_off_no_evidence

canonical config:
  configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml
```

Verification:

```text
tests.test_raw_intelligence_eval_script
tests.test_model_config
tests.test_training_checkpoint_init
149 tests OK
py_compile OK
git diff --check OK
```

Next gate: train S120, then compare donor-only, guided, delta-off, gate-off,
and core-off under no-retrieval raw-intelligence evaluation.

Smoke result:

```text
checkpoint:
  runs/qwen35_2b_4090_donor_preserving_logit_guider_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:        9/24
  core_steps_2:        9/24
  core_steps_4:        9/24
  core_steps_8:        9/24
  delta_off:           9/24
  residual_gate_off:  10/24
```

Decision: not accepted. The gate-off mode beating the full gated mode suggests
the learned residual gate is currently too conservative or undertrained. Next
candidate is KISS bounded delta without a learned residual gate; donor
preservation remains enforced by donor scale 1.0, clamp, donor KL, and
donor-correct margin.

Follow-up v2:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_bounded_delta_nogate_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:        9/24
  core_steps_2:        9/24
  core_steps_4:        9/24
  core_steps_8:        9/24
  delta_off:           9/24
  residual_gate_off:   9/24
```

Decision: v2 is also not accepted. Removing the gate did not create a causal
raw-reasoning gain. Root hypothesis update: the first two runs trained on
generic text/math/mm data, not on the pure-recursive preference data used by
the raw-intelligence gate. Next candidate must train the donor-preserving delta
on `data/filtered/pure_recursive_reasoning_preferences_train.jsonl`.

## [2026-05-06] research | Donor-preserving controller method selected

Web search over 2025-2026 reasoning-control prior found the best next path:

```text
ThinkLogit / Proxy-Tuning: logit arithmetic guider over frozen target
ReFT / GLoRE / BREP: bounded representation interventions in frozen LMs
BuPO: internal layer policies, especially relevant to Qwen-series structure
Dead Weights: frozen LMs can communicate through learned residual hooks
Speculative Thinking / Thinking Intervention: intervene only at reasoning points
```

Decision: implement the simpler bounded donor-logit guider before attempting
Qwen residual-stream hooks.

```text
final_logits = donor_logits + alpha * gate * clamp(qtrm_delta_logits)
```

Artifact:

```text
docs/wiki/sources/donor-preserving-reasoning-control-2026.md
docs/wiki/decisions/donor-preserving-controller-next-method.md
```

## [2026-05-06] renderer | Hidden bridge S080 rejected

Added a zero-init answer-state hidden bridge and trained only that bridge from
the accepted answer-halt S080 checkpoint.

Result:

```text
generation smoke4:
  full:              0/4
  hidden_bridge_off: 0/4
  halt_gate_off:     0/4

causal forced-choice smoke4:
  full:              0/4
  hidden_bridge_off: 4/4
```

Decision: reject and delete the generated checkpoint. This is a strong
root-architecture signal: the bridge does not open generation and actively
damages the accepted forced-choice behavior. Next candidate should preserve the
donor decoder/language path and inject QTRM as a residual cognitive controller,
instead of stacking more private QTRM renderer patches.

Artifact:

```text
docs/wiki/decisions/ouro-hidden-bridge-s080-reject.md
```

## [2026-05-06] renderer | LM-head-only decoder alignment rejected

After donor-unembedding head surgery failed, added `trainable_param_policy:
lm_head_only` to train only the untied final LM head from the accepted
answer-halt S080 checkpoint.

Result:

```text
generation smoke4:
  full/gate_off: 0/8

causal forced-choice smoke8:
  full:          0/8
  halt_gate_off: 0/5 observed before early stop
```

Decision: reject and delete the generated checkpoint. The update changed token
priors but destroyed the accepted forced-choice reasoning signal and still did
not make greedy generation correct. The next renderer candidate should align
answer-state hidden vectors through an ablatable LM-compatible hidden bridge,
not train only the final vocabulary head.

Artifact:

```text
docs/wiki/decisions/ouro-lm-head-only-decoder-alignment.md
```

## [2026-05-06] renderer | Donor-unembedding head surgery rejected

Web/prior search focused on output embedding geometry and latent-thought
rendering:

```text
Press/Wolf output embedding
Inan/Khosravi/Socher tied word vectors
PonderLM-2 latent thoughts before token prediction
Coconut official latent reasoning implementation
```

Implemented `scripts/246_build_donor_unembedding_aligned_checkpoint.py` to
project Qwen donor output embeddings into QTRM width and replace only the
untied LM head of the accepted Ouro answer-halt checkpoint.

Result:

```text
pinv donor-output head:
  generation smoke4: 0/8
  causal forced-choice smoke8 full: 6/8

direct projection donor-output head:
  generation smoke4: 0/8
```

Decision: reject both and delete both generated checkpoint weights. The failure
is not solved by vocab-head geometry alone; answer-state hidden vectors are not
yet aligned to an LM-compatible hidden manifold. Next candidate should train a
decoder/projection from answer-state hidden to tokenizer logits, PonderLM-style,
instead of adding more small margin losses to the current head.

Artifact:

```text
docs/wiki/decisions/ouro-donor-unembedding-head-probe.md
```

## [2026-05-06] renderer | Causal-prefix self-rollout rejected

Added online self-rollout causal-prefix supervision to the Ouro answer renderer
probe. The training path now can build prefix examples from the model's own
greedy generated tokens, then supervise the next gold token from that generated
prefix.

Result:

```text
generation smoke4:
  full/gate_off: 0/8

causal forced-choice smoke8:
  full:          8/8
  halt_gate_off: 0/8
```

Decision: reject as renderer and delete the checkpoint. This preserves the
accepted answer-halt forced-choice gate but does not make the model render the
answer autoregressively. The active bottleneck is no longer teacher-forcing
exposure bias alone; the answer-state-to-token rendering path itself is too
weak.

Additional beam diagnostic on the accepted S080 baseline:

```text
beam_width: 16
max_new_tokens: 7
heldout rows: 4
hits: 0/4
best samples: "1 1 1", "UNKNOWN 1 1"
```

Interpretation: the generation failure is not greedy-only decoding. Short beam
search also collapses to the same invalid token priors, so the next renderer
candidate should align answer-state hidden vectors to tokenizer logits more
directly instead of adding another small CE patch.

Artifact:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Typed Algorithmic Value-State Probe

Added a role-separated typed value-state probe for the accepted mixed
composition len11/13 checkpoint:

```text
raw_list_offsets
doubled_list_offsets
scalar_coeff
scalar_residual
final_residual
```

Result:

```text
held-out len11/13 mixed-only:
  content-field accuracy: 352/1024 = 0.34375
  head-off content acc:     0/1024 = 0.0
  trace exact:              0/32

action-code:
  len11/13 exact: 32/32
  len7/9 exact:   32/32
```

Decision:

```text
Accept only as a causal Stage-2 probe improvement. The typed field path learns
some value content and preserves the action controller, but it is not an exact
neural transition model yet. Next step is a recurrent typed value-transition
cell with delta-off/shuffle ablations.
```

Decision doc:

```text
docs/wiki/decisions/typed-algorithmic-value-state-len1113-s080.md
```

## [2026-05-06] raw-intelligence | Terminal-depth CE rejected

Added terminal-only answer CE:

```text
--terminal-depth-ce-weight
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
action-code: 32/32
```

Decision: reject. The terminal CE loss is active, but it only preserves the
accepted 4/8 score and removes the full-vs-bridge-off causal gap. Next
candidate should gate answer-state updates inside the recurrence with finality,
not add another post-hoc CE or selector.

Artifact:

```text
docs/wiki/decisions/ouro-terminal-depth-ce-s020-reject.md
```

## [2026-05-06] raw-intelligence | Subtract-tail counterfactual rejected, depth overshoot found

Added train-only counterfactual negatives for mixed subtract-tail failures:

```text
--subtract-tail-counterfactual-margin-weight
--subtract-tail-counterfactual-margin
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 4/8
full core8:   3/8
bridge_off:   2/8
action-code: 32/32
```

Decision: reject. The bridge path is causal again, but full core8 regresses
below the accepted 4/8 baseline. New bottleneck: recursive-depth overshoot.
Depth 4 matches the baseline while depth 8 worsens.

Artifact:

```text
docs/wiki/decisions/ouro-subtract-tail-counterfactual-s020-reject.md
```

## [2026-05-06] raw-intelligence | Trajectory monotonic process credit rejected

Added adjacent-depth process-credit training:

```text
--depth-trajectory-monotonic-weight
--depth-trajectory-monotonic-margin
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
action-code: 32/32
```

Decision: reject. Full preserves the accepted 4/8 score but bridge-off matches
full, so the objective does not prove a causal raw-intelligence gain. Logged
`depth_trajectory_monotonic` was already 0.0000, which means this was not the
active bottleneck.

Artifact:

```text
docs/wiki/decisions/ouro-trajectory-monotonic-s020-reject.md
```

## [2026-05-06] raw-intelligence | Bridge contrast creates causal dependency

Added `transition_joint_answer_bridge_contrastive_loss`, which contrasts the
full answer path against the same path with
`disable_transition_state_joint_answer_bridge=True`.

Result:

```text
train:
  final_path_ce: 4.0516 -> 2.6656
  bridge contrast delta: -0.0525 -> -0.0059

held-out LM forced-choice:
  full:       2/8
  bridge off: 0/8

action-code:
  exact: 32/32
```

Decision: accept as a smoke probe. This does not improve full accuracy over
S80, but it fixes the previous bridge failure class: the transition-joint
bridge is no longer ignorable under held-out ablation. Scale only with
save-every validation and reject continuations where bridge-off matches full.

Artifact:

```text
docs/wiki/decisions/ouro-transition-joint-answer-bridge-contrast-s020.md
```

Update: continued S020 for 60 more steps with validation checkpoints. Step20
and step60 both preserve the same held-out causal pattern:

```text
full:       2/8
bridge off: 0/8
action-code at step60: 32/32
```

Decision: accept causal preservation through S080-from-S020, but do not claim
quality improvement. The next objective needs to break the 2/8 smoke ceiling,
not only preserve bridge causality.

## [2026-05-06] raw-intelligence | Transition joint answer bridge rejected

Implemented a direct causal bridge from `transition_state_joint_logits` into
the recurrent answer-state loop and trained 20 steps from the accepted S80
checkpoint.

Result:

```text
train:
  final_path_ce: 3.9788 -> 2.6370

held-out LM forced-choice:
  full:       2/8
  bridge off: 2/8

action-code:
  exact: 32/32
```

Decision: reject as canonical. The answer path can now see the transition
trace, but it does not causally depend on it under the current objective. Next
candidate should add bridge-contrast or depth-wise answer-process supervision,
then require full > bridge-off before any scale-up.

Artifact:

```text
docs/wiki/decisions/ouro-transition-joint-answer-bridge-s020.md
```

## [2026-05-06] raw-intelligence | Finality selector rejected

Tested whether the S80 Ouro recurrent checkpoint's already-correct
transition-state finality signal can choose a better answer-state-loop depth
without retraining.

Result:

```text
soft selector:
  full:         2/8
  selector off: 2/8

hard_first selector:
  full:         2/8
  selector off: 2/8
  action-code:  32/32
```

Decision: reject as canonical. The transition trace remains correct, but a
post-hoc selector does not causally improve answer-token logits. Next candidate
should feed transition code/state into answer-state recurrence at every step.

Artifact:

```text
docs/wiki/decisions/ouro-finality-selector-zeroshot.md
```

## 2026-05-05: Universal LLM Causal Path Principle Added

Added a canonical principle for future QTRM architecture work:

```text
prompt/chat-template tokens
-> tokenizer
-> donor/token hidden states
-> QTRM workspace/core/memory
-> LM logits
-> autoregressive text
```

Structured modules such as typed registers, operation selectors, verifier
heads, and memory readers may be used only as learned and ablatable internal
bottlenecks. They must not become external calculators, hidden answer channels,
or runtime rule solvers that compute the final answer while the LLM only
formats it.

Artifacts:

```text
skill:
  ~/.agents/skills/research-driven-architecture-debugging/SKILL.md

wiki:
  docs/wiki/architecture/universal-llm-causal-path-contract.md
  docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md
  docs/wiki/architecture/canonical-architecture-matrix.md
```

Implication for the next typed-register executor:

```text
It must preserve the general LLM path. The verifier may judge register updates
during training/eval, but the learned model must perform the update causally at
inference and final answers must come through LM logits.
```

## [2026-05-05] raw-intelligence | Role-value slot candidate opened

Added the role/filler variable-slot prior page and the next falsifiable
candidate after generic algorithmic slots failed.

Artifacts:

```text
docs/wiki/sources/role-filler-variable-slots.md
docs/wiki/decisions/role-value-state-candidate.md
references/papers/role_value_slots/
```

Decision status: active candidate. The next gate is a short role-value state
train/eval: content role accuracy must beat the 0.05 generic-slot baseline,
trace exact must rise above 0/32, and the accepted action-code path must remain
32/32.

Update: rejected after the 480-step gate. Value accuracy rose to 48/624
(0.0769), but trace exact stayed 0/32 and step exact stayed 0/256. Action-code
behavior remained 32/32. Next candidate: make role-bound values part of the
mandatory recurrent state rather than a readout-only probe.

## [2026-05-05] raw-intelligence | Core role-value state scaffold partially accepted

Moved role-bound value tokens into the mandatory recurrent core and trained the
new core-role path from the accepted mixed-composition action checkpoint.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     0/256
  value accuracy: 100/624 = 0.1603

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: accept only as a scaffold. It beats readout-only role values
(0.0769 -> 0.1603) without breaking the accepted action controller, but exact
latent value-state remains 0/32. The next candidate is joint core training with
action preservation plus core role-value CE, not another readout head.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-s480.md
```

## [2026-05-05] raw-intelligence | Joint core role-value training improves value accuracy

Opened the recurrent core together with answer-state loop, transition-state
joint head, and core role-value tokens. The objective preserved action-code
behavior while continuing to train role-bound values.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     0/256
  value accuracy: 144/624 = 0.2308

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: accept the direction, reject final exact-state claim. The progression
is now 0.0500 -> 0.0769 -> 0.1603 -> 0.2308, while action-code stays 32/32.
Next bottleneck: add explicit previous-role-state to next-role-state
transition supervision.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-joint-s240.md
```

## [2026-05-05] raw-intelligence | Len579 gate exposes prompt-to-role binding bottleneck

Found a gate/data confound: the previous value-state train split had mixed
list length 5 only, while eval used lengths 7 and 9. Added
`--train-list-lengths` to the mixed composition builder and generated a
length-5/7/9 mixed-only train split.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     16/256 = 0.0625
  value accuracy: 184/624 = 0.2949

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: current best value-state scaffold, still not final. The progression
is now 0.0500 -> 0.0769 -> 0.1603 -> 0.2308 -> 0.2949, with first non-zero
step exact. Next bottleneck is prompt-to-role binding: role tokens need a
direct role-conditioned extraction path from prompt token states before the
mandatory recurrent core.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-len579-s240.md
```

## [2026-05-05] raw-intelligence | Prompt-extract role binding candidate rejected

Implemented gated role-conditioned prompt extraction before the recurrent core
and trained it for 240 steps from the best len579 scaffold.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     16/256 = 0.0625
  value accuracy: 130/624 = 0.2083

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: reject as implemented. It preserves action but regresses from the
best len579 value accuracy of 0.2949. Keep the code as experimental; do not
make it canonical without extractor pretraining or a better causal gate.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-prompt-extract-s240.md
```

## [2026-05-05] raw-intelligence | Algorithmic value-state slots rejected

Tested structured neural-algorithmic value targets instead of digit strings:
kind=list/scalar plus relative list/scalar slots.

Result:

```text
pad-including CE:
  kind acc:         1.0000
  content slot acc: 0.0000
  trace exact:      0/32

content-slot CE:
  kind acc:         1.0000
  content slot acc: 0.0500
  trace exact:      0/32

action-code preservation:
  trace exact:      32/32
  halted exact:     32/32
```

Decision: reject. The model learns phase/kind but not exact value content.
Next candidate is typed algorithmic fields rather than one generic slot head.

Artifact:

```text
docs/wiki/decisions/algorithmic-value-state-s480.md
```

## [2026-05-05] raw-intelligence | Factorized value-state path rejected

Implemented the first state-factorized candidate: separate recurrent value
slots conditioned by prompt context and the frozen accepted action-state
trajectory.

Result:

```text
value-state:
  trace exact: 0/32
  token acc:   0.3713

action-code preservation:
  trace exact: 32/32
  halted:      32/32
```

Decision: reject. The action policy is preserved, but digit-sequence CE still
does not create exact algorithmic value state. Next step is structured
neural-algorithmic state targets: list element slots, accumulator slots, masks,
and explicit transition loss.

## [2026-05-05] raw-intelligence | Value-state-only control rejected

Ran the control from the state-factorized plan: freeze the accepted core/action
path and train only the compact value-state head.

Result:

```text
value-state:
  trace exact: 0/32
  token acc:   0.3794

action-code preservation:
  trace exact: 32/32
  halted:      32/32
```

Decision: reject value-readout-only as a solution. The accepted action policy
is preserved when frozen, but exact value content is not present in the current
latent trajectory. Proceed to state-factorized core instead of more readout
tuning.

## [2026-05-05] raw-intelligence | State-factorized core plan opened

Opened the research-backed architecture plan for the current bottleneck:
preserve the accepted latent action controller while adding causal
value-bearing recurrent state.

Artifact:

```text
docs/wiki/decisions/state-factorized-qtrm-core-research-plan.md
```

Prior axes:

```text
Neural Algorithmic Reasoning: latent processor and intermediate state.
Dreamer/RSSM: separated recurrent/action/value latent signals.
Factored Latent Action World Models: factorized action/state transition.
TRM/latent reasoning: recursive latent compute remains the raw-intelligence core.
```

Immediate experiment: freeze core/action and train only the compact
value-state readout to test whether accepted latent trajectories already
contain decodable value content.

## [2026-05-05] raw-intelligence | Compact value-state head rejected

Tested whether a compact digit/comma/minus value-state head can recover
value-bearing latent state without damaging the accepted mixed-composition
action policy.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-value-state-s480.md

evaluator:
  scripts/237_eval_qtrm_value_state.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_s480_from_mixed_s720/last.pt
```

Result:

```text
value-state:
  trace exact:    0/32
  token acc:      0.3713

action-code preservation:
  trace exact:   10/32
  step acc:       0.8477
```

Decision: reject. The compact value probe partially learns digits, but joint
training contaminates the accepted latent action policy. The next candidate
must factor action-state and value-state so values become causal without
regressing the action controller.

## [2026-05-05] raw-intelligence | State-sequence bottleneck rejected

Tested whether the accepted mixed-composition checkpoint contains decodable
value-bearing internal state, not only a correct latent action code.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-state-sequence-bottleneck-s480.md

evaluator:
  scripts/236_eval_qtrm_core_state_sequence.py
```

Result:

```text
accepted action-code checkpoint:
  state trace exact: 0/32
  state token acc:   0.0040

joint core/readout sequence CE:
  state token acc:   0.3713
  action-code exact: 0/32

readout-only:
  state token acc:   0.2273
  action-code exact: 32/32

direct transition-state sequence head:
  state token acc:   0.1418
  action-code exact: 32/32
```

Decision: reject state-content claims for the current core. The next bottleneck
is value preservation inside the recurrent latent update, not another readout
head.

## [2026-05-05] raw-intelligence | Mixed-family composition gate accepted

Accepted the next Stage 1 gate after list length/value transfer: list state now
feeds an arithmetic aggregation/subtraction step.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-s720.md

builder:
  scripts/235_build_mixed_family_composition_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt
```

Result:

```text
full held-out:
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  exact:         0/32
  halted exact:  0/32

code shuffle:
  exact:         0/32
  halted exact:  0/32

code dropout:
  exact:         0/32
  halted exact:  0/32
```

Key architecture correction: `dynamic_halt_v3` separates action identity from
halt/finality, and the mixed-train rows are interleaved because the training
script consumes rows deterministically.

## [2026-05-05] raw-intelligence | Long list transfer gate accepted

Scaled the accepted list paraphrase transfer gate to held-out list lengths and
value ranges.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-list-transfer-long-s120.md

builder:
  scripts/234_build_list_transfer_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/last.pt
```

Split:

```text
train:
  list variants: 0, 1, 2, 3, 4, 5
  list length:   5

eval:
  list variants: 6, 7
  list lengths:  7, 9
  rows:          32
```

Result:

```text
full:
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  exact:         0/32
  halted exact:  0/32

code shuffle:
  exact:         0/32
  halted exact:  0/32

code dropout:
  exact:         0/32
  halted exact:  0/32
```

Interpretation: accepted as within-family length/value/surface transfer. Do not
promote to full operation-family zero-shot or open-ended reasoning.

## [2026-05-04] raw-intelligence | Prompt-conditioned primitive transition head

Fixed a causal information-path failure in the QTRM primitive transition
experiment.

Previous path:

```text
core_depth_states only -> primitive operation logits
```

Observed failure:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_s240_oponly/last.pt

heldout first64:
  operation accuracy: 128/256 = 0.5000

failure:
  family-insensitive operation schedule collapse
```

Implemented path:

```text
canonical prompt tokens
-> prelude text context
-> masked prompt pooled context
-> concat(core_depth_state, prompt_context)
-> primitive operation logits
```

Config:

```text
primitive_transition_prompt_context_enabled: true
```

Result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptctx_s720_oponly/last.pt

train:
  operation accuracy: 512/512 = 1.0000

heldout existing:
  operation accuracy: 250/256 = 0.9766

heldout7000:
  operation accuracy: 507/512 = 0.9902
```

Remaining failure is localized to list transforms:

```text
double_filtered -> second_mapping
```

Decision record:

```text
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240.md
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240-summary.json
```

## [2026-05-03] gate | Temporal-spatial context causal gate wired

Added a concrete train/eval/gate path for temporal-spatial context conditioning:

```text
scripts/207_build_temporal_spatial_context_cases.py
  builds SSOT-derived temporal_freshness and spatial_relation JSONL cases

scripts/196_train_pure_recursive_depth_supervised.py
  now forwards row.temporal_spatial_context to student and teacher models

scripts/192_eval_raw_intelligence.py
  now forwards case.temporal_spatial_context and supports
  qtrm_core_steps_N_temporal_spatial_off_no_evidence

src/qtrm_mm/eval/raw_intelligence_gate.py
  gate_type=temporal_spatial_context

scripts/208_run_temporal_spatial_context_gate.sh
  builds train/eval data, trains the context probe, runs context-on/off eval,
  and writes the wiki gate report
```

Generated case files:

```text
data/train/temporal_spatial_context_train_120.jsonl
data/eval/temporal_spatial_context_heldout_24.jsonl
```

No capability claim is made until
`qtrm_core_steps_8_no_evidence` beats
`qtrm_core_steps_8_temporal_spatial_off_no_evidence` on the held-out gate.

Pipeline smoke:

```text
command shape:
  PYTHON_BIN=.venv/bin/python TRAIN_CASES_PER_FAMILY=1 EVAL_CASES_PER_FAMILY=1
  STEPS=1 bash scripts/208_run_temporal_spatial_context_gate.sh

result:
  pipeline completed
  eval records: 4
  smoke status: rejected
```

The rejection is expected for a 1-step smoke. It proves the runner, model
forward path, eval ablation, and gate writer are connected; it does not test
capability.

Full S240 gate result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/temporal_spatial_context_probe/s240/last.pt

eval:
  local_eval/temporal_spatial_context_gate.jsonl

gate:
  docs/wiki/decisions/temporal-spatial-context-gate.md
  docs/wiki/decisions/temporal-spatial-context-gate-summary.json

status:
  rejected

context_on:
  qtrm_core_steps_8_no_evidence = 4/24

context_off:
  qtrm_core_steps_8_temporal_spatial_off_no_evidence = 4/24

task family:
  temporal_freshness = 4/12 on, 4/12 off
  spatial_relation = 0/12 on, 0/12 off
```

Interpretation: the new context path is wired and ablatable, but the S240
training recipe did not make it causally useful on held-out cases. The next
architecture/training step should not claim temporal/spatial intelligence; it
should attack spatial failure and context-off tie directly.

Delta analysis:

```text
script:
  scripts/209_analyze_temporal_spatial_context_delta.py

summary:
  docs/wiki/decisions/temporal-spatial-context-delta-summary.json

paired cases:
  24

changed completions:
  0

context-on only correct:
  0

context-off only correct:
  0

mean chosen-logprob delta:
  +0.00048
```

Interpretation: the context path has near-zero behavioral effect. This is a
stronger failure than merely tying accuracy; it means the current training
recipe did not make the model depend on the temporal/spatial prefix tokens.

Implemented next diagnostic pressure:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --temporal-spatial-context-contrast-weight
  --temporal-spatial-context-contrast-margin

loss:
  max(0, margin - (logp_context_on(answer) - logp_context_off(answer)))

scripts/208_run_temporal_spatial_context_gate.sh
  TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT default = 0.5
```

1-step runtime smoke passed with contrast enabled and logged
`context_contrast` plus `context_contrast_target_logp_delta`. The next full
run should compare whether this changes the previously near-zero delta.

## [2026-05-03] architecture | Temporal-spatial context conditioning added

Added a small QTRM forward-path extension for time/space awareness:

```text
src/qtrm_mm/config.py
  temporal_spatial_context_enabled
  temporal_spatial_context_dim
  temporal_spatial_context_max_tokens

src/qtrm_mm/qtrm_model.py
  temporal_spatial_context -> projected prefix tokens
  disable_temporal_spatial_context ablation flag
  temporal_spatial_context_token_count telemetry

src/qtrm_mm/training/train.py
  trainable_param_policy=core_and_temporal_spatial_context
```

This does not change the root architecture. It keeps the same
SSOT -> workspace -> recursive core -> coda/readout path, but lets the core see
SSOT-derived time/space facts as model tokens rather than hidden runtime state.

Decision doc:

```text
docs/wiki/decisions/temporal-spatial-context-conditioning.md
```

## [2026-05-03] recovery | Metacognitive checkpoint rebuild path added

The old metacognitive baseline/candidate checkpoints are still unreadable from
`/mnt/sdb1`, so exact recovery is blocked unless a readable backup exists. Added
a separate recovery path that builds a new matched pair on a healthy disk:

```text
script:
  scripts/206_run_metacog_pair_rebuild.sh

default destination:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild

baseline:
  no_warmup_rebuilt_s001/last.pt

candidate:
  unknown_teacher_kl_conservative_rebuilt_s040/last.pt
```

The trainer now supports an explicit random-init recovery mode:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --allow-random-init
```

Without `--init-checkpoint` or `--allow-random-init`, training fails. This keeps
artifact recovery honest: a random-init matched pair can restore the eval loop,
but it is not the same result as the old unreadable s001/conservative-s040
checkpoints.

Also wired `ALL_DEPTH_CE_WEIGHT` through
`scripts/197_run_pure_recursive_depth_supervised_train.sh`, matching the
documented conservative teacher-KL setting.

Decision doc:

```text
docs/wiki/decisions/metacog-checkpoint-rebuild-recovery.md
```

## [2026-05-03] architecture | donor-QTRM conflict gate probe added

Full 40-case fusion-scale sweep was attempted after the smoke8 result, but the
`runs` symlink points to `/mnt/sdb1/ws-sky-data/qtrm-runs` and that mount began
returning I/O errors:

```text
runs -> /mnt/sdb1/ws-sky-data/qtrm-runs
findmnt -T runs:
  /mnt/sdb1 /dev/sdb1 ext4 rw,noatime,emergency_ro

failure:
  OSError: [Errno 5] Input/output error
  path: runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
```

The failure is an artifact-storage/checkpoint-read blocker, not a model result.
Do not trust the partial full-sweep file under `runs/eval`.

Implemented the next KISS fusion diagnostic instead:

```text
src/qtrm_mm/config.py
  donor_qtrm_conflict_gate_enabled
  donor_qtrm_conflict_qtrm_scale

src/qtrm_mm/qtrm_model.py
  donor/QTRM top-token conflict gate before donor-logit fusion

scripts/192_eval_raw_intelligence.py
  --donor-qtrm-conflict-gate
  --donor-qtrm-conflict-qtrm-scale

scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
  writes full 40-case fusion sweep JSONL files to local_eval/
  compares candidate plain and candidate conflict-gate against baseline
  preflights checkpoint byte-read and output-dir write/delete

scripts/205_localize_metacog_checkpoints.sh
  copies baseline/candidate checkpoints to /mnt/nvme1n1p2/qtrm-local-checkpoints
  verifies source/destination sha256
  currently blocked on source checkpoint I/O error

docs/wiki/decisions/donor-qtrm-conflict-gate-probe.md
```

Verification:

```text
forced-choice eval now writes donor_qtrm_conflict_gate_mean per choice
calibration reports aggregate mean_predicted_conflict_gate and mean_choice_conflict_gate
localize helper fails fast at source checkpoint read, before copying
```

Decision:
this is an ablation probe, not a promoted architecture. Once `/mnt/sdb1` is
stable or checkpoints are copied to a healthy disk, rerun the 40-case fused
calibration gate with and without this conflict gate. If it helps, replace the
heuristic with a learned fusion calibration/router. If it does not, move to
trained reliability targets rather than more core-only loss tuning.

## [2026-05-03] evaluation | Split metacognitive gates isolate QTRM core gain from fusion failure

Added profile-aware metacognitive calibration gates:

```text
script:
  scripts/202_build_metacognitive_calibration_gate.py

profiles:
  strict: all records, existing behavior
  qtrm_core: qtrm_core_steps_8_no_evidence + qtrm_core_steps_8_qtrm_only_no_evidence
  fused: qtrm_core_steps_8_no_evidence + qtrm_core_steps_8_low_donor_no_evidence
```

Added explicit fusion-scale eval modes:

```text
script:
  scripts/192_eval_raw_intelligence.py

examples:
  qtrm_core_steps_8_donor_scale_0p50_no_evidence
  qtrm_core_steps_8_qtrm_scale_0p75_donor_scale_0p50_no_evidence
```

Held-out split on the conservative unknown-only teacher-KL checkpoint:

```text
qtrm_core report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-qtrm-core-gate.md
  status: accepted
  accuracy: 0.600000 -> 0.600000
  ECE:      0.408397 -> 0.394158  delta -0.014239
  Brier:    0.399219 -> 0.398868  delta -0.000351

fused report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-fused-gate.md
  status: rejected
  accuracy: 0.600000 -> 0.600000
  ECE:      0.348036 -> 0.362485  delta +0.014450
  Brier:    0.364187 -> 0.366383  delta +0.002196

fusion-scale smoke8:
  docs/wiki/decisions/metacog-fusion-scale-sweep-conservative-s040-smoke8.md
  status: rejected
  accuracy: 0.500000 -> 0.500000
  ECE:      0.498750 -> 0.499257  delta +0.000507
  Brier:    0.497586 -> 0.498591  delta +0.001005
  donor_scale_0p25 ECE delta: +0.000002
  donor_scale_1p0  ECE delta: +0.001601
```

Interpretation:
the current evidence supports a narrow QTRM-core metacognition improvement, not
a full architecture promotion. The donor/QTRM fusion path is now the active
failure class, especially `qtrm_core_steps_8_low_donor_no_evidence`. The next
architecture move should target fusion calibration directly instead of further
tuning the pure core loss and hoping the fused path follows. The small scale
sweep suggests the degradation grows as donor scale rises, but it is only an
8-case smoke and should be followed by a full 40-case sweep before adding a
learned fusion module.

## [2026-05-03] evaluation | Conservative unknown-only teacher-KL probe still rejected

Reduced update strength after the unknown-only S080 run worsened global and
low-donor fused calibration:

```text
checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt
eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_conservative_s040_40.jsonl
report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-calibration-heldout40.md

settings:
  steps: 40
  lr: 2.0e-6
  teacher_depth_kl_weight: 5.0
  all_depth_ce_weight: 0.10
  choice_margin_weight: 0.25
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.340911  delta +0.006972
Brier:                     0.286778 -> 0.287452  delta +0.000674
avg confidence when wrong: 0.635804 -> 0.637578  delta +0.001773
status: rejected
```

Signal:

```text
qtrm_core_steps_8_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_low_donor_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier worsened slightly
```

Interpretation:
stronger teacher KL and lower update strength move in the right direction, but
the fused path still rejects. The next architectural/training fix should
separate QTRM-only calibration from donor-fusion calibration, likely by adding a
small learned donor/QTRM fusion calibration gate or by evaluating the QTRM-only
metacognition claim separately from fused-generation behavior.

## [2026-05-03] evaluation | Unknown-only teacher-KL metacognitive probe rejected

Added a selective preference filter and trained only on `expected_unknown=true`
metacognitive rows with teacher-depth KL preservation.

Code/data:

```text
scripts/194_build_pure_recursive_reasoning_preferences.py
  --only-expected-unknown

tests/test_pure_recursive_reasoning_preferences.py

data/filtered/metacognitive_calibration_unknown_preferences_train.jsonl
  72 preference rows
```

Run:

```text
checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_s080/last.pt
eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_s080_40.jsonl
report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-s080-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.363077  delta +0.029138
Brier:                     0.286778 -> 0.298315  delta +0.011536
avg confidence when wrong: 0.635804 -> 0.649499  delta +0.013695
status: rejected
```

Interpretation:
selecting only UNKNOWN/OOD/contradiction rows improves expected-unknown slices
but makes answerable and low-donor fused calibration worse. It preserves
QTRM-only accuracy, but the checkpoint is not a valid metacognitive gain. The
next conservative check should reduce update strength or increase teacher-KL
weight before introducing more architecture.

## [2026-05-03] evaluation | Teacher-depth KL metacognitive probe rejected but preserved QTRM-only policy

Ran the next metacognitive probe with checkpoint preservation:

```text
teacher checkpoint:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
student init:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
checkpoint:
  runs/qwen35_2b_4090_metacog_teacher_kl_s080_v2/last.pt
eval:
  runs/eval/metacognitive_calibration_teacher_kl_s080_v2_40.jsonl
report:
  docs/wiki/decisions/metacog-teacher-kl-s080-v2-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.352499  delta +0.018560
Brier:                     0.286778 -> 0.292267  delta +0.005489
avg confidence when wrong: 0.635804 -> 0.643463  delta +0.007658
status: rejected
```

Mode-specific signal:

```text
qtrm_core_steps_8_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_low_donor_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier worsened
```

Interpretation:
teacher-depth KL fixed the previous direct forced-choice failure where QTRM-only
accuracy dropped to 0.425. It is useful as policy preservation. It is not enough
as the final metacognitive objective because global calibration and low-donor
fused calibration worsen. Next candidate: keep teacher KL, but apply calibration
pressure selectively to overconfident wrong/UNKNOWN rows rather than all
answerable rows.

## [2026-05-03] evaluation | Direct metacognitive forced-choice probe rejected

Built a separate metacognitive train split and trained a direct known/unknown
forced-choice probe from the no-warmup baseline.

Artifacts:

```text
train cases:
  data/filtered/metacognitive_calibration_train_40.jsonl
preferences:
  data/filtered/metacognitive_calibration_preferences_train.jsonl
checkpoint:
  runs/qwen35_2b_4090_metacog_forced_choice_s080/last.pt
eval:
  runs/eval/metacognitive_calibration_forced_choice_s080_40.jsonl
report:
  docs/wiki/decisions/metacog-forced-choice-s080-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.433333  delta -0.033333
ECE:                       0.333939 -> 0.295489  delta -0.038450
Brier:                     0.286778 -> 0.293834  delta +0.007056
avg confidence when wrong: 0.635804 -> 0.534910  delta -0.100895
status: rejected
```

Mode-specific signal:

```text
qtrm_core_steps_8_low_donor_no_evidence:
  accuracy 0.60 -> 0.75
  ECE/Brier worsened

qtrm_core_steps_8_no_evidence:
  accuracy 0.60 -> 0.425
  ECE/Brier improved because the model became less confident

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy 0.60 -> 0.425
  same failure as full QTRM-only path
```

Interpretation:
the probe taught caution, but it damaged the pure QTRM answer policy. This is
not accepted as raw metacognitive intelligence. The next candidate should add
policy-preservation pressure, such as teacher-depth KL from the accepted
no-warmup core path, while applying calibration loss only where the baseline is
overconfident or wrong.

## [2026-05-03] training/evaluation | Uniform noise warm-up probe added but strict QTRM-mode gate rejects

The 40-case random-label warm-up result showed that a random-label CE objective
does not robustly improve metacognition. Added a smaller follow-up objective:
`--noise-warmup-uniform-weight`, which trains random-token inputs toward
high-entropy logits instead of a random target token.

Code:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  random_noise_warmup_loss(..., uniform_weight=...)
  --noise-warmup-uniform-weight

scripts/197_run_pure_recursive_depth_supervised_train.sh
  NOISE_WARMUP_UNIFORM_WEIGHT

tests/test_pure_recursive_depth_supervised_train_script.py
  overconfident-noise-logit penalty
  runner argument plumbing
```

Smoke checkpoint:

```text
runs/qwen35_2b_4090_noise_uniform_warmup_smoke_s001/last.pt
```

Matched 40-case metacognitive result:

```text
eval:
  runs/eval/metacognitive_calibration_noise_uniform_warmup_s001_40.jsonl
report:
  docs/wiki/decisions/noise-uniform-warmup-metacognitive-calibration-heldout40-s001.md

accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.332757  delta -0.001182
Brier:                     0.286778 -> 0.286715  delta -0.000063
avg confidence when wrong: 0.635804 -> 0.634379  delta -0.001426
status: rejected
```

Why rejected:
the global average improved, but the stricter gate now checks critical QTRM
modes separately. `qtrm_core_steps_8_no_evidence` and
`qtrm_core_steps_8_qtrm_only_no_evidence` both worsened on ECE and Brier. This
means the apparent gain is not yet a clean QTRM-core metacognitive gain.

Next:
build a direct forced-choice calibration training objective on known/unknown,
contradiction, and OOD rows, then require no worsening in core-on/QTRM-only
critical modes.

## [2026-05-03] evaluation | 40-case random-noise metacognitive gate rejected

Added a held-out metacognitive calibration dataset and regenerated the matched
no-warmup versus random-noise-warmup gate with category breakdowns.

Code and data:

```text
scripts/203_build_metacognitive_calibration_cases.py
scripts/192_eval_raw_intelligence.py
scripts/202_build_metacognitive_calibration_gate.py
tests/test_metacognitive_calibration_cases.py
tests/test_raw_intelligence_eval_script.py
tests/test_metacognitive_calibration_gate_script.py
data/eval/metacognitive_calibration_heldout_40.jsonl
```

Compared checkpoints:

```text
baseline checkpoint:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
baseline eval:
  runs/eval/metacognitive_calibration_no_warmup_s001_40.jsonl

candidate checkpoint:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
candidate eval:
  runs/eval/metacognitive_calibration_noise_warmup_s001_40.jsonl
```

Result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.337422  delta +0.003483
Brier:                     0.286778 -> 0.286384  delta -0.000394
avg confidence when wrong: 0.635804 -> 0.635978  delta +0.000174
status: rejected
failed check: candidate_ece_worse
```

Interpretation:
the earlier 1-case result remains only a smoke. The 40-case gate rejects the
current two-step warm-up because global ECE and wrong-answer confidence worsen.
UNKNOWN/OOD slices improve slightly, but answerable boolean rows degrade. Next
step is a stronger QTRM-side calibration objective, not promotion of this
checkpoint.

Report:

```text
docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001.md
docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001-summary.json
```

## [2026-05-03] training | Random-noise warm-up option added for QTRM-only calibration probes

Implemented a QTRM-side adaptation of the Random2 / Nature NMI 2026 random
noise calibration prior.

Reference code cloned:

```text
references/official/cogilab-random2 798a76cb1d98
references/official/cogilab-random  e96aadd13903
```

Code:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  build_random_noise_warmup_batch()
  random_noise_warmup_loss()
  --noise-warmup-* arguments

scripts/197_run_pure_recursive_depth_supervised_train.sh
  NOISE_WARMUP_* environment variables

tests/test_pure_recursive_depth_supervised_train_script.py
```

Smoke:

```text
command shape:
  NOISE_WARMUP_STEPS=2
  NOISE_WARMUP_SEQ_LEN=8
  NOISE_WARMUP_TARGET_VOCAB_SIZE=128
  STEPS=1
  MAX_CASES=1

checkpoint:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt

summary:
  docs/wiki/decisions/qwen35-2b-noise-warmup-smoke-s001-depth-gate-1-summary.json

raw-depth smoke:
  donor 0/1
  core_off 0/1
  core8 1/1
  status accepted
```

Interpretation:
this is a plumbing acceptance only. It proves the warm-up option runs while the
Qwen donor remains frozen and the standard raw-depth harness still executes. It
does not prove that random-noise warm-up improves QTRM calibration or reasoning.
The next gate must train matched no-warmup vs warmup checkpoints and compare
ECE/Brier/UNKNOWN/OOD/selective accuracy under donor-only, fused, low-donor,
QTRM-only, and core-off modes.

## [2026-05-03] evaluation | Matched random-noise metacognitive smoke gate

Added a choice-score calibration gate:

```text
scripts/202_build_metacognitive_calibration_gate.py
tests/test_metacognitive_calibration_gate_script.py
```

The gate reads raw forced-choice eval JSONL records, converts `choice_scores`
to softmax confidence, then computes:

```text
accuracy
mean confidence
ECE
Brier
average confidence when wrong
per-mode deltas
```

Matched smoke:

```text
baseline:
  checkpoint: runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
  eval: runs/eval/qwen35_2b_noise_warmup_matched_no_warmup_s001_depth_gate_1.jsonl

candidate:
  checkpoint: runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
  eval: runs/eval/qwen35_2b_noise_warmup_smoke_s001_depth_gate_1.jsonl

report:
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001.md
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001-summary.json
```

Result:

```text
accuracy: 0.500000 -> 0.500000
ECE:      0.180124 -> 0.176691  delta -0.003433
Brier:    0.247725 -> 0.243216  delta -0.004509
status: accepted
```

Interpretation:
accepted as smoke/evaluator plumbing only. This does not prove robust
metacognition. It proves the harness can detect matched calibration changes
without MemoryOS/retrieval and without making Qwen donor trainable. Next
required step: larger held-out metacognitive dataset with answerable,
UNKNOWN, contradiction, OOD/random-token, and core-off/low-donor comparisons.

## [2026-05-03] architecture | Qwen donor boundary and metacognition risk documented

Decision:
using a frozen Qwen donor is not inherently a problem. It is valid as tokenizer
contract, hidden-state provider, base language-policy baseline, and bounded
residual scaffold. It becomes a problem when donor fluency, donor confidence, or
fused-logit quality is mistaken for QTRM raw intelligence.

Updated:

```text
docs/wiki/sources/random-noise-calibration.md
docs/wiki/decisions/qwen-donor-risk-and-metacognition.md
docs/wiki/index.md
```

New canonical rule:

```text
QTRM claims require donor_only, qtrm_fused, qtrm_only/low_donor, core_off, and
memory_off comparisons. Metacognitive calibration claims additionally require
ECE/Brier/UNKNOWN/OOD/selective-accuracy gates and must show causal drop under
the claimed QTRM component ablation.
```

Random-noise warm-up is now mapped as a QTRM-trainable-path calibration probe:
freeze Qwen donor, warm up only QTRM modules, then measure QTRM-only and fused
calibration before/after. It is not proof of donor-free language ability.

## [2026-05-03] distillation | Subliminal Learning changes Qwen3.6 teacher policy

Reviewed:

```text
paper: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
arXiv: https://arxiv.org/abs/2507.14805
local PDF: references/papers/donor_annealing/2507.14805.pdf
```

Decision:
direct Qwen3.6-to-QTRM teacher-answer imitation is no longer canonical.
Teacher-generated data can carry hidden behavioral signals even when the
surface text is unrelated to the trait and filtered. The risk is especially
relevant for Qwen-family teacher/donor/student setups.

Canonical replacement:

```text
teacher proposes candidates / critiques / hard negatives
-> verifier or gold process decides labels
-> QTRM trains on verified labels and explicit checked preferences
```

Updated:

```text
docs/wiki/sources/donor-annealing-distillation.md
docs/wiki/concepts/donor-annealing-roadmap.md
docs/wiki/decisions/qwen36-online-distillation-roadmap.md
docs/wiki/concepts/cognitive-core-data-quality.md
```

Next data direction:
prefer verified public datasets first: GSM8K, MATH-500, NuminaMath verifiable,
OpenR1 verified answer-only subsets, ProofWriter, CLUTRR, bAbI, MBPP, MBPP+,
and HumanEval. Qwen3.6/GPT-5.5 can still help generate candidates or hard
cases, but not unverified gold labels.

## [2026-05-02] architecture | Raw intelligence gates become first priority

Correction:
the strict mandatory-core 72-case result proves a causal answer path on the
current MemoryOS-style gate, but it is not proof of raw intelligence. ASI
progress now requires no-retrieval recursive-depth and trainable-memory
ablation gates.

Implemented:

```text
src/qtrm_mm/eval/raw_intelligence_gate.py
scripts/190_build_pure_recursive_reasoning_cases.py
scripts/191_build_raw_intelligence_gate.py
scripts/192_eval_raw_intelligence.py
scripts/193_run_pure_recursive_reasoning_depth_gate.sh
data/eval/pure_recursive_reasoning_heldout_72.jsonl
docs/wiki/decisions/raw-intelligence-gates.md
```

Canonical first gate:

```text
donor_only_no_evidence
qtrm_core_off_no_evidence
qtrm_core_steps_1_no_evidence
qtrm_core_steps_2_no_evidence
qtrm_core_steps_4_no_evidence
qtrm_core_steps_8_no_evidence
```

Acceptance requires deep core to beat donor and core-off, plus a positive
depth-scaling gain, with zero MemoryOS/retrieval/hidden-evidence shortcut
records.

First smoke:

```text
report: docs/wiki/decisions/pure-recursive-reasoning-depth-gate-smoke4.md
status: rejected
donor/core_off/core_steps_1/2/4/8: all 1/4
shortcut records: 0
```

This is the correct failure to expose. It means the current mandatory-core
checkpoint is not yet a raw recursive-reasoning model. The next canonical work
is raw-core training or redesign with a core-off causal loss, not more
MemoryOS/RAG/answer-format tuning.

Implemented next training path:

```text
config: configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml
runner: scripts/195_run_pure_recursive_reasoning_core_train.sh
train cases: data/filtered/pure_recursive_reasoning_train256_cases.jsonl
preference rows: data/filtered/pure_recursive_reasoning_preferences_train.jsonl
preference rows count: 640
canonical causal ablation: core_off
```

S160 result:

```text
checkpoint: runs/qwen35_2b_4090_pure_recursive_reasoning_core_s160/last.pt
report: docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md
failure ledger: docs/wiki/decisions/pure-recursive-reasoning-core-s160-failure-ledger.md
status: rejected
donor/core_off/core_steps_1/2/4/8: all 3/8
```

The core path changed some answers, so it is no longer completely inert.
However, depth 1/2/4/8 produced identical scores and outputs. The next
architecture change must supervise depth-progressive state updates; another
answer-only SFT pass is not canonical progress.

## [2026-05-02] architecture | Full-sequence KISS raw-core probes reject current recursive core

Implemented fixes for the raw recursive-depth path:

```text
src/qtrm_mm/qtrm_model.py: final logits now use qtrm_residual_logits without donor logits
scripts/196_train_pure_recursive_depth_supervised.py: full answer-token CE
scripts/196_train_pure_recursive_depth_supervised.py: Cartesian row/depth schedule
scripts/197_run_pure_recursive_depth_supervised_train.sh: TARGET_MODE switch
src/qtrm_mm/training/train.py: core_only trainable policy
configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
```

Result:

```text
failure ledger: docs/wiki/decisions/pure-recursive-depth-fullseq-kiss-failure-ledger.md
KISS plain-coda train8: core_off 7/8, core8 7/8
core_only train8: core_off 2/8, core8 3/8
heldout: no run beats donor and core_off with positive depth scaling
status: rejected
```

Interpretation:
the corrected KISS path proves the prior setup had real SSOT/KISS violations
and training bugs. It also falsifies the stronger claim that the current
recursive core is already a useful raw reasoning engine. Non-core paths can
memorize the tiny train set; the core-only path cannot yet beat donor or show
depth-scaled reasoning.

## [2026-05-02] architecture | Strict mandatory-core answer path passes first 8-case gate

The previous mandatory identity-safe core still allowed the answer path to
behave like `core_off`. The corrected strict path makes the answer residual
require a running core:

```text
canonical prompt tokens
-> frozen donor hidden/logits
-> QTRM workspace
-> mandatory recursive core
-> core-conditioned answer bottleneck
-> QTRM residual logits + donor logits
-> greedy answer
```

New config:

```text
QTRMConfig.answer_bottleneck_requires_core
```

When `answer_bottleneck_requires_core: true`, `core_off` and `workspace_off`
cannot use the answer-bottleneck residual. They fall back to donor-fused output
instead of silently taking a coreless residual shortcut.

Failure steps that led here:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_causal_s080.yaml
report: docs/wiki/decisions/mandatory-identity-core-causal-s080-gate-8.md
status: rejected
reason: core_off same completions 8/8

config: configs/qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120.yaml
report: docs/wiki/decisions/mandatory-core-answer-bottleneck-causal-s120-gate-8.md
status: rejected
reason: bypass removed, but full QTRM still tied donor-only at 5/8
```

The final fine-tune used cleaned intervention preferences:

```text
builder: scripts/186_build_clean_intervention_preferences.py
data: data/filtered/memory_reasoning_intervention_preferences_clean_train24.jsonl
config: configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
script: scripts/187_run_mandatory_core_intervention_preference_train.sh
checkpoint: runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt
```

8-case result:

```text
report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8.md
status: accepted
causal_gate_status: accepted
full mandatory core: 6/8
donor-only: 5/8
core_off: 5/8
workspace_off: 5/8
same completion rate core_off/workspace_off: 3/8
```

Interpretation:
this is the first accepted small gate for a mandatory-core answer path. It is
not proof of a solved architecture or ASI. It proves that the current candidate
beats donor-only on the 8-case held-out slice and loses that advantage when
core/workspace are disabled. The next required check is 16/32/72 held-out
scale-up plus a failure ledger for the remaining misses.

Scale-up result:

```text
16-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-16.md
16-case full mandatory core: 13/16
16-case donor/core_off/workspace_off: 10/16

32-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-32.md
32-case full mandatory core: 20/32
32-case donor/core_off/workspace_off: 17/32

72-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-72.md
72-case full mandatory core: 50/72
72-case donor/core_off/workspace_off: 39/72
72-case core/workspace hit drop: 11
```

This upgrades the candidate from "8-case smoke accepted" to "72-case held-out
root gate accepted" for the strict mandatory-core answer path. Remaining
failure classes are not architectural optional-core failures anymore; they are
mostly abstention and temporal/conflict calibration failures.

## [2026-05-02] architecture | Mandatory identity-safe core replaces coreless as target

Correction:
the latent reasoning loop is the core QTRM claim. `coreless` must not be the
final architecture. It is only a lower-bound/teacher baseline showing that the
old recursive core was harmful when connected directly to answer logits.

Implemented a mandatory, identity-safe core path:

```text
workspace
-> recursive core always runs
-> learned output blend starts near identity(workspace)
-> answer bottleneck / residual governor
-> greedy answer
```

New fields:

```text
QTRMConfig.core_output_blend_enabled
QTRMConfig.core_output_blend_init_bias
QTRMConfig.core_output_blend_min
```

Config and script:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml
script: scripts/183_run_mandatory_identity_core_candidate_gate.sh
checkpoint: runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt
```

8-case result:

```text
report: docs/wiki/decisions/mandatory-identity-core-candidate-gate-8.md
status: accepted by current broad gate
full mandatory core: 6/8
donor-only: 5/8
core_off: 6/8
workspace_off: 5/8
core_steps: 2 for full path
```

Interpretation:
this is a safe mandatory-core starting point, not proof that the core has
learned useful reasoning yet. `core_off` still matches full output, which is
expected from the identity blend initialization. The next training goal is to
open the core blend only when the loop improves over the lower-bound coreless
teacher.

## [2026-05-02] architecture | Coreless workspace-answer lower-bound passes 8/16-case active gates

After the full intervention-preference checkpoint failed, the `core_off`
ablation exposed a lower-bound/teacher path:

```text
donor + SSOT prompt tokens
-> QTRM workspace answer bottleneck
-> answer residual governor
-> donor-fused greedy answer
```

The recursive core is disabled for this candidate:

```text
config: configs/qwen35_2b_4090_coreless_workspace_answer_candidate.yaml
script: scripts/182_run_coreless_workspace_answer_candidate_gate.sh
checkpoint: runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt
```

New config support:

```text
QTRMConfig.core_enabled: false
```

This makes the ordinary `qtrm_residual_with_evidence` path behave like the
previous `core_off` ablation, without needing a runtime ablation flag.

Results:

```text
8-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-8.md
8-case status: accepted
8-case QTRM coreless: 6/8
8-case donor-only: 5/8
8-case workspace_off: 5/8

16-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-16.md
16-case status: accepted
16-case QTRM coreless: 11/16
16-case donor-only: 10/16
16-case workspace_off: 10/16
```

Interpretation:
this is not the final QTRM architecture because it removes the latent loop. It
is a lower-bound/teacher baseline for training the mandatory identity-safe core:
the mandatory core must first match this behavior without harm, then beat it
with a core-off causal drop.

## [2026-05-02] architecture | On-policy intervention preference rejected; root path still non-causal

Generated train-split on-policy intervention data from the rejected preference
checkpoint:

```text
train eval: runs/eval/canonical_answer_preference_s160_train24_answer_gate.jsonl
audit: docs/wiki/decisions/canonical-answer-preference-s160-train24-intervention-audit.json
data: data/filtered/memory_reasoning_intervention_preferences_train24.jsonl
rows: 6
preserve_donor: 2
allow_qtrm: 2
suppress_core_override: 2
```

Added:

- `configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml`
- `scripts/181_run_intervention_preference_train.sh`
- `tests/test_intervention_preference_train_script.py`

Held-out result:

```text
report: docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-8.md
active-path report: docs/wiki/decisions/intervention-preference-train24-s080-active-path-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only: 5/8
core_off: 6/8
answer_residual_governor_off: 6/8
intervention audit: docs/wiki/decisions/intervention-preference-train24-s080-intervention-audit.json
donor_hit_qtrm_miss: 0/8
qtrm_hit_donor_miss: 0/8
core_off_beats_qtrm: 1/8
```

Important mixed result:
the old `synthetic-authority-vault-0100` failure improved from
`stone-arch` to `opal-river`, so intervention preference can repair some
unsafe overrides. It still fails the root architecture gate because critical
component-off paths match or beat full QTRM. The next step should be a root
path change, not another local margin/scale/loss tweak.

Evaluation-tool correction:
`scripts/148_build_root_architecture_gate.py` now accepts repeated
`--critical-mode` and `--comparison-mode` overrides. The broad gate still
detects non-causal probe-only paths, while the active-path gate checks only the
components that should matter for the current SSOT answer path. The active-path
gate also rejected, so the conclusion does not depend on inactive probe modes.

## [2026-05-02] architecture | Donor-preserve and preference fixes rejected by strict promotion gate

Tested two donor-preserving follow-ups to the answer residual governor:

- `configs/qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml`
- `configs/qwen35_2b_4090_canonical_answer_preference_s160.yaml`
- `scripts/176_run_canonical_answer_governor_preserve_train.sh`
- `scripts/177_build_canonical_plain_answer_preferences.py`
- `scripts/178_run_canonical_answer_preference_train.sh`

The preference run originally OOMed because it combined rejected-sample
preference forwards with canonical causal ablation forwards. The preference
config now keeps canonical causal ablation loss off during training and relies
on the strict held-out gate after training:

```text
loss_canonical_causal_weight: 0.0
canonical_causal_ablation_modes: []
```

Results:

```text
preserve report: docs/wiki/decisions/canonical-answer-governor-preserve-s120-answer-gate-8.md
preserve status: rejected
preserve full QTRM: 5/8
preserve donor-only: 5/8
preserve core_off: 5/8

preference report: docs/wiki/decisions/canonical-answer-preference-s160-answer-gate-8.md
preference status: rejected
preference causal_gate_status: accepted
preference full QTRM: 5/8
preference donor-only: 5/8
preference answer_residual_governor_off: 4/8
preference core_off: 6/8
```

Interpretation:
donor-correct preservation removed one earlier `core_off > full` symptom in the
preserve run, but did not make QTRM beat donor-only. Preference training did
not fix the key authority-conflict failure: on
`synthetic-authority-vault-0100`, donor/core-off answer `opal-river`, while
full QTRM still answers `stone-arch`. The next step should not be another
local loss. The latent core must be redesigned or gated so it cannot override
donor-correct answers unless it has an independently verified advantage.

Follow-up scale sweep:

```text
report: docs/wiki/decisions/canonical-answer-preference-s160-qscale030-answer-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only: 5/8
causal modes: none
```

`QTRM_LOGITS_SCALE=0.30` fixed the `opal-river` override case, but removed the
causal governor drop and hurt abstention behavior. This rejects the simpler
"just lower residual scale" explanation. The failure is intervention policy:
QTRM needs a learned, causal permit/verify path for when it may override donor,
not a global residual scale.

Added:

- `scripts/179_audit_intervention_policy.py`
- `scripts/180_build_intervention_preferences.py`
- `docs/wiki/decisions/intervention-policy-failure-ledger.md`

The diagnostic builder produced 3 held-out correction rows
(`allow_qtrm=1`, `preserve_donor=1`, `suppress_core_override=1`). These rows
are not for final training claims; they define the on-policy correction shape
that should be generated on a train split next.

## [2026-05-02] architecture | Answer residual governor causal signal accepted, strict promotion rejected

Added a donor-preserving answer residual governor:

- `QTRMConfig.answer_residual_governor_enabled`
- `QTRMConfig.answer_residual_governor_init_bias`
- `TrainConfig.loss_answer_residual_governor_weight`
- `src/qtrm_mm/losses.py::answer_residual_governor_loss`
- `qtrm_answer_residual_governor_off_with_evidence` ablation mode
- `configs/qwen35_2b_4090_canonical_answer_governor_s120.yaml`
- `scripts/175_run_canonical_answer_governor_train.sh`

Also tightened the root gate:

```text
causal_gate_status = component changed answer-level behavior
status             = strict promotion result
```

Strict promotion now requires:

```text
full QTRM > donor-only
no critical component-off mode > full QTRM
at least one critical component-off causal drop
```

Result:

```text
report: docs/wiki/decisions/canonical-answer-governor-s120-answer-gate-8.md
status: rejected
causal_gate_status: accepted
full QTRM: 5/8
donor-only with evidence: 5/8
answer_residual_governor_off: 4/8
core_off: 6/8
```

Interpretation:
the governor is a real causal/protective component on this split, but the full
architecture is not promoted. QTRM still ties donor-only, and disabling the
latent core improves the hit count. The next candidate must make the latent core
help rather than merely letting the governor limit damage.

## [2026-05-02] architecture | Plain-answer KISS reset rejected

Added:

- `scripts/173_build_canonical_plain_answer_data.py`
- `scripts/174_run_canonical_plain_answer_kiss_train.sh`
- `configs/qwen35_2b_4090_canonical_plain_answer_kiss_s120.yaml`
- `tests/test_canonical_plain_answer_data.py`
- `tests/test_canonical_plain_answer_train_script.py`
- `docs/wiki/decisions/canonical-greedy-margin-and-plain-answer-kiss-result.md`

Result:

```text
report: docs/wiki/decisions/canonical-plain-answer-kiss-s120-answer-gate-8.md
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 4/8
qtrm_workspace_off_with_evidence: 5/8
qtrm_core_off_with_evidence: 5/8
causal modes: none
```

Interpretation:
matching the training rows to the canonical plain `Answer:` contract reduced
format leakage, but it did not make the latent workspace/core path causal.
Workspace/core-off ablations still matched or improved full QTRM. This rejects
the idea that another local loss tweak on the current donor-fused residual path
is enough.

## [2026-05-02] architecture | Greedy-token margin rejected on canonical gate

Added greedy-token margin pressure to the training loss:

- `src/qtrm_mm/losses.py::greedy_token_margin_loss`
- `TrainConfig.loss_greedy_token_margin_weight`
- `TrainConfig.greedy_token_margin`
- `TrainConfig.greedy_token_margin_only_donor_errors`
- `configs/qwen35_2b_4090_canonical_greedy_margin_s120.yaml`
- `scripts/172_run_canonical_greedy_margin_train.sh`
- `tests/test_greedy_token_margin_train_script.py`

Result:

```text
report: docs/wiki/decisions/canonical-greedy-margin-s120-answer-gate-8.md
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 5/8
qtrm_core_to_text_off_with_evidence: 6/8
exact match: 0/8
causal modes: none
```

Interpretation:
the margin objective can pressure the target token against the top non-target
competitor, but the held-out answer gate still does not show latent causality.
The result also exposed a training/eval contract mismatch: decision-token rows
train `Verify/Decision/Answer`, while the canonical gate expects a short greedy
`Answer:` completion.

## [2026-05-02] architecture | KISS/YAGNI/DRY/SSOT contract enforced

Added:

- `src/qtrm_mm/eval/ssot_contract.py`
- `tests/test_ssot_contract.py`
- `docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md`

Implementation:
the canonical answer-path constants and validation now live in one module:

```text
CANONICAL_EVIDENCE_INJECTION = "ssot"
CANONICAL_ANSWER_CHANNEL = "greedy"
```

`scripts/95_eval_memory_retrieval.py` imports that module instead of keeping
its own local validation logic.

Decision:
KISS/YAGNI/DRY/SSOT is now an enforceable engineering contract:

```text
compiled prompt tokens -> donor + QTRM -> greedy autoregressive answer
```

Span-copy, workspace-only evidence, dual evidence, source masks, and post-hoc
answer decision paths remain probe-only until they pass held-out causal
ablation gates.

## [2026-05-02] architecture | Canonical SSoT answer gate added

Added a hard contract for user-facing QTRM answer evaluation:

```text
--require-canonical-ssot
--evidence-injection ssot
--answer-channel greedy
```

Implementation:

- `scripts/95_eval_memory_retrieval.py` now has
  `--require-canonical-ssot` and rejects workspace/dual hidden-evidence or
  span-copy answer channels when that flag is set.
- `scripts/166_run_canonical_ssot_answer_gate.sh` runs the canonical
  autoregressive answer gate against donor-only, full-QTRM, core-off,
  workspace-off, evidence-bottleneck-off, and no-evidence baselines.
- `scripts/153_run_reasoning_safe_span_copy_gate.sh` is explicitly labeled
  `PROBE-ONLY` and now defaults to `EVIDENCE_INJECTION=ssot`.

Decision:
span-copy, hidden workspace evidence, and dual evidence paths remain diagnostic
tools only. They are not accepted as evidence that the main QTRM architecture
can answer as a general LLM. The main gate is now compiled prompt tokens into
donor + QTRM, followed by greedy autoregressive generation.

Smoke result:

- command: `MAX_CASES=2 bash scripts/166_run_canonical_ssot_answer_gate.sh`
  with explicit output paths under `runs/eval/canonical_ssot_answer_gate_smoke2.*`;
- report: `docs/wiki/decisions/canonical-ssot-answer-gate-smoke2.md`;
- summary: `docs/wiki/decisions/canonical-ssot-answer-gate-smoke2-summary.json`;
- status: `rejected`;
- full QTRM accuracy: `1/2 = 0.500`;
- donor-only with evidence accuracy: `1/2 = 0.500`;
- all critical ablations had `hit_drop=0` and `same_completion_rate=1.000`;
- missing critical modes: `none`.

Interpretation:
the canonical SSoT greedy path now gives a sharper failure signal. The current
checkpoint can use visible evidence as donor/prompt context, but the latent
workspace/core/evidence paths are not causally changing the answer in this
smoke. Next work should train the greedy answer path under answer-level causal
pressure instead of relying on span-copy extraction.

## [2026-05-02] eval | Model-only language stability sweep recorded

Ran the first ASI sufficiency sanity gate without MemoryOS:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
OUT_DIR=runs/language_stability/asi_sufficiency_20260502 \
bash scripts/152_run_residual_language_stability_sweep.sh
```

Artifacts:

- `runs/language_stability/asi_sufficiency_20260502/summary.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p0.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p05.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p10.jsonl`
- `docs/wiki/decisions/asi-sufficiency-gate-2026-05-02.md`

Result:
scales 0.00, 0.05, and 0.10 all had `clean_rate=1.0`,
`repeat_failure_rate=0.0`, `visible_reasoning_rate=0.0`, and
`answer_drift_rate=0.0` over 4 plain prompts.

Decision:
this provisionally accepts the model-only gate only as a narrow repetition and
format sanity check. It does not prove reasoning. The arithmetic sample still
answered incorrectly (`x = 7 - 4 = x + 1` instead of `x = 4`), so the next gate
must measure answer-level correctness and causal latent contribution against
donor-only and component-off ablations.

## [2026-05-02] decision | ASI sufficiency rejected, next gates ranked

Added:

- `docs/wiki/decisions/asi-sufficiency-gate-2026-05-02.md`

Decision:
the current QTRM architecture is not sufficient for a general ASI claim. The
MemoryOS/model boundary is now cleaner, but ASI-oriented progress still needs
separate gates:

1. model-only plain-prompt language competence;
2. causal latent-answer improvement over donor-only and component-off baselines;
3. verified self-improvement runtime with regression protection;
4. long-horizon task-level reward, not only action imitation.

Immediate priority:
run the residual language stability sweep, then SSOT answer ablations, before
adding another large architecture component.

## [2026-05-02] architecture | MemoryOS moved outside model boundary

Corrected the architecture claim boundary:

```text
QTRM model architecture:
canonical token stream -> Qwen donor -> QTRM workspace/core/residual -> answer

QTRM runtime with MemoryOS:
user prompt -> optional retrieval/rerank -> context compiler
-> canonical token stream -> QTRM model
```

Added:

- `docs/wiki/architecture/model-vs-runtime-boundary.md`

Updated:

- `docs/wiki/architecture/qtrm-forward-pass.md`
- `docs/wiki/architecture/canonical-architecture-matrix.md`
- `docs/wiki/architecture/paper-diagram-prompts.md`
- `docs/wiki/concepts/qtrm-terminology.md`
- `docs/wiki/concepts/workspace-memory-architecture.md`
- `docs/wiki/concepts/workspace-evidence-path.md`
- `docs/wiki/decisions/qtrm-goal-and-scope.md`

Boundary rule:
MemoryOS is optional external memory/RAG/runtime infrastructure. It can prepare
the canonical prompt, but it is not an internal QTRM model component. The base
QTRM model must still run on plain prompts without MemoryOS.

## [2026-05-02] architecture | MemoryOS evidence path corrected to SSOT

Corrected the QTRM MemoryOS architecture from a misleading two-path drawing to
a single-source-of-truth evidence path.

```text
User prompt
-> retrieval query
-> MemoryOS retrieval/rerank
-> Context Compiler / chat-template builder
-> one canonical donor-visible token stream
-> Frozen Qwen donor + QTRM core
-> answer channel
```

Implementation changes:

- `scripts/95_eval_memory_retrieval.py` now defaults to
  `--evidence-injection ssot`.
- SSOT span-copy sets `evidence_span_reader_context="input"`, so the evidence
  span reader scores canonical prompt tokens instead of requiring a separate
  workspace evidence text.
- Learned source masks now annotate token spans in the canonical prompt and do
  not include the user prompt after the final evidence source.
- `workspace` and `dual` remain available only as ablation/probe paths.

Claim boundary:
MemoryOS retrieval is not a second model input reality. It is pre-forward
context engineering that must compile back into one chat-template token stream.

## [2026-05-02] experiment | Learned source span-mask accepted

Trained a learned answer-source selector and applied it as a span-logit mask
while preserving the full workspace evidence context.

```text
report: docs/wiki/decisions/evidence-source-selector-truthcal72.md
selector checkpoint: runs/evidence_source_selector_truthcal_s500/selector.pt
selector summary: docs/wiki/decisions/evidence-source-selector-truthcal72-summary.json
answer records: docs/wiki/decisions/evidence-source-selector-span-mask-truthcal72-thr065-records.jsonl
train source selector: 504 / 504 source records, case success 1.0000
heldout source selector: 252 / 252 source records, case success 1.0000
answer accuracy: 71 / 72 = 0.9861
unknown negatives: 24 / 24
gate: accepted learned source span-mask
```

Important correction:
the accepted mechanism is not evidence pruning. The model still receives full
workspace evidence. The learned selector only masks the final span-copy argmax
to answer-bearing source text. This avoids the distribution break that made the
reliability source governor fall to 48/72.

Remaining miss:
`synthetic-authority-vault-0102` selects the correct source but the span reader
returns `no_answer` with high no-answer probability. Next step is no-answer
head recalibration under source-masked decoding.

## [2026-05-02] experiment | Boundary REVISE accepted, source pruning rejected

Added a deterministic answer-renderer `REVISE` branch for evidence span-copy
answers that were cut inside atomic identifiers.

```text
report: docs/wiki/decisions/evidence-span-boundary-revise-truthcal72.md
accepted records: docs/wiki/decisions/evidence-span-truthcal-72-boundary-revision-ablation-records.jsonl
full in-model + boundary revise: 67 / 72 = 0.9306
feature-off: 55 / 72 = 0.7639
decision-head-off: 55 / 72 = 0.7639
revised spans: 6
span-boundary fixes: 5
negative UNKNOWN retained: 24 / 24
gate: accepted narrow renderer REVISE
```

Also tested a non-label reliability source governor:

```text
records: docs/wiki/decisions/evidence-span-truthcal-72-source-governor-records.jsonl
full mode: 48 / 72 = 0.6667
gate: rejected
```

Important correction:
source pruning before answer selection is harmful with the current calibrated
answer-decision path. The next source-selection step should be learned or
distilled and should preserve full evidence unless calibrated confidence is
high.

## [2026-05-02] experiment | In-model answer-decision head accepted

Moved the answer-decision gate into an ablatable QTRM checkpoint path.

```text
report: docs/wiki/decisions/inmodel-answer-decision-head-truthcal-s200.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl
train mix: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
bootstrap script: scripts/164_bootstrap_answer_decision_feature_head.py
full in-model: 62 / 72 = 0.8611
feature-off: 49 / 72 = 0.6806
decision-head-off: 49 / 72 = 0.6806
blocked candidates: 14
block improved: 13
block harmed: 0
remaining expected-unknown false positives: 0
gate: accepted in-model causal gate
```

Important correction:
hidden-only and linear telemetry variants were insufficient. The accepted
variant uses raw answer-channel telemetry through a feature MLP in the QTRM
checkpoint and keeps feature-off/head-off ablations to prove causality. The
next failure class is positive wrong answers, so the next architecture branch
should be `REVISE` or `SEARCH_MORE`, not more UNKNOWN blocking.

## [2026-05-01] experiment | Action loop fixed, answer bottleneck identified

Trained an action-first strict runtime transition-state controller and reran
the task-level answer gate.

Action-policy result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_actionfirst_s200/last.pt
summary: docs/wiki/decisions/transition-state-controller-runtime-actionfirst-s200-summary.json
train sequences: 256
state_loss_weight: 0.2
held-out action accuracy: 1.0000
RETRIEVE_MEMORY: 72 / 72
VERIFY_EVIDENCE: 72 / 72
ANSWER: 72 / 72
zero_transition_state: 0.3333
transition_state_drop: 0.6667
state_prediction_binary_accuracy: 0.9913
gate: accepted
```

Task-level answer result:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-actionfirst-gate.md
learned_state_qtrm: 4 / 8 = 0.5000
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 1.0000
gate: rejected
```

Interpretation:
the previous near-miss answer gain was partly an accidental UNKNOWN from an
action failure. Once action stability is fixed, the learned loop matches the
scripted answer baseline. The bottleneck is answer formation after
verification, not retrieval or action ordering.

Follow-up answer-channel probe:

```text
config: configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml
checkpoint: runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
answer_channel: evidence_span_copy
truth_gate: true
qtrm_residual_with_evidence: 49 / 72 = 0.6806
span_reader_off: 24 / 72 = 0.3333
donor_only: 24 / 72 = 0.3333
```

Conclusion:
the next architecture step is an answer-decision loop where verifier results
can choose `ANSWER`, `ABSTAIN`, `REVISE`, or `SEARCH_MORE`; another fixed
retrieve-verify-answer controller will not prove reasoning progress.

Static answer-decision threshold probe:

```text
script: scripts/160_calibrate_answer_decision_gate.py
report: docs/wiki/decisions/answer-decision-gate-truthcal-72.md
calibration baseline -> gated: 0.6111 -> 0.8056
heldout baseline -> gated: 0.7500 -> 0.4167
heldout false positives: 2 -> 0
heldout blocked positives: 16
gate: rejected
```

Conclusion:
a static threshold can remove false positives but over-abstains badly on
heldout. The next verifier must be trained with counterfactual positives and
negatives rather than tuned as a scalar threshold.

Learned answer-decision head:

```text
script: scripts/161_train_answer_decision_head.py
report: docs/wiki/decisions/answer-decision-head-truthcal-train144-eval72.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
train records: docs/wiki/decisions/evidence-span-truthcal-train144-answer-channel-records.jsonl
eval records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
train baseline -> learned: 0.7569 -> 0.9444
eval baseline -> learned: 0.6806 -> 0.8611
eval false positives: 13 -> 0
eval block improved: 13
eval block harmed: 0
gate: accepted
```

Conclusion:
the answer-decision signal is learnable. The next architecture step is to wire
this decision into the runtime answer loop and then move it from a post-hoc MLP
into an ablatable in-model QTRM head.

Runtime answer-decision integration:

```text
script: scripts/95_eval_memory_retrieval.py
report: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
decision checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
baseline span/truth: 49 / 72 = 0.6806
runtime decision: 62 / 72 = 0.8611
blocked candidates: 14
expected-unknown false positives: 0
gate: accepted runtime integration
```

Conclusion:
the answer-decision improvement survives the real eval path. The remaining
failure class is positive wrong answers in conflict or multi-hop cases, which
requires a `REVISE` or `SEARCH_MORE` action rather than more UNKNOWN blocking.

## [2026-05-01] experiment | Strict runtime learned-state answer-loop near-miss

Added a task-level gate for the learned transition-state controller. Unlike the
previous smoke, runtime rows hide trace-step and phase-specific state-summary
text. The controller prompt also hides evidence until `RETRIEVE_MEMORY` places
retrieved evidence into the previous-observation path.

Changed:

- `scripts/159_eval_learned_state_answer_loop.py`;
- `tests/test_learned_state_answer_loop_script.py`;
- `scripts/158_train_transition_state_controller.py --strict-runtime-state-inputs`;
- `docs/wiki/decisions/transition-state-controller-runtime-state-s120.md`;
- `docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md`.

Strict runtime-state controller result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_state_s120/last.pt
held-out action accuracy: 0.9630
RETRIEVE_MEMORY: 0.8889
VERIFY_EVIDENCE: 1.0000
ANSWER: 1.0000
zero_transition_state: 0.3333
transition_state_drop: 0.6296
state_prediction_binary_accuracy: 0.8287
gate: rejected
```

Task-level answer-loop gate:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md
learned_state_qtrm: 5 / 8 = 0.6250
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 7 / 8 = 0.8750
gate: rejected
failed_check: learned_action_loop_not_stable
```

Interpretation:
this is the first small task-level answer reward gain from the learned-state
loop, but it is not accepted. The next fix should target first-step action
stability under strict runtime inputs, not add another answer-format guard.

## [2026-05-01] experiment | Learned transition-state controller smoke

Added a learned transition-state predictor on top of QTRM row features. The
controller no longer receives `previous_action`, and its direct QTRM feature
path is disabled with `controller_feature_scale=0.0`, so action selection must
go through the predicted state vector.

Changed:

- `TransitionStatePredictor` and `transition_state_prediction_loss`;
- `scripts/158_train_transition_state_controller.py --learn-transition-state`;
- state-prediction gate requiring held-out binary accuracy >= `0.90`;
- `docs/wiki/decisions/transition-state-controller-learned-state-smoke.md`.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_learned_state_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-learned-state-smoke-summary.json
feature_scale: 1.0
controller_feature_scale: 0.0
learn_transition_state: true
use_prev_action: false
held-out eval_full: 1.0000
held-out state_prediction_binary_accuracy: 0.9974
zero_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this is the first learned-state loop smoke, but it is still a trace-phase task.
It does not prove task-level answer reward, factual verification, or a general
world model. The next gate must run the learned-state loop through answer
generation and compare against scripted/donor harnesses with state/world/verifier
ablations.

## [2026-05-01] experiment | Explicit observation/verifier transition-state smoke

Extended the transition controller so the action loop can read an explicit
state vector derived from the previous observation, reward, and previous
world/verifier signals. This test disables QTRM latent features
(`feature_scale=0.0`) and disables `previous_action`, so the loop cannot pass by
memorizing the fixed action order alone.

Changed:

- `TransitionStateController` now accepts `transition_state_dim` and supports
  transition-state ablations at inference time;
- `scripts/158_train_transition_state_controller.py` adds
  `--use-transition-state` and builds 9 inspectable state features;
- sequence grouping now uses `task_id + hash(prompt, workspace)` to preserve
  augmented variants;
- tests cover explicit state input, state collation, and variant grouping;
- `docs/wiki/decisions/transition-state-controller-explicit-state-smoke.md`.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_explicit_state_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-explicit-state-smoke-summary.json
feature_scale: 0.0
use_transition_state: true
use_prev_action: false
reset_hidden: true
held-out eval_full: 1.0000
zero_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this proves causal explicit observation/verifier-state wiring, not learned
world-model reasoning. The next gate must replace hand-built state features
with learned observation/world/verifier predictors and compare against this
scripted-state baseline.

## [2026-05-01] experiment | Explicit transition-state controller smoke

Added a sequence-level transition controller after the learned-signal collapse.
The key change is that action policy is no longer treated as independent
per-row classification. Trace rows are grouped by `task_id` and learned as a
transition sequence.

Added:

- `src/qtrm_mm/agentic/transition_controller.py`;
- `scripts/158_train_transition_state_controller.py`;
- `tests/test_transition_state_controller.py`;
- `return_features_only=True` in `QTRMMultimodalModel.forward`;
- `docs/wiki/decisions/transition-state-controller-markov-smoke.md`.

Important loader fix:
the signal trace file had duplicate `0,1,2` rows under the same `task_id`.
The transition sequence reader now deduplicates by `step`; otherwise the model
sees contradictory previous-action transitions.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_markov_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-markov-smoke-summary.json
controller_mode: explicit_markov_transition_state
feature_scale: 0.0
use_prev_action: true
reset_hidden: true
held-out eval_full: 1.0000
reset_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this proves only the explicit transition-state action loop, not QTRM latent
reasoning. `feature_scale=1.0` and hidden-only recurrent variants failed on
held-out, so QTRM feature grounding remains the next unsolved problem.

## [2026-05-01] experiment | Learned controller-signal replacement rejected

Replaced the oracle `controller_signal` scaffold with a learned
core-derived signal head and evaluated two variants:

```text
full learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_s300/last.pt
  held-out action accuracy: 0.3333
  collapse mode: predicts VERIFY_EVIDENCE for all rows
  gate status: rejected

head-only learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_head_s300/last.pt
  held-out action accuracy: 0.3333
  collapse mode: predicts ANSWER for all rows
  qtrm_controller_signal_off: 0.3333
  qtrm_verifier_off: 0.3333
  gate status: rejected

learned-readout diagnostic:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_readout_s300/last.pt
  held-out action accuracy: 0.3704
  collapse mode: mostly ANSWER, with 8/72 VERIFY_EVIDENCE correct
  qtrm_latent_core_off: 0.5926
  qtrm_workspace_off: 0.6296
  gate status: rejected
```

Interpretation:
the oracle signal path is causally useful, but a stateless per-row signal head
over the current latent core is not enough to replace it. A prompt/coda readout
diagnostic also fails, so the problem is not just `z_h` pooling. This is not a
language-model quality issue; it is an architecture issue. The next controller
design must learn a transition-state policy or recurrent planner over the trace,
not only a two-bit bottleneck readout from one frozen latent state.

## [2026-05-01] experiment | Controller-signal causal-loop scaffold

Added a Stage-1.5 controller-signal path so future learned world-model and
verifier outputs can affect the action controller through a measurable tensor
path.

Added:

- `controller_signal_enabled` and `controller_signal_dim` in `QTRMConfig`;
- `controller_signal_proj` in `QTRMModel`;
- `controller_signal` parsing and batching in `JsonlTextVisionDataset`;
- signal-conditioned trace replay rows in
  `scripts/155_build_controller_trace_replay.py`;
- signal-aware policy eval and ablation modes in
  `scripts/156_eval_controller_trace_policy.py` and
  `scripts/157_eval_asi_controller_causal_loop.py`;
- `configs/qwen35_2b_4090_controller_signal_s300.yaml`;
- `docs/wiki/decisions/asi-controller-signal-causal-loop-s300.md`.

Run:

```text
checkpoint: runs/qwen35_2b_4090_controller_signal_s300/last.pt
train rows: data/filtered/asi_controller_signal_trace_replay.jsonl
held-out rows: data/eval/asi_controller_signal_trace_replay_heldout_72.jsonl
final train action_policy loss: 0.1599
final train batch action_acc: 1.0000
held-out action accuracy: 0.9444
```

Causal-loop ablation:

```text
qtrm_harness: 0.9444
qtrm_world_model_off: 0.3333
qtrm_verifier_off: 0.6111
qtrm_controller_signal_off: 0.3333
qtrm_latent_core_off: 1.0000
standard gate status: rejected
```

Interpretation:
the new signal path is causal because zeroing the world-model/verifier signal
dimensions sharply reduces held-out action accuracy. This is still an oracle
scaffold, not a learned world-model/verifier proof. The ASI gate correctly
rejects because QTRM does not beat the scripted/donor harness baselines and the
latent core is bypassed rather than causally required.

## [2026-05-01] architecture | ASI root architecture reset

Reset the recommended direction away from free-form donor replacement and
toward a verified cognitive loop:

```text
typed context tape -> MemoryOS/search -> shared evidence SSoT
-> Qwen donor prior + QTRM recurrent residual controller
-> world-model prediction -> candidate action or answer
-> verifier gate -> answer/tool/memory write
-> trace store -> verified training buffer
```

Added:

- `src/qtrm_mm/agentic/cognitive_loop.py`;
- `src/qtrm_mm/agentic/context_tape.py`;
- `src/qtrm_mm/agentic/causal_gate.py`;
- `src/qtrm_mm/agentic/harness.py`;
- `src/qtrm_mm/agentic/trace_replay.py`;
- `tests/test_asi_cognitive_loop_contract.py`;
- `tests/test_asi_causal_loop_gate_script.py`;
- `scripts/154_build_asi_causal_loop_gate.py`;
- `docs/wiki/decisions/asi-root-architecture-reset.md`;
- new ASI roadmap PDFs for agent harness residual-role, autonomous agent
  memory, and autonomous memory agents.

Interpretation:
QTRM should be measured as a residual cognitive controller over donor and
harness baselines. Generated memory writes default to quarantine unless
grounding, contradiction, novelty, usefulness, and regression gates pass.
The Stage-0 scripted harness now records
`RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER` transitions so future QTRM
controller claims have a simple baseline to beat.
The harness now uses `TypedContextTape` as a single source of truth; prompt,
workspace, verifier input, and replay records share the same `context_hash`.
The ASI causal-loop gate rejects QTRM claims unless QTRM beats donor/scripted
harness baselines and the latent-core/world-model/verifier-off ablations drop.

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_asi_cognitive_loop_contract.py \
  tests/test_asi_causal_loop_gate_script.py -q
# 8 passed
```

## [2026-05-01] research | ASI-oriented literature roadmap

Reviewed current self-improvement, agent RL, world-model, RAG verification, and
evolutionary-code-discovery papers as architecture constraints rather than ASI
claims.

Added:

- `docs/wiki/decisions/asi-research-roadmap.md`;
- ASI wiki index link;
- ASI roadmap PDFs under `references/papers/asi_roadmap`;
- `docs/REFERENCE_BASELINE.md` entries for the new paper set;
- 2026 review delta for SEAL, RAGEN, Agent Lightning, AlphaEvolve/CodeEvolve,
  V-JEPA 2/LeWorldModel, and validated RAG write-back.

Boundary:

```text
ASI is the ambition, not the current claim.
Progress requires causal gates for evidence, latent core, world model,
verification, self-improvement, and long-horizon agency.
```

## [2026-05-01] implementation | Dual-path evidence conditioning

Corrected the evidence-routing architecture. Workspace-only evidence remains a
causality probe, but the practical RAG-compatible path now keeps retrieved
evidence in the visible prompt and also encodes the same evidence as workspace
memory.

Added:

- `--evidence-injection dual` in `scripts/95_eval_memory_retrieval.py`;
- `build_shared_evidence_context()` as the SSoT evidence materialization point;
- `build_case_prompt_and_workspace_memory(..., evidence_injection="dual")`;
- `workspace_evidence_injection_mode: dual` support in
  `JsonlTextVisionDataset`;
- `EVIDENCE_INJECTION="${EVIDENCE_INJECTION:-dual}"` in
  `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- wiki updates in `workspace-evidence-path.md` and the canonical architecture
  matrix.

Boundary:

```text
workspace = hidden-only causality probe
prompt    = ordinary visible RAG context
dual      = final architecture candidate: one shared evidence context
            -> visible RAG view + latent workspace view
```

Smoke run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=2 \
  OUT=runs/eval/reasoning_safe_span_copy_dual_smoke_2.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_dual_smoke_2_audit.jsonl \
  ROOT_MD=docs/wiki/decisions/reasoning-safe-span-copy-dual-smoke-2.md \
  ROOT_JSON=docs/wiki/decisions/reasoning-safe-span-copy-dual-smoke-2-summary.json \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Result: `qtrm_residual_with_evidence` was `2/2` on the two-case smoke; the root
gate was accepted. This only proves the dual path executes and preserves the
small span-copy behavior. It does not yet prove broad generation quality.

## [2026-05-01] implementation | Truth-gated evidence answer channel

Clarified the architecture boundary: Loop-RM/TRM-style recurrence is a latent
compute substrate, not truth judgment by itself. Truth-sensitive answering now
has an explicit runtime gate in the `evidence_span_copy` answer channel.

Added:

- `evidence_truth_gate_from_outputs()` in
  `scripts/95_eval_memory_retrieval.py`;
- `--truth-gate` plus support/causal/refute/missing thresholds;
- span-copy blocking with `answer_channel_meta.status=truth_gate_blocked`;
- optional `TRUTH_GATE=1` runner support in
  `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- concept note:
  `docs/wiki/concepts/truth-gated-evidence-answer-channel.md`.

Current interpretation:

```text
Loop/TRM core = repeated latent state update capacity
truth gate = explicit support/refute/missing/causal answer-path contract
```

The gate is disabled by default until calibration is proven on held-out
truth-gated evals.

Smoke eval:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src TRUTH_GATE=1 \
  MAX_CASES=4 \
  OUT=runs/eval/reasoning_safe_span_copy_truth_gate_4.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_truth_gate_4_audit.jsonl \
  ROOT_MD=docs/wiki/decisions/reasoning-safe-span-copy-truth-gate-4.md \
  ROOT_JSON=docs/wiki/decisions/reasoning-safe-span-copy-truth-gate-4-summary.json \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Result: `qtrm_residual_with_evidence` was `2/4`; the root gate was
`rejected`. Metadata shows the current head logits sit near `0.5`, so
`support_low` blocks valid multi-hop answers by tiny margins. Conclusion:
runtime wiring is correct, but the logical heads need explicit calibration
training before `TRUTH_GATE=1` should be the default.

## [2026-05-01] experiment | Reasoning-safe span-copy gate improves hidden-evidence QA

Implemented a confidence-gated `evidence_span_copy` answer channel so QTRM can
improve evidence-sensitive reasoning without touching donor language logits.

Added:

- `--evidence-span-min-score` to `scripts/95_eval_memory_retrieval.py`;
- low-span-score abstention: selected spans below the confidence floor return
  `Answer: UNKNOWN`;
- `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- tests for the CLI, low-score abstention, and runner defaults.

Run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT=runs/eval/reasoning_safe_span_copy_confidence_gate_16_runner_full.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_confidence_gate_16_runner_full_audit.jsonl \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Settings:

```text
checkpoint: runs/qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt
answer_channel: evidence_span_copy
evidence_injection: workspace
no_answer_threshold: 0.1
min_span_score: 12
max_cases: 16
```

Result:

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| `donor_only_with_evidence` | `5/16` | `0.3125` |
| `qtrm_residual_with_evidence` | `13/16` | `0.8125` |
| `qtrm_workspace_off_with_evidence` | `5/16` | `0.3125` |
| `qtrm_workspace_memory_off_with_evidence` | `5/16` | `0.3125` |
| `qtrm_evidence_span_reader_off_with_evidence` | `5/16` | `0.3125` |

Compared with the previous uncalibrated span-copy baseline,
`qtrm_residual_with_evidence` improved from `11/16` to `13/16` by abstaining on
low-confidence spans. Root gate:
`docs/wiki/decisions/reasoning-safe-span-copy-confidence-gate.md` reports
`accepted` with causal drops for workspace, workspace-memory, and evidence-span
reader ablations.

Interpretation:
this is a real reasoning-path improvement under the language-safe constraint:
the donor's surface language policy is not modified, and the answer is produced
through a bounded span-copy/UNKNOWN channel. It is not yet proof that the
recursive core is doing the reasoning. `core_off`, `core_context_off`, and
`evidence_bottleneck_off` matched the baseline, so the current causal path is
best described as:

```text
hidden workspace evidence -> prompt-conditioned evidence span reader
-> confidence/UNKNOWN gate -> short copied answer
```

## [2026-05-01] experiment | Residual language-stability sweep with donor preserved

Implemented and ran a repeatable residual language-stability sweep for the
current donor-backed residual candidate.

Added:

- `--donor-logits-scale`, `--qtrm-logits-scale`, and
  `--qtrm-residual-clamp` overrides to `scripts/92_eval_qtrm_logits.py`;
- `--stop-after-sentence` and `--min-new-tokens-before-stop` to the same eval
  path, matching the interactive language-safe guard;
- `scripts/152_run_residual_language_stability_sweep.sh`;
- tests for the new CLI controls and sweep runner.

Run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT_DIR=runs/language_stability/residual_s010_safe_sweep_96 \
  SCALES="0.0 0.05 0.10" MAX_NEW_TOKENS=96 \
  bash scripts/152_run_residual_language_stability_sweep.sh
```

Result:

| QTRM scale | Records | Clean rate | Repeat failure | Visible reasoning | Answer drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |
| `0.05` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |
| `0.10` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |

Artifact:

```text
runs/language_stability/residual_s010_safe_sweep_96/summary.jsonl
```

Interpretation:
`qtrm_logits_scale <= 0.10` is language-stable under the guarded donor-backed
runtime on these Korean/English/math smoke prompts. This is not yet evidence
that QTRM improves reasoning: residual telemetry showed no donor argmax shifts
on the prompt probes, and the math prompt's wrong answer was already present in
donor-only output. The next gate must measure QTRM usefulness on
evidence-sensitive tasks against donor-only, not only fluency preservation.

## [2026-05-01] implementation | Language-safe donor-preserving inference default

Fixed the immediate language-collapse path in interactive inference.

Diagnosis:
`scripts/90_infer_with_donor.sh` used config defaults where
`qtrm_logits_scale=1.0` and `donor_logits_scale=0.0`, so the "QTRM + donor"
script was actually free-running from the immature QTRM language head. That is
why old checkpoints produced attractors such as `Freeze Freeze ...` or
`world of the world ...`.

Default runtime guard:

```text
LANGUAGE_SAFE=1
DONOR_LOGITS_SCALE=1.0
QTRM_LOGITS_SCALE=0.0
QTRM_RESIDUAL_CLAMP=0.0
SUPPRESS_VISIBLE_REASONING=1
NO_REPEAT_NGRAM_SIZE=2
STOP_AFTER_SENTENCE=1
```

This is a donor-preserving inference mode, not proof that QTRM has learned a
standalone language policy. Raw QTRM experiments remain available with
`LANGUAGE_SAFE=0` and explicit scale overrides. History rows use mode
`language_safe_donor` for the default guard so these samples are not confused
with QTRM-residual capability evidence.

Smoke result:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  CONFIG=configs/qwen35_2b_4090_extended.yaml \
  CHECKPOINT=runs/qwen35_2b_4090_extended/last.pt \
  HISTORY_JSONL=runs/history/generations/language_safe_smoke.jsonl \
  bash scripts/90_infer_with_donor.sh "양자 컴퓨팅이란 무엇인가요?"
```

Output:

```text
양자 컴퓨팅이란 무엇인가요?

양자 컴퓨팅은 양자역학의 원리를 이용하여 컴퓨터를 설계하는 것을 말합니다.

Generated 22 new tokens
```

Verification:

```bash
bash -n scripts/90_infer_with_donor.sh
PYTHONPATH=src uv run python -m py_compile \
  src/qtrm_mm/history.py scripts/95_eval_memory_retrieval.py
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_infer_with_donor_script.py \
  tests/test_generation_history.py \
  tests/test_memory_eval_script.py \
  tests/test_losses.py -q
# 51 passed
```

## [2026-05-01] implementation | Generation history JSONL

Added persistent generation history so QTRM outputs can compound into failure
ledgers, regression tests, and curated training data instead of disappearing in
terminal scrollback.

Added:

- `src/qtrm_mm/history.py`
- `tests/test_generation_history.py`
- `HISTORY_JSONL=auto` support in `scripts/90_infer_with_donor.sh`
- `--history-jsonl-out auto|none|PATH` support in
  `scripts/95_eval_memory_retrieval.py`
- concept page: `docs/wiki/concepts/generation-history.md`

Default paths:

```text
runs/history/generations/YYYY-MM-DD.jsonl
runs/history/evals/YYYY-MM-DD.jsonl
```

Rule:
raw history must not be used as training data directly. It first needs a
validator or human label so that repetition, decoy-copy, hallucination, and
format leakage become explicit failure classes rather than self-training noise.

## [2026-05-01] decision | Limit-aware model objective

Clarified the "model with no limitations" goal. The wiki now treats it as a
limit-aware architecture objective, not as a literal claim that any fixed model
has no limits.

Added:

- `docs/wiki/decisions/limit-aware-model-objective.md`

Key rule:

```text
no known failure class remains unmeasured, unhandled, undocumented, or without
a falsifiable improvement plan
```

Also linked the objective from the limitations roadmap and wiki index. This
keeps future architecture work tied to explicit failure ledgers, prior research,
ablation gates, and wiki preservation instead of adding complexity blindly.

## [2026-05-01] research-driven debug | Answer-channel contract beats post-hoc rerank

Applied the root-architecture doubt gate to the generation verifier path.

Root architecture claim under test:
post-hoc generation-verifier reranking can improve QTRM sampled candidates.

Falsifier:
reranking lowers format-aware candidate quality below the baseline first
candidate.

Result:

- 8 prompts, 3 sampled candidates each;
- baseline quality rate under format-aware labels: `0.625`;
- verifier-reranked quality rate: `0.250`;
- oracle quality rate: `0.750`;
- selected candidate changed on `0.750` of prompts;
- conclusion: current verifier reranker is rejected.

The larger observed failure is visible answer-channel leakage. Raw QTRM/Qwen
sampling frequently emits `<think>` blocks or meta-reasoning before the answer.
Official Qwen docs describe non-thinking generation controls through
`enable_thinking=False` and `/no_think`; QTRM's raw donor-backed path did not
have an equivalent visible-answer contract.

Added:

- `--suppress-visible-reasoning-tokens` to `scripts/92_eval_qtrm_logits.py`;
- `SUPPRESS_VISIBLE_REASONING=1` support in `scripts/90_infer_with_donor.sh`;
- `scripts/147_summarize_generation_format.py`;
- format-aware `quality_target` in
  `scripts/141_build_generation_verifier_dataset.py`;
- concept page:
  `docs/wiki/concepts/answer-channel-contract.md`.

20-prompt single-candidate comparison with drift-aware format detector:

| Mode | Visible Reasoning | Repeat Failure | Answer Drift | Clean / Quality-Like |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.150 | 0.300 | 0.200 | 0.550 |
| suppress `<think>` + no-repeat-2 | 0.000 | 0.000 | 0.150 | 0.850 |
| direct prompt contract + suppress + no-repeat-2 | 0.000 | 0.000 | 0.300 | 0.700 |

Decision:
answer-channel and prefix-time degeneration controls are higher priority than
post-hoc verifier reranking. Narrow `<think>` suppression plus no-repeat-2 is
the current best runtime guard on this smoke. The direct prompt suffix is
rejected as a default because it increases instruction/answer drift with the raw
base donor.

## [2026-05-01] research-driven debug | Generation verifier split calibration

Failure ledger:

- Failure: generation verifier had in-sample best-threshold signal but no
  held-out calibration proof; fixed `0.5` thresholds were poor and stop was
  weak.
- Evidence: all-50 smoke best-threshold F1 was `0.757` repeat and `0.800`
  quality, but fixed-threshold F1 was poor; stop best-threshold F1 was `0.250`.
- Likely component: coda-text verifier has signal, but calibration and stop
  labels are underpowered.
- Prior work checked: unlikelihood training, SimCTG/contrastive search, FUDGE
  future discriminators, and generator-reranker systems.
- Recommended candidate: full-candidate verifier reranker path, starting with a
  train/calibration/holdout split. This is the smallest falsifiable step before
  a hard decoding gate.

Added:

- source notes:
  `docs/wiki/sources/generation-verifier-reranking.md`;
- deterministic splitter:
  `scripts/144_split_generation_verifier_dataset.py`;
- calibration evaluator:
  `scripts/145_calibrate_generation_verifier_eval.py`;
- split smoke config:
  `configs/qwen35_2b_4090_generation_verifier_split_s020.yaml`;
- tests:
  `tests/test_generation_verifier_split_script.py` and
  `tests/test_generation_verifier_calibration_script.py`.

Split result from the 50-row verifier dataset:

- train: 29 rows, 11 repeat failures, 3 stop failures, 16 quality passes;
- calibration: 10 rows, 3 repeat failures, 1 stop failure, 6 quality passes;
- holdout: 11 rows, 4 repeat failures, 2 stop failures, 6 quality passes.

Ran the split-only 20-step verifier smoke from:

```text
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_train.jsonl
```

Saved:

```text
runs/qwen35_2b_4090_generation_verifier_split_s020/last.pt
```

Calibration-selected holdout report:

| Head | Calibration Threshold | Calibration F1 | Holdout F1 | Holdout Precision | Holdout Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat | 0.4948 | 0.667 | 0.571 | 0.667 | 0.500 |
| stop | 0.3495 | 0.286 | 0.222 | 0.143 | 0.500 |
| quality | 0.4918 | 0.800 | 0.750 | 0.600 | 1.000 |

Decision: repeat and quality retain weak held-out signal; stop does not. The
next architecture experiment should be candidate reranking with verifier scores
and a larger on-policy generated dataset, not hard threshold gating.

## [2026-05-01] implementation | Generation verifier head smoke

Implemented the first QTRM-generated output-failure verifier path:

- config flags:
  `generation_verifier_enabled` and
  `loss_generation_verifier_weight`;
- model heads:
  `generation_repeat_head`, `generation_stop_head`, and
  `generation_quality_head`;
- data labels:
  `generation_verifier_repeat_target`,
  `generation_verifier_stop_target`,
  `generation_verifier_quality_target`, and
  `generation_verifier_sample_weight`;
- trainable policy:
  `generation_verifier_only`;
- dataset builder:
  `scripts/141_build_generation_verifier_dataset.py`;
- smoke train runner:
  `scripts/142_run_generation_verifier_s020.sh`;
- evaluator:
  `scripts/143_eval_generation_verifier.py`;
- concept page:
  `docs/wiki/concepts/generation-verifier.md`.

Built the first 50-row verifier dataset from QTRM v2 generated outputs:

- source eval:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_eval.jsonl`;
- prompt metadata:
  `runs/qwen_scope/qtrm_repeat_gate_prompts_s50.jsonl`;
- output:
  `data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier.jsonl`;
- label distribution:
  18 repeat failures, 6 stop failures, 28 quality passes.

Important architecture correction: the first verifier-head placement used pooled
`z_h` and learned mostly label priors. The heads now read the coda/post-norm
last valid text hidden state, which has access to the actual candidate
completion. This makes the verifier an output-failure reader rather than a
latent-state prior classifier.

Second 20-step verifier-only smoke:

- checkpoint:
  `runs/qwen35_2b_4090_generation_verifier_s020/last.pt`;
- eval:
  `runs/qwen35_2b_4090_generation_verifier_s020/eval_generation_verifier_s50.json`.

In-sample evaluation after the coda-text fix:

| Head | Fixed 0.5 F1 | Best in-sample F1 | Decision |
| --- | ---: | ---: | --- |
| repeat | 0.100 | 0.757 | Signal exists, calibration needed. |
| stop | 0.000 | 0.250 | Weak; needs more positives and better labels. |
| quality | 0.000 | 0.800 | Signal exists, calibration needed. |

Decision: keep generation verifier as probe-only. It should next be evaluated
on a larger train/holdout split and used as a reranker before any hard decoding
gate. Do not claim repetition is solved from this smoke.

## [2026-05-01] implementation | Qwen-Scope repeat candidate scorer

Added a Qwen-Scope repeat-candidate scoring path:

- API: `score_qwen_scope_candidate_features` in `src/qtrm_mm/qwen_scope.py`;
- CLI: `scripts/138_score_qwen_scope_repeat_candidates.py`;
- tests:
  `tests/test_qwen_scope.py` and `tests/test_qwen_scope_score_script.py`;
- artifact:
  `runs/qwen_scope/qtrm_v2_repeat_candidate_scores.json`.

The scorer takes layer-specific SAE feature candidates and reports prompt-level
hit counts, summed activation values, max activation, and optional generation
repeat metrics from QTRM eval JSONL.

First actual-v2 result with candidates `12:847` and `23:29838,31860`:

- prompt `3`: `repeated_2gram_rate=0.254`, score `78.820`, rank 1;
- prompt `2`: `repeated_2gram_rate=0.762`, score `46.713`, rank 2;
- prompt `0`: `repeated_2gram_rate=0.016`, score `23.338`, false-positive
  risk;
- prompt `4`: `repeated_2gram_rate=0.175`, score `0.500`, missed mild-repeat
  risk.

Conclusion: Qwen-Scope candidates are useful as a severe-loop diagnostic smoke
signal, especially for prompt-copy loops, but they are not yet a clean binary
repetition detector or causal control feature. The next gate needs 50-100
generated samples and transparent layer-23 rank/value thresholds before any
training or decoding governor is built from these features.

Ran the 50-sample gate:

- prompt suite:
  `runs/qwen_scope/qtrm_repeat_gate_prompts_s50.jsonl`;
- QTRM eval:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_eval.jsonl`;
- Qwen-Scope records:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_layers_12_23.jsonl`;
- threshold summaries:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_threshold_summary_rep15.json`,
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_threshold_summary_severe25.json`,
  and sparse-candidate variants.

Generation distribution:

- 50 prompts total;
- rep2 >= 0.15: 18/50;
- rep2 >= 0.25: 11/50;
- category with highest average rep2: `math_reasoning` at `0.238`;
- evidence-check includes severe claim/evidence copy loops;
- several prompts have low n-gram repetition but still fail to stop before the
  64-token cap.

Detector result:

- original overlap candidates `12:847`, `23:29838,31860` are not clean:
  rep2 >= 0.15 F1 `0.652`, severe rep2 >= 0.25 F1 `0.625`;
- sparse in-sample candidates can fit the 50-sample severe split perfectly, but
  this is not evidence of generalization;
- crude train/holdout check using indices `0-24` for discovery and `25-49` for
  holdout failed: train F1 `1.0`, holdout F1 `0.0`.

Decision: Qwen-Scope should remain diagnostic tooling. Do not build a hard
Qwen-Scope decoding governor yet. The next implementation target should be a
QTRM-generated output-failure dataset plus an explicit stop/format/repetition
verifier head or candidate reranker, with Qwen-Scope features used only for
analysis.

## [2026-04-30] decision | Qwen3.6 teacher distillation order

Added `decisions/qwen36-online-distillation-roadmap.md` to fix the execution
order for DGX-backed teacher distillation.

Decision:

- use DGX as the Qwen3.6-27B teacher/data machine;
- distill QTRM first because the current blocker is generation-side evidence
  use and repetition, not only memory retrieval;
- validate MSA separately as a donor fork with routing/healing losses;
- combine QTRM+MSA only after each path passes its own smoke gates.

Checklist order:

1. stabilize custom full-MSA checkpoint save/load;
2. define Qwen3.6 teacher data schema;
3. configure DGX teacher environment under `/mnt/data4tb`;
4. generate 100-case offline teacher smoke data;
5. train/evaluate QTRM offline distillation;
6. add online top-k teacher KL;
7. add on-policy QTRM correction;
8. add MSA routing-label loss and real 2B healing dry run;
9. run joint QTRM+MSA ablations.

Claim boundary: online distillation is not accepted by lower loss alone. It
must improve held-out generation and show causal use through workspace/core/MSA
ablation.

Implemented the first two execution gates after documenting the plan:

- custom full-MSA checkpoint save/load roundtrip in
  `src/qtrm_mm/qwen35_full_msa.py`;
- Qwen3.6 teacher record schema in
  `src/qtrm_mm/distill/teacher_schema.py`;
- OpenAI-compatible teacher message builder and JSON response parser in
  `src/qtrm_mm/distill/qwen36_teacher_client.py`.

The teacher schema now validates the shared QTRM/MSA distillation record:
`prompt`, `answer`, optional evidence ids/spans, rejected answer, trace summary,
MSA memory docs, target doc ids, and optional top-k teacher logprobs.

Added the first public HF dataset intake path so offline data does not need to
be generated from scratch:

- manifest: `configs/hf_distill_datasets.yaml`;
- source page: `sources/hf-distillation-datasets.md`;
- converter: `src/qtrm_mm/distill/hf_dataset_convert.py`;
- CLI: `scripts/131_convert_hf_distill_dataset.py`;
- first-wave runner: `scripts/132_convert_first_wave_hf_distill_smoke.sh`.

First-wave sources:

- `Yana/ft-llm-2026-reasoning-dpo` for QTRM preference/CoT warmup;
- `AMAImedia/NOESIS-50K-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54`
  for multilingual reasoning warmup;
- `F4biian/RAGognize` for MSA routing/evidence gates;
- `lrsbrgrn/HalluClaim-76k` for hallucination/evidence bottleneck training.

HF smoke note: `datasets` streaming successfully wrote Yana samples but exited
with code 134 during cleanup. The converter now defaults to non-streaming mode;
use `--streaming` only for datasets that cannot be loaded normally.

Actual smoke results:

- Yana non-streaming converted 3/3 rows successfully, but the first rows have
  identical chosen/rejected final answers, so the converter now drops identical
  `rejected_answer` fields.
- RAGognize converted 3/3 rows and produced `memory_docs`; valid responses can
  produce `target_doc_ids`, hallucinated/unsupported responses may have no
  routing target and should be used for evidence-gate negatives.
- HalluClaim converted 3/3 rows after mapping actual columns `doc` and `type`.
- NOESIS converted 3 rows after skipping 3 invalid prompt-only or unclosed
  `<think>` rows; unclosed think rows are skipped to avoid visible reasoning
  leakage.

Built the first mixed QTRM HF warmup JSONL:

- script: `scripts/133_build_hf_distill_training_mix.py`;
- output: `data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl`;
- config: `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml`;
- runner: `scripts/134_run_hf_first_wave_warmup.sh`;
- rows: 400 total, 100 each from Yana, NOESIS, RAGognize, and HalluClaim;
- targets: 171 preference rows, 200 workspace-evidence rows, 166 positive
  `target_doc_ids` rows;
- safety conversion: 34 RAGognize rows with memory docs but no evidence target
  are converted to `chosen=NEEDS_SEARCH` and rejected unsupported answers,
  preventing hallucinated answers from becoming SFT targets.

Also fixed mixed-batch collation in `src/qtrm_mm/data/jsonl_dataset.py` so a
single batch can contain SFT, preference, evidence, and non-evidence rows. The
collator now fills missing optional preference/evidence targets with zero
weights instead of depending on the first sample's key set.

Verification:

- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_hf_distill_training_mix.py tests/test_jsonl_dataset_supervised.py tests/test_hf_distill_manifest.py tests/test_hf_distill_converters.py tests/test_hf_distill_convert_script.py tests/test_qwen36_teacher_distill_schema.py tests/test_qwen36_teacher_client.py`
  passed with 26 tests.
- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_losses.py tests/test_training_checkpoint_init.py tests/test_jsonl_dataset_supervised.py tests/test_hf_distill_training_mix.py`
  passed with 48 tests.
- Data-loader smoke on mixed source-boundary rows produced a 6-row batch with
  workspace tokens, preference weights, logical support targets, and logical
  missing targets present.

Ran the 400-step HF first-wave warmup and saved:
`runs/qwen35_2b_4090_hf_first_wave_warmup_s400/last.pt`.

Safety checks:

- final checkpoint size: 775 MB;
- floating tensors checked: 276;
- NaN/Inf tensors: 0.

Diagnosis after sanity eval:

- The original warmup config had
  `evidence_bottleneck_suppress_without_workspace: true` and the evidence
  bottleneck was applied to the whole QTRM residual.
- Therefore ordinary prompts with no external workspace evidence became
  donor-only at final-logit fusion, even though the latent workspace/core still
  ran internally.
- This is correct only for a strict evidence-causality proof gate, not for the
  general QTRM/loop-LM objective.

Fix:

- added `model.evidence_bottleneck_applies_to_residual`;
- default remains `true` to preserve existing evidence-only proof configs;
- `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml` sets it to `false`,
  making the evidence heads verifier-only while keeping the general QTRM
  residual active without external workspace evidence.

Post-fix eval on the same checkpoint:

- output:
  `runs/qwen35_2b_4090_hf_first_wave_warmup_s400/logit_eval_general_residual_sanity.jsonl`;
- ordinary prompts now have nonzero residual telemetry
  (`residual_linf` about `0.84-0.875`, `donor_to_fused_kl > 0`);
- the evidence-style prompt changes donor argmax, proving QTRM residual is no
  longer shut off;
- however the claim/evidence generation repeats the prompt/evidence pattern,
  so the next run must add format/answer-only supervision and rerun with this
  corrected residual boundary.

Verification:

- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_model_config.py tests/test_hf_distill_training_mix.py tests/test_jsonl_dataset_supervised.py`
  passed with 32 tests.

## [2026-04-30] implementation | Qwen3.5-2B full-MSA fork scaffold

Started the aggressive full-MSA donor fork path requested for Qwen3.5-2B.
Added:

- `src/qtrm_mm/msa_qwen35.py`;
- `scripts/129_prepare_qwen35_full_msa_fork.py`;
- `tests/test_qwen35_full_msa_fork.py`;
- `decisions/qwen35-full-msa-fork.md`.

The converter reads `references/model_configs/qwen35_2b_base/config.json`,
rewrites all 24 text layers to Hugging Face's allowed `sparse` type with
`qtrm_full_msa_fork=true`, and writes a conversion manifest. The manifest makes
the destructive boundary explicit:
Qwen3.5-2B has 18 `linear_attention` layers and 6 `full_attention` layers. The
6 full-attention layers can seed Qwen3.5-native MSA projections, while the 18
linear-attention layers must be replaced and healed because GatedDeltaNet
conv/recurrent weights do not map cleanly to MSA q/k/v/o plus router weights.

Added the first custom text-forward prototype in
`src/qtrm_mm/qwen35_full_msa.py`. It implements Qwen3.5-native MSA attention
with the gated q projection, q/k RMSNorm, partial/mRoPE position embeddings,
chunk-pooled doc-id routing, and sparse selected-document attention. The new
test `tests/test_qwen35_full_msa_model.py` verifies a tiny random
Qwen3.5-style config runs a `doc_ids` forward pass. This is still not a trained
full-MSA donor: HF model registration, weight-copy, Memory Parallel cache
runtime, and donor-healing training remain next.

Added the first safe-healing smoke path:

- `src/qtrm_mm/qwen35_full_msa_healing.py`;
- `scripts/130_train_qwen35_full_msa_healing.py`;
- `tests/test_qwen35_full_msa_healing.py`.

The tiny smoke trains only MSA attention/router parameters while freezing
copied embeddings, MLPs, norms, and LM head. The loss is next-token LM CE plus
donor KL against the original Qwen3.5-style teacher. This is the right safety
shape before a real 2B healing run because it tests parameter freezing,
teacher-preservation, doc-id routing, optimizer update, and report writing
without risking a large donor checkpoint.

Ran the tiny smoke for 2 steps:

- report: `runs/qwen35_full_msa_healing_tiny_smoke/healing_report.json`;
- trainable/frozen params: `11296 / 20640`;
- updated trainable L1: `15.0766`;
- final `loss=4.9102`, `lm_loss=4.8438`, `donor_kl=0.0664`.

## [2026-04-30] research | agentic closed-loop planner references

Added the closed-loop planning research pack for the next QTRM/MemoryOS stage:

- downloaded 19 PDFs into
  `references/papers/agentic_closed_loop_planner/`;
- cloned official repos:
  `agent-lightning@0b40cb724a0a`, `ragen@20daedc47558`,
  `agentgym@c3b300f0381a`,
  `language-agent-tree-search@853d81614607`,
  `ace@4f679bef3b78`, `dynamic-cheatsheet@5cfe3c37e8e5`,
  `skillrl@299909b2f5e2`, and `aworld-rl@2082e70bcd54`;
- added `sources/agentic-closed-loop-planning.md`;
- added `concepts/agentic-closed-loop-planner.md`;
- updated the long-horizon agent architecture and canonical architecture
  matrix.

Conclusion:
QTRM is not yet an autonomous closed-loop agent. The prior-backed path is
trace-first: build an external replayable `AgentHarness`, log state/action/
observation/verifier transitions, then train controller heads and only later
attempt Agent-Lightning/RAGEN-style turn-level RL. RAGEN-2 and ASTER add a
hard requirement for collapse diagnostics: entropy is insufficient; track
input-dependence, reward variance, repeated action templates, and tool/evidence
interaction density.

## [2026-04-30] process | research-driven architecture debugging skill

Added local agent skill
`/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md`.
Future QTRM architecture failures should follow this loop: stabilize the run,
write a failure ledger, search primary prior work and official implementations,
map the failure to a research axis, implement the smallest falsifiable
architecture/training experiment, and require generation plus ablation gates
before accepting the change.

This process was added after the bounded-preference run showed the same pattern
as earlier experiments: pairwise or loss improvements can coexist with visible
generation failures, repeated formats, retrieval-but-wrong answers, and weak
workspace/core causality.

Updated the skill so it is active architecture proposal guidance, not just a
debugging checklist. Once a limitation is clear, the agent must propose 2-3
concrete prior-backed architecture candidates, rank them, choose the most
testable one, and proceed to a minimal experiment unless explicitly paused.

Installed external research-support skills:

- `arxiv-research`: arXiv paper fetch/compile for LLM, agents, RAG, reasoning,
  and AI infrastructure.
- `literature-review`: systematic paper search, triage, and synthesis workflow.
- `deep-research`: multi-source citation-tracked research reports with evidence
  persistence and claim verification.
- `research-lookup`: current scientific/research lookup with backend routing.

Validated all five research-related skills, including the local
`research-driven-architecture-debugging` orchestrator. Adjusted two installed
skill frontmatters so they pass the local skill validator.

## [2026-04-30] diagnosis | preference loss is not enough for causal workspace use

Ran the workspace-evidence preference and preference+repeatguard probes after
the 500-step preference-only run still repeated `Ilya Chen Chen...`.

The preference+repeatguard checkpoint learned the small pairwise objective:
`workspace_evidence_preference_repeatguard_pair_eval_s050.jsonl` reported
11/11 preference accuracy and margin pass. That did not transfer to answer
generation. A 4-case quick workspace-evidence ablation wrote
`docs/wiki/decisions/workspace-evidence-preference-repeatguard-quick-ablation.md`
and showed the critical failure mode:

- retrieval is not the bottleneck in this gate (`retrieved_target_rate=1.0`,
  `target_recall_mean=0.9167`);
- `qtrm_residual_with_evidence` scored 0/4;
- workspace/core/core-context/workspace-gate/workspace-memory off modes produced
  identical completions to the full residual path (`same_completion_rate=1.0`);
- outputs still contain action/text/number repetition.

Interpretation: the current model can overfit preference pairs, but the latent
workspace path is not yet behaviorally causal in generation. The next probe
therefore moves back to a donor-preserving residual architecture instead of
training a freer residual head: bounded residual clamp, normalized residual
gate with floor, student-LM pressure, donor-KL preservation, conservative repeat
unlikelihood, and pairwise preference. Added
`configs/qwen35_2b_4090_workspace_evidence_bounded_preference_s050.yaml` and
`scripts/123_run_workspace_evidence_bounded_preference_train.sh`. This starts
from `runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt`, not the
repetition-damaged preference checkpoint, and saves 100-step checkpoints so the
best non-repeating midpoint can be selected if the final checkpoint regresses.

## [2026-04-30] docs | QTRM terminology and reasoning boundary

Added `concepts/qtrm-terminology.md` as the canonical glossary for donor,
donor-backed residual adapter, cognitive sidecar, LatentWorkspace,
workspace-only evidence, gated core context injection, donor annealing,
internalized context engineering, actual reasoning, and inference cliffs.

Updated the goal/scope and forward-pass pages to state the reasoning boundary:
QTRM is intended to learn a latent state-update process, not just a visible
CoT text pattern, but the claim is valid only when held-out evals and
workspace/core/context ablations prove the latent computation is causally
necessary.

Added `decisions/actual-reasoning-architecture-roadmap.md` to separate the
current proof gate from future improvements: true TRM carry, per-sequence ACT,
latent distillation, state-trajectory consistency, residual usefulness gating,
retrieval feedback, and donor annealing.

Added eval telemetry for the next proof gate: `scripts/95_eval_memory_retrieval.py`
now records `latent_gates.workspace_update_gate_*` and
`latent_gates.core_context_gate_*`, and the strict causality runner includes
`qtrm_core_context_off_with_evidence`. Updated `docs/REFERENCE_BASELINE.md`
with the rule that core-context claims require both the off-ablation and gate
telemetry.

## [2026-04-29] diagnosis | donor annealing needs student-only LM loss

Ran the first donor-anneal probe and confirmed a failure mode: training CE on
fused logits lets donor logits carry fluency while QTRM-only logits remain
weak. The fused-loss-only checkpoint was coherent at donor scale `1.0` and
`0.5`, but collapsed at `0.25` and `0.0`.

Added `qtrm_logits` to `QTRMMultimodalModel` outputs before donor fusion, plus
`TrainConfig.loss_student_lm_weight` and a `student_lm` metric. Donor KL now
distills into QTRM-only logits when available. A 200-step patched probe exposed
the real gap: `lm=2.22` while `student_lm=12.46` at start; by the end
`student_lm` only reached `11.42`, and donor `0.0` still repeated. Added
`configs/qwen35_2b_4090_student_lm_pretrain_probe.yaml` for the next gate:
keep donor logits fixed while pretraining QTRM-only LM before attempting full
donor detach.

Ran the 500-step fixed-donor student pretrain probe. `student_lm` moved from
`12.44` to the 8-ish range (`9.89` at step 200, `8.58` at step 300, `7.85` at
step 400, noisy `8.00` at step 450), confirming the training signal works.
Generation still collapses with donor `0.0`, and QTRM residual scale `0.5`
damages even donor-backed output. With `qtrm_logits_scale=0.1`, donor scales
`1.0`, `0.5`, and `0.25` stayed fluent on the Korean quantum-computing prompt.
Conclusion: keep donor attached, train student longer, and cap/gate residual
amplitude before lowering donor logits further.

## [2026-04-29] implementation | bounded residual gate

Added bounded residual fusion behind config flags:
`qtrm_residual_clamp`, `qtrm_residual_gate_enabled`, and
`qtrm_residual_gate_init_bias`. The model now exposes `qtrm_residual_logits`
and `qtrm_residual_gate` for telemetry. The eval script records
`residual_gate`, and `scripts/90_infer_with_donor.sh` can sweep the bounded
residual settings through `QTRM_RESIDUAL_CLAMP`, `QTRM_RESIDUAL_GATE`, and
`QTRM_RESIDUAL_GATE_BIAS`.

Smoke result on the 500-step student LM pretrain checkpoint: the previously
bad setting `donor_logits_scale=0.5`, `qtrm_logits_scale=0.5` collapsed into
number repetition without bounds, but became fluent with clamp `1.0` and the
gate enabled. The same bounded setting stayed fluent at donor scale `0.25`.

Ran `configs/qwen35_2b_4090_bounded_residual_probe.yaml` for 500 steps. The
QTRM-only student LM signal improved from `12.42` to `10.98`, but the learned
gate saturated closed at about `3.3e-6`. Root cause: the gate linear layer was
fed unnormalized latent states, so a small learned weight still produced a
large negative pre-sigmoid value. Clamp-only evaluation proved the opposite
failure mode: full residual strength preserved some prompts but damaged low
donor-scale generation.

Patched the gate to RMS-normalize its latent input and added
`qtrm_residual_gate_min`. With a `0.05` floor and `qtrm_logits_scale=0.5`, the
corrected 500-step checkpoint reached `student_lm=10.98` and keeps donor scales
`1.0`, `0.5`, and `0.25` fluent on Korean and English smoke prompts. The final
sweep had gate mean about `0.061`, residual L-infinity about `0.0625`, no donor
argmax shift, and repeated 2/3-gram rates at `0.0`. Donor `0.0` still
collapses to `,, and`, so this is not yet donor replacement.

## [2026-04-29] implementation | donor annealing references and KL hook

Downloaded donor-annealing/distillation papers under
`references/papers/donor_annealing`: foundational KD, sequence-level KD,
Annealing-KD, Pro-KD, Distilling step-by-step, MiniLLM, GKD/on-policy
distillation, Universal Logit Distillation, and Multi-Level OT.

Cloned implementation references under `references/official`: TRL
GKD/MiniLLM, Microsoft LMOps MiniLLM, Google distilling-step-by-step,
EasyDistill, MiniPLM, Multi-Level OT, and Teacher Assistant KD. Added
`docs/wiki/sources/donor-annealing-distillation.md` and linked it from the
donor annealing roadmap.

Added the first implementation hook beyond static scaling:
`loss_donor_kl_weight`, `donor_kl_beta`, and `donor_kl_temperature`. The loss
uses generalized donor-logit distillation so the QTRM/fused policy can be
trained against Qwen donor logits while `donor_logits_scale` is annealed down.

## [2026-04-29] probe | core halt training gate

Added `configs/qwen35_2b_4090_core_halt_probe.yaml` and
`scripts/107_run_core_halt_probe.sh`. This is the first executable gate for
the latent-loop early-exit idea: train with `core_halt_auto_targets=true`,
`loss_core_halt_weight>0`, `outer_steps=3`, and Qwen donor logits preserved as
the base language policy.

Extended `scripts/92_eval_qtrm_logits.py` with `--enable-core-halt` and
`core_halt` JSON/text telemetry. The post-eval artifacts now expose
`core_steps`, `core_halted`, and halt-head logit summaries, so we can check
whether a trained checkpoint actually exits early instead of only having a
halt loss in code.

Ran the 300-step probe from the memory-synth checkpoint. The training path is
healthy, but the first automatic target rule is too conservative for early
exit: `core_halt` loss collapsed to near zero, and post-eval on 16 samples
reported `core_steps={3:16}` and `core_halted={False:16}`. This means the next
gate needs balanced positive halt labels, teacher-depth labels, or a target
availability report before expecting runtime savings.

Added halt target availability diagnostics to `qtrm_smoke_loss`. Auto-target
runs now report `halt_target_pos_rate`, `halt_target_neg_rate`,
`exact_next_token_pass_rate`, and donor KL gate stats. A one-batch check on the
core-halt probe checkpoint reported `halt_target_pos_rate=0.0`,
`exact_next_token_pass_rate=0.0`, and `donor_kl_pass_rate=1.0`, confirming that
the exact next-token gate, not the KL gate, is blocking early-exit positives.

## [2026-04-29] implementation | CoT-to-latent halting hook

Added `core_halt_loss` and `TrainConfig.loss_core_halt_weight` so the QTRM
halt head can be trained when a future dataset supplies `core_halt_targets`.
The loss is wired into `qtrm_smoke_loss` without changing default training
behavior because the weight defaults to `0.0`.

Added `infer_core_halt_targets` plus `core_halt_auto_targets` and
`core_halt_donor_kl_threshold`. The first automatic target rule is conservative:
exact next-token correctness, optional verifier pass/fail, and optional
fused-vs-donor KL stability must all pass before teaching the core to halt.

Downloaded and indexed the CoT-to-latent reference set: Coconut, CODI,
HybridCoT, and Do Latent Tokens Think. Cloned CODI at `2c23146`. Added source
and concept wiki pages for using explicit CoT as teacher supervision while QTRM
performs repeated latent workspace computation.

Added the first TRM-style halting hook to `QTRMRecursiveCore`: optional
`core_halt_enabled`, `core_halt_min_steps`, `core_halt_use_continue`,
`halt_head`, `enable_core_halt`, and telemetry outputs. This is not full TRM
ACT yet; persistent carry, per-sequence reset, exploration, and halt loss remain
future work.

Extended the MemoryOS answer-level eval with `qtrm_workspace_off_*` and
`qtrm_core_off_*` modes plus `scripts/106_run_ablation_proof.sh`, because
latent-reasoning claims require answer-level component ablations, not only
next-token telemetry.

## [2026-04-29] decision | innovation claim boundary

Clarified the innovation claim. QTRM should be framed as a frozen-donor,
trainable cognitive-sidecar architecture for evidence-sensitive residual
correction, not as a tiny standalone model that generally beats trillion-scale
LLMs. The comparison target is donor-only Qwen, ordinary RAG, LoRA/SFT adapters,
prompt/tool agents, and direct long-context scaling.

## [2026-04-29] docs | limitation-mitigation diagram prompt

Updated the paper-diagram prompt bank so limitation fixes are explicitly
visible inside generated figures. The main architecture prompt now requires a
`Limitations -> Mitigations` panel, and the new overlay prompt maps donor
dependency, residual fluency risk, unproven latent reasoning, misleading
retrieval, and long-context cost to concrete mitigation modules and eval gates.

## [2026-04-29] implementation | ablation modes and paper prompts

Added fixed eval ablation modes for `donor_only`, `residual`, `workspace_off`,
and `core_off`. The same checkpoint can now isolate donor policy, full residual
path, latent workspace contribution, and recursive-core contribution. Added a
paper-diagram prompt bank for QTRM limitation mitigation, ablation lanes, and
the realistic claim boundary for a tiny cognitive core.

## [2026-04-29] implementation | residual telemetry

Added residual-logit telemetry as the first mitigation-roadmap implementation
step. `residual_logit_telemetry` compares scaled donor logits with fused logits
and reports argmax changes, KL divergence, and residual norms. `92_eval_qtrm_logits.py`
now includes this in JSON/text eval output when donor logits are available.

## [2026-04-29] decision | limitations and mitigation roadmap

Added `decisions/limitations-mitigation-roadmap.md` to keep the current QTRM
limits tied to concrete mitigations: Proxy-Tuning/DExperts for logit residuals,
Side-Tuning/Ladder Side-Tuning for the frozen-donor sidecar pattern,
KL/distillation for donor preservation, looped-transformer/Parcae references for
recursive stability, and Self-RAG/CRAG/RAPTOR-style gates for MemoryOS. The next
engineering order is telemetry, ablations, gated residual, and KL-to-donor loss
before longer training.

## [2026-04-29] plan | current architecture pretrain viability probe

Added `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` and
`scripts/105_run_current_arch_pretrain_probe.sh`. This locks the next
architecture gate before more reasoning-data expansion: Qwen donor logits remain
the base language policy, QTRM contributes only a bounded residual
(`qtrm_logits_scale=0.10`), workspace is enabled, and JEPA/aux losses stay off.

The probe trains 2000 steps on `data/filtered/qtrm_clean_pilot.jsonl`, runs live
prompt diagnostics, and writes `post_eval.jsonl` plus `post_eval_prompts.txt`
under `runs/qwen35_2b_4090_current_arch_pretrain_probe`.

## [2026-04-29] implementation | Bongakgyo critical synthesis case expansion

Added `src/qtrm_mm/training/bongak_critical_synthesis_cases.py` and
`scripts/104_build_bongak_critical_synthesis_cases.py`. Built 30
`bongak_critical_synthesis` cases from the local
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_요약.md`
and
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_매뉴얼.md`
sources.

Outputs:
`data/filtered/critical_synthesis_bongak_cases.jsonl` and
`data/filtered/critical_synthesis_bongak_traces.jsonl`, 30 rows each. These
rows intentionally treat the local doctrine as source material rather than
truth labels: the target answers must critique weak or controlling claims,
preserve useful practice values, bracket unverifiable metaphysics, and end with
a positive constructive conclusion.

## [2026-04-29] implementation | critical synthesis trace builder

Added `src/qtrm_mm/training/critical_synthesis_data.py` and
`scripts/103_build_critical_synthesis_traces.py`. Built
`data/filtered/critical_synthesis_traces.jsonl` from
`data/eval/critical_synthesis_probe.jsonl`. The generated rows preserve the
required structure: critique, preserve, risks, reframe, and positive
conclusion.

## [2026-04-29] implementation | fact verification and critical synthesis gates

Added a deterministic fact-verification gate:
`src/qtrm_mm/eval/fact_verification.py`,
`src/qtrm_mm/training/fact_verification_data.py`,
`scripts/102_eval_fact_verification_memoryos.py`, and
`data/eval/fact_verification_probe.jsonl`. The gate separates verdict accuracy,
action accuracy, retrieval recall, temporal priority, authority priority, and
conflict handling before generation.

Added the critical-synthesis axis for religion/value questions:
`src/qtrm_mm/eval/critical_synthesis.py` and
`data/eval/critical_synthesis_probe.jsonl`. The target is not blind skepticism;
it requires critique, preservation of value, risk checks, reframing, and a
constructive positive conclusion. Added wiki pages for value/critical synthesis
and the 본각교 handling contract.

## [2026-04-29] ingest | fact verification and fake-info references

Downloaded fact-verification and fake-info papers under
`references/papers/fact_verification`: FEVER, FActScore, LongFact/SAFE,
RAGTruth, OpenFActScore, CIBER, CONFACT, CARE-RAG, verifiable misinformation
agent, LiveFact, and ArbGraph. Cloned official repositories for FEVER baseline,
FActScore, LongFact/SAFE, RAGTruth, OpenFActScore, CONFACT, and ArbGraph under
`references/official`.

Added wiki pages for fact-verification reasoning. The design now separates
predictive latent intuition from factual truth: LeWorldModel remains the
world-model prior, while MemoryOS needs retrieval, source metadata, atomic claim
verification, conflict arbitration, temporal labels, and `NEEDS_SEARCH` routing.

## [2026-04-29] implementation | self-improvement preference rows

Downloaded current self-improvement/hallucination references under
`references/papers/self_improvement` and added wiki pages for the self-
improvement loop. The design now treats `UNKNOWN` as a closed-evidence eval
label, not the final open-world MemoryOS answer. In agentic mode, insufficient
evidence routes to `Action: NEEDS_SEARCH`, then retrieval/search expands the
evidence set before answering or reporting bounded non-verification.

Added `src/qtrm_mm/training/self_improvement_data.py` and
`scripts/101_build_self_improvement_preferences.py`. Built
`data/filtered/memory_self_improvement_preferences_analysis.jsonl` from the
held-out MemoryOS eval: 11 analysis-only rows, with 6 `needs_search` states and
5 answer-correction states. These rows are explicitly `analysis_only` to avoid
held-out train leakage.

## [2026-04-29] eval | held-out MemoryOS generalization

Ran the new held-out MemoryOS reasoning gate after synthetic trace expansion.
The checkpoint `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
evaluated on `data/eval/memory_reasoning_heldout_probe.jsonl` with Harrier
top-30, Qwen3-Reranker-0.6B, and top-5 evidence.

Result: donor-only stayed at 6/12, while QTRM residual improved to 9/12.
QTRM residual solved abstention 4/4 but missed two Korean conflict cases and
one English multi-hop case. This confirms the earlier 9-case trace fine-tune
was overfit, while the broader synthetic trace set generalizes partially.

Next gate: fix Korean temporal/authority conflict selection, the remaining
multi-hop miss, and UNKNOWN repetition artifacts before scaling MemoryOS.

## [2026-04-29] synthesis | long-horizon agent references

Added `sources/long-horizon-agent-references.md` and
`concepts/long-horizon-agent-architecture.md`. The wiki now records the current
agentic/long-context reference bundle: Externalization in LLM Agents, Memory for
Autonomous LLM Agents, MemGPT, Memex(RL), Recursive Language Models, ReAct,
Reflexion, Voyager, SWE-agent, Self-RAG, CRAG, and RAPTOR.

Decision captured: QTRM should not be redesigned as a standalone long-running
agent. Long-running behavior belongs in MemoryOS plus an agent harness with
mode routing, indexed evidence, trace memory, reflection memory, skill memory,
sandboxed execution, budgets, and verification gates. RLM is a future inference
mode, not the default training target.

## [2026-04-29] ingest | Titans neural memory

Added `references/official/titans-pytorch` at commit `714a14c` and downloaded
`references/papers/long_term_memory/titans_2501.00663.pdf`. Added wiki pages
for Titans and neural long-term memory. QTRM now treats Titans as a future
long-term/test-time memory ablation, separate from the current in-context
`LatentWorkspace` and donor-logit residual path.

Clarified architecture wording: QTRM performs looped latent-workspace
computation over donor representations, but it should be described as a
Qwen-backed looped latent-workspace residual adapter rather than a standalone
loop LM or a proven latent-reasoning LM.

## [2026-04-29] experiment | residual 0.10 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s010_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`. Live diagnostics at
200/400/600/800/1000 steps remained stable, and independent reload evaluation
passed the same language-stability gate as residual `0.05`.

Metrics: Korean quantum prompt `loss=2.1777`, `ppl=8.83`,
`rank_mean=6.62`, `rep2=0.000`; English entanglement prompt `loss=2.8766`,
`ppl=17.75`, `rank_mean=15.40`, `rep2=0.032`; math prompt `loss=2.2573`,
`ppl=9.56`, `rank_mean=53.67`, `rep2=0.115`. The math prompt generated the
correct answer and stopped without continuing into another algebra example.

Decision: `qtrm_logits_scale=0.10` is inside the current safe residual range
and becomes the next working candidate. Do not scale residual strength further
until donor-only versus QTRM-residual memory/retrieval evals exist.

## [2026-04-29] experiment | residual 0.05 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s005_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s005_1000/last.pt`. Independent reload
evaluation passed the language-stability gate: Korean and English generations
remained coherent with no `Freeze`, no `world of the world`, and no single-token
collapse. Metrics: Korean quantum prompt `loss=2.2096`, `ppl=9.11`,
`rank_mean=6.88`, `rep2=0.000`; English entanglement prompt `loss=2.9217`,
`ppl=18.57`, `rank_mean=16.40`, `rep2=0.048`. The math prompt answered `x=4`
then continued into another synthetic algebra example; this is template
continuation rather than the earlier collapse mode, but future math evals need
answer-only stop criteria.

Verification: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`
ran 36 tests successfully, and `py_compile` passed for the modified model,
training, donor, loss, and eval files.

## [2026-04-29] architecture | QTRM forward pass diagrams

Added `docs/wiki/architecture/qtrm-forward-pass.md` with Mermaid flowcharts,
generation sequence, tensor shape ledger, and current donor-logit residual
contract. The page explicitly separates latent-workspace computation from
claims of independent latent reasoning or standalone QTRM language generation.

## [2026-04-29] synthesis | Karpathy cognitive core and QTRM goal

Added corrected wiki notes for the Karpathy/Dwarkesh cognitive-core claim. The
wiki now distinguishes the supported hypothesis from the social-media overclaim
that a clean 1B model directly equals a 1.8T frontier model. Added QTRM goal and
scope decision: prioritize donor baseline, data quality, tiny-overfit,
target-token-rank, entropy/repetition diagnostics, memory traces, and ablations
before another long training run.

## [2026-04-29] implementation | QTRM training diagnostics

Added diagnostic utilities and scripts:
`scripts/91_donor_only_generate.sh`, `scripts/92_eval_qtrm_logits.py`,
`scripts/92_eval_qtrm_logits.sh`, and
`scripts/93_tiny_overfit_donor_adapter.sh`. The first checkpoint probe on
`runs/qwen35_2b_4090_extended/last.pt` shows `Freeze` as the top next token and
greedy completion collapse with donor on and off. The checkpoint also predates
the current LeWM/SIGReg JEPA module layout, causing 45 missing new JEPA keys and
4 unexpected old `jepa.net.*` keys during non-strict loading.

Follow-up diagnostics: Qwen donor-only generation is coherent on Korean prompts.
Current-code donor-on tiny-overfit reaches `rank=1.0` and `top1=1.0`, and a
saved tiny checkpoint continues `Quantum entanglement means` without `Freeze`.
Found and fixed a real data pipeline bug: HF-tokenized JSONL samples now carry
an `attention_mask` based on the tokenizer pad id instead of assuming pad id
`0` in `collate_jsonl`.

Ran `configs/qwen35_2b_4090_fixed_pilot.yaml` for 120 real-data steps after
the mask fix. The pilot loss moved from roughly `lm=12.51` to `lm=9.19`.
Evaluation on the Korean quantum prompt no longer selected `Freeze`; the top
tokens were high-frequency punctuation/function words and greedy decoding
formed `, and the.` loops. Interpretation: current code and fixed masking avoid
the old `Freeze` attractor in this short pilot, but the run is far below a
usable LM loss and still needs clean data, longer training, and live
entropy/repetition gates.

## [2026-04-29] ingest | OpenMythos and recurrent depth

Added `references/official/openmythos` at `8c68c1f` and
`references/official/parcae` at `dee8363`. Downloaded recurrent-depth papers:
Parcae stable looped LMs, Looped Transformers are Better at Learning Learning
Algorithms, Reasoning with Latent Thoughts, and a negative/limited latent-CoT
probing paper. Updated recursive-core docs with stable injection, depth-sweep,
and recurrence telemetry requirements.

## [2026-04-29] ingest | Architecture search and composition

Added architecture-search papers under `references/papers/architecture_search`:
Transformer modification transfer, NAS survey, RegNet/design spaces,
EfficientNet, ConvNeXt, task arithmetic, NAS search-phase evaluation, and No
Free Lunch. Added wiki pages for treating QTRM as a compositional architecture
that must be validated through ablations rather than assumed optimal.

## [2026-04-29] ingest | Training diagnostics

Added training-diagnostics papers under
`references/papers/training_diagnostics`: LLM scaling laws, Chinchilla
compute-optimal training, gradient noise scale, LC-PFN learning-curve
extrapolation, ACL 2025 LLM training dynamics, and Prechelt early stopping.
Added wiki pages for using these papers as QTRM run-failure probes.

## [2026-04-29] ingest | LeWorldModel

Added `references/official/le-wm` and `references/papers/leworldmodel_2603.19312.pdf`.
Changed the JEPA target from older stop-grad/EMA style to LeWM-style
end-to-end next-embedding prediction plus SIGReg.

## [2026-04-29] ingest | Tiny Recursive Models

Added `references/official/tiny-recursive-models` and
`references/papers/tiny_recursive_models_2510.04871.pdf`. Marked current QTRM
recursive core as needing comparison against TRM carry, no-grad cycles,
carry-detach, and ACT halting.

## [2026-04-29] ingest | Gated DeltaNet

Added `references/official/gated-delta-net` and
`references/papers/gated_delta_networks_2412.06464.pdf`. Marked current
`TorchGatedDeltaMixer` as a smoke/debug fallback, not official Gated DeltaNet.

## [2026-04-29] implementation | Gated DeltaNet adapter

Updated `src/qtrm_mm/mixers.py` to prefer the official FLA import path
`from fla.layers import GatedDeltaNet`. Added adapter tests for strict mode,
non-strict fallback, mask forwarding, and backend registry detection.
Updated the 4B adapter config and production backend docs to prefer
`delta_backend: fla_gated_delta`.

## [2026-04-29] ingest | Karpathy LLM Wiki

Added wiki schema under `docs/wiki`. Adopted the raw-source / synthesized-wiki /
schema split for QTRM architecture work.

## [2026-04-29] ingest | Qwen3.5 donor architecture

Downloaded Qwen3.5 2B Base and chat model cards/configs into
`references/model_configs`. Added Qwen3.5 Omni technical report
`references/papers/qwen35_omni_technical_report_2604.15804.pdf`. Created wiki
pages for Qwen3.5 architecture and donor integration constraints.

## [2026-04-29] ingest | Qwen donor lineage

Added `references/official/qwen35`,
`references/papers/qwen3_vl_technical_report_2511.21631.pdf`, and
`references/papers/qwen3_omni_technical_report_2509.17765.pdf`. These cover
Qwen3.5 release guidance plus Qwen3-VL/Qwen3-Omni design lineage.

## [2026-04-29] ingest | Transfer, merge, healing

Added model merging and healing-tune references under
`references/official/model-merging` and `references/papers/model_merging`.
Covered MergeKit, TIES, DARE, DELLA, Model Soups, Branch-Train-Merge,
continual/domain-adaptive pretraining, model-merging surveys, and
merge-friendly fine-tuning. Added QTRM wiki pages for the transfer/healing axis.

## [2026-04-29] diagnostics | QTRM collapse probes

Added donor-only, logit, and tiny-overfit diagnostics. The donor-only Qwen
baseline generates coherent text, and a 16-sample donor-on tiny-overfit run
reaches `rank=1.0`/`top1=1.0`, so the current path can learn. The old extended
checkpoint is invalid for current architecture comparisons because it has JEPA
state mismatch and collapses to `Freeze`.

Fixed HF padding-mask handling in `jsonl_dataset`: padding is now derived from
the tokenizer pad id instead of assuming `0`. Added a clean text-only pilot data
builder and built `data/filtered/qtrm_clean_pilot.jsonl` with 6000 accepted
rows. The 500-step clean pilot reached roughly `lm=6.86` from an initial
`lm=12.52`. It removed the `Freeze` top-token failure but still free-runs into
dialogue markers and repeated high-frequency phrases, so the next work is an
objective/data/architecture ablation rather than another blind long run.

Also fixed autoregressive donor handling in `scripts/90_infer_with_donor.sh`
and training diagnostics: donor states are refreshed against the full generated
sequence instead of being held at the initial prompt length. Re-running the clean
checkpoint with refreshed donor states still repeats, confirming this was a
real consistency bug but not the remaining collapse's primary cause.

Added configurable train loss weights and ran
`configs/qwen35_2b_4090_clean_lm_only_pilot.yaml` with `loss_jepa_weight=0` and
`loss_aux_weight=0`. The 300-step LM-only run reached about `lm=6.73` but still
generated `world of the world` loops, so JEPA/aux should not be treated as the
main collapse source.

## [2026-04-29] ingest | Workspace and memory references

Added official/near-official references for latent workspace and memory:
DeepMind Perceiver, Salesforce LAVIS/BLIP-2 Q-Former, OpenFlamingo
PerceiverResampler, and ARMT. Downloaded related PDFs under
`references/papers/memory_workspace`. The current QTRM workspace is now treated
as an in-context working-memory adapter, not a proven persistent memory system.

## [2026-04-29] implementation | Donor logits and Perceiver-style workspace

Added donor-logit residual generation through `donor_logits_scale` and
`qtrm_logits_scale`. `donor_logits_scale=1.0, qtrm_logits_scale=0.0` removes the
generation collapse and produces Qwen-quality completions, proving the previous
loop was caused by forcing a random QTRM LM head to learn the full language
distribution. Added Perceiver/OpenFlamingo-style workspace depth controls:
`workspace_layers`, `workspace_ff_mult`, and `workspace_include_latents_in_kv`.

Ran `configs/qwen35_2b_4090_donor_residual_workspace_pilot.yaml` for 120 steps.
The run stayed near donor-quality generation on both Korean and English prompts
with no `world of the world` collapse. This establishes the next baseline:
Qwen donor as base policy, QTRM as a small residual workspace/memory adapter.

## [2026-04-29] ingest | In-Place TTT

Added ByteDance Seed `In-Place-TTT` at `references/official/in-place-ttt` and
downloaded `references/papers/test_time_training/in_place_ttt_2604.06169.pdf`.
This is relevant to QTRM as a donor-side adaptive-memory axis: selected Qwen/LLaMA
MLP down-projection fast weights are updated during inference with a
next-token-prediction-aligned objective. It should be evaluated as a separate
ablation after donor-logit residual generation remains stable.

## [2026-04-29] plan | Residual ablation

Added `docs/wiki/decisions/residual-ablation-plan.md` to preserve the immediate
experiment plan. Added 1000-step residual configs for `qtrm_logits_scale=0.05`
and `0.10`. The first run is `0.05` with donor logits as the base policy,
Perceiver-style workspace depth enabled, and JEPA/aux disabled.

## [2026-04-29] eval | Memory retrieval probe

Added `data/eval/memory_retrieval_probe.jsonl`,
`src/qtrm_mm/eval/memory_retrieval.py`, and
`scripts/95_eval_memory_retrieval.py`. The script compares donor-only logits
against QTRM residual logits with and without fixed MemoryOS-style evidence, and
scores only generated completion text so answers present in the prompt are not
counted as hits.

Initial result on
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`: donor-only with
evidence `5/5`, QTRM residual with evidence `5/5`, donor-only no evidence `0/5`,
and QTRM residual no evidence `0/5`. Interpretation: evidence injection works,
but the behavior is already provided by Qwen donor logits. QTRM residual `0.10`
does not break the path, but this probe does not yet prove a distinct
QTRM-specific memory policy.

## [2026-04-29] eval | Harrier MemoryOS retrieval

Changed the MemoryOS text embedding default to
`microsoft/harrier-oss-v1-270m` and updated Harrier query encoding to use the
official SentenceTransformers `prompt_name="web_search_query"` path, with a
custom-instruction fallback for other embedders. Added
`scripts/96_build_memory_retrieval_probe_index.py` to build probe indexes that
preserve `case_id`, `evidence_role`, and `is_target` metadata.

Built `runs/eval/memory_retrieval_memoryos_harrier270m_index` from
`data/eval/memory_retrieval_distractor_probe.jsonl` and ran
`scripts/95_eval_memory_retrieval.py --evidence-mode memoryos`. Result:
retrieved target `10/10`, answer hit `10/10` across donor-only and QTRM
residual modes. This verifies the real MemoryOS retrieval path, but donor-only
still matches QTRM residual, so the next task is a memory trace or distractor
task where QTRM must improve over donor evidence copying.

## [2026-04-29] eval | Qwen3 reranked MemoryOS

Added `src/qtrm_mm/memoryos/rerank.py` with `none`, `lexical`, and
`cross_encoder` reranking backends. `cross_encoder` supports
`Qwen/Qwen3-Reranker-0.6B` through SentenceTransformers and caches the model
within the process. Updated `retrieve.py` and
`scripts/95_eval_memory_retrieval.py` to support
`retrieve_top_n -> rerank -> retrieval_top_k` evidence selection.

Smoke-tested Qwen3-Reranker-0.6B on two documents: it ranked the target archive
code document above the vault distractor (`10.75` vs `-4.6875`). Full
MemoryOS eval with Harrier top-20 and Qwen3 top-3 reranking scored retrieval
target `10/10` and answer hit `10/10` across donor-only and QTRM residual
modes. The reranker improves evidence ordering, but this probe still does not
show a QTRM-specific advantage over donor-only generation.

## [2026-04-29] eval | Hard MemoryOS reasoning probe

Added `data/eval/memory_reasoning_probe.jsonl` with temporal conflict,
authority conflict, English/Korean multi-hop, and negative missing-answer
cases. Added target recall/all-target retrieval metrics to the memory eval
helpers and JSONL output.

Built `runs/eval/memory_reasoning_harrier270m_index` and ran Harrier top-20 ->
Qwen3-Reranker-0.6B -> top-5 evidence generation. Result: donor-only `5/6`,
QTRM residual `5/6`, with all target evidence retrieved in both modes. The
single failure is the negative missing-answer case: both modes answered a seen
distractor passphrase instead of `UNKNOWN`. This isolates a real next weakness:
abstention/contradiction handling, not retrieval recall.

## [2026-04-29] plan | 100M MemoryOS and MSA

Cloned the MSA reference implementation to `references/official/msa` at
`30405b2`. Added `docs/wiki/sources/memory-sparse-attention.md` and
`docs/wiki/decisions/memoryos-100m-scale-plan.md`.

Added `src/qtrm_mm/memoryos/scale_plan.py` and
`scripts/97_plan_memoryos_scale.py` to estimate large MemoryOS builds before
ingestion. The default 100M-token estimate with 512-token chunks, 64-token
overlap, and Harrier 270M 640-dimensional embeddings is 223,215 chunks and
about 0.532 GiB of float32 embedding storage. Decision: 100M+ tokens is a
MemoryOS external-memory target, while active model context remains a much
smaller retrieved/reranked/compressed working set.

## [2026-04-29] eval | Abstention metrics before scale

Expanded `data/eval/memory_reasoning_probe.jsonl` from 6 to 9 cases with more
negative missing-answer, authority-conflict, and temporal missing-current-state
checks. Updated `src/qtrm_mm/eval/memory_retrieval.py` and
`scripts/95_eval_memory_retrieval.py` so records carry `category`,
`task_family`, and `expected_unknown`, and summaries report accuracy/retrieval
metrics by mode, category, and task family.

Added `src/qtrm_mm/memoryos/scale_benchmark.py` and
`scripts/98_benchmark_memoryos_scale.py` to write staged 1M/10M planning records
before attempting larger MemoryOS ingestion. Default output is
`runs/eval/memoryos_scale_plan_1m_10m.jsonl`. Current order: fix "retrieved but
answered wrong" on the small hard probe before treating 100M-scale architecture
as the main bottleneck.

Rebuilt `runs/eval/memory_reasoning_harrier270m_index` from the expanded
9-case probe, then reran Harrier top-20 -> Qwen3-Reranker-0.6B -> top-5
generation into `runs/eval/memory_reasoning_qwen3_rerank_32tok_expanded.jsonl`.
Result: donor-only `6/9`, QTRM residual `6/9`, all target evidence retrieved
`18/18`. By task family, conflict `8/8`, multi-hop `4/4`, abstention `0/6`.
This confirms the current blocker is not search recall; it is answer selection
when the correct response is `UNKNOWN`.

## [2026-04-29] train | Memory trace abstention fine-tune

Added `src/qtrm_mm/training/memory_trace_data.py` and
`scripts/99_build_memory_trace_data.py` to convert hard MemoryOS reasoning cases
into supervised traces. Wrote
`data/filtered/memory_abstention_traces.jsonl` with 27 rows: `target`, `all`,
and `lexical` evidence variants for each of the 9 probe cases.

Changed `JsonlTextVisionDataset` and `qtrm_smoke_loss` so rows with
`prompt`/`answer` train only answer tokens via `labels=-100` on prompt tokens.
Added `--init-checkpoint` to `qtrm_mm.training.train` so memory trace runs can
continue from the stable residual checkpoint.

Strict prompting alone did not fix the issue: with the original residual
checkpoint, donor-only and QTRM residual still failed abstention. A first
trace run at `qtrm_logits_scale=0.1` learned some UNKNOWN behavior but was too
weak; scale sweeps showed `0.5` was useful while `1.0` over-abstained.

Final run:

- Config: `configs/qwen35_2b_4090_memory_abstention_trace_s050.yaml`
- Init: `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Output: `runs/qwen35_2b_4090_memory_abstention_trace_s050/last.pt`
- Eval: `runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl`

Result on the expanded 9-case hard probe with all target evidence retrieved:
donor-only `5/9`, QTRM residual `9/9`. QTRM residual task-family accuracy:
conflict `4/4`, multi-hop `2/2`, abstention `3/3`. This fixes the current
small-scale "retrieved but answered wrong" blocker, but it is not yet proof of
general MemoryOS reasoning beyond this synthetic probe.

## [2026-04-29] train | Teacher-depth core halt probe

Found and fixed a wiring issue in the early-halt probe: the global
`attn_every=4` combined with `n_coda_layers=2` left the coda with no attention
layer, so the recursive core prefix could fail to affect text logits. Added
`QTRMConfig.coda_attn_every` and set the probe config to `coda_attn_every=2`,
leaving the core's 3:1 delta/attention schedule unchanged.

Added teacher-depth halt targets based on per-depth QTRM residual last-token
logits. The target compares each depth to the final depth with top-1 match,
centered-logit cosine, and KL checks. Donor logits are deliberately excluded
from this comparison because they are depth-invariant and can hide whether the
QTRM core itself has stabilized.

Probe run:

- Config: `configs/qwen35_2b_4090_core_halt_probe.yaml`
- Init: `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
- Output: `runs/qwen35_2b_4090_core_halt_probe/last.pt`
- Eval: `runs/qwen35_2b_4090_core_halt_probe/post_eval_core_halt.jsonl`

Result after 300 steps: post-eval on 8 clean-pilot samples reported
`core_steps={1:8}` and `core_halted={true:8}`; greedy generation over 40
generated steps also used `core_steps={1:40}`. Average teacher-forced loss was
`1.934`, top-1 accuracy `0.527`, and repeated-2gram rate `0.0`. This proves the
halt path can learn and execute early exit with donor-assisted residual
generation. It is not yet proof that step-1 latent reasoning is sufficient on
hard reasoning tasks; the next gate is comparing halted vs full-depth answers
on the MemoryOS hard probe.

## [2026-04-29] eval | Core halt MemoryOS depth gate

Added `--core-halt-mode {config,enabled,disabled}` to
`scripts/95_eval_memory_retrieval.py` and recorded prompt-level `core_halt`
telemetry in each MemoryOS eval record. This lets the same checkpoint be
evaluated in full-depth mode (`enable_core_halt=False`) and early-exit mode
(`enable_core_halt=True`).

Ran the teacher-depth halt checkpoint with `--qtrm-logits-scale 0.5`,
Harrier retrieval, Qwen3-Reranker-0.6B, and top-5 evidence.

9-case hard probe:

- Full depth: `runs/eval/memory_reasoning_corehalt_full_depth_32tok.jsonl`
- Halted: `runs/eval/memory_reasoning_corehalt_enabled_32tok.jsonl`
- Result: full-depth `5/9` with `core_steps={3:9}`; halted `5/9` with
  `core_steps={1:9}`; hit changes `0`.

12-case held-out probe:

- Full depth:
  `runs/eval/memory_reasoning_heldout_corehalt_full_depth_32tok.jsonl`
- Halted:
  `runs/eval/memory_reasoning_heldout_corehalt_enabled_32tok.jsonl`
- Result: full-depth `7/12` with `core_steps={3:12}`; halted `7/12` with
  `core_steps={1:12}`; hit changes `0`.

Interpretation: early exit did not add answer-level regressions in these two
MemoryOS gates, and it reduced recursive outer depth from 3 to 1. However, the
core-halt fine-tune checkpoint is weaker than the earlier synthetic MemoryOS
checkpoint on held-out accuracy (`7/12` here versus prior `9/12`). The next
training pass should preserve MemoryOS behavior while teaching halting, either
by training halt targets on MemoryOS traces or by freezing most residual-path
weights and training the halt head/coda gate more narrowly.

## [2026-04-29] train | MemoryOS-preserving halt-only probe

Added `TrainConfig.trainable_param_policy` and
`configure_trainable_parameters`. The first narrow policy is
`core_halt_only`, which freezes the full QTRM residual path and trains only
`core.halt_head.weight` and `core.halt_head.bias`. This lets halt behavior be
learned without changing full-depth MemoryOS logits.

Run:

- Config: `configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml`
- Script: `scripts/108_run_memory_halt_preserve.sh`
- Init: `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
- Data: `data/filtered/memory_reasoning_synth_traces.jsonl`
- Trainable tensors: 2
- Trainable params: 1,026
- Output: `runs/qwen35_2b_4090_memory_halt_preserve_s050/last.pt`

The checkpoint kept the synthetic MemoryOS generalization result while adding
early exit:

| Eval | Full-depth hits | Halted hits | Full steps | Halted steps | Hit changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9-case hard probe | 6/9 | 6/9 | 2 x 9 | 1 x 9 | 0 |
| 12-case held-out probe | 9/12 | 9/12 | 2 x 12 | 1 x 12 | 0 |

Outputs:

- `runs/eval/memory_reasoning_halt_preserve_full_depth_32tok.jsonl`
- `runs/eval/memory_reasoning_halt_preserve_enabled_32tok.jsonl`
- `runs/eval/memory_reasoning_heldout_halt_preserve_full_depth_32tok.jsonl`
- `runs/eval/memory_reasoning_heldout_halt_preserve_enabled_32tok.jsonl`

Interpretation: the previous regression was not caused by early halt itself; it
was caused by fine-tuning the residual path on clean LM text. Freezing the
residual path and training only the halt head preserves the 12-case held-out
MemoryOS score (`9/12`) while reducing recursive outer depth from 2 to 1.

## [2026-04-29] train | Bounded student-LM 2K continuation

Added a longer bounded student-LM continuation config:

- Config: `configs/qwen35_2b_4090_bounded_residual_studentlm_2k.yaml`
- Init: `runs/qwen35_2b_4090_bounded_residual_probe/last.pt`
- Output: `runs/qwen35_2b_4090_bounded_residual_studentlm_2k/last.pt`
- Eval: `runs/qwen35_2b_4090_bounded_residual_studentlm_2k/evals/donor_scale_sweep_qtrm1p0_final.jsonl`

The important change from earlier student-LM probes is
`qtrm_logits_scale: 1.0`, while keeping donor logits fixed at `1.0` during
training and retaining normalized residual gating with a `0.05` gate floor.
This made the student-only loss move much faster: the run started around
`student_lm=11.05` and reached the noisy `5.7-6.2` range by the second half of
the 2K-step run.

Final donor-scale sweep:

| donor_logits_scale | Behavior |
| --- | --- |
| `1.0` | fluent Korean/English greedy text; residual gate about `0.063`; no donor argmax shift |
| `0.5` | still fluent; no donor argmax shift |
| `0.25` | still fluent; no donor argmax shift |
| `0.0` | collapsed to `world of the world` / repeated chapter-pattern text |

Interpretation: the residual safety rail now works at higher QTRM scale, but
donor-logit detach is still not solved. Low-donor fluency is still carried by
the donor logits, and QTRM-only generation remains unstable. The next training
step should be on-policy distillation/GKD-style: sample low-donor or QTRM-only
continuations, have the donor/teacher score or correct those continuations, and
train on the actual student distribution instead of only teacher-forced clean
text.

## [2026-04-29] research | Donor-free collapse solution search

Web search for the `donor_logits_scale=0.0` collapse pointed to a clear
training-distribution problem, not just a step-count problem. The most relevant
references are:

- GKD / on-policy distillation: `https://arxiv.org/abs/2306.13649`
- 2026 OPD failure/success recipe: `https://arxiv.org/abs/2604.13016`
- OPD survey: `https://arxiv.org/abs/2604.00626`
- DistiLLM: `https://arxiv.org/abs/2402.03898`
- DistiLLM-2 contrastive distillation:
  `https://arxiv.org/abs/2503.07067`
- Residual KD: `https://openreview.net/forum?id=Dh6KxUxG20`
- Concrete Score Distillation:
  `https://openreview.net/forum?id=bZBJFrxH1H`
- Distillation scaling laws: `https://arxiv.org/abs/2502.08606`
- Capacity-gap law: `https://arxiv.org/abs/2311.07052`
- Minitron / Sheared LLaMA as fallback compression paths:
  `https://github.com/NVlabs/Minitron`,
  `https://arxiv.org/abs/2310.06694`

Decision: do not keep extending teacher-forced student-LM training as the only
path. The next QTRM detach experiment should collect QTRM's own bad low-donor
rollouts, score them with Qwen donor feedback, and train on those
student-visited states. Use teacher/reference continuations as positives and
collapsed QTRM continuations as negatives. Add repeated n-gram unlikelihood as
a local anti-collapse auxiliary loss, but keep the main fix
GKD/OPD/DistiLLM-style because the real failure is exposure bias.

## [2026-04-29] decision | Residual adapter validation reset

Clarified the current architecture name and validation order. The current QTRM
should be treated as a donor-backed residual adapter, more specifically a
donor-backed residual cognitive adapter:

```text
Qwen hidden states -> QTRM workspace/core/coda -> residual logits
Qwen donor logits  -> base language policy
fused logits       -> donor logits + bounded QTRM residual
```

The donor-free probes skipped ahead to a standalone-student claim. That was a
useful stress test, but it is not the next required proof. The immediate proof
is residual-adapter usefulness: with donor logits intact, QTRM residual should
beat donor-only on evidence-sensitive MemoryOS tasks while preserving donor
fluency. Only after that gate should low-donor and donor-free student behavior
be promoted again.

Next action: rerun MemoryOS hard/held-out ablation with donor-only versus QTRM
residual, plus workspace/core disabled modes where applicable.

## [2026-04-29] research | LM2 memory reference added

Added LM2 as a memory-architecture reference:

- Paper: `https://arxiv.org/abs/2502.06049`
- Repo: `https://github.com/convergence-ai/lm2`

Interpretation: LM2 is relevant to QTRM, but not as the direct fix for
`donor_logits_scale=0.0` language collapse. It supports the residual-adapter
memory direction: keep the base Transformer path intact, add a complementary
memory pathway, connect it through cross-attention and gates, and verify that
general ability is not degraded. This maps well to QTRM's near-term goal as a
donor-backed residual cognitive adapter with explicit MemoryOS/workspace
ablation gates.

## [2026-04-29] eval | Residual adapter proof package

Added a fixed proof package for the current near-term claim:

- Script: `scripts/109_build_residual_adapter_proof.py`
- Module: `src/qtrm_mm/eval/residual_adapter_proof.py`
- Markdown: `docs/wiki/decisions/residual-adapter-proof.md`
- JSON: `docs/wiki/decisions/residual-adapter-proof-summary.json`

The package summarizes existing eval JSONL files instead of rerunning training:

| Eval | Donor-only | QTRM residual | Delta |
| --- | ---: | ---: | ---: |
| hard memory probe | 5/9 | 9/9 | +4 |
| held-out memory probe | 6/12 | 9/12 | +3 |
| aggregate | 11/21 | 18/21 | +7 |

The aggregate task-family delta shows the current gain is concentrated in
abstention: donor-only `0/7`, QTRM residual `7/7`. Conflict and multi-hop are
currently tied. This supports the residual-adapter usefulness claim on current
MemoryOS probes, not a donor-free standalone-LM claim.

## [2026-04-30] eval | Expanded residual adapter gate

Added and ran the 72-case expanded held-out MemoryOS gate:

- Cases: `data/eval/memory_reasoning_heldout_expanded_72.jsonl`
- Builder: `scripts/110_build_expanded_memory_reasoning_heldout.py`
- Runner: `scripts/111_run_residual_adapter_expanded_gate.sh`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl`

The expanded gate is balanced: 24 conflict, 24 multi-hop, and 24 abstention
cases. It is ID-disjoint from the hard probe, original held-out probe, and
synthetic training cases. The actual MemoryOS retrieval/rerank path retrieved a
target for every case, with all-target retrieval on 58/72 cases per mode.

Result:

| Eval | Donor-only | QTRM residual | Delta |
| --- | ---: | ---: | ---: |
| expanded held-out memory probe | 26/72 | 49/72 | +23 |

Updated the residual-adapter proof package to include this gate. Aggregate proof
is now donor-only `37/93`, QTRM residual `67/93`, delta `+30`. The gain is still
largest on abstention (`1/31 -> 25/31`), but the expanded gate also shows
nonzero multi-hop improvement (`11/30 -> 16/30`). The remaining weakness is
multi-hop retrieval/answer composition, especially cases where not all linked
targets were retrieved.

## [2026-04-30] eval | Expanded workspace/core ablation

Added and ran the expanded workspace/core ablation gate:

- Runner: `scripts/112_run_expanded_workspace_core_ablation.sh`
- Proof builder: `scripts/113_build_expanded_ablation_proof.py`
- Module: `src/qtrm_mm/eval/architecture_ablation_proof.py`
- Markdown: `docs/wiki/decisions/expanded-workspace-core-ablation.md`
- JSON: `docs/wiki/decisions/expanded-workspace-core-ablation-summary.json`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_synth_generalization_s050.jsonl`

Result on the same 72-case expanded MemoryOS gate:

| Mode | Hits | Drop vs full residual |
| --- | ---: | ---: |
| `qtrm_residual_with_evidence` | 49/72 | 0 |
| `qtrm_workspace_off_with_evidence` | 49/72 | 0 |
| `qtrm_core_off_with_evidence` | 49/72 | 0 |

Task-family drops are also zero: abstention `18/24`, conflict `20/24`, and
multi-hop `11/24` for all three modes.

The generated completions are also identical across full residual,
workspace-off, and core-off for all 72 cases. Interpretation: the current
expanded-gate gain is real versus donor-only, but this ablation does not
localize the gain to the latent workspace or recursive core. The measured
behavior is currently consistent with a donor-hidden residual side path that can
solve the synthetic MemoryOS prompts even when workspace/core are disabled. The
next architecture task is to add a stricter causality gate: workspace-only
memory injection, coda-off/residual-head ablations, and training losses that
force evidence into workspace/core states before claiming latent workspace or
recursive-core reasoning.

## [2026-04-30] eval | Strict causality ablation

Added strict causality ablation modes and ran them on the same expanded 72-case
MemoryOS gate:

- `qtrm_coda_off_with_evidence`
- `qtrm_residual_head_off_with_evidence`
- `qtrm_donor_hidden_off_with_evidence`
- `qtrm_workspace_only_with_evidence`
- Runner: `scripts/114_run_expanded_strict_causality_ablation.sh`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_synth_generalization_s050.jsonl`

Result:

| Mode | Hits | Drop vs full residual |
| --- | ---: | ---: |
| `qtrm_residual_with_evidence` | 49/72 | 0 |
| `qtrm_coda_off_with_evidence` | 39/72 | +10 |
| `qtrm_residual_head_off_with_evidence` | 26/72 | +23 |
| `qtrm_donor_hidden_off_with_evidence` | 49/72 | 0 |
| `qtrm_workspace_only_with_evidence` | 49/72 | 0 |

Interpretation: the residual head is the main measured source of the current
gain over donor-only, and coda contributes. Direct projected donor hidden states
are not the cause on this gate: removing them keeps `49/72` and identical
completions. `workspace_only` also keeps `49/72`, but because `workspace_off`
also keeps `49/72`, this still does not prove latent-workspace causality. The
next proof must make the task impossible without workspace state, for example
by putting evidence only into workspace-side memory tokens while hiding it from
the normal prompt/donor path.

## [2026-04-30] architecture | Gated latent workspace memory

Added the first LM2/G-MemLLM-inspired gated latent memory path inside
`LatentWorkspace`.

Implementation:

- `src/qtrm_mm/workspace.py` now supports `workspace_memory_gate_enabled` and a
  conservative `workspace_memory_gate_init_bias`.
- Each workspace layer can apply update/reset/candidate gates after context
  cross-attention, letting latent slots preserve, reset, or overwrite state.
- `src/qtrm_mm/qtrm_model.py` returns `workspace_update_gate_mean` telemetry.
- `disable_workspace_memory_gate=True` supports runtime causal ablation.
- `scripts/95_eval_memory_retrieval.py` and
  `scripts/114_run_expanded_strict_causality_ablation.sh` include
  `qtrm_workspace_gate_off_with_evidence`.
- `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` enables the gated
  workspace for the next current-architecture probe.

Reference ingest:

- cloned `references/official/lm2` at `5f56b197b735`;
- cloned `references/official/lightmem` at `b11eccd23c7c`;
- downloaded LM2, G-MemLLM, LightMem, MemCoT, ATLAS, and MIRAS PDFs under
  `references/papers`.

Interpretation: this is not full LM2, G-MemLLM, Titans, ATLAS, or MSA. It is a
small, ablatable gated-memory lane for in-context workspace state. The claim is
valid only if future `workspace_gate_off` runs drop below full residual on a
held-out MemoryOS gate.

## [2026-04-30] eval | Layered MemoryOS scoring

Added stricter MemoryOS answer scoring:

- `hit`: backward-compatible permissive normalized alias containment;
- `exact_match`: canonical short answer exactly equals an alias;
- `normalized_exact`: canonical short answer equals an alias after
  case/punctuation/spacing normalization;
- `normalized_contains`: old loose alias-containment behavior;
- `unknown_correct`: UNKNOWN-specific correctness;
- `needs_human_audit` and `audit_reasons`: queue loose hits, UNKNOWN
  repetitions, and misses for human or LLM-as-judge review.

Implementation:

- `src/qtrm_mm/eval/memory_retrieval.py`
- `scripts/95_eval_memory_retrieval.py`
- `scripts/116_rescore_memory_eval.py`
- `docs/wiki/concepts/memory-evaluation-metrics.md`

Rescored the gated-workspace outputs without rerunning model generation:

| Mode | Permissive hits | Exact / normalized exact | Audit items |
| --- | ---: | ---: | ---: |
| `donor_only_with_evidence` | 26/72 | 9/72 | 63/72 |
| `qtrm_residual_with_evidence` | 49/72 | 24/72 | 48/72 |
| `qtrm_workspace_gate_off_with_evidence` | 49/72 | 24/72 | 48/72 |

Interpretation: QTRM residual still improves over donor-only, but exact
short-answer quality is much lower than permissive hit rate. Architecture claims
should now require improvement in permissive hit rate, normalized exact rate,
and audit rate together.

## [2026-04-30] decision | Perceiver/Q-Former workspace status

Reviewed the current latent-workspace lineage against newer memory references.
Decision: do not delete Perceiver/Q-Former/OpenFlamingo-style learned slots, but
demote them to connector/bottleneck baseline.

Current classification:

| Axis | Role |
| --- | --- |
| Perceiver / Q-Former / PerceiverResampler | compact learned-slot connector |
| Qwen2.5-VL-style merger | modern simple multimodal connector reference |
| LM2 / G-MemLLM | internal gated-memory causality target |
| MSA | large external sparse-memory routing target |
| LightMem / MemCoT | external MemoryOS filtering and iterative search target |

Added `docs/wiki/decisions/latent-workspace-prior-decision.md` and updated the
workspace/memory concept pages. The next architecture step should not be "make
the Perceiver stack deeper"; it should be workspace-side evidence injection
where `workspace_off` and `workspace_gate_off` actually reduce score.

## [2026-04-30] architecture | Workspace evidence-only path

Implemented the stricter workspace evidence path.

Previous MemoryOS evals put retrieved evidence directly into the donor-visible
prompt. That made the residual gain useful, but not a proof of latent workspace
reasoning. The new path keeps the visible prompt to task/question text and
encodes retrieved evidence separately as workspace-side donor hidden states.

Implementation:

- `src/qtrm_mm/qtrm_model.py` accepts `workspace_text_states` and
  `workspace_attention_mask`.
- `disable_workspace_memory_context=True` removes this separate evidence memory
  path for ablation.
- `src/qtrm_mm/multimodal_projector.py` supports feature masks for evidence
  memory tokens.
- `scripts/95_eval_memory_retrieval.py` adds
  `--evidence-injection workspace`.
- `qtrm_workspace_memory_off_with_evidence` now measures whether retrieved
  evidence is actually flowing through workspace memory.
- `scripts/117_run_workspace_evidence_path_probe.sh` runs the expanded 72-case
  MemoryOS gate in this stricter mode.
- `src/qtrm_mm/data/jsonl_dataset.py` can split MemoryOS supervised rows into
  visible prompt tokens plus `workspace_input_ids`.
- `src/qtrm_mm/training/train.py` encodes those workspace ids through the frozen
  donor and forwards them as `workspace_text_states`.
- `configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml` and
  `scripts/118_run_workspace_evidence_path_train.sh` define the first training
  run for this path.

Boundary: this is an architecture/eval-path change. Existing checkpoints were
not trained with evidence hidden from the prompt, so this probe is expected to
be a causality harness first and a performance improvement only after retraining
against the new path.

## [2026-04-30] concept | Internalized context engineering

Added a wiki source/concept pair for QTRM as internalized context engineering.

Definition: QTRM is not replacing RAG. It is trying to move selected parts of
external context engineering into trainable, gated, auditable memory routes.

Similar paper families:

- Context Engineering Survey names the broader field.
- ACE is the closest external-context counterpart: it evolves structured
  contexts without changing model weights.
- REALM, RETRO, DSI, and Atlas show retrieval entering model behavior rather
  than staying as prompt stuffing.
- LM2 and G-MemLLM are the closest memory-lane and frozen-backbone gated-memory
  references.
- MSA is the closest current reference for sparse memory routing at 100M-token
  scale.
- LightMem and MemCoT keep the external MemoryOS side relevant through memory
  filtering, consolidation, and iterative search.

QTRM staged path:

```text
MemoryOS retrieval + rerank
-> ACE-style evolving context playbooks
-> workspace-only evidence injection
-> gated core context injection
-> LM2/G-MemLLM-style internal memory lane
-> MSA-style sparse memory routing
```

Decision: keep retrieval quality, workspace causality, core-context causality,
and memory-gate causality as separate ablation gates. Do not claim full
end-to-end memory until each stage passes its own off-switch test.

## [2026-04-30] architecture | Gated core context injection

Downloaded/collected the internalized-context reference set:

- context-engineering and ACE PDFs;
- REALM, RETRO, DSI, Atlas, Retro-Li, and retrieval-enhanced generalization
  PDFs;
- persistent latent memory and LatentMem PDFs;
- cloned ACE, Atlas, REALM sparse checkout, RETRO-pytorch, labml RETRO,
  Retro-Li, DSI-transformers, MemoryLLM, and MemGen.

Implemented the next architecture step after workspace-only evidence:

- `QTRMConfig.core_context_enabled`;
- `QTRMConfig.core_context_gate_init_bias`;
- `QTRMRecursiveCore` gated cross-attention from `z_l`/`z_h` to prelude context;
- `disable_core_context` runtime ablation;
- `core_context_gate_mean` telemetry;
- `qtrm_core_context_off_with_evidence` eval mode.

Reasoning: this follows the LM2/G-MemLLM principle of preserving the base path
while adding a complementary gated memory/context route. It also follows
RETRO/Atlas by allowing retrieved evidence to influence architectural attention
paths, not only prompt strings.

Required proof: after training with `core_context_enabled: true`,
`qtrm_core_context_off_with_evidence` must drop below
`qtrm_residual_with_evidence`. If it does not drop, the path exists but has not
become behaviorally important.

## [2026-04-30] training | Human-like loss first slice

Added the first general preference-loss path and kept repetition handling as an
optional guard:

- JSONL rows with `prompt`, `chosen`, and `rejected` now train the chosen answer
  as SFT and score the rejected answer through a second model forward pass.
- `sequence_average_logprob` computes sequence-level average log-probs.
- `simpo_margin_loss` trains chosen/rejected answer pairs with an explicit
  margin and optional row-level `preference_weight`/`confidence`.
- `loss_preference_weight`, `preference_beta`, and `preference_margin` wire this
  into normal training.
- `scripts/120_run_workspace_evidence_preference_train.sh` runs the
  workspace-evidence preference probe on
  `data/filtered/memory_self_improvement_preferences_analysis.jsonl`.
- `scripts/121_eval_preference_pairs.py` evaluates whether a checkpoint
  actually assigns higher sequence average log-prob to `chosen` than
  `rejected`, reporting raw and weighted accuracy plus margin pass rate.
- `repetition_unlikelihood_loss` still exists, but only as a local guard for
  confident adjacent repeat candidates when the previous token is not the
  current gold target.
- `loss_repeat_unlikelihood_weight` wires the repeat guard into normal training
  while keeping the default disabled.
- `configs/qwen35_2b_4090_workspace_evidence_repeatguard_s050.yaml` defines a
  conservative next probe with repeat unlikelihood weight `0.02`.

Decision boundary: this is not a claim that QTRM now has human cognition. It is
the first practical step toward behavior that learns from bad generations:
ordinary CE for correct answers, preference loss against rejected generations,
optional unlikelihood for repeated failure modes, and a future
process-supervision path for verified correction traces.

## [2026-04-30] training | Preference-only early stop

Ran the first workspace-evidence preference-only probe from
`runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt` using
`memory_self_improvement_preferences_analysis.jsonl`.

Observed signal:

- preference margins improved very quickly on training pairs;
- by step 25, the logged margin had already flipped positive;
- by steps 300, 400, and 500, the live diagnostic still produced
  `Ilya Chen Chen Chen...` with high repeated-2gram rate;
- `repeat_ul` was logged but had zero training weight in the preference-only
  config.

Decision: preference loss alone is not sufficient for this architecture's
current generation stability. The next probe must train the same pairwise
chosen/rejected objective together with a weak repetition unlikelihood guard.

Added
`configs/qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050.yaml`
and `scripts/122_run_workspace_evidence_preference_repeatguard_train.sh` for
that balanced probe.

## [2026-04-30] architecture | Counterfactual workspace causality probe

Added a research-driven architecture-debugging cycle for the
retrieval-found-but-wrong failure:

- source map and decision ledger:
  `docs/wiki/decisions/research-driven-next-architecture.md`;
- counterfactual data builder:
  `scripts/125_build_workspace_counterfactual_preferences.py`;
- generated data:
  `data/filtered/memory_self_improvement_preferences_workspace_counterfactual.jsonl`;
- training config:
  `configs/qwen35_2b_4090_workspace_evidence_counterfactual_s050.yaml`;
- runner:
  `scripts/126_run_workspace_evidence_counterfactual_train.sh`;
- checkpoint:
  `runs/qwen35_2b_4090_workspace_evidence_counterfactual_s050/last.pt`;
- proof:
  `docs/wiki/decisions/workspace-evidence-counterfactual-trained-ablation.md`.

Mechanism: for the same visible prompt and chosen answer, train true workspace
evidence to score higher than shuffled/counterfactual workspace evidence. This
tests whether workspace evidence actually changes the chosen answer log-prob.

Result:

- pair preference still passes: `preference_accuracy=0.909`,
  `margin_pass_rate=0.909`;
- quick retrieval eval fails: `qtrm_residual_with_evidence=0/4` even though
  `retrieved_target_rate=1.0`;
- workspace/core/memory ablations do not reduce hit rate.

Conclusion: counterfactual workspace loss is useful instrumentation, but the
current architecture still lacks a causal evidence-to-answer bottleneck. The
next implementation target is an Evidence-Bottleneck Decoder or a
verifier-controlled answer gate, not more blind preference tuning.

## [2026-04-30] architecture | Logical-causal evidence bottleneck

Added the next falsifiable architecture probe for the
retrieval-found-but-wrong failure.

New source map and papers:

- `docs/wiki/sources/logical-causal-trust.md`;
- downloaded ProoFVer, FOLIO, LINC, NL2LOGIC, insufficient-evidence,
  RAGONITE, CF-RAG, and FaithfulRAG PDFs under
  `references/papers/logical-causal-trust/`;
- cloned `references/official/linc`, `references/official/folio`, and
  `references/official/nl2logic`.

Implemented mechanism:

- support/refute/missing verifier heads over the recursive workspace state;
- causal evidence gate;
- optional suppression of QTRM residual logits when no workspace evidence is
  present;
- logical evidence and causal gate losses using true/counterfactual workspace
  pairs;
- ablation mode `qtrm_evidence_bottleneck_off_with_evidence`.

Experiment entry point:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/127_run_logical_causal_bottleneck_train.sh
```

Acceptance gate: this is not accepted until full evidence beats the previous
`0/4` quick result and memory/gate-off ablations cause a measurable answer
change.

## [2026-04-30] architecture | LeWM core trajectory prediction

Mapped the official LeWorldModel implementation
`references/official/le-wm@bf04d3e8c375` onto QTRM's TRM-style recursive core.

Mechanism:

- keep the existing token-latent LeWM/JEPa head;
- add optional LeWM prediction over recursive `z_H` core trajectories;
- build a simple action trace for the first probe:
  `OBSERVE` or `RETRIEVE` -> `VERIFY` -> `ANSWER`;
- train with `MSE(predicted next z_H, next z_H) + SIGReg(z_H trajectory)`;
- expose `core_world_model` metric in the normal training loop.

Added:

- `core_world_model_enabled` model config fields;
- `loss_core_world_model_weight` train config field;
- `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`;
- `scripts/128_run_lewm_core_world_model_probe.sh`;
- tests for action routing, core-world-model outputs, and loss metrics.

This is still not a full LeWorldModel reproduction: it uses QTRM text/workspace
latents, not pixel trajectories, and the action trace is fixed probe metadata.

## [2026-04-30] architecture | Canonical active-vs-disabled ledger

Added a canonical architecture matrix so implemented features are no longer
implicitly treated as active architecture.

New/updated docs:

- `docs/wiki/architecture/canonical-architecture-matrix.md`
- `docs/wiki/architecture/qtrm-forward-pass.md`
- `docs/wiki/architecture/paper-diagram-prompts.md`

Key boundary recorded:

- The current canonical probe is
  `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`.
- Active path includes donor logits, bounded residual gate, workspace-only
  evidence, workspace memory gate, core context gate, evidence bottleneck,
  counterfactual/preference/repeat/donor-KL losses, and core LeWM trajectory
  loss.
- Token-level JEPA, controller aux heads, core halting, donor annealing,
  multimodal scaffold, external reranker, symbolic verifier, and persistent
  neural memory are implemented or referenced but not active canonical claims.
- Donor-free generation is not yet equivalent to donor-backed gated residual
  fusion because the gate applies to `qtrm_residual_logits` in donor-fusion
  mode.

## [2026-05-01] architecture | OpenMythos-style core-to-text path

Failure ledger:

- Failure: after the evidence-bottleneck fix, ordinary prompts no longer forced
  QTRM to donor-only, but recursive `z_H` could still be weakly causal because
  the prelude text context had an easy bypass into coda.
- Evidence: `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml` used
  `n_coda_layers: 2` with inherited `attn_every: 4`, so no explicit coda
  attention layer was guaranteed. The model could rely on donor/text logits and
  leave the latent recurrent prefix underused.
- Prior checked: `references/official/openmythos@8c68c1f`. It is speculative,
  not official Claude evidence, but its useful architectural invariant is
  Prelude -> recurrent state -> Coda without an unchanged hidden-state bypass.
- Recommended candidate: keep QTRM donor-backed and latent-workspace based, but
  add a gated `z_H -> text_context` cross-attention bridge before coda.

Implemented:

- `QTRMConfig.core_to_text_enabled`
- `QTRMConfig.core_to_text_gate_init_bias`
- `QTRMConfig.core_to_text_gate_min`
- `QTRMMultimodalModel.forward(..., disable_core_to_text=True)`
- output telemetry `core_to_text_gate_mean`
- v2 config `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`
- runner `scripts/135_run_hf_first_wave_warmup_v2.sh`

The v2 config also sets `coda_attn_every: 1`, so text tokens can explicitly
attend to the recurrent workspace prefix. This is a conservative adaptation of
OpenMythos's recurrent-output-to-coda idea, not an OpenMythos clone.

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_model_config.py \
  tests/test_core_halting.py \
  tests/test_hf_distill_training_mix.py \
  tests/test_jsonl_dataset_supervised.py
```

Result: `39 passed`.

Next gate: train the v2 warmup checkpoint, then compare residual, donor-only,
core-off, and `disable_core_to_text` evals. Acceptance requires residual logits
and generation behavior to change under the core-to-text ablation on prompts
where latent recurrence is expected to matter.

## [2026-05-01] tooling | Qwen-Scope donor feature logger

Added Qwen-Scope tooling for donor residual-stream analysis.

Source:

- `docs/wiki/sources/qwen-scope.md`
- Hugging Face collection: <https://huggingface.co/collections/Qwen/qwen-scope>
- Qwen3.5-2B Base SAE repo:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100>

Implemented:

- `src/qtrm_mm/qwen_scope.py`
  - loads official SAE tensor dicts;
  - validates `W_enc`, `W_dec`, `b_enc`, `b_dec`;
  - extracts top-k feature ids/values from donor residual states;
  - emits JSON-ready feature records;
  - uses `attention_mask` for last-nonpad token selection so padded EOS tokens
    are not mistaken for prompt endings.
- `scripts/136_qwen_scope_probe.py`
  - loads Qwen donor with `AutoModelForCausalLM`;
  - downloads selected Qwen-Scope SAE layers;
  - writes prompt/layer/token feature records as JSONL.
- tests:
  - `tests/test_qwen_scope.py`
  - `tests/test_qwen_scope_probe_script.py`

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py
```

Initial result: `5 passed`.

Follow-up execution:

- v2 warmup finished and saved
  `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`;
- checkpoint NaN/Inf scan: `floating_tensors=284`, `bad_tensors=0`;
- first Qwen-Scope probe showed a padding bug: short prompts recorded
  `<|endoftext|>` as the last token;
- fixed `qwen_scope_feature_records(..., attention_mask=..., token_position="last_nonpad")`;
- reran the probe:
  `runs/qwen_scope/qwen35_2b_base_layers_0_12_23.jsonl`;
- records: 9 rows, 3 prompts x 3 layers.

Post-fix verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py \
  tests/test_model_config.py \
  tests/test_hf_distill_training_mix.py
```

Result: `35 passed`.

Current boundary: Qwen-Scope is not an architecture replacement. It is a donor
diagnostics instrument for repetition, evidence-missing, unsupported-answer,
and donor-vs-QTRM residual feature analysis. Do not run the heavy probe while
the v2 QTRM training process is occupying GPU memory.

## [2026-05-01] analysis | Qwen-Scope repeat-vs-normal probe

Standardized the one-off Qwen-Scope comparison into reusable tooling:

- `compare_qwen_scope_feature_groups` in `src/qtrm_mm/qwen_scope.py`;
- `scripts/137_compare_qwen_scope_groups.py`;
- `tests/test_qwen_scope_compare_script.py`.

Ran a curated repeat-vs-normal probe:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
uv run python scripts/136_qwen_scope_probe.py \
  --device cuda --load-in-4bit --dtype bfloat16 --top-k 50 \
  --layer 0 --layer 12 --layer 23 \
  --prompt "Explain quantum entanglement in simple terms." \
  --prompt "양자 컴퓨팅이란 무엇인가요?" \
  --prompt "Determine whether the claim is supported by the evidence. Claim: The Eiffel Tower is in Berlin." \
  --prompt "양자 컴퓨팅이란 무엇인가요? Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze" \
  --prompt "Explain quantum entanglement in simple terms. This is a crucial role in the world of the world of the world of the world of the world" \
  --prompt "Explain quantum entanglement in simple terms. This is a crucial role in the world of the world of the world of the world of" \
  --out runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_layers_0_12_23.jsonl
```

Then compared normal indices `0,1,2` against repeated indices `3,4,5`:

```bash
PYTHONPATH=src uv run python scripts/137_compare_qwen_scope_groups.py \
  --input runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_layers_0_12_23.jsonl \
  --out runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_summary.json \
  --normal 0,1,2 \
  --repeat 3,4,5 \
  --feature-limit 20 \
  --top-output 15
```

Result summary:

- layer 0: no shared repeat top features; repeat-shared-not-normal examples
  include `16910`, `7605`;
- layer 12: strongest small-set separation; repeat common top features are
  `847`, `2761`, `22167`, `24725`, `25296`, `26397`;
- layer 23: repeat common top features are `29838`, `30452`, `31860`, with
  repeat-shared-not-normal examples `2248`, `12018`, `5121`.

Interpretation: these are candidate diagnostic features, not proven causal
repetition features. The next gate is to run QTRM-generated failures through the
same logger and compare generated repeats against normal completions.

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py \
  tests/test_qwen_scope_compare_script.py
```

Result: `9 passed`.

## [2026-05-01] analysis | Qwen-Scope on actual QTRM generated repeats

Ran v2 QTRM generation using:

- config:
  `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`
- checkpoint:
  `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`
- output:
  `runs/qwen_scope/qtrm_v2_generated_for_scope.jsonl`

Generation split:

- normal/low-repeat: prompt indices `0,1`;
- repeat failures: prompt indices `2,3,4`.

Observed failures:

- index `2`: claim/evidence prompt repeats claim/evidence statements,
  `repeated_2gram_rate=0.762`, `repeated_3gram_rate=0.742`;
- index `3`: math prompt begins a second copied problem,
  `repeated_2gram_rate=0.254`;
- index `4`: Korean history prompt repeats multiple-choice options,
  `repeated_2gram_rate=0.175`.

Ran Qwen-Scope over the generated texts:

- prompt file:
  `runs/qwen_scope/qtrm_v2_generated_texts_for_scope.txt`;
- SAE output:
  `runs/qwen_scope/qtrm_v2_generated_layers_0_12_23.jsonl`;
- comparison summary:
  `runs/qwen_scope/qtrm_v2_generated_repeat_vs_normal_summary.json`.

Overlap between curated repeated-string candidates and actual QTRM-generated
repeat candidates:

- layer 0: none;
- layer 12: `847`;
- layer 23: `29838`, `31860`.

Interpretation: Qwen-Scope now gives a concrete repeat-state diagnostic signal
on actual generated QTRM failures, especially in late donor layer 23. This is
not causal proof yet. The next falsifiable step is a larger generated-failure
set and a detector that predicts high repeated-ngram rate from SAE candidate
activations before any architecture/steering changes are considered.

## [2026-05-01] architecture | Root-architecture causality gate

Added an automated big-structure gate:

- module: `src/qtrm_mm/eval/root_architecture_gate.py`;
- runner: `scripts/148_build_root_architecture_gate.py`;
- tests: `tests/test_root_architecture_gate.py`;
- output: `docs/wiki/decisions/root-architecture-causality-gate.md`;
- summary: `docs/wiki/decisions/root-architecture-causality-gate-summary.json`.

Result on
`runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_counterfactual_32tok_trained_s050.jsonl`:

| Check | Result |
| --- | --- |
| status | `rejected` |
| baseline | `qtrm_residual_with_evidence=0/4` |
| workspace memory off | `0/4`, same completions `4/4` |
| core context off | `0/4`, same completions `4/4` |
| failed checks | baseline no successes, no critical causal drop, critical ablations match baseline identity |

Interpretation: this checkpoint does not prove that the latent workspace/core
path is causally necessary. The next architecture work should move answer
formation onto a forced workspace/evidence bottleneck instead of adding another
local loss or reranker.

## [2026-05-01] experiment | Logical-causal bottleneck quick gate

Ran the implemented logical-causal evidence bottleneck probe:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/127_run_logical_causal_bottleneck_train.sh
```

Artifacts:

- checkpoint: `runs/qwen35_2b_4090_logical_causal_bottleneck_s050/last.pt`;
- pair eval: `runs/eval/logical_causal_bottleneck_pair_eval_s050.jsonl`;
- memory eval:
  `runs/eval/memory_reasoning_heldout_expanded_logical_causal_bottleneck_32tok_trained_s050.jsonl`;
- ablation proof:
  `docs/wiki/decisions/logical-causal-bottleneck-trained-ablation.md`;
- root gate:
  `docs/wiki/decisions/logical-causal-bottleneck-root-gate.md`.

Results:

| Metric | Value |
| --- | ---: |
| preference accuracy | `0.727` |
| margin pass rate | `0.727` |
| memory residual hit rate | `0/4` |
| retrieved target rate | `1.0` |
| target recall mean | `0.917` |
| root gate | `rejected` |

Diagnostic generations at steps 100, 200, and 300 still repeated the hidden
workspace-evidence phrase and leaked `<think>` on the second prompt. The memory
eval showed evidence copying and prompt continuation instead of short answers.

Interpretation: the bottleneck is not completely inert, because
`evidence_bottleneck_off` changes some completions. But it is not a useful
causal answer path yet, because the full residual baseline remains `0/4`.
Next fix: force a visible answer channel during MemoryOS decoding/training and
then retest the same root gate.

## [2026-05-01] eval | MemoryOS guarded decoding

Added the same runtime guards used by `92_eval_qtrm_logits.py` to
`scripts/95_eval_memory_retrieval.py`:

- `--suppress-visible-reasoning-tokens`;
- `--no-repeat-ngram-size`;
- runner env in `scripts/117_run_workspace_evidence_path_probe.sh`:
  `SUPPRESS_VISIBLE_REASONING=1`, `NO_REPEAT_NGRAM_SIZE=2`.

Guarded eval command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  CONFIG=configs/qwen35_2b_4090_logical_causal_bottleneck_s050.yaml \
  CHECKPOINT=runs/qwen35_2b_4090_logical_causal_bottleneck_s050/last.pt \
  MAX_CASES=4 SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 \
  OUT=runs/eval/memory_reasoning_heldout_expanded_logical_causal_bottleneck_32tok_guarded_s050.jsonl \
  bash scripts/117_run_workspace_evidence_path_probe.sh
```

Result:

- memory residual hit rate remains `0/4`;
- root gate remains `rejected`;
- `workspace_memory_off` same-completion rate remains `0.75`;
- `core_context_off` same-completion rate remains `1.0`;
- outputs still copy evidence text or continue the prompt.

Conclusion:
the answer-channel runtime guard is useful for generic generation, but it does
not solve MemoryOS answer extraction. The next architecture candidate must train
and/or force a short answer decoder, not just suppress bad tokens.

## [2026-05-01] experiment | Workspace-evidence supervised path

Ran the direct hidden-workspace evidence supervised path:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  MAX_CASES=4 SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 \
  SAVE_EVERY=200 bash scripts/118_run_workspace_evidence_path_train.sh
```

Artifacts:

- checkpoint: `runs/qwen35_2b_4090_workspace_evidence_path_s050/last.pt`;
- eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_trained_s050.jsonl`;
- ablation proof:
  `docs/wiki/decisions/workspace-evidence-path-trained-ablation.md`;
- root gate:
  `docs/wiki/decisions/workspace-evidence-path-trained-root-gate.md`.

Quick-gate result:

| Mode | Hits |
| --- | ---: |
| donor-only | `0/4` |
| QTRM residual | `3/4` |
| workspace off | `3/4` |
| core off | `3/4` |
| workspace memory off | `3/4` |
| evidence bottleneck off | `3/4` |
| residual head off | `0/4` |
| coda off | `2/4` |

Root gate: `rejected`.

Interpretation: this is an important negative result. The model learned a
residual/coda answer prior that improves the quick score, but hidden workspace
evidence is still not causally necessary. More steps on this same architecture
would likely increase prior/memorization and repetition, not prove latent
workspace reasoning.

Next required gate: counterfactual workspace-swap eval. Keep the visible
question fixed, swap only hidden workspace evidence, and require the answer to
change or become `UNKNOWN`. Without that, QTRM is still a residual answer-style
adapter, not a hidden-evidence reasoner.

## [2026-05-01] eval | Workspace counterfactual swap root gate

Added a counterfactual workspace-swap case builder:

- `scripts/build_workspace_counterfactual_eval_cases.py`;
- `tests/test_workspace_counterfactual_eval_cases.py`;
- generated quick set:
  `data/eval/memory_reasoning_heldout_workspace_swap_4.jsonl`.

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml \
  --checkpoint runs/qwen35_2b_4090_workspace_evidence_path_s050/last.pt \
  --cases data/eval/memory_reasoning_heldout_workspace_swap_4.jsonl \
  --evidence-mode all --evidence-injection workspace \
  --max-new-tokens 32 --suppress-visible-reasoning-tokens \
  --no-repeat-ngram-size 2 --no-logit-shift
```

Result:

| Mode | Hits |
| --- | ---: |
| QTRM residual | `3/4` |
| workspace off | `3/4` |
| core off | `3/4` |
| core context off | `3/4` |
| workspace memory off | `3/4` |
| evidence bottleneck off | `3/4` |
| coda off | `4/4` |
| residual head off | `1/4` |

Root gate: `rejected`.

Interpretation: the model can emit `UNKNOWN` on some swapped cases, but the
same completions appear when workspace/core/memory/evidence-bottleneck paths
are disabled. This rejects the current hidden-workspace causality claim more
strongly than the normal held-out eval. The next implementation should force
the answer logits through a selected-evidence/short-answer bottleneck instead
of letting the residual head learn a general answer prior.

## [2026-05-01] architecture | Answer-bottleneck repetition stop

A first `workspace_answer_bottleneck` prototype was trained for 300 steps.
The run was interrupted before completing full root-gate eval because the live
diagnostic showed strong repetition:

- `Answer: 666666...` at step 100;
- `Answer: 555555...` at step 300;
- Korean diagnostic also drifted into repeated structure.

Important diagnosis:
the live training diagnostic did not inject hidden workspace evidence. It was a
plain prompt diagnostic. The new answer-bottleneck path was still active even
when `workspace_memory_present == 0`, so the model could drive QTRM residual
logits from prompt-only latent state and collapse into repeated local tokens.

Architecture fix:

- added `answer_bottleneck_requires_workspace_memory`;
- when enabled, answer-bottleneck residual logits are multiplied by
  `workspace_memory_present`;
- with no workspace memory, QTRM residual is zero and generation falls back to
  donor logits;
- default `scripts/129_run_workspace_answer_bottleneck_train.sh` diagnostics
  are disabled because plain prompt diagnostics are invalid for this hidden
  workspace path.

Next gate:
rerun answer-bottleneck only with workspace-injected evals. If repetition still
appears when workspace evidence is present, escalate again to an explicit
short-answer/stop-governed output channel instead of training longer.

## [2026-05-01] eval | Governed answer-bottleneck and causal-gated follow-up

After adding `answer_bottleneck_requires_workspace_memory`, the
workspace-injected answer-bottleneck eval no longer showed the numeric
`666...` / `555...` repetition pattern. A decode-time short-answer governor was
added to keep scoring focused on the first answer line:

- `scripts/95_eval_memory_retrieval.py --short-answer-governor`;
- `scripts/117_run_workspace_evidence_path_probe.sh` wiring;
- default enabled in `scripts/129_run_workspace_answer_bottleneck_train.sh`;
- tests in `tests/test_memory_eval_script.py`.

Governed quick gate:

- eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_answer_bottleneck_24tok_governed_s050.jsonl`;
- root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-governed-root-gate.md`;
- `qtrm_residual_with_evidence=1/4`;
- `workspace_off=0/4`, `core_off=0/4`,
  `workspace_memory_off=0/4`;
- `core_context_off=1/4` with `4/4` same completions;
- `evidence_bottleneck_off=1/4` with `4/4` same completions.

Interpretation:
the repeated-token failure was fixed as an information-path bug, but answer
selection was not solved. The model still often wrote plausible answer-shaped
text instead of extracting the requested alias from hidden evidence.

A stricter causal follow-up was implemented:

- trainable policy:
  `answer_bottleneck_evidence_only`;
- config:
  `configs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml`;
- runner:
  `scripts/149_run_workspace_answer_bottleneck_causal_train.sh`;
- only answer-bottleneck and logical/causal evidence heads are trained
  (`~1.05M` parameters), initialized from
  `runs/qwen35_2b_4090_workspace_answer_bottleneck_s050/last.pt`;
- counterfactual workspace data:
  `data/filtered/memory_self_improvement_preferences_workspace_counterfactual.jsonl`.

Run result:

- checkpoint:
  `runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt`;
- pair preference eval:
  `preference_accuracy=0.727`, `margin_mean=1.60`;
- training log signal:
  `workspace_margin_logp` stayed near `0.0001`, so true/counterfactual
  workspace evidence was still barely separated;
- normal root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-root-gate.md`,
  status `accepted` only because workspace/core/memory-off drop the single
  successful `UNKNOWN` case;
- swap root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-swap-root-gate.md`,
  status `rejected`;
- swap baseline `qtrm_residual_with_evidence=0/4`;
- `evidence_bottleneck_off` did better than the full residual on swap
  (`1/4` vs `0/4`), which means the evidence gate is not a reliable answer-use
  mechanism yet.

Conclusion:
do not treat this as solved. The answer-bottleneck/gate path can influence
generation, but it still does not implement robust evidence-to-answer
extraction. The next root-structure candidate should add an explicit
evidence-span/copy or evidence-selector reader objective instead of another
scalar loss on free-form residual logits.

## [2026-05-01] architecture | 2026 prompt-conditioned memory reader update

Research update:
2026 memory work does not support defending QTRM as a literal
`prompt`/`workspace` split in the way ordinary user-facing LLM APIs work.
Long-context systems expose visible context windows and retrieval/tool results,
while memory-LM papers separate memory capacity from reasoning compute through
task-conditioned memory reads, gated memory banks, cross-attention, or sparse
memory routing.

Source ledger:

- `docs/wiki/sources/2026-memory-context-architecture.md`
- `docs/wiki/decisions/research-driven-next-architecture.md`
- `docs/wiki/architecture/paper-diagram-prompts.md` Figure 7

Implemented scaffold:

- `model.evidence_span_reader_enabled`
- `train.loss_evidence_span_reader_weight`
- `trainable_param_policy=evidence_span_reader_only`
- `scripts/build_evidence_span_reader_dataset.py`
- `configs/qwen35_2b_4090_evidence_span_reader_s050.yaml`
- `scripts/150_run_evidence_span_reader_train.sh`
- ablation mode `qtrm_evidence_span_reader_off_with_evidence`

Generated data:

```text
data/filtered/memory_reasoning_synth_span_reader.jsonl
rows=432
found=288
no_answer=144
```

Architecture correction:

```text
visible prompt/question states -> query projection
hidden workspace evidence tokens -> start/end span reader + UNKNOWN head
selected evidence span -> answer-only channel -> bounded donor residual
```

This replaces the rejected pattern:

```text
hidden evidence -> latent workspace -> free-form residual logits
```

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_evidence_span_reader_dataset.py \
  tests/test_model_config.py \
  tests/test_losses.py \
  tests/test_training_checkpoint_init.py \
  tests/test_memory_ablation_modes.py \
  tests/test_root_architecture_gate.py -q
# 94 passed

PYTHONPATH=src uv run python -m py_compile \
  scripts/build_evidence_span_reader_dataset.py \
  scripts/95_eval_memory_retrieval.py \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  src/qtrm_mm/losses.py \
  src/qtrm_mm/data/jsonl_dataset.py \
  src/qtrm_mm/training/train.py \
  src/qtrm_mm/eval/root_architecture_gate.py

bash -n scripts/150_run_evidence_span_reader_train.sh
```

Training result:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/150_run_evidence_span_reader_train.sh

init: runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt
out:  runs/qwen35_2b_4090_evidence_span_reader_s050/last.pt
trainable: 1,837,057 params, 9 tensors
evidence_span_reader: 10.3774 -> 0.5896
evidence_span_start_acc: 0.0000 -> 1.0000
evidence_span_end_acc: 0.0000 -> 1.0000
```

Decision:
the prompt-conditioned reader is now a trained probe, not just a scaffold. It
does prove that prompt states can select hidden workspace evidence spans on the
synthetic MemoryOS span task. It does not yet prove final generative answering:
the selected span still must be wired into an answer-only copy/decoder channel
and then tested under normal and workspace-swap gates.

## [2026-05-01] architecture | ASI controller trace-replay training hook

Root-architecture correction:
before trying to make QTRM replace donor language, train the cognitive loop as
an action controller over verified traces.

Implemented:

- stable `Action.id` mapping:
  `OBSERVE=0`, `RETRIEVE_MEMORY=1`, `VERIFY_EVIDENCE=2`, `ANSWER=3`;
- `trace_replay` JSONL rows in `JsonlTextVisionDataset`;
- `action_policy_loss` over `ControllerHeads.action`;
- `train.loss_action_policy_weight`;
- `trainable_param_policy=controller_only`;
- `scripts/155_build_controller_trace_replay.py`;
- `configs/qwen35_2b_4090_controller_trace_s050.yaml`;
- `scripts/155_run_controller_trace_train.sh`.

This is not yet ASI evidence. It is the first causal training bridge:

```text
TypedContextTape / MemoryOS rows
-> trace_replay action targets
-> frozen QTRM/donor representation
-> ControllerHeads.action
-> later ASI causal-loop gate
```

Next gate:
train the controller head, then compare scripted harness, donor+harness,
QTRM-controller+harness, latent-core-off, world-model-off, and verifier-off.

## [2026-05-01] architecture | Controller trace-replay suspicion closed

Suspicion:
the first controller SFT result collapsed toward a single action because the
trace rows did not give the controller a distinguishable causal state. Step 0,
step 1, and step 2 could share almost the same prompt while requiring different
actions.

Fixes:

- trace-replay inputs now include `trace_step`, `state_summary`, and
  `previous_observation`;
- the same action query is repeated near the sequence tail so the pooled
  controller state can see the current decision;
- `ControllerHeads.action` now pools from the final valid text/coda state
  instead of the last latent workspace slot;
- the evaluation script now bounds default evaluation by JSONL line count, so
  finite trace datasets do not silently run forever.

Full held-in trace-replay result:

```text
checkpoint: runs/qwen35_2b_4090_controller_trace_s300/last.pt
data: data/filtered/asi_controller_trace_replay.jsonl
samples: 1296
accuracy: 1.0000
RETRIEVE_MEMORY: 432/432
VERIFY_EVIDENCE: 432/432
ANSWER: 432/432
missing_keys: []
unexpected_keys: []
summary: docs/wiki/decisions/controller-trace-s300-eval-summary.json
```

Held-out contract check:

```text
source cases: data/eval/memory_reasoning_heldout_expanded_72.jsonl
trace data: data/eval/asi_controller_trace_replay_heldout_72.jsonl
summary: docs/wiki/decisions/controller-trace-s300-heldout-eval-summary.json
samples: 216
accuracy: 1.0000
RETRIEVE_MEMORY: 72/72
VERIFY_EVIDENCE: 72/72
ANSWER: 72/72
```

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_jsonl_dataset_supervised.py \
  tests/test_losses.py \
  tests/test_training_checkpoint_init.py \
  tests/test_model_config.py \
  tests/test_controller_trace_replay_script.py \
  tests/test_controller_trace_policy_eval_script.py \
  tests/test_asi_cognitive_loop_contract.py \
  tests/test_asi_causal_loop_gate_script.py -q
# 105 passed

PYTHONPATH=src python3 -m py_compile \
  src/qtrm_mm/agentic/cognitive_loop.py \
  src/qtrm_mm/data/jsonl_dataset.py \
  src/qtrm_mm/losses.py \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  src/qtrm_mm/training/train.py \
  scripts/155_build_controller_trace_replay.py \
  scripts/156_eval_controller_trace_policy.py \
  scripts/154_build_asi_causal_loop_gate.py

bash -n scripts/155_run_controller_trace_train.sh
```

Decision:
this closes the local action-policy wiring bug and shows the explicit
step/state controller contract transfers to unseen MemoryOS eval cases. It
does not prove ASI, open-domain reasoning, answer correctness, or donor-free
language ability. The next gate must test the trained controller inside the
causal-loop harness against donor/scripted baselines and component-off
ablations.

## [2026-05-01] architecture | Stage-1 ASI controller causal-loop gate

Ran the trained controller through an explicit action-policy causal-loop gate.

Artifacts:

- script: `scripts/157_eval_asi_controller_causal_loop.py`;
- summary: `docs/wiki/decisions/asi-controller-causal-loop-s300-summary.json`;
- report: `docs/wiki/decisions/asi-controller-causal-loop-s300.md`;
- metrics for standard gate:
  `docs/wiki/decisions/asi-controller-causal-loop-s300-metrics.json`;
- standard ASI gate:
  `docs/wiki/decisions/asi-causal-loop-gate.md`;
- standard ASI gate summary:
  `docs/wiki/decisions/asi-causal-loop-gate-summary.json`.

Held-out action-policy metrics:

```text
qtrm_harness:               1.0000
qtrm_latent_core_off:       0.7037
qtrm_workspace_off:         0.6667
qtrm_workspace_memory_off:  1.0000
qtrm_core_to_text_off:      1.0000
samples:                   216
```

Standard ASI gate result:

```text
status: rejected
gain_over_donor_harness:    0.0000
gain_over_scripted_harness: 0.0000
latent_core_drop:           0.2963
world_model_drop:           0.0000
verifier_drop:              0.0000
failed:
  qtrm_does_not_beat_donor_harness
  qtrm_does_not_beat_scripted_harness
  world_model_not_causal
  verifier_not_causal
```

Interpretation:
this is a useful rejection. It proves the latent core/workspace participate in
the controller's final `ANSWER` decision, but the current action-policy metric
does not prove QTRM beats a scripted/donor harness. It also confirms the world
model and verifier are not yet causal in action selection. The next
architecture step is to wire verifier/world-model outcomes into controller
state or rewards, then rerun the same gate.

## [2026-05-02] architecture | Canonical SSOT causal training loss

Added direct training pressure for the simplest answer path:

```text
full canonical SSOT greedy path
> core_off/workspace_off/evidence_bottleneck_off ablations
```

Artifacts:

- loss: `src/qtrm_mm/losses.py::canonical_causal_ablation_loss`;
- config: `configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml`;
- script: `scripts/168_run_canonical_ssot_greedy_causal_train.sh`;
- contract: `docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md`.

Important detail:
ablation log-probs are stop-gradient targets. The objective should improve the
full path, not teach shared weights to degrade ablated paths.

Status:
unit/config/script tests pass. The architecture claim is still rejected until
the canonical answer gate shows held-out full-QTRM improvement over donor-only
and causal drops for component-off ablations.

## [2026-05-02] architecture | Core-to-text bridge tried for canonical causal path

Finding:
the first canonical causal loss used donor-fused logits, which hid QTRM causal
differences. It now uses `qtrm_residual_logits`. After that fix, the plain
canonical config still had `canonical_causal_margin=0.0000`, meaning
`core_off`/`workspace_off` did not change the QTRM residual answer path.

Architecture change:

- added `configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml`;
- enabled `core_context_enabled`;
- enabled `core_to_text_enabled`;
- set `coda_attn_every: 1`;
- added `qtrm_core_to_text_off_with_evidence` to the canonical answer gate.

Result:

```text
checkpoint: runs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050/last.pt
report: docs/wiki/decisions/canonical-ssot-coretotext-answer-gate-after-causal-train-4.md
status: rejected
full QTRM: 2 / 4
donor-only with evidence: 2 / 4
workspace_off/core_off/core_to_text_off: 2 / 4
```

Interpretation:
`core_to_text` is the first candidate that makes canonical residual logits and
some completions differ under ablation, so it is a better root bridge than the
previous config. It is not accepted: no held-out answer-score drop appears when
the bridge is disabled. Next work should train a stronger forced answer
bottleneck or increase the causal objective on a larger split, while keeping
the SSOT greedy answer gate as the acceptance test.

## [2026-05-02] architecture | Canonical causal margin probe added

Added a fast residual-logprob diagnostic before slow generation gates:

```text
scripts/169_probe_canonical_causal_margin.py
```

It compares the full canonical SSoT path against `core_off`, `workspace_off`,
`core_context_off`, `core_to_text_off`, and `evidence_bottleneck_off` on
`qtrm_residual_logits`. This catches the specific failure where training loss
looks causal but donor-fused greedy output hides the effect.

Result for `canonical_ssot_coretotext_forced_s150`:

```text
core_off margin:        0.1735
workspace_off margin:   0.1866
core_to_text_off margin:0.1348
```

The following greedy gate was still rejected because all critical ablation
strings matched the full model. Conclusion: the bridge moved confidence, but
not top-1 answer decisions.

## [2026-05-02] architecture | Core answer bottleneck accepted on 4-case SSoT gate

Implemented the stronger canonical candidate:

```text
configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150.yaml
```

Key design:
all evidence remains in the canonical prompt token stream, but QTRM residual
answer logits are generated through a latent answer bottleneck over `z_h`
instead of the ordinary text coda path.

Artifacts:

```text
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150/last.pt
margin probe: runs/eval/canonical_causal_margin_probe_core_answer_bottleneck_s150_4_summary.json
gate report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-4.md
```

Result:

```text
status: accepted
full QTRM: 3/4
donor-only with evidence: 2/4
workspace_off: 2/4
workspace_off same-completion rate: 0/4
core_off same-completion rate: 0/4
```

Residual margin probe:

```text
core_off:      0.6939
workspace_off: 2.8459
```

Remaining limitation:
`core_off` changes the strings but does not yet reduce hit count on the 4-case
gate. Scale the accepted candidate to more cases and add a stronger core-only
drop gate before claiming the recursive core is solved.

## [2026-05-02] architecture | Core answer bottleneck rejected on 8-case scale-up

The 4-case acceptance did not scale:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    6 / 8
workspace_off_with_evidence: 6 / 8
core_off_with_evidence:      6 / 8
causal modes:                none
```

Interpretation:
the answer bottleneck is a real causal path at small scale, but it is not a
reliable improvement. It overrode donor-correct outputs on 3-hop/location and
temporal/conflict cases.

Safe-gate eval:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_safe_gate_eval.yaml
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-safe-gate-eval-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
causal modes:                none
```

Decision:
do not promote `answer_bottleneck` as the canonical default. Keep it as a
diagnostic architecture showing that a latent path can affect greedy tokens,
then move to selective residual control: donor as default, QTRM intervention
only when calibrated by verifier/reasoning signals.

## [2026-05-02] architecture | Selective residual-gate fine-tune rejected

Ran a conservative gate fine-tune:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150.yaml
init: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150/last.pt
```

Training kept donor logits at `1.0` and enabled `qtrm_residual_gate`. The final
canonical margin was high (`1.1902`), so the path learned to matter in logprob
space.

Held-out 8-case greedy SSoT gate:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-selective-gate-s150-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
core_off_with_evidence:      6 / 8
causal modes:                none
```

Conclusion:
do not spend another run on this checkpoint with only scale/gate/margin
tweaks. The next architecture must train explicit verifier and decision
targets: allow, abstain, revise, or search-more. The accepted in-model answer
decision head is the strongest prior artifact, but it must be adapted without
violating the canonical SSoT/greedy contract.

## [2026-05-02] architecture | Canonical decision-token SFT rejected as answer architecture

Added SSOT decision-token data generation:

```text
script: scripts/170_build_canonical_decision_token_data.py
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
rows: 144
decision targets: ANSWER=96, ABSTAIN=48
hidden workspace fields: 0
```

Trained a donor-preserving residual checkpoint:

```text
config: configs/qwen35_2b_4090_canonical_decision_tokens_s120.yaml
runner: scripts/171_run_canonical_decision_token_train.sh
checkpoint: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
final canonical_causal_margin: 0.0725
```

The residual-logprob probe passed on decision-token rows:

```text
core_off margin:      0.0792
workspace_off margin: 0.0933
```

But the canonical greedy answer gate rejected the checkpoint. I also found and
fixed an eval mismatch: the generic answer-gate script had hardcoded
`QTRM_LOGITS_SCALE=0.10`, while this config uses `0.30`. The generic runner now
uses the config unless a scale override is explicit, and the decision-token
runner sets `0.30`. The corrected run still failed:

```text
report: docs/wiki/decisions/canonical-decision-tokens-s120-qscale030-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

Decision:
decision-token labels alone are not enough. The next architecture must make
the verifier/decision state causally control the final answer logits, not only
improve average label logprob under a donor-dominated greedy decode.

## [2026-05-02] architecture | Hidden-only answer-decision head rejected

Trained a minimal in-model answer-decision head on SSOT prompt rows:

```text
config: configs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200.yaml
data: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
init: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200/last.pt
trainable params: 513
```

Held-out 8-case gate with `--model-answer-decision`:

```text
report: docs/wiki/decisions/canonical-ssot-hidden-answer-decision-s200-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

Observed block probabilities on `qtrm_residual_with_evidence`:

```text
blocked: 0 / 8
range: 0.00255..0.00280
```

Interpretation:
the currently frozen hidden representation does not expose answer validity to a
tiny linear decision head. The next viable path is not another hidden-only
threshold. We need an explicit verifier-conditioned answer policy or internally
learned telemetry features that are causal for final logits.

## [2026-05-02] raw intelligence | Depth-supervised recursive core rejected

Added a stricter pure-recursive gate check:

```text
failed check: depth_outputs_identical_across_steps
```

This rejects runs where `core_steps=1/2/4/8` all select the same answer on
every comparable held-out case, even if the core path is active.

Ran the S160 raw-core checkpoint and two S080 depth-supervised follow-ups:

```text
S160 report:
  docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md

S080 internal depth probe:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-depth-supervised-s080-depth-gate-8.md

S080 final-path CE:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_finalpath_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-depth-supervised-finalpath-s080-depth-gate-8.md
```

All three runs were rejected on the same 8-case no-evidence gate:

```text
donor_only_no_evidence: 3/8
qtrm_core_off_no_evidence: 3/8
qtrm_core_steps_1/2/4/8_no_evidence: 3/8
depth output diversity: 0/8 cases changed by depth
shortcut records: 0
```

Decision:
this is no longer a RAG, SSOT, formatting, or final-token-loss problem. The
recursive core reaches a fixed-point-like answer after one step. The next
architecture candidate must use depth-conditioned state transitions or exact
intermediate-state supervision so deeper steps have different causal roles.

## [2026-05-02] raw intelligence | Mandatory core-loop readout implemented

Added a stricter loop-causal answer path:

```text
prompt -> frozen embed/prelude/workspace -> recursive z_L/z_H loop
-> core_loop_readout_cross -> LM head -> final text logits
```

Code:

```text
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/config.py
src/qtrm_mm/training/train.py
```

Config updated:

```text
configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
core_loop_readout_enabled: true
core_loop_readout_requires_core: true
n_coda_layers: 0
core_to_text_enabled: false
trainable_param_policy: core_and_loop_readout
```

Purpose:
the loop is no longer just a latent prefix that coda may or may not use. In
this probe, the final answer residual must be read from the recursive loop
state, and `disable_core=true` zeroes that path. Acceptance still requires the
raw-intelligence gate: deeper loop depth must beat donor and core-off without
MemoryOS or retrieval.

Smoke:

```text
run: pure_recursive_loop_readout_smoke_s001
steps: 1
max_cases: 1
trainable policy: core_and_loop_readout
result: runner completed, raw gate rejected as expected for smoke
report: docs/wiki/decisions/pure-recursive-loop-readout-smoke-s001-depth-gate-1.md
```

## [2026-05-03] raw intelligence | First staged loop-readout gate accepted

Fixed a raw-eval forced-choice tie bug:

```text
problem: all-zero logits from core_off could win by first-choice ordering
fix: top logprob ties become __FORCED_CHOICE_TIE__ and score as miss
script: scripts/192_eval_raw_intelligence.py
test: tests/test_raw_intelligence_eval_script.py
```

Ran mandatory loop-readout S160:

```text
final-target S160:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_s160/last.pt
  heldout8: rejected
  donor 2/8, core_off 0/8, core8 4/8
  failure: no depth scaling; depth outputs identical

staged-target S160:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_staged_s160/last.pt
  heldout8: accepted
  donor 2/8, core_off 0/8, core8 3/8
  depth changed 2/8
  heldout16: rejected
  donor 5/16, core_off 0/16, core8 4/16
  depth changed 3/16
```

Decision:
mandatory loop readout and staged targets are the first combination that passes
the small pure-recursive gate. The result is weak and not yet scalable; do not
claim ASI or robust reasoning. Next work should resume with staged S320/S640 or
a stronger explicit recurrent state-machine core.

## [2026-05-03] ingest | Formal CoT vs latent thought

Added:

```text
paper: A Formal Comparison Between Chain of Thought and Latent Thought
arXiv: https://arxiv.org/abs/2509.25239
pdf: references/papers/recurrent_depth/formal_comparison_cot_latent_thought_2509.25239.pdf
code: references/official/cot-vs-loop@783fa90
wiki: docs/wiki/sources/formal-cot-vs-latent-thought.md
```

Decision:
the paper is a guardrail against overclaiming pure latent recurrence. It argues
for a formal separation: latent thought favors parallelizable computation,
while CoT has advantages for stochastic approximate counting and sampling.
QTRM raw-intelligence gates should therefore split cases by task family instead
of treating one aggregate score as proof that latent loop reasoning works or
does not work.

Implemented:

```text
case metadata:
  reasoning_family
  expected_paradigm
  requires_stochasticity
  parallel_depth_estimate
  serial_trace_length_estimate

gate metadata:
  by_task_family
  by_reasoning_family
  by_expected_paradigm
```

Regenerated:

```text
data/eval/pure_recursive_reasoning_heldout_72.jsonl
data/filtered/pure_recursive_reasoning_train256_cases.jsonl
data/filtered/pure_recursive_reasoning_preferences_train.jsonl
```

## [2026-05-03] architecture | TRM-style answer-state loop

Added an explicit answer-state loop candidate:

```text
y_0 = prompt text hidden states
for each recursive depth t:
  y_t = norm(y_{t-1} + gate(y_{t-1}) * cross_attn(y_{t-1}, z_H_t))
final logits = LMHead(y_T)
```

Code:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_answer_state_loop_s160.yaml
```

Reason:
the previous `core_loop_readout` was causal but still a final readout from the
loop state. The new path makes the visible answer state itself the recurrent
object, closer to TRM's `x, y, z -> z' -> y'` correction loop.

Verification:

```text
smoke:
  run: pure_recursive_answer_state_loop_smoke_s001
  status: rejected as expected
  hits: 0/6
  purpose: checkpoint/eval path only

S160 staged:
  run: pure_recursive_answer_state_loop_s160
  report: docs/wiki/decisions/pure-recursive-answer-state-loop-s160-depth-gate-16.md
  status: rejected
  donor: 5/16
  core_off: 0/16
  core_steps_1: 3/16
  core_steps_2: 4/16
  core_steps_4: 4/16
  core_steps_8: 5/16
  changed_by_depth: 6/16
```

Decision:
this is not accepted raw-intelligence improvement yet because deepest core ties
the donor instead of beating it. It is still a stronger architecture signal
than the previous frozen-output failures: disabling the core collapses the
answer path, deeper latent steps change outputs, and the ladder improves from
depth 1 to depth 8. Next work should target the failure families directly,
especially list transforms and cases where deeper boolean steps flip away from
the correct early answer.

## [2026-05-03] eval+training | causal prefix raw gate

Finding:
the previous forced-choice path fed `prompt + candidate answer` to QTRM in one
forward pass. That is standard for causal donor scoring, but QTRM's
workspace/core path can cross-attend to the whole sequence, including future
answer tokens. This made the raw-intelligence gate less strict than intended.

Implemented:

```text
eval:
  scripts/192_eval_raw_intelligence.py
  scoring=causal_forced_choice
  each answer token is scored from prompt + previous answer tokens only

runner:
  scripts/193_run_pure_recursive_reasoning_depth_gate.sh
  default SCORING=causal_forced_choice

training:
  scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-supervision
  prompt-only input predicts the first answer token

runner:
  scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_SUPERVISION=1
```

Rejected intermediate:

```text
run: pure_recursive_answer_state_loop_hard_s320
change: hard-family oversampling + first-token choice margin, but still
  full-answer training
causal gate: rejected
donor: 5/16
core8: 4/16
reason: no_depth_scaling_gain, deep_core_does_not_beat_donor
```

Accepted result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_s160
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-s160-depth-gate-16.md
status: accepted
donor: 5/16
core_off: 0/16
core1: 3/16
core2: 4/16
core4: 6/16
core8: 6/16
changed_by_depth: 3/16
passed:
  deep_core_beats_core_off
  deep_core_beats_donor
  depth_scaling_gain_present
  depth_outputs_not_all_identical
  no_retrieval_or_memoryos_shortcut
```

Decision:
this is the cleanest raw-recursive signal so far because both training and
evaluation now avoid future-answer leakage. The improvement is still small
and family-limited: arithmetic improves to 3/4 at depth 8, boolean stays 2/4,
symbolic stays 1/4, and list transforms remain 0/4. Next promotion requires a
larger held-out sweep and a second-stage fix for multi-token/list outputs.

## [2026-05-03] training | causal prefix multi-token rejected

Implemented:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-max-target-tokens N
  _prepare_causal_prefix_answer_examples()

scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_MAX_TARGET_TOKENS

tests/test_pure_recursive_depth_supervised_train_script.py
  prefix examples:
    prompt -> answer token 0
    prompt + answer token 0 -> answer token 1
    ...
```

Purpose:
the accepted `causal_prefix_s160` checkpoint only trains the first answer
token. The multi-token variant tests whether answer-state recursion can learn
the next-token chain without leaking future answer tokens through the workspace
or core path.

Verification:

```text
smoke:
  run: pure_recursive_answer_state_loop_causal_prefix_multitoken_smoke_s001
  status: accepted
  purpose: plumbing only
  observed: causal_prefix_examples=4

quality:
  run: pure_recursive_answer_state_loop_causal_prefix_multitoken_s080
  config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080.yaml
  checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-multitoken-s080-depth-gate-16.md
  status: rejected
  donor: 5/16
  core_off: 0/16
  core1: 3/16
  core2: 4/16
  core4: 4/16
  core8: 4/16
  failed: deep_core_does_not_beat_donor
```

Comparison to canonical baseline:

```text
causal_prefix_s160:
  core8: 6/16
  accepted

causal_prefix_multitoken_s080:
  core8: 4/16
  rejected
```

Decision:
do not promote multi-token causal-prefix training as canonical. The
implementation is useful as an experiment tool, but the result shows that
naively adding later answer-token loss can dilute the raw recursive reasoning
signal. Keep `causal_prefix_s160` as the current canonical raw-intelligence
baseline. The next architecture move should not be "more tokens with the same
loss"; it should explicitly preserve first-token/depth reasoning gains while
adding a separate causal sequence-readout objective or a teacher-generated
latent-state target.

## [2026-05-03] training | split causal-prefix later-token loss rejected

Implemented:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-later-token-weight
  first answer token weight: 1.0
  later answer token weight: configurable, tested at 0.1

scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_LATER_TOKEN_WEIGHT

tests/test_pure_recursive_depth_supervised_train_script.py
  _causal_prefix_example_loss_weight()
```

Hypothesis:
the naive multi-token experiment failed because later-token continuation loss
overpowered the first-token recursive decision objective. Lowering later-token
weight should preserve the accepted depth signal while adding a weak sequence
readout pressure.

Result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_split_s080
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-split-s080-depth-gate-16.md
status: rejected
donor: 5/16
core_off: 0/16
core1: 3/16
core2: 5/16
core4: 5/16
core8: 5/16
failed: deep_core_does_not_beat_donor
```

Decision:
split later-token loss is also not canonical. It is less damaging than equal
multi-token loss (`core8 5/16` versus `4/16`), but it still loses the accepted
baseline (`6/16`) and does not solve list transforms (`0/4`). The canonical
raw-intelligence checkpoint remains:

```text
runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
```

Next architecture implication:
do not continue local CE-weight tuning for sequence continuation. The failure
class is architectural: answer sequence continuation and latent recursive
reasoning need separate acceptance gates. The next serious candidate should
train the recurrent state transition or teacher-latent target directly, then
keep answer-token CE as a downstream readout probe.

## [2026-05-03] architecture | LeWM core-world-model raw gate rejected

Implemented raw-trainer support for LeWorldModel-style recursive-core
prediction:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --core-world-model-weight
  core_world_model_actions
  jepa_world_model_loss over recursive core states

src/qtrm_mm/training/train.py
  core_and_answer_state_loop_and_world_model trainable policy

src/qtrm_mm/qtrm_model.py
  core_world_model predictor max length handles dynamic depth schedules

configs/qwen35_2b_4090_verified_reasoning_lewm_core_s200.yaml
```

Smoke:

```text
runs/qwen35_2b_4090_verified_reasoning_lewm_core_smoke_s020/last.pt
core_world_model loss appears at depth 2/4/8
```

Held-out raw reasoning eval:

```text
baseline:
  runs/eval/verified_reasoning_baseline_interleaved_max5.jsonl
  total 2/30

CE-only S200:
  runs/eval/verified_reasoning_s200_interleaved_max5.jsonl
  total 2/30

CE + LeWM core S200:
  runs/eval/verified_reasoning_lewm_core_s200_interleaved_max5.jsonl
  total 2/30
```

Decision:
LeWM is now wired and trainable, but this experiment does not show raw
reasoning gain. Do not promote it as canonical. The next useful experiment must
directly test transition prediction quality and its correlation with answer
accuracy, instead of assuming next-latent MSE will automatically improve raw
reasoning.

## [2026-05-03] evaluation | LeWM transition quality gate

Added:

```text
scripts/200_eval_core_world_model_transition.py
tests/test_core_world_model_transition_eval.py
docs/wiki/decisions/lewm-transition-quality-gate.md
```

The gate computes masked MSE between:

```text
core_world_model_pred
core_world_model_target
```

and merges existing raw-answer `hit` labels from raw eval JSONL.

Result:

```text
CE-only S200 under LeWM config:
  runs/eval/verified_reasoning_ce_only_s200_transition_max5_summary.json
  core_steps_2 MSE: 1.3253
  core_steps_4 MSE: 1.3295
  core_steps_8 MSE: 1.3322
  QTRM hit rate: 0.0

CE + LeWM core S200:
  runs/eval/verified_reasoning_lewm_core_s200_transition_max5_summary.json
  core_steps_2 MSE: 0.00816
  core_steps_4 MSE: 0.00786
  core_steps_8 MSE: 0.00778
  QTRM hit rate: 0.0
```

Interpretation:
LeWM successfully learns the current recursive-core latent dynamics, but those
dynamics are not yet semantically anchored to answer correctness. This falsifies
the weak claim that lower next-latent MSE alone is enough for raw reasoning.

Next:
build a verifiable symbolic transition-state gate where latent transitions have
known state-update targets, then only promote LeWM if transition quality predicts
final answer correctness and world-model-off ablation drops.

## [2026-05-03] evaluation | Symbolic transition gate for staged LeWM

Added:

```text
scripts/201_eval_symbolic_transition_gate.py
tests/test_symbolic_transition_gate_script.py
configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml
docs/wiki/decisions/pure-recursive-lewm-staged-s200-symbolic-transition-gate.md
```

Runner update:

```text
scripts/197_run_pure_recursive_depth_supervised_train.sh
  CORE_WORLD_MODEL_WEIGHT -> --core-world-model-weight
```

Trained pure-recursive staged LeWM from the canonical causal-prefix baseline:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_lewm_staged_s200/last.pt
raw depth gate:
  runs/eval/pure_recursive_lewm_staged_s200_depth_gate_16.jsonl
  status: accepted
  donor: 5/16
  core_off: 0/16
  core1/core2/core4/core8: 4/16, 5/16, 6/16, 6/16
```

Symbolic transition gate:

```text
canonical S160:
  runs/eval/symbolic_transition_canonical_s160_max16_summary.json
  18/64

verified-reasoning LeWM S200:
  runs/eval/symbolic_transition_lewm_core_s200_max16_summary.json
  19/64

pure-recursive staged LeWM S200:
  runs/eval/symbolic_transition_pure_recursive_lewm_staged_s200_max16_summary.json
  18/64
```

Latent transition quality for pure-recursive staged LeWM:

```text
runs/eval/pure_recursive_lewm_staged_s200_transition_max16_summary.json
core2 MSE: 0.00647
core4 MSE: 0.00740
core8 MSE: 0.00783
transition_mse_hit_pearson: 0.06636
```

Decision:
LeWM learns the recursive latent dynamics, and the raw depth gate remains
accepted, but symbolic intermediate-state accuracy is unchanged from canonical.
Do not promote LeWM as semantic reasoning architecture yet. Next candidate is a
semantic transition head attached to the same recurrent state used by answer
formation, followed by all-depth answer-state CE if the diagnostic shows the
state contains the target but the answer readout ignores it.

## [2026-05-03] architecture | LeWM demoted from canonical answer path

Decision:
The canonical QTRM raw-intelligence path is now the single-trace TRM
answer-state loop, not the LeWM/core-world-model probe.

Updated:

```text
src/qtrm_mm/eval/ssot_contract.py
scripts/95_eval_memory_retrieval.py
tests/test_ssot_contract.py
tests/test_memory_eval_script.py
docs/wiki/architecture/canonical-architecture-matrix.md
docs/wiki/architecture/qtrm-forward-pass.md
docs/wiki/components/qtrm-world-model.md
docs/wiki/concepts/leworldmodel.md
docs/wiki/decisions/lewm-demoted-from-canonical-single-trace-trm.md
```

Canonical settings:

```text
core_world_model_enabled: false
loss_core_world_model_weight: 0.0
donor_logits_scale: 0.0 for the raw-intelligence answer path
answer_state_loop_enabled: true
answer_state_loop_requires_core: true
```

Reason:
LeWM reduced self-latent transition MSE, but it did not improve symbolic
intermediate-state accuracy or answer accuracy. Until a semantic transition,
answer-causal, or planner gate passes, LeWM stays probe-only.

## [2026-05-03] raw intelligence | LeWM-free single-trace rebuild S160 rejected on 72 cases

Rebuilt the canonical single-trace TRM path from the healthy metacog baseline:

```text
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/no_warmup_rebuilt_s001/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160/last.pt
core_world_model_weight: 0.0
```

The first eval attempt wrote to `runs/eval` and failed with `OSError: [Errno 5]
Input/output error`, so the 72-case eval was rerun to `local_eval/`.

Held-out result:

```text
eval:
  local_eval/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160_depth_gate_72.jsonl
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-rebuild-s160-depth-gate-72-summary.json
status: rejected
donor_only: 22/72
core_off: 0/72
core1/core2/core4/core8: 19/72, 18/72, 18/72, 18/72
depth outputs identical: 69/72
```

Interpretation:
The core path is causal because `core_off` collapses, but recurrent depth is not
doing useful iterative computation yet. Next experiment should add semantic
depth pressure rather than reintroducing LeWM.

Failure ledger:

```text
docs/wiki/decisions/pure-recursive-single-trace-rebuild-s160-failure-ledger.md
```

## [2026-05-03] raw intelligence | Semantic-depth S120 rejected

Trained the LeWM-free canonical answer-state loop from the S160 rebuild with
staged internal first-token CE and no core-world-model loss:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_semantic_depth_s120/last.pt
core_world_model_weight: 0.0
staged_internal_first_token_ce_weight: 0.5
```

Held-out smoke16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-depth-gate-16-summary.json
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 2/16, 2/16, 2/16, 2/16
depth outputs identical: 16/16
```

Train-distribution smoke16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-train-slice-16-summary.json
status: rejected
donor_only: 3/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 4/16, 4/16
depth outputs identical: 16/16
```

Symbolic transition train16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-symbolic-transition-train16-summary.json
accuracy: 17/64
depth1/depth2/depth4/depth8: 4/16, 5/16, 4/16, 4/16
list_transform: 0/16
```

Interpretation:
The core path remains causal, but staged first-token pressure did not create a
reliable intermediate-state transition. One bounded multi-token causal-prefix
probe is allowed because the prior run trained mostly first answer tokens while
the gate scores full answer strings.

## [2026-05-03] raw intelligence | Multi-token depth S080 rejected

Continued from semantic-depth S120 with multi-token causal-prefix supervision
and hard-family repeats:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_multitoken_depth_s080/last.pt
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-multitoken-depth-s080-depth-gate-16-summary.json
CAUSAL_PREFIX_MAX_TARGET_TOKENS: 6
FAMILY_REPEAT: list_transform=4,arithmetic_chain=3,symbolic_binding=2
core_world_model_weight: 0.0
```

Held-out smoke16:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 3/16, 4/16, 4/16, 4/16
depth outputs changed: 1/16
list_transform core8: 0/4
```

Interpretation:
The first-token-only defect was not the root cause. The current
answer-state-loop-only design can make the core causal and can create a tiny
depth ladder, but it still cannot beat donor-only or solve list transforms.
The failure ledger now promotes the next candidate to an explicit recurrent
transition-state core.

## [2026-05-04] raw intelligence | Transition-state core implementation surface

Implemented the explicit transition-state candidate surface:

```text
model:
  core_depth_states -> transition_state_features -> answer_state_loop
  disable_transition_state=True ablation

loss/eval:
  canonical causal ablation mode: transition_state_off
  raw eval mode: qtrm_core_steps_8_transition_state_off_no_evidence
  depth gate runner: INCLUDE_TRANSITION_STATE_OFF=1
  depth-supervised contrast:
    --transition-state-contrast-weight
    --transition-state-contrast-margin

config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
```

Verification:

```text
transition-state model/eval/loss focused tests: passed
related core-halting/checkpoint/raw-gate tests: passed
```

Next gate:
Train the S080 candidate and reject it unless core8 beats donor-only/core-off
and `transition_state_off` drops below full core8 on held-out prompt-only
reasoning.

## [2026-05-04] raw intelligence | Transition-state S080 rejected

Ran the first in-model explicit transition-state candidate:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080/last.pt
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 5/16, 4/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 3/16
state-off output changes: 0/16
shortcuts: 0
```

Passed:

```text
deep_core_beats_core_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
deep_core_does_not_beat_transition_state_off
```

Interpretation:
The recursive core remains causal relative to core-off, but the new
transition-state branch is not causally consumed by the answer path. Since
state-off produces identical outputs on the 16-case gate, this is not yet an
explicit transition-state reasoning model. The next change must put the state
features on a stronger answer-critical path or add direct symbolic
state-supervision before more scale-up.

## [2026-05-04] raw intelligence | Direct transition-state CE S080 rejected

Ran the direct transition-state CE follow-up:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_ce_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_ce_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-ce-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 4/16
state-off output changes: 0/16
training transition_state_first_token_acc: 0.0
```

Interpretation:
The direct CE surface did not make the transition-state branch answer-causal.
The result strengthens the root-structure doubt: a tiny continuous sigmoid
state vector plus gated residual into the answer-state loop is too easy for the
renderer to ignore. The next candidate should use a compact supervised
state-code/token path and require the answer renderer to consume it, instead of
adding more CE weight to the same side readout.

Verification:

```text
165 related unittest cases: passed
py_compile for touched model/train/eval modules: passed
bash -n for pure-recursive eval/train runners: passed
```

## [2026-05-04] raw intelligence | State-code transition path S080 rejected

Implemented and tested a compact state-code path:

```text
model:
  transition_state_code_enabled
  transition_state_codebook_size
  core_depth_states -> transition_state_code_logits
  soft code embedding -> answer_state_loop
  disable_transition_state=True zeroes code embeddings

training:
  --transition-state-code-ce-weight
  transition_state_code_targets = staged first-token id modulo codebook
  transition_state_code_ce_loss

config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_s080.yaml
```

Run:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 5/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 4/16
state-off output changes: 0/16
shortcuts: 0
```

Training signal:

```text
early:
  transition_state_code_ce 6.48 -> 0.0047
  transition_state_code_acc 0.0 -> 1.0

late hard windows:
  transition_state_code_ce rose again to 4.95-12.37
  transition_state_code_acc fell to 0.0 on logged windows
```

Interpretation:
The compact code surface is learnable on some staged states, but the final
answer is still not state-code causal. Because full and state-off completions
match 16/16, the active blocker is now the answer renderer, not another state
side loss. The next experiment should remove the ordinary trajectory
cross-attention bypass or replace `answer_state_loop` with a stricter
state-token decoder whose LM logits must be emitted from recurrent state.

Verification:

```text
213 related unittest cases: passed
py_compile for touched model/train/eval modules: passed
bash -n for pure-recursive eval/train runners: passed
```

## [2026-05-04] raw intelligence | Code-only state decoder becomes causal but still rejected

Added a stricter answer renderer option:

```text
model:
  transition_state_code_only_answer_loop

behavior:
  when enabled, answer_state_loop cross-attends to the state-code token only
  rather than concatenating [state_code_token, trajectory_workspace_state].
```

This removes the ordinary trajectory cross-attention bypass that let previous
transition-state candidates ignore the explicit state path.

Run:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_only_s080.yaml
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_only_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-only-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 4/16, 5/16
transition_state_off: 3/16
state-off output changes: 6/16
shortcuts: 0
```

Passed:

```text
deep_core_beats_core_off
deep_core_beats_transition_state_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
```

Interpretation:
This is the first explicit transition-state candidate where the state path is
held-out answer-causal. The previous failure class was architectural: the
answer renderer could ignore the state token. The current failure class is now
capability/training: the causal state-token decoder ties donor-only but does not
beat it.

Next:

```text
Continue from the code-only checkpoint with a longer S240/S500 run, keep
transition_state_off in the gate, and tune only objectives that improve hard
families without losing the new state-off causal drop.
```

## [2026-05-04] raw intelligence | Code-only S240 keeps causal state path but loses to donor

Continued the strict code-only state decoder from the S080 checkpoint:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_only_s080.yaml
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s240/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_only_s240_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-only-s240-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 3/16, 3/16, 4/16, 4/16
transition_state_off: 3/16
state-off output changes: 4/16
shortcuts: 0
```

Mode semantics:

```text
donor_only:
  true donor baseline; QTRM residual logits are forced off and donor logits are used.
core_off:
  internal QTRM ablation; QTRM forward still runs with disable_core=True and
  donor fallback is not forced. It is not equivalent to donor_only.
transition_state_off:
  QTRM core still runs, but the explicit transition-state/code path is disabled.
```

Passed:

```text
deep_core_beats_core_off
deep_core_beats_transition_state_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
```

Interpretation:
S240 confirms the code-only state-token path is causal, but longer training did
not improve raw reasoning. It regressed from the S080 core8 tie with donor
`5/16` to `4/16`, while donor stayed `5/16`. The final training windows also
showed hard-family spikes (`depth_final_acc=0.0`, high CE), so the next
experiment should not be a blind S500. The active failure is hard-family policy
learning and state-code target stability, not hidden MemoryOS/RAG leakage or
state-path non-causality.

Next:

```text
Do not promote S240 as baseline.
Keep S080 code-only as the canonical causal checkpoint for now.
Build a hard-family curriculum/overfit proof that first makes list_transform
and arithmetic_chain improve under core depth, then re-run the 16-case gate.

## [2026-05-04] evaluation | Forced-choice length normalization fixed hard-family gate bias

The hard-family overfit8 proof exposed a gate-level confound before it exposed a
new model-architecture failure. The first S120 run used legacy summed
answer-token logprob scoring:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_overfit8_s120/last.pt
legacy eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-depth-gate-8-summary.json
status: rejected
donor_only: 1/8
core8: 1/8
transition_state_off: 0/8
list_transform: 0/4
```

Failure inspection showed `list_transform` answers like `208,204` were losing
to the one-token distractor `EMPTY` because the eval ranked candidates by
summed logprob. This was an evaluation-length bias, not clean evidence that the
state-code path could not learn.

Implemented eval contract fix:

```text
scripts/192_eval_raw_intelligence.py
  --choice-score-normalization mean
  choice_scores[].logprob_sum
  choice_scores[].token_count

scripts/193_run_pure_recursive_reasoning_depth_gate.sh
  CHOICE_SCORE_NORMALIZATION=mean

src/qtrm_mm/eval/raw_intelligence_gate.py
  eval_contract in JSON/markdown reports
```

Same checkpoint, same 8 cases, mean-normalized causal forced-choice:

```text
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-mean-depth-gate-8-summary.json
status: accepted
donor_only: 1/8
core_off: 0/8
core1/core2/core4/core8: 2/8, 3/8, 3/8, 3/8
transition_state_off: 0/8
list_transform core8: 2/4
shortcuts: 0
```

Interpretation:
This is a real local raw-intelligence signal for the strict code-only
transition-state path on the overfit8 hard-family proof: core8 beats donor-only
and transition_state_off. It is not yet a held-out scale proof. The next gate
must use mean-normalized scoring and test different held-out indices before any
broader ASI/architecture claim.

## [2026-05-04] raw intelligence | Hard-family heldout200 rejected after local overfit proof

Ran the same S120 hard-family checkpoint on different generated hard-family
indices with the corrected eval contract:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_overfit8_s120/last.pt
heldout cases:
  data/eval/pure_recursive_hard_family_heldout200_cases.jsonl
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-heldout200-mean-depth-gate-8-summary.json
scoring:
  causal_forced_choice + mean choice-score normalization
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 1/8, 1/8, 1/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Interpretation:
The previous accepted result is only a local overfit proof. On heldout index
200, QTRM often keeps outputs in the training numeric range, for example
list-transform expected values around `408,404` but core8 selects values like
`204,202`. That means the current state-code path can memorize a narrow
hard-family slice, but has not learned the abstract operation robustly.

Next:

```text
Train a broader hard-family generalization run:
  train_start_index=100
  train_cases_per_family >= 16
  heldout_start_index=200
  mean-normalized causal forced-choice gate

Acceptance:
  core8 > donor_only
  core8 > transition_state_off
  list_transform core8 > 0/4 on heldout200
```

## [2026-05-04] raw intelligence | Hard-family S240 generalization still rejected

Broadened training from the local overfit slice to 16 cases per hard family:

```text
runner:
  scripts/211_run_pure_recursive_hard_family_generalization_s240.sh
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_generalization_s240/last.pt
heldout:
  start_index=200, cases_per_family=4, families=arithmetic_chain,list_transform
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_generalization_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-generalization-s240-depth-gate-8-summary.json
scoring:
  causal_forced_choice + mean choice-score normalization
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 2/8, 2/8, 2/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Failure:
The current `transition_state_code_targets` objective was still using
first answer-token id modulo the codebook. That makes the code label a token
artifact, not a semantic transition state. It can make the state path causal
locally, but it does not teach stable operations like "filtered evens" versus
"doubled filtered evens" across train/heldout index ranges.

Architecture change started:

```text
semantic transition-state codes:
  arithmetic_chain: sum -> product -> final
  list_transform: filtered -> doubled/final
  symbolic_binding: one-hop -> two-hop/final
  boolean_logic: not_q -> and -> final

preference margin:
  legacy first-token margin kept
  new sequence mode aligns training pressure with mean forced-choice eval

runner:
  scripts/212_run_pure_recursive_semantic_state_sequence_margin_s240.sh
```

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_semantic_state_sequence_margin_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_semantic_state_sequence_margin_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-semantic-state-sequence-margin-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 0/8, 1/8, 1/8, 1/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Interpretation:
Semantic stage codes are still too small. They distinguish "filtered" from
"doubled/final", but they do not carry value-bearing intermediate state. The
list-transform failures stayed half-scale (`204,202`) instead of producing the
heldout doubled values (`408,404`). The next root-architecture candidate must
use a value-bearing transition state, for example continuous transition-state
features or explicit text/numeric state targets, not only a categorical code.

Prepared next root-architecture runner:

```text
scripts/213_run_pure_recursive_value_state_sequence_margin_s240.sh
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
mechanism:
  continuous transition_state_features -> answer loop
  transition_state CE on staged first-token targets
  sequence preference margin
```

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_value_state_sequence_margin_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_value_state_sequence_margin_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-value-state-sequence-margin-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 2/8, 2/8, 2/8
transition_state_off: 1/8
list_transform core8: 0/4
```

Interpretation:
Continuous transition-state features are weakly answer-causal because core8
beats transition_state_off, but they still do not beat donor-only. Training
also kept `transition_state_first_token_acc=0.0`, so first-token staged-state
supervision is not enough to make the latent state carry values. The next
candidate must supervise full value-bearing intermediate states, not just a
stage code or a first token.

Implemented next candidate:

```text
training:
  --staged-internal-sequence-ce-weight
  --staged-internal-sequence-max-target-tokens

target:
  depth_targets[depth] token sequence, not only first token

path:
  prompt tokens -> recurrent core depth state -> core_depth_text_logits per
  depth -> full intermediate value text

runner:
  scripts/214_run_pure_recursive_full_state_sequence_s240.sh
```

Reason:
This keeps the architecture simple and uses the existing causal depth readout.
It directly teaches depth 1/2/4/8 to emit the matching intermediate value
sequence, so list_transform can be penalized for stopping at filtered
half-scale values.

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_full_state_sequence_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_full_state_sequence_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-full-state-sequence-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 1/8, 2/8, 2/8
transition_state_off: 2/8
list_transform core8: 0/4
```

Interpretation:
The full-state sequence loss did not make the transition-state path causal on
heldout: core8 tied both donor-only and transition_state_off. Training also
kept `staged_internal_sequence_acc=0.0` through the run. This is now the third
state-objective failure after semantic code and continuous first-token state,
so the local donor-residual/readout route should not receive more loss-weight
tuning. The next candidate must be a cleaner recurrent state-machine core
trained on solver traces, with state prediction and answer readout measured
separately.

Implemented trace-data foundation for that replacement:

```text
case schema:
  solver_trace:
    - depth
    - operation
    - state_text

builder:
  scripts/215_build_pure_recursive_solver_trace_dataset.py

train cases:
  data/filtered/pure_recursive_solver_trace_train_cases.jsonl
train trace rows:
  data/filtered/pure_recursive_solver_trace_train.jsonl

heldout cases:
  data/eval/pure_recursive_solver_trace_heldout200_cases.jsonl
heldout trace rows:
  data/eval/pure_recursive_solver_trace_heldout200.jsonl
```

Example:

```text
arith-chain-100:
  d1 add_operands: "" -> 110
  d2 multiply_sum: 110 -> 220
  d4 subtract_offset: 220 -> 217
  d8 hold_final: 217 -> 217
```

## 2026-05-04: Solver State-Machine Probe

Added a donor-free, MemoryOS-free recurrent state-machine probe:

```text
src/qtrm_mm/agentic/solver_state_machine.py
scripts/216_train_pure_recursive_solver_state_machine.py
tests/test_solver_state_machine.py
tests/test_pure_recursive_solver_state_machine_train_script.py
```

Key result:

```text
small compact-input probe:
  train rollout_final_exact: 0.90625
  heldout rollout_final_exact: 0.0

large probe:
  train rollout_final_exact: 0.46484375
  heldout rollout_final_exact: 0.0

primitive transition ceiling:
  heldout_large state: 256/256
  heldout_large final: 64/64
```

Decision:
direct recurrent state-string generation is rejected for the next canonical
raw-intelligence architecture. The core should select/compose causal primitive
state transitions, then render from the updated state.

Follow-up:

```text
char operation policy:
  heldout operation_exact: 0.625
  heldout rollout_final_exact: 0.5

structured operation policy:
  heldout operation_exact: 1.0
  heldout rollout_state_exact: 1.0
  heldout rollout_final_exact: 1.0

all-family structured primitive route:
  families: arithmetic_chain, symbolic_binding, boolean_logic, list_transform
  primitive ceiling: state 512/512, final 128/128
  learned structured policy: operation/state/final all 1.0
```

New canonical raw-core candidate:

```text
prompt/question
-> recurrent latent core
-> structured transition metadata head
-> primitive transition executor
-> explicit state
-> answer renderer
```

QTRM integration surface added:

```text
QTRMConfig.primitive_transition_enabled
QTRMConfig.primitive_transition_num_operations
QTRMConfig.primitive_transition_hidden_dim
model output: primitive_transition_operation_logits
trainer arg: --primitive-transition-operation-ce-weight
target: solver_trace.operation
```

Details:

```text
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240.md
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240-summary.json
```

## 2026-05-04: Primitive Transition Token Grounding

The mean-pooled prompt context primitive head reached 507/512 on heldout7000
but kept confusing list transform operations:

```text
double_filtered -> second_mapping
```

Implemented a token-level prompt attention path:

```text
core_depth_state query
-> cross-attend over canonical prompt token context
-> primitive transition operation logits
```

This keeps SSOT: prompt/task/evidence information still enters through the same
canonical token stream. There is no hidden MemoryOS/RAG evidence path.

Result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

heldout existing:
  operation accuracy: 256/256 = 1.0000

heldout7000:
  operation accuracy: 512/512 = 1.0000
```

Decision:
token-level prompt grounding is the current canonical primitive transition
selector for this synthetic raw-recursive-core gate.

## 2026-05-04: Primitive Transition Rollout Gate

Added the rollout causality evaluator:

```text
scripts/221_eval_qtrm_primitive_transition_rollout.py
qtrm_mm.agentic.solver_state_machine.rollout_solver_trace_from_operations
```

The evaluator executes QTRM-predicted primitive operations through the explicit
solver transition function and scores state/final answers.

Heldout7000:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

operation exact: 512/512
state exact:     512/512
final exact:     128/128
```

Core-off ablation:

```text
operation exact: 0/512
state exact:     0/512
final exact:     0/128
```

This makes the result a causal primitive-reasoning gate: recurrent core
operation sequence -> explicit state update -> final answer.

## 2026-05-04: Runtime Primitive Answer Path

Added a label-free primitive answer runtime:

```text
scripts/222_infer_qtrm_primitive_transition_answer.py
qtrm_mm.agentic.solver_state_machine.answer_from_primitive_operations
```

Runtime contract:

```text
prompt
-> QTRM recurrent core operation logits
-> predicted operation sequence
-> primitive transition executor
-> answer
```

Smokes with the prompt-attention checkpoint:

```text
arithmetic:
  ops: add_operands -> multiply_sum -> subtract_offset -> hold_final
  states: 7010 -> 14020 -> 14017
  answer: 14017

list_transform:
  ops: filter_even -> double_filtered -> hold_final -> hold_final
  states: 7004,7002 -> 14008,14004
  answer: 14008,14004
```

Boundary:
this is not general LM generation yet. It is the first runtime path where QTRM
core predictions produce final answers through a primitive state executor
without using `solver_trace` labels at inference time.

## 2026-05-04: Answer-Only Primitive Runtime Gate

Added the stricter prompt-only answer runtime evaluator:

```text
scripts/223_eval_qtrm_primitive_answer_runtime.py
```

Runtime input excludes `solver_trace`, intermediate states, and the chosen
answer. Scoring only compares the produced answer with `chosen/answer`.

Heldout7000:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000.json

answer exact:
  128/128
```

Core-off:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000_coreoff.json

answer exact:
  0/128
```

This is now the strongest synthetic primitive-reasoning gate: prompt-only QTRM
core operation predictions produce final answers, and disabling the core
destroys the path.

## 2026-05-04: Donor-Only Baseline Comparison

Added donor-only baseline evaluator:

```text
scripts/224_eval_donor_only_baseline.py
```

Heldout7000 comparison:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

donor-only forced_choice:
   61/128 = 0.4766

donor-only greedy:
   29/128 = 0.2266
```

Donor-only by family:

```text
forced_choice:
  arithmetic_chain: 15/32
  symbolic_binding: 17/32
  boolean_logic:    29/32
  list_transform:    0/32

greedy:
  arithmetic_chain:  0/32
  symbolic_binding:  0/32
  boolean_logic:    29/32
  list_transform:    0/32
```

Conclusion:
On this synthetic primitive-reasoning gate, QTRM primitive runtime beats
donor-only forced-choice and greedy baselines. This remains a scoped benchmark
claim, not a broad claim that QTRM is a better general-purpose LLM than the
donor.

Added an explicit benchmark-definition section to the decision doc. It now
spells out the four families with examples:

```text
arithmetic_chain:
  ((7007 + 3) * 2) - 3 -> 14017

list_transform:
  [7001, 7004, 7002, 7007, 7003] -> 14008,14004

symbolic_binding:
  A -> green -> violet

boolean_logic:
  (P AND NOT Q) OR R
```

This keeps the benchmark claim scoped: procedure selection and explicit state
updates on synthetic primitive families, not open-domain general LLM quality.

## 2026-05-04: OOD Surface-Form Primitive Gate

Added an OOD surface-form gate:

```text
scripts/225_build_pure_recursive_ood_surface_cases.py
data/eval/pure_recursive_primitive_transition_ood_surface_heldout8000_preferences.jsonl
```

Result:

```text
QTRM primitive answer runtime:
  107/128 = 0.8359

QTRM core-off:
    0/128 = 0.0000

donor-only forced_choice:
   71/128 = 0.5547

donor-only greedy:
   32/128 = 0.2500
```

QTRM by family:

```text
arithmetic_chain: 11/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Arithmetic variant breakdown:

```text
variant 0 "Solve this arithmetic expression exactly": 11/11
variant 1 "What integer do you get after evaluating": 0/11
variant 2 "Evaluate the expression ... report result": 0/10
```

Interpretation:
QTRM still beats donor-only on the OOD surface gate, but arithmetic routing is
surface-form sensitive. The next gate should add arithmetic prompt
augmentation and require OOD arithmetic recovery before claiming broader
arithmetic-language generalization.

## 2026-05-04: Surface-Form Augmentation Recovery Gate

Added a canonical plus OOD-surface training mix:

```text
scripts/226_build_pure_recursive_surface_aug_mix.py
data/filtered/pure_recursive_primitive_transition_surface_aug_mix_train.jsonl
```

Resumed from the prompt-attention primitive checkpoint and trained only the
primitive transition operation CE path:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt
```

Result on OOD surface heldout8000:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

QTRM core-off:
    0/128 = 0.0000

arithmetic_chain: 32/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Canonical heldout7000 regression check:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000
```

Interpretation:
The surface-form arithmetic routing failure is recoverable through the causal
primitive-operation path without canonical heldout regression. This is still a
scoped synthetic primitive-runtime result, not broad open-domain LLM quality.

## 2026-05-05: Fixed Operation To General Recursive Core Roadmap

Added a roadmap that explicitly separates the accepted fixed-operation
primitive probe from the final QTRM target:

```text
docs/wiki/decisions/fixed-operation-to-general-recursive-core-roadmap.md
```

The documented stage path is:

```text
fixed operation policy
-> learned latent operation codebook
-> neural state-transition model
-> open-ended recursive reasoning core
-> memory/metacognition composition gates
```

The roadmap also adds promotion, rejection, and workflow checklists so future
work does not confuse a neuro-symbolic scaffold with broad open-ended
reasoning.

## 2026-05-05: Larger OOD Paraphrase Stress Recovery

Added a larger paraphrase stress builder and heldout-separated train/eval split:

```text
scripts/227_build_pure_recursive_ood_paraphrase_stress_cases.py
data/eval/pure_recursive_primitive_transition_ood_paraphrase_stress_heldout10000_preferences.jsonl
data/filtered/pure_recursive_primitive_transition_ood_paraphrase_stress_train11000_preferences.jsonl
```

Baseline from the surface-augmentation checkpoint:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt

stress heldout10000:
  total:            191/256 = 0.7461
  arithmetic_chain:  58/64
  symbolic_binding:  50/64
  boolean_logic:     64/64
  list_transform:    19/64

core-off:
  total:              0/256
```

Trained a heldout-separated recovery checkpoint using only the primitive
transition operation CE path:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/last.pt
```

Recovered results:

```text
stress heldout10000 raw argmax:
  total:            252/256 = 0.9844
  arithmetic_chain:  64/64
  symbolic_binding:  64/64
  boolean_logic:     64/64
  list_transform:    60/64

stress heldout10000 with state-constrained operation decoding:
  total:            256/256 = 1.0000

canonical heldout7000:
  total:            128/128 = 1.0000

stress core-off:
  total:              0/256 = 0.0000

stress state-constrained core-off:
  total:              0/256 = 0.0000
```

Residual failure:

```text
list_transform variant 7:
  4/8

observed wrong operation sequence:
  filter_even, multiply_sum, hold_final, hold_final

needed second operation:
  double_filtered
```

Interpretation:
The larger paraphrase failure is mostly recoverable through the mandatory
recursive primitive-transition path without canonical regression. The remaining
raw failures are invalid operation choices after a correct first list step. A
state-constrained operation decoder masks those invalid choices and closes the
runtime answer gate to 256/256 while keeping core-off at 0/256. This strengthens
the Stage 0 scaffold runtime, but the raw operation argmax bottleneck remains.
The next raw-intelligence proof is operation-family transfer or a raw
list-transform routing fix, not ASI-scale claims.

## 2026-05-05: Operation-Family Holdout Feasibility Gate

Added a diagnostic builder for fixed-operation family holdout:

```text
scripts/228_build_pure_recursive_operation_family_holdout.py
data/filtered/pure_recursive_primitive_transition_family_holdout_list_train.jsonl
data/eval/pure_recursive_primitive_transition_family_holdout_list_eval.jsonl
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/operation_family_holdout_list_summary.json
```

Generated split:

```text
holdout family: list_transform
train rows:     192
eval rows:       64

unseen eval operations:
  double_filtered, filter_even

shared eval operation:
  hold_final
```

Conclusion:
Full list-family holdout is a reject/diagnostic gate for the current
fixed-label primitive scaffold, not a fair acceptance gate. The eval set needs
`filter_even` and `double_filtered`, but the train split never contains those
operation labels. More training on this split would mostly test impossible
label invention, not recursive raw intelligence.

The architecture implication is to keep the current result as Stage 0 causal
operation routing over supported primitives and move promotion work toward a
learned latent operation codebook or neural transition model.

## 2026-05-05: Stage 1 Latent Action Codebook Feasibility

Added a Stage 1 diagnostic builder that replaces fixed operation-string
targets with family-agnostic latent action roles:

```text
scripts/229_build_pure_recursive_latent_action_codebook_cases.py
data/filtered/pure_recursive_latent_action_codebook_family_holdout_list_train.jsonl
data/eval/pure_recursive_latent_action_codebook_family_holdout_list_eval.jsonl
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/latent_action_codebook_family_holdout_list_summary.json
```

Latent action codebook:

```text
0: extract_or_unary_transform
1: compose_from_previous
2: final_compose_from_previous
3: hold_final
```

Generated split:

```text
holdout family: list_transform
train rows:     192
eval rows:       64

fixed operation view:
  unseen eval operations:
    double_filtered, filter_even

latent action view:
  train latent action codes:
    0, 1, 2, 3

  eval latent action codes:
    0, 1, 3

  unseen eval latent action codes:
    none
```

Conclusion:
The latent-action codebook removes the immediate impossibility found in the
fixed-label operation-family holdout. The list eval family still contains
unseen fixed operations, but it no longer contains unseen latent action codes.

This is a feasibility result only. It does not prove neural execution or broad
reasoning. The next required experiment is to train the `transition_state_code`
path on this split and run core-off, transition-state-off, and action-code
shuffle/dropout ablations.

## 2026-05-05: Latent Action Codebook S120 Rejection

Trained the Stage 1 `transition_state_code` path on list-family holdout splits.
The objective was code CE only; answer/depth CE weights were set to zero.

Artifacts:

```text
v1 config:
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_s120.yaml

v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/last.pt

v1 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_transition_off.json

v2 config:
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_v2_s120.yaml

v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/last.pt

v2 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_transition_off.json
```

Results:

```text
role_v1:
  full step_acc:                 0.7500
  transition-state-off step_acc: 0.2500
  trace exact:                   0/64

terminal_v2:
  full step_acc:                 0.5000
  transition-state-off step_acc: 0.2500
  trace exact:                   0/64
```

Failure pattern:

```text
role_v1:
  depth 1, 2, 8 correct
  depth 4 wrong: predicted final_compose, target hold

terminal_v2:
  depth 1, 8 correct
  depth 2 wrong: predicted nonterminal compose, target terminal compose
  depth 4 wrong: predicted final compose, target hold
```

Conclusion:
The code path is causal because full beats transition-state-off on step
accuracy, but Stage 1 is rejected because trace exact is still 0/64. The
remaining blocker is prompt-grounded termination and semantic transition
grounding for unseen families, not merely the number or names of latent action
codes.

## 2026-05-05: Transition-State Finality S120 Narrow Acceptance

Trained a finality-only transition-state head on the same list-family holdout
split. The run used finality BCE only; answer and code losses were disabled.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_finality_s120.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_transition_off.json
```

Results:

```text
full finality step_acc:                 0.7500
transition-state-off finality step_acc: 0.2500
trace exact:                            0/64
```

Decision:
Accept the finality head only as a narrow causal signal. Reject it as Stage 1
reasoning progress because trace exact is still 0/64 and action/semantic
transition reasoning remains unsolved.

## 2026-05-05: Transition-State Text S120 Rejection

Trained a transition-state text head on the same list-family holdout split.
This tests semantic intermediate-state token prediction without fixed operation
ids.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_text_s120.yaml

eval script:
  scripts/231_eval_qtrm_transition_state_text.py

low-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_s120_from_oodstress/last.pt

high-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_lr1e3_s120_from_oodstress/last.pt
```

Results:

```text
lr=5e-5:
  full step_acc:                 0.0000
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64

lr=1e-3:
  full step_acc:                 0.2500
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64
```

Decision:
Accept only the narrow causal semantic-token signal from the high-lr run.
Reject recursive semantic transition because the model collapses to the depth-1
token at every depth and does not learn the depth-varying state update.

## 2026-05-05: Transition-State Text Depth-Contrast Rejection

Added a row-local depth-contrast loss to separate labelled intermediate-state
tokens across recurrent depths.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/eval_transition_text_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/eval_transition_text_list_holdout_transition_off.json
```

Results:

```text
full step_acc:                 0.2500
transition-state-off step_acc: 0.0000
trace exact:                   0/64
```

Decision:
Reject this as Stage 1 progress. The path is causal, but it still predicts the
same depth-1 content token at every depth. Do not keep stacking scalar losses
onto the frozen full-vocabulary projection. The next candidate is a compact
semantic state head or state embedding that the recurrent core must update
before any language-vocabulary readout.

## 2026-05-05: Code+Finality And Joint-State Rejection

Tested two compact alternatives after the full-vocabulary transition-text path
collapsed:

```text
code + finality checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_finality_s120_from_oodstress/last.pt

joint checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_s120_from_oodstress/last.pt
```

Results on the held-out list-family split:

```text
code + finality:
  full code step_acc:      0.7500
  full finality step_acc:  0.7500
  full halted exact:       0/64
  off code/finality acc:   0.2500 / 0.2500

joint:
  full code step_acc:      0.7500
  full finality step_acc:  0.7500
  full halted exact:       0/64
  off code/finality acc:   0.2500 / 0.2500
```

Decision:
Reject both as Stage 1 promotion. They prove a causal transition-state path,
but strict trace exact and halted exact remain 0/64. The joint head shows the
current bottleneck more clearly: it gets the list prefix right, then fires a
final/hold state at an unlabelled intermediate step. The next experiment should
use dense 1..N transition targets instead of sparse 1/2/4/8 labels.

## 2026-05-05: Dense Joint-State Rejection

Added dense 1..N transition targets and retrained compact joint-state heads for
both role_v1 and terminal_v2 codebooks.

Artifacts:

```text
builder:
  scripts/232_build_dense_transition_targets.py

decision:
  docs/wiki/decisions/transition-joint-dense-s120.md

role_v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/last.pt

terminal_v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/last.pt
```

Results:

```text
role_v1 dense:
  full step_acc:      0.8750
  off step_acc:       0.1250
  halted exact:       0/64

terminal_v2 dense:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64
```

Decision:
Reject as Stage 1 promotion. Dense targets prove the transition-state path is
causal, but held-out list still maps to the arithmetic/nonterminal compose path
at depth 2 and strict halted trace exact remains 0/64. The next bottleneck is
prompt-grounded terminal routing and semantic family transfer, not just sparse
transition labels.

## 2026-05-05: Terminality Fix And All-Families Sanity Control

Added terminality counterfactual data and an action-terminal finality mode for
dense targets.

Artifacts:

```text
builder:
  scripts/233_build_terminality_counterfactual_targets.py

decision:
  docs/wiki/decisions/transition-joint-terminality-and-sanity-s120.md
```

Results:

```text
terminality augmentation, list still fully held out:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64

all-families sanity control, list included in train but held out by index:
  full step_acc:      1.0000
  off step_acc:       0.1250
  halted exact:       64/64
```

Decision:
Accept only the all-families mechanism sanity control. The recurrent
transition-state path can learn list traces and is causally necessary, but full
list-family zero-shot transfer remains rejected.

## 2026-05-05: Core Role-Value Transition Auxiliary Rejected

Added a disabled-by-default `core_role_value_transition_logits` path and
`--algorithmic-role-value-transition-ce-weight` to test whether previous
role-state to next-role-state supervision improves the mandatory latent loop.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-role-value-transition-aux-s120.md

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_transition_joint_s120.yaml

rejected checkpoints:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_from_len579_s240/last.pt
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_w01_from_len579_s240/last.pt
```

Results:

```text
canonical baseline:
  value:     184/624 = 0.2949
  step exact: 16/256 = 0.0625

transition aux weight 1.0:
  value:      96/624 = 0.1538
  step exact:  0/256

transition aux weight 0.1:
  value:      96/624 = 0.1538
  step exact:  0/256

baseline loaded with transition-enabled config:
  value:     184/624 = 0.2949
  step exact: 16/256 = 0.0625
```

Decision:
Reject the transition auxiliary as canonical. The module presence is safe, but
auxiliary training regresses held-out value state. Core-step sweep also shows
that current role-value slots are a structured probe/scaffold, not yet a
depth-improving raw recursive reasoning mechanism.

## 2026-05-05: List Paraphrase Transfer Gate Accepted

Built a fairer Stage 1 list transfer gate where list_transform is present in
train, but list paraphrase variants 6 and 7 are held out.

Artifacts:

```text
builder:
  scripts/234_build_list_transfer_gate.py

decision:
  docs/wiki/decisions/transition-joint-list-transfer-s120.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/last.pt
```

Results:

```text
full:
  exact:         16/16
  halted exact:  16/16

transition-state-off:
  exact:         0/16
  halted exact:  0/16

code shuffle:
  exact:         0/16
  halted exact:  0/16

code dropout:
  exact:         0/16
  halted exact:  0/16
```

Decision:
Accept as within-family list-surface transfer. Do not claim full family-zero-shot
reasoning; that remains rejected.

## 2026-05-05: Core State-Carry Role-Value Rejected

Added a gated recurrent carry update on the core role-value token slice plus a
`core_state_carry_only` trainable policy to separate architecture damage from
baseline-preserving adapter training.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-state-carry-role-value-s120.md

configs:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_carry_joint_s120.yaml
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_state_carry_only_s120.yaml

eval ablation:
  scripts/238_eval_qtrm_algorithmic_value_state.py --disable-core-state-carry
```

Results:

```text
canonical baseline:
  value:      184/624 = 0.2949
  step exact:  16/256 = 0.0625
  action:      32/32

core_and_role_value_state + carry:
  value:       80/624 = 0.1282
  step exact:   0/256

same checkpoint, carry disabled at eval:
  value:       80/624 = 0.1282
  step exact:   0/256

core_state_carry_only:
  value:      158/624 = 0.2532
  step exact:  16/256 = 0.0625
  action:      32/32
```

Decision:
Reject state-carry as canonical. Carry-only proves the safer baseline-preserving
workflow, but the module does not beat the held-out value-state baseline. The
next candidate should be a compact value-delta state rather than a free MLP
carry over role tokens.

## 2026-05-05: Core Role-Value Delta-Only Rejected

Added a small recurrent delta adapter over core role-value token trajectories,
plus `core_role_value_delta_only` training and a delta-off value eval ablation.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-role-value-delta-only-s120.md

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_delta_only_s120.yaml

checkpoints:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_from_len579_s240/last.pt
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_lr1e5_from_len579_s240/last.pt
```

Results:

```text
baseline / untrained delta config:
  value:      184/624 = 0.2949
  step exact:  16/256

delta-only lr 1e-4:
  value:      112/624 = 0.1795
  step exact:   0/256
  delta off:  184/624 = 0.2949
  action:      32/32

delta-only lr 1e-5:
  value:      184/624 = 0.2949
  step exact:  16/256
```

Decision:
Reject as canonical. The adapter is isolated and baseline-preserving, but it
does not improve held-out value-state reasoning. The next candidate should make
the missing state variable explicit: action code -> typed value-delta code ->
value-state update -> role-value logits.

## 2026-05-05: Breakthrough-Prior Architecture Pivot

The repeated failure of local carry/delta MLPs triggers the big-structure doubt
gate. The next candidate is no longer another continuous hidden-state adapter.

Artifacts:

```text
decision:
  docs/wiki/decisions/breakthrough-prior-next-architecture.md

downloaded papers:
  references/papers/recurrent_depth/loopformer_2602.11451.pdf
  references/papers/recurrent_depth/ouro_looplm_2510.25741.pdf
  references/papers/recurrent_depth/rltt_latent_thought_trajectory_2602.10520.pdf
  references/papers/recurrent_depth/looprpt_2603.19714.pdf
  references/papers/role_value_slots/discrete_neural_algorithmic_reasoning_icml2025.pdf
  references/papers/role_value_slots/transnar_transformers_meet_nar_2406.09308.pdf

official code:
  references/official/loopformer @ 59a8ae8
  references/official/dnar @ 12f3f0b
  references/official/clrs @ bfd042f
```

Prior-backed pivot:

```text
Discrete NAR-style typed execution bottleneck
+ LoopFormer/Ouro/RLTT-style trajectory credit
+ QTRM mandatory recursive core
```

Next falsifiable experiment:

```text
core_depth_state
-> value_delta_code_logits
-> straight-through one-hot value_delta_code
-> typed register update features
-> role-value logits
```

Implementation scaffold:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml

model:
  core_value_delta_code_logits
  core_value_delta_code_gate_mean

training:
  trainable_param_policy: core_value_delta_code_only
  --core-value-delta-code-ce-weight

eval:
  --disable-core-value-delta-code
```

Acceptance:

```text
held-out value accuracy > 184/624
step exact             > 16/256
trace exact            > 0/32
action-code exact      = 32/32
delta-code ablation    drops below full
depth 8                beats depth 1 on held-out value/step metrics
```

## 2026-05-05: Core Value-Delta Code Only S120 Rejection

Decision:

```text
docs/wiki/decisions/core-value-delta-code-only-s120.md
```

Result:

```text
full trained path:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

code-off ablation:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

direct code-logit readout:
  value accuracy:  63/624 = 0.1009615385
  step exact:       0/256 = 0.0
  trace exact:      0/32

action-code:
  exact: 32/32
```

Decision:

```text
Reject as canonical. The finite code path does not carry held-out value-state
signal, and deeper recurrence does not improve value-state accuracy.
```

Next:

```text
Stop adding local hidden-state adapters for value execution. Rebuild the next
candidate as a TransNAR/CLRS-style typed register executor:

prompt -> role binder -> mandatory core -> operation selector ->
typed registers -> verifier-checked update -> role-value readout
```

## 2026-05-05: Typed Register Executor Only S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-executor-only-s120.md
```

Implemented the first persistent typed-register executor while preserving the
universal LLM causal path:

```text
prompt tokens
-> QTRM core trajectory
-> learned operation selector
-> persistent typed registers
-> role-value logits
```

Results:

```text
untrained typed-register:
  value accuracy:   0/624 = 0.0

trained full:
  value accuracy: 106/624 = 0.1698717949
  step exact:       0/256
  trace exact:      0/32

typed-register-off:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256

action-code:
  exact: 32/32
```

Decision:

```text
Reject as canonical. The executor is causal but harmful: disabling it restores
the stronger baseline. Keep the scaffold, but the next candidate needs
operation/process supervision rather than value CE alone.
```

## 2026-05-05: Typed Register Executor V2 Process-Credit Rejection

Decision:

```text
docs/wiki/decisions/typed-register-executor-v2-process-credit-plan.md
```

Research-driven update:

```text
Use LoopLM/latent lookahead/process-credit prior to add training-only process
CE on the existing typed-register operation selector.
```

Implementation target:

```text
core_typed_register_operation_logits
+ transition_state_codes[depth] CE
+ role-value CE
```

Result:

```text
full value accuracy:       102/624 = 0.1634615385
typed-register-off:        184/624 = 0.2948717949
full step exact:             0/256
typed-register-off exact:   16/256
action-code exact:          32/32
```

Decision:

```text
Reject. Process-code CE preserves the action controller but the typed-register
value path remains harmful. Next candidate needs register-transition
consistency and latent-lookahead/process reward, not another isolated value
head.
```

## 2026-05-05: Typed Register Prompt Binder S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-prompt-binder-s120.md
```

Hypothesis tested:

```text
Maybe typed registers fail because they only see mean-pooled context and cannot
bind exact prompt values.
```

Result:

```text
baseline value accuracy:      184/624
prompt-binder full:           104/624
prompt-binder register-off:   168/624
action-code exact:             32/32
```

Decision:

```text
Reject. Token-addressed prompt access alone is not enough and also disturbs the
previous role-value baseline. The next candidate should make exact value update
a recurrent transition objective, not another readout/binder patch.
```

## 2026-05-05: Typed Register Transition Consistency S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-transition-consistency-s120.md
```

Hypothesis:

```text
The typed-register path needs teacher-forced recurrent value-transition credit:
register_state[t] -> role_value_state[t+1].
```

Implementation:

```text
core_typed_register_transition_logits
--core-typed-register-transition-ce-weight
```

Result:

```text
full value accuracy:      104/624
typed-register-off:       184/624
action-code exact:         32/32
trace exact:                0/32
```

Decision:

```text
Reject. The transition head was auxiliary and did not become the evaluated
value path. Next candidate must make depth>1 value readout come from the
previous-register transition prediction itself.
```

## 2026-05-05: Typed Register Strict Transition Readout S120 Plan

Decision:

```text
docs/wiki/decisions/typed-register-strict-transition-readout-s120.md
```

Hypothesis:

```text
Depth > 1 role-value logits must be the transition prediction itself, not an
auxiliary head beside the evaluated value head.
```

Implementation:

```text
core_typed_register_transition_readout_enabled
```

Result:

```text
full value accuracy:      64/624 = 0.1025641026
typed-register-off:      184/624 = 0.2948717949
full step exact:           0/256
typed-register-off exact: 16/256
action-code exact:        32/32
```

Decision:

```text
Reject. Making the transition head the causal value readout worsens the value
path. Since typed-register-off restores the baseline exactly, the bottleneck is
not prompt access or auxiliary loss wiring; the typed-register value mechanism
itself is the wrong local patch. Next work should move back to the universal
LLM causal path and test a root latent-state candidate instead of another
parallel role-vocab head.
```

## 2026-05-05: Final Answer Bridge S120 Opened

Decision:

```text
docs/wiki/decisions/final-answer-bridge-s120.md
```

Failure:

```text
The best len579 value-state checkpoint scores 0/32 on mixed-composition
causal forced-choice for every tested mode. It ranks intermediate list states
above final scalar answers.
```

Next experiment:

```text
Train answer_state_loop LM logits with causal-prefix final-answer CE while
preserving transition_state_joint CE. No typed-register or role-value answer
channel is active.
```

Result:

```text
action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000

LM causal forced-choice smoke8:
  donor_only:   0/8
  core_off:     0/8
  core_steps_1: 0/8
  core_steps_8: 0/8
```

Decision:

```text
Reject. Final-answer CE lowers training loss and preserves the action policy,
but it does not transfer to held-out final-value answers. The next candidate
must put a neural value-state transition into the causal answer path; answer
rendering alone is not enough.
```

## 2026-05-05: Role-Value Answer Bridge S120 Opened

Decision:

```text
docs/wiki/decisions/role-value-answer-bridge-s120.md
```

Architecture change:

```text
core_role_value_state_logits
-> soft value embeddings + role embeddings
-> gated internal answer-loop tokens
-> answer_state_loop LM logits
```

Why:

```text
The previous final-answer bridge trained the renderer but did not make the
computed value state causal. This probe forces the answer loop to read the
core's role-value state through a learned, ablatable internal bottleneck.
```

Promotion gate:

```text
action-code exact remains 32/32
core_steps_8 LM forced-choice beats donor_only and core_off
role_value_answer_bridge_off causes a held-out drop
```

Result:

```text
S80 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_role_value_answer_bridge_len579_s080_from_len579_s240/last.pt

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000

LM causal forced-choice smoke8:
  donor_only:              0/8
  core_off:                0/8
  core_steps_8:            0/8
  core_steps_8 bridge_off: 0/8
```

Decision:

```text
Reject. The bridge trains and preserves action-code, but bridge-off does not
change held-out LM answer selection. Next candidate should move to an
Ouro/LoopLM-style recurrent answer-hidden-state block with depth allocation,
not another side-head bridge.
```

## 2026-05-06: Ouro Answer Recurrent S080 Opened

Decision:

```text
docs/wiki/decisions/ouro-answer-recurrent-s080.md
```

Architecture change:

```text
answer_state_loop cross-attention
-> shared causal recurrent answer block
-> LM logits
```

Why:

```text
Ouro/LoopLM is the better general-LLM prior than task-specific TRM. The next
probe updates the answer hidden state itself instead of feeding side-state
tokens into an otherwise unchanged answer loop.
```

Promotion gate:

```text
action-code exact remains 32/32
core_steps_8 beats donor_only/core_off on held-out LM causal forced-choice
answer_state_recurrent_off drops versus full
```

Result:

```text
S80 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s080_from_len579_s240/last.pt

training:
  final_path_ce: 10.9580 -> 2.6068
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       2/8
  core_steps_8 answer_recurrent_off:  0/8
```

Decision:

```text
Accept as a smoke-level causal gain. This is not a general reasoning claim yet,
but it is the first positive answer-path ablation in this branch. Continue to
S240/S480 before adding a SubQ/MSA-like selective context router.
```

S240 continuation result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s240_from_s080/last.pt

training:
  final_path_ce: 4.3030 -> 1.0415
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       0/8
  core_steps_8 answer_recurrent_off:  0/8
```

Decision update:

```text
Reject S240 continuation. CE-only continuation improves in-sample loss but
erases the S80 held-out answer-path gain. Keep S80 as the best observed
checkpoint and add validation-gated selection or stronger final-value/ranking
loss before longer training.
```

Implementation follow-up:

```text
Added --save-every to scripts/196_train_pure_recursive_depth_supervised.py.
Future recurrent-answer runs should save step snapshots and evaluate LM
causal forced-choice before overwriting the best observed answer-path state.
```

Choice-margin continuation result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_choice_margin_len579_s080_from_s080/step_000080.pt

training:
  final_path_ce: 0.9753 on final logged batch
  final_path_acc: 1.0000
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  step40 core_steps_8:                       0/8
  step40 core_steps_8 answer_recurrent_off:  0/8
  step80 core_steps_8:                       0/8
```

Decision update:

```text
Reject choice-margin continuation as a canonical recurrent-answer objective.
The implementation is useful infrastructure, but this run repeats the S240
failure: transition/action state remains perfect while held-out LM answer
selection loses the accepted S80 causal gain.
```

## 2026-05-06: Ablation-Aware Checkpoint Gate And Selective Router Probe

Added an evaluation artifact selector:

```text
scripts/240_select_qtrm_checkpoint_by_gate.py
```

It selects checkpoints by held-out LM causal forced-choice, action-code
preservation, and optional component ablation drop. This prevents accepting a
lower training CE or equal-score component that is not causally useful.

Selector result:

```text
S80 baseline:
  full:            2/8
  recurrent_off:   0/8
  action-code:     32/32
  accepted:        true

S240 CE-only:
  full:            0/8
  action-code:     32/32
  accepted:        false

choice-margin step80:
  full:            0/8
  action-code:     32/32
  accepted:        false
```

Implemented the minimal SubQ/SSA-style answer selective context router:

```text
answer hidden state
+ core/workspace states
+ prompt states
-> learned top-k selector
-> answer-state cross-attention
-> recurrent answer block
-> LM logits
```

Probe result:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_s080.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_len579_s020_from_s080/step_000020.pt

LM causal forced-choice:
  full:       2/8
  router_off: 2/8

action-code:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
```

Decision:

```text
Reject selective_s020 as a causal architecture gain. It preserves S80 but
router-off does not hurt. Keep the router code as a scaffold; future router
runs need dense-vs-sparse alignment or explicit router supervision before
claiming a SubQ/SSA-style improvement.
```

Decision doc:

```text
docs/wiki/decisions/ouro-selective-context-s020.md
```

Dense-alignment v2 update:

```text
change:
  added force_answer_state_loop_dense_context
  added --answer-selective-context-alignment-weight
  added answer_selective_context_alignment_loss

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_align_len579_s020_from_s080/step_000020.pt

training:
  final_path_ce: 4.3567 -> 2.0881
  answer_selective_context_alignment_kl: 0.0228 -> 0.0198

LM causal forced-choice:
  full:       2/8
  router_off: 2/8

action-code:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
```

Decision:

```text
Reject dense-alignment S020. The KL teacher path is implemented, but router-off
still ties full mode, so the router has no causal answer-path contribution.
S80 Ouro recurrent remains the selected baseline.
```

## 2026-05-06 - Ouro Causal-Prefix Tail S020

Failure isolated:

```text
mixed-list arithmetic often computes the doubled sum but skips the final
subtract step, e.g. gold 300015 vs model 300024.
```

The hard negative was already present in `choices`, so the next falsifier used
causal-prefix multi-token supervision and sequence margin instead of another
dataset-only change.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/action_code_eval32.json
```

Result:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   3/8
full core8:   4/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     3
  pre_subtract_sum:  4
  tie:               1
```

Decision:

```text
Accept as a smoke improvement, not a broad breakthrough.
The next bottleneck remains final subtract-tail retention.
```

Storage hygiene:

```text
Deleted generated local_eval/**/step_*.pt snapshots after / reached 100%.
Kept last.pt checkpoints and eval JSON artifacts.
```

## 2026-05-06 - All-Prefix Bridge Tail Reject

Implemented:

```text
--transition-joint-answer-bridge-contrast-all-prefix-tokens
```

Rationale:

```text
The accepted tail checkpoint improved full to 4/8, but bridge-off also reached
3/8. The hypothesis was that bridge contrast only touched the first answer
token, while the final subtract error lives in later answer digits.
```

Result:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   2/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     2
  pre_subtract_sum:  4
  doubled_list:      2
```

Decision:

```text
Reject. All-prefix bridge contrast preserves the action controller but regresses
the answer path and eliminates the full-vs-bridge-off gap.
```

Decision doc:

```text
docs/wiki/decisions/ouro-allprefix-bridge-tail-s020-reject.md
```

## 2026-05-06 - Tail-Negative S020

Implemented a narrow preterminal-state negative objective:

```text
--tail-negative-margin-weight
--tail-negative-margin
--tail-negative-family-filter
```

Accepted smoke:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/last.pt

donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   4/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     2
  pre_subtract_sum:  6
```

Decision:

```text
Accept as causal-gap improvement only. It keeps full 4/8 and lowers bridge-off
from 3/8 to 2/8, but it does not reduce full pre-subtract failures.
```

Rejected follow-up:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/last.pt

mixedx4 S040:
  donor_only:   0/8
  core_off:     0/8
  bridge_off:   4/8
  full core8:   3/8
  action-code: 32/32
```

Decision doc:

```text
docs/wiki/decisions/ouro-tail-negative-s020.md
```

## 2026-05-06 - Ouro Final-Answer Binder Reject

Implemented an ablatable final-answer binder and tested two variants:

```text
v1: transition joint-code/finality distribution -> answer delta
v2: finality-weighted core_depth_state -> answer delta
```

Both were trained for S020 from the accepted tail-negative checkpoint and
evaluated with the same smoke8 causal forced-choice gate.

Results:

```text
v1:
  donor_only:        0/8
  core_off:          0/8
  full core8:        2/8
  binder_off:        2/8
  joint_bridge_off:  2/8

v2:
  donor_only:        0/8
  core_off:          0/8
  full core8:        2/8
  binder_off:        2/8
  joint_bridge_off:  2/8
```

Decision:

```text
Reject. The accepted baseline is still 4/8, and binder-off ties full. The next
architecture step should train latent trajectory quality directly, using
LoopLM/COCONUT/LoopFormer/LoopRPT-style process credit rather than adding
another answer-side readout patch.
```

Decision doc:

```text
docs/wiki/decisions/ouro-final-answer-binder-s020-reject.md
```

## 2026-05-06 - Ouro Answer Halt Head S080 Accepted

Implemented a PonderNet/ACT-style answer-state halt head and tested two
variants:

```text
rejected:
  transition finality -> in-loop answer freeze
  result: full 2/8, gate_off 4/8

accepted:
  answer hidden -> learned halt head
  train with halt gate disabled
  eval with hard-first in-loop halt gate
```

Key metrics:

```text
smoke8:
  core_steps4:       8/8
  core_steps8 full:  8/8
  halt_gate_off:     0/8
  bridge_off:        8/8

smoke16:
  core_steps4:       10/16
  core_steps8 full:  10/16
  halt_gate_off:      0/16
  bridge_off:        10/16

action-code:
  exact:             32/32
  step_acc:          1.0
  finality_acc:      1.0
  halted_exact:      32/32

generation smoke4:
  full/gate_off:      0/8
```

Decision:

```text
Accept as the current Ouro raw-recursive answer-path candidate. The answer
halt gate is causal because disabling it collapses the answer path to 0/16.
Demote transition_joint_answer_bridge for this checkpoint because bridge_off
ties full. Do not claim generation readiness: greedy generation still fails.
```

Decision doc:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Typed CE Demoted To Probe-Only

Scope correction:

```text
typed algorithmic CE is not a universal LLM objective.
```

It remains useful for diagnosing numeric/register binding, but canonical QTRM
claims must improve the single prompt-token-to-LM-logits path.

The current canonical raw-recursive answer-path baseline is now recorded as a
machine gate:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080-raw-gate.md

donor_only: 0/8
core_off:   0/8
depth1:     0/8
depth2:     0/8
depth4:     8/8
depth8:     8/8
```

Next bottleneck:

```text
greedy autoregressive rendering remains 0/8 while forced-choice is accepted.
Improve the answer-token path without typed answer channels or donor-logit
shortcuts.
```

## 2026-05-07 - Ouro Halt-Head Partial Scale32 Depth Gate

Attempted a full scale32 forced-choice sweep for the accepted halt-head
checkpoint. The full 8-mode sweep was stopped after 170/256 rows because it was
too slow for the current loop.

The completed depth4 subset was preserved as a partial gate:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080-depth4-scale32-partial-gate.md

donor_only: 0/32
core_off:   0/32
depth1:     4/32
depth2:     4/32
depth4:    16/32
```

Decision:

```text
accept as partial depth-scaling evidence only.
not accepted as full scale32 halt-off/depth8 verification.
```

## 2026-05-07 - Renderer Root Redesign Toward Latent Lookahead

Web-search-backed prior check:

```text
Latent Lookahead Training
PonderLM-2
Reasoning with Latent Tokens in Diffusion LMs
Parcae stable looped models
Autoregressive LMs as EBMs
Hidden Capacity for One-Step Text Generation
Draft/Verify/Improve and Weaver
```

New diagnostic:

```text
scripts/247_probe_qtrm_gold_token_ranks.py
```

Accepted halt-head rank probe, 4 cases:

```text
core_steps_4:
  first@1: 4/4
  all@1:   0/4
  all<=10: 0/4
  max-rank mean: 740

example answer 300015:
  ranks: [1, 14, 5, 5, 5, 4, 1222]
```

Donor-preserving core-forced generation smoke:

```text
hits: 0/48
```

Decision:

```text
Stop stacking LM-head/bridge/logit-scale patches.
Next candidate is same-prefix latent lookahead / future-token auxiliary:
the answer-state loop must learn the next K answer tokens from the same
prefix, while runtime output remains normal LM logits/autoregressive text.
```

Docs:

```text
docs/wiki/sources/latent-lookahead-renderer.md
docs/wiki/decisions/qtrm-renderer-root-redesign-latent-lookahead.md
```

## 2026-05-07 - Depth Router Heads Rejected, Trajectory Training Promoted

Result:

```text
fixed-depth oracle from core-forced readout:
  16/24

best final-state route head:
  12/24

best core_depth_states trajectory route head:
  12/24
```

Decision:

```text
Reject route-head-only as the depth selector path.
The core has useful fixed-depth behavior, but frozen core states do not expose
a reliable routing signal to a small classifier.
```

Implementation note:

```text
Added controller_signal_source=learned_core_trajectory so diagnostic heads can
read the full core_depth_states sequence.
```

Next candidate:

```text
LoopFormer-style variable trajectory / shortcut-consistency training for the
recursive core itself, with depth sweeps and core-off/delta-off ablations.
```

Decision/source docs:

```text
docs/wiki/decisions/donor-preserving-controller-next-method.md
docs/wiki/sources/adaptive-depth-test-time-compute.md
```

## 2026-05-06 - Ouro Renderer Probes Rejected

Tested three ways to turn the accepted answer-halt S080 forced-choice scorer
into a greedy autoregressive renderer:

```text
naive answer-loop CE:
  generation:           0/8
  causal forced-choice: 0/8
  decision:             reject, checkpoint deleted

zero-init LM adapter:
  generation:           0/8
  causal forced-choice: full 8/8, halt_gate_off 0/8
  decision:             reject as renderer, checkpoint deleted

greedy-margin adapter:
  generation:           0/8
  causal forced-choice: full 4/8, halt_gate_off 0/8
  sample:               "24434264752"
  decision:             reject, checkpoint deleted

donor-logit fusion sanity:
  donor_logits_scale=1.0, qtrm_logits_scale=1.0
  generation: 0/4
```

Conclusion:

```text
The halt-head checkpoint remains canonical for raw forced-choice reasoning.
Greedy rendering is a separate causal bottleneck. Output-only patches either
do nothing or damage the accepted gate. The next candidate needs
autoregressive rollout/prefix training while preserving the halt-head
forced-choice baseline.
```

Decision doc:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Latent Lookahead Renderer Scaffold

Implemented the next renderer falsifier:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/247_probe_qtrm_gold_token_ranks.py
scripts/248_run_qtrm_ouro_future_token_lookahead_s040.sh
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_future_token_lookahead_s040.yaml
```

Decision:

```text
same-prefix future-token lookahead is auxiliary-only
runtime answer path remains outputs["logits"] autoregressive LM generation
future-token CE requires causal-prefix supervision to avoid answer leakage
rank probe now separates strict rank from unique_top1 to avoid all-zero tie false positives
```

Decision doc:

```text
docs/wiki/decisions/qtrm-renderer-root-redesign-latent-lookahead.md
```

## 2026-05-07 - Causal Talker S040 Reject

Implemented a canonical-path Talker renderer:

```text
QTRM answer-state hidden + latent trajectory summary
-> causal Talker block
-> LM head
-> autoregressive logits
```

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040.yaml
scripts/249_run_qtrm_ouro_causal_talker_s040.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8: 0/8

causal forced-choice smoke4:
  donor/core_off/core4/core8/halt_off: 0/4
```

Decision:

```text
reject S040 as a checkpoint.
The code path is cleaner than aux lookahead, but answer-loop-only S040 training
does not create a usable LM-compatible renderer and regresses the accepted
forced-choice signal.
```

Decision doc:

```text
docs/wiki/decisions/ouro-causal-talker-s040-reject.md
```

## 2026-05-07 - Causal Talker-Only S080 Reject

Narrowed the previous Talker experiment by freezing the accepted halt-head
checkpoint and training only the new causal Talker parameters.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080.yaml
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_eval_gate.yaml
scripts/250_run_qtrm_ouro_causal_talker_only_s080.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8/core8_halt_off/core8_talker_off: 0/8

causal forced-choice smoke4 with answer halt gate enabled:
  donor/core_off: 0/4
  core4: 4/4
  core8: 4/4
  core8_halt_off: 0/4
  core8_talker_off: 4/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The forced-choice signal is preserved by the answer halt gate, not caused by
the Talker, and greedy generation remains 0/8.
```

Decision doc:

```text
docs/wiki/decisions/ouro-causal-talker-only-s080-reject.md
```

## 2026-05-07 - Donor-Guided Adapter S060 Reject

Fixed the depth-supervised trainer so donor-preserving configs actually pass
donor logits into the QTRM forward path. Then tested a low-rank answer-state LM
adapter with Qwen as the renderer:

```text
final_logits = donor_logits + clamp(answer_halt_state_adapter_delta)
qtrm_logits_scale = 0.0
donor_logits_scale = 1.0
trainable = answer_state_loop_lm_adapter_only
```

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml
scripts/251_run_qtrm_ouro_donor_guided_adapter_s060.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core4/core8/delta_off/halt_off: 0/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The donor-logit path is now correctly trained, but the adapter reinforces
intermediate trace strings instead of final answers and destroys the accepted
halt-gated forced-choice signal.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-adapter-s060-reject.md
```

## 2026-05-07 - Donor-Guided Adapter Final-Only S060 Reject

Retested the donor-guided adapter after removing the staged-target
contamination risk. This run trained only final-answer tokens at depth 8:

```text
target_mode = final
depth_steps = 8
final_logits = donor_logits + clamp(answer_halt_state_adapter_delta)
qtrm_logits_scale = 0.0
donor_logits_scale = 1.0
trainable = answer_state_loop_lm_adapter_only
```

Artifacts:

```text
scripts/252_run_qtrm_ouro_donor_guided_adapter_final_s060.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_final_s060_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/delta_off/halt_off: 0/4

gold-token rank probe:
  accepted halt-head core8 first_unique@1: 4/4
  accepted halt-head core8 all<=10:        0/4
  hard-negative core8 all<=10:             1/4
  hard-negative donor/delta-off all<=10:   3/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
Final-only supervision removes the obvious trace-string contamination but
does not recover the accepted halt-gated forced-choice signal or open greedy
generation. The next target should be a token-local final-answer
discriminator/scorer with hard negatives, not another blind adapter sweep.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-adapter-final-s060-reject.md
```

## 2026-05-07 - Donor-Guided Hard-Negative S080 Reject

Tested the next donor-renderer falsifier with final-only causal-prefix
supervision plus explicit hard negatives:

```text
choice negatives from row.choices
preterminal trace negatives
final-answer +/- 1 numeric counterfactuals
```

Artifacts:

```text
scripts/253_run_qtrm_ouro_donor_guided_hardneg_s080.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/delta_off/halt_off: 0/4
```

Representative core8 misses:

```text
300015 -> 100002
300015 -> 50001
400037 -> 100000
400037 -> 50004
```

Decision:

```text
reject as a promoted renderer checkpoint.
Hard negatives did not recover generation or forced-choice. The bottleneck is
not only final-answer-vs-trace discrimination; answer-state hidden is not
renderer-ready for autoregressive numeric generation under adapter-only tuning.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-hardneg-s080-reject.md
```

## 2026-05-07 - Ouro Next-Token Decoder S080 Reject

Added an in-loop tokenizer-aligned answer decoder before the shared LM head and
trained only that decoder from the accepted answer-halt checkpoint.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080.yaml
scripts/254_run_qtrm_ouro_next_token_decoder_s080.sh
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/
```

Training signal at step 80:

```text
final_path_ce=3.3559
final_path_acc=0.4286
final_greedy_token_margin=0.4388
causal_prefix_self_rollout_examples=0
```

Gold-token rank probe:

```text
donor_only:    first_unique@1 0/4, all<=10 3/4
core_off:      first_unique@1 0/4, all<=10 4/4
core8 full:    first_unique@1 4/4, all<=10 2/4
decoder_off:   first_unique@1 4/4, all<=10 0/4
halt_gate_off: first_unique@1 0/4, all<=10 0/4
```

Smoke gates:

```text
generation smoke8:
  donor/core_off/core8/decoder_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/halt_off: 0/4
  decoder_off: 4/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The decoder improves some teacher-forced rank diagnostics, but the full
runtime path still fails generation and regresses the accepted forced-choice
signal. The next falsifier should add self-rollout causal-prefix correction
instead of another decoder-only sweep.
```

Decision doc:

```text
docs/wiki/decisions/ouro-next-token-decoder-s080-reject.md
```

## 2026-05-07 - Ouro Next-Token Decoder Self-Rollout S040 Reject

Tested generated-prefix correction for the in-loop tokenizer-aligned decoder.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040.yaml
scripts/255_run_qtrm_ouro_next_token_decoder_selfrollout_s040.sh
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040_from_s080/
```

Training signal:

```text
step 1 self-rollout mismatch rate: 1.0000
around step 10 final_path_acc: 0.5000
final step 40 final_path_acc: 1.0000
final step 40 self-rollout mismatch rate: 0.0000
```

Gold-token rank probe:

```text
donor_only:    first_unique@1 0/4, all<=10 3/4
core_off:      first_unique@1 0/4, all<=10 4/4
core8 full:    first_unique@1 4/4, all<=10 2/4
decoder_off:   first_unique@1 4/4, all<=10 0/4
halt_gate_off: first_unique@1 0/4, all<=10 0/4
```

Smoke gates:

```text
generation smoke8:
  donor/core_off/core8/decoder_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/halt_off: 0/4
  decoder_off: 4/4
```

Representative full core8 failure:

```text
300015 -> 1600000
400037 -> 1600000
300032 -> 1600000
400051 -> 1600000
```

Decision:

```text
reject as a promoted renderer checkpoint.
Self-rollout is wired and train metrics move, but decoder-only continuation
does not repair held-out generation or sequence scoring. The next attempt
must train answer-state recurrent block, halt gate, and decoder jointly.
```

Decision doc:

```text
docs/wiki/decisions/ouro-next-token-decoder-selfrollout-s040-reject.md
```

## 2026-05-07 - TRM Canonical Core / Mythos Demotion

Architecture decision:

```text
TRM/QTRM recursive core = primary reasoning core
answer-state loop = readout/renderer/control
Mythos/OpenMythos recurrence = stability reference or rejected probe, not a
second answer-side reasoning core
```

Reason:

```text
The Mythos-style answer-loop joint decoder S040 experiment still has
generation smoke8 0/8, and forced-choice smoke4 shows core8 full 2/4 while
decoder_off reaches 4/4.
```

Decision docs:

```text
docs/wiki/decisions/trm-canonical-core-mythos-demotion.md
docs/wiki/decisions/ouro-answer-loop-joint-decoder-s040-reject.md
```

## 2026-05-07 - TRM Core ACT Runtime Scaffold

Implemented two official-TRM-aligned core mechanics behind config flags:

```text
core_halt_init_bias=-5.0
core_trm_no_grad_inner_cycles_enabled=true
core_halt_freeze_halted_state_enabled=true
core_halt_exploration_prob/core_halt_exploration_min_steps
QTRMCoreCarry continuation API
```

Verification:

```text
halt-head conservative init test: OK
TRM no-grad inner-cycle test: OK
outer torch.no_grad preservation test: OK
per-sequence halt freeze test: OK
detached carry continuation tests: OK
model forward carry reuse test: OK
halt exploration train-only delay test: OK
```

Decision doc:

```text
docs/wiki/decisions/trm-core-act-runtime-scaffold.md
```

## 2026-05-07 - TRM Halt Q-Value Loss

Implemented optional q-value supervision for the core halt head:

```text
core_halt_loss_mode=q_value
core_halt_q_value_gamma
```

Meaning:

```text
q_halt     -> value of stopping at the current recurrent depth
q_continue -> discounted value of continuing one more depth step
runtime    -> halt when q_halt > q_continue
```

This closes the first Q-style halt/continue training gap. It is still only a
training mechanism until a held-out depth/ACT gate proves raw-reasoning gain.

## 2026-05-07 - Core Carry Eval Harness

Added raw-intelligence eval mode:

```text
qtrm_core_halt_carry_steps_N_no_evidence
```

This mode keeps retrieval/MemoryOS off, enables core halt, and reuses
`core_carry` across token/prefix forwards in causal forced-choice and
generation scoring. It is the first runtime harness for testing whether QTRM's
latent recurrent state continuation helps raw reasoning beyond fixed-depth
recomputation.

Status:

```text
harness wired: yes
promotion: pending held-out carry > no-carry/core-off/donor gate
```

## 2026-05-07 - Core Carry ACT Smoke4 Result

Ran the accepted S080 checkpoint with the correct eval-gate config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
```

Result:

```text
donor_only:                    0/4
core_off:                      0/4
core_steps8:                   4/4
answer_halt_gate_off:          0/4
core_halt_carry_steps8:        2/4
core_steps4:                   4/4
core_halt_carry_steps4:        4/4
```

Decision:

```text
carry harness is valid, but depth8 carry is rejected for this fixed-depth-4
smoke because it drifts toward an intermediate answer. Depth4 carry preserves
the accepted raw-recursive behavior.
```

Decision doc:

```text
docs/wiki/decisions/core-carry-act-smoke4.md
```

## 2026-05-07 - Core Carry Mixed-Depth ACT Gate

Ran mixed-depth causal forced-choice gate on:

```text
data/eval/pure_recursive_reasoning_heldout_72.jsonl
```

Smoke8 result:

```text
donor_only:                 2/8
core_off:                   0/8
core_steps2:                3/8
core_steps4:                3/8
core_steps8:                2/8
core_halt_carry_steps2:     3/8
core_halt_carry_steps4:     3/8
core_halt_carry_steps8:     2/8
answer_halt_gate_off:       2/8
```

Decision:

```text
reject S080 as a promoted mixed-depth ACT checkpoint. The carry harness is
useful diagnostically, but no carry mode beats the best fixed-depth modes at
smoke8.
```

Runner:

```text
scripts/257_run_core_carry_mixed_depth_act_gate.sh
```

Decision doc:

```text
docs/wiki/decisions/core-carry-mixed-depth-act-gate.md
```

## 2026-05-07 - Mixed-Depth ACT S160 Training

Fixed the depth-supervised training path for `pure_recursive_reasoning_*` rows:

```text
answer_aliases[0] is accepted as canonical answer
terminal depth can be inferred from depth_targets matching the final answer
staged choice-margin excludes the current staged answer from rejects
eval telemetry records answer-state-loop halt logits
```

Results:

```text
answer_loop_only_s160:
  core_steps8:          3/8
  halt_gate_off steps8: 3/8
  decision: reject as ACT

core_joint_ce_s160:
  core_steps8:          4/8
  halt_gate_off steps8: 4/8
  decision: partial fixed-depth raw-core improvement, reject as ACT
```

Decision doc:

```text
docs/wiki/decisions/mixed-depth-act-s160-results.md
```

## 2026-05-07 - List Transform Process S096 Reject

Added a list-transform failure summarizer and ran a list-focused process-state
experiment.

Failure ledger on the core-joint CE S160 checkpoint:

```text
list_transform qtrm_core_steps8: 0/2
errors:
  filtered_state_selected: 1
  reversed_final_selected: 1
```

Process-state run:

```text
init: core_joint_ce_s160
steps: 96
family_repeat: list_transform=6
staged_internal_sequence_ce_weight: 0.45
policy: core_and_answer_state_loop
```

Result:

```text
core_steps8:       3/8
list_transform:    0/2
decision: reject
```

The rejected checkpoint was deleted; eval JSON and ledgers were preserved.

Decision docs:

```text
docs/wiki/decisions/list-transform-failure-ledger-smoke8.md
docs/wiki/decisions/list-transform-process-s096-reject.md
```
## [2026-05-07] experiment | Reverse mixed composition prompt-context ledger

Added
`docs/wiki/decisions/transition-joint-reverse-composition-prompt-context.md`.

The accepted list-to-arithmetic checkpoint fails the reverse
arithmetic-to-list order by emitting the old fixed action code sequence.
Head-only S240 and prompt-context S240 both reject at `32/64`, while
prompt-context repeat4 S1000 improves reverse step accuracy to `0.6914` but
still has reverse exact `0/32`. The active gate remains exact trace transfer,
not partial step accuracy.
## 2026-05-07 General LLM Bottleneck Roadmap

- Added `docs/wiki/decisions/general-llm-bottleneck-roadmap.md`.
- Decision: reverse-composition accept would clear only the first major
  bottleneck: prompt-conditioned latent operation order.
- Remaining gates are now tracked as 10 bottlenecks: recursive depth/halting,
  value binding, latent-to-text rendering, donor override, trainable memory,
  reasoning-memory composition, metacognition, context routing, and
  agentic/multimodal grounding.

## 2026-05-07 Reverse Composition Token-Attention Reject

- Checkpoint:
  `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_promptctx_tokattn_aug_joint_only_s5500_from_len1113`.
- Gate:
  `data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_eval82000_v6to7_len1113_aug.jsonl`.
- Result: `26/64` overall.
- Family split: `mixed_arithmetic_list=26/32`,
  `mixed_list_arithmetic=0/32`.
- Decision: reject and delete checkpoint weights. Prompt token-attention alone
  made the reverse family better than the original baseline but catastrophically
  regressed the old-order policy. Next candidate should factorize operation
  order/phase rather than adding more prompt-attention capacity to the same
  joint code head.

## 2026-05-08 Reverse Composition Source-Router Rejects

Updated
`docs/wiki/decisions/transition-joint-reverse-composition-prompt-context.md`
with primitive/source-router experiments.

Results on the 128-case reverse mixed-composition holdout:

```text
joint on primitive checkpoint:       113/128
primitive source only:                60/128
mean source router S500:             113/128
token-attention source router S800:   88/128
mean+token router S800 aug lengths:   87/128
```

Decision: reject source-router promotion. Primitive operation factorization is
useful for reverse order, but routing between joint and primitive policies is
not robust; token attention learns a length/OOD shortcut and sends len13 old
order to primitive. Next candidate should fold operation factorization into
the canonical joint transition policy rather than using a separate source
classifier.

## 2026-05-08 Reverse Composition Operation-Residual Best Partial

Added an internal operation residual:

```text
primitive operation probabilities -> zero-init residual -> joint transition logits
```

Best result so far:

```text
checkpoint:
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_joint_opres_scale05_joint_s0400_from_opres/last.pt

eval config scale: 0.45
holdout: 128 cases, eval lengths 11/13
overall: 123/128
reverse arithmetic_to_list: 62/64
old list_to_arithmetic: 61/64
```

Decision: best partial, still reject. This is better than source routing
because the improvement stays inside the canonical joint transition path.
Remaining failures require phase/order-state pressure rather than path
selection.

Follow-up hard-range fine-tune at train start index `174000` rejected:

```text
overall: 102/128
reverse arithmetic_to_list: 58/64
old list_to_arithmetic: 44/64
```

The rejected checkpoint was deleted. Broader value-range exposure alone is not
the next lever.

Transition-state code residual also rejected:

```text
overall: 64/128
reverse arithmetic_to_list: 0/64
old list_to_arithmetic: 64/64
```

The checkpoint was deleted and the default experiment config now keeps
`transition_state_code_enabled=false` and
`transition_state_joint_code_residual_enabled=false`. A plain state-code
auxiliary path collapses back to the old canonical order.

Joint order-contrast S200 also rejected:

```text
overall: 119/128
reverse arithmetic_to_list: 62/64
old list_to_arithmetic: 57/64
```

The checkpoint was deleted. The best active partial remains the operation
residual checkpoint at eval scale `0.45`: `123/128`.
## 2026-05-09 Source-Pointer L2 Local Acceptance

Fixed two target/architecture blockers in the QTRM source-position pointer
gate:

- `role_value_list_class_mode` is now applied to every loaded row, so
  `source_position` is not silently overridden by row metadata.
- `list_transform` single-number states such as `"8"` are interpreted as
  one-element list states, not scalar arithmetic states.
- The accepted gate uses 12 role/value state roles, giving five raw source
  slots, five doubled slots, and two scalar slots for length-5 list data.

Accepted local gate:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_l2_source_pointer_roles12_targetfix_s120
checkpoint: accepted_l2_source_pointer_roles12_step_000040.pt
full trace exact: 23/128 = 0.1796875
full value accuracy: 0.5093123209169055
primitive-off value accuracy: 0.0
source-binder-off value accuracy: 0.03939828080229226
decision: accepted_l2
```

Scope: L2 local state/probe acceptance only. This is not L3/L4 and does not
yet prove canonical LM answer rendering or token-numeric causal dependence.

## 2026-05-09 Source-Pointer Primitive-Core L3 Acceptance

Added an L3 hard perturbation gate and found a real state-codec bug: list
targets did not supervise padded/null roles, so cardinality was weakly
specified. Hard rows now use `role_value_supervise_null_slots=true`, which
labels padded source-pointer list roles as class `0`.

Accepted primitive-core L3:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_l3_source_pointer_roles12_null_tune_s200
checkpoint: train/accepted_l3_primitive_core_null_step_000050.pt
report: manual_l3_null_decision_step_000050.json
audit report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l3_source_pointer_roles12_null_step50_l3_audit/report.json
```

Key metrics:

```text
full trace exact: 0.2890625
full value accuracy: 0.63125
primitive-off value accuracy: 0.0
primitive value drop: 0.63125
variant value accuracy:
  duplicate_even_binding: 1.0
  fifth_position_single_even: 0.4
  range_shift_v32to63: 0.58125
  surface_paraphrase: 0.54375
```

Important caveat: strict source-binder causality is still rejected
(`source_binder_value_drop = 0.115625`). Treat the source binder as auxiliary;
the canonical L3 claim is the primitive recurrent core plus null-slot state
codec.

## 2026-05-09 L4 Canonical LM Path Candidate Rejected

Added the first post-L3 LM-path runner:

```text
config: configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml
runner: scripts/322_run_source_pointer_l4_lm_path_gate.py
report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_source_pointer_lm_path_s080/report.json
```

Result:

```text
decision: rejected_l4_candidate
full greedy generation: 3/32 = 0.09375
donor-only: 5/32 = 0.15625
core-off: 5/32 = 0.15625
primitive-off: 3/32 = 0.09375
bridge-off: 4/32 = 0.125
```

Follow-up diagnostics:

```text
causal forced-choice:
  full = donor = core-off = primitive-off = bridge-off = 10/32

donor scale sweep:
  full default: 3/32
  donor_scale_0.25: 0/32
  qtrm_scale_1.0 donor_scale_0.25: 0/32
  qtrm_only: 0/32
```

Conclusion: L4 is still open. The accepted L3 primitive state exists, but the
current answer bridge/LM adapter does not causally render that state into token
logits. Next work must focus on a better state-to-token renderer, not MemoryOS,
RAG, MSA, or larger-scale training.

## 2026-05-09 L4 Bridge Family Rejected

Ran three post-L4 follow-ups against the accepted L3 source-pointer checkpoint:

```text
primitive-only bridge S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_primitive_lm_path_s020/report.json
  full: 3/16
  donor/core-off: 2/16
  primitive-off/bridge-off/final-binder-off: 3/16
  decision: rejected_l4_candidate

forced bridge S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_forced_bridge_s020/report.json
  full: 3/12
  donor/core-off: 2/12
  primitive-off/bridge-off/final-binder-off: 3/12
  decision: rejected_l4_candidate

forced bridge + adapter-only bottleneck S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_bridge_adapter_only_s020/report.json
  full: 2/12
  donor/core-off: 2/12
  primitive-off/bridge-off/final-binder-off: 2/12
  decision: rejected_l4_candidate
```

Do not count the 5-step adapter-only smoke as L4 acceptance; it used relaxed
negative thresholds and all causal margins were zero.

Code changes:

```text
configs/qwen35_2b_4090_source_pointer_l4_forced_bridge_roles12_s080.yaml
scripts/322_run_source_pointer_l4_lm_path_gate.py
src/qtrm_mm/training/train.py
tests/test_training_checkpoint_init.py
```

Validation:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_training_checkpoint_init tests.test_raw_intelligence_eval_script

116 tests OK
```

Conclusion: the current bridge-token cross-attention renderer family is the
wrong L4 path. Some runs perturb generation above donor/core-off, but the
primitive/bridge/final-binder ablations do not drop, so the accepted L3 state is
not the causal source. Next candidate should be a state-to-vocab pointer/copy
residual that feeds the canonical LM logits and has its own renderer-off
ablation.

## 2026-05-09 L4 State-to-Vocab Renderer Diagnostics

Implemented and tested a direct `core_role_value_state_vocab_renderer` path:

```text
canonical input tokens
-> donor hidden states / QTRM core
-> role-value answer bridge tokens
-> direct state-to-vocab residual logits
-> donor-fused LM logits
-> autoregressive generation
```

Code changes:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/322_run_source_pointer_l4_lm_path_gate.py
tests/test_core_halting.py
tests/test_pure_recursive_depth_supervised_train_script.py
tests/test_raw_intelligence_eval_script.py
```

Key fixes:

```text
1. Added direct renderer-off eval mode.
2. Added renderer-only trainable policy.
3. Added direct renderer CE / greedy-margin loss.
4. Removed unnecessary core-depth vocab logits from final-path-only training to avoid OOM.
5. Added primitive-off renderer contrast loss.
6. Added optional primitive-operation tokens to renderer cross-attention memory.
7. Added generation completion-delta reporting to the L4 runner.
```

Representative reports:

```text
S005/S010 direct renderer:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_s005/report.json
    qtrm_l4_source_pointer_vocab_renderer_lr1e3_s010/report.json
  result: rejected; full == donor == core-off == renderer-off

direct CE S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_s008/report.json
  result: rejected; full accuracy ties donor, but full generations differ
          from bridge-off/renderer-off on several cases.

direct CE + primitive contrast S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_primitive_contrast_s008/report.json
  result: rejected; primitive-off still does not drop in accuracy.

focused S024:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_primitive_contrast_s024_focus/report.json
  result: rejected; full accuracy drops to 0/4, indicating prompt/source-copy collapse.

operation-context S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_vocab_renderer_opctx_direct_ce_s008/report.json
  result: rejected; operation tokens did not change the held-out generation pattern.
```

Current root cause:

```text
The renderer is now causal enough to alter generated strings, but it is not
yet a correct reasoning renderer. It tends to copy source numbers or partial
intermediate values instead of rendering the final transformed answer.
```

Important warning:

```text
The L3 checkpoint chain still prints:
  [init] skipped shape-mismatched keys: core_role_value_state_embed.weight

The older base checkpoint used a 10-role embed while the current source-pointer
configs use 12 roles. L4 runs train/save a fresh 12-role embed, but the accepted
L3 baseline should not be treated as a fully clean full-state checkpoint.
```

Conclusion: L4 is still open. The next architecture should not merely make the
renderer stronger; it must make the final LM logits depend on the primitive
recurrent state in a way that renders transformed values, not source-copy
tokens.

## 2026-05-10 Source-Copy Renderer Role-Mask Repair

Found and fixed a contract mismatch between the source-copy alignment probe and
the actual LM renderer. The probe scored answer roles `0..3`, but the renderer
was preserving the later doubled-role block and masking the answer roles to
NULL. This made an alignment success look stronger than what generation really
used.

Changes:

```text
src/qtrm_mm/qtrm_model.py
tests/test_source_pointer_l4_lm_path_gate.py
```

The corrected test now requires the renderer to preserve answer roles `0..3`
and null out non-answer roles. Targeted tests passed.

Corrected source-copy alignment:

```text
S002:
  row_content_exact 83/128, rejected

S002 source-binder-off:
  row_content_exact 0/128, causal drop confirmed

S040 final-state CE:
  row_content_exact 128/128, accepted as L2 source-copy alignment diagnostic

S020 staged CE:
  row_content_exact 84/128, rejected
```

Generation still rejects L4:

```text
S040 after mask fix, smoke8:
  donor/core_off/vocab_renderer_off/source_binder_off: 5/8
  primitive_off: 4/8
  full: 3/8

S003 renderer-only continuation from S040:
  same smoke8 result, full 3/8
  failed checkpoint deleted; eval JSONL preserved
```

Gold-token rank probe shows the correct tokens are not hopelessly far away:

```text
S040 maskfix smoke8:
  donor/core_off/vocab_renderer_off all<=10: 8/8
  full all<=10: 8/8
  full all@1: 2/8, donor all@1: 3/8
```

Conclusion: exact source-position state is achievable, but the answer-role
decoder/query still fails to render that state into autoregressive LM tokens.
The next candidate should be a pointer-generator style decoder/cursor pressure
on the canonical LM path, not more MemoryOS/RAG/MSA work.

## 2026-05-10 Source-Copy Strict Scoring Repair

Found an evaluation weakness in `scripts/192_eval_raw_intelligence.py`: generation
hits were based on `normalized_contains`. For source-copy lexicalization this
can mark an overlong answer as correct, for example a completion that contains
the gold comma list as a substring but includes an extra copied source value.

Fix:

```text
source_copy_lexicalization now requires:
  exact_match or normalized_exact

general QA keeps the previous contains-based scoring.
```

Strict-rescored S040 maskfix smoke8:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_maskfix_smoke8/eval.strict.jsonl

donor/core_off/vocab_renderer_off/source_binder_off: 5/8
primitive_off: 4/8
full: 2/8
```

This makes the L4 rejection stronger. The previous full 3/8 included one loose
substring hit.

## 2026-05-10 Source-Copy Cursor Bias Rejected

Implemented an optional visible-prefix cursor bias for the source-copy renderer:

```text
config fields:
  core_role_value_state_vocab_renderer_source_copy_cursor_enabled
  core_role_value_state_vocab_renderer_source_copy_cursor_bias

code:
  QTRMMultimodalModel._compute_source_copy_cursor_role_bias
```

The idea was prior-backed by pointer-generator/copy-attention cursor or coverage
bias: use the visible generated prefix only to decide which answer role should
be read next, while still relying on the learned source-position state for the
actual copied value.

Smoke result on S040:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.jsonl

donor/core_off/vocab_renderer_off/source_binder_off: 5/8
primitive_off: 2/8
full: 1/8
```

Strict rescoring keeps the same full result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.strict.jsonl

full: 1/8
donor/core_off/vocab_renderer_off/source_binder_off: 5/8
```

Decision: reject as canonical default. The cursor over-biases copy tokens and
worsens autoregressive generation. The optional code path remains for controlled
ablation, but `configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml`
keeps it disabled.

## 2026-05-10 Source-Copy Span Lexicalization Repair

Root cause narrowed further: source-copy was keeping only the first tokenizer
piece for each source slot. Under Qwen tokenization, values such as `44`,
`40`, and `32` are multi-token spans:

```text
44,40,32 -> ["4", "4", ",", "4", "0", ",", "3", "2"]
```

Implemented the canonical-path span repair:

```text
src/qtrm_mm/algorithmic_value_state.py
  token_numeric_source_slot_token_spans(...)
  preserves all tokenizer pieces per compact source slot.

src/qtrm_mm/qtrm_model.py
  token_numeric_source_slot_token_span_ids / mask forward inputs
  _compute_source_copy_span_next_token_ids(...)
  source-copy renderer can continue the next token piece of the selected
  source span instead of copying only the first piece.

configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
  core_role_value_state_vocab_renderer_source_copy_span_enabled: true
```

This is still not a hidden answer channel: source positions must come from the
QTRM source-position state, and the result still enters LM logits through the
autoregressive renderer.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

108 tests OK
```

Partial smoke on S040 with span-copy enabled:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_spancopy_smoke8/eval.jsonl

completed before manual stop:
  donor_only: 5/8
  core_off: 5/8
  full QTRM: 3/8
  vocab_renderer_off: 3/3 partial
```

Decision: do not promote. Span-copy fixed one lexicalization failure
(`full 2/8 -> 3/8` versus strict S040), but full QTRM still loses to
donor/core-off. The remaining bottleneck is answer-role/order selection during
generation: the model can emit complete source spans more faithfully, but it
still chooses extra or wrong source slots in several cases.

## 2026-05-10 Source-Copy Answer-Role Cursor Candidate

Implemented a second, narrower pointer-generator style cursor after the
span-copy repair. The rejected visible-prefix cursor counted source token ids
and failed on multi-token/duplicate source values. The new cursor counts
completed source spans beyond the prompt source list and uses separators only
to decide which answer role should be read next.

Code:

```text
src/qtrm_mm/config.py
  core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled
  core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias
  core_role_value_state_vocab_renderer_source_copy_answer_role_separator_token_ids

src/qtrm_mm/qtrm_model.py
  _compute_source_copy_answer_role_cursor_bias(...)
```

Canonical boundary:

```text
Allowed:
  prompt token spans + visible generated prefix choose the active answer role.

Still required:
  QTRM source-position state chooses which source slot that role copies.

Not allowed:
  computing the final answer as a hidden side channel.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

111 tests OK
```

S040 smoke with role cursor enabled:

```text
full QTRM:
  first4: 4/4
  tail4 was split because full 8-case process exited early after 4 rows:
    eval_tail4 first2: 2/2
    eval_tail2 remaining2: 2/2
  combined smoke8: 8/8

previous span-only S040:
  full: 3/8
  donor/core_off: 5/8

tail4 ablations on the harder second source group:
  renderer_off: 1/4
  source_binder_off: 1/4
  primitive_off tail2: 0/2
```

Artifacts:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_rolecursor_smoke8/
```

Decision: promising L4 candidate, not promotion yet. It passes the local
smoke pattern and shows causal drops on the harder tail subset, but the full
held-out gate must still run in one reproducible command over a broader split
with donor/core_off/renderer_off/source_binder_off/primitive_off on the same
cases.

## 2026-05-10 Source-Copy Rolecursor L4 Smoke8 Accepted Candidate

The rolecursor L4 gate was made reproducible with a chunked/resumable runner:

```text
scripts/329_run_source_copy_rolecursor_l4_eval.py
```

Reason for the runner: repeated full generation evals can terminate or compete
for GPU with stale sessions. The runner executes each mode/chunk as an isolated
process, reuses complete chunks with `--resume`, and writes the full command log
to `report.json` while keeping stdout compact.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_copy_rolecursor_l4_eval \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

115 tests OK
```

Accepted smoke8 result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_rolecursor_l4_eval_smoke8/report.json

full QTRM:        8/8 = 1.000
donor-only:       5/8 = 0.625
core-off:         5/8 = 0.625
primitive-off:    3/8 = 0.375
source-slot-off:  5/8 = 0.625
source-binder-off:5/8 = 0.625
vocab-renderer-off:5/8 = 0.625
```

Interpretation:

```text
full - donor/core_off/source_binder_off/renderer_off = +0.375
full - primitive_off = +0.625
```

This is the first clean L4 source-copy LM-path smoke acceptance for the
rolecursor path: the recursive/core path, source binding, and vocab renderer
are all causally needed on the same eight cases, and the final answer is
emitted through autoregressive text generation.

Boundary:

```text
Accepted: smoke8 L4 candidate.
Not accepted yet: broad 16/32/128 case L4 promotion or general LM reasoning.
```

Next action: run the same resumable gate on 16/32/128 cases, then add
mixed-family cases where the answer cannot be solved by source-copy alone.

## 2026-05-10 Source-Copy Rolecursor L4 Smoke16 Accepted Candidate

The same resumable L4 gate was expanded from 8 to 16 held-out source-copy
lexicalization cases.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 16 \
  --chunk-size 2 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_smoke16 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke16/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:         15/16 = 0.9375
donor-only:         8/16 = 0.5000
core-off:           8/16 = 0.5000
primitive-off:      6/16 = 0.3750
source-slot-off:    8/16 = 0.5000
source-binder-off:  8/16 = 0.5000
vocab-renderer-off: 8/16 = 0.5000
```

Interpretation:

```text
full - donor/core_off/source_slot/source_binder/renderer_off = +0.4375
full - primitive_off = +0.5625
```

This strengthens the smoke8 result. The rolecursor source-copy path is now a
repeatable 16-case L4 candidate where the canonical LM output depends on the
recursive primitive state, source binding, source slots, and vocab renderer.

Boundary remains unchanged:

```text
Accepted: source-copy lexicalization L4 candidate at 16-case smoke scale.
Not accepted yet: 32/128 broad gate, mixed-family general reasoning, or
general-purpose LLM promotion.
```

## 2026-05-10 Source-Copy Rolecursor L4 Smoke32 Accepted Candidate

The same gate was expanded to 32 cases.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 32 \
  --chunk-size 2 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_smoke32 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke32/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:         27/32 = 0.84375
donor-only:        14/32 = 0.43750
core-off:          14/32 = 0.43750
primitive-off:     12/32 = 0.37500
source-slot-off:   14/32 = 0.43750
source-binder-off: 14/32 = 0.43750
vocab-renderer-off:14/32 = 0.43750
```

Interpretation:

```text
full - donor/core_off/source_slot/source_binder/renderer_off = +0.40625
full - primitive_off = +0.46875
```

The 32-case run confirms the source-copy rolecursor path is not just an 8/16
case artifact. Accuracy drops as cases broaden, but the causal margin remains
large and all required ablations collapse to donor/core-off scale.

Updated boundary:

```text
Accepted: reproducible 32-case source-copy L4 candidate.
Still not accepted: 128-case standard source-copy gate or mixed-family
general reasoning.
```

## 2026-05-10 Source-Copy Rolecursor L4 Standard128 Accepted

The full 128-case source-copy lexicalization gate passed.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 128 \
  --chunk-size 4 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_standard128 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_standard128/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:          100/128 = 0.78125
donor-only:          44/128 = 0.34375
core-off:            44/128 = 0.34375
primitive-off:       37/128 = 0.28906
source-slot-off:     41/128 = 0.32031
source-binder-off:   41/128 = 0.32031
vocab-renderer-off:  44/128 = 0.34375
```

Interpretation:

```text
full - donor/core_off/renderer_off = +0.43750
full - primitive_off = +0.49219
full - source_slot/source_binder_off = +0.46094
```

This completes the source-copy lexicalization L4 standard gate. The canonical
LM path emits the answer autoregressively, and the gain disappears when the
recursive primitive state, source slots, source binder, or vocab renderer are
ablated.

Boundary:

```text
Accepted: source-copy lexicalization L4 standard128.
Still not accepted: broad/general LM promotion.
Next required gate: mixed-family/non-copy reasoning where the answer cannot be
obtained by lexical source-copy.
```

## 2026-05-10 Source-Copy Pointer L2 Accepted

A contract mismatch was found in the source-copy trainer: the probe/renderer
reads final answer roles `4..7`, but source-copy CE was still aimed at the
early/raw roles. The trainer now has
`--core-role-value-source-copy-answer-role-targets` to supervise the same answer
role block that inference reads.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_source_pointer_batch_trainer \
  tests.test_source_position_logits_probe \
  tests.test_source_pointer_l4_lm_path_gate

54 tests OK
```

Accepted source-copy probe:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s020_b2_answerroles_from_s060/train/last.pt

eval128:
  full copy answer accuracy: 1.000
  source-slot-off copy answer accuracy: 0.000
  source-binder-off copy answer accuracy: 0.000
```

Decision: accept L2 source-position/copy-logits prerequisite only. L4 remains
rejected because a post-L2 generation smoke still degraded full autoregressive
generation:

```text
partial smoke:
  donor_only: 5/8
  core_off: 5/8
  full QTRM: 2/8
```

Next bottleneck: repair vocab renderer / donor fusion. The pointer state is now
causal and correct; generation is the failing layer.

## 2026-05-10 Mixed Non-Copy LM Gate Rejected

Added a reproducible post-source-copy diagnostic:

```text
scripts/330_run_mixed_noncopy_lm_gate.py
tests/test_mixed_noncopy_lm_gate.py
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest tests.test_mixed_noncopy_lm_gate

5 tests OK
```

Run:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
  --max-cases 16 \
  --chunk-size 4 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_mixed_noncopy_lm_gate_diag16 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_mixed_noncopy_lm_gate_diag16/report.json
```

Result:

```text
decision: rejected_noncopy_lm_gate

full QTRM:  0/16 = 0.0
donor-only: 0/16 = 0.0
core-off:   0/16 = 0.0
```

Interpretation:

```text
The source-copy L4 standard128 result is real but narrow. It does not solve
computed mixed-family answers. The next bottleneck is non-copy answer synthesis
from latent state into the canonical autoregressive LM path.
```

Forced-choice follow-up:

```text
cases: 4 mixed-family non-copy rows
modes: donor_only, core_off, full QTRM
result: 0/12 hits
tail class: 12/12 doubled_list
```

This rules out a pure greedy-rendering explanation. Even with candidates
provided, the model prefers the intermediate doubled list over the final scalar
answer. The next gate must target scalar reduction/accumulator/final-answer
state in the recurrent path before another broad LM renderer run.

## 2026-05-10 Ouro Recurrent L2 Len11/13 Recheck Rejected

The preserved L2 recurrent-answer checkpoint was rechecked on the harder
mixed-family non-copy split:

```text
checkpoint:
  local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0_s40_eval4/accepted.pt

eval:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_l2_len1113_forced_choice8.jsonl

tail summary:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_l2_len1113_tail_error_summary8.json
```

Result:

```text
donor_only:        0/8
core_off:          0/8
recurrent_off:     0/8
full recurrent:    0/8

full tail classes:
  doubled_list:      6/8
  pre_subtract_sum:  2/8
```

Interpretation:

```text
The earlier len7 recurrent-answer success was a narrow local L2 result. It
does not scale to len11/13 non-copy reductions. The next architecture gate
must train and test length-stable scalar reduction plus final subtract
retention, not another source-copy or private renderer patch.
```

## 2026-05-10 Ouro Recurrent Len11/13 From Joint Controller Rejected

Ran the orthodox retry: start from the accepted len11/13 transition-joint
controller and train only the Ouro/LoopLM recurrent answer path.

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_len1113_s020_eval4_from_jointonly

train:
  steps=20
  final_logit_ce=1.0
  depth_final_ce=1.0
  validation gate: causal forced-choice, eval4
```

Result:

```text
decision: rejected

donor_only:        0/4
core_off:          0/4
recurrent_off:     0/4
full recurrent:    0/4

action/finality sanity:
  trace exact:     32/32
  finality exact:  32/32
  halted exact:    32/32

tail summary:
  full recurrent:  doubled_list=4/4
```

Interpretation:

```text
The transition/action controller is not the blocker. More final-token CE on
the same answer loop does not make the model prefer the final scalar. The next
architecture must add a causal value accumulator / process-supervised scalar
state update that the recurrent answer path actually consumes.
```

## 2026-05-10 Relative Source-Slot LM Path Smoke

Implemented and smoke-tested relative source-slot ids through the canonical
QTRM LM path.

Commits:

```text
b39f72f feat(qtrm): add relative parity source slot mode
afc8694 feat(qtrm): propagate source slot id mode through gates
48ea7b2 fix(qtrm): preserve source copy spans for relative slots
```

Run:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_relative_parity_smoke_s001

flags:
  --token-numeric-source-slot-id-mode relative_parity
  --token-numeric-source-slot-vocab-size 3
  --steps 1
  --max-eval-cases 4
```

Result:

```text
decision: rejected_l4_candidate
full_generation_accuracy:                    0.25
donor_generation_accuracy:                   0.25
core_off_generation_accuracy:                0.25
source_slot_off_generation_accuracy:         0.25
source_binder_off_generation_accuracy:       0.25

full_minus_donor:                            0.0
full_minus_core_off:                         0.0
full_minus_source_slot_off:                  0.0
full_minus_source_binder_off:                0.0
```

Observed completion perturbation:

```text
surface_paraphrase:
  donor/core_off: 2,
  full:           56,
```

Interpretation:

```text
The relative source-slot path is executable and can perturb LM output, but it
is not yet a causal reasoning improvement. The same 1/4 hit rate appears under
donor/core-off/source-slot-off/source-binder-off ablations. Treat this as an
input-path repair plus L4 smoke evidence, not promotion. The next useful gate
must force scalar reduction/accumulator state to improve exact answers and drop
under primitive/source-slot/source-binder/answer-bridge ablations.
```

## 2026-05-10 List-Transform Eval Strictness Fix

The follow-up 20-step relative source-slot smoke rejected:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_relative_parity_smoke_s020

reported before strict fix:
  full_generation_accuracy:       1/8 = 0.125
  donor_generation_accuracy:      1/8 = 0.125
  core_off_generation_accuracy:   1/8 = 0.125
  full_minus_donor:               0.0
  full_minus_core_off:            0.0
```

Audit finding:

```text
The 1/8 hits were loose contains matches such as target "52" inside output
"52,54". That is acceptable for open QA but invalid for algorithmic CSV
answers. `list_transform` and `sequential_list_transform` now require strict
exact/normalized-exact generation scoring, matching the existing
source-copy strictness.
```

Offline strict rescore of the same JSONL:

```text
donor_only_no_evidence:                 0/8
qtrm_core_off_no_evidence:              0/8
qtrm_core_steps_8_no_evidence:          0/8
all listed L4 ablations:                0/8
```

Interpretation:

```text
The current relative source-slot LM path can alter text, but it has not yet
solved exact non-copy answer synthesis at all. The next architecture step must
target a process-supervised scalar/list accumulator that feeds the normal LM
logits path and is ablated causally.
```

## 2026-05-10 Relative Source-Slot Trainable Policy Fix

Audit finding:

```text
The relative source-slot L4 smoke changed the source-slot id space from
absolute value classes to a compact parity vocabulary. That causes
`token_numeric_source_slot_embed.weight` and some source-position binder
parameters to initialize fresh when loading the source-copy checkpoint.

The previous default L4 trainable policy only trained answer bridge / answer
loop / vocab renderer parameters. It could leave those fresh source-binding
parameters frozen, making the relative-mode smoke underpowered.
```

Implementation:

```text
Added trainable policy:
  source_slot_binder_answer_bridge_loop_vocab_renderer

It trains:
  token_numeric_source_slot_*
  core_source_position_binder_*
  answer_state_loop_*
  core_role_value_state_embed.*
  core_role_value_state_answer_*
  core_role_value_state_vocab_renderer_*
```

Interpretation:

```text
This is still the canonical prompt-token -> latent binder/core -> LM logits
path. It is not a solver. It makes the next relative source-slot experiment a
fairer test because newly initialized prompt-derived binding modules are no
longer frozen.
```

## 2026-05-10 Direct L4 Sufficient-Condition Gate

Question:

```text
Can we skip more loose L2/L3-style diagnostics and go directly to a sufficient
L4 condition for a general non-copy LM path?
```

Answer:

```text
Yes, but only by making the gate stricter, not easier.
```

Implementation:

```text
scripts/330_run_mixed_noncopy_lm_gate.py now scores strict exact generation
only. Loose contains-style `hit` fields are ignored for this gate.

Required modes:
  donor_only_no_evidence
  qtrm_core_off_no_evidence
  qtrm_core_steps_8_no_evidence
  qtrm_core_steps_8_primitive_role_value_off_no_evidence
  qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence
  qtrm_core_steps_8_core_source_position_binder_off_no_evidence
  qtrm_core_steps_8_role_value_answer_bridge_off_no_evidence
  qtrm_core_steps_8_typed_value_answer_bridge_off_no_evidence
  qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence
  qtrm_core_steps_8_answer_state_recurrent_off_no_evidence
```

Sufficient condition:

```text
full strict generation accuracy >= threshold
full > donor-only by margin
full > core-off by margin
full drops under primitive-off
full drops under source-slot-off
full drops under source-binder-off
full drops under answer-bridge-off
full drops under typed-value-answer-bridge-off
full drops under vocab-renderer-off
full drops under answer-recurrent-off
all eval commands must complete
all required modes must be present
```

Checkpoint audit:

```text
The source-copy default checkpoint is not currently runnable as a sufficient
gate input because its trainable-delta base chain points to missing file:

  local_eval/research_gate_runner/
  primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt

This is a checkpoint hygiene problem, not a sufficient-gate design problem.
```

Runnable smoke:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_answer_bridge_final_choice_s020_from_s040/last.pt

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_answer_bridge_s040.yaml

run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  l4_sufficient_noncopy_gate_typed_value_s020_smoke_1case

decision:
  rejected_noncopy_lm_gate

strict exact:
  donor_only:        0/1
  core_off:          0/1
  full:              0/1
  every ablation:    0/1
```

Interpretation:

```text
The sufficient-condition runner is now executable and conservative. The
current typed-value checkpoint still fails the actual L4 condition: full QTRM
does not produce the final scalar answer, and disabling core paths does not
cause a measurable accuracy drop because the full path is already at zero.

The next architecture change should target scalar/list accumulator state that
causally feeds the canonical LM logits path. A source-copy renderer repair or
loose contains metric cannot count as general L4 progress.
```
