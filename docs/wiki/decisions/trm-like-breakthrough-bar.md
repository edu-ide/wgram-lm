# TRM-Like Breakthrough Bar

Status: hard promotion rule, 2026-05-17.

## Position

QTRM is meaningful only if it produces a TRM-like innovation signal. Incremental
formatting gains, answer-only CE, retrieval improvements, or donor preservation
are not enough.

The project must show:

```text
same/smaller parameter budget
-> more latent recurrent compute at inference
-> clearly better reasoning accuracy
-> held-out scale-out beyond the tiny diagnostic slice
-> normal LM output path remains intact
```

## Non-Negotiable Gate

The canonical breakthrough gate is:

```bash
bash scripts/410_run_trm_breakthrough_gate.sh
```

It requires both sides:

```text
M6 scoped raw reasoning:
  accepted M6 manifest
  QTRM full exact >= 0.50
  QTRM margin over Qwen3.6 proxy >= 0.20
  recurrent core gain >= 0.25
  destructive ablation drop >= 0.25
  min family exact >= 0.30

M7B public-style scale-out:
  accepted M7B report
  >= 256 held-out cases
  full depth accuracy >= 0.18
  deeper recurrent core gain >= 0.03 over depth0
  deeper recurrent core gain >= 0.03 over best shallow depth
```

If either side fails, do not call the result TRM-like.

## Current Status

Current M6 is strong:

```text
QTRM full_generation_exact: 0.6067708333333334
Qwen3.6 DGX proxy score: 0.146484375
core_gain: 0.5859375
ablation_drop: 0.5716145833333334
min_family_generation_exact: 0.4140625
```

Current M7B 256 is rejected:

```text
depth0 accuracy: 0.15234375
depth4 accuracy: 0.09375
gain_vs_baseline: -0.05859375
gain_vs_best_shallow: -0.05859375
```

Therefore the current state is:

```text
TRM-like scoped raw-reasoning candidate: yes
TRM-like public-style breakthrough: no
next bottleneck: scale raw-reasoning depth first, then return to public-style MCQ
```

DGX depth-repair triage, 2026-05-17:

```text
report:
  DGX local_eval/dgx_qtrm_native_m7b_depth256_triage_depthrepair_20260517_190439/m7b_gate_report.json

result:
  depth0: 0.09375
  depth1: 0.15625
  depth4: 0.09375
  depth8: 0.046875
  decision: rejected
```

Interpretation:

```text
Adding final-token mixed-depth CE, full-vs-shallow margin, and depth6->depth8
trajectory KL did not create a TRM-like public-MCQ depth signal. MMLU-style
MCQ is currently confounded by missing knowledge/language ability. The next
TRM-like test should return to raw reasoning scale-out: keep the same
modchain/revchain/checksum family but increase program_len from 4 to 8/12 and
require recurrent depth to scale with problem length.
```

Raw scale-out transfer, 2026-05-17:

```text
len4 accepted checkpoint:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/last.pt
  full_generation_exact: 0.6067708333333334
  full_minus_think0: 0.5859375
  full_minus_worst_ablation: 0.5716145833333334
  min_family_generation_exact: 0.4140625

len8 from scratch, 1200 steps:
  DGX local_eval/dgx_trm_raw_scaleout_len8_medium_len8_20260517_190940/report.json
  decision: rejected
  full_generation_exact: 0.041666666666666664
  full_minus_think0: 0.041666666666666664
  full_minus_worst_ablation: 0.010416666666666664
  min_family_generation_exact: 0.015625

len4 -> len8 transfer:
  DGX local_eval/dgx_trm_raw_scaleout_len8_resume_len4_to_len8_20260517_191305/report.json
  decision: accepted_trm_raw_scaleout_len8
  full_generation_exact: 0.15104166666666666
  think0_generation_exact: 0.0
  full_minus_think0: 0.15104166666666666
  full_minus_worst_ablation: 0.109375
  min_family_generation_exact: 0.03125

len8 -> len12 transfer:
  DGX local_eval/dgx_trm_raw_scaleout_len12_resume_len8_to_len12_20260517_191414/report.json
  decision: accepted_trm_raw_scaleout_len12
  full_generation_exact: 0.10416666666666667
  think0_generation_exact: 0.0
  full_minus_think0: 0.10416666666666667
  full_minus_worst_ablation: 0.06770833333333334
  min_family_generation_exact: 0.015625

len12 -> len16 transfer:
  DGX local_eval/dgx_trm_raw_scaleout_len16_resume_len12_to_len16_20260517_191926/report.json
  decision: accepted_trm_raw_scaleout_len16
  full_generation_exact: 0.109375
  think0_generation_exact: 0.0
  full_minus_think0: 0.109375
  full_minus_worst_ablation: 0.0859375
  min_family_generation_exact: 0.05813953488372093

len16 standalone checkpoint rerun, 512 held-out cases:
  DGX local_eval/dgx_trm_raw_scaleout_len16_standalone_len16_ckpt512_20260517_192654/report.json
  decision: accepted_trm_raw_scaleout_len16
  full_generation_exact: 0.09375
  think0_generation_exact: 0.0
  full_minus_think0: 0.09375
  full_minus_worst_ablation: 0.0703125
  min_family_generation_exact: 0.05263157894736842

strong family-DRO len16 repair:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_familydro_repair_20260517_193004/report.json
  decision: rejected
  full_generation_exact: 0.076171875
  full_minus_think0: 0.076171875
  full_minus_worst_ablation: 0.05078125
  min_family_generation_exact: 0.023391812865497075

low-pressure family-floor selection:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_familyfloor_select_20260517_193542/report.json
  decision: rejected
  full_generation_exact: 0.09375
  full_minus_think0: 0.09375
  full_minus_worst_ablation: 0.0703125
  min_family_generation_exact: 0.05263157894736842
  best periodic step: 0

len16 h-state probe:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_probe_h_20260517_194508/report.json
  decision: accepted_trm_raw_scaleout_len16
  greedy full_generation_exact: 0.09375
  core_answer_probe_exact: 0.115234375
  core_step_probe_by_depth:
    depth1: 0.048828125
    depth8: 0.046875
    depth10: 0.064453125
    depth12: 0.0625
    depth16: 0.08984375

state-trace depth stabilization:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_trace_depth_family_20260517_194749/report.json
  decision: rejected
  restored best checkpoint: step 0
  high-water full exact: 0.1875 at step600
  step600 min family exact: 0.03488372093023256

operation breakdown:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_opbreakdown_20260517_195401/report.json
  modchain by last op:
    op01 add1: 0 / 29 = 0.0
    op04 mul2: 0 / 27 = 0.0
    op07 affine3: 1 / 31 = 0.03225806451612903
    op06 affine2: 5 / 19 = 0.2631578947368421

late-op hard replay:
  DGX local_eval/dgx_trm_raw_scaleout_len16_len16_modchain_lateop_replay_20260517_195528/report.json
  decision: rejected
  restored best checkpoint: step 0
  high-water full exact: 0.19140625 at step700
  step700 min family exact: 0.023255813953488372
```

Interpretation:

```text
The first real TRM-like scaling signal is not the public MCQ path. It is the
curriculum transfer path: len4 -> len8 -> len12 -> len16. In the accepted
transfer runs, think0 cannot solve the task, full recurrent depth solves a
nonzero slice, and state/op ablations remove a large part of the gain.

This is still not a project-level breakthrough. The family floor is weak,
especially modchain/revchain, and the result needs seed stability, a stricter
family-balanced gate, and then a return to M7B/public benchmarks. Strong
family-DRO damaged the accepted trajectory, so the next repair should use
low-pressure family-floor periodic selection or family-specific replay instead
of a high family-DRO weight.

The h-state probe partially matches the Looped-LM/Ouro diagnostic expectation:
the final recurrent state exposes more answer information than greedy decoding
alone, and later depth16 state is more probe-readable than depth1. It does not
yet show clean monotonic answer sharpening across all depths, so the next
objective must stabilize the trajectory rather than only increasing answer CE.

The follow-up stabilization and hard-replay runs showed the failure mode:
overall accuracy can be raised during training, but the family floor drops.
Therefore the next architectural candidate should add route-conditioned
recurrent transitions or family/order-conditioned state updates, and acceptance
must require each route/family to keep a causal depth gain.
```

## Research Direction

Do not widen the model or add RAG before the breakthrough gate passes. The next
changes must target the recurrent trajectory itself:

```text
1. depth curriculum:
   Train on mixed depths and evaluate depth0/1/2/4/8.

2. trajectory consistency:
   Correct stable answers should become more stable with additional loops.

3. attractor-style stabilization:
   After the model approaches a solution, later loops should not destructively
   move the latent state.

4. early-exit/halt:
   More loops must be optional at inference. If the state converges early, stop.

5. family-balanced hard negatives:
   Repair only the public-style families where depth hurts.
```

## External Prior Details To Import

The raw scale-out result is not enough by itself. The architecture and gates
must import the key details that make recent recursive/latent-reasoning work
convincing:

```text
TRM / TinyRecursiveModels:
  source: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
  key detail:
    Recursive reasoning must progressively improve an answer/state, not merely
    add hidden compute. The proof should show answer refinement across steps.

Recurrent Depth LM:
  source: https://arxiv.org/abs/2502.05171
  key detail:
    The model should scale test-time compute by iterating a recurrent block in
    latent space, then show benchmark improvement as recurrent depth increases.

Ouro / Looped LM:
  source: https://www.aimodels.fyi/papers/arxiv/scaling-latent-reasoning-via-looped-language-models
  key detail:
    Linear probes or equivalent diagnostics should show that the correct answer
    becomes more accessible at later loop iterations even without new input.

Parallel Loop Transformer:
  source: https://arxiv.org/abs/2510.24824
  key detail:
    If sequential loops become too slow, the later system needs cross-loop
    parallelism or KV-sharing. This is not today's raw-intelligence blocker,
    but it matters before deployment-scale claims.

Coconut and latent-token critiques:
  sources:
    https://papers.cool/arxiv/2412.06769
    https://arxiv.org/abs/2512.21711
  key detail:
    Latent thoughts can be useful, but they can also become shortcut tokens.
    QTRM gates must therefore include causal/adversarial ablations, not only
    higher accuracy.
```

Immediate consequence:

```text
The next accepted run must not only pass len16/len20 exact accuracy. It should
also log a depth trajectory: answer exact by depth, family exact by depth,
state/op ablation by depth, and ideally probe-style evidence that the final
answer becomes more accessible in later recurrent states.
```

Raw scale-out runner:

```bash
bash scripts/411_dgx_trm_raw_scaleout_gate.sh plan
PROGRAM_LEN=8 THINK_STEPS=8 bash scripts/411_dgx_trm_raw_scaleout_gate.sh run
PROGRAM_LEN=12 THINK_STEPS=12 bash scripts/411_dgx_trm_raw_scaleout_gate.sh run
RESUME_FROM=<accepted len12 last.pt> PROGRAM_LEN=16 THINK_STEPS=16 bash scripts/411_dgx_trm_raw_scaleout_gate.sh run
RESUME_FROM=<accepted len16 last.pt> STEPS=0 PROGRAM_LEN=16 THINK_STEPS=16 EVAL_CASES=512 bash scripts/411_dgx_trm_raw_scaleout_gate.sh run
```

## Accepted Route-Conditioned Repair

The first accepted post-len16 architecture repair is
`single_order_router` with `chain_vs_checksum` router targets:

```text
source checkpoint:
  local_eval/dgx_trm_raw_scaleout_len16_resume_len12_to_len16_20260517_191926/last.pt

accepted report:
  DGX local_eval/dgx_single_order_router_chain_target_s200_20260517_204312/report.json

decision:
  accepted_single_order_router_chain_target_len16

metrics:
  full_generation_exact: 0.12890625
  full_minus_think0: 0.12890625
  full_minus_worst_ablation: 0.11328125
  min_family_generation_exact: 0.06976744186046512

family exact:
  checksum: 0.24705882352941178
  modchain: 0.06976744186046512
  revchain: 0.07058823529411765
```

Why this matters:

```text
The accepted single recurrent transition remains route0 and is preserved:
forced route0 full exact is 0.109375.  The additive route1 path is derived from
the same token stream and feeds the same recurrent/LM-logit answer path.  The
router learned to send ordered chain tasks to route1 while checksum remains
mostly route0.
```

This is still not a TRM-like breakthrough. It is a valid architecture foothold
because it improves the len16 family floor without changing the base tensors.
Promotion requires a standalone 512-case rerun, route-specialization stability,
and len20/longer transfer.

Standalone 512-case rerun:

```text
report:
  DGX local_eval/dgx_single_order_router_chain_target_512rerun_20260517_205432/report.json

decision:
  accepted_single_order_router_chain_target_len16_512rerun

metrics:
  full_generation_exact: 0.140625
  think0_generation_exact: 0.0
  full_minus_think0: 0.140625
  full_minus_worst_ablation: 0.119140625
  min_family_generation_exact: 0.06432748538011696
  state_reset_generation_exact: 0.021484375
  op_zero_generation_exact: 0.01953125

family exact:
  checksum: 0.27647058823529413
  modchain: 0.08187134502923976
  revchain: 0.06432748538011696

router last_hlh_prob:
  checksum: 0.17065902054309845
  modchain: 0.9107509851455688
  revchain: 0.9774150252342224
```

Decision consequence:

```text
The repair is now reproduced on the stronger standalone gate. This confirms
causal recurrent compute and route-conditioned family separation, but the
absolute score and forced route1 score are still too small for a breakthrough
claim. The next falsifiable gate is len20 transfer with the same destructive
ablations.
```

Len20 route-conditioned transfer:

```text
accepted report:
  DGX local_eval/dgx_single_order_router_len20_familyfloor_select_20260517_222156/report.json

decision:
  accepted_single_order_router_len20_familyfloor

metrics:
  full_generation_exact: 0.1953125
  think0_generation_exact: 0.0
  full_minus_think0: 0.1953125
  full_minus_worst_ablation: 0.158203125
  min_family_generation_exact: 0.07602339181286549
  state_reset_generation_exact: 0.03125
  op_zero_generation_exact: 0.037109375

family exact:
  checksum: 0.43529411764705883
  modchain: 0.07602339181286549
  revchain: 0.07602339181286549

best periodic eval:
  step: 600
  generation_exact: 0.1953125
  min_family_generation_exact: 0.07602339181286549
```

Why this matters:

```text
The previous len20 attempts failed mostly because the ordered-chain families
were under-selected, not because the route-conditioned core was unusable.
Balanced chain data, family-DRO pressure, and family-floor checkpoint selection
converted a near miss into an accepted len20 gate while preserving the normal
LM-logit answer path and destructive core ablations.
```

Promotion constraint:

```text
Do not call this a breakthrough yet. It is an accepted synthetic len20 raw
reasoning result on the selection seed. Promotion requires an independent
eval-seed rerun, then len24 or a public-style gate where the same
recurrent-core/depth ablation remains causal.
```

Independent seed check:

```text
report:
  DGX local_eval/dgx_single_order_router_len20_seed9338_20260517_230036/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.1640625
  full_minus_think0: 0.1640625
  full_minus_worst_ablation: 0.1328125
  min_family_generation_exact: 0.029239766081871343

family exact:
  checksum: 0.4117647058823529
  modchain: 0.029239766081871343
  revchain: 0.05263157894736842
```

Decision consequence:

```text
The len20 route-conditioned core remains a useful architecture foothold, but it
is not stable enough for promotion. The next work is data-scale and multi-seed
family-floor selection for modchain/revchain, not len24 transfer.
```

Data-scale seed repair:

```text
report:
  DGX local_eval/dgx_single_order_router_len20_seed9338_datascale_20260517_230608/report.json

decision:
  rejected

settings:
  train_cases: 32768
  steps: 1200
  eval_seed: 9338
  periodic_score_mode: family_floor

metrics:
  full_generation_exact: 0.166015625
  full_minus_think0: 0.166015625
  full_minus_worst_ablation: 0.134765625
  min_family_generation_exact: 0.04093567251461988
  state_reset_generation_exact: 0.03125
  op_zero_generation_exact: 0.03125

best periodic eval:
  step: 200
  generation_exact: 0.166015625
  min_family_generation_exact: 0.04093567251461988
```

Decision consequence:

```text
More synthetic cases and longer route-only continuation did not repair the
seed instability. The bottleneck is now classified as ordered-chain transition
generalization, not merely dataset quantity. Future long runs must use
checkpoint saving, and the next promoted candidate must change the transition
objective/state representation before repeating the multi-seed gate.
```

Operational guardrail:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports:
  --save-every-steps
  --save-best-periodic-checkpoint

Use these for long DGX runs so interruptions do not destroy all intermediate
training state.
```

Next candidate after data-scale rejection:

```text
mechanism:
  forced_route_prefix_depth_anchor_loss

target:
  ordered-chain transition generalization on route1

causal path:
  causal-prefix prompt tokens
  -> tokenizer
  -> QTRM single_order_router with forced route1
  -> recurrent core at matching prefix depth
  -> normal LM logits

why allowed:
  The objective changes training pressure only. It does not add a hidden answer
  channel, rule solver, or runtime sidecar.

promotion gate:
  len20 seed9338 family-floor pass
  len20 original seed retention pass
  destructive state/op/depth ablations still remove the gain
```

Implemented repair path:

```text
scripts/400_train_qtrm_native_public_mcq_final_token.py

new objective knobs:
  --multi-depth-ce-weight
  --multi-depth-ce-depths
  --depth-gain-weight
  --depth-gain-shallow-depths
  --trajectory-kl-weight
  --trajectory-kl-anchor-depth
  --trajectory-kl-compare-depths

DGX fastlane default:
  think_steps=8
  multi_depth_ce_depths=4,8
  depth_gain_shallow_depths=0,1,2,4
  trajectory_kl_anchor_depth=8
  trajectory_kl_compare_depths=6
```

## Kill / Pivot Rule

If repeated DGX runs cannot make M7B 256 pass while preserving M6 and language
non-regression, stop claiming the current QTRM-native architecture is a TRM-like
breakthrough. At that point the correct move is a root redesign of the recurrent
transition objective, not more final-answer tuning.
