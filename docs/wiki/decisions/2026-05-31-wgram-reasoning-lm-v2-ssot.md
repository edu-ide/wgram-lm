# 2026-05-31 W-GRAM Reasoning LM V2 SSOT

## Decision

Create `src/wgram_lm/v2/` as the clean canonical path for the next
W-GRAM-LM reasoning-language model.

```text
byte input
-> dynamic BLT causal chunk summary
-> GatedDeltaNet2-style 3:1 recurrent/attention core
-> IMTA same-body latent trajectories
-> own-latent prediction auxiliary
-> answer-start/body/stop contract losses
-> learned speaker position encoding
-> answer-prefix memory planner with optional causal prompt-context read
-> causal token maturation / latent refinement
-> hnet causal speaker
-> same LM head
-> free generation only
```

## Why V2 Exists

The previous active BLT path was powerful but overloaded. Too many historical
ideas lived near the answer path: old candidate selection, LeWM probes, hnet
byte speakers, and several meanings of "one body." V2 exists to remove that
ambiguity. It is not a second evaluator; it is the single path future RI claims
must pass through.

## Non-Negotiable Rules

- Boundary state source is `causal_chunk_summary`.
- Fastlane must not force fixed BLT boundaries. Fixed boundaries are allowed
  only for tiny smoke tests or causality unit tests.
- Token-level dechunking must use only completed chunk states or BOS context.
  A token must never read the chunk boundary after that token, because that
  leaks future response-prefix inputs during teacher-forced training.
- The 3:1 attention leg inside the recurrent core is causal. GatedDeltaNet-2
  recurrent state is not enough if the explicit attention layer can see future
  chunks.
- Promotion evaluation with official GatedDeltaNet-2 must use the full-prefix
  chunk kernel unless an explicit cache-correct token-by-token decode path is
  implemented and separately verified. The official layer auto-selects
  `fused_recurrent` for short eval sequences, but V2's current gate recomputes
  the whole prefix at every step; mixing those paths can make the generated
  first token disagree with the teacher-forced first-token diagnostic.
- Final answer path is `hnet_causal_speaker_same_lm_head`.
- IMTA/GRAM/PTRM means internal latent trajectories, not external answer
  candidates.
- Own-latent prediction predicts the next causal chunk latent, not the next
  already-selected answer hidden by residual copying.
- Repetition is measured in the free-generation gate. Older anti-repeat knobs
  such as repeat unlikelihood or repetition penalty are diagnostic-only and
  off by default in the primary fastlane; they must not be used to claim model
  reasoning quality.
- The first response token must beat answer-stop tokens through the same LM
  head during training. This is an answer-contract repair, not a decoding
  penalty, and it exists because the 2026-05-31 official-core fastlane runs
  showed `top1=<|box_end|>` before any answer token.
- Response body tokens must also beat answer-stop tokens through the same LM
  head. The route-floor continuation still produced short answers like
  `2<|box_end|>` for multi-token labels such as `24<|box_end|>`, so V2 uses a
  same-head continuation-stop margin on non-stop body positions as a scheduled
  diagnostic/repair lever. It is not a primary default until a free-generation
  gate shows it does not revive body-token loops.
- The speaker must receive positional information. A same-head decoder without
  position encoding cannot reliably distinguish "the first generated 1" from
  "the next generated 1", which matches the observed `1111...` collapse.
- The speaker must mature token hidden states before discrete commitment.
  Repetition penalty is an old decode-time patch; V2 instead delays commitment
  by refining the causal hidden state and still projects through the same LM
  head.
- The latent-to-speaker bridge uses an input-dependent selective gate. A single
  global byte/latent scalar is too blunt when donor surface language must be
  preserved while recurrent latent state is still learning.
- Multi-token answer semantics require a prompt-grounded internal answer plan.
  If the model can start and stop but still writes `20/200...` for labels such
  as `24`, the missing piece is not another stop penalty; it is a same-head
  latent answer-prefix memory that binds several upcoming answer tokens before
  token commitment.
- Answer-prefix memory has an optional causal same-body prompt-context read.
  A single compressed response-start vector is likely too narrow for semantic
  grounding, but the 2026-05-31 short gate showed that turning this read on
  immediately can create a `{`/low-diversity structural-token attractor. Keep
  it implemented as a staged-grounding diagnostic, not as the primary fastlane
  default until free generation proves it helps.
- Answer-prefix memory must be committed through the normal speaker state, not
  only softly injected as a residual suggestion. The commitment path uses the
  same response positions and the same LM head. Its commitment projection is
  identity-initialized so a newly added layer does not scramble an already
  norm-aligned plan state. Unlike answer-memory injection, commitment is not
  confidence-floor gated by default; otherwise the model only learns to use the
  plan after it is already confident.
- IMTA/GRAM/PTRM route breadth must be alive during training. If selector
  entropy collapses toward zero, the model no longer has meaningful internal
  trajectory exploration even if `imta_trajectories=3`; V2 therefore uses a
  small same-body route probability floor plus train-time route entropy and
  route-balance losses as internal anti-collapse regularizers, never as
  external answer selection.
- BLT-style byte, position, and chunk-length embeddings must use small
  initialization (`std=0.02` in the local implementation). Large default
  embedding initialization destabilizes short from-scratch runs and can mask
  whether the recurrent path itself is useful.
- Response-phase embeddings are part of the primary fastlane answer path. They
  give the speaker a causal, response-relative "answer line" signal, which is
  distinct from absolute prompt position and keeps answer-prefix planning from
  leaning on high-frequency continuation shortcuts. Tied input/output
  embeddings remain optional until a clean gate proves they help under the
  current causal dechunk/core path.
- Promotion evidence must report `promotion_evidence_eligible=true`; any
  diagnostic/stochastic decode knob makes that field false.
- Promotion evaluation is free autoregressive generation only.
- Promotion runs must not use the torch smoke core.
- Latest-method discipline is mandatory. If a proposed repair is only an old
  decoding workaround, it may be logged as a diagnostic but cannot become the
  promoted path unless the same free-generation gate proves it is unnecessary
  for the claimed improvement.
- Primary training must use explicit optimizer-step accounting. `steps` means
  optimizer updates; `grad_accum_steps` means micro-batches per update. This
  prevents a small local run from being mistaken for a larger effective-batch
  training recipe.
- Primary fastlane runs must log optimizer schedule, warmup, gradient norm,
  effective tokens per optimizer step, TensorBoard logdir, and Aim experiment
  metadata. A run without these fields is a smoke/debug run, not promotion
  evidence.
- The free-generation gate must also expose internal answer-plan grounding.
  This is not forced choice: no candidate answers are supplied and no plan is
  used to select among outputs. It only reports whether the same-head latent
  answer-prefix planner ranks the gold upcoming tokens highly before the model
  commits to free generation.

## Latest-Method Anchors

The primary path should track current architecture evidence, not old decoding
workarounds:

| date | source | mechanism | local failure explained | candidate implication |
| --- | --- | --- | --- | --- |
| 2026-05-28 | [Reasoning in Memory](https://arxiv.org/abs/2605.30343) | Use fixed memory blocks for latent reasoning instead of generating visible thought tokens. | Extra generated thought text is not the same as better internal computation. | Keep reasoning in the internal path; do not add visible candidate thoughts or forced-choice selection. |
| 2026-05-26 | [Learn from your own latents and not from tokens](https://arxiv.org/abs/2605.27734) | Predict own latent representations of related views instead of relying only on token prediction. | Token-only supervision is sample hungry and does not force stable internal hierarchy. | Keep own-latent prediction, but target the next causal chunk state to avoid residual identity copying. |
| 2026-05-21 | [GatedDeltaNet-2](https://arxiv.org/abs/2605.22791) / [NVlabs code](https://github.com/NVlabs/GatedDeltaNet-2) | Decouple erase and write gates in the recurrent linear-attention state. | A compressed recurrent state can scramble memory if erase/write are tied. | Promotion core is official GDN2, not torch smoke. |
| 2026-05-20 | [Equilibrium Reasoners](https://arxiv.org/abs/2605.21488) | Learn solution-aligned attractors and scale depth/breadth at test time. | More think steps do not help if the latent dynamics converge to the wrong basin. | Track depth benefit and IMTA trajectory breadth; do not call attractor active from loss alone. |
| 2026-05-14 | [$\phi$-Balancing for Mixture-of-Experts Training](https://arxiv.org/abs/2605.15403) | Balance expected routing distribution with differentiable population-level routing pressure. | The 2026-05-31 V2 continuation showed IMTA selector entropy near zero, so multiple trajectories existed in name but not in use. | Add internal route probability floor, entropy/balance regularization, and telemetry; keep final answer free generation through the same LM head. |
| 2026-05-10 | [LoopUS](https://arxiv.org/abs/2605.11011) | Stable looped latent refinement with selective gating and deep supervision. | Looped hidden states can drift or collapse without input-dependent control. | IMTA/GDN2 loops need telemetry for selector entropy, route diversity, and causal answer impact. |
| 2026-05-08 | [Fast BLT](https://arxiv.org/abs/2605.08044) | BLT generation improves through block/draft/verification-style training and decoding. | Byte-level generation is slow and easy to destabilize if byte/patch contracts are wrong. | Keep tokenizer-aware stop handling and causal chunk summaries; later speed work can use verification-style BLT, not old repetition tricks. |
| 2026-02-03 | [Reasoning with Latent Tokens in Diffusion Language Models](https://arxiv.org/abs/2602.03769) | Jointly predict multiple unresolved/latent tokens, improving planning and global coherence; also useful as an auxiliary objective for autoregressive models. | V2 first-token memory could start answers but not plan the next few answer tokens. | Answer memory is now a same-head, norm-aligned, jointly refined answer-prefix planner rather than a single-token attractor. |
| 2026-04-06 | [Projected Autoregression](https://arxiv.org/abs/2601.04854) | Predict/refine in continuous state space and project to tokens only at commitment time. | Repeated greedy token selection commits too early and amplifies loops. | Add causal token maturation before the same LM head; do not hide loops with repetition penalty. |
| 2025-12-06 | [Self-Autoregressive Refinement](https://arxiv.org/abs/2512.06421) | Train on lightweight model rollouts to reduce train-test mismatch and add student-forcing supervision for self-generated contexts. | V2 learns the first answer token under teacher forcing but collapses once its own token becomes the next input. | Add same-head self-rollout continuation loss; do not use it as candidate reranking. |
| 2024-12-13 | [BLT](https://arxiv.org/abs/2412.09871) / [Meta code](https://github.com/facebookresearch/blt) | Dynamic byte patches allocate compute where byte entropy is high. | Boundary-byte shortcuts lose information inside patches. | Boundary state source is causal chunk summary. |
| 2024-12-08 | [Coconut](https://arxiv.org/abs/2412.06769) | Continuous latent thought can replace explicit CoT tokens for some reasoning phases. | Latent thinking by name does not guarantee decoded answer quality. | RI claims require free-generation and depth/causality evidence. |

## Fastlane Priority

The active priority is the single core V2 recipe:

```text
WGRAMReasoningLMV2
core_implementation = official_gated_delta2
official_gdn2_force_chunk_eval = true
grad_accum_steps = 4 by default
lr = 2.2e-4 by default
lr_schedule = warmup_cosine
optimizer_warmup_fraction = 0.03
min_lr_ratio = 0.1
tensorboard_logdir = /tmp/wgram_eval/<run_name>
aim_repo = /tmp/wgram_aim
imta_trajectories = 3
own_latent_prediction_weight > 0
imta_diversity_weight > 0
imta_route_min_probability > 0
imta_route_entropy_floor > 0
imta_route_entropy_weight > 0
imta_route_balance_weight > 0
repeat_unlikelihood_weight = 0 by default
premature_stop_loss_weight > 0
response_start_loss_weight > 0
response_start_stop_margin_weight > 0
response_continue_stop_margin_weight = 0 by default
response_continue_stop_margin_schedule = steps_relative_delayed_linear_ramp
response_body_loss_weight > 0
response_stop_loss_weight > 0
response_stop_loss_schedule = steps_relative_delayed_linear_ramp
self_rollout_loss_weight > 0
balanced_response_sampler = true
force_fixed_boundaries = false
speaker_position_encoding = learned_absolute
token_maturation_steps = 2
token_maturation_aux_loss_weight > 0
answer_memory_steps = 2
answer_memory_plan_tokens = 4
answer_memory_plan_layers = 1
answer_memory_prompt_context = false by default
answer_memory_prompt_context_gate_init = -1.0
answer_memory_aux_loss_weight > 0
answer_memory_confidence_gate = same_lm_head_plan_confidence
answer_memory_commitment_scale = 1.0
answer_memory_commitment_confidence_gate = false
answer_prefix_commitment_loss_weight > 0
answer_memory_commitment_schedule = plan_grounding_then_open
answer_memory_injection_schedule = grounded_first_then_delayed_ramp
adaptive_latent_bridge = true
optional_donor_initialization = BLT PrefixLM surface path transplant
optional_v2_initialization = previous W-GRAM V2 full-body checkpoint
speaker_response_phase_encoding = learned_segment_relative
tie_input_output_embeddings = false by default
generation_repetition_penalty = 1.0 by default
evaluation_policy = free_generation_only
```

Non-core comparisons are deferred. Do not spend the next iteration on K sweeps,
own-latent-off sweeps, candidate rerankers, forced-choice gates, or historical
track comparisons unless the single core recipe fails and the failure must be
localized.

## Current Implementation Status

- `src/wgram_lm/v2/config.py`: V2 config.
- `src/wgram_lm/v2/contracts.py`: fail-fast SSOT validation.
- `src/wgram_lm/v2/chunk_encoder.py`: causal BLT chunk summaries.
  The boundary scorer is initialized conservatively and feeds differentiable
  boundary probabilities into the chunk summary, so the dynamic BLT path is not
  just a detached hard-threshold label. Dechunking uses
  `completed_chunk_or_bos`, not future chunk boundaries.
- `src/wgram_lm/v2/recurrent_core.py`: torch smoke implementation and official
  GatedDeltaNet-2 3:1 recurrent/attention adapter. The explicit attention leg
  uses a causal mask. Promotion/eval runs force the official GDN2 chunk kernel
  for full-prefix recomputation; fused recurrent eval remains an explicit
  diagnostic switch, not the default free-generation evidence path.
- `src/wgram_lm/v2/imta.py`: same-body latent trajectory adapter with route
  symmetry breaking before and after the recurrent core. It now reports
  selector effective route count, route mass min/max, entropy-floor violation,
  route probability floor, and route-balance loss so route collapse cannot hide
  behind a nominal `imta_trajectories=3` setting.
- `src/wgram_lm/v2/latent_prediction.py`: own-latent auxiliary targeting next
  causal chunk state.
- `src/wgram_lm/v2/speaker.py`: causal byte speaker with the same LM head.
  It includes learned absolute position embeddings inside the speaker path.
  It also includes causal token maturation: a shared latent refinement block
  with input-dependent gates, confidence telemetry, and optional deep
  supervision before final projection through the same LM head.
  Before token maturation, a prompt-grounded answer memory planner reads the
  first response prediction state, refines it internally, expands it into a
  short position-conditioned answer-prefix plan, can optionally re-read the
  causally visible prompt context through a gated same-body attention path,
  jointly refines the latent plan tokens, commits the answer-prefix plan into
  the same response positions, injects that plan into response positions, and
  optionally receives
  multi-token prefix supervision through the same decoder norm plus same LM
  head. Injection is confidence-gated by the same LM-head plan distribution, so
  an ungrounded planner cannot fully overwrite the normal speaker just because
  the schedule reached scale `1.0`. Commitment is schedule-gated rather than
  confidence-gated by default and its projection is identity-initialized; this
  prevents a low-confidence planner from being permanently ignored and prevents
  a random new projection from corrupting the answer path. The primary trainer
  can ground this memory first and delay its injection; this follows the
  RiM-style lesson that latent memory blocks must be grounded before they are
  trusted as the answer path.
  The latent bridge is selectively gated per token so donor language state and
  recurrent latent state can be blended without an all-or-nothing scalar.
  Response-phase embeddings are now the primary fastlane default because the
  causal post-audit gate removed corrected short-period loops without using
  decode-time repetition controls. Input/output tying remains implemented but
  optional until it has fresh causal-path evidence.
- `src/wgram_lm/v2/model.py`: `WGRAMReasoningLMV2`.
- `src/wgram_lm/v2/generation.py`: free-generation-only decoding API with
  recorded repetition statistics. Repetition penalty exists only as a labeled
  diagnostic switch and is disabled by default. The repetition metric detects
  both single-token collapse and short-period loops such as `2{2{2{...`, so
  alternating loops cannot hide behind a low adjacent-repeat count. It also
  records whether the autoregressive first generated token matches the
  teacher-forced top-1 token under deterministic greedy free generation.
- `scripts/590_train_wgram_v2_prefixlm.py`: minimal DataIO PrefixLM V2 trainer
  and free-generation-only checkpoint gate. The gate records exact match,
  tokenizer-aware decoded samples, generated loop stats, EOS/answer-stop
  token rates, byte-shift applicability, decoded-text stats, and teacher-forced
  first-response-token rank/top-5 evidence. It also records answer-memory plan
  token accuracy/top-5/rank/gold-probability diagnostics under the same LM head;
  these diagnostics never provide candidate answers and never select the final
  completion. The gate additionally records first-token consistency between
  teacher-forced top-1 and deterministic autoregressive generation; a mismatch
  rejects the evidence as an eval-path problem before any architecture claim.
  The trainer can warm-start V2's byte embedding and causal speaker
  decoder/head from an older BLT PrefixLM checkpoint; this is a donor
  surface-path transplant, not a candidate selector or alternate answer head.
  It can also warm-start the full V2 body from a previous V2 checkpoint so
  improvements like answer-memory grounding are not discarded between core
  runs. The trainer now uses explicit optimizer-step accounting with optional
  gradient accumulation, warmup/cosine scheduling, gradient-norm telemetry,
  effective-token accounting, TensorBoard scalar/text logging, and optional Aim
  experiment tracking. This is part of the training contract, not a dashboard
  convenience.
- `scripts/591_wgram_v2_fastlane.py`: single-recipe W-GRAM V2 fastlane runner that writes
  a manifest, trains the core V2 recipe, then runs the free-generation gate.
  It restores the PrefixLM answer-start contract by default with response-start
  CE, response-start stop-margin loss, response-body continuation CE, true
  response-stop CE, and premature answer-stop suppression. A scheduled
  response-body stop-margin diagnostic is implemented but disabled by default
  after the immediate/scheduled 120-step gates revived `2000...` loops. The
  stop CE uses a delayed linear ramp so early training first learns to start
  and continue an answer before being strongly pushed to close it. It also
  trains a short
  same-head self-rollout continuation pass so the decoder sees its own
  generated response prefix during training. The fastlane also uses a
  response-balanced sampler so short direct answers and frequent first digits do
  not dominate early scratch training. It now defaults to `warmup_cosine`
  optimization, `grad_accum_steps=4`, `/tmp/wgram_eval/<run_name>` TensorBoard
  logs, and `/tmp/wgram_aim` Aim tracking;
  these are train-time contract losses, not decode-time anti-repeat penalties.
  The old `scripts/590_train_qtrm_v2_prefixlm.py` and
  `scripts/591_qtrm_v2_fastlane.py` names remain thin compatibility wrappers
  only.

`core_implementation="official_gated_delta2"` wires the existing
`OfficialGatedDeltaNet2Mixer` adapter and refuses CPU forward passes because the
official kernels require CUDA/Triton. Torch smoke runs remain non-promotion.

## Promotion Gap

This file does not claim that V2 is already a good reasoning model. The current
V2 implementation provides a tested canonical path and smoke core. Promotion
still requires:

- long official GatedDeltaNet-2 training/eval runs with the latest-method
  defaults;
- free-generation depth scaling, not forced choice;
- teacher-forced first-token rank and EOS/answer-stop-token diagnostics;
- deterministic first-token teacher-forced/autoregressive consistency under
  greedy free generation;
- answer-memory plan grounding diagnostics for upcoming answer-prefix tokens;
- repetition and byte-decodability gates without legacy anti-repeat masking;
- optimizer schedule/gradient/effective-batch logs in TensorBoard or Aim;
- RI-1 through RI-7 evidence under matched data.

## 2026-05-31 Local Gate Evidence

Current official-core short runs remain rejected. The useful part is that the
failure is now localized.

| run | key changes | free-generation result | interpretation |
| --- | --- | --- | --- |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_DYNBLT_STOPMARGIN` | dynamic BLT + answer-start/body/stop losses | `1111...`, `exact_fraction=0.0`, `loop_like_fraction=1.0` | stop collapse was fixed, but continuation collapsed to a frequent digit. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_DYNBLT_POSINIT` | plus speaker position embeddings and small BLT-style init | `2222...`, `exact_fraction=0.0`, `loop_like_fraction=1.0` | first-token rank improved, but body continuation still failed. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_DYNBLT_TIED` | plus tied input/output embeddings and response-phase embeddings | `4444...`, `exact_fraction=0.0`, `loop_like_fraction=1.0` | tying/phase did not solve continuation in the short scratch setting. |
| `20260531_QTRM_V2_FASTLANE_CORE_S180_BAL_SELFROLL` | plus response-balanced sampling and same-head self-rollout continuation | `1111...`, `exact_fraction=0.0`, `loop_like_fraction=1.0` | self-rollout trains, but the tiny run still falls into the high-frequency digit attractor. |
| `20260531_QTRM_V2_CORE_S180_BAL_SELFROLL_POSONLY` | disables tied/phase while keeping balanced sampling + self-rollout | `1111...`, `exact_fraction=0.0`, `loop_like_fraction=1.0` | the remaining issue is not just tied/phase; it is a stronger decoder/head/capacity or scale bottleneck. |
| `20260531_QTRM_V2_FASTLANE_CORE_S180_CAUSAL_STOPCE` | causal dechunk/core attention plus response-stop CE weight `0.5` | `<|box_end|>`, `stop_fraction=1.0`, `first_token_stop_fraction=1.0` | stop CE is needed but a high early weight flips the failure back to premature stop. |
| `20260531_QTRM_V2_FASTLANE_CORE_S180_CAUSAL_STOPCE005` | causal dechunk/core attention plus response-stop CE weight `0.05` | `1111...`, `stop_fraction=0.0`, `loop_like_fraction=1.0` | low stop CE is too weak at 180 steps; the next fix needs scheduling or more scale, not another static scalar. |
| `20260531_QTRM_V2_FASTLANE_CORE_S240_STOPRAMP` | delayed stop CE ramp, target `0.3`, 240 steps | `1111...`, `stop_fraction=0.0`, `loop_like_fraction=1.0` | ramp was active but too weak at 240 steps. |
| `20260531_QTRM_V2_FASTLANE_CORE_S600_STOPRAMP` | delayed stop CE ramp, target `0.3`, 600 steps | mixed early close / short loops, `stop_fraction=1.0`, `first_token_stop_fraction=0.8125`, `loop_like_fraction=0.0625` | ramp solved most loop collapse but target `0.3` over-pushed premature stop. |
| `20260531_QTRM_V2_FASTLANE_CORE_S600_STOPRAMP02` | delayed stop CE ramp, target `0.2`, starts at 120 | mixed early close / loops, `stop_fraction=1.0`, `first_token_stop_fraction=0.75`, `loop_like_fraction=0.25` | lower target helped only slightly; stop supervision still starts too early. |
| `20260531_QTRM_V2_FASTLANE_CORE_S600_STOPRAMP02_LATE` | delayed stop CE ramp, target `0.2`, starts at 300 | `<|box_end|>`, `stop_fraction=1.0`, `first_token_stop_fraction=1.0` | delaying the ramp did not prevent stop from overtaking answer-start; target still too high for this scale. |
| `20260531_QTRM_V2_FASTLANE_CORE_S600_STOPRAMP01_LATE` | delayed stop CE ramp, target `0.1`, starts at 300 | `2222...`, `stop_fraction=0.0`, `loop_like_fraction=1.0` | target `0.1` is too weak; static midpoint hunting is a local minimum. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_MATURATION` | causal token maturation, no donor | `1111...`, `exact_fraction=0.0`, `loop_like_fraction=1.0`, `stop_fraction=0.0` | maturation alone does not fix a scratch surface-language collapse. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_MATURATION_DONOR` | BLT PrefixLM surface donor plus token maturation | mixed short answers, mostly `3<|box_end|>`, `loop_like_fraction=0.125`, `stop_fraction=1.0` | donor transplant breaks the infinite loop and learns stop timing, but first-token semantics still collapse to a frequent digit. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_MATURATION_DONOR_SELECTIVE` | donor plus input-dependent latent bridge gate | `3<|box_end|>` / `36<|box_end|>`, `loop_like_fraction=0.0`, `stop_fraction=1.0` | selective bridge removes loop collapse in the quick gate, but answer correctness is still absent; the next bottleneck is first-token conditioning and longer semantic learning, not repetition decoding. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_DONOR_SELECTIVE_ANSWERMEM` | direct prompt-grounded answer memory injection | `3<|box_end|>`, `loop_like_fraction=0.0`, `stop_fraction=1.0` | direct injection did not improve first-token semantics and reduced output variety. Treat this as evidence for a grounding-first injection schedule, not as a reason to remove internal memory. |
| `20260531_QTRM_V2_FASTLANE_CORE_S300_DONOR_SELECTIVE_ANSWERMEM_SCHED` | answer memory auxiliary from step 1, injection ramp from step 195 to 300 | `3<|box_end|>`, `loop_like_fraction=0.0`, `stop_fraction=1.0` | the schedule works mechanically, but 300 steps is not enough to ground the memory; the remaining bottleneck is semantic first-token learning under the current tiny local sample/step budget. |
| `20260531_QTRM_V2_FASTLANE_CORE_S1200_DONOR_SELECTIVE_ANSWERMEM_SCHED` | same scheduled answer-memory recipe, 1200 steps, 4096 rows | varied but still wrong answers (`1`, `10`, `B`, `(-1`, `We...`), `exact_fraction=0.0`, `loop_like_fraction=0.0625`, `stop_fraction=0.9375`, first-token mean rank `7.5` | longer grounding breaks the single-token `3` attractor and greatly improves first-token rank, but free-generation correctness is still missing. Next bottleneck is semantic supervision/scale, not decode-time repetition control. |
| `20260531_QTRM_V2_FASTLANE_CORE_CONT600_FROM_S1200` | full V2 continuation from the 1200-step checkpoint, injection and self-rollout active from step 1 | `exact_fraction=0.0625`, `loop_like_fraction=0.0`, `stop_fraction=0.9375`, first-token mean rank `15.75`, but `imta_selector_entropy` fell near zero | first exact free-generation sample appeared and loop collapse stayed fixed, but IMTA route collapse means GRAM/PTRM-style breadth is not yet alive. The next code change is internal route anti-collapse, not forced-choice evaluation or repetition decoding. |
| `20260531_QTRM_V2_FASTLANE_CORE_ROUTEGUARD_CONT240_FROM_S1200` | route entropy/balance losses without a route probability floor | `exact_fraction=0.0`, `loop_like_fraction=0.0`, `stop_fraction=1.0`, top-5 first-token fraction `0.8125`; final effective routes only `1.14` | loss-only route pressure is too weak once the selector has a strong single-route basin. V2 escalates to a small same-body route probability floor so unused trajectories receive answer-path gradient. |
| `20260531_QTRM_V2_FASTLANE_CORE_ROUTEFLOOR_CONT120_FROM_S1200` | route entropy/balance losses plus `imta_route_min_probability=0.05` | `exact_fraction=0.0`, `loop_like_fraction=0.0`, `stop_fraction=1.0`, top-5 first-token fraction `0.6875`, first-token mean rank `11.81`; selected effective routes rose from `1.48` at step 1 to `1.75` at step 120 | the route floor keeps the answer path from degenerating to a single trajectory and raw route entropy begins to recover, but answer correctness still needs stronger semantic training/data scale. This is a mechanism repair, not a solved RI model. |
| `20260531_QTRM_V2_FASTLANE_CORE_CONTINUESTOP_CONT120_FROM_ROUTEFLOOR` | continuation-stop margin active from step 1 | `exact_fraction=0.0`, `loop_like_fraction=0.25`, `stop_fraction=0.9375`; examples extended from `2<|box_end|>` to `20/200...<|box_end|>` | the margin addresses early close but over-extends if switched on immediately. Do not make it a primary default. |
| `20260531_QTRM_V2_FASTLANE_CORE_CONTINUESTOP_SCHED120_FROM_ROUTEFLOOR` | continuation-stop margin delayed/ramped from step 78 to 120 | `exact_fraction=0.0`, `loop_like_fraction=0.125`, `stop_fraction=1.0`; examples still include `2000000<|box_end|>` | scheduling reduces but does not eliminate the loop revival. Keep this margin as diagnostic until answer-prefix planning is grounded. |
| `20260531_QTRM_V2_FASTLANE_CORE_PREFIXPLAN_CONT120_FROM_ROUTEFLOOR` | answer-prefix memory planner plus scheduled continuation-stop margin | `exact_fraction=0.0`, `loop_like_fraction=0.1875`, `stop_fraction=0.875`; answer-memory plan aux tokens active but accuracy still `0.0` at 120 steps | prefix planning is mechanically active, but coupling it immediately with continuation margin is still too unstable. The next primary gate should train prefix planning without continuation margin. |
| `20260531_QTRM_V2_FASTLANE_CORE_PREFIXPLAN_NOMARGIN_CONT120_FROM_ROUTEFLOOR` | answer-prefix memory planner, continuation-stop margin disabled | `exact_fraction=0.0`, `loop_like_fraction=0.0625`, `stop_fraction=1.0`; plan aux active but accuracy still `0.0` at 120 steps | disabling the margin restores loop stability, but the new plan module needs more grounding/scale before it improves answer correctness. |
| `20260531_QTRM_V2_FASTLANE_CORE_NORMPLAN_CONT120_FROM_ROUTEFLOOR` | prefix planner passes through decoder norm and one joint latent-token refinement layer | `exact_fraction=0.0`, `loop_like_fraction=0.0625`, `stop_fraction=1.0`; plan aux accuracy reached `0.25` at 120 steps | norm-aligning the planner to the same LM-head interface makes the plan objective learnable in the small gate, though generation correctness is still absent. |
| `20260531_QTRM_V2_FASTLANE_CORE_NORMPLAN_SCHED300_FROM_ROUTEFLOOR` | norm-aligned planner, delayed injection schedule | `exact_fraction=0.0`, `loop_like_fraction=0.0`, `stop_fraction=1.0`, first-token top-5 `0.75` | delayed injection keeps loops down and improves first-token shortlist, but full-scale injection still hurts plan grounding. V2 adds same-LM-head confidence-gated injection next. |
| `20260531_QTRM_V2_FASTLANE_CORE_CONFPLAN_IMMEDIATE120_FROM_ROUTEFLOOR` | same-LM-head confidence-gated planner injection, injection schedule forced active from step 1 | `exact_fraction=0.0`, `loop_like_fraction=0.0625`, `stop_fraction=1.0`, first-token top-5 `0.6875`; injection confidence scale was `0.0` at steps 1 and 120, briefly `0.069` at step 60 | the confidence gate prevents low-confidence plans from fully overwriting the speaker even when nominal injection scale is `1.0`. It is a useful safety guard, not yet a correctness solution. |
| `20260531_QTRM_V2_LATESTMETHOD_NORMSTABLE_S300` | activation/logit stabilization, decoder-normed final LM head, IMTA output norm | `exact_fraction=0.0`, `loop_like_fraction=1.0`, `stop_fraction=0.0`; all rows generated `2222...`; nonfinite logits `0` | normalization removed NaN/inf instability and preserved route breadth, but the normal speaker still fell into a high-frequency digit attractor. |
| `20260531_QTRM_V2_LATESTMETHOD_TOPKMASS_S300` | same-LM-head answer-memory confidence changed from top-1 probability to top-5 probability mass | `exact_fraction=0.0`, `loop_like_fraction=1.0`, `stop_fraction=0.0`; plan top-5 improved to `0.426`, mean top-5 mass `0.328` | the planner-to-speaker gate now opens during training, but the opened plan is still too weak to change free generation; the speaker remains locked on `2`. |
| `20260531_QTRM_V2_LATESTMETHOD_STOPPLAN_S300` | answer-memory plan premature-stop margin added to discourage `<|box_end|>` before answer-prefix tokens | `exact_fraction=0.0`, `loop_like_fraction=1.0`, `stop_fraction=0.0`; plan accuracy fell to `0.085`, plan top-5 fell to `0.191` | suppressing premature stop alone is the wrong pressure at this scale. It removes one planner collapse mode but weakens answer-token commitment, so the next move must strengthen answer-prefix commitment rather than add decode-time repetition controls. |
| `20260531_QTRM_V2_LATESTMETHOD_COMMITPREFIX_CONT240_FROM_ROUTEFLOOR` | answer-prefix commitment added, but commitment still shared the same confidence-floor scale as injection | `exact_fraction=0.0625`, `loop_like_fraction=0.0` under the older metric, `stop_fraction=1.0`; final commitment scale was nominally `1.0` but plan top-k mass fell below the `0.2` floor in many batches | this exposed a contract bug: confidence-gating the commitment path means the model only receives commitment when the planner is already confident. Keep confidence gating for injection, not for the train-time commitment route. |
| `20260531_QTRM_V2_LATESTMETHOD_COMMITPREFIX_OPEN_CONT180_FROM_ROUTEFLOOR` | commitment confidence gate disabled, so schedule opens the answer-prefix path even when planner confidence is low | `exact_fraction=0.0625`, `loop_like_fraction=0.0` under the older metric, `stop_fraction=1.0`, first-token rank worsened to `29.56` | opening commitment without fixing the new projection initialization still pushes hidden states through a randomly initialized map. The causal route is right, but the initialization contract is wrong. |
| `20260531_QTRM_V2_LATESTMETHOD_COMMITPREFIX_IDENTITY_CONT180_FROM_ROUTEFLOOR` | commitment confidence gate disabled and commitment projection identity-initialized | under the corrected periodic-loop metric: `exact_fraction=0.125`, `loop_like_fraction=0.1875`, `stop_fraction=0.9375`, first-token mean rank `21.69`, plan top-k mass `0.431` | identity commitment improves exact free generation from 1/16 to 2/16 and keeps same-head plan mass alive, but it also reveals short-period loops such as `2{2{...`. This is progress, not promotion. The next bottleneck is continuation policy/semantic grounding, not decode-time repetition masking. |
| `20260531_QTRM_V2_LATESTMETHOD_PREFIXONLY_CONT180_FROM_ROUTEFLOOR` | answer-prefix plan injection restricted to the first planned response tokens only; no tail clamp of the last plan state over the whole response | `exact_fraction=0.125`, `loop_like_fraction=0.125`, `stop_fraction=0.9375`, first-token mean rank `22.56`, first-token top-5 `0.375`, plan top-5 mass `0.435` | removing tail-clamped plan injection reduces the corrected periodic loops from 3/16 to 2/16, but generated text still falls into `2{...` and short symbolic answers. The bug was real; the reasoning model is still not solved. |
| `20260531_QTRM_V2_LATESTMETHOD_PHASEPREFIX_CONT180_FROM_ROUTEFLOOR` | response-phase/relative answer-position embeddings enabled under the causal post-audit path, with repetition penalty still disabled | `exact_fraction=0.125`, `loop_like_fraction=0.0`, `stop_fraction=1.0`, first-token mean rank `22.63`, first-token top-5 `0.5625`, plan top-5 mass `0.309` | response-phase signals eliminate the corrected short-period loops in this 16-row greedy free-generation gate and improve first-token top-5, but exact accuracy remains 2/16. Make phase embeddings the primary fastlane default for stability; do not claim RI success until semantic first-token grounding improves. |
| `20260531_QTRM_V2_LATESTMETHOD_CONTEXTPLAN_CONT180_FROM_PHASEPREFIX` | optional causal prompt-context read added to answer-prefix memory and continued from the phase-prefix checkpoint | chunk-eval strict-loop re-eval: `exact_fraction=0.125`, `loop_like_fraction=0.0625`, `stop_fraction=1.0`, first-token consistency `1.0`, first-token top-5 `0.625`, plan top-5 `0.489`, plan top-5 mass `0.464`; one long low-diversity `Evaluating t 2...` sample is flagged | prompt-context read increases internal plan grounding, but it does not beat phase-prefix free-generation correctness and can create a structural-token/low-diversity attractor. Keep the module as staged-grounding diagnostic code, off in the primary fastlane until a later curriculum proves it helps decoded answers. The earlier `0.875` first-token consistency was an eval-path issue from automatic fused-recurrent GDN2 eval, not model reasoning. |
| `20260531_QTRM_V2_LATESTMETHOD_DISCIPLINE_CONT300_FROM_PHASEPREFIX` | restored training discipline: `grad_accum_steps=4`, `lr=2.2e-4`, warmup/cosine, TensorBoard/Aim logging, 300 optimizer steps from phase-prefix checkpoint | `exact_fraction=0.0625`, `loop_like_fraction=0.0`, `stop_fraction=1.0`, first-token consistency `1.0`, first-token top-5 `0.6875`, first-token mean rank `45.94`, plan accuracy `0.191`, plan top-5 `0.553`, plan top-5 mass `0.461` | the latest-method training loop is mechanically healthy and preserves no-loop free generation, while internal answer-plan grounding improves. It still degrades exact generation versus phase-prefix and often maps answers to frequent short strings such as `14`, `B`, `10`, or `\text{1}`. Do not call this a good reasoning model. The next move is longer semantic grounding or a stronger planner-to-speaker curriculum, not repetition penalty or forced-choice evaluation. |

Post-audit correction: those rejected runs also used a train/generation-mismatched
dechunk path. Earlier token positions gathered the next chunk boundary latent,
and the core's explicit attention layer was non-causal. The current V2 code
fixes both with `completed_chunk_or_bos` dechunking and causal core attention.
Any new gate must be interpreted separately from the rejected table above.

The post-causality STOP-CE checks show a narrow stop-transition problem:
`0.5` closes immediately, while `0.05` never stops. Do not promote either
static setting. The current code therefore uses delayed linear stop-loss
scheduling. The 600-step target-`0.3` ramp reduced `loop_like_fraction` to
`0.0625` but over-closed answers. Target `0.2`, even with a later ramp start,
still closed too early; target `0.1` loops. Do not keep hunting scalar
midpoints as the main strategy. The active fastlane now uses a steps-relative
continuation-first schedule (`start_fraction=0.65`, `warmup_fraction=0.35`,
target `0.15`) so 300/600/long runs preserve the same curriculum shape. If
that still oscillates between digit repetition and early close, the next move
is a stronger continuation/stop transition architecture.

Observed data prior on the 4,096-row local sample:

- first response token `1`: 702 rows (`17.1%`);
- first response token `2`: 480 rows (`11.7%`);
- body tokens are also dominated by `1`, `2`, and `0`.

Therefore repeated `1/2/4` outputs should be read as a high-frequency token
attractor under insufficient continuation learning, not as successful reasoning.
Do not promote any of these runs.

Current active diagnosis after the latest-method commitment gates:

- The run is free-generation only: greedy decoding, `repetition_penalty=1.0`,
  no forced choice, no rerank, no oracle pass@K.
- The free-generation gate now requires deterministic first-token consistency:
  under greedy promotion settings, the first autoregressive token must match
  the teacher-forced top-1 token for the same prefix. The official GDN2
  full-prefix eval path is chunk-forced by default because the auto
  `fused_recurrent` short-sequence path produced a false 2/16 mismatch on the
  context-plan checkpoint.
- NaN/logit instability is no longer the primary blocker.
- IMTA route floor keeps nominal selected breadth alive, but raw selector
  entropy still often collapses toward one route by the end of short
  continuations. Treat GRAM/PTRM breadth as partially alive, not solved.
- The answer planner has partial same-head signal and commitment now reaches
  the normal speaker path, but semantic first-token grounding is still weak and
  continuation falls into short-period high-frequency token loops.
- The corrected loop metric now flags alternating patterns, so future RI-2
  claims cannot rely on the old adjacent-repeat-only loop score. It also
  catches medium-length low-diversity answer drift such as
  `Evaluating t 2 ... 2 2`.
- Response-phase/relative answer-position embeddings are now default in the
  fastlane because the causal post-audit gate reduced corrected periodic loops
  to zero without changing the free-generation policy.
- Optional prompt-context answer-memory read improves plan diagnostics but
  failed the short free-generation gate as a primary default. The code remains
  for staged grounding, but SSOT keeps it disabled by default to avoid adding a
  new route that improves internal metrics while hurting decoded answers.
- The previous short V2 gates used small effective batches and constant or
  implicit learning-rate updates. Treat them as architecture diagnostics, not
  final learning-capacity evidence. The next primary run must use the restored
  optimizer discipline: gradient accumulation, warmup/cosine decay, TensorBoard
  and Aim telemetry, and reported effective tokens per optimizer step.
- The first restored-discipline run confirms that optimizer hygiene alone does
  not solve semantic answering. It keeps loops at zero and raises answer-plan
  top-5 signal, but exact free generation remains worse than the best
  phase-prefix gate. The model is still learning answer-shape priors more than
  task semantics.
- The next architecture move should target continuation semantics and
  planner-to-speaker grounding at larger scale or with stronger semantic
  supervision. Do not spend more GPU on scalar confidence-floor tuning or
  decode-time penalties.

## RI Evidence Ledger

- RI-1: free-generation quality must improve with useful `think_steps`; the
  same checkpoint must expose first-token rank so failure is visible before
  long decoded strings.
- RI-2: generated samples must avoid repeated-token collapse and premature EOS;
  loop-like fraction, max run, EOS fraction, answer-stop fraction, and decoded
  token diagnostics are required gate fields.
- RI-2 continuation note: if answer-start is fixed but the model repeats one
  ordinary token, use same-head response-body continuation CE before considering
  any decode-time repetition workaround.
- RI-2 stop-collapse note: if free generation emits only `<|box_end|>`, the
  first fix is same-head response-start stop-margin training plus gate metrics
  for `gold_minus_best_stop_logit`; do not use repetition penalty to hide it.
- RI-2 short-answer note: if free generation starts with a plausible answer
  token but closes early (`2<|box_end|>` for `24<|box_end|>`), add same-head
  continuation-stop margin on body positions. This is not a decode-time minimum
  length rule; it teaches the normal LM head that body gold tokens must beat
  stop tokens until the true stop position.
- RI-2 no-stop note: if free generation never emits `<|box_end|>`, add
  same-head response-stop CE at the true stop positions. Premature-stop
  suppression alone teaches "not yet", not "stop now." Keep the stop CE
  conservative; high early weights reproduce premature close.
- RI-2 continuation-collapse note: if the model chooses a plausible first
  token but then repeats it (`2222...` or `4444...`), train on short
  self-rollout response prefixes with the same LM head. This directly addresses
  train-test mismatch and is not a decoding-time penalty.
- RI-2 frequency-collapse note: if the repeated token is also a high-frequency
  first/body token in the sampled data, use response-balanced sampling before
  escalating to another architecture module. This keeps the normal answer path
  intact while reducing a data-prior shortcut.
- RI-2 escalation note: if balanced sampling plus self-rollout still repeats a
  frequent token, stop adding local scalar losses. Escalate to a decoder/head
  capacity change or a longer language pretraining phase before RI claims.
- RI-3: mechanism evidence is not accepted from auxiliary loss alone. Later
  perturbation/ablation checks may be debug-only, but promotion needs proof
  that latent trajectories affect decoded answers.
- RI-4: memory must remain inside the recurrent latent loop.
- RI-5: the 3:1 GatedDeltaNet-2/attention hybrid is the promotion core.
- RI-6: telemetry must show active mechanisms, not dead uniform selectors or
  trivially copied own-latent targets.
- RI-7: data efficiency must be judged on matched free-generation gates, not
  candidate/rerank scores.

Matched K=1/K>1 and own-latent on/off comparisons are demoted to debug-only
follow-ups. They are not the next priority unless the fastlane run fails in a
way that requires mechanism localization.
