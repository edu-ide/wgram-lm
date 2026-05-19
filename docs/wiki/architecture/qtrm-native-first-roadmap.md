# QTRM-Native First Roadmap

Date: 2026-05-12

Status: canonical direction, not a completed capability claim.

## Strategic Update 2026-05-19: Pretrained-Init First

HRM-Text shows that recurrent latent-reasoning LMs can be pretrained from
scratch with far less compute/data than conventional scaling recipes, but its
reference resource targets are still outside a fast single-DGX-Spark loop:

```text
HRM-Text L  / 0.6B:  8 x H100 for about 50 hours
HRM-Text XL / 1.0B: 16 x H100 for about 46 hours
```

Therefore, the near-term canonical execution path is:

```text
Qwen3.5 pretrained initialization
+ strict TRM shared z_H/z_L recurrent core
+ L=3 recurrent schedule
+ dedicated halt/depth objectives
+ language healing without erasing core causality
```

Do not attempt a 0.6B/1B from-scratch HRM-Text-style run on a single DGX Spark
as the main path. Use from-scratch runs only for small proof models and
architecture gates. The acceptance target remains:

```text
QTRM/Qwen-pretrained-init model beats Qwen3.6-27B on the same benchmark suite
while preserving destructive core-ablation causality.
```

HRM-Text training lessons to import:

```text
1. keep a real hierarchical/nested recurrent forward, not an optional sidecar;
2. train with packed PrefixLM-style language data once core causality is stable;
3. select checkpoints by causal gates, not only loss;
4. preserve language logits while adding recurrent computation;
5. use clean, structured public data before scaling raw token volume;
6. treat from-scratch HRM-Text-size pretraining as a later multi-H100 path,
   not the fastest single-DGX-Spark route.
```

Current Qwen3.5-preinit evidence:

```text
local_eval/qwen35_preinit_checksum_traj_w2_eval512_20260519

512-case accepted: true
gain: 0.0234375
language_top1_agreement: 1.0

family gains:
  chain5:      +0.0467836257
  checksum4:   0.0
  select_pair: +0.0235294118
```

Implication:

```text
HRM-Text-style trajectory shaping helps stabilize an aggregate mandatory-core
gain to 512 cases without damaging language logits. It still does not solve
composition: checksum4 gain is 0.0. The next canonical architecture candidate
must make recurrent step states causally feed the final LM residual path.
```

Trajectory carry update:

```text
local_eval/qwen35_preinit_trajcarry_mean_w2_eval512_20260519

512-case accepted: true
gain: 0.037109375
language_top1_agreement: 0.875

family gains:
  chain5:      +0.0643274854
  checksum4:  +0.0058479532
  select_pair:+0.0411764706

carry-off ablation:
  gain: 0.02734375
  checksum4 gain: 0.0
```

Updated implication:

```text
The recurrent trajectory now has causal evidence in the LM-logit path: turning
off carry removes the first positive checksum4 gain. This is the current best
Qwen3.5-preinit QTRM-native signal. It is not yet a robust public-benchmark
claim because the checksum gain is small and language top1 drops to 0.875.

Route-only frozen-Qwen repeat:
  gain: 0.0234375
  checksum4 gain: 0.0
  language_top1_agreement: 1.0

Therefore, frozen mean carry is language-safe but too weak. The next step is a
learned carry mixer or a deliberately small Qwen healing scope with stricter
language preservation.

Follow-up:

```text
learned carry route-only 512:
  gain: 0.00390625
  checksum4 gain: 0.0
  decision: rejected

mean carry route-only 512:
  gain: 0.00390625
  checksum4 gain: 0.0
  decision: rejected
```

Updated next step:

```text
Do not continue frozen route-only carry sweeps. The current evidence says the
route is useful only when paired with a small pretrained-backbone healing
scope. Future promotion runs must use 512-case periodic selection and
language-aware checkpoint scoring.
```

512-selected result:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_s100_20260519

selected directly on 512 cases
gain: 0.021484375
checksum4 gain: +0.0058479532
min_family_gain: +0.0058479532
language_top1_agreement: 0.875
decision: accepted

carry-off ablation:
  gain: 0.0078125
  checksum4 gain: 0.0
  decision: rejected
```

Current best claim:

```text
Qwen3.5-preinit QTRM with mean trajectory carry now has a 512-selected
carry-dependent reasoning gain. The recurrent trajectory is causally useful in
the LM-logit path on this synthetic gate. The remaining blocker is language
preservation at a stronger probe scale.
```

Extended language update:

```text
local_eval/qwen35_preinit_trajcarry_mean_512select_lang0875_eval512_extendedlang_20260519

language_probe_set: extended
num_prompts: 32
accepted: true
gain: 0.021484375
language_top1_agreement: 0.96875
checksum4 gain: +0.0058479532
```

Updated best claim:

```text
The current best Qwen3.5-preinit QTRM checkpoint preserves the Qwen language
path on a 32-prompt English/Korean probe while keeping the 512-case
carry-dependent reasoning gain. The next bottleneck is no longer immediate
language collapse. It is converting this small synthetic causal signal into
broader language-reasoning improvement.
```

HRM-Text-style healing result:

```text
local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_s40_20260519

response-only clean language healing
accepted: true
gain: 0.03125
language_top1_agreement: 0.96875
checksum4 gain: +0.0058479532

carry-off ablation:
  local_eval/qwen35_preinit_trajcarry_mean_hrmtext_heal_s40_carryoff_20260519
  accepted: false
  gain: 0.0
```

Updated best claim:

```text
The current best local result is an HRM-Text-inspired training improvement, not
a new architecture claim: response-only clean language healing raises the
512-case QTRM gain from 0.021484375 to 0.03125, preserves extended language
top-1 agreement at 0.96875, and the gain disappears when trajectory carry is
disabled. This is the first clean sign that language healing and latent
recursive gain can coexist in the Qwen3.5-preinit QTRM path.
```

HRM-Text application rule:

```text
Apply HRM-Text only as training discipline:
  packed PrefixLM-style clean language healing
  recurrent/nested forward always mandatory
  checkpoint selection by causal reasoning gates plus language preservation
  destructive core/carry ablations

Do not treat HRM-Text as a shortcut to instant innovation. Its from-scratch
resource recipe is not the same as our Qwen3.5-preinit route, and its benchmark
claims do not transfer until our own public benchmarks pass.
```

The current scaling evidence says:

```text
len4 short gate:
  HRM-style separate H/L learns faster.

len6/len7/len8 scaling:
  strict TRM shared z_H/z_L is the better baseline on exact/family floor.

len8 bottleneck:
  exact improves, but worst-ablation drop is still negative, so core causality
  must be repaired before claiming a robust reasoning core.
```

## Hard Lock 2026-05-16: TRM-Paper-Condition QTRM-Native

The canonical target is now stricter than "QTRM is executed during forward".
The target is:

```text
QTRM-native loop reasoning model satisfying the TRM paper conditions.
```

Canonical causal path:

```text
chat-template / prompt tokens
-> tokenizer
-> native token embeddings
-> native QTRM backbone
-> mandatory TRM-style recursive latent loop
   z_L / z_H state is updated for multiple recurrent steps
   the recurrent state is carried inside the answer-producing path
   optional halt/early-exit may stop only after the loop has computed state
-> core-dependent decoder/readout
-> LM head
-> autoregressive text
```

"Mandatory" means causally necessary, not merely present. A candidate is not
canonical if the base LM/Qwen path can answer while QTRM only adds an optional
residual, reranking score, hidden answer field, MemoryOS result, or formatting
controller.

Promotion requires TRM-style destructive evidence:

```text
full QTRM-native > no-loop / shallow-loop baseline
full QTRM-native > core_off / think0
deeper loop steps beat shallower loop steps on held-out cases
z_L/z_H reset, zero, shuffle, detach, or corruption reduces the same metric
core-dependent readout off reduces the same metric
final answer comes from normal LM logits and autoregressive decoding
```

Do not promote as QTRM architecture progress:

```text
Qwen donor/residual adapter improvement alone
Qwen-preservation-first tuning where QTRM is only a bounded delta
forced-choice gain without greedy/LM-logit generation gain
MemoryOS/RAG/tool/verifier success hiding the recursive core
post-hoc answer arbitration, candidate solver, or side renderer
MMLU/MCQ gain without core-depth and state-corruption causality
```

Current implication:

```text
The recent Qwen-integrated residual path is diagnostic/legacy, not the
canonical QTRM-native claim. It may provide baselines, tokenizer/backbone
lessons, or language-healing data, but the main proof must return to a
single native token->recursive-loop->LM-logit model that satisfies the TRM
causal ablations above.
```

Correct next order:

```text
1. prove the donorless/native TRM loop improves a small held-out reasoning
   or language-reasoning gate through LM logits;
2. prove depth/state causality with destructive ablations in the same run;
3. only then attach larger pretrained backbones or language-healing curricula;
4. reject any path where residual_scale=0 or core_off preserves the claimed
   gain.
```

Milestone form:

```text
M-A Recursive-core reasoning proof:
  prove the recursive core actually improves reasoning, with depth/state
  ablations, before relying on any donor, retrieval, verifier, or sidecar.

M-B Core-to-LM attachment:
  attach that proven core to the normal LM path so the answer is produced by
  token embeddings -> recursive loop -> core-dependent readout -> LM logits.

M-C Language healing:
  only after M-A and M-B, run language healing / pretrained initialization /
  partial unfreeze / corpus scaling so the model speaks well without erasing
  the recursive-core causal gain.
```

## Hard Lock 2026-05-14

QTRM-native is now the mandatory canonical architecture path.

Donor/residual/Qwen-sidecar work is diagnostic only. It must not be proposed
as the main route for proving the architecture unless the user explicitly
reverses this lock.

Canonical path:

```text
prompt tokens
-> tokenizer
-> native token embeddings
-> native encoder/backbone
-> mandatory TRM/QTRM recursive thinking core
-> native decoder/readout
-> LM head
-> autoregressive text
```

Immediate reset:

```text
Do not continue trying to promote synthetic TRM reasoning before native
language viability. The next canonical target is tiny native LM pretraining:
loss down, non-degenerate greedy text, and recurrence/core ablations that do
not reveal a hidden shortcut.
```

## Decision

The canonical architecture target is QTRM-native:

```text
tokenizer / token ids
-> token embeddings
-> native encoder
-> mandatory TRM/QTRM recursive thinking core
-> native decoder/readout
-> LM head
-> autoregressive text
```

The donor-sidecar architecture remains useful for diagnostics, but it is no
longer the canonical route for proving a general-LLM reasoning architecture.

## Why We Pivoted

The donor-sidecar path repeatedly exposed the same failure:

```text
Qwen donor hidden states
-> QTRM workspace / recursive sidecar core
-> separate answer/readout loop
-> LM logits
```

Observed failures:

```text
Freeze / world-of repetition on donor-adapter generation
00000000 / 66666666 numeric collapse
forced-choice or rank improvements without strict greedy generation
answer/readout components that do not causally beat decoder-off
core_state_zero sometimes closer to the gold answer than the full core
```

The decisive structural issue is that the donor and recursive core are not a
single language-model residual path. A core can compute a useful latent signal
and still fail to express it through the frozen donor-facing output geometry.

## Native First Does Not Mean "No ETD"

Qwen-surgery ETD is not the final target:

```text
Qwen early layers
-> repeated Qwen middle block
-> Qwen later layers
-> Qwen LM head
```

That is a bridge if we need to preserve Qwen language ability.

QTRM-native still keeps the ETD structural lesson:

```text
encode
-> repeated thinking block / TRM recursive core
-> decode
-> LM head
```

The difference is that the weights are native QTRM weights, not a frozen donor
with a sidecar adapter.

## Official TRM Is The Canonical Core

The project goal is not merely to build another looped LM/Ouro-style repeated
transformer. MHA ETD is allowed as a viability baseline, warm-start source, or
performance floor, but it is not the final architecture claim.

Canonical QTRM-native reasoning must eventually use an official-TRM-style
recursive core:

```text
tokens
-> native embeddings / causal encoder
-> mandatory z_L / z_H official TRM recursive core
-> z_H-dependent decoder/readout
-> LM logits
-> autoregressive text
```

Promotion rule:

```text
Do not promote an MHA ETD recurrent result as the final QTRM architecture.
Promote only if an official-TRM-core variant beats or matches the recurrent LM
floor and still fails under think0, state_reset, op_zero, z_l_zero, and z_h_zero
ablations on the same LM-generation metric.
```

Current diagnosis:

```text
The MHA ETD path proved that the native LM scaffold and curriculum can work.
It did not prove the official TRM hypothesis. Official TRM variants currently
fail mainly because z_L/z_H state is not yet a strong enough causal source for
LM logits as length increases. The next work should therefore improve the
official TRM core-to-logits bridge and recurrent state training, not keep
scaling MHA ETD as if it were the final model.
```

## Short-Term Goal

The immediate goal is not human-level language. It is native LM viability:

```text
Can a minimal QTRM-native model, with no donor, learn next-token prediction and
produce stable greedy text through its own recursive path?
```

Minimum success criteria:

```text
1. donor disabled
2. no hidden evidence, MemoryOS, RAG, answer renderer, candidate solver, or sidecar
3. token embedding -> native encoder -> recursive thinking core -> decoder -> LM head
4. next-token loss decreases on tiny text or synthetic language
5. tiny overfit succeeds
6. greedy generation does not collapse into repeated tokens
7. think_steps/depth improves a held-out metric
8. think_steps=0, thinking_block_off, state_reset/core_zero, and op_zero ablations drop
9. on-policy samples do not replay the same prompt/answer blocks
```

Current language-generation status, 2026-05-15:

```text
QTRM-native can now learn a small clean text curriculum and Qwen-tokenized
teacher-continuation corpus without donor inference in the model path.
The first strict surface-answer run was rejected as broad language evidence
because it repeated short User/Assistant snippets even though loss and depth
metrics were good. EOS-separated Qwen-tokenized records and a larger diverse
answer-only corpus repaired the immediate fixed-block replay.

strict rejected report:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_surface_strict_s1200_20260515/report.json

reject reason:
  on_policy_unique_line_fraction_too_low

strict heldout false-accept control:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_heldout_eos_strict_v2_s1200_20260515/report.json

reject reason:
  on_policy_extra_assistant_marker

accepted controlled instruction/EOS repair:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_heldout_eos_repair_s1200_20260515/report.json

accepted repair meaning:
  The model can produce short controlled answer-only instruction responses and
  stop at EOS through the donorless native path.

accepted semantic repair:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_semantic_eos_repair_s1200_20260515/report.json

accepted semantic meaning:
  The same controlled instruction/EOS path now also requires keyword-level
  semantic relevance for heldout prompts.

rejected paraphrase generalization gate:
  local_eval/qtrm_native_language_generalization_gate_semantic_repair_20260515/report.json

reject reason:
  on_policy_semantic_relevance_too_low

rejected gate meaning:
  The model keeps answer format and EOS behavior on less familiar paraphrases,
  but semantic keyword coverage is still too weak. This blocks any broad
  language-ability claim today.

accepted paraphrase curriculum repair:
  local_eval/qtrm_native_language_generalization_gate_paraphrase_repair_20260515/report.json

accepted repair meaning:
  After adding a small controlled paraphrase curriculum, the same heldout
  generalization gate passes. This proves quick repairability of a narrow
  semantic paraphrase gate, not broad language ability.

unseen2 stress:
  local_eval/qtrm_native_language_generalization_gate_unseen2_uncertainty_repair_20260515/report.json

unseen2 result:
  rejected. Uncertainty-family examples recover anti-guessing prompts, but the
  paragraph/readers prompt regresses to a generic good-answer response.

family-balanced curriculum:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_family_balanced_s1200_20260515/report.json

family-balanced result:
  accepted as a controlled language bootstrap. Depth helps strongly on eval
  loss: depth4 0.3257 vs think0 2.6102.

family-balanced unseen stress:
  local_eval/qtrm_native_language_generalization_gate_unseen2_family_balanced_20260515/report.json

family-balanced unseen result:
  rejected. Paragraph/readers, Korean anti-guessing, and uncertain-facts prompts
  pass keyword relevance, but the claims/support/trust prompt regresses to the
  wrong answer family.

next repair target:
  family-DRO or a larger external answer-only dataset. Do not keep appending
  isolated examples as if that proved broad language ability. Any next
  acceptance must pass unseen family stress plus core-depth/core-off
  non-regression.

external dataset policy:
  If the language bottleneck is semantic-family interference, answer-family
  routing, Korean/English coverage, or open-domain fluency, download or sample
  external datasets instead of hand-authoring more tiny examples.

  Current approved offline sources:
    - HuggingFaceH4/ultrachat_200k, default/train_sft
    - HuggingFaceFW/fineweb-edu, sample-10BT/train or equivalent edu text
    - beomi/KoAlpaca-v1.1a, default/train
    - FreedomIntelligence/alpaca-gpt4-korean if KoAlpaca coverage is weak

  Builder:
    scripts/357_build_external_language_corpus.py

  Example command:
    PYTHONPATH=src .venv/bin/python scripts/357_build_external_language_corpus.py \
      --out local_eval/external_language_corpus/qtrm_native_external_mix_20260515.jsonl \
      --max-records 1200 \
      --max-record-chars 1000 \
      --min-record-chars 50

  Required artifact:
    JSONL corpus plus `.report.json` with dataset/config/split, source counts,
    character counts, rejected counts, and filters.

  Boundary:
    External data is allowed only as offline training/evaluation data. It must
    not become a runtime donor, hidden retrieval path, visible CoT target, or
    sidecar answer solver. After using it, rerun unseen family stress and
    native core depth/core-off non-regression.

external dataset results, 2026-05-15:
  first external mix:
    accepted bootstrap and passed unseen2 semantic stress.

  wider unseen3:
    rejected. The model still routes many wider prompts to the wrong small
    answer family.

  external-dominant:
    rejected. Pushing external-stage steps too hard caused repeated-number
    collapse.

  balanced source corpus:
    built with source caps:
      UltraChat 400, FineWeb-Edu 400, KoAlpaca 400.

  balanced d128:
    accepted bootstrap but failed unseen3.

  balanced d256:
    rejected; larger d_model alone did not fix semantic routing.

  tied embeddings:
    implemented and tested; mixed result, still rejected.

  next diagnosis:
    Qwen/Qwen3.5 tokenizer has a 248k vocabulary. A d128/d256 native model is
    spending too much capacity learning output token geometry from scratch.
    Before scaling reasoning claims, test vocabulary pressure directly with a
    reduced/custom tokenizer, staged active-vocab curriculum, sampled softmax,
    or stronger offline logit/KD artifacts.

not proven:
  broad language ability
  semantic generalization beyond keyword-level controlled prompts
  open-domain instruction following

next language gate:
  larger answer-only corpus
  shuffled diverse prompts
  on-policy loop rejection
  extra Assistant/User marker rejection
  semantic relevance rejection
  core-depth/core-off non-regression
  no visible CoT in the surface language channel
```

TiDAR boundary note, 2026-05-15:

```text
TiDAR ("Think in Diffusion, Talk in Autoregression") is useful to track for a
later QTRM-native decoder/serving stage, not for the current language
acquisition bottleneck. The idea maps cleanly to:

  QTRM recurrent core -> AR answer logits
  plus parallel diffusion-style draft slots
  plus AR verification/acceptance

It should not replace the native recursive core or the AR final answer path.
Use only after the native language bootstrap and raw reasoning gates are stable.
```

## Backbone And Memory Schedule

Because QTRM-native is a new model, the final architecture should leave room
for both a fast local backbone and a scalable memory route. That does not mean
all components should be enabled in the first viability gate.

Canonical staged design:

```text
tokens
-> local language backbone
   baseline: MHA ETD block
   candidate: Qwen3.5-style Delta / Delta / Delta / Attention hybrid
-> mandatory TRM/QTRM recursive thinking core
-> decoder/readout backbone
-> LM head
-> autoregressive text

later long-memory route:
external docs / long context
-> MSA-style offline memory encoding
-> chunk pooled K/V/Kr memory
-> router top-k
-> sparse memory attention in upper layers
```

Promotion order:

```text
1. prove native MHA ETD L1 viability;
2. compare MHA ETD vs Qwen3.5-style 3:1 hybrid under the same gate;
3. promote the hybrid only if tokens/sec, loss, generation exactness, and
   depth/core ablations improve or remain non-regressive;
4. add MSA as an off-by-default interface after L1/L2, not as a hidden answer
   path;
5. promote MSA only after memory_off/router_off/chunk_shuffle ablations drop
   the same LM-generation metric.
```

Reason:

```text
Qwen3.5-style DeltaNet 3:1 is a local sequence/backbone efficiency candidate.
MSA is a long-memory scaling candidate.
Neither replaces the mandatory recursive reasoning core.
```

## Gate Ladder

## Canonical Runner

QTRM-native gates are now first-class entries in the research gate runner:

```text
qtrm_native_l1_mha
qtrm_native_l1_hybrid
qtrm_native_l2_curriculum_depth
qtrm_native_l3_language_slice
qtrm_native_l4_mixed_text_reasoning
qtrm_native_l5_multifamily
qtrm_native_l5_language_nonregression
```

The short wrapper is:

```bash
bash scripts/338_run_qtrm_native_gate.sh
```

Default wrapper target:

```text
gate: qtrm_native_l4_mixed_text_reasoning
profile: standard
```

Runtime-only smoke:

```bash
PROFILE=smoke WRITE_WIKI=0 bash scripts/338_run_qtrm_native_gate.sh
```

Smoke must not be treated as a capability claim. It proves that the runner can
execute the probe, parse `report.json`, and preserve decisive metrics. Some
smoke profiles intentionally relax training-depth-sensitive thresholds so the
CI-sized wiring check can complete in a few seconds.

## Autoresearch-Style Operations

Karpathy's `autoresearch` repository is not a QTRM architecture prior. It is
an experiment-operation prior: fixed-budget runs, one decisive metric, a
results ledger, and keep/discard discipline.

QTRM adapts it this way:

```text
autoresearch fixed 5-minute val_bpb
  -> QTRM fixed triage budgets with raw-intelligence metrics

autoresearch results.tsv
  -> QTRM experiment ledger plus wiki promotion/rejection notes

autoresearch keep/discard commits
  -> QTRM keep/discard checkpoints and recipes; rejected runs are not
     canonical even if they finish successfully

autoresearch single editable train.py
  -> QTRM one changed mechanism per run: architecture, loss, or schedule, not
     all at once
```

Required one-line result for every serious QTRM-native run:

```text
run_id
changed_mechanism
baseline_checkpoint
full_generation_exact
min_family_generation_exact
full_minus_think0
full_minus_worst_ablation
status: keep / discard / probe / crash
next_action
```

The default wrappers now write the operation ledger automatically:

```bash
bash scripts/300_run_research_gate.sh
bash scripts/338_run_qtrm_native_gate.sh
```

Default ledger path:

```text
local_eval/research_gate_runner/results.tsv
```

Ledger status semantics:

```text
keep    = gate accepted; candidate may become the next baseline after review
discard = gate ran but failed acceptance; do not promote checkpoint/recipe
probe   = dry-run or diagnostic-only result
crash   = command failed without an acceptable report
```

This applies to experiment operations only. It does not justify random
architecture hacking, hidden solvers, MemoryOS shortcuts, or changing the eval
harness to make a run pass.

Promotion evidence must come from `standard` runs or stricter profiles with:

```text
full generation exactness above threshold
think0 lower than full
state_reset/op_zero lower than full
final answer through LM logits
no donor, no MemoryOS, no hidden side solver
```

Seed-stability promotion is separate from a single standard run:

```bash
PYTHONPATH=src .venv/bin/python scripts/339_qtrm_native_seed_sweep.py \
  --profile standard --seeds 337 338 339
```

The current policy is strict:

```text
min_pass_rate: 1.0
min_exact per seed: 0.70
```

This means the accepted seed-337 L4 run is a canonical scaffold, not yet a
stable architecture claim across seeds. The next promotion requires the sweep
summary to report `decision: accepted_seed_stability`.

Current seed-stability result:

```text
out_dir:
  local_eval/qtrm_native_l4_seed_sweep

decision:
  accepted_seed_stability

seeds:
  337, 338, 339

pass_count:
  3 / 3

full_generation_exact:
  seed 337: 0.7421875
  seed 338: 0.74609375
  seed 339: 0.716796875

ablation floor:
  think0:      0.025390625 .. 0.03515625
  state_reset: 0.02734375  .. 0.03515625
  op_zero:     0.03125     .. 0.046875
```

Interpretation:

```text
The L4 QTRM-native scaffold is now seed-stable on the small mixed text
reasoning benchmark. This still is not broad LLM capability. It promotes the
native recurrent causal path from one-seed scaffold to a reproducible small
raw-reasoning result.
```

Next promotion gates:

```text
L5A: larger text vocabulary / BPE tokenizer, no char-only dependency
L5B: broader task families, not only modular arithmetic text prompts [accepted seed-stable]
L5C: language non-regression under recurrence on larger natural text [accepted seed-stable]
L5D: compare MHA ETD vs Qwen3.5-style hybrid under the same L4/L5 gates
L6: only then add MSA/LM2 memory with memory_off/router_off/chunk_shuffle
    ablations
```

### L5B Broader Reasoning Families

Purpose:

```text
prove the same native recurrent LM path can handle more than one reasoning
family from text-form prompts, with the family specified inside the same token
stream rather than through a side channel
```

Implemented families:

```text
modchain:
  apply op_1, op_2, ... in prompt order

revchain:
  apply op_n, ..., op_2, op_1 in reverse prompt order

checksum:
  answer = start + sum(op_ids) mod modulus
```

Prompt format:

```text
task modchain start 19 ops 06 05 06 02 answer 28\n
task revchain start 12 ops 01 04 03 03 answer 13\n
task checksum start 25 ops 05 06 03 05 answer 12\n
```

The `task <family>` tag is fixed-width and part of the normal text prompt. It
is not a hidden field or side channel.

Current standard run:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard

decision:
  accepted_l5_multifamily

setup:
  steps: 12000
  train_cases: 24576
  eval_cases: 768
  families: modchain, revchain, checksum
  d_model: 128
  n_heads: 8
  d_ff: 256
  backbone: mha_etd
  depth_intermediate_loss_weight: 0.5
  active_len_curriculum: true

thresholds:
  full_exact >= 0.60
  depth_gain >= 0.10
  ablation_drop >= 0.10
  min_family_exact >= 0.40

results:
  full_generation_exact: 0.6067708333333334
  think0_generation_exact: 0.020833333333333332
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.03515625
  full_minus_think0: 0.5859375
  full_minus_worst_ablation: 0.5716145833333334
  min_family_generation_exact: 0.4140625

by_family:
  checksum: 0.9375
  modchain: 0.46875
  revchain: 0.4140625
```

Interpretation:

```text
L5B has a first standard acceptance, but it is marginal. The recurrent path
still clearly matters because think0/state_reset/op_zero are near chance.
However, family balance is weak: checksum is easy, while modchain and revchain
are the active bottlenecks. Do not promote this to stable L5 until a seed sweep
passes and revchain/modchain are less brittle.
```

Hard-balanced seed stability:

```text
script:
  scripts/340_qtrm_native_l5_seed_sweep.py

out_dir:
  local_eval/qtrm_native_l5_multifamily_hardbalanced_sweep

decision:
  accepted_l5_seed_stability

policy:
  seeds: 337, 338, 339
  min_pass_rate: 1.0
  min_exact_per_seed: 0.60
  min_family_exact_per_seed: 0.40
  train families: modchain, revchain, modchain, revchain, checksum
  eval families: modchain, revchain, checksum

summary:
  pass_count: 3 / 3
  min_full_generation_exact: 0.7057291666666666
  max_full_generation_exact: 0.7565104166666666
  min_family_generation_exact: 0.55078125
  max_family_generation_exact: 0.62109375

seed 337:
  full_generation_exact: 0.7330729166666666
  min_family_generation_exact: 0.5625

seed 338:
  full_generation_exact: 0.7057291666666666
  min_family_generation_exact: 0.55078125

seed 339:
  full_generation_exact: 0.7565104166666666
  min_family_generation_exact: 0.62109375
```

Interpretation:

```text
L5B is now seed-stable under the hard-balanced family schedule. This promotes
the native recurrent LM path beyond a single modular arithmetic family. It
still remains a synthetic reasoning result, not broad language capability.
```

### L5C Language Non-Regression

Purpose:

```text
prove the native recurrent path does not win small reasoning gates by damaging
ordinary next-token language behavior
```

Gate:

```text
runner gate:
  qtrm_native_l5_language_nonregression

script:
  scripts/336_train_qtrm_native_text_probe.py

standard out_dir:
  local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_standard
```

Acceptance checks:

```text
full recurrent eval loss must beat random-loss threshold
greedy sample must avoid repeated-character collapse
full recurrent loss must not regress badly vs:
  think_steps=0
  thinking_block_off
  separately trained think0 baseline
```

Standard result:

```text
decision:
  accepted_l5_language_nonregression

setup:
  text_file: docs/wiki/architecture/qtrm-native-first-roadmap.md
  steps: 800
  baseline_steps: 800
  seq_len: 96
  d_model: 64
  n_heads: 4
  d_ff: 128

metrics:
  think_eval_loss: 1.8944039690879084
  think0_loss: 4.506251942726873
  thinking_block_off_loss: 4.506251942726873
  think0_baseline_loss: 1.8811764822852226
  full_vs_think0: 0.4203945969211712
  full_vs_thinking_block_off: 0.4203945969211712
  full_vs_baseline: 1.0070314970058616
  unique_chars: 21
  max_run_fraction: 0.015151515151515152
```

Interpretation:

```text
The recurrent model preserves the larger text-slice LM path: its recurrent
loss is effectively tied with the separately trained think0 baseline and much
better than disabling the thinking path inside the recurrent model. This is
one standard run. Seed stability is required before treating L5C as promoted.
```

Seed-stability result:

```text
script:
  scripts/341_qtrm_native_l5_language_seed_sweep.py

out_dir:
  local_eval/qtrm_native_l5_language_nonregression_seed_sweep

decision:
  accepted_l5c_seed_stability

policy:
  seeds: 337, 338, 339
  min_pass_rate: 1.0
  max_full_vs_baseline: 1.35

summary:
  pass_count: 3 / 3
  min_full_vs_baseline: 0.975550581055019
  max_full_vs_baseline: 1.003826688355419
  min_full_vs_think0: 0.4441652548042246
  max_full_vs_think0: 0.49055043638467777
  min_full_vs_off: 0.4441652548042246
  max_full_vs_off: 0.49055043638467777
```

Interpretation:

```text
L5C is now seed-stable on the larger roadmap text slice. The recurrent path is
not degrading next-token language loss versus a separately trained think0
baseline, and the same model becomes much worse if its thinking path is turned
off at eval time. The next orthodox step is L5D backbone comparison under the
already accepted L4/L5 reasoning gates.
```

### L1 Native LM Viability

Purpose:

```text
prove the model can be an autoregressive LM without Qwen donor help
```

Required evidence:

```text
tiny overfit loss decreases
held-out loss beats random/simple baseline
greedy generation emits answer/EOS without collapse
```

### L2 Native Recursive Gain

Purpose:

```text
prove the recursive thinking block causes a measured improvement
```

Required evidence:

```text
think_steps=4 or 8 > think_steps=0
thinking_block_off drops
state_reset/core_state_zero drops
op_zero or operation-shuffle drops on algorithmic tasks
final answer is generated by LM logits
```

### L3 Native Language Slice

Purpose:

```text
prove recurrence does not destroy basic natural language modeling
```

Required evidence:

```text
small Korean/English text slice
loss decreases
short greedy samples are non-degenerate
recursive block ablations do not falsely improve the language metric
```

Current smoke:

```text
script:
  scripts/336_train_qtrm_native_text_probe.py

out_dir:
  local_eval/qtrm_native_text_l3_smoke_cuda_s800

setup:
  donor disabled
  char-level tiny text slice
  seq_len: 64
  steps: 800
  backbone: mha_etd
  train/eval think_steps: 4

result:
  decision: accepted_l3_language_slice
  vocab_size: 33
  random_loss: 3.4965
  think_eval_loss: 0.04335
  think0_loss: 4.3128
  thinking_block_off_loss: 4.3128
  unique_chars: 30
  max_run_fraction: 0.01515
```

Interpretation:

```text
This is only a tiny language-slice smoke, but it confirms that the native
recurrent path can learn basic autoregressive text without immediate repetition
collapse. It does not yet prove broad natural-language capability.
```

### L4 Native Reasoning + Language

Purpose:

```text
prove native recurrence helps reasoning while preserving language behavior
```

Required evidence:

```text
mixed reasoning tasks with normal prompts
generation exactness, not forced-choice only
depth sweep gain
core/thinking-block ablation drop
language slice non-regression
```

Current scaffold:

```text
script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

prompt format:
  start 19 ops 06 05 06 02 answer

answer format:
  28\n
```

Best current run:

```text
out_dir: local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s10000
program_len: 4
modulus: 32
steps: 10000
decision: rejected
threshold exact >= 0.70
think4_generation_exact: 0.693359375
think0_generation_exact: 0.017578125
state_reset_generation_exact: 0.03515625
op_zero_generation_exact: 0.044921875
full_minus_think0: 0.67578125
full_minus_worst_ablation: 0.6484375
```

12000-step confirmation:

```text
out_dir: local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s12000
decision: rejected
think4_generation_exact: 0.693359375
think0_generation_exact: 0.009765625
state_reset_generation_exact: 0.033203125
op_zero_generation_exact: 0.03515625
full_minus_think0: 0.68359375
full_minus_worst_ablation: 0.658203125
```

Accepted L4 capacity-up run:

```text
out_dir: local_eval/qtrm_native_mixed_text_l4_program4_mod32_d128_cuda_s8000
program_len: 4
modulus: 32
steps: 8000
d_model: 128
n_heads: 8
decision: accepted_l4_mixed_text_reasoning
threshold exact >= 0.70
think4_generation_exact: 0.7421875
think0_generation_exact: 0.02734375
state_reset_generation_exact: 0.03515625
op_zero_generation_exact: 0.03125
full_minus_think0: 0.71484375
full_minus_worst_ablation: 0.70703125
```

Interpretation:

```text
The L4 mixed text path is now accepted in the small scaffold setting. This does
not prove broad language reasoning, but it does prove that a donorless
QTRM-native recurrent model can read a text-form prompt, perform multi-step
reasoning through the recurrent path, and generate a text-form answer with
strong destructive-ablation drops.
```

## Current Minimal Probe

Implemented scaffold:

```text
scripts/335_train_qtrm_native_etd_probe.py
tests/test_qtrm_native_etd_probe.py
```

The scaffold now supports two native backbone modes:

```text
--backbone mha_etd
--backbone qtrm_hybrid_3to1
```

`qtrm_hybrid_3to1` reuses the local `QTRMBlockStack` pattern:

```text
Delta mixer
Delta mixer
Delta mixer
Grouped-query attention
```

Important naming boundary:

```text
qtrm_hybrid_3to1 is Qwen3.5-style only.
It is not a strict official Qwen3.5 backbone reproduction.
```

Current local implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  NativeQTRMETDLM(backbone="qtrm_hybrid_3to1")

src/qtrm_mm/blocks.py
  QTRMBlockStack
  attn_every=4 gives 3 delta/recurrent blocks then 1 attention block

src/qtrm_mm/mixers.py
  torch_gated_delta is a local reference/debug fallback
```

Official implementation references now cloned locally:

```text
references/official/gated-deltanet
  upstream: https://github.com/NVlabs/GatedDeltaNet
  commit: b53d6d3a161267432a79c1c04af69fa52bddc921

references/official/flash-linear-attention
  upstream: https://github.com/fla-org/flash-linear-attention
  commit: 74d011f1bd58367b7bc6519dbc4d177d29b063e0

references/official/qwen35
  upstream: https://github.com/QwenLM/Qwen3.5
  commit: f1443092c29978643fd041ebe959676259e934f1

references/official/mamba
  upstream: https://github.com/state-spaces/mamba
  commit: a14b1dff0454a3bc27d9eb31355dc01e4b2490ec
```

Qwen3.5-2B config evidence:

```text
references/model_configs/qwen35_2b_base/config.json

num_hidden_layers: 24
layer_types: 6 x [linear_attention, linear_attention, linear_attention, full_attention]
full_attention_interval: 4
hidden_size: 2048
intermediate_size: 6144
linear_num_key_heads: 16
linear_num_value_heads: 16
linear_key_head_dim: 128
linear_value_head_dim: 128
num_attention_heads: 8
num_key_value_heads: 2
head_dim: 256
rope_theta: 10000000
partial_rotary_factor: 0.25
```

Qwen3.5 3:1 terminology correction:

```text
The Qwen3.5-2B config does not describe the 1-in-4 layer as
`gated_attention`. It describes the schedule as:

  linear_attention
  linear_attention
  linear_attention
  full_attention

repeated six times.

The Qwen README describes the efficient hybrid architecture as Gated Delta
Networks plus MoE. For the dense 2B config, the local architectural evidence is
therefore:

  3 x linear_attention, implemented through the GatedDeltaNet/FLA path
  1 x full_attention, the ordinary full-attention layer

Do not rename `full_attention` to `gated attention` unless a model source file
or config explicitly proves that this full attention block has a gated-attention
variant. In QTRM docs, use:

  Qwen3.5-style 3:1 = 3 GatedDeltaNet-like linear-attention layers
                       + 1 full-attention layer
```

Mamba-3 reference boundary:

```text
Official source:
  references/official/mamba

Paper:
  Mamba-3: Improved Sequence Modeling using State Space Principles
  https://arxiv.org/abs/2603.15569

Local implementation evidence:
  references/official/mamba/mamba_ssm/modules/mamba3.py
  class Mamba3

Key implementation knobs:
  d_state: 128
  expand: 2
  headdim: 64
  rope_fraction: 0.5 or 1.0
  is_mimo: false/true
  mimo_rank: 4
  chunk_size: 64 for SISO, 64 / mimo_rank for MIMO
```

Mamba-3 is a legitimate future backbone candidate because it is an
inference-first SSM with official code. It is not the same family as
GatedDeltaNet. It should not replace the currently accepted L5D placement
without a separate gate:

```text
MHA ETD encode -> Mamba-3 recurrent think -> MHA ETD decode -> LM head
```

Promotion requirements for a Mamba-3 candidate:

```text
1. strict official-source import from references/official/mamba or installed
   mamba-ssm source build;
2. no fallback toy SSM counted as Mamba-3;
3. same L5D seed-stability gate versus MHA ETD and official_fla_think;
4. same staged-placement language non-regression gate;
5. CUDA/kernel dependency status recorded, because Mamba-3 MIMO requires extra
   kernels such as TileLang and decode uses specialized step kernels.
```

Current priority:

```text
Do not interrupt the accepted L5D path to chase Mamba-3. The first direct
runtime verification has now been attempted:

  scripts/342_qtrm_native_l5d_backbone_compare.py
  --profile smoke
  --candidates mha_etd,official_fla_think,official_mamba3_think

Result:
  official_mamba3_think: command_failed

Failure class:
  official Mamba-3 module imports through the local reference adapter, but its
  Triton SISO kernel fails during compilation at the QK skip-connection
  tl.dot site under the current local toolchain.

Artifact:
  local_eval/qtrm_native_l5d_mamba3_compare_smoke

Therefore that first evidence did not prove GatedDeltaNet > Mamba-3 in model
quality. It proved only that Mamba-3 was blocked before scoring.
```

Follow-up runtime fix:

```text
cause:
  references/official/mamba/pyproject.toml requires triton>=3.5.0
  current .venv has triton 3.3.1 from torch 2.7.1

non-destructive local fix:
  install triton==3.5.1 into local_deps/mamba3_runtime
  keep the main .venv unchanged

command shape:
  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python ...

micro-forward:
  OfficialMamba3Mixer(d_model=64, n_heads=4, strict=True)
  CUDA forward OK
```

Mamba-3 short comparison after the runtime fix:

```text
artifact:
  local_eval/qtrm_native_l5d_mamba3_compare_short_triton351

candidates:
  mha_etd
  official_fla_think
  official_mamba3_think

winner:
  official_mamba3_think

full_generation_exact:
  mha_etd:                 0.015625
  official_fla_think:      0.036458333333333336
  official_mamba3_think:   0.046875

causal checks:
  official_fla_think:
    full_minus_think0:          0.03125
    full_minus_worst_ablation:  0.010416666666666668

  official_mamba3_think:
    full_minus_think0:          0.046875
    full_minus_worst_ablation:  0.010416666666666664
```

Interpretation:

```text
This is the first direct evidence that official Mamba-3 can be wired as the
QTRM recurrent think-core and can beat official_fla_think on a single short
seed. It is not yet canonical promotion. Promotion still requires seed
stability and language non-regression under the same gate standards.
```

Mamba-3 seed-stability follow-up:

```text
artifact:
  local_eval/qtrm_native_l5d_mamba3_seed_sweep_short_triton351

target_candidate:
  official_mamba3_think

decision:
  accepted_l5d_placement_seed_stability

promoted_count:
  3 / 3

causal_ok_count:
  3 / 3

backend_ok_count:
  3 / 3

min_delta_vs_mha:
  0.005208333333

max_delta_vs_mha:
  0.03125

min_full_generation_exact:
  0.046875

max_full_generation_exact:
  0.052083333333333336
```

Per-seed short comparison:

```text
seed 337:
  winner: official_mamba3_think
  mha_etd:               0.015625
  official_fla_think:    0.036458333333333336
  official_mamba3_think: 0.046875

seed 338:
  winner: official_fla_think by tie ordering
  mha_etd:               0.046875
  official_fla_think:    0.052083333333333336
  official_mamba3_think: 0.052083333333333336

seed 339:
  winner: official_mamba3_think
  mha_etd:               0.046875
  official_fla_think:    0.015625
  official_mamba3_think: 0.052083333333333336
```

Updated interpretation:

```text
The earlier statement "GatedDeltaNet is better than Mamba-3" is false for the
current evidence. After fixing the Mamba-3 runtime, the short seed sweep favors
official_mamba3_think as the stronger recurrent think-core candidate on this
synthetic L5D scaffold.

Do not yet demote official_fla_think from the broader canonical path: FLA has
already passed language non-regression and scaled-reasoning gates, while Mamba-3
has only passed runtime, short comparison, and short seed-stability. The next
promotion gates for Mamba-3 are:

  1. language non-regression with MHA encode/decode + Mamba-3 think;
  2. scaled reasoning under the same profile as qtrm_native_l5d_placement_scaled_reasoning;
  3. optional longer-context/memory stress if Mamba-3 remains stable.
```

Mamba-3 encode/decode isolation:

```text
artifact:
  local_eval/qtrm_native_l5d_all_mamba3_seed_sweep_short_triton351

question:
  Should encode/decode also move away from MHA ETD, or should Mamba-3 stay only
  in the recurrent thinking core?

candidates:
  mha_etd
  official_mamba3_think
  official_fla_encode_decode_mamba3_think
  official_mamba3

target summaries:
  placement_seed_sweep_summary.official_mamba3_think.json
  placement_seed_sweep_summary.official_mamba3.json
  placement_seed_sweep_summary.official_fla_encode_decode_mamba3_think.json
```

Result:

```text
official_mamba3_think:
  decision: accepted_l5d_placement_seed_stability
  promoted_count: 3 / 3
  causal_ok_count: 3 / 3
  backend_ok_count: 3 / 3
  min_delta_vs_mha: 0.005208333333
  max_delta_vs_mha: 0.03125
  min_full_generation_exact: 0.046875
  max_full_generation_exact: 0.052083333333333336

official_mamba3:
  decision: rejected
  promoted_count: 2 / 3
  causal_ok_count: 2 / 3
  backend_ok_count: 3 / 3
  min_delta_vs_mha: -0.005208333333
  max_delta_vs_mha: 0.036458333333
  min_full_generation_exact: 0.041666666666666664
  max_full_generation_exact: 0.078125

official_fla_encode_decode_mamba3_think:
  decision: rejected
  promoted_count: 2 / 3
  causal_ok_count: 3 / 3
  backend_ok_count: 3 / 3
  min_delta_vs_mha: -0.005208333333
  max_delta_vs_mha: 0.026041666667
  min_full_generation_exact: 0.041666666666666664
  max_full_generation_exact: 0.057291666666666664
```

Per-seed exact generation:

```text
seed 337:
  mha_etd:                                  0.015625
  official_mamba3_think:                   0.046875
  official_fla_encode_decode_mamba3_think: 0.041666666666666664
  official_mamba3:                         0.052083333333333336

seed 338:
  mha_etd:                                  0.046875
  official_mamba3_think:                   0.052083333333333336
  official_fla_encode_decode_mamba3_think: 0.041666666666666664
  official_mamba3:                         0.078125

seed 339:
  mha_etd:                                  0.046875
  official_mamba3_think:                   0.052083333333333336
  official_fla_encode_decode_mamba3_think: 0.057291666666666664
  official_mamba3:                         0.041666666666666664
```

Interpretation:

```text
Mamba-3 in all three stages can win individual seeds, but it is not stable
enough for promotion: seed 339 falls below MHA ETD and op_zero beats the full
path. FLA encode/decode with Mamba-3 think also fails seed stability.

The current Mamba-3 result therefore supports this staged placement only:

  MHA ETD encode -> official Mamba-3 recurrent think -> MHA ETD decode

Do not replace encode/decode MHA ETD yet. The encode/decode stages should stay
simple and stable until an all-Mamba or FLA-encode/decode candidate passes the
same seed-stability, causal-ablation, language-nonregression, and scaled
reasoning gates.
```

Mamba-3 language + scaled reasoning verification:

```text
artifact root:
  local_eval/research_gate_runner_mamba3_verify_20260512

architecture:
  MHA ETD encode
  -> official Mamba-3 recurrent think
  -> MHA ETD decode
  -> LM head

strict backend:
  mamba3_mixers: 1
  official_mamba3_mixers: 1
  torch_delta_mixers: 0
  all_mamba3_mixers_official: true
```

Language non-regression:

```text
gate:
  qtrm_native_l5d_mamba3_placement_language_nonregression

profile:
  standard

decision:
  accepted_l5d_mamba3_placement_language_nonregression

report:
  local_eval/research_gate_runner_mamba3_verify_20260512/
  qtrm_native_l5d_mamba3_placement_language_nonregression_standard/report.json

metrics:
  last_loss: 1.447218418121338
  think_eval_loss: 1.7053431420461507
  think0_loss: 5.173120769203132
  thinking_block_off_loss: 5.173120769203132
  think0_baseline_loss: 1.9871065920971809
  full_vs_think0: 0.32965461626151865
  full_vs_thinking_block_off: 0.32965461626151865
  full_vs_baseline: 0.8582041591670939
  sample_unique_chars: 21
  sample_max_run_fraction: 0.015151515151515152
```

Scaled reasoning:

```text
gate:
  qtrm_native_l5d_mamba3_placement_scaled_reasoning

profile:
  standard

decision:
  accepted_l5d_mamba3_placement_scaled_reasoning

report:
  local_eval/research_gate_runner_mamba3_verify_20260512/
  qtrm_native_l5d_mamba3_placement_scaled_reasoning_standard/report.json

setup:
  steps: 1200
  train_cases: 4096
  eval_cases: 384
  d_model: 64
  n_heads: 4
  d_ff: 128

metrics:
  full_generation_exact: 0.171875
  think0_generation_exact: 0.0
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.026041666666666668
  full_minus_think0: 0.171875
  full_minus_worst_ablation: 0.14583333333333334
  min_family_generation_exact: 0.046875
```

Optimization note:

```text
The first Mamba-3 scaled-reasoning standard run used d_model=96, which implies
headdim=24 in the current NativeMamba3Block wrapper. That run diverged to NaN
before step 200. Lowering LR to 1e-4 did not fix it.

The stable accepted setting uses d_model=64, n_heads=4, headdim=16. Do not
re-promote a wider Mamba-3 placement until the d_model=96/headdim=24 NaN
failure is fixed and rerun.
```

Interpretation:

```text
MHA encode/decode + official Mamba-3 think-core now has:

1. short seed-stability evidence;
2. standard language non-regression evidence;
3. standard scaled-reasoning evidence with causal ablation drops.

This is an architecture-level proof for the current native scaffold, not a
claim of broad general LLM capability. A stricter future proof should train a
single joint language+reasoning checkpoint and evaluate both tasks from that
same checkpoint.
```

Strict promotion rule:

```text
Do not call the current hybrid "official Qwen3.5".
Promote only after a separate official-backend gate uses FLA/NVlabs
GatedDeltaNet or Transformers Qwen3.5 modules in strict mode, follows the
Qwen3.5 config layer schedule, and passes the same L4/L5/L5C gates against
MHA ETD.
```

Official FLA runtime gate:

```text
runner gate:
  qtrm_native_l5d_official_fla_runtime

script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

backend:
  --backbone qtrm_hybrid_3to1
  --delta-backend fla_gated_delta
  --strict-backends
```

Smoke result:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_official_fla_runtime_smoke

decision:
  accepted_l5d_official_fla_runtime

backend_summary:
  fla_delta_mixers: 9
  official_fla_delta_mixers: 9
  torch_delta_mixers: 0
  all_fla_mixers_official: true
```

Short standard runtime result:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_official_fla_runtime_standard

decision:
  accepted_l5d_official_fla_runtime

setup:
  steps: 200
  train_cases: 1024
  eval_cases: 96
  d_model: 64
  n_heads: 4
  delta_head_dim: 16
  delta_num_v_heads: 4

backend_summary:
  fla_delta_mixers: 9
  official_fla_delta_mixers: 9
  torch_delta_mixers: 0
  all_fla_mixers_official: true

metrics:
  full_generation_exact: 0.03125
  think0_generation_exact: 0.020833333333333332
  state_reset_generation_exact: 0.010416666666666666
  op_zero_generation_exact: 0.0
```

Interpretation:

```text
This is a runtime/wiring acceptance only. It proves that the official FLA
GatedDeltaNet backend can run inside the native recurrent causal LM path with
no `torch_gated_delta` fallback. It does not prove that the official backend
outperforms MHA ETD. The next L5D step is a performance comparison under the
accepted L4/L5/L5C gates.
```

L5D short comparison:

```text
script:
  scripts/342_qtrm_native_l5d_backbone_compare.py

out_dir:
  local_eval/qtrm_native_l5d_backbone_compare_short

profile:
  short

candidates:
  mha_etd
  official_fla

full_generation_exact:
  mha_etd: 0.015625
  official_fla: 0.026041666666666668

delta:
  official_fla - mha_etd: 0.010416666667

official_fla_backend_ok:
  true

official_fla_causal_ok:
  false

official_fla_full_minus_think0:
  -0.005208333333333332

official_fla_full_minus_worst_ablation:
  -0.026041666666666668

official_fla_promoted:
  false
```

Interpretation:

```text
The official FLA backend is wired correctly and slightly beats MHA ETD in this
very short exact-generation comparison, but it fails the causal reasoning
requirement: think0/state_reset/op_zero are not lower than the full recurrent
path. Therefore official FLA is not promoted as the canonical backbone yet.
Next work should tune or train the official-FLA variant until the same causal
ablation standards that MHA ETD passed at L4/L5 are satisfied.
```

Depth-8 official FLA probe:

```text
out_dir:
  local_eval/qtrm_native_l5d_official_fla_depth8_short

setup:
  official FLA GatedDeltaNet strict backend
  train_think_steps: 8
  eval_think_steps: 8
  steps: 600
  train_cases: 4096
  eval_cases: 256

backend_summary:
  fla_delta_mixers: 9
  official_fla_delta_mixers: 9
  torch_delta_mixers: 0
  all_fla_mixers_official: true

metrics:
  full_generation_exact: 0.03515625
  think0_generation_exact: 0.03515625
  state_reset_generation_exact: 0.0390625
  op_zero_generation_exact: 0.03125
  full_minus_think0: 0.0
  full_minus_worst_ablation: -0.00390625
```

Interpretation:

```text
Increasing official-FLA recurrent depth from 4 to 8 does not fix causal use.
The full recurrent path ties think0 and loses to state_reset. The bottleneck is
not simply too-few thinking steps. Next L5D experiments should isolate where
FLA belongs:

1. FLA only in encode/decode with MHA thinking block;
2. MHA encode/decode with FLA thinking block;
3. lower learning rate or longer warmup for FLA;
4. compare teacher-forced answer loss and generation exact separately.
```

L5D placement isolation:

```text
shared setup:
  script:
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  steps: 400
  train_cases: 2048
  eval_cases: 192
  train_think_steps: 4
  eval_think_steps: 4
  strict_backends: true
  delta_backend: fla_gated_delta
  official FLA fallback count:
    torch_delta_mixers: 0
```

Placement A:

```text
out_dir:
  local_eval/qtrm_native_l5d_fla_encode_decode_mha_think_short

encode_backbone:
  qtrm_hybrid_3to1 / official FLA

think_backbone:
  mha_etd

decode_backbone:
  qtrm_hybrid_3to1 / official FLA

metrics:
  full_generation_exact: 0.057291666666666664
  think0_generation_exact: 0.041666666666666664
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.036458333333333336
  full_minus_think0: 0.015625
  full_minus_worst_ablation: 0.02083333333333333
  min_family_generation_exact: 0.03125
```

Placement B:

```text
out_dir:
  local_eval/qtrm_native_l5d_mha_encode_decode_fla_think_short

encode_backbone:
  mha_etd

think_backbone:
  qtrm_hybrid_3to1 / official FLA

decode_backbone:
  mha_etd

metrics:
  full_generation_exact: 0.08333333333333333
  think0_generation_exact: 0.041666666666666664
  state_reset_generation_exact: 0.036458333333333336
  op_zero_generation_exact: 0.046875
  full_minus_think0: 0.041666666666666664
  full_minus_worst_ablation: 0.03645833333333333
  min_family_generation_exact: 0.0625
```

Interpretation:

```text
The failed all-official-FLA run was not enough to reject FLA entirely. The
placement isolation shows that official FLA can be causally useful when placed
inside the recursive thinking block while the encode/decode stages remain MHA
ETD. This is the current L5D candidate.

Do not promote full qtrm_hybrid_3to1 as the canonical backbone yet. Promote
only the staged candidate:

  MHA encode -> official FLA recurrent think -> MHA decode -> LM head

Next requirement: rerun the staged candidate with a seed sweep and compare it
against the current MHA ETD canonical baseline under the same L5B/L5C gates.
```

L5D placement seed stability:

```text
script:
  scripts/343_qtrm_native_l5d_placement_seed_sweep.py

out_dir:
  local_eval/qtrm_native_l5d_placement_seed_sweep_short

profile:
  short

candidates:
  mha_etd
  official_fla_think

target_candidate:
  official_fla_think

seeds:
  337, 338, 339

decision:
  accepted_l5d_placement_seed_stability

promoted_count:
  3 / 3

causal_ok_count:
  3 / 3

backend_ok_count:
  3 / 3

min_delta_vs_mha:
  0.005208333333

max_delta_vs_mha:
  0.067708333333

min_full_generation_exact:
  0.052083333333333336

max_full_generation_exact:
  0.08333333333333333
```

Per-seed summary:

```text
seed 337:
  winner: official_fla_think
  full_generation_exact: 0.08333333333333333
  delta_vs_mha: 0.067708333333
  full_minus_think0: 0.041666666666666664
  full_minus_worst_ablation: 0.03645833333333333

seed 338:
  winner: official_fla_think
  full_generation_exact: 0.078125
  delta_vs_mha: 0.03125
  full_minus_think0: 0.06770833333333333
  full_minus_worst_ablation: 0.020833333333333336

seed 339:
  winner: official_fla_think
  full_generation_exact: 0.052083333333333336
  delta_vs_mha: 0.005208333333
  full_minus_think0: 0.052083333333333336
  full_minus_worst_ablation: 0.005208333333333336
```

Interpretation:

```text
The staged candidate is now seed-stable at the short L5D level. The canonical
QTRM-native L5D placement is therefore:

  MHA ETD encode -> official FLA GatedDeltaNet recurrent think -> MHA ETD decode

This is still not a full Qwen3.5 architecture. It is a native QTRM placement
where the official GatedDeltaNet layer is used only inside the mandatory
recursive thinking core. The next orthodox step is not another placement
search; it is to scale the accepted placement under stricter L5/L6 gates:

1. longer training and higher eval-cases for the accepted placement;
2. natural-text language preservation against MHA ETD;
3. then memory/long-context gates such as MSA or LM2.
```

L5D placement language non-regression:

```text
script:
  scripts/336_train_qtrm_native_text_probe.py

runner gate:
  qtrm_native_l5d_placement_language_nonregression

out_dir:
  local_eval/qtrm_native_l5d_official_fla_think_language_nonregression_standard

placement:
  MHA ETD encode
  official FLA GatedDeltaNet recurrent think
  MHA ETD decode

decision:
  accepted_l5d_placement_language_nonregression

metrics:
  last_loss: 0.6685409545898438
  think_eval_loss: 1.888507903768466
  think0_loss: 5.344207181380345
  thinking_block_off_loss: 5.344207181380345
  think0_baseline_loss: 2.025272998672265
  full_vs_think0_loss_ratio: 0.35337475507090776
  full_vs_thinking_block_off_loss_ratio: 0.35337475507090776
  full_vs_baseline_loss_ratio: 0.9324707854232689
  sample_unique_chars: 33
  sample_max_run_fraction: 0.022727272727272728
```

Interpretation:

```text
The staged placement does not win the short reasoning gate by destroying the
native LM path. On the small natural-text slice, recurrent inference is better
than think0/off ablations and stays within the non-regression threshold against
the separately trained think0 baseline. This makes the accepted L5D placement:

  MHA ETD encode -> official FLA GatedDeltaNet recurrent think -> MHA ETD decode

the current canonical QTRM-native backbone placement for the next scale-up.
```

L5D placement scaled reasoning:

```text
runner gate:
  qtrm_native_l5d_placement_scaled_reasoning

script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_placement_scaled_reasoning_standard

placement:
  MHA ETD encode
  official FLA GatedDeltaNet recurrent think
  MHA ETD decode

setup:
  steps: 1200
  train_cases: 4096
  eval_cases: 384
  d_model: 96
  n_heads: 4
  n_kv_heads: 2
  d_ff: 192
  train_think_steps: 4
  eval_think_steps: 4
  strict_backends: true

decision:
  accepted_l5d_placement_scaled_reasoning

backend:
  official_fla_delta_mixers: 3
  torch_delta_mixers: 0
  all_fla_mixers_official: true

metrics:
  full_generation_exact: 0.14583333333333334
  think0_generation_exact: 0.0
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.033854166666666664
  full_minus_think0: 0.14583333333333334
  full_minus_worst_ablation: 0.11197916666666669
  min_family_generation_exact: 0.0625
```

Per-family exact:

```text
checksum: 0.296875
modchain: 0.078125
revchain: 0.0625
```

Interpretation:

```text
The staged placement survives a larger reasoning check. This is stronger than
the short seed sweep because the exact rate rises to 0.1458 and think0 remains
0.0, while state_reset/op_zero still fall well below the full recurrent path.

This does not yet justify moving to a final general LLM claim. It does justify
using the staged placement as the current canonical reasoning core for the next
memory/long-context experiments.
```

This probe intentionally uses a tiny synthetic modular-program language:

```text
BOS START value ops... ANS -> answer EOS
```

It is not the final QTRM-native architecture. It is the smallest falsifier for:

```text
tokens -> encode -> repeated shared thinking block -> decode -> LM head
```

Acceptance requires:

```text
full_generation_exact >= threshold
full > think0
full > state_reset/op_zero
```

First smoke:

```text
command:
  PYTHONPATH=src .venv/bin/python scripts/335_train_qtrm_native_etd_probe.py \
    --out-dir local_eval/qtrm_native_etd_smoke_cpu_s20 \
    --steps 20 --train-cases 64 --eval-cases 16 \
    --program-len 2 --modulus 8 --d-model 32 --n-heads 4 --d-ff 64 \
    --batch-size 16 --device cpu --log-every 10 \
    --accept-min-exact 0.5 --accept-min-depth-gain 0.05 \
    --accept-min-ablation-drop 0.05

result:
  loss: 3.8876 -> 1.8404
  full_generation_exact: 0.0
  think0_generation_exact: 0.0625
  state_reset_generation_exact: 0.0
  op_zero_generation_exact: 0.0
  decision: rejected
```

Interpretation:

```text
The native path trains enough for loss to fall, but greedy generation collapses
to immediate EOS under the short smoke. This is not evidence against the
native-first direction; it defines the first native bottleneck: answer/EOS
decoding and on-policy generation need to be fixed before reasoning-depth
claims can be made.
```

First Qwen3.5-style hybrid runtime smoke:

```text
command:
  PYTHONPATH=src .venv/bin/python scripts/335_train_qtrm_native_etd_probe.py \
    --out-dir local_eval/qtrm_native_hybrid_3to1_smoke_cpu_s1 \
    --steps 1 --train-cases 8 --eval-cases 2 \
    --program-len 2 --modulus 8 --d-model 16 --n-heads 4 \
    --n-kv-heads 2 --d-ff 32 --backbone qtrm_hybrid_3to1 \
    --hybrid-layers 4 --attn-every 4 --train-think-steps 1 \
    --eval-think-steps 1 --batch-size 2 --device cpu

result:
  runtime: completed, report written
  decision: rejected
  full_generation_exact: 0.5
  think0_generation_exact: 0.0
  thinking_block_off_generation_exact: 0.0
  state_reset_generation_exact: 0.5
```

Interpretation:

```text
The hybrid path runs through LM logits, but it is not promoted. The strict
state-reset ablation still matches full, so the recurrent trajectory is not yet
causally proven.
```

First accepted native L1 runs after answer/EOS loss split:

```text
loss change:
  answer and EOS token losses are weighted separately
  first answer token receives an answer-vs-EOS margin
  evaluation now records answer_token_accuracy and first_token_eos_rate
```

MHA ETD strict run:

```text
out_dir: local_eval/qtrm_native_mha_answer_eos_cuda_s600_strict
thresholds:
  exact >= 0.90
  depth_gain >= 0.10
  ablation_drop >= 0.10
result:
  decision: accepted_l1_native_etd
  think4_generation_exact: 0.98046875
  think0_generation_exact: 0.16796875
  state_reset_generation_exact: 0.32421875
  op_zero_generation_exact: 0.12109375
  full_minus_think0: 0.8125
  full_minus_worst_ablation: 0.65625
  first_token_eos_rate: 0.0
```

Qwen3.5-style hybrid strict run:

```text
out_dir: local_eval/qtrm_native_hybrid_3to1_cuda_s600_strict
thresholds:
  exact >= 0.90
  depth_gain >= 0.10
  ablation_drop >= 0.10
result:
  decision: accepted_l1_native_etd
  think4_generation_exact: 1.0
  think0_generation_exact: 0.05859375
  state_reset_generation_exact: 0.59765625
  op_zero_generation_exact: 0.15625
  full_minus_think0: 0.94140625
  full_minus_worst_ablation: 0.40234375
  first_token_eos_rate: 0.0
```

Interpretation:

```text
The first native donorless LM path is now viable on a tiny algorithmic language.
The result is not a general LLM claim yet. It proves that the causal route
tokens -> native backbone -> repeated think block -> LM head can overfit and
generalize held-out synthetic cases with destructive ablations.
```

L2 ladder probes:

```text
L2A intermediate:
  program_len: 3
  modulus: 16
  out_dir: local_eval/qtrm_native_mha_l2a_program3_mod16_cuda_s2500
  decision: accepted_l1_native_etd
  threshold exact >= 0.70
  think4_generation_exact: 0.763671875
  think0_generation_exact: 0.060546875
  state_reset_generation_exact: 0.203125
  op_zero_generation_exact: 0.05859375
  full_minus_think0: 0.703125
  full_minus_worst_ablation: 0.560546875

L2B harder:
  program_len: 4
  modulus: 32
  out_dir: local_eval/qtrm_native_mha_l2_program4_mod32_cuda_s5000
  decision: rejected
  threshold exact >= 0.70
  think4_generation_exact: 0.248046875
  think0_generation_exact: 0.046875
  state_reset_generation_exact: 0.111328125
  op_zero_generation_exact: 0.021484375
  full_minus_think0: 0.201171875
  full_minus_worst_ablation: 0.13671875
```

Step-wise depth target probe:

```text
change:
  optional --depth-intermediate-loss-weight
  depth 1 predicts state after op 1
  depth 2 predicts state after op 2
  ...
  final answer still uses ordinary LM logits

out_dir: local_eval/qtrm_native_mha_l2_program4_mod32_depth_targets_cuda_s2500
program_len: 4
modulus: 32
steps: 2500
depth_intermediate_loss_weight: 0.5
decision: rejected
think4_generation_exact: 0.291015625
think0_generation_exact: 0.0
state_reset_generation_exact: 0.0234375
op_zero_generation_exact: 0.025390625
full_minus_think0: 0.291015625
full_minus_worst_ablation: 0.265625
```

Accepted L2B with curriculum + step-wise depth targets:

```text
change:
  --active-len-curriculum
  --active-len-curriculum-min 1
  --active-len-curriculum-warmup-frac 0.5
  --depth-intermediate-loss-weight 0.5

out_dir: local_eval/qtrm_native_mha_l2_program4_mod32_curriculum_depth_cuda_s5000
program_len: 4
modulus: 32
steps: 5000
decision: accepted_l1_native_etd
threshold exact >= 0.70
think4_generation_exact: 0.955078125
think0_generation_exact: 0.0
state_reset_generation_exact: 0.013671875
op_zero_generation_exact: 0.03125
full_minus_think0: 0.955078125
full_minus_worst_ablation: 0.923828125
first_token_eos_rate: 0.0
```

Interpretation:

```text
Native recurrence is causally useful beyond the tiny L1 setting. The harder
program_len=4/mod32 distribution was rejected under direct training and under
depth targets alone, but passed when active-length curriculum was combined with
step-wise depth supervision. This is the first strong L2B-style evidence that
the recurrent trajectory itself carries the multi-step value transition.
```

## Rejection Rules

Do not promote a result if:

```text
the answer comes from a symbolic side solver
only teacher-forced loss improves while greedy generation collapses
think_steps=0 matches full
state_reset or core_state_zero matches or beats full
op_zero/op_shuffle matches full on algorithmic tasks
the result depends on Qwen donor hidden states
```

## Relation To Donor Work

Donor-sidecar work remains useful for:

```text
diagnosing output collapse
studying tokenizer/output-head geometry
collecting teacher traces
comparing language preservation
building future distillation targets
```

But the canonical claim has moved:

```text
from:
  Qwen donor + QTRM sidecar

to:
  QTRM-native LM with recursive thinking in the causal token path
```

## L5E Official-TRM Dual-State Check

Question:

```text
Is the current native recurrent block actually the Samsung TRM loop, or only
TRM-inspired ETD?
```

Answer:

```text
The earlier QTRM-native ETD scaffold was not identical to Samsung TRM. It used
a single recurrent hidden state:

  encode -> h -> shared think block repeated N times -> decode -> LM head

The official TinyRecursiveModels implementation uses a dual latent carry:

  z_L, z_H
  for H_cycles:
    for L_cycles:
      z_L = shared_reasoning_module(z_L, z_H + input_embeddings)
    z_H = shared_reasoning_module(z_H, z_L)
  logits = lm_head(z_H)

The native probe now supports this as:

  --think-structure trm_dual_z
```

Implementation:

```text
script:
  scripts/335_train_qtrm_native_etd_probe.py

new args:
  --think-structure single|trm_dual_z
  --trm-l-cycles N
  --trm-full-grad-cycles

important detail:
  The think block is shared between z_L and z_H updates, matching the official
  TRM single-network recurrence pattern. It is not two separate low/high
  networks.
```

Official TRM cannot be copied byte-for-byte into the canonical general-LM path:

```text
Samsung TRM is a puzzle/ARC-style recursive reasoner with non-causal sequence
attention and task carry/halting. QTRM-native is an autoregressive LM path, so
the token path must remain causal to avoid next-token leakage.
```

Short comparison:

```text
out_root:
  local_eval/qtrm_native_l5e_trm_dual_z_compare_short_20260512

profile:
  short, 400 steps, d_model=64, eval_cases=192

candidates:
  mha_etd
  official_fla_think
  official_mamba3_think
  trm_dual_z_fla_think
  trm_dual_z_mamba3_think
```

Results:

```text
mha_etd:
  full_generation_exact: 0.015625

official_fla_think:
  full_generation_exact: 0.036458333333333336
  full_minus_think0: 0.03125
  full_minus_worst_ablation: 0.010416666666666668
  promoted: true

official_mamba3_think:
  full_generation_exact: 0.046875
  full_minus_think0: 0.046875
  full_minus_worst_ablation: 0.010416666666666664
  promoted: true

trm_dual_z_fla_think:
  full_generation_exact: 0.026041666666666668
  full_minus_think0: 0.026041666666666668
  full_minus_worst_ablation: -0.020833333333333332
  promoted: false

trm_dual_z_mamba3_think:
  full_generation_exact: 0.020833333333333332
  full_minus_think0: -0.005208333333333336
  full_minus_worst_ablation: -0.015625000000000003
  promoted: false
```

Interpretation:

```text
Do not hard-code Mamba-3 just because it won the earlier single-state probe.
GatedDeltaNet/FLA remains a real candidate: it also beat MHA and passed the
causal promotion checks in the short single-state placement.

However, the first official-TRM-style z_L/z_H graft did not promote. In both
dual-state candidates, state_reset/op_zero ablations matched or beat the full
trajectory. That means the current z_L/z_H loop is implemented and runnable,
but it is not yet causally useful under the short mixed-text reasoning gate.

Canonical status:
  current strongest validated placement remains
    MHA encode -> official Mamba-3 single recurrent think -> MHA decode

  official-TRM z_L/z_H is now an experimental candidate, not yet canonical.
```

Next TRM-specific work:

```text
1. Compare trm_dual_z with full-grad cycles vs official no-grad inner cycles.
2. Sweep L_cycles=1/2/4.
3. Add an official-TRM block option closer to post-norm Attention/SwiGLU.
4. Only promote dual-state TRM if full > think0 and full > state_reset/op_zero
   on the same LM-logit generation metric.
```

## L5F Official-TRM Block Check

The next orthodox TRM step was to stop testing only the `z_L/z_H` loop and to
also reproduce the official TRM block bias more closely.

Official TRM block pattern:

```text
hidden = rms_norm(hidden + self_attention(hidden))
hidden = rms_norm(hidden + SwiGLU(hidden))
```

QTRM-native adaptation:

```text
backbone:
  trm_official

implementation:
  NativeTRMOfficialBlock

LM constraint:
  attention remains causal in the autoregressive LM path
```

Code:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds NativeRMSNorm
  adds NativeSwiGLU
  adds NativeTRMOfficialBlock
  adds trm_official to SUPPORTED_BACKBONES

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds official_trm_think
  adds trm_dual_z_official_trm_think
  adds trm_dual_z_official_trm_l2_think
  adds trm_dual_z_official_trm_fullgrad_think
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/336_train_qtrm_native_text_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_qtrm_native_l5d_backbone_compare

result:
  55 tests OK
```

Short reasoning comparison:

```text
out_root:
  local_eval/qtrm_native_l5f_official_trm_block_compare_short_20260512

profile:
  short, 400 steps, d_model=64, eval_cases=192

full_generation_exact:
  mha_etd:                                  0.015625
  official_fla_think:                       0.036458333333333336
  official_mamba3_think:                    0.046875
  official_trm_think:                       0.057291666666666664
  trm_dual_z_official_trm_think:            0.046875
  trm_dual_z_official_trm_l2_think:         0.041666666666666664
  trm_dual_z_official_trm_fullgrad_think:   0.020833333333333332

winner:
  official_trm_think
```

Promotion status:

```text
official_trm_think:
  promoted: true
  full_minus_think0: 0.031249999999999997
  full_minus_worst_ablation: 0.02083333333333333

trm_dual_z_official_trm_think:
  promoted: true
  full_minus_think0: 0.03125
  full_minus_worst_ablation: 0.015625

trm_dual_z_official_trm_l2_think:
  promoted: true
  full_minus_think0: 0.010416666666666664
  full_minus_worst_ablation: 0.005208333333333329

trm_dual_z_official_trm_fullgrad_think:
  promoted: false
  full_minus_think0: -0.015625000000000003
  full_minus_worst_ablation: -0.03125
```

Language non-regression:

```text
out_dir:
  local_eval/qtrm_native_l5f_official_trm_language_nonregression_short_20260512

decision:
  accepted_l5f_official_trm_language_nonregression

metrics:
  think_eval_loss: 0.04991341052068905
  think0_loss: 2.7112910639155996
  thinking_block_off_loss: 2.7112910639155996
  think0_baseline_loss: 0.05648966429924423
  full_vs_think0: 0.018409462261349752
  full_vs_baseline: 0.883584831665513
  sample_unique_chars: 30
  sample_max_run_fraction: 0.015151515151515152
```

Interpretation:

```text
The first dual-z failure was not evidence against TRM as a family. It was
evidence that z_L/z_H alone, grafted onto the earlier block, was not enough.

Adding the official-TRM-style post-norm RMSNorm/SwiGLU block changed the result:

1. official_trm_think beat MHA, FLA-think, and Mamba3-think in the short
   reasoning comparison;
2. trm_dual_z_official_trm_think now passed causal promotion checks;
3. full-grad inner cycles were worse than the official-style no-grad inner
   cycle default on this short gate;
4. official_trm_think also passed a short language non-regression gate.

Canonical update:
  promote official_trm_think to the current strongest short-gate candidate.

Not yet final:
  dual-z official TRM is now viable, but it did not beat single official_trm
  on this gate. It remains a serious candidate for longer/deeper recursive
  tasks, not the default canonical placement yet.
```

## L5G TRM-Shell Mixer Comparison

Question:

```text
Can the official-TRM shell be strengthened by replacing or extending its
attention mixer with Mamba3, GatedDeltaNet, Qwen3.5-style 3:1
GatedDelta/Attention, or a three-way GatedDelta/Mamba3/Attention mixer?
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds NativeTRMMixerLayer
  adds NativeTRMMixerBlock
  adds NativeTRMMamba3Block
  adds NativeTRMGatedDeltaBlock
  adds NativeTRMQwen35HybridBlock
  adds NativeTRMTriMixerBlock

new supported backbones:
  trm_mamba3
  trm_gated_delta
  trm_qwen35_3to1
  trm_tri_mixer

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds trm_dual_z_trm_mamba3_think
  adds trm_dual_z_trm_gated_delta_think
  adds trm_dual_z_trm_qwen35_3to1_think
  adds trm_dual_z_trm_tri_mixer_think
```

Architecture rule:

```text
Do not parallel-fuse the mixers.

Use the same causal hidden stream:

tokens -> MHA encode -> mandatory TRM dual-z recurrence
  -> sequential TRM-shell mixer stack
  -> MHA decode -> LM logits

trm_tri_mixer schedule:
  GatedDelta -> Mamba3 -> GatedDelta -> Attention
```

Reason:

```text
Sequential mixing preserves the universal LM causal path. Each mixer updates
the same latent stream before LM-logit generation. Parallel fusion would make
the ablation story harder and can hide which path actually carries the answer.
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/336_train_qtrm_native_text_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py \
  src/qtrm_mm/mixers.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_qtrm_native_l5d_backbone_compare

result:
  63 tests OK
```

Smoke backend check:

```text
out_root:
  local_eval/qtrm_native_l5g_trm_shell_mixer_compare_smoke_20260512_rerun

result:
  all new candidates ran
  official Mamba3 mixer count is detected
  official FLA GatedDelta mixer count is detected
  torch_delta_mixers is 0 under strict official backend candidates
```

Short reasoning comparison:

```text
out_root:
  local_eval/qtrm_native_l5g_trm_shell_mixer_compare_short_20260512

profile:
  short, 400 steps, d_model=64, eval_cases=192

full_generation_exact:
  mha_etd:                            0.015625
  official_trm_think:                 0.057291666666666664
  trm_dual_z_official_trm_think:      0.046875
  trm_dual_z_trm_mamba3_think:        0.026041666666666668
  trm_dual_z_trm_gated_delta_think:   0.052083333333333336
  trm_dual_z_trm_qwen35_3to1_think:   0.026041666666666668
  trm_dual_z_trm_tri_mixer_think:     0.026041666666666668

winner:
  official_trm_think
```

Promotion status:

```text
official_trm_think:
  promoted: true
  full_minus_think0: 0.031249999999999997
  full_minus_worst_ablation: 0.02083333333333333

trm_dual_z_official_trm_think:
  promoted: true
  full_minus_think0: 0.03125
  full_minus_worst_ablation: 0.015625

trm_dual_z_trm_mamba3_think:
  promoted: true
  full_minus_think0: 0.026041666666666668
  full_minus_worst_ablation: 0.010416666666666668

trm_dual_z_trm_gated_delta_think:
  promoted: true
  full_minus_think0: 0.04166666666666667
  full_minus_worst_ablation: 0.005208333333333336

trm_dual_z_trm_qwen35_3to1_think:
  promoted: true
  full_minus_think0: 0.010416666666666668
  full_minus_worst_ablation: 0.005208333333333336

trm_dual_z_trm_tri_mixer_think:
  promoted: false
  full_minus_think0: 0.026041666666666668
  full_minus_worst_ablation: -0.010416666666666668
```

Interpretation:

```text
Mixing Mamba3, GatedDeltaNet, and Attention in one TRM-shell block is a valid
runtime candidate, but it is not the current canonical architecture. It fails
the causal ablation promotion rule because state_reset/op_zero can outperform
the full recurrent path on this short gate.

GatedDelta-only inside the TRM shell is the strongest new candidate:
  exact 0.052083333333333336

But it still does not beat:
  official_trm_think exact 0.057291666666666664

Canonical update:
  keep official_trm_think as the default short-gate QTRM-native reasoning
  core.

Candidate queue:
  1. official_trm_think remains canonical.
  2. trm_dual_z_trm_gated_delta_think is the best strict FLA follow-up.
  3. trm_tri_mixer is diagnostic only until it passes a positive ablation drop.
```

## L5H ELF-Inspired Latent Readout Candidate

Reference:

```text
paper:
  ELF: Embedded Language Flows

arxiv:
  https://arxiv.org/abs/2605.10938

submitted:
  2026-05-11

official implementation:
  https://github.com/lillian039/ELF

authors include:
  Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li,
  Yoon Kim, Jacob Andreas, Kaiming He
```

What ELF claims:

```text
ELF is a continuous diffusion / flow-matching language model that keeps
generation in continuous embedding space until the final step. Only the final
step maps embeddings back to discrete tokens through a shared-weight network.

This differs from discrete diffusion language models that operate primarily
over token states.
```

Core idea:

```text
noise embedding
-> continuous-time flow matching / denoising trajectory
-> clean language embedding
-> final discrete-token projection
```

Why it matters for QTRM:

```text
Our current QTRM-native bottleneck is not only the recursive core. It is also
how the final latent trajectory is converted into stable LM tokens.

TRM-style recurrence gives a latent trajectory:

  z_L / z_H over think steps

The current readout is simple:

  final latent -> LM head -> greedy autoregressive token

ELF suggests a stronger readout/training target:

  noisy or partial latent -> refined answer embedding trajectory
  -> shared token projection only at the end
```

Important boundary:

```text
ELF is not a replacement for the TRM/QTRM reasoning core.

It should not replace:
  token input
  mandatory recursive latent core
  causal LM path

It is a candidate for:
  latent transition supervision
  latent readout stabilization
  final token projection from a refined embedding
```

Diffusion boundary:

```text
Do not turn QTRM-native into a diffusion reasoning model.

Rejected as canonical core:
  noisy latent
  -> diffusion sampler / flow sampler
  -> answer tokens

Reason:
  QTRM-native's canonical claim is recursive latent reasoning through a normal
  LM path. The core must remain deterministic/recurrent enough that think depth,
  state reset, and core-off ablations can show causal reasoning gains.

Allowed as auxiliary:
  ELF-style continuous embedding refinement loss
  latent MTP over future answer-token embeddings
  shared-LM-head final projection
```

Potential QTRM-native adaptation:

```text
tokens
-> encoder
-> mandatory TRM dual-z recursive core
-> z_H / z_L trajectory
-> ELF-inspired latent refinement readout
-> shared LM head / token projection
-> autoregressive text
```

Training sketch:

```text
1. Keep the canonical TRM core.
2. Collect latent states from multiple think depths.
3. Add an ELF-inspired embedding refinement auxiliary:
     early/partial latent -> target answer-token embedding(s)
4. Add latent MTP:
     from the same prefix latent, predict multiple future answer-token
     embeddings through the shared LM head.
5. Use a shared LM projection only at the final readout.
6. Preserve normal LM CE so the model remains a general LM path.
7. Require causal ablations:
   core_off, think0, state_reset, op_zero, readout_off.
```

Why this may help:

```text
1. It gives the recurrent core a smoother target than immediate discrete token
   accuracy.
2. It may reduce repeated-token collapse because the model first refines a
   continuous answer embedding.
3. It gives think steps a natural interpretation:
     early latent = noisy / under-refined
     later latent = cleaner / answer-ready
4. It directly targets the current readout bottleneck rather than only swapping
   MHA, Mamba3, or GatedDelta mixers.
```

Risks:

```text
1. ELF is not an autoregressive transformer LM; naive adoption could leave the
   canonical LM path.
2. Sampling may require multiple flow steps, increasing inference cost.
3. The official implementation is JAX/TPU first; PyTorch support is not yet
   the main path.
4. A flow readout can become a side decoder unless it is kept inside the same
   LM-logit causal path and passes ablation checks.
```

Canonical status:

```text
Not canonical yet.

ELF is promoted to the candidate queue as an L5H latent-readout research
direction, not as a replacement for official_trm_think.
```

Candidate queue update:

```text
1. Keep official_trm_think as the current canonical short-gate reasoning core.
2. Keep trm_dual_z_trm_gated_delta_think as the best strict FLA follow-up.
3. Treat trm_tri_mixer as diagnostic only.
4. Add ELF-inspired latent refinement readout as the next readout-side
   architecture candidate.
```

Immediate next step:

```text
Implement L5H as an auxiliary-only experiment, not a core replacement.

Target experiment:
  official_trm_think
  + ELF-style latent refinement auxiliary
  + latent MTP over future answer-token embeddings
  + shared LM head

Acceptance:
  full_generation_exact improves over official_trm_think or at least over
  trm_dual_z_official_trm_think on the same short gate;
  full_minus_think0 > 0;
  full_minus_worst_ablation > 0;
  readout_off reduces the gain;
  normal greedy autoregressive output remains the evaluated path.

Reject if:
  a side flow decoder computes the answer;
  only teacher-forced embedding loss improves;
  generation exact does not improve;
  ablations show the recursive core is not causally needed.
```

## L5H.1 Latent Diffusion Reasoning Prior Check

Question:

```text
Is the current QTRM-native failure partly caused by strictly autoregressive
commitment, where an early wrong token/readout decision cannot be revised?
```

Prior evidence:

```text
Latent Diffusion for Language Generation
  arxiv: https://arxiv.org/abs/2212.09462
  lesson:
    A language autoencoder can map text to a continuous latent space, run
    diffusion in that latent space, and decode back to language. Diffusion and
    pretrained language models can be complementary rather than exclusive.

Diffusion Forcing
  arxiv: https://arxiv.org/abs/2407.01392
  lesson:
    Per-token noise levels can combine causal next-token prediction with
    full-sequence diffusion guidance. This is relevant because QTRM wants
    causal generation but also needs trajectory-level correction.

Diffusion of Thoughts
  arxiv: https://arxiv.org/abs/2402.07754
  lesson:
    Reasoning steps can be diffused over time instead of committed strictly
    left-to-right, giving a compute/quality knob and self-correction behavior.

LaDiR: Latent Diffusion Reasoner
  arxiv: https://arxiv.org/abs/2510.04573
  lesson:
    Existing LLM reasoning can be encoded into latent thought blocks, denoised
    with blockwise bidirectional attention, and decoded into diverse reasoning
    trajectories. This is the closest prior to a QTRM latent-thought refiner.

LLaDA / Large Language Diffusion Models
  arxiv: https://arxiv.org/abs/2502.09992
  lesson:
    A diffusion LM can scale and support instruction behavior, so core LLM
    abilities are not necessarily tied to autoregressive models.

Masked Diffusion Language Models / SEDD / Block Diffusion
  arxiv:
    https://arxiv.org/abs/2406.07524
    https://arxiv.org/abs/2310.16834
    https://arxiv.org/abs/2503.09573
  lesson:
    Discrete/masked/block diffusion narrows the gap to AR LMs, supports
    infilling and partial revision, and can interpolate between AR and full
    diffusion.

Latent Refinement Decoding
  arxiv: https://arxiv.org/abs/2510.11052
  lesson:
    Premature commitment and discarded uncertainty are explicit failure modes.
    Keeping distributional belief states and finalizing only confident tokens
    can improve reasoning/coding while preserving early stopping.

Continuous Latent Diffusion Language Model / Cola DLM
  arxiv: https://arxiv.org/abs/2605.06548
  lesson:
    A newer latent diffusion route separates global semantic organization from
    local text realization through Text-VAE latent mapping, latent prior
    modeling, and conditional decoding.

DiffCoT
  arxiv: https://arxiv.org/abs/2601.03559
  lesson:
    AR CoT suffers from exposure bias and irreversible early mistakes. A
    diffusion-styled CoT process can retrospectively correct intermediate
    reasoning while preserving token-level autoregression.
```

Implication for QTRM-native:

```text
The prior strongly supports the suspicion that a pure greedy AR readout can be
too brittle for multi-step reasoning. However, the prior does not require
abandoning TRM/QTRM.

Preferred adaptation:
  prompt tokens
  -> native encoder
  -> mandatory dual-z TRM/QTRM recursive core
  -> latent thought/readout states over think depth
  -> diffusion-style latent refinement auxiliary
  -> shared LM head
  -> normal autoregressive answer tokens

Rejected shortcut:
  prompt
  -> separate diffusion decoder
  -> answer

Reason:
  That would make the diffusion sampler the real reasoner and would no longer
  prove QTRM-native recursive core intelligence.
```

Candidate experiment:

```text
L5H-latent-refine:
  base:
    accepted MHA ETD L6 floor and/or official_trm_think
  add:
    latent_noise_level embedding over intermediate core states
    denoise head that predicts clean answer-token embeddings
    optional belief-state KL between early and late readout distributions
    confidence/finalize loss inspired by Latent Refinement Decoding
  keep:
    final metric is greedy autoregressive generation_exact
    final projection uses shared LM head
    core ablations remain mandatory

Accept only if:
  generation_exact improves, not only latent loss;
  think0/state_reset/op_zero/readout_refine_off reduce the gain;
  no side decoder or solver computes the final answer.
```

Implementation status:

```text
implemented:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new CLI:
  --latent-refine-loss-weight
  --latent-refine-min-depth
  --latent-refine-noise-std
  --latent-refine-depth-weight-power
  --latent-refine-final-kl-weight

mechanism:
  model.forward_with_runtime(..., return_state_trace=True)
  -> core_state_trace_h depth states
  -> optional depth-scaled Gaussian noise
  -> existing model.decode stack
  -> existing model.norm + shared lm_head
  -> answer_text_loss on normal answer token positions
  -> optional KL to final canonical logits

boundary:
  No separate diffusion decoder, solver, renderer, or answer side channel was
  added. The auxiliary only teaches intermediate latent states to become
  answer-ready through the existing LM path.

verification:
  unit tests cover finite gradient loss, missing trace rejection, and parser
  options. A 1-step CPU smoke confirms the training loop accepts the new
  auxiliary and writes a normal report/checkpoint.

status:
  implemented as an auxiliary experiment; not promoted. Promotion requires a
  real same-budget comparison against the accepted L6 floor and/or
  official_trm_think with core/readout ablations.
```

### L5H.2 Diffusive-TRM v1

Goal:

```text
Turn the TRM update itself into a diffusion-style latent refinement loop without
leaving the canonical LM path.
```

Implemented structure:

```text
think_structure:
  trm_dual_z_diffusive

path:
  prompt tokens
  -> token embeddings
  -> native encoder
  -> z_L / z_H TRM state
  -> denoise-time-conditioned TRM update
  -> native decoder
  -> shared LM head
  -> autoregressive answer text
```

Core mechanism:

```text
For each recurrent H step:
  progress = denoise progress in [0, 1]
  noise_level = 1 - progress
  time_features = [progress, noise_level]
  time_state = MLP(time_features)
  step_state = learned TRM step embedding
  diffusion_context = step_state + time_state

  z_L update:
    context = encoded + z_H + diffusion_context
    candidate_L = TRMBlock(norm(z_L + context))
    z_L = gated_update(z_L, candidate_L, context)

  z_H update:
    context = encoded + z_L + diffusion_context
    candidate_H = TRMBlock(norm(z_H + context))
    z_H = gated_update(z_H, candidate_H, context)
```

What this is:

```text
Diffusion-style time/noise conditioning inside the mandatory recursive TRM core.
It gives every think step an explicit "noisy -> cleaner" role and pairs with
the L5H latent-refine auxiliary.
```

What this is not:

```text
It is not a separate diffusion decoder.
It does not sample answer tokens through a flow/diffusion sampler.
It does not bypass decoder -> shared LM head -> autoregressive generation.
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds trm_dual_z_diffusive
  adds trm_diffusion_time_mlp
  adds trm_diffusion_input_norm
  adds z_L/z_H diffusion update gates

scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  accepts --think-structure trm_dual_z_diffusive
  can combine it with --latent-refine-loss-weight
```

Verification:

```text
unit:
  tests.test_qtrm_native_etd_probe
  tests.test_qtrm_native_mixed_text_reasoning_probe
  130 tests passing

smoke:
  local_eval/qtrm_native_diffusive_trm_smoke_20260513/report.json
  threshold-zero CPU smoke only; not a performance promotion.

triage comparison:
  local_eval/qtrm_native_diffusive_vs_official_len6_cpu_triage_20260513/summary.json

  setup:
    CPU only because local CUDA failed with driver/library version mismatch.
    len6, d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64,
    steps=80. This is not a promotion run.

  result:
    official:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.015625
    diffusive_trm + latent_refine:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0625
      full_minus_think0: -0.015625
      full_minus_worst_ablation: -0.015625

  interpretation:
    Diffusive-TRM did not improve exact accuracy in this triage and failed the
    causal core-gain check because think0/op_zero were not worse than the full
    recurrent path. Keep it experimental; do not promote until a same-budget
    GPU run shows positive depth gain and positive ablation drop.
```

v2 follow-up:

```text
Question:
  Are Mamba3 / GatedDeltaNet AR-only blocks that cannot be used in
  Diffusive-TRM?

Answer:
  No. Mamba3 and GatedDeltaNet are causal recurrent/linear sequence mixers.
  They are commonly used in autoregressive LMs, but the mixer itself is not
  restricted to final AR decoding. In QTRM-native Diffusive-TRM they can be
  used as the latent denoising/update mixer inside:

    z_t + prompt-conditioned state + denoise-time embedding -> z_{t-1}

  The final answer path must still stay canonical:

    prompt tokens -> native encoder -> mandatory recursive core
    -> native decoder -> shared LM head -> AR text

Implementation update:
  scripts/335_train_qtrm_native_etd_probe.py
    trm_dual_z_diffusive now initializes z_L/z_H from encoded prompt state via
    trm_init_l_proj / trm_init_h_proj. This prevents the diffusion loop from
    starting as an almost prompt-detached learned constant state.

  scripts/346_run_dual_trm_3to1_length_gate.sh
    adds:
      diffusive_gated_delta = trm_dual_z_diffusive + trm_gated_delta
      diffusive_mamba3      = trm_dual_z_diffusive + trm_mamba3

Verification:
  unit:
    PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
      tests.test_qtrm_native_etd_probe \
      tests.test_qtrm_native_mixed_text_reasoning_probe
    130 tests passing

  static:
    bash -n scripts/346_run_dual_trm_3to1_length_gate.sh

v2 CPU triage:
  local_eval/qtrm_native_diffusive_v2_vs_official_len6_cpu_triage_20260513/summary.json

  setup:
    CPU only because local CUDA is unavailable.
    len6, d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64,
    steps=80. This is still triage, not a promotion run.

  results:
    official:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.015625

    diffusive_trm:
      full_generation_exact: 0.015625
      think0_generation_exact: 0.015625
      full_minus_think0: 0.0
      full_minus_worst_ablation: 0.015625

    diffusive_gated_delta:
      full_generation_exact: 0.0625
      think0_generation_exact: 0.015625
      full_minus_think0: 0.046875
      full_minus_worst_ablation: -0.03125

    diffusive_mamba3:
      path: local_eval/qtrm_native_diffusive_mamba3_len6_cpu_triage_20260513/summary.json
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.0

Decision:
  diffusive_gated_delta is interesting because it improved short-run exact
  accuracy over official on this tiny CPU triage. It is not promoted because
  the worst ablation beat the full model, so the improvement is not yet
  causally attributable to the intended recursive latent path.

  diffusive_mamba3 matched official exact and had positive depth gain, but it
  is also not promoted because z_l_zero matched the full model, leaving no
  strict ablation drop.

  official_trm remains the cleaner canonical baseline for now because its
  causal ablation semantics are still less suspicious than the diffusive
  variants in these CPU triage runs.

Next repair target:
  If Diffusive-TRM remains worth pursuing, repair diffusive_gated_delta so
  state_reset, z_L/z_H zeroing, and op/backbone ablations become worse than
  the full recurrent path. Do not count raw exact alone as architecture
  success.

Primary prior anchors:
  Mamba-3: https://arxiv.org/abs/2603.15569
  Gated Delta Networks: https://arxiv.org/abs/2412.06464
```

v3 reversed hybrid 3:1:

```text
Question:
  What if z_L uses Mamba3 while z_H uses GatedDeltaNet, and both sides use a
  3:1 mixer-to-attention sync pattern inside the diffusion loop?

Implemented structure:
  think_structure:
    trm_dual_z_diffusive_reversed_hybrid_3to1

  z_L update block:
    Mamba3 -> Mamba3 -> Mamba3 -> causal attention

  z_H update block:
    GatedDeltaNet -> GatedDeltaNet -> GatedDeltaNet -> causal attention

  diffusion path:
    prompt tokens
    -> native encoder
    -> prompt-conditioned z_L / z_H
    -> denoise-time-conditioned z_L Mamba3 trajectory update
    -> denoise-time-conditioned z_H GatedDelta correction update
    -> native decoder
    -> shared LM head
    -> AR answer text

Interpretation:
  z_L is treated as the recurrent trajectory/state-flow carrier.
  z_H is treated as the high-level correction/update carrier.
  The attention layer in each 3:1 block is a periodic causal grounding sync,
  not a separate answer channel.

Implementation:
  scripts/335_train_qtrm_native_etd_probe.py
    NativeTRMMamba3AttentionHybridBlock:
      ("mamba3", "mamba3", "mamba3", "attention")

    NativeTRMGatedDeltaAttentionHybridBlock:
      ("gated_delta", "gated_delta", "gated_delta", "attention")

    _run_trm_diffusive_reversed_hybrid_h_cycle:
      uses the Mamba3-heavy block for z_L and GatedDelta-heavy block for z_H.

  scripts/346_run_dual_trm_3to1_length_gate.sh
    adds candidate:
      diffusive_reversed_hybrid_3to1

Verification:
  red/green unit tests:
    test_trm_dual_z_diffusive_reversed_hybrid_uses_mamba_l_delta_h_attention_sync
    test_parser_accepts_diffusive_reversed_hybrid_3to1_thinking_structure

CPU triage:
  local_eval/qtrm_native_diffusive_reversed_hybrid_3to1_len6_cpu_triage_20260513/summary.json

  setup:
    CPU only because local CUDA is unavailable.
    len6, d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64,
    steps=80. This is still triage, not a promotion run.

  results:
    official:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.015625

    diffusive_gated_delta:
      full_generation_exact: 0.0625
      think0_generation_exact: 0.015625
      full_minus_think0: 0.046875
      full_minus_worst_ablation: -0.03125

    diffusive_mamba3:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.0

    diffusive_reversed_hybrid_3to1:
      full_generation_exact: 0.078125
      think0_generation_exact: 0.03125
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.03125

Decision:
  This is the strongest Diffusive-TRM candidate so far on CPU triage:
    higher exact than official and single-mixer diffusive candidates;
    positive depth gain;
    positive worst-ablation drop;
    z_L and z_H zero ablations both reduce the full result.

  Do not call it canonical yet. Promote only after a GPU same-budget
  standard run and at least one seed sweep confirm the gain.
```

v4 non-diffusive reversed hybrid 3:1 control:

```text
Question:
  Is the reversed 3:1 mixer split itself enough, or did the v3 gain require
  diffusion time/noise conditioning and latent-refine training?

Implemented control:
  think_structure:
    trm_dual_z_reversed_hybrid_3to1

  z_L update block:
    Mamba3 -> Mamba3 -> Mamba3 -> causal attention

  z_H update block:
    GatedDeltaNet -> GatedDeltaNet -> GatedDeltaNet -> causal attention

  non-diffusion path:
    prompt tokens
    -> native encoder
    -> prompt-conditioned z_L / z_H
    -> recurrent z_L Mamba3 trajectory update
    -> recurrent z_H GatedDelta correction update
    -> native decoder
    -> shared LM head
    -> AR answer text

  Difference from v3:
    no trm_diffusion_time_mlp
    no denoise progress/noise embedding
    no default latent_refine auxiliary in the runner candidate

Verification:
  red/green unit tests:
    test_trm_dual_z_reversed_hybrid_uses_mamba_l_delta_h_without_diffusion_time
    test_parser_accepts_reversed_hybrid_3to1_thinking_structure

CPU triage:
  local_eval/qtrm_native_reversed_vs_diffusive_reversed_len6_cpu_triage_20260513/summary.json

  setup:
    CPU only because local CUDA is unavailable.
    len6, d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64,
    steps=80.

  results:
    official:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.015625

    reversed_hybrid_3to1:
      full_generation_exact: 0.015625
      think0_generation_exact: 0.0
      full_minus_think0: 0.015625
      full_minus_worst_ablation: -0.03125

    diffusive_reversed_hybrid_3to1:
      full_generation_exact: 0.078125
      think0_generation_exact: 0.03125
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.03125

Decision:
  The reversed 3:1 mixer split alone is not sufficient in this CPU triage.
  It underperformed official and failed the causal ablation check because
  op_zero/z_l_zero were better than the full model.

  The positive v3 result appears to depend on the diffusion-style step
  conditioning and/or latent-refine auxiliary, not merely on adding Mamba3,
  GatedDeltaNet, and attention in the reversed layout.
```

v5 length-scaling check for diffusive reversed hybrid 3:1:

```text
Question:
  Does the len6 gain survive when the program length is raised to len8?

Run:
  local_eval/qtrm_native_diffusive_reversed_hybrid_len6_8_cpu_triage_20260513/summary.json

  setup:
    CPU smoke/triage only because local CUDA is unavailable.
    candidates: official, diffusive_reversed_hybrid_3to1
    lengths: 6, 8
    d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64, steps=80

  len6:
    official:
      full_generation_exact: 0.046875
      think0_generation_exact: 0.0
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.015625

    diffusive_reversed_hybrid_3to1:
      full_generation_exact: 0.078125
      think0_generation_exact: 0.03125
      full_minus_think0: 0.046875
      full_minus_worst_ablation: 0.03125

  len8:
    official:
      full_generation_exact: 0.03125
      think0_generation_exact: 0.0
      full_minus_think0: 0.03125
      full_minus_worst_ablation: -0.015625

    diffusive_reversed_hybrid_3to1:
      full_generation_exact: 0.03125
      think0_generation_exact: 0.0
      full_minus_think0: 0.03125
      full_minus_worst_ablation: 0.0

Decision:
  Not robust across len6 -> len8 yet.

  The diffusive reversed hybrid is better than official at len6 in this smoke
  setting, but at len8 it ties official on exact accuracy and loses the
  positive worst-ablation margin. This means the result is an interesting
  len6 architecture candidate, not a length-robust canonical architecture.

  The next valid promotion test is a GPU standard run with longer training,
  seed sweep, and strict causal ablations. Do not claim length robustness from
  the current CPU smoke.
```

v6 joint-readout fix for len8 causal robustness:

```text
Failure addressed:
  In v5, diffusive_reversed_hybrid_3to1 did not remain robust at len8:
    full_generation_exact: 0.03125
    full_minus_worst_ablation: 0.0

  The failure signal was architectural, not just a training-length issue.
  z_L was designed as the trajectory carrier, but the final recurrent state
  handed to the decoder was z_H only. This let the model partially route around
  z_L at longer lengths.

Implemented candidate:
  think_structure:
    trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout

  candidate alias in scripts/346:
    diffusive_reversed_hybrid_3to1_joint_readout

  recurrent update:
    same as v3/v5:
      z_L: Mamba3 -> Mamba3 -> Mamba3 -> causal attention
      z_H: GatedDeltaNet -> GatedDeltaNet -> GatedDeltaNet -> causal attention
      diffusion time/noise conditioning
      latent_refine auxiliary in the runner

  final readout:
    old:
      decoder input = z_H

    new:
      decoder input = LayerNorm(z_H + Linear([encoded, z_L, z_H]))

  This keeps the answer path canonical:
    prompt tokens
    -> token embeddings
    -> native encoder
    -> recurrent z_L/z_H core
    -> joint core-dependent decoder state
    -> LM head
    -> AR answer text

CPU len8 triage:
  local_eval/qtrm_native_diffusive_reversed_hybrid_joint_readout_len8_cpu_triage_20260513/summary.json

  setup:
    CPU smoke/triage only because local CUDA is unavailable.
    len8, d_model=24, d_ff=48, batch=8, train_cases=256, eval_cases=64,
    steps=80.

  previous len8 diffusive_reversed_hybrid_3to1:
    full_generation_exact: 0.03125
    think0_generation_exact: 0.0
    full_minus_think0: 0.03125
    full_minus_worst_ablation: 0.0

  joint-readout len8:
    full_generation_exact: 0.09375
    think0_generation_exact: 0.0
    full_minus_think0: 0.09375
    full_minus_worst_ablation: 0.03125
    min_family_generation_exact: 0.047619047619047616
    state_reset_generation_exact: 0.0
    op_zero_generation_exact: 0.03125
    z_l_zero_generation_exact: 0.0625
    z_h_zero_generation_exact: 0.0
    generation_format_valid: 0.828125

Decision:
  CPU smoke suggested that len8 needed architecture repair. Joint readout
  restored a positive depth gain and positive causal ablation margin in that
  tiny setting.

  Do not promote yet:
    generation_format_valid regressed to 0.828125;
    active_len 5 and 6 remain 0.0 in this tiny run;
    this is one CPU seed and not a GPU standard run.

  Next promotion test:
    run len6/len8 same-budget comparison with:
      official
      diffusive_reversed_hybrid_3to1
      diffusive_reversed_hybrid_3to1_joint_readout
    require:
      exact improvement over both baselines;
      full_minus_worst_ablation > 0;
      format_valid near 1.0;
      no dead active-length buckets.
```

v6 local CUDA reproduction after driver reboot:

```text
Artifact:
  local_eval/qtrm_native_local_joint_readout_len8_s200_20260513/summary.json

Setup:
  local RTX 4090, len8, d_model=64, d_ff=128, batch=32,
  train_cases=1024, eval_cases=128, steps=200.

Results:
  official:
    full_generation_exact: 0.0625
    think0_generation_exact: 0.03125
    full_minus_think0: 0.03125
    full_minus_worst_ablation: 0.0234375

  diffusive_reversed_hybrid_3to1:
    full_generation_exact: 0.015625
    think0_generation_exact: 0.0390625
    full_minus_think0: -0.0234375
    full_minus_worst_ablation: -0.0703125

  diffusive_reversed_hybrid_3to1_joint_readout:
    full_generation_exact: 0.015625
    think0_generation_exact: 0.03125
    full_minus_think0: -0.015625
    full_minus_worst_ablation: -0.0390625

Conclusion:
  The CPU joint-readout improvement did not reproduce under the local CUDA
  len8 s200 gate. `official` remains the canonical len8 short-run baseline.
  The diffusive and joint-readout variants are demoted to diagnostic ideas.

  Likely bottleneck:
    diffusion-style latent-refine pressure and/or hybrid mixer dynamics harm
    the TRM depth path at this training budget. Do not keep architecture
    shopping around Mamba3/GatedDelta inside the TRM core until official TRM
    length curriculum and readout-only ablations are exhausted.

Follow-up no-latent-refine ablation:
  local_eval/qtrm_native_local_joint_readout_no_latent_refine_len8_s200_20260513/summary.json

  setup:
    same local RTX 4090 len8 s200 gate as above, but with:
      DIFFUSIVE_LATENT_REFINE_LOSS_WEIGHT=0
      DIFFUSIVE_LATENT_REFINE_NOISE_STD=0
      DIFFUSIVE_LATENT_REFINE_FINAL_KL_WEIGHT=0

  result:
    diffusive_reversed_hybrid_3to1_joint_readout:
      full_generation_exact: 0.015625
      think0_generation_exact: 0.0234375
      full_minus_think0: -0.0078125
      full_minus_worst_ablation: -0.046875
      min_family_generation_exact: 0.0
      decision: rejected_length_gate

  conclusion:
    Turning off latent-refine loss does not rescue joint_readout. The problem is
    not only the diffusion auxiliary. The hybrid diffusive/joint readout path
    itself is currently weaker than official TRM for len8.

Evaluation hygiene fix:
  scripts/346_run_dual_trm_3to1_length_gate.sh now writes a wrapper-level strict
  length-gate decision into summary.json. This avoids trusting the underlying
  train script's loose `accepted=true` when acceptance thresholds are disabled
  for exploratory sweeps.

  strict accepted requires:
    full_generation_exact > 0
    full_minus_think0 > 0
    full_minus_worst_ablation > 0
    min_family_generation_exact > 0

Official TRM len8 short baseline after strict wrapper:
  local_eval/qtrm_native_local_official_len8_short_20260513/summary.json

  setup:
    local RTX 4090, official TRM, len8, d_model=64, d_ff=128,
    train_cases=4096, eval_cases=192, steps=800.

  result:
    full_generation_exact: 0.036458333333333336
    think0_generation_exact: 0.026041666666666668
    full_minus_think0: 0.010416666666666668
    full_minus_worst_ablation: 0.0
    min_family_generation_exact: 0.015625
    active_len:
      len2: 0.03333333333333333
      len3: 0.037037037037037035
      len4: 0.07407407407407407
      len5: 0.07407407407407407
      len6: 0.037037037037037035
      len7: 0.0
      len8: 0.0
    decision: rejected_length_gate

  conclusion:
    Official TRM remains the best local len8 candidate, but more steps alone did
    not solve length robustness. The decisive failure is causal: `op_zero` ties
    full exact, so the recurrent core is not yet forced to depend on operation
    state strongly enough at len8. The next work should target operation/state
    transition causality, not another mixer swap.

Operation/state causality probes after len8 failure:
  Goal:
    Break the `op_zero == full` tie without leaving the canonical LM path:
      prompt tokens -> recurrent TRM core -> LM logits -> AR answer.

  Implemented training signal:
    operation_counterfactual_loss in
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

    Mechanism:
      chosen = logprob(gold answer | original ops prompt)
      counterfactual = logprob(same gold answer | zero-op prompt)
      loss = relu(margin - (chosen - counterfactual))

    This is a training-only counterfactual contrast. It does not add a hidden
    answer channel, solver, renderer, or non-LM inference path.

  Tests:
    tests/test_qtrm_native_mixed_text_reasoning_probe.py
      test_operation_counterfactual_loss_uses_zero_ops_prompt_with_gold_answer
      test_parser_accepts_active_len_halt_target_and_eval_cycle

  Failed existing probes:
    local_eval/qtrm_native_official_len8_core_step_codec_s800_20260513/report.json
      full_generation_exact: 0.015625
      full_minus_think0: -0.010416666666666668
      full_minus_worst_ablation: -0.03125
      op_zero_generation_exact: 0.046875
      conclusion: core_step_codec worsened both accuracy and causal margin.

    local_eval/qtrm_native_official_len8_state_trace_anticollapse_s800_20260513/report.json
      full_generation_exact: 0.015625
      full_minus_think0: -0.010416666666666668
      full_minus_worst_ablation: -0.026041666666666664
      op_zero_generation_exact: 0.041666666666666664
      conclusion: anti-collapse did not fix the operation-use failure.

  Counterfactual loss sweep:
    baseline:
      local_eval/qtrm_native_local_official_len8_short_20260513/len8_official/report.json
      full_generation_exact: 0.036458333333333336
      full_minus_think0: 0.010416666666666668
      full_minus_worst_ablation: 0.0
      op_zero_generation_exact: 0.036458333333333336

    weight 0.05:
      local_eval/qtrm_native_official_len8_op_counterfactual_w005_s800_20260513/report.json
      full_generation_exact: 0.036458333333333336
      full_minus_think0: 0.010416666666666668
      full_minus_worst_ablation: 0.0
      op_zero_generation_exact: 0.036458333333333336
      conclusion: preserves baseline but does not make op use causal.

    weight 0.075:
      local_eval/qtrm_native_official_len8_op_counterfactual_w0075_s800_20260513/report.json
      full_generation_exact: 0.036458333333333336
      full_minus_think0: 0.010416666666666668
      full_minus_worst_ablation: 0.0
      op_zero_generation_exact: 0.036458333333333336
      conclusion: still too weak.

    weight 0.1:
      local_eval/qtrm_native_official_len8_op_counterfactual_s800_20260513/report.json
      full_generation_exact: 0.03125
      full_minus_think0: 0.005208333333333332
      full_minus_worst_ablation: 0.010416666666666668
      op_zero_generation_exact: 0.020833333333333332
      min_family_generation_exact: 0.03125
      conclusion: first run in this series that makes `op_zero` worse than
        full, but it lowers exact accuracy below baseline.

    weight 0.1, warmup 400, active_len 6..8 only:
      local_eval/qtrm_native_official_len8_op_counterfactual_warm400_len6_8_w01_s800_20260513/report.json
      full_generation_exact: 0.03125
      full_minus_think0: 0.005208333333333332
      full_minus_worst_ablation: -0.005208333333333336
      state_reset_generation_exact: 0.015625
      op_zero_generation_exact: 0.036458333333333336
      min_family_generation_exact: 0.03125
      conclusion: rejected. Delaying the loss and applying it only to long
        active lengths removes the useful causal pressure; `op_zero` again
        beats the full model.

    weight 0.1, warmup 400, all active lengths:
      local_eval/qtrm_native_official_len8_op_counterfactual_warm400_all_w01_s800_20260513/report.json
      full_generation_exact: 0.036458333333333336
      full_minus_think0: 0.010416666666666668
      full_minus_worst_ablation: 0.0
      state_reset_generation_exact: 0.015625
      op_zero_generation_exact: 0.036458333333333336
      min_family_generation_exact: 0.015625
      conclusion: rejected. Removing the active-length filter restores baseline
        exact accuracy, but `op_zero` still ties the full model. Therefore the
        warmup itself kills the counterfactual causal signal; the loss must
        shape the model early.

    weight 0.1, early-only steps 1..400, all active lengths:
      local_eval/qtrm_native_official_len8_op_counterfactual_early400_all_w01_s800_20260513/report.json
      full_generation_exact: 0.026041666666666668
      full_minus_think0: 0.0
      full_minus_worst_ablation: -0.010416666666666668
      state_reset_generation_exact: 0.026041666666666668
      op_zero_generation_exact: 0.036458333333333336
      min_family_generation_exact: 0.015625
      conclusion: rejected. Early-only shaping does not persist; full collapses
        to think0 while `op_zero` beats full. Counterfactual pressure must stay
        active if it is used.

    weight 0.1 plus answer-space ranking 0.05:
      local_eval/qtrm_native_official_len8_opcf_w01_answer_rank_w005_s800_20260513/report.json
      full_generation_exact: 0.036458333333333336
      full_minus_think0: 0.010416666666666668
      full_minus_worst_ablation: 0.0
      state_reset_generation_exact: 0.020833333333333332
      op_zero_generation_exact: 0.036458333333333336
      min_family_generation_exact: 0.015625
      conclusion: rejected. Answer-space ranking recovers baseline exact
        accuracy but also recovers the zero-op shortcut, so it cancels the
        useful causal effect of constant counterfactual pressure.

    resume causal weight 0.1 checkpoint for 800 more steps:
      local_eval/qtrm_native_official_len8_op_counterfactual_w01_resume800_20260513/report.json
      resume_from: local_eval/qtrm_native_official_len8_op_counterfactual_s800_20260513/last.pt
      full_generation_exact: 0.010416666666666666
      full_minus_think0: 0.0
      full_minus_worst_ablation: -0.036458333333333336
      state_reset_generation_exact: 0.041666666666666664
      op_zero_generation_exact: 0.046875
      min_family_generation_exact: 0.0
      mean_halt_steps: 4.953125
      halted_fraction: 1.0
      conclusion: rejected. Continued training lowers loss and learns the
        active_len halt target, but destroys answer accuracy and causality.
        This suggests the adaptive-halt auxiliary can become a competing
        shortcut objective at len8.

    fixed-depth, no adaptive halt, weight 0.1:
      local_eval/qtrm_native_official_len8_fixed_depth_opcf_w01_s800_20260513/report.json
      full_generation_exact: 0.041666666666666664
      full_minus_think0: 0.015624999999999997
      full_minus_worst_ablation: 0.0
      state_reset_generation_exact: 0.026041666666666668
      op_zero_generation_exact: 0.041666666666666664
      min_family_generation_exact: 0.03125
      conclusion: rejected. Removing adaptive halt improves exact accuracy and
        depth gain, but `op_zero` ties full. Halt was an interference source,
        not the whole shortcut problem.

    fixed-depth, no adaptive halt, weight 0.15:
      local_eval/qtrm_native_official_len8_fixed_depth_opcf_w015_s800_20260513/report.json
      full_generation_exact: 0.041666666666666664
      full_minus_think0: 0.015624999999999997
      full_minus_worst_ablation: -0.005208333333333336
      state_reset_generation_exact: 0.046875
      op_zero_generation_exact: 0.036458333333333336
      min_family_generation_exact: 0.03125
      conclusion: rejected. Stronger op counterfactual starts to lower
        `op_zero`, but `state_reset` becomes better than full. This points to
        another auxiliary shortcut: depth-intermediate supervision trains
        shallow/reset-like paths to solve the answer.

    fixed-depth, no intermediate, op counterfactual 0.15:
      local_eval/qtrm_native_official_len8_fixed_depth_nointermediate_opcf_w015_s800_20260513/report.json
      full_generation_exact: 0.03125
      full_minus_think0: 0.0
      full_minus_worst_ablation: 0.0
      state_reset_generation_exact: 0.03125
      op_zero_generation_exact: 0.0
      min_family_generation_exact: 0.03125
      conclusion: rejected. Removing intermediate supervision makes `op_zero`
        causal but removes full-depth advantage.

    fixed-depth, no intermediate, op counterfactual 0.15 plus depth counterfactual 0.1:
      local_eval/qtrm_native_official_len8_fixed_depth_nointermediate_opcf_w015_depthcf_w01_s800_20260513/report.json
      full_generation_exact: 0.041666666666666664
      full_minus_think0: 0.041666666666666664
      full_minus_worst_ablation: 0.0
      state_reset_generation_exact: 0.041666666666666664
      op_zero_generation_exact: 0.0
      min_family_generation_exact: 0.03125
      conclusion: rejected but important. Depth counterfactual fixes think0
        and op_zero, while preserving the best fixed-depth full exact. The
        only remaining strict ablation tie is `state_reset`.

    fixed-depth, no intermediate, op/depth/state counterfactual:
      local_eval/qtrm_native_official_len8_fixed_depth_op_depth_state_cf_s800_20260513/report.json
      full_generation_exact: 0.0625
      full_minus_think0: 0.0625
      full_minus_worst_ablation: 0.0625
      state_reset_generation_exact: 0.0
      op_zero_generation_exact: 0.0
      z_l_zero_generation_exact: 0.0
      z_h_zero_generation_exact: 0.0
      min_family_generation_exact: 0.03125
      conclusion: best strict-causal scaffold so far. It is still rejected by
        the absolute L4 exact threshold, but it is the first len8 run in this
        series where full recurrence beats think0, state_reset, op_zero, and
        z-state zero ablations at the same time.

    resume the strict-causal scaffold for 1600 more steps:
      local_eval/qtrm_native_official_len8_fixed_depth_op_depth_state_cf_resume1600_20260513/report.json
      full_generation_exact: 0.026041666666666668
      full_minus_think0: 0.0
      full_minus_worst_ablation: -0.005208333333333332
      state_reset_generation_exact: 0.03125
      op_zero_generation_exact: 0.0
      min_family_generation_exact: 0.015625
      conclusion: rejected. More training from the best causal checkpoint loses
        the strict depth/state advantage, so the 800-step checkpoint should be
        treated as the current peak for this small-capacity recipe.

    modest capacity scaling, d_model 96, same op/depth/state counterfactual recipe:
      local_eval/qtrm_native_official_len8_fixed_depth_op_depth_state_cf_d96_s800_20260513/report.json
      full_generation_exact: 0.052083333333333336
      full_minus_think0: 0.052083333333333336
      full_minus_worst_ablation: 0.052083333333333336
      state_reset_generation_exact: 0.0
      op_zero_generation_exact: 0.0
      z_l_zero_generation_exact: 0.0
      z_h_zero_generation_exact: 0.0
      min_family_generation_exact: 0.03125
      format_valid: 1.0
      conclusion: rejected. Capacity scaling preserves the strict causal
        ablation pattern, but does not beat the d_model 64 peak
        full_generation_exact 0.0625. Width alone is therefore not the next
        sufficient fix; the useful signal is the counterfactual recipe, not
        merely parameter count.

  Decision:
    Operation counterfactual contrast is a real causal lever. It is not yet an
    accepted len8 solution because the useful weight reduces exact accuracy.
    A naive warmup plus active_len 6..8 filter is worse than the constant
    all-length version. Warmup-only preserves exact but loses causality, so the
    early-only version was tested and also rejected. The remaining local path is
    to simplify the training objective before adding new architecture:
      keep counterfactual contrast active from the beginning through the end;
      use fixed-depth recurrence as the cleaner objective base;
      remove depth-intermediate supervision when testing strict causal
      ablations, because it explicitly rewards shallow paths;
      add a state-reset counterfactual contrast so recurrent state continuity,
      not only depth count and operation tokens, becomes causally necessary;
      use the op/depth/state counterfactual recipe as the current causal
      scaffold; longer training from the same small model did not help, and
      d_model 96 capacity scaling preserved causality but did not improve
      absolute exact accuracy;
      keep strict gate: full > think0, full > op_zero/state_reset, min family > 0.
```

Canonical status:

```text
Experimental. Not promoted.

Promotion requires a same-budget comparison:
  base official_trm_think or accepted MHA ETD L6 floor
  vs trm_dual_z_diffusive + latent_refine

Accept only if:
  generation_exact improves;
  think0/state_reset/op_zero/z_l_zero/z_h_zero reduce the gain;
  no generation-format regression;
  latent-refine loss alone is not counted as success.
```

Decision:

```text
Latent diffusion is worth testing as a readout/trajectory refinement mechanism.
It should not replace the TRM core yet. The highest-signal next implementation
is an auxiliary latent-refinement loss that gives the recurrent core a way to
revise uncertain intermediate states before final AR token commitment.
```

## L5I Coupled Dual-TRM Architecture Experiment

Goal:

```text
Create a dual-TRM architecture that beats official_trm_think while preserving
the orthodox causal LM path:

tokens
-> embeddings
-> encoder
-> mandatory z_L / z_H recursive TRM core
-> decoder
-> shared LM head
-> autoregressive text
```

Architecture candidates tested:

```text
1. trm_dual_z_coupled_residual_official_trm_think

   z_L / z_H official TRM core
   + z_L GatedDeltaNet local-state proposal
   + z_H Mamba3 slow-state proposal
   + residual readout

2. trm_dual_z_coupled_official_trm_think

   z_L / z_H official TRM core
   + learned bidirectional cross-state coupling
   + no Mamba3/GatedDeltaNet proposal
   + no residual readout shortcut
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds think_structure=trm_dual_z_coupled
  adds think_structure=trm_dual_z_coupled_residual

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds candidate=trm_dual_z_coupled_official_trm_think
  adds candidate=trm_dual_z_coupled_residual_official_trm_think
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_l5d_backbone_compare

Result:
  56 tests OK
```

Short-gate results, seed 337 / eval seed 9337:

```text
Baseline comparison:
  official_trm_think:                          0.057291666666666664
  trm_dual_z_official_trm_think:               0.046875
  trm_dual_z_coupled_residual_official_trm:    0.041666666666666664

Follow-up pure coupling:
  trm_dual_z_coupled_official_trm_think:       0.07291666666666667
```

Pure coupling decisive metrics:

```text
full_generation_exact:          0.07291666666666667
think0_generation_exact:        0.036458333333333336
state_reset_generation_exact:   0.046875
op_zero_generation_exact:       0.03125
full_minus_think0:              0.036458333333333336
full_minus_worst_ablation:      0.02604166666666667
min_family_generation_exact:    0.046875
```

Interpretation:

```text
Accepted as the new short-gate winner.

The useful change was not adding Mamba3/GatedDeltaNet proposal branches. The
proposal version had official FLA/Mamba3 backends working, but it reduced
generation exact and weakened the depth gain.

The useful change was cleaner dual-state coupling:
  z_L -> z_H bottom-up summary
  z_H -> z_L top-down control

while keeping the official TRM attention block as the recurrent update and
keeping the final answer on the normal LM-logit path.
```

Canonical status:

```text
Promote trm_dual_z_coupled_official_trm_think to the current L5I short-gate
candidate.

Do not promote Mamba3/GatedDeltaNet proposal adapters yet. They remain
diagnostic/optional until they beat pure coupled dual-TRM and pass the same
depth/ablation checks.
```

Immediate next step:

```text
1. Add explicit coupling ablations:
   coupling_off, z_L_zero, z_H_zero.
2. Run a seed sweep against official_trm_think and trm_dual_z_official_trm.
3. If stable, promote coupled dual-TRM from short-gate winner to canonical
   L5 native recursive reasoning core.
4. Only then revisit Mamba3/GatedDeltaNet as proposal adapters.
```

## L5J Mamba3/GatedDeltaNet Proposal Adapter Check

Question:

```text
Can Mamba3 or GatedDeltaNet improve the new L5I winner if they are added
conservatively, one branch at a time?
```

Candidates:

```text
baseline:
  trm_dual_z_coupled_official_trm_think

proposal variants:
  trm_dual_z_coupled_delta_l_only_official_trm_think
    z_L gets a small GatedDeltaNet local-state proposal

  trm_dual_z_coupled_mamba_h_only_official_trm_think
    z_H gets a small Mamba3 slow-state proposal

  trm_dual_z_coupled_gated_proposal_official_trm_think
    z_L gets GatedDeltaNet and z_H gets Mamba3, both behind near-zero
    learned gates
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds think_structure=trm_dual_z_coupled_delta_l_only
  adds think_structure=trm_dual_z_coupled_mamba_h_only
  adds think_structure=trm_dual_z_coupled_gated_proposal

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds the matching compare candidates
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_l5d_backbone_compare

Result:
  63 tests OK
```

Smoke:

```text
local_eval/qtrm_native_l5j_coupled_proposals_smoke_20260512

All three proposal variants ran.
Backend checks passed:
  delta_l_only: official FLA GatedDeltaNet
  mamba_h_only: official Mamba3
  gated_proposal: official FLA GatedDeltaNet + official Mamba3
```

Short-gate result:

```text
local_eval/qtrm_native_l5j_coupled_proposals_short_20260512

trm_dual_z_coupled_official_trm_think:                  0.07291666666666667
trm_dual_z_coupled_delta_l_only_official_trm_think:     0.046875
trm_dual_z_coupled_mamba_h_only_official_trm_think:     0.052083333333333336
trm_dual_z_coupled_gated_proposal_official_trm_think:   0.026041666666666668
```

Decisive metrics:

```text
pure coupled:
  full_minus_think0:          0.036458333333333336
  full_minus_worst_ablation:  0.02604166666666667
  causal_ok:                 true

delta_l_only:
  full_minus_think0:          0.020833333333333332
  full_minus_worst_ablation:  0.010416666666666664
  causal_ok:                 true

mamba_h_only:
  full_minus_think0:          0.026041666666666668
  full_minus_worst_ablation:  0.015625
  causal_ok:                 true

gated_proposal:
  full_minus_think0:          0.0
  full_minus_worst_ablation: -0.010416666666666668
  causal_ok:                 false
```

Interpretation:

```text
Reject Mamba3/GatedDeltaNet proposal adapters for the current L5 canonical
core.

They are valid runtime modules and their official backends work, but they did
not beat the simpler coupled dual-TRM. The stronger architecture signal is:

  keep official TRM attention update
  keep z_L/z_H dual recurrence
  keep bidirectional coupling
  avoid extra proposal adapters until a harder gate shows a need for them
```

Canonical status:

```text
Keep trm_dual_z_coupled_official_trm_think as the current winner.

Demote:
  trm_dual_z_coupled_delta_l_only_official_trm_think
  trm_dual_z_coupled_mamba_h_only_official_trm_think
  trm_dual_z_coupled_gated_proposal_official_trm_think

Do not spend more cycles on Mamba3/GatedDeltaNet proposal adapters until the
pure coupled dual-TRM passes coupling-specific ablations and seed sweep.
```

## L5K TRM Attention Update Variants

Question:

```text
Can the current coupled dual-TRM winner be improved by changing only the
official TRM attention update?
```

Candidates:

```text
baseline:
  trm_dual_z_coupled_official_trm_think

variant A:
  trm_dual_z_coupled_gated_attention_think
  - keeps the official TRM attention/SwiGLU shape
  - adds learned scalar residual gates on attention and MLP updates

variant B:
  trm_dual_z_coupled_qwen_attention_think
  - replaces nn.MultiheadAttention with a Qwen-style attention update
  - uses RMSNorm pre-norm, bias-free Q/K/V/O projections, QK-norm, SwiGLU
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds backbone=trm_gated_attention
  adds backbone=trm_qwen_attention

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds candidate=trm_dual_z_coupled_gated_attention_think
  adds candidate=trm_dual_z_coupled_qwen_attention_think
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_l5d_backbone_compare

Result:
  67 tests OK
```

Short-gate results:

```text
local_eval/qtrm_native_l5k_gated_attention_short_20260512

trm_dual_z_coupled_official_trm_think:   0.07291666666666667
trm_dual_z_coupled_gated_attention:      0.041666666666666664

local_eval/qtrm_native_l5k_qwen_attention_short_20260512

trm_dual_z_coupled_official_trm_think:   0.07291666666666667
trm_dual_z_coupled_qwen_attention:       0.03125
```

Decisive metrics:

```text
official coupled:
  causal_ok:                 true
  full_minus_think0:          0.036458333333333336
  full_minus_worst_ablation:  0.02604166666666667

gated attention:
  causal_ok:                 false
  full_minus_think0:          0.020833333333333332
  full_minus_worst_ablation:  0.0

Qwen-style attention:
  causal_ok:                 false
  full_minus_think0:         -0.010416666666666664
  full_minus_worst_ablation:  0.0
```

Interpretation:

```text
Reject both attention-update variants for the current L5 canonical path.

The winning signal is still the simpler official TRM attention block inside
the coupled z_L/z_H recurrent loop. Extra gating weakened ablation separation,
and the Qwen-style local attention update made 4-step thinking worse than
think0 on this gate.
```

Canonical status:

```text
Keep trm_dual_z_coupled_official_trm_think as the current L5 short-gate
winner.

Do not replace the official TRM attention update until a candidate beats it
on all three conditions:
  1. higher full_generation_exact
  2. positive full_minus_think0
  3. positive full_minus_worst_ablation
```

## Qwen3.5-Style Backbone Deferral Policy

Decision:

```text
Do not force Qwen3.5-style hybrid blocks into the canonical recursive core yet.
Defer them until the native LM backbone scaling phase.
```

Reason:

```text
Qwen3.5-style architecture gains are whole-model gains, not a guaranteed
drop-in recurrent-core upgrade.

They depend on:
  - large-scale LM pretraining
  - the full backbone being trained around the hybrid block pattern
  - tokenizer/data/optimizer/schedule co-adaptation
  - long-context and high-token-count regimes where linear/hybrid attention
    inductive bias becomes useful

The current L5 gate is different:
  - tiny synthetic reasoning task
  - short training run
  - newly initialized small blocks
  - MHA encode/decode with only the recurrent think block swapped
  - evaluation focused on causal depth gain, not long-context throughput
```

Observed evidence:

```text
Mamba3/GatedDeltaNet proposal adapters did not beat pure coupled dual-TRM.
Qwen-style attention update did not beat pure coupled dual-TRM.

Current winner:
  trm_dual_z_coupled_official_trm_think

Current rejected substitutions:
  trm_dual_z_coupled_delta_l_only_official_trm_think
  trm_dual_z_coupled_mamba_h_only_official_trm_think
  trm_dual_z_coupled_gated_proposal_official_trm_think
  trm_dual_z_coupled_gated_attention_think
  trm_dual_z_coupled_qwen_attention_think
```

Canonical order:

```text
Now:
  prove the TRM/QTRM recursive core itself
  keep official TRM attention inside coupled z_L/z_H recurrence
  require depth/core/coupling ablations to show causal gain

Later:
  scale the native LM backbone
  then reintroduce Qwen3.5-style hybrid architecture as a full-backbone
  candidate, not as a small isolated recurrent-core patch
```

Promotion rule for Qwen3.5-style ideas:

```text
Promote only if the Qwen-style backbone candidate improves the same canonical
LM path:

  tokens
  -> token embeddings
  -> native backbone
  -> mandatory coupled TRM/QTRM recurrent core
  -> LM logits
  -> autoregressive text

and passes:
  1. higher generation exact or lower LM loss than the simple backbone
  2. recurrence depth improves held-out reasoning
  3. core_off/state_reset/coupling_off ablations reduce the same metric
  4. language non-regression stays acceptable
```

Practical takeaway:

```text
Qwen3.5-style hybrid design is not rejected.
It is deferred from "core proof" to "native backbone scaling".

The current proof target is narrower:
  official TRM attention + coupled z_L/z_H recurrence must first become a
  stable causal reasoning core.
```

## L5L More TRM-Like Attention Update Variants

Question:

```text
Can the official TRM attention update be made more TRM-like without importing
Qwen/Mamba/GatedDelta backbone complexity?
```

Candidates:

```text
baseline:
  trm_dual_z_coupled_official_trm_think

variant A:
  trm_dual_z_coupled_cross_attention_think
  - keeps official TRM attention as the recurrent update
  - adds causal z_L -> z_H and z_H -> z_L cross-attention links
  - intended to make low/high state communication explicit instead of only
    additive projection coupling

variant B:
  trm_dual_z_coupled_step_conditioned_attention_think
  - keeps official TRM attention as the recurrent update
  - injects learned recurrent step embeddings into the coupled update
  - intended to let the shared recurrent block specialize by depth/iteration
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  adds think_structure=trm_dual_z_coupled_cross_attention
  adds think_structure=trm_dual_z_coupled_step_conditioned_attention

scripts/342_qtrm_native_l5d_backbone_compare.py
  adds candidate=trm_dual_z_coupled_cross_attention_think
  adds candidate=trm_dual_z_coupled_step_conditioned_attention_think
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_l5d_backbone_compare

Result:
  71 tests OK
```

Short-gate result:

```text
local_eval/qtrm_native_l5l_trm_attention_variants_short_20260512

trm_dual_z_coupled_official_trm_think:                  0.07291666666666667
trm_dual_z_coupled_cross_attention_think:               0.036458333333333336
trm_dual_z_coupled_step_conditioned_attention_think:    0.052083333333333336
```

Decisive metrics:

```text
official coupled:
  causal_ok:                 true
  full_minus_think0:          0.036458333333333336
  full_minus_worst_ablation:  0.02604166666666667

cross attention:
  causal_ok:                 true
  full_minus_think0:          0.020833333333333336
  full_minus_worst_ablation:  0.005208333333333336

step-conditioned attention:
  causal_ok:                 true
  full_minus_think0:          0.026041666666666668
  full_minus_worst_ablation:  0.015625
```

Interpretation:

```text
Reject both L5L variants as canonical replacements.

Both variants preserve causal depth/ablation behavior, which is better than
the L5K gated/Qwen attention substitutions. However, neither beats the simpler
coupled dual-TRM baseline.

Step conditioning is the more promising failed variant because it keeps a
reasonable depth gain while adding only a small learned recurrent-step signal.
It may be worth retrying after seed sweep or after the core is scaled, but it
does not replace the current winner.
```

Canonical status:

```text
Keep trm_dual_z_coupled_official_trm_think as the canonical L5 short-gate
winner.

Demote:
  trm_dual_z_coupled_cross_attention_think
  trm_dual_z_coupled_step_conditioned_attention_think

Current lesson:
  explicit cross-state attention adds capacity but weakens the simple signal;
  recurrent step conditioning is cleaner, but not strong enough yet.
```

## L5M Coupled Core Causal Ablation Gate

Decision:

```text
Architecture shopping is paused.
The current task is to prove and scale the canonical winner:

  trm_dual_z_coupled_official_trm_think
```

Why this gate exists:

```text
The previous ablations checked:
  think0
  state_reset
  op_zero
  thinking_block_off

That was not enough to prove that the coupled z_L/z_H core itself was the
causal source of the gain. L5M adds direct coupled-core ablations:

  coupling_off
  z_l_zero
  z_h_zero
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py
  NativeQTRMETDLM.forward now accepts:
    coupling_off
    z_l_zero
    z_h_zero

  coupled recurrent cycles apply these flags inside the loop, not only at
  final readout.

scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  evaluate() now supports:
    coupling_off
    z_l_zero
    z_h_zero

  train_probe() now writes these metrics into eval_metrics.

  make_decision() now computes full_minus_worst_ablation across:
    state_reset
    op_zero
    coupling_off
    z_l_zero
    z_h_zero
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_qtrm_native_l5d_backbone_compare

Result:
  86 tests OK
```

Short-gate result:

```text
local_eval/qtrm_native_l5m_coupling_ablation_short_20260512

full_generation_exact:          0.07291666666666667
think0_generation_exact:        0.036458333333333336
state_reset_generation_exact:   0.046875
op_zero_generation_exact:       0.03125
coupling_off_generation_exact:  0.046875
z_l_zero_generation_exact:      0.0
z_h_zero_generation_exact:      0.0

full_minus_think0:              0.036458333333333336
full_minus_worst_ablation:      0.02604166666666667
causal_ok:                     true
```

Interpretation:

```text
The current winner survives the stronger causal gate.

The strongest evidence is:
  z_l_zero -> 0.0
  z_h_zero -> 0.0

So the output is not merely coming from the encoder/decoder or from answer
formatting. Both recurrent latent states are necessary for the measured gain.

coupling_off falls to 0.046875, equal to state_reset and below full 0.0729167.
This means explicit coupling helps, but the most decisive evidence is that
zeroing either latent state destroys the result completely.
```

Canonical status:

```text
Previous short-gate working candidate:
  trm_dual_z_coupled_official_trm_think

Do not add new architecture variants until these proof/scaling tasks are done:
  1. seed sweep with L5M ablations
  2. harder reasoning gates: program_len 6/8, modulus 64
  3. longer training: 2k/5k steps
  4. language non-regression check
```

Post-fix correction, 2026-05-12:

```text
The first L5M strict seed sweep exposed an evaluation semantics problem:
coupling/state ablations were being recorded even for structures where they
were not applicable, such as `single` MHA and single official TRM.

Code fix:
  scripts/335_train_qtrm_native_etd_probe.py
    added applicable_ablation_names(think_structure)
    applies z_l_zero / z_h_zero inside non-coupled dual-z recurrent cycles

  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
    formal decisions now use only applicable ablations
    train_probe() now records only applicable ablations

Verification:
  93 related tests OK
```

Post-fix seed-338 diagnostic:

```text
local_eval/qtrm_native_l5m_seed338_postfix_compare_short_20260512

mha_etd:                                0.046875
official_trm_think:                    0.046875
trm_dual_z_official_trm_think:         0.062500  <- promoted on this seed
trm_dual_z_coupled_official_trm_think: 0.052083
```

Post-fix 3-seed strict sweep:

```text
local_eval/qtrm_native_l5m_dual_z_seed_sweep_short_postfix_20260512

target_candidate:
  trm_dual_z_official_trm_think

decision:
  rejected

promoted_count:
  2 / 3

reject_reasons:
  promoted_rate_below_threshold
  seed_delta_below_threshold

per-seed:
  seed 337:
    winner: trm_dual_z_coupled_official_trm_think
    dual_z promoted: true

  seed 338:
    winner: trm_dual_z_official_trm_think
    dual_z promoted: true

  seed 339:
    winner: trm_dual_z_coupled_official_trm_think
    dual_z promoted: false
```

Interpretation:

```text
Do not promote either dual-z or coupled dual-z as stable L5M canonical yet.

What is proven:
  recurrent thinking depth can improve over think0 in individual seeds;
  z_l/z_h ablations are now wired causally for dual-z structures;
  the evaluation gate no longer counts nonexistent coupling/state ablations.

What is not proven:
  no short-scale candidate wins with seed-stable 3/3 promotion;
  explicit coupling is not reliably causal across seeds;
  short 400-step runs are too noisy for final architecture selection.
```

Next mandatory steps:

```text
1. Stop adding new architecture variants.
2. Add a standard-scale placement profile matching the accepted L5 regime
   instead of relying on 400-step short gates. [done]
3. Run only the minimal KISS comparison:
     mha_etd
     official_trm_think
     trm_dual_z_official_trm_think
     trm_dual_z_coupled_official_trm_think
4. Promote only if a candidate passes:
     3/3 seed promoted
     positive delta vs MHA
     positive depth gain vs think0
     positive applicable-ablation drop
     family floor preserved
5. Only after that, run language non-regression and scaled reasoning.
```

Standard-scale runner:

```text
scripts/342_qtrm_native_l5d_backbone_compare.py
scripts/343_qtrm_native_l5d_placement_seed_sweep.py

new profile:
  --profile standard

scale:
  steps: 12000
  train_cases: 24576
  eval_cases: 768
  d_model: 128
  n_heads: 8
  d_ff: 256

acceptance:
  accept_min_exact: 0.60
  accept_min_depth_gain: 0.10
  accept_min_ablation_drop: 0.10
  accept_min_family_exact: 0.40
```

## Native Adaptive Halt Runtime

Question:

```text
Does QTRM-native have the TRM-style early termination path?
```

Current answer:

```text
Yes as a runtime scaffold, not yet as a promoted reasoning result.
```

Implementation:

```text
scripts/335_train_qtrm_native_etd_probe.py

NativeQTRMETDLM now has:
  core_halt_head
  forward_with_runtime(...)
  adaptive_halt
  halt_threshold
  halt_min_steps

Runtime telemetry:
  logits
  core_q_halt_logits
  core_halted
  halt_steps
  executed_think_steps

The halt path is batch-shared for now:
  if all batch items cross the halt threshold after halt_min_steps, the
  recurrent thinking loop stops before max think_steps.
```

Mixed text evaluation:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new eval options:
  --adaptive-halt-eval
  --halt-threshold
  --halt-min-steps

When enabled, eval_metrics["adaptive_halt"] records:
  mean_halt_steps
  executed_think_steps
  halted_fraction
  core_q_halt_shape
```

Teacher-depth halt training:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new training option:
  --adaptive-halt-loss-weight

Target construction:
  run fixed depths 1..D
  mark a depth as correct if its answer text tokens match the target
  earliest eligible correct depth becomes the first halt point
  all later depths are halt=1
  no correct depth means halt=0 for all depths

This keeps halt learning inside the normal LM-logit answer path. It is not a
rule solver and it does not compute the answer externally.
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_qtrm_native_l5d_backbone_compare \
  tests.test_qtrm_native_l5d_placement_seed_sweep

Result:
  100 tests OK
```

Runtime smoke:

```text
local_eval/qtrm_native_adaptive_halt_smoke_20260513

adaptive_halt:
  mean_halt_steps: 4.0
  executed_think_steps: 4
  halted_fraction: 0.0
  core_q_halt_shape: [6, 4]
```

Halt-loss smoke:

```text
local_eval/qtrm_native_adaptive_halt_loss_smoke_20260513

args:
  --adaptive-halt-eval
  --adaptive-halt-loss-weight 0.1

adaptive_halt:
  mean_halt_steps: 4.0
  executed_think_steps: 4
  halted_fraction: 0.0
  core_q_halt_shape: [6, 4]
```

Short diagnostic:

```text
local_eval/qtrm_native_adaptive_halt_loss_short_20260513

args:
  --steps 400
  --eval-cases 192
  --adaptive-halt-eval
  --adaptive-halt-loss-weight 0.1

decision:
  accepted_l4_mixed_text_reasoning

decisive_metrics:
  full_generation_exact: 0.0572916667
  think0_generation_exact: 0.0364583333
  full_minus_think0: 0.0208333333
  full_minus_worst_ablation: 0.0052083333
  z_l_zero_generation_exact: 0.0
  z_h_zero_generation_exact: 0.0

adaptive_halt:
  generation_exact: 0.0572916667
  mean_halt_steps: 4.0
  executed_think_steps: 4
  halted_fraction: 0.0
  core_q_halt_shape: [192, 4]
```

Active-length halt target diagnostics:

```text
Implementation:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new helpers/options:
  effective_program_len(...)
  active_len_halt_targets(...)
  active_len_first_halt_targets(...)
  adaptive_halt_active_len_loss(...)
  --adaptive-halt-target-mode active_len
  --adaptive-halt-active-len-target first_step|cumulative
  --active-len-batch-cycle
  --eval-active-len-cycle

Reason:
  teacher-depth correctness gives almost no positive halt target before the
  model already answers correctly. For synthetic active/noop-tail probes, the
  task metadata can supervise "minimum useful recurrent depth" without
  computing the answer externally.
```

Failed active-length variants:

```text
local_eval/qtrm_native_adaptive_halt_active_len_short_20260513
  target: cumulative
  train: active_len_curriculum
  result: mean_halt_steps 4.0, halted_fraction 0.0
  interpretation: full-length curriculum tail dominated; halt stayed continue.

local_eval/qtrm_native_adaptive_halt_batchmix_short_20260513
  target: cumulative
  train: active_len_batch_cycle
  result: mean_halt_steps 1.0, halted_fraction 1.0
  interpretation: cumulative BCE over-produced positive labels and collapsed
  to always halt at step 1.

local_eval/qtrm_native_adaptive_halt_batchmix_firststep_short_20260513
  target: first_step
  train: active_len_batch_cycle
  result: mean_halt_steps 4.0, halted_fraction 0.0
  interpretation: first-step target was better shaped but undertrained at 400
  steps.
```

First useful adaptive halt diagnostic:

```text
local_eval/qtrm_native_adaptive_halt_batchmix_firststep_s1200_telemetry_20260513

args:
  --steps 1200
  --active-len-batch-cycle
  --eval-active-len-cycle
  --adaptive-halt-loss-weight 5.0
  --adaptive-halt-target-mode active_len
  --adaptive-halt-active-len-target first_step

fixed-depth:
  think4.generation_exact: 0.3229166667

adaptive_halt:
  generation_exact: 0.3229166667
  mean_halt_steps: 1.5364583731
  halted_fraction: 0.953125
  executed_think_steps: 4

halt_by_active_len:
  0: mean_halt_steps 1.0769, halted_fraction 0.9744
  1: mean_halt_steps 1.2308, halted_fraction 0.9231
  2: mean_halt_steps 1.1579, halted_fraction 0.9474
  3: mean_halt_steps 1.2368, halted_fraction 0.9211
  4: mean_halt_steps 3.0000, halted_fraction 1.0000
```

Teacher-forced adaptive-halt post-hoc decision:

```text
Decision helper:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py::make_decision

new acceptance flags:
  --accept-require-adaptive-halt
  --accept-max-adaptive-halt-exact-drop
  --accept-max-mean-halt-steps
  --accept-min-halted-fraction

Applied to:
  local_eval/qtrm_native_adaptive_halt_batchmix_firststep_s1200_telemetry_20260513

criteria:
  accept_min_exact: 0.20
  accept_min_depth_gain: 0.01
  accept_min_ablation_drop: -1.0
  accept_max_adaptive_halt_exact_drop: 0.01
  accept_max_mean_halt_steps: 2.0
  accept_min_halted_fraction: 0.90

result:
  accepted: true
  telemetry_source: teacher_forced
  full_generation_exact: 0.3229166667
  adaptive_halt_generation_exact: 0.3229166667
  full_minus_adaptive_halt: 0.0
  adaptive_halt_mean_steps: 1.5364583731
  adaptive_halt_halted_fraction: 0.953125
```

Generation-path correction:

```text
local_eval/qtrm_native_adaptive_halt_batchmix_firststep_s1200_gen_telemetry_20260513

adaptive_halt:
  generation_exact: 0.3229166667
  teacher_forced_mean_halt_steps: 1.5364583731
  teacher_forced_halted_fraction: 0.953125
  generation_mean_executed_think_steps: 3.1631944444
  generation_mean_halted_fraction: 0.3229166667

strict decision with generation telemetry preferred:
  accepted: false
  reject_reasons:
    adaptive_halt_mean_steps_above_threshold
    adaptive_halt_fraction_below_threshold
  adaptive_halt_telemetry_source: generation
```

Halt-context and threshold sweep:

```text
new training option:
  --adaptive-halt-loss-context full|prompt|prefixes

Reason:
  full context trains halt with teacher-forced answer prefix visible.
  prompt context trains halt on the first generation context only.
  prefixes context trains halt on prompt plus every gold answer prefix.

Sweep results:

local_eval/qtrm_native_adaptive_halt_batchmix_firststep_s1200_gen_telemetry_20260513
  context: full
  threshold 0.2: exact 0.3229, generation_exec_steps 3.0694, generation_halted 0.3333
  threshold 0.5: exact 0.3229, generation_exec_steps 3.1632, generation_halted 0.3229

local_eval/qtrm_native_adaptive_halt_promptctx_s1200_20260513
  context: prompt
  threshold 0.2: exact 0.3542, generation_exec_steps 3.8524, generation_halted 0.1042
  threshold 0.5: exact 0.3542, generation_exec_steps 3.9410, generation_halted 0.0677

local_eval/qtrm_native_adaptive_halt_prefixctx_s1200_20260513
  context: prefixes
  threshold 0.5: exact 0.3229, generation_exec_steps 3.8698, generation_halted 0.1979

Stable halt-head pooling probe:

local_eval/qtrm_native_adaptive_halt_meanpool_fullctx_s1200_20260513
  halt_pooling: mean
  context: full
  threshold 0.5:
    fixed_exact: 0.34375
    adaptive_exact: 0.33333
    generation_exec_steps: 3.75694
    generation_halted: 0.23090
    think0_exact: 0.03125
    state_reset_exact: 0.30729
    thinking_block_off_exact: 0.03125

Interpretation:
  threshold tuning alone does not solve the generation-path compute gate.
  prompt/prefix context losses did not improve adaptive halt; prompt context
  improved fixed-depth exact to 0.3542 and gave a larger think0/depth gap, but
  halt stayed mostly off. Prefix context increased training cost and still did
  not reduce generation compute. Mean-pooling halt did not solve compute either,
  but it did preserve a larger depth-vs-think0 gap and a small state_reset gap.
```

Dedicated halt-state probe:

```text
Implementation:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new option:
  --halt-pooling dedicated

Mechanism:
  initialize a separate halt state from the encoded prompt/core summary
  update it once per recurrent thinking step with a GRUCell
  compute halt logits from this dedicated state instead of the last LM token
  keep final answer generation on normal LM logits

Result:
  local_eval/qtrm_native_adaptive_halt_dedicated_fullctx_s1200_20260513

args:
  --halt-pooling dedicated
  --adaptive-halt-loss-weight 5.0
  --adaptive-halt-target-mode active_len
  --adaptive-halt-active-len-target first_step
  --adaptive-halt-loss-context full
  --active-len-batch-cycle
  --eval-active-len-cycle

fixed-depth:
  think4.generation_exact: 0.3645833333
  think0.generation_exact: 0.015625
  state_reset.generation_exact: 0.328125
  op_zero.generation_exact: 0.2239583333
  z_l_zero.generation_exact: 0.0
  z_h_zero.generation_exact: 0.0
  min_family_generation_exact: 0.28125

adaptive_halt:
  generation_exact: 0.3333333333
  generation_mean_executed_think_steps: 2.1875
  generation_mean_halted_fraction: 1.0

halt_by_active_len:
  0: mean_halt_steps 1.0, halted_fraction 1.0
  1: mean_halt_steps 1.0, halted_fraction 1.0
  2: mean_halt_steps 2.0, halted_fraction 1.0
  3: mean_halt_steps 3.0, halted_fraction 1.0
  4: mean_halt_steps 4.0, halted_fraction 1.0

strict adaptive-halt decision:
  criteria:
    accept_min_exact: 0.30
    accept_min_depth_gain: 0.10
    accept_min_ablation_drop: 0.03
    accept_min_family_exact: 0.25
    accept_max_adaptive_halt_exact_drop: 0.04
    accept_max_mean_halt_steps: 2.50
    accept_min_halted_fraction: 0.95
  accepted: true
  adaptive_halt_telemetry_source: generation
  full_minus_adaptive_halt: 0.03125
  adaptive_halt_mean_steps: 2.1875
  adaptive_halt_halted_fraction: 1.0

Comparison:
  last-token halt full context: generation_exec_steps 3.1632, rejected
  mean-pooled halt full context: generation_exec_steps 3.7569, rejected
  dedicated halt state: generation_exec_steps 2.1875, accepted under the
  diagnostic adaptive-halt runtime gate above
```

Dedicated halt seed-stability check:

```text
Shared criteria:
  accept_min_exact: 0.30
  accept_min_depth_gain: 0.10
  accept_min_ablation_drop: 0.03
  accept_min_family_exact: 0.25
  accept_max_adaptive_halt_exact_drop: 0.04
  accept_max_mean_halt_steps: 2.50
  accept_min_halted_fraction: 0.95

seed 337:
  report: local_eval/qtrm_native_adaptive_halt_dedicated_fullctx_s1200_20260513
  strict_decision: accepted
  full_exact: 0.3645833333
  adaptive_exact: 0.3333333333
  generation_exec_steps: 2.1875
  generation_halted: 1.0
  state_reset_exact: 0.328125
  full_minus_worst_ablation: 0.0364583333

seed 338:
  report: local_eval/qtrm_native_adaptive_halt_dedicated_fullctx_s1200_seed338_20260513
  strict_decision: rejected
  reject_reason: ablation_drop_below_threshold
  full_exact: 0.3333333333
  adaptive_exact: 0.3541666667
  generation_exec_steps: 2.1875
  generation_halted: 1.0
  state_reset_exact: 0.3697916667
  full_minus_worst_ablation: -0.0364583333

seed 339:
  report: local_eval/qtrm_native_adaptive_halt_dedicated_fullctx_s1200_seed339_20260513
  strict_decision: rejected
  reject_reason: ablation_drop_below_threshold
  full_exact: 0.3125
  adaptive_exact: 0.3385416667
  generation_exec_steps: 2.1875
  generation_halted: 1.0
  state_reset_exact: 0.3385416667
  full_minus_worst_ablation: -0.0260416667
```

Interpretation:

```text
The default halt head is initialized to continue, so adaptive halt does not
silently remove reasoning depth before it has been trained.

This is the correct safety default. It means the mechanism is wired, but it is
not yet evidence that the model learned when to stop. The 400-step diagnostic
shows the same: the recursive core still uses all 4 thinking steps even when
the teacher-depth halt loss is enabled.

The active-length batch-mix experiment is the first useful adaptive halt result:
fixed-depth and adaptive-halt generation exact match while sample mean halt
steps drop from 4 to 1.54 in teacher-forced batch evaluation. However, actual
autoregressive generation still averages 3.16 executed thinking steps and only
halts on about 32% of token-generation calls. This is not yet an adaptive-halt
promotion.

Prompt/prefix context training did not fix this. The halt head's state source
was the important architecture issue: last-token and mean-pooled LM-token halt
heads were too sensitive to answer-prefix changes. A dedicated recurrent halt
state is the first variant that passes the diagnostic generation-path adaptive
halt gate.

It is still not a full raw-reasoning promotion. The dedicated halt runtime is
stable across seeds, but recurrent carry is not: seed 338/339 have state_reset
matching or beating the full recurrent path. This means the model has learned a
useful runtime-control policy, but the recurrent state transition itself is not
yet reliably better than repeated reset passes.

Use the strict adaptive-halt gate only for runtime-control claims. Do not let it
replace the raw recursive-reasoning gate, which still needs causal depth/state
ablation gains.
```

Next required gate:

```text
Scale adaptive halt beyond the diagnostic:
  mix active lengths inside every batch
  keep first-step halt targets for active_len metadata
  do not rely on threshold sweep alone
  use the dedicated halt state as the current best runtime-control scaffold
  verify actual per-sample generation compute, not only full-batch telemetry
  enlarge the task scale and require stronger core/state ablation drop before
  calling it raw reasoning

Next raw-intelligence bottleneck:
  recurrent carry must beat state_reset across seeds. Until that holds, the
  architecture has adaptive compute control, but not a stable recursive-state
  reasoning advantage.

Fixed-base recurrent carry correction:

```text
Do not continue architecture shopping. The fixed base is:

  token ids
  -> native MHA encode
  -> official TRM thinking block
  -> dual z_L/z_H recurrent state
  -> dedicated halt state
  -> native MHA decode
  -> LM logits

The failed variant was trm_dual_z + dedicated halt. Its halt runtime was stable,
but seed 338/339 failed because state_reset matched or beat full recurrence.

Direct linear L-H coupling was tested as trm_dual_z_coupled. It improved seed
338 full exact, but coupling_off matched or slightly beat full, so the direct
coupling residual is not canonical evidence. KISS decision: remove that direct
coupling knob and keep only the useful interactive dual-z state update:

  z_L <- TRM(z_L + z_H + encoded)
  z_H <- TRM(z_H + z_L)

Implementation:
  scripts/335_train_qtrm_native_etd_probe.py
  think_structure: trm_dual_z_interactive

Unit verification:
  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
    scripts/335_train_qtrm_native_etd_probe.py \
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
    tests.test_qtrm_native_etd_probe \
    tests.test_qtrm_native_mixed_text_reasoning_probe

  result after interactive base + active_len diagnostics:
    79 tests OK

Seed 338 strict result:
  report:
    local_eval/qtrm_native_adaptive_halt_interactive_fullctx_s1200_seed338_depth025_20260513
  settings:
    think_structure: trm_dual_z_interactive
    halt_pooling: dedicated
    depth_intermediate_loss_weight: 0.25
    adaptive_halt_loss_weight: 5.0
    active_len_batch_cycle: true
    eval_active_len_cycle: true
  decision: accepted
  full_generation_exact: 0.3802083333
  think0_generation_exact: 0.0833333333
  full_minus_think0: 0.296875
  state_reset_generation_exact: 0.3489583333
  op_zero_generation_exact: 0.234375
  z_l_zero_generation_exact: 0.0
  z_h_zero_generation_exact: 0.0
  full_minus_worst_ablation: 0.03125
  adaptive_halt_generation_exact: 0.359375
  full_minus_adaptive_halt: 0.0208333333
  adaptive_halt_mean_steps: 2.1875
  adaptive_halt_halted_fraction: 1.0

Depth-loss sweep on the old coupled base:
  depth 0.5:
    full 0.421875, state_reset 0.40625, coupling_off 0.427083, adaptive 0.416667
    rejected: coupling_off beats full
  depth 0.25:
    full 0.390625, state_reset 0.385417, coupling_off 0.385417, adaptive 0.369792
    rejected: ablation gap only 0.005208
  depth 0.125:
    full 0.359375, state_reset 0.3125, coupling_off 0.354167, adaptive 0.322917
    rejected: ablation gap only 0.005208
  depth 0.0:
    full 0.302083, state_reset 0.166667, coupling_off 0.302083, adaptive 0.1875
    rejected: adaptive halt exact drop too large

Interpretation:
  The intermediate depth loss cannot be simply removed. It is needed to keep
  early-halt answer quality, but too much of it makes reset/shallow paths too
  strong. The first accepted fixed-base setting is interactive dual-z with
  depth_intermediate_loss_weight=0.25.

Seed sweep after fixing the base:
  seed 337, 1200 steps:
    full 0.3229167
    state_reset 0.3125
    op_zero 0.2291667
    adaptive 0.328125
    rejected: ablation gap only 0.0104167
  seed 338, 1200 steps:
    full 0.3802083
    state_reset 0.3489583
    op_zero 0.234375
    adaptive 0.359375
    accepted
  seed 339, 1200 steps:
    full 0.3385417
    state_reset 0.3125
    op_zero 0.2447917
    adaptive 0.3229167
    rejected: ablation gap only 0.0260417
  seed 337, 2400 steps:
    full 0.4114583
    state_reset 0.3958333
    op_zero 0.2291667
    adaptive 0.40625
    rejected: ablation gap only 0.015625

Current base decision:
  Keep trm_dual_z_interactive + dedicated halt as the fixed base.
  Do not keep shopping among Mamba3, GatedDeltaNet, Qwen hybrid, coupled
  residual, or attention-update variants for this gate. They are parked until
  the fixed base proves hard recursive reasoning.

Why the aggregate score was misleading:
  The 2400-step seed-337 run looked stronger in aggregate, but active_len
  breakdown showed that easy active_len 0/1 examples dominated:

    full active_len 0: 1.0
    full active_len 1: 0.8718
    full active_len 2: 0.1053
    full active_len 3: 0.0526
    full active_len 4: 0.0

  Hard-only active_len 2..4 training/eval exposed the real bottleneck:

    report:
      local_eval/qtrm_native_adaptive_halt_interactive_hardlen24_s2400_seed337_depth025_20260513
    full_generation_exact: 0.078125
    think0_generation_exact: 0.0052083
    state_reset_generation_exact: 0.03125
    op_zero_generation_exact: 0.0416667
    z_l_zero_generation_exact: 0.0
    z_h_zero_generation_exact: 0.0
    adaptive_halt_generation_exact: 0.0572917
    adaptive_halt_mean_steps: 3.0
    adaptive_halt_halted_fraction: 1.0

  Longer fixed-base hard-only training improved the same metric without
  changing architecture:

    report:
      local_eval/qtrm_native_adaptive_halt_interactive_hardlen24_s8000_seed337_depth025_20260513
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      adaptive_halt_exact_drop_above_threshold
    full_generation_exact:
      0.2552083
    think0_generation_exact:
      0.0
    state_reset_generation_exact:
      0.0260417
    op_zero_generation_exact:
      0.046875
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    adaptive_halt_generation_exact:
      0.1770833
    adaptive_halt_mean_steps:
      3.0
    hard active_len exact:
      active_len 2: 0.515625
      active_len 3: 0.140625
      active_len 4: 0.109375

  Correctness-target halt was also tested on the same fixed base:

    report:
      local_eval/qtrm_native_adaptive_halt_interactive_hardlen24_s8000_seed337_correcthalt_20260513
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      adaptive_halt_mean_steps_above_threshold
      adaptive_halt_fraction_below_threshold
    full_generation_exact:
      0.203125
    adaptive_halt_generation_exact:
      0.1875
    full_minus_adaptive_halt:
      0.015625
    adaptive_halt_mean_steps:
      3.7395833
    adaptive_halt_halted_fraction:
      0.2361111

  Interpretation of the correctness-target halt:
    It reduces the adaptive-vs-fixed quality drop, but it becomes too
    conservative and also hurts full-depth learning. It is not the next default.
    Keep active_len halt for now and improve the recursive core so answers are
    already stable near the intended halt step.

  Resume support was added to avoid restarting the same fixed-base experiment:

    script:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
    flag:
      --resume-from <previous last.pt>
    verification:
      tests.test_qtrm_native_mixed_text_reasoning_probe
      34 tests OK
    reproduction wrapper:
      PROFILE=standard SEED=337 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    broader-family wrapper option:
      TASK_FAMILIES=modchain,revchain,checksum
    length-scaling wrapper options:
      PROGRAM_LEN=6
      THINK_STEPS=6
      ACTIVE_LEN_MIN=3
      ACTIVE_LEN_MAX=6
      ACCEPT_MAX_MEAN_HALT_STEPS=4.6
      TRM_L_CYCLES=2
      D_MODEL=96
      D_FF=192
    smoke wiring check:
      PROFILE=smoke OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_smoke_20260513_rerun3 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
      result:
        accepted_l4_mixed_text_reasoning
    broader-family smoke:
      PROFILE=smoke TASK_FAMILIES=modchain,revchain,checksum \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_multifamily_smoke_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
      result:
        wiring OK
    length-6 broader-family smoke:
      PROFILE=smoke PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        TASK_FAMILIES=modchain,revchain,checksum \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_multifamily_smoke_20260513_rerun \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
      result:
        wiring OK
    length-6 low-level-cycle smoke:
      PROFILE=smoke PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        TRM_L_CYCLES=2 TASK_FAMILIES=modchain \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_lcycles2_smoke_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
      result:
        wiring OK
    eval-grid correction:
      apply_eval_active_len_cycle now balances active_len independently per
      family. This avoids the bad evaluation pattern where a 3-family order and
      a 3-length active_len cycle accidentally bind one family to one length.
      verification:
        tests.test_qtrm_native_mixed_text_reasoning_probe
        35 tests OK
    wrapper behavior:
      standard runs execute stage1 active_len, stage2 lower-LR resume, and
      stage3 halt-depth final loss. If stage3 narrowly fails the strict gate
      but writes a report, the wrapper runs a stage4 2000-step halt-depth refine
      and treats that report as the final result.

  Resuming the active_len checkpoint with lower LR improved the fixed-depth
  hard score but still failed adaptive drop:

    report:
      local_eval/qtrm_native_adaptive_halt_interactive_hardlen24_s16000_seed337_depth025_resume_20260513
    decision:
      rejected
    reject_reasons:
      adaptive_halt_exact_drop_above_threshold
    full_generation_exact:
      0.3697917
    adaptive_halt_generation_exact:
      0.2916667
    full_minus_adaptive_halt:
      0.078125
    state_reset_generation_exact:
      0.0364583
    op_zero_generation_exact:
      0.0364583
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0

  Threshold sweep on that checkpoint did not change the adaptive result for
  halt_threshold 0.6, 0.7, 0.8, or 0.9. The halt logits were saturated, so the
  issue was not runtime thresholding.

  The accepted repair was a halt-depth final-answer loss:

    idea:
      At each sample's intended halt depth, train the model to produce the
      final answer directly. This keeps the same fixed base and strengthens the
      causal LM path used by adaptive halt.
    implementation:
      halt_depth_final_answer_loss(...)
      --halt-depth-final-loss-weight
    report:
      local_eval/qtrm_native_adaptive_halt_interactive_hardlen24_s20000_seed337_haltdepth1_resume_20260513
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.4270833
    adaptive_halt_generation_exact:
      0.4010417
    full_minus_adaptive_halt:
      0.0260417
    adaptive_halt_mean_steps:
      3.0
    adaptive_halt_halted_fraction:
      1.0
    think0_generation_exact:
      0.0364583
    state_reset_generation_exact:
      0.0364583
    op_zero_generation_exact:
      0.0260417
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    hard active_len exact:
      full active_len 2: 0.875
      full active_len 3: 0.28125
      full active_len 4: 0.125
      adaptive active_len 2: 0.796875
      adaptive active_len 3: 0.28125
      adaptive active_len 4: 0.125

  Seed 338 using the same standard wrapper also passed:

    command:
      SEED=338 EVAL_SEED=9337 OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_seed338_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_seed338_20260513/stage3_halt_depth_s4000/report.json
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.6354167
    adaptive_halt_generation_exact:
      0.5989583
    full_minus_adaptive_halt:
      0.0364583
    adaptive_halt_mean_steps:
      3.0
    adaptive_halt_halted_fraction:
      1.0
    think0_generation_exact:
      0.0416667
    state_reset_generation_exact:
      0.03125
    op_zero_generation_exact:
      0.0052083
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    hard active_len exact:
      full active_len 2: 0.9375
      full active_len 3: 0.703125
      full active_len 4: 0.265625
      adaptive active_len 2: 0.953125
      adaptive active_len 3: 0.578125
      adaptive active_len 4: 0.265625

  Seed 339 required the automatic refine stage and then passed:

    command:
      SEED=339 EVAL_SEED=9337 OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_seed339_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    stage3:
      rejected only by adaptive_halt_exact_drop_above_threshold
      full_minus_adaptive_halt: 0.0416667
    final report:
      local_eval/qtrm_native_hardlen_recipe_seed339_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.4947917
    adaptive_halt_generation_exact:
      0.5
    full_minus_adaptive_halt:
      -0.0052083
    adaptive_halt_mean_steps:
      3.0
    adaptive_halt_halted_fraction:
      1.0
    think0_generation_exact:
      0.0
    state_reset_generation_exact:
      0.03125
    op_zero_generation_exact:
      0.0416667
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    hard active_len exact:
      full active_len 2: 0.90625
      full active_len 3: 0.421875
      full active_len 4: 0.15625
      adaptive active_len 2: 0.890625
      adaptive active_len 3: 0.453125
      adaptive active_len 4: 0.15625

  Broader family gate after the eval-grid correction:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain,revchain,checksum \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_multifamily_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    corrected grid eval report:
      local_eval/qtrm_native_hardlen_recipe_multifamily_seed337_20260513/stage3_halt_depth_s4000_grid_eval/report.json
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.5677083
    adaptive_halt_generation_exact:
      0.5364583
    full_minus_adaptive_halt:
      0.03125
    adaptive_halt_mean_steps:
      2.984375
    adaptive_halt_halted_fraction:
      1.0
    min_family_generation_exact:
      0.34375
    state_reset_generation_exact:
      0.0208333
    op_zero_generation_exact:
      0.0416667
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by family:
      checksum full/adaptive: 0.96875 / 0.984375
      modchain full/adaptive: 0.390625 / 0.328125
      revchain full/adaptive: 0.34375 / 0.296875
    by active_len:
      active_len 2 full/adaptive: 0.7424242 / 0.6515152
      active_len 3 full/adaptive: 0.5555556 / 0.5555556
      active_len 4 full/adaptive: 0.3968254 / 0.3968254

  Broader family seed 338 also passed after the automatic refine stage:

    command:
      SEED=338 EVAL_SEED=9337 TASK_FAMILIES=modchain,revchain,checksum \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_multifamily_seed338_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_multifamily_seed338_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.5260417
    adaptive_halt_generation_exact:
      0.4947917
    full_minus_adaptive_halt:
      0.03125
    adaptive_halt_mean_steps:
      2.984375
    adaptive_halt_halted_fraction:
      1.0
    min_family_generation_exact:
      0.28125
    state_reset_generation_exact:
      0.0208333
    op_zero_generation_exact:
      0.03125
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by family:
      checksum full/adaptive: 0.96875 / 0.9375
      modchain full/adaptive: 0.328125 / 0.296875
      revchain full/adaptive: 0.28125 / 0.25

  Broader family seed 339 also passed after the automatic refine stage:

    command:
      SEED=339 EVAL_SEED=9337 TASK_FAMILIES=modchain,revchain,checksum \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_multifamily_seed339_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_multifamily_seed339_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      accepted_l4_mixed_text_reasoning
    full_generation_exact:
      0.546875
    adaptive_halt_generation_exact:
      0.5416667
    full_minus_adaptive_halt:
      0.0052083
    adaptive_halt_mean_steps:
      2.984375
    adaptive_halt_halted_fraction:
      1.0
    min_family_generation_exact:
      0.3125
    state_reset_generation_exact:
      0.0208333
    op_zero_generation_exact:
      0.0416667
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by family:
      checksum full/adaptive: 0.984375 / 0.953125
      modchain full/adaptive: 0.3125 / 0.328125
      revchain full/adaptive: 0.34375 / 0.34375

  Broader-family seed stability summary:

    seeds:
      337, 338, 339
    pass_count:
      3 / 3
    full_generation_exact:
      0.5677083, 0.5260417, 0.546875
    adaptive_halt_generation_exact:
      0.5364583, 0.4947917, 0.5416667
    min_family_generation_exact:
      0.34375, 0.28125, 0.3125
    adaptive_halt_mean_steps:
      2.984375 for all three final reports
    causal ablations:
      state_reset <= 0.0208333
      op_zero <= 0.0416667
      z_l_zero = 0.0
      z_h_zero = 0.0

  Length-6 broader-family scaling attempt:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain,revchain,checksum \
        PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        ACCEPT_MAX_MEAN_HALT_STEPS=4.6 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_multifamily_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_len6_multifamily_seed337_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      rejected
    reject_reasons:
      family_exact_below_threshold
    full_generation_exact:
      0.3854167
    adaptive_halt_generation_exact:
      0.3645833
    full_minus_adaptive_halt:
      0.0208333
    adaptive_halt_mean_steps:
      4.5
    min_family_generation_exact:
      0.078125
    by family:
      checksum full/adaptive: 0.984375 / 0.96875
      modchain full/adaptive: 0.078125 / 0.046875
      revchain full/adaptive: 0.09375 / 0.078125
    interpretation:
      Length scaling preserves causal ablation and adaptive halt behavior, but
      the model collapses to the easy checksum family and fails hard sequential
      transformation families.

  Length-6 hard-family oversampling refine:

    command:
      resume from the length-6 stage4 checkpoint, train with
      TASK_FAMILIES=modchain,revchain,modchain,revchain,checksum and
      EVAL_TASK_FAMILIES=modchain,revchain,checksum
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_multifamily_seed337_20260513/stage5_hard_family_refine_s4000/report.json
    decision:
      rejected
    reject_reasons:
      family_exact_below_threshold
    full_generation_exact:
      0.3645833
    adaptive_halt_generation_exact:
      0.3333333
    min_family_generation_exact:
      0.0625
    by family:
      checksum full/adaptive: 0.953125 / 0.921875
      modchain full/adaptive: 0.078125 / 0.03125
      revchain full/adaptive: 0.0625 / 0.046875
    interpretation:
      Simple oversampling of hard families did not fix length-6. The next
      diagnostic must separate family interference from raw capacity/depth by
      training length-6 modchain and revchain alone under the same causal gate.

  Length-6 modchain-only diagnostic:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain \
        PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        ACCEPT_MAX_MEAN_HALT_STEPS=4.6 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_modchain_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_seed337_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      family_exact_below_threshold
    full_generation_exact:
      0.1614583
    adaptive_halt_generation_exact:
      0.1666667
    adaptive_halt_mean_steps:
      4.5
    state_reset_generation_exact:
      0.0416667
    op_zero_generation_exact:
      0.0260417
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by active_len:
      active_len 3 full/adaptive: 0.3333333 / 0.3541667
      active_len 4 full/adaptive: 0.1666667 / 0.1458333
      active_len 5 full/adaptive: 0.0833333 / 0.1041667
      active_len 6 full/adaptive: 0.0625 / 0.0625
    interpretation:
      Length-6 failure is not only multi-family interference. Modchain alone is
      weak beyond active_len 3 at the current d_model=64 / think6 setting.
      The next controlled axis is capacity/depth, not a new backbone.

  Length-6 modchain-only width diagnostic:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain \
        PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        ACCEPT_MAX_MEAN_HALT_STEPS=4.6 D_MODEL=96 D_FF=192 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      rejected
    full_generation_exact:
      0.1822917
    adaptive_halt_generation_exact:
      0.2083333
    state_reset_generation_exact:
      0.0416667
    op_zero_generation_exact:
      0.0364583
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by active_len:
      active_len 3 full/adaptive: 0.3125 / 0.4166667
      active_len 4 full/adaptive: 0.1666667 / 0.2291667
      active_len 5 full/adaptive: 0.1666667 / 0.1041667
      active_len 6 full/adaptive: 0.0833333 / 0.0833333
    interpretation:
      Increasing width from d_model=64 to 96 is only a small improvement and
      does not solve length-6 modchain. The next more TRM-specific axis is
      low-level recurrent cycles, not more backbone shopping.

  Length-6 modchain-only low-level cycle diagnostic:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain \
        PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        ACCEPT_MAX_MEAN_HALT_STEPS=4.6 TRM_L_CYCLES=2 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_modchain_lcycles2_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_lcycles2_seed337_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      rejected
    full_generation_exact:
      0.1354167
    adaptive_halt_generation_exact:
      0.1302083
    state_reset_generation_exact:
      0.0364583
    op_zero_generation_exact:
      0.0416667
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by active_len:
      active_len 3 full/adaptive: 0.2916667 / 0.2291667
      active_len 4 full/adaptive: 0.1666667 / 0.1666667
      active_len 5 full/adaptive: 0.0416667 / 0.0833333
      active_len 6 full/adaptive: 0.0416667 / 0.0416667
    interpretation:
      Increasing low-level cycles from 1 to 2 did not solve length-6 and was
      worse than the d_model=96 width check. The current recipe is likely
      missing a curriculum/optimization ingredient for longer transformations.

  Length-6 modchain curriculum diagnostic:

    command:
      SEED=337 EVAL_SEED=9337 TASK_FAMILIES=modchain \
        PROGRAM_LEN=6 THINK_STEPS=6 ACTIVE_LEN_MIN=3 ACTIVE_LEN_MAX=6 \
        ACCEPT_MAX_MEAN_HALT_STEPS=4.6 \
        STAGE1_ACTIVE_LEN_MODE=curriculum STAGE2_ACTIVE_LEN_MODE=batch_cycle \
        STAGE3_ACTIVE_LEN_MODE=batch_cycle STAGE4_ACTIVE_LEN_MODE=batch_cycle \
        CURRICULUM_MIN=1 CURRICULUM_WARMUP_FRAC=0.75 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_len6_modchain_curriculum_seed337_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
    final report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_curriculum_seed337_20260513/stage4_halt_depth_refine_s2000/report.json
    decision:
      rejected
    full_generation_exact:
      0.1354167
    adaptive_halt_generation_exact:
      0.1354167
    state_reset_generation_exact:
      0.0416667
    op_zero_generation_exact:
      0.03125
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by active_len:
      active_len 3 full/adaptive: 0.1875 / 0.2291667
      active_len 4 full/adaptive: 0.1666667 / 0.2083333
      active_len 5 full/adaptive: 0.125 / 0.0416667
      active_len 6 full/adaptive: 0.0625 / 0.0625
    interpretation:
      Easy-to-hard curriculum pretraining did not improve length-6. The next
      controlled optimization axis is targeted long-length training: train on
      active_len 5..6 while evaluating the full 3..6 grid.

  Length-6 modchain train-len56 focused refine:

    command:
      resume from the d_model=96 stage4 checkpoint, keep evaluation on
      active_len 3..6, but train the active_len batch cycle only on 5..6
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000/report.json
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      family_exact_below_threshold
    full_generation_exact:
      0.1770833
    adaptive_halt_generation_exact:
      0.2135417
    state_reset_generation_exact:
      0.046875
    op_zero_generation_exact:
      0.03125
    z_l_zero_generation_exact:
      0.0
    z_h_zero_generation_exact:
      0.0
    by active_len:
      active_len 3 full/adaptive: 0.25 / 0.2916667
      active_len 4 full/adaptive: 0.125 / 0.2291667
      active_len 5 full/adaptive: 0.2083333 / 0.2083333
      active_len 6 full/adaptive: 0.125 / 0.125
    interpretation:
      Focusing training on active_len 5..6 slightly improves the hardest
      lengths relative to the earlier d96 run, but it does not lift the whole
      length-6 gate. The full/adaptive path remains causal because state_reset,
      op_zero, and z-state ablations drop, but the absolute reasoning accuracy
      is too low.

  Length-6 late-depth weighted intermediate-loss refine:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --depth-intermediate-min-depth and --depth-intermediate-weight-power.
      Defaults preserve the previous uniform intermediate-depth loss.
    command:
      resume from the train-len56 focused checkpoint, train 4000 steps with
      depth_intermediate_min_depth=4 and depth_intermediate_weight_power=1.0
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_late_depth_weighted_refine_s4000/report.json
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      depth_gain_below_threshold
      family_exact_below_threshold
      adaptive_halt_mean_steps_above_threshold
    full_generation_exact:
      0.0833333
    adaptive_halt_generation_exact:
      0.09375
    full_minus_think0:
      0.0625
    adaptive_halt_mean_steps:
      4.6232639
    generation_format_valid:
      think6: 1.0
      adaptive_halt: 1.0
    by active_len:
      active_len 3 full/adaptive: 0.1041667 / 0.0625
      active_len 4 full/adaptive: 0.0833333 / 0.125
      active_len 5 full/adaptive: 0.1041667 / 0.1458333
      active_len 6 full/adaptive: 0.0416667 / 0.0416667
    interpretation:
      Late-depth reweighting is not the missing ingredient. It preserves valid
      answer formatting but worsens exactness and depth gain, likely because
      strong intermediate supervision distorts the final recurrent trajectory.
      Keep the option as a diagnostic knob, but do not promote it as the next
      recipe.

  Answer-format telemetry diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now reports
      generation_format_valid globally, by family, by active_len, and per
      example. Valid answer format is exactly two digits plus newline.
    tests:
      PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
        tests.test_qtrm_native_mixed_text_reasoning_probe
      36 tests passed.
    length-6 format eval report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_format_eval/report.json
    length-6 result:
      think6 generation_format_valid: 1.0
      adaptive_halt generation_format_valid: 1.0
      think0 generation_format_valid: 0.53125
      thinking_block_off generation_format_valid: 0.53125
    accepted control report:
      local_eval/qtrm_native_hardlen_recipe_multifamily_seed337_20260513/stage3_halt_depth_s4000_format_eval/report.json
    accepted control result:
      think4 generation_format_valid: 1.0
      adaptive_halt generation_format_valid: 1.0
      full_generation_exact: 0.5677083
      adaptive_halt_generation_exact: 0.5364583
    interpretation:
      Current length-6 failure is not mainly a lexicalization or answer-format
      failure. Once the recursive core is active, the model emits valid
      two-digit answers, but it often emits the wrong two-digit answer. The
      bottleneck is therefore the learned recurrent state transition for longer
      sequential transformations, not the final text renderer.

  Length-6 depth-sweep diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --eval-depth-sweep. It reports exact/format-valid by depth, global best
      depth, and best depth by active_len.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_depth_sweep_eval_v2/report.json
    global exact_by_depth:
      depth 0: 0.015625
      depth 1: 0.046875
      depth 2: 0.0416667
      depth 3: 0.109375
      depth 4: 0.125
      depth 5: 0.1354167
      depth 6: 0.1770833
    global best_depth:
      6
    best depth by active_len:
      active_len 3: best depth 3, exact 0.2916667
      active_len 4: best depth 4, exact 0.2291667
      active_len 5: best depth 5, exact 0.2083333
      active_len 6: best depth 6, exact 0.125
    interpretation:
      The model is not suffering from an overthinking collapse where later
      recursion destroys an already-good answer. The best depth tracks the
      required active program length. Therefore early-exit/halt can save compute
      after the core becomes accurate, but it is not the primary fix for L6.
      The primary fix must improve per-step recurrent state-transition accuracy.

  Length-6 no-intermediate final-answer refine:

    command:
      resume from the train-len56 focused checkpoint, keep train active_len 5..6,
      disable depth_intermediate_loss, keep halt_depth_final_loss
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_no_intermediate_final_refine_s4000/report.json
    decision:
      rejected
    reject_reasons:
      full_exact_below_threshold
      depth_gain_below_threshold
      family_exact_below_threshold
      adaptive_halt_mean_steps_above_threshold
    full_generation_exact:
      0.109375
    adaptive_halt_generation_exact:
      0.1145833
    full_minus_think0:
      0.09375
    depth_sweep:
      best_depth: 6
      best_generation_exact: 0.109375
      best_minus_full: 0.0
    interpretation:
      Removing partial intermediate supervision does not solve L6 and is worse
      than the previous train-len56 focused checkpoint. Intermediate supervision
      is not the whole problem; the remaining bottleneck is still the quality
      of the learned per-step state transition.

  Number-aware tokenizer diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --tokenizer-mode char|number. Number mode tokenizes two-digit values such
      as "25" as one token and can limit extra numeric tokens with
      --number-tokenizer-max-value.
    wrapper:
      scripts/344_run_qtrm_native_hardlen_recipe.sh now accepts TOKENIZER_MODE,
      NUMBER_TOKENIZER_MAX_VALUE, and EVAL_DEPTH_SWEEP.
    smoke:
      PROFILE=smoke TOKENIZER_MODE=number NUMBER_TOKENIZER_MAX_VALUE=31 \
        EVAL_DEPTH_SWEEP=1 \
        OUT_ROOT=local_eval/qtrm_native_hardlen_recipe_number_token_smoke_20260513 \
        bash scripts/344_run_qtrm_native_hardlen_recipe.sh
      completed successfully.
    direct length-6 diagnostics:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_number_token_seed337_s4000/report.json
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_number_token_max31_seed337_s4000/report.json
    number token max99:
      vocab_size: 111
      prompt_len: 31
      answer_len: 2
      full_generation_exact: 0.0364583
      adaptive_halt_generation_exact: 0.046875
    number token max31:
      vocab_size: 43
      prompt_len: 31
      answer_len: 2
      full_generation_exact: 0.0520833
      adaptive_halt_generation_exact: 0.0520833
    interpretation:
      Atomic two-digit value tokens do not immediately solve length-6. The
      max31 version is better than max99 but still far below the char-token
      trained checkpoint. Tokenization remains a valid L5A comparison axis, but
      current evidence does not justify replacing the fixed char-token baseline
      for the L6 recurrent-transition investigation.

  Prior work update for the L6 bottleneck:

    downloaded references:
      references/papers/latent_reasoning/seq_vcr_iclr2025.pdf
      references/papers/latent_reasoning/lsrl_process_supervised_grpo_2025.pdf
      references/papers/latent_reasoning/ood_recursive_latent_space_reasoning_2026_openreview.pdf
    Seq-VCR:
      Relevance: arithmetic reasoning can fail because intermediate
      representations collapse. Candidate fixed-base diagnostic: measure
      variance/covariance of recurrent states across depth and add a light
      anti-collapse regularizer only if collapse is observed.
    LSRL:
      Relevance: latent-recurrent models benefit from process supervision on
      recurrent states, not only final-answer CE. Candidate fixed-base
      diagnostic: score each depth state with the existing exact/rank telemetry
      and train from dense per-depth rewards or weighted targets rather than a
      flat hand-set margin.
    OOD recursive latent-space reasoning:
      Relevance: input-adaptive recurrence, algorithmic supervision, anchored
      latent representations, and explicit error correction directly match the
      L6 state-transition bottleneck. Candidate fixed-base diagnostic: add a
      discrete/anchored latent-state probe and an error-correction pass, but
      only after proving state collapse or unstable state drift.

  State-trace collapse diagnostic:

    implementation:
      scripts/335_train_qtrm_native_etd_probe.py can now return
      core_state_trace_h/l from forward_with_runtime(return_state_trace=True).
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py exposes this
      via --eval-state-trace.
    L6 report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_state_trace_eval/report.json
    L6 z_h:
      variance by depth: 0.6583, 0.6819, 0.6080, 0.4591, 0.4089, 0.4149
      step delta norm: 9.4336, 7.2477, 5.5136, 3.6721, 2.9435
      consecutive cosine: 0.4282, 0.6605, 0.7889, 0.9050, 0.9416
    accepted L4 control:
      local_eval/qtrm_native_hardlen_recipe_multifamily_seed337_20260513/stage3_halt_depth_s4000_state_trace_eval/report.json
    L4 z_h:
      variance by depth: 0.6211, 0.7270, 0.7511, 0.7360
      step delta norm: 6.7153, 5.4859, 3.5958
      consecutive cosine: 0.5410, 0.6750, 0.8584
    interpretation:
      The rejected L6 checkpoint shows late-depth state convergence:
      z_h variance falls from about 0.66 to 0.41 and consecutive-depth cosine
      rises to 0.94. The accepted L4 control keeps higher variance through the
      final depth. This supports a Seq-VCR-style anti-collapse diagnostic as
      the next fixed-base training experiment.
    anti-collapse implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --state-trace-anti-collapse-loss-weight, --state-trace-min-variance, and
      --state-trace-min-delta-norm.
    anti-collapse reports:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_state_trace_anticollapse_w005_s2000/report.json
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_state_trace_anticollapse_w02_s1000/report.json
    anti-collapse results:
      weight 0.05 full/adaptive: 0.171875 / 0.1614583
      weight 0.20 full/adaptive: 0.171875 / 0.1875
    anti-collapse interpretation:
      The state-collapse diagnostic is real, but a simple variance/delta hinge
      regularizer does not beat the stage5 baseline. The next fixed-base idea
      should anchor states to semantic transition correctness rather than only
      forcing states to spread out.

  Teacher-forced vs greedy diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now reports
      teacher_forced_token_accuracy and teacher_forced_sequence_exact for the
      answer span.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_tf_eval/report.json
    think6:
      greedy generation_exact: 0.1770833
      teacher_forced_token_accuracy: 0.6111111
      teacher_forced_sequence_exact: 0.1770833
      teacher_forced_answer_loss: 0.8832551
    adaptive_halt:
      greedy generation_exact: 0.2135417
      teacher_forced_token_accuracy: 0.6111111
      teacher_forced_sequence_exact: 0.1770833
    interpretation:
      This is not primarily an exposure-bias or decoding problem. Under
      teacher forcing, the model's argmax answer sequence is correct at the
      same rate as greedy generation. The core/readout path needs better
      answer-logit separation, not a more elaborate decoder.

  Answer-token rank and margin-loss diagnostic:

    rank report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_rank_eval/report.json
    rank result:
      full_generation_exact: 0.1770833
      teacher_forced_mean_token_rank: 1.75
      teacher_forced_token_top3: 0.9131945
      teacher_forced_token_top5: 0.9930556
    interpretation:
      The correct answer tokens are usually near the top but not always rank 1.
      This made margin loss a plausible low-risk training diagnostic.
    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --answer-margin-loss-weight and --answer-margin.
    margin refine report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_answer_margin_refine_s4000/report.json
    margin refine result:
      decision: rejected
      full_generation_exact: 0.140625
      adaptive_halt_generation_exact: 0.15625
      teacher_forced_mean_token_rank: 1.796875
      teacher_forced_token_top3: 0.9131945
      teacher_forced_token_top5: 0.9895833
    interpretation:
      A strong answer-margin refine does not solve L6 and degrades the current
      best checkpoint. The rank signal is useful diagnostically, but the next
      recipe should not promote margin loss at this weight. If revisited, it
      must be a weaker auxiliary and compared against the stage5 baseline.
    weak margin report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_answer_margin_w01_refine_s2000/report.json
    weak margin result:
      decision: rejected
      full_generation_exact: 0.1197917
      adaptive_halt_generation_exact: 0.1666667
      teacher_forced_mean_token_rank: 1.7847222
      teacher_forced_token_top5: 0.9913195
    weak margin interpretation:
      Lowering answer-margin weight to 0.1 still degrades the baseline. Margin
      loss is therefore not the current shortest path.

  Plain continuation and periodic-eval checkpointing:

    plain continuation report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_s4000/report.json
    plain continuation result:
      decision: rejected
      full_generation_exact: 0.1354167
      adaptive_halt_generation_exact: 0.1666667
      teacher_forced_answer_loss: 0.8713582
      teacher_forced_sequence_exact: 0.1354167
    interpretation:
      Continuing the same recipe lowers teacher-forced loss but lowers exactness
      relative to the stage5 baseline. Train loss alone is not a reliable
      selection signal.
    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --eval-during-training-every, --eval-during-training-cases, and
      --restore-best-eval-checkpoint. Best checkpoint selection uses
      generation exact, then teacher-forced sequence exact, then lower token
      rank, then lower answer loss.
      scripts/344_run_qtrm_native_hardlen_recipe.sh exposes the same controls
      as EVAL_DURING_TRAINING_EVERY, EVAL_DURING_TRAINING_CASES, and
      RESTORE_BEST_EVAL_CHECKPOINT.
    smoke:
      local_eval/qtrm_native_periodic_eval_smoke_20260513_rerun/report.json
      accepted: true
      best_periodic_eval step: 4
      local_eval/qtrm_native_hardlen_recipe_periodic_smoke_20260513/stage3_halt_depth_s8/report.json
      accepted: true
      wrapper best_periodic_eval step: 8
    L6 periodic subset report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_periodic_s4000/report.json
    L6 periodic subset result:
      subset best_periodic_eval step: 3000
      subset best_generation_exact: 0.21875
      restored full_generation_exact: 0.1458333
    correction:
      For L6, 64-case periodic eval was not representative enough. The script
      now treats --eval-during-training-cases <= 0 as "use all eval cases" so
      future checkpoint selection can use the full held-out grid.
    L6 full-periodic report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_fullperiodic_s2000/report.json
    L6 full-periodic result:
      best_periodic_eval step: 1000
      best_periodic_eval exact: 0.1354167
      restored full_generation_exact: 0.1354167
      adaptive_halt_generation_exact: 0.1458333
    conclusion:
      Full-grid periodic selection confirms that continuing from the stage5
      checkpoint does not improve L6. The current L6 best remains the stage5
      train-len56 focused checkpoint.
    interpretation:
      Future long L6 runs should use periodic eval / best restore, because
      several refines show falling loss with falling exactness.

  Full-gradient recurrent-cycle diagnostic:

    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_fullgrad_cycles_s1000/report.json
    setup:
      Resume from the current stage5 best checkpoint, keep the same fixed base,
      and change only the TRM recurrent-cycle training mode with
      --trm-full-grad-cycles. This tests whether no-grad inner recurrence is
      the L6 bottleneck.
    result:
      decision: rejected
      full_generation_exact: 0.1510417
      adaptive_halt_generation_exact: 0.15625
      state_reset_generation_exact: 0.0416667
      op_zero_generation_exact: 0.03125
      teacher_forced_sequence_exact: 0.1510417
      teacher_forced_mean_token_rank: 1.7326389
    interpretation:
      Full-gradient cycles preserve causal core dependence but do not beat the
      stage5 baseline. The fixed base stays canonical; full-grad recurrence is
      not promoted as the next recipe.

  Prefix-depth anchor diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --prefix-depth-anchor-loss-weight, --prefix-depth-anchor-min-depth, and
      --prefix-depth-anchor-weight-power.
    purpose:
      Keep the same fixed base, but add a causal-prefix auxiliary loss: at
      depth d, train the same LM answer path on a prompt where operations after
      d are replaced by NOOP. This tests whether L6 fails because intermediate
      process supervision on the full prompt is contaminated by future
      operations rather than anchored to the stepwise transition.
    smoke:
      local_eval/qtrm_native_hardlen_recipe_prefix_anchor_smoke_20260513/stage3_halt_depth_s8/report.json
      accepted: true
    L6 report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_prefix_depth_anchor_w025_s1000/report.json
    L6 result:
      decision: rejected
      best_periodic_eval step: 500
      full_generation_exact: 0.171875
      adaptive_halt_generation_exact: 0.1770833
      state_reset_generation_exact: 0.0416667
      op_zero_generation_exact: 0.0208333
      teacher_forced_sequence_exact: 0.171875
      teacher_forced_mean_token_rank: 1.7013889
    interpretation:
      Prefix-depth anchoring is causal and valid, but it does not beat the
      stage5 baseline. The L6 bottleneck is not solved by adding more partial
      prefix CE alone. The next diagnostic should inspect whether failures are
      concentrated by operation id or operation position before adding more
      training signal.

  Operation-breakdown diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --eval-operation-breakdown. It records generation exactness by last
      operation id, by position:operation id, and by modular error delta.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_operation_breakdown_eval/report.json
    think6 by last op:
      op01: 0.2444444
      op02: 0.1481481
      op03: 0.1904762
      op04: 0.1818182
      op05: 0.1666667
      op06: 0.1304348
      op07: 0.1333333
    adaptive_halt by last op:
      op01: 0.3111111
      op02: 0.1851852
      op03: 0.2857143
      op04: 0.1818182
      op05: 0.125
      op06: 0.2173913
      op07: 0.1333333
    error-delta observation:
      For think6, wrong valid answers are heavily concentrated on even modular
      deltas. This suggests the model often preserves a coarse residue/parity
      signal but fails to keep the exact modular value through long
      transitions.
    interpretation:
      The next fixed-base data/training experiment should target exact-value
      transition fidelity rather than formatting, decoding, full-grad cycles,
      or additional prefix CE. A hard-op or residue-aware curriculum is allowed
      as a training diagnostic only if final evaluation remains the same normal
      answer-generation path.

  Hard-op oversampling diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --train-hard-op-ids and --train-hard-op-probability. The wrapper exposes
      TRAIN_HARD_OP_IDS and TRAIN_HARD_OP_PROBABILITY.

      It also supports --train-hard-op-positions, with wrapper env
      TRAIN_HARD_OP_POSITIONS, so hard-op oversampling can be restricted to
      selected 1-indexed operation positions. Empty positions keep the previous
      all-position behavior.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_hardop_02050607_p07_s1000/report.json
    setup:
      Resume from stage5, train 1000 steps with op02/op05/op06/op07 sampled
      with probability 0.7 in train cases, keep eval unchanged.
    result:
      decision: rejected
      full_generation_exact: 0.1770833
      adaptive_halt_generation_exact: 0.1510417
      active_len 5 full: 0.2916667
      active_len 6 full: 0.1041667
    interpretation:
      Hard-op oversampling does not improve the uniform eval gate. It shifts
      accuracy toward some lengths/ops but hurts adaptive behavior and does not
      improve full exact over the stage5 baseline.

  Residue auxiliary diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --residue-aux-loss-weight and --residue-aux-moduli. It uses fixed-width
      answer tags such as answer2, answer4, and answer8 so tokenizer chars and
      max sequence length remain compatible with the stage5 checkpoint. Final
      evaluation remains the canonical answer-space prompt.
    report 1:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_residue_aux_w02_s1000/report.json
    result 1:
      decision: rejected
      full_generation_exact: 0.1666667
      adaptive_halt_generation_exact: 0.1614583
      active_len 6 full: 0.1875
    report 2:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_residue_aux_w02_trainlen36_s1000/report.json
    result 2:
      decision: rejected
      full_generation_exact: 0.171875
      adaptive_halt_generation_exact: 0.1927083
      active_len 3 adaptive: 0.3541667
      active_len 6 adaptive: 0.1041667
    interpretation:
      Residue supervision is not useless: in the train-len56 variant it lifts
      active_len 6 full exact from 0.125 to 0.1875. But it does not improve the
      full held-out grid, and train-len36 mainly improves shorter adaptive
      cases while leaving active_len 6 weak. Do not promote residue aux as the
      next recipe unless it is paired with a better recurrent-state/readout
      coupling diagnostic.

  Beam/readout diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --eval-beam-width. Beam evaluation is diagnostic only; it does not change
      the accepted generation path.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_beam5_eval/report.json
    stage5 result:
      think6 greedy_generation_exact: 0.1770833
      think6 beam5_oracle_exact: 0.5364583
      think5 beam5_oracle_exact: 0.53125
      think4 beam5_oracle_exact: 0.34375
    interpretation:
      The correct answer is often present in a small candidate set, but the
      greedy LM readout ranks another candidate first. This is a real readout
      ranking bottleneck. However, an oracle beam is not canonical progress.
      Any promoted fix must teach the same recursive LM path to rank the
      correct sequence higher, not use an external verifier to choose answers.

  Sequence-preference diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --sequence-preference-loss-weight, --sequence-preference-deltas, and
      --sequence-preference-margin. It compares correct answer sequence
      log-probability against same-prompt rejected answers at modular deltas.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_seqpref_w01_s1000/report.json
    result:
      decision: rejected
      full_generation_exact: 0.1614583
      adaptive_halt_generation_exact: 0.1510417
      think6 beam5_oracle_exact: 0.5364583
    interpretation:
      Simple hand-picked modular-delta sequence preference does not convert the
      beam oracle headroom into greedy accuracy. The readout bottleneck remains,
      but the next preference attempt must use mined beam negatives or a better
      process/readout coupling signal; do not continue hand-picking deltas.

  Online greedy-preference diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      bounded online greedy negative mining:
      --online-greedy-preference-loss-weight
      --online-greedy-preference-margin
      --online-greedy-preference-max-cases
      --online-greedy-preference-every
      The bounded controls are required because mining every case in every
      batch is too slow for the fixed-base loop.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_online_greedy_pref_w005_bounded_s500/report.json
    result:
      decision: rejected
      full_generation_exact: 0.1614583
      adaptive_halt_generation_exact: 0.1458333
      adaptive_halt_mean_steps: 4.6059028
      state_reset_generation_exact: 0.046875
      op_zero_generation_exact: 0.0260417
      think6 beam5_oracle_exact: 0.5625
    interpretation:
      Online greedy negatives preserve the same causal LM path, but this short
      bounded run does not beat the stage5 baseline. It slightly increases
      beam-oracle headroom while lowering greedy exact, so the current problem
      is not just "seen wrong answer versus gold" preference. The bottleneck is
      still readout ranking/process-state alignment. Keep stage5 as canonical;
      do not promote online-greedy preference yet.

  Answer-space ranking diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      full answer-space ranking over the canonical LM sequence score:
      --answer-space-ranking-loss-weight
      --answer-space-ranking-max-cases
      --answer-space-ranking-every
      --answer-space-ranking-temperature
      For the mod32 probe this compares the gold answer against all 32 possible
      answer strings during training. It is a diagnostic readout-alignment
      signal, not an inference-time verifier or promoted architecture.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_answer_space_rank_w005_s500/report.json
    result:
      decision: rejected
      full_generation_exact: 0.1510417
      adaptive_halt_generation_exact: 0.1770833
      adaptive_halt_mean_steps: 4.6006944
      state_reset_generation_exact: 0.046875
      op_zero_generation_exact: 0.0260417
      think6 beam5_oracle_exact: 0.578125
      think6 teacher_forced_mean_token_rank: 1.7239584
    interpretation:
      The candidate set becomes richer, but greedy exact drops below the
      stage5 baseline. This says the current failure is not simply "gold
      answer has low teacher-forced score"; the model can place useful mass
      near the answer while autoregressive greedy still follows the wrong
      local token path. The next diagnostic should measure exact answer-space
      argmax from teacher-forced sequence scores and compare it with greedy and
      beam oracle. If answer-space argmax is high, the bottleneck is decoding
      path/local token commitment; if it is low, the bottleneck remains
      recurrent state -> readout score alignment.

  Answer-space argmax evaluation:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --eval-answer-space-argmax and --eval-answer-space-argmax-batch-size. It
      scores all 32 possible answer strings through the same LM sequence-score
      path and reports whether the gold answer is the argmax. This is an eval
      diagnostic only.
    stage5 report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_answer_space_argmax_eval/report.json
    stage5 result:
      greedy_generation_exact: 0.1770833
      answer_space_argmax_exact: 0.171875
      answer_space_gold_mean_rank: 6.0989585
      answer_space_gold_top5: 0.546875
      beam5_oracle_exact: 0.5364583
    answer-space-ranking checkpoint report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_answer_space_rank_w005_s500_argmax_eval/report.json
    answer-space-ranking checkpoint result:
      greedy_generation_exact: 0.1510417
      answer_space_argmax_exact: 0.1354167
      answer_space_gold_mean_rank: 5.8489585
      answer_space_gold_top5: 0.578125
      beam5_oracle_exact: 0.578125
    interpretation:
      The answer-space argmax does not beat greedy. Therefore the main failure
      is not just local autoregressive search. The gold answer is often in the
      top-5 mass, but the recursive state/readout path does not put it first.
      Training should focus on state-transition/readout alignment, not external
      beam selection.

  Prefix-state alignment diagnostic:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
      --prefix-state-alignment-loss-weight
      --prefix-state-alignment-max-cases
      --prefix-state-alignment-every
      It aligns the prompt-boundary recurrent state of a full prompt at depth d
      to the detached state of a prefix/noop prompt containing only the first d
      active operations. This is meant to teach the recursive core a stepwise
      process state without adding a sidecar answer solver.
    report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_prefix_state_align_w002_s500/report.json
    result:
      decision: rejected
      full_generation_exact: 0.171875
      adaptive_halt_generation_exact: 0.1354167
      adaptive_halt_mean_steps: 4.5902778
      state_reset_generation_exact: 0.046875
      op_zero_generation_exact: 0.0208333
      think6 beam5_oracle_exact: 0.5416667
      answer_space_argmax_exact: 0.1614583
    interpretation:
      The fixed core remains causally active, but this state-alignment loss
      does not beat stage5 and harms adaptive halt. The likely issue is that
      forcing hidden-state equality between full and prefix prompts conflicts
      with halt/readout specialization. Keep the implementation as a diagnostic
      tool, but do not promote this recipe. The next process-level attempt
      should avoid direct hidden equality and instead supervise observable
      depth outputs or halt-conditioned scores.

  Plain refine controls and adaptive-aware checkpoint selection:

    implementation:
      scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now records
      adaptive_halt_generation_exact, fixed_minus_adaptive_halt,
      adaptive_halt_mean_steps, and adaptive_halt_halted_fraction during
      periodic eval when --adaptive-halt-eval is enabled. periodic_eval_score
      now prefers checkpoints with good fixed/adaptive balance instead of
      fixed-depth accuracy alone.
    plain final-only report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_s1000/report.json
    plain final-only result:
      decision: rejected
      full_generation_exact: 0.15625
      adaptive_halt_generation_exact: 0.1614583
      active_len 6 full: 0.1458333
      interpretation: more low-lr training improves active_len 6 slightly but
      hurts other lengths, so average accuracy drops.
    fixed-only periodic restore report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_periodic_s1000/report.json
    fixed-only periodic restore result:
      decision: rejected
      best_periodic_step: 600
      full_generation_exact: 0.203125
      adaptive_halt_generation_exact: 0.15625
      interpretation: fixed-depth reasoning improved, but the old checkpoint
      selector ignored adaptive halt and picked a checkpoint that violates the
      strict adaptive gate.
    halt-threshold sweep:
      reports:
        stage6_plain_refine_periodic_s1000_halt_0.3/report.json
        stage6_plain_refine_periodic_s1000_halt_0.4/report.json
        stage6_plain_refine_periodic_s1000_halt_0.6/report.json
        stage6_plain_refine_periodic_s1000_halt_0.7/report.json
        stage6_plain_refine_periodic_s1000_halt_0.8/report.json
        stage6_plain_refine_periodic_s1000_halt_0.9/report.json
      result:
        adaptive_halt_generation_exact stayed around 0.1510-0.15625.
      interpretation:
        The adaptive failure was not a simple threshold choice.
    adaptive-aware periodic restore report:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_adaptiveaware_s1000/report.json
    adaptive-aware periodic restore result:
      decision: rejected
      best_periodic_step: 200
      periodic fixed/adaptive exact at best step: 0.21875 / 0.2395833
      final full_generation_exact: 0.1822917
      final adaptive_halt_generation_exact: 0.1927083
      final full_minus_adaptive_halt: -0.0104167
    adaptive-aware argmax eval:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_adaptiveaware_s1000_argmax_eval/report.json
      beam5_oracle_exact: 0.5625
      answer_space_argmax_exact: 0.1614583
    interpretation:
      The adaptive-aware selector fixed the selection bug and gives a small
      fixed-depth improvement over stage5, but it still does not beat the
      stage5 adaptive baseline of 0.2135417 and remains far below the 0.30
      strict target. Do not replace the canonical stage5 baseline yet. Keep
      adaptive-aware periodic selection as the default for future strict-gate
      training.

    active-len floor selection:
      implementation:
        periodic_eval_score now supports --periodic-eval-score-mode
        strict|active_floor. strict remains the default. active_floor promotes
        checkpoints by the worst active-length bucket before overall exact, so
        it can be used specifically when length tradeoff is the measured
        bottleneck.
      report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_activefloor_s1000/report.json
      result:
        decision: rejected
        best_periodic_step: 800
        full_generation_exact: 0.2083333
        adaptive_halt_generation_exact: 0.1822917
        active_len full exact:
          len3: 0.2916667
          len4: 0.1875
          len5: 0.2291667
          len6: 0.125
      interpretation:
        active_floor selection gives the best fixed-depth score observed in
        this stage and keeps the adaptive drop within the strict tolerance, but
        it still misses the 0.30 threshold and does not improve active_len6
        beyond the original stage5 value. This is useful as a fixed-depth
        diagnostic checkpoint, not a canonical replacement for stage5.

    len6-only refine control:
      report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_len6_only_refine_activefloor_s1000/report.json
      setup:
        Resume stage5, train only active_len 6 for 1000 steps, select periodic
        checkpoint by active_floor.
      result:
        decision: rejected
        best_periodic_step: 1000
        full_generation_exact: 0.15625
        adaptive_halt_generation_exact: 0.1770833
        active_len full exact:
          len3: 0.1458333
          len4: 0.1458333
          len5: 0.1666667
          len6: 0.1666667
      interpretation:
        The len6 transition is learnable: active_len6 improves from the stage5
        baseline 0.125 to 0.1666667. But focusing only on len6 erases shorter
        length performance and lowers the overall score. The next useful
        training signal is not more len6-only tuning; it is length-aware
        retention/distillation so len6 can improve without forgetting len3-5.

    reference-retention KL control:
      reports:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_len6_only_retention_w01_s1000/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_len6_only_retention_w01_haltfinal_s1000/report.json
      setup:
        Resume stage5, train only active_len 6, keep a frozen copy of the
        resumed checkpoint as the reference, and add KL retention on
        active_len 3..5 answer logits. The first run accidentally left
        halt_depth_final_loss_weight at 0.0, so the halt-final matched run is
        the fairer comparison to the previous controls.
      result:
        retention_w01_no_haltfinal:
          decision: rejected
          full_generation_exact: 0.1875
          adaptive_halt_generation_exact: 0.1666667
          active_len full exact:
            len3: 0.2291667
            len4: 0.125
            len5: 0.2708333
            len6: 0.125
        retention_w01_haltfinal:
          decision: rejected
          full_generation_exact: 0.15625
          adaptive_halt_generation_exact: 0.15625
          active_len full exact:
            len3: 0.1458333
            len4: 0.1041667
            len5: 0.2291667
            len6: 0.1458333
      interpretation:
        Reference-retention KL is a useful non-regression diagnostic, but this
        weak output-logit KL is not enough to solve the active_len6 bottleneck.
        It either preserves/reweights shorter cases without improving len6, or
        improves len6 slightly while still lowering total and adaptive exact.
        Do not promote these checkpoints. Keep stage5 as canonical and treat
        retention KL as optional training support, not an architecture change.

    active_len replay CE control:
      report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_len6_replayce_w05_s1000/report.json
      setup:
        Resume stage5, train active_len 6, and replay active_len 3..5 gold
        answer CE every two steps.
      result:
        decision: rejected
        full_generation_exact: 0.1510417
        adaptive_halt_generation_exact: 0.1666667
        active_len full exact:
          len3: 0.1875
          len4: 0.1041667
          len5: 0.1875
          len6: 0.125
      interpretation:
        Simple gold replay CE does not fix forgetting or len6. The bottleneck
        is not solved by adding more short-length answer CE on top of the same
        transition objective.

    core answer probe diagnostic:
      reports:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_core_answer_probe_h/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_core_answer_probe_both/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_core_answer_probe_both_flatten/report.json
      setup:
        Freeze stage5. Feed prompt-only text through the recursive core, then
        train a small linear probe from core latent state to the answer value.
        Probe variants use z_H last-token, z_H+z_L last-token, and flattened
        z_H+z_L over all prompt positions.
      result:
        z_H last-token probe:
          train_exact: 0.1640625
          eval_exact: 0.0677083
          len6_exact: 0.0416667
        z_H+z_L last-token probe:
          train_exact: 0.2270508
          eval_exact: 0.0520833
          len6_exact: 0.0625
        z_H+z_L flattened prompt probe:
          train_exact: 0.6914063
          eval_exact: 0.1041667
          len6_exact: 0.125
      interpretation:
        A linear probe can overfit train cases from flattened state but does
        not generalize. This weakens the hypothesis that the core already has
        a clean hidden answer and only the LM readout is bad. The stronger
        hypothesis is that held-out transition representation/generalization
        is the main bottleneck.

    core step probe diagnostic:
      implementation:
        scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now
        supports --eval-core-step-probe. It freezes the checkpoint, reads the
        recurrent state trace at every think depth, and trains a small linear
        probe to predict the intermediate prefix answer for that depth. It
        reuses the core-answer probe controls for state source, pooling,
        train/eval cases, probe steps, batch size, lr, and weight decay.
      reports:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_core_step_probe_both_last/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_core_step_probe_both_flatten/report.json
      result:
        z_H+z_L last-token step probe:
          train_exact: 0.2460124
          eval_exact: 0.1770833
          by_depth:
            depth1: 0.53125
            depth2: 0.2083333
            depth3: 0.1302083
            depth4: 0.0833333
            depth5: 0.0572917
            depth6: 0.0520833
        z_H+z_L flattened-prompt step probe:
          train_exact: 0.3883464
          eval_exact: 0.2560764
          by_depth:
            depth1: 0.6041667
            depth2: 0.3385417
            depth3: 0.203125
            depth4: 0.1145833
            depth5: 0.1458333
            depth6: 0.1302083
      interpretation:
        The recurrent trajectory contains some early-step information and some
        distributed prompt-level signal, but the depth-local state does not
        cleanly encode the intermediate answer at later recursion depths. This
        further supports "transition representation/readout coupling" as the
        root bottleneck. A better next architecture is a canonical internal
        scratch/state token or state codec that every recursive step must write
        through before the LM head reads the answer, rather than more hard-op
        sampling or output-only ranking loss.

    core-step codec training control:
      implementation:
        scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now
        supports a training-only --core-step-codec-loss-weight with
        --core-step-codec-state-source and --core-step-codec-pooling. The codec
        reads only prompt-prefix recurrent state from the state trace and
        predicts the intermediate prefix answer for each depth. It is not used
        at inference and does not replace the canonical LM-logit answer path.
      reports:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_core_step_codec_w01_fullgrad_s1000/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_core_step_codec_w003_fullgrad_s1000/report.json
      result:
        w0.10 full-grad:
          decision: rejected
          full_generation_exact: 0.1979167
          adaptive_halt_generation_exact: 0.171875
          active_len full exact:
            len3: 0.2083333
            len4: 0.1875
            len5: 0.25
            len6: 0.1458333
          core_step_probe_exact: 0.1788194
          core_step_probe depth6: 0.078125
        w0.03 full-grad:
          decision: rejected
          full_generation_exact: 0.15625
          adaptive_halt_generation_exact: 0.1927083
          active_len full exact:
            len3: 0.1458333
            len4: 0.1458333
            len5: 0.2083333
            len6: 0.125
          core_step_probe_exact: 0.1597222
          core_step_probe depth6: 0.0572917
      interpretation:
        A training-only codec can slightly lift fixed-depth exact at w0.10, but
        it does not make the recurrent state linearly readable at late depths
        and it hurts adaptive halt. Lower weight is worse. Therefore the next
        improvement should be structural: make a dedicated causal scratch/read
        position or state-aggregation path part of the recurrent LM path,
        instead of relying on an auxiliary head to discover a clean state inside
        the existing distributed prompt-token trajectory.

    causal prefix-scratch architecture control:
      implementation:
        scripts/335_train_qtrm_native_etd_probe.py now includes
        think_structure=trm_dual_z_interactive_prefix_scratch. It keeps the
        TRM dual-z interactive update, then applies a causal prefix aggregation
        to z_H through a small gated scratch projection. This is inside the
        normal causal LM path: no retrieval, no side answer channel, no external
        verifier. scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
        also supports --resume-allow-missing for additive architecture probes,
        so the stage5 checkpoint can initialize shared parameters while the new
        scratch parameters start fresh.
      report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_prefix_scratch_resume_s1000/report.json
      result:
        decision: rejected
        full_generation_exact: 0.15625
        adaptive_halt_generation_exact: 0.1614583
        adaptive_halt_mean_steps: 4.7083333
        active_len full exact:
          len3: 0.25
          len4: 0.0833333
          len5: 0.1875
          len6: 0.1041667
        core_step_probe_exact: 0.1727431
        core_step_probe by depth:
          depth1: 0.5
          depth2: 0.2135417
          depth3: 0.1041667
          depth4: 0.078125
          depth5: 0.0572917
          depth6: 0.0833333
      interpretation:
        The naive prefix-scratch aggregation hurts both fixed and adaptive
        generation compared with canonical stage5, and it does not repair
        late-depth step-state readability. Do not promote this architecture.
        The failure is still useful: simply averaging/projecting prefix state
        after the TRM update is too blunt. A future structural attempt needs a
        more targeted recurrent state carrier, likely trained from scratch or
        with a staged warmup, rather than an additive scratch module bolted onto
        a mature checkpoint.

    interactive residual-readout architecture control:
      implementation:
        scripts/335_train_qtrm_native_etd_probe.py now includes
        think_structure=trm_dual_z_interactive_residual_readout. It keeps the
        interactive z_L/z_H recurrent update, but final readout becomes
        LayerNorm(encoded + alpha * (z_L + z_H)) before the normal decoder and
        LM head. This was tested because core-step probes suggested weak
        state/readout coupling.
      report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_interactive_residual_readout_resume_s1000/report.json
      result:
        decision: rejected
        reject_reasons:
          full_exact_below_threshold
          depth_gain_below_threshold
          ablation_drop_below_threshold
        full_generation_exact: 0.125
        adaptive_halt_generation_exact: 0.1041667
        full_minus_think0: 0.078125
        full_minus_worst_ablation: 0.0625
        active_len full exact:
          len3: 0.1458333
          len4: 0.125
          len5: 0.1041667
          len6: 0.125
        core_step_probe_exact: 0.1449653
        core_step_probe depth6: 0.015625
      interpretation:
        Residual readout is not the answer. It creates an encoded shortcut
        that weakens the measured causal contribution of the recursive z-state
        and hurts generation. Keep the canonical stage5 interactive readout.
        Future readout changes must increase core-dependent accuracy without
        weakening z_l_zero/z_h_zero/state_reset ablation drops.

    visible prompt state-anchor control:
      implementation:
        scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
        --prompt-state-anchor. This changes the normal visible prompt from
        "ops ... answer " to "ops ... state answer ". It is not a hidden
        evidence channel or renderer; it remains inside the same tokenizer ->
        embeddings -> recursive core -> decoder -> LM logits path. The wrapper
        script supports PROMPT_STATE_ANCHOR=1.
      reports:
        local_eval/qtrm_native_prompt_anchor_compare_20260513/baseline_no_anchor_s1200/report.json
        local_eval/qtrm_native_prompt_anchor_compare_20260513/state_anchor_s1200/report.json
        local_eval/qtrm_native_prompt_anchor_compare_20260513/tail_state_anchor_s1200/report.json
        local_eval/qtrm_native_prompt_anchor_compare_20260513/state_anchor_flatten_probe_eval/report.json
      matched short result:
        baseline no-anchor:
          decision: rejected
          full_generation_exact: 0.0520833
          think0_generation_exact: 0.0052083
          state_reset_generation_exact: 0.046875
          op_zero_generation_exact: 0.0520833
          adaptive_halt_generation_exact: 0.0520833
          full_minus_worst_ablation: 0.0
          active_len full exact:
            len3: 0.0625
            len4: 0.0208333
            len5: 0.0416667
            len6: 0.0833333
          core_step_probe_exact: 0.0355903
        visible state-anchor:
          decision: rejected
          full_generation_exact: 0.0625
          think0_generation_exact: 0.0260417
          state_reset_generation_exact: 0.0208333
          op_zero_generation_exact: 0.03125
          adaptive_halt_generation_exact: 0.03125
          full_minus_worst_ablation: 0.03125
          active_len full exact:
            len3: 0.0833333
            len4: 0.1041667
            len5: 0.0416667
            len6: 0.0208333
          core_step_probe_exact: 0.0295139
        tail visible state-anchor ("answer state "):
          decision: rejected
          full_generation_exact: 0.0520833
          think0_generation_exact: 0.0208333
          state_reset_generation_exact: 0.0572917
          op_zero_generation_exact: 0.0364583
          adaptive_halt_generation_exact: 0.0520833
          full_minus_worst_ablation: -0.0052083
          active_len6_full_exact: 0.0833333
          core_step_probe_exact: 0.0260417
        state-anchor flatten probe eval:
          checkpoint:
            local_eval/qtrm_native_prompt_anchor_compare_20260513/state_anchor_s1200/last.pt
          probe_pooling: flatten
          core_step_probe_exact: 0.0269097
          depth6_probe_exact: 0.015625
      interpretation:
        The visible state-anchor is a useful diagnostic but not a canonical
        fix. It creates a nonzero ablation drop against op_zero/state_reset and
        slightly lifts fixed-depth exact in this short run, so a stable prompt
        boundary can help causal use of the recurrent path. However, adaptive
        halt accuracy drops, len6 gets worse, and the core-step probe is lower.
        This means the bottleneck is not merely "add a state word before
        answer"; the recurrent trajectory still lacks a reliably readable
        late-depth state. Treat prompt-state-anchor as a diagnostic option only
        until it improves both generation and depth-local probe/readout metrics.
        Moving the visible anchor to the tail ("answer state ") also fails:
        state_reset beats the full recurrent path and the step probe degrades.
        A flatten probe on the state-anchor checkpoint is also low, so the
        missing signal is not merely "distributed across prompt positions but
        invisible at the final readout token". Therefore the next candidate
        should not be another prompt-word anchor. The carrier/readout problem
        needs a learned recurrent state objective or a native recurrence change
        that improves the actual transition trajectory while preserving causal
        drops and improving len6/depth-local probes together.

    prefix-state contrastive transition objective:
      implementation:
        scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
        --prefix-state-contrastive-loss-weight with max-cases/every/temperature/
        state-source/pooling options. This is a training-only objective: for
        each depth d, the full-prompt recurrent state at depth d is contrasted
        against the detached recurrent state from a prefix/noop prompt
        containing only the first d active operations. It keeps final answers
        on the normal LM-logit path and is a softer alternative to the rejected
        MSE prefix-state alignment loss.
      reports:
        local_eval/qtrm_native_prompt_anchor_compare_20260513/state_contrastive_w002_s1200/report.json
        local_eval/qtrm_native_prompt_anchor_compare_20260513/state_contrastive_w0005_every8_s1200/report.json
      matched short result:
        w0.02 every4 max_cases8 temp0.1:
          decision: rejected
          full_generation_exact: 0.0208333
          think0_generation_exact: 0.0208333
          state_reset_generation_exact: 0.0364583
          op_zero_generation_exact: 0.0208333
          adaptive_halt_generation_exact: 0.0208333
          full_minus_worst_ablation: -0.015625
          active_len6_full_exact: 0.0
          core_step_probe_exact: 0.0338542
        w0.005 every8 max_cases8 temp0.2:
          decision: rejected
          full_generation_exact: 0.0520833
          think0_generation_exact: 0.0208333
          state_reset_generation_exact: 0.0364583
          op_zero_generation_exact: 0.0520833
          adaptive_halt_generation_exact: 0.0520833
          full_minus_worst_ablation: 0.0
          active_len6_full_exact: 0.0833333
          core_step_probe_exact: 0.03125
      interpretation:
        Contrastive prefix-state alignment is implemented and verified, but it
        is not a fix. At moderate weight it actively harms the recurrent path:
        state_reset beats full, len6 drops to zero, and depth gain disappears.
        At very low weight it is mostly neutral and does not improve the
        baseline's op_zero tie or core-step probe. Do not promote this loss.
        This narrows the L6 bottleneck further: externally aligning full states
        to prefix states, whether by MSE or contrast, is too indirect. The next
        useful change should alter the recurrent transition itself or its native
        answer readout, not add another prefix-state regularizer.

    interactive transition-gate recurrent update:
      implementation:
        scripts/335_train_qtrm_native_etd_probe.py now supports
        trm_dual_z_interactive_transition_gate. This keeps the official
        TRM-style recurrent attention update as the candidate transition, but
        inserts learned scalar gates that interpolate between previous z_L/z_H
        state and the candidate state. The output still flows through the
        normal native LM readout; there is no answer side channel or external
        renderer.
      report:
        local_eval/qtrm_native_transition_gate_compare_20260513/transition_gate_s1200/report.json
      matched short result:
        decision: rejected
        reject_reasons:
          - full_exact_below_threshold
          - depth_gain_below_threshold
          - ablation_drop_below_threshold
        full_generation_exact: 0.0520833
        think0_generation_exact: 0.0416667
        state_reset_generation_exact: 0.0520833
        op_zero_generation_exact: 0.0520833
        adaptive_halt_generation_exact: 0.0520833
        full_minus_worst_ablation: 0.0
        active_len6_full_exact: 0.0833333
        core_step_probe_exact: 0.0390625
      interpretation:
        The failure is now more specific than "TRM block missing". A
        TRM-style recursive candidate update is present, and even a learned
        transition gate does not make the final LM answer depend causally on
        the recurrent trajectory: state_reset and op_zero exactly tie the full
        model. Therefore the current bottleneck is not solved by preserving
        more previous state inside the official TRM update. The next canonical
        attempt should address the native recurrent state readout/carrier:
        the model must expose a late-depth state that the LM path can read,
        while still failing under core-state ablations. Do not promote
        transition-gate as an architecture improvement.

    recurrent core-carrier readout control:
      implementation:
        scripts/335_train_qtrm_native_etd_probe.py now supports
        think_structure=trm_dual_z_interactive_core_carrier. It keeps the
        same interactive z_L/z_H TRM update, then passes concat(z_L, z_H,
        encoded) through a small causal GRU carrier and adds it back to z_H
        before the normal decoder/LM head. This is still inside the native
        autoregressive LM path: no candidate solver, no answer renderer, and
        no retrieval/tool shortcut.
      reports:
        local_eval/qtrm_native_core_carrier_compare_20260513/core_carrier_s1200/report.json
        local_eval/qtrm_native_core_carrier_compare_20260513/core_carrier_resume_stage5_s1000/report.json
      matched from-scratch short result:
        decision: rejected
        full_generation_exact: 0.0520833
        think0_generation_exact: 0.0208333
        state_reset_generation_exact: 0.0260417
        op_zero_generation_exact: 0.0364583
        full_minus_worst_ablation: 0.015625
        active_len6_full_exact: 0.0
        core_step_probe_exact: 0.0390625
      stage5-resume result:
        decision: rejected
        full_generation_exact: 0.125
        think0_generation_exact: 0.015625
        state_reset_generation_exact: 0.0416667
        op_zero_generation_exact: 0.0208333
        full_minus_worst_ablation: 0.0833333
        active_len full exact:
          len3: 0.1041667
          len4: 0.1458333
          len5: 0.1875
          len6: 0.0625
        adaptive_halt_generation_exact: 0.1354167
        core_step_probe_exact: 0.1545139
      interpretation:
        The recurrent carrier creates some causal sensitivity compared with
        the transition-gate run, but it does not improve the actual solved
        task. From scratch it leaves full accuracy at the same weak 0.052
        level and collapses len5/len6. Resuming from the stronger stage5
        checkpoint also regresses full accuracy from 0.177 to 0.125 and
        reduces the ablation drop. Therefore this carrier is not a canonical
        fix. The current evidence says "readout can be made causal" is not
        enough; the hard part is still learning stable operation composition
        in the recurrent state.

    operation breakdown and hard-op curriculum:
      breakdown report:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_operation_breakdown_eval/report.json
      hard-op reports:
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_hardop257_refine_s1000/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_hardop257_balanced_p035_s1000/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_latepos_hardop257_p075_s1000/report.json
        local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_hardop257_retention_w005_s1000/report.json
      breakdown result:
        hardest fixed-depth positions include:
          position 5 op05: exact 0.0 over 13 cases
          position 6 op02: exact 0.0 over 10 cases
          position 6 op07: exact 0.0 over 6 cases
      hard-op oversampling result:
        p0.75, train active_len 5..6:
          decision: rejected
          full_generation_exact: 0.1822917
          adaptive_halt_generation_exact: 0.171875
          len6_exact: 0.2083333
        p0.35, train active_len 3..6:
          decision: rejected
          full_generation_exact: 0.1666667
          adaptive_halt_generation_exact: 0.2083333
          len6_exact: 0.1666667
        p0.75, train active_len 5..6, hard-op positions 5..6 only:
          decision: rejected
          full_generation_exact: 0.1770833
          adaptive_halt_generation_exact: 0.1770833
          active_len full exact:
            len3: 0.125
            len4: 0.2083333
            len5: 0.2291667
            len6: 0.1458333
        p0.75 all positions + stage5 retention KL w0.05 on active_len 3..5:
          decision: rejected
          full_generation_exact: 0.171875
          adaptive_halt_generation_exact: 0.171875
          active_len full exact:
            len3: 0.1666667
            len4: 0.125
            len5: 0.2916667
            len6: 0.1041667
      interpretation:
        Hard-op oversampling proves len6 can move without changing the
        architecture, but the gain trades off against shorter lengths or
        fixed/adaptive balance. Late-position-only oversampling reduces the
        distribution change but still does not beat the canonical stage5
        fixed/adaptive baseline; it mostly shifts accuracy from len3 into
        len4/len5 with only a small len6 lift. Adding weak reference-retention
        KL to the hard-op run also fails and lowers len6, so this is not just
        catastrophic forgetting. The next fix should be a transition objective
        or recurrent-state/readout design that teaches late-step composition
        without distribution skew, not another backbone swap or another
        hard-op curriculum tweak.

  Depth and representation controls:

    think8 eval reports:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage5_train_len56_refine_s4000_think8_eval/report.json
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_d96_seed337_20260513/stage6_plain_refine_adaptiveaware_s1000_think8_eval/report.json
    think8 result:
      stage5 full/adaptive/active_len6:
        0.1354167 / 0.2135417 / 0.0625
      adaptive-aware full/adaptive/active_len6:
        0.1302083 / 0.1927083 / 0.0208333
    interpretation:
      Extra recurrent depth at eval time hurts fixed-depth accuracy and does
      not solve active_len6. The len6 bottleneck is not "just run more loops";
      it must be learned as a better six-step transition/readout.

    program4-to-program6 flexible curriculum transfer:
      implementation:
        scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now has
        load_model_state_flexible(). With --resume-allow-missing it loads
        same-shaped tensors normally and prefix-copies safe length/vocab
        tensors such as pos_embed.weight. This lets a validated shorter-context
        native LM checkpoint initialize a longer program_len run without
        discarding learned token/encoder/think/decode weights.
      source checkpoint:
        local_eval/qtrm_native_l5_multifamily_hardbalanced_seed339/seed_339/last.pt
        - program_len: 4
        - backbone: mha_etd
        - think_structure: single
        - previous accepted full exact: 0.7565104
      transfer reports:
        local_eval/qtrm_native_program4_to6_transfer_20260513/mha_etd_seed339_s3000/report.json
        local_eval/qtrm_native_program4_to6_transfer_20260513/mha_etd_seed339_refine_s3000/report.json
        local_eval/qtrm_native_program4_to6_transfer_20260513/mha_etd_seed337_s3000/report.json
      result:
        first transfer stage:
          decision: rejected
          full_generation_exact: 0.6458333
          full_minus_worst_ablation: 0.6197917
          family exact:
            checksum: 0.9101563
            modchain: 0.5429688
            revchain: 0.484375
          active_len exact:
            len3: 0.921875
            len4: 0.8177083
            len5: 0.5052083
            len6: 0.3385417
          resume_load_summary:
            pos_embed.weight copied 48/54 rows
        refine stage:
          decision: accepted_l4_mixed_text_reasoning
          full_generation_exact: 0.7278646
          full_minus_think0: 0.7005208
          full_minus_worst_ablation: 0.7083333
          family exact:
            checksum: 0.9960938
            modchain: 0.6210938
            revchain: 0.5664063
          active_len exact:
            len3: 0.9427083
            len4: 0.8645833
            len5: 0.6041667
            len6: 0.5
        seed337 reproduction:
          decision: accepted_l4_mixed_text_reasoning
          full_generation_exact: 0.7252604
          full_minus_think0: 0.7005208
          full_minus_worst_ablation: 0.7044271
          family exact:
            checksum: 0.9726563
            modchain: 0.609375
            revchain: 0.59375
          active_len exact:
            len3: 0.9427083
            len4: 0.78125
            len5: 0.703125
            len6: 0.4739583
      interpretation:
        This is the first strong L6 result. The earlier dual-z TRM/offical-TRM
        architecture attempts were not failing because "program_len6 is
        impossible"; a shorter-context accepted native recurrent model can be
        extended to program_len6 by curriculum transfer and modest refinement.
        The accepted path is still a normal token -> recurrent think -> LM
        logits path and it fails sharply under think0/state_reset/op_zero
        ablations. Seed337 reproduces the same accepted result, so this is not
        only a seed339 accident. However, it is an MHA ETD recurrent baseline,
        not the orthodox dual-z TRM core. Treat it as the current L6
        performance floor: any future dual-z/TRM-native architecture must beat
        this baseline, not only the weaker from-scratch len6 runs.

    program6-to-program8 flexible curriculum transfer:
      baseline:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_s3000/report.json
        decision: rejected
        full_generation_exact: 0.3723958333333333
        full_minus_think0: 0.35416666666666663
        full_minus_worst_ablation: 0.3411458333333333
        min_family_generation_exact: 0.140625
        state_reset_generation_exact: 0.03125
        op_zero_generation_exact: 0.0234375
        family exact:
          checksum: 0.72265625
          modchain: 0.25390625
          revchain: 0.140625
        active_len exact:
          len3: 0.06201550387596899
          len4: 0.31007751937984496
          len5: 0.6511627906976745
          len6: 0.43410852713178294
          len7: 0.42857142857142855
          len8: 0.3492063492063492
      weak answer-space ranking refinement:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_answer_rank_w002_s2000/report.json
        decision: rejected
        full_generation_exact: 0.43359375
        full_minus_think0: 0.40234375
        full_minus_worst_ablation: 0.40234375
        min_family_generation_exact: 0.25390625
        state_reset_generation_exact: 0.03125
        op_zero_generation_exact: 0.022135416666666668
        family exact:
          checksum: 0.70703125
          modchain: 0.33984375
          revchain: 0.25390625
        active_len exact:
          len3: 0.06201550387596899
          len4: 0.2713178294573643
          len5: 0.7984496124031008
          len6: 0.5581395348837209
          len7: 0.47619047619047616
          len8: 0.4365079365079365
        conclusion:
          Answer-space ranking is useful on the strong MHA ETD transfer path:
          it improves overall exact and the worst family while preserving strong
          depth/state/op ablation drops. It is not a side answer solver because
          the training signal is computed from the same canonical LM logits.
          The new bottleneck is active_len 3/4 collapse caused by focusing the
          transfer refinement on active_len 5..8. Next refine should keep the
          ranking loss but train active_len 3..8 to recover shorter lengths
          without sacrificing len8.
      active_len 3..8 recovery refinement:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_answer_rank_w002_len3_8_s2500/report.json
        decision: accepted_l4_mixed_text_reasoning
        full_generation_exact: 0.7044270833333334
        full_minus_think0: 0.6705729166666667
        full_minus_worst_ablation: 0.6731770833333334
        min_family_generation_exact: 0.54296875
        state_reset_generation_exact: 0.03125
        op_zero_generation_exact: 0.03125
        family exact:
          checksum: 0.99609375
          modchain: 0.57421875
          revchain: 0.54296875
        active_len exact:
          len3: 0.8992248062015504
          len4: 0.937984496124031
          len5: 0.813953488372093
          len6: 0.6124031007751938
          len7: 0.5396825396825397
          len8: 0.4126984126984127
        conclusion:
          This is the first accepted L8 curriculum-transfer checkpoint. The
          key was not a new backbone; it was preserving the strong L6 native LM
          path, adding weak answer-space ranking on LM logits, and widening the
          active_len replay/refine range back to 3..8 so short lengths did not
          collapse. Treat this as a candidate, not final proof, until it is
          reproduced on at least one additional seed or a disjoint holdout seed.
      holdout eval seed 19339:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_answer_rank_len3_8_holdout_eval_seed19339/report.json
        decision: rejected
        full_generation_exact: 0.6829427083333334
        full_minus_think0: 0.6529947916666667
        full_minus_worst_ablation: 0.6516927083333334
        min_family_generation_exact: 0.50390625
        state_reset_generation_exact: 0.024088541666666668
        op_zero_generation_exact: 0.03125
        family exact:
          checksum: 0.994140625
          modchain: 0.55078125
          revchain: 0.50390625
        active_len exact:
          len3: 0.937984496124031
          len4: 0.8488372093023255
          len5: 0.7529411764705882
          len6: 0.615686274509804
          len7: 0.5058823529411764
          len8: 0.43137254901960786
        conclusion:
          The accepted seed339 checkpoint is near-threshold but not robustly
          accepted on a larger disjoint holdout. Causality survives strongly.
          The next bottleneck is not short active_len anymore; it is long
          active_len 7/8 and the modchain/revchain families. The next shortest
          intervention is a long-tail refine on active_len 6..8 with retention
          or replay for active_len 3..5.
      long-tail len6..8 refine with short-length retention/replay:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_longtail_len6_8_replay_s3000/report.json
        decision: accepted_l4_mixed_text_reasoning
        full_generation_exact: 0.7161458333333334
        full_minus_think0: 0.6848958333333334
        full_minus_worst_ablation: 0.6848958333333334
        min_family_generation_exact: 0.5625
        state_reset_generation_exact: 0.03125
        op_zero_generation_exact: 0.028645833333333332
        family exact:
          checksum: 0.99609375
          modchain: 0.58984375
          revchain: 0.5625
        active_len exact:
          len3: 0.8992248062015504
          len4: 0.9302325581395349
          len5: 0.7906976744186046
          len6: 0.6511627906976745
          len7: 0.5634920634920635
          len8: 0.4523809523809524
      long-tail holdout seed 19339:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_longtail_len6_8_replay_holdout_seed19339/report.json
        decision: accepted_l4_mixed_text_reasoning
        full_generation_exact: 0.7063802083333334
        full_minus_think0: 0.6783854166666667
        full_minus_worst_ablation: 0.67578125
        min_family_generation_exact: 0.53515625
        state_reset_generation_exact: 0.024088541666666668
        op_zero_generation_exact: 0.030598958333333332
        family exact:
          checksum: 0.998046875
          modchain: 0.5859375
          revchain: 0.53515625
        active_len exact:
          len3: 0.9651162790697675
          len4: 0.8565891472868217
          len5: 0.7686274509803922
          len6: 0.6705882352941176
          len7: 0.5254901960784314
          len8: 0.4470588235294118
      long-tail holdout seed 29339:
        local_eval/qtrm_native_program6_to8_transfer_20260513/mha_etd_seed339_longtail_len6_8_replay_holdout_seed29339/report.json
        decision: accepted_l4_mixed_text_reasoning
        full_generation_exact: 0.705078125
        full_minus_think0: 0.6770833333333334
        full_minus_worst_ablation: 0.6770833333333334
        min_family_generation_exact: 0.52734375
        state_reset_generation_exact: 0.027994791666666668
        op_zero_generation_exact: 0.0234375
        family exact:
          checksum: 1.0
          modchain: 0.587890625
          revchain: 0.52734375
        active_len exact:
          len3: 0.9534883720930233
          len4: 0.875968992248062
          len5: 0.7764705882352941
          len6: 0.615686274509804
          len7: 0.5372549019607843
          len8: 0.4666666666666667
      current L8 status:
        Reproduced robust baseline. The MHA ETD native recurrent path passes
        the L8 mixed text reasoning gate on the original eval seed and two
        disjoint 1536-case holdout seeds, with strong drops under think0,
        state_reset, and op_zero. The remaining bottleneck is still the long
        active_len tail, especially len8 and revchain/modchain. The next
        research step is L10/L12 curriculum transfer or a stronger long-tail
        recurrent-state training signal, not more backbone shopping.

    program8-to-program10 flexible curriculum transfer:
      first transfer:
        local_eval/qtrm_native_program8_to10_transfer_20260513/mha_etd_seed339_s3000/report.json
        decision: rejected
        full_generation_exact: 0.640625
        full_minus_think0: 0.6184895833333334
        full_minus_worst_ablation: 0.60546875
        min_family_generation_exact: 0.4375
        state_reset_generation_exact: 0.03515625
        op_zero_generation_exact: 0.029947916666666668
        resume_load_summary:
          pos_embed.weight copied 60/66 rows
        family exact:
          checksum: 0.98828125
          modchain: 0.49609375
          revchain: 0.4375
        active_len exact:
          len3: 0.9479166666666666
          len4: 0.90625
          len5: 0.8125
          len6: 0.59375
          len7: 0.6145833333333334
          len8: 0.4270833333333333
          len9: 0.40625
          len10: 0.4166666666666667
        conclusion:
          L8->L10 transfer preserves causal recurrence and most short/mid
          lengths, but does not cross the 0.70 gate. The bottleneck is again
          the long active_len tail and modchain/revchain. Apply the same
          long-tail refine pattern that promoted L8: train active_len 8..10
          while retaining/replaying active_len 3..7.

    number-tokenizer reports:
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_number_token_seed337_s4000/report.json
      local_eval/qtrm_native_hardlen_recipe_len6_modchain_number_token_max31_seed337_s4000/report.json
    number-tokenizer result:
      number/default full/adaptive: 0.0364583 / 0.046875
      number/max31 full/adaptive: 0.0520833 / 0.0520833
    interpretation:
      Replacing char tokenization with number tokens is much worse than the
      current char baseline. Do not pursue tokenizer replacement as the next
      fix.

Interpretation:
  The fixed base is causal on hard cases because think0/state_reset/op_zero and
  z-state ablations drop. Longer training moved hard-only full exact from
  0.078125 to 0.2552083, so the fixed base is learning. But absolute hard
  active_len 3/4 accuracy is still weaker than active_len 2 in the single-family
  runs, but the corrected broader-family grid is now accepted for
  modchain/revchain/checksum. The halt-depth final-answer loss fixed the
  adaptive-drop issue without changing the architecture. The same fixed-base
  recipe is accepted on seeds 337, 338, and 339 on the hard active_len 2..4
  modchain gate, and the corrected broader-family grid passes on seeds 337,
  338, and 339. Length-6 multi-family does not pass yet, and length-6
  modchain-only also fails. A small width increase, low-level-cycle increase,
  easy-to-hard curriculum, active_len 5..6 focused refine, late-depth
  reweighted intermediate supervision, and removing intermediate supervision
  are insufficient. Answer-format telemetry shows the full/adaptive path emits
  valid answer text. Depth-sweep telemetry shows the best depth tracks the
  required active length, so early exit is not the primary fix. The next
  bottleneck is recurrent state-transition learning and answer-logit separation
  for longer sequential transformations, not replacing the fixed base, adding a
  renderer, or changing decoding.

Next step:
  Stay on the fixed base and improve only training signal/curriculum/capacity
  for length-6 sequential transformations. New backbone architecture candidates
  are rejected for now unless this base cannot learn after controlled
  state-transition diagnostics and the failure is documented.
```

Promote adaptive halt only if:
  adaptive_halt accuracy >= fixed-depth accuracy within tolerance
  mean_halt_steps < fixed eval_think_steps
  halt_head_off or threshold forcing changes halt telemetry as expected
  no language non-regression failure
```

## Official TRM Dual H3/L6 Architecture Search

Date: 2026-05-13

Canonical constraint:

```text
Use the official TinyRecursiveModels rhythm:
  H_cycles = 3
  L_cycles = 6

Do not tune H/L during architecture shopping. Keep the dual z_L/z_H TRM shell
fixed and vary only the proposal/mixer block inside the update.
```

Reference:

```text
references/repos/TinyRecursiveModels/config/arch/trm.yaml
  H_cycles: 3
  L_cycles: 6
```

Implementation:

```text
scripts/347_run_official_trm_dual_arch_search.sh
  H_CYCLES="3"
  L_CYCLES="6"
```

Added candidate:

```text
reversed_hybrid_3to1
  z_L update: Mamba3, Mamba3, Mamba3, Attention
  z_H update: GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, Attention
  final output: z_H -> decoder -> LM logits -> autoregressive answer text
```

Short comparison:

```text
OUT_ROOT=local_eval/qtrm_native_official_trm_dual_h3l6_reversed_short_20260513
PROFILE=short
CANDIDATES=official,qwen35_3to1,reversed_hybrid_3to1
bash scripts/347_run_official_trm_dual_arch_search.sh
```

Results:

```text
reversed_hybrid_3to1:
  full_generation_exact: 0.06770833333333333
  full_minus_think0: 0.0
  full_minus_worst_ablation: 0.015624999999999993
  state_reset_generation_exact: 0.041666666666666664
  op_zero_generation_exact: 0.052083333333333336
  z_l_zero_generation_exact: 0.026041666666666668
  z_h_zero_generation_exact: 0.0
  core_step_probe_exact: 0.1267361044883728
  official_fla_delta_mixers: 3
  official_mamba3_mixers: 3

official:
  full_generation_exact: 0.052083333333333336
  full_minus_think0: 0.020833333333333336
  full_minus_worst_ablation: 0.015625
  state_reset_generation_exact: 0.036458333333333336
  op_zero_generation_exact: 0.010416666666666666
  z_l_zero_generation_exact: 0.0
  z_h_zero_generation_exact: 0.0
  core_step_probe_exact: 0.0833333358168602

qwen35_3to1:
  full_generation_exact: 0.036458333333333336
  full_minus_think0: 0.010416666666666668
  full_minus_worst_ablation: 0.010416666666666668
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.026041666666666668
  z_l_zero_generation_exact: 0.0
  z_h_zero_generation_exact: 0.0
  core_step_probe_exact: 0.1180555522441864
  official_fla_delta_mixers: 3
```

Interpretation:

```text
The reversed 3:1 split is the best short H3/L6 architecture candidate so far
by full exact and core-state probe. It also uses real official Mamba3 and FLA
GatedDelta backends, not torch fallback.

Do not promote it yet. full_minus_think0 is 0.0, so it has not proven that
deeper H cycles improve the answer. z_H is causal, but op_zero/state_reset are
too close to full. The next experiment should keep reversed_hybrid_3to1 and add
depth-gain/state-transition pressure:
  depth counterfactual
  state-reset counterfactual
  operation counterfactual
  latent-refine or answer-space ranking tuned for z_H
```

### Reversed Hybrid Readout Fix

Question:

```text
Can reversed_hybrid_3to1 be moved toward the strong MHA ETD baseline without
leaving the canonical LM path?
```

Negative result:

```text
local_eval/qtrm_native_reversed_h3l6_depth_cf_latent_refine_len4_s1200_20260513

reversed_hybrid_3to1 + depth/state/op counterfactual + latent refine:
  full_generation_exact: 0.0234375
  think0_generation_exact: 0.0
  full_minus_think0: 0.0234375
  full_minus_worst_ablation: -0.0390625
  state_reset_generation_exact: 0.0625
  op_zero_generation_exact: 0.0234375
  z_h_zero_generation_exact: 0.0

Conclusion:
  loss pressure alone is insufficient. state_reset > full means the recurrent
  carry path is still harmful or unstable.
```

Readout experiment:

```text
local_eval/qtrm_native_reversed_joint_readout_short_20260513

reversed_hybrid_3to1:
  full_generation_exact: 0.036458333333333336
  full_minus_think0: 0.010416666666666668
  full_minus_worst_ablation: -0.026041666666666664
  z_h_zero_generation_exact: 0.0

reversed_hybrid_3to1_joint_readout:
  full_generation_exact: 0.046875
  full_minus_think0: 0.046875
  full_minus_worst_ablation: -0.010416666666666664
  z_h_zero_generation_exact: 0.057291666666666664

Conclusion:
  joint readout improves depth gain, but it is not causal enough because the
  output can improve when z_H is zeroed.
```

Accepted next candidate:

```text
trm_dual_z_reversed_hybrid_3to1_core_gated_readout

Architecture:
  z_L update: Mamba3, Mamba3, Mamba3, Attention
  z_H update: GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, Attention
  readout: z_H + alpha * proj(encoded, z_L, z_H) * tanh(z_H)
  final output: readout -> decoder -> LM logits -> autoregressive answer text

Rationale:
  The readout may use encoded/z_L as context, but z_H multiplicatively gates
  the bridge. If z_H is zeroed, the bridge contribution is also zeroed. This
  preserves the canonical causal TRM path while giving the decoder an ETD-like
  state extraction bridge.
```

Short result:

```text
local_eval/qtrm_native_reversed_core_gated_short_20260513

reversed_hybrid_3to1_core_gated_readout:
  full_generation_exact: 0.08333333333333333
  full_minus_think0: 0.04687499999999999
  full_minus_worst_ablation: 0.031249999999999993
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.041666666666666664
  z_l_zero_generation_exact: 0.052083333333333336
  z_h_zero_generation_exact: 0.0
  core_step_probe_exact: 0.125
  official_fla_delta_mixers: 3
  official_mamba3_mixers: 3
```

Interpretation:

```text
This is the first reversed 3:1 variant in this branch that improves full exact,
shows positive depth gain, shows positive ablation drop, and keeps z_H causal.
It is not close to the MHA ETD L8 baseline yet, but it is the correct next
base for curriculum transfer.
```

Next:

```text
1. Keep H=3/L=6 and core_gated_readout fixed.
2. Train len4 longer and preserve the best checkpoint.
3. Transfer len4 -> len6 -> len8 with retention/replay.
4. Only after len8 is stable, reintroduce stricter depth/state/op
   counterfactual losses.
5. Promote only if full exact, depth gain, and ablation drop all improve over
   the non-gated reversed baseline.
```

### Official TRM Implementation Audit

Reference:

```text
references/repos/TinyRecursiveModels/models/recursive_reasoning/trm.py
references/repos/TinyRecursiveModels/config/arch/trm.yaml
```

Official TRM core:

```text
H_cycles: 3
L_cycles: 6
L_layers: 2
H_layers: ignored

for H_step in range(H_cycles - 1), no_grad:
  for L_step in range(L_cycles):
    z_L = L_level(z_L, z_H + input_embeddings)
  z_H = L_level(z_H, z_L)

final H_step, grad:
  for L_step in range(L_cycles):
    z_L = L_level(z_L, z_H + input_embeddings)
  z_H = L_level(z_H, z_L)

output = lm_head(z_H)
new_carry = detach(z_H, z_L)
```

What our QTRM-native probe matched:

```text
H/L loop order: matched
H=3/L=6 in canonical wrapper: matched
H-1 no-grad, final H grad: matched
single shared think module for z_L and z_H updates: matched
final answer path through z_H -> decoder -> LM logits: matched in concept
```

Mismatch found and fixed:

```text
Official L_level has L_layers=2.
Our NativeTRMOfficialBlock was a single block.

Fix:
  NativeTRMOfficialStack now wraps two NativeTRMOfficialBlock layers.
  stage_backbone=trm_official now returns NativeTRMOfficialStack(..., layers=2).

Smoke:
  local_eval/qtrm_native_official_stack2_smoke_20260513
```

Remaining intentional differences:

```text
Official TRM attention is non-causal because puzzle labels are separate from
the input. Our LM probe uses causal attention to avoid answer-token leakage in
autoregressive training.

Official uses RoPE inside the reasoning attention. Our current official-style
block uses PyTorch MHA without RoPE.

Official attention uses bias-free qkv/o projections. PyTorch MHA includes its
own projection layout and biases.

Official RMSNorm is parameter-free. Our NativeRMSNorm has a learnable scale.

Official SwiGLU computes hidden size from expansion and rounds to a multiple
of 256. Our probe uses the explicit d_ff value.

Official uses scaled token embeddings, puzzle prefix embeddings, ACT carry,
q_halt/q_continue heads, stablemax CE, and bfloat16 casted weights. The probe
uses a normal causal LM training path with ordinary CE and optional halt probes.
```

Conclusion:

```text
The previous "official" candidate was official-loop-correct but not an exact
official implementation. It should be described as causal official-style TRM
adapted to a general LM path.

The L_layers=2 mismatch was a real oversight and is now corrected. Remaining
differences are either required for autoregressive LM training or should be
tested as separate ablations before claiming official-TRM fidelity.
```

Stack2 short rerun:

```text
local_eval/qtrm_native_official_stack2_short_20260513

official, seed777:
  full_generation_exact: 0.057291666666666664
  full_minus_think0: 0.041666666666666664
  full_minus_worst_ablation: 0.0
  state_reset_generation_exact: 0.046875
  op_zero_generation_exact: 0.057291666666666664

core_gated_readout, seed979:
  full_generation_exact: 0.078125
  full_minus_think0: 0.015625
  full_minus_worst_ablation: 0.03125
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.03125
  z_h_zero_generation_exact: 0.0

local_eval/qtrm_native_official_stack2_reversed_only_short_20260513

reversed_hybrid_3to1, seed777:
  full_generation_exact: 0.041666666666666664
  full_minus_think0: -0.015625
  full_minus_worst_ablation: 0.0

local_eval/qtrm_native_official_stack2_core_gated_seed777_short_20260513

core_gated_readout, seed777:
  full_generation_exact: 0.036458333333333336
  full_minus_think0: -0.041666666666666664
  full_minus_worst_ablation: 0.0

local_eval/qtrm_native_official_stack2_reversed_seed979_short_20260513

reversed_hybrid_3to1, seed979:
  full_generation_exact: 0.036458333333333336
  full_minus_think0: 0.0
  full_minus_worst_ablation: 0.005208333333333336
```

Interpretation:

```text
After L_layers=2, core_gated_readout is promising but seed-sensitive. It beats
the original reversed variant on seed979 and keeps z_H causal, but it does not
beat the original on seed777. Do not promote it as a robust architecture yet.

Next fidelity fix:
  Official SwiGLU hidden size should use expansion=4 and round to a multiple
  of 256, instead of the probe's explicit d_ff=128 path.
```

### Official SwiGLU Fidelity Fix

Reference difference:

```text
Official TRM SwiGLU:
  hidden = ceil_to_multiple(round(expansion * hidden_size * 2 / 3), 256)
  expansion = 4

Previous probe:
  hidden = explicit d_ff, usually 128 for d_model=64
```

Fix:

```text
NativeOfficialSwiGLU added.
NativeTRMOfficialBlock now uses official expansion=4 width rule.

d_model=64:
  official-style hidden: 256
  gate_up out_features: 512
```

Smoke:

```text
local_eval/qtrm_native_official_stack2_swiglu_smoke_20260513
```

Short result:

```text
local_eval/qtrm_native_official_stack2_swiglu_short_20260513

reversed_hybrid_3to1:
  full_generation_exact: 0.07291666666666667
  full_minus_think0: 0.036458333333333336
  full_minus_worst_ablation: 0.015625000000000007
  state_reset_generation_exact: 0.046875
  op_zero_generation_exact: 0.057291666666666664
  z_l_zero_generation_exact: 0.026041666666666668
  z_h_zero_generation_exact: 0.0
  core_step_probe_exact: 0.140625

official:
  full_generation_exact: 0.06770833333333333
  full_minus_think0: 0.02083333333333333
  full_minus_worst_ablation: 0.005208333333333329
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0625

core_gated_readout:
  full_generation_exact: 0.0625
  full_minus_think0: -0.02083333333333333
  full_minus_worst_ablation: 0.026041666666666664
  z_h_zero_generation_exact: 0.0
```

Interpretation:

```text
The official SwiGLU fidelity fix improved both official and reversed variants.
The previous core-gated readout no longer wins under this more official stack.
Current base candidate:
  trm_dual_z_reversed_hybrid_3to1

Do not promote yet:
  ablation drop is still small, and op_zero remains close to full. The next
  check is a second-seed short run before longer curriculum.
```

Second seed:

```text
local_eval/qtrm_native_official_stack2_swiglu_reversed_seed979_short_20260513

reversed_hybrid_3to1:
  full_generation_exact: 0.046875
  full_minus_think0: 0.0
  full_minus_worst_ablation: -0.005208333333333336
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.041666666666666664
  z_l_zero_generation_exact: 0.052083333333333336
  z_h_zero_generation_exact: 0.0
```

Interpretation:

```text
The architecture is still seed-sensitive. Do not start long curriculum until
the short gate is more stable. The next official-fidelity hypothesis is the
training loss: official TRM uses stablemax_cross_entropy, while the probe uses
ordinary softmax CE.
```

### Stablemax Loss Probe

Implementation:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  --answer-loss-type cross_entropy
  --answer-loss-type stablemax_cross_entropy

scripts/347_run_official_trm_dual_arch_search.sh
  ANSWER_LOSS_TYPE=stablemax_cross_entropy
```

Smoke:

```text
local_eval/qtrm_native_stablemax_smoke_20260513
```

Short result:

```text
local_eval/qtrm_native_stablemax_short_seed777_20260513

official + stablemax:
  full_generation_exact: 0.0625
  full_minus_think0: 0.026041666666666664
  full_minus_worst_ablation: 0.015625
  state_reset_generation_exact: 0.036458333333333336
  op_zero_generation_exact: 0.046875

reversed_hybrid_3to1 + stablemax:
  full_generation_exact: 0.026041666666666668
  full_minus_think0: 0.0
  full_minus_worst_ablation: -0.026041666666666668
  core_step_probe_exact: 0.1788194477558136
```

Interpretation:

```text
Stablemax is not a general improvement in this LM probe. It may slightly
improve official ablation drop, but it severely hurts reversed_hybrid_3to1.
Keep cross_entropy as the default for the reversed branch. Treat stablemax as
an optional official-fidelity ablation, not the canonical training loss.
```

### Pre-Norm Ablation

Question:

```text
Is the recurrent TRM branch weak because the official block is post-norm?
```

Reference:

```text
Official TinyRecursiveModels TRM is post-norm:
  x = rms_norm(x + attention(x))
  x = rms_norm(x + swiglu(x))

This matches the official implementation, but causal autoregressive LMs often
prefer pre-norm for optimization stability:
  x = x + attention(norm(x))
  x = x + swiglu(norm(x))
```

Implementation:

```text
trm_official_prenorm
trm_dual_z_reversed_hybrid_3to1_prenorm
```

Smoke:

```text
local_eval/qtrm_native_prenorm_smoke_fix_20260513
```

Short comparison:

```text
local_eval/qtrm_native_prenorm_short_seed777_20260513

reversed_hybrid_3to1:
  full_generation_exact: 0.08333333333333333
  full_minus_think0: 0.04687499999999999
  full_minus_worst_ablation: 0.05729166666666666
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.026041666666666668
  z_h_zero_generation_exact: 0.0

reversed_hybrid_3to1_prenorm:
  full_generation_exact: 0.0625
  full_minus_think0: 0.057291666666666664
  full_minus_worst_ablation: 0.026041666666666664
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.020833333333333332
  z_h_zero_generation_exact: 0.0

official_prenorm:
  full_generation_exact: 0.041666666666666664
  full_minus_think0: 0.015624999999999997
  full_minus_worst_ablation: 0.010416666666666664

official post-norm:
  full_generation_exact: 0.03125
  full_minus_think0: -0.020833333333333336
  full_minus_worst_ablation: -0.015625
```

Interpretation:

```text
Pre-norm helps the pure official baseline, so the user's suspicion was valid
for that branch. It does not beat post-norm on the stronger reversed hybrid
branch in this seed. Current best short candidate remains:
  trm_dual_z_reversed_hybrid_3to1

However, pre-norm reversed has the largest depth gain in this comparison, so it
should remain as a stability candidate for longer or harder length sweeps.
```

### Reversed Hybrid Training-Efficiency Path

Question:

```text
Can reversed_hybrid_3to1 borrow the training stability observed in the strong
MHA ETD path without changing the canonical reversed_hybrid_3to1 architecture?
```

Rule:

```text
Allowed:
  same reversed_hybrid_3to1 architecture
  same H=3 / L=6 recursive schedule
  same causal LM logits answer path
  larger fair capacity
  LR warmup + cosine decay
  len4 -> len6 -> len8 curriculum using the same reversed checkpoint family
  depth intermediate loss, answer-space ranking, active-length replay

Not allowed for canonical promotion:
  warm-starting from MHA ETD and claiming pure reversed_hybrid_3to1 evidence
  hidden renderer/sidecar answer path
  MemoryOS/RAG shortcut
  answer computed outside the LM logits path
```

Implementation:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  --lr-schedule constant
  --lr-schedule linear_warmup_cosine
  --lr-warmup-steps N
  --lr-min-ratio R

scripts/348_train_reversed_hybrid_efficiency.sh
  STAGE=smoke
  STAGE=len4
  STAGE=len6 RESUME_FROM=<len4 last.pt>
  STAGE=len8 RESUME_FROM=<len6 last.pt>
```

Canonical target:

```text
reversed_hybrid_3to1 should improve because the recursive core becomes easier
to optimize, not because ETD computes the answer. Promotion still requires:

  full > think0
  full > state_reset / op_zero / z_L_zero / z_H_zero
  deeper loop improves held-out generation
  held-out active lengths and families do not collapse
  final answer is generated through LM logits
```

First len4 result:

```text
local_eval/qtrm_native_reversed_efficiency_len4_lrsched_20260513/len4_seed777

full_generation_exact: 0.1328125
think0_generation_exact: 0.028645833333333332
full_minus_think0: 0.10416666666666667
full_minus_worst_ablation: 0.0859375
state_reset_generation_exact: 0.03125
op_zero_generation_exact: 0.028645833333333332
z_l_zero_generation_exact: 0.046875
z_h_zero_generation_exact: 0.0
min_family_generation_exact: 0.0390625
format_valid: 1.0
```

Interpretation:

```text
Training-efficiency recipe improved the reversed branch from the prior short
8.3% region to 13.3% while preserving a real causal core gap. The result is
not L4/L5-grade yet because modchain and active_len4 remain weak.

The output format is already stable, so the bottleneck is not lexicalization.
The hidden core-step probe is much stronger at depth1 than depth2/depth3,
suggesting that the early intermediate loss may be over-shaping shallow states.
The next same-architecture run should resume this checkpoint with lower
depth_intermediate weight and keep answer/ranking/replay active.
```

Second len4 result, same architecture phase2:

```text
local_eval/qtrm_native_reversed_efficiency_len4_phase2_20260513/len4_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_lrsched_20260513/len4_seed777/last.pt

recipe changes:
  depth_intermediate_loss_weight: 0.20 -> 0.05
  answer_space_ranking_loss_weight: 0.02 -> 0.03
  active_len_replay_loss_weight: 0.02 -> 0.04
  lr: 8e-5 -> 4e-5

full_generation_exact: 0.2682291666666667
think0_generation_exact: 0.044270833333333336
full_minus_think0: 0.22395833333333334
full_minus_worst_ablation: 0.21093750000000003
state_reset_generation_exact: 0.026041666666666668
op_zero_generation_exact: 0.033854166666666664
z_l_zero_generation_exact: 0.057291666666666664
z_h_zero_generation_exact: 0.0
format_valid: 1.0

by_family:
  checksum: 0.625
  modchain: 0.0703125
  revchain: 0.109375

by_active_len:
  len2: 0.35658914728682173
  len3: 0.2558139534883721
  len4: 0.19047619047619047
```

Interpretation:

```text
This is the first clear same-architecture training-efficiency gain for
reversed_hybrid_3to1. It moved from 13.3% to 26.8% and increased the causal
core gap from 10.4 points to 22.4 points.

The next bottleneck is not output formatting or core causality; it is family
imbalance. Checksum is already learnable, while modchain/revchain remain weak.
Next run should target hard-family replay with retention against the phase2
checkpoint, not another architecture change.
```

Third len4 result, hard-family replay:

```text
local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_phase2_20260513/len4_seed777/last.pt

recipe changes:
  task_families: modchain,revchain,modchain,revchain,checksum
  eval_task_families: checksum,modchain,revchain
  depth_intermediate_loss_weight: 0.03
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.02

full_generation_exact: 0.359375
think0_generation_exact: 0.026041666666666668
full_minus_think0: 0.3333333333333333
full_minus_worst_ablation: 0.296875
state_reset_generation_exact: 0.020833333333333332
op_zero_generation_exact: 0.028645833333333332
z_l_zero_generation_exact: 0.0625
z_h_zero_generation_exact: 0.0
min_family_generation_exact: 0.140625

by_family:
  checksum: 0.75
  modchain: 0.1875
  revchain: 0.140625

by_active_len:
  len2: 0.4728682170542636
  len3: 0.3798449612403101
  len4: 0.2222222222222222
```

Interpretation:

```text
Hard-family replay improved the same reversed_hybrid_3to1 architecture from
26.8% to 35.9% and increased the causal core gap to 33.3 points over think0.
This confirms that ETD-like training stability can be borrowed as optimizer,
curriculum, loss phasing, and replay without replacing the canonical recursive
core.

Still not enough for promotion. The next bottleneck is length-4 and hard
families, not architecture or formatting. The next canonical move is either:
  1. one more len4 hard-family continuation until min_family is substantially
     higher; or
  2. len6 curriculum transfer from this checkpoint, treating low len6 as a
     depth-scaling failure rather than a root-architecture win.
```

Fourth len4 result, rejected len4-only focus:

```text
local_eval/qtrm_native_reversed_efficiency_len4_len4focus_20260513/len4_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

recipe:
  train_active_len_cycle_min/max: 4/4
  eval_active_len_cycle_min/max: 2/4
  task_families: modchain,revchain,modchain,revchain,modchain,revchain,checksum
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.03

full_generation_exact: 0.3151041666666667
full_minus_think0: 0.2916666666666667
full_minus_worst_ablation: 0.2421875
min_family_generation_exact: 0.125

by_family:
  checksum: 0.6796875
  modchain: 0.140625
  revchain: 0.125

by_active_len:
  len2: 0.40310077519379844
  len3: 0.2868217054263566
  len4: 0.25396825396825395
```

Decision:

```text
Reject as canonical continuation. It slightly improves len4 over phase3
0.222 -> 0.254, but hurts full exact 0.359 -> 0.315 and min_family 0.141 ->
0.125. Pure len4 focus is too narrow; keep phase3 as the current canonical
reversed_hybrid_3to1 checkpoint and test len6 transfer from phase3 instead.
```

First len6 transfer:

```text
local_eval/qtrm_native_reversed_efficiency_len6_transfer_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

full_generation_exact: 0.20703125
think0_generation_exact: 0.01953125
full_minus_think0: 0.1875
full_minus_worst_ablation: 0.1640625
state_reset_generation_exact: 0.017578125
op_zero_generation_exact: 0.0234375
z_l_zero_generation_exact: 0.04296875
z_h_zero_generation_exact: 0.0
min_family_generation_exact: 0.04093567251461988

by_family:
  checksum: 0.52046783625731
  modchain: 0.04093567251461988
  revchain: 0.058823529411764705

by_active_len:
  len3: 0.17829457364341086
  len4: 0.26356589147286824
  len5: 0.234375
  len6: 0.15079365079365079
```

Interpretation:

```text
Len6 transfer is not collapsed: recursive core ablations still reduce the
answer path, and len6 exact is nonzero. But hard-family performance almost
collapses while checksum remains learnable. The next len6 move should repeat
the phase2 pattern: hard-family replay with retention, not an architecture
change.
```

Len6 hard-family continuation:

```text
local_eval/qtrm_native_reversed_efficiency_len6_hardfamily_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len6_transfer_20260513/len6_seed777/last.pt

recipe:
  task_families: modchain,revchain,modchain,revchain,modchain,revchain,checksum
  depth_intermediate_loss_weight: 0.02
  answer_space_ranking_loss_weight: 0.04
  active_len_replay_loss_weight: 0.04
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.03

full_generation_exact: 0.20703125
think0_generation_exact: 0.021484375
full_minus_think0: 0.185546875
full_minus_worst_ablation: 0.173828125
min_family_generation_exact: 0.041176470588235294

by_family:
  checksum: 0.5263157894736842
  modchain: 0.05263157894736842
  revchain: 0.041176470588235294

by_active_len:
  len3: 0.23255813953488372
  len4: 0.23255813953488372
  len5: 0.203125
  len6: 0.15873015873015872
```

Decision:

```text
Reject as a promotion. Hard-family continuation preserves the causal core gap
but does not meaningfully improve modchain/revchain. The bottleneck is not
general collapse; it is ordered composition in the hard families.
```

Revchain intermediate-target bug fix:

```text
file:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

issue:
  revchain semantics applies operations right-to-left, but intermediate depth
  teacher targets were built from left prefixes. That gave directionally wrong
  shallow-depth supervision for revchain.

fix:
  revchain intermediate targets now use active operation suffixes, preserving
  the same right-to-left causal order as compute_answer().

test:
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
  test_intermediate_answer_targets_follow_family_direction

verification:
  .venv/bin/python -m unittest tests.test_qtrm_native_mixed_text_reasoning_probe -v
  87 tests passed
```

Len6 transfer after revchain target fix:

```text
local_eval/qtrm_native_reversed_efficiency_len6_revtargetfix_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

full_generation_exact: 0.197265625
think0_generation_exact: 0.025390625
full_minus_think0: 0.171875
full_minus_worst_ablation: 0.13671875
min_family_generation_exact: 0.03529411764705882

by_family:
  checksum: 0.5146198830409356
  modchain: 0.04093567251461988
  revchain: 0.03529411764705882

by_active_len:
  len3: 0.2558139534883721
  len4: 0.14728682170542637
  len5: 0.2265625
  len6: 0.15873015873015872
```

Interpretation:

```text
The target fix is mandatory because the old revchain supervision was wrong,
but it is not sufficient by itself. Periodic training curves improved earlier
in the run, yet final exactness did not beat the previous len6 transfer.
Future len6 experiments must keep this fix, but the remaining bottleneck is
still recurrent state composition.
```

Fast family-operation breakdown:

```text
implementation:
  generation_operation_breakdown now reports nested by_family breakdowns.
  scripts/348_train_reversed_hybrid_efficiency.sh can disable heavy
  eval-answer-space-argmax and eval-core-step-probe via env flags.

diagnostic run:
  local_eval/qtrm_native_reversed_efficiency_len6_family_opbreakdown_fast_20260513/len6_seed777

checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_hardfamily_20260513/len6_seed777/last.pt

eval_cases: 192
eval_answer_space_argmax: false
eval_core_step_probe: false
eval_operation_breakdown: true

full_generation_exact: 0.20833333333333334
min_family_generation_exact: 0.0625

by_family:
  checksum: 0.484375
  modchain: 0.078125
  revchain: 0.0625

by_active_len:
  len3: 0.20833333333333334
  len4: 0.22916666666666666
  len5: 0.25
  len6: 0.14583333333333334
```

Diagnosis:

```text
The output format is valid for all three families. The model is not failing
because it cannot render answers. Checksum remains much easier, while
modchain/revchain fail across many last-op and error-delta buckets. Therefore
the len6 bottleneck is ordered recurrent composition, not a single bad opcode
or a lexicalization/renderer defect.
```

No-intermediate len6 triage:

```text
local_eval/qtrm_native_reversed_efficiency_len6_nointermediate_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

recipe:
  depth_intermediate_loss_weight: 0.0
  answer_space_ranking_loss_weight: 0.03
  active_len_replay_loss_weight: 0.04
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.02
  eval_answer_space_argmax: false
  eval_core_step_probe: false

periodic:
  step 600 exact: 0.11458333333333333
  step 1000 exact: 0.125

final:
  full_generation_exact: 0.142578125
  think0_generation_exact: 0.021484375
  full_minus_think0: 0.12109375
  full_minus_worst_ablation: 0.087890625
  min_family_generation_exact: 0.052941176470588235

by_family:
  checksum: 0.3157894736842105
  modchain: 0.05847953216374269
  revchain: 0.052941176470588235

by_active_len:
  len3: 0.18604651162790697
  len4: 0.15503875968992248
  len5: 0.09375
  len6: 0.1349206349206349
```

Decision:

```text
Reject. Removing intermediate depth supervision weakens the full score and
the causal ablation gap. The bug-fixed intermediate targets should stay, but
the remaining question is how to schedule length and family difficulty, not
whether to remove intermediate supervision entirely.
```

Active-length curriculum triage:

```text
local_eval/qtrm_native_reversed_efficiency_len6_len_curriculum_triage_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

recipe:
  active_len_batch_cycle: false
  active_len_curriculum: true
  active_len_curriculum_min: 3
  active_len_curriculum_warmup_frac: 0.7
  depth_intermediate_loss_weight: 0.03
  eval_cases: 192

periodic:
  step 600 exact: 0.052083333333333336
  step 600 min_active_len_exact: 0.0
```

Decision:

```text
Stopped early. Replacing batch length cycling with a simple curriculum makes
the len6 transfer much worse. Keep active_len_batch_cycle as the default. The
next useful check is not another schedule-only change; it is whether len6
needs the stronger original depth_intermediate weight near 0.20 after the
revchain target fix.
```

Depth-intermediate 0.20 triage:

```text
local_eval/qtrm_native_reversed_efficiency_len6_depth020_triage_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

recipe:
  active_len_batch_cycle: true
  depth_intermediate_loss_weight: 0.20
  eval_cases: 192
  eval_answer_space_argmax: false
  eval_core_step_probe: false

periodic:
  step 600 exact: 0.10416666666666667
  step 600 min_active_len_exact: 0.041666666666666664
```

Decision:

```text
Stopped early and discarded under the autoresearch-style triage rule. Restoring
stronger intermediate supervision alone does not recover the original len6
transfer. The active bottleneck remains ordered composition across modchain and
revchain, not output format, not checksum arithmetic, and not a single scalar
loss weight.
```

Causal-prefix depth target fix:

```text
issue:
  intermediate_answer_targets had already been fixed for revchain, but other
  depth/trace losses and probes still used left-prefix active programs. For
  revchain, causal order is right-to-left, so depth=1 must refer to the last
  operation, not the first operation.

fix:
  Added case_with_causal_prefix_len().
  Updated prefix_state_alignment_loss, prefix_state_contrastive_loss,
  core_step_probe labels, core_step_codec_loss labels, intermediate targets,
  and prefix_depth_anchor_loss to use family-correct causal prefixes.

tests:
  test_causal_prefix_len_follows_family_order
  test_core_step_codec_labels_follow_family_causal_order

verification:
  .venv/bin/python -m py_compile scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  bash -n scripts/348_train_reversed_hybrid_efficiency.sh
  .venv/bin/python -m unittest tests.test_qtrm_native_mixed_text_reasoning_probe -v
  91 tests passed
```

Core-step codec triage:

```text
local_eval/qtrm_native_reversed_efficiency_len6_corecodec_triage_20260513/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

recipe:
  core_step_codec_loss_weight: 0.05
  core_step_codec_state_source: both
  core_step_codec_pooling: last
  depth_intermediate_loss_weight: 0.03
  active_len_batch_cycle: true
  eval_cases: 192

periodic:
  step 600 exact: 0.14583333333333334
  step 600 min_active_len_exact: 0.10416666666666667
  step 1000 exact: 0.11979166666666667
  step 1000 min_active_len_exact: 0.0625

final:
  restored_best_eval_checkpoint: true
  full_generation_exact: 0.14583333333333334
  full_minus_think0: 0.13541666666666669
  full_minus_worst_ablation: 0.08854166666666669
  min_family_generation_exact: 0.03125

by_family:
  checksum: 0.328125
  modchain: 0.078125
  revchain: 0.03125

by_active_len:
  len3: 0.22916666666666666
  len4: 0.14583333333333334
  len5: 0.10416666666666667
  len6: 0.10416666666666667
```

Decision:

```text
Treat as probe/discard, not canonical. The codec loss improved the active
length floor relative to other short triages, so it touches the state-depth
bottleneck. But it failed the actual hard-family objective because revchain
fell to 0.03125. The next operational fix is to make periodic checkpoint
selection track min_family_generation_exact, not only full exactness and
active-length floor.
```

Periodic family-floor checkpoint selection:

```text
implementation:
  periodic records now include min_family_generation_exact.
  Added periodic_eval_score(mode="family_floor").
  scripts/348_train_reversed_hybrid_efficiency.sh exposes:
    PERIODIC_EVAL_SCORE_MODE=family_floor

purpose:
  when the active bottleneck is modchain/revchain balance, best-checkpoint
  selection must not restore a checkpoint that improves full exactness by
  sacrificing the weakest family.
```

Family-floor triage result:

```text
local_eval/qtrm_native_reversed_efficiency_len6_familyfloor_triage_20260514/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

best_periodic_eval:
  restored_best_eval_checkpoint: true
  restored_step: 800
  step800 full_generation_exact: 0.125
  step800 min_active_len_generation_exact: 0.0625
  step800 min_family_generation_exact: 0.046875

final:
  full_generation_exact: 0.125
  full_minus_think0: 0.11458333333333333
  full_minus_worst_ablation: 0.08333333333333334
  min_family_generation_exact: 0.046875

by_family:
  checksum: 0.28125
  modchain: 0.046875
  revchain: 0.046875

by_active_len:
  len3: 0.14583333333333334
  len4: 0.125
  len5: 0.16666666666666666
  len6: 0.0625
```

Decision:

```text
Keep as an experiment-operations improvement, not an architecture promotion.
The selector correctly restored step 800 instead of the later step 1000,
because step 1000 had a worse family floor. This proves the autoresearch-style
keep/discard checkpoint rule is useful for QTRM runs.

Do not call this a len6 solution. The core still fails the hard-family ordered
composition bottleneck: modchain and revchain remain at 0.046875, and len6 is
only 0.0625.
```

Family-aware active-length fix:

```text
problem:
  revchain executes operations in right-to-left causal order.
  active-length curriculum/eval previously kept the left prefix for every
  family, which made revchain active-len examples semantically inconsistent
  with the causal prefix targets used by intermediate/depth losses.

fix:
  case_with_active_program_len now keeps the first active operations in each
  family's causal order.
  modchain/checksum: keep left prefix.
  revchain: keep right suffix in prompt order.
  effective_program_len now counts non-NOOP operations instead of relying on a
  trailing-NOOP tail.

verification:
  .venv/bin/python -m unittest tests.test_qtrm_native_mixed_text_reasoning_probe -v
  .venv/bin/python -m py_compile scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  bash -n scripts/348_train_reversed_hybrid_efficiency.sh
```

Causal-active triage:

```text
local_eval/qtrm_native_reversed_efficiency_len6_causalactive_triage_20260514/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/last.pt

best_periodic_eval:
  restored_best_eval_checkpoint: true
  restored_step: 800
  step800 full_generation_exact: 0.109375
  step800 min_active_len_generation_exact: 0.041666666666666664
  step800 min_family_generation_exact: 0.03125

final:
  full_generation_exact: 0.109375
  full_minus_think0: 0.109375
  full_minus_worst_ablation: 0.07291666666666666
  min_family_generation_exact: 0.03125

by_family:
  checksum: 0.265625
  modchain: 0.03125
  revchain: 0.03125

by_active_len:
  len3: 0.14583333333333334
  len4: 0.10416666666666667
  len5: 0.14583333333333334
  len6: 0.041666666666666664
```

Decision:

```text
Keep the semantic bugfix, discard the checkpoint as a performance promotion.
The corrected eval is stricter and confirms that the remaining bottleneck is
not renderer format or hidden evidence; it is still ordered composition in the
normal LM path. Next run should add exactly one mechanism on top of the fixed
semantics, starting with prefix_depth_anchor.
```

Prefix-depth anchor on fixed active-length semantics:

```text
local_eval/qtrm_native_reversed_efficiency_len6_prefixanchor_triage_20260514/len6_seed777

changed_mechanism:
  prefix_depth_anchor_loss_weight: 0.05

periodic:
  step400 full_generation_exact: 0.078125
  step400 min_active_len_generation_exact: 0.020833333333333332
  step400 min_family_generation_exact: 0.015625
  step800 full_generation_exact: 0.15104166666666666
  step800 min_active_len_generation_exact: 0.08333333333333333
  step800 min_family_generation_exact: 0.0

final_restored_by_family_floor:
  restored_step: 400
  full_generation_exact: 0.078125
  full_minus_think0: 0.06770833333333333
  full_minus_worst_ablation: 0.026041666666666664
  min_family_generation_exact: 0.015625

by_family:
  checksum: 0.171875
  modchain: 0.015625
  revchain: 0.046875
```

Decision:

```text
Discard. Prefix-depth anchor can raise the active-length floor at a later
checkpoint, but it collapses the weakest family. Because family_floor restored
step 400, the final checkpoint is weaker than causalactive_triage.

New root suspicion:
  The source len4 checkpoint was trained before the revchain active-length
  semantic fix. Continuing len6 from that checkpoint may be a contaminated
  transfer. The next orthodox step is to rebuild the len4 canonical baseline
  from scratch under fixed family-aware active-length semantics, then transfer
  len4 -> len6 again.
```

Fixed-semantics len4 scratch rebase:

```text
local_eval/qtrm_native_reversed_efficiency_len4_causalactive_rebase_20260514/len4_seed777

recipe:
  resume_from: none
  steps: 1800
  lr: 3.0e-5
  depth_intermediate_loss_weight: 0.03
  answer_space_ranking_loss_weight: 0.03
  active_len_replay_loss_weight: 0.04
  retention_kl_loss_weight: 0.0

best_periodic_eval:
  restored_step: 600
  step600 full_generation_exact: 0.036458333333333336
  step600 min_active_len_generation_exact: 0.015873015873015872
  step600 min_family_generation_exact: 0.03125

final:
  full_generation_exact: 0.049479166666666664
  full_minus_think0: 0.026041666666666664
  full_minus_worst_ablation: -0.005208333333333336
  min_family_generation_exact: 0.03125

by_family:
  checksum: 0.0703125
  modchain: 0.046875
  revchain: 0.03125
```

Decision:

```text
Discard. The fixed-semantics scratch hardfamily recipe does not reproduce the
old len4 baseline. It also fails the causal ablation requirement because
z_l_zero beats the full model.

Revised diagnosis:
  The old len4 baseline came from a staged path:
    lrsched scratch -> phase2 -> hardfamily
  The next run should reproduce the first lrsched stage under fixed semantics,
  not jump directly to the hardfamily fine-tune recipe.
```

Fixed-semantics lrsched stage:

```text
local_eval/qtrm_native_reversed_efficiency_len4_lrsched_causalactive_20260514/len4_seed777

recipe:
  resume_from: none
  steps: 1800
  lr: 8.0e-5
  depth_intermediate_loss_weight: 0.20
  answer_space_ranking_loss_weight: 0.02
  active_len_replay_loss_weight: 0.02
  task_families: checksum,modchain,revchain

periodic:
  step600 full_generation_exact: 0.052083333333333336
  step600 min_family_generation_exact: 0.015625
  step1200 full_generation_exact: 0.09895833333333333
  step1200 min_family_generation_exact: 0.0625
  step1800 full_generation_exact: 0.16145833333333334
  step1800 min_family_generation_exact: 0.0625

final:
  full_generation_exact: 0.17447916666666666
  full_minus_think0: 0.12760416666666666
  full_minus_worst_ablation: 0.10677083333333333
  min_family_generation_exact: 0.0546875

by_family:
  checksum: 0.3984375
  modchain: 0.0703125
  revchain: 0.0546875

by_active_len:
  len2: 0.24031007751937986
  len3: 0.16279069767441862
  len4: 0.11904761904761904
```

Decision:

```text
Keep as fixed-semantics stage1. This beats the old lrsched checkpoint on full
exactness and preserves a positive causal ablation gap. It does not solve the
len4 hard-family bottleneck, but it is a valid baseline for rebuilding the
old staged path under corrected revchain active-length semantics.

Next:
  fixed lrsched -> fixed phase2 -> fixed hardfamily -> len6 transfer.
```

Fixed-semantics phase2 stage:

```text
local_eval/qtrm_native_reversed_efficiency_len4_phase2_causalactive_20260514/len4_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_lrsched_causalactive_20260514/len4_seed777/last.pt

recipe:
  steps: 1600
  lr: 4.0e-5
  depth_intermediate_loss_weight: 0.05
  answer_space_ranking_loss_weight: 0.03
  active_len_replay_loss_weight: 0.04
  task_families: checksum,modchain,revchain

periodic:
  step400 full_generation_exact: 0.20833333333333334
  step400 min_family_generation_exact: 0.078125
  step800 full_generation_exact: 0.22395833333333334
  step800 min_family_generation_exact: 0.09375
  step1200 full_generation_exact: 0.296875
  step1200 min_family_generation_exact: 0.0625
  step1600 full_generation_exact: 0.2864583333333333
  step1600 min_family_generation_exact: 0.078125

final_restored_by_family_floor:
  restored_step: 800
  full_generation_exact: 0.23697916666666666
  full_minus_think0: 0.20833333333333331
  full_minus_worst_ablation: 0.16666666666666666
  min_family_generation_exact: 0.0625

by_family:
  checksum: 0.515625
  modchain: 0.0625
  revchain: 0.1328125

by_active_len:
  len2: 0.26356589147286824
  len3: 0.2713178294573643
  len4: 0.1746031746031746
```

Decision:

```text
Keep as fixed-semantics stage2. This is slightly below the old phase2 full and
family floor, but it has a strong causal ablation gap under the corrected
revchain active-length semantics. Continue the staged reconstruction with the
hardfamily fine-tune.
```

Fixed-semantics hardfamily stage:

```text
local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_causalactive_20260514/len4_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_phase2_causalactive_20260514/len4_seed777/last.pt

recipe:
  steps: 1200
  lr: 3.0e-5
  depth_intermediate_loss_weight: 0.03
  answer_space_ranking_loss_weight: 0.03
  active_len_replay_loss_weight: 0.04
  retention_kl_loss_weight: 0.02
  task_families: modchain,revchain,modchain,revchain,checksum

periodic:
  step300 full_generation_exact: 0.2916666666666667
  step300 min_family_generation_exact: 0.125
  step600 full_generation_exact: 0.2552083333333333
  step600 min_family_generation_exact: 0.109375
  step900 full_generation_exact: 0.3072916666666667
  step900 min_family_generation_exact: 0.078125
  step1200 full_generation_exact: 0.3020833333333333
  step1200 min_family_generation_exact: 0.09375

final_restored_by_family_floor:
  restored_step: 300
  full_generation_exact: 0.2630208333333333
  full_minus_think0: 0.23177083333333331
  full_minus_worst_ablation: 0.1953125
  min_family_generation_exact: 0.09375

by_family:
  checksum: 0.5859375
  modchain: 0.109375
  revchain: 0.09375

by_active_len:
  len2: 0.3333333333333333
  len3: 0.27906976744186046
  len4: 0.1746031746031746
```

Decision:

```text
Keep as corrected len4 baseline. It is below the old contaminated len4
canonical checkpoint, but it was trained and evaluated under fixed family-aware
active-length semantics and has a strong causal ablation gap. Use this for
the next len4 -> len6 transfer attempt.
```

Corrected len4 -> len6 transfer:

```text
local_eval/qtrm_native_reversed_efficiency_len6_causalactive_transfer_20260514/len6_seed777

resume_from:
  local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_causalactive_20260514/len4_seed777/last.pt

recipe:
  steps: 1200
  lr: 3.0e-5
  depth_intermediate_loss_weight: 0.03
  answer_space_ranking_loss_weight: 0.03
  active_len_replay_loss_weight: 0.04
  retention_kl_loss_weight: 0.02
  task_families: modchain,revchain,modchain,revchain,checksum

periodic:
  step400 full_generation_exact: 0.125
  step400 min_family_generation_exact: 0.0625
  step800 full_generation_exact: 0.18229166666666666
  step800 min_active_len_generation_exact: 0.125
  step800 min_family_generation_exact: 0.0625
  step1200 full_generation_exact: 0.17708333333333334
  step1200 min_family_generation_exact: 0.015625

final_restored_by_family_floor:
  restored_step: 800
  full_generation_exact: 0.18229166666666666
  full_minus_think0: 0.16145833333333331
  full_minus_worst_ablation: 0.13020833333333331
  min_family_generation_exact: 0.0625

by_family:
  checksum: 0.375
  modchain: 0.0625
  revchain: 0.109375

by_active_len:
  len3: 0.14583333333333334
  len4: 0.2708333333333333
  len5: 0.125
  len6: 0.1875
```

Decision:

```text
Keep as the corrected len6 transfer baseline. It does not solve len6, but it
beats the earlier corrected len6 triages and confirms the staged fixed
semantics path is the right reset. The current weakest family is modchain, so
the next falsifiable run should bias the data schedule toward modchain while
keeping revchain/checksum through retention and mixed eval.
```

### Len6 Modchain Refine Seesaw

Autoresearch-style operation note: this run changes one mechanism only, the
family sampling schedule. It is not an architecture promotion.

```text
run_id: len6_modchain_refine
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_causalactive_transfer_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_modchain_refine_20260514/len6_seed777/report.json

changed_mechanism:
  TASK_FAMILIES=modchain,modchain,modchain,revchain,checksum
  STEPS=800
  LR=2e-5
  RETENTION_KL_LOSS_WEIGHT=0.03

periodic:
  step400 full_generation_exact: 0.20833333333333334
  step400 min_family_generation_exact: 0.046875
  step800 full_generation_exact: 0.21875
  step800 min_family_generation_exact: 0.03125

final_restored_by_family_floor:
  restored_step: 400
  full_generation_exact: 0.20833333333333334
  full_minus_think0: 0.17708333333333334
  full_minus_worst_ablation: 0.15104166666666669
  min_family_generation_exact: 0.046875

by_family:
  checksum: 0.484375
  modchain: 0.09375
  revchain: 0.046875

by_active_len:
  len3: 0.1875
  len4: 0.22916666666666666
  len5: 0.20833333333333334
  len6: 0.20833333333333334
```

Decision:

```text
Probe-only. The schedule does what it was designed to do: it improves mean
accuracy and raises modchain from 0.0625 to 0.09375. It fails the promotion
rule because revchain falls from 0.109375 to 0.046875 and the family floor
drops from 0.0625 to 0.046875. This exposes a family-seesaw bottleneck.

Next falsifiable operation: checkpoint interpolation/model-soup between the
balanced len6 transfer checkpoint and the modchain-refine checkpoint. Accept
only if full_generation_exact increases without lowering min_family below the
balanced checkpoint.
```

### Len6 Soup Triage

Autoresearch-style operation note: this is an experiment-operation mechanism,
not a new model architecture. It tests whether the balanced transfer checkpoint
and the modchain-biased checkpoint live in a compatible basin.

```text
run_id: len6_soup_triage
summary:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/summary.json

base_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_causalactive_transfer_20260514/len6_seed777/last.pt
candidate_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_modchain_refine_20260514/len6_seed777/last.pt

mechanism:
  linear_model_soup
  formula: (1-alpha) * base + alpha * candidate
```

Triage results:

```text
baseline transfer:
  full_generation_exact: 0.18229166666666666
  min_family_generation_exact: 0.0625
  by_family:
    checksum: 0.375
    modchain: 0.0625
    revchain: 0.109375

modchain-refine parent:
  full_generation_exact: 0.20833333333333334
  min_family_generation_exact: 0.046875
  by_family:
    checksum: 0.484375
    modchain: 0.09375
    revchain: 0.046875

alpha=0.25:
  full_generation_exact: 0.171875
  min_family_generation_exact: 0.0625

alpha=0.50:
  full_generation_exact: 0.203125
  min_family_generation_exact: 0.078125

alpha=0.625:
  full_generation_exact: 0.203125
  min_family_generation_exact: 0.0625

alpha=0.75:
  checkpoint:
    local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
  report:
    local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/eval_alpha_075/len6_seed777/report.json
  full_generation_exact: 0.21875
  full_minus_think0: 0.19270833333333334
  full_minus_worst_ablation: 0.16145833333333334
  min_family_generation_exact: 0.078125
  by_family:
    checksum: 0.484375
    modchain: 0.09375
    revchain: 0.078125
  by_active_len:
    len3: 0.25
    len4: 0.22916666666666666
    len5: 0.22916666666666666
    len6: 0.16666666666666666

alpha=0.875:
  full_generation_exact: 0.19791666666666666
  min_family_generation_exact: 0.046875
```

Decision:

```text
Keep alpha=0.75 as the current len6 soup candidate. It improves both decisive
metrics over the corrected len6 transfer baseline:

  full_generation_exact: 0.18229166666666666 -> 0.21875
  min_family_generation_exact: 0.0625 -> 0.078125

It also preserves a causal gap:

  full_minus_think0: 0.19270833333333334
  full_minus_worst_ablation: 0.16145833333333334

This does not solve len6 or promote a final architecture. It resolves one
operational bottleneck: family-specialized checkpoints can be combined better
than either parent through interpolation. The next run should start from
alpha_075 and use a short balanced fine-tune with family-floor checkpoint
selection.
```

### Len6 Soup Balanced Fine-Tune Failure

```text
run_id: len6_soup_balanced_ft
resume_from:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_balanced_ft_20260514/len6_seed777/report.json

changed_mechanism:
  short balanced fine-tune from alpha_075
  STEPS=600
  LR=1e-5
  RETENTION_REFERENCE_CHECKPOINT=resume
  RETENTION_KL_LOSS_WEIGHT=0.04
```

Result:

```text
periodic step200:
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0

periodic step400:
  full_generation_exact: 0.14583333333333334
  min_family_generation_exact: 0.015625

periodic step600 / restored final:
  full_generation_exact: 0.23958333333333334
  full_minus_think0: 0.21354166666666669
  full_minus_worst_ablation: 0.17708333333333334
  min_family_generation_exact: 0.046875
  by_family:
    checksum: 0.625
    modchain: 0.046875
    revchain: 0.046875
```

Decision:

```text
Discard as a checkpoint even though mean accuracy improved. It regresses the
alpha_075 soup family floor from 0.078125 to 0.046875 and shows checksum
dominance. The next canonical len6 baseline remains:

  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt

New operations bottleneck found: periodic restore did not consider the initial
resume checkpoint as a best candidate, so a fine-tune run could overwrite a
better starting point. The training script now supports:

  --eval-initial-checkpoint

Wrapper env:

  EVAL_INITIAL_CHECKPOINT=1

CUDA smoke:

  local_eval/qtrm_native_eval_initial_smoke_cuda_20260514/smoke_seed777/report.json

The smoke confirms `periodic_eval[0].step == 0`,
`periodic_eval[0].source == initial_checkpoint`, and
`restored_best_eval_checkpoint == true`.
```

### Len6 Soup Hard-Family Fine-Tune Failure

```text
run_id: len6_soup_hardfamily_ft
resume_from:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_hardfamily_ft_20260514/len6_seed777/report.json

changed_mechanism:
  TASK_FAMILIES=modchain,revchain,modchain,revchain
  STEPS=400
  LR=5e-6
  RETENTION_REFERENCE_CHECKPOINT=resume
  RETENTION_KL_LOSS_WEIGHT=0.08
  EVAL_INITIAL_CHECKPOINT=1
```

Result:

```text
step0:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125

step200:
  full_generation_exact: 0.18229166666666666
  min_family_generation_exact: 0.046875

step400:
  full_generation_exact: 0.11979166666666667
  min_family_generation_exact: 0.046875

final restored:
  restored_best_eval_checkpoint: true
  best_periodic_eval.step: 0
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125
```

Decision:

```text
Discard as a training improvement. The step0 guard worked and preserved the
soup baseline, but the hard-family CE fine-tune did not improve modchain or
revchain. This strengthens the diagnosis: the next bottleneck is hard-family
reasoning quality, not simply more gradient steps on the hard families.
```

### Len6 Soup Plus Fine-Tune Soup Failure

```text
run_id: len6_soup_ft_soup_triage
summary:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_ft_soup_triage_20260514/summary.json

base_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
candidate_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_balanced_ft_20260514/len6_seed777/last.pt
```

Results:

```text
baseline alpha_075:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125
  by_family:
    checksum: 0.484375
    modchain: 0.09375
    revchain: 0.078125

alpha=0.125:
  full_generation_exact: 0.22395833333333334
  min_family_generation_exact: 0.0625
  by_family:
    checksum: 0.53125
    modchain: 0.078125
    revchain: 0.0625

alpha=0.25:
  full_generation_exact: 0.22916666666666666
  min_family_generation_exact: 0.0625
  by_family:
    checksum: 0.5625
    modchain: 0.0625
    revchain: 0.0625

alpha=0.50:
  full_generation_exact: 0.234375
  min_family_generation_exact: 0.046875
  by_family:
    checksum: 0.59375
    modchain: 0.046875
    revchain: 0.0625
```

Decision:

```text
Discard this interpolation direction. It imports the checksum/mean gain from
the balanced fine-tune, but every tested ratio lowers the family floor below
the alpha_075 soup baseline. The current candidate remains:

  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
```

### Len6 Family-DRO Fine-Tune Failure

```text
run_id: len6_family_dro_ft
report:
  local_eval/qtrm_native_reversed_efficiency_len6_family_dro_ft_20260514/len6_seed777/report.json

baseline_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt

changed_mechanism:
  family_dro_loss_weight: 0.25
  family_dro_temperature: 0.0
  LR: 5e-6
  steps: 400
  restore_best_eval_checkpoint: true
  eval_initial_checkpoint: true
```

Result:

```text
step0:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125

step200:
  full_generation_exact: 0.234375
  min_family_generation_exact: 0.0625

step400:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.046875

final restored:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125
  full_minus_think0: 0.19270833333333334
  full_minus_worst_ablation: 0.16145833333333334
```

Decision:

```text
Discard as a checkpoint improvement. The operation was useful because the
step0 restore guard preserved the alpha_075 baseline, but worst-family answer
loss alone did not improve the weak modchain/revchain floor. The hard-family
bottleneck is therefore not just average CE imbalance. The next mechanism must
target the transition/credit-assignment path for ordered composition, not only
reweight final answer loss.
```

### Len6 Depth-Intermediate Family-DRO Failure

```text
run_id: len6_depth_family_dro_ft
report:
  local_eval/qtrm_native_reversed_efficiency_len6_depth_family_dro_ft_20260514/len6_seed777/report.json

baseline_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt

changed_mechanism:
  depth_intermediate_family_dro: true
  depth_intermediate_family_dro_temperature: 0.0
  depth_intermediate_loss_weight: 0.03
  final family_dro_loss_weight: 0.0
  LR: 5e-6
  steps: 400
  restore_best_eval_checkpoint: true
  eval_initial_checkpoint: true
```

Result:

```text
step0:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125

step200:
  full_generation_exact: 0.234375
  min_family_generation_exact: 0.0625

step400:
  full_generation_exact: 0.22916666666666666
  min_family_generation_exact: 0.046875

final restored:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.078125
  full_minus_think0: 0.19270833333333334
  full_minus_worst_ablation: 0.16145833333333334
```

Decision:

```text
Discard as a checkpoint improvement. This is a stronger negative result than
the final-answer family-DRO failure: even reweighting the per-depth transition
targets toward the weakest family does not improve the hard-family floor. The
next experiment should not be another CE reweighting variant. It should repair
the coupling between recurrent state transition and the LM answer readout, or
diagnose whether the learned state contains the hard-family answer while the
decoder fails to expose it.
```

### Len6 Alpha075 Core-State Probe Diagnostic

```text
run_id:
  len6_core_probe_alpha075
  len6_core_probe_alpha075_mean
  len6_core_probe_alpha075_flatten

reports:
  local_eval/qtrm_native_reversed_efficiency_len6_core_probe_alpha075_20260514/len6_seed777/report.json
  local_eval/qtrm_native_reversed_efficiency_len6_core_probe_alpha075_mean_20260514/len6_seed777/report.json
  local_eval/qtrm_native_reversed_efficiency_len6_core_probe_alpha075_flatten_20260514/len6_seed777/report.json

baseline_checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
```

Generation baseline:

```text
full_generation_exact: 0.21875
by_family:
  checksum: 0.484375
  modchain: 0.09375
  revchain: 0.078125
```

Core answer probe:

```text
pooling=last:
  exact: 0.1041666641831398
  by_family:
    checksum: 0.203125
    modchain: 0.046875
    revchain: 0.0625

pooling=mean:
  exact: 0.03125
  by_family:
    checksum: 0.0625
    modchain: 0.015625
    revchain: 0.015625

pooling=flatten:
  exact: 0.0572916679084301
  by_family:
    checksum: 0.109375
    modchain: 0.03125
    revchain: 0.03125
```

Core step probe:

```text
pooling=last:
  exact: 0.1232638880610466
  by_depth:
    depth1: 0.19791666666666666
    depth2: 0.10416666666666667
    depth3: 0.06770833333333333
  by_family:
    checksum: 0.17708333333333334
    modchain: 0.125
    revchain: 0.06770833333333333

pooling=mean:
  exact: 0.09375
  by_depth:
    depth1: 0.19270833333333334
    depth2: 0.046875
    depth3: 0.041666666666666664

pooling=flatten:
  exact: 0.0989583358168602
  by_depth:
    depth1: 0.16145833333333334
    depth2: 0.078125
    depth3: 0.057291666666666664
```

Interpretation:

```text
The failure is not simply that the LM readout ignores a strong hidden answer.
Linear probes over h+l states are weaker than the LM generation result, and
the step probe degrades sharply from depth1 to depth3. `last` pooling is the
best probe, so the answer is not merely hidden in other token positions.

Active bottleneck:
  recurrent state-transition semantic drift / decay across deeper thinking
  steps, especially revchain.

Next non-repeat direction:
  change the recurrent transition or its semantic preservation mechanism.
  Do not spend more runs on final-answer CE reweighting, intermediate CE
  reweighting, mean/flatten readout pooling, or anti-collapse alone.
```

## Len6 Semantic-Carry Transition Probe

Run:

```text
run_id: len6_semantic_carry_ft
baseline: local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
changed mechanism:
  trm_dual_z_reversed_hybrid_3to1_semantic_carry
  adds small learned gates that pull z_L/z_H toward their initial
  prompt-conditioned anchor state after each recurrent update.
```

Triage result at 192 eval cases:

```text
step0:
  full_generation_exact: 0.21875
  min_family_generation_exact: 0.0625

best restored step200:
  full_generation_exact: 0.23958333333333334
  min_family_generation_exact: 0.0625
  full_minus_think0: 0.21354166666666669
  full_minus_worst_ablation: 0.17708333333333334

step400:
  full_generation_exact: 0.2760416666666667
  min_family_generation_exact: 0.046875
```

Fair 512-case re-evaluation:

```text
full_generation_exact: 0.21484375
min_family_generation_exact: 0.047058823529411764
full_minus_think0: 0.1796875
full_minus_worst_ablation: 0.15625
by_family:
  checksum: 0.543859649122807
  modchain: 0.05263157894736842
  revchain: 0.047058823529411764
```

Decision:

```text
discard-familyfloor as a canonical checkpoint.
keep the semantic-carry structure as a diagnostic/probe option only.
```

Interpretation:

```text
Semantic carry can improve mean/full exactness for short triage and makes
z_H strongly causal, but it still overfits the easy checksum family and does
not repair the hard-family transition. The bottleneck is now narrower:

  recurrent transition needs family/operation-order-balanced state evolution,
  not only prompt-anchor preservation.
```

Next non-repeat direction:

```text
Do not rerun plain semantic-carry fine-tuning. The next candidate must target
operation-order-balanced transition dynamics directly, such as an explicit
reverse/forward family-conditioned update schedule, per-family transition
router inside the core, or a state transition objective that prevents checksum
dominance while preserving revchain/modchain.
```

## Len6 Order-Router Transition Probe

Run:

```text
run_id: len6_order_router_small_ft
baseline: local_eval/qtrm_native_reversed_efficiency_len6_soup_triage_20260514/alpha_075/avg.pt
changed mechanism:
  trm_dual_z_reversed_hybrid_3to1_order_router
  computes both L->H and reverse-primed H->L->H transition orders, then mixes
  them with a prompt-state router initialized toward the old L->H path.
```

Operational note:

```text
The first batch20 run with answer-space ranking crashed with CUDA OOM.
The successful triage used batch8 and disabled answer-space ranking so the
result isolates transition dynamics rather than auxiliary candidate scoring.
```

Small 128-case triage:

```text
step0:
  full_generation_exact: 0.2109375
  min_family_generation_exact: 0.046511627906976744

step150:
  full_generation_exact: 0.203125
  min_family_generation_exact: 0.023809523809523808

step300:
  full_generation_exact: 0.2421875
  min_active_len_generation_exact: 0.21875
  min_family_generation_exact: 0.047619047619047616
  full_minus_think0: 0.203125
  full_minus_worst_ablation: 0.1796875
```

Decision:

```text
probe-mean-gain only.
Do not promote without a 512-case family-floor improvement over alpha_075.
```

Interpretation:

```text
Order-router confirms that transition order is a real knob: full exactness and
active-length floor can rise after only 300 small-batch steps. But the family
floor is still too low and checksum still dominates:

  checksum: 0.6046511627906976
  modchain: 0.06976744186046512
  revchain: 0.047619047619047616

The next non-repeat step should not add more readout/loss reweighting. It
should make the router explicitly measurable: log route probabilities by
family and test whether revchain/modchain actually choose the reverse-primed
path. If they do not, train the router with a small family/order auxiliary or
change the order candidates.
```

No-train route telemetry on the trained order-router checkpoint:

```text
run_id: len6_order_router_route_probe
checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_probe_20260514/len6_seed777/report.json
eval_cases: 128
full_generation_exact: 0.2421875
min_family_generation_exact: 0.047619047619047616
full_minus_think0: 0.203125
full_minus_worst_ablation: 0.1796875

router route names:
  route0 = L->H
  route1 = H->L->H

overall:
  last_lh_prob: 0.9822192192077637
  last_hlh_prob: 0.017780806869268417

by_family last_hlh_prob:
  checksum: 0.01776897720992565
  modchain: 0.017777688801288605
  revchain: 0.01779610849916935
```

Diagnosis:

```text
The router did not learn family/order specialization. It remains pinned to the
initial L->H-biased path for checksum, modchain, and revchain alike. Therefore
the current order-router result is not evidence that reverse-primed routing
helps revchain/modchain; it only shows that the extra structure can preserve or
slightly alter the old L->H transition dynamics.
```

Next non-repeat operation:

```text
Do not rerun plain order_router_small_ft. The next candidate must make route
choice causally trainable and measurable:

  option A: add a small router auxiliary target from family causal order
            (modchain/checksum prefer L->H, revchain prefer H->L->H),
            with low weight and keep LM-logit answer path unchanged.

  option B: remove the free router and use deterministic family-token-gated
            order during training only as a teacher/probe, then distill back
            into a learned router.

  option C: change the H->L->H candidate because a route that is never chosen
            may be too weak or too expensive to learn through answer CE alone.

Acceptance remains family-floor based; route specialization alone is not a
promotion unless full generation and ablation gaps improve.
```

## Len6 Order-Router Auxiliary Fastprobe

Implementation note:

```text
Added an optional family-order router auxiliary:

  --order-router-aux-loss-weight
  --order-router-aux-target-mode family_order
  --order-router-lr-multiplier

The target is not an answer sidecar. It only marks the intended recurrent
update order:

  checksum/modchain -> route0 L->H
  revchain          -> route1 H->L->H

The answer still comes from the normal recursive core -> decoder -> LM logits
path.
```

Restore-best run:

```text
run_id: len6_order_router_aux_ft
checkpoint:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
aux_weight: 0.2
router_lr_multiplier: 1.0

periodic:
  step0:
    full_generation_exact: 0.2421875
    min_family_generation_exact: 0.047619047619047616
  step150:
    full_generation_exact: 0.171875
    min_family_generation_exact: 0.023255813953488372
  step300:
    full_generation_exact: 0.21875
    min_family_generation_exact: 0.023255813953488372

decision: discard-restored
```

No-restore diagnostics:

```text
run_id: len6_order_router_aux_norestore_probe
aux_weight: 0.2
router_lr_multiplier: 1.0

full_generation_exact: 0.203125
min_family_generation_exact: 0.0
by_family:
  checksum: 0.5454545454545454
  modchain: 0.047619047619047616
  revchain: 0.0
by_family last_hlh_prob:
  checksum: 0.018175365403294563
  modchain: 0.01820817030966282
  revchain: 0.018283013254404068
```

Fast router-LR diagnostic:

```text
run_id: len6_order_router_aux_fastprobe
aux_weight: 0.2
router_lr_multiplier: 1000.0

full_generation_exact: 0.203125
min_family_generation_exact: 0.0
full_minus_think0: 0.171875
full_minus_worst_ablation: 0.140625
by_family:
  checksum: 0.5454545454545454
  modchain: 0.047619047619047616
  revchain: 0.0
by_family last_hlh_prob:
  checksum: 0.10700041800737381
  modchain: 0.30114322900772095
  revchain: 0.43261656165122986
```

Diagnosis:

```text
The router can be moved when given a separate LR, so the original failure was
partly a trainability problem. But moving revchain into the reverse-primed
H->L->H path makes revchain generation worse, not better. Therefore the active
bottleneck is no longer just "router does not learn"; it is:

  the H->L->H reverse candidate does not yet produce a useful causal state for
  the LM answer path.
```

Next non-repeat direction:

```text
Do not keep increasing router auxiliary weight. That only forces selection of
a weak route. The next candidate must improve or pretrain the reverse candidate
itself before routing can help:

  1. add route-specific transition distillation:
       train H->L->H to match correct intermediate revchain prefixes while
       keeping final answers on LM logits;

  2. add a route ablation report:
       force route0, force route1, and learned route on the same checkpoint to
       measure whether route1 is intrinsically weak;

  3. if route1 is weak, replace H->L->H with a suffix-conditioned revchain
       transition candidate instead of another router/loss tweak.
```

## Len6 Order-Router Force Ablation

Autoresearch-style operation result:

```text
run_id: len6_order_router_force_ablation
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_force_ablation_20260514/len6_seed777/report.json

learned/full generation_exact: 0.2421875
forced route0 L->H:           0.2421875
forced route1 H->L->H:        0.1015625

by_family forced route0:
  checksum: 0.6046511627906976
  modchain: 0.06976744186046512
  revchain: 0.047619047619047616

by_family forced route1:
  checksum: 0.16279069767441862
  modchain: 0.046511627906976744
  revchain: 0.09523809523809523
```

Diagnosis:

```text
The learned router is pinned to route0, but forcing route1 is not a hidden win.
It slightly improves revchain while destroying checksum and full accuracy. This
means the active bottleneck is not "the router cannot choose route1"; it is:

  route1 H->L->H does not yet create a generally useful causal state for the
  LM answer path.

Therefore, the next experiment should not be another router auxiliary weight
sweep. The next experiment must improve the reverse transition itself, preferably
with a small route-specific transition pretraining/distillation objective that
still renders through the normal LM logits path.
```

Autoresearch operating rule for this stage:

```text
Use short fixed-budget probes with one decisive metric:

  metric: family-floor generation exact on held-out len6 mixed families
  keep:   full and family floor improve without losing causal ablation gaps
  discard: mean-only gains that lower revchain/modchain floor
  branch: every rejected run must name one narrower bottleneck

This is the part of karpathy/autoresearch that applies directly to QTRM. The
part that should not be copied directly is unconstrained free editing of all
training code; QTRM needs locked gate scripts, explicit ablations, and a ledger
because otherwise architecture search can produce non-causal shortcut wins.
```

## Len6 Forced-Route Answer CE

Implementation note:

```text
Added optional forced-route LM answer supervision:

  --forced-route-answer-loss-weight
  --forced-route-answer-route
  --forced-route-answer-families
  --forced-route-answer-max-cases
  --forced-route-answer-every

This is not a router auxiliary. It forces route1 during the forward pass and
applies the normal answer-token CE through the existing decoder/LM head. The
router receives no gradient from this loss; the intended pressure is on the
route candidate state transition and LM readout.
```

Diagnostic run:

```text
run_id: len6_forced_route_answer_ft
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_forced_route_answer_ft_20260514/len6_seed777/report.json

forced_route_answer_loss_weight: 0.5
forced_route_answer_route: 1
forced_route_answer_families: revchain
forced_route_answer_max_cases: 4
steps: 300

learned/full generation_exact:
  before: 0.2421875
  after:  0.1875

forced route1 generation_exact:
  before: 0.1015625
  after:  0.1875

forced route1 by_family after:
  checksum: 0.3953488372093023
  modchain: 0.09302325581395349
  revchain: 0.07142857142857142
```

Diagnosis:

```text
Final-answer CE alone is not enough. It raises route1's aggregate score, but
the gain comes mostly from checksum/modchain behavior while revchain, the target
family, gets worse than the forced-route ablation baseline.

This rejects "just train route1 on final answers" as the next canonical fix.
The next non-repeat experiment should supervise the route transition before the
final answer, for example:

  route1 forced depth-intermediate CE on revchain causal suffix prefixes

or replace route1 with a suffix-conditioned transition that has an explicit
reverse-order state update. The acceptance metric remains family floor and
forced-route revchain, not mean-only route1 improvement.
```

## Len6 Forced-Route Depth CE

Implementation note:

```text
Added optional forced-route depth supervision:

  --forced-route-depth-loss-weight
  --forced-route-depth-route
  --forced-route-depth-families
  --forced-route-depth-max-cases
  --forced-route-depth-every
  --forced-route-depth-min-depth
  --forced-route-depth-weight-power

This forces a selected route and applies the existing intermediate-depth answer
targets through normal LM logits. It still does not train the router.
```

Diagnostic run:

```text
run_id: len6_forced_route_depth_ft
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_forced_route_depth_ft_20260514/len6_seed777/report.json

forced_route_depth_loss_weight: 0.3
forced_route_depth_route: 1
forced_route_depth_families: revchain
forced_route_depth_max_cases: 4
steps: 240

learned/full generation_exact:
  before: 0.2421875
  after:  0.1171875

forced route1 generation_exact:
  before: 0.1015625
  after:  0.1015625

forced route1 revchain:
  before: 0.09523809523809523
  after:  0.09523809523809523
```

Diagnosis:

```text
This rejects both simple CE variants:

  final answer CE on forced route1
  intermediate depth CE on forced route1

Neither turns H->L->H into a useful reverse transition. The active bottleneck is
therefore structural, not just loss placement:

  route1's H->L->H candidate is not an adequate reverse-order recurrent state
  transition for revchain.

Next non-repeat action:

  remove or replace H->L->H as route1. Test an explicit suffix-conditioned
  transition where revchain route1 reads the causal suffix state directly,
  instead of hoping a generic pre-H update discovers reverse order.
```

## Len6 Recent Order-Router Probe

Implementation note:

```text
Added:

  trm_dual_z_reversed_hybrid_3to1_recent_order_router

This keeps the same universal LM path and same route0 as the order-router
variant. Only route1 changes: instead of plain encoded context, it uses a
causal recency-biased EMA context before the H->L->H candidate. This avoids
non-causal sequence reversal while making the route more sensitive to the
suffix-heavy revchain prompt layout.
```

No-train probe:

```text
run_id: len6_recent_order_router_probe
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_recent_order_router_probe_20260514/len6_seed777/report.json

learned/full generation_exact:
  original order-router: 0.2421875
  recent route1:         0.234375

forced route1 generation_exact:
  original H->L->H:      0.1015625
  recent route1:         0.0859375

forced route1 revchain:
  original H->L->H:      0.09523809523809523
  recent route1:         0.07142857142857142
```

Diagnosis:

```text
Reject this structure. A generic causal recency bias is still not enough to
create reverse-order computation. It slightly preserves learned/full because
the learned router remains pinned to route0, but forced route1 gets worse.

This narrows the bottleneck again:

  the model needs an explicit operation/order state representation for revchain,
  not just another route order, final CE, intermediate CE, or suffix bias.

Next viable family:

  explicit causal operation-state transition inside the recursive core, with
  answer still rendered through LM logits and with ablations proving the state
  is used.
```

## Len6 State-GRU Order-Router Probe

Implementation note:

```text
Added:

  trm_dual_z_reversed_hybrid_3to1_state_gru_order_router

This keeps the normal QTRM-native causal path:

  prompt tokens -> native encoder -> mandatory recursive core -> LM logits

Only route1 changes. Instead of plain encoded context or hand-built recency
context, route1 receives a trainable causal GRU order-state context:

  encoded -> GRU order state -> residual LayerNorm -> H->L->H candidate

The intent was to test whether a learned causal operation/order state can repair
the reverse-route candidate without adding a sidecar answer channel.
```

Diagnostic run:

```text
run_id: len6_state_gru_order_router_probe
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_state_gru_order_router_probe_20260514/len6_seed777/report.json

steps: 300
forced_route_depth_loss_weight: 0.2
forced_route_depth_route: 1
forced_route_depth_families: revchain

learned/full generation_exact:
  before: 0.2421875
  after:  0.1328125

forced route1 generation_exact:
  before: 0.1015625
  after:  0.09375

forced route1 revchain:
  before: 0.09523809523809523
  after:  0.11904761904761904
```

Diagnosis:

```text
Reject this as canonical despite the small route1 revchain improvement.

Positive signal:
  route1 revchain improves from 0.0952 to 0.1190.

Negative signal:
  learned/full collapses from 0.2422 to 0.1328.
  route1 full falls from 0.1016 to 0.0938.
  the learned router remains pinned to route0 with mean_hlh_prob around 0.018.

Conclusion:
  A trainable generic GRU order state is too weak and too unstable. It can carry
  a little reverse-family signal, but it does not improve the universal LM path.

Next non-repeat action:
  stop adding generic context wrappers around H->L->H. The next candidate must
  make operation order part of the recurrent transition itself, for example a
  position-bound recurrent state update whose ablations prove that source
  positions/ops, not a side answer head, drive the final LM logits.
```

## Len6 Transition-State Order-Router Probe

Implementation note:

```text
Added:

  trm_dual_z_reversed_hybrid_3to1_transition_state_order_router

This is stricter than the previous state-GRU context wrapper. Route1 now has a
trainable order-state that is recurrently updated inside each H/L transition:

  current z_L, z_H, context
  -> order transition GRUCell
  -> order state
  -> update-gated H/L candidate context
  -> LM logits

The answer still goes through the normal QTRM-native LM path. There is no
sidecar solver or answer head.
```

Diagnostic run:

```text
run_id: len6_transition_state_order_router_probe
baseline:
  local_eval/qtrm_native_reversed_efficiency_len6_order_router_small_ft_20260514/len6_seed777/last.pt
report:
  local_eval/qtrm_native_reversed_efficiency_len6_transition_state_order_router_probe_20260514/len6_seed777/report.json

steps: 300
lr: 4e-5
forced_route_depth_loss_weight: 0.15
retention_reference_checkpoint: resume
retention_kl_loss_weight: 0.05

learned/full generation_exact:
  before: 0.2421875
  after:  0.234375

forced route1 generation_exact:
  before: 0.1015625
  after:  0.0703125

forced route1 revchain:
  before: 0.09523809523809523
  after:  0.047619047619047616

forced route0 generation_exact:
  after: 0.2421875
```

Diagnosis:

```text
Reject this structure.

The retention KL did its job: route0/full remains close to the previous
baseline, so the run isolates the new route1 candidate more cleanly than the
state-GRU context-wrapper run.

But route1 itself gets worse. This rejects the idea that a generic learned
transition-order state is enough. The active bottleneck is more specific:

  the model is not reliably binding source operation positions and values into
  the recurrent state before applying reverse-order computation.

Next non-repeat action:
  stop adding generic order-state wrappers. Use the earlier accepted L1
  source-position/value binding insight: create a token-path source-position
  operation binder or trace-supervised transition target that feeds the
  recurrent LM path, then require source-binding-off and core-state ablations.
```

## Donor-Integrated Bridge Boundary

Question:

```text
If the donor is no longer frozen, can the donor path behave like QTRM-native?
```

Decision:

```text
Not strictly. A trainable donor can become a QTRM-integrated donor, but the
`QTRM-native` label remains reserved for donor-disabled runs.
```

The bridge is still useful. The intended causal path is:

```text
prompt tokens
-> Qwen donor with LoRA or partial unfreeze
-> donor hidden states
-> mandatory QTRM/TRM recurrent core
-> core-conditioned QTRM logits/residual
-> autoregressive LM output
```

Promotion requirements:

```text
full model beats donor-only on the same held-out reasoning gate;
full model beats core-off;
deeper recurrent steps beat shallow steps;
core-state/z_L/z_H ablations reduce the same LM-logit metric;
donor-logit scale can be annealed downward without repeated-token collapse.
```

Implementation added:

```text
src/qtrm_mm/qwen_donor.py
  donor.train_lora now enables PEFT LoRA instead of being ignored
  donor.train_last_n_layers supports partial full-precision unfreeze

src/qtrm_mm/training/train.py
  trainable donor parameters are added to the optimizer
  donor hidden states can keep gradient into donor LoRA/partial layers
  donor logits stay detached for teacher/baseline use by default
  optional train.loss_donor_lm_weight adds donor self-LM healing loss

configs/qwen35_2b_4090_qtrm_integrated_donor_lora_healing_s080.yaml
  first QTRM-integrated donor LoRA healing scaffold
```

Naming rule:

```text
frozen donor + QTRM sidecar       -> donor-backed QTRM adapter
trainable donor + mandatory core -> QTRM-integrated donor
donor disabled                   -> QTRM-native
```

## Dual-Path Reverse Active Core Freeze

Decision document:

```text
docs/wiki/decisions/qtrm-native-dual-path-reverse-active-architecture.md
```

Current active native hypothesis:

```text
dual_path_reverse
  implementation: trm_dual_z_reversed_hybrid_3to1
  baseline: official_trm_think
```

The active QTRM-native path is now:

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
-> LM logits
-> autoregressive text
```

Runner:

```bash
PYTHONPATH=src .venv/bin/python scripts/352_run_qtrm_native_dual_path_reverse_gate.py \
  --out-dir local_eval/qtrm_native_dual_path_reverse_gate \
  --profile short \
  --lengths 4,6,8 \
  --candidates official,dual_path_reverse
```

Research gate:

```bash
PYTHONPATH=src .venv/bin/python scripts/300_research_gate_runner.py \
  --gate qtrm_native_dual_path_reverse_length_gate \
  --profile standard
```

Rule:

```text
Do not resume architecture shopping while this gate is unresolved. If the gate
rejects, repair the failed axis in the same dual_path_reverse architecture:
exactness, depth gain, ablation margin, family coverage, or official-baseline
loss.
```

## Runner Result 2026-05-14T19:45:09

```text
gate: qtrm_native_l5_language_nonregression
target_level: L5C native language non-regression
profile: triage
decision: accepted_l5_language_nonregression
accepted: True
next_action: stay QTRM-native: if this was triage, run the standard language non-regression gate; if standard passes, broaden the native text slice before any backbone comparison
```

Decisive metrics:

```json
{
  "last_loss": 2.953779935836792,
  "eval_metrics.think_eval_loss": 2.988300154949057,
  "eval_metrics.think0_loss": 3.9122430414989076,
  "eval_metrics.thinking_block_off_loss": 3.9122430414989076,
  "eval_metrics.think0_baseline_loss": 2.94579855738015,
  "eval_metrics.loss_ratios.full_vs_think0": 0.7638329529251695,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.7638329529251695,
  "eval_metrics.loss_ratios.full_vs_baseline": 1.0144278696390923,
  "eval_metrics.sample_degeneracy.unique_chars": 12.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.06060606060606061,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_triage/report.json`

## Runner Result 2026-05-14T19:47:04

```text
gate: qtrm_native_l5_language_nonregression
target_level: L5C native language non-regression
profile: standard
decision: accepted_l5_language_nonregression
accepted: True
next_action: stay QTRM-native: if this was triage, run the standard language non-regression gate; if standard passes, broaden the native text slice before any backbone comparison
```

Decisive metrics:

```json
{
  "last_loss": 1.8722347021102905,
  "eval_metrics.think_eval_loss": 2.1474249631138047,
  "eval_metrics.think0_loss": 3.5913524429950403,
  "eval_metrics.thinking_block_off_loss": 3.5913524429950403,
  "eval_metrics.think0_baseline_loss": 2.0925279042627607,
  "eval_metrics.loss_ratios.full_vs_think0": 0.5979432531893029,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.5979432531893029,
  "eval_metrics.loss_ratios.full_vs_baseline": 1.0262348037219533,
  "eval_metrics.sample_degeneracy.unique_chars": 16.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.015151515151515152,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_standard/report.json`

## Runner Result 2026-05-14T19:53:51

```text
gate: qtrm_native_l4_mixed_text_reasoning
target_level: L4 native reasoning + language
profile: standard
decision: rejected
accepted: False
next_action: do not add MSA or external retrieval; fix core-to-text generation, capacity, curriculum, or depth supervision first
```

Decisive metrics:

```json
{
  "last_loss": 0.16500693559646606,
  "eval_metrics.think4.generation_exact": 0.6640625,
  "eval_metrics.think0.generation_exact": 0.02734375,
  "eval_metrics.state_reset.generation_exact": 0.033203125,
  "eval_metrics.op_zero.generation_exact": 0.044921875,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false,
  "decisive_metrics.full_generation_exact": 0.6640625,
  "decisive_metrics.think0_generation_exact": 0.02734375,
  "decisive_metrics.state_reset_generation_exact": 0.033203125,
  "decisive_metrics.op_zero_generation_exact": 0.044921875,
  "decisive_metrics.full_minus_think0": 0.63671875,
  "decisive_metrics.full_minus_worst_ablation": 0.619140625,
  "decisive_metrics.min_family_generation_exact": 0.6640625
}
```

Report: `local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/report.json`

## Runner Result 2026-05-14T20:02:48

```text
gate: qtrm_native_dual_reverse_l4_baseline_compare
target_level: L4 dual reverse versus single baseline
profile: triage
decision: command_failed
accepted: False
next_action: keep single recurrent MHA ETD as the active baseline; do not promote dual reverse until it beats 0.664 with depth and ablation margins
```

Decisive metrics:

```json
{}
```

Report: `local_eval/research_gate_runner/qtrm_native_dual_reverse_l4_baseline_compare_triage/report.json`

## Official-Schedule Split-Mixer Correction 2026-05-14

Decision:

```text
Do not use "reverse" to describe the canonical Mamba3/GatedDelta split-mixer
variant unless the TRM update order is actually reversed.
```

Canonical experiment:

```text
think_structure:
  trm_dual_z_nested_official_schedule_split_mixer_3to1

gate:
  qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
```

Fixed schedule:

```text
H outer steps: 3
L inner cycles per H step: 6

z_L mixer:
  Mamba3, Mamba3, Mamba3, Attention

z_H mixer:
  GatedDelta, GatedDelta, GatedDelta, Attention
```

Why:

```text
This preserves the official TRM H/L rhythm while testing the Qwen-style 3:1
linear-attention/attention idea inside the recursive z_L/z_H core. The old
`trm_dual_z_nested_reversed_hybrid_3to1` name is retained only for legacy
reports, not for future canonical promotion.
```

## Runner Result 2026-05-14T20:11:53

```text
gate: qtrm_native_dual_reverse_l4_baseline_compare
target_level: L4 dual reverse versus single baseline
profile: triage
decision: command_failed
accepted: False
next_action: keep single recurrent MHA ETD as the active baseline; do not promote dual reverse until it beats 0.664 with depth and ablation margins
```

Decisive metrics:

```json
{}
```

Report: `local_eval/research_gate_runner/qtrm_native_dual_reverse_l4_baseline_compare_triage/report.json`

## Runner Result 2026-05-14T21:52:50

```text
gate: qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
target_level: L4 nested official-schedule split-mixer 3:1 versus single baseline
profile: smoke
decision: smoke_passed_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
accepted: False
next_action: do not replace the accepted nested MHA repair; inspect whether the official H=3/L=6 Mamba3/GatedDelta split-mixer failed exact accuracy, z_L/z_H causality, or both before any further architecture shopping
```

Decisive metrics:

```json
{
  "last_loss": 4.68924617767334,
  "eval_metrics.think0.generation_exact": 0.0,
  "eval_metrics.state_reset.generation_exact": 0.0,
  "eval_metrics.op_zero.generation_exact": 0.0,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 3,
  "backend_summary.official_mamba3_mixers": 3,
  "backend_summary.torch_delta_mixers": 6,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": true,
  "decisive_metrics.full_generation_exact": 0.0,
  "decisive_metrics.think0_generation_exact": 0.0,
  "decisive_metrics.state_reset_generation_exact": 0.0,
  "decisive_metrics.op_zero_generation_exact": 0.0,
  "decisive_metrics.full_minus_think0": 0.0,
  "decisive_metrics.full_minus_worst_ablation": 0.0,
  "decisive_metrics.min_family_generation_exact": 0.0
}
```

Report: `local_eval/research_gate_runner/qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare_smoke/report.json`

## 2026 Diffusion / TST / Fast-Slow Update 2026-05-14

Sources:

```text
Mercury-2:
  https://www.inceptionlabs.ai/blog/introducing-mercury-2

Mercury diffusion LLM:
  https://arxiv.org/abs/2506.17298

Token Superposition Training:
  https://arxiv.org/abs/2605.06546

Fast-Slow Training:
  https://arxiv.org/abs/2605.12484

source note:
  docs/wiki/sources/diffusion-fast-slow-llm-2026.md
```

Architecture decision:

```text
Mercury-style diffusion is not the new canonical QTRM core. It is a future
decoder/readout candidate after the recursive core passes causality gates.
```

Current priority:

```text
Keep running:
  qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare

Active command:
  scripts/300_research_gate_runner.py
    --gate qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
    --profile standard

Active log:
  local_eval/background_logs/split_mixer_standard_20260514_220000.log
```

Why not pivot to Mercury now:

```text
The current bottleneck is not only decoding latency. It is whether native
z_L/z_H recursive state causally improves final LM-logit generation. A
standalone diffusion decoder could hide that bottleneck by becoming a new
answer generator.
```

How Mercury can be used later:

```text
qtrm_native_parallel_refinement_decoder_gate

prompt tokens
-> native encoder
-> mandatory QTRM/TRM core
-> core-conditioned parallel/diffusion answer refinement
-> shared token projection / LM logits
-> final text

required ablations:
  core_off
  state_reset
  z_L_zero
  z_H_zero
  refinement_off
```

How Token Superposition Training can be used later:

```text
qtrm_native_tst_pretraining_efficiency_gate

Use after architecture promotion, not before:
  train native backbone/core with TST-style packed multi-token targets
  keep next-token CE as reference
  require language non-regression and recursive-depth non-regression
```

How Fast-Slow Training applies now:

```text
qtrm_native_fast_slow_latent_update_gate

z_L:
  fast working state, updated every L cycle

z_H:
  slow abstract/control state, updated every H step

candidate losses:
  z_L fast-state counterfactual loss
  z_H slow-state retention loss
  fast/slow disentanglement loss
  replay/KL to prevent language/backbone forgetting
```

Decision tree:

```text
If official-schedule split-mixer passes:
  run seed stability
  run language non-regression
  only then test Mercury/TST extensions

If official-schedule split-mixer fails:
  do not continue mixer shopping
  implement Fast-Slow-style latent update pressure as the next targeted repair
```

## Runner Result 2026-05-14T22:14:54

```text
gate: qtrm_native_fast_slow_latent_update_l4_repair
target_level: L4 Fast-Slow latent update repair
profile: smoke
decision: smoke_passed_fast_slow_latent_update_l4_repair
accepted: False
next_action: inspect whether z_L-zero, z_H-zero, exact accuracy, or language retention is the limiting factor before changing mixers again
```

Decisive metrics:

```json
{
  "last_loss": 4.714169979095459,
  "eval_metrics.think0.generation_exact": 0.0,
  "eval_metrics.state_reset.generation_exact": 0.0,
  "eval_metrics.op_zero.generation_exact": 0.0,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 3,
  "backend_summary.official_mamba3_mixers": 3,
  "backend_summary.torch_delta_mixers": 6,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": true,
  "decisive_metrics.full_generation_exact": 0.0,
  "decisive_metrics.think0_generation_exact": 0.0,
  "decisive_metrics.state_reset_generation_exact": 0.0,
  "decisive_metrics.op_zero_generation_exact": 0.0,
  "decisive_metrics.full_minus_think0": 0.0,
  "decisive_metrics.full_minus_worst_ablation": 0.0,
  "decisive_metrics.min_family_generation_exact": 0.0
}
```

Report: `local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_smoke/report.json`

## Runner Result 2026-05-14T21:59:55

```text
gate: qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
target_level: L4 nested official-schedule split-mixer 3:1 versus single baseline
profile: standard
decision: rejected
accepted: False
next_action: do not replace the accepted nested MHA repair; inspect whether the official H=3/L=6 Mamba3/GatedDelta split-mixer failed exact accuracy, z_L/z_H causality, or both before any further architecture shopping
```

Decisive metrics:

```json
{
  "last_loss": 0.5775094032287598,
  "eval_metrics.think0.generation_exact": 0.03125,
  "eval_metrics.state_reset.generation_exact": 0.0546875,
  "eval_metrics.op_zero.generation_exact": 0.01953125,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 3,
  "backend_summary.official_mamba3_mixers": 3,
  "backend_summary.torch_delta_mixers": 6,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": true,
  "decisive_metrics.full_generation_exact": 0.23046875,
  "decisive_metrics.think0_generation_exact": 0.03125,
  "decisive_metrics.state_reset_generation_exact": 0.0546875,
  "decisive_metrics.op_zero_generation_exact": 0.01953125,
  "decisive_metrics.full_minus_think0": 0.19921875,
  "decisive_metrics.full_minus_worst_ablation": 0.17578125,
  "decisive_metrics.min_family_generation_exact": 0.23046875
}
```

Report: `local_eval/research_gate_runner/qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare_standard/report.json`

## Runner Result 2026-05-14T22:27:09

```text
gate: qtrm_native_fast_slow_latent_update_l4_repair
target_level: L4 Fast-Slow latent update repair
profile: standard
decision: rejected
accepted: False
next_action: inspect whether z_L-zero, z_H-zero, exact accuracy, or language retention is the limiting factor before changing mixers again
```

Decisive metrics:

```json
{
  "last_loss": 0.5424272418022156,
  "eval_metrics.think0.generation_exact": 0.03125,
  "eval_metrics.state_reset.generation_exact": 0.0546875,
  "eval_metrics.op_zero.generation_exact": 0.0234375,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 3,
  "backend_summary.official_mamba3_mixers": 3,
  "backend_summary.torch_delta_mixers": 6,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": true,
  "decisive_metrics.full_generation_exact": 0.2421875,
  "decisive_metrics.think0_generation_exact": 0.03125,
  "decisive_metrics.state_reset_generation_exact": 0.0546875,
  "decisive_metrics.op_zero_generation_exact": 0.0234375,
  "decisive_metrics.full_minus_think0": 0.2109375,
  "decisive_metrics.full_minus_worst_ablation": 0.1875,
  "decisive_metrics.min_family_generation_exact": 0.2421875
}
```

Report: `local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_standard/report.json`

## L5 Multi-Family Repaired Seed Stability 2026-05-14

```text
summary:
  local_eval/qtrm_native_l5_multifamily_repaired_seed_stability_20260514_summary.json

decision:
  accepted_l5_multifamily_repaired_seed_stability

pass_rate:
  3 / 3

minimum metrics:
  min_full_generation_exact: 0.6067708333333334
  min_family_generation_exact: 0.4140625
  min_full_minus_worst_ablation: 0.5716145833333334
```

Seed reports:

```text
337:
  source: local_eval/qtrm_native_l5_multifamily_seed_sweep/seed_337/report.json
  decision: accepted_l5_multifamily
  full_generation_exact: 0.6067708333333334
  min_family_generation_exact: 0.4140625

338:
  source: local_eval/qtrm_native_l5_multifamily_seed338_single_modchain_weak_repair_s1500_20260514/report.json
  decision: accepted_l5_multifamily_single_modchain_weak_seed338_repair
  full_generation_exact: 0.74609375
  min_family_generation_exact: 0.60546875

339:
  source: local_eval/qtrm_native_l5_multifamily_seed339_single_modchain_weak_repair_s1500_20260514/report.json
  decision: accepted_l5_multifamily_single_modchain_weak_seed339_repair
  full_generation_exact: 0.6822916666666666
  min_family_generation_exact: 0.5078125
```

Architecture interpretation:

```text
Promote:
  QTRM-native single recurrent MHA/ETD path as the current stable L5
  multi-family baseline.

Do not promote yet:
  nested dual-z as mandatory L5 baseline.
  Fast-Slow latent loss as fixed training loss.
  Mamba3/GatedDelta split-mixer 3:1.

Reason:
  nested dual-z is locally useful on L4 but broad multi-family z_L causality is
  still unstable. The stable L5 result came from low-LR weak-family repair on
  the single recurrent native core.
```

## L6 Len6 Transfer From Repaired L5 2026-05-14

Question:

```text
Does the repaired L5 multi-family single recurrent MHA/ETD baseline transfer
to longer len6 programs without changing the canonical QTRM-native path?
```

Reports:

```text
zero-shot:
  local_eval/qtrm_native_l6_len6_transfer_from_l5_repair_seed339_init_20260514/report.json

focused fine-tune:
  local_eval/qtrm_native_l6_len6_from_l5_repair_focused_s2500_20260514/report.json

answer-space diagnostic:
  local_eval/qtrm_native_l6_len6_focused_s2500_answer_space_eval_20260514/report.json
```

Results:

```text
zero-shot:
  decision: rejected
  full_generation_exact: 0.0078125
  min_family_generation_exact: 0.00390625
  full_minus_worst_ablation: 0.00390625

focused final restored checkpoint:
  decision: rejected
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.05859375
  full_minus_worst_ablation: 0.05989583333333333

focused periodic trend:
  step0 full: 0.0078125, min_family: 0.00390625
  step1500 full: 0.0859375, min_family: 0.05859375
  step2500 full: 0.21354166666666666, min_family: 0.046875

answer-space diagnostic at restored checkpoint:
  greedy full_generation_exact: 0.0859375
  answer_space_argmax_exact: 0.0794270858168602
  answer_space_gold_mean_rank: 10.708333015441895
  answer_space_gold_top3: 0.2161458283662796
  answer_space_gold_top5: 0.3515625
```

Interpretation:

```text
The L6 failure is not primarily answer rendering. If it were mostly a greedy
renderer issue, answer_space_argmax would be much higher than greedy exact.
Instead, argmax is slightly lower than greedy exact and the gold answer mean
rank is still poor. The model is learning during focused fine-tuning, but
hard-family balance remains weak and longer ordered recurrent transition is
not solved.
```

Promotion decision:

```text
Do not freeze nested dual-z or Fast-Slow as canonical.

Current stable baseline:
  QTRM-native single recurrent MHA/ETD, repaired through L5 multi-family seed
  stability.

Current bottleneck:
  L6 length-scaling / ordered recurrent transition under hard-family balance.

Next experiment class:
  length-aware training or transition-state improvement on the same canonical
  token -> native core -> LM logits path. Do not switch back to donor,
  MemoryOS, sidecar solvers, or architecture shopping to claim progress.
```

Active-length batch-cycle repair:

```text
report:
  local_eval/qtrm_native_l6_len6_from_l5_repair_active_batch_cycle_s2500_20260514/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.052083333333333336
  min_family_generation_exact: 0.046875
  full_minus_think0: 0.03515625
  full_minus_worst_ablation: 0.01953125

interpretation:
  Mixing active lengths 1..6 inside each batch is not enough. It improves over
  zero-shot transfer but is worse than the prior focused len6 fine-tune. The
  L6 bottleneck is therefore not solved by a simple exposure schedule.
```

State/readout diagnostic:

```text
report:
  local_eval/qtrm_native_l6_len6_focused_state_probe_eval_20260515/report.json

metrics:
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.05859375
  depth_sweep_exact:
    depth0: 0.010416666666666666
    depth1: 0.016927083333333332
    depth2: 0.03125
    depth3: 0.016927083333333332
    depth4: 0.0859375
  z_h_variance_by_depth:
    0.24019193649291992
    0.3616328239440918
    0.5645048022270203
    0.8666419386863708
  core_step_probe_exact: 0.09765625
  core_step_probe_by_family:
    modchain: 0.08984375
    revchain: 0.0498046875
    checksum: 0.1533203125

interpretation:
  The state trajectory is active, not collapsed. But the depth-local latent
  state is weakly readable as the causal intermediate calculation. This points
  at transition-state representation/readout coupling, not output formatting.
```

Latent-refinement repair:

```text
report:
  local_eval/qtrm_native_l6_len6_from_l5_repair_latent_refine_s2500_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.08854166666666667
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.06640625
  core_step_probe_exact: 0.0960286483168602

periodic:
  step1500 full: 0.08854166666666667, min_family: 0.0546875
  step2500 full: 0.23046875, min_family: 0.05078125

interpretation:
  The auxiliary decode-ready state objective does not make the recurrent state
  more linearly readable and does not raise the weakest family. It confirms
  that the next L6 work should target hard-family recurrent transition balance
  or a genuine state-carrier/readout change, not another output-only loss.
```

## Block/Topology Naming Correction 2026-05-15

Problem:

```text
The phrase `trm_official_prenorm backbone` is ambiguous if it is read as a
replacement for the dual/nested QTRM topology. In this codebase, backbone is a
local stage/block choice; think_structure is the macro recursive topology.
```

Correct interpretation:

```text
backbone / encode_backbone / think_backbone / decode_backbone:
  which local block to run inside each stage

think_structure:
  single, dual-z, nested dual-z, official schedule, coupled variants, etc.

canonical QTRM-native candidate:
  must keep a mandatory recursive think_structure, preferably the accepted
  dual/nested reference when testing dual/nested claims.
```

Rejected diagnostic:

```text
L6 all-stage official-prenorm no-abs smoke:
  report: local_eval/qtrm_native_l6_dual_nested_trm_official_prenorm_noabs_smoke_20260514_235644/report.json
  think_structure: trm_dual_z_nested_reversed_mha_etd
  encode/think/decode_backbone: trm_official_prenorm
  position_embedding_mode: none
  H/L: train_think_steps=3, trm_l_cycles=6
  full_generation_exact: 0.0
  full_minus_worst_ablation: 0.0

L4 think-only official-prenorm no-abs triage:
  report: local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_noabs_triage_20260514_235922/report.json
  resume: local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/last.pt
  think_structure: trm_dual_z_nested_reversed_mha_etd
  think_backbone: trm_official_prenorm
  position_embedding_mode: none
  full_generation_exact: 0.041666666666666664
  full_minus_worst_ablation: -0.03125000000000001

L4 think-only official-prenorm learned-position control:
  report: local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_learnedpos_triage_20260515_000034/report.json
  same as above but position_embedding_mode: learned
  full_generation_exact: 0.052083333333333336
  full_minus_worst_ablation: -0.010416666666666664
  z_l_zero_generation_exact: 0.052083333333333336
```

Decision:

```text
Do not replace the accepted nested MHA/ETD reference with
trm_official_prenorm. The learned-position control shows the failure is not
only caused by disabling learned absolute positions. The official-prenorm
think block is not yet causally useful inside the dual/nested path because
z_l_zero matches full accuracy in the control.
```

Next useful hypothesis:

```text
Target the state carrier/readout coupling directly. A candidate must beat the
accepted nested MHA/ETD L4 gate or the L5/L6 family floor under ablations
before it can be called a dual/nested improvement.
```

## Nested MHA/ETD Joint Readout 2026-05-15

Claim under test:

```text
The L6 bottleneck is partly caused by weak readout coupling from the recurrent
state trajectory. The accepted nested MHA/ETD topology should be kept, but the
final readout should see encoded, z_L, and z_H together.
```

Implemented candidates:

```text
replacement:
  trm_dual_z_nested_reversed_mha_etd_joint_readout
  readout = LN(z_H + W[encoded, z_L, z_H])

residual:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout
  base = z_H + tanh(alpha) * (z_L - z_H)
  readout = base + tanh(beta) * LN(W[encoded, z_L, z_H])
```

Why residual is the promoted candidate:

```text
The replacement readout exposes z_L/z_H more strongly but reduces exactness.
The residual readout preserves the accepted base path and adds a trainable
state bridge initialized as a no-op when loaded from the accepted nested
checkpoint.
```

Results:

```text
replacement triage:
  report: local_eval/qtrm_native_l4_nested_mha_joint_readout_triage_20260515_000707/report.json
  decision: rejected
  full_generation_exact: 0.5729166666666666
  full_minus_worst_ablation: 0.35416666666666663

replacement full-train continuation:
  report: local_eval/qtrm_native_l4_nested_mha_joint_readout_standard_cont_20260515_000749/report.json
  decision: rejected
  full_generation_exact: 0.1328125
  full_minus_worst_ablation: 0.07421875

replacement added-params continuation:
  report: local_eval/qtrm_native_l4_nested_mha_joint_readout_addedparams_cont_20260515_000946/report.json
  decision: rejected
  full_generation_exact: 0.55078125
  full_minus_worst_ablation: 0.46484375

residual from accepted nested checkpoint:
  report: local_eval/qtrm_native_l4_nested_mha_residual_joint_readout_from_accepted_20260515_001357/report.json
  decision: accepted_nested_mha_residual_joint_readout_l4
  full_generation_exact: 0.671875
  full_minus_think0: 0.640625
  full_minus_worst_ablation: 0.14453125
```

Decision:

```text
Promote residual joint-readout only to the next diagnostic stage. It matches
the accepted nested MHA/ETD L4 level but does not yet prove L5/L6 improvement.
The next valid test is L5 multi-family or L6 transfer from the residual
checkpoint, with base-path preservation and family-floor acceptance.
```

### L5 Residual Joint-Readout Transfer 2026-05-15

Tokenizer/vocab transfer fix:

```text
Problem:
  Direct L4 -> L5 resume changes the tokenizer char set because revchain and
  checksum introduce new letters. Prefix-copying token rows is unsafe because
  sorted token rows shift when new chars appear before old chars.

Fix:
  When --resume-allow-missing is used, token_embed.weight and lm_head.weight
  are now remapped by token string. Positional embeddings still use the
  configured position resize strategy.

Verification:
  test_flexible_load_remaps_vocab_rows_by_token
```

Direct L4 residual checkpoint to L5:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_multifamily_seed337_vocabremap_20260515_0028/report.json

decision:
  rejected

decisive:
  full_generation_exact: 0.0859375
  full_minus_think0: 0.07421875
  full_minus_worst_ablation: 0.0625
  min_family_generation_exact: 0.023529411764705882

meaning:
  The remap works, but the L4 single-family nested residual checkpoint does not
  directly become an L5 multi-family model.
```

Attach nested residual path to the accepted L5 single recurrent checkpoint:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_from_l5_single_missingonly_seed339_20260515_0035/report.json

resume:
  local_eval/qtrm_native_l5_multifamily_seed339_single_modchain_weak_repair_s1500_20260514/last.pt

decision:
  rejected by ablation_drop_below_threshold

best periodic:
  generation_exact: 0.734375
  min_family_generation_exact: 0.5581395348837209

decisive:
  full_generation_exact: 0.68359375
  full_minus_think0: 0.65625
  full_minus_worst_ablation: -0.01171875
  min_family_generation_exact: 0.5232558139534884
  z_l_zero_generation_exact: 0.6953125
  z_h_zero_generation_exact: 0.0
```

Decision:

```text
Do not promote residual joint-readout to L5. It can preserve or recover L5
accuracy when attached to an accepted single recurrent checkpoint, but z_L is
not causally required. The active L5 dual/nested bottleneck is now z_L
state-carrier causality, not vocabulary transfer, readout format, or broad
text generation.
```

z_L causal repair:

```text
first repair:
  report: local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair_seed339_20260515_0048/report.json
  decision: rejected
  full_generation_exact: 0.68359375
  full_minus_worst_ablation: 0.05078125
  z_l_zero_generation_exact: 0.6328125

second repair:
  report: local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair2_seed339_20260515_0056/report.json
  decision: accepted_l5_residual_joint_readout_zl_causal_repair2
  full_generation_exact: 0.703125
  think0_generation_exact: 0.02734375
  full_minus_think0: 0.67578125
  full_minus_worst_ablation: 0.296875
  min_family_generation_exact: 0.5581395348837209
  z_l_zero_generation_exact: 0.40625
  z_h_zero_generation_exact: 0.0
```

Updated decision:

```text
The current accepted L5 dual/nested candidate is:

  accepted L5 single recurrent checkpoint
  -> nested residual joint-readout attachment
  -> TRM-only z_L causal repair with strong z_l_counterfactual loss

This does not make Fast-Slow mandatory. Fast-Slow remains optional until it
passes the same L5/L6 causal gates. Nested residual joint-readout is now the
candidate to seed-test and L6-transfer, because it has crossed the L5
multi-family exact/family/depth/ablation gate with z_L causality restored.
```

### Seed Stability And L6 Check 2026-05-15

Seed338 reproduction:

```text
direct attach + strong z_L repair:
  report: local_eval/qtrm_native_l5_residual_joint_readout_seed338_direct_zl_repair_20260515_0110/report.json
  decision: rejected
  full_generation_exact: 0.74609375
  min_family_generation_exact: 0.5882352941176471
  z_l_zero_generation_exact: 0.75
  full_minus_worst_ablation: -0.00390625

second-stage repair:
  report: local_eval/qtrm_native_l5_residual_joint_readout_seed338_zl_repair2_20260515_0118/report.json
  decision: rejected
  full_generation_exact: 0.75
  min_family_generation_exact: 0.611764705882353
  z_l_zero_generation_exact: 0.734375
  full_minus_worst_ablation: 0.015625
```

Meaning:

```text
The seed338 path solves L5 exactness but not dual-state causality. The current
accepted nested residual result is therefore seed339-local, not seed-stable.
```

L6 transfer from accepted seed339 z_L-causal checkpoint:

```text
zero-shot:
  report: local_eval/qtrm_native_l6_len6_from_l5_zl_causal_seed339_init_20260515_0126/report.json
  decision: rejected
  full_generation_exact: 0.0234375
  min_family_generation_exact: 0.0
  full_minus_worst_ablation: -0.0078125

fine-tune:
  report: local_eval/qtrm_native_l6_len6_from_l5_zl_causal_finetune_s2500_20260515_0130/report.json
  decision: rejected
  best_periodic_generation_exact: 0.08854166666666667
  full_generation_exact: 0.0703125
  min_family_generation_exact: 0.03125
  full_minus_worst_ablation: 0.04427083333333333

prior single L6 hard-family balance:
  full_generation_exact: 0.12239583333333333
```

Decision:

```text
Nested residual z_L-causal is not the L6 answer yet. Keep it as an L5
seed339-local candidate. The next architecture/training work should target
seed-stable z_L causality and len6 ordered transition generalization, not
Fast-Slow hard-freezing or another mixer swap.
```

### L5 z_L Causality Stabilized With Core-Step Codec 2026-05-15

Core-step codec repair:

```text
idea:
  Add auxiliary supervision that reads intermediate causal values from z_L
  state traces. This does not replace the final answer path; the accepted
  metric is still normal autoregressive LM generation with z_L-zero ablation.

why:
  seed338 showed that z_L counterfactual loss alone can keep high exactness but
  fail causal ablation. The codec gives z_L a positive intermediate-state job
  instead of only making the ablated path worse.
```

Seed338:

```text
report:
  local_eval/qtrm_native_l5_seed338_zl_codec_repair_final_s1200_20260515_0145/report.json

decision:
  accepted_l5_seed338_zl_codec_repair

decisive:
  full_generation_exact: 0.75390625
  full_minus_think0: 0.7421875
  full_minus_worst_ablation: 0.28125
  min_family_generation_exact: 0.6235294117647059
  z_l_zero_generation_exact: 0.47265625
```

Seed337:

```text
report:
  local_eval/qtrm_native_l5_seed337_zl_codec_direct_repair_s1200_20260515_0152/report.json

decision:
  accepted_l5_seed337_zl_codec_direct_repair

decisive:
  full_generation_exact: 0.703125
  full_minus_think0: 0.6875
  full_minus_worst_ablation: 0.23828125
  min_family_generation_exact: 0.5465116279069767
  z_l_zero_generation_exact: 0.46484375
```

Seed-stable L5 candidate:

```text
seed337:
  accepted_l5_seed337_zl_codec_direct_repair
seed338:
  accepted_l5_seed338_zl_codec_repair
seed339:
  accepted_l5_residual_joint_readout_zl_causal_repair2
```

Decision:

```text
Promote nested residual joint-readout with z_L causal repair to the active L5
dual/nested candidate. Qualification: it is still not L6-capable. The next
gate is len6 ordered transition generalization using the stabilized z_L codec
recipe, not another readout or mixer swap.
```

### L6 Depth/Length Alignment 2026-05-15

Finding:

```text
The earlier len6 transfer failures were partly confounded by a depth/length
mismatch: program_len=6 while train_think_steps/eval_think_steps stayed at the
script default 4. A len6 ordered program should first be tested with recurrent
depth aligned to the active program length before changing architecture.
```

Aligned L6 checks:

```text
nested residual z_L-codec depth6:
  report: local_eval/qtrm_native_l6_len6_seed338_zl_codec_depth6_s3000_20260515_0215/report.json
  decision: rejected
  full_generation_exact: 0.08072916666666667
  min_family_generation_exact: 0.0390625

nested residual depth6 continuation:
  report: local_eval/qtrm_native_l6_len6_seed338_zl_codec_depth6_cont_s3000_20260515_0228/report.json
  decision: rejected
  full_generation_exact: 0.07552083333333333
  min_family_generation_exact: 0.0546875

single recurrent depth6:
  report: local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/report.json
  decision: accepted_l6_len6_single_seed338_depth6
  full_generation_exact: 0.3671875
  think0_generation_exact: 0.0
  full_minus_worst_ablation: 0.34375
  min_family_generation_exact: 0.0703125
```

Decision:

```text
The current L6 winner is not the dual/nested branch. It is the single
recurrent MHA/ETD scaffold with train/eval think_steps=program_len=6. Keep
dual/nested as the active L5 direction, but do not force it into L6 until it
beats this aligned-depth scaffold under the same causal ablations.
```

Continuation check:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_cont_s3000_lr1e5_20260515_011502/report.json

result:
  accepted by the diagnostic L6 thresholds, but not improved.

base:
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125

continued:
  full_generation_exact: 0.3541666666666667
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.3229166666666667
```

Updated next step:

```text
Do not keep plain-continuing this checkpoint. The weak families are modchain
and revchain, while checksum is already high. The next aligned L6 work should
try a family-floor repair or seed-stability run without changing the
token->core->logits path.
```

Weak-family repair check:

```text
first repair:
  report: local_eval/qtrm_native_l6_len6_single_seed338_family_floor_repair_s2500_20260515_011837/report.json
  decision: rejected
  full_generation_exact: 0.3567708333333333
  min_family_generation_exact: 0.078125
  reject_reason: family_exact_below_threshold

second modchain-focused repair:
  report: local_eval/qtrm_native_l6_len6_single_seed338_modchain_floor_repair_s1500_20260515_012038/report.json
  decision: rejected
  result: no improvement beyond the first repair; restore-best selected the
    initial checkpoint.
```

Interpretation:

```text
Family-DRO pressure is useful but insufficient. It moves the L6 family floor
from 0.0703125 to 0.078125, but the thresholded improvement gate remains
rejected. The next roadmap item is seed-stability of the aligned depth6 recipe
or a new transition objective, not more architecture shopping.
```

Seed-stability check:

```text
seed338:
  report: local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/report.json
  decision: accepted_l6_len6_single_seed338_depth6
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.34375

seed339:
  report: local_eval/qtrm_native_l6_len6_single_seed339_depth6_s3000_20260515_012244/report.json
  decision: accepted_l6_len6_single_seed339_depth6
  full_generation_exact: 0.15364583333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.1328125
```

Roadmap implication:

```text
Aligned depth6 is now a reproducible L6 scaffold across two retained seeds,
but the variance is large. Treat it as the current floor for future L6
transition/objective work. It does not settle the dual/nested question; it
only proves that any dual/nested L6 candidate must beat this aligned single
baseline under the same depth and ablation settings.
```

Core-step codec check:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_corecodec_h_s2500_20260515_012658/report.json

decision:
  rejected

result:
  restore-best selected the initial checkpoint.
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.34375
```

Roadmap implication:

```text
Auxiliary state readability alone does not repair L6. The next candidate
should make the canonical LM path answer correctly across active program
prefixes, for example with a bounded active-length replay objective, or change
the transition structure itself.
```

Active-length replay check:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_active_replay_s2000_20260515_012955/report.json

decision:
  rejected

recipe:
  resume:
    local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt
  active_len_replay_loss_weight: 0.02
  active_len_replay_min/max: 1/6
  active_len_replay_max_cases: 16
  active_len_replay_every: 4
  train/eval_think_steps: 6

result:
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.34375
  reject_reason: family_exact_below_threshold
  restored best checkpoint: initial checkpoint
```

Roadmap implication:

```text
Prefix replay does not move the aligned L6 scaffold. The next experiment should
test whether the canonical LM path needs an explicit transition carrier rather
than more replay or auxiliary state-readability pressure. Start with the
least-invasive prompt_state_anchor smoke; if it fails, move to an internal
scratch/state token in the model path.
```

Transition-carrier result:

```text
visible prompt_state_anchor:
  report:
    local_eval/qtrm_native_l6_len6_single_seed338_prompt_anchor_s2500_20260515_013530/report.json
  decision: rejected
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.30729166666666663

internal single_core_carrier:
  implementation:
    scripts/335_train_qtrm_native_etd_probe.py
  report:
    local_eval/qtrm_native_l6_len6_single_seed338_single_core_carrier_s2500_20260515_013530/report.json
  decision: accepted_l6_len6_single_seed338_single_core_carrier
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.09375
  full_minus_worst_ablation: 0.3307291666666667
  state_reset_generation_exact: 0.0026041666666666665
  op_zero_generation_exact: 0.036458333333333336
  family exact:
    checksum: 0.9140625
    modchain: 0.09375
    revchain: 0.09375
```

Roadmap implication:

```text
The first accepted L6 repair is structural and internal: `single_core_carrier`
keeps the same prompt-token -> recurrent-core -> decoder -> LM-logit path, but
adds a causal GRU carrier inside each single recurrent step. It beats the
visible prompt anchor and raises both hard families above the current floor
threshold. Use this as the active L6 scaffold. Next required checks:

1. seed-stability from seed339 or another retained L5/single checkpoint;
2. optional length-8 smoke with train/eval depth aligned to length;
3. only after stability, decide whether dual/nested should absorb the same
   carrier idea.
```

Seed-stability follow-up:

```text
seed339 standard carrier:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_s2500_20260515_014403/report.json
  decision: rejected
  full_generation_exact: 0.19010416666666666
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.1640625

seed339 near-off carrier gate:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_gateoff_s2500_20260515_014659/report.json
  decision: rejected
  reject_reason: full_exact_below_threshold
  full_generation_exact: 0.12760416666666666
  min_family_generation_exact: 0.0859375
  full_minus_worst_ablation: 0.10156249999999999
```

Updated roadmap implication:

```text
The carrier is an accepted seed338 scaffold, not a seed-stable architecture.
The next L6 work should target carrier optimization stability. The useful
signal is split: standard init keeps higher full exact but weak family floor;
near-off gate improves family floor but loses full exact. A proper next run
should combine these pressures explicitly, for example by using the standard
carrier init with a stronger validation-gated family-floor objective, rather
than changing the macro topology again.
```

Family-DRO 0.15 check:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_famdro015_s2500_20260515_014659/report.json

decision:
  rejected

decisive:
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.15364583333333334
  reject_reason: family_exact_below_threshold
```

Roadmap update:

```text
Do not keep sweeping family-DRO on seed339. The next carrier-stability work
must add a better supervision/curriculum signal for the carrier itself, or
test a third seed to estimate whether seed339 is an outlier.
```

Third-seed carrier check:

```text
seed337 baseline:
  report:
    local_eval/qtrm_native_l6_len6_single_seed337_depth6_s3000_20260515_015547/report.json
  decision: rejected
  full_generation_exact: 0.23697916666666666
  min_family_generation_exact: 0.03125
  full_minus_worst_ablation: 0.19010416666666666

seed337 single_core_carrier:
  report:
    local_eval/qtrm_native_l6_len6_single_seed337_single_core_carrier_s2500_20260515_015547/report.json
  decision: rejected
  full_generation_exact: 0.1640625
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.140625
```

Roadmap implication:

```text
The carrier is no longer just "seed339 is an outlier"; it is accepted only on
seed338 and rejected on seed337/seed339. It still points to the right failure
class: L6 needs a causal transition-state carrier, but the current GRU carrier
recipe is not stable enough. Next work should be carrier-specific curriculum or
state supervision, not renaming the backbone or promoting single-core as final.

`trm_official_prenorm` means a pre-norm official-TRM-style block. It can be
used inside single, dual-z, or nested schedules. It does not discard dual/nested
TRM. The correct policy is to keep dual/nested as the macro reasoning schedule,
then test which backbone block makes the z_L/z_H transition trainable under the
same L6 gate.
```

Carrier supervision check:

```text
seed339 single_core_carrier + h-mean core-step codec:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hmean_w008_s2500_20260515_020418/report.json
  decision: rejected
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.16666666666666669
  by_family:
    checksum: 0.4609375
    modchain: 0.0546875
    revchain: 0.0625
```

Roadmap implication:

```text
Carrier-specific supervision is still the right bottleneck class, but
mean-pooled h-state supervision is too weak or too indirect. The next minimal
variant should supervise the last prompt state, because that is closer to the
canonical autoregressive answer boundary. If last-state supervision also fails,
stop codec-weight variants and implement a direct transition-consistency loss
on the recurrent state itself.
```

Last-state codec check:

```text
seed339 single_core_carrier + h-last core-step codec:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hlast_w008_s2500_20260515_020735/report.json
  decision: rejected
  full_generation_exact: 0.2109375
  min_family_generation_exact: 0.046875
  full_minus_worst_ablation: 0.1875
  by_family:
    checksum: 0.5078125
    modchain: 0.046875
    revchain: 0.078125
```

Roadmap implication:

```text
Stop carrier codec variants. They preserve causal ablation drops and improve
some aggregate scores, but they do not lift the weakest family. The next
minimal state-level candidate is prefix/full recurrent state consistency,
using the existing `prefix_state_alignment_loss`, before writing a new loss.
```

Prefix/full state alignment check:

```text
seed339 single_core_carrier + prefix_state_alignment:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_align_w002_s2500_20260515_021141/report.json
  decision: rejected
  full_generation_exact: 0.20572916666666666
  min_family_generation_exact: 0.046875
  full_minus_worst_ablation: 0.18229166666666666
  by_family:
    checksum: 0.5
    modchain: 0.046875
    revchain: 0.0703125
```

Roadmap implication:

```text
MSE prefix/full state alignment is not enough. Run one contrastive state
alignment check because it preserves case discrimination; if that also fails,
the next repair should change the carrier transition objective itself rather
than adding more auxiliary readers.
```

Contrastive state alignment check:

```text
seed339 single_core_carrier + prefix_state_contrastive:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_contrast_w002_s2500_20260515_021514/report.json
  decision: rejected
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.16666666666666669
  by_family:
    checksum: 0.453125
    modchain: 0.0546875
    revchain: 0.0703125
```

Roadmap implication:

```text
Stop auxiliary state alignment for this repair line. The failure pattern is
consistent: state/readability losses preserve causality but do not lift the
hard-family floor. The next minimal curriculum is carrier warmup: train only
new `single_carrier_*` parameters against the frozen L6 single baseline, then
fine-tune the full model. This tests whether seed instability comes from
introducing a random carrier into a trained recurrent LM all at once.
```

Carrier warmup and gate control:

```text
warmup only:
  report: local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_warmup_missing_s1500_20260515_021914/report.json
  decision: rejected
  full_generation_exact: 0.17708333333333334
  min_family_generation_exact: 0.0546875

warmup then full fine-tune:
  report: local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_warmup_then_full_s1500_20260515_022043/report.json
  decision: rejected
  full_generation_exact: 0.17708333333333334
  min_family_generation_exact: 0.0546875

carrier_gate_init=-4.0:
  report: local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_gateinit_m4_s2500_20260515_022504/report.json
  decision: rejected
  full_generation_exact: 0.12760416666666666
  min_family_generation_exact: 0.0859375
  family exact:
    checksum: 0.2109375
    modchain: 0.0859375
    revchain: 0.0859375
```

Roadmap implication:

```text
Carrier warmup does not solve seed339. Lowering the carrier gate init shows the
right tradeoff but not a pass: it lifts the family floor above 0.08 while
dropping full exact below 0.15. This is useful evidence because it localizes the
instability to carrier strength/timing, not to the existence of a causal
carrier path.
```

Checkpoint-soup diagnostic:

```text
source checkpoints:
  base:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_s2500_20260515_014403/last.pt
  candidate:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_gateinit_m4_s2500_20260515_022504/last.pt

alpha0.1:
  report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p1_eval_20260515/report.json
  full_generation_exact: 0.12760416666666666
  min_family_generation_exact: 0.0546875
  decision: rejected

alpha0.2:
  report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p2_eval_20260515/report.json
  full_generation_exact: 0.09895833333333333
  min_family_generation_exact: 0.0625
  decision: rejected

alpha0.3:
  report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p3_eval_20260515/report.json
  full_generation_exact: 0.08072916666666667
  min_family_generation_exact: 0.0625
  decision: rejected

alpha0.4:
  report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p4_eval_20260515/report.json
  full_generation_exact: 0.0703125
  min_family_generation_exact: 0.0546875
  decision: rejected

alpha0.5:
  report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha05_eval_20260515/report.json
  full_generation_exact: 0.07291666666666667
  min_family_generation_exact: 0.0625
  decision: rejected
```

Roadmap implication:

```text
Do not use linear checkpoint soup as the seed339 carrier repair. The two
solutions are not linearly composable under this metric. The next canonical
architecture move should return to the dual/nested macro schedule, but with the
lesson from carrier diagnostics: recurrent transition state must enter the
core before LM rendering, and its strength must be learned without erasing the
baseline state.

Fast-Slow and nested remain candidate macro/training principles, not fixed
defaults. They should be fixed only after the same L6 strict gate plus
ablation suite passes. Until then:
  canonical scaffold: aligned single recurrent depth6;
  accepted structural hint: seed338 single_core_carrier;
  rejected repairs: codec, prefix alignment, warmup, gate init, checkpoint soup.
```

Order-invariant L6 gate update:

```text
Problem:
  The previous eval case builder could change per-family held-out examples
  when only the order of eval_task_families changed.

Fix:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py now supports
  --eval-family-order-invariant.

Canonical use:
  All future L6/L7 family-floor claims should use:
    --eval-task-families modchain,revchain,checksum
    --eval-family-order-invariant

Reason:
  Near the family-floor threshold, evaluation order must not become an
  accidental architecture knob.
```

Current L6 dual/nested status under the order-invariant gate:

```text
best causal dual/nested scaffold:
  local_eval/qtrm_native_l6_seed338_order_invariant_revchain_fullperiodic_repair_s800_20260515/report.json

decision:
  rejected only by family floor

metrics:
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.3125
  z_l_zero_generation_exact: 0.0
  z_h_zero_generation_exact: 0.0

family:
  checksum: 0.84375
  modchain: 0.0859375
  revchain: 0.078125
```

Roadmap implication:

```text
This is not an accepted L6 result, but it is a useful bottleneck:
  the causal recurrent path works;
  fixed depth 6 is necessary;
  over-running to depth 7/8 collapses;
  revchain reverse-order binding is the limiting family.

Next architecture step:
  stop generic CE/family-DRO/answer-rank tuning;
  add a reverse-order transition-state repair that is still internal to the
  QTRM-native token -> recurrent core -> LM logits path.
```

L6 dual/nested core-carrier update:

```text
active candidate:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier

meaning:
  dual/nested macro topology is still active.
  The carrier is an internal recurrent-state repair, not a donor/RAG/renderer
  path.

terminology:
  `trm_official_prenorm backbone` names a local block style only. It must not
  be read as "single-core" or "dual/nested abandoned".

seed338 order-invariant eval:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_nested_core_carrier_identity_init_carrieroff_eval2_20260515/report.json
  decision:
    accepted_l6_nested_core_carrier_identity_init
  full_generation_exact:
    0.3411458333333333
  min_family_generation_exact:
    0.09375
  carrier_off_generation_exact:
    0.3359375
  full_minus_carrier_off:
    0.005208333333333315

promotion status:
  L6 seed338 scaffold accepted.
  Not yet seed-stable L6.
  Not yet language-preserved L7/L4 general-LM promotion.
```

Next roadmap gate:

```text
1. Re-run the same carrier candidate across seeds.
2. If only seed338 passes, train only the missing carrier/gate parameters with
   the same full/family/depth/ablation gate.
3. Preserve the accepted dual/nested residual joint readout path.
4. Reject any repair that improves the family floor by using a side renderer,
   hidden solver, donor shortcut, or MemoryOS/RAG path.
```

Seed-stability result:

```text
summary:
  local_eval/qtrm_native_l6_nested_core_carrier_seed_stability_summary_20260515.json

decision:
  rejected_seed_stability

pass_count:
  1 / 3

meaning:
  nested dual-z remains the active macro candidate, but it is not promoted as a
  fixed default. The in-core carrier is useful but seed-brittle.

fast-slow status:
  not fixed. The prior Fast-Slow standard gate was rejected, so Fast-Slow may
  be tested as a repair but must not become a default unless it passes the
  same strict L6/L7 gate.
```

Updated immediate roadmap:

```text
1. Keep QTRM-native and dual/nested macro path as the active research line.
2. Do not freeze carrier, nested, or Fast-Slow as canonical defaults yet.
3. Repair the seed brittleness by reducing carrier random-init dependence.
4. Require the same 3-seed order-invariant family-floor gate before promotion.
5. Only after seed-stable L6, run language non-regression.
```

Carrier brittleness repair result:

```text
summary:
  local_eval/qtrm_native_l6_seed337_carrier_brittleness_repair_summary_20260515.json

decision:
  rejected_deterministic_carrier_and_seed337_repairs

tested:
  deterministic carrier modes: encoded, state_mean, state_delta
  carrier-only learning
  carrier-only family-DRO
  low-LR transition+carrier tuning with retention

result:
  none crossed the 0.08 weak-family floor on seed337.
```

Roadmap adjustment:

```text
Do not spend more cycles on local carrier tuning. The next prerequisite is a
readout-vs-transition audit:

1. Run answer-space argmax/rank on weak seed337/339.
2. If the gold answer is ranked well but greedy text fails, repair the LM
   readout/generation path.
3. If the gold answer is not ranked well, repair recurrent transition learning
   before adding new carrier variants.
```

Answer-space audit result:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_carrier_answer_space_audit_20260515/report.json

answer_space_argmax_exact:
  0.3177083432674408

generation_exact:
  0.3203125

answer_space_gold_mean_rank:
  8.098958015441895

decision:
  readout-only repair is not the next best path. The weak-seed failure is
  recurrent transition/state accuracy.
```

State-trace depth repair closure:

```text
reports:
  local_eval/qtrm_native_l6_seed337_nested_state_trace_depth_repair_s900_20260515/report.json
  local_eval/qtrm_native_l6_seed337_nested_state_trace_depth_repair_strong_s1200_20260515/report.json

strong repair:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 450
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.0
  answer_space_argmax_exact: 0.34375
  answer_space_gold_mean_rank: 7.916666507720947

interpretation:
  depth-wise state supervision improves the readout rank only marginally and
  does not repair the modchain/revchain floor. The carrier path is not
  contributing under the strong run because carrier_off equals full.

roadmap consequence:
  Stop scalar state-trace/carrier tuning for this bottleneck. Keep
  QTRM-native dual/nested as the active macro direction, but do not promote it
  as canonical until a transition-binding repair passes the seed337/338/339
  family-floor and causal-ablation gate.

next allowed candidate class:
  recurrent transition/state-update redesign inside the mandatory dual/nested
  core. Examples: contrastive transition consistency, stricter z_L/z_H state
  binding, learned operation-conditioned transition routing, or a TRM-native
  depth curriculum that improves latent state accuracy before LM logits.
```

Prefix contrast transition-binding check:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_binding_prefix_contrast_s900_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3333333333333333
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: -0.002604166666666685
  answer_space_argmax_exact: 0.3411458432674408
  answer_space_gold_mean_rank: 7.640625

roadmap consequence:
  Contrastive prefix/full state alignment is not sufficient and may blur the
  distinction required by modchain/revchain. Do not keep sweeping its weight.
  The next bounded repair is operation/position transition-codec pressure. If
  that fails under the same gate, move from auxiliary objectives to an actual
  recurrent state-update redesign.
```

Transition-codec repair check:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_codec_repair_s700_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3125
  min_family_generation_exact: 0.0625
  full_minus_carrier_off: -0.010416666666666685
  answer_space_argmax_exact: 0.3177083432674408
  answer_space_gold_mean_rank: 7.919270992279053

roadmap consequence:
  State-codec pressure is also insufficient. This closes the bounded
  auxiliary-objective repair sequence for the seed337 weak-family floor.
  Continue only with an actual recurrent state-update redesign under the same
  QTRM-native hard lock.
```

Cross-exchange structural attempt:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange

report:
  local_eval/qtrm_native_l6_seed337_nested_core_carrier_cross_exchange_s800_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3385416666666667
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: 0.01302083333333337
  coupling_off_generation_exact: 0.3333333333333333
  answer_space_argmax_exact: 0.3463541567325592

roadmap consequence:
  Generic z_L/z_H delta exchange is not enough. It creates a small measurable
  dependency but fails the family floor and causal-drop threshold. The next
  architecture candidate must make operation-order binding explicit inside the
  recurrent update instead of only sharing generic latent deltas.
```

Step-conditioned recurrent update:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned

reports:
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_s1000_20260515/report.json
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_step_only_repair_s600_20260515/report.json

best result:
  decision: rejected
  full_generation_exact: 0.3619791666666667
  min_family_generation_exact: 0.0703125
  coupling_off_generation_exact: 0.359375
  modchain: 0.0703125
  revchain: 0.0859375

roadmap consequence:
  Step conditioning is directionally useful but not accepted. It improves the
  weak seed more than generic cross-exchange, but the coupling_off ablation
  shows the new path is not yet causally necessary. Next candidate should be a
  nested order-router that changes update order or route, not just an additive
  step delta.
```

Nested order-router and sequence-router checks:

```text
new think_structures:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router

reports:
  local_eval/qtrm_native_l6_seed337_nested_order_router_aux_forced_s600_20260515/report.json
  local_eval/qtrm_native_l6_seed337_nested_sequence_order_router_aux_forced_s600_20260515/report.json

order-router result:
  decision: rejected
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.036458333333333315
  route0_generation_exact: 0.2994791666666667
  route1_generation_exact: 0.041666666666666664
  modchain: 0.0625
  revchain: 0.0546875

sequence-router result:
  decision: rejected
  full_generation_exact: 0.09635416666666667
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: -0.23177083333333331
  route0_generation_exact: 0.328125
  route1_generation_exact: 0.041666666666666664
  modchain: 0.0546875
  revchain: 0.0625

roadmap consequence:
  The router implementation is now general enough to train/probe nested
  routers, but the route candidate is the failure. Route1 H->L->H is too weak:
  forcing it gives only 0.0417 overall. Sequence-level routing proves the issue
  is not per-token route noise; route locking makes the model worse because it
  routes whole cases through the weak transition.

  Do not keep sweeping order-router LR, route aux weights, or forced-route CE.
  The next L6 structural candidate must replace route1 with a better recurrent
  transition update that binds operation order/source position inside z_L/z_H
  before LM logits.
```

Order-bound route1 replacement:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router

report:
  local_eval/qtrm_native_l6_seed337_nested_order_bound_router_aux_forced_s600_20260515/report.json

result:
  decision: rejected
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.0390625
  coupling_off_generation_exact: 0.296875
  route0_generation_exact: 0.296875
  route1_generation_exact: 0.03125
  checksum: 0.875
  modchain: 0.078125
  revchain: 0.0546875

roadmap consequence:
  Re-attending to visible prompt tokens inside route1 is not enough. It remains
  QTRM-native and causal-path compliant, but it does not improve the weak
  revchain family or make route1 viable. The L6 path should now move away from
  H->L->H router variants and toward a transition-binding core or a reduced
  diagnostic gate that directly measures operation-order state updates.
```

Route1 capacity isolation:

```text
report:
  local_eval/qtrm_native_l6_seed337_order_bound_route1_revchain_capacity_s400_20260515/report.json

setup:
  task_families: revchain
  eval_task_families: revchain
  strong route1 router/forced-route losses
  no promotion claim

result:
  decision: rejected
  full_generation_exact: 0.04296875
  min_family_generation_exact: 0.04296875
  full_minus_worst_ablation: 0.01953125
  order_route0_generation_exact: 0.0078125
  order_route1_generation_exact: 0.046875
  last route1 probability: 0.999332070350647

roadmap consequence:
  The reduced task confirms that route selection is not the limiting factor.
  Even when the router chooses route1, the route1 transition is near chance.
  The next roadmap item is a direct transition-binding diagnostic, not another
  router, route attention, or route aux-weight sweep.
```

Reduced operation-order transition diagnostic:

```text
script:
  scripts/353_train_operation_order_transition_probe.py

model path:
  prompt tokens -> native embeddings -> recurrent transition state -> LM logits

not included:
  donor, MemoryOS, RAG, symbolic answer solver, hidden answer channel

held-out smoke:
  report: local_eval/operation_order_transition_probe_smoke_20260515/report.json
  decision: rejected
  full_generation_exact: 0.044921875
  full_minus_transition_off: 0.009765625
  full_minus_order_shuffle: 0.01171875

held-out capacity:
  report: local_eval/operation_order_transition_probe_capacity_mod16_20260515/report.json
  decision: rejected
  full_generation_exact: 0.431640625
  transition_off_generation_exact: 0.07421875
  order_shuffle_generation_exact: 0.2421875
  full_minus_transition_off: 0.357421875
  full_minus_order_shuffle: 0.189453125

same-seed capacity control:
  report: local_eval/operation_order_transition_probe_capacity_mod16_trainseed_eval_20260515/report.json
  decision: accepted_operation_order_transition_diagnostic
  full_generation_exact: 0.999755859375
  transition_off_generation_exact: 0.0712890625
  order_shuffle_generation_exact: 0.54150390625
  full_minus_transition_off: 0.928466796875
  full_minus_order_shuffle: 0.458251953125

roadmap consequence:
  This proves the reduced recurrent transition can become causal and solve the
  generated cases when the distribution is memorized. It does not yet prove
  held-out combinatorial generalization. The next architecture change should
  not be another router. It should add a generalization-oriented state/value
  codec or transition objective, then reinsert that mechanism into the
  dual/nested L6 core and require the same held-out LM-logit metric to improve.
```

Reduced transition trace/value-codec acceptance:

```text
implemented in:
  scripts/353_train_operation_order_transition_probe.py

tests:
  tests/test_operation_order_transition_probe.py

accepted reduced gates:
  learned codec + trace, mod16:
    report: local_eval/operation_order_transition_probe_learned_trace_mod16_20260515/report.json
    full_generation_exact: 0.828125
    full_minus_transition_off: 0.765625
    full_minus_order_shuffle: 0.361328125

  circular codec + trace, mod16:
    report: local_eval/operation_order_transition_probe_circular_trace_mod16_20260515/report.json
    full_generation_exact: 0.970703125
    full_minus_transition_off: 0.935546875
    full_minus_order_shuffle: 0.427734375

  circular codec + trace, mod32/d256/s3000:
    report: local_eval/operation_order_transition_probe_circular_trace_mod32_d256_s3000_20260515/report.json
    full_generation_exact: 0.89453125
    transition_off_generation_exact: 0.0390625
    order_shuffle_generation_exact: 0.4609375
    state_reset_generation_exact: 0.0
    full_minus_transition_off: 0.85546875
    full_minus_order_shuffle: 0.43359375

roadmap consequence:
  The L6 route1 bottleneck now has a concrete repair target. The winning
  reduced recipe is not another route wrapper. It is:

  1. expose recurrent transition traces;
  2. supervise those traces on stepwise operation results through the LM/value
     token path;
  3. use a stable value codec instead of relying only on random value-token
     embeddings;
  4. keep transition_off/order_shuffle/state_reset ablations on the same
     final LM-logit metric.

  The next QTRM-native L6 experiment should use the existing state-trace depth
  machinery as the trace-supervision route and add the missing stable value
  codec to the native ETD model before another dual/nested promotion attempt.
```

First L6 trace/value-codec transplant result:

```text
code changes:
  NativeQTRMETDLM:
    value_codec=learned|circular
    value_token_ids for arbitrary tokenizer mappings

  mixed text runner:
    --value-codec circular
    guard: circular requires --tokenizer-mode number
    value_token_ids_for_tokenizer maps 00..31 to the model's value codec

experiments:
  char tokenizer:
    report: local_eval/qtrm_native_l6_seed338_circular_value_trace_repair_s1200_20260515/report.json
    decision: rejected
    full_generation_exact: 0.0546875
    invalid contract: char tokens are not atomic values

  number tokenizer:
    report: local_eval/qtrm_native_l6_seed338_number_circular_trace_repair_s1200_20260515/report.json
    decision: rejected
    full_generation_exact: 0.068359375
    min_family_generation_exact: 0.023391812865497075
    full_minus_worst_ablation: 0.0078125

roadmap consequence:
  The reduced diagnostic separated OP tokens from VALUE tokens. The text L6
  task currently represents both operation ids and numeric values with the
  same two-digit tokens. Circular value coding therefore helps value manifold
  consistency but blurs operation role identity. Next full model step:

  1. add role-conditioned token embeddings for number tokens based on prompt
     segment/position, or introduce separate atomic OP tokens and VALUE tokens;
  2. keep state-trace depth supervision;
  3. rerun the same L6 dual/nested gate only after op/value roles are no
     longer sharing one embedding manifold.
```

Updated L6 d256 QTRM-native result:

```text
accepted checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

report:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/report.json

decision:
  accepted_l6_d256_number_oprole_circular_trace_revrepair

architecture:
  tokenizer-mode: number
  op-role tokens: enabled
  value codec: circular shared LM readout
  d_model: 256
  think_structure:
    trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier

training route:
  from-scratch d256 warmup:
    local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_s3000_20260515/report.json
  continuation:
    local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_cont_s7000_20260515/report.json
  revchain-heavy retention repair:
    local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/report.json

decisive metrics:
  full_generation_exact: 0.37890625
  min_family_generation_exact: 0.11695906432748537
  full_minus_think0: 0.3515625
  full_minus_worst_ablation: 0.33203125
  full_minus_carrier_off: 0.09765625
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.046875
  z_l_zero_generation_exact: 0.021484375
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.28125

by_family:
  checksum: 0.8705882352941177
  modchain: 0.15204678362573099
  revchain: 0.11695906432748537

roadmap consequence:
  This is now the strongest local L6 QTRM-native result. The earlier failed
  migrated number/op-role branch was not enough; the winning path required
  native d256 capacity, circular value readout, trace/depth supervision, and a
  short revchain-heavy retention repair.

  Do not yet call this a general LLM or L7 result. Next required gates:
    1. seed-stability over at least three eval seeds;
    2. checkpoint midpoint/rollback audit around the revchain repair;
    3. language non-regression on the native text path;
    4. length generalization beyond program_len=6.
```

L6 d256 revrepair eval-seed stability:

```text
summary:
  local_eval/qtrm_native_l6_d256_revrepair_seed_stability_summary_20260515.json

decision:
  accepted_l6_d256_revrepair_seed_stable_3eval

eval seeds:
  9340:
    full_generation_exact: 0.37890625
    min_family_generation_exact: 0.11695906432748537
    full_minus_worst_ablation: 0.33203125
  9341:
    full_generation_exact: 0.36328125
    min_family_generation_exact: 0.09941520467836257
    full_minus_worst_ablation: 0.294921875
  9342:
    full_generation_exact: 0.3828125
    min_family_generation_exact: 0.1111111111111111
    full_minus_worst_ablation: 0.34765625

roadmap consequence:
  The accepted d256 revrepair checkpoint is now seed-stable under the immediate
  3-eval-seed L6 gate. The next promotion blockers are no longer L6 family
  floor or causal ablation. They are:
    1. midpoint/rollback audit around the revchain-heavy repair;
    2. language non-regression;
    3. length generalization beyond program_len=6.
```

L6 d256 midpoint audit:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_mid1500_audit_20260515/report.json

decision:
  accepted_l6_d256_revrepair_mid1500_audit

metrics:
  full_generation_exact: 0.34375
  min_family_generation_exact: 0.08771929824561403
  full_minus_worst_ablation: 0.298828125
  full_minus_carrier_off: 0.123046875

roadmap consequence:
  The revchain-heavy repair passes at step 1500 as well as the restored-best
  final checkpoint. This satisfies the immediate midpoint/rollback audit.
  Remaining gates:
    1. language non-regression;
    2. program_len generalization beyond 6.
```

L6 d256 revrepair len7 zero-shot audit:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_eval_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

setup:
  steps: 0
  program_len: 7
  train_think_steps/eval_think_steps: 7
  resume_allow_missing: true
  pos_embed_resize_strategy: random_tail

resume load:
  pos_embed.weight resized from [46, 256] to [48, 256]
  copied_rows: 46
  random_tail filled_rows: 2

decision:
  rejected

metrics:
  full_generation_exact: 0.015625
  min_family_generation_exact: 0.0
  full_minus_think0: 0.009765625
  full_minus_worst_ablation: -0.0078125
  full_minus_carrier_off: -0.00390625
  state_reset_generation_exact: 0.017578125
  op_zero_generation_exact: 0.0234375
  z_l_zero_generation_exact: 0.015625
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.01953125

roadmap consequence:
  The accepted L6 d256 revrepair checkpoint is a valid L6 raw-reasoning
  baseline, not a length-general L7 model. The len7 zero-shot run fails both
  accuracy and causal-ablation criteria. The next valid promotion path is:
    1. preserve the accepted L6 checkpoint as canonical baseline;
    2. train a length-curriculum or positional-generalization repair from it;
    3. rerun len6 retention and len7 acceptance gates;
    4. reject any L7 claim unless both pass on normal LM logits.
```

L6 d256 revrepair len7 curriculum repair:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_curriculum_s1500_20260515/report.json

canonical argmax eval:
  local_eval/qtrm_native_l6_d256_revrepair_len7_curriculum_argmax_eval_20260515/report.json
  same decisive metrics; still rejected

resume checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

setup:
  steps: 1500
  program_len: 7
  train_think_steps/eval_think_steps: 7
  active_len_curriculum_min: 4
  active_len_replay_loss_weight: 0.02
  retention_kl_loss_weight: 0.05
  pos_embed_resize_strategy: repeat_last

decision:
  rejected

metrics:
  full_generation_exact: 0.154296875
  min_family_generation_exact: 0.023391812865497075
  full_minus_think0: 0.12890625
  full_minus_worst_ablation: 0.11328125
  full_minus_carrier_off: 0.08984375

by_family:
  checksum: 0.40588235294117647
  modchain: 0.03508771929824561
  revchain: 0.023391812865497075

roadmap consequence:
  Length curriculum is not enough to accept L7, but it is not a dead end. It
  recovers a causal recurrent signal on len7, while zero-shot had no useful
  depth or ablation gain. The next experiment should not change topology first;
  it should run a hard-family len7 repair from this curriculum checkpoint and
  require:
    1. len7 full exact >= 0.30;
    2. len7 min-family exact >= 0.08;
    3. full-minus-think0 and full-minus-worst-ablation remain positive;
    4. a separate len6 retention eval still passes the accepted L6 gate.
```

L7 hard-family continuation from len7 curriculum:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_hardfamily_s1500_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.189453125
  min_family_generation_exact: 0.023391812865497075
  full_minus_think0: 0.16015625
  full_minus_worst_ablation: 0.146484375
  full_minus_carrier_off: 0.10546875

by_family:
  checksum: 0.49411764705882355
  modchain: 0.023391812865497075
  revchain: 0.05263157894736842

roadmap consequence:
  Hard-family continuation moves the len7 model in the right direction, but
  still does not pass. The remaining bottleneck is not lack of recursive
  causality; the depth/ablation gains are now strong. The bottleneck is
  modchain family generalization at length 7. The next valid local repair is
  modchain-focused continuation with 512-case periodic family-floor selection,
  not a new backbone/topology.
```

L7 modchain-focused continuation:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.2578125
  min_family_generation_exact: 0.04678362573099415
  full_minus_think0: 0.232421875
  full_minus_worst_ablation: 0.212890625
  full_minus_carrier_off: 0.16796875

by_family:
  checksum: 0.6823529411764706
  modchain: 0.04678362573099415
  revchain: 0.04678362573099415

roadmap consequence:
  This is the strongest len7 result so far, but still rejected. The correct
  next action is not architecture shopping. Continue the same QTRM-native
  dual/nested recurrent path with a low-LR modchain/revchain balance repair and
  keep 512-case family-floor periodic selection. Only after a len7 acceptance
  should we run len6 retention and language non-regression gates.
```

L7 balanced hard-family continuation:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_balanced_hard_s900_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt

setup:
  steps: 900
  program_len: 7
  task_families: modchain x5, revchain x5, checksum x1
  family_dro_loss_weight: 0.15
  periodic_eval_score_mode: family_floor
  eval_during_training_cases: 512
  eval_answer_space_argmax: true
  lr: 8e-6

periodic trend on 512-case eval:
  step0 exact/min_family: 0.2578125 / 0.04678362573099415
  step300 exact/min_family: 0.248046875 / 0.029239766081871343
  step600 exact/min_family: 0.244140625 / 0.017543859649122806
  step900 exact/min_family: 0.23828125 / 0.03508771929824561

decision:
  rejected

metrics:
  full_generation_exact: 0.2578125
  think0_generation_exact: 0.025390625
  full_minus_think0: 0.232421875
  full_minus_worst_ablation: 0.212890625
  min_family_generation_exact: 0.04678362573099415
  full_minus_carrier_off: 0.16796875
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0
  z_l_zero_generation_exact: 0.044921875
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.08984375

by_family:
  checksum: 0.6823529411764706
  modchain: 0.04678362573099415
  revchain: 0.04678362573099415

roadmap consequence:
  Balanced continuation does not improve over the modchain-focused checkpoint.
  The restored best checkpoint is step0, and later training degrades both full
  exact and family floor. The next valid step is not another blind continuation;
  run an operation/position/family breakdown for len7 modchain and revchain,
  then repair the specific transition failure that the breakdown exposes.
```

L7 operation breakdown and reduced transition control:

```text
full-QTRM operation breakdown:
  report:
    local_eval/qtrm_native_l6_d256_revrepair_len7_opbreakdown_2048_20260515/report.json
  eval cases: 2048
  decision: rejected
  full_generation_exact: 0.23291015625
  min_family_generation_exact: 0.03513909224011713
  full_minus_think0: 0.20166015625
  full_minus_worst_ablation: 0.1904296875

full-QTRM by_family:
  checksum: 0.624633431085044
  modchain: 0.03513909224011713
  revchain: 0.03953147877013177

worst modchain last-op slices:
  op07: 0.0
  op06: 0.010638297872340425
  op05: 0.0297029702970297

state-transition codec repair:
  report:
    local_eval/qtrm_native_l6_d256_revrepair_len7_state_transition_codec_s600_20260515/report.json
  resume:
    local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt
  decision: rejected
  best checkpoint: step0
  full_generation_exact: 0.2578125
  min_family_generation_exact: 0.04678362573099415

reduced transition control:
  report:
    local_eval/operation_order_transition_probe_len7_circular_trace_mod32_d256_s10000_20260515/report.json
  decision:
    accepted_operation_order_transition_diagnostic
  full_generation_exact: 0.998046875
  fwd: 0.998046875
  rev: 0.998046875
  transition_off_generation_exact: 0.0283203125
  order_shuffle_generation_exact: 0.5166015625
  state_reset_generation_exact: 0.0
  full_minus_transition_off: 0.9697265625
  full_minus_order_shuffle: 0.4814453125

roadmap consequence:
  Program_len7 ordered recurrent transition is learnable. The full-QTRM failure
  is therefore not an impossibility result and not a reason to abandon QTRM
  native. The missing mechanism is a faithful transplant of the reduced
  transition cell's causal recipe into the full token->core->LM path:
  family-conditioned ordered op read, recurrent state update, circular value
  manifold, and trace loss that directly supervises the state trajectory. Short
  low-LR full-QTRM continuations and detached auxiliary codec heads are now
  rejected as the shortest path.
```

Stronger full-QTRM transplant triage:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_transition_transplant_lr3e5_s1200_20260515/report.json

resume:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt

setup:
  lr: 3e-5
  steps: 1200
  program_len: 7
  same nested dual-z core-carrier path
  state_trace_depth_loss_weight: 0.5
  core_step_codec_loss_weight: 0.2
  core_step_op_codec_loss_weight: 0.1
  core_step_position_codec_loss_weight: 0.05

periodic trend:
  step0 exact/min_family: 0.2578125 / 0.04678362573099415
  step300 exact/min_family: 0.21484375 / 0.03508771929824561
  step600 exact/min_family: 0.244140625 / 0.017543859649122806
  step900 exact/min_family: 0.17578125 / 0.029239766081871343
  step1200 exact/min_family: 0.244140625 / 0.04093567251461988

decision:
  rejected

roadmap consequence:
  Raising LR and strengthening detached auxiliary state/value/op/position
  losses still restores the initial checkpoint as best. The next step should be
  a canonical transition-cell transplant inside the main core path, not another
  stronger side-head continuation.
```

Qwen-assisted native language bootstrap:

```text
question:
  Can Qwen3.5-2B be used to make a QTRM-native LM quickly through healing tune?

answer:
  Yes for language bootstrapping, no for proving QTRM-native reasoning by
  itself. Qwen may provide tokenizer, initialization hints, cached logits, or
  teacher distributions. It must not be present in the final inference path if
  the result is called QTRM-native.

native-allowed path:
  Qwen tokenizer / text corpus / optional cached teacher logits
  -> QTRM-native token embeddings
  -> mandatory QTRM recursive core
  -> native LM head
  -> autoregressive text

not native:
  prompt -> Qwen donor forward -> donor hidden states/logits -> QTRM sidecar

recommended stages:
  1. Qwen tokenizer + QTRM-native LM smoke on ordinary text.
  2. Offline Qwen top-k logit distillation or CE+KL language healing.
  3. Core-on language non-regression: recurrence must not cause repetition.
  4. Add raw reasoning gates only after the native text path is stable.
  5. Promote only if Qwen is absent at inference and core ablations still matter.

expectation:
  This can make the native LM stop degenerating much faster than training from
  scratch, but it will not instantly give Qwen-level knowledge or reasoning.
  It is a language prior transfer, not a replacement for native recursive-core
  acceptance.
```

Web reference update, 2026-05-15:

```text
Qwopus3.5-27B-v3:
  Source:
    https://huggingface.co/Jackrong/Qwopus3.5-27B-v3
    https://github.com/R6410418/Jackrong-llm-finetuning-guide

  Observed pattern:
    Base model: Qwen/Qwen3.5-27B
    Training: Unsloth + LoRA SFT
    Masking: response-only training around assistant/<think> region
    Data: high-fidelity reasoning/CoT/coding/chat distillation sets
    Runtime: one merged/fine-tuned model, not a separate donor sidecar
    Evidence for this model being a layer-stack frankenmerge: not found in the
      model card. The public frankenmerge/heal-tune description found during
      search refers to Qwopus-GLM-18B-Merged, a separate 64-layer merge of two
      Qwen3.5-9B finetunes, followed by a 1000-step QLoRA healing run.

  Meaning for QTRM:
    Qwopus is a precedent for "donor-assisted but native-at-inference" only in
    the teacher/data/initialization sense. It is not a precedent for keeping a
    frozen donor forward pass inside the final architecture.

  Useful pieces to copy:
    1. response-only SFT mask;
    2. high-quality reasoning scaffold data;
    3. LoRA/QLoRA first, merge/export later;
    4. teacher-generated answer/continuation text as a language scaffold;
       visible CoT must not be copied into QTRM-native language bootstrap;
    5. benchmark before/after against the base model.

  Not enough for our claim:
    Qwopus improves a Qwen-family model by post-training. It does not prove a
    new QTRM-native recurrent architecture. For us, Qwen/Qwopus should be a
    language/reasoning teacher used offline; final inference must still be:

      prompt tokens
      -> QTRM-native embeddings
      -> mandatory nested dual-z recurrent core
      -> native LM head
      -> autoregressive text
```

Language-first bootstrap literature update, 2026-05-15:

```text
goal:
  Make QTRM-native speak coherent text quickly before rejoining the L7 raw
  reasoning path.

why language first:
  If the native token->core->LM-head path cannot predict ordinary text, every
  reasoning gain will be trapped behind a broken renderer/generator. Language
  viability is therefore a prerequisite, not a distraction.

most relevant prior:
  TinyStories:
    Synthetic simple stories can make very small LMs produce fluent and
    coherent text. Use this as the first non-degenerate text smoke curriculum.
    https://huggingface.co/papers/2305.07759

  Textbooks Are All You Need / phi:
    High-quality textbook-style data and synthetic exercises can move small
    models far faster than broad noisy web data.
    https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need/
    https://arxiv.org/abs/2309.05463

  FineWeb-Edu and DCLM:
    Dataset curation, model-based filtering, deduplication, and educational
    quality selection materially improve pretraining efficiency.
    https://huggingface.co/papers/2406.17557
    https://arxiv.org/abs/2406.11794

  MiniLLM:
    Reverse-KL style distillation is better suited to generative LMs than naive
    forward-KL and improves precision/calibration/long generation.
    https://www.microsoft.com/en-us/research/publication/knowledge-distillation-of-large-language-models/

  GKD:
    On-policy distillation trains on the student's own generated sequences to
    reduce train/inference distribution mismatch.
    https://huggingface.co/papers/2306.13649

  Pre-training distillation / sparse logits:
    Teacher logits help pretraining, but naive tiny top-k caches can miscalibrate
    the student. Prefer full/logit-rich cache when affordable, or top-p/top-k
    with residual-tail handling.
    https://aclanthology.org/2025.acl-long.181.pdf
    https://aclanthology.org/2025.acl-long.885.pdf

  Orca / Distilling step-by-step:
    Reasoning traces are useful after basic language works, but they should not
    be the first language curriculum because small models can imitate trace
    style without learning the process.
    https://arxiv.org/abs/2306.02707
    https://arxiv.org/abs/2305.02301

  GaLore / ReLoRA:
    If full native pretraining is memory limited, prefer full-parameter learning
    with memory-efficient optimizers or periodic low-rank update merging over
    pure adapter-only training for the base language phase.
    https://arxiv.org/abs/2403.03507
    https://arxiv.org/abs/2307.05695

chosen QTRM-native recipe:
  Stage A: tiny coherent-text curriculum
    Data: TinyStories-style + short Korean/English educational passages.
    Loss: standard CE first.
    Acceptance: greedy generation is non-degenerate; repetition metrics pass.

  Stage B: textbook/edu corpus
    Data: FineWeb-Edu/DCLM-style filtered text plus synthetic textbook snippets.
    Loss: CE + optional teacher KD.
    Acceptance: validation CE falls and language sample quality improves.

  Stage C: offline Qwen/Qwopus teacher cache
    Data: same texts passed through Qwen/Qwopus teacher.
    Store: top-p/top-k logits with enough tail correction, or use CE on
      teacher-generated continuations when logits are too expensive.
    Loss: CE + reverse-KL/temperature KD; avoid tiny top-k-only KL.
    Boundary: language teacher data is answer-only/continuation-only. Do not
      train visible CoT or <think> text into the native LM surface channel.

  Stage D: on-policy repair
    Generate QTRM-native samples, have teacher/verifier mark degeneracy,
    grammar failures, or short continuation targets, then train back on those.
    This follows the GKD intuition and directly attacks inference-time mismatch.

  Stage E: reasoning traces only after language acceptance
    Add answer-only reasoning examples and optional latent/core supervision.
    Visible CoT remains non-canonical because QTRM's reasoning claim is latent
    recurrence, not language-space chain-of-thought imitation.
    Keep L7 raw reasoning gates separate.

do not:
  Do not start with long CoT-only data.
  Do not put visible CoT, <think>, or teacher reasoning prose into the
  language bootstrap corpus.
  Do not use teacher at runtime.
  Do not call a donor-sidecar fluent result QTRM-native.
  Do not rely on tiny top-k logits without calibration/tail handling.
  Do not use LoRA-only as the canonical native base unless followed by merge or
  full native healing; the base LM path must actually learn.
```

Implemented bootstrap tooling:

```text
scripts:
  scripts/354_train_qtrm_native_language_bootstrap.py
  scripts/354_run_qtrm_native_language_bootstrap.sh

features:
  - donorless QTRM-native training only;
  - built-in TinyStories-style and textbook-style bilingual seed corpus;
  - optional external text files/globs;
  - optional offline teacher text/JSONL ingestion;
  - optional Hugging Face tokenizer path via --tokenizer-name;
  - depth sweep, thinking-block-off eval, repetition metrics;
  - on-policy candidate JSONL export for later teacher/verifier repair.

first CUDA triage, 2026-05-15:
  command:
    PYTHONPATH=src .venv/bin/python \
      scripts/354_train_qtrm_native_language_bootstrap.py \
      --out-dir local_eval/qtrm_native_language_bootstrap_triage_20260515 \
      --device cuda --stage-a-steps 120 --stage-b-steps 240 \
      --stage-c-steps 0 --tiny-repeats 128 --textbook-repeats 96 \
      --seq-len 64 --d-model 64 --n-heads 4 --n-kv-heads 2 \
      --d-ff 128 --batch-size 64

  result:
    accepted: true
    tiny_last_loss: 2.1199
    edu_last_loss: 1.3097
    eval_think_loss_depth4: 1.5084
    eval_think0_loss: 3.2798
    depth_sweep:
      0: 3.2798
      1: 2.7307
      2: 2.1051
      4: 1.5084
    sample_unique_chars: 26
    max_run_fraction: 0.0168

  interpretation:
    This is not broad language ability yet. It is the first positive
    language-first bootstrap signal: the native path learns the corpus quickly,
    generation is non-degenerate, and the recurrent depth helps LM loss.
    Next work should move from char tokenizer to Qwen tokenizer and add offline
    teacher text/logit artifacts.

Qwen tokenizer smoke:

```text
report:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_smoke/report.json

command:
  HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python \
    scripts/354_train_qtrm_native_language_bootstrap.py \
    --tokenizer-name Qwen/Qwen3.5-2B-Base \
    --device cuda --stage-a-steps 1 --stage-b-steps 0 --stage-c-steps 0 \
    --tiny-repeats 1 --textbook-repeats 1 --seq-len 16 \
    --d-model 16 --n-heads 4 --n-kv-heads 2 --d-ff 32 --batch-size 2

result:
  accepted: true under permissive tokenizer-smoke thresholds
  tokenizer_kind: hf
  tokenizer_name: Qwen/Qwen3.5-2B-Base
  vocab_size: 248077

interpretation:
  This proves the Qwen tokenizer path is wired. It is not a quality claim:
  one step on a 248k-vocab model remains near random loss. The next real run
  should use a larger d_model and enough CE/KD steps.
```

Offline teacher cache pipeline, 2026-05-15:

```text
scripts:
  scripts/355_build_qtrm_language_teacher_cache.py
  scripts/355_build_qtrm_language_teacher_cache.sh

purpose:
  Generate offline Qwen/Qwopus language artifacts for QTRM-native bootstrap.
  The teacher is not part of QTRM inference.

artifact schema:
  prompt: instruction prompt used only for teacher generation
  seed_text: original clean source text
  answer: teacher continuation
  text / teacher_text: seed_text + answer, without instruction prompt
  topk_logprobs: optional future KD payload

guards:
  - /no_think switch enabled by default;
  - bad_words_ids suppress visible <think> tokens;
  - strip/reject visible think blocks;
  - repetition and minimum-length filters.
  - 354 bootstrap also strips visible think blocks from arbitrary teacher
    JSONL/text before using it as language data.

real Qwen2B smoke:
  report:
    local_eval/qtrm_language_teacher_cache_qwen2b_smoke_v3/teacher_text.jsonl.report.json
  output:
    local_eval/qtrm_language_teacher_cache_qwen2b_smoke_v3/teacher_text.jsonl
  model:
    Qwen/Qwen3.5-2B-Base
  written:
    2 records
  key fix:
    Earlier prompt versions produced option lists or <think> leakage. v3 stores
    clean continuation text only and keeps the teacher instruction out of the
    training text.

interop smoke:
  report:
    local_eval/qtrm_native_language_bootstrap_teacher_jsonl_qwen_tokenizer_smoke_v3/report.json
  result:
    accepted under permissive 2-step smoke thresholds
  meaning:
    teacher JSONL -> Qwen tokenizer -> donorless QTRM-native bootstrap path is
    wired. This remains a plumbing test, not a language quality claim.
```

Teacher boundary correction:

```text
User concern:
  QTRM-native is a latent-reasoning architecture, not a language-space CoT
  imitation model. Therefore teacher-generated CoT should not become the
  training target for language bootstrap.

decision:
  Correct. Teacher usage is allowed only for:
    - surface language distribution: clean continuation/answer text;
    - optional future logits/KD over normal next tokens;
    - optional verifier labels or latent targets for core training.

  Teacher usage is not allowed for:
    - visible CoT SFT in the language stage;
    - copying <think> traces into QTRM-native outputs;
    - replacing causal latent-depth/ablation evidence with fluent rationale
      imitation.

code guard:
  scripts/354_train_qtrm_native_language_bootstrap.py strips visible <think>
  blocks from teacher text/JSONL. scripts/355_build_qtrm_language_teacher_cache.py
  generates continuation-only records and keeps the teacher instruction out of
  teacher_text.
```

Tokenizer and language-bootstrap update, 2026-05-15:

```text
problem:
  The first external-language runs showed that broad Qwen-tokenizer training
  leaves a tiny QTRM-native model fighting a 248k-vocab LM head. Character
  training makes the optimization easy but the surface language becomes too
  loose. Therefore tokenizer/output geometry is a real bottleneck.

implemented:
  scripts/354_train_qtrm_native_language_bootstrap.py:
    - --compact-hf-vocab for active Qwen-vocab remapping
    - --train-byte-bpe-tokenizer for local byte-level BPE
    - --repair-jsonl / --repair-jsonl-repeats for on-policy family repair

  scripts/356_eval_qtrm_native_language_generalization.py:
    - restores byte-BPE checkpoints
    - keeps tokenizer JSON out of reports

best current checkpoint:
  local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_repair_s1900_20260515/last.pt

best current recipe:
  tokenizer:
    byte-level BPE, vocab_size=8192, explicit <|qtrm_eos|>

  corpus:
    local_eval/external_language_corpus/qtrm_native_external_balanced_20260515.jsonl
    plus targeted answer-only repair:
      local_eval/qtrm_native_language_repair_unseen_failures_20260515.jsonl

  metrics:
    think_eval_loss_depth4: 4.2884
    think0_loss: 7.0308
    depth0: 7.0308
    depth1: 5.4790
    depth2: 4.6146
    depth4: 4.2884

accepted:
  bootstrap gate
  unseen2 near-family generalization gate
  unseen3 near-family generalization gate

rejected:
  wider paraphrase unseen4 gate

meaning:
  This is the strongest donorless native language scaffold so far. It is not
  broad language ability. It is a small coherent answer model with causal
  recurrent depth gains and limited family generalization.

next required work:
  1. Build a larger paraphrase-diverse answer-only corpus from external
     datasets or offline teacher artifacts.
  2. Keep byte-BPE 8k/16k as the canonical tokenizer axis unless evidence
     disproves it.
  3. Fix a broad unseen suite before further repair so we cannot overfit to the
     currently failing prompts.
  4. Promote only if broad unseen passes and depth4/core-on still beats
     think0/thinking-block-off.
  5. Only after language broadening, rejoin QTRM-native raw reasoning gates.
```

Qwen3 multilingual lesson and MSA ordering, 2026-05-15:

```text
question:
  Should we start from MSA architecture now, or first obtain multilingual
  general language ability?

decision:
  Do not start with MSA as the next canonical implementation step.

why:
  Qwen3's multilingual capability is not a memory architecture trick. The
  official Qwen3 report/blog describes:
    - 119 languages and dialects;
    - about 36T pretraining tokens;
    - staged pretraining from general language to knowledge/STEM/code/reasoning
      data and then long-context data;
    - post-training that combines thinking and non-thinking behavior.

  Therefore the lesson for QTRM-native is:
    language coverage and multilingual robustness are primarily data/tokenizer/
    curriculum/post-training problems before they are MSA problems.

current local evidence:
  best checkpoint:
    local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_repair_s1900_20260515/last.pt

  accepted:
    bootstrap
    near-unseen2
    near-unseen3

  rejected fixed broad unseen suite:
    data/eval/qtrm_native_language_broad_unseen_20260515.jsonl
    local_eval/qtrm_native_language_generalization_gate_broad_unseen_byte_bpe8k_repair_20260515/report.json

  failure:
    semantic paraphrases still route to nearby memorized answer templates.

ordering:
  1. QTRM-native multilingual/broad language:
     byte-BPE 8k/16k, answer-only multilingual corpus, fixed broad unseen
     suite, depth/core-off ablations.

  2. QTRM-native raw reasoning:
     recursive core gates after language is not broken.

  3. MSA/LM2/neural memory:
     add only when the model can already read/write normal text and solve
     short-context reasoning. Promote MSA only if memory_on beats memory_off,
     router_off, and chunk_shuffle on length/distractor sweeps.

MSA role:
  MSA is for long-context/trainable-memory scaling, not for rescuing missing
  base multilingual language ability.
```

Bilingual-first language target, 2026-05-15:

```text
minimum target:
  QTRM-native must first become reliable in English and Korean.

why this scope:
  English gives broad technical/general coverage.
  Korean is a first-class target language for this project.
  Other languages should remain architecturally extensible, but they are not
  required for the first native-language acceptance gate.

required before MSA:
  1. English/Korean answer-only corpus large enough to avoid template routing.
  2. Byte-BPE 8k/16k tokenizer trained with English/Korean coverage.
  3. Fixed bilingual core suite:
     data/eval/qtrm_native_language_bilingual_core_20260515.jsonl
  4. Broad paraphrase suite:
     data/eval/qtrm_native_language_broad_unseen_20260515.jsonl
  5. Depth/core-off evidence:
     depth4/core-on must beat think0 and thinking-block-off.

extension path:
  Add more languages later by adding language-balanced corpus shards, extending
  the tokenizer if needed, adding per-language core suites, and re-running the
  same recurrence/core-off gates. Do not change the architecture merely because
  another language is added.

current result:
  local_eval/qtrm_native_language_generalization_gate_bilingual_core_byte_bpe8k_repair_20260515/report.json

decision:
  rejected

diagnosis:
  The current byte-BPE repair checkpoint is not bilingual-ready. English works
  on some close families, but Korean and farther English paraphrases still
  collapse to memorized nearby answer templates. Continue with larger
  English/Korean answer-only data and fixed bilingual evaluation before MSA.
```

Bilingual scaffold promotion, 2026-05-15:

```text
promoted checkpoint:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv4_s4200_20260515/last.pt

why promoted:
  - donorless QTRM-native inference path;
  - byte-level BPE 16k;
  - English/Korean answer-only repair;
  - depth4 loss is far below depth0/think0;
  - fixed bilingual core gate accepted;
  - fixed broad unseen gate accepted;
  - direct inference smoke works in English and Korean.

accepted reports:
  bootstrap:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv4_s4200_20260515/report.json

  bilingual:
    local_eval/qtrm_native_language_generalization_gate_bilingual_core_bpe16k_d192_repairv4_groups_20260515/report.json

  broad unseen:
    local_eval/qtrm_native_language_generalization_gate_broad_unseen_bpe16k_d192_repairv4_groups_20260515/report.json

inference wrapper:
  scripts/359_infer_qtrm_native_language.py
  scripts/359_infer_qtrm_native_language.sh

important evaluator change:
  Broad/bilingual gates now use expected_keyword_groups rather than brittle
  exact flat keywords. Each group is still a required meaning slot, but valid
  paraphrases are allowed.

next bottleneck:
  The model is still a small scaffold. Do not call it a broad LLM. Next:
  larger external English/Korean corpus, wider heldout suite, then reasoning
  gates reattached without losing language.
```

External data and TST next step, 2026-05-15:

```text
new external corpus:
  local_eval/external_language_corpus/qtrm_native_external_bilingual_4500_20260515.jsonl

corpus report:
  local_eval/external_language_corpus/qtrm_native_external_bilingual_4500_20260515.jsonl.report.json

records:
  4500 balanced across UltraChat, FineWeb-Edu, and KoAlpaca

TST reference:
  docs/wiki/sources/token-superposition-training.md
  references/papers/2605.06546-efficient-pre-training-with-token-superposition.pdf

local primitive:
  src/qtrm_mm/tst.py

ordering:
  1. Train non-TST larger-corpus baseline.
  2. Evaluate bilingual/broad gates and depth/core-off non-regression.
  3. Add TST phase-1 superposed-bag pretraining as an efficiency experiment.
  4. Promote TST only if it reaches equal or better language gates faster
     without hurting greedy English/Korean inference or recurrent-depth gains.
```

External4500 baseline result, 2026-05-15:

```text
baseline checkpoint:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

accepted:
  bootstrap
  bilingual core
  broad unseen

depth evidence:
  depth0_loss: 5.3371
  depth4_loss: 1.5977
  thinking_block_off_loss: 5.3371

default inference now points here:
  scripts/359_infer_qtrm_native_language.sh

next canonical experiment:
  qtrm_native_language_tst_phase_smoke

TST accept rule:
  TST must preserve this baseline's English/Korean generation gates and
  recurrent-depth advantage while reducing training time/token budget to reach
  comparable loss or heldout behavior.
```

TST b4 smoke result, 2026-05-15:

```text
checkpoint:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b4_s3600_20260515/last.pt

result:
  rejected

why:
  bootstrap accepted, but bilingual and broad-unseen gates rejected.
  depth4 loss was also slightly worse than the non-TST external4500 baseline.

keep canonical baseline:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

allowed future TST work:
  bag_size=2 or shorter TST phase ratio with longer recovery.

not allowed:
  claiming TST progress from bootstrap loss alone.
```

TST b2 and b2-short result, 2026-05-15:

```text
baseline remains:
  local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

baseline depth4_loss:
  1.5977

bag_size=2 / 300 TST / 2400 CE:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_s3600_20260515/last.pt

  depth4_loss:
    1.6091

  bilingual gate:
    rejected

  broad unseen gate:
    rejected

bag_size=2 / 150 TST / 2550 CE:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_short_s3600_20260515/last.pt

  depth4_loss:
    1.6811

  decision:
    rejected before heldout promotion because it regressed more than b2 and
    produced generic sample drift.

current canonical rule:
  TST is not promoted for the current small QTRM-native English/Korean language
  scaffold. Keep TST available as a future large-scale throughput experiment,
  but do not spend the next iteration on TST tuning unless the objective changes
  to wall-clock/token-efficiency research.

next practical bottleneck:
  Improve the non-TST native language scaffold with more diverse English/Korean
  data and normal CE or gated on-policy repair, then reattach reasoning gates.
```

External9000 result and stop rule, 2026-05-15:

```text
external9000 corpus:
  local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl

source balance:
  UltraChat 3000
  FineWeb-Edu 3000
  KoAlpaca 3000

new continuation support:
  scripts/354_train_qtrm_native_language_bootstrap.py --init-checkpoint

why it matters:
  Repair can now continue from a QTRM-native language checkpoint while reusing
  the exact saved tokenizer and model-shape args. This avoids full retraining
  for every hard-family repair attempt.

runs:
  external9000_s4200:
    depth4_loss: 3.0979
    heldout: rejected

  external9000_s9900:
    depth4_loss: 2.3362
    heldout: rejected

  external9000_s9900_hardrepair_s1000:
    heldout: rejected
    diagnosis: hard-only oversampling caused retention loss

  external9000_s9900_balancedrepair_s800:
    heldout: rejected

  external9000_s9900_microrepair_s400:
    heldout: rejected
    remaining failures:
      What makes a repeated test useful?
      출처의 날짜는 왜 중요한가요?
      문장을 고칠 때 무엇을 먼저 확인해야 하나요?

decision:
  Do not promote external9000 yet. The canonical checkpoint remains:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

stop rule:
  Stop single-family repair loops when they trade one semantic slot for another.
  The next language experiment must use balanced family sampling or an automated
  retained-family replay builder, not manual oversampling of the latest failed
  prompts alone.
```

Qwen-width 4090 optimizer decision, 2026-05-15:

```text
goal:
  Make QTRM-native able to use Qwen3.5-2B width/weights on a single RTX 4090
  without returning to a runtime donor path.

canonical runtime path:
  prompt tokens
  -> native embeddings initialized from Qwen rows
  -> native encoder
  -> mandatory QTRM/TRM thinking core
  -> native decoder
  -> LM logits

chosen local full-ish optimizer:
  auto -> galore_adamw8bit on CUDA when galore-torch is installed

why:
  GaLore is the most directly relevant 4090 prior because it targets
  full-parameter learning by low-rank gradient projection and reports 7B
  pretraining feasibility on 24GB consumer GPUs. Q-GaLore/APOLLO/BAdam remain
  valid follow-up candidates, but GaLore 8-bit is the shortest working path in
  this repo because galore-torch + bitsandbytes are available now.

implementation:
  src/qtrm_mm/training_optimizers.py

supported flags:
  --optimizer auto|adamw|adamw8bit|paged_adamw8bit|galore_adamw|galore_adamw8bit
  --galore-rank
  --galore-update-proj-gap
  --galore-scale
  --galore-proj-type
  --galore-min-dim
  --galore-include-embeddings

default quality rule:
  Do not apply GaLore projection to token embeddings or LM head by default.
  Those tensors may carry imported Qwen lexical geometry and should be updated
  normally through the chosen 8-bit optimizer unless memory forces otherwise.

preflight:
  local_eval/qtrm_native_qwenwidth_galore_preflight_s1_20260515/report.json

preflight result:
  d_model: 2048
  d_ff: 6144
  n_heads: 8
  compact Qwen vocab: 4096
  pretrained_init.runtime_donor: false
  optimizer: galore_adamw8bit
  trainable params: 134,428,673
  one training step completed on RTX 4090

interpretation:
  This proves the memory-efficient optimizer path runs on 4090 for a
  Qwen-width native scaffold. It does not prove language quality because the
  run used only one step and was rejected by generation gates.

next:
  Run a longer Qwen-width compact-vocab language bootstrap with balanced
  heldout gates. If SVD/projection wall time is too high, compare:
    - galore_adamw8bit with smaller rank or larger update_proj_gap
    - adamw8bit baseline
    - APOLLO or BAdam after integration
```

Qwen3.5-style hybrid backend stabilization, 2026-05-15:

```text
canonical local backend:
  --delta-backend fla_gated_delta
  --strict-backends

reason:
  FLA GatedDeltaNet has high first kernel compile/warmup cost, but the warmed
  training path is materially faster than the PyTorch fallback. It is also the
  path closest to the Qwen3.5/Qwen3-Next-style 3:1 hybrid idea.

measured:
  d=128 qtrm_hybrid_3to1:
    torch fallback repeat fwd+bwd: ~22.4 ms
    FLA GatedDelta repeat fwd+bwd: ~9.3 ms

  d=2048 single 4-layer hybrid stack:
    trainable params: 224,682,464
    first compile fwd+bwd: ~47.6 s
    repeat fwd+bwd: ~10.5 ms
    peak allocated: ~2.2 GiB

native LM smoke:
  d=2048 encode/think/decode all qtrm_hybrid_3to1:
    runtime donor: false
    Qwen rows copied into native embeddings/LM head
    trainable params: 675,042,721
    official FLA mixers: 9/9
    result: accepted 1-step runtime smoke

  d=512 short training check:
    steps: 50
    train loss: 7.19 -> 6.37
    depth1 eval loss: 6.2005
    depth0 eval loss: 6.4210
    result: runtime/training path works, but generation collapsed to comma
      repetition under intentionally lax smoke gates

operational rule:
  Direct Python runs of scripts/354_train_qtrm_native_language_bootstrap.py must
  keep --max-text-chars explicit for smoke runs. The parser now defaults to
  120000 to avoid accidental full-corpus window expansion, but serious runs
  should still record the chosen limit in the command.

promotion rule:
  This only proves runtime viability. It is not a language-quality acceptance.
  Next promotion requires longer d=512/1024 language bootstrap, English/Korean
  non-degenerate generation, and depth/core ablations that do not damage normal
  next-token behavior.
```

Language scaffold promotion update, 2026-05-15:

```text
accepted small scaffold:
  local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_family_repair_v2_s400_20260515/last.pt

architecture:
  tokenizer:
    native byte-BPE, no runtime donor
  encode/think/decode:
    qtrm_hybrid_3to1
  delta backend:
    fla_gated_delta with --strict-backends
  recurrence:
    mandatory depth2 QTRM-native path
  trainable params:
    ~7.1M

why this matters:
  The accepted scaffold proves a donorless QTRM-native path can generate short
  English/Korean answers without punctuation loops, without extra role-marker
  leakage, and with a causal recurrent-depth advantage on next-token loss.

depth evidence:
  broad expansion checkpoint:
    depth0 loss: ~9.92
    depth1 loss: ~4.57
    depth2 loss: ~0.99

held-out regate:
  local_eval/qtrm_native_hybrid_fla_bytebpe_d192_broad_expand_heldout_regate_20260515/report.json
  result:
    accepted

temporal hard-negative regate:
  local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_family_repair_v2_semantic_regate_20260515/report.json
  result:
    accepted
  retained default gate:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_v2_default_retention_regate_20260515/report.json
    accepted

do not claim yet:
  - broad general LLM capability
  - Qwen-width language quality
  - robust temporal/date reasoning beyond the small hard-negative family
  - ASI or frontier-level reasoning

next promotion gates:
  1. Run seed stability on d192 byte-BPE.
  2. Add broader bilingual unseen semantic gates across temporal, evidence,
     writing, answer-quality, and uncertainty families.
  3. Scale the same accepted recipe to d512 with FLA strict backends.
  4. Revisit Qwen compact/Qwen-width only after byte-BPE d512 is stable.
```

d512 Qwen pretrained-init A/B update, 2026-05-15:

```text
question:
  Is native pretrained initialization actually better than random init for the
  QTRM-native language bootstrap?

setup:
  architecture:
    d_model: 512
    tokenizer: Qwen/Qwen3.5-2B-Base compact HF vocab, 8192 active rows
    encode/think/decode: qtrm_hybrid_3to1
    delta backend: fla_gated_delta with --strict-backends
    recurrent depth: train/eval think_steps=2
    runtime donor: false
  data:
    external bilingual 4500 jsonl + bilingual/core/broad repair jsonl
  schedule:
    stage_a=100, stage_b=200, stage_c=900

A. random init:
  checkpoint:
    local_eval/qtrm_native_d512_hybrid_compact_random_ab_s1200_20260515/last.pt
  bootstrap decision:
    rejected
  reason:
    sample_repetition_run_too_high on the generic free seed
  eval:
    broad unseen: rejected
    missed family:
      repeated measurements / science
  depth:
    depth0 loss: 4.5649
    depth1 loss: 1.7323
    depth2 loss: 1.4831

B. Qwen pretrained init:
  checkpoint:
    local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_20260515/last.pt
  initialization:
    Qwen/Qwen3.5-2B-Base embedding and LM-head rows copied into the native
    compact vocab through random projection 2048 -> 512.
  runtime donor:
    false
  bootstrap decision:
    rejected
  reason:
    sample_repetition_run_too_high on the generic free seed
  eval:
    broad unseen:
      local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_broad_unseen_eval_20260515/report.json
      accepted_language_generalization
    bilingual core:
      local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_bilingual_core_eval_20260515/report.json
      accepted_language_generalization
  depth:
    depth0 loss: 4.6104
    depth1 loss: 2.1403
    depth2 loss: 1.5099

interpretation:
  Pretrained init is not a magical loss win: eval loss is roughly tied with
  random init in this short run. Its concrete advantage is semantic coverage.
  Under the same d512 compact-vocab schedule, random init missed a broad unseen
  family while Qwen pretrained init passed both broad unseen and bilingual core.

promotion status:
  Qwen pretrained init is now the preferred d512 language-bootstrap candidate,
  but it is not fully canonical until the generic free-sample newline loop is
  repaired or the bootstrap gate seed is replaced with chat-template answer
  prompts that match the intended use.

original-settings rule:
  Because Qwen pretrained init beat random init on semantic gates, future
  Qwen-preinit work should preserve Qwen's original settings as much as
  hardware allows. The d512 random-projection result is a scout result, not the
  final orthodox form.

  Priority:
    1. Use the full Qwen tokenizer/vocabulary when feasible. The local
       Qwen/Qwen3.5-2B-Base tokenizer reports 248077 tokens.
    2. Use Qwen hidden width 2048 so embedding/LM-head rows can be copied
       directly instead of projected.
    3. Preserve Qwen's tied embedding policy when applicable.
    4. Preserve Qwen-like head/intermediate/norm/activation choices where they
       are compatible with the QTRM-native recurrent core.
    5. Use compact vocab or 2048->512 projection only for fast triage,
       not for the final pretrained-init claim.

canonical next architecture:
  Qwen-width QTRM-native:
    tokenizer: full Qwen tokenizer/vocab
    d_model: 2048
    pretrained init: direct Qwen embedding/head copy
    runtime donor: false
    core: mandatory QTRM recurrent path
    acceptance: chat-style generation + broad unseen + bilingual core +
      depth/core ablation

next:
  1. Run full-vocab Qwen-width d2048 smoke with explicit memory limits.
  2. If it fits on 4090, run a short chat-style healing pass.
  3. Re-run broad unseen and bilingual core.
  4. Only if full-vocab d2048 is infeasible, fall back to d2048 compact-vocab
     or d512 projection.
```

## Qwen-Backbone QTRM Native Bridge

```text
date:
  2026-05-15

why this replaces direct d2048 copy as the next step:
  Directly copying Qwen embeddings and LM head into a non-Qwen QTRM backbone is
  not enough. The hidden states between embedding and head are not Qwen hidden
  states, so Qwen language ability is mostly lost.

canonical Qwen3.5-integrated QTRM-native path:
  Keep the real Qwen3.5-2B token -> backbone -> hidden -> LM-head stream.
  Insert QTRMRecursiveCore as a mandatory recurrent transformation on the final
  hidden states before the same Qwen LM head:

    Qwen final hidden
      -> core_in_norm
      -> QTRM recurrent core
      -> core_out_norm
      -> mandatory residual transform
      -> same Qwen LM head

important boundary:
  This is not the old "frozen donor hidden states -> QTRM sidecar -> separate
  decoder" design. QTRM is now in the same causal logits path as Qwen's LM head.
  The report field `runtime_donor` is false because there is no separate donor
  branch at inference. The normal canonical path must set `mandatory_core=true`;
  `force_core_off` and `core_gate_override=0` remain diagnostic ablations only.

implemented:
  src/qtrm_mm/qwen_backbone_qtrm.py
  scripts/361_qwen_backbone_qtrm_smoke.py
  tests/test_qwen_backbone_qtrm.py

smoke:
  local_eval/qwen_backbone_qtrm_gate_smoke_20260515/report.json

acceptance:
  max_abs_delta_base_vs_hidden_path_gate0: 0.0
  max_abs_delta_base_vs_core_on: 0.65625
  accepted_equivalence: true
  accepted_core_causality: true
  accepted: true

meaning:
  gate=0 proves exact Qwen preservation.
  gate>0 proves QTRM can causally affect logits through the preserved Qwen
  LM-head path.

next gates:
  1. freeze Qwen, train the mandatory QTRM core on a small
     reasoning-language mixture;
  2. require language non-regression against Qwen gate=0;
  3. require reasoning gain with core_on > core_off;
  4. only then test partial Qwen unfreeze / healing tune.
```

## Qwen Integrated Native Correction

```text
date:
  2026-05-16

correction:
  The intended C path is not a compact bridge and not a tiny randomly
  initialized native-only model. The canonical direction is Qwen3.5-integrated
  QTRM-native:

    Qwen3.5 tokenizer/full vocab
    -> Qwen3.5 embedding
    -> Qwen3.5 original backbone/layers
    -> mandatory QTRM recursive core in the same causal hidden path
    -> Qwen3.5 LM head
    -> autoregressive text

implementation update:
  src/qtrm_mm/qwen_backbone_qtrm.py:
    QwenBackboneQTRM now supports `mandatory_core=True`.
    In that mode, the normal forward path uses core gate 1.0 and the gate logit
    is not trainable. `force_core_off` and `core_gate_override=0` are retained
    only for ablation and equivalence diagnostics.

  scripts/361_qwen_backbone_qtrm_smoke.py:
    default `core_impl` is now `qwen_layer_wrapped`, with Qwen layer 3 as the
    default transition prior.

  scripts/362_train_qwen_backbone_qtrm_core_gate.py:
    adds `--mandatory-core` and `--train-qwen`.
    reports `qtrm_native_integrated=true`, `standalone_graph=true`, and
    `runtime_donor=false`.

  scripts/386_run_qwen35_integrated_mandatory_core_gate.sh:
    one-command canonical Stage-1 runner.
    Qwen3.5 remains the integrated backbone; QTRM core is mandatory; Qwen is
    frozen by default; only the recurrent core/adapter path is trained.

pretrained init rule:
  Pretrained init does not imply every weight is immediately trainable. The
  fast path is:

    Stage 1: freeze Qwen, train mandatory QTRM core.
    Stage 2: unfreeze selected Qwen layers or adapters for healing tune.
    Stage 3: only if stable, expand Qwen trainability.

acceptance:
  Promote only if Qwen+mandatory-QTRM beats Qwen/core_off, deeper or better
  recurrent state improves the same metric, and bilingual/open-ended language
  non-regression still passes.
```

## Qwen Integrated Mandatory Core Gate

```text
date:
  2026-05-16

runner:
  scripts/386_run_qwen35_integrated_mandatory_core_gate.sh

report:
  local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516/report.json

checkpoint:
  local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516/last_core.pt

setup:
  model:
    Qwen/Qwen3.5-2B-Base
  qwen_trainable:
    false
  runtime_donor:
    false
  integrated_qwen_backbone:
    true
  standalone_graph:
    true
  mandatory_core:
    true
  core_impl:
    qwen_layer_wrapped
  qwen_core_layer_indices:
    [3]
  normal_core_gate:
    1.0
  residual_scale:
    0.05
  core_adapter_dim:
    128

gate:
  case_mode:
    hard_v1
  train_cases:
    768
  eval_cases:
    512
  steps:
    300

result:
  accepted:
    true
  base/core_off accuracy:
    0.029296875
  core_on accuracy:
    0.1171875
  gain:
    0.087890625
  min_family_gain:
    0.058823529411764705
  min_family_core_accuracy:
    0.09941520467836257
  language_top1_agreement:
    1.0

interpretation:
  This is the first accepted Qwen3.5-integrated mandatory-core result after the
  C-path correction. It is not a public benchmark win; it is a causal
  architecture gate showing that the mandatory QTRM core can improve the same
  LM-logit answer metric over the Qwen/core_off path while preserving a small
  language non-regression probe.
```

## Qwen Integrated Native Milestones

```text
date:
  2026-05-16

principle:
  Preserve the original Qwen3.5 backbone as much as possible. QTRM is not a
  side donor adapter; it is a mandatory recurrent core inside the same causal
  hidden-state-to-LM-logit path.

canonical graph:
  prompt/chat-template tokens
  -> Qwen3.5 tokenizer/full vocabulary
  -> Qwen3.5 token embeddings
  -> Qwen3.5 original backbone/layers
  -> mandatory QTRM recursive core
  -> Qwen3.5 LM head
  -> autoregressive text

M0 - Integrated Path Lock:
  status:
    accepted
  requirement:
    runtime_donor=false
    integrated_qwen_backbone=true
    standalone_graph=true
    mandatory_core=true
    normal_core_gate=1.0
  evidence:
    local_eval/qwen35_integrated_mandatory_core_smoke_20260516/report.json

M1 - Freeze Qwen, Train Mandatory QTRM Core:
  status:
    accepted
  trainable:
    QTRM core, norms, adapter
  frozen:
    Qwen3.5 embedding/backbone/LM head
  requirement:
    core_on > core_off on the same LM-logit metric
    language non-regression passes
  evidence:
    local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516/report.json

M2 - Partial Qwen Unfreeze:
  status:
    partial accepted; promotion still guarded by seed-stability and public
    language/benchmark gates
  trainable candidates:
    selected upper Qwen layers
    selected Qwen transition layer used by QTRM
    QTRM adapter/norm/core path
  frozen by default:
    tokenizer
    token embeddings
    most Qwen backbone layers
    LM head unless explicit healing requires it
  requirement:
    improve over M1 without losing language non-regression
    core_off ablation still removes the claimed gain

M3 - Healing Tune:
  status:
    accepted first local gate; still needs seed-stability and public benchmark
    promotion
  purpose:
    align the mandatory QTRM core with Qwen hidden space while preserving Qwen
    fluency and bilingual behavior.
  data:
    small reasoning-language mixture first
    then broader English/Korean language non-regression corpus
  losses:
    answer CE / next-token CE where appropriate
    donor/core_off KL only as preservation regularizer
    no teacher sidecar at inference
  requirement:
    M1/M2 reasoning gain retained
    English/Korean generation remains non-degenerate
    MMLU-Pro subset score improves over compact-preinit baseline

M4 - Public Benchmark Recheck:
  status:
    public-subset core gain accepted on 64-case, 256-case, and 512-case
    MMLU-Pro subsets; 1024-case recheck rejected; not Qwen3.6-27B parity
  suite:
    MMLU-Pro balanced subset first, then full split
  requirement:
    score improves over previous Qwen3.5 compact-preinit native score
    no claim of Qwen3.6-27B parity until public benchmark gap closes
    seed stability and full-split checks remain required before promotion

M5 - Scale/Release Candidate:
  status:
    pending
  allowed only after:
    M2/M3 pass seed-stability
    core depth/state ablations remain causal
    language and public MCQ gates do not regress
```

## M2 Partial Qwen Unfreeze Results

```text
date:
  2026-05-16

implementation:
  src/qtrm_mm/qwen_backbone_qtrm.py
    set_qwen_partial_trainable(...)

  scripts/362_train_qwen_backbone_qtrm_core_gate.py
    --unfreeze-qwen-layer-indices
    --qwen-lr
    --qwen-weight-decay
    finite-logit acceptance check

  scripts/387_run_qwen35_integrated_partial_unfreeze_gate.sh
    M2 runner

safety finding:
  Direct fp16 partial unfreeze can create non-finite language deltas. M2 defaults
  were changed to bfloat16 and very low Qwen LR.

M2 candidate A:
  unfreeze:
    Qwen layer 3
  report:
    local_eval/qwen35_integrated_partial_unfreeze_l3_s200_20260516/report.json
  decision:
    rejected
  reason:
    select_pair family gain stayed negative.
  metrics:
    gain: 0.046875
    min_family_gain: -0.0058823529411764774
    min_family_core_accuracy: 0.0935672514619883
    language_top1: 1.0
    finite_logits: true

M2 candidate B:
  unfreeze:
    Qwen layer 23
  report:
    local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/report.json
  checkpoint:
    local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/last_core.pt
  decision:
    accepted_m2_partial_unfreeze_family_floor
  metrics:
    base/core_off accuracy: 0.060546875
    core_on accuracy: 0.107421875
    gain: 0.046875
    min_family_gain: 0.0
    min_family_core_accuracy: 0.08187134502923976
    language_top1: 1.0
    finite_logits: true
    qwen_trainable_parameters: 52433408

interpretation:
  Opening the first full-attention transition layer (3) is too disruptive for
  family balance. Opening the final full-attention layer (23) is a safer M2
  healing direction: it preserves finite logits and language top1 while
  repairing the negative family floor on this seed. It is not yet a claim that
  M2 beats M1 aggregate reasoning; it is the first accepted partial-unfreeze
  safety/family-floor gate. Next promotion requires seed stability and broader
  language generation checks.
```

## M3 Qwen-Integrated Healing Tune Results

```text
date:
  2026-05-16

runner:
  scripts/388_run_qwen35_integrated_healing_tune.sh

language gate:
  scripts/389_run_qwen35_integrated_healing_language_gate.sh

init checkpoint:
  local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/last_core.pt

output checkpoint:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt

training:
  partial Qwen layer 23 unfreeze
  mandatory QTRM core on normal path
  runtime_donor=false
  bfloat16
  Qwen LR: 1.0e-6
  QTRM LR: 5.0e-5
  KL to core_off/base language path: 0.10
  steps: 100

reasoning report:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/report.json

reasoning metrics:
  base/core_off accuracy: 0.05078125
  core_on accuracy: 0.142578125
  gain: 0.091796875
  min_family_gain: 0.04678362573099415
  min_family_core_accuracy: 0.1286549707602339
  language_top1_agreement: 0.875
  finite_logits: true

language generation report:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_language_gate_20260516/report.json

language generation gate:
  accepted: true
  accepted_top1: true
  accepted_top5: true
  accepted_repetition: true
  accepted_unique_ratio: true
  accepted_finite_logits: true
  prompts: English and Korean short assistant prompts

decision:
  accepted_m3_first_healing_gate

interpretation:
  This is the first Qwen-integrated QTRM-native checkpoint that keeps the
  original Qwen tokenizer/full vocabulary/backbone/LM head inside one standalone
  graph, keeps the QTRM core mandatory on the normal path, improves the local
  hard_v1 reasoning gate over core_off, and passes a broad English/Korean
  non-degenerate generation gate. It is not yet a claim of Qwen3.6-27B parity;
  the next promotion step is M4 public benchmark recheck plus seed-stability.
```

## M4 Qwen-Integrated Public MCQ Recheck

```text
date:
  2026-05-16

evaluator:
  scripts/390_eval_qwen35_integrated_public_mcq.py

runner:
  scripts/390_run_qwen35_integrated_m4_public_mcq.sh

suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl
  MMLU-Pro validation, 64 category-balanced cases

checkpoint:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt

report:
  local_eval/qwen35_integrated_m4_mmlu_pro64_20260516/report.json

scorer:
  next-token option-letter log likelihood

comparison:
  same Qwen3.5 tokenizer/full vocab/backbone/LM head
  base/core_off vs mandatory QTRM core_on

result:
  accepted_core_gain: true
  accepted_parity: false
  finite_logits: true
  base/core_off: 22/64 = 0.34375
  core_on: 24/64 = 0.375
  core_gain_over_base: 0.03125
  min_core_gain: 0.01

balanced 256 recheck:
  report:
    local_eval/qwen35_integrated_m4_mmlu_pro_balanced256_20260516/report.json
  decision:
    rejected_m4_public_mcq_core_gain
  base/core_off: 92/256 = 0.359375
  core_on: 92/256 = 0.359375
  core_gain_over_base: 0.0

public-MCQ healing:
  trainer:
    scripts/391_train_qwen35_integrated_public_mcq_healing.py
  runner:
    scripts/391_run_qwen35_integrated_public_mcq_healing.sh
  checkpoint:
    local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt
  note:
    Directory name says coreonly, but this run used layer-23-open bookkeeping
    because the runner defaulted empty UNFREEZE_QWEN_LAYER_INDICES to 23 before
    the env-handling fix. Qwen LR was 0.0, so Qwen weights were not updated.
    Treat it as layer23-open/core-updated public-MCQ healing, not as a Qwen
    weight healing result.
  training report:
    local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/report.json
  independent verification:
    local_eval/qwen35_integrated_public_mcq_healing_l23_verified_m4_256_20260516/report.json
  decision:
    accepted_m4_public_mcq_core_gain
  verified result:
    base/core_off: 92/256 = 0.359375
    core_on: 96/256 = 0.375
    core_gain_over_base: 0.015625
    accepted_parity: false
  stronger language gate:
    report:
      local_eval/qwen35_integrated_public_mcq_healing_l23open_resid0p06_language_gate_20260516/report.json
    decision:
      accepted
    top1_agreement: 0.8333333730697632
    base_top1_in_core_top5: 1.0
    max_repeated_token_run: 1
    mean_unique_ratio: 0.7864583333333334
  true core-only control:
    report:
      local_eval/qwen35_integrated_public_mcq_healing_true_coreonly_val64_to_test256_s80_20260516/report.json
    decision:
      rejected_public_mcq_healing
    base/core_off: 92/256 = 0.359375
    core_on: 94/256 = 0.3671875
    core_gain_over_base: 0.0078125

512-case balanced recheck:
  suite:
    local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_512.jsonl
  materialization report:
    local_eval/m7_public_reasoning_suite/report_balanced_512.json
  best accepted checkpoint:
    local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt
  inference setting:
    residual_scale: 0.06
  report:
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_resid0p06_20260516/report.json
  decision:
    accepted_m4_public_mcq_core_gain
  result:
    base/core_off: 191/512 = 0.373046875
    core_on: 197/512 = 0.384765625
    core_gain_over_base: 0.01171875
    accepted_parity: false
  controls:
    residual_scale 0.05:
      report: local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_20260516/report.json
      result: 191/512 -> 196/512, gain 0.009765625, rejected by threshold
    residual_scale 0.04:
      report: local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_resid0p04_20260516/report.json
      result: 191/512 -> 193/512, gain 0.00390625, rejected
    seed20260520 residual_scale 0.05:
      report: local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260520_m4_512_20260516/report.json
      result: 191/512 -> 194/512, gain 0.005859375, rejected
    seed19/20 soup alpha 0.5:
      report: local_eval/qwen35_integrated_public_mcq_healing_soup_seed19_20_a050_m4_512_20260516/report.json
      result: 191/512 -> 194/512, gain 0.005859375, rejected

interpretation:
  M4 now has a verified positive core gain on a 512-case balanced public
  MMLU-Pro subset after targeted layer23-open/core-updated public-MCQ healing
  and a residual-scale recheck. The gain is still small but causal on the same
  canonical LM-logit path: the same Qwen3.5 graph with core_off scores 191/512
  while mandatory core_on scores 197/512. This is far below the Qwen3.6-27B
  target band and must not be treated as parity. The remaining bottleneck is
  scaling this from small public-subset gain to stable 1K/full-suite gain while
  reducing health/law regressions and preserving language.

1024-case bottleneck update:
  suite:
    local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_1024.jsonl
  materialization report:
    local_eval/m7_public_reasoning_suite/report_balanced_1024.json
  512-accepted checkpoint, residual_scale 0.06:
    report: local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_1024_resid0p06_20260516/report.json
    result: 378/1024 -> 381/1024, gain 0.0029296875, rejected
  512-accepted checkpoint, residual_scale 0.08:
    report: local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_1024_resid0p08_20260516/report.json
    result: 378/1024 -> 378/1024, gain 0.0, rejected

auxiliary non-test repair:
  materializer:
    scripts/392_materialize_aux_public_mcq.py
  source:
    cais/mmlu dev/validation/auxiliary_train
  policy:
    Do not train or select checkpoints on MMLU-Pro test labels.
  loss addition:
    base-wrong option-margin loss in
    scripts/391_train_qwen35_integrated_public_mcq_healing.py
  targeted aux train:
    local_eval/m7_public_reasoning_suite/mmlu_aux_targeted_validation_train_20260516.jsonl
    401 cases across health/chemistry/economics/law
  targeted aux dev:
    local_eval/m7_public_reasoning_suite/mmlu_aux_targeted_dev_select_20260516.jsonl
    60 cases across health/chemistry/economics/law
  CE-only targeted repair:
    report: local_eval/qwen35_integrated_public_mcq_aux_targeted_from_m3_devselect_s120_20260516/report.json
    aux dev: 33/60 -> 33/60, gain 0.0, rejected
  base-wrong margin targeted repair:
    report: local_eval/qwen35_integrated_public_mcq_aux_targeted_margin_from_m3_devselect_s160_20260516/report.json
    aux dev: 33/60 -> 34/60, gain 0.016666666666666607, accepted
    independent 1024: 378/1024 -> 383/1024, gain 0.0048828125, rejected
    independent report:
      local_eval/qwen35_integrated_public_mcq_aux_targeted_margin_from_m3_m4_1024_20260516/report.json
  broad all-aux blend:
    train file: local_eval/m7_public_reasoning_suite/mmlu_aux_blend_targeted401_all1024_train_20260516.jsonl
    aux dev: 33/60 -> 34/60, gain 0.016666666666666607, accepted
    independent 1024: 378/1024 -> 379/1024, gain 0.0009765625, rejected
    independent report:
      local_eval/qwen35_integrated_public_mcq_aux_blend_margin_from_m3_m4_1024_20260516/report.json
  all-validation/all-dev partial Qwen layer-23 repair:
    report: local_eval/qwen35_integrated_public_mcq_aux_all_margin_l23lr1e7_from_m3_devselect_s120_20260516/report.json
    setting: qwen_lr=1.0e-7, QTRM lr=5.0e-5, base-wrong margin, language KL
    dev result: 169/285 -> 170/285, gain 0.0035087719298245723, rejected
    language: accepted
  category-regression guard implementation:
    scripts/391_train_qwen35_integrated_public_mcq_healing.py
    supports category_gain_summary, regression-penalized checkpoint selection,
    balanced category sampling, and ce_focus=base_wrong
  guarded/base-wrong CE follow-up:
    all-dev balanced: 169/285 -> 171/285, rejected
    targeted balanced: 33/60 -> 33/60, rejected
    targeted base-wrong CE on targeted dev: 33/60 -> 34/60, accepted
    targeted base-wrong CE on MMLU-Pro validation64: 22/64 -> 24/64, accepted
    independent 1024 for both base-wrong CE accepted selection runs:
      378/1024 -> 378/1024, rejected

current M4 decision:
  512-case subset gain remains accepted.
  1024-case gain remains rejected.
  The strongest 1024 result is targeted base-wrong margin:
    +5 hits / 1024
  The acceptance threshold is:
    >= +11 hits / 1024
  Next action:
    Stop residual-scale, tiny-LR partial-Qwen, and tiny selection-set sweeps.
    Use a larger non-test selection pool closer to MMLU-Pro distribution, or
    move to larger-scale language/knowledge healing before retrying 1024/full
    public benchmark gates.

latest external-pool/core-scale follow-up:
  External non-test MCQ pool was materialized from ARC-Challenge, ARC-Easy,
  OpenBookQA, and CommonsenseQA:
    local_eval/m7_public_reasoning_suite/external_mcq_train_pool_2000_20260516.jsonl

  The M3 integrated checkpoint can now warm-start larger adapters with:
    checkpoint_load_mode=skip_mismatch
    core_adapter_dim=512

  Plumbing smoke:
    local_eval/qwen35_integrated_public_mcq_warmstart_ad512_smoke_s2_20260516/report.json
    shape-mismatched adapter tensors skipped: 2
    qtrm parameters: 2111489

  External2000 AD512 selected on MMLU-Pro validation64:
    local_eval/qwen35_integrated_public_mcq_external2000_ad512_basewrongce_margin_from_m3_mmluproval64_s120_20260516/report.json
    22/64 -> 24/64, gain 0.03125, accepted on selection set

  Independent MMLU-Pro 1024:
    local_eval/qwen35_integrated_public_mcq_external2000_ad512_basewrongce_margin_from_m3_mmluproval64_m4_1024_20260516/report.json
    380/1024 -> 377/1024, gain -0.0029296875, rejected

  Conclusion:
    Larger adapter capacity and a non-test external MCQ pool did not solve
    1K transfer. Do not keep optimizing tiny selection sets. The next serious
    promotion attempt needs larger language/knowledge healing or a stronger
    recurrent-core curriculum, then the same independent 1024 gate.

integrated language/knowledge healing follow-up:
  New stage:
    scripts/394_train_qwen35_integrated_language_knowledge_healing.py
    scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

  Purpose:
    Train the same standalone Qwen3.5 -> mandatory QTRM core -> Qwen3.5 LM
    head path on external bilingual/knowledge text and non-test external MCQ,
    with core_off KL to preserve the source model distribution. This is a
    language/knowledge scaffold, not a public benchmark claim.

  Standard run:
    local_eval/qwen35_integrated_language_knowledge_healing_external9000_s120_20260516/report.json
    accepted language/knowledge scaffold
    text rows: 6000
    external MCQ rows: 2000
    language top1 agreement: 0.9166666865348816
    max repeated token run: 1

  Independent MMLU-Pro 1024:
    local_eval/qwen35_integrated_language_knowledge_healing_external9000_m4_1024_20260516/report.json
    380/1024 -> 377/1024
    gain -0.0029296875
    rejected

  Updated conclusion:
    The architecture scaffold can preserve language while training on external
    text, but 120-step shallow healing is still not a public-reasoning upgrade.
    The 1K bottleneck is category-level transfer/regression, not just missing
    plumbing. Continue only with larger validation-controlled knowledge healing
    or stronger recurrent-core curriculum; do not promote this checkpoint.
```

## Causal Core For The LM Path

```text
decision:
  For the Qwen-backbone general LM path, use causal QTRMBlockStack updates.
  Non-causal TRM-style attention remains valid for fully observed puzzle probes,
  but it is not the right default inside an autoregressive next-token LM.

implementation:
  QTRMConfig.core_causal:
    false by default for legacy probes
    true by default in build_qtrm_core_config_from_qwen()

  QTRMRecursiveCore:
    fast_stack = QTRMBlockStack(... causal=cfg.core_causal)
    slow_stack = QTRMBlockStack(... causal=cfg.core_causal)

latest smoke:
  local_eval/qwen_backbone_qtrm_causal_gate_smoke_20260515/report.json

latest metrics:
  core_config.core_causal: true
  max_abs_delta_base_vs_hidden_path_gate0: 0.0
  max_abs_delta_base_vs_core_on: 0.6875
  accepted: true

pretrained-init boundary:
  QTRMBlockStack can be Qwen3.5-style:
    d_model=2048
    n_heads=8
    n_kv_heads=2
    d_ff=6144
    RMSNorm + SwiGLU
    GatedDelta/GQA 3:1 schedule

  But it is not fully Qwen3.5-pretrained by default:
    Qwen's pretrained layers are the main language backbone.
    QTRMBlockStack has z_L/z_H recurrence and repeated fast/slow stacks.
    Full 1:1 Qwen layer weight transplant would require replacing the block
    stack with exact Qwen layer modules or writing a careful partial mapper.

practical rule:
  Preserve full pretrained Qwen in the language backbone.
  Train the causal QTRM core as the new recurrent reasoning residual.
  Only attempt partial Qwen-layer transplant into QTRMBlockStack after causal
  core training shows a reasoning gain without language regression.
```

## Qwen-Layer-Wrapped And Ouro-Style Core Candidates

```text
date:
  2026-05-15

reason:
  The direct QTRMBlockStack path only copies Qwen settings. The stronger bridge
  reuses selected pretrained Qwen decoder layers as frozen recurrent transition
  blocks inside z_L/z_H. This is closer to pretrained initialization than a
  hand-written Qwen-style block.

source prior:
  Ouro model:
    https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
  Ouro paper:
    https://arxiv.org/abs/2510.25741
  Ouro relevance:
    Ouro/LoopLM uses iterative latent computation, recurrent depth, and learned
    depth allocation. Our local variant keeps Qwen3.5 as the language backbone
    and tests whether shared recurrent transition behavior helps the QTRM core.

implemented core_impl values:
  qtrm_block_stack:
    existing hand-written QTRMBlockStack path
  qwen_layer_wrapped:
    fast/slow recurrent updates call selected frozen Qwen decoder layers
  ouro_shared_qwen_layer:
    fast/slow updates share the same wrapped Qwen transition stack

implemented files:
  src/qtrm_mm/qwen_backbone_qtrm.py
  scripts/361_qwen_backbone_qtrm_smoke.py
  tests/test_qwen_backbone_qtrm.py

unit tests:
  PYTHONPATH=src .venv/bin/python -m unittest tests.test_qwen_backbone_qtrm
  OK, 5 tests

smoke A:
  name:
    Qwen-layer-wrapped causal dual-nested core
  report:
    local_eval/qwen_backbone_qtrm_qwen_layer_wrapped_smoke_20260515/report.json
  core_impl:
    qwen_layer_wrapped
  selected Qwen layer:
    3
  core_causal:
    true
  qwen_trainable_parameters:
    0
  qtrm_trainable_parameters:
    12289
  gate=0 equivalence delta:
    0.0
  gate>0 logit delta:
    0.46875
  accepted:
    true

smoke B:
  name:
    Ouro-style shared Qwen-layer loop
  report:
    local_eval/qwen_backbone_qtrm_ouro_shared_qwen_layer_smoke_20260515/report.json
  core_impl:
    ouro_shared_qwen_layer
  shared_stack:
    true
  selected Qwen layer:
    3
  core_causal:
    true
  qwen_trainable_parameters:
    0
  qtrm_trainable_parameters:
    12289
  gate=0 equivalence delta:
    0.0
  gate>0 logit delta:
    0.46875
  accepted:
    true

promotion rule:
  These are now viable architecture candidates, not proven reasoning wins.
  Promote only after a training gate shows:
    1. Qwen gate=0 language behavior remains intact;
    2. core_on beats core_off on a reasoning benchmark;
    3. no repetition/regression on bilingual chat-style prompts.
```

## Actual Ouro-Weight Wrapped Candidate

```text
date:
  2026-05-15

user correction:
  "Ouro-style" shared Qwen-layer recurrence is not the same as using Ouro
  weights. Keep the concepts separate.

implemented distinction:
  ouro_shared_qwen_layer:
    Qwen weights only.
    Shared fast/slow recurrent transition stack.
    Useful as a LoopLM/Ouro-style recurrence probe.

  ouro_weight_wrapped:
    Loads actual ByteDance/Ouro-2.6B-Thinking weights.
    Freezes Ouro and uses selected Ouro decoder layer(s) as the recurrent
    transition block inside the same Qwen final-hidden -> gated core residual
    -> Qwen LM-head causal path.

download path:
  /mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking

source:
  model:
    https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
  paper:
    https://arxiv.org/abs/2510.25741

implemented files:
  src/qtrm_mm/qwen_backbone_qtrm.py
    OuroLayerWrappedStack
    OuroWeightWrappedRecursiveCore
    core_impl=ouro_weight_wrapped
  scripts/361_qwen_backbone_qtrm_smoke.py
    --core-impl ouro_weight_wrapped
    --ouro-model-id /mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking
    --ouro-core-layer-indices 24
  scripts/362_train_qwen_backbone_qtrm_core_gate.py
    same true-Ouro option for short train gates
  scripts/363_run_qwen_backbone_ouro_weight_gate.sh
    one-command actual Ouro smoke + short train gate

verification before real-weight run:
  PYTHONPATH=src .venv/bin/python -m py_compile \
    src/qtrm_mm/qwen_backbone_qtrm.py \
    scripts/361_qwen_backbone_qtrm_smoke.py \
    scripts/362_train_qwen_backbone_qtrm_core_gate.py
  PYTHONPATH=src .venv/bin/python -m unittest \
    tests.test_qwen_backbone_qtrm \
    tests.test_qwen_backbone_qtrm_core_gate_trainer
  result:
    OK, 9 tests

promotion gate after download:
  1. actual Ouro smoke:
     gate=0 must exactly match Qwen logits;
     gate>0 must change logits through the same Qwen LM head.
  2. actual Ouro short train:
     freeze Qwen and Ouro;
     train only QTRM z-state/norm/adapter/gate;
     require core_on > core_off and language top1 agreement >= threshold.
  3. compare against:
     qwen_layer_wrapped
     ouro_shared_qwen_layer
     qtrm_block_stack
```

## Actual Ouro-Weight Results

```text
date:
  2026-05-15

download:
  file:
    /mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking/model.safetensors
  bytes:
    5336011242
  sha256:
    c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080

download method:
  scripts/364_download_ouro_weight_parallel.sh
  resume-safe parallel byte-range downloader.
  Existing prefix/part files are reused; missing ranges only are fetched.

partial true-Ouro smoke:
  report:
    local_eval/qwen_backbone_qtrm_ouro_weight_partial_l18_smoke_20260515/report.json
  core_impl:
    ouro_weight_wrapped
  ouro_core_layer_indices:
    [18]
  gate=0 equivalence delta:
    0.0
  gate>0 logit delta:
    0.5
  accepted:
    true

partial true-Ouro train gate:
  report:
    local_eval/qwen_backbone_qtrm_ouro_weight_partial_l18_train_gate_s80_20260515/report.json
  accepted:
    true
  base/core_off accuracy:
    0.14583333333333334
  core_on accuracy:
    0.20833333333333334
  reasoning gain:
    0.0625
  language top1 agreement:
    1.0

full true-Ouro smoke:
  report:
    local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_smoke_20260515/report.json
  core_impl:
    ouro_weight_wrapped
  ouro_core_layer_indices:
    [24]
  gate=0 equivalence delta:
    0.0
  gate>0 logit delta:
    0.5
  accepted:
    true

full true-Ouro train gate:
  report:
    local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_train_gate_s80_20260515/report.json
  accepted:
    true
  base/core_off accuracy:
    0.14583333333333334
  core_on accuracy:
    0.21875
  reasoning gain:
    0.07291666666666667
  language top1 agreement:
    1.0

comparison:
  qwen_layer_wrapped s80:
    gain 0.0625, language top1 0.75
  ouro_shared_qwen_layer s80:
    gain 0.07291666666666667, language top1 0.75
  ouro_weight_wrapped full l24 s80:
    gain 0.07291666666666667, language top1 1.0

interpretation:
  The actual Ouro-weight transition is now a viable candidate. It does not yet
  prove a large architectural win, but it matches the best short-gate reasoning
  gain and improves the small language non-regression probe versus the earlier
  shared-Qwen-loop candidate.

layer sweep:
  layer 12:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l12_train_gate_s80_20260515/report.json
    decision:
      rejected
    gain:
      0.041666666666666664
    language top1:
      1.0
  layer 24:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_train_gate_s80_20260515/report.json
    decision:
      accepted
    gain:
      0.07291666666666667
    language top1:
      1.0
  layer 36:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l36_train_gate_s80_20260515/report.json
    decision:
      rejected
    gain:
      0.041666666666666664
    language top1:
      0.75
  layers 18,24:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l18_24_train_gate_s80_20260515/report.json
    decision:
      rejected
    gain:
      0.03125
    language top1:
      0.75

current canonical Ouro-weight candidate:
  Use a single frozen Ouro layer 24 transition in `ouro_weight_wrapped`.
  Do not promote multi-layer wrapping yet; in this gate it diluted the gain.
```

## QTRM Core Transition Candidate Naming

```text
date:
  2026-05-15

clarification:
  Qwen wrapping and Ouro wrapping are both QTRM-core designs. The QTRM core is
  not removed in either case. The difference is the frozen transition prior used
  inside the z_L/z_H recurrent update.

common path:
  Qwen3.5 backbone
  -> final hidden states
  -> QTRM recurrent core
  -> gated residual
  -> same Qwen LM head

candidate A:
  name:
    qwen_layer_wrapped
  meaning:
    QTRM recurrent core + frozen Qwen layer 3 transition.
  advantage:
    simplest deployment; one model family; no extra Ouro checkpoint required.

candidate B:
  name:
    ouro_weight_wrapped
  meaning:
    QTRM recurrent core + frozen Ouro layer 24 transition.
  advantage:
    actual looped-LM/Ouro transition prior; strongest language top1 in the
    small gate.

current policy:
  Treat both as QTRM-core transition candidates. Do not describe this as
  "Qwen core vs Ouro core" in a way that implies QTRM is absent. The precise
  comparison is "QTRM core with Qwen transition prior" versus "QTRM core with
  Ouro transition prior".
```

## Eval512 Transition Comparison

```text
date:
  2026-05-15

gate:
  train_cases:
    512
  eval_cases:
    512
  steps:
    200
  seed:
    20260515
  language probe:
    4 prompts, top1 agreement against core_off/Qwen path

candidate A:
  name:
    QTRM core with Qwen transition prior
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/report.json
  accepted:
    true
  base/core_off accuracy:
    0.130859375
  core_on accuracy:
    0.1875
  gain:
    0.056640625
  language top1:
    1.0

candidate B:
  name:
    QTRM core with Ouro layer 24 transition prior
  report:
    local_eval/qwen_backbone_qtrm_ouro_transition_l24_eval512_s200_20260515/report.json
  accepted:
    true
  base/core_off accuracy:
    0.130859375
  core_on accuracy:
    0.185546875
  gain:
    0.0546875
  language top1:
    1.0

decision:
  Both are viable QTRM-core transition candidates. For the default canonical
  path, use Qwen layer wrapping because it is simpler, uses one model family,
  and is slightly stronger on the eval512/200-step gate. Keep Ouro layer 24 as a
  strong alternate transition prior because it is consistently accepted and
  may be useful in later looped-LM/early-exit experiments.
```

## Bilingual General-Language Gate

```text
date:
  2026-05-15

candidate:
  QTRM core with Qwen layer 3 transition prior

checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/last_core.pt

script:
  scripts/367_eval_qwen_backbone_language_gate.py

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_bilingual_generation_gate_20260515/report.json

prompt coverage:
  12 prompts:
    English general explanation / verification / translation / study prompts
    Korean explanation / evidence / greeting / uncertainty / research-note prompts

top-k metrics:
  top1 agreement:
    1.0
  base top1 in core top5:
    1.0
  mean_abs_delta:
    0.35411715507507324

generation metrics:
  generated prompts:
    12
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.9791666666666666

decision:
  accepted

interpretation:
  The eval512/s200 Qwen-transition QTRM checkpoint preserves broad bilingual
  generation behavior under this gate while retaining the accepted reasoning
  gain. This is still a non-regression gate, not proof of improved open-ended
  language ability. It does show that the recurrent core did not recreate the
  earlier repetition-collapse failure.
```

## Long-Generation Language Non-Regression Gate

```text
date:
  2026-05-15

candidate:
  QTRM core with Qwen layer 3 transition prior

checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/last_core.pt

script:
  scripts/367_eval_qwen_backbone_language_gate.py

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_longgen64_gate_20260515/report.json

generation setting:
  max_new_tokens:
    64
  prompts:
    12 English/Korean general prompts

top-k metrics:
  top1 agreement:
    1.0
  base top1 in core top5:
    1.0
  mean_abs_delta:
    0.35411715507507324

generation metrics:
  generated prompts:
    12
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.7513020833333334

decision:
  accepted

interpretation:
  This extends the bilingual/general language non-regression check from short
  generation to 64-token generation. The recurrent core still preserves the
  base Qwen top-token behavior and does not trigger the earlier "Freeze/world
  of..." repetition collapse. This remains a non-regression gate; it does not
  by itself prove that the QTRM core improves open-ended language ability.
```

## Hard Family Gate-Open QTRM Result

```text
date:
  2026-05-15

candidate:
  QTRM core with Qwen layer 3 transition prior, stronger residual path

change:
  core_adapter_dim:
    128
  core_gate_init:
    -2.0
  residual_scale:
    0.5

checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/last_core.pt

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/report.json

gate:
  case_mode:
    hard_v1
  families:
    chain5
    checksum4
    select_pair
  train_cases:
    768
  eval_cases:
    512
  steps:
    300
  acceptance_metric:
    full_vocab

result:
  accepted:
    true
  base/core_off accuracy:
    0.056640625
  core_on accuracy:
    0.125
  gain:
    0.068359375
  learned core gate:
    0.11465207487344742

family floor:
  chain5 gain:
    0.1286549707602339
  checksum4 gain:
    0.05847953216374269
  select_pair gain:
    0.017647058823529405
  min_family_gain:
    0.017647058823529405
  min_family_core_accuracy:
    0.1111111111111111

contrastive failures before this:
  hard_repair_v1 oversampling:
    rejected; select_pair gain stayed negative
  family_loss_weights select_pair=2.0,checksum4=1.5:
    rejected; overall gain dropped to 0.0390625
  qwen layers 3,7 + adapter128 with small gate:
    rejected; select_pair gain stayed negative

interpretation:
  The previous hard_v1 bottleneck was not fixed by data weighting or simply
  adding another frozen transition layer. The decisive change was opening the
  core residual path enough for the recurrent core to causally affect the LM
  logits while keeping Qwen frozen. This is the strongest Qwen-backbone QTRM
  hard-family result so far, but it remains a synthetic reasoning gate, not a
  broad benchmark win over Qwen3.5-2B.
```

## Gate-Open Long-Generation Non-Regression

```text
date:
  2026-05-15

checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/last_core.pt

script:
  scripts/367_eval_qwen_backbone_language_gate.py

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_longgen64_20260515/report.json

generation setting:
  max_new_tokens:
    64
  prompts:
    12 English/Korean general prompts

top-k metrics:
  top1 agreement:
    0.9166666865348816
  base top1 in core top5:
    1.0
  mean_abs_delta:
    0.22324642539024353

generation metrics:
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.7747395833333334

decision:
  accepted

interpretation:
  Opening the QTRM residual path enough to pass hard_v1 does not recreate the
  earlier repetition collapse in 64-token English/Korean generation. Top1
  preservation is slightly weaker than the smaller-gate checkpoint, but top5
  preservation and repetition metrics remain inside threshold.
```

## Qwen-Wrapper Nested Direction

```text
date:
  2026-05-15

decision:
  For the Qwen-backbone bridge path, the next canonical recursive experiment is
  Qwen-wrapper nested recurrence, not Mamba/GatedDelta hybrid nested recurrence.

reason:
  The current QTRM gain comes from a frozen Qwen3.5 backbone plus a Qwen-layer
  transition prior inside the QTRM core. Replacing this transition with
  Mamba/GatedDelta at this stage would confound the experiment: it would test a
  new backbone/mixer family rather than whether the Qwen-derived recurrent core
  can become a stronger TRM-style reasoning path.

canonical nested schedule to test:
  core_impl:
    qwen_layer_wrapped
  qwen_core_layer_indices:
    3
  h_cycles:
    3
  l_cycles:
    6
  outer_steps:
    1 initially

separate concept:
  H=3/L=6 is the nested recurrent thinking schedule. Early exit requires a halt
  head or multi-outer-step controller and should be tested after the nested
  fixed-depth path is stable.

smoke result:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_smoke_s30_20260515/report.json
  accepted:
    true
  base/core_off accuracy:
    0.08333333333333333
  core_on accuracy:
    0.125
  gain:
    0.041666666666666664
  learned core gate:
    0.119376040995121
  language top1:
    1.0

limitation:
  This was only a tiny smoke gate with relaxed family-floor thresholds. It
  proves that Qwen-wrapper nested H3/L6 runs and can produce positive gain, but
  it does not yet prove that nested is better than the current gate-open
  non-nested checkpoint. The next valid comparison is a matched-budget nested
  hard_v1 gate plus language non-regression.
```

## Qwen-Wrapper Nested Matched-Transition Triage

```text
date:
  2026-05-15

script:
  scripts/369_run_qwen_wrapper_nested_compare.sh

purpose:
  Compare the current Qwen-wrapper non-nested path against Qwen-wrapper H3/L6
  nested recurrence while roughly matching core transition calls.

approximate transition accounting:
  non-nested:
    h=1, l=1 -> 2 Qwen-layer transition calls per optimizer step
    steps=210 -> approx 420 core transition calls
  nested:
    h=3, l=6 -> 21 Qwen-layer transition calls per optimizer step
    steps=20 -> approx 420 core transition calls

shared settings:
  model:
    Qwen/Qwen3.5-2B-Base frozen backbone
  core_impl:
    qwen_layer_wrapped
  qwen_core_layer_indices:
    3
  core_adapter_dim:
    128
  core_gate_init:
    -2.0
  case_mode:
    hard_v1
  train/eval cases:
    192 / 192

non-nested result:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_gateopen_nonnested_compare_seed20260515_s210_t420_20260515/report.json
  accepted:
    false
  gain:
    -0.005208333333333333
  min_family_gain:
    -0.015625
  min_family_core_accuracy:
    0.078125
  language_top1:
    1.0

nested H3/L6 result, residual_scale=0.5:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_gateopen_nested_h3_l6_compare_seed20260515_s20_t420_20260515/report.json
  accepted:
    false
  gain:
    -0.041666666666666664
  min_family_gain:
    -0.125
  min_family_core_accuracy:
    0.0625
  failure mode:
    select_pair degraded strongly
  language_top1:
    1.0

nested H3/L6 result, residual_scale=0.1:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_r01_compare_seed20260515_s20_t420_20260515/report.json
  accepted:
    false
  gain:
    0.0
  min_family_gain:
    0.0
  min_family_core_accuracy:
    0.078125
  interpretation:
    Lower residual scale prevents the select_pair collapse but also removes the
    reasoning gain.

decision:
  Qwen-wrapper nested remains the correct canonical nested direction, but fixed
  H3/L6 is not automatically better than the current non-nested gate-open
  bridge. It needs a nested-specific training strategy: gradual depth
  curriculum, residual/gate schedule, periodic family-floor selection, and only
  then halt/early-exit. Do not switch to Mamba/GatedDelta hybrid to hide this
  problem; solve the Qwen-wrapper nested training dynamics first.
```

## Convergence-Based Early Exit For Nested Qwen Wrapper

```text
date:
  2026-05-15

motivation:
  Fixed H3/L6 nested recurrence is expensive and can over-iterate. The next
  canonical direction is adaptive early exit over the same Qwen-wrapper nested
  path, inspired by convergence/fixed-point style recurrence control such as:
    https://arxiv.org/abs/2605.12466

implementation:
  files:
    src/qtrm_mm/config.py
    src/qtrm_mm/qwen_backbone_qtrm.py
    scripts/362_train_qwen_backbone_qtrm_core_gate.py

  config fields:
    core_convergence_halt_enabled
    core_convergence_halt_threshold
    core_convergence_halt_min_outer

  mechanism:
    After each outer nested recurrence block, compute relative z_H state delta:
      rms(z_H_new - z_H_prev) / rms(z_H_prev)
    If every batch item is below threshold after min_outer, stop early.

  telemetry:
    qtrm_core_outer_iterations
    qtrm_core_converged
    qtrm_core_convergence_delta
    mean_core_outer_iterations in eval reports
    core_converged_fraction in eval reports

smoke:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_convergence_halt_telemetry_smoke_s5_20260515/report.json
  schedule:
    h_cycles=3
    l_cycles=6
    outer_steps=3
  convergence threshold:
    0.05
  mean_core_outer_iterations:
    3.0
  core_converged_fraction:
    0.0

interpretation:
  The convergence-halt path is now implemented and instrumented, but threshold
  0.05 does not halt early on this smoke. The next useful experiments are
  threshold sweeps and curriculum training. Do not claim early-exit speedup
  until mean_core_outer_iterations drops below the fixed outer_steps while
  preserving hard_v1 gain and language non-regression.
```

## Convergence Threshold Sweep

```text
date:
  2026-05-15

script:
  scripts/370_sweep_nested_convergence_halt_threshold.py

report:
  local_eval/qwen_backbone_qtrm_nested_h3_l6_convergence_threshold_sweep_20260515/report.json

setting:
  Qwen-wrapper nested H3/L6
  outer_steps:
    3
  residual_scale:
    0.1
  eval_cases:
    96 hard_v1

threshold results:
  threshold 0.02:
    mean_outer_iterations: 3.0
    converged_fraction: 0.0
    gain: 0.0
  threshold 0.05:
    mean_outer_iterations: 3.0
    converged_fraction: 0.0
    gain: 0.0
  threshold 0.1:
    mean_outer_iterations: 3.0
    converged_fraction: 0.0
    gain: 0.0
  threshold 0.2:
    mean_outer_iterations: 3.0
    converged_fraction: 0.3333333333333333
    gain: 0.0
  threshold 0.5:
    mean_outer_iterations: 2.0
    converged_fraction: 1.0
    gain: 0.0
  threshold 1.0:
    mean_outer_iterations: 1.0
    converged_fraction: 1.0
    gain: 0.0

interpretation:
  The convergence threshold controls compute as expected. Threshold 0.5 reduces
  outer iterations from 3 to 2, and threshold 1.0 reduces them to 1. However,
  this sweep used an unpromoted smoke setting and shows no reasoning gain. The
  next valid step is to combine nested curriculum training with threshold
  sweep on a checkpoint that already shows positive hard_v1 gain.
```

## Current M4/M5 Language And Knowledge Healing Decision

```text
date:
  2026-05-16

canonical path:
  Qwen3.5 tokenizer/full vocabulary
  -> Qwen3.5 backbone inside one standalone graph
  -> mandatory QTRM core
  -> Qwen3.5 LM head
  -> autoregressive text

new gate:
  external text CE must not regress
  external validation MCQ must not fall below core_off/base
  category movement is tracked before public MMLU-Pro promotion
  best checkpoint is selected on non-test external validation, not public test

best current validation-controlled run:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_external9000_basewrong_margin_valctrl_s120_20260516/report.json
  decision:
    rejected
  language:
    accepted
  text CE:
    accepted, core_ce_delta 0.000014291144907474518
  external validation MCQ:
    base 192/256
    core 191/256
    gain -0.00390625
  category movement:
    commonsense 0
    science -1

roadmap consequence:
  The Qwen-integrated native scaffold is valid as plumbing, but architecture is
  not complete. Public reasoning promotion is blocked by stable core-side
  knowledge/reasoning transfer. Do not claim Qwen3.5-2B or Qwen3.6-27B
  benchmark improvement from this stage. The next stage must make the
  mandatory core causally improve held-out MCQ/reasoning without damaging
  language, then rerun independent MMLU-Pro 1024.
```

## Cloned-Core Triage 2026-05-16

The Qwen-integrated native path now has a stronger core option:

```text
Qwen3.5 tokenizer/full vocabulary
-> frozen or partially trainable Qwen3.5 backbone
-> mandatory QTRM core containing deep-copied trainable Qwen layer modules
-> Qwen3.5 LM head
-> autoregressive text
```

This is still QTRM-native because the model is a single standalone graph with
`runtime_donor=false`; the cloned layer is inside the mandatory causal core, not
a donor sidecar.

Current result:

```text
clone-core smoke:
  accepted as plumbing

true frozen-Qwen core-only:
  rejected
  validation MCQ base 191/256, core 190/256

layer23 partial-unfreeze + cloned core:
  best current scaffold ties validation MCQ at 191/256 vs 191/256
  language and text CE pass
  positive-gain threshold still rejects

strong preservation KL:
  removes base_correct_core_wrong flips
  but also produces no base_wrong_core_correct flips

strong base_wrong retry correction:
  increases correction pressure
  but introduces preservation regressions before creating held-out corrections

dual-stream retry3 correction/preservation:
  separates correction sampling from preservation KL
  still rejected
  validation MCQ base 191/256, core 190/256
```

Roadmap consequence:

```text
The architecture is not complete; it is a viable Qwen-integrated native
scaffold. The next promotion gate must require positive held-out MCQ gain:
full core > core_off/base, not merely a tie. Do not rerun public MMLU-Pro 1024
until the external validation pool shows reliable base_wrong_core_correct flips
with low or zero base_correct_core_wrong regressions. Loss-weight-only tuning is
not enough so far; the next architecture/training candidate should change where
the core affects the Qwen path or use a stronger non-test verifier/preference
target for base-wrong correction.
```

## Nested H/L Controls Reopened 2026-05-16

Important correction:

```text
The M4/M5 Qwen-integrated healing path had been running with:
  h_cycles=1
  l_cycles=1
  outer_steps=1

That is not the intended TRM-style nested reasoning schedule. It is a shallow
one-pass core residual after the Qwen backbone.
```

Current code now exposes:

```text
N_CORE_LAYERS
H_CYCLES
L_CYCLES
OUTER_STEPS
CORE_CONVERGENCE_HALT_ENABLED
CORE_STEP_CONDITIONING_ENABLED
```

Current canonical defaults:

```text
h_cycles: 3
l_cycles: 6
outer_steps: 3
core_convergence_halt_enabled: true
core_convergence_halt_threshold: 0.2
core_step_conditioning_enabled: true
```

Smoke result:

```text
local_eval/qwen35_integrated_language_knowledge_healing_nested_h3l6_smoke_s2_20260516/report.json

H/L:
  h_cycles=3
  l_cycles=6
  outer_steps=1

result:
  executable
  language preserved
  16-case MCQ tie
```

Default smoke:

```text
local_eval/qwen35_integrated_language_knowledge_healing_nested_default_smoke_s0_20260516/report.json

result:
  default path now reports H=3, L=6, outer=3, convergence halt on, step
  conditioning on.
```

Roadmap consequence:

```text
The correct next experiment is not more residual-scale tuning. It is a
nested-core validation gate, then a core-placement experiment if H/L recurrence
still only ties base. A positive promotion still requires held-out
base_wrong_core_correct flips, not just language preservation or MCQ ties.
```

## Mid-layer Causal Insertion 2026-05-16

Correction:

```text
The final-hidden residual QTRM path was too late in the causal computation:

  Qwen all layers -> QTRM residual delta -> LM head

This can preserve language but has weak leverage over Qwen's answer formation.
The canonical integrated path now supports:

  Qwen prefix layers
  -> mandatory H/L QTRM recurrent core
  -> Qwen suffix layers
  -> LM head
```

Implementation knobs:

```text
CORE_INSERTION_MODE=mid_layer_suffix
CORE_INSERT_AFTER_LAYER=11
```

First smoke:

```text
local_eval/qwen35_integrated_midlayer_suffix_default_smoke_s0_20260516/report.json

result:
  executable
  mean_core_outer_iterations: 2.75
  core_converged_fraction: 0.25
```

First small validation:

```text
local_eval/qwen35_integrated_midlayer_suffix_shared_posgain_s80_20260516/report.json

decision:
  rejected

metrics:
  base_hits: 94 / 128
  core_hits: 93 / 128
  base_wrong_core_correct: 1
  base_correct_core_wrong: 2
```

Interpretation:

```text
This is not a benchmark win. It is, however, the first integrated public-MCQ
run where the QTRM path produced a base-wrong correction flip while running in
the causal Qwen path. The next milestone is not more architecture shopping; it
is making mid-layer correction net-positive by reducing base-correct
regressions.
```

## Strict Integrated Gain 2026-05-16

Canonical accepted scaffold:

```text
Qwen3.5 tokenizer / embeddings
-> Qwen3.5 prefix layers 0..11
-> mandatory QTRM H/L recurrent core
-> adapter-only hidden delta
-> Qwen3.5 suffix layers 12..end
-> Qwen3.5 LM head
```

Accepted report:

```text
local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/report.json
```

Acceptance metrics:

```text
base: 191 / 256
core: 195 / 256
net gain: +4 hits
base_wrong_core_correct: 12
base_correct_core_wrong: 8
commonsense delta: +1
science delta: +3
language top1 agreement: 0.9166667
```

Key lesson:

```text
MCQ correction loss alone creates a global option-letter bias. Language-anchor
KL on ordinary English/Korean prompts is required so that the core learns a
contextual correction rather than a universal "pick option token" shift.
```

Next gates:

```text
1. Reproduce the gain with another seed.
2. Run independent scripts/390 public MCQ eval at 512/1024 cases.
3. Tighten regression: reduce base_correct_core_wrong while preserving the
   positive category deltas.
4. Only then consider partial Qwen healing/unfreeze.
```

Independent 390 recheck:

```text
external commonsense/science 64:
  report: local_eval/qwen35_integrated_midlayer_suffix_langanchor_390_external64_20260516/report.json
  base: 48 / 64
  core: 50 / 64
  gain: +0.03125
  accepted: true

MMLU-Pro 64:
  report: local_eval/qwen35_integrated_midlayer_suffix_langanchor_390_mmlupro64_20260516/report.json
  base: 25 / 64
  core: 24 / 64
  gain: -0.015625
  accepted: false
```

Roadmap update:

```text
Do not overclaim the accepted result as broad benchmark progress. It is a
causal integrated-QTRM gain on the commonsense/science repair distribution.
Broad MMLU-Pro requires a balanced MMLU-Pro repair stage or curriculum before
any 27B-parity claim.
```

## Current MMLU-Pro Bottleneck 2026-05-16

The current canonical integrated path remains:

```text
Qwen3.5 prefix
-> mandatory QTRM H=3/L=6 recurrent core
-> adapter-only hidden delta
-> Qwen3.5 suffix
-> Qwen3.5 LM head
```

Recent repair work found one concrete infrastructure bug and one architecture
bottleneck:

```text
infrastructure bug:
  batched next-token checks were reading right-padding logits in 394/367.
  This is fixed with last_nonpad_logits.

architecture bottleneck:
  the core has useful MMLU flips, but harmful flips are not separable by the
  current hidden token gate after short training.
```

Rejected MMLU repair attempts:

```text
adaptive token gate:
  23 -> 22, gain -0.015625

supervised token gate:
  23 -> 21, gain -0.03125

fast-LR token gate:
  23 -> 21, gain -0.03125

trainable cloned wrapped core layer:
  23 -> 20, gain -0.046875
```

Next principled step:

```text
Add a confidence/arbitration head that decides whether the mandatory core delta
should control the final answer distribution. This should be trained/evaluated
from base/core score geometry and flip accounting, not by blindly increasing
MCQ CE.

The core remains mandatory: it is always computed. The arbitration head is a
no-harm decision layer that prevents useful latent corrections from being
washed out by harmful option flips.
```

## Autoresearch Probe Result 2026-05-16

`karpathy/autoresearch` was cloned locally as an experiment-operations
reference:

```text
references/official/autoresearch
commit: 228791fb499afffb54b46200aca536f79142f117
```

The useful part adopted here is not the nanochat model code. It is the research
operating loop:

```text
fixed small budget
one decisive metric
keep/discard result
ledger before further scaling
```

Implemented QTRM probe:

```text
scripts/395_autoresearch_arbitration_probe.py
scripts/395_run_autoresearch_arbitration_probe.sh
```

MMLU-Pro64 split result:

```text
report:
  local_eval/qwen35_integrated_autoresearch_arbitration_probe_mmlupro64_20260516/report.json

decision:
  rejected_arbitration_probe

fit:
  base 12 / 32
  core 9 / 32
  arbitration 12 / 32

held-out eval:
  base 11 / 32
  core 10 / 32
  arbitration 11 / 32
```

Roadmap consequence:

```text
The current accepted QTRM core does not expose a robust enough option-score
geometry signal for simple arbitration on MMLU-Pro. Keep the external
commonsense/science gain as canonical, but do not promote MMLU repair. The next
fast loop should either:

1. improve the core signal on MMLU-style knowledge/reasoning before arbitration,
2. add richer non-label separability features, or
3. train a learned arbitration head only after a fixed-budget probe shows
   positive held-out arbitration gain.
```

Linear arbitration follow-up:

```text
report:
  local_eval/qwen35_integrated_autoresearch_linear_arbitration_probe_mmlupro64_20260516/report.json

policy:
  linear score-geometry head over margins, confidence, entropy, switch_adv, and
  switch-candidate features

decision:
  rejected_arbitration_probe

held-out eval:
  base 11 / 32
  core 10 / 32
  linear arbitration 11 / 32
  switches 0
```

Consequence:

```text
Do not spend the next loop on more post-hoc arbitration over the same signal.
The fixed-budget autoresearch loop has now rejected both threshold and linear
arbitration. The next candidate mechanism must change the core signal itself:
for example, a small MMLU-style core-signal repair that is judged by the same
keep/discard ledger before any long run.
```

## MCQ Scorer SSOT Correction 2026-05-16

A scorer mismatch was found after the option-only MMLU repair attempt:

```text
old 390 scorer:
  max over acceptable one-token option-letter renderings

394/395 scorer and training loss:
  logsumexp probability mass over acceptable one-token option-letter renderings
```

The max scorer produced a misleading positive read on the option-only
checkpoint. After making 390 use the same probability-mass scorer as 394/395,
the canonical result is:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_optiononly_mmlupro64_ssot_eval_20260516/report.json

decision:
  rejected_m4_public_mcq_core_gain

base:
  26 / 64

core:
  25 / 64

gain:
  -0.015625
```

Roadmap consequence:

```text
The current option-only repair is discarded. Future public-MCQ claims must use
the shared option-letter probability-mass scorer and non-pad final-token
gathering before any architecture or training conclusion is accepted.
```

Additional infrastructure correction:

```text
scripts/391_train_qwen35_integrated_public_mcq_healing.py also used
logits[:, -1, :] on padded batches. It now gathers the final real token using
attention_mask.sum(dim=1)-1.
```

Preservation repair result:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_optiononly_preserve_repair_s80_20260516/report.json

decision:
  rejected_integrated_language_knowledge_healing

best checkpoint:
  step 0

base/core:
  23 / 21

gain:
  -0.03125
```

## SSOT Revalidation Consequence 2026-05-16

The post-fix revalidation runner is:

```text
scripts/396_run_qwen35_integrated_ssot_revalidation.sh
```

Canonical ledger:

```text
local_eval/qwen35_integrated_ssot_revalidation_20260516/summary.jsonl
```

Accepted after SSOT:

```text
midlayer_external64:
  base/core: 49 / 50
  gain: +0.015625
```

Rejected after SSOT:

```text
midlayer_mmlupro64:
  base/core: 26 / 23
  gain: -0.046875

optiononly_mmlupro64:
  base/core: 26 / 25
  gain: -0.015625

public_coreonly_mmlu256:
  base/core: 93 / 93
  gain: 0
```

MMLU-Pro64 scale sweep:

```text
local_eval/qwen35_integrated_ssot_scale_sweep_mmlupro64_20260516/summary.jsonl

scale 0.00: 26 / 26, gain 0
scale 0.25: 26 / 23, gain -0.046875
scale 0.50: 26 / 23, gain -0.046875
scale 0.75: 26 / 22, gain -0.0625
scale 1.00: 26 / 23, gain -0.046875
```

Architecture consequence:

```text
Do not continue by increasing residual strength or adding post-hoc arbitration.
For broad MMLU-style reasoning, the current QTRM residual is anti-causal:
turning it on makes the model worse. The next candidate must change what the
core learns or how it is integrated, then pass this same SSOT revalidation.
```

## M7A Native Answer-Surface Gate 2026-05-16

Result:

```text
accepted gate:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/m7a_gate_report.json

checkpoint:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt

strict public-style MCQ slice:
  64 held-out MMLU-Pro balanced-test cases

metrics:
  invalid_pred_rate: 0.0
  prompt_echo_rate: 0.0
  max_pred_fraction: 0.234375
  accuracy: 0.140625
```

Architecture/training lesson:

```text
For native public MCQ work, final-token answer supervision is required before
benchmark claims. Whole-sequence CE over MCQ prompts/options teaches the model
to reproduce option-line syntax and can create A. loops. A-D-only auxiliary
data is also insufficient for MMLU-Pro's A-J answer space.

The accepted M7A path uses:
  full Qwen tokenizer/vocab native checkpoint
  MMLU-Pro validation A-J labels
  final answer token CE/margin on the canonical " X" token
  strict greedy-generation gate
```

Next required promotion:

```text
M7B:
  improve correctness above the M7A surface baseline while preserving:
    invalid_pred_rate <= 0.05
    prompt_echo_rate <= 0.05
    max_pred_fraction <= 0.60
  and add core-depth/core-off ablations on the same public-style eval.
```

## M7B Native Core-Depth Gate 2026-05-16

Accepted:

```text
report:
  local_eval/qtrm_native_m7b_core_depth_gate_m7a_s300_20260516/m7b_gate_report.json

runner:
  scripts/403_run_qtrm_native_m7b_core_depth_gate.sh

scorer:
  scripts/402_score_m7b_core_depth_gate.py
```

Result:

```text
depth0:
  accuracy: 0.09375
  max_pred_fraction: 1.0

depth1:
  accuracy: 0.109375

depth2:
  accuracy: 0.109375

depth4:
  accuracy: 0.140625
  invalid_pred_rate: 0.0
  prompt_echo_rate: 0.0
  max_pred_fraction: 0.234375

gain_vs_baseline:
  0.046875

gain_vs_best_shallow:
  0.03125
```

Consequence:

```text
The native recursive core has now produced a small public-style MCQ strict
generation gain through the normal LM answer path. This does not solve public
benchmark parity. It only establishes that the next stage can optimize
correctness without treating answer formatting and core-depth causality as
unproven.

M7C should keep scripts/398 and 402 as hard gates while increasing accuracy on
larger public-style slices.
```

## M7C Rejected Path 2026-05-16

Rejected promotion strategy:

```text
Take the accepted M7A/M7B checkpoint and continue training final option-token
CE on more MCQ rows.
```

Why rejected:

```text
The path regressed correctness below the M7A/M7B baseline even when guarded by:
  base checkpoint option-KL preservation
  low learning rate / short runs
  think/core-only trainable parameter filtering
  natural external MCQ data
  A-J remapped auxiliary data
  depth-gain trajectory loss
  deeper recurrent execution at depth6/8/12
```

Representative reports:

```text
local_eval/qtrm_native_m7c_aj_remap_s300_20260516/report.json
local_eval/qtrm_native_m7c_aj_remap_s300_m7b_20260516/m7b_gate_report.json
local_eval/qtrm_native_m7c_external_preservekl_s100_lr1e5_20260516/report.json
local_eval/qtrm_native_m7c_coreonly_mmluaux_preservekl_s150_20260516/report.json
local_eval/qtrm_native_m7c_coreonly_ajremap_preservekl_s60_20260516/report.json
local_eval/qtrm_native_m7c_depthgain_coreonly_val64_s100_20260516/report.json
local_eval/qtrm_native_m7c_depth_sweep_m7a_20260516/m7b_gate_report.json
```

Canonical consequence:

```text
M7C must stop treating public-MCQ correctness as a final-token-only surface
problem. M7A solved the surface. M7B proved a small core-depth effect. The next
architecture/training candidate must strengthen the recursive core trajectory
itself and then pass the same strict LM-generation gate.

The current checkpoint should stay at depth4 for public MCQ evaluation. Higher
depth is not automatically better and currently acts like overthinking/noise
accumulation.
```

## M7D Rejected Path 2026-05-16

Rejected strategy:

```text
Build non-test MCQ answer-content language data, continue language bootstrap,
then re-open the answer-letter surface.
```

Outcome:

```text
knowledge-only CE:
  rejected, repeated-word language degradation

preservation-mix low-LR knowledge CE:
  non-degenerate language sample, but did not improve strict MCQ

answer repair after knowledge mix:
  next-token diagnostic recovered to 9 / 64
  strict generation collapsed to 2 / 64 with many empty outputs
```

Canonical consequence:

```text
The current 64M QTRM-native checkpoint cannot be promoted by small CE patches
on MCQ-derived language data. A larger native language/knowledge pretraining
stage or a stronger integrated pretrained backbone is required before public
MMLU-style correctness can move materially. Until then, keep the accepted
M7A/M7B checkpoint as canonical and treat M7C/M7D attempts as rejected probes.
```

Next candidate shape:

```text
prompt tokens
-> native embeddings/backbone
-> mandatory recursive core
-> trajectory objective/curriculum on core states
-> LM logits
-> strict answer generation

Acceptance:
  M7A surface stays valid
  M7B depth4 > depth0/depth1/depth2 still holds
  strict correctness exceeds 9 / 64
```

## Qwen-Integrated M4 SSOT Revalidation 2026-05-16

The Qwen-integrated path remains a native-candidate only if it is one
standalone graph:

```text
Qwen tokenizer/backbone
-> mandatory QTRM core inside the same hidden/logit path
-> Qwen LM head
-> autoregressive text
```

However, the old public-MCQ accepted-looking reports are no longer canonical.
Standalone SSOT reruns did not reproduce the required public benchmark gain:

```text
256-case core-only rerun:
  local_eval/qwen35_integrated_repro_public_coreonly_mmlu256_rerun2_20260516/report.json
  core_off 93 / 256
  core_on  93 / 256
  gain 0.0
  rejected

256-case l23open seed20260520 rerun:
  local_eval/qwen35_integrated_repro_l23open_seed20260520_mmlu256_20260516/report.json
  core_off 93 / 256
  core_on  94 / 256
  gain 0.00390625
  rejected

512-case residual_scale=0.06 rerun:
  local_eval/qwen35_integrated_repro_public_coreonly_mmlu512_resid0p06_20260516/report.json
  core_off 194 / 512
  core_on  196 / 512
  gain 0.00390625
  rejected
```

Decision:

```text
Qwen-integrated residual tuning is not current architecture evidence for
public MMLU-style reasoning. It can remain a language/backbone bridge and a
diagnostic baseline, but promotion requires standalone rerun evidence where
core_on beats core_off beyond the configured threshold and the same checkpoint
passes strict generation or explicit LM-logit causality checks.
```

## DGX Qwen3.6 Proxy And M7B Scale-Out 2026-05-17

DGX now has a reproducible llama-server proxy for Qwen3.6-27B MTP GGUF:

```text
server:
  /mnt/data4tb/llama-cpp-turboquant-cuda/build/bin/llama-server

model:
  /mnt/data4tb/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf

endpoint:
  http://192.168.219.113:18082/v1

scripts:
  local control: scripts/407_dgx_qwen36_mtp_llama_server.sh
  DGX direct: /mnt/data4tb/qwen36-mtp-llama-server.sh
```

M6 scoped raw-reasoning baseline rerun:

```text
Qwen3.6-27B-MTP-GGUF DGX llama-server:
  64 cases:  7 / 64  = 0.109375
  256 cases: 35 / 256 = 0.13671875
  512 cases: 75 / 512 = 0.146484375

accepted QTRM-native L5 baseline:
  0.6067708333333334 over 768 cases
  core_gain: 0.5859375
  ablation_drop: 0.5716145833333334
  min_family_generation_exact: 0.4140625
```

M6 consequence:

```text
The scoped synthetic raw-reasoning win remains accepted against the DGX
Qwen3.6-MTP-GGUF proxy. This is still a narrow custom-suite result, not public
benchmark parity.
```

M7B public MCQ scale-out:

```text
64-case DGX rerun:
  depth0: 6 / 64 = 0.09375
  depth4: 9 / 64 = 0.140625
  decision: accepted

256-case DGX rerun:
  depth0: 39 / 256 = 0.15234375
  depth4: 24 / 256 = 0.09375
  decision: rejected
```

Architecture consequence:

```text
Do not promote the current M7B public-MCQ core-depth signal beyond a small
diagnostic slice. The recursive core shows a real 64-case effect, but the same
checkpoint loses to depth0 at 256 cases. The next architecture/training step is
core-depth scale-out repair, not more answer-token CE or benchmark claims.
```

## HRM-Text Prior Update 2026-05-19

HRM-Text is now a first-class text-LM prior for QTRM-native:

```text
source:
  https://huggingface.co/sapientinc/HRM-Text-1B
  https://github.com/sapientinc/HRM-Text

local clone:
  references/official/hrm-text@f99410a
```

Architectural correction:

```text
TRM:
  dual latent states z_H / z_L
  shared recurrent update block

HRM:
  dual latent states z_H / z_L
  separate H_module / L_module

HRM-Text:
  HRM-style dual-module recurrence ported to PrefixLM text pretraining
```

QTRM-native consequence:

```text
Do not describe TRM as single-state. The open question is not z_H/z_L versus
one state; the open question is shared recurrent block versus separate H/L
modules under the same QTRM-native LM-generation gate.
```

Next canonical comparison:

```text
candidate A: TRM-style shared-block z_H/z_L core
candidate B: HRM-style separate H/L core
candidate C: current best nested QTRM baseline

All candidates must use:
  same tokenizer/data/eval seed
  same H_cycles x L_cycles budget
  same greedy LM-generation metric
  same think0/state_reset/z_l_zero/z_h_zero/op_zero destructive ablations
```

Do not jump from HRM-Text to a capability claim. HRM-Text's strong numbers came
from real text pretraining; QTRM still has to prove the recurrent core first,
then run language healing/pretraining without erasing that causal gain.

## Qwen3.5 Pretrained-Init Strict TRM Gate 2026-05-19

Canonical QTRM-native pretrained-init path:

```text
prompt/chat text
-> Qwen3.5 tokenizer + token embeddings
-> Qwen3.5 original backbone
-> mandatory shared z_H/z_L TRM-style recurrent core
-> Qwen3.5 LM head
-> AR text/logits
```

Implementation status:

```text
core_impl:
  qwen_shared_layer_wrapped

meaning:
  reuse a selected original Qwen3.5 layer as the shared recurrent update block
  keep the core mandatory
  train the QTRM core first
  optionally unfreeze the matching Qwen layer with a lower LR for healing
```

DGX short-gate results:

```text
frozen Qwen, core-only S80:
  accepted: false
  gain: +0.0039
  min_family_core_accuracy: 0.0465
  language_top1_agreement: 1.0

partial layer-3 unfreeze S80:
  accepted: false
  gain: +0.0234
  accepted_reasoning_gain: true
  min_family_core_accuracy: 0.0465
  language_top1_agreement: 1.0

checksum repair, layer-3 still partial, S80:
  accepted: false
  gain: +0.0156
  accepted_family_core_accuracy: true
  min_family_core_accuracy: 0.1395
  language_top1_agreement: 1.0
```

Interpretation:

```text
The pretrained-init path runs and preserves the Qwen language surface on the
small gate. Partial unfreeze of the Qwen layer makes the recurrent core produce
a measurable reasoning gain, but the gain and family-floor criteria have not
yet been satisfied at the same checkpoint.

This is not a Qwen3.6-27B-beating model. It is the first executable
Qwen3.5-pretrained, QTRM-native, mandatory-core gate with non-regressed
language logits.
```

Next action:

```text
Do not change the whole architecture yet.
Target the measured bottleneck:
  core gain and family floor are currently separable.

Next candidate should make the core route family-balanced without allowing the
base Qwen path to absorb the same improvement:
  1. keep Qwen layer-3 partial healing at low LR
  2. add balanced family/DRO selection on the same acceptance metric
  3. keep language KL/top1 non-regression
  4. promote only when gain >= 0.02 and min_family_core_accuracy >= 0.08 on the
     same checkpoint
```

Mid-layer suffix note:

```text
final_residual:
  safer language preservation
  weaker causal influence because the core only perturbs the final hidden state

mid_layer_suffix:
  stronger causal route because the core perturbs an intermediate Qwen hidden
  state and the remaining Qwen layers reinterpret it
  but unsafe at residual_scale=0.05 on first smoke
```

DGX smoke:

```text
mid_layer_suffix, insert_after_layer=3, residual_scale=0.05:
  language_top1_agreement: 0.375 -> rejected

mid_layer_suffix, insert_after_layer=3, residual_scale=0.005:
  language_top1_agreement: 0.75 -> barely passes the language threshold
  reasoning_gain: 0.0 on the 16-case smoke
```

Consequence:

```text
Do not promote mid_layer_suffix yet. It needs a residual/gate warmup schedule
or token_mlp gate training before it can become the default QTRM-native
pretrained-init route.
```

## Causal Advantage And Interpolation Breakthrough 2026-05-19

Measured bottleneck:

```text
partial layer-3 healing:
  passes aggregate core gain
  misses family floor

checksum repair:
  passes family floor
  misses aggregate core gain
```

Added objective:

```text
core_advantage_loss:
  compare core_on logits against force_core_off logits
  require the core path to improve the correct answer margin

modes:
  target_logp
  label_choice_margin
```

The direct advantage continuations did not pass the gate. The useful discovery
was that the two rejected checkpoints are complementary in weight space.

Checkpoint interpolation:

```text
A:
  local_eval/qwen35_preinit_strict_trm_partial_l3_gate_s80_20260519/last_core.pt

B:
  local_eval/qwen35_preinit_strict_trm_partial_l3_checksum_repair_s80_20260519/last_core.pt

alpha:
  interpolated = (1 - alpha) * A + alpha * B
```

Best alpha:

```text
alpha=0.25
128-case gate:
  accepted: true
  gain: +0.0390625
  min_family_gain: 0.0
  min_family_core_accuracy: 0.1395348837
  language_top1_agreement: 1.0

256-case gate:
  accepted: false
  gain: +0.01953125
  min_family_gain: 0.0
  min_family_core_accuracy: 0.0930232558
  language_top1_agreement: 1.0
```

Interpretation:

```text
This is the first Qwen3.5-pretrained QTRM-native checkpoint composition that
passes the small strict mandatory-core gate while preserving language logits.
It is not yet a public benchmark or Qwen3.6-27B result. It is a real
architecture-training signal: reasoning-gain and family-floor capabilities can
coexist in the same weight basin, and interpolation can expose the overlap.
```

Next promotion requirement:

```text
Turn the alpha=0.25 interpolation into a trained/stable checkpoint:
  1. rerun independent seeds
  2. pass 256-case by margin, not near-miss
  3. run core destructive ablations
  4. only then attach public MCQ/language healing gates
```

## Qwen3.5 Preinit Interpolation Stabilization Attempt 2026-05-19

Decision record:

```text
docs/wiki/decisions/qwen35-preinit-interpolation-stabilization.md
```

Result:

```text
Direct continuation from alpha=0.25:
  rejected
  gain: 0.015625
  language_top1_agreement: 1.0

Best scalar/selective interpolation on 256 cases:
  q0.25_c0.30 / q0.25_c0.32 / q0.25_c0.35
  rejected
  gain: 0.01953125
  language_top1_agreement: 1.0

Group interpolation:
  qa25_qm25_qn25_cs50_ca25
  128-case accepted
  256-case rejected, gain: 0.01953125
```

Interpretation:

```text
The 128-case accepted basin is real but fragile. Coarse interpolation and
low-LR continuation do not stabilize it. The next credible step is not more
architecture shopping or alpha hunting. It is a fixed 256-case family-balanced
training/selection loop that optimizes the actual promotion gate while
preserving language non-regression.
```

Follow-up:

```text
scripts/412_run_qwen35_preinit_family_balanced_selection.sh
local_eval/qwen35_preinit_family_balanced_select_s120_20260519

result:
  rejected
  best_periodic_gain: 0.01953125
  final_gain: 0.01953125
  language_top1_agreement: 1.0

consequence:
  selection loop works, but objective still cannot move checksum4. Next target
  is checksum-specific counterfactual/recurrent trajectory supervision.
```

Checksum counterfactual follow-up:

```text
scripts/362_train_qwen_backbone_qtrm_core_gate.py
  --checksum-counterfactual-weight
  --checksum-counterfactual-variants

run:
  local_eval/qwen35_preinit_checksum_cf_w05_v2_s80_20260519

256-case:
  accepted: true
  gain: 0.0234375
  language_top1_agreement: 1.0

512-case:
  accepted: false
  gain: 0.015625
  language_top1_agreement: 1.0

remaining bottleneck:
  checksum4 gain remains 0.0, so the next objective must target checksum4
  base-error cases directly. The 256-case checkpoint is a useful candidate, not
  a robust promotion.
```

Base-error targeted follow-up:

```text
run:
  local_eval/qwen35_preinit_checksum_baseerr_w06_s80_20260519

256-case:
  accepted: true
  gain: 0.0234375
  language_top1_agreement: 1.0

family gains:
  chain5:      +0.0352941176
  checksum4:   0.0
  select_pair: +0.0352941176

consequence:
  final-token answer-margin pressure is not enough for checksum4. Next step is
  a checksum4 diagnostic/probe: record base/core digit predictions, target
  ranks, logit margins, and z_H/z_L operand-binding evidence.
```

Checksum latent probe result:

```text
script:
  scripts/413_probe_qwen35_preinit_checksum_latents.py

outputs:
  local_eval/qwen35_preinit_checksum_probe_alpha025_20260519
  local_eval/qwen35_preinit_checksum_probe_cf_w05_v2_20260519
  local_eval/qwen35_preinit_checksum_probe_baseerr_w06_20260519

shared checksum4 eval:
  base_accuracy: 0.0930232555
  core_accuracy: 0.0930232555
  core_fixes_base_errors: 0

latent finding:
  z_h contains operand information, but not the composed checksum answer.

next:
  add latent answer-composition supervision before more promotion runs.
```

Latent answer auxiliary result:

```text
implemented:
  qtrm_core_hidden / qtrm_core_delta outputs
  --checksum-latent-answer-weight
  --checksum-latent-answer-source z_h|delta_h

runs:
  local_eval/qwen35_preinit_latent_answer_zh_w05_s100_20260519
  local_eval/qwen35_preinit_latent_answer_delta_w05_s80_20260519

256-case:
  accepted: true
  gain: 0.0234375
  language_top1_agreement: 1.0
  checksum4 gain: 0.0

consequence:
  final latent answer auxiliary is rejected as a checksum4 fix. Next HRM-Text
  adaptation must supervise recurrent trajectory/residue states.
```
