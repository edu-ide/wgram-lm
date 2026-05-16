# QTRM-Native Hard Lock

Date: 2026-05-14

Status: canonical research constraint.

## 2026-05-16 Addendum: TRM Paper Condition Lock

QTRM-native is not just "donorless" and not just "a recurrent block exists".
The architecture claim is locked to a loop-reasoning model that satisfies the
spirit of the TRM paper:

```text
prompt tokens
-> tokenizer
-> native embeddings
-> native backbone
-> mandatory recursive latent state loop
   z_L / z_H or an equivalent low/high recurrent state
   repeated state transition before answer readout
   loop depth or halt policy is part of the model's computation
-> core-dependent readout
-> LM logits
-> autoregressive answer
```

The recursive state must be the reason the answer improves. It is not enough
for the loop to run in parallel while an ordinary LM path, a residual adapter,
or a sidecar solver carries the result.

Hard reject as canonical:

```text
Qwen donor path answers and QTRM only nudges logits
QTRM is an optional residual that can be scaled to zero without losing gain
MemoryOS/RAG/verifier/tooling computes the answer outside the model
candidate scoring or forced-choice improves but greedy LM generation does not
core_off, think0, z_L_zero, z_H_zero, or state_reset keeps the same result
```

Minimum promotion evidence:

```text
full > no-loop or shallow-loop baseline
full > core_off / think0
deeper loop > shallower loop on held-out cases
state reset/corruption damages the same answer metric
readout-off damages the same answer metric
output remains ordinary LM logits and autoregressive text
```

This addendum overrides older residual/donor-preservation language in the
wiki when discussing the canonical architecture. Donor, Qwen, Ouro, MemoryOS,
MSA, RAG, and verifier work remain allowed as scaffolds or later extensions,
but they cannot be the proof of QTRM-native loop reasoning.

## Decision

QTRM architecture research is locked to the QTRM-native path.

Canonical model path:

```text
chat-template / prompt tokens
-> tokenizer
-> native token embeddings
-> native encoder/backbone
-> mandatory TRM/QTRM recursive thinking core
-> native decoder/readout
-> LM head
-> autoregressive text
```

The donor/residual/Qwen-sidecar path is no longer a main architecture route.
It may be used only as a diagnostic scaffold, teacher/source of comparison, or
temporary data-generation/probing tool.

## Why

The project goal is not merely a useful Qwen adapter. The goal is a new
native recursive LM architecture whose reasoning loop is part of the model
itself.

The donor/residual path can be useful, but it weakens the core claim:

```text
donor language ability
-> sidecar/controller/residual
-> apparent reasoning improvement
```

That does not prove a native LM can learn language and reasoning through its
own recurrent core.

## What Is Canonical

Canonical progress must satisfy all of the following:

```text
donor disabled
no MemoryOS/RAG hidden answer path
no side renderer or symbolic solver carrying the answer
normal token IDs and token embeddings
mandatory native recursive core
normal LM logits and autoregressive generation
held-out metric improves
core/depth/state ablations reduce the same metric
```

## What Is Diagnostic Only

These remain allowed only as probes:

```text
Qwen donor hidden states
Qwen-Scope interpretability
donor/residual adapters
teacher datasets or teacher logits
MemoryOS/RAG retrieval
external verifiers
typed executors
answer renderers
candidate-list or forced-choice heads
```

None of these can be used to claim QTRM-native architecture progress unless
the same mechanism is reproduced inside the donorless native token-to-logit
path.

## New Research Order

The immediate plan is reset to native tiny-LM-first:

```text
1. Tiny native LM viability:
   next-token loss decreases on tiny text and greedy generation is not
   repeated-token collapse.

2. Native recurrent non-regression:
   recurrence must not damage basic LM loss versus think0 or core-off.

3. Native recursive causality:
   deeper/core-on path must beat shallow/core-off on the same LM metric.

4. Native synthetic reasoning:
   only after language viability, reintroduce small reasoning gates.

5. Larger native pretraining:
   only after the tiny-LM and recursive-causality gates pass.
```

Canonical milestone dependency:

```text
M-A. Recursive core improves reasoning:
     first prove the loop core itself creates a held-out reasoning gain.
     Required ablations: core_off/think0, shallow depth, state reset/corrupt.

M-B. Core is attached to the LM path:
     the proven core must feed the answer-producing hidden path and LM logits.
     No separate answer channel, candidate solver, or residual-only shortcut.

M-C. Language healing:
     after M-A and M-B, heal language with corpus training, pretrained init,
     partial unfreeze, or distillation-style artifacts while preserving the
     same core-causal ablation gains.
```

## Canonical Stage Goals 2026-05-14

The active roadmap is:

```text
0. Principle lock:
   QTRM-native only. Donor, Qwen sidecars, RAG, MemoryOS, MSA, and runtime
   verifiers are diagnostic or later-stage tools, not canonical progress.

1. Native LM viability:
   donorless token->native embedding->core->LM logits learns non-degenerate
   next-token text.

2. Core depth causality:
   deeper recurrent/core steps beat shallow/core-off on the same LM metric.

3. Language non-regression:
   mandatory recurrence must not destroy ordinary text loss or greedy
   generation.

4. Native mixed text reasoning:
   normal text-form prompt and answer must improve through the same
   donorless recurrent LM path.

5. Seed stability:
   accepted reasoning gains must survive multiple seeds, not one lucky run.

6. L5 multi-family reasoning:
   the same path must work across modchain, revchain, checksum, and related
   families.

7. Language expansion:
   move from tiny char-level probes to broader corpus, larger tokenizer, and
   longer context while keeping recurrence causal.

8. Later extensions:
   only after native language+reasoning acceptance, add memory, MSA/LM2,
   metacognition, agent loops, and larger pretraining.
```

Current bottleneck:

```text
L4 native mixed text reasoning:
  latest single recurrent standard run reached 0.6640625 exact
  threshold is 0.70
  depth/core ablations are strongly causal
  next work is to close the 0.664 -> 0.70 gap without changing the canonical
  path or adding external shortcuts
```

## Slow-Progress Rule

If progress is slow, do not fall back to donor/residual QTRM as the canonical
answer.

Instead:

```text
shrink the native gate;
run triage-first;
reduce model/data size;
fix tokenization/decoder/loss/recurrence;
preserve rejected results in the wiki;
promote only after donorless native ablations pass.
```

## Current Implication

The previous dual_path_reverse len8 work is preserved as research evidence,
but it is no longer the next immediate promotion target.

The next active target is:

```text
QTRM-native tiny LM viability before synthetic recursive reasoning promotion.
```

Default runner:

```bash
bash scripts/300_run_research_gate.sh
```

is now equivalent to:

```bash
bash scripts/300_run_research_gate.sh qtrm_native_tiny_lm_first standard
```

Fast triage:

```bash
PROFILE=triage bash scripts/300_run_research_gate.sh
```

Gate meaning:

```text
qtrm_native_tiny_lm_first:
  donorless tiny text LM probe
  QTRM-native token embeddings
  mandatory recurrent path active during training/eval
  normal LM logits
  reject on random-like loss, repeated-token collapse, or recurrent path
  non-regression failure versus think0/core-off
```

## Runner Result 2026-05-14T19:36:41

```text
gate: qtrm_native_tiny_lm_first
target_level: L1 native tiny LM first
profile: triage
decision: accepted_native_tiny_lm_first
accepted: True
next_action: continue native-only: add a small recursive-depth LM ablation or proceed to qtrm_native_l3_language_slice; do not switch to donor/residual QTRM
```

Decisive metrics:

```json
{
  "last_loss": 1.6067042350769043,
  "eval_metrics.think_eval_loss": 1.5494265830379792,
  "eval_metrics.think0_loss": 3.4571088374346153,
  "eval_metrics.thinking_block_off_loss": 3.4571088374346153,
  "eval_metrics.loss_ratios.full_vs_think0": 0.44818565335876087,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.44818565335876087,
  "eval_metrics.sample_degeneracy.unique_chars": 19.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.022727272727272728,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_tiny_lm_first_triage/report.json`

## Runner Result 2026-05-14T19:38:14

```text
gate: qtrm_native_tiny_lm_first
target_level: L1 native tiny LM first
profile: standard
decision: accepted_native_tiny_lm_first
accepted: True
next_action: continue native-only: add a small recursive-depth LM ablation or proceed to qtrm_native_l3_language_slice; do not switch to donor/residual QTRM
```

Decisive metrics:

```json
{
  "last_loss": 0.04423641040921211,
  "eval_metrics.think_eval_loss": 0.043676364574242725,
  "eval_metrics.think0_loss": 3.7994057698683306,
  "eval_metrics.thinking_block_off_loss": 3.7994057698683306,
  "eval_metrics.loss_ratios.full_vs_think0": 0.011495577787617126,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.011495577787617126,
  "eval_metrics.sample_degeneracy.unique_chars": 30.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.015151515151515152
}
```

Interpretation:

```text
The native tiny LM path is viable at standard budget. The recurrent path is
not a cosmetic no-op on this tiny text slice: think0/core-off loss is much
higher than the active recurrent path. The next native-only gate is an explicit
depth sweep, not synthetic reasoning yet.
```

New gate:

```text
qtrm_native_tiny_lm_depth_ablation:
  same donorless native text probe
  evaluates depth sweep 0,1,2,4
  rejects if full recurrent depth regresses versus the best shallow depth
```

Example:

```bash
bash scripts/300_run_research_gate.sh qtrm_native_tiny_lm_depth_ablation standard
```

## Runner Result 2026-05-14T19:38:14

```text
gate: qtrm_native_tiny_lm_first
target_level: L1 native tiny LM first
profile: standard
decision: accepted_native_tiny_lm_first
accepted: True
next_action: continue native-only: add a small recursive-depth LM ablation or proceed to qtrm_native_l3_language_slice; do not switch to donor/residual QTRM
```

Decisive metrics:

```json
{
  "last_loss": 0.04423641040921211,
  "eval_metrics.think_eval_loss": 0.043676364574242725,
  "eval_metrics.think0_loss": 3.7994057698683306,
  "eval_metrics.thinking_block_off_loss": 3.7994057698683306,
  "eval_metrics.loss_ratios.full_vs_think0": 0.011495577787617126,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.011495577787617126,
  "eval_metrics.sample_degeneracy.unique_chars": 30.0,
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

Report: `local_eval/research_gate_runner/qtrm_native_tiny_lm_first_standard/report.json`

## Runner Result 2026-05-14T19:41:46

```text
gate: qtrm_native_tiny_lm_depth_ablation
target_level: L2 native tiny LM depth ablation
profile: standard
decision: accepted_native_tiny_lm_depth_ablation
accepted: True
next_action: continue native-only: broaden the text slice or reintroduce small synthetic reasoning while preserving the same native token->core->logits path
```

Decisive metrics:

```json
{
  "last_loss": 0.04423641040921211,
  "eval_metrics.think_eval_loss": 0.043676364574242725,
  "eval_metrics.think0_loss": 3.7994057698683306,
  "eval_metrics.thinking_block_off_loss": 3.7994057698683306,
  "eval_metrics.loss_ratios.full_vs_think0": 0.011495577787617126,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.011495577787617126,
  "eval_metrics.loss_ratios.full_vs_best_shallow_depth": 0.1457017641560008,
  "eval_metrics.best_shallow_depth_loss": 0.29976551641117444,
  "eval_metrics.sample_degeneracy.unique_chars": 30.0,
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

Report: `local_eval/research_gate_runner/qtrm_native_tiny_lm_depth_ablation_standard/report.json`

## Runner Result 2026-05-14T19:45:09

This is not a general-LLM acceptance. It is a broader native language
non-regression triage result: donorless QTRM-native recurrence still improves
the same next-token loss over core-off while staying close to a no-recurrence
baseline on a wiki-text slice.

```text
gate: qtrm_native_l5_language_nonregression
target_level: L5C native language non-regression
profile: triage
decision: accepted_l5_language_nonregression
accepted: True
next_action: run the standard native language non-regression gate before any
  backbone comparison or larger reasoning promotion
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
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.06060606060606061
}
```

Report:
`local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_triage/report.json`

## Runner Result 2026-05-14T19:47:04

The standard version of the same native language non-regression gate also
passed. This strengthens the hard-lock baseline, but it still does not prove
usable open-ended language. The generated sample remains tiny-corpus language
fragments; the accepted claim is narrower: the mandatory recurrent core
improves loss over core-off without materially regressing against a native
baseline on the same wiki text.

```text
gate: qtrm_native_l5_language_nonregression
target_level: L5C native language non-regression
profile: standard
decision: accepted_l5_language_nonregression
accepted: True
next_action: broaden the native text slice before any backbone comparison
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
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.015151515151515152
}
```

Report:
`local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_standard/report.json`

## Runner Result 2026-05-14T19:50:49

```text
gate: qtrm_native_broad_wiki_text_nonregression
target_level: L5C broad native wiki text non-regression
profile: triage
decision: rejected
accepted: False
next_action: treat the prior single-file language result as too narrow; fix corpus loading, capacity, tokenizer, or recurrence placement
```

Decisive metrics:

```json
{
  "last_loss": 2.821063995361328,
  "eval_metrics.think_eval_loss": 2.9197988465627036,
  "eval_metrics.think0_loss": 3.5613443400065106,
  "eval_metrics.thinking_block_off_loss": 3.5613443400065106,
  "eval_metrics.think0_baseline_loss": 2.874905122121175,
  "eval_metrics.loss_ratios.full_vs_think0": 0.8198586173662067,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.8198586173662067,
  "eval_metrics.loss_ratios.full_vs_baseline": 1.0156157238359242,
  "eval_metrics.sample_degeneracy.unique_chars": 11.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.6287878787878788,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_broad_wiki_text_nonregression_triage/report.json`

## Runner Result 2026-05-14T19:51:17

```text
gate: qtrm_native_broad_wiki_text_nonregression
target_level: L5C broad native wiki text non-regression
profile: standard
decision: accepted_broad_wiki_text_nonregression
accepted: True
next_action: stay QTRM-native: add a broad-corpus depth sweep or only then return to native mixed reasoning under the same token->core->logits path
```

Decisive metrics:

```json
{
  "last_loss": 1.8918626308441162,
  "eval_metrics.think_eval_loss": 1.9832347703895081,
  "eval_metrics.think0_loss": 3.65362258938584,
  "eval_metrics.thinking_block_off_loss": 3.65362258938584,
  "eval_metrics.think0_baseline_loss": 1.952862384413351,
  "eval_metrics.loss_ratios.full_vs_think0": 0.5428132550283149,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.5428132550283149,
  "eval_metrics.loss_ratios.full_vs_baseline": 1.0155527528301904,
  "eval_metrics.sample_degeneracy.unique_chars": 16.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.007575757575757576,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_broad_wiki_text_nonregression_standard/report.json`

## Runner Result 2026-05-14T19:52:49

```text
gate: qtrm_native_broad_wiki_depth_ablation
target_level: L5C broad native wiki depth ablation
profile: standard
decision: accepted_broad_wiki_depth_ablation
accepted: True
next_action: stay QTRM-native: return to native mixed reasoning only with the broad corpus language/depth baselines kept as regression gates
```

Decisive metrics:

```json
{
  "last_loss": 1.8918626308441162,
  "eval_metrics.think_eval_loss": 1.9832347703895081,
  "eval_metrics.think0_loss": 3.65362258938584,
  "eval_metrics.thinking_block_off_loss": 3.65362258938584,
  "eval_metrics.think0_baseline_loss": 1.952862384413351,
  "eval_metrics.loss_ratios.full_vs_think0": 0.5428132550283149,
  "eval_metrics.loss_ratios.full_vs_thinking_block_off": 0.5428132550283149,
  "eval_metrics.loss_ratios.full_vs_baseline": 1.0155527528301904,
  "eval_metrics.loss_ratios.full_vs_best_shallow_depth": 0.8785494939281318,
  "eval_metrics.best_shallow_depth_loss": 2.2573967478168546,
  "eval_metrics.sample_degeneracy.unique_chars": 16.0,
  "eval_metrics.sample_degeneracy.max_run_fraction": 0.007575757575757576,
  "backend_summary.fla_delta_mixers": 0,
  "backend_summary.official_fla_delta_mixers": 0,
  "backend_summary.mamba3_mixers": 0,
  "backend_summary.official_mamba3_mixers": 0,
  "backend_summary.torch_delta_mixers": 0,
  "backend_summary.all_fla_mixers_official": false,
  "backend_summary.all_mamba3_mixers_official": false
}
```

Report: `local_eval/research_gate_runner/qtrm_native_broad_wiki_depth_ablation_standard/report.json`

## Dual Reverse Recheck 2026-05-14

Question:

```text
Should dual_path_reverse / trm_dual_z_reversed_hybrid_3to1 replace the current
single recurrent MHA ETD baseline?
```

Decision:

```text
No promotion yet.
```

Reason:

```text
latest active single recurrent L4 baseline:
  full_generation_exact: 0.6640625
  report: local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/report.json

best stored trm_dual_z_reversed_hybrid_3to1 local report found:
  full_generation_exact: 0.359375
  report: local_eval/qtrm_native_reversed_efficiency_len4_hardfamily_20260513/len4_seed777/report.json

new comparison gate added:
  qtrm_native_dual_reverse_l4_baseline_compare

gate candidate updated after warm-start repair:
  trm_dual_z_reversed_mha_etd

gate recipe:
  resume from current single MHA ETD L4 baseline
  load matching tensors flexibly
  train only newly introduced dual-reverse parameters
  keep baseline LM/core weights frozen during the repair probe
  add soft z_L counterfactual loss

attempted triage:
  local_eval/research_gate_runner/qtrm_native_dual_reverse_l4_baseline_compare_triage
  result: command_failed because the dual reverse runner was too slow for the
  reduced triage budget and was manually stopped before a model report was
  produced.
```

Rule:

```text
dual reverse can be reconsidered only if it beats 0.664 exact through the same
native token->core->logits path and keeps positive depth/ablation margins.
Until then, single recurrent MHA ETD remains the active baseline.
```

## Dual Reverse Warm-Start Repair 2026-05-14

What changed:

```text
added think_structure:
  trm_dual_z_reversed_mha_etd

purpose:
  keep dual-z reverse state routing, but reuse the proven MHA ETD `think.*`
  checkpoint weights instead of replacing the recurrent core with randomly
  initialized Mamba/Delta hybrid proposal blocks.

added training controls:
  --train-only-resume-missing-params
  --train-param-name-regex
  --z-l-counterfactual-loss-weight
  --z-l-counterfactual-margin
  --z-l-counterfactual-every
```

Key evidence:

```text
baseline-preserving init eval:
  report: local_eval/qtrm_native_dual_reverse_mha_alpha_bias6_init_eval/report.json
  full_generation_exact: 0.6640625
  z_l_zero_generation_exact: 0.6640625
  result: baseline preserved, but z_L is not causal yet

soft z_L counterfactual:
  report: local_eval/qtrm_native_dual_reverse_mha_missing_only_zlcf_soft_s1000/report.json
  full_generation_exact: 0.67578125
  z_l_zero_generation_exact: 0.62109375
  full_minus_worst_ablation: 0.0546875
  result: exact passes 0.665, but z_L causal drop is below 0.10

strong z_L counterfactual:
  report: local_eval/qtrm_native_dual_reverse_mha_missing_only_zlcf_s1000/report.json
  full_generation_exact: 0.65234375
  z_l_zero_generation_exact: 0.4609375
  full_minus_worst_ablation: 0.19140625
  result: z_L causal drop passes, but exact falls below 0.665
```

Current conclusion:

```text
dual-reverse is no longer collapsing from random core replacement, but it is
not promoted yet. The remaining bottleneck is simultaneous satisfaction of:

  full_generation_exact >= 0.665
  full_minus_worst_ablation >= 0.10

The best exact checkpoint has insufficient z_L causality. The best causal
checkpoint has insufficient exact accuracy. This means the next experiment
should improve z_L/H synergy, not add more backbone variants.
```

## Nested Learning Application 2026-05-14

Source:

```text
Nested Learning: The Illusion of Deep Learning Architectures
arXiv:2512.24695
local paper: references/papers/arxiv_2512_24695_nested_learning.pdf
local implementation reference: references/external/nested_learning

Useful idea:
  treat model components as nested optimization/update levels with different
  context flows and update frequencies.

Not used as:
  a full HOPE/CMS architecture replacement for the current L4 gate.
```

Interpretation for QTRM-native:

```text
z_L:
  fast working latent state
  updated multiple times inside each H step

z_H:
  slower higher-level latent state
  updated once per outer H step

current bottleneck:
  exact-passing checkpoints still let z_H dominate;
  strongly causal z_L checkpoints lose exact accuracy.

Nested Learning repair hypothesis:
  add a learned update rule to z_L and z_H, but keep the same native
  token->core->logits path and keep the warm-startable MHA ETD core.
```

Implemented candidate:

```text
think_structure:
  trm_dual_z_nested_reversed_mha_etd

path:
  prompt tokens
  -> native embeddings
  -> MHA ETD encoder
  -> dual z_L/z_H reversed recurrent core
  -> baseline-preserving MHA ETD think block
  -> learned nested update MLP for fast z_L
  -> learned nested update MLP for slow z_H
  -> shared LM logits

baseline preservation:
  uses the existing single MHA ETD `think.*` checkpoint tensors
  initializes nested update final layers to zero
  initializes nested gates near off
  trains only newly introduced parameters during the first comparison probe
```

New gate:

```text
qtrm_native_nested_dual_reverse_l4_baseline_compare
```

Acceptance rule:

```text
Promote only if:
  full_generation_exact >= 0.665
  full_minus_worst_ablation >= 0.10
  depth gain remains positive

Reject if:
  learned update only preserves baseline without z_L/z_H causal drop
  or improves ablation drop by sacrificing exact accuracy
```

First standard result:

```text
report:
  local_eval/research_gate_runner/qtrm_native_nested_dual_reverse_l4_baseline_compare_standard/report.json

result:
  rejected

metrics:
  full_generation_exact: 0.67578125
  think0_generation_exact: 0.03125
  full_minus_think0: 0.64453125
  full_minus_worst_ablation: 0.05859375
  z_l_zero_generation_exact: 0.6171875
  z_h_zero_generation_exact: 0.0

interpretation:
  Nested update preserves the exact gain and matches the best soft z_L
  counterfactual exact result, but z_L removal still leaves 0.617 exact.
  Therefore the learned update is not yet causally strong enough to promote.

next repair direction:
  train nested update with a stronger z_L/H synergy objective, not more
  backbone variants. The specific failure is z_H still being too competent
  without z_L, while z_H removal is destructive.
```

Accepted repair result:

```text
report:
  local_eval/qtrm_native_nested_dual_reverse_zlcf_mid_s1000/report.json

recipe:
  trm_dual_z_nested_reversed_mha_etd
  resume from current single MHA ETD L4 baseline
  train only missing nested/dual parameters
  z_l_counterfactual_loss_weight: 0.10
  z_l_counterfactual_margin: 0.15

decision:
  accepted_nested_dual_reverse_l4_baseline_compare

metrics:
  full_generation_exact: 0.67578125
  think0_generation_exact: 0.03125
  full_minus_think0: 0.64453125
  full_minus_worst_ablation: 0.140625
  z_l_zero_generation_exact: 0.53515625
  z_h_zero_generation_exact: 0.0

interpretation:
  This is the first accepted Nested Learning inspired dual-reverse repair.
  It does not prove broad L4/L5 generality, but it clears the local baseline
  comparison: exact beats the 0.664 single baseline and both z_L and z_H are
  causally relevant under ablation.

canonical status:
  promote to a candidate, not the final architecture.
  next required checks are seed stability and broad language/depth
  non-regression.
```

## TRM Split-Mixer Naming Correction 2026-05-14

Problem:

```text
The earlier name `reversed_hybrid_3to1` overloaded two different ideas:

1. reversing or changing the TRM update schedule;
2. keeping the official TRM schedule but assigning different mixers to z_L
   and z_H.

For the canonical QTRM-native path this ambiguity is harmful. A result should
not be called "reverse" unless the update order itself is reversed.
```

Canonical replacement:

```text
think_structure:
  trm_dual_z_nested_official_schedule_split_mixer_3to1

gate:
  qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
```

Architecture:

```text
prompt tokens
-> native embeddings
-> native MHA ETD encoder
-> official TRM schedule:
     H outer steps = 3
     L inner cycles per H step = 6
-> z_L fast path:
     Mamba3, Mamba3, Mamba3, Attention
-> z_H slow path:
     GatedDelta, GatedDelta, GatedDelta, Attention
-> nested learned update MLPs for z_L and z_H
-> native MHA ETD decoder/readout
-> LM logits / autoregressive text
```

Compatibility note:

```text
trm_dual_z_nested_reversed_hybrid_3to1 remains only as a legacy experiment
identifier. It is not the canonical name for the official H=3/L=6 split-mixer
architecture.
```

Acceptance rule:

```text
Promote only if:
  full_generation_exact >= 0.665
  full_minus_worst_ablation >= 0.10
  full_minus_think0 > 0
  z_L and z_H zero ablations both reduce the same LM-logit generation metric

Reject if:
  the split-mixer path only improves a probe,
  needs a hidden renderer,
  breaks the native token->core->logits path,
  or fails to beat the accepted nested MHA repair.
```

## 2026 Diffusion / TST / Fast-Slow Decision 2026-05-14

Sources:

```text
Mercury-2 product note:
  https://www.inceptionlabs.ai/blog/introducing-mercury-2

Mercury diffusion LLM paper:
  https://arxiv.org/abs/2506.17298

Token Superposition Training:
  https://arxiv.org/abs/2605.06546

Fast-Slow Training:
  https://arxiv.org/abs/2605.12484

wiki source note:
  docs/wiki/sources/diffusion-fast-slow-llm-2026.md
```

Decision:

```text
Do not pivot the canonical QTRM-native architecture from recursive TRM/QTRM
core to a standalone diffusion language model.
```

Reason:

```text
Mercury-style diffusion is most relevant to answer decoding speed and
multi-token refinement. It does not by itself prove stronger recursive raw
intelligence. QTRM-native progress still requires:

  prompt tokens
  -> native encoder
  -> mandatory recursive core
  -> core-dependent readout
  -> LM logits / text

with core/depth/state ablations reducing the same final metric.
```

Mapping:

```text
Mercury / diffusion LLM:
  later decoder/readout candidate.
  use only as core-conditioned parallel refinement, not as an answer sidecar.

Token Superposition Training:
  later native pretraining throughput candidate.
  not an immediate fix for the L4 z_L/z_H causality bottleneck.

Fast-Slow Training:
  immediate architectural-training inspiration.
  maps naturally to z_L fast working state, z_H slow abstract state, and nested
  learned update pressure.
```

Next if current split-mixer fails:

```text
Try a Fast-Slow-style latent update gate before more architecture shopping:

  qtrm_native_fast_slow_latent_update_l4_repair

Goal:
  make z_L causally necessary for fast per-instance state;
  keep z_H as slower high-level control;
  prevent z_H-only solving from passing the gate.
```

Implementation status:

```text
implemented:
  fast_slow_latent_counterfactual_loss

location:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  scripts/300_research_gate_runner.py

smoke:
  local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_smoke/report.json

decision:
  smoke_passed_fast_slow_latent_update_l4_repair

boundary:
  This is a QTRM latent-state adaptation of Fast-Slow Training, not a full
  GEPA/textual fast-weight reproduction.
```

Standard result:

```text
report:
  local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_standard/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.2421875
  think0_generation_exact: 0.03125
  full_minus_think0: 0.2109375
  full_minus_worst_ablation: 0.1875
  z_L_zero_generation_exact: 0.0234375
  z_H_zero_generation_exact: 0.0

interpretation:
  Fast-Slow pressure slightly improves the official-schedule split-mixer
  3:1 path over 0.23046875 and makes z_L ablation more destructive, but exact
  generation remains far below the 0.665 L4 threshold. The bottleneck is no
  longer merely "z_L/z_H are unused"; it is that the Mamba3/GatedDelta split
  mixer fails to recover the accepted MHA/ETD answer-generation capability.

decision:
  Do not promote the split-mixer 3:1 branch. Freeze the accepted
  trm_dual_z_nested_reversed_mha_etd result as the active QTRM-native baseline
  and move to seed stability / multi-family repair / language non-regression.
```

Qualification after L5 multi-family repair:

```text
Do not hard-freeze nested dual-z or Fast-Slow globally.

Evidence:
  nested dual-z MHA/ETD is accepted on the local L4 modchain baseline gate, but
  when attached to the L5 multi-family seed339 checkpoint it initially preserves
  full accuracy while failing z_L causality:

    full_generation_exact: 0.6223958333333334
    min_family_generation_exact: 0.40234375
    z_L_zero_generation_exact: 0.62109375
    full_minus_worst_ablation: 0.0013020833333333703

  stronger z_L counterfactual training made z_L causal but damaged full
  multi-family accuracy:

    full_generation_exact: 0.3346354166666667
    min_family_generation_exact: 0.07421875

Decision:
  nested dual-z stays a local L4 candidate, not the mandatory L5/L6 baseline.
  Fast-Slow stays a repair hypothesis, not a fixed loss.

Current stable L5 path:
  single recurrent MHA/ETD QTRM-native core with weak-family repair:
    low LR: 2e-5
    weak-family oversampling: modchain,modchain,revchain,checksum
    family_dro_loss_weight: 0.05
    retention_kl_loss_weight: 0.50
```

Repaired L5 multi-family seed stability:

```text
summary:
  local_eval/qtrm_native_l5_multifamily_repaired_seed_stability_20260514_summary.json

decision:
  accepted_l5_multifamily_repaired_seed_stability

pass_rate:
  3 / 3

minimum metrics across the three accepted seed reports:
  min_full_generation_exact: 0.6067708333333334
  min_family_generation_exact: 0.4140625
  min_full_minus_worst_ablation: 0.5716145833333334

note:
  seed337 uses the original accepted report because its checkpoint file was not
  retained. seeds338 and 339 use the weak-family repair recipe.
```

L6 length-scaling constraint:

```text
zero-shot len6:
  report: local_eval/qtrm_native_l6_len6_transfer_from_l5_repair_seed339_init_20260514/report.json
  decision: rejected
  full_generation_exact: 0.0078125
  min_family_generation_exact: 0.00390625

focused len6:
  report: local_eval/qtrm_native_l6_len6_from_l5_repair_focused_s2500_20260514/report.json
  decision: rejected
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.05859375

answer-space diagnostic:
  report: local_eval/qtrm_native_l6_len6_focused_s2500_answer_space_eval_20260514/report.json
  answer_space_argmax_exact: 0.0794270858168602
  answer_space_gold_top5: 0.3515625
```

Decision:

```text
The accepted L5 single recurrent MHA/ETD baseline does not yet solve len6.
Because answer-space argmax is not materially above greedy exact, this is not
mainly a print-only renderer failure. Treat len6 ordered recurrent transition
and hard-family balance as the next bottleneck. Do not hard-freeze nested
dual-z, Fast-Slow, or Mamba3/GatedDelta split-mixer 3:1 globally until they
beat this same L6 gate with causal ablations and seed stability.
```

Active-length batch-cycle control:

```text
report: local_eval/qtrm_native_l6_len6_from_l5_repair_active_batch_cycle_s2500_20260514/report.json
decision: rejected
full_generation_exact: 0.052083333333333336
min_family_generation_exact: 0.046875
full_minus_worst_ablation: 0.01953125
```

Decision update:

```text
Simple active-length exposure is not the missing ingredient. The next fix must
improve the recurrent transition/state representation or use a more principled
length curriculum, not globally freezing nested/Fast-Slow.
```

L6 state/readout diagnostic:

```text
report: local_eval/qtrm_native_l6_len6_focused_state_probe_eval_20260515/report.json
core_step_probe_exact: 0.09765625
z_h_variance_by_depth: 0.24019193649291992, 0.3616328239440918, 0.5645048022270203, 0.8666419386863708
```

Latent-refinement control:

```text
report: local_eval/qtrm_native_l6_len6_from_l5_repair_latent_refine_s2500_20260515/report.json
decision: rejected
full_generation_exact: 0.08854166666666667
min_family_generation_exact: 0.0546875
core_step_probe_exact: 0.0960286483168602
```

Decision update:

```text
The recurrent state moves but is not a clean causal transition-state carrier.
Latent refinement does not fix that. Keep QTRM-native locked, but the next
canonical improvement must address hard-family transition representation rather
than globally promoting nested dual-z, Fast-Slow, or output-only losses.
```

Block/topology clarification:

```text
`trm_official_prenorm` is a local block/backbone candidate, not a replacement
for the dual/nested macro topology. A run only counts as dual/nested if
think_structure is explicitly dual/nested, for example:

  trm_dual_z_nested_reversed_mha_etd
  trm_dual_z_nested_official_schedule_split_mixer_3to1

The single-core `trm_official_prenorm` path is diagnostic only.
```

Official-prenorm dual/nested diagnostic:

```text
no-abs L4 triage:
  report: local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_noabs_triage_20260514_235922/report.json
  decision: rejected
  full_generation_exact: 0.041666666666666664
  full_minus_worst_ablation: -0.03125000000000001

learned-position L4 control:
  report: local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_learnedpos_triage_20260515_000034/report.json
  decision: rejected
  full_generation_exact: 0.052083333333333336
  full_minus_worst_ablation: -0.010416666666666664
  z_l_zero_generation_exact: 0.052083333333333336
```

Decision update:

```text
Do not promote `trm_official_prenorm` into the canonical dual/nested path. It
does not beat the accepted nested MHA/ETD L4 reference, and the learned-position
control is not causally dependent on z_L. Keep the hard lock on QTRM-native and
the local dual/nested reference, but focus the next architecture change on the
state carrier/readout coupling rather than another backbone swap.
```

State/readout coupling update:

```text
new structures:
  trm_dual_z_nested_reversed_mha_etd_joint_readout
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout

accepted residual report:
  local_eval/qtrm_native_l4_nested_mha_residual_joint_readout_from_accepted_20260515_001357/report.json

decision:
  accepted_nested_mha_residual_joint_readout_l4

metrics:
  full_generation_exact: 0.671875
  full_minus_think0: 0.640625
  full_minus_worst_ablation: 0.14453125
  z_l_zero_generation_exact: 0.52734375
  z_h_zero_generation_exact: 0.0
```

Decision update:

```text
The residual joint-readout is now the next diagnostic candidate because it
preserves the accepted nested MHA/ETD L4 level while slightly improving the
ablation margin. It does not replace the stable baseline until it passes L5/L6
family-floor gates. Full retuning of the replacement joint-readout is rejected
because it collapsed full_generation_exact to 0.1328125.
```

L5 follow-up:

```text
Direct L4 residual -> L5 multi-family:
  report: local_eval/qtrm_native_l5_residual_joint_readout_multifamily_seed337_vocabremap_20260515_0028/report.json
  decision: rejected
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.023529411764705882

Attach nested residual path to accepted L5 single checkpoint:
  report: local_eval/qtrm_native_l5_residual_joint_readout_from_l5_single_missingonly_seed339_20260515_0035/report.json
  decision: rejected
  reject_reason: ablation_drop_below_threshold
  best_periodic_generation_exact: 0.734375
  best_periodic_min_family_generation_exact: 0.5581395348837209
  final_full_generation_exact: 0.68359375
  z_l_zero_generation_exact: 0.6953125
```

Decision update:

```text
Residual joint-readout remains an L4 diagnostic candidate, not the L5 canonical
path. It can coexist with an accepted L5 single checkpoint, but the L5 gain is
not dual-state causal because z_L zeroing does not reduce the metric. The next
dual/nested work must target z_L as an actual intermediate state carrier.
```

z_L causal repair acceptance:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair2_seed339_20260515_0056/report.json

recipe:
  resume:
    local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair_seed339_20260515_0048/last.pt
  train only:
    ^trm_
  z_l_counterfactual_loss_weight:
    1.0
  z_l_counterfactual_margin:
    0.50

decision:
  accepted_l5_residual_joint_readout_zl_causal_repair2

decisive:
  full_generation_exact: 0.703125
  think0_generation_exact: 0.02734375
  full_minus_think0: 0.67578125
  full_minus_worst_ablation: 0.296875
  min_family_generation_exact: 0.5581395348837209
  z_l_zero_generation_exact: 0.40625
  z_h_zero_generation_exact: 0.0
```

Decision update:

```text
The L5 dual/nested path is no longer rejected in principle. The accepted route
is conservative: preserve the accepted L5 single recurrent base, attach nested
residual joint-readout, then repair z_L causality by training only TRM
parameters. Do not hard-freeze Fast-Slow as mandatory; it has not earned the
same L5 causal acceptance.
```

Seed/L6 qualification:

```text
seed338 reproduction:
  report: local_eval/qtrm_native_l5_residual_joint_readout_seed338_zl_repair2_20260515_0118/report.json
  decision: rejected
  full_generation_exact: 0.75
  min_family_generation_exact: 0.611764705882353
  z_l_zero_generation_exact: 0.734375
  full_minus_worst_ablation: 0.015625

L6 transfer from accepted seed339:
  init report: local_eval/qtrm_native_l6_len6_from_l5_zl_causal_seed339_init_20260515_0126/report.json
  fine-tune report: local_eval/qtrm_native_l6_len6_from_l5_zl_causal_finetune_s2500_20260515_0130/report.json
  fine-tune decision: rejected
  fine-tune full_generation_exact: 0.0703125
  fine-tune min_family_generation_exact: 0.03125
```

Decision update:

```text
Do not call nested residual z_L-causal the stable canonical baseline yet. It is
accepted for L5 seed339, but not seed-stable and not len6-capable. The hard
lock remains QTRM-native; the next bottleneck is ordered transition
generalization with stable z_L causality.
```

Seed-stability update:

```text
seed337:
  report: local_eval/qtrm_native_l5_seed337_zl_codec_direct_repair_s1200_20260515_0152/report.json
  decision: accepted_l5_seed337_zl_codec_direct_repair
  full_generation_exact: 0.703125
  full_minus_worst_ablation: 0.23828125
  min_family_generation_exact: 0.5465116279069767
  z_l_zero_generation_exact: 0.46484375

seed338:
  report: local_eval/qtrm_native_l5_seed338_zl_codec_repair_final_s1200_20260515_0145/report.json
  decision: accepted_l5_seed338_zl_codec_repair
  full_generation_exact: 0.75390625
  full_minus_worst_ablation: 0.28125
  min_family_generation_exact: 0.6235294117647059
  z_l_zero_generation_exact: 0.47265625

seed339:
  report: local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair2_seed339_20260515_0056/report.json
  decision: accepted_l5_residual_joint_readout_zl_causal_repair2
  full_generation_exact: 0.703125
  full_minus_worst_ablation: 0.296875
  min_family_generation_exact: 0.5581395348837209
  z_l_zero_generation_exact: 0.40625
```

Decision update:

```text
Nested residual joint-readout with z_L causal repair is now the active L5
dual/nested candidate. Core-step codec on z_L is the stabilizing recipe for
seeds where counterfactual loss alone leaves z_L non-causal. This is not yet a
L6/generalization claim.
```

L6 depth/length alignment update:

```text
finding:
  The earlier len6 transfer runs used program_len=6 while train/eval
  think_steps stayed at the script default 4. Those runs are useful diagnostics
  but they are not clean falsifiers of the architecture, because the recurrent
  core had fewer steps than the ordered program length.

nested residual z_L-codec depth6:
  report: local_eval/qtrm_native_l6_len6_seed338_zl_codec_depth6_s3000_20260515_0215/report.json
  decision: rejected
  full_generation_exact: 0.08072916666666667
  min_family_generation_exact: 0.0390625

single recurrent depth6:
  report: local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/report.json
  decision: accepted_l6_len6_single_seed338_depth6
  full_generation_exact: 0.3671875
  think0_generation_exact: 0.0
  full_minus_worst_ablation: 0.34375
  min_family_generation_exact: 0.0703125
```

Decision update:

```text
Before changing mixer/backbone/topology for a longer-length failure, first
align train_think_steps and eval_think_steps with the active program length and
compare against the default-depth run. The current L6 scaffold is single
recurrent MHA/ETD with depth6. The L5 dual/nested z_L-codec path remains the
active L5 candidate, but it is not the L6 winner yet.
```

L6 continuation check:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_cont_s3000_lr1e5_20260515_011502/report.json

decision:
  accepted_l6_len6_single_seed338_depth6_cont

comparison:
  base full_generation_exact: 0.3671875
  continued full_generation_exact: 0.3541666666666667
  base min_family_generation_exact: 0.0703125
  continued min_family_generation_exact: 0.0703125

continued family exact:
  checksum: 0.90625
  modchain: 0.0859375
  revchain: 0.0703125
```

Decision update:

```text
Lower-lr continuation did not improve the L6 scaffold. Do not spend more runs
on plain continuation. The next L6 work should target weak-family floor repair
for modchain/revchain or seed-stability measurement, while preserving
program_len=6 and train/eval think_steps=6.
```

Weak-family repair update:

```text
first repair:
  report: local_eval/qtrm_native_l6_len6_single_seed338_family_floor_repair_s2500_20260515_011837/report.json
  decision: rejected
  reject_reason: family_exact_below_threshold
  full_generation_exact: 0.3567708333333333
  min_family_generation_exact: 0.078125
  family exact:
    checksum: 0.90625
    modchain: 0.078125
    revchain: 0.0859375

second modchain-focused repair:
  report: local_eval/qtrm_native_l6_len6_single_seed338_modchain_floor_repair_s1500_20260515_012038/report.json
  decision: rejected
  result: restored the initial checkpoint; stronger DRO did not improve.
```

Decision update:

```text
The L6 aligned-depth scaffold has a small family-floor repair signal, but it
is still weak. Do not call L6 solved. Next choose either seed-stability
measurement for the aligned single-depth6 recipe or a more principled
transition objective; do not return to plain continuation or stronger DRO.
```

Active-length replay update:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_active_replay_s2000_20260515_012955/report.json

decision:
  rejected

result:
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.34375
  restored best checkpoint: initial checkpoint
```

Decision update:

```text
Simple prefix replay is not the missing ingredient. The next L6 repair should
test a structural transition carrier that stays on the normal prompt-token ->
recurrent-core -> LM-logit path, rather than more continuation, stronger DRO,
or auxiliary codec weighting.
```

Transition-carrier decision:

```text
visible prompt anchor:
  report:
    local_eval/qtrm_native_l6_len6_single_seed338_prompt_anchor_s2500_20260515_013530/report.json
  decision: rejected
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.30729166666666663

internal single-core carrier:
  implementation:
    scripts/335_train_qtrm_native_etd_probe.py
    think_structure=single_core_carrier
  report:
    local_eval/qtrm_native_l6_len6_single_seed338_single_core_carrier_s2500_20260515_013530/report.json
  decision: accepted_l6_len6_single_seed338_single_core_carrier
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.09375
  full_minus_worst_ablation: 0.3307291666666667
  state_reset_generation_exact: 0.0026041666666666665
  op_zero_generation_exact: 0.036458333333333336
```

Decision update:

```text
Promote `single_core_carrier` to the active L6 scaffold. This does not
abandon dual/nested: dual/nested remains the active L5 macro-topology, but at
L6 the strongest canonical evidence currently comes from the single recurrent
core plus an internal causal carrier. The next validation must be seed
stability or length scaling, not another visible prompt anchor.
```

Seed-stability caveat:

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

Decision update:

```text
The carrier result is promising but not seed-stable. Keep it as the active L6
scaffold because it is the first accepted structural transition-carrier repair,
but do not call L6 solved. The next bottleneck is robust optimization/stability
of the carrier across seeds, not more visible prompt anchoring.
```

Family-DRO 0.15 follow-up:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_famdro015_s2500_20260515_014659/report.json

decision:
  rejected

result:
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.15364583333333334
```

Decision update:

```text
Do not keep raising family-DRO for seed339. The next seed-stability repair
needs carrier-specific supervision/curriculum, or a third-seed check before
changing the architecture again.
```

Third-seed carrier update:

```text
seed337 single_core_carrier:
  report:
    local_eval/qtrm_native_l6_len6_single_seed337_single_core_carrier_s2500_20260515_015547/report.json
  decision: rejected
  reject_reason: family_exact_below_threshold
  full_generation_exact: 0.1640625
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.140625
```

Decision update:

```text
The current `single_core_carrier` implementation is not seed-stable. It remains
a useful L6 transition-state hypothesis, but not the canonical architecture.

`trm_official_prenorm` is only a backbone/block choice: official-TRM-style
attention/SwiGLU with pre-norm residual updates for LM stability. It does not
mean abandoning dual-z or nested TRM. Dual/nested remains the intended macro
structure once the core transition-state training is robust.
```

Carrier supervision update:

```text
seed339 single_core_carrier + h-mean core-step codec:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hmean_w008_s2500_20260515_020418/report.json
  decision: rejected
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.16666666666666669
```

Decision update:

```text
Do not promote mean-pooled carrier codec. It confirms the carrier path remains
causal, but it does not solve family-floor stability. Try last-state carrier
supervision once; if it also fails, move from auxiliary codec supervision to
direct transition-state consistency.
```

Last-state carrier codec update:

```text
seed339 single_core_carrier + h-last core-step codec:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hlast_w008_s2500_20260515_020735/report.json
  decision: rejected
  full_generation_exact: 0.2109375
  min_family_generation_exact: 0.046875
  full_minus_worst_ablation: 0.1875
```

Decision update:

```text
Stop codec-pooling variants. They are not solving the hard-family floor. The
next allowed L6 repair is direct recurrent state consistency, not more codec
heads or backbone renaming.
```

Prefix/full state consistency update:

```text
seed339 single_core_carrier + prefix_state_alignment:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_align_w002_s2500_20260515_021141/report.json
  decision: rejected
  full_generation_exact: 0.20572916666666666
  min_family_generation_exact: 0.046875
  full_minus_worst_ablation: 0.18229166666666666
```

Decision update:

```text
Plain MSE prefix/full state consistency is not enough. Try contrastive
prefix/full state alignment once. If rejected, stop auxiliary-state losses and
change the transition mechanism or curriculum itself.
```

Contrastive state consistency update:

```text
seed339 single_core_carrier + prefix_state_contrastive:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_contrast_w002_s2500_20260515_021514/report.json
  decision: rejected
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.16666666666666669
```

Decision update:

```text
Stop auxiliary state losses for this seed339 repair. The next permissible
experiment is a curriculum change: warm up only the newly added carrier
parameters before full fine-tuning. This keeps the same canonical LM path and
tests whether the instability is from abrupt random carrier insertion.
```

Seed-stability update:

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

Decision update:

```text
Aligned single recurrent depth6 is seed-reproducible at the diagnostic
threshold, but not strong or stable. It is the current L6 scaffold, not a final
L6 solution and not evidence that dual/nested should be abandoned. The next
architecture work should improve robust ordered transition generalization and
then re-test dual/nested against this scaffold.
```

Core-step codec update:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_corecodec_h_s2500_20260515_012658/report.json

recipe:
  core_step_codec_loss_weight: 0.25
  core_step_codec_state_source: h
  restore_best_eval_checkpoint: true

decision:
  rejected

decisive:
  full_generation_exact: 0.3671875
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.34375
```

Decision update:

```text
The h-state core-step codec did not improve L6; restore-best selected the
initial checkpoint. Do not keep sweeping codec weights. Prefer an LM-path
prefix/replay consistency objective or a structural transition repair.
```

Next if current split-mixer passes:

```text
Do not add Mercury/TST immediately.
Run seed stability and broad language non-regression first.
Only then add:
  qtrm_native_parallel_refinement_decoder_gate
  qtrm_native_tst_pretraining_efficiency_gate
```

L6 carrier repair decision:

```text
single_core_carrier seed338:
  accepted as a structural hint, not as a seed-stable architecture

seed337/seed339 carrier:
  rejected on family floor or full exact

tested repair classes:
  core-step codec:
    rejected
  prefix/full state alignment:
    rejected
  contrastive state alignment:
    rejected
  carrier-only warmup:
    rejected
  carrier_gate_init=-4.0:
    rejected; family floor improved but full exact fell below threshold
  linear checkpoint soup:
    rejected for alpha 0.1, 0.2, 0.3, 0.4, 0.5
```

Decision:

```text
Keep QTRM-native hard lock. Do not pivot to donor, MemoryOS, RAG, or renderer
shortcuts.

Do not interpret `trm_official_prenorm backbone` as abandoning dual/nested TRM.
It is a block choice only. The macro choices are still:
  single recurrent core;
  dual z_L/z_H TRM;
  nested dual TRM.

The current canonical L6 scaffold is the aligned single recurrent depth6 path.
The next architecture candidate should re-test dual/nested with an internal
transition-state carrier, because the carrier diagnostics show causal state can
help but must be integrated into the recurrent core more robustly.
```

Evaluation update:

```text
Use --eval-family-order-invariant for future family-floor promotion claims.

Rationale:
  A family-floor gate is only meaningful if a prompt-order or family-order
  change cannot silently swap the held-out examples under each family.

Current order-invariant dual/nested status:
  best report:
    local_eval/qtrm_native_l6_seed338_order_invariant_revchain_fullperiodic_repair_s800_20260515/report.json
  decision:
    rejected
  bottleneck:
    revchain is 10/128, while the strict >=0.08 floor needs 11/128.
  causal evidence:
    z_l_zero_generation_exact: 0.0
    z_h_zero_generation_exact: 0.0
    full_minus_worst_ablation: 0.3125
```

Hard lock implication:

```text
Do not use an eval-order change, MemoryOS/RAG side path, donor fallback, or
renderer shortcut to claim progress. The next promotion must come from the
native recurrent transition itself improving the same order-invariant LM-logit
generation metric.
```

Dual/nested terminology lock:

```text
`trm_official_prenorm backbone` is not a macro-topology decision.

Allowed interpretation:
  local pre-norm official-TRM-style block/backbone material.

Disallowed interpretation:
  abandoning dual z_L/z_H;
  abandoning nested TRM;
  replacing the QTRM-native hard lock with a donor/residual path.

Macro topology remains explicit in think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier
```

Current accepted seed338 scaffold:

```text
report:
  local_eval/qtrm_native_l6_seed338_order_invariant_nested_core_carrier_identity_init_carrieroff_eval2_20260515/report.json

decision:
  accepted_l6_nested_core_carrier_identity_init

canonical path:
  prompt tokens -> native embedding -> dual/nested z_L/z_H recurrent core
  -> in-core carrier delta -> residual joint readout -> LM logits
  -> autoregressive answer text

guardrail:
  The carrier is only valid because it is inside the recurrent core and
  carrier_off returns to the previous near-pass. It must not become an external
  answer channel.

promotion limit:
  This is a seed338 L6 scaffold. It must survive seed stability and language
  non-regression before becoming the default QTRM-native architecture.
```

Seed-stability decision:

```text
summary:
  local_eval/qtrm_native_l6_nested_core_carrier_seed_stability_summary_20260515.json

decision:
  rejected_seed_stability

result:
  seed337 rejected;
  seed338 accepted;
  seed339 rejected.

hard-lock consequence:
  Do not promote the core-carrier as default yet.
  Do not freeze Fast-Slow as default.
  Do not pivot to donor/RAG/MemoryOS to hide the family-floor weakness.

next allowed repair:
  a QTRM-native recurrent-state repair that makes the dual/nested carrier less
  random-init-dependent and passes the same family-floor/depth/ablation gate.
```

Carrier repair audit update:

```text
summary:
  local_eval/qtrm_native_l6_seed337_carrier_brittleness_repair_summary_20260515.json

decision:
  rejected_deterministic_carrier_and_seed337_repairs

hard-lock consequence:
  deterministic carrier modes and local carrier-only training are not enough.
  The next repair must first determine whether the weak-seed failure is a
  readout/generation problem or a recurrent transition problem.

required next evidence:
  answer-space argmax/rank on weak seeds under the same QTRM-native path.
```

Answer-space evidence:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_carrier_answer_space_audit_20260515/report.json

finding:
  answer_space_argmax_exact is 0.3177083432674408 while greedy generation is
  0.3203125. Gold mean rank is 8.098958015441895.

decision:
  This is not a renderer/readout-only bottleneck. The next hard-lock-compliant
  repair must improve recurrent transition/state accuracy before the LM logits.
```

State-trace repair evidence:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_state_trace_depth_repair_strong_s1200_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.0
  answer_space_argmax_exact: 0.34375
  answer_space_gold_mean_rank: 7.916666507720947

hard-lock consequence:
  The strong state-trace depth loss does not promote the architecture. It
  supports the same conclusion as the answer-space audit: the weak-seed floor
  is a recurrent transition/state-binding problem, not a renderer, sidecar, or
  carrier scalar-weighting problem.

canonical next step:
  Keep the QTRM-native token -> native embedding -> mandatory dual/nested
  core -> LM logits path. Redesign the recurrent transition/state update inside
  that path before any donor, MemoryOS, RAG, renderer, or decoding shortcut.
```

Transition-binding prefix contrast:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_binding_prefix_contrast_s900_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3333333333333333
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: -0.002604166666666685

hard-lock consequence:
  This remains QTRM-native and uses no sidecar answer path, but it does not
  pass. Do not promote prefix contrast. The next repair should be more specific
  to transition binding, such as operation/position state-codec pressure or a
  structural state-update redesign.
```

Transition-codec evidence:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_codec_repair_s700_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3125
  min_family_generation_exact: 0.0625
  full_minus_carrier_off: -0.010416666666666685

hard-lock consequence:
  Operation/position state-codec pressure does not solve the seed337
  weak-family floor. Do not promote codec heads or treat them as canonical
  reasoning components. The next valid change must modify the recurrent
  z_L/z_H state update while preserving token -> core -> LM logits generation.
```

Cross-exchange structural evidence:

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

hard-lock consequence:
  This is a valid QTRM-native structural experiment because the exchange lives
  inside the recurrent core and the output remains LM logits. It is still not
  accepted: weak-family accuracy does not improve, and the cross-exchange
  ablation drop is too small. Do not promote it as canonical.
```

Step-conditioned recurrent update evidence:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned

reports:
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_s1000_20260515/report.json
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_step_only_repair_s600_20260515/report.json

decision:
  rejected

best metrics:
  full_generation_exact: 0.3619791666666667
  min_family_generation_exact: 0.0703125
  coupling_off_generation_exact: 0.359375

hard-lock consequence:
  Step conditioning is allowed because it is internal to the native recurrent
  core and answers still pass through LM logits. It is not accepted because
  the modchain family floor remains below threshold and coupling_off does not
  produce a meaningful drop. The next valid path is a nested order-router or
  another structural update-order mechanism, not a donor/RAG/renderer shortcut.
```

Nested order-router hard-lock evidence:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router

report:
  local_eval/qtrm_native_l6_seed337_nested_order_router_aux_forced_s600_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.036458333333333315
  coupling_off_generation_exact: 0.2994791666666667
  order_route0_generation_exact: 0.2994791666666667
  order_route1_generation_exact: 0.041666666666666664

hard-lock consequence:
  This remains a valid QTRM-native experiment because the router lives inside
  the recurrent core and final answers remain normal LM logits. It is not
  accepted. The router learns a family/order signal, but the H->L->H route it
  selects is not a useful transition. Do not count router movement as progress
  unless route forcing and same-metric ablations also improve.
```

Sequence-level order-router hard-lock evidence:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router

report:
  local_eval/qtrm_native_l6_seed337_nested_sequence_order_router_aux_forced_s600_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.09635416666666667
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: -0.23177083333333331
  coupling_off_generation_exact: 0.328125
  order_route0_generation_exact: 0.328125
  order_route1_generation_exact: 0.041666666666666664

hard-lock consequence:
  One-route-per-case routing makes the model worse. This falsifies the idea
  that the main problem was token-wise route inconsistency. The next valid L6
  change is not more routing; it is replacing the weak route1 transition with a
  stronger operation-order/state-binding update inside the mandatory native
  recurrent core.
```

Order-bound route1 hard-lock evidence:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router

report:
  local_eval/qtrm_native_l6_seed337_nested_order_bound_router_aux_forced_s600_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.0390625
  coupling_off_generation_exact: 0.296875
  order_route0_generation_exact: 0.296875
  order_route1_generation_exact: 0.03125

hard-lock consequence:
  The experiment is canonical-path compliant because route1 attention reads
  the same prompt-token encoded state and still emits LM logits. It still does
  not pass. Prompt-source attention alone does not make the reverse route a
  useful recurrent transition. Do not continue H->L->H route variants unless a
  reduced diagnostic first proves route1 can learn operation-order state.
```

Route1 revchain-only diagnostic:

```text
report:
  local_eval/qtrm_native_l6_seed337_order_bound_route1_revchain_capacity_s400_20260515/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.04296875
  min_family_generation_exact: 0.04296875
  full_minus_worst_ablation: 0.01953125
  coupling_off_generation_exact: 0.0078125
  order_route1_generation_exact: 0.046875
  last route1 probability: 0.999332070350647

hard-lock consequence:
  The router can select route1, but route1 still cannot compute revchain. This
  rejects router-selection and multi-family conflict as primary explanations.
  The next valid hard-lock experiment must change the recurrent transition
  mechanism itself or first pass a smaller operation-order transition gate.
```

Reduced operation-order transition diagnostic:

```text
script:
  scripts/353_train_operation_order_transition_probe.py

hard-lock status:
  diagnostic-only, QTRM-native compliant

why compliant:
  The answer is emitted through LM logits from the same prompt-token stream.
  There is no donor, retrieval path, symbolic solver, or hidden final-answer
  channel.

held-out capacity result:
  report: local_eval/operation_order_transition_probe_capacity_mod16_20260515/report.json
  decision: rejected
  full_generation_exact: 0.431640625
  transition_off_generation_exact: 0.07421875
  order_shuffle_generation_exact: 0.2421875

same-seed capacity control:
  report: local_eval/operation_order_transition_probe_capacity_mod16_trainseed_eval_20260515/report.json
  decision: accepted_operation_order_transition_diagnostic
  full_generation_exact: 0.999755859375
  transition_off_generation_exact: 0.0712890625
  order_shuffle_generation_exact: 0.54150390625

hard-lock consequence:
  The recurrent transition can be causally necessary, but the present token
  embedding transition does not yet generalize well enough on held-out
  operation/value combinations. Do not promote it directly into L6. Use it as
  the next scaffold for a state/value codec or transition objective, then
  re-test inside the mandatory dual/nested QTRM-native path.
```

Reduced trace/value-codec hard-lock update:

```text
status:
  accepted diagnostic, not yet canonical L6

accepted evidence:
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

hard-lock interpretation:
  This is still QTRM-native compliant because inference remains prompt tokens
  -> native embeddings/value codec -> recurrent transition -> LM logits. The
  symbolic step results are used only as training targets, not as a runtime
  answer channel. Promote the idea only after the same trace/value-codec recipe
  improves the dual/nested QTRM-native L6 model under the same final-answer
  ablations.
```

L6 transplant hard-lock note:

```text
implemented:
  NativeQTRMETDLM can now receive explicit value_token_ids so a tokenizer can
  define which token ids correspond to values.

guard:
  --value-codec circular is invalid with char tokenizer and now requires
  --tokenizer-mode number in the mixed text runner.

evidence:
  number-tokenizer transplant report:
    local_eval/qtrm_native_l6_seed338_number_circular_trace_repair_s1200_20260515/report.json

  result:
    decision: rejected
    full_generation_exact: 0.068359375
    min_family_generation_exact: 0.023391812865497075
    full_minus_worst_ablation: 0.0078125

hard-lock consequence:
  Do not claim the reduced trace/value success has transferred to L6 yet. The
  full text runner still mixes operation ids and numeric values into the same
  two-digit token ids. The next valid transplant must separate op-role and
  value-role embeddings or apply a role-conditioned codec, while preserving the
  same prompt-token -> recurrent core -> LM-logit path.
```
