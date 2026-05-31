# Renderer Canonical LM Gate

Date: 2026-05-08

Status: active L3-candidate gate.

## Gate Header

```text
Target level:
  L3 candidate

Major bottleneck:
  bottleneck 4, latent-state to autoregressive text renderer

Baseline to beat:
  donor-only, core-off, and renderer-module-off generation modes

Required score:
  full greedy/autoregressive generation accuracy >= 0.50 on smoke8

Required ablation drop:
  full - core_off >= 0.25
  full - renderer_module_off >= 0.25 when the ablation mode exists

Promotion decision if pass:
  broaden to held-out generation gate and preserve the primitive/forced-choice
  signal

Kill decision if fail:
  do not expand memory/metacognition claims; design a donor-compatible text
  renderer first
```

## Scope

This gate is intentionally stricter than primitive runtime or forced-choice
ranking. It uses generation JSONL rows with `hit` labels:

```text
prompt
-> QTRM/donor path
-> LM logits
-> greedy/autoregressive completion
-> exact answer match
```

It rejects scaffold-only success. If primitive operation selection works but
normal text generation remains wrong, bottleneck 4 remains open.

## Runner

```bash
PYTHONPATH=src .venv/bin/python scripts/300_research_gate_runner.py \
  --gate renderer_canonical_lm \
  --profile standard \
  --write-wiki
```

Underlying gate builder:

```text
scripts/302_build_renderer_canonical_lm_gate.py
tests/test_renderer_canonical_lm_gate.py
```

## Runner Result 2026-05-08T09:51:00

```text
gate: renderer_canonical_lm
target_level: L3 candidate
profile: standard
decision: rejected
accepted: False
next_action: renderer remains bottleneck; design a donor-compatible text renderer before memory/metacognition expansion
```

Decisive metrics:

```json
{
  "metrics.full_minus_core_off": 0.0,
  "metrics.full_minus_donor": 0.0,
  "metrics.full_generation_accuracy": 0.0,
  "metrics.core_off_generation_accuracy": 0.0,
  "metrics.donor_generation_accuracy": 0.0,
  "metrics.ablation_generation_accuracy": 0.0,
  "metrics.full_minus_ablation": 0.0
}
```

Report: `local_eval/research_gate_runner/renderer_canonical_lm_standard/report.json`

## Source-Copy Renderer Mask Repair 2026-05-10

Orthodox status: L2/L3 prerequisite repair, not L4 promotion.

The source-copy path was rechecked after the pointer/copy oracle passed. The
important correction is that the probe and the actual renderer must score the
same role block. For `role_value_source_copy_no_doubled=true`, final selected
source positions live in answer roles `0..answer_roles-1`. The renderer was
masking those roles to NULL and reading the later doubled-role block instead.
That made earlier alignment reports overstate what the LM renderer actually
used.

Code/test repair:

```text
src/wgram_lm/wgram_model.py
  _mask_source_copy_position_logits_to_answer_roles now preserves roles 0..3
  for the 10-role source-copy scaffold and masks roles 4..9 to NULL.

tests/test_source_pointer_l4_lm_path_gate.py
  test_l4_source_copy_masks_non_answer_roles_to_null now enforces the same
  answer-role contract as the source-copy probe.
```

Corrected diagnostics:

```text
S002 renderer-true source-copy alignment:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_alignment_probe_s002_renderertrue_eval128/report.json

  content_position_accuracy: 330/384 = 0.8594
  row_content_exact: 83/128 = 0.6484
  decision: rejected_l2_source_copy_alignment_probe

S002 source-binder-off alignment:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_alignment_probe_s002_renderertrue_eval128_binder_off/report.json

  content_position_accuracy: 0/384 = 0.0
  row_content_exact: 0/128 = 0.0

S040 final-state CE alignment:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_alignment_probe_state_ce_s040_fix_renderertrue_eval128/report.json

  content_position_accuracy: 384/384 = 1.0
  row_content_exact: 128/128 = 1.0
  decision: accepted_l2_source_copy_alignment_probe

S020 staged CE alignment:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_alignment_probe_state_ce_staged_s020_lr1e5_renderertrue_eval128/report.json

  row_content_exact: 84/128 = 0.6562
  decision: rejected_l2_source_copy_alignment_probe
```

Generation after the role-mask fix still rejects L4:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_state_ce_s040_fix/last.pt

smoke8 report jsonl:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_maskfix_smoke8/eval.jsonl

donor_only:          5/8
core_off:            5/8
vocab_renderer_off:  5/8
source_binder_off:   5/8
primitive_off:       4/8
full:                3/8
```

Strict scoring correction:

```text
The raw eval previously counted normalized substring matches as hits. For the
source-copy lexicalization gate this is too weak: a completion such as
"55,40,32,44" contains "40,32,44" but is not the requested final answer.

scripts/192_eval_raw_intelligence.py now treats
task_family/category=source_copy_lexicalization as strict exact:
  hit = exact_match or normalized_exact
```

Strict-rescored S040 maskfix smoke8:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_maskfix_smoke8/eval.strict.jsonl

donor_only:          5/8
core_off:            5/8
vocab_renderer_off:  5/8
source_binder_off:   5/8
primitive_off:       4/8
full:                2/8
```

A short S003 renderer-only continuation from S040 did not improve the smoke8
generation result:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_renderer_maskfix_s003_from_state_ce_s040/train/last.pt
  deleted after no-change rejection; eval JSONL preserved.

smoke8:
  full: 3/8
  donor/core_off/vocab_renderer_off/source_binder_off: 5/8
```

Rejected cursor-bias candidate:

```text
candidate:
  source-copy visible-prefix cursor bias

prior family:
  pointer-generator / copy-attention cursor or coverage bias

implementation status:
  optional code path exists behind
  core_role_value_state_vocab_renderer_source_copy_cursor_enabled.
  It is disabled in the source-copy scaffold config after rejection.

report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.jsonl

result:
  donor_only:          5/8
  core_off:            5/8
  vocab_renderer_off:  5/8
  source_binder_off:   5/8
  primitive_off:       2/8
  full:                1/8

strict-rescored report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.strict.jsonl

strict result:
  donor_only/core_off/vocab_renderer_off/source_binder_off: 5/8
  primitive_off: 2/8
  full: 1/8

decision:
  reject as canonical default. The visible-prefix cursor over-biases copy
  tokens and worsens generation.
```

Gold-token rank probe:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_gold_rank_state_ce_s040_maskfix_smoke8.jsonl

donor/core_off/vocab_renderer_off:
  all@1: 3/8
  all<=10: 8/8

full:
  all@1: 2/8
  all<=10: 8/8
```

Interpretation:

```text
The source-position state can now be made exact, and source-binder-off proves
that the alignment probe is causally connected. But exact source-position state
does not yet imply correct autoregressive LM generation. The remaining blocker
is the answer-role decoder/query: the model must learn which answer role to
read at each generated token step without relying on a hidden answer channel.

Do not promote S040 or S003 to L4.
The next orthodox candidate should add a prior-backed pointer-generator style
decoder/cursor pressure or answer-position supervision over the canonical LM
logits, then require full > donor/core_off and renderer/source/primitive
ablations to drop on held-out generation.
```

## Orthodox L4 Repair Audit 2026-05-10

The L4 source-copy path was re-audited after the earlier `Freeze`/copy-collapse
failures. The key finding is that the renderer failure must be debugged as a
causal path contract, not as more token tuning.

Orthodox contracts now enforced:

```text
1. Trainable-only checkpoints must be loaded with recursive base checkpoints.
   Otherwise frozen L3 modules are silently random at probe time.

2. L4 configs must preserve the accepted L3 core architecture.
   The accepted L3 source-pointer checkpoint used the typed-register executor
   and primitive typed selector; disabling them creates a different model.

3. Source-copy rendering must read the accepted primitive recurrent state.
   L3 acceptance was measured on core_primitive_role_value_state_logits, so the
   copy path must be allowed to use that state, not only prompt binder logits or
   a later selector state.

4. Source-position classes are 1-based:
   0 = NULL, 1..N = source slots. Copy-token alignment must preserve that null
   class instead of treating class 0 as source slot 0.
```

Implemented/updated:

```text
scripts/328_probe_qtrm_source_position_logits.py
  Probes prompt source logits, selected renderer copy logits, primitive copy
  logits, valid slot masking, one-based source classes, and recursive checkpoint
  loading.

configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml
  Preserve L3 typed core modules and enable primitive-state source-copy.

src/wgram_lm/wgram_model.py
  Adds primitive-state source-copy selection and keeps one-based NULL alignment.
```

Diagnostic artifacts:

```text
oracle pointer-copy:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/oracle_pointer_copy_lexicalizer_eval128/report.json
  accepted_l1_oracle_pointer_copy, 128/128

pre-repair probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/source_position_logits_probe_current_eval2_one_based/report.json
  raw source binder often selected fixed/invalid positions.

recursive L3-only primitive probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/source_copy_primitive_logits_probe_l3only_recursive_eval2/report.json
  selected primitive state is partially correct but not exact on the source-copy
  lexicalization pair: role accuracy 1/3 on the first two cases.

2-step L4 smoke after contract repair:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_l4_source_copy_pointer_l3preserved_s002_rerun/report.json
  train chunks passed; full multi-mode generation eval was killed, so this is
  not an L4 result.

post-train source-copy probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/source_copy_probe_l3preserved_s002_after_train/report.json
  still rejected; valid primitive pointer role accuracy remains 1/3 on the
  two-case smoke.
```

Current decision:

```text
L3 remains canonical.
L4 remains rejected.

Do not spend long runs on the old renderer settings.
The next valid step is a narrow L3-preserving source-copy adaptation:
  keep recursive checkpoint loading,
  keep accepted L3 core modules enabled,
  first prove source-position logits on source-copy data with the same batch CE
  path that produced L3,
  then train the LM/copy renderer side,
  and require source-copy probe improvement before full generation gates.
```

### Follow-up Repair Findings 2026-05-10

Additional orthodox contract bugs were found and fixed:

```text
1. Final answer role block mismatch.
   Accepted L3 stores final source-position answers in roles 4..7 for the
   10-role state layout. The L4 source-copy probe/renderer had been reading
   roles 0..3, which are the early/initial block. The probe and renderer now
   use the final answer role block.

2. Probe runtime mismatch.
   The source-copy probe now mirrors the L4/L3 source-slot runtime:
   token source slots enabled, source-slot predicate feedback enabled,
   core source-position binder enabled, state straight-through enabled,
   gate_min=1.0, state_gate_min=0.25, and raw source slots enabled.

3. Dataset mismatch.
   `qtrm_source_pointer_l3_hard_eval128` asks for doubled values. It is valid
   for source-position reasoning, but invalid as a source-copy renderer
   acceptance set. Source-copy rendering must be checked on
   `qtrm_source_copy_lexicalization_eval128`, where the final answer is copied
   source values.
```

New diagnostic artifacts:

```text
L3 hard with corrected runtime:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_source_copy_probe_s001_20260510/
  source_copy_probe_runtime_contract_fix.json

  pointer_exact_accuracy: 1.0 on the first 4 L3-hard rows
  copy_answer_accuracy: 0.0 because the task requires doubling, not copying

source-copy lexicalization with corrected runtime:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_source_copy_probe_s001_20260510/
  source_copy_lexicalization_probe_eval8.json

  full pointer_exact_accuracy: 0.625
  full pointer_role_accuracy: 0.875
  source_slot_off/binder_off copy accuracy: 0.0
```

Interpretation:

```text
The causal source-slot/core-binder path is real: source-slot/binder ablations
zero the probe.

The remaining blocker is not donor fluency or MemoryOS. It is source-position
generalization on the source-copy lexicalization distribution, especially the
third output role choosing NULL or a neighboring source class.

Do not promote L4 yet.
Run the existing L3 batch CE trainer on source-copy lexicalization rows first:
scripts/324_train_qtrm_source_pointer_batch.py
  init: accepted_l3_last.pt
  train: data/filtered/qtrm_source_copy_lexicalization_train512.jsonl
  eval: data/eval/qtrm_source_copy_lexicalization_eval128.jsonl
  policy: token_numeric_source_slot_context_binder_primitive_role_value_state_machine

Only after source-copy probe reaches the acceptance range should the expensive
multi-mode generation gate be rerun.
```

### Source-Copy Batch CE Result 2026-05-10

The first valid adaptation used the existing L3 batch trainer rather than a new
renderer trick:

```text
script:
  scripts/324_train_qtrm_source_pointer_batch.py

init:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_position_l3_hard_batch_s240_b8_eval/accepted_l3_last.pt

train:
  data/filtered/qtrm_source_copy_lexicalization_train512.jsonl

eval:
  data/eval/qtrm_source_copy_lexicalization_eval128.jsonl

policy:
  token_numeric_source_slot_context_binder_primitive_role_value_state_machine
```

Artifacts:

```text
10-step stable smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s010_b2_from_l3/train/last.pt

held-out probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s010_b2_from_l3/source_copy_probe_eval32.json

result:
  full_copy_answer_accuracy: 0.8125 on 32 held-out rows
  full_pointer_role_accuracy: 0.9375
  source_slot_off_copy_answer_accuracy: 0.0
  source_binder_off_copy_answer_accuracy: 0.0
```

This is an improvement over the corrected-runtime pre-adaptation probe:

```text
pre-adaptation source-copy eval8:
  full_copy_answer_accuracy: 0.625

post 10-step source-copy eval32:
  full_copy_answer_accuracy: 0.8125
```

The 30-step continuation remained train-stable but did not solve the remaining
held-out pattern on the first 8 rows:

```text
artifact:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s030_b2_from_s010/source_copy_probe_eval8.json

result:
  full_copy_answer_accuracy: 0.75 on 8 held-out rows

dominant error:
  role 5 often predicts the previous source slot, e.g. [1,2,5] instead of
  [1,3,5]. This is a role-order/source-position discrimination issue, not a
  donor fluency issue.
```

Orthodox decision:

```text
L4 remains rejected.
The causal source-position path is now proven useful and ablatable, but not yet
accurate enough for promotion.

Next experiment should target role-5 ordering specifically:
  hard-negative paired permutations,
  role-order contrast / trace contrast,
  or source-position batch CE with stronger paired groups.
Do not rerun full generation gates until the source-copy probe is >=0.95.
```

### Source-Copy Pointer L2 Accepted 2026-05-10

The source-copy pointer path was repaired using the same causal contract that
the probe and renderer read. The important fix was not another renderer trick:
the trainer now can place source-copy targets in the final answer role block
instead of the early/raw role block.

Orthodox-method audit:

```text
Prior family:
  pointer/copy binding with hard-negative paired permutations.

General LLM path:
  visible prompt -> tokenizer/donor states -> source slots -> recursive QTRM
  role state -> source-position logits. No hidden solver computes the answer.

Causal necessity:
  source-slot-off and source-binder-off both collapse to 0/128.

Scope honesty:
  This is L2 source-position/copy-logits acceptance only. It is not L4
  autoregressive LM acceptance.
```

Implementation changes:

```text
scripts/324_train_qtrm_source_pointer_batch.py
  added source-slot hard-negative margin loss
  added --core-role-value-source-copy-answer-role-targets
  shifted source-copy supervision to roles 4..7 for the 10-role answer block

tests/test_qtrm_source_pointer_batch_trainer.py
  added parser, source-margin, and answer-role target tests
```

Training/eval artifacts:

```text
wider source-copy train split:
  data/filtered/qtrm_source_copy_lexicalization_train2048_v0to64_s4326.jsonl
  data/filtered/qtrm_source_copy_lexicalization_train2048_v0to64_s4326.summary.json

best checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s020_b2_answerroles_from_s060/train/last.pt

eval32 probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s020_b2_answerroles_from_s060/
  source_copy_probe_eval32.json

eval128 probe:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s020_b2_answerroles_from_s060/
  source_copy_probe_eval128.json
```

Accepted probe metrics:

```text
eval32:
  full copy answer accuracy: 1.000
  source-slot-off copy answer accuracy: 0.000
  source-binder-off copy answer accuracy: 0.000

eval128:
  full copy answer accuracy: 1.000
  source-slot-off copy answer accuracy: 0.000
  source-binder-off copy answer accuracy: 0.000
```

This establishes the first clean post-L3 prerequisite:

```text
QTRM can bind source positions causally in latent/recurrent state on the
source-copy lexicalization gate.
```

### Post-L2 Generation Smoke 2026-05-10

A small autoregressive generation smoke was run after the L2 pointer acceptance:

```text
artifact:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_answerroles_s020_smoke8/eval.jsonl

status:
  process exited early, so this is partial diagnostic evidence only
```

Partial metrics:

```text
donor_only_no_evidence:                         5/8
qtrm_core_off_no_evidence:                      5/8
qtrm_core_steps_8_no_evidence:                  2/8
source_slot_off:                                5/8
source_binder_off:                              5/8
vocab_renderer_off:                             2/2 partial
```

Interpretation:

```text
The source-position pointer is no longer the blocker.
The blocker moved to L4: primitive/source-copy logits are correct, but the
vocab renderer / donor fusion path degrades autoregressive text generation.

Do not promote L4.
The next proper step is not more source-pointer training. It is a renderer
fusion repair where the accepted pointer logits are injected without
overriding donor fluency or reversing source order.
```

Premature renderer generation smoke confirms the same rule:

```text
artifact:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_renderer_maskfix_s003_smoke8/eval.jsonl

summary:
  donor_only exact: 5/8
  core_off exact: 5/8
  full QTRM exact: 3/8
  primitive_off exact: 3/8
  vocab_renderer_off exact: 5/8

interpretation:
  current renderer integration can degrade donor generation.
  This is not L4 progress; it reinforces that the source-copy pointer probe
  must be fixed before full generation promotion.
```

## Pointer Copy Oracle L1 2026-05-10

Orthodox status:

```text
method_class: official/minimal reproduction
target_level: L1 scaffold
prior_family: pointer-generator / copy attention
canonical_path:
  prompt source tokens -> selected source positions -> pointer/copy LM-token rendering
```

Result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  pointer_copy_oracle_l1_source_copy_eval128/report.json

decision: accepted_l1_pointer_copy_oracle
rows: 128
full_exact_rows: 128
renderer_off_exact_rows: 0
non_empty_full_exact_rows: 128
non_empty_renderer_off_exact_rows: 0
```

Interpretation:

```text
This does not promote QTRM to L4.
It only proves that the source-copy lexicalization data and the minimal
pointer/copy rendering contract are coherent.

The active L4 blocker is now narrower:
  Does QTRM's source-position binder produce the selected source positions
  that the oracle copy renderer requires?
```

Next causal diagnostic:

```text
Probe core_source_position_prompt_logits against oracle selected source
positions on the source-copy eval split.

If source-position alignment is high but generation fails:
  repair the answer-step renderer/query path.

If source-position alignment is low:
  repair the source-position binder interface before any more L4 training.
```

## L4 4090 Executable Smoke 2026-05-10

Orthodox status: prerequisite repair for L4, not L4 promotion.

Why this run was needed:

```text
Earlier L4 source-slot LM-path runs failed with CUDA OOM before the causal
question could be measured. The current runner now uses target-logit-only
training, chunked subprocess training, no self-rollout by default, and a
shorter max-length smoke setup to test whether the L4 gate is operational on a
24GB 4090.
```

Command class:

```text
script: scripts/322_run_source_pointer_l4_lm_path_gate.py
out_dir: /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_l4_source_slot_lm_path_smoke_exec_s002
init: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_source_position_l3_hard_batch_s240_b8_eval/accepted_l3_last.pt
steps: 2, chunk_steps: 1, max_length: 128, eval_cases: 2, max_new_tokens: 4
```

Result:

```text
train_chunk_0001: OK
train_chunk_0002: OK
eval: OK
decision: rejected_l4_candidate

full_generation_accuracy: 0.50
donor_generation_accuracy: 0.50
core_off_generation_accuracy: 0.50
primitive_off_generation_accuracy: 0.50
source_slot_off_generation_accuracy: 0.50
source_binder_off_generation_accuracy: 0.50
bridge_off_generation_accuracy: 0.50
vocab_renderer_off_generation_accuracy: 0.50
answer_recurrent_off_generation_accuracy: 0.50
all full-minus-ablation margins: 0.0
completion deltas versus full: 0/2 for every ablation
```

Interpretation:

```text
The executable-run blocker is repaired: the L4 smoke gate can run on the 4090
without OOM. The architecture claim is still rejected because the full model,
donor-only, core-off, primitive-off, source-slot-off, bridge-off, renderer-off,
and answer-loop-off paths all emit the same completions on the smoke cases.

Example failure: for the prompt asking to keep even values in [49,57,40,55,60]
and double them, the target alias is 80,120 but donor/core/full all emit 98,.
This is a state-to-LM-logit causality failure, not a retrieval, MemoryOS, or
L3 source-state failure.
```

Next bottleneck:

```text
Keep the accepted L3 source-pointer checkpoint canonical, but do not promote
L4. The next falsifier should inspect the state-to-logit bridge directly with
gold-token rank / residual-logit probes before another long training run. If
full, primitive-off, source-slot-off, and renderer-off rank the same gold tokens,
the bridge is not using the accepted L3 state and needs architectural redesign.
```

Gold-token rank probe:

```text
probe: /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_l4_source_slot_lm_path_smoke_exec_s002/gold_rank_probe.jsonl

donor_only:
  content_first_rank_mean = 3.50
  max_rank_mean = 3.50
core_off:
  content_first_rank_mean = 3.50
  max_rank_mean = 3.50
full:
  content_first_rank_mean = 3.50
  max_rank_mean = 3.50
primitive_off/source_slot_off/source_binder_off/bridge_off/renderer_off/
answer_recurrent_off:
  content_first_rank_mean = 3.50
  max_rank_mean = 3.50
```

Hard-token example:

```text
case: source-pointer-l3-s321-range_shift_v32to63-0000
target: 80,120
first content target token: 8

donor/core/full rank the target token 8 at rank 6.
top predicted content tokens are still 9, 4, 5, 1, 2.
```

Decision:

```text
The L4 bridge is not yet state-causal at the token-rank level. More steps on
the same bridge/renderer family are not justified unless a new architectural
change creates a measurable full-vs-ablation rank/logprob margin first.
```

QTRM-only rank probe:

```text
probe: /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_l4_source_slot_lm_path_smoke_exec_s002/gold_rank_probe_qtrm_only.jsonl
override: qtrm_logits_scale=1.0, donor_logits_scale=0.0

qtrm_core_off:
  content_first_rank_mean = 1.00
  invalid zero-logit/tie path; first_unique@1 = 0/2

full:
  content_first_rank_mean = 123733.50
  max_rank_mean = 197962.00

primitive_off:
  content_first_rank_mean = 85913.00

renderer_off:
  content_first_rank_mean = 123738.00
```

Interpretation:

```text
This rejects the hypothesis that QTRM already knows the answer but the donor
overpowers it. Without donor logits, the QTRM renderer ranks the gold content
tokens extremely poorly. The donor is still carrying the normal language/token
distribution, while QTRM has not learned a usable token-space lexicalizer.
```

## Source-Copy Pointer Reset 2026-05-10

Orthodox status:

```text
Roadmap target: L4/general-LM promotion.
Active gate: L2/L3 prerequisite repair for renderer/core causal path.
Classification: faithful QTRM adaptation attempt, rejected as L4 candidate.
```

Mechanism tested:

```text
prompt/chat-template text
-> tokenizer / frozen donor hidden states
-> token numeric source slots
-> source-position binder
-> role-value bridge tokens
-> source-copy residual scattered into tokenizer-id LM logits
-> autoregressive generation
```

Prior mapping:

```text
Pointer/copy and pointer-generator style decoding:
  use source-position attention to bias/copy prompt tokens through the LM logit
  path instead of using a hidden answer channel.
QTRM adaptation:
  the copy residual is driven by the accepted source-position state and is
  ablatable through source-slot, source-binder, and vocab-renderer flags.
```

Current OOM-safe smoke rerun:

```text
run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_l4_source_copy_pointer_current_s002_eval2

config:
  configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml

data:
  data/filtered/qtrm_source_copy_lexicalization_train512.jsonl
  data/eval/qtrm_source_copy_lexicalization_eval128.jsonl

runtime:
  steps=2, chunk_steps=1, max_length=128, max_eval_cases=2

decision:
  rejected_l4_candidate

decisive metrics:
  full_generation_accuracy: 0/2
  donor_generation_accuracy: 0/2
  core_off_generation_accuracy: 0/2
  primitive_off/source_slot_off/source_binder_off/bridge_off/renderer_off: 0/2
  all full-minus-accuracy margins: 0.0
```

Interpretation:

```text
The OOM-safe runner can execute the pointer/copy lexicalization gate, and the
renderer can perturb some completions. It still does not produce a causal
accuracy gain. This reinforces the L4 blocker: QTRM needs a stronger token-space
lexicalizer/decoder before broader LM promotion.
```

Orthodox minimal reproduction:

```text
script:
  scripts/327_eval_oracle_pointer_copy_lexicalizer.py

test:
  tests/test_oracle_pointer_copy_lexicalizer.py

run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/oracle_pointer_copy_lexicalizer_eval128/report.json

result:
  decision: accepted_l1_oracle_pointer_copy
  rows: 128
  full_exact: 128/128
  renderer_off_exact: 0/128
  nonempty_pointer_drop: 1.0
```

Meaning:

```text
The source-copy data and minimal pointer/copy lexicalizer contract are valid:
given correct selected source positions, the renderer can produce exact text
and renderer-off drops completely.

Therefore the L4 failure is not a data-contract failure. The next bottleneck is
replacing oracle positions with QTRM source-position logits at answer time. The
QTRM L4 candidate must prove that those logits point to the same source
positions before another broad LM generation run is meaningful.
```

Run:

```text
config:
  configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
train:
  data/filtered/qtrm_source_copy_hard_token_train64_from_probe.jsonl
eval:
  data/eval/qtrm_source_copy_hard_token_eval64_from_probe.jsonl
checkpoint:
  local_eval/l4_source_copy_hard_pointer_s8/train/last.pt
report:
  local_eval/l4_source_copy_hard_pointer_s8/report.json
rank probe:
  local_eval/l4_source_copy_hard_pointer_s8/gold_rank_probe13.jsonl
```

Generation gate result:

```text
decision: rejected_l4_candidate
full_generation_accuracy: 0/13
donor_generation_accuracy: 0/13
core_off_generation_accuracy: 0/13
all component-off generation accuracies: 0/13
```

Rank probe:

```text
donor-only:
  content_first@1 = 0/13
  content_first_rank_mean = 2.23
  max_rank_mean = 2.69

core-off:
  content_first@1 = 0/13
  content_first_rank_mean = 2.23
  max_rank_mean = 2.69

vocab-renderer-off:
  content_first@1 = 0/13
  content_first_rank_mean = 2.23
  max_rank_mean = 2.69

source-binder-off:
  content_first@1 = 1/13
  content_first_rank_mean = 2.15
  max_rank_mean = 2.62

full:
  content_first@1 = 1/13
  content_first_rank_mean = 2.15
  max_rank_mean = 2.62
```

Rejected claim:

```text
The source-copy pointer renderer did not prove that QTRM core/source binding
causally improves final LM generation.
```

Failed causal condition:

```text
full did not beat generation baselines, and full tied source-binder-off on the
rank probe. The copy path may slightly move logits, but the measured gain is
not attributable to the intended source-position binder.
```

Next falsifiable hypothesis:

```text
The bottleneck is not only "copy source token into vocab"; it is ordered,
multi-token answer planning. The next repair must test whether a recurrent
answer-state loop can maintain output order and comma/token boundaries before
claiming an L4 LM renderer.

Kill criterion:
  if full ties answer-loop-off, source-binder-off, or core-off on the hard
  source-copy rank/generation gate, reject again.
```

This resets the active work to `renderer/core causal-path repair`. It is not
L4 promotion until the same canonical LM path beats donor/core-off and drops
under the relevant component ablations.

## Answer-Loop Future Decoder Probe 2026-05-10

Orthodox status:

```text
Roadmap target: L4/general-LM promotion.
Active gate: L2/L3 prerequisite repair for ordered multi-token answer planning.
Classification: faithful QTRM adaptation attempt, rejected as L4 candidate.
```

Prior-to-implementation contract:

```text
Prior principle:
  recurrent/looped LM decoding and auxiliary lookahead prediction. The answer
  state loop should maintain an internal state for multi-token answer planning,
  while future-token CE is auxiliary pressure, not a hidden answer channel.

QTRM tensor path:
  prompt tokens -> donor hidden states -> QTRM recursive trajectory/source state
  -> answer_state_loop recurrent block / next-token decoder
  -> LM logits -> autoregressive text.

Causal ablation:
  qtrm_core_steps_8_answer_state_recurrent_off_no_evidence
  qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence

Kill criterion:
  if full ties recurrent-off or next-token-decoder-off on generation and rank,
  reject the answer-loop claim.
```

Implementation artifacts:

```text
config:
  configs/qwen35_2b_4090_source_copy_answer_loop_future_decoder_scaffold.yaml
runner changes:
  scripts/322_run_source_pointer_l4_lm_path_gate.py now forwards
  --answer-state-loop-future-token-ce-weight and evaluates
  qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence.
tests:
  tests/test_source_pointer_l4_lm_path_gate.py
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_future_s8
train:
  data/filtered/qtrm_source_copy_hard_token_train64_from_probe.jsonl
eval:
  data/eval/qtrm_source_copy_hard_token_eval64_from_probe.jsonl
steps:
  8
```

Generation result:

```text
decision: rejected_l4_candidate
full_generation_accuracy: 0/13
donor_generation_accuracy: 0/13
core_off_generation_accuracy: 0/13
answer_recurrent_off_generation_accuracy: 0/13
answer_next_token_decoder_off_generation_accuracy: 0/13

completion deltas versus full:
  donor/core-off/source-binder-off/vocab-renderer-off changed 9/13
  primitive-off changed 6/13
  answer_recurrent_off changed 0/13
  answer_next_token_decoder_off changed 0/13
```

Rank probe:

```text
donor/core-off/vocab-renderer-off:
  content_first@1 = 0/13
  content_first_rank_mean = 2.23
  max_rank_mean = 2.69

full:
  content_first@1 = 1/13
  content_first_rank_mean = 2.15
  max_rank_mean = 2.62

answer_recurrent_off:
  content_first@1 = 1/13
  content_first_rank_mean = 2.15
  max_rank_mean = 2.62

answer_next_token_decoder_off:
  content_first@1 = 1/13
  content_first_rank_mean = 2.15
  max_rank_mean = 2.62
```

Rejected claim:

```text
The answer-state recurrent/next-token decoder did not causally improve ordered
multi-token answer rendering. The small rank movement comes from the
vocab-renderer path, not the answer-loop internals.
```

Failed causal condition:

```text
full tied recurrent-off and next-token-decoder-off on both generation and rank.
Therefore the current architecture is not yet a looped-LM answer planner; it is
still a state-to-vocab perturbation path.
```

Next falsifiable hypothesis:

```text
The answer-loop block is downstream but too weakly coupled to the final
residual path. The next repair must make answer-loop logits replace or dominate
the final QTRM residual under a bounded gate, then require:
  full > donor/core-off,
  full > answer_recurrent_off,
  full > answer_next_token_decoder_off,
  full > vocab_renderer_off
on the same hard source-copy gate.
```

## Answer-Loop Additive Residual Probe 2026-05-10

Root-cause check:

```text
The previous future-decoder config had
core_role_value_state_vocab_renderer_replace_residual_enabled=true.
In wgram_model.py this zeroes qtrm_residual_logits immediately before adding
core_role_value_vocab_renderer_logits, so the answer_state_loop_logits computed
earlier are overwritten in the final residual path.
```

Repair:

```text
configs/qwen35_2b_4090_source_copy_answer_loop_future_decoder_scaffold.yaml
sets:
  core_role_value_state_vocab_renderer_replace_residual_enabled: false
```

This makes the path additive:

```text
answer_state_loop_logits + core_role_value_vocab_renderer_logits
-> donor fusion -> autoregressive text
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_additive_s4
steps:
  4
eval cases:
  8
```

Result:

```text
decision: rejected_l4_candidate
full_generation_accuracy: 0/8
donor/core-off/primitive-off/source-off/vocab-off accuracies: 0/8
answer_recurrent_off_generation_accuracy: 0/8
answer_next_token_decoder_off_generation_accuracy: 0/8

completion deltas versus full:
  donor/core-off/primitive-off: 6/8 changed
  source-slot/source-binder/bridge-off: 4/8 changed
  final-binder-off: 3/8 changed
  vocab-renderer-off: 5/8 changed
  answer_recurrent_off: 0/8 changed
  answer_next_token_decoder_off: 0/8 changed
```

Conclusion:

```text
The overwrite issue was real and is fixed in the config, but the answer-loop
internals still are not a causal text-generation component. The learned effect
is carried by source/bridge/vocab paths, not by recurrent answer planning.
```

Next falsifiable hypothesis:

```text
Training pressure is leaking into source/bridge/vocab paths because
role_value_answer_bridge_loop_vocab_renderer_only trains all of them together.
The next repair should isolate answer_state_loop_only or
answer_state_loop_next_token_decoder_only on a rank/generation probe, then
promote back only if recurrent-off or next-token-decoder-off drops.
```

## Answer-Loop-Only Diagnostic 2026-05-10

Orthodox status:

```text
Roadmap target: L4/general-LM promotion.
Active gate: L2/L3 diagnostic for answer-loop causal viability.
Classification: diagnostic probe, not a canonical L4 claim.
```

Reason:

```text
The additive residual probe still routed most pressure through source/bridge/
vocab paths. To test whether the recurrent answer loop can affect generation at
all, trainable_param_policy was narrowed to answer_state_loop_only and renderer/
bridge/primitive contrast losses were set to 0.
```

Run:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_only_s4
trainable_param_policy:
  answer_state_loop_only
steps:
  4
eval cases:
  4
```

Result:

```text
decision: rejected_l4_candidate
full_generation_accuracy: 0/4
donor/core-off/all ablation accuracies: 0/4

completion deltas versus full:
  donor/core-off: 2/4 changed
  primitive-off: 3/4 changed
  source-slot/source-binder/bridge/vocab-renderer-off: 2/4 changed
  answer_state_recurrent_off: 1/4 changed
  answer_next_token_decoder_off: 0/4 changed
```

Conclusion:

```text
The recurrent answer loop is not completely inert when training pressure is
isolated, but it is far too weak and not accurate. The next-token decoder path
still shows no causal generation effect.
```

Rank probe:

```text
donor/core-off:
  content_first@1 = 0/4
  content_first_rank_mean = 2.00
  max_rank_mean = 2.50

full:
  content_first@1 = 0/4
  content_first_rank_mean = 2.00
  max_rank_mean = 2.00

answer_recurrent_off:
  content_first@1 = 0/4
  content_first_rank_mean = 2.00
  max_rank_mean = 2.00

answer_next_token_decoder_off:
  content_first@1 = 0/4
  content_first_rank_mean = 2.00
  max_rank_mean = 2.00

vocab_renderer_off:
  content_first@1 = 0/4
  content_first_rank_mean = 2.00
  max_rank_mean = 2.00
```

Rank conclusion:

```text
The max-rank improvement over donor/core-off does not depend on recurrent-off,
next-token-decoder-off, or vocab-renderer-off. It is therefore not evidence for
a causal answer-loop planner.
```

Next falsifiable hypothesis:

```text
Keep answer_state_loop_only as the diagnostic lane, but add direct rank/CE
pressure to answer_state_loop_logits and evaluate rank before generation. Do
not re-enable source/vocab joint training until answer_recurrent_off produces a
clear held-out rank drop.
```

## Direct Answer-Loop Logit CE 2026-05-10

Orthodox status:

```text
Roadmap target:
  L4/general-LM promotion.

Active gate:
  L2/L3 answer-loop causal-path repair.

Classification:
  diagnostic probe, not L4 promotion.
```

Implementation:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  added --answer-state-loop-logit-ce-weight
  added answer_state_loop_logit_ce_loss(outputs["answer_state_loop_logits"], target_ids)
  validation requires --causal-prefix-supervision and
  model.answer_state_loop_enabled=true

scripts/322_run_source_pointer_l4_lm_path_gate.py
  forwards --answer-state-loop-logit-ce-weight

tests:
  direct helper + parser + runner forwarding + all-prefix training contract
```

Validation:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_source_copy_lexicalization_builder \
  tests.test_gold_token_rank_probe \
  tests.test_hard_token_lexicalization_gate_builder \
  tests.test_pure_recursive_depth_supervised_train_script \
  tests.test_model_config

196 tests OK
```

Run A:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_direct_ce_s8
trainable_param_policy:
  answer_state_loop_only
steps:
  8
losses:
  answer_state_loop_logit_ce=1.0
  answer_state_loop_future_token_ce=0.5
result:
  rejected_l4_candidate
```

Decisive metrics:

```text
full_generation_accuracy: 0.25
donor_generation_accuracy: 0.25
core_off_generation_accuracy: 0.25
answer_recurrent_off_generation_accuracy: 0.25
answer_next_token_decoder_off_generation_accuracy: 0.25

full_minus_donor: 0.00
full_minus_core_off: 0.00
full_minus_answer_recurrent_off: 0.00
full_minus_answer_next_token_decoder_off: 0.00
```

Completion deltas:

```text
answer_state_recurrent_off changed 2/4 completions
answer_next_token_decoder_off changed 1/4 completions
```

Rank probe:

```text
fused donor+QTRM logits:
  full content_first_rank_mean = 3.50
  donor/core-off content_first_rank_mean = 3.25
  answer_recurrent_off content_first_rank_mean = 3.50
  answer_next_token_decoder_off content_first_rank_mean = 3.25

qtrm-only diagnostic:
  full and answer-loop ablations rank the gold token extremely poorly
  core-off qtrm-only is an invalid zero-logit tie, not a real success
```

Interpretation:

```text
Direct CE makes answer-loop ablations perturb text, but it does not make the
answer-loop improve gold-token rank or greedy accuracy. The fused donor path is
not helped, and the QTRM-only residual is not a learned answer policy.
```

Run B:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_direct_ce_skipws_s24
change:
  --causal-prefix-skip-leading-whitespace-targets
steps:
  24
result:
  rejected_l4_candidate
```

Decisive metrics:

```text
full_generation_accuracy: 0.00
donor/core_off_generation_accuracy: 0.25
answer_recurrent_off_generation_accuracy: 0.00
answer_next_token_decoder_off_generation_accuracy: 0.00

full_minus_donor: -0.25
full_minus_core_off: -0.25
full_minus_answer_recurrent_off: 0.00
full_minus_answer_next_token_decoder_off: 0.00
```

Rank probe:

```text
full content_first_rank_mean = 3.50
answer_recurrent_off content_first_rank_mean = 3.00
answer_next_token_decoder_off content_first_rank_mean = 3.50
donor/core-off content_first_rank_mean = 3.25
```

Interpretation:

```text
Skipping leading whitespace targets did not fix the causal path. The recurrent
block became mildly harmful on rank, not helpful.
```

Training-contract repair:

```text
answer_state_loop_logit_ce was originally limited to example_index == 0.
That was wrong for the autoregressive LM path: generation queries the loop on
every generated-prefix state. The CE is now applied to every causal-prefix
example; future-token CE remains prompt-only because it is an auxiliary
lookahead head, not the runtime LM path.
```

Run C:

```text
out:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/l4_source_copy_answer_loop_direct_ce_allprefix_skipws_s24
change:
  all-prefix answer_state_loop_logit_ce
  --causal-prefix-skip-leading-whitespace-targets
steps:
  24
result:
  rejected_l4_candidate
```

Decisive metrics:

```text
full_generation_accuracy: 0.00
donor_generation_accuracy: 0.25
core_off_generation_accuracy: 0.25
primitive_off_generation_accuracy: 0.25
answer_recurrent_off_generation_accuracy: 0.00
answer_next_token_decoder_off_generation_accuracy: 0.00

full_minus_donor: -0.25
full_minus_core_off: -0.25
full_minus_primitive_off: -0.25
full_minus_answer_recurrent_off: 0.00
full_minus_answer_next_token_decoder_off: 0.00
```

Conclusion:

```text
The direct answer-loop residual-logit path remains rejected. The loop can
perturb generation, but it is not an accepted causal LM renderer because full
does not beat donor/core-off and answer-loop ablations do not remove a real
held-out gain.
```

Next falsifiable hypothesis:

```text
Stop treating answer_state_loop_logits as the canonical L4 renderer. The more
orthodox donor-preserving path is the soft-prefix/teacher-forced-to-greedy
renderer reset, because that prior family already produced an accepted L1
greedy-generation scaffold on arithmetic. The next promotion attempt should
improve that path on heldout18 before returning to mixed source-pointer L4.
```

## Donor-Aligned LM-Head Renderer Probe 2026-05-10

Orthodox status:

```text
Roadmap target:
  L4/general-LM promotion.

Active gate:
  L2/L3 prerequisite repair for lexicalization causality.

Why this is prerequisite repair:
  L4 cannot be interpreted until QTRM full beats donor/core-off and a
  renderer/source/primitive ablation drops the same held-out LM metric.
```

Prior-to-implementation contract:

```text
Prior principle:
  Use the donor token embedding/unembedding geometry as the lexicalization
  prior instead of asking a random low-rank renderer to discover the whole
  vocabulary map from weak supervision.

QTRM tensor path:
  accepted L3 source/role/value state -> source-state renderer tokens ->
  lm_head-based residual logits -> autoregressive text.

Causal ablation:
  full must improve over donor/core-off and drop under primitive-off,
  source-binder-off, or vocab-renderer-off.

Kill criterion:
  if full == donor/core-off or component-off ties full on generation/rank,
  reject as non-causal for L4.
```

Implementation:

```text
checkpoint:
  local_eval/l4_donor_aligned_init/last.pt

config:
  local_eval/l4_donor_aligned_init/config.yaml

delta run:
  local_eval/l4_donor_aligned_s2_smoke/train/last.pt
```

The aligned checkpoint was built by resolving the accepted L3 base-chain, then
initializing `lm_head.weight` from the Qwen donor output embedding projected
into the QTRM hidden space. The config sets:

```yaml
tie_embeddings: false
core_role_value_state_vocab_renderer_use_lm_head: true
```

Smoke generation result:

```text
decision: rejected_l4_candidate
full_generation_accuracy: 0.5
donor_generation_accuracy: 0.5
core_off_generation_accuracy: 0.5
primitive_off_generation_accuracy: 0.5
source_binder_off_generation_accuracy: 0.5
vocab_renderer_off_generation_accuracy: 0.5
all full_minus_* margins: 0.0
```

Training diagnostics:

```text
core_role_value_vocab_renderer_ce: 12.41 -> 12.45
core_role_value_vocab_renderer_acc: 0.0
source_binder_target_logp_delta: about -0.01
primitive_target_logp_delta: -0.023 -> +0.005
```

Gold-token rank probe after fixing the probe to ignore the leading whitespace
token:

```text
probe:
  local_eval/l4_donor_aligned_s2_smoke/gold_rank_probe.jsonl

max_cases: 2
modes:
  donor_only_no_evidence
  qtrm_core_off_no_evidence
  qtrm_core_steps_8_no_evidence
  qtrm_core_steps_8_primitive_role_value_off_no_evidence
  qtrm_core_steps_8_core_source_position_binder_off_no_evidence
  qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence

result:
  all modes content_first@1 = 1/2
  all modes content_first_rank_mean = 3.50
  all modes all<=10 = 2/2
```

Interpretation:

```text
The donor-aligned renderer is technically wired, but it is not causally used.
The first visible answer token is already near the top under donor/core-off,
and QTRM does not improve the rank or generation result. Turning off primitive,
source binder, or renderer does not hurt the measured metric.
```

Decision:

```text
reject donor-aligned lm_head renderer as L4 candidate.
do not run longer S8/S80 until a harder lexicalization gate shows a causal
rank/logprob margin that donor/core-off cannot match.
```

Next hypothesis:

```text
The current L4 gate mixes two problems:
  1. donor often already puts easy numeric answer tokens near the top;
  2. QTRM does not add a causal margin on hard tokens where donor is wrong.

The next repair must use a hard-token lexicalization gate:
  select cases/tokens where donor/core-off ranks the gold content token poorly,
  train only a canonical prompt -> state/core -> LM-logit path,
  and accept only if full improves those ranks while primitive/source/renderer
  ablations drop them.
```

## Hard-Token Lexicalization Repair 2026-05-10

Hard-token gate construction:

```text
train donor rank probe:
  local_eval/l4_donor_aligned_s2_smoke/train_gold_rank_probe_64.jsonl

train hard subset:
  data/filtered/qtrm_source_pointer_l4_hard_token_train64_from_probe.jsonl
  selected 23 / 512 cases

eval hard subset:
  data/eval/qtrm_source_pointer_l4_hard_token_eval16_from_probe.jsonl
  selected 6 / 128 cases
```

The gate selects cases where donor-only ranks the first non-whitespace answer
token below top-1. This avoids the misleading leading-space token and removes
easy cases where donor/core-off already puts the answer token near the top.

Hard-token s8 repair result:

```text
run:
  local_eval/l4_donor_aligned_hard_token_s8

decision:
  rejected_l4_candidate

full_generation_accuracy:
  1/6

donor_generation_accuracy:
  0/6

core_off_generation_accuracy:
  0/6

primitive_off/source_slot_off/source_binder_off/bridge_off/
final_binder_off/vocab_renderer_off/answer_recurrent_off/
answer_halt_gate_off:
  all 1/6
```

Interpretation:

```text
This is the first hard-token run where full beats donor/core-off on generation.
However, every claimed primitive/source/renderer/answer-loop ablation ties the
full model. The 1/6 improvement is therefore a generic core-residual effect,
not a proven primitive/source renderer effect.
```

Additional answer-loop ablations:

```text
file:
  local_eval/l4_donor_aligned_hard_token_s8/hard_token_answer_loop_extra_ablation.jsonl

full:
  1/6
answer_selective_context_off:
  1/6
answer_hidden_bridge_off:
  1/6
answer_next_token_decoder_off:
  1/6
vocab_renderer_off:
  1/6
```

Decision:

```text
reject as L4.
keep as useful evidence that hard-token focusing can beat donor/core-off,
but do not attribute the gain to the intended source/primitive renderer path.
```

Next causal hypothesis:

```text
The current QTRM residual path is too broad. Even when the vocab renderer or
primitive/source binders are disabled, a generic core residual can still produce
the same held-out hit. The next repair should add a forced renderer bottleneck:

  qtrm base residual -> zeroed or bypassed for this gate
  source/primitive/role-value state -> vocab renderer -> LM residual logits

Accept only if:
  full > donor/core-off
  vocab_renderer_off drops
  primitive/source-binder-off drops
```

## Forced Renderer Bottleneck Result 2026-05-10

Implementation:

```text
config flag:
  core_role_value_state_vocab_renderer_replace_residual_enabled

config:
  configs/qwen35_2b_4090_source_pointer_l4_hard_token_renderer_bottleneck.yaml

run:
  local_eval/l4_renderer_bottleneck_hard_token_s8

trainable policy:
  role_value_vocab_renderer_only
```

Purpose:

```text
Prevent the generic QTRM residual or answer-loop residual from producing a
held-out hit that survives renderer/source/primitive ablations. For this gate,
the QTRM residual is zeroed before adding the role-value vocab renderer.
```

Unit proof:

```text
tests.test_source_pointer_l4_lm_path_gate.
  test_l4_vocab_renderer_can_replace_generic_qtrm_residual

The test verifies:
  qtrm_residual_logits == core_role_value_vocab_renderer_logits
  renderer_off -> qtrm_residual_logits == 0
```

Generation result:

```text
decision:
  rejected_l4_candidate

full:
  0/6

donor/core-off:
  0/6

all primitive/source/renderer/answer-loop ablations:
  0/6
```

Gold-token rank result:

```text
probe:
  local_eval/l4_renderer_bottleneck_hard_token_s8/gold_rank_probe6.jsonl

donor/core-off/vocab-renderer-off:
  content_first@1 = 0/6
  content_first_rank_mean = 4.33

full:
  content_first@1 = 0/6
  content_first_rank_mean = 4.67

source-binder-off:
  content_first@1 = 0/6
  content_first_rank_mean = 4.67
```

Training signal:

```text
core_role_value_vocab_renderer_acc briefly rose up to 0.5 on tiny train
batches, but held-out generation and rank did not improve.
```

Interpretation:

```text
The forced bottleneck correctly removed the false positive from the broad core
residual path. However, the current low-rank/lm_head vocab renderer is too weak
or too weakly supervised to lexicalize hard numeric answers from the accepted
L3 state in 8 steps.
```

Decision:

```text
Do not promote.
Do not run longer until a cheaper rank/logprob gate shows the renderer can
improve hard content-token ranks over donor/core-off.
```

Next architecture direction:

```text
Replace the broad full-vocabulary renderer with a prior-backed copy/generate
or candidate-token bottleneck that still ends in LM logits:

  prompt tokens/source states -> recurrent primitive state
  -> small candidate token set or pointer/copy distribution
  -> projected LM-logit residual

Acceptance must still require:
  full > donor/core-off
  renderer_off drop
  primitive/source-binder-off drop
  no hidden solver or external answer channel
```

## Candidate-Token Renderer Repair 2026-05-10

Orthodox status:

```text
Roadmap target:
  L4/general-LM promotion.

Active gate:
  L2/L3 prerequisite repair for hard-token lexicalization.

Method classification:
  diagnostic probe, not a promoted architecture.
```

Why this is only a probe:

```text
Candidate-token masking can isolate whether the accepted latent state can move
the right answer token when the output space is small. It is not itself a
general LLM solution, because a fixed candidate vocabulary is evaluator-shaped.
It can only justify the next step if the canonical LM-logit path improves and
the source/primitive/renderer ablations drop.
```

Prior mapping:

```text
Prior family:
  pointer/copy and copy-generate lexicalization.

QTRM adaptation:
  prompt/donor hidden states -> source-position/primitive recurrent state ->
  candidate-token LM residual logits -> autoregressive text.

Shortcut exclusion:
  no external answer solver; reject if donor/core-off ties full or if
  primitive/source/renderer-off ties full.
```

Candidate-token S8 result:

```text
run:
  local_eval/l4_candidate_token_renderer_hard_token_s8

decision:
  rejected_l4_candidate

generation:
  all modes 0/6

gold-rank probe:
  donor/core-off/vocab-off content_first_rank_mean = 4.33
  full/source-binder-off content_first_rank_mean = 4.83
```

Interpretation:

```text
The candidate bottleneck did not improve held-out hard-token rank. The next
small repair checks a specific supervision mismatch: the hard-token gate is
selected on the first non-whitespace answer token, but training still spends
its first target step on the leading whitespace token.
```

Content-first S8 result:

```text
run:
  local_eval/l4_candidate_token_contentfirst_s8

training change:
  --causal-prefix-skip-leading-whitespace-targets

generation decision:
  rejected_l4_candidate

generation:
  full/donor/core-off/all ablations = 0/6

gold-rank probe:
  donor/core-off/vocab-renderer-off content_first_rank_mean = 4.33
  full/source-binder-off content_first_rank_mean = 4.83
  content_first@1 = 0/6 for all modes
```

Interpretation:

```text
Skipping the leading whitespace target did not repair lexicalization. Full QTRM
slightly improves the trivial leading whitespace rank but worsens the first
content-token rank relative to donor/core-off. The source-binder-off path ties
full, so the accepted L3 source state is still not causally improving LM token
selection.
```

Decision:

```text
Reject candidate-token/content-first repair.
Do not spend more runs on fixed candidate vocab masking. It is evaluator-shaped
and did not produce the required causal rank or generation gain.
```

Next experiment gate:

```text
Target level:
  L2/L3 repair, not L4 promotion.

Major bottleneck:
  recurrent/source state to first content answer token.

Baseline to beat:
  donor-only and core-off on hard-token rank/generation.

Required ablation drop:
  vocab-renderer-off, source-binder-off, or primitive-off must reduce the same
  hard-token metric.

Perturbation/held-out split:
  data/eval/qtrm_source_pointer_l4_hard_token_eval16_from_probe.jsonl

Kill decision if fail:
  reject candidate-token/content-first repair and reset to a more faithful
  copy/pointer or donor-token-space reproduction before more local tuning.
```

Actual kill decision:

```text
Triggered. Reset to a faithful pointer/copy or donor-token-space reproduction
before attempting another integrated L4 renderer run.
```

## Pointer/Copy Renderer Reset 2026-05-10

Orthodox status:

```text
Target:
  faithful reproduction scaffold before another L4 renderer attempt.

Prior family:
  pointer/copy and copy-generate lexicalization.

Scope:
  L2/L3 prerequisite repair, not L4 promotion.
```

Implementation scaffold:

```text
model flag:
  core_role_value_state_vocab_renderer_source_copy_enabled

path:
  source-position logits + role bridge state
  -> scatter residual logits onto prompt/source token ids
  -> normal LM logits/autoregressive generation

unit test:
  test_l4_vocab_renderer_can_scatter_source_copy_logits_to_prompt_tokens
```

Critical alignment finding:

```text
The accepted source-slot L3 state predicts list-element/source-slot positions.
A faithful pointer/copy renderer copies prompt token positions or tokenizer ids.
Those are not the same coordinate system.
```

Therefore:

```text
Do not claim source-copy success by scattering numeric source-slot ids as token
ids. That would be another evaluator-shaped shortcut.

Before the next training run, build or adapt a gate where the supervised
source-position target aligns with actual prompt/token positions, or add an
explicit donor-token-space lexicalizer whose mapping is part of the model path
and ablated.
```

New data scaffold:

```text
script:
  scripts/326_build_source_copy_lexicalization_gate.py

purpose:
  create a source-copy task where answers are copied even values, not doubled
  transformed values.

artifacts:
  data/filtered/qtrm_source_copy_lexicalization_train512.jsonl
  data/eval/qtrm_source_copy_lexicalization_eval128.jsonl
  data/filtered/qtrm_source_copy_lexicalization_train512.summary.json

rows:
  train 512, eval 128

remaining prerequisite:
  align source-position labels to tokenizer/prompt positions before using this
  as a canonical pointer/copy reproduction gate.
```

Target semantics fix:

```text
source-copy rows set:
  role_value_source_copy_no_doubled: true

Reason:
  existing role-value target code assumed depth>1 means doubled values.
  The source-copy gate must keep source-position targets on copied even values
  at every depth.
```

Source-slot tokenizer-id alignment:

```text
Added:
  token_numeric_source_slot_token_ids

Reason:
  source-slot class ids are value+1 labels, not tokenizer ids. The copy
  renderer must scatter onto the actual tokenizer id observed in the prompt,
  otherwise source-copy becomes an evaluator-shaped shortcut.

Updated paths:
  training: scripts/196_train_pure_recursive_depth_supervised.py
  eval: scripts/192_eval_raw_intelligence.py
  rank probe: scripts/247_probe_qtrm_gold_token_ranks.py
```

First source-copy pointer smoke:

```text
config:
  configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml

run:
  local_eval/l4_source_copy_pointer_s4

data:
  data/filtered/qtrm_source_copy_lexicalization_train512.jsonl
  data/eval/qtrm_source_copy_lexicalization_eval128.jsonl

generation:
  full/donor/core-off/all ablations = 0/6

completion deltas:
  primitive-off changed 6/6
  vocab-renderer-off changed 3/6
  donor/core-off changed 3/6
```

Rank probe:

```text
file:
  local_eval/l4_source_copy_pointer_s4/gold_rank_probe6.jsonl

donor/core-off/vocab-renderer-off:
  content_first@1 = 4/6
  content_first_rank_mean = 1.33
  max_rank_mean = 1.83

full/source-binder-off:
  content_first@1 = 4/6
  content_first_rank_mean = 1.33
  max_rank_mean = 1.67
```

Interpretation:

```text
The copy path now changes completions and slightly improves max token rank, but
it is not yet causally attributable to the source binder because source-binder
off ties full. The eval slice is also too easy for donor/core-off on the first
content token, so broad source-copy eval is not a decisive gate.
```

Next required gate:

```text
Build hard source-copy lexicalization splits where donor/core-off fail the
content answer token rank, then require:
  full > donor/core-off
  vocab-renderer-off drops
  source-binder-off drops
```

## Orthodox Method Audit 2026-05-10

Current answer to "is this orthodox or hacky?":

```text
The validation method is orthodox.
The current L4 architecture candidate is not yet promoted.
```

Why the method is orthodox:

```text
1. Prior family:
   recurrent latent reasoning + pointer/copy/state-to-vocab rendering.

2. Universal LLM path:
   prompt/chat text -> tokenizer/donor hidden states -> QTRM latent/core path
   -> LM logits -> autoregressive text.

3. Causal ablation:
   every L4 claim is rejected unless full generation beats donor-only/core-off
   and drops when primitive/source/renderer paths are disabled.

4. Baseline pressure:
   donor-only, core-off, primitive-off, source-slot-off, source-binder-off,
   bridge-off, renderer-off, recurrent-off, and halt-gate-off are evaluated.

5. Scope honesty:
   L3 source-position reasoning remains canonical.
   L4 general LM rendering remains rejected until causal drops appear.
```

The latest `typed_register` bypass repair disabled the typed-register executor
and typed primitive selector in the L4 bridge config. This was necessary because
those modules can behave like a diagnostic shortcut around the accepted L3
primitive/source-state path.

Smoke result:

```text
run: local_eval/l4_source_slot_renderer_typedoff_s2_smoke
decision: rejected_l4_candidate
full_generation_accuracy: 0.5
donor_generation_accuracy: 0.5
core_off_generation_accuracy: 0.5
primitive_off_generation_accuracy: 0.5
source_slot_off_generation_accuracy: 0.5
source_binder_off_generation_accuracy: 0.5
all ablation drops: 0.0
```

Interpretation:

```text
Disabling the bypass did not yet make the primitive/source state causally
necessary for generation. Therefore the L4 candidate is still not a model
architecture advance. It remains a diagnostic bridge/renderer experiment.
```

Next orthodox step:

```text
Do not tune this as if it is accepted.
Design the next renderer so that the accepted L3 source-position state is a
direct input to the canonical LM logits, then require primitive/source/renderer
ablations to reduce held-out generation accuracy before any L4 claim.
```

Follow-up 8-step result:

```text
run: local_eval/l4_source_slot_renderer_typedoff_s8_eval8
decision: rejected_l4_candidate
full_generation_accuracy: 0.125
donor_generation_accuracy: 0.125
core_off_generation_accuracy: 0.125
primitive_off_generation_accuracy: 0.125
source_slot_off_generation_accuracy: 0.125
source_binder_off_generation_accuracy: 0.125
bridge_off_generation_accuracy: 0.125
vocab_renderer_off_generation_accuracy: 0.125
all accuracy drops: 0.0
```

Observed deltas:

```text
full vs donor/core-off changed completions on 4/8 cases,
full vs primitive-off changed completions on 3/8,
full vs renderer-off changed completions on 1/8,
but no mode changed hit accuracy.
```

Conclusion:

```text
The L4 path can perturb surface completions, but it does not yet improve held-
out answers or make primitive/source state causally necessary. This is an
orthodox negative result, not an accepted architecture.
```

## Direct Source-State Renderer Candidate 2026-05-10

Prior-To-Implementation Contract:

```text
Prior principle:
  Pointer/copy attention and state-to-vocab residual rendering. The model
  should not receive an external answer; it receives latent source-position
  state derived from the same prompt stream and uses it inside the LM-logit
  path.

QTRM tensor path:
  prompt/chat text -> token numeric source slots -> source-position binder
  logits -> source-position state delta tokens -> vocab-renderer state memory
  -> LM residual logits -> autoregressive text.

  The existing primitive role-value state remains the reasoning path through
  core_role_value_state_answer_bridge tokens. The new source-state tokens only
  expose the accepted L3 source binding directly to the renderer, because the
  previous bridge made source_slot_off/source_binder_off non-causal.

Causal ablation:
  qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence,
  qtrm_core_steps_8_core_source_position_binder_off_no_evidence,
  qtrm_core_steps_8_primitive_role_value_off_no_evidence, and
  qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence must reduce
  held-out generation accuracy or target-token probability.

Shortcut risk:
  The renderer could become a source-copy patch. Reject unless full generation
  improves toward final answers and primitive-off also drops.

Kill criterion:
  If full == donor/core-off or source_slot/source_binder/primitive/renderer
  ablations tie full accuracy on the held-out gate, the candidate remains a
  diagnostic L4 repair attempt, not a promoted general-LM architecture.
```

Implementation result:

```text
Added:
  model.core_role_value_state_vocab_renderer_source_state_tokens_enabled
  L4 config enables direct source-state renderer tokens
  source-binder contrast loss for renderer target log-prob

Unit evidence:
  test_l4_vocab_renderer_can_read_direct_source_state_tokens passes.
  Disabling the source binder changes renderer logits in a small forward test.
```

Gate results:

```text
run: local_eval/l4_source_state_renderer_s2_smoke
decision: rejected_l4_candidate
generation: full == donor == core_off == primitive_off == source_off == renderer_off

run: local_eval/l4_source_state_contrast_s2_smoke
decision: rejected_l4_candidate
source-binder contrast metric exists but generation remains tied.

run: local_eval/l4_source_state_contrast_s8_eval8
decision: rejected_l4_candidate
full_generation_accuracy: 0.125
donor_generation_accuracy: 0.125
core_off_generation_accuracy: 0.125
primitive_off_generation_accuracy: 0.125
source_slot_off_generation_accuracy: 0.125
source_binder_off_generation_accuracy: 0.125
vocab_renderer_off_generation_accuracy: 0.125
all accuracy drops: 0.0
```

Training diagnostic:

```text
core_role_value_vocab_renderer_source_binder_final_target_logp_delta stayed
near zero and slightly negative through 8 steps. The source-state tensor reaches
the renderer, but the current low-rank vocab renderer does not learn to use it
for held-out answer tokens.
```

Conclusion:

```text
Direct source-state context fixed a local tensor-path hole but not the L4
renderer bottleneck. The remaining problem is lexicalization: converting
accepted latent source/primitive value state into normal autoregressive LM
tokens without becoming a task-specific hidden solver.
```

Next candidate:

```text
Replace the generic low-rank state-to-vocab residual with a lexicalization
bridge that maps latent value/state tokens into donor-token space through a
copy/generate style residual. It must still emit normal LM logits and must be
rejected unless primitive/source/renderer ablations reduce held-out generation
or causal forced-choice target probability.
```

## L4 Candidate Result 2026-05-09

After the source-pointer primitive-core L3 checkpoint was accepted, the next
gate tested whether that latent state can improve the normal autoregressive LM
path.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml
runner:
  scripts/322_run_source_pointer_l4_lm_path_gate.py
init checkpoint:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l3_source_pointer_roles12_null_tune_s200/train/
  accepted_l3_primitive_core_null_step_000050.pt
report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_source_pointer_lm_path_s080/report.json
```

Decision:

```text
decision: rejected_l4_candidate
accepted: false
full generation accuracy: 3/32 = 0.09375
donor-only generation accuracy: 5/32 = 0.15625
core-off generation accuracy: 5/32 = 0.15625
primitive-off generation accuracy: 3/32 = 0.09375
bridge-off generation accuracy: 4/32 = 0.125
```

The causal forced-choice diagnostic also rejected the LM-path claim:

```text
causal forced-choice full: 10/32 = 0.3125
donor-only: 10/32 = 0.3125
core-off: 10/32 = 0.3125
primitive-off: 10/32 = 0.3125
bridge-off: 10/32 = 0.3125
```

A donor-scale sweep did not rescue generation:

```text
full default: 3/32 = 0.09375
full donor_scale_0.25: 0/32
full qtrm_scale_1.0 donor_scale_0.25: 0/32
full qtrm_only: 0/32
```

Conclusion: L4 is not blocked by donor scale alone. The current answer bridge
does not causally transfer the accepted L3 primitive recurrent state into LM
token logits. Keep L3 as the canonical accepted baseline and treat bottleneck 4
as open: the next architecture work must redesign the state-to-token renderer
instead of adding memory, retrieval, or broader ASI claims.

## L4 Runner Source-Slot Repair 2026-05-10

Orthodox status:
  prerequisite repair for the L4 LM-path gate, not L4 promotion.

Why lower-gate work is justified:
  The latest accepted L3 checkpoint is the source-slot/source-binder recurrent
  state gate:

```text
/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
qtrm_source_position_l3_hard_batch_s240_b8_eval/accepted_l3_last.pt
```

The older L4 runner still defaulted to the pre-source-slot L3 checkpoint and
the legacy token-numeric-value feature path. That makes any L4 result
ambiguous: a generation win could bypass the accepted L3 source-slot causal
path, and a generation failure could be a runner mismatch rather than a true
state-to-token bottleneck.

Repair:

```text
scripts/322_run_source_pointer_l4_lm_path_gate.py
  default init checkpoint -> latest accepted source-slot L3 checkpoint
  train/eval command -> token-numeric-source-slots
  train/eval command -> source-slots-only + raw-source-slots binder
  L4 decision -> require source-slot-off and source-binder-off drops

scripts/192_eval_raw_intelligence.py
  generation eval can now build prompt-derived source-slot tensors
  eval modes can ablate token_numeric_source_slots and core_source_position_binder

tests/test_source_pointer_l4_lm_path_gate.py
  locks the updated checkpoint, command flags, and ablation modes
```

What would make it non-orthodox:
  claiming L4 success from this repair alone. The repair only makes the next L4
  gate interpretable. L4 still requires full generation to beat donor-only and
  core-off while dropping under primitive-off, source-slot-off,
  source-binder-off, and renderer/bridge-off ablations.

Validation:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_source_pointer_l3_hard_gate \
  tests.test_qtrm_source_pointer_batch_trainer \
  tests.test_qtrm_source_pointer_state_gate \
  tests.test_raw_intelligence_eval_script

29 tests OK
```

## L4 Training Runtime And Renderer Repair 2026-05-10

Orthodox status:
  prerequisite repair for the L4 LM-path gate, not L4 promotion.

Failure found:

```text
single-process L4 training failed on step 2:
  first blocker: full-sequence lm_head(seq) full-vocab logits OOM
  second blocker after target-logit repair: self-rollout extra forwards left
      too little VRAM for the next core FFN allocation

runner/config mismatch:
  L4 runner trained with core_role_value_vocab_renderer_* losses and required
  vocab-renderer-off drops, but the default L4 config did not enable
  core_role_value_state_vocab_renderer_enabled and the trainable policy did
  not include core_role_value_state_vocab_renderer_* parameters.
```

Repair:

```text
src/wgram_lm/wgram_model.py
  forward(logit_token_indices=...) computes LM logits only for supervised
  target positions during training. Autoregressive eval/generation remains on
  the normal full path.

scripts/196_train_pure_recursive_depth_supervised.py
  --target-logit-positions-only uses the target-position logits for
  final-path causal-prefix CE and matching ablation contrasts.

scripts/322_run_source_pointer_l4_lm_path_gate.py
  L4 training now passes --target-logit-positions-only.
  self-rollout default is 0.0 for the 4090 gate; rollout is optional because it
  triples the forward count and can hide the causal-path result behind VRAM
  failure.

configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml
  enables core_role_value_state_vocab_renderer_*.
  default trainable policy is role_value_answer_bridge_loop_vocab_renderer_only.

src/wgram_lm/training/train.py
  adds role_value_answer_bridge_loop_vocab_renderer_only.
```

Validation:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_training_checkpoint_init

129 tests OK

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_source_pointer_l3_hard_gate \
  tests.test_qtrm_source_pointer_batch_trainer \
  tests.test_qtrm_source_pointer_state_gate \
  tests.test_raw_intelligence_eval_script

36 tests OK
```

Runtime evidence:

```text
local_eval/l4_source_slot_target_logits_adamw_no_rollout_s2_smoke/report.json
  train: OK
  eval: OK
  decision: rejected_l4_candidate

local_eval/l4_source_slot_renderer_on_s8_eval8/report.json
  train: OK
  eval: OK
  decision: rejected_l4_candidate
  full_generation_accuracy: 0.125
  donor_generation_accuracy: 0.125
  core_off_generation_accuracy: 0.125
  primitive/source-slot/source-binder drops: 0.0
  bridge_off_accuracy: 0.25
```

Interpretation:

```text
The L4 gate is now executable and the renderer/bridge path is real, but L4 is
not accepted. The full model still ties donor-only and core-off on accuracy.
More importantly, primitive-off, source-slot-off, and source-binder-off still
do not reduce the metric. Therefore the accepted L3 source/primitive state is
not yet causal in the autoregressive LM answer.
```

Next bottleneck:

```text
The state-to-token renderer can influence surface completions, but the current
supervision does not force it to depend on the accepted L3 primitive/source
state. The next experiment must target primitive/source-conditioned rendering,
not MemoryOS, retrieval, larger donors, or broader instruction data.
```

## Source-Copy Span Lexicalization Repair 2026-05-10

Orthodox status: prerequisite repair for the L4 LM-path gate, not L4 promotion.

Failure narrowed:

```text
The source-position state can be exact, but the copy renderer was given only
the first tokenizer piece of each source slot.

Example under Qwen tokenization:
  source slot 44 -> ["4", "4"]
  old copy coordinate -> ["4"]
```

Implementation:

```text
src/wgram_lm/algorithmic_value_state.py
  token_numeric_source_slot_token_spans(...)

src/wgram_lm/wgram_model.py
  forward(... token_numeric_source_slot_token_span_ids, mask ...)
  _compute_source_copy_span_next_token_ids(...)
  span-aware source-copy logits inside
  _compute_core_role_value_state_vocab_renderer_logits(...)

scripts/192_eval_raw_intelligence.py
  generation and forced-choice paths now pass source slot token spans when
  --token-numeric-source-slots is enabled.

configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
  core_role_value_state_vocab_renderer_source_copy_span_enabled: true
```

This preserves the universal LM causal path:

```text
prompt/chat text
-> tokenizer offsets
-> compact source slots + source token spans
-> QTRM source-position state
-> source-copy logits
-> autoregressive LM generation
```

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

Smoke result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_spancopy_smoke8/eval.jsonl

completed rows:
  donor_only: 5/8
  core_off: 5/8
  full QTRM: 3/8
  vocab_renderer_off: 3/3 partial
```

Decision:

```text
Rejected for promotion. Span-copy improves strict S040 full generation from
2/8 to 3/8, but full still loses to donor/core-off. The remaining bottleneck is
answer-role/order selection during autoregressive generation, not single-token
lexicalization alone.
```

## Answer-Role Cursor Candidate 2026-05-10

Orthodox status: L4 candidate, not accepted L4.

Prior family:

```text
pointer-generator / copy-attention decoder cursor
```

Why this is different from the rejected cursor:

```text
rejected:
  counted source token ids in the visible prefix.
  This breaks on multi-token values and duplicate digit tokens.

new candidate:
  counts completed source token spans beyond the prompt source list.
  A just-completed span receives no copy bias so the LM can emit a separator.
  After a separator, the cursor advances to the next answer role.
```

Canonical path:

```text
source token spans from prompt
-> QTRM source-position/primitive state
-> answer-role cursor over visible generated prefix
-> source-copy LM logits
-> autoregressive text
```

The cursor does not choose the answer source values. It only decides which
ordered answer role is active. The chosen source slot still comes from the
QTRM state and must drop under source/primitive/renderer ablations.

Implementation:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
tests/test_source_pointer_l4_lm_path_gate.py
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

Smoke artifacts:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_generation_state_ce_s040_rolecursor_smoke8/
```

Smoke result:

```text
full QTRM:
  first4: 4/4
  tail first2: 2/2
  tail remaining2: 2/2
  combined smoke8: 8/8

previous span-only full:
  3/8

previous donor/core_off:
  5/8

tail4 ablations:
  renderer_off: 1/4
  source_binder_off: 1/4
  primitive_off tail2: 0/2
```

Decision:

```text
Promising but not accepted. The evidence is strong enough to justify a broader
held-out gate, but not enough to claim general L4 promotion because the full
8-case result was assembled from split runs and the broad eval has not yet run
with all ablations on the same case set.
```

Next acceptance gate:

```text
Run one reproducible held-out eval with:
  donor_only
  core_off
  full
  renderer_off
  source_binder_off
  primitive_off

Accept only if:
  full > donor/core_off
  full > renderer/source/primitive ablations
  strict exact source-copy scoring is used
  no hidden answer channel is introduced
```

## L4 Bridge Follow-ups 2026-05-09

After the first L4 rejection, three smaller falsifiers tested whether the
problem was only a weak selector/gate or trainable-parameter shortcut.

Artifacts:

```text
primitive-only config:
  configs/qwen35_2b_4090_source_pointer_l4_primitive_lm_bridge_roles12_s080.yaml
forced bridge config:
  configs/qwen35_2b_4090_source_pointer_l4_forced_bridge_roles12_s080.yaml
runner:
  scripts/322_run_source_pointer_l4_lm_path_gate.py
```

Results:

```text
primitive-only bridge S020:
  decision: rejected_l4_candidate
  full: 3/16 = 0.1875
  donor/core-off: 2/16 = 0.125
  primitive-off: 3/16 = 0.1875
  bridge-off: 3/16 = 0.1875

forced bridge S020:
  decision: rejected_l4_candidate
  full: 3/12 = 0.25
  donor/core-off: 2/12 = 0.1667
  primitive-off: 3/12 = 0.25
  bridge-off: 3/12 = 0.25

forced bridge + adapter-only bottleneck S020:
  decision: rejected_l4_candidate
  full: 2/12 = 0.1667
  donor/core-off: 2/12 = 0.1667
  primitive-off: 2/12 = 0.1667
  bridge-off: 2/12 = 0.1667
```

The 5-step adapter-only run was intentionally threshold-relaxed for smoke and
must not be counted as an accepted L4 result; all scores tied and all causal
margins were zero.

Interpretation:

```text
full > donor/core-off appears in some small runs,
but full == primitive-off == bridge-off == final-binder-off.
```

That pattern rejects the current bridge-token cross-attention family as a
canonical L4 renderer. The QTRM/core path can perturb generation, but the
accepted L3 primitive state is not the causal source of the improvement.

Next architecture candidate: replace the bridge-token renderer with a
state-to-vocab pointer/copy residual. The accepted source-position state should
produce a token-level logit bias through the canonical LM logits, and the gate
must require:

```text
full > donor-only
full > core-off
full > primitive-off
full > pointer/copy-renderer-off
```

This keeps the universal LM path intact while making the L3 state-to-token link
directly ablatable.

## State-to-Vocab Renderer Follow-up 2026-05-09

The direct renderer candidate was implemented as a canonical LM residual:

```text
role-value answer bridge tokens
+ optional primitive operation/transition context
-> cross-attend from text positions
-> low-rank vocab projection
-> residual logits added before donor fusion
```

Acceptance status: rejected.

Observed progression:

```text
S005/S010:
  renderer logits were too weak or too non-specific;
  full == donor == core-off == renderer-off.

direct renderer CE S008:
  full generations began to differ from bridge-off and renderer-off,
  so the renderer is causally connected to LM generation.
  Accuracy still tied donor/core-off.

direct CE + primitive contrast:
  did not produce a primitive-off accuracy drop.

focused stronger S024:
  overpowered the donor and collapsed toward source-copy outputs
  such as "40,80" or repeated short numeric fragments.

operation-context S008:
  adding primitive operation tokens to renderer memory did not yet change
  the held-out generation pattern.
```

Current interpretation:

```text
The previous bottleneck "renderer has no effect" is partially solved.
The new bottleneck is "renderer effect is not aligned with final reasoning".
It can bias generation, but it mostly copies source/intermediate values.
```

Updated reject rule:

```text
Do not promote a renderer merely because it changes generation.
Promote only if:
  full beats donor/core-off,
  primitive-off drops,
  renderer-off drops,
  and generated text improves toward exact final answers rather than
  source-copy/intermediate traces.
```

Next candidate:

```text
Use a transformed-value renderer or a recurrent answer-state renderer that is
trained from primitive recurrent state to final answer tokens. It must retain
the canonical LM path but should not rely on a generic low-rank vocab residual
to discover arithmetic/value rendering from weak supervision alone.
```

## Source-Copy Rolecursor Smoke8 2026-05-10

Status: accepted L4 smoke candidate, not broad L4 promotion.

Implemented support:

```text
scripts/329_run_source_copy_rolecursor_l4_eval.py
  chunked/resumable L4 generation runner

src/wgram_lm/wgram_model.py
  source-copy answer-role cursor bias
  source-token span copy bias
```

The runner was added because repeated generation evals can die mid-run or
compete with stale Codex/eval processes. It writes complete artifacts under the
gate out_dir and supports `--resume` so partial runs are not thrown away.

Smoke8 report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke8/report.json
```

Decisive metrics:

```text
full_generation_accuracy:          1.000  (8/8)
donor_generation_accuracy:         0.625  (5/8)
core_off_generation_accuracy:      0.625  (5/8)
primitive_off_generation_accuracy: 0.375  (3/8)
source_slot_off_generation_accuracy: 0.625
source_binder_off_generation_accuracy: 0.625
vocab_renderer_off_generation_accuracy: 0.625

full_minus_donor:         +0.375
full_minus_core_off:      +0.375
full_minus_primitive_off: +0.625
full_minus_source_binder: +0.375
full_minus_renderer_off:  +0.375
```

Interpretation:

```text
The rolecursor source-copy path now has a same-case smoke result where:
  full > donor-only
  full > core-off
  full > primitive-off
  full > source-binder-off
  full > vocab-renderer-off
```

This is stronger than the earlier split/manual tail checks because all seven
modes are evaluated on the same eight cases and assembled into a single report.

Promotion boundary:

```text
Do not call this a general LLM or broad L4 result yet.
It is a source-copy lexicalization smoke acceptance.
Promotion requires:
  16/32/128 case expansion,
  non-source-copy mixed reasoning families,
  and depth/memory ablations that still show causal gains.
```

## Source-Copy Rolecursor Smoke16 2026-05-10

Status: accepted L4 source-copy candidate at 16-case smoke scale.

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke16/report.json
```

Decisive metrics:

```text
full_generation_accuracy:             0.9375  (15/16)
donor_generation_accuracy:            0.5000   (8/16)
core_off_generation_accuracy:         0.5000   (8/16)
primitive_off_generation_accuracy:    0.3750   (6/16)
source_slot_off_generation_accuracy:  0.5000   (8/16)
source_binder_off_generation_accuracy:0.5000   (8/16)
vocab_renderer_off_generation_accuracy:0.5000  (8/16)

full_minus_donor:         +0.4375
full_minus_core_off:      +0.4375
full_minus_primitive_off: +0.5625
full_minus_source_slot:   +0.4375
full_minus_source_binder: +0.4375
full_minus_renderer_off:  +0.4375
```

Decision:

```text
The 8-case acceptance replicated and strengthened at 16 cases.
This promotes the result from manual/split smoke to reproducible source-copy
L4 candidate, but not to broad/general L4.
```

Next gates:

```text
1. 32-case and 128-case source-copy expansion.
2. Mixed-family cases that cannot be solved by source-copy lexicalization.
3. Recursive-depth and memory-composition gates for general LM promotion.
```

## Source-Copy Rolecursor Smoke32 2026-05-10

Status: accepted L4 source-copy candidate at 32-case smoke scale.

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke32/report.json
```

Decisive metrics:

```text
full_generation_accuracy:              0.84375  (27/32)
donor_generation_accuracy:             0.43750  (14/32)
core_off_generation_accuracy:          0.43750  (14/32)
primitive_off_generation_accuracy:     0.37500  (12/32)
source_slot_off_generation_accuracy:   0.43750  (14/32)
source_binder_off_generation_accuracy: 0.43750  (14/32)
vocab_renderer_off_generation_accuracy:0.43750  (14/32)

full_minus_donor:         +0.40625
full_minus_core_off:      +0.40625
full_minus_primitive_off: +0.46875
full_minus_source_slot:   +0.40625
full_minus_source_binder: +0.40625
full_minus_renderer_off:  +0.40625
```

Decision:

```text
The 16-case acceptance replicated at 32 cases. The full canonical LM path keeps
a large margin over donor/core-off and all required ablations.
```

Remaining promotion blockers:

```text
1. 128-case standard gate.
2. Mixed-family LM reasoning where output is not simple source-copy.
3. Recursive depth scaling and memory-composition gates beyond lexicalization.
```

## Source-Copy Rolecursor Standard128 2026-05-10

Status: accepted source-copy lexicalization L4 standard gate.

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_standard128/report.json
```

Decisive metrics:

```text
full_generation_accuracy:               0.78125  (100/128)
donor_generation_accuracy:              0.34375   (44/128)
core_off_generation_accuracy:           0.34375   (44/128)
primitive_off_generation_accuracy:      0.28906   (37/128)
source_slot_off_generation_accuracy:    0.32031   (41/128)
source_binder_off_generation_accuracy:  0.32031   (41/128)
vocab_renderer_off_generation_accuracy: 0.34375   (44/128)

full_minus_donor:         +0.43750
full_minus_core_off:      +0.43750
full_minus_primitive_off: +0.49219
full_minus_source_slot:   +0.46094
full_minus_source_binder: +0.46094
full_minus_renderer_off:  +0.43750
```

Decision:

```text
Promote the rolecursor source-copy path to accepted source-copy L4 standard.
The result is no longer just a smoke candidate.
```

Scope boundary:

```text
This is not yet a broad/general LLM result.
The gate is still source-copy lexicalization. General promotion requires
mixed-family reasoning, non-copy answer synthesis, recursive depth scaling,
and memory-composition gates.
```

## Mixed Non-Copy LM Gate 2026-05-10

Status: rejected; this is the current post-source-copy blocker.

Runner:

```text
scripts/330_run_mixed_noncopy_lm_gate.py
tests/test_mixed_noncopy_lm_gate.py
```

Command:

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

Metrics:

```text
full QTRM:  0/16 = 0.0
donor-only: 0/16 = 0.0
core-off:   0/16 = 0.0
```

Tail-error summary:

```text
generation:
  donor_only: 16/16 numeric_other_miss
  core_off:   16/16 numeric_other_miss
  full QTRM:  16/16 numeric_other_miss

causal forced-choice diagnostic, 4 cases x 3 modes:
  donor_only: 0/4, all select doubled_list
  core_off:   0/4, all select doubled_list
  full QTRM:  0/4, all select doubled_list
```

Decision:

```text
rejected_noncopy_lm_gate
```

Interpretation:

```text
The accepted source-copy L4 path does not transfer to mixed-family non-copy
reasoning. This is not a source-binding failure anymore; it is a non-copy
answer synthesis failure. The next architecture work must make computed latent
state influence normal autoregressive answer tokens, not only copy source
spans.

The forced-choice failure is especially important: even when answer candidates
are provided, the model ranks the intermediate doubled list above the final
computed scalar answer. Therefore the active bottleneck is not just greedy
lexicalization. The recurrent path must learn a causal scalar
reduction/accumulator/final-answer state before another broad LM renderer run
is meaningful.
```

## Runner Result 2026-05-08T09:53:22

```text
gate: renderer_canonical_lm
target_level: L3 candidate
profile: standard
decision: rejected
accepted: False
next_action: renderer remains bottleneck; design a donor-compatible text renderer before memory/metacognition expansion
```

Decisive metrics:

```json
{
  "metrics.full_minus_core_off": 0.0,
  "metrics.full_minus_donor": 0.0,
  "metrics.full_generation_accuracy": 0.0,
  "metrics.core_off_generation_accuracy": 0.0,
  "metrics.donor_generation_accuracy": 0.0,
  "metrics.ablation_generation_accuracy": 0.0,
  "metrics.full_minus_ablation": 0.0
}
```

Report: `local_eval/research_gate_runner/renderer_canonical_lm_standard/report.json`
