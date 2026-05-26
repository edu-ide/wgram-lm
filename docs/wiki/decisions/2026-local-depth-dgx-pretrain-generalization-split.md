# 2026 Local Depth / DGX Pretrain Generalization Split

## Decision

Use the two machines for different questions.

```text
Local 4090:
  Fast microscope.
  Test whether the architecture gets better when it thinks longer.

DGX:
  Slow incubator.
  Test whether pretraining becomes more data-efficient and stays general,
  instead of merely lowering loss.
```

Do not mix these roles in one experiment. If local depth does not help, DGX
pretraining should not be used to hide that architecture failure. If DGX loss
falls but Generalization Dynamics margins get worse, do not call the pretrain
efficient.

## Plain-Language Model

The model needs two different exams.

```text
Local exam:
  Give the student more thinking time.
  If more time does not improve the answer, the thinking organ is not doing
  useful work yet.

DGX exam:
  Let the student read a lot more.
  But keep checking whether the student is understanding, or only becoming
  better at repeating familiar-looking answer patterns.
```

This is the reason for the split:

- local tests the thinking mechanism;
- DGX tests the reading/pretraining process;
- TensorBoard shows the heartbeat;
- GD-lite/OPUS gates decide whether the heartbeat is meaningful.

## Local Track: Depth Scaling Architecture Gate

Question:

```text
Does recurrent depth improve the normal answer path?
```

Required local matrix:

```text
depth: 1, 2, 4, 8, 16
mode: full, core_off, state_frozen or depth_off when available
metric: heldout loss, free generation hit, GD-lite margin, repetition/EOS rate
```

Promote only if:

- deeper thinking improves held-out loss or generation on the same prompts;
- the gain disappears when the recurrent core/depth path is disabled;
- GD-lite anti-parrot margin does not get worse;
- generation samples improve, not only teacher-forced CE.

Reject if:

- all depths produce the same answer distribution;
- depth helps train loss but not held-out or generation;
- core-off matches full;
- the model becomes more repetitive as depth increases.

## DGX Track: Pretrain Efficiency And Generalization Gate

Question:

```text
Can pretraining use better data windows to learn faster while preserving
generalization?
```

The two paper mechanisms used on DGX are:

```text
OPUS:
  Choose useful data dynamically during pretraining.
  Current implementation uses scripts/614_score_opus_projected_utility.py to
  score rows by AdamW-shaped projected update alignment against a heldout proxy
  direction, then scripts/555_prepare_byte_prefixlm_sample.py materializes the
  selected byte window.

Generalization Dynamics / GDsuite:
  Check whether the checkpoint prefers generalizing answers over parrot-like
  shortcut answers.
```

Promote only if:

- held-out PrefixLM loss improves;
- GD-lite intelligence-vs-parrot margin improves or stays positive;
- language generation samples improve;
- no critical family regresses under the same checkpoint selection rule.

Reject if:

- loss improves while GD-lite margin flips negative;
- the model copies repetitive/successive answer patterns;
- it prefers intuitive traps over slow calculation;
- checkpoint selection is based only on train/eval CE.

Implementation boundary:

```text
Implemented now:
  OPUS-lite projected utility scorer:
    proxy gradient direction
    candidate gradient direction
    AdamW optimizer-state preconditioning when a last.pt optimizer checkpoint
    is supplied
    deterministic CountSketch projection
    redundancy-adjusted utility order
    Stage95 utility-window materialization
    Stage95 partial-static -> last.pt optimizer checkpoint -> full OPUS window
    automation when FULL_SELECTION_MODE=utility
    Generalization Dynamics rows in the OPUS proxy only when proxy row limits
    are large enough to include the GD file
    OPUS reports proxy_bucket_counts/proxy_source_counts for future audits
    automatic GD-lite gate after full training

Not yet claimed:
  full OPUS every-iteration online buffer selection inside the trainer.
```

Paper-fidelity correction:

```text
OPUS paper:
  every optimizer step:
    build/score a candidate buffer;
    use optimizer-induced effective updates;
    compare to a refreshed proxy direction;
    train the selected subset in that same step.

Current code:
  score a finite candidate set once from an optimizer-bearing checkpoint;
  materialize an offline byte PrefixLM sample window;
  train normally on that selected window.

Therefore:
  call it OPUS-style or OPUS-inspired data-window selection.
  Do not call it full OPUS or paper-faithful OPUS.

Generalization Dynamics / GDsuite:
  current `generalization_dynamics_lite_probe.jsonl` is a 6-row local smoke
  adapted from the blog/GDsuite task families.
  Do not call it full GDsuite until the official Jiaxin-Wen/GDsuite data/eval
  path is used or an explicitly equivalent full suite is materialized.
```

## 2026-05-26 Stage95I OPUS/GD Contract Audit

Current DGX run:

```text
out:
  /mnt/data4tb/qtrm_multimodal_memoryos/local_eval/20260525_STAGE95I_DGX_1B_OPUS_GD_OFFICIAL_GDN2_ONEBODY_FULL

sample:
  /mnt/data4tb/qtrm_multimodal_memoryos/local_eval/stage95_blt_foundation_byte_curriculum_broad_240k_opus_gd/sampled
```

What is proven applied:

```text
OPUS-style data-window selection:
  metadata.data_selection_contract.selection_mode = utility
  metadata.data_selection_contract.utility_scores_loaded = 202

OPUS scorer:
  contract = opus_projected_utility_v1
  status = pass
  optimizer_state_source = adamw_state_checkpoint
  preconditioner = adamw_state
  proxy_rows = 8
  proxy_target_tokens = 157
  adamw_preconditioned_update_calls = 2233
  identity_fallback_update_calls = 0
  missing_exp_avg_sq_parameter_tensors = 0
  exp_avg_sq_shape_mismatch_parameter_tensors = 0
```

Corrected plain-language read:

```text
The DGX student is not reading a random full textbook. The full byte window was
chosen by an optimizer-shaped audition.

However, this audit does not prove true OPUS/GD. The scorer loaded proxy rows
with a global proxy_rows = 8 cap, and the language-heldout file alone supplies
8 usable rows before the GD-lite file is reached. Therefore Stage95I should be
read as OPUS language-proxy selection unless a later report proves GD rows were
included through proxy_bucket_counts/proxy_source_counts or proxy_rows >= 14.
```

What is not proven applied as a training objective:

```text
Generalization Dynamics is not an in-loop loss for Stage95I.
For Stage95I as audited above, it is guaranteed only as:
  1. a post-training GD-lite gate run by
     scripts/559_run_stage95_blt_partial_then_full_dgx.sh when
     GD_LITE_ENABLED=1.

For a corrected Stage95J-style run, it may also be a proxy component for OPUS
row selection, but only if the OPUS report proves the GD-lite source rows were
loaded.
```

Important launch hygiene:

```text
The script default is:
  GD_LITE_ENABLED=1

The active run-full parent processes did not have GD_LITE_ENABLED=0 in their
environment, so the script default should run GD-lite after full training.

However, one outer watchdog command line contained a stale manual relaunch
override:
  GD_LITE_ENABLED=0

That override is not the canonical contract. Any Stage95I relaunch/watchdog
must remove it or set GD_LITE_ENABLED=1 explicitly.
```

Operational fix applied:

```text
script:
  scripts/620_watch_stage95i_dgx_gd_lite_on.sh

DGX copy:
  /mnt/data4tb/qtrm_multimodal_memoryos/scripts/620_watch_stage95i_dgx_gd_lite_on.sh

watchdog log:
  /tmp/20260525_STAGE95I_GD_LITE_ON_WATCHDOG.log

current behavior:
  preserve the active trainer;
  if the trainer disappears or the log goes stale, relaunch Stage95I with
  GD_LITE_ENABLED=1;
  if report.json appears, exit cleanly.
```

## 2026-05-26 Stage106 Local OPUS/GD A/B

Question:

```text
Does the OPUS/GD data-window choice help locally, or is the DGX signal only a
scale artifact?
```

Why local was not already doing it:

```text
Stage105 local was the BPE reading-stability baseline. OPUS/GD currently scores
byte/BLT rows, so adding it directly to Stage105D would mix tokenizer, data
selection, and architecture changes in one blurry experiment.
```

Local causal test:

```text
script:
  scripts/622_run_local_opus_gd_blt_ablation.sh

tmux:
  stage106_local_opus_gd_blt_ablation
  stage106b_local_opus_gd_proxy14

work root:
  local_eval/20260526_STAGE106_LOCAL_OPUS_GD_BLT_ABLATION
  local_eval/20260526_STAGE106B_LOCAL_OPUS_GD_PROXY14_BLT_ABLATION

comparison:
  static_first_window/sampled
  vs
  opus_gd_utility_window/sampled          # proxy8, language-only by accident
  opus_gd_proxy14_window/sampled          # corrected: language + GD-lite

shared eval:
  shared_eval_window/sampled
```

Important hygiene:

```text
The old Stage94Y BLT checkpoint contained inactive legacy fallback delta keys.
Stage106 does not resume those keys. It first writes:
  base_official_sanitized_model_only.pt

Then it runs a short official-GDN2 warmup to create an optimizer-bearing:
  base_warm_optimizer_seed/last.pt

Only after that does OPUS score rows with:
  preconditioner = adamw_state

  This keeps the A/B from becoming a fallback-vs-official comparison.
```

Stage106 result:

```text
base warm:
  completed
  checkpoint = base_warm_optimizer_seed/last.pt
  final_eval_loss = 0.2478742787
  delta_runtime_fallback_active_count = 0

Stage106 proxy8:
  completed
  proxy_rows = 8
  corrected read = OPUS language-proxy only, not true OPUS/GD
  reason = language-heldout supplied all 8 rows before GD-lite was reached

Stage106B proxy14:
  completed
  proxy_rows = 14
  source_counts:
    prefixlm_language_heldout.jsonl = 8
    generalization_dynamics_lite_probe.jsonl = 6
  preconditioner = adamw_state
  identity_fallback_update_calls = 0
  delta_runtime_fallback_active_count = 0

Stage106D minimax proxy14:
  completed
  proxy_rows = 14
  proxy_grouping = source_file_bucket
  proxy_score_mode = minimax_mean
  proxy_mean_weight = 0.25
  source_counts:
    prefixlm_language_heldout.jsonl = 8
    generalization_dynamics_lite_probe.jsonl = 6
  preconditioner = adamw_state
  identity_fallback_update_calls = 0
  delta_runtime_fallback_active_count = 0
```

Stage106B A/B metrics:

```text
static:
  final_eval_loss = 0.0908529028
  best_eval_loss = 0.0537150951 @ step240
  GD-lite accuracy = 0.3333
  GD-lite mean_margin = -0.7303
  generation prefix_token_accuracy = 0.6960
  generation ended_with_eoa_fraction = 0.25

OPUS/GD proxy14:
  final_eval_loss = 0.0980483626
  best_eval_loss = 0.0539917022 @ step240
  GD-lite accuracy = 0.3333
  GD-lite mean_margin = -0.8753
  generation prefix_token_accuracy = 0.7815
  generation ended_with_eoa_fraction = 0.125
```

Decision:

```text
Do not promote the current OPUS/GD proxy14 recipe.

It improved free-generation prefix overlap but lost on final heldout loss,
slightly lost on best heldout loss, and worsened GD-lite mean margin. That
means the data audition is not yet selecting a more general student; it is
mostly selecting rows that make surface continuation easier.
```

Stage106D minimax A/B metrics:

```text
static:
  final_eval_loss = 0.0908529028
  best_eval_loss = 0.0537150951 @ step240
  GD-lite accuracy = 0.3333
  GD-lite mean_margin = -0.7303
  GD-lite min_margin = -2.3796
  generation prefix_token_accuracy = 0.6960
  generation ended_with_eoa_fraction = 0.25

OPUS/GD minimax proxy14:
  final_eval_loss = 0.0915803693
  best_eval_loss = 0.0581947468 @ step240
  GD-lite accuracy = 0.3333
  GD-lite mean_margin = -0.6836
  GD-lite min_margin = -2.5544
  generation prefix_token_accuracy = 0.8906
  generation ended_with_eoa_fraction = 0.0
```

Decision:

```text
Do not promote Stage106D as a training recipe, but keep the scorer change.

The minimax selector fixed the measured failure mode partially:
  average OPUS/GD worsened GD mean margin to -0.8753;
  minimax OPUS/GD improved it to -0.6836 versus static -0.7303.

But it still rejects because GD accuracy remains 0.3333, min_margin worsened,
best heldout loss worsened, and free generation still has 0 exact hits.

Plain-language read:
  The new audition no longer forgets the anti-parrot exam, but it can only
  choose from the existing textbook pages. If the textbook lacks enough
  explicit anti-parrot lessons, selecting pages more carefully cannot create
  the missing lesson. The next move should change the data contract, not only
  the row-selection score.
```

Plain-language read:

```text
The local exam now asks:
  If two identical students start from the same cleaned-up body and optimizer
  state, does the student who receives the OPUS/GD-chosen worksheet learn
  heldout language/generalization faster than the student who receives the
  first-seen worksheet?
```

Promote only if:

- OPUS/GD branch beats static on the same heldout eval window;
- GD-lite/generation gates do not regress;
- the branch base is `base_warm_optimizer_seed/last.pt`;
- `delta_runtime_fallback_active_count = 0`.

Checkpoint-selection fix applied:

```text
The active DGX run may have started before this trainer change, but the
canonical BLT trainer now supports best-eval preservation:

  scripts/557_train_blt_d_prefixlm_dataio.py
    --save-best-eval-checkpoint
    writes best_eval_model.pt and copy_best_eval_model.pt

This matters because long runs can improve language/generation early and then
overfit or drift later. Acceptance should inspect both:

  last_model.pt
  best_eval_model.pt, if present

and should prefer the checkpoint that passes heldout/generation/GD-lite gates,
not automatically the final scheduled step.
```

Plain-language guardrail:

```text
Do not grade only the student at bedtime. Also keep the notebook from the hour
when the student was most accurate on unseen questions.
```

Allowed claim:

```text
Stage95I is an OPUS-selected full-byte pretraining run with a language-heldout
proxy and GD-lite expected as a post-training acceptance gate.
```

Forbidden claim:

```text
Stage95I trains with a Generalization Dynamics loss or has already passed the
GD-lite gate before generalization_dynamics_lite_report.json exists.

Stage95I used GD rows in the OPUS proxy, unless the report proves GD source
rows through proxy_source_counts/proxy_bucket_counts or a corrected
proxy_rows >= 14 audit.
```

Plain-language read:

```text
OPUS is no longer just "maybe use a score file".
The model now has a data audition: a candidate page is useful only if training
on it pushes the model in the same direction as the heldout proxy exam.
But the audition currently happens before sample binding, not every optimizer
iteration inside the main training loop.

Proper OPUS needs AdamW memory. If the run only saves `last_model.pt`, the
student has an answer sheet but not the optimizer's handwriting history. The
Stage95 launcher therefore keeps partial selection static by default, saves
`PARTIAL_OUT/last.pt` when the full window is OPUS-selected, and uses that
optimizer-bearing checkpoint to score the broad/full data window.

The default promoted Stage95 path is now:

SELECTION_MODE=utility
PARTIAL_SELECTION_MODE=first
FULL_SELECTION_MODE=utility
OPUS_PROXY_JSONL=prefixlm_language_heldout + generalization_dynamics_lite_probe
OPUS_PROXY_MAX_ROWS>=14
OPUS_PROXY_GROUPING=source_file_bucket
OPUS_PROXY_SCORE_MODE=minimax_mean
GD_LITE_ENABLED=1

Meaning: the first small lesson creates optimizer memory; the full lesson is
OPUS-selected using both normal language and anti-parrot Generalization
Dynamics pressure only if the proxy row cap actually reaches both files; the
checkpoint is then judged by GD-lite after training.
```

## TensorBoard Contract

Use:

```bash
scripts/568_start_tensorboards.sh start both
scripts/568_start_tensorboards.sh status both
```

URLs:

```text
local: http://127.0.0.1:6007/
DGX:   http://192.168.219.113:6008/
```

The TensorBoard process is infrastructure only. It proves observability, not
model quality. A run still needs the gates above.

## Immediate Next Action

1. Run the local depth gate first on the smallest current one-body/BLT
   checkpoint that can generate.
2. Run GD-lite on Stage96C and Stage96D checkpoints as a pretraining sanity
   check.
3. If Stage96D has lower loss but worse GD-lite margin than Stage96C, treat the
   full-data pretrain as a data/window failure, not as model progress.
4. Only after local depth improves should LT2/GDN mixer optimization become a
   promoted DGX-scale experiment.

## Hard Lock

```text
No long DGX run is promoted from now on unless it answers one of these:

1. Local depth scaling survived destructive ablation.
2. DGX pretrain improved loss and did not lose GD-lite generalization margin.
```

This prevents the old failure mode: doing more training while the actual story
of the model still does not make sense.

## Current Evidence Snapshot

Local Stage99I one-body checkpoint:

```text
checkpoint:
  local_eval/20260524_STAGE99I_LOCAL_ONE_BODY_GATE400/last_model.pt

depth report:
  local_eval/20260524_STAGE99I_LOCAL_ONE_BODY_DEPTH_RESIDUAL_PROBE256/report.json

depth 1 loss: 2.5619886004, residual: 0.6702648781
depth 2 loss: 2.5603724661, residual: 0.1889533062
depth 4 loss: 2.5783546231, residual: 0.0994255078
depth 8 loss: 2.6085943811, residual: 0.0543557573

verdict:
  rejected as depth-scaling architecture progress
```

Plain-language read:

```text
The thought state becomes more stable as depth increases, but the answer gets
worse after depth 2. This means the model is settling, but not settling into a
better answer basin. Do not spend DGX pretraining to hide this local
architecture failure.
```

DGX Stage96 evidence:

```text
Stage96C partial:
  GD-lite report exists:
    local_eval/20260525_STAGE100_GD_LITE_STAGE96C_FULL6/report.json
  accuracy = 0.3333
  mean_margin = 0.0137
  min_margin = -0.7227
  passed = repetitive_answer_icl, persona_multihop_icl
  failed = flipped_answer_icl, intuitive_answer_zero_shot,
           successive_answer_icl, truthy_answer_icl
  accepted = false

Stage96D full:
  GD-lite report exists:
    local_eval/20260525_STAGE100_GD_LITE_STAGE96D_FULL6/report.json
  accuracy = 0.3333
  mean_margin = 0.0457
  min_margin = -0.3032
  passed = repetitive_answer_icl, persona_multihop_icl
  failed = flipped_answer_icl, intuitive_answer_zero_shot,
           successive_answer_icl, truthy_answer_icl
  accepted = false
```

Plain-language read:

```text
Stage96D is less bad than Stage96C on soft margins: mean margin improves and
the worst shortcut margin is less negative. But it still does not pass the
Generalization Dynamics gate. More pretraining made the student slightly less
parrot-like, not generally reliable.
```

## Local Generalization + Efficiency Depth Sweep

Run:

```text
local_eval/20260525_STAGE100_LOCAL_STAGE99I_GD_DEPTH_SWEEP/summary.json
local_eval/20260525_STAGE100_LOCAL_STAGE99I_GD_DEPTH_SWEEP/tensorboard
```

Checkpoint:

```text
local_eval/20260524_STAGE99I_LOCAL_ONE_BODY_GATE400/last_model.pt
```

Result:

```text
depth | GD-lite acc | mean margin | min margin | heldout loss | residual | elapsed
1     | 0.3333      | -0.0080     | -0.5378    | 2.5620       | 0.6703   | 20.69s
2     | 0.3333      |  0.0036     | -0.4535    | 2.5604       | 0.1890   | 20.63s
4     | 0.3333      | -0.0049     | -0.4384    | 2.5784       | 0.0994   | 20.81s
8     | 0.5000      | -0.0452     | -0.4961    | 2.6086       | 0.0544   | 21.31s
16    | 0.5000      | -0.0575     | -0.5451    | n/a          | n/a      | 21.89s
```

Verdict:

```text
accepted = false
```

Plain-language read:

```text
Longer thinking makes the hidden state calmer, and by depth 8/16 it passes one
extra GD-lite item. But it does not become broadly wiser: average anti-parrot
margin gets worse, the hardest shortcut axes remain failed, and held-out loss
already worsens after depth 2.
```

Failed axes that must be fixed by the next architecture:

```text
flipped_answer_icl:
  cannot reliably obey a local rule when it contradicts memorized sentiment.

successive_answer_icl:
  still follows answer-sequence temptation instead of doing arithmetic.

truthy_answer_icl:
  still confuses what sounds plausible with what is true.
```

Consequence:

```text
Do not promote "more recurrent depth" as the fix. The next local architecture
change must make repeated thinking directly repair these failed axes, then
rerun this same sweep.
```

Solution-aligned attractor gate:

```text
script:
  scripts/569_eval_solution_aligned_answer_attractor_gate.py

report:
  local_eval/20260525_STAGE100_LOCAL_STAGE99I_GD_DEPTH_SWEEP/solution_aligned_answer_attractor_gate.json

baseline:
  depth 2

candidate:
  depth 8

accepted:
  false

failed checks:
  gd_mean_margin_improves
  critical_tasks_pass
  heldout_loss_not_regressed
```

Plain-language read:

```text
Depth 8 looks more thoughtful than depth 2 in a narrow sense: residual is lower
and one extra GD-lite task passes. But it is not a solution attractor. The
average intelligence-vs-parrot margin gets worse, the three critical shortcut
axes remain failed, and held-out loss regresses. The student is calmer, not
wiser.
```

Next local experiment must be designed to pass this exact gate, not only the
older residual gate.

## 2026-05-26 Max-Full OPUS/GD Upgrade

What changed:

```text
GD:
  Official GDsuite repo is cloned at:
    references/official/GDsuite

  Official choice-family probe is materialized at:
    data/eval/official_gdsuite_choice_probe.jsonl

  Rows:
    66,164

  Covered official families:
    flipped_answer
    repetitive_answer
    successive_answer
    truthy_answer
    intuitive_answer

  Still missing:
    multihop_persona_qa, because the official version is generative + regex
    based rather than a correct-vs-parrot logprob choice.

OPUS:
  Offline OPUS scorer now defaults to:
    proxy = prefixlm_language_heldout + official_gdsuite_choice_probe
    proxy_grouping = source_file_bucket
    proxy_score_mode = minimax_mean
    proxy_max_rows = 0
    proxy_max_rows_per_group = 8

  Trainer now supports online candidate-batch selection:
    scripts/557_train_blt_d_prefixlm_dataio.py --online-opus-enabled

  Stage95 launcher now defaults ONLINE_OPUS_ENABLED=1, so full runs use
  in-loop candidate selection unless explicitly disabled.
```

Plain-language read:

```text
The old setup was checking one tiny anti-parrot flashcard and then choosing a
data window before training. The upgraded setup checks the official GDsuite
choice curriculum as a broad anti-shortcut map, and the trainer can now choose
between live candidate batches while training.

This is much closer to the intended "eat the data that improves the future
model" idea. It is still not paper-faithful full OPUS, because Ghost/Muon and
the full rolling-buffer machinery are not implemented, and it is not full
GDsuite until persona generation is handled.
```

Smoke evidence:

```text
online trainer smoke:
  local_eval/20260526_ONLINE_OPUS_TRAINER_SMOKE/report.json

result:
  2 steps completed
  online_opus_active = 1.0 on both logged steps
  selected_index was logged
  eval clean loss moved 6.3008 -> 6.1371

verification:
  py_compile:
    scripts/557_train_blt_d_prefixlm_dataio.py
    scripts/567_eval_blt_generalization_dynamics_probe.py
    scripts/614_score_opus_projected_utility.py
    scripts/623_build_official_gdsuite_choice_probe.py

  bash -n:
    scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh
    scripts/559_run_stage95_blt_partial_then_full_dgx.sh
    scripts/622_run_local_opus_gd_blt_ablation.sh

  unittest:
    tests.test_opus_projected_utility_scorer, 10 tests OK
```
