# 0001 Active Decision Index

## Purpose

This file is the first page an agent should read before using decision records.
It prevents old rejected decisions from silently becoming current instructions.

## Rule

```text
If a decision is not listed here as active, treat it as context, not policy.
```

Historical decision files can explain why something failed, but they do not
override this active index.

## Active Decisions

### Stage101 One-Body Answer Attractor

```text
file:
  2026-05-25-stage101-solution-aligned-answer-attractor.md

status:
  active

current read:
  The model must solve through one path:
    prompt/source
    -> recurrent thought
    -> same LM head answer

  Stage101L/M/N/O/P/Q/R/S show the current bottleneck:
    old-anchor replay can preserve shortcut resistance,
    source binding can be taught on narrow rows,
    real evidence judgment improves when belief is numeric and factorized,
    but low/neutral numeric strings are not yet speakable.

  Stage101N is not promoted:
    it preserved old anchors,
    but source heldout stayed at 0.25 and source margin worsened versus M.

  Stage101O is not promoted:
    it improved its own counterfactual source-binding heldout,
    but older source-balance still failed and Stage101P exposed
    source-obedience risk.

  Stage101P is still a gate:
    reliable evidence may revise belief,
    untrusted evidence must be ignored,
    conflicts require reliability judgment,
    insufficient evidence must answer Unknown.

  Stage101Q is a rejected numeric diagnostic:
    graded support/reliability/sufficiency is a better target than bare
    True/False/Unknown, but the full ledger string did not yet train through.

  Stage101R is the active numeric route:
    factorized support/reliability/sufficiency moved heldout accuracy from
    0.1667 to 0.7083 while preserving the old Stage101B anchor, but it still
    fails low/neutral scalar states.

  Stage101S is not promoted:
    direct scalar-prior calibration raised mean margins but did not flip any
    +0.00, +0.80, 0.10, or 0.50 target rows. It mostly strengthened the already
    easy 0.90 habit.

  Stage101T is not promoted:
    bucket-to-number disentanglement was useful as a diagnostic, but it still
    treated low/neutral/insufficient as labels to name or translate. It did not
    force the model to infer reliability from source role or support/sufficiency
    from evidence relevance and polarity.

  Stage101U is the active data-causality route:
    evidence judgment must be stepwise and causal:
      source role -> source reliability
      evidence relevance/polarity -> support and sufficiency
      parent chain -> numeric support/reliability/sufficiency

  Stage101V is the active curiosity diagnostic:
    curiosity means metacognitive evidence acquisition, not personality.
    The model should answer_now only when evidence is trusted and sufficient;
    otherwise it should ask_more and name the missing evidence type.
    The first V smoke is not promoted because it overcorrected into
    over-curiosity: ask_more appeared, but trusted-and-sufficient heldout rows
    began asking too.

next expected move:
  Stage101W9 paired latent feature-difference reader:
    W4/W5/W6/W7 all rejected.
    W4 added causal-plausibility labels.
    W5 added minimal counterfactual repair labels.
    W6 removed labels and used A/B counterfactual worlds.
    W7 scored the two worlds independently with siamese yes/no energy.
    W8 added latent feature heads over decoder hidden state. It improved
    answer_permission to 0.875 heldout, but all-feature accuracy stayed 0.3906
    and detail/conflict remained weak.

    Stop launching independent one-world feature-label variants as the main
    mechanism. Build paired examples where exactly one feature changes, then
    train pairwise margins over latent feature states. The immediate weak axes
    are detail sufficiency and conflict status.

    W9 is the last nearby diagnostic in this rejected family. If it rejects on
    the same detail/conflict bottleneck, do not launch W9-with-a-stronger-loss.
    Jump to Stage101X: counterfactual-imagination answer attractor, where the
    same one-body LM-head answer path is trained on original vs minimally
    imagined counterfactual worlds.
  Promote only if:
    trusted+sufficient rows answer_now,
    untrusted/irrelevant/partial/conflict rows ask_more,
    request type is correct,
    U/R heldouts do not regress,
    old Stage101B anchor remains accepted.
```

### Internal Multi-Trajectory Answer Attractor

```text
file:
  ../architecture/internal-multitrajectory-answer-attractor-ssot.md

status:
  active paper-candidate architecture contract

current read:
  "GRAM + PTRM + top-k selector" is not a sufficient novelty claim.
  The paper-candidate claim is stricter:
    candidate thoughts are generated internally,
    checked for answer-attractor convergence,
    selected or aggregated inside the one-body path,
    and spoken through the same LM head.

  External answer tables, oracle-only selected accuracy, detached verifiers,
  and sidecar speakers are diagnostic-only.

next expected move:
  Do Stage101W9 before IMTA-K:
    paired latent feature-difference reader,
    exact one-feature counterfactual pairs,
    all-depth pairwise feature-margin supervision,
    answer-permission dependency on the feature state,
    feature-off / answer-attractor-off ablations.

  Then run IMTA-K only after W-style evidence/curiosity gates stop rejecting:
    K=1/3/8 stochastic recurrent trajectories,
    depth scaling,
    same LM-head answer margin,
    checker-off / stochastic-off / one-body-state-off ablations.
  HRM-Text DataIO remains the required language-training contract for any
  one-body language or efficiency claim: DataIO rows -> PrefixLM input_ids,
  response-only labels, attention_mask, same LM-head generation gates.
  MSA belongs after these gates as the sparse long-memory route into the IMTA
  reader, not as a shortcut for current short-context evidence failures.

promote only if:
  K>1 beats K=1 without oracle-only scoring,
  answer-attractor margins improve through the same LM head,
  the gain disappears under the required ablations,
  Stage101B anchor remains accepted,
  DataIO/PrefixLM heldout and free-generation gates do not regress,
  free generation is non-degenerate,
  and any MSA claim passes memory_off/router_off/chunk_shuffle ablations.
```

### Stage104 BPE Reading-Stability Control

```text
file:
  2026-05-26-stage104-bpe-control-blt-reading-bottleneck.md

status:
  active diagnostic anchor

current read:
  BPE is not declared the final tokenizer, but it is now the short reasoning
  microscope control for stable reading. Stage104B shows that the same native
  recurrent PrefixLM body trains stably on the Stage103 rows with official
  HRM-Text/Data-IO BPE:
    eval loss 11.0748 -> 2.0018 in 240 steps,
    eval_nonfinite_batches = 0,
    depth4 beats depth1,
    generation is non-degenerate but not solved: exact 1/16, loops 0/16.

  Corrected Stage103D finite-row aggregation shows BLT is not hopeless:
    depth1 loss 1.7889 -> depth8 loss 1.7039.
  But it is not promoted because non-finite depth rows remain.

  Qwen3.6-27B comparison records are narrow Stage58 modulo-10 synthetic OOD
  answer-only baselines, not broad public-benchmark wins:
    saved Qwen3.6 full answer-only baseline = 324/768 = 0.4219.
  A local QTRM win there may be reported only as a scoped private-suite signal.
  It is not a general LLM, official benchmark, or agentic-tool win.

  Stage104C continued the BPE control to 1200 steps. It finished normally and
  stayed numerically stable, but train loss collapsed while eval loss bottomed
  near step840 and worsened by step1200. Read this as clean overfit on the tiny
  microscope, not as proof that longer local training solves free generation.

  Generation did improve versus Stage104B:
    Stage104B step240 exact generation = 1/16, first-token acc = 0.18125.
    Stage104C step1200 exact generation = 6/16, first-token acc = 0.44375.
  This proves the BPE one-body mouth is learnable, but not robust yet: later
  arithmetic rows still fall into wrong answer basins such as 290.

  Operational correction: native/BPE and BLT PrefixLM trainers now preserve
  best_eval_model.pt and copy_best_eval_model.pt when eval improves. Future
  long microscopes must judge best-eval and last checkpoints separately, so a
  late overfit step does not erase the best generalizing student.

  Stage105A broad HRM-Text split probe kept the same BPE one-body architecture
  and changed the data contract to a 90/10 split over gsm8k/math/omnimath/
  openbookqa. At 120 local steps:
    train loss 11.1632 -> 5.7894,
    heldout eval loss 11.0939 -> 4.3923,
    eval_nonfinite_batches = 0,
    best_eval_model.pt was saved.
  Current read: Stage104C's overfit is mainly a narrow worksheet/data-contract
  problem, not proof that the one-body architecture inherently overfits.

  Stage105B continued the same broad split to 600 steps over all train rows.
  Heldout eval kept falling:
    step120 4.3923,
    step240 3.5739,
    step360 3.2487,
    step480 3.0587,
    step600 2.9360.
  Free generation is still early: 3/12 exact on a tiny generation gate, 12/12
  clean EOA stops, 0/12 repeated loops. Current read: keep training; the model
  is learning the broad textbook but still falls into easy answer basins.

  Stage105C completed to 2000 steps:
    out = local_eval/20260526_STAGE105C_LOCAL_BPE_BROAD_HRM_TEXT_CONT2000_ALLROWS
    final/best heldout eval loss = 2.4680 at step2000.

  Stage105D completed as the batch-increased continuation:
    out = local_eval/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS
    log = /tmp/20260526_STAGE105D_LOCAL_BPE_BROAD_HRM_TEXT_BS48_CHUNK64_CONT5000_ALLROWS.log
    resume = Stage105C/last.pt
    batch = 48
    loss_chunk_size = 64
    target step = 5000.
  Capacity audit:
    batch64 passed only 1-step smoke but rejected on 20-step smoke with
    cross_entropy CUDA OOM even at loss_chunk_size 32.
    batch48 + loss_chunk_size 64 passed 20-step smoke and is promoted locally.
  Stage105D eval improved at first:
    Stage105C step2000 eval = 2.4680,
    Stage105D step2040 eval = 2.4012,
    Stage105D step2160 eval = 2.3736,
    Stage105D step2280 eval = 2.2555,
    Stage105D step2400 eval = 2.2072,
    Stage105D step2520 eval = 2.1619.
  But the longer continuation overfit:
    best eval loss = 2.077648 at step3240,
    final eval loss = 2.553959 at step5000.
  Read: the BPE one-body mouth can learn broad HRM-Text-style rows, but the
  small local continuation is still not a robust answer-preference model.

  Stage112 tested whether the Stage110/111 GD rejection was only a BLT reading
  problem by evaluating stable BPE checkpoints on the same GD choice gate:
    file = 2026-05-26-stage112-bpe-gd-restoration-gate.md
    Stage104C BPE recurrent control: accuracy 0.4000, mean margin -0.1592.
    Stage105D BPE broad best: accuracy 0.2000, mean margin -0.0374.
  Decision: both reject. BPE is a stable reading control, but it does not by
  itself teach anti-parrot answer preference.

  Stage113 changed the causal route by training answer preference through the
  same BPE PrefixLM logits:
    file = 2026-05-26-stage113-bpe-gd-preference-restoration.md
    baseline Stage105D: accuracy 0.2000, mean margin -0.0374.
    Stage113A step40: accuracy 0.6500, mean margin 0.8979.
    Stage113B continued: accuracy 0.7000, mean margin 1.2324.
  Decision: partial positive causal signal, not accepted. The missing
  anti-parrot habit is trainable in the same LM answer path, but algebra/CRT
  remain weak and short generation regressed slightly.

next expected move:
  Treat BLT/H-Net/semantic-BLT as a reading-compression research thread until
  it matches the BPE control on:
    no non-finite eval/depth rows,
    depth gain or preservation versus depth1,
    non-degenerate free generation.

promote only if:
  BLT beats or matches BPE on the same rows and the same one-body answer path,
  not only on filtered finite rows.

  Do not promote OPUS/GD full selection or BLT boundary changes to DGX as a
  generalization claim until a local gate improves GD accuracy, GD margin, and
  free-generation samples together. The next local move must train/evaluate an
  explicit answer-preference or candidate-exposure path, with ablations that
  prove the normal LM answer route uses it.

  After Stage113, the concrete next local move is Stage114:
    BPE same-LM preference training,
    hard-family replay for algebra/CRT,
    language-preservation CE/KL mix,
    heldout GD smoke plus language/generation preservation gates.
  Do not scale this route to DGX until Stage114 keeps the Stage113 margin gain
  while removing the algebra/CRT misses and avoiding generation regression.

  Stage114 partially succeeded and Stage115/116 localized the remaining wall:
    file = 2026-05-26-stage114-stage116-hard-algebra-followup.md
    Stage114: accuracy 0.8000, mean margin 1.3043, CRT passed, language and
      generation preserved versus Stage113B.
    Stage115: algebra-only replay stayed at accuracy 0.8000.
    Stage116: stronger chosen CE stayed at accuracy 0.8000.
  Current read:
    preference pressure is real and useful,
    language preservation works,
    but the last algebra rows need an internal calculation curriculum, not
    another replay/CE scalar sweep.

  Next expected move:
    Stage117 generated non-heldout algebra traps:
      misleading repeated demo answer,
      final equation requiring a = x - y or a = y - x,
      same LM answer path,
      language CE preservation,
      heldout smoke rows only for evaluation.

  Stage118 (fixed-parrot algebra diagnostic, 60 steps) was evaluated as a direct follow-up.
  Result on the same GD gate:
    - Stage118 last: accuracy 1.0, mean_margin +1.208, min_margin +0.022, accepted True
    - All algebra variants now pass (big improvement over Stage117's 0.85 / negative min margin).
  Language heldout actually improved vs Stage117.
  However, direct generation quality regressed (exact 0/12 vs 1/12, prefix accuracy 0.133 vs 0.317).

  Decision (per handoff promotion rule):
    Stage118 is a strong diagnostic but is **not promoted** as the new local anchor
    because of material regression on the generation gate.
    Current local anchor remains Stage117.

  The persistent wall after two rounds of preference pressure strongly suggests
  that scalar "more exposure to the failure mode" has reached diminishing returns.
  The next move should be a structural change to how the recurrent state binds
  and computes the final equation (one-body equation-state readback direction).

  Stage119 experimental contract defined (see dedicated decision record):
    - Hypothesis: forcing the recurrent state itself to explicitly bind the
      final equation operands/operation and maintain a "solved" representation
      will give a cleaner causal gain on hard algebra cases than additional
      final-answer preference pressure.
    - Minimal addition: lightweight equation-binding auxiliary objective on
      generated algebra trap data (leveraging or extending existing typed
      register / state supervision machinery).
    - Falsification gate: short local probe from current anchor + standard
      44-row GD smoke + language/generation preservation + state ablation.
    - Status: ready for minimal implementation and cheap local probe.
```

### Stage95I DGX OPUS/GD Contract Audit

```text
file:
  2026-local-depth-dgx-pretrain-generalization-split.md

status:
  active launch/evidence contract

current read:
  DGX Stage95I uses an OPUS-style utility-selected byte window:
    data_selection_contract.selection_mode = utility,
    utility_scores_loaded = 202,
    opus_projected_utility_report.status = pass,
    preconditioner = adamw_state,
    identity_fallback_update_calls = 0.

  Corrected audit:
    Stage95I's audited OPUS report had proxy_rows = 8.
    Because prefixlm_language_heldout supplies 8 usable rows before the
    GD-lite file is reached, Stage95I must be read as OPUS language-proxy
    selection unless a later report proves GD rows through proxy_source_counts,
    proxy_bucket_counts, or proxy_rows >= 14.

  Generalization Dynamics is required as a post-training GD-lite acceptance
  gate. It is not an in-loop training loss for Stage95I.

  The canonical script default is GD_LITE_ENABLED=1. A stale outer watchdog
  command line had GD_LITE_ENABLED=0 in its relaunch branch; this is not the
  promoted contract and must not be copied.

  Operational guard now exists:
    scripts/620_watch_stage95i_dgx_gd_lite_on.sh
  It preserves the active trainer and relaunches only missing/stale Stage95I
  training with GD_LITE_ENABLED=1.

  The canonical BLT trainer also supports best-eval checkpoint preservation
  through --save-best-eval-checkpoint, writing best_eval_model.pt and
  copy_best_eval_model.pt. Use these for acceptance gates when present instead
  of assuming last_model.pt is the best checkpoint.

  Local Stage106/106B/106D is the completed OPUS/GD microscope:
    script = scripts/622_run_local_opus_gd_blt_ablation.sh
    tmux =
      stage106_local_opus_gd_blt_ablation,
      stage106b_local_opus_gd_proxy14,
      stage106d_local_opus_gd_minimax
    work root = local_eval/20260526_STAGE106_LOCAL_OPUS_GD_BLT_ABLATION
    corrected work root =
      local_eval/20260526_STAGE106B_LOCAL_OPUS_GD_PROXY14_BLT_ABLATION
    minimax corrected work root =
      local_eval/20260526_STAGE106D_LOCAL_OPUS_GD_MINIMAX_BLT_ABLATION
  It compares static-first byte/BLT rows against OPUS/GD utility-selected
  byte/BLT rows from the same cleaned official-GDN2 warm checkpoint. This is
  the local answer to "does OPUS/GD actually help?", separate from the Stage105D
  BPE reading baseline.

  Stage106 proxy8 was not true OPUS/GD; the row cap excluded GD-lite rows.
  Stage106B proxy14 did include 8 language rows + 6 GD-lite rows.
  Stage106B is not promoted:
    static best eval loss = 0.0537150951, final = 0.0908529028,
    OPUS/GD14 best eval loss = 0.0539917022, final = 0.0980483626,
    static GD-lite mean margin = -0.7303,
    OPUS/GD14 GD-lite mean margin = -0.8753.

  Stage106D changed OPUS scoring to source_file_bucket + minimax_mean.
  Keep this scorer audit fix, but do not promote the recipe:
    OPUS/GD minimax best eval loss = 0.0581947468, final = 0.0915803693,
    OPUS/GD minimax GD-lite accuracy = 0.3333,
    OPUS/GD minimax GD-lite mean margin = -0.6836,
    OPUS/GD minimax min margin = -2.5544,
    generation prefix accuracy = 0.8906,
    exact generation = 0.
  Read: minimax data audition helps the average anti-parrot margin versus
  static, but it cannot create missing anti-parrot lessons from the existing
  rows. Next move should change the data contract, not merely tune OPUS.

promote only if:
  Stage95I writes generalization_dynamics_lite_report.json and the report does
  not reject, or the rejection is explicitly recorded as a data/window failure.
  Stage106 promotes only if OPUS/GD beats static on the shared local heldout
  window without reintroducing fallback runtime.

  New hard audit rule:
    An OPUS/GD run is not allowed to call itself OPUS/GD unless its report
    proves GD rows were included. The practical minimum for the current
    two-file proxy is OPUS_PROXY_MAX_ROWS >= 14, or explicit
    proxy_source_counts/proxy_bucket_counts containing GD-lite rows.
    The promoted scorer default is OPUS_PROXY_GROUPING=source_file_bucket and
    OPUS_PROXY_SCORE_MODE=minimax_mean; aggregate mean scoring is diagnostic
    only for OPUS/GD because Stage106B showed it can worsen GD margin.

  Paper-fidelity rule:
    OPUS now has two implementation levels:
      1. offline projected-utility data-window selection;
      2. online candidate-batch selection inside
         scripts/557_train_blt_d_prefixlm_dataio.py via --online-opus-enabled.
    Allowed claim: "online OPUS-style candidate batch selection."
    Forbidden claim: "paper-faithful full OPUS" until Ghost/Muon/hybrid
    optimizer geometry and the complete rolling-buffer setup are implemented.

    Generalization Dynamics now has two implementation levels:
      1. GD-lite 6-row smoke;
      2. official GDsuite choice-family adapter:
         data/eval/official_gdsuite_choice_probe.jsonl
         with 66,164 official rows across the five logprob families.
    Allowed claim: "official GDsuite choice-family adapter."
    Forbidden claim: "full GDsuite pass" until the multi-hop persona
    generative family and trajectory sweep are also implemented.
```

### Official GDN2 Runtime Contract

```text
file:
  2026-05-25-official-gdn2-runtime-contract.md

status:
  active launch policy

current read:
  official_gated_delta2 is fail-fast.
  No Torch fallback, no runtime fallback, no auto ptxas fallback, and no
  fallback checkpoint resume under an official-GDN2 experiment name.

  DGX official GDN2 runs must set both:
    REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
    TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas

  Current local RTX 4090 official-GDN2 runs must set both:
    REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
    TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas

  Do not use /usr/local/cuda/bin/ptxas locally right now; it points at CUDA
  13.0 and Triton 3.3.1 rejected it before Stage105A smoke.

  Before continuing Stage95-style work, run:
    bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh preflight

  Legacy Stage95B partial checkpoint is blocked:
    legacy fallback delta-mixer keys are present,
    decoder_latent_mode=add while current policy expects one_body.

next expected move:
  Do not resume new official-GDN2 one-body runs from Stage95B.
  The Stage95 launcher now defaults to clean 20260525 Stage95G/H dirs.
  Start clean, or resume only from a checkpoint that passes
  scripts/613_preflight_official_gdn2_contract.py.
```

### Stage101A First Smoke

```text
file:
  2026-05-25-stage101a-solution-attractor-smoke.md

status:
  superseded by Stage101 one-body answer-attractor chain

current read:
  Useful first proof that answer-facing contrast can move the same LM head.
  Not a promoted checkpoint and not the current training recipe.
```

### Decision File Ordering

```text
file:
  0000-decision-file-ordering.md

status:
  active process rule

current read:
  New and active decision files must use ordered names. Full archive migration
  should be scripted with link rewriting and status labels.
```

## Archived-By-Default Rule

All unordered decision files are currently archived by default unless this file
explicitly marks them active.

Plain-language reason:

```text
The archive is useful memory, but it is not the driver's seat.
```

### Stage119 One-Body Equation-State Readback (L1 Probe)

file:
  docs/wiki/decisions/2026-05-26-stage119-one-body-equation-state-readback.md
  docs/wiki/decisions/stage119/probe_results_2026-05-26.md
  docs/wiki/decisions/stage119/failure_ledger.md
  src/qtrm_mm/losses/equation_state_binding.py (full)
  scripts/625_train_bpe_gd_preference.py (guarded hook)
  scripts/627_run_stage119_equation_probe.py (self-contained runnable)

status:
  active L1 diagnostic probe

current read (direct run 2026-05-26, synthetic algebra traps):
  before_exact: 0.000
  after_exact: 0.125
  ablation_drop: -0.5625 (proxy model)
  language_proxy: 0.75
  aux active and decreasing
  verdict: probe

  Small exact lift on target family under the new compute_equation_state_binding_loss
  (logit margin on final components + readback enforcement). Ablation not yet causal
  in micro-proxy. Language held.

next expected move (factual only):
  Real anchor continuation from Stage117/118 checkpoint using the now-runnable
  627/625 integration + recurrent state capture. Re-record numbers on actual data.
  Promote only on same-LM-metric exact lift + state-off drop.

### RI-4 Dynamic Slot Memory (Inference Write Lock Unlocked)

file:
  2026-05-29-ri4-inference-write-lock-unlocked.md

status:
  active

current read:
  Dynamic selective slot updates are now fully integrated and executed inside 
  OneBodyParallelHybridBlock.forward at every micro-step during inference.
  
  Heldout 72-case reasoning results (step 50 checkpoint):
    Slots-On (Active Memory): 34.72%
    Slots-Off (Memory Ablated): 29.17%
    Ablation Margin: +5.55% (4 cases)

### Donorless Born-One-Body Revival (Parallel Track)

file:
  2026-05-29-donorless-one-body-revival-plan.md

status:
  planning / active investigation

current read:
  Per user request (2026-05-29), after RI-4 memory write lock was resolved in the hybrid track,
  we are now also advancing the pure donorless / HRM-Text-style born-one-body track in parallel.
  Goal: uncontaminated architectural evidence that the recurrent core + dynamic memory system
  delivers reasoning gains with zero external backbone. Minimal viable revival plan documented.

next expected move:
  Surprise-Driven Write Trigger tuning: Condition slot updates on L2 surprise relative to slot std
  to down-weight low-utility/non-surprising memory writes and break the 34.72% ceiling.

### LoRA-Steered Loop LM Recurrent Reasoning (M1/M2 Restoration)

file:
  2026-06-missing-inductive-biases-restoration-roadmap.md

status:
  active / completed M1 & M2

current read:
  We successfully completed the M1 and M2 phases of the 2026-06 Restoration Roadmap. We wired and
  activated the learnable elastic depth policy (dynamic early-exiting & supervised BCE Halting Loss)
  and trained loop-wise Mythos LoRA adapters to steer the latent manifold under 8 steps of recurrence.
  
  Causal forced-choice evaluation verified that under 8 core steps, the trained Loop LM achieves
  perfect correct mathematical answers (exact match '300015' on list-arithmetic), successfully
  healing the latent drift bottleneck where unadapted baselines completely fail.

next expected move:
  Transition to M3 (Full 5.56 Composite Rehearsal Curriculum) and evaluate downstream hard-family
  state ablation margins under long-horizon multi-step reasoning.
