# Training Diagnostics

Purpose: decide quickly whether a QTRM training run is learning, collapsed, or
structurally miswired before spending hours on a long run.

Required probes:

1. Donor-only baseline generation.
   If Qwen3.5 alone generates coherent text and QTRM generates repeated tokens,
   the problem is in adapter training, target construction, decoding, or config
   alignment rather than in the donor checkpoint.

2. Tiny overfit run.
   Train on a fixed 32-128 sample shard. Loss should drop aggressively. Failure
   to overfit is a hard signal for label shift, tokenizer mismatch, masking
   errors, frozen trainable path, or incompatible projection sizes.

3. Train/validation loss split.
   Both losses flat means optimization or architecture failure. Train down with
   validation flat means data/generalization failure. Validation down with bad
   free generation means decoding or exposure-bias diagnostics are next.

4. Teacher-forced target-token rank.
   Track the rank/probability of the true next token. This catches progress
   earlier than free-running generation and distinguishes bad sampling from bad
   learning.

5. Logit entropy and repetition checks.
   Repetition such as one token dominating every step should be treated as a
   collapse signal. Log top-k probabilities, entropy, EOS/special-token rates,
   and repeated n-gram counts.

6. Loss-curve extrapolation.
   Pilot runs should log enough points to fit or at least inspect early loss
   slope and deceleration. A long run is justified only if the pilot curve and
   validation probes show consistent movement.

7. Data-quality and trace-shape audit.
   Before increasing steps, inspect examples for prompt/target boundaries,
   repeated boilerplate, special-token leakage, low-entropy synthetic outputs,
   and language/domain imbalance. Karpathy-style cognitive-core arguments imply
   better traces, not simply more noisy tokens.

8. Padding-mask audit.
   When using an HF tokenizer, do not assume the pad token id is `0`. Verify the
   dataset sample carries an `attention_mask` derived from the tokenizer pad id.
   Otherwise padded positions become LM targets and long runs can learn padding
   or special-token artifacts.

9. Worked-before/reference-contract audit.
   If a run fails, do not immediately add a new auxiliary loss or invent a new
   architecture name. First reopen the last working or least-bad run and the
   closest official reference implementation. Record the exact command,
   checkpoint, data shard, optimizer settings, and the metric that made the
   prior run better. Then diff concrete contracts: input/label shift,
   prefix/causal attention, generation prefill, stop-token handling, optimizer,
   warmup, global token batch, weight decay, beta values, and evaluation gates.

10. HRM-Text exact-contract gate.
    For born-one-body recurrent language experiments, HRM-Text is the reference
    standard until superseded. Before promoting GRAM/PTRM/GDN2 changes, first
    match the HRM-Text contract closely enough to make a fair comparison:
    `dataset_new.py` PrefixLM shift, `lm_head.py` response-token CE/accuracy,
    `simple_inference_engine.py` prefill/decode generation, and
    `config/cfg_pretrain.yaml` optimizer rhythm (`lr=2.2e-4`,
    warmup `2000`, `weight_decay=0.1`, `beta2=0.95`, large token batch).
    If this baseline is not in place, failed generation is not evidence against
    the architecture; it is evidence that the training contract is not yet
    controlled.

11. Consecutive-reject stop rule.
    If two or more runs reject for the same plain-language failure, stop
    ideating. The next action must be a worked-before audit: reopen the best
    accepted run, the least-bad reject, and the official/reference path; write
    the exact contract diff; then restore or change only one contract. Do not
    launch another acronym, auxiliary loss, or backbone swap until this table
    exists.

    Required table before a new run:

    ```text
    anchor/reference | exact command/checkpoint | what worked | failed run diff
    expected causal repair | accept/reject gate
    ```

    문과적으로 말하면: 답안지를 또 새로 꾸미기 전에, 먼저 "언제는 왜 답이
    써졌는지"를 다시 열어봐야 한다. 성공했던 호흡을 잃어버린 상태에서 새
    장식을 붙이면 학습이 아니라 삽질이 된다.

Current QTRM interpretation:

- `lm=5.42` with a 248k-token Qwen vocabulary is better than random cross
  entropy, but repeated `Freeze` generation suggests collapse or misalignment.
- The next engineering step is not just more steps. Add donor-only baseline,
  tiny-overfit, validation loss, target-token-rank, and logit-entropy probes.
- If those probes fail, scaling from 5k to larger step counts is expected to
  amplify the same attractor rather than produce a stable cognitive core.
- A donor-on tiny-overfit probe on 16 fixed samples reached `rank=1.0` and
  `top1=1.0`, so the current QTRM path can learn small data. This shifts the
  main failure search toward checkpoint/code mismatch, dirty training data, and
  masking/eval gaps.
- A 120-step fixed-mask real-data pilot did not reproduce the `Freeze` top
  token, but it produced a punctuation/function-word loop. Treat this as
  undertrained/high-frequency collapse, not success.
- A 500-step clean text-only pilot also did not reproduce `Freeze`, but still
  collapsed into dialogue markers and high-frequency English phrases such as
  `world of the world`. The padding fix and data filter removed one failure
  mode; they did not make the architecture/data objective sufficient yet.
- Autoregressive diagnostics and inference must refresh donor states for the
  full current token sequence. Keeping only the initial prompt donor states is a
  train/inference mismatch because training encodes the complete sample.
- A 300-step LM-only ablation with `loss_jepa_weight=0` and `loss_aux_weight=0`
  still collapsed into high-frequency text. This makes JEPA/aux unlikely to be
  the primary cause of the current free-generation failure.
- The current next gate is
  `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` via
  `scripts/105_run_current_arch_pretrain_probe.sh`: 2000 steps, clean text
  pilot data, Qwen donor logits as the base policy, QTRM residual scale `0.10`,
  workspace enabled, and JEPA/aux disabled. Passing this gate means train loss,
  target-token rank, entropy, and greedy repetition stay healthy after a longer
  continued-pretraining probe. It still does not prove reasoning gains over the
  donor; that requires donor-only versus QTRM residual held-out tasks.

Stage80-84 HRM-Text lesson:

- Stage80/81 showed that eval loss can improve while first-response generation
  is still broken. On 512 teacher-forced first response positions, top-1 was
  `<|box_end|>` for every row and first-token accuracy was `0%`.
- Stage82 proved that directly punishing early `<|box_end|>` can remove the
  immediate stop but cause `111111...` repetition. This is a failed workaround,
  not a causal language fix.
- Stage83 showed that row-balanced response loss did not immediately restore
  first-token accuracy. More auxiliary loss tuning is therefore lower priority
  than restoring the HRM-Text reference training contract.
- Stage84 restored the HRM-Text optimizer rhythm first: no auxiliary loss,
  `lr=2.2e-4`, `lr_warmup_steps=2000`, `weight_decay=0.1`, `beta2=0.95`.
  This immediately fixed the actual failure mode: by step 3000,
  first-response-token accuracy reached `92.2%`, `<|box_end|>` first-token
  top-1 rate reached `0%`, and a 5-row greedy generation probe produced 5/5
  exact answers. This is the current proof that reference-contract restoration
  beat special-loss tinkering.
- Current rule: when HRM-Text-like from-scratch language training fails, first
  repair the reference contract and compare to the best previous run. Do not
  keep stacking special losses unless the exact-contract baseline shows the
  same failure and the new loss has a narrow falsifiable gate.

Current success anchor:

```text
Stage84_LOCAL82M_HRMTEXT_OPTIM_BS64
  command family:
    scripts/534_train_native_prefixlm_dataio.py
  key contract:
    no special auxiliary loss,
    HRM-Text-like optimizer rhythm,
    lr=2.2e-4,
    lr_warmup_steps=2000,
    weight_decay=0.1,
    beta2=0.95
  observed training-split gate at step 3000:
    first-response-token accuracy 92.2%,
    first-token <|box_end|> top1 rate 0%,
    5/5 greedy sample exact on the probed rows
  observed heldout teacher-forced gate at step 5000:
    eval_loss 0.0879 on epoch1 eval target tokens
  observed heldout generation gate at step 7000:
    first-response-token accuracy 98.4% on 512 epoch1 rows,
    first-token <|box_end|> top1 rate 0%,
    first-token gold probability 0.970,
    greedy free generation exact 59/64 on epoch1 rows,
    starts-with-<|box_end|> 0/64,
    repeated-token loops 0/64
  expanded heldout generation gate at step 8000:
    first-response-token accuracy 98.3% on 2048 epoch1 rows,
    first-token <|box_end|> top1 rate 0%,
    first-token gold probability 0.971,
    greedy free generation exact 468/512 on epoch1 rows,
    starts-with-<|box_end|> 0/512,
    ended-with-<|box_end|> 500/512,
    repeated-token loops 1/512
  warning:
    this is a strong small heldout gate, not a full capability claim. Before
    broad capability claims, expand beyond this math/Data-IO slice and keep
    comparing against this exact anchor.
  later local retention gate at step 17000:
    first-response-token accuracy 99.2% on 4096 epoch1 rows,
    first-token <|box_end|> top1 rate 0%,
    first-token gold probability 0.988,
    greedy free generation exact 1890/2048 on epoch1 rows,
    starts-with-<|box_end|> 0/2048,
    ended-with-<|box_end|> 1963/2048,
    repeated-token loops 0/2048.
```

Current active run board:

```text
Local Stage84:
  status:
    keep running as the success anchor
  latest checked:
    step 50000 eval_loss 0.00634
  role:
    local anchor for heldout/OOD generation gates

DGX Stage80:
  status:
    rejected and stopped
  reason:
    old optimizer contract, lr=4e-4, no HRM-Text warmup/beta2 contract
  observed failure:
    eval_loss stayed around 4.48 through step 36000
  lesson:
    do not burn DGX on pre-Stage84 contracts when Stage84 has already shown the
    correct training rhythm.

DGX Stage85:
  status:
    launched as the corrected DGX run
  contract:
    aux-free, lr=2.2e-4, lr_warmup_steps=2000, beta1=0.9, beta2=0.95,
    weight_decay=0.1, batch_size=64
  early signal:
    step 400 train loss fell from 11.10 to 5.80 during warmup
    step 1000 eval_loss 2.366, matching the Stage84 step1000 curve
    step 1000 heldout generation is still early, not rejected:
      first-response-token accuracy 27.5% on 512 epoch1 rows,
      first-token <|box_end|> top1 rate 0%,
      greedy free generation exact 3/64
    this matches the local Stage84 early phase, where first-response accuracy
    was only about 28.5% around step 1000 before rising sharply later.
    step 2000 eval_loss 0.954, slightly ahead of the Stage84 step2000 0.990
    curve.
    step 2000 heldout generation:
      first-response-token accuracy 70.7% on 512 epoch1 rows,
      first-token <|box_end|> top1 rate 0%,
      greedy free generation exact 30/64,
      repeated-token loops 0/64.
    step 3000 eval_loss 0.293, continuing the accepted Stage84-like curve.
    step 3000 heldout generation:
      first-response-token accuracy 93.4% on 512 epoch1 rows,
      first-token <|box_end|> top1 rate 0%,
      first-token gold probability 0.884,
      greedy free generation exact 45/64,
      starts-with-<|box_end|> 0/64,
      ended-with-<|box_end|> 64/64,
      repeated-token loops 0/64.
    step 6000 eval_loss 0.0755.
    step 7000 eval_loss 0.0503, still following the accepted Stage84-like
    downward curve.
    step 10000 eval_loss 0.0429.
  dgx runtime note:
    generation/eval commands on GB10 must pin the ptxas contract explicitly:
      REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
      TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
    If either is missing, the launcher should stop with
    `missing required ptxas contract` or `missing required ptxas`.
    Stage95 showed the sharper rule: the backend name is not enough.
    `official_gated_delta2` must fail fast when the official module, ptxas, or
    CUDA kernel path is unavailable. Do not call a run official unless
    `actual_delta_runtime=official_runtime` and
    `delta_runtime_fallback_active_count=0`.
  role:
    reproduce Stage84's success rhythm on DGX before scaling model size or
    adding architecture mechanisms.

Today gates:
  local:
    keep Stage84 running, then rerun heldout generation at a later checkpoint
    with at least 2048 rows.
  dgx:
    Stage85 must reach the Stage84-like curve before it is allowed to scale:
    eval_loss should fall sharply by step 1000/2000, then run the same heldout
    first-response/free-generation gate.
  reject rule:
    if DGX does not follow the Stage84 curve, do not add a new mechanism. First
    diff data, script, optimizer, tokenizer, CUDA/backend, and checkpoint
    contract.
```

Local diagnostic scripts:

- `scripts/91_donor_only_generate.sh`: Qwen donor-only generation baseline.
- `scripts/92_eval_qtrm_logits.py`: checkpoint logit diagnostics, target-token
  rank, entropy, top-k next token, and greedy repetition stats.
- `scripts/93_tiny_overfit_donor_adapter.sh`: fixed-shard overfit probe for
  deciding whether the QTRM path can learn at all.
- `scripts/94_build_clean_pilot_data.sh`: builds a text-only filtered pilot
  dataset with boilerplate, image-token, and repetition filters.
- `scripts/105_run_current_arch_pretrain_probe.sh`: runs the current-architecture
  2000-step residual pretraining viability probe and writes post-eval artifacts.
- `scripts/539_eval_prefixlm_generation_gate.py`: evaluates HRM-Text Data-IO
  PrefixLM checkpoints with heldout first-response-token and free-generation
  gates, including `<|box_end|>` first-token and repetition checks.

Memory and optimizer policy for scaling:

- 문과적으로 말하면 AdamW는 모델마다 "개인 장부 두 권"을 더 들고 훈련하는
  방식이다. 1B급 모델에서는 모델 자체보다 이 장부가 먼저 VRAM을 먹는다.
  그래서 큰 모델로 갈 때 첫 최적화 대상은 architecture 이름이 아니라
  optimizer state다.
- `scripts/534_train_native_prefixlm_dataio.py` supports
  `--optimizer adamw`, `adamw8bit`, `paged_adamw8bit`, `galore_adamw`, and
  `galore_adamw8bit`, plus bitsandbytes `ademamix8bit` and
  `paged_ademamix8bit`. `auto` prefers GaLore 8-bit on CUDA when installed,
  then bitsandbytes 8-bit AdamW, then plain AdamW.
- Local CUDA smoke verified that `--optimizer auto` resolves to
  `galore_adamw8bit` with the current environment.
- Local CUDA smoke also verified `--optimizer paged_ademamix8bit`.
  AdEMAMix is a learning-dynamics optimizer, not primarily a memory optimizer;
  use its 8-bit/paged form if trying it on 4090.
- Plain AdamW is acceptable for 82M/300M/700M pilots. 900M-1.15B class runs
  should use `adamw8bit`, `paged_adamw8bit`, or `galore_adamw8bit` before
  reducing model size. This preserves the one-body training path while cutting
  the optimizer-state bottleneck.
- PyTorch 2.6 changed `torch.load` to default `weights_only=True`. Full training
  checkpoints that include optimizer state, especially GaLore projector state,
  must be resumed with `weights_only=False` when the checkpoint was produced by
  this trusted training job. Otherwise a healthy run can fail to resume even
  though the checkpoint is intact.
- Apple CCE / Liger fused linear CE is directly relevant to our large-vocab
  PrefixLM loss. It avoids materializing the full supervised-token-by-vocab
  logits table before CE. `scripts/534_train_native_prefixlm_dataio.py` now has
  `--loss-kernel torch|auto|liger_fused_linear_ce`; local CUDA smoke verified
  that `auto` resolves to Liger and backward works.
- Attention/kernel memory is a separate lever from optimizer memory. PyTorch
  SDPA has flash and memory-efficient kernels enabled in the local CUDA stack;
  avoid dense masks when possible so the fused path is not blocked. For pure
  causal unmasked self-attention, `GroupedQueryAttention` now uses
  `is_causal=True` instead of constructing a dense `T x T` causal mask.
- Current PrefixLM training does not pass padding masks into the model forward,
  so the causal SDPA fast path is usable in the active training path. Padding
  tokens are ignored in the loss but still consume compute. The next data-side
  efficiency lever is therefore length packing or variable-length batching, not
  MLA.
- Current safe packing step: batches are trimmed to each batch's max valid
  length by default (`--trim-batch-to-max-length`), with
  `--no-trim-batch-to-max-length` left for fixed-shape debugging.
- Implemented length bucketing step:
  - `scripts/534_train_native_prefixlm_dataio.py` has
    `--length-bucketed-batches` and `--length-bucket-size-multiplier`.
  - It keeps randomization at the bucket/batch level but groups similar row
    lengths so batch trimming wastes fewer padding tokens.
  - CUDA smoke verified the combined fitting stack:
    `--length-bucketed-batches --activation-checkpointing --loss-kernel auto
    --optimizer paged_ademamix8bit`.
  - Keep it opt-in when comparing exactly against the Stage84 HRM-Text contract;
    enable it for 4090 fitting/scaling runs.
- 4090 capability boundary:
  - realistic: 82M/300M/700M pilots, and careful 900M-1.1B experiments with
    8-bit/paged optimizer, Liger fused CE, trimming, small microbatch, and
    activation care.
  - not realistic: full 27B training, unconstrained long-context pretraining,
    or claiming HRM-Text-1B parity from a tiny token budget.
  - plain-language rule: these optimizations clear the desk, but they do not
    make the room infinitely large.

913M Stage88/91 operational lesson:

- Local Stage88 stopped at the step 5000 eval gate, not because it converged or
  because training loss collapsed, but because one eval batch stayed non-finite
  even after the previous fallback path. This is a judging-system failure mode,
  not automatic architecture evidence.
- The eval path now records three separate counters:
  `eval_nonfinite_batches`, `eval_fallback_batches`, and
  `eval_unresolved_nonfinite_batches`. A single unresolved eval batch is a
  yellow flag. It becomes a reject signal only if it repeats across eval gates
  or is accompanied by train loss/grad collapse.
- Stage88 resumed from the trusted step 4000 checkpoint after explicitly loading
  the full checkpoint with `weights_only=False`. It passed the old failure point:
  step 5000 eval produced `eval_loss=2.9011` with one unresolved batch, then
  step 6000 eval recovered to `eval_loss=2.8013` with all nonfinite counters at
  zero. It then passed step 7000, 8000, 9000, and 10000 without any repeated
  nonfinite eval batch.
- 문과적으로 말하면: 시험지 한 줄이 번졌다고 학생을 퇴학시키지 않는다. 같은
  번짐이 계속 나오면 펜/손/책상 문제를 조사하고, 한 번만 나오고 다음 시험지가
  깨끗하면 계속 풀게 둔다.
- DGX Stage90 proved explicit `galore_adamw8bit` can train/evaluate/checkpoint a
  913M batch-8 run. The stall came after the learning work, during Aim metadata
  close/write on `/mnt/data4tb/qtrm_aim_stage90`, with the process in
  `rq_qos_wait`. Therefore DGX long runs should avoid Aim or move Aim/checkpoint
  I/O off the hot path until async checkpointing is implemented.
- DGX Stage91 is the corrected operating pattern for now: no Aim logging,
  TensorBoard only, larger checkpoint interval, and `TRITON_PTXAS_PATH` set for
  GB10/sm_121 compatibility. The DGX "special advantage" should first be spent
  on bigger batch and uninterrupted wall-clock, not on NVFP4 experiments while
  numeric stability is still being characterized. Stage91 passed the important
  step 2000 gate: eval loss improved to `2.3728`, nonfinite counters were zero,
  and a 4.6GB `last.pt` checkpoint wrote successfully while training continued.

Current 913M active board:

```text
Batch comparison rule:
  do not compare local and DGX by raw step alone.
  Local Stage88 uses batch_size=2; DGX Stage91 uses batch_size=8.
  Therefore DGX sees roughly 4x more examples per optimizer step, while each
  step takes longer. The fair comparison axis is `target_tokens_seen`, then
  wall-clock/target-token throughput.

  plain-language:
    step is "how many times the teacher said update."
    target_tokens_seen is "how many answer tokens the student actually studied."
    batch size is "how many exam sheets are open on the desk at once."
    A bigger desk makes the gradient less noisy; it does not prove the student
    has a different brain.

  current evidence:
    around the same target-token exposure:
      Local Stage88 step8000 target_tokens_seen=149736 eval_loss=2.6552
      DGX Stage91 step2000 target_tokens_seen=150407 eval_loss=2.3728
      Local Stage88 step12000 target_tokens_seen=225799 eval_loss=2.2197
      DGX Stage91 step3000 target_tokens_seen=226890 eval_loss=2.0130
      Local Stage88 step16000 target_tokens_seen=300275 eval_loss=2.1432
      DGX Stage91 step4000 target_tokens_seen=301653 eval_loss=1.6589
      Local Stage88 step17000 target_tokens_seen=319428 eval_loss=2.1802
      Local Stage88 step18000 target_tokens_seen=338410 eval_loss=2.2023
      Local Stage88 step20000 target_tokens_seen=375155 eval_loss=2.1424
      DGX Stage91 step5000 target_tokens_seen=378066 eval_loss=1.2554
      Local Stage88 step21000 target_tokens_seen=395064 eval_loss=1.9579
      DGX Stage91 step6000 target_tokens_seen=450733 eval_loss=1.1206
      DGX Stage91 step7000 target_tokens_seen=525560 eval_loss=0.9180
    This is evidence that the larger DGX batch is giving a cleaner early
    optimization signal, but not yet a final capability claim. DGX step3000 had
    3 unresolved nonfinite eval batches, then step4000 through step6000 each had
    1 unresolved batch. Step7000 then cleared all nonfinite counters while
    eval_loss fell below 1.0, so the numeric yellow flag is weaker than before.
    Generation gates remain required.

Local Stage88 913M:
  command shape:
    batch_size=2, galore_adamw8bit, loss_kernel=auto, length buckets,
    resumed from trusted step4000 checkpoint
  latest meaningful gates:
    step5000 eval_loss 2.9011,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=1
    step6000 eval_loss 2.8013,
      all eval nonfinite counters returned to 0
    step7000 eval_loss 2.5741,
      all eval nonfinite counters remained 0
    step8000 eval_loss 2.6552,
      all eval nonfinite counters remained 0
    step9000 eval_loss 2.5522,
      all eval nonfinite counters remained 0,
      checkpoint write succeeded
    step10000 eval_loss 2.5246,
      all eval nonfinite counters remained 0,
      checkpoint write succeeded
    step11000 eval_loss 2.4222,
      all eval nonfinite counters remained 0,
      checkpoint write succeeded
    step12000 eval_loss 2.2197,
      all eval nonfinite counters remained 0,
      checkpoint write succeeded
    step13000 eval_loss 2.2376,
      all eval nonfinite counters remained 0
    step14000 eval_loss 2.1944,
      all eval nonfinite counters remained 0
    step15000 eval_loss 2.2674,
      all eval nonfinite counters remained 0
    step16000 eval_loss 2.1432,
      all eval nonfinite counters remained 0
    step17000 eval_loss 2.1802,
      all eval nonfinite counters remained 0
    step18000 eval_loss 2.2023,
      all eval nonfinite counters remained 0
    step19000 eval_loss 2.2409,
      all eval nonfinite counters remained 0
    step20000 eval_loss 2.1424,
      all eval nonfinite counters remained 0,
      checkpoint write succeeded at 2026-05-23 20:52 KST
    step21000 eval_loss 1.9579,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=0
    step30000 final:
      final_logged_loss=2.9522,
      eval_loss=1.7023,
      target_tokens_seen=561676,
      eval_nonfinite_batches=0,
      eval_fallback_batches=0,
      eval_unresolved_nonfinite_batches=0,
      final checkpoint wrote at 2026-05-23 21:26 KST.
    generation gate on epoch1:
      first-response-token accuracy 56.1% on 512 rows,
      first-token <|box_end|> top1 rate 0%,
      first-token gold probability 0.353,
      greedy free generation exact 13/64,
      starts-with-<|box_end|> 0/64,
      ended-with-<|box_end|> 60/64,
      repeated-token loops 0/64.
  interpretation:
    accepted as a local fitting/speaking proof, but not as the best training
    recipe. It learned to answer without EOA collapse or repetition loops, yet
    batch2 reached much weaker answer quality than DGX batch8 at a similar
    target-token scale. In plain language, the student can now speak, but the
    small desk made the lesson much noisier.

DGX Stage91 913M:
  command shape:
    batch_size=8, galore_adamw8bit, loss_kernel=auto, TensorBoard only,
    Aim disabled, checkpoint_every=2000, TRITON_PTXAS_PATH set
  latest meaningful gates:
    step1 eval_loss 11.1099,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=0
    step1000 train loss already reached about 2.014 during warmup
    step1000 eval_loss 2.9486,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=0
    step2000 train loss 1.3121 after warmup reached full lr
    step2000 eval_loss 2.3728,
      eval_nonfinite_batches=0,
      eval_fallback_batches=0,
      eval_unresolved_nonfinite_batches=0
    step2000 checkpoint:
      4.6GB last.pt wrote successfully at 2026-05-23 20:20 KST,
      process continued to step2050/2100 afterward
    step3000 train loss 1.8991
    step3000 eval_loss 2.0130,
      eval_nonfinite_batches=3,
      eval_fallback_batches=3,
      eval_unresolved_nonfinite_batches=3
    step4000 train loss 1.2427
    step4000 eval_loss 1.6589,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=1
    step4000 checkpoint:
      4.6GB last.pt wrote successfully at 2026-05-23 20:36 KST,
      process entered `rq_qos_wait` during checkpoint flush, then resumed and
      reached step4050. The step4000->4050 interval was about 187 seconds, so
      checkpoint I/O is now a real DGX throughput bottleneck.
    live tail:
      process is still active and reached step4600 with
      target_tokens_seen=347005. It recovered from the step4000 checkpoint I/O
      stall and is again producing regular training logs.
    step5000 eval_loss 1.2554,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=1
    step6000 train loss 0.3701
    step6000 eval_loss 1.1206,
      eval_nonfinite_batches=1,
      eval_fallback_batches=1,
      eval_unresolved_nonfinite_batches=1
    step6000 checkpoint:
      4.6GB last.pt wrote successfully at 2026-05-23 20:53 KST, and the
      process resumed to step6050. The step6000->6050 interval was about 187
      seconds, matching the earlier checkpoint flush stall.
    live tail:
      process is still active and reached step6150 with
      target_tokens_seen=461991.
    step7000 eval_loss 0.9180,
      eval_nonfinite_batches=0,
      eval_fallback_batches=0,
      eval_unresolved_nonfinite_batches=0
    step8000 final:
      final_logged_loss=0.0736,
      eval_loss=0.7308,
      target_tokens_seen=601806,
      eval_nonfinite_batches=3,
      eval_fallback_batches=3,
      eval_unresolved_nonfinite_batches=3,
      final checkpoint wrote at 2026-05-23 21:11 KST,
      report wrote at 2026-05-23 21:13 KST.
    generation gate on epoch1:
      first-response-token accuracy 91.4% on 512 rows,
      first-token <|box_end|> top1 rate 0%,
      first-token gold probability 0.861,
      greedy free generation exact 46/64,
      starts-with-<|box_end|> 0/64,
      ended-with-<|box_end|> 61/64,
      repeated-token loops 0/64.
  interpretation:
    this is now a real accepted learning signal for the 913M born-one-body
    path, not merely a fitting smoke. The model is speaking answers without EOA
    collapse or repetition loops after only about 602k supervised answer tokens.
    Keep two yellow flags separate. First, step8000 still had 3 unresolved eval
    batches, so numeric eval instrumentation remains necessary in the next run.
    The trainer now records unresolved eval batch indices and hidden
    nonfinite-element counts for future runs; Stage91 loaded the older code
    before that instrumentation was copied over. Second,
    checkpoint I/O is costing minutes. If the run is extended, increase
    checkpoint interval or implement safer async/model-only checkpointing before
    spending more DGX time on frequent full optimizer-state saves.

DGX Stage92 913M continuation:

- Role:
  continue Stage91 from its accepted `last.pt` instead of launching a new
  architecture. This is the same 913M born-one-body PrefixLM model, same
  HRM-Text-like optimizer rhythm, batch8, and `train_think_steps=2`.
- Start:
  resumed from
  `/mnt/data4tb/wgram-lm/local_eval/20260523_STAGE91_DGX913M_BS8_GALORE_NOAIM_500K/last.pt`
  at `step=8000`, `target_tokens_seen=601806`.
- Observed continuation:
  step9000 `eval_loss=0.7973`,
  `target_tokens_seen=676044`,
  `eval_nonfinite_batches=2`,
  `eval_unresolved_nonfinite_batches=1`.
  step10000 `eval_loss=0.6032`,
  `target_tokens_seen=752584`,
  and all eval nonfinite/fallback/unresolved counters returned to `0`.
  step11000 `eval_loss=0.6002`,
  `target_tokens_seen=827579`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step12000 `eval_loss=0.5635`,
  `target_tokens_seen=903020`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step13000 `eval_loss=0.5259`,
  `target_tokens_seen=976211`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step14000 `eval_loss=0.4745`,
  `target_tokens_seen=1053541`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step15000 `eval_loss=0.4139`,
  `target_tokens_seen=1128351`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step16000 `eval_loss=0.4467`,
  `target_tokens_seen=1203136`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step17000 `eval_loss=0.4571`,
  `target_tokens_seen=1277310`,
  with all nonfinite/fallback/unresolved counters still `0`.
  step18000 `eval_loss=0.4156`,
  `target_tokens_seen=1353633`,
  with all nonfinite/fallback/unresolved counters still `0`.
- Direct-answer generation gate at step12000:
  report
  `/mnt/data4tb/wgram-lm/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K/gates/direct_generation_gate_step12000_512_64.json`.
  The gate filters `epoch_1` to `condition=direct`, keeping `13664` of
  `14981` valid rows. It reached first-response-token accuracy `95.3%` on
  512 rows, first-token gold probability `0.933`, and greedy free-generation
  exact `50/64 = 78.125%`. It had starts-with-`<|box_end|>` `0/64`,
  ended-with-`<|box_end|>` `64/64`, and repeated-token loops `0/64`.
- Interpretation:
  step12000 is both a cleaner teacher-forced eval and a stronger direct-answer
  speaker than the Stage91 64-row all-condition gate. 문과적으로 말하면: 필기
  점수도 좋아졌고, direct 문제 구술시험에서도 답을 직접 말하는 힘이
  올라왔다.
- Direct-answer generation gate at step14000:
  report
  `/mnt/data4tb/wgram-lm/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K/gates/direct_generation_gate_next_model_512_512.json`.
  The gate uses the same `epoch_1`, `condition=direct` slice and expands free
  generation from 64 rows to 512 rows. It reached first-response-token accuracy
  `95.3%` on 512 rows, first-token gold probability `0.932`, greedy
  free-generation exact `441/512 = 86.13%`, starts-with-`<|box_end|>` `0/512`,
  ended-with-`<|box_end|>` `510/512`, and repeated-token loops `0/512`.
- Step14000 interpretation:
  this is the strongest Stage92 evidence so far. The teacher-forced eval kept
  improving, the 512-row direct-answer free-generation gate is substantially
  above the local Qwen3.6 GGUF proxy on the same narrow answer-only slice, and
  the model still shows no early-EOA or repetition-collapse symptom.
  문과적으로 말하면: 이제는 64문항 찍기 운이 아니라, 512문항짜리 큰 시험지에서도
  답안지를 안정적으로 채우기 시작했다. 아직 "범용 사고력" 증명은 아니지만,
  HRM-Text식 한 몸 학습 경로가 실제로 자라고 있다는 증거는 강해졌다.
- Step16000 training-only update:
  the run is still active and training-only on DGX. Step15000 became the
  current best teacher-forced eval point (`0.4139`), while step16000 rebounded
  slightly to `0.4467` and step17000 to `0.4571`. Because all
  nonfinite/fallback/unresolved counters remain zero, this is a normal noisy
  learning curve bump, not an architecture rejection signal. Do not interrupt
  the active DGX run for another long Direct/CoT generation gate; wait for the
  next stage boundary or final checkpoint.
  문과적으로 말하면: 시험 점수가 한 번 살짝 출렁였지만, 학생이 쓰러진 것은
  아니다. 지금은 달리기 중간에 붙잡고 구술시험을 시킬 때가 아니라, 코스를
  끝까지 달리게 두고 큰 시험은 정류장에서 봐야 한다.
- Stage92 next-action gate:
  keep the active DGX process training-only until at least the next built-in
  teacher-forced eval. If step18000 or step19000 returns below the step15000
  best (`0.4139`), treat that as renewed learning and continue to the planned
  stage boundary. If two consecutive evals rise above `0.60` or any
  nonfinite/fallback/unresolved counter returns, stop adding compute and inspect
  the checkpoint/data contract before launching generation gates. Long Direct
  or CoT generation remains a separate exam after stopping or at a final/stage
  checkpoint, not an active-training side job.
  문과적으로 말하면: 다음 시험에서 다시 좋아지면 계속 달린다. 두 번 연속
  크게 나빠지거나 답안지가 깨지기 시작하면 운동을 멈추고 건강검진을 한다.
  길게 말시키는 구술시험은 운동 중이 아니라 휴식 지점에서 본다.
- Step18000 gate update:
  step18000 recovered to `0.4156`, very close to the step15000 best `0.4139`
  but not below it. This is not a new best checkpoint, but it does falsify the
  collapse interpretation of the step16000/17000 bump. All instability counters
  remain zero, so keep DGX training-only and wait for step19000 or the planned
  stage boundary before any long generation gate.
  문과적으로 말하면: 16000/17000에서 잠깐 흔들린 뒤 18000에서 거의 원래
  컨디션으로 돌아왔다. 아직 최고 점수를 새로 쓴 것은 아니지만, 학생이
  무너진 것은 아니다. 계속 달리게 둔다.
- Step18000 checkpoint boundary:
  `last_model.pt` was refreshed at step18000 and is about `3.5GB`; the Stage92
  run directory is about `8.0GB` with `last.pt` plus `last_model.pt`. This
  explains a short post-eval pause as checkpoint I/O, not a training failure.
  Keep the active run untouched. If a long generation exam is needed later,
  use this or a later selected stage-boundary checkpoint, not every transient
  checkpoint.
  문과적으로 말하면: 학생이 답안지를 새로 복사해서 보관하는 시간이다. 공부가
  멈춘 게 아니라 기록을 남기는 시간이고, 이때 옆에서 긴 구술시험을 걸면
  오히려 흐름만 망친다.

Stage88 vs Stage91 plain-language conclusion:

- Same 913M body, same HRM-Text-like contract, same `train_think_steps=2`.
- Local batch2 after `561676` target tokens:
  eval_loss `1.7023`, first-response `56.1%`, greedy exact `13/64`.
- DGX batch8 after `601806` target tokens:
  eval_loss `0.7308`, first-response `91.4%`, greedy exact `46/64`.
- Therefore the big jump here is not an acronym change. It is the larger
  effective batch and cleaner gradient path making the same born-one-body model
  actually learn to speak. 문과적으로 말하면: 같은 학생에게 같은 교재를
  줬는데, DGX는 넓은 책상에 여러 문제를 펼쳐놓고 규칙을 잡았고, 로컬은
  좁은 책상에서 한두 문제씩 보느라 방향이 훨씬 흔들렸다.

Local vs DGX operating rule:

- Local 4090 is now the maintenance bench, not the judge.
- Use local for:
  code-path smoke, 1-step/100-step backward checks, CLI compatibility,
  checkpoint load/save sanity, tiny generation gate validation, tokenizer/data
  contract checks, quick NaN/collapse detection, Qwen/proxy comparison baselines,
  and long free-generation gates only when the target checkpoint is already
  local.
- Do not use local for:
  accepted/reject decisions on 913M+ learning efficiency, long from-scratch
  growth claims, or architecture capability judgment.
- DGX is the training ground and the judge for accepted/reject decisions.
- During an active DGX training run, do not run long free-generation gates on
  the DGX GPU. Direct/CoT generation probes compete with training for the same
  GPU and can slow the run enough to distort throughput. Use the trainer's
  built-in small teacher-forced eval while training is active.
- If a long generation gate is needed:
  1. Prefer local 4090 if the checkpoint is already local.
  2. If the checkpoint is not local and the link is slow, do not copy every
     checkpoint. Wait for a stage boundary or final checkpoint, then copy once.
  3. If same-day evidence is needed before copying, pause or finish the DGX
     training run first, then run the generation gate on DGX as a separate
     evaluation job.
- A bad local long-learning result is not architecture evidence unless DGX
  reproduces the same failure under the accepted batch/data/optimizer contract.
- Plain-language rule:
  로컬은 정비소다. 엔진이 켜지는지, 핸들이 붙어 있는지, 계기판이 읽히는지
  보는 곳이다. DGX는 주행 시험장이다. 실제로 속도가 나는지, 오래 달리는지,
  코너를 도는지는 DGX에서 판단한다.
  긴 구술시험은 주행 중인 차 안에서 치르지 않는다. 차는 DGX 도로에서 계속
  달리게 두고, 구술시험은 차를 세운 뒤 하거나 로컬 시험장으로 옮겨서 본다.

Local Qwen3.6 comparison bench:

- Local should prepare comparison exams while DGX trains. Do not spend 50+
  minutes copying a 1.3GB+ DGX checkpoint over a slow link merely to run a gate
  locally; run Stage92 generation gates on DGX and use local for small judge
  plumbing.
- `scripts/541_build_prefixlm_openai_suite.py` materializes HRM-Text PrefixLM
  sampled rows into OpenAI-compatible JSONL for Qwen3.6-style baselines.
  Current local artifact:
  `/tmp/qtrm_eval/qwen36_prefixlm_epoch1_direct_512/cases.jsonl`.
- The suite uses `epoch_1`, `condition=direct`, 512 rows, and prompt protocol
  `hrm_text_data_io_answer_only_v1`. It strips `<|im_start|>`, condition
  markers, `<|im_end|>`, and `<|box_end|>` so the adult baseline receives a
  normal problem and an answer-only target.
- Local smoke:
  `Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL` ran 16 cases from that suite with
  `answer_format=exact_text` and scored `7/16 = 43.75%`. Treat this as a
  wiring/contract smoke only, not a full Qwen3.6 capability number.
- Local 64-case proxy baseline:
  same suite/scorer/model proxy scored `24/64 = 37.5%`, report at
  `/tmp/qtrm_eval/qwen36_prefixlm_epoch1_direct_64/report.json`. This is a
  Qwen3.6 GGUF proxy number on our answer-only direct slice, not a public
  benchmark score and not a full-precision Qwen3.6 claim.
- Local 512-case proxy baseline:
  same suite/scorer/model proxy scored `180/512 = 35.15625%`, report at
  `/tmp/qtrm_eval/qwen36_prefixlm_epoch1_direct_512_full/report.json`. This
  replaces the 16-row smoke and 64-row proxy as the stronger local comparison
  anchor for this narrow direct-answer slice.
- Local CoT proxy smoke:
  `scripts/541_build_prefixlm_openai_suite.py` now supports
  `--prompt-style auto`, so `condition=cot` rows ask the baseline to return the
  full solution and end with the final answer in `\boxed{}`. The OpenAI-compatible
  evaluator now supports `--answer-format boxed_text`, which extracts the final
  boxed expression instead of requiring a byte-for-byte match to the reference
  reasoning trace. This matters because CoT evaluation should ask whether the
  final reasoned answer is right, not whether the model copied the exact same
  prose.
  Current Qwen3.6 GGUF proxy CoT smoke:
  `/tmp/qtrm_eval/qwen36_prefixlm_epoch1_cot_32_smoke/report.json`,
  `6/32 = 18.75%` with `scorer=final boxed text exact match`.
- Same-slice comparison:
  Stage92 step12000 direct generation gate scored `50/64 = 78.125%` on the
  corresponding `epoch_1` direct-answer slice. Stage92 step14000 then expanded
  the same direct slice to 512 free-generation rows and scored
  `441/512 = 86.13%`. This is a narrow HRM-Text Data-IO answer-only comparison,
  not a broad model capability claim. Within that classroom, however, the 913M
  born-one-body model is now answering the worksheet much better than the local
  Qwen3.6 GGUF proxy (`180/512 = 35.16%`).
```

Batch/sequence/activation/checkpointing research map:

- Plain-language invariant:
  - `batch` is how many notebooks are open on the desk.
  - `seq_len` is how long each notebook is.
  - `activation` is the scratch work kept for backprop.
  - `checkpointing` is throwing away scratch work and redoing it later.
  - Therefore one lever cannot solve the whole problem. The correct stack is:
    reduce wasted sequence length, avoid materializing huge temporary tensors,
    compress/shard optimizer ledgers, and only then recompute selected
    activations.
- Paper-backed bottleneck split:
  1. Attention memory: FlashAttention/FlashAttention-2 reduce HBM traffic and
     avoid the standard attention memory blow-up without approximation. This is
     the default answer for long `seq_len` on a single GPU.
     Sources:
       https://arxiv.org/abs/2205.14135
       https://arxiv.org/abs/2307.08691
  2. Loss-head memory: Apple CCE and Liger fused CE avoid materializing the full
     supervised-token-by-vocab logit table. This matters especially for our
     rounded 65k-vocab PrefixLM path.
     Sources:
       https://arxiv.org/abs/2411.09009
       https://arxiv.org/abs/2410.10989
  3. Optimizer memory: GaLore/8-bit GaLore reduce optimizer-state memory while
     keeping full-parameter learning, unlike adapter-only training.
     Sources:
       https://arxiv.org/abs/2403.03507
       https://arxiv.org/abs/2407.08296
  4. Activation memory: activation checkpointing is not the final answer; it is
     a compute-for-memory trade. Prefer selective activation recomputation and
     long-context-aware checkpointing over blanket checkpointing.
     Sources:
       https://arxiv.org/abs/2205.05198
       https://arxiv.org/abs/2604.27089
  5. Checkpoint I/O: runtime checkpoint files are a separate bottleneck from
     activation checkpointing. DataStates-LLM/UCP-style systems matter when
     frequent large checkpoints start pausing DGX jobs.
     Sources:
       https://arxiv.org/abs/2406.10707
       https://arxiv.org/abs/2406.18820

- Immediate local path:
  1. Use variable length before exotic architecture. Dataset Decomposition
     (NeurIPS 2024) shows that fixed concat-and-chunk training pays fixed
     attention cost even when examples are short, while variable sequence length
     makes compute follow actual document length and can train an 8k-context 1B
     run at about the cost of a 2k fixed baseline.
     Source: https://arxiv.org/abs/2405.13226
  2. Use PyTorch selective activation checkpointing before custom activation
     compression. PyTorch's current AC/SAC guidance says activation memory grows
     with depth, batch, and sequence length; normal AC trades memory for
     recompute, while SAC lets us avoid recomputing expensive matmuls/attention.
     Source: https://pytorch.org/blog/activation-checkpointing-techniques/
  3. Keep Liger/CCE enabled for large vocab. Apple CCE and Liger attack the
     supervised-token-by-vocab loss table, which is unusually large in our
     rounded 65k vocab PrefixLM path.
     Sources:
       https://machinelearning.apple.com/research/cut-your-losses
       https://arxiv.org/abs/2410.10989

- Near-term if 1B still OOMs:
  1. Add gradient accumulation plus small microbatch. This solves batch memory
     but not wall-clock speed. Treat it as fitting insurance, not a speed trick.
  2. Add block/stack activation checkpointing around QTRMBlockStack and the TRM
     recurrent thought cycles. Use selective checkpointing if available so
     pointwise/norm/dropout are recomputed while matmuls/flash attention are
     saved.
  3. Add length-bucketed sampling or variable sequence curriculum so local 4090
     sees short rows early and long rows only when needed.

- Implemented first activation checkpointing step:
  - `scripts/534_train_native_prefixlm_dataio.py` has
    `--activation-checkpointing`, passed into the native one-body model.
  - `scripts/335_train_qtrm_native_etd_probe.py` checkpoints
    encode/think/decode stage calls when training, gradients are enabled, and
    the stage input requires grad.
  - CUDA smoke verified the combined fitting stack:
    `--activation-checkpointing --loss-kernel auto --optimizer paged_ademamix8bit
    --trim-batch-to-max-length`.
  - Important smoke note: tiny `official_gated_delta2` with head_dim < 16 can
    fail Triton dot shape constraints. Use realistic head dims for GDN2 smoke.

- Multi-GPU/DGX sequence path:
  1. DeepSpeed Ulysses and Ring Attention distribute the sequence dimension
     across devices for long context. Good for DGX long-context runs, not a
     single-4090 fix.
     Sources:
       https://arxiv.org/abs/2309.14509
       https://arxiv.org/abs/2310.01889
  2. AutoSP (2026) is the newest practical direction found for DGX-style
     long-context training: compiler-chosen sequence parallelism plus
     long-context-aware activation checkpointing. Treat it as a DGX integration
     target, not a local 4090 prerequisite.
     Source: https://arxiv.org/abs/2604.27089
  3. ALST (2025) shows the intended shape of the DGX path: combine
     attention-agnostic single-GPU and multi-GPU memory optimizations so HF
     models can train much longer sequences without rewriting the model first.
     Source: https://arxiv.org/abs/2506.13996

- Riskier research path:
  1. Adacc (2025) combines adaptive activation compression and recomputation at
     tensor level. It is conceptually right, but too complex for the first local
     implementation.
     Source: https://arxiv.org/abs/2508.00806
  2. PRAC (2026) compresses activations through principal-random subspaces and
     reports total memory reduction with little degradation. Promising, but it
     is a new training algorithm and should be gated after ordinary
     checkpointing/packing.
     Source: https://arxiv.org/abs/2602.23111
  3. Activation Compression in LLMs (2026) gives the cleaner theoretical
     caution: unbiased compression is safer around linear operators and riskier
     around nonlinear operators. If we add compression, start with linear-layer
     activations/gradients, not blanket tensor compression.
     Source: https://arxiv.org/abs/2605.01255
  4. ZeRO-Offload and newer offload systems move optimizer/activation state to
     CPU/NVMe. They can fit larger models, but single-4090 throughput may fall
     sharply if PCIe/NVMe becomes the bottleneck.
     Sources:
       https://arxiv.org/abs/2101.06840
       https://arxiv.org/abs/2509.02480

- Reject-before-implementation rules:
  1. Do not solve batch OOM by lowering model size first. Try microbatch +
     gradient accumulation + 8-bit/GaLore optimizer first.
  2. Do not solve sequence OOM by adding memory architecture first. Try
     Flash/SDPA, length trimming, length buckets, and variable-length curriculum
     first.
  3. Do not solve activation OOM by blanket checkpointing first. Try
     block-level/selective checkpointing, then compression only if the simple
     stack still fails.
  4. Do not use inference-only memory papers as pretraining evidence. KV-cache,
     MLA serving, flash-storage paging, and speculative decoding are later
     deployment levers unless the paper explicitly covers training.

Practical decision:

```text
4090 first:
  length trim/buckets + Flash/SDPA + Liger/CCE + 8-bit/GaLore optimizer
  + block/selective activation checkpointing

DGX first:
  Ulysses/Ring/ALST/AutoSP-style sequence parallelism, plus async checkpoint I/O
  only when checkpoint writes start stalling training

Research later:
  PRAC / activation-gradient co-compression / Adacc / multi-tier offload
```

Learning-speed policy:

- 문과적으로 구분:
  - 학습 효율: 같은 토큰을 먹고 더 빨리 똑똑해지는 것.
  - 처리 속도: 초당 더 많은 토큰을 먹는 것.
  - 메모리 효율: 더 큰 몸/긴 문맥/큰 batch가 책상 위에 올라가는 것.
  이 셋을 섞으면 "빨라졌다"는 말이 빈말이 된다. 모든 run report는 가능하면
  `tokens/sec`, `target_tokens/sec`, `eval_loss per target token`, and wall-clock
  time to gate를 같이 기록한다.
- Immediate speed levers:
  1. Liger/CCE, Flash/SDPA, length trimming/bucketing: 처리 속도와 메모리 효율을
     동시에 올리는 저위험 기법이다. 먼저 켠다.
  2. GaLore/8-bit optimizer: optimizer 장부를 줄여 더 큰 모델이나 batch를 가능하게
     한다. 이것은 직접적인 초당 속도 기법이라기보다 scaling/fitting 기법이다.
  3. Gradient accumulation: 큰 effective batch를 흉내 내지만 wall-clock은 보통
     느려진다. OOM 보험이지 속도 기술이 아니다.
- Optimizer candidates for real learning efficiency:
  1. Muon: 2025 pretraining 연구에서 AdamW 대비 compute-time tradeoff를 개선한다고
     보고되었다. 우리에게는 "같은 토큰으로 더 빨리 loss가 내려가는가"를 볼
     1순위 learning-efficiency 후보지만, hyperparameter transfer를 별도 gate로
     검증해야 한다.
     Source: https://arxiv.org/abs/2505.02222
  2. Adam-mini / Q-Adam-mini / COSMOS: optimizer-state 메모리를 줄이면서 AdamW급
     성능을 노리는 후보. 4090에서는 의미가 있지만, 현재 이미 GaLore 8-bit가
     들어간 상태이므로 다음 후보군으로 둔다.
     Sources:
       https://arxiv.org/abs/2406.16793
       https://openreview.net/forum?id=sa3uVJLEsR
       https://www.microsoft.com/en-us/research/publication/cosmos-a-hybrid-adaptive-optimizer-for-memory-efficient-training-of-llms/
  3. FlashOptim / APOLLO-style optimizer compression: 2025-2026 메모리 효율
     후보지만, 현재 로컬 코드에 없는 새 optimizer path이므로 Stage86 fitting
     이후 작은 A/B gate로만 검증한다.
     Sources:
       https://arxiv.org/abs/2602.23349
       https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf
- Gate:
  - A speed trick is accepted only if it improves at least one of:
    `tokens/sec`, `target_tokens/sec`, or wall-clock time to the same eval-loss
    gate without hurting heldout generation.
  - An optimizer trick is accepted only if it improves loss-vs-target-token or
    time-to-gate, not merely if it fits in memory.
- Early-eval interpretation rule:
  - Very early generation gates are health checks only. They can reject obvious
    collapse such as immediate EOA, repeated-token loops, NaNs, or no loss
    movement. They must not reject capability or architecture before the run has
    eaten a comparable amount of supervised answer tokens.
  - Current reference scale: Stage85 82M reached `93.4%` first-response accuracy
    and `45/64` greedy exact at step 3000 after about `1.80M`
    `target_tokens_seen`. Stage86 913M step 5000 saw only `47.3k`
    target tokens, about `1/38` of that exposure. Therefore Stage86 step 5000
    is a format/collapse check, not an ability judgment.
  - 문과적으로 말하면: 82M은 답안지를 180만 글자쯤 써본 뒤 채점한 것이고,
    913M은 4.7만 글자만 써본 뒤 본 것이다. 후자는 "손이 움직이나"를 보는
    순간이지, "이 학생이 못한다"를 판정할 순간이 아니다.
- Implemented measurement:
  - `scripts/534_train_native_prefixlm_dataio.py` now logs cumulative and
    interval speed fields in every train row:
    `elapsed_sec`, `interval_sec`, `steps_per_sec`, `tokens_per_sec`,
    `target_tokens_per_sec`, `compute_tokens_per_sec`, and interval variants.
  - `compute_tokens_seen` is saved in checkpoints and reports. Plain-language
    read: `tokens_seen` is useful text, `target_tokens_seen` is graded answer
    text, and `compute_tokens_seen` is how much paper the GPU actually had to
    read after trimming/bucketing.
  - Current Stage86 was launched before this metric patch, so it remains a
    loss/fit smoke. Relaunch or resume after the patch when comparing speed
    tricks such as Muon vs GaLore.
- Implemented eval non-finite guard:
  - `evaluate_prefixlm_loss` now checks each eval batch loss for finite values.
    If a batch returns NaN/Inf and the model supports hidden-loss evaluation, it
    retries that same batch with fp32 `torch` chunked CE. The eval row records
    `eval_nonfinite_batches` and `eval_fallback_batches`.
  - This does not change training gradients. It only prevents a transient
    grading-kernel glitch from being mistaken for an architecture reject.
  - 문과적으로 말하면: 학생이 답을 쓰는 손은 그대로 두고, 채점기가 한 줄을
    잘못 읽었을 때 같은 답안을 더 느리지만 확실한 채점기로 다시 읽게 만든
    것이다.

4090 capacity planner:

- `scripts/540_plan_prefixlm_4090_capacity.py` builds candidate PrefixLM models
  on PyTorch `meta` device, so it can count parameters without allocating real
  weights.
- It reports a floor memory estimate before activations:
  model weights + gradients + optimizer ledger. This is not a complete VRAM
  guarantee; activations are still controlled by batch size, sequence length,
  checkpointing, length bucketing, and loss kernel.
- Latest local plan with `--optimizer galore_adamw8bit --batch-size 1
  --seq-len 128`:

```text
current_82m  0.082B  floor=0.72GB   green_for_4090_smoke
probe_225m   0.225B  floor=1.99GB   green_for_4090_smoke
probe_357m   0.357B  floor=3.16GB   green_for_4090_smoke
safe_695m    0.695B  floor=6.15GB   green_for_4090_smoke
risk_913m    0.913B  floor=8.08GB   green_for_4090_smoke
risk_1150m   1.150B  floor=10.17GB  yellow_requires_small_microbatch
```

- Plain-language read: 1.15B is no longer absurd for a 4090 smoke, but it is
  not a casual run. Start with 695M or 913M to validate throughput and memory,
  then try 1.15B with batch size 1, bf16 autocast, Liger CE, length bucketing,
  trimming, and only checkpointing modes that have passed a same-backbone
  backward smoke.
- Stage86 local 913M launcher:
  - `scripts/launch_stage86_local_913m_optimized_smoke.sh`
  - launches the 0.913B candidate with the full 4090 fitting stack:
    length buckets, trim, Liger CCE, bf16 autocast, and `galore_adamw8bit` by
    default.
  - Activation checkpointing is now opt-in through
    `ACTIVATION_CHECKPOINTING=1`. The first 913M debug run showed that blanket
    stage-level checkpointing can fail with a PyTorch recomputation metadata
    mismatch, while the same 913M stack without activation checkpointing
    completed a one-step backward smoke. This is a checkpoint placement issue,
    not evidence that 913M does not fit on 4090.
  - The launcher refuses to start if another
    `scripts/534_train_native_prefixlm_dataio.py` process is already running,
    unless `FORCE=1` is set. This protects clean speed/VRAM measurements.
  - Current launch status: Stage84 local was stopped after a strong 60k
    checkpoint; Stage86 no-AC one-step debug passed. The full launcher should
    now run with default `ACTIVATION_CHECKPOINTING=0`.
  - Wrapper reliability note: plain `nohup` inside the launcher exited with an
    empty log in this environment, while the same command survived when run
    directly. The launcher now uses `setsid` plus a command array and redirects
    stdin from `/dev/null`. A `STEPS=1` launcher smoke completed with
    `total_parameters=912896129`, `activation_checkpointing=false`, initial
    `loss=11.1457`, and `eval_loss=11.1269`.
  - Full local Stage86 smoke is now running as PID `3018854` with
    `steps=5000`. First logged row: `step=1`, `loss=11.1455`,
    `eval_loss=11.1269`, `activation_checkpointing=false`. Observed local
    VRAM after startup was about `13.1GB / 24.6GB`.
  - Stage86 step-2000 status: `eval_loss` moved from `4.1802` at step 1000 to
    `3.8603` at step 2000. This is a valid fitting/learning smoke, but not yet
    a training-efficiency claim because `batch_size=1` only exposed about
    `19.6k` target tokens by step 2000.
  - Stage86 step-5000 result: fitting smoke completed. Eval loss moved
    `11.1269 -> 4.1802 -> 3.8603 -> 3.6802 -> 3.5635 -> 3.2167`, with
    `target_tokens_seen=47315`.
  - Stage86 generation gate at step 5000:
    first-response accuracy `25.0%`, first-token `<|box_end|>` top-1 `0%`,
    free-generation exact `0/32`, starts-with-EOA `0/32`, ended-with-EOA
    `32/32`, repeated loops `0/32`. Plain-language read: the model learned to
    write and close answers instead of collapsing, but it has not eaten enough
    answer tokens to be correct. This is a batch/throughput problem before it is
    an architecture problem.
  - The launcher now exposes `BATCH_SIZE`, `SEQ_LEN`, `CHECKPOINT_EVERY`,
    `EVAL_EVERY`, `EVAL_BATCH_SIZE`, and `LOG_EVERY` environment knobs. Next
    local scaling gate should try metric-enabled `BATCH_SIZE=2`, then 4 only if
    VRAM stays below the safe margin.
- Stage87 local 913M `BATCH_SIZE=2` metric smoke:
  - command family:
    `RUN_NAME=STAGE87_LOCAL913M_BS2_METRIC_SMOKE ... BATCH_SIZE=2 STEPS=1000
    bash scripts/launch_stage86_local_913m_optimized_smoke.sh`
  - status:
    completed normally with checkpoint and report under
    `/tmp/qtrm_eval/20260523_STAGE87_LOCAL913M_BS2_METRIC_SMOKE`.
  - memory:
    local 4090 peaked around `13.2GB / 24.6GB`, so physical batch 2 fits for
    the 913M candidate with no activation checkpointing.
  - speed:
    final cumulative `target_tokens_per_sec=88.6`,
    `compute_tokens_per_sec=602.9`; healthy interval rows usually showed about
    `900-1100` compute tokens/sec, with periodic checkpoint/eval stalls.
  - learning:
    train loss moved `11.0677 -> 2.9869`; eval moved
    `11.1269 -> 3.7644` by step 1000, after
    `target_tokens_seen=19,716`.
  - caution:
    step 500 reported a transient `eval_loss=NaN`, while training loss stayed
    finite and the step 1000 eval returned finite. Treat this as a numeric/eval
    stability warning before trying batch 4, not as model collapse.
  - plain-language read:
    batch 2 means the student can keep two notebooks open on the 4090 desk.
    The desk still has room, but one grading pass briefly produced nonsense, so
    do not open four notebooks until the grading path is trusted.
- Stage88 local 913M `BATCH_SIZE=2` target-token ramp:
  - command family:
    `RUN_NAME=STAGE88_LOCAL913M_BS2_TARGET600K_RAMP ... STEPS=30000
    BATCH_SIZE=2 bash scripts/launch_stage86_local_913m_optimized_smoke.sh`
  - status:
    launched locally as the first 913M run intended to reach an ability-relevant
    token exposure rather than an early collapse check.
  - target:
    about `600k` supervised answer tokens if the Stage87 token rate holds.
    This is still below Stage85 step-3000's `1.8M` target-token exposure, but
    high enough to judge whether the 913M curve is bending toward the accepted
    82M contract.
  - first logged gate:
    step 1 eval loss was finite and the new guard recorded
    `eval_nonfinite_batches=0`, `eval_fallback_batches=0`.
  - do-not-reject-before:
    do not judge 913M generation quality from the first few tens of thousands
    of target tokens. Before `~600k` target tokens, use eval only to detect
    collapse, NaNs, repeated loops, or total loss stagnation.
- MLA, QuantSpec, KV-cache quantization, and Apple `LLM in a Flash` are mostly
  inference-memory or long-context-serving levers. They matter after the model
  can speak and reason, or when training a new MLA-shaped model from scratch.
  They should not replace optimizer-state compression for the immediate 4090
  from-scratch scaling bottleneck.
- Apple-related source classification:
  - `LLM in a Flash` stores parameters in flash and streams needed weights into
    DRAM for inference; it enables models up to about twice available DRAM and
    speeds naive loading, but it is not a full-pretraining optimizer.
    Source: https://machinelearning.apple.com/research/efficient-large-language
  - `Memory-Efficient Backpropagation` targets mobile fine-tuning under 1GB
    memory, not our full from-scratch GPU pretraining path.
    Source: https://machinelearning.apple.com/research/memory-efficient-backpropagation
  - `QuantSpec` targets long-context inference with hierarchical 4-bit KV cache
    and 4-bit weights; useful later for serving, not first for training.
    Source: https://machinelearning.apple.com/research/quantspec
  - `Pretraining with Hierarchical Memories` is architecture-relevant for
    separating long-tail knowledge from the small model, but it is a pretraining
    design choice and must be tested as a separate one-body architecture.
    Source: https://machinelearning.apple.com/research/hierarchical-memories

## Stage92 Data Sufficiency Audit

- Current DGX Stage92 is not data-rich pretraining yet. The sampled PrefixLM
  shard actually fed to the run is:
  - path:
    `/mnt/data4tb/wgram-lm/local_eval/stage67_local_sampled_prefixlm/sampled`
  - disk size: about `15MB`
  - `tokens.npy`: `3,495,331` tokens
  - per epoch: `22,461` instruction/response rows, about `1,711,795`
    instruction tokens and `1,729,075` response target tokens.
- Plain-language read:
  the library is not empty, but the student is currently studying from a thin
  handout. This is enough to prove that the training loop, answer-token path,
  checkpointing, and early loss descent are alive. It is not enough to claim
  serious 913M/1B general-language growth.
- Existing local data is much larger:
  `/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515` is about `40GB`,
  split into about `2.1GB` `data/` JSONL and about `37GB`
  `data_clustered/` parquet. Large sources include `principia_collection`,
  `numinamath`, `natural_reasoning`, `webinstruct_verified`, and clustered
  `SYNTH` parquet shards.
- DGX also has the cleaned HRM-Text source data, and it is larger:
  `/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515` is about `320GB`,
  with about `2.1GB` `data/` JSONL and about `309GB` `data_clustered/`
  parquet. It contains `19` JSONL files and `5,198` parquet files, including
  `flan`, `openmathinstruct2`, `openthoughts2`, `acereason`,
  `textbookreasoning`, `SYNTH`, and related clusters.
- Full tokenized/sample outputs for that 40GB cleaned dataset are not currently
  present under the local cleaned-data directory or the official `data_io`
  directory, and DGX source data currently has `0` `tokens.npy` files under the
  cleaned dataset path. The missing pipeline step is tokenization plus large
  stratified sampling.
- DGX caveat:
  `scripts/533_prepare_hrm_text_dataio_sample.sh` still defaults to the local
  path `/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515`. On DGX, set
  `CLEANED_DATA_PATH=/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515`
  explicitly before building Stage93 data.
- Decision:
  do not download more data before exhausting the existing HRM-Text cleaned
  dataset. The next real scaling stage should build a Stage93 large sampled
  dataset from both `data/` and `data_clustered/`, then train on that.
- Download more only after Stage93 if the target explicitly expands to broad
  natural language, code/tool use, agentic benchmarks, or multimodal ability.
  For the immediate HRM-Text-like one-body language/thought path, the missing
  step is bigger sampling from existing data, not new collection.

## Stage93 Continue-Training Contract

- Stage93 is continued learning, not a reset. The larger dataset changes the
  textbook, not the student.
- Default resume checkpoint:
  `/mnt/data4tb/wgram-lm/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K/last_model.pt`
- Why model-only by default:
  Stage92 trained on a tiny `15MB / 3.5M token` shard. Stage93 moves to a much
  larger HRM-Text Data-IO shard. Keeping the model weights preserves learned
  reading/speaking behavior, while restarting optimizer moments avoids letting
  the tiny-shard optimizer statistics over-steer the new broad-data phase.
- Full optimizer continuation remains available with `last.pt`, but it is a
  more literal continuation and less clean when the data distribution changes.
- Launcher:
  `scripts/536_launch_stage93_dgx_continue_prefixlm.sh`
- Guard:
  the launcher refuses to run until the Stage93 sampled dataset has
  `tokens.npy`, and it refuses to start while another PrefixLM training process
  is active unless `FORCE=1` is set.
- Plain-language read:
  Stage92 is the student who learned from the small handout. Stage93 hands the
  same student a larger textbook. We keep the student's brain, but reset the
  study rhythm for the new textbook.

## Native Text Versus Native Multimodal

- Final project direction:
  native multimodal QTRM-MemoryOS. The target system should read text, images,
  OCR/layout, retrieved text evidence, retrieved visual evidence, and tool
  outputs through one learned answer path.
- Current Stage92/Stage93 direction:
  native text PrefixLM. The active trainer is
  `scripts/534_train_native_prefixlm_dataio.py`, whose dataset path maps
  HRM-Text Data-IO rows into `input_ids`, `labels`, `attention_mask`, and
  `response_start_mask`. No image tensors, vision encoder outputs, OCR boxes,
  or visual memory rows are in the active Stage92/Stage93 training path.
- Why it is still "native":
  the model is not a small side adapter on top of Qwen for this run. The
  913M one-body model learns token embeddings, recurrent thought blocks, and
  LM logits as its own body.
- Why it is not yet "native multimodal":
  multimodal ability requires a trained visual reader/projector/resampler and
  multimodal data/eval, such as image QA, OCR, chart QA, visual evidence
  retrieval, and visual grounding. Those are scaffolded in project docs and
  dataset scripts, but they are not part of the active Stage92/Stage93 PrefixLM
  run.
- Operating rule:
  do not claim multimodal ability from Stage92/Stage93. Claim only
  text-language, text-reasoning, and one-body learning-efficiency evidence
  until the multimodal reader path is actually trained and evaluated.
- Plain-language read:
  the current student is being taught to read and think in text first. The
  final organism should also see, inspect screenshots, read charts, and use
  visual memory. But right now we are growing the language/thought spine before
  attaching the eyes.

## Stage94 Multimodal Graft Rule

- Do not restart from scratch for the first multimodal continuation.
- Default route:
  use the strongest Stage93 text-native checkpoint as the language/thought
  spine, then add a visual reader/projector/resampler and train visual data
  through the same recurrent core and LM head.
- Freeze first:
  token embeddings, most recurrent thought blocks, and the LM head. Train the
  visual reader adapter, multimodal projector, workspace resampler, and
  OCR/layout/chart source embeddings.
- Partially unfreeze only after the visual path is stable:
  late recurrent thought blocks and answer-facing projection layers.
- Why:
  from scratch multimodal pretraining would throw away the expensive text
  language/thought spine. The high-probability route is to preserve the student
  and teach a new sense organ to speak the student's latent language.
- Native multimodal evidence requires:
  1. text regression does not erase Stage93 language ability;
  2. visual perturbations change answers when they should;
  3. disabling the visual projector drops visual-task performance;
  4. answers come through the same recurrent core and LM head;
  5. OCR/chart/evidence answers can be grounded to visual source regions or
     source tokens.
- Reject:
  a side OCR/vision solver that produces the answer externally and lets the
  model merely copy text. That is a useful tool pipeline, but it is not native
  multimodal reasoning.
- Plain-language read:
  Stage93 grows the reader-thinker-speaker. Stage94 attaches eyes to that same
  person. We do not raise a second person from zero unless the graft route fails
  under clear ablation evidence.

## Stage93B General-Language Focus

- Failure named plainly:
  Stage93A learned the tiny math/reasoning handout, not ordinary speaking. A
  falling micro PrefixLM loss is therefore not evidence that the model can
  answer simple natural-language prompts.
- HRM-Text contract to preserve:
  tokens enter the same one-body recurrent model, response tokens are trained
  with PrefixLM target-only CE, and free generation must come from the same LM
  head. Do not add a side answerer or a separate evaluator-only probe to hide
  weak generation.
- Current evidence:
  the Stage93A step-40000 language gates failed general speaking. The
  `general_language_generation_probe` had `0/16` hits and produced math-like
  fragments for ordinary prompts such as sky color, water, fruits, thanks, and
  simple word knowledge. The `general_language_heldout_loss` was still high
  enough to block any claim of usable language ability.
- Causal read:
  the model is not silent; it is speaking in the dialect of the handout it saw.
  In human terms, it was drilled on problem-book continuations, so it tries to
  turn every question into algebraic filler.
- Mandatory next action:
  before another architecture idea, run a broad HRM-Text-style curriculum
  continuation that includes ordinary instruction/dialogue/QA/translation rows
  through the normal PrefixLM path. The automation entrypoint is
  `scripts/552_run_stage93b_general_language_pipeline_dgx.sh`.
- Scheduling rule:
  Stage93B must not be blocked by old `reasoning_nonflan` sample writers. It
  waits for active PrefixLM training to avoid GPU contention, but by default it
  does not wait for obsolete reasoning-only samplers
  (`WAIT_FOR_OLD_REASONING_SAMPLERS=0`). Set that variable to `1` only when
  protecting disk I/O matters more than starting the general-language repair.
- Duplicate-prep guard:
  the pipeline must keep exactly one general-language data-prep path active.
  It uses exact process matching for the pipeline, tokenizer, and
  `sample_tokenized.py`, plus a short relaunch grace period
  (`DATA_PREP_RELAUNCH_GRACE_SECONDS=300`) so the watcher does not mistake the
  launch-to-tokenizer handoff gap for a failed prep process. If a duplicate
  tokenizer ever starts on the same output directory, move the partial
  `tokenized`/`sampled` directory aside and restart cleanly.
- Stage93B gate:
  pass/fail is based on TensorBoard-backed language gates, not train loss:
  `eval/general_language_heldout/loss`,
  `eval/general_language_heldout/token_accuracy`,
  `eval/general_language_generation/accuracy`,
  `eval/general_language_generation/degenerate_repetition_rate`, and free
  generation samples.
- Plain-language read:
  first teach the same student to speak ordinary sentences from ordinary books.
  Only after that is true should we ask whether the recurrent thinker, GRAM,
  PTRM, or memory machinery improves reasoning beyond HRM-Text.

## Stage95 BLT From-Scratch Foundation Curriculum

- Failure to avoid:
  do not make another model that can complete a math worksheet but cannot speak
  ordinary language. The Stage93A lesson applies even more strongly to BLT/1B:
  low loss on a reasoning-only or math-heavy sample can create a narrow problem-
  book dialect, not a general LLM.
- Plain-language rule:
  first raise a person who can read and speak, then teach calculation,
  reasoning, memory, and later tool work. Do not start with a calculator and
  hope it grows into a person.
- Stage95 data mix must include all first-order raw-intelligence shelves:

  ```text
  general language:
    dialogue, ordinary instruction, QA, summarization, world text, narrative

  Korean/English and multilingual:
    Korean direct QA/instruction where available, English-Korean translation,
    WMT/translation, XQuAD/MLQA/TyDi/XNLI-style multilingual reading

  reasoning:
    natural_reasoning, webinstruct_verified, acereason, textbookreasoning,
    openbook/commonsense/science QA

  math/symbolic:
    gsm8k, math, numinamath, openmathinstruct2, omnimath, SYNTH math traces

  memory/context:
    long QA, multi-document/case-file style rows, summarization and retrieval-
    like evidence rows when available
  ```

- Agentic/tool data is intentionally later:
  tool-use traces, function-call formatting, browser actions, and long-horizon
  coding-agent trajectories belong after the model can speak and reason through
  the normal LM path. Otherwise the model may learn the shape of a function
  call without the judgment needed to choose and use the tool.
- Stage95 implementation note:
  `scripts/555_prepare_byte_prefixlm_sample.py` now accepts both `.jsonl` and
  `.parquet` sources and supports `--source-globs`. This is mandatory for a real
  BLT/byte-latent foundation sample, because the DGX HRM-Text cleaned dataset
  stores most broad shelves under `data_clustered/**/*.parquet`.
  The builder also supports `--max-scan-rows-per-file`, so a broad run can
  sample every shelf without getting stuck searching one oversized parquet file
  for acceptable rows. In plain language: open every subject book, but do not
  let one giant shelf stop the student from seeing the rest of the curriculum.
- Stage95 preparation entrypoint:
  `scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh` documents and
  builds the broad byte sample. Its plan must list general language, reasoning,
  math, multilingual, and memory/context shelves. On DGX it can automatically
  use `/mnt/data4tb/venv_sglang_pr23000/bin/python` when the project `.venv`
  lacks `pyarrow`, because parquet support is required for the broad shelves.
- Stage95 overnight automation:
  use `scripts/559_run_stage95_blt_partial_then_full_dgx.sh launch` for the
  partial-then-full route. The script resumes or starts the broad byte sample
  build, builds a smaller byte partial from the same curriculum, trains the 1B
  BLT body on the partial with model-only checkpoints, waits for the broad
  sample to finish, then continues on the full sample with
  `--resume PARTIAL_OUT/last_model.pt`. In plain language: the model starts
  reading the thin temporary book immediately, and when the full book is bound
  it keeps learning with the same body instead of restarting.
- Minimum Stage95 source rule:
  do not launch a long BLT/1B from-scratch run from only
  `data/gsm8k_train.jsonl`, `data/natural_reasoning.jsonl`, or another narrow
  math/reasoning subset. A long run must show its selected `source_files` /
  `source_globs` in metadata and include general-language, reasoning, math,
  multilingual, and memory/context shelves.
- Minimum Stage95 gates:
  TensorBoard-backed loss is not enough. Run ordinary language heldout,
  free-generation, Korean/English multilingual, arithmetic/reasoning,
  symbolic-manipulation, and context-memory probes through the same BLT byte
  speaker. Do not promote a checkpoint that passes math while failing simple
  prompts such as water, sky color, greeting, translation, or short QA.
- BPE relation:
  keep the strongest BPE Stage92/93 checkpoint as a baseline and fallback. BLT
  Stage95 may become the new from-scratch path only if it preserves general
  language while gaining byte/tokenizer robustness or compute/context leverage.
  Do not erase the BPE baseline just because BLT is architecturally cleaner.

## Qwen-Like Versus HRM-Text-Like Language Learning

- Similarity:
  both Qwen-style LLMs and the current Stage93B route learn by reading token
  sequences and producing answer tokens through a normal LM head. Broad
  natural-language, instruction, dialogue, QA, and translation data are the
  right food if the target is ordinary language generation.
- Difference:
  Qwen3.5 is born from broad general pretraining over very large web/code/
  multilingual corpora. Stage93B is not that yet. It is a 913M born-one-body
  recurrent PrefixLM model that first overlearned a small reasoning/math
  handout and is now being repaired with broader HRM-Text/Data-IO curriculum.
- Operating rule:
  do not call Stage93B "Qwen-like" as a capability claim. It is Qwen-like only
  in the broad language-model training direction. Architecturally and
  experimentally it should be judged as HRM-Text-like: same recurrent body,
  same PrefixLM answer path, same LM mouth, and free-generation gates.
- Plain-language read:
  Qwen is like someone raised from childhood in a huge multilingual library.
  Stage93B is a student who learned a narrow problem-book dialect first and is
  now being moved into ordinary books. The repair can move in the Qwen
  direction, but it has not earned Qwen-scale language breadth until the data
  and generation gates prove it.

## Multilingual Status

- Architecture/tokenizer status:
  multilingual-capable, not yet multilingual-proven. The active PrefixLM
  tokenizer is a `65,536` vocab BPE/byte-level tokenizer, so it can encode
  non-English text rather than hard-failing on Korean, Spanish, Turkish,
  German, Telugu, and related scripts.
- Data source status:
  DGX HRM-Text cleaned data contains FLAN translation/multilingual files. A
  quick source scan found `50` FLAN files matching translation/WMT/XQuAD/
  WikiLingua-style names, including `wmt16_translate_de-en`,
  `wmt16_translate_tr-en`, `xquad_es`, `xquad-ca`, and English-Telugu
  aligned translation tasks.
- Current active Stage93 data status:
  `PROFILE=reasoning_nonflan`, with `INCLUDE_FLAN=0`. This means the current
  Stage93 data build is strong for reasoning/instruction text, but it is not
  the multilingual curriculum yet.
- Operating rule:
  do not claim multilingual ability from Stage92 or the current non-FLAN
  Stage93 run. Claim only multilingual readiness until a multilingual
  curriculum run and Korean/translation eval gates are completed.
- Data profile added:
  `PROFILE=multilingual_curriculum` in
  `scripts/535_prepare_stage93_hrm_text_large_dataio.sh`. It includes the
  normal reasoning shelves plus targeted FLAN translation/multilingual files
  using `FLAN_INCLUDE_REGEX`.
- Updated DGX source check:
  the broader multilingual regex
  `translate|translation|wmt|xquad|wiki_lingua|cc_alligned|xnli|tydi|mlqa|paws-x|xstory`
  matches `832` FLAN parquet files on DGX. This is the intended mix:
  HRM-Text reasoning shelves plus targeted multilingual FLAN, not the full
  264GB FLAN bulk.
- Current Stage93B general-language curriculum snapshot:
  `FLAN_MAX_FILES=192` is active for the broad repair run. A source scan of the
  selected FLAN subset found `33` explicitly multilingual/translation files,
  mainly `wmt14_translate_fr-en`, `wmt16_translate_cs-en`,
  `wmt16_translate_de-en`, `wmt16_translate_fi-en`,
  `wmt16_translate_ro-en`, `wmt16_translate_ru-en`,
  `wmt16_translate_tr-en`, and WikiLingua English. This makes Stage93B
  multilingual-translation-aware, but still not a full Asian-language or
  global multilingual curriculum.
- Mixing rule:
  yes, HRM-Text and multilingual data should be trained together for the
  multilingual stage. HRM-Text keeps the reasoning spine; multilingual FLAN
  teaches the same spine to operate across languages. Do not train a separate
  multilingual side model.
- Minimum multilingual gate:
  Korean direct QA, Korean instruction following, English-Korean translation,
  WMT-style English/French/German/Turkish/Russian/Romanian/Finnish/Czech word
  translation, non-English XQuAD-style reading, and code-switched prompt
  stability must all be evaluated through the normal LM answer path.
- Probe assets:
  `data/eval/prefixlm_multilingual_probe.jsonl` is the small smoke exam, and
  `scripts/542_eval_prefixlm_multilingual_probe.py` runs it by wrapping each
  instruction with the same HRM-Text Data-IO PrefixLM markers and greedily
  generating through the checkpoint's normal recurrent LM head.
- Evaluation rule:
  run this probe after a `PROFILE=multilingual_curriculum` checkpoint. Passing
  the probe does not prove broad multilingual skill, but failing it blocks any
  claim that the HRM-Text plus multilingual mix has actually transferred into
  the one-body answer path.
- Plain-language read:
  the model has a mouth that can pronounce many languages and the library has
  multilingual books. Stage93B starts putting some translation books on the
  desk, especially European-language translation. But Korean/Japanese/Chinese/
  Arabic-style broad multilingual ability is not earned by that. To call the
  student multilingual, each language family must appear in the schedule and
  pass free-generation gates through the same LM mouth.

## Multilingual And Native Multimodal Paper Watch

As of 2026-05-23, multilingual and native multimodal design should not be
treated as solved by the current HRM-Text Data-IO path. The current path is a
valid text/thought spine, but tokenizer and multimodal choices need explicit
paper-backed gates before broad claims.

Latest-first mechanism table:

```text
date        | source                              | mechanism
2026-05-08  | Fast Byte Latent Transformer         | tokenizer-free BLT generation accelerated by block diffusion, self-speculation, and verification
2026-04-22  | LLaDA2.0-Uni                         | native unified text/vision via semantic discrete visual tokenizer, dLLM backbone, diffusion decoder
2026 ICML   | Next Implicit Token Prediction       | NTP plus auxiliary prediction of next-token shallow representation; train-time-only representation shaping
2026-03-23  | Teaching Old Tokenizers New Words v2 | continued BPE training plus pruning for controlled tokenizer adaptation
2026-03     | ByteFlow                            | tokenizer-free learned/adaptive byte compression by latent coding-rate chunking
2025-11-27  | Qwen3-VL v2                          | interleaved text/image/video context, interleaved-MRoPE, DeepStack multi-level ViT features
2025-08-07  | H-Net++                              | hierarchical dynamic chunking for tokenizer-free morphologically rich language modeling
2025-07-17  | FLEXITOKENS                          | byte-level learnable boundary tokenizer to reduce multilingual/domain over-fragmentation
2025-07-10  | H-Net                                | end-to-end learned dynamic chunking that replaces tokenizer/LM/detokenizer with one hierarchy
2025-06-12  | One Tokenizer To Rule Them All        | train tokenizer over broader languages than pretraining mix to improve later language plasticity
2025-03-27  | UGen                                 | single AR transformer over text/image discrete tokens with progressive visual vocabulary activation
2025-03-26  | Qwen2.5-Omni                         | end-to-end multimodal Thinker-Talker, block-wise audio/visual encoders, TMRoPE
2025-02-19  | Qwen2.5-VL                           | dynamic-resolution ViT, native spatial/temporal perception, document/chart/layout grounding
2024-12-13  | Byte Latent Transformer              | tokenizer-free byte patches with entropy-based dynamic patching
```

Local implication:

- Stage93 multilingual:
  keep the current 65k byte-level BPE for the immediate run, but measure
  Korean/Spanish/German/translation fragmentation and generation before
  claiming multilingual ability. Do not redesign the tokenizer unless the
  multilingual probe or token-length audit shows over-fragmentation as the
  bottleneck.
- Tokenizer next candidate:
  if multilingual fails because Korean or other scripts explode into long
  inefficient token chains, first test a tokenizer audit and controlled
  continued-BPE/adaptive-tokenizer small run. Do not restart 913M pretraining
  just because a tokenizer paper is new.
- Latent-first interpretation:
  tokenizer-free/byte-latent, NITP, and PV-GRAM touch different layers of the
  same general move away from surface-only next-token fitting:

  ```text
  tokenizer-free / byte-latent = change what raw language unit enters the model
  NITP                         = change what representation target training sees
  PV-GRAM / GRAM / PTRM         = change how the model thinks over latent state
  ```

  Plain-language rule: byte-latent changes the reading material, NITP changes
  the exam, and GRAM changes the thinking habit. They are compatible, but do
  not turn all three knobs in the active Stage93 continuation. Stage93B keeps
  the current tokenizer and data path. NITP is a cheap auxiliary-loss candidate
  for a small controlled run. Byte-latent/tokenizer-free is a from-scratch
  ablation candidate only until it beats the BPE baseline on the same
  Korean/English language gates.
- Stage94 82M latent-first ablation:
  the immediate controlled order is:

  ```text
  A. BPE 82M baseline
  B. BPE 82M + NITP-style latent target loss
  C. byte-latent/tokenizer-free 82M
  D. byte-latent/tokenizer-free 82M + NITP
  ```

  `scripts/534_train_native_prefixlm_dataio.py` now supports
  `--nitp-loss-weight`, `--nitp-hidden-dim`, and `--nitp-max-targets`.
  `scripts/553_run_stage94_latent_first_82m_ablation.sh` runs A/B on the
  normal BPE PrefixLM path once a Data-IO `sampled` directory is ready.
  Tokenizer-free remains a separate C/D ablation; do not mix it into the
  Stage93 continuation checkpoint.
- Stage94 tokenizer-free / Fast-BLT-D decision log:
  after the user explicitly selected `2605.08044`, raw byte-free was treated
  only as a control, not as the final scalable architecture. The scalable path
  is BLT-style byte-latent patching: local bytes -> latent patch -> global
  recurrent core -> local byte decoder.

  ```text
  run        | contract                         | final eval clean loss | note
  Stage94A   | BPE 82M baseline                 | 4.3205                | normal BPE PrefixLM path, 2000-step baseline
  Stage94B   | BPE 82M + NITP                   | 4.3189                | tiny improvement, not meaningful enough to promote NITP
  Stage94C   | raw byte-free                    | 1.8895                | best small-run loss, but global core sees every byte
  Stage94D   | BLT-D-4, shifted latent only     | 2.9635                | 4x compression, weak boundary prediction
  Stage94E   | BLT-D-4, boundary current latent | 2.8323                | boundary fix helped
  Stage94F   | BLT-D-4, diffusion weight 0.05   | 2.8542                | lower diffusion did not help enough
  Stage94G   | BLT-4 clean-only                 | 2.8384                | diffusion is not the sole bottleneck
  Stage94H   | BLT-4 clean-only, local4         | 2.8744                | larger local decoder did not help
  Stage94I   | BLT-2 clean-only                 | 2.3202                | 2x compression is much stronger than fixed 4-byte folding
  Stage94J   | BLT-2, latent cross-attn decoder | 2.3464                | slightly worse than Stage94I; not first bottleneck
  Stage94K   | BLT dynamic proxy, soft3         | 2.7982                | 2.6x compression is too coarse for current small model
  Stage94L   | BLT dynamic proxy, soft2         | 2.6740                | same compression as BLT-2 but worse boundary heuristic
  Stage94M   | BLT-2 + local-decoder NITP 0.05  | 2.3453                | NITP learns cosine target but hurts clean byte loss
  Stage94N   | BLT ByteFlow-proxy boundary      | 2.6982                | shorter latent sequence, but cheap embedding-change boundary loses language signal
  Stage94O   | HBF-BLT v0 heuristic hierarchy   | 2.7490                | UTF-8-safe H-Net/ByteFlow-style boundary did not beat fixed BLT-2
  Stage94P   | learned-primary semantic BLT v0  | 2.8859                | trainable chunker works, but 4x compression loses too much signal
  Stage94Q   | learned-primary BLT-2-scale      | 2.3540                | same 2x compression as BLT-2, but learned soft chunk embedding still trails fixed BLT-2
  Stage94R   | learned-boundary H-Net v0         | 2.8124                | learned scorer changes latent length, but BLT local decoder is the wrong dechunk/speaker contract
  Stage94S   | H-Net dechunk/speaker v0          | 2.5309                | dechunk path helped, but boundary collapsed toward coarse max-patch behavior
  Stage94T   | H-Net dechunk + prior 0.5         | 2.4817                | best H-Net variant so far, still worse than fixed BLT-2
  Stage94U   | H-Net dechunk + prior 0.65        | 2.5008                | more selected boundary information alone did not help
  Stage94W   | BLT n-gram entropy patcher        | 2.5601                | newer BLT entropy-patcher proxy; beats Stage94N, still worse than Stage94T and Stage94I
  Stage94X   | raw byte-free seq768 capacity     | 2.1387@400            | 106k-row longer-context raw-byte path fits on 4090 and learns fast
  Stage94Y   | BLT-2 + raw-byte teacher distill  | 2.2999                | first compressed-latent path to beat Stage94I at the full 1200-step gate
  Stage94Y0  | BLT-2 teacher-off same seed       | 2.3345                | same seed 9682 control; confirms teacher distill is the causal improvement
  ```

  Plain-language read: raw byte-free is a student reading every letter with
  full attention; BLT is a student folding nearby letters into shorter notes
  before thinking. Folding is the scalable route, but fixed 4-byte notes are
  currently too coarse: Stage94I shows that 2-byte notes recover much of the
  lost language signal. Stage94H also says "make the local decoder bigger" is
  not the first fix. Stage94J then tested BLT-style latent cross-attention, but
  the 1200-step gate came back slightly worse than Stage94I. The highest-signal
  next move was therefore dynamic/entropy-style patching: keep raw bytes, keep
  a shorter latent sequence, but stop forcing every span into the same fixed
  patch size. The cheap UTF-8/byte-class heuristic did not work: soft3 compressed
  harder but lost too much language signal, and soft2 matched BLT-2 compression
  but still trained worse. Do not promote heuristic dynamic patching. If
  dynamic patching is revisited, use a learned entropy patcher or a stronger
  official-reference implementation, not this hand-built boundary rule.
  Stage94N then tested a ByteFlow-inspired proxy that used learned byte
  embedding change as a cheap stand-in for coding-rate chunking. It compressed
  the eval latent sequence from Stage94I's roughly `178.36` mean latent length
  to about `152.61`, but final eval loss worsened from `2.3202` to `2.6982`.
  This rejects the proxy, not the ByteFlow paper: the real mechanism is
  information-theoretic/learned chunking, while the proxy was only a local
  embedding-change heuristic.
  Stage94O then tried the requested Fast-BLT + ByteFlow + H-Net integration as
  a small HBF-BLT v0 gate: keep the BLT local byte decoder, use UTF-8-safe
  H-Net-style hierarchy constraints, and close patches with a ByteFlow-style
  coding/change score. It preserved roughly the same compression as Stage94I
  (`mean_latent_len` about `176.6-176.9` versus Stage94I `178.36`), but final
  eval loss was `2.7490`, far worse than Stage94I `2.3202`. This rejects the
  heuristic HBF boundary rule. It does not reject a true learned H-Net/ByteFlow
  chunker, but it proves that "merge the paper ideas as hand-written boundary
  rules" is not the high-probability path.
  Stage94P then removed the permanent BLT-2 fallback and made the learned
  semantic chunker the main answer path. The learned gate did not simply
  collapse: by the 1200-step run it logged gate means around `0.3`, entropy
  around `0.5`, and std around `0.2`. However, it compressed much harder than
  BLT-2 (`mean_latent_len=89.41` versus Stage94I `178.36`) and final eval loss
  was `2.8859`. This rejects learned-primary v0 at 4-byte chunks. It does not
  reject learned-primary BLT as a direction; it says the first learned-primary
  version was too compressed/weak and needs a gentler 2-byte or multi-scale
  learned path before it can challenge BLT-2.
  Stage94Q then ran that gentler 2-byte learned-primary gate. It exactly
  matched Stage94I's eval compression (`mean_latent_len=178.36`) and kept the
  learned chunk gate alive (`gate_mean` about `0.31-0.34`, entropy about
  `0.55`, std about `0.18`). However, final eval loss was `2.3540`, still
  worse than Stage94I `2.3202`. This rejects the current learned-primary soft
  chunk embedding even when compression is no longer the excuse. The failure is
  now plain: the model has learned to move a soft boundary knob, but that knob
  is not yet a better reading habit than fixed two-byte notes.
  Stage94R then made the boundary scorer change the actual latent sequence.
  This was the first true-boundary smoke in this repo: eval mean latent length
  moved to about `162.3-168.9`, close to the Stage94I BLT-2 baseline
  `178.36`, and the logs included `learned_boundary_prob_mean`,
  `learned_boundary_prob_std`, and `learned_boundary_valid_boundaries`. It
  still lost badly: final eval loss `2.8124` versus Stage94I `2.3202`.
  Therefore the missing piece is no longer "can the boundary change the global
  sequence?" It can. The missing piece is the H-Net dechunk/speaker contract:
  after shortening, the model must map the shortened thought back onto the
  original byte positions, not just stuff ragged variable chunks into the old
  BLT local patch decoder.
  Stage94S/T/U tested that dechunk/speaker story directly. The plain-language
  result is: giving the shortened thought its own mouth helped a lot compared
  with Stage94R, and a mild boundary prior helped more, but it still did not
  beat fixed BLT-2. Increasing the prior target from `0.5` to `0.65` made the
  result worse, so the next fix is not "keep more boundaries" or "tune H-Net
  harder." The remaining H-Net issue is contract-level correctness only: use
  official-style EMA dechunk if H-Net is compared again, but do not spend long
  runs on H-Net until a newer boundary mechanism beats the fixed BLT-2 anchor.

  Stage97A local H-Net teacher gate:
  2026-05-24 local-only gate tested `hnet_dechunk + raw-byte teacher distill`
  for 400 steps, using the same Stage94 byte sample and the Stage94C raw-byte
  teacher. This was the smallest test for the plain-language idea: "a student
  that cuts meaning chunks should recover language signal if a raw-byte teacher
  shows it how to speak." The path loaded, trained, used the official GDN2
  runtime, kept H-Net boundaries alive, and produced no eval nonfinite batches,
  but it did not beat the existing anchors:

  ```text
  run                              | eval@400 | mean latent | plain read
  Stage94I BLT-2 clean             | 2.7642   | 178.36      | fixed 2-byte notes still stronger
  Stage94Y BLT-2 raw-byte teacher  | 2.7412   | 178.36      | teacher helps fixed BLT-2
  Stage94T H-Net dechunk prior     | 2.8644   | 130.81      | H-Net dechunk helps but trails BLT-2
  Stage97A H-Net raw-byte teacher  | 3.0016   | 92.03       | teacher cannot rescue too-coarse chunking
  ```

  문과적으로 말하면: 지금 H-Net 학생은 선생의 발음을 듣긴 하지만, 노트를 너무
  크게 접어서 원문 맛을 잃는다. 그래서 "H-Net + teacher"는 바로 큰 run으로
  승격하지 않는다. H-Net++가 필요하다면 boundary loss나 teacher만 더하는 방식이
  아니라, `hierarchical router -> chunk mixer -> document/sequence prior ->
  dechunk/speaker` 계약을 제대로 구현한 H-Net++-mini gate로만 다시 비교한다.

  Stage97B local H++Flow-BLT teacher gate:
  2026-05-24 local-only gate implemented `hnetpp_flow_dechunk`: the normal
  H-Net dechunk/speaker answer path remains intact, but ByteFlow-style adjacent
  embedding change can open an extra boundary when the learned semantic scorer
  would otherwise fold too much text together. This is the strongest
  plain-language combination so far:

  ```text
  BLT-2 fixed chunking      = safe training wheels / stable note folding
  raw-byte teacher distill  = teacher that preserves original byte-language taste
  H-Net++-style scorer      = student learning where meaning units begin
  ByteFlow-style gate       = auditor that stops over-compression from hiding information
  ```

  The implementation passed the local boundary/parser test gate and the full
  `tests.test_blt_hbf_boundary` suite. The 400-step local loss gate rejected it:

  ```text
  run                                | eval@400 | mean latent | plain read
  Stage97A H-Net raw-byte teacher    | 3.0016   | 92.03       | too coarse, but slightly lower loss
  Stage97B H++Flow-BLT teacher mini  | 3.0233   | 142.84      | preserves more information, but loss worse
  Stage94Y BLT-2 raw-byte teacher    | 2.7412   | 178.36      | strongest 400-step compressed anchor
  ```

  문과적으로 말하면: Stage97B는 노트를 너무 크게 접는 문제는 완화했다. 그러나
  아직 "어디서 접어야 의미가 산다"를 배운 진짜 H-Net++가 아니라, embedding 변화가
  큰 곳을 사람이 만든 규칙으로 더 여는 H++Flow-mini다. 정보는 더 남겼지만 글을
  더 잘 말하지는 못했다. Therefore `hnetpp_flow_dechunk` is useful as a
  diagnostic mode, not as the next foundation-training default. The default
  compressed path remains Stage94Y-style `BLT-2 + raw-byte teacher distill`
  until a learned hierarchical chunker beats it on the same 400-step gate.

  Stage97C local BLT-2 add-cross teacher gate:
  2026-05-24 local-only gate tested the anchor-preserving alternative:
  keep Stage94Y fixed `BLT-2 + raw-byte teacher distill`, but switch
  `decoder_latent_mode` from `add` to `add_cross` so the local byte speaker can
  cross-attend to previous latent notes. This did not change the boundary
  contract, so it was a cleaner test than another H-Net boundary rule.

  ```text
  run                              | eval@400 | mean latent | plain read
  Stage94Y BLT-2 teacher, add      | 2.6969   | 178.36      | best local 400-step compressed anchor
  Stage97C BLT-2 teacher, add_cross| 2.7146   | 178.36      | more reading paths, slightly worse loss
  ```

  문과적으로 말하면: 말하는 학생에게 노트를 더 많이 보여주면 좋아질 것처럼
  보였지만, 초반 학습에서는 통로가 많아져 산만해졌다. 더 많은 cross-attention은
  "의미를 더 잘 안다"가 아니라 "읽을 곳이 더 많다"일 뿐이다. Therefore
  `add_cross` is not promoted for this BLT-2 teacher path. The current default
  remains plain `decoder_latent_mode=add` with raw-byte teacher distill.

  Stage97D local learned hierarchical chunker gate:
  2026-05-24 local-only gate implemented and tested `decoder_latent_mode=hier_add`.
  This is the first "true learned hierarchy" test in this BLT branch that does
  not use a hand boundary rule: fixed BLT-2 micro-patches remain the safe base,
  then a differentiable learned gate/projection builds an upper-level memory
  from the previous two latent notes and adds it to the normal byte-speaking
  path. The unit test verifies that the hierarchy receives gradient from
  `forward_losses`, so this is not a side metric.

  ```text
  run                              | eval@400 | mean latent | plain read
  Stage94Y BLT-2 teacher, add      | 2.696948 | 178.36      | current compressed anchor
  Stage97D BLT-2 teacher, hier_add | 2.696974 | 178.36      | learned hierarchy opens but ties/slightly trails
  ```

  The learned hierarchy did activate: eval `hier_chunk_gate_mean` moved from
  `0.498` at step 1 to `0.997` at step 400, and `hier_chunk_memory_norm`
  increased from `3.81` to `4.81`. 문과적으로 말하면: 학생이 상위 메모장을
  실제로 펼쳐서 쓰기 시작했지만, 현재 과제와 400-step gate에서는 기존 BLT-2
  teacher 필기만으로도 거의 같은 답을 냈다. Therefore `hier_add` is a valid
  diagnostic and a plausible longer-run candidate, but it is not promoted over
  Stage94Y unless it beats the anchor on the same eval gate or shows a clear
  reasoning/generation gain that loss alone misses.

  Stage94W then pivoted away from H-Net, following the user's instruction not
  to cling to a weak family. It implemented a newer BLT-style entropy patcher
  proxy: build unigram/bigram byte surprisal from the actual local corpus, keep
  a fixed patch budget like official BLT entropy patching, and start patches at
  the most surprising byte transitions. At the 400-step gate it reached
  `3.1604`, slightly better than Stage94N ByteFlow-proxy `3.1907@400`, but far
  behind Stage94I fixed BLT-2 `2.7642@400`. At the final 1200-step gate it
  reached `2.5601`: better than Stage94N `2.6982` and HBF-BLT Stage94O
  `2.7490`, but worse than Stage94T H-Net prior `2.4817` and still clearly
  worse than Stage94I `2.3202`. This means "information boundary" is a better
  story than random embedding-change boundaries, but a cheap n-gram entropy
  proxy is still not the big jump.

  Stage94W decision: do not keep tuning cheap hand-made entropy boundaries.
  The next high-probability tokenizer-free move is either an official/neural
  BLT entropy model, a true ByteFlow coding-rate router, or a raw-byte
  efficiency path that preserves Stage94C's strong loss while making compute
  tractable. If none of those is available immediately, keep fixed BLT-2 as the
  scalable baseline and spend research time on the language/reasoning data
  curriculum rather than another boundary heuristic.

  Stage94X then tested the raw-byte efficiency assumption directly instead of
  repeating the phrase "raw bytes are not scalable." It used the same native
  raw-byte PrefixLM path as Stage94C, but raised `seq_len` from `384` to `768`
  with `batch_size=4` on the local RTX 4090. This unlocked far more usable rows
  in the same sampled corpus (`38,499` rows at seq384 versus `106,729` rows at
  seq768), finished a 400-step capacity probe without OOM, and reached
  `eval_loss=2.1387` on a larger eval set with `32,015` target tokens. Speed was
  still practical at about `10.4k` compute tokens/sec over the run. The
  plain-language lesson is important: the student who reads every letter is not
  obviously too slow at this scale; the fixed BLT-2 student folds notes neatly,
  but currently loses more meaning than it saves.

  Stage94Y then made the latent route more honest: instead of asking a hand
  boundary rule to guess where meaning lives, it used the strong raw-byte model
  as a teacher and trained the BLT-2 student's actual byte-speaking logits to
  match the teacher's next-byte distribution. In plain language, the teacher
  reads every letter and tells the note-taking student what answer distribution
  it should still hear after folding nearby bytes into a shorter latent note.
  This is not a side probe: the KL is applied to `forward_logits`, the same
  byte path that produces evaluated language loss. The first 400-step local
  gate reached `eval_clean_loss=2.6969`, beating the Stage94I fixed BLT-2
  400-step anchor `2.7642` while preserving the same `mean_latent_len=178.36`.
  The full 1200-step gate then reached `eval_clean_loss=2.2999`, beating the
  Stage94I final anchor `2.3202`. In the 1200-step run, eval loss moved
  `6.3111 -> 2.7412 -> 2.4363 -> 2.2999` at steps `1/400/800/1200`, and
  teacher KL fell from `4.37` at step 1 to `0.55` at step 1200. This is the
  first compressed-latent result in Stage94 that beats fixed BLT-2 without
  giving up the 2x latent compression.

  Same-seed teacher-off control then removed the teacher KL while keeping seed
  `9682`, data, BLT-2 patching, model size, optimizer, eval rows, and
  `mean_latent_len=178.36` fixed. It reached eval losses
  `6.3111 -> 2.7777 -> 2.4713 -> 2.3345` at steps `1/400/800/1200`. Therefore
  the teacher path improved the full gate by `0.0346` loss (`2.3345 -> 2.2999`)
  and improved every logged eval point. This is a real but small causal gain:
  raw-byte teacher distillation helps the compressed latent student preserve
  meaning, but it has not closed the raw-byte quality gap.

  Current Stage94 decision: raw byte-free is the active tokenizer-free quality
  baseline, and fixed BLT-2 clean-only is only the active compressed-latent
  baseline. BLT-2 plus raw-byte teacher distillation is now the leading
  compressed-latent candidate because it is the first variant to beat fixed
  BLT-2 at the full 1200-step gate and it beats a same-seed teacher-off
  control. Do not discard raw byte-free as "unscalable" without a measured
  long-sequence or larger-model capacity
  failure. Hand-made boundary rules are paused: they repeatedly lost language
  signal. Promote a compressed tokenizer-free path only if raw-byte
  distillation, an official/neural BLT entropy model, true ByteFlow coding-rate
  router, BLT-DV generation gate, or another measured raw-byte efficiency
  mechanism beats the raw-byte quality curve or preserves it with a clear
  compute win.

  HBF-BLT v0 lesson:

  ```text
  rejected:
    UTF-8-safe heuristic hierarchy
    + embedding-change/coding proxy boundary
    + BLT local decoder

  reason:
    it changed the actual answer path and preserved compression, but it made
    next-byte modeling worse. The model did not benefit from irregular
    hand-written boundaries at this scale.

  allowed next version:
    learned-primary semantic chunking. Fixed BLT-2 may be used only as a
    teacher, warm-start, baseline, or ablation; it must not remain mixed into
    the normal forward path if the run is claimed as "self-learned semantic
    BLT".
  ```

  Stage94P principle:

  ```text
  main path:
    UTF-8 bytes
    -> local byte encoder
    -> learned semantic chunker
    -> learned latent chunks
    -> recurrent/global thought core
    -> local byte decoder
    -> clean next-byte CE

  fixed BLT-2 role:
    teacher / warm-start / baseline / ablation only
    not a permanent fallback in the normal answer path
  ```

  문과적으로 말하면: BLT-2는 선생님이지 몸이 아니다. 처음에는 "두 byte씩
  읽으면 이런 모양이 된다"는 교재로 쓸 수 있지만, 최종 경로는 모델이 스스로
  "여기까지가 한 뜻이다"라고 접어야 한다.

  Stage94P result:

  ```text
  rejected v0:
    learned-primary chunker with max 4-byte chunks

  useful signal:
    the learned gate was trainable and did not collapse to a constant, so the
    causal answer path is alive.

  failure:
    4x compression is too aggressive for this small 82M gate. The model learns
    faster throughput but loses language signal relative to BLT-2.

  next allowed variant:
    not another scalar tweak. First audit the official/reference contract and
    strongest anchor. A future learned chunker must add a real boundary
    objective, warm-start/teacher schedule, or official H-Net/ByteFlow-style
    coding-rate mechanism; changing patch-size alone has now been tested.
  ```

  Stage94Q result:

  ```text
  anchor:
    Stage94I BLT-2 clean-only
    final_eval_loss=2.3202
    mean_latent_len=178.36

  tested repair:
    Stage94Q learned-primary BLT-2-scale
    patch_size=2
    final_eval_loss=2.3540
    mean_latent_len=178.36

  useful signal:
    the learned chunk gate stayed trainable rather than collapsing.

  failure:
    same compression, worse language loss. The learned soft chunk embedding is
    not yet carrying byte identity/order as cleanly as the fixed two-byte fold.

  next action:
    freeze boundary-rule ideation. Reopen the strongest working path and the
    closest official/reference implementation, then change one contract only.
  ```

  Stage94R result:

  ```text
  tested repair:
    learned boundary scorer controls the actual patch sequence seen by the
    global core.

  evidence that the repair was real:
    step1200 mean_latent_len=168.875
    learned_boundary_prob_mean=0.5003
    learned_boundary_valid_boundaries=536.14

  reject:
    final_eval_loss=2.8124
    Stage94I fixed BLT-2 anchor=2.3202

  plain-language failure:
    the model learned where to fold the paper, but then tried to speak through
    the old fixed-patch mouth. The answer path needs a dechunker: shortened
    thought must be spread back onto every original byte position before the
    LM head speaks.

  next allowed variant:
    H-Net-style boundary -> chunk -> global core -> dechunk -> byte LM head.
    Do not launch another boundary-threshold run before this dechunk contract
    is tested.
  ```

  Default loss contract:

  ```text
  default:
    UTF-8 bytes
    -> BLT-2 fixed latent patches
    -> recurrent/global thought core
    -> local byte decoder
    -> next-byte logits
    -> clean next-byte cross entropy

  not default:
    NITP representation-shaping loss
    diffusion reconstruction loss
    hand-written boundary reward
    side probe / auxiliary answer head
  ```

  Do not add NITP by habit. The active default is the normal answer path's
  clean next-byte CE loss because it directly trains the state that speaks.
  Auxiliary losses are allowed only after the clean path has a measured failure
  that the auxiliary target specifically explains.

  Latent-tokenizer ranking:

  ```text
  1. BLT-2 clean-only:
     active baseline because it is already implemented and currently has the
     best scalable byte-latent loss in this repo.

  2. Real ByteFlow-style learned/coding-rate chunker:
     highest-probability next latent-tokenizer upgrade. It should be tested as
     a proper learned boundary/chunking objective, not as another hand-written
     byte-class or embedding-delta rule.

  3. H-Net/H-Net++ dynamic hierarchy:
     stronger architectural reset candidate for Korean/code/morphology, but it
     changes more of the model body and should follow a small controlled gate.

  4. NITP:
     useful representation-shaping auxiliary only after the answer path is
     healthy; it is not a tokenizer replacement and should not be the default
     first move after Stage94M.
  ```

  문과적으로 말하면: Fast-BLT는 이미 글자를 접어 읽는 학생을 "더 빨리 말하게"
  만드는 논문이고, ByteFlow/H-Net은 "어디까지가 한 덩어리인지 스스로 느끼게"
  만드는 논문이다. 지금 우리에게 더 근본적인 latent tokenizer 후보는
  ByteFlow/H-Net 쪽이지만, 현재 구현 증거로는 BLT-2 fixed patch가 가장 덜
  망가지는 기준선이다.

  Stage94Q paper-escalation decision:

  ```text
  problem:
    Stage94P/Q proved that our learned_primary path is not yet a true learned
    tokenizer. It still packs fixed windows, then learns soft weights inside
    each window. That is a learned note-taking style inside a fixed notebook,
    not a learned decision about where a semantic note begins and ends.

  stronger references now pinned locally:
    references/official/blt
      official BLT code; use its entropy patcher and patch-length contract as
      the BLT reference, not our simplified BLT-D trainer.
    references/official/hnet
      official H-Net code; use RoutingModule -> ChunkLayer -> DeChunkLayer as
      the reference for true learned/dynamic chunking.
    references/official/flexitokens
      learnable boundary-tokenizer reference; use it to avoid forcing one fixed
      compression rate across all samples/languages.

  paper priority:
    1. H-Net / Dynamic Chunking:
       true end-to-end learned boundary, chunk, and dechunk body.
    2. ByteFlow:
       coding-rate / Top-K adaptive byte compression. Prefer the real
       information/compression criterion over embedding-delta proxies.
    3. FLEXITOKENS:
       boundary predictor and flexible compression-rate objective, especially
       relevant for Korean/English multilingual over-fragmentation.
    4. Fast BLT:
       useful later for decoding speed and BLT-DV verification, but not the
       first fix for Stage94Q's boundary-learning failure.

  next run rule:
    Do not call the next tokenizer-free run "learned semantic BLT" unless the
    boundary mask itself changes the shortened sequence that the global core
    sees, and the decoder/dechunk path maps that shortened sequence back to
    byte positions on the normal answer path.

  Stage94R satisfied the first half of this rule and failed the second half.
  The next run must therefore repair the dechunk/speaker path, not the boundary
  threshold.
  ```

  NITP placement lesson:
  NITP should be attached only to hidden states on the normal answer path. For
  BLT this means the local decoder hidden that directly feeds the byte LM head,
  not a side probe on global patch state. Stage94M implemented that causal
  placement. The auxiliary target was learnable (`nitp_cosine_similarity`
  reached about `0.36`), but the actual clean eval loss worsened from Stage94I
  `2.3202` to `2.3453`. Therefore "BLT plus NITP" is not automatically better:
  keep NITP as a low-weight/late-phase candidate, not a default ingredient.

  문과적으로 말하면: BLT는 글자를 접는 독서 방식이고, NITP는 입이 다음 글자
  모양을 미리 떠올리는 습관이다. Stage94M에서는 그 습관 자체는 생겼지만,
  실제 답안을 더 정확히 쓰지는 못했다. 습관이 생겼다는 것과 답이 좋아졌다는
  것은 다른 증거다.

- LT2 / Linear-Time Looped Transformers placement:
  `2605.20670` is relevant to the recurrent/looped compute engine, not to the
  tokenizer or immediate speaker alignment problem. In this project:

  ```text
  GRAM/PTRM = how many candidate thoughts/paths are produced, corrected, and selected
  LT2       = how cheaply the looped thought engine can be run for more steps/context
  ```

  Current GRAM/PTRM runs are configurable/test-time compute, but not fully
  adaptive compute unless the model also learns when to halt, when to widen
  search, and when extra thinking is unnecessary. Do not replace the current
  BLT/tokenizer-free gate with LT2. First prove depth scaling:

  ```text
  think_steps 1/2/4/8 -> heldout loss/generation improves monotonically or by family
  ```

  If depth scaling is real but expensive, then test LT2-style GDN+sparse/full
  hybrid mixer as the cheaper looped engine. If extra depth does not improve
  language/generation, LT2 only makes a non-useful loop cheaper.

  Architecture decision: LT2 is not discarded. The active LT2 choice is fixed
  to the Full+GDN 3:1 schedule:

  ```text
  GatedDelta/GDN
  GatedDelta/GDN
  GatedDelta/GDN
  Full attention
  ```

  In code this is `--delta-backend official_gated_delta2 --attn-every 4`.
  `attn_every=2`, `attn_every=8`, GDN-only, and GDN+DSA/sparse variants are not
  the main path; they are ablations only. 문과적으로 말하면: 빠른 생각 세 번
  뒤에 한 번은 전체 문맥을 보고 정신을 다시 맞추는 방식으로 고정한다.
  Therefore the next research question is not "whether to use LT2"; it is
  whether EqR/attractor training can make this fixed LT2 engine improve with
  deeper `think_steps`.

  Stage98A local EqR-style depth/residual probe:
  2026-05-24 added `scripts/560_eval_blt_depth_residual_probe.py` to evaluate
  BLT checkpoints with an EqR-style question: when `think_steps` increases, do
  heldout loss and fixed-point residual improve together? The probe writes both
  a JSON report and JSONL rows, and computes token-weighted heldout loss plus
  token-weighted mean residual between the last two latent depths.

  On the current strongest compressed anchor,
  `20260524_STAGE94Y_LOCAL_BLT2_RAWTEACHER_DISTILL1200/last_model.pt`, using
  the same Stage94 byte sample eval rows:

  ```text
  think_steps | heldout loss | mean fixed-point residual | plain read
  1           | 2.3061       | 0.3917                    | under-thought
  2           | 2.3000       | 0.1673                    | best loss
  4           | 2.3176       | 0.1219                    | more settled, worse answer loss
  8           | 2.3850       | 0.0842                    | most settled, clearly worse loss
  ```

  문과적으로 말하면: 이 모델은 오래 생각하면 점점 조용하고 안정된 생각 상태로
  들어가지만, 그 안정점이 더 좋은 답 상태는 아니다. 즉 현재 Stage94Y에는
  "수렴"은 있지만 EqR식 "좋은 attractor로 수렴"은 아직 없다. Therefore do not
  promote LT2 or longer looped inference as the next fix for this checkpoint.
  The next latent-reasoning fix must train the attractor landscape itself:
  depth-randomized training, residual-to-correctness alignment, or
  convergence-based selection on tasks where correctness/generation is measured.

  Stage98B EqR architecture/training patch:
  `scripts/557_train_blt_d_prefixlm_dataio.py` now has an optional
  EqR-style attractor regularizer for BLT/H-Net runs. It compares shallow,
  previous, and deep recurrent depths on the same supervised byte targets:

  ```text
  --eqr-shallow-think-steps
  --eqr-deep-think-steps
  --eqr-deep-supervision-weight
  --eqr-consistency-weight
  --eqr-residual-weight
  --eqr-improvement-weight
  --eqr-every
  ```

  문과적으로 말하면: 얕은 생각이 바로 말하게 두는 대신, 깊은 생각이 더 좋은
  답으로 수렴하는지 확인하고, 마지막 두 생각이 계속 흔들리면 벌점을 주며, 깊은
  생각이 얕은 생각보다 나쁘면 벌점을 준다. LT2는 이 다음 단계의 최적화다.
  Current code already has the LT2-relevant compute knobs:
  `--delta-backend official_gated_delta2`, `--hybrid-layers`, and
  `--attn-every`. Promote LT2-style GDN+sparse/full-hybrid only after the EqR
  gate shows that deeper `think_steps` improves heldout/generation rather than
  merely reducing fixed-point residual.

  Stage98C local EqR gate result:
  `20260524_STAGE98C_LOCAL_BLT2_EQR_RAWTEACHER_GATE400` trained the fixed
  LT2 Full+GDN 3:1 engine with EqR enabled every 4 steps:

  ```text
  EqR: shallow=1, previous=3, deep=4
  weights: deep_ce=0.05, consistency=0.03, residual=0.05, improvement=0.10
  backend: official_gated_delta2, attn_every=4
  fallback: 0
  ```

  Compared with the no-EqR 400-step anchor
  `20260524_STAGE94Y_LOCAL_BLT2_RAWTEACHER_DISTILL400`, heldout eval loss
  improved from `2.6969` to `2.5655`. This is a useful local signal: EqR is not
  just decorative, it improved the same short gate.

  But the depth/residual probe is still not accepted:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5768       | 0.6000
  2           | 2.5655       | 0.2119
  4           | 2.5963       | 0.1105
  8           | 2.6432       | 0.0750
  ```

  문과적으로 말하면: EqR는 학생을 더 차분하게 만들고 기본 시험 점수도 올렸지만,
  아직 "오래 생각할수록 더 정답"인 학생은 아니다. 지금 다음 실험은 LT2 비율
  쇼핑이 아니라, 고정된 LT2 3:1 몸통 위에서 EqR/attractor가 깊은 생각을 더 좋은
  답으로 밀도록 만드는 것이다.

  Stage98D/98E local EqR bridge results:

  ```text
  run      main CE depth  EqR cadence/weight change             eval loss
  Stage98C 2              deep CE 0.05 every 4 steps            2.5655
  Stage98D 4              same EqR, main answer path depth 4    2.9902
  Stage98E 2              deep CE 0.20 every 1 step             2.5676
  ```

  Stage98D rejects the naive fix "just train and evaluate at depth 4." It made
  the student worse. Stage98E shows that stronger depth-4 answer supervision is
  not catastrophic, but it does not improve the gate over Stage98C.

  Depth probes remain rejected:

  ```text
  Stage98D best depth = 2, best loss = 2.9564
    depth 1  loss 2.9654  residual 1.6706
    depth 2  loss 2.9564  residual 0.2540
    depth 4  loss 2.9902  residual 0.0565
    depth 8  loss 2.9747  residual 0.1854

  Stage98E best depth = 2, best loss = 2.5675
    depth 1  loss 2.5820  residual 0.8429
    depth 2  loss 2.5675  residual 0.2616
    depth 4  loss 2.5880  residual 0.1337
    depth 8  loss 2.6223  residual 0.0691
  ```

  Attractor-style adaptive stopping preflight:
  `scripts/561_eval_blt_attractor_adaptive_depth_from_probe.py` reads depth
  probe rows and asks the 2605.12466-style first cheap question: if convergence
  residual chooses the stopping depth, does the selected depth beat fixed depth
  2?

  Result: no. For Stage98C/D/E the best residual threshold is `0.3`, which
  simply selects depth 2 and exactly recovers the depth-2 loss. Tighter
  thresholds select depth 4 or 8, but answer loss worsens.

  문과적으로 말하면: 지금 반복 코어는 "조용해지는 법"은 배우고 있지만, "정답으로
  조용해지는 법"은 아직 배우지 못했다. Therefore a simple halt head or residual
  threshold is not enough. The next real Attractor-Model experiment must train
  the answer attractor itself: the recurrent state should be pulled toward a
  stable answer state whose logits improve, not merely toward any low-motion
  state.

  Stage99B answer-attractor gate:
  `20260524_STAGE99B_LOCAL_ANSWER_ATTRACTOR_GATE400` added the first direct
  answer-attractor training loss on the canonical LT2 3:1 body:

  ```text
  answer depths: 1, 2, 4
  ce weight: 0.05
  monotonic weight: 0.20
  residual-wrong weight: 0.10
  backend: official_gated_delta2, attn_every=4
  fallback: 0
  ```

  This is the right kind of intervention because it compares the normal answer
  path at multiple recurrent depths and penalizes the loop when deeper thinking
  becomes a worse answer state. It is no longer only asking the state to move
  less.

  The 400-step result is promising but not accepted:

  ```text
  run      heldout eval loss
  Stage98C EqR shallow/deep bridge          2.5655
  Stage98E stronger deep bridge             2.5676
  Stage99B answer-attractor                 2.5504
  ```

  Depth probe:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5571       | 0.6630
  2           | 2.5503       | 0.1988
  4           | 2.5611       | 0.1030
  8           | 2.5863       | 0.0562
  ```

  Offline oracle/adaptive analysis:

  ```text
  oracle best loss: 2.5475
  oracle depth counts: depth1=18, depth2=39, depth4=6, depth8=1
  best residual threshold: 0.3
  best adaptive loss: 2.5503
  selected depth: always depth2
  ```

  Interpretation: Stage99B starts carving the correct kind of basin because it
  improves the eval gate slightly and its answer-attractor metrics are active.
  But the basin is still shallow and centered around depth 2. Deeper depth is
  quieter, not more correct. Do not spend DGX-scale compute on halt selection
  alone; the next attractor version must make the deep state itself
  answer-causal.

  Next accepted-likelihood direction:

  ```text
  1. Keep the normal answer path mandatory.
  2. Train a fixed answer embedding/state target, not only scalar CE at each
     depth.
  3. Penalize stable-wrong states harder: low residual plus worse CE is the
     exact failure.
  4. Add an ablation: disable attractor refinement/readback; the gain must
     disappear.
  5. Promote only if depth4 or convergence-selected depth beats depth2 on
     heldout loss and generation.
  ```

  Stage99C fixed answer-state attractor:
  `20260524_STAGE99C_LOCAL_FIXED_ANSWER_STATE_GATE400` implemented the next
  candidate above. It exposes the decoder hidden state used by the normal
  answer head, then pulls supervised positions toward the gold answer speaker
  embedding row. This gives the attractor an explicit semantic center instead
  of only a scalar CE slope.

  ```text
  answer-state depths: 1, 2, 4
  state weight: 0.03
  monotonic weight: 0.10
  residual-wrong weight: 0.05
  backend: official_gated_delta2, attn_every=4
  fallback: 0
  ```

  The implementation is wired and trainable:

  ```text
  answer_state_attractor_distance:
    step 1   0.9910
    step 400 0.6533
  answer_state best depth at step 400: 4
  answer CE best depth at step 400:   2
  eval loss: 2.5504
  ```

  But the depth probe still rejects:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5552       | 0.6858
  2           | 2.5503       | 0.2024
  4           | 2.5629       | 0.0987
  8           | 2.5859       | 0.0521
  ```

  Offline adaptive stopping also still chooses depth 2:

  ```text
  oracle best loss: 2.5475
  oracle depth counts: depth1=22, depth2=32, depth4=8, depth8=2
  best residual threshold: 0.3
  best adaptive loss: 2.5503
  selected depth: always depth2
  ```

  문과적으로 말하면: Stage99C는 학생의 "마음 방향"을 정답 단어 쪽으로 돌리는
  데에는 성공했다. 그런데 그 마음 방향을 실제 답안 문장으로 읽는 입은 아직
  깊은 생각을 더 좋은 답으로 바꾸지 못한다. 따라서 다음 후보는 scalar loss나
  hidden cosine을 더 키우는 것이 아니라, attractor state를 normal speaker가
  읽는 방식까지 answer-causal하게 연결해야 한다.

  Stage99D answer-causal readback:
  `20260524_STAGE99D_LOCAL_ANSWER_READBACK_GATE400` implemented the first
  minimal answer-causal readback. The normal answer path now performs a
  preliminary speaker projection, converts that into an expected speaker
  embedding, and gates it back into the decoder hidden state before the final
  byte speaker. This is deliberately small: it tests whether the speaker can
  reread its own answer expectation as a global-workspace hint.

  ```text
  answer readback mode: self_embedding
  gate init: -2.0
  observed gate: 0.1192 -> 0.1225
  expected embedding norm: 0.0356 -> 0.2947
  backend: official_gated_delta2, attn_every=4
  fallback: 0
  eval loss: 2.5479
  ```

  This is a small eval improvement over Stage99B/C (`~2.5504`), so the
  readback path is not dead. However, the depth probe still rejects:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5523       | 0.6896
  2           | 2.5486       | 0.1976
  4           | 2.5594       | 0.0993
  8           | 2.5824       | 0.0526
  ```

  Adaptive stopping confirms the same failure:

  ```text
  oracle best loss: 2.5453
  oracle depth counts: depth1=23, depth2=30, depth4=9, depth8=2
  best residual threshold: 0.3
  best adaptive loss: 2.5486
  selected depth: always depth2
  ```

  문과적으로 말하면: Stage99D는 "생각을 입으로 다시 읽히는 통로"를 열었고
  시험 점수는 아주 조금 좋아졌다. 하지만 깊게 생각할수록 정답이 좋아지는
  사람 같은 구조는 아직 아니다. 더 오래 생각하면 마음은 안정되지만, 입이
  더 좋은 답을 말하지는 않는다. 따라서 다음 수술은 self-embedding readback을
  더 세게 하는 것이 아니라, verifier-selected/global-workspace readback처럼
  "어떤 생각을 답안지에 올릴지"를 선택한 뒤 그 선택이 final logits를 바꾸는
  구조여야 한다.

  Stage99E callosal latent-CoT bridge:
  `20260524_STAGE99E_LOCAL_CALLOSAL_LATENT_COT_GATE400` added an inner-speech
  anchor head and `answer_readback_mode=anchor_embedding`. The intent was a
  corpus-callosum-like bridge between latent thought and short language anchors:
  latent state proposes a compact verbal anchor, the anchor is converted through
  the normal speaker embedding space, and the result is gated back into the
  final byte speaker.

  ```text
  answer readback mode: anchor_embedding
  cot anchor loss weight: 0.05
  gate init: -2.0
  eval loss: 2.5554

  cot_anchor_loss:                ~6.39 -> ~2.97
  cot_anchor_accuracy:             0.002 -> 0.243
  cot_anchor_readback_confidence:  0.009 -> 0.261
  ```

  The bridge is therefore learnable, but the 400-step gate rejects it because
  main held-out loss is worse than Stage99D (`2.5479`). The depth probe also
  rejects EqR-style convergence:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5577       | 0.6281
  2           | 2.5558       | 0.1781
  4           | 2.5690       | 0.0964
  8           | 2.6023       | 0.0482
  ```

  Offline adaptive stopping confirms the same failure:

  ```text
  oracle best loss: 2.5521
  oracle depth counts: depth1=24, depth2=32, depth4=8
  best residual threshold: 0.2
  best adaptive loss: 2.5558
  selected depth: always depth2
  ```

  문과적으로 말하면: 언어 앵커는 배웠지만, 그 말풍선이 정답을 더 잘 말하게
  만드는 뇌량은 아직 아니다. 현재 통로는 "생각과 말 사이에 작은 번역기를
  붙인 것"에 가깝고, "여러 생각 중 어떤 생각을 전면 작업대에 올릴지 고르는
  방송국"은 아니다. 다음 실험은 CoT anchor loss를 더 키우는 것이 아니라,
  readback-off ablation, anchor-first/readback-later schedule, 또는
  verifier-selected/global-workspace readback이어야 한다.

  Stage99F selected workspace readback:
  `20260524_STAGE99F_LOCAL_SELECTED_WORKSPACE_READBACK_GATE400` replaced local
  per-token anchor readback with `answer_readback_mode=selected_anchor_embedding`.
  The inner-speech anchor is first converted into speaker embeddings, a small
  workspace selector then picks a single broadcast vector over valid positions,
  and that vector is gated back into the same final byte speaker. This is the
  first implementation of the "callosal workspace" route:

  ```text
  latent thought -> inner-speech anchor -> workspace selector
  -> broadcast readback vector -> same byte speaker
  ```

  The implementation is alive and the default diffusion path bug found during
  the smoke was fixed (`valid = grouped_mask[..., None]`). But the 400-step gate
  rejects:

  ```text
  Stage99D self readback eval loss:        2.5479
  Stage99E anchor readback eval loss:      2.5554
  Stage99F selected workspace eval loss:   2.5563

  cot_anchor_loss:                         ~6.32 -> ~2.64
  cot_anchor_accuracy:                      0.010 -> 0.325
  cot_anchor_readback_confidence:           0.010 -> 0.236
  answer_workspace_selection_confidence:    0.0147 -> 0.0085
  ```

  The depth probe still rejects:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5616       | 0.6665
  2           | 2.5566       | 0.1904
  4           | 2.5703       | 0.0947
  8           | 2.5964       | 0.0495
  ```

  Adaptive stopping again collapses to depth 2:

  ```text
  oracle best loss: 2.5544
  oracle depth counts: depth1=19, depth2=41, depth4=3, depth8=1
  best residual threshold: 0.3
  best adaptive loss: 2.5566
  selected depth: always depth2
  ```

  문과적으로 말하면: Stage99F는 방송국 건물을 만들었지만 편집장이 없다. 언어
  앵커는 더 잘 배웠지만, workspace selector는 어떤 생각을 전면에 올려야 하는지
  배우지 못해 거의 전체를 흐릿하게 평균낸다. 따라서 다음 단계는 단순
  `selected_anchor_embedding`이 아니라, selector 자체에 "이 후보를 올리면
  final speaker loss가 내려간다"는 verifier/critic 신호를 줘야 한다. 가능한
  최소 다음 gate는 readback-off ablation과 selector-supervised gate다:

  ```text
  candidate depths/positions -> score by answer CE or verifier
  -> train selector toward the better candidate
  -> broadcast selected vector
  -> same byte speaker
  ```

  Stage99G selector critic:
  `20260524_STAGE99G_LOCAL_SELECTOR_CRITIC_GATE400` trained the workspace
  selector with the low-anchor-CE candidate as the editor target. This proved
  that the editor can learn a sharp target, but it still failed the main gate:

  ```text
  eval loss: 2.5581
  accepted: false

  answer_workspace_selector_selection_confidence: ~0.10 -> 0.53
  answer_workspace_selector_target_argmax_match:   0.0 -> 0.875
  answer_workspace_selection_confidence:           ~0.015 -> 0.186
  cot_anchor_accuracy:                             ~0.010 -> 0.335
  ```

  The depth probe again rejected EqR-style deeper thinking:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5663       | 0.6874
  2           | 2.5588       | 0.1986
  4           | 2.5754       | 0.0966
  8           | 2.6031       | 0.0502
  ```

  Adaptive stopping also collapsed to depth 2:

  ```text
  oracle best loss: 2.5570
  oracle depth counts: depth1=13, depth2=49, depth4=2
  best residual threshold: 0.3
  best adaptive loss: 2.5588
  selected depth: always depth2
  ```

  문과적으로 말하면: 편집장은 생겼지만 시험 점수표가 아니라 말풍선 점수표를
  보고 편집했다. selector는 anchor CE가 낮은 위치를 잘 고르게 됐지만, 그
  선택이 최종 speaker가 정답을 더 잘 말하게 만들지는 못했다. 따라서 다음
  실험은 selector weight를 키우는 것이 아니라, selector target을 final speaker
  CE 또는 readback-on/off causal delta로 바꾸는 것이어야 한다.

  Stage99H final-speaker causal critic:
  `20260524_STAGE99H_LOCAL_FINAL_CE_SELECTOR_GATE400` changed the critic
  scorecard from anchor CE to final speaker CE: each candidate anchor broadcast
  is temporarily applied, then the same final speaker CE decides which
  candidate the selector should prefer.

  ```text
  eval loss: 2.5556
  accepted: false

  cot_anchor_accuracy:                                ~0.010 -> 0.330
  final-ce selector target confidence:                0.126
  final-ce selector target argmax match:              0.0
  final-ce best-vs-mean CE improvement:               0.0017
  answer_workspace_selection_confidence:              0.0046
  ```

  The key diagnostic is the target confidence: with 8 candidates, a uniform
  target is 0.125. Stage99H ended at 0.126, so the final speaker saw almost no
  meaningful difference between candidate broadcasts. The selector did not fail
  because it was too small; it failed because there was almost nothing
  answer-causal to select.

  Stage99H depth/adaptive probes confirm the same depth-2 basin:

  ```text
  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5614       | 0.6695
  2           | 2.5560       | 0.1914
  4           | 2.5703       | 0.0951
  8           | 2.5970       | 0.0500

  oracle best loss: 2.5539
  oracle depth counts: depth1=17, depth2=42, depth4=4, depth8=1
  best residual threshold: 0.3
  best adaptive loss: 2.5560
  selected depth: always depth2
  ```

  HRM-Text contrast:
  HRM-Text works because reading, recurrent thinking, and speaking are trained
  as one native body: token reader -> recurrent hidden state -> same LM head.
  The current BLT/PV-GRAM bridge is still a later-added connection:
  byte/latent reader -> recurrent thought -> anchor/readback bridge -> final
  byte speaker. The anchor can learn, but the final speaker does not yet treat
  the readback as answer-causal evidence. In plain Korean: HRM-Text는 태어날 때부터
  한 몸으로 배운 학생이고, 우리는 아직 생각기관과 말기관 사이에 통역사를
  붙이는 중이다. 통역사가 단어를 조금 배웠지만, 말하는 쪽이 그 신호를
  답안지 점수로 읽지 못한다.

  Stage99 reset:

  ```text
  Stop treating bridge/readback/anchor/selector as the main solution.

  The next main architecture must be HRM-Text-style one body:

  input bytes/tokens
  -> native reader
  -> mandatory recurrent thought/core/search
  -> same decoder hidden state
  -> same LM head
  -> text
  ```

  Bridge/readback/anchor/selector modules are diagnostic-only after Stage99H.
  They may be used only to reproduce or ablate the failure, not as the proposed
  next fix. The actual next fix is to remove the shortcut where the byte/local
  speaker can lower CE without depending on recurrent thought. In plain Korean:
  더 좋은 통역사를 붙이는 문제가 아니라, 처음부터 읽고 생각하고 말하는 몸을
  하나로 다시 묶는 문제다.

  Stage99I first one-body decoder gate:
  `20260524_STAGE99I_LOCAL_ONE_BODY_GATE400` implemented
  `decoder_latent_mode=one_body`. This blocks the direct grouped-byte decoder
  shortcut in the final clean decoder input; the LM head receives decoder states
  built from recurrent thought conditioning plus byte-position embeddings, not
  raw grouped byte embeddings as an answer shortcut. Stage99 bridge/readback
  knobs were left off and are now blocked by default unless a command is
  explicitly marked diagnostic.

  ```text
  eval loss: 2.5603
  accepted: false
  final train clean_loss: 2.4694

  think_steps | heldout loss | mean fixed-point residual
  1           | 2.5620       | 0.6703
  2           | 2.5604       | 0.1890
  4           | 2.5784       | 0.0994
  8           | 2.6086       | 0.0544

  oracle best loss: 2.5574
  oracle depth counts: depth1=32, depth2=30, depth4=1, depth8=1
  best residual threshold: 0.3
  best adaptive loss: 2.5604
  selected depth: always depth2
  ```

  Interpretation: Stage99I removes the worst shortcut, but it does not yet
  create the HRM-Text-like answer attractor. The state becomes quieter at depth
  4/8, but not more correct. In plain Korean: 한 몸으로 묶는 첫 끈은 제대로
  걸었지만, 아직 그 몸이 오래 생각할수록 정답 쪽으로 가는 습관을 배운 것은
  아니다. Therefore the next main work remains one-body training contract and
  causality, not a return to bridge/readback/selector.

  Do not promote BLT-D-4 to the main from-scratch run until it beats raw
  byte-free on the same byte Data-IO eval or wins a generation/throughput gate
  that matters for the target scale. Keep raw byte-free as a control only; do
  not use it as the final architecture for large runs because its global
  compute does not scale with long byte sequences.
- VPO placement:
  Vector Policy Optimization / candidate-diversity training belongs after
  ordinary free generation and candidate proposer/verifier plumbing work. It is
  not the first pretraining fix. Use it later to improve pass@k, best@k,
  unique valid candidates, and verifier-selected accuracy once the model can
  already produce non-degenerate answer candidates.
- Local tokenizer audit:
  `scripts/543_audit_prefixlm_multilingual_tokenizer.py` measures
  `tokens_per_nonspace_char` on the multilingual probe set before declaring
  the current tokenizer sufficient. The first local smoke audit on
  `data/eval/prefixlm_multilingual_probe.jsonl` warned on Korean
  fragmentation: Korean max `tokens_per_nonspace_char = 2.6667`, while
  Spanish/German/English probe rows stayed below `0.51`.
- Decision from the audit:
  do not swap tokenizers before Stage93 multilingual training, because changing
  the tokenizer would invalidate checkpoint continuation. But do not claim
  multilingual efficiency yet; Korean needs generation + fragmentation evidence
  after multilingual curriculum training.
- Stage94 native multimodal:
  current high-probability route is still text-spine graft: Stage93 checkpoint
  -> visual reader/projector/resampler -> same recurrent core and LM head.
  Qwen3-VL/Qwen2.5-VL imply that dynamic resolution, spatial/temporal position
  handling, and multi-level visual features are important. LLaDA2.0-Uni/UGen
  imply that a true unified visual tokenizer matters later if we want image
  generation/editing, not just visual question answering.
- Native claim rule:
  a multimodal paper is relevant only if it changes the normal causal path:
  visual/text input -> shared thought state -> same answer logits. External OCR
  or a side visual solver is useful tooling, but not native multimodal
  reasoning.

Plain-language read:
the current student can become multilingual by reading multilingual books with
the same brain. For vision, first attach eyes to the same student. A full visual
tokenizer is like teaching the student to also draw images, not merely answer
questions about images; that is a later, larger stage.

Primary-source anchors:

- LLaDA2.0-Uni: https://arxiv.org/abs/2604.20796
- Teaching Old Tokenizers New Words: https://arxiv.org/abs/2512.03989
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- FLEXITOKENS: https://arxiv.org/abs/2507.12720
- One Tokenizer To Rule Them All: https://arxiv.org/abs/2506.10766
- UGen: https://arxiv.org/abs/2503.21193
- Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Fast Byte Latent Transformer: https://arxiv.org/abs/2605.08044
- Next Implicit Token Prediction: https://github.com/aHapBean/NITP

## Agentic Multi-Turn And Tool-Calling Paper Watch

As of 2026-05-23, agent ability is a separate curriculum and evaluation axis.
It should not be inferred from ordinary language loss, single-turn instruction
following, or one-shot function-call formatting.

Plain-language target:

```text
An assistant is not only a speaker.
It must remember what the user wanted, choose a tool, fill arguments, observe
the result, update its working memory, and answer or continue the task without
losing the thread.
```

Latest-first mechanism table:

```text
date        | source             | mechanism
2026-04-13  | UniToolCall         | unified Query-Action-Observation-Answer representation, 22k+ tool pool, 390k+ training instances, cross-turn Anchor Linkage
2026-02-13  | MT-AgentRisk        | multi-turn tool-agent safety benchmark and ToolShield defense for unsafe long-horizon tool use
2025-08-11  | MCPToolBench++      | large MCP tool-use benchmark over 4k+ MCP servers and multi-step tool calls
2025-07-29  | MemTool             | short-term tool/MCP context memory management across 100 consecutive interactions
2025-07-07  | MemoryAgentBench    | memory-agent benchmark covering retrieval, test-time learning, long-range understanding, selective forgetting
2025-06-09  | tau^2-bench         | dual-control conversational agents where user and agent both act in a shared tool environment
2025-05-22  | T1                  | multi-turn tool-oriented planning dataset with inter-tool dependencies and cache reuse/replanning
2025-05-19  | DialogTool          | stateful multi-turn tool lifecycle: tool creation, awareness, selection, execution, role-consistent response
2024-08-08  | ToolSandbox         | stateful conversational tool-use benchmark with dynamic intermediate/final milestone evaluation
2024-06-17  | tau-bench           | realistic tool-agent-user interaction with database-state end evaluation and pass^k reliability
2024-04-30  | OSWorld             | executable desktop/GUI environment for multimodal computer-use agents
2023-07-31  | ToolLLM/ToolBench   | real-world API instruction generation and solution-path annotation
2023-02-09  | Toolformer          | self-supervised API call insertion: when to call, what arguments, how to use results
2022-10-06  | ReAct               | interleaved reasoning/action/observation loop
```

Local implication for QTRM/PV-GRAM native training:

- Stage93 text-spine:
  continue ordinary HRM-Text/data-io language and reasoning first. Do not claim
  agent ability from Stage93.
- Agentic continuation:
  add a structured trajectory format after the base speaker is stable:

  ```text
  user/query -> internal thought/register -> tool_call JSON/token block
  -> tool_result/observation -> memory update -> final answer or next action
  ```

- Tool-call representation:
  prefer a unified QAOA-style grammar over ad hoc text. The model should learn
  tool name, arguments, observation ingestion, and final answer as one token
  path.
- Memory requirement:
  multi-turn agentic ability requires short-term tool-context memory and
  longer-term task memory. A model that forgets prior tool results after one
  turn is a chatbot with tools, not an agent.
- Evaluation gate:
  start with small local stateful tool simulations before DGX-scale training:
  1. single-turn strict JSON/function-call accuracy;
  2. multi-turn argument carryover;
  3. observation-grounded final answer;
  4. state-changing tool sequence success;
  5. refusal/safety when a tool would cause harm;
  6. ablation: removing observation/memory should remove the gain.
- Native path rule:
  tool calls must come from the normal model output path. A hard-coded router
  can be used as scaffolding, but the promoted model must emit and consume tool
  calls/observations through the same recurrent thought and LM-token speaker.

Plain-language read:
the next agentic stage is like teaching the same student office procedure:
remember the request, pick the right form, fill it, read the receipt, update the
case file, then answer. A side script that secretly chooses the tool is useful
engineering, but it is not proof that the student learned agentic work.

Primary-source anchors:

- UniToolCall: https://arxiv.org/abs/2604.11557
- MT-AgentRisk / ToolShield: https://arxiv.org/abs/2602.13379
- MCPToolBench++: https://arxiv.org/abs/2508.07575
- MemTool: https://arxiv.org/abs/2507.21428
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- tau^2-bench: https://arxiv.org/abs/2506.07982
- T1: https://arxiv.org/abs/2505.16986
- DialogTool: https://arxiv.org/abs/2505.13328
- ToolSandbox: https://arxiv.org/abs/2408.04682
- tau-bench: https://arxiv.org/abs/2406.12045
- OSWorld: https://arxiv.org/abs/2404.07972
- ToolLLM: https://arxiv.org/abs/2307.16789
- Toolformer: https://arxiv.org/abs/2302.04761
- ReAct: https://arxiv.org/abs/2210.03629

## Codex-Like Long-Horizon Coding Agent Implication

The current Codex-style "work for hours in the background" capability should be
understood as a system architecture, not only as a base-model property.

Plain-language decomposition:

```text
model brain:
  understands repo/task and proposes code actions

tool body:
  reads files, edits files, runs shell commands, runs tests

workspace memory:
  keeps stable task semantics, current findings, failed attempts, and pending
  TODOs without stuffing every raw token forever

sandbox:
  lets the agent try changes safely and produce verifiable evidence

human review gate:
  lets a person inspect diffs, logs, tests, and request revisions before merge
```

Local architecture lesson:

- Do not expect a 913M/1B from-scratch text model to become a 5-hour coding
  agent just because it can generate text.
- Long-horizon agent ability needs explicit training/eval for:
  1. repository navigation;
  2. file edit planning;
  3. test execution and failure interpretation;
  4. context compression/memory update;
  5. environment setup;
  6. partial progress recovery after interruptions;
  7. final diff/log/test evidence.
- For our model, the later agentic stage should use a structured trajectory:

  ```text
  task -> inspect -> edit -> run -> observe -> compact/update memory
       -> next inspect/edit/run -> final answer with evidence
  ```

- Promotion gate:
  run small local repo tasks first, then SWE-Bench-style tasks. Passing a
  single function-call JSON test is not evidence of Codex-like autonomy.

Primary-source anchors:

- OpenAI Codex product: https://openai.com/index/introducing-codex/
- OpenAI Codex app: https://openai.com/index/introducing-the-codex-app/
- GPT-5.2-Codex: https://openai.com/index/introducing-gpt-5-2-codex/
- Context as a Tool: https://arxiv.org/abs/2512.22087
- SWE-EVO: https://arxiv.org/abs/2512.18470
- SWE-Bench Pro: https://arxiv.org/abs/2509.16941
- SetupBench: https://arxiv.org/abs/2507.09063
- SWE-agent: https://arxiv.org/abs/2405.15793
- OpenHands: https://arxiv.org/abs/2407.16741
- SWE-bench: https://arxiv.org/abs/2310.06770

## Copy-Safe Checkpoint Discipline

2026-05-24 Stage93 update:

- Keep overwrite/resume checkpoints and local-transfer checkpoints as separate
  filenames.
- Resume files:
  - `last.pt`: full training state, including optimizer when enabled.
  - `last_model.pt`: model/verifier state without optimizer for lighter resume
    and evaluation.
- Copy-safe aliases:
  - `copy_last.pt`
  - `copy_last_model.pt`
- Plain-language rule:

  ```text
  last*.pt = the notebook still on the student's desk
  copy_last*.pt = the photocopy we can carry to another machine
  ```

- Implementation rule:
  write the checkpoint once to a hidden temporary file, atomically replace the
  resume file, then publish the copy alias by hardlink when possible. This
  avoids a second multi-GB `torch.save` and prevents local copy jobs from seeing
  half-written checkpoints.
- For already-running trainers that cannot load edited Python code, run
  `scripts/550_watch_copy_safe_checkpoints.sh` with `OUT_DIR`, `LOG_FILE`, and
  `TARGET_STEP` so the alias follows stable checkpoint boundaries until the run
  finishes.
