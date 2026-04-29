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
