# Free Transformer Latent For QTRM

Date: 2026-05-12

Status: implemented scaffold, not accepted L4.

## Why

The current QTRM bottleneck is not raw teacher-forced CE. Several runs improve
gold-token logits, but greedy generation still collapses to repeated numeric
patterns such as `00000000` or `66666666`.

The Free Transformer paper is relevant because it explicitly argues that a
plain autoregressive decoder has to infer global latent decisions only through
previous generated tokens. That can send generation off track after early token
mistakes. QTRM currently has the same symptom: the recursive core can perturb
the answer path, but the answer decoder does not reliably carry a stable latent
decision through autoregressive rollout.

## Adopted idea

QTRM keeps the TRM/QTRM recurrent core as the primary reasoning core.

Free Transformer is added only as an answer-path latent conditioning scaffold:

```text
prompt tokens / donor states
-> QTRM workspace + mandatory recurrent core
-> answer-state recurrent loop
-> Free-Transformer-style latent conditioning
   training: posterior latent from the full training context
   inference: prior latent from the answer hidden state
   loss: KL/free-bits between posterior and prior
-> mandatory next-token decoder
-> LM logits
-> autoregressive text
```

This is not a side solver and it does not create a hidden answer channel. The
latent is only allowed to modulate hidden states before normal LM logits.

## Files

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/330_run_mixed_noncopy_lm_gate.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_free_transformer_latent_s040.yaml
scripts/332_run_free_transformer_latent_smoke.sh
```

The training script now uses two pressures:

```text
KL/free-bits:
  posterior latent must stay close enough to the inference prior.

final contrast:
  full LM logits must assign higher target log-prob than the same forward pass
  with Free Transformer latent conditioning disabled.
```

## Acceptance condition

This scaffold counts only if strict generation improves:

```text
full QTRM > donor-only
full QTRM > core_off
core_state_zero hurts
answer_recurrent_off hurts
answer_next_token_decoder_off hurts
free_latent_off hurts
free latent KL stays nonzero but bounded
greedy generation improves, not only teacher-forced logits
```

If KL goes to zero and generation is unchanged, the latent collapsed.
If KL grows while generation improves only in teacher-forced mode, posterior
latents are leaking training information and not transferring to inference.

## Quick Smoke 2026-05-12

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  STEPS=1 MAX_TARGET_TOKENS=2 SELF_ROLLOUT_WEIGHT=0.0 \
  OUT_BASE=local_eval/free_transformer_latent_contrast_smoke_quick \
  bash scripts/332_run_free_transformer_latent_smoke.sh
```

Compile and static scaffold tests passed.

Training produced a valid checkpoint at:

```text
local_eval/free_transformer_latent_contrast_smoke_quick/train_s1/last.pt
```

Key one-step training metrics:

```text
answer_free_transformer_latent_final_contrast: 1.1735
answer_free_transformer_latent_final_target_logp_delta: -1.0735
answer_free_transformer_latent_kl: 0.6436
answer_free_transformer_latent_kl_loss: 0.5936
answer_free_transformer_gate_mean: 1.0000
final_path_acc: 0.0000
```

Gate report:

```text
local_eval/free_transformer_latent_contrast_smoke_quick/gate_1case_s1/report.json
```

Decision:

```text
rejected_noncopy_lm_gate
full_generation_accuracy: 0.0
full_minus_answer_free_transformer_latent_off: 0.0
```

Interpretation:

```text
The scaffold is runnable and the new contrast path is active, but a 1-step run
does not prove causal greedy-generation improvement. This remains L2/L3
prerequisite repair for the latent-state-to-autoregressive-answer bottleneck,
not L4 promotion.
```

Environment note:

```text
/mnt/nvme0n1p2 was full during the first smoke attempt, so the verified smoke
was written under local_eval/ on the root filesystem.
```

## Stage-5 Diagnostics 2026-05-12

Runs:

```text
local_eval/free_transformer_latent_contrast_stage_s5_t2
local_eval/free_transformer_latent_contrast_stage_s5_t2/gate_1case_s5_norepeat1
local_eval/free_transformer_latent_contrast_stage_s5_t2_depth8
local_eval/free_transformer_latent_contrast_stage_s5_t2_depth8_selfroll
```

Results:

```text
S5 mixed depth:
  train final_path_acc reached 0.5 on the tiny slice.
  strict gate full_generation_accuracy=0.0.
  full completion: 66666666
  free_latent_off completion: 66666666

S5 no_repeat_ngram_size=1 diagnostic:
  strict gate full_generation_accuracy=0.0.
  full completion: 604: UNKNOWN5Answer1
  interpretation: repetition ban changes surface form but does not reveal the
  correct answer, so the bottleneck is not only repetition decoding.

S5 depth8-only:
  train/eval depth mismatch removed.
  train final_path_acc reached 0.5.
  strict gate full_generation_accuracy=0.0.
  full completion: 66666666
  free_latent_off completion: 66666666

S5 depth8-only + self-rollout:
  self_rollout_prefix_mismatch_rate remained 1.0.
  train final_path_acc reached 0.5.
  strict gate full_generation_accuracy=0.0.
  full completion: 66666666
  free_latent_off completion: 66666666

S5 depth8-only + skip-leading-whitespace:
  strict gate full_generation_accuracy=0.0.
  full completion: 60060066
  free_latent_off completion: 60060066
  rank probe:
    6 rank 1
    0 rank 1
    0 rank 1
    0 rank 2
    5 rank 8
    4 rank 5
    EOS rank 55

S3 target-token-8 from base + skip-leading-whitespace:
  strict gate full_generation_accuracy=0.0.
  full completion: 00000000
  rank probe:
    first 6 rank 2
    max rank 36

S3 staged T2 checkpoint -> target-token-8:
  strict gate full_generation_accuracy=0.0.
  full completion: 00000000

S3 staged T2 checkpoint -> target-token-8, later_token_weight=0.25:
  rank probe:
    first 6 rank 2
    max rank 59

S3 staged T2 step5 checkpoint -> target-token-4:
  source checkpoint:
    local_eval/free_transformer_latent_contrast_t2_depth8_skipws_saveevery_s5/train_s5/step_000005.pt
  rank probes:
    step_000001: 6 rank 2, 0 rank 1, 0 rank 1, 0 rank 1, 5 rank 9, 4 rank 6, EOS rank 73
    step_000002: 6 rank 2, 0 rank 1, 0 rank 1, 0 rank 1, 5 rank 9, 4 rank 6, EOS rank 101
    step_000003: 6 rank 2, 0 rank 1, 0 rank 1, 0 rank 1, 5 rank 10, 4 rank 6, EOS rank 124
```

Decision:

```text
Rejected as a current L4 repair. Free Transformer latent conditioning remains a
valid scaffold, but it is not yet causally useful in greedy generation.
The only clear improvement came from target alignment:
`--causal-prefix-skip-leading-whitespace-targets`.

Target-token coverage also exposed checkpoint instability. Extending from two
answer tokens to the full answer can break the already learned prefix. Future
runs must use validation-gated checkpoint selection (`SAVE_EVERY`) before any
longer training is interpreted.

The T2->T4 stage confirms the same issue at a smaller curriculum jump: the
first answer token regresses from rank 1 to rank 2, so Free Transformer latent
conditioning is not stabilizing the autoregressive prefix.

The next orthodox step is not to add another answer channel; it is to repair
the core-state-to-token synthesis path so the recurrent state can produce
stable answer-token logits under autoregressive rollout.
```
