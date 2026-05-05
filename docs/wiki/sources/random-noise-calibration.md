# Random Noise Calibration References

Status: source note, 2026-05-03.

Purpose: track the prior work behind random-noise warm-up, uncertainty
calibration, and metacognitive "do not overclaim" gates for QTRM.

## Primary Sources

| Source | Link | QTRM relevance |
| --- | --- | --- |
| Cheon and Paik, 2026, "Brain-inspired warm-up training with random noise for uncertainty calibration" | `https://www.nature.com/articles/s42256-026-01215-x` | Direct source for short random input + random label warm-up improving calibration, OOD confidence, and ECE. |
| Preprint, "Pretraining with random noise for uncertainty calibration" | `https://arxiv.org/abs/2412.17411` | Same line of work; states random initialization can create overconfident untrained networks and random noise/label pretraining moves confidence toward chance. |
| Official implementation | `https://github.com/cogilab/Random2` / `references/official/cogilab-random2` | Official code for the Nature Machine Intelligence 2026 paper. Use as the main implementation reference. |
| Cheon, Lee, Paik, NeurIPS 2024, "Pretraining with Random Noise for Fast and Robust Learning without Weight Transport" | `https://arxiv.org/abs/2405.16731` | Related random-noise pretraining result for faster learning, generalization, and task-agnostic adaptation. |
| NeurIPS 2024 proceedings | `https://proceedings.neurips.cc/paper_files/paper/2024/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html` | Peer-reviewed version of the 2024 random-noise pretraining line. |
| Official NeurIPS 2024 implementation | `https://github.com/cogilab/Random` / `references/official/cogilab-random` | Implementation reference for random-noise pretraining under feedback alignment. |

## LLM Uncertainty Adjacent Sources

| Source | Link | QTRM relevance |
| --- | --- | --- |
| Semantic Self-Distillation for Language Model Uncertainty, 2026 | `https://arxiv.org/abs/2602.04577` | Distills sampled semantic uncertainty into a lightweight student before generation. Useful for future prompt-level uncertainty heads. |
| Improving Semantic Uncertainty Quantification in Language Model QA via Token-Level Temperature Scaling, 2026 | `https://arxiv.org/abs/2604.07172` | Reminds us to measure both calibration and discrimination for semantic uncertainty. |
| Unified Uncertainty Calibration, 2023/2024 | `https://arxiv.org/abs/2310.01202` | Treats "I do not know" as more than a single reject threshold; useful for metacognitive gate design. |
| UnifiedUncertaintyCalibration implementation | `https://github.com/facebookresearch/UnifiedUncertaintyCalibration` | Code reference for uncertainty calibration experiments. Archived, but still useful for metrics and method structure. |

## QTRM Adaptation

The Nature paper's fine-tuning setup is the closest analogy to QTRM:

```text
pretrained backbone frozen during noise warm-up
new classifier/head updated on random input + random label
then normal downstream training resumes
```

QTRM mapping:

```text
frozen Qwen donor
-> random token/hidden-state warm-up
-> train only QTRM workspace/core/coda/readout/residual heads
-> normal raw-reasoning training resumes
```

This is not a direct proof that random-noise warm-up improves LLM reasoning.
It is a prior-backed initialization/calibration probe.

## Required Metrics

Random-noise warm-up is accepted only if it improves metacognitive calibration
without hiding failures:

```text
ECE / Brier / NLL on answerable cases
confidence on random-token and OOD prompts
UNKNOWN/abstain/selective-accuracy curves
donor-only vs QTRM-only vs fused QTRM
core-off and memory-off confidence ablations
```

Do not promote the result if calibration improves only after a threshold,
external verifier, or donor-only fusion hides the QTRM failure.

## Current Decision

Random-noise warm-up is a legitimate QTRM experiment because overconfidence is
now considered a raw-intelligence failure. It must remain scoped:

```text
canonical: QTRM trainable-path calibration probe
not canonical: proof of ASI, proof of donor-free language ability, or a
replacement for recursive-depth/memory gates
```

## Local QTRM Implementation

Implemented on 2026-05-03:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --noise-warmup-steps
  --noise-warmup-seq-len
  --noise-warmup-batch-size
  --noise-warmup-core-steps
  --noise-warmup-target-vocab-size
  --noise-warmup-final-ce-weight
  --noise-warmup-depth-ce-weight
  --noise-warmup-uniform-weight

scripts/197_run_pure_recursive_depth_supervised_train.sh
  NOISE_WARMUP_* environment variables

tests/test_pure_recursive_depth_supervised_train_script.py
  random batch shape/range
  matching-label loss preference
  runner argument plumbing

scripts/202_build_metacognitive_calibration_gate.py
  choice-score softmax confidence
  ECE / Brier / wrong-confidence summaries
  matched no-warmup versus warmup comparison

tests/test_metacognitive_calibration_gate_script.py
```

Smoke:

```text
run: qwen35_2b_noise_warmup_smoke_s001
checkpoint: runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
eval: runs/eval/qwen35_2b_noise_warmup_smoke_s001_depth_gate_1.jsonl
summary: docs/wiki/decisions/qwen35-2b-noise-warmup-smoke-s001-depth-gate-1-summary.json
status: plumbing accepted on 1 held-out case
```

This smoke only proves the warm-up path runs through the frozen Qwen donor and
updates QTRM trainable modules without breaking the raw-depth gate harness. It
does not prove a calibration or reasoning gain. The next real test must compare
matched checkpoints with and without warm-up on a held-out metacognitive
calibration gate.

Matched calibration smoke:

```text
baseline:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
  runs/eval/qwen35_2b_noise_warmup_matched_no_warmup_s001_depth_gate_1.jsonl

candidate:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
  runs/eval/qwen35_2b_noise_warmup_smoke_s001_depth_gate_1.jsonl

report:
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001.md
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001-summary.json

result:
  accuracy_delta: 0.0
  ece_delta: -0.003433
  brier_delta: -0.004509
  status: accepted as smoke only
```

The positive smoke is too small to promote. It just proves the evaluator can
see a calibration delta without using MemoryOS/retrieval or changing the Qwen
donor.

Matched held-out calibration gate for random-label CE warm-up:

```text
dataset:
  data/eval/metacognitive_calibration_heldout_40.jsonl
  40 cases: answerable arithmetic, answerable boolean, unknown missing,
  contradiction, and random-token OOD

baseline:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
  runs/eval/metacognitive_calibration_no_warmup_s001_40.jsonl

candidate:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
  runs/eval/metacognitive_calibration_noise_warmup_s001_40.jsonl

report:
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001.md
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001-summary.json

result:
  accuracy_delta: +0.000000
  ece_delta: +0.003483
  brier_delta: -0.000394
  wrong_confidence_delta: +0.000174
  status: rejected
```

Interpretation:
the current two-step random-noise warm-up is not accepted. It slightly improves
Brier and several UNKNOWN/OOD category slices, but global ECE and wrong-answer
confidence worsen. The failure is most visible on answerable boolean rows,
where Brier and ECE both degrade. The next experiment should not merely extend
the same smoke setting; it should add an explicit uncertainty/calibration
objective on held-out-like known/unknown/OOD hard cases and re-run the same
matched gate.

High-entropy random-noise warm-up follow-up:

```text
change:
  add --noise-warmup-uniform-weight
  set random-label CE weights to 0.0 in the smoke
  train QTRM modules to keep random-input logits closer to uniform/high entropy

checkpoint:
  runs/qwen35_2b_4090_noise_uniform_warmup_smoke_s001/last.pt

eval:
  runs/eval/metacognitive_calibration_noise_uniform_warmup_s001_40.jsonl

report:
  docs/wiki/decisions/noise-uniform-warmup-metacognitive-calibration-heldout40-s001.md
  docs/wiki/decisions/noise-uniform-warmup-metacognitive-calibration-heldout40-s001-summary.json

global result:
  accuracy_delta: +0.000000
  ece_delta: -0.001182
  brier_delta: -0.000063
  wrong_confidence_delta: -0.001426
```

The global numbers improve, but the stricter gate still rejects this checkpoint:
`qtrm_core_steps_8_no_evidence` and
`qtrm_core_steps_8_qtrm_only_no_evidence` both worsen on ECE and Brier. This
shows why donor-only/core-off rows cannot be allowed to hide failures in the
actual QTRM core-on path. The next experiment should target core-on and
QTRM-only calibration directly, likely with forced-choice known/unknown
calibration rows rather than only random-token warm-up.

Direct forced-choice calibration follow-up:

```text
train cases:
  data/filtered/metacognitive_calibration_train_40.jsonl
  start_index=100, no overlap with heldout40

preference rows:
  data/filtered/metacognitive_calibration_preferences_train.jsonl

checkpoint:
  runs/qwen35_2b_4090_metacog_forced_choice_s080/last.pt

eval:
  runs/eval/metacognitive_calibration_forced_choice_s080_40.jsonl

report:
  docs/wiki/decisions/metacog-forced-choice-s080-calibration-heldout40.md
  docs/wiki/decisions/metacog-forced-choice-s080-calibration-heldout40-summary.json

global result:
  accuracy_delta: -0.033333
  ece_delta: -0.038450
  brier_delta: +0.007056
  wrong_confidence_delta: -0.100895
  status: rejected
```

Interpretation:
direct known/unknown training successfully lowers overconfidence, but it also
drops accuracy and worsens Brier. The important split is mode-specific:
`qtrm_core_steps_8_low_donor_no_evidence` improves accuracy from 0.60 to 0.75,
but `qtrm_core_steps_8_no_evidence` and
`qtrm_core_steps_8_qtrm_only_no_evidence` drop from 0.60 to 0.425. This is not
yet a donor-independent QTRM metacognitive improvement. The next candidate
should preserve the no-warmup QTRM core policy while adding calibration, for
example teacher-depth KL from the no-warmup checkpoint plus known/unknown
preference loss.

Teacher-depth KL preservation follow-up:

```text
teacher checkpoint:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt

student init:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt

checkpoint:
  runs/qwen35_2b_4090_metacog_teacher_kl_s080_v2/last.pt

eval:
  runs/eval/metacognitive_calibration_teacher_kl_s080_v2_40.jsonl

report:
  docs/wiki/decisions/metacog-teacher-kl-s080-v2-calibration-heldout40.md
  docs/wiki/decisions/metacog-teacher-kl-s080-v2-calibration-heldout40-summary.json

global result:
  accuracy_delta: +0.000000
  ece_delta: +0.018560
  brier_delta: +0.005489
  wrong_confidence_delta: +0.007658
  status: rejected
```

Interpretation:
teacher-depth KL did preserve the QTRM-only/core-on answer policy:
`qtrm_core_steps_8_no_evidence` and
`qtrm_core_steps_8_qtrm_only_no_evidence` stayed at 0.60 accuracy, unlike the
direct forced-choice run which dropped to 0.425. It also slightly improved
QTRM-only ECE/Brier. However, the low-donor fused path worsened and global
ECE/Brier increased. This rejects the checkpoint but validates teacher-depth KL
as a useful preservation tool. The next candidate should reduce calibration
pressure on already-correct answerable rows and apply it primarily to
overconfident wrong/UNKNOWN rows, while keeping teacher KL active.

Unknown-only teacher-depth KL follow-up:

```text
preference rows:
  data/filtered/metacognitive_calibration_unknown_preferences_train.jsonl
  only expected_unknown=true rows

checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_s080/last.pt

eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_s080_40.jsonl

report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-s080-calibration-heldout40.md
  docs/wiki/decisions/metacog-unknown-teacher-kl-s080-calibration-heldout40-summary.json

global result:
  accuracy_delta: +0.000000
  ece_delta: +0.029138
  brier_delta: +0.011536
  wrong_confidence_delta: +0.013695
  status: rejected
```

Interpretation:
unknown-only selection improves the expected-unknown category slices, but still
worsens global, answerable, and low-donor fused calibration. This means row
selection alone is not enough. The next conservative candidate is lower update
strength or stronger teacher KL before adding another component.

Conservative unknown-only teacher-depth KL follow-up:

```text
checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt

eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_conservative_s040_40.jsonl

report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-calibration-heldout40.md
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-calibration-heldout40-summary.json

settings:
  steps: 40
  lr: 2.0e-6
  teacher_depth_kl_weight: 5.0
  all_depth_ce_weight: 0.10
  choice_margin_weight: 0.25

global result:
  accuracy_delta: +0.000000
  ece_delta: +0.006972
  brier_delta: +0.000674
  wrong_confidence_delta: +0.001773
  status: rejected
```

Interpretation:
the conservative setting nearly preserves the baseline and improves QTRM-only
ECE/Brier while keeping QTRM-only accuracy fixed at 0.60. It still rejects
because the low-donor fused path worsens. This points to a fusion-calibration
problem rather than only a pure QTRM-core calibration problem.

Profile-split follow-up:

```text
qtrm_core profile:
  report: docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-qtrm-core-gate.md
  status: accepted
  accuracy_delta: +0.000000
  ece_delta: -0.014239
  brier_delta: -0.000351

fused profile:
  report: docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-fused-gate.md
  status: rejected
  accuracy_delta: +0.000000
  ece_delta: +0.014450
  brier_delta: +0.002196
```

Interpretation:
profile filtering confirms that the conservative teacher-KL update gives a
narrow QTRM-core calibration gain, while donor/QTRM fusion remains the active
failure. The next experiment should calibrate the fusion coefficient or routing
policy directly, not only the recursive core.

Fusion-scale sweep support:

```text
scripts/192_eval_raw_intelligence.py now accepts:
  qtrm_core_steps_8_donor_scale_0p50_no_evidence
  qtrm_core_steps_8_qtrm_scale_0p75_donor_scale_0p50_no_evidence
```

This keeps the next test KISS: sweep fusion coefficients before adding a new
learned fusion module.

Smoke8 result:

```text
baseline eval:
  runs/eval/metacognitive_fusion_scale_sweep_no_warmup_s001_smoke8.jsonl

candidate eval:
  runs/eval/metacognitive_fusion_scale_sweep_unknown_teacher_kl_conservative_s040_smoke8.jsonl

report:
  docs/wiki/decisions/metacog-fusion-scale-sweep-conservative-s040-smoke8.md

global:
  accuracy_delta: +0.000000
  ece_delta: +0.000507
  brier_delta: +0.001005
  status: rejected

by donor scale:
  0p25 ECE delta: +0.000002
  0p50 ECE delta: +0.000007
  0p75 ECE delta: +0.000419
  1p0  ECE delta: +0.001601
```

Interpretation:
the 8-case smoke is too small for promotion, but it shows the conservative
teacher-KL checkpoint does not improve fusion calibration and its degradation
grows with higher donor scale. Run the same sweep on all 40 cases before adding
a learned fusion router.

Full-sweep blocker:

```text
runs -> /mnt/sdb1/ws-sky-data/qtrm-runs
mount: /mnt/sdb1 ext4 rw,noatime,emergency_ro

failed read:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
  OSError: [Errno 5] Input/output error
```

Do not interpret this as a model failure. The next valid full-sweep run needs a
healthy checkpoint path or local copies outside `/mnt/sdb1`.

Checkpoint-localization helper:

```bash
bash scripts/205_localize_metacog_checkpoints.sh
```

It copies the two required checkpoints to:

```text
/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_fusion_sweep
```

and verifies SHA-256 before printing the `BASELINE_CHECKPOINT=...` and
`CANDIDATE_CHECKPOINT=...` command for the fusion sweep runner. It currently
fails at source read with the same `/mnt/sdb1` I/O error, confirming the blocker
is the source artifact path.

Implemented next diagnostic:

```text
docs/wiki/decisions/donor-qtrm-conflict-gate-probe.md
```

This optional probe downscales QTRM residual logits on donor/QTRM top-token
conflict. It is a falsification tool for the fusion boundary, not a canonical
metacognition solution.

Telemetry update:

```text
scripts/192_eval_raw_intelligence.py:
  choice_scores[].donor_qtrm_conflict_gate_mean
  choice_scores[].donor_qtrm_conflict_gate_observations

scripts/202_build_metacognitive_calibration_gate.py:
  mean_predicted_conflict_gate
  mean_choice_conflict_gate
```

This lets the full-sweep report show whether the conflict gate actually
attenuated QTRM residuals on predicted choices.

Rebuild recovery path:

```bash
bash scripts/206_run_metacog_pair_rebuild.sh
```

This creates a new matched baseline/candidate pair under:

```text
/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild
```

Use `INIT_CHECKPOINT=/healthy/path/last.pt` when a healthy prior checkpoint is
available. If no prior checkpoint is readable, `ALLOW_RANDOM_INIT=1` creates a
fresh random-init baseline and a conservative teacher-KL candidate from that
baseline. That mode restores the evaluation loop only; it must not be described
as exact recovery of the old unreadable s001/conservative-s040 checkpoints.
