# S041 Donor-Preserving Free Generation Smoke

Status: active design direction, negative promotion smoke, 2026-05-30.

## Purpose

Test whether the S040 rank-8 LoRA checkpoint can recover free generation by
keeping Qwen donor logits as the fluent language path and adding a bounded QTRM
residual under a donor/QTRM conflict gate.

## Research Basis

Source page:

```text
docs/wiki/sources/2026-donor-preserving-looplm-freegen-repair.md
```

Core priors:

- Relaxed Recursive Transformers supports loop/depth LoRA as a legitimate
  parameter-efficient recurrent-depth relaxation.
- LoopUS/Ouro/LoopFormer/Parcae all point to trained latent loops with stable
  input injection, trajectory/depth conditioning, and bounded recurrence.
- ReFT/Proxy-Tuning/ThinkLogit support preserving the frozen donor language path
  and injecting a small representation/logit delta.
- Unlikelihood, scheduled sampling, and on-policy distillation address the
  observed teacher-forcing/free-generation gap.

## Local Smoke Contract

Runner:

```text
scripts/262_run_s041_donor_preserving_freegen_sweep.sh
```

Checkpoint:

```text
/mnt/nvme0n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_loop_joint_decoder_s040_from_selfrollout/last.pt
```

Eval settings:

```text
scoring: generation
cases: first 8 rows of data/eval/pure_recursive_reasoning_heldout_72.jsonl
depths: 2, 4, 8
donor-preserving guided modes: donor_scale=1.0, qtrm_scale in {0.25, 0.5, 1.0}
donor_qtrm_conflict_gate: on
conflict qtrm scale: 0.25
no_repeat_ngram_size: 2
```

Artifacts:

```text
reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.jsonl
reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.summary.json
reports/s041_donor_preserving_freegen/s041_conflict_gated_free_generation_smoke8.summary.md
```

## Result

Free-generation aggregate:

```text
records: 112
modes: 14
hits: 23/112
```

Free-generation exact-match summary:

| Mode family | Exact | Read |
| --- | ---: | --- |
| `donor_only_no_evidence` | 2/8 | donor mouth remains fluent but weak |
| `qtrm_core_off_no_evidence` | 0/8 | no QTRM/donor contract |
| `qtrm_core_steps_{2,4,8}_no_evidence` | 0/8 exact | QTRM-only renderer still collapses |
| `depth {2,4,8} + donor_scale=1.0 + qtrm_scale {0.25,0.5,1.0}` | 2/8 each | donor fluency preserved, no donor-beating gain |

The guided modes remove the most severe QTRM-only loop strings in several rows,
but the best exact score ties donor-only rather than beating it.

## Follow-Up CFC Reasoning Smoke

User challenge:

```text
그럼 이제 추론 성능 올라가는지 테스트 다시해야되는거 아니야?
```

Follow-up artifact:

```text
reports/s041_donor_preserving_freegen/s041_conflict_gated_cfc_reasoning_smoke8.jsonl
reports/s041_donor_preserving_freegen/s041_conflict_gated_cfc_reasoning_smoke8.summary.md
```

Causal forced-choice aggregate:

```text
records: 112
modes: 14
hits: 29/112
```

CFC exact-match summary:

| Mode family | Exact | Read |
| --- | ---: | --- |
| `donor_only_no_evidence` | 2/8 | donor baseline |
| `qtrm_core_off_no_evidence` | 0/8 | no answer path |
| canonical `qtrm_core_steps_{2,4,8}_no_evidence` | 3/8 each | narrow reasoning/candidate-discrimination gain |
| donor-preserving `donor_scale=1.0` guided modes | 2/8 each | donor blend masks the extra QTRM CFC gain |

Interpretation: CFC does show a small reasoning lift for the canonical QTRM
path over donor-only, but the S041 donor-preserving inference-time mix does not
preserve that lift.  The next training objective must teach the donor-preserving
gate when to let the QTRM delta override donor bias.

## Decision

Do not promote S041 as solved.

The smoke supports the donor-preserving hypothesis as the right path to train,
but it rejects inference-only alpha/conflict-gate sweeping as enough to open
free generation or to preserve the narrow CFC reasoning gain.

## UltraData Implication

Training on all UltraData should help coverage, instruction following, and
general answer formatting only if the training objective is changed.  A plain
teacher-forced SFT run risks strengthening forced-choice/rank signals while
leaving free generation collapsed.

The next UltraData run should therefore use:

```text
donor_logits as the base mouth
bounded QTRM delta/gate as the intervention
response-only CE
first-answer-token margin on donor-wrong rows
donor-correct KL/margin preservation
unlikelihood for Answer:Answer / bang loops / numeric attractors
self-rollout repair on generated prefixes
depth and qtrm/donor scale sweeps as the promotion gate
```

Promotion requires free-generation exact improvement over donor-only plus
causal drops under core/delta/gate ablations.
