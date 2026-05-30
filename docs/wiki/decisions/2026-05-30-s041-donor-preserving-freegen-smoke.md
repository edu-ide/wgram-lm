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

## S042 Follow-Up: Adaptive-Margin Conflict Gate

User follow-up:

```text
해결 해봐 그럼
```

Root-cause test:

```text
reports/s041_donor_preserving_freegen/s042_no_conflict_high_alpha_cfc_smoke8.jsonl
reports/s041_donor_preserving_freegen/s042_no_conflict_high_alpha_cfc_smoke8.summary.md
```

Finding:

| Mode | Exact |
| --- | ---: |
| `donor_only_no_evidence` | 2/8 |
| no-conflict `qtrm_scale=2, donor_scale=1`, depth2 | 3/8 |
| no-conflict `qtrm_scale=2, donor_scale=1`, depth4 | 3/8 |
| no-conflict `qtrm_scale=2, donor_scale=1`, depth8 | 3/8 |

This falsifies "donor 사용 자체가 QTRM reasoning을 불가능하게 만든다".  The
specific masking cause was the downscale-only conflict gate, which reduced QTRM
residuals whenever donor and QTRM disagreed.

Implemented repair:

```text
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/config.py
scripts/192_eval_raw_intelligence.py
tests/test_donor_qtrm_conflict_gate.py
```

New policy:

```text
donor_qtrm_conflict_gate_mode = adaptive_margin
if donor_top != qtrm_top:
  keep or boost QTRM when QTRM top-token margin >= donor margin + threshold
  otherwise downscale QTRM to the legacy conflict scale
```

Validation:

```text
PYTHONPATH=. python3 tests/test_donor_qtrm_conflict_gate.py
PYTHONPATH=. python3 -m py_compile src/qtrm_mm/qtrm_model.py src/qtrm_mm/config.py scripts/192_eval_raw_intelligence.py tests/test_donor_qtrm_conflict_gate.py
```

Adaptive CFC artifacts:

```text
reports/s041_donor_preserving_freegen/s042_adaptive_margin_cfc_smoke8.jsonl
reports/s041_donor_preserving_freegen/s042_adaptive_margin_cfc_smoke8.summary.md
```

Adaptive CFC result:

| Mode | Exact |
| --- | ---: |
| `donor_only_no_evidence` | 2/8 |
| `qtrm_core_off_no_evidence` | 0/8 |
| canonical QTRM depth2/4/8 | 2/8 each |
| adaptive donor-preserving `qtrm_scale=2, donor_scale=1`, depth2 | 3/8 |
| adaptive donor-preserving `qtrm_scale=2, donor_scale=1`, depth4 | 3/8 |
| adaptive donor-preserving `qtrm_scale=2, donor_scale=1`, depth8 | 3/8 |

Decision: the donor-preserving CFC masking bug is repaired.  The current
promotion recipe for candidate-discrimination probes is:

```text
donor_qtrm_conflict_gate: on
donor_qtrm_conflict_gate_mode: adaptive_margin
donor_qtrm_conflict_qtrm_scale: 0.25
qtrm_scale: 2.0
donor_scale: 1.0
depth sweep: 2, 4, 8
```

Free-generation follow-up artifacts:

```text
reports/s041_donor_preserving_freegen/s042_adaptive_margin_free_generation_smoke8.jsonl
reports/s041_donor_preserving_freegen/s042_adaptive_margin_free_generation_smoke8.summary.md
reports/s041_donor_preserving_freegen/s042_adaptive_margin_beam_generation_smoke8.jsonl
reports/s041_donor_preserving_freegen/s042_adaptive_margin_beam_generation_smoke8.summary.md
```

Free-generation result:

| Decoder | Best guided exact | Donor-only exact | Read |
| --- | ---: | ---: | --- |
| greedy generation | 2/8 | 2/8 | guided depth2 ties donor, does not beat it |
| beam generation, beam8 mean | 1/8 exact, 2/8 loose hit | 0/8 exact, 2/8 loose hit | beam does not unlock the answer path |

DGX UltraData rehearsal checkpoint check:

```text
/mnt/data4tb/qtrm_multimodal_memoryos/checkpoints/qwen35_2b_ultradata_rehearsal_sft/last.pt
generation_smoke8: hits=0/40
causal_forced_choice_smoke4: hits=2/20
```

Decision: S042 is not generation-ready.  It repairs the donor/QTRM CFC blend,
but free generation remains a training objective problem, not a decoding-only
problem.  The next accepted path must train the free-running donor mouth with
self-rollout repair, unlikelihood against observed collapse strings, and
first-answer-token / answer-boundary objectives.  Plain 40-step UltraData SFT is
not enough evidence that full UltraData volume alone will solve free generation.
