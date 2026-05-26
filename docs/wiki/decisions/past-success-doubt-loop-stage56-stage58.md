# Past-Success Doubt Loop: Stage56/58 vs One-Body Language

Date: 2026-05-24

Status: diagnostic rule for future model-architecture decisions.

## Plain-Language Conclusion

The old local arithmetic runs did work, but they worked in a narrower exam.
They should make us curious, not complacent.

Stage56/58 was like giving the model many candidate answer sheets for a small
math game, then training a judge to pick the best one. That can beat a much
larger fluent model on that game because the answer space is tiny and the
search/verifier loop is doing real work.

The current from-scratch one-body language model is a different exam. It must
read natural text, think, and speak through its own normal language path. A high
selected/oracle score on the old synthetic gate does not prove this broader
ability.

## Evidence Snapshot

| Run | Exact Metric | Value | What It Proves | What It Does Not Prove |
| --- | --- | ---: | --- | --- |
| Stage56 PTRM K128 | `mean_selected_accuracy_oracle_depth` | 0.7682 | K-sampled PTRM candidates plus selector can find compact synthetic arithmetic answers. | Free language generation, multilingual ability, or one-body recurrent LM reasoning. |
| Stage56 PTRM K128 | `mean_oracle_accuracy` | 0.7747 | Candidate pool contains correct answers at high rate. | The model can select or verbalize arbitrary natural-language answers. |
| Stage58B PTRM K64 top3 | `mean_selected_accuracy_oracle_depth` | 0.9336 | Top-k candidate exposure plus verifier selection is very strong on the old small state-space. | General reasoning beyond the synthetic task contract. |
| Stage58B PTRM K64 top3 | `mean_oracle_accuracy` | 0.9401 | The generator/search side has high answer coverage. | That the normal LM head owns the answer path. |
| Stage94 raw-byte 82M local | `final_eval_loss` | 2.1387 | Raw byte language learning can drop held-out CE quickly on the local sample. | Good free generation or depth-scaled recurrent reasoning. |
| Stage94 BLT2 raw-teacher 1200 | `final_eval_loss` | 2.2999 | BLT-style latent byte path is trainable and improves over its initial loss. | That BLT already beats raw byte or solves the speaker path. |

## Reproducibility Status

Current status: replayable, but not fully sealed.

This is now backed by an executable manifest:

```text
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.json
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.md
```

Manifest verdict:

```text
overall_status = replayable_not_sealed
can_replay_any = true
all_fully_sealed = false
```

What is still present:

```text
Stage54B generator checkpoint:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt
  sha256 ef171d5b9e49abff9c3ca4f2458a2798dcc67363b1b10726a192f02fdce54eb0

Stage55 extractor checkpoint:
  /mnt/sdc1/tripleyoung/qtrm_eval/20260522_093200_LOCAL_STAGE55_stage54B_generator_tokenlocal_selector_seed103/best_token_local_register_extractor.pt
  sha256 cd8f3eea645ad04072242a4f1460d69281dcecd92bf6bb12fab48469949489fe

Stage56 summary:
  selected=0.7682, oracle=0.7747, packed=0.9935
  samples=128, eval_count=128, eval_seed=10042, eval_depths=4/6/8/10/12/14
  sha256 530f75ec62a7daebdb98a9cf29906ffd0a91dd8ba0eb19a10e7d64477d766a10

Stage58B summary:
  selected=0.9336, oracle=0.9401, packed=0.9935
  samples=64, candidate_topk_per_sample=3, eval_count=128, eval_seed=10042
  sha256 94162524b44fb78a8000ab37b77dbc28ebb71a56181f4ac8eeb3b8e77f461747
```

What is not fully sealed:

```text
1. No immutable code commit/hash is recorded in the run summaries.
2. The full materialized eval JSONL rows are not stored beside the summaries.
3. Stochastic high-level evaluation is enabled, so exact replay can depend on
   seed handling and GPU/kernel determinism.
4. The current worktree has evolved since the run.
```

Plain Korean verdict:

```text
재현성은 "다시 시도할 수 있는 수준"은 있다.
하지만 논문식 완전 재현 패키지라고 말하려면 부족하다.
완전 재현으로 승격하려면 code snapshot/hash, checkpoint sha256,
materialized eval rows, one-command rerun script, tolerance 비교 리포트를
같이 묶어야 한다.
```

Primary local evidence paths:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260522_101900_LOCAL_STAGE56_PTRM_evalonly_K128_scale1p0/summary.json
/mnt/sdc1/tripleyoung/qtrm_eval/20260522_113500_LOCAL_STAGE58B_PTRM_evalonly_K64_top3_stage54B_seed10042/summary.json
/tmp/20260524_STAGE94X_LOCAL_RAWBYTE_SEQ768_CAPACITY400.log
/tmp/20260524_STAGE94Y_LOCAL_BLT2_RAWTEACHER_DISTILL1200.log
```

Generated report artifacts:

```text
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.report.json
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.report.md
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.json
docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.md
```

The generated report now includes a `recommended_comparison_row`. For the
current evidence, it selects Stage58B as the strongest old success and says the
next restoration gate must log candidate coverage, selected-vs-oracle accuracy,
free generation samples, and recurrent/depth-off loss deltas on the same
heldout rows.

Launch guard:

```text
scripts/557_train_blt_d_prefixlm_dataio.py now exposes:
  --past-success-report-json
  --past-success-restoration-gate-json
  --past-success-preflight-min-steps
  --allow-missing-past-success-preflight
  --acknowledge-past-success-restoration-gap

For long `--decoder-latent-mode one_body` runs, the report is required unless
the run is explicitly marked as a diagnostic override. If the report says
`do_not_launch_long_run_until_restoration_gate_exists`, the launcher requires a
passing restoration-gate report from
`scripts/564_check_past_success_restoration_gate.py`, or an explicit diagnostic
override via `--acknowledge-past-success-restoration-gap`. A gate whose
`current_checkpoint_recommendation` is `do_not_promote_current_checkpoint` does
not satisfy the guard.
```

Rebuild command:

```bash
.venv/bin/python scripts/562_build_past_success_doubt_report.py \
  --old-ptrm Stage56_K128=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_101900_LOCAL_STAGE56_PTRM_evalonly_K128_scale1p0/summary.json \
  --old-ptrm Stage58B_K64_top3=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_113500_LOCAL_STAGE58B_PTRM_evalonly_K64_top3_stage54B_seed10042/summary.json \
  --current-language Stage94_raw_byte_82M=/tmp/20260524_STAGE94X_LOCAL_RAWBYTE_SEQ768_CAPACITY400.log \
  --current-language Stage94_BLT2_rawteacher_1200=/tmp/20260524_STAGE94Y_LOCAL_BLT2_RAWTEACHER_DISTILL1200.log \
  --out-json docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.report.json \
  --out-md docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.report.md

.venv/bin/python scripts/563_build_past_success_repro_manifest.py \
  --stage Stage56_K128=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_101900_LOCAL_STAGE56_PTRM_evalonly_K128_scale1p0/summary.json \
  --stage Stage58B_K64_top3=/mnt/sdc1/tripleyoung/qtrm_eval/20260522_113500_LOCAL_STAGE58B_PTRM_evalonly_K64_top3_stage54B_seed10042/summary.json \
  --out-json docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.json \
  --out-md docs/wiki/decisions/past-success-doubt-loop-stage56-stage58.repro_manifest.md

.venv/bin/python scripts/564_check_past_success_restoration_gate.py \
  --teacher-forced-report PATH_TO_LANGUAGE_LOSS_JSON \
  --generation-report PATH_TO_GENERATION_GATE_JSON \
  --depth-report PATH_TO_DEPTH_OR_RECURRENT_ABLATION_JSON \
  --search-report PATH_TO_SELECTED_ORACLE_JSON \
  --require-search-split \
  --out-json PATH_TO_RESTORATION_GATE_JSON \
  --out-md PATH_TO_RESTORATION_GATE_MD
```

## Why This Is Not A Contradiction

"Wrong attractor convergence" and "Stage56/58 reached 78-93%" can both be true.

The old run's success attractor was a compact synthetic answer state. The model
could explore several candidate states and a verifier could choose among them.
That is closer to solving a small puzzle with many attempts.

The current one-body language run asks for a broader attractor: the same state
must become a fluent next-token distribution. If depth residual shrinks but
held-out loss or generation does not improve, the model may be settling into a
stable but unhelpful thought state.

In Korean:

```text
예전 성공은 "작은 문제집에서 후보를 많이 내고 잘 고른 성공"이다.
지금 실패는 "읽고, 생각하고, 말하는 한 몸의 언어 경로가 아직 정답
상태로 수렴하지 못하는 실패"다.
```

## Causal Ingredient To Preserve

The useful part of Stage56/58 is not the old typed-register machinery itself.
The useful ingredients are:

```text
1. Diverse candidate generation.
2. A verifier that sees enough of the completed candidate to judge it.
3. A selected answer path whose metric is separated from oracle coverage.
4. Logs that distinguish selected accuracy, oracle accuracy, packed-state
   accuracy, teacher-forced loss, and free generation.
```

The parts not to copy into the main architecture:

```text
1. Oracle-only selection as a claimed model ability.
2. Task-specific synthetic heads as general LM reasoning.
3. A separate side register that the normal speaker can ignore.
4. Any metric that hides whether the LM head can actually speak the answer.
```

## Required Future Comparison Row

Before a new failed run inspires a new architecture name, write this row:

| Old success | Exact metric | Causal ingredient | Missing in current run | Smallest restoration test |
| --- | --- | --- | --- | --- |
| Example: Stage58B | selected 0.9336, oracle 0.9401 | candidate diversity plus verifier over completed answers | one-body language path lacks useful depth/generation gain | K-scaling candidate coverage, selected-vs-oracle split, free-generation sample gate |

If this row cannot be filled, do not launch a long local or DGX run.

## Immediate Implication

The next serious one-body language experiment should not merely lower CE. It
must log all of these at the same time:

```text
candidate coverage / oracle accuracy if search is used
selected accuracy if verifier selection is used
teacher-forced heldout loss
free generation samples
first-response-token top-k/rank
repetition and EOS/special-token rates
depth-off or recurrent-core-off ablation
```

Only the combination can tell whether we kept the real Stage56/58 success
condition while moving to a general language path.

Executable gate:

```text
scripts/564_check_past_success_restoration_gate.py
```

This checker does not claim the model is good. It checks that the experiment is
observable enough to avoid fooling ourselves: teacher-forced loss, free
generation samples, first-response-token stats, repetition/EOS rates,
selected-vs-oracle split when search is used, and depth/recurrent ablation must
all be visible before a long one-body run is treated as serious.

Current Stage99I local one-body result:

```text
teacher-forced loss: 6.3601 -> 2.5603
first response accuracy: 1.0000 on 128 rows
free generation exact: 0/8
ended with EOS: 0/8
depth residual probe: rejected; best loss at depth 2, not deepest
restoration gate: observable but current checkpoint is do_not_promote
```
