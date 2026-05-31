# Residual Ablation Plan

Decision:

QTRM should be trained and evaluated as a residual adapter over Qwen donor
logits, not as a replacement language model head.

## Fixed Architecture Assumption

- Qwen donor logits are the base language policy.
- QTRM contributes a small residual through `qtrm_logits_scale`.
- `LatentWorkspace` uses Perceiver/OpenFlamingo-style repeated latent layers.
- JEPA/aux losses stay disabled for the first residual language stability
  ablations.
- In-Place TTT is a later donor-side adaptation axis, not part of this first
  residual run.

## Ablation Matrix

| Run | Config | Status | Purpose |
| --- | --- | --- | --- |
| donor passthrough | `configs/qwen35_2b_4090_donor_passthrough.yaml` | passed earlier | Qwen-only generation baseline |
| residual 0.05 | `configs/qwen35_2b_4090_donor_residual_s005_1000.yaml` | passed stability gate | first production candidate |
| residual 0.10 | `configs/qwen35_2b_4090_donor_residual_s010_1000.yaml` | passed stability gate | stronger residual comparison |

2026-05-01 guarded rerun:
`scripts/152_run_residual_language_stability_sweep.sh` rechecked the
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt` candidate at
`qtrm_logits_scale=0.00/0.05/0.10` with donor logits fixed at `1.0`,
visible-reasoning suppression, no-repeat-2, sentence stop, and 96 generated
tokens. All three scales reached `clean_rate=1.0` on the four-prompt smoke:

```text
runs/language_stability/residual_s010_safe_sweep_96/summary.jsonl
```

This proves fluency preservation for the guarded residual range, not residual
usefulness. The same smoke showed no donor argmax shifts and donor-only already
failed the math correctness prompt, so the next proof must be donor-only versus
QTRM-residual on evidence-sensitive tasks.

## First Run

Run `qtrm_logits_scale=0.05` for 1000 steps on
`data/filtered/qtrm_clean_pilot.jsonl` with live diagnostics:

- Korean prompt: `양자 컴퓨팅이란 무엇인가요?`
- English prompt: `Quantum entanglement means`
- math prompt: `Solve step by step: if x + 3 = 7, what is x?`

Result:

- Checkpoint: `runs/qwen35_2b_4090_donor_residual_s005_1000/last.pt`
- Saved after 1000 steps.
- Live diagnostics at 200/400/600/1000 steps stayed stable with no `Freeze`,
  `world of the world`, or single-token collapse.
- Independent reload evaluation also stayed stable.

Independent reload metrics:

| Prompt | Loss | PPL | Rank mean | Top1 | Repetition |
| --- | ---: | ---: | ---: | ---: | --- |
| `양자 컴퓨팅이란 무엇인가요?` | 2.2096 | 9.11 | 6.88 | 0.500 | `rep2=0.000`, `rep3=0.000` |
| `Quantum entanglement means` | 2.9217 | 18.57 | 16.40 | 0.400 | `rep2=0.048`, `rep3=0.016` |
| `Solve step by step: if x + 3 = 7, what is x?` | 2.2420 | 9.41 | 49.28 | 0.500 | `rep2=0.270`, `rep3=0.194` |

Interpretation:

- Korean and English prompts pass the language-stability gate.
- The math prompt answers correctly, then continues with another synthetic
  algebra example. That is not the previous collapse mode, but future math evals
  should use stop criteria or answer-only prompts to avoid counting template
  continuation as reasoning quality.
- `qtrm_logits_scale=0.05` is currently a safe residual scale.

## Go / No-Go Criteria

Promising:

- free generation remains close to donor baseline;
- repeated 2-gram/3-gram rates remain low;
- target-token rank does not regress materially from donor passthrough;
- QTRM residual changes top-k distribution without dominating it;
- no return of `world of the world`, `Freeze`, or dialogue-marker collapse.

Not promising:

- repetition rises while LM loss still decreases;
- QTRM residual overwhelms donor logits;
- Korean generation degrades relative to donor baseline;
- improvements only appear on one prompt with no metric movement.

## Next Step After This Run

Residual `0.10` was run for 1000 steps using the same dataset and diagnostics.
It preserved donor stability and did not show the previous repetition collapse.

Result:

- Checkpoint: `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Saved after 1000 steps.
- Live diagnostics at 200/400/600/800/1000 steps stayed stable.
- Independent reload evaluation also stayed stable.

Independent reload metrics:

| Prompt | Loss | PPL | Rank mean | Top1 | Repetition |
| --- | ---: | ---: | ---: | ---: | --- |
| `양자 컴퓨팅이란 무엇인가요?` | 2.1777 | 8.83 | 6.62 | 0.500 | `rep2=0.000`, `rep3=0.000` |
| `Quantum entanglement means` | 2.8766 | 17.75 | 15.40 | 0.400 | `rep2=0.032`, `rep3=0.016` |
| `Solve step by step: if x + 3 = 7, what is x?` | 2.2573 | 9.56 | 53.67 | 0.500 | `rep2=0.115`, `rep3=0.040` |

Interpretation:

- `qtrm_logits_scale=0.10` remains inside the current safe residual range.
- Compared with `0.05`, it slightly improves Korean/English loss and reduces
  English/math repetition in this small prompt set.
- This is still a stability result, not proof that QTRM adds reasoning over the
  donor. The next eval must target memory/retrieval tasks where donor-only and
  QTRM-residual can be separated.

After the 0.05 vs 0.10 scale boundary is known, train a small memory/retrieval
trace dataset and measure whether QTRM residual improves answers that require
local context or retrieved evidence. If 0.10 is unstable, keep the safe scale
near 0.05 or reduce to 0.02 before adding memory traces.

Current decision:

- Use `qtrm_logits_scale=0.10` as the next working residual candidate.
- Do not increase scale again until memory/retrieval evals exist.
- Next architecture axis: add a controlled memory benchmark before integrating
  Titans-style neural memory or In-Place TTT.

## Memory Retrieval Probe

Added a synthetic fixed-evidence probe:

- Cases: `data/eval/memory_retrieval_probe.jsonl`
- Script: `scripts/95_eval_memory_retrieval.py`
- Checkpoint:
  `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Command shape:
  `HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/95_eval_memory_retrieval.py --config configs/qwen35_2b_4090_donor_residual_s010_1000.yaml --checkpoint runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt --max-length 256 --max-new-tokens 24`

Result:

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| donor-only with evidence | 5/5 | 1.00 |
| QTRM residual with evidence | 5/5 | 1.00 |
| donor-only no evidence | 0/5 | 0.00 |
| QTRM residual no evidence | 0/5 | 0.00 |

Interpretation:

- External text evidence is usable by the current inference path.
- The positive behavior is already present in donor-only Qwen logits.
- QTRM residual `0.10` does not break evidence use in this small probe.
- This is not yet proof that QTRM learned a distinct memory/reasoning policy.
  The next eval must separate donor and QTRM by using either trained memory
  traces, residual-only diagnostics, harder distractor retrieval, or a task
  where the residual changes the answer distribution in a measurable way.

## Distractor And MemoryOS Probe

Added a harder distractor set and a real MemoryOS vector-index path:

- Cases: `data/eval/memory_retrieval_distractor_probe.jsonl`
- Index builder: `scripts/96_build_memory_retrieval_probe_index.py`
- Index:
  `runs/eval/memory_retrieval_memoryos_harrier270m_index`
- Retrieval model: `microsoft/harrier-oss-v1-270m`
- Eval output:
  `runs/eval/memory_retrieval_memoryos_harrier270m_24tok.jsonl`

Result with `--evidence-mode memoryos`, case-scoped top-k filtering, and
`max_new_tokens=24`:

| Mode | Retrieval target | Answer hits |
| --- | ---: | ---: |
| donor-only with MemoryOS evidence | 5/5 | 5/5 |
| QTRM residual with MemoryOS evidence | 5/5 | 5/5 |

Interpretation:

- The actual MemoryOS path now works end to end:
  Harrier embedding index -> FAISS retrieval -> MemoryOS evidence prompt ->
  Qwen donor/QTRM generation.
- The behavior is still not QTRM-specific because donor-only also scores 5/5.
- First-step residual-shift telemetry shows QTRM residual changes logits
  measurably (`max_abs_delta` around `1.50` to `1.75` on this probe), but only
  one of five QTRM records changed the first-token argmax. That is residual
  influence, not yet proven useful reasoning.

## Reranked MemoryOS Probe

Added reranking to the MemoryOS evidence path:

- Reranker module: `src/wgram_lm/memoryos/rerank.py`
- Eval flags:
  `--retrieve-top-n 20 --retrieval-top-k 3 --rerank-backend cross_encoder --reranker-model-id Qwen/Qwen3-Reranker-0.6B`
- Output:
  `runs/eval/memory_retrieval_memoryos_qwen3_rerank_24tok.jsonl`

Result:

| Mode | Retrieval target | Answer hits |
| --- | ---: | ---: |
| donor-only with Qwen3-reranked MemoryOS evidence | 5/5 | 5/5 |
| QTRM residual with Qwen3-reranked MemoryOS evidence | 5/5 | 5/5 |

Rerank behavior:

- `code-vx913`: target score `10.125`, distractors `4.625`, `1.25`
- `korean-lumina17`: target score `11.0`, distractors `7.0`, `-8.125`
- `date-marigold`: target score `10.438`, distractors `5.375`, `-12.125`
- `phrase-jade-circuit`: target score `9.062`, distractors `4.938`, `-3.312`
- `korean-northstar42`: target score `12.312`, distractors `5.75`, `-8.938`

Interpretation:

- Harrier top-20 plus Qwen3-Reranker top-3 works end to end.
- Qwen3-Reranker cleanly separates target from distractors on this synthetic
  multilingual probe.
- The final answer result still matches donor-only, so the next benchmark must
  be harder: conflicting evidence, multi-hop memory traces, temporal priority,
  or missing/directly contradicted distractors.

## Hard Memory Reasoning Probe

Added a harder synthetic reasoning probe:

- Cases: `data/eval/memory_reasoning_probe.jsonl`
- Current case count: 9.
- Categories: temporal conflict, authority conflict, English multi-hop, Korean
  multi-hop, Korean temporal conflict, negative missing-answer reasoning,
  Korean negative missing-answer reasoning, Korean authority conflict, and
  temporal missing-current-state reasoning.
- Index: `runs/eval/memory_reasoning_harrier270m_index`
- Eval output: `runs/eval/memory_reasoning_qwen3_rerank_32tok_expanded.jsonl`
- Eval shape: Harrier top-20 -> Qwen3-Reranker-0.6B -> top-5 evidence ->
  generation.
- Metrics now summarize by mode, category, and task family. Task families are
  inferred as `abstention`, `conflict`, `multi_hop`, or the raw category when
  no known family applies.

Result:

| Mode | Answer hits | All targets retrieved | Target recall |
| --- | ---: | ---: | ---: |
| donor-only with reranked MemoryOS evidence | 6/9 | 9/9 | 1.00 |
| QTRM residual with reranked MemoryOS evidence | 6/9 | 9/9 | 1.00 |

Task-family result:

| Task family | Answer hits | All targets retrieved | Target recall |
| --- | ---: | ---: | ---: |
| conflict | 8/8 | 8/8 | 1.00 |
| multi_hop | 4/4 | 4/4 | 1.00 |
| abstention | 0/6 | 6/6 | 1.00 |

Key observations:

- Retrieval/reranking is not the bottleneck: every case retrieved all target
  evidence after reranking.
- Both donor-only and QTRM residual solved temporal, authority, and multi-hop
  copy/follow cases.
- Both failed every abstention case. The model answered with nearby retrieved
  distractors such as `Polaris-42`, `북극성-42`, or `Dr. Mina Vale` instead of
  returning `UNKNOWN`.
- QTRM residual again did not show a measurable advantage over donor-only.

Next implication:

- The next training/eval target should be abstention and contradiction handling,
  not another easier retrieval-copy benchmark.
- Add supervised memory traces where the correct behavior is `UNKNOWN`, “not in
  evidence”, or “newer/signed evidence overrides older/anonymous evidence”.
- Before any 100M-scale ingestion, rerun this small probe and require separate
  reporting for retrieval recall, rerank recall, answer accuracy, and
  abstention accuracy. The known failure mode is "retrieved enough evidence, but
  answered from a nearby distractor anyway."

## Memory Trace Fine-Tune

Added supervised MemoryOS traces:

- Trace builder: `scripts/99_build_memory_trace_data.py`
- Trace data: `data/filtered/memory_abstention_traces.jsonl`
- Rows: 27 (`target`, `all`, and `lexical` variants for each of 9 cases)
- Training config: `configs/qwen35_2b_4090_memory_abstention_trace_s050.yaml`
- Init checkpoint:
  `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Fine-tuned checkpoint:
  `runs/qwen35_2b_4090_memory_abstention_trace_s050/last.pt`
- Eval output:
  `runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl`

Implementation notes:

- `JsonlTextVisionDataset` now masks `prompt` tokens for rows with
  `prompt`/`answer`, so supervised traces train only answer tokens instead of
  spending loss on copying the evidence prompt.
- `scripts/95_eval_memory_retrieval.py` supports `--qtrm-logits-scale` for
  scale sweeps.
- A scale sweep showed `0.1` was too weak, `1.0` over-abstained, and `0.5`
  was the stable training/eval point for this probe.

Result with Harrier top-20 -> Qwen3-Reranker-0.6B -> top-5 evidence:

| Mode | Answer hits | All targets retrieved | Target recall |
| --- | ---: | ---: | ---: |
| donor-only with reranked MemoryOS evidence | 5/9 | 9/9 | 1.00 |
| QTRM residual with trace fine-tune | 9/9 | 9/9 | 1.00 |

QTRM residual by task family:

| Task family | Answer hits | All targets retrieved | Target recall |
| --- | ---: | ---: | ---: |
| conflict | 4/4 | 4/4 | 1.00 |
| multi_hop | 2/2 | 2/2 | 1.00 |
| abstention | 3/3 | 3/3 | 1.00 |

Interpretation:

- The first small-scale blocker is fixed on the current hard probe: QTRM
  residual now improves over donor-only specifically when evidence is retrieved
  but answer selection requires abstention or conflict handling.
- This is still a 9-case synthetic probe. Do not treat it as a general
  MemoryOS solution until the trace set is expanded and held-out abstention
  cases are added.

## Held-Out MemoryOS Generalization Gate

Added a separate held-out reasoning probe:

- Held-out cases: `data/eval/memory_reasoning_heldout_probe.jsonl`
- Case count: 12
- Task families: 4 conflict, 4 multi-hop, 4 abstention
- Held-out index:
  `runs/eval/memory_reasoning_heldout_harrier270m_index`

Initial result before synthetic generalization traces:

| Mode | Answer hits | Accuracy |
| --- | ---: | ---: |
| donor-only with reranked MemoryOS evidence | 6/12 | 0.50 |
| QTRM residual with 9-case trace fine-tune | 4/12 | 0.33 |

Interpretation: the 9-case trace fine-tune overfit and did not generalize.

Then added a broader synthetic MemoryOS trace set:

- Case generator: `src/wgram_lm/training/synthetic_memory_cases.py`
- Builder: `scripts/100_build_synthetic_memory_cases.py`
- Training cases: `data/filtered/memory_reasoning_synth_train_cases.jsonl`
- Training traces: `data/filtered/memory_reasoning_synth_traces.jsonl`
- Trace rows: 288 balanced across conflict, multi-hop, and abstention
- Training config:
  `configs/qwen35_2b_4090_memory_synth_generalization_s050.yaml`
- Checkpoint:
  `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
- Eval output:
  `runs/eval/memory_reasoning_heldout_qwen3_rerank_32tok_synth_generalization_s050.jsonl`

Held-out result with Harrier top-30 -> Qwen3-Reranker-0.6B -> top-5 evidence:

| Mode | Answer hits | Accuracy |
| --- | ---: | ---: |
| donor-only with reranked MemoryOS evidence | 6/12 | 0.50 |
| QTRM residual with synthetic trace fine-tune | 9/12 | 0.75 |

Per-family result:

| Mode | Conflict | Multi-hop | Abstention |
| --- | ---: | ---: | ---: |
| donor-only | 3/4 | 3/4 | 0/4 |
| QTRM residual | 2/4 | 3/4 | 4/4 |

Remaining failures for QTRM residual:

- `heldout-temporal-korean-east-hangar`: expected `해돋이-31`, answered
  `달빛-10`.
- `heldout-authority-korean-comms-room`: expected `바람문-27`, answered
  `구름-27`.
- `heldout-multihop-project-ember-maintainer`: expected `Ilya Chen`, answered
  `Mira Sol`.

Decision:

- The synthetic trace expansion is useful: it improves held-out accuracy from
  4/12 to 9/12 and fixes abstention from 0/4 donor-only to 4/4 QTRM residual.
- The next blocker is conflict resolution under Korean temporal/authority
  evidence plus one English multi-hop all-target retrieval/selection failure.
- Do not scale MemoryOS to larger corpora until this held-out gate reaches at
  least 11/12 with no UNKNOWN repetition artifacts.

## Core-Halt Depth Gate

Added a direct halted versus full-depth comparison to the MemoryOS evaluator:

```bash
--core-halt-mode disabled  # force full recursive depth
--core-halt-mode enabled   # force learned early exit
```

Using `runs/qwen35_2b_4090_core_halt_probe/last.pt` with
`--qtrm-logits-scale 0.5`:

| Eval | Full-depth hits | Halted hits | Full steps | Halted steps | Hit changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9-case hard probe | 5/9 | 5/9 | 3 x 9 | 1 x 9 | 0 |
| 12-case held-out probe | 7/12 | 7/12 | 3 x 12 | 1 x 12 | 0 |

Interpretation:

- The learned halt head can reduce the recursive loop from 3 outer steps to 1
  on these MemoryOS probes without changing answer-level hit/miss labels.
- This does not prove step-1 latent reasoning is enough in general; it only
  shows that the current final answers are already determined by the first
  recursive state under this checkpoint and decoding setup.
- The core-halt checkpoint regresses from the earlier synthetic MemoryOS
  checkpoint's held-out `9/12` result to `7/12`. The next training gate should
  combine MemoryOS trace preservation with halt supervision instead of training
  the halt probe on clean-pilot LM text alone.

## MemoryOS-Preserving Halt-Only Gate

Added a narrow training policy:

```yaml
train:
  trainable_param_policy: core_halt_only
```

This freezes every QTRM tensor except:

- `core.halt_head.weight`
- `core.halt_head.bias`

Using `configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml`, the run starts
from `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`, trains on
`data/filtered/memory_reasoning_synth_traces.jsonl`, and writes
`runs/qwen35_2b_4090_memory_halt_preserve_s050/last.pt`.

Result with Harrier retrieval, Qwen3-Reranker-0.6B, top-5 evidence,
`--qtrm-logits-scale 0.5`, and `--no-logit-shift`:

| Eval | Full-depth hits | Halted hits | Full steps | Halted steps | Hit changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9-case hard probe | 6/9 | 6/9 | 2 x 9 | 1 x 9 | 0 |
| 12-case held-out probe | 9/12 | 9/12 | 2 x 12 | 1 x 12 | 0 |

Decision:

- This is the preferred halt training path for now.
- Early halt can be added without losing the held-out MemoryOS gain when the
  residual path is frozen.
- The remaining blocker is not halt; it is answer quality on the three held-out
  failures: two Korean conflict cases and one English multi-hop maintainer case.
