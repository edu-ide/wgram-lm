# Research-Driven Next Architecture

Date: 2026-04-30

Status: active architecture-debug ledger after the workspace-evidence
preference/repeatguard/counterfactual probes.

## Root Architecture Doubt Gate

This project must not keep stacking local fixes if the root architecture is
wrong. Before adding another loss, threshold, verifier, or decoding script,
write the following ledger:

```text
Root architecture claim:
Falsifier:
Simpler baseline:
Causal path under test:
Evidence that the path is actually used:
Local fixes already tried:
Replacement architecture if rejected:
Kill criterion:
```

Current QTRM root claims that must stay under suspicion:

| Claim | Falsifier | Simpler Baseline |
| --- | --- | --- |
| Latent workspace/core is causally useful. | `workspace_off`, `core_off`, or gate-off outputs match full model on held-out tasks. | Qwen donor-only or residual-only adapter. |
| Workspace evidence is used, not merely retrieved. | Correct evidence, shuffled evidence, and no evidence produce same answer. | External RAG/reranker without QTRM. |
| QTRM residual improves donor instead of damaging it. | Donor-only is more fluent/accurate or residual mainly changes format/repetition. | Frozen donor with decoding controls. |
| Verifier/reranker is an architecture improvement. | It only hides bad generations after the fact and does not improve held-out candidate selection. | Direct donor sampling plus external reranker. |
| Loop/latent reasoning is real computation. | More loops improve loss but not held-out behavior, or loop ablation does not change answers. | Non-loop adapter with same parameter budget. |

Escalation rule:
if two local fixes improve only in-sample metrics or fail causal ablation, stop
local tuning and propose a root-structure alternative. At least one candidate in
each research cycle must be a replacement architecture, not a patch.

Automated gate:
`scripts/148_build_root_architecture_gate.py` converts existing ablation JSONL
into an explicit `accepted`, `rejected`, or `inconclusive` decision. Use it
after every workspace/core/evidence ablation before starting another training
run.

Current gate:
`docs/wiki/decisions/root-architecture-causality-gate.md` is `rejected` on the
held-out workspace-evidence counterfactual run. The baseline is `0/4`, and
`workspace_memory_off` plus `core_context_off` keep the same completions as the
full residual path. That means the current checkpoint cannot support the claim
that latent workspace/core/evidence paths are causally necessary.

## Observed Failure

Failure:
MemoryOS retrieval finds the target evidence, but QTRM still generates wrong or
generic answers. Workspace/core ablations do not reduce score.

Evidence:

- `runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_counterfactual_32tok_trained_s050.jsonl`
- `docs/wiki/decisions/workspace-evidence-counterfactual-trained-ablation.md`
- quick gate: `retrieved_target_rate=1.0`, `target_recall_mean=0.9167`,
  `qtrm_residual_with_evidence=0/4`, `workspace_off=0/4`,
  `core_off=0/4`, `workspace_memory_off=0/4`
- completion identity still mostly matches when workspace memory/gates are
  disabled: `workspace_gate_off=4/4`, `workspace_memory_off=4/4`,
  `core_context_off=4/4`
- pair preference still passes: `preference_accuracy=0.909`,
  `margin_pass_rate=0.909`, `margin_mean=2.50`

Likely component:
The evidence path exists, but answer logits are not causally forced through
workspace-selected evidence. Static chosen/rejected preference pairs can pass
without teaching generation-time evidence use.

Alternative explanations:

- 11 counterfactual rows are too few.
- Qwen donor logits dominate the residual path.
- Diagnostic prompts without actual evidence are not representative.
- Reranker/eval retrieval may include noisy but target-containing contexts.

## Prior Work Checked

| Source | Mechanism | Local reference |
| --- | --- | --- |
| [Self-RAG](https://arxiv.org/abs/2310.11511), [repo](https://github.com/AkariAsai/self-rag) | retrieve/generate/critique reflection tokens; do not retrieve/use context blindly | `references/official/self-rag@1fcdc420e48f` |
| [CRAG](https://arxiv.org/abs/2401.15884), [repo](https://github.com/HuskyInSalt/CRAG) | retrieval evaluator and corrective action when retrieved docs are weak | `references/official/crag@de7c2961ae62` |
| [FiD](https://arxiv.org/abs/2007.0128), [repo](https://github.com/facebookresearch/fid) | retrieved passages are fused in decoder attention; reader scores can train retriever | `references/official/fid@fe769f30e371` |
| [RETRO](https://arxiv.org/abs/2112.04426) | chunked cross-attention to retrieved neighbors; retrieval becomes an architectural input | `references/official/retro-pytorch@d08661313f3a`, `references/official/labml-retro@33ab02281c2b` |
| [RAGChecker](https://arxiv.org/abs/2408.08067), [repo](https://github.com/amazon-science/RAGChecker) | separate retrieval and generation diagnostics | `references/official/ragchecker@6091f08c00e6` |
| [FaithfulRAG](https://arxiv.org/abs/2506.08938), [repo](https://github.com/XMUDeepLIT/Faithful-RAG) | fact-level conflict modeling between retrieved context and parametric knowledge | `references/official/faithful-rag@9181b1132f2f` |
| [SFR-RAG](https://arxiv.org/abs/2409.09916), [repo](https://github.com/SalesforceAIResearch/SFR-RAG) | instruction tuning for context-faithful, unanswerable, counterfactual RAG behavior | `references/official/sfr-rag@afc44fb51c12` |
| [Coconut](https://arxiv.org/abs/2412.06769), [repo](https://github.com/facebookresearch/coconut) | continuous latent thought via feeding hidden states back as inputs | `references/official/coconut@27273cb8cca4` |
| [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416) | looped transformers can simulate latent thought through effective depth | paper source only |
| [Ouro](https://arxiv.org/abs/2510.25741), [site](https://ouro-llm.github.io/) | looped LM pretraining with entropy-regularized depth allocation | official site says code coming soon |
| [TRM](https://arxiv.org/abs/2510.04871), [repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | tiny recursive reasoning with explicit iterative refinement | `references/official/tiny-recursive-models@c01103738605` |
| [SoftCoT](https://arxiv.org/abs/2502.12134), [repo](https://github.com/xuyige/SoftCoT) | soft thought tokens projected into LLM representation space | `references/official/softcot@fa7f537d1d0a` |

## Architecture Candidates

### 1. Evidence-Bottleneck Decoder

Limitation solved:
Retrieved evidence exists but the answer path bypasses it.

Prior mechanism:
FiD/RETRO make retrieved passages architectural attention inputs; Self-RAG,
CRAG, FaithfulRAG, and SFR-RAG add explicit evidence-use checks.

Architecture change:
Insert an answer bottleneck between workspace-selected evidence states and the
QTRM residual logits. For MemoryOS tasks, the residual should not freely write
answer logits from text context alone. It should first produce an evidence-use
state:

```text
workspace evidence states
-> evidence selector / verifier gate
-> answer bottleneck state
-> bounded residual logits
```

Training/eval change:

- keep counterfactual workspace contrastive loss;
- add verifier labels for `answer_supported`, `evidence_conflict`,
  `unanswerable`;
- add ablation where correct evidence, shuffled evidence, and no evidence must
  produce different answer logits;
- accept only if `workspace_memory_off` and shuffled-evidence modes drop.

Why this may work:
It makes evidence use measurable at the information path level rather than only
at the loss level.

Main risk:
Over-constraining the residual may hurt donor fluency or make the model answer
UNKNOWN too often.

Minimal prototype:
Add an `EvidenceBottleneckHead` that pools workspace tokens and gates the text
residual before final logits. Train it only on MemoryOS short-answer rows.

Reject if:
Pair eval improves but `workspace_memory_off` or shuffled-evidence ablations
still match full residual.

### 2. Verifier-Controlled RAG Sidecar

Limitation solved:
Generation chooses unsupported text even when retrieved target exists.

Prior mechanism:
Self-RAG critique tokens, CRAG evaluator, RAGChecker diagnostics,
FaithfulRAG fact-conflict checks.

Architecture change:
Keep the QTRM generator unchanged, but add a separate verifier/reranker that
scores candidate answers against retrieved evidence and suppresses unsupported
outputs.

Training/eval change:
Generate multiple candidates, score support against retrieved chunks, then
choose only supported answers or UNKNOWN.

Why this may work:
It is the fastest route to user-visible correctness.

Main risk:
It proves a better RAG pipeline, not a better internal latent-reasoning model.

Minimal prototype:
Add `scripts/127_eval_memory_verifier_rerank.py` over existing eval JSONL.

Reject if:
Verifier improves hit rate but QTRM component ablations still show no causal
workspace use.

### 3. Looped Workspace With Halting

Limitation solved:
QTRM needs more latent computation before writing answer logits.

Prior mechanism:
Coconut, looped latent thoughts, Ouro, TRM, SoftCoT.

Architecture change:
Turn the current fixed recursive core into a budgeted loop with learned
evidence-use/answer-ready halting.

Training/eval change:
Use depth sweeps and require later loops to increase answer support while early
halts are allowed for easy cases.

Why this may work:
It aligns with the long-term QTRM goal of latent state-update reasoning.

Main risk:
It does not by itself solve retrieved-evidence faithfulness; without the
evidence bottleneck it can still reason over the wrong signal.

Minimal prototype:
Train `core_halt` only after the evidence bottleneck has a measurable causal
drop.

Reject if:
More loops improve training loss but not retrieval-found answer accuracy.

## Recommended Next Step

Implement Candidate 1 first: **Evidence-Bottleneck Decoder**.

Reason:
The current failure is not lack of latent recursion. It is lack of causal
evidence use. Counterfactual loss alone still leaves `workspace_memory_off`
matching the full model. The next architecture must make the answer residual
depend on evidence-selected workspace states.

Acceptance gate:

- `qtrm_residual_with_evidence` improves over the current 0/4 quick gate;
- `workspace_memory_off` or shuffled-evidence drops below full residual;
- `core_context_off` changes completion identity on evidence-dependent rows;
- pair preference remains above 0.85 but is not treated as sufficient;
- no repeated-token collapse in diagnostic generation.

## Logical-Causal Bottleneck Implementation

Status:
implemented as the next falsifiable probe, not accepted as solved.

Additional prior sources added:

- [ProoFVer](https://aclanthology.org/2022.tacl-1.59/): proof-style
  support/refute natural-logic verification.
- [FOLIO](https://arxiv.org/abs/2209.00840): natural-language reasoning with
  first-order logic annotations.
- [LINC](https://arxiv.org/abs/2310.15164) and
  `references/official/linc@718dfe8fae34`: LLM-to-FOL prover pipeline.
- [NL2LOGIC](https://arxiv.org/abs/2602.13237) and
  `references/official/nl2logic@3d625c5b33ae`: AST-guided NL-to-FOL
  translation.
- [Fact Checking with Insufficient Evidence](https://aclanthology.org/2022.tacl-1.43/):
  insufficient evidence is a separate epistemic label.
- [RAGONITE](https://arxiv.org/abs/2412.10571): counterfactual attribution for
  evidence contribution.
- [CF-RAG](https://openreview.net/forum?id=9U51rOnGko): counterfactual query
  arbitration.
- [FaithfulRAG](https://arxiv.org/abs/2506.08938): fact-level conflict between
  parametric knowledge and retrieved context.

Implemented files:

- `docs/wiki/sources/logical-causal-trust.md`
- `docs/wiki/concepts/logical-causal-evidence-bottleneck.md`
- `configs/qwen35_2b_4090_logical_causal_bottleneck_s050.yaml`
- `scripts/127_run_logical_causal_bottleneck_train.sh`
- model/loss/data wiring for `support/refute/missing/causal_gate`

Architecture change:

```text
workspace evidence
-> recursive workspace state
-> support/refute/missing + causal evidence gate
-> evidence-gated bounded QTRM residual logits
-> donor logits + gated residual
```

Why this is the right next probe:
The previous counterfactual loss showed that pair margins can improve without
generation using the evidence. This implementation changes the residual path
itself, so `workspace_memory_off` and `evidence_bottleneck_off` are meaningful
causality ablations.

Reject if:
the new run still reports `qtrm_residual_with_evidence=0/4`, or if
`workspace_memory_off` and `evidence_bottleneck_off` produce the same answers as
the full model.

Quick run result:

- command: `HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 bash scripts/127_run_logical_causal_bottleneck_train.sh`;
- checkpoint: `runs/qwen35_2b_4090_logical_causal_bottleneck_s050/last.pt`;
- preference pair eval: `preference_accuracy=0.727`, `margin_pass_rate=0.727`,
  `margin_mean=1.88`;
- memory gate: `qtrm_residual_with_evidence=0/4`,
  `retrieved_target_rate=1.0`, `target_recall_mean=0.917`;
- root gate: `docs/wiki/decisions/logical-causal-bottleneck-root-gate.md` is
  `rejected`;
- `evidence_bottleneck_off` changes completions on some rows, but no critical
  ablation reduces a successful baseline because the full residual baseline is
  still `0/4`.

Conclusion:
the current logical-causal bottleneck is not enough. It creates some output
influence, but it does not teach the model to produce the short supported
answer. The next structural fix must combine a forced answer channel with a
causal workspace/evidence path, not just another scalar gate.

Guarded decoding follow-up:

- added MemoryOS eval controls:
  `--suppress-visible-reasoning-tokens` and `--no-repeat-ngram-size`;
- runner env:
  `SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 bash scripts/117_run_workspace_evidence_path_probe.sh`;
- output:
  `docs/wiki/decisions/logical-causal-bottleneck-guarded-root-gate.md`.

Result:
guarded decoding is also `rejected` with the same `0/4` residual baseline. It
removes some surface degeneration pressure, but the model still copies evidence
or continues the prompt instead of extracting the short answer. Therefore the
remaining failure is not just visible `<think>` or n-gram repetition; it is an
answer-extraction / answer-channel training problem.

## Workspace-Evidence Path Supervised Run

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  MAX_CASES=4 SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 \
  SAVE_EVERY=200 bash scripts/118_run_workspace_evidence_path_train.sh
```

Result:

- checkpoint: `runs/qwen35_2b_4090_workspace_evidence_path_s050/last.pt`;
- eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_trained_s050.jsonl`;
- root gate:
  `docs/wiki/decisions/workspace-evidence-path-trained-root-gate.md`;
- residual quick score improved from `0/4` to `3/4`;
- donor-only remained `0/4`;
- residual-head-off dropped to `0/4`;
- coda-off dropped to `2/4`;
- but `workspace_off`, `core_off`, `workspace_memory_off`,
  `core_context_off`, and `evidence_bottleneck_off` all stayed `3/4` with
  `4/4` same completions.

Interpretation:
the supervised hidden-evidence run taught the residual/coda path a useful answer
style or dataset prior, but it still did not make the hidden workspace evidence
causally necessary. This is a hard rejection of "more of the same training" as
the next step.

Next architecture candidate:
replace the current "workspace evidence as optional side context" with a
**forced evidence-to-answer bottleneck**:

```text
visible question
+ hidden workspace evidence
-> evidence selector state
-> short-answer bottleneck state
-> answer-only decoder head
```

The next eval must include counterfactual workspace swaps: keep the visible
question fixed, swap the hidden evidence, and require the answer to change or
fall to `UNKNOWN`. Workspace-off and counterfactual-swap must fail if the full
model succeeds.

## Counterfactual Probe Result

Implemented:

- `loss_workspace_contrastive_weight`
- `workspace_counterfactual_text_states` donor encoding
- `scripts/125_build_workspace_counterfactual_preferences.py`
- `configs/qwen35_2b_4090_workspace_evidence_counterfactual_s050.yaml`
- `scripts/126_run_workspace_evidence_counterfactual_train.sh`

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/126_run_workspace_evidence_counterfactual_train.sh
```

Result:

- checkpoint: `runs/qwen35_2b_4090_workspace_evidence_counterfactual_s050/last.pt`
- step 300: no immediate repeated-token collapse in diagnostics, but answer
  text still generic or leaks `<think>` in eval;
- pair eval: `preference_accuracy=0.909`, `margin_pass_rate=0.909`,
  `margin_mean=2.50`;
- quick memory eval: `qtrm_residual_with_evidence=0/4`,
  `retrieved_target_rate=1.0`, `target_recall_mean=0.9167`;
- ablations: `workspace_off=0/4`, `core_off=0/4`,
  `workspace_memory_off=0/4`, `workspace_gate_off=0/4`.

Conclusion:
The counterfactual loss path is useful instrumentation but does not solve the
architecture failure. It supports the need for an explicit evidence bottleneck
or verifier-controlled answer gate.

## Workspace-Swap Gate Result

Implemented an eval-only counterfactual gate that keeps the visible question
fixed, swaps only hidden workspace evidence from another case, and requires
`UNKNOWN` when the swapped evidence does not contain the requested answer.

Artifacts:

- case builder: `scripts/build_workspace_counterfactual_eval_cases.py`;
- test: `tests/test_workspace_counterfactual_eval_cases.py`;
- cases: `data/eval/memory_reasoning_heldout_workspace_swap_4.jsonl`;
- eval:
  `runs/eval/memory_reasoning_heldout_workspace_swap_4_workspace_evidence_path_s050.jsonl`;
- root gate:
  `docs/wiki/decisions/workspace-counterfactual-swap-root-gate.md`.

Result:

- residual: `3/4`;
- workspace off: `3/4`;
- core off: `3/4`;
- core context off: `3/4`;
- workspace memory off: `3/4`;
- evidence bottleneck off: `3/4`;
- coda off: `4/4`;
- residual head off: `1/4`;
- critical same-completion rate: `4/4` for all critical off modes.

Conclusion:
the current architecture can learn an `UNKNOWN`-emitting answer style, but the
workspace/core/evidence-bottleneck paths are not causally necessary. This
rejects the current design as a hidden-evidence reasoner. More SFT,
counterfactual preference loss, or runtime repetition guards on this exact
information path are now low-priority.

Replacement architecture direction:

```text
visible question states
hidden workspace evidence states
-> forced evidence selector
-> short-answer latent bottleneck
-> answer-only decoder logits
```

Acceptance gate:
full model must beat donor-only and residual-head-off, and at least one of
`workspace_off`, `workspace_memory_off`, `core_off`, or
`evidence_bottleneck_off` must drop on both normal held-out and workspace-swap
evals.

## Answer-Bottleneck Repetition Stop

Failure:
the first answer-bottleneck run reached 300 steps but live diagnostics produced
degenerate repetitions such as `666666...` and `555555...`.

Root architecture hypothesis:
answer-bottleneck logits should only act when hidden workspace evidence exists.
If they act on a plain prompt without workspace memory, they become another
prompt-prior residual generator and can repeat.

Evidence:
`run_prompt_diagnostics` encodes only the visible diagnostic prompt and donor
states. It does not split MemoryOS evidence into `workspace_text_states`.
Therefore the diagnostic was testing a prompt-only path, not the intended
hidden workspace path.

Architecture change:

```text
if answer_bottleneck_enabled and answer_bottleneck_requires_workspace_memory:
    qtrm_residual_logits *= workspace_memory_present
```

This makes the hidden-evidence answer path silent when workspace memory is
absent. The donor remains available through `donor_logits_scale`.

Kill criterion:
if a workspace-injected eval still repeats after this change, do not continue
plain CE training. Move to a stricter output channel:

```text
workspace evidence
-> selector/support state
-> short-answer state
-> bounded answer-token decoder
-> stop/repeat governor
```

That output channel should train stop/repeat/quality heads on generated failures
and should allow QTRM residuals only while the answer governor is open.

## Answer-Bottleneck Causal Gate Result

Implemented a stricter follow-up to avoid re-opening the entire QTRM model:

- `trainable_param_policy=answer_bottleneck_evidence_only`;
- `configs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml`;
- `scripts/149_run_workspace_answer_bottleneck_causal_train.sh`;
- `loss_workspace_contrastive_weight=0.1`;
- `loss_logical_evidence_weight=0.5`;
- `loss_causal_evidence_gate_weight=0.5`;
- `loss_repeat_unlikelihood_weight=0.02`.

Training was intentionally narrow:

```text
frozen base QTRM + donor
train answer_bottleneck_* modules
train evidence_support/refute/missing/causal_gate heads
keep residual bounded and workspace-required
```

Artifacts:

- checkpoint:
  `runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt`;
- pair eval:
  `runs/eval/workspace_answer_bottleneck_causal_pair_eval_s050.jsonl`;
- normal eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_answer_bottleneck_causal_32tok_s050.jsonl`;
- normal root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-root-gate.md`;
- swap eval:
  `runs/eval/memory_reasoning_heldout_workspace_swap_4_workspace_answer_bottleneck_causal_s050.jsonl`;
- swap root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-swap-root-gate.md`.

Results:

| Gate | Result |
| --- | --- |
| pair preference | `0.727` accuracy, margin mean `1.60` |
| training workspace margin | stayed near `0.0001` |
| normal residual | `1/4` |
| normal root gate | `accepted`, but only on one `UNKNOWN` success |
| swap residual | `0/4` |
| swap root gate | `rejected` |

Interpretation:
this candidate fixed the no-workspace repetition bug and made some
workspace/core/memory ablations matter, but it still failed the more important
workspace-swap test. A scalar evidence gate plus free-form residual logits is
not enough to learn reliable evidence-to-answer extraction from 11
counterfactual rows.

Updated root-architecture doubt:

```text
Root claim:
  QTRM can answer from hidden workspace evidence through bounded residual logits.

Falsifier now observed:
  When hidden workspace evidence is swapped away, the model does not reliably
  answer UNKNOWN, and evidence-bottleneck-off can outperform the full path.

Replacement architecture:
  Move from free-form residual generation to an explicit evidence reader:
  workspace evidence tokens -> selector/span/copy state -> answer-only decoder.
```

## Next Candidate: Evidence-Span Reader

The next architecture should stop asking the residual LM head to discover the
answer span implicitly. For MemoryOS facts, many answers are literal spans in
retrieved evidence. That gives a sharper training signal:

```text
visible question states
hidden workspace evidence token states
-> evidence selector / support verifier
-> start/end or answer-span reader over workspace evidence
-> answer-only copy/decoder channel
-> donor logits only for surface realization
```

Minimal prototype:

- build labels that mark the first answer alias span inside the hidden
  workspace evidence;
- train a span reader on donor-encoded workspace tokens;
- use `UNKNOWN` as a separate no-span class;
- only allow the generative residual to realize the selected span, not invent a
  new answer;
- evaluate with the same normal and workspace-swap root gates.

Acceptance gate:

- normal held-out residual must exceed `1/4`;
- workspace-swap residual must answer `UNKNOWN` when swapped evidence lacks the
  requested answer;
- `workspace_memory_off` and span-reader-off must drop;
- the model must not regain correctness only through `evidence_bottleneck_off`.

Reject if:
the span reader selects no span or the generated answer still ignores the
selected span after two small probes. In that case the next fallback is a
verifier-controlled RAG sidecar, which is less pure as latent reasoning but
more likely to produce user-visible correctness.

## 2026 Memory-Architecture Update

Source ledger:
`docs/wiki/sources/2026-memory-context-architecture.md`.

Correction:
the current workspace split must not be defended as "what latest LLMs do" in a
literal sense. Commercial long-context systems expose a visible context window
and tool/retrieval interfaces. Research memory LMs separate memory capacity
from reasoning compute, but the memory read is still conditioned by the
question or task. QTRM should therefore evolve from a workspace-only sidecar
into a prompt-conditioned memory reader.

Updated root claim:

```text
QTRM uses Qwen donor states for linguistic competence, but answer correctness
on MemoryOS tasks must flow through a prompt-conditioned hidden-evidence
reader, not through free-form residual logits over an unconditioned workspace.
```

New implemented scaffold:

- source notes: `docs/wiki/sources/2026-memory-context-architecture.md`;
- dataset builder: `scripts/build_evidence_span_reader_dataset.py`;
- dataset: `data/filtered/memory_reasoning_synth_span_reader.jsonl`;
- model flag: `model.evidence_span_reader_enabled`;
- train loss: `train.loss_evidence_span_reader_weight`;
- train policy: `evidence_span_reader_only`;
- config: `configs/qwen35_2b_4090_evidence_span_reader_s050.yaml`;
- runner: `scripts/150_run_evidence_span_reader_train.sh`.
- ablation mode: `qtrm_evidence_span_reader_off_with_evidence`.

Current generated span-label stats:

```text
rows=432
found=288
no_answer=144
missing=0
```

Updated architecture:

```text
visible prompt/question states
-> query projection

hidden workspace evidence token states
-> prompt-conditioned start/end span scores
-> no-answer/UNKNOWN score
-> selected evidence span
-> answer-only channel and bounded donor residual
```

Why this replaces the previous local fixes:
the failed answer-bottleneck run made some gates matter, but the free-form
residual still did not reliably extract the answer from swapped hidden
evidence. The span reader puts the requested answer location directly on the
loss surface, making workspace use easier to test causally.

Span-reader probe result:

```text
runner: scripts/150_run_evidence_span_reader_train.sh
init:   runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt
out:    runs/qwen35_2b_4090_evidence_span_reader_s050/last.pt

trainable params: 1,837,057
evidence_span_reader loss: 10.3774 -> 0.5896
start accuracy: 0.0000 -> 1.0000
end accuracy:   0.0000 -> 1.0000
```

Interpretation:
the model can learn a prompt-conditioned hidden-evidence span reader on the
synthetic MemoryOS span task. This is a real architecture improvement over the
workspace-only/free-form residual path because the question now causally
conditions the evidence read. It is still a probe. QTRM has not yet solved
hidden-evidence answering until the selected span is wired into an answer-only
copy/decoder channel and normal/workspace-swap gates pass.

Next experiment:
connect the selected span to the answer channel, then run normal and
workspace-swap evals with `qtrm_evidence_span_reader_off_with_evidence`. Do not
claim QTRM has solved hidden-evidence reasoning until both gates pass.
