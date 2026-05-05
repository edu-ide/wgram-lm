# QTRM Terminology

Status: working glossary, 2026-04-30.

This page fixes the vocabulary used in the QTRM wiki. The main rule is that
architecture terms must map to code paths, telemetry, and ablations. If a term
cannot be tested, it should be treated as a hypothesis, not a result.

## Core Terms

| Term | Meaning in QTRM | Not the same as | Required evidence |
| --- | --- | --- | --- |
| Donor | A frozen source LLM that provides tokenizer behavior, hidden states, and optionally base logits. Current donor: Qwen3.5-2B. | A trainable QTRM component. | Donor-only baseline and donor+QTRM comparison. |
| Donor-backed residual adapter | QTRM reads donor hidden states and adds bounded residual logits on top of donor logits. | LoRA, because QTRM does not edit donor weights. | Fluency preserved while held-out task score improves. |
| Cognitive sidecar | The trainable QTRM path around the donor: workspace, recursive core, coda, residual head, gates, and telemetry. | A standalone language model. | Component ablations show which sidecar part caused a gain. |
| QTRM model | Tokenizer/donor/QTRM forward path from canonical token stream to logits/answer channel. | MemoryOS retrieval, reranking, tools, or history store. | Plain-prompt donor/QTRM eval works without MemoryOS. |
| QTRM runtime | The system around the model: optional MemoryOS retrieval, rerank, context compiler, tools, history, eval harness. | The model architecture itself. | Runtime steps are logged separately from model forward telemetry. |
| SSOT context compiler | Runtime step that turns user prompt, retrieved evidence, tool results, and memory records into one canonical token stream before the model call. | A second hidden prompt or independent workspace reality. | Evidence masks are token-aligned annotations over the same stream. |
| LatentWorkspace | Learned latent slots that compress and route prompt/donor/evidence states before the recursive core. | Persistent long-term memory. | `workspace_off` or `workspace_memory_off` causes a measurable drop. |
| Workspace-only evidence | Retrieved evidence encoded separately and injected only into the workspace path, not appended to visible text. This is now an ablation/probe mode. | The canonical QTRM user-facing architecture. | Evidence helps only when workspace memory is enabled. |
| Gated workspace memory | A gated latent update over workspace slots inspired by LM2/G-MemLLM-style memory lanes. | A faithful LM2 or G-MemLLM clone. | `workspace_gate_off` hurts a target metric and gate telemetry is nontrivial. |
| Recursive core | The repeated `z_l`/`z_h` update loop that performs latent-space computation over workspace states. | Proven TRM ACT or human reasoning. | Depth sweeps, `core_off`, halt telemetry, and teacher-depth probes. |
| Gated core context injection | An ablatable cross-attention route from prelude prompt/evidence context into the recursive core. | Leaking evidence into the visible text path. | `core_context_off` reduces score when prompt/evidence context matters. |
| Coda | The post-core block stack that lets text positions attend back to latent core prefix states before LM head projection. | The donor decoder. | Core-prefix ablations affect residual logits or answer score. |
| Donor-logit fusion | The final logit combination of donor base logits plus QTRM residual logits. | Replacing the donor. | Residual scale/gate telemetry and donor argmax-shift rate. |
| Donor annealing | A staged reduction of donor reliance after QTRM-only logits are trained and stable. | Immediately setting donor scale to zero. | QTRM-only/student loss, repetition metrics, and held-out eval pass. |
| Metacognitive calibration | QTRM's trainable state estimates whether to answer, search, defer, or abstain under uncertainty. | Donor confidence, threshold-only abstention, or an external verifier sidecar. | ECE/Brier/UNKNOWN/OOD/selective-accuracy gains that drop under the claimed core/memory ablation. |
| MemoryOS retrieval | Optional external search/rerank/evidence packaging before model forward, followed by SSOT context compilation. | QTRM model architecture, neural memory inside QTRM weights, or a second model input reality. | Retrieval recall, rerank quality, token-aligned source masks, and missing-answer controls. |
| Internalized context engineering | Moving some context selection/routing/compression from prompt assembly into trainable workspace/core memory paths. | End-to-end learned web search. | External retrieval is measured, then internal routes are ablated. |

## Reasoning Terms

### Pattern Imitation

Pattern imitation means the model learns surface correlations in training
answers or traces. It may output plausible answers, but the intermediate
workspace/core state is not proven to be causally necessary.

Symptoms:

- training loss falls but free generation repeats tokens;
- CoT-shaped data improves formatting but not held-out reasoning;
- disabling workspace/core does not change the answer;
- the model succeeds only when the answer is copied from visible context.

### Actual Reasoning, Operational Meaning

In QTRM, "actual reasoning" must be defined operationally, not metaphysically.
The strong claim is allowed only when:

```text
intermediate latent computation changes the final answer
and the change improves correctness
and disabling the claimed computation removes the gain
and the behavior transfers to held-out cases
```

Therefore the target is not merely:

```text
learn the text pattern of reasoning traces
```

The target is:

```text
learn a compact state-update process that searches, compares, preserves,
rejects, halts, or asks for more evidence before changing the donor policy
```

This includes metacognition: the model must learn when its own state is
underdetermined, contradictory, out-of-distribution, or too weak to justify a
confident answer. Overconfidence is therefore a raw-intelligence failure, not
only a safety or formatting problem.

This is why QTRM needs workspace/core ablations, depth sweeps, halt telemetry,
retrieval controls, and verifier labels. Without those, "latent reasoning" is
only a convenient description of hidden computation.

### Inference Cliff

An inference cliff is a failure where a model appears competent under a narrow
prompt, teacher-forced trace, or short eval, but collapses when it must generate
freely, reason for more steps, hide the CoT, lower donor support, or handle a
held-out distractor/conflict case.

QTRM reduces this risk by requiring staged gates:

1. donor-only fluency baseline;
2. QTRM residual does not damage donor fluency;
3. QTRM improves held-out answer score;
4. workspace/core/context ablations explain the gain;
5. repeated latent depth converges or halts safely;
6. donor annealing happens only after QTRM-only logits are stable.

This does not guarantee that no inference cliff exists. It only makes cliffs
visible earlier and prevents architecture claims from outrunning evidence.

## Current Correct Label

The most accurate current label is:

```text
Qwen-backed looped latent-workspace residual cognitive adapter
```

Short form:

```text
donor-backed residual cognitive adapter
```

Avoid these labels for now:

- standalone Qwen replacement;
- proven AGI reasoning core;
- fully TRM-faithful implementation;
- MemoryOS-in-the-model architecture;
- internal long-term memory system;
- donor-free student model.

## Architecture Improvement Boundary

There are still real architecture improvements to make, but the next step is
not to keep adding components. The current architecture already has the minimum
hooks needed to test the key claim:

```text
Does a gated latent workspace/core path improve evidence-sensitive answers
over donor-only and ordinary visible-context baselines?
```

Only after that gate passes should QTRM add heavier mechanisms such as:

- true TRM-style persistent carry and per-sequence ACT;
- stronger latent distillation from explicit CoT/verifier traces;
- learned retrieval/rerank feedback into MemoryOS;
- sparse memory routing for very large external memory pools;
- donor-annealed or donor-free student training.

## Related Wiki Pages

- [QTRM Goal And Scope](../decisions/qtrm-goal-and-scope.md)
- [Qwen Donor Risk And Metacognition Boundary](../decisions/qwen-donor-risk-and-metacognition.md)
- [QTRM Forward Pass](../architecture/qtrm-forward-pass.md)
- [Internalized Context Engineering](internalized-context-engineering.md)
- [CoT To Latent Transfer](cot-to-latent-transfer.md)
- [Workspace Evidence Path](workspace-evidence-path.md)
- [Gated Core Context Injection](gated-core-context-injection.md)
- [QTRM Limitations And Mitigation Roadmap](../decisions/limitations-mitigation-roadmap.md)
