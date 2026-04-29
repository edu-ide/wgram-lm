# QTRM Limitations And Mitigation Roadmap

Status: working architecture decision, 2026-04-29.

## Current Position

QTRM is a Qwen-backed latent-workspace residual adapter, not a standalone
language model. The current stable path keeps Qwen donor logits as the base
language policy and lets QTRM contribute a bounded residual:

```text
final_logits = donor_logits_scale * donor_logits
             + qtrm_logits_scale  * qtrm_logits
```

The immediate objective is not to replace Qwen. It is to prove that QTRM can add
measurable reasoning, memory, or verification behavior without regressing donor
fluency.

## Limitation Map

| Limitation | Mitigation | Prior references | Required gate |
| --- | --- | --- | --- |
| Donor dependency | Treat Qwen as the base policy and train QTRM only where a residual is useful | Proxy-Tuning, DExperts | donor-only vs donor+QTRM fixed eval |
| QTRM contribution is hard to isolate | Run component ablations: residual off, workspace off, core off, memory off | Side-Tuning, Ladder Side-Tuning, AdapterFusion | answer-level delta plus logit telemetry |
| Residual can damage donor fluency | Add gated residual, residual clamp, and KL-to-donor preservation loss | InstructGPT KL-to-reference, Learning without Forgetting, distillation | entropy/repetition and KL bounds |
| Not standalone | Keep donor-backed mode as the main path; only attempt QTRM-only after distillation | Knowledge distillation, Proxy-Tuning | donor_logits_scale=0 held-out eval |
| Inference cost doubles | Cache donor KV/hidden/logits, batch evals, and optionally apply residual only on evidence-sensitive steps | HF generation/cache patterns | throughput and VRAM benchmark |
| Latent reasoning is unproven | Use depth sweeps, workspace shuffle/zero tests, and synthetic tasks requiring latent state | Parcae, looped transformers, Tiny Recursive Models | performance changes causally with workspace/core |
| MemoryOS may be evidence-copy only | Add distractors, conflicts, missing-answer cases, temporal cases, and source labels | Atlas, Self-RAG, CRAG, RAPTOR | QTRM improves over donor-only on retrieved evidence |
| Multi-loss interference | Phase training: LM stability, then memory traces, then verification traces, then JEPA/world-model | JEPA/LeWM plus training-diagnostics references | no regression at each phase |

## Target Architecture Change

Replace the fixed residual scale with a bounded gated residual:

```text
gate = sigmoid(controller(z_h, evidence_summary))
residual = clamp(qtrm_logits, min=-r, max=r)
final_logits = donor_logits + gate * residual_scale * residual
```

Training should include a donor-preservation term:

```text
loss = task_loss
     + alpha * KL(fused_logits || donor_logits)
     + beta  * repetition_or_entropy_penalty
```

The gate must be inspected, not merely trained. A useful QTRM residual should
turn on for retrieval, contradiction, abstention, and reasoning cases, and remain
near-zero on ordinary donor-fluent text.

## Telemetry Required Before Scaling

Every residual experiment should log:

- donor top-1 token and probability
- QTRM residual top-1 token and probability
- fused top-1 token and probability
- whether the fused argmax changed
- KL divergence between fused and donor distributions
- residual norm and gated residual norm
- entropy, repeated n-gram rate, and max token run
- answer-level correctness under donor-only and donor+QTRM

Implementation status:

- `src/qtrm_mm/diagnostics.py` provides `residual_logit_telemetry`.
- `scripts/92_eval_qtrm_logits.py` emits `residual_telemetry` in JSON records
  and prints a compact text summary when donor logits are available.
- `tests/test_residual_telemetry.py` covers argmax shifts, donor scaling, and
  eval-script integration.

## Evaluation Matrix

Minimum modes:

```text
donor only
donor + fixed QTRM residual
donor + gated QTRM residual
donor + QTRM without workspace
donor + QTRM without recursive core
donor + MemoryOS evidence
donor + MemoryOS evidence + QTRM residual
```

Minimum task families:

- language fluency and Korean prompts
- arithmetic and short reasoning
- retrieval evidence copying
- distractor evidence rejection
- temporal conflict
- authority/source conflict
- missing-answer abstention
- critical synthesis with positive conclusion

## Go / No-Go Rules

Promote a QTRM change only when:

- donor-only remains coherent;
- QTRM does not increase repetition collapse or special-token loops;
- target-token rank and held-out task score improve together;
- at least one task family improves over donor-only;
- component ablations show which QTRM part caused the improvement.

Reject or roll back a QTRM change when:

- improvement appears only in one hand-picked sample;
- donor fluency regresses on ordinary prompts;
- residual argmax changes are frequent but not answer-improving;
- memory retrieval recall succeeds but generation still hallucinates;
- workspace/core ablations show the same result without the claimed component.

## Next Engineering Order

1. Done: add residual telemetry to the eval script.
2. Add fixed ablation modes for donor-only, residual, workspace-off, and core-off.
3. Add gated residual behind a config flag.
4. Add KL-to-donor loss behind a config flag.
5. Re-run the current hard MemoryOS held-out probes.
6. Only then resume longer training or JEPA/world-model losses.
