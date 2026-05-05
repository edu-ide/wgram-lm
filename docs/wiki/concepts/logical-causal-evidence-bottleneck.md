# Logical-Causal Evidence Bottleneck

Primary source map:
[Logical Causal Trust References](../sources/logical-causal-trust.md).

## Problem

The current MemoryOS/QTRM probe can retrieve the right evidence and still
answer incorrectly. That means retrieval success is not enough. The model must
be forced to decide whether workspace evidence supports, refutes, or fails to
answer the prompt before QTRM residual logits are allowed to affect the donor
answer.

## Mechanism

The implemented bottleneck adds four lightweight verifier heads over the final
recursive workspace state:

| Head | Meaning |
| --- | --- |
| `evidence_support_head` | Workspace evidence supports the chosen answer. |
| `evidence_refute_head` | Workspace evidence contradicts the chosen answer. |
| `evidence_missing_head` | Workspace evidence is insufficient. |
| `evidence_causal_gate_head` | Evidence is relevant enough to let QTRM residual logits write. |

The final gate is:

```text
sigmoid(causal_gate + support - refute - missing)
```

When `evidence_bottleneck_suppress_without_workspace` is enabled, this gate is
multiplied by `workspace_memory_present`. No workspace evidence means no
evidence-backed residual.

Important boundary: this gate must not be the default controller for all QTRM
reasoning. General QTRM/loop-LM-style latent reasoning should still write a
bounded residual from the prompt even when no external workspace evidence is
provided. The implementation therefore has
`evidence_bottleneck_applies_to_residual`:

- `true`: evidence-proof mode; the evidence gate controls QTRM residual logits;
- `false`: verifier-only mode; support/refute/missing/causal heads are trained
  and logged, but the general QTRM residual remains active.

The HF first-wave warmup uses verifier-only mode. Evidence-only causality
probes can still use the stricter residual-gating mode.

## Why This Is Not Just Another Loss

Counterfactual workspace preference loss can pass while generation still ignores
evidence. This bottleneck changes the information path: the QTRM residual is
bounded, donor-preserving, and evidence-gated.

The acceptance condition is not only lower loss. A successful run must show a
drop or completion change when evidence memory is disabled, shuffled, or the
evidence bottleneck is bypassed.

## Current Implementation

Files:

- `src/qtrm_mm/qtrm_model.py`: evidence verifier heads and residual gate.
- `src/qtrm_mm/losses.py`: logical evidence and causal evidence gate losses.
- `src/qtrm_mm/data/jsonl_dataset.py`: default support/gate targets for
  workspace-evidence rows.
- `configs/qwen35_2b_4090_logical_causal_bottleneck_s050.yaml`: first probe
  config.
- `scripts/127_run_logical_causal_bottleneck_train.sh`: training, pair eval,
  and ablation runner.

## Risks

- Too strict a gate can collapse to donor-only behavior.
- Applying the evidence gate to the whole residual makes ordinary prompts
  donor-only when no workspace evidence is present; that is wrong for a
  general latent-reasoning core.
- Too weak a gate will pass metrics without changing answers.
- The current default targets assume workspace evidence is supportive unless a
  row supplies explicit refute/missing labels.
- This is not a replacement for a symbolic prover on formal logic tasks.

## Next Acceptance Gate

Run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/127_run_logical_causal_bottleneck_train.sh
```

Accept the direction only if:

- `qtrm_residual_with_evidence` improves over the previous `0/4` quick gate;
- `qtrm_workspace_memory_off_with_evidence` or shuffled/counterfactual evidence
  causes a measurable drop or answer change;
- `qtrm_evidence_bottleneck_off_with_evidence` is different enough to prove the
  gate has causal effect;
- pair preference remains high but is not treated as sufficient.
