# Prompt Source-Position Binder Probe

## Purpose

This L1 probe isolates the new bottleneck found after the rejected
`qtrm_source_pointer_state` gate:

```text
Can a small neural binder read the prompt token stream and identify which
source list positions should be selected?
```

If this fails, QTRM recurrent-state or renderer tuning is premature. The model
does not yet have a reliable prompt-position binding substrate.

## Prior

The result matches recent numerical-representation literature: numerical
reasoning failures often come from how numbers are tokenized or represented,
not only from insufficient Transformer depth.

Relevant prior:

- Value-Aware Numerical Representations for Transformer Language Models
  (`arXiv:2601.09706`): injects explicit magnitude information into the input
  space while staying compatible with decoder-only Transformers.
- A Triadic Suffix Tokenization Scheme for Numerical Reasoning
  (`arXiv:2604.11582`): argues that inconsistent number fragmentation loses
  positional and decimal structure.
- NumeroLogic (`arXiv:2404.00459`): adds number encodings to improve numerical
  reasoning.
- LUNA (`arXiv:2212.02691`): uses number plugins / number embeddings for
  numerical understanding.

## Experiment

Data:

```text
train: data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl
eval: data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl
target: source positions for depth 1 filter_even state
```

Model:

```text
prompt tokens
-> token embedding or frozen Qwen donor hidden states
-> slot-query Transformer binder
-> source-position class per output slot
```

This is a diagnostic L1 scaffold. It does not compute the final answer and does
not claim canonical QTRM progress.

## Results 2026-05-09

Token embedding, 300 steps:

```text
out: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_s300
best eval exact_acc: 0.640625
best eval slot_acc: 0.884765625
```

Larger token embedding binder, 1000 steps:

```text
out: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_s1000_h512_l2
best eval exact_acc: 0.859375
best eval slot_acc: 0.95703125
```

Frozen Qwen donor hidden binder, 500 steps:

```text
out: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_donor_s500
best eval exact_acc: 0.375
best eval slot_acc: 0.806640625
```

Numeric-aware value embedding, 300 steps:

```text
out: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_numeric_s300
decision: accepted_l1
best eval exact_acc: 1.0
best eval slot_acc: 1.0
```

Token path + numeric value embedding, 300 steps:

```text
out: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_plus_numeric_standard
decision: accepted_l1
best eval exact_acc: 0.9453125
best eval slot_acc: 0.986328125
```

## Decision

Mixed decision:

```text
prompt-token only: rejected
frozen Qwen donor hidden: rejected
numeric-aware input representation: accepted_l1
```

The token-only binder can overfit train exact to 1.0 but held-out exact plateaus
below 0.90. Frozen Qwen donor hidden states are worse for this small probe.
The numeric-aware value embedding reaches exact 1.0 on the held-out split.
The canonical token-path numeric variant also passes the L1 threshold at
0.9453125 while keeping numeric information aligned to tokenizer positions.

Therefore the bottleneck is not just recurrent depth. It is numeric
source-position binding from the prompt representation. The next QTRM candidate
should not keep relying on raw BPE token hidden states alone for numeric source
slots.

## Next Architecture Candidate

The next candidate should add a numeric-aware input representation before QTRM
state recurrence. Prefer the token-path variant over the separate numeric-only
source-slot side channel:

```text
chat template / prompt text
-> tokenizer
-> token embeddings + token-aligned numeric value embeddings
-> recurrent source-position binder
-> QTRM recursive core
-> LM logits
```

The first direct QTRM promotion attempt showed that this binder should not be
introduced as a fresh, ungated module on top of a trained recurrent-state
checkpoint. QTRM now gates both token-numeric residuals and internal
source-position binder logits, but the L2 gate still rejects. The next
promotion attempt should reuse the accepted L1 binder weights or run a
controlled binder-pretrain stage before primitive recurrent-state training.

Promotion rule:

```text
Pass L1 only if prompt/source-position exact_acc >= 0.90 on the corrected
held-out combination split. The numeric-aware representation now satisfies
this criterion.

Promote to QTRM L2 only if core state improves over numeric-binder-only and
degrades under numeric-feature-off / recurrent-core-off ablations.
```

## Runner Result 2026-05-09T10:44:36

```text
gate: prompt_source_position_binder
target_level: L1 scaffold
profile: smoke
decision: rejected
accepted: False
next_action: add numeric-aware input representation or digit/value features before retrying recurrent pointer-state QTRM L2
```

Decisive metrics:

```json
{
  "best_exact_acc": 0.0
}
```

Report: `local_eval/research_gate_runner/prompt_source_position_binder_smoke/report.json`

## Runner Result 2026-05-09T10:48:08

```text
gate: prompt_source_position_binder_numeric
target_level: L1 scaffold
profile: standard
decision: accepted_l1
accepted: True
next_action: port numeric-aware source-slot embeddings into QTRM and require numeric-feature-off plus core-off ablation drops
```

Decisive metrics:

```json
{
  "best_exact_acc": 1.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_numeric_standard/report.json`

## Runner Result 2026-05-09T11:27:10

```text
gate: prompt_source_position_binder_token_plus_numeric
target_level: L1 scaffold
profile: smoke
decision: rejected
accepted: False
next_action: canonical token-path numeric binding is still insufficient; improve token-aligned value representation before QTRM L2
```

Decisive metrics:

```json
{
  "best_exact_acc": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_plus_numeric_smoke/report.json`

## Runner Result 2026-05-09T11:27:30

```text
gate: prompt_source_position_binder_token_plus_numeric
target_level: L1 scaffold
profile: standard
decision: accepted_l1
accepted: True
next_action: replace side-channel numeric source features with token-path value-aware embeddings in QTRM source-pointer L2
```

Decisive metrics:

```json
{
  "best_exact_acc": 0.9453125
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/prompt_source_position_binder_token_plus_numeric_standard/report.json`

## Hard L3 Split Recheck 2026-05-09

The L3 hard source-pointer split exposed a parser bug in this probe: a
single-number list-transform state such as `"2"` must be treated as a
one-element list state, not as a non-list scalar. The probe now uses
`row_list_state_values(...)`, matching the corrected QTRM state-codec behavior.

Validated with:

```bash
PYTHONPATH=src .venv/bin/python tests/test_prompt_source_position_binder_probe.py
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/320_train_prompt_source_position_binder_probe.py
```

Result:

```text
7 tests OK
py_compile OK
```

Hard split data:

```text
train: data/filtered/qtrm_source_pointer_l3_hard_train512_s1321.jsonl
eval:  data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl
target: depth-1 source-position state, including singleton list results
```

Results:

```text
token_plus_numeric, h256/l1/s300:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          prompt_source_binder_hard_tokenplus_s300/report.json
  decision: rejected
  best_exact_acc: 0.765625

numeric_value_embedding, h256/l1/s300:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          prompt_source_binder_hard_numeric_s300/report.json
  decision: accepted_l1
  best_exact_acc: 0.9609375

token_plus_numeric, h512/l2/s600:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          prompt_source_binder_hard_tokenplus_l2h512_s600/report.json
  decision: rejected
  best_exact_acc: 0.8046875

donor_hidden, h512/l1/s300:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          prompt_source_binder_hard_donorhidden_s300/report.json
  decision: rejected
  best_exact_acc: 0.671875

token_numeric_source_slots, h256/l1/s300:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          prompt_source_binder_hard_token_numeric_slots_s300/report.json
  decision: accepted_l1
  best_exact_acc: 0.9921875
```

Interpretation:

```text
The hard split is learnable from a compact numeric source-slot sequence.
It is also learnable when that compact sequence is derived from tokenizer
offsets over the visible prompt. It is not yet reliably learnable from the
full token-aligned numeric span sequence or frozen Qwen donor hidden states.
```

Decision:

```text
numeric source-slot probe: accepted_l1 diagnostic
token-derived compact source-slot probe: accepted_l1 canonical candidate
token/donor prompt binder on hard split: rejected
QTRM L2/L4 retry: blocked until source binding is made canonical and causal
```

The next architecture candidate should not jump to renderer or LeWM work. It
should port `token_numeric_source_slots` into the QTRM causal path as compact
source-slot latent tokens derived from tokenizer offsets over the prompt, then
require both:

```text
source-slot path on/off drop
recurrent core on/off drop
```

before another L2/L3/L4 promotion attempt.
