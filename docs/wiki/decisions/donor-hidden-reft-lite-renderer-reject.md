# Donor Hidden ReFT-Lite Renderer Reject

Date: 2026-05-08

Status: rejected L0/L1 renderer falsifier.

## Gate

```text
Target level:
  L0/L1 renderer falsifier

Major bottleneck:
  latent-core-to-autoregressive-text

Baseline to beat:
  donor_only_no_evidence and qtrm_core_off_no_evidence

Required score:
  greedy generation must improve above donor-only on held-out rows

Required ablation drop:
  qtrm_core_steps_8 > qtrm_core_off
  qtrm_core_steps_8 > delta_off

Kill decision if fail:
  do not keep tuning QTRM private logits or final-hidden-only adapters.
```

## Implementation

Added:

```text
scripts/303_train_donor_hidden_reft_lite.py
tests/test_donor_hidden_reft_lite.py
```

Path:

```text
prompt tokens
-> frozen Qwen final hidden states
-> QTRM core_loop_readout_hidden
-> low-rank ReFT-lite delta
-> donor final hidden + delta
-> frozen donor lm_head
-> greedy generation
```

This preserves the universal LLM path better than a private QTRM head, but it
is still only a final-hidden intervention. It does not hook internal donor
layers.

## Run 1: Multi-Token Adapter

Artifact:

```text
local_eval/research_gate_runner/donor_hidden_reft_lite_s300/report.json
```

Result:

```text
decision: rejected
teacher-forced donor_top1:     0.3958
teacher-forced reft_full_top1: 0.4583
teacher-forced core_off_top1:  0.3958
generation donor:             0/8
generation qtrm_core_steps_8:  0/8
generation core_off:          0/8
```

Interpretation:

```text
The bridge can move some next-token ranks, but it does not make greedy
autoregressive text correct.
```

## Run 2: Visible-Reasoning Suppression Eval

Artifact:

```text
local_eval/research_gate_runner/donor_hidden_reft_lite_s300_suppress_think_eval/report.json
```

Result:

```text
decision: rejected
generation donor:            0/8
generation qtrm_core_steps_8: 0/8
```

Suppression removed `<think>` leakage but exposed numeric instability:

```text
gold 300015 -> qtrm completion 1000077 / 7777777
gold 400037 -> qtrm completion 1000077 / 7777777
```

## Run 3: First-Token-Only Adapter

Artifact:

```text
local_eval/research_gate_runner/donor_hidden_reft_lite_firsttok_s500/report.json
```

Result:

```text
decision: rejected
teacher-forced donor_top1:     0.0000
teacher-forced reft_full_top1: 1.0000
teacher-forced core_off_top1:  1.0000
generation donor:             0/8
generation qtrm_core_steps_8:  0/8
generation core_off:          0/8
```

Interpretation:

```text
The adapter learned a non-causal format/token bias. Since core_off also reaches
1.0 teacher-forced first-token top1, the gain is not caused by recursive core
reasoning.
```

## Decision

Final-hidden ReFT-lite is rejected.

Do not promote:

```text
donor_hidden_last + P(qtrm_core_state) -> donor lm_head
```

as the canonical renderer.

## Next Candidate

Move one level earlier in the donor path:

```text
Candidate A:
  true internal donor-layer residual hook
  donor_hidden[layer, answer-token-position] += alpha * gate * P(qtrm_core_state)

Candidate B:
  core-conditioned soft prefix / prefix-tuning
  prompt embeddings + learned virtual tokens from qtrm_core_state
  -> frozen donor transformer
  -> donor lm_head
```

Promotion still requires:

```text
generation > donor_only
generation > core_off
delta_off returns donor behavior
core_off loses the improvement
no hidden answer channel or external solver
```

