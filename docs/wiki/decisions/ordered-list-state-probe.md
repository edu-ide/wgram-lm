
## Runner Result 2026-05-09T08:53:19

```text
gate: ordered_list_state
target_level: L1 scaffold
profile: smoke
decision: rejected
accepted: False
next_action: do not tune answer bridges; redesign the ordered recurrent state until filter->double composition is accepted in isolation
```

Decisive metrics:

```json
{
  "eval_metrics.depth4_final_exact": 0.0,
  "eval_metrics.depth1_final_exact": 0.0,
  "eval_metrics.depth2_final_exact": 0.0,
  "eval_metrics.depth2_state_exact": 0.0,
  "ablations.state_reset.depth4_final_exact": 0.0,
  "ablations.state_reset.depth2_final_exact": 0.0,
  "ablations.op_zero.depth4_final_exact": 0.0,
  "ablations.op_zero.depth2_final_exact": 0.0,
  "ablations.order_shuffle.depth4_final_exact": 0.0,
  "ablations.order_shuffle.depth2_final_exact": 0.0,
  "last_loss": 4.4350152015686035
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/ordered_list_state_smoke/report.json`

## Runner Result 2026-05-09T08:53:33

```text
gate: ordered_list_state
target_level: L1 scaffold
profile: standard
decision: rejected
accepted: False
next_action: do not tune answer bridges; redesign the ordered recurrent state until filter->double composition is accepted in isolation
```

Decisive metrics:

```json
{
  "eval_metrics.depth4_final_exact": 0.88671875,
  "eval_metrics.depth1_final_exact": 0.01953125,
  "eval_metrics.depth2_final_exact": 0.86328125,
  "eval_metrics.depth2_state_exact": 0.86328125,
  "ablations.state_reset.depth4_final_exact": 0.017578125,
  "ablations.state_reset.depth2_final_exact": 0.017578125,
  "ablations.op_zero.depth4_final_exact": 0.873046875,
  "ablations.op_zero.depth2_final_exact": 0.841796875,
  "ablations.order_shuffle.depth4_final_exact": 0.072265625,
  "ablations.order_shuffle.depth2_final_exact": 0.0625,
  "last_loss": 0.06985588371753693
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/ordered_list_state_standard/report.json`

## Accepted L1 Interpretation

Date: 2026-05-09

Decision:

```text
accepted_l1
```

Accepted checkpoint:

```text
/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/ordered_list_state_standard/accepted_l1_ordered_list_state.pt
sha256: 49e38698e1b5fd82da1c54162a76975a4198e7d1000a43e484da28c3cc0ad116
```

What this proves:

```text
A learned recurrent ordered-slot state can preserve list order and compose
filter -> double on held-out synthetic list cases.
```

What this does not prove:

```text
It is not L3/L4 and not a universal LLM-path result. The probe is donorless,
structured, and still outside the canonical prompt -> tokenizer -> QTRM core
-> LM logits -> autoregressive text path.
```

Why op_zero is not decisive here:

```text
This L1 probe uses a fixed operation sequence. Since every task is always
filter-then-double, the recurrence can legitimately encode step count without
needing a variable operation token. Therefore the required causal ablations are
state_reset and order_shuffle. op_zero remains a diagnostic metric, not a hard
reject condition for this fixed-operation scaffold.
```

Next action:

```text
Port the ordered-slot transition into QTRM as a causal state path. The next
gate must prove that final LM logits depend on that ordered recurrent state:

full QTRM > donor_only/core_off/state_off
and ordered-state-off must drop on held-out list-transform forced-choice or
generation.
```

## Runner Result 2026-05-09T08:55:01

```text
gate: ordered_list_state
target_level: L1 scaffold
profile: standard
decision: accepted_l1
accepted: True
next_action: port the ordered-slot transition into QTRM so final LM logits depend on the ordered recurrent state; require source/state-off ablation drop before L3
```

Decisive metrics:

```json
{
  "eval_metrics.depth4_final_exact": 0.9140625,
  "eval_metrics.depth1_final_exact": 0.01953125,
  "eval_metrics.depth2_final_exact": 0.8828125,
  "eval_metrics.depth2_state_exact": 0.8828125,
  "ablations.state_reset.depth4_final_exact": 0.017578125,
  "ablations.state_reset.depth2_final_exact": 0.017578125,
  "ablations.op_zero.depth4_final_exact": 0.8828125,
  "ablations.op_zero.depth2_final_exact": 0.845703125,
  "ablations.order_shuffle.depth4_final_exact": 0.068359375,
  "ablations.order_shuffle.depth2_final_exact": 0.06640625,
  "last_loss": 0.014939426444470882
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/ordered_list_state_standard/report.json`
