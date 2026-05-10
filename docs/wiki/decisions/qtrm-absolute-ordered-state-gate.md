
## Status Correction 2026-05-09

The `2026-05-09T09:14:48` smoke `accepted_l2` entry below is invalid. The
smoke thresholds were accidentally set to zero, so a zero-score run was marked
accepted. Later smoke/standard runs with nonzero thresholds are authoritative.

The first absolute-value gate also used the old train/eval split:

```text
train: data/filtered/pure_recursive_reasoning_smallrange_train256_cases.jsonl
eval:  data/eval/pure_recursive_reasoning_heldout_72.jsonl
```

That split is invalid for an absolute value-class head because held-out eval
values include classes never seen during train. The corrected split is:

```text
train: data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl
eval:  data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl
summary: data/filtered/qtrm_absolute_ordered_state_train512_v0to31.summary.json
```

The corrected split holds out combinations while preserving target-class
coverage. Result: the gate is still rejected, but it now shows a small causal
value signal (`full_value_accuracy` about 0.06-0.08, primitive-off 0.0) instead
of a pure 0.0/0.0 tie.

The prompt-open 300-step result also rejects:

```text
full_trace_exact_accuracy: 0.0
full_value_accuracy: 0.057692307692307696
primitive_off_value_accuracy: 0.0
```

Training-time prompt/value metrics can improve on the sampled train row, but
held-out ordered trace exact remains zero. This makes the absolute-class path a
poor L2 promotion route. The next architecture candidate should use the
already accepted source-position pointer state plus a pointer/copy-style value
renderer, not a flat absolute class head.

## Runner Result 2026-05-09T09:14:48

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: smoke
decision: accepted_l2
accepted: True
next_action: open canonical LM renderer gate: require final LM logits or generation to improve and drop under ordered-state-off ablation
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_smoke/report.json`

## Runner Result 2026-05-09T09:16:14

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_smoke/report.json`

## Runner Result 2026-05-09T09:17:16

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_standard/report.json`

## Runner Result 2026-05-09T09:21:23

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_standard/report.json`

## Runner Result 2026-05-09T09:32:44

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.06666666666666667,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.06666666666666667
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_smoke/report.json`

## Runner Result 2026-05-09T09:33:47

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.07692307692307693,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.07692307692307693
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_standard/report.json`

## Runner Result 2026-05-09T09:38:35

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: smoke
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.06666666666666667,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.06666666666666667
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_smoke/report.json`

## Runner Result 2026-05-09T09:39:33

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.0641025641025641,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.0641025641025641
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_standard/report.json`

## Runner Result 2026-05-09T09:42:51

```text
gate: qtrm_absolute_ordered_state
target_level: L2 local gate
profile: standard
decision: rejected
accepted: False
next_action: do not add answer bridges; fix QTRM ordered state learning or port the donorless ordered-slot transition more directly
```

Decisive metrics:

```json
{
  "full_trace_exact_accuracy": 0.0,
  "full_value_accuracy": 0.057692307692307696,
  "full_step_exact_accuracy": 0.0,
  "ablation_trace_exact_accuracy": 0.0,
  "ablation_value_accuracy": 0.0,
  "ablation_step_exact_accuracy": 0.0,
  "trace_drop": 0.0,
  "value_drop": 0.057692307692307696
}
```

Report: `/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_absolute_ordered_state_standard/report.json`
