# Donorless Recurrent Depth Probe

Date: 2026-05-08

Status: active L1 scaffold.

## Gate Header

```text
Target level:
  L1 scaffold

Major bottleneck:
  prerequisite reset for bottleneck 2, recursive depth scaling and halting

Baseline to beat:
  shallow recurrent depth and component ablations

Required score:
  depth8 final exact >= 0.95 on held-out modular programs

Required ablation drop:
  depth8 full - worst(state_reset, op_zero, op_shuffle) >= 0.25

Perturbation/held-out split:
  train and eval use different RNG seeds over start values and operation
  sequences

Promotion decision if pass:
  accept only as L1 evidence that a learned donorless recurrent latent state can
  benefit from depth; do not count as a major QTRM bottleneck

Kill decision if fail:
  do not keep integrated donor-QTRM tuning; inspect whether the recurrence cell
  or task design fails before moving back into QTRM
```

## Why This Probe

The integrated QTRM reverse-composition path has accumulated too many local
rejects. Per the major-gate discipline, the next step is to reset to the
smallest donorless recurrence that can show a depth gain before adding donor,
MemoryOS, renderer, metacognition, or MSA paths.

## Implemented Path

```text
start token
-> learned latent state
-> recurrent GRUCell update over one operation token per depth
-> state logits after each depth
```

No Qwen donor, retrieval, MemoryOS, symbolic solver, hidden evidence, or answer
channel is used. The task is modular program execution over held-out synthetic
operation sequences. Intermediate prefix targets supervise the recurrent state
at every depth.

Code:

```text
scripts/260_train_donorless_recurrent_depth_probe.py
tests/test_donorless_recurrent_depth_probe.py
scripts/300_research_gate_runner.py
tests/test_research_gate_runner.py
```

## Acceptance Meaning

If accepted, this only proves:

```text
learned recurrent state + more steps can solve a controlled transition task
better than shallow depth and ablations
```

It does not prove:

```text
QTRM integrated donor path works
normal language generation works
major bottleneck 2 is solved
ASI/general LLM progress
```

The next step after an L1 pass is to port the same minimal recurrence pressure
back into the QTRM core and require:

```text
donor-only < QTRM
core_off < QTRM
depth1 < depth4/depth8
held-out exact improves without hidden answer channels
```

## Runner Result 2026-05-08T09:35:58

```text
gate: donorless_recurrent_depth
target_level: L1 scaffold
profile: standard
decision: rejected
accepted: False
next_action: stop integrated donor-QTRM tuning; redesign the donorless recurrence/task until an isolated depth gain is accepted
```

Decisive metrics:

```json
{
  "eval_metrics.depth8_final_exact": 0.08203125,
  "eval_metrics.depth4_final_exact": 0.0234375,
  "eval_metrics.depth1_final_exact": 0.005859375,
  "ablations.state_reset.depth8_final_exact": 0.005859375,
  "ablations.op_zero.depth8_final_exact": 0.00390625,
  "ablations.op_shuffle.depth8_final_exact": 0.03125,
  "last_loss": 0.5248516798019409
}
```

Report: `local_eval/donorless_recurrent_depth_probe_s1200/report.json`

## Runner Result 2026-05-08T09:40:35

```text
gate: donorless_recurrent_depth
target_level: L1 scaffold
profile: standard
decision: rejected
accepted: False
next_action: stop integrated donor-QTRM tuning; redesign the donorless recurrence/task until an isolated depth gain is accepted
```

Decisive metrics:

```json
{
  "eval_metrics.depth8_final_exact": 0.08203125,
  "eval_metrics.depth4_final_exact": 0.0234375,
  "eval_metrics.depth1_final_exact": 0.005859375,
  "ablations.state_reset.depth8_final_exact": 0.005859375,
  "ablations.op_zero.depth8_final_exact": 0.00390625,
  "ablations.op_shuffle.depth8_final_exact": 0.03125,
  "last_loss": 0.5248516798019409
}
```

Report: `local_eval/research_gate_runner/donorless_recurrent_depth_standard/report.json`

## Runner Result 2026-05-08T09:44:25

```text
gate: donorless_recurrent_depth
target_level: L1 scaffold
profile: standard
decision: rejected
accepted: False
next_action: stop integrated donor-QTRM tuning; redesign the donorless recurrence/task until an isolated depth gain is accepted
```

Decisive metrics:

```json
{
  "eval_metrics.depth8_final_exact": 0.017578125,
  "eval_metrics.depth4_final_exact": 0.013671875,
  "eval_metrics.depth1_final_exact": 0.005859375,
  "ablations.state_reset.depth8_final_exact": 0.009765625,
  "ablations.op_zero.depth8_final_exact": 0.01171875,
  "ablations.op_shuffle.depth8_final_exact": 0.017578125,
  "last_loss": 3.789130687713623
}
```

Report: `local_eval/research_gate_runner/donorless_recurrent_depth_transition_table_standard/report.json`

## Runner Result 2026-05-08T09:45:12

```text
gate: donorless_recurrent_depth
target_level: L1 scaffold
profile: standard
decision: accepted_l1
accepted: True
next_action: open qtrm_minimal_depth gate: port the same recurrence pressure into QTRM and require donor-only < QTRM plus core_off < QTRM
```

Decisive metrics:

```json
{
  "eval_metrics.depth8_final_exact": 1.0,
  "eval_metrics.depth4_final_exact": 0.017578125,
  "eval_metrics.depth1_final_exact": 0.005859375,
  "ablations.state_reset.depth8_final_exact": 0.0078125,
  "ablations.op_zero.depth8_final_exact": 0.01171875,
  "ablations.op_shuffle.depth8_final_exact": 0.05078125,
  "last_loss": 2.115734338760376
}
```

Report: `local_eval/research_gate_runner/donorless_recurrent_depth_transition_table_directce_standard/report.json`

## Runner Result 2026-05-08T09:53:03

```text
gate: donorless_recurrent_depth
target_level: L1 scaffold
profile: standard
decision: accepted_l1
accepted: True
next_action: open qtrm_minimal_depth gate: port the same recurrence pressure into QTRM and require donor-only < QTRM plus core_off < QTRM
```

Decisive metrics:

```json
{
  "eval_metrics.depth8_final_exact": 1.0,
  "eval_metrics.depth4_final_exact": 0.017578125,
  "eval_metrics.depth1_final_exact": 0.005859375,
  "ablations.state_reset.depth8_final_exact": 0.0078125,
  "ablations.op_zero.depth8_final_exact": 0.01171875,
  "ablations.op_shuffle.depth8_final_exact": 0.05078125,
  "last_loss": 2.115734338760376
}
```

Report: `local_eval/research_gate_runner/donorless_recurrent_depth_standard/report.json`
