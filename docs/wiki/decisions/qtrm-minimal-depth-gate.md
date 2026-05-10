# QTRM Minimal Depth Gate

Date: 2026-05-08

Status: active L2 local gate.

## Gate Header

```text
Target level:
  L2 local gate

Major bottleneck:
  minimal QTRM depth scaffold after donorless recurrence L1

Baseline to beat:
  donor-only and core-off primitive runtime artifacts

Required score:
  full QTRM primitive runtime answer accuracy >= 0.95

Required ablation drop:
  full - core_off >= 0.50
  full - donor forced-choice >= 0.25 when donor artifact is present

Promotion decision if pass:
  accept only as L2 evidence that QTRM core can drive the primitive scaffold
  better than donor/core-off

Kill decision if fail:
  redesign QTRM minimal depth path before renderer, memory, or metacognition
```

## Scope

This gate is deliberately narrow. It evaluates the existing primitive runtime
scaffold:

```text
prompt
-> donor hidden states
-> QTRM recurrent core
-> primitive operation logits
-> explicit primitive executor
-> answer
```

That means an accepted result does not prove normal autoregressive text
generation. It only says the QTRM core is causally useful for the scaffolded
operation-selection path.

## Runner

```bash
PYTHONPATH=src .venv/bin/python scripts/300_research_gate_runner.py \
  --gate qtrm_minimal_depth \
  --profile standard \
  --write-wiki
```

Underlying gate builder:

```text
scripts/301_build_qtrm_minimal_depth_gate.py
tests/test_qtrm_minimal_depth_gate.py
```

If accepted, the next bottleneck is the renderer/canonical LLM path:

```text
primitive scaffold success
-> latent/core state must affect normal LM logits
-> greedy/autoregressive answer must improve without external executor
```

## Runner Result 2026-05-08T09:48:21

```text
gate: qtrm_minimal_depth
target_level: L2 local gate
profile: standard
decision: accepted_l2
accepted: True
next_action: open renderer/canonical-LLM-path gate; primitive executor success is not yet normal autoregressive text generation
```

Decisive metrics:

```json
{
  "metrics.full_answer_accuracy": 1.0,
  "metrics.core_off_answer_accuracy": 0.0,
  "metrics.full_minus_core_off": 1.0,
  "metrics.donor_forced_choice_accuracy": 0.4765625,
  "metrics.donor_greedy_accuracy": 0.2265625,
  "metrics.full_minus_donor": 0.5234375
}
```

Report: `local_eval/research_gate_runner/qtrm_minimal_depth_standard/report.json`

## Runner Result 2026-05-08T09:53:16

```text
gate: qtrm_minimal_depth
target_level: L2 local gate
profile: standard
decision: accepted_l2
accepted: True
next_action: open renderer/canonical-LLM-path gate; primitive executor success is not yet normal autoregressive text generation
```

Decisive metrics:

```json
{
  "metrics.full_answer_accuracy": 1.0,
  "metrics.core_off_answer_accuracy": 0.0,
  "metrics.full_minus_core_off": 1.0,
  "metrics.donor_forced_choice_accuracy": 0.4765625,
  "metrics.donor_greedy_accuracy": 0.2265625,
  "metrics.full_minus_donor": 0.5234375
}
```

Report: `local_eval/research_gate_runner/qtrm_minimal_depth_standard/report.json`
