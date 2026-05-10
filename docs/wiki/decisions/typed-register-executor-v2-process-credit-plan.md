# Typed Register Executor V2 Process-Credit Plan

Date: 2026-05-05

Status: rejected experiment.

## Failure

`typed_register_executor_only_s120` was causal but harmful:

```text
baseline core role-value:     184/624
typed-register full:          106/624
typed-register-off ablation:  184/624
trace exact:                    0/32
```

The executor was in the answer/value path, but value CE alone asked a randomly
initialized recurrent register machine to discover operation/process semantics
from sparse final role slots. That is not enough credit assignment.

## Prior Signal

Checked 2026-05-05:

| Source | Relevant mechanism | Adaptation |
| --- | --- | --- |
| https://arxiv.org/abs/2603.20219 | Latent lookahead recursively feeds hidden states before token commitment and supervises latent predictions against future ground-truth tokens. | Add supervised process credit to latent/register steps before final answer CE. |
| https://arxiv.org/abs/2602.10520 | RLTT argues LoopLMs need reward over latent thought trajectories, not only final-state credit. | Treat operation/process-code CE as dense trajectory credit for each recurrent step. |
| https://arxiv.org/abs/2603.19714 | LoopRPT applies reinforcement pretraining signals directly to latent steps for looped models. | Keep reward/process signal inside training; do not add an inference-time solver. |
| https://arxiv.org/abs/2510.01265 | RLP moves exploration/reasoning reward earlier into pretraining and uses information gain over future-token prediction. | Prefer process-aware pretraining-style objectives over post-hoc output-only tuning. |
| https://arxiv.org/abs/2601.08058 | Reasoning can be activated by latent internal states, not only visible CoT. | Preserve the universal LLM path and improve hidden recurrent state instead of adding visible CoT shortcuts. |

## Architecture Claim

V2 claim:

```text
prompt tokens
-> frozen donor hidden-state context
-> QTRM workspace + mandatory recurrent core
-> typed-register process selector
-> persistent typed registers
-> role-value readout
-> LM answer path
```

The process selector is learned from labels during training only. It is not a
runtime rule solver and does not compute the final answer outside the model.

## Minimal Change

Add one training objective:

```text
core_typed_register_operation_ce
```

Target source:

```text
transition_state_codes[depth]
```

Why this is still KISS:

```text
Raw-intelligence axis: pure recursive reasoning
SSOT source: one canonical prompt token stream plus supervised process labels
Smallest path: existing typed-register operation head, no new model branch
Needed now because: value-only CE regressed below baseline
Duplicated logic avoided: no external solver/verifier in inference
Canonical gate: full > baseline and typed-register-off < full
```

## Experiment

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_v2_process_s120.yaml
```

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_v2_process_s120.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt \
  --steps 120 --depth-steps 1,2,4,8 \
  --out-dir local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_executor_v2_process_len579_s120_from_len579_s240 \
  --final-logit-ce-weight 0.0 --depth-final-ce-weight 0.0 --progress-margin-weight 0.0 \
  --core-typed-register-ce-weight 1.0 \
  --core-typed-register-operation-ce-weight 0.5
```

## Acceptance Gate

Promote only if:

```text
value accuracy full       > 184/624
step exact full           > 16/256
trace exact full          > 0/32
action-code exact          = 32/32
typed-register-off        < full
```

Reject if full is below the 184/624 baseline or disabling the typed-register
path restores/improves performance.

## Result

Command completed and saved:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_executor_v2_process_len579_s120_from_len579_s240/last.pt
```

Held-out value-state full:

```text
value accuracy: 102/624 = 0.1634615385
step exact:       0/256 = 0.0
trace exact:      0/32
```

Typed-register-off:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Action-code preservation:

```text
exact:        32/32
step acc:      1.0
finality acc:  1.0
halted exact: 32/32
```

## Decision

Reject `typed_register_executor_v2_process_s120` as canonical.

The added process-code CE preserves the accepted action-code controller, but it
does not make the typed-register value path useful. The decisive ablation is:

```text
typed-register full: 102/624
typed-register off:  184/624
```

This means the typed-register value path is still harmful. Process-code CE
alone is not enough credit assignment for exact value updates.

## Next If Rejected

Do not add another isolated value head. Escalate to a root-structure change:

```text
operation/process CE
+ register-transition consistency
+ latent lookahead rollout across depths 1/2/4/8
+ verifier/process reward as training-only signal
```

The next proof must still preserve:

```text
prompt -> tokens -> core/register/memory -> LM logits -> text
```
