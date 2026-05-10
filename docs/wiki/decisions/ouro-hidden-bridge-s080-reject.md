# Ouro Hidden Bridge S080 Reject

Date: 2026-05-06

## Failure

Donor-unembedding head surgery and LM-head-only tuning both failed. The next
minimal hypothesis was:

```text
Maybe answer-state hidden vectors need a trainable hidden bridge before the
existing tokenizer LM head, rather than a direct vocab-head patch.
```

## Prior Mapping

This was the smallest PonderLM/Coconut-compatible bridge probe available in the
current architecture:

```text
answer-state hidden
-> zero-init residual hidden bridge
-> existing LM head
-> autoregressive text
```

Unlike the prior low-rank LM adapter, this probe changes hidden geometry before
the LM head, not logits after the head.

## Implementation

Added:

```text
model.answer_state_loop_hidden_bridge_enabled
model.answer_state_loop_hidden_bridge_hidden_dim
model.answer_state_loop_hidden_bridge_scale
trainable_param_policy: answer_state_loop_hidden_bridge_only
eval mode: qtrm_core_steps_N_answer_hidden_bridge_off_no_evidence
```

Files:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
scripts/192_eval_raw_intelligence.py
tests/test_core_halting.py
tests/test_training_checkpoint_init.py
tests/test_raw_intelligence_eval_script.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_hidden_bridge_s080.yaml
```

The bridge is zero-init and no-op safe at initialization. Its ablation is
explicitly testable.

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt

trainable:
  answer_state_loop_hidden_bridge_only
  525,824 params, 5 tensors

steps:
  80
```

## Results

Generation smoke4:

```text
full:              0/4
hidden_bridge_off: 0/4
halt_gate_off:     0/4

full sample:
  "10040 02030"

hidden_bridge_off sample:
  "1  UNKNOWN -1 UNKNOWN UNKNOWN-"

halt_gate_off sample:
  "Answer:AnswerAnswer Answer2Answer1Answer-Answer"
```

Causal forced-choice smoke4 was stopped early after the decisive rows:

```text
full:              0/4
hidden_bridge_off: 4/4
halt_gate_off:     0/1 observed before early stop
```

Decision:

```text
reject, checkpoint deleted
```

## Interpretation

This is stronger than a neutral failure:

```text
The trained hidden bridge actively damages the accepted forced-choice signal.
Turning the bridge off restores the smoke4 forced-choice hits, while generation
still fails either way.
```

Updated root diagnosis:

```text
The current private QTRM answer-state space is good enough for narrow
forced-choice scoring under the accepted halt gate, but not stable enough to
become a standalone autoregressive language renderer through small head/bridge
patches.
```

## Architecture Implication

Do not keep stacking renderer patches on the private QTRM head:

```text
rejected:
  low-rank LM adapter
  greedy-token margin adapter
  causal-prefix self-rollout
  donor-unembedding head surgery
  LM-head-only tuning
  hidden bridge tuning
```

Next root candidate should preserve the donor decoder/language path more
directly:

```text
canonical prompt tokens
-> frozen donor decoder hidden states
-> QTRM recursive cognitive controller
-> small residual/modulation into donor-compatible hidden path
-> donor-compatible LM head / donor logits
-> autoregressive text
```

This keeps QTRM responsible for raw reasoning/control, but stops asking a
small randomly initialized QTRM decoder head to relearn fluent token rendering.
