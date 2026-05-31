# Ouro Answer Halt Head S080

Date: 2026-05-06

## Decision

Promote as the current canonical Ouro raw-recursive answer-path candidate.

The accepted part is narrow:

- answer-state hidden gets its own learned halt head;
- the halt head is trained with gate disabled;
- eval enables a hard-first in-loop halt gate;
- no retrieval, MemoryOS, donor logits, hidden evidence, or external solver is used.

This is not yet a broad ASI or general reasoning claim. It is a first causal
single-trace answer-loop result on the mixed-list arithmetic forced-choice
smoke gate. Autoregressive answer rendering remains unsolved.

## Prior Work Used

- ACT: learned adaptive computation depth for recurrent networks.
  <https://arxiv.org/abs/1603.08983>
- Universal Transformer: recurrent Transformer depth plus dynamic halting.
  <https://arxiv.org/abs/1807.03819>
- PonderNet: learned computation steps balancing accuracy, compute, and
  generalization.
  <https://arxiv.org/abs/2107.05407>
- PonderLM: continuous-space pondering within token generation.
  <https://arxiv.org/abs/2505.20674>
- FR-Ponder: latent steering controller for adaptive reasoning depth.
  <https://arxiv.org/abs/2509.24238>

## Implemented Change

Code:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
scripts/192_eval_raw_intelligence.py
scripts/196_train_pure_recursive_depth_supervised.py
```

New configs:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080.yaml
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
```

Experiment artifacts:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/lm_causal_forced_choice_smoke8_eval_gate.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/lm_causal_forced_choice_smoke16_eval_gate.jsonl
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/action_code_eval32_eval_gate.json
```

## Method

The answer-state loop now exposes:

```text
answer_state_loop_halt_logits: [batch, depth]
```

Training:

```text
answer_state_loop_halt_gate_enabled: false
trainable_param_policy: answer_state_loop_only
--depth-steps 4,8
--answer-state-loop-halt-ce-weight 1.0
```

Evaluation:

```text
answer_state_loop_halt_gate_enabled: true
answer_state_loop_halt_gate_mode: hard_first
```

The hard-first gate freezes the answer hidden state at the first depth whose
learned halt logit crosses zero. This makes halting causal inside the answer
path rather than a post-hoc selector.

## Results

S080 halt-head training:

```text
answer_state_halt_ce: 1.3863 -> 0.0427
answer_state_halt_acc: 0.0 -> 1.0
```

Smoke8 causal forced-choice:

```text
donor_only:        0/8
core_off:          0/8
core_steps1:       0/8
core_steps2:       0/8
core_steps4:       8/8
core_steps8 full:  8/8
halt_gate_off:     0/8
bridge_off:        8/8
```

Smoke16 causal forced-choice:

```text
core_steps4:       10/16
core_steps8 full:  10/16
halt_gate_off:      0/16
bridge_off:        10/16
```

Greedy generation sanity:

```text
max_cases: 4
modes: core_steps8, halt_gate_off
hits: 0/8
full sample: "1  UNKNOWN -1 UNKNOWN UNKNOWN-"
gate_off sample: multilingual/noisy token loop
```

Action-code sanity:

```text
action-code exact: 32/32
step_acc:          1.0
finality_acc:      1.0
halted_exact:      32/32
```

## Interpretation

Accepted:

```text
The answer halt gate is causal. Turning it off collapses the answer path to
0/16 on the expanded smoke while the full in-loop gate reaches 10/16.
```

Rejected or demoted:

```text
Transition-joint answer bridge is not the active causal path in this checkpoint:
bridge_off ties full at 10/16.
```

Depth interpretation:

```text
The useful answer state is reached by depth 4 on this task. Depth 8 succeeds
because hard-first halting freezes the depth-4 answer state, not because later
depths add useful reasoning.
```

Main remaining bottleneck:

```text
This is still a fixed-depth-4 smoke behavior. The next gate must test whether
the halt head adapts across mixed tasks requiring different depths, not merely
whether it can freeze one known terminal depth.
```

Generation bottleneck:

```text
The model can rank the correct answer under causal forced-choice, but greedy
generation still emits invalid text. The next renderer experiment must train
the same causal answer-halt path under autoregressive generation pressure
without reintroducing donor-logit shortcuts.
```

## Renderer Follow-Up Probes

These probes were run after accepting the halt-head checkpoint. None replaces
the canonical S080 halt-head baseline.

Naive teacher-forced answer-loop CE:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_renderer_s080_from_halt_s080/last.pt

generation smoke4:
  full/gate_off: 0/8
  sample: "UNKNOWNAnswer: UNKNOWN5Answer00:0AnswerAnswer"

causal forced-choice smoke8:
  full:          0/8
  halt_gate_off: 0/8

decision:
  reject, checkpoint deleted
```

Zero-init low-rank answer-state LM adapter:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_renderer_adapter_s120.yaml

trainable:
  answer_state_loop_lm_adapter_down/up only

generation smoke4:
  full/gate_off: 0/8
  sample: "1  UNKNOWN -11 UNKNOWN UNKNOWN-"

causal forced-choice smoke8:
  full:          8/8
  halt_gate_off: 0/8

decision:
  reject as renderer, checkpoint deleted
  note: forced-choice reasoning was preserved, but greedy rendering did not improve
```

Greedy-top margin adapter:

```text
new loss:
  final/depth gold token must beat current top non-gold token by a margin

generation smoke4:
  full/gate_off: 0/8
  sample: "24434264752"

causal forced-choice smoke8:
  full:          4/8
  halt_gate_off: 0/8

decision:
  reject, checkpoint deleted
  note: moved UNKNOWN repetition into wrong digit streams and damaged the
        accepted forced-choice gate
```

Causal-prefix self-rollout adapter:

```text
new loss:
  train causal-prefix targets on the model's own greedy rollout prefixes
  instead of only gold teacher-forced prefixes

training telemetry:
  causal_prefix_self_rollout_examples: 6
  causal_prefix_self_rollout_prefix_tokens: 5
  causal_prefix_self_rollout_prefix_mismatch_rate: 0.8-1.0

generation smoke4:
  full/gate_off: 0/8
  full sample: "1  UNKNOWN -11 UNKNOWN UNKNOWN-"
  gate_off sample: multilingual/noisy token loop

causal forced-choice smoke8:
  full:          8/8
  halt_gate_off: 0/8

decision:
  reject as renderer, checkpoint deleted
  note: self-rollout preserves the accepted causal answer scorer but still
        fails to turn it into a stable autoregressive answer generator
```

Frozen donor-logit fusion sanity:

```text
checkpoint:
  accepted halt-head S080 baseline

eval:
  donor_logits_scale=1.0
  qtrm_logits_scale=1.0

generation smoke4:
  full: 0/4
  samples: "10011216171", "10011213141"

decision:
  donor language logits alone do not solve answer rendering
```

Beam-search renderer diagnostic:

```text
checkpoint:
  accepted halt-head S080 baseline

beam:
  width: 16
  max_new_tokens: 7
  cases: first 4 held-out mixed-composition rows

result:
  hits: 0/4
  best samples: "1 1 1", "UNKNOWN 1 1"

diagnosis:
  the failure is not greedy-only decoding. The answer token "3" can appear in
  the early top-k on one row, but the short beam still collapses to the same
  high-prior invalid token pattern.
```

Renderer conclusion:

```text
The accepted core can act as a causal answer scorer, but the answer hidden is
not yet a stable autoregressive digit/value generator. Low-rank output
patching, greedy-token margins, frozen donor-logit fusion, and online
self-rollout prefix training are all insufficient. Beam search also does not
recover the correct short answers. The next renderer candidate
should stop treating rendering as a small output-head patch and instead add a
tokenizer-aligned answer-state decoder/readout that is trained and ablated as
part of the same causal loop.
```

## Rejected Sibling Probe

The direct in-loop transition-finality gate was rejected:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_gate_zeroshot.yaml

smoke8:
  full with finality gate: 2/8
  finality_gate_off:      4/8
```

Reason:

```text
Transition finality means operation terminality, not answer-readiness. Using it
as the answer halt signal prematurely freezes the wrong state.
```

The first answer-halt S020 with gate enabled during training was also not
promoted:

```text
full:      4/8
gate_off:  4/8
```

Reason:

```text
The new halt head had not learned enough, so the gate had no measurable causal
effect. Separating halt-head training from hard-gate evaluation fixed this.
```

## Next Gate

Use the accepted S080 checkpoint as the next baseline and run:

```text
1. 32/64-case scale-up with full, halt_gate_off, core_steps1/2/4/8.
2. Mixed-depth tasks where correct terminal depth is not always 4.
3. Bridge removal cleanup: treat transition_joint_answer_bridge as noncanonical
   unless future ablations prove it adds accuracy.
4. Generation test, not only causal forced-choice, after the forced-choice gate
   holds on larger heldout sets.
```
