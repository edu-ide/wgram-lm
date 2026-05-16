# Orthodox TRM General-LLM Direction

Date: 2026-05-11

Status: canonical direction, not a completed capability claim.

## 2026-05-16 Correction: What "TRM Conditions" Mean Here

For this project, "TRM-inspired" is no longer enough. A QTRM result counts as
orthodox only when it behaves like a TRM-style loop reasoning model inside the
normal LLM path:

```text
prompt tokens
-> tokenizer
-> native embeddings/backbone
-> repeated recursive latent state transition
   z_L / z_H or equivalent state
   multiple loop steps or learned halt
   state carried into the answer path
-> core-dependent readout
-> LM logits
-> autoregressive answer
```

The key test is causal necessity:

```text
If the answer survives core_off, think0, state_reset, z_L_zero, z_H_zero,
or readout_off, then the result is not evidence for TRM-style reasoning.
```

Therefore the Qwen-integrated residual-adapter path is diagnostic unless it
can be converted into a single native model where the recurrent core is inside
the answer-producing hidden path and destructive state ablations remove the
gain. The current canonical proof target is QTRM-native first, not
Qwen-preserving residual improvement first.

## Decision

TRM-style recurrent latent reasoning is a valid direction for a general LLM
only when it remains inside the single prompt-token-to-logits path:

```text
prompt / chat template / compiled context
-> tokenizer
-> token embeddings or frozen donor hidden states
-> mandatory TRM/QTRM recurrent latent core
-> core-dependent answer readout
-> LM logits
-> autoregressive text
```

This keeps QTRM a general language model rather than a task-specific solver
attached to a language model.

## What This Means For QTRM

The primary reasoning core is the TRM/QTRM latent core, not MemoryOS, RAG, a
typed executor, a renderer, or a parallel answer loop.

Support modules are allowed, but only with this role:

- answer-state loop: read out and stabilize core trajectory state;
- typed register/value state: provide process-supervised internal state that
  feeds the LM answer path;
- source pointer/copy path: bind prompt positions when the task demands
  copying from visible text;
- MemoryOS/RAG: prepare context outside the model boundary, then compile it
  into the same visible token stream;
- verifier/solver: create labels, rewards, or evaluation gates, not inference
  answers.

If a support module can solve the task while the core trajectory is zeroed or
disabled, it is not evidence for TRM-style raw reasoning.

## Current Lesson

Recent L4 sufficient-condition work made the generation target stricter and
fixed termination/EOS behavior, but it also exposed the main architectural
weakness:

```text
full generation can improve on tiny cases,
but core_state_zero and several upstream ablations can still preserve the
answer.
```

That means the current successful surface behavior is not yet enough to prove
that the recurrent latent core is the causal reasoning mechanism. The next
work must make the core trajectory state necessary, not merely present.

## Acceptance Gate

A TRM-general-LLM result can be promoted only if the same held-out runner shows:

```text
full QTRM > donor-only/simple baseline
full QTRM > core_off
core_steps=8 > core_steps=1 or another clear depth-sweep gain
core_state_zero reduces or removes the gain
answer/readout path off reduces or removes the gain
normal LM logits/autoregressive generation produces the answer
```

For broader L4 promotion, add:

```text
held-out perturbations preserve the advantage
generation exactness is strict, not loose contains matching
the donor language path is not degraded on donor-correct examples
no hidden answer channel, candidate-time solver, or renderer shortcut exists
```

## Reject Rules

Reject or demote to diagnostic if:

```text
the answer loop works after core state is zeroed;
typed CE improves internal fields but not LM generation;
span-copy solves only copy tasks and fails non-copy synthesis;
MemoryOS/RAG success hides a weak recursive core;
the model relies on labels, operation names, or schemas unavailable in normal
chat-template inference;
the final answer is computed by external code and only formatted by the LM.
```

## Shortest Next Path

1. Freeze the canonical path: prompt tokens -> donor/token states -> mandatory
   TRM core -> core-dependent readout -> LM logits.
2. Remove or gate shortcuts where answer-state recurrence can bypass the core
   trajectory.
3. Add same-run destructive ablations: `core_off`, `core_state_zero`,
   `answer_recurrent_off`, and path-specific off modes.
4. Pass a tiny held-out raw-recursive gate before another long integrated
   training run.
5. Only after the core-to-logits path is causal, promote MSA/LM2 memory,
   MemoryOS retrieval, donor annealing, or agentic loops as extensions.

## Relationship To ASI Ambition

This document does not claim QTRM is ASI. It defines the minimum orthodox path
for making an ASI-relevant model-architecture claim: the model must improve
raw reasoning through its own recurrent latent state and produce the final
answer through the normal LLM generation path.

## 2026-05-11 Implementation Update

Implemented the first orthodox answer-loop constraint:

```text
config:
  answer_state_loop_core_state_only_enabled

effect:
  visible text/donor states may be used as the query into core trajectory
  state, but answer-state cross-attention no longer appends raw text states as
  value/context when this mode is enabled.

intended pressure:
  the answer residual hidden state must be formed from core trajectory deltas,
  then mapped to LM logits.
```

Added tests proving the new mode keeps raw text out of answer-loop cross-value
context. The previous behavior appended workspace/core state plus raw text
states; in the minimal unit test that meant context length `10` instead of the
strict core/workspace length `4`.

Added strict candidate configs:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_
typed_value_fullpath_scalar_codec_core_state_only_s060.yaml

configs/qwen35_2b_4090_source_copy_pointer_renderer_core_state_only_scaffold.yaml
```

Updated the mixed non-copy L4 sufficient gate default to use the self-contained
typed-value core-state-only candidate rather than the previous source-copy
delta checkpoint whose base chain was missing.

Smoke:

```text
command:
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py
    --max-cases 1
    --chunk-size 1
    --max-length 192
    --max-new-tokens 8
    --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
      mixed_noncopy_typed_core_state_only_smoke_1case

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  l4_sufficient_onecase_overfit/train_eos_s020/last.pt

report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_typed_core_state_only_smoke_1case/report.json
```

Result:

```text
decision: rejected_noncopy_lm_gate
full_generation_accuracy: 0/1
donor_generation_accuracy: 0/1
core_off_generation_accuracy: 0/1
core_state_zero_generation_accuracy: 0/1
answer_recurrent_off_generation_accuracy: 0/1
```

Important observation:

```text
full completion:             66666666
core_state_zero completion:  55555555
answer_recurrent_off:        00000000
target:                      600054
```

The strict path now runs and ablations perturb the output surface, but the
model still fails the mixed non-copy answer. This is not an accepted L4 result.
It is a cleaner rejection: the remaining blocker is scalar/list accumulator to
LM-logit synthesis under the core-state-only constraint.

Training smoke:

```text
command:
  .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py
    --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_
      typed_value_fullpath_scalar_codec_core_state_only_s060.yaml
    --data-jsonl data/eval/pure_recursive_transition_joint_dynamic_halt_v3_
      mixed_composition_len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl
    --init-checkpoint /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
      l4_sufficient_onecase_overfit/train_eos_s020/last.pt
    --steps 1
    --depth-steps 8
    --target-mode final
    --max-length 192
    --target-logit-positions-only
    --causal-prefix-supervision
    --causal-prefix-max-target-tokens 8
    --causal-prefix-skip-leading-whitespace-targets
    --causal-prefix-append-eos-target
    --final-path-only-supervision
    --answer-state-loop-logit-ce-weight 1.0
    --final-logit-ce-weight 1.0
    --depth-final-ce-weight 0.0
    --progress-margin-weight 0.0
    --lr 1e-5
    --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
      smoke_core_state_only_causal_prefix_s1
```

Observed:

```text
saved:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_core_state_only_causal_prefix_s1/last.pt

final_path_ce: 6.9826
answer_state_loop_logit_ce: 6.9826
final_path_acc: 0.1429
answer_state_loop_logit_acc: 0.1429
causal_prefix_examples: 7
core_steps: 8
```

This confirms the strict core-state-only path is trainable with causal-prefix
EOS supervision. It does not prove recovery; the next required experiment is a
short multi-step overfit/recovery run followed by the same 1-case strict gate.

## 2026-05-11 Gate-Contrast Training Update

Added trainer support for strict gate ablation contrast on the final LM path:

```text
--core-state-zero-final-contrast-weight
--core-state-zero-final-contrast-margin
--core-state-zero-final-contrast-all-prefix-tokens

--answer-state-recurrent-final-contrast-weight
--answer-state-recurrent-final-contrast-margin
--answer-state-recurrent-final-contrast-all-prefix-tokens
```

Principle:

```text
full target logp must beat the same forward pass with:
  zero_core_trajectory=True
  disable_answer_state_loop_recurrent=True

This trains the same destructive ablations that the strict L4 runner uses.
It is still normal LM-logit supervision, not a sidecar solver.
```

Smoke:

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_core_state_only_gate_contrast_s001

observed:
  final_path_ce: 2.8354
  final_path_acc: 0.4286
  core_state_zero_final_target_logp_delta: 9.2229
  answer_state_recurrent_final_target_logp_delta: 1.9025
```

Strict 1-case gate after the smoke:

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_gate_contrast_s001_gate_1case

decision: rejected_noncopy_lm_gate
target:   600054
full:     00000000
donor:    10000
core_off: !!!!!!!!
zero:     55555555
```

Greedy-margin retry:

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_gate_contrast_greedy_s010

gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_gate_contrast_greedy_s010_gate_1case

decision: rejected_noncopy_lm_gate
full:     00000000
typed_value_answer_bridge_off: 60060060
zero:     Vega5555555
```

Interpretation:

```text
The core trajectory is not inert: zeroing it changes the generated surface.
However, the full path remains stuck in a repeated-zero attractor. The typed
value answer bridge is not yet canonical; in this smoke it appears to suppress
the closer typed-bridge-off output rather than improve final synthesis.

Do not promote typed bridge, renderer, or scalar codec components unless
strict generation improves and the same-run ablations prove the component is
causal for the correct LM answer.
```

## 2026-05-11 Answer-Next-Token-Decoder Gate-Contrast Update

Ran focused contrast runs adding strict final-path pressure against
`disable_answer_state_loop_next_token_decoder`:

```text
command (s020):
  .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py
    --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_core_state_only_s060.yaml
    --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_train40000_v0to5_mixed_only.jsonl
    --init-checkpoint /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_sufficient_onecase_overfit/train_eos_s020/last.pt
    --steps 20
    --depth-steps 8
    --target-mode final
    --max-length 192
    --target-logit-positions-only
    --causal-prefix-supervision
    --causal-prefix-max-target-tokens 8
    --causal-prefix-skip-leading-whitespace-targets
    --causal-prefix-append-eos-target
    --final-path-only-supervision
    --answer-state-loop-logit-ce-weight 1.0
    --final-logit-ce-weight 1.0
    --depth-final-ce-weight 0.0
    --progress-margin-weight 0.0
    --lr 1e-5
    --out-dir .../smoke_core_state_only_gate_contrast_s020_nexttok
    --core-state-zero-final-contrast-weight 0.1
    --core-state-zero-final-contrast-margin 0.05
    --core-state-zero-final-contrast-all-prefix-tokens
    --answer-state-recurrent-final-contrast-weight 0.1
    --answer-state-recurrent-final-contrast-margin 0.05
    --answer-state-recurrent-final-contrast-all-prefix-tokens
    --answer-next-token-decoder-final-contrast-weight 0.1
    --answer-next-token-decoder-final-contrast-margin 0.05
    --answer-next-token-decoder-final-contrast-all-prefix-tokens

command (s040):
  same as above with only
    --answer-next-token-decoder-final-contrast-weight 0.5
    --answer-next-token-decoder-final-contrast-margin 0.1
    --answer-next-token-decoder-final-contrast-all-prefix-tokens
  and recurrent/core-state-zero final-contrast weights 0.0

--all others unchanged from s020 run
```

Observed gate outcomes:

```text
out:
  .../mixed_noncopy_core_state_only_gate_contrast_s020_nexttok_gate_1case

decision: rejected_noncopy_lm_gate
full_generation_accuracy: 0.0
full_minus_answer_next_token_decoder_off: 0.0
```

and

```text
out:
  .../mixed_noncopy_core_state_only_gate_contrast_s040_nexttok_gate_1case

decision: rejected_noncopy_lm_gate
full_generation_accuracy: 0.0
full_minus_answer_next_token_decoder_off: 0.0
```

Conclusion:

```text
strict final-path contrast against both `core_state_zero` and
`disable_answer_state_loop_next_token_decoder` is now wired and exercised by
the same-run gate, but these short contrast-only retrains did not recover
non-copy exact synthesis. The bottleneck remains downstream of these current
path ablations.
```

## 2026-05-11 Orthodox Status Reset

Status:

```text
active level: L2/L3 prerequisite repair
roadmap target: L4/general-LM promotion
promotion status: blocked
blocking condition: recurrent core state has not yet produced the final
  non-copy answer through strict autoregressive LM generation.
```

The orthodox architecture is now defined as:

```text
chat-template prompt / compiled context
-> tokenizer
-> token embeddings or frozen donor hidden states
-> mandatory recurrent TRM/QTRM latent core
-> core-state-only answer/readout path
-> LM logits
-> greedy/beam autoregressive text
```

Non-canonical shortcuts remain excluded from promotion:

```text
MemoryOS/RAG as a separate model path
external verifier/solver computing the answer
```

## 2026-05-11 Next-Token Decoder Ablation Gate Update

Executed a strict 1-case mixed-noncopy gate run after adding
`answer_next_token_decoder_off` as a required destructive ablation mode:

```text
command:
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py
    --max-cases 1
    --chunk-size 1
    --max-length 192
    --max-new-tokens 8
    --out-dir /tmp/qtrm_mixed_noncopy_nexttok_gate_smoke
```

Outcome:

```text
decision: rejected_noncopy_lm_gate
full_generation_accuracy: 0.0
full_minus_answer_next_token_decoder_off: 0.0
reject_reasons include:
  - full_generation_accuracy_below_min
  - full_does_not_beat_donor
  - full_does_not_beat_core_off
  - answer_next_token_decoder_off_drop_below_min
```

Follow-up smoke validation with a resolvable config/checkpoint pair completed
all required modes end-to-end (all command exit codes 0):

```text
command:
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
    --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_core_state_only_s060.yaml \
    --checkpoint /tmp/qtrm_nexttok_contrast_s008/last.pt \
    --out-dir /tmp/qtrm_nexttok_smoke_check2 \
    --max-cases 1 \
    --chunk-size 1 \
    --max-length 192 \
    --max-new-tokens 8 \
    --min-full-accuracy 0.0 \
    --min-donor-margin 0.0 \
    --min-core-off-margin 0.0

decision: rejected_noncopy_lm_gate
full_generation_accuracy: 0.0
full_minus_answer_next_token_decoder_off: 0.0
reject_reasons:
  - full_does_not_beat_donor
  - full_does_not_beat_core_off
  - answer_next_token_decoder_off_drop_below_min
```

I then ran a short contrast-focused retrain and re-gated with the same strict
1-case contract:

```text
training:
  .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
    --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_\
typed_value_fullpath_scalar_codec_core_state_only_s060.yaml \
    --steps 8 \
    --answer-next-token-decoder-final-contrast-weight 0.5 \
    --answer-next-token-decoder-final-contrast-margin 0.04 \
    --answer-next-token-decoder-final-contrast-all-prefix-tokens \
    --out-dir /tmp/qtrm_nexttok_contrast_s008

eval:
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
    --checkpoint /tmp/qtrm_nexttok_contrast_s008/last.pt \
    --max-cases 1 \
    --chunk-size 1 \
    --max-length 192 \
    --max-new-tokens 8 \
    --out-dir /tmp/qtrm_nexttok_gate_after_s008
```

Outcome remained reject; strict generative exactness is unchanged (`0/1` full and
all ablation modes). The blocker is still non-copy latent-to-greedy synthesis,
not just missing ablation wiring.
Latest diagnostic findings:

```text
core-state-only gate-contrast + greedy-margin:
  target=600054
  full=00000000
  typed_value_answer_bridge_off=60060060
  core_state_zero=Vega5555555
  decision=rejected_noncopy_lm_gate

stripped+EOS gold-token rank probe:
  target tokens: 6 0 0 0 5 4 <eos>
  full ranks:    1 1 1 1 6 4 10
  finding: first digits are partly reachable, but tail 5/4/EOS is unstable.

self-rollout continuation:
  final_path_acc reached 0.5714 on the supervised slice,
  but strict generation still produced 00000000.

beam probe:
  beam_size=8 also missed the answer and produced 60000000.

Follow-up contrast contrast weight sweep on the same strict 1-case contract:

- `/tmp/qtrm_nexttok_contrast_s004`: `answer-next-token-decoder-final-contrast-weight=0.5`, eval rejected, `full_minus_answer_next_token_decoder_off=0.0`
- `/tmp/qtrm_nexttok_contrast_s004_w1`: `answer-next-token-decoder-final-contrast-weight=1.0`, eval rejected, `full_minus_answer_next_token_decoder_off=0.0`
- `/tmp/qtrm_nexttok_margin_s004`: final margin path (`--final-greedy-token-margin-weight 1.0 --final-greedy-token-margin 0.5`), eval rejected, `full_minus_answer_next_token_decoder_off=0.0`
```

Conclusion:

```text
The current failure is not only greedy decoding. Beam search also misses the
correct scalar tail, and teacher-forced metrics do not transfer to final text.
The active bottleneck is latent-state-to-autoregressive text synthesis under
the mandatory core-state-only path.
```

Allowed next moves:

```text
1. Tail-token/EOS weighting or scheduled/self-rollout loss on the canonical LM
   path, followed by the same strict generation gate.
2. If the typed value bridge remains worse than typed_bridge_off, demote it and
   test a simpler KISS core-state-only readout.
3. If two more LM-path repairs fail, reset to the smallest prior-backed
   recurrent-core-to-text reproduction before adding MSA, MemoryOS, larger
   donors, or online distillation.
```

## 2026-05-11 Tail-Weight And KISS Readout Results

Tail/EOS-weighted continuation:

```text
train:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_tail_weight_s005

strict gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_tail_weight_s005_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  target=600054
  donor=10000
  core_off=!!!!!!!!
  full=40000000
  typed_value_answer_bridge_off=44444444
  core_state_zero=Vega  555555
  answer_recurrent_off=00000000

rank probe, stripped target + EOS, full:
  target tokens: 6 0 0 0 5 4 <eos>
  ranks:         8 1 1 1 3 2 5
```

Interpretation:

```text
Tail/EOS weighting did not repair the tail. It damaged the first digit: the
gold first token 6 fell to rank 8 and the model generated 40000000. This loss
family should not be extended as an orthodox repair.
```

KISS no-typed-bridge config:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_
  core_state_only_kiss_answer_loop_s040.yaml

change:
  keep core + core-state-only answer_state_loop;
  disable typed_algorithmic_value_state and typed bridge;
  trainable policy = core_and_answer_state_loop.
```

No-train A/B from the self-rollout checkpoint:

```text
strict gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_no_train_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  target=600054
  full=60060060
  core_state_zero=!!!!!!!!
  answer_recurrent_off=00000000
```

This is the closest current surface output and it has visible destructive
ablation sensitivity, but exact accuracy is still zero and every reported
accuracy/drop threshold fails.

KISS 5-step continuation:

```text
train:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_kiss_answer_loop_s005

strict gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_s005_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  full=00000000
  core_state_zero=!!!!!!!!
  answer_recurrent_off=00000000
```

Conclusion:

```text
The best immediate simplification is the no-train KISS readout, not another
short CE continuation. Training the current answer-loop objective collapses the
closer 60060060 surface back to 00000000.

The next orthodox move is a root reset of the recurrent-core-to-next-token
readout objective: reproduce a minimal prior-backed recurrent decoder/readout
that learns autoregressive next-token synthesis from latent state under
on-policy or scheduled-sampling conditions, then port only that minimal
mechanism back into QTRM.

Update (2026-05-12):
- `answer_next_token_decoder` final-path contrast is now wired in training and gate.
- L4 mixed non-copy gate now includes `answer_next_token_decoder_off` as a required
  destructive ablation path with threshold tracking.
- New scoped config:
`configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_answer_loop_next_token_decoder_s040.yaml`.
```
