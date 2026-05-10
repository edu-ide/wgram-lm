# Transition Joint Reverse Composition Prompt Context

Date: 2026-05-07

Status: active experiment ledger.

Update 2026-05-08: the reverse split was cleaned after finding hidden surface
overlap in list-to-arithmetic variants. Earlier best partial results from the
old split are now historical diagnostics only, not canonical proof.

## Failure

The accepted mixed-composition checkpoint solves the original
list-to-arithmetic order, but fails the reverse arithmetic-to-list order.

Baseline on
`data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_eval80000_v6to7_len1113.jsonl`:

- overall: `32/64`
- `mixed_list_arithmetic`: `32/32`
- `mixed_arithmetic_list`: `0/32`
- reverse prediction pattern: `[0,1,2,3,4,4,4,4]`
- reverse target pattern: `[0,2,3,1,1,4,4,4]`

The model had learned a canonical action sequence, not a prompt-conditioned
composition order.

## Local Fixes Tested

1. `transition_state_joint_only` S240 without prompt context:
   - held-out reverse: `32/64`
   - reverse family: `0/32`
   - decision: reject

2. `core_and_answer_state_loop` S160 on interleaved reverse data:
   - held-out reverse: `0/64`
   - old order regressed from `32/32` to `0/32`
   - decision: reject

3. `transition_state_joint_only` S240 with prompt-context conditioning:
   - held-out reverse: `32/64`
   - train reverse: `0/144`
   - decision: reject as insufficient exposure/pressure

4. `transition_state_joint_only` S1000 with prompt-context conditioning and
   `mixed_repeat=4`:
   - held-out reverse overall: `32/64`
   - old order: `32/32`
   - reverse family: `0/32`
   - reverse step accuracy improved from `0.50` to `0.6914`
   - decision: partial, not accepted

5. Prompt-context augmented S4500:
   - held-out reverse overall: `59/64`
   - old order: `32/32`
   - reverse family: `27/32`
   - decision: best partial checkpoint, still reject because reverse exact is
     below `64/64`

6. Prompt token-attention residual S5500:
   - held-out reverse overall: `26/64`
   - old order regressed to `0/32`
   - reverse family: `26/32`
   - decision: reject; token attention improved the reverse family relative to
     the original baseline but destroyed the old-order policy, so prompt
     attention alone is not a stable fix

7. Primitive operation head S1000 from the S4500 partial checkpoint:
   - evaluation source=`primitive`, 128-case holdout:
     - overall: `60/128`
     - `mixed_arithmetic_list`: `56/64`
     - `mixed_list_arithmetic`: `4/64`
   - evaluation source=`joint` on the same checkpoint:
     - overall: `113/128`
     - `mixed_arithmetic_list`: `49/64`
     - `mixed_list_arithmetic`: `64/64`
   - decision: partial; primitive operation factorization helps reverse order
     but destroys the old-order policy, so it cannot be the sole transition
     source

8. Source-router-only S500 from primitive checkpoint:
   - router train CE converged, but held-out routed eval was identical to
     joint: `113/128`
   - source split showed old order routed to joint, but most reverse rows also
     routed to joint
   - decision: reject; mean prompt context is too weak to choose primitive for
     reverse order

9. Source router with prompt-token attention S800:
   - held-out routed eval: `88/128`
   - `mixed_arithmetic_list`: `56/64`
   - `mixed_list_arithmetic`: `32/64`
   - source split: reverse rows all routed to primitive, but many old rows also
     routed to primitive
   - decision: reject; token attention flips the failure mode and over-routes
     long old-order prompts to primitive

10. Source router with mean+token prompt context S800 and length-augmented
    training lengths `5,7,9,15,17`:
    - held-out routed eval on lengths `11,13`: `87/128`
    - `mixed_arithmetic_list`: `55/64`
    - `mixed_list_arithmetic`: `32/64`
    - source split: reverse mostly primitive, but len13 old-order rows still
      routed to primitive
    - decision: reject; the source-router design is brittle under length/OOD
      transfer and should not be promoted

11. Operation residual inside the joint transition policy:
    - architecture:
      `primitive operation probabilities -> zero-init linear residual -> joint
      transition logits`
    - residual-only S800 on length-augmented train data:
      - scale `1.0`: `96/128`; reverse `64/64`, old `32/64`
      - scale `0.5`: `120/128`; reverse `63/64`, old `57/64`
    - joint S400 fine-tune from the residual checkpoint:
      - best eval scale `0.45`: `123/128`
      - reverse: `62/64`
      - old order: `61/64`
    - decision: best partial, still reject. This is the first candidate that
      improves both orders within the canonical joint transition path, but it
      still misses exact acceptance.

12. Hard-range joint fine-tune from the best partial checkpoint:
    - training data start index shifted to `174000`, still excluding eval
      variants `6,7`
    - held-out eval scale `0.45`: `102/128`
    - reverse: `58/64`
    - old order: `44/64`
    - decision: reject and delete checkpoint. More value-range exposure alone
      worsened transfer; the remaining issue is not solved by simple data
      broadening.

13. Transition-state code residual from the best partial checkpoint:
    - architecture:
      `transition_state_code probabilities -> zero-init residual -> joint
      transition logits`
    - trainable params: code head, code embedding, code residual
    - training: S800 with code CE + joint CE
    - held-out eval: `64/128`
    - reverse: `0/64`
    - old order: `64/64`
    - decision: reject and delete checkpoint. Code residual collapsed to the
      old canonical order and erased the operation-residual reverse gain.

14. Joint order-contrast fine-tune from the best partial checkpoint:
    - loss: target joint logit vs opposite-order joint logit margin
    - training: S200, low LR, joint CE weight `0.2`, order contrast weight `1.0`
    - held-out eval: `119/128`
    - reverse: `62/64`
    - old order: `57/64`
    - decision: reject and delete checkpoint. The hand-defined opposite-order
      contrast was not enough to fix the remaining held-out cases and slightly
      worsened the best partial checkpoint.

15. Hidden split audit and clean split rebuild:
    - bug: `_rewrite_mixed_case` had only 8 list-to-arithmetic surface variants,
      while train requested variants `8..23`; modulo wrapping leaked train
      surfaces into eval variants `6,7`.
    - fix: added real list-to-arithmetic variants `8..23`.
    - clean train:
      `data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_train74000_v0to5_8to23_len5791517_cases16_repeat3.jsonl`
    - clean eval:
      `data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_eval84000_v6to7_len1113_cases16.jsonl`
    - verified variant overlap: none.
    - decision: previous `123/128` is demoted to historical/leaky diagnostic.

16. Clean primitive operation retrain from the accepted len11/13 joint-only
    baseline:
    - checkpoint:
      `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_clean_primitive_s1000_from_jointonly/last.pt`
    - primitive eval: `93/128`
    - reverse `mixed_arithmetic_list`: `61/64`
    - old `mixed_list_arithmetic`: `32/64`
    - joint eval on same checkpoint: `64/128`, reverse `0/64`, old `64/64`
    - decision: useful diagnostic. Primitive operation factorization transfers
      to clean reverse, but it is not sufficient as the canonical joint policy.

17. Clean operation residual retrain from the clean primitive checkpoint:
    - checkpoint:
      `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_clean_opres_s0800_from_clean_primitive/last.pt`
    - train joint CE converged with joint train accuracy `1.0`.
    - held-out clean eval:
      - scale `0.45`: `64/128`, reverse `0/64`, old `64/64`
      - scale `0.8`: `64/128`, reverse `0/64`, old `64/64`
      - scale `1.0`: `64/128`, reverse `0/64`, old `64/64`, finality
        regressed to `32/128`
    - decision: reject. A linear residual from primitive operation
      probabilities to joint logits fits train but does not carry clean
      order/phase transfer.

18. Clean joint+operation-residual fine-tune:
    - checkpoint:
      `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_clean_joint_opres_s0400_from_clean_opres/last.pt`
    - held-out clean eval: `32/128`
    - reverse: `0/64`
    - old order: `32/64`
    - checkpoint weights deleted; logs/eval retained.
    - decision: reject. Reopening the joint head after the clean residual does
      not repair held-out reverse transfer.

19. Clean phase residual from the clean primitive checkpoint:
    - mechanism:
      `transition phase logits + core depth states -> zero-init MLP residual
      -> joint transition logits`
    - first single-stage phase+residual train: `57/128`, reverse `0/64`,
      old `57/64`; checkpoint weights deleted.
    - phase+joint fine-tune from that checkpoint:
      - best clean partial: `99/128`
      - `mixed_arithmetic_list`: `55/64`
      - `mixed_list_arithmetic`: `44/64`
      - finality exact: `121/128`
    - residual scale sweep showed a brittle switch:
      - scale `0.8`: old `64/64`, reverse `0/64`
      - scale `1.0`: old `44/64`, reverse `55/64`
      - scale `1.2`: old `0/64`, reverse `64/64`
    - decision: keep as best clean partial, still reject. The phase residual
      can flip policies but does not compose both policies stably.

20. Centered/gated phase residual variants:
    - centered phase residual: `63/128`, reverse `43/64`, old `20/64`;
      checkpoint weights deleted.
    - non-reference gated phase residual: `78/128`, reverse `60/64`, old
      `18/64`; checkpoint weights deleted.
    - two-stage phase-only then phase-residual-only:
      - eval: `50/128`
      - `mixed_arithmetic_list`: `50/64`
      - `mixed_list_arithmetic`: `0/64`
      - checkpoint weights deleted.
    - phase classifier audit:
      - phase-only checkpoint: `625/1024` phase steps on clean eval.
      - phase+joint best partial: `976/1024` phase steps on clean eval.
    - decision: reject. Freezing a phase classifier before residual training is
      not enough; the held-out order signal must remain causal while the joint
      transition is optimized.

21. Global prompt-query phase latent:
    - mechanism:
      one learned query attends over prompt token states, producing a global
      order latent broadcast to every recursive depth.
    - held-out clean eval: `40/128`
    - `mixed_arithmetic_list`: `0/64`
    - `mixed_list_arithmetic`: `40/64`
    - checkpoint weights deleted.
    - decision: reject. Reading a cleaner global order latent does not by
      itself fix the transition policy; the failure is in how order conditions
      the recursive transition update.

22. Factorized primitive action + finality/halt head:
    - mechanism:
      `core depth state -> primitive operation logits` and
      `core depth state -> explicit finality logits`.
    - held-out clean primitive eval: `58/128`
    - `mixed_arithmetic_list`: code exact `58/64`, finality exact `0/64`
    - `mixed_list_arithmetic`: code exact `0/64`, finality exact `64/64`
    - halted exact: `0/128`
    - checkpoint weights deleted.
    - decision: reject. Action and halt separated cleanly, but they specialize
      to opposite composition orders. The order signal must condition the
      state transition itself, not only a separate readout head.

23. Full recurrent core reopen with factorized action+halt:
    - mechanism:
      train `core.*`, primitive operation head, and explicit finality head
      together from the clean primitive checkpoint.
    - trainable parameters: about `32.8M`.
    - training became unstable late in the run; primitive operation CE spiked
      on logged samples.
    - held-out clean primitive eval: `0/128`
    - `mixed_arithmetic_list`: code step accuracy `0.555`, finality exact
      `0/64`
    - `mixed_list_arithmetic`: code step accuracy `0.744`, finality exact
      `0/64`
    - checkpoint weights deleted.
    - decision: reject. Reopening the full core has too much blast radius and
      destroys the already-useful primitive policy. The next root candidate
      must condition the recurrent transition through a smaller adapter or
      gated context path rather than updating all core weights at once.

24. Narrow core-context adapter with factorized action+halt:
    - mechanism:
      train only `core.context_cross_l/h`, `core.context_gate_l/h`, primitive
      operation head, and explicit finality head.
    - trainable parameters: about `3.3M`.
    - training remained more stable than full-core reopen.
    - held-out clean primitive eval: `49/128`
    - `mixed_arithmetic_list`: code exact `48/64`, finality exact `0/64`
    - `mixed_list_arithmetic`: code exact `1/64`, finality exact `64/64`
    - halted exact: `29/128`
    - checkpoint weights deleted.
    - decision: reject. A small context adapter preserves primitive behavior
      better than full-core training but still does not make order-conditioned
      transition causal enough. The key missing mechanism is feedback from
      predicted/teacher-forced transition state into the next recurrent step.

## Prior Work Checked On 2026-05-08

- CLRS Algorithmic Reasoning Benchmark, ICML 2022:
  <https://proceedings.mlr.press/v162/velickovic22a.html>
  - relevance: algorithmic reasoning models use intermediate hints as part of
    recurrent algorithm execution, not only final-output supervision.
- Recursive Algorithmic Reasoning, PMLR 2024:
  <https://proceedings.mlr.press/v231/jurss24a/jurss24a.pdf>
  - relevance: predicted hints can be aggregated with processed features and
    algorithm inputs to form the next recurrent step; stack-like state is used
    when recursion requires push/pop memory.
- Neural Algorithmic Reasoning without Intermediate Supervision, NeurIPS 2023:
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf>
  - relevance: warns that fixed hint imitation can be suboptimal, but still
    uses representation regularization to align executions; this suggests QTRM
    should avoid blind hint CE and instead make hints causally affect state.
- Unlocking OOD Generalization in Transformers via Recursive Latent Space
  Reasoning, arXiv 2025:
  <https://huggingface.co/papers/2510.14095>
  - relevance: combines adaptive recurrence, algorithmic supervision,
    discrete latent bottlenecks, and error correction for OOD algorithmic
    generalization.
- Thinking Deeper, Not Longer, arXiv 2026:
  <https://arxiv.org/abs/2603.21676>
  - relevance: depth-recurrent Transformers need stable recurrence mechanisms
    such as identity-biased recurrence and LayerScale; full-core reopening
    without stability controls is consistent with the observed collapse.

## Architecture Change

Added optional prompt-conditioned transition joint logits:

```text
prompt text -> donor/prelude hidden states -> masked prompt summary
core depth states + prompt summary projection -> transition joint logits
```

This keeps the universal LLM causal path. It does not add a hidden answer
channel or external solver; the prompt summary is derived from the same token
stream used by the model.

The projection is zero-initialized, so older checkpoints initially preserve
their previous transition policy.

## Root Hypothesis

The reverse-composition failure is not a MemoryOS or retrieval issue. It is a
raw recursive action-policy failure: the transition policy lacks enough causal
pressure to bind prompt-specified operation order to depth-specific action
states.

If longer prompt-context training still fails, the next architecture candidate
should stop adding scalar heads and move to a structured but causal operation
state:

```text
prompt tokens -> recursive core -> operation/state-factorized transition latent
-> joint action/finality logits -> answer path
```

The operation/state factorization must remain internal and learned. It must not
execute the operation outside the model.

The source-router experiments sharpen this hypothesis: selecting between two
separate transition policies is less stable than making the canonical joint
transition policy itself prompt-conditioned and operation-factorized. A router
can memorize which path works on training distributions, but it is not a robust
raw-intelligence improvement unless held-out length/order transfer stays
correct under ablation.

The operation-residual experiment initially looked promising, but the clean
split invalidated it as a canonical fix. The clean primitive head still shows
that the donor/core features contain enough information to classify reverse
operations, while the joint path fails to preserve the prompt-specified
composition order.

The phase and factorized-action experiments sharpen the root hypothesis:
separate order readers, residual switches, routers, primitive heads, and
finality heads can each solve one side of the distribution, but they do not
force the recursive latent state transition to update under the selected
composition order. The next candidate should therefore make the order/phase
latent part of the recurrent transition state update before the action readout:

```text
prompt tokens -> order/phase latent
previous recurrent state + order/phase latent -> next recurrent state
next recurrent state -> action/finality logits
```

This phase latent must be ablatable and must not compute the final answer. It
is accepted only if the canonical transition path improves on clean held-out
order variants and order-latent-off drops.

## 25. Core Transition Feedback Recurrent Hint

Status: rejected.

Experiment:

- config: `configs/qwen35_2b_4090_pure_recursive_transition_feedback_action_halt_s120.yaml`
- checkpoint dir: `local_eval/qwen35_2b_pure_recursive_transition_feedback_action_halt_s0400_from_clean_primitive`
- init: `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_clean_primitive_s1000_from_jointonly/last.pt`
- trainable policy: `core_transition_feedback_and_readouts`
- trainable params: `1,202,843`
- storage fix: `--save-trainable-only` saved `qtrm_trainable_delta_v1` instead of rewriting frozen donor weights

Change:

```text
core state at step t -> operation/finality feedback logits
soft operation/finality prediction -> gated feedback vector
feedback vector -> first core latent token for step t+1
```

Result on clean held-out reverse gate:

```text
exact_rows: 64/128
mixed_arithmetic_list: 64/64
mixed_list_arithmetic: 0/64
finality_exact_rows: 64/128
halted_exact_rows: 0/128
```

Decision:

- reject canonical promotion;
- delete the checkpoint weights;
- preserve logs and eval JSON for the failure ledger.

Interpretation:

The recurrent feedback path can learn on training examples, but predicted
feedback alone still collapses to a family-specific policy on held-out reverse
composition. This does not beat the current canonical partial baseline
(`99/128`). The next architecture candidate should test supervised/teacher-
forced recurrent hint feedback or scheduled feedback so the recurrent state is
first forced to carry the correct transition hint before relying on its own
predictions.

## 26. Teacher-Forced Core Transition Feedback

Status: rejected, but diagnostically useful.

Experiment:

- config: `configs/qwen35_2b_4090_pure_recursive_transition_feedback_action_halt_s120.yaml`
- checkpoint dir: `local_eval/qwen35_2b_pure_recursive_transition_feedback_teacherforced_action_halt_s0400_from_clean_primitive`
- init: clean primitive checkpoint
- added training flag: `--core-transition-feedback-teacher-forcing`
- storage: delta checkpoint, `qtrm_trainable_delta_v1`

Change:

During training only, the recurrent feedback update consumes gold primitive
operation/finality hints while the predicted feedback heads are still trained.
At evaluation, no gold hints are available, so the model uses its own predicted
feedback.

Primitive readout result:

```text
exact_rows: 81/128
mixed_arithmetic_list: 51/64
mixed_list_arithmetic: 30/64
finality_exact_rows: 64/128
halted_exact_rows: 30/128
```

Core-feedback readout diagnostic:

```text
exact_rows: 64/128
mixed_arithmetic_list: 0/64
mixed_list_arithmetic: 64/64
```

Decision:

- reject canonical promotion because it remains below the current `99/128`
  partial baseline and far below the `128/128` acceptance gate;
- keep the small delta checkpoint temporarily because it is the first feedback
  variant that gives non-zero exact rows on both composition families.

Interpretation:

Teacher forcing confirms the root hypothesis direction: correct recurrent
hints improve family balance (`64/128 -> 81/128`). However, the model still
does not learn a robust self-generated hint policy. The feedback head itself is
more family-biased than the primitive readout, so promoting feedback logits as
the answer readout is rejected.

## 27. Teacher-Forced Then Self-Feedback Finetune

Status: rejected.

Experiment:

- init: teacher-forced feedback checkpoint above
- finetune: 200 steps, no teacher forcing, `lr=1e-4`
- checkpoint dir:
  `local_eval/qwen35_2b_pure_recursive_transition_feedback_teacherforced_then_self_s0200_from_teacherforced`

Result:

```text
exact_rows: 61/128
mixed_arithmetic_list: 47/64
mixed_list_arithmetic: 14/64
finality_exact_rows: 64/128
halted_exact_rows: 14/128
```

Decision:

- reject;
- delete checkpoint weights;
- preserve log/eval JSON.

Interpretation:

Simple scheduled handoff from gold feedback to self feedback causes regression.
This suggests the remaining blocker is not only exposure bias. The core needs a
more direct recurrent transition objective or a state representation whose
operation-order variable is identifiable and stable under ablation.

## 28. Prompt-Derived Core Order Bottleneck

Status: rejected.

Architecture change:

- added `core_transition_order_bottleneck_enabled`;
- prompt tokens are read by a learned query/cross-attention bottleneck;
- the resulting 2-class order latent is inserted into the core input as a
  causal latent token;
- primitive prompt-context readout was disabled in the strict variant so the
  readout cannot solve the task by bypassing the recursive core.

Strict bottleneck-only experiment:

- config:
  `configs/qwen35_2b_4090_pure_recursive_transition_order_bottleneck_action_halt_s120.yaml`
- checkpoint dir:
  `local_eval/qwen35_2b_pure_recursive_transition_order_bottleneck_strict_s0400_from_clean_primitive`
- trainable policy: `core_transition_order_bottleneck_and_readouts`
- result:

```text
exact_rows: 0/128
step_acc: 0.7500
mixed_arithmetic_list: 0/64 exact, 0.8750 step_acc
mixed_list_arithmetic: 0/64 exact, 0.6250 step_acc
finality_exact_rows: 0/128
```

Core-open experiment:

- config:
  `configs/qwen35_2b_4090_pure_recursive_transition_order_bottleneck_core_action_halt_s120.yaml`
- checkpoint dir:
  `local_eval/qwen35_2b_pure_recursive_transition_order_bottleneck_core_s0400_from_clean_primitive`
- trainable policy: `core_and_transition_order_bottleneck_and_readouts`
- result:

```text
exact_rows: 74/128
step_acc: 0.8564
mixed_arithmetic_list: 58/64 exact, 0.9883 step_acc
mixed_list_arithmetic: 16/64 exact, 0.7246 step_acc
finality_exact_rows: 86/128
halted_exact_rows: 76/128
```

Decision:

- reject both variants;
- delete rejected checkpoint weights and preserve logs/eval JSON;
- keep the architecture code as a diagnostic/prototype path because it proves
  prompt-derived order can be placed on the core causal path, but it does not
  yet solve reverse composition.

Interpretation:

The strict variant proves that a single prompt-derived order token is not
enough for a frozen core. Opening the core improves exactness from `0/128` to
`74/128`, so the frozen-core hypothesis is partly supported. However it remains
below the current `99/128` partial baseline and strongly over-selects the
`mixed_arithmetic_list` transition sequence. In the failed family, many
`mixed_list_arithmetic` rows are decoded as the opposite operation order.

Next hypothesis:

The order variable must condition the recurrent update at each step, not only
exist as an extra token that attention may ignore. The next candidate is
order-conditioned recurrent step conditioning: derive the order latent from the
prompt, then inject it into each recursive step together with the step embedding
so path-off/order-shuffle ablations can test causality.

## 29. Order-Conditioned Recurrent Step Conditioning

Status: rejected.

Architecture change:

- added `core_transition_order_step_conditioning_enabled`;
- the prompt-derived order latent is passed into `QTRMRecursiveCore`;
- at every outer recursive step, the core adds a gated normalized order vector
  to both `z_l` and `z_h`, so order can affect the recurrent update directly
  rather than relying on attention to an appended token.

Base step-conditioned experiment:

- config:
  `configs/qwen35_2b_4090_pure_recursive_transition_order_step_conditioned_core_action_halt_s120.yaml`
- checkpoint dir:
  `local_eval/qwen35_2b_pure_recursive_transition_order_step_conditioned_core_s0400_from_clean_primitive`
- trainable policy: `core_and_transition_order_bottleneck_and_readouts`
- result:

```text
exact_rows: 83/128
step_acc: 0.8936
mixed_arithmetic_list: 51/64 exact, 0.9746 step_acc
mixed_list_arithmetic: 32/64 exact, 0.8125 step_acc
finality_exact_rows: 96/128
halted_exact_rows: 83/128
```

Hard-family finetune:

- init: base step-conditioned checkpoint above
- curriculum: `family_repeat=mixed_list_arithmetic=4,mixed_arithmetic_list=1`
- checkpoint dir:
  `local_eval/qwen35_2b_pure_recursive_transition_order_step_conditioned_core_hardlist_s0200_from_s0400`
- result:

```text
exact_rows: 73/128
step_acc: 0.9092
mixed_arithmetic_list: 47/64 exact, 0.9258 step_acc
mixed_list_arithmetic: 26/64 exact, 0.8926 step_acc
finality_exact_rows: 117/128
halted_exact_rows: 73/128
```

Decision:

- reject both checkpoints and delete weights;
- preserve logs/eval JSON;
- keep the code path as a diagnostic because it improves the weak family over
  the previous core-open order-token result (`16/64 -> 32/64`), but it is still
  below the `99/128` partial baseline and far below the `128/128` gate.

Interpretation:

Direct order conditioning at every recurrent step reduces but does not solve
family collapse. Hard-family oversampling increases step accuracy and finality
but reduces exact traces, indicating that simple curriculum weighting trades
one family against the other rather than producing a stable shared algorithm.

Next hypothesis:

The model needs an explicit per-step latent transition target that separates
the operation class from the order family without forcing one canonical
sequence. Candidate: factorized transition state with independent
`operation`, `order`, and `finality` readouts plus an order-conditioned recurrent
state update. Accept only if both families improve together and order-shuffle
ablation drops.

## Acceptance Gate

Accept only if all are true:

- reverse held-out full: `128/128`
- `mixed_list_arithmetic`: `64/64`
- `mixed_arithmetic_list`: `64/64`
- transition-off or code-shuffle drops strongly
- canonical len11/13 regression remains accepted

Partial step-accuracy improvement without exact reverse traces is not accepted.
