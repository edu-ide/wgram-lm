# Role-Filler Variable Slots

Date: 2026-05-05

Purpose: ground the next QTRM value-state candidate in prior work on variable
binding, role/filler representations, and neural algorithmic reasoning. This is
the response to the current failure: the accepted recursive action controller
solves the operation trace, but generic value slots do not bind exact numeric
content.

## Downloaded References

```text
references/papers/role_value_slots/slot_attention_2006.15055.pdf
references/papers/role_value_slots/transformer_variable_binding_wu25.pdf
references/papers/role_value_slots/open_book_neural_algorithmic_reasoning_2024.pdf
references/papers/role_value_slots/basic_reasoning_tpr_1601.02745.pdf
references/papers/role_value_slots/clrs_algorithmic_reasoning_2205.15659.pdf
```

## Prior Notes

- Slot Attention learns a set of exchangeable slots that bind to objects through
  iterative competitive attention. For QTRM, the useful prior is not the vision
  task; it is the idea that a fixed set of slots can bind different input
  entities and support unseen compositions.
- Tensor Product Representations model symbolic reasoning as role/filler
  binding. For QTRM, this motivates stable role embeddings such as
  `input_slot_0`, `doubled_slot_0`, `coefficient`, and `residual`, with values
  predicted through a shared value classifier.
- The 2025 transformer variable-binding analysis shows that transformer
  residual streams can behave as addressable memory for variable dereferencing
  after training, and that causal interventions can verify the mechanism.
- CLRS and Open-Book Neural Algorithmic Reasoning are relevant as evaluation
  pressure: exact intermediate state, held-out values/lengths, distractors, and
  ablations matter more than fluent text for raw intelligence.

## QTRM Mapping

```text
stable role embeddings
  = typed variable addresses

factorized recurrent value slots
  = trainable filler/state reservoir

role cross-attention over value slots
  = neural binding/readout operation

role-value CE
  = scaffold/probe for exact state binding

field/head-off or role-shuffle ablation
  = causal test that the binding path is used
```

This is intentionally less task-ad-hoc than one hard-coded head per arithmetic
field, but more structured than one generic slot vocabulary. It tests whether
QTRM can learn stable role/value binding before moving the value transition
fully into the recurrent core.

## Architecture Implication

The current role-value path is a candidate, not canonical final architecture.

Acceptance requires:

```text
content role accuracy clearly above the generic-slot baseline
trace exact > 0/32 on held-out len7/9 mixed composition
action-code exact remains 32/32
role/value path ablation drops value metrics
```

Rejection means the next architecture should stop adding readout probes and
make role-bound values part of the mandatory recurrent state update itself.

