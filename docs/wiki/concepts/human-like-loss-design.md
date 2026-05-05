# Human-Like Loss Design

QTRM should not claim that a loss function makes the model think like a human.
The defensible claim is narrower:

```text
human-like behavior =
  fluent answer imitation
  + repeated-failure avoidance
  + preference for verified corrections
  + uncertainty/search triggers
  + memory of past mistakes
```

## Current Implementation

The current implementation has two layers.

### General Preference Layer

This is the preferred latest-style path for QTRM training:

- dataset rows can use `prompt`, `chosen`, and `rejected`;
- the chosen answer is also used as ordinary SFT supervision;
- the rejected answer is scored by a second model forward pass;
- `sequence_average_logprob` computes the sequence-level average log-prob;
- `simpo_margin_loss` trains:

```text
chosen_logprob - rejected_logprob >= preference_margin
```

- row-level `preference_weight` or `confidence` can down-weight noisy
  preference pairs, matching the robust-preference direction of RPO/ROPO-style
  work.

Config knobs:

- `loss_preference_weight`
- `preference_beta`
- `preference_margin`

Script:

- `scripts/120_run_workspace_evidence_preference_train.sh`
- `scripts/121_eval_preference_pairs.py`

The preference eval gate reports:

- `preference_accuracy`: fraction where `chosen_logp > rejected_logp`;
- `margin_pass_rate`: fraction where
  `chosen_logp - rejected_logp >= preference_margin`;
- weighted variants using `preference_weight` or `confidence`;
- `margin_mean`, `margin_min`, and `margin_max`.

### Narrow Repetition Layer

This is now treated as an optional guard, not the main alignment objective. It
addresses the collapse pattern observed in donor adapter probes:

```text
"Freeze Freeze Freeze ..."
"world of the world of the world ..."
```

The code now has:

- `repetition_unlikelihood_loss(logits, input_ids, labels=...)`
  - penalizes high probability on repeating the previous token when that token
    is not the current supervised target;
  - skips positions where the gold target is a legitimate adjacent repeat;
  - returns zero when every candidate is masked or a legitimate gold repeat.
- `simpo_margin_loss(chosen_logps, rejected_logps)`
  - preference primitive for chosen/rejected answer pairs;
  - uses sequence-level average log-prob margins.
- `TrainConfig.loss_repeat_unlikelihood_weight`
  - default `0.0`;
  - enables conservative repeat-guard experiments without changing old configs.

## Why Start Here

Preference loss is more general than repetition unlikelihood because it can
represent many failure modes:

- wrong answer over correct answer;
- stale evidence over current evidence;
- unsigned evidence over signed evidence;
- guessed answer over `UNKNOWN`;
- verbose but unsupported answer over short grounded answer.

Repetition unlikelihood remains useful only as a local safety valve.

It is intentionally not a hard decoding rule. A hard no-repeat rule can hide a
training failure and can break legitimate repeats. Unlikelihood keeps the model
trainable while lowering probability mass on the known-bad attractor.

## Next Loss Steps

1. Expand preference rows beyond answer pairs:

```json
{
  "prompt": "...",
  "chosen": "verified answer",
  "rejected": "bad generation",
  "preference_weight": 0.8
}
```

2. Add stronger robust weighting:

```text
weight = inferred_label_reliability * source_quality * temporal_validity
```

3. Train with:

```text
CE_answer + SimPO_margin + optional_repeat_UL
```

4. Add process preference:

```json
{
  "prompt": "...",
  "chosen_trace": ["search", "compare sources", "answer with citation"],
  "rejected_trace": ["guess", "answer without evidence"]
}
```

5. Add verifier-weighted preference:

```text
weight = source_quality * temporal_validity * contradiction_resolution
```

## Claim Boundary

This is closer to human learning than plain next-token prediction because it can
learn from mistakes and preferences. It is still not a full human cognition
model. A stronger claim requires:

- behaviorally important memory gates;
- process-level supervision;
- search/verification traces;
- ablation showing QTRM improves over donor-only;
- robustness tests on held-out prompts where repeated text was not in training.

## References

See [Human-Like Loss Design Sources](../sources/human-like-loss-design.md).
