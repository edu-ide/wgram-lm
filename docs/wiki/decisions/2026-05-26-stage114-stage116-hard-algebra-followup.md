# 2026-05-26 Stage114-116 Hard Algebra Follow-Up

## Decision

Stage114 improves the Stage113 answer-preference route, but Stage115 and
Stage116 show that the remaining algebra wall is no longer solved by more
nearby replay or stronger chosen-answer CE.

```text
Stage113B:
  GD smoke accuracy: 0.700000
  mean margin: 1.232448
  min margin: -0.744675

Stage114 hard-family replay + language preservation:
  GD smoke accuracy: 0.800000
  mean margin: 1.304333
  min margin: -0.595507

Stage115 algebra-only replay:
  GD smoke accuracy: 0.800000
  mean margin: 1.307387
  min margin: -0.462224

Stage116 algebra replay + stronger chosen CE:
  GD smoke accuracy: 0.800000
  mean margin: 1.304350
  min margin: -0.432504
```

Stage114 is the best promoted local checkpoint in this family so far because it
improves GD while preserving language/generation.

## Plain-Language Read

Stage113 taught the model:

```text
Do not blindly copy the repeated answer.
Prefer the answer that follows the rule.
```

Stage114 added:

```text
Practice the hard families more,
but keep reading/speaking normal language while practicing.
```

That helped. CRT flipped from failing to passing, and GD smoke rose to 0.80.

But the remaining algebra rows are a different bottleneck:

```text
The model knows copying 83 or 13 can be wrong,
but it still has not reliably performed the small subtraction:
  89 = 73 + a  -> a = 16
  -17 = -94 + a -> a = 77
```

So the current wall is not "lack of preference pressure." It is a missing
internal calculation routine for algebra under misleading demonstrations.

## Evidence

Stage114 fixed CRT:

```text
intuitive_answer/crt:
  accuracy: 1.000000
  mean margin: 0.155282
  min margin: 0.010564
```

Stage114 preserved small language/generation gates versus Stage113B:

```text
language heldout loss:
  Stage113B: 11.106347
  Stage114:  11.073960

language token accuracy:
  Stage113B: 0.105263
  Stage114:  0.105263

generation prefix token accuracy:
  Stage113B: 0.317073
  Stage114:  0.317073

first response accuracy:
  Stage113B: 0.234375
  Stage114:  0.250000
```

Depth did not solve the remaining algebra wall:

```text
Stage115 depth 4:
  accuracy: 0.800000
  mean margin: 1.307387

Stage115 depth 8:
  accuracy: 0.800000
  mean margin: 1.089566

Stage115 depth 12:
  accuracy: 0.700000
  mean margin: 1.039298
```

The only depth-specific improvement was the numbered algebra variant, but
overall accuracy did not improve.

## Remaining Failed Rows

Stage114 remaining negative rows are all algebra variants:

```text
repetitive_answer/algebra/original:
  target: 16
  parrot: 83
  min margin: -0.502426

repetitive_answer/algebra/v2fmt:
  target: 16
  parrot: 83
  min margin: -0.595507

repetitive_answer/algebra/numbered:
  target: 77
  parrot: 13
  min margin: -0.029528

repetitive_answer/algebra/instruction:
  target: 77
  parrot: 13
  min margin: -0.373402
```

The prompts explicitly create a trap by repeating the wrong answer in the
demonstrations. The model must solve the final equation, not imitate the answer
frequency.

## Implementation Notes

The Stage113 preference trainer now supports:

```text
--focus-tasks
--focus-replay-factor
--language-loss-weight
--language-sampled-data
--language-batch-size
--language-loss-chunk-size
```

The row-id report was changed from huge full row dumps to:

```text
row_id_count
row_ids_sha256
row_id_examples
```

to keep reports readable and avoid bloating logs.

## Consequence

Do not launch another Stage115-like replay-only or Stage116-like CE-only run as
the main move.

The next high-probability move must change the algebra computation route while
staying inside the same LM answer path.

## Next Move

Stage117 should add an internal algebra calculation curriculum, not a side
solver:

```text
prompt with misleading repeated answers
-> same BPE reader
-> recurrent thought state must bind the final equation
-> train on answer-only plus small hidden/calculation contrast
-> same LM head says the final numeric answer
```

Fast local gate:

```text
1. Use heldout smoke rows only for evaluation.
2. Train on generated non-heldout algebra traps:
   repeated wrong answer in demos,
   final equation requiring a = x - y or a = y - x.
3. Keep language CE preservation.
4. Pass only if the four remaining algebra smoke rows flip without losing CRT,
   code tracing, letter counting, successive, truthy, or generation gates.
```

Hard reject:

```text
Do not add an external algebra calculator as the promoted architecture.
Do not score with an oracle selector.
Do not call it accepted if only the train algebra rows improve.
```
