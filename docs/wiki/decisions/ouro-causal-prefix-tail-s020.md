# Ouro Causal-Prefix Tail S020

Status: accepted smoke probe, 2026-05-06.

## Failure

The mixed list-to-arithmetic gate exposed a specific final-operation failure:

```text
gold:      doubled_sum - offset
model:     doubled_sum
example:   gold 300015, model 300024
```

The existing `choices` already included the pre-subtract sum as a hard
negative, so the problem was not missing evaluation negatives. The likely
training mismatch was that normal answer CE can condition on the full answer
string, while the held-out `causal_forced_choice` metric scores each candidate
from prefix-only conditions.

## Change

Fine-tuned the accepted mixed-repeat bridge checkpoint with causal-prefix
answer-token supervision:

```text
--causal-prefix-supervision
--causal-prefix-max-target-tokens 8
--causal-prefix-later-token-weight 2.0
--choice-margin-mode sequence
--choice-margin-weight 0.5
--transition-joint-answer-bridge-contrast-weight 0.5
--staged-internal-sequence-ce-weight 0.25
```

This makes later answer tokens, including the digits affected by the final
subtract step, receive direct prefix-conditioned pressure instead of only
first-token pressure.

## Artifacts

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/lm_causal_forced_choice_smoke8.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/tail_error_summary_smoke8.json
```

## Result

Baseline-inclusive held-out smoke8:

```text
donor_only:       0/8
core_off:         0/8
bridge_off:       3/8
full core8:       4/8
action-code:     32/32
finality_acc:     1.0000
halted_exact:    32/32
```

Tail-error breakdown:

```text
donor_only:
  doubled_list:       8

core_off:
  forced_choice_tie:  8

bridge_off:
  correct_final:      3
  pre_subtract_sum:   4
  forced_choice_tie:  1

full core8:
  correct_final:      4
  pre_subtract_sum:   4
```

Previous mixed-repeat smoke:

```text
bridge_off:       2/8
full core8:       3/8
```

## Decision

Accept as a small but meaningful smoke improvement.

This is the first mixed-composition checkpoint in this sequence where the
universal LLM path beats donor-only, core-off, and bridge-off simultaneously on
the same smoke slice while preserving the transition/action controller.

Do not claim a broad raw-intelligence breakthrough yet. The score is still only
4/8, and the remaining misses are all the same final subtract-tail class.

## Next

```text
1. Promote causal-prefix sequence supervision to the default for numeric and
   symbolic forced-choice probes.
2. Add a stricter tail-negative gate that separately reports:
   correct_sum_wrong_tail vs correct_final_answer.
3. Expand from smoke8 to smoke32 only after saving-space hygiene is enforced.
4. Consider applying bridge contrast to all causal-prefix answer tokens, not
   only the first token, if the tail misses persist.
```

## Storage Note

The first run failed at checkpoint save because `/` was full. Removed generated
`local_eval/**/step_*.pt` snapshots and kept `last.pt` checkpoints plus eval
JSON artifacts.
