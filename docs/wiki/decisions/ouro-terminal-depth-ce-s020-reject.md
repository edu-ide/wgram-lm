# Ouro Terminal-Depth CE S020 Reject

Status: rejected probe, 2026-05-06.

## Question

After subtract-tail counterfactual training, `core_steps=4` scored 4/8 while
`core_steps=8` regressed to 3/8. This suggested a possible terminal-depth
overshoot problem: once the latent transition trace becomes final, later
recursive depths should keep the final answer instead of drifting.

## Implementation

Added train-only terminal-depth CE:

```text
--terminal-depth-ce-weight
```

Mechanism:

```text
Use row.transition_finality_targets to build a per-depth mask.
Apply answer-token CE only to depths where finality == 1.
Do not force nonterminal depths to emit the final answer.
```

This keeps the universal LLM path:

```text
prompt -> donor hidden states -> recursive core / answer-state loop
-> LM logits
```

## Artifacts

```text
runner:
  scripts/244_run_qtrm_ouro_terminal_depth_ce_s020.sh

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_terminal_depth_ce_s020_from_tail_s020/last.pt
  deleted after rejection to recover local disk; eval JSON artifacts retained.

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_terminal_depth_ce_s020_from_tail_s020/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_terminal_depth_ce_s020_from_tail_s020/action_code_eval32.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_terminal_depth_ce_s020_from_tail_s020/tail_error_summary_smoke8.json
```

## Result

Held-out smoke8 causal forced-choice:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
```

Action-code gate:

```text
exact:        32/32
step_acc:     1.0000
finality_acc: 1.0000
halted_exact: 32/32
```

Training telemetry confirmed that terminal CE was active:

```text
terminal_depth_ce:    3.6091
terminal_depth_acc:   0.5000
terminal_depth_count: 2.0000
```

## Decision

Reject as canonical.

The full model preserves the accepted 4/8 score, but bridge-off also reaches
4/8. The terminal-depth CE therefore does not prove that the transition bridge
or recursive terminal signal is causally improving raw answer accuracy.

## Next Hypothesis

The answer path still uses the correct action/finality trace only weakly. The
next architecture direction should make terminal readout causality explicit
rather than adding more target CE:

```text
finality-gated answer-state update:
  while nonterminal: update answer state from next recursive state
  once terminal: freeze or select the terminal answer state
```

This differs from the rejected post-hoc finality selector because the gate must
act inside the answer-state recurrence, not only select a depth after the whole
loop has already run.
