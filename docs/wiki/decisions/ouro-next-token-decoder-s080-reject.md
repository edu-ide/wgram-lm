# Ouro Next-Token Decoder S080 Reject

Date: 2026-05-07

## Purpose

This run tested the renderer bottleneck directly: instead of adding another
post-hoc donor adapter, put a tokenizer-aligned next-token decoder inside the
answer-state loop before the shared LM head.

Canonical path:

```text
prompt tokens
-> frozen Qwen donor hidden states
-> QTRM recursive answer-state loop
-> in-loop next-token decoder
-> shared LM head
-> autoregressive text
```

No MemoryOS, retrieval, answer side channel, or candidate solver was used.

## Implementation

Code and config:

```text
src/wgram_lm/config.py
src/wgram_lm/wgram_model.py
src/wgram_lm/training/train.py
scripts/192_eval_raw_intelligence.py
scripts/247_probe_qtrm_gold_token_ranks.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080.yaml
scripts/254_run_qtrm_ouro_next_token_decoder_s080.sh
```

New runtime ablation:

```text
qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence
```

New trainable policy:

```text
trainable_param_policy: answer_state_loop_next_token_decoder_only
```

The run started from the accepted answer-halt checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
```

The checkpoint was saved outside the nearly-full root filesystem:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/last.pt
```

## Training Signal

Final telemetry at step 80:

```text
final_path_ce: 3.3559
final_path_acc: 0.4286
final_greedy_token_margin: 0.4388
final_greedy_token_win_rate: 0.4286
causal_prefix_self_rollout_examples: 0
```

The decoder learned some teacher-forced local token signal, but it was trained
without self-rollout correction.

## Evaluation

Artifacts:

```text
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/gold_token_rank_probe4.jsonl
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/generation_smoke8.jsonl
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/causal_forced_choice_smoke4.jsonl
```

Gold-token rank probe on 4 held-out cases:

```text
donor_only:       first_unique@1 0/4, all<=10 3/4, max_rank_mean 9.75
core_off:         first_unique@1 0/4, all<=10 4/4, max_rank_mean 1.00
core8 full:       first_unique@1 4/4, all<=10 2/4, max_rank_mean 12.00
decoder_off:      first_unique@1 4/4, all<=10 0/4, max_rank_mean 740.00
halt_gate_off:    first_unique@1 0/4, all<=10 0/4, max_rank_mean 78.00
```

Generation smoke8:

```text
donor_only:       0/8
core_off:         0/8
core8 full:       0/8
decoder_off:      0/8
halt_gate_off:    0/8
```

Causal forced-choice smoke4:

```text
donor_only:       0/4
core_off:         0/4
core8 full:       0/4
decoder_off:      4/4
halt_gate_off:    0/4
```

Representative generation failures:

```text
expected 300015 -> core8 full completion 2400000
expected 400037 -> core8 full completion 1400000
expected 300015 -> decoder-off completion 1 1 1 1
expected 400051 -> decoder-off completion 1 1 1 1
```

## Decision

Reject as a promoted renderer checkpoint.

The in-loop next-token decoder is not useless: it improves full-sequence
gold-token rank versus decoder-off on the 4-case probe. But the promoted
runtime path still fails both real gates:

```text
greedy generation: 0/8
causal forced-choice: 0/4
```

Worse, disabling the new decoder restores forced-choice to 4/4 under the
accepted answer-halt gate. That means this decoder changes local token ranks
without preserving the sequence-level scoring behavior that made the accepted
halt-head checkpoint valuable.

## Interpretation

The bottleneck is exposure bias plus answer-state/token-state mismatch:

```text
teacher-forced rank can improve
but greedy rollout still leaves the valid answer manifold
and sequence scoring can regress
```

Do not continue blind decoder-only sweeps.

Next falsifier:

```text
keep the in-loop tokenizer-aligned decoder scaffold
add self-rollout causal-prefix correction
train with generated-prefix examples, not only gold prefixes
require: generation_smoke8 > 0/8
require: forced-choice does not regress below the accepted halt-head baseline
require: decoder/core/halt ablation loses the gain
```
