# Past-Success Doubt Report

Old selected/oracle arithmetic wins are search-verifier evidence, not general LM ability. Preserve the causal ingredient, not the overclaim.

In short: selected/oracle wins are useful search evidence, not general LM ability.

## Old Success Rows

| Label | Metric Family | Exact Metric | Proves | Does Not Prove |
| --- | --- | --- | --- | --- |
| Stage56_K128 | selected_oracle_search | selected=0.7682, oracle=0.7747, packed=0.9935 | candidate search plus verifier selection works on a compact synthetic arithmetic/state-space exam | free language generation, multilingual ability, or one-body general LM reasoning |
| Stage58B_K64_top3 | selected_oracle_search | selected=0.9336, oracle=0.9401, packed=0.9935 | candidate search plus verifier selection works on a compact synthetic arithmetic/state-space exam | free language generation, multilingual ability, or one-body general LM reasoning |

## Current Rows

| Label | Metric Family | Exact Metric | Proves | Does Not Prove |
| --- | --- | --- | --- | --- |
| Stage94_raw_byte_82M | teacher_forced_loss | eval_loss 6.5724 -> 2.1387 | heldout CE fell under teacher forcing on this data contract | free generation, candidate selection, depth scaling, or general reasoning |
| Stage94_BLT2_rawteacher_1200 | teacher_forced_loss | eval_loss 6.3111 -> 2.2999 | heldout CE fell under teacher forcing on this data contract | free generation, candidate selection, depth scaling, or general reasoning |

## Required Comparison Row

| old_success | exact_metric | causal_ingredient | missing_in_current_run | smallest_restoration_test |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Recommended Comparison Row

| old_success | exact_metric | causal_ingredient | missing_in_current_run | smallest_restoration_test |
| --- | --- | --- | --- | --- |
| Stage58B_K64_top3 | selected=0.9336, oracle=0.9401, packed=0.9935 | candidate diversity plus verifier-selected compact answers | free generation samples; selected-vs-oracle split on the normal one-body answer path; candidate diversity/coverage if search is claimed; depth or recurrent-core-off ablation | Run a small one-body language gate that logs candidate coverage, selected-vs-oracle accuracy, free generation samples, and recurrent/depth-off loss deltas on the same heldout rows. |

Launch recommendation: `do_not_launch_long_run_until_restoration_gate_exists`

Do not launch a long run until that row is filled.
