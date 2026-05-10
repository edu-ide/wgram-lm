# List Transform Failure Ledger

Mode: `qtrm_core_steps_8_no_evidence`

## Summary

```text
hits: 0/2
by_error: {'filtered_state_selected': 1, 'reversed_final_selected': 1}
mean_correct_minus_selected_score: -0.4942706780774251
```

## Records

### list-transform-000

```text
hit: False
error_type: filtered_state_selected
completion: 4,2
answer: 8,4
correct_rank: 3
correct_minus_selected_score: -0.9858449101448059
depth_targets: {'1': '4,2', '2': '8,4', '4': '8,4', '8': '8,4'}
```

### list-transform-001

```text
hit: False
error_type: reversed_final_selected
completion: 8,16,4
answer: 4,16,8
correct_rank: 2
correct_minus_selected_score: -0.002696446010044262
depth_targets: {'1': '2,8,4', '2': '4,16,8', '4': '4,16,8', '8': '4,16,8'}
```

