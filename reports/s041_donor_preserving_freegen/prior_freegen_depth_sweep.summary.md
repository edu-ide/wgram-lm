# Prior S040 Free Generation Depth Sweep Summary

Source: `local_eval/necessary_condition_smoke/local_lora_mythos_free_generation_depth_sweep_smoke8.jsonl`

## Aggregate

- Records: 48
- Modes: 6
- Hits: 0/48

## Per-Mode Results

| Mode | Depth | QTRM scale | Donor scale | Hits | Exact | Avg tokens | Top completion | Collapse flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| donor_only_no_evidence | - | 0.0 | 1.0 | 0/8 | 0/8 | 6.38 | `50004` | low_diversity:2 |
| qtrm_core_off_no_evidence | - | - | - | 0/8 | 0/8 | 8.00 | `!!!!!!!!` | bang_loop:8, low_diversity:8 |
| qtrm_core_steps_1_no_evidence | 1 | 1.0 | 0.0 | 0/8 | 0/8 | 8.00 | `Answer:Answer:Answer:Answer:` | answer_loop:8 |
| qtrm_core_steps_2_no_evidence | 2 | 1.0 | 0.0 | 0/8 | 0/8 | 8.00 | `Answer:Answer:Answer:Answer:` | answer_loop:8 |
| qtrm_core_steps_4_no_evidence | 4 | 1.0 | 0.0 | 0/8 | 0/8 | 8.00 | `1600000` | numeric_attractor_1600000:8 |
| qtrm_core_steps_8_no_evidence | 8 | 1.0 | 0.0 | 0/8 | 0/8 | 8.00 | `1600000` | numeric_attractor_1600000:8 |

## Interpretation Contract

This report is a smoke diagnostic, not a promotion gate. A mode can only be promoted if it improves free generation over donor-only, keeps delta/core ablations causal, and does not replace the donor language path with a private renderer.
