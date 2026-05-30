# S042 Adaptive-Margin Donor-Preserving Beam Generation Smoke8

Source: `reports/s041_donor_preserving_freegen/s042_adaptive_margin_beam_generation_smoke8.jsonl`

## Aggregate

- Records: 32
- Modes: 4
- Hits: 6/32

## Per-Mode Results

| Mode | Depth | QTRM scale | Donor scale | Hits | Exact | Avg tokens | Top completion | Collapse flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| donor_only_no_evidence | - | 0.0 | 1.0 | 2/8 | 0/8 | 12.00 | `<think>\nWe are asked: "Compute ((7 +` | - |
| qtrm_core_steps_2_qtrm_scale_2_donor_scale_1_no_evidence | 2 | 2.0 | 1.0 | 2/8 | 1/8 | 8.12 | `green` | low_diversity:1, numeric_attractor_1600000:1 |
| qtrm_core_steps_4_qtrm_scale_2_donor_scale_1_no_evidence | 4 | 2.0 | 1.0 | 1/8 | 0/8 | 9.25 | `green` | numeric_attractor_1600000:1 |
| qtrm_core_steps_8_qtrm_scale_2_donor_scale_1_no_evidence | 8 | 2.0 | 1.0 | 1/8 | 0/8 | 9.25 | `green` | numeric_attractor_1600000:1 |

## Interpretation Contract

This report is a smoke diagnostic, not a promotion gate. A mode can only be promoted if it improves free generation over donor-only, keeps delta/core ablations causal, and does not replace the donor language path with a private renderer.
