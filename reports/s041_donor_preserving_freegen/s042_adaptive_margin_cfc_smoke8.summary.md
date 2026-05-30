# S042 Adaptive-Margin Donor-Preserving CFC Smoke8

Source: `reports/s041_donor_preserving_freegen/s042_adaptive_margin_cfc_smoke8.jsonl`

## Aggregate

- Records: 88
- Modes: 11
- Hits: 23/88

## Per-Mode Results

| Mode | Depth | QTRM scale | Donor scale | Hits | Exact | Avg tokens | Top completion | Collapse flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| donor_only_no_evidence | - | 0.0 | 1.0 | 2/8 | 2/8 | 0.00 | `16` | - |
| qtrm_core_off_no_evidence | - | - | - | 0/8 | 0/8 | 0.00 | `__FORCED_CHOICE_TIE__` | - |
| qtrm_core_steps_2_no_evidence | 2 | 1.0 | 0.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_2_qtrm_scale_1_donor_scale_1_no_evidence | 2 | 1.0 | 1.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_2_qtrm_scale_2_donor_scale_1_no_evidence | 2 | 2.0 | 1.0 | 3/8 | 3/8 | 0.00 | `green` | - |
| qtrm_core_steps_4_no_evidence | 4 | 1.0 | 0.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_4_qtrm_scale_1_donor_scale_1_no_evidence | 4 | 1.0 | 1.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_4_qtrm_scale_2_donor_scale_1_no_evidence | 4 | 2.0 | 1.0 | 3/8 | 3/8 | 0.00 | `green` | - |
| qtrm_core_steps_8_no_evidence | 8 | 1.0 | 0.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_8_qtrm_scale_1_donor_scale_1_no_evidence | 8 | 1.0 | 1.0 | 2/8 | 2/8 | 0.00 | `green` | - |
| qtrm_core_steps_8_qtrm_scale_2_donor_scale_1_no_evidence | 8 | 2.0 | 1.0 | 3/8 | 3/8 | 0.00 | `green` | - |

## Interpretation Contract

This report is a smoke diagnostic, not a promotion gate. A mode can only be promoted if it improves free generation over donor-only, keeps delta/core ablations causal, and does not replace the donor language path with a private renderer.
