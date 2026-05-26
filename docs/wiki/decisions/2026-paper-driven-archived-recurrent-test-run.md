# 2026 Paper-Driven Archived Recurrent Test Run

## Scope

This run exercises the paper-driven test plan against the strongest local
evidence currently available in this workspace.

It is not a Stage93/913M result. The Stage93 checkpoint paths documented for
DGX live under `/mnt/data4tb`, and that mount/checkpoint is not present in the
current local environment.

## Inputs

Archived recurrent eval artifact:

```text
/mnt/nvme1n1p2/qtrm-archive/local_eval/
  qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/
    causal_forced_choice_smoke4_eval_gate.jsonl
    generation_smoke8_eval_gate.jsonl
```

Local report outputs:

```text
local_eval/20260524_paper_driven_tests/
  pure_recursive_reasoning_gate.md
  pure_recursive_reasoning_gate.json
  depth_breadth_report_from_archived_causal_smoke4.json
  vpo_candidate_diversity_from_archived_rows.json
```

## EqR / Raw Recurrent Gate

Command:

```bash
PYTHONPATH=src python scripts/191_build_raw_intelligence_gate.py \
  --gate-type pure_recursive_reasoning \
  --eval-jsonl /mnt/nvme1n1p2/qtrm-archive/local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/causal_forced_choice_smoke4_eval_gate.jsonl \
  --markdown-out local_eval/20260524_paper_driven_tests/pure_recursive_reasoning_gate.md \
  --json-out local_eval/20260524_paper_driven_tests/pure_recursive_reasoning_gate.json
```

Result:

```text
status = rejected

passed:
  deep_core_beats_core_off
  deep_core_beats_donor
  no_retrieval_or_memoryos_shortcut

failed:
  no_depth_scaling_gain
  depth_outputs_identical_across_steps
```

Metrics:

```text
donor_only_no_evidence:        0/4
qtrm_core_off_no_evidence:     0/4
qtrm_core_steps_4_no_evidence: 4/4
qtrm_core_steps_8_no_evidence: 4/4
```

Interpretation:

```text
This archived smoke shows that enabling the core helps over donor/core-off on
the four forced-choice cases, but it does not prove depth scaling. Depth 4 and
depth 8 produce identical outputs on all comparable cases.
```

## EqR Depth/Breadth Report

Command:

```bash
PYTHONPATH=src python scripts/548_build_depth_breadth_probe_report.py \
  --rows /mnt/nvme1n1p2/qtrm-archive/local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/causal_forced_choice_smoke4_eval_gate.jsonl \
  --out local_eval/20260524_paper_driven_tests/depth_breadth_report_from_archived_causal_smoke4.json
```

Result:

```text
status = inconclusive

failed:
  rows_missing_depth
  no_residual_records
  no_depth_trajectory_gain
```

Depth ladder from rows with explicit depth:

```text
depth 4: 4 trajectories, 4 hits, accuracy 1.000
depth 8: 12 trajectories, 8 hits, accuracy 0.667
```

Interpretation:

```text
The archive does not contain fixed-point/convergence residual telemetry, so it
cannot test the EqR attractor claim. The next real EqR run must log depth,
restart_id, completion, hit, and fixed_point_residual or convergence_residual.
```

Implementation update:

```text
scripts/192_eval_raw_intelligence.py now exports EqR-compatible recurrent
telemetry for forced-choice and causal-forced-choice rows when the model output
contains core_depth_states.

Exported fields promoted to the top-level row:
  residual_curve
  fixed_point_residual
  core_fixed_point_residual
  mean_fixed_point_residual
  fixed_point_residual_observations
```

Verification:

```bash
PYTHONPATH=src python -m unittest tests.test_raw_intelligence_eval_script
PYTHONPATH=src python tests/test_depth_breadth_probe.py
python -m py_compile scripts/192_eval_raw_intelligence.py tests/test_raw_intelligence_eval_script.py src/qtrm_mm/eval/depth_breadth_probe.py scripts/548_build_depth_breadth_probe_report.py
```

Result:

```text
tests.test_raw_intelligence_eval_script: 16 tests OK
tests/test_depth_breadth_probe.py:       3 tests OK
py_compile:                             OK
```

## VPO Candidate-Diversity Smoke

Command:

```bash
python - <<'PY'
# one-off candidate diversity aggregation over archived JSONL rows
PY
```

Output:

```text
local_eval/20260524_paper_driven_tests/vpo_candidate_diversity_from_archived_rows.json
```

Result:

```text
causal_forced_choice_smoke4:
  case_count = 4
  mean_candidates_per_case = 6.0
  mean_unique_completions_per_case = 3.0
  oracle_accuracy = 1.0

generation_smoke8:
  case_count = 8
  mean_candidates_per_case = 6.0
  mean_unique_completions_per_case = 4.0
  oracle_accuracy = 0.0
```

Interpretation:

```text
The candidate set has forced-choice oracle coverage, but visible generation has
zero oracle hits and degenerate completions such as repeated punctuation,
repeated "Answer", and repeated digit fragments. This supports the existing
renderer/generation warning: candidate diversity alone is not useful unless the
normal answer channel can produce selectable correct candidates.
```

## MemoryOS / PEEK / MeMo Baseline Tests

Commands:

```bash
PYTHONPATH=src python tests/test_memory_retrieval_eval.py
PYTHONPATH=src python tests/test_memory_eval_script.py
```

Results:

```text
tests/test_memory_retrieval_eval.py: 20 tests OK
tests/test_memory_eval_script.py:     39 tests OK
```

Interpretation:

```text
The existing MemoryOS scoring/retrieval/eval-script layer is green. This is only
the baseline. It does not yet prove a PEEK-style persistent context map, because
there is not yet a context-map mode with map budget, stale-map correction, and
repeated-query metrics.
```

## Next Required Test

The next run must use a real Stage93 or current recurrent checkpoint with fresh
rows shaped like:

```json
{
  "case_id": "case-id",
  "depth": 8,
  "restart_id": 3,
  "completion": "answer",
  "hit": true,
  "fixed_point_residual": 0.0012
}
```

Minimum matrix:

```text
depth:      1, 2, 4, 8, 16
restarts:   0..7
modes:      full, core_off, residual_head_off, state_frozen
metrics:    accuracy, unique completion count, residual top-1 accuracy,
            majority vote accuracy, oracle@k, tokens/sec, VRAM
```

Plain-language verdict:

```text
We have tested the archived recurrent evidence and the MemoryOS baseline layer.
The archived core signal is real but not depth-scaling. The EqR attractor test
is still pending because the local environment lacks the Stage93 checkpoint and
the archive lacks convergence residual telemetry.
```
