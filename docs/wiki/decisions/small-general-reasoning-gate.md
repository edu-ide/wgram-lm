# Small General Reasoning Gate

Date: 2026-05-08

## Question

Before claiming broader LLM progress, can this path beat donor-only on a small
mixed reasoning gate?

```text
prompt
-> tokenizer
-> frozen Qwen donor hidden states
-> QTRM recursive core
-> explicit state codec
-> donor soft-prefix final answer path
-> autoregressive text
```

The required comparison is:

```text
soft_full_no_evidence > donor_only_no_evidence
soft_full_no_evidence > soft_core_off_no_evidence
soft_full_no_evidence > soft_state_off_no_evidence
```

If `core_off` or `state_off` matches full, the renderer may have learned a
prompt/style shortcut, but the QTRM recursive core or state codec did not
causally improve the answer.

## Implementation

Added:

```text
scripts/308_run_small_general_reasoning_gate.py
```

It builds a mixed train/eval JSONL from multiple reasoning sources, then calls:

```text
scripts/304_train_core_soft_prefix_donor.py
```

The default state codec is the typed algorithmic state:

```text
typed_algorithmic_kind_logits
typed_algorithmic_raw_list_offset_logits
typed_algorithmic_doubled_list_offset_logits
typed_algorithmic_scalar_coeff_logits
typed_algorithmic_scalar_residual_logits
typed_algorithmic_final_residual_logits
```

The gate is also wired into:

```text
scripts/300_research_gate_runner.py --gate small_general_reasoning
```

## Smoke Result

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/300_research_gate_runner.py \
  --profile smoke \
  --gate small_general_reasoning \
  --out-dir local_eval/research_gate_runner/small_general_reasoning_smoke_run
```

Report:

```text
local_eval/research_gate_runner/small_general_reasoning_smoke_run/report.json
```

Result:

```text
decision: rejected
full:      0.00
donor:     0.00
core_off:  0.00
state_off: 0.00
families:  2
```

This only confirms that the gate runs end-to-end.

## L2 Probe Result

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/308_run_small_general_reasoning_gate.py \
  --out-dir local_eval/research_gate_runner/small_general_reasoning_l2_probe \
  --max-train-per-source 4 \
  --max-eval-per-source 2 \
  --max-train-cases 8 \
  --max-eval-cases 4 \
  --soft-prefix-steps 80 \
  --max-new-tokens 8 \
  --append-eos-target \
  --suppress-visible-reasoning-tokens \
  --log-every 20 \
  --min-full-accuracy 0.25
```

Report:

```text
local_eval/research_gate_runner/small_general_reasoning_l2_probe/report.json
```

Result:

```text
decision: rejected
full_generation_accuracy:      0.25
donor_generation_accuracy:     0.00
core_off_generation_accuracy:  0.25
state_off_generation_accuracy: 0.25
full_minus_donor:              0.25
full_minus_core_off:           0.00
full_minus_state_off:          0.00
eval_family_count:             2
```

Family breakdown:

```text
arithmetic_chain:
  full:      1/2
  core_off:  1/2
  state_off: 1/2
  donor:     0/2

mixed_list_arithmetic:
  full:      0/2
  core_off:  0/2
  state_off: 0/2
  donor:     0/2
```

Teacher-forced token accuracy:

```text
soft_full:      0.7083
soft_core_off:  0.6250
soft_state_off: 0.7083
```

## Decision

Reject as L3 candidate.

The path beats donor-only on aggregate, but it does not beat the core/state
ablations:

```text
full == core_off == state_off
```

Therefore the gain is not yet causal to the recursive core or state codec.
The current adapter can learn a small arithmetic answer pattern, but the mixed
list-arithmetic family still fails and the typed state features do not improve
generation over state-off.

## Next Constraint

Do not promote this to a general LLM architecture claim.

The next gate must fix the causal state bottleneck:

```text
1. typed/final scalar state must improve over state_off
2. mixed_list_arithmetic must have at least one held-out hit
3. full must beat donor, core_off, and state_off
4. family coverage must not be arithmetic-only
```

Most likely blocker:

```text
final_residual / scalar_residual state codec is not carrying exact enough value
information into the final answer path.
```

