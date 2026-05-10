# Reproducible Gate Preservation Protocol

Date: 2026-05-08

## Problem

The earlier Ouro recurrent-answer smoke success could not be reproduced after
the accepted checkpoint was deleted.

Root cause:

```text
the init checkpoint did not contain the tested recurrent answer block
```

The rebuild run reported:

```text
missing keys: 12
unexpected keys: 5
```

The missing keys were the actual tested component:

```text
answer_state_loop_recurrent_norm.weight
answer_state_loop_recurrent_stack.layers.0.*
answer_state_loop_recurrent_gate.*
```

Therefore the component was randomly initialized. Because the training script
did not previously record a seed, a small `2/8` smoke win could not be exactly
recovered after the checkpoint was lost.

## Rule

No smoke success may be promoted unless these are preserved:

```text
seed
command
config snapshot
init checkpoint SHA256
missing/unexpected load keys
train/eval data paths
step checkpoints
raw eval JSONL/JSON
best.pt
accepted.pt when the gate passes
```

If missing keys include the new component under test, the result is considered
random-init sensitive until repeated seeds or an accepted checkpoint prove it.

## Implementation

Updated:

```text
scripts/196_train_pure_recursive_depth_supervised.py
```

New behavior:

```text
--seed sets python/random, numpy, torch, and CUDA seeds
checkpoints store training_metadata
training_metadata includes seed, command, config, init checkpoint,
train data, missing keys, unexpected keys, and trainable policy
```

Added:

```text
scripts/309_run_validation_gated_ouro_recurrent.py
tests/test_ouro_validation_runner.py
```

The runner uses this pattern:

```text
train with --save-every
evaluate every step checkpoint with causal_forced_choice
choose the best checkpoint by full-vs-baseline margin
hardlink/copy best.pt
hardlink/copy accepted.pt only if the L2 gate passes
write run_manifest.json and report.json
```

Acceptance requires:

```text
full core8 > donor_only
full core8 > core_off
full core8 > answer_state_recurrent_off
action-code exact remains 32/32
```

## Dry-Run Check

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/309_run_validation_gated_ouro_recurrent.py \
  --out-dir local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_dry_run \
  --steps 4 \
  --save-every 2 \
  --eval-max-cases 2 \
  --seed 0 \
  --dry-run
```

Result:

```text
decision: dry_run
manifest written
config snapshot written
init checkpoint SHA256 recorded
train command includes --seed, --save-every, and --save-trainable-only
```

## Smoke Pipeline Check

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/309_run_validation_gated_ouro_recurrent.py \
  --out-dir local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_smoke_seed0 \
  --steps 4 \
  --save-every 2 \
  --eval-max-cases 1 \
  --action-eval-max-cases 4 \
  --seed 0
```

Result:

```text
decision: rejected
best: step_000002
full: 1/1
donor_only: 0/1
core_off: 0/1
answer_state_recurrent_off: 1/1
action-code exact: 4/4
```

This is the intended reject behavior. The full path hit the single smoke case,
but recurrent-off also hit it, so the runner refused to call it a causal
recurrent-answer gain.

Preserved artifacts:

```text
run_manifest.json
report.json
config_snapshot.yaml
step_000002.pt
step_000004.pt
last.pt
best.pt
evals/*causal_forced_choice.jsonl
evals/*action_code_eval.json
```

The step checkpoint metadata now records:

```text
seed: 0
init_missing_keys: 12
init_unexpected_keys: 5
```

The missing keys include the tested answer recurrent block, so future reports
must treat this branch as random-init sensitive unless a seed sweep shows a
stable causal gain.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/196_train_pure_recursive_depth_supervised.py \
  scripts/309_run_validation_gated_ouro_recurrent.py \
  tests/test_ouro_validation_runner.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_ouro_validation_runner \
  tests.test_research_gate_runner
```

Result:

```text
8 tests OK
```

## Next

Use the validation-gated runner instead of direct S80 training:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/309_run_validation_gated_ouro_recurrent.py \
  --out-dir local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0 \
  --steps 40 \
  --save-every 10 \
  --eval-max-cases 4 \
  --seed 0
```

If seed0 rejects, repeat with a small seed sweep before claiming the recurrent
answer path is impossible:

```text
seed 0, 1, 2
```

## 2026-05-08 Correction

The raw-intelligence eval loader had a second reproducibility bug: it loaded
trainable-only checkpoints directly and ignored `base_checkpoint` metadata.
That means non-trainable base weights could be random during evaluation.

Updated:

```text
scripts/192_eval_raw_intelligence.py
tests/test_raw_intelligence_eval_script.py
```

Corrected re-evaluation of the previous Ouro recurrent validation-gated run:

```text
local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0_s40_eval4_fixedload
```

Result:

```text
decision: rejected
step_000010 full=0/4
step_000020 full=0/4
step_000030 full=0/4
step_000040 full=0/4
last       full=0/4
```

Therefore the earlier Ouro recurrent L2 acceptance is invalid. L3 work must
resume from the value-state L2 gate rather than from the Ouro renderer path.

If only one seed passes, preserve it but mark the result unstable. If multiple
seeds pass with the same ablation pattern, reopen the L2 scale-up.
