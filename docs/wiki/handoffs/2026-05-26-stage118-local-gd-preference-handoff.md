# 2026-05-26 Stage118 Local GD Preference Handoff

## Agent Prompt

You are taking over the local QTRM / PrefixLM generalization-dynamics work in:

```text
/home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos
```

Continue from the evidence, not from new speculation.

Plain-language state:

```text
The model can read the prompt well enough for the small GD gate.
The current wall is that misleading repeated answers still pull the final
answer path away from small algebra calculation.

Stage113 taught "prefer the rule-following answer over the familiar answer."
Stage114 preserved language while improving the hard families.
Stage117 added generated non-heldout algebra traps and became the current best.
Stage118 has finished training but has not yet been evaluated.
```

## Current Best Checkpoint

Use this as the promoted local anchor until Stage118 proves otherwise:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE117_LOCAL_GENERATED_ALGEBRA_TRAPS_100STEP/last.pt
```

Stage117 heldout GD smoke:

```text
valid rows: 20 / 44
accuracy: 0.850000
mean margin: 1.167116
min margin: -0.455690
accepted: false
```

Language and generation did not regress materially:

```text
language heldout loss: 11.051448
language token accuracy: 0.105263
direct generation first-response accuracy: 0.250000
direct generation exact: 1 / 12
direct generation prefix accuracy: 0.317073
repeated loops: 0
```

Remaining Stage117 failures are algebra trap variants:

```text
repetitive_answer/algebra/original:    target 16, parrot 83, margin -0.340394
repetitive_answer/algebra/v2fmt:       target 16, parrot 83, margin -0.455690
repetitive_answer/algebra/instruction: target 77, parrot 13, margin -0.291939
```

## Stage118 Status

Stage118 was launched as a diagnostic fixed-parrot algebra run and appears to
have completed training.

Output directory:

```text
/mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE118_LOCAL_FIXED_PARROT_ALGEBRA_DIAG_60STEP
```

Files present:

```text
checkpoint_pref_step000030.pt
checkpoint_pref_step000060.pt
last.pt
preference_train_report.json
tensorboard/events.out.tfevents...
```

Training report last row:

```text
step: 60
loss: 3.611039
preference_loss: 0.586966
chosen_ce: 2.061763
language_loss: 2.749326
mean_margin: 0.280351
min_margin: -0.383416
win_rate: 0.750000
```

Do not promote Stage118 until heldout smoke plus language/generation gates are
run. It is diagnostic because the fixed parrot numbers are close to the failed
heldout parrot values.

## Immediate Next Command

First evaluate Stage118. Do not start a new architecture run before this.

```bash
cd /home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos

ROOT_OUT=/mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE118_LOCAL_FIXED_PARROT_ALGEBRA_DIAG_60STEP
PROBE=local_eval/20260526_STAGE107L_LOCAL_ONLINE_OPUS_EFFECT_GDSUITE_SMOKE/official_gdsuite_choice_probe_2pertask.jsonl

for tag in step30 last; do
  if [ "$tag" = step30 ]; then
    CKPT="$ROOT_OUT/checkpoint_pref_step000030.pt"
    STEP=30
  else
    CKPT="$ROOT_OUT/last.pt"
    STEP=60
  fi

  REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
  TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
  .venv/bin/python scripts/624_eval_bpe_generalization_dynamics_probe.py \
    --checkpoint "$CKPT" \
    --probe-jsonl "$PROBE" \
    --device cuda \
    --out "$ROOT_OUT/stage118_${tag}_gdsuite_smoke44.json" \
    --tensorboard-dir "$ROOT_OUT/tensorboard" \
    --tensorboard-prefix "eval/stage118_${tag}_bpe_gdsuite" \
    --tensorboard-step "$STEP"
done
```

Summarize the result with:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE118_LOCAL_FIXED_PARROT_ALGEBRA_DIAG_60STEP')
for name in ['stage118_step30_gdsuite_smoke44.json', 'stage118_last_gdsuite_smoke44.json']:
    p = root / name
    print('\\n', p)
    data = json.loads(p.read_text())
    s = data.get('summary', data)
    print('accuracy', s.get('accuracy'))
    print('mean_margin', s.get('mean_margin'))
    print('min_margin', s.get('min_margin'))
    print('accepted', s.get('accepted'))
    for task, row in sorted(s.get('tasks', {}).items()):
        if not row.get('passed'):
            print('FAIL', task, row)
PY
```

## If Stage118 Improves

If Stage118 reaches at least:

```text
accuracy >= 0.90
mean_margin positive
no new non-algebra failures
```

then run language and generation preservation before claiming progress:

```bash
cd /home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos

ROOT_OUT=/mnt/sdc1/tripleyoung/qtrm_eval/20260526_STAGE118_LOCAL_FIXED_PARROT_ALGEBRA_DIAG_60STEP

REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
.venv/bin/python scripts/544_eval_prefixlm_language_heldout_loss.py \
  --checkpoint "$ROOT_OUT/last.pt" \
  --probe-jsonl data/eval/prefixlm_general_language_heldout.jsonl \
  --device cuda \
  --max-cases 8 \
  --out "$ROOT_OUT/language_heldout_stage118_8.json" \
  --tensorboard-dir "$ROOT_OUT/tensorboard" \
  --tensorboard-prefix eval/stage118_language \
  --tensorboard-step 60

REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas \
.venv/bin/python scripts/539_eval_prefixlm_generation_gate.py \
  --checkpoint "$ROOT_OUT/last.pt" \
  --probe-jsonl data/eval/prefixlm_general_language_generation_probe.jsonl \
  --device cuda \
  --max-cases 12 \
  --max-new-tokens 24 \
  --out "$ROOT_OUT/generation_stage118_direct12.json" \
  --tensorboard-dir "$ROOT_OUT/tensorboard" \
  --tensorboard-prefix eval/stage118_generation \
  --tensorboard-step 60
```

Promotion rule:

```text
Stage118 can replace Stage117 only if GD improves and language/generation do
not materially regress. Otherwise Stage117 remains the anchor.
```

## If Stage118 Does Not Improve

Do not keep tuning fixed-parrot Stage118. Treat it as a diagnostic reject.

The likely read is:

```text
The model does not need more exposure to the exact wrong answer token.
It needs a stronger normal answer path for binding the final equation and
performing the small calculation before the LM head speaks.
```

The next high-probability move should be a real route change, not another
stronger replay or stronger CE variant.

Candidate next route:

```text
Stage119: one-body equation-state readback

same BPE prompt reader
-> recurrent state must reconstruct the final equation fields
-> same hidden state must prefer the solved answer
-> same LM head emits the final answer

No external calculator.
No oracle selector.
No side verifier as the promoted answer path.
```

Fast falsification gate:

```text
1. Train only on generated non-heldout algebra traps.
2. Evaluate the same heldout 44-row GD smoke.
3. Require all algebra variants to flip without hurting already-passed tasks.
4. Require language/generation preservation.
```

## Do Not Do

Hard rejects for the next agent:

```text
Do not pivot back to BLT/tokenizer-free as the main explanation for this
specific GD algebra wall.

Do not restart from scratch or DGX pretraining to answer this local question.

Do not add a side calculator/verifier and call it the model's answer path.

Do not overclaim old 78% selected/oracle synthetic results as general LLM
reasoning or as proof that this path is solved.

Do not launch another Stage115/116-style "more replay / stronger CE" run unless
it is explicitly labeled a cheap diagnostic.

Do not write new checkpoints under repo root; root filesystem is nearly full.
Use /mnt/sdc1/tripleyoung/qtrm_eval.
```

Disk status at handoff:

```text
/      about 1.6G free, effectively full
/mnt/sdc1 about 9.9G free
```

## Relevant Files

Training / eval scripts:

```text
scripts/625_train_bpe_gd_preference.py
scripts/624_eval_bpe_generalization_dynamics_probe.py
scripts/626_build_algebra_trap_preference_probe.py
scripts/544_eval_prefixlm_language_heldout_loss.py
scripts/539_eval_prefixlm_generation_gate.py
```

Tests:

```text
tests/test_bpe_gd_preference_train.py
tests/test_stage117_algebra_trap_probe.py
```

Decision docs:

```text
docs/wiki/decisions/2026-05-26-stage113-bpe-gd-preference-restoration.md
docs/wiki/decisions/2026-05-26-stage114-stage116-hard-algebra-followup.md
```

Need to add after evaluation:

```text
docs/wiki/decisions/2026-05-26-stage117-stage118-generated-algebra-traps.md
```

## Verification Already Run

These passed before handoff:

```bash
.venv/bin/python -m py_compile scripts/625_train_bpe_gd_preference.py scripts/626_build_algebra_trap_preference_probe.py
.venv/bin/python -m unittest tests.test_bpe_gd_preference_train tests.test_stage117_algebra_trap_probe -v
```

## One-Sentence Mental Model

The project is not currently blocked on "more thinking depth" or "new
tokenizer"; it is blocked on making the one-body LM answer path do the small
calculation after reading a misleading prompt, then speak through the same
head without being pulled back into the repeated wrong answer.
