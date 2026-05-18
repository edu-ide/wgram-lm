# QTRM LLM Wiki Log

## [2026-05-12] research | DeltaNet official implementation boundary

Searched and cloned the official DeltaNet/GatedDeltaNet reference stack:

```text
references/official/gated-deltanet
  upstream: https://github.com/NVlabs/GatedDeltaNet
  commit: b53d6d3a161267432a79c1c04af69fa52bddc921

references/official/flash-linear-attention
  upstream: https://github.com/fla-org/flash-linear-attention
  commit: 74d011f1bd58367b7bc6519dbc4d177d29b063e0

references/official/qwen35
  upstream: https://github.com/QwenLM/Qwen3.5
  commit: f1443092c29978643fd041ebe959676259e934f1

references/official/mamba
  upstream: https://github.com/state-spaces/mamba
  commit: a14b1dff0454a3bc27d9eb31355dc01e4b2490ec
```

Decision:

```text
The current `qtrm_hybrid_3to1` backbone is Qwen3.5-style, not a strict official
Qwen3.5 implementation. Keep it as a candidate/probe only.
```

Why:

```text
Current code uses local `QTRMBlockStack` plus local `torch_gated_delta`
fallback. Official promotion requires FLA/NVlabs GatedDeltaNet or the
Transformers Qwen3.5 modules in strict mode, plus the Qwen3.5 config schedule:
3 x linear_attention followed by 1 x full_attention.
```

Qwen3.5 3:1 terminology correction:

```text
The local Qwen3.5-2B config says:

  full_attention_interval: 4
  layer_types:
    linear_attention
    linear_attention
    linear_attention
    full_attention
    ...

Therefore describe the 3:1 schedule as:

  3 x GatedDeltaNet-like linear_attention
  1 x full_attention

not as:

  GatedDeltaNet + gated attention

unless a separate official model source proves that the full_attention layer is
implemented as a gated-attention variant.
```

Mamba-3 boundary:

```text
Mamba-3 is present in the official state-spaces/mamba repo:

  references/official/mamba/mamba_ssm/modules/mamba3.py
  class Mamba3

It is an SSM candidate, not a GatedDeltaNet or full-attention replacement by
name. It may be compared later as:

  MHA ETD encode -> Mamba-3 recurrent think -> MHA ETD decode -> LM head

but only after the accepted official-FLA thinking-core placement is scaled.
Promotion requires the same seed-stability and language non-regression gates
plus strict official-kernel/runtime evidence.
```

Mamba-3 direct verification update:

```text
script:
  scripts/342_qtrm_native_l5d_backbone_compare.py

candidate added:
  official_mamba3_think

candidate path:
  MHA ETD encode -> official Mamba-3 recurrent think -> MHA ETD decode -> LM head

smoke artifact:
  local_eval/qtrm_native_l5d_mamba3_compare_smoke

command:
  PYTHONPATH=src .venv/bin/python scripts/342_qtrm_native_l5d_backbone_compare.py \
    --profile smoke \
    --out-root local_eval/qtrm_native_l5d_mamba3_compare_smoke \
    --candidates mha_etd,official_fla_think,official_mamba3_think \
    --seed 337 \
    --eval-seed 9337

result:
  official_mamba3_think: command_failed
  returncode: 1

failure:
  the official Mamba-3 module can be imported through the local reference
  adapter, but the Triton SISO kernel fails to compile at the QK skip-connection
  tl.dot site under the current local toolchain.

interpretation:
  This does not prove GatedDeltaNet is intrinsically better than Mamba-3.
  It proves only that the accepted official_fla_think path is currently
  executable/scored, while official_mamba3_think is blocked before scoring.
```

Mamba-3 runtime fix and short comparison:

```text
root cause:
  official Mamba-3 requires triton>=3.5.0
  current .venv has triton 3.3.1 from torch 2.7.1

non-destructive local runtime:
  local_deps/mamba3_runtime
  triton==3.5.1 installed there only

micro-forward:
  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python ...
  OfficialMamba3Mixer CUDA forward OK

short artifact:
  local_eval/qtrm_native_l5d_mamba3_compare_short_triton351

short command:
  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python \
    scripts/342_qtrm_native_l5d_backbone_compare.py \
    --profile short \
    --out-root local_eval/qtrm_native_l5d_mamba3_compare_short_triton351 \
    --candidates mha_etd,official_fla_think,official_mamba3_think \
    --seed 337 \
    --eval-seed 9337

winner:
  official_mamba3_think

full_generation_exact:
  mha_etd:                 0.015625
  official_fla_think:      0.036458333333333336
  official_mamba3_think:   0.046875

official_fla_think:
  backend_ok: true
  causal_ok: true
  full_minus_think0: 0.03125
  full_minus_worst_ablation: 0.010416666666666668

official_mamba3_think:
  backend_ok: true
  causal_ok: true
  full_minus_think0: 0.046875
  full_minus_worst_ablation: 0.010416666666666664

interpretation:
  Single-seed short evidence now favors official_mamba3_think over
  official_fla_think, but this is not canonical promotion until seed-stability
  and language non-regression pass under the same standards.
```

Mamba-3 short seed-stability sweep:

```text
artifact:
  local_eval/qtrm_native_l5d_mamba3_seed_sweep_short_triton351

command:
  PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python \
    scripts/343_qtrm_native_l5d_placement_seed_sweep.py \
    --profile short \
    --out-root local_eval/qtrm_native_l5d_mamba3_seed_sweep_short_triton351 \
    --candidates mha_etd,official_fla_think,official_mamba3_think \
    --target-candidate official_mamba3_think

decision:
  accepted_l5d_placement_seed_stability

target_candidate:
  official_mamba3_think

promoted_count:
  3 / 3

causal_ok_count:
  3 / 3

backend_ok_count:
  3 / 3

min_delta_vs_mha:
  0.005208333333

max_delta_vs_mha:
  0.03125

min_full_generation_exact:
  0.046875

max_full_generation_exact:
  0.052083333333333336

seed 337:
  mha_etd:               0.015625
  official_fla_think:    0.036458333333333336
  official_mamba3_think: 0.046875

seed 338:
  mha_etd:               0.046875
  official_fla_think:    0.052083333333333336
  official_mamba3_think: 0.052083333333333336

seed 339:
  mha_etd:               0.046875
  official_fla_think:    0.015625
  official_mamba3_think: 0.052083333333333336

interpretation:
  The current evidence no longer supports "GatedDeltaNet > Mamba-3". On the
  short L5D synthetic scaffold, official_mamba3_think is seed-stable and at
  least ties or beats official_fla_think across the checked seeds. It still
  needs Mamba-3 language non-regression and scaled-reasoning gates before
  replacing official_fla_think as the broader canonical placement.
```

Next implementation target:

```text
Add an official-backend L5D comparison gate:
  MHA ETD baseline
  qtrm_hybrid_3to1 local candidate
  FLA/NVlabs/Qwen3.5 official-backend candidate

Promotion requires passing the same L4/L5/L5C gates and beating or matching
MHA ETD without weakening ablation criteria.
```

Implemented the first official-backend gate:

```text
runner gate:
  qtrm_native_l5d_official_fla_runtime

script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

backend:
  --backbone qtrm_hybrid_3to1
  --delta-backend fla_gated_delta
  --strict-backends
```

Smoke result:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_official_fla_runtime_smoke

decision:
  accepted_l5d_official_fla_runtime

backend_summary:
  fla_delta_mixers: 9
  official_fla_delta_mixers: 9
  torch_delta_mixers: 0
  all_fla_mixers_official: true
```

Short standard runtime result:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_official_fla_runtime_standard

decision:
  accepted_l5d_official_fla_runtime

backend_summary:
  fla_delta_mixers: 9
  official_fla_delta_mixers: 9
  torch_delta_mixers: 0
  all_fla_mixers_official: true

metrics:
  full_generation_exact: 0.03125
  think0_generation_exact: 0.020833333333333332
  state_reset_generation_exact: 0.010416666666666666
  op_zero_generation_exact: 0.0
```

Interpretation:

```text
This is not a performance claim. It only proves that the official FLA
GatedDeltaNet path is actually used inside QTRM-native with strict backend
wiring. The next L5D task is performance comparison against MHA ETD under the
accepted L4/L5/L5C gates.
```

Added the first comparison harness:

```text
script:
  scripts/342_qtrm_native_l5d_backbone_compare.py

tests:
  tests/test_qtrm_native_l5d_backbone_compare.py
```

Short comparison result:

```text
out_dir:
  local_eval/qtrm_native_l5d_backbone_compare_short

winner:
  official_fla

full_generation_exact:
  mha_etd: 0.015625
  official_fla: 0.026041666666666668

full_exact_delta_official_fla_minus_mha:
  0.010416666667

official_fla_backend_ok:
  true

official_fla_causal_ok:
  false

official_fla_full_minus_think0:
  -0.005208333333333332

official_fla_full_minus_worst_ablation:
  -0.026041666666666668

official_fla_promoted:
  false
```

Interpretation:

```text
The official FLA backend is running and slightly ahead on short exactness, but
it fails the causal ablation standard. Do not promote it over MHA ETD yet.
The next L5D task is to improve official-FLA training or configuration until
full > think0 and full > destructive ablations under the same benchmark.
```

Depth-8 follow-up:

```text
out_dir:
  local_eval/qtrm_native_l5d_official_fla_depth8_short

setup:
  official FLA GatedDeltaNet strict backend
  train_think_steps: 8
  eval_think_steps: 8
  steps: 600
  train_cases: 4096
  eval_cases: 256

metrics:
  full_generation_exact: 0.03515625
  think0_generation_exact: 0.03515625
  state_reset_generation_exact: 0.0390625
  op_zero_generation_exact: 0.03125
  full_minus_think0: 0.0
  full_minus_worst_ablation: -0.00390625
```

Interpretation:

```text
More recurrent depth did not fix official-FLA causal use. The next experiment
should isolate FLA placement: FLA encode/decode with MHA think block, versus
MHA encode/decode with FLA think block.
```

Placement isolation:

```text
shared setup:
  script:
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  steps: 400
  train_cases: 2048
  eval_cases: 192
  train_think_steps: 4
  eval_think_steps: 4
  strict_backends: true
  delta_backend: fla_gated_delta

placement_a:
  out_dir:
    local_eval/qtrm_native_l5d_fla_encode_decode_mha_think_short
  encode_backbone:
    qtrm_hybrid_3to1 / official FLA
  think_backbone:
    mha_etd
  decode_backbone:
    qtrm_hybrid_3to1 / official FLA
  backend:
    official_fla_delta_mixers: 6
    torch_delta_mixers: 0
  metrics:
    full_generation_exact: 0.057291666666666664
    think0_generation_exact: 0.041666666666666664
    state_reset_generation_exact: 0.020833333333333332
    op_zero_generation_exact: 0.036458333333333336
    full_minus_think0: 0.015625
    full_minus_worst_ablation: 0.02083333333333333
    min_family_generation_exact: 0.03125

placement_b:
  out_dir:
    local_eval/qtrm_native_l5d_mha_encode_decode_fla_think_short
  encode_backbone:
    mha_etd
  think_backbone:
    qtrm_hybrid_3to1 / official FLA
  decode_backbone:
    mha_etd
  backend:
    official_fla_delta_mixers: 3
    torch_delta_mixers: 0
  metrics:
    full_generation_exact: 0.08333333333333333
    think0_generation_exact: 0.041666666666666664
    state_reset_generation_exact: 0.036458333333333336
    op_zero_generation_exact: 0.046875
    full_minus_think0: 0.041666666666666664
    full_minus_worst_ablation: 0.03645833333333333
    min_family_generation_exact: 0.0625
```

Interpretation:

```text
The all-FLA candidate still is not promoted. The best strict-official-FLA
placement so far is:

  MHA encode -> official FLA recurrent think -> MHA decode -> LM head

This is important because it keeps the QTRM-native universal LM path and makes
the official FLA block causally useful in the mandatory recursive core. The
next step is a seed-sweep comparison against the current MHA ETD baseline.
```

Seed-stability follow-up:

```text
script:
  scripts/343_qtrm_native_l5d_placement_seed_sweep.py

out_dir:
  local_eval/qtrm_native_l5d_placement_seed_sweep_short

profile:
  short

candidates:
  mha_etd
  official_fla_think

target_candidate:
  official_fla_think

decision:
  accepted_l5d_placement_seed_stability

promoted_count:
  3 / 3

causal_ok_count:
  3 / 3

backend_ok_count:
  3 / 3

min_delta_vs_mha:
  0.005208333333

max_delta_vs_mha:
  0.067708333333

min_full_generation_exact:
  0.052083333333333336

max_full_generation_exact:
  0.08333333333333333
```

Per-seed:

```text
seed 337:
  full_generation_exact: 0.08333333333333333
  delta_vs_mha: 0.067708333333
  full_minus_think0: 0.041666666666666664
  full_minus_worst_ablation: 0.03645833333333333

seed 338:
  full_generation_exact: 0.078125
  delta_vs_mha: 0.03125
  full_minus_think0: 0.06770833333333333
  full_minus_worst_ablation: 0.020833333333333336

seed 339:
  full_generation_exact: 0.052083333333333336
  delta_vs_mha: 0.005208333333
  full_minus_think0: 0.052083333333333336
  full_minus_worst_ablation: 0.005208333333333336
```

Interpretation:

```text
The staged official-FLA thinking-core placement is now seed-stable at short
scale. Promote the L5D placement, not the all-FLA backbone:

  MHA ETD encode -> official FLA GatedDeltaNet recurrent think -> MHA ETD decode

This preserves the native LM causal path and keeps the recursive core
mandatory. Next work should scale this exact placement and test natural-text
language preservation before moving to MSA/LM2 memory gates.
```

Language non-regression follow-up:

```text
runner gate:
  qtrm_native_l5d_placement_language_nonregression

script:
  scripts/336_train_qtrm_native_text_probe.py

out_dir:
  local_eval/qtrm_native_l5d_official_fla_think_language_nonregression_standard

placement:
  MHA ETD encode
  official FLA GatedDeltaNet recurrent think
  MHA ETD decode

decision:
  accepted_l5d_placement_language_nonregression

metrics:
  last_loss: 0.6685409545898438
  think_eval_loss: 1.888507903768466
  think0_loss: 5.344207181380345
  thinking_block_off_loss: 5.344207181380345
  think0_baseline_loss: 2.025272998672265
  full_vs_think0_loss_ratio: 0.35337475507090776
  full_vs_thinking_block_off_loss_ratio: 0.35337475507090776
  full_vs_baseline_loss_ratio: 0.9324707854232689
  sample_unique_chars: 33
  sample_max_run_fraction: 0.022727272727272728
```

Interpretation:

```text
The accepted L5D placement does not collapse the small native autoregressive
language path. It is now both seed-stable on the short reasoning placement
gate and accepted on the staged-placement language non-regression gate.
```

Scaled reasoning follow-up:

```text
runner gate:
  qtrm_native_l5d_placement_scaled_reasoning

script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

out_dir:
  local_eval/research_gate_runner/qtrm_native_l5d_placement_scaled_reasoning_standard

placement:
  MHA ETD encode
  official FLA GatedDeltaNet recurrent think
  MHA ETD decode

setup:
  steps: 1200
  train_cases: 4096
  eval_cases: 384
  d_model: 96
  n_heads: 4
  n_kv_heads: 2
  d_ff: 192
  strict_backends: true

decision:
  accepted_l5d_placement_scaled_reasoning

backend:
  official_fla_delta_mixers: 3
  torch_delta_mixers: 0
  all_fla_mixers_official: true

metrics:
  full_generation_exact: 0.14583333333333334
  think0_generation_exact: 0.0
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.033854166666666664
  full_minus_think0: 0.14583333333333334
  full_minus_worst_ablation: 0.11197916666666669
  min_family_generation_exact: 0.0625

per_family:
  checksum: 0.296875
  modchain: 0.078125
  revchain: 0.0625
```

Interpretation:

```text
The accepted staged placement survives a larger reasoning gate. The full
recurrent path is substantially above think0 and destructive ablations, so the
next work can move to memory/long-context mechanisms without changing the
canonical reasoning core.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_gated_delta_adapter \
  tests.test_research_gate_runner

WRITE_WIKI=0 PYTHONPATH=src bash scripts/338_run_qtrm_native_gate.sh \
  qtrm_native_l5d_official_fla_runtime smoke

WRITE_WIKI=0 PYTHONPATH=src bash scripts/338_run_qtrm_native_gate.sh \
  qtrm_native_l5d_official_fla_runtime standard

PYTHONPATH=src .venv/bin/python scripts/342_qtrm_native_l5d_backbone_compare.py \
  --profile short --out-root local_eval/qtrm_native_l5d_backbone_compare_short
```

## [2026-05-12] evaluation | QTRM-native L5B seed-stable and L5C language non-regression accepted

Confirmed the hard-balanced L5B multi-family seed sweep:

```text
out_dir:
  local_eval/qtrm_native_l5_multifamily_hardbalanced_sweep

decision:
  accepted_l5_seed_stability

summary:
  pass_count: 3 / 3
  min_full_generation_exact: 0.7057291666666666
  max_full_generation_exact: 0.7565104166666666
  min_family_generation_exact: 0.55078125
  max_family_generation_exact: 0.62109375
```

Added an L5C language non-regression gate:

```text
runner gate:
  qtrm_native_l5_language_nonregression

script:
  scripts/336_train_qtrm_native_text_probe.py
  scripts/341_qtrm_native_l5_language_seed_sweep.py

checks:
  full recurrent LM loss vs random threshold
  sample degeneracy
  full recurrent loss vs think0
  full recurrent loss vs thinking_block_off
  full recurrent loss vs separately trained think0 baseline
```

Standard L5C run:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5_language_nonregression_standard

decision:
  accepted_l5_language_nonregression

metrics:
  think_eval_loss: 1.8944039690879084
  think0_loss: 4.506251942726873
  thinking_block_off_loss: 4.506251942726873
  think0_baseline_loss: 1.8811764822852226
  full_vs_baseline: 1.0070314970058616
  unique_chars: 21
  max_run_fraction: 0.015151515151515152
```

L5C seed sweep:

```text
out_dir:
  local_eval/qtrm_native_l5_language_nonregression_seed_sweep

decision:
  accepted_l5c_seed_stability

summary:
  pass_count: 3 / 3
  min_full_vs_baseline: 0.975550581055019
  max_full_vs_baseline: 1.003826688355419
  min_full_vs_think0: 0.4441652548042246
  max_full_vs_think0: 0.49055043638467777
```

Interpretation:

```text
The native recurrent causal path has now passed broader synthetic reasoning
families and a seed-stable larger text-slice language non-regression gate.
This still is not broad LLM capability. The next orthodox target is L5D MHA ETD
vs Qwen3.5-style hybrid comparison under the same L4/L5 gates.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_text_probe \
  tests.test_research_gate_runner \
  tests.test_qtrm_native_l5_language_seed_sweep

WRITE_WIKI=0 PYTHONPATH=src bash scripts/338_run_qtrm_native_gate.sh \
  qtrm_native_l5_language_nonregression smoke

WRITE_WIKI=0 PYTHONPATH=src bash scripts/338_run_qtrm_native_gate.sh \
  qtrm_native_l5_language_nonregression standard

PYTHONPATH=src .venv/bin/python scripts/341_qtrm_native_l5_language_seed_sweep.py \
  --profile standard --seeds 337 338 339
```

## [2026-05-12] evaluation | QTRM-native L5B multi-family standard accepted

Added multi-family support to the L4/L5 native text reasoning probe:

```text
script:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

families:
  modchain
  revchain
  checksum

runner gate:
  qtrm_native_l5_multifamily
```

Important architectural constraint:

```text
The task family is encoded as fixed-width text in the same prompt:
  task modchain start ...
  task revchain start ...
  task checksum start ...

There is no hidden family side channel.
```

Standard run:

```text
out_dir:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard

decision:
  accepted_l5_multifamily

thresholds:
  full_exact >= 0.60
  depth_gain >= 0.10
  ablation_drop >= 0.10
  min_family_exact >= 0.40
```

Metrics:

```text
full_generation_exact: 0.6067708333333334
think0_generation_exact: 0.020833333333333332
state_reset_generation_exact: 0.03515625
op_zero_generation_exact: 0.03515625
full_minus_think0: 0.5859375
full_minus_worst_ablation: 0.5716145833333334
min_family_generation_exact: 0.4140625
```

Family breakdown:

```text
checksum: 0.9375
modchain: 0.46875
revchain: 0.4140625
```

Interpretation:

```text
This is the first L5B acceptance: the same QTRM-native recurrent LM path can
handle three tagged reasoning families through generated text answers. It is
not yet stable L5. The pass is marginal and dominated by checksum. The next
work should be an L5 seed sweep and then targeted fixes for modchain/revchain,
not a premature Qwen3.5-style backbone or MSA switch.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_research_gate_runner

passed
```

## [2026-05-12] evaluation | QTRM-native L4 seed stability accepted

Ran the standard L4 mixed text reasoning seed-stability gate:

```text
script:
  scripts/339_qtrm_native_seed_sweep.py

out_dir:
  local_eval/qtrm_native_l4_seed_sweep

decision:
  accepted_seed_stability

policy:
  min_seeds: 3
  min_pass_rate: 1.0
  min_exact_per_seed: 0.70
```

Results:

```text
seed 337:
  full_generation_exact: 0.7421875
  think0_generation_exact: 0.02734375
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.03125

seed 338:
  full_generation_exact: 0.74609375
  think0_generation_exact: 0.03515625
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.046875

seed 339:
  full_generation_exact: 0.716796875
  think0_generation_exact: 0.025390625
  state_reset_generation_exact: 0.02734375
  op_zero_generation_exact: 0.03125
```

Summary:

```text
pass_count: 3 / 3
pass_rate: 1.0
min_full_generation_exact: 0.716796875
max_full_generation_exact: 0.74609375
reject_reasons: []
```

Interpretation:

```text
This is the strongest QTRM-native result so far. The small L4 mixed text
reasoning scaffold is no longer a one-seed accident: the recurrent core
causally improves generated text answers across three seeds, while think0,
state_reset, and op_zero remain near chance. This still does not prove broad
LLM capability; it promotes the native recurrent causal path to a reproducible
small raw-reasoning baseline.
```

Method correction made before the run:

```text
scripts/339_qtrm_native_seed_sweep.py
  --min-seeds default: 3
  --reuse-existing added
  min_exact is now an actual rejection condition, not just a reported field
```

## [2026-05-12] evaluation | QTRM-native L4 seed sweep guard added

Added a seed-stability guard for the QTRM-native L4 scaffold:

```text
scripts/339_qtrm_native_seed_sweep.py
tests/test_qtrm_native_seed_sweep.py
```

Purpose:

```text
Do not promote a one-seed L4 acceptance into a stable architecture claim.
Require the same mixed text reasoning gate to pass across multiple seeds.
```

Default standard sweep:

```bash
PYTHONPATH=src .venv/bin/python scripts/339_qtrm_native_seed_sweep.py \
  --profile standard --seeds 337 338 339
```

Promotion policy:

```text
min_pass_rate: 1.0
min_exact per seed: 0.70
decision required: accepted_seed_stability
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest tests.test_qtrm_native_seed_sweep

Ran 3 tests: OK
```

Dry-run:

```text
PYTHONPATH=src .venv/bin/python scripts/339_qtrm_native_seed_sweep.py \
  --profile standard --seeds 337 338 \
  --out-root local_eval/qtrm_native_l4_seed_sweep_dryrun --dry-run
```

Result:

```text
decision: dry_run
commands: 2
seed 337 -> eval_seed 9337
seed 338 -> eval_seed 9338
```

Interpretation:

```text
The accepted seed-337 L4 run remains a canonical scaffold. It is not yet a
seed-stable architecture claim until this sweep passes.
```

## [2026-05-12] tooling | QTRM-native gates promoted into research runner

QTRM-native is now exposed as the canonical one-click gate path:

```text
qtrm_native_l1_mha
qtrm_native_l1_hybrid
qtrm_native_l2_curriculum_depth
qtrm_native_l3_language_slice
qtrm_native_l4_mixed_text_reasoning
```

Added:

```text
scripts/338_run_qtrm_native_gate.sh
```

The wrapper defaults to:

```text
gate: qtrm_native_l4_mixed_text_reasoning
profile: standard
```

Important correction:

```text
smoke profile keeps real acceptance thresholds
```

So smoke is a runtime/report-parsing check, not a fake capability acceptance.
The runner now preserves `report.json` even when a gate script exits non-zero
because the model was rejected. This matters for QTRM-native probes because
rejected experiments are expected to return exit code 1 while still writing the
metrics needed for architecture debugging.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_research_gate_runner

Ran 30 tests: OK
```

Wrapper smoke:

```text
WRITE_WIKI=0 bash scripts/338_run_qtrm_native_gate.sh \
  qtrm_native_l4_mixed_text_reasoning smoke
```

Result:

```text
exit_code: 1
decision: rejected
accepted: false
eval_metrics.think4.generation_exact: 0.0
```

Interpretation:

```text
This is the correct smoke behavior: runtime path works and metrics are parsed,
but no L4 capability claim is made from a two-step CPU run.
```

## [2026-05-12] architecture | QTRM-native backbone comparison path added

Added an explicit QTRM-native backbone schedule:

```text
MHA ETD baseline
vs
Qwen3.5-style Delta / Delta / Delta / Attention hybrid
```

The native probe now exposes:

```text
scripts/335_train_qtrm_native_etd_probe.py --backbone mha_etd
scripts/335_train_qtrm_native_etd_probe.py --backbone qtrm_hybrid_3to1
```

This does not promote the hybrid yet. It only makes the comparison falsifiable
under the same L1/L2 native gates. MSA is recorded as a final-architecture
memory route, but remains off-by-default until native LM viability and recursive
core causality are proven. Promotion requires `memory_off`, `router_off`, and
chunk-shuffle ablations to drop the same LM-generation metric.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe

Ran 5 tests: OK
```

Runtime smoke:

```text
PYTHONPATH=src .venv/bin/python scripts/335_train_qtrm_native_etd_probe.py \
  --out-dir local_eval/qtrm_native_hybrid_3to1_smoke_cpu_s1 \
  --steps 1 --train-cases 8 --eval-cases 2 \
  --program-len 2 --modulus 8 --d-model 16 --n-heads 4 \
  --n-kv-heads 2 --d-ff 32 --backbone qtrm_hybrid_3to1 \
  --hybrid-layers 4 --attn-every 4 --train-think-steps 1 \
  --eval-think-steps 1 --batch-size 2 --device cpu
```

Result:

```text
completed and wrote report, but decision was rejected
full_generation_exact: 0.5
think0_generation_exact: 0.0
state_reset_generation_exact: 0.5
op_zero_generation_exact: 0.0
```

Interpretation: the hybrid path runs, but is not promoted because state reset
still matches full.

## [2026-05-12] training | Native L1 accepted after answer/EOS loss split

Implemented the native probe's first decoder-loss repair:

```text
answer CE and EOS CE are weighted separately
answer-vs-EOS margin is applied at the first generated token position
evaluation reports answer_token_accuracy and first_token_eos_rate
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe

Ran 6 tests: OK
```

Short MHA smoke:

```text
local_eval/qtrm_native_mha_answer_eos_s80
decision: rejected
loss: 3.9079 -> 1.3863
think4_generation_exact: 0.1875
think0_generation_exact: 0.125
state_reset_generation_exact: 0.09375
op_zero_generation_exact: 0.03125
first_token_eos_rate: 0.0
```

Strict MHA acceptance:

```text
local_eval/qtrm_native_mha_answer_eos_cuda_s600_strict
decision: accepted_l1_native_etd
thresholds: exact>=0.90, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 0.98046875
think0_generation_exact: 0.16796875
state_reset_generation_exact: 0.32421875
op_zero_generation_exact: 0.12109375
full_minus_think0: 0.8125
full_minus_worst_ablation: 0.65625
first_token_eos_rate: 0.0
```

Strict Qwen3.5-style hybrid acceptance:

```text
local_eval/qtrm_native_hybrid_3to1_cuda_s600_strict
decision: accepted_l1_native_etd
thresholds: exact>=0.90, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 1.0
think0_generation_exact: 0.05859375
state_reset_generation_exact: 0.59765625
op_zero_generation_exact: 0.15625
full_minus_think0: 0.94140625
full_minus_worst_ablation: 0.40234375
first_token_eos_rate: 0.0
```

Interpretation: this is the first accepted donorless native LM-path scaffold.
It is still a tiny synthetic-language result, not a general LLM claim. The next
promotion target is L2: harder program length/modulus, seed stability, and
language-slice non-regression.

## [2026-05-12] evaluation | Native L2 ladder boundary found

Ran harder donorless native recursive probes after L1 acceptance.

Rejected full L2B candidate:

```text
local_eval/qtrm_native_mha_l2_program4_mod32_cuda_s5000
program_len: 4
modulus: 32
steps: 5000
decision: rejected
threshold exact>=0.70, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 0.248046875
think0_generation_exact: 0.046875
state_reset_generation_exact: 0.111328125
op_zero_generation_exact: 0.021484375
full_minus_think0: 0.201171875
full_minus_worst_ablation: 0.13671875
```

Rejected hybrid L2B candidate:

```text
local_eval/qtrm_native_hybrid_l2_program4_mod32_cuda_s1200
program_len: 4
modulus: 32
steps: 1200
decision: rejected
think4_generation_exact: 0.1953125
think0_generation_exact: 0.021484375
state_reset_generation_exact: 0.13671875
op_zero_generation_exact: 0.015625
```

Accepted intermediate L2A:

```text
local_eval/qtrm_native_mha_l2a_program3_mod16_cuda_s2500
program_len: 3
modulus: 16
steps: 2500
decision: accepted_l1_native_etd
threshold exact>=0.70, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 0.763671875
think0_generation_exact: 0.060546875
state_reset_generation_exact: 0.203125
op_zero_generation_exact: 0.05859375
full_minus_think0: 0.703125
full_minus_worst_ablation: 0.560546875
```

Interpretation:

```text
The current native recursive path is real but not enough for the harder L2B
distribution. The immediate bottleneck is exact value generalization over
longer operation chains, not first-token EOS collapse.
```

Added optional step-wise depth supervision:

```text
scripts/335_train_qtrm_native_etd_probe.py
  --depth-intermediate-loss-weight

tests/test_qtrm_native_etd_probe.py
  depth_target_tokens follows step-wise program state
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe

Ran 7 tests: OK
```

Depth-target L2B probe:

```text
local_eval/qtrm_native_mha_l2_program4_mod32_depth_targets_cuda_s2500
program_len: 4
modulus: 32
steps: 2500
depth_intermediate_loss_weight: 0.5
decision: rejected
think4_generation_exact: 0.291015625
think0_generation_exact: 0.0
state_reset_generation_exact: 0.0234375
op_zero_generation_exact: 0.025390625
full_minus_think0: 0.291015625
full_minus_worst_ablation: 0.265625
```

Interpretation: step-wise depth targets strengthen causal separation and beat
the 5000-step no-depth-target exact rate slightly at half the steps, but still
do not solve L2B. More structure is needed for exact value-state generalization.

Added active-length curriculum:

```text
scripts/335_train_qtrm_native_etd_probe.py
  --active-len-curriculum
  --active-len-curriculum-min
  --active-len-curriculum-warmup-frac

tests/test_qtrm_native_etd_probe.py
  case_with_active_program_len replaces tail ops with NOOP and recomputes answer
  active_program_len_for_step warms up then uses full program
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe

Ran 9 tests: OK
```

Accepted L2B-style run:

```text
local_eval/qtrm_native_mha_l2_program4_mod32_curriculum_depth_cuda_s5000
program_len: 4
modulus: 32
steps: 5000
depth_intermediate_loss_weight: 0.5
active_len_curriculum: true
active_len_curriculum_warmup_frac: 0.5
decision: accepted_l1_native_etd
threshold exact>=0.70, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 0.955078125
think0_generation_exact: 0.0
state_reset_generation_exact: 0.013671875
op_zero_generation_exact: 0.03125
full_minus_think0: 0.955078125
full_minus_worst_ablation: 0.923828125
first_token_eos_rate: 0.0
```

Interpretation:

```text
Direct L2B and depth-target-only L2B were rejected. Active-length curriculum
plus step-wise depth targets passed strongly. This is the current best evidence
that the native recurrent trajectory can perform multi-step value transitions
inside the ordinary LM logits path.
```

## [2026-05-12] evaluation | Native L3 language-slice smoke accepted

Added a donorless native text preservation probe:

```text
scripts/336_train_qtrm_native_text_probe.py
tests/test_qtrm_native_text_probe.py
```

Purpose:

```text
Check that the native recurrent LM path can learn a small text slice without
repetition collapse. This is a language-preservation smoke, not a general
language-capability claim.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/336_train_qtrm_native_text_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe

Ran 12 tests: OK
```

Accepted smoke:

```text
local_eval/qtrm_native_text_l3_smoke_cuda_s800
steps: 800
seq_len: 64
backbone: mha_etd
decision: accepted_l3_language_slice
vocab_size: 33
random_loss: 3.4965
think_eval_loss: 0.04335
think0_loss: 4.3128
thinking_block_off_loss: 4.3128
unique_chars: 30
max_run_fraction: 0.01515
sample:
  QTRM native language probe. A small recurrent model should preserve ordinary
  next-token language behavior while its thinking block i
```

Interpretation:

```text
The native recurrent path can preserve basic autoregressive text behavior on a
small slice. The next L4 target is a mixed reasoning+language gate using normal
prompt text rather than only symbolic tokens or tiny text memorization.
```

## [2026-05-12] evaluation | Native L4 mixed text reasoning scaffold added but rejected

Added a mixed text reasoning probe:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
tests/test_qtrm_native_mixed_text_reasoning_probe.py
```

The probe uses ordinary fixed-width text prompts and text answers:

```text
prompt: start 19 ops 06 05 06 02 answer
answer: 28\n
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/336_train_qtrm_native_text_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe

Ran 15 tests: OK
```

Runs:

```text
local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s3000
  decision: rejected
  think4_generation_exact: 0.326171875
  think0_generation_exact: 0.01953125
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.033203125

local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s8000
  decision: rejected
  think4_generation_exact: 0.677734375
  think0_generation_exact: 0.005859375
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.029296875

local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s10000
  decision: rejected
  think4_generation_exact: 0.693359375
  think0_generation_exact: 0.017578125
  state_reset_generation_exact: 0.03515625
  op_zero_generation_exact: 0.044921875

local_eval/qtrm_native_mixed_text_l4_program4_mod32_cuda_s12000
  decision: rejected
  think4_generation_exact: 0.693359375
  think0_generation_exact: 0.009765625
  state_reset_generation_exact: 0.033203125
  op_zero_generation_exact: 0.03515625
```

Interpretation:

```text
The L4 path is not accepted because exact text-answer generation remains below
the 0.70 threshold. The positive signal is strong causal recurrence:
full-minus-worst-ablation is ~0.65. The next bottleneck is answer text/value
rendering under normal prompt format.
```

Capacity-up L4 acceptance:

```text
local_eval/qtrm_native_mixed_text_l4_program4_mod32_d128_cuda_s8000
steps: 8000
d_model: 128
n_heads: 8
d_ff: 256
decision: accepted_l4_mixed_text_reasoning
threshold exact>=0.70, depth_gain>=0.10, ablation_drop>=0.10
think4_generation_exact: 0.7421875
think0_generation_exact: 0.02734375
state_reset_generation_exact: 0.03515625
op_zero_generation_exact: 0.03125
full_minus_think0: 0.71484375
full_minus_worst_ablation: 0.70703125
```

Interpretation:

```text
This is the first accepted L4 mixed normal-prompt scaffold. It remains a small
synthetic text task, but it proves the native recurrent path can parse text-form
inputs and emit text-form answers while destructive ablations collapse the same
generation exact metric.
```

## [2026-05-11] implementation | Core-state-only answer loop scaffold

Implemented `answer_state_loop_core_state_only_enabled`.

Effect:

```text
Text/donor hidden states can query the core trajectory, but raw text hidden
states are not appended as answer-loop cross-attention values.
```

Files:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
tests/test_core_halting.py
tests/test_mixed_noncopy_lm_gate.py
scripts/330_run_mixed_noncopy_lm_gate.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_core_state_only_s060.yaml
configs/qwen35_2b_4090_source_copy_pointer_renderer_core_state_only_scaffold.yaml
```

Verification:

```text
.venv/bin/python -m unittest \
  tests.test_core_halting \
  tests.test_model_config \
  tests.test_raw_intelligence_eval_script \
  tests.test_mixed_noncopy_lm_gate

Ran 153 tests: OK
```

Smoke:

```text
.venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
  --max-cases 1 \
  --chunk-size 1 \
  --max-length 192 \
  --max-new-tokens 8 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/mixed_noncopy_typed_core_state_only_smoke_1case
```

Result:

```text
decision: rejected_noncopy_lm_gate
full: 0/1
donor: 0/1
core_off: 0/1
core_state_zero: 0/1
answer_recurrent_off: 0/1

full completion:             66666666
core_state_zero completion:  55555555
answer_recurrent_off:        00000000
target:                      600054
```

Decision:

```text
Accept the scaffold as a stricter orthodox path, not as an L4 capability.
The remaining blocker is non-copy scalar/list accumulator synthesis into the
LM-logit path under core-state-only constraints.
```

Training smoke:

```text
.venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_fullpath_scalar_codec_core_state_only_s060.yaml \
  --data-jsonl data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl \
  --init-checkpoint /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_sufficient_onecase_overfit/train_eos_s020/last.pt \
  --steps 1 \
  --depth-steps 8 \
  --target-mode final \
  --max-length 192 \
  --target-logit-positions-only \
  --causal-prefix-supervision \
  --causal-prefix-max-target-tokens 8 \
  --causal-prefix-skip-leading-whitespace-targets \
  --causal-prefix-append-eos-target \
  --final-path-only-supervision \
  --answer-state-loop-logit-ce-weight 1.0 \
  --final-logit-ce-weight 1.0 \
  --depth-final-ce-weight 0.0 \
  --progress-margin-weight 0.0 \
  --lr 1e-5 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/smoke_core_state_only_causal_prefix_s1
```

Observed:

```text
saved: /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/smoke_core_state_only_causal_prefix_s1/last.pt
final_path_ce: 6.9826
answer_state_loop_logit_ce: 6.9826
final_path_acc: 0.1429
answer_state_loop_logit_acc: 0.1429
causal_prefix_examples: 7
```

## [2026-05-11] architecture | Orthodox TRM general-LLM direction recorded

Decision:

```text
TRM/QTRM remains the primary reasoning-core direction for general LLM work,
but only if the answer follows one canonical causal path:
prompt tokens -> donor/token states -> mandatory recurrent core ->
core-dependent readout -> LM logits -> autoregressive text.
```

Artifact:

```text
docs/wiki/decisions/orthodox-trm-general-llm-direction.md
```

Operational change:

```text
Future promotions require same-run destructive ablations:
core_off, core_state_zero, answer/readout path off, and depth sweep.
If answer-state recurrence or a renderer works after core state is zeroed,
the result is diagnostic, not a TRM-general-LLM architecture claim.
```

## [2026-05-07] experiment | Mixed composition length 11/13 accepted

Extended the accepted dynamic-halt mixed list-to-arithmetic Stage 1 gate from
held-out list lengths 7/9 to 11/13.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-len1113-s080.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len1113_jointonly_s080_from_s720/last.pt
```

Result:

```text
S720 baseline on len11/13:
  trace exact:    32/32
  halted exact:  30/32

joint-only S080 recovery:
  len11/13 full:             32/32
  len11/13 transition-off:    0/32
  len11/13 code-shuffle:      0/32
  len11/13 code-dropout:      0/32
  canonical len7/9 full:     32/32
```

Decision:

```text
Accept as a Stage 1 length-transfer extension. The fix used only
transition-state joint CE; answer-token CE was disabled.
```

## [2026-05-07] experiment | Depth-text process CE rejected

Goal: test whether direct process credit on every recurrent depth's answer
logits can make deeper QTRM loops useful.

Implementation:

```text
new loss:
  core_depth_text_ce_loss

main forward:
  return_core_depth_text_logits=True when the loss is enabled

memory fix:
  drop return_core_depth_text_logits from preference rejected, counterfactual,
  short-trajectory, and canonical ablation forwards

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_depth_text_ce_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_depth_text_ce_s080/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_depth_text_ce_s080_causal_fc_24.jsonl
```

Training signal:

```text
core_depth_text_ce: about 11.44 -> 11.13
```

Held-out 24-case causal forced-choice:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:            9/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off step8:         9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject. The accepted baseline remains core_steps_1 14/24 and core_steps_4
13/24. Depth-text CE did not create deeper-loop gains.
```

Branch decision:

```text
Depth router heads, shortcut consistency, variable trajectory, and depth-text
process CE are all rejected for this donor-preserving controller branch.
The next architecture must train explicit verifiable recurrent state updates.
```

## [2026-05-07] experiment | Variable-trajectory v1 rejected

Goal: test the stronger version of the LoopFormer-inspired idea after
same-run shortcut consistency failed.

Implementation:

```text
long path:
  normal forward with outer_steps=4

short path:
  second forward on the same batch with outer_steps=1

new loss:
  core_variable_trajectory_consistency_loss

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_variable_traj_s080_causal_fc_24.jsonl
```

Training proxy moved:

```text
state cosine: about 0.22 -> 0.43
long-short logp margin: about -0.03 -> +0.27
```

Held-out 24-case causal forced-choice:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           13/24
core_steps_2:            9/24
core_steps_4:           11/24
core_steps_8:            8/24
delta_off step8:         9/24
residual_gate_off step8: 8/24
```

Decision:

```text
Reject. The accepted baseline remains core_steps_1 14/24 and core_steps_4
13/24. Variable-trajectory v1 optimized internal proxy metrics but did not
improve raw intelligence.
```

Next:

```text
Stop adding depth alignment/router heads for this branch.
Move to a recurrent transition objective: each loop step must perform a
verifiable state update, and the final answer path must lose the gain when
that update path is ablated.
```

## [2026-05-07] experiment | Shortcut-consistency v1 rejected

Goal: test whether a LoopFormer-inspired trajectory regularizer stabilizes the
accepted donor-preserving core-forced readout across recursive depths.

Implementation:

```text
new loss:
  core_trajectory_shortcut_consistency_loss

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120.yaml

init checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

trained checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_shortcut_outer4_s120/last.pt
  deleted after rejection; eval JSONL preserved

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_shortcut_outer4_s120_causal_fc_24.jsonl
```

Training signal was active:

```text
core_trajectory_shortcut cosine reached about 0.56 near the final log
```

24-case causal forced-choice result:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:           10/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off step8:         9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject shortcut-consistency v1 as a canonical improvement.
It preserves the known step-1 gain but does not improve it, lowers step-4
from 13/24 to 12/24, and collapses step-8 to donor/core_off.
```

Next:

```text
Do not add another post-hoc route head.
Implement a true variable-trajectory short/long schedule experiment with
correctness or verifier gating before claiming LoopFormer-style training.
```

## [2026-05-07] experiment | Adaptive halt probe on core-forced readout

Goal: test whether the accepted core-forced donor-preserving readout can choose
its own recursive depth instead of relying on unstable fixed-depth sweeps.

Implementation and eval hygiene:

```text
raw eval:
  fixed-depth modes now pass enable_core_halt=false explicitly
  qtrm_core_halt_steps_N_no_evidence is the only halt-enabled mode
  donor_only now disables the core instead of merely setting qtrm scale to 0
  choice telemetry records actual core_steps_mean

config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_halt_teacher_depth_s080.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_halt_teacher_depth_s080/last.pt

eval:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_halt_teacher_depth_s080_causal_fc_24_v2.jsonl
```

The halt probe initialized from the accepted core-forced checkpoint and trained
only `core.halt_head.*` with teacher-depth stability targets. Training signal
was real but mostly told the model to continue to step 8:

```text
core_halt loss: 1.23 -> 0.69
teacher_depth_earliest_step_mean: 8.0
teacher_depth_step1_stable_rate: 0.0
```

24-case causal forced-choice result:

```text
donor_only:          9/24, core_steps_mean=0
core_off:            9/24, core_steps_mean=0
core_steps_1:       14/24, core_steps_mean=1
core_steps_2:        9/24, core_steps_mean=2
core_steps_4:       13/24, core_steps_mean=4
core_steps_8:       10/24, core_steps_mean=8
core_halt_steps_8:  10/24, core_steps_mean=8
delta_off:           9/24, core_steps_mean=8
```

Decision: reject teacher-depth stability halting as the next depth solution.
It learned a valid no-early-exit policy and reproduced fixed depth 8, but it
did not select the useful depths 1 or 4. The next depth selector must optimize
answer correctness / verifier preference directly, not only final-depth logit
stability.

Next candidate:

```text
supervised depth router:
  run fixed-depth candidates offline
  label each case with the shallowest correct depth, or UNKNOWN if none
  train a small router/verifier to select depth before answer scoring
  keep donor/core_off/delta_off ablations
```

Implemented the first label builder:

```text
script:
  scripts/depth_router_labels.py

labels:
  /mnt/nvme0n1p2/qtrm-runs/eval/donor_preserving_core_forced_depth_router_labels_24.jsonl

summary:
  cases: 24
  donor_hits: 9
  oracle fixed-depth hits: 16
  causal_core_gains: 7
  unknown_routes: 8

target_route distribution:
  donor: 9
  core_steps_1: 5
  core_steps_4: 1
  core_steps_8: 1
  unknown: 8
```

Interpretation: the fixed-depth oracle is already 16/24, so there is routeable
signal above donor-only 9/24. The immediate bottleneck is learning/predicting
the route, not inventing another core path.

## [2026-05-06] experiment | Core-forced donor-preserving readout

Tested the third donor-preserving controller candidate after v1/v2 failed:

```text
donor logits remain the base policy
QTRM residual is bounded
n_coda_layers = 0
core_loop_readout_enabled = true
core_loop_readout_requires_core = true
```

This makes the QTRM delta collapse to zero when `core_off`; the model can no
longer get a raw-reasoning gain from a non-recursive coda side path.

Pure-recursive preference run:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_pure_recursive_pref_s160/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:           11/24
  core_steps_1:       14/24
  core_steps_2:       10/24
  core_steps_4:       11/24
  core_steps_8:       11/24
  delta_off:           9/24
  residual_gate_off:  11/24
```

Decision: partial signal only. The delta learned something because
`delta_off` returns to donor, but `core_off` also beats donor. The recursive
core is not yet the causal source.

Core-forced run:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:       14/24
  core_steps_2:        9/24
  core_steps_4:       13/24
  core_steps_8:       10/24
  delta_off:           9/24
  residual_gate_off:  10/24
```

Decision: accept as a causal architecture improvement, not as a final raw
recursive intelligence result. The gain now disappears under `core_off` and
`delta_off`, so the improved modes depend on the core-forced QTRM delta.
However, fixed deeper recursion is unstable: depth 1 and 4 help, depth 8
degrades.

Outer4 continuation:

```text
config:
  configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_outer4_s120.yaml

checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_outer4_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:       14/24
  core_steps_2:        9/24
  core_steps_4:       11/24
  core_steps_8:        8/24
  delta_off:           9/24
  residual_gate_off:   8/24
```

Decision: reject. Fixed deeper training did not improve deeper inference; it
regressed depth 4 and 8. Next candidate should be adaptive early-exit / depth
selection over the core-forced donor-preserving readout, not longer fixed
recursion.

## [2026-05-06] implementation | Donor-preserving logit guider gate wired

Implemented the first falsifier for the donor-preserving controller path:

```text
final logits = donor logits + bounded/gated QTRM residual delta
```

Added:

```text
forward ablation:
  disable_qtrm_residual_gate

raw eval modes:
  qtrm_core_steps_8_delta_off_no_evidence
  qtrm_core_steps_8_residual_gate_off_no_evidence

canonical config:
  configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml
```

Verification:

```text
tests.test_raw_intelligence_eval_script
tests.test_model_config
tests.test_training_checkpoint_init
149 tests OK
py_compile OK
git diff --check OK
```

Next gate: train S120, then compare donor-only, guided, delta-off, gate-off,
and core-off under no-retrieval raw-intelligence evaluation.

Smoke result:

```text
checkpoint:
  runs/qwen35_2b_4090_donor_preserving_logit_guider_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:        9/24
  core_steps_2:        9/24
  core_steps_4:        9/24
  core_steps_8:        9/24
  delta_off:           9/24
  residual_gate_off:  10/24
```

Decision: not accepted. The gate-off mode beating the full gated mode suggests
the learned residual gate is currently too conservative or undertrained. Next
candidate is KISS bounded delta without a learned residual gate; donor
preservation remains enforced by donor scale 1.0, clamp, donor KL, and
donor-correct margin.

Follow-up v2:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/qwen35_2b_4090_donor_preserving_bounded_delta_nogate_s120/last.pt

24-case causal forced-choice:
  donor_only:          9/24
  core_off:            9/24
  core_steps_1:        9/24
  core_steps_2:        9/24
  core_steps_4:        9/24
  core_steps_8:        9/24
  delta_off:           9/24
  residual_gate_off:   9/24
```

Decision: v2 is also not accepted. Removing the gate did not create a causal
raw-reasoning gain. Root hypothesis update: the first two runs trained on
generic text/math/mm data, not on the pure-recursive preference data used by
the raw-intelligence gate. Next candidate must train the donor-preserving delta
on `data/filtered/pure_recursive_reasoning_preferences_train.jsonl`.

## [2026-05-06] research | Donor-preserving controller method selected

Web search over 2025-2026 reasoning-control prior found the best next path:

```text
ThinkLogit / Proxy-Tuning: logit arithmetic guider over frozen target
ReFT / GLoRE / BREP: bounded representation interventions in frozen LMs
BuPO: internal layer policies, especially relevant to Qwen-series structure
Dead Weights: frozen LMs can communicate through learned residual hooks
Speculative Thinking / Thinking Intervention: intervene only at reasoning points
```

Decision: implement the simpler bounded donor-logit guider before attempting
Qwen residual-stream hooks.

```text
final_logits = donor_logits + alpha * gate * clamp(qtrm_delta_logits)
```

Artifact:

```text
docs/wiki/sources/donor-preserving-reasoning-control-2026.md
docs/wiki/decisions/donor-preserving-controller-next-method.md
```

## [2026-05-06] renderer | Hidden bridge S080 rejected

Added a zero-init answer-state hidden bridge and trained only that bridge from
the accepted answer-halt S080 checkpoint.

Result:

```text
generation smoke4:
  full:              0/4
  hidden_bridge_off: 0/4
  halt_gate_off:     0/4

causal forced-choice smoke4:
  full:              0/4
  hidden_bridge_off: 4/4
```

Decision: reject and delete the generated checkpoint. This is a strong
root-architecture signal: the bridge does not open generation and actively
damages the accepted forced-choice behavior. Next candidate should preserve the
donor decoder/language path and inject QTRM as a residual cognitive controller,
instead of stacking more private QTRM renderer patches.

Artifact:

```text
docs/wiki/decisions/ouro-hidden-bridge-s080-reject.md
```

## [2026-05-06] renderer | LM-head-only decoder alignment rejected

After donor-unembedding head surgery failed, added `trainable_param_policy:
lm_head_only` to train only the untied final LM head from the accepted
answer-halt S080 checkpoint.

Result:

```text
generation smoke4:
  full/gate_off: 0/8

causal forced-choice smoke8:
  full:          0/8
  halt_gate_off: 0/5 observed before early stop
```

Decision: reject and delete the generated checkpoint. The update changed token
priors but destroyed the accepted forced-choice reasoning signal and still did
not make greedy generation correct. The next renderer candidate should align
answer-state hidden vectors through an ablatable LM-compatible hidden bridge,
not train only the final vocabulary head.

Artifact:

```text
docs/wiki/decisions/ouro-lm-head-only-decoder-alignment.md
```

## [2026-05-06] renderer | Donor-unembedding head surgery rejected

Web/prior search focused on output embedding geometry and latent-thought
rendering:

```text
Press/Wolf output embedding
Inan/Khosravi/Socher tied word vectors
PonderLM-2 latent thoughts before token prediction
Coconut official latent reasoning implementation
```

Implemented `scripts/246_build_donor_unembedding_aligned_checkpoint.py` to
project Qwen donor output embeddings into QTRM width and replace only the
untied LM head of the accepted Ouro answer-halt checkpoint.

Result:

```text
pinv donor-output head:
  generation smoke4: 0/8
  causal forced-choice smoke8 full: 6/8

direct projection donor-output head:
  generation smoke4: 0/8
```

Decision: reject both and delete both generated checkpoint weights. The failure
is not solved by vocab-head geometry alone; answer-state hidden vectors are not
yet aligned to an LM-compatible hidden manifold. Next candidate should train a
decoder/projection from answer-state hidden to tokenizer logits, PonderLM-style,
instead of adding more small margin losses to the current head.

Artifact:

```text
docs/wiki/decisions/ouro-donor-unembedding-head-probe.md
```

## [2026-05-06] renderer | Causal-prefix self-rollout rejected

Added online self-rollout causal-prefix supervision to the Ouro answer renderer
probe. The training path now can build prefix examples from the model's own
greedy generated tokens, then supervise the next gold token from that generated
prefix.

Result:

```text
generation smoke4:
  full/gate_off: 0/8

causal forced-choice smoke8:
  full:          8/8
  halt_gate_off: 0/8
```

Decision: reject as renderer and delete the checkpoint. This preserves the
accepted answer-halt forced-choice gate but does not make the model render the
answer autoregressively. The active bottleneck is no longer teacher-forcing
exposure bias alone; the answer-state-to-token rendering path itself is too
weak.

Additional beam diagnostic on the accepted S080 baseline:

```text
beam_width: 16
max_new_tokens: 7
heldout rows: 4
hits: 0/4
best samples: "1 1 1", "UNKNOWN 1 1"
```

Interpretation: the generation failure is not greedy-only decoding. Short beam
search also collapses to the same invalid token priors, so the next renderer
candidate should align answer-state hidden vectors to tokenizer logits more
directly instead of adding another small CE patch.

Artifact:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Typed Algorithmic Value-State Probe

Added a role-separated typed value-state probe for the accepted mixed
composition len11/13 checkpoint:

```text
raw_list_offsets
doubled_list_offsets
scalar_coeff
scalar_residual
final_residual
```

Result:

```text
held-out len11/13 mixed-only:
  content-field accuracy: 352/1024 = 0.34375
  head-off content acc:     0/1024 = 0.0
  trace exact:              0/32

action-code:
  len11/13 exact: 32/32
  len7/9 exact:   32/32
```

Decision:

```text
Accept only as a causal Stage-2 probe improvement. The typed field path learns
some value content and preserves the action controller, but it is not an exact
neural transition model yet. Next step is a recurrent typed value-transition
cell with delta-off/shuffle ablations.
```

Decision doc:

```text
docs/wiki/decisions/typed-algorithmic-value-state-len1113-s080.md
```

## [2026-05-06] raw-intelligence | Terminal-depth CE rejected

Added terminal-only answer CE:

```text
--terminal-depth-ce-weight
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
action-code: 32/32
```

Decision: reject. The terminal CE loss is active, but it only preserves the
accepted 4/8 score and removes the full-vs-bridge-off causal gap. Next
candidate should gate answer-state updates inside the recurrence with finality,
not add another post-hoc CE or selector.

Artifact:

```text
docs/wiki/decisions/ouro-terminal-depth-ce-s020-reject.md
```

## [2026-05-06] raw-intelligence | Subtract-tail counterfactual rejected, depth overshoot found

Added train-only counterfactual negatives for mixed subtract-tail failures:

```text
--subtract-tail-counterfactual-margin-weight
--subtract-tail-counterfactual-margin
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 4/8
full core8:   3/8
bridge_off:   2/8
action-code: 32/32
```

Decision: reject. The bridge path is causal again, but full core8 regresses
below the accepted 4/8 baseline. New bottleneck: recursive-depth overshoot.
Depth 4 matches the baseline while depth 8 worsens.

Artifact:

```text
docs/wiki/decisions/ouro-subtract-tail-counterfactual-s020-reject.md
```

## [2026-05-06] raw-intelligence | Trajectory monotonic process credit rejected

Added adjacent-depth process-credit training:

```text
--depth-trajectory-monotonic-weight
--depth-trajectory-monotonic-margin
```

Result on held-out mixed-composition smoke8:

```text
donor_only:   0/8
core_off:     0/8
core_steps 1: 2/8
core_steps 2: 2/8
core_steps 4: 2/8
full core8:   4/8
bridge_off:   4/8
action-code: 32/32
```

Decision: reject. Full preserves the accepted 4/8 score but bridge-off matches
full, so the objective does not prove a causal raw-intelligence gain. Logged
`depth_trajectory_monotonic` was already 0.0000, which means this was not the
active bottleneck.

Artifact:

```text
docs/wiki/decisions/ouro-trajectory-monotonic-s020-reject.md
```

## [2026-05-06] raw-intelligence | Bridge contrast creates causal dependency

Added `transition_joint_answer_bridge_contrastive_loss`, which contrasts the
full answer path against the same path with
`disable_transition_state_joint_answer_bridge=True`.

Result:

```text
train:
  final_path_ce: 4.0516 -> 2.6656
  bridge contrast delta: -0.0525 -> -0.0059

held-out LM forced-choice:
  full:       2/8
  bridge off: 0/8

action-code:
  exact: 32/32
```

Decision: accept as a smoke probe. This does not improve full accuracy over
S80, but it fixes the previous bridge failure class: the transition-joint
bridge is no longer ignorable under held-out ablation. Scale only with
save-every validation and reject continuations where bridge-off matches full.

Artifact:

```text
docs/wiki/decisions/ouro-transition-joint-answer-bridge-contrast-s020.md
```

Update: continued S020 for 60 more steps with validation checkpoints. Step20
and step60 both preserve the same held-out causal pattern:

```text
full:       2/8
bridge off: 0/8
action-code at step60: 32/32
```

Decision: accept causal preservation through S080-from-S020, but do not claim
quality improvement. The next objective needs to break the 2/8 smoke ceiling,
not only preserve bridge causality.

## [2026-05-06] raw-intelligence | Transition joint answer bridge rejected

Implemented a direct causal bridge from `transition_state_joint_logits` into
the recurrent answer-state loop and trained 20 steps from the accepted S80
checkpoint.

Result:

```text
train:
  final_path_ce: 3.9788 -> 2.6370

held-out LM forced-choice:
  full:       2/8
  bridge off: 2/8

action-code:
  exact: 32/32
```

Decision: reject as canonical. The answer path can now see the transition
trace, but it does not causally depend on it under the current objective. Next
candidate should add bridge-contrast or depth-wise answer-process supervision,
then require full > bridge-off before any scale-up.

Artifact:

```text
docs/wiki/decisions/ouro-transition-joint-answer-bridge-s020.md
```

## [2026-05-06] raw-intelligence | Finality selector rejected

Tested whether the S80 Ouro recurrent checkpoint's already-correct
transition-state finality signal can choose a better answer-state-loop depth
without retraining.

Result:

```text
soft selector:
  full:         2/8
  selector off: 2/8

hard_first selector:
  full:         2/8
  selector off: 2/8
  action-code:  32/32
```

Decision: reject as canonical. The transition trace remains correct, but a
post-hoc selector does not causally improve answer-token logits. Next candidate
should feed transition code/state into answer-state recurrence at every step.

Artifact:

```text
docs/wiki/decisions/ouro-finality-selector-zeroshot.md
```

## 2026-05-05: Universal LLM Causal Path Principle Added

Added a canonical principle for future QTRM architecture work:

```text
prompt/chat-template tokens
-> tokenizer
-> donor/token hidden states
-> QTRM workspace/core/memory
-> LM logits
-> autoregressive text
```

Structured modules such as typed registers, operation selectors, verifier
heads, and memory readers may be used only as learned and ablatable internal
bottlenecks. They must not become external calculators, hidden answer channels,
or runtime rule solvers that compute the final answer while the LLM only
formats it.

Artifacts:

```text
skill:
  ~/.agents/skills/research-driven-architecture-debugging/SKILL.md

wiki:
  docs/wiki/architecture/universal-llm-causal-path-contract.md
  docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md
  docs/wiki/architecture/canonical-architecture-matrix.md
```

Implication for the next typed-register executor:

```text
It must preserve the general LLM path. The verifier may judge register updates
during training/eval, but the learned model must perform the update causally at
inference and final answers must come through LM logits.
```

## [2026-05-05] raw-intelligence | Role-value slot candidate opened

Added the role/filler variable-slot prior page and the next falsifiable
candidate after generic algorithmic slots failed.

Artifacts:

```text
docs/wiki/sources/role-filler-variable-slots.md
docs/wiki/decisions/role-value-state-candidate.md
references/papers/role_value_slots/
```

Decision status: active candidate. The next gate is a short role-value state
train/eval: content role accuracy must beat the 0.05 generic-slot baseline,
trace exact must rise above 0/32, and the accepted action-code path must remain
32/32.

Update: rejected after the 480-step gate. Value accuracy rose to 48/624
(0.0769), but trace exact stayed 0/32 and step exact stayed 0/256. Action-code
behavior remained 32/32. Next candidate: make role-bound values part of the
mandatory recurrent state rather than a readout-only probe.

## [2026-05-05] raw-intelligence | Core role-value state scaffold partially accepted

Moved role-bound value tokens into the mandatory recurrent core and trained the
new core-role path from the accepted mixed-composition action checkpoint.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     0/256
  value accuracy: 100/624 = 0.1603

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: accept only as a scaffold. It beats readout-only role values
(0.0769 -> 0.1603) without breaking the accepted action controller, but exact
latent value-state remains 0/32. The next candidate is joint core training with
action preservation plus core role-value CE, not another readout head.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-s480.md
```

## [2026-05-05] raw-intelligence | Joint core role-value training improves value accuracy

Opened the recurrent core together with answer-state loop, transition-state
joint head, and core role-value tokens. The objective preserved action-code
behavior while continuing to train role-bound values.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     0/256
  value accuracy: 144/624 = 0.2308

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: accept the direction, reject final exact-state claim. The progression
is now 0.0500 -> 0.0769 -> 0.1603 -> 0.2308, while action-code stays 32/32.
Next bottleneck: add explicit previous-role-state to next-role-state
transition supervision.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-joint-s240.md
```

## [2026-05-05] raw-intelligence | Len579 gate exposes prompt-to-role binding bottleneck

Found a gate/data confound: the previous value-state train split had mixed
list length 5 only, while eval used lengths 7 and 9. Added
`--train-list-lengths` to the mixed composition builder and generated a
length-5/7/9 mixed-only train split.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     16/256 = 0.0625
  value accuracy: 184/624 = 0.2949

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: current best value-state scaffold, still not final. The progression
is now 0.0500 -> 0.0769 -> 0.1603 -> 0.2308 -> 0.2949, with first non-zero
step exact. Next bottleneck is prompt-to-role binding: role tokens need a
direct role-conditioned extraction path from prompt token states before the
mandatory recurrent core.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-len579-s240.md
```

## [2026-05-05] raw-intelligence | Prompt-extract role binding candidate rejected

Implemented gated role-conditioned prompt extraction before the recurrent core
and trained it for 240 steps from the best len579 scaffold.

Result:

```text
held-out value:
  trace exact:    0/32
  step exact:     16/256 = 0.0625
  value accuracy: 130/624 = 0.2083

action-code preservation:
  trace exact:    32/32
  halted exact:   32/32
  step accuracy:  1.0000
  finality acc:   1.0000
```

Decision: reject as implemented. It preserves action but regresses from the
best len579 value accuracy of 0.2949. Keep the code as experimental; do not
make it canonical without extractor pretraining or a better causal gate.

Artifact:

```text
docs/wiki/decisions/core-role-value-state-prompt-extract-s240.md
```

## [2026-05-05] raw-intelligence | Algorithmic value-state slots rejected

Tested structured neural-algorithmic value targets instead of digit strings:
kind=list/scalar plus relative list/scalar slots.

Result:

```text
pad-including CE:
  kind acc:         1.0000
  content slot acc: 0.0000
  trace exact:      0/32

content-slot CE:
  kind acc:         1.0000
  content slot acc: 0.0500
  trace exact:      0/32

action-code preservation:
  trace exact:      32/32
  halted exact:     32/32
```

Decision: reject. The model learns phase/kind but not exact value content.
Next candidate is typed algorithmic fields rather than one generic slot head.

Artifact:

```text
docs/wiki/decisions/algorithmic-value-state-s480.md
```

## [2026-05-05] raw-intelligence | Factorized value-state path rejected

Implemented the first state-factorized candidate: separate recurrent value
slots conditioned by prompt context and the frozen accepted action-state
trajectory.

Result:

```text
value-state:
  trace exact: 0/32
  token acc:   0.3713

action-code preservation:
  trace exact: 32/32
  halted:      32/32
```

Decision: reject. The action policy is preserved, but digit-sequence CE still
does not create exact algorithmic value state. Next step is structured
neural-algorithmic state targets: list element slots, accumulator slots, masks,
and explicit transition loss.

## [2026-05-05] raw-intelligence | Value-state-only control rejected

Ran the control from the state-factorized plan: freeze the accepted core/action
path and train only the compact value-state head.

Result:

```text
value-state:
  trace exact: 0/32
  token acc:   0.3794

action-code preservation:
  trace exact: 32/32
  halted:      32/32
```

Decision: reject value-readout-only as a solution. The accepted action policy
is preserved when frozen, but exact value content is not present in the current
latent trajectory. Proceed to state-factorized core instead of more readout
tuning.

## [2026-05-05] raw-intelligence | State-factorized core plan opened

Opened the research-backed architecture plan for the current bottleneck:
preserve the accepted latent action controller while adding causal
value-bearing recurrent state.

Artifact:

```text
docs/wiki/decisions/state-factorized-qtrm-core-research-plan.md
```

Prior axes:

```text
Neural Algorithmic Reasoning: latent processor and intermediate state.
Dreamer/RSSM: separated recurrent/action/value latent signals.
Factored Latent Action World Models: factorized action/state transition.
TRM/latent reasoning: recursive latent compute remains the raw-intelligence core.
```

Immediate experiment: freeze core/action and train only the compact
value-state readout to test whether accepted latent trajectories already
contain decodable value content.

## [2026-05-05] raw-intelligence | Compact value-state head rejected

Tested whether a compact digit/comma/minus value-state head can recover
value-bearing latent state without damaging the accepted mixed-composition
action policy.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-value-state-s480.md

evaluator:
  scripts/237_eval_qtrm_value_state.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_s480_from_mixed_s720/last.pt
```

Result:

```text
value-state:
  trace exact:    0/32
  token acc:      0.3713

action-code preservation:
  trace exact:   10/32
  step acc:       0.8477
```

Decision: reject. The compact value probe partially learns digits, but joint
training contaminates the accepted latent action policy. The next candidate
must factor action-state and value-state so values become causal without
regressing the action controller.

## [2026-05-05] raw-intelligence | State-sequence bottleneck rejected

Tested whether the accepted mixed-composition checkpoint contains decodable
value-bearing internal state, not only a correct latent action code.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-state-sequence-bottleneck-s480.md

evaluator:
  scripts/236_eval_qtrm_core_state_sequence.py
```

Result:

```text
accepted action-code checkpoint:
  state trace exact: 0/32
  state token acc:   0.0040

joint core/readout sequence CE:
  state token acc:   0.3713
  action-code exact: 0/32

readout-only:
  state token acc:   0.2273
  action-code exact: 32/32

direct transition-state sequence head:
  state token acc:   0.1418
  action-code exact: 32/32
```

Decision: reject state-content claims for the current core. The next bottleneck
is value preservation inside the recurrent latent update, not another readout
head.

## [2026-05-05] raw-intelligence | Mixed-family composition gate accepted

Accepted the next Stage 1 gate after list length/value transfer: list state now
feeds an arithmetic aggregation/subtraction step.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-mixed-composition-s720.md

builder:
  scripts/235_build_mixed_family_composition_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt
```

Result:

```text
full held-out:
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  exact:         0/32
  halted exact:  0/32

code shuffle:
  exact:         0/32
  halted exact:  0/32

code dropout:
  exact:         0/32
  halted exact:  0/32
```

Key architecture correction: `dynamic_halt_v3` separates action identity from
halt/finality, and the mixed-train rows are interleaved because the training
script consumes rows deterministically.

## [2026-05-05] raw-intelligence | Long list transfer gate accepted

Scaled the accepted list paraphrase transfer gate to held-out list lengths and
value ranges.

Artifacts:

```text
decision:
  docs/wiki/decisions/transition-joint-list-transfer-long-s120.md

builder:
  scripts/234_build_list_transfer_gate.py

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/last.pt
```

Split:

```text
train:
  list variants: 0, 1, 2, 3, 4, 5
  list length:   5

eval:
  list variants: 6, 7
  list lengths:  7, 9
  rows:          32
```

Result:

```text
full:
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  exact:         0/32
  halted exact:  0/32

code shuffle:
  exact:         0/32
  halted exact:  0/32

code dropout:
  exact:         0/32
  halted exact:  0/32
```

Interpretation: accepted as within-family length/value/surface transfer. Do not
promote to full operation-family zero-shot or open-ended reasoning.

## [2026-05-04] raw-intelligence | Prompt-conditioned primitive transition head

Fixed a causal information-path failure in the QTRM primitive transition
experiment.

Previous path:

```text
core_depth_states only -> primitive operation logits
```

Observed failure:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_s240_oponly/last.pt

heldout first64:
  operation accuracy: 128/256 = 0.5000

failure:
  family-insensitive operation schedule collapse
```

Implemented path:

```text
canonical prompt tokens
-> prelude text context
-> masked prompt pooled context
-> concat(core_depth_state, prompt_context)
-> primitive operation logits
```

Config:

```text
primitive_transition_prompt_context_enabled: true
```

Result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptctx_s720_oponly/last.pt

train:
  operation accuracy: 512/512 = 1.0000

heldout existing:
  operation accuracy: 250/256 = 0.9766

heldout7000:
  operation accuracy: 507/512 = 0.9902
```

Remaining failure is localized to list transforms:

```text
double_filtered -> second_mapping
```

Decision record:

```text
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240.md
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240-summary.json
```

## [2026-05-03] gate | Temporal-spatial context causal gate wired

Added a concrete train/eval/gate path for temporal-spatial context conditioning:

```text
scripts/207_build_temporal_spatial_context_cases.py
  builds SSOT-derived temporal_freshness and spatial_relation JSONL cases

scripts/196_train_pure_recursive_depth_supervised.py
  now forwards row.temporal_spatial_context to student and teacher models

scripts/192_eval_raw_intelligence.py
  now forwards case.temporal_spatial_context and supports
  qtrm_core_steps_N_temporal_spatial_off_no_evidence

src/qtrm_mm/eval/raw_intelligence_gate.py
  gate_type=temporal_spatial_context

scripts/208_run_temporal_spatial_context_gate.sh
  builds train/eval data, trains the context probe, runs context-on/off eval,
  and writes the wiki gate report
```

Generated case files:

```text
data/train/temporal_spatial_context_train_120.jsonl
data/eval/temporal_spatial_context_heldout_24.jsonl
```

No capability claim is made until
`qtrm_core_steps_8_no_evidence` beats
`qtrm_core_steps_8_temporal_spatial_off_no_evidence` on the held-out gate.

Pipeline smoke:

```text
command shape:
  PYTHON_BIN=.venv/bin/python TRAIN_CASES_PER_FAMILY=1 EVAL_CASES_PER_FAMILY=1
  STEPS=1 bash scripts/208_run_temporal_spatial_context_gate.sh

result:
  pipeline completed
  eval records: 4
  smoke status: rejected
```

The rejection is expected for a 1-step smoke. It proves the runner, model
forward path, eval ablation, and gate writer are connected; it does not test
capability.

Full S240 gate result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/temporal_spatial_context_probe/s240/last.pt

eval:
  local_eval/temporal_spatial_context_gate.jsonl

gate:
  docs/wiki/decisions/temporal-spatial-context-gate.md
  docs/wiki/decisions/temporal-spatial-context-gate-summary.json

status:
  rejected

context_on:
  qtrm_core_steps_8_no_evidence = 4/24

context_off:
  qtrm_core_steps_8_temporal_spatial_off_no_evidence = 4/24

task family:
  temporal_freshness = 4/12 on, 4/12 off
  spatial_relation = 0/12 on, 0/12 off
```

Interpretation: the new context path is wired and ablatable, but the S240
training recipe did not make it causally useful on held-out cases. The next
architecture/training step should not claim temporal/spatial intelligence; it
should attack spatial failure and context-off tie directly.

Delta analysis:

```text
script:
  scripts/209_analyze_temporal_spatial_context_delta.py

summary:
  docs/wiki/decisions/temporal-spatial-context-delta-summary.json

paired cases:
  24

changed completions:
  0

context-on only correct:
  0

context-off only correct:
  0

mean chosen-logprob delta:
  +0.00048
```

Interpretation: the context path has near-zero behavioral effect. This is a
stronger failure than merely tying accuracy; it means the current training
recipe did not make the model depend on the temporal/spatial prefix tokens.

Implemented next diagnostic pressure:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --temporal-spatial-context-contrast-weight
  --temporal-spatial-context-contrast-margin

loss:
  max(0, margin - (logp_context_on(answer) - logp_context_off(answer)))

scripts/208_run_temporal_spatial_context_gate.sh
  TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT default = 0.5
```

1-step runtime smoke passed with contrast enabled and logged
`context_contrast` plus `context_contrast_target_logp_delta`. The next full
run should compare whether this changes the previously near-zero delta.

## [2026-05-03] architecture | Temporal-spatial context conditioning added

Added a small QTRM forward-path extension for time/space awareness:

```text
src/qtrm_mm/config.py
  temporal_spatial_context_enabled
  temporal_spatial_context_dim
  temporal_spatial_context_max_tokens

src/qtrm_mm/qtrm_model.py
  temporal_spatial_context -> projected prefix tokens
  disable_temporal_spatial_context ablation flag
  temporal_spatial_context_token_count telemetry

src/qtrm_mm/training/train.py
  trainable_param_policy=core_and_temporal_spatial_context
```

This does not change the root architecture. It keeps the same
SSOT -> workspace -> recursive core -> coda/readout path, but lets the core see
SSOT-derived time/space facts as model tokens rather than hidden runtime state.

Decision doc:

```text
docs/wiki/decisions/temporal-spatial-context-conditioning.md
```

## [2026-05-03] recovery | Metacognitive checkpoint rebuild path added

The old metacognitive baseline/candidate checkpoints are still unreadable from
`/mnt/sdb1`, so exact recovery is blocked unless a readable backup exists. Added
a separate recovery path that builds a new matched pair on a healthy disk:

```text
script:
  scripts/206_run_metacog_pair_rebuild.sh

default destination:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild

baseline:
  no_warmup_rebuilt_s001/last.pt

candidate:
  unknown_teacher_kl_conservative_rebuilt_s040/last.pt
```

The trainer now supports an explicit random-init recovery mode:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --allow-random-init
```

Without `--init-checkpoint` or `--allow-random-init`, training fails. This keeps
artifact recovery honest: a random-init matched pair can restore the eval loop,
but it is not the same result as the old unreadable s001/conservative-s040
checkpoints.

Also wired `ALL_DEPTH_CE_WEIGHT` through
`scripts/197_run_pure_recursive_depth_supervised_train.sh`, matching the
documented conservative teacher-KL setting.

Decision doc:

```text
docs/wiki/decisions/metacog-checkpoint-rebuild-recovery.md
```

## [2026-05-03] architecture | donor-QTRM conflict gate probe added

Full 40-case fusion-scale sweep was attempted after the smoke8 result, but the
`runs` symlink points to `/mnt/sdb1/ws-sky-data/qtrm-runs` and that mount began
returning I/O errors:

```text
runs -> /mnt/sdb1/ws-sky-data/qtrm-runs
findmnt -T runs:
  /mnt/sdb1 /dev/sdb1 ext4 rw,noatime,emergency_ro

failure:
  OSError: [Errno 5] Input/output error
  path: runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
```

The failure is an artifact-storage/checkpoint-read blocker, not a model result.
Do not trust the partial full-sweep file under `runs/eval`.

Implemented the next KISS fusion diagnostic instead:

```text
src/qtrm_mm/config.py
  donor_qtrm_conflict_gate_enabled
  donor_qtrm_conflict_qtrm_scale

src/qtrm_mm/qtrm_model.py
  donor/QTRM top-token conflict gate before donor-logit fusion

scripts/192_eval_raw_intelligence.py
  --donor-qtrm-conflict-gate
  --donor-qtrm-conflict-qtrm-scale

scripts/204_run_metacog_fusion_conflict_gate_sweep.sh
  writes full 40-case fusion sweep JSONL files to local_eval/
  compares candidate plain and candidate conflict-gate against baseline
  preflights checkpoint byte-read and output-dir write/delete

scripts/205_localize_metacog_checkpoints.sh
  copies baseline/candidate checkpoints to /mnt/nvme1n1p2/qtrm-local-checkpoints
  verifies source/destination sha256
  currently blocked on source checkpoint I/O error

docs/wiki/decisions/donor-qtrm-conflict-gate-probe.md
```

Verification:

```text
forced-choice eval now writes donor_qtrm_conflict_gate_mean per choice
calibration reports aggregate mean_predicted_conflict_gate and mean_choice_conflict_gate
localize helper fails fast at source checkpoint read, before copying
```

Decision:
this is an ablation probe, not a promoted architecture. Once `/mnt/sdb1` is
stable or checkpoints are copied to a healthy disk, rerun the 40-case fused
calibration gate with and without this conflict gate. If it helps, replace the
heuristic with a learned fusion calibration/router. If it does not, move to
trained reliability targets rather than more core-only loss tuning.

## [2026-05-03] evaluation | Split metacognitive gates isolate QTRM core gain from fusion failure

Added profile-aware metacognitive calibration gates:

```text
script:
  scripts/202_build_metacognitive_calibration_gate.py

profiles:
  strict: all records, existing behavior
  qtrm_core: qtrm_core_steps_8_no_evidence + qtrm_core_steps_8_qtrm_only_no_evidence
  fused: qtrm_core_steps_8_no_evidence + qtrm_core_steps_8_low_donor_no_evidence
```

Added explicit fusion-scale eval modes:

```text
script:
  scripts/192_eval_raw_intelligence.py

examples:
  qtrm_core_steps_8_donor_scale_0p50_no_evidence
  qtrm_core_steps_8_qtrm_scale_0p75_donor_scale_0p50_no_evidence
```

Held-out split on the conservative unknown-only teacher-KL checkpoint:

```text
qtrm_core report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-qtrm-core-gate.md
  status: accepted
  accuracy: 0.600000 -> 0.600000
  ECE:      0.408397 -> 0.394158  delta -0.014239
  Brier:    0.399219 -> 0.398868  delta -0.000351

fused report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-fused-gate.md
  status: rejected
  accuracy: 0.600000 -> 0.600000
  ECE:      0.348036 -> 0.362485  delta +0.014450
  Brier:    0.364187 -> 0.366383  delta +0.002196

fusion-scale smoke8:
  docs/wiki/decisions/metacog-fusion-scale-sweep-conservative-s040-smoke8.md
  status: rejected
  accuracy: 0.500000 -> 0.500000
  ECE:      0.498750 -> 0.499257  delta +0.000507
  Brier:    0.497586 -> 0.498591  delta +0.001005
  donor_scale_0p25 ECE delta: +0.000002
  donor_scale_1p0  ECE delta: +0.001601
```

Interpretation:
the current evidence supports a narrow QTRM-core metacognition improvement, not
a full architecture promotion. The donor/QTRM fusion path is now the active
failure class, especially `qtrm_core_steps_8_low_donor_no_evidence`. The next
architecture move should target fusion calibration directly instead of further
tuning the pure core loss and hoping the fused path follows. The small scale
sweep suggests the degradation grows as donor scale rises, but it is only an
8-case smoke and should be followed by a full 40-case sweep before adding a
learned fusion module.

## [2026-05-03] evaluation | Conservative unknown-only teacher-KL probe still rejected

Reduced update strength after the unknown-only S080 run worsened global and
low-donor fused calibration:

```text
checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_conservative_s040/last.pt
eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_conservative_s040_40.jsonl
report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-conservative-s040-calibration-heldout40.md

settings:
  steps: 40
  lr: 2.0e-6
  teacher_depth_kl_weight: 5.0
  all_depth_ce_weight: 0.10
  choice_margin_weight: 0.25
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.340911  delta +0.006972
Brier:                     0.286778 -> 0.287452  delta +0.000674
avg confidence when wrong: 0.635804 -> 0.637578  delta +0.001773
status: rejected
```

Signal:

```text
qtrm_core_steps_8_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_low_donor_no_evidence:
  accuracy unchanged at 0.60
  ECE/Brier worsened slightly
```

Interpretation:
stronger teacher KL and lower update strength move in the right direction, but
the fused path still rejects. The next architectural/training fix should
separate QTRM-only calibration from donor-fusion calibration, likely by adding a
small learned donor/QTRM fusion calibration gate or by evaluating the QTRM-only
metacognition claim separately from fused-generation behavior.

## [2026-05-03] evaluation | Unknown-only teacher-KL metacognitive probe rejected

Added a selective preference filter and trained only on `expected_unknown=true`
metacognitive rows with teacher-depth KL preservation.

Code/data:

```text
scripts/194_build_pure_recursive_reasoning_preferences.py
  --only-expected-unknown

tests/test_pure_recursive_reasoning_preferences.py

data/filtered/metacognitive_calibration_unknown_preferences_train.jsonl
  72 preference rows
```

Run:

```text
checkpoint:
  runs/qwen35_2b_4090_metacog_unknown_teacher_kl_s080/last.pt
eval:
  runs/eval/metacognitive_calibration_unknown_teacher_kl_s080_40.jsonl
report:
  docs/wiki/decisions/metacog-unknown-teacher-kl-s080-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.363077  delta +0.029138
Brier:                     0.286778 -> 0.298315  delta +0.011536
avg confidence when wrong: 0.635804 -> 0.649499  delta +0.013695
status: rejected
```

Interpretation:
selecting only UNKNOWN/OOD/contradiction rows improves expected-unknown slices
but makes answerable and low-donor fused calibration worse. It preserves
QTRM-only accuracy, but the checkpoint is not a valid metacognitive gain. The
next conservative check should reduce update strength or increase teacher-KL
weight before introducing more architecture.

## [2026-05-03] evaluation | Teacher-depth KL metacognitive probe rejected but preserved QTRM-only policy

Ran the next metacognitive probe with checkpoint preservation:

```text
teacher checkpoint:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
student init:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
checkpoint:
  runs/qwen35_2b_4090_metacog_teacher_kl_s080_v2/last.pt
eval:
  runs/eval/metacognitive_calibration_teacher_kl_s080_v2_40.jsonl
report:
  docs/wiki/decisions/metacog-teacher-kl-s080-v2-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.352499  delta +0.018560
Brier:                     0.286778 -> 0.292267  delta +0.005489
avg confidence when wrong: 0.635804 -> 0.643463  delta +0.007658
status: rejected
```

Mode-specific signal:

```text
qtrm_core_steps_8_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier improved slightly

qtrm_core_steps_8_low_donor_no_evidence:
  accuracy 0.60 -> 0.60
  ECE/Brier worsened
```

Interpretation:
teacher-depth KL fixed the previous direct forced-choice failure where QTRM-only
accuracy dropped to 0.425. It is useful as policy preservation. It is not enough
as the final metacognitive objective because global calibration and low-donor
fused calibration worsen. Next candidate: keep teacher KL, but apply calibration
pressure selectively to overconfident wrong/UNKNOWN rows rather than all
answerable rows.

## [2026-05-03] evaluation | Direct metacognitive forced-choice probe rejected

Built a separate metacognitive train split and trained a direct known/unknown
forced-choice probe from the no-warmup baseline.

Artifacts:

```text
train cases:
  data/filtered/metacognitive_calibration_train_40.jsonl
preferences:
  data/filtered/metacognitive_calibration_preferences_train.jsonl
checkpoint:
  runs/qwen35_2b_4090_metacog_forced_choice_s080/last.pt
eval:
  runs/eval/metacognitive_calibration_forced_choice_s080_40.jsonl
report:
  docs/wiki/decisions/metacog-forced-choice-s080-calibration-heldout40.md
```

Held-out result:

```text
accuracy:                  0.466667 -> 0.433333  delta -0.033333
ECE:                       0.333939 -> 0.295489  delta -0.038450
Brier:                     0.286778 -> 0.293834  delta +0.007056
avg confidence when wrong: 0.635804 -> 0.534910  delta -0.100895
status: rejected
```

Mode-specific signal:

```text
qtrm_core_steps_8_low_donor_no_evidence:
  accuracy 0.60 -> 0.75
  ECE/Brier worsened

qtrm_core_steps_8_no_evidence:
  accuracy 0.60 -> 0.425
  ECE/Brier improved because the model became less confident

qtrm_core_steps_8_qtrm_only_no_evidence:
  accuracy 0.60 -> 0.425
  same failure as full QTRM-only path
```

Interpretation:
the probe taught caution, but it damaged the pure QTRM answer policy. This is
not accepted as raw metacognitive intelligence. The next candidate should add
policy-preservation pressure, such as teacher-depth KL from the accepted
no-warmup core path, while applying calibration loss only where the baseline is
overconfident or wrong.

## [2026-05-03] training/evaluation | Uniform noise warm-up probe added but strict QTRM-mode gate rejects

The 40-case random-label warm-up result showed that a random-label CE objective
does not robustly improve metacognition. Added a smaller follow-up objective:
`--noise-warmup-uniform-weight`, which trains random-token inputs toward
high-entropy logits instead of a random target token.

Code:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  random_noise_warmup_loss(..., uniform_weight=...)
  --noise-warmup-uniform-weight

scripts/197_run_pure_recursive_depth_supervised_train.sh
  NOISE_WARMUP_UNIFORM_WEIGHT

tests/test_pure_recursive_depth_supervised_train_script.py
  overconfident-noise-logit penalty
  runner argument plumbing
```

Smoke checkpoint:

```text
runs/qwen35_2b_4090_noise_uniform_warmup_smoke_s001/last.pt
```

Matched 40-case metacognitive result:

```text
eval:
  runs/eval/metacognitive_calibration_noise_uniform_warmup_s001_40.jsonl
report:
  docs/wiki/decisions/noise-uniform-warmup-metacognitive-calibration-heldout40-s001.md

accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.332757  delta -0.001182
Brier:                     0.286778 -> 0.286715  delta -0.000063
avg confidence when wrong: 0.635804 -> 0.634379  delta -0.001426
status: rejected
```

Why rejected:
the global average improved, but the stricter gate now checks critical QTRM
modes separately. `qtrm_core_steps_8_no_evidence` and
`qtrm_core_steps_8_qtrm_only_no_evidence` both worsened on ECE and Brier. This
means the apparent gain is not yet a clean QTRM-core metacognitive gain.

Next:
build a direct forced-choice calibration training objective on known/unknown,
contradiction, and OOD rows, then require no worsening in core-on/QTRM-only
critical modes.

## [2026-05-03] evaluation | 40-case random-noise metacognitive gate rejected

Added a held-out metacognitive calibration dataset and regenerated the matched
no-warmup versus random-noise-warmup gate with category breakdowns.

Code and data:

```text
scripts/203_build_metacognitive_calibration_cases.py
scripts/192_eval_raw_intelligence.py
scripts/202_build_metacognitive_calibration_gate.py
tests/test_metacognitive_calibration_cases.py
tests/test_raw_intelligence_eval_script.py
tests/test_metacognitive_calibration_gate_script.py
data/eval/metacognitive_calibration_heldout_40.jsonl
```

Compared checkpoints:

```text
baseline checkpoint:
  runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
baseline eval:
  runs/eval/metacognitive_calibration_no_warmup_s001_40.jsonl

candidate checkpoint:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
candidate eval:
  runs/eval/metacognitive_calibration_noise_warmup_s001_40.jsonl
```

Result:

```text
accuracy:                  0.466667 -> 0.466667  delta +0.000000
ECE:                       0.333939 -> 0.337422  delta +0.003483
Brier:                     0.286778 -> 0.286384  delta -0.000394
avg confidence when wrong: 0.635804 -> 0.635978  delta +0.000174
status: rejected
failed check: candidate_ece_worse
```

Interpretation:
the earlier 1-case result remains only a smoke. The 40-case gate rejects the
current two-step warm-up because global ECE and wrong-answer confidence worsen.
UNKNOWN/OOD slices improve slightly, but answerable boolean rows degrade. Next
step is a stronger QTRM-side calibration objective, not promotion of this
checkpoint.

Report:

```text
docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001.md
docs/wiki/decisions/noise-warmup-metacognitive-calibration-heldout40-s001-summary.json
```

## [2026-05-03] training | Random-noise warm-up option added for QTRM-only calibration probes

Implemented a QTRM-side adaptation of the Random2 / Nature NMI 2026 random
noise calibration prior.

Reference code cloned:

```text
references/official/cogilab-random2 798a76cb1d98
references/official/cogilab-random  e96aadd13903
```

Code:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  build_random_noise_warmup_batch()
  random_noise_warmup_loss()
  --noise-warmup-* arguments

scripts/197_run_pure_recursive_depth_supervised_train.sh
  NOISE_WARMUP_* environment variables

tests/test_pure_recursive_depth_supervised_train_script.py
```

Smoke:

```text
command shape:
  NOISE_WARMUP_STEPS=2
  NOISE_WARMUP_SEQ_LEN=8
  NOISE_WARMUP_TARGET_VOCAB_SIZE=128
  STEPS=1
  MAX_CASES=1

checkpoint:
  runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt

summary:
  docs/wiki/decisions/qwen35-2b-noise-warmup-smoke-s001-depth-gate-1-summary.json

raw-depth smoke:
  donor 0/1
  core_off 0/1
  core8 1/1
  status accepted
```

Interpretation:
this is a plumbing acceptance only. It proves the warm-up option runs while the
Qwen donor remains frozen and the standard raw-depth harness still executes. It
does not prove that random-noise warm-up improves QTRM calibration or reasoning.
The next gate must train matched no-warmup vs warmup checkpoints and compare
ECE/Brier/UNKNOWN/OOD/selective accuracy under donor-only, fused, low-donor,
QTRM-only, and core-off modes.

## [2026-05-03] evaluation | Matched random-noise metacognitive smoke gate

Added a choice-score calibration gate:

```text
scripts/202_build_metacognitive_calibration_gate.py
tests/test_metacognitive_calibration_gate_script.py
```

The gate reads raw forced-choice eval JSONL records, converts `choice_scores`
to softmax confidence, then computes:

```text
accuracy
mean confidence
ECE
Brier
average confidence when wrong
per-mode deltas
```

Matched smoke:

```text
baseline:
  checkpoint: runs/qwen35_2b_4090_noise_warmup_matched_no_warmup_s001/last.pt
  eval: runs/eval/qwen35_2b_noise_warmup_matched_no_warmup_s001_depth_gate_1.jsonl

candidate:
  checkpoint: runs/qwen35_2b_4090_noise_warmup_smoke_s001/last.pt
  eval: runs/eval/qwen35_2b_noise_warmup_smoke_s001_depth_gate_1.jsonl

report:
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001.md
  docs/wiki/decisions/noise-warmup-metacognitive-calibration-smoke-s001-summary.json
```

Result:

```text
accuracy: 0.500000 -> 0.500000
ECE:      0.180124 -> 0.176691  delta -0.003433
Brier:    0.247725 -> 0.243216  delta -0.004509
status: accepted
```

Interpretation:
accepted as smoke/evaluator plumbing only. This does not prove robust
metacognition. It proves the harness can detect matched calibration changes
without MemoryOS/retrieval and without making Qwen donor trainable. Next
required step: larger held-out metacognitive dataset with answerable,
UNKNOWN, contradiction, OOD/random-token, and core-off/low-donor comparisons.

## [2026-05-03] architecture | Qwen donor boundary and metacognition risk documented

Decision:
using a frozen Qwen donor is not inherently a problem. It is valid as tokenizer
contract, hidden-state provider, base language-policy baseline, and bounded
residual scaffold. It becomes a problem when donor fluency, donor confidence, or
fused-logit quality is mistaken for QTRM raw intelligence.

Updated:

```text
docs/wiki/sources/random-noise-calibration.md
docs/wiki/decisions/qwen-donor-risk-and-metacognition.md
docs/wiki/index.md
```

New canonical rule:

```text
QTRM claims require donor_only, qtrm_fused, qtrm_only/low_donor, core_off, and
memory_off comparisons. Metacognitive calibration claims additionally require
ECE/Brier/UNKNOWN/OOD/selective-accuracy gates and must show causal drop under
the claimed QTRM component ablation.
```

Random-noise warm-up is now mapped as a QTRM-trainable-path calibration probe:
freeze Qwen donor, warm up only QTRM modules, then measure QTRM-only and fused
calibration before/after. It is not proof of donor-free language ability.

## [2026-05-03] distillation | Subliminal Learning changes Qwen3.6 teacher policy

Reviewed:

```text
paper: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
arXiv: https://arxiv.org/abs/2507.14805
local PDF: references/papers/donor_annealing/2507.14805.pdf
```

Decision:
direct Qwen3.6-to-QTRM teacher-answer imitation is no longer canonical.
Teacher-generated data can carry hidden behavioral signals even when the
surface text is unrelated to the trait and filtered. The risk is especially
relevant for Qwen-family teacher/donor/student setups.

Canonical replacement:

```text
teacher proposes candidates / critiques / hard negatives
-> verifier or gold process decides labels
-> QTRM trains on verified labels and explicit checked preferences
```

Updated:

```text
docs/wiki/sources/donor-annealing-distillation.md
docs/wiki/concepts/donor-annealing-roadmap.md
docs/wiki/decisions/qwen36-online-distillation-roadmap.md
docs/wiki/concepts/cognitive-core-data-quality.md
```

Next data direction:
prefer verified public datasets first: GSM8K, MATH-500, NuminaMath verifiable,
OpenR1 verified answer-only subsets, ProofWriter, CLUTRR, bAbI, MBPP, MBPP+,
and HumanEval. Qwen3.6/GPT-5.5 can still help generate candidates or hard
cases, but not unverified gold labels.

## [2026-05-02] architecture | Raw intelligence gates become first priority

Correction:
the strict mandatory-core 72-case result proves a causal answer path on the
current MemoryOS-style gate, but it is not proof of raw intelligence. ASI
progress now requires no-retrieval recursive-depth and trainable-memory
ablation gates.

Implemented:

```text
src/qtrm_mm/eval/raw_intelligence_gate.py
scripts/190_build_pure_recursive_reasoning_cases.py
scripts/191_build_raw_intelligence_gate.py
scripts/192_eval_raw_intelligence.py
scripts/193_run_pure_recursive_reasoning_depth_gate.sh
data/eval/pure_recursive_reasoning_heldout_72.jsonl
docs/wiki/decisions/raw-intelligence-gates.md
```

Canonical first gate:

```text
donor_only_no_evidence
qtrm_core_off_no_evidence
qtrm_core_steps_1_no_evidence
qtrm_core_steps_2_no_evidence
qtrm_core_steps_4_no_evidence
qtrm_core_steps_8_no_evidence
```

Acceptance requires deep core to beat donor and core-off, plus a positive
depth-scaling gain, with zero MemoryOS/retrieval/hidden-evidence shortcut
records.

First smoke:

```text
report: docs/wiki/decisions/pure-recursive-reasoning-depth-gate-smoke4.md
status: rejected
donor/core_off/core_steps_1/2/4/8: all 1/4
shortcut records: 0
```

This is the correct failure to expose. It means the current mandatory-core
checkpoint is not yet a raw recursive-reasoning model. The next canonical work
is raw-core training or redesign with a core-off causal loss, not more
MemoryOS/RAG/answer-format tuning.

Implemented next training path:

```text
config: configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml
runner: scripts/195_run_pure_recursive_reasoning_core_train.sh
train cases: data/filtered/pure_recursive_reasoning_train256_cases.jsonl
preference rows: data/filtered/pure_recursive_reasoning_preferences_train.jsonl
preference rows count: 640
canonical causal ablation: core_off
```

S160 result:

```text
checkpoint: runs/qwen35_2b_4090_pure_recursive_reasoning_core_s160/last.pt
report: docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md
failure ledger: docs/wiki/decisions/pure-recursive-reasoning-core-s160-failure-ledger.md
status: rejected
donor/core_off/core_steps_1/2/4/8: all 3/8
```

The core path changed some answers, so it is no longer completely inert.
However, depth 1/2/4/8 produced identical scores and outputs. The next
architecture change must supervise depth-progressive state updates; another
answer-only SFT pass is not canonical progress.

## [2026-05-02] architecture | Full-sequence KISS raw-core probes reject current recursive core

Implemented fixes for the raw recursive-depth path:

```text
src/qtrm_mm/qtrm_model.py: final logits now use qtrm_residual_logits without donor logits
scripts/196_train_pure_recursive_depth_supervised.py: full answer-token CE
scripts/196_train_pure_recursive_depth_supervised.py: Cartesian row/depth schedule
scripts/197_run_pure_recursive_depth_supervised_train.sh: TARGET_MODE switch
src/qtrm_mm/training/train.py: core_only trainable policy
configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
```

Result:

```text
failure ledger: docs/wiki/decisions/pure-recursive-depth-fullseq-kiss-failure-ledger.md
KISS plain-coda train8: core_off 7/8, core8 7/8
core_only train8: core_off 2/8, core8 3/8
heldout: no run beats donor and core_off with positive depth scaling
status: rejected
```

Interpretation:
the corrected KISS path proves the prior setup had real SSOT/KISS violations
and training bugs. It also falsifies the stronger claim that the current
recursive core is already a useful raw reasoning engine. Non-core paths can
memorize the tiny train set; the core-only path cannot yet beat donor or show
depth-scaled reasoning.

## [2026-05-02] architecture | Strict mandatory-core answer path passes first 8-case gate

The previous mandatory identity-safe core still allowed the answer path to
behave like `core_off`. The corrected strict path makes the answer residual
require a running core:

```text
canonical prompt tokens
-> frozen donor hidden/logits
-> QTRM workspace
-> mandatory recursive core
-> core-conditioned answer bottleneck
-> QTRM residual logits + donor logits
-> greedy answer
```

New config:

```text
QTRMConfig.answer_bottleneck_requires_core
```

When `answer_bottleneck_requires_core: true`, `core_off` and `workspace_off`
cannot use the answer-bottleneck residual. They fall back to donor-fused output
instead of silently taking a coreless residual shortcut.

Failure steps that led here:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_causal_s080.yaml
report: docs/wiki/decisions/mandatory-identity-core-causal-s080-gate-8.md
status: rejected
reason: core_off same completions 8/8

config: configs/qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120.yaml
report: docs/wiki/decisions/mandatory-core-answer-bottleneck-causal-s120-gate-8.md
status: rejected
reason: bypass removed, but full QTRM still tied donor-only at 5/8
```

The final fine-tune used cleaned intervention preferences:

```text
builder: scripts/186_build_clean_intervention_preferences.py
data: data/filtered/memory_reasoning_intervention_preferences_clean_train24.jsonl
config: configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml
script: scripts/187_run_mandatory_core_intervention_preference_train.sh
checkpoint: runs/qwen35_2b_4090_mandatory_core_intervention_preference_s080/last.pt
```

8-case result:

```text
report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-8.md
status: accepted
causal_gate_status: accepted
full mandatory core: 6/8
donor-only: 5/8
core_off: 5/8
workspace_off: 5/8
same completion rate core_off/workspace_off: 3/8
```

Interpretation:
this is the first accepted small gate for a mandatory-core answer path. It is
not proof of a solved architecture or ASI. It proves that the current candidate
beats donor-only on the 8-case held-out slice and loses that advantage when
core/workspace are disabled. The next required check is 16/32/72 held-out
scale-up plus a failure ledger for the remaining misses.

Scale-up result:

```text
16-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-16.md
16-case full mandatory core: 13/16
16-case donor/core_off/workspace_off: 10/16

32-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-32.md
32-case full mandatory core: 20/32
32-case donor/core_off/workspace_off: 17/32

72-case report: docs/wiki/decisions/mandatory-core-intervention-preference-s080-gate-72.md
72-case full mandatory core: 50/72
72-case donor/core_off/workspace_off: 39/72
72-case core/workspace hit drop: 11
```

This upgrades the candidate from "8-case smoke accepted" to "72-case held-out
root gate accepted" for the strict mandatory-core answer path. Remaining
failure classes are not architectural optional-core failures anymore; they are
mostly abstention and temporal/conflict calibration failures.

## [2026-05-02] architecture | Mandatory identity-safe core replaces coreless as target

Correction:
the latent reasoning loop is the core QTRM claim. `coreless` must not be the
final architecture. It is only a lower-bound/teacher baseline showing that the
old recursive core was harmful when connected directly to answer logits.

Implemented a mandatory, identity-safe core path:

```text
workspace
-> recursive core always runs
-> learned output blend starts near identity(workspace)
-> answer bottleneck / residual governor
-> greedy answer
```

New fields:

```text
QTRMConfig.core_output_blend_enabled
QTRMConfig.core_output_blend_init_bias
QTRMConfig.core_output_blend_min
```

Config and script:

```text
config: configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml
script: scripts/183_run_mandatory_identity_core_candidate_gate.sh
checkpoint: runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt
```

8-case result:

```text
report: docs/wiki/decisions/mandatory-identity-core-candidate-gate-8.md
status: accepted by current broad gate
full mandatory core: 6/8
donor-only: 5/8
core_off: 6/8
workspace_off: 5/8
core_steps: 2 for full path
```

Interpretation:
this is a safe mandatory-core starting point, not proof that the core has
learned useful reasoning yet. `core_off` still matches full output, which is
expected from the identity blend initialization. The next training goal is to
open the core blend only when the loop improves over the lower-bound coreless
teacher.

## [2026-05-02] architecture | Coreless workspace-answer lower-bound passes 8/16-case active gates

After the full intervention-preference checkpoint failed, the `core_off`
ablation exposed a lower-bound/teacher path:

```text
donor + SSOT prompt tokens
-> QTRM workspace answer bottleneck
-> answer residual governor
-> donor-fused greedy answer
```

The recursive core is disabled for this candidate:

```text
config: configs/qwen35_2b_4090_coreless_workspace_answer_candidate.yaml
script: scripts/182_run_coreless_workspace_answer_candidate_gate.sh
checkpoint: runs/qwen35_2b_4090_intervention_preference_train24_s080/last.pt
```

New config support:

```text
QTRMConfig.core_enabled: false
```

This makes the ordinary `qtrm_residual_with_evidence` path behave like the
previous `core_off` ablation, without needing a runtime ablation flag.

Results:

```text
8-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-8.md
8-case status: accepted
8-case QTRM coreless: 6/8
8-case donor-only: 5/8
8-case workspace_off: 5/8

16-case report: docs/wiki/decisions/coreless-workspace-answer-candidate-gate-16.md
16-case status: accepted
16-case QTRM coreless: 11/16
16-case donor-only: 10/16
16-case workspace_off: 10/16
```

Interpretation:
this is not the final QTRM architecture because it removes the latent loop. It
is a lower-bound/teacher baseline for training the mandatory identity-safe core:
the mandatory core must first match this behavior without harm, then beat it
with a core-off causal drop.

## [2026-05-02] architecture | On-policy intervention preference rejected; root path still non-causal

Generated train-split on-policy intervention data from the rejected preference
checkpoint:

```text
train eval: runs/eval/canonical_answer_preference_s160_train24_answer_gate.jsonl
audit: docs/wiki/decisions/canonical-answer-preference-s160-train24-intervention-audit.json
data: data/filtered/memory_reasoning_intervention_preferences_train24.jsonl
rows: 6
preserve_donor: 2
allow_qtrm: 2
suppress_core_override: 2
```

Added:

- `configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml`
- `scripts/181_run_intervention_preference_train.sh`
- `tests/test_intervention_preference_train_script.py`

Held-out result:

```text
report: docs/wiki/decisions/intervention-preference-train24-s080-answer-gate-8.md
active-path report: docs/wiki/decisions/intervention-preference-train24-s080-active-path-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only: 5/8
core_off: 6/8
answer_residual_governor_off: 6/8
intervention audit: docs/wiki/decisions/intervention-preference-train24-s080-intervention-audit.json
donor_hit_qtrm_miss: 0/8
qtrm_hit_donor_miss: 0/8
core_off_beats_qtrm: 1/8
```

Important mixed result:
the old `synthetic-authority-vault-0100` failure improved from
`stone-arch` to `opal-river`, so intervention preference can repair some
unsafe overrides. It still fails the root architecture gate because critical
component-off paths match or beat full QTRM. The next step should be a root
path change, not another local margin/scale/loss tweak.

Evaluation-tool correction:
`scripts/148_build_root_architecture_gate.py` now accepts repeated
`--critical-mode` and `--comparison-mode` overrides. The broad gate still
detects non-causal probe-only paths, while the active-path gate checks only the
components that should matter for the current SSOT answer path. The active-path
gate also rejected, so the conclusion does not depend on inactive probe modes.

## [2026-05-02] architecture | Donor-preserve and preference fixes rejected by strict promotion gate

Tested two donor-preserving follow-ups to the answer residual governor:

- `configs/qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml`
- `configs/qwen35_2b_4090_canonical_answer_preference_s160.yaml`
- `scripts/176_run_canonical_answer_governor_preserve_train.sh`
- `scripts/177_build_canonical_plain_answer_preferences.py`
- `scripts/178_run_canonical_answer_preference_train.sh`

The preference run originally OOMed because it combined rejected-sample
preference forwards with canonical causal ablation forwards. The preference
config now keeps canonical causal ablation loss off during training and relies
on the strict held-out gate after training:

```text
loss_canonical_causal_weight: 0.0
canonical_causal_ablation_modes: []
```

Results:

```text
preserve report: docs/wiki/decisions/canonical-answer-governor-preserve-s120-answer-gate-8.md
preserve status: rejected
preserve full QTRM: 5/8
preserve donor-only: 5/8
preserve core_off: 5/8

preference report: docs/wiki/decisions/canonical-answer-preference-s160-answer-gate-8.md
preference status: rejected
preference causal_gate_status: accepted
preference full QTRM: 5/8
preference donor-only: 5/8
preference answer_residual_governor_off: 4/8
preference core_off: 6/8
```

Interpretation:
donor-correct preservation removed one earlier `core_off > full` symptom in the
preserve run, but did not make QTRM beat donor-only. Preference training did
not fix the key authority-conflict failure: on
`synthetic-authority-vault-0100`, donor/core-off answer `opal-river`, while
full QTRM still answers `stone-arch`. The next step should not be another
local loss. The latent core must be redesigned or gated so it cannot override
donor-correct answers unless it has an independently verified advantage.

Follow-up scale sweep:

```text
report: docs/wiki/decisions/canonical-answer-preference-s160-qscale030-answer-gate-8.md
status: rejected
causal_gate_status: rejected
full QTRM: 5/8
donor-only: 5/8
causal modes: none
```

`QTRM_LOGITS_SCALE=0.30` fixed the `opal-river` override case, but removed the
causal governor drop and hurt abstention behavior. This rejects the simpler
"just lower residual scale" explanation. The failure is intervention policy:
QTRM needs a learned, causal permit/verify path for when it may override donor,
not a global residual scale.

Added:

- `scripts/179_audit_intervention_policy.py`
- `scripts/180_build_intervention_preferences.py`
- `docs/wiki/decisions/intervention-policy-failure-ledger.md`

The diagnostic builder produced 3 held-out correction rows
(`allow_qtrm=1`, `preserve_donor=1`, `suppress_core_override=1`). These rows
are not for final training claims; they define the on-policy correction shape
that should be generated on a train split next.

## [2026-05-02] architecture | Answer residual governor causal signal accepted, strict promotion rejected

Added a donor-preserving answer residual governor:

- `QTRMConfig.answer_residual_governor_enabled`
- `QTRMConfig.answer_residual_governor_init_bias`
- `TrainConfig.loss_answer_residual_governor_weight`
- `src/qtrm_mm/losses.py::answer_residual_governor_loss`
- `qtrm_answer_residual_governor_off_with_evidence` ablation mode
- `configs/qwen35_2b_4090_canonical_answer_governor_s120.yaml`
- `scripts/175_run_canonical_answer_governor_train.sh`

Also tightened the root gate:

```text
causal_gate_status = component changed answer-level behavior
status             = strict promotion result
```

Strict promotion now requires:

```text
full QTRM > donor-only
no critical component-off mode > full QTRM
at least one critical component-off causal drop
```

Result:

```text
report: docs/wiki/decisions/canonical-answer-governor-s120-answer-gate-8.md
status: rejected
causal_gate_status: accepted
full QTRM: 5/8
donor-only with evidence: 5/8
answer_residual_governor_off: 4/8
core_off: 6/8
```

Interpretation:
the governor is a real causal/protective component on this split, but the full
architecture is not promoted. QTRM still ties donor-only, and disabling the
latent core improves the hit count. The next candidate must make the latent core
help rather than merely letting the governor limit damage.

## [2026-05-02] architecture | Plain-answer KISS reset rejected

Added:

- `scripts/173_build_canonical_plain_answer_data.py`
- `scripts/174_run_canonical_plain_answer_kiss_train.sh`
- `configs/qwen35_2b_4090_canonical_plain_answer_kiss_s120.yaml`
- `tests/test_canonical_plain_answer_data.py`
- `tests/test_canonical_plain_answer_train_script.py`
- `docs/wiki/decisions/canonical-greedy-margin-and-plain-answer-kiss-result.md`

Result:

```text
report: docs/wiki/decisions/canonical-plain-answer-kiss-s120-answer-gate-8.md
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 4/8
qtrm_workspace_off_with_evidence: 5/8
qtrm_core_off_with_evidence: 5/8
causal modes: none
```

Interpretation:
matching the training rows to the canonical plain `Answer:` contract reduced
format leakage, but it did not make the latent workspace/core path causal.
Workspace/core-off ablations still matched or improved full QTRM. This rejects
the idea that another local loss tweak on the current donor-fused residual path
is enough.

## [2026-05-02] architecture | Greedy-token margin rejected on canonical gate

Added greedy-token margin pressure to the training loss:

- `src/qtrm_mm/losses.py::greedy_token_margin_loss`
- `TrainConfig.loss_greedy_token_margin_weight`
- `TrainConfig.greedy_token_margin`
- `TrainConfig.greedy_token_margin_only_donor_errors`
- `configs/qwen35_2b_4090_canonical_greedy_margin_s120.yaml`
- `scripts/172_run_canonical_greedy_margin_train.sh`
- `tests/test_greedy_token_margin_train_script.py`

Result:

```text
report: docs/wiki/decisions/canonical-greedy-margin-s120-answer-gate-8.md
status: rejected
donor_only_with_evidence: 5/8
qtrm_residual_with_evidence: 5/8
qtrm_core_to_text_off_with_evidence: 6/8
exact match: 0/8
causal modes: none
```

Interpretation:
the margin objective can pressure the target token against the top non-target
competitor, but the held-out answer gate still does not show latent causality.
The result also exposed a training/eval contract mismatch: decision-token rows
train `Verify/Decision/Answer`, while the canonical gate expects a short greedy
`Answer:` completion.

## [2026-05-02] architecture | KISS/YAGNI/DRY/SSOT contract enforced

Added:

- `src/qtrm_mm/eval/ssot_contract.py`
- `tests/test_ssot_contract.py`
- `docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md`

Implementation:
the canonical answer-path constants and validation now live in one module:

```text
CANONICAL_EVIDENCE_INJECTION = "ssot"
CANONICAL_ANSWER_CHANNEL = "greedy"
```

`scripts/95_eval_memory_retrieval.py` imports that module instead of keeping
its own local validation logic.

Decision:
KISS/YAGNI/DRY/SSOT is now an enforceable engineering contract:

```text
compiled prompt tokens -> donor + QTRM -> greedy autoregressive answer
```

Span-copy, workspace-only evidence, dual evidence, source masks, and post-hoc
answer decision paths remain probe-only until they pass held-out causal
ablation gates.

## [2026-05-02] architecture | Canonical SSoT answer gate added

Added a hard contract for user-facing QTRM answer evaluation:

```text
--require-canonical-ssot
--evidence-injection ssot
--answer-channel greedy
```

Implementation:

- `scripts/95_eval_memory_retrieval.py` now has
  `--require-canonical-ssot` and rejects workspace/dual hidden-evidence or
  span-copy answer channels when that flag is set.
- `scripts/166_run_canonical_ssot_answer_gate.sh` runs the canonical
  autoregressive answer gate against donor-only, full-QTRM, core-off,
  workspace-off, evidence-bottleneck-off, and no-evidence baselines.
- `scripts/153_run_reasoning_safe_span_copy_gate.sh` is explicitly labeled
  `PROBE-ONLY` and now defaults to `EVIDENCE_INJECTION=ssot`.

Decision:
span-copy, hidden workspace evidence, and dual evidence paths remain diagnostic
tools only. They are not accepted as evidence that the main QTRM architecture
can answer as a general LLM. The main gate is now compiled prompt tokens into
donor + QTRM, followed by greedy autoregressive generation.

Smoke result:

- command: `MAX_CASES=2 bash scripts/166_run_canonical_ssot_answer_gate.sh`
  with explicit output paths under `runs/eval/canonical_ssot_answer_gate_smoke2.*`;
- report: `docs/wiki/decisions/canonical-ssot-answer-gate-smoke2.md`;
- summary: `docs/wiki/decisions/canonical-ssot-answer-gate-smoke2-summary.json`;
- status: `rejected`;
- full QTRM accuracy: `1/2 = 0.500`;
- donor-only with evidence accuracy: `1/2 = 0.500`;
- all critical ablations had `hit_drop=0` and `same_completion_rate=1.000`;
- missing critical modes: `none`.

Interpretation:
the canonical SSoT greedy path now gives a sharper failure signal. The current
checkpoint can use visible evidence as donor/prompt context, but the latent
workspace/core/evidence paths are not causally changing the answer in this
smoke. Next work should train the greedy answer path under answer-level causal
pressure instead of relying on span-copy extraction.

## [2026-05-02] eval | Model-only language stability sweep recorded

Ran the first ASI sufficiency sanity gate without MemoryOS:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm \
OUT_DIR=runs/language_stability/asi_sufficiency_20260502 \
bash scripts/152_run_residual_language_stability_sweep.sh
```

Artifacts:

- `runs/language_stability/asi_sufficiency_20260502/summary.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p0.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p05.jsonl`
- `runs/language_stability/asi_sufficiency_20260502/scale_0p10.jsonl`
- `docs/wiki/decisions/asi-sufficiency-gate-2026-05-02.md`

Result:
scales 0.00, 0.05, and 0.10 all had `clean_rate=1.0`,
`repeat_failure_rate=0.0`, `visible_reasoning_rate=0.0`, and
`answer_drift_rate=0.0` over 4 plain prompts.

Decision:
this provisionally accepts the model-only gate only as a narrow repetition and
format sanity check. It does not prove reasoning. The arithmetic sample still
answered incorrectly (`x = 7 - 4 = x + 1` instead of `x = 4`), so the next gate
must measure answer-level correctness and causal latent contribution against
donor-only and component-off ablations.

## [2026-05-02] decision | ASI sufficiency rejected, next gates ranked

Added:

- `docs/wiki/decisions/asi-sufficiency-gate-2026-05-02.md`

Decision:
the current QTRM architecture is not sufficient for a general ASI claim. The
MemoryOS/model boundary is now cleaner, but ASI-oriented progress still needs
separate gates:

1. model-only plain-prompt language competence;
2. causal latent-answer improvement over donor-only and component-off baselines;
3. verified self-improvement runtime with regression protection;
4. long-horizon task-level reward, not only action imitation.

Immediate priority:
run the residual language stability sweep, then SSOT answer ablations, before
adding another large architecture component.

## [2026-05-02] architecture | MemoryOS moved outside model boundary

Corrected the architecture claim boundary:

```text
QTRM model architecture:
canonical token stream -> Qwen donor -> QTRM workspace/core/residual -> answer

QTRM runtime with MemoryOS:
user prompt -> optional retrieval/rerank -> context compiler
-> canonical token stream -> QTRM model
```

Added:

- `docs/wiki/architecture/model-vs-runtime-boundary.md`

Updated:

- `docs/wiki/architecture/qtrm-forward-pass.md`
- `docs/wiki/architecture/canonical-architecture-matrix.md`
- `docs/wiki/architecture/paper-diagram-prompts.md`
- `docs/wiki/concepts/qtrm-terminology.md`
- `docs/wiki/concepts/workspace-memory-architecture.md`
- `docs/wiki/concepts/workspace-evidence-path.md`
- `docs/wiki/decisions/qtrm-goal-and-scope.md`

Boundary rule:
MemoryOS is optional external memory/RAG/runtime infrastructure. It can prepare
the canonical prompt, but it is not an internal QTRM model component. The base
QTRM model must still run on plain prompts without MemoryOS.

## [2026-05-02] architecture | MemoryOS evidence path corrected to SSOT

Corrected the QTRM MemoryOS architecture from a misleading two-path drawing to
a single-source-of-truth evidence path.

```text
User prompt
-> retrieval query
-> MemoryOS retrieval/rerank
-> Context Compiler / chat-template builder
-> one canonical donor-visible token stream
-> Frozen Qwen donor + QTRM core
-> answer channel
```

Implementation changes:

- `scripts/95_eval_memory_retrieval.py` now defaults to
  `--evidence-injection ssot`.
- SSOT span-copy sets `evidence_span_reader_context="input"`, so the evidence
  span reader scores canonical prompt tokens instead of requiring a separate
  workspace evidence text.
- Learned source masks now annotate token spans in the canonical prompt and do
  not include the user prompt after the final evidence source.
- `workspace` and `dual` remain available only as ablation/probe paths.

Claim boundary:
MemoryOS retrieval is not a second model input reality. It is pre-forward
context engineering that must compile back into one chat-template token stream.

## [2026-05-02] experiment | Learned source span-mask accepted

Trained a learned answer-source selector and applied it as a span-logit mask
while preserving the full workspace evidence context.

```text
report: docs/wiki/decisions/evidence-source-selector-truthcal72.md
selector checkpoint: runs/evidence_source_selector_truthcal_s500/selector.pt
selector summary: docs/wiki/decisions/evidence-source-selector-truthcal72-summary.json
answer records: docs/wiki/decisions/evidence-source-selector-span-mask-truthcal72-thr065-records.jsonl
train source selector: 504 / 504 source records, case success 1.0000
heldout source selector: 252 / 252 source records, case success 1.0000
answer accuracy: 71 / 72 = 0.9861
unknown negatives: 24 / 24
gate: accepted learned source span-mask
```

Important correction:
the accepted mechanism is not evidence pruning. The model still receives full
workspace evidence. The learned selector only masks the final span-copy argmax
to answer-bearing source text. This avoids the distribution break that made the
reliability source governor fall to 48/72.

Remaining miss:
`synthetic-authority-vault-0102` selects the correct source but the span reader
returns `no_answer` with high no-answer probability. Next step is no-answer
head recalibration under source-masked decoding.

## [2026-05-02] experiment | Boundary REVISE accepted, source pruning rejected

Added a deterministic answer-renderer `REVISE` branch for evidence span-copy
answers that were cut inside atomic identifiers.

```text
report: docs/wiki/decisions/evidence-span-boundary-revise-truthcal72.md
accepted records: docs/wiki/decisions/evidence-span-truthcal-72-boundary-revision-ablation-records.jsonl
full in-model + boundary revise: 67 / 72 = 0.9306
feature-off: 55 / 72 = 0.7639
decision-head-off: 55 / 72 = 0.7639
revised spans: 6
span-boundary fixes: 5
negative UNKNOWN retained: 24 / 24
gate: accepted narrow renderer REVISE
```

Also tested a non-label reliability source governor:

```text
records: docs/wiki/decisions/evidence-span-truthcal-72-source-governor-records.jsonl
full mode: 48 / 72 = 0.6667
gate: rejected
```

Important correction:
source pruning before answer selection is harmful with the current calibrated
answer-decision path. The next source-selection step should be learned or
distilled and should preserve full evidence unless calibrated confidence is
high.

## [2026-05-02] experiment | In-model answer-decision head accepted

Moved the answer-decision gate into an ablatable QTRM checkpoint path.

```text
report: docs/wiki/decisions/inmodel-answer-decision-head-truthcal-s200.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl
train mix: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
bootstrap script: scripts/164_bootstrap_answer_decision_feature_head.py
full in-model: 62 / 72 = 0.8611
feature-off: 49 / 72 = 0.6806
decision-head-off: 49 / 72 = 0.6806
blocked candidates: 14
block improved: 13
block harmed: 0
remaining expected-unknown false positives: 0
gate: accepted in-model causal gate
```

Important correction:
hidden-only and linear telemetry variants were insufficient. The accepted
variant uses raw answer-channel telemetry through a feature MLP in the QTRM
checkpoint and keeps feature-off/head-off ablations to prove causality. The
next failure class is positive wrong answers, so the next architecture branch
should be `REVISE` or `SEARCH_MORE`, not more UNKNOWN blocking.

## [2026-05-01] experiment | Action loop fixed, answer bottleneck identified

Trained an action-first strict runtime transition-state controller and reran
the task-level answer gate.

Action-policy result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_actionfirst_s200/last.pt
summary: docs/wiki/decisions/transition-state-controller-runtime-actionfirst-s200-summary.json
train sequences: 256
state_loss_weight: 0.2
held-out action accuracy: 1.0000
RETRIEVE_MEMORY: 72 / 72
VERIFY_EVIDENCE: 72 / 72
ANSWER: 72 / 72
zero_transition_state: 0.3333
transition_state_drop: 0.6667
state_prediction_binary_accuracy: 0.9913
gate: accepted
```

Task-level answer result:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-actionfirst-gate.md
learned_state_qtrm: 4 / 8 = 0.5000
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 1.0000
gate: rejected
```

Interpretation:
the previous near-miss answer gain was partly an accidental UNKNOWN from an
action failure. Once action stability is fixed, the learned loop matches the
scripted answer baseline. The bottleneck is answer formation after
verification, not retrieval or action ordering.

Follow-up answer-channel probe:

```text
config: configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml
checkpoint: runs/qwen35_2b_4090_evidence_span_reader_truthcal_s300/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
answer_channel: evidence_span_copy
truth_gate: true
qtrm_residual_with_evidence: 49 / 72 = 0.6806
span_reader_off: 24 / 72 = 0.3333
donor_only: 24 / 72 = 0.3333
```

Conclusion:
the next architecture step is an answer-decision loop where verifier results
can choose `ANSWER`, `ABSTAIN`, `REVISE`, or `SEARCH_MORE`; another fixed
retrieve-verify-answer controller will not prove reasoning progress.

Static answer-decision threshold probe:

```text
script: scripts/160_calibrate_answer_decision_gate.py
report: docs/wiki/decisions/answer-decision-gate-truthcal-72.md
calibration baseline -> gated: 0.6111 -> 0.8056
heldout baseline -> gated: 0.7500 -> 0.4167
heldout false positives: 2 -> 0
heldout blocked positives: 16
gate: rejected
```

Conclusion:
a static threshold can remove false positives but over-abstains badly on
heldout. The next verifier must be trained with counterfactual positives and
negatives rather than tuned as a scalar threshold.

Learned answer-decision head:

```text
script: scripts/161_train_answer_decision_head.py
report: docs/wiki/decisions/answer-decision-head-truthcal-train144-eval72.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
train records: docs/wiki/decisions/evidence-span-truthcal-train144-answer-channel-records.jsonl
eval records: docs/wiki/decisions/evidence-span-truthcal-72-answer-channel-records.jsonl
train baseline -> learned: 0.7569 -> 0.9444
eval baseline -> learned: 0.6806 -> 0.8611
eval false positives: 13 -> 0
eval block improved: 13
eval block harmed: 0
gate: accepted
```

Conclusion:
the answer-decision signal is learnable. The next architecture step is to wire
this decision into the runtime answer loop and then move it from a post-hoc MLP
into an ablatable in-model QTRM head.

Runtime answer-decision integration:

```text
script: scripts/95_eval_memory_retrieval.py
report: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
decision checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
baseline span/truth: 49 / 72 = 0.6806
runtime decision: 62 / 72 = 0.8611
blocked candidates: 14
expected-unknown false positives: 0
gate: accepted runtime integration
```

Conclusion:
the answer-decision improvement survives the real eval path. The remaining
failure class is positive wrong answers in conflict or multi-hop cases, which
requires a `REVISE` or `SEARCH_MORE` action rather than more UNKNOWN blocking.

## [2026-05-01] experiment | Strict runtime learned-state answer-loop near-miss

Added a task-level gate for the learned transition-state controller. Unlike the
previous smoke, runtime rows hide trace-step and phase-specific state-summary
text. The controller prompt also hides evidence until `RETRIEVE_MEMORY` places
retrieved evidence into the previous-observation path.

Changed:

- `scripts/159_eval_learned_state_answer_loop.py`;
- `tests/test_learned_state_answer_loop_script.py`;
- `scripts/158_train_transition_state_controller.py --strict-runtime-state-inputs`;
- `docs/wiki/decisions/transition-state-controller-runtime-state-s120.md`;
- `docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md`.

Strict runtime-state controller result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_state_s120/last.pt
held-out action accuracy: 0.9630
RETRIEVE_MEMORY: 0.8889
VERIFY_EVIDENCE: 1.0000
ANSWER: 1.0000
zero_transition_state: 0.3333
transition_state_drop: 0.6296
state_prediction_binary_accuracy: 0.8287
gate: rejected
```

Task-level answer-loop gate:

```text
report: docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md
learned_state_qtrm: 5 / 8 = 0.6250
scripted_qtrm_answer_channel: 4 / 8 = 0.5000
scripted_donor_answer_channel: 4 / 8 = 0.5000
state_off: 2 / 8 = 0.2500
action_success_rate: 7 / 8 = 0.8750
gate: rejected
failed_check: learned_action_loop_not_stable
```

Interpretation:
this is the first small task-level answer reward gain from the learned-state
loop, but it is not accepted. The next fix should target first-step action
stability under strict runtime inputs, not add another answer-format guard.

## [2026-05-01] experiment | Learned transition-state controller smoke

Added a learned transition-state predictor on top of QTRM row features. The
controller no longer receives `previous_action`, and its direct QTRM feature
path is disabled with `controller_feature_scale=0.0`, so action selection must
go through the predicted state vector.

Changed:

- `TransitionStatePredictor` and `transition_state_prediction_loss`;
- `scripts/158_train_transition_state_controller.py --learn-transition-state`;
- state-prediction gate requiring held-out binary accuracy >= `0.90`;
- `docs/wiki/decisions/transition-state-controller-learned-state-smoke.md`.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_learned_state_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-learned-state-smoke-summary.json
feature_scale: 1.0
controller_feature_scale: 0.0
learn_transition_state: true
use_prev_action: false
held-out eval_full: 1.0000
held-out state_prediction_binary_accuracy: 0.9974
zero_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this is the first learned-state loop smoke, but it is still a trace-phase task.
It does not prove task-level answer reward, factual verification, or a general
world model. The next gate must run the learned-state loop through answer
generation and compare against scripted/donor harnesses with state/world/verifier
ablations.

## [2026-05-01] experiment | Explicit observation/verifier transition-state smoke

Extended the transition controller so the action loop can read an explicit
state vector derived from the previous observation, reward, and previous
world/verifier signals. This test disables QTRM latent features
(`feature_scale=0.0`) and disables `previous_action`, so the loop cannot pass by
memorizing the fixed action order alone.

Changed:

- `TransitionStateController` now accepts `transition_state_dim` and supports
  transition-state ablations at inference time;
- `scripts/158_train_transition_state_controller.py` adds
  `--use-transition-state` and builds 9 inspectable state features;
- sequence grouping now uses `task_id + hash(prompt, workspace)` to preserve
  augmented variants;
- tests cover explicit state input, state collation, and variant grouping;
- `docs/wiki/decisions/transition-state-controller-explicit-state-smoke.md`.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_explicit_state_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-explicit-state-smoke-summary.json
feature_scale: 0.0
use_transition_state: true
use_prev_action: false
reset_hidden: true
held-out eval_full: 1.0000
zero_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this proves causal explicit observation/verifier-state wiring, not learned
world-model reasoning. The next gate must replace hand-built state features
with learned observation/world/verifier predictors and compare against this
scripted-state baseline.

## [2026-05-01] experiment | Explicit transition-state controller smoke

Added a sequence-level transition controller after the learned-signal collapse.
The key change is that action policy is no longer treated as independent
per-row classification. Trace rows are grouped by `task_id` and learned as a
transition sequence.

Added:

- `src/qtrm_mm/agentic/transition_controller.py`;
- `scripts/158_train_transition_state_controller.py`;
- `tests/test_transition_state_controller.py`;
- `return_features_only=True` in `QTRMMultimodalModel.forward`;
- `docs/wiki/decisions/transition-state-controller-markov-smoke.md`.

Important loader fix:
the signal trace file had duplicate `0,1,2` rows under the same `task_id`.
The transition sequence reader now deduplicates by `step`; otherwise the model
sees contradictory previous-action transitions.

Result:

```text
checkpoint: runs/qwen35_2b_4090_transition_state_controller_markov_smoke/last.pt
summary: docs/wiki/decisions/transition-state-controller-markov-smoke-summary.json
controller_mode: explicit_markov_transition_state
feature_scale: 0.0
use_prev_action: true
reset_hidden: true
held-out eval_full: 1.0000
reset_transition_state: 0.3333
transition_state_drop: 0.6667
gate: accepted
```

Boundary:
this proves only the explicit transition-state action loop, not QTRM latent
reasoning. `feature_scale=1.0` and hidden-only recurrent variants failed on
held-out, so QTRM feature grounding remains the next unsolved problem.

## [2026-05-01] experiment | Learned controller-signal replacement rejected

Replaced the oracle `controller_signal` scaffold with a learned
core-derived signal head and evaluated two variants:

```text
full learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_s300/last.pt
  held-out action accuracy: 0.3333
  collapse mode: predicts VERIFY_EVIDENCE for all rows
  gate status: rejected

head-only learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_head_s300/last.pt
  held-out action accuracy: 0.3333
  collapse mode: predicts ANSWER for all rows
  qtrm_controller_signal_off: 0.3333
  qtrm_verifier_off: 0.3333
  gate status: rejected

learned-readout diagnostic:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_readout_s300/last.pt
  held-out action accuracy: 0.3704
  collapse mode: mostly ANSWER, with 8/72 VERIFY_EVIDENCE correct
  qtrm_latent_core_off: 0.5926
  qtrm_workspace_off: 0.6296
  gate status: rejected
```

Interpretation:
the oracle signal path is causally useful, but a stateless per-row signal head
over the current latent core is not enough to replace it. A prompt/coda readout
diagnostic also fails, so the problem is not just `z_h` pooling. This is not a
language-model quality issue; it is an architecture issue. The next controller
design must learn a transition-state policy or recurrent planner over the trace,
not only a two-bit bottleneck readout from one frozen latent state.

## [2026-05-01] experiment | Controller-signal causal-loop scaffold

Added a Stage-1.5 controller-signal path so future learned world-model and
verifier outputs can affect the action controller through a measurable tensor
path.

Added:

- `controller_signal_enabled` and `controller_signal_dim` in `QTRMConfig`;
- `controller_signal_proj` in `QTRMModel`;
- `controller_signal` parsing and batching in `JsonlTextVisionDataset`;
- signal-conditioned trace replay rows in
  `scripts/155_build_controller_trace_replay.py`;
- signal-aware policy eval and ablation modes in
  `scripts/156_eval_controller_trace_policy.py` and
  `scripts/157_eval_asi_controller_causal_loop.py`;
- `configs/qwen35_2b_4090_controller_signal_s300.yaml`;
- `docs/wiki/decisions/asi-controller-signal-causal-loop-s300.md`.

Run:

```text
checkpoint: runs/qwen35_2b_4090_controller_signal_s300/last.pt
train rows: data/filtered/asi_controller_signal_trace_replay.jsonl
held-out rows: data/eval/asi_controller_signal_trace_replay_heldout_72.jsonl
final train action_policy loss: 0.1599
final train batch action_acc: 1.0000
held-out action accuracy: 0.9444
```

Causal-loop ablation:

```text
qtrm_harness: 0.9444
qtrm_world_model_off: 0.3333
qtrm_verifier_off: 0.6111
qtrm_controller_signal_off: 0.3333
qtrm_latent_core_off: 1.0000
standard gate status: rejected
```

Interpretation:
the new signal path is causal because zeroing the world-model/verifier signal
dimensions sharply reduces held-out action accuracy. This is still an oracle
scaffold, not a learned world-model/verifier proof. The ASI gate correctly
rejects because QTRM does not beat the scripted/donor harness baselines and the
latent core is bypassed rather than causally required.

## [2026-05-01] architecture | ASI root architecture reset

Reset the recommended direction away from free-form donor replacement and
toward a verified cognitive loop:

```text
typed context tape -> MemoryOS/search -> shared evidence SSoT
-> Qwen donor prior + QTRM recurrent residual controller
-> world-model prediction -> candidate action or answer
-> verifier gate -> answer/tool/memory write
-> trace store -> verified training buffer
```

Added:

- `src/qtrm_mm/agentic/cognitive_loop.py`;
- `src/qtrm_mm/agentic/context_tape.py`;
- `src/qtrm_mm/agentic/causal_gate.py`;
- `src/qtrm_mm/agentic/harness.py`;
- `src/qtrm_mm/agentic/trace_replay.py`;
- `tests/test_asi_cognitive_loop_contract.py`;
- `tests/test_asi_causal_loop_gate_script.py`;
- `scripts/154_build_asi_causal_loop_gate.py`;
- `docs/wiki/decisions/asi-root-architecture-reset.md`;
- new ASI roadmap PDFs for agent harness residual-role, autonomous agent
  memory, and autonomous memory agents.

Interpretation:
QTRM should be measured as a residual cognitive controller over donor and
harness baselines. Generated memory writes default to quarantine unless
grounding, contradiction, novelty, usefulness, and regression gates pass.
The Stage-0 scripted harness now records
`RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER` transitions so future QTRM
controller claims have a simple baseline to beat.
The harness now uses `TypedContextTape` as a single source of truth; prompt,
workspace, verifier input, and replay records share the same `context_hash`.
The ASI causal-loop gate rejects QTRM claims unless QTRM beats donor/scripted
harness baselines and the latent-core/world-model/verifier-off ablations drop.

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_asi_cognitive_loop_contract.py \
  tests/test_asi_causal_loop_gate_script.py -q
# 8 passed
```

## [2026-05-01] research | ASI-oriented literature roadmap

Reviewed current self-improvement, agent RL, world-model, RAG verification, and
evolutionary-code-discovery papers as architecture constraints rather than ASI
claims.

Added:

- `docs/wiki/decisions/asi-research-roadmap.md`;
- ASI wiki index link;
- ASI roadmap PDFs under `references/papers/asi_roadmap`;
- `docs/REFERENCE_BASELINE.md` entries for the new paper set;
- 2026 review delta for SEAL, RAGEN, Agent Lightning, AlphaEvolve/CodeEvolve,
  V-JEPA 2/LeWorldModel, and validated RAG write-back.

Boundary:

```text
ASI is the ambition, not the current claim.
Progress requires causal gates for evidence, latent core, world model,
verification, self-improvement, and long-horizon agency.
```

## [2026-05-01] implementation | Dual-path evidence conditioning

Corrected the evidence-routing architecture. Workspace-only evidence remains a
causality probe, but the practical RAG-compatible path now keeps retrieved
evidence in the visible prompt and also encodes the same evidence as workspace
memory.

Added:

- `--evidence-injection dual` in `scripts/95_eval_memory_retrieval.py`;
- `build_shared_evidence_context()` as the SSoT evidence materialization point;
- `build_case_prompt_and_workspace_memory(..., evidence_injection="dual")`;
- `workspace_evidence_injection_mode: dual` support in
  `JsonlTextVisionDataset`;
- `EVIDENCE_INJECTION="${EVIDENCE_INJECTION:-dual}"` in
  `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- wiki updates in `workspace-evidence-path.md` and the canonical architecture
  matrix.

Boundary:

```text
workspace = hidden-only causality probe
prompt    = ordinary visible RAG context
dual      = final architecture candidate: one shared evidence context
            -> visible RAG view + latent workspace view
```

Smoke run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=2 \
  OUT=runs/eval/reasoning_safe_span_copy_dual_smoke_2.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_dual_smoke_2_audit.jsonl \
  ROOT_MD=docs/wiki/decisions/reasoning-safe-span-copy-dual-smoke-2.md \
  ROOT_JSON=docs/wiki/decisions/reasoning-safe-span-copy-dual-smoke-2-summary.json \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Result: `qtrm_residual_with_evidence` was `2/2` on the two-case smoke; the root
gate was accepted. This only proves the dual path executes and preserves the
small span-copy behavior. It does not yet prove broad generation quality.

## [2026-05-01] implementation | Truth-gated evidence answer channel

Clarified the architecture boundary: Loop-RM/TRM-style recurrence is a latent
compute substrate, not truth judgment by itself. Truth-sensitive answering now
has an explicit runtime gate in the `evidence_span_copy` answer channel.

Added:

- `evidence_truth_gate_from_outputs()` in
  `scripts/95_eval_memory_retrieval.py`;
- `--truth-gate` plus support/causal/refute/missing thresholds;
- span-copy blocking with `answer_channel_meta.status=truth_gate_blocked`;
- optional `TRUTH_GATE=1` runner support in
  `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- concept note:
  `docs/wiki/concepts/truth-gated-evidence-answer-channel.md`.

Current interpretation:

```text
Loop/TRM core = repeated latent state update capacity
truth gate = explicit support/refute/missing/causal answer-path contract
```

The gate is disabled by default until calibration is proven on held-out
truth-gated evals.

Smoke eval:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src TRUTH_GATE=1 \
  MAX_CASES=4 \
  OUT=runs/eval/reasoning_safe_span_copy_truth_gate_4.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_truth_gate_4_audit.jsonl \
  ROOT_MD=docs/wiki/decisions/reasoning-safe-span-copy-truth-gate-4.md \
  ROOT_JSON=docs/wiki/decisions/reasoning-safe-span-copy-truth-gate-4-summary.json \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Result: `qtrm_residual_with_evidence` was `2/4`; the root gate was
`rejected`. Metadata shows the current head logits sit near `0.5`, so
`support_low` blocks valid multi-hop answers by tiny margins. Conclusion:
runtime wiring is correct, but the logical heads need explicit calibration
training before `TRUTH_GATE=1` should be the default.

## [2026-05-01] experiment | Reasoning-safe span-copy gate improves hidden-evidence QA

Implemented a confidence-gated `evidence_span_copy` answer channel so QTRM can
improve evidence-sensitive reasoning without touching donor language logits.

Added:

- `--evidence-span-min-score` to `scripts/95_eval_memory_retrieval.py`;
- low-span-score abstention: selected spans below the confidence floor return
  `Answer: UNKNOWN`;
- `scripts/153_run_reasoning_safe_span_copy_gate.sh`;
- tests for the CLI, low-score abstention, and runner defaults.

Run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT=runs/eval/reasoning_safe_span_copy_confidence_gate_16_runner_full.jsonl \
  AUDIT_OUT=runs/eval/reasoning_safe_span_copy_confidence_gate_16_runner_full_audit.jsonl \
  bash scripts/153_run_reasoning_safe_span_copy_gate.sh
```

Settings:

```text
checkpoint: runs/qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt
answer_channel: evidence_span_copy
evidence_injection: workspace
no_answer_threshold: 0.1
min_span_score: 12
max_cases: 16
```

Result:

| Mode | Hits | Accuracy |
| --- | ---: | ---: |
| `donor_only_with_evidence` | `5/16` | `0.3125` |
| `qtrm_residual_with_evidence` | `13/16` | `0.8125` |
| `qtrm_workspace_off_with_evidence` | `5/16` | `0.3125` |
| `qtrm_workspace_memory_off_with_evidence` | `5/16` | `0.3125` |
| `qtrm_evidence_span_reader_off_with_evidence` | `5/16` | `0.3125` |

Compared with the previous uncalibrated span-copy baseline,
`qtrm_residual_with_evidence` improved from `11/16` to `13/16` by abstaining on
low-confidence spans. Root gate:
`docs/wiki/decisions/reasoning-safe-span-copy-confidence-gate.md` reports
`accepted` with causal drops for workspace, workspace-memory, and evidence-span
reader ablations.

Interpretation:
this is a real reasoning-path improvement under the language-safe constraint:
the donor's surface language policy is not modified, and the answer is produced
through a bounded span-copy/UNKNOWN channel. It is not yet proof that the
recursive core is doing the reasoning. `core_off`, `core_context_off`, and
`evidence_bottleneck_off` matched the baseline, so the current causal path is
best described as:

```text
hidden workspace evidence -> prompt-conditioned evidence span reader
-> confidence/UNKNOWN gate -> short copied answer
```

## [2026-05-01] experiment | Residual language-stability sweep with donor preserved

Implemented and ran a repeatable residual language-stability sweep for the
current donor-backed residual candidate.

Added:

- `--donor-logits-scale`, `--qtrm-logits-scale`, and
  `--qtrm-residual-clamp` overrides to `scripts/92_eval_qtrm_logits.py`;
- `--stop-after-sentence` and `--min-new-tokens-before-stop` to the same eval
  path, matching the interactive language-safe guard;
- `scripts/152_run_residual_language_stability_sweep.sh`;
- tests for the new CLI controls and sweep runner.

Run:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  OUT_DIR=runs/language_stability/residual_s010_safe_sweep_96 \
  SCALES="0.0 0.05 0.10" MAX_NEW_TOKENS=96 \
  bash scripts/152_run_residual_language_stability_sweep.sh
```

Result:

| QTRM scale | Records | Clean rate | Repeat failure | Visible reasoning | Answer drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |
| `0.05` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |
| `0.10` | 4 | `1.00` | `0.00` | `0.00` | `0.00` |

Artifact:

```text
runs/language_stability/residual_s010_safe_sweep_96/summary.jsonl
```

Interpretation:
`qtrm_logits_scale <= 0.10` is language-stable under the guarded donor-backed
runtime on these Korean/English/math smoke prompts. This is not yet evidence
that QTRM improves reasoning: residual telemetry showed no donor argmax shifts
on the prompt probes, and the math prompt's wrong answer was already present in
donor-only output. The next gate must measure QTRM usefulness on
evidence-sensitive tasks against donor-only, not only fluency preservation.

## [2026-05-01] implementation | Language-safe donor-preserving inference default

Fixed the immediate language-collapse path in interactive inference.

Diagnosis:
`scripts/90_infer_with_donor.sh` used config defaults where
`qtrm_logits_scale=1.0` and `donor_logits_scale=0.0`, so the "QTRM + donor"
script was actually free-running from the immature QTRM language head. That is
why old checkpoints produced attractors such as `Freeze Freeze ...` or
`world of the world ...`.

Default runtime guard:

```text
LANGUAGE_SAFE=1
DONOR_LOGITS_SCALE=1.0
QTRM_LOGITS_SCALE=0.0
QTRM_RESIDUAL_CLAMP=0.0
SUPPRESS_VISIBLE_REASONING=1
NO_REPEAT_NGRAM_SIZE=2
STOP_AFTER_SENTENCE=1
```

This is a donor-preserving inference mode, not proof that QTRM has learned a
standalone language policy. Raw QTRM experiments remain available with
`LANGUAGE_SAFE=0` and explicit scale overrides. History rows use mode
`language_safe_donor` for the default guard so these samples are not confused
with QTRM-residual capability evidence.

Smoke result:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  CONFIG=configs/qwen35_2b_4090_extended.yaml \
  CHECKPOINT=runs/qwen35_2b_4090_extended/last.pt \
  HISTORY_JSONL=runs/history/generations/language_safe_smoke.jsonl \
  bash scripts/90_infer_with_donor.sh "양자 컴퓨팅이란 무엇인가요?"
```

Output:

```text
양자 컴퓨팅이란 무엇인가요?

양자 컴퓨팅은 양자역학의 원리를 이용하여 컴퓨터를 설계하는 것을 말합니다.

Generated 22 new tokens
```

Verification:

```bash
bash -n scripts/90_infer_with_donor.sh
PYTHONPATH=src uv run python -m py_compile \
  src/qtrm_mm/history.py scripts/95_eval_memory_retrieval.py
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_infer_with_donor_script.py \
  tests/test_generation_history.py \
  tests/test_memory_eval_script.py \
  tests/test_losses.py -q
# 51 passed
```

## [2026-05-01] implementation | Generation history JSONL

Added persistent generation history so QTRM outputs can compound into failure
ledgers, regression tests, and curated training data instead of disappearing in
terminal scrollback.

Added:

- `src/qtrm_mm/history.py`
- `tests/test_generation_history.py`
- `HISTORY_JSONL=auto` support in `scripts/90_infer_with_donor.sh`
- `--history-jsonl-out auto|none|PATH` support in
  `scripts/95_eval_memory_retrieval.py`
- concept page: `docs/wiki/concepts/generation-history.md`

Default paths:

```text
runs/history/generations/YYYY-MM-DD.jsonl
runs/history/evals/YYYY-MM-DD.jsonl
```

Rule:
raw history must not be used as training data directly. It first needs a
validator or human label so that repetition, decoy-copy, hallucination, and
format leakage become explicit failure classes rather than self-training noise.

## [2026-05-01] decision | Limit-aware model objective

Clarified the "model with no limitations" goal. The wiki now treats it as a
limit-aware architecture objective, not as a literal claim that any fixed model
has no limits.

Added:

- `docs/wiki/decisions/limit-aware-model-objective.md`

Key rule:

```text
no known failure class remains unmeasured, unhandled, undocumented, or without
a falsifiable improvement plan
```

Also linked the objective from the limitations roadmap and wiki index. This
keeps future architecture work tied to explicit failure ledgers, prior research,
ablation gates, and wiki preservation instead of adding complexity blindly.

## [2026-05-01] research-driven debug | Answer-channel contract beats post-hoc rerank

Applied the root-architecture doubt gate to the generation verifier path.

Root architecture claim under test:
post-hoc generation-verifier reranking can improve QTRM sampled candidates.

Falsifier:
reranking lowers format-aware candidate quality below the baseline first
candidate.

Result:

- 8 prompts, 3 sampled candidates each;
- baseline quality rate under format-aware labels: `0.625`;
- verifier-reranked quality rate: `0.250`;
- oracle quality rate: `0.750`;
- selected candidate changed on `0.750` of prompts;
- conclusion: current verifier reranker is rejected.

The larger observed failure is visible answer-channel leakage. Raw QTRM/Qwen
sampling frequently emits `<think>` blocks or meta-reasoning before the answer.
Official Qwen docs describe non-thinking generation controls through
`enable_thinking=False` and `/no_think`; QTRM's raw donor-backed path did not
have an equivalent visible-answer contract.

Added:

- `--suppress-visible-reasoning-tokens` to `scripts/92_eval_qtrm_logits.py`;
- `SUPPRESS_VISIBLE_REASONING=1` support in `scripts/90_infer_with_donor.sh`;
- `scripts/147_summarize_generation_format.py`;
- format-aware `quality_target` in
  `scripts/141_build_generation_verifier_dataset.py`;
- concept page:
  `docs/wiki/concepts/answer-channel-contract.md`.

20-prompt single-candidate comparison with drift-aware format detector:

| Mode | Visible Reasoning | Repeat Failure | Answer Drift | Clean / Quality-Like |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.150 | 0.300 | 0.200 | 0.550 |
| suppress `<think>` + no-repeat-2 | 0.000 | 0.000 | 0.150 | 0.850 |
| direct prompt contract + suppress + no-repeat-2 | 0.000 | 0.000 | 0.300 | 0.700 |

Decision:
answer-channel and prefix-time degeneration controls are higher priority than
post-hoc verifier reranking. Narrow `<think>` suppression plus no-repeat-2 is
the current best runtime guard on this smoke. The direct prompt suffix is
rejected as a default because it increases instruction/answer drift with the raw
base donor.

## [2026-05-01] research-driven debug | Generation verifier split calibration

Failure ledger:

- Failure: generation verifier had in-sample best-threshold signal but no
  held-out calibration proof; fixed `0.5` thresholds were poor and stop was
  weak.
- Evidence: all-50 smoke best-threshold F1 was `0.757` repeat and `0.800`
  quality, but fixed-threshold F1 was poor; stop best-threshold F1 was `0.250`.
- Likely component: coda-text verifier has signal, but calibration and stop
  labels are underpowered.
- Prior work checked: unlikelihood training, SimCTG/contrastive search, FUDGE
  future discriminators, and generator-reranker systems.
- Recommended candidate: full-candidate verifier reranker path, starting with a
  train/calibration/holdout split. This is the smallest falsifiable step before
  a hard decoding gate.

Added:

- source notes:
  `docs/wiki/sources/generation-verifier-reranking.md`;
- deterministic splitter:
  `scripts/144_split_generation_verifier_dataset.py`;
- calibration evaluator:
  `scripts/145_calibrate_generation_verifier_eval.py`;
- split smoke config:
  `configs/qwen35_2b_4090_generation_verifier_split_s020.yaml`;
- tests:
  `tests/test_generation_verifier_split_script.py` and
  `tests/test_generation_verifier_calibration_script.py`.

Split result from the 50-row verifier dataset:

- train: 29 rows, 11 repeat failures, 3 stop failures, 16 quality passes;
- calibration: 10 rows, 3 repeat failures, 1 stop failure, 6 quality passes;
- holdout: 11 rows, 4 repeat failures, 2 stop failures, 6 quality passes.

Ran the split-only 20-step verifier smoke from:

```text
data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier_train.jsonl
```

Saved:

```text
runs/qwen35_2b_4090_generation_verifier_split_s020/last.pt
```

Calibration-selected holdout report:

| Head | Calibration Threshold | Calibration F1 | Holdout F1 | Holdout Precision | Holdout Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat | 0.4948 | 0.667 | 0.571 | 0.667 | 0.500 |
| stop | 0.3495 | 0.286 | 0.222 | 0.143 | 0.500 |
| quality | 0.4918 | 0.800 | 0.750 | 0.600 | 1.000 |

Decision: repeat and quality retain weak held-out signal; stop does not. The
next architecture experiment should be candidate reranking with verifier scores
and a larger on-policy generated dataset, not hard threshold gating.

## [2026-05-01] implementation | Generation verifier head smoke

Implemented the first QTRM-generated output-failure verifier path:

- config flags:
  `generation_verifier_enabled` and
  `loss_generation_verifier_weight`;
- model heads:
  `generation_repeat_head`, `generation_stop_head`, and
  `generation_quality_head`;
- data labels:
  `generation_verifier_repeat_target`,
  `generation_verifier_stop_target`,
  `generation_verifier_quality_target`, and
  `generation_verifier_sample_weight`;
- trainable policy:
  `generation_verifier_only`;
- dataset builder:
  `scripts/141_build_generation_verifier_dataset.py`;
- smoke train runner:
  `scripts/142_run_generation_verifier_s020.sh`;
- evaluator:
  `scripts/143_eval_generation_verifier.py`;
- concept page:
  `docs/wiki/concepts/generation-verifier.md`.

Built the first 50-row verifier dataset from QTRM v2 generated outputs:

- source eval:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_eval.jsonl`;
- prompt metadata:
  `runs/qwen_scope/qtrm_repeat_gate_prompts_s50.jsonl`;
- output:
  `data/filtered/qtrm_generated_verifier/qtrm_v2_repeat_gate_s50_generation_verifier.jsonl`;
- label distribution:
  18 repeat failures, 6 stop failures, 28 quality passes.

Important architecture correction: the first verifier-head placement used pooled
`z_h` and learned mostly label priors. The heads now read the coda/post-norm
last valid text hidden state, which has access to the actual candidate
completion. This makes the verifier an output-failure reader rather than a
latent-state prior classifier.

Second 20-step verifier-only smoke:

- checkpoint:
  `runs/qwen35_2b_4090_generation_verifier_s020/last.pt`;
- eval:
  `runs/qwen35_2b_4090_generation_verifier_s020/eval_generation_verifier_s50.json`.

In-sample evaluation after the coda-text fix:

| Head | Fixed 0.5 F1 | Best in-sample F1 | Decision |
| --- | ---: | ---: | --- |
| repeat | 0.100 | 0.757 | Signal exists, calibration needed. |
| stop | 0.000 | 0.250 | Weak; needs more positives and better labels. |
| quality | 0.000 | 0.800 | Signal exists, calibration needed. |

Decision: keep generation verifier as probe-only. It should next be evaluated
on a larger train/holdout split and used as a reranker before any hard decoding
gate. Do not claim repetition is solved from this smoke.

## [2026-05-01] implementation | Qwen-Scope repeat candidate scorer

Added a Qwen-Scope repeat-candidate scoring path:

- API: `score_qwen_scope_candidate_features` in `src/qtrm_mm/qwen_scope.py`;
- CLI: `scripts/138_score_qwen_scope_repeat_candidates.py`;
- tests:
  `tests/test_qwen_scope.py` and `tests/test_qwen_scope_score_script.py`;
- artifact:
  `runs/qwen_scope/qtrm_v2_repeat_candidate_scores.json`.

The scorer takes layer-specific SAE feature candidates and reports prompt-level
hit counts, summed activation values, max activation, and optional generation
repeat metrics from QTRM eval JSONL.

First actual-v2 result with candidates `12:847` and `23:29838,31860`:

- prompt `3`: `repeated_2gram_rate=0.254`, score `78.820`, rank 1;
- prompt `2`: `repeated_2gram_rate=0.762`, score `46.713`, rank 2;
- prompt `0`: `repeated_2gram_rate=0.016`, score `23.338`, false-positive
  risk;
- prompt `4`: `repeated_2gram_rate=0.175`, score `0.500`, missed mild-repeat
  risk.

Conclusion: Qwen-Scope candidates are useful as a severe-loop diagnostic smoke
signal, especially for prompt-copy loops, but they are not yet a clean binary
repetition detector or causal control feature. The next gate needs 50-100
generated samples and transparent layer-23 rank/value thresholds before any
training or decoding governor is built from these features.

Ran the 50-sample gate:

- prompt suite:
  `runs/qwen_scope/qtrm_repeat_gate_prompts_s50.jsonl`;
- QTRM eval:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_eval.jsonl`;
- Qwen-Scope records:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_layers_12_23.jsonl`;
- threshold summaries:
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_threshold_summary_rep15.json`,
  `runs/qwen_scope/qtrm_v2_repeat_gate_s50_threshold_summary_severe25.json`,
  and sparse-candidate variants.

Generation distribution:

- 50 prompts total;
- rep2 >= 0.15: 18/50;
- rep2 >= 0.25: 11/50;
- category with highest average rep2: `math_reasoning` at `0.238`;
- evidence-check includes severe claim/evidence copy loops;
- several prompts have low n-gram repetition but still fail to stop before the
  64-token cap.

Detector result:

- original overlap candidates `12:847`, `23:29838,31860` are not clean:
  rep2 >= 0.15 F1 `0.652`, severe rep2 >= 0.25 F1 `0.625`;
- sparse in-sample candidates can fit the 50-sample severe split perfectly, but
  this is not evidence of generalization;
- crude train/holdout check using indices `0-24` for discovery and `25-49` for
  holdout failed: train F1 `1.0`, holdout F1 `0.0`.

Decision: Qwen-Scope should remain diagnostic tooling. Do not build a hard
Qwen-Scope decoding governor yet. The next implementation target should be a
QTRM-generated output-failure dataset plus an explicit stop/format/repetition
verifier head or candidate reranker, with Qwen-Scope features used only for
analysis.

## [2026-04-30] decision | Qwen3.6 teacher distillation order

Added `decisions/qwen36-online-distillation-roadmap.md` to fix the execution
order for DGX-backed teacher distillation.

Decision:

- use DGX as the Qwen3.6-27B teacher/data machine;
- distill QTRM first because the current blocker is generation-side evidence
  use and repetition, not only memory retrieval;
- validate MSA separately as a donor fork with routing/healing losses;
- combine QTRM+MSA only after each path passes its own smoke gates.

Checklist order:

1. stabilize custom full-MSA checkpoint save/load;
2. define Qwen3.6 teacher data schema;
3. configure DGX teacher environment under `/mnt/data4tb`;
4. generate 100-case offline teacher smoke data;
5. train/evaluate QTRM offline distillation;
6. add online top-k teacher KL;
7. add on-policy QTRM correction;
8. add MSA routing-label loss and real 2B healing dry run;
9. run joint QTRM+MSA ablations.

Claim boundary: online distillation is not accepted by lower loss alone. It
must improve held-out generation and show causal use through workspace/core/MSA
ablation.

Implemented the first two execution gates after documenting the plan:

- custom full-MSA checkpoint save/load roundtrip in
  `src/qtrm_mm/qwen35_full_msa.py`;
- Qwen3.6 teacher record schema in
  `src/qtrm_mm/distill/teacher_schema.py`;
- OpenAI-compatible teacher message builder and JSON response parser in
  `src/qtrm_mm/distill/qwen36_teacher_client.py`.

The teacher schema now validates the shared QTRM/MSA distillation record:
`prompt`, `answer`, optional evidence ids/spans, rejected answer, trace summary,
MSA memory docs, target doc ids, and optional top-k teacher logprobs.

Added the first public HF dataset intake path so offline data does not need to
be generated from scratch:

- manifest: `configs/hf_distill_datasets.yaml`;
- source page: `sources/hf-distillation-datasets.md`;
- converter: `src/qtrm_mm/distill/hf_dataset_convert.py`;
- CLI: `scripts/131_convert_hf_distill_dataset.py`;
- first-wave runner: `scripts/132_convert_first_wave_hf_distill_smoke.sh`.

First-wave sources:

- `Yana/ft-llm-2026-reasoning-dpo` for QTRM preference/CoT warmup;
- `AMAImedia/NOESIS-50K-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54`
  for multilingual reasoning warmup;
- `F4biian/RAGognize` for MSA routing/evidence gates;
- `lrsbrgrn/HalluClaim-76k` for hallucination/evidence bottleneck training.

HF smoke note: `datasets` streaming successfully wrote Yana samples but exited
with code 134 during cleanup. The converter now defaults to non-streaming mode;
use `--streaming` only for datasets that cannot be loaded normally.

Actual smoke results:

- Yana non-streaming converted 3/3 rows successfully, but the first rows have
  identical chosen/rejected final answers, so the converter now drops identical
  `rejected_answer` fields.
- RAGognize converted 3/3 rows and produced `memory_docs`; valid responses can
  produce `target_doc_ids`, hallucinated/unsupported responses may have no
  routing target and should be used for evidence-gate negatives.
- HalluClaim converted 3/3 rows after mapping actual columns `doc` and `type`.
- NOESIS converted 3 rows after skipping 3 invalid prompt-only or unclosed
  `<think>` rows; unclosed think rows are skipped to avoid visible reasoning
  leakage.

Built the first mixed QTRM HF warmup JSONL:

- script: `scripts/133_build_hf_distill_training_mix.py`;
- output: `data/filtered/hf_distill_smoke/qtrm_hf_first_wave_mix_s400.jsonl`;
- config: `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml`;
- runner: `scripts/134_run_hf_first_wave_warmup.sh`;
- rows: 400 total, 100 each from Yana, NOESIS, RAGognize, and HalluClaim;
- targets: 171 preference rows, 200 workspace-evidence rows, 166 positive
  `target_doc_ids` rows;
- safety conversion: 34 RAGognize rows with memory docs but no evidence target
  are converted to `chosen=NEEDS_SEARCH` and rejected unsupported answers,
  preventing hallucinated answers from becoming SFT targets.

Also fixed mixed-batch collation in `src/qtrm_mm/data/jsonl_dataset.py` so a
single batch can contain SFT, preference, evidence, and non-evidence rows. The
collator now fills missing optional preference/evidence targets with zero
weights instead of depending on the first sample's key set.

Verification:

- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_hf_distill_training_mix.py tests/test_jsonl_dataset_supervised.py tests/test_hf_distill_manifest.py tests/test_hf_distill_converters.py tests/test_hf_distill_convert_script.py tests/test_qwen36_teacher_distill_schema.py tests/test_qwen36_teacher_client.py`
  passed with 26 tests.
- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_losses.py tests/test_training_checkpoint_init.py tests/test_jsonl_dataset_supervised.py tests/test_hf_distill_training_mix.py`
  passed with 48 tests.
- Data-loader smoke on mixed source-boundary rows produced a 6-row batch with
  workspace tokens, preference weights, logical support targets, and logical
  missing targets present.

Ran the 400-step HF first-wave warmup and saved:
`runs/qwen35_2b_4090_hf_first_wave_warmup_s400/last.pt`.

Safety checks:

- final checkpoint size: 775 MB;
- floating tensors checked: 276;
- NaN/Inf tensors: 0.

Diagnosis after sanity eval:

- The original warmup config had
  `evidence_bottleneck_suppress_without_workspace: true` and the evidence
  bottleneck was applied to the whole QTRM residual.
- Therefore ordinary prompts with no external workspace evidence became
  donor-only at final-logit fusion, even though the latent workspace/core still
  ran internally.
- This is correct only for a strict evidence-causality proof gate, not for the
  general QTRM/loop-LM objective.

Fix:

- added `model.evidence_bottleneck_applies_to_residual`;
- default remains `true` to preserve existing evidence-only proof configs;
- `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml` sets it to `false`,
  making the evidence heads verifier-only while keeping the general QTRM
  residual active without external workspace evidence.

Post-fix eval on the same checkpoint:

- output:
  `runs/qwen35_2b_4090_hf_first_wave_warmup_s400/logit_eval_general_residual_sanity.jsonl`;
- ordinary prompts now have nonzero residual telemetry
  (`residual_linf` about `0.84-0.875`, `donor_to_fused_kl > 0`);
- the evidence-style prompt changes donor argmax, proving QTRM residual is no
  longer shut off;
- however the claim/evidence generation repeats the prompt/evidence pattern,
  so the next run must add format/answer-only supervision and rerun with this
  corrected residual boundary.

Verification:

- `PYTHONPATH=src uv run --with pytest --with pyyaml pytest tests/test_model_config.py tests/test_hf_distill_training_mix.py tests/test_jsonl_dataset_supervised.py`
  passed with 32 tests.

## [2026-04-30] implementation | Qwen3.5-2B full-MSA fork scaffold

Started the aggressive full-MSA donor fork path requested for Qwen3.5-2B.
Added:

- `src/qtrm_mm/msa_qwen35.py`;
- `scripts/129_prepare_qwen35_full_msa_fork.py`;
- `tests/test_qwen35_full_msa_fork.py`;
- `decisions/qwen35-full-msa-fork.md`.

The converter reads `references/model_configs/qwen35_2b_base/config.json`,
rewrites all 24 text layers to Hugging Face's allowed `sparse` type with
`qtrm_full_msa_fork=true`, and writes a conversion manifest. The manifest makes
the destructive boundary explicit:
Qwen3.5-2B has 18 `linear_attention` layers and 6 `full_attention` layers. The
6 full-attention layers can seed Qwen3.5-native MSA projections, while the 18
linear-attention layers must be replaced and healed because GatedDeltaNet
conv/recurrent weights do not map cleanly to MSA q/k/v/o plus router weights.

Added the first custom text-forward prototype in
`src/qtrm_mm/qwen35_full_msa.py`. It implements Qwen3.5-native MSA attention
with the gated q projection, q/k RMSNorm, partial/mRoPE position embeddings,
chunk-pooled doc-id routing, and sparse selected-document attention. The new
test `tests/test_qwen35_full_msa_model.py` verifies a tiny random
Qwen3.5-style config runs a `doc_ids` forward pass. This is still not a trained
full-MSA donor: HF model registration, weight-copy, Memory Parallel cache
runtime, and donor-healing training remain next.

Added the first safe-healing smoke path:

- `src/qtrm_mm/qwen35_full_msa_healing.py`;
- `scripts/130_train_qwen35_full_msa_healing.py`;
- `tests/test_qwen35_full_msa_healing.py`.

The tiny smoke trains only MSA attention/router parameters while freezing
copied embeddings, MLPs, norms, and LM head. The loss is next-token LM CE plus
donor KL against the original Qwen3.5-style teacher. This is the right safety
shape before a real 2B healing run because it tests parameter freezing,
teacher-preservation, doc-id routing, optimizer update, and report writing
without risking a large donor checkpoint.

Ran the tiny smoke for 2 steps:

- report: `runs/qwen35_full_msa_healing_tiny_smoke/healing_report.json`;
- trainable/frozen params: `11296 / 20640`;
- updated trainable L1: `15.0766`;
- final `loss=4.9102`, `lm_loss=4.8438`, `donor_kl=0.0664`.

## [2026-04-30] research | agentic closed-loop planner references

Added the closed-loop planning research pack for the next QTRM/MemoryOS stage:

- downloaded 19 PDFs into
  `references/papers/agentic_closed_loop_planner/`;
- cloned official repos:
  `agent-lightning@0b40cb724a0a`, `ragen@20daedc47558`,
  `agentgym@c3b300f0381a`,
  `language-agent-tree-search@853d81614607`,
  `ace@4f679bef3b78`, `dynamic-cheatsheet@5cfe3c37e8e5`,
  `skillrl@299909b2f5e2`, and `aworld-rl@2082e70bcd54`;
- added `sources/agentic-closed-loop-planning.md`;
- added `concepts/agentic-closed-loop-planner.md`;
- updated the long-horizon agent architecture and canonical architecture
  matrix.

Conclusion:
QTRM is not yet an autonomous closed-loop agent. The prior-backed path is
trace-first: build an external replayable `AgentHarness`, log state/action/
observation/verifier transitions, then train controller heads and only later
attempt Agent-Lightning/RAGEN-style turn-level RL. RAGEN-2 and ASTER add a
hard requirement for collapse diagnostics: entropy is insufficient; track
input-dependence, reward variance, repeated action templates, and tool/evidence
interaction density.

## [2026-04-30] process | research-driven architecture debugging skill

Added local agent skill
`/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md`.
Future QTRM architecture failures should follow this loop: stabilize the run,
write a failure ledger, search primary prior work and official implementations,
map the failure to a research axis, implement the smallest falsifiable
architecture/training experiment, and require generation plus ablation gates
before accepting the change.

This process was added after the bounded-preference run showed the same pattern
as earlier experiments: pairwise or loss improvements can coexist with visible
generation failures, repeated formats, retrieval-but-wrong answers, and weak
workspace/core causality.

Updated the skill so it is active architecture proposal guidance, not just a
debugging checklist. Once a limitation is clear, the agent must propose 2-3
concrete prior-backed architecture candidates, rank them, choose the most
testable one, and proceed to a minimal experiment unless explicitly paused.

Installed external research-support skills:

- `arxiv-research`: arXiv paper fetch/compile for LLM, agents, RAG, reasoning,
  and AI infrastructure.
- `literature-review`: systematic paper search, triage, and synthesis workflow.
- `deep-research`: multi-source citation-tracked research reports with evidence
  persistence and claim verification.
- `research-lookup`: current scientific/research lookup with backend routing.

Validated all five research-related skills, including the local
`research-driven-architecture-debugging` orchestrator. Adjusted two installed
skill frontmatters so they pass the local skill validator.

## [2026-04-30] diagnosis | preference loss is not enough for causal workspace use

Ran the workspace-evidence preference and preference+repeatguard probes after
the 500-step preference-only run still repeated `Ilya Chen Chen...`.

The preference+repeatguard checkpoint learned the small pairwise objective:
`workspace_evidence_preference_repeatguard_pair_eval_s050.jsonl` reported
11/11 preference accuracy and margin pass. That did not transfer to answer
generation. A 4-case quick workspace-evidence ablation wrote
`docs/wiki/decisions/workspace-evidence-preference-repeatguard-quick-ablation.md`
and showed the critical failure mode:

- retrieval is not the bottleneck in this gate (`retrieved_target_rate=1.0`,
  `target_recall_mean=0.9167`);
- `qtrm_residual_with_evidence` scored 0/4;
- workspace/core/core-context/workspace-gate/workspace-memory off modes produced
  identical completions to the full residual path (`same_completion_rate=1.0`);
- outputs still contain action/text/number repetition.

Interpretation: the current model can overfit preference pairs, but the latent
workspace path is not yet behaviorally causal in generation. The next probe
therefore moves back to a donor-preserving residual architecture instead of
training a freer residual head: bounded residual clamp, normalized residual
gate with floor, student-LM pressure, donor-KL preservation, conservative repeat
unlikelihood, and pairwise preference. Added
`configs/qwen35_2b_4090_workspace_evidence_bounded_preference_s050.yaml` and
`scripts/123_run_workspace_evidence_bounded_preference_train.sh`. This starts
from `runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt`, not the
repetition-damaged preference checkpoint, and saves 100-step checkpoints so the
best non-repeating midpoint can be selected if the final checkpoint regresses.

## [2026-04-30] docs | QTRM terminology and reasoning boundary

Added `concepts/qtrm-terminology.md` as the canonical glossary for donor,
donor-backed residual adapter, cognitive sidecar, LatentWorkspace,
workspace-only evidence, gated core context injection, donor annealing,
internalized context engineering, actual reasoning, and inference cliffs.

Updated the goal/scope and forward-pass pages to state the reasoning boundary:
QTRM is intended to learn a latent state-update process, not just a visible
CoT text pattern, but the claim is valid only when held-out evals and
workspace/core/context ablations prove the latent computation is causally
necessary.

Added `decisions/actual-reasoning-architecture-roadmap.md` to separate the
current proof gate from future improvements: true TRM carry, per-sequence ACT,
latent distillation, state-trajectory consistency, residual usefulness gating,
retrieval feedback, and donor annealing.

Added eval telemetry for the next proof gate: `scripts/95_eval_memory_retrieval.py`
now records `latent_gates.workspace_update_gate_*` and
`latent_gates.core_context_gate_*`, and the strict causality runner includes
`qtrm_core_context_off_with_evidence`. Updated `docs/REFERENCE_BASELINE.md`
with the rule that core-context claims require both the off-ablation and gate
telemetry.

## [2026-04-29] diagnosis | donor annealing needs student-only LM loss

Ran the first donor-anneal probe and confirmed a failure mode: training CE on
fused logits lets donor logits carry fluency while QTRM-only logits remain
weak. The fused-loss-only checkpoint was coherent at donor scale `1.0` and
`0.5`, but collapsed at `0.25` and `0.0`.

Added `qtrm_logits` to `QTRMMultimodalModel` outputs before donor fusion, plus
`TrainConfig.loss_student_lm_weight` and a `student_lm` metric. Donor KL now
distills into QTRM-only logits when available. A 200-step patched probe exposed
the real gap: `lm=2.22` while `student_lm=12.46` at start; by the end
`student_lm` only reached `11.42`, and donor `0.0` still repeated. Added
`configs/qwen35_2b_4090_student_lm_pretrain_probe.yaml` for the next gate:
keep donor logits fixed while pretraining QTRM-only LM before attempting full
donor detach.

Ran the 500-step fixed-donor student pretrain probe. `student_lm` moved from
`12.44` to the 8-ish range (`9.89` at step 200, `8.58` at step 300, `7.85` at
step 400, noisy `8.00` at step 450), confirming the training signal works.
Generation still collapses with donor `0.0`, and QTRM residual scale `0.5`
damages even donor-backed output. With `qtrm_logits_scale=0.1`, donor scales
`1.0`, `0.5`, and `0.25` stayed fluent on the Korean quantum-computing prompt.
Conclusion: keep donor attached, train student longer, and cap/gate residual
amplitude before lowering donor logits further.

## [2026-04-29] implementation | bounded residual gate

Added bounded residual fusion behind config flags:
`qtrm_residual_clamp`, `qtrm_residual_gate_enabled`, and
`qtrm_residual_gate_init_bias`. The model now exposes `qtrm_residual_logits`
and `qtrm_residual_gate` for telemetry. The eval script records
`residual_gate`, and `scripts/90_infer_with_donor.sh` can sweep the bounded
residual settings through `QTRM_RESIDUAL_CLAMP`, `QTRM_RESIDUAL_GATE`, and
`QTRM_RESIDUAL_GATE_BIAS`.

Smoke result on the 500-step student LM pretrain checkpoint: the previously
bad setting `donor_logits_scale=0.5`, `qtrm_logits_scale=0.5` collapsed into
number repetition without bounds, but became fluent with clamp `1.0` and the
gate enabled. The same bounded setting stayed fluent at donor scale `0.25`.

Ran `configs/qwen35_2b_4090_bounded_residual_probe.yaml` for 500 steps. The
QTRM-only student LM signal improved from `12.42` to `10.98`, but the learned
gate saturated closed at about `3.3e-6`. Root cause: the gate linear layer was
fed unnormalized latent states, so a small learned weight still produced a
large negative pre-sigmoid value. Clamp-only evaluation proved the opposite
failure mode: full residual strength preserved some prompts but damaged low
donor-scale generation.

Patched the gate to RMS-normalize its latent input and added
`qtrm_residual_gate_min`. With a `0.05` floor and `qtrm_logits_scale=0.5`, the
corrected 500-step checkpoint reached `student_lm=10.98` and keeps donor scales
`1.0`, `0.5`, and `0.25` fluent on Korean and English smoke prompts. The final
sweep had gate mean about `0.061`, residual L-infinity about `0.0625`, no donor
argmax shift, and repeated 2/3-gram rates at `0.0`. Donor `0.0` still
collapses to `,, and`, so this is not yet donor replacement.

## [2026-04-29] implementation | donor annealing references and KL hook

Downloaded donor-annealing/distillation papers under
`references/papers/donor_annealing`: foundational KD, sequence-level KD,
Annealing-KD, Pro-KD, Distilling step-by-step, MiniLLM, GKD/on-policy
distillation, Universal Logit Distillation, and Multi-Level OT.

Cloned implementation references under `references/official`: TRL
GKD/MiniLLM, Microsoft LMOps MiniLLM, Google distilling-step-by-step,
EasyDistill, MiniPLM, Multi-Level OT, and Teacher Assistant KD. Added
`docs/wiki/sources/donor-annealing-distillation.md` and linked it from the
donor annealing roadmap.

Added the first implementation hook beyond static scaling:
`loss_donor_kl_weight`, `donor_kl_beta`, and `donor_kl_temperature`. The loss
uses generalized donor-logit distillation so the QTRM/fused policy can be
trained against Qwen donor logits while `donor_logits_scale` is annealed down.

## [2026-04-29] probe | core halt training gate

Added `configs/qwen35_2b_4090_core_halt_probe.yaml` and
`scripts/107_run_core_halt_probe.sh`. This is the first executable gate for
the latent-loop early-exit idea: train with `core_halt_auto_targets=true`,
`loss_core_halt_weight>0`, `outer_steps=3`, and Qwen donor logits preserved as
the base language policy.

Extended `scripts/92_eval_qtrm_logits.py` with `--enable-core-halt` and
`core_halt` JSON/text telemetry. The post-eval artifacts now expose
`core_steps`, `core_halted`, and halt-head logit summaries, so we can check
whether a trained checkpoint actually exits early instead of only having a
halt loss in code.

Ran the 300-step probe from the memory-synth checkpoint. The training path is
healthy, but the first automatic target rule is too conservative for early
exit: `core_halt` loss collapsed to near zero, and post-eval on 16 samples
reported `core_steps={3:16}` and `core_halted={False:16}`. This means the next
gate needs balanced positive halt labels, teacher-depth labels, or a target
availability report before expecting runtime savings.

Added halt target availability diagnostics to `qtrm_smoke_loss`. Auto-target
runs now report `halt_target_pos_rate`, `halt_target_neg_rate`,
`exact_next_token_pass_rate`, and donor KL gate stats. A one-batch check on the
core-halt probe checkpoint reported `halt_target_pos_rate=0.0`,
`exact_next_token_pass_rate=0.0`, and `donor_kl_pass_rate=1.0`, confirming that
the exact next-token gate, not the KL gate, is blocking early-exit positives.

## [2026-04-29] implementation | CoT-to-latent halting hook

Added `core_halt_loss` and `TrainConfig.loss_core_halt_weight` so the QTRM
halt head can be trained when a future dataset supplies `core_halt_targets`.
The loss is wired into `qtrm_smoke_loss` without changing default training
behavior because the weight defaults to `0.0`.

Added `infer_core_halt_targets` plus `core_halt_auto_targets` and
`core_halt_donor_kl_threshold`. The first automatic target rule is conservative:
exact next-token correctness, optional verifier pass/fail, and optional
fused-vs-donor KL stability must all pass before teaching the core to halt.

Downloaded and indexed the CoT-to-latent reference set: Coconut, CODI,
HybridCoT, and Do Latent Tokens Think. Cloned CODI at `2c23146`. Added source
and concept wiki pages for using explicit CoT as teacher supervision while QTRM
performs repeated latent workspace computation.

Added the first TRM-style halting hook to `QTRMRecursiveCore`: optional
`core_halt_enabled`, `core_halt_min_steps`, `core_halt_use_continue`,
`halt_head`, `enable_core_halt`, and telemetry outputs. This is not full TRM
ACT yet; persistent carry, per-sequence reset, exploration, and halt loss remain
future work.

Extended the MemoryOS answer-level eval with `qtrm_workspace_off_*` and
`qtrm_core_off_*` modes plus `scripts/106_run_ablation_proof.sh`, because
latent-reasoning claims require answer-level component ablations, not only
next-token telemetry.

## [2026-04-29] decision | innovation claim boundary

Clarified the innovation claim. QTRM should be framed as a frozen-donor,
trainable cognitive-sidecar architecture for evidence-sensitive residual
correction, not as a tiny standalone model that generally beats trillion-scale
LLMs. The comparison target is donor-only Qwen, ordinary RAG, LoRA/SFT adapters,
prompt/tool agents, and direct long-context scaling.

## [2026-04-29] docs | limitation-mitigation diagram prompt

Updated the paper-diagram prompt bank so limitation fixes are explicitly
visible inside generated figures. The main architecture prompt now requires a
`Limitations -> Mitigations` panel, and the new overlay prompt maps donor
dependency, residual fluency risk, unproven latent reasoning, misleading
retrieval, and long-context cost to concrete mitigation modules and eval gates.

## [2026-04-29] implementation | ablation modes and paper prompts

Added fixed eval ablation modes for `donor_only`, `residual`, `workspace_off`,
and `core_off`. The same checkpoint can now isolate donor policy, full residual
path, latent workspace contribution, and recursive-core contribution. Added a
paper-diagram prompt bank for QTRM limitation mitigation, ablation lanes, and
the realistic claim boundary for a tiny cognitive core.

## [2026-04-29] implementation | residual telemetry

Added residual-logit telemetry as the first mitigation-roadmap implementation
step. `residual_logit_telemetry` compares scaled donor logits with fused logits
and reports argmax changes, KL divergence, and residual norms. `92_eval_qtrm_logits.py`
now includes this in JSON/text eval output when donor logits are available.

## [2026-04-29] decision | limitations and mitigation roadmap

Added `decisions/limitations-mitigation-roadmap.md` to keep the current QTRM
limits tied to concrete mitigations: Proxy-Tuning/DExperts for logit residuals,
Side-Tuning/Ladder Side-Tuning for the frozen-donor sidecar pattern,
KL/distillation for donor preservation, looped-transformer/Parcae references for
recursive stability, and Self-RAG/CRAG/RAPTOR-style gates for MemoryOS. The next
engineering order is telemetry, ablations, gated residual, and KL-to-donor loss
before longer training.

## [2026-04-29] plan | current architecture pretrain viability probe

Added `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` and
`scripts/105_run_current_arch_pretrain_probe.sh`. This locks the next
architecture gate before more reasoning-data expansion: Qwen donor logits remain
the base language policy, QTRM contributes only a bounded residual
(`qtrm_logits_scale=0.10`), workspace is enabled, and JEPA/aux losses stay off.

The probe trains 2000 steps on `data/filtered/qtrm_clean_pilot.jsonl`, runs live
prompt diagnostics, and writes `post_eval.jsonl` plus `post_eval_prompts.txt`
under `runs/qwen35_2b_4090_current_arch_pretrain_probe`.

## [2026-04-29] implementation | Bongakgyo critical synthesis case expansion

Added `src/qtrm_mm/training/bongak_critical_synthesis_cases.py` and
`scripts/104_build_bongak_critical_synthesis_cases.py`. Built 30
`bongak_critical_synthesis` cases from the local
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_요약.md`
and
`/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_매뉴얼.md`
sources.

Outputs:
`data/filtered/critical_synthesis_bongak_cases.jsonl` and
`data/filtered/critical_synthesis_bongak_traces.jsonl`, 30 rows each. These
rows intentionally treat the local doctrine as source material rather than
truth labels: the target answers must critique weak or controlling claims,
preserve useful practice values, bracket unverifiable metaphysics, and end with
a positive constructive conclusion.

## [2026-04-29] implementation | critical synthesis trace builder

Added `src/qtrm_mm/training/critical_synthesis_data.py` and
`scripts/103_build_critical_synthesis_traces.py`. Built
`data/filtered/critical_synthesis_traces.jsonl` from
`data/eval/critical_synthesis_probe.jsonl`. The generated rows preserve the
required structure: critique, preserve, risks, reframe, and positive
conclusion.

## [2026-04-29] implementation | fact verification and critical synthesis gates

Added a deterministic fact-verification gate:
`src/qtrm_mm/eval/fact_verification.py`,
`src/qtrm_mm/training/fact_verification_data.py`,
`scripts/102_eval_fact_verification_memoryos.py`, and
`data/eval/fact_verification_probe.jsonl`. The gate separates verdict accuracy,
action accuracy, retrieval recall, temporal priority, authority priority, and
conflict handling before generation.

Added the critical-synthesis axis for religion/value questions:
`src/qtrm_mm/eval/critical_synthesis.py` and
`data/eval/critical_synthesis_probe.jsonl`. The target is not blind skepticism;
it requires critique, preservation of value, risk checks, reframing, and a
constructive positive conclusion. Added wiki pages for value/critical synthesis
and the 본각교 handling contract.

## [2026-04-29] ingest | fact verification and fake-info references

Downloaded fact-verification and fake-info papers under
`references/papers/fact_verification`: FEVER, FActScore, LongFact/SAFE,
RAGTruth, OpenFActScore, CIBER, CONFACT, CARE-RAG, verifiable misinformation
agent, LiveFact, and ArbGraph. Cloned official repositories for FEVER baseline,
FActScore, LongFact/SAFE, RAGTruth, OpenFActScore, CONFACT, and ArbGraph under
`references/official`.

Added wiki pages for fact-verification reasoning. The design now separates
predictive latent intuition from factual truth: LeWorldModel remains the
world-model prior, while MemoryOS needs retrieval, source metadata, atomic claim
verification, conflict arbitration, temporal labels, and `NEEDS_SEARCH` routing.

## [2026-04-29] implementation | self-improvement preference rows

Downloaded current self-improvement/hallucination references under
`references/papers/self_improvement` and added wiki pages for the self-
improvement loop. The design now treats `UNKNOWN` as a closed-evidence eval
label, not the final open-world MemoryOS answer. In agentic mode, insufficient
evidence routes to `Action: NEEDS_SEARCH`, then retrieval/search expands the
evidence set before answering or reporting bounded non-verification.

Added `src/qtrm_mm/training/self_improvement_data.py` and
`scripts/101_build_self_improvement_preferences.py`. Built
`data/filtered/memory_self_improvement_preferences_analysis.jsonl` from the
held-out MemoryOS eval: 11 analysis-only rows, with 6 `needs_search` states and
5 answer-correction states. These rows are explicitly `analysis_only` to avoid
held-out train leakage.

## [2026-04-29] eval | held-out MemoryOS generalization

Ran the new held-out MemoryOS reasoning gate after synthetic trace expansion.
The checkpoint `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
evaluated on `data/eval/memory_reasoning_heldout_probe.jsonl` with Harrier
top-30, Qwen3-Reranker-0.6B, and top-5 evidence.

Result: donor-only stayed at 6/12, while QTRM residual improved to 9/12.
QTRM residual solved abstention 4/4 but missed two Korean conflict cases and
one English multi-hop case. This confirms the earlier 9-case trace fine-tune
was overfit, while the broader synthetic trace set generalizes partially.

Next gate: fix Korean temporal/authority conflict selection, the remaining
multi-hop miss, and UNKNOWN repetition artifacts before scaling MemoryOS.

## [2026-04-29] synthesis | long-horizon agent references

Added `sources/long-horizon-agent-references.md` and
`concepts/long-horizon-agent-architecture.md`. The wiki now records the current
agentic/long-context reference bundle: Externalization in LLM Agents, Memory for
Autonomous LLM Agents, MemGPT, Memex(RL), Recursive Language Models, ReAct,
Reflexion, Voyager, SWE-agent, Self-RAG, CRAG, and RAPTOR.

Decision captured: QTRM should not be redesigned as a standalone long-running
agent. Long-running behavior belongs in MemoryOS plus an agent harness with
mode routing, indexed evidence, trace memory, reflection memory, skill memory,
sandboxed execution, budgets, and verification gates. RLM is a future inference
mode, not the default training target.

## [2026-04-29] ingest | Titans neural memory

Added `references/official/titans-pytorch` at commit `714a14c` and downloaded
`references/papers/long_term_memory/titans_2501.00663.pdf`. Added wiki pages
for Titans and neural long-term memory. QTRM now treats Titans as a future
long-term/test-time memory ablation, separate from the current in-context
`LatentWorkspace` and donor-logit residual path.

Clarified architecture wording: QTRM performs looped latent-workspace
computation over donor representations, but it should be described as a
Qwen-backed looped latent-workspace residual adapter rather than a standalone
loop LM or a proven latent-reasoning LM.

## [2026-04-29] experiment | residual 0.10 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s010_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`. Live diagnostics at
200/400/600/800/1000 steps remained stable, and independent reload evaluation
passed the same language-stability gate as residual `0.05`.

Metrics: Korean quantum prompt `loss=2.1777`, `ppl=8.83`,
`rank_mean=6.62`, `rep2=0.000`; English entanglement prompt `loss=2.8766`,
`ppl=17.75`, `rank_mean=15.40`, `rep2=0.032`; math prompt `loss=2.2573`,
`ppl=9.56`, `rank_mean=53.67`, `rep2=0.115`. The math prompt generated the
correct answer and stopped without continuing into another algebra example.

Decision: `qtrm_logits_scale=0.10` is inside the current safe residual range
and becomes the next working candidate. Do not scale residual strength further
until donor-only versus QTRM-residual memory/retrieval evals exist.

## [2026-04-29] experiment | residual 0.05 stability gate

Finished `configs/qwen35_2b_4090_donor_residual_s005_1000.yaml` and saved
`runs/qwen35_2b_4090_donor_residual_s005_1000/last.pt`. Independent reload
evaluation passed the language-stability gate: Korean and English generations
remained coherent with no `Freeze`, no `world of the world`, and no single-token
collapse. Metrics: Korean quantum prompt `loss=2.2096`, `ppl=9.11`,
`rank_mean=6.88`, `rep2=0.000`; English entanglement prompt `loss=2.9217`,
`ppl=18.57`, `rank_mean=16.40`, `rep2=0.048`. The math prompt answered `x=4`
then continued into another synthetic algebra example; this is template
continuation rather than the earlier collapse mode, but future math evals need
answer-only stop criteria.

Verification: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`
ran 36 tests successfully, and `py_compile` passed for the modified model,
training, donor, loss, and eval files.

## [2026-04-29] architecture | QTRM forward pass diagrams

Added `docs/wiki/architecture/qtrm-forward-pass.md` with Mermaid flowcharts,
generation sequence, tensor shape ledger, and current donor-logit residual
contract. The page explicitly separates latent-workspace computation from
claims of independent latent reasoning or standalone QTRM language generation.

## [2026-04-29] synthesis | Karpathy cognitive core and QTRM goal

Added corrected wiki notes for the Karpathy/Dwarkesh cognitive-core claim. The
wiki now distinguishes the supported hypothesis from the social-media overclaim
that a clean 1B model directly equals a 1.8T frontier model. Added QTRM goal and
scope decision: prioritize donor baseline, data quality, tiny-overfit,
target-token-rank, entropy/repetition diagnostics, memory traces, and ablations
before another long training run.

## [2026-04-29] implementation | QTRM training diagnostics

Added diagnostic utilities and scripts:
`scripts/91_donor_only_generate.sh`, `scripts/92_eval_qtrm_logits.py`,
`scripts/92_eval_qtrm_logits.sh`, and
`scripts/93_tiny_overfit_donor_adapter.sh`. The first checkpoint probe on
`runs/qwen35_2b_4090_extended/last.pt` shows `Freeze` as the top next token and
greedy completion collapse with donor on and off. The checkpoint also predates
the current LeWM/SIGReg JEPA module layout, causing 45 missing new JEPA keys and
4 unexpected old `jepa.net.*` keys during non-strict loading.

Follow-up diagnostics: Qwen donor-only generation is coherent on Korean prompts.
Current-code donor-on tiny-overfit reaches `rank=1.0` and `top1=1.0`, and a
saved tiny checkpoint continues `Quantum entanglement means` without `Freeze`.
Found and fixed a real data pipeline bug: HF-tokenized JSONL samples now carry
an `attention_mask` based on the tokenizer pad id instead of assuming pad id
`0` in `collate_jsonl`.

Ran `configs/qwen35_2b_4090_fixed_pilot.yaml` for 120 real-data steps after
the mask fix. The pilot loss moved from roughly `lm=12.51` to `lm=9.19`.
Evaluation on the Korean quantum prompt no longer selected `Freeze`; the top
tokens were high-frequency punctuation/function words and greedy decoding
formed `, and the.` loops. Interpretation: current code and fixed masking avoid
the old `Freeze` attractor in this short pilot, but the run is far below a
usable LM loss and still needs clean data, longer training, and live
entropy/repetition gates.

## [2026-04-29] ingest | OpenMythos and recurrent depth

Added `references/official/openmythos` at `8c68c1f` and
`references/official/parcae` at `dee8363`. Downloaded recurrent-depth papers:
Parcae stable looped LMs, Looped Transformers are Better at Learning Learning
Algorithms, Reasoning with Latent Thoughts, and a negative/limited latent-CoT
probing paper. Updated recursive-core docs with stable injection, depth-sweep,
and recurrence telemetry requirements.

## [2026-04-29] ingest | Architecture search and composition

Added architecture-search papers under `references/papers/architecture_search`:
Transformer modification transfer, NAS survey, RegNet/design spaces,
EfficientNet, ConvNeXt, task arithmetic, NAS search-phase evaluation, and No
Free Lunch. Added wiki pages for treating QTRM as a compositional architecture
that must be validated through ablations rather than assumed optimal.

## [2026-04-29] ingest | Training diagnostics

Added training-diagnostics papers under
`references/papers/training_diagnostics`: LLM scaling laws, Chinchilla
compute-optimal training, gradient noise scale, LC-PFN learning-curve
extrapolation, ACL 2025 LLM training dynamics, and Prechelt early stopping.
Added wiki pages for using these papers as QTRM run-failure probes.

## [2026-04-29] ingest | LeWorldModel

Added `references/official/le-wm` and `references/papers/leworldmodel_2603.19312.pdf`.
Changed the JEPA target from older stop-grad/EMA style to LeWM-style
end-to-end next-embedding prediction plus SIGReg.

## [2026-04-29] ingest | Tiny Recursive Models

Added `references/official/tiny-recursive-models` and
`references/papers/tiny_recursive_models_2510.04871.pdf`. Marked current QTRM
recursive core as needing comparison against TRM carry, no-grad cycles,
carry-detach, and ACT halting.

## [2026-04-29] ingest | Gated DeltaNet

Added `references/official/gated-delta-net` and
`references/papers/gated_delta_networks_2412.06464.pdf`. Marked current
`TorchGatedDeltaMixer` as a smoke/debug fallback, not official Gated DeltaNet.

## [2026-04-29] implementation | Gated DeltaNet adapter

Updated `src/qtrm_mm/mixers.py` to prefer the official FLA import path
`from fla.layers import GatedDeltaNet`. Added adapter tests for strict mode,
non-strict fallback, mask forwarding, and backend registry detection.
Updated the 4B adapter config and production backend docs to prefer
`delta_backend: fla_gated_delta`.

## [2026-04-29] ingest | Karpathy LLM Wiki

Added wiki schema under `docs/wiki`. Adopted the raw-source / synthesized-wiki /
schema split for QTRM architecture work.

## [2026-04-29] ingest | Qwen3.5 donor architecture

Downloaded Qwen3.5 2B Base and chat model cards/configs into
`references/model_configs`. Added Qwen3.5 Omni technical report
`references/papers/qwen35_omni_technical_report_2604.15804.pdf`. Created wiki
pages for Qwen3.5 architecture and donor integration constraints.

## [2026-04-29] ingest | Qwen donor lineage

Added `references/official/qwen35`,
`references/papers/qwen3_vl_technical_report_2511.21631.pdf`, and
`references/papers/qwen3_omni_technical_report_2509.17765.pdf`. These cover
Qwen3.5 release guidance plus Qwen3-VL/Qwen3-Omni design lineage.

## [2026-04-29] ingest | Transfer, merge, healing

Added model merging and healing-tune references under
`references/official/model-merging` and `references/papers/model_merging`.
Covered MergeKit, TIES, DARE, DELLA, Model Soups, Branch-Train-Merge,
continual/domain-adaptive pretraining, model-merging surveys, and
merge-friendly fine-tuning. Added QTRM wiki pages for the transfer/healing axis.

## [2026-04-29] diagnostics | QTRM collapse probes

Added donor-only, logit, and tiny-overfit diagnostics. The donor-only Qwen
baseline generates coherent text, and a 16-sample donor-on tiny-overfit run
reaches `rank=1.0`/`top1=1.0`, so the current path can learn. The old extended
checkpoint is invalid for current architecture comparisons because it has JEPA
state mismatch and collapses to `Freeze`.

Fixed HF padding-mask handling in `jsonl_dataset`: padding is now derived from
the tokenizer pad id instead of assuming `0`. Added a clean text-only pilot data
builder and built `data/filtered/qtrm_clean_pilot.jsonl` with 6000 accepted
rows. The 500-step clean pilot reached roughly `lm=6.86` from an initial
`lm=12.52`. It removed the `Freeze` top-token failure but still free-runs into
dialogue markers and repeated high-frequency phrases, so the next work is an
objective/data/architecture ablation rather than another blind long run.

Also fixed autoregressive donor handling in `scripts/90_infer_with_donor.sh`
and training diagnostics: donor states are refreshed against the full generated
sequence instead of being held at the initial prompt length. Re-running the clean
checkpoint with refreshed donor states still repeats, confirming this was a
real consistency bug but not the remaining collapse's primary cause.

Added configurable train loss weights and ran
`configs/qwen35_2b_4090_clean_lm_only_pilot.yaml` with `loss_jepa_weight=0` and
`loss_aux_weight=0`. The 300-step LM-only run reached about `lm=6.73` but still
generated `world of the world` loops, so JEPA/aux should not be treated as the
main collapse source.

## [2026-04-29] ingest | Workspace and memory references

Added official/near-official references for latent workspace and memory:
DeepMind Perceiver, Salesforce LAVIS/BLIP-2 Q-Former, OpenFlamingo
PerceiverResampler, and ARMT. Downloaded related PDFs under
`references/papers/memory_workspace`. The current QTRM workspace is now treated
as an in-context working-memory adapter, not a proven persistent memory system.

## [2026-04-29] implementation | Donor logits and Perceiver-style workspace

Added donor-logit residual generation through `donor_logits_scale` and
`qtrm_logits_scale`. `donor_logits_scale=1.0, qtrm_logits_scale=0.0` removes the
generation collapse and produces Qwen-quality completions, proving the previous
loop was caused by forcing a random QTRM LM head to learn the full language
distribution. Added Perceiver/OpenFlamingo-style workspace depth controls:
`workspace_layers`, `workspace_ff_mult`, and `workspace_include_latents_in_kv`.

Ran `configs/qwen35_2b_4090_donor_residual_workspace_pilot.yaml` for 120 steps.
The run stayed near donor-quality generation on both Korean and English prompts
with no `world of the world` collapse. This establishes the next baseline:
Qwen donor as base policy, QTRM as a small residual workspace/memory adapter.

## [2026-04-29] ingest | In-Place TTT

Added ByteDance Seed `In-Place-TTT` at `references/official/in-place-ttt` and
downloaded `references/papers/test_time_training/in_place_ttt_2604.06169.pdf`.
This is relevant to QTRM as a donor-side adaptive-memory axis: selected Qwen/LLaMA
MLP down-projection fast weights are updated during inference with a
next-token-prediction-aligned objective. It should be evaluated as a separate
ablation after donor-logit residual generation remains stable.

## [2026-04-29] plan | Residual ablation

Added `docs/wiki/decisions/residual-ablation-plan.md` to preserve the immediate
experiment plan. Added 1000-step residual configs for `qtrm_logits_scale=0.05`
and `0.10`. The first run is `0.05` with donor logits as the base policy,
Perceiver-style workspace depth enabled, and JEPA/aux disabled.

## [2026-04-29] eval | Memory retrieval probe

Added `data/eval/memory_retrieval_probe.jsonl`,
`src/qtrm_mm/eval/memory_retrieval.py`, and
`scripts/95_eval_memory_retrieval.py`. The script compares donor-only logits
against QTRM residual logits with and without fixed MemoryOS-style evidence, and
scores only generated completion text so answers present in the prompt are not
counted as hits.

Initial result on
`runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`: donor-only with
evidence `5/5`, QTRM residual with evidence `5/5`, donor-only no evidence `0/5`,
and QTRM residual no evidence `0/5`. Interpretation: evidence injection works,
but the behavior is already provided by Qwen donor logits. QTRM residual `0.10`
does not break the path, but this probe does not yet prove a distinct
QTRM-specific memory policy.

## [2026-04-29] eval | Harrier MemoryOS retrieval

Changed the MemoryOS text embedding default to
`microsoft/harrier-oss-v1-270m` and updated Harrier query encoding to use the
official SentenceTransformers `prompt_name="web_search_query"` path, with a
custom-instruction fallback for other embedders. Added
`scripts/96_build_memory_retrieval_probe_index.py` to build probe indexes that
preserve `case_id`, `evidence_role`, and `is_target` metadata.

Built `runs/eval/memory_retrieval_memoryos_harrier270m_index` from
`data/eval/memory_retrieval_distractor_probe.jsonl` and ran
`scripts/95_eval_memory_retrieval.py --evidence-mode memoryos`. Result:
retrieved target `10/10`, answer hit `10/10` across donor-only and QTRM
residual modes. This verifies the real MemoryOS retrieval path, but donor-only
still matches QTRM residual, so the next task is a memory trace or distractor
task where QTRM must improve over donor evidence copying.

## [2026-04-29] eval | Qwen3 reranked MemoryOS

Added `src/qtrm_mm/memoryos/rerank.py` with `none`, `lexical`, and
`cross_encoder` reranking backends. `cross_encoder` supports
`Qwen/Qwen3-Reranker-0.6B` through SentenceTransformers and caches the model
within the process. Updated `retrieve.py` and
`scripts/95_eval_memory_retrieval.py` to support
`retrieve_top_n -> rerank -> retrieval_top_k` evidence selection.

Smoke-tested Qwen3-Reranker-0.6B on two documents: it ranked the target archive
code document above the vault distractor (`10.75` vs `-4.6875`). Full
MemoryOS eval with Harrier top-20 and Qwen3 top-3 reranking scored retrieval
target `10/10` and answer hit `10/10` across donor-only and QTRM residual
modes. The reranker improves evidence ordering, but this probe still does not
show a QTRM-specific advantage over donor-only generation.

## [2026-04-29] eval | Hard MemoryOS reasoning probe

Added `data/eval/memory_reasoning_probe.jsonl` with temporal conflict,
authority conflict, English/Korean multi-hop, and negative missing-answer
cases. Added target recall/all-target retrieval metrics to the memory eval
helpers and JSONL output.

Built `runs/eval/memory_reasoning_harrier270m_index` and ran Harrier top-20 ->
Qwen3-Reranker-0.6B -> top-5 evidence generation. Result: donor-only `5/6`,
QTRM residual `5/6`, with all target evidence retrieved in both modes. The
single failure is the negative missing-answer case: both modes answered a seen
distractor passphrase instead of `UNKNOWN`. This isolates a real next weakness:
abstention/contradiction handling, not retrieval recall.

## [2026-04-29] plan | 100M MemoryOS and MSA

Cloned the MSA reference implementation to `references/official/msa` at
`30405b2`. Added `docs/wiki/sources/memory-sparse-attention.md` and
`docs/wiki/decisions/memoryos-100m-scale-plan.md`.

Added `src/qtrm_mm/memoryos/scale_plan.py` and
`scripts/97_plan_memoryos_scale.py` to estimate large MemoryOS builds before
ingestion. The default 100M-token estimate with 512-token chunks, 64-token
overlap, and Harrier 270M 640-dimensional embeddings is 223,215 chunks and
about 0.532 GiB of float32 embedding storage. Decision: 100M+ tokens is a
MemoryOS external-memory target, while active model context remains a much
smaller retrieved/reranked/compressed working set.

## [2026-04-29] eval | Abstention metrics before scale

Expanded `data/eval/memory_reasoning_probe.jsonl` from 6 to 9 cases with more
negative missing-answer, authority-conflict, and temporal missing-current-state
checks. Updated `src/qtrm_mm/eval/memory_retrieval.py` and
`scripts/95_eval_memory_retrieval.py` so records carry `category`,
`task_family`, and `expected_unknown`, and summaries report accuracy/retrieval
metrics by mode, category, and task family.

Added `src/qtrm_mm/memoryos/scale_benchmark.py` and
`scripts/98_benchmark_memoryos_scale.py` to write staged 1M/10M planning records
before attempting larger MemoryOS ingestion. Default output is
`runs/eval/memoryos_scale_plan_1m_10m.jsonl`. Current order: fix "retrieved but
answered wrong" on the small hard probe before treating 100M-scale architecture
as the main bottleneck.

Rebuilt `runs/eval/memory_reasoning_harrier270m_index` from the expanded
9-case probe, then reran Harrier top-20 -> Qwen3-Reranker-0.6B -> top-5
generation into `runs/eval/memory_reasoning_qwen3_rerank_32tok_expanded.jsonl`.
Result: donor-only `6/9`, QTRM residual `6/9`, all target evidence retrieved
`18/18`. By task family, conflict `8/8`, multi-hop `4/4`, abstention `0/6`.
This confirms the current blocker is not search recall; it is answer selection
when the correct response is `UNKNOWN`.

## [2026-04-29] train | Memory trace abstention fine-tune

Added `src/qtrm_mm/training/memory_trace_data.py` and
`scripts/99_build_memory_trace_data.py` to convert hard MemoryOS reasoning cases
into supervised traces. Wrote
`data/filtered/memory_abstention_traces.jsonl` with 27 rows: `target`, `all`,
and `lexical` evidence variants for each of the 9 probe cases.

Changed `JsonlTextVisionDataset` and `qtrm_smoke_loss` so rows with
`prompt`/`answer` train only answer tokens via `labels=-100` on prompt tokens.
Added `--init-checkpoint` to `qtrm_mm.training.train` so memory trace runs can
continue from the stable residual checkpoint.

Strict prompting alone did not fix the issue: with the original residual
checkpoint, donor-only and QTRM residual still failed abstention. A first
trace run at `qtrm_logits_scale=0.1` learned some UNKNOWN behavior but was too
weak; scale sweeps showed `0.5` was useful while `1.0` over-abstained.

Final run:

- Config: `configs/qwen35_2b_4090_memory_abstention_trace_s050.yaml`
- Init: `runs/qwen35_2b_4090_donor_residual_s010_1000/last.pt`
- Output: `runs/qwen35_2b_4090_memory_abstention_trace_s050/last.pt`
- Eval: `runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl`

Result on the expanded 9-case hard probe with all target evidence retrieved:
donor-only `5/9`, QTRM residual `9/9`. QTRM residual task-family accuracy:
conflict `4/4`, multi-hop `2/2`, abstention `3/3`. This fixes the current
small-scale "retrieved but answered wrong" blocker, but it is not yet proof of
general MemoryOS reasoning beyond this synthetic probe.

## [2026-04-29] train | Teacher-depth core halt probe

Found and fixed a wiring issue in the early-halt probe: the global
`attn_every=4` combined with `n_coda_layers=2` left the coda with no attention
layer, so the recursive core prefix could fail to affect text logits. Added
`QTRMConfig.coda_attn_every` and set the probe config to `coda_attn_every=2`,
leaving the core's 3:1 delta/attention schedule unchanged.

Added teacher-depth halt targets based on per-depth QTRM residual last-token
logits. The target compares each depth to the final depth with top-1 match,
centered-logit cosine, and KL checks. Donor logits are deliberately excluded
from this comparison because they are depth-invariant and can hide whether the
QTRM core itself has stabilized.

Probe run:

- Config: `configs/qwen35_2b_4090_core_halt_probe.yaml`
- Init: `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
- Output: `runs/qwen35_2b_4090_core_halt_probe/last.pt`
- Eval: `runs/qwen35_2b_4090_core_halt_probe/post_eval_core_halt.jsonl`

Result after 300 steps: post-eval on 8 clean-pilot samples reported
`core_steps={1:8}` and `core_halted={true:8}`; greedy generation over 40
generated steps also used `core_steps={1:40}`. Average teacher-forced loss was
`1.934`, top-1 accuracy `0.527`, and repeated-2gram rate `0.0`. This proves the
halt path can learn and execute early exit with donor-assisted residual
generation. It is not yet proof that step-1 latent reasoning is sufficient on
hard reasoning tasks; the next gate is comparing halted vs full-depth answers
on the MemoryOS hard probe.

## [2026-04-29] eval | Core halt MemoryOS depth gate

Added `--core-halt-mode {config,enabled,disabled}` to
`scripts/95_eval_memory_retrieval.py` and recorded prompt-level `core_halt`
telemetry in each MemoryOS eval record. This lets the same checkpoint be
evaluated in full-depth mode (`enable_core_halt=False`) and early-exit mode
(`enable_core_halt=True`).

Ran the teacher-depth halt checkpoint with `--qtrm-logits-scale 0.5`,
Harrier retrieval, Qwen3-Reranker-0.6B, and top-5 evidence.

9-case hard probe:

- Full depth: `runs/eval/memory_reasoning_corehalt_full_depth_32tok.jsonl`
- Halted: `runs/eval/memory_reasoning_corehalt_enabled_32tok.jsonl`
- Result: full-depth `5/9` with `core_steps={3:9}`; halted `5/9` with
  `core_steps={1:9}`; hit changes `0`.

12-case held-out probe:

- Full depth:
  `runs/eval/memory_reasoning_heldout_corehalt_full_depth_32tok.jsonl`
- Halted:
  `runs/eval/memory_reasoning_heldout_corehalt_enabled_32tok.jsonl`
- Result: full-depth `7/12` with `core_steps={3:12}`; halted `7/12` with
  `core_steps={1:12}`; hit changes `0`.

Interpretation: early exit did not add answer-level regressions in these two
MemoryOS gates, and it reduced recursive outer depth from 3 to 1. However, the
core-halt fine-tune checkpoint is weaker than the earlier synthetic MemoryOS
checkpoint on held-out accuracy (`7/12` here versus prior `9/12`). The next
training pass should preserve MemoryOS behavior while teaching halting, either
by training halt targets on MemoryOS traces or by freezing most residual-path
weights and training the halt head/coda gate more narrowly.

## [2026-04-29] train | MemoryOS-preserving halt-only probe

Added `TrainConfig.trainable_param_policy` and
`configure_trainable_parameters`. The first narrow policy is
`core_halt_only`, which freezes the full QTRM residual path and trains only
`core.halt_head.weight` and `core.halt_head.bias`. This lets halt behavior be
learned without changing full-depth MemoryOS logits.

Run:

- Config: `configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml`
- Script: `scripts/108_run_memory_halt_preserve.sh`
- Init: `runs/qwen35_2b_4090_memory_synth_generalization_s050/last.pt`
- Data: `data/filtered/memory_reasoning_synth_traces.jsonl`
- Trainable tensors: 2
- Trainable params: 1,026
- Output: `runs/qwen35_2b_4090_memory_halt_preserve_s050/last.pt`

The checkpoint kept the synthetic MemoryOS generalization result while adding
early exit:

| Eval | Full-depth hits | Halted hits | Full steps | Halted steps | Hit changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9-case hard probe | 6/9 | 6/9 | 2 x 9 | 1 x 9 | 0 |
| 12-case held-out probe | 9/12 | 9/12 | 2 x 12 | 1 x 12 | 0 |

Outputs:

- `runs/eval/memory_reasoning_halt_preserve_full_depth_32tok.jsonl`
- `runs/eval/memory_reasoning_halt_preserve_enabled_32tok.jsonl`
- `runs/eval/memory_reasoning_heldout_halt_preserve_full_depth_32tok.jsonl`
- `runs/eval/memory_reasoning_heldout_halt_preserve_enabled_32tok.jsonl`

Interpretation: the previous regression was not caused by early halt itself; it
was caused by fine-tuning the residual path on clean LM text. Freezing the
residual path and training only the halt head preserves the 12-case held-out
MemoryOS score (`9/12`) while reducing recursive outer depth from 2 to 1.

## [2026-04-29] train | Bounded student-LM 2K continuation

Added a longer bounded student-LM continuation config:

- Config: `configs/qwen35_2b_4090_bounded_residual_studentlm_2k.yaml`
- Init: `runs/qwen35_2b_4090_bounded_residual_probe/last.pt`
- Output: `runs/qwen35_2b_4090_bounded_residual_studentlm_2k/last.pt`
- Eval: `runs/qwen35_2b_4090_bounded_residual_studentlm_2k/evals/donor_scale_sweep_qtrm1p0_final.jsonl`

The important change from earlier student-LM probes is
`qtrm_logits_scale: 1.0`, while keeping donor logits fixed at `1.0` during
training and retaining normalized residual gating with a `0.05` gate floor.
This made the student-only loss move much faster: the run started around
`student_lm=11.05` and reached the noisy `5.7-6.2` range by the second half of
the 2K-step run.

Final donor-scale sweep:

| donor_logits_scale | Behavior |
| --- | --- |
| `1.0` | fluent Korean/English greedy text; residual gate about `0.063`; no donor argmax shift |
| `0.5` | still fluent; no donor argmax shift |
| `0.25` | still fluent; no donor argmax shift |
| `0.0` | collapsed to `world of the world` / repeated chapter-pattern text |

Interpretation: the residual safety rail now works at higher QTRM scale, but
donor-logit detach is still not solved. Low-donor fluency is still carried by
the donor logits, and QTRM-only generation remains unstable. The next training
step should be on-policy distillation/GKD-style: sample low-donor or QTRM-only
continuations, have the donor/teacher score or correct those continuations, and
train on the actual student distribution instead of only teacher-forced clean
text.

## [2026-04-29] research | Donor-free collapse solution search

Web search for the `donor_logits_scale=0.0` collapse pointed to a clear
training-distribution problem, not just a step-count problem. The most relevant
references are:

- GKD / on-policy distillation: `https://arxiv.org/abs/2306.13649`
- 2026 OPD failure/success recipe: `https://arxiv.org/abs/2604.13016`
- OPD survey: `https://arxiv.org/abs/2604.00626`
- DistiLLM: `https://arxiv.org/abs/2402.03898`
- DistiLLM-2 contrastive distillation:
  `https://arxiv.org/abs/2503.07067`
- Residual KD: `https://openreview.net/forum?id=Dh6KxUxG20`
- Concrete Score Distillation:
  `https://openreview.net/forum?id=bZBJFrxH1H`
- Distillation scaling laws: `https://arxiv.org/abs/2502.08606`
- Capacity-gap law: `https://arxiv.org/abs/2311.07052`
- Minitron / Sheared LLaMA as fallback compression paths:
  `https://github.com/NVlabs/Minitron`,
  `https://arxiv.org/abs/2310.06694`

Decision: do not keep extending teacher-forced student-LM training as the only
path. The next QTRM detach experiment should collect QTRM's own bad low-donor
rollouts, score them with Qwen donor feedback, and train on those
student-visited states. Use teacher/reference continuations as positives and
collapsed QTRM continuations as negatives. Add repeated n-gram unlikelihood as
a local anti-collapse auxiliary loss, but keep the main fix
GKD/OPD/DistiLLM-style because the real failure is exposure bias.

## [2026-04-29] decision | Residual adapter validation reset

Clarified the current architecture name and validation order. The current QTRM
should be treated as a donor-backed residual adapter, more specifically a
donor-backed residual cognitive adapter:

```text
Qwen hidden states -> QTRM workspace/core/coda -> residual logits
Qwen donor logits  -> base language policy
fused logits       -> donor logits + bounded QTRM residual
```

The donor-free probes skipped ahead to a standalone-student claim. That was a
useful stress test, but it is not the next required proof. The immediate proof
is residual-adapter usefulness: with donor logits intact, QTRM residual should
beat donor-only on evidence-sensitive MemoryOS tasks while preserving donor
fluency. Only after that gate should low-donor and donor-free student behavior
be promoted again.

Next action: rerun MemoryOS hard/held-out ablation with donor-only versus QTRM
residual, plus workspace/core disabled modes where applicable.

## [2026-04-29] research | LM2 memory reference added

Added LM2 as a memory-architecture reference:

- Paper: `https://arxiv.org/abs/2502.06049`
- Repo: `https://github.com/convergence-ai/lm2`

Interpretation: LM2 is relevant to QTRM, but not as the direct fix for
`donor_logits_scale=0.0` language collapse. It supports the residual-adapter
memory direction: keep the base Transformer path intact, add a complementary
memory pathway, connect it through cross-attention and gates, and verify that
general ability is not degraded. This maps well to QTRM's near-term goal as a
donor-backed residual cognitive adapter with explicit MemoryOS/workspace
ablation gates.

## [2026-04-29] eval | Residual adapter proof package

Added a fixed proof package for the current near-term claim:

- Script: `scripts/109_build_residual_adapter_proof.py`
- Module: `src/qtrm_mm/eval/residual_adapter_proof.py`
- Markdown: `docs/wiki/decisions/residual-adapter-proof.md`
- JSON: `docs/wiki/decisions/residual-adapter-proof-summary.json`

The package summarizes existing eval JSONL files instead of rerunning training:

| Eval | Donor-only | QTRM residual | Delta |
| --- | ---: | ---: | ---: |
| hard memory probe | 5/9 | 9/9 | +4 |
| held-out memory probe | 6/12 | 9/12 | +3 |
| aggregate | 11/21 | 18/21 | +7 |

The aggregate task-family delta shows the current gain is concentrated in
abstention: donor-only `0/7`, QTRM residual `7/7`. Conflict and multi-hop are
currently tied. This supports the residual-adapter usefulness claim on current
MemoryOS probes, not a donor-free standalone-LM claim.

## [2026-04-30] eval | Expanded residual adapter gate

Added and ran the 72-case expanded held-out MemoryOS gate:

- Cases: `data/eval/memory_reasoning_heldout_expanded_72.jsonl`
- Builder: `scripts/110_build_expanded_memory_reasoning_heldout.py`
- Runner: `scripts/111_run_residual_adapter_expanded_gate.sh`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl`

The expanded gate is balanced: 24 conflict, 24 multi-hop, and 24 abstention
cases. It is ID-disjoint from the hard probe, original held-out probe, and
synthetic training cases. The actual MemoryOS retrieval/rerank path retrieved a
target for every case, with all-target retrieval on 58/72 cases per mode.

Result:

| Eval | Donor-only | QTRM residual | Delta |
| --- | ---: | ---: | ---: |
| expanded held-out memory probe | 26/72 | 49/72 | +23 |

Updated the residual-adapter proof package to include this gate. Aggregate proof
is now donor-only `37/93`, QTRM residual `67/93`, delta `+30`. The gain is still
largest on abstention (`1/31 -> 25/31`), but the expanded gate also shows
nonzero multi-hop improvement (`11/30 -> 16/30`). The remaining weakness is
multi-hop retrieval/answer composition, especially cases where not all linked
targets were retrieved.

## [2026-04-30] eval | Expanded workspace/core ablation

Added and ran the expanded workspace/core ablation gate:

- Runner: `scripts/112_run_expanded_workspace_core_ablation.sh`
- Proof builder: `scripts/113_build_expanded_ablation_proof.py`
- Module: `src/qtrm_mm/eval/architecture_ablation_proof.py`
- Markdown: `docs/wiki/decisions/expanded-workspace-core-ablation.md`
- JSON: `docs/wiki/decisions/expanded-workspace-core-ablation-summary.json`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_synth_generalization_s050.jsonl`

Result on the same 72-case expanded MemoryOS gate:

| Mode | Hits | Drop vs full residual |
| --- | ---: | ---: |
| `qtrm_residual_with_evidence` | 49/72 | 0 |
| `qtrm_workspace_off_with_evidence` | 49/72 | 0 |
| `qtrm_core_off_with_evidence` | 49/72 | 0 |

Task-family drops are also zero: abstention `18/24`, conflict `20/24`, and
multi-hop `11/24` for all three modes.

The generated completions are also identical across full residual,
workspace-off, and core-off for all 72 cases. Interpretation: the current
expanded-gate gain is real versus donor-only, but this ablation does not
localize the gain to the latent workspace or recursive core. The measured
behavior is currently consistent with a donor-hidden residual side path that can
solve the synthetic MemoryOS prompts even when workspace/core are disabled. The
next architecture task is to add a stricter causality gate: workspace-only
memory injection, coda-off/residual-head ablations, and training losses that
force evidence into workspace/core states before claiming latent workspace or
recursive-core reasoning.

## [2026-04-30] eval | Strict causality ablation

Added strict causality ablation modes and ran them on the same expanded 72-case
MemoryOS gate:

- `qtrm_coda_off_with_evidence`
- `qtrm_residual_head_off_with_evidence`
- `qtrm_donor_hidden_off_with_evidence`
- `qtrm_workspace_only_with_evidence`
- Runner: `scripts/114_run_expanded_strict_causality_ablation.sh`
- Eval output:
  `runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_synth_generalization_s050.jsonl`

Result:

| Mode | Hits | Drop vs full residual |
| --- | ---: | ---: |
| `qtrm_residual_with_evidence` | 49/72 | 0 |
| `qtrm_coda_off_with_evidence` | 39/72 | +10 |
| `qtrm_residual_head_off_with_evidence` | 26/72 | +23 |
| `qtrm_donor_hidden_off_with_evidence` | 49/72 | 0 |
| `qtrm_workspace_only_with_evidence` | 49/72 | 0 |

Interpretation: the residual head is the main measured source of the current
gain over donor-only, and coda contributes. Direct projected donor hidden states
are not the cause on this gate: removing them keeps `49/72` and identical
completions. `workspace_only` also keeps `49/72`, but because `workspace_off`
also keeps `49/72`, this still does not prove latent-workspace causality. The
next proof must make the task impossible without workspace state, for example
by putting evidence only into workspace-side memory tokens while hiding it from
the normal prompt/donor path.

## [2026-04-30] architecture | Gated latent workspace memory

Added the first LM2/G-MemLLM-inspired gated latent memory path inside
`LatentWorkspace`.

Implementation:

- `src/qtrm_mm/workspace.py` now supports `workspace_memory_gate_enabled` and a
  conservative `workspace_memory_gate_init_bias`.
- Each workspace layer can apply update/reset/candidate gates after context
  cross-attention, letting latent slots preserve, reset, or overwrite state.
- `src/qtrm_mm/qtrm_model.py` returns `workspace_update_gate_mean` telemetry.
- `disable_workspace_memory_gate=True` supports runtime causal ablation.
- `scripts/95_eval_memory_retrieval.py` and
  `scripts/114_run_expanded_strict_causality_ablation.sh` include
  `qtrm_workspace_gate_off_with_evidence`.
- `configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml` enables the gated
  workspace for the next current-architecture probe.

Reference ingest:

- cloned `references/official/lm2` at `5f56b197b735`;
- cloned `references/official/lightmem` at `b11eccd23c7c`;
- downloaded LM2, G-MemLLM, LightMem, MemCoT, ATLAS, and MIRAS PDFs under
  `references/papers`.

Interpretation: this is not full LM2, G-MemLLM, Titans, ATLAS, or MSA. It is a
small, ablatable gated-memory lane for in-context workspace state. The claim is
valid only if future `workspace_gate_off` runs drop below full residual on a
held-out MemoryOS gate.

## [2026-04-30] eval | Layered MemoryOS scoring

Added stricter MemoryOS answer scoring:

- `hit`: backward-compatible permissive normalized alias containment;
- `exact_match`: canonical short answer exactly equals an alias;
- `normalized_exact`: canonical short answer equals an alias after
  case/punctuation/spacing normalization;
- `normalized_contains`: old loose alias-containment behavior;
- `unknown_correct`: UNKNOWN-specific correctness;
- `needs_human_audit` and `audit_reasons`: queue loose hits, UNKNOWN
  repetitions, and misses for human or LLM-as-judge review.

Implementation:

- `src/qtrm_mm/eval/memory_retrieval.py`
- `scripts/95_eval_memory_retrieval.py`
- `scripts/116_rescore_memory_eval.py`
- `docs/wiki/concepts/memory-evaluation-metrics.md`

Rescored the gated-workspace outputs without rerunning model generation:

| Mode | Permissive hits | Exact / normalized exact | Audit items |
| --- | ---: | ---: | ---: |
| `donor_only_with_evidence` | 26/72 | 9/72 | 63/72 |
| `qtrm_residual_with_evidence` | 49/72 | 24/72 | 48/72 |
| `qtrm_workspace_gate_off_with_evidence` | 49/72 | 24/72 | 48/72 |

Interpretation: QTRM residual still improves over donor-only, but exact
short-answer quality is much lower than permissive hit rate. Architecture claims
should now require improvement in permissive hit rate, normalized exact rate,
and audit rate together.

## [2026-04-30] decision | Perceiver/Q-Former workspace status

Reviewed the current latent-workspace lineage against newer memory references.
Decision: do not delete Perceiver/Q-Former/OpenFlamingo-style learned slots, but
demote them to connector/bottleneck baseline.

Current classification:

| Axis | Role |
| --- | --- |
| Perceiver / Q-Former / PerceiverResampler | compact learned-slot connector |
| Qwen2.5-VL-style merger | modern simple multimodal connector reference |
| LM2 / G-MemLLM | internal gated-memory causality target |
| MSA | large external sparse-memory routing target |
| LightMem / MemCoT | external MemoryOS filtering and iterative search target |

Added `docs/wiki/decisions/latent-workspace-prior-decision.md` and updated the
workspace/memory concept pages. The next architecture step should not be "make
the Perceiver stack deeper"; it should be workspace-side evidence injection
where `workspace_off` and `workspace_gate_off` actually reduce score.

## [2026-04-30] architecture | Workspace evidence-only path

Implemented the stricter workspace evidence path.

Previous MemoryOS evals put retrieved evidence directly into the donor-visible
prompt. That made the residual gain useful, but not a proof of latent workspace
reasoning. The new path keeps the visible prompt to task/question text and
encodes retrieved evidence separately as workspace-side donor hidden states.

Implementation:

- `src/qtrm_mm/qtrm_model.py` accepts `workspace_text_states` and
  `workspace_attention_mask`.
- `disable_workspace_memory_context=True` removes this separate evidence memory
  path for ablation.
- `src/qtrm_mm/multimodal_projector.py` supports feature masks for evidence
  memory tokens.
- `scripts/95_eval_memory_retrieval.py` adds
  `--evidence-injection workspace`.
- `qtrm_workspace_memory_off_with_evidence` now measures whether retrieved
  evidence is actually flowing through workspace memory.
- `scripts/117_run_workspace_evidence_path_probe.sh` runs the expanded 72-case
  MemoryOS gate in this stricter mode.
- `src/qtrm_mm/data/jsonl_dataset.py` can split MemoryOS supervised rows into
  visible prompt tokens plus `workspace_input_ids`.
- `src/qtrm_mm/training/train.py` encodes those workspace ids through the frozen
  donor and forwards them as `workspace_text_states`.
- `configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml` and
  `scripts/118_run_workspace_evidence_path_train.sh` define the first training
  run for this path.

Boundary: this is an architecture/eval-path change. Existing checkpoints were
not trained with evidence hidden from the prompt, so this probe is expected to
be a causality harness first and a performance improvement only after retraining
against the new path.

## [2026-04-30] concept | Internalized context engineering

Added a wiki source/concept pair for QTRM as internalized context engineering.

Definition: QTRM is not replacing RAG. It is trying to move selected parts of
external context engineering into trainable, gated, auditable memory routes.

Similar paper families:

- Context Engineering Survey names the broader field.
- ACE is the closest external-context counterpart: it evolves structured
  contexts without changing model weights.
- REALM, RETRO, DSI, and Atlas show retrieval entering model behavior rather
  than staying as prompt stuffing.
- LM2 and G-MemLLM are the closest memory-lane and frozen-backbone gated-memory
  references.
- MSA is the closest current reference for sparse memory routing at 100M-token
  scale.
- LightMem and MemCoT keep the external MemoryOS side relevant through memory
  filtering, consolidation, and iterative search.

QTRM staged path:

```text
MemoryOS retrieval + rerank
-> ACE-style evolving context playbooks
-> workspace-only evidence injection
-> gated core context injection
-> LM2/G-MemLLM-style internal memory lane
-> MSA-style sparse memory routing
```

Decision: keep retrieval quality, workspace causality, core-context causality,
and memory-gate causality as separate ablation gates. Do not claim full
end-to-end memory until each stage passes its own off-switch test.

## [2026-04-30] architecture | Gated core context injection

Downloaded/collected the internalized-context reference set:

- context-engineering and ACE PDFs;
- REALM, RETRO, DSI, Atlas, Retro-Li, and retrieval-enhanced generalization
  PDFs;
- persistent latent memory and LatentMem PDFs;
- cloned ACE, Atlas, REALM sparse checkout, RETRO-pytorch, labml RETRO,
  Retro-Li, DSI-transformers, MemoryLLM, and MemGen.

Implemented the next architecture step after workspace-only evidence:

- `QTRMConfig.core_context_enabled`;
- `QTRMConfig.core_context_gate_init_bias`;
- `QTRMRecursiveCore` gated cross-attention from `z_l`/`z_h` to prelude context;
- `disable_core_context` runtime ablation;
- `core_context_gate_mean` telemetry;
- `qtrm_core_context_off_with_evidence` eval mode.

Reasoning: this follows the LM2/G-MemLLM principle of preserving the base path
while adding a complementary gated memory/context route. It also follows
RETRO/Atlas by allowing retrieved evidence to influence architectural attention
paths, not only prompt strings.

Required proof: after training with `core_context_enabled: true`,
`qtrm_core_context_off_with_evidence` must drop below
`qtrm_residual_with_evidence`. If it does not drop, the path exists but has not
become behaviorally important.

## [2026-04-30] training | Human-like loss first slice

Added the first general preference-loss path and kept repetition handling as an
optional guard:

- JSONL rows with `prompt`, `chosen`, and `rejected` now train the chosen answer
  as SFT and score the rejected answer through a second model forward pass.
- `sequence_average_logprob` computes sequence-level average log-probs.
- `simpo_margin_loss` trains chosen/rejected answer pairs with an explicit
  margin and optional row-level `preference_weight`/`confidence`.
- `loss_preference_weight`, `preference_beta`, and `preference_margin` wire this
  into normal training.
- `scripts/120_run_workspace_evidence_preference_train.sh` runs the
  workspace-evidence preference probe on
  `data/filtered/memory_self_improvement_preferences_analysis.jsonl`.
- `scripts/121_eval_preference_pairs.py` evaluates whether a checkpoint
  actually assigns higher sequence average log-prob to `chosen` than
  `rejected`, reporting raw and weighted accuracy plus margin pass rate.
- `repetition_unlikelihood_loss` still exists, but only as a local guard for
  confident adjacent repeat candidates when the previous token is not the
  current gold target.
- `loss_repeat_unlikelihood_weight` wires the repeat guard into normal training
  while keeping the default disabled.
- `configs/qwen35_2b_4090_workspace_evidence_repeatguard_s050.yaml` defines a
  conservative next probe with repeat unlikelihood weight `0.02`.

Decision boundary: this is not a claim that QTRM now has human cognition. It is
the first practical step toward behavior that learns from bad generations:
ordinary CE for correct answers, preference loss against rejected generations,
optional unlikelihood for repeated failure modes, and a future
process-supervision path for verified correction traces.

## [2026-04-30] training | Preference-only early stop

Ran the first workspace-evidence preference-only probe from
`runs/qwen35_2b_4090_memory_gated_workspace_s050/last.pt` using
`memory_self_improvement_preferences_analysis.jsonl`.

Observed signal:

- preference margins improved very quickly on training pairs;
- by step 25, the logged margin had already flipped positive;
- by steps 300, 400, and 500, the live diagnostic still produced
  `Ilya Chen Chen Chen...` with high repeated-2gram rate;
- `repeat_ul` was logged but had zero training weight in the preference-only
  config.

Decision: preference loss alone is not sufficient for this architecture's
current generation stability. The next probe must train the same pairwise
chosen/rejected objective together with a weak repetition unlikelihood guard.

Added
`configs/qwen35_2b_4090_workspace_evidence_preference_repeatguard_s050.yaml`
and `scripts/122_run_workspace_evidence_preference_repeatguard_train.sh` for
that balanced probe.

## [2026-04-30] architecture | Counterfactual workspace causality probe

Added a research-driven architecture-debugging cycle for the
retrieval-found-but-wrong failure:

- source map and decision ledger:
  `docs/wiki/decisions/research-driven-next-architecture.md`;
- counterfactual data builder:
  `scripts/125_build_workspace_counterfactual_preferences.py`;
- generated data:
  `data/filtered/memory_self_improvement_preferences_workspace_counterfactual.jsonl`;
- training config:
  `configs/qwen35_2b_4090_workspace_evidence_counterfactual_s050.yaml`;
- runner:
  `scripts/126_run_workspace_evidence_counterfactual_train.sh`;
- checkpoint:
  `runs/qwen35_2b_4090_workspace_evidence_counterfactual_s050/last.pt`;
- proof:
  `docs/wiki/decisions/workspace-evidence-counterfactual-trained-ablation.md`.

Mechanism: for the same visible prompt and chosen answer, train true workspace
evidence to score higher than shuffled/counterfactual workspace evidence. This
tests whether workspace evidence actually changes the chosen answer log-prob.

Result:

- pair preference still passes: `preference_accuracy=0.909`,
  `margin_pass_rate=0.909`;
- quick retrieval eval fails: `qtrm_residual_with_evidence=0/4` even though
  `retrieved_target_rate=1.0`;
- workspace/core/memory ablations do not reduce hit rate.

Conclusion: counterfactual workspace loss is useful instrumentation, but the
current architecture still lacks a causal evidence-to-answer bottleneck. The
next implementation target is an Evidence-Bottleneck Decoder or a
verifier-controlled answer gate, not more blind preference tuning.

## [2026-04-30] architecture | Logical-causal evidence bottleneck

Added the next falsifiable architecture probe for the
retrieval-found-but-wrong failure.

New source map and papers:

- `docs/wiki/sources/logical-causal-trust.md`;
- downloaded ProoFVer, FOLIO, LINC, NL2LOGIC, insufficient-evidence,
  RAGONITE, CF-RAG, and FaithfulRAG PDFs under
  `references/papers/logical-causal-trust/`;
- cloned `references/official/linc`, `references/official/folio`, and
  `references/official/nl2logic`.

Implemented mechanism:

- support/refute/missing verifier heads over the recursive workspace state;
- causal evidence gate;
- optional suppression of QTRM residual logits when no workspace evidence is
  present;
- logical evidence and causal gate losses using true/counterfactual workspace
  pairs;
- ablation mode `qtrm_evidence_bottleneck_off_with_evidence`.

Experiment entry point:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/127_run_logical_causal_bottleneck_train.sh
```

Acceptance gate: this is not accepted until full evidence beats the previous
`0/4` quick result and memory/gate-off ablations cause a measurable answer
change.

## [2026-04-30] architecture | LeWM core trajectory prediction

Mapped the official LeWorldModel implementation
`references/official/le-wm@bf04d3e8c375` onto QTRM's TRM-style recursive core.

Mechanism:

- keep the existing token-latent LeWM/JEPa head;
- add optional LeWM prediction over recursive `z_H` core trajectories;
- build a simple action trace for the first probe:
  `OBSERVE` or `RETRIEVE` -> `VERIFY` -> `ANSWER`;
- train with `MSE(predicted next z_H, next z_H) + SIGReg(z_H trajectory)`;
- expose `core_world_model` metric in the normal training loop.

Added:

- `core_world_model_enabled` model config fields;
- `loss_core_world_model_weight` train config field;
- `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`;
- `scripts/128_run_lewm_core_world_model_probe.sh`;
- tests for action routing, core-world-model outputs, and loss metrics.

This is still not a full LeWorldModel reproduction: it uses QTRM text/workspace
latents, not pixel trajectories, and the action trace is fixed probe metadata.

## [2026-04-30] architecture | Canonical active-vs-disabled ledger

Added a canonical architecture matrix so implemented features are no longer
implicitly treated as active architecture.

New/updated docs:

- `docs/wiki/architecture/canonical-architecture-matrix.md`
- `docs/wiki/architecture/qtrm-forward-pass.md`
- `docs/wiki/architecture/paper-diagram-prompts.md`

Key boundary recorded:

- The current canonical probe is
  `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`.
- Active path includes donor logits, bounded residual gate, workspace-only
  evidence, workspace memory gate, core context gate, evidence bottleneck,
  counterfactual/preference/repeat/donor-KL losses, and core LeWM trajectory
  loss.
- Token-level JEPA, controller aux heads, core halting, donor annealing,
  multimodal scaffold, external reranker, symbolic verifier, and persistent
  neural memory are implemented or referenced but not active canonical claims.
- Donor-free generation is not yet equivalent to donor-backed gated residual
  fusion because the gate applies to `qtrm_residual_logits` in donor-fusion
  mode.

## [2026-05-01] architecture | OpenMythos-style core-to-text path

Failure ledger:

- Failure: after the evidence-bottleneck fix, ordinary prompts no longer forced
  QTRM to donor-only, but recursive `z_H` could still be weakly causal because
  the prelude text context had an easy bypass into coda.
- Evidence: `configs/qwen35_2b_4090_hf_first_wave_warmup_s400.yaml` used
  `n_coda_layers: 2` with inherited `attn_every: 4`, so no explicit coda
  attention layer was guaranteed. The model could rely on donor/text logits and
  leave the latent recurrent prefix underused.
- Prior checked: `references/official/openmythos@8c68c1f`. It is speculative,
  not official Claude evidence, but its useful architectural invariant is
  Prelude -> recurrent state -> Coda without an unchanged hidden-state bypass.
- Recommended candidate: keep QTRM donor-backed and latent-workspace based, but
  add a gated `z_H -> text_context` cross-attention bridge before coda.

Implemented:

- `QTRMConfig.core_to_text_enabled`
- `QTRMConfig.core_to_text_gate_init_bias`
- `QTRMConfig.core_to_text_gate_min`
- `QTRMMultimodalModel.forward(..., disable_core_to_text=True)`
- output telemetry `core_to_text_gate_mean`
- v2 config `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`
- runner `scripts/135_run_hf_first_wave_warmup_v2.sh`

The v2 config also sets `coda_attn_every: 1`, so text tokens can explicitly
attend to the recurrent workspace prefix. This is a conservative adaptation of
OpenMythos's recurrent-output-to-coda idea, not an OpenMythos clone.

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_model_config.py \
  tests/test_core_halting.py \
  tests/test_hf_distill_training_mix.py \
  tests/test_jsonl_dataset_supervised.py
```

Result: `39 passed`.

Next gate: train the v2 warmup checkpoint, then compare residual, donor-only,
core-off, and `disable_core_to_text` evals. Acceptance requires residual logits
and generation behavior to change under the core-to-text ablation on prompts
where latent recurrence is expected to matter.

## [2026-05-01] tooling | Qwen-Scope donor feature logger

Added Qwen-Scope tooling for donor residual-stream analysis.

Source:

- `docs/wiki/sources/qwen-scope.md`
- Hugging Face collection: <https://huggingface.co/collections/Qwen/qwen-scope>
- Qwen3.5-2B Base SAE repo:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100>

Implemented:

- `src/qtrm_mm/qwen_scope.py`
  - loads official SAE tensor dicts;
  - validates `W_enc`, `W_dec`, `b_enc`, `b_dec`;
  - extracts top-k feature ids/values from donor residual states;
  - emits JSON-ready feature records;
  - uses `attention_mask` for last-nonpad token selection so padded EOS tokens
    are not mistaken for prompt endings.
- `scripts/136_qwen_scope_probe.py`
  - loads Qwen donor with `AutoModelForCausalLM`;
  - downloads selected Qwen-Scope SAE layers;
  - writes prompt/layer/token feature records as JSONL.
- tests:
  - `tests/test_qwen_scope.py`
  - `tests/test_qwen_scope_probe_script.py`

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py
```

Initial result: `5 passed`.

Follow-up execution:

- v2 warmup finished and saved
  `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`;
- checkpoint NaN/Inf scan: `floating_tensors=284`, `bad_tensors=0`;
- first Qwen-Scope probe showed a padding bug: short prompts recorded
  `<|endoftext|>` as the last token;
- fixed `qwen_scope_feature_records(..., attention_mask=..., token_position="last_nonpad")`;
- reran the probe:
  `runs/qwen_scope/qwen35_2b_base_layers_0_12_23.jsonl`;
- records: 9 rows, 3 prompts x 3 layers.

Post-fix verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py \
  tests/test_model_config.py \
  tests/test_hf_distill_training_mix.py
```

Result: `35 passed`.

Current boundary: Qwen-Scope is not an architecture replacement. It is a donor
diagnostics instrument for repetition, evidence-missing, unsupported-answer,
and donor-vs-QTRM residual feature analysis. Do not run the heavy probe while
the v2 QTRM training process is occupying GPU memory.

## [2026-05-01] analysis | Qwen-Scope repeat-vs-normal probe

Standardized the one-off Qwen-Scope comparison into reusable tooling:

- `compare_qwen_scope_feature_groups` in `src/qtrm_mm/qwen_scope.py`;
- `scripts/137_compare_qwen_scope_groups.py`;
- `tests/test_qwen_scope_compare_script.py`.

Ran a curated repeat-vs-normal probe:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
uv run python scripts/136_qwen_scope_probe.py \
  --device cuda --load-in-4bit --dtype bfloat16 --top-k 50 \
  --layer 0 --layer 12 --layer 23 \
  --prompt "Explain quantum entanglement in simple terms." \
  --prompt "양자 컴퓨팅이란 무엇인가요?" \
  --prompt "Determine whether the claim is supported by the evidence. Claim: The Eiffel Tower is in Berlin." \
  --prompt "양자 컴퓨팅이란 무엇인가요? Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze Freeze" \
  --prompt "Explain quantum entanglement in simple terms. This is a crucial role in the world of the world of the world of the world of the world" \
  --prompt "Explain quantum entanglement in simple terms. This is a crucial role in the world of the world of the world of the world of" \
  --out runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_layers_0_12_23.jsonl
```

Then compared normal indices `0,1,2` against repeated indices `3,4,5`:

```bash
PYTHONPATH=src uv run python scripts/137_compare_qwen_scope_groups.py \
  --input runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_layers_0_12_23.jsonl \
  --out runs/qwen_scope/qwen35_2b_base_repeat_vs_normal_curated_summary.json \
  --normal 0,1,2 \
  --repeat 3,4,5 \
  --feature-limit 20 \
  --top-output 15
```

Result summary:

- layer 0: no shared repeat top features; repeat-shared-not-normal examples
  include `16910`, `7605`;
- layer 12: strongest small-set separation; repeat common top features are
  `847`, `2761`, `22167`, `24725`, `25296`, `26397`;
- layer 23: repeat common top features are `29838`, `30452`, `31860`, with
  repeat-shared-not-normal examples `2248`, `12018`, `5121`.

Interpretation: these are candidate diagnostic features, not proven causal
repetition features. The next gate is to run QTRM-generated failures through the
same logger and compare generated repeats against normal completions.

Verification:

```bash
PYTHONPATH=src uv run --with pytest pytest \
  tests/test_qwen_scope.py \
  tests/test_qwen_scope_probe_script.py \
  tests/test_qwen_scope_compare_script.py
```

Result: `9 passed`.

## [2026-05-01] analysis | Qwen-Scope on actual QTRM generated repeats

Ran v2 QTRM generation using:

- config:
  `configs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400.yaml`
- checkpoint:
  `runs/qwen35_2b_4090_hf_first_wave_warmup_v2_s400/last.pt`
- output:
  `runs/qwen_scope/qtrm_v2_generated_for_scope.jsonl`

Generation split:

- normal/low-repeat: prompt indices `0,1`;
- repeat failures: prompt indices `2,3,4`.

Observed failures:

- index `2`: claim/evidence prompt repeats claim/evidence statements,
  `repeated_2gram_rate=0.762`, `repeated_3gram_rate=0.742`;
- index `3`: math prompt begins a second copied problem,
  `repeated_2gram_rate=0.254`;
- index `4`: Korean history prompt repeats multiple-choice options,
  `repeated_2gram_rate=0.175`.

Ran Qwen-Scope over the generated texts:

- prompt file:
  `runs/qwen_scope/qtrm_v2_generated_texts_for_scope.txt`;
- SAE output:
  `runs/qwen_scope/qtrm_v2_generated_layers_0_12_23.jsonl`;
- comparison summary:
  `runs/qwen_scope/qtrm_v2_generated_repeat_vs_normal_summary.json`.

Overlap between curated repeated-string candidates and actual QTRM-generated
repeat candidates:

- layer 0: none;
- layer 12: `847`;
- layer 23: `29838`, `31860`.

Interpretation: Qwen-Scope now gives a concrete repeat-state diagnostic signal
on actual generated QTRM failures, especially in late donor layer 23. This is
not causal proof yet. The next falsifiable step is a larger generated-failure
set and a detector that predicts high repeated-ngram rate from SAE candidate
activations before any architecture/steering changes are considered.

## [2026-05-01] architecture | Root-architecture causality gate

Added an automated big-structure gate:

- module: `src/qtrm_mm/eval/root_architecture_gate.py`;
- runner: `scripts/148_build_root_architecture_gate.py`;
- tests: `tests/test_root_architecture_gate.py`;
- output: `docs/wiki/decisions/root-architecture-causality-gate.md`;
- summary: `docs/wiki/decisions/root-architecture-causality-gate-summary.json`.

Result on
`runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_counterfactual_32tok_trained_s050.jsonl`:

| Check | Result |
| --- | --- |
| status | `rejected` |
| baseline | `qtrm_residual_with_evidence=0/4` |
| workspace memory off | `0/4`, same completions `4/4` |
| core context off | `0/4`, same completions `4/4` |
| failed checks | baseline no successes, no critical causal drop, critical ablations match baseline identity |

Interpretation: this checkpoint does not prove that the latent workspace/core
path is causally necessary. The next architecture work should move answer
formation onto a forced workspace/evidence bottleneck instead of adding another
local loss or reranker.

## [2026-05-01] experiment | Logical-causal bottleneck quick gate

Ran the implemented logical-causal evidence bottleneck probe:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/127_run_logical_causal_bottleneck_train.sh
```

Artifacts:

- checkpoint: `runs/qwen35_2b_4090_logical_causal_bottleneck_s050/last.pt`;
- pair eval: `runs/eval/logical_causal_bottleneck_pair_eval_s050.jsonl`;
- memory eval:
  `runs/eval/memory_reasoning_heldout_expanded_logical_causal_bottleneck_32tok_trained_s050.jsonl`;
- ablation proof:
  `docs/wiki/decisions/logical-causal-bottleneck-trained-ablation.md`;
- root gate:
  `docs/wiki/decisions/logical-causal-bottleneck-root-gate.md`.

Results:

| Metric | Value |
| --- | ---: |
| preference accuracy | `0.727` |
| margin pass rate | `0.727` |
| memory residual hit rate | `0/4` |
| retrieved target rate | `1.0` |
| target recall mean | `0.917` |
| root gate | `rejected` |

Diagnostic generations at steps 100, 200, and 300 still repeated the hidden
workspace-evidence phrase and leaked `<think>` on the second prompt. The memory
eval showed evidence copying and prompt continuation instead of short answers.

Interpretation: the bottleneck is not completely inert, because
`evidence_bottleneck_off` changes some completions. But it is not a useful
causal answer path yet, because the full residual baseline remains `0/4`.
Next fix: force a visible answer channel during MemoryOS decoding/training and
then retest the same root gate.

## [2026-05-01] eval | MemoryOS guarded decoding

Added the same runtime guards used by `92_eval_qtrm_logits.py` to
`scripts/95_eval_memory_retrieval.py`:

- `--suppress-visible-reasoning-tokens`;
- `--no-repeat-ngram-size`;
- runner env in `scripts/117_run_workspace_evidence_path_probe.sh`:
  `SUPPRESS_VISIBLE_REASONING=1`, `NO_REPEAT_NGRAM_SIZE=2`.

Guarded eval command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  CONFIG=configs/qwen35_2b_4090_logical_causal_bottleneck_s050.yaml \
  CHECKPOINT=runs/qwen35_2b_4090_logical_causal_bottleneck_s050/last.pt \
  MAX_CASES=4 SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 \
  OUT=runs/eval/memory_reasoning_heldout_expanded_logical_causal_bottleneck_32tok_guarded_s050.jsonl \
  bash scripts/117_run_workspace_evidence_path_probe.sh
```

Result:

- memory residual hit rate remains `0/4`;
- root gate remains `rejected`;
- `workspace_memory_off` same-completion rate remains `0.75`;
- `core_context_off` same-completion rate remains `1.0`;
- outputs still copy evidence text or continue the prompt.

Conclusion:
the answer-channel runtime guard is useful for generic generation, but it does
not solve MemoryOS answer extraction. The next architecture candidate must train
and/or force a short answer decoder, not just suppress bad tokens.

## [2026-05-01] experiment | Workspace-evidence supervised path

Ran the direct hidden-workspace evidence supervised path:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  MAX_CASES=4 SUPPRESS_VISIBLE_REASONING=1 NO_REPEAT_NGRAM_SIZE=2 \
  SAVE_EVERY=200 bash scripts/118_run_workspace_evidence_path_train.sh
```

Artifacts:

- checkpoint: `runs/qwen35_2b_4090_workspace_evidence_path_s050/last.pt`;
- eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_path_32tok_trained_s050.jsonl`;
- ablation proof:
  `docs/wiki/decisions/workspace-evidence-path-trained-ablation.md`;
- root gate:
  `docs/wiki/decisions/workspace-evidence-path-trained-root-gate.md`.

Quick-gate result:

| Mode | Hits |
| --- | ---: |
| donor-only | `0/4` |
| QTRM residual | `3/4` |
| workspace off | `3/4` |
| core off | `3/4` |
| workspace memory off | `3/4` |
| evidence bottleneck off | `3/4` |
| residual head off | `0/4` |
| coda off | `2/4` |

Root gate: `rejected`.

Interpretation: this is an important negative result. The model learned a
residual/coda answer prior that improves the quick score, but hidden workspace
evidence is still not causally necessary. More steps on this same architecture
would likely increase prior/memorization and repetition, not prove latent
workspace reasoning.

Next required gate: counterfactual workspace-swap eval. Keep the visible
question fixed, swap only hidden workspace evidence, and require the answer to
change or become `UNKNOWN`. Without that, QTRM is still a residual answer-style
adapter, not a hidden-evidence reasoner.

## [2026-05-01] eval | Workspace counterfactual swap root gate

Added a counterfactual workspace-swap case builder:

- `scripts/build_workspace_counterfactual_eval_cases.py`;
- `tests/test_workspace_counterfactual_eval_cases.py`;
- generated quick set:
  `data/eval/memory_reasoning_heldout_workspace_swap_4.jsonl`.

Command:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src uv run python \
  scripts/95_eval_memory_retrieval.py \
  --config configs/qwen35_2b_4090_workspace_evidence_path_s050.yaml \
  --checkpoint runs/qwen35_2b_4090_workspace_evidence_path_s050/last.pt \
  --cases data/eval/memory_reasoning_heldout_workspace_swap_4.jsonl \
  --evidence-mode all --evidence-injection workspace \
  --max-new-tokens 32 --suppress-visible-reasoning-tokens \
  --no-repeat-ngram-size 2 --no-logit-shift
```

Result:

| Mode | Hits |
| --- | ---: |
| QTRM residual | `3/4` |
| workspace off | `3/4` |
| core off | `3/4` |
| core context off | `3/4` |
| workspace memory off | `3/4` |
| evidence bottleneck off | `3/4` |
| coda off | `4/4` |
| residual head off | `1/4` |

Root gate: `rejected`.

Interpretation: the model can emit `UNKNOWN` on some swapped cases, but the
same completions appear when workspace/core/memory/evidence-bottleneck paths
are disabled. This rejects the current hidden-workspace causality claim more
strongly than the normal held-out eval. The next implementation should force
the answer logits through a selected-evidence/short-answer bottleneck instead
of letting the residual head learn a general answer prior.

## [2026-05-01] architecture | Answer-bottleneck repetition stop

A first `workspace_answer_bottleneck` prototype was trained for 300 steps.
The run was interrupted before completing full root-gate eval because the live
diagnostic showed strong repetition:

- `Answer: 666666...` at step 100;
- `Answer: 555555...` at step 300;
- Korean diagnostic also drifted into repeated structure.

Important diagnosis:
the live training diagnostic did not inject hidden workspace evidence. It was a
plain prompt diagnostic. The new answer-bottleneck path was still active even
when `workspace_memory_present == 0`, so the model could drive QTRM residual
logits from prompt-only latent state and collapse into repeated local tokens.

Architecture fix:

- added `answer_bottleneck_requires_workspace_memory`;
- when enabled, answer-bottleneck residual logits are multiplied by
  `workspace_memory_present`;
- with no workspace memory, QTRM residual is zero and generation falls back to
  donor logits;
- default `scripts/129_run_workspace_answer_bottleneck_train.sh` diagnostics
  are disabled because plain prompt diagnostics are invalid for this hidden
  workspace path.

Next gate:
rerun answer-bottleneck only with workspace-injected evals. If repetition still
appears when workspace evidence is present, escalate again to an explicit
short-answer/stop-governed output channel instead of training longer.

## [2026-05-01] eval | Governed answer-bottleneck and causal-gated follow-up

After adding `answer_bottleneck_requires_workspace_memory`, the
workspace-injected answer-bottleneck eval no longer showed the numeric
`666...` / `555...` repetition pattern. A decode-time short-answer governor was
added to keep scoring focused on the first answer line:

- `scripts/95_eval_memory_retrieval.py --short-answer-governor`;
- `scripts/117_run_workspace_evidence_path_probe.sh` wiring;
- default enabled in `scripts/129_run_workspace_answer_bottleneck_train.sh`;
- tests in `tests/test_memory_eval_script.py`.

Governed quick gate:

- eval:
  `runs/eval/memory_reasoning_heldout_expanded_workspace_answer_bottleneck_24tok_governed_s050.jsonl`;
- root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-governed-root-gate.md`;
- `qtrm_residual_with_evidence=1/4`;
- `workspace_off=0/4`, `core_off=0/4`,
  `workspace_memory_off=0/4`;
- `core_context_off=1/4` with `4/4` same completions;
- `evidence_bottleneck_off=1/4` with `4/4` same completions.

Interpretation:
the repeated-token failure was fixed as an information-path bug, but answer
selection was not solved. The model still often wrote plausible answer-shaped
text instead of extracting the requested alias from hidden evidence.

A stricter causal follow-up was implemented:

- trainable policy:
  `answer_bottleneck_evidence_only`;
- config:
  `configs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml`;
- runner:
  `scripts/149_run_workspace_answer_bottleneck_causal_train.sh`;
- only answer-bottleneck and logical/causal evidence heads are trained
  (`~1.05M` parameters), initialized from
  `runs/qwen35_2b_4090_workspace_answer_bottleneck_s050/last.pt`;
- counterfactual workspace data:
  `data/filtered/memory_self_improvement_preferences_workspace_counterfactual.jsonl`.

Run result:

- checkpoint:
  `runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt`;
- pair preference eval:
  `preference_accuracy=0.727`, `margin_mean=1.60`;
- training log signal:
  `workspace_margin_logp` stayed near `0.0001`, so true/counterfactual
  workspace evidence was still barely separated;
- normal root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-root-gate.md`,
  status `accepted` only because workspace/core/memory-off drop the single
  successful `UNKNOWN` case;
- swap root gate:
  `docs/wiki/decisions/workspace-answer-bottleneck-causal-swap-root-gate.md`,
  status `rejected`;
- swap baseline `qtrm_residual_with_evidence=0/4`;
- `evidence_bottleneck_off` did better than the full residual on swap
  (`1/4` vs `0/4`), which means the evidence gate is not a reliable answer-use
  mechanism yet.

Conclusion:
do not treat this as solved. The answer-bottleneck/gate path can influence
generation, but it still does not implement robust evidence-to-answer
extraction. The next root-structure candidate should add an explicit
evidence-span/copy or evidence-selector reader objective instead of another
scalar loss on free-form residual logits.

## [2026-05-01] architecture | 2026 prompt-conditioned memory reader update

Research update:
2026 memory work does not support defending QTRM as a literal
`prompt`/`workspace` split in the way ordinary user-facing LLM APIs work.
Long-context systems expose visible context windows and retrieval/tool results,
while memory-LM papers separate memory capacity from reasoning compute through
task-conditioned memory reads, gated memory banks, cross-attention, or sparse
memory routing.

Source ledger:

- `docs/wiki/sources/2026-memory-context-architecture.md`
- `docs/wiki/decisions/research-driven-next-architecture.md`
- `docs/wiki/architecture/paper-diagram-prompts.md` Figure 7

Implemented scaffold:

- `model.evidence_span_reader_enabled`
- `train.loss_evidence_span_reader_weight`
- `trainable_param_policy=evidence_span_reader_only`
- `scripts/build_evidence_span_reader_dataset.py`
- `configs/qwen35_2b_4090_evidence_span_reader_s050.yaml`
- `scripts/150_run_evidence_span_reader_train.sh`
- ablation mode `qtrm_evidence_span_reader_off_with_evidence`

Generated data:

```text
data/filtered/memory_reasoning_synth_span_reader.jsonl
rows=432
found=288
no_answer=144
```

Architecture correction:

```text
visible prompt/question states -> query projection
hidden workspace evidence tokens -> start/end span reader + UNKNOWN head
selected evidence span -> answer-only channel -> bounded donor residual
```

This replaces the rejected pattern:

```text
hidden evidence -> latent workspace -> free-form residual logits
```

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_evidence_span_reader_dataset.py \
  tests/test_model_config.py \
  tests/test_losses.py \
  tests/test_training_checkpoint_init.py \
  tests/test_memory_ablation_modes.py \
  tests/test_root_architecture_gate.py -q
# 94 passed

PYTHONPATH=src uv run python -m py_compile \
  scripts/build_evidence_span_reader_dataset.py \
  scripts/95_eval_memory_retrieval.py \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  src/qtrm_mm/losses.py \
  src/qtrm_mm/data/jsonl_dataset.py \
  src/qtrm_mm/training/train.py \
  src/qtrm_mm/eval/root_architecture_gate.py

bash -n scripts/150_run_evidence_span_reader_train.sh
```

Training result:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/150_run_evidence_span_reader_train.sh

init: runs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050/last.pt
out:  runs/qwen35_2b_4090_evidence_span_reader_s050/last.pt
trainable: 1,837,057 params, 9 tensors
evidence_span_reader: 10.3774 -> 0.5896
evidence_span_start_acc: 0.0000 -> 1.0000
evidence_span_end_acc: 0.0000 -> 1.0000
```

Decision:
the prompt-conditioned reader is now a trained probe, not just a scaffold. It
does prove that prompt states can select hidden workspace evidence spans on the
synthetic MemoryOS span task. It does not yet prove final generative answering:
the selected span still must be wired into an answer-only copy/decoder channel
and then tested under normal and workspace-swap gates.

## [2026-05-01] architecture | ASI controller trace-replay training hook

Root-architecture correction:
before trying to make QTRM replace donor language, train the cognitive loop as
an action controller over verified traces.

Implemented:

- stable `Action.id` mapping:
  `OBSERVE=0`, `RETRIEVE_MEMORY=1`, `VERIFY_EVIDENCE=2`, `ANSWER=3`;
- `trace_replay` JSONL rows in `JsonlTextVisionDataset`;
- `action_policy_loss` over `ControllerHeads.action`;
- `train.loss_action_policy_weight`;
- `trainable_param_policy=controller_only`;
- `scripts/155_build_controller_trace_replay.py`;
- `configs/qwen35_2b_4090_controller_trace_s050.yaml`;
- `scripts/155_run_controller_trace_train.sh`.

This is not yet ASI evidence. It is the first causal training bridge:

```text
TypedContextTape / MemoryOS rows
-> trace_replay action targets
-> frozen QTRM/donor representation
-> ControllerHeads.action
-> later ASI causal-loop gate
```

Next gate:
train the controller head, then compare scripted harness, donor+harness,
QTRM-controller+harness, latent-core-off, world-model-off, and verifier-off.

## [2026-05-01] architecture | Controller trace-replay suspicion closed

Suspicion:
the first controller SFT result collapsed toward a single action because the
trace rows did not give the controller a distinguishable causal state. Step 0,
step 1, and step 2 could share almost the same prompt while requiring different
actions.

Fixes:

- trace-replay inputs now include `trace_step`, `state_summary`, and
  `previous_observation`;
- the same action query is repeated near the sequence tail so the pooled
  controller state can see the current decision;
- `ControllerHeads.action` now pools from the final valid text/coda state
  instead of the last latent workspace slot;
- the evaluation script now bounds default evaluation by JSONL line count, so
  finite trace datasets do not silently run forever.

Full held-in trace-replay result:

```text
checkpoint: runs/qwen35_2b_4090_controller_trace_s300/last.pt
data: data/filtered/asi_controller_trace_replay.jsonl
samples: 1296
accuracy: 1.0000
RETRIEVE_MEMORY: 432/432
VERIFY_EVIDENCE: 432/432
ANSWER: 432/432
missing_keys: []
unexpected_keys: []
summary: docs/wiki/decisions/controller-trace-s300-eval-summary.json
```

Held-out contract check:

```text
source cases: data/eval/memory_reasoning_heldout_expanded_72.jsonl
trace data: data/eval/asi_controller_trace_replay_heldout_72.jsonl
summary: docs/wiki/decisions/controller-trace-s300-heldout-eval-summary.json
samples: 216
accuracy: 1.0000
RETRIEVE_MEMORY: 72/72
VERIFY_EVIDENCE: 72/72
ANSWER: 72/72
```

Verification:

```bash
PYTHONPATH=src uv run --with pytest --with pyyaml pytest \
  tests/test_jsonl_dataset_supervised.py \
  tests/test_losses.py \
  tests/test_training_checkpoint_init.py \
  tests/test_model_config.py \
  tests/test_controller_trace_replay_script.py \
  tests/test_controller_trace_policy_eval_script.py \
  tests/test_asi_cognitive_loop_contract.py \
  tests/test_asi_causal_loop_gate_script.py -q
# 105 passed

PYTHONPATH=src python3 -m py_compile \
  src/qtrm_mm/agentic/cognitive_loop.py \
  src/qtrm_mm/data/jsonl_dataset.py \
  src/qtrm_mm/losses.py \
  src/qtrm_mm/config.py \
  src/qtrm_mm/qtrm_model.py \
  src/qtrm_mm/training/train.py \
  scripts/155_build_controller_trace_replay.py \
  scripts/156_eval_controller_trace_policy.py \
  scripts/154_build_asi_causal_loop_gate.py

bash -n scripts/155_run_controller_trace_train.sh
```

Decision:
this closes the local action-policy wiring bug and shows the explicit
step/state controller contract transfers to unseen MemoryOS eval cases. It
does not prove ASI, open-domain reasoning, answer correctness, or donor-free
language ability. The next gate must test the trained controller inside the
causal-loop harness against donor/scripted baselines and component-off
ablations.

## [2026-05-01] architecture | Stage-1 ASI controller causal-loop gate

Ran the trained controller through an explicit action-policy causal-loop gate.

Artifacts:

- script: `scripts/157_eval_asi_controller_causal_loop.py`;
- summary: `docs/wiki/decisions/asi-controller-causal-loop-s300-summary.json`;
- report: `docs/wiki/decisions/asi-controller-causal-loop-s300.md`;
- metrics for standard gate:
  `docs/wiki/decisions/asi-controller-causal-loop-s300-metrics.json`;
- standard ASI gate:
  `docs/wiki/decisions/asi-causal-loop-gate.md`;
- standard ASI gate summary:
  `docs/wiki/decisions/asi-causal-loop-gate-summary.json`.

Held-out action-policy metrics:

```text
qtrm_harness:               1.0000
qtrm_latent_core_off:       0.7037
qtrm_workspace_off:         0.6667
qtrm_workspace_memory_off:  1.0000
qtrm_core_to_text_off:      1.0000
samples:                   216
```

Standard ASI gate result:

```text
status: rejected
gain_over_donor_harness:    0.0000
gain_over_scripted_harness: 0.0000
latent_core_drop:           0.2963
world_model_drop:           0.0000
verifier_drop:              0.0000
failed:
  qtrm_does_not_beat_donor_harness
  qtrm_does_not_beat_scripted_harness
  world_model_not_causal
  verifier_not_causal
```

Interpretation:
this is a useful rejection. It proves the latent core/workspace participate in
the controller's final `ANSWER` decision, but the current action-policy metric
does not prove QTRM beats a scripted/donor harness. It also confirms the world
model and verifier are not yet causal in action selection. The next
architecture step is to wire verifier/world-model outcomes into controller
state or rewards, then rerun the same gate.

## [2026-05-02] architecture | Canonical SSOT causal training loss

Added direct training pressure for the simplest answer path:

```text
full canonical SSOT greedy path
> core_off/workspace_off/evidence_bottleneck_off ablations
```

Artifacts:

- loss: `src/qtrm_mm/losses.py::canonical_causal_ablation_loss`;
- config: `configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml`;
- script: `scripts/168_run_canonical_ssot_greedy_causal_train.sh`;
- contract: `docs/wiki/architecture/kiss-yagni-dry-ssot-contract.md`.

Important detail:
ablation log-probs are stop-gradient targets. The objective should improve the
full path, not teach shared weights to degrade ablated paths.

Status:
unit/config/script tests pass. The architecture claim is still rejected until
the canonical answer gate shows held-out full-QTRM improvement over donor-only
and causal drops for component-off ablations.

## [2026-05-02] architecture | Core-to-text bridge tried for canonical causal path

Finding:
the first canonical causal loss used donor-fused logits, which hid QTRM causal
differences. It now uses `qtrm_residual_logits`. After that fix, the plain
canonical config still had `canonical_causal_margin=0.0000`, meaning
`core_off`/`workspace_off` did not change the QTRM residual answer path.

Architecture change:

- added `configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml`;
- enabled `core_context_enabled`;
- enabled `core_to_text_enabled`;
- set `coda_attn_every: 1`;
- added `qtrm_core_to_text_off_with_evidence` to the canonical answer gate.

Result:

```text
checkpoint: runs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050/last.pt
report: docs/wiki/decisions/canonical-ssot-coretotext-answer-gate-after-causal-train-4.md
status: rejected
full QTRM: 2 / 4
donor-only with evidence: 2 / 4
workspace_off/core_off/core_to_text_off: 2 / 4
```

Interpretation:
`core_to_text` is the first candidate that makes canonical residual logits and
some completions differ under ablation, so it is a better root bridge than the
previous config. It is not accepted: no held-out answer-score drop appears when
the bridge is disabled. Next work should train a stronger forced answer
bottleneck or increase the causal objective on a larger split, while keeping
the SSOT greedy answer gate as the acceptance test.

## [2026-05-02] architecture | Canonical causal margin probe added

Added a fast residual-logprob diagnostic before slow generation gates:

```text
scripts/169_probe_canonical_causal_margin.py
```

It compares the full canonical SSoT path against `core_off`, `workspace_off`,
`core_context_off`, `core_to_text_off`, and `evidence_bottleneck_off` on
`qtrm_residual_logits`. This catches the specific failure where training loss
looks causal but donor-fused greedy output hides the effect.

Result for `canonical_ssot_coretotext_forced_s150`:

```text
core_off margin:        0.1735
workspace_off margin:   0.1866
core_to_text_off margin:0.1348
```

The following greedy gate was still rejected because all critical ablation
strings matched the full model. Conclusion: the bridge moved confidence, but
not top-1 answer decisions.

## [2026-05-02] architecture | Core answer bottleneck accepted on 4-case SSoT gate

Implemented the stronger canonical candidate:

```text
configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150.yaml
```

Key design:
all evidence remains in the canonical prompt token stream, but QTRM residual
answer logits are generated through a latent answer bottleneck over `z_h`
instead of the ordinary text coda path.

Artifacts:

```text
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150/last.pt
margin probe: runs/eval/canonical_causal_margin_probe_core_answer_bottleneck_s150_4_summary.json
gate report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-4.md
```

Result:

```text
status: accepted
full QTRM: 3/4
donor-only with evidence: 2/4
workspace_off: 2/4
workspace_off same-completion rate: 0/4
core_off same-completion rate: 0/4
```

Residual margin probe:

```text
core_off:      0.6939
workspace_off: 2.8459
```

Remaining limitation:
`core_off` changes the strings but does not yet reduce hit count on the 4-case
gate. Scale the accepted candidate to more cases and add a stronger core-only
drop gate before claiming the recursive core is solved.

## [2026-05-02] architecture | Core answer bottleneck rejected on 8-case scale-up

The 4-case acceptance did not scale:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-s150-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    6 / 8
workspace_off_with_evidence: 6 / 8
core_off_with_evidence:      6 / 8
causal modes:                none
```

Interpretation:
the answer bottleneck is a real causal path at small scale, but it is not a
reliable improvement. It overrode donor-correct outputs on 3-hop/location and
temporal/conflict cases.

Safe-gate eval:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_safe_gate_eval.yaml
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-safe-gate-eval-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
causal modes:                none
```

Decision:
do not promote `answer_bottleneck` as the canonical default. Keep it as a
diagnostic architecture showing that a latent path can affect greedy tokens,
then move to selective residual control: donor as default, QTRM intervention
only when calibrated by verifier/reasoning signals.

## [2026-05-02] architecture | Selective residual-gate fine-tune rejected

Ran a conservative gate fine-tune:

```text
config: configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150.yaml
init: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150/last.pt
```

Training kept donor logits at `1.0` and enabled `qtrm_residual_gate`. The final
canonical margin was high (`1.1902`), so the path learned to matter in logprob
space.

Held-out 8-case greedy SSoT gate:

```text
report: docs/wiki/decisions/canonical-ssot-core-answer-bottleneck-selective-gate-s150-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
core_off_with_evidence:      6 / 8
causal modes:                none
```

Conclusion:
do not spend another run on this checkpoint with only scale/gate/margin
tweaks. The next architecture must train explicit verifier and decision
targets: allow, abstain, revise, or search-more. The accepted in-model answer
decision head is the strongest prior artifact, but it must be adapted without
violating the canonical SSoT/greedy contract.

## [2026-05-02] architecture | Canonical decision-token SFT rejected as answer architecture

Added SSOT decision-token data generation:

```text
script: scripts/170_build_canonical_decision_token_data.py
data: data/filtered/memory_reasoning_canonical_decision_tokens.jsonl
rows: 144
decision targets: ANSWER=96, ABSTAIN=48
hidden workspace fields: 0
```

Trained a donor-preserving residual checkpoint:

```text
config: configs/qwen35_2b_4090_canonical_decision_tokens_s120.yaml
runner: scripts/171_run_canonical_decision_token_train.sh
checkpoint: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
final canonical_causal_margin: 0.0725
```

The residual-logprob probe passed on decision-token rows:

```text
core_off margin:      0.0792
workspace_off margin: 0.0933
```

But the canonical greedy answer gate rejected the checkpoint. I also found and
fixed an eval mismatch: the generic answer-gate script had hardcoded
`QTRM_LOGITS_SCALE=0.10`, while this config uses `0.30`. The generic runner now
uses the config unless a scale override is explicit, and the decision-token
runner sets `0.30`. The corrected run still failed:

```text
report: docs/wiki/decisions/canonical-decision-tokens-s120-qscale030-answer-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

Decision:
decision-token labels alone are not enough. The next architecture must make
the verifier/decision state causally control the final answer logits, not only
improve average label logprob under a donor-dominated greedy decode.

## [2026-05-02] architecture | Hidden-only answer-decision head rejected

Trained a minimal in-model answer-decision head on SSOT prompt rows:

```text
config: configs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200.yaml
data: data/filtered/memory_reasoning_answer_decision_truthcal_train144.jsonl
init: runs/qwen35_2b_4090_canonical_decision_tokens_s120/last.pt
checkpoint: runs/qwen35_2b_4090_canonical_ssot_hidden_answer_decision_s200/last.pt
trainable params: 513
```

Held-out 8-case gate with `--model-answer-decision`:

```text
report: docs/wiki/decisions/canonical-ssot-hidden-answer-decision-s200-gate-8.md
status: rejected
qtrm_residual_with_evidence: 5 / 8
donor_only_with_evidence:    5 / 8
workspace_off_with_evidence: 5 / 8
core_off_with_evidence:      5 / 8
```

Observed block probabilities on `qtrm_residual_with_evidence`:

```text
blocked: 0 / 8
range: 0.00255..0.00280
```

Interpretation:
the currently frozen hidden representation does not expose answer validity to a
tiny linear decision head. The next viable path is not another hidden-only
threshold. We need an explicit verifier-conditioned answer policy or internally
learned telemetry features that are causal for final logits.

## [2026-05-02] raw intelligence | Depth-supervised recursive core rejected

Added a stricter pure-recursive gate check:

```text
failed check: depth_outputs_identical_across_steps
```

This rejects runs where `core_steps=1/2/4/8` all select the same answer on
every comparable held-out case, even if the core path is active.

Ran the S160 raw-core checkpoint and two S080 depth-supervised follow-ups:

```text
S160 report:
  docs/wiki/decisions/pure-recursive-reasoning-core-s160-depth-gate-8.md

S080 internal depth probe:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-depth-supervised-s080-depth-gate-8.md

S080 final-path CE:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_depth_supervised_finalpath_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-depth-supervised-finalpath-s080-depth-gate-8.md
```

All three runs were rejected on the same 8-case no-evidence gate:

```text
donor_only_no_evidence: 3/8
qtrm_core_off_no_evidence: 3/8
qtrm_core_steps_1/2/4/8_no_evidence: 3/8
depth output diversity: 0/8 cases changed by depth
shortcut records: 0
```

Decision:
this is no longer a RAG, SSOT, formatting, or final-token-loss problem. The
recursive core reaches a fixed-point-like answer after one step. The next
architecture candidate must use depth-conditioned state transitions or exact
intermediate-state supervision so deeper steps have different causal roles.

## [2026-05-02] raw intelligence | Mandatory core-loop readout implemented

Added a stricter loop-causal answer path:

```text
prompt -> frozen embed/prelude/workspace -> recursive z_L/z_H loop
-> core_loop_readout_cross -> LM head -> final text logits
```

Code:

```text
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/config.py
src/qtrm_mm/training/train.py
```

Config updated:

```text
configs/qwen35_2b_4090_pure_recursive_depth_fullseq_kiss_s160.yaml
core_loop_readout_enabled: true
core_loop_readout_requires_core: true
n_coda_layers: 0
core_to_text_enabled: false
trainable_param_policy: core_and_loop_readout
```

Purpose:
the loop is no longer just a latent prefix that coda may or may not use. In
this probe, the final answer residual must be read from the recursive loop
state, and `disable_core=true` zeroes that path. Acceptance still requires the
raw-intelligence gate: deeper loop depth must beat donor and core-off without
MemoryOS or retrieval.

Smoke:

```text
run: pure_recursive_loop_readout_smoke_s001
steps: 1
max_cases: 1
trainable policy: core_and_loop_readout
result: runner completed, raw gate rejected as expected for smoke
report: docs/wiki/decisions/pure-recursive-loop-readout-smoke-s001-depth-gate-1.md
```

## [2026-05-03] raw intelligence | First staged loop-readout gate accepted

Fixed a raw-eval forced-choice tie bug:

```text
problem: all-zero logits from core_off could win by first-choice ordering
fix: top logprob ties become __FORCED_CHOICE_TIE__ and score as miss
script: scripts/192_eval_raw_intelligence.py
test: tests/test_raw_intelligence_eval_script.py
```

Ran mandatory loop-readout S160:

```text
final-target S160:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_s160/last.pt
  heldout8: rejected
  donor 2/8, core_off 0/8, core8 4/8
  failure: no depth scaling; depth outputs identical

staged-target S160:
  checkpoint: runs/qwen35_2b_4090_pure_recursive_loop_readout_staged_s160/last.pt
  heldout8: accepted
  donor 2/8, core_off 0/8, core8 3/8
  depth changed 2/8
  heldout16: rejected
  donor 5/16, core_off 0/16, core8 4/16
  depth changed 3/16
```

Decision:
mandatory loop readout and staged targets are the first combination that passes
the small pure-recursive gate. The result is weak and not yet scalable; do not
claim ASI or robust reasoning. Next work should resume with staged S320/S640 or
a stronger explicit recurrent state-machine core.

## [2026-05-03] ingest | Formal CoT vs latent thought

Added:

```text
paper: A Formal Comparison Between Chain of Thought and Latent Thought
arXiv: https://arxiv.org/abs/2509.25239
pdf: references/papers/recurrent_depth/formal_comparison_cot_latent_thought_2509.25239.pdf
code: references/official/cot-vs-loop@783fa90
wiki: docs/wiki/sources/formal-cot-vs-latent-thought.md
```

Decision:
the paper is a guardrail against overclaiming pure latent recurrence. It argues
for a formal separation: latent thought favors parallelizable computation,
while CoT has advantages for stochastic approximate counting and sampling.
QTRM raw-intelligence gates should therefore split cases by task family instead
of treating one aggregate score as proof that latent loop reasoning works or
does not work.

Implemented:

```text
case metadata:
  reasoning_family
  expected_paradigm
  requires_stochasticity
  parallel_depth_estimate
  serial_trace_length_estimate

gate metadata:
  by_task_family
  by_reasoning_family
  by_expected_paradigm
```

Regenerated:

```text
data/eval/pure_recursive_reasoning_heldout_72.jsonl
data/filtered/pure_recursive_reasoning_train256_cases.jsonl
data/filtered/pure_recursive_reasoning_preferences_train.jsonl
```

## [2026-05-03] architecture | TRM-style answer-state loop

Added an explicit answer-state loop candidate:

```text
y_0 = prompt text hidden states
for each recursive depth t:
  y_t = norm(y_{t-1} + gate(y_{t-1}) * cross_attn(y_{t-1}, z_H_t))
final logits = LMHead(y_T)
```

Code:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_answer_state_loop_s160.yaml
```

Reason:
the previous `core_loop_readout` was causal but still a final readout from the
loop state. The new path makes the visible answer state itself the recurrent
object, closer to TRM's `x, y, z -> z' -> y'` correction loop.

Verification:

```text
smoke:
  run: pure_recursive_answer_state_loop_smoke_s001
  status: rejected as expected
  hits: 0/6
  purpose: checkpoint/eval path only

S160 staged:
  run: pure_recursive_answer_state_loop_s160
  report: docs/wiki/decisions/pure-recursive-answer-state-loop-s160-depth-gate-16.md
  status: rejected
  donor: 5/16
  core_off: 0/16
  core_steps_1: 3/16
  core_steps_2: 4/16
  core_steps_4: 4/16
  core_steps_8: 5/16
  changed_by_depth: 6/16
```

Decision:
this is not accepted raw-intelligence improvement yet because deepest core ties
the donor instead of beating it. It is still a stronger architecture signal
than the previous frozen-output failures: disabling the core collapses the
answer path, deeper latent steps change outputs, and the ladder improves from
depth 1 to depth 8. Next work should target the failure families directly,
especially list transforms and cases where deeper boolean steps flip away from
the correct early answer.

## [2026-05-03] eval+training | causal prefix raw gate

Finding:
the previous forced-choice path fed `prompt + candidate answer` to QTRM in one
forward pass. That is standard for causal donor scoring, but QTRM's
workspace/core path can cross-attend to the whole sequence, including future
answer tokens. This made the raw-intelligence gate less strict than intended.

Implemented:

```text
eval:
  scripts/192_eval_raw_intelligence.py
  scoring=causal_forced_choice
  each answer token is scored from prompt + previous answer tokens only

runner:
  scripts/193_run_pure_recursive_reasoning_depth_gate.sh
  default SCORING=causal_forced_choice

training:
  scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-supervision
  prompt-only input predicts the first answer token

runner:
  scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_SUPERVISION=1
```

Rejected intermediate:

```text
run: pure_recursive_answer_state_loop_hard_s320
change: hard-family oversampling + first-token choice margin, but still
  full-answer training
causal gate: rejected
donor: 5/16
core8: 4/16
reason: no_depth_scaling_gain, deep_core_does_not_beat_donor
```

Accepted result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_s160
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-s160-depth-gate-16.md
status: accepted
donor: 5/16
core_off: 0/16
core1: 3/16
core2: 4/16
core4: 6/16
core8: 6/16
changed_by_depth: 3/16
passed:
  deep_core_beats_core_off
  deep_core_beats_donor
  depth_scaling_gain_present
  depth_outputs_not_all_identical
  no_retrieval_or_memoryos_shortcut
```

Decision:
this is the cleanest raw-recursive signal so far because both training and
evaluation now avoid future-answer leakage. The improvement is still small
and family-limited: arithmetic improves to 3/4 at depth 8, boolean stays 2/4,
symbolic stays 1/4, and list transforms remain 0/4. Next promotion requires a
larger held-out sweep and a second-stage fix for multi-token/list outputs.

## [2026-05-03] training | causal prefix multi-token rejected

Implemented:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-max-target-tokens N
  _prepare_causal_prefix_answer_examples()

scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_MAX_TARGET_TOKENS

tests/test_pure_recursive_depth_supervised_train_script.py
  prefix examples:
    prompt -> answer token 0
    prompt + answer token 0 -> answer token 1
    ...
```

Purpose:
the accepted `causal_prefix_s160` checkpoint only trains the first answer
token. The multi-token variant tests whether answer-state recursion can learn
the next-token chain without leaking future answer tokens through the workspace
or core path.

Verification:

```text
smoke:
  run: pure_recursive_answer_state_loop_causal_prefix_multitoken_smoke_s001
  status: accepted
  purpose: plumbing only
  observed: causal_prefix_examples=4

quality:
  run: pure_recursive_answer_state_loop_causal_prefix_multitoken_s080
  config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080.yaml
  checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080/last.pt
  report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-multitoken-s080-depth-gate-16.md
  status: rejected
  donor: 5/16
  core_off: 0/16
  core1: 3/16
  core2: 4/16
  core4: 4/16
  core8: 4/16
  failed: deep_core_does_not_beat_donor
```

Comparison to canonical baseline:

```text
causal_prefix_s160:
  core8: 6/16
  accepted

causal_prefix_multitoken_s080:
  core8: 4/16
  rejected
```

Decision:
do not promote multi-token causal-prefix training as canonical. The
implementation is useful as an experiment tool, but the result shows that
naively adding later answer-token loss can dilute the raw recursive reasoning
signal. Keep `causal_prefix_s160` as the current canonical raw-intelligence
baseline. The next architecture move should not be "more tokens with the same
loss"; it should explicitly preserve first-token/depth reasoning gains while
adding a separate causal sequence-readout objective or a teacher-generated
latent-state target.

## [2026-05-03] training | split causal-prefix later-token loss rejected

Implemented:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --causal-prefix-later-token-weight
  first answer token weight: 1.0
  later answer token weight: configurable, tested at 0.1

scripts/197_run_pure_recursive_depth_supervised_train.sh
  CAUSAL_PREFIX_LATER_TOKEN_WEIGHT

tests/test_pure_recursive_depth_supervised_train_script.py
  _causal_prefix_example_loss_weight()
```

Hypothesis:
the naive multi-token experiment failed because later-token continuation loss
overpowered the first-token recursive decision objective. Lowering later-token
weight should preserve the accepted depth signal while adding a weak sequence
readout pressure.

Result:

```text
run: pure_recursive_answer_state_loop_causal_prefix_split_s080
config: configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080.yaml
checkpoint: runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_split_s080/last.pt
report: docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-split-s080-depth-gate-16.md
status: rejected
donor: 5/16
core_off: 0/16
core1: 3/16
core2: 5/16
core4: 5/16
core8: 5/16
failed: deep_core_does_not_beat_donor
```

Decision:
split later-token loss is also not canonical. It is less damaging than equal
multi-token loss (`core8 5/16` versus `4/16`), but it still loses the accepted
baseline (`6/16`) and does not solve list transforms (`0/4`). The canonical
raw-intelligence checkpoint remains:

```text
runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
```

Next architecture implication:
do not continue local CE-weight tuning for sequence continuation. The failure
class is architectural: answer sequence continuation and latent recursive
reasoning need separate acceptance gates. The next serious candidate should
train the recurrent state transition or teacher-latent target directly, then
keep answer-token CE as a downstream readout probe.

## [2026-05-03] architecture | LeWM core-world-model raw gate rejected

Implemented raw-trainer support for LeWorldModel-style recursive-core
prediction:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  --core-world-model-weight
  core_world_model_actions
  jepa_world_model_loss over recursive core states

src/qtrm_mm/training/train.py
  core_and_answer_state_loop_and_world_model trainable policy

src/qtrm_mm/qtrm_model.py
  core_world_model predictor max length handles dynamic depth schedules

configs/qwen35_2b_4090_verified_reasoning_lewm_core_s200.yaml
```

Smoke:

```text
runs/qwen35_2b_4090_verified_reasoning_lewm_core_smoke_s020/last.pt
core_world_model loss appears at depth 2/4/8
```

Held-out raw reasoning eval:

```text
baseline:
  runs/eval/verified_reasoning_baseline_interleaved_max5.jsonl
  total 2/30

CE-only S200:
  runs/eval/verified_reasoning_s200_interleaved_max5.jsonl
  total 2/30

CE + LeWM core S200:
  runs/eval/verified_reasoning_lewm_core_s200_interleaved_max5.jsonl
  total 2/30
```

Decision:
LeWM is now wired and trainable, but this experiment does not show raw
reasoning gain. Do not promote it as canonical. The next useful experiment must
directly test transition prediction quality and its correlation with answer
accuracy, instead of assuming next-latent MSE will automatically improve raw
reasoning.

## [2026-05-03] evaluation | LeWM transition quality gate

Added:

```text
scripts/200_eval_core_world_model_transition.py
tests/test_core_world_model_transition_eval.py
docs/wiki/decisions/lewm-transition-quality-gate.md
```

The gate computes masked MSE between:

```text
core_world_model_pred
core_world_model_target
```

and merges existing raw-answer `hit` labels from raw eval JSONL.

Result:

```text
CE-only S200 under LeWM config:
  runs/eval/verified_reasoning_ce_only_s200_transition_max5_summary.json
  core_steps_2 MSE: 1.3253
  core_steps_4 MSE: 1.3295
  core_steps_8 MSE: 1.3322
  QTRM hit rate: 0.0

CE + LeWM core S200:
  runs/eval/verified_reasoning_lewm_core_s200_transition_max5_summary.json
  core_steps_2 MSE: 0.00816
  core_steps_4 MSE: 0.00786
  core_steps_8 MSE: 0.00778
  QTRM hit rate: 0.0
```

Interpretation:
LeWM successfully learns the current recursive-core latent dynamics, but those
dynamics are not yet semantically anchored to answer correctness. This falsifies
the weak claim that lower next-latent MSE alone is enough for raw reasoning.

Next:
build a verifiable symbolic transition-state gate where latent transitions have
known state-update targets, then only promote LeWM if transition quality predicts
final answer correctness and world-model-off ablation drops.

## [2026-05-03] evaluation | Symbolic transition gate for staged LeWM

Added:

```text
scripts/201_eval_symbolic_transition_gate.py
tests/test_symbolic_transition_gate_script.py
configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml
docs/wiki/decisions/pure-recursive-lewm-staged-s200-symbolic-transition-gate.md
```

Runner update:

```text
scripts/197_run_pure_recursive_depth_supervised_train.sh
  CORE_WORLD_MODEL_WEIGHT -> --core-world-model-weight
```

Trained pure-recursive staged LeWM from the canonical causal-prefix baseline:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_lewm_staged_s200/last.pt
raw depth gate:
  runs/eval/pure_recursive_lewm_staged_s200_depth_gate_16.jsonl
  status: accepted
  donor: 5/16
  core_off: 0/16
  core1/core2/core4/core8: 4/16, 5/16, 6/16, 6/16
```

Symbolic transition gate:

```text
canonical S160:
  runs/eval/symbolic_transition_canonical_s160_max16_summary.json
  18/64

verified-reasoning LeWM S200:
  runs/eval/symbolic_transition_lewm_core_s200_max16_summary.json
  19/64

pure-recursive staged LeWM S200:
  runs/eval/symbolic_transition_pure_recursive_lewm_staged_s200_max16_summary.json
  18/64
```

Latent transition quality for pure-recursive staged LeWM:

```text
runs/eval/pure_recursive_lewm_staged_s200_transition_max16_summary.json
core2 MSE: 0.00647
core4 MSE: 0.00740
core8 MSE: 0.00783
transition_mse_hit_pearson: 0.06636
```

Decision:
LeWM learns the recursive latent dynamics, and the raw depth gate remains
accepted, but symbolic intermediate-state accuracy is unchanged from canonical.
Do not promote LeWM as semantic reasoning architecture yet. Next candidate is a
semantic transition head attached to the same recurrent state used by answer
formation, followed by all-depth answer-state CE if the diagnostic shows the
state contains the target but the answer readout ignores it.

## [2026-05-03] architecture | LeWM demoted from canonical answer path

Decision:
The canonical QTRM raw-intelligence path is now the single-trace TRM
answer-state loop, not the LeWM/core-world-model probe.

Updated:

```text
src/qtrm_mm/eval/ssot_contract.py
scripts/95_eval_memory_retrieval.py
tests/test_ssot_contract.py
tests/test_memory_eval_script.py
docs/wiki/architecture/canonical-architecture-matrix.md
docs/wiki/architecture/qtrm-forward-pass.md
docs/wiki/components/qtrm-world-model.md
docs/wiki/concepts/leworldmodel.md
docs/wiki/decisions/lewm-demoted-from-canonical-single-trace-trm.md
```

Canonical settings:

```text
core_world_model_enabled: false
loss_core_world_model_weight: 0.0
donor_logits_scale: 0.0 for the raw-intelligence answer path
answer_state_loop_enabled: true
answer_state_loop_requires_core: true
```

Reason:
LeWM reduced self-latent transition MSE, but it did not improve symbolic
intermediate-state accuracy or answer accuracy. Until a semantic transition,
answer-causal, or planner gate passes, LeWM stays probe-only.

## [2026-05-03] raw intelligence | LeWM-free single-trace rebuild S160 rejected on 72 cases

Rebuilt the canonical single-trace TRM path from the healthy metacog baseline:

```text
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild/no_warmup_rebuilt_s001/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160/last.pt
core_world_model_weight: 0.0
```

The first eval attempt wrote to `runs/eval` and failed with `OSError: [Errno 5]
Input/output error`, so the 72-case eval was rerun to `local_eval/`.

Held-out result:

```text
eval:
  local_eval/pure_recursive_answer_state_loop_causal_prefix_rebuild_s160_depth_gate_72.jsonl
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-causal-prefix-rebuild-s160-depth-gate-72-summary.json
status: rejected
donor_only: 22/72
core_off: 0/72
core1/core2/core4/core8: 19/72, 18/72, 18/72, 18/72
depth outputs identical: 69/72
```

Interpretation:
The core path is causal because `core_off` collapses, but recurrent depth is not
doing useful iterative computation yet. Next experiment should add semantic
depth pressure rather than reintroducing LeWM.

Failure ledger:

```text
docs/wiki/decisions/pure-recursive-single-trace-rebuild-s160-failure-ledger.md
```

## [2026-05-03] raw intelligence | Semantic-depth S120 rejected

Trained the LeWM-free canonical answer-state loop from the S160 rebuild with
staged internal first-token CE and no core-world-model loss:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_semantic_depth_s120/last.pt
core_world_model_weight: 0.0
staged_internal_first_token_ce_weight: 0.5
```

Held-out smoke16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-depth-gate-16-summary.json
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 2/16, 2/16, 2/16, 2/16
depth outputs identical: 16/16
```

Train-distribution smoke16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-train-slice-16-summary.json
status: rejected
donor_only: 3/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 4/16, 4/16
depth outputs identical: 16/16
```

Symbolic transition train16:

```text
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-semantic-depth-s120-symbolic-transition-train16-summary.json
accuracy: 17/64
depth1/depth2/depth4/depth8: 4/16, 5/16, 4/16, 4/16
list_transform: 0/16
```

Interpretation:
The core path remains causal, but staged first-token pressure did not create a
reliable intermediate-state transition. One bounded multi-token causal-prefix
probe is allowed because the prior run trained mostly first answer tokens while
the gate scores full answer strings.

## [2026-05-03] raw intelligence | Multi-token depth S080 rejected

Continued from semantic-depth S120 with multi-token causal-prefix supervision
and hard-family repeats:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_answer_state_loop_multitoken_depth_s080/last.pt
summary:
  docs/wiki/decisions/pure-recursive-answer-state-loop-multitoken-depth-s080-depth-gate-16-summary.json
CAUSAL_PREFIX_MAX_TARGET_TOKENS: 6
FAMILY_REPEAT: list_transform=4,arithmetic_chain=3,symbolic_binding=2
core_world_model_weight: 0.0
```

Held-out smoke16:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 3/16, 4/16, 4/16, 4/16
depth outputs changed: 1/16
list_transform core8: 0/4
```

Interpretation:
The first-token-only defect was not the root cause. The current
answer-state-loop-only design can make the core causal and can create a tiny
depth ladder, but it still cannot beat donor-only or solve list transforms.
The failure ledger now promotes the next candidate to an explicit recurrent
transition-state core.

## [2026-05-04] raw intelligence | Transition-state core implementation surface

Implemented the explicit transition-state candidate surface:

```text
model:
  core_depth_states -> transition_state_features -> answer_state_loop
  disable_transition_state=True ablation

loss/eval:
  canonical causal ablation mode: transition_state_off
  raw eval mode: qtrm_core_steps_8_transition_state_off_no_evidence
  depth gate runner: INCLUDE_TRANSITION_STATE_OFF=1
  depth-supervised contrast:
    --transition-state-contrast-weight
    --transition-state-contrast-margin

config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
```

Verification:

```text
transition-state model/eval/loss focused tests: passed
related core-halting/checkpoint/raw-gate tests: passed
```

Next gate:
Train the S080 candidate and reject it unless core8 beats donor-only/core-off
and `transition_state_off` drops below full core8 on held-out prompt-only
reasoning.

## [2026-05-04] raw intelligence | Transition-state S080 rejected

Ran the first in-model explicit transition-state candidate:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_multitoken_s080/last.pt
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 5/16, 4/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 3/16
state-off output changes: 0/16
shortcuts: 0
```

Passed:

```text
deep_core_beats_core_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
deep_core_does_not_beat_transition_state_off
```

Interpretation:
The recursive core remains causal relative to core-off, but the new
transition-state branch is not causally consumed by the answer path. Since
state-off produces identical outputs on the 16-case gate, this is not yet an
explicit transition-state reasoning model. The next change must put the state
features on a stronger answer-critical path or add direct symbolic
state-supervision before more scale-up.

## [2026-05-04] raw intelligence | Direct transition-state CE S080 rejected

Ran the direct transition-state CE follow-up:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_ce_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_ce_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-ce-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 4/16
state-off output changes: 0/16
training transition_state_first_token_acc: 0.0
```

Interpretation:
The direct CE surface did not make the transition-state branch answer-causal.
The result strengthens the root-structure doubt: a tiny continuous sigmoid
state vector plus gated residual into the answer-state loop is too easy for the
renderer to ignore. The next candidate should use a compact supervised
state-code/token path and require the answer renderer to consume it, instead of
adding more CE weight to the same side readout.

Verification:

```text
165 related unittest cases: passed
py_compile for touched model/train/eval modules: passed
bash -n for pure-recursive eval/train runners: passed
```

## [2026-05-04] raw intelligence | State-code transition path S080 rejected

Implemented and tested a compact state-code path:

```text
model:
  transition_state_code_enabled
  transition_state_codebook_size
  core_depth_states -> transition_state_code_logits
  soft code embedding -> answer_state_loop
  disable_transition_state=True zeroes code embeddings

training:
  --transition-state-code-ce-weight
  transition_state_code_targets = staged first-token id modulo codebook
  transition_state_code_ce_loss

config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_s080.yaml
```

Run:

```text
init:
  runs/qwen35_2b_4090_pure_recursive_transition_state_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 5/16, 5/16, 5/16
transition_state_off: 5/16
depth outputs changed: 4/16
state-off output changes: 0/16
shortcuts: 0
```

Training signal:

```text
early:
  transition_state_code_ce 6.48 -> 0.0047
  transition_state_code_acc 0.0 -> 1.0

late hard windows:
  transition_state_code_ce rose again to 4.95-12.37
  transition_state_code_acc fell to 0.0 on logged windows
```

Interpretation:
The compact code surface is learnable on some staged states, but the final
answer is still not state-code causal. Because full and state-off completions
match 16/16, the active blocker is now the answer renderer, not another state
side loss. The next experiment should remove the ordinary trajectory
cross-attention bypass or replace `answer_state_loop` with a stricter
state-token decoder whose LM logits must be emitted from recurrent state.

Verification:

```text
213 related unittest cases: passed
py_compile for touched model/train/eval modules: passed
bash -n for pure-recursive eval/train runners: passed
```

## [2026-05-04] raw intelligence | Code-only state decoder becomes causal but still rejected

Added a stricter answer renderer option:

```text
model:
  transition_state_code_only_answer_loop

behavior:
  when enabled, answer_state_loop cross-attends to the state-code token only
  rather than concatenating [state_code_token, trajectory_workspace_state].
```

This removes the ordinary trajectory cross-attention bypass that let previous
transition-state candidates ignore the explicit state path.

Run:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_only_s080.yaml
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s080/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_only_s080_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-only-s080-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 4/16, 4/16, 4/16, 5/16
transition_state_off: 3/16
state-off output changes: 6/16
shortcuts: 0
```

Passed:

```text
deep_core_beats_core_off
deep_core_beats_transition_state_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
```

Interpretation:
This is the first explicit transition-state candidate where the state path is
held-out answer-causal. The previous failure class was architectural: the
answer renderer could ignore the state token. The current failure class is now
capability/training: the causal state-token decoder ties donor-only but does not
beat it.

Next:

```text
Continue from the code-only checkpoint with a longer S240/S500 run, keep
transition_state_off in the gate, and tune only objectives that improve hard
families without losing the new state-off causal drop.
```

## [2026-05-04] raw intelligence | Code-only S240 keeps causal state path but loses to donor

Continued the strict code-only state decoder from the S080 checkpoint:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_code_only_s080.yaml
init:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s080/last.pt
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_transition_state_code_only_s240/last.pt
eval jsonl:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_transition_state_code_only_s240_depth_gate_16.jsonl
summary:
  docs/wiki/decisions/pure-recursive-transition-state-code-only-s240-depth-gate-16-summary.json
```

Gate result:

```text
status: rejected
donor_only: 5/16
core_off: 0/16
core1/core2/core4/core8: 3/16, 3/16, 4/16, 4/16
transition_state_off: 3/16
state-off output changes: 4/16
shortcuts: 0
```

Mode semantics:

```text
donor_only:
  true donor baseline; QTRM residual logits are forced off and donor logits are used.
core_off:
  internal QTRM ablation; QTRM forward still runs with disable_core=True and
  donor fallback is not forced. It is not equivalent to donor_only.
transition_state_off:
  QTRM core still runs, but the explicit transition-state/code path is disabled.
```

Passed:

```text
deep_core_beats_core_off
deep_core_beats_transition_state_off
depth_scaling_gain_present
depth_outputs_not_all_identical
no_retrieval_or_memoryos_shortcut
```

Failed:

```text
deep_core_does_not_beat_donor
```

Interpretation:
S240 confirms the code-only state-token path is causal, but longer training did
not improve raw reasoning. It regressed from the S080 core8 tie with donor
`5/16` to `4/16`, while donor stayed `5/16`. The final training windows also
showed hard-family spikes (`depth_final_acc=0.0`, high CE), so the next
experiment should not be a blind S500. The active failure is hard-family policy
learning and state-code target stability, not hidden MemoryOS/RAG leakage or
state-path non-causality.

Next:

```text
Do not promote S240 as baseline.
Keep S080 code-only as the canonical causal checkpoint for now.
Build a hard-family curriculum/overfit proof that first makes list_transform
and arithmetic_chain improve under core depth, then re-run the 16-case gate.

## [2026-05-04] evaluation | Forced-choice length normalization fixed hard-family gate bias

The hard-family overfit8 proof exposed a gate-level confound before it exposed a
new model-architecture failure. The first S120 run used legacy summed
answer-token logprob scoring:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_overfit8_s120/last.pt
legacy eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-depth-gate-8-summary.json
status: rejected
donor_only: 1/8
core8: 1/8
transition_state_off: 0/8
list_transform: 0/4
```

Failure inspection showed `list_transform` answers like `208,204` were losing
to the one-token distractor `EMPTY` because the eval ranked candidates by
summed logprob. This was an evaluation-length bias, not clean evidence that the
state-code path could not learn.

Implemented eval contract fix:

```text
scripts/192_eval_raw_intelligence.py
  --choice-score-normalization mean
  choice_scores[].logprob_sum
  choice_scores[].token_count

scripts/193_run_pure_recursive_reasoning_depth_gate.sh
  CHOICE_SCORE_NORMALIZATION=mean

src/qtrm_mm/eval/raw_intelligence_gate.py
  eval_contract in JSON/markdown reports
```

Same checkpoint, same 8 cases, mean-normalized causal forced-choice:

```text
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-mean-depth-gate-8-summary.json
status: accepted
donor_only: 1/8
core_off: 0/8
core1/core2/core4/core8: 2/8, 3/8, 3/8, 3/8
transition_state_off: 0/8
list_transform core8: 2/4
shortcuts: 0
```

Interpretation:
This is a real local raw-intelligence signal for the strict code-only
transition-state path on the overfit8 hard-family proof: core8 beats donor-only
and transition_state_off. It is not yet a held-out scale proof. The next gate
must use mean-normalized scoring and test different held-out indices before any
broader ASI/architecture claim.

## [2026-05-04] raw intelligence | Hard-family heldout200 rejected after local overfit proof

Ran the same S120 hard-family checkpoint on different generated hard-family
indices with the corrected eval contract:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_overfit8_s120/last.pt
heldout cases:
  data/eval/pure_recursive_hard_family_heldout200_cases.jsonl
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_overfit8_s120_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-overfit8-s120-heldout200-mean-depth-gate-8-summary.json
scoring:
  causal_forced_choice + mean choice-score normalization
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 1/8, 1/8, 1/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Interpretation:
The previous accepted result is only a local overfit proof. On heldout index
200, QTRM often keeps outputs in the training numeric range, for example
list-transform expected values around `408,404` but core8 selects values like
`204,202`. That means the current state-code path can memorize a narrow
hard-family slice, but has not learned the abstract operation robustly.

Next:

```text
Train a broader hard-family generalization run:
  train_start_index=100
  train_cases_per_family >= 16
  heldout_start_index=200
  mean-normalized causal forced-choice gate

Acceptance:
  core8 > donor_only
  core8 > transition_state_off
  list_transform core8 > 0/4 on heldout200
```

## [2026-05-04] raw intelligence | Hard-family S240 generalization still rejected

Broadened training from the local overfit slice to 16 cases per hard family:

```text
runner:
  scripts/211_run_pure_recursive_hard_family_generalization_s240.sh
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_hard_family_generalization_s240/last.pt
heldout:
  start_index=200, cases_per_family=4, families=arithmetic_chain,list_transform
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_hard_family_generalization_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-hard-family-generalization-s240-depth-gate-8-summary.json
scoring:
  causal_forced_choice + mean choice-score normalization
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 2/8, 2/8, 2/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Failure:
The current `transition_state_code_targets` objective was still using
first answer-token id modulo the codebook. That makes the code label a token
artifact, not a semantic transition state. It can make the state path causal
locally, but it does not teach stable operations like "filtered evens" versus
"doubled filtered evens" across train/heldout index ranges.

Architecture change started:

```text
semantic transition-state codes:
  arithmetic_chain: sum -> product -> final
  list_transform: filtered -> doubled/final
  symbolic_binding: one-hop -> two-hop/final
  boolean_logic: not_q -> and -> final

preference margin:
  legacy first-token margin kept
  new sequence mode aligns training pressure with mean forced-choice eval

runner:
  scripts/212_run_pure_recursive_semantic_state_sequence_margin_s240.sh
```

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_semantic_state_sequence_margin_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_semantic_state_sequence_margin_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-semantic-state-sequence-margin-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 0/8, 1/8, 1/8, 1/8
transition_state_off: 0/8
list_transform core8: 0/4
```

Interpretation:
Semantic stage codes are still too small. They distinguish "filtered" from
"doubled/final", but they do not carry value-bearing intermediate state. The
list-transform failures stayed half-scale (`204,202`) instead of producing the
heldout doubled values (`408,404`). The next root-architecture candidate must
use a value-bearing transition state, for example continuous transition-state
features or explicit text/numeric state targets, not only a categorical code.

Prepared next root-architecture runner:

```text
scripts/213_run_pure_recursive_value_state_sequence_margin_s240.sh
config:
  configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml
mechanism:
  continuous transition_state_features -> answer loop
  transition_state CE on staged first-token targets
  sequence preference margin
```

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_value_state_sequence_margin_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_value_state_sequence_margin_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-value-state-sequence-margin-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 2/8, 2/8, 2/8
transition_state_off: 1/8
list_transform core8: 0/4
```

Interpretation:
Continuous transition-state features are weakly answer-causal because core8
beats transition_state_off, but they still do not beat donor-only. Training
also kept `transition_state_first_token_acc=0.0`, so first-token staged-state
supervision is not enough to make the latent state carry values. The next
candidate must supervise full value-bearing intermediate states, not just a
stage code or a first token.

Implemented next candidate:

```text
training:
  --staged-internal-sequence-ce-weight
  --staged-internal-sequence-max-target-tokens

target:
  depth_targets[depth] token sequence, not only first token

path:
  prompt tokens -> recurrent core depth state -> core_depth_text_logits per
  depth -> full intermediate value text

runner:
  scripts/214_run_pure_recursive_full_state_sequence_s240.sh
```

Reason:
This keeps the architecture simple and uses the existing causal depth readout.
It directly teaches depth 1/2/4/8 to emit the matching intermediate value
sequence, so list_transform can be penalized for stopping at filtered
half-scale values.

Result:

```text
checkpoint:
  /mnt/nvme1n1p2/qtrm-local-checkpoints/pure_recursive_full_state_sequence_s240/last.pt
eval:
  /mnt/nvme1n1p2/qtrm-eval/pure_recursive_full_state_sequence_s240_heldout200_mean_depth_gate_8.jsonl
summary:
  docs/wiki/decisions/pure-recursive-full-state-sequence-s240-depth-gate-8-summary.json
status: rejected
donor_only: 2/8
core_off: 0/8
core1/core2/core4/core8: 1/8, 1/8, 2/8, 2/8
transition_state_off: 2/8
list_transform core8: 0/4
```

Interpretation:
The full-state sequence loss did not make the transition-state path causal on
heldout: core8 tied both donor-only and transition_state_off. Training also
kept `staged_internal_sequence_acc=0.0` through the run. This is now the third
state-objective failure after semantic code and continuous first-token state,
so the local donor-residual/readout route should not receive more loss-weight
tuning. The next candidate must be a cleaner recurrent state-machine core
trained on solver traces, with state prediction and answer readout measured
separately.

Implemented trace-data foundation for that replacement:

```text
case schema:
  solver_trace:
    - depth
    - operation
    - state_text

builder:
  scripts/215_build_pure_recursive_solver_trace_dataset.py

train cases:
  data/filtered/pure_recursive_solver_trace_train_cases.jsonl
train trace rows:
  data/filtered/pure_recursive_solver_trace_train.jsonl

heldout cases:
  data/eval/pure_recursive_solver_trace_heldout200_cases.jsonl
heldout trace rows:
  data/eval/pure_recursive_solver_trace_heldout200.jsonl
```

Example:

```text
arith-chain-100:
  d1 add_operands: "" -> 110
  d2 multiply_sum: 110 -> 220
  d4 subtract_offset: 220 -> 217
  d8 hold_final: 217 -> 217
```

## 2026-05-04: Solver State-Machine Probe

Added a donor-free, MemoryOS-free recurrent state-machine probe:

```text
src/qtrm_mm/agentic/solver_state_machine.py
scripts/216_train_pure_recursive_solver_state_machine.py
tests/test_solver_state_machine.py
tests/test_pure_recursive_solver_state_machine_train_script.py
```

Key result:

```text
small compact-input probe:
  train rollout_final_exact: 0.90625
  heldout rollout_final_exact: 0.0

large probe:
  train rollout_final_exact: 0.46484375
  heldout rollout_final_exact: 0.0

primitive transition ceiling:
  heldout_large state: 256/256
  heldout_large final: 64/64
```

Decision:
direct recurrent state-string generation is rejected for the next canonical
raw-intelligence architecture. The core should select/compose causal primitive
state transitions, then render from the updated state.

Follow-up:

```text
char operation policy:
  heldout operation_exact: 0.625
  heldout rollout_final_exact: 0.5

structured operation policy:
  heldout operation_exact: 1.0
  heldout rollout_state_exact: 1.0
  heldout rollout_final_exact: 1.0

all-family structured primitive route:
  families: arithmetic_chain, symbolic_binding, boolean_logic, list_transform
  primitive ceiling: state 512/512, final 128/128
  learned structured policy: operation/state/final all 1.0
```

New canonical raw-core candidate:

```text
prompt/question
-> recurrent latent core
-> structured transition metadata head
-> primitive transition executor
-> explicit state
-> answer renderer
```

QTRM integration surface added:

```text
QTRMConfig.primitive_transition_enabled
QTRMConfig.primitive_transition_num_operations
QTRMConfig.primitive_transition_hidden_dim
model output: primitive_transition_operation_logits
trainer arg: --primitive-transition-operation-ce-weight
target: solver_trace.operation
```

Details:

```text
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240.md
docs/wiki/decisions/pure-recursive-solver-state-machine-probe-s240-summary.json
```

## 2026-05-04: Primitive Transition Token Grounding

The mean-pooled prompt context primitive head reached 507/512 on heldout7000
but kept confusing list transform operations:

```text
double_filtered -> second_mapping
```

Implemented a token-level prompt attention path:

```text
core_depth_state query
-> cross-attend over canonical prompt token context
-> primitive transition operation logits
```

This keeps SSOT: prompt/task/evidence information still enters through the same
canonical token stream. There is no hidden MemoryOS/RAG evidence path.

Result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

heldout existing:
  operation accuracy: 256/256 = 1.0000

heldout7000:
  operation accuracy: 512/512 = 1.0000
```

Decision:
token-level prompt grounding is the current canonical primitive transition
selector for this synthetic raw-recursive-core gate.

## 2026-05-04: Primitive Transition Rollout Gate

Added the rollout causality evaluator:

```text
scripts/221_eval_qtrm_primitive_transition_rollout.py
qtrm_mm.agentic.solver_state_machine.rollout_solver_trace_from_operations
```

The evaluator executes QTRM-predicted primitive operations through the explicit
solver transition function and scores state/final answers.

Heldout7000:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/last.pt

operation exact: 512/512
state exact:     512/512
final exact:     128/128
```

Core-off ablation:

```text
operation exact: 0/512
state exact:     0/512
final exact:     0/128
```

This makes the result a causal primitive-reasoning gate: recurrent core
operation sequence -> explicit state update -> final answer.

## 2026-05-04: Runtime Primitive Answer Path

Added a label-free primitive answer runtime:

```text
scripts/222_infer_qtrm_primitive_transition_answer.py
qtrm_mm.agentic.solver_state_machine.answer_from_primitive_operations
```

Runtime contract:

```text
prompt
-> QTRM recurrent core operation logits
-> predicted operation sequence
-> primitive transition executor
-> answer
```

Smokes with the prompt-attention checkpoint:

```text
arithmetic:
  ops: add_operands -> multiply_sum -> subtract_offset -> hold_final
  states: 7010 -> 14020 -> 14017
  answer: 14017

list_transform:
  ops: filter_even -> double_filtered -> hold_final -> hold_final
  states: 7004,7002 -> 14008,14004
  answer: 14008,14004
```

Boundary:
this is not general LM generation yet. It is the first runtime path where QTRM
core predictions produce final answers through a primitive state executor
without using `solver_trace` labels at inference time.

## 2026-05-04: Answer-Only Primitive Runtime Gate

Added the stricter prompt-only answer runtime evaluator:

```text
scripts/223_eval_qtrm_primitive_answer_runtime.py
```

Runtime input excludes `solver_trace`, intermediate states, and the chosen
answer. Scoring only compares the produced answer with `chosen/answer`.

Heldout7000:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000.json

answer exact:
  128/128
```

Core-off:

```text
report:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_promptattn_s480_from_s720/eval_answer_runtime_heldout7000_coreoff.json

answer exact:
  0/128
```

This is now the strongest synthetic primitive-reasoning gate: prompt-only QTRM
core operation predictions produce final answers, and disabling the core
destroys the path.

## 2026-05-04: Donor-Only Baseline Comparison

Added donor-only baseline evaluator:

```text
scripts/224_eval_donor_only_baseline.py
```

Heldout7000 comparison:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

donor-only forced_choice:
   61/128 = 0.4766

donor-only greedy:
   29/128 = 0.2266
```

Donor-only by family:

```text
forced_choice:
  arithmetic_chain: 15/32
  symbolic_binding: 17/32
  boolean_logic:    29/32
  list_transform:    0/32

greedy:
  arithmetic_chain:  0/32
  symbolic_binding:  0/32
  boolean_logic:    29/32
  list_transform:    0/32
```

Conclusion:
On this synthetic primitive-reasoning gate, QTRM primitive runtime beats
donor-only forced-choice and greedy baselines. This remains a scoped benchmark
claim, not a broad claim that QTRM is a better general-purpose LLM than the
donor.

Added an explicit benchmark-definition section to the decision doc. It now
spells out the four families with examples:

```text
arithmetic_chain:
  ((7007 + 3) * 2) - 3 -> 14017

list_transform:
  [7001, 7004, 7002, 7007, 7003] -> 14008,14004

symbolic_binding:
  A -> green -> violet

boolean_logic:
  (P AND NOT Q) OR R
```

This keeps the benchmark claim scoped: procedure selection and explicit state
updates on synthetic primitive families, not open-domain general LLM quality.

## 2026-05-04: OOD Surface-Form Primitive Gate

Added an OOD surface-form gate:

```text
scripts/225_build_pure_recursive_ood_surface_cases.py
data/eval/pure_recursive_primitive_transition_ood_surface_heldout8000_preferences.jsonl
```

Result:

```text
QTRM primitive answer runtime:
  107/128 = 0.8359

QTRM core-off:
    0/128 = 0.0000

donor-only forced_choice:
   71/128 = 0.5547

donor-only greedy:
   32/128 = 0.2500
```

QTRM by family:

```text
arithmetic_chain: 11/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Arithmetic variant breakdown:

```text
variant 0 "Solve this arithmetic expression exactly": 11/11
variant 1 "What integer do you get after evaluating": 0/11
variant 2 "Evaluate the expression ... report result": 0/10
```

Interpretation:
QTRM still beats donor-only on the OOD surface gate, but arithmetic routing is
surface-form sensitive. The next gate should add arithmetic prompt
augmentation and require OOD arithmetic recovery before claiming broader
arithmetic-language generalization.

## 2026-05-04: Surface-Form Augmentation Recovery Gate

Added a canonical plus OOD-surface training mix:

```text
scripts/226_build_pure_recursive_surface_aug_mix.py
data/filtered/pure_recursive_primitive_transition_surface_aug_mix_train.jsonl
```

Resumed from the prompt-attention primitive checkpoint and trained only the
primitive transition operation CE path:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt
```

Result on OOD surface heldout8000:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000

QTRM core-off:
    0/128 = 0.0000

arithmetic_chain: 32/32
symbolic_binding: 32/32
boolean_logic:    32/32
list_transform:   32/32
```

Canonical heldout7000 regression check:

```text
QTRM primitive answer runtime:
  128/128 = 1.0000
```

Interpretation:
The surface-form arithmetic routing failure is recoverable through the causal
primitive-operation path without canonical heldout regression. This is still a
scoped synthetic primitive-runtime result, not broad open-domain LLM quality.

## 2026-05-05: Fixed Operation To General Recursive Core Roadmap

Added a roadmap that explicitly separates the accepted fixed-operation
primitive probe from the final QTRM target:

```text
docs/wiki/decisions/fixed-operation-to-general-recursive-core-roadmap.md
```

The documented stage path is:

```text
fixed operation policy
-> learned latent operation codebook
-> neural state-transition model
-> open-ended recursive reasoning core
-> memory/metacognition composition gates
```

The roadmap also adds promotion, rejection, and workflow checklists so future
work does not confuse a neuro-symbolic scaffold with broad open-ended
reasoning.

## 2026-05-05: Larger OOD Paraphrase Stress Recovery

Added a larger paraphrase stress builder and heldout-separated train/eval split:

```text
scripts/227_build_pure_recursive_ood_paraphrase_stress_cases.py
data/eval/pure_recursive_primitive_transition_ood_paraphrase_stress_heldout10000_preferences.jsonl
data/filtered/pure_recursive_primitive_transition_ood_paraphrase_stress_train11000_preferences.jsonl
```

Baseline from the surface-augmentation checkpoint:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_primitive_transition_surface_aug_s640_from_promptattn/last.pt

stress heldout10000:
  total:            191/256 = 0.7461
  arithmetic_chain:  58/64
  symbolic_binding:  50/64
  boolean_logic:     64/64
  list_transform:    19/64

core-off:
  total:              0/256
```

Trained a heldout-separated recovery checkpoint using only the primitive
transition operation CE path:

```text
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/last.pt
```

Recovered results:

```text
stress heldout10000 raw argmax:
  total:            252/256 = 0.9844
  arithmetic_chain:  64/64
  symbolic_binding:  64/64
  boolean_logic:     64/64
  list_transform:    60/64

stress heldout10000 with state-constrained operation decoding:
  total:            256/256 = 1.0000

canonical heldout7000:
  total:            128/128 = 1.0000

stress core-off:
  total:              0/256 = 0.0000

stress state-constrained core-off:
  total:              0/256 = 0.0000
```

Residual failure:

```text
list_transform variant 7:
  4/8

observed wrong operation sequence:
  filter_even, multiply_sum, hold_final, hold_final

needed second operation:
  double_filtered
```

Interpretation:
The larger paraphrase failure is mostly recoverable through the mandatory
recursive primitive-transition path without canonical regression. The remaining
raw failures are invalid operation choices after a correct first list step. A
state-constrained operation decoder masks those invalid choices and closes the
runtime answer gate to 256/256 while keeping core-off at 0/256. This strengthens
the Stage 0 scaffold runtime, but the raw operation argmax bottleneck remains.
The next raw-intelligence proof is operation-family transfer or a raw
list-transform routing fix, not ASI-scale claims.

## 2026-05-05: Operation-Family Holdout Feasibility Gate

Added a diagnostic builder for fixed-operation family holdout:

```text
scripts/228_build_pure_recursive_operation_family_holdout.py
data/filtered/pure_recursive_primitive_transition_family_holdout_list_train.jsonl
data/eval/pure_recursive_primitive_transition_family_holdout_list_eval.jsonl
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/operation_family_holdout_list_summary.json
```

Generated split:

```text
holdout family: list_transform
train rows:     192
eval rows:       64

unseen eval operations:
  double_filtered, filter_even

shared eval operation:
  hold_final
```

Conclusion:
Full list-family holdout is a reject/diagnostic gate for the current
fixed-label primitive scaffold, not a fair acceptance gate. The eval set needs
`filter_even` and `double_filtered`, but the train split never contains those
operation labels. More training on this split would mostly test impossible
label invention, not recursive raw intelligence.

The architecture implication is to keep the current result as Stage 0 causal
operation routing over supported primitives and move promotion work toward a
learned latent operation codebook or neural transition model.

## 2026-05-05: Stage 1 Latent Action Codebook Feasibility

Added a Stage 1 diagnostic builder that replaces fixed operation-string
targets with family-agnostic latent action roles:

```text
scripts/229_build_pure_recursive_latent_action_codebook_cases.py
data/filtered/pure_recursive_latent_action_codebook_family_holdout_list_train.jsonl
data/eval/pure_recursive_latent_action_codebook_family_holdout_list_eval.jsonl
local_eval/qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/latent_action_codebook_family_holdout_list_summary.json
```

Latent action codebook:

```text
0: extract_or_unary_transform
1: compose_from_previous
2: final_compose_from_previous
3: hold_final
```

Generated split:

```text
holdout family: list_transform
train rows:     192
eval rows:       64

fixed operation view:
  unseen eval operations:
    double_filtered, filter_even

latent action view:
  train latent action codes:
    0, 1, 2, 3

  eval latent action codes:
    0, 1, 3

  unseen eval latent action codes:
    none
```

Conclusion:
The latent-action codebook removes the immediate impossibility found in the
fixed-label operation-family holdout. The list eval family still contains
unseen fixed operations, but it no longer contains unseen latent action codes.

This is a feasibility result only. It does not prove neural execution or broad
reasoning. The next required experiment is to train the `transition_state_code`
path on this split and run core-off, transition-state-off, and action-code
shuffle/dropout ablations.

## 2026-05-05: Latent Action Codebook S120 Rejection

Trained the Stage 1 `transition_state_code` path on list-family holdout splits.
The objective was code CE only; answer/depth CE weights were set to zero.

Artifacts:

```text
v1 config:
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_s120.yaml

v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/last.pt

v1 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_s120_from_oodstress/eval_latent_action_codebook_list_holdout_transition_off.json

v2 config:
  configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_v2_s120.yaml

v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/last.pt

v2 eval:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_v2_s120_from_oodstress/eval_latent_action_codebook_v2_list_holdout_transition_off.json
```

Results:

```text
role_v1:
  full step_acc:                 0.7500
  transition-state-off step_acc: 0.2500
  trace exact:                   0/64

terminal_v2:
  full step_acc:                 0.5000
  transition-state-off step_acc: 0.2500
  trace exact:                   0/64
```

Failure pattern:

```text
role_v1:
  depth 1, 2, 8 correct
  depth 4 wrong: predicted final_compose, target hold

terminal_v2:
  depth 1, 8 correct
  depth 2 wrong: predicted nonterminal compose, target terminal compose
  depth 4 wrong: predicted final compose, target hold
```

Conclusion:
The code path is causal because full beats transition-state-off on step
accuracy, but Stage 1 is rejected because trace exact is still 0/64. The
remaining blocker is prompt-grounded termination and semantic transition
grounding for unseen families, not merely the number or names of latent action
codes.

## 2026-05-05: Transition-State Finality S120 Narrow Acceptance

Trained a finality-only transition-state head on the same list-family holdout
split. The run used finality BCE only; answer and code losses were disabled.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_finality_s120.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_transition_off.json
```

Results:

```text
full finality step_acc:                 0.7500
transition-state-off finality step_acc: 0.2500
trace exact:                            0/64
```

Decision:
Accept the finality head only as a narrow causal signal. Reject it as Stage 1
reasoning progress because trace exact is still 0/64 and action/semantic
transition reasoning remains unsolved.

## 2026-05-05: Transition-State Text S120 Rejection

Trained a transition-state text head on the same list-family holdout split.
This tests semantic intermediate-state token prediction without fixed operation
ids.

Artifacts:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_text_s120.yaml

eval script:
  scripts/231_eval_qtrm_transition_state_text.py

low-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_s120_from_oodstress/last.pt

high-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_lr1e3_s120_from_oodstress/last.pt
```

Results:

```text
lr=5e-5:
  full step_acc:                 0.0000
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64

lr=1e-3:
  full step_acc:                 0.2500
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64
```

Decision:
Accept only the narrow causal semantic-token signal from the high-lr run.
Reject recursive semantic transition because the model collapses to the depth-1
token at every depth and does not learn the depth-varying state update.

## 2026-05-05: Transition-State Text Depth-Contrast Rejection

Added a row-local depth-contrast loss to separate labelled intermediate-state
tokens across recurrent depths.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/eval_transition_text_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/eval_transition_text_list_holdout_transition_off.json
```

Results:

```text
full step_acc:                 0.2500
transition-state-off step_acc: 0.0000
trace exact:                   0/64
```

Decision:
Reject this as Stage 1 progress. The path is causal, but it still predicts the
same depth-1 content token at every depth. Do not keep stacking scalar losses
onto the frozen full-vocabulary projection. The next candidate is a compact
semantic state head or state embedding that the recurrent core must update
before any language-vocabulary readout.

## 2026-05-05: Code+Finality And Joint-State Rejection

Tested two compact alternatives after the full-vocabulary transition-text path
collapsed:

```text
code + finality checkpoint:
  local_eval/qwen35_2b_pure_recursive_latent_action_codebook_finality_s120_from_oodstress/last.pt

joint checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_s120_from_oodstress/last.pt
```

Results on the held-out list-family split:

```text
code + finality:
  full code step_acc:      0.7500
  full finality step_acc:  0.7500
  full halted exact:       0/64
  off code/finality acc:   0.2500 / 0.2500

joint:
  full code step_acc:      0.7500
  full finality step_acc:  0.7500
  full halted exact:       0/64
  off code/finality acc:   0.2500 / 0.2500
```

Decision:
Reject both as Stage 1 promotion. They prove a causal transition-state path,
but strict trace exact and halted exact remain 0/64. The joint head shows the
current bottleneck more clearly: it gets the list prefix right, then fires a
final/hold state at an unlabelled intermediate step. The next experiment should
use dense 1..N transition targets instead of sparse 1/2/4/8 labels.

## 2026-05-05: Dense Joint-State Rejection

Added dense 1..N transition targets and retrained compact joint-state heads for
both role_v1 and terminal_v2 codebooks.

Artifacts:

```text
builder:
  scripts/232_build_dense_transition_targets.py

decision:
  docs/wiki/decisions/transition-joint-dense-s120.md

role_v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/last.pt

terminal_v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/last.pt
```

Results:

```text
role_v1 dense:
  full step_acc:      0.8750
  off step_acc:       0.1250
  halted exact:       0/64

terminal_v2 dense:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64
```

Decision:
Reject as Stage 1 promotion. Dense targets prove the transition-state path is
causal, but held-out list still maps to the arithmetic/nonterminal compose path
at depth 2 and strict halted trace exact remains 0/64. The next bottleneck is
prompt-grounded terminal routing and semantic family transfer, not just sparse
transition labels.

## 2026-05-05: Terminality Fix And All-Families Sanity Control

Added terminality counterfactual data and an action-terminal finality mode for
dense targets.

Artifacts:

```text
builder:
  scripts/233_build_terminality_counterfactual_targets.py

decision:
  docs/wiki/decisions/transition-joint-terminality-and-sanity-s120.md
```

Results:

```text
terminality augmentation, list still fully held out:
  full step_acc:      0.7500
  off step_acc:       0.1250
  halted exact:       0/64

all-families sanity control, list included in train but held out by index:
  full step_acc:      1.0000
  off step_acc:       0.1250
  halted exact:       64/64
```

Decision:
Accept only the all-families mechanism sanity control. The recurrent
transition-state path can learn list traces and is causally necessary, but full
list-family zero-shot transfer remains rejected.

## 2026-05-05: Core Role-Value Transition Auxiliary Rejected

Added a disabled-by-default `core_role_value_transition_logits` path and
`--algorithmic-role-value-transition-ce-weight` to test whether previous
role-state to next-role-state supervision improves the mandatory latent loop.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-role-value-transition-aux-s120.md

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_transition_joint_s120.yaml

rejected checkpoints:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_from_len579_s240/last.pt
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_w01_from_len579_s240/last.pt
```

Results:

```text
canonical baseline:
  value:     184/624 = 0.2949
  step exact: 16/256 = 0.0625

transition aux weight 1.0:
  value:      96/624 = 0.1538
  step exact:  0/256

transition aux weight 0.1:
  value:      96/624 = 0.1538
  step exact:  0/256

baseline loaded with transition-enabled config:
  value:     184/624 = 0.2949
  step exact: 16/256 = 0.0625
```

Decision:
Reject the transition auxiliary as canonical. The module presence is safe, but
auxiliary training regresses held-out value state. Core-step sweep also shows
that current role-value slots are a structured probe/scaffold, not yet a
depth-improving raw recursive reasoning mechanism.

## 2026-05-05: List Paraphrase Transfer Gate Accepted

Built a fairer Stage 1 list transfer gate where list_transform is present in
train, but list paraphrase variants 6 and 7 are held out.

Artifacts:

```text
builder:
  scripts/234_build_list_transfer_gate.py

decision:
  docs/wiki/decisions/transition-joint-list-transfer-s120.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/last.pt
```

Results:

```text
full:
  exact:         16/16
  halted exact:  16/16

transition-state-off:
  exact:         0/16
  halted exact:  0/16

code shuffle:
  exact:         0/16
  halted exact:  0/16

code dropout:
  exact:         0/16
  halted exact:  0/16
```

Decision:
Accept as within-family list-surface transfer. Do not claim full family-zero-shot
reasoning; that remains rejected.

## 2026-05-05: Core State-Carry Role-Value Rejected

Added a gated recurrent carry update on the core role-value token slice plus a
`core_state_carry_only` trainable policy to separate architecture damage from
baseline-preserving adapter training.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-state-carry-role-value-s120.md

configs:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_carry_joint_s120.yaml
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_state_carry_only_s120.yaml

eval ablation:
  scripts/238_eval_qtrm_algorithmic_value_state.py --disable-core-state-carry
```

Results:

```text
canonical baseline:
  value:      184/624 = 0.2949
  step exact:  16/256 = 0.0625
  action:      32/32

core_and_role_value_state + carry:
  value:       80/624 = 0.1282
  step exact:   0/256

same checkpoint, carry disabled at eval:
  value:       80/624 = 0.1282
  step exact:   0/256

core_state_carry_only:
  value:      158/624 = 0.2532
  step exact:  16/256 = 0.0625
  action:      32/32
```

Decision:
Reject state-carry as canonical. Carry-only proves the safer baseline-preserving
workflow, but the module does not beat the held-out value-state baseline. The
next candidate should be a compact value-delta state rather than a free MLP
carry over role tokens.

## 2026-05-05: Core Role-Value Delta-Only Rejected

Added a small recurrent delta adapter over core role-value token trajectories,
plus `core_role_value_delta_only` training and a delta-off value eval ablation.

Artifacts:

```text
decision:
  docs/wiki/decisions/core-role-value-delta-only-s120.md

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_delta_only_s120.yaml

checkpoints:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_from_len579_s240/last.pt
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_lr1e5_from_len579_s240/last.pt
```

Results:

```text
baseline / untrained delta config:
  value:      184/624 = 0.2949
  step exact:  16/256

delta-only lr 1e-4:
  value:      112/624 = 0.1795
  step exact:   0/256
  delta off:  184/624 = 0.2949
  action:      32/32

delta-only lr 1e-5:
  value:      184/624 = 0.2949
  step exact:  16/256
```

Decision:
Reject as canonical. The adapter is isolated and baseline-preserving, but it
does not improve held-out value-state reasoning. The next candidate should make
the missing state variable explicit: action code -> typed value-delta code ->
value-state update -> role-value logits.

## 2026-05-05: Breakthrough-Prior Architecture Pivot

The repeated failure of local carry/delta MLPs triggers the big-structure doubt
gate. The next candidate is no longer another continuous hidden-state adapter.

Artifacts:

```text
decision:
  docs/wiki/decisions/breakthrough-prior-next-architecture.md

downloaded papers:
  references/papers/recurrent_depth/loopformer_2602.11451.pdf
  references/papers/recurrent_depth/ouro_looplm_2510.25741.pdf
  references/papers/recurrent_depth/rltt_latent_thought_trajectory_2602.10520.pdf
  references/papers/recurrent_depth/looprpt_2603.19714.pdf
  references/papers/role_value_slots/discrete_neural_algorithmic_reasoning_icml2025.pdf
  references/papers/role_value_slots/transnar_transformers_meet_nar_2406.09308.pdf

official code:
  references/official/loopformer @ 59a8ae8
  references/official/dnar @ 12f3f0b
  references/official/clrs @ bfd042f
```

Prior-backed pivot:

```text
Discrete NAR-style typed execution bottleneck
+ LoopFormer/Ouro/RLTT-style trajectory credit
+ QTRM mandatory recursive core
```

Next falsifiable experiment:

```text
core_depth_state
-> value_delta_code_logits
-> straight-through one-hot value_delta_code
-> typed register update features
-> role-value logits
```

Implementation scaffold:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_value_delta_code_only_s120.yaml

model:
  core_value_delta_code_logits
  core_value_delta_code_gate_mean

training:
  trainable_param_policy: core_value_delta_code_only
  --core-value-delta-code-ce-weight

eval:
  --disable-core-value-delta-code
```

Acceptance:

```text
held-out value accuracy > 184/624
step exact             > 16/256
trace exact            > 0/32
action-code exact      = 32/32
delta-code ablation    drops below full
depth 8                beats depth 1 on held-out value/step metrics
```

## 2026-05-05: Core Value-Delta Code Only S120 Rejection

Decision:

```text
docs/wiki/decisions/core-value-delta-code-only-s120.md
```

Result:

```text
full trained path:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

code-off ablation:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

direct code-logit readout:
  value accuracy:  63/624 = 0.1009615385
  step exact:       0/256 = 0.0
  trace exact:      0/32

action-code:
  exact: 32/32
```

Decision:

```text
Reject as canonical. The finite code path does not carry held-out value-state
signal, and deeper recurrence does not improve value-state accuracy.
```

Next:

```text
Stop adding local hidden-state adapters for value execution. Rebuild the next
candidate as a TransNAR/CLRS-style typed register executor:

prompt -> role binder -> mandatory core -> operation selector ->
typed registers -> verifier-checked update -> role-value readout
```

## 2026-05-05: Typed Register Executor Only S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-executor-only-s120.md
```

Implemented the first persistent typed-register executor while preserving the
universal LLM causal path:

```text
prompt tokens
-> QTRM core trajectory
-> learned operation selector
-> persistent typed registers
-> role-value logits
```

Results:

```text
untrained typed-register:
  value accuracy:   0/624 = 0.0

trained full:
  value accuracy: 106/624 = 0.1698717949
  step exact:       0/256
  trace exact:      0/32

typed-register-off:
  value accuracy: 184/624 = 0.2948717949
  step exact:      16/256

action-code:
  exact: 32/32
```

Decision:

```text
Reject as canonical. The executor is causal but harmful: disabling it restores
the stronger baseline. Keep the scaffold, but the next candidate needs
operation/process supervision rather than value CE alone.
```

## 2026-05-05: Typed Register Executor V2 Process-Credit Rejection

Decision:

```text
docs/wiki/decisions/typed-register-executor-v2-process-credit-plan.md
```

Research-driven update:

```text
Use LoopLM/latent lookahead/process-credit prior to add training-only process
CE on the existing typed-register operation selector.
```

Implementation target:

```text
core_typed_register_operation_logits
+ transition_state_codes[depth] CE
+ role-value CE
```

Result:

```text
full value accuracy:       102/624 = 0.1634615385
typed-register-off:        184/624 = 0.2948717949
full step exact:             0/256
typed-register-off exact:   16/256
action-code exact:          32/32
```

Decision:

```text
Reject. Process-code CE preserves the action controller but the typed-register
value path remains harmful. Next candidate needs register-transition
consistency and latent-lookahead/process reward, not another isolated value
head.
```

## 2026-05-05: Typed Register Prompt Binder S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-prompt-binder-s120.md
```

Hypothesis tested:

```text
Maybe typed registers fail because they only see mean-pooled context and cannot
bind exact prompt values.
```

Result:

```text
baseline value accuracy:      184/624
prompt-binder full:           104/624
prompt-binder register-off:   168/624
action-code exact:             32/32
```

Decision:

```text
Reject. Token-addressed prompt access alone is not enough and also disturbs the
previous role-value baseline. The next candidate should make exact value update
a recurrent transition objective, not another readout/binder patch.
```

## 2026-05-05: Typed Register Transition Consistency S120 Rejection

Decision:

```text
docs/wiki/decisions/typed-register-transition-consistency-s120.md
```

Hypothesis:

```text
The typed-register path needs teacher-forced recurrent value-transition credit:
register_state[t] -> role_value_state[t+1].
```

Implementation:

```text
core_typed_register_transition_logits
--core-typed-register-transition-ce-weight
```

Result:

```text
full value accuracy:      104/624
typed-register-off:       184/624
action-code exact:         32/32
trace exact:                0/32
```

Decision:

```text
Reject. The transition head was auxiliary and did not become the evaluated
value path. Next candidate must make depth>1 value readout come from the
previous-register transition prediction itself.
```

## 2026-05-05: Typed Register Strict Transition Readout S120 Plan

Decision:

```text
docs/wiki/decisions/typed-register-strict-transition-readout-s120.md
```

Hypothesis:

```text
Depth > 1 role-value logits must be the transition prediction itself, not an
auxiliary head beside the evaluated value head.
```

Implementation:

```text
core_typed_register_transition_readout_enabled
```

Result:

```text
full value accuracy:      64/624 = 0.1025641026
typed-register-off:      184/624 = 0.2948717949
full step exact:           0/256
typed-register-off exact: 16/256
action-code exact:        32/32
```

Decision:

```text
Reject. Making the transition head the causal value readout worsens the value
path. Since typed-register-off restores the baseline exactly, the bottleneck is
not prompt access or auxiliary loss wiring; the typed-register value mechanism
itself is the wrong local patch. Next work should move back to the universal
LLM causal path and test a root latent-state candidate instead of another
parallel role-vocab head.
```

## 2026-05-05: Final Answer Bridge S120 Opened

Decision:

```text
docs/wiki/decisions/final-answer-bridge-s120.md
```

Failure:

```text
The best len579 value-state checkpoint scores 0/32 on mixed-composition
causal forced-choice for every tested mode. It ranks intermediate list states
above final scalar answers.
```

Next experiment:

```text
Train answer_state_loop LM logits with causal-prefix final-answer CE while
preserving transition_state_joint CE. No typed-register or role-value answer
channel is active.
```

Result:

```text
action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000

LM causal forced-choice smoke8:
  donor_only:   0/8
  core_off:     0/8
  core_steps_1: 0/8
  core_steps_8: 0/8
```

Decision:

```text
Reject. Final-answer CE lowers training loss and preserves the action policy,
but it does not transfer to held-out final-value answers. The next candidate
must put a neural value-state transition into the causal answer path; answer
rendering alone is not enough.
```

## 2026-05-05: Role-Value Answer Bridge S120 Opened

Decision:

```text
docs/wiki/decisions/role-value-answer-bridge-s120.md
```

Architecture change:

```text
core_role_value_state_logits
-> soft value embeddings + role embeddings
-> gated internal answer-loop tokens
-> answer_state_loop LM logits
```

Why:

```text
The previous final-answer bridge trained the renderer but did not make the
computed value state causal. This probe forces the answer loop to read the
core's role-value state through a learned, ablatable internal bottleneck.
```

Promotion gate:

```text
action-code exact remains 32/32
core_steps_8 LM forced-choice beats donor_only and core_off
role_value_answer_bridge_off causes a held-out drop
```

Result:

```text
S80 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_role_value_answer_bridge_len579_s080_from_len579_s240/last.pt

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000

LM causal forced-choice smoke8:
  donor_only:              0/8
  core_off:                0/8
  core_steps_8:            0/8
  core_steps_8 bridge_off: 0/8
```

Decision:

```text
Reject. The bridge trains and preserves action-code, but bridge-off does not
change held-out LM answer selection. Next candidate should move to an
Ouro/LoopLM-style recurrent answer-hidden-state block with depth allocation,
not another side-head bridge.
```

## 2026-05-06: Ouro Answer Recurrent S080 Opened

Decision:

```text
docs/wiki/decisions/ouro-answer-recurrent-s080.md
```

Architecture change:

```text
answer_state_loop cross-attention
-> shared causal recurrent answer block
-> LM logits
```

Why:

```text
Ouro/LoopLM is the better general-LLM prior than task-specific TRM. The next
probe updates the answer hidden state itself instead of feeding side-state
tokens into an otherwise unchanged answer loop.
```

Promotion gate:

```text
action-code exact remains 32/32
core_steps_8 beats donor_only/core_off on held-out LM causal forced-choice
answer_state_recurrent_off drops versus full
```

Result:

```text
S80 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s080_from_len579_s240/last.pt

training:
  final_path_ce: 10.9580 -> 2.6068
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       2/8
  core_steps_8 answer_recurrent_off:  0/8
```

Decision:

```text
Accept as a smoke-level causal gain. This is not a general reasoning claim yet,
but it is the first positive answer-path ablation in this branch. Continue to
S240/S480 before adding a SubQ/MSA-like selective context router.
```

S240 continuation result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_len579_s240_from_s080/last.pt

training:
  final_path_ce: 4.3030 -> 1.0415
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  donor_only:                         0/8
  core_off:                           0/8
  core_steps_8:                       0/8
  core_steps_8 answer_recurrent_off:  0/8
```

Decision update:

```text
Reject S240 continuation. CE-only continuation improves in-sample loss but
erases the S80 held-out answer-path gain. Keep S80 as the best observed
checkpoint and add validation-gated selection or stronger final-value/ranking
loss before longer training.
```

Implementation follow-up:

```text
Added --save-every to scripts/196_train_pure_recursive_depth_supervised.py.
Future recurrent-answer runs should save step snapshots and evaluate LM
causal forced-choice before overwriting the best observed answer-path state.
```

Choice-margin continuation result:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_choice_margin_len579_s080_from_s080/step_000080.pt

training:
  final_path_ce: 0.9753 on final logged batch
  final_path_acc: 1.0000
  transition_state_joint_acc: 1.0000

action-code heldout:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
  halted_exact: 32/32

LM causal forced-choice smoke8:
  step40 core_steps_8:                       0/8
  step40 core_steps_8 answer_recurrent_off:  0/8
  step80 core_steps_8:                       0/8
```

Decision update:

```text
Reject choice-margin continuation as a canonical recurrent-answer objective.
The implementation is useful infrastructure, but this run repeats the S240
failure: transition/action state remains perfect while held-out LM answer
selection loses the accepted S80 causal gain.
```

## 2026-05-06: Ablation-Aware Checkpoint Gate And Selective Router Probe

Added an evaluation artifact selector:

```text
scripts/240_select_qtrm_checkpoint_by_gate.py
```

It selects checkpoints by held-out LM causal forced-choice, action-code
preservation, and optional component ablation drop. This prevents accepting a
lower training CE or equal-score component that is not causally useful.

Selector result:

```text
S80 baseline:
  full:            2/8
  recurrent_off:   0/8
  action-code:     32/32
  accepted:        true

S240 CE-only:
  full:            0/8
  action-code:     32/32
  accepted:        false

choice-margin step80:
  full:            0/8
  action-code:     32/32
  accepted:        false
```

Implemented the minimal SubQ/SSA-style answer selective context router:

```text
answer hidden state
+ core/workspace states
+ prompt states
-> learned top-k selector
-> answer-state cross-attention
-> recurrent answer block
-> LM logits
```

Probe result:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_s080.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_len579_s020_from_s080/step_000020.pt

LM causal forced-choice:
  full:       2/8
  router_off: 2/8

action-code:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
```

Decision:

```text
Reject selective_s020 as a causal architecture gain. It preserves S80 but
router-off does not hurt. Keep the router code as a scaffold; future router
runs need dense-vs-sparse alignment or explicit router supervision before
claiming a SubQ/SSA-style improvement.
```

Decision doc:

```text
docs/wiki/decisions/ouro-selective-context-s020.md
```

Dense-alignment v2 update:

```text
change:
  added force_answer_state_loop_dense_context
  added --answer-selective-context-alignment-weight
  added answer_selective_context_alignment_loss

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_align_len579_s020_from_s080/step_000020.pt

training:
  final_path_ce: 4.3567 -> 2.0881
  answer_selective_context_alignment_kl: 0.0228 -> 0.0198

LM causal forced-choice:
  full:       2/8
  router_off: 2/8

action-code:
  exact:        32/32
  step_acc:      1.0000
  finality_acc:  1.0000
```

Decision:

```text
Reject dense-alignment S020. The KL teacher path is implemented, but router-off
still ties full mode, so the router has no causal answer-path contribution.
S80 Ouro recurrent remains the selected baseline.
```

## 2026-05-06 - Ouro Causal-Prefix Tail S020

Failure isolated:

```text
mixed-list arithmetic often computes the doubled sum but skips the final
subtract step, e.g. gold 300015 vs model 300024.
```

The hard negative was already present in `choices`, so the next falsifier used
causal-prefix multi-token supervision and sequence margin instead of another
dataset-only change.

Artifacts:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/last.pt

eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/lm_causal_forced_choice_smoke8_with_baselines.jsonl
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_prefix_tail_s020_from_mixedrepeat/action_code_eval32.json
```

Result:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   3/8
full core8:   4/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     3
  pre_subtract_sum:  4
  tie:               1
```

Decision:

```text
Accept as a smoke improvement, not a broad breakthrough.
The next bottleneck remains final subtract-tail retention.
```

Storage hygiene:

```text
Deleted generated local_eval/**/step_*.pt snapshots after / reached 100%.
Kept last.pt checkpoints and eval JSON artifacts.
```

## 2026-05-06 - All-Prefix Bridge Tail Reject

Implemented:

```text
--transition-joint-answer-bridge-contrast-all-prefix-tokens
```

Rationale:

```text
The accepted tail checkpoint improved full to 4/8, but bridge-off also reached
3/8. The hypothesis was that bridge contrast only touched the first answer
token, while the final subtract error lives in later answer digits.
```

Result:

```text
donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   2/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     2
  pre_subtract_sum:  4
  doubled_list:      2
```

Decision:

```text
Reject. All-prefix bridge contrast preserves the action controller but regresses
the answer path and eliminates the full-vs-bridge-off gap.
```

Decision doc:

```text
docs/wiki/decisions/ouro-allprefix-bridge-tail-s020-reject.md
```

## 2026-05-06 - Tail-Negative S020

Implemented a narrow preterminal-state negative objective:

```text
--tail-negative-margin-weight
--tail-negative-margin
--tail-negative-family-filter
```

Accepted smoke:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_s020_from_tail_s020/last.pt

donor_only:   0/8
core_off:     0/8
bridge_off:   2/8
full core8:   4/8
action-code: 32/32
```

Tail breakdown:

```text
full core8:
  correct_final:     4
  pre_subtract_sum:  4

bridge_off:
  correct_final:     2
  pre_subtract_sum:  6
```

Decision:

```text
Accept as causal-gap improvement only. It keeps full 4/8 and lowers bridge-off
from 3/8 to 2/8, but it does not reduce full pre-subtract failures.
```

Rejected follow-up:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_tailneg_mixedx4_s040_from_tail_s020/last.pt

mixedx4 S040:
  donor_only:   0/8
  core_off:     0/8
  bridge_off:   4/8
  full core8:   3/8
  action-code: 32/32
```

Decision doc:

```text
docs/wiki/decisions/ouro-tail-negative-s020.md
```

## 2026-05-06 - Ouro Final-Answer Binder Reject

Implemented an ablatable final-answer binder and tested two variants:

```text
v1: transition joint-code/finality distribution -> answer delta
v2: finality-weighted core_depth_state -> answer delta
```

Both were trained for S020 from the accepted tail-negative checkpoint and
evaluated with the same smoke8 causal forced-choice gate.

Results:

```text
v1:
  donor_only:        0/8
  core_off:          0/8
  full core8:        2/8
  binder_off:        2/8
  joint_bridge_off:  2/8

v2:
  donor_only:        0/8
  core_off:          0/8
  full core8:        2/8
  binder_off:        2/8
  joint_bridge_off:  2/8
```

Decision:

```text
Reject. The accepted baseline is still 4/8, and binder-off ties full. The next
architecture step should train latent trajectory quality directly, using
LoopLM/COCONUT/LoopFormer/LoopRPT-style process credit rather than adding
another answer-side readout patch.
```

Decision doc:

```text
docs/wiki/decisions/ouro-final-answer-binder-s020-reject.md
```

## 2026-05-06 - Ouro Answer Halt Head S080 Accepted

Implemented a PonderNet/ACT-style answer-state halt head and tested two
variants:

```text
rejected:
  transition finality -> in-loop answer freeze
  result: full 2/8, gate_off 4/8

accepted:
  answer hidden -> learned halt head
  train with halt gate disabled
  eval with hard-first in-loop halt gate
```

Key metrics:

```text
smoke8:
  core_steps4:       8/8
  core_steps8 full:  8/8
  halt_gate_off:     0/8
  bridge_off:        8/8

smoke16:
  core_steps4:       10/16
  core_steps8 full:  10/16
  halt_gate_off:      0/16
  bridge_off:        10/16

action-code:
  exact:             32/32
  step_acc:          1.0
  finality_acc:      1.0
  halted_exact:      32/32

generation smoke4:
  full/gate_off:      0/8
```

Decision:

```text
Accept as the current Ouro raw-recursive answer-path candidate. The answer
halt gate is causal because disabling it collapses the answer path to 0/16.
Demote transition_joint_answer_bridge for this checkpoint because bridge_off
ties full. Do not claim generation readiness: greedy generation still fails.
```

Decision doc:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Typed CE Demoted To Probe-Only

Scope correction:

```text
typed algorithmic CE is not a universal LLM objective.
```

It remains useful for diagnosing numeric/register binding, but canonical QTRM
claims must improve the single prompt-token-to-LM-logits path.

The current canonical raw-recursive answer-path baseline is now recorded as a
machine gate:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080-raw-gate.md

donor_only: 0/8
core_off:   0/8
depth1:     0/8
depth2:     0/8
depth4:     8/8
depth8:     8/8
```

Next bottleneck:

```text
greedy autoregressive rendering remains 0/8 while forced-choice is accepted.
Improve the answer-token path without typed answer channels or donor-logit
shortcuts.
```

## 2026-05-07 - Ouro Halt-Head Partial Scale32 Depth Gate

Attempted a full scale32 forced-choice sweep for the accepted halt-head
checkpoint. The full 8-mode sweep was stopped after 170/256 rows because it was
too slow for the current loop.

The completed depth4 subset was preserved as a partial gate:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080-depth4-scale32-partial-gate.md

donor_only: 0/32
core_off:   0/32
depth1:     4/32
depth2:     4/32
depth4:    16/32
```

Decision:

```text
accept as partial depth-scaling evidence only.
not accepted as full scale32 halt-off/depth8 verification.
```

## 2026-05-07 - Renderer Root Redesign Toward Latent Lookahead

Web-search-backed prior check:

```text
Latent Lookahead Training
PonderLM-2
Reasoning with Latent Tokens in Diffusion LMs
Parcae stable looped models
Autoregressive LMs as EBMs
Hidden Capacity for One-Step Text Generation
Draft/Verify/Improve and Weaver
```

New diagnostic:

```text
scripts/247_probe_qtrm_gold_token_ranks.py
```

Accepted halt-head rank probe, 4 cases:

```text
core_steps_4:
  first@1: 4/4
  all@1:   0/4
  all<=10: 0/4
  max-rank mean: 740

example answer 300015:
  ranks: [1, 14, 5, 5, 5, 4, 1222]
```

Donor-preserving core-forced generation smoke:

```text
hits: 0/48
```

Decision:

```text
Stop stacking LM-head/bridge/logit-scale patches.
Next candidate is same-prefix latent lookahead / future-token auxiliary:
the answer-state loop must learn the next K answer tokens from the same
prefix, while runtime output remains normal LM logits/autoregressive text.
```

Docs:

```text
docs/wiki/sources/latent-lookahead-renderer.md
docs/wiki/decisions/qtrm-renderer-root-redesign-latent-lookahead.md
```

## 2026-05-07 - Depth Router Heads Rejected, Trajectory Training Promoted

Result:

```text
fixed-depth oracle from core-forced readout:
  16/24

best final-state route head:
  12/24

best core_depth_states trajectory route head:
  12/24
```

Decision:

```text
Reject route-head-only as the depth selector path.
The core has useful fixed-depth behavior, but frozen core states do not expose
a reliable routing signal to a small classifier.
```

Implementation note:

```text
Added controller_signal_source=learned_core_trajectory so diagnostic heads can
read the full core_depth_states sequence.
```

Next candidate:

```text
LoopFormer-style variable trajectory / shortcut-consistency training for the
recursive core itself, with depth sweeps and core-off/delta-off ablations.
```

Decision/source docs:

```text
docs/wiki/decisions/donor-preserving-controller-next-method.md
docs/wiki/sources/adaptive-depth-test-time-compute.md
```

## 2026-05-06 - Ouro Renderer Probes Rejected

Tested three ways to turn the accepted answer-halt S080 forced-choice scorer
into a greedy autoregressive renderer:

```text
naive answer-loop CE:
  generation:           0/8
  causal forced-choice: 0/8
  decision:             reject, checkpoint deleted

zero-init LM adapter:
  generation:           0/8
  causal forced-choice: full 8/8, halt_gate_off 0/8
  decision:             reject as renderer, checkpoint deleted

greedy-margin adapter:
  generation:           0/8
  causal forced-choice: full 4/8, halt_gate_off 0/8
  sample:               "24434264752"
  decision:             reject, checkpoint deleted

donor-logit fusion sanity:
  donor_logits_scale=1.0, qtrm_logits_scale=1.0
  generation: 0/4
```

Conclusion:

```text
The halt-head checkpoint remains canonical for raw forced-choice reasoning.
Greedy rendering is a separate causal bottleneck. Output-only patches either
do nothing or damage the accepted gate. The next candidate needs
autoregressive rollout/prefix training while preserving the halt-head
forced-choice baseline.
```

Decision doc:

```text
docs/wiki/decisions/ouro-answer-halt-head-s080.md
```

## 2026-05-07 - Latent Lookahead Renderer Scaffold

Implemented the next renderer falsifier:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/247_probe_qtrm_gold_token_ranks.py
scripts/248_run_qtrm_ouro_future_token_lookahead_s040.sh
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_future_token_lookahead_s040.yaml
```

Decision:

```text
same-prefix future-token lookahead is auxiliary-only
runtime answer path remains outputs["logits"] autoregressive LM generation
future-token CE requires causal-prefix supervision to avoid answer leakage
rank probe now separates strict rank from unique_top1 to avoid all-zero tie false positives
```

Decision doc:

```text
docs/wiki/decisions/qtrm-renderer-root-redesign-latent-lookahead.md
```

## 2026-05-07 - Causal Talker S040 Reject

Implemented a canonical-path Talker renderer:

```text
QTRM answer-state hidden + latent trajectory summary
-> causal Talker block
-> LM head
-> autoregressive logits
```

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040.yaml
scripts/249_run_qtrm_ouro_causal_talker_s040.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_s040_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8: 0/8

causal forced-choice smoke4:
  donor/core_off/core4/core8/halt_off: 0/4
```

Decision:

```text
reject S040 as a checkpoint.
The code path is cleaner than aux lookahead, but answer-loop-only S040 training
does not create a usable LM-compatible renderer and regresses the accepted
forced-choice signal.
```

Decision doc:

```text
docs/wiki/decisions/ouro-causal-talker-s040-reject.md
```

## 2026-05-07 - Causal Talker-Only S080 Reject

Narrowed the previous Talker experiment by freezing the accepted halt-head
checkpoint and training only the new causal Talker parameters.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080.yaml
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_eval_gate.yaml
scripts/250_run_qtrm_ouro_causal_talker_only_s080.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_causal_talker_only_s080_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8/core8_halt_off/core8_talker_off: 0/8

causal forced-choice smoke4 with answer halt gate enabled:
  donor/core_off: 0/4
  core4: 4/4
  core8: 4/4
  core8_halt_off: 0/4
  core8_talker_off: 4/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The forced-choice signal is preserved by the answer halt gate, not caused by
the Talker, and greedy generation remains 0/8.
```

Decision doc:

```text
docs/wiki/decisions/ouro-causal-talker-only-s080-reject.md
```

## 2026-05-07 - Donor-Guided Adapter S060 Reject

Fixed the depth-supervised trainer so donor-preserving configs actually pass
donor logits into the QTRM forward path. Then tested a low-rank answer-state LM
adapter with Qwen as the renderer:

```text
final_logits = donor_logits + clamp(answer_halt_state_adapter_delta)
qtrm_logits_scale = 0.0
donor_logits_scale = 1.0
trainable = answer_state_loop_lm_adapter_only
```

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml
scripts/251_run_qtrm_ouro_donor_guided_adapter_s060.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core4/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core4/core8/delta_off/halt_off: 0/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The donor-logit path is now correctly trained, but the adapter reinforces
intermediate trace strings instead of final answers and destroys the accepted
halt-gated forced-choice signal.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-adapter-s060-reject.md
```

## 2026-05-07 - Donor-Guided Adapter Final-Only S060 Reject

Retested the donor-guided adapter after removing the staged-target
contamination risk. This run trained only final-answer tokens at depth 8:

```text
target_mode = final
depth_steps = 8
final_logits = donor_logits + clamp(answer_halt_state_adapter_delta)
qtrm_logits_scale = 0.0
donor_logits_scale = 1.0
trainable = answer_state_loop_lm_adapter_only
```

Artifacts:

```text
scripts/252_run_qtrm_ouro_donor_guided_adapter_final_s060.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_final_s060_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/delta_off/halt_off: 0/4

gold-token rank probe:
  accepted halt-head core8 first_unique@1: 4/4
  accepted halt-head core8 all<=10:        0/4
  hard-negative core8 all<=10:             1/4
  hard-negative donor/delta-off all<=10:   3/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
Final-only supervision removes the obvious trace-string contamination but
does not recover the accepted halt-gated forced-choice signal or open greedy
generation. The next target should be a token-local final-answer
discriminator/scorer with hard negatives, not another blind adapter sweep.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-adapter-final-s060-reject.md
```

## 2026-05-07 - Donor-Guided Hard-Negative S080 Reject

Tested the next donor-renderer falsifier with final-only causal-prefix
supervision plus explicit hard negatives:

```text
choice negatives from row.choices
preterminal trace negatives
final-answer +/- 1 numeric counterfactuals
```

Artifacts:

```text
scripts/253_run_qtrm_ouro_donor_guided_hardneg_s080.sh
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_hardneg_s080_from_halt_s080/
```

Result:

```text
generation smoke8:
  donor/core_off/core8/delta_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/delta_off/halt_off: 0/4
```

Representative core8 misses:

```text
300015 -> 100002
300015 -> 50001
400037 -> 100000
400037 -> 50004
```

Decision:

```text
reject as a promoted renderer checkpoint.
Hard negatives did not recover generation or forced-choice. The bottleneck is
not only final-answer-vs-trace discrimination; answer-state hidden is not
renderer-ready for autoregressive numeric generation under adapter-only tuning.
```

Decision doc:

```text
docs/wiki/decisions/ouro-donor-guided-hardneg-s080-reject.md
```

## 2026-05-07 - Ouro Next-Token Decoder S080 Reject

Added an in-loop tokenizer-aligned answer decoder before the shared LM head and
trained only that decoder from the accepted answer-halt checkpoint.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080.yaml
scripts/254_run_qtrm_ouro_next_token_decoder_s080.sh
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_s080_from_halt_s080/
```

Training signal at step 80:

```text
final_path_ce=3.3559
final_path_acc=0.4286
final_greedy_token_margin=0.4388
causal_prefix_self_rollout_examples=0
```

Gold-token rank probe:

```text
donor_only:    first_unique@1 0/4, all<=10 3/4
core_off:      first_unique@1 0/4, all<=10 4/4
core8 full:    first_unique@1 4/4, all<=10 2/4
decoder_off:   first_unique@1 4/4, all<=10 0/4
halt_gate_off: first_unique@1 0/4, all<=10 0/4
```

Smoke gates:

```text
generation smoke8:
  donor/core_off/core8/decoder_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/halt_off: 0/4
  decoder_off: 4/4
```

Decision:

```text
reject as a promoted renderer checkpoint.
The decoder improves some teacher-forced rank diagnostics, but the full
runtime path still fails generation and regresses the accepted forced-choice
signal. The next falsifier should add self-rollout causal-prefix correction
instead of another decoder-only sweep.
```

Decision doc:

```text
docs/wiki/decisions/ouro-next-token-decoder-s080-reject.md
```

## 2026-05-07 - Ouro Next-Token Decoder Self-Rollout S040 Reject

Tested generated-prefix correction for the in-loop tokenizer-aligned decoder.

Artifacts:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040.yaml
scripts/255_run_qtrm_ouro_next_token_decoder_selfrollout_s040.sh
/mnt/nvme1n1p2/qtrm-runs/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_next_token_decoder_selfrollout_s040_from_s080/
```

Training signal:

```text
step 1 self-rollout mismatch rate: 1.0000
around step 10 final_path_acc: 0.5000
final step 40 final_path_acc: 1.0000
final step 40 self-rollout mismatch rate: 0.0000
```

Gold-token rank probe:

```text
donor_only:    first_unique@1 0/4, all<=10 3/4
core_off:      first_unique@1 0/4, all<=10 4/4
core8 full:    first_unique@1 4/4, all<=10 2/4
decoder_off:   first_unique@1 4/4, all<=10 0/4
halt_gate_off: first_unique@1 0/4, all<=10 0/4
```

Smoke gates:

```text
generation smoke8:
  donor/core_off/core8/decoder_off/halt_off: 0/8

causal forced-choice smoke4:
  donor/core_off/core8/halt_off: 0/4
  decoder_off: 4/4
```

Representative full core8 failure:

```text
300015 -> 1600000
400037 -> 1600000
300032 -> 1600000
400051 -> 1600000
```

Decision:

```text
reject as a promoted renderer checkpoint.
Self-rollout is wired and train metrics move, but decoder-only continuation
does not repair held-out generation or sequence scoring. The next attempt
must train answer-state recurrent block, halt gate, and decoder jointly.
```

Decision doc:

```text
docs/wiki/decisions/ouro-next-token-decoder-selfrollout-s040-reject.md
```

## 2026-05-07 - TRM Canonical Core / Mythos Demotion

Architecture decision:

```text
TRM/QTRM recursive core = primary reasoning core
answer-state loop = readout/renderer/control
Mythos/OpenMythos recurrence = stability reference or rejected probe, not a
second answer-side reasoning core
```

Reason:

```text
The Mythos-style answer-loop joint decoder S040 experiment still has
generation smoke8 0/8, and forced-choice smoke4 shows core8 full 2/4 while
decoder_off reaches 4/4.
```

Decision docs:

```text
docs/wiki/decisions/trm-canonical-core-mythos-demotion.md
docs/wiki/decisions/ouro-answer-loop-joint-decoder-s040-reject.md
```

## 2026-05-07 - TRM Core ACT Runtime Scaffold

Implemented two official-TRM-aligned core mechanics behind config flags:

```text
core_halt_init_bias=-5.0
core_trm_no_grad_inner_cycles_enabled=true
core_halt_freeze_halted_state_enabled=true
core_halt_exploration_prob/core_halt_exploration_min_steps
QTRMCoreCarry continuation API
```

Verification:

```text
halt-head conservative init test: OK
TRM no-grad inner-cycle test: OK
outer torch.no_grad preservation test: OK
per-sequence halt freeze test: OK
detached carry continuation tests: OK
model forward carry reuse test: OK
halt exploration train-only delay test: OK
```

Decision doc:

```text
docs/wiki/decisions/trm-core-act-runtime-scaffold.md
```

## 2026-05-07 - TRM Halt Q-Value Loss

Implemented optional q-value supervision for the core halt head:

```text
core_halt_loss_mode=q_value
core_halt_q_value_gamma
```

Meaning:

```text
q_halt     -> value of stopping at the current recurrent depth
q_continue -> discounted value of continuing one more depth step
runtime    -> halt when q_halt > q_continue
```

This closes the first Q-style halt/continue training gap. It is still only a
training mechanism until a held-out depth/ACT gate proves raw-reasoning gain.

## 2026-05-07 - Core Carry Eval Harness

Added raw-intelligence eval mode:

```text
qtrm_core_halt_carry_steps_N_no_evidence
```

This mode keeps retrieval/MemoryOS off, enables core halt, and reuses
`core_carry` across token/prefix forwards in causal forced-choice and
generation scoring. It is the first runtime harness for testing whether QTRM's
latent recurrent state continuation helps raw reasoning beyond fixed-depth
recomputation.

Status:

```text
harness wired: yes
promotion: pending held-out carry > no-carry/core-off/donor gate
```

## 2026-05-07 - Core Carry ACT Smoke4 Result

Ran the accepted S080 checkpoint with the correct eval-gate config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
```

Result:

```text
donor_only:                    0/4
core_off:                      0/4
core_steps8:                   4/4
answer_halt_gate_off:          0/4
core_halt_carry_steps8:        2/4
core_steps4:                   4/4
core_halt_carry_steps4:        4/4
```

Decision:

```text
carry harness is valid, but depth8 carry is rejected for this fixed-depth-4
smoke because it drifts toward an intermediate answer. Depth4 carry preserves
the accepted raw-recursive behavior.
```

Decision doc:

```text
docs/wiki/decisions/core-carry-act-smoke4.md
```

## 2026-05-07 - Core Carry Mixed-Depth ACT Gate

Ran mixed-depth causal forced-choice gate on:

```text
data/eval/pure_recursive_reasoning_heldout_72.jsonl
```

Smoke8 result:

```text
donor_only:                 2/8
core_off:                   0/8
core_steps2:                3/8
core_steps4:                3/8
core_steps8:                2/8
core_halt_carry_steps2:     3/8
core_halt_carry_steps4:     3/8
core_halt_carry_steps8:     2/8
answer_halt_gate_off:       2/8
```

Decision:

```text
reject S080 as a promoted mixed-depth ACT checkpoint. The carry harness is
useful diagnostically, but no carry mode beats the best fixed-depth modes at
smoke8.
```

Runner:

```text
scripts/257_run_core_carry_mixed_depth_act_gate.sh
```

Decision doc:

```text
docs/wiki/decisions/core-carry-mixed-depth-act-gate.md
```

## 2026-05-07 - Mixed-Depth ACT S160 Training

Fixed the depth-supervised training path for `pure_recursive_reasoning_*` rows:

```text
answer_aliases[0] is accepted as canonical answer
terminal depth can be inferred from depth_targets matching the final answer
staged choice-margin excludes the current staged answer from rejects
eval telemetry records answer-state-loop halt logits
```

Results:

```text
answer_loop_only_s160:
  core_steps8:          3/8
  halt_gate_off steps8: 3/8
  decision: reject as ACT

core_joint_ce_s160:
  core_steps8:          4/8
  halt_gate_off steps8: 4/8
  decision: partial fixed-depth raw-core improvement, reject as ACT
```

Decision doc:

```text
docs/wiki/decisions/mixed-depth-act-s160-results.md
```

## 2026-05-07 - List Transform Process S096 Reject

Added a list-transform failure summarizer and ran a list-focused process-state
experiment.

Failure ledger on the core-joint CE S160 checkpoint:

```text
list_transform qtrm_core_steps8: 0/2
errors:
  filtered_state_selected: 1
  reversed_final_selected: 1
```

Process-state run:

```text
init: core_joint_ce_s160
steps: 96
family_repeat: list_transform=6
staged_internal_sequence_ce_weight: 0.45
policy: core_and_answer_state_loop
```

Result:

```text
core_steps8:       3/8
list_transform:    0/2
decision: reject
```

The rejected checkpoint was deleted; eval JSON and ledgers were preserved.

Decision docs:

```text
docs/wiki/decisions/list-transform-failure-ledger-smoke8.md
docs/wiki/decisions/list-transform-process-s096-reject.md
```
## [2026-05-07] experiment | Reverse mixed composition prompt-context ledger

Added
`docs/wiki/decisions/transition-joint-reverse-composition-prompt-context.md`.

The accepted list-to-arithmetic checkpoint fails the reverse
arithmetic-to-list order by emitting the old fixed action code sequence.
Head-only S240 and prompt-context S240 both reject at `32/64`, while
prompt-context repeat4 S1000 improves reverse step accuracy to `0.6914` but
still has reverse exact `0/32`. The active gate remains exact trace transfer,
not partial step accuracy.
## 2026-05-07 General LLM Bottleneck Roadmap

- Added `docs/wiki/decisions/general-llm-bottleneck-roadmap.md`.
- Decision: reverse-composition accept would clear only the first major
  bottleneck: prompt-conditioned latent operation order.
- Remaining gates are now tracked as 10 bottlenecks: recursive depth/halting,
  value binding, latent-to-text rendering, donor override, trainable memory,
  reasoning-memory composition, metacognition, context routing, and
  agentic/multimodal grounding.

## 2026-05-07 Reverse Composition Token-Attention Reject

- Checkpoint:
  `local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_promptctx_tokattn_aug_joint_only_s5500_from_len1113`.
- Gate:
  `data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_eval82000_v6to7_len1113_aug.jsonl`.
- Result: `26/64` overall.
- Family split: `mixed_arithmetic_list=26/32`,
  `mixed_list_arithmetic=0/32`.
- Decision: reject and delete checkpoint weights. Prompt token-attention alone
  made the reverse family better than the original baseline but catastrophically
  regressed the old-order policy. Next candidate should factorize operation
  order/phase rather than adding more prompt-attention capacity to the same
  joint code head.

## 2026-05-08 Reverse Composition Source-Router Rejects

Updated
`docs/wiki/decisions/transition-joint-reverse-composition-prompt-context.md`
with primitive/source-router experiments.

Results on the 128-case reverse mixed-composition holdout:

```text
joint on primitive checkpoint:       113/128
primitive source only:                60/128
mean source router S500:             113/128
token-attention source router S800:   88/128
mean+token router S800 aug lengths:   87/128
```

Decision: reject source-router promotion. Primitive operation factorization is
useful for reverse order, but routing between joint and primitive policies is
not robust; token attention learns a length/OOD shortcut and sends len13 old
order to primitive. Next candidate should fold operation factorization into
the canonical joint transition policy rather than using a separate source
classifier.

## 2026-05-08 Reverse Composition Operation-Residual Best Partial

Added an internal operation residual:

```text
primitive operation probabilities -> zero-init residual -> joint transition logits
```

Best result so far:

```text
checkpoint:
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_reverse_joint_opres_scale05_joint_s0400_from_opres/last.pt

eval config scale: 0.45
holdout: 128 cases, eval lengths 11/13
overall: 123/128
reverse arithmetic_to_list: 62/64
old list_to_arithmetic: 61/64
```

Decision: best partial, still reject. This is better than source routing
because the improvement stays inside the canonical joint transition path.
Remaining failures require phase/order-state pressure rather than path
selection.

Follow-up hard-range fine-tune at train start index `174000` rejected:

```text
overall: 102/128
reverse arithmetic_to_list: 58/64
old list_to_arithmetic: 44/64
```

The rejected checkpoint was deleted. Broader value-range exposure alone is not
the next lever.

Transition-state code residual also rejected:

```text
overall: 64/128
reverse arithmetic_to_list: 0/64
old list_to_arithmetic: 64/64
```

The checkpoint was deleted and the default experiment config now keeps
`transition_state_code_enabled=false` and
`transition_state_joint_code_residual_enabled=false`. A plain state-code
auxiliary path collapses back to the old canonical order.

Joint order-contrast S200 also rejected:

```text
overall: 119/128
reverse arithmetic_to_list: 62/64
old list_to_arithmetic: 57/64
```

The checkpoint was deleted. The best active partial remains the operation
residual checkpoint at eval scale `0.45`: `123/128`.
## 2026-05-09 Source-Pointer L2 Local Acceptance

Fixed two target/architecture blockers in the QTRM source-position pointer
gate:

- `role_value_list_class_mode` is now applied to every loaded row, so
  `source_position` is not silently overridden by row metadata.
- `list_transform` single-number states such as `"8"` are interpreted as
  one-element list states, not scalar arithmetic states.
- The accepted gate uses 12 role/value state roles, giving five raw source
  slots, five doubled slots, and two scalar slots for length-5 list data.

Accepted local gate:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_l2_source_pointer_roles12_targetfix_s120
checkpoint: accepted_l2_source_pointer_roles12_step_000040.pt
full trace exact: 23/128 = 0.1796875
full value accuracy: 0.5093123209169055
primitive-off value accuracy: 0.0
source-binder-off value accuracy: 0.03939828080229226
decision: accepted_l2
```

Scope: L2 local state/probe acceptance only. This is not L3/L4 and does not
yet prove canonical LM answer rendering or token-numeric causal dependence.

## 2026-05-09 Source-Pointer Primitive-Core L3 Acceptance

Added an L3 hard perturbation gate and found a real state-codec bug: list
targets did not supervise padded/null roles, so cardinality was weakly
specified. Hard rows now use `role_value_supervise_null_slots=true`, which
labels padded source-pointer list roles as class `0`.

Accepted primitive-core L3:

```text
run: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/qtrm_l3_source_pointer_roles12_null_tune_s200
checkpoint: train/accepted_l3_primitive_core_null_step_000050.pt
report: manual_l3_null_decision_step_000050.json
audit report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l3_source_pointer_roles12_null_step50_l3_audit/report.json
```

Key metrics:

```text
full trace exact: 0.2890625
full value accuracy: 0.63125
primitive-off value accuracy: 0.0
primitive value drop: 0.63125
variant value accuracy:
  duplicate_even_binding: 1.0
  fifth_position_single_even: 0.4
  range_shift_v32to63: 0.58125
  surface_paraphrase: 0.54375
```

Important caveat: strict source-binder causality is still rejected
(`source_binder_value_drop = 0.115625`). Treat the source binder as auxiliary;
the canonical L3 claim is the primitive recurrent core plus null-slot state
codec.

## 2026-05-09 L4 Canonical LM Path Candidate Rejected

Added the first post-L3 LM-path runner:

```text
config: configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml
runner: scripts/322_run_source_pointer_l4_lm_path_gate.py
report:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
  qtrm_l4_source_pointer_lm_path_s080/report.json
```

Result:

```text
decision: rejected_l4_candidate
full greedy generation: 3/32 = 0.09375
donor-only: 5/32 = 0.15625
core-off: 5/32 = 0.15625
primitive-off: 3/32 = 0.09375
bridge-off: 4/32 = 0.125
```

Follow-up diagnostics:

```text
causal forced-choice:
  full = donor = core-off = primitive-off = bridge-off = 10/32

donor scale sweep:
  full default: 3/32
  donor_scale_0.25: 0/32
  qtrm_scale_1.0 donor_scale_0.25: 0/32
  qtrm_only: 0/32
```

Conclusion: L4 is still open. The accepted L3 primitive state exists, but the
current answer bridge/LM adapter does not causally render that state into token
logits. Next work must focus on a better state-to-token renderer, not MemoryOS,
RAG, MSA, or larger-scale training.

## 2026-05-09 L4 Bridge Family Rejected

Ran three post-L4 follow-ups against the accepted L3 source-pointer checkpoint:

```text
primitive-only bridge S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_primitive_lm_path_s020/report.json
  full: 3/16
  donor/core-off: 2/16
  primitive-off/bridge-off/final-binder-off: 3/16
  decision: rejected_l4_candidate

forced bridge S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_forced_bridge_s020/report.json
  full: 3/12
  donor/core-off: 2/12
  primitive-off/bridge-off/final-binder-off: 3/12
  decision: rejected_l4_candidate

forced bridge + adapter-only bottleneck S020:
  report: /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
          qtrm_l4_source_pointer_bridge_adapter_only_s020/report.json
  full: 2/12
  donor/core-off: 2/12
  primitive-off/bridge-off/final-binder-off: 2/12
  decision: rejected_l4_candidate
```

Do not count the 5-step adapter-only smoke as L4 acceptance; it used relaxed
negative thresholds and all causal margins were zero.

Code changes:

```text
configs/qwen35_2b_4090_source_pointer_l4_forced_bridge_roles12_s080.yaml
scripts/322_run_source_pointer_l4_lm_path_gate.py
src/qtrm_mm/training/train.py
tests/test_training_checkpoint_init.py
```

Validation:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_training_checkpoint_init tests.test_raw_intelligence_eval_script

116 tests OK
```

Conclusion: the current bridge-token cross-attention renderer family is the
wrong L4 path. Some runs perturb generation above donor/core-off, but the
primitive/bridge/final-binder ablations do not drop, so the accepted L3 state is
not the causal source. Next candidate should be a state-to-vocab pointer/copy
residual that feeds the canonical LM logits and has its own renderer-off
ablation.

## 2026-05-09 L4 State-to-Vocab Renderer Diagnostics

Implemented and tested a direct `core_role_value_state_vocab_renderer` path:

```text
canonical input tokens
-> donor hidden states / QTRM core
-> role-value answer bridge tokens
-> direct state-to-vocab residual logits
-> donor-fused LM logits
-> autoregressive generation
```

Code changes:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
src/qtrm_mm/training/train.py
scripts/196_train_pure_recursive_depth_supervised.py
scripts/322_run_source_pointer_l4_lm_path_gate.py
tests/test_core_halting.py
tests/test_pure_recursive_depth_supervised_train_script.py
tests/test_raw_intelligence_eval_script.py
```

Key fixes:

```text
1. Added direct renderer-off eval mode.
2. Added renderer-only trainable policy.
3. Added direct renderer CE / greedy-margin loss.
4. Removed unnecessary core-depth vocab logits from final-path-only training to avoid OOM.
5. Added primitive-off renderer contrast loss.
6. Added optional primitive-operation tokens to renderer cross-attention memory.
7. Added generation completion-delta reporting to the L4 runner.
```

Representative reports:

```text
S005/S010 direct renderer:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_s005/report.json
    qtrm_l4_source_pointer_vocab_renderer_lr1e3_s010/report.json
  result: rejected; full == donor == core-off == renderer-off

direct CE S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_s008/report.json
  result: rejected; full accuracy ties donor, but full generations differ
          from bridge-off/renderer-off on several cases.

direct CE + primitive contrast S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_primitive_contrast_s008/report.json
  result: rejected; primitive-off still does not drop in accuracy.

focused S024:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_source_pointer_vocab_renderer_direct_ce_primitive_contrast_s024_focus/report.json
  result: rejected; full accuracy drops to 0/4, indicating prompt/source-copy collapse.

operation-context S008:
  /mnt/nvme1n1p2/qtrm-runs/research_gate_runner/
    qtrm_l4_vocab_renderer_opctx_direct_ce_s008/report.json
  result: rejected; operation tokens did not change the held-out generation pattern.
```

Current root cause:

```text
The renderer is now causal enough to alter generated strings, but it is not
yet a correct reasoning renderer. It tends to copy source numbers or partial
intermediate values instead of rendering the final transformed answer.
```

Important warning:

```text
The L3 checkpoint chain still prints:
  [init] skipped shape-mismatched keys: core_role_value_state_embed.weight

The older base checkpoint used a 10-role embed while the current source-pointer
configs use 12 roles. L4 runs train/save a fresh 12-role embed, but the accepted
L3 baseline should not be treated as a fully clean full-state checkpoint.
```

Conclusion: L4 is still open. The next architecture should not merely make the
renderer stronger; it must make the final LM logits depend on the primitive
recurrent state in a way that renders transformed values, not source-copy
tokens.

## 2026-05-10 Source-Copy Renderer Role-Mask Repair

Found and fixed a contract mismatch between the source-copy alignment probe and
the actual LM renderer. The probe scored answer roles `0..3`, but the renderer
was preserving the later doubled-role block and masking the answer roles to
NULL. This made an alignment success look stronger than what generation really
used.

Changes:

```text
src/qtrm_mm/qtrm_model.py
tests/test_source_pointer_l4_lm_path_gate.py
```

The corrected test now requires the renderer to preserve answer roles `0..3`
and null out non-answer roles. Targeted tests passed.

Corrected source-copy alignment:

```text
S002:
  row_content_exact 83/128, rejected

S002 source-binder-off:
  row_content_exact 0/128, causal drop confirmed

S040 final-state CE:
  row_content_exact 128/128, accepted as L2 source-copy alignment diagnostic

S020 staged CE:
  row_content_exact 84/128, rejected
```

Generation still rejects L4:

```text
S040 after mask fix, smoke8:
  donor/core_off/vocab_renderer_off/source_binder_off: 5/8
  primitive_off: 4/8
  full: 3/8

S003 renderer-only continuation from S040:
  same smoke8 result, full 3/8
  failed checkpoint deleted; eval JSONL preserved
```

Gold-token rank probe shows the correct tokens are not hopelessly far away:

```text
S040 maskfix smoke8:
  donor/core_off/vocab_renderer_off all<=10: 8/8
  full all<=10: 8/8
  full all@1: 2/8, donor all@1: 3/8
```

Conclusion: exact source-position state is achievable, but the answer-role
decoder/query still fails to render that state into autoregressive LM tokens.
The next candidate should be a pointer-generator style decoder/cursor pressure
on the canonical LM path, not more MemoryOS/RAG/MSA work.

## 2026-05-10 Source-Copy Strict Scoring Repair

Found an evaluation weakness in `scripts/192_eval_raw_intelligence.py`: generation
hits were based on `normalized_contains`. For source-copy lexicalization this
can mark an overlong answer as correct, for example a completion that contains
the gold comma list as a substring but includes an extra copied source value.

Fix:

```text
source_copy_lexicalization now requires:
  exact_match or normalized_exact

general QA keeps the previous contains-based scoring.
```

Strict-rescored S040 maskfix smoke8:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_maskfix_smoke8/eval.strict.jsonl

donor/core_off/vocab_renderer_off/source_binder_off: 5/8
primitive_off: 4/8
full: 2/8
```

This makes the L4 rejection stronger. The previous full 3/8 included one loose
substring hit.

## 2026-05-10 Source-Copy Cursor Bias Rejected

Implemented an optional visible-prefix cursor bias for the source-copy renderer:

```text
config fields:
  core_role_value_state_vocab_renderer_source_copy_cursor_enabled
  core_role_value_state_vocab_renderer_source_copy_cursor_bias

code:
  QTRMMultimodalModel._compute_source_copy_cursor_role_bias
```

The idea was prior-backed by pointer-generator/copy-attention cursor or coverage
bias: use the visible generated prefix only to decide which answer role should
be read next, while still relying on the learned source-position state for the
actual copied value.

Smoke result on S040:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.jsonl

donor/core_off/vocab_renderer_off/source_binder_off: 5/8
primitive_off: 2/8
full: 1/8
```

Strict rescoring keeps the same full result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_cursor_smoke8/eval.strict.jsonl

full: 1/8
donor/core_off/vocab_renderer_off/source_binder_off: 5/8
```

Decision: reject as canonical default. The cursor over-biases copy tokens and
worsens autoregressive generation. The optional code path remains for controlled
ablation, but `configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml`
keeps it disabled.

## 2026-05-10 Source-Copy Span Lexicalization Repair

Root cause narrowed further: source-copy was keeping only the first tokenizer
piece for each source slot. Under Qwen tokenization, values such as `44`,
`40`, and `32` are multi-token spans:

```text
44,40,32 -> ["4", "4", ",", "4", "0", ",", "3", "2"]
```

Implemented the canonical-path span repair:

```text
src/qtrm_mm/algorithmic_value_state.py
  token_numeric_source_slot_token_spans(...)
  preserves all tokenizer pieces per compact source slot.

src/qtrm_mm/qtrm_model.py
  token_numeric_source_slot_token_span_ids / mask forward inputs
  _compute_source_copy_span_next_token_ids(...)
  source-copy renderer can continue the next token piece of the selected
  source span instead of copying only the first piece.

configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml
  core_role_value_state_vocab_renderer_source_copy_span_enabled: true
```

This is still not a hidden answer channel: source positions must come from the
QTRM source-position state, and the result still enters LM logits through the
autoregressive renderer.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

108 tests OK
```

Partial smoke on S040 with span-copy enabled:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_spancopy_smoke8/eval.jsonl

completed before manual stop:
  donor_only: 5/8
  core_off: 5/8
  full QTRM: 3/8
  vocab_renderer_off: 3/3 partial
```

Decision: do not promote. Span-copy fixed one lexicalization failure
(`full 2/8 -> 3/8` versus strict S040), but full QTRM still loses to
donor/core-off. The remaining bottleneck is answer-role/order selection during
generation: the model can emit complete source spans more faithfully, but it
still chooses extra or wrong source slots in several cases.

## 2026-05-10 Source-Copy Answer-Role Cursor Candidate

Implemented a second, narrower pointer-generator style cursor after the
span-copy repair. The rejected visible-prefix cursor counted source token ids
and failed on multi-token/duplicate source values. The new cursor counts
completed source spans beyond the prompt source list and uses separators only
to decide which answer role should be read next.

Code:

```text
src/qtrm_mm/config.py
  core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled
  core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias
  core_role_value_state_vocab_renderer_source_copy_answer_role_separator_token_ids

src/qtrm_mm/qtrm_model.py
  _compute_source_copy_answer_role_cursor_bias(...)
```

Canonical boundary:

```text
Allowed:
  prompt token spans + visible generated prefix choose the active answer role.

Still required:
  QTRM source-position state chooses which source slot that role copies.

Not allowed:
  computing the final answer as a hidden side channel.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

111 tests OK
```

S040 smoke with role cursor enabled:

```text
full QTRM:
  first4: 4/4
  tail4 was split because full 8-case process exited early after 4 rows:
    eval_tail4 first2: 2/2
    eval_tail2 remaining2: 2/2
  combined smoke8: 8/8

previous span-only S040:
  full: 3/8
  donor/core_off: 5/8

tail4 ablations on the harder second source group:
  renderer_off: 1/4
  source_binder_off: 1/4
  primitive_off tail2: 0/2
```

Artifacts:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_generation_state_ce_s040_rolecursor_smoke8/
```

Decision: promising L4 candidate, not promotion yet. It passes the local
smoke pattern and shows causal drops on the harder tail subset, but the full
held-out gate must still run in one reproducible command over a broader split
with donor/core_off/renderer_off/source_binder_off/primitive_off on the same
cases.

## 2026-05-10 Source-Copy Rolecursor L4 Smoke8 Accepted Candidate

The rolecursor L4 gate was made reproducible with a chunked/resumable runner:

```text
scripts/329_run_source_copy_rolecursor_l4_eval.py
```

Reason for the runner: repeated full generation evals can terminate or compete
for GPU with stale sessions. The runner executes each mode/chunk as an isolated
process, reuses complete chunks with `--resume`, and writes the full command log
to `report.json` while keeping stdout compact.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_source_copy_rolecursor_l4_eval \
  tests.test_prompt_source_position_binder_probe \
  tests.test_source_pointer_l4_lm_path_gate \
  tests.test_raw_intelligence_eval_script \
  tests.test_qtrm_source_copy_alignment_probe \
  tests.test_model_config

115 tests OK
```

Accepted smoke8 result:

```text
report:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_rolecursor_l4_eval_smoke8/report.json

full QTRM:        8/8 = 1.000
donor-only:       5/8 = 0.625
core-off:         5/8 = 0.625
primitive-off:    3/8 = 0.375
source-slot-off:  5/8 = 0.625
source-binder-off:5/8 = 0.625
vocab-renderer-off:5/8 = 0.625
```

Interpretation:

```text
full - donor/core_off/source_binder_off/renderer_off = +0.375
full - primitive_off = +0.625
```

This is the first clean L4 source-copy LM-path smoke acceptance for the
rolecursor path: the recursive/core path, source binding, and vocab renderer
are all causally needed on the same eight cases, and the final answer is
emitted through autoregressive text generation.

Boundary:

```text
Accepted: smoke8 L4 candidate.
Not accepted yet: broad 16/32/128 case L4 promotion or general LM reasoning.
```

Next action: run the same resumable gate on 16/32/128 cases, then add
mixed-family cases where the answer cannot be solved by source-copy alone.

## 2026-05-10 Source-Copy Rolecursor L4 Smoke16 Accepted Candidate

The same resumable L4 gate was expanded from 8 to 16 held-out source-copy
lexicalization cases.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 16 \
  --chunk-size 2 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_smoke16 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke16/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:         15/16 = 0.9375
donor-only:         8/16 = 0.5000
core-off:           8/16 = 0.5000
primitive-off:      6/16 = 0.3750
source-slot-off:    8/16 = 0.5000
source-binder-off:  8/16 = 0.5000
vocab-renderer-off: 8/16 = 0.5000
```

Interpretation:

```text
full - donor/core_off/source_slot/source_binder/renderer_off = +0.4375
full - primitive_off = +0.5625
```

This strengthens the smoke8 result. The rolecursor source-copy path is now a
repeatable 16-case L4 candidate where the canonical LM output depends on the
recursive primitive state, source binding, source slots, and vocab renderer.

Boundary remains unchanged:

```text
Accepted: source-copy lexicalization L4 candidate at 16-case smoke scale.
Not accepted yet: 32/128 broad gate, mixed-family general reasoning, or
general-purpose LLM promotion.
```

## 2026-05-10 Source-Copy Rolecursor L4 Smoke32 Accepted Candidate

The same gate was expanded to 32 cases.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 32 \
  --chunk-size 2 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_smoke32 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_smoke32/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:         27/32 = 0.84375
donor-only:        14/32 = 0.43750
core-off:          14/32 = 0.43750
primitive-off:     12/32 = 0.37500
source-slot-off:   14/32 = 0.43750
source-binder-off: 14/32 = 0.43750
vocab-renderer-off:14/32 = 0.43750
```

Interpretation:

```text
full - donor/core_off/source_slot/source_binder/renderer_off = +0.40625
full - primitive_off = +0.46875
```

The 32-case run confirms the source-copy rolecursor path is not just an 8/16
case artifact. Accuracy drops as cases broaden, but the causal margin remains
large and all required ablations collapse to donor/core-off scale.

Updated boundary:

```text
Accepted: reproducible 32-case source-copy L4 candidate.
Still not accepted: 128-case standard source-copy gate or mixed-family
general reasoning.
```

## 2026-05-10 Source-Copy Rolecursor L4 Standard128 Accepted

The full 128-case source-copy lexicalization gate passed.

Command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/329_run_source_copy_rolecursor_l4_eval.py \
  --max-cases 128 \
  --chunk-size 4 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_source_copy_rolecursor_l4_eval_standard128 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_source_copy_rolecursor_l4_eval_standard128/report.json
```

Result:

```text
decision: accepted_l4_candidate

full QTRM:          100/128 = 0.78125
donor-only:          44/128 = 0.34375
core-off:            44/128 = 0.34375
primitive-off:       37/128 = 0.28906
source-slot-off:     41/128 = 0.32031
source-binder-off:   41/128 = 0.32031
vocab-renderer-off:  44/128 = 0.34375
```

Interpretation:

```text
full - donor/core_off/renderer_off = +0.43750
full - primitive_off = +0.49219
full - source_slot/source_binder_off = +0.46094
```

This completes the source-copy lexicalization L4 standard gate. The canonical
LM path emits the answer autoregressively, and the gain disappears when the
recursive primitive state, source slots, source binder, or vocab renderer are
ablated.

Boundary:

```text
Accepted: source-copy lexicalization L4 standard128.
Still not accepted: broad/general LM promotion.
Next required gate: mixed-family/non-copy reasoning where the answer cannot be
obtained by lexical source-copy.
```

## 2026-05-10 Source-Copy Pointer L2 Accepted

A contract mismatch was found in the source-copy trainer: the probe/renderer
reads final answer roles `4..7`, but source-copy CE was still aimed at the
early/raw roles. The trainer now has
`--core-role-value-source-copy-answer-role-targets` to supervise the same answer
role block that inference reads.

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qtrm_source_pointer_batch_trainer \
  tests.test_source_position_logits_probe \
  tests.test_source_pointer_l4_lm_path_gate

54 tests OK
```

Accepted source-copy probe:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  qtrm_source_copy_pointer_batch_s020_b2_answerroles_from_s060/train/last.pt

eval128:
  full copy answer accuracy: 1.000
  source-slot-off copy answer accuracy: 0.000
  source-binder-off copy answer accuracy: 0.000
```

Decision: accept L2 source-position/copy-logits prerequisite only. L4 remains
rejected because a post-L2 generation smoke still degraded full autoregressive
generation:

```text
partial smoke:
  donor_only: 5/8
  core_off: 5/8
  full QTRM: 2/8
```

Next bottleneck: repair vocab renderer / donor fusion. The pointer state is now
causal and correct; generation is the failing layer.

## 2026-05-10 Mixed Non-Copy LM Gate Rejected

Added a reproducible post-source-copy diagnostic:

```text
scripts/330_run_mixed_noncopy_lm_gate.py
tests/test_mixed_noncopy_lm_gate.py
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest tests.test_mixed_noncopy_lm_gate

5 tests OK
```

Run:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  .venv/bin/python scripts/330_run_mixed_noncopy_lm_gate.py \
  --max-cases 16 \
  --chunk-size 4 \
  --out-dir /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/qtrm_mixed_noncopy_lm_gate_diag16 \
  --resume
```

Report:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
qtrm_mixed_noncopy_lm_gate_diag16/report.json
```

Result:

```text
decision: rejected_noncopy_lm_gate

full QTRM:  0/16 = 0.0
donor-only: 0/16 = 0.0
core-off:   0/16 = 0.0
```

Interpretation:

```text
The source-copy L4 standard128 result is real but narrow. It does not solve
computed mixed-family answers. The next bottleneck is non-copy answer synthesis
from latent state into the canonical autoregressive LM path.
```

Forced-choice follow-up:

```text
cases: 4 mixed-family non-copy rows
modes: donor_only, core_off, full QTRM
result: 0/12 hits
tail class: 12/12 doubled_list
```

This rules out a pure greedy-rendering explanation. Even with candidates
provided, the model prefers the intermediate doubled list over the final scalar
answer. The next gate must target scalar reduction/accumulator/final-answer
state in the recurrent path before another broad LM renderer run.

## 2026-05-10 Ouro Recurrent L2 Len11/13 Recheck Rejected

The preserved L2 recurrent-answer checkpoint was rechecked on the harder
mixed-family non-copy split:

```text
checkpoint:
  local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated_seed0_s40_eval4/accepted.pt

eval:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_l2_len1113_forced_choice8.jsonl

tail summary:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_l2_len1113_tail_error_summary8.json
```

Result:

```text
donor_only:        0/8
core_off:          0/8
recurrent_off:     0/8
full recurrent:    0/8

full tail classes:
  doubled_list:      6/8
  pre_subtract_sum:  2/8
```

Interpretation:

```text
The earlier len7 recurrent-answer success was a narrow local L2 result. It
does not scale to len11/13 non-copy reductions. The next architecture gate
must train and test length-stable scalar reduction plus final subtract
retention, not another source-copy or private renderer patch.
```

## 2026-05-10 Ouro Recurrent Len11/13 From Joint Controller Rejected

Ran the orthodox retry: start from the accepted len11/13 transition-joint
controller and train only the Ouro/LoopLM recurrent answer path.

```text
out:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  ouro_answer_recurrent_len1113_s020_eval4_from_jointonly

train:
  steps=20
  final_logit_ce=1.0
  depth_final_ce=1.0
  validation gate: causal forced-choice, eval4
```

Result:

```text
decision: rejected

donor_only:        0/4
core_off:          0/4
recurrent_off:     0/4
full recurrent:    0/4

action/finality sanity:
  trace exact:     32/32
  finality exact:  32/32
  halted exact:    32/32

tail summary:
  full recurrent:  doubled_list=4/4
```

Interpretation:

```text
The transition/action controller is not the blocker. More final-token CE on
the same answer loop does not make the model prefer the final scalar. The next
architecture must add a causal value accumulator / process-supervised scalar
state update that the recurrent answer path actually consumes.
```

## 2026-05-10 Relative Source-Slot LM Path Smoke

Implemented and smoke-tested relative source-slot ids through the canonical
QTRM LM path.

Commits:

```text
b39f72f feat(qtrm): add relative parity source slot mode
afc8694 feat(qtrm): propagate source slot id mode through gates
48ea7b2 fix(qtrm): preserve source copy spans for relative slots
```

Run:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_relative_parity_smoke_s001

flags:
  --token-numeric-source-slot-id-mode relative_parity
  --token-numeric-source-slot-vocab-size 3
  --steps 1
  --max-eval-cases 4
```

Result:

```text
decision: rejected_l4_candidate
full_generation_accuracy:                    0.25
donor_generation_accuracy:                   0.25
core_off_generation_accuracy:                0.25
source_slot_off_generation_accuracy:         0.25
source_binder_off_generation_accuracy:       0.25

full_minus_donor:                            0.0
full_minus_core_off:                         0.0
full_minus_source_slot_off:                  0.0
full_minus_source_binder_off:                0.0
```

Observed completion perturbation:

```text
surface_paraphrase:
  donor/core_off: 2,
  full:           56,
```

Interpretation:

```text
The relative source-slot path is executable and can perturb LM output, but it
is not yet a causal reasoning improvement. The same 1/4 hit rate appears under
donor/core-off/source-slot-off/source-binder-off ablations. Treat this as an
input-path repair plus L4 smoke evidence, not promotion. The next useful gate
must force scalar reduction/accumulator state to improve exact answers and drop
under primitive/source-slot/source-binder/answer-bridge ablations.
```

## 2026-05-10 List-Transform Eval Strictness Fix

The follow-up 20-step relative source-slot smoke rejected:

```text
/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/l4_relative_parity_smoke_s020

reported before strict fix:
  full_generation_accuracy:       1/8 = 0.125
  donor_generation_accuracy:      1/8 = 0.125
  core_off_generation_accuracy:   1/8 = 0.125
  full_minus_donor:               0.0
  full_minus_core_off:            0.0
```

Audit finding:

```text
The 1/8 hits were loose contains matches such as target "52" inside output
"52,54". That is acceptable for open QA but invalid for algorithmic CSV
answers. `list_transform` and `sequential_list_transform` now require strict
exact/normalized-exact generation scoring, matching the existing
source-copy strictness.
```

Offline strict rescore of the same JSONL:

```text
donor_only_no_evidence:                 0/8
qtrm_core_off_no_evidence:              0/8
qtrm_core_steps_8_no_evidence:          0/8
all listed L4 ablations:                0/8
```

Interpretation:

```text
The current relative source-slot LM path can alter text, but it has not yet
solved exact non-copy answer synthesis at all. The next architecture step must
target a process-supervised scalar/list accumulator that feeds the normal LM
logits path and is ablated causally.
```

## 2026-05-10 Relative Source-Slot Trainable Policy Fix

Audit finding:

```text
The relative source-slot L4 smoke changed the source-slot id space from
absolute value classes to a compact parity vocabulary. That causes
`token_numeric_source_slot_embed.weight` and some source-position binder
parameters to initialize fresh when loading the source-copy checkpoint.

The previous default L4 trainable policy only trained answer bridge / answer
loop / vocab renderer parameters. It could leave those fresh source-binding
parameters frozen, making the relative-mode smoke underpowered.
```

Implementation:

```text
Added trainable policy:
  source_slot_binder_answer_bridge_loop_vocab_renderer

It trains:
  token_numeric_source_slot_*
  core_source_position_binder_*
  answer_state_loop_*
  core_role_value_state_embed.*
  core_role_value_state_answer_*
  core_role_value_state_vocab_renderer_*
```

Interpretation:

```text
This is still the canonical prompt-token -> latent binder/core -> LM logits
path. It is not a solver. It makes the next relative source-slot experiment a
fairer test because newly initialized prompt-derived binding modules are no
longer frozen.
```

## 2026-05-10 Direct L4 Sufficient-Condition Gate

Question:

```text
Can we skip more loose L2/L3-style diagnostics and go directly to a sufficient
L4 condition for a general non-copy LM path?
```

Answer:

```text
Yes, but only by making the gate stricter, not easier.
```

Implementation:

```text
scripts/330_run_mixed_noncopy_lm_gate.py now scores strict exact generation
only. Loose contains-style `hit` fields are ignored for this gate.

Required modes:
  donor_only_no_evidence
  qtrm_core_off_no_evidence
  qtrm_core_steps_8_no_evidence
  qtrm_core_steps_8_primitive_role_value_off_no_evidence
  qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence
  qtrm_core_steps_8_core_source_position_binder_off_no_evidence
  qtrm_core_steps_8_role_value_answer_bridge_off_no_evidence
  qtrm_core_steps_8_typed_value_answer_bridge_off_no_evidence
  qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence
  qtrm_core_steps_8_answer_state_recurrent_off_no_evidence
```

Sufficient condition:

```text
full strict generation accuracy >= threshold
full > donor-only by margin
full > core-off by margin
full drops under primitive-off
full drops under source-slot-off
full drops under source-binder-off
full drops under answer-bridge-off
full drops under typed-value-answer-bridge-off
full drops under vocab-renderer-off
full drops under answer-recurrent-off
all eval commands must complete
all required modes must be present
```

Checkpoint audit:

```text
The source-copy default checkpoint is not currently runnable as a sufficient
gate input because its trainable-delta base chain points to missing file:

  local_eval/research_gate_runner/
  primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt

This is a checkpoint hygiene problem, not a sufficient-gate design problem.
```

Runnable smoke:

```text
checkpoint:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  typed_value_answer_bridge_final_choice_s020_from_s040/last.pt

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_typed_value_answer_bridge_s040.yaml

run:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  l4_sufficient_noncopy_gate_typed_value_s020_smoke_1case

decision:
  rejected_noncopy_lm_gate

strict exact:
  donor_only:        0/1
  core_off:          0/1
  full:              0/1
  every ablation:    0/1
```

Interpretation:

```text
The sufficient-condition runner is now executable and conservative. The
current typed-value checkpoint still fails the actual L4 condition: full QTRM
does not produce the final scalar answer, and disabling core paths does not
cause a measurable accuracy drop because the full path is already at zero.

The next architecture change should target scalar/list accumulator state that
causally feeds the canonical LM logits path. A source-copy renderer repair or
loose contains metric cannot count as general L4 progress.
```

## 2026-05-11 Core-State-Only Gate Contrast

Implementation:

```text
scripts/196_train_pure_recursive_depth_supervised.py
  added final LM-path contrast options for:
    core_state_zero
    answer_state_recurrent_off

tests:
  tests/test_pure_recursive_depth_supervised_train_script.py
```

RED/GREEN:

```text
RED:
  parser rejected --core-state-zero-final-contrast-weight and
  --answer-state-recurrent-final-contrast-weight.
  script text did not contain zero_core_trajectory=True or
  disable_answer_state_loop_recurrent=True trainer ablation forwards.

GREEN:
  .venv/bin/python -m unittest
    tests.test_pure_recursive_depth_supervised_train_script.
    PureRecursiveDepthSupervisedTrainScriptTests.
    test_parser_accepts_core_zero_and_answer_recurrent_final_contrast
    tests.test_pure_recursive_depth_supervised_train_script.
    PureRecursiveDepthSupervisedTrainScriptTests.
    test_training_script_has_gate_ablation_final_contrast_for_core_zero_and_recurrent_off
  OK
```

Smoke and gate:

```text
core-state-only 20-step CE recovery:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_causal_prefix_s020_from_eos
  final_path_ce reached 4.8107 in the logged window.

strict gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_s020_gate_1case
  decision: rejected_noncopy_lm_gate
  target=600054, full=00000000, donor=10000,
  core_off=!!!!!!!!, core_state_zero=55555555

gate-contrast smoke:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  smoke_core_state_only_gate_contrast_s001
  final_path_ce=2.8354, final_path_acc=0.4286
  core_state_zero_final_target_logp_delta=9.2229
  answer_state_recurrent_final_target_logp_delta=1.9025

greedy-margin continuation:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_gate_contrast_greedy_s010
  strict gate: rejected_noncopy_lm_gate
  full=00000000, typed_value_answer_bridge_off=60060060,
  core_state_zero=Vega5555555
```

Interpretation:

```text
The stricter training path now optimizes the same core-state-zero and
answer-recurrent-off ablations used by the L4 gate. The core affects generated
surface text, but the model still collapses to repeated digits and has not
converted the latent scalar state into the correct autoregressive answer.
Typed value bridge is demoted to diagnostic until it improves strict generation
and shows a positive same-run ablation drop.
```

## 2026-05-11 Orthodox Direction Skill/Wiki Reset

Updated the research-driven architecture skill and wiki to make the orthodox
direction explicit:

```text
canonical path:
  prompt/chat-template tokens
  -> tokenizer
  -> token embeddings or frozen donor states
  -> mandatory recurrent TRM/QTRM core
  -> core-state-dependent readout
  -> LM logits
  -> autoregressive text

active status:
  L2/L3 prerequisite repair, not L4 promotion.

active blocker:
  non-copy latent-state-to-autoregressive text synthesis.
```

Recorded the current reject evidence:

```text
strict generation:
  target=600054
  full=00000000
  typed_value_answer_bridge_off=60060060
  core_state_zero=Vega5555555

gold-token ranks with stripped target + EOS:
  target tokens: 6 0 0 0 5 4 <eos>
  full ranks:    1 1 1 1 6 4 10

self-rollout:
  teacher-forced/final-path accuracy improved, but strict generation stayed
  collapsed at 00000000.

beam search:
  beam_size=8 produced 60000000, so the tail failure is not only greedy
  decoding.
```

Decision:

```text
No MemoryOS/RAG, MSA/LM2, larger donor, online distillation, typed renderer,
or side solver should be promoted until the mandatory core-state-only LM path
beats donor/core-off/core-state-zero/path-off under strict generation.
```

## 2026-05-11 Tail-Weight And KISS Readout Gates

Tail/EOS-weighted continuation:

```text
train:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_tail_weight_s005

gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_tail_weight_s005_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  target=600054
  full=40000000
  typed_value_answer_bridge_off=44444444
  core_state_zero=Vega  555555
  answer_recurrent_off=00000000

rank probe:
  target tokens: 6 0 0 0 5 4 <eos>
  full ranks:    8 1 1 1 3 2 5
```

KISS no-typed-bridge candidate:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_
  core_state_only_kiss_answer_loop_s040.yaml

load check:
  answer_state_loop_core_state_only_enabled=True
  typed_algorithmic_value_state_enabled=False
  trainable_param_policy=core_and_answer_state_loop
```

No-train A/B:

```text
gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_no_train_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  full=60060060
  core_state_zero=!!!!!!!!
  answer_recurrent_off=00000000
```

5-step KISS continuation:

```text
train:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  core_state_only_kiss_answer_loop_s005

trainable:
  core_and_answer_state_loop
  35,811,332 trainable params
  90 unexpected typed/checkpoint keys intentionally dropped

gate:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  mixed_noncopy_core_state_only_kiss_s005_gate_1case

decision:
  rejected_noncopy_lm_gate

generation:
  full=00000000
  core_state_zero=!!!!!!!!
  answer_recurrent_off=00000000
```

Conclusion:

```text
Tail weighting damages the first token. KISS no-train gives the closest output
and a visible core-state-zero/recurrent-off perturbation, but current short
training collapses it. The active blocker remains the root LM readout problem:
learned recurrent latent state is not being converted into stable
autoregressive next tokens.

Next orthodox step: reset to a minimal prior-backed recurrent
latent-state-to-next-token reproduction before adding more QTRM-specific heads,
bridges, MemoryOS, MSA, larger donors, or distillation.
```

## 2026-05-11 Minimal Latent Readout Reproduction

Implemented:

```text
scripts/331_train_latent_readout_reproduction.py
tests/test_latent_readout_reproduction_script.py
```

Purpose:

```text
Isolate the current QTRM bottleneck:
  latent state -> recurrent next-token decoder -> greedy digit/EOS text.
```

RED/GREEN:

```text
RED:
  tests failed because scripts/331_train_latent_readout_reproduction.py did
  not exist.

GREEN:
  .venv/bin/python -m unittest tests.test_latent_readout_reproduction_script
  Ran 4 tests: OK
```

Runs:

```text
teacher forcing only:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_teacher_forcing_s200/report.json

scheduled sampling:
  /mnt/nvme0n1p2/qtrm-runs/research_gate_runner/
  latent_readout_repro_scheduled_s200/report.json
```

Both profiles:

```text
teacher_forced_token_acc: 1.0
teacher_forced_exact:    1.0
greedy_token_acc:        1.0
greedy_exact:            1.0
decision:                accepted
```

Decision:

```text
Accept as L1 minimal readout reproduction only. This proves the readout problem
is solvable when latent states are sufficient and token-aligned. It does not
prove QTRM core states are sufficient or token-aligned.

Next QTRM step: port the minimal contract
  core trajectory -> per-output-step latent readout -> recurrent next-token
  readout -> LM logits
and require strict greedy generation plus core_state_zero and recurrent-off
destructive drops.
```

## [2026-05-12] architecture | Free Transformer latent scaffold for QTRM

Implemented a Free-Transformer-style latent conditioning scaffold inside the
QTRM answer loop:

```text
training: answer posterior latent from full training context
inference: answer prior latent from QTRM answer hidden state
loss: KL/free-bits hook in scripts/196
output: normal LM logits only
```

Files:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
scripts/196_train_pure_recursive_depth_supervised.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_free_transformer_latent_s040.yaml
scripts/332_run_free_transformer_latent_smoke.sh
docs/wiki/decisions/free-transformer-latent-for-qtrm.md
```

Decision:

```text
This is a scaffold, not an accepted L4 result. It is promoted only if strict
greedy generation improves and core/answer/next-token/free-latent ablations
show causal drops.
```

Smoke command:

```text
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  bash scripts/332_run_free_transformer_latent_smoke.sh
```

Follow-up implementation:

```text
Added answer_free_transformer_latent_final_contrast so full LM logits must beat
the same forward pass with Free Transformer latent conditioning disabled.
This makes `answer_free_transformer_latent_off` a trained causal ablation,
not only an evaluation toggle.
```

Quick smoke:

```text
command:
  HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src \
  STEPS=1 MAX_TARGET_TOKENS=2 SELF_ROLLOUT_WEIGHT=0.0 \
  OUT_BASE=local_eval/free_transformer_latent_contrast_smoke_quick \
  bash scripts/332_run_free_transformer_latent_smoke.sh

train:
  answer_free_transformer_latent_final_contrast=1.1735
  answer_free_transformer_latent_final_target_logp_delta=-1.0735
  answer_free_transformer_latent_kl=0.6436
  answer_free_transformer_gate_mean=1.0000

gate:
  decision=rejected_noncopy_lm_gate
  full_generation_accuracy=0.0
  full_minus_answer_free_transformer_latent_off=0.0
  report=local_eval/free_transformer_latent_contrast_smoke_quick/gate_1case_s1/report.json
```

The scaffold is runnable, but this is still prerequisite repair, not L4.

Stage-5 diagnostics:

```text
S5 mixed-depth:
  full=66666666
  free_latent_off=66666666
  decision=rejected_noncopy_lm_gate

S5 no-repeat diagnostic:
  full=604: UNKNOWN5Answer1
  decision=rejected_noncopy_lm_gate
  interpretation=not just a repetition-decoding problem

S5 depth8-only:
  full=66666666
  free_latent_off=66666666
  decision=rejected_noncopy_lm_gate

S5 depth8-only + self-rollout:
  self_rollout_prefix_mismatch_rate=1.0
  full=66666666
  free_latent_off=66666666
  decision=rejected_noncopy_lm_gate
```

Conclusion:

```text
Free Transformer latent conditioning is wired correctly but has no greedy
causal gain yet. Continue with the core-state-to-token synthesis bottleneck;
do not promote this to L4 and do not hide the failure with no-repeat decoding.
```

Follow-up diagnostics:

```text
S5 depth8-only + skip-leading-whitespace:
  full=60060066
  free_latent_off=60060066
  rank: 6@1, 0@1, 0@1, 0@2, 5@8, 4@5, EOS@55

S3 target-token-8 from base:
  full=00000000
  first 6 rank=2

S3 staged T2->T8:
  full=00000000

S3 staged T2->T8 later_token_weight=0.25:
  first 6 rank=2
  max rank=59

S3 staged T2 step5 -> T4:
  source=local_eval/free_transformer_latent_contrast_t2_depth8_skipws_saveevery_s5/train_s5/step_000005.pt
  step_000001 rank=6@2,0@1,0@1,0@1,5@9,4@6,EOS@73
  step_000002 rank=6@2,0@1,0@1,0@1,5@9,4@6,EOS@101
  step_000003 rank=6@2,0@1,0@1,0@1,5@10,4@6,EOS@124
```

Decision:

```text
The useful fix was target alignment, not Free Transformer latent causality.
Full-answer token coverage is necessary but destabilizes the learned prefix.
Even the smaller T2->T4 stage regresses the first answer token from rank 1 to
rank 2, so this is not only a full-answer-length jump problem.
The smoke wrapper now exposes SAVE_EVERY so future T8 runs can use
validation-gated checkpoint selection instead of trusting the final step.
```

## [2026-05-12] architecture | next-token future auxiliary QTRM port

Evaluator cleanup:

```text
scripts/192_eval_raw_intelligence.py now records gold_answer and
canonical_completion separately. The old canonical_answer field is retained
for compatibility but is completion-derived.

Verification:
  .venv/bin/python -m unittest tests.test_raw_intelligence_eval_script
  Ran 14 tests: OK
```

Training alignment fix:

```text
answer_state_loop_future_token_targets now uses causal_prefix_answer_token_ids,
so future-token CE shares skip-leading-whitespace and EOS settings with the
main causal-prefix answer path.

Verification:
  .venv/bin/python -m unittest tests.test_pure_recursive_depth_supervised_train_script
  Ran 124 tests: OK
```

Implemented:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_mandatory_next_token_decoder_future_aux_s040.yaml
scripts/333_run_nexttok_future_aux_smoke.sh
```

Baseline regate:

```text
artifact=local_eval/orthodox_mandatory_nexttok_regate_goldfields
gold_answer=600054
full=00000000
answer_next_token_decoder_off=00000000
core_state_zero=!!!!!!!!
decision=rejected_noncopy_lm_gate
```

Future-token S5:

```text
artifact=local_eval/nexttok_future_aux_smoke_s5
full=66666666
answer_recurrent_off=00000000
answer_next_token_decoder_off=66666666
decision=rejected_noncopy_lm_gate

rank:
  step_000001 full=6@2,0@1,0@1,0@1,5@4,4@3,EOS@18
  step_000004 full=6@1,0@2,0@2,0@2,5@4,4@3,EOS@17
  step_000005 full=6@1,0@2,0@2,0@2,5@4,4@3,EOS@16
```

T4 self-rollout continuation:

```text
artifact=local_eval/nexttok_future_aux_stage_s5_to_t4_selfroll_s3
best training telemetry at step_000002:
  final_path_acc=0.75
  final_greedy_token_win_rate=0.75
  causal_prefix_self_rollout_prefix_mismatch_rate=0.3333

rank:
  step_000001 full=6@2,0@1,0@1,0@1,5@4,4@3,EOS@17
  step_000002 full=6@2,0@1,0@1,0@1,5@6,4@3,EOS@19
  step_000003 full=6@2,0@1,0@1,0@1,5@7,4@3,EOS@30
```

Decision:

```text
Reject as L4/general-LM promotion.

The future-token auxiliary creates a partial readout-direction signal: it can
move the first token from 0 toward 6 and recurrent_off changes the output.
However, it still collapses into digit repetition and next_token_decoder_off
ties full in greedy generation. The current decoder is not yet a faithful
port of the L1 latent-readout reproduction because it does not explicitly
consume previous generated answer-token embeddings.

Next orthodox step:
  implement a previous-token-conditioned latent readout before more weight
  sweeps or Free-Transformer-style scaffolds.
```
## 2026-05-12 12:38 KST - ETD Thinking-Block Correction

Corrected the `think block` reference set.

Primary paper:

```text
Encode, Think, Decode: Scaling test-time reasoning with recursive latent
thoughts
arXiv:2510.07358
```

Local artifact:

```text
references/papers/recurrent_depth/encode_think_decode_2510.07358.pdf
```

QTRM implication:

```text
ETD is a more direct prior than generic text `<think>` blocks. It splits a
transformer into encode / repeated middle thinking block / decode and scales
test-time latent recursion inside the normal LM path. This strengthens the
case that QTRM should test a compact recurrent reasoning block feeding the
ordinary LM head before adding more external answer-loop renderer patches.
```

## 2026-05-12 12:52 KST - Previous-Token Latent Readout Scaffold

Implemented the next orthodox renderer repair:

```text
previous input/generated token embedding
+ QTRM answer/core latent hidden
-> previous-token fusion
-> next-token decoder block
-> LM head
```

Files:

```text
src/qtrm_mm/config.py
src/qtrm_mm/qtrm_model.py
configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_core_state_only_kiss_prev_token_readout_s040.yaml
scripts/334_run_prev_token_readout_smoke.sh
tests/test_prev_token_latent_readout.py
```

Verification:

```text
.venv/bin/python -m py_compile src/qtrm_mm/qtrm_model.py src/qtrm_mm/config.py \
  scripts/196_train_pure_recursive_depth_supervised.py \
  scripts/330_run_mixed_noncopy_lm_gate.py

PYTHONPATH=src .venv/bin/python -m unittest tests.test_prev_token_latent_readout
Ran 3 tests: OK
```

Decision:

```text
This is a scaffold, not a promoted result. Next action is
scripts/334_run_prev_token_readout_smoke.sh and strict gate/ablation reading.
```

S5 result:

```text
artifact=local_eval/prev_token_readout_smoke_s5
gold_answer=600054
full=00000000
donor_only=10000
core_off=!!!!!!!!
core_state_zero=6054
answer_recurrent_off=00000000
answer_next_token_decoder_off=00000000
decision=rejected_noncopy_lm_gate
```

Rank probe:

```text
full:            6@2,0@1,0@1,0@1,5@4,4@3,EOS@15
decoder_off:     6@2,0@1,0@1,0@1,5@6,4@4,EOS@12
recurrent_off:   6@4,0@1,0@1,0@1,5@8,4@6,EOS@16
core_state_zero: 6@1,0@1,0@3,0@3,5@1,4@1,EOS@1
```

Conclusion:

```text
Rejected. The previous-token readout is implemented correctly as a scaffold,
but readout-only tuning is not sufficient. The full core is worse than
core_state_zero on gold-token ranks, so the next run must unlock the recurrent
core with the answer loop and require full > core_state_zero. If that fails,
the architecture should pivot toward an ETD-style in-path thinking block rather
than more renderer patches.
```

Core+answer loop unlock:

```text
artifact=local_eval/prev_token_readout_core_answer_s5
trainable_param_policy=core_and_answer_state_loop
gold_answer=600054
full=60000066
core_state_zero=6054
answer_recurrent_off=00000000
answer_next_token_decoder_off=66666666
decision=rejected_noncopy_lm_gate
```

Rank probe:

```text
full:            6@1,0@1,0@1,0@1,5@4,4@3,EOS@16
decoder_off:     6@1,0@2,0@2,0@2,5@5,4@3,EOS@13
recurrent_off:   6@3,0@1,0@1,0@1,5@8,4@6,EOS@22
core_state_zero: 6@1,0@1,0@2,0@2,5@1,4@1,EOS@1
```

Suffix-pressure continuation:

```text
artifact=local_eval/prev_token_readout_core_answer_t5_later2
init=local_eval/prev_token_readout_core_answer_s5/train_s5/last.pt
LATER_TOKEN_WEIGHT=2.0
SELF_ROLLOUT_WEIGHT=0.5
full=00000000
core_state_zero=6054
decision=rejected_noncopy_lm_gate
```

Decision:

```text
Stop this as a weight sweep. The core+answer unlock produced a partial signal
but did not beat core_state_zero, and suffix-pressure continuation regressed.
The next orthodox step is an ETD-style in-path thinking-block probe rather than
more external answer-loop renderer patches.
```
## 2026-05-12 13:42 KST - QTRM-Native First Pivot

Canonical direction updated:

```text
tokenizer / token ids
-> token embeddings
-> native encoder
-> mandatory TRM/QTRM recursive thinking core
-> native decoder/readout
-> LM head
-> autoregressive text
```

Reason:

```text
The donor-sidecar path repeatedly produced non-causal or harmful-core results:
strict greedy generation failed, answer-loop/readout patches did not promote,
and core_state_zero sometimes ranked the gold answer better than the full core.
This suggests the core and donor are not one integrated LM residual path.
```

Updated:

```text
/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md
docs/wiki/architecture/qtrm-native-first-roadmap.md
docs/wiki/index.md
```

Current native scaffold:

```text
scripts/335_train_qtrm_native_etd_probe.py
tests/test_qtrm_native_etd_probe.py
```

Claim:

```text
Not solved. This is a native-first L1 falsifier for whether a donorless
encode -> repeated thinking block -> decode -> LM-head path can learn stable
autoregressive outputs and show depth/ablation gain.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py

PYTHONPATH=src .venv/bin/python -m unittest tests.test_qtrm_native_etd_probe
Ran 4 tests: OK
```

First smoke:

```text
artifact=local_eval/qtrm_native_etd_smoke_cpu_s20
loss=3.8876 -> 1.8404
full_generation_exact=0.0
think0_generation_exact=0.0625
state_reset_generation_exact=0.0
op_zero_generation_exact=0.0
decision=rejected
```

Conclusion:

```text
Native path is implemented and trainable enough for loss to drop, but the first
smoke collapses to immediate EOS in greedy generation. Next bottleneck is
native answer/EOS decoding and on-policy generation, not donor integration.
```

## 2026-05-12 18:55 KST - Mamba-3 Encode/Decode Isolation

Question:

```text
After Mamba-3 passed the short recurrent think-core seed sweep, should
encode/decode also move from MHA ETD to Mamba-3 or FLA/GatedDeltaNet?
```

Experiment:

```text
script:
  scripts/342_qtrm_native_l5d_backbone_compare.py

seed sweep:
  scripts/343_qtrm_native_l5d_placement_seed_sweep.py

artifact:
  local_eval/qtrm_native_l5d_all_mamba3_seed_sweep_short_triton351

candidates:
  mha_etd
  official_mamba3_think
  official_fla_encode_decode_mamba3_think
  official_mamba3
```

Result:

```text
official_mamba3_think:
  accepted_l5d_placement_seed_stability
  promoted_count: 3 / 3
  min_delta_vs_mha: 0.005208333333
  exact range: 0.046875 .. 0.052083333333333336

official_mamba3:
  rejected
  promoted_count: 2 / 3
  min_delta_vs_mha: -0.005208333333
  exact range: 0.041666666666666664 .. 0.078125

official_fla_encode_decode_mamba3_think:
  rejected
  promoted_count: 2 / 3
  min_delta_vs_mha: -0.005208333333
  exact range: 0.041666666666666664 .. 0.057291666666666664
```

Decision:

```text
Do not replace encode/decode MHA ETD yet. Mamba-3 is currently supported only
as the recurrent thinking core:

  MHA ETD encode -> official Mamba-3 recurrent think -> MHA ETD decode

All-Mamba3 and FLA-encode/decode remain research candidates until they pass the
same seed-stability and causal-ablation gates.
```

## 2026-05-12 19:08 KST - FLA 3:1 Think-Core Verification

Question:

```text
Was Qwen3.5-style 3:1 actually tested in the thinking backbone, and was it bad?
```

Answer:

```text
It was tested as `official_fla_think`:

  MHA ETD encode -> Qwen3.5-style official FLA/GatedDeltaNet 3:1 think
  -> MHA ETD decode -> LM head

It was not bad. It passed the short L5D seed-stability gate:

  artifact: local_eval/qtrm_native_l5d_placement_seed_sweep_short
  decision: accepted_l5d_placement_seed_stability
  promoted_count: 3 / 3
  causal_ok_count: 3 / 3
  backend_ok_count: 3 / 3
  min_delta_vs_mha: 0.005208333333
  max_delta_vs_mha: 0.067708333333
  exact range: 0.052083333333333336 .. 0.08333333333333333
```

Fresh runtime smoke:

```text
artifact:
  local_eval/qtrm_native_l5d_verify_fla_mamba3_think_smoke_20260512

candidates:
  mha_etd
  official_fla_think
  official_mamba3_think

result:
  completed_l5d_backbone_compare

backend:
  official_fla_think:
    official_fla_delta_mixers: 3
    torch_delta_mixers: 0
    all_fla_mixers_official: true

  official_mamba3_think:
    official_mamba3_mixers: 1
    all_mamba3_mixers_official: true
```

Decision:

```text
3:1 official FLA/GatedDeltaNet is a valid thinking-core candidate and should
not be described as failed. The failed variants were full/all-stage replacement
or unstable encode/decode replacement. The current stronger short candidate is
Mamba-3 think-core, but it still needs the broader language non-regression and
scaled-reasoning gates that FLA already passed earlier.
```

## 2026-05-12 20:15 KST - Mamba3 Think-Core Language And Scaled Reasoning

Objective:

```text
Verify whether the selected architecture preserves language behavior and
causal reasoning:

  MHA ETD encode -> official Mamba-3 recurrent think -> MHA ETD decode -> LM head
```

Implementation updates:

```text
scripts/336_train_qtrm_native_text_probe.py
  accepts mamba3 stage backbones
  records backend_summary for FLA/Mamba3 strict-backend evidence

scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  accepts --target-level for runner-specific reports

scripts/300_research_gate_runner.py
  adds qtrm_native_l5d_mamba3_placement_language_nonregression
  adds qtrm_native_l5d_mamba3_placement_scaled_reasoning
```

Language gate:

```text
gate:
  qtrm_native_l5d_mamba3_placement_language_nonregression

profile:
  standard

decision:
  accepted_l5d_mamba3_placement_language_nonregression

report:
  local_eval/research_gate_runner_mamba3_verify_20260512/
  qtrm_native_l5d_mamba3_placement_language_nonregression_standard/report.json

metrics:
  think_eval_loss: 1.7053431420461507
  think0_loss: 5.173120769203132
  thinking_block_off_loss: 5.173120769203132
  think0_baseline_loss: 1.9871065920971809
  full_vs_think0: 0.32965461626151865
  full_vs_baseline: 0.8582041591670939
  sample_unique_chars: 21
  sample_max_run_fraction: 0.015151515151515152
```

Scaled reasoning gate:

```text
gate:
  qtrm_native_l5d_mamba3_placement_scaled_reasoning

profile:
  standard

decision:
  accepted_l5d_mamba3_placement_scaled_reasoning

report:
  local_eval/research_gate_runner_mamba3_verify_20260512/
  qtrm_native_l5d_mamba3_placement_scaled_reasoning_standard/report.json

metrics:
  full_generation_exact: 0.171875
  think0_generation_exact: 0.0
  state_reset_generation_exact: 0.020833333333333332
  op_zero_generation_exact: 0.026041666666666668
  full_minus_think0: 0.171875
  full_minus_worst_ablation: 0.14583333333333334
  min_family_generation_exact: 0.046875
```

Backend evidence:

```text
mamba3_mixers: 1
official_mamba3_mixers: 1
torch_delta_mixers: 0
all_mamba3_mixers_official: true
```

Important failure and fix:

```text
The initial scaled reasoning standard config used d_model=96/headdim=24 and
diverged to NaN before step 200. Lowering LR to 1e-4 still produced NaN.

The accepted standard config uses d_model=64/headdim=16. This should remain
the runner default until the wider Mamba-3 instability is fixed.
```

Conclusion:

```text
For the current native scaffold, MHA encode/decode + official Mamba-3 think
has now passed:

1. short seed-stability;
2. standard language non-regression;
3. standard scaled reasoning with causal ablation drops.

This is architecture-level evidence, not a broad general-LLM claim. A stricter
future gate should train one joint language+reasoning checkpoint and evaluate
both from the same checkpoint.
```

## 2026-05-12 - Official TRM Dual-State Native Candidate

Implemented the first QTRM-native official-TRM-style thinking structure.

Code:

```text
scripts/335_train_qtrm_native_etd_probe.py
scripts/336_train_qtrm_native_text_probe.py
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
scripts/342_qtrm_native_l5d_backbone_compare.py
```

New args:

```text
--think-structure trm_dual_z
--trm-l-cycles
--trm-full-grad-cycles
```

Structural correction:

```text
Official Samsung TRM uses shared reasoning weights for both z_L and z_H
updates. The native implementation therefore reuses one think block:

  z_L = think(z_L + z_H + encoded_tokens)
  z_H = think(z_H + z_L)

with z_H feeding decode -> LM head.
```

Verification:

```text
PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/336_train_qtrm_native_text_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  scripts/342_qtrm_native_l5d_backbone_compare.py

PYTHONPATH=local_deps/mamba3_runtime:src .venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_text_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe \
  tests.test_qtrm_native_l5d_backbone_compare

result:
  48 tests OK
```

Runtime smoke:

```text
out_root:
  local_eval/qtrm_native_l5e_trm_dual_z_compare_smoke_20260512

candidates:
  mha_etd
  official_fla_think
  official_mamba3_think
  trm_dual_z_fla_think
  trm_dual_z_mamba3_think

result:
  all candidates executed with backend guards satisfied
```

Short comparison:

```text
out_root:
  local_eval/qtrm_native_l5e_trm_dual_z_compare_short_20260512

winner:
  official_mamba3_think

full_generation_exact:
  mha_etd:                 0.015625
  official_fla_think:      0.036458333333333336
  official_mamba3_think:   0.046875
  trm_dual_z_fla_think:    0.026041666666666668
  trm_dual_z_mamba3_think: 0.020833333333333332
```

Promotion status:

```text
official_fla_think:
  promoted: true
  full_minus_think0: 0.03125
  full_minus_worst_ablation: 0.010416666666666668

official_mamba3_think:
  promoted: true
  full_minus_think0: 0.046875
  full_minus_worst_ablation: 0.010416666666666664

trm_dual_z_fla_think:
  promoted: false
  full_minus_worst_ablation: -0.020833333333333332

trm_dual_z_mamba3_think:
  promoted: false
  full_minus_think0: -0.005208333333333336
  full_minus_worst_ablation: -0.015625000000000003
```

Decision:

```text
The official-TRM-style z_L/z_H loop is now implemented, but the first short
gate does not prove it as the canonical QTRM-native core. The strongest current
validated placement remains MHA encode/decode + official Mamba-3 single
recurrent think-core. GatedDeltaNet/FLA is not eliminated because it also
promoted in the same single-state comparison.
```

## Wiki Update 2026-05-14T22:05:00

```text
topic:
  Mercury-2, Mercury diffusion LLM, Token Superposition Training, and
  Fast-Slow Training mapped to QTRM-native.

files:
  docs/wiki/sources/diffusion-fast-slow-llm-2026.md
  docs/wiki/decisions/qtrm-native-hard-lock.md
  docs/wiki/architecture/qtrm-native-first-roadmap.md
  docs/wiki/index.md

decision:
  Mercury-style diffusion is a future decoder/readout candidate, not a
  replacement for the mandatory QTRM recursive core.

  Token Superposition Training is a later native pretraining throughput
  candidate, not the immediate L4 causality fix.

  Fast-Slow Training is the most directly relevant near-term idea because it
  maps to z_L fast state, z_H slow state, and nested learned update pressure.

active background gate:
  qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare
  profile: standard
  log: local_eval/background_logs/split_mixer_standard_20260514_220000.log
```

## Implementation 2026-05-14T22:15:00

```text
topic:
  Fast-Slow latent update repair scaffold.

files:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  scripts/300_research_gate_runner.py
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
  tests/test_research_gate_runner.py
  docs/wiki/sources/diffusion-fast-slow-llm-2026.md
  docs/wiki/decisions/qtrm-native-hard-lock.md

new loss:
  fast_slow_latent_counterfactual_loss

new gate:
  qtrm_native_fast_slow_latent_update_l4_repair

verification:
  .venv/bin/python -m py_compile \
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
    scripts/300_research_gate_runner.py

  .venv/bin/python -m unittest \
    tests.test_qtrm_native_mixed_text_reasoning_probe \
    tests.test_research_gate_runner -v

  result:
    147 tests OK

smoke:
  local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_smoke/report.json

decision:
  smoke_passed_fast_slow_latent_update_l4_repair
```

## Experiment 2026-05-14T22:58:41

```text
topic:
  Fast-Slow latent update standard gate result.

gate:
  qtrm_native_fast_slow_latent_update_l4_repair

report:
  local_eval/research_gate_runner/qtrm_native_fast_slow_latent_update_l4_repair_standard/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.2421875
  think0_generation_exact: 0.03125
  full_minus_think0: 0.2109375
  full_minus_worst_ablation: 0.1875
  z_l_zero_generation_exact: 0.0234375
  z_h_zero_generation_exact: 0.0

interpretation:
  Fast-Slow gives a small gain over the raw official-schedule split-mixer
  branch, but it does not close the gap to the accepted nested MHA/ETD branch
  at 0.67578125. The active path remains
  trm_dual_z_nested_reversed_mha_etd; the next bottleneck is multi-family seed
  stability, not more Mamba3/GatedDelta mixer shopping.
```

## Experiment 2026-05-14T23:25:00

```text
topic:
  L5 multi-family seed stability repair.

finding:
  nested dual-z should not be globally frozen yet. On seed339 multi-family it
  preserves full accuracy at initialization but z_L ablation has almost no
  effect. Strong z_L counterfactual training makes z_L causal but collapses
  full/min-family accuracy.

accepted repair:
  single recurrent MHA/ETD
  resume from seed339 multi-family checkpoint
  lr: 2e-5
  task_families: modchain,modchain,revchain,checksum
  family_dro_loss_weight: 0.05
  retention_kl_loss_weight: 0.50

accepted report:
  local_eval/qtrm_native_l5_multifamily_seed339_single_modchain_weak_repair_s1500_20260514/report.json

seed339 metrics:
  full_generation_exact: 0.6822916666666666
  min_family_generation_exact: 0.5078125
  full_minus_worst_ablation: 0.6549479166666666

repaired seed-stability summary:
  local_eval/qtrm_native_l5_multifamily_repaired_seed_stability_20260514_summary.json

summary decision:
  accepted_l5_multifamily_repaired_seed_stability

summary metrics:
  pass_rate: 3 / 3
  min_full_generation_exact: 0.6067708333333334
  min_family_generation_exact: 0.4140625
  min_full_minus_worst_ablation: 0.5716145833333334
```

## Experiment 2026-05-14T23:55:00

```text
topic:
  L6 len6 transfer from the repaired L5 multi-family baseline.

reports:
  zero-shot:
    local_eval/qtrm_native_l6_len6_transfer_from_l5_repair_seed339_init_20260514/report.json
  focused:
    local_eval/qtrm_native_l6_len6_from_l5_repair_focused_s2500_20260514/report.json
  answer-space diagnostic:
    local_eval/qtrm_native_l6_len6_focused_s2500_answer_space_eval_20260514/report.json

zero-shot decision:
  rejected

zero-shot metrics:
  full_generation_exact: 0.0078125
  min_family_generation_exact: 0.00390625
  full_minus_worst_ablation: 0.00390625

focused decision:
  rejected

focused final metrics:
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.05859375
  full_minus_worst_ablation: 0.05989583333333333

focused trend:
  step0 full: 0.0078125, min_family: 0.00390625
  step1500 full: 0.0859375, min_family: 0.05859375
  step2500 full: 0.21354166666666666, min_family: 0.046875

answer-space diagnostic:
  answer_space_argmax_exact: 0.0794270858168602
  answer_space_gold_mean_rank: 10.708333015441895
  answer_space_gold_top3: 0.2161458283662796
  answer_space_gold_top5: 0.3515625

interpretation:
  L6 is not mainly a greedy renderer problem. Answer-space argmax is nearly
  the same as greedy exact, so the model does not merely know the answer but
  fail to print it. The current bottleneck is ordered recurrent transition
  scaling under longer programs and hard-family balance.

decision:
  Do not hard-freeze nested dual-z or Fast-Slow globally. Keep the accepted
  L5 single recurrent MHA/ETD baseline as the stable checkpoint family, and
  treat L6 as the next separate length-scaling bottleneck.
```

## Experiment 2026-05-15T00:10:00

```text
topic:
  L6 active-length batch-cycle repair attempt.

report:
  local_eval/qtrm_native_l6_len6_from_l5_repair_active_batch_cycle_s2500_20260514/report.json

recipe:
  resume from accepted L5 seed339 weak-family repair
  program_len: 6
  active_len_batch_cycle: true
  train_active_len_cycle_min: 1
  train_active_len_cycle_max: 6
  family_dro_loss_weight: 0.05
  retention_kl_loss_weight: 0.10

decision:
  rejected

metrics:
  full_generation_exact: 0.052083333333333336
  min_family_generation_exact: 0.046875
  full_minus_think0: 0.03515625
  full_minus_worst_ablation: 0.01953125

interpretation:
  Batch-level active-length mixing does not solve L6 and is worse than the
  prior focused len6 fine-tune on full exact. The bottleneck is not a simple
  exposure schedule issue. Next useful work should target the recurrent
  transition/state representation or a more principled length curriculum,
  while keeping the canonical QTRM-native token->core->logits path.
```

## Diagnostic 2026-05-15T00:25:00

```text
topic:
  L6 state/readout probe on the focused len6 checkpoint.

report:
  local_eval/qtrm_native_l6_len6_focused_state_probe_eval_20260515/report.json

result:
  full_generation_exact: 0.0859375
  min_family_generation_exact: 0.05859375
  depth_sweep exact:
    depth0: 0.010416666666666666
    depth1: 0.016927083333333332
    depth2: 0.03125
    depth3: 0.016927083333333332
    depth4: 0.0859375
  state_trace z_h variance:
    0.24019193649291992
    0.3616328239440918
    0.5645048022270203
    0.8666419386863708
  core_step_probe_exact: 0.09765625
  core_step_probe_by_family:
    modchain: 0.08984375
    revchain: 0.0498046875
    checksum: 0.1533203125

interpretation:
  The recurrent state is not collapsed; variance and step delta grow with
  depth. The problem is that the state is not cleanly readable as the causal
  intermediate calculation. L6 needs a stronger transition/state
  representation, not merely anti-collapse, answer-space reranking, or a
  visible prompt anchor.
```

## Experiment 2026-05-15T00:35:00

```text
topic:
  L6 latent-refinement state objective.

report:
  local_eval/qtrm_native_l6_len6_from_l5_repair_latent_refine_s2500_20260515/report.json

recipe:
  resume from accepted L5 seed339 weak-family repair
  latent_refine_loss_weight: 0.05
  latent_refine_noise_std: 0.05
  latent_refine_depth_weight_power: 1.0
  latent_refine_final_kl_weight: 0.1
  family_dro_loss_weight: 0.05
  periodic_eval_score_mode: family_floor

decision:
  rejected

metrics:
  full_generation_exact: 0.08854166666666667
  min_family_generation_exact: 0.0546875
  full_minus_think0: 0.07291666666666667
  full_minus_worst_ablation: 0.06640625
  core_step_probe_exact: 0.0960286483168602

periodic:
  step1500 full: 0.08854166666666667, min_family: 0.0546875
  step2500 full: 0.23046875, min_family: 0.05078125

interpretation:
  Latent refinement improves the high-full late checkpoint slightly but does
  not improve weakest-family accuracy or core-step readability. The hard
  family remains modchain, with revchain second. The next trial should target
  hard-family transition balance or change the recurrent state carrier itself.
```

## Experiment 2026-05-15T00:55:00

```text
topic:
  Clarify block/backbone versus dual/nested topology, then test whether a
  TRM official pre-norm think block helps inside the dual/nested QTRM-native
  path.

terminology:
  backbone / stage backbone:
    the local block used for encoder, think, or decoder stages.
  think_structure:
    the macro recurrent topology such as single, dual-z, nested dual-z, or
    official H/L schedule.

rule:
  trm_official_prenorm is a block candidate. It does not replace or abandon
  dual/nested by itself. A canonical dual/nested experiment must explicitly
  set a dual/nested think_structure.

smoke:
  report:
    local_eval/qtrm_native_l6_dual_nested_trm_official_prenorm_noabs_smoke_20260514_235644/report.json
  structure:
    think_structure: trm_dual_z_nested_reversed_mha_etd
    encode_backbone: trm_official_prenorm
    think_backbone: trm_official_prenorm
    decode_backbone: trm_official_prenorm
    position_embedding_mode: none
    H/L: train_think_steps=3, trm_l_cycles=6
  decision:
    rejected
  decisive:
    full_generation_exact: 0.0
    min_family_generation_exact: 0.0
    full_minus_worst_ablation: 0.0

triage no-abs:
  report:
    local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_noabs_triage_20260514_235922/report.json
  structure:
    resume from local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/last.pt
    train_only_resume_missing_params: true
    think_structure: trm_dual_z_nested_reversed_mha_etd
    think_backbone: trm_official_prenorm
    encode/decode_backbone: mha_etd
    position_embedding_mode: none
  decision:
    rejected
  decisive:
    full_generation_exact: 0.041666666666666664
    full_minus_think0: 0.041666666666666664
    full_minus_worst_ablation: -0.03125000000000001
    z_l_zero_generation_exact: 0.03125
    z_h_zero_generation_exact: 0.0

triage learned-pos control:
  report:
    local_eval/qtrm_native_l4_dual_nested_think_official_prenorm_learnedpos_triage_20260515_000034/report.json
  structure:
    same as no-abs triage, but position_embedding_mode: learned
  decision:
    rejected
  decisive:
    full_generation_exact: 0.052083333333333336
    full_minus_think0: 0.04166666666666667
    full_minus_worst_ablation: -0.010416666666666664
    z_l_zero_generation_exact: 0.052083333333333336
    z_h_zero_generation_exact: 0.0

interpretation:
  The official pre-norm think block does not currently improve the accepted
  nested MHA/ETD path. The learned-position control is only slightly better
  than no-abs and still far below the accepted nested MHA L4 baseline
  full_generation_exact=0.67578125. Because z_l_zero is at or near full
  accuracy in the learned-pos control, this is not a causally useful
  dual/nested improvement.

decision:
  Do not promote trm_official_prenorm as the dual/nested think block. Keep the
  accepted nested MHA/ETD path as the local dual/nested reference, and treat
  official-prenorm/no-abs as rejected diagnostics until a new state-carrier
  hypothesis beats the L4 gate with causal ablations.
```

## Architecture 2026-05-15T01:15:00

```text
topic:
  Add a conservative residual joint-readout variant for the accepted nested
  MHA/ETD dual-z path.

motivation:
  L6 diagnostics showed that recurrent state moves but is not cleanly readable
  as the causal intermediate calculation. A full joint-readout replacement
  improves causal ablation margins but lowers L4 exactness, so the next
  candidate preserves the accepted MHA/ETD readout and adds a small residual
  joint state bridge.

new think_structure:
  trm_dual_z_nested_reversed_mha_etd_joint_readout
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout

residual form:
  base = z_H + tanh(alpha) * (z_L - z_H)
  residual = LN(W[encoded, z_L, z_H])
  readout = base + tanh(beta) * residual

code:
  scripts/335_train_qtrm_native_etd_probe.py
  tests/test_qtrm_native_etd_probe.py
  tests/test_qtrm_native_mixed_text_reasoning_probe.py

verification:
  .venv/bin/python -m py_compile scripts/335_train_qtrm_native_etd_probe.py scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  .venv/bin/python -m unittest tests.test_qtrm_native_etd_probe tests.test_qtrm_native_mixed_text_reasoning_probe -v
  result: 183 tests OK
```

Joint-readout replacement diagnostic:

```text
report:
  local_eval/qtrm_native_l4_nested_mha_joint_readout_triage_20260515_000707/report.json

decision:
  rejected only by full_exact threshold

metrics:
  full_generation_exact: 0.5729166666666666
  full_minus_think0: 0.5625
  full_minus_worst_ablation: 0.35416666666666663
  z_l_zero_generation_exact: 0.10416666666666667
  z_h_zero_generation_exact: 0.21875

interpretation:
  The replacement joint-readout makes z_L/z_H much more causally visible, but
  it falls below the accepted L4 exactness threshold. It is useful evidence for
  the readout-coupling hypothesis but not a promoted baseline.
```

Full-train continuation control:

```text
report:
  local_eval/qtrm_native_l4_nested_mha_joint_readout_standard_cont_20260515_000749/report.json

decision:
  rejected

metrics:
  full_generation_exact: 0.1328125
  full_minus_worst_ablation: 0.07421875

interpretation:
  Training all loaded base parameters after the good triage destroys the
  previously stable path. Future continuation from accepted checkpoints should
  freeze the base path unless the explicit purpose is a full retune.
```

Added-params continuation control:

```text
report:
  local_eval/qtrm_native_l4_nested_mha_joint_readout_addedparams_cont_20260515_000946/report.json

decision:
  rejected by full exact

metrics:
  full_generation_exact: 0.55078125
  full_minus_worst_ablation: 0.46484375

interpretation:
  Freezing the base path preserves the causal readout gain much better than
  full retuning, but pure replacement joint-readout still stays below the
  accepted exactness threshold.
```

Residual joint-readout accepted diagnostic:

```text
report:
  local_eval/qtrm_native_l4_nested_mha_residual_joint_readout_from_accepted_20260515_001357/report.json

recipe:
  resume:
    local_eval/research_gate_runner/qtrm_native_nested_dual_reverse_l4_baseline_compare_standard/last.pt
  think_structure:
    trm_dual_z_nested_reversed_mha_etd_residual_joint_readout
  train_only_resume_missing_params:
    true
  new trained parameters:
    trm_nested_mha_joint_readout_alpha
    trm_joint_readout_norm
    trm_joint_readout_proj

decision:
  accepted_nested_mha_residual_joint_readout_l4

metrics:
  full_generation_exact: 0.671875
  think0_generation_exact: 0.03125
  full_minus_think0: 0.640625
  full_minus_worst_ablation: 0.14453125
  state_reset_generation_exact: 0.0546875
  op_zero_generation_exact: 0.03515625
  z_l_zero_generation_exact: 0.52734375
  z_h_zero_generation_exact: 0.0

comparison:
  accepted nested MHA/ETD L4 reference:
    full_generation_exact: 0.67578125
    full_minus_worst_ablation: 0.140625
  residual joint-readout:
    full_generation_exact: 0.671875
    full_minus_worst_ablation: 0.14453125

interpretation:
  The residual joint-readout is not a large accuracy improvement, but it is a
  conservative accepted architecture change: it preserves the accepted L4
  exactness level while making the state/readout coupling slightly stronger.
  It is now the next candidate to test on L5/L6, not a final replacement.
```

## Experiment 2026-05-15T00:45:00

```text
topic:
  Test whether the accepted L4 nested residual joint-readout can promote to L5
  multi-family reasoning.

infrastructure fix:
  L4 -> L5 resume initially failed because the checkpoint tokenizer chars did
  not match the multi-family tokenizer. The flexible loader now remaps
  token_embed.weight and lm_head.weight by token string when
  --resume-allow-missing is set, instead of unsafe row-prefix copying.

verification:
  .venv/bin/python -m py_compile scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  .venv/bin/python -m unittest tests.test_qtrm_native_mixed_text_reasoning_probe.QTRMNativeMixedTextReasoningProbeTests.test_flexible_load_remaps_vocab_rows_by_token tests.test_qtrm_native_mixed_text_reasoning_probe.QTRMNativeMixedTextReasoningProbeTests.test_flexible_resume_prefix_copies_position_embeddings tests.test_qtrm_native_mixed_text_reasoning_probe.QTRMNativeMixedTextReasoningProbeTests.test_flexible_load_can_tail_shift_new_position_rows -v
  result: 3 tests OK
```

Direct L4 residual -> L5 multi-family transfer:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_multifamily_seed337_vocabremap_20260515_0028/report.json

resume:
  local_eval/qtrm_native_l4_nested_mha_residual_joint_readout_from_accepted_20260515_001357/last.pt

result:
  decision: rejected
  full_generation_exact: 0.0859375
  full_minus_think0: 0.07421875
  full_minus_worst_ablation: 0.0625
  min_family_generation_exact: 0.023529411764705882

resume_load_summary:
  token_embed/lm_head copied 21 existing token rows by token_remap
  new target tokens: c, d, h, i, k, m, u, v

interpretation:
  The vocab expansion path is now correct, but a single-family L4 residual
  checkpoint does not directly transfer to L5 multi-family reasoning.
```

Attach nested residual path to an accepted L5 single recurrent checkpoint:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_from_l5_single_missingonly_seed339_20260515_0035/report.json

resume:
  local_eval/qtrm_native_l5_multifamily_seed339_single_modchain_weak_repair_s1500_20260514/last.pt

training:
  --train-only-resume-missing-params
  train only newly missing nested residual / z_L / z_H parameters

periodic best:
  step: 500
  generation_exact: 0.734375
  min_family_generation_exact: 0.5581395348837209

final decision:
  rejected
  reject_reasons: ablation_drop_below_threshold

decisive:
  full_generation_exact: 0.68359375
  think0_generation_exact: 0.02734375
  full_minus_think0: 0.65625
  full_minus_worst_ablation: -0.01171875
  min_family_generation_exact: 0.5232558139534884
  z_l_zero_generation_exact: 0.6953125
  z_h_zero_generation_exact: 0.0

interpretation:
  The accepted L5 single path can carry the accuracy while the added nested
  residual path is present, but z_L is not causally required. This is not a
  valid dual-state promotion. The next bottleneck is not text generation
  format or vocab transfer; it is making z_L a genuine intermediate state
  carrier under the L5 family gate.
```

z_L causal repair:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair_seed339_20260515_0048/report.json

recipe:
  resume from:
    local_eval/qtrm_native_l5_residual_joint_readout_from_l5_single_missingonly_seed339_20260515_0035/last.pt
  train_param_name_regex:
    ^trm_
  z_l_counterfactual_loss_weight:
    0.50
  z_l_counterfactual_margin:
    0.30

decision:
  rejected by ablation_drop_below_threshold

decisive:
  full_generation_exact: 0.68359375
  full_minus_worst_ablation: 0.05078125
  min_family_generation_exact: 0.5294117647058824
  z_l_zero_generation_exact: 0.6328125

interpretation:
  Directionally useful: z_l_zero dropped from 0.6953125 to 0.6328125 and the
  ablation margin moved from -0.01171875 to 0.05078125, but it still missed the
  0.10 strict causal threshold.
```

z_L causal repair 2:

```text
report:
  local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair2_seed339_20260515_0056/report.json

recipe:
  resume from:
    local_eval/qtrm_native_l5_residual_joint_readout_zl_causal_repair_seed339_20260515_0048/last.pt
  train_param_name_regex:
    ^trm_
  z_l_counterfactual_loss_weight:
    1.0
  z_l_counterfactual_margin:
    0.50
  lr:
    5e-5

decision:
  accepted_l5_residual_joint_readout_zl_causal_repair2

decisive:
  full_generation_exact: 0.703125
  think0_generation_exact: 0.02734375
  full_minus_think0: 0.67578125
  full_minus_worst_ablation: 0.296875
  min_family_generation_exact: 0.5581395348837209
  state_reset_generation_exact: 0.02734375
  op_zero_generation_exact: 0.01953125
  z_l_zero_generation_exact: 0.40625
  z_h_zero_generation_exact: 0.0

interpretation:
  This is the first accepted L5 result for the nested residual joint-readout
  route with z_L causality restored. The key was not Fast-Slow; it was
  preserving the accepted L5 single path, attaching the nested residual
  dual-state path, and then applying a stronger z_L counterfactual only to
  TRM parameters.
```

## Experiment 2026-05-15T01:35:00

```text
topic:
  Check whether the L5 nested residual z_L-causal result is seed-stable and
  whether it transfers to len6.
```

Seed338 reproduction:

```text
direct attach + strong z_L repair:
  report:
    local_eval/qtrm_native_l5_residual_joint_readout_seed338_direct_zl_repair_20260515_0110/report.json
  resume:
    local_eval/qtrm_native_l5_multifamily_seed338_single_modchain_weak_repair_s1500_20260514/last.pt
  decision:
    rejected
  full_generation_exact:
    0.74609375
  min_family_generation_exact:
    0.5882352941176471
  z_l_zero_generation_exact:
    0.75
  full_minus_worst_ablation:
    -0.00390625

second-stage repair:
  report:
    local_eval/qtrm_native_l5_residual_joint_readout_seed338_zl_repair2_20260515_0118/report.json
  decision:
    rejected
  full_generation_exact:
    0.75
  min_family_generation_exact:
    0.611764705882353
  z_l_zero_generation_exact:
    0.734375
  full_minus_worst_ablation:
    0.015625

interpretation:
  Seed338 keeps strong L5 accuracy but does not make z_L causal. Therefore the
  seed339 acceptance is a valid local result but not yet seed-stable.
```

L6 transfer from the accepted seed339 z_L-causal checkpoint:

```text
zero-shot init:
  report:
    local_eval/qtrm_native_l6_len6_from_l5_zl_causal_seed339_init_20260515_0126/report.json
  decision:
    rejected
  full_generation_exact:
    0.0234375
  min_family_generation_exact:
    0.0
  full_minus_worst_ablation:
    -0.0078125

fine-tune:
  report:
    local_eval/qtrm_native_l6_len6_from_l5_zl_causal_finetune_s2500_20260515_0130/report.json
  decision:
    rejected
  best_periodic_generation_exact:
    0.08854166666666667
  full_generation_exact:
    0.0703125
  min_family_generation_exact:
    0.03125
  full_minus_worst_ablation:
    0.04427083333333333

comparison:
  previous single L6 hard-family balance best:
    full_generation_exact: 0.12239583333333333
```

Decision:

```text
Do not promote nested residual z_L-causal as the canonical L6 route. It is an
accepted L5 seed339 candidate only. The next real bottlenecks are:
  1. seed-stable z_L causality;
  2. len6 ordered transition generalization.
```

## Experiment 2026-05-15T02:00:00

```text
topic:
  Repair seed-stable z_L causality using core-step codec supervision.

hypothesis:
  z_L counterfactual pressure alone can fail by keeping the answer path in the
  base recurrent state while z_L remains non-causal. Adding a core-step codec
  loss on z_L should make z_L carry intermediate causal state, while the final
  answer remains normal LM logits.
```

Seed338 codec repair:

```text
report:
  local_eval/qtrm_native_l5_seed338_zl_codec_repair_final_s1200_20260515_0145/report.json

resume:
  local_eval/qtrm_native_l5_residual_joint_readout_seed338_zl_repair2_20260515_0118/last.pt

recipe:
  train_param_name_regex: ^trm_
  z_l_counterfactual_loss_weight: 1.5
  z_l_counterfactual_margin: 0.70
  core_step_codec_loss_weight: 0.35
  core_step_codec_state_source: l
  restore_best_eval_checkpoint: false

decision:
  accepted_l5_seed338_zl_codec_repair

decisive:
  full_generation_exact: 0.75390625
  think0_generation_exact: 0.01171875
  full_minus_think0: 0.7421875
  full_minus_worst_ablation: 0.28125
  min_family_generation_exact: 0.6235294117647059
  z_l_zero_generation_exact: 0.47265625
  z_h_zero_generation_exact: 0.0
```

Seed337 codec direct repair:

```text
report:
  local_eval/qtrm_native_l5_seed337_zl_codec_direct_repair_s1200_20260515_0152/report.json

resume:
  local_eval/qtrm_native_l5_multifamily_hardbalanced_sweep/seed_337/last.pt

recipe:
  train_only_resume_missing_params: true
  z_l_counterfactual_loss_weight: 1.5
  z_l_counterfactual_margin: 0.70
  core_step_codec_loss_weight: 0.35
  core_step_codec_state_source: l

decision:
  accepted_l5_seed337_zl_codec_direct_repair

decisive:
  full_generation_exact: 0.703125
  think0_generation_exact: 0.015625
  full_minus_think0: 0.6875
  full_minus_worst_ablation: 0.23828125
  min_family_generation_exact: 0.5465116279069767
  z_l_zero_generation_exact: 0.46484375
  z_h_zero_generation_exact: 0.0
```

Seed-stability update:

```text
accepted L5 nested residual / z_L-causal checkpoints:
  seed337: accepted_l5_seed337_zl_codec_direct_repair
  seed338: accepted_l5_seed338_zl_codec_repair
  seed339: accepted_l5_residual_joint_readout_zl_causal_repair2

interpretation:
  The L5 dual/nested route is now seed-stable enough to promote as the active
  L5 candidate. The essential recipe is residual joint-readout plus z_L
  counterfactual pressure; core-step codec on z_L is the stabilizer for seeds
  where z_L otherwise remains non-causal.
```

## Experiment 2026-05-15T02:45:00

```text
topic:
  Re-evaluate L6 after noticing a depth/length mismatch.

finding:
  Previous len6 experiments used program_len=6 but left train_think_steps and
  eval_think_steps at the script default 4. That made the model solve a
  six-operation program with only four recurrent steps.
```

Nested residual z_L-codec with depth6:

```text
report:
  local_eval/qtrm_native_l6_len6_seed338_zl_codec_depth6_s3000_20260515_0215/report.json

recipe:
  resume:
    local_eval/qtrm_native_l5_seed338_zl_codec_repair_final_s1200_20260515_0145/last.pt
  program_len:
    6
  train_think_steps/eval_think_steps:
    6
  core_step_codec_state_source:
    l

decision:
  rejected

best_periodic_generation_exact:
  0.11458333333333333

final_decisive:
  full_generation_exact: 0.08072916666666667
  min_family_generation_exact: 0.0390625
  full_minus_worst_ablation: 0.03906250000000001
```

Nested residual continuation:

```text
report:
  local_eval/qtrm_native_l6_len6_seed338_zl_codec_depth6_cont_s3000_20260515_0228/report.json

decision:
  rejected

final_decisive:
  full_generation_exact: 0.07552083333333333
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.03645833333333333

interpretation:
  Adding the L5 dual/nested z_L-codec path does not currently help len6.
```

Single recurrent depth6 baseline:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/report.json

resume:
  local_eval/qtrm_native_l5_multifamily_seed338_single_modchain_weak_repair_s1500_20260514/last.pt

recipe:
  think_structure: single
  program_len: 6
  train_think_steps: 6
  eval_think_steps: 6

decision:
  accepted_l6_len6_single_seed338_depth6

decisive:
  full_generation_exact: 0.3671875
  think0_generation_exact: 0.0
  full_minus_think0: 0.3671875
  full_minus_worst_ablation: 0.34375
  min_family_generation_exact: 0.0703125
  state_reset_generation_exact: 0.0078125
  op_zero_generation_exact: 0.0234375

comparison:
  previous single L6 hard-family balance with default depth4:
    full_generation_exact: 0.12239583333333333
  nested residual z_L-codec depth6:
    full_generation_exact: 0.08072916666666667
```

Decision:

```text
L6 is not blocked by language output formatting. The immediate fix was
depth/length alignment. The current active L6 scaffold is single recurrent
MHA/ETD with train/eval think_steps=program_len=6. The dual/nested z_L-codec
path remains the L5 candidate but should not be forced into L6 until the single
depth6 transition scaffold is stable across seeds/families.
```

## Experiment 2026-05-15T03:20:00

```text
topic:
  Continue the accepted L6 single recurrent depth6 scaffold with a lower lr.

hypothesis:
  If the L6 scaffold is still undertrained, continuing from the accepted
  checkpoint at lr=1e-5 should raise exactness or the weak family floor without
  changing architecture.
```

Run:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_cont_s3000_lr1e5_20260515_011502/report.json

resume:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt

recipe:
  think_structure: single
  program_len: 6
  train_think_steps: 6
  eval_think_steps: 6
  lr: 1e-5
  steps: 3000
```

Decision:

```text
accepted by the diagnostic L6 thresholds, but not an improvement over the
initial accepted checkpoint.
```

Decisive:

```text
base full_generation_exact: 0.3671875
continued full_generation_exact: 0.3541666666666667
base min_family_generation_exact: 0.0703125
continued min_family_generation_exact: 0.0703125
continued full_minus_worst_ablation: 0.3229166666666667

continued family exact:
  checksum: 0.90625
  modchain: 0.0859375
  revchain: 0.0703125
```

Conclusion:

```text
Plain longer training is not the missing ingredient. L6 remains accepted only
as a weak scaffold; the next bottleneck is weak-family floor repair for
modchain/revchain while preserving the causal depth6 path.
```

## Experiment 2026-05-15T03:35:00

```text
topic:
  L6 single depth6 weak-family floor repair.

hypothesis:
  Since checksum is already high but modchain/revchain are weak, a stronger
  family-DRO objective and depth-intermediate family-DRO may raise the
  min-family floor without changing the token->core->logits path.
```

First repair:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_family_floor_repair_s2500_20260515_011837/report.json

recipe:
  resume:
    local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt
  family_dro_loss_weight:
    0.25
  depth_intermediate_family_dro:
    true
  restore_best_eval_checkpoint:
    true
  accept_min_family_exact:
    0.08

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.3567708333333333
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.3359375

family exact:
  checksum: 0.90625
  modchain: 0.078125
  revchain: 0.0859375
```

Second modchain-focused repair:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_modchain_floor_repair_s1500_20260515_012038/report.json

recipe:
  resume:
    local_eval/qtrm_native_l6_len6_single_seed338_family_floor_repair_s2500_20260515_011837/last.pt
  family_dro_loss_weight:
    0.35
  lr:
    5e-6
  restore_best_eval_checkpoint:
    true

decision:
  rejected

result:
  best checkpoint was the initial checkpoint, so stronger DRO did not improve
  over the first repair.
```

Conclusion:

```text
Family-DRO gives a small real signal, raising the floor from 0.0703125 to
0.078125, but it does not cross the 0.08 improvement threshold. Do not keep
raising DRO. The next informative step is seed-stability for the aligned
single-depth6 scaffold, then a more principled transition/objective repair.
```

## Experiment 2026-05-15T03:50:00

```text
topic:
  L6 single recurrent depth6 seed-stability check on seed339.

hypothesis:
  If the depth/length alignment fix is real rather than seed338-local, the
  same single recurrent depth6 recipe should pass at least the diagnostic L6
  thresholds on another retained L5 single checkpoint.
```

Run:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_depth6_s3000_20260515_012244/report.json

resume:
  local_eval/qtrm_native_l5_multifamily_seed339_single_modchain_weak_repair_s1500_20260514/last.pt

recipe:
  think_structure: single
  program_len: 6
  train_think_steps: 6
  eval_think_steps: 6
  lr: 2e-5
  steps: 3000
```

Decision:

```text
accepted_l6_len6_single_seed339_depth6
```

Decisive:

```text
full_generation_exact: 0.15364583333333334
think0_generation_exact: 0.013020833333333334
full_minus_think0: 0.140625
full_minus_worst_ablation: 0.1328125
min_family_generation_exact: 0.0546875
state_reset_generation_exact: 0.020833333333333332
op_zero_generation_exact: 0.005208333333333333

family exact:
  checksum: 0.3359375
  modchain: 0.0546875
  revchain: 0.0703125
```

Conclusion:

```text
Aligned single recurrent depth6 is not a one-seed artifact: seed338 and seed339
both pass the diagnostic L6 thresholds with causal ablation drops. However,
seed339 is much weaker than seed338, so L6 is not seed-stable enough to promote
as a strong architecture level. The next bottleneck is robust transition
generalization, not another plain continuation.
```

## Experiment 2026-05-15T04:05:00

```text
topic:
  Core-step codec on the aligned L6 single depth6 scaffold.

hypothesis:
  If the L6 bottleneck is hidden-state transition readability, a training-only
  core-step codec on `core_state_trace_h` should improve the weak family floor
  while the final accepted metric remains normal LM generation.
```

Run:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_corecodec_h_s2500_20260515_012658/report.json

resume:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt

recipe:
  core_step_codec_loss_weight: 0.25
  core_step_codec_state_source: h
  core_step_codec_pooling: last
  restore_best_eval_checkpoint: true
  accept_min_family_exact: 0.08
```

Decision:

```text
rejected
```

Decisive:

```text
full_generation_exact: 0.3671875
min_family_generation_exact: 0.0703125
full_minus_worst_ablation: 0.34375
reject_reason: family_exact_below_threshold
restore_best_eval_checkpoint: true
best checkpoint: initial checkpoint
```

Conclusion:

```text
Core-step codec on the single h-state does not improve the aligned L6 scaffold.
Do not keep sweeping codec weights on this path. The next transition objective
should train consistency across active program prefixes through the same LM
answer path, not an auxiliary state reader.
```

## Experiment 2026-05-15T04:20:00

```text
topic:
  Active-length replay on the aligned L6 single recurrent depth6 scaffold.

hypothesis:
  If the weak L6 families fail because the LM path only sees full-length
  supervision, bounded replay over active prefixes should improve the family
  floor while preserving the canonical token->core->logits path.
```

Run:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_active_replay_s2000_20260515_012955/report.json

resume:
  local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt

recipe:
  active_len_replay_loss_weight: 0.02
  active_len_replay_min: 1
  active_len_replay_max: 6
  active_len_replay_max_cases: 16
  active_len_replay_every: 4
  restore_best_eval_checkpoint: true
  train/eval_think_steps: 6
```

Decision:

```text
rejected
```

Decisive:

```text
full_generation_exact: 0.3671875
min_family_generation_exact: 0.0703125
full_minus_worst_ablation: 0.34375
state_reset_generation_exact: 0.0078125
op_zero_generation_exact: 0.0234375
reject_reason: family_exact_below_threshold
best checkpoint: initial checkpoint
```

Conclusion:

```text
Simple active-length replay does not repair L6. The next candidate should not
be more prefix-replay weight sweeping. It should add a structural transition
carrier that remains inside the normal prompt-token -> recurrent core -> LM
logit path, for example a visible prompt-state anchor smoke first and then a
learned internal state/scratch token if the visible anchor is not sufficient.
```

## Experiment 2026-05-15T04:45:00

```text
topic:
  L6 transition-carrier comparison: visible prompt anchor vs internal
  single-core carrier.

orthodox-method audit:
  prior family:
    recurrent state carrier / causal sequence-state update.
  canonical path:
    prompt tokens -> native embeddings -> encoder -> mandatory recurrent core
    -> decoder -> LM logits. No solver, renderer, retrieval, or hidden answer
    channel.
  shortcut exclusion:
    the carrier only changes latent state inside the recurrent core. The final
    answer remains greedy autoregressive text from the normal LM head.
```

Visible prompt-state anchor:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed338_prompt_anchor_s2500_20260515_013530/report.json

recipe:
  resume:
    local_eval/qtrm_native_l6_len6_single_seed338_depth6_s3000_20260515_0242/last.pt
  prompt_state_anchor: true
  prompt_state_anchor_position: before_answer
  pos_embed_resize_strategy: tail_shift
  train/eval_think_steps: 6

decision:
  rejected

decisive:
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.30729166666666663
  reject_reason: family_exact_below_threshold
```

Internal single-core carrier:

```text
implementation:
  scripts/335_train_qtrm_native_etd_probe.py
  think_structure: single_core_carrier

mechanism:
  Each single recurrent think step runs the normal think block, then passes
  concat(current_state, encoded_prompt_state) through a causal GRU carrier and
  adds the gated carrier state back into the recurrent state before decode.

report:
  local_eval/qtrm_native_l6_len6_single_seed338_single_core_carrier_s2500_20260515_013530/report.json

decision:
  accepted_l6_len6_single_seed338_single_core_carrier

decisive:
  full_generation_exact: 0.3671875
  think0_generation_exact: 0.0026041666666666665
  full_minus_think0: 0.3645833333333333
  full_minus_worst_ablation: 0.3307291666666667
  min_family_generation_exact: 0.09375
  state_reset_generation_exact: 0.0026041666666666665
  op_zero_generation_exact: 0.036458333333333336

family exact:
  checksum: 0.9140625
  modchain: 0.09375
  revchain: 0.09375
```

Conclusion:

```text
The first useful L6 transition-carrier result is internal, not textual.
Prompt-state anchoring nearly reaches the family-floor threshold but does not
pass. The internal `single_core_carrier` raises the weak-family floor from the
base 0.0703125 to 0.09375 while preserving strong state-reset/op-zero
ablation drops. Promote it as the active L6 scaffold, but do not call L6
solved until a second seed or a longer-length gate confirms it.
```

## Experiment 2026-05-15T05:10:00

```text
topic:
  Seed339 stability check for the internal single-core carrier.
```

Standard carrier init:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_s2500_20260515_014403/report.json

resume:
  local_eval/qtrm_native_l6_len6_single_seed339_depth6_s3000_20260515_012244/last.pt

decision:
  rejected

decisive:
  full_generation_exact: 0.19010416666666666
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.1640625
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.026041666666666668
```

Near-off gate init diagnostic:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_gateoff_s2500_20260515_014659/report.json

decision:
  rejected

reject_reason:
  full_exact_below_threshold

decisive:
  full_generation_exact: 0.12760416666666666
  min_family_generation_exact: 0.0859375
  full_minus_worst_ablation: 0.10156249999999999
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.020833333333333332
```

Conclusion:

```text
`single_core_carrier` is a real seed338 improvement but not yet seed-stable.
On seed339 the standard carrier improves full exact over the seed339 base but
does not lift the family floor; the near-off gate lifts the family floor but
loses too much overall exact. Keep the seed338 carrier checkpoint as the active
L6 scaffold, but treat stability as the next bottleneck. Do not promote this
to a solved L6 architecture until either seed339 is repaired or another seed
passes under the same gate.
```

Family-DRO 0.15 follow-up:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_famdro015_s2500_20260515_014659/report.json

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.19270833333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.15364583333333334
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.0390625
```

Updated conclusion:

```text
Seed339 is not fixed by simply increasing family-DRO. The next stability fix
should change the optimization curriculum or carrier supervision, not keep
sweeping family-DRO weights.
```

## Experiment 2026-05-15T05:35:00

```text
topic:
  Third-seed check for the internal single-core carrier.
```

Seed337 baseline:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed337_depth6_s3000_20260515_015547/report.json

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.23697916666666666
  min_family_generation_exact: 0.03125
  full_minus_think0: 0.20833333333333331
  full_minus_worst_ablation: 0.19010416666666666
```

Seed337 carrier:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed337_single_core_carrier_s2500_20260515_015547/report.json

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.1640625
  think0_generation_exact: 0.036458333333333336
  full_minus_think0: 0.12760416666666666
  full_minus_worst_ablation: 0.140625
  min_family_generation_exact: 0.0546875
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.0234375

family exact:
  checksum: 0.3671875
  modchain: 0.0703125
  revchain: 0.0546875
```

Conclusion:

```text
`single_core_carrier` remains seed338-only. It improves the weak-family floor
relative to the seed337 baseline, but it lowers full exact and still misses the
0.08 family floor. Therefore it is not a seed-stable L6 solution. Treat it as
a useful transition-state idea, not a canonical fixed architecture.

`trm_official_prenorm` is a backbone/block implementation choice, not a
decision to abandon dual or nested TRM schedules. Dual/nested remains a macro
structure to re-test after the transition-state training problem is repaired.
Do not promote single-carrier over dual/nested by naming alone; promote only by
standard runs plus ablations.
```

## Experiment 2026-05-15T06:04:18

```text
topic:
  Carrier plus h-state mean-pooled step codec on seed339.
```

Run:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hmean_w008_s2500_20260515_020418/report.json

recipe:
  resume:
    local_eval/qtrm_native_l6_len6_single_seed339_depth6_s3000_20260515_012244/last.pt
  think_structure: single_core_carrier
  core_step_codec_loss_weight: 0.08
  core_step_codec_state_source: h
  core_step_codec_pooling: mean
  train/eval_think_steps: 6

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.19270833333333334
  think0_generation_exact: 0.013020833333333334
  full_minus_think0: 0.1796875
  full_minus_worst_ablation: 0.16666666666666669
  min_family_generation_exact: 0.0546875
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.026041666666666668

family exact:
  checksum: 0.4609375
  modchain: 0.0546875
  revchain: 0.0625

best periodic:
  step: 2000
  generation_exact: 0.21354166666666666
  min_family_generation_exact: 0.0625
```

Conclusion:

```text
Mean-pooled h-state codec does not repair seed339 carrier stability. It keeps
the recurrent ablation signal and improves checksum, but the hard families are
still below the 0.08 floor. The next narrow check is last-token h-state codec,
because the LM answer path reads from the prompt/answer boundary more directly
than from a mean-pooled prompt representation.
```

Last-token codec follow-up:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_codec_hlast_w008_s2500_20260515_020735/report.json

recipe:
  think_structure: single_core_carrier
  core_step_codec_loss_weight: 0.08
  core_step_codec_state_source: h
  core_step_codec_pooling: last

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.2109375
  think0_generation_exact: 0.018229166666666668
  full_minus_think0: 0.19270833333333334
  full_minus_worst_ablation: 0.1875
  min_family_generation_exact: 0.046875
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.020833333333333332

family exact:
  checksum: 0.5078125
  modchain: 0.046875
  revchain: 0.078125
```

Conclusion:

```text
Last-token codec improves full exact over the standard seed339 carrier but
worsens the minimum family floor. Codec supervision is therefore not enough:
it teaches a readable state for easier/favored families without forcing robust
ordered transition generalization. Stop codec-pooling variants here. The next
candidate is direct prefix/full state consistency on the recurrent trajectory.
```

Prefix/full state alignment:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_align_w002_s2500_20260515_021141/report.json

recipe:
  think_structure: single_core_carrier
  prefix_state_alignment_loss_weight: 0.02
  prefix_state_alignment_max_cases: 8
  prefix_state_alignment_every: 4

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.20572916666666666
  think0_generation_exact: 0.018229166666666668
  full_minus_think0: 0.1875
  full_minus_worst_ablation: 0.18229166666666666
  min_family_generation_exact: 0.046875
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.018229166666666668

family exact:
  checksum: 0.5
  modchain: 0.046875
  revchain: 0.0703125
```

Conclusion:

```text
Direct MSE alignment between full and prefix recurrent state does not repair
the weak family. It keeps the causal ablation signal, but like the codec
variants it mainly improves checksum/aggregate exact. The next existing
state-level candidate is contrastive prefix/full alignment so the state is not
only close to its prefix target but also separated from other cases.
```

Contrastive prefix/full state alignment:

```text
report:
  local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_prefix_contrast_w002_s2500_20260515_021514/report.json

recipe:
  think_structure: single_core_carrier
  prefix_state_contrastive_loss_weight: 0.02
  prefix_state_contrastive_max_cases: 16
  prefix_state_contrastive_every: 4
  prefix_state_contrastive_state_source: h
  prefix_state_contrastive_pooling: last

decision:
  rejected

reject_reason:
  family_exact_below_threshold

decisive:
  full_generation_exact: 0.19270833333333334
  think0_generation_exact: 0.013020833333333334
  full_minus_think0: 0.1796875
  full_minus_worst_ablation: 0.16666666666666669
  min_family_generation_exact: 0.0546875
  state_reset_generation_exact: 0.026041666666666668
  op_zero_generation_exact: 0.026041666666666668

family exact:
  checksum: 0.453125
  modchain: 0.0546875
  revchain: 0.0703125
```

Conclusion:

```text
Contrastive state alignment also fails. Stop auxiliary state-reader/alignment
losses for this seed339 carrier repair. The repeated pattern suggests the
carrier is being introduced as a random new module into an already trained
single-core model and is not being integrated stably. The next minimal
curriculum is a carrier warmup: train only the newly added `single_carrier_*`
parameters first, then run a short full fine-tune.
```

Carrier warmup / gate-init / soup follow-up:

```text
carrier warmup only:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_warmup_missing_s1500_20260515_021914/report.json
  decision: rejected
  full_generation_exact: 0.17708333333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.15104166666666669

carrier warmup then full fine-tune:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_warmup_then_full_s1500_20260515_022043/report.json
  decision: rejected
  full_generation_exact: 0.17708333333333334
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.15104166666666669

carrier_gate_init=-4.0:
  report:
    local_eval/qtrm_native_l6_len6_single_seed339_single_core_carrier_gateinit_m4_s2500_20260515_022504/report.json
  decision: rejected
  reject_reason: full_exact_below_threshold
  full_generation_exact: 0.12760416666666666
  min_family_generation_exact: 0.0859375
  full_minus_worst_ablation: 0.10156249999999999
  family exact:
    checksum: 0.2109375
    modchain: 0.0859375
    revchain: 0.0859375

checkpoint soup standard/gateinit_m4:
  alpha0.1:
    report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p1_eval_20260515/report.json
    decision: rejected
    full_generation_exact: 0.12760416666666666
    min_family_generation_exact: 0.0546875
  alpha0.2:
    report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p2_eval_20260515/report.json
    decision: rejected
    full_generation_exact: 0.09895833333333333
    min_family_generation_exact: 0.0625
  alpha0.3:
    report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p3_eval_20260515/report.json
    decision: rejected
    full_generation_exact: 0.08072916666666667
    min_family_generation_exact: 0.0625
  alpha0.4:
    report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha0p4_eval_20260515/report.json
    decision: rejected
    full_generation_exact: 0.0703125
    min_family_generation_exact: 0.0546875
  alpha0.5:
    report: local_eval/qtrm_native_l6_seed339_carrier_soup_standard_gateinit_m4_alpha05_eval_20260515/report.json
    decision: rejected
    full_generation_exact: 0.07291666666666667
    min_family_generation_exact: 0.0625
```

Decision update:

```text
`single_core_carrier` remains a causal transition-state scaffold, but not a
seed-stable L6 architecture. Warmup, auxiliary state supervision, gate init,
and checkpoint soup all failed to satisfy full exact and family floor together.

The useful diagnostic split is now clear:
  standard carrier: higher full exact, weak family floor;
  near-off gate: better family floor, lower full exact.

Therefore the next orthodox step should not be more scalar blending or
backbone renaming. It should either:
  1. return to the dual/nested macro schedule and inject a carrier/state path
     there under the same L6 gate; or
  2. redesign the recurrent transition objective so hard-family state updates
     are learned before final LM rendering.

`trm_official_prenorm` remains only a pre-norm official-TRM-style backbone
choice. It does not mean dual or nested TRM has been abandoned.
```

### 2026-05-15 - L6 Dual/Nested Family-Floor Audit And Eval-Order Fix

Key finding:

```text
The L6 dual/nested residual joint readout path can preserve strong causal
dependence on the recurrent core, but it saturates just below the strict
family-floor gate.
```

New guardrail:

```text
Added --eval-family-order-invariant to
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py.

Reason:
  eval cases were previously generated by a single RNG while cycling families.
  Therefore changing eval family order changed the per-family held-out samples.
  That can create a proxy pass/fail signal near the 0.08 family threshold.

Test:
  tests.test_qtrm_native_mixed_text_reasoning_probe.
  QTRMNativeMixedTextReasoningProbeTests.
  test_order_invariant_eval_cases_keep_per_family_samples_stable
```

Canonical order-invariant re-eval:

```text
z_L causal repair:
  report:
    local_eval/qtrm_native_l6_seed338_zl_causal_repair_order_invariant_reeval_20260515/report.json
  decision: rejected
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.3125
  z_l_zero_generation_exact: 0.0026041666666666665
  by_family:
    checksum: 0.8671875
    modchain: 0.0859375
    revchain: 0.0703125

hard-op repair:
  report:
    local_eval/qtrm_native_l6_seed338_hardop_order_invariant_reeval_20260515/report.json
  decision: rejected
  full_generation_exact: 0.3463541666666667
  min_family_generation_exact: 0.0703125
  full_minus_worst_ablation: 0.3151041666666667
  z_l_zero_generation_exact: 0.0
  by_family:
    checksum: 0.8828125
    modchain: 0.0859375
    revchain: 0.0703125
```

Repair attempts:

```text
revchain hard-op repair:
  report:
    local_eval/qtrm_native_l6_seed338_nested_residual_joint_readout_revchain_hardop_repair_trmonly_s1500_20260515/report.json
  decision: rejected
  full_generation_exact: 0.3619791666666667
  min_family_generation_exact: 0.0625
  full_minus_worst_ablation: 0.3385416666666667
  by_family:
    checksum: 0.9140625
    modchain: 0.109375
    revchain: 0.0625

order-invariant full-periodic repair:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_revchain_fullperiodic_repair_s800_20260515/report.json
  decision: rejected
  best_periodic_step: 100
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.078125
  full_minus_worst_ablation: 0.3125
  z_l_zero_generation_exact: 0.0
  by_family:
    checksum: 0.84375
    modchain: 0.0859375
    revchain: 0.078125

one-case continuation:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_revchain_onecase_repair_s400_20260515/report.json
  decision: rejected
  min_family_generation_exact: 0.078125
```

Decision:

```text
The current dual/nested residual joint readout is not accepted, but it is the
best causal L6 dual/nested scaffold so far:
  core ablations are strong;
  z_L zeroing is destructive;
  depth 6 is necessary;
  family floor is one held-out revchain case below threshold.

Do not keep sweeping scalar loss weights. The next architecture change should
target family-floor saturation directly:
  1. make eval-family-order-invariant the canonical L6 gate;
  2. add a transition-state objective that improves reverse-order binding
     without trading away modchain;
  3. test with the same order-invariant full/family/depth/ablation gate.
```

### 2026-05-15 - L6 Dual/Nested Core-Carrier Seed338 Pass

Terminology correction:

```text
`trm_official_prenorm backbone` is a local block/backbone choice. It does not
replace the macro topology. Dual/nested remains controlled by think_structure,
for example:

  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier

The current candidate is still QTRM-native:
  prompt tokens -> native embedding -> dual/nested recurrent z_L/z_H core
  -> core-dependent readout -> LM logits -> autoregressive text.
```

Failed auxiliary repairs:

```text
op-codec repair:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_opcodec_revchain_repair_s500_20260515/report.json
  decision: rejected; restore-best selected the initial checkpoint

position-codec repair:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_positioncodec_revchain_repair_s500_20260515/report.json
  decision: rejected; restore-best selected the initial checkpoint

Conclusion:
  operation-id and operation-position auxiliary losses did not solve the
  transition-to-answer saturation. The next useful change had to be structural.
```

New structural candidate:

```text
name:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier

class:
  faithful QTRM-native adaptation / original transition-state hypothesis

mechanism:
  keep the dual/nested residual joint-readout path;
  add an identity-safe in-core GRU carrier from [z_L, z_H, encoded];
  inject only a small normalized carrier delta into z_L/z_H;
  do not normalize the whole loaded baseline state.

bug fixed:
  the first carrier version normalized the full base state and broke warm-start.
  The fixed version preserves identity when the carrier gate is near zero.
```

Order-invariant eval:

```text
identity-safe carrier, gate_init=-4.0:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_nested_core_carrier_identity_init_carrieroff_eval2_20260515/report.json
  decision:
    accepted_l6_nested_core_carrier_identity_init
  full_generation_exact:
    0.3411458333333333
  min_family_generation_exact:
    0.09375
  by_family:
    checksum: 0.8203125
    modchain: 0.09375
    revchain: 0.109375
  full_minus_worst_ablation:
    0.3125
  z_l_zero_generation_exact:
    0.0026041666666666665
  z_h_zero_generation_exact:
    0.0
  carrier_off_generation_exact:
    0.3359375
  full_minus_carrier_off:
    0.005208333333333315

gate_off control:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_nested_core_carrier_gateoff_init_eval_20260515/report.json
  decision:
    rejected; returns to the old near-pass
```

Interpretation:

```text
This is the first strict order-invariant L6 pass for the dual/nested line, but
it is not a global L6 solution yet.

Why it matters:
  the path is still native token -> recurrent core -> LM logits;
  depth 6 is necessary;
  z_L/z_H and reset/op ablations remain destructive;
  carrier_off returns to the previous near-pass, so the small in-core carrier
  accounts for the one-case family-floor rescue.

Why it is provisional:
  the carrier parameters are new missing parameters initialized under seed338;
  the gain over carrier_off is small;
  seed stability and language non-regression are not yet verified.

Next:
  1. run seed stability for the same carrier structure;
  2. train only the carrier/gate narrowly if seed stability fails;
  3. keep full/family/depth/ablation gate unchanged;
  4. only after that promote it beyond seed338 scaffold.
```

### 2026-05-15 - L6 Core-Carrier Seed Stability Rejected

Question:

```text
Should nested and Fast-Slow be fixed as defaults?
```

Answer:

```text
No. Nested remains the active macro candidate, but it is not globally fixed.
Fast-Slow remains a rejected/diagnostic auxiliary unless it passes the same
strict family-floor gate. Do not hard-code either as a promotion default yet.
```

Seed-stability sweep:

```text
summary:
  local_eval/qtrm_native_l6_nested_core_carrier_seed_stability_summary_20260515.json

decision:
  rejected_seed_stability

pass_count:
  1 / 3

seed337:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_identity_init_carrieroff_eval_20260515/report.json
  decision: rejected
  full_generation_exact: 0.3203125
  min_family_generation_exact: 0.0390625
  carrier_off_generation_exact: 0.3333333333333333
  full_minus_carrier_off: -0.013020833333333315
  by_family:
    checksum: 0.859375
    modchain: 0.0390625
    revchain: 0.0625

seed338:
  report:
    local_eval/qtrm_native_l6_seed338_order_invariant_nested_core_carrier_identity_init_carrieroff_eval2_20260515/report.json
  decision: accepted_l6_nested_core_carrier_identity_init
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.09375
  carrier_off_generation_exact: 0.3359375
  full_minus_carrier_off: 0.005208333333333315

seed339:
  report:
    local_eval/qtrm_native_l6_seed339_order_invariant_nested_core_carrier_identity_init_carrieroff_eval_20260515/report.json
  decision: rejected
  full_generation_exact: 0.3307291666666667
  min_family_generation_exact: 0.046875
  carrier_off_generation_exact: 0.3255208333333333
  full_minus_carrier_off: 0.00520833333333337
  by_family:
    checksum: 0.8671875
    modchain: 0.078125
    revchain: 0.046875
```

Carrier-only repair:

```text
report:
  local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_only_repair_s600_20260515/report.json

recipe:
  resume from seed338 near-pass checkpoint;
  train only missing carrier parameters;
  steps: 600;
  lr: 1e-4;
  restore_best_eval_checkpoint by family_floor.

decision:
  rejected

best_periodic_step:
  300

metrics:
  full_generation_exact: 0.34375
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.010416666666666685
  by_family:
    checksum: 0.9140625
    modchain: 0.0625
    revchain: 0.0546875
```

Decision:

```text
The identity-safe in-core carrier is a useful structural signal, not a
promotion-ready default. It can improve full exact and remain causal, but it
does not yet stabilize the weak-family floor across seeds.

Do not freeze Fast-Slow as a default. The previous standard Fast-Slow gate was
rejected, and this seed-stability failure is about family-specific transition
binding, not a proven fast/slow loss default.

Next causal hypothesis:
  preserve the dual/nested macro path;
  make the carrier less random-init-dependent;
  add a deterministic near-identity carrier initialization or trainable gate
  schedule that targets weak-family balance without worsening carrier_off
  controls.
```

### 2026-05-15 - Deterministic Carrier And Seed337 Repair Rejected

Code change:

```text
files:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
  tests/test_qtrm_native_etd_probe.py
  tests/test_qtrm_native_mixed_text_reasoning_probe.py

new option:
  --carrier-state-mode

modes:
  gru
  encoded
  state_mean
  state_delta
  encoded_state_mean

purpose:
  test whether the nested core-carrier failure is mainly random GRU
  initialization dependence.
```

Verification:

```text
.venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

.venv/bin/python -m unittest \
  tests.test_qtrm_native_etd_probe \
  tests.test_qtrm_native_mixed_text_reasoning_probe -v

result:
  195 tests OK
```

Seed337 repair summary:

```text
summary:
  local_eval/qtrm_native_l6_seed337_carrier_brittleness_repair_summary_20260515.json

decision:
  rejected_deterministic_carrier_and_seed337_repairs
```

Deterministic carrier evals:

```text
state_mean:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_state_mean_init_eval_20260515/report.json
  full_generation_exact: 0.2578125
  min_family_generation_exact: 0.0234375
  full_minus_carrier_off: -0.07552083333333331

encoded:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_encoded_init_eval_20260515/report.json
  full_generation_exact: 0.23697916666666666
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: -0.09635416666666666

state_delta:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_state_delta_init_eval_20260515/report.json
  full_generation_exact: 0.296875
  min_family_generation_exact: 0.03125
  full_minus_carrier_off: -0.036458333333333315
```

Learned repair evals:

```text
carrier_only:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_only_repair_s600_20260515/report.json
  full_generation_exact: 0.34375
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.010416666666666685

carrier_only_family_dro:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_core_carrier_only_famdro_repair_s900_20260515/report.json
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.0078125

transition_carrier_low_lr:
  report:
    local_eval/qtrm_native_l6_seed337_order_invariant_nested_transition_carrier_repair_s900_20260515/report.json
  full_generation_exact: 0.3385416666666667
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: -0.0026041666666666297
```

Failure breakdown:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_carrier_failure_opbreakdown_20260515/report.json

finding:
  failures are broad across modchain/revchain operation positions. This is not
  a single hard op that can be fixed by another hard-op replay sweep.
```

Decision:

```text
The bottleneck is no longer "carrier random init" by itself.

Rejected:
  deterministic carrier state replacement;
  carrier-only learning;
  carrier-only family-DRO;
  low-LR transition+carrier tuning with retention.

Current interpretation:
  the seed337/339 failures are a broader answer-space/generalization gap in
  the L6 prompt -> recurrent state -> LM logits path. The core is causal, but
  the recurrent state is not accurate enough on modchain/revchain to cross the
  family floor.

Next direction:
  stop local carrier tuning;
  audit answer-space argmax vs greedy generation on the weak seeds;
  if answer-space rank is high but greedy fails, repair LM readout;
  if answer-space rank is also low, return to recurrent transition learning.
```

Answer-space audit:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_carrier_answer_space_audit_20260515/report.json

metrics:
  generation_exact: 0.3203125
  answer_space_argmax_exact: 0.3177083432674408
  answer_space_gold_mean_rank: 8.098958015441895
  answer_space_gold_top3: 0.4348958432674408
  answer_space_gold_top5: 0.5286458134651184

interpretation:
  answer-space argmax is not materially better than greedy generation, and
  the gold answer is not ranked near the top often enough. Therefore the
  active weak-seed bottleneck is not mainly text rendering or greedy decoding.
  It is recurrent transition learning / state accuracy.
```

Next repair class:

```text
Return to recurrent transition learning. The next candidate should improve
state accuracy before logits, for example a transition-consistency objective
or a stricter depth-wise state supervision path. Do not add another carrier
variant unless it directly improves answer-space rank and weak-family floor
under the same gate.
```

State-trace depth repair result:

```text
reports:
  local_eval/qtrm_native_l6_seed337_nested_state_trace_depth_repair_s900_20260515/report.json
  local_eval/qtrm_native_l6_seed337_nested_state_trace_depth_repair_strong_s1200_20260515/report.json

strong settings:
  think_structure:
    trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier
  state_trace_depth_loss_weight: 1.0
  state_trace_depth_state_source: both
  retention_kl_loss_weight: 0.25
  lr: 5.0e-5
  steps: 1200
  periodic_eval_score_mode: family_floor

strong result:
  decision: rejected
  best_periodic_eval_step: 450
  full_generation_exact: 0.3411458333333333
  min_family_generation_exact: 0.0546875
  full_minus_carrier_off: 0.0
  state_reset_generation_exact: 0.013020833333333334
  op_zero_generation_exact: 0.018229166666666668
  z_l_zero_generation_exact: 0.005208333333333333
  z_h_zero_generation_exact: 0.0
  answer_space_argmax_exact: 0.34375
  answer_space_gold_mean_rank: 7.916666507720947
  answer_space_gold_top3: 0.4557291567325592
  answer_space_gold_top5: 0.5390625

by_family:
  checksum: 0.90625
  modchain: 0.0625
  revchain: 0.0546875

decision:
  The state-trace depth objective slightly improves answer-space rank/top-k
  versus the initial answer-space audit, but it does not cross the weak-family
  floor and the carrier_off metric matches full accuracy. This closes the
  scalar state-trace/carrier repair ladder for seed337.

next hypothesis:
  The remaining weak-seed bottleneck is transition binding inside the
  dual/nested recurrent core. The next experiment should change the recurrent
  transition objective or state update itself, not add another renderer,
  decoder trick, carrier variant, MemoryOS path, or scalar loss sweep.
```

Transition-binding prefix contrast result:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_binding_prefix_contrast_s900_20260515/report.json

path:
  prompt tokens -> native embedding -> dual/nested z_L/z_H recurrent core
  -> in-core carrier -> residual joint readout -> LM logits

mechanism:
  prefix_state_contrastive_loss aligns the depth-wise state from the full
  prompt with the state produced when only the causal prefix up to that depth
  is visible. This is a transition-state objective, not an answer sidecar.

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 750
  full_generation_exact: 0.3333333333333333
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: -0.002604166666666685
  answer_space_argmax_exact: 0.3411458432674408
  answer_space_gold_mean_rank: 7.640625

by_family:
  checksum: 0.90625
  modchain: 0.046875
  revchain: 0.046875

decision:
  Prefix/full contrast improves gold mean rank a little, but it reduces the
  weak-family floor and does not make carrier causally helpful. Do not use
  prefix contrast as the next repair ladder for this bottleneck.

next candidate:
  Try a narrower transition-codec repair that asks the recurrent state to bind
  the operation and operation position at each depth, while still requiring the
  final answer to pass through LM logits. If that also fails, the bottleneck is
  likely architectural state update rather than auxiliary objective choice.
```

Transition op/position codec repair result:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_transition_codec_repair_s700_20260515/report.json

mechanism:
  core_step_codec_loss + core_step_op_codec_loss +
  core_step_position_codec_loss train the recurrent state to expose depth-wise
  answer/op/position information. The codec heads are training probes only;
  acceptance still requires normal LM-logit generation.

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 500
  full_generation_exact: 0.3125
  min_family_generation_exact: 0.0625
  full_minus_carrier_off: -0.010416666666666685
  answer_space_argmax_exact: 0.3177083432674408
  answer_space_gold_mean_rank: 7.919270992279053

by_family:
  checksum: 0.8125
  modchain: 0.0625
  revchain: 0.0625

decision:
  The codec repair does not cross the 0.08 family floor and degrades full
  accuracy from the accepted seed338 scaffold. Because carrier_off is better
  than full, the carrier remains non-causal on seed337.

next step:
  Stop auxiliary-objective repair for this bottleneck. The next work item is a
  real recurrent state-update redesign that keeps the dual/nested QTRM-native
  causal path but changes how z_L/z_H exchange and retain transition state.
```

Cross-exchange structural repair:

```text
code:
  scripts/335_train_qtrm_native_etd_probe.py

new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange

mechanism:
  Preserve the accepted dual/nested core-carrier path, but add a warm-start
  identity cross-exchange gate:
    z_L nested delta -> small residual into z_H
    z_H nested delta -> small residual into z_L
  The exchange is controlled by coupling_off and remains inside the native
  token -> recurrent core -> LM logits path.

test:
  tests/test_qtrm_native_etd_probe.py
  test_trm_dual_z_nested_core_carrier_cross_exchange_is_inside_core

experiment:
  local_eval/qtrm_native_l6_seed337_nested_core_carrier_cross_exchange_s800_20260515/report.json

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 600
  full_generation_exact: 0.3385416666666667
  min_family_generation_exact: 0.046875
  full_minus_carrier_off: 0.01302083333333337
  coupling_off_generation_exact: 0.3333333333333333
  carrier_off_generation_exact: 0.3255208333333333
  answer_space_argmax_exact: 0.3463541567325592
  answer_space_gold_mean_rank: 7.6328125

by_family:
  checksum: 0.921875
  modchain: 0.046875
  revchain: 0.046875

decision:
  Cross-exchange gives a small same-metric dependency for carrier/coupling and
  slightly improves answer-space argmax, but it does not repair the weak-family
  floor and the coupling drop is far below the 0.05 causal threshold. Do not
  promote this structure.

next structural hypothesis:
  The next architecture must bind operation order more explicitly inside the
  recurrent update, not merely exchange generic z_L/z_H deltas. Candidate:
  a step/operation-conditioned transition router that modulates the z_L/z_H
  update from encoded token state while preserving the same LM-logit output
  path and requiring coupling/router ablation drops.

verification:
  .venv/bin/python -m py_compile \
    scripts/335_train_qtrm_native_etd_probe.py \
    scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

  .venv/bin/python -m unittest \
    tests.test_qtrm_native_etd_probe \
    tests.test_qtrm_native_mixed_text_reasoning_probe -v

  result: 197 tests OK
```

Step-conditioned recurrent update:

```text
code:
  scripts/335_train_qtrm_native_etd_probe.py

new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned

mechanism:
  Adds recurrent-step embeddings and encoded-token-conditioned z_L/z_H delta
  optimizers inside the existing dual/nested core-carrier path. The update is
  disabled by coupling_off and final answers still use the native LM logits.

focused test:
  tests/test_qtrm_native_etd_probe.py
  test_trm_dual_z_nested_core_carrier_step_conditioned_is_inside_core

experiment:
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_s1000_20260515/report.json

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 900
  full_generation_exact: 0.3619791666666667
  min_family_generation_exact: 0.0703125
  full_minus_carrier_off: 0.015625
  coupling_off_generation_exact: 0.359375
  z_l_zero_generation_exact: 0.059895833333333336
  answer_space_argmax_exact: 0.3567708432674408

by_family:
  checksum: 0.9296875
  modchain: 0.0703125
  revchain: 0.0859375

interpretation:
  This is the best seed337 result in this repair sequence and revchain crosses
  the 0.08 family floor, but modchain remains below threshold and coupling_off
  is almost unchanged. The improvement is not yet causally attributable to the
  new step-conditioned path.
```

Step-only causal repair:

```text
report:
  local_eval/qtrm_native_l6_seed337_nested_step_conditioned_step_only_repair_s600_20260515/report.json

settings:
  resume:
    local_eval/qtrm_native_l6_seed337_nested_step_conditioned_s1000_20260515/last.pt
  train_param_name_regex:
    ^trm_nested_step_
  lr: 1.0e-4
  steps: 600

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 0
  full_generation_exact: 0.3619791666666667
  min_family_generation_exact: 0.0703125
  coupling_off_generation_exact: 0.359375

decision:
  Step-only training did not improve beyond the initial checkpoint. Keep the
  step-conditioned result as a useful directional signal, but do not promote it
  until an order/step path produces both a family-floor pass and a meaningful
  coupling/router ablation drop.

next structural hypothesis:
  A nested order-router should choose or blend update order, not just add a
  generic step-conditioned delta. The route must be derived from encoded token
  state and final answers must remain LM-logit generation.
```

Nested order-router recurrent update:

```text
code:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router

mechanism:
  Keeps the dual/nested core-carrier path, then blends two internal update
  orders from the same encoded token stream:
    route0: L -> H
    route1: H-primed H -> L -> H
  Final answers still use the native LM logits. The generic router utilities
  now support both trm_order_router and trm_nested_order_router.

focused tests:
  tests/test_qtrm_native_etd_probe.py
    test_trm_dual_z_nested_core_carrier_order_router_stays_inside_native_core
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
    test_order_router_family_order_loss_supports_nested_router
    test_forced_route_answer_loss_supports_nested_router_force_attr

experiment:
  local_eval/qtrm_native_l6_seed337_nested_order_router_aux_forced_s600_20260515/report.json

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 300
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.036458333333333315
  coupling_off_generation_exact: 0.2994791666666667
  carrier_off_generation_exact: 0.3333333333333333
  order_route0_generation_exact: 0.2994791666666667
  order_route1_generation_exact: 0.041666666666666664

by_family:
  checksum: 0.890625
  modchain: 0.0625
  revchain: 0.0546875

router probe:
  overall last route1 probability: 0.42304137349128723
  checksum last route1 probability: 0.17793744802474976
  modchain last route1 probability: 0.5396702289581299
  revchain last route1 probability: 0.5515165328979492

interpretation:
  The auxiliary can move the router toward route1 for modchain/revchain, but
  route1 itself is weak. Forced route1 is only 0.0417 overall, while the full
  model still misses the family floor and does not clear the causal ablation
  threshold. Do not keep sweeping router LR or auxiliary weights.
```

Sequence-level nested order-router:

```text
new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router

mechanism:
  Uses one route distribution per whole case instead of token-wise route
  blending, to test whether inconsistent per-token routing is the bottleneck.

focused test:
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
    test_sequence_level_nested_order_router_uses_one_route_distribution_per_case

experiment:
  local_eval/qtrm_native_l6_seed337_nested_sequence_order_router_aux_forced_s600_20260515/report.json

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 300
  full_generation_exact: 0.09635416666666667
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: -0.23177083333333331
  coupling_off_generation_exact: 0.328125
  carrier_off_generation_exact: 0.08854166666666667
  order_route0_generation_exact: 0.328125
  order_route1_generation_exact: 0.041666666666666664

by_family:
  checksum: 0.171875
  modchain: 0.0546875
  revchain: 0.0625

interpretation:
  Sequence-level route locking is worse. The full model collapses because it
  commits too much of the whole state to the weak H->L->H route. The bottleneck
  is not token-vs-sequence routing; it is the route1 transition candidate
  itself. Close the order-router ladder unless route1 is replaced by a stronger
  transition mechanism.

next structural hypothesis:
  Replace route1, not the router. The next candidate must bind operation order
  inside the recurrent z_L/z_H transition itself while preserving the native
  token -> core -> LM-logits path. Do not add a side renderer, typed executor,
  MemoryOS/RAG path, or more scalar router tuning.
```

Order-bound route1 replacement:

```text
code:
  scripts/335_train_qtrm_native_etd_probe.py

new think_structure:
  trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_bound_router

mechanism:
  Keeps route0 as L->H, keeps the nested router, but replaces route1 with an
  H->L->H route that injects route1-only attention over the encoded prompt
  tokens. This tests whether route1 failed because it could not re-bind source
  operation positions from the visible prompt.

focused test:
  tests/test_qtrm_native_etd_probe.py
    test_trm_dual_z_nested_order_bound_router_replaces_route1_transition

experiment:
  local_eval/qtrm_native_l6_seed337_nested_order_bound_router_aux_forced_s600_20260515/report.json

result:
  accepted: false
  decision: rejected
  best_periodic_eval_step: 300
  full_generation_exact: 0.3359375
  min_family_generation_exact: 0.0546875
  full_minus_worst_ablation: 0.0390625
  coupling_off_generation_exact: 0.296875
  carrier_off_generation_exact: 0.3333333333333333
  order_route0_generation_exact: 0.296875
  order_route1_generation_exact: 0.03125

by_family:
  checksum: 0.875
  modchain: 0.078125
  revchain: 0.0546875

interpretation:
  Source attention inside route1 does not solve the bottleneck. It slightly
  helps modchain reach the 0.078125 edge, but revchain remains below the floor
  and forced route1 is worse than the previous H->L->H route. The missing
  mechanism is not just prompt-source re-attention. Close this route1-attention
  branch.

next structural hypothesis:
  Stop building alternative routers around H->L->H. The next candidate should
  be a direct recurrent transition-binding mechanism that updates the core
  state in operation order, or a smaller diagnostic task that proves such a
  transition can learn before returning to L6.
```

Route1 revchain-only capacity diagnostic:

```text
report:
  local_eval/qtrm_native_l6_seed337_order_bound_route1_revchain_capacity_s400_20260515/report.json

purpose:
  Test whether the new order-bound route1 can learn revchain when the whole
  dataset is revchain and the router/forced-route losses push strongly toward
  route1. This is diagnostic only, not a canonical promotion gate.

result:
  accepted: false
  decision: rejected
  full_generation_exact: 0.04296875
  min_family_generation_exact: 0.04296875
  full_minus_worst_ablation: 0.01953125
  coupling_off_generation_exact: 0.0078125
  order_route0_generation_exact: 0.0078125
  order_route1_generation_exact: 0.046875

router probe:
  last route1 probability: 0.999332070350647
  mean route1 probability: 0.7590198516845703

interpretation:
  This isolates the bottleneck. The router can be forced to select route1, but
  route1 still cannot solve revchain. Therefore the weak L6 result is not a
  router-selection problem or a multi-family conflict problem. It is a route1
  transition-capacity/state-binding problem.

next structural hypothesis:
  Do not continue H->L->H route wrappers. Build a direct operation-order
  recurrent transition diagnostic: the state update itself must learn ordered
  composition before it is reinserted into the L6 nested dual core.
```

Reduced operation-order transition diagnostic:

```text
script:
  scripts/353_train_operation_order_transition_probe.py

focused test:
  tests/test_operation_order_transition_probe.py

purpose:
  Isolate one question before changing L6 again:
  can a learned recurrent transition read the same prompt tokens, compose the
  operations in fwd or rev order, and emit the answer through normal LM logits?

smoke held-out run:
  local_eval/operation_order_transition_probe_smoke_20260515/report.json

smoke held-out result:
  decision: rejected
  full_generation_exact: 0.044921875
  transition_off_generation_exact: 0.03515625
  order_shuffle_generation_exact: 0.033203125
  full_minus_transition_off: 0.009765625
  full_minus_order_shuffle: 0.01171875

capacity run:
  local_eval/operation_order_transition_probe_capacity_mod16_20260515/report.json

capacity held-out result:
  decision: rejected
  full_generation_exact: 0.431640625
  transition_off_generation_exact: 0.07421875
  order_shuffle_generation_exact: 0.2421875
  full_minus_transition_off: 0.357421875
  full_minus_order_shuffle: 0.189453125

same-seed capacity control:
  local_eval/operation_order_transition_probe_capacity_mod16_trainseed_eval_20260515/report.json

same-seed capacity control result:
  decision: accepted_operation_order_transition_diagnostic
  full_generation_exact: 0.999755859375
  transition_off_generation_exact: 0.0712890625
  order_shuffle_generation_exact: 0.54150390625
  state_reset_generation_exact: 0.059326171875
  full_minus_transition_off: 0.928466796875
  full_minus_order_shuffle: 0.458251953125

interpretation:
  The recurrent transition is not inert. On the same generated distribution it
  can memorize and solve almost perfectly, and transition/state/order ablations
  are strongly causal. However, held-out seed generalization is only 0.4316
  under the stronger mod16 capacity run. The current bottleneck is therefore
  not "the transition cannot learn at all"; it is compositional
  value-operation generalization from token embeddings. Reinsert this
  mechanism into L6 only after adding a better state/value codec or another
  generalization-oriented transition objective.
```

Reduced transition trace/value-codec repair:

```text
script:
  scripts/353_train_operation_order_transition_probe.py

new knobs:
  --value-codec learned|circular
  --trace-loss-weight

mechanism:
  The recurrent transition still reads prompt tokens and emits LM logits. The
  circular value codec gives value tokens a continuous latent basis, while
  trace loss supervises each intermediate recurrent state on the stepwise
  operation result. The step targets are training-only; inference does not call
  a symbolic solver.

mod16 controls:
  circular codec only:
    report: local_eval/operation_order_transition_probe_circular_mod16_20260515/report.json
    decision: rejected
    full_generation_exact: 0.447265625
    full_minus_transition_off: 0.388671875
    full_minus_order_shuffle: 0.201171875

  learned codec + trace loss:
    report: local_eval/operation_order_transition_probe_learned_trace_mod16_20260515/report.json
    decision: accepted_operation_order_transition_diagnostic
    full_generation_exact: 0.828125
    full_minus_transition_off: 0.765625
    full_minus_order_shuffle: 0.361328125

  circular codec + trace loss:
    report: local_eval/operation_order_transition_probe_circular_trace_mod16_20260515/report.json
    decision: accepted_operation_order_transition_diagnostic
    full_generation_exact: 0.970703125
    full_minus_transition_off: 0.935546875
    full_minus_order_shuffle: 0.427734375

mod32 scale check:
  undertrained d128:
    report: local_eval/operation_order_transition_probe_circular_trace_mod32_20260515/report.json
    decision: rejected
    full_generation_exact: 0.18359375

  d256/s3000:
    report: local_eval/operation_order_transition_probe_circular_trace_mod32_d256_s3000_20260515/report.json
    decision: accepted_operation_order_transition_diagnostic
    full_generation_exact: 0.89453125
    transition_off_generation_exact: 0.0390625
    order_shuffle_generation_exact: 0.4609375
    state_reset_generation_exact: 0.0
    full_minus_transition_off: 0.85546875
    full_minus_order_shuffle: 0.43359375

interpretation:
  This is the first reduced gate that cleanly solves held-out operation-order
  generalization with strong causal ablations. Trace supervision is the major
  unlock; circular value coding improves the ceiling and scaling. The next L6
  move should transplant this recipe into the QTRM-native dual/nested core:
  use state-trace/depth supervision as the main operation-order objective and
  replace flat random value embeddings/readout with a value codec that gives
  the recurrent core a stable state manifold.
```

L6 trace/value-codec transplant attempt:

```text
implementation:
  scripts/335_train_qtrm_native_etd_probe.py
    NativeQTRMETDLM now accepts value_codec and explicit value_token_ids.

  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
    --value-codec circular is wired through to the native model.
    It is guarded to require --tokenizer-mode number.
    value_token_ids_for_tokenizer maps 00..31 tokens to value indices.

tests:
  tests/test_qtrm_native_etd_probe.py
    circular value codec and custom value_token_ids

  tests/test_qtrm_native_mixed_text_reasoning_probe.py
    parser accepts value_codec and number tokenizer;
    value_token_ids_for_tokenizer follows two-digit values.

invalid char-tokenizer transplant:
  report: local_eval/qtrm_native_l6_seed338_circular_value_trace_repair_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.0546875
  reason: char tokenizer has no atomic value tokens; applying circular codec
          to char ids corrupts the token semantics.

valid number-tokenizer transplant:
  report: local_eval/qtrm_native_l6_seed338_number_circular_trace_repair_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.068359375
  min_family_generation_exact: 0.023391812865497075
  full_minus_worst_ablation: 0.0078125
  transition/core signal: weak

interpretation:
  The reduced trace/value recipe does not transfer directly into the current
  text runner. With number tokenizer, value tokens are atomic, but operation
  ids and numeric values share the same two-digit tokens. The reduced
  diagnostic separated op tokens from value tokens; the text runner does not.
  The next full-L6 transplant must separate operation-role embeddings from
  value-role embeddings, or add role-conditioned value/op codecs, before using
  circular value coding as canonical evidence.
```

L6 op-role tokenizer transplant:

```text
implementation:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
    --number-tokenizer-op-role-tokens
      encodes visible ops-segment numbers as internal opNN tokens while decode
      preserves the original prompt text.

    flexible checkpoint load now composes unseen number/op tokens from source
    character rows when migrating a char-token checkpoint:
      "05"   <- mean("0", "5")
      "op05" <- mean("o", "p", "0", "5")

tests:
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
    number tokenizer separates operation-role tokens;
    flexible load composes number/op tokens from source chars.

random-init op-role run:
  report: local_eval/qtrm_native_l6_seed338_number_oprole_circular_trace_repair_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.08203125
  min_family_generation_exact: 0.029239766081871343
  full_minus_think0: 0.08203125
  full_minus_worst_ablation: 0.025390625
  carrier_off_generation_exact: 0.095703125

composed-init op-role run:
  report: local_eval/qtrm_native_l6_seed338_number_oprole_composedinit_circular_trace_repair_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.0546875
  min_family_generation_exact: 0.03508771929824561
  full_minus_think0: 0.029296875
  full_minus_worst_ablation: 0.009765625
  carrier_off_generation_exact: 0.048828125

interpretation:
  OP/VALUE token separation fixes the representational contract but does not
  solve L6. Random op-role tokens slightly improve full exact over the earlier
  number-circular run, but the carrier path is not causally helpful because
  carrier_off is stronger than full. Composed char-to-number initialization
  prevents random-token migration but does not improve the gate.

roadmap consequence:
  Stop treating tokenizer/codec repair as the remaining bottleneck. The next
  L6 fix must make the recurrent state update itself learn ordered
  operation-role transitions, likely by porting the reduced probe's transition
  trace objective more directly into z_L/z_H rather than adding more router,
  carrier, or tokenizer variants.
```

L6 shared-LM/value-codec readout audit:

```text
implementation correction:
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py
    state_trace_depth_answer_loss and latent_refinement_loss now call the
    model's canonical shared LM/value-codec readout instead of bypassing it
    with direct lm_head(norm(hidden)).

tests:
  tests/test_qtrm_native_mixed_text_reasoning_probe.py
    state trace depth loss uses shared _lm_logits path;
    latent refinement loss uses shared _lm_logits path.

shared readout number/op-role run:
  report: local_eval/qtrm_native_l6_seed338_number_oprole_sharedlm_circular_trace_repair_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.072265625
  min_family_generation_exact: 0.017543859649122806
  full_minus_think0: 0.072265625
  full_minus_worst_ablation: 0.00390625
  z_l_zero_generation_exact: 0.068359375
  carrier_off_generation_exact: 0.0625

z_L-direct step-codec run:
  report: local_eval/qtrm_native_l6_seed338_zl_direct_stepcodec_sharedlm_s1200_20260515/report.json
  decision: rejected
  full_generation_exact: 0.052734375
  min_family_generation_exact: 0.029239766081871343
  full_minus_think0: 0.041015625
  full_minus_worst_ablation: 0.0
  z_l_zero_generation_exact: 0.052734375
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.064453125

interpretation:
  The readout fix was necessary, but it does not make the number/op-role
  transplant a valid L6 path. Direct auxiliary pressure on z_L also fails:
  zeroing z_L leaves full accuracy unchanged, while zeroing z_H collapses the
  output. This means the transplanted number/op-role setup is no longer using
  the accepted dual/nested z_L causal scaffold.

roadmap consequence:
  Stop auxiliary-head/tokenizer sweeps for this branch. The accepted char-token
  dual/nested core-carrier scaffold remains the active causal scaffold. Next
  work should improve family balance and transition generalization inside that
  accepted native path, not replace it with a migrated number-token branch.
```

L6 d256 native number/op-role path:

```text
from-scratch d256 run:
  report: local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_s3000_20260515/report.json
  decision: rejected
  full_generation_exact: 0.134765625
  best_periodic_generation_exact: 0.1484375
  min_family_generation_exact: 0.04678362573099415
  full_minus_worst_ablation: 0.037109375

interpretation:
  Initial d256 training is undertrained but rising. Unlike the d128 migrated
  branch, the from-scratch path has a real depth/core signal and improves with
  more optimization.

continuation:
  report: local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_cont_s7000_20260515/report.json
  decision: rejected
  full_generation_exact: 0.29296875
  min_family_generation_exact: 0.07017543859649122
  full_minus_worst_ablation: 0.240234375
  full_minus_carrier_off: 0.052734375
  best_periodic_generation_exact: 0.2994791666666667
  best_periodic_min_family: 0.0859375

interpretation:
  The d256 native path becomes strongly causal and nearly crosses L6. The
  remaining blocker is revchain/family floor, not format validity or lack of
  recursive-depth signal.

revchain-heavy retention repair:
  report: local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/report.json
  decision: accepted_l6_d256_number_oprole_circular_trace_revrepair
  full_generation_exact: 0.37890625
  min_family_generation_exact: 0.11695906432748537
  full_minus_think0: 0.3515625
  full_minus_worst_ablation: 0.33203125
  full_minus_carrier_off: 0.09765625
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.046875
  z_l_zero_generation_exact: 0.021484375
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.28125

by_family:
  checksum: 0.8705882352941177
  modchain: 0.15204678362573099
  revchain: 0.11695906432748537

decision:
  Promote this as the current strongest QTRM-native L6 checkpoint. It keeps the
  universal path:
    prompt tokens -> number/op-role tokenizer -> native embeddings ->
    dual/nested recurrent core -> shared circular value LM logits ->
    autoregressive answer.

  This supersedes the seed338 char-token scaffold as the strongest local L6
  evidence, but it still requires seed-stability and language non-regression
  before any L7/general-LM claim.
```

L6 d256 revrepair seed-stability eval:

```text
summary:
  local_eval/qtrm_native_l6_d256_revrepair_seed_stability_summary_20260515.json

checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

decision:
  accepted_l6_d256_revrepair_seed_stable_3eval

eval_seed 9340:
  report: local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/report.json
  full_generation_exact: 0.37890625
  min_family_generation_exact: 0.11695906432748537
  full_minus_worst_ablation: 0.33203125

eval_seed 9341:
  report: local_eval/qtrm_native_l6_d256_revrepair_seedstab_eval9341_20260515/report.json
  full_generation_exact: 0.36328125
  min_family_generation_exact: 0.09941520467836257
  full_minus_worst_ablation: 0.294921875

eval_seed 9342:
  report: local_eval/qtrm_native_l6_d256_revrepair_seedstab_eval9342_20260515/report.json
  full_generation_exact: 0.3828125
  min_family_generation_exact: 0.1111111111111111
  full_minus_worst_ablation: 0.34765625

decision:
  This passes the immediate 3-eval-seed stability gate for L6. It is still not
  a language-capable L7/general-LLM checkpoint. Next gates are:
    1. midpoint/rollback audit around the revchain repair;
    2. language non-regression;
    3. program_len generalization beyond 6.
```

L6 d256 revrepair midpoint audit:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_mid1500_audit_20260515/report.json

decision:
  accepted_l6_d256_revrepair_mid1500_audit

metrics:
  full_generation_exact: 0.34375
  min_family_generation_exact: 0.08771929824561403
  full_minus_think0: 0.314453125
  full_minus_worst_ablation: 0.298828125
  full_minus_carrier_off: 0.123046875
  state_reset_generation_exact: 0.0234375
  op_zero_generation_exact: 0.044921875
  z_l_zero_generation_exact: 0.03125
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.220703125

decision:
  The revchain-heavy repair is not a single lucky final checkpoint. The 1500
  step midpoint independently passes the same strict L6 gate. Remaining
  blockers for promotion are language non-regression and length generalization.
```

L6 d256 revrepair len7 zero-shot audit:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_eval_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

setup:
  steps: 0
  program_len: 7
  train_think_steps/eval_think_steps: 7
  resume_allow_missing: true
  pos_embed_resize_strategy: random_tail

resume load:
  pos_embed.weight resized from [46, 256] to [48, 256]
  copied_rows: 46
  random_tail filled_rows: 2

decision:
  rejected

reject reasons:
  full_exact_below_threshold
  depth_gain_below_threshold
  ablation_drop_below_threshold
  family_exact_below_threshold

metrics:
  full_generation_exact: 0.015625
  think0_generation_exact: 0.005859375
  full_minus_think0: 0.009765625
  full_minus_worst_ablation: -0.0078125
  min_family_generation_exact: 0.0
  full_minus_carrier_off: -0.00390625
  state_reset_generation_exact: 0.017578125
  op_zero_generation_exact: 0.0234375
  z_l_zero_generation_exact: 0.015625
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.01953125

interpretation:
  The accepted L6 d256 revrepair checkpoint is seed-stable and midpoint-stable,
  but it is still length-specialized. The len7 zero-shot eval collapses to
  near chance and the causal ablation pattern disappears. The random-tail
  positional rows are a concrete OOD factor, but the main result is that L6
  success must not be promoted to L7 or general length reasoning.

next action:
  Do not claim L7. Run a length-curriculum or positional-generalization repair
  from the accepted L6 checkpoint, then require both len6 retention and len7
  acceptance under the same final LM-logit/causal-ablation gate.
```

L6 d256 revrepair len7 curriculum repair:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_curriculum_s1500_20260515/report.json

canonical argmax eval:
  local_eval/qtrm_native_l6_d256_revrepair_len7_curriculum_argmax_eval_20260515/report.json
  same decisive metrics; still rejected

resume checkpoint:
  local_eval/qtrm_native_l6_d256_number_oprole_circular_trace_revrepair_s3000_20260515/last.pt

setup:
  steps: 1500
  program_len: 7
  train_think_steps/eval_think_steps: 7
  active_len_curriculum: true
  active_len_curriculum_min: 4
  active_len_curriculum_warmup_frac: 0.6
  active_len_replay_loss_weight: 0.02
  active_len_replay_min/max: 1/6
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.05
  pos_embed_resize_strategy: repeat_last

resume load:
  pos_embed.weight resized from [46, 256] to [48, 256]
  copied_rows: 46
  repeat_last filled_rows: 2

periodic trend:
  step0 exact: 0.0
  step300 exact: 0.0234375
  step600 exact: 0.0234375
  step900 exact: 0.09375
  step1200 exact: 0.140625
  step1500 exact: 0.1484375

decision:
  rejected

reject reasons:
  full_exact_below_threshold
  family_exact_below_threshold

metrics:
  full_generation_exact: 0.154296875
  think0_generation_exact: 0.025390625
  full_minus_think0: 0.12890625
  full_minus_worst_ablation: 0.11328125
  min_family_generation_exact: 0.023391812865497075
  full_minus_carrier_off: 0.08984375
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0
  z_l_zero_generation_exact: 0.041015625
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.064453125

by_family:
  checksum: 0.40588235294117647
  modchain: 0.03508771929824561
  revchain: 0.023391812865497075

interpretation:
  The curriculum repair substantially improves len7 over zero-shot and restores
  a real recursive-depth/ablation signal. It is still not L7 because the model
  overfits toward checksum while modchain/revchain remain near chance. The next
  useful repair should target hard-family balance on len7, while preserving the
  restored depth and ablation gains.
```

L7 hard-family continuation from len7 curriculum:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_hardfamily_s1500_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_revrepair_len7_curriculum_s1500_20260515/last.pt

setup:
  steps: 1500
  program_len: 7
  task_families: revchain x5, modchain x4, checksum x1
  family_dro_loss_weight: 0.1
  periodic_eval_score_mode: family_floor
  eval_answer_space_argmax: true
  retention_reference_checkpoint: resume
  retention_kl_loss_weight: 0.05
  active_len_replay_loss_weight: 0.02

periodic trend on 128-case eval:
  step0 exact/min_family: 0.1484375 / 0.0
  step300 exact/min_family: 0.171875 / 0.046511627906976744
  step600 exact/min_family: 0.234375 / 0.06976744186046512
  step900 exact/min_family: 0.203125 / 0.0
  step1200 exact/min_family: 0.1875 / 0.0
  step1500 exact/min_family: 0.21875 / 0.023255813953488372

decision:
  rejected

metrics on 512-case final eval:
  full_generation_exact: 0.189453125
  think0_generation_exact: 0.029296875
  full_minus_think0: 0.16015625
  full_minus_worst_ablation: 0.146484375
  min_family_generation_exact: 0.023391812865497075
  full_minus_carrier_off: 0.10546875
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0
  z_l_zero_generation_exact: 0.04296875
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.083984375

by_family:
  checksum: 0.49411764705882355
  modchain: 0.023391812865497075
  revchain: 0.05263157894736842

interpretation:
  The hard-family continuation improves full exact and strengthens causal
  depth/ablation gaps, but it still fails L7 because modchain remains the
  limiting family. The 128-case periodic gate briefly reached min-family 0.0698
  at step600, so the direction is not useless; the next repair should focus
  specifically on modchain while using 512-case periodic selection to avoid
  accepting a small-sample family-floor artifact.
```

L7 modchain-focused continuation:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_revrepair_len7_hardfamily_s1500_20260515/last.pt

setup:
  steps: 900
  program_len: 7
  task_families: modchain x7, revchain x3, checksum x1
  family_dro_loss_weight: 0.12
  periodic_eval_score_mode: family_floor
  eval_during_training_cases: 512
  eval_answer_space_argmax: true
  lr: 1e-5

periodic trend on 512-case eval:
  step0 exact/min_family: 0.189453125 / 0.023391812865497075
  step300 exact/min_family: 0.20703125 / 0.023391812865497075
  step600 exact/min_family: 0.216796875 / 0.03508771929824561
  step900 exact/min_family: 0.2578125 / 0.04678362573099415

decision:
  rejected

metrics:
  full_generation_exact: 0.2578125
  think0_generation_exact: 0.025390625
  full_minus_think0: 0.232421875
  full_minus_worst_ablation: 0.212890625
  min_family_generation_exact: 0.04678362573099415
  full_minus_carrier_off: 0.16796875
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0
  z_l_zero_generation_exact: 0.044921875
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.08984375

by_family:
  checksum: 0.6823529411764706
  modchain: 0.04678362573099415
  revchain: 0.04678362573099415

interpretation:
  This is the strongest len7 run so far. It remains rejected, but the failure
  is now narrow: full exact is approaching the 0.30 threshold and the recursive
  causal gaps are strong. The next shortest path is another low-LR hard-family
  continuation, balanced between modchain and revchain, with 512-case periodic
  family-floor selection. Do not change backbone/topology yet.
```

L7 balanced hard-family continuation:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_balanced_hard_s900_20260515/report.json

resume checkpoint:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt

setup:
  steps: 900
  program_len: 7
  task_families: modchain x5, revchain x5, checksum x1
  family_dro_loss_weight: 0.15
  periodic_eval_score_mode: family_floor
  eval_during_training_cases: 512
  eval_answer_space_argmax: true
  lr: 8e-6

periodic trend on 512-case eval:
  step0 exact/min_family: 0.2578125 / 0.04678362573099415
  step300 exact/min_family: 0.248046875 / 0.029239766081871343
  step600 exact/min_family: 0.244140625 / 0.017543859649122806
  step900 exact/min_family: 0.23828125 / 0.03508771929824561

decision:
  rejected

metrics:
  full_generation_exact: 0.2578125
  think0_generation_exact: 0.025390625
  full_minus_think0: 0.232421875
  full_minus_worst_ablation: 0.212890625
  min_family_generation_exact: 0.04678362573099415
  full_minus_carrier_off: 0.16796875
  state_reset_generation_exact: 0.015625
  op_zero_generation_exact: 0.0
  z_l_zero_generation_exact: 0.044921875
  z_h_zero_generation_exact: 0.0
  carrier_off_generation_exact: 0.08984375

by_family:
  checksum: 0.6823529411764706
  modchain: 0.04678362573099415
  revchain: 0.04678362573099415

interpretation:
  Balanced continuation does not improve over the modchain-focused checkpoint.
  The restored best checkpoint is step0, and later training degrades both full
  exact and family floor. The next valid step is not another blind continuation;
  run an operation/position/family breakdown for len7 modchain and revchain,
  then repair the specific transition failure that the breakdown exposes.
```

L7 operation breakdown and reduced transition control:

```text
full-QTRM operation breakdown:
  report:
    local_eval/qtrm_native_l6_d256_revrepair_len7_opbreakdown_2048_20260515/report.json
  eval cases: 2048
  decision: rejected
  full_generation_exact: 0.23291015625
  min_family_generation_exact: 0.03513909224011713
  full_minus_think0: 0.20166015625
  full_minus_worst_ablation: 0.1904296875

full-QTRM by_family:
  checksum: 0.624633431085044
  modchain: 0.03513909224011713
  revchain: 0.03953147877013177

worst modchain last-op slices:
  op07: 0.0
  op06: 0.010638297872340425
  op05: 0.0297029702970297

state-transition codec repair:
  report:
    local_eval/qtrm_native_l6_d256_revrepair_len7_state_transition_codec_s600_20260515/report.json
  resume:
    local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt
  decision: rejected
  best checkpoint: step0
  full_generation_exact: 0.2578125
  min_family_generation_exact: 0.04678362573099415

reduced transition control:
  report:
    local_eval/operation_order_transition_probe_len7_circular_trace_mod32_d256_s10000_20260515/report.json
  decision:
    accepted_operation_order_transition_diagnostic
  full_generation_exact: 0.998046875
  fwd: 0.998046875
  rev: 0.998046875
  transition_off_generation_exact: 0.0283203125
  order_shuffle_generation_exact: 0.5166015625
  state_reset_generation_exact: 0.0
  full_minus_transition_off: 0.9697265625
  full_minus_order_shuffle: 0.4814453125

interpretation:
  Program_len7 ordered recurrent transition is learnable. The full-QTRM failure
  is therefore not an impossibility result and not a reason to abandon QTRM
  native. The missing mechanism is a faithful transplant of the reduced
  transition cell's causal recipe into the full token->core->LM path:
  family-conditioned ordered op read, recurrent state update, circular value
  manifold, and trace loss that directly supervises the state trajectory. Short
  low-LR full-QTRM continuations and detached auxiliary codec heads are now
  rejected as the shortest path.
```

Stronger full-QTRM transplant triage:

```text
report:
  local_eval/qtrm_native_l6_d256_revrepair_len7_transition_transplant_lr3e5_s1200_20260515/report.json

resume:
  local_eval/qtrm_native_l6_d256_revrepair_len7_modchain_focus_s900_20260515/last.pt

setup:
  lr: 3e-5
  steps: 1200
  program_len: 7
  same nested dual-z core-carrier path
  state_trace_depth_loss_weight: 0.5
  core_step_codec_loss_weight: 0.2
  core_step_op_codec_loss_weight: 0.1
  core_step_position_codec_loss_weight: 0.05

periodic trend:
  step0 exact/min_family: 0.2578125 / 0.04678362573099415
  step300 exact/min_family: 0.21484375 / 0.03508771929824561
  step600 exact/min_family: 0.244140625 / 0.017543859649122806
  step900 exact/min_family: 0.17578125 / 0.029239766081871343
  step1200 exact/min_family: 0.244140625 / 0.04093567251461988

decision:
  rejected

interpretation:
  Raising LR and strengthening detached auxiliary state/value/op/position
  losses still restores the initial checkpoint as best. The next step should be
  a canonical transition-cell transplant inside the main core path, not another
  stronger side-head continuation.
```

Qwen-assisted native language bootstrap:

```text
question:
  Can Qwen3.5-2B be used to make a QTRM-native LM quickly through healing tune?

answer:
  Yes for language bootstrapping, no for proving QTRM-native reasoning by
  itself. Qwen may provide tokenizer, initialization hints, cached logits, or
  teacher distributions. It must not be present in the final inference path if
  the result is called QTRM-native.

native-allowed path:
  Qwen tokenizer / text corpus / optional cached teacher logits
  -> QTRM-native token embeddings
  -> mandatory QTRM recursive core
  -> native LM head
  -> autoregressive text

not native:
  prompt -> Qwen donor forward -> donor hidden states/logits -> QTRM sidecar

recommended stages:
  1. Qwen tokenizer + QTRM-native LM smoke on ordinary text.
  2. Offline Qwen top-k logit distillation or CE+KL language healing.
  3. Core-on language non-regression: recurrence must not cause repetition.
  4. Add raw reasoning gates only after the native text path is stable.
  5. Promote only if Qwen is absent at inference and core ablations still matter.

expectation:
  This can make the native LM stop degenerating much faster than training from
  scratch, but it will not instantly give Qwen-level knowledge or reasoning.
  It is a language prior transfer, not a replacement for native recursive-core
  acceptance.
```

Web reference update, 2026-05-15:

```text
Qwopus3.5-27B-v3:
  Source:
    https://huggingface.co/Jackrong/Qwopus3.5-27B-v3
    https://github.com/R6410418/Jackrong-llm-finetuning-guide

  Observed pattern:
    Base model: Qwen/Qwen3.5-27B
    Training: Unsloth + LoRA SFT
    Masking: response-only training around assistant/<think> region
    Data: high-fidelity reasoning/CoT/coding/chat distillation sets
    Runtime: one merged/fine-tuned model, not a separate donor sidecar
    Evidence for this model being a layer-stack frankenmerge: not found in the
      model card. The public frankenmerge/heal-tune description found during
      search refers to Qwopus-GLM-18B-Merged, a separate 64-layer merge of two
      Qwen3.5-9B finetunes, followed by a 1000-step QLoRA healing run.

  Meaning for QTRM:
    Qwopus is a precedent for "donor-assisted but native-at-inference" only in
    the teacher/data/initialization sense. It is not a precedent for keeping a
    frozen donor forward pass inside the final architecture.

  Useful pieces to copy:
    1. response-only SFT mask;
    2. high-quality reasoning scaffold data;
    3. LoRA/QLoRA first, merge/export later;
    4. teacher-generated answer/continuation text as a language scaffold;
       visible CoT must not be copied into QTRM-native language bootstrap;
    5. benchmark before/after against the base model.

  Not enough for our claim:
    Qwopus improves a Qwen-family model by post-training. It does not prove a
    new QTRM-native recurrent architecture. For us, Qwen/Qwopus should be a
    language/reasoning teacher used offline; final inference must still be:

      prompt tokens
      -> QTRM-native embeddings
      -> mandatory nested dual-z recurrent core
      -> native LM head
      -> autoregressive text
```

Language-first bootstrap literature update, 2026-05-15:

```text
decision:
  Temporarily prioritize QTRM-native language viability before more L7
  reasoning continuation. A recursive core that cannot render ordinary text
  cannot yet be promoted as a general LM path.

literature-backed recipe:
  1. TinyStories-style coherent-text curriculum for non-degenerate generation.
  2. Textbooks/phi-style high-quality educational synthetic data instead of
     broad noisy web first.
  3. FineWeb-Edu/DCLM-style filtering, deduplication, and model-based data
     quality selection for the next corpus.
  4. Offline Qwen/Qwopus teacher cache with CE + reverse-KL or calibrated
     sparse-logit KD; avoid tiny top-k-only KL because it can miscalibrate.
  5. GKD-style on-policy repair on QTRM-native generated mistakes.
  6. Add Orca/distilling-step-by-step reasoning traces only after language
     acceptance, and do not put visible CoT into the language surface channel.
  7. Use GaLore/ReLoRA-style memory-efficient full native learning if optimizer
     memory blocks full training.

sources:
  TinyStories:
    https://huggingface.co/papers/2305.07759
  Textbooks Are All You Need / phi:
    https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need/
    https://arxiv.org/abs/2309.05463
  FineWeb-Edu / DCLM:
    https://huggingface.co/papers/2406.17557
    https://arxiv.org/abs/2406.11794
  MiniLLM / GKD:
    https://www.microsoft.com/en-us/research/publication/knowledge-distillation-of-large-language-models/
    https://huggingface.co/papers/2306.13649
  Pretraining and sparse-logit distillation:
    https://aclanthology.org/2025.acl-long.181.pdf
    https://aclanthology.org/2025.acl-long.885.pdf
  Reasoning-trace distillation:
    https://arxiv.org/abs/2306.02707
    https://arxiv.org/abs/2305.02301
  Memory-efficient native learning:
    https://arxiv.org/abs/2403.03507
    https://arxiv.org/abs/2307.05695
```

Implemented language-first bootstrap tooling:

```text
files:
  scripts/354_train_qtrm_native_language_bootstrap.py
  scripts/354_run_qtrm_native_language_bootstrap.sh
  tests/test_qtrm_native_language_bootstrap.py

capabilities:
  - donorless QTRM-native CE training on built-in TinyStories/textbook-style
    bilingual seed corpus;
  - optional external text, teacher text, and teacher JSONL ingestion;
  - optional Hugging Face tokenizer via --tokenizer-name;
  - depth sweep / core-off eval / repetition metrics;
  - on-policy candidate JSONL export for later GKD-style repair.

verification:
  PYTHONPATH=src .venv/bin/python -m py_compile \
    scripts/354_train_qtrm_native_language_bootstrap.py
  PYTHONPATH=src .venv/bin/python -m unittest \
    tests.test_qtrm_native_language_bootstrap \
    tests.test_qtrm_native_text_probe
  git diff --check

smoke:
  local_eval/qtrm_native_language_bootstrap_smoke/report.json
  accepted: true under permissive 2-step smoke thresholds only

CUDA triage:
  report:
    local_eval/qtrm_native_language_bootstrap_triage_20260515/report.json

  decision:
    accepted

  metrics:
    tiny_last_loss: 2.1199
    edu_last_loss: 1.3097
    think_eval_loss_depth4: 1.5084
    think0_loss: 3.2798
    thinking_block_off_loss: 3.2798
    depth0: 3.2798
    depth1: 2.7307
    depth2: 2.1051
    depth4: 1.5084
    unique_chars: 26
    max_run_fraction: 0.0168

conclusion:
  First positive language-first native bootstrap signal. This does not prove
  broad language ability, but it shows the donorless QTRM-native LM path can
  learn a clean text curriculum quickly and that recurrent depth can reduce LM
  loss. Next step is Qwen tokenizer + offline Qwen/Qwopus teacher artifacts.

Qwen tokenizer smoke:
  report:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_smoke/report.json

  result:
    accepted: true under permissive tokenizer-smoke thresholds
    tokenizer_kind: hf
    tokenizer_name: Qwen/Qwen3.5-2B-Base
    vocab_size: 248077

  interpretation:
    The HF/Qwen tokenizer path is wired through the donorless native LM.
    This is not a quality claim because the one-step run remains near random
    loss. It only clears the next implementation step: real Qwen-tokenized
    native CE/KD training.

Offline teacher cache pipeline:
  files:
    scripts/355_build_qtrm_language_teacher_cache.py
    scripts/355_build_qtrm_language_teacher_cache.sh
    tests/test_qtrm_language_teacher_cache.py

  schema:
    prompt: instruction prompt for teacher only
    seed_text: clean source text
    answer: teacher continuation
    text / teacher_text: seed_text + answer, used by QTRM-native bootstrap

  quality guards:
    /no_think switch, visible <think> suppression, think-block stripping,
    think-leak rejection, repetition filter, and minimum answer length.

  dry-run:
    local_eval/qtrm_language_teacher_cache_dryrun_v2/teacher_text.jsonl.report.json

  real Qwen2B smoke:
    local_eval/qtrm_language_teacher_cache_qwen2b_smoke_v3/teacher_text.jsonl.report.json
    written: 2 records
    model: Qwen/Qwen3.5-2B-Base

  interop smoke:
    local_eval/qtrm_native_language_bootstrap_teacher_jsonl_qwen_tokenizer_smoke_v3/report.json
    result: accepted under permissive 2-step smoke thresholds

  conclusion:
    The offline teacher artifact path is now usable:
    Qwen teacher -> clean teacher JSONL -> Qwen tokenizer -> donorless
    QTRM-native bootstrap. This is the required bridge before longer native
    language training.

Teacher boundary correction:
  User correctly pointed out that QTRM-native is not a language-space CoT model.
  The language bootstrap teacher is therefore restricted to clean surface
  continuation/answer text. Visible CoT and <think> content are contamination
  for this stage.

  Code change:
    scripts/354_train_qtrm_native_language_bootstrap.py now strips visible
    think blocks and explicit reasoning labels from teacher text/JSONL before
    building stage C. scripts/355 remains continuation-only and keeps the
    generation instruction out of teacher_text.

  Architecture rule:
    Reasoning must be proven through latent recurrent depth and destructive
    ablations. Teacher traces, if ever used, are diagnostics or latent/core
    supervision, not visible answer text.

Language generation quality check:
  question:
    Does the current QTRM-native bootstrap generate language well?

  answer:
    Not yet. It now avoids the earlier character/token collapse and can learn a
    tiny clean bilingual curriculum, but the Qwen-tokenized surface-answer run
    still replays the same short User/Assistant blocks. That is prompt-pattern
    memorization, not broad language ability.

  strict report:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_surface_strict_s1200_20260515/report.json

  decision:
    rejected

  reject reason:
    on_policy_unique_line_fraction_too_low

  metrics:
    stage_a_tiny_last_loss: 2.1382
    stage_b_edu_last_loss: 0.0939
    stage_c_teacher_last_loss: 0.0942
    think_eval_loss_depth4: 0.4016
    think0_loss: 2.0356
    depth0: 2.0356
    depth1: 0.6014
    depth2: 0.4317
    depth4: 0.4016
    on_policy_unique_line_fraction:
      seed0: 0.4286
      seed1: 0.4286
      seed2: 0.5000

  interpretation:
    The recurrent core is causally useful for next-token loss on this tiny
    corpus, but current generation quality is still only a language-viability
    signal. The next real language gate must use a larger, more diverse
    answer-only corpus and must reject sample loops, not just repeated
    characters.

  code change:
    scripts/354_train_qtrm_native_language_bootstrap.py now adds on-policy
    line/block-loop metrics and rejects low unique-line-fraction samples.
    tests/test_qtrm_native_language_bootstrap.py covers replay detection.

Qwen-tokenized instruction/EOS bootstrap repair:
  problem:
    The first strict surface-answer run stopped fixed block replay only after
    adding EOS boundaries, but one heldout instruction produced an extra
    `Assistant:` marker inside the answer. That was a false accept because the
    model was still crossing record boundaries.

  code changes:
    scripts/336_train_qtrm_native_text_probe.py stops greedy generation when
    the tokenizer exposes eos_token_id and predicts EOS.

    scripts/354_train_qtrm_native_language_bootstrap.py adds:
      - deterministic diverse answer-only snippets;
      - Qwen-tokenized auto record separator `<|endoftext|>`;
      - heldout/paraphrase instruction seeds;
      - on-policy answer-surface checks for minimum continuation length,
        next User leakage, extra Assistant markers, and visible think leakage.

  rejected control:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_heldout_eos_strict_v2_s1200_20260515/report.json

    decision: rejected
    reject_reason: on_policy_extra_assistant_marker

  accepted repair:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_heldout_eos_repair_s1200_20260515/report.json

    decision: accepted_qtrm_native_language_bootstrap
    think_eval_loss_depth4: 0.3330
    think0_loss: 2.3638
    depth0: 2.3638
    depth1: 0.6556
    depth2: 0.3735
    depth4: 0.3330
    full_vs_think0: 0.1409
    full_vs_best_shallow_depth: 0.8915

  accepted on-policy samples:
    User: Why should evidence be checked?
    Assistant: Evidence should be checked because unsupported claims can sound
    convincing while still being wrong.

    User: How can writing become clearer?
    Assistant: Writing becomes clearer when sentences are short, subjects are
    explicit, and reasons are connected.

    User: 좋은 답변은 무엇인가요?
    Assistant: 좋은 답변은 질문에 직접 답하고, 근거를 분명히 말하며, 모르면
    추측하지 않는 답변이다.

  interpretation:
    This is the first clean Qwen-tokenized QTRM-native instruction/EOS
    bootstrap result. It proves only a tiny controlled language surface:
    prompt tokens -> native embeddings -> recurrent core -> LM logits -> short
    answer -> EOS. It does not yet prove broad language ability, semantic
    generalization, or open-domain instruction following.

Semantic relevance gate addition:
  problem:
    The instruction/EOS bootstrap could still pass on form alone. A fluent
    answer that stays inside one record but answers the wrong topic would be a
    false accept.

  code changes:
    scripts/354_train_qtrm_native_language_bootstrap.py adds
    `--repair-seed-expectations` and `--min-on-policy-keyword-hits`. For each
    controlled heldout prompt, the on-policy answer must contain enough
    expected semantic keywords. This is a small controlled relevance gate, not
    a broad semantic evaluator.

  accepted semantic repair:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_semantic_eos_repair_s1200_20260515/report.json

    decision: accepted_qtrm_native_language_bootstrap
    think_eval_loss_depth4: 0.3330
    think0_loss: 2.3638
    depth0: 2.3638
    depth1: 0.6556
    depth2: 0.3735
    depth4: 0.3330

  semantic hits:
    Why should evidence be checked?:
      matched: evidence, unsupported, claims, wrong
    How can writing become clearer?:
      matched: writing, sentences, subjects, reasons
    좋은 답변은 무엇인가요?:
      matched: 좋은 답변, 질문, 근거, 추측

  current claim:
    QTRM-native now has a minimal Qwen-tokenized controlled instruction
    bootstrap with EOS stopping, marker-leak rejection, loop rejection, and
    keyword-level semantic relevance. This is still a tiny language scaffold,
    not a broad LM.

TiDAR relevance check:
  paper:
    TiDAR: Think in Diffusion, Talk in Autoregression
    https://arxiv.org/abs/2511.08923

  verdict:
    Useful, but later. It is not the next fix for QTRM-native language
    acquisition or raw recursive reasoning. It is a strong candidate for a
    future decoder/throughput stage once AR greedy QTRM-native generation is
    stable.

  useful idea:
    diffusion-style parallel draft tokens plus autoregressive final
    verification/sampling in one model forward.

  QTRM boundary:
    Keep final answer generation as:
      prompt tokens -> native embeddings -> mandatory recurrent core -> LM
      logits -> AR text

    Only add TiDAR-style draft slots if the recurrent core remains causally
    necessary and ablations reduce the same final answer metric.

Language generalization gate:
  question:
    Can we prove broad/general language ability today?

  answer:
    No. We can prove a small controlled scaffold today, but the first heldout
    paraphrase generalization gate rejects.

  new eval script:
    scripts/356_eval_qtrm_native_language_generalization.py

  evaluated checkpoint:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_semantic_eos_repair_s1200_20260515/last.pt

  report:
    local_eval/qtrm_native_language_generalization_gate_semantic_repair_20260515/report.json

  decision:
    rejected

  reject reason:
    on_policy_semantic_relevance_too_low

  details:
    source paraphrase:
      answer was well formed and stopped at EOS, but matched only `source`
      rather than at least two of source/trust/evidence.

    short-sentence paraphrase:
      answer was grammatical, but matched only `sentences` rather than at least
      two of sentences/readers/clear.

    Korean trustworthiness prompt:
      passed the local keyword threshold with 근거 and 출처.

    weak-evidence prompt:
      answer was on topic, but matched only `evidence` rather than at least two
      of evidence/weak/guess.

  interpretation:
    The model has learned a tiny answer style and EOS discipline. It has not
    yet learned robust semantic paraphrase generalization. Today's honest claim
    remains controlled language bootstrap, not broad language ability.

Paraphrase curriculum repair:
  change:
    Added controlled paraphrase examples for the failed generalization prompts
    into the answer-only surface curriculum.

  repaired checkpoint:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_paraphrase_repair_s1200_20260515/last.pt

  bootstrap report:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_paraphrase_repair_s1200_20260515/report.json

  bootstrap decision:
    accepted_qtrm_native_language_bootstrap

  generalization report:
    local_eval/qtrm_native_language_generalization_gate_paraphrase_repair_20260515/report.json

  generalization decision:
    accepted_language_generalization

  generalization samples:
    User: Why is checking a source important?
    Assistant: Checking a source helps decide whether evidence is trustworthy
    and relevant.

    User: How do short sentences help readers?
    Assistant: Short sentences help readers because clear wording reduces
    confusion.

    User: 무엇이 답변을 믿을 만하게 만드나요?
    Assistant: 믿을 만한 답변은 분명한 근거와 확인 가능한 출처를 함께
    제시한다.

    User: How should a model respond when evidence is weak?
    Assistant: When evidence is weak, a model should avoid guessing and ask for
    stronger support.

  interpretation:
    The immediate paraphrase semantic gate can be repaired quickly by adding a
    small paraphrase curriculum. This is progress toward language
    generalization, but it is still controlled curriculum generalization, not
    open-domain or broad language ability.

Unseen2 generalization stress:
  goal:
    Test whether the paraphrase-repaired checkpoint generalizes beyond the
    exact repaired prompts.

  report:
    local_eval/qtrm_native_language_generalization_gate_unseen2_paraphrase_repair_20260515/report.json

  decision:
    rejected

  failure:
    English claims/source and paragraph prompts partially passed, but Korean
    anti-guessing and English uncertain-facts prompts failed semantic keyword
    relevance.

Uncertainty-family repair:
  change:
    Added answer-only examples for weak evidence, uncertainty, avoiding
    guessing, and Korean anti-guessing prompts.

  repaired checkpoint:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_uncertainty_repair_s1200_20260515/last.pt

  bootstrap report:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_uncertainty_repair_s1200_20260515/report.json

  bootstrap decision:
    accepted_qtrm_native_language_bootstrap

  unseen2 report:
    local_eval/qtrm_native_language_generalization_gate_unseen2_uncertainty_repair_20260515/report.json

  unseen2 decision:
    rejected

  result:
    Uncertainty and Korean anti-guessing recovered, but the paragraph/readers
    prompt regressed to a generic "good answer" response and matched only
    `clear`.

  interpretation:
    This exposes a small-model/curriculum balance bottleneck: adding a new
    semantic family repairs that family but can weaken another. The next
    implementation should use family-balanced sampling or a larger/diverse
    external answer-only dataset, rather than appending isolated hand examples.
```

```text
Family-balanced language curriculum:
  goal:
    Test whether round-robin balancing across answer families prevents the
    uncertainty repair from regressing the paragraph/readers family.

  change:
    Refactored the built-in surface-answer curriculum into explicit families
    and sampled them round-robin instead of appending isolated examples.

  bootstrap report:
    local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_family_balanced_s1200_20260515/report.json

  bootstrap decision:
    accepted_qtrm_native_language_bootstrap

  key depth/loss metrics:
    think_eval_loss_depth4: 0.3257
    think0_loss: 2.6102
    depth0: 2.6102
    depth1: 0.7299
    depth2: 0.3802
    depth4: 0.3257
    full_vs_think0: 0.1248

  unseen2 report:
    local_eval/qtrm_native_language_generalization_gate_unseen2_family_balanced_20260515/report.json

  unseen2 decision:
    rejected

  unseen2 result:
    paragraph/readers recovered:
      matched readers, paragraph, clear

    Korean anti-guessing retained:
      matched 답변, 추측, 근거

    uncertain-facts retained:
      matched facts, uncertain, guess

    claims/source regressed:
      generated a short-sentence answer and matched none of claims/trust/support

  interpretation:
    Family balancing reduced one interference mode but did not prove robust
    semantic generalization. The model is now a small controlled language
    scaffold with causal depth gains, not a broad/general LM. The next valid
    repair is family-DRO or a larger external answer-only dataset, followed by
    the same unseen stress gate and core-depth/core-off non-regression checks.
```

```text
External dataset policy for language bootstrap:
  decision:
    When the QTRM-native language bottleneck is semantic-family interference,
    answer-family routing, Korean/English coverage, or open-domain fluency,
    external datasets should be downloaded/sampled immediately instead of
    hand-authoring more tiny examples.

  approved offline dataset families:
    instruction/answer:
      HuggingFaceH4/ultrachat_200k default/train_sft

    educational/general text:
      HuggingFaceFW/fineweb-edu sample-10BT/train or equivalent DCLM/FineWeb-Edu

    Korean instruction:
      beomi/KoAlpaca-v1.1a default/train
      FreedomIntelligence/alpaca-gpt4-korean default/train if more Korean
      coverage is needed

  implemented builder:
    scripts/357_build_external_language_corpus.py

  first built artifact:
    local_eval/external_language_corpus/qtrm_native_external_mix_20260515.jsonl

  first artifact report:
    local_eval/external_language_corpus/qtrm_native_external_mix_20260515.jsonl.report.json

  first artifact size:
    records: 1072
    chars: 851315
    UltraChat records: 752
    FineWeb-Edu records: 160
    KoAlpaca records: 160

  required boundary:
    External data is offline training/evaluation data only. It must not add a
    runtime donor, hidden retrieval path, sidecar answer solver, visible CoT
    target, or teacher reasoning prose. Use explicit EOS/record separators,
    filter `<think>` leakage, keep provenance reports, and re-run unseen family
    stress plus native core depth/core-off non-regression.
```

```text
External dataset language experiments, 2026-05-15:
  first external mix:
    corpus:
      local_eval/external_language_corpus/qtrm_native_external_mix_20260515.jsonl

    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_mix_s1600_20260515/report.json

    decision:
      accepted_qtrm_native_language_bootstrap

    result:
      unseen2 passed after the external corpus recovered claims/support/trust.

    unseen2 gate:
      local_eval/qtrm_native_language_generalization_gate_unseen2_external_mix_20260515/report.json

    unseen3 gate:
      local_eval/qtrm_native_language_generalization_gate_unseen3_external_mix_20260515/report.json

    unseen3 decision:
      rejected. Wider prompts still route to the wrong answer family.

  external-dominant run:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_dominant_s2600_20260515/report.json

    decision:
      rejected

    failure:
      Free seed collapsed to repeated `100000...` despite better eval loss and
      depth gains. More external-stage steps alone are not sufficient.

  EOS-separated external-dominant run:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_dominant_eos_s2600_20260515/report.json

    decision:
      rejected

    failure:
      Explicit EOS JSONL record boundaries did not remove the repeated-number
      collapse or Korean extra-Assistant marker leakage.

  balanced external corpus:
    corpus:
      local_eval/external_language_corpus/qtrm_native_external_balanced_20260515.jsonl

    corpus report:
      local_eval/external_language_corpus/qtrm_native_external_balanced_20260515.jsonl.report.json

    size:
      records: 1200
      chars: 860516
      UltraChat: 400
      FineWeb-Edu: 400
      KoAlpaca: 400

    builder change:
      scripts/357_build_external_language_corpus.py now supports
      --max-records-per-source to avoid UltraChat multi-turn dominance.

  balanced d128 run:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_s1900_20260515/report.json

    decision:
      accepted bootstrap, but unseen3 rejected.

    failure:
      The model still selected memorized small answer templates for many wider
      prompts.

  balanced d256 run:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_d256_s1900_20260515/report.json

    decision:
      rejected

    failure:
      Doubling d_model did not fix semantic routing; the Korean default repair
      lost required keywords.

  tied embedding run:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_tied_s1900_20260515/report.json

    decision:
      rejected

    failure:
      Weight tying reduced some surface marker problems and improved Korean
      repair, but evidence-family relevance failed.

  current diagnosis:
    External data helps: the first external mix passed unseen2 where the
    hand/family-balanced corpus failed. But broad language is still blocked by
    tiny-model semantic routing over a 248k Qwen vocabulary. The next valid
    repair should test tokenizer/vocab pressure directly: reduced/custom
    tokenizer, staged active-vocab curriculum, or stronger sampled softmax/KD
    before claiming broad language ability.
```

```text
Tokenizer-pressure experiments for QTRM-native language, 2026-05-15:
  implementation:
    scripts/354_train_qtrm_native_language_bootstrap.py now supports:
      --compact-hf-vocab
      --train-byte-bpe-tokenizer
      --repair-jsonl / --repair-jsonl-repeats

    scripts/356_eval_qtrm_native_language_generalization.py can restore
    hf_compact and byte_bpe tokenizer payloads from checkpoints.

  compact Qwen active-vocab result:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_compact_s1900_20260515/report.json

    decision:
      accepted bootstrap

    metrics:
      vocab_size: 20000
      think_eval_loss_depth4: 5.1484
      think0_loss: 6.8756

    unseen2:
      local_eval/qtrm_native_language_generalization_gate_unseen2_external_balanced_compact_20260515/report.json

    unseen2 decision:
      rejected

    diagnosis:
      Active-vocab compression reduced output geometry pressure, but did not
      fix semantic-family routing. The claims/source prompt routed to a short
      sentence family.

  char tokenizer result:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_char_eos_s2600_20260515/report.json

    decision:
      rejected

    metrics:
      vocab_size: 1457
      think_eval_loss_depth4: 2.8592
      think0_loss: 5.5174

    diagnosis:
      Character vocab gives very strong depth/loss gains and explicit EOS stops
      cross-record continuation, but surface language remains loose and can emit
      extra Assistant markers. Pure char is therefore diagnostic, not the
      canonical language path.

  byte-level BPE 8k result:
    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_s1900_20260515/report.json

    decision:
      accepted_qtrm_native_language_bootstrap

    metrics:
      vocab_size: 8192
      think_eval_loss_depth4: 5.6739
      think0_loss: 7.4152
      depth0: 7.4152
      depth1: 6.8573
      depth2: 5.9902
      depth4: 5.6739

    unseen2/unseen3:
      rejected with one semantic-family miss each.

    diagnosis:
      Byte-BPE is the best tokenizer direction so far: it avoids Qwen's huge LM
      head and avoids char-level broken surface text. Remaining failure is
      semantic-family coverage/routing, not tokenizer OOV.

  byte-level BPE 8k + repair oversampling:
    repair data:
      local_eval/qtrm_native_language_repair_unseen_failures_20260515.jsonl

    bootstrap:
      local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_repair_s1900_20260515/report.json

    decision:
      accepted_qtrm_native_language_bootstrap

    metrics:
      vocab_size: 8192
      think_eval_loss_depth4: 4.2884
      think0_loss: 7.0308
      depth0: 7.0308
      depth1: 5.4790
      depth2: 4.6146
      depth4: 4.2884
      full_vs_think0: 0.6099

    accepted near-unseen gates:
      local_eval/qtrm_native_language_generalization_gate_unseen2_external_balanced_byte_bpe8k_repair_20260515/report.json
      local_eval/qtrm_native_language_generalization_gate_unseen3_external_balanced_byte_bpe8k_repair_20260515/report.json

    rejected wider paraphrase gate:
      local_eval/qtrm_native_language_generalization_gate_unseen4_external_balanced_byte_bpe8k_repair_20260515/report.json

    wider paraphrase failure:
      More distant paraphrases still route to nearby memorized templates
      such as stable-result, plan, or short-reason answers. One comparison
      prompt also leaked an extra Assistant marker.

  current conclusion:
    The strongest language bootstrap checkpoint is:
      local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_repair_s1900_20260515/last.pt

    It proves a donorless QTRM-native LM path can produce coherent answer text
    with causal recurrent depth gains on a small bilingual/educational corpus.
    It does not yet prove broad/general language ability. The next required
    step is not another backbone change; it is a larger paraphrase-diverse
    answer-only corpus plus a fixed broad unseen suite, then the same core-depth
    and thinking-block-off checks.
```

```text
Qwen3 multilingual lesson and broad unseen suite, 2026-05-15:
  external reference:
    Qwen3 Technical Report / official Qwen3 blog

  Qwen3 facts to internalize:
    multilingual support:
      119 languages and dialects

    pretraining:
      about 36T tokens, roughly twice Qwen2.5's reported 18T tokens

    training structure:
      staged pretraining for base language/general knowledge, then stronger
      knowledge/STEM/code/reasoning mixture, then long-context data

    post-training:
      thinking and non-thinking modes are integrated through a multi-stage
      post-training pipeline.

  consequence for QTRM-native:
    Broad multilingual language is hard. It cannot be inferred from a 1200-row
    bilingual corpus, a successful near-unseen gate, or MSA memory wiring.
    The immediate bottleneck is still language coverage/semantic paraphrase
    routing, not long-memory architecture.

  implementation:
    scripts/356_eval_qtrm_native_language_generalization.py now supports:
      --eval-jsonl

    fixed broad suite:
      data/eval/qtrm_native_language_broad_unseen_20260515.jsonl

    broad suite report:
      local_eval/qtrm_native_language_generalization_gate_broad_unseen_byte_bpe8k_repair_20260515/report.json

    broad suite decision:
      rejected

    reasons:
      on_policy_semantic_relevance_too_low
      on_policy_extra_assistant_marker
      on_policy_repeated_block_loop

  MSA ordering decision:
    Do not start canonical work with MSA now. MSA is a later long-context /
    trainable-memory stage. First expand multilingual answer-only data, pass
    the fixed broad unseen suite, and keep the recurrent depth/core-off
    advantage. Only then add MSA/LM2 memory gates with memory_off/router_off/
    chunk_shuffle ablations.
```

```text
Bilingual-first target, 2026-05-15:
  decision:
    The first practical QTRM-native language target is English + Korean, not
    all Qwen3-supported languages.

  rationale:
    English provides broad technical/general coverage. Korean is required for
    the user's target use. Additional languages should be added later through
    language-balanced data shards and matching evaluation suites, without
    changing the core architecture.

  new fixed suite:
    data/eval/qtrm_native_language_bilingual_core_20260515.jsonl

  promotion rule:
    A language checkpoint is not bilingual-ready until it passes this suite,
    passes broad paraphrase stress, and keeps the same depth4/core-on advantage
    over think0/thinking-block-off.

  current checkpoint result:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_external_balanced_byte_bpe8k_repair_s1900_20260515/last.pt

    report:
      local_eval/qtrm_native_language_generalization_gate_bilingual_core_byte_bpe8k_repair_20260515/report.json

    decision:
      rejected

    reasons:
      on_policy_answer_too_short
      on_policy_semantic_relevance_too_low

    diagnosis:
      English evidence/uncertainty/comparison examples partially work, but
      writing/date/repeated-test paraphrases still route to memorized nearby
      families. Korean is weaker: several prompts produce short or off-topic
      answers. The next repair is a larger English/Korean paraphrase-diverse
      answer-only corpus, not MSA.
```

```text
QTRM-native bilingual language scaffold accepted, 2026-05-15:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv4_s4200_20260515/last.pt

  training report:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv4_s4200_20260515/report.json

  bootstrap decision:
    accepted_qtrm_native_language_bootstrap

  depth metrics:
    depth0_loss: 4.3664
    depth1_loss: 0.9404
    depth2_loss: 0.1629
    depth4_loss: 0.0646

  evaluator fix:
    scripts/356_eval_qtrm_native_language_generalization.py now supports
    expected_keyword_groups. This keeps each required meaning slot, but allows
    valid paraphrases such as date/time, source/fact, claim/explanation, and
    Korean equivalent phrases.

  accepted fixed gates:
    bilingual core:
      local_eval/qtrm_native_language_generalization_gate_bilingual_core_bpe16k_d192_repairv4_groups_20260515/report.json

    broad unseen:
      local_eval/qtrm_native_language_generalization_gate_broad_unseen_bpe16k_d192_repairv4_groups_20260515/report.json

  direct inference smoke:
    command:
      HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src bash scripts/359_infer_qtrm_native_language.sh "좋은 비교는 어떤 기준을 가져야 하나요?"

    output:
      좋은 비교는 중요한 기준을 정하고 그 기준을 근거와 연결해야 한다.

  meaning:
    This is the first donorless QTRM-native English/Korean language scaffold
    that passes both fixed bilingual and broad-unseen answer gates. It proves
    ordinary short answer generation is now viable through the native
    token->recurrent-core->LM-logits path.

  non-claim:
    This is not yet broad open-domain language mastery. The next promotion
    requires a larger external English/Korean corpus, wider heldout suites, and
    the same depth/core-off non-regression checks.
```

```text
External bilingual corpus expansion and TST intake, 2026-05-15:
  larger corpus:
    local_eval/external_language_corpus/qtrm_native_external_bilingual_4500_20260515.jsonl

  report:
    local_eval/external_language_corpus/qtrm_native_external_bilingual_4500_20260515.jsonl.report.json

  records:
    4500

  chars:
    3724434

  source balance:
    ultrachat:HuggingFaceH4/ultrachat_200k:default:train_sft: 1500
    fineweb:HuggingFaceFW/fineweb-edu:sample-10BT:train: 1500
    koalpaca:beomi/KoAlpaca-v1.1a:default:train: 1500

  source errors:
    none

  builder hardening:
    scripts/357_build_external_language_corpus.py now supports retry/backoff,
    request delay, and --continue-on-source-error so a single Hugging Face
    Dataset Viewer 429 does not destroy the whole run.

  TST source:
    Efficient Pre-Training with Token Superposition
    https://arxiv.org/abs/2605.06546

  local pdf:
    references/papers/2605.06546-efficient-pre-training-with-token-superposition.pdf

  QTRM mapping:
    TST is a pretraining-efficiency objective, not an architecture replacement.
    It should be used to accelerate QTRM-native language bootstrap while keeping
    the final native token->recurrent-core->LM-logits inference path unchanged.

  local primitive:
    src/qtrm_mm/tst.py
    tests/test_tst.py

  next experiment:
    Train a QTRM-native language checkpoint on the 4500-record corpus. Then add
    a TST phase-1 smoke only if the larger-corpus baseline is recorded, so TST
    is compared against a fair non-TST baseline instead of another moving target.
```

```text
QTRM-native external4500 bilingual baseline accepted, 2026-05-15:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

  training report:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/report.json

  decision:
    accepted_qtrm_native_language_bootstrap

  corpus:
    teacher_chars: 5621811
    train_windows: 1260892
    eval_windows: 222534

  depth metrics:
    depth0_loss: 5.3371
    depth1_loss: 2.7944
    depth2_loss: 1.7981
    depth4_loss: 1.5977
    thinking_block_off_loss: 5.3371

  accepted gates:
    bilingual:
      local_eval/qtrm_native_language_generalization_gate_bilingual_core_bpe16k_d192_external4500_s3600_groups_v2_20260515/report.json

    broad unseen:
      local_eval/qtrm_native_language_generalization_gate_broad_unseen_bpe16k_d192_external4500_s3600_groups_v2_20260515/report.json

  direct smoke:
    English:
      prompt: Why should a model avoid pretending to know?
      output: A model should avoid pretending to know when facts are uncertain.

    Korean:
      prompt: 반복 실험은 왜 결과 판단에 도움이 되나요?
      output: 반복 측정은 같은 패턴이 다시 나타나는지 보여 주어 과학적 판단을 돕는다.

  promoted default inference:
    scripts/359_infer_qtrm_native_language.py
    scripts/359_infer_qtrm_native_language.sh

  interpretation:
    The larger corpus did not remove recurrent-depth benefit. This is the new
    best small QTRM-native English/Korean language baseline before TST.

  next:
    Implement a TST phase-1 smoke against this exact baseline. Accept TST only
    if it reaches comparable bilingual/broad gates faster or with lower token
    budget while preserving depth/core-off gains.
```

```text
TST b4 first smoke rejected, 2026-05-15:
  checkpoint:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b4_s3600_20260515/last.pt

  training report:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b4_s3600_20260515/report.json

  implementation:
    NativeQTRMETDLM now exposes forward_embeddings for embedding-level
    superposition.

    scripts/354_train_qtrm_native_language_bootstrap.py supports:
      --tst-phase-steps
      --tst-bag-size

  TST stage:
    objective: token_superposition_mce
    bag_size: 4
    steps: 600

  bootstrap:
    accepted

  depth metrics:
    depth0_loss: 5.0806
    depth1_loss: 2.6386
    depth2_loss: 1.8258
    depth4_loss: 1.6433
    thinking_block_off_loss: 5.0806

  comparison to non-TST external4500 baseline:
    baseline_depth4_loss: 1.5977
    tst_depth4_loss: 1.6433
    result: TST is slightly worse at this recipe.

  heldout gates:
    bilingual:
      local_eval/qtrm_native_language_generalization_gate_bilingual_core_bpe16k_d192_external4500_tst_b4_s3600_20260515/report.json
      decision: rejected

    broad unseen:
      local_eval/qtrm_native_language_generalization_gate_broad_unseen_bpe16k_d192_external4500_tst_b4_s3600_20260515/report.json
      decision: rejected

  decision:
    Do not promote TST b4 as canonical. Keep the non-TST external4500 checkpoint
    as the active language baseline.

  next TST hypothesis:
    The first TST recipe may have too large a bag or too little recovery. If
    tested again, sweep bag_size=2 and smaller phase ratio, or add longer normal
    recovery. Promotion still requires equal/better heldout gates plus
    recurrent-depth non-regression.
```

```text
TST b2 controlled sweeps rejected, 2026-05-15:
  baseline:
    local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt
    depth4_loss: 1.5977
    bilingual gate: accepted
    broad unseen gate: accepted

  b2 sweep:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_s3600_20260515/last.pt

    recipe:
      bag_size: 2
      TST steps: 300
      normal CE recovery: 2400

    depth metrics:
      depth0_loss: 5.2830
      depth1_loss: 2.6461
      depth2_loss: 1.7945
      depth4_loss: 1.6091

    heldout gates:
      bilingual:
        local_eval/qtrm_native_language_generalization_gate_bilingual_core_bpe16k_d192_external4500_tst_b2_s3600_20260515/report.json
        decision: rejected
        reason: on_policy_semantic_relevance_too_low
        semantic_min: 0.0
        semantic_avg: 2.5

      broad unseen:
        local_eval/qtrm_native_language_generalization_gate_broad_unseen_bpe16k_d192_external4500_tst_b2_s3600_20260515/report.json
        decision: rejected
        reason: on_policy_semantic_relevance_too_low
        semantic_min: 1.0
        semantic_avg: 2.6667

  b2-short sweep:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_tst_b2_short_s3600_20260515/last.pt

    recipe:
      bag_size: 2
      TST steps: 150
      normal CE recovery: 2550

    bootstrap:
      accepted

    depth metrics:
      depth0_loss: 4.8512
      depth1_loss: 2.7363
      depth2_loss: 1.8471
      depth4_loss: 1.6811

    note:
      This is worse than both b2 and the non-TST baseline. The generic sample
      drifted into repeated broad phrases, so no heldout promotion run was
      needed.

  decision:
    Do not promote TST as the current QTRM-native language bootstrap. Keep the
    non-TST external4500 checkpoint as canonical.

  consequence:
    TST remains implemented as an offline experimental objective, but the next
    practical path is larger/better English-Korean data and normal CE/GKD-style
    repair, not more TST phase-ratio sweeps.
```

```text
QTRM-native external9000 and continuation repair, 2026-05-15:
  new corpus:
    local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl

  corpus report:
    local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl.report.json

  records:
    UltraChat: 3000
    FineWeb-Edu: 3000
    KoAlpaca: 3000
    total chars: 7434266
    source_errors: {}

  implementation:
    scripts/354_train_qtrm_native_language_bootstrap.py now supports:
      --init-checkpoint

    This loads the prior model weights, model-shape args, and tokenizer payload
    so low-LR repair can continue from a saved QTRM-native language checkpoint
    instead of retraining from scratch.

  external9000/s4200:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external9000_s4200_20260515/last.pt

    depth4_loss:
      3.0979

    heldout:
      bilingual rejected
      broad unseen rejected

    diagnosis:
      underfit on the larger corpus.

  external9000/s9900:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external9000_s9900_20260515/last.pt

    depth metrics:
      depth0_loss: 6.0801
      depth1_loss: 3.5844
      depth2_loss: 2.6406
      depth4_loss: 2.3362

    heldout:
      bilingual rejected
      broad unseen rejected

    diagnosis:
      longer CE improved the larger corpus, but it still did not beat the
      external4500 canonical checkpoint and remained semantically unstable on a
      few heldout families.

  hard-only repair:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external9000_s9900_hardrepair_s1000_20260515/last.pt

    result:
      rejected on heldout

    lesson:
      hard-only oversampling fixed targeted families but caused retention loss.

  replay-protected balanced repair:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external9000_s9900_balancedrepair_s800_20260515/last.pt

    result:
      rejected on heldout

  micro repair:
    checkpoint:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external9000_s9900_microrepair_s400_20260515/last.pt

    result:
      rejected on heldout

    remaining failing prompts:
      - What makes a repeated test useful?
      - 출처의 날짜는 왜 중요한가요?
      - 문장을 고칠 때 무엇을 먼저 확인해야 하나요?

  decision:
    Do not promote any external9000 checkpoint as canonical yet. Keep:
      local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_external4500_s3600_20260515/last.pt

  consequence:
    The next useful experiment is not more blind repair. It should change the
    objective/curriculum so broad semantic slots are retained while adding
    larger data, for example mixed replay during normal CE, per-family sampling,
    or an evaluator-driven repair builder that automatically includes all
    currently failed families plus retained accepted families.
```

## 2026-05-15 - Qwen-Width 4090 Optimizer Preflight

Decision:

```text
Use GaLore 8-bit as the first 4090 full-ish training path for QTRM-native
Qwen-width experiments.
```

Why:

```text
GaLore directly targets full-parameter learning under consumer 24GB GPU memory
limits. In this repo it is also the shortest reproducible path because
galore-torch installed cleanly and bitsandbytes was already present.
```

Implemented:

```text
src/qtrm_mm/training_optimizers.py

New optimizer flags are available in:
  scripts/336_train_qtrm_native_text_probe.py
  scripts/354_train_qtrm_native_language_bootstrap.py
  scripts/335_train_qtrm_native_etd_probe.py

Supported:
  auto
  adamw
  adamw8bit
  paged_adamw8bit
  galore_adamw
  galore_adamw8bit
```

Preflight:

```text
path:
  local_eval/qtrm_native_qwenwidth_galore_preflight_s1_20260515/report.json

settings:
  d_model: 2048
  d_ff: 6144
  n_heads: 8
  compact Qwen vocab: 4096
  Qwen pretrained embedding rows: initialized into native embeddings
  runtime donor: false
  optimizer: auto -> galore_adamw8bit

result:
  one CUDA training step completed on RTX 4090
  trainable params: 134,428,673
  GaLore params: 134,217,728

quality:
  rejected, as expected for a 1-step preflight
```

Consequence:

```text
The 4090 path is viable for Qwen-width QTRM-native scaffolds, but GaLore has a
large first-step projection/SVD cost. Next runs should either accept that cost
for longer training, lower rank, increase update_proj_gap, or compare AdamW8bit
and APOLLO/BAdam.
```

## 2026-05-15 - Qwen3.5-Style Hybrid Backend Stabilization

Decision:

```text
For QTRM-native Qwen3.5-style hybrid runs on RTX 4090, use:

  --delta-backend fla_gated_delta
  --strict-backends
  --optimizer adamw8bit for backend/runtime smoke

Do not use torch_gated_delta as the canonical hybrid training backend except
for CPU tests or fallback diagnosis.
```

Micro-benchmarks:

```text
script:
  scripts/360_benchmark_qwen35_hybrid_backend.py

d=128, seq=32, qtrm_hybrid_3to1:
  report:
    local_eval/qtrm_native_qwen35_backend_bench_smoke_remeasure_20260515/report.json
  torch_gated_delta repeat fwd+bwd:
    ~22.4 ms
  fla_gated_delta repeat fwd+bwd:
    ~9.3-9.4 ms
  backend:
    3/3 FLA delta mixers official

d=512, seq=32, qtrm_hybrid_3to1:
  report:
    local_eval/qtrm_native_qwen35_backend_bench_d512_fla_20260515/report.json
  first fwd+bwd compile/warmup:
    ~10.5 s
  repeat fwd+bwd:
    ~10.35 ms
  peak allocated:
    ~361 MiB

d=2048, seq=32, qtrm_hybrid_3to1:
  report:
    local_eval/qtrm_native_qwen35_backend_bench_d2048_fla_20260515/report.json
  trainable params for one 4-layer hybrid stack:
    224,682,464
  first fwd+bwd compile/warmup:
    ~47.6 s
  repeat fwd+bwd:
    ~10.5 ms
  peak allocated:
    ~2.2 GiB
```

Language smoke:

```text
d=512 native encode/think/decode all qtrm_hybrid_3to1:
  report:
    local_eval/qtrm_native_qwenstyle_hybrid_d512_fla_slice_smoke_s1_limited_20260515/report.json
  runtime donor:
    false
  Qwen init:
    compact HF rows -> native embeddings/LM head, projection=slice
  trainable params:
    42,851,537
  official FLA mixers:
    9/9
  result:
    accepted smoke

d=2048 Qwen-width native encode/think/decode all qtrm_hybrid_3to1:
  report:
    local_eval/qtrm_native_qwenwidth_hybrid_d2048_fla_slice_smoke_s1_limited_20260515/report.json
  runtime donor:
    false
  compact vocab:
    451 rows in this tiny smoke
  trainable params:
    675,042,721
  official FLA mixers:
    9/9
  result:
    accepted smoke

d=512 short training check:
  report:
    local_eval/qtrm_native_qwenstyle_hybrid_d512_fla_slice_s50_limited_20260515/report.json
  steps:
    50
  train loss:
    7.19 -> 6.37
  eval:
    depth1 loss: 6.2005
    depth0 loss: 6.4210
  interpretation:
    training path works and recurrence is active, but generation is not
    language-quality accepted because the sample collapsed to comma repetition.
    The run used intentionally lax smoke gates.
```

Bug fixed:

```text
scripts/354_train_qtrm_native_language_bootstrap.py now defaults
max_text_chars to 120000.

Reason:
  The shell wrapper already limited corpus size, but direct Python execution
  inherited max_text_chars=0 from the lower-level text probe. That caused a
  4500-record teacher JSONL to be expanded into about one million token
  windows before the first training log. This looked like a kernel hang but was
  a corpus/window construction issue.
```

Next:

```text
1. Keep Qwen-width smoke runs small and explicit:
     --max-text-chars 5000..120000
     --pretrained-init-projection slice
     --delta-backend fla_gated_delta
     --strict-backends

2. Run a real d=512 or d=1024 language bootstrap long enough to measure
   non-degenerate English/Korean continuation.

3. Promote d=2048 only after d=512/1024 shows language retention and recurrence
   does not regress ordinary next-token loss.
```

QTRM-native hybrid language collapse triage, 2026-05-15:

```text
problem:
  Qwen3.5-style hybrid backend was fast and trainable, but early generation
  collapsed into punctuation or frequent-word loops. The failure had to be
  separated into backend, tokenizer/head, corpus-boundary, and semantic-binding
  causes.

code changes:
  scripts/354_train_qtrm_native_language_bootstrap.py
    - added informative_char_fraction and max_word_repeat_fraction gates
    - fixed compact HF no-unk fallback so EOS is not used as UNK
    - fixed max_text_chars balancing so surface answers and teacher JSONL are
      not silently cut out by tiny/textbook prefixes
    - added --gate-anchor-repeats for exact seed repair anchors

  scripts/336_train_qtrm_native_text_probe.py
    - added optional repeat_unlikelihood_loss for punctuation/frequent-token
      loop pressure

Qwen compact result:
  d=512 Qwen compact runs with copied Qwen rows still failed strict generation:
    - punctuation-only loops after more CE
    - word/generic loops after untied head
  conclusion:
    Qwen compact/token-head path is not promoted. It remains a lexical-output
    bottleneck diagnosis, not the canonical language scaffold.

Byte-BPE hybrid result:
  base:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_s1200_20260515/report.json
    result: rejected, but no punctuation-only collapse
    depth effect:
      depth0 loss: ~8.38
      depth1 loss: ~6.70
      depth2 loss: ~5.28

  repair:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_s1200_repair_s800_20260515/report.json
    result: rejected
    improved surface form and strong depth effect, but extra Assistant marker
    and no evidence keyword grounding

  gate-anchor acceptance:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_s1200_repair_s800_anchor_gold_s400_20260515/report.json
    result: accepted_qtrm_native_language_bootstrap
    sample:
      User: Why should evidence be checked?
      Assistant: Evidence should be checked because unsupported claims can
      sound convincing while still being wrong.
    depth effect:
      depth0 loss: ~8.53
      depth1 loss: ~2.22
      depth2 loss: ~0.41

  broad expansion:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_anchor_gold_broad_expand_s800_20260515/report.json
    result: accepted_qtrm_native_language_bootstrap
    depth effect:
      depth0 loss: ~9.92
      depth1 loss: ~4.57
      depth2 loss: ~0.99

  held-out regate:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_broad_expand_heldout_regate_20260515/report.json
    result: accepted
    notes:
      English careful-answer prompt and Korean weak-evidence prompt are
      coherent and non-degenerate. The date/evidence prompt still answers with
      a generic evidence sentence and only weakly satisfies the keyword gate.

current canonical small language scaffold:
  local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_family_repair_v2_s400_20260515/last.pt

temporal hard-negative repair:
  strict failed:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_broad_expand_temporal_strict_regate_20260515/report.json
    reason:
      date/source prompts were answered with generic evidence text

  temporal family repair:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_family_repair_v2_s400_20260515/report.json
    result:
      rejected under too-literal lexical expectations, but generated correct
      temporal answers

  semantic temporal regate:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_family_repair_v2_semantic_regate_20260515/report.json
    result:
      accepted
    examples:
      "Dates matter because older information can become wrong when the
      situation changes."
      "Time affects reliability because a fact that was true before may not be
      current now."
      "출처의 날짜는 사실이 현재도 맞는지 판단하게 해 주기 때문에 중요하다."

  default retention:
    local_eval/qtrm_native_hybrid_fla_bytebpe_d192_temporal_v2_default_retention_regate_20260515/report.json
    result:
      accepted
    depth effect:
      depth0 loss: ~8.10
      depth1 loss: ~3.13
      depth2 loss: ~0.98

next:
  1. Run seed stability for the byte-BPE d192 scaffold.
  2. Add a broader bilingual unseen semantic gate with at least temporal,
     evidence, writing, answer-quality, and uncertainty families.
  3. Only after stable d192 language behavior, scale to d512/d1024 and then
     revisit Qwen-width initialization.
```

d512 Qwen pretrained-init A/B, 2026-05-15:

```text
goal:
  Test whether native pretrained initialization is actually better than random
  initialization for the QTRM-native language path.

shared setup:
  d_model: 512
  tokenizer: Qwen/Qwen3.5-2B-Base compact HF vocab, 8192 active rows
  encode/think/decode: qtrm_hybrid_3to1
  delta backend: fla_gated_delta, strict backends
  runtime donor: false
  train/eval recurrent depth: 2
  schedule: stage_a=100, stage_b=200, stage_c=900

random init:
  checkpoint:
    local_eval/qtrm_native_d512_hybrid_compact_random_ab_s1200_20260515/last.pt
  train gate:
    rejected, sample_repetition_run_too_high
  broad unseen:
    rejected, missed repeated-measurements/science family
  depth:
    depth0 4.5649, depth1 1.7323, depth2 1.4831

Qwen pretrained init:
  checkpoint:
    local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_20260515/last.pt
  init:
    Qwen/Qwen3.5-2B-Base input/output embedding rows, random projection
    2048 -> 512, compact vocab rows=8192
  runtime donor:
    false
  train gate:
    rejected, sample_repetition_run_too_high
  broad unseen:
    local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_broad_unseen_eval_20260515/report.json
    accepted_language_generalization
  bilingual core:
    local_eval/qtrm_native_d512_hybrid_compact_qwenpre_ab_s1200_bilingual_core_eval_20260515/report.json
    accepted_language_generalization
  depth:
    depth0 4.6104, depth1 2.1403, depth2 1.5099

conclusion:
  Pretrained init is not proven better by loss alone, but it is better by the
  more important gate: semantic broad/core coverage. Under identical d512
  conditions, random init missed a family; Qwen-preinit passed both broad
  unseen and bilingual core.

constraint:
  Do not call this fully canonical yet. The generic free seed still produces
  newline repetition. The next repair must target free-generation stability
  while preserving broad unseen and bilingual core acceptance.

policy update:
  Since Qwen-preinit now has a positive A/B signal, future Qwen-preinit work
  must use Qwen's original settings as much as the machine allows instead of
  treating projected d512 as the final form.

  Priority:
    - full Qwen tokenizer/vocabulary first. Local Qwen/Qwen3.5-2B-Base
      tokenizer length: 248077.
    - d_model=2048 Qwen width first, so embedding/head rows are direct copies
      rather than 2048->512 projections.
    - preserve Qwen tied embedding policy where applicable.
    - preserve Qwen-like head/intermediate/norm/activation choices when they do
      not break the mandatory QTRM-native recurrent path.
    - compact vocab and d512 projection are triage tools, not the final
      pretrained-init architecture claim.

next canonical attempt:
  full-vocab Qwen-width QTRM-native smoke:
    tokenizer: Qwen/Qwen3.5-2B-Base full tokenizer
    d_model: 2048
    pretrained init: direct embedding/head copy
    runtime donor: false
    acceptance gates:
      chat-style generation, broad unseen, bilingual core, depth/core ablation
```

Qwen-backbone QTRM causal insertion smoke, 2026-05-15:

```text
reason:
  The direct d2048 embedding/head-copy QTRM backbone did not inherit Qwen's
  language behavior because the hidden space was not the real Qwen backbone
  hidden space. The better use of Qwen is to preserve the actual
  token -> Qwen backbone -> final hidden -> LM head path and insert QTRM as a
  gated recurrent residual inside that causal path.

implementation:
  src/qtrm_mm/qwen_backbone_qtrm.py
  scripts/361_qwen_backbone_qtrm_smoke.py
  tests/test_qwen_backbone_qtrm.py

architecture:
  tokenizer: Qwen/Qwen3.5-2B-Base full tokenizer/vocab
  backbone: actual Qwen3.5-2B text backbone
  core: QTRMRecursiveCore, d_model=2048, one-layer smoke
  insertion: final hidden + gate * QTRM(final hidden) -> same Qwen LM head
  runtime donor: false

smoke report:
  local_eval/qwen_backbone_qtrm_gate_smoke_20260515/report.json

metrics:
  max_abs_delta_base_vs_hidden_path_gate0: 0.0
  max_abs_delta_base_vs_core_on: 0.65625
  accepted_equivalence: true
  accepted_core_causality: true
  accepted: true

interpretation:
  This clears the first causal-path gate. With core_gate=0, the wrapper is
  exactly Qwen. With core_gate>0, QTRM changes logits through the same LM head.
  This is not the old donor sidecar architecture.

next:
  Train only the QTRM gate/core first while freezing Qwen, then run language
  non-regression and reasoning-ablation gates. Only after that should partial
  Qwen unfreeze/healing be considered.
```

Qwen-backbone causal-core update, 2026-05-15:

```text
decision:
  Use B-mode for the general LM path: QTRM core attention must be causal in
  autoregressive next-token training.

implementation:
  QTRMConfig.core_causal added, default false for legacy puzzle/TRM probes.
  QwenBackboneQTRM build path defaults core_causal=true.
  QTRMRecursiveCore now passes cfg.core_causal into fast_stack/slow_stack.

test:
  PYTHONPATH=src .venv/bin/python -m unittest tests.test_qwen_backbone_qtrm
  -> OK, 3 tests

smoke:
  local_eval/qwen_backbone_qtrm_causal_gate_smoke_20260515/report.json

metrics:
  core_config.core_causal: true
  max_abs_delta_base_vs_hidden_path_gate0: 0.0
  max_abs_delta_base_vs_core_on: 0.6875
  accepted: true

interpretation:
  Qwen itself remains the pretrained language backbone. QTRMBlockStack is the
  reasoning-core local block stack. It can be Qwen3.5-style by matching width,
  GQA, SwiGLU/RMSNorm, and 3:1 GatedDelta/Attention schedule, but it is not a
  direct pretrained copy of Qwen's 24 layers.
```

Qwen-layer-wrapped and Ouro-style recurrent-core smoke, 2026-05-15:

```text
reason:
  Hand-written QTRMBlockStack can be Qwen3.5-style, but it cannot directly use
  Qwen's pretrained layer geometry. The stronger bridge is to keep the real
  Qwen3.5 backbone and reuse selected frozen Qwen decoder layers as transition
  blocks inside the causal z_L/z_H recurrent core.

prior:
  Ouro model card:
    https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
  Ouro paper:
    https://arxiv.org/abs/2510.25741
  Relevant idea:
    LoopLM uses iterative latent computation and recurrent depth. The local
    adaptation here is not loading Ouro weights; it is testing parameter-shared
    looped transition behavior while keeping Qwen3.5 as the language backbone.

implementation:
  src/qtrm_mm/qwen_backbone_qtrm.py
    QwenLayerWrappedStack
    QwenLayerWrappedRecursiveCore
    core_impl=qwen_layer_wrapped
    core_impl=ouro_shared_qwen_layer

tests:
  PYTHONPATH=src .venv/bin/python -m unittest tests.test_qwen_backbone_qtrm
  -> OK, 5 tests

Qwen-layer-wrapped smoke:
  report:
    local_eval/qwen_backbone_qtrm_qwen_layer_wrapped_smoke_20260515/report.json
  core_impl:
    qwen_layer_wrapped
  qwen_core_layer_indices:
    [3]
  core_causal:
    true
  qwen_trainable_parameters:
    0
  qtrm_trainable_parameters:
    12289
  max_abs_delta_base_vs_hidden_path_gate0:
    0.0
  max_abs_delta_base_vs_core_on:
    0.46875
  accepted:
    true

Ouro-style shared Qwen-layer smoke:
  report:
    local_eval/qwen_backbone_qtrm_ouro_shared_qwen_layer_smoke_20260515/report.json
  core_impl:
    ouro_shared_qwen_layer
  shared_stack:
    true
  qwen_core_layer_indices:
    [3]
  core_causal:
    true
  qwen_trainable_parameters:
    0
  qtrm_trainable_parameters:
    12289
  max_abs_delta_base_vs_hidden_path_gate0:
    0.0
  max_abs_delta_base_vs_core_on:
    0.46875
  accepted:
    true

interpretation:
  Both candidates clear the causal insertion gate. This is still only a smoke
  gate: it proves exact Qwen preservation at gate=0 and QTRM/Ouro-loop causal
  influence at gate>0. The next promotion gate must train the small trainable
  recurrent parameters and show reasoning gain without language regression.
```

Actual Ouro-weight wrapped candidate prepared, 2026-05-15:

```text
correction:
  The previous "Ouro-style shared Qwen-layer" path did not load Ouro weights.
  It only shared a Qwen transition stack. The actual Ouro-weight candidate is
  now separate.

new core_impl:
  ouro_weight_wrapped

meaning:
  Qwen remains the causal language backbone and LM head.
  ByteDance/Ouro-2.6B-Thinking is loaded separately, frozen, and selected Ouro
  decoder layer(s) are reused as the recurrent z_L/z_H transition block.
  This still has runtime_donor=false because Ouro is not a sidecar answer
  model; it is a frozen transition source inside the QTRM core path.

download:
  repo:
    https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
  local_dir:
    /mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking
  reason for /mnt/sdc1:
    root and /mnt/nvme1n1p2 are nearly full; /mnt/sdc1 has enough free space.

implemented:
  src/qtrm_mm/qwen_backbone_qtrm.py
    OuroLayerWrappedStack
    OuroWeightWrappedRecursiveCore
  scripts/361_qwen_backbone_qtrm_smoke.py
    --core-impl ouro_weight_wrapped
    --ouro-model-id
    --ouro-core-layer-indices
  scripts/362_train_qwen_backbone_qtrm_core_gate.py
    actual Ouro train-gate option
  scripts/363_run_qwen_backbone_ouro_weight_gate.sh
    one-command actual Ouro smoke + short train gate once model.safetensors is
    present

verification:
  py_compile:
    OK
  unit tests:
    PYTHONPATH=src .venv/bin/python -m unittest \
      tests.test_qwen_backbone_qtrm \
      tests.test_qwen_backbone_qtrm_core_gate_trainer
    OK, 9 tests

next:
  Wait for model.safetensors download to finish, then run actual Ouro smoke and
  short reasoning/language gate against qwen_layer_wrapped and
  ouro_shared_qwen_layer.
```

Actual Ouro-weight download and gates completed, 2026-05-15:

```text
download:
  file:
    /mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking/model.safetensors
  bytes:
    5336011242
  sha256:
    c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080
  script:
    scripts/364_download_ouro_weight_parallel.sh
  note:
    The downloader is now resume-safe. It preserves completed prefix/part files
    and fetches only missing byte ranges.

partial actual Ouro:
  smoke:
    local_eval/qwen_backbone_qtrm_ouro_weight_partial_l18_smoke_20260515/report.json
    accepted=true
    gate0_delta=0.0
    core_on_delta=0.5
  train:
    local_eval/qwen_backbone_qtrm_ouro_weight_partial_l18_train_gate_s80_20260515/report.json
    accepted=true
    base_accuracy=0.14583333333333334
    core_accuracy=0.20833333333333334
    gain=0.0625
    language_top1=1.0

full actual Ouro:
  smoke:
    local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_smoke_20260515/report.json
    accepted=true
    gate0_delta=0.0
    core_on_delta=0.5
  train:
    local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_train_gate_s80_20260515/report.json
    accepted=true
    base_accuracy=0.14583333333333334
    core_accuracy=0.21875
    gain=0.07291666666666667
    language_top1=1.0

decision:
  Keep ouro_weight_wrapped as a real candidate. It matches the best short
  reasoning gain from ouro_shared_qwen_layer and gives better language top1
  agreement on the tiny non-regression probe. The next gate should be a
  multi-layer/layer-sweep or a longer held-out synthetic+bilingual gate, not
  another naming-only architecture change.
```

Actual Ouro-weight layer sweep, 2026-05-15:

```text
goal:
  Check whether the Ouro recurrent transition should use an early, middle, late,
  or multi-layer frozen Ouro block.

results:
  layer 12:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l12_train_gate_s80_20260515/report.json
    accepted:
      false
    gain:
      0.041666666666666664
    language_top1:
      1.0

  layer 24:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l24_train_gate_s80_20260515/report.json
    accepted:
      true
    gain:
      0.07291666666666667
    language_top1:
      1.0

  layer 36:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l36_train_gate_s80_20260515/report.json
    accepted:
      false
    gain:
      0.041666666666666664
    language_top1:
      0.75

  layers 18,24:
    report:
      local_eval/qwen_backbone_qtrm_ouro_weight_full_l18_24_train_gate_s80_20260515/report.json
    accepted:
      false
    gain:
      0.03125
    language_top1:
      0.75

decision:
  Keep single Ouro layer 24 as the current canonical actual-Ouro transition.
  Multi-layer wrapping is not automatically better; in the short gate it reduced
  both reasoning gain and language preservation.
```

QTRM transition-candidate naming clarification, 2026-05-15:

```text
clarification:
  Qwen wrapping and Ouro wrapping are both QTRM-core designs. The distinction is
  the frozen transition prior inside the recurrent z_L/z_H update, not whether
  QTRM is present.

common path:
  Qwen3.5 backbone
  -> final hidden states
  -> QTRM recurrent core
  -> gated residual
  -> same Qwen LM head

candidate A:
  qwen_layer_wrapped:
    QTRM recurrent core + frozen Qwen layer 3 transition

candidate B:
  ouro_weight_wrapped:
    QTRM recurrent core + frozen Ouro layer 24 transition

policy:
  Use the wording "QTRM core with Qwen transition prior" and "QTRM core with
  Ouro transition prior" to avoid implying that one path is non-QTRM.
```

Eval512 QTRM transition comparison, 2026-05-15:

```text
gate:
  train_cases=512
  eval_cases=512
  steps=200
  seed=20260515

QTRM core with Qwen transition prior:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/report.json
  accepted:
    true
  base_accuracy:
    0.130859375
  core_accuracy:
    0.1875
  gain:
    0.056640625
  language_top1:
    1.0

QTRM core with Ouro layer 24 transition prior:
  report:
    local_eval/qwen_backbone_qtrm_ouro_transition_l24_eval512_s200_20260515/report.json
  accepted:
    true
  base_accuracy:
    0.130859375
  core_accuracy:
    0.185546875
  gain:
    0.0546875
  language_top1:
    1.0

decision:
  Default canonical transition prior is Qwen layer wrapping for now. It is
  simpler and slightly stronger on the larger gate. Ouro layer 24 remains a
  strong accepted alternate, not a rejected path.
```

Bilingual general-language gate for canonical Qwen-transition QTRM, 2026-05-15:

```text
checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/last_core.pt

script:
  scripts/367_eval_qwen_backbone_language_gate.py

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_bilingual_generation_gate_20260515/report.json

accepted:
  true

top-k:
  prompts:
    12 English/Korean general prompts
  top1_agreement:
    1.0
  base_top1_in_core_top5:
    1.0
  mean_abs_delta:
    0.35411715507507324

generation:
  generated_prompts:
    12
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.9791666666666666

decision:
  The canonical Qwen-transition QTRM checkpoint passes the bilingual/general
  non-regression gate. It preserves base generation behavior closely and shows
  no repetition collapse in the sampled English/Korean prompts.
```

Long-generation language non-regression for canonical Qwen-transition QTRM,
2026-05-15:

```text
checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_20260515/last_core.pt

script:
  scripts/367_eval_qwen_backbone_language_gate.py

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_eval512_s200_longgen64_gate_20260515/report.json

generation:
  max_new_tokens:
    64
  prompts:
    12 English/Korean general prompts

accepted:
  true

top-k:
  top1_agreement:
    1.0
  base_top1_in_core_top5:
    1.0
  mean_abs_delta:
    0.35411715507507324

generation_metrics:
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.7513020833333334

decision:
  The same canonical checkpoint also passes the longer 64-token generation
  non-regression gate. This is stronger evidence against repetition collapse,
  but it is still a preservation gate rather than an open-ended language
  improvement claim.
```

Hard-family gate-open QTRM result, 2026-05-15:

```text
candidate:
  QTRM core with Qwen layer 3 transition prior

change:
  core_adapter_dim=128
  core_gate_init=-2.0
  residual_scale=0.5

checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/last_core.pt

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/report.json

accepted:
  true

hard_v1:
  base_accuracy:
    0.056640625
  core_accuracy:
    0.125
  gain:
    0.068359375
  learned_core_gate:
    0.11465207487344742

family_floor:
  chain5_gain:
    0.1286549707602339
  checksum4_gain:
    0.05847953216374269
  select_pair_gain:
    0.017647058823529405
  min_family_gain:
    0.017647058823529405
  min_family_core_accuracy:
    0.1111111111111111

negative_controls:
  hard_repair_v1 oversampling:
    rejected
  weighted family loss:
    rejected
  qwen layers 3,7 + adapter128 with small gate:
    rejected

interpretation:
  Hard_v1 required opening the residual core path. Data weighting and simply
  adding a second frozen Qwen transition layer did not solve select_pair. The
  accepted result is synthetic reasoning evidence only, not a claim that the
  model broadly beats Qwen3.5-2B.
```

Gate-open long-generation non-regression, 2026-05-15:

```text
checkpoint:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_familyfloor_20260515/last_core.pt

report:
  local_eval/qwen_backbone_qtrm_qwen_transition_hardv1_gateopen_ad128_s300_longgen64_20260515/report.json

accepted:
  true

top-k:
  top1_agreement:
    0.9166666865348816
  base_top1_in_core_top5:
    1.0
  mean_abs_delta:
    0.22324642539024353

generation:
  max_core_repeated_token_run:
    1
  mean_core_unique_ratio:
    0.7747395833333334

decision:
  The stronger gate-open checkpoint keeps 64-token English/Korean generation
  inside non-regression thresholds. This validates it as the current strongest
  Qwen-backbone QTRM synthetic-reasoning checkpoint.
```

Qwen-wrapper nested direction, 2026-05-15:

```text
decision:
  On the Qwen-backbone bridge path, use Qwen-wrapper nested recurrence as the
  next canonical TRM-style direction. Do not switch this path to
  Mamba/GatedDelta hybrid nested yet.

why:
  The accepted bridge result depends on Qwen3.5 hidden states, frozen Qwen
  transition prior, and gated QTRM residual logits. A Mamba/GatedDelta hybrid
  would be a different mixer/backbone experiment and would confound whether the
  Qwen-derived recurrent core itself scales.

nested smoke:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_smoke_s30_20260515/report.json
  core_impl:
    qwen_layer_wrapped
  h_cycles:
    3
  l_cycles:
    6
  outer_steps:
    1
  accepted:
    true
  base_accuracy:
    0.08333333333333333
  core_accuracy:
    0.125
  gain:
    0.041666666666666664
  learned_core_gate:
    0.119376040995121
  language_top1:
    1.0

limitation:
  This is a tiny relaxed smoke, not a promotion. The next real comparison is a
  matched hard_v1 nested run against the current non-nested gate-open baseline.
```

Qwen-wrapper nested matched-transition triage, 2026-05-15:

```text
script:
  scripts/369_run_qwen_wrapper_nested_compare.sh

comparison:
  non-nested h=1,l=1:
    steps:
      210
    approx_core_transitions:
      420
    report:
      local_eval/qwen_backbone_qtrm_qwen_transition_gateopen_nonnested_compare_seed20260515_s210_t420_20260515/report.json
    accepted:
      false
    gain:
      -0.005208333333333333
    min_family_gain:
      -0.015625

  nested h=3,l=6,residual_scale=0.5:
    steps:
      20
    approx_core_transitions:
      420
    report:
      local_eval/qwen_backbone_qtrm_qwen_transition_gateopen_nested_h3_l6_compare_seed20260515_s20_t420_20260515/report.json
    accepted:
      false
    gain:
      -0.041666666666666664
    min_family_gain:
      -0.125
    failure:
      select_pair collapse

  nested h=3,l=6,residual_scale=0.1:
    report:
      local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_r01_compare_seed20260515_s20_t420_20260515/report.json
    accepted:
      false
    gain:
      0.0
    min_family_gain:
      0.0

decision:
  Qwen-wrapper nested is still the correct nested direction, but fixed H3/L6 is
  not automatically better. The next attempt should not switch to
  Mamba/GatedDelta hybrid; it should stabilize Qwen-wrapper nested with gradual
  depth curriculum, residual/gate schedule, and periodic family-floor
  checkpoint selection.
```

Convergence-based early exit for Qwen-wrapper nested, 2026-05-15:

```text
motivation:
  Fixed H3/L6 is not enough. Add adaptive early exit over the same Qwen-wrapper
  nested path, inspired by convergence/fixed-point recurrence control:
    https://arxiv.org/abs/2605.12466

implemented:
  src/qtrm_mm/config.py:
    core_convergence_halt_enabled
    core_convergence_halt_threshold
    core_convergence_halt_min_outer

  src/qtrm_mm/qwen_backbone_qtrm.py:
    relative z_H delta after each outer block
    early break when every batch item converges
    output qtrm_core_outer_iterations / qtrm_core_converged /
      qtrm_core_convergence_delta

  scripts/362_train_qwen_backbone_qtrm_core_gate.py:
    CLI flags for convergence halt
    eval report mean_core_outer_iterations and core_converged_fraction

smoke:
  report:
    local_eval/qwen_backbone_qtrm_qwen_transition_nested_h3_l6_convergence_halt_telemetry_smoke_s5_20260515/report.json
  h/l/outer:
    3 / 6 / 3
  threshold:
    0.05
  mean_core_outer_iterations:
    3.0
  core_converged_fraction:
    0.0

decision:
  The early-exit mechanism is now present and measurable, but this threshold
  did not halt early. Next step is threshold/curriculum search. Promotion
  requires lower mean outer iterations plus preserved hard_v1 gain and language
  non-regression.
```

Convergence threshold sweep, 2026-05-15:

```text
script:
  scripts/370_sweep_nested_convergence_halt_threshold.py

report:
  local_eval/qwen_backbone_qtrm_nested_h3_l6_convergence_threshold_sweep_20260515/report.json

setting:
  Qwen-wrapper nested H3/L6
  outer_steps=3
  residual_scale=0.1
  eval_cases=96 hard_v1

results:
  threshold 0.02:
    mean_outer_iterations=3.0
    converged_fraction=0.0
    gain=0.0
  threshold 0.05:
    mean_outer_iterations=3.0
    converged_fraction=0.0
    gain=0.0
  threshold 0.1:
    mean_outer_iterations=3.0
    converged_fraction=0.0
    gain=0.0
  threshold 0.2:
    mean_outer_iterations=3.0
    converged_fraction=0.3333333333333333
    gain=0.0
  threshold 0.5:
    mean_outer_iterations=2.0
    converged_fraction=1.0
    gain=0.0
  threshold 1.0:
    mean_outer_iterations=1.0
    converged_fraction=1.0
    gain=0.0

decision:
  Early-exit compute control works mechanically. It is not yet a reasoning win.
  Next step is nested curriculum training, then applying this threshold sweep
  to the trained nested checkpoint.
```
# 2026-05-15 - QTRM-Native 2B/3B vs Qwen3.6-27B target contract

Added the canonical milestone contract for the project goal:

```text
QTRM-Native-2B/3B beats Qwen3.6-27B on reasoning/memory-relevant benchmarks
with about 10x fewer parameters.
```

Files:

- `docs/wiki/sources/qwen36-27b-benchmarks.md`
- `docs/wiki/decisions/qtrm-native-27b-benchmark-milestones.md`
- `scripts/372_qtrm_native_27b_milestone_status.py`
- `tests/test_qtrm_native_27b_milestone_status.py`

Current status:

```text
M0 target contract: accepted
M1 bridge causal signal: rejected by stability gate
  gain: 0.064453125
  min_family_gain: 0.011764705882352955
  min_family_core_accuracy: 0.0935672514619883
  required family accuracy: 0.10

3-seed stability:
  num_accepted: 1 / 3
  min_gain: 0.037109375
  mean_gain: 0.0546875
  min_family_gain: -0.01764705882352942
  min_family_core_accuracy: 0.07602339181286549
```

Interpretation:

The Qwen-backbone bridge result is a useful causal signal but remains
diagnostic. It does not count as QTRM-Native progress until reproduced in a
donorless native token-to-logit path.

Fast-path update:

```text
M1 is now closed as a rejected diagnostic bridge.
Do not spend more time tuning bridge checkpoints unless a later native result
requires a targeted bridge probe.
Next action: M2 Native Tiny LM on the local 4090.
4090 is sufficient for M0-M3 and M6 scoped gates.
DGX is expected for serious M4+ language bootstrap and M7/M8 public benchmark
parity/win attempts.
```

## [2026-05-15] milestone | M2/M3/M4 native fast-path update

Status gate update:

```text
script:
  scripts/372_qtrm_native_27b_milestone_status.py

new inputs:
  --native-report
  --native-core-report
```

M2/M4 evidence:

```text
local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_s4000_20260515/report.json

decision: accepted_qtrm_native_language_bootstrap
vocab_size: 16000
think_eval_loss: 0.06253129527702948
think0_loss: 4.404038346539084
thinking_block_off_loss: 4.404038346539084
sample max_run_fraction: 0.013333333333333334
```

M3 evidence:

```text
local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_core_ablation_20260515/report.json

decision: accepted_qtrm_native_core_causality
think_eval_loss: 0.06253067243833912
think0_loss: 4.403806257955005
thinking_block_off_loss: 4.403806257955005
state_reset_loss: 1.0610178813987357
full_vs_best_shallow_depth: 0.39065462690292
```

Interpretation:

```text
M2 native tiny LM: accepted.
M3 native core causality: accepted on the small native language checkpoint.
M4 small native language bootstrap: accepted.
Next action: M5 public-target Qwen3.6 evaluation harness.
Correction: DGX is not required to use published Qwen3.6-27B benchmark
numbers. DGX/server is optional and only needed for a direct Qwen rerun on
custom/scoped suites.
```

## [2026-05-15] milestone | M5 public-target Qwen3.6 eval harness accepted

Artifacts:

```text
manifest:
  local_eval/qwen36_public_target_manifest/report.json

status report:
  local_eval/qtrm_native_27b_milestone_status/report.json
```

Decision:

```text
M5_QWEN36_EVAL_HARNESS: accepted
comparison_mode: public_qwen36_target_scores
direct_qwen36_rerun_required: false
benchmark_count: 13
artifact_count: 2
next_action: M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING
```

Interpretation:

```text
DGX is not required just to compare against published Qwen3.6-27B benchmark
targets. The current M5 result is not a model-win claim; it is the accepted
comparison contract that keeps QTRM-Native outputs, public target scores, and
scorer mappings in one manifest.
```

## [2026-05-15] milestone | M6 scoped raw-reasoning gate attempted

Artifacts:

```text
manifest:
  local_eval/m6_scoped_raw_reasoning_manifest/report.json

suite:
  local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl

suite metadata:
  local_eval/m6_scoped_raw_reasoning_suite/metadata.json

Qwen3.6 baseline runner:
  scripts/378_eval_qwen36_scoped_raw_reasoning_baseline.py

Qwen3.6 baseline wrapper:
  scripts/379_run_m6_qwen36_baseline.sh

M6 status refresher:
  scripts/380_refresh_m6_status.sh

status report:
  local_eval/qtrm_native_27b_milestone_status/report.json

best QTRM-native report:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json
```

Metrics:

```text
suite_id:
  qtrm_native_text_reasoning_modchain_revchain_checksum_program4_mod32

prompt_protocol:
  operation_definitions_v1

QTRM full_generation_exact:
  0.6067708333333334

think0:
  0.020833333333333332

core_gain:
  0.5859375

ablation_drop:
  0.5716145833333334

min_family_generation_exact:
  0.4140625
```

Decision:

```text
M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING: rejected
reject reason:
  matched Qwen3.6-27B baseline score is missing for the same deterministic
  scoped suite.
```

Interpretation:

```text
The QTRM-native candidate has a real causal recursive-core signal on the scoped
raw-reasoning suite, but this is not yet a Qwen3.6-27B win. The next artifact
must be a Qwen3.6-27B baseline report for the exact same suite_id.

DGX status:
  ssh dgx failed with "No route to host"
  ssh sk@edgexpert-5b20 failed DNS resolution
```

## [2026-05-16] ops | M6 Qwen3.6 baseline runner prepared, DGX unreachable

Prepared the exact M6 baseline path:

```text
suite:
  local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl

runner:
  scripts/379_run_m6_qwen36_baseline.sh

status refresher:
  scripts/380_refresh_m6_status.sh
```

DGX network check:

```text
ssh dgx:
  ssh: connect to host 192.168.219.113 port 22: No route to host

ping 192.168.219.113:
  Destination Host Unreachable

ssh sk@edgexpert-5b20:
  Could not resolve hostname edgexpert-5b20
```

Current blocker:

```text
M6 cannot be accepted until Qwen3.6-27B is run on the same suite with
prompt_protocol=operation_definitions_v1.
```

## [2026-05-16] eval | M6 accepted against local Qwen3.6-27B-MTP GGUF proxy

Corrected the local baseline from the accidentally downloaded Qwen3.5 MTP GGUF
to `unsloth/Qwen3.6-27B-MTP-GGUF`, file
`Qwen3.6-27B-UD-Q4_K_XL.gguf`.

Artifacts:

```text
model:
  /mnt/nvme0n1p2/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf

runner:
  scripts/382_run_m6_qwen36_mtp_proxy_baseline.sh

baseline report:
  local_eval/m6_qwen36_mtp_proxy_baseline/report.json

M6 manifest:
  local_eval/m6_scoped_raw_reasoning_manifest/report.json

status:
  local_eval/qtrm_native_27b_milestone_status/report.json
```

Scores:

```text
QTRM-Native:
  full_generation_exact: 0.6067708333333334
  core_gain: 0.5859375
  ablation_drop: 0.5716145833333334
  min_family_generation_exact: 0.4140625

Qwen3.6-27B-MTP-GGUF proxy:
  generation_exact: 0.15364583333333334
  checksum: 0.38671875
  modchain: 0.046875
  revchain: 0.02734375
```

Decision:

```text
M6_NATIVE_SMALL_BEATS_27B_ON_SCOPED_RAW_REASONING: accepted
decision: accepted_m6_scoped_raw_reasoning_win
```

Limitation:

```text
This is a scoped custom-suite win over a local quantized Qwen3.6 MTP GGUF proxy.
It is not public benchmark parity, and it is not a full-precision Qwen3.6 rerun.
The next target is M7 public benchmark parity.
```

## [2026-05-16] eval | M7 public MMLU-Pro balanced smoke rejected

Added the first public benchmark path for M7:

```text
suite materializer:
  scripts/383_materialize_m7_public_reasoning_suite.py

evaluator:
  scripts/384_eval_qtrm_native_public_mcq.py

status refresher:
  scripts/385_refresh_m7_status.sh

unit tests:
  tests/test_m7_public_reasoning_suite.py
  tests/test_qtrm_native_public_mcq_eval.py
```

Dataset:

```text
TIGER-Lab/MMLU-Pro
source:
  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

HF Dataset Viewer validation split:
  70 rows

HF Dataset Viewer test split:
  12032 rows
```

Run:

```text
suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl

checkpoint:
  local_eval/qtrm_native_pretrained_init_qwen35_compact_external4500_s3600_20260515/last.pt

checkpoint type:
  QTRM-Native with Qwen3.5 compact tokenizer/pretrained initialization
  runtime_donor: false

report:
  local_eval/m7_qtrm_native_qwen35pre_mmlu_pro_balanced256_eval/report.json
```

Result:

```text
QTRM-Native Qwen3.5-preinit:
  hits: 37 / 256
  accuracy: 0.14453125

Previous byte-BPE tiny native:
  hits: 16 / 256
  accuracy: 0.0625

Qwen3.6-27B MMLU-Pro target:
  0.862

parity floor:
  0.842

decision:
  rejected_m7_public_benchmark_parity
```

Important limitation:

```text
This is a category-balanced 256-case subset generated through the local
`datasets` backend after the Dataset Viewer balanced scan hit HTTP 429. It is
much better than the first category-ordered business-only smoke, but still not
the full 12032-case MMLU-Pro test split.
```

Interpretation:

```text
The Qwen3.5-preinit native path is a real native model: it copies Qwen3.5
tokenizer/embedding information into QTRM initialization but does not call a
Qwen donor at runtime. It more than doubles the byte-BPE tiny score, but still
does not have public benchmark language/knowledge competence. The next step is
a larger native language/knowledge bootstrap and then balanced/full MMLU-Pro
scoring.
```

## 2026-05-16 - Qwen3.5 Integrated Native Correction

```text
correction:
  The intended C path is Qwen3.5-integrated QTRM-native, not a tiny compact
  bridge. Using Qwen3.5 inside the same standalone model graph is allowed; using
  Qwen as an external donor sidecar is not canonical.

canonical path:
  prompt/chat-template tokens
  -> Qwen3.5 tokenizer/full vocab
  -> Qwen3.5 embedding
  -> Qwen3.5 original backbone/layers
  -> mandatory QTRM recursive core in the same causal hidden path
  -> Qwen3.5 LM head
  -> autoregressive text

code update:
  src/qtrm_mm/qwen_backbone_qtrm.py
    added mandatory_core mode.
    normal forward uses core gate 1.0 when mandatory_core=true.
    force_core_off and core_gate_override=0 remain ablation-only.

  scripts/361_qwen_backbone_qtrm_smoke.py
    default core_impl is now qwen_layer_wrapped.

  scripts/362_train_qwen_backbone_qtrm_core_gate.py
    added --mandatory-core and --train-qwen.
    reports qtrm_native_integrated=true, standalone_graph=true, runtime_donor=false.

  scripts/386_run_qwen35_integrated_mandatory_core_gate.sh
    one-command Stage-1 runner for Qwen3.5 integrated mandatory-core training.

training order:
  1. freeze Qwen; train mandatory QTRM core.
  2. require core_on > core_off and language non-regression.
  3. partial Qwen unfreeze / healing tune only after the core gain is stable.
```

Accepted Stage-1 gate:

```text
runner:
  scripts/386_run_qwen35_integrated_mandatory_core_gate.sh

report:
  local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516/report.json

checkpoint:
  local_eval/qwen35_integrated_mandatory_core_gate_s300_20260516/last_core.pt

model path:
  Qwen3.5 tokenizer/full vocab/backbone/LM head
  -> mandatory QTRM core
  -> same LM head

runtime_donor:
  false

qwen_trainable:
  false

core_impl:
  qwen_layer_wrapped

normal_core_gate:
  1.0

base/core_off accuracy:
  0.029296875

core_on accuracy:
  0.1171875

gain:
  0.087890625

min_family_gain:
  0.058823529411764705

min_family_core_accuracy:
  0.09941520467836257

language top1 agreement:
  1.0

decision:
  accepted
```

Milestone plan added:

```text
file:
  docs/wiki/architecture/qtrm-native-first-roadmap.md

section:
  Qwen Integrated Native Milestones

sequence:
  M0 Integrated Path Lock
  M1 Freeze Qwen, Train Mandatory QTRM Core
  M2 Partial Qwen Unfreeze
  M3 Healing Tune
  M4 Public Benchmark Recheck
  M5 Scale/Release Candidate

policy:
  Qwen3.5 original backbone is preserved as much as possible.
  QTRM core is mandatory in the same causal path.
  Partial unfreeze and healing tune happen only after frozen-Qwen mandatory-core
  gain is accepted.
```

## 2026-05-16 - M2 partial Qwen unfreeze gate

```text
implementation:
  added selective Qwen unfreeze support:
    --unfreeze-qwen-layer-indices
    --qwen-lr
    --qwen-weight-decay

  added finite-logit acceptance:
    reasoning eval logits must be finite
    language non-regression logits must be finite

runner:
  scripts/387_run_qwen35_integrated_partial_unfreeze_gate.sh

safety correction:
  fp16 direct partial unfreeze produced non-finite language delta in smoke.
  M2 default now uses bfloat16 and Qwen LR 2e-6.

candidate A:
  unfreeze layer:
    3
  report:
    local_eval/qwen35_integrated_partial_unfreeze_l3_s200_20260516/report.json
  decision:
    rejected
  gain:
    0.046875
  min_family_gain:
    -0.0058823529411764774
  language_top1:
    1.0
  finite_logits:
    true

candidate B:
  unfreeze layer:
    23
  report:
    local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/report.json
  checkpoint:
    local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/last_core.pt
  decision:
    accepted_m2_partial_unfreeze_family_floor
  base/core_off accuracy:
    0.060546875
  core_on accuracy:
    0.107421875
  gain:
    0.046875
  min_family_gain:
    0.0
  min_family_core_accuracy:
    0.08187134502923976
  language_top1:
    1.0
  finite_logits:
    true

interpretation:
  Partial unfreeze is now technically wired and has one safe accepted candidate.
  Layer 23 is safer than layer 3. This is a family-floor/healing acceptance, not
  a broad aggregate improvement over M1 yet.
```

## 2026-05-16 - M3 Qwen-integrated healing tune accepted

```text
runner:
  scripts/388_run_qwen35_integrated_healing_tune.sh

language gate:
  scripts/389_run_qwen35_integrated_healing_language_gate.sh

init checkpoint:
  local_eval/qwen35_integrated_partial_unfreeze_l23_s200_20260516/last_core.pt

output checkpoint:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt

canonical status:
  qtrm_native_integrated=true
  standalone_graph=true
  runtime_donor=false
  mandatory_core=true
  normal_core_gate=1.0

training:
  partial Qwen layer 23 unfreeze
  Qwen LR: 1.0e-6
  QTRM LR: 5.0e-5
  language KL to core_off/base path: 0.10
  dtype: bfloat16
  steps: 100

reasoning report:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/report.json

reasoning result:
  accepted: true
  base/core_off accuracy: 0.05078125
  core_on accuracy: 0.142578125
  gain: 0.091796875
  min_family_gain: 0.04678362573099415
  min_family_core_accuracy: 0.1286549707602339
  language_top1_agreement: 0.875
  finite_logits: true

language generation report:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_language_gate_20260516/report.json

language generation result:
  accepted: true
  accepted_top1: true
  accepted_top5: true
  accepted_repetition: true
  accepted_unique_ratio: true
  accepted_finite_logits: true

interpretation:
  This is the first accepted M3 healing gate on the Qwen-integrated
  QTRM-native path. It preserves the original Qwen tokenizer/full vocab,
  backbone, and LM head inside one standalone graph while making the QTRM core
  mandatory. It improves the local hard_v1 reasoning gate over core_off and
  does not collapse English/Korean generation. Do not claim 27B parity yet;
  M4 public benchmark and seed-stability gates are next.
```

## 2026-05-16 - M4 Qwen-integrated public MCQ smoke accepted

```text
implementation:
  scripts/390_eval_qwen35_integrated_public_mcq.py
  scripts/390_run_qwen35_integrated_m4_public_mcq.sh
  tests/test_qwen35_integrated_public_mcq_eval.py

suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl
  MMLU-Pro validation, 64 category-balanced cases

checkpoint:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt

report:
  local_eval/qwen35_integrated_m4_mmlu_pro64_20260516/report.json

canonical status:
  qtrm_native_integrated=true
  standalone_graph=true
  runtime_donor=false
  mandatory_core=true
  normal_core_gate=1.0

scorer:
  next-token option-letter log likelihood

result:
  decision: accepted_m4_public_mcq_core_gain
  accepted_core_gain: true
  accepted_parity: false
  accepted_finite_logits: true
  base/core_off: 22/64 = 0.34375
  core_on: 24/64 = 0.375
  core_gain_over_base: 0.03125
  min_core_gain: 0.01

balanced 256 recheck:
  report:
    local_eval/qwen35_integrated_m4_mmlu_pro_balanced256_20260516/report.json
  decision:
    rejected_m4_public_mcq_core_gain
  base/core_off: 92/256 = 0.359375
  core_on: 92/256 = 0.359375
  core_gain_over_base: 0.0

interpretation:
  The M3 checkpoint gives a small positive public-subset gain over the same
  Qwen3.5 graph with core_off on 64 validation cases. The 256 balanced subset
  is neutral, so this is useful as a smoke signal but not stable public
  benchmark progress yet. It tests the actual LM-logit path rather than a
  sidecar or synthetic-only renderer. It remains a smoke gate, not a 27B parity
  claim. Next required checks: seed-stability, category-error analysis, and then
  targeted healing/data if the gain does not hold.
```

## 2026-05-16 - M4 256-case public MCQ healing accepted

```text
implementation:
  scripts/391_train_qwen35_integrated_public_mcq_healing.py
  scripts/391_run_qwen35_integrated_public_mcq_healing.sh
  tests/test_qwen35_integrated_public_mcq_healing.py

train suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl

heldout eval suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl

init checkpoint:
  local_eval/qwen35_integrated_healing_l23_langkl_s100_20260516/last_core.pt

output checkpoint:
  local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt

important naming correction:
  The output directory contains "coreonly", but this accepted run used partial
  Qwen layer-23 unfreeze because the runner defaulted an empty
  UNFREEZE_QWEN_LAYER_INDICES value to 23 before the fix. Treat this as l23
  public-MCQ healing. A true core-only control was run separately and rejected.

training report:
  local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/report.json

independent verification report:
  local_eval/qwen35_integrated_public_mcq_healing_l23_verified_m4_256_20260516/report.json

verified result:
  decision: accepted_m4_public_mcq_core_gain
  accepted_core_gain: true
  accepted_parity: false
  base/core_off: 92/256 = 0.359375
  core_on: 96/256 = 0.375
  core_gain_over_base: 0.015625

true core-only control:
  report:
    local_eval/qwen35_integrated_public_mcq_healing_true_coreonly_val64_to_test256_s80_20260516/report.json
  decision:
    rejected_public_mcq_healing
  base/core_off: 92/256 = 0.359375
  core_on: 94/256 = 0.3671875
  core_gain_over_base: 0.0078125

interpretation:
  M4 now has a verified positive core gain on a 256-case balanced MMLU-Pro
  public subset through the canonical Qwen-integrated QTRM-native LM-logit path.
  This is real progress beyond the 64-case smoke, but it is still a small
  subset gain and not Qwen3.6-27B parity. Remaining work: seed-stability,
  category-error analysis, broader/full MMLU-Pro evaluation, and larger
  language/knowledge healing without language regression.
```

## 2026-05-16 - M4 512-case public MCQ recheck accepted

```text
suite materialization:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_512.jsonl
  local_eval/m7_public_reasoning_suite/report_balanced_512.json

checkpoint:
  local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt

important naming/optimizer detail:
  Directory name says coreonly, but the run had Qwen layer 23 marked trainable.
  Qwen LR was 0.0, so Qwen weights were not updated. Treat this as
  layer23-open/core-updated public-MCQ healing, not as Qwen weight healing.

accepted 512 report:
  local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_resid0p06_20260516/report.json

accepted setting:
  residual_scale: 0.06

result:
  decision: accepted_m4_public_mcq_core_gain
  accepted_core_gain: true
  accepted_parity: false
  base/core_off: 191/512 = 0.373046875
  core_on: 197/512 = 0.384765625
  core_gain_over_base: 0.01171875

language gate at residual_scale 0.06:
  report:
    local_eval/qwen35_integrated_public_mcq_healing_l23open_resid0p06_language_gate_20260516/report.json
  accepted: true
  top1_agreement: 0.8333333730697632
  base_top1_in_core_top5: 1.0
  max_repeated_token_run: 1
  mean_unique_ratio: 0.7864583333333334

rejected controls:
  residual_scale 0.05:
    191/512 -> 196/512, gain 0.009765625
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_20260516/report.json
  residual_scale 0.04:
    191/512 -> 193/512, gain 0.00390625
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_512_resid0p04_20260516/report.json
  seed20260520 residual_scale 0.05:
    191/512 -> 194/512, gain 0.005859375
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260520_m4_512_20260516/report.json
  seed19/20 soup alpha 0.5:
    191/512 -> 194/512, gain 0.005859375
    local_eval/qwen35_integrated_public_mcq_healing_soup_seed19_20_a050_m4_512_20260516/report.json

category deltas at accepted setting:
  biology: +1
  business: +2
  computer science: +2
  history: +1
  other: +2
  physics: +1
  health: -2
  law: -1
  chemistry/economics/math/philosophy/psychology: 0

interpretation:
  M4 has moved from a 256-case public-subset gain to a 512-case balanced
  public-subset gain while preserving the language gate. This remains a small
  subset improvement and is not a Qwen3.6-27B parity claim. The next hard
  bottleneck is stable 1K/full-suite improvement, especially reducing
  health/law regressions without erasing the positive categories.
```

## 2026-05-16 - M4 1024-case public MCQ bottleneck and aux repair attempts

```text
suite materialization:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_1024.jsonl
  local_eval/m7_public_reasoning_suite/report_balanced_1024.json

1024 baseline recheck from the 512-accepted checkpoint:
  report:
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_1024_resid0p06_20260516/report.json
  decision:
    rejected_m4_public_mcq_core_gain
  base/core_off:
    378/1024 = 0.369140625
  core_on:
    381/1024 = 0.3720703125
  gain:
    0.0029296875

residual_scale 0.08 control:
  report:
    local_eval/qwen35_integrated_public_mcq_healing_l23open_seed20260519_m4_1024_resid0p08_20260516/report.json
  base/core_off:
    378/1024 = 0.369140625
  core_on:
    378/1024 = 0.369140625
  gain:
    0.0

implementation added:
  scripts/392_materialize_aux_public_mcq.py
  tests/test_aux_public_mcq_materializer.py

loss update:
  scripts/391_train_qwen35_integrated_public_mcq_healing.py
  scripts/391_run_qwen35_integrated_public_mcq_healing.sh

  Added base-wrong option-margin training:
    CE(chosen option)
    + margin_weight * relu(margin - score_core(gold) + score_core(base_pred))
    + optional language KL

  Added --skip-train-eval to avoid expensive full train-set scoring when the
  train JSONL is only an auxiliary repair source.

auxiliary data policy:
  Use cais/mmlu dev/validation/auxiliary_train for training/checkpoint
  selection. Do not use MMLU-Pro test labels for training or checkpoint
  selection.

targeted aux train:
  local_eval/m7_public_reasoning_suite/mmlu_aux_targeted_validation_train_20260516.jsonl
  cases: 401
  categories:
    health: 96
    chemistry: 30
    economics: 81
    law: 194

targeted aux dev selection:
  local_eval/m7_public_reasoning_suite/mmlu_aux_targeted_dev_select_20260516.jsonl
  cases: 60
  categories:
    health: 20
    chemistry: 10
    economics: 15
    law: 15

CE-only aux repair:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_from_m3_devselect_s120_20260516/report.json
  decision:
    rejected_public_mcq_healing
  aux dev:
    33/60 -> 33/60, gain 0.0
  language:
    accepted

targeted base-wrong margin aux repair:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_margin_from_m3_devselect_s160_20260516/report.json
  decision:
    accepted_public_mcq_healing
  aux dev:
    33/60 -> 34/60, gain 0.016666666666666607
  language:
    accepted
  independent MMLU-Pro 1024 report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_margin_from_m3_m4_1024_20260516/report.json
  independent MMLU-Pro 1024:
    378/1024 -> 383/1024, gain 0.0048828125
    rejected; threshold is >= 0.01

targeted margin residual_scale 0.06 control:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_margin_from_m3_m4_1024_resid0p06_20260516/report.json
  independent MMLU-Pro 1024:
    378/1024 -> 379/1024, gain 0.0009765625
    rejected

all-auxiliary blend:
  broad auxiliary file:
    local_eval/m7_public_reasoning_suite/mmlu_aux_all_auxtrain_1024_20260516.jsonl
  blended train file:
    local_eval/m7_public_reasoning_suite/mmlu_aux_blend_targeted401_all1024_train_20260516.jsonl
  cases:
    1425

  training report:
    local_eval/qwen35_integrated_public_mcq_aux_blend_margin_from_m3_devselect_s220_20260516/report.json
  aux dev:
    33/60 -> 34/60, gain 0.016666666666666607
  independent MMLU-Pro 1024 report:
    local_eval/qwen35_integrated_public_mcq_aux_blend_margin_from_m3_m4_1024_20260516/report.json
  independent MMLU-Pro 1024:
    378/1024 -> 379/1024, gain 0.0009765625
    rejected

all-validation/all-dev partial Qwen layer-23 margin repair:
  train file:
    local_eval/m7_public_reasoning_suite/mmlu_aux_all_validation_train_20260516.jsonl
  dev selection file:
    local_eval/m7_public_reasoning_suite/mmlu_aux_all_dev_select_20260516.jsonl
  report:
    local_eval/qwen35_integrated_public_mcq_aux_all_margin_l23lr1e7_from_m3_devselect_s120_20260516/report.json
  setting:
    Qwen layer 23 trainable with qwen_lr=1.0e-7
    QTRM lr=5.0e-5
    base-wrong option-margin weight=0.75
    language KL weight=0.25
  dev result:
    169/285 -> 170/285, gain 0.0035087719298245723
    rejected; threshold is >= 0.01
  language:
    accepted
    top1 agreement 0.9166666865348816
  interpretation:
    A tiny Qwen LR preserves language but does not solve the public-MCQ
    transfer bottleneck. Do not promote this to 1024 evaluation.

interpretation:
  The 512-case signal does not yet scale to 1024. Base-wrong margin loss is
  better than CE-only and improves the independent 1024 result from +3 hits to
  +5 hits, but it is still below the +11 hit acceptance threshold. Broad
  auxiliary_train blending hurts transfer and should not be promoted. The
  all-validation/all-dev partial-Qwen attempt also misses the dev gate. The
  next valid move is not more residual-scale sweeping; it is a stronger
  held-out public-MCQ repair protocol with per-category regression guards or a
  larger-scale language/knowledge healing run.
```

## 2026-05-16 - M4 public MCQ regression guards and base-wrong CE

Implementation:

```text
scripts/391_train_qwen35_integrated_public_mcq_healing.py
  added category_gain_summary
  added category-regression-penalized checkpoint selection
  added balanced category sampling
  added ce_focus=base_wrong

scripts/391_run_qwen35_integrated_public_mcq_healing.sh
  added BALANCED_CATEGORY_SAMPLING
  added CATEGORY_REGRESSION_PENALTY
  added MIN_EVAL_CATEGORY_GAIN / MIN_EVAL_CATEGORY_HIT_DELTA
  added CE_FOCUS

tests:
  tests/test_qwen35_integrated_public_mcq_healing.py
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/391_train_qwen35_integrated_public_mcq_healing.py \
  scripts/392_materialize_aux_public_mcq.py

bash -n scripts/391_run_qwen35_integrated_public_mcq_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_public_mcq_healing \
  tests.test_aux_public_mcq_materializer \
  tests.test_qwen35_integrated_public_mcq_eval

result:
  13 tests OK
```

New repair attempts:

```text
all-validation/all-dev balanced category + margin:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_all_balcat_margin_from_m3_devselect_s180_20260516/report.json
  result:
    169/285 -> 171/285, gain 0.007017543859649145
    category regressions: 0
    language: accepted
    decision: rejected

targeted balanced category + margin:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_balcat_margin_from_m3_devselect_s160_20260516/report.json
  result:
    33/60 -> 33/60, gain 0.0
    language: accepted
    decision: rejected

targeted base-wrong CE + margin, selected on cais/mmlu targeted dev:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_basewrongce_margin_from_m3_devselect_s160_20260516/report.json
  result:
    33/60 -> 34/60, gain 0.016666666666666607
    category regressions: 0
    language: accepted
    decision: accepted
  independent MMLU-Pro 1024:
    report:
      local_eval/qwen35_integrated_public_mcq_aux_targeted_basewrongce_margin_from_m3_m4_1024_20260516/report.json
    result:
      378/1024 -> 378/1024, gain 0.0
      decision: rejected

targeted base-wrong CE + margin, selected on MMLU-Pro validation 64:
  report:
    local_eval/qwen35_integrated_public_mcq_aux_targeted_basewrongce_margin_from_m3_mmluproval64_s120_20260516/report.json
  result:
    22/64 -> 24/64, gain 0.03125
    category regressions: 0
    language: accepted
    decision: accepted
  independent MMLU-Pro 1024:
    report:
      local_eval/qwen35_integrated_public_mcq_aux_targeted_basewrongce_margin_from_m3_mmluproval64_m4_1024_20260516/report.json
    result:
      378/1024 -> 378/1024, gain 0.0
      decision: rejected
```

Interpretation:

```text
The new guards are useful diagnostics but do not solve 1K transfer. Small
selection sets can show clean category non-regression while still failing the
independent 1024 public subset. The strongest 1024 result remains the earlier
targeted base-wrong margin run:
  378/1024 -> 383/1024

The next step should stop optimizing tiny selection sets. Public benchmark
progress now needs either:
  1. a larger non-test selection pool closer to MMLU-Pro distribution, or
  2. larger-scale language/knowledge healing that changes the base competence,
     followed by the same core-on/core-off 1K gate.
```

## 2026-05-16 - External MCQ pool and AD512 warm-start probe

Implementation:

```text
scripts/393_materialize_external_mcq_pool.py
  materializes non-test external MCQ datasets into the QTRM public-MCQ JSONL
  schema.

scripts/391_train_qwen35_integrated_public_mcq_healing.py
  added checkpoint_load_mode=skip_mismatch so larger core adapters can
  warm-start from older smaller-adapter checkpoints without losing compatible
  Qwen/core tensors.

scripts/390_run_qwen35_integrated_m4_public_mcq.sh
scripts/391_run_qwen35_integrated_public_mcq_healing.sh
  expose QWEN_CORE_LAYER_INDICES, CORE_ADAPTER_DIM, CORE_DELTA_ADAPTER_MODE,
  and RESIDUAL_SCALE.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/390_eval_qwen35_integrated_public_mcq.py \
  scripts/391_train_qwen35_integrated_public_mcq_healing.py \
  scripts/393_materialize_external_mcq_pool.py

bash -n scripts/390_run_qwen35_integrated_m4_public_mcq.sh
bash -n scripts/391_run_qwen35_integrated_public_mcq_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_public_mcq_eval \
  tests.test_qwen35_integrated_public_mcq_healing \
  tests.test_external_mcq_pool_materializer

result:
  14 tests OK
```

External pool:

```text
train pool:
  local_eval/m7_public_reasoning_suite/external_mcq_train_pool_2000_20260516.jsonl
  cases: 2000
  sources:
    allenai/ai2_arc ARC-Challenge train: 500
    allenai/ai2_arc ARC-Easy train: 500
    allenai/openbookqa main train: 500
    tau/commonsense_qa default train: 500
  policy:
    non-test external MCQ pool; do not use public target test labels for
    checkpoint selection.
```

AD512 warm-start probe:

```text
smoke:
  report:
    local_eval/qwen35_integrated_public_mcq_warmstart_ad512_smoke_s2_20260516/report.json
  result:
    core_adapter_dim 128 -> 512 loaded with checkpoint_load_mode=skip_mismatch
    skipped mismatched tensors:
      core_delta_adapter.1.weight
      core_delta_adapter.3.weight
    qtrm parameters:
      538625 -> 2111489

external2000 base-wrong CE + margin, selected on MMLU-Pro validation64:
  report:
    local_eval/qwen35_integrated_public_mcq_external2000_ad512_basewrongce_margin_from_m3_mmluproval64_s120_20260516/report.json
  result:
    22/64 -> 24/64, gain 0.03125
    best periodic step: 20
    language: accepted, top1 agreement 0.9166666865348816
    decision: accepted

independent MMLU-Pro 1024:
  report:
    local_eval/qwen35_integrated_public_mcq_external2000_ad512_basewrongce_margin_from_m3_mmluproval64_m4_1024_20260516/report.json
  result:
    380/1024 -> 377/1024, gain -0.0029296875
    decision: rejected
```

Interpretation:

```text
The loader/core-scale plumbing is now correct: larger adapters can be
warm-started safely from the M3 integrated checkpoint. But AD512 plus external
MCQ selection still does not transfer to the 1024 public target. This points
away from small public-MCQ repair and toward a larger knowledge/language
healing stage or a more substantial recurrent-core training curriculum.
```

## 2026-05-16 - Integrated language/knowledge healing scaffold

Implementation:

```text
scripts/394_train_qwen35_integrated_language_knowledge_healing.py
  trains the Qwen3.5-integrated QTRM-native graph on external text next-token
  CE, optional non-test external MCQ option CE, and core_off KL.

scripts/394_run_qwen35_integrated_language_knowledge_healing.sh
  default data:
    local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl
    local_eval/m7_public_reasoning_suite/external_mcq_train_pool_2000_20260516.jsonl

tests:
  tests/test_qwen35_integrated_language_knowledge_healing.py
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py

bash -n scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_language_knowledge_healing

result:
  5 tests OK
```

Smoke:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_smoke_s2_20260516/report.json

result:
  accepted_integrated_language_knowledge_healing
  core_ce_delta: 0.000007748603820800781
  language top1 agreement: 0.9166666865348816
```

Standard 120-step run:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_external9000_s120_20260516/report.json

data:
  text rows: 6000
  non-test external MCQ rows: 2000

result:
  accepted_integrated_language_knowledge_healing
  before core CE: 2.2167024994269013
  after core CE: 2.216939340811223
  core_ce_delta: 0.0002368413843214512
  language top1 agreement: 0.9166666865348816
  max repeated token run: 1
  external eval MCQ: 87/128 -> 87/128
```

Independent MMLU-Pro 1024 after language/knowledge healing:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_external9000_m4_1024_20260516/report.json

result:
  base: 380/1024
  core: 377/1024
  gain: -0.0029296875
  decision: rejected_m4_public_mcq_core_gain

category movement:
  gains:
    biology +1
    business +1
    law +1
    history +1
    other +1
  regressions:
    chemistry -2
    health -3
    philosophy -1
    physics -1
    economics -1
```

Interpretation:

```text
The integrated language/knowledge scaffold works and preserves generation, but
120 local steps are not enough to improve public MMLU-Pro 1024 transfer. The
same pattern remains: early slices can look positive, but the full 1024 subset
exposes category regressions. Next valid work is either a much larger
knowledge-healing run with stronger category-balanced validation, or a stronger
recurrent-core curriculum before public-MCQ retesting.
```

## 2026-05-16 - Validation-controlled language/knowledge healing

Implementation update:

```text
scripts/394_train_qwen35_integrated_language_knowledge_healing.py
  added:
    external MCQ validation jsonl
    periodic validation scoring
    best checkpoint restore
    per-category MCQ summaries
    category-regression selection penalty
    base_wrong MCQ CE focus
    base_wrong preference-margin option loss

scripts/394_run_qwen35_integrated_language_knowledge_healing.sh
  default validation:
    local_eval/m7_public_reasoning_suite/external_mcq_validation_pool_20260516.jsonl
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py

bash -n scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_language_knowledge_healing

result:
  9 tests OK
```

Smoke:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_basewrong_margin_smoke_s2_20260516/report.json

result:
  accepted_integrated_language_knowledge_healing
  validation MCQ: 12/16 -> 12/16
  language top1 agreement: 0.9166666865348816
```

Standard validation-controlled runs:

```text
all-CE run:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_external9000_valctrl_s120_20260516/report.json
  decision:
    rejected
  best step:
    120
  text core CE delta:
    0.0003188513219356537
  validation MCQ:
    base 192/256
    core 191/256
    gain -0.00390625
  category movement:
    commonsense -1
    science 0

base_wrong + margin run:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_external9000_basewrong_margin_valctrl_s120_20260516/report.json
  decision:
    rejected
  best step:
    80
  text core CE delta:
    0.000014291144907474518
  validation MCQ:
    base 192/256
    core 191/256
    gain -0.00390625
  category movement:
    commonsense 0
    science -1
```

Interpretation:

```text
The validation-controlled scaffold is now safer: it no longer promotes a run
that only preserves language while losing auxiliary MCQ accuracy. The
base_wrong + margin loss reduces text CE regression by ~22x compared with
all-CE and repairs the commonsense regression, but the core still falls one hit
behind base on science. This is not a public-benchmark promotion.

Next valid work is not another tiny public sweep. The bottleneck is that the
mandatory core is still too weak to create stable option-level knowledge gains
over Qwen core_off. The next experiment should strengthen the core-side
training signal or use a larger non-test validation curriculum before any
independent MMLU-Pro 1024 rerun.
```

## 2026-05-16 - Cloned Qwen core and positive-gain triage

Implementation update:

```text
src/qtrm_mm/qwen_backbone_qtrm.py
  added clone_qwen_core_layers support so the QTRM core can own trainable
  deep-copied Qwen layer modules instead of only reusing frozen/shared Qwen
  layers.

scripts/394_train_qwen35_integrated_language_knowledge_healing.py
  added:
    clone_qwen_core_layers reporting
    skip_save_checkpoint for full-disk-safe rejected runs
    MCQ flip counts:
      both_correct
      both_wrong
      base_correct_core_wrong
      base_wrong_core_correct
    base_wrong_mcq_retries for correction-focused sampling
    base_correct_kl_extra_batch_size for a separate preservation KL stream

scripts/394_run_qwen35_integrated_language_knowledge_healing.sh
  fixed:
    UNFREEZE_QWEN_LAYER_INDICES now preserves an explicit empty value.
    Before this fix, UNFREEZE_QWEN_LAYER_INDICES='' still became layer 23
    because ${VAR:-23} treats empty as unset.
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py

bash -n scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_language_knowledge_healing

result:
  11 tests OK
```

Clone-core smoke:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_clonecore_smoke_s1_20260516/report.json

result:
  accepted smoke
  qtrm_parameters: 105405441
  qtrm_trainable_parameters: 105405440
  qwen_core_layers_cloned: true
```

Important correction:

```text
The earlier "core-only" interpretation was wrong. The runner bug meant layer
23 was still trainable. After fixing the runner, the true frozen-Qwen
core-only run was rejected:

report:
  local_eval/qwen35_integrated_language_knowledge_healing_clonecore_true_coreonly_basewrong_margin_preserve_valctrl_s120_20260516/report.json

result:
  qwen_trainability.mode: frozen
  text accepted
  language accepted
  external validation MCQ:
    base 191/256
    core 190/256
    gain -0.00390625
  decision:
    rejected
```

Partial-unfreeze cloned-core runs:

```text
layer23 + residual_scale 0.05:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_clonecore_coreonly_basewrong_margin_preserve_valctrl_s120_20260516/report.json
  corrected interpretation:
    not core-only; layer 23 was trainable
  result:
    accepted only because min_eval_mcq_gain was 0.0
    base 191/256
    core 191/256
    gain 0.0
    not public promotion

layer23 + residual_scale 0.08 + positive-gain threshold:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_clonecore_l23_r008_basewrong_margin_preserve_posgain_s120_20260516/report.json
  result:
    rejected
    best validation gain 0.0
    threshold required +1/256

layer23 + stronger base-correct KL:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_clonecore_l23_preserve02_margin08_posgain_s120_20260516/report.json
  result:
    rejected
    base 191/256
    core 191/256
    gain 0.0
    flip_counts after:
      both_correct 191
      both_wrong 65
      base_correct_core_wrong 0
      base_wrong_core_correct 0

layer23 + base_wrong retry4 correction:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_clonecore_l23_basewrong_retry4_posgain_s120_20260516/report.json
  result:
    rejected
    base 192/256
    core 191/256
    gain -0.00390625
    flip_counts after:
      both_correct 191
      both_wrong 64
      base_correct_core_wrong 1
      base_wrong_core_correct 0

layer23 + dual-stream retry3 correction/preservation:
  report:
    local_eval/qwen35_integrated_language_knowledge_healing_clonecore_l23_dualstream_retry3_posgain_s120_20260516/report.json
  result:
    rejected
    base 191/256
    core 190/256
    gain -0.00390625
    core_ce_delta 0.000021344516426324844
    flip_counts after:
      both_correct 190
      both_wrong 65
      base_correct_core_wrong 1
      base_wrong_core_correct 0
```

Interpretation:

```text
Cloning a Qwen layer into the mandatory QTRM core gives the core enough
capacity to preserve language and nearly tie the base. It still does not create
reliable base-wrong corrections on held-out external MCQ validation. Stronger
preservation KL prevents regressions but also leaves zero corrections. Stronger
base_wrong retry/correction pressure destabilizes preservation before it learns
new correct answers. A first dual-stream correction/preservation objective also
remained negative, so the next improvement likely needs stronger core placement,
larger non-test correction data, or a better preference/verifier target rather
than only loss-weight tuning.

Therefore this stage is not an architecture-complete result and not a public
benchmark improvement. The next bottleneck is not checkpoint selection or
residual scale. It is the objective/data path for producing causal
base_wrong_core_correct flips without introducing base_correct_core_wrong
regressions.
```

## 2026-05-16 - Nested H/L core controls opened

Diagnosis:

```text
The integrated Qwen language/knowledge healing script was not actually using
TRM-style nested H/L recurrence. It hardcoded:

  n_core_layers=1
  h_cycles=1
  l_cycles=1
  outer_steps=1

This made the mandatory QTRM path a shallow one-pass post-backbone residual,
not a nested learner/reasoner. That can explain why language was preserved but
base_wrong_core_correct flips did not emerge.
```

Implementation:

```text
scripts/394_train_qwen35_integrated_language_knowledge_healing.py
  added CLI args:
    --n-core-layers
    --h-cycles
    --l-cycles
    --outer-steps
    --core-convergence-halt-enabled / --no-core-convergence-halt
    --core-step-conditioning-enabled / --no-core-step-conditioning

scripts/394_run_qwen35_integrated_language_knowledge_healing.sh
  added env controls:
    N_CORE_LAYERS
    H_CYCLES
    L_CYCLES
    OUTER_STEPS
    CORE_CONVERGENCE_HALT_ENABLED
    CORE_STEP_CONDITIONING_ENABLED

default correction:
  direct script defaults and runner defaults are now:
    h_cycles=3
    l_cycles=6
    outer_steps=3
    core_convergence_halt_enabled=true
    core_step_conditioning_enabled=true
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py

bash -n scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen35_integrated_language_knowledge_healing

result:
  11 tests OK
```

Nested smoke:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_nested_h3l6_smoke_s2_20260516/report.json

settings:
  h_cycles: 3
  l_cycles: 6
  outer_steps: 1
  qwen_trainability.mode: frozen
  clone_qwen_core_layers: true

result:
  executable
  text accepted
  language accepted
  16-case MCQ tie: base 12/16, core 12/16

interpretation:
  This is only an execution smoke, not a promotion. The next useful run is a
  small positive-gain validation gate with the same nested controls.
```

Default nested smoke:

```text
report:
  local_eval/qwen35_integrated_language_knowledge_healing_nested_default_smoke_s0_20260516/report.json

result:
  executable
  h_cycles: 3
  l_cycles: 6
  outer_steps: 3
  core_convergence_halt_enabled: true
  core_step_conditioning_enabled: true
  language accepted

interpretation:
  The default `394` path is no longer a 1/1/1 shallow residual path. Any future
  1/1/1 use must be explicitly labeled as a smoke/ablation, not canonical TRM
  nested learning.
```

## 2026-05-16 - Mid-layer QTRM causal insertion

Diagnosis:

```text
The H3/L6/outer3 final-hidden residual path still failed public MCQ validation:

  base_hits: 191 / 256
  core_hits: 189 / 256
  base_wrong_core_correct: 0
  base_correct_core_wrong: 2
  core_converged_fraction: 0.0

This means the core was recurrent, but it was too late in the causal path. It
perturbed Qwen's final hidden state after Qwen had already completed its answer
computation.
```

Implementation:

```text
src/qtrm_mm/qwen_backbone_qtrm.py
  added core_insertion_mode:
    final_residual
    mid_layer_suffix

  mid_layer_suffix path:
    Qwen full forward with hidden_states
    -> take hidden state after core_insert_after_layer
    -> mandatory QTRM core
    -> rerun remaining Qwen suffix layers
    -> Qwen final norm / LM head

scripts/394_train_qwen35_integrated_language_knowledge_healing.py
scripts/394_run_qwen35_integrated_language_knowledge_healing.sh
  default:
    CORE_INSERTION_MODE=mid_layer_suffix
    CORE_INSERT_AFTER_LAYER=11
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/qtrm_mm/qwen_backbone_qtrm.py \
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py

bash -n scripts/394_run_qwen35_integrated_language_knowledge_healing.sh

PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_qwen_backbone_qtrm \
  tests.test_qwen35_integrated_language_knowledge_healing

result:
  24 tests OK
```

Smoke:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_default_smoke_s0_20260516/report.json

settings:
  core_insertion_mode: mid_layer_suffix
  core_insert_after_layer: 11
  h_cycles: 3
  l_cycles: 6
  outer_steps: 3

result:
  executable
  MCQ 4-case tie
  mean_core_outer_iterations: 2.75
  core_converged_fraction: 0.25
```

Small validation:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_shared_posgain_s80_20260516/report.json

decision:
  rejected

metrics:
  base_hits: 94 / 128
  core_hits: 93 / 128
  gain: -0.0078125
  base_wrong_core_correct: 1
  base_correct_core_wrong: 2
  mean_core_outer_iterations: 2.8125
  core_converged_fraction: 0.1953125

interpretation:
  Mid-layer placement is directionally better than final-hidden residual
  because it produced the first public-MCQ correction flip and nonzero
  convergence. It is still not accepted because regressions exceed corrections.
```

Roadmap consequence:

```text
Do not revert to final-hidden residual tuning. The next bottleneck is
preservation/correction separation in the mid-layer path. Try adapter-only or
smaller residual starts with a clean/mismatched adapter reset, stronger
base-correct preservation, and selection by flip balance:

  base_wrong_core_correct > base_correct_core_wrong
  full core gain > 0
```

## 2026-05-16 - First Strict Mid-layer Integrated Gain

Result:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/report.json

decision:
  accepted_integrated_language_knowledge_healing

path:
  Qwen3.5 prefix layers
  -> mandatory H=3/L=6 QTRM recurrent core at layer 11
  -> Qwen3.5 suffix layers
  -> Qwen3.5 LM head

settings:
  core_insertion_mode: mid_layer_suffix
  core_delta_adapter_mode: adapter_only
  core_adapter_dim: 512
  qwen_trainable_parameters: 0
  language_anchor_weight: 1.5
  base_correct_option_kl_weight: 0.8
```

Metrics:

```text
before:
  base_hits: 191 / 256
  core_hits: 191 / 256

after:
  base_hits: 191 / 256
  core_hits: 195 / 256
  gain: +0.015625
  base_wrong_core_correct: 12
  base_correct_core_wrong: 8
  commonsense hit_delta: +1
  science hit_delta: +3
  language top1 agreement: 0.9166667
  generation max repeated token run: 1
```

Important negative control:

```text
Without the language-anchor KL, the same correction objective reached
base_wrong_core_correct=15 but collapsed ordinary prompt top1 agreement to
0.0833 by pushing option-letter tokens such as "B" into the global LM logits.
This proves the bottleneck was not "core cannot learn"; it was correction loss
leaking into the general language channel.
```

Roadmap consequence:

```text
This is the first non-preservation-only Qwen-integrated QTRM result where the
mandatory core beats donor-only on held-out public MCQ validation while keeping
normal language generation usable.

Next promotion gate:
  run independent scripts/390 public-MCQ evaluation at larger eval size
  keep language-anchor KL
  tighten base_correct regression after confirming the gain reproduces
```

## 2026-05-16 - Independent 390 Recheck

External commonsense/science 64-case recheck:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_langanchor_390_external64_20260516/report.json

decision:
  accepted_m4_public_mcq_core_gain

metrics:
  base: 48 / 64 = 0.7500
  core: 50 / 64 = 0.78125
  gain: +0.03125
```

MMLU-Pro 64-case recheck:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_langanchor_390_mmlupro64_20260516/report.json

decision:
  rejected_m4_public_mcq_core_gain

metrics:
  base: 25 / 64 = 0.390625
  core: 24 / 64 = 0.375
  gain: -0.015625
```

Interpretation:

```text
The accepted mid-layer QTRM core gain is real on the external
commonsense/science validation distribution and survives an independent scorer.
It is not yet a broad MMLU-Pro gain. The next bottleneck is data/task
coverage, not another macro-architecture rewrite.

Next action:
  train or repair on a balanced MMLU-Pro train/dev slice while keeping the
  language-anchor KL and strict base/core flip accounting.
```

## 2026-05-16 - MMLU Repair Diagnostics

Implementation correction:

```text
Bug found:
  Qwen3.5 tokenizer padding_side is right.
  Several batched MCQ/language paths used logits[:, -1, :].
  For short rows inside a padded batch this reads the pad-token position, not
  the actual final prompt token.

Fix:
  scripts/394_train_qwen35_integrated_language_knowledge_healing.py now uses
  last_nonpad_logits(...) for batched MCQ and language-anchor next-token losses.
  scripts/367_eval_qwen_backbone_language_gate.py now uses the same rule for
  top-k language preservation checks.
```

Experiments after the correction:

```text
adaptive gate:
  report: local_eval/qwen35_integrated_midlayer_suffix_adaptive_gate_mmlupro_s80_20260516/report.json
  base/core: 23/22
  gain: -0.015625
  corrections/regressions: 2 / 3

supervised adaptive gate:
  report: local_eval/qwen35_integrated_midlayer_suffix_adaptive_gate_supervised_mmlupro_s80_20260516/report.json
  base/core: 23/21
  gain: -0.03125
  corrections/regressions: 1 / 3

fast-LR gate:
  report: local_eval/qwen35_integrated_midlayer_suffix_adaptive_gate_fastlr_mmlupro_s80_20260516/report.json
  base/core: 23/21
  gain: -0.03125
  corrections/regressions: 0 / 2

trainable cloned core layer:
  report: local_eval/qwen35_integrated_midlayer_suffix_clonecore_mmlupro_s40_20260516/report.json
  base/core: 23/20
  gain: -0.046875
  corrections/regressions: 1 / 4
```

Diagnostic conclusion:

```text
MMLU-Pro is not improved by simply opening a token gate or making the wrapped
core layer trainable for a short repair run. The current core signal contains
some useful flips, but also higher-priority harmful flips in health/philosophy.

An option-score arbitration sweep on the accepted mid-layer checkpoint found a
non-label rule that can preserve base and take one separable correction:
  condition: base_margin <= 5, core_margin >= 0.25, switch_adv >= 1
  result on the 64-case shuffled MMLU-Pro slice: base 23 -> arbitrated 24

Roadmap consequence:
  Do not scale this repair until the decision/gating problem is made explicit.
  The next useful stage is a learned confidence/arbitration head trained to
  predict "apply core delta" from base/core score geometry, while keeping the
  QTRM core mandatory in the causal path.
```

## 2026-05-16 - Karpathy Autoresearch Probe Applied

Reference material:

```text
repo:
  https://github.com/karpathy/autoresearch

local clone:
  references/official/autoresearch

commit:
  228791fb499afffb54b46200aca536f79142f117
```

Applied pattern:

```text
fixed-budget experiment
one decisive metric
keep/discard ledger
small scoped change before scaling
```

New probe:

```text
script:
  scripts/395_autoresearch_arbitration_probe.py

runner:
  scripts/395_run_autoresearch_arbitration_probe.sh

report:
  local_eval/qwen35_integrated_autoresearch_arbitration_probe_mmlupro64_20260516/report.json

ledger:
  local_eval/qwen35_integrated_autoresearch_arbitration_probe/results.tsv
```

Result:

```text
decision:
  rejected_arbitration_probe

fit split:
  base: 12 / 32
  core: 9 / 32
  arbitration: 12 / 32

held-out eval split:
  base: 11 / 32
  core: 10 / 32
  arbitration: 11 / 32

best rule:
  base_margin_max: 0.0
  core_margin_min: 0.0
  switch_adv_min: -1.0
```

Interpretation:

```text
The fixed-budget autoresearch probe did not find a keep-worthy score-geometry
arbitration rule on the split evaluation. It selected no-switch because the fit
split contained no useful separable correction. Therefore, do not scale blind
MCQ repair or simple post-hoc arbitration. The next action is to improve the
core signal or add richer separability features before training a learned
arbitration head.
```

Follow-up linear policy probe:

```text
script:
  scripts/395_autoresearch_arbitration_probe.py --policy linear

report:
  local_eval/qwen35_integrated_autoresearch_linear_arbitration_probe_mmlupro64_20260516/report.json

decision:
  rejected_arbitration_probe

fit:
  base: 12 / 32
  core: 9 / 32
  linear arbitration: 12 / 32
  switches: 0

held-out eval:
  base: 11 / 32
  core: 10 / 32
  linear arbitration: 11 / 32
  switches: 0

ledger:
  local_eval/qwen35_integrated_autoresearch_arbitration_probe/results.tsv
```

Updated conclusion:

```text
Both threshold and linear score-geometry arbitration reject. This narrows the
next useful work: MMLU-Pro needs a better recurrent core signal, not another
post-hoc gate over the current base/core option scores.
```

## 2026-05-16 - MMLU Scorer SSOT Correction

Issue found:

```text
scripts/390_eval_qwen35_integrated_public_mcq.py scored option letters with
max over acceptable one-token renderings:
  A, " A", "\nA"

scripts/394_train_qwen35_integrated_language_knowledge_healing.py and
scripts/395_autoresearch_arbitration_probe.py scored the same renderings with
logsumexp probability mass.

This made a small false-positive possible: the option-only checkpoint looked
like base 25/64 -> core 26/64 under the old 390 max scorer, but rejected under
the training/autoresearch scorer.
```

Fix:

```text
390 now uses option_score_from_log_probs(...), i.e. logsumexp over acceptable
one-token renderings, matching 394/395.

391 public-MCQ healing also used logits[:, -1, :] on padded batches. It now
gathers the final non-pad token logits just like 394.

test:
  tests/test_qwen35_integrated_public_mcq_eval.py
  tests/test_qwen35_integrated_public_mcq_healing.py
```

Canonical recheck after the fix:

```text
checkpoint:
  local_eval/qwen35_integrated_midlayer_suffix_optiononly_mmlupro_s80_20260516/last_core.pt

report:
  local_eval/qwen35_integrated_midlayer_suffix_optiononly_mmlupro64_ssot_eval_20260516/report.json

result:
  rejected_m4_public_mcq_core_gain
  base: 26 / 64
  core: 25 / 64
  gain: -0.015625
```

Consequence:

```text
Do not claim MMLU-Pro improvement from the option-only run. The current
bottleneck remains recurrent core signal quality, not just evaluator/gating.
All future MCQ gates must use the same option-letter probability-mass scorer
and non-pad final-token gathering.
```

Preservation repair follow-up:

```text
report:
  local_eval/qwen35_integrated_midlayer_suffix_optiononly_preserve_repair_s80_20260516/report.json

decision:
  rejected_integrated_language_knowledge_healing

best periodic step:
  0

result:
  base: 23 / 64
  core: 21 / 64
  gain: -0.03125
  base_wrong_core_correct: 1
  base_correct_core_wrong: 3

interpretation:
  stronger base-correct preservation loss did not repair the harmful
  health/philosophy flips. This supports changing the core signal/capacity or
  integration path instead of adding more post-hoc or preservation-only loss.
```

## 2026-05-16 - SSOT Revalidation Of Accepted QTRM Candidates

New runner:

```text
scripts/396_run_qwen35_integrated_ssot_revalidation.sh
```

Purpose:

```text
Do not trust historical accepted=true flags after the scorer/padding fixes.
Re-run canonical candidates through the same 390 public-MCQ evaluator using:
  - option-letter probability mass scorer
  - explicit checkpoint/suite/settings
  - strict gain threshold
  - summary.jsonl ledger
```

Summary:

```text
ledger:
  local_eval/qwen35_integrated_ssot_revalidation_20260516/summary.jsonl

midlayer_external64:
  checkpoint: local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/last_core.pt
  result: accepted
  base/core: 49 / 50
  gain: +0.015625

midlayer_mmlupro64:
  checkpoint: local_eval/qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor_s160_20260516/last_core.pt
  result: rejected
  base/core: 26 / 23
  gain: -0.046875

optiononly_mmlupro64:
  checkpoint: local_eval/qwen35_integrated_midlayer_suffix_optiononly_mmlupro_s80_20260516/last_core.pt
  result: rejected
  base/core: 26 / 25
  gain: -0.015625

public_coreonly_mmlu256:
  checkpoint: local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt
  result: rejected
  base/core: 93 / 93
  gain: 0
```

Scale sweep on MMLU-Pro64:

```text
ledger:
  local_eval/qwen35_integrated_ssot_scale_sweep_mmlupro64_20260516/summary.jsonl

residual_scale 0.00:
  base/core: 26 / 26
  gain: 0

residual_scale 0.25:
  base/core: 26 / 23
  gain: -0.046875

residual_scale 0.50:
  base/core: 26 / 23
  gain: -0.046875

residual_scale 0.75:
  base/core: 26 / 22
  gain: -0.0625

residual_scale 1.00:
  base/core: 26 / 23
  gain: -0.046875
```

Diagnosis:

```text
The current mid-layer QTRM core is real but narrow. It helps the external
commonsense/science validation slice, but harms broad MMLU-Pro as soon as its
residual is allowed to influence logits. This is not a "more scale" problem.
The next architecture/training loop must improve the core signal itself before
claiming broad reasoning improvement.
```

## 2026-05-16 QTRM-Native TRM-Condition Lock

Decision:

```text
The canonical QTRM target is now explicitly:
  TRM-paper-condition QTRM-native loop reasoning model.
```

Meaning:

```text
mandatory core = causally necessary recursive latent loop
not merely a block that is executed during forward
```

Required path:

```text
prompt tokens
-> tokenizer
-> native embeddings/backbone
-> repeated TRM-style latent loop (z_L/z_H or equivalent)
-> core-dependent readout
-> LM logits
-> autoregressive text
```

Promotion requires:

```text
full > no-loop/shallow-loop
full > core_off/think0
deeper loop > shallow loop
state reset/zero/shuffle/corruption damages the same metric
readout_off damages the same metric
normal LM logits generate the answer
```

Demoted to diagnostic:

```text
Qwen donor/residual adapter improvement
Qwen-preservation-first tuning
MemoryOS/RAG/tool/verifier success
forced-choice gain without greedy LM generation gain
any result where residual_scale=0 or core_off preserves the claim
```

Updated:

```text
docs/wiki/architecture/qtrm-native-first-roadmap.md
docs/wiki/decisions/qtrm-native-hard-lock.md
docs/wiki/decisions/orthodox-trm-general-llm-direction.md
docs/wiki/index.md
```

Milestone dependency added:

```text
M-A: prove recursive core actually improves reasoning.
M-B: attach the proven core to the normal LM path.
M-C: heal language only after the core is causal in that LM path.
```

## 2026-05-16 Executable M-A/M-B/M-C Gate

Implemented:

```text
scripts/372_qtrm_native_27b_milestone_status.py
  adds core_to_lm_to_healing_dependencies:
    M_A_RECURSIVE_CORE_REASONING_PROOF
    M_B_CORE_TO_LM_ATTACHMENT
    M_C_LANGUAGE_HEALING_AFTER_CORE

scripts/380_refresh_m6_status.sh
scripts/385_refresh_m7_status.sh
  pass QTRM_REPORT as --core-reasoning-report
```

Current regenerated status:

```text
report:
  local_eval/qtrm_native_27b_milestone_status/report.json
  local_eval/qtrm_native_27b_milestone_status/report.md

M-A:
  accepted
  source: local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/report.json
  full_generation_exact: 0.6067708333333334
  min_family_generation_exact: 0.4140625
  full_minus_think0: 0.5859375
  full_minus_worst_ablation: 0.5716145833333334

M-B:
  accepted
  same report proves normal LM-generation evidence because the score is
  full_generation_exact with ablation drops.

M-C:
  accepted
  source: local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_repairv3_s4000_20260515/report.json

next_action:
  work on M7_NATIVE_2B_3B_PUBLIC_BENCH_PARITY
```

Verification:

```text
PYTHONPATH=src .venv/bin/python -m unittest tests.test_qtrm_native_27b_milestone_status
.venv/bin/python -m py_compile scripts/372_qtrm_native_27b_milestone_status.py
bash -n scripts/380_refresh_m6_status.sh scripts/385_refresh_m7_status.sh
```

## 2026-05-16 M7 Strict Scorer Correction

Problem:

```text
The previous M7 public MCQ evaluator parsed option letters from the whole
generated completion. When the native model echoed the prompt/options, the
scorer could read the echoed "A." option as the answer.
```

Fix:

```text
scripts/384_eval_qtrm_native_public_mcq.py
  extract_answer_text() now separates answer-bearing text from prompt echo.
  The scorer no longer counts letters from echoed options.
  Metrics now report invalid_pred_rate, prompt_echo_rate, and pred histogram.

scripts/372_qtrm_native_27b_milestone_status.py
  propagates M7 invalid/prompt-echo metrics into the milestone status.
```

Strict re-eval:

```text
report:
  local_eval/m7_qtrm_native_qwen35pre_mmlu_pro_balanced256_eval_strict_20260516/report.json

suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl

checkpoint:
  local_eval/qtrm_native_pretrained_init_qwen35_compact_external4500_s3600_20260515/last.pt

accuracy:
  1 / 256 = 0.00390625

invalid_pred_rate:
  0.98046875

prompt_echo_rate:
  1.0

pred_answer_histogram:
  <empty>: 251
  A: 5
```

Diagnosis:

```text
M7 is blocked before public knowledge/reasoning parity. The current native
checkpoint usually reconstructs/echoes the MCQ prompt instead of producing a
single answer letter after Assistant:. The next repair target is strict
instruction-following / answer-only generation on public-MCQ format while
preserving M-A/M-B core causality.
```

Next action:

```text
Build an M7 answer-only MCQ healing gate:
  train/eval on public-style MCQ prompts;
  reject prompt echo;
  require pred_answer_histogram not dominated by A or empty;
  then re-run strict M7 before any larger public benchmark claim.
```

## 2026-05-16 M7A Answer-Only Gate Accepted

Implementation:

```text
scripts/384_eval_qtrm_native_public_mcq.py
  fixed generation scoring to decode only newly generated suffix token ids.
  The earlier strict scorer decoded prompt+completion, which made compact
  tokenizer prompt reconstruction look like prompt echo.

scripts/397_build_m7a_public_mcq_answer_only_corpus.py
  builds public-style answer-only rows.

scripts/398_score_m7a_answer_only_gate.py
  rejects invalid answers, prompt echo, and single-label collapse.

scripts/400_train_qtrm_native_public_mcq_final_token.py
  trains only the final answer token after the prompt, instead of sequence CE
  over the whole MCQ prompt/options text.

scripts/401_run_qtrm_native_m7a_final_token_healing.sh
  one-command M7A final-token runner.
```

Key diagnosis:

```text
The compact Qwen-preinit checkpoint is not suitable for public MCQ answer
generation. Its compact tokenizer has unk_compact_id == eos_compact_id, so OOV
prompt pieces become EOS/UNK and the model learns empty output.

The full-vocab Qwen-tokenizer checkpoint can produce valid answer letters, but
plain sequence CE over whole MCQ records teaches option-line patterns such as
"A." and can collapse to a single option.

The correct M7A repair objective is final-token option CE/margin on the answer
position, with A-J training data. A-D-only auxiliary MMLU data is structurally
insufficient for MMLU-Pro A-J answer space.
```

Accepted run:

```text
train:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516

runner-equivalent:
  scripts/401_run_qtrm_native_m7a_final_token_healing.sh

init checkpoint:
  local_eval/qtrm_native_language_bootstrap_qwen_tokenizer_quality_s1000_20260515/last.pt

train suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl
  A-J labels present

strict eval suite:
  local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl
  first 64 cases

gate report:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/m7a_gate_report.json

decision:
  accepted_m7a_public_mcq_answer_only_gate

strict generation metrics:
  accuracy: 9 / 64 = 0.140625
  invalid_pred_rate: 0.0
  prompt_echo_rate: 0.0
  max_pred_fraction: 0.234375
  pred_answer_histogram:
    B: 11
    D: 1
    E: 15
    F: 9
    H: 6
    I: 9
    J: 13
```

Interpretation:

```text
M7A is not public benchmark parity. It only closes the answer-surface bottleneck:
the model now emits a single option letter without prompt echo, empty output,
or A-only collapse on a strict held-out public-style MCQ slice.

Next bottleneck:
  improve actual knowledge/reasoning accuracy while preserving this accepted
  answer path. The next experiment should optimize public-MCQ correctness with
  non-test A-J validation data and core-depth/core-off ablations, not return to
  compact-vocab sequence CE or residual-only architecture shopping.
```

## 2026-05-16 M7B Core-Depth Gate Accepted

Implementation:

```text
scripts/402_score_m7b_core_depth_gate.py
  scores whether a full-depth native recurrent core improves strict greedy MCQ
  accuracy over no/shallow thinking while preserving the M7A answer surface.

scripts/403_run_qtrm_native_m7b_core_depth_gate.sh
  runs depth0/depth1/depth2/depth4 strict MCQ evals and then scores M7B.
```

Run:

```text
OUT_ROOT=local_eval/qtrm_native_m7b_core_depth_gate_m7a_s300_20260516 \
CHECKPOINT=local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt \
MAX_CASES=64 DEVICE=cuda PYTHONPATH=src \
bash scripts/403_run_qtrm_native_m7b_core_depth_gate.sh
```

Gate report:

```text
local_eval/qtrm_native_m7b_core_depth_gate_m7a_s300_20260516/m7b_gate_report.json
```

Decision:

```text
accepted_m7b_public_mcq_core_depth_gate
```

Depth sweep:

```text
depth0:
  6 / 64 = 0.09375
  histogram: A 64

depth1:
  7 / 64 = 0.109375
  histogram: A 3, B 2, E 59

depth2:
  7 / 64 = 0.109375
  histogram: A 1, B 8, E 53, F 1, H 1

depth4:
  9 / 64 = 0.140625
  histogram: B 11, D 1, E 15, F 9, H 6, I 9, J 13
```

Acceptance checks:

```text
gain_vs_baseline:
  +0.046875

gain_vs_best_shallow:
  +0.03125

surface:
  invalid_pred_rate: 0.0
  prompt_echo_rate: 0.0
  max_pred_fraction: 0.234375
```

Interpretation:

```text
M7B is still not public benchmark parity. It is the first public-style MCQ
native result where:
  1. the answer surface is valid under strict greedy generation;
  2. full recursive depth beats no/shallow thinking on the same held-out slice;
  3. the gain is measured through normal LM token generation, not a sidecar.

Next bottleneck:
  M7C should increase correctness beyond this 9/64 baseline while keeping the
  same M7A/M7B gates. Required next experiments should use broader non-test
  A-J MCQ supervision or a stronger native/integrated core curriculum, then
  rerun M7B plus larger 256/512 public slices.
```

## 2026-05-16 M7C Answer-CE Repair Attempts Rejected

Implementation added:

```text
scripts/404_materialize_aj_remap_mcq.py
  builds non-test A-J remapped MCQ training data without using MMLU-Pro test
  labels.

scripts/400_train_qtrm_native_public_mcq_final_token.py
  gained optional base-checkpoint option-KL preservation and trainable
  parameter name filtering.
```

Built data:

```text
local_eval/m7_public_reasoning_suite/mmlu_aux_all_validation_train_aj_remap_plus_mmluproval64_20260516.jsonl
cases: 2112
answer distribution:
  A 219, B 229, C 199, D 228, E 208, F 194, G 216, H 233, I 180, J 206
```

Rejected runs:

```text
local_eval/qtrm_native_m7c_aj_remap_s300_20260516/report.json
  final_eval: 7 / 64 = 0.109375
  M7B gate:
    local_eval/qtrm_native_m7c_aj_remap_s300_m7b_20260516/m7b_gate_report.json
    rejected: gain_vs_baseline and gain_vs_best_shallow both 0.015625

local_eval/qtrm_native_m7c_aj_remap_s120_lr5e5_20260516/report.json
  final_eval: 5 / 64 = 0.078125

local_eval/qtrm_native_m7c_external_natural_s120_lr2e5_20260516/report.json
  final_eval: 7 / 64 = 0.109375

local_eval/qtrm_native_m7c_external_preservekl_s100_lr1e5_20260516/report.json
  option-KL preserve on MMLU-Pro validation 64
  final_eval: 8 / 64 = 0.125

local_eval/qtrm_native_m7c_mmluaux_preservekl_s80_lr1e5_20260516/report.json
  option-KL preserve on MMLU-Pro validation 64
  final_eval: 8 / 64 = 0.125

local_eval/qtrm_native_m7c_coreonly_mmluaux_preservekl_s150_20260516/report.json
  trainable: think/core_halt only, 132609 / 63913729 params
  final_eval: 8 / 64 = 0.125

local_eval/qtrm_native_m7c_coreonly_ajremap_preservekl_s60_20260516/report.json
  trainable: think/core_halt only, 132609 / 63913729 params
  final_eval: 7 / 64 = 0.109375

local_eval/qtrm_native_m7c_depthgain_coreonly_val64_s100_20260516/report.json
  core-only depth-gain trajectory loss
  final_eval: 5 / 64 = 0.078125

local_eval/qtrm_native_m7c_depth_sweep_m7a_20260516/m7b_gate_report.json
  no training, accepted M7A checkpoint depth sweep
  depth4: 9 / 64 = 0.140625
  depth6: 6 / 64 = 0.09375
  depth8: 5 / 64 = 0.078125
  depth12: 2 / 64 = 0.03125
```

Decision:

```text
M7C answer-token CE repair is rejected for now. It preserves the answer surface
better than whole-sequence CE, but it still pulls the model away from the
accepted M7A/M7B checkpoint. This held for:
  full-model tuning
  low-LR short tuning
  option-KL preserve tuning
  core-only tuning
  natural external MCQ and A-J remapped MCQ data
  depth-gain trajectory tuning
```

Research consequence:

```text
Do not continue M7C by simply adding more MCQ answer CE. The next correctness
step must train the recursive core's trajectory, not just the final option
token. A stronger TRM-style curriculum should supervise or regularize latent
recursive state transitions, then promote only if:
  depth4 still beats depth0/depth1/depth2;
  final output remains normal LM generation;
  strict M7A surface does not regress;
  correctness improves beyond 9/64 on the held-out public-style slice.

The accepted M7A checkpoint also has a depth sweet spot at 4 recursive steps.
Blindly increasing loop count is rejected: depth6/8/12 all regressed.
```

## 2026-05-16 M7D Knowledge Bootstrap Smoke Rejected

Added:

```text
scripts/405_eval_m7c_checkpoint_soup_in_memory.py
  evaluates checkpoint interpolation without writing full `last.pt` files.

scripts/406_build_mcq_knowledge_text_corpus.py
  converts non-test MCQ rows into answer-content language records:
    question -> correct option text
```

Non-test corpus:

```text
local_eval/m7_public_reasoning_suite/mcq_knowledge_text_aux_mmlu_external_20260516.jsonl
records: 3531
sources:
  local_eval/m7_public_reasoning_suite/mmlu_aux_all_validation_train_20260516.jsonl
  local_eval/m7_public_reasoning_suite/external_mcq_train_pool_2000_20260516.jsonl
```

M7C soup triage:

```text
summary:
  local_eval/qtrm_native_m7c_soup_triage_20260516/summary.json

best:
  9 / 64 = 0.140625
  no improvement over accepted M7A/M7B baseline

decision:
  rejected as promotion path
```

Knowledge bootstrap attempts:

```text
knowledge-only CE:
  local_eval/qtrm_native_m7d_knowledge_bootstrap_aux_s120_20260516/report.json
  rejected
  symptom: repeated-word loop / "the the" language degradation

knowledge-only -> answer repair:
  local_eval/qtrm_native_m7d_knowledge_then_m7a_val70_s240_20260516/report.json
  final next-token diagnostic: 2 / 64
  rejected

preservation-mix low-LR knowledge CE:
  local_eval/qtrm_native_m7d_knowledge_mix_low_lr_s80_20260516/report.json
  rejected by semantic relevance gate, but language sample was non-degenerate

preservation-mix -> answer repair:
  local_eval/qtrm_native_m7d_knowledge_mix_then_m7a_val70_s240_20260516/report.json
  next-token diagnostic: 8 / 64

preservation-mix -> answer repair -> KL continuation:
  local_eval/qtrm_native_m7d_knowledge_mix_then_m7a_val70_s240_klcont_s120_20260516/report.json
  next-token diagnostic: 9 / 64
  strict M7B:
    local_eval/qtrm_native_m7d_knowledge_mix_then_m7a_val70_s240_klcont_s120_m7b_20260516/m7b_gate_report.json
    rejected
    strict depth4: 2 / 64
    invalid_pred_rate: 0.796875
```

Important evaluator lesson:

```text
Trainer `initial_eval` / `final_eval` in
scripts/400_train_qtrm_native_public_mcq_final_token.py scores next-token
option log-probability. It can look acceptable while strict greedy generation
emits empty output. Promotion must use strict M7A/M7B generation gates, not the
trainer diagnostic alone.
```

Decision:

```text
M7D knowledge-text CE on the current 64M native backbone is rejected as a fast
route to public-MCQ improvement. It either destroys language fluency or creates
a logprob/generation mismatch. The baseline remains:
  local_eval/qtrm_native_m7a_final_token_space_mmluproval64_s300_20260516/last.pt
  depth4 strict generation: 9 / 64
```

## 2026-05-16 Qwen-Integrated Public MCQ SSOT Revalidation Rejected

Context:

```text
The Qwen-integrated path is a native-candidate only when Qwen tokenizer,
backbone, QTRM core, and LM head live in one standalone graph with
runtime_donor=false. It is not the canonical donor-sidecar path.
```

Existing result audit:

```text
Aggregated existing qwen35_integrated*/report.json files.
Some old reports showed accepted public-MCQ core gains, but the strongest
ones were training-internal/best-periodic or non-SSOT reports.
```

Revalidation commands used standalone public evaluator:

```text
scripts/390_run_qwen35_integrated_m4_public_mcq.sh
scorer: next-token option-letter probability mass
core_off: force_core_off=True
core_on: mandatory QTRM core in the same Qwen graph
```

Results:

```text
public_coreonly_mmlu256 SSOT summary:
  local_eval/qwen35_integrated_ssot_revalidation_20260516/public_coreonly_mmlu256/report.json
  checkpoint: local_eval/qwen35_integrated_public_mcq_healing_coreonly_val64_to_test256_s120_20260516/last_core.pt
  core_off 93 / 256
  core_on  93 / 256
  gain 0.0
  decision rejected

same checkpoint direct rerun:
  local_eval/qwen35_integrated_repro_public_coreonly_mmlu256_rerun2_20260516/report.json
  core_off 93 / 256
  core_on  93 / 256
  gain 0.0
  decision rejected

l23open seed20260520 rerun:
  local_eval/qwen35_integrated_repro_l23open_seed20260520_mmlu256_20260516/report.json
  core_off 93 / 256
  core_on  94 / 256
  gain 0.00390625
  decision rejected

512 residual_scale=0.06 rerun:
  local_eval/qwen35_integrated_repro_public_coreonly_mmlu512_resid0p06_20260516/report.json
  core_off 194 / 512
  core_on  196 / 512
  gain 0.00390625
  decision rejected
```

Decision:

```text
The Qwen-integrated public-MCQ core gain is not currently canonical. Earlier
96/256 and 197/512 accepted-looking reports should be treated as historical
diagnostics until reproduced by standalone SSOT evaluation.
```

Research consequence:

```text
Do not keep tuning this residual integrated path for benchmark claims until
the core effect is stronger than the evaluator noise floor. The next acceptable
route is either:
  1. return to QTRM-native TRM-condition proof: recursive depth/state ablations
     improve held-out LM-logit or strict generation metrics; or
  2. redesign the integrated core objective so standalone public eval shows
     repeatable core_on > core_off by at least the configured threshold.

Training-periodic eval, best-checkpoint selection metrics, and old reports are
not promotion evidence unless the exact checkpoint is rerun through the
standalone SSOT gate.
```

## 2026-05-17 DGX llama-server baseline and M7B scale-out audit

Set up DGX llama-server for the Qwen3.6-27B MTP GGUF baseline:

```text
DGX server script:
  /mnt/data4tb/qwen36-mtp-llama-server.sh

DGX repo copy:
  /mnt/data4tb/qtrm_multimodal_memoryos/scripts/407_qwen36_mtp_llama_server_dgx_local.sh

local control script:
  scripts/407_dgx_qwen36_mtp_llama_server.sh

endpoint:
  http://192.168.219.113:18082/v1

server:
  /mnt/data4tb/llama-cpp-turboquant-cuda/build/bin/llama-server

model:
  /mnt/data4tb/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf
```

Runtime:

```text
ctx: 131072
reasoning: off
MTP draft: on
ngram-mod: on
KV: V turbo4, K auto-upgraded to q8_0 for GQA quality
```

M6 scoped raw-reasoning baseline reruns:

```text
64-case:
  local_eval/dgx_qwen36_mtp_proxy_baseline_64_20260517/report.json
  score: 7 / 64 = 0.109375

256-case:
  local_eval/dgx_qwen36_mtp_proxy_baseline_256_20260517/report.json
  score: 35 / 256 = 0.13671875

512-case:
  local_eval/dgx_qwen36_mtp_proxy_baseline_512_20260517/report.json
  score: 75 / 512 = 0.146484375

DGX 512 manifest:
  local_eval/m6_scoped_raw_reasoning_manifest_dgx512_20260517/report.json
  decision: accepted_m6_scoped_raw_reasoning_win
```

M6 interpretation:

```text
The QTRM-native L5 scoped raw-reasoning result remains accepted against the
DGX Qwen3.6-MTP-GGUF proxy:
  QTRM: 0.6067708333333334 over 768 cases
  DGX Qwen proxy: 0.146484375 over 512 cases

This remains a scoped custom-suite win, not public benchmark parity.
```

M7B public MCQ core-depth scale-out:

```text
64-case DGX rerun:
  local_eval/dgx_m7b_core_depth_gate_m7a_s300_20260517/m7b_gate_report.json
  decision: accepted
  depth0: 6 / 64 = 0.09375
  depth4: 9 / 64 = 0.140625

256-case DGX rerun:
  local_eval/dgx_m7b_core_depth_gate_m7a_s300_256_20260517/m7b_gate_report.json
  decision: rejected
  depth0: 39 / 256 = 0.15234375
  depth4: 24 / 256 = 0.09375
```

Decision:

```text
M7B remains useful as a small-slice causal-depth diagnostic, but it cannot be
promoted as a public-MCQ reasoning result. The next work item is core-depth
scale-out repair on 256/512 public-style cases.
```

## 2026-05-17 TRM Raw Scale-Out Transfer Accepted To Len16

Public-style M7B repair stayed rejected, so the work returned to the raw
reasoning scale-out ladder where knowledge/language confounds are removed.

Accepted transfer chain:

```text
len4 accepted checkpoint:
  local_eval/research_gate_runner/qtrm_native_l5_multifamily_standard/last.pt
  full: 0.6067708333333334
  full_minus_think0: 0.5859375
  ablation_drop: 0.5716145833333334
  min_family: 0.4140625

len4 -> len8:
  DGX local_eval/dgx_trm_raw_scaleout_len8_resume_len4_to_len8_20260517_191305/report.json
  decision: accepted_trm_raw_scaleout_len8
  full: 0.15104166666666666
  full_minus_think0: 0.15104166666666666
  ablation_drop: 0.109375
  min_family: 0.03125

len8 -> len12:
  DGX local_eval/dgx_trm_raw_scaleout_len12_resume_len8_to_len12_20260517_191414/report.json
  decision: accepted_trm_raw_scaleout_len12
  full: 0.10416666666666667
  full_minus_think0: 0.10416666666666667
  ablation_drop: 0.06770833333333334
  min_family: 0.015625

len12 -> len16:
  DGX local_eval/dgx_trm_raw_scaleout_len16_resume_len12_to_len16_20260517_191926/report.json
  decision: accepted_trm_raw_scaleout_len16
  full: 0.109375
  full_minus_think0: 0.109375
  ablation_drop: 0.0859375
  min_family: 0.05813953488372093

len16 standalone checkpoint rerun, 512 held-out cases:
  DGX local_eval/dgx_trm_raw_scaleout_len16_standalone_len16_ckpt512_20260517_192654/report.json
  decision: accepted_trm_raw_scaleout_len16
  full: 0.09375
  full_minus_think0: 0.09375
  ablation_drop: 0.0703125
  min_family: 0.05263157894736842
```

Interpretation:

```text
This is the first clean recurrent-compute scaling ladder in this repo:
depth0 cannot solve the longer tasks, full recurrent depth solves a nonzero
slice, and state/op ablations remove much of the gain. It is still not public
benchmark parity or a full TRM-like breakthrough. The next bottlenecks are
seed stability, higher family-balanced floors, and then returning to public
MCQ/language gates.
```

Rejected family-DRO repair:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_familydro_repair_20260517_193004/report.json
decision: rejected
full: 0.076171875
full_minus_think0: 0.076171875
ablation_drop: 0.05078125
min_family: 0.023391812865497075
reject_reasons:
  full_exact_below_threshold
  family_exact_below_threshold
```

Interpretation:

```text
Strong family-DRO plus retention KL over-corrected and damaged the accepted
len16 recurrent trajectory. Continue with lower-pressure family-floor periodic
selection rather than treating family-DRO as the canonical repair.
```

Low-pressure family-floor selection:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_familyfloor_select_20260517_193542/report.json
decision: rejected
full: 0.09375
full_minus_think0: 0.09375
ablation_drop: 0.0703125
min_family: 0.05263157894736842
best_periodic_eval: step 0
```

This did not improve the family floor over the accepted len16 checkpoint.
Training did briefly raise overall exact to 0.171875 at step 800, but the
minimum family exact stayed 0.047058823529411764, so the family-balanced gate
still rejected it.

Ouro-style h-state probe:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_probe_h_20260517_194508/report.json
decision: accepted_trm_raw_scaleout_len16
greedy full: 0.09375
core_answer_probe_exact: 0.115234375

core_step_probe_by_depth:
  depth1: 0.048828125
  depth8: 0.046875
  depth10: 0.064453125
  depth12: 0.0625
  depth16: 0.08984375
```

Interpretation:

```text
The recurrent h-state contains answer information beyond greedy decoding, and
the final depth16 state is more probe-readable than early depth1. This is a
real latent-state signal, but it is not yet the clean monotonic sharpening
reported for strong looped-LM style systems. The next objective should
stabilize depth trajectories and family-specific transitions, not merely add
more answer-token CE.
```

Trajectory-stabilization trial:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_trace_depth_family_20260517_194749/report.json
decision: rejected
full: 0.09375
full_minus_think0: 0.09375
ablation_drop: 0.0703125
min_family: 0.05263157894736842
best_periodic_eval: step 0

periodic high-water marks:
  step400 full: 0.15234375, min_family: 0.046511627906976744
  step600 full: 0.1875, min_family: 0.03488372093023256
```

The state-trace depth loss improved overall exact during training but did not
improve the family floor. This confirms that the accepted trajectory can be
overfit toward easier families.

Operation breakdown on accepted len16:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_opbreakdown_20260517_195401/report.json

family exact:
  checksum: 0.15294117647058825
  modchain: 0.05263157894736842
  revchain: 0.07602339181286549

modchain by last op:
  op01 add1: 0 / 29 = 0.0
  op04 mul2: 0 / 27 = 0.0
  op07 affine3: 1 / 31 = 0.03225806451612903
  op06 affine2: 5 / 19 = 0.2631578947368421
```

Late-op hard replay:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_len16_modchain_lateop_replay_20260517_195528/report.json
decision: rejected
full: 0.09375
full_minus_think0: 0.09375
ablation_drop: 0.0703125
min_family: 0.05263157894736842
best_periodic_eval: step 0

periodic high-water marks:
  step400 full: 0.1484375, min_family: 0.046511627906976744
  step700 full: 0.19140625, min_family: 0.023255813953488372
```

Interpretation:

```text
Hard replay of late modchain op01/op04 raised overall exact during training but
made family balance worse. The next architecture change should separate or
condition transition routes by family/causal order, then force every route to
retain a recurrent-depth gain. More global CE, family-DRO, or late-op replay is
not enough.
```

## 2026-05-17 - Preserved Single-Order Router Accepted

The len16 trajectory-routing bottleneck was addressed with a preservation-first
architecture change:

```text
new think_structure:
  single_order_router

mechanism:
  route0 = the existing accepted single recurrent transition
  route1 = causal GRU + suffix-biased context, then the same think block
  router = token-stream-derived 2-route selector

preservation rule:
  forced route0 must match the accepted single checkpoint before any route
  training. No route-output LayerNorm is allowed because it changes route0.
```

The first smoke exposed a bug in the design: a post-blend LayerNorm broke
route0 preservation and collapsed full exact to 0. That was removed, and the
router bias was initialized to `[8, -8]` so additive resume starts as the
accepted single model.

Preservation smoke:

```text
DGX local_eval/dgx_trm_raw_scaleout_len16_single_order_router_preserve_smoke_20260517_202312/report.json
steps: 0
resume: local_eval/dgx_trm_raw_scaleout_len16_resume_len12_to_len16_20260517_191926/last.pt

full: 0.109375
order_route0: 0.109375
order_route1: 0.125
min_family: 0.09090909090909091
```

The first route-training attempt used the old `family_order` target
(`revchain -> route1`, others route0). It improved full exact but missed the
family floor by a hair:

```text
DGX local_eval/dgx_single_order_router_fast_s200_20260517_203439/report.json
decision: rejected
full: 0.1328125
full_minus_think0: 0.1328125
ablation_drop: 0.1171875
min_family: 0.05813953488372093
```

The accepted repair changed the router target to `chain_vs_checksum`: both
ordered chain families (`modchain`, `revchain`) are route1, while checksum stays
route0. Only the new missing route/router parameters were trainable; the
accepted single checkpoint tensors were frozen.

```text
DGX local_eval/dgx_single_order_router_chain_target_s200_20260517_204312/report.json
decision: accepted_single_order_router_chain_target_len16

full: 0.12890625
think0: 0.0
full_minus_think0: 0.12890625
ablation_drop: 0.11328125
min_family: 0.06976744186046512

by_family:
  checksum: 0.24705882352941178
  modchain: 0.06976744186046512
  revchain: 0.07058823529411765

router last_hlh_prob:
  checksum: 0.16707628965377808
  modchain: 0.9165406227111816
  revchain: 0.9740087985992432

forced route0:
  full: 0.109375

forced route1:
  full: 0.078125
```

Interpretation:

```text
This is a real architecture result, not just more CE:
  - route0 preserves the accepted checkpoint exactly enough to remain usable;
  - only additive route parameters were trained;
  - the learned router moves chain families onto the new route;
  - depth/state/op ablations still remove the gain;
  - the final output remains normal LM logits.

The route1 candidate itself is still weak when forced globally, so the next
step is not to claim breakthrough. The next step is route specialization:
checksum route0, chain route1, and then a 512-case standalone rerun plus
len20 transfer.
```

512-case standalone rerun:

```text
DGX local_eval/dgx_single_order_router_chain_target_512rerun_20260517_205432/report.json
decision: accepted_single_order_router_chain_target_len16_512rerun

full: 0.140625
think0: 0.0
full_minus_think0: 0.140625
ablation_drop: 0.119140625
min_family: 0.06432748538011696
state_reset: 0.021484375
op_zero: 0.01953125

by_family:
  checksum: 0.27647058823529413
  modchain: 0.08187134502923976
  revchain: 0.06432748538011696

router last_hlh_prob:
  checksum: 0.17065902054309845
  modchain: 0.9107509851455688
  revchain: 0.9774150252342224

forced route0:
  full: 0.09375

forced route1:
  full: 0.078125
```

Interpretation:

```text
The route-conditioned repair reproduced on a 512-case standalone gate.
This is stronger than the 200-case report because it confirms:
  - no-think/core-off cannot solve the task;
  - 16 recurrent steps are causally useful;
  - corrupting recurrent state or operations removes most of the gain;
  - the router consistently separates checksum from ordered-chain families.

This is still not an ASI or public-benchmark breakthrough. It is the first
preservation-first architectural foothold after many rejected route variants.
The canonical next experiment is length transfer from len16 to len20 while
retaining the same route-conditioned core.
```

## 2026-05-17 - Len20 Route-Conditioned Core Accepted

The len20 transfer showed that the remaining bottleneck was not just
architecture. It was data/selection pressure on the ordered-chain families.

Naive len20 transfer from the accepted len16 route-conditioned checkpoint:

```text
DGX local_eval/dgx_single_order_router_len20_transfer_resize_20260517_210209/report.json
decision: rejected
reason: family_exact_below_threshold

full: 0.11328125
think0: 0.0
full_minus_think0: 0.11328125
ablation_drop: 0.08203125
min_family: 0.029239766081871343

by_family:
  checksum: 0.27647058823529413
  modchain: 0.03508771929824561
  revchain: 0.029239766081871343
```

Route1-only repair nearly passed:

```text
DGX local_eval/dgx_single_order_router_len20_route1_repair_20260517_213141/report.json
decision: rejected
reason: family_exact_below_threshold

full: 0.173828125
think0: 0.0
full_minus_think0: 0.173828125
ablation_drop: 0.13671875
min_family: 0.05847953216374269

by_family:
  checksum: 0.38823529411764707
  modchain: 0.05847953216374269
  revchain: 0.07602339181286549
```

A modchain-only continuation over-corrected and damaged the family floor:

```text
DGX local_eval/dgx_single_order_router_len20_modchain_floor_20260517_220346/report.json
decision: rejected
reason: family_exact_below_threshold

full: 0.17578125
min_family: 0.04093567251461988

by_family:
  checksum: 0.43529411764705883
  modchain: 0.05263157894736842
  revchain: 0.04093567251461988
```

The accepted run used balanced chain data, family-DRO losses, and periodic
family-floor checkpoint selection:

```text
DGX local_eval/dgx_single_order_router_len20_familyfloor_select_20260517_222156/report.json
decision: accepted_single_order_router_len20_familyfloor

full: 0.1953125
think0: 0.0
full_minus_think0: 0.1953125
ablation_drop: 0.158203125
min_family: 0.07602339181286549
state_reset: 0.03125
op_zero: 0.037109375

by_family:
  checksum: 0.43529411764705883
  modchain: 0.07602339181286549
  revchain: 0.07602339181286549

best periodic eval:
  step: 600
  full: 0.1953125
  family_floor: 0.07602339181286549
  teacher_forced_answer_loss: 0.857588529586792
```

Interpretation:

```text
The data/selection hypothesis was correct for len20:
  - checksum stayed learnable across rejected runs;
  - modchain/revchain were the limiting families;
  - route1-only pressure moved the family floor to the acceptance boundary;
  - blind single-family repair overfit the target family and hurt the other;
  - family-floor periodic selection accepted the same architecture without
    changing the final LM-logit path.

This is the first accepted len20 QTRM-native route-conditioned recurrent core
on the selection seed. It is still a synthetic raw-reasoning gate, not
public-benchmark intelligence. The next canonical step is an independent
eval-seed rerun of this len20 checkpoint, followed by len24 transfer only if
the rerun holds.
```

Independent eval seed 9338 did not hold:

```text
DGX local_eval/dgx_single_order_router_len20_seed9338_20260517_230036/report.json
decision: rejected
reason: family_exact_below_threshold

full: 0.1640625
think0: 0.0
full_minus_think0: 0.1640625
ablation_drop: 0.1328125
min_family: 0.029239766081871343

by_family:
  checksum: 0.4117647058823529
  modchain: 0.029239766081871343
  revchain: 0.05263157894736842
```

Consequence:

```text
Len20 is architecturally alive but not seed-stable. This strengthens the
data/selection diagnosis: checksum is robust, recursive ablations are causal,
but ordered-chain generalization is still under-trained. Do not promote to
len24 or public-style claims until a multi-seed len20 gate holds.
```

Data-scale repair against seed 9338 also rejected:

```text
DGX local_eval/dgx_single_order_router_len20_seed9338_datascale_20260517_230608/report.json
decision: rejected
reason: family_exact_below_threshold

settings:
  train_cases: 32768
  steps: 1200
  eval_seed: 9338
  periodic_selection: family_floor

full: 0.166015625
think0: 0.0
full_minus_think0: 0.166015625
ablation_drop: 0.134765625
min_family: 0.04093567251461988
state_reset: 0.03125
op_zero: 0.03125

best periodic eval:
  step: 200
  full: 0.166015625
  family_floor: 0.04093567251461988
  teacher_forced_answer_loss: 0.8592787981033325
```

Interpretation:

```text
Simply increasing synthetic case count and step budget did not stabilize
seed9338. The core remains causal, but ordered-chain generalization does not
cross the family-floor gate. The next architecture/debugging action is not a
larger blind run; it is a smaller diagnostic that changes the transition
objective or state representation and then reruns the same multi-seed gate.
```

Operational fix added after this long run:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new flags:
  --save-every-steps N
  --save-best-periodic-checkpoint

smoke:
  PYTHONPATH=src .venv/bin/python scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py ...
  local_eval/checkpoint_save_smoke/checkpoint_step_000001.pt
  local_eval/checkpoint_save_smoke/latest.pt
  local_eval/checkpoint_save_smoke/best_periodic.pt
  local_eval/checkpoint_save_smoke/latest_progress.json
```

Long runs should now use, at minimum:

```text
--save-every-steps 100
--save-best-periodic-checkpoint
```

## 2026-05-18 - Forced Route Prefix-Anchor Objective

After the seed9338 data-scale run rejected, the bottleneck was reclassified as
ordered-chain transition generalization rather than dataset quantity. The next
candidate is not another larger run. It adds a route-local transition objective:
force route1 and train it on causal-prefix prompts through the normal LM logits
path.

Implementation:

```text
scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

new function:
  forced_route_prefix_depth_anchor_loss

new flags:
  --forced-route-prefix-depth-anchor-loss-weight
  --forced-route-prefix-depth-anchor-route
  --forced-route-prefix-depth-anchor-families
  --forced-route-prefix-depth-anchor-max-cases
  --forced-route-prefix-depth-anchor-every
  --forced-route-prefix-depth-anchor-min-depth
  --forced-route-prefix-depth-anchor-weight-power
```

Rationale:

```text
forced_route_intermediate_depth_loss:
  full prompt stays full length, route is asked to expose intermediate answers
  after each recurrent depth.

forced_route_prefix_depth_anchor_loss:
  prompt itself is shortened to each causal prefix, route1 is forced, and the
  answer is still produced by LM logits. This gives route1 local transition
  supervision before it must carry the same transition inside a full program.
```

Verification:

```text
.venv/bin/python -m py_compile scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

PYTHONPATH=src .venv/bin/python scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py \
  --out-dir local_eval/forced_route_prefix_anchor_smoke \
  --target-level forced-route-prefix-anchor-smoke \
  --steps 1 \
  --train-cases 8 \
  --eval-cases 8 \
  --task-families modchain,revchain,checksum \
  --eval-task-families modchain,revchain,checksum \
  --eval-family-order-invariant \
  --include-family-tag \
  --program-len 2 \
  --modulus 8 \
  --d-model 32 \
  --n-heads 4 \
  --d-ff 64 \
  --batch-size 4 \
  --lr 1e-3 \
  --device cpu \
  --train-think-steps 2 \
  --eval-think-steps 2 \
  --backbone mha_etd \
  --think-structure single_order_router \
  --forced-route-prefix-depth-anchor-loss-weight 0.1 \
  --forced-route-prefix-depth-anchor-route 1 \
  --forced-route-prefix-depth-anchor-families modchain,revchain \
  --forced-route-prefix-depth-anchor-max-cases 4 \
  --forced-route-prefix-depth-anchor-min-depth 1 \
  --forced-route-prefix-depth-anchor-weight-power 1.0 \
  --accept-min-exact 0 \
  --accept-min-depth-gain 0 \
  --accept-min-ablation-drop 0 \
  --accept-min-family-exact 0 \
  --accepted-decision forced_route_prefix_anchor_smoke
```

Next DGX candidate once connectivity returns:

```text
scripts/412_dgx_len20_prefix_anchor_gate.sh run
scripts/412_dgx_len20_prefix_anchor_gate.sh submit
scripts/412_dgx_len20_prefix_anchor_gate.sh status
scripts/412_dgx_len20_prefix_anchor_gate.sh tail

Use the accepted len20 selection checkpoint or the seed9338 failed checkpoint,
train only route1/router, and add:

--forced-route-prefix-depth-anchor-loss-weight 0.2
--forced-route-prefix-depth-anchor-route 1
--forced-route-prefix-depth-anchor-families modchain,revchain
--forced-route-prefix-depth-anchor-max-cases 64
--forced-route-prefix-depth-anchor-min-depth 1
--forced-route-prefix-depth-anchor-weight-power 1.0
--save-every-steps 100
--save-best-periodic-checkpoint

Promote only if seed9338 and the original selection seed both pass len20
family-floor gates.
```

## 2026-05-18 - Continuous Literature Watch For QTRM Bottlenecks

The research loop is now explicit: QTRM architecture work should keep checking
recent papers while experiments run, but every paper idea must be translated
into a bottleneck-specific, falsifiable local gate. Newness alone is not a
reason to mutate the architecture.

Latest scan targets for the current ordered-chain / latent recurrence
bottleneck:

```text
Depth-recurrent latent reasoning:
  Thinking Deeper, Not Longer: Depth-Recurrent Transformers for
  Compositional Generalization
  https://arxiv.org/abs/2603.21676

Relevance:
  shared-weight latent recurrence, final-answer-only silent thinking,
  LayerScale/identity-biased recurrence, and explicit depth scaling are directly
  relevant to QTRM core stability.

Latent reasoning framing:
  LLM Reasoning Is Latent, Not the Chain of Thought
  https://arxiv.org/abs/2604.15726

Relevance:
  supports evaluating hidden-state trajectories separately from visible CoT and
  serial compute; matches QTRM's raw-intelligence gate philosophy.

Latent CoT limits:
  Capabilities and Fundamental Limits of Latent Chain-of-Thought
  https://arxiv.org/abs/2602.01148

Relevance:
  warns that latent CoT can excel on exploration-like tasks but fail on exact
  computation; reinforces curriculum and family-floor gates.

Abstract latent tokens:
  Thinking Without Words: Efficient Latent Reasoning with Abstract
  Chain-of-Thought
  https://arxiv.org/abs/2604.22709

Relevance:
  reserved latent tokens may be useful later for language-model integration,
  but current QTRM-native gates should first prove recurrent core causality.

Adaptive stopping:
  Adaptive Stopping for Multi-Turn LLM Reasoning
  https://arxiv.org/abs/2604.01413

Relevance:
  useful for future halt/uncertainty policy, but only after the recurrent core
  produces a measurable family-floor gain.

Recursive sparse structure:
  ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and
  Long-Context Reasoning
  https://arxiv.org/abs/2510.01585

Relevance:
  recurrent memory and sparse structure are relevant to later long-context
  scaling, not the immediate len20 ordered-chain repair.
```

Immediate consequence:

```text
If the current prefix-anchor len20 run rejects, the next architecture repair
should not be another data-scale continuation. The most relevant prior-guided
candidate is recurrence stabilization:

  identity-biased recurrent update
  LayerScale/small residual update around the recursive core
  final-answer-only or delayed-depth supervision variant

The gate remains the same:
  seed9338 len20 family-floor pass
  original-seed retention pass
  destructive core/depth/route ablations still remove the gain
```

## 2026-05-18 - Len20 Prefix-Anchor DGX Result

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_prefix_anchor_seed9338_20260518_090836/report.json
```

Summary:

```text
decision: rejected
reject_reasons:
  family_exact_below_threshold

decisive_metrics:
  full_generation_exact: 0.171875
  think0_generation_exact: 0.0
  full_minus_think0: 0.171875
  full_minus_worst_ablation: 0.140625
  min_family_generation_exact: 0.0409357
  state_reset_generation_exact: 0.03125
  op_zero_generation_exact: 0.0273438

best_periodic:
  step: 200
  generation_exact: 0.171875
  min_family_generation_exact: 0.0409357
```

Interpretation:

```text
Prefix-anchor did not solve the len20 worst-family bottleneck. It preserved a
strong causal recurrent-core signal:

  full - think0 = 0.171875
  full - worst ablation = 0.140625

But it failed the promotion threshold because the weakest family stayed below
0.06. The failure is therefore not "no core effect"; it is unstable
ordered-chain family generalization.
```

Next architecture direction:

```text
Stop increasing dataset size or prefix-anchor pressure. The next candidate
should target recurrence stability directly, guided by current
depth-recurrent/latent-reasoning literature:

  identity-biased recurrent update
  LayerScale/small residual recurrent delta
  delayed-depth or final-answer-only supervision variant

Acceptance remains unchanged:
  seed9338 len20 family-floor >= 0.06
  original seed retention pass
  destructive recurrent-state ablations remove the gain
```

## 2026-05-18 - Recurrent LayerScale Repair Candidate

Implemented candidate:

```text
think_structure:
  single_order_router_residual_scale

files:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/414_dgx_len20_recurrent_layerscale_gate.sh
```

Mechanism:

```text
old single_order_router:
  next_state = route_mix(state)

new single_order_router_residual_scale:
  mixed = route_mix(state)
  next_state = state + layer_scale * (mixed - state)

layer_scale:
  shape [1, 1, d_model]
  initialized to 1.0
```

Why this is preservation-first:

```text
When layer_scale = 1.0, the new structure reproduces the previous router path.
This lets the accepted len20 checkpoint be loaded with --resume-allow-missing
without destroying the accepted route. Training then only learns whether some
recurrent dimensions should move less or more aggressively.
```

Local verification:

```text
.venv/bin/python -m py_compile \
  scripts/335_train_qtrm_native_etd_probe.py \
  scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py

random preservation smoke:
  old single_order_router vs new single_order_router_residual_scale
  max_abs_diff: 7.152557373046875e-07

runner smoke:
  RESUME_FROM=none REMOTE_PYTHON=.venv/bin/python ... \
    scripts/414_dgx_len20_recurrent_layerscale_gate.sh run-local
  decision: layerscale_runner_local_smoke
```

DGX command:

```text
scripts/414_dgx_len20_recurrent_layerscale_gate.sh submit
```

Promotion gate:

```text
seed9338:
  full >= 0.10
  full - think0 >= 0.06
  full - worst ablation >= 0.06
  min_family >= 0.06

then original-seed retention rerun with the same gate
```

## 2026-05-18 - Continuous Latest-Paper Search Rule Reaffirmed

Answer to the process question:

```text
Yes. QTRM architecture work must keep searching current papers while improving
the model. But the loop is not paper shopping. A new paper is useful only if it
maps to the measured bottleneck and can be converted into one falsifiable gate.
```

Current measured bottleneck:

```text
The accepted len20 recurrent route has a real causal core signal, but hard
ordered-chain families still fail the family-floor gate. Recent failed repairs
show that weak after-the-fact anti-collapse loss, scalar time conditioning,
time-gated routing, and zero-init state stream adapters do not break the
0.0409 min-family plateau.
```

Fresh literature implications checked on 2026-05-18:

```text
LoopFormer
  https://loopformer.github.io/
  Mechanism: variable loop trajectories, t/dt conditioning, shortcut
  consistency. Already tested in minimal form through time-conditioned and
  time-gated routers; result plateaued, so the next attempt needs stronger
  trajectory-level supervision, not another scalar time adapter.

RLTT: Rewarding Latent Thought Trajectories
  https://arxiv.org/abs/2602.10520
  Mechanism: assign credit across the whole latent reasoning trajectory rather
  than only final answer state. This directly targets the QTRM failure where
  final LM CE preserves core causality but does not teach hard-family
  transitions well enough.

Solve the Loop: Attractor Models for Language and Reasoning
  https://huggingface.co/papers/2605.12466
  Mechanism: fixed-point iterative refinement with implicit differentiation
  and adaptive convergence. This is relevant as a future larger redesign if
  explicit recurrent unroll training keeps showing trajectory collapse.

Latent Lookahead Training
  https://machinelearning.apple.com/research/latent-lookahead
  Mechanism: recursively feed hidden states forward for selected tokens and
  supervise multiple future targets. This supports a prefix/depth trajectory
  objective for QTRM rather than only final-token answer CE.

ADEPT
  https://arxiv.org/abs/2601.03700
  Mechanism: adaptive early exit. Useful for halt/compute policy later, but not
  the next fix until the recurrent trajectory itself improves.

Continuous Latent Diffusion Language Model / Latent-DARM
  https://huggingface.co/papers/2605.06548
  https://arxiv.org/abs/2603.09184
  Mechanism: globally revisable or diffusion-like latent planning. Relevant as
  a long-term alternative branch, but it would change the core training regime;
  it should not replace the current TRM/LoopLM proof gate without a separate
  minimal comparison.
```

Next research-driven action:

```text
Stop adding small route adapters. The next candidate should implement
trajectory-level credit/supervision for the recurrent core:

  per-depth latent trajectory targets
  or RLTT-style dense trajectory reward/proxy
  or latent-lookahead prefix/depth supervision

Promotion is unchanged:
  seed9338 min_family >= 0.06
  original-seed retention pass
  think0/state_reset/op_zero/route ablations remove the gain
```

## 2026-05-18 - Time-Conditioned Router DGX Result

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_time_conditioned_seed9338_20260518_163145/report.json
```

Decision:

```text
rejected
reject_reasons:
  family_exact_below_threshold
```

Metrics:

```text
full_generation_exact: 0.173828125
think0_generation_exact: 0.0
full_minus_think0: 0.173828125
full_minus_worst_ablation: 0.142578125
min_family_generation_exact: 0.04093567251461988
state_reset_generation_exact: 0.03125
op_zero_generation_exact: 0.02734375
```

Comparison:

```text
seed9338 diagnostic:
  min_family: 0.0292398

chain trace anti-collapse:
  min_family: 0.0350877

time-conditioned router:
  min_family: 0.0409357
```

Trace comparison:

```text
modchain:
  late_cosine:     0.999044 -> 0.999106
  final_variance:  2.43117  -> 2.17447
  exact:           0.029240 -> 0.040936

revchain:
  late_cosine:     0.999169 -> 0.999189
  final_variance:  2.38877  -> 2.24329
  exact:           0.052632 -> 0.052632
```

Interpretation:

```text
Time conditioning improved the family-floor metric more than the weak
anti-collapse penalty, but it did not fix the latent trajectory collapse.
The improvement came without making late-depth chain states less collapsed.

Therefore route1 context-level time bias is too weak or placed too late. The
next candidate should inject trajectory conditioning directly into the
route1 update gate, still zero-initialized and preservation-first:

  route1_gate_logits =
    update_gate([route0, route1_candidate, route1_context])
    + time_gate([t, dt])

Promote only with the same seed9338/original-seed family-floor gate.
```

## 2026-05-18 - Time-Gated Router Candidate

Implemented candidate:

```text
think_structure:
  single_order_router_time_gate

files:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/417_dgx_len20_time_conditioned_router_gate.sh
```

Mechanism:

```text
route1_gate_logits =
  single_order_route1_update_gate([route0, route1_candidate, route1_context])
  + single_order_time_gate([t, dt])

t  = (step_index + 1) / total_steps
dt = 1 / total_steps
```

Why this is the next sharper candidate:

```text
The rejected time-conditioned router added time only to route1 context. That
barely changed the late-depth chain traces. Time-gated router puts trajectory
conditioning directly on the route1 update gate, so each recurrent step can
learn how much to accept the route1 transition.
```

Preservation check:

```text
old:
  single_order_router

new:
  single_order_router_time_gate

load old state into new with strict=False:
  missing: single_order_time_gate.weight
  unexpected: none
  max_abs_diff at zero init: 0.0
  allclose_1e-6: true
```

Local runner smoke:

```text
RESUME_FROM=none REMOTE_PYTHON=.venv/bin/python \
RUN_LABEL=time_gate_router OUT_PREFIX=time_gate TARGET_LABEL=time-gate \
THINK_STRUCTURE=single_order_router_time_gate \
TRAIN_PARAM_NAME_REGEX='single_order_time_gate|single_order_route1|trm_order_router' \
... scripts/417_dgx_len20_time_conditioned_router_gate.sh run-local

result:
  completed
```

## 2026-05-18 - State-Stream Router Early Stop

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_state_stream_seed9338_20260518_175534
```

Mid-run:

```text
step100:
  best remains initial checkpoint

step200:
  last_loss: 3.029082775115967
  best remains initial checkpoint

best_periodic_eval:
  step: 0
  generation_exact: 0.1640625
  min_family_generation_exact: 0.029239766081871343
```

Action:

```text
Stopped early to save GPU time.
```

Interpretation:

```text
The zero-init state stream preserved the accepted path but did not produce an
early learning signal. This is weaker than both time-conditioned and time-gated
routes, which reached min_family 0.0409357 by step200.

Current evidence after this sequence:
  accepted seed9338 diagnostic: 0.0292398 min_family
  anti-collapse:                0.0350877 min_family
  time-conditioned:             0.0409357 min_family
  time-gated:                   0.0409357 min_family
  state-stream zero-init:       no improvement by step200

Do not claim innovation yet. The next useful work is not another blind adapter.
The next design must either:
  strengthen the core training objective so route1 transition state is directly
  supervised at intermediate prefixes, or
  move from adapter repair to a native recurrent pretraining curriculum where
  the recurrent transition is learned before len20 family-floor promotion.
```

## 2026-05-18 - Time-Gated Router DGX Result

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_time_gate_seed9338_20260518_171439/report.json
```

Decision:

```text
rejected
reject_reasons:
  family_exact_below_threshold
```

Metrics:

```text
full_generation_exact: 0.173828125
think0_generation_exact: 0.0
full_minus_think0: 0.173828125
full_minus_worst_ablation: 0.142578125
min_family_generation_exact: 0.04093567251461988
state_reset_generation_exact: 0.03125
op_zero_generation_exact: 0.02734375
```

Comparison:

```text
time-conditioned router:
  min_family: 0.04093567251461988

time-gated router:
  min_family: 0.04093567251461988

trace metrics:
  nearly identical to time-conditioned router
```

Interpretation:

```text
Time-gate did not add measurable benefit over context-level time conditioning.
Both variants form the same plateau: a small improvement over the accepted
seed9338 diagnostic and anti-collapse candidate, but far below the 0.06
family-floor gate.

Conclusion:
  scalar trajectory conditioning is insufficient. Proceed to the state-stream
  candidate, which changes how recurrent state moves across prompt positions
  while preserving the LM path and zero-init route preservation.
```

## 2026-05-18 - Time-Gated Router Mid-Run Signal

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_time_gate_seed9338_20260518_171439
```

Mid-run:

```text
step100:
  generation_exact: 0.169921875
  min_family_generation_exact: 0.029239766081871343

step200:
  generation_exact: 0.173828125
  min_family_generation_exact: 0.04093567251461988

step300:
  best remains step200
```

Interpretation:

```text
Time-gate matches the time-conditioned router curve and has not broken through
the 0.0409 plateau. This suggests that small route-level time adapters are
insufficient. The route's local update can preserve the accepted path and give
a small family-floor gain, but it does not fix the hard-family transition
representation.
```

Next candidate if final report also rejects:

```text
SST-style state stream candidate:
  maintain a lightweight recurrent state over prompt positions
  blend encoded token state with previous-position stream state
  feed the resulting stream into route1 context
  initialize stream gate near zero to preserve accepted path
  train only stream gate/projection plus route1/router first

Why:
  current failures look like answer-position-only trajectory correction is too
  weak. The model may need position-wise state transport across the prompt
  before the recurrent answer state can compose len20 ordered chains.

Reject constraints:
  no external solver
  no hidden answer path
  same token -> recurrent core -> LM logits path
  same seed9338/original-seed family-floor gate
```

## 2026-05-18 - State-Stream Router Candidate

Implemented candidate:

```text
think_structure:
  single_order_router_state_stream

file:
  scripts/335_train_qtrm_native_etd_probe.py
```

Mechanism:

```text
shifted_state[:, 1:, :] = state[:, :-1, :]

state_stream =
  LayerNorm(Linear([encoded, shifted_state]))

route1_context =
  route1_context + state_stream
```

Why this is different from previous route adapters:

```text
time-conditioned and time-gated variants only changed the local route update.
They did not move the chain-family late-depth collapse enough.

state-stream adds a position-wise transport path from the current recurrent
state at the previous prompt position into the next position. This directly
targets the hypothesis that answer-position-only correction cannot carry a
len20 ordered chain through the prompt.
```

Preservation check:

```text
old:
  single_order_router

new:
  single_order_router_state_stream

load old state into new with strict=False:
  missing:
    single_order_state_stream_in.weight
    single_order_state_stream_norm.weight
    single_order_state_stream_norm.bias
  unexpected: none
  max_abs_diff at zero init: 0.0
  allclose_1e-6: true
```

Local runner smoke:

```text
RESUME_FROM=none REMOTE_PYTHON=.venv/bin/python \
RUN_LABEL=state_stream_router OUT_PREFIX=state_stream TARGET_LABEL=state-stream \
THINK_STRUCTURE=single_order_router_state_stream \
TRAIN_PARAM_NAME_REGEX='single_order_state_stream|single_order_route1|trm_order_router' \
... scripts/417_dgx_len20_time_conditioned_router_gate.sh run-local

result:
  completed
```

## 2026-05-18 - Recurrent LayerScale DGX Result

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_layerscale_seed9338_20260518_100529/report.json
```

Summary:

```text
decision: rejected
reject_reasons:
  family_exact_below_threshold

decisive_metrics:
  full_generation_exact: 0.175781
  think0_generation_exact: 0.0
  full_minus_think0: 0.175781
  full_minus_worst_ablation: 0.144531
  min_family_generation_exact: 0.0409357
  state_reset_generation_exact: 0.03125
  op_zero_generation_exact: 0.0273438

best_periodic:
  step: 200
  generation_exact: 0.175781
  min_family_generation_exact: 0.0409357
```

Interpretation:

```text
LayerScale-style recurrent stabilization slightly improved average exact over
prefix-anchor:

  prefix-anchor full: 0.171875
  layerscale full:    0.175781

But it did not move the worst-family floor:

  prefix-anchor min_family: 0.0409357
  layerscale min_family:    0.0409357

Therefore the current bottleneck is not just recurrence step-size instability.
The remaining failure is family-specific transition representation: the model
keeps a causal core advantage, but the weakest ordered-chain family remains
underrepresented or entangled in the latent state.
```

Next research-driven candidate:

```text
Do not rerun the same objective with more steps as the primary action.
The next candidate should make the transition family/state explicitly
separable inside the canonical recurrent LM path, then ablate it:

  family-conditioned recurrent delta factor
  or contrastive family-state separation loss on core_state_trace
  or route-local state trace probe with stop-gradient target

The gate remains unchanged:
  seed9338 min_family >= 0.06
  original-seed retention pass
  state_reset/op_zero/route ablations remove the gain
```

## 2026-05-18 - Continuous Literature Watch Refresh

The literature loop is active and must continue during QTRM work. The rule is
not "add every new paper"; the rule is:

```text
latest paper
-> measured QTRM bottleneck
-> one causal mechanism
-> one small falsifiable gate
-> wiki result, accepted or rejected
```

Fresh scan additions relevant to the current len20 ordered-chain bottleneck:

```text
Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
https://arxiv.org/abs/2604.07822

Relevance:
  identifies overthinking as a recurrent-depth failure mode where excessive
  recurrence degrades predictions. This matches the current QTRM diagnostic:
  hard chain families keep core causality, but late-depth z_h trajectories
  become nearly frozen.

Latent Chain-of-Thought Improves Structured-Data Transformers
https://arxiv.org/abs/2605.11262

Relevance:
  recurrent latent feedback tokens improve structured prediction, but the
  immediate QTRM action is not to add a new token interface. The useful
  mechanism is measuring whether repeated latent feedback preserves
  discriminative state for hard families.

The Recurrent Transformer: Greater Effective Depth and Efficient Decoding
https://arxiv.org/abs/2604.21215

Relevance:
  supports recurrence as a depth/parameter tradeoff for language pretraining.
  It strengthens the long-term QTRM-native direction, but does not replace the
  current family-floor gate.
```

Current diagnostic consequence:

```text
LayerScale did not fix the worst-family floor. State-trace diagnostics show a
more specific failure:

  checksum:
    higher final state variance
    lower late-depth consecutive cosine
    much better exact accuracy

  modchain/revchain:
    route1 selected almost always
    late-depth consecutive cosine near 0.999
    weak exact accuracy

This points to chain-family late-depth over-smoothing / trajectory collapse,
not simply lack of data, lack of route selection, or global recurrence
instability.
```

Next falsifiable candidate:

```text
Add a chain-family state-trace anti-collapse objective:
  penalize late-depth consecutive cosine above a threshold for modchain/revchain
  preserve or raise final-state variance for modchain/revchain
  keep the same canonical LM answer path

Promote only if:
  seed9338 min_family >= 0.06
  original-seed retention also passes
  core/depth/op ablations still remove the gain
```

## 2026-05-18 - Literature Watch While Chain Trace Gate Runs

Additional current papers/pages found while the chain-family anti-collapse gate
is running:

```text
LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via
Shortcut Modulation
https://loopformer.github.io/

Mechanism:
  condition each loop on internal time t and step size delta-t;
  align shortcut/coarser trajectories to full trajectories;
  make looped depth budget-aware instead of fixed-depth only.

QTRM implication if the current gate rejects:
  QTRM should not only penalize collapse after it happens. It may need an
  explicit trajectory-conditioning signal, because len20 hard-family failure
  looks like a fixed-depth trajectory problem.

State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence
for Latent Space Reasoning
https://arxiv.org/abs/2605.00206

Mechanism:
  stream latent residual state horizontally across positions through learned
  blending; use latent deliberation per position; analyze distinct semantic
  basins and first-token latent survivability under extra computation.

QTRM implication if the current gate rejects:
  investigate whether z_H-only recurrent correction is too local at the final
  answer position. A future candidate may need a lightweight state stream over
  prompt positions, not an external memory or side solver.

Tracing the Traces: Latent Temporal Signals for Efficient and Accurate
Reasoning
https://www.microsoft.com/en-us/research/publication/tracing-the-traces-latent-temporal-signals-for-efficient-and-accurate-reasoning/

Mechanism:
  use latent trajectory change, accumulated movement, and movement toward the
  final state as predictors of successful reasoning traces.

QTRM implication:
  the new state-trace comparator follows this spirit: do not inspect only
  final exact; inspect whether the recurrent trajectory itself is productive.
```

Current run:

```text
script:
  scripts/415_dgx_len20_chain_trace_anticollapse_gate.sh

mid-run seed9338 at step100:
  generation_exact: 0.166015625
  min_family_generation_exact: 0.029239766081871343

mid-run seed9338 at step200:
  generation_exact: 0.166015625
  min_family_generation_exact: 0.03508771929824561

Interpretation:
  slight worst-family movement, but still below the 0.06 gate. Let the 600-step
  run finish, then compare state traces before selecting the next candidate.
```

## 2026-05-18 - Chain Trace Anti-Collapse DGX Result

Run:

```text
/mnt/data4tb/qtrm_multimodal_memoryos_gate/local_eval/
  dgx_single_order_router_len20_chain_trace_anticollapse_seed9338_20260518_154718/report.json
```

Decision:

```text
rejected
reject_reasons:
  family_exact_below_threshold
```

Metrics:

```text
full_generation_exact: 0.171875
think0_generation_exact: 0.0
full_minus_think0: 0.171875
full_minus_worst_ablation: 0.140625
min_family_generation_exact: 0.03508771929824561
state_reset_generation_exact: 0.03125
op_zero_generation_exact: 0.025390625
```

Trace comparison against the seed9338 diagnostic:

```text
checksum:
  late_cosine:     0.879184 -> 0.877564
  final_variance:  6.10126  -> 5.90741
  exact:           0.411765 -> 0.441176

modchain:
  late_cosine:     0.999044 -> 0.998876
  final_variance:  2.43117  -> 2.62017
  exact:           0.029240 -> 0.040936

revchain:
  late_cosine:     0.999169 -> 0.999090
  final_variance:  2.38877  -> 2.52960
  exact:           0.052632 -> 0.035088
```

Interpretation:

```text
The loss moved the intended trace metrics only slightly. It did not move the
worst-family floor enough and regressed revchain. Therefore the failure is not
solved by a weak after-the-fact anti-collapse penalty.

Next candidate should modify the recurrent trajectory itself while preserving
the accepted route. The most direct literature-driven candidate is
LoopFormer-style time/step conditioning:

  route state receives normalized t and dt
  initialization preserves the old route exactly
  train only the new time-condition path plus route1/router first
  gate remains seed9338 family floor and destructive ablations
```

## 2026-05-18 - Time-Conditioned Router Candidate

Implemented candidate:

```text
think_structure:
  single_order_router_time_conditioned

files:
  scripts/335_train_qtrm_native_etd_probe.py
  scripts/417_dgx_len20_time_conditioned_router_gate.sh
```

Mechanism:

```text
At each recurrent step, route1 receives a learned zero-initialized bias from:

  t  = (step_index + 1) / total_steps
  dt = 1 / total_steps

This is a minimal LoopFormer-inspired trajectory-conditioning test. It does
not add a side solver or hidden answer path; it only modulates the accepted
recurrent route's internal trajectory.
```

Preservation check:

```text
old:
  single_order_router

new:
  single_order_router_time_conditioned

load old state into new with strict=False:
  missing: single_order_time_condition.weight
  unexpected: none
  max_abs_diff at zero init: 0.0
  allclose_1e-6: true
```

Local runner smoke:

```text
RESUME_FROM=none REMOTE_PYTHON=.venv/bin/python ... \
  scripts/417_dgx_len20_time_conditioned_router_gate.sh run-local

result:
  completed
```

Promotion gate:

```text
seed9338:
  full >= 0.10
  full - think0 >= 0.06
  full - worst ablation >= 0.06
  min_family >= 0.06

then original-seed retention rerun with the same gate
```
