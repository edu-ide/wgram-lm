# TRM-Like Breakthrough Bar

Status: hard promotion rule, 2026-05-17.

## Position

QTRM is meaningful only if it produces a TRM-like innovation signal. Incremental
formatting gains, answer-only CE, retrieval improvements, or donor preservation
are not enough.

The project must show:

```text
same/smaller parameter budget
-> more latent recurrent compute at inference
-> clearly better reasoning accuracy
-> held-out scale-out beyond the tiny diagnostic slice
-> normal LM output path remains intact
```

## Non-Negotiable Gate

The canonical breakthrough gate is:

```bash
bash scripts/410_run_trm_breakthrough_gate.sh
```

It requires both sides:

```text
M6 scoped raw reasoning:
  accepted M6 manifest
  QTRM full exact >= 0.50
  QTRM margin over Qwen3.6 proxy >= 0.20
  recurrent core gain >= 0.25
  destructive ablation drop >= 0.25
  min family exact >= 0.30

M7B public-style scale-out:
  accepted M7B report
  >= 256 held-out cases
  full depth accuracy >= 0.18
  deeper recurrent core gain >= 0.03 over depth0
  deeper recurrent core gain >= 0.03 over best shallow depth
```

If either side fails, do not call the result TRM-like.

## Current Status

Current M6 is strong:

```text
QTRM full_generation_exact: 0.6067708333333334
Qwen3.6 DGX proxy score: 0.146484375
core_gain: 0.5859375
ablation_drop: 0.5716145833333334
min_family_generation_exact: 0.4140625
```

Current M7B 256 is rejected:

```text
depth0 accuracy: 0.15234375
depth4 accuracy: 0.09375
gain_vs_baseline: -0.05859375
gain_vs_best_shallow: -0.05859375
```

Therefore the current state is:

```text
TRM-like synthetic candidate: yes
TRM-like public-style breakthrough: no
next bottleneck: M7B core-depth scale-out repair
```

## Research Direction

Do not widen the model or add RAG before the breakthrough gate passes. The next
changes must target the recurrent trajectory itself:

```text
1. depth curriculum:
   Train on mixed depths and evaluate depth0/1/2/4/8.

2. trajectory consistency:
   Correct stable answers should become more stable with additional loops.

3. attractor-style stabilization:
   After the model approaches a solution, later loops should not destructively
   move the latent state.

4. early-exit/halt:
   More loops must be optional at inference. If the state converges early, stop.

5. family-balanced hard negatives:
   Repair only the public-style families where depth hurts.
```

Implemented repair path:

```text
scripts/400_train_qtrm_native_public_mcq_final_token.py

new objective knobs:
  --multi-depth-ce-weight
  --multi-depth-ce-depths
  --depth-gain-weight
  --depth-gain-shallow-depths
  --trajectory-kl-weight
  --trajectory-kl-anchor-depth
  --trajectory-kl-compare-depths

DGX fastlane default:
  think_steps=8
  multi_depth_ce_depths=4,8
  depth_gain_shallow_depths=0,1,2,4
  trajectory_kl_anchor_depth=8
  trajectory_kl_compare_depths=6
```

## Kill / Pivot Rule

If repeated DGX runs cannot make M7B 256 pass while preserving M6 and language
non-regression, stop claiming the current QTRM-native architecture is a TRM-like
breakthrough. At that point the correct move is a root redesign of the recurrent
transition objective, not more final-answer tuning.
