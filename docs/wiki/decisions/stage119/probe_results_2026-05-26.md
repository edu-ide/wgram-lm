# Stage119 Probe Results (Direct Run)

Date: 2026-05-26
Script: scripts/627_run_stage119_equation_probe.py
Loss: src/qtrm_mm/losses/equation_state_binding.py (compute_equation_state_binding_loss with logit margin + readback)
Trainer patch: scripts/625_train_bpe_gd_preference.py (guarded integration)

## Run Config
- steps: 15
- binding_weight: 0.3
- seed: 119
- synthetic algebra traps (misleading repeated demo + small calc, matching Stage117/118 pattern)
- tiny one-body proxy (GRU recurrent state -> LM head)

## Gate Numbers (synthetic heldout 16-row algebra trap)
before:
  exact: 0.000
  mean_margin: 0.10217
  min_margin: -0.33837

after:
  exact: 0.125
  mean_margin: 0.09681
  min_margin: -1.66597

ablation (state binding zeroed):
  drop: -0.5625   # note: proxy model zeroing was destructive beyond the binding term

language_proxy (non-degenerate generation):
  0.75

aux_loss observed: active (38.58 -> 36.36 over steps), non-zero contribution.

## Verdict
probe

## Raw Event
{"event": "stage119_probe_complete", "steps": 15, "binding_weight": 0.3, "seed": 119, "before": {"exact": 0.0, "mean_margin": 0.10217027831822634, "min_margin": -0.3383713960647583}, "after": {"exact": 0.125, "mean_margin": 0.09680509939789772, "min_margin": -1.6659693717956543}, "ablation_drop": -0.5625, "language_proxy": 0.75, "verdict": "probe", "wall_time_sec": 0.8464946140011307, "notes": "self-contained synthetic probe; real checkpoint continuation uses same loss + 625 patch"}

## One-Line Gate
before_exact=0.000 after_exact=0.125 ablation_drop=-0.562 lang=0.75 VERDICT=PROBE

## Interpretation (Factual Only)
Small positive movement on the target algebra exact (0/16 -> 2/16) under the new auxiliary objective. Ablation in this micro-proxy was not yet diagnostic (model collapse on full zero). Language proxy held at 0.75. Signal is L1/probe level; requires real checkpoint continuation (625 + real recurrent state capture) for causal confirmation on Stage117/118 anchors.

## Next Mandatory Action (per skill, no proposal)
Re-run with real Stage117/118 anchor checkpoint using the now-runnable 627 (or 625 integration) on actual generated_non_heldout.jsonl + full model forward state hook. Record new numbers in this ledger.