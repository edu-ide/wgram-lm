# PoC Verification Plan: Necessary Conditions for 1B Model to Surpass 27B-scale Models (Qwen3.6-27B class)

**Date**: 2026-06-03  
**Context**: The ultimate goal is architecture innovation that enables a ~1B model to meaningfully outperform much larger models (e.g., Qwen3.6-27B class) through superior per-parameter efficiency, test-time scaling, and inductive biases.

This document treats the verification of the key necessary conditions as the highest-value work. We will use controlled experiments (primarily the clean "pure stochastic effect" probe + state robustness probe, with real 642 gold where possible) on the current OneBodyParallelHybridBlock vs the historical QTRMRecursiveCore.

## Key Necessary Conditions for 1B >> 27B (in this project's context)

1. **Dramatically Higher Effective Compute per Parameter**  
   - The architecture must produce significantly higher quality latent states / reasoning traces per parameter and per FLOP than standard scale-based approaches.

2. **Strong, Reliable Test-Time Compute Scaling**  
   - Increasing recurrence depth / latent steps / thinking time at inference must produce large, predictable gains (not just marginal or saturating early).

3. **Superior Long-Horizon State Stability and Attractor Dynamics**  
   - State quality must remain high over long sequences without rapid degradation. Strong attractors and protection mechanisms must be structurally effective.

4. **Successful Causal Integration of High-Signal Inductive Biases** (the 5.56 signals)  
   - Stochastic breadth, gold structural injection, attractor protection during rehearsal, and scheduled decay must not only be ported but must show large causal contribution in the new architecture.

5. **Efficient Parallel Hybrid Synergy without Destructive Interference**  
   - Combining recurrence and attention inside the same block (One-Body) must create positive synergy rather than the attention branch diluting the recurrence strengths (or vice versa).

6. **Low Training Waste / High Capacity Utilization**  
   - During training, the model must be forced to learn useful iterative computation rather than relying on shortcuts or memorization. Ablations must show that the core mechanisms are actually being used.

7. **Better Data Efficiency through Stronger Priors**  
   - The architecture + curriculum must allow the model to extract more useful capability from the same (or less) high-quality data.

## PoC Experiments (Using Current Infrastructure)

We will use the two main scripts:
- `scripts/train_556_on_parallel_hybrid_minimal.py` (hybrid with real gold support)
- `scripts/run_s2_historical_baseline.py` (old core direct baseline with real gold support)
- The clean probes: `compute_pure_stochastic_contribution` and `simple_state_robustness_probe`

### PoC 1: Test-Time Compute Scaling (Condition 2)
- Experiment: Run both architectures at 80 / 120 / 150 / 200 steps with the same real 642 gold.
- Metric: pure_stochastic_effect + robustness as function of horizon.
- Success signal: Hybrid shows continued gains or much slower saturation than old core.
- Status (as of 2026-06-03): 80/100/120/150 step data collected on both. Hybrid stable/high (~1.46 at 150s), old core flat/slightly declining (~0.25).

### PoC 2: Long-Horizon State Stability & Attractor Quality (Condition 3)
- Experiment: Use the robustness probe at increasing horizons on both architectures with real gold.
- Metric: Degradation under state ablation as horizon increases.
- Success signal: Hybrid maintains high robustness much better than old core at long horizons.
- Status: Initial data from 150-step runs shows hybrid at 1.000 robustness; historical needs more systematic probing.

### PoC 3: Causal Contribution of 5.56 Inductive Biases (Condition 4)
- Experiment: Run full ablation matrix (full, stoch_zero, gold_off, protection_off) on hybrid with real gold at 120 steps using the clean probe.
- Compare causal drops to historical.
- Success signal: Large, clean drops when ablating the 5.56 signals on hybrid (proving successful deep integration).
- Status: Infrastructure ready; high-value next run.

### PoC 4: Parallel Hybrid Synergy vs Interference (Condition 5)
- Experiment: Compare pure recurrence (old core or hybrid with attention heads = 0) vs full hybrid on the clean probe and robustness, with real gold.
- Success signal: Full hybrid significantly outperforms pure recurrence version on the key metrics.
- Status: Can be derived from existing ablations + controlled runs.

### PoC 5: Overall Per-Parameter Efficiency Gap (Conditions 1 + 6)
- Experiment: Head-to-head at matched horizon + real gold using the clean "pure stochastic effect" as a proxy for effective reasoning compute.
- Track the gap as we improve the hybrid (Gating v2, better fusion, etc.).
- Success signal: Consistent large gap (5x+) in favor of the new architecture under real gold.

## Current Status & Next Immediate High-Value Experiments (as of 2026-06-03)

Completed / In Progress:
- 150-step hybrid + historical with real gold → strong evidence for PoC 1 and PoC 2 (hybrid scales/stabilizes much better).
- Multi-seed 120-step real-gold on hybrid.

Highest-value next experiments:
1. Full ablation matrix on hybrid at 120 steps with real gold (PoC 3 — highest priority for causal claim).
2. One more 150-200 step run on hybrid with real gold + systematic robustness probing at multiple horizons (PoC 1 + 2).
3. Controlled "hybrid vs pure recurrence" run on hybrid architecture itself (PoC 5).

All experiments use the existing clean probes and real gold infrastructure. No major new infrastructure required for the first wave of PoCs.

This structured verification of the necessary conditions is now treated as the main "most valuable work" track.

**Raw Intelligence sharpening**: The reasoning-specific version of these conditions (with emphasis on no-retrieval causal latent computation) is defined in the 2026-06 SSOT [Raw Intelligence / Actual Reasoning Necessary Conditions](../docs/wiki/decisions/raw-intelligence-necessary-conditions-2026-06.md). The two documents are intentionally aligned (RI-1..RI-7 map to the 7 S2 conditions).

Detailed execution plan for completing the raw intelligence conditions on current hybrid infrastructure: [RI Raw Intelligence PoC Execution Plan](../docs/roadmaps/RI_Raw_Intelligence_PoC_Execution_Plan_2026-06.md).