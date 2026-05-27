# S2 Hybrid Baseline — 100 Step Validation Report (2026-06-01)

**Backbone**: OneBodyParallelHybridBlock (recurrence: TorchGatedDeltaNet2MixerV2 / official GDN2 preference, attention: official FLA MLA, vector gated fusion v0.2)  
**S1.1 Decisions**: Fully applied  
**Rehearsal Logic**: S1.5 improved faithful 5.56 simulation (scheduled decay 0.40→0.04, gold injection modulated by decay, attractor protection 0.7)  
**Metric**: Clean `pure_stochastic_contribution` (with-noise vs without-noise from identical state)

## Results

### Full 5.56 Recipe (100 steps)
- Pure stochastic effect: **1.32 ~ 1.46** throughout (very stable)
- Final (step 100): **1.3984**
- State robustness probe: **1.000** (perfect)
- Official MLA loaded and used without issues

### Stochastic Breadth Ablation Zero (100 steps)
- Pure stochastic effect: **exactly 0.0000** for the entire 100 steps
- State robustness: **1.000**
- Contract: **Perfect** — turning off stochastic breadth produces zero effect on the final state, as required by Reverse I→G→A.

## Interpretation for S2

This 100-step data constitutes a high-quality, reproducible baseline for the **new hybrid backbone** under 5.56-style curriculum dynamics.

Key strengths:
- Stochastic breadth remains strongly causal even at 100 steps.
- The effect does not decay or collapse over long horizon (unlike some earlier prototypes).
- Ablation contract is clean (zero arm stays at 0.0000).

This is now ready to be compared against historical 5.56 gold recipe runs from the evidence package (`docs/5.56_Promotion_Gate_Evidence_2026-05-30/`) once we have matched-length checkpoints and run the identical clean probe on them.

**Next for S2**: 
- Extract or re-run old backbone 100-step equivalents.
- Run the same clean metric suite.
- Produce direct delta table (hybrid vs old 5.56).

This report + the 100-step artifacts from the improved prototype are the current S2 foundation on the candidate (hybrid) side.
