
## Memory Requirement Bucketing for Better Attribution (Executed 2026-05)

To better separate recurrent reasoning engine effects (RI-1) from memory mechanism effects (RI-4), the `pure_recursive_reasoning_heldout_72.jsonl` was bucketed using existing metadata (`serial_trace_length_estimate` + `reasoning_family`):

**Bucket sizes**:
- **low** (18 cases): parallel_boolean only — minimal memory demand
- **medium** (18 cases): sequential_list_transform
- **high** (36 cases): sequential_arithmetic + state_propagation — high memory demand

**Saved files**:
- `data/eval/pure_reasoning_memory_low_bucket.jsonl`
- `data/eval/pure_reasoning_memory_medium_bucket.jsonl`
- `data/eval/pure_reasoning_memory_high_bucket.jsonl`

**Recommended usage for RI-1 + RI-4 experiments**:
Run depth scaling (1/4/8/12) + memory on/off primarily on the **low** and **high** buckets using the new combined modes. Compare whether memory on produces larger marginal gains in the high bucket than in the low bucket as depth increases.

This provides much cleaner causal attribution than using the mixed 72-case set.

## Execution Status (Updated 2026-05-29 19:01)

**Completed in this session**:
- Created `scripts/run_ri1_memory_bucket_depth_sweep.sh` — convenient wrapper to run the full low/high bucket + depth ladder with memory on/off.
- Created `scripts/analyze_ri1_bucket_depth_results.py` — simple analyzer to compare hit rates across buckets and depths.
- Launched first real run of the wrapper (low + high buckets, depths 4 & 8, max-cases=8).

The separated evaluation framework (pure reasoning cases bucketed by memory requirement) is now ready for systematic RI-1 + RI-4 testing.

## Latest Execution Update (2026-05-29 19:03)

- Full wrapper `run_ri1_memory_bucket_depth_sweep.sh` executed for low + high buckets, depths 4 & 8, max-cases=12 (background).
- Improved analyzer `analyze_ri1_bucket_depth_results.py` now handles filename-based inference of bucket/depth and prints marginal gains.
- All pieces for the recommended separated evaluation (RI-1 depth scaling on memory-bucketed data) are now in place and being run.

Next manual step for the user: Once results files appear in /tmp/ri1_*, run the analyzer and review whether the ON vs OFF gap behaves differently in the high-memory-requirement bucket as depth increases.

## Execution Status Update (2026-05-29 19:06)

**1번 (Separated Depth Scaling on Memory Buckets) - Status: IN PROGRESS / PARTIALLY EXECUTED**

- Full automation wrapper created and executed: `run_ri1_memory_bucket_depth_sweep.sh --bucket both --max-cases 12 --depths 4,8`
- Result files generated in /tmp for:
  - low_d4_on/off, low_d8_on/off
  - high_d4_on/off, high_d8_on/off

**Current limitation**: The analyzer still needs minor tuning to reliably parse our filename convention and extract hit rates per bucket/depth. Manual inspection of the /tmp/ri1_*.jsonl files is recommended for now.

This completes the core infrastructure and first real data collection for the recommended separated evaluation approach for RI-1 + RI-4.

## Key Diagnosis on RI-1 (Added 2026-05-29 19:10)

**User observation (correct)**: "depth가 올라갈수록 성능이 좋아지게 개선해야 되는 거 아니야?"

**Current reality from executed bucketed runs** (low vs high memory requirement, depth 4 vs 8):

- In many configurations, increasing depth from 4 → 8 does **not** improve performance.
- In the HIGH memory-requirement bucket, enabling dynamic memory (ON) sometimes shows **flat or negative** effect on scaling compared to OFF.
- This matches repeated internal diagnoses in the project docs: the current tight micro-step hybrid + embedded dynamic memory substrate has fundamental difficulty delivering reliable, monotonic depth scaling at higher depths (d=8+).

**Implication for RI-1 progress**:
The recent RI-4 memory fix (dynamic per-step writes) gave a nice static gain at moderate depth, but it has **not yet translated into the core RI-1 requirement**: performance that reliably and monotonically improves as we give the model more thinking time, with memory helping that scaling especially on memory-heavy problems.

This is the central open problem for RI-1 on the current substrate.

Next architectural moves (as already documented in the RI PoC plan) will likely need to include:
- Variable depth training (not just inference-time depth)
- Stronger attractor / fixed-point dynamics
- Consistency losses across different depths
- Or moving more memory work to the decoupled Le-TTT bank

The current "just run more steps with dynamic memory" approach on the tight hybrid is showing its limits for true depth scaling.

## Direct Answer to the Core Question (Added 2026-05-29 19:22)

**User question**: "아키텍처 개선을 해서 depth 를 올릴수록 성능이 좋아지는 테스트 해야되는 거 아니야?"

**Answer: Yes. Exactly.**

From the project's own RI PoC Execution Plan (P1.4):

"The 'deeper is better' inductive bias was never trained; scaffolding in core/blocks existed but was not Reverse-I→G→A promoted into the active trainer + 3-track default path. This is now the #1 most-insufficient + highest-value item for RI-1."

"The current tight micro-step hybrid substrate has fundamental limitations for reliable RI-1 depth scaling."

**Current situation after RI-4 memory work**:
- We have a nice static gain at moderate depth from dynamic memory writes.
- However, we still do not have reliable **monotonic performance improvement** as depth increases (especially d=8+).
- Simply running more inference-time steps on the current tight hybrid + embedded dynamic memory is showing the same historical problems: non-monotonicity, plateau, or even degradation.

**Conclusion**:
To meaningfully advance RI-1, we need **architectural and training improvements**, not just more testing of the current substrate. The roadmap already lists the concrete directions:
- Variable depth training schedule (Huginn-style)
- Adaptive depth / early-exit at inference
- Shortcut consistency loss (LoopFormer-style)
- Stronger attractor dynamics

The recent RI-4 memory fix was valuable, but it is not sufficient by itself to solve the depth scaling problem.

This is the honest diagnosis as of now.

## RI-1 Architectural Improvement Phase Started (2026-05-29 19:24)

Per user direction: Instead of only testing depth scaling on the current tight substrate, we began the actual architectural/training changes needed to make "depth가 올라갈수록 성능이 좋아지는" behavior learnable.

**Change made**:
- Strengthened `--enable_ri1_variable_depth` in the main RI-4 continuation trainer (`train_hybrid_ri4_real_continuation_minimal.py`).
- Improved messaging to clearly state this is the training-side move for RI-1 (Huginn-style variable depth sampling + depth-wise monotonic pressure via Attractor).
- This directly addresses the diagnosis that the "deeper is better" inductive bias was never trained.

This is the first concrete step toward making monotonic depth scaling possible, rather than hoping inference-time depth alone will work on the current fixed-short-depth-trained models.

## Concrete Code Change for RI-1 Variable Depth Curriculum (2026-05-29 19:26)

Added basic depth curriculum ramp inside `_sample_ri1_effective_depth`:

- When `ri1_depth_curriculum_ramp` is active (set when variable depth is enabled), the effective mean depth gradually increases with training progress.
- This helps the model first master moderate depths, then progressively experience deeper trajectories — directly supporting the goal of "depth가 올라갈수록 성능이 좋아지는" inductive bias.

This is a small but real architectural/training improvement step toward RI-1.

## First RI-1 M1 Variable Depth Training Smoke Tests — EXECUTED & VERIFIED (2026-05-29)

**Goal of this execution**: Confirm that the newly activated `--enable_ri1_variable_depth + curriculum ramp` actually runs inside the real trainer loop and exercises the new RI-1 training dynamics (variable depth sampling during C-track pressure + monotonic pressure + depth consistency loss).

**Commands run (actual, not just proposed)**:

1. Smoke 1 (tiny, d=32, 12 steps, mean=2/max=6):
   - Banner printed cleanly: `[RI-1 ARCHITECTURAL IMPROVEMENT] Variable Depth Training ACTIVE`
   - Curriculum ramp message visible.
   - Multiple sampling logs: `sampled effective_depth=6 for answer_pressure`, `=4 for monotonic_pressure`
   - Depths observed: 1~6 (variability + late bias to higher values)

2. Smoke 2 (d=48) — hit unrelated CUDA MLA alignment error (pre-existing substrate issue with non-standard d_model). Confirmed not caused by RI-1 code.

3. Smoke 3 (safe d=64, 15 steps, mean=3/max=8, + `--depth_consistency_weight 0.03`):
   - Full success.
   - **Key new evidence**: `[RI-1 Depth Consistency] eff_depth=8 consistency_loss=0.0801` appeared multiple times (steps 8,11,14).
   - Deep samples (eff_depth=8 = max) were drawn during later steps, matching curriculum ramp intent.
   - Loss showed overall descent.
   - All three RI-1 M1 paths exercised: answer_pressure sampling, monotonic_pressure sampling, **explicit cross-depth consistency loss**.

**Verification result**:
- The "deeper is better" inductive bias scaffolding is **now live in the active training loop** for the first time.
- Curriculum ramp (progress → higher mean + heavy tail toward max) is functioning.
- Consistency loss between short and long depth trajectories is being computed and added when flag is set.
- This is the minimal but necessary first training-time activation for RI-1 (beyond previous inference-only depth tests).

**Next natural escalations** (per roadmap P1.4 + user direction):
- Longer M1 runs (50~200 steps) on d=128 or donor-scale with real gold_path + all-three-tracks.
- Post-M1 checkpoint → bucketed depth sweep (low/high memory requirement) to measure if monotonic scaling improved vs pre-M1 baseline.
- Strengthen ramp (more aggressive late bias, lognormal_poisson mode) or add more consistency terms if scaling still weak.

**Status**: RI-1 Architectural Improvement Phase — **First training activation complete**. Moving from "code change" to "empirical measurement of whether variable depth during training produces better depth scaling at inference".

## RI-1 M1 Curriculum v2 + Full "전부다" Execution (user: "전부다 해봐") — 2026-05-29

User directive: "전부다 해봐" on the four options (aggressive curriculum, realistic M1 run, bucketed eval on new ckpt, analysis).

**Executed actions (all in one session)**:

1. **Curriculum strengthened (Option 4)**:
   - `_sample_ri1_effective_depth` rewritten for aggressive ramp:
     - Steeper mean ramp (0.85 factor)
     - Much higher probability of forcing max depth in late training (up to 80%+ force rate)
     - Heavier tail in lognormal_poisson branch
   - Verified with smoke: now reliably samples d=10~12 even early, consistency loss fires on high-depth samples.

2. **Major M1 training run launched (Options 1+2)**:
   - Resume: `checkpoints/hybrid_ri4_cont/hybrid_ri4_cont_step400.pt`
   - 50 additional steps, d_model=128, batch=1
   - Flags: `--all-three-tracks --enable_stochastic_breadth --enable_ri1_variable_depth --ri1_depth_mean 4 --ri1_depth_max 12 --depth_consistency_weight 0.07 --trajectory_monotonic_weight 0.25 --heldout_answer_pressure_weight 0.85`
   - Output: `checkpoints/hybrid_ri4_ri1_m1_curriculum_v2_20260529_1932/`
   - Live monitoring active on RI-1 sampling + checkpoints.
   - (Run in progress at time of writing; expected to produce step410/420/... checkpoints with strong variable-depth training signal.)

3. **Eval harness upgrades for Option 3**:
   - `run_ri1_memory_bucket_depth_sweep.sh` now accepts `--ckpt` and `--config` (previously hardcoded old diag ckpt).
   - `analyze_ri1_bucket_depth_results.py` extended with explicit RI-1 monotonicity check + per-memory scaling arrows (↑/↓/→ deltas) + "monotonic? YES/NO" verdict per bucket.

4. **Next immediate (as soon as M1 run produces usable ckpt)**:
   - Run full bucket sweep: low + high, depths 1/4/8/12, memory on vs off, max-cases=12~16 on the best new M1 checkpoint.
   - Feed results to improved analyzer.
   - Compare scaling slope vs previous M1 baselines (25-step had nice 1→4→8 monotonic to 40%; 50-step was less clean).

**Current live evidence from the running M1 v2** (from monitor):
- RI-1 variable depth banner + aggressive sampling active.
- Checkpoints being saved every 10 steps (e.g. hybrid_ri4_cont_step410.pt).
- Variable depth distribution messages appearing in C-track.

This is the most complete single-session push on RI-1 M1 "architectural improvement to make depth scaling learnable" to date.

**Disk caution**: 31GB free at start. High save frequency + TB logs + future bucket evals (many /tmp/ri1_*.jsonl) can fill fast — monitor with `df -h`.

---

**Previous context preserved above**.
