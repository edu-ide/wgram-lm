# First Explicit Attractor Solver Substrate Diagnostic (June 2026)

**Status**: Ready to run (prototype + wiring script complete)

**Strategic Context (Densing Law Framing)**: This diagnostic is the minimal instrument for measuring *Inference Densing* gains from the Section 7 substrate (see main wiki Section 7 "Densing Law Framing of the Overhaul" and 7.1 Risks). Success is ultimately defined not only by raw accuracy but by improved performance per inference FLOP (solver steps / effective recurrence depth vs. answer quality). See also the IKP nuance (arXiv:2604.24827) on procedural vs. factual capacity.

**Goal**: Smallest falsifiable 20–50 step diagnostic for the Section 7 substrate (Proposal Engine + Dedicated AttractorSolverModule + Parcae stability + SOT + Equilibrium Internalization) before promoting to full native 72 RI-1 measurement.

## Recommended First Command (copy-paste)

```bash
python scripts/diag_explicit_attractor_solver_20step.py \
  --steps 20 \
  --sot_segment_length 5 \
  --attractor_solver_weight 0.15 \
  --internalization_weight 0.12 \
  --out_dir checkpoints/diag_attractor_solver_20step \
  --log_every 2
```

Expected healthy signals (first 20 steps):
- `solver_res` monotonically or mostly decreasing (Parcae + persistent injection working)
- `int` (internalization) beginning to show non-zero and decreasing trend once the proposal projection is replaced by the real hybrid citizen
- No NaN / loss explosion

## Minimal Ablation Matrix (Priority Order for First Wave)

Run each for 20–40 steps. Log solver_res, internalization_loss, primary_on_equilibrium, total.

| ID | Command delta | What it tests | Expected diagnostic value | Priority |
|----|---------------|---------------|---------------------------|----------|
| A0 | (baseline above) | Full stack (Parcae + SOT h=5 + int_w=0.12) | Reference curve | 1 |
| A1 | `--sot_segment_length 3` | Shorter SOT segments (more frequent opt steps) | Does more frequent landscape shaping help or hurt early internalization? | 2 |
| A2 | `--sot_segment_length 7` | Longer segments | Closer to classic unroll behavior | 2 |
| A3 | `--attractor_solver_weight 0.05` | Very light solver pressure | Does the solver still refine when the auxiliary is weak? | 3 |
| A4 | `--attractor_solver_weight 0.25` | Strong solver pressure | Early sign of loss interference (risk #3) | 3 |
| A5 | `--internalization_weight 0.05` | Weak internalization curriculum | How much does the "drop solver at inference" promise depend on strong L_int? | 1 |
| A6 | `--internalization_weight 0.20` | Strong internalization | Risk of proposal collapse or gradient conflict | 4 |
| A7 | (add `--parcae_negative_diag_enabled false` after wiring) | No Parcae stability | Does residual explode or training become unstable? (directly tests Parcae contribution) | 1 (critical) |
| A8 | `--ri_ni_scale 0.0` | No RI/NI (EqR basin shaping off) | Spurious attractor formation speed (risk #2) | 2 |

## Integration Roadmap (after clean 20-step)

1. Confirm residual ↓ + internalization starting to move in the minimal harness.
2. Replace the toy `proposal_proj` with a real call to the existing `OneBodyParallelHybridBlock + BrainMimeticTripleMemory` (reuse the `_hybrid_forward_only` pattern from the main trainer).
3. Wire the equilibrium output into the final LM head path (or add it as a parallel strong loss term) inside the main training loop.
4. Run the same 20-step command with `--brain_triple_memory --internal_fast_recurrent --use_explicit_attractor_solver` using the real proposal engine.
5. If A7 (no Parcae) shows clear instability → Parcae negative diagonal is validated as necessary.
6. Only after the above: promote to full native batched 72 heldout RI-1 measurement under the same strict-B + Principle Gate contract.

## Current Known Early Observation (from first 20-step runs)

In the ultra-minimal harness (toy linear proposal), `internalization_loss` stayed 0.0 because proposal and equilibrium are too close by construction. This is actually **useful data** — it previews risk #1 ("Equilibrium Internalization Fails to Materialize") when the proposal engine is already strong.

When the real rich hybrid citizen (FastGated + TripleMemory + ChunkedSlow) becomes the proposal engine, we expect `int_loss` to become meaningful and then (hopefully) decrease over training. This must be the first thing measured in the next wiring iteration.

All numbers and failure modes will be appended here after the first real rich-proposal runs.

**Long-term Direction Note (Diffusion-Style Attractor Iteration)**

See main wiki Section 7 for the full description of the long-term "Proposal as noisy latent + Solver as learned denoiser" direction.

First small experiments to consider:
- Add controlled Gaussian (and later structured) noise to the proposal before solver steps.
- Implement simple linear/cosine noise schedule wrapper.
- Measure whether explicit noise + timestep conditioning improves:
  - Recovery from bad/spurious initial proposals
  - Final equilibrium quality
  - Inference Densing curve (quality vs. total solver steps)
- Compare against current plain attractor solver baseline.

This direction is deliberately exploratory and should only be pursued after the near-term and medium-term improvements (selective looping, curriculum internalization, densing metrics) show stable signals.
