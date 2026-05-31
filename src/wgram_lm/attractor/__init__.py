"""
Attractor substrate components for the June 2026 Fundamental Overhaul (Solve-the-Loop + Parcae + EqR SOT).

This package implements the explicit Proposal Engine + Dedicated Attractor Solver separation
described in Section 7 of brain_attractor_centric_recurrent_architecture_2026.md.

Design goals (non-negotiable):
- Persistent proposal (y0) injection at every solver step (Attractor Models paper).
- Stability via Parcae-style negative-diagonal parameterization on the solver operator.
- SOT (Segmented Online Training) + RI/NI as first-class training primitives (EqR).
- Equilibrium Internalization as a strong, first-class loss (drives proposal quality).
- Clean contract with existing InferenceState + BrainMimeticTripleMemory (slow context).
- 100% one-body causal path + Principle Gate compatible when wired.
- Full ablatability (solver_off, internalization_off, negative_diag_off, sot_off, etc.).
"""

from .attractor_solver import (
    AttractorSolverModule,
    ProposalEngineAdapter,
    EquilibriumInternalizationLoss,
    SOTSegmentedSolverTrainer,
    make_attractor_solver,
    solve_fixed_point_with_persistent_injection,
)

__all__ = [
    "AttractorSolverModule",
    "ProposalEngineAdapter",
    "EquilibriumInternalizationLoss",
    "SOTSegmentedSolverTrainer",
    "make_attractor_solver",
    "solve_fixed_point_with_persistent_injection",
]
