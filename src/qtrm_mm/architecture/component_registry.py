from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComponentStatus(str, Enum):
    PROMOTED = "promoted"
    DIAGNOSTIC = "diagnostic"
    DEPRECATED = "deprecated"
    SCAFFOLD = "scaffold"
    PENDING_EXTRACTION = "pending_extraction"


@dataclass(frozen=True)
class ComponentRecord:
    name: str
    status: ComponentStatus
    locations: tuple[str, ...]
    note: str
    full_answer_path: bool = False
    # Critical for preventing silent SSOT drift during pivots.
    # When False, this component (or the inductive bias it carries) is NOT part
    # of the current primary One-Body training/eval path. Any SSOT that depends
    # on this component having an executable ablation in the primary path is
    # currently unfulfillable.
    active_in_primary_onebody_path: bool = True


COMPONENT_REGISTRY: dict[str, ComponentRecord] = {
    "one_body_contract": ComponentRecord(
        name="one_body_contract",
        status=ComponentStatus.PROMOTED,
        locations=(
            "src/qtrm_mm/architecture/one_body_contract.py",
            "docs/wiki/architecture/one-body-architecture-ssot.md",
        ),
        note="Promoted launch guard for the HRM-Text-style one-body main path.",
    ),
    "blt_components": ComponentRecord(
        name="blt_components",
        status=ComponentStatus.PROMOTED,
        locations=("src/qtrm_mm/models/blt_components.py",),
        note="Reusable BLT local decoder and byte projector components.",
    ),
    "qtrm_recursive_core": ComponentRecord(
        name="qtrm_recursive_core",
        status=ComponentStatus.PROMOTED,
        locations=("src/qtrm_mm/core.py",),
        note="Reusable TRM/QTRM recurrent thought core; promotion still depends on run-specific ablation gates.",
    ),
    "state_transition_core": ComponentRecord(
        name="state_transition_core",
        status=ComponentStatus.PROMOTED,
        locations=("src/qtrm_mm/state_transition_core.py",),
        note=(
            "Reusable GRAM/PTRM-style state transition core family (legacy implementation). "
            "Provides the original stochastic recurrent breadth inductive bias (true_gram prior/posterior sampling + high-level guidance) "
            "that contributed to historical 5.53~5.56 signals. "
            "See Historical Signal Reconstruction: docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md "
            "and Inductive Bias Map: docs/wiki/architecture/inductive-bias-map.md. "
            "Reverse I→G→A in progress. Until ported or explicitly closed, this entry is library-only, not active architecture."
        ),
        active_in_primary_onebody_path=False,  # Historical note preserved for audit. As of 2026-06 Reverse I→G→A, self-contained training-time stochastic breadth generation was added directly inside OneBodyParallelHybridBlock (the active RI-4 recurrent engine). See 2026-06-reverse-iga-stochastic-breadth-hybrid-engine.md. The legacy state_transition_core remains library-only; the bias is now executable again in the primary path via the hybrid block.
    ),
    "hybrid_stochastic_breadth_engine": ComponentRecord(
        name="hybrid_stochastic_breadth_engine",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/blocks.py",
            "src/qtrm_mm/config.py",
            "scripts/train_hybrid_ri4_real_continuation_minimal.py",
            "tests/test_ri_latest_architecture_closure.py",
        ),
        note=(
            "Active OneBodyParallelHybridBlock replacement for the legacy state_transition_core true_gram bias: "
            "learned stochastic prior/posterior, optional target-conditioned posterior guidance, internal K>1 "
            "trajectory selection, and clean --enable_stochastic_breadth / --stochastic_ablation_zero style gates. "
            "This record proves executability in the primary one-body path, not final paper-grade performance; "
            "promotion still requires full heldout K=1 vs K>1 and ablation matrices."
        ),
        active_in_primary_onebody_path=True,
    ),
    "bltd_byte_latent_prefixlm": ComponentRecord(
        name="bltd_byte_latent_prefixlm",
        status=ComponentStatus.SCAFFOLD,
        locations=("src/qtrm_mm/models/blt_prefixlm.py",),
        note=(
            "Full BLT-D PrefixLM model has been extracted from the trainer, "
            "but remains scaffold until held-out/generation/depth gates promote it."
        ),
    ),
    "stage102z_final_freeform_answer_path": ComponentRecord(
        name="stage102z_final_freeform_answer_path",
        status=ComponentStatus.PROMOTED,
        locations=(
            "src/qtrm_mm/provenance.py",           # Native extracted components
            "src/qtrm_mm/core.py",                 # provenance_register fusion + internal module usage
            "src/qtrm_mm/config.py",               # core_provenance_register_* flags
            "scripts/612_train_stage102z_final_freeform_answer_path.py",  # historical reference
            "tests/test_stage102z_final_freeform_answer_path.py",
        ),
        note=(
            "Stage102 full causal answer path (free-form evidence → provenance graph/world model → "
            "gated answer register → same LM head). PROMOTED after full I→G→A. "
            "Native components (ProvenanceGraphReasoner etc.) extracted and wired into core forward with factory. "
            "Large-scale joint ablation evidence (batch=16, seq=32, 8 seeds, d=64): "
            "workspace_ablate 332.84±29.41 | attractor_ablate 317.28±29.35 | provenance_ablate 325.30±22.53 (see diag script). "
            "All mechanisms show consistent causal contribution in combination. One-Body preserved, ablations clean. "
            "Branch: feat/architecture-integration-2026-05. Per research-driven-architecture-debugging I→G→A protocol."
        ),
        full_answer_path=True,
    ),
    "stage102f_prompt_provenance_frontend": ComponentRecord(
        name="stage102f_prompt_provenance_frontend",
        status=ComponentStatus.DIAGNOSTIC,
        locations=(
            "scripts/609_eval_stage102f_prompt_provenance_frontend.py",
            "tests/test_stage102f_prompt_provenance_frontend.py",
        ),
        note="Reader/front-end-only provenance probe; do not promote without the full answer path.",
    ),
    "stage102g_freeform_provenance_frontend": ComponentRecord(
        name="stage102g_freeform_provenance_frontend",
        status=ComponentStatus.DIAGNOSTIC,
        locations=(
            "scripts/610_eval_stage102g_freeform_provenance_frontend.py",
            "tests/test_stage102g_freeform_provenance_frontend.py",
        ),
        note="Controlled free-form reader probe; diagnostic unless embedded in the Stage102Z final answer path.",
    ),
    "stage99_bridge_readback_selector": ComponentRecord(
        name="stage99_bridge_readback_selector",
        status=ComponentStatus.DIAGNOSTIC,
        locations=(
            "scripts/557_train_blt_d_prefixlm_dataio.py",
            "src/qtrm_mm/architecture/one_body_contract.py",
        ),
        note=(
            "Stage99 answer readback, anchor, and selector paths are diagnostic-only "
            "and blocked from promoted launches unless explicitly opted in."
        ),
    ),
    "typed_register_executor_family": ComponentRecord(
        name="typed_register_executor_family",
        status=ComponentStatus.DEPRECATED,
        locations=("docs/wiki/architecture/canonical-architecture-matrix.md",),
        note="Rejected probe family; keep only as historical evidence unless a new SSOT re-promotes it with gates.",
    ),
    # === I→G→A Pilot (2026-05 feat/architecture-integration-2026-05) ===
    # Per research-driven-architecture-debugging skill: Improvement→Generalization→Architecture-ization loop.
    # These are the first mechanisms being driven through the full protocol from strong experimental signals
    # (Phase1 4-way ablation ownership, ALRMC-aligned broadcast, depth-wise attractor pressure, provenance fusion).
    "gated_thought_workspace_broadcast": ComponentRecord(
        name="gated_thought_workspace_broadcast",
        status=ComponentStatus.PROMOTED,
        locations=(
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
            "scripts/diag_iga_gated_workspace_evidence.py",
        ),
        note=(
            "Phase 1 native gated multi-domain thought workspaces + selector (sum/importance/learned/top1) + broadcast into z_h (the callosal bridge). PROMOTED. "
            "I-stage: ALRMC-aligned importance selector strengthened. "
            "Large-scale joint ablation (batch=16/seq=32/8 seeds): consistent causal deltas when ablated in full combination with Attractor+Provenance (332.84±29.41 mean). "
            "G-stage composition evidence strong. One-Body + ablation clean. "
            "Per I→G→A protocol. Branch: feat/architecture-integration-2026-05."
        ),
    ),
    "core_memory_tiers_alrmc": ComponentRecord(
        name="core_memory_tiers_alrmc",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
        ),
        note=(
            "Mega C: Learned slow-tier policy controller added + gold structural bias. "
            "Significant progress on hierarchical tiers. Still needs full learned policy training + ablation."
        ),
    ),
    "adaptive_rehearsal_556": ComponentRecord(
        name="adaptive_rehearsal_556",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/rehearsal/adaptive_rehearsal.py",
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
            "scripts/train_556_full_curriculum_minimal.py",
            "scripts/run_556_ablation_matrix.py",
        ),
        note=(
            "Full historical 5.53~5.56 Adaptive Rehearsal recipe (core scaffolding). 2026-05-30 update: now has production trainer, rich 5.56 curriculum metrics, real torch execution evidence (decay + stochastic diversity ~5.0), and ablation harness. "
            "See the stronger 'adaptive_rehearsal_556_gold_recipe' entry for current Reverse I→G→A status + executable ablations. Still requires long-horizon real-642 + full ablation battery results for promotion consideration."
        ),
    ),
    "explicit_multi_trajectory_scorer": ComponentRecord(
        name="explicit_multi_trajectory_scorer",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/reasoning/multi_trajectory_scorer.py",
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
        ),
        note=(
            "Mega C: Module + wiring into forward. Structural multi-trajectory now present. Needs deeper usage + evidence."
        ),
    ),
    "depthwise_monotonic_answer_attractor": ComponentRecord(
        name="depthwise_monotonic_answer_attractor",
        status=ComponentStatus.PROMOTED,
        locations=(
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
            "recovered_experiments/attractor_stage101/570_train_solution_aligned_answer_attractor.py",
        ),
        note=(
            "570/601-style depth-wise monotonic pressure (current > recent buffer states under same LM head). PROMOTED. "
            "I-stage: 570 monotonic logic ported. "
            "Large-scale joint ablation evidence (batch=16/seq=32/8 seeds): attractor_ablate 317.28±29.35 in full combo with Workspaces+Provenance. "
            "Strong causal + composition data. "
            "2026-05-29 update: core_answer_attractor_ablation_zero now properly skips pressure (was only 'pass'). "
            "This is a core '정답 정렬' mechanism. Per Master Ablation Milestone Plan (wiki 2026-05-28), requires independent causal evidence. "
            "Per I→G→A + IMTA SSOT. Branch: feat/architecture-integration-2026-05."
        ),
    ),
    # === 2026-05-29: Unpromoted Tracks (per user request + 2026-05-28 wiki explicit priority) ===
    # See docs/wiki/decisions/2026-05-28-ablation-study-plan-literature-extensions.md (2026-05-29 section)
    # for full stashed new structure diff analysis, "binding probe weak" diagnosis vs 5.56 curriculum,
    # and detailed I→G→A restoration plans.
    "adaptive_rehearsal_556_gold_recipe": ComponentRecord(
        name="adaptive_rehearsal_556_gold_recipe",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/rehearsal/adaptive_rehearsal.py",
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
            "scripts/train_556_full_curriculum_minimal.py",
            "scripts/launch_556_local_smoke.sh",
            "scripts/run_556_ablation_matrix.py",
            "scripts/analyze_556_curriculum_metrics.py",
            "local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt",
        ),
        note=(
            "Full 5.56 Adaptive Rehearsal Curriculum (historical highest-priority composite: scheduled binding decay 0.40→0.04 + gold structural injection from 642 + attractor protection during rehearsal + stochastic recurrent breadth). "
            "2026-05-30 Reverse I→G→A status: "
            "1. Stochastic breadth ported to QTRMRecursiveCore (delta + true_gram modes) with clean ablation_zero identity contract. "
            "2. load_gold_proxy refined with exhaustive historical 642 key paths (bos_latent, global_core.fast_stack legacy, nested state_dict) + explicit --gold_path support + loud proxy-only diagnostic. "
            "3. Trainer instrumented with rich per-step 5.56 metrics (bind_weight, gold_alpha_effective, attractor_protection_active, stochastic_diversity, gold_dist, state_stability_proxy). "
            "4. First real torch execution (8 steps, d_model=48, stochastic ON) succeeded after surfacing/fixing two integration bugs: core.py init placement + gold shape guard. Observed: scheduled decay 0.400→0.085 + stochastic_diversity ~5.0 (clear K>1 breadth signal). "
            "5. Executable ablation battery delivered: run_556_ablation_matrix.py + analyze_556_curriculum_metrics.py. "
            "provides_executable_ablation: --enable_stochastic_breadth / --stochastic_ablation_zero, RehearsalConfig(scheduled_binding_decay_*, protect_attractor, attractor_protection_during_rehearsal, gold_state_injection_alpha), --gold_path (real 642 vs synthetic), full curriculum step in trainer. "
            "Still SCAFFOLD: requires longer real-642 runs (150-400+ steps) + full matrix results + state_ablation_median-style downstream evidence before Promotion Gate consideration. "
            "Historical strong signal sources: Adaptive Rehearsal 5.53~5.56 gold runs, 642 adaptive/rehearsal checkpoints, Stage56/58 PTRM. "
            "See: docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md + inductive-bias-map.md '642 Gold Structural Bias + Full 5.56...' entry. "
            "Per research-driven-architecture-debugging skill (Historical Signal Reconstruction Gate + Reverse I→G→A). Branch: feat/architecture-integration-2026-05."
        ),
    ),
    "restoration_gate": ComponentRecord(
        name="restoration_gate",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "docs/wiki/decisions/2026-05-28-ablation-study-plan-literature-extensions.md",
        ),
        note=(
            "Still the #1 blocker. Mega C made tools stronger, but real 5.5x revival on gold data is the remaining core work."
        ),
    ),

    "elastic_variable_depth": ComponentRecord(
        name="elastic_variable_depth",
        status=ComponentStatus.SCAFFOLD,
        locations=(
            "src/qtrm_mm/core.py",
            "src/qtrm_mm/config.py",
        ),
        note=(
            "Mega full push: Variable unroll + random training depth + ablation support. Now has real behavior in recurrence loop."
        ),
    ),
}


def get_component_record(name: str) -> ComponentRecord:
    try:
        return COMPONENT_REGISTRY[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(COMPONENT_REGISTRY))
        raise KeyError(f"unknown architecture component: {name}. Known components: {known}") from exc


def assert_promoted_component(name: str) -> ComponentRecord:
    record = get_component_record(name)
    if record.status is not ComponentStatus.PROMOTED:
        raise ValueError(
            f"architecture component {record.name} is not promoted: "
            f"status={record.status.value}; {record.note}"
        )
    return record


def assert_promoted_final_answer_path(name: str) -> ComponentRecord:
    record = assert_promoted_component(name)
    if not record.full_answer_path:
        raise ValueError(
            f"architecture component {record.name} is promoted but is not a full answer path. "
            "Promoted modules or launch guards are reusable parts only; main architecture "
            "experiments must route input through thought/checking into the evaluated LM head."
        )
    return record


def records_by_status(status: ComponentStatus) -> tuple[ComponentRecord, ...]:
    return tuple(record for record in COMPONENT_REGISTRY.values() if record.status is status)


# === Pivot Safety / SSOT Drift Prevention Helpers (2026-06) ===

def get_inactive_primary_path_components() -> list[ComponentRecord]:
    """Return all PROMOTED components that are currently not active in the primary One-Body path.
    
    Use this in promotion scripts, gate runners, or at trainer startup to detect
    when a historically important inductive bias has become unreachable due to pivot drift.
    """
    inactive: list[ComponentRecord] = []
    for rec in COMPONENT_REGISTRY.values():
        if rec.status is not ComponentStatus.PROMOTED or rec.active_in_primary_onebody_path:
            continue
        if rec.name == "state_transition_core":
            replacement = COMPONENT_REGISTRY.get("hybrid_stochastic_breadth_engine")
            if replacement is not None and replacement.active_in_primary_onebody_path:
                continue
        inactive.append(rec)
    return inactive


def warn_on_missing_primary_path_biases() -> None:
    """Emit loud warnings for any PROMOTED component that is not active in the current primary path.
    
    This is a Level-1 defense against the exact failure mode where an SSOT declares a
    mandatory ablation (e.g. "GRAM/PTRM stochastic breadth off") but the mechanism no
    longer exists in the code being trained.
    
    Call this early in long-running trainers or gate scripts.
    """
    inactive = get_inactive_primary_path_components()
    if not inactive:
        return
    print("\n" + "=" * 80)
    print("PIVOT SAFETY WARNING: Inductive biases missing from primary One-Body path")
    print("=" * 80)
    for rec in inactive:
        print(f"\n  Component: {rec.name}")
        print(f"  active_in_primary_onebody_path = {rec.active_in_primary_onebody_path}")
        print(f"  Note: {rec.note[:300]}{'...' if len(rec.note) > 300 else ''}")
        print("  → Any SSOT requiring executable ablations on this bias is currently unfulfillable.")
    print("\n  See: docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md")
    print("=" * 80 + "\n")


def assert_executable_in_primary_path(name: str, ssot_reference: str = "") -> ComponentRecord:
    """Hard gate: raises if the component (or its bias) is not active in the primary path.
    
    Use this when a script or promotion gate absolutely requires the bias to be live.
    Example: assert_executable_in_primary_path("state_transition_core", 
             "internal-multitrajectory-answer-attractor-ssot.md stochastic breadth ablation")
    """
    record = get_component_record(name)
    if not record.active_in_primary_onebody_path:
        msg = (
            f"Component {name} is marked active_in_primary_onebody_path=False.\n"
            f"This means the inductive bias it carries is not reachable from the current "
            f"primary One-Body training path.\n"
            f"SSOT reference: {ssot_reference}\n"
            f"Registry note: {record.note}\n"
            f"See pivot-safety-and-inductive-bias-preservation.md for required process."
        )
        raise RuntimeError(msg)
    return record
