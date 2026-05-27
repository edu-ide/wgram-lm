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
        note="Reusable GRAM/PTRM-style state transition core family.",
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
            "Minimal isolated memory tiers + MSA-style sparse rehearsal signal inside recurrent core (Option 2 track). "
            "I→G→A in progress. Currently scaffold; requires full generalization + attractor/provenance composition evidence before promotion consideration."
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
            "Per I→G→A. Branch: feat/architecture-integration-2026-05."
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
