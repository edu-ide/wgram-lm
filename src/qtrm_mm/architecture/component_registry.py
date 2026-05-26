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
            "scripts/612_train_stage102z_final_freeform_answer_path.py",
            "tests/test_stage102z_final_freeform_answer_path.py",
            "docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md",
        ),
        note=(
            "Stage102 promoted full causal answer path: free-form evidence text -> "
            "provenance graph/world model -> gated answer register -> same BLT LM head."
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
