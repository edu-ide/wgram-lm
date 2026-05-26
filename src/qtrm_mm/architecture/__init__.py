"""Architecture-level contracts shared by trainers and eval scripts."""

from .component_registry import (
    COMPONENT_REGISTRY,
    ComponentRecord,
    ComponentStatus,
    assert_promoted_final_answer_path,
    assert_promoted_component,
    get_component_record,
    records_by_status,
)
from .one_body_contract import (
    BridgeContractFields,
    collect_bridge_contract_fields,
    validate_one_body_architecture_contract,
)

__all__ = [
    "COMPONENT_REGISTRY",
    "BridgeContractFields",
    "ComponentRecord",
    "ComponentStatus",
    "assert_promoted_final_answer_path",
    "assert_promoted_component",
    "collect_bridge_contract_fields",
    "get_component_record",
    "records_by_status",
    "validate_one_body_architecture_contract",
]
