import importlib.util
from pathlib import Path

import torch


def _load_stage523():
    path = Path(__file__).resolve().parents[1] / "scripts" / "523_train_state_text_speaker.py"
    spec = importlib.util.spec_from_file_location("stage523_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyTokenizer:
    def __call__(self, prompts, *, return_tensors, padding, truncation, max_length, add_special_tokens):
        assert return_tensors == "pt"
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }


class CapturingQTRM:
    def __init__(self):
        self.operation_ids = None
        self.operation_arg_ids = None
        self.source_numeric_features = None
        self.source_numeric_feature_mask = None

    def __call__(
        self,
        input_ids,
        *,
        attention_mask=None,
        n_steps=None,
        return_dict=True,
        operation_ids=None,
        operation_arg_ids=None,
        source_numeric_features=None,
        source_numeric_feature_mask=None,
    ):
        self.operation_ids = operation_ids.detach().cpu() if operation_ids is not None else None
        self.operation_arg_ids = operation_arg_ids.detach().cpu() if operation_arg_ids is not None else None
        self.source_numeric_features = (
            source_numeric_features.detach().cpu() if source_numeric_features is not None else None
        )
        self.source_numeric_feature_mask = (
            source_numeric_feature_mask.detach().cpu() if source_numeric_feature_mask is not None else None
        )
        batch = int(input_ids.size(0))
        steps = int(n_steps)
        return {
            "qtrm_readout_state": torch.zeros(batch, 4),
            "qtrm_core_step_states": torch.zeros(batch, steps + 1, 4),
            "qtrm_workspace": torch.zeros(batch, input_ids.size(1), 4),
            "qtrm_working_register_trajectory": torch.zeros(batch, steps + 1, 2, 4),
            "qtrm_typed_value_register_trajectory": torch.zeros(batch, steps + 1, 4, 4),
        }


def test_trace_operation_ids_are_compacted_and_padded_with_hold():
    module = _load_stage523()
    rows = [
        {
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "4010"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "8020"},
                {"depth": 4, "operation": "subtract_offset", "state_text": "8017"},
                {"depth": 8, "operation": "hold_final", "state_text": "8017"},
            ]
        },
        {
            "solver_trace": [
                {"depth": 1, "operation": "filter_even", "state_text": "4004,4002"},
                {"depth": 2, "operation": "double_filtered", "state_text": "8008,8004"},
            ]
        },
    ]

    operation_ids, operation_arg_ids = module.trace_operation_tensors(rows, n_steps=5, device=torch.device("cpu"))

    assert operation_ids.tolist()[0] == [
        module.TRACE_OPERATION_TO_ID["add_operands"],
        module.TRACE_OPERATION_TO_ID["multiply_sum"],
        module.TRACE_OPERATION_TO_ID["subtract_offset"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
    ]
    assert operation_ids.tolist()[1] == [
        module.TRACE_OPERATION_TO_ID["filter_even"],
        module.TRACE_OPERATION_TO_ID["double_filtered"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
    ]
    assert operation_arg_ids.shape == operation_ids.shape
    assert operation_arg_ids.sum().item() == 0


def test_thought_context_passes_trace_operations_to_qtrm_forward():
    module = _load_stage523()
    qtrm = CapturingQTRM()
    rows = [
        {
            "prompt": "Question: Compute ((4007 + 3) * 2) - 3. Answer:",
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "4010"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "8020"},
                {"depth": 4, "operation": "subtract_offset", "state_text": "8017"},
            ],
        }
    ]

    module.thought_context_for_batch(
        qtrm,
        TinyTokenizer(),
        rows,
        max_length=16,
        n_steps=4,
        device=torch.device("cpu"),
        detach=True,
        condition_on_trace_operations=True,
    )

    assert qtrm.operation_ids is not None
    assert qtrm.operation_ids.tolist()[0] == [
        module.TRACE_OPERATION_TO_ID["add_operands"],
        module.TRACE_OPERATION_TO_ID["multiply_sum"],
        module.TRACE_OPERATION_TO_ID["subtract_offset"],
        module.TRACE_OPERATION_TO_ID["hold_final"],
    ]


def test_source_number_features_extract_visible_numbers_without_solving():
    module = _load_stage523()
    rows = [
        {"question": "Compute ((4007 + 3) * 2) - 3."},
        {"question": "From the list [4001, 4004, 4002], keep only even numbers."},
    ]

    features, mask = module.source_number_feature_tensors(
        rows,
        max_slots=4,
        feature_dim=18,
        value_scale=10000.0,
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert torch.allclose(features[0, :, 0], torch.tensor([0.4007, 0.0003, 0.0002, 0.0003]))
    assert features[0, :, 1].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.allclose(features[1, :3, 0], torch.tensor([0.4001, 0.4004, 0.4002]))
    assert features[1, :3, 1].tolist() == [0.0, 1.0, 1.0]


def test_thought_context_passes_source_number_features_to_qtrm_forward():
    module = _load_stage523()
    qtrm = CapturingQTRM()
    rows = [
        {
            "prompt": "Question: Compute ((4007 + 3) * 2) - 3. Answer:",
            "solver_trace": [{"depth": 1, "operation": "add_operands", "state_text": "4010"}],
        }
    ]

    module.thought_context_for_batch(
        qtrm,
        TinyTokenizer(),
        rows,
        max_length=16,
        n_steps=2,
        device=torch.device("cpu"),
        detach=True,
        use_source_number_slots=True,
        source_number_slots=4,
        source_number_feature_dim=18,
        source_number_value_scale=10000.0,
    )

    assert qtrm.source_numeric_features is not None
    assert qtrm.source_numeric_features.shape == (1, 4, 18)
    assert qtrm.source_numeric_feature_mask.tolist() == [[1, 1, 1, 1]]


def test_thought_context_returns_typed_value_register_trajectory():
    module = _load_stage523()
    qtrm = CapturingQTRM()
    rows = [{"prompt": "Question: Compute ((4007 + 3) * 2) - 3. Answer:"}]

    context = module.thought_context_for_batch(
        qtrm,
        TinyTokenizer(),
        rows,
        max_length=16,
        n_steps=2,
        device=torch.device("cpu"),
        detach=True,
    )

    assert context["typed_value_register_trajectory"] is not None
    assert context["typed_value_register_trajectory"].shape == (1, 3, 4, 4)
