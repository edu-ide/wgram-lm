import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_stage530():
    path = Path(__file__).resolve().parents[1] / "scripts" / "530_train_final_typed_register_answerer.py"
    spec = importlib.util.spec_from_file_location("stage530_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_typed_value_trace_targets_parse_step_values_and_list_slots():
    module = _load_stage530()
    rows = [
        {
            "solver_trace": [
                {"depth": 1, "state_text": "4010"},
                {"depth": 4, "state_text": "8017"},
            ]
        },
        {
            "solver_trace": [
                {"depth": 1, "state_text": "4004,4002"},
                {"depth": 2, "state_text": "8008,8004"},
            ]
        },
        {"solver_trace": [{"depth": 1, "state_text": "true"}]},
    ]

    batch_indices, step_indices, value_targets, presence_targets = module.typed_value_trace_targets(
        rows,
        max_register_steps=5,
        max_value_slots=3,
        value_scale=10000.0,
        device=torch.device("cpu"),
    )

    assert batch_indices.tolist() == [0, 0, 1, 1]
    assert step_indices.tolist() == [1, 4, 1, 2]
    assert torch.allclose(
        value_targets,
        torch.tensor(
            [
                [0.4010, 0.0, 0.0],
                [0.8017, 0.0, 0.0],
                [0.4004, 0.4002, 0.0],
                [0.8008, 0.8004, 0.0],
            ]
        ),
    )
    assert presence_targets.tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]


def test_compute_typed_value_trace_loss_uses_only_present_slots():
    module = _load_stage530()
    head = module.TypedValueTraceHead(d_state=2, hidden_dim=4, value_scale=10000.0)
    value_registers = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value_targets = torch.tensor([[0.25, 0.0]])
    presence_targets = torch.tensor([[1.0, 0.0]])

    loss, metrics = module.compute_typed_value_trace_loss(
        head,
        value_registers,
        value_targets,
        presence_targets,
    )

    assert loss.item() > 0.0
    assert metrics["presence_accuracy"] >= 0.0
    assert metrics["value_mae"] >= 0.0


def test_typed_value_digit_trace_targets_right_align_digits_and_lists():
    module = _load_stage530()
    rows = [
        {"solver_trace": [{"depth": 4, "state_text": "8017"}]},
        {"solver_trace": [{"depth": 2, "state_text": "8008,8004"}]},
        {"solver_trace": [{"depth": 1, "state_text": "true"}]},
    ]

    batch_indices, step_indices, digit_targets, presence_targets = module.typed_value_digit_trace_targets(
        rows,
        max_register_steps=5,
        max_value_slots=3,
        max_digits=6,
        device=torch.device("cpu"),
    )

    assert batch_indices.tolist() == [0, 1]
    assert step_indices.tolist() == [4, 2]
    assert digit_targets[0, 0].tolist() == [
        module.IGNORE_INDEX,
        module.IGNORE_INDEX,
        8,
        0,
        1,
        7,
    ]
    assert presence_targets[0, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert digit_targets[1, 0].tolist() == [
        module.IGNORE_INDEX,
        module.IGNORE_INDEX,
        8,
        0,
        0,
        8,
    ]
    assert digit_targets[1, 1].tolist() == [
        module.IGNORE_INDEX,
        module.IGNORE_INDEX,
        8,
        0,
        0,
        4,
    ]
    assert presence_targets[1, 2].tolist() == [0.0] * 6


def test_digit_transition_pretraining_examples_shift_previous_to_next_states():
    module = _load_stage530()
    rows = [
        {
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "4010"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "8020"},
                {"depth": 4, "operation": "subtract_offset", "state_text": "8017"},
            ]
        },
        {
            "solver_trace": [
                {"depth": 1, "operation": "filter_even", "state_text": "4004,4002"},
                {"depth": 2, "operation": "double_filtered", "state_text": "8008,8004"},
            ]
        },
    ]

    examples = module.digit_transition_pretraining_examples(
        rows,
        max_value_slots=3,
        max_digits=4,
        device=torch.device("cpu"),
    )

    assert examples.operation_names == [
        "multiply_sum",
        "subtract_offset",
        "double_filtered",
    ]
    assert examples.previous_digit_targets.shape == (3, 3, 4)
    assert examples.next_digit_targets.shape == (3, 3, 4)
    assert examples.previous_digit_targets[0, 0].tolist() == [4, 0, 1, 0]
    assert examples.next_digit_targets[0, 0].tolist() == [8, 0, 2, 0]
    assert examples.previous_digit_targets[2, 1].tolist() == [4, 0, 0, 2]
    assert examples.next_digit_targets[2, 1].tolist() == [8, 0, 0, 4]


def test_digit_transition_pretraining_examples_can_include_initial_source_state():
    module = _load_stage530()
    rows = [
        {
            "question": "Compute ((4007 + 3) * 2) - 3.",
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "4010"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "8020"},
            ],
        },
        {
            "question": "From the list [4001, 4004, 4002], keep only even numbers, double each kept number.",
            "solver_trace": [
                {"depth": 1, "operation": "filter_even", "state_text": "4004,4002"},
                {"depth": 2, "operation": "double_filtered", "state_text": "8008,8004"},
            ],
        },
    ]

    examples = module.digit_transition_pretraining_examples(
        rows,
        max_value_slots=4,
        max_digits=4,
        include_initial_source=True,
        device=torch.device("cpu"),
    )

    assert examples.operation_names == [
        "add_operands",
        "multiply_sum",
        "filter_even",
        "double_filtered",
    ]
    assert examples.previous_digit_targets[0, 0].tolist() == [4, 0, 0, 7]
    assert examples.previous_digit_targets[0, 1].tolist() == [
        module.IGNORE_INDEX,
        module.IGNORE_INDEX,
        module.IGNORE_INDEX,
        3,
    ]
    assert examples.next_digit_targets[0, 0].tolist() == [4, 0, 1, 0]
    assert examples.previous_digit_targets[2, 1].tolist() == [4, 0, 0, 4]
    assert examples.next_digit_targets[2, 0].tolist() == [4, 0, 0, 4]


def test_digit_transition_pretraining_examples_keep_source_anchor_features():
    module = _load_stage530()
    rows = [
        {
            "question": "Compute ((4007 + 3) * 2) - 3.",
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "4010"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "8020"},
            ],
        }
    ]

    examples = module.digit_transition_pretraining_examples(
        rows,
        max_value_slots=4,
        max_digits=4,
        include_initial_source=True,
        source_feature_dim=32,
        source_value_scale=10000.0,
        device=torch.device("cpu"),
    )

    assert examples.source_numeric_features is not None
    assert examples.source_numeric_feature_mask is not None
    assert examples.source_numeric_features.shape == (2, 4, 32)
    assert examples.source_numeric_feature_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert torch.allclose(
        examples.source_numeric_features[:, :, 0],
        torch.tensor([[0.4007, 0.0003, 0.0002, 0.0003], [0.4007, 0.0003, 0.0002, 0.0003]]),
    )


def test_digit_transition_pretraining_passes_source_anchor_features_to_executor():
    module = _load_stage530()

    class CapturingExecutor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_state = 2
            self.source_feature_dim = 32
            self.digit_state_embed = torch.nn.Embedding(10, self.d_state)
            self.digit_bias = torch.nn.Parameter(torch.zeros(10))
            self.presence_bias = torch.nn.Parameter(torch.zeros(()))
            self.seen_source_features = None
            self.seen_source_mask = None

        def forward(
            self,
            digit_trajectory,
            *,
            operation_ids,
            operation_arg_ids,
            source_numeric_features=None,
            source_numeric_feature_mask=None,
            transition_off=False,
            column_procedure_off=False,
            return_logits=False,
        ):
            self.seen_source_features = source_numeric_features.detach().clone() if source_numeric_features is not None else None
            self.seen_source_mask = source_numeric_feature_mask.detach().clone() if source_numeric_feature_mask is not None else None
            batch, steps, slots, _d_state = digit_trajectory.shape
            digit_logits = self.digit_bias.view(1, 1, 1, 10).expand(batch, steps, slots, 10)
            presence_logits = self.presence_bias.view(1, 1, 1).expand(batch, steps, slots)
            return module.TypedDigitNextStateOutput(
                trajectory=digit_trajectory,
                digit_logits=digit_logits,
                presence_logits=presence_logits,
            )

    rows = [
        {
            "question": "Compute ((4007 + 3) * 2) - 3.",
            "solver_trace": [{"depth": 1, "operation": "add_operands", "state_text": "4010"}],
        }
    ]
    executor = CapturingExecutor()

    module.pretrain_digit_transition_executor(
        executor,
        rows,
        max_value_slots=4,
        max_digits=4,
        epochs=1,
        batch_size=1,
        lr=1e-3,
        grad_clip=1.0,
        include_initial_source=True,
        device=torch.device("cpu"),
    )

    assert executor.seen_source_features is not None
    assert executor.seen_source_mask is not None
    assert torch.allclose(executor.seen_source_features[0, :, 0], torch.tensor([0.4007, 0.0003, 0.0002, 0.0003]))
    assert executor.seen_source_mask.tolist() == [[1, 1, 1, 1]]


def test_digit_targets_to_executor_seed_trajectory_places_digit_embeddings_and_carry_slots():
    module = _load_stage530()
    executor = module.TypedDigitNextStateExecutor(d_state=3, n_operations=4, scan_digits=2)
    with torch.no_grad():
        executor.digit_state_embed.weight.copy_(torch.arange(30, dtype=torch.float32).view(10, 3))
    digit_targets = torch.tensor([[[1, 2], [module.IGNORE_INDEX, 3]]])
    presence_targets = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])

    trajectory = module.digit_targets_to_executor_seed_trajectory(
        executor,
        digit_targets,
        presence_targets,
        max_digits=2,
    )

    assert trajectory.shape == (1, 2, 6, 3)
    assert torch.equal(trajectory[0, 0, 0], executor.digit_state_embed.weight[1])
    assert torch.equal(trajectory[0, 0, 1], executor.digit_state_embed.weight[2])
    assert torch.equal(trajectory[0, 0, 2], torch.zeros(3))
    assert torch.equal(trajectory[0, 0, 3], torch.zeros(3))
    assert torch.equal(trajectory[0, 0, 4], executor.digit_state_embed.weight[3])
    assert torch.equal(trajectory[0, 1], torch.zeros_like(trajectory[0, 1]))


def test_compute_typed_value_digit_trace_loss_reads_present_digits():
    module = _load_stage530()
    head = module.TypedValueDigitTraceHead(d_state=2, max_digits=4, hidden_dim=4)
    value_registers = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    digit_targets = torch.tensor(
        [
            [
                [module.IGNORE_INDEX, 8, 0, 7],
                [module.IGNORE_INDEX, module.IGNORE_INDEX, module.IGNORE_INDEX, module.IGNORE_INDEX],
            ]
        ]
    )
    presence_targets = torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    loss, metrics = module.compute_typed_value_digit_trace_loss(
        head,
        value_registers,
        digit_targets,
        presence_targets,
    )

    assert loss.item() > 0.0
    assert metrics["digit_accuracy"] >= 0.0
    assert metrics["presence_accuracy"] >= 0.0


def test_compute_typed_digit_register_trace_loss_reads_digit_columns_not_value_slot():
    module = _load_stage530()
    head = module.TypedDigitRegisterTraceHead(d_state=2, max_digits=4, hidden_dim=4)
    # One numeric value slot has four digit-column registers plus one carry pocket.
    digit_registers = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.0],
            ]
        ]
    )
    digit_targets = torch.tensor([[[module.IGNORE_INDEX, 8, 0, 7]]])
    presence_targets = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

    loss, metrics = module.compute_typed_digit_register_trace_loss(
        head,
        digit_registers,
        digit_targets,
        presence_targets,
    )

    assert loss.item() > 0.0
    assert metrics["digit_accuracy"] >= 0.0
    assert metrics["presence_accuracy"] >= 0.0


def test_typed_register_answerer_digit_output_bridge_reads_typed_digit_trajectory():
    module = _load_stage530()
    answerer = module.TypedRegisterAnswerer(
        d_state=4,
        vocab_size=12,
        max_candidates=1,
        max_candidate_chars=4,
        n_heads=1,
        dropout=0.0,
        digit_register_output_bridge=True,
        typed_digit_register_digits=4,
        digit_char_ids=list(range(10)),
    )
    answerer.eval()
    readout = torch.zeros(1, 4)
    register_trajectory = torch.zeros(1, 2, 1, 4)
    teacher_char_targets = torch.tensor([[[0, 1, 2, 3]]])
    digit_trajectory_a = torch.zeros(1, 2, 5, 4)
    digit_trajectory_b = digit_trajectory_a.clone()
    digit_trajectory_b[0, -1, 0, 0] = 4.0
    digit_trajectory_b[0, -1, 1, 1] = 3.0

    with torch.no_grad():
        logits_a = answerer(
            readout=readout,
            register_trajectory=register_trajectory,
            typed_digit_register_trajectory=digit_trajectory_a,
            teacher_char_targets=teacher_char_targets,
        )[0]
        logits_b = answerer(
            readout=readout,
            register_trajectory=register_trajectory,
            typed_digit_register_trajectory=digit_trajectory_b,
            teacher_char_targets=teacher_char_targets,
        )[0]

    assert not torch.allclose(logits_a[..., :10], logits_b[..., :10])


def test_trace_digit_register_context_selects_matching_solver_steps_only_when_bridge_enabled():
    module = _load_stage530()
    context = {
        "typed_digit_register_trajectory": torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
    }
    batch_indices = torch.tensor([0, 1])
    step_indices = torch.tensor([2, 4])

    selected = module.trace_digit_register_context_for_steps(
        context,
        batch_indices,
        step_indices,
        use_digit_bridge=True,
    )

    assert selected is not None
    assert selected.shape == (2, 1, 3, 4)
    assert torch.equal(selected[0, 0], context["typed_digit_register_trajectory"][0, 2])
    assert torch.equal(selected[1, 0], context["typed_digit_register_trajectory"][1, 4])
    assert (
        module.trace_digit_register_context_for_steps(
            context,
            batch_indices,
            step_indices,
            use_digit_bridge=False,
        )
        is None
    )


def test_trace_operation_tensors_routes_arithmetic_operand_arguments():
    module = _load_stage530()
    rows = [
        {
            "prompt": (
                "Answer with only the final answer. Do not write reasoning.\n"
                "Question: Compute ((4007 + 3) * 2) - 3.\n"
                "Answer:"
            ),
            "solver_trace": [
                {"operation": "add_operands", "state_text": "4010"},
                {"operation": "multiply_sum", "state_text": "8020"},
                {"operation": "subtract_offset", "state_text": "8017"},
                {"operation": "hold_final", "state_text": "8017"},
            ],
        }
    ]

    operation_ids, argument_ids = module.stage523.trace_operation_tensors(
        rows,
        n_steps=4,
        device=torch.device("cpu"),
    )

    assert operation_ids[0].tolist() == [
        module.stage523.TRACE_OPERATION_TO_ID["add_operands"],
        module.stage523.TRACE_OPERATION_TO_ID["multiply_sum"],
        module.stage523.TRACE_OPERATION_TO_ID["subtract_offset"],
        module.stage523.TRACE_OPERATION_TO_ID["hold_final"],
    ]
    assert argument_ids[0].tolist() == [3, 2, 3, 0]


def test_trace_operation_tensors_routes_list_transform_arguments():
    module = _load_stage530()
    rows = [
        {
            "prompt": (
                "Question: From the list [4001, 4004, 4002], keep only even numbers, "
                "double each kept number, and return comma-separated values with no spaces."
            ),
            "solver_trace": [
                {"operation": "filter_even", "state_text": "4004,4002"},
                {"operation": "double_filtered", "state_text": "8008,8004"},
                {"operation": "hold_final", "state_text": "8008,8004"},
            ],
        }
    ]

    _operation_ids, argument_ids = module.stage523.trace_operation_tensors(
        rows,
        n_steps=4,
        device=torch.device("cpu"),
    )

    assert argument_ids[0].tolist() == [2, 2, 0, 0]


def test_typed_digit_next_state_executor_is_ablatable_and_operation_conditioned():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=6,
        hidden_dim=8,
    )
    digit_trajectory = torch.zeros(2, 3, 5, 4)
    digit_trajectory[:, 0, :, 0] = 1.0
    operation_ids = torch.tensor([[1, 2], [2, 1]])
    operation_arg_ids = torch.tensor([[3, 4], [4, 3]])
    source_features = torch.randn(2, 4, 6)
    source_mask = torch.ones(2, 4)

    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        transition_off=True,
    )
    on = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    assert off.shape == digit_trajectory.shape
    assert torch.equal(off, digit_trajectory)
    assert on.shape == digit_trajectory.shape
    assert not torch.allclose(on[:, 1], digit_trajectory[:, 1])
    assert not torch.allclose(on[0, 1], on[1, 1])


def test_digit_transition_executor_context_replaces_digit_trajectory_only_when_enabled():
    module = _load_stage530()

    class AddOneExecutor(torch.nn.Module):
        def forward(self, digit_trajectory, **_kwargs):
            return digit_trajectory + 1.0

    context = {"typed_digit_register_trajectory": torch.zeros(1, 2, 3, 4)}
    operation_ids = torch.zeros(1, 1, dtype=torch.long)
    operation_arg_ids = torch.zeros(1, 1, dtype=torch.long)

    disabled = module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=None,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=False,
    )
    enabled = module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=AddOneExecutor(),
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=False,
    )

    assert disabled is context
    assert enabled is not context
    assert torch.equal(context["typed_digit_register_trajectory"], torch.zeros(1, 2, 3, 4))
    assert torch.equal(enabled["typed_digit_register_trajectory"], torch.ones(1, 2, 3, 4))


def test_digit_transition_executor_context_preserves_direct_logits_when_requested():
    module = _load_stage530()

    class LogitExecutor(torch.nn.Module):
        def forward(self, digit_trajectory, **kwargs):
            assert kwargs["return_logits"] is True
            return module.TypedDigitNextStateOutput(
                trajectory=digit_trajectory + 1.0,
                digit_logits=torch.ones(1, 2, 3, 10),
                presence_logits=torch.ones(1, 2, 3) * 2.0,
            )

    context = {"typed_digit_register_trajectory": torch.zeros(1, 2, 3, 4)}
    output = module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=LogitExecutor(),
        operation_ids=torch.zeros(1, 1, dtype=torch.long),
        operation_arg_ids=torch.zeros(1, 1, dtype=torch.long),
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=False,
        return_logits=True,
    )

    assert torch.equal(output["typed_digit_register_trajectory"], torch.ones(1, 2, 3, 4))
    assert output["digit_transition_executor_digit_logits"].shape == (1, 2, 3, 10)
    assert torch.equal(output["digit_transition_executor_presence_logits"], torch.ones(1, 2, 3) * 2.0)


def test_digit_transition_executor_context_passes_column_procedure_off_ablation():
    module = _load_stage530()

    class CaptureExecutor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.column_procedure_off = None

        def forward(self, digit_trajectory, **kwargs):
            self.column_procedure_off = kwargs["column_procedure_off"]
            return digit_trajectory + 1.0

    executor = CaptureExecutor()
    context = {"typed_digit_register_trajectory": torch.zeros(1, 2, 3, 4)}

    module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=executor,
        operation_ids=torch.zeros(1, 1, dtype=torch.long),
        operation_arg_ids=torch.zeros(1, 1, dtype=torch.long),
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=False,
        column_procedure_off=True,
    )

    assert executor.column_procedure_off is True


def test_committed_digit_writeback_updates_main_working_register_trajectory():
    module = _load_stage530()
    torch.manual_seed(0)
    writeback = module.TypedDigitCommittedWriteback(d_state=4, n_heads=2, gate_init_bias=2.0)
    working = torch.zeros(1, 2, 3, 4)
    digits = torch.zeros(1, 2, 5, 4)
    digits[:, 1, 0, 0] = 2.0

    updated = writeback(working, digits, writeback_off=False)
    disabled = writeback(working, digits, writeback_off=True)

    assert updated.shape == working.shape
    assert torch.equal(disabled, working)
    assert not torch.allclose(updated[:, 1], working[:, 1])
    assert torch.allclose(updated[:, 0], working[:, 0])


def test_typed_digit_ledger_attractor_refines_draft_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    attractor = module.TypedDigitLedgerAttractor(d_state=4, hidden_dim=8, n_refine_steps=2)
    draft = torch.zeros(1, 3, 5, 4)
    draft[:, 1, 0, 0] = 1.0
    digit_logits = torch.zeros(1, 3, 5, 10)
    presence_logits = torch.zeros(1, 3, 5)

    refined = attractor(
        module.TypedDigitNextStateOutput(
            trajectory=draft,
            digit_logits=digit_logits,
            presence_logits=presence_logits,
        ),
        attractor_off=False,
    )
    disabled = attractor(
        module.TypedDigitNextStateOutput(
            trajectory=draft,
            digit_logits=digit_logits,
            presence_logits=presence_logits,
        ),
        attractor_off=True,
    )

    assert refined.trajectory.shape == draft.shape
    assert refined.digit_logits.shape == digit_logits.shape
    assert refined.presence_logits.shape == presence_logits.shape
    assert torch.equal(disabled.trajectory, draft)
    assert torch.equal(disabled.digit_logits, digit_logits)
    assert torch.equal(disabled.presence_logits, presence_logits)
    assert not torch.allclose(refined.trajectory[:, 1], draft[:, 1])


def test_attractor_context_refines_digit_trajectory_before_committed_writeback():
    module = _load_stage530()

    class DraftExecutor(torch.nn.Module):
        def forward(self, digit_trajectory, **_kwargs):
            return module.TypedDigitNextStateOutput(
                trajectory=digit_trajectory + 1.0,
                digit_logits=torch.ones(1, 2, 3, 10),
                presence_logits=torch.ones(1, 2, 3),
            )

    class AddTwoAttractor(torch.nn.Module):
        def forward(self, executor_output, **kwargs):
            assert kwargs["attractor_off"] is False
            return module.TypedDigitNextStateOutput(
                trajectory=executor_output.trajectory + 2.0,
                digit_logits=executor_output.digit_logits + 2.0,
                presence_logits=executor_output.presence_logits + 2.0,
            )

    class CaptureWriteback(torch.nn.Module):
        def forward(self, working_register_trajectory, typed_digit_register_trajectory, **_kwargs):
            assert torch.equal(typed_digit_register_trajectory, torch.ones(1, 2, 3, 4) * 3.0)
            return working_register_trajectory + typed_digit_register_trajectory.mean(dim=2, keepdim=True)

    context = {
        "typed_digit_register_trajectory": torch.zeros(1, 2, 3, 4),
        "working_register_trajectory": torch.zeros(1, 2, 1, 4),
    }
    output = module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=DraftExecutor(),
        digit_ledger_attractor=AddTwoAttractor(),
        digit_committed_writeback=CaptureWriteback(),
        operation_ids=torch.zeros(1, 1, dtype=torch.long),
        operation_arg_ids=torch.zeros(1, 1, dtype=torch.long),
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=False,
        attractor_off=False,
        return_logits=True,
    )

    assert torch.equal(output["typed_digit_register_trajectory"], torch.ones(1, 2, 3, 4) * 3.0)
    assert torch.equal(output["digit_transition_executor_digit_logits"], torch.ones(1, 2, 3, 10) * 3.0)
    assert torch.equal(output["digit_transition_executor_presence_logits"], torch.ones(1, 2, 3) * 3.0)
    assert torch.equal(output["working_register_trajectory"], torch.ones(1, 2, 1, 4) * 3.0)


def test_executor_off_also_disables_attractor_for_clean_ablation():
    module = _load_stage530()

    class AddOneExecutor(torch.nn.Module):
        def forward(self, digit_trajectory, **kwargs):
            assert kwargs["transition_off"] is True
            return module.TypedDigitNextStateOutput(
                trajectory=digit_trajectory,
                digit_logits=torch.zeros(1, 2, 3, 10),
                presence_logits=torch.zeros(1, 2, 3),
            )

    class BadAttractor(torch.nn.Module):
        def forward(self, executor_output, **kwargs):
            assert kwargs["attractor_off"] is True
            return module.TypedDigitNextStateOutput(
                trajectory=executor_output.trajectory,
                digit_logits=executor_output.digit_logits,
                presence_logits=executor_output.presence_logits,
            )

    context = {"typed_digit_register_trajectory": torch.zeros(1, 2, 3, 4)}
    output = module.apply_digit_transition_executor_context(
        context,
        digit_transition_executor=AddOneExecutor(),
        digit_ledger_attractor=BadAttractor(),
        operation_ids=torch.zeros(1, 1, dtype=torch.long),
        operation_arg_ids=torch.zeros(1, 1, dtype=torch.long),
        source_numeric_features=None,
        source_numeric_feature_mask=None,
        transition_off=True,
        attractor_off=False,
        return_logits=True,
    )

    assert torch.equal(output["typed_digit_register_trajectory"], context["typed_digit_register_trajectory"])


def test_typed_digit_next_state_executor_returns_digit_logits_with_discrete_feedback():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=0,
        hidden_dim=8,
        discrete_state_feedback=True,
    )
    digit_trajectory = torch.zeros(2, 3, 5, 4)
    operation_ids = torch.tensor([[1, 2], [2, 1]])
    operation_arg_ids = torch.tensor([[3, 4], [4, 3]])

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        return_logits=True,
    )

    assert output.trajectory.shape == digit_trajectory.shape
    assert output.digit_logits.shape == (2, 3, 5, 10)
    assert output.presence_logits.shape == (2, 3, 5)
    assert not torch.allclose(output.trajectory[:, 1], digit_trajectory[:, 1])


def test_ledger_causal_renderer_changes_full_answer_logits_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    answerer = module.TypedRegisterAnswerer(
        d_state=4,
        vocab_size=12,
        max_candidates=2,
        max_candidate_chars=4,
        hidden_dim=8,
        n_heads=2,
        digit_register_output_bridge=False,
        ledger_causal_renderer=True,
        ledger_causal_renderer_scale=1.0,
    )
    with torch.no_grad():
        answerer.ledger_renderer_head.bias[3] = 5.0
    readout = torch.zeros(1, 4)
    working = torch.zeros(1, 2, 3, 4)
    digits = torch.zeros(1, 2, 5, 4)
    digits[:, -1, 0, 0] = 3.0

    on_logits = answerer(
        readout=readout,
        register_trajectory=working,
        typed_digit_register_trajectory=digits,
        ledger_causal_renderer_off=False,
    )[0]
    off_logits = answerer(
        readout=readout,
        register_trajectory=working,
        typed_digit_register_trajectory=digits,
        ledger_causal_renderer_off=True,
    )[0]

    assert on_logits.shape == off_logits.shape == (1, 2, 4, 12)
    assert not torch.allclose(on_logits, off_logits)


def test_ledger_forced_copy_renderer_maps_digit_logits_to_char_ids_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    digit_char_ids = [3 + digit for digit in range(10)]
    answerer = module.TypedRegisterAnswerer(
        d_state=4,
        vocab_size=14,
        max_candidates=1,
        max_candidate_chars=3,
        hidden_dim=8,
        n_heads=2,
        digit_register_output_bridge=False,
        ledger_forced_copy_renderer=True,
        ledger_forced_copy_renderer_scale=1.0,
        digit_char_ids=digit_char_ids,
    )
    with torch.no_grad():
        answerer.char_head.weight.zero_()
        answerer.char_head.bias.zero_()
        answerer.ledger_copy_digit_head.weight.zero_()
        answerer.ledger_copy_digit_head.bias.zero_()
        answerer.ledger_copy_digit_head.bias[7] = 8.0
        answerer.ledger_copy_presence_head.weight.zero_()
        answerer.ledger_copy_presence_head.bias.fill_(8.0)
    readout = torch.zeros(1, 4)
    working = torch.zeros(1, 2, 3, 4)
    digits = torch.zeros(1, 2, 5, 4)
    digits[:, -1, 0, 0] = 3.0

    on_logits = answerer(
        readout=readout,
        register_trajectory=working,
        typed_digit_register_trajectory=digits,
        ledger_forced_copy_renderer_off=False,
    )[0]
    off_logits = answerer(
        readout=readout,
        register_trajectory=working,
        typed_digit_register_trajectory=digits,
        ledger_forced_copy_renderer_off=True,
    )[0]

    assert on_logits.shape == off_logits.shape == (1, 1, 3, 14)
    assert int(on_logits[0, 0, 0].argmax()) == digit_char_ids[7]
    assert not torch.allclose(on_logits, off_logits)


def test_ledger_pact_renderer_packs_numeric_and_list_answers_from_digit_logits():
    module = _load_stage530()
    digit_logits = torch.full((3, 4, 14, 10), -8.0)
    presence_logits = torch.full((3, 4, 14), -8.0)

    def write_value(row_index, value_slot, digits):
        places = 7
        start = value_slot * places
        pad = 6 - len(digits)
        for offset, digit in enumerate(digits):
            slot = start + pad + offset
            digit_logits[row_index, -1, slot, int(digit)] = 8.0
            presence_logits[row_index, -1, slot] = 8.0

    write_value(0, 0, "8017")
    write_value(1, 0, "8008")
    write_value(1, 1, "8004")
    rows = [
        {"task_family": "arithmetic_chain", "answer_aliases": ["8017"]},
        {"task_family": "list_transform", "answer_aliases": ["8008,8004"]},
        {"task_family": "boolean_logic", "answer_aliases": ["TRUE"]},
    ]

    rendered, mask = module.render_ledger_pact_texts(
        rows,
        digit_logits,
        presence_logits,
        max_digits=6,
    )

    assert rendered == ["8017", "8008,8004", ""]
    assert mask.tolist() == [True, True, False]


def test_ledger_pact_candidates_override_only_numeric_answer_slots():
    module = _load_stage530()
    candidates = [["old", "", ""], ["old-list", "", ""], ["TRUE", "FALSE", ""]]
    rendered = ["8017", "8008,8004", ""]
    mask = torch.tensor([True, True, False])

    patched, selected = module.apply_ledger_pact_candidates(
        candidates,
        rendered,
        mask,
        answer_slot_index=0,
    )

    assert patched == [["8017", "", ""], ["8008,8004", "", ""], ["TRUE", "FALSE", ""]]
    assert selected == [0, 0, None]


def test_ledger_candidate_exposure_includes_second_best_digit_alternative():
    module = _load_stage530()
    digit_logits = torch.full((1, 2, 7, 10), -8.0)
    presence_logits = torch.full((1, 2, 7), -8.0)
    # Right-aligned six-digit value. Greedy final digit is 6, but 7 is the
    # second-best ledger belief, so candidate exposure should place 8017 on the
    # table for the verifier.
    for offset, digit in enumerate("8016"):
        slot = 2 + offset
        digit_logits[0, -1, slot, int(digit)] = 8.0
        presence_logits[0, -1, slot] = 8.0
    digit_logits[0, -1, 5, 7] = 7.5
    rows = [{"task_family": "arithmetic_chain", "answer_aliases": ["8017"]}]

    candidates, mask = module.render_ledger_candidate_texts(
        rows,
        digit_logits,
        presence_logits,
        max_digits=6,
        max_candidates=4,
        digit_alt_topk=2,
    )

    assert mask.tolist() == [True]
    assert candidates[0][0] == "8016"
    assert "8017" in candidates[0]


def test_ledger_candidate_exposure_overwrites_multiple_numeric_slots():
    module = _load_stage530()
    candidates = [["free0", "free1", "free2", "free3"], ["TRUE", "FALSE", "", ""]]
    ledger_candidates = [["8016", "8017"], [""]]
    mask = torch.tensor([True, False])

    patched, selected = module.apply_ledger_candidate_exposure(
        candidates,
        ledger_candidates,
        mask,
        answer_slot_index=1,
        max_candidates=4,
    )

    assert patched == [["free0", "8016", "8017", "free3"], ["TRUE", "FALSE", "", ""]]
    assert selected == [1, None]


def test_ledger_candidate_exposure_reads_all_arithmetic_value_slots():
    module = _load_stage530()
    digit_logits = torch.full((1, 2, 14, 10), -8.0)
    presence_logits = torch.full((1, 2, 14), -8.0)

    def write_value(value_slot, digits):
        places = 7
        start = value_slot * places
        pad = 6 - len(digits)
        for offset, digit in enumerate(digits):
            slot = start + pad + offset
            digit_logits[0, -1, slot, int(digit)] = 8.0
            presence_logits[0, -1, slot] = 8.0

    write_value(0, "604")
    write_value(1, "8017")
    rows = [{"task_family": "arithmetic_chain", "answer_aliases": ["8017"]}]

    candidates, mask = module.render_ledger_candidate_texts(
        rows,
        digit_logits,
        presence_logits,
        max_digits=6,
        max_candidates=4,
        digit_alt_topk=2,
    )

    assert mask.tolist() == [True]
    assert "604" in candidates[0]
    assert "8017" in candidates[0]


def test_choice_verifier_candidates_select_supplied_choice_and_are_ablatable():
    module = _load_stage530()
    rows = [
        {"choices": ["8020", "8017", "8016"], "answer_aliases": ["8017"]},
        {"choices": ["FALSE", "TRUE"], "answer_aliases": ["FALSE"]},
    ]
    verifier_logits = torch.tensor([[0.1, 4.0, -1.0], [3.0, 0.5, -9.0]])
    choice_mask = torch.tensor([[True, True, True], [True, True, False]])

    candidates, selected = module.apply_choice_verifier_candidates(
        rows,
        verifier_logits,
        choice_mask,
        max_candidates=3,
        choice_verifier_off=False,
    )
    off_candidates, off_selected = module.apply_choice_verifier_candidates(
        rows,
        verifier_logits,
        choice_mask,
        max_candidates=3,
        choice_verifier_off=True,
    )

    assert candidates == [["8020", "8017", "8016"], ["FALSE", "TRUE", ""]]
    assert selected == [1, 0]
    assert off_candidates == [["", "", ""], ["", "", ""]]
    assert off_selected == [None, None]


def test_procedure_choice_verifier_masks_invalid_choices():
    module = _load_stage530()
    torch.manual_seed(0)
    verifier = module.ProcedureChoiceVerifier(
        d_state=4,
        vocab_size=8,
        max_choice_chars=3,
        hidden_dim=8,
    )
    thought_state = torch.randn(2, 4)
    choice_ids = torch.tensor(
        [
            [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
            [[5, 0, 0], [6, 7, 0], [0, 0, 0]],
        ]
    )
    choice_mask = torch.tensor([[True, True, False], [True, True, False]])

    logits = verifier(thought_state, choice_ids, choice_mask)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits[:, :2]).all()
    assert logits[:, 2].lt(-1e8).all()


def test_pairwise_tournament_scores_sum_valid_pairwise_wins():
    module = _load_stage530()
    pairwise_logits = torch.full((1, 3, 3), -100.0)
    choice_mask = torch.tensor([[True, True, True]])
    pairwise_logits[0, 1, 0] = 3.0
    pairwise_logits[0, 1, 2] = 4.0
    pairwise_logits[0, 0, 1] = -3.0
    pairwise_logits[0, 2, 1] = -4.0
    pairwise_logits[0, 0, 2] = 1.0
    pairwise_logits[0, 2, 0] = -1.0

    scores = module.pairwise_tournament_scores(pairwise_logits, choice_mask)

    assert scores.shape == (1, 3)
    assert int(scores.argmax(dim=-1).item()) == 1


def test_pairwise_procedure_choice_verifier_masks_invalid_pairs():
    module = _load_stage530()
    torch.manual_seed(0)
    verifier = module.PairwiseProcedureChoiceVerifier(
        d_state=4,
        vocab_size=8,
        max_choice_chars=3,
        hidden_dim=8,
    )
    thought_state = torch.randn(2, 4)
    choice_ids = torch.tensor(
        [
            [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
            [[5, 0, 0], [6, 7, 0], [0, 0, 0]],
        ]
    )
    choice_mask = torch.tensor([[True, True, False], [True, True, False]])

    pairwise_logits = verifier(thought_state, choice_ids, choice_mask)
    scores = module.pairwise_tournament_scores(pairwise_logits, choice_mask)

    assert pairwise_logits.shape == (2, 3, 3)
    assert torch.isfinite(pairwise_logits[:, :2, :2]).all()
    assert pairwise_logits[:, 2].lt(-1e8).all()
    assert pairwise_logits[:, :, 2].lt(-1e8).all()
    assert scores.shape == (2, 3)
    assert scores[:, 2].lt(-1e8).all()


def test_pairwise_choice_verifier_loss_prefers_target_choice():
    module = _load_stage530()
    pairwise_logits = torch.full((2, 3, 3), -8.0)
    choice_mask = torch.tensor([[True, True, True], [True, True, False]])
    targets = torch.tensor([1, 0])
    pairwise_logits[0, 1, 0] = 8.0
    pairwise_logits[0, 1, 2] = 8.0
    pairwise_logits[1, 0, 1] = 8.0

    loss, metrics = module.compute_pairwise_choice_verifier_loss(
        pairwise_logits,
        choice_mask,
        targets,
    )

    assert loss.item() < 0.1
    assert metrics["accuracy"] == 1.0


def test_generated_candidate_verifier_inputs_target_generated_gold_only():
    module = _load_stage530()
    rows = [
        {"answer_aliases": ["42"]},
        {"answer_aliases": ["green"]},
    ]
    candidates = [
        ["41", "42", ""],
        ["red", "blue", ""],
    ]

    choice_ids, choice_mask, targets = module.encode_candidate_string_verifier_inputs(
        candidates,
        rows,
        allowed_chars=[module.PAD, module.EOS, "0", "1", "2", "4", "b", "d", "e", "g", "l", "n", "r", "u"],
        max_candidates=3,
        max_candidate_chars=5,
        device=torch.device("cpu"),
    )

    assert choice_ids.shape == (2, 3, 5)
    assert choice_mask.tolist() == [[True, True, False], [True, True, False]]
    assert targets.tolist() == [1, module.IGNORE_INDEX]


def test_generated_choice_verifier_candidates_select_generated_not_supplied():
    module = _load_stage530()
    candidates = [["wrong", "right", ""], ["old", "new", ""]]
    logits = torch.tensor([[0.0, 3.0, -1e9], [4.0, 1.0, -1e9]])
    mask = torch.tensor([[True, True, False], [True, True, False]])

    patched, selected = module.apply_candidate_string_verifier_candidates(
        candidates,
        logits,
        mask,
        max_candidates=3,
    )

    assert patched == [["wrong", "right", ""], ["old", "new", ""]]
    assert selected == [1, 0]


def test_choice_verifier_owned_targets_mask_free_mouth_supervision():
    module = _load_stage530()
    char_targets = torch.ones(2, 3, 4, dtype=torch.long)
    select_targets = torch.tensor([0, 1])

    masked_chars, masked_select = module.mask_choice_verifier_owned_answer_targets(
        char_targets,
        select_targets,
        choice_verifier_owns_answer=True,
    )
    unmasked_chars, unmasked_select = module.mask_choice_verifier_owned_answer_targets(
        char_targets,
        select_targets,
        choice_verifier_owns_answer=False,
    )

    assert masked_chars.eq(module.IGNORE_INDEX).all()
    assert masked_select.eq(module.IGNORE_INDEX).all()
    assert torch.equal(unmasked_chars, char_targets)
    assert torch.equal(unmasked_select, select_targets)


def test_ledger_pact_renderer_is_inactive_when_typed_register_body_is_off():
    module = _load_stage530()

    class Args:
        answerer_ledger_pact_renderer = True

    assert module.ledger_pact_renderer_active(
        Args(),
        typed_register_off=False,
        ledger_pact_renderer_off=False,
    )
    assert not module.ledger_pact_renderer_active(
        Args(),
        typed_register_off=True,
        ledger_pact_renderer_off=False,
    )


def test_typed_primitive_pact_renderer_solves_boolean_without_answer_label():
    module = _load_stage530()
    row = {
        "task_family": "boolean_logic",
        "question": "Let P=FALSE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE.",
        "answer_aliases": ["WRONG"],
        "solver_trace": [
            {"operation": "not_q"},
            {"operation": "and_with_p"},
            {"operation": "or_with_r"},
            {"operation": "hold_final"},
        ],
    }
    digit_logits = torch.zeros(1, 4, 14, 10)
    presence_logits = torch.zeros(1, 4, 14)

    rendered, mask = module.render_typed_primitive_pact_texts(
        [row],
        digit_logits,
        presence_logits,
        max_digits=6,
    )

    assert rendered == ["FALSE"]
    assert mask.tolist() == [True]


def test_typed_primitive_pact_renderer_follows_symbolic_mapping_without_answer_label():
    module = _load_stage530()
    row = {
        "task_family": "symbolic_binding",
        "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
        "answer_aliases": ["WRONG"],
        "solver_trace": [
            {"operation": "first_mapping"},
            {"operation": "second_mapping"},
            {"operation": "hold_final"},
        ],
    }
    digit_logits = torch.zeros(1, 4, 14, 10)
    presence_logits = torch.zeros(1, 4, 14)

    rendered, mask = module.render_typed_primitive_pact_texts(
        [row],
        digit_logits,
        presence_logits,
        max_digits=6,
    )

    assert rendered == ["green"]
    assert mask.tolist() == [True]


def test_typed_primitive_pact_renderer_is_inactive_when_typed_register_body_is_off():
    module = _load_stage530()

    class Args:
        answerer_typed_primitive_pact_renderer = True

    assert module.typed_primitive_pact_renderer_active(
        Args(),
        typed_register_off=False,
        typed_primitive_pact_renderer_off=False,
    )
    assert not module.typed_primitive_pact_renderer_active(
        Args(),
        typed_register_off=True,
        typed_primitive_pact_renderer_off=False,
    )


def test_typed_primitive_pact_renderer_can_ablate_boolean_lane_only():
    module = _load_stage530()
    rows = [
        {
            "task_family": "boolean_logic",
            "question": "Let P=FALSE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE.",
            "solver_trace": [
                {"operation": "not_q"},
                {"operation": "and_with_p"},
                {"operation": "or_with_r"},
            ],
        },
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
            "solver_trace": [
                {"operation": "first_mapping"},
                {"operation": "second_mapping"},
            ],
        },
    ]
    digit_logits = torch.zeros(2, 4, 14, 10)
    presence_logits = torch.zeros(2, 4, 14)

    rendered, mask = module.render_typed_primitive_pact_texts(
        rows,
        digit_logits,
        presence_logits,
        max_digits=6,
        boolean_lane_off=True,
    )

    assert rendered == ["", "green"]
    assert mask.tolist() == [False, True]


def test_typed_primitive_pact_renderer_can_ablate_symbolic_lane_only():
    module = _load_stage530()
    rows = [
        {
            "task_family": "boolean_logic",
            "question": "Let P=FALSE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE.",
            "solver_trace": [
                {"operation": "not_q"},
                {"operation": "and_with_p"},
                {"operation": "or_with_r"},
            ],
        },
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
            "solver_trace": [
                {"operation": "first_mapping"},
                {"operation": "second_mapping"},
            ],
        },
    ]
    digit_logits = torch.zeros(2, 4, 14, 10)
    presence_logits = torch.zeros(2, 4, 14)

    rendered, mask = module.render_typed_primitive_pact_texts(
        rows,
        digit_logits,
        presence_logits,
        max_digits=6,
        symbolic_lane_off=True,
    )

    assert rendered == ["FALSE", ""]
    assert mask.tolist() == [True, False]


def test_boolean_lane_source_and_trace_targets_do_not_use_answer_label():
    module = _load_stage530()
    rows = [
        {
            "task_family": "boolean_logic",
            "question": "Let P=FALSE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE.",
            "answer_aliases": ["WRONG"],
            "solver_trace": [
                {"operation": "not_q"},
                {"operation": "and_with_p"},
                {"operation": "or_with_r"},
                {"operation": "hold_final"},
            ],
        }
    ]

    source_values, trace_targets, trace_mask = module.boolean_lane_source_and_trace_tensors(
        rows,
        n_steps=4,
        device=torch.device("cpu"),
    )

    assert source_values.tolist() == [[0.0, 0.0, 0.0]]
    assert trace_targets.tolist() == [[1, 0, 0, 0]]
    assert trace_mask.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_symbolic_lane_source_and_trace_targets_follow_pointer_without_answer_label():
    module = _load_stage530()
    rows = [
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
            "answer_aliases": ["WRONG"],
            "solver_trace": [
                {"operation": "first_mapping"},
                {"operation": "second_mapping"},
                {"operation": "hold_final"},
            ],
        }
    ]
    vocab = ["B", "E", "green", "red"]

    source_index, mapping_table, trace_targets, trace_mask = module.symbolic_lane_source_and_trace_tensors(
        rows,
        symbol_vocab=vocab,
        n_steps=3,
        device=torch.device("cpu"),
    )

    assert source_index.tolist() == [1]
    assert mapping_table[0].tolist() == [0, 3, 0, 2]
    assert trace_targets.tolist() == [[3, 2, 2]]
    assert trace_mask.tolist() == [[1.0, 1.0, 1.0]]


def test_build_symbolic_lane_vocab_collects_mapping_symbols_in_stable_order():
    module = _load_stage530()
    rows = [
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
        },
        {
            "task_family": "boolean_logic",
            "question": "Let P=TRUE, Q=FALSE, R=TRUE. Evaluate (P AND NOT Q) OR R.",
        },
    ]

    assert module.build_symbolic_lane_vocab(rows) == ["B", "E", "green", "red"]


def test_typed_primitive_lane_executor_returns_boolean_and_symbolic_trace_logits():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedPrimitiveLaneExecutor(
        d_state=8,
        n_operations=12,
        symbolic_vocab_size=4,
        hidden_dim=16,
    )
    boolean_sources = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    symbolic_source_index = torch.tensor([1, 2])
    symbolic_mapping_table = torch.tensor([[0, 3, 0, 2], [1, 2, 3, 0]])
    operation_ids = torch.tensor(
        [
            [
                module.stage523.TRACE_OPERATION_TO_ID["not_q"],
                module.stage523.TRACE_OPERATION_TO_ID["and_with_p"],
                module.stage523.TRACE_OPERATION_TO_ID["or_with_r"],
            ],
            [
                module.stage523.TRACE_OPERATION_TO_ID["first_mapping"],
                module.stage523.TRACE_OPERATION_TO_ID["second_mapping"],
                module.stage523.TRACE_OPERATION_TO_ID["hold_final"],
            ],
        ]
    )

    output = executor(
        boolean_sources=boolean_sources,
        symbolic_source_index=symbolic_source_index,
        symbolic_mapping_table=symbolic_mapping_table,
        operation_ids=operation_ids,
    )

    assert output.boolean_logits.shape == (2, 3, 2)
    assert output.symbolic_logits.shape == (2, 3, 4)


def test_compute_typed_primitive_lane_trace_loss_prefers_correct_logits():
    module = _load_stage530()
    good_boolean_logits = torch.zeros(1, 2, 2)
    good_boolean_logits[0, 0, 1] = 5.0
    good_boolean_logits[0, 1, 0] = 5.0
    bad_boolean_logits = good_boolean_logits.flip(-1)
    good_symbolic_logits = torch.zeros(1, 2, 4)
    good_symbolic_logits[0, 0, 3] = 5.0
    good_symbolic_logits[0, 1, 2] = 5.0
    bad_symbolic_logits = good_symbolic_logits.roll(shifts=1, dims=-1)
    boolean_targets = torch.tensor([[1, 0]])
    symbolic_targets = torch.tensor([[3, 2]])
    trace_mask = torch.tensor([[1.0, 1.0]])

    good_loss, good_metrics = module.compute_typed_primitive_lane_trace_loss(
        module.TypedPrimitiveLaneOutput(good_boolean_logits, good_symbolic_logits),
        boolean_targets=boolean_targets,
        boolean_mask=trace_mask,
        symbolic_targets=symbolic_targets,
        symbolic_mask=trace_mask,
    )
    bad_loss, _bad_metrics = module.compute_typed_primitive_lane_trace_loss(
        module.TypedPrimitiveLaneOutput(bad_boolean_logits, bad_symbolic_logits),
        boolean_targets=boolean_targets,
        boolean_mask=trace_mask,
        symbolic_targets=symbolic_targets,
        symbolic_mask=trace_mask,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["boolean_accuracy"] == 1.0
    assert good_metrics["symbolic_accuracy"] == 1.0


def test_learned_primitive_lane_renderer_uses_logits_not_answer_label():
    module = _load_stage530()
    rows = [
        {
            "task_family": "boolean_logic",
            "question": "Let P=TRUE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R.",
            "answer_aliases": ["WRONG"],
        },
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
            "answer_aliases": ["WRONG"],
        },
        {
            "task_family": "arithmetic_chain",
            "question": "Compute ((4007 + 3) * 2) - 3.",
            "answer_aliases": ["8017"],
        },
    ]
    boolean_logits = torch.zeros(3, 2, 2)
    boolean_logits[0, -1, 1] = 6.0
    symbolic_logits = torch.zeros(3, 2, 4)
    symbolic_logits[1, -1, 2] = 6.0
    output = module.TypedPrimitiveLaneOutput(
        boolean_logits=boolean_logits,
        symbolic_logits=symbolic_logits,
    )

    rendered, mask = module.render_learned_primitive_lane_texts(
        rows,
        output,
        symbol_vocab=["B", "E", "green", "red"],
    )

    assert rendered == ["TRUE", "green", ""]
    assert mask.tolist() == [True, True, False]


def test_learned_primitive_lane_renderer_ablation_masks_one_lane_only():
    module = _load_stage530()
    rows = [
        {"task_family": "boolean_logic"},
        {"task_family": "symbolic_binding"},
    ]
    boolean_logits = torch.zeros(2, 1, 2)
    boolean_logits[0, 0, 0] = 5.0
    symbolic_logits = torch.zeros(2, 1, 3)
    symbolic_logits[1, 0, 2] = 5.0
    output = module.TypedPrimitiveLaneOutput(boolean_logits, symbolic_logits)

    rendered_boolean_off, mask_boolean_off = module.render_learned_primitive_lane_texts(
        rows,
        output,
        symbol_vocab=["A", "B", "C"],
        boolean_lane_off=True,
    )
    rendered_symbolic_off, mask_symbolic_off = module.render_learned_primitive_lane_texts(
        rows,
        output,
        symbol_vocab=["A", "B", "C"],
        symbolic_lane_off=True,
    )

    assert rendered_boolean_off == ["", "C"]
    assert mask_boolean_off.tolist() == [False, True]
    assert rendered_symbolic_off == ["FALSE", ""]
    assert mask_symbolic_off.tolist() == [True, False]


def test_learned_primitive_lane_renderer_is_inactive_when_typed_register_body_is_off():
    module = _load_stage530()

    class Args:
        answerer_learned_primitive_lane_renderer = True

    assert module.learned_primitive_lane_renderer_active(
        Args(),
        typed_register_off=False,
        learned_primitive_lane_renderer_off=False,
    )
    assert not module.learned_primitive_lane_renderer_active(
        Args(),
        typed_register_off=True,
        learned_primitive_lane_renderer_off=False,
    )


def test_learned_numeric_procedure_renderer_is_inactive_when_typed_register_body_is_off():
    module = _load_stage530()

    class Args:
        answerer_learned_numeric_procedure_renderer = True

    assert module.learned_numeric_procedure_renderer_active(
        Args(),
        typed_register_off=False,
        learned_numeric_procedure_renderer_off=False,
    )
    assert not module.learned_numeric_procedure_renderer_active(
        Args(),
        typed_register_off=True,
        learned_numeric_procedure_renderer_off=False,
    )


def test_typed_primitive_lane_batch_inputs_bundle_sources_trace_targets_and_operations():
    module = _load_stage530()

    class Args:
        n_steps = 3

    rows = [
        {
            "task_family": "boolean_logic",
            "question": "Let P=FALSE, Q=FALSE, R=FALSE. Evaluate (P AND NOT Q) OR R.",
            "solver_trace": [
                {"operation": "not_q"},
                {"operation": "and_with_p"},
                {"operation": "or_with_r"},
            ],
        },
        {
            "task_family": "symbolic_binding",
            "question": "If E maps to red, red maps to green, and green maps to B, what does E map to after two mappings?",
            "solver_trace": [
                {"operation": "first_mapping"},
                {"operation": "second_mapping"},
                {"operation": "hold_final"},
            ],
        },
    ]

    batch_inputs = module.typed_primitive_lane_batch_inputs(
        rows,
        args=Args(),
        symbol_vocab=["B", "E", "green", "red"],
        device=torch.device("cpu"),
    )

    assert batch_inputs["operation_ids"].shape == (2, 3)
    assert batch_inputs["boolean_sources"].tolist()[0] == [0.0, 0.0, 0.0]
    assert batch_inputs["boolean_targets"].tolist()[0] == [1, 0, 0]
    assert batch_inputs["boolean_mask"].tolist()[0] == [1.0, 1.0, 1.0]
    assert batch_inputs["symbolic_source_index"].tolist()[1] == 1
    assert batch_inputs["symbolic_mapping_table"][1].tolist() == [0, 3, 0, 2]
    assert batch_inputs["symbolic_targets"].tolist()[1] == [3, 2, 2]
    assert batch_inputs["symbolic_mask"].tolist()[1] == [1.0, 1.0, 1.0]


def test_primitive_lane_owned_targets_mask_boolean_and_symbolic_free_mouth_supervision():
    module = _load_stage530()
    rows = [
        {"task_family": "arithmetic_chain"},
        {"task_family": "boolean_logic"},
        {"task_family": "symbolic_binding"},
    ]
    char_targets = torch.tensor(
        [
            [[1, 2, 3]],
            [[4, 5, 6]],
            [[7, 8, 9]],
        ]
    )
    select_targets = torch.tensor([0, 0, 0])

    masked_chars, masked_select = module.mask_primitive_lane_owned_answer_targets(
        rows,
        char_targets,
        select_targets,
        primitive_lane_owns_answer=True,
    )

    assert masked_chars[0].tolist() == [[1, 2, 3]]
    assert masked_select[0].item() == 0
    assert masked_chars[1].tolist() == [[module.IGNORE_INDEX] * 3]
    assert masked_chars[2].tolist() == [[module.IGNORE_INDEX] * 3]
    assert masked_select.tolist() == [0, module.IGNORE_INDEX, module.IGNORE_INDEX]


def test_numeric_procedure_owned_targets_mask_numeric_and_list_free_mouth_supervision():
    module = _load_stage530()
    rows = [
        {"task_family": "arithmetic_chain"},
        {"task_family": "list_transform"},
        {"task_family": "boolean_logic"},
        {"task_family": "symbolic_binding"},
    ]
    char_targets = torch.tensor(
        [
            [[1, 2, 3]],
            [[4, 5, 6]],
            [[7, 8, 9]],
            [[1, 3, 5]],
        ]
    )
    select_targets = torch.tensor([0, 0, 0, 0])

    masked_chars, masked_select = module.mask_primitive_lane_owned_answer_targets(
        rows,
        char_targets,
        select_targets,
        primitive_lane_owns_answer=True,
        numeric_procedure_owns_answer=True,
    )

    assert masked_chars.tolist() == [[[module.IGNORE_INDEX] * 3]] * 4
    assert masked_select.tolist() == [module.IGNORE_INDEX] * 4


def test_procedure_generator_outputs_action_logits_and_has_off_ablation():
    module = _load_stage530()
    torch.manual_seed(0)
    generator = module.ProcedureGenerator(
        d_state=4,
        n_operations=12,
        n_actions=7,
        source_feature_dim=32,
        hidden_dim=8,
    )
    current_state = torch.zeros(2, 5, 4)
    operation_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    source_features = torch.randn(2, 3, 32)
    source_mask = torch.ones(2, 3)

    output = generator(
        current_state,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )
    off = generator(
        current_state,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        procedure_generator_off=True,
    )

    assert output.action_logits.shape == (2, 3, 7)
    assert output.procedure_state.shape == (2, 3, 4)
    assert not torch.allclose(output.action_logits, torch.zeros_like(output.action_logits))
    assert torch.equal(off.action_logits, torch.zeros_like(off.action_logits))
    assert torch.equal(off.procedure_state, torch.zeros_like(off.procedure_state))


def test_learned_numeric_procedure_executor_consumes_generated_actions_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.LearnedNumericProcedureExecutor(
        d_state=4,
        n_actions=7,
        source_feature_dim=32,
        hidden_dim=8,
    )
    digit_trajectory = torch.zeros(2, 3, 5, 4)
    procedure_output = module.ProcedureGeneratorOutput(
        action_logits=torch.randn(2, 2, 7),
        procedure_state=torch.randn(2, 2, 4),
    )
    source_features = torch.randn(2, 3, 32)
    source_mask = torch.ones(2, 3)

    output = executor(
        digit_trajectory,
        procedure_output=procedure_output,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )
    off = executor(
        digit_trajectory,
        procedure_output=procedure_output,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        procedure_executor_off=True,
    )

    assert output.trajectory.shape == digit_trajectory.shape
    assert output.digit_logits.shape == (2, 3, 5, 10)
    assert output.presence_logits.shape == (2, 3, 5)
    assert not torch.allclose(output.trajectory[:, 1], digit_trajectory[:, 1])
    assert torch.equal(off.trajectory, digit_trajectory)
    assert torch.equal(off.digit_logits, torch.zeros_like(off.digit_logits))
    assert torch.equal(off.presence_logits, torch.zeros_like(off.presence_logits))


def test_apply_learned_numeric_procedure_context_wires_generator_executor_and_logits():
    module = _load_stage530()
    torch.manual_seed(0)
    generator = module.ProcedureGenerator(
        d_state=4,
        n_operations=12,
        n_actions=7,
        source_feature_dim=32,
        hidden_dim=8,
    )
    executor = module.LearnedNumericProcedureExecutor(
        d_state=4,
        n_actions=7,
        source_feature_dim=32,
        hidden_dim=8,
    )
    context = {"typed_digit_register_trajectory": torch.zeros(2, 3, 5, 4)}
    operation_ids = torch.tensor([[1, 2], [3, 4]])
    source_features = torch.randn(2, 3, 32)
    source_mask = torch.ones(2, 3)

    patched = module.apply_learned_numeric_procedure_context(
        context,
        procedure_generator=generator,
        learned_numeric_procedure_executor=executor,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        procedure_generator_off=False,
        procedure_executor_off=False,
    )
    off = module.apply_learned_numeric_procedure_context(
        context,
        procedure_generator=generator,
        learned_numeric_procedure_executor=executor,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        procedure_generator_off=True,
        procedure_executor_off=False,
    )

    assert patched["typed_digit_register_trajectory"].shape == context["typed_digit_register_trajectory"].shape
    assert patched["learned_numeric_procedure_action_logits"].shape == (2, 2, 7)
    assert patched["learned_numeric_procedure_digit_logits"].shape == (2, 3, 5, 10)
    assert patched["learned_numeric_procedure_presence_logits"].shape == (2, 3, 5)
    assert not torch.allclose(
        patched["typed_digit_register_trajectory"][:, 1],
        context["typed_digit_register_trajectory"][:, 1],
    )
    assert torch.equal(off["typed_digit_register_trajectory"], context["typed_digit_register_trajectory"])
    assert torch.equal(
        off["learned_numeric_procedure_digit_logits"],
        torch.zeros_like(off["learned_numeric_procedure_digit_logits"]),
    )


def test_lbecp_executor_reads_source_digit_columns_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.LearnedBaseEquivariantColumnProcedureExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        scan_digits=4,
    )
    digit_trajectory = torch.zeros(1, 3, 10, 4)
    operation_ids = torch.tensor([[1, 2]])
    operation_arg_ids = torch.tensor([[3, 4]])
    source_features = torch.zeros(1, 2, 32)
    source_mask = torch.ones(1, 2)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    source_features[0, 0, digit_start + 5] = 7.0 / 9.0
    source_features[0, 0, presence_start + 5] = 1.0
    changed_source = source_features.clone()
    changed_source[0, 0, digit_start + 5] = 2.0 / 9.0

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )
    changed = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=changed_source,
        source_numeric_feature_mask=source_mask,
    )
    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        column_procedure_off=True,
    )

    assert output.trajectory.shape == digit_trajectory.shape
    assert output.digit_logits.shape == (1, 3, 10, 10)
    assert output.presence_logits.shape == (1, 3, 10)
    assert not torch.allclose(output.trajectory[:, 1], digit_trajectory[:, 1])
    assert not torch.allclose(output.trajectory[:, 1], changed.trajectory[:, 1])
    assert torch.equal(off.trajectory, digit_trajectory)
    assert torch.equal(off.digit_logits, torch.zeros_like(off.digit_logits))
    assert torch.equal(off.presence_logits, torch.zeros_like(off.presence_logits))


def test_lbecp_executor_seeds_visible_source_digits_before_rollout():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.LearnedBaseEquivariantColumnProcedureExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        scan_digits=4,
    )
    digit_trajectory = torch.zeros(1, 3, 10, 4)
    operation_ids = torch.tensor([[1, 2]])
    source_features = torch.zeros(1, 2, 32)
    source_mask = torch.ones(1, 2)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    source_features[0, 0, digit_start + 5] = 7.0 / 9.0
    source_features[0, 0, presence_start + 5] = 1.0

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )
    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        column_procedure_off=True,
    )

    assert not torch.allclose(output.trajectory[:, 0], digit_trajectory[:, 0])
    assert torch.equal(off.trajectory, digit_trajectory)


def test_lbecp_executor_commits_predicted_digits_back_to_next_rollout_state():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.LearnedBaseEquivariantColumnProcedureExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        scan_digits=4,
    )
    digit_trajectory = torch.zeros(1, 3, 10, 4)
    operation_ids = torch.tensor([[1, 2]])
    source_features = torch.zeros(1, 2, 32)
    source_mask = torch.ones(1, 2)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    source_features[0, 0, digit_start + 5] = 7.0 / 9.0
    source_features[0, 0, presence_start + 5] = 1.0

    first = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )
    with torch.no_grad():
        executor.digit_state_embed.weight.add_(1.0)
    changed = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    assert not torch.allclose(first.trajectory[:, 2], changed.trajectory[:, 2])


def test_dual_axis_lbecp_compacts_even_value_slots_before_digit_scan():
    module = _load_stage530()
    executor = module.LearnedBaseEquivariantColumnProcedureExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        scan_digits=2,
        value_slot_scan=True,
    )
    columns = torch.zeros(1, 3, 3, 2)
    # Last digit place is index 1 when scan_digits=2. Values: 31, 34, 32.
    columns[0, 0, 0, 0] = 3.0 / 9.0
    columns[0, 0, 1, 0] = 1.0 / 9.0
    columns[0, 0, :2, 1] = 1.0
    columns[0, 1, 0, 0] = 3.0 / 9.0
    columns[0, 1, 1, 0] = 4.0 / 9.0
    columns[0, 1, :2, 1] = 1.0
    columns[0, 2, 0, 0] = 3.0 / 9.0
    columns[0, 2, 1, 0] = 2.0 / 9.0
    columns[0, 2, :2, 1] = 1.0

    compacted = executor._compact_even_value_columns(columns)

    assert torch.allclose(compacted[0, 0, :, 0], columns[0, 1, :, 0])
    assert torch.allclose(compacted[0, 1, :, 0], columns[0, 2, :, 0])
    assert compacted[0, 2, :, 1].sum().item() == 0.0


def test_digit_transition_executor_lbecp_mode_uses_column_procedure():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        executor_mode="lbecp_column",
        scan_digits=4,
    )
    digit_trajectory = torch.zeros(1, 3, 10, 4)
    operation_ids = torch.tensor([[1, 2]])
    operation_arg_ids = torch.tensor([[3, 4]])
    source_features = torch.zeros(1, 2, 32)
    source_mask = torch.ones(1, 2)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    source_features[0, 0, digit_start + 5] = 7.0 / 9.0
    source_features[0, 0, presence_start + 5] = 1.0

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )
    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        column_procedure_off=True,
        return_logits=True,
    )

    assert isinstance(output, module.TypedDigitNextStateOutput)
    assert output.trajectory.shape == digit_trajectory.shape
    assert output.digit_logits.shape == (1, 3, 10, 10)
    assert not torch.allclose(output.trajectory[:, 1], digit_trajectory[:, 1])
    assert torch.equal(off.trajectory, digit_trajectory)
    assert torch.equal(off.digit_logits, torch.zeros_like(off.digit_logits))


def test_digit_transition_executor_dual_axis_lbecp_mode_compacts_list_slots():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        executor_mode="dual_axis_lbecp",
        scan_digits=2,
    )
    digit_trajectory = torch.zeros(1, 3, 9, 4)
    operation_ids = torch.tensor([[module.stage523.TRACE_OPERATION_TO_ID["filter_even"], 0]])
    operation_arg_ids = torch.tensor([[2, 0]])
    source_features = torch.zeros(1, 3, 32)
    source_mask = torch.ones(1, 3)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    for slot, digit in enumerate([1, 4, 2]):
        source_features[0, slot, 1] = 1.0 if digit % 2 == 0 else 0.0
        source_features[0, slot, digit_start + 5] = float(digit) / 9.0
        source_features[0, slot, presence_start + 5] = 1.0

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )
    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        column_procedure_off=True,
        return_logits=True,
    )

    assert isinstance(output, module.TypedDigitNextStateOutput)
    assert output.trajectory.shape == digit_trajectory.shape
    assert not torch.allclose(output.trajectory[:, 1], digit_trajectory[:, 1])
    assert torch.equal(off.trajectory, digit_trajectory)


def test_learned_numeric_procedure_renderer_reads_executor_logits_not_benape_values():
    module = _load_stage530()
    row = {
        "task_family": "arithmetic_chain",
        "question": "Compute ((4007 + 3) * 2) - 3.",
        "answer_aliases": ["WRONG"],
    }
    digit_logits = torch.full((1, 2, 5, 10), -8.0)
    presence_logits = torch.full((1, 2, 5), -8.0)
    for slot, digit in enumerate([1, 2, 3, 4]):
        digit_logits[0, -1, slot, digit] = 8.0
        presence_logits[0, -1, slot] = 8.0
    output = module.TypedDigitNextStateOutput(
        trajectory=torch.zeros(1, 2, 5, 4),
        digit_logits=digit_logits,
        presence_logits=presence_logits,
    )

    rendered, mask = module.render_learned_numeric_procedure_texts(
        [row],
        output,
        max_digits=4,
    )
    rendered_off, mask_off = module.render_learned_numeric_procedure_texts(
        [row],
        output,
        max_digits=4,
        numeric_procedure_off=True,
    )

    assert rendered == ["1234"]
    assert mask.tolist() == [True]
    assert rendered_off == [""]
    assert mask_off.tolist() == [False]


def test_cross_entropy_ignore_or_zero_returns_zero_when_every_target_is_ignored():
    module = _load_stage530()
    logits = torch.randn(3, 5)
    targets = torch.full((3,), module.IGNORE_INDEX)

    loss = module.cross_entropy_ignore_or_zero(logits, targets)

    assert loss.item() == 0.0


def test_benape_primitive_solves_arithmetic_chain_from_source_digit_columns():
    module = _load_stage530()
    row = {
        "task_family": "arithmetic_chain",
        "question": "Compute ((4007 + 3) * 2) - 3.",
        "answer_aliases": ["8017"],
    }
    source_features, source_mask = module.stage523.source_number_feature_tensors(
        [row],
        max_slots=8,
        feature_dim=32,
        value_scale=10000.0,
        device=torch.device("cpu"),
    )
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        executor_mode="benape_primitive",
        scan_digits=6,
    )
    digit_trajectory = torch.zeros(1, 4, 8 * 7, 4)
    operation_ids = torch.tensor(
        [[
            module.stage523.TRACE_OPERATION_TO_ID["add_operands"],
            module.stage523.TRACE_OPERATION_TO_ID["multiply_sum"],
            module.stage523.TRACE_OPERATION_TO_ID["subtract_offset"],
        ]]
    )
    operation_arg_ids = torch.tensor([[3, 2, 3]])

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )
    rendered, mask = module.render_ledger_pact_texts(
        [row],
        output.digit_logits,
        output.presence_logits,
        max_digits=6,
    )

    assert rendered == ["8017"]
    assert mask.tolist() == [True]


def test_benape_primitive_filters_and_doubles_list_from_source_digit_columns():
    module = _load_stage530()
    row = {
        "task_family": "list_transform",
        "question": (
            "From the list [4001, 4004, 4002, 4007, 4003], keep only even numbers, "
            "double each kept number, and return comma-separated values with no spaces."
        ),
        "answer_aliases": ["8008,8004"],
    }
    source_features, source_mask = module.stage523.source_number_feature_tensors(
        [row],
        max_slots=8,
        feature_dim=32,
        value_scale=10000.0,
        device=torch.device("cpu"),
    )
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        executor_mode="benape_primitive",
        scan_digits=6,
    )
    digit_trajectory = torch.zeros(1, 3, 8 * 7, 4)
    operation_ids = torch.tensor(
        [[
            module.stage523.TRACE_OPERATION_TO_ID["filter_even"],
            module.stage523.TRACE_OPERATION_TO_ID["double_filtered"],
        ]]
    )
    operation_arg_ids = torch.tensor([[2, 2]])

    output = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )
    rendered, mask = module.render_ledger_pact_texts(
        [row],
        output.digit_logits,
        output.presence_logits,
        max_digits=6,
    )

    assert rendered == ["8008,8004"]
    assert mask.tolist() == [True]


def test_typed_digit_next_state_executor_column_scan_couples_neighbor_digit_columns():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=0,
        hidden_dim=8,
        executor_mode="column_scan",
        scan_digits=4,
    )
    base = torch.zeros(1, 2, 5, 4)
    changed_rightmost = base.clone()
    changed_rightmost[:, 0, 3, 0] = 1.0
    operation_ids = torch.tensor([[1]])
    operation_arg_ids = torch.tensor([[7]])

    base_output = executor(
        base,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
    )
    changed_output = executor(
        changed_rightmost,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
    )

    assert base_output.shape == changed_output.shape == base.shape
    assert not torch.allclose(base_output[:, 1, 2], changed_output[:, 1, 2])


def test_typed_digit_next_state_executor_becto_reads_source_digit_columns_and_is_ablatable():
    module = _load_stage530()
    torch.manual_seed(0)
    executor = module.TypedDigitNextStateExecutor(
        d_state=4,
        n_operations=12,
        source_feature_dim=32,
        hidden_dim=8,
        executor_mode="becto_column",
        scan_digits=4,
    )
    digit_trajectory = torch.zeros(1, 2, 10, 4)
    operation_ids = torch.tensor([[1]])
    operation_arg_ids = torch.tensor([[3]])
    source_features = torch.zeros(1, 2, 32)
    source_mask = torch.ones(1, 2)
    digit_feature_count = 6
    digit_start = 14
    presence_start = digit_start + digit_feature_count
    source_features[0, 0, digit_start + 5] = 7.0 / 9.0
    source_features[0, 0, presence_start + 5] = 1.0

    changed_source = source_features.clone()
    changed_source[0, 0, digit_start + 5] = 2.0 / 9.0

    off = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        column_procedure_off=True,
        return_logits=True,
    )
    on = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )
    changed = executor(
        digit_trajectory,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=changed_source,
        source_numeric_feature_mask=source_mask,
        return_logits=True,
    )

    assert torch.equal(off.trajectory, digit_trajectory)
    assert torch.equal(off.digit_logits, torch.zeros_like(off.digit_logits))
    assert on.trajectory.shape == digit_trajectory.shape
    assert not torch.allclose(on.trajectory[:, 1], digit_trajectory[:, 1])
    assert not torch.allclose(on.trajectory[:, 1], changed.trajectory[:, 1])


def test_compute_digit_transition_executor_trace_loss_reads_direct_logits():
    module = _load_stage530()
    digit_logits = torch.zeros(1, 2, 5, 10)
    presence_logits = torch.zeros(1, 2, 5)
    digit_logits[0, 1, 0, 8] = 5.0
    digit_logits[0, 1, 1, 0] = 5.0
    digit_logits[0, 1, 2, 1] = 5.0
    digit_logits[0, 1, 3, 7] = 5.0
    presence_logits[0, 1, :4] = 5.0
    presence_logits[0, 1, 4] = -5.0
    batch_indices = torch.tensor([0])
    step_indices = torch.tensor([1])
    digit_targets = torch.tensor([[[8, 0, 1, 7]]])
    presence_targets = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

    good_loss, good_metrics = module.compute_digit_transition_executor_trace_loss(
        digit_logits,
        presence_logits,
        batch_indices,
        step_indices,
        digit_targets,
        presence_targets,
        max_digits=4,
    )
    bad_loss, _bad_metrics = module.compute_digit_transition_executor_trace_loss(
        digit_logits,
        presence_logits,
        batch_indices,
        step_indices,
        torch.tensor([[[7, 1, 0, 8]]]),
        presence_targets,
        max_digits=4,
    )

    assert good_loss.item() < bad_loss.item()
    assert good_metrics["digit_accuracy"] == 1.0


def test_residual_thought_graft_starts_identity_and_is_ablatable():
    module = _load_stage530()
    graft = module.ResidualThoughtGraft(d_state=4, hidden_dim=3)
    context = {
        "readout": torch.randn(2, 4),
        "working_register_trajectory": torch.randn(2, 3, 5, 4),
        "typed_digit_register_trajectory": torch.randn(2, 3, 6, 4),
    }

    identity = module.apply_residual_thought_graft_context(
        context,
        residual_thought_graft=graft,
        graft_off=False,
        graft_base="qtrm_readout",
    )
    off = module.apply_residual_thought_graft_context(
        context,
        residual_thought_graft=graft,
        graft_off=True,
    )

    assert torch.equal(identity["readout"], context["readout"])
    assert torch.equal(off["readout"], context["readout"])

    with torch.no_grad():
        graft.up.weight.fill_(0.25)

    on = module.apply_residual_thought_graft_context(
        context,
        residual_thought_graft=graft,
        graft_off=False,
        graft_base="qtrm_readout",
    )
    off_after_update = module.apply_residual_thought_graft_context(
        context,
        residual_thought_graft=graft,
        graft_off=True,
    )

    assert not torch.allclose(on["readout"], context["readout"])
    assert torch.equal(off_after_update["readout"], context["readout"])
    assert torch.equal(on["working_register_trajectory"], context["working_register_trajectory"])
    assert torch.equal(on["typed_digit_register_trajectory"], context["typed_digit_register_trajectory"])


def test_residual_thought_graft_requires_qwen_compatible_readout_shape():
    module = _load_stage530()
    graft = module.ResidualThoughtGraft(d_state=4, hidden_dim=3)
    context = {
        "readout": torch.randn(2, 1, 4),
        "working_register_trajectory": torch.randn(2, 3, 5, 4),
    }

    try:
        module.apply_residual_thought_graft_context(
            context,
            residual_thought_graft=graft,
            graft_off=False,
            graft_base="qtrm_readout",
        )
    except ValueError as exc:
        assert "readout" in str(exc)
    else:
        raise AssertionError("expected invalid readout shape to fail")


def test_residual_thought_graft_validation_allows_register_mean_answer_path():
    module = _load_stage530()
    args = SimpleNamespace(
        residual_thought_graft=True,
        answerer_use_qtrm_readout=False,
        residual_thought_graft_base="qtrm_readout",
    )

    try:
        module.validate_residual_thought_graft_flags(args)
    except SystemExit as exc:
        assert "--residual-thought-graft-base qtrm_readout requires --answerer-use-qtrm-readout" in str(exc)
    else:
        raise AssertionError("expected qtrm-readout graft without qtrm answer path to fail validation")

    args.residual_thought_graft_base = "register_mean"
    module.validate_residual_thought_graft_flags(args)


def test_residual_thought_graft_register_mean_base_sets_answer_readout():
    module = _load_stage530()
    graft = module.ResidualThoughtGraft(d_state=4, hidden_dim=3)
    register_trajectory = torch.randn(2, 3, 5, 4)
    context = {
        "readout": torch.randn(2, 4),
        "working_register_trajectory": register_trajectory,
        "typed_digit_register_trajectory": torch.randn(2, 3, 6, 4),
    }

    grafted = module.apply_residual_thought_graft_context(
        context,
        residual_thought_graft=graft,
        graft_off=False,
        graft_base="register_mean",
    )
    expected = register_trajectory[:, -1].mean(dim=1)

    assert torch.equal(grafted["residual_thought_graft_readout"], expected)
    assert torch.equal(
        module.answer_readout_from_context(
            grafted,
            typed_register_off=False,
            use_qtrm_readout=False,
        ),
        expected,
    )


def test_residual_thought_graft_active_respects_ablation_and_typed_register_off():
    module = _load_stage530()
    args = SimpleNamespace(residual_thought_graft=True)

    assert module.residual_thought_graft_active(
        args,
        typed_register_off=False,
        residual_thought_graft_off=False,
    )
    assert not module.residual_thought_graft_active(
        args,
        typed_register_off=True,
        residual_thought_graft_off=False,
    )
    assert not module.residual_thought_graft_active(
        args,
        typed_register_off=False,
        residual_thought_graft_off=True,
    )


def test_qwen_lm_answer_mouth_uses_qwen_lm_head_from_grafted_readout():
    module = _load_stage530()

    class DummyQtrm:
        def __init__(self):
            self.seen_state = None
            self.proj = torch.nn.Linear(4, 7, bias=False)

        def _lm_head_logits_from_state(self, state):
            self.seen_state = state
            return state, self.proj(state)

    qtrm = DummyQtrm()
    mouth = module.QwenLmAnswerMouth(
        d_state=4,
        vocab_size=7,
        max_answer_tokens=3,
        logit_mode="qwen_lm_head",
    )
    grafted = torch.randn(2, 4)
    context = {
        "readout": torch.randn(2, 4),
        "residual_thought_graft_readout": grafted,
        "working_register_trajectory": torch.randn(2, 3, 5, 4),
    }

    logits = module.qwen_lm_mouth_forward(
        qtrm,
        mouth,
        context,
        typed_register_off=False,
    )

    assert logits.shape == (2, 3, 7)
    assert qtrm.seen_state is not None
    assert qtrm.seen_state.shape == (2, 3, 4)


def test_qwen_lm_answer_targets_use_tokenizer_eos():
    module = _load_stage530()

    class DummyTokenizer:
        eos_token_id = 99

        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 10 for char in text]

    rows = [
        {"answer_aliases": ["A7"]},
        {"answer": "B"},
    ]

    targets = module.encode_qwen_lm_answer_targets(
        DummyTokenizer(),
        rows,
        max_answer_tokens=4,
        device=torch.device("cpu"),
    )

    assert targets.tolist() == [
        [5, 5, 99, module.IGNORE_INDEX],
        [6, 99, module.IGNORE_INDEX, module.IGNORE_INDEX],
    ]


def test_qwen_lm_answer_target_validation_rejects_truncation():
    module = _load_stage530()

    class DummyTokenizer:
        eos_token_id = 99

        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 10 for char in text]

    rows = [
        {"answer_aliases": ["ABCDE"]},
        {"answer": "B"},
    ]

    try:
        module.validate_qwen_lm_answer_target_lengths(
            DummyTokenizer(),
            rows,
            max_answer_tokens=4,
            split_name="train",
        )
    except SystemExit as exc:
        message = str(exc)
        assert "train rows contain 1 answer(s) longer than --qwen-lm-mouth-max-answer-tokens=4" in message
        assert "ABCDE requires 6 tokens including EOS" in message
    else:
        raise AssertionError("expected overlong qwen lm mouth answer target to fail validation")


def test_qwen_lm_mouth_validation_rejects_generated_choice_verifier():
    module = _load_stage530()
    args = SimpleNamespace(
        qwen_lm_mouth_answerer=True,
        answerer_choice_verifier=True,
        choice_verifier_candidate_source="generated",
    )

    try:
        module.validate_qwen_lm_mouth_flags(args)
    except SystemExit as exc:
        assert "--qwen-lm-mouth-answerer cannot use generated choice verifier candidates" in str(exc)
    else:
        raise AssertionError("expected generated choice verifier with qwen lm mouth to fail validation")


def test_qwen_lm_answer_mouth_register_prefix_reads_full_register_memory():
    module = _load_stage530()

    class DummyQtrm:
        def __init__(self):
            self.proj = torch.nn.Linear(4, 7, bias=False)

        def _lm_head_logits_from_state(self, state):
            return state, self.proj(state)

    mouth = module.QwenLmAnswerMouth(
        d_state=4,
        vocab_size=7,
        max_answer_tokens=3,
        logit_mode="qwen_lm_head",
        context_mode="register_prefix",
        n_heads=2,
    )
    qtrm = DummyQtrm()
    context = {
        "readout": torch.randn(2, 4),
        "residual_thought_graft_readout": torch.randn(2, 4),
        "working_register_trajectory": torch.randn(2, 3, 5, 4),
        "typed_digit_register_trajectory": torch.randn(2, 3, 6, 4),
    }
    changed_context = dict(context)
    changed_context["working_register_trajectory"] = context["working_register_trajectory"] + 5.0

    logits = module.qwen_lm_mouth_forward(
        qtrm,
        mouth,
        context,
        typed_register_off=False,
    )
    changed_logits = module.qwen_lm_mouth_forward(
        qtrm,
        mouth,
        changed_context,
        typed_register_off=False,
    )

    assert logits.shape == (2, 3, 7)
    assert not torch.allclose(logits, changed_logits)


def test_qwen_lm_answer_mouth_ledger_token_reader_adds_digit_logits_and_turns_off():
    module = _load_stage530()

    class DummyQtrm:
        def _lm_head_logits_from_state(self, state):
            return state, state.new_zeros(state.size(0), state.size(1), 12)

    qtrm = DummyQtrm()
    mouth = module.QwenLmAnswerMouth(
        d_state=4,
        vocab_size=12,
        max_answer_tokens=2,
        logit_mode="qwen_lm_head",
        ledger_token_reader=True,
        ledger_token_reader_scale=2.0,
        digit_token_ids=list(range(10)),
        comma_token_id=10,
        eos_token_id=11,
    )
    with torch.no_grad():
        mouth.ledger_token_digit_head.weight.zero_()
        mouth.ledger_token_digit_head.bias.zero_()
        mouth.ledger_token_digit_head.bias[3] = 1.5
        mouth.ledger_token_control_head.weight.zero_()
        mouth.ledger_token_control_head.bias.zero_()

    context = {
        "readout": torch.randn(2, 4),
        "residual_thought_graft_readout": torch.randn(2, 4),
        "working_register_trajectory": torch.randn(2, 3, 5, 4),
        "typed_digit_register_trajectory": torch.randn(2, 3, 7, 4),
    }

    logits = module.qwen_lm_mouth_forward(
        qtrm,
        mouth,
        context,
        typed_register_off=False,
        ledger_token_reader_off=False,
    )
    off_logits = module.qwen_lm_mouth_forward(
        qtrm,
        mouth,
        context,
        typed_register_off=False,
        ledger_token_reader_off=True,
    )

    assert torch.allclose(logits[..., 3], torch.full_like(logits[..., 3], 3.0))
    assert torch.allclose(off_logits, torch.zeros_like(off_logits))


def test_qwen_lm_mouth_direct_ledger_renderer_diagnoses_numeric_ledger_before_mouth():
    module = _load_stage530()

    class DummyEval:
        def eval(self):
            return self

    digit_logits = torch.zeros(2, 1, 5, 10)
    for slot, digit in enumerate([8, 0, 1, 7]):
        digit_logits[0, 0, slot, digit] = 9.0
    presence_logits = torch.full((2, 1, 5), -9.0)
    presence_logits[0, 0, :4] = 9.0

    old_context_fn = module.stage523.thought_context_for_batch

    def fake_context(*_args, **_kwargs):
        return {
            "readout": torch.zeros(2, 4),
            "working_register_trajectory": torch.zeros(2, 1, 1, 4),
            "digit_transition_executor_digit_logits": digit_logits,
            "digit_transition_executor_presence_logits": presence_logits,
        }

    module.stage523.thought_context_for_batch = fake_context
    try:
        summary, records = module.evaluate(
            wgram_model=DummyEval(),
            tokenizer=object(),
            answerer=DummyEval(),
            digit_transition_executor=None,
            procedure_generator=None,
            learned_numeric_procedure_executor=None,
            typed_primitive_lane_executor=None,
            choice_verifier=None,
            digit_ledger_attractor=None,
            digit_committed_writeback=None,
            residual_thought_graft=None,
            qwen_lm_mouth=DummyEval(),
            rows=[
                {
                    "id": "arith",
                    "task_family": "arithmetic_chain",
                    "answer_aliases": ["8017"],
                },
                {
                    "id": "bool",
                    "task_family": "boolean_logic",
                    "answer_aliases": ["TRUE"],
                },
            ],
            allowed_chars=list("0123456789,TRUEFALS"),
            symbol_vocab=[],
            args=SimpleNamespace(
                eval_batch_size=2,
                max_length=8,
                n_steps=1,
                condition_on_trace_operations=False,
                use_source_number_slots=False,
                source_number_slots=0,
                source_number_feature_dim=0,
                source_number_value_scale=1.0,
                qwen_lm_mouth_answerer=True,
                typed_digit_register_digits=4,
            ),
            device=torch.device("cpu"),
            qwen_lm_mouth_direct_ledger_renderer=True,
        )
    finally:
        module.stage523.thought_context_for_batch = old_context_fn

    assert summary["qwen_lm_mouth_direct_ledger_renderer"]
    assert records[0]["selection_mode"] == "qwen_lm_mouth_direct_ledger_renderer"
    assert records[0]["selected"] == "8017"
    assert records[0]["exact"]
    assert records[1]["selected"] == ""
    assert summary["by_family"]["arithmetic_chain"]["accuracy"] == 1.0
    assert summary["by_family"]["boolean_logic"]["accuracy"] == 0.0
