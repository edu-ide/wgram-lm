import importlib.util
import json
import sys
from pathlib import Path


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self):
        self.last_text = None

    def __call__(
        self,
        text,
        *,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None,
        add_special_tokens=False,
    ):
        import torch

        self.last_text = text
        ids = [min(98, (ord(ch) % 90) + 1) for ch in text]
        ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad
            mask = mask + [0] * pad
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def load_training_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "511_train_qwen_state_transition_hrmtext.py"
    spec = importlib.util.spec_from_file_location("train_511_hrmtext", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_reasoning_prompt_contains_operands_and_operations():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.SyntheticDataset(tokenizer, count=1, seed=3, max_length=96)
    row = dataset[0]

    prompt = row["prompt_text"]
    assert "start=" in prompt
    assert "ops=" in prompt
    assert any(name in prompt for name in ("add_a=", "digits=", "a="))
    assert row["input_ids"].shape[0] == 96
    assert row["state_labels"].shape[0] == 4


def test_generalized_synthetic_schema_matches_eval_surface_and_pads_depths():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.SyntheticDataset(
        tokenizer,
        count=32,
        seed=7,
        max_length=160,
        schema="generalized",
        depths=[4, 6, 8],
        max_operation_steps=8,
        condition_prefix="synth",
    )

    depths = {dataset[i]["depth"] for i in range(len(dataset))}
    assert depths <= {4, 6, 8}
    assert len(depths) >= 2

    row = next(dataset[i] for i in range(len(dataset)) if dataset[i]["depth"] < 8)
    prompt = row["prompt_text"]
    assert prompt.startswith("Condition: synth,")
    assert "Reasoning task:" in prompt
    assert "Return the final digit.\nAnswer:" in prompt
    assert row["operation_ids"].shape[0] == 8
    assert row["state_labels"].shape[0] == 8
    assert row["operation_ids"][row["depth"] :].eq(module.OP_TO_ID["copy"]).all()
    assert row["state_labels"][row["depth"] :].eq(row["reasoning_labels"]).all()


def test_generalized_synthetic_schema_can_balance_family_mix():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.SyntheticDataset(
        tokenizer,
        count=64,
        seed=11,
        max_length=160,
        schema="generalized",
        depths=[4, 6],
        max_operation_steps=6,
        family_mix="checksum2_chain1",
    )

    families = [dataset[i]["family"] for i in range(len(dataset))]
    assert "checksum" in families
    assert "chain" in families
    assert families.count("checksum") > families.count("chain")


def test_generalized_synthetic_schema_can_stratify_depth_and_family_mix():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.SyntheticDataset(
        tokenizer,
        count=12,
        seed=13,
        max_length=160,
        schema="generalized",
        depths=[4, 6],
        max_operation_steps=6,
        family_mix="balanced",
        sampling_strategy="stratified",
    )

    depths = [dataset[i]["depth"] for i in range(len(dataset))]
    families = [dataset[i]["family"] for i in range(len(dataset))]
    assert depths.count(4) == depths.count(6)
    assert families.count("chain") == families.count("checksum")


def test_generalized_synthetic_schema_can_use_depth_family_pattern():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.SyntheticDataset(
        tokenizer,
        count=8,
        seed=17,
        max_length=160,
        schema="generalized",
        depths=[4, 6, 8],
        max_operation_steps=8,
        sampling_strategy="stratified",
        depth_family_pattern=module.parse_depth_family_pattern(
            ["chain:4", "chain:6", "checksum:8", "checksum:8"]
        ),
    )

    depth_family = [(dataset[i]["depth"], dataset[i]["family"]) for i in range(len(dataset))]
    assert depth_family == [
        (4, "chain"),
        (6, "chain"),
        (8, "checksum"),
        (8, "checksum"),
        (4, "chain"),
        (6, "chain"),
        (8, "checksum"),
        (8, "checksum"),
    ]


def test_healing_dataset_uses_prefix_response_boundary_and_response_only_labels():
    module = load_training_module()
    tokenizer = TinyTokenizer()

    dataset = module.HRMTextHealingDataset(
        tokenizer,
        rows=[
            {
                "condition": "synth,cot",
                "instruction": "Add 2 and 3.",
                "response": "5678",
            }
        ],
        max_length=96,
        target_tokens=3,
    )
    row = dataset[0]

    assert row["is_healing"] is True
    assert row["prefix_len"].item() > 0
    assert row["response_start"].item() == row["prefix_len"].item()
    assert row["labels"][: row["prefix_len"].item()].eq(module.IGNORE_INDEX).all()
    assert row["labels"][row["prefix_len"].item() :].ne(module.IGNORE_INDEX).any()
    assert row["healing_target_ids"].shape[0] == 3
    assert row["healing_target_ids"].ne(module.IGNORE_INDEX).sum().item() == 3
    assert row["token_type_ids"][: row["prefix_len"].item()].eq(1).all()
    assert row["condition"] == "synth,cot"


def test_load_hrm_text_rows_from_cleaned_jsonl_path(tmp_path):
    module = load_training_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "gsm8k_train.jsonl"
    rows = [
        {"condition": "cot", "instruction": "What is 2+3?", "response": "5"},
        {"condition": "direct", "instruction": "", "response": "skip"},
        {"condition": "direct", "instruction": "Say hi", "response": "Hi"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = module.load_hrm_text_rows_from_path(str(tmp_path), count=4, seed=1, include_globs=["data/*.jsonl"])

    assert loaded == [
        {"condition": "cot", "instruction": "What is 2+3?", "response": "5"},
        {"condition": "direct", "instruction": "Say hi", "response": "Hi"},
    ]


def test_fit_step_sequence_pads_or_truncates_to_requested_depth():
    module = load_training_module()
    import torch

    values = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    assert torch.equal(module.fit_step_sequence(values, 2), torch.tensor([[0, 1], [4, 5]]))
    assert torch.equal(
        module.fit_step_sequence(values, 6),
        torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 7, 7, 7]]),
    )


def test_step_answer_auxiliary_loss_supervises_each_transition_state():
    module = load_training_module()
    import torch
    from torch import nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core_out_norm = nn.Identity()
            self.answer_head = nn.Identity()

    labels = torch.tensor([[2, 4, 7], [1, 3, 5]])
    trajectory = torch.zeros((2, 4, 10), dtype=torch.float32)
    for batch_idx in range(labels.size(0)):
        for step_idx in range(labels.size(1)):
            trajectory[batch_idx, step_idx + 1, labels[batch_idx, step_idx]] = 20.0

    loss, logits = module.compute_step_answer_loss(DummyModel(), trajectory, labels)

    assert logits.shape == (2, 3, 10)
    assert loss.item() < 1e-3


def test_operation_supervision_loss_supervises_each_transition_operation():
    module = load_training_module()
    import torch

    operation_ids = torch.tensor([[0, 1, 2], [3, 0, 1]])
    logits = torch.zeros((2, 3, 4), dtype=torch.float32)
    for batch_idx in range(operation_ids.size(0)):
        for step_idx in range(operation_ids.size(1)):
            logits[batch_idx, step_idx, operation_ids[batch_idx, step_idx]] = 20.0

    loss = module.compute_operation_supervision_loss(logits, operation_ids)

    assert loss.item() < 1e-3


def test_depth_consistency_loss_uses_detached_teacher_distribution():
    module = load_training_module()
    import torch

    teacher_logits = torch.tensor([[4.0, 0.0, -2.0]], requires_grad=True)
    student_logits = torch.tensor([[0.0, 3.0, -1.0]], requires_grad=True)

    loss = module.compute_depth_consistency_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=1.0,
    )
    loss.backward()

    assert loss.item() > 0
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum().item() > 0
    assert teacher_logits.grad is None


def test_trajectory_anchor_loss_distills_teacher_direction_without_teacher_grad():
    module = load_training_module()
    import torch

    trajectory = torch.zeros((2, 4, 3), dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    loss = module.compute_trajectory_anchor_loss(
        state_trajectory=trajectory + teacher.detach().unsqueeze(1),
        teacher_state=teacher,
        min_step=1,
    )
    loss.backward()

    assert loss.item() < 1e-6
    assert trajectory.grad is not None
    assert teacher.grad is None
