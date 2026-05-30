import torch

from qtrm_mm.qtrm_model import compute_donor_qtrm_conflict_gate


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_conflict_gate_downscale_mode_matches_legacy_behavior() -> None:
    qtrm_logits = torch.tensor([[[0.0, 4.0, 1.0], [3.0, 0.0, 1.0]]])
    donor_logits = torch.tensor([[[5.0, 0.0, 1.0], [2.0, 0.0, 1.0]]])

    gate = compute_donor_qtrm_conflict_gate(
        qtrm_logits,
        donor_logits,
        enabled=True,
        mode="downscale",
        conflict_scale=0.25,
        boost_scale=1.0,
        margin_threshold=0.0,
    )

    assert_close(gate, torch.tensor([[0.25, 1.0]]))


def test_conflict_gate_adaptive_margin_preserves_stronger_qtrm_signal() -> None:
    qtrm_logits = torch.tensor([[[0.0, 7.0, 1.0], [3.0, 0.0, 1.0]]])
    donor_logits = torch.tensor([[[4.0, 0.0, 1.0], [0.0, 5.0, 1.0]]])

    gate = compute_donor_qtrm_conflict_gate(
        qtrm_logits,
        donor_logits,
        enabled=True,
        mode="adaptive_margin",
        conflict_scale=0.25,
        boost_scale=1.0,
        margin_threshold=0.0,
    )

    assert_close(gate, torch.tensor([[1.0, 0.25]]))


def test_conflict_gate_adaptive_margin_can_boost_clear_qtrm_wins() -> None:
    qtrm_logits = torch.tensor([[[0.0, 9.0, 1.0]]])
    donor_logits = torch.tensor([[[4.0, 0.0, 1.0]]])

    gate = compute_donor_qtrm_conflict_gate(
        qtrm_logits,
        donor_logits,
        enabled=True,
        mode="adaptive_margin",
        conflict_scale=0.25,
        boost_scale=2.0,
        margin_threshold=0.0,
    )

    assert_close(gate, torch.tensor([[2.0]]))


if __name__ == "__main__":
    test_conflict_gate_downscale_mode_matches_legacy_behavior()
    test_conflict_gate_adaptive_margin_preserves_stronger_qtrm_signal()
    test_conflict_gate_adaptive_margin_can_boost_clear_qtrm_wins()
