from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from qtrm_mm.qwen35_full_msa_healing import (
    build_tiny_healing_models,
    freeze_for_stage1_healing,
    qwen35_full_msa_healing_loss,
    run_tiny_healing_smoke,
    synthetic_doc_batch,
)


def test_stage1_healing_freezes_stable_backbone_and_trains_msa_attention() -> None:
    _, target = build_tiny_healing_models(seed=7)

    summary = freeze_for_stage1_healing(target)

    trainable_names = [name for name, p in target.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in target.named_parameters() if not p.requires_grad]
    assert summary["trainable_param_count"] > 0
    assert summary["frozen_param_count"] > summary["trainable_param_count"]
    assert any(".self_attn." in name for name in trainable_names)
    assert any("router_q_proj" in name for name in trainable_names)
    assert not any(".mlp." in name for name in trainable_names)
    assert any(".mlp." in name for name in frozen_names)


def test_healing_loss_is_finite_for_teacher_kl_and_lm() -> None:
    source, target = build_tiny_healing_models(seed=11)
    batch = synthetic_doc_batch(batch_size=2, seq_len=8, vocab_size=128)

    loss, metrics = qwen35_full_msa_healing_loss(
        source,
        target,
        batch,
        lm_weight=1.0,
        donor_kl_weight=0.5,
        temperature=1.5,
    )

    assert torch.isfinite(loss)
    assert metrics["lm_loss"] > 0
    assert metrics["donor_kl"] >= 0
    assert metrics["loss"] > 0


def test_tiny_healing_smoke_writes_report_and_updates_trainable_params(tmp_path: Path) -> None:
    report = run_tiny_healing_smoke(
        out_dir=tmp_path,
        steps=2,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        seed=13,
    )

    report_path = tmp_path / "healing_report.json"
    assert report_path.is_file()
    written = json.loads(report_path.read_text())
    assert written["steps"] == 2
    assert report["updated_trainable_l1"] > 0
    assert report["freeze_summary"]["trainable_param_count"] > 0
    assert report["final_metrics"]["loss"] > 0
