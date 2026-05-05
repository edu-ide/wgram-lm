from __future__ import annotations

from pathlib import Path

from qtrm_mm.msa_qwen35 import (
    build_qwen35_full_msa_fork,
    load_json,
    write_fork_artifacts,
)


def test_qwen35_full_msa_fork_converts_all_text_layers() -> None:
    source = load_json("references/model_configs/qwen35_2b_base/config.json")

    plan = build_qwen35_full_msa_fork(source)

    text_cfg = plan.config["text_config"]
    assert plan.config["model_type"] == "qwen3_5_full_msa_fork"
    assert plan.config["architectures"] == ["Qwen3_5FullMSAForConditionalGeneration"]
    assert text_cfg["qtrm_full_msa_fork"] is True
    assert set(text_cfg["layer_types"]) == {"sparse"}
    assert len(text_cfg["layer_types"]) == text_cfg["num_hidden_layers"]
    assert len(plan.manifest["linear_attention_layers_replaced_by_msa"]) == 18
    assert len(plan.manifest["full_attention_layers_reused_as_msa_seed"]) == 6
    assert text_cfg["msa_config"]["top_k_docs"] == 8
    assert text_cfg["msa_config"]["aux_loss_method"] == "INFONCE"


def test_qwen35_full_msa_fork_writes_artifacts(tmp_path: Path) -> None:
    source_path = Path("references/model_configs/qwen35_2b_base/config.json")
    plan = build_qwen35_full_msa_fork(load_json(source_path))

    write_fork_artifacts(tmp_path, plan, source_config_path=source_path)

    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "conversion_manifest.json").is_file()
    readme = (tmp_path / "README.md").read_text()
    assert "Qwen3.5-2B Full-MSA Fork Plan" in readme
    assert "Linear-attention layers replaced by MSA" in readme
