from __future__ import annotations

import yaml


def test_hf_distill_manifest_lists_first_wave_sources() -> None:
    manifest = yaml.safe_load(open("configs/hf_distill_datasets.yaml"))  # noqa: PTH123
    datasets = {item["name"]: item for item in manifest["datasets"]}

    assert "yana_reasoning_dpo" in datasets
    assert "noesis_50k_reasoning_sft" in datasets
    assert "ragognize_evidence" in datasets
    assert "halluclaim_76k" in datasets
    assert datasets["yana_reasoning_dpo"]["adapter"] == "yana_reasoning_dpo"
    assert datasets["ragognize_evidence"]["role"] == "msa_routing_evidence_gate"
    assert datasets["halluclaim_76k"]["priority"] == 1


def test_manifest_entries_have_required_fields() -> None:
    manifest = yaml.safe_load(open("configs/hf_distill_datasets.yaml"))  # noqa: PTH123

    for item in manifest["datasets"]:
        assert item["name"]
        assert item["hf_id"]
        assert item["split"]
        assert item["adapter"]
        assert item["role"]
        assert item["expected_columns"]
        assert item["target_outputs"]
