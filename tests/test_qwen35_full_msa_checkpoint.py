from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from wgram_lm.qwen35_full_msa import (
    Qwen35FullMsaForCausalLM,
    load_qwen35_full_msa_checkpoint,
    save_qwen35_full_msa_checkpoint,
)


def tiny_config() -> Qwen3_5TextConfig:
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["sparse", "sparse"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    cfg.qtrm_original_layer_types = ["linear_attention", "full_attention"]
    cfg.qtrm_full_msa_fork = True
    cfg.msa_config = {
        "top_k_docs": 1,
        "pooling_kernel_size": 2,
        "head_reduce_method": "mean",
        "query_reduce_method": "max",
        "chunk_reduce_method": "max",
        "decouple_router": True,
        "aux_loss_method": "INFONCE",
    }
    return cfg


def test_qwen35_full_msa_checkpoint_round_trips_logits(tmp_path) -> None:
    torch.manual_seed(7)
    model = Qwen35FullMsaForCausalLM(tiny_config())
    model.eval()
    input_ids = torch.tensor([[11, 12, 21, 22, 31, 32]])
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.tensor([[1, 1, 2, 2, 0, 0]])

    with torch.no_grad():
        before = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            doc_ids=doc_ids,
        ).logits

    save_qwen35_full_msa_checkpoint(
        model,
        tmp_path,
        metadata={"source": "unit-test", "stage": "roundtrip"},
    )
    loaded = load_qwen35_full_msa_checkpoint(tmp_path, map_location="cpu")
    loaded.eval()

    with torch.no_grad():
        after = loaded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            doc_ids=doc_ids,
        ).logits

    assert torch.allclose(before, after)
    assert loaded.config.qtrm_full_msa_fork is True
    assert loaded.config.msa_config["top_k_docs"] == 1
    assert (tmp_path / "qtrm_full_msa_checkpoint.json").exists()
