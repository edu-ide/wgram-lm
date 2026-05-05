from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from qtrm_mm.qwen35_full_msa import Qwen35FullMsaTextModel


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


def test_qwen35_full_msa_text_model_runs_doc_ids_forward() -> None:
    model = Qwen35FullMsaTextModel(tiny_config())
    input_ids = torch.tensor([[11, 12, 13, 21, 22, 31, 32, 2]])
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0, 0]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, doc_ids=doc_ids)

    assert out.last_hidden_state.shape == (1, 8, 32)
    assert len(out.selected_doc_ids_by_layer) == 2
    assert len(out.selected_doc_ids_by_layer[0][0]) == 1
    assert out.selected_doc_ids_by_layer[0][0][0] in {1, 2}
