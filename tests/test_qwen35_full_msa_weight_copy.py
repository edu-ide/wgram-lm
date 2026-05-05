from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from qtrm_mm.qwen35_full_msa import (
    Qwen35FullMsaForCausalLM,
    copy_qwen35_text_weights_into_full_msa,
)


def source_config() -> Qwen3_5TextConfig:
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
    )
    return cfg


def target_config() -> Qwen3_5TextConfig:
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


def test_weight_copy_reuses_safe_weights_and_reports_replaced_layers() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(1)
    source = Qwen3_5ForCausalLM(source_config())
    torch.manual_seed(2)
    target = Qwen35FullMsaForCausalLM(target_config())

    report = copy_qwen35_text_weights_into_full_msa(source, target)

    assert report["copied_groups"]["embeddings"] == 1
    assert report["copied_groups"]["lm_head"] == 1
    assert report["copied_groups"]["mlp"] == 2
    assert report["copied_groups"]["layer_norms"] == 5
    assert report["copied_groups"]["full_attention_seed"] == 1
    assert report["reinitialized_msa_layers"] == [0]
    assert report["seeded_msa_layers"] == [1]
    assert torch.equal(source.model.embed_tokens.weight, target.model.embed_tokens.weight)
    assert torch.equal(source.lm_head.weight, target.lm_head.weight)
    assert torch.equal(
        source.model.layers[1].self_attn.q_proj.weight,
        target.model.layers[1].self_attn.q_proj.weight,
    )
    assert not torch.equal(
        source.model.layers[0].linear_attn.out_proj.weight,
        target.model.layers[0].self_attn.o_proj.weight,
    )


def test_full_msa_causal_lm_returns_logits_after_weight_copy() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    source = Qwen3_5ForCausalLM(source_config())
    target = Qwen35FullMsaForCausalLM(target_config())
    copy_qwen35_text_weights_into_full_msa(source, target)

    input_ids = torch.tensor([[11, 12, 21, 22, 31, 32]])
    attention_mask = torch.ones_like(input_ids)
    doc_ids = torch.tensor([[1, 1, 2, 2, 0, 0]])

    out = target(input_ids=input_ids, attention_mask=attention_mask, doc_ids=doc_ids)

    assert out.logits.shape == (1, 6, 128)
    assert len(out.selected_doc_ids_by_layer) == 2
