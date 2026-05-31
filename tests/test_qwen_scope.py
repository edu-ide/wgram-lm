from __future__ import annotations

import json

import torch


def test_qwen_scope_sae_loads_layer_file_and_validates_shapes(tmp_path):
    from wgram_lm.qwen_scope import load_qwen_scope_sae_file

    path = tmp_path / "layer3.sae.pt"
    torch.save(
        {
            "W_enc": torch.randn(5, 3),
            "W_dec": torch.randn(3, 5),
            "b_enc": torch.randn(5),
            "b_dec": torch.randn(3),
        },
        path,
    )

    sae = load_qwen_scope_sae_file(path, layer=3, device="cpu")

    assert sae.layer == 3
    assert sae.d_model == 3
    assert sae.d_sae == 5


def test_qwen_scope_topk_features_match_sparse_autoencoder_projection():
    from wgram_lm.qwen_scope import QwenScopeSAE, extract_topk_features

    sae = QwenScopeSAE(
        layer=0,
        W_enc=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        W_dec=torch.zeros(3, 4),
        b_enc=torch.tensor([0.0, 0.0, 0.0, -0.5]),
        b_dec=torch.zeros(3),
    )
    residual = torch.tensor([[[2.0, 1.0, -3.0]]])

    features = extract_topk_features(sae, residual, top_k=2)

    assert features.indices.tolist() == [[[3, 0]]]
    assert torch.allclose(features.values, torch.tensor([[[2.5, 2.0]]]))
    assert features.pre_activation_shape == (1, 1, 4)


def test_qwen_scope_records_last_token_features_as_json_ready_rows():
    from wgram_lm.qwen_scope import QwenScopeSAE, qwen_scope_feature_records

    sae = QwenScopeSAE(
        layer=2,
        W_enc=torch.eye(3),
        W_dec=torch.eye(3),
        b_enc=torch.zeros(3),
        b_dec=torch.zeros(3),
    )
    residual = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 0.5, 1.0]],
            [[1.0, 4.0, 0.0], [0.0, 2.0, 5.0]],
        ]
    )

    rows = qwen_scope_feature_records(
        sae,
        residual,
        prompts=["first", "second"],
        token_ids=torch.tensor([[10, 11], [20, 21]]),
        token_texts=[["a", "b"], ["c", "d"]],
        top_k=2,
        token_position=-1,
    )

    assert rows == [
        {
            "prompt_index": 0,
            "prompt": "first",
            "layer": 2,
            "token_index": 1,
            "token_id": 11,
            "token": "b",
            "feature_ids": [0, 2],
            "feature_values": [3.0, 1.0],
        },
        {
            "prompt_index": 1,
            "prompt": "second",
            "layer": 2,
            "token_index": 1,
            "token_id": 21,
            "token": "d",
            "feature_ids": [2, 1],
            "feature_values": [5.0, 2.0],
        },
    ]
    json.dumps(rows, ensure_ascii=False)


def test_qwen_scope_records_last_nonpad_token_when_attention_mask_is_supplied():
    from wgram_lm.qwen_scope import QwenScopeSAE, qwen_scope_feature_records

    sae = QwenScopeSAE(
        layer=1,
        W_enc=torch.eye(3),
        W_dec=torch.eye(3),
        b_enc=torch.zeros(3),
        b_dec=torch.zeros(3),
    )
    residual = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [9.0, 9.0, 9.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 5.0], [2.0, 0.0, 0.0]],
        ]
    )
    rows = qwen_scope_feature_records(
        sae,
        residual,
        prompts=["short", "longer"],
        token_ids=torch.tensor([[10, 11, 99], [20, 21, 22]]),
        token_texts=[["a", "real", "pad"], ["x", "y", "real"]],
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        top_k=1,
        token_position="last_nonpad",
    )

    assert rows[0]["token_index"] == 1
    assert rows[0]["token_id"] == 11
    assert rows[0]["token"] == "real"
    assert rows[0]["feature_ids"] == [1]
    assert rows[1]["token_index"] == 2
    assert rows[1]["token_id"] == 22
    assert rows[1]["token"] == "real"
    assert rows[1]["feature_ids"] == [0]


def test_qwen_scope_compare_feature_groups_finds_repeat_enriched_features():
    from wgram_lm.qwen_scope import compare_qwen_scope_feature_groups

    records = [
        {
            "prompt_index": 0,
            "layer": 12,
            "token": ".",
            "feature_ids": [1, 2, 3],
            "feature_values": [1.0, 0.5, 0.25],
        },
        {
            "prompt_index": 1,
            "layer": 12,
            "token": "?",
            "feature_ids": [1, 4, 5],
            "feature_values": [1.1, 0.4, 0.3],
        },
        {
            "prompt_index": 2,
            "layer": 12,
            "token": " Freeze",
            "feature_ids": [9, 1, 6],
            "feature_values": [3.0, 1.2, 0.8],
        },
        {
            "prompt_index": 3,
            "layer": 12,
            "token": " world",
            "feature_ids": [9, 7, 1],
            "feature_values": [2.5, 1.5, 1.1],
        },
    ]

    summary = compare_qwen_scope_feature_groups(
        records,
        normal_prompt_indices={0, 1},
        repeat_prompt_indices={2, 3},
        feature_limit=3,
    )

    layer = summary["layers"]["12"]
    assert layer["normal_tokens"] == [".", "?"]
    assert layer["repeat_tokens"] == [" Freeze", " world"]
    assert layer["repeat_common_top_features"] == [1, 9]
    assert layer["repeat_enriched_features"][0]["feature_id"] == 9
    assert layer["repeat_enriched_features"][0]["repeat_freq"] == 1.0
    assert layer["repeat_enriched_features"][0]["normal_freq"] == 0.0


def test_qwen_scope_candidate_scores_group_hits_by_prompt_and_layer():
    from wgram_lm.qwen_scope import score_qwen_scope_candidate_features

    records = [
        {
            "prompt_index": 0,
            "prompt": "normal",
            "layer": 12,
            "feature_ids": [847, 1],
            "feature_values": [0.4, 0.1],
        },
        {
            "prompt_index": 0,
            "prompt": "normal",
            "layer": 23,
            "feature_ids": [5, 31860],
            "feature_values": [0.9, 0.2],
        },
        {
            "prompt_index": 1,
            "prompt": "repeat",
            "layer": 12,
            "feature_ids": [2, 3],
            "feature_values": [0.6, 0.5],
        },
        {
            "prompt_index": 1,
            "prompt": "repeat",
            "layer": 23,
            "feature_ids": [29838, 31860],
            "feature_values": [1.5, 1.0],
        },
    ]

    scores = score_qwen_scope_candidate_features(
        records,
        candidate_features_by_layer={12: [847], 23: [29838, 31860]},
    )

    assert [row["prompt_index"] for row in scores] == [0, 1]
    assert scores[0]["total_hit_count"] == 2
    assert scores[0]["total_value_sum"] == 0.6
    assert scores[0]["total_value_max"] == 0.4
    assert scores[0]["layers"]["12"]["hits"] == [
        {"feature_id": 847, "rank": 1, "value": 0.4}
    ]
    assert scores[1]["total_hit_count"] == 2
    assert scores[1]["total_value_sum"] == 2.5
    assert scores[1]["total_value_max"] == 1.5
    assert scores[1]["layers"]["23"]["hits"] == [
        {"feature_id": 29838, "rank": 1, "value": 1.5},
        {"feature_id": 31860, "rank": 2, "value": 1.0},
    ]


def test_qwen_scope_candidate_scores_can_limit_top_features():
    from wgram_lm.qwen_scope import score_qwen_scope_candidate_features

    records = [
        {
            "prompt_index": 0,
            "layer": 23,
            "feature_ids": [5, 31860],
            "feature_values": [0.9, 0.2],
        },
    ]

    scores = score_qwen_scope_candidate_features(
        records,
        candidate_features_by_layer={23: [31860]},
        feature_limit=1,
    )

    assert scores[0]["total_hit_count"] == 0
    assert scores[0]["layers"]["23"]["hits"] == []


def test_qwen_scope_repeat_score_threshold_summary_selects_best_f1():
    from wgram_lm.qwen_scope import summarize_qwen_scope_repeat_score_thresholds

    scores = [
        {"prompt_index": 0, "total_value_sum": 0.4, "repeat_label": "normal"},
        {"prompt_index": 1, "total_value_sum": 2.5, "repeat_label": "repeat"},
        {"prompt_index": 2, "total_value_sum": 1.0, "repeat_label": "normal"},
        {"prompt_index": 3, "total_value_sum": 3.0, "repeat_label": "repeat"},
    ]

    summary = summarize_qwen_scope_repeat_score_thresholds(scores)

    assert summary["positive_label"] == "repeat"
    assert summary["score_field"] == "total_value_sum"
    assert summary["label_counts"] == {"normal": 2, "repeat": 2}
    assert summary["best_threshold"]["threshold"] == 2.5
    assert summary["best_threshold"]["precision"] == 1.0
    assert summary["best_threshold"]["recall"] == 1.0
    assert summary["best_threshold"]["f1"] == 1.0


def test_qwen_scope_repeat_score_threshold_summary_can_label_from_repeat_rate():
    from wgram_lm.qwen_scope import summarize_qwen_scope_repeat_score_thresholds

    scores = [
        {"prompt_index": 0, "total_value_sum": 0.4, "repeated_2gram_rate": 0.05},
        {"prompt_index": 1, "total_value_sum": 2.5, "repeated_2gram_rate": 0.30},
    ]

    summary = summarize_qwen_scope_repeat_score_thresholds(
        scores,
        label_field=None,
        repeat_rate_threshold=0.15,
    )

    assert summary["label_counts"] == {"normal": 1, "repeat": 1}
    assert summary["best_threshold"]["threshold"] == 2.5
