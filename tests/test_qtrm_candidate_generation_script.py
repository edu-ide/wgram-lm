import importlib.util
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/92_eval_qtrm_logits.py")
    spec = importlib.util.spec_from_file_location("eval_qtrm_logits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_candidate_generation_cli_exposes_sampling_controls() -> None:
    module = load_module()

    args = module.build_arg_parser().parse_args(
        [
            "--num-candidates",
            "4",
            "--do-sample",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--seed",
            "23",
        ]
    )

    assert args.num_candidates == 4
    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.seed == 23


def test_answer_contract_suffix_adds_direct_answer_instructions() -> None:
    module = load_module()

    text = module.apply_answer_contract("What is photosynthesis?", "direct")

    assert "/no_think" in text
    assert "Answer directly" in text
    assert "Do not create multiple-choice options" in text


def test_answer_contract_none_keeps_prompt_unchanged() -> None:
    module = load_module()

    assert module.apply_answer_contract("Prompt", "none") == "Prompt"


def test_cli_exposes_visible_reasoning_suppression() -> None:
    module = load_module()

    args = module.build_arg_parser().parse_args(
        [
            "--suppress-visible-reasoning-tokens",
            "--no-repeat-ngram-size",
            "3",
            "--answer-contract",
            "direct",
        ]
    )

    assert args.suppress_visible_reasoning_tokens is True
    assert args.no_repeat_ngram_size == 3
    assert args.answer_contract == "direct"


def test_cli_exposes_logit_scale_overrides_for_residual_sweeps() -> None:
    module = load_module()

    args = module.build_arg_parser().parse_args(
        [
            "--donor-logits-scale",
            "1.0",
            "--qtrm-logits-scale",
            "0.05",
            "--qtrm-residual-clamp",
            "0.75",
        ]
    )

    assert args.donor_logits_scale == 1.0
    assert args.qtrm_logits_scale == 0.05
    assert args.qtrm_residual_clamp == 0.75


def test_cli_exposes_sentence_stop_for_language_stability_gate() -> None:
    module = load_module()

    args = module.build_arg_parser().parse_args(
        [
            "--stop-after-sentence",
            "--min-new-tokens-before-stop",
            "8",
        ]
    )

    assert args.stop_after_sentence is True
    assert args.min_new_tokens_before_stop == 8


def test_should_stop_after_sentence_uses_completion_not_prompt() -> None:
    module = load_module()

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "질문? 답입니다." if len(ids) >= 4 else "질문? 답"

    assert module.should_stop_after_sentence(
        [1, 2, 3, 4],
        prompt_len=2,
        tokenizer=FakeTokenizer(),
        enabled=True,
        min_new_tokens_before_stop=2,
    )
    assert not module.should_stop_after_sentence(
        [1, 2, 3],
        prompt_len=2,
        tokenizer=FakeTokenizer(),
        enabled=True,
        min_new_tokens_before_stop=2,
    )


def test_logit_scale_overrides_update_model_config() -> None:
    import types

    module = load_module()
    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            donor_logits_scale=0.0,
            qtrm_logits_scale=1.0,
            qtrm_residual_clamp=0.0,
        )
    )
    args = types.SimpleNamespace(
        donor_logits_scale=1.0,
        qtrm_logits_scale=0.1,
        qtrm_residual_clamp=0.5,
    )

    module.apply_logit_scale_overrides(model, args)

    assert model.cfg.donor_logits_scale == 1.0
    assert model.cfg.qtrm_logits_scale == 0.1
    assert model.cfg.qtrm_residual_clamp == 0.5


def test_select_next_token_uses_argmax_without_sampling() -> None:
    module = load_module()
    logits = torch.tensor([0.1, 2.0, 1.0])

    token_id = module.select_next_token(
        logits,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        generator=None,
    )

    assert token_id == 1


def test_sampled_token_respects_top_p_filter() -> None:
    module = load_module()
    logits = torch.tensor([5.0, 1.0, 0.5])
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(11)

    token_id = module.select_next_token(
        logits,
        do_sample=True,
        temperature=1.0,
        top_p=0.1,
        generator=generator,
    )

    assert token_id == 0


def test_select_next_token_applies_suppressed_ids_before_argmax() -> None:
    module = load_module()
    logits = torch.tensor([0.1, 3.0, 2.0])

    token_id = module.select_next_token(
        logits,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        generator=None,
        suppressed_token_ids=[1],
    )

    assert token_id == 2


def test_no_repeat_ngram_bans_tokens_that_would_repeat_existing_ngram() -> None:
    module = load_module()

    assert module.no_repeat_ngram_banned_tokens([1, 2, 1], 2) == [2]
    assert module.no_repeat_ngram_banned_tokens([1, 2, 3, 1, 2], 3) == [3]
    assert module.no_repeat_ngram_banned_tokens([1, 2], 3) == []
