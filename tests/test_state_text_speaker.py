import torch

from wgram_lm.eval.state_text_speaker import (
    IGNORE_INDEX,
    DirectVocabLogitHead,
    LowRankVocabLogitAdapter,
    PooledContextTextSpeaker,
    RestrictedVocabLogitHead,
    StateTextSpeaker,
    TrajectoryAwareTextSpeaker,
    build_answer_char_vocab,
    build_answer_token_vocab,
    decode_answer_char_indices,
    decode_answer_token_ids,
    encode_answer_char_targets,
    encode_answer_targets,
    encode_restricted_answer_targets,
    first_answer_alias,
    restricted_indices_to_token_ids,
)


class TinyTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"8017": 1, "green": 2, "TRUE": 3, "8008": 4, ",": 5, "8004": 6}
        self.inv_vocab = {value: key for key, value in self.vocab.items()}
        self.inv_vocab[self.eos_token_id] = "<eos>"

    def encode(self, text, add_special_tokens=False):
        if text == "8008,8004":
            return [4, 5, 6]
        return [self.vocab[str(text)]]

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for token_id in ids:
            if skip_special_tokens and int(token_id) == self.eos_token_id:
                continue
            pieces.append(self.inv_vocab[int(token_id)])
        return "".join(pieces).replace("8017green", "8017 green")


def test_state_text_speaker_expands_readout_state_to_answer_positions():
    speaker = StateTextSpeaker(d_state=8, max_answer_tokens=5, hidden_dim=16, dropout=0.0)
    states = speaker(torch.randn(3, 8))

    assert states.shape == (3, 5, 8)


def test_trajectory_aware_speaker_reads_trajectory_and_workspace():
    speaker = TrajectoryAwareTextSpeaker(
        d_state=8,
        max_answer_tokens=5,
        hidden_dim=16,
        n_heads=2,
        dropout=0.0,
    )
    states = speaker(
        torch.randn(3, 8),
        state_trajectory=torch.randn(3, 4, 8),
        workspace=torch.randn(3, 6, 8),
        workspace_attention_mask=torch.ones(3, 6, dtype=torch.long),
    )

    assert states.shape == (3, 5, 8)


def test_pooled_context_speaker_reads_pooled_features():
    speaker = PooledContextTextSpeaker(
        d_state=8,
        max_answer_tokens=5,
        hidden_dim=16,
        dropout=0.0,
    )
    states = speaker(
        torch.randn(3, 8),
        state_trajectory=torch.randn(3, 4, 8),
        workspace=torch.randn(3, 6, 8),
        workspace_attention_mask=torch.ones(3, 6, dtype=torch.long),
    )

    assert states.shape == (3, 5, 8)


def test_low_rank_vocab_adapter_outputs_vocab_residual_logits():
    adapter = LowRankVocabLogitAdapter(d_state=8, vocab_size=17, rank=4, dropout=0.0)
    logits = adapter(torch.randn(3, 5, 8))

    assert logits.shape == (3, 5, 17)


def test_direct_vocab_head_outputs_full_vocab_logits():
    head = DirectVocabLogitHead(d_state=8, vocab_size=17, hidden_dim=16, dropout=0.0)
    logits = head(torch.randn(3, 5, 8))

    assert logits.shape == (3, 5, 17)


def test_restricted_vocab_head_outputs_small_vocab_logits():
    head = RestrictedVocabLogitHead(d_state=8, restricted_vocab_size=7, hidden_dim=16, dropout=0.0)
    logits = head(torch.randn(3, 5, 8))

    assert logits.shape == (3, 5, 7)


def test_answer_target_encoding_adds_eos_and_padding():
    tokenizer = TinyTokenizer()
    targets = encode_answer_targets(tokenizer, ["green", "8008,8004"], max_answer_tokens=5)

    assert targets.tolist()[0] == [2, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert targets.tolist()[1] == [4, 5, 6, 0, IGNORE_INDEX]
    assert decode_answer_token_ids(tokenizer, [4, 5, 6, 0, 2]) == "8008,8004"


def test_restricted_answer_vocab_roundtrip():
    tokenizer = TinyTokenizer()
    rows = [{"answer_aliases": ["green"]}, {"answer_aliases": ["8008,8004"]}]
    allowed = build_answer_token_vocab(tokenizer, rows, max_answer_tokens=5)
    targets = encode_restricted_answer_targets(
        tokenizer,
        ["8008,8004"],
        allowed_token_ids=allowed,
        max_answer_tokens=5,
    )
    token_ids = restricted_indices_to_token_ids(targets.tolist()[0][:4], allowed_token_ids=allowed)

    assert decode_answer_token_ids(tokenizer, token_ids) == "8008,8004"


def test_answer_char_vocab_roundtrip():
    rows = [{"answer_aliases": ["green"]}, {"answer_aliases": ["8008,8004"]}]
    allowed = build_answer_char_vocab(rows)
    targets = encode_answer_char_targets(
        ["8008,8004"],
        allowed_chars=allowed,
        max_answer_chars=12,
    )

    assert decode_answer_char_indices(targets.tolist()[0], allowed_chars=allowed) == "8008,8004"


def test_answer_char_vocab_includes_solver_trace_state_text():
    rows = [
        {
            "answer_aliases": ["17"],
            "solver_trace": [{"depth": 1, "state_text": "8008,8004"}],
        }
    ]
    allowed = build_answer_char_vocab(rows)

    assert "," in allowed
    assert "8" in allowed


def test_first_answer_alias_prefers_answer_aliases():
    assert first_answer_alias({"answer_aliases": ["green"], "answer": "red"}) == "green"
    assert first_answer_alias({"answer": "8017"}) == "8017"
