"""General answer speaker for recurrent state-transition thought states."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

IGNORE_INDEX = -100


class StateTextSpeaker(nn.Module):
    """Map one recurrent thought state to a short parallel answer-token sequence.

    The speaker deliberately stays small. It does not solve the task; it only
    learns how to express an existing thought state through the frozen Qwen
    LM-head language.
    """

    def __init__(
        self,
        *,
        d_state: int,
        max_answer_tokens: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.max_answer_tokens = int(max_answer_tokens)
        hidden = int(hidden_dim or d_state * 2)
        self.position_embed = nn.Embedding(self.max_answer_tokens, self.d_state)
        self.norm = nn.LayerNorm(self.d_state)
        self.adapter = nn.Sequential(
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.d_state),
        )
        self.gate = nn.Linear(self.d_state, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, readout_state: torch.Tensor) -> torch.Tensor:
        if readout_state.ndim != 2:
            raise ValueError("readout_state must have shape (batch, d_state)")
        batch = int(readout_state.size(0))
        positions = torch.arange(self.max_answer_tokens, device=readout_state.device)
        pos = self.position_embed(positions).unsqueeze(0).expand(batch, -1, -1)
        base = readout_state.unsqueeze(1).expand(-1, self.max_answer_tokens, -1) + pos
        hidden = self.norm(base)
        delta = self.adapter(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return self.norm(base + gate * delta)


class TrajectoryAwareTextSpeaker(nn.Module):
    """Map a recurrent thought trajectory and prompt workspace to answer tokens.

    This is the smallest architecture-clean "mouth" for Stage59: the answer
    speaker is allowed to look at the thought path and the reader workspace,
    but it still has to speak through Qwen-compatible token logits.
    """

    def __init__(
        self,
        *,
        d_state: int,
        max_answer_tokens: int,
        hidden_dim: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.max_answer_tokens = int(max_answer_tokens)
        self.n_heads = int(n_heads)
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_state % self.n_heads != 0:
            raise ValueError("d_state must be divisible by n_heads")
        hidden = int(hidden_dim or d_state * 2)
        self.position_embed = nn.Embedding(self.max_answer_tokens, self.d_state)
        self.query_norm = nn.LayerNorm(self.d_state)
        self.trajectory_attn = nn.MultiheadAttention(
            self.d_state,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.workspace_attn = nn.MultiheadAttention(
            self.d_state,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.trajectory_norm = nn.LayerNorm(self.d_state)
        self.workspace_norm = nn.LayerNorm(self.d_state)
        self.adapter = nn.Sequential(
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.d_state),
        )
        self.gate = nn.Linear(self.d_state, 1)
        self.out_norm = nn.LayerNorm(self.d_state)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def _key_padding_mask(attention_mask: torch.Tensor | None, memory: torch.Tensor) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (batch, seq)")
        if attention_mask.size(0) != memory.size(0) or attention_mask.size(1) != memory.size(1):
            raise ValueError("attention_mask shape must match memory batch/sequence dimensions")
        mask = attention_mask.to(device=memory.device, dtype=torch.bool).logical_not()
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask = mask.clone()
            mask[all_masked] = False
        return mask

    def forward(
        self,
        readout_state: torch.Tensor,
        *,
        state_trajectory: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        workspace_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if readout_state.ndim != 2:
            raise ValueError("readout_state must have shape (batch, d_state)")
        batch = int(readout_state.size(0))
        positions = torch.arange(self.max_answer_tokens, device=readout_state.device)
        pos = self.position_embed(positions).unsqueeze(0).expand(batch, -1, -1)
        query = self.query_norm(readout_state.unsqueeze(1).expand(-1, self.max_answer_tokens, -1) + pos)

        if state_trajectory is not None:
            if state_trajectory.ndim != 3:
                raise ValueError("state_trajectory must have shape (batch, steps, d_state)")
            trajectory_context, _ = self.trajectory_attn(
                query,
                state_trajectory.to(query.dtype),
                state_trajectory.to(query.dtype),
                need_weights=False,
            )
            query = self.trajectory_norm(query + trajectory_context)

        if workspace is not None:
            if workspace.ndim != 3:
                raise ValueError("workspace must have shape (batch, seq, d_state)")
            workspace_context, _ = self.workspace_attn(
                query,
                workspace.to(query.dtype),
                workspace.to(query.dtype),
                key_padding_mask=self._key_padding_mask(workspace_attention_mask, workspace),
                need_weights=False,
            )
            query = self.workspace_norm(query + workspace_context)

        delta = self.adapter(query)
        gate = torch.sigmoid(self.gate(query))
        return self.out_norm(query + gate * delta)


class PooledContextTextSpeaker(nn.Module):
    """Small speaker over pooled readout, trajectory, and workspace features."""

    def __init__(
        self,
        *,
        d_state: int,
        max_answer_tokens: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        hidden = int(hidden_dim or d_state * 2)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.d_state * 3),
            nn.Linear(self.d_state * 3, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.d_state),
        )
        self.token_speaker = StateTextSpeaker(
            d_state=self.d_state,
            max_answer_tokens=max_answer_tokens,
            hidden_dim=hidden,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.context_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _masked_mean(memory: torch.Tensor | None, mask: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
        if memory is None:
            return fallback
        if mask is None:
            return memory.mean(dim=1)
        weights = mask.to(device=memory.device, dtype=memory.dtype).unsqueeze(-1)
        return (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def forward(
        self,
        readout_state: torch.Tensor,
        *,
        state_trajectory: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        workspace_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if readout_state.ndim != 2:
            raise ValueError("readout_state must have shape (batch, d_state)")
        trajectory_mean = self._masked_mean(state_trajectory, None, readout_state)
        workspace_mean = self._masked_mean(workspace, workspace_attention_mask, readout_state)
        context = self.context_proj(torch.cat([readout_state, trajectory_mean, workspace_mean], dim=-1))
        return self.token_speaker(context)


class LowRankVocabLogitAdapter(nn.Module):
    """Small vocabulary-logit residual adapter for a frozen LM head."""

    def __init__(
        self,
        *,
        d_state: int,
        vocab_size: int,
        rank: int = 64,
        dropout: float = 0.05,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(self.d_state)
        self.down = nn.Linear(self.d_state, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(self.rank, self.vocab_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, answer_states: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.down(self.norm(answer_states)))
        return self.up(self.dropout(hidden)) * self.scale.to(hidden.dtype)


class DirectVocabLogitHead(nn.Module):
    """Diagnostic full-vocab head for answer-token supervision.

    This is intentionally stronger than the low-rank residual adapter. It tests
    whether the speaker states contain the answer signal at all before we spend
    more time forcing them through a frozen LM head bottleneck.
    """

    def __init__(
        self,
        *,
        d_state: int,
        vocab_size: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.vocab_size = int(vocab_size)
        hidden = int(hidden_dim or d_state * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_state),
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.vocab_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, answer_states: torch.Tensor) -> torch.Tensor:
        return self.net(answer_states)


class RestrictedVocabLogitHead(nn.Module):
    """Answer-token head over a deliberately small token alphabet."""

    def __init__(
        self,
        *,
        d_state: int,
        restricted_vocab_size: int,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_state = int(d_state)
        self.restricted_vocab_size = int(restricted_vocab_size)
        hidden = int(hidden_dim or d_state * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_state),
            nn.Linear(self.d_state, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.restricted_vocab_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, answer_states: torch.Tensor) -> torch.Tensor:
        return self.net(answer_states)


def first_answer_alias(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    for key in ("answer", "answer_text", "gold_answer"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def encode_answer_targets(
    tokenizer: Any,
    answers: Sequence[str],
    *,
    max_answer_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    target_rows: list[list[int]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for answer in answers:
        ids = [int(token_id) for token_id in tokenizer.encode(str(answer), add_special_tokens=False)]
        if eos_id is not None:
            ids.append(int(eos_id))
        ids = ids[: int(max_answer_tokens)]
        padded = ids + [IGNORE_INDEX] * (int(max_answer_tokens) - len(ids))
        target_rows.append(padded)
    return torch.tensor(target_rows, dtype=torch.long, device=device)


def build_answer_token_vocab(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    max_answer_tokens: int,
) -> list[int]:
    """Build a small token alphabet from answer strings.

    This is a diagnostic/general-interface alphabet, not an answer oracle: it
    exposes possible characters/subtokens, not which sequence is correct.
    """

    token_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        token_ids.add(int(eos_id))
    for row in rows:
        answer = first_answer_alias(row)
        ids = tokenizer.encode(str(answer), add_special_tokens=False)
        for token_id in ids[: int(max_answer_tokens)]:
            token_ids.add(int(token_id))
    return sorted(token_ids)


def encode_restricted_answer_targets(
    tokenizer: Any,
    answers: Sequence[str],
    *,
    allowed_token_ids: Sequence[int],
    max_answer_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    allowed_index = {int(token_id): index for index, token_id in enumerate(allowed_token_ids)}
    full_targets = encode_answer_targets(
        tokenizer,
        answers,
        max_answer_tokens=max_answer_tokens,
        device=None,
    )
    rows: list[list[int]] = []
    for target_row in full_targets.tolist():
        restricted_row: list[int] = []
        for token_id in target_row:
            value = int(token_id)
            if value == IGNORE_INDEX:
                restricted_row.append(IGNORE_INDEX)
            elif value in allowed_index:
                restricted_row.append(int(allowed_index[value]))
            else:
                raise ValueError(f"target token {value} is missing from restricted answer vocab")
        rows.append(restricted_row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def restricted_indices_to_token_ids(
    restricted_indices: Sequence[int],
    *,
    allowed_token_ids: Sequence[int],
) -> list[int]:
    allowed = [int(token_id) for token_id in allowed_token_ids]
    return [allowed[int(index)] for index in restricted_indices]


CHAR_EOS = "<eos>"


def build_answer_char_vocab(rows: Sequence[dict[str, Any]]) -> list[str]:
    chars: set[str] = {CHAR_EOS}
    for row in rows:
        chars.update(str(first_answer_alias(row)))
        for step in row.get("solver_trace") or ():
            if isinstance(step, dict) and step.get("state_text") is not None:
                chars.update(str(step["state_text"]))
        depth_targets = row.get("depth_targets")
        if isinstance(depth_targets, dict):
            for value in depth_targets.values():
                chars.update(str(value))
    return [CHAR_EOS] + sorted(char for char in chars if char != CHAR_EOS)


def encode_answer_char_targets(
    answers: Sequence[str],
    *,
    allowed_chars: Sequence[str],
    max_answer_chars: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    char_index = {char: index for index, char in enumerate(allowed_chars)}
    if CHAR_EOS not in char_index:
        raise ValueError("allowed_chars must include CHAR_EOS")
    rows: list[list[int]] = []
    for answer in answers:
        values = [char_index[char] for char in str(answer)]
        values.append(char_index[CHAR_EOS])
        values = values[: int(max_answer_chars)]
        rows.append(values + [IGNORE_INDEX] * (int(max_answer_chars) - len(values)))
    return torch.tensor(rows, dtype=torch.long, device=device)


def decode_answer_char_indices(indices: Sequence[int], *, allowed_chars: Sequence[str]) -> str:
    chars: list[str] = []
    for index in indices:
        char = str(allowed_chars[int(index)])
        if char == CHAR_EOS:
            break
        chars.append(char)
    return "".join(chars).strip()


def decode_answer_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    clean: list[int] = []
    for token_id in token_ids:
        value = int(token_id)
        if value == IGNORE_INDEX:
            continue
        if eos_id is not None and value == int(eos_id):
            break
        clean.append(value)
    return str(tokenizer.decode(clean, skip_special_tokens=True)).strip()


def speaker_vocab_logits(wgram_model: Any, speaker: StateTextSpeaker, readout_state: torch.Tensor) -> torch.Tensor:
    answer_states = speaker(readout_state)
    _, vocab_logits = wgram_model._lm_head_logits_from_state(answer_states)
    return vocab_logits
