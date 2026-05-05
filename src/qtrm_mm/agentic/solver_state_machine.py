from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn


SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass(frozen=True)
class CharVocab:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharVocab":
        chars = sorted({ch for text in texts for ch in str(text)})
        tokens = list(SPECIAL_TOKENS)
        tokens.extend(ch for ch in chars if ch not in SPECIAL_TOKENS)
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id, id_to_token=tuple(tokens))

    def encode(
        self,
        text: str,
        *,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in str(text)]
        if add_eos:
            ids.append(self.eos_id)
        if max_len is not None:
            ids = ids[: int(max_len)]
            ids.extend([self.pad_id] * max(0, int(max_len) - len(ids)))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars: list[str] = []
        for raw_id in ids:
            idx = int(raw_id)
            if idx < 0 or idx >= len(self.id_to_token):
                token = "<unk>"
            else:
                token = self.id_to_token[idx]
            if token == "<eos>":
                break
            if token in {"<pad>", "<bos>"}:
                continue
            chars.append("?" if token == "<unk>" else token)
        return "".join(chars)


@dataclass(frozen=True)
class OperationVocab:
    operation_to_id: dict[str, int]
    id_to_operation: tuple[str, ...]

    @classmethod
    def build(cls, operations: Iterable[str]) -> "OperationVocab":
        labels = sorted({str(operation) for operation in operations})
        return cls(
            operation_to_id={label: idx for idx, label in enumerate(labels)},
            id_to_operation=tuple(labels),
        )

    def encode(self, operation: str) -> int:
        return int(self.operation_to_id[str(operation)])

    def decode(self, operation_id: int) -> str:
        return self.id_to_operation[int(operation_id)]


def state_machine_input_text(
    row: Mapping[str, object],
    previous_state: str | None = None,
) -> str:
    prev = row.get("previous_state_text", "") if previous_state is None else previous_state
    task = row.get("question") or row.get("prompt", "")
    return "\n".join(
        [
            f"Operation: {row.get('operation', '')}",
            f"Depth: {row.get('depth', '')}",
            f"Previous state: {prev}",
            f"Task: {task}",
            "Target state:",
        ]
    )


def operation_policy_input_text(
    row: Mapping[str, object],
    previous_state: str | None = None,
) -> str:
    prev = row.get("previous_state_text", "") if previous_state is None else previous_state
    task = row.get("question") or row.get("prompt", "")
    return "\n".join(
        [
            f"Task: {task}",
            f"Previous state: {prev}",
            f"Task family: {row.get('task_family') or row.get('category') or ''}",
            f"Trace index: {row.get('trace_index', '')}",
            f"Depth: {row.get('depth', '')}",
            "Operation:",
        ]
    )


def target_tensors(
    vocab: CharVocab,
    target_text: str,
    *,
    max_target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = int(max_target_len)
    target_ids = vocab.encode(str(target_text), add_eos=True)
    decoder_input_ids = [vocab.bos_id] + target_ids
    labels = list(target_ids)
    decoder_input_ids = decoder_input_ids[:max_len]
    labels = labels[:max_len]
    decoder_input_ids.extend([vocab.pad_id] * max(0, max_len - len(decoder_input_ids)))
    labels.extend([vocab.pad_id] * max(0, max_len - len(labels)))
    return (
        torch.tensor(decoder_input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _question_text(row: Mapping[str, object]) -> str:
    return str(row.get("question") or row.get("prompt") or "")


def _parse_arithmetic_question(question: str) -> tuple[int, int, int]:
    match = re.search(r"\(\(([-]?\d+)\s*\+\s*([-]?\d+)\)\s*\*\s*([-]?\d+)\)\s*-\s*([-]?\d+)", question)
    if not match:
        raise ValueError(f"unsupported arithmetic question: {question!r}")
    a, b, c, subtract = (int(group) for group in match.groups())
    if subtract != b:
        raise ValueError("arithmetic primitive expects final subtract operand to match b")
    return a, b, c


def _parse_list_question(question: str) -> list[int]:
    match = re.search(r"\[([^\]]*)\]", question)
    if not match:
        raise ValueError(f"unsupported list question: {question!r}")
    values: list[int] = []
    for part in match.group(1).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _parse_symbolic_mappings(question: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for source, target in re.findall(r"\b([A-Za-z]+)\s+maps\s+to\s+([A-Za-z]+)\b", question):
        mappings[str(source)] = str(target)
    if not mappings:
        raise ValueError(f"unsupported symbolic question: {question!r}")
    return mappings


def _parse_boolean_bindings(question: str) -> dict[str, bool]:
    bindings: dict[str, bool] = {}
    for name, value in re.findall(r"\b([PQR])=(TRUE|FALSE)\b", question):
        bindings[name] = value == "TRUE"
    if set(bindings) != {"P", "Q", "R"}:
        raise ValueError(f"unsupported boolean question: {question!r}")
    return bindings


def _bool_text(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def _text_bool(value: str) -> bool:
    text = str(value).strip().upper()
    if text == "TRUE":
        return True
    if text == "FALSE":
        return False
    raise ValueError(f"expected TRUE/FALSE state, got {value!r}")


def _parse_csv_ints(text: str) -> list[int]:
    if not str(text).strip() or str(text).strip() == "EMPTY":
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values) if values else "EMPTY"


def execute_solver_transition(
    row: Mapping[str, object],
    previous_state: str,
) -> str:
    operation = str(row.get("operation") or "")
    question = _question_text(row)
    if operation == "hold_final":
        return str(previous_state)
    if operation in {"add_operands", "multiply_sum", "subtract_offset"}:
        a, b, c = _parse_arithmetic_question(question)
        if operation == "add_operands":
            return str(a + b)
        if operation == "multiply_sum":
            return str(int(previous_state) * c)
        return str(int(previous_state) - b)
    if operation in {"filter_even", "double_filtered"}:
        if operation == "filter_even":
            return _format_int_list([value for value in _parse_list_question(question) if value % 2 == 0])
        return _format_int_list([value * 2 for value in _parse_csv_ints(previous_state)])
    if operation in {"first_mapping", "second_mapping"}:
        mappings = _parse_symbolic_mappings(question)
        if operation == "first_mapping":
            first_source = next(iter(mappings))
            return mappings[first_source]
        return mappings[str(previous_state)]
    if operation in {"not_q", "and_with_p", "or_with_r"}:
        bindings = _parse_boolean_bindings(question)
        if operation == "not_q":
            return _bool_text(not bindings["Q"])
        if operation == "and_with_p":
            return _bool_text(bindings["P"] and _text_bool(previous_state))
        return _bool_text(_text_bool(previous_state) or bindings["R"])
    raise ValueError(f"unsupported solver operation: {operation!r}")


def rollout_solver_trace_from_operations(
    row: Mapping[str, object],
    operations: Sequence[str],
) -> dict[str, object]:
    """Execute a solver trace with predicted primitive operation names."""
    trace = row.get("solver_trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise ValueError("row must include solver_trace")
    if len(operations) < len(trace):
        raise ValueError("operations must include at least one prediction per trace step")

    previous_state = ""
    records: list[dict[str, object]] = []
    operation_hits = 0
    state_hits = 0
    for index, raw_step in enumerate(trace):
        if not isinstance(raw_step, Mapping):
            raise ValueError("solver_trace entries must be mappings")
        operation = str(operations[index])
        target_operation = str(raw_step.get("operation") or "")
        target_state = str(raw_step.get("state_text") or "")
        step_row = dict(row)
        step_row.update(raw_step)
        step_row["operation"] = operation
        error = ""
        try:
            predicted_state = execute_solver_transition(step_row, previous_state)
        except Exception as exc:  # noqa: BLE001 - eval records should preserve bad predictions
            predicted_state = ""
            error = f"{type(exc).__name__}: {exc}"
        operation_exact = operation == target_operation
        state_exact = predicted_state == target_state
        operation_hits += int(operation_exact)
        state_hits += int(state_exact)
        records.append(
            {
                "trace_index": index,
                "depth": raw_step.get("depth", index + 1),
                "operation": operation,
                "target_operation": target_operation,
                "operation_exact_match": operation_exact,
                "previous_state_text": previous_state,
                "predicted_state_text": predicted_state,
                "target_state_text": target_state,
                "state_exact_match": state_exact,
                "error": error,
            }
        )
        previous_state = predicted_state

    target_final = str(row.get("chosen") or row.get("answer") or previous_state)
    return {
        "records": records,
        "predicted_final": previous_state,
        "target_final": target_final,
        "final_exact_match": previous_state == target_final,
        "operation_exact_count": operation_hits,
        "state_exact_count": state_hits,
        "total_steps": len(records),
        "operation_exact": f"{operation_hits}/{len(records)}",
        "state_exact": f"{state_hits}/{len(records)}",
    }


def answer_from_primitive_operations(
    row: Mapping[str, object],
    operations: Sequence[str],
    *,
    stop_on_hold_final: bool = True,
) -> dict[str, object]:
    """Run predicted primitive operations and return the final state as answer."""
    previous_state = ""
    states: list[str] = []
    executed_operations: list[str] = []
    records: list[dict[str, object]] = []
    for index, operation in enumerate(operations):
        operation = str(operation)
        if operation == "hold_final" and bool(stop_on_hold_final):
            break
        step_row = dict(row)
        step_row["operation"] = operation
        predicted_state = execute_solver_transition(step_row, previous_state)
        records.append(
            {
                "trace_index": index,
                "operation": operation,
                "previous_state_text": previous_state,
                "predicted_state_text": predicted_state,
            }
        )
        executed_operations.append(operation)
        states.append(predicted_state)
        previous_state = predicted_state
    return {
        "answer": previous_state,
        "states": states,
        "executed_operations": executed_operations,
        "records": records,
    }


def operation_names_from_logits(
    operation_logits: torch.Tensor,
    id_to_operation: Mapping[int, str],
    *,
    row: Mapping[str, object] | None = None,
    state_constrained: bool = False,
) -> list[str]:
    logits = operation_logits.detach()
    if logits.ndim == 3:
        if int(logits.shape[0]) != 1:
            raise ValueError("batched operation logits must have batch size 1")
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError("operation_logits must have shape [steps, operations]")
    if bool(state_constrained) and row is None:
        raise ValueError("row is required when state_constrained=True")
    pred_ids = logits.float().argmax(dim=-1).detach().cpu().tolist()
    names: list[str] = []
    previous_state = ""
    previous_required_operations = {
        "multiply_sum",
        "subtract_offset",
        "double_filtered",
        "second_mapping",
        "and_with_p",
        "or_with_r",
        "hold_final",
    }
    for step_index, pred_id in enumerate(pred_ids):
        idx = int(pred_id)
        if idx not in id_to_operation:
            raise ValueError(f"unknown predicted operation id: {idx}")
        if not bool(state_constrained):
            names.append(str(id_to_operation[idx]))
            continue
        if not previous_state:
            selected_operation = str(id_to_operation[idx])
            names.append(selected_operation)
            step_row = dict(row or {})
            step_row["operation"] = selected_operation
            try:
                previous_state = execute_solver_transition(step_row, previous_state)
            except Exception:
                previous_state = ""
            continue
        row_logits = logits[step_index].float()
        ranked_ids = row_logits.argsort(descending=True).detach().cpu().tolist()
        selected_operation = str(id_to_operation[idx])
        selected_state: str | None = None
        for candidate_id in ranked_ids:
            candidate_idx = int(candidate_id)
            if candidate_idx not in id_to_operation:
                continue
            candidate_operation = str(id_to_operation[candidate_idx])
            if not previous_state and candidate_operation in previous_required_operations:
                continue
            step_row = dict(row or {})
            step_row["operation"] = candidate_operation
            try:
                candidate_state = execute_solver_transition(step_row, previous_state)
            except Exception:
                continue
            selected_operation = candidate_operation
            selected_state = candidate_state
            break
        names.append(selected_operation)
        if selected_state is not None:
            previous_state = selected_state
    return names


class SolverStateMachine(nn.Module):
    """Small seq2seq recurrent probe for explicit solver-state transitions."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        hidden_dim: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.encoder = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, self.vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_id >= 0 and self.pad_id < self.embedding.weight.shape[0]:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].zero_()
        for gru in (self.encoder, self.decoder):
            for name, param in gru.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [batch, target_len]")
        encoder_emb = self.embedding(input_ids)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            encoder_emb = encoder_emb * attention_mask.to(encoder_emb.dtype).unsqueeze(-1)
        _, hidden = self.encoder(encoder_emb)
        decoder_emb = self.embedding(decoder_input_ids)
        decoder_states, _ = self.decoder(decoder_emb, hidden)
        return self.output(decoder_states)


class OperationPolicy(nn.Module):
    """Predict the next solver primitive from question, depth, and current state."""

    def __init__(
        self,
        vocab_size: int,
        num_operations: int,
        *,
        d_model: int = 128,
        hidden_dim: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_operations = int(num_operations)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.encoder = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, self.num_operations)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_id >= 0 and self.pad_id < self.embedding.weight.shape[0]:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].zero_()
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            emb = emb * attention_mask.to(emb.dtype).unsqueeze(-1)
        _, hidden = self.encoder(emb)
        return self.output(hidden[-1])


class StructuredOperationPolicy(nn.Module):
    """Predict solver primitive from structured trace metadata."""

    def __init__(
        self,
        *,
        num_families: int,
        num_trace_indices: int,
        num_depths: int,
        num_operations: int,
        d_model: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_families = int(num_families)
        self.num_trace_indices = int(num_trace_indices)
        self.num_depths = int(num_depths)
        self.num_operations = int(num_operations)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.family_embedding = nn.Embedding(self.num_families, self.d_model)
        self.trace_index_embedding = nn.Embedding(self.num_trace_indices, self.d_model)
        self.depth_embedding = nn.Embedding(self.num_depths, self.d_model)
        self.net = nn.Sequential(
            nn.Linear(self.d_model * 3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_operations),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.family_embedding,
            self.trace_index_embedding,
            self.depth_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        *,
        family_ids: torch.Tensor,
        trace_index_ids: torch.Tensor,
        depth_ids: torch.Tensor,
    ) -> torch.Tensor:
        if family_ids.ndim != 1 or trace_index_ids.ndim != 1 or depth_ids.ndim != 1:
            raise ValueError("structured ids must have shape [batch]")
        features = torch.cat(
            [
                self.family_embedding(family_ids),
                self.trace_index_embedding(trace_index_ids),
                self.depth_embedding(depth_ids),
            ],
            dim=-1,
        )
        return self.net(features)


def rollout_trace_rows(
    rows: Sequence[Mapping[str, object]],
    predict: Callable[[Mapping[str, object], str], str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (str(row.get("source_id", "")), int(row.get("trace_index", 0))),
    )
    current_source: str | None = None
    previous_state = ""
    for row in sorted_rows:
        source_id = str(row.get("source_id", ""))
        if source_id != current_source:
            current_source = source_id
            previous_state = str(row.get("previous_state_text", ""))
        predicted = str(predict(row, previous_state))
        target = str(row.get("target_state_text", ""))
        record = dict(row)
        record["rollout_previous_state_text"] = previous_state
        record["predicted_state_text"] = predicted
        record["state_exact_match"] = predicted == target
        records.append(record)
        previous_state = predicted
    return records
