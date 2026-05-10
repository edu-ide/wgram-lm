from __future__ import annotations

import json
import re
from typing import Any


TYPED_ALGORITHMIC_FIELD_KEYS = (
    "kind",
    "raw_list_offsets",
    "doubled_list_offsets",
    "scalar_coeff",
    "scalar_offset",
    "scalar_residual",
    "final_residual",
)


def apply_role_value_list_class_mode(
    rows: list[dict[str, Any]],
    list_class_mode: str | None,
) -> str:
    mode = str(list_class_mode or "source_position").strip() or "source_position"
    for row in rows:
        row["role_value_list_class_mode"] = mode
    return mode


def row_mixed_list_base(row: dict[str, Any]) -> int | None:
    raw = row.get("list_value_start", row.get("base_value"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_int_list_state(text: str) -> list[int] | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        values = parsed
    elif "," in stripped:
        values = stripped.split(",")
    else:
        return None
    result: list[int] = []
    for value in values:
        try:
            result.append(int(str(value).strip()))
        except (TypeError, ValueError):
            return None
    return result


def row_input_list(row: dict[str, Any]) -> list[int] | None:
    raw = row.get("input_list")
    if raw is None:
        raw = row.get("list")
    if raw is None:
        base = row_mixed_list_base(row)
        raw_length = row.get("list_length")
        if base is not None and raw_length is not None:
            try:
                length = int(raw_length)
            except (TypeError, ValueError):
                length = 0
            if length > 0:
                return [int(base) + offset for offset in range(length)]
    if raw is None:
        for key in ("question", "prompt"):
            text = str(row.get(key) or "")
            match = re.search(r"\blist\s+(\[[^\]]+\])", text)
            if match is None:
                match = re.search(r"(\[[^\]]+\])", text)
            if match:
                raw = match.group(1)
                break
    if raw is None:
        return None
    if isinstance(raw, list):
        values = raw
    else:
        parsed = parse_int_list_state(str(raw))
        if parsed is None:
            return None
        return parsed
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            return None
    return result


def source_numeric_value_spans(
    row: dict[str, Any],
    *,
    value_vocab_size: int,
) -> tuple[tuple[int, int, int], ...]:
    """Map source-list numeric values to character spans in the prompt.

    The returned class id is one-based so 0 remains the "no numeric value"
    token class.
    """

    prompt = str(row.get("prompt") or "")
    values = row_input_list(row)
    if values is None:
        raise ValueError("row has no input_list")
    list_start = prompt.find("[")
    list_end = prompt.find("]", list_start + 1) if list_start >= 0 else -1
    if list_start < 0 or list_end < 0:
        raise ValueError("prompt does not contain a bracketed source list")
    spans: list[tuple[int, int, int]] = []
    cursor = list_start + 1
    for value in values:
        text = str(int(value))
        start = prompt.find(text, cursor, list_end)
        if start < 0:
            raise ValueError(f"could not align source value in prompt: {text}")
        class_id = int(value) + 1
        if class_id <= 0 or class_id >= int(value_vocab_size):
            raise ValueError(f"numeric value class out of range: {class_id}")
        end = start + len(text)
        spans.append((start, end, class_id))
        cursor = end
    return tuple(spans)


def token_numeric_value_ids(
    row: dict[str, Any],
    *,
    offsets: Any,
    value_vocab_size: int,
) -> tuple[int, ...]:
    """Encode token-aligned source numeric values for canonical token input."""

    spans = source_numeric_value_spans(
        row,
        value_vocab_size=int(value_vocab_size),
    )
    ids: list[int] = []
    for raw_start, raw_end in offsets:
        start = int(raw_start)
        end = int(raw_end)
        class_id = 0
        if end > start:
            for span_start, span_end, span_class in spans:
                if max(start, span_start) < min(end, span_end):
                    class_id = int(span_class)
                    break
        ids.append(class_id)
    return tuple(ids)


def token_numeric_source_slot_ids(
    row: dict[str, Any],
    *,
    offsets: Any,
    max_list_len: int,
    value_vocab_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build compact source-slot ids from token-aligned prompt numeric ids.

    The source slots are derived from tokenizer offsets over the visible prompt,
    then consecutive token pieces overlapping the same source number are
    collapsed into one slot. This preserves the canonical prompt/token source
    while giving recurrent state code a compact list-shaped input.
    """

    token_ids = token_numeric_value_ids(
        row,
        offsets=offsets,
        value_vocab_size=int(value_vocab_size),
    )
    slots: list[int] = []
    previous = 0
    for raw_id in token_ids:
        class_id = int(raw_id)
        if class_id <= 0:
            previous = 0
            continue
        if class_id != previous:
            slots.append(class_id)
            if len(slots) >= int(max_list_len):
                break
        previous = class_id
    ids = slots[: int(max_list_len)]
    mask = [1] * len(ids)
    while len(ids) < int(max_list_len):
        ids.append(0)
        mask.append(0)
    return tuple(ids), tuple(mask)


def relative_source_slot_parity_ids(
    row: dict[str, Any],
    *,
    max_list_len: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode compact source slots as relative parity classes.

    Class ids are intentionally tiny and independent of absolute numeric
    magnitude: 0=pad, 1=odd source value, 2=even source value.
    """

    values = row_input_list(row)
    if values is None:
        raise ValueError("row has no input_list")
    ids: list[int] = []
    mask: list[int] = []
    for value in values[: int(max_list_len)]:
        ids.append(2 if int(value) % 2 == 0 else 1)
        mask.append(1)
    while len(ids) < int(max_list_len):
        ids.append(0)
        mask.append(0)
    return tuple(ids), tuple(mask)


def token_numeric_source_slot_token_ids(
    row: dict[str, Any],
    *,
    offsets: Any,
    input_ids: Any,
    max_list_len: int,
    value_vocab_size: int,
) -> tuple[int, ...]:
    """Return tokenizer ids for the first token piece of each compact source slot."""

    spans = source_numeric_value_spans(
        row,
        value_vocab_size=int(value_vocab_size),
    )
    input_id_list = [int(token_id) for token_id in input_ids]
    if len(input_id_list) != len(offsets):
        raise ValueError("input_ids and offsets must have the same length")
    ids: list[int] = []
    previous = 0
    for token_id, raw_offset in zip(input_id_list, offsets):
        raw_start, raw_end = raw_offset
        start = int(raw_start)
        end = int(raw_end)
        class_id = 0
        if end > start:
            for span_start, span_end, span_class in spans:
                if max(start, span_start) < min(end, span_end):
                    class_id = int(span_class)
                    break
        if class_id <= 0:
            previous = 0
            continue
        if class_id != previous:
            ids.append(int(token_id))
            if len(ids) >= int(max_list_len):
                break
        previous = class_id
    ids = ids[: int(max_list_len)]
    while len(ids) < int(max_list_len):
        ids.append(0)
    return tuple(ids)


def token_numeric_source_slot_token_spans(
    row: dict[str, Any],
    *,
    offsets: Any,
    input_ids: Any,
    max_list_len: int,
    max_token_pieces: int,
    value_vocab_size: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return all tokenizer pieces for each compact source slot.

    `token_numeric_source_slot_token_ids` intentionally keeps only the first
    piece for older single-token copy probes. Span-copy lexicalization needs the
    full token sequence, e.g. source value 44 -> ["4", "4"] under Qwen's
    tokenizer.
    """

    if int(max_token_pieces) <= 0:
        raise ValueError("max_token_pieces must be positive")
    spans = source_numeric_value_spans(
        row,
        value_vocab_size=int(value_vocab_size),
    )
    input_id_list = [int(token_id) for token_id in input_ids]
    if len(input_id_list) != len(offsets):
        raise ValueError("input_ids and offsets must have the same length")

    slot_pieces: list[list[int]] = []
    current_class = 0
    current_pieces: list[int] = []

    def flush_current() -> None:
        nonlocal current_class, current_pieces
        if current_class > 0 and len(slot_pieces) < int(max_list_len):
            slot_pieces.append(list(current_pieces[: int(max_token_pieces)]))
        current_class = 0
        current_pieces = []

    for token_id, raw_offset in zip(input_id_list, offsets):
        raw_start, raw_end = raw_offset
        start = int(raw_start)
        end = int(raw_end)
        class_id = 0
        if end > start:
            for span_start, span_end, span_class in spans:
                if max(start, span_start) < min(end, span_end):
                    class_id = int(span_class)
                    break
        if class_id <= 0:
            flush_current()
            continue
        if current_class == 0:
            current_class = class_id
            current_pieces = [int(token_id)]
            continue
        if class_id == current_class:
            current_pieces.append(int(token_id))
            continue
        flush_current()
        current_class = class_id
        current_pieces = [int(token_id)]
    flush_current()

    span_rows: list[tuple[int, ...]] = []
    mask_rows: list[tuple[int, ...]] = []
    for pieces in slot_pieces[: int(max_list_len)]:
        row_ids = list(pieces[: int(max_token_pieces)])
        row_mask = [1] * len(row_ids)
        while len(row_ids) < int(max_token_pieces):
            row_ids.append(0)
            row_mask.append(0)
        span_rows.append(tuple(row_ids))
        mask_rows.append(tuple(row_mask))
    while len(span_rows) < int(max_list_len):
        span_rows.append(tuple(0 for _ in range(int(max_token_pieces))))
        mask_rows.append(tuple(0 for _ in range(int(max_token_pieces))))
    return tuple(span_rows), tuple(mask_rows)


def numeric_source_feature_matrix(
    row: dict[str, Any],
    *,
    visual_dim: int,
    max_list_len: int = 5,
    value_vocab_size: int = 128,
) -> tuple[list[list[float]], list[int]]:
    """Encode input-list numbers as source-slot features.

    This is an input representation, not a solver: it exposes numeric value and
    source position so a learned binder/core can decide which slots matter.
    """

    if int(visual_dim) <= 0:
        raise ValueError("visual_dim must be positive")
    if int(max_list_len) <= 0:
        raise ValueError("max_list_len must be positive")
    if int(value_vocab_size) <= 1:
        raise ValueError("value_vocab_size must be greater than 1")
    values = row_input_list(row) or []
    features = [[0.0] * int(visual_dim) for _ in range(int(max_list_len))]
    mask = [0] * int(max_list_len)
    pos_den = float(max(1, int(max_list_len) - 1))
    value_den = float(max(1, int(value_vocab_size) - 1))
    for index, value in enumerate(values[: int(max_list_len)]):
        value = int(value)
        class_id = max(0, min(value, int(value_vocab_size) - 1))
        row_features = features[index]
        row_features[0] = float(class_id) / value_den
        if int(visual_dim) > 1:
            row_features[1] = 1.0 if value % 2 == 0 else 0.0
        if int(visual_dim) > 2:
            row_features[2] = float(index) / pos_den
        if int(visual_dim) > 3:
            row_features[3] = 1.0
        value_offset = 4 + class_id
        if 0 <= value_offset < int(visual_dim):
            row_features[value_offset] = 1.0
        pos_offset = 4 + int(value_vocab_size) + index
        if 0 <= pos_offset < int(visual_dim):
            row_features[pos_offset] = 1.0
        mask[index] = 1
    return features, mask


def parse_int_scalar_state(text: str) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def row_list_state_values(row: dict[str, Any], raw: Any) -> list[int] | None:
    values = parse_int_list_state(str(raw))
    if values is not None:
        return values
    family = str(row.get("task_family") or row.get("category") or "").strip()
    if family != "list_transform":
        return None
    scalar = parse_int_scalar_state(str(raw))
    if scalar is not None:
        return [int(scalar)]
    if str(raw).strip().upper() == "EMPTY":
        return []
    return None


def mixed_even_offsets(
    row: dict[str, Any],
    depth_targets: dict[str, Any] | None,
) -> list[int] | None:
    base = row_mixed_list_base(row)
    if base is None:
        return None
    raw_length = row.get("list_length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            return None
        return [offset for offset in range(length) if (base + offset) % 2 == 0]
    if isinstance(depth_targets, dict):
        first_state = depth_targets.get("1")
        values = parse_int_list_state(str(first_state)) if first_state else None
        if values is not None:
            return [int(value) - base for value in values]
    return None


def relative_list_value_classes(
    values: list[int],
    *,
    base: int,
    max_slots: int,
    slot_vocab_size: int,
) -> list[int] | None:
    if max_slots <= 0 or slot_vocab_size <= 1:
        return None
    doubled_base = 2 * base
    anchor = doubled_base if values and min(values) >= doubled_base else base
    classes: list[int] = []
    for value in values[:max_slots]:
        class_id = int(value) - int(anchor) + 1
        if class_id <= 0 or class_id >= int(slot_vocab_size):
            return None
        classes.append(class_id)
    classes.extend([0] * (int(max_slots) - len(classes)))
    return classes


def absolute_list_value_classes(
    values: list[int],
    *,
    max_slots: int,
    slot_vocab_size: int,
) -> list[int] | None:
    if max_slots <= 0 or slot_vocab_size <= 1:
        return None
    classes: list[int] = []
    for value in values[:max_slots]:
        class_id = int(value) + 1
        if class_id <= 0 or class_id >= int(slot_vocab_size):
            return None
        classes.append(class_id)
    classes.extend([0] * (int(max_slots) - len(classes)))
    return classes


def source_position_list_classes(
    row: dict[str, Any],
    values: list[int],
    *,
    doubled: bool,
    max_slots: int,
    slot_vocab_size: int,
) -> list[int] | None:
    if max_slots <= 0 or slot_vocab_size <= 1:
        return None
    source_values = row_input_list(row)
    if not source_values:
        return None
    used = [False] * len(source_values)
    classes: list[int] = []
    for value in values[:max_slots]:
        source_value = int(value)
        if doubled:
            if source_value % 2 != 0:
                return None
            source_value //= 2
        match_index: int | None = None
        for index, candidate in enumerate(source_values):
            if used[index]:
                continue
            if int(candidate) == source_value:
                match_index = index
                break
        if match_index is None:
            return None
        used[match_index] = True
        class_id = int(match_index) + 1
        if class_id <= 0 or class_id >= int(slot_vocab_size):
            return None
        classes.append(class_id)
    classes.extend([0] * (int(max_slots) - len(classes)))
    return classes


def relative_scalar_value_classes(
    value: int,
    *,
    base: int,
    even_offsets: list[int],
    max_slots: int,
    slot_vocab_size: int,
) -> list[int] | None:
    if max_slots <= 0 or slot_vocab_size <= 1 or not even_offsets:
        return None
    coeff = 2 * len(even_offsets)
    residual = int(value) - int(coeff) * int(base)
    classes = [coeff + 1, residual + 1]
    if any(class_id < 0 or class_id >= int(slot_vocab_size) for class_id in classes):
        return None
    padded = classes[: int(max_slots)]
    padded.extend([0] * (int(max_slots) - len(padded)))
    return padded


def relative_scalar_affine_value_classes(
    row: dict[str, Any],
    value: int,
    *,
    base: int,
    max_slots: int,
    slot_vocab_size: int,
) -> list[int] | None:
    if max_slots <= 0 or slot_vocab_size <= 1:
        return None
    raw_coeff = row.get("scalar_coeff", row.get("affine_coeff"))
    if raw_coeff is None:
        return None
    try:
        coeff = int(raw_coeff)
    except (TypeError, ValueError):
        return None
    residual = int(value) - int(coeff) * int(base)
    classes = [coeff + 1, residual + 1]
    if any(class_id < 0 or class_id >= int(slot_vocab_size) for class_id in classes):
        return None
    padded = classes[: int(max_slots)]
    padded.extend([0] * (int(max_slots) - len(padded)))
    return padded


def algorithmic_targets_from_row(
    row: dict[str, Any],
    *,
    num_steps: int,
    max_slots: int,
    slot_vocab_size: int,
    list_class_mode: str | None = None,
) -> tuple[list[int], list[list[int]]]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(max_slots) <= 0:
        raise ValueError("max_slots must be positive")
    if int(slot_vocab_size) <= 1:
        raise ValueError("slot_vocab_size must be greater than 1")
    base = row_mixed_list_base(row)
    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        return (
            [-100] * int(num_steps),
            [[-100] * int(max_slots) for _ in range(int(num_steps))],
        )
    even_offsets = mixed_even_offsets(row, depth_targets) or []
    list_mode = str(
        list_class_mode
        or row.get("role_value_list_class_mode")
        or row.get("list_class_mode")
        or "source_position"
    ).strip().lower()
    source_copy_no_doubled = bool(row.get("role_value_source_copy_no_doubled"))
    kind_targets: list[int] = []
    slot_targets: list[list[int]] = []
    for depth in range(1, int(num_steps) + 1):
        raw = depth_targets.get(str(depth))
        if raw is None:
            kind_targets.append(-100)
            slot_targets.append([-100] * int(max_slots))
            continue
        values = row_list_state_values(row, raw)
        if values is not None:
            if base is None:
                classes = None
                if list_mode != "absolute":
                    classes = source_position_list_classes(
                        row,
                        values,
                        doubled=(int(depth) > 1 and not source_copy_no_doubled),
                        max_slots=int(max_slots),
                        slot_vocab_size=int(slot_vocab_size),
                    )
                if classes is None:
                    classes = absolute_list_value_classes(
                        values,
                        max_slots=int(max_slots),
                        slot_vocab_size=int(slot_vocab_size),
                    )
            else:
                classes = relative_list_value_classes(
                    values,
                    base=int(base),
                    max_slots=int(max_slots),
                    slot_vocab_size=int(slot_vocab_size),
                )
            if classes is None:
                kind_targets.append(-100)
                slot_targets.append([-100] * int(max_slots))
            else:
                kind_targets.append(1)
                slot_targets.append(classes)
            continue
        value = parse_int_scalar_state(str(raw))
        if value is not None:
            if base is None:
                kind_targets.append(-100)
                slot_targets.append([-100] * int(max_slots))
                continue
            classes = relative_scalar_value_classes(
                value,
                base=int(base),
                even_offsets=even_offsets,
                max_slots=int(max_slots),
                slot_vocab_size=int(slot_vocab_size),
            )
            if classes is None:
                kind_targets.append(-100)
                slot_targets.append([-100] * int(max_slots))
            else:
                kind_targets.append(2)
                slot_targets.append(classes)
            continue
        kind_targets.append(-100)
        slot_targets.append([-100] * int(max_slots))
    return kind_targets, slot_targets


def role_value_targets_from_row(
    row: dict[str, Any],
    *,
    num_steps: int,
    num_roles: int,
    value_vocab_size: int,
    list_class_mode: str | None = None,
    supervise_nulls: bool = False,
) -> list[list[int]]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(num_roles) < 4:
        raise ValueError("num_roles must be at least 4")
    if int(value_vocab_size) <= 1:
        raise ValueError("value_vocab_size must be greater than 1")
    base = row_mixed_list_base(row)
    depth_targets = row.get("depth_targets")
    targets = [[-100] * int(num_roles) for _ in range(int(num_steps))]
    if not isinstance(depth_targets, dict):
        return targets
    max_list_fields = max(1, (int(num_roles) - 2) // 2)
    doubled_role_start = max_list_fields
    scalar_coeff_role = 2 * max_list_fields
    scalar_residual_role = scalar_coeff_role + 1
    if scalar_residual_role >= int(num_roles):
        raise ValueError("num_roles must fit list/doubled/scalar roles")
    even_offsets = mixed_even_offsets(row, depth_targets) or []
    list_mode = str(
        list_class_mode
        or row.get("role_value_list_class_mode")
        or row.get("list_class_mode")
        or "source_position"
    ).strip().lower()
    source_copy_no_doubled = bool(row.get("role_value_source_copy_no_doubled"))
    supervise_nulls = bool(
        supervise_nulls or row.get("role_value_supervise_null_slots")
    )
    for depth in range(1, int(num_steps) + 1):
        raw = depth_targets.get(str(depth))
        if raw is None:
            continue
        values = row_list_state_values(row, raw)
        if values is not None:
            if base is None:
                is_doubled = int(depth) > 1 and not source_copy_no_doubled
                classes = None
                if list_mode != "absolute":
                    classes = source_position_list_classes(
                        row,
                        values,
                        doubled=is_doubled,
                        max_slots=max_list_fields,
                        slot_vocab_size=int(value_vocab_size),
                    )
                if classes is None:
                    classes = absolute_list_value_classes(
                        values,
                        max_slots=max_list_fields,
                        slot_vocab_size=int(value_vocab_size),
                    )
            else:
                classes = relative_list_value_classes(
                    values,
                    base=int(base),
                    max_slots=max_list_fields,
                    slot_vocab_size=int(value_vocab_size),
                )
            if classes is None:
                continue
            is_doubled = (
                is_doubled if base is None else bool(values and min(values) >= 2 * int(base))
            )
            role_start = doubled_role_start if is_doubled else 0
            for index, class_id in enumerate(classes):
                role_index = role_start + index
                if role_index < int(num_roles):
                    if int(class_id) > 0 or supervise_nulls:
                        targets[depth - 1][role_index] = int(class_id)
            continue
        value = parse_int_scalar_state(str(raw))
        if value is not None:
            if base is None:
                continue
            classes = relative_scalar_value_classes(
                value,
                base=int(base),
                even_offsets=even_offsets,
                max_slots=2,
                slot_vocab_size=int(value_vocab_size),
            )
            if classes is None or len(classes) < 2:
                continue
            targets[depth - 1][scalar_coeff_role] = int(classes[0])
            targets[depth - 1][scalar_residual_role] = int(classes[1])
    return targets


def role_value_initial_targets_from_row(
    row: dict[str, Any],
    *,
    num_steps: int,
    num_roles: int,
    value_vocab_size: int,
    list_class_mode: str | None = None,
    include_metadata: bool = False,
) -> list[list[int]]:
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(num_roles) < 4:
        raise ValueError("num_roles must be at least 4")
    if int(value_vocab_size) <= 1:
        raise ValueError("value_vocab_size must be greater than 1")
    targets = [[-100] * int(num_roles) for _ in range(int(num_steps))]
    base = row_mixed_list_base(row)
    max_list_fields = max(1, (int(num_roles) - 2) // 2)
    if base is None:
        list_mode = str(
            list_class_mode
            or row.get("role_value_list_class_mode")
            or row.get("list_class_mode")
            or "source_position"
        ).strip().lower()
        values = row_input_list(row)
        if not values:
            return targets
        for role_index, value in enumerate(values[:max_list_fields]):
            class_id = (
                int(value) + 1
                if list_mode == "absolute"
                else int(role_index) + 1
            )
            if 0 <= class_id < int(value_vocab_size):
                targets[0][role_index] = int(class_id)
        return targets

    raw_length = row.get("list_length")
    try:
        length = int(raw_length)
    except (TypeError, ValueError):
        length = 0
    doubled_role_start = max_list_fields
    for offset in range(min(length, max_list_fields)):
        class_id = int(offset) + 1
        if 0 <= class_id < int(value_vocab_size):
            targets[0][offset] = int(class_id)
    scalar_coeff_role = 2 * max_list_fields
    scalar_residual_role = scalar_coeff_role + 1
    if scalar_coeff_role < int(num_roles):
        targets[0][scalar_coeff_role] = int(base) % 2 + 1
    if bool(include_metadata) and scalar_residual_role < int(num_roles):
        depth_targets = row.get("depth_targets")
        even_offsets = (
            mixed_even_offsets(row, depth_targets)
            if isinstance(depth_targets, dict)
            else None
        )
        if even_offsets:
            coeff_class = 2 * len(even_offsets) + 1
            if 0 <= coeff_class < int(value_vocab_size):
                targets[0][doubled_role_start] = int(coeff_class)
        try:
            offset_class = int(row.get("mixed_offset")) + 1
        except (TypeError, ValueError):
            offset_class = -1
        if 0 <= offset_class < int(value_vocab_size):
            targets[0][scalar_residual_role] = int(offset_class)
    return targets


def _last_labelled_depth(depth_targets: dict[str, Any]) -> int | None:
    depths: list[int] = []
    for raw_key, raw_value in depth_targets.items():
        if raw_value is None:
            continue
        try:
            depths.append(int(raw_key))
        except (TypeError, ValueError):
            continue
    return max(depths) if depths else None


def typed_algorithmic_field_targets_from_row(
    row: dict[str, Any],
    *,
    num_steps: int,
    max_list_slots: int,
    offset_vocab_size: int,
    scalar_vocab_size: int,
) -> dict[str, Any]:
    """Return role-separated algorithmic state targets for mixed list arithmetic.

    Kind ids are field types, not padding:
      0 = raw list, 1 = doubled list, 2 = scalar.
    """

    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(max_list_slots) <= 0:
        raise ValueError("max_list_slots must be positive")
    if int(offset_vocab_size) <= 1:
        raise ValueError("offset_vocab_size must be greater than 1")
    if int(scalar_vocab_size) <= 1:
        raise ValueError("scalar_vocab_size must be greater than 1")
    targets = {
        "kind": [-100] * int(num_steps),
        "raw_list_offsets": [
            [-100] * int(max_list_slots) for _ in range(int(num_steps))
        ],
        "doubled_list_offsets": [
            [-100] * int(max_list_slots) for _ in range(int(num_steps))
        ],
        "scalar_coeff": [-100] * int(num_steps),
        "scalar_offset": [-100] * int(num_steps),
        "scalar_residual": [-100] * int(num_steps),
        "scalar_residual_delta": [-100] * int(num_steps),
        "final_residual": [-100] * int(num_steps),
    }
    base = row_mixed_list_base(row)
    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        return targets
    even_offsets = mixed_even_offsets(row, depth_targets) or []
    final_depth = _last_labelled_depth(depth_targets)
    finality_targets = row.get("transition_finality_targets")
    if not isinstance(finality_targets, dict):
        finality_targets = {}
    previous_scalar_residual: int | None = None
    residual_delta_center = int(scalar_vocab_size) // 2
    scalar_offset_class: int | None = None
    raw_offset = row.get("mixed_offset", row.get("subtract_offset"))
    if raw_offset is not None:
        try:
            parsed_offset = int(raw_offset)
        except (TypeError, ValueError):
            parsed_offset = None
        if parsed_offset is not None:
            candidate = int(parsed_offset) + 1
            if 0 <= candidate < int(scalar_vocab_size):
                scalar_offset_class = int(candidate)
    for depth in range(1, int(num_steps) + 1):
        raw = depth_targets.get(str(depth))
        if raw is None:
            continue
        values = row_list_state_values(row, raw)
        if values is not None:
            if base is None:
                is_doubled = int(depth) > 1
                classes = source_position_list_classes(
                    row,
                    values,
                    doubled=is_doubled,
                    max_slots=int(max_list_slots),
                    slot_vocab_size=int(offset_vocab_size),
                )
                if classes is None:
                    classes = absolute_list_value_classes(
                        values,
                        max_slots=int(max_list_slots),
                        slot_vocab_size=int(offset_vocab_size),
                    )
            else:
                is_doubled = bool(values and min(values) >= 2 * int(base))
                classes = relative_list_value_classes(
                    values,
                    base=int(base),
                    max_slots=int(max_list_slots),
                    slot_vocab_size=int(offset_vocab_size),
                )
            if classes is None:
                continue
            step_index = depth - 1
            if is_doubled:
                targets["kind"][step_index] = 1
                targets["doubled_list_offsets"][step_index] = classes
            else:
                targets["kind"][step_index] = 0
                targets["raw_list_offsets"][step_index] = classes
            continue
        value = parse_int_scalar_state(str(raw))
        if value is not None:
            if base is None:
                continue
            classes = None
            if even_offsets:
                classes = relative_scalar_value_classes(
                    value,
                    base=int(base),
                    even_offsets=even_offsets,
                    max_slots=2,
                    slot_vocab_size=int(scalar_vocab_size),
                )
            if classes is None:
                classes = relative_scalar_affine_value_classes(
                    row,
                    value,
                    base=int(base),
                    max_slots=2,
                    slot_vocab_size=int(scalar_vocab_size),
                )
            if classes is None or len(classes) < 2:
                continue
            step_index = depth - 1
            targets["kind"][step_index] = 2
            targets["scalar_coeff"][step_index] = int(classes[0])
            if scalar_offset_class is not None:
                targets["scalar_offset"][step_index] = int(scalar_offset_class)
            targets["scalar_residual"][step_index] = int(classes[1])
            if previous_scalar_residual is not None:
                signed_delta = int(classes[1]) - int(previous_scalar_residual)
                delta_class = int(signed_delta) + int(residual_delta_center)
                if 0 <= delta_class < int(scalar_vocab_size):
                    targets["scalar_residual_delta"][step_index] = int(delta_class)
            previous_scalar_residual = int(classes[1])
            finality_value = finality_targets.get(str(depth))
            try:
                is_final = int(finality_value) == 1
            except (TypeError, ValueError):
                is_final = final_depth is not None and int(depth) == int(final_depth)
            if is_final:
                targets["final_residual"][step_index] = int(classes[1])
    return targets


def score_typed_algorithmic_field_predictions(
    *,
    predicted: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    for key in TYPED_ALGORITHMIC_FIELD_KEYS:
        if key not in predicted or key not in target:
            raise ValueError(f"missing typed algorithmic field: {key}")
    if len(predicted["kind"]) != len(target["kind"]):
        raise ValueError("predicted and target kind sequences must have equal length")

    correct_fields = 0
    total_fields = 0
    correct_content_fields = 0
    total_content_fields = 0
    exact_steps = 0
    total_steps = 0
    labelled_fields: list[dict[str, Any]] = []

    def _record(
        *,
        step_index: int,
        field: str,
        predicted_id: int,
        target_id: int,
        slot_index: int | None = None,
    ) -> bool:
        nonlocal correct_fields, total_fields
        nonlocal correct_content_fields, total_content_fields
        hit = int(predicted_id) == int(target_id)
        total_fields += 1
        correct_fields += int(hit)
        if int(target_id) > 0:
            total_content_fields += 1
            correct_content_fields += int(hit)
        labelled_fields.append(
            {
                "step_index": int(step_index),
                "depth": int(step_index) + 1,
                "field": field,
                "slot_index": None if slot_index is None else int(slot_index),
                "predicted": int(predicted_id),
                "target": int(target_id),
                "correct": bool(hit),
            }
        )
        return bool(hit)

    for step_index, (pred_kind, target_kind) in enumerate(
        zip(predicted["kind"], target["kind"])
    ):
        if int(target_kind) < 0:
            continue
        step_total = 0
        step_correct = 0
        if _record(
            step_index=step_index,
            field="kind",
            predicted_id=int(pred_kind),
            target_id=int(target_kind),
        ):
            step_correct += 1
        step_total += 1

        for field in ("raw_list_offsets", "doubled_list_offsets"):
            pred_values = predicted[field][step_index]
            target_values = target[field][step_index]
            if len(pred_values) != len(target_values):
                raise ValueError(f"{field} width mismatch")
            for slot_index, (pred_id, target_id) in enumerate(
                zip(pred_values, target_values)
            ):
                if int(target_id) < 0:
                    continue
                if _record(
                    step_index=step_index,
                    field=field,
                    slot_index=slot_index,
                    predicted_id=int(pred_id),
                    target_id=int(target_id),
                ):
                    step_correct += 1
                step_total += 1

        for field in (
            "scalar_coeff",
            "scalar_offset",
            "scalar_residual",
            "final_residual",
        ):
            target_id = int(target[field][step_index])
            if target_id < 0:
                continue
            if _record(
                step_index=step_index,
                field=field,
                predicted_id=int(predicted[field][step_index]),
                target_id=target_id,
            ):
                step_correct += 1
            step_total += 1

        if step_total:
            total_steps += 1
            exact_steps += int(step_correct == step_total)

    return {
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "field_accuracy": float(correct_fields) / float(total_fields)
        if total_fields
        else 0.0,
        "correct_content_fields": correct_content_fields,
        "total_content_fields": total_content_fields,
        "content_field_accuracy": float(correct_content_fields)
        / float(total_content_fields)
        if total_content_fields
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
        "trace_exact": bool(total_steps and exact_steps == total_steps),
        "labelled_fields": labelled_fields,
    }


def score_role_value_predictions(
    *,
    predicted_values: list[list[int]],
    target_values: list[list[int]],
) -> dict[str, Any]:
    if len(predicted_values) != len(target_values):
        raise ValueError("predicted_values and target_values must have the same length")
    correct_values = 0
    total_values = 0
    exact_steps = 0
    total_steps = 0
    labelled_values: list[dict[str, Any]] = []
    for step_index, (predicted, target) in enumerate(
        zip(predicted_values, target_values)
    ):
        if len(predicted) != len(target):
            raise ValueError("predicted and target role values must have equal width")
        step_total = 0
        step_correct = 0
        for role_index, (predicted_id, target_id) in enumerate(zip(predicted, target)):
            if int(target_id) < 0:
                continue
            hit = int(predicted_id) == int(target_id)
            step_total += 1
            step_correct += int(hit)
            total_values += 1
            correct_values += int(hit)
            labelled_values.append(
                {
                    "step_index": int(step_index),
                    "depth": int(step_index) + 1,
                    "role_index": int(role_index),
                    "predicted": int(predicted_id),
                    "target": int(target_id),
                    "correct": bool(hit),
                }
            )
        if step_total:
            total_steps += 1
            exact_steps += int(step_correct == step_total)
    return {
        "correct_values": correct_values,
        "total_values": total_values,
        "value_accuracy": float(correct_values) / float(total_values)
        if total_values
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
        "trace_exact": bool(total_steps and exact_steps == total_steps),
        "labelled_values": labelled_values,
    }


def score_algorithmic_sequences(
    *,
    predicted_kinds: list[int],
    predicted_slots: list[list[int]],
    target_kinds: list[int],
    target_slots: list[list[int]],
) -> dict[str, Any]:
    if len(predicted_kinds) != len(target_kinds):
        raise ValueError("predicted_kinds and target_kinds must have the same length")
    if len(predicted_slots) != len(target_slots):
        raise ValueError("predicted_slots and target_slots must have the same length")
    correct_kinds = 0
    total_kinds = 0
    correct_slots = 0
    total_slots = 0
    correct_content_slots = 0
    total_content_slots = 0
    exact_steps = 0
    total_steps = 0
    labelled_steps: list[dict[str, Any]] = []
    for step_index, (pred_kind, target_kind) in enumerate(
        zip(predicted_kinds, target_kinds)
    ):
        if int(target_kind) >= 0:
            total_kinds += 1
            correct_kinds += int(int(pred_kind) == int(target_kind))
        if int(target_kind) < 0:
            continue
        if step_index >= len(predicted_slots) or step_index >= len(target_slots):
            raise ValueError("slot sequence length must match kind sequence length")
        if len(predicted_slots[step_index]) != len(target_slots[step_index]):
            raise ValueError("predicted and target slots must have equal width")
        step_slot_total = 0
        step_slot_correct = 0
        for slot_index, (pred_slot, target_slot) in enumerate(
            zip(predicted_slots[step_index], target_slots[step_index])
        ):
            if int(target_slot) < 0:
                continue
            hit = int(pred_slot) == int(target_slot)
            step_slot_total += 1
            step_slot_correct += int(hit)
            total_slots += 1
            correct_slots += int(hit)
            if int(target_slot) > 0:
                total_content_slots += 1
                correct_content_slots += int(hit)
            labelled_steps.append(
                {
                    "step_index": int(step_index),
                    "depth": int(step_index) + 1,
                    "slot_index": int(slot_index),
                    "predicted": int(pred_slot),
                    "target": int(target_slot),
                    "correct": bool(hit),
                }
            )
        if step_slot_total:
            exact_steps += int(
                int(pred_kind) == int(target_kind)
                and step_slot_correct == step_slot_total
            )
            total_steps += 1
    return {
        "correct_kinds": correct_kinds,
        "total_kinds": total_kinds,
        "kind_accuracy": float(correct_kinds) / float(total_kinds)
        if total_kinds
        else 0.0,
        "correct_slots": correct_slots,
        "total_slots": total_slots,
        "slot_accuracy": float(correct_slots) / float(total_slots)
        if total_slots
        else 0.0,
        "correct_content_slots": correct_content_slots,
        "total_content_slots": total_content_slots,
        "content_slot_accuracy": float(correct_content_slots)
        / float(total_content_slots)
        if total_content_slots
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
        "trace_exact": bool(total_steps and exact_steps == total_steps),
        "labelled_steps": labelled_steps,
    }
