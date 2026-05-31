from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable


@dataclass(frozen=True)
class VerifiedSourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    adapter: str


DEFAULT_VERIFIED_SOURCES: dict[str, VerifiedSourceSpec] = {
    "gsm8k_train": VerifiedSourceSpec(
        name="gsm8k_train",
        dataset="openai/gsm8k",
        config="main",
        split="train",
        adapter="gsm8k",
    ),
    "math500_test": VerifiedSourceSpec(
        name="math500_test",
        dataset="HuggingFaceH4/MATH-500",
        config="default",
        split="test",
        adapter="math_answer",
    ),
    "numina_verifiable_train": VerifiedSourceSpec(
        name="numina_verifiable_train",
        dataset="gravermistakes/NuminaMath-1.5-RL-Verifiable",
        config="default",
        split="train",
        adapter="numina_math_verifiable",
    ),
    "openr1_math_verified_train": VerifiedSourceSpec(
        name="openr1_math_verified_train",
        dataset="open-r1/OpenR1-Math-220k",
        config="default",
        split="train",
        adapter="openr1_math_verified",
    ),
    "openmathinstruct2_train": VerifiedSourceSpec(
        name="openmathinstruct2_train",
        dataset="nvidia/OpenMathInstruct-2",
        config="default",
        split="train",
        adapter="openmathinstruct2",
    ),
    "proofwriter_validation": VerifiedSourceSpec(
        name="proofwriter_validation",
        dataset="renma/ProofWriter",
        config="default",
        split="validation",
        adapter="proofwriter",
    ),
    "clutrr_train": VerifiedSourceSpec(
        name="clutrr_train",
        dataset="CLUTRR/v1",
        config="gen_train234_test2to10",
        split="train",
        adapter="clutrr",
    ),
    "bbh_boolean_test": VerifiedSourceSpec(
        name="bbh_boolean_test",
        dataset="lukaemon/bbh",
        config="boolean_expressions",
        split="test",
        adapter="bbh",
    ),
}


def convert_verified_row(
    row: dict[str, Any],
    *,
    adapter: str,
    source_name: str,
    row_index: int,
) -> dict[str, Any]:
    converters: dict[str, Callable[[dict[str, Any]], tuple[str, str, str, list[str]]]] = {
        "gsm8k": _convert_gsm8k,
        "math_answer": _convert_math_answer,
        "numina_math_verifiable": _convert_numina_math_verifiable,
        "openr1_math_verified": _convert_openr1_math_verified,
        "openmathinstruct2": _convert_openmathinstruct2,
        "proofwriter": _convert_proofwriter,
        "clutrr": _convert_clutrr,
        "bbh": _convert_bbh,
    }
    if adapter not in converters:
        raise ValueError(f"unknown verified reasoning adapter: {adapter}")
    question, answer, task_family, choices = converters[adapter](row)
    question = _collapse_ws(question)
    answer = _clean_answer(answer)
    if not question:
        raise ValueError("verified row has empty question")
    if not answer:
        raise ValueError("verified row has empty answer")
    prompt = (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    choices = _unique([answer, *choices, *_default_negative_choices(answer, task_family)])
    return {
        "id": f"{source_name}-{int(row_index):06d}",
        "raw_intelligence_axis": "verified_reasoning_dataset",
        "category": task_family,
        "task_family": task_family,
        "reasoning_family": task_family,
        "expected_paradigm": "latent_recursive_or_direct",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 0,
        "source_dataset": source_name,
        "source_adapter": adapter,
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "answer_aliases": [answer],
        "choices": choices,
        "depth_targets": {"1": answer, "2": answer, "4": answer, "8": answer},
        "expected_unknown": answer.upper() == "UNKNOWN",
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "verified_label_source": "hf_gold_answer_or_test",
        "distill_policy": "verified_dataset_no_teacher_imitation",
    }


def _convert_gsm8k(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question = str(row.get("question", "")).strip()
    answer = _extract_hash_answer(str(row.get("answer", "")))
    return question, answer, "math_word_problem", []


def _convert_math_answer(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question = str(row.get("problem") or row.get("question") or "").strip()
    answer = str(row.get("answer") or row.get("expected_answer") or "").strip()
    subject = str(row.get("subject") or row.get("problem_type") or "math").strip().lower()
    return question, answer, f"math_{_slug(subject)}", []


def _convert_numina_math_verifiable(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    if str(row.get("problem_is_valid", "Yes")).strip().lower() not in {"yes", "true", "1"}:
        raise ValueError("NuminaMath problem is not verified valid")
    if str(row.get("solution_is_valid", "Yes")).strip().lower() not in {"yes", "true", "1"}:
        raise ValueError("NuminaMath solution is not verified valid")
    question = str(row.get("problem") or "").strip()
    answer = str(row.get("answer") or "").strip()
    kind = str(row.get("problem_type") or "math").strip().lower()
    return question, answer, f"math_{_slug(kind)}", []


def _convert_openr1_math_verified(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    correctness = row.get("correctness_math_verify")
    if isinstance(correctness, list) and correctness and not any(bool(item) for item in correctness):
        raise ValueError("OpenR1 row has no math-verified generation")
    question = str(row.get("problem") or "").strip()
    answer = str(row.get("answer") or "").strip()
    kind = str(row.get("problem_type") or row.get("question_type") or "math").strip().lower()
    return question, answer, f"math_{_slug(kind)}", []


def _convert_openmathinstruct2(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question = str(row.get("problem") or "").strip()
    answer = str(row.get("expected_answer") or row.get("answer") or "").strip()
    source = str(row.get("problem_source") or "openmath").strip().lower()
    return question, answer, f"math_{_slug(source)}", []


def _convert_proofwriter(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = _proofwriter_answer(row)
    full_question = f"{context}\n\n{question}" if context else question
    return full_question, answer, "logical_entailment", ["TRUE", "FALSE", "UNKNOWN"]


def _convert_clutrr(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    story = str(row.get("clean_story") or row.get("story") or "").strip()
    query = str(row.get("query") or "").strip()
    answer = str(row.get("target_text") or row.get("target") or "").strip()
    question = (
        f"{story}\n\nWhat is the family relationship for query {query}? "
        "Answer with the relation word only."
    )
    return question, answer, "relation_reasoning", []


def _convert_bbh(row: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question = str(row.get("input") or row.get("question") or "").strip()
    answer = str(row.get("target") or row.get("answer") or "").strip()
    return question, answer, "bbh_reasoning", []


def _extract_hash_answer(text: str) -> str:
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return text.strip()


def _proofwriter_answer(row: dict[str, Any]) -> str:
    answer = str(row.get("answer") or "").strip()
    options = row.get("options") or []
    if answer.upper() in {"A", "B", "C"}:
        index = ord(answer.upper()) - ord("A")
        if isinstance(options, list) and 0 <= index < len(options):
            answer = str(options[index])
    lowered = answer.lower()
    if "true" in lowered or lowered in {"a", "1"}:
        return "TRUE"
    if "false" in lowered or lowered in {"b", "0"}:
        return "FALSE"
    if "unknown" in lowered or lowered in {"c", "2"}:
        return "UNKNOWN"
    raise ValueError(f"cannot map ProofWriter answer: {answer!r}")


def _clean_answer(text: str) -> str:
    text = str(text).strip()
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if boxed:
        text = boxed.group(1).strip()
    return _collapse_ws(text)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or "unknown"


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
    return out


def _default_negative_choices(answer: str, task_family: str) -> list[str]:
    normalized = str(answer).strip()
    if re.fullmatch(r"-?\d+", normalized):
        value = int(normalized)
        candidates = [value - 1, value + 1]
        if value != 0:
            candidates.append(value * 2)
        candidates.append(value + 10)
        return [str(item) for item in candidates if item != value]
    upper = normalized.upper()
    if upper in {"TRUE", "FALSE"}:
        return ["FALSE" if upper == "TRUE" else "TRUE", "UNKNOWN"]
    if upper == "UNKNOWN":
        return ["TRUE", "FALSE"]
    if task_family == "relation_reasoning":
        return [
            "mother",
            "father",
            "daughter",
            "son",
            "sister",
            "brother",
            "wife",
            "husband",
            "aunt",
            "uncle",
            "niece",
            "nephew",
        ]
    if task_family.startswith("math_"):
        return ["0", "1", "-1", "2"]
    return []
