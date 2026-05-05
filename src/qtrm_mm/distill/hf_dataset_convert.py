from __future__ import annotations

import re
from typing import Any, Callable

from .teacher_schema import Qwen36TeacherRecord, TeacherEvidenceDoc


def convert_hf_row(row: dict[str, Any], *, adapter: str) -> Qwen36TeacherRecord:
    converters: dict[str, Callable[[dict[str, Any]], Qwen36TeacherRecord]] = {
        "yana_reasoning_dpo": _convert_yana_reasoning_dpo,
        "noesis_text_sft": _convert_noesis_text_sft,
        "ragognize": _convert_ragognize,
        "halluclaim_76k": _convert_halluclaim_76k,
    }
    if adapter not in converters:
        raise ValueError(f"unknown HF distill adapter: {adapter}")
    return converters[adapter](row)


def _convert_yana_reasoning_dpo(row: dict[str, Any]) -> Qwen36TeacherRecord:
    prompt = _strip_xml_tag(str(row.get("prompt", "")), "question").strip()
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    answer = _clean_answer(_extract_answer(chosen) or chosen)
    rejected_answer = _clean_answer(_extract_answer(rejected) or rejected)
    if _normalize_text(answer) == _normalize_text(rejected_answer):
        rejected_answer = ""
    return Qwen36TeacherRecord(
        prompt=prompt or _extract_problem(chosen),
        answer=answer,
        rejected_answer=rejected_answer,
        trace_summary=_extract_thinking(chosen),
        teacher_model="hf:Yana/ft-llm-2026-reasoning-dpo",
    )


def _convert_noesis_text_sft(row: dict[str, Any]) -> Qwen36TeacherRecord:
    text = str(row.get("text", ""))
    prompt, answer = _split_user_assistant(text)
    if _normalize_text(prompt) == _normalize_text(answer):
        raise ValueError("NOESIS row does not contain a separate assistant answer")
    if "<think>" in answer.lower() and "</think>" not in answer.lower():
        raise ValueError("NOESIS row contains an unclosed think block")
    thinking = _extract_think_block(answer)
    answer = _remove_think_blocks(answer).strip()
    return Qwen36TeacherRecord(
        prompt=prompt,
        answer=answer,
        trace_summary=thinking,
        teacher_model="hf:AMAImedia/NOESIS-50K-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54",
    )


def _convert_ragognize(row: dict[str, Any]) -> Qwen36TeacherRecord:
    prompt = str(row.get("user_prompt") or row.get("question") or "").strip()
    docs = [
        TeacherEvidenceDoc(
            doc_id=idx + 1,
            text=str(doc.get("text", "")),
            source=str(doc.get("title", "")),
        )
        for idx, doc in enumerate(row.get("documents") or [])
        if isinstance(doc, dict) and str(doc.get("text", "")).strip()
    ]
    answer = _first_valid_ragognize_answer(row.get("responses"))
    evidence_doc_ids = _match_evidence_doc_ids(answer, docs)
    evidence_spans = [docs[idx - 1].text for idx in evidence_doc_ids if 0 < idx <= len(docs)]
    if not evidence_doc_ids and docs:
        evidence_doc_ids = [1] if bool(row.get("answerable", True)) else []
    if not evidence_spans and evidence_doc_ids:
        evidence_spans = [docs[evidence_doc_ids[0] - 1].text]

    return Qwen36TeacherRecord(
        prompt=prompt,
        answer=answer or ("UNKNOWN" if not row.get("answerable", True) else ""),
        evidence_ids=[f"doc-{idx}" for idx in evidence_doc_ids],
        evidence_spans=evidence_spans,
        memory_docs=docs,
        target_doc_ids=evidence_doc_ids,
        teacher_model="hf:F4biian/RAGognize",
    )


def _convert_halluclaim_76k(row: dict[str, Any]) -> Qwen36TeacherRecord:
    claim = str(row.get("claim") or row.get("statement") or row.get("response") or "").strip()
    document = str(
        row.get("document")
        or row.get("doc")
        or row.get("context")
        or row.get("evidence")
        or ""
    ).strip()
    label = str(
        row.get("label")
        or row.get("type")
        or row.get("gold")
        or row.get("classification")
        or ""
    ).strip().lower()
    supported = label in {"supported", "support", "true", "entailed", "faithful", "0"}
    hallucinated = "hallucinated" in label or label in {"unsupported", "refuted", "false", "1"}
    if hallucinated:
        supported = False
    answer = "SUPPORTED" if supported else "REFUTED"
    rejected = "REFUTED" if supported else "SUPPORTED"
    docs = [TeacherEvidenceDoc(doc_id=1, text=document)] if document else []
    return Qwen36TeacherRecord(
        prompt=f"Determine whether the claim is supported by the evidence.\nClaim: {claim}",
        answer=answer,
        rejected_answer=rejected,
        evidence_spans=[document] if document else [],
        memory_docs=docs,
        target_doc_ids=[1] if document else [],
        teacher_model="hf:lrsbrgrn/HalluClaim-76k",
    )


def _strip_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def _extract_problem(text: str) -> str:
    return _strip_xml_tag(text, "Problem").strip()


def _extract_thinking(text: str) -> str:
    return _strip_xml_tag(text, "Thinking").strip()


def _extract_answer(text: str) -> str:
    return _strip_xml_tag(text, "Answer").strip()


def _clean_answer(text: str) -> str:
    text = text.strip()
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if boxed:
        return boxed.group(1).strip()
    return text


def _split_user_assistant(text: str) -> tuple[str, str]:
    match = re.search(r"User:\s*(.*?)\s*Assistant:\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if "\n" in text:
        prompt, answer = text.split("\n", 1)
        return prompt.strip(), answer.strip()
    return text.strip(), text.strip()


def _extract_think_block(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _first_valid_ragognize_answer(responses: Any) -> str:
    if not isinstance(responses, dict):
        return ""
    fallback = ""
    for value in responses.values():
        if not isinstance(value, dict):
            continue
        output = str(
            value.get("text")
            or value.get("output")
            or value.get("response")
            or value.get("answer")
            or ""
        ).strip()
        if not output:
            continue
        fallback = fallback or output
        result = (
            value.get("details", {})
            .get("annotations", {})
            .get("result", {})
        )
        if isinstance(result, dict) and result.get("all_valid") is True:
            return output
    return fallback


def _match_evidence_doc_ids(answer: str, docs: list[TeacherEvidenceDoc]) -> list[int]:
    if not answer:
        return []
    answer_lower = answer.lower()
    matched = []
    for doc in docs:
        text_lower = doc.text.lower()
        if text_lower and (text_lower in answer_lower or _overlap_score(answer_lower, text_lower) >= 0.35):
            matched.append(doc.doc_id)
    return matched[:3]


def _overlap_score(answer_lower: str, text_lower: str) -> float:
    answer_words = {w for w in re.findall(r"\w+", answer_lower) if len(w) > 2}
    text_words = {w for w in re.findall(r"\w+", text_lower) if len(w) > 2}
    if not answer_words or not text_words:
        return 0.0
    return len(answer_words & text_words) / len(answer_words)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
