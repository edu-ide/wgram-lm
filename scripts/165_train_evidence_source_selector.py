#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgram_lm.eval.memory_retrieval import (
    _lex_terms,
    canonical_answer_text,
    case_task_family,
    evidence_records,
    expected_unknown_case,
    load_cases,
    normalize_answer,
)


_DATE_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_AUTHORITY_POSITIVE_CUES = (
    "signed",
    "supervisor",
    "official",
    "verified",
    "서명",
    "운영 공지",
    "공지",
)
_AUTHORITY_NEGATIVE_CUES = (
    "anonymous",
    "unverified",
    "rumor",
    "익명",
)
_TEMPORAL_CURRENT_CUES = (
    "current",
    "currently",
    "latest",
    "newest",
    "현재",
    "최신",
)
_TEMPORAL_STALE_CUES = (
    "previous",
    "older",
    "old ",
    "deprecated",
    "discarded",
    "이전",
    "폐기",
    "과거",
)
_DECOY_CUES = (
    "decoy",
    "other-",
    "other_",
    "project other",
    "bay decoy",
    "가짜",
)


@dataclass(frozen=True)
class SourceExample:
    case: dict[str, Any]
    record: dict[str, Any]
    features: list[float]
    label_answer_source: int


class EvidenceSourceSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _text_non_ascii_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def _cue_count(text: str, cues: Iterable[str]) -> float:
    haystack = text.casefold()
    return float(sum(1 for cue in cues if cue in haystack))


def _latest_date_value(text: str) -> int:
    values: list[int] = []
    for year, month, day in _DATE_RE.findall(text):
        values.append(int(year) * 10000 + int(month) * 100 + int(day))
    return max(values) if values else 0


def _answer_alias_in_record(case: dict[str, Any], record: dict[str, Any]) -> bool:
    if expected_unknown_case(case):
        return False
    text_norm = normalize_answer(str(record.get("text", "")))
    for alias in case.get("answer_aliases") or []:
        alias_norm = normalize_answer(canonical_answer_text(str(alias)))
        if alias_norm and alias_norm in text_norm:
            return True
    return False


def feature_names() -> list[str]:
    return [
        "retrieval_score_scaled",
        "rank_scaled",
        "total_records_scaled",
        "query_overlap_scaled",
        "query_overlap_fraction",
        "source_overlap_scaled",
        "record_len_scaled",
        "question_len_scaled",
        "record_non_ascii_fraction",
        "question_non_ascii_fraction",
        "authority_positive_cues",
        "authority_negative_cues",
        "temporal_current_cues",
        "temporal_stale_cues",
        "decoy_cues",
        "date_year_scaled",
        "date_month_scaled",
        "date_day_scaled",
        "question_authority_cue",
        "question_temporal_cue",
        "question_who_cue",
        "question_where_cue",
        "question_code_cue",
        "question_label_cue",
        "task_conflict_hint",
        "task_multi_hop_hint",
        "task_abstention_hint",
    ]


def extract_features(
    case: dict[str, Any],
    record: dict[str, Any],
    *,
    retrieval_score: float = 1.0,
    rank: int = 0,
    total_records: int = 1,
) -> list[float]:
    question = str(case.get("question", ""))
    rec_text = f"{record.get('source', '')} {record.get('text', '')}"
    q_terms = _lex_terms(question)
    r_terms = _lex_terms(rec_text)
    overlap = q_terms & r_terms
    source_terms = _lex_terms(str(record.get("source", "")))
    source_overlap = q_terms & source_terms
    date_value = _latest_date_value(rec_text)
    year = date_value // 10000 if date_value else 0
    month = (date_value // 100) % 100 if date_value else 0
    day = date_value % 100 if date_value else 0
    q_cf = question.casefold()
    task = case_task_family(case)
    return [
        min(1.0, _safe_float(retrieval_score) / 10.0),
        min(1.0, max(0, int(rank)) / 16.0),
        min(1.0, max(1, int(total_records)) / 16.0),
        min(1.0, len(overlap) / 8.0),
        len(overlap) / max(1.0, float(len(q_terms))),
        min(1.0, len(source_overlap) / 8.0),
        min(1.0, len(str(record.get("text", ""))) / 256.0),
        min(1.0, len(question) / 256.0),
        _text_non_ascii_fraction(str(record.get("text", ""))),
        _text_non_ascii_fraction(question),
        min(1.0, _cue_count(rec_text, _AUTHORITY_POSITIVE_CUES) / 3.0),
        min(1.0, _cue_count(rec_text, _AUTHORITY_NEGATIVE_CUES) / 3.0),
        min(1.0, _cue_count(rec_text, _TEMPORAL_CURRENT_CUES) / 3.0),
        min(1.0, _cue_count(rec_text, _TEMPORAL_STALE_CUES) / 3.0),
        min(1.0, _cue_count(rec_text, _DECOY_CUES) / 2.0),
        max(0.0, min(1.0, (year - 2000) / 40.0)) if year else 0.0,
        month / 12.0 if month else 0.0,
        day / 31.0 if day else 0.0,
        1.0 if any(token in q_cf for token in ("passphrase", "인증", "암구호")) else 0.0,
        1.0 if any(token in q_cf for token in ("current", "현재", "최신")) else 0.0,
        1.0 if any(token in q_cf for token in ("who", "누구", "담당", "maintain")) else 0.0,
        1.0 if any(token in q_cf for token in ("where", "어디", "위치")) else 0.0,
        1.0 if any(token in q_cf for token in ("code", "코드", "문구", "표식")) else 0.0,
        1.0 if any(token in q_cf for token in ("label", "라벨", "badge")) else 0.0,
        1.0 if task == "conflict" else 0.0,
        1.0 if task == "multi_hop" else 0.0,
        1.0 if task == "abstention" else 0.0,
    ]


def build_examples(cases: Iterable[dict[str, Any]]) -> list[SourceExample]:
    examples: list[SourceExample] = []
    for case in cases:
        records = evidence_records(case, include_distractors=True)
        total = len(records)
        for rank, rec in enumerate(records):
            examples.append(
                SourceExample(
                    case=case,
                    record=rec,
                    features=extract_features(
                        case,
                        rec,
                        retrieval_score=1.0,
                        rank=rank,
                        total_records=total,
                    ),
                    label_answer_source=int(_answer_alias_in_record(case, rec)),
                )
            )
    return examples


def stable_split(
    cases: list[dict[str, Any]],
    *,
    calibration_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    def key(row: dict[str, Any]) -> str:
        raw = str(row.get("id", "")) + "\0" + str(row.get("question", ""))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    ordered = sorted(cases, key=key)
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
    return ordered[:cut], ordered[cut:]


def tensorize(examples: list[SourceExample]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([example.features for example in examples], dtype=torch.float32)
    y = torch.tensor([example.label_answer_source for example in examples], dtype=torch.float32)
    return x, y


def train_selector(
    examples: list[SourceExample],
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
) -> EvidenceSourceSelector:
    if not examples:
        raise ValueError("no source selector training examples")
    torch.manual_seed(seed)
    x, y = tensorize(examples)
    model = EvidenceSourceSelector(x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    positives = float(y.sum().item())
    negatives = float((1.0 - y).sum().item())
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def source_probabilities(
    model: EvidenceSourceSelector,
    examples: list[SourceExample],
) -> list[float]:
    if not examples:
        return []
    x, _ = tensorize(examples)
    probs = torch.sigmoid(model(x)).detach().cpu().tolist()
    return [float(value) for value in probs]


def evaluate_selector(
    examples: list[SourceExample],
    probabilities: list[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    case_rows: dict[str, list[tuple[SourceExample, float]]] = {}
    for example, prob in zip(examples, probabilities):
        predicted = prob >= threshold
        label = bool(example.label_answer_source)
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
        case_rows.setdefault(str(example.case.get("id", "")), []).append((example, prob))
    case_success = 0
    positive_cases = 0
    positive_success = 0
    negative_cases = 0
    negative_clean = 0
    for rows in case_rows.values():
        labels = [bool(example.label_answer_source) for example, _ in rows]
        preds = [prob >= threshold for _, prob in rows]
        if any(labels):
            positive_cases += 1
            if any(label and pred for label, pred in zip(labels, preds)):
                positive_success += 1
                case_success += 1
        else:
            negative_cases += 1
            if not any(preds):
                negative_clean += 1
                case_success += 1
    count = len(examples)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "count": count,
        "accuracy": (tp + tn) / count if count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "case_count": len(case_rows),
        "case_success": case_success,
        "case_success_rate": case_success / max(1, len(case_rows)),
        "positive_cases": positive_cases,
        "positive_case_recall": positive_success / max(1, positive_cases) if positive_cases else 0.0,
        "negative_cases": negative_cases,
        "negative_clean": negative_clean,
        "negative_clean_rate": negative_clean / max(1, negative_cases) if negative_cases else 0.0,
    }


def select_threshold(
    examples: list[SourceExample],
    probabilities: list[float],
) -> tuple[float, dict[str, Any]]:
    best: tuple[tuple[float, float, float, float], float, dict[str, Any]] | None = None
    for idx in range(5, 96):
        threshold = idx / 100.0
        metrics = evaluate_selector(examples, probabilities, threshold=threshold)
        rank = (
            float(metrics["case_success_rate"]),
            float(metrics["f1"]),
            -float(metrics["fp"]),
            -threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    if best is None:
        raise ValueError("no threshold candidates")
    return best[1], best[2]


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Evidence Source Selector",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Setup",
            "",
            f"- train cases: `{report['train_cases_jsonl']}`",
            f"- eval cases: `{report['eval_cases_jsonl']}`",
            f"- selected threshold: `{report['selected_threshold']:.2f}`",
            "",
            "## Metrics",
            "",
            "| Split | Case Success | F1 | Precision | Recall | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| train | {report['train_learned']['case_success_rate']:.4f} | "
                f"{report['train_learned']['f1']:.4f} | {report['train_learned']['precision']:.4f} | "
                f"{report['train_learned']['recall']:.4f} | {report['train_learned']['fp']} | "
                f"{report['train_learned']['fn']} |"
            ),
            (
                f"| eval | {report['eval_learned']['case_success_rate']:.4f} | "
                f"{report['eval_learned']['f1']:.4f} | {report['eval_learned']['precision']:.4f} | "
                f"{report['eval_learned']['recall']:.4f} | {report['eval_learned']['fp']} | "
                f"{report['eval_learned']['fn']} |"
            ),
            "",
            "## Boundary",
            "",
            "This selector learns which retrieved source record contains the answer-bearing "
            "span. It should be applied as a span-logit mask while preserving the full "
            "workspace evidence context, not as pre-forward evidence pruning.",
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a learned evidence answer-source selector.")
    ap.add_argument("--train-cases-jsonl", required=True)
    ap.add_argument("--eval-cases-jsonl", default="")
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--calibration-fraction", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3.0e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-eval-case-success", type=float, default=0.75)
    ap.add_argument("--min-eval-f1", type=float, default=0.70)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_cases = load_cases(args.train_cases_jsonl)
    if args.eval_cases_jsonl:
        eval_cases = load_cases(args.eval_cases_jsonl)
    else:
        train_cases, eval_cases = stable_split(
            train_cases,
            calibration_fraction=float(args.calibration_fraction),
        )
    train_examples = build_examples(train_cases)
    eval_examples = build_examples(eval_cases)
    model = train_selector(
        train_examples,
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        seed=int(args.seed),
    )
    train_probs = source_probabilities(model, train_examples)
    selected_threshold, train_learned = select_threshold(train_examples, train_probs)
    eval_probs = source_probabilities(model, eval_examples)
    eval_learned = evaluate_selector(
        eval_examples,
        eval_probs,
        threshold=selected_threshold,
    )
    failed: list[str] = []
    if float(eval_learned["case_success_rate"]) < float(args.min_eval_case_success):
        failed.append("eval_case_success_too_low")
    if float(eval_learned["f1"]) < float(args.min_eval_f1):
        failed.append("eval_f1_too_low")
    report = {
        "train_cases_jsonl": args.train_cases_jsonl,
        "eval_cases_jsonl": args.eval_cases_jsonl or args.train_cases_jsonl,
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "feature_names": feature_names(),
        "selected_threshold": selected_threshold,
        "train_learned": train_learned,
        "eval_learned": eval_learned,
        "failed_checks": failed,
        "status": "accepted" if not failed else "rejected",
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_names": report["feature_names"],
            "selected_threshold": selected_threshold,
            "args": vars(args),
            "report": report,
        },
        out_pt,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_md = Path(args.markdown_out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
