#!/usr/bin/env python3
"""Build deterministic English/Korean answer-only repair data.

This is not a broad-language proof. It creates a reusable bilingual scaffold
for the current QTRM-native language gate failures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FAMILIES = {
    "evidence": {
        "en_prompts": [
            "Why should evidence be checked before trusting a claim?",
            "How can evidence make a claim more trustworthy?",
            "Why is unsupported information risky?",
            "What should we inspect before accepting a claim?",
        ],
        "en_answers": [
            "Evidence should be checked because a claim can sound convincing even when it is not trustworthy.",
            "A trustworthy claim needs evidence that is specific, relevant, and possible to verify.",
            "Unsupported information is risky because it can lead to a confident but false conclusion.",
            "Before accepting a claim, inspect the evidence, the source, and whether the support really matches the claim.",
        ],
        "ko_prompts": [
            "주장을 믿기 전에 왜 근거를 확인해야 하나요?",
            "근거는 어떻게 주장의 신뢰도를 높이나요?",
            "근거 없는 정보는 왜 위험한가요?",
            "주장을 받아들이기 전에 무엇을 확인해야 하나요?",
        ],
        "ko_answers": [
            "주장을 믿기 전에 근거를 확인하면 그 주장이 신뢰할 만한지 판단할 수 있다.",
            "구체적이고 확인 가능한 근거는 주장의 신뢰도를 높인다.",
            "근거 없는 정보는 그럴듯해 보여도 잘못된 결론으로 이어질 수 있어 위험하다.",
            "주장을 받아들이기 전에는 근거, 출처, 그리고 그 근거가 주장과 맞는지 확인해야 한다.",
        ],
    },
    "writing": {
        "en_prompts": [
            "How can short sentences make writing clearer?",
            "How can a writer make one idea easy to follow?",
            "What should a writer check when revising a sentence?",
            "Why do clear subjects help readers?",
        ],
        "en_answers": [
            "Short sentences make writing clearer because each sentence can carry one idea without confusing the reader.",
            "A writer can make one idea easy to follow by using clear words, explicit subjects, and connected reasons.",
            "When revising a sentence, a writer should check the subject, the predicate, and whether the idea is clear.",
            "Clear subjects help readers understand who or what the sentence is about.",
        ],
        "ko_prompts": [
            "짧은 문장은 글을 어떻게 더 분명하게 만드나요?",
            "한 가지 생각을 쉽게 따라가게 하려면 어떻게 써야 하나요?",
            "문장을 고칠 때 무엇을 먼저 확인해야 하나요?",
            "분명한 주어는 독자에게 어떤 도움을 주나요?",
        ],
        "ko_answers": [
            "짧은 문장은 한 문장에 한 가지 생각을 담게 해 글을 더 분명하게 만든다.",
            "한 가지 생각을 쉽게 따라가게 하려면 글에서 주어와 서술어를 분명하게 쓰고 이유를 연결해야 한다.",
            "문장을 고칠 때는 먼저 주어와 서술어가 맞는지, 글의 뜻이 분명한지 확인해야 한다.",
            "분명한 주어는 독자가 문장에서 누가 무엇을 하는지 쉽게 이해하게 돕는다.",
        ],
    },
    "uncertainty": {
        "en_prompts": [
            "What should an answer do when facts are uncertain?",
            "Why should a model avoid pretending to know?",
            "How should weak evidence be handled?",
            "What is a careful answer when information is missing?",
        ],
        "en_answers": [
            "When facts are uncertain, an answer should say what is uncertain and avoid a confident guess.",
            "A model should avoid pretending to know because false confidence can mislead the user.",
            "Weak evidence should be handled by saying the support is weak and looking for stronger facts.",
            "When information is missing, a careful answer explains the gap and avoids guessing beyond the evidence.",
        ],
        "ko_prompts": [
            "사실이 불확실할 때 답변은 어떻게 해야 하나요?",
            "모델은 왜 아는 척을 피해야 하나요?",
            "근거가 약할 때는 어떻게 답해야 하나요?",
            "정보가 부족한 문제에는 어떤 답변이 조심스러운가요?",
        ],
        "ko_answers": [
            "사실이 불확실할 때 답변은 불확실한 부분을 말하고 근거 없는 추측을 피해야 한다.",
            "모델이 아는 척을 피해야 하는 이유는 거짓 확신이 사용자를 잘못 이끌 수 있기 때문이다.",
            "근거가 약할 때는 근거가 약하다고 밝히고 더 강한 사실을 확인해야 한다.",
            "정보가 부족하면 부족한 부분을 설명하고 근거를 넘어선 추측을 하지 않는 답변이 조심스럽다.",
        ],
    },
    "source_date": {
        "en_prompts": [
            "Why does the date of a source matter?",
            "Why can old information become unreliable?",
            "How does time affect whether a fact is current?",
            "When should a source be checked again?",
        ],
        "en_answers": [
            "The date of a source matters because a fact can change and old information may no longer be current.",
            "Old information can become unreliable when the situation changes after the source was written.",
            "Time affects reliability because a fact that was true before may not be current now.",
            "A source should be checked again when the topic changes quickly or when the source is old.",
        ],
        "ko_prompts": [
            "출처의 날짜는 왜 중요한가요?",
            "오래된 정보는 왜 믿기 어려울 수 있나요?",
            "시간은 사실의 현재성에 어떤 영향을 주나요?",
            "언제 출처를 다시 확인해야 하나요?",
        ],
        "ko_answers": [
            "출처의 날짜는 사실이 현재도 맞는지 판단하게 해 주기 때문에 중요하다.",
            "오래된 정보는 상황이 바뀌면 현재 사실과 달라질 수 있어 믿기 어려울 수 있다.",
            "시간이 지나면 사실이 바뀔 수 있으므로 정보가 현재에도 유효한지 확인해야 한다.",
            "주제가 빠르게 변하거나 출처가 오래되었으면 출처를 다시 확인해야 한다.",
        ],
    },
    "comparison": {
        "en_prompts": [
            "How can a comparison help choose between two claims?",
            "What makes a comparison useful?",
            "Why compare two explanations?",
            "How can differences reveal stronger support?",
        ],
        "en_answers": [
            "A comparison helps choose between claims by showing where they differ and which one has stronger support.",
            "A useful comparison names the important differences and connects them to evidence.",
            "Comparing two explanations helps reveal which one fits the facts more directly.",
            "Differences can reveal stronger support when one claim explains the evidence better than the other.",
        ],
        "ko_prompts": [
            "두 주장을 비교하면 무엇을 알 수 있나요?",
            "좋은 비교는 어떤 기준을 가져야 하나요?",
            "두 설명을 비교하는 이유는 무엇인가요?",
            "차이를 보면 더 강한 근거를 어떻게 알 수 있나요?",
        ],
        "ko_answers": [
            "두 주장을 비교하면 차이점과 공통점을 보고 어떤 주장이 더 강한 근거를 갖는지 알 수 있다.",
            "좋은 비교는 중요한 기준을 정하고 그 기준을 근거와 연결해야 한다.",
            "두 설명을 비교하면 어떤 설명이 사실과 더 잘 맞는지 판단할 수 있다.",
            "차이를 보면 어느 주장이 근거를 더 잘 설명하는지 확인할 수 있다.",
        ],
    },
    "repeated_test": {
        "en_prompts": [
            "What makes a repeated test useful?",
            "How do repeated measurements help science?",
            "What is a stable result?",
            "Why repeat an experiment under the same conditions?",
        ],
        "en_answers": [
            "A repeated test is useful because it shows whether a result is stable rather than accidental.",
            "Repeated measurements help science by showing whether the same pattern appears again.",
            "A stable result appears again when the same test is repeated under fair conditions.",
            "Repeating an experiment under the same conditions helps check whether the result is reliable.",
        ],
        "ko_prompts": [
            "반복 실험은 왜 결과 판단에 도움이 되나요?",
            "반복 측정은 과학에서 어떤 도움을 주나요?",
            "안정적인 결과는 무엇인가요?",
            "같은 조건에서 실험을 반복하는 이유는 무엇인가요?",
        ],
        "ko_answers": [
            "반복 실험은 결과가 우연이 아니라 안정적인지 판단하는 데 도움이 된다.",
            "반복 측정은 같은 패턴이 다시 나타나는지 보여 주어 과학적 판단을 돕는다.",
            "안정적인 결과는 같은 조건에서 반복해도 비슷하게 나타나는 결과다.",
            "같은 조건에서 실험을 반복하면 결과가 믿을 만한지 확인할 수 있다.",
        ],
    },
}


def make_records(repeats: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family_name, family in FAMILIES.items():
        for language in ("en", "ko"):
            prompts = family[f"{language}_prompts"]
            answers = family[f"{language}_answers"]
            for repeat in range(max(1, int(repeats))):
                for index, prompt in enumerate(prompts):
                    answer = answers[(index + repeat) % len(answers)]
                    rows.append(
                        {
                            "text": f"User: {prompt}\nAssistant: {answer}",
                            "source": f"bilingual_core_repair:{family_name}:{language}",
                        }
                    )
    return rows


def build_repair(args: argparse.Namespace) -> dict[str, object]:
    rows = make_records(int(args.repeats))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = Counter(str(row["source"]) for row in rows)
    report = {
        "status": "complete",
        "out": str(out_path),
        "records": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "repeats": int(args.repeats),
    }
    out_path.with_suffix(out_path.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="local_eval/qtrm_native_bilingual_core_repair_20260515.jsonl",
    )
    parser.add_argument("--repeats", type=int, default=4)
    return parser


def main() -> None:
    print(json.dumps(build_repair(build_arg_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
