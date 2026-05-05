#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


PROMPTS_BY_CATEGORY = {
    "normal_qa": [
        "Explain quantum entanglement in simple terms.",
        "What is photosynthesis? Answer in three short sentences.",
        "Why do seasons happen on Earth?",
        "Explain what a compiler does.",
        "What is the difference between RAM and storage?",
        "Describe how vaccines help the immune system.",
        "Why is the sky blue during the day?",
        "What is a database index used for?",
        "Explain inflation in economics in plain language.",
        "What is a black hole?",
    ],
    "math_reasoning": [
        "Solve step by step: if x + 3 = 7, what is x?",
        "A box has 12 red balls and 8 blue balls. What fraction are blue?",
        "If a train travels 90 km in 1.5 hours, what is its average speed?",
        "Solve: 2x + 5 = 17.",
        "A rectangle is 6 cm by 4 cm. What is its area and perimeter?",
        "If 15% of a number is 30, what is the number?",
        "A shop gives a 20% discount on a $50 item. What is the final price?",
        "There are 24 students split equally into 6 groups. How many per group?",
        "If a recipe for 4 people uses 2 cups of rice, how much for 10 people?",
        "What is the next number in the sequence 2, 4, 8, 16?",
    ],
    "evidence_check": [
        "Determine whether the claim is supported by the evidence. Claim: The Eiffel Tower is in Berlin. Evidence: The Eiffel Tower is in Paris.",
        "Determine whether the claim is supported by the evidence. Claim: Water boils at 100 C at sea level. Evidence: At standard atmospheric pressure, water boils at 100 C.",
        "Determine whether the claim is supported by the evidence. Claim: Marie Curie won two Nobel Prizes. Evidence: Marie Curie received Nobel Prizes in Physics and Chemistry.",
        "Determine whether the claim is supported by the evidence. Claim: The Moon is larger than Earth. Evidence: Earth has a larger diameter than the Moon.",
        "Determine whether the claim is supported by the evidence. Claim: Python is a compiled-only language. Evidence: Python code is usually executed by an interpreter.",
        "Given the evidence, answer yes or no. Evidence: Ada Lovelace wrote notes on the Analytical Engine. Claim: Ada Lovelace is associated with early computing.",
        "Given the evidence, answer yes or no. Evidence: The capital of Canada is Ottawa. Claim: Toronto is the capital of Canada.",
        "Given the evidence, answer yes or no. Evidence: Bees pollinate many flowering plants. Claim: Bees can help plants reproduce.",
        "Given the evidence, answer yes or no. Evidence: The Pacific Ocean is the largest ocean. Claim: The Atlantic Ocean is the largest ocean.",
        "Given the evidence, answer yes or no. Evidence: Insulin helps regulate blood sugar. Claim: Insulin is unrelated to blood sugar.",
    ],
    "korean_qa": [
        "양자 컴퓨팅이란 무엇인가요?",
        "한국어는 어떻게 발전했나요?",
        "광합성이 무엇인지 쉽게 설명해 주세요.",
        "인공지능과 머신러닝의 차이를 설명해 주세요.",
        "블록체인의 핵심 아이디어는 무엇인가요?",
        "기후 변화의 주요 원인은 무엇인가요?",
        "민주주의의 장점과 한계를 짧게 설명해 주세요.",
        "조선 시대 과학 기술의 예를 하나 들어 설명해 주세요.",
        "컴퓨터 메모리와 저장장치의 차이는 무엇인가요?",
        "수학에서 함수란 무엇인가요?",
    ],
    "repeat_stress": [
        "Answer once only: What is the capital of France?",
        "Give a direct answer only: 5 + 7 = ?",
        "Do not repeat the prompt. Explain recursion in one sentence.",
        "Classify the claim as supported or refuted, then stop. Claim: Cats are mammals. Evidence: Cats are warm-blooded vertebrates that nurse their young.",
        "Write exactly one sentence about gravity.",
        "Answer with one word: Is ice frozen water?",
        "Summarize this without repeating phrases: Data helps decisions when it is accurate and relevant.",
        "Choose A, B, C, or D only. Which is a prime number? A) 4 B) 6 C) 7 D) 9",
        "State the final answer only. If y - 2 = 6, what is y?",
        "Explain the term evidence in one concise sentence and stop.",
    ],
}


def build_prompt_suite() -> list[dict]:
    rows = []
    prompt_id = 0
    for category, prompts in PROMPTS_BY_CATEGORY.items():
        for text in prompts:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "text": text,
                }
            )
            prompt_id += 1
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a fixed prompt suite for Qwen-Scope repeat-gate evaluation.")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = build_prompt_suite()
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
