#!/usr/bin/env python3
"""QTRM-native language-first bootstrap runner.

This is the language-first counterpart to the reasoning gates. It keeps the
runtime path donorless:

    tokens -> QTRM-native embeddings -> mandatory recurrent core -> LM logits

Teacher/Qwen/Qwopus artifacts may appear only as offline text data. No teacher
model is loaded by this script.
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wgram_lm.tst import multi_hot_cross_entropy, next_token_bags, superpose_embeddings


def load_text_probe_module():
    path = Path(__file__).with_name("336_train_qtrm_native_text_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_text_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_text_probe = load_text_probe_module()


TINY_STORIES = (
    "Mina has a red cup. She fills it with water. The cup is not full, so she adds a little more. "
    "When the cup is full, Mina stops. She smiles because she solved the small problem.\n",
    "A boy puts three blocks on the table. One block falls down. He counts again and sees two blocks. "
    "He picks up the fallen block and makes the tower stable.\n",
    "The sun is bright in the morning. A small plant leans toward the light. The child turns the pot, "
    "and the plant slowly grows straight again.\n",
    "A robot reads a short note. The note says to bring the blue book. The robot checks the color first, "
    "then carries the correct book to the desk.\n",
    "민아는 작은 컵에 물을 담았다. 컵이 가득 차자 민아는 물을 멈추었다. 민아는 차분하게 확인하고 웃었다.\n",
    "학생은 공책에 짧은 문장을 썼다. 문장이 틀리면 다시 읽고 고쳤다. 그래서 글은 조금씩 더 분명해졌다.\n",
)


TEXTBOOK_SNIPPETS = (
    "A clear sentence has a subject and a predicate. The subject tells who or what the sentence is about. "
    "The predicate tells what happens or what is true.\n",
    "Addition combines quantities. If a basket has two apples and we add three apples, the basket has five apples. "
    "The answer follows from counting all items together.\n",
    "A cause is an event that helps produce another event. An effect is what happens after the cause. "
    "Good explanations connect causes and effects in order.\n",
    "When reading evidence, first identify the claim. Then find the facts that support or weaken it. "
    "If two facts conflict, compare their source and time.\n",
    "한국어 문장은 보통 주어와 서술어로 의미를 만든다. 글을 이해하려면 먼저 누가 무엇을 하는지 확인한다.\n",
    "좋은 설명은 결론만 말하지 않는다. 이유를 짧게 제시하고, 그 이유가 결론과 어떻게 연결되는지 보여준다.\n",
)

SURFACE_ANSWER_SNIPPETS = (
    "User: A clear explanation starts with\nAssistant: a simple claim, a short reason, and a clear connection.\n",
    "User: Explain a small cause and effect.\nAssistant: A cause is what makes something happen. An effect is what happens after that cause.\n",
    "User: Continue the story about Mina.\nAssistant: Mina puts the cup on the table and checks that the water stays inside.\n",
    "User: What makes writing easier to understand?\nAssistant: Short sentences, clear subjects, and connected reasons make writing easier to understand.\n",
    "User: 한국어로 짧게 설명하면\nAssistant: 좋은 설명은 핵심을 먼저 말하고, 그 이유를 짧게 덧붙인다.\n",
    "User: 문장을 잘 쓰려면?\nAssistant: 먼저 주어와 서술어를 분명히 하고, 필요한 이유를 차분하게 이어 쓴다.\n",
)

DEFAULT_REPAIR_SEED_TEXTS = (
    "User: Why should evidence be checked?\nAssistant: ||"
    "User: How can writing become clearer?\nAssistant: ||"
    "User: 좋은 답변은 무엇인가요?\nAssistant: "
)

DEFAULT_REPAIR_SEED_EXPECTATIONS = {
    "Why should evidence be checked?": ["evidence", "unsupported", "claims", "wrong"],
    "How can writing become clearer?": ["writing", "sentences", "subjects", "reasons"],
    "좋은 답변은 무엇인가요?": ["좋은 답변", "질문", "근거", "추측"],
}

GATE_ANCHOR_SNIPPETS = (
    (
        "User: Why should evidence be checked?\n"
        "Assistant: Evidence should be checked because unsupported claims can sound convincing while still being wrong.\n"
    ),
    (
        "User: How can writing become clearer?\n"
        "Assistant: Writing becomes clearer when sentences are short, subjects are explicit, and reasons are connected.\n"
    ),
    (
        "User: 좋은 답변은 무엇인가요?\n"
        "Assistant: 좋은 답변은 질문에 직접 답하고, 근거를 분명히 말하며, 모르면 추측하지 않는 답변이다.\n"
    ),
    (
        "User: What should an answer do when facts are uncertain?\n"
        "Assistant: When facts are uncertain, an answer should say what is uncertain and avoid guessing.\n"
    ),
    (
        "User: Why does the date of a source matter?\n"
        "Assistant: The date of a source matters because old information may not be current when facts change.\n"
    ),
    (
        "User: What makes a repeated test useful?\n"
        "Assistant: A repeated test is useful because it shows whether a result is reliable under the same conditions.\n"
    ),
    (
        "User: 반복 실험은 왜 결과 판단에 도움이 되나요?\n"
        "Assistant: 반복 실험은 같은 결과가 다시 나오는지 보여 주어 결과가 안정적인지 판단하게 해 준다.\n"
    ),
    (
        "User: 주장을 믿기 전에 왜 근거를 확인해야 하나요?\n"
        "Assistant: 주장을 믿기 전에 근거를 확인하면 그 주장이 신뢰할 만한지 판단할 수 있다.\n"
    ),
    (
        "User: 출처가 왜 답변의 신뢰도를 높이나요?\n"
        "Assistant: 출처는 답변이 어떤 근거에 기대는지 보여 주어 답변의 신뢰도를 높인다.\n"
    ),
    (
        "User: Why should a model avoid pretending to know?\n"
        "Assistant: A model should avoid pretending to know because uncertain answers can mislead people.\n"
    ),
    (
        "User: 문장을 고칠 때 무엇을 먼저 확인해야 하나요?\n"
        "Assistant: 문장을 고칠 때는 주어와 서술어가 맞는지 먼저 확인해야 한다.\n"
    ),
    (
        "User: How can a summary stay faithful to the original text?\n"
        "Assistant: A faithful summary keeps the central meaning and removes details that do not change it.\n"
    ),
    (
        "User: How can an answer separate a claim from evidence?\n"
        "Assistant: An answer can separate a claim from evidence by stating the claim first and then listing the evidence that supports it.\n"
    ),
)


def _qa_rows(items: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(f"User: {question}\nAssistant: {answer}\n" for question, answer in items)


def build_surface_answer_families() -> dict[str, tuple[str, ...]]:
    """Build answer-only prompt families for balanced language bootstrap."""
    basics = (
        ("Why does ice melt?", "Ice melts when heat gives its molecules enough energy to move more freely."),
        ("Why do plants need light?", "Plants use light to make food, so light helps them grow."),
        ("Why does practice help?", "Practice helps because each attempt gives feedback that can guide the next attempt."),
        ("What is an example?", "An example is a concrete case that makes an abstract idea easier to see."),
        ("What is a contrast?", "A contrast shows how two things differ in a way that matters for the question."),
        ("What is a stable result?", "A stable result appears again when the same test is repeated under fair conditions."),
        ("What makes a repeated test useful?", "A repeated test is useful because it shows whether a result is reliable under the same conditions."),
        ("반복 실험은 왜 결과 판단에 도움이 되나요?", "반복 실험은 같은 결과가 다시 나오는지 보여 주어 결과가 안정적인지 판단하게 해 준다."),
    )
    answer_quality = (
        ("What is a careful answer?", "A careful answer states the main point and avoids adding details it cannot support."),
        ("What is a good answer?", "A good answer addresses the question directly, gives a clear reason, and stops before guessing."),
        ("좋은 답변은 어떻게 시작하나요?", "좋은 답변은 먼저 핵심 결론을 말하고, 그다음 필요한 이유를 짧게 붙인다."),
        ("좋은 답변은 무엇인가요?", "좋은 답변은 질문에 직접 답하고, 근거를 분명히 말하며, 모르면 추측하지 않는 답변이다."),
    )
    evidence = (
        ("Why should evidence be checked?", "Evidence should be checked because unsupported claims can sound convincing while still being wrong."),
        ("Why is checking a source important?", "Checking a source helps decide whether evidence is trustworthy and relevant."),
        ("What makes a source useful?", "A useful source is relevant, specific, and clear about where its information came from."),
        ("Why check dates in evidence?", "Dates matter because older information can become wrong when the situation changes."),
        ("Why can old information become wrong?", "Old information can become wrong when the situation changes after the source was written."),
        ("Why can old information become unreliable?", "Old information can become unreliable when a source is old and the situation has changed."),
        ("What should we do before trusting a claim?", "Before trusting a claim, we should look for support and compare the evidence."),
        ("왜 근거를 확인해야 하나요?", "근거를 확인하면 주장과 사실을 구분할 수 있고, 잘못된 결론을 줄일 수 있다."),
        ("주장을 믿기 전에 왜 근거를 확인해야 하나요?", "주장을 믿기 전에 근거를 확인하면 그 주장이 신뢰할 만한지 판단할 수 있다."),
        ("주장을 받아들이기 전에 무엇을 확인해야 하나요?", "주장을 받아들이기 전에는 근거, 출처, 그리고 그 근거가 주장과 맞는지 확인해야 한다."),
        ("신뢰도는 어떻게 보나요?", "출처, 날짜, 구체성, 다른 근거와의 일치 여부를 함께 보면 신뢰도를 판단하기 쉽다."),
        ("무엇이 답변을 믿을 만하게 만드나요?", "믿을 만한 답변은 분명한 근거와 확인 가능한 출처를 함께 제시한다."),
        ("출처가 왜 답변의 신뢰도를 높이나요?", "출처는 답변이 어떤 근거에 기대는지 보여 주어 답변의 신뢰도를 높인다."),
        ("출처의 날짜는 왜 중요한가요?", "출처의 날짜는 사실이 현재도 맞는지 판단하게 해 주기 때문에 중요하다."),
        ("오래된 정보는 왜 틀릴 수 있나요?", "오래된 정보는 상황이 바뀌면 현재 사실과 달라질 수 있어 틀릴 수 있다."),
        ("How can an answer separate a claim from evidence?", "An answer can separate a claim from evidence by stating the claim first and then listing the evidence that supports it."),
    )
    writing = (
        ("How should a short paragraph begin?", "A short paragraph should begin with one clear idea that the next sentences explain."),
        ("How can writing become clearer?", "Writing becomes clearer when sentences are short, subjects are explicit, and reasons are connected."),
        ("How do short sentences help readers?", "Short sentences help readers because clear wording reduces confusion."),
        ("What helps readers understand a paragraph quickly?", "A clear paragraph helps readers by using short sentences and one main idea."),
        ("Why keep sentences short?", "Short sentences reduce ambiguity and make the relationship between ideas easier to follow."),
        ("Why revise writing?", "Revision catches unclear wording and helps the final text match the intended meaning."),
        ("문장을 쉽게 쓰려면?", "한 문장에 한 가지 생각을 담고, 주어와 서술어를 분명하게 맞춘다."),
        ("문장을 고칠 때 무엇을 먼저 확인해야 하나요?", "문장을 고칠 때는 주어와 서술어가 맞는지 먼저 확인해야 한다."),
        ("글을 고쳐 쓰는 이유는?", "고쳐 쓰면 어색한 표현을 줄이고 말하려는 뜻을 더 정확하게 만들 수 있다."),
    )
    planning_summary = (
        ("What is a simple plan?", "A simple plan names the goal, the next action, and the sign that the action worked."),
        ("Why compare two claims?", "Comparing claims helps find what they share, where they conflict, and which has better support."),
        ("What is a good summary?", "A good summary keeps the central idea and removes details that do not change the meaning."),
        ("How can a summary stay faithful to the original text?", "A faithful summary keeps the central meaning and removes details that do not change it."),
        ("요약은 무엇인가요?", "요약은 중요한 뜻을 남기고 반복되거나 덜 중요한 세부사항을 줄이는 것이다."),
        ("비교가 왜 필요한가요?", "비교를 하면 두 주장 사이의 공통점과 차이를 보고 더 강한 근거를 고를 수 있다."),
        ("작은 계획은 무엇인가요?", "작은 계획은 목표와 바로 할 일, 그리고 성공을 확인할 기준을 함께 정하는 것이다."),
    )
    uncertainty = (
        ("How can a model avoid guessing?", "It can answer only when evidence is enough and ask for more information when it is not."),
        ("How should a model respond when evidence is weak?", "When evidence is weak, a model should avoid guessing and ask for stronger support."),
        ("What should an answer do if the facts are uncertain?", "If facts are uncertain, an answer should say so and avoid a confident guess."),
        ("How can an answer avoid pretending to know?", "It can avoid pretending by naming uncertainty and asking for better facts."),
        ("What should an answer do when facts are uncertain?", "When facts are uncertain, an answer should say what is uncertain and avoid guessing."),
        ("Why should a model avoid pretending to know?", "A model should avoid pretending to know because uncertain answers can mislead people."),
        ("모르는 문제는 어떻게 다뤄야 하나요?", "정보가 부족하면 추측하지 말고 부족한 부분을 밝힌 뒤 확인할 방법을 찾는다."),
        ("답변이 추측이 아닌지 어떻게 알 수 있나요?", "답변이 근거와 출처를 제시하고 불확실한 부분을 밝히면 추측을 줄일 수 있다."),
        ("불확실한 사실에는 어떻게 답해야 하나요?", "사실이 불확실하면 단정하지 말고 근거가 부족하다고 말해야 한다."),
        ("근거가 약하면 어떻게 해야 하나요?", "근거가 약하면 추측하지 말고 더 믿을 만한 자료를 확인해야 한다."),
    )
    stories = (
        ("Mina checks the cup", "Mina sees the cup is full, moves it slowly, and keeps the water from spilling."),
        ("The robot finds the book", "The robot reads the label, compares the colors, and brings the blue book to the desk."),
        ("A child fixes a tower", "The child notices the weak block, moves it to the bottom, and builds the tower again."),
        ("A student studies rain", "The student watches clouds gather, writes a note, and explains why the ground becomes wet."),
        ("The cook reads a recipe", "The cook checks the steps, measures the flour, and waits until the bread turns brown."),
        ("The team tests a bridge", "The team adds weight slowly, records each result, and stops before the bridge bends."),
        ("A gardener turns a pot", "The gardener turns the pot toward the light and checks the plant again the next day."),
        ("The class solves a puzzle", "The class lists the clues, removes the impossible answers, and chooses the one that fits."),
    )
    korean_context = (
        ("시간 정보는 왜 중요한가요?", "시간 정보는 어떤 사실이 지금도 유효한지 판단하는 데 필요하다."),
        ("실험 결과를 믿으려면?", "같은 조건에서 다시 확인했을 때 비슷한 결과가 나와야 더 믿을 수 있다."),
        ("설명에서 예시는 왜 쓰나요?", "예시는 추상적인 말을 실제 상황처럼 보여 주어 이해를 돕는다."),
    )
    bilingual_pairs = (
        ("Translate simply: clear reason", "간단히 옮기면 '분명한 이유'라는 뜻이다."),
        ("Translate simply: careful reading", "간단히 옮기면 '주의 깊은 읽기'라는 뜻이다."),
        ("Translate simply: trusted source", "간단히 옮기면 '믿을 만한 출처'라는 뜻이다."),
        ("한국어로 answer-only를 설명하면?", "답변 표면에는 최종 답만 두고, 사고 과정은 모델 내부에서 처리한다는 뜻이다."),
        ("한국어로 native model을 설명하면?", "외부 donor 없이 자체 embedding과 recurrent core로 logits를 만드는 모델이라는 뜻이다."),
        ("한국어로 strict gate를 설명하면?", "느슨한 성공 신호가 아니라 실제 실패 패턴까지 잡는 평가 기준이라는 뜻이다."),
    )

    return {
        "basics": _qa_rows(basics),
        "answer_quality": _qa_rows(answer_quality),
        "evidence": _qa_rows(evidence),
        "writing": _qa_rows(writing),
        "planning_summary": _qa_rows(planning_summary),
        "uncertainty": _qa_rows(uncertainty),
        "stories": tuple(
            f"User: Continue the story: {title}.\nAssistant: {continuation}\n"
            for title, continuation in stories
        ),
        "korean_context": _qa_rows(korean_context),
        "bilingual": _qa_rows(bilingual_pairs),
    }


def round_robin_family_rows(families: dict[str, tuple[str, ...]], max_items: int) -> tuple[str, ...]:
    if int(max_items) <= 0:
        return ()
    rows: list[str] = []
    names = sorted(families)
    index = 0
    while len(rows) < int(max_items):
        progressed = False
        for name in names:
            family = families[name]
            if index < len(family):
                rows.append(family[index])
                progressed = True
                if len(rows) >= int(max_items):
                    break
        if not progressed:
            index = 0
            continue
        index += 1
    return tuple(rows)


def build_diverse_surface_answer_snippets(max_items: int) -> tuple[str, ...]:
    """Build deterministic answer-only prompt/response snippets."""
    families = build_surface_answer_families()
    rows = round_robin_family_rows(families, int(max_items))
    if len(rows) >= int(max_items):
        return tuple(rows[: int(max_items)])
    return rows


@dataclass(frozen=True)
class HFTokenizerAdapter:
    name: str
    tokenizer: Any

    @classmethod
    def from_name(cls, name: str) -> "HFTokenizerAdapter":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "transformers is required for --tokenizer-name. "
                "Install transformers or omit --tokenizer-name for char mode."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(name=name, tokenizer=tokenizer)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def eos_token_id(self) -> int | None:
        value = self.tokenizer.eos_token_id
        return int(value) if value is not None else None

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))


@dataclass(frozen=True)
class CompactHFTokenizerAdapter:
    name: str
    tokenizer: Any
    compact_to_hf: tuple[int, ...]
    hf_to_compact: dict[int, int]
    unk_compact_id: int
    eos_compact_id: int | None

    @classmethod
    def from_text(
        cls,
        name: str,
        text: str,
        *,
        max_size: int,
        min_count: int,
        extra_texts: tuple[str, ...] = (),
    ) -> "CompactHFTokenizerAdapter":
        base = HFTokenizerAdapter.from_name(name)
        counts: Counter[int] = Counter(base.encode(text))
        for extra_text in extra_texts:
            counts.update(base.encode(extra_text))
        ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))

        required_ids = {
            token_id
            for token_id in (
                base.tokenizer.eos_token_id,
                base.tokenizer.pad_token_id,
                base.tokenizer.unk_token_id,
            )
            if token_id is not None
        }
        min_count = max(1, int(min_count))
        selected = {
            token_id
            for token_id, count in counts.items()
            if int(count) >= min_count
        }
        selected.update(int(token_id) for token_id in required_ids)
        if int(max_size) > 0 and len(selected) > int(max_size):
            kept: set[int] = set(int(token_id) for token_id in required_ids)
            for token_id, _count in ranked:
                kept.add(int(token_id))
                if len(kept) >= int(max_size):
                    break
            selected = kept

        compact_to_hf = tuple(sorted(int(token_id) for token_id in selected))
        if not compact_to_hf:
            raise ValueError("compact tokenizer vocabulary is empty")
        hf_to_compact = {int(token_id): index for index, token_id in enumerate(compact_to_hf)}
        unk_hf_id = base.tokenizer.unk_token_id
        if unk_hf_id is None or int(unk_hf_id) not in hf_to_compact:
            eos_id = base.tokenizer.eos_token_id
            unk_hf_id = None
            for token_id, _count in ranked:
                candidate = int(token_id)
                if candidate in hf_to_compact and (
                    eos_id is None or candidate != int(eos_id)
                ):
                    unk_hf_id = candidate
                    break
        if unk_hf_id is None or int(unk_hf_id) not in hf_to_compact:
            unk_hf_id = base.tokenizer.eos_token_id
        if unk_hf_id is None or int(unk_hf_id) not in hf_to_compact:
            unk_hf_id = compact_to_hf[0]
        eos_id = base.tokenizer.eos_token_id
        return cls(
            name=name,
            tokenizer=base.tokenizer,
            compact_to_hf=compact_to_hf,
            hf_to_compact=hf_to_compact,
            unk_compact_id=int(hf_to_compact[int(unk_hf_id)]),
            eos_compact_id=(
                int(hf_to_compact[int(eos_id)])
                if eos_id is not None and int(eos_id) in hf_to_compact
                else None
            ),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CompactHFTokenizerAdapter":
        name = str(payload.get("name", ""))
        base = HFTokenizerAdapter.from_name(name)
        compact_to_hf = tuple(int(token_id) for token_id in payload.get("compact_to_hf", ()))
        if not compact_to_hf:
            raise ValueError("compact tokenizer checkpoint is missing compact_to_hf")
        hf_to_compact = {int(token_id): index for index, token_id in enumerate(compact_to_hf)}
        eos_compact_id = payload.get("eos_compact_id")
        return cls(
            name=name,
            tokenizer=base.tokenizer,
            compact_to_hf=compact_to_hf,
            hf_to_compact=hf_to_compact,
            unk_compact_id=int(payload.get("unk_compact_id", 0)),
            eos_compact_id=int(eos_compact_id) if eos_compact_id is not None else None,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.compact_to_hf)

    @property
    def eos_token_id(self) -> int | None:
        return self.eos_compact_id

    def encode(self, text: str) -> list[int]:
        hf_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            self.hf_to_compact.get(int(token_id), int(self.unk_compact_id))
            for token_id in hf_ids
        ]

    def decode(self, token_ids: list[int]) -> str:
        hf_ids: list[int] = []
        for token_id in token_ids:
            index = int(token_id)
            if 0 <= index < len(self.compact_to_hf):
                hf_ids.append(int(self.compact_to_hf[index]))
            else:
                hf_ids.append(int(self.compact_to_hf[int(self.unk_compact_id)]))
        return str(self.tokenizer.decode(hf_ids, skip_special_tokens=True))


@dataclass(frozen=True)
class ByteBPETokenizerAdapter:
    """Small project-local byte-level BPE tokenizer for QTRM-native bootstrap."""

    name: str
    tokenizer: Any
    eos_token: str
    unk_token: str
    eos_id: int | None

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        vocab_size: int,
        min_frequency: int,
        eos_token: str,
        unk_token: str,
        extra_texts: tuple[str, ...] = (),
    ) -> "ByteBPETokenizerAdapter":
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "tokenizers is required for --train-byte-bpe-tokenizer."
            ) from exc

        tokenizer = Tokenizer(models.BPE(unk_token=str(unk_token)))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=max(256, int(vocab_size)),
            min_frequency=max(1, int(min_frequency)),
            special_tokens=[str(eos_token), str(unk_token)],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        corpus = [str(text), *(str(extra) for extra in extra_texts if str(extra))]
        tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
        eos_id = tokenizer.token_to_id(str(eos_token))
        return cls(
            name=f"byte_bpe_{int(vocab_size)}",
            tokenizer=tokenizer,
            eos_token=str(eos_token),
            unk_token=str(unk_token),
            eos_id=int(eos_id) if eos_id is not None else None,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ByteBPETokenizerAdapter":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("tokenizers is required to load byte-BPE checkpoints.") from exc
        raw_json = str(payload.get("tokenizer_json", ""))
        if not raw_json:
            raise ValueError("byte-BPE tokenizer checkpoint is missing tokenizer_json")
        tokenizer = Tokenizer.from_str(raw_json)
        eos_id = payload.get("eos_id")
        return cls(
            name=str(payload.get("name", "byte_bpe")),
            tokenizer=tokenizer,
            eos_token=str(payload.get("eos_token", "<|qtrm_eos|>")),
            unk_token=str(payload.get("unk_token", "<|qtrm_unk|>")),
            eos_id=int(eos_id) if eos_id is not None else None,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    @property
    def eos_token_id(self) -> int | None:
        return self.eos_id

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(str(text)).ids)

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode([int(token_id) for token_id in token_ids]))


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_globs(patterns: list[str]) -> str:
    parts: list[str] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for raw_path in sorted(glob.glob(str(pattern), recursive=True)):
            path = Path(raw_path)
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                parts.append(f"\n\n## FILE: {path}\n\n{read_text_file(path)}")
    return "\n".join(parts)


def extract_jsonl_text_records(path: str | Path) -> list[str]:
    parts: list[str] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(item, str):
            cleaned = clean_teacher_language_text(item)
            if cleaned:
                parts.append(cleaned)
            continue
        if not isinstance(item, dict):
            continue
        for key in (
            "text",
            "teacher_text",
            "completion",
            "continuation",
            "response",
            "output",
            "answer",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = clean_teacher_language_text(value)
                if cleaned:
                    parts.append(cleaned)
                break
        else:
            messages = item.get("messages")
            if isinstance(messages, list):
                joined: list[str] = []
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        joined.append(str(message["content"]))
                if joined:
                    cleaned = clean_teacher_language_text("\n".join(joined))
                    if cleaned:
                        parts.append(cleaned)
    return parts


def extract_jsonl_texts(path: str | Path, *, separator: str = "\n\n") -> str:
    return str(separator).join(extract_jsonl_text_records(path))


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r".*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def clean_teacher_language_text(text: str) -> str:
    """Keep language bootstrap answer-only/continuation-only.

    QTRM-native uses latent recurrence for reasoning, so visible teacher CoT is
    treated as language-data contamination here.
    """
    cleaned = strip_think_blocks(text)
    if "<think" in cleaned.lower() or "</think" in cleaned.lower():
        return ""
    # Drop common explicit reasoning labels from teacher artifacts.
    cleaned = re.sub(
        r"(?im)^\s*(thinking|reasoning|chain[- ]of[- ]thought)\s*:.*$",
        "",
        cleaned,
    ).strip()
    return cleaned


def repeated_block(
    lines: tuple[str, ...],
    repeats: int,
    *,
    shuffle: bool,
    seed: int,
    separator: str = "\n",
) -> str:
    rng = random.Random(int(seed))
    parts: list[str] = []
    for _ in range(max(0, int(repeats))):
        rows = list(lines)
        if bool(shuffle):
            rng.shuffle(rows)
        parts.extend(rows)
    return str(separator).join(parts)


def cap_text(text: str, limit: int) -> str:
    if int(limit) <= 0:
        return str(text)
    return str(text)[: int(limit)]


def capped_join(parts: tuple[str, ...], *, limit: int) -> str:
    clean = [str(part) for part in parts if str(part).strip()]
    if int(limit) <= 0:
        return "\n".join(clean)
    if not clean:
        return ""
    if len(clean) == 1:
        return cap_text(clean[0], int(limit))
    per_part = max(1, int(limit) // len(clean))
    capped = [part[:per_part] for part in clean]
    remaining = int(limit) - sum(len(part) for part in capped)
    if remaining > 0:
        for index, part in enumerate(clean):
            if remaining <= 0:
                break
            already = len(capped[index])
            extra = part[already : already + remaining]
            capped[index] += extra
            remaining -= len(extra)
    return "\n".join(part for part in capped if part.strip())


def language_record_separator(args: argparse.Namespace) -> str:
    separator = str(getattr(args, "record_separator", "auto"))
    if separator == "auto":
        if bool(getattr(args, "train_byte_bpe_tokenizer", False)):
            return f"\n{str(getattr(args, 'byte_bpe_eos_token', '<|qtrm_eos|>'))}\n"
        if str(getattr(args, "tokenizer_name", "")):
            return "\n<|endoftext|>\n"
        return "\n"
    return separator.encode("utf-8").decode("unicode_escape")


def build_stage_texts(args: argparse.Namespace) -> dict[str, str]:
    record_separator = language_record_separator(args)
    tiny = repeated_block(
        TINY_STORIES,
        int(args.tiny_repeats),
        shuffle=bool(args.shuffle_corpus),
        seed=int(args.seed),
        separator=record_separator,
    )
    textbook = repeated_block(
        TEXTBOOK_SNIPPETS,
        int(args.textbook_repeats),
        shuffle=bool(args.shuffle_corpus),
        seed=int(args.seed) + 17,
        separator=record_separator,
    )
    surface_answer_lines = SURFACE_ANSWER_SNIPPETS + build_diverse_surface_answer_snippets(
        int(args.diverse_surface_answer_count)
    )
    surface_answers = repeated_block(
        surface_answer_lines,
        int(args.surface_answer_repeats),
        shuffle=bool(args.shuffle_corpus),
        seed=int(args.seed) + 31,
        separator=record_separator,
    )

    external_parts: list[str] = []
    if args.text_file:
        external_parts.append(f"\n\n## FILE: {args.text_file}\n\n{read_text_file(args.text_file)}")
    if args.text_glob:
        external_parts.append(read_globs(list(args.text_glob)))
    external = "\n".join(part for part in external_parts if part.strip())

    teacher_parts: list[str] = []
    gate_anchor_repeats = int(getattr(args, "gate_anchor_repeats", 0))
    if gate_anchor_repeats > 0:
        anchors = repeated_block(
            GATE_ANCHOR_SNIPPETS,
            gate_anchor_repeats,
            shuffle=bool(args.shuffle_corpus),
            seed=int(args.seed) + 43,
            separator=record_separator,
        )
        teacher_parts.append(f"\n\n## GATE_ANCHORS\n\n{anchors}")
    for path in args.teacher_text_file or []:
        cleaned = clean_teacher_language_text(read_text_file(path))
        if cleaned:
            teacher_parts.append(f"\n\n## TEACHER_TEXT: {path}\n\n{cleaned}")
    for path in args.teacher_jsonl or []:
        records = extract_jsonl_texts(path, separator=record_separator)
        if records:
            teacher_parts.append(f"\n\n## TEACHER_JSONL: {path}\n\n{records}")
    for path in args.repair_jsonl or []:
        records = extract_jsonl_texts(path, separator=record_separator)
        if records:
            repeated_records = record_separator.join(
                records for _ in range(max(1, int(args.repair_jsonl_repeats)))
            )
            teacher_parts.append(f"\n\n## REPAIR_JSONL: {path}\n\n{repeated_records}")
    teacher = "\n".join(part for part in teacher_parts if part.strip())

    limit = int(args.max_text_chars)
    if limit > 0:
        stage_a = cap_text(tiny, limit)
        stage_b = capped_join((tiny, textbook, surface_answers, external), limit=limit)
        stage_c = capped_join((stage_b, teacher), limit=limit)
    else:
        stage_a = tiny
        stage_b = "\n".join(
            part for part in (tiny, textbook, surface_answers, external) if part.strip()
        )
        stage_c = "\n".join(part for part in (stage_b, teacher) if part.strip())
    return {"tiny": stage_a, "edu": stage_b, "teacher": stage_c}


def build_tokenizer(args: argparse.Namespace, *, all_text: str):
    if bool(getattr(args, "train_byte_bpe_tokenizer", False)):
        extra_texts = (
            str(getattr(args, "seed_text", "")),
            str(getattr(args, "repair_seed_texts", "")),
            str(getattr(args, "repair_seed_expectations", "")),
        )
        return ByteBPETokenizerAdapter.from_text(
            all_text,
            vocab_size=int(getattr(args, "byte_bpe_vocab_size", 8192)),
            min_frequency=int(getattr(args, "byte_bpe_min_frequency", 2)),
            eos_token=str(getattr(args, "byte_bpe_eos_token", "<|qtrm_eos|>")),
            unk_token=str(getattr(args, "byte_bpe_unk_token", "<|qtrm_unk|>")),
            extra_texts=extra_texts,
        )
    if args.tokenizer_name:
        if bool(getattr(args, "compact_hf_vocab", False)):
            extra_texts = (
                str(getattr(args, "seed_text", "")),
                str(getattr(args, "repair_seed_texts", "")),
                str(getattr(args, "repair_seed_expectations", "")),
            )
            return CompactHFTokenizerAdapter.from_text(
                str(args.tokenizer_name),
                all_text,
                max_size=int(getattr(args, "compact_vocab_max_size", 0)),
                min_count=int(getattr(args, "compact_vocab_min_count", 1)),
                extra_texts=extra_texts,
            )
        return HFTokenizerAdapter.from_name(str(args.tokenizer_name))
    return _text_probe.CharTokenizer.from_text(all_text)


def tokenizer_from_payload(tokenizer_payload: dict[str, object], args: argparse.Namespace):
    kind = str(tokenizer_payload.get("kind", ""))
    if kind == "byte_bpe":
        args.train_byte_bpe_tokenizer = True
        args.tokenizer_name = ""
        args.byte_bpe_eos_token = str(tokenizer_payload.get("eos_token", args.byte_bpe_eos_token))
        args.byte_bpe_unk_token = str(tokenizer_payload.get("unk_token", args.byte_bpe_unk_token))
        return ByteBPETokenizerAdapter.from_payload(tokenizer_payload)
    if kind == "hf_compact":
        args.tokenizer_name = str(tokenizer_payload.get("name") or args.tokenizer_name)
        args.compact_hf_vocab = True
        return CompactHFTokenizerAdapter.from_payload(tokenizer_payload)
    if kind == "hf" or args.tokenizer_name:
        name = str(tokenizer_payload.get("name") or args.tokenizer_name)
        args.tokenizer_name = name
        return HFTokenizerAdapter.from_name(name)
    chars = tuple(str(ch) for ch in tokenizer_payload.get("chars", ()))
    return _text_probe.CharTokenizer(chars=chars, char_to_id={ch: i for i, ch in enumerate(chars)})


def _dtype_from_name(name: str):
    value = str(name).lower()
    if value in {"auto", ""}:
        return "auto"
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unknown pretrained init dtype: {name}")


def compact_or_full_hf_token_ids(tokenizer) -> list[int]:
    if isinstance(tokenizer, CompactHFTokenizerAdapter):
        return [int(token_id) for token_id in tokenizer.compact_to_hf]
    if isinstance(tokenizer, HFTokenizerAdapter):
        return list(range(int(tokenizer.vocab_size)))
    raise ValueError(
        "pretrained LM initialization requires --tokenizer-name, preferably "
        "with --compact-hf-vocab for a bounded active vocabulary."
    )


def project_pretrained_rows(
    rows: torch.Tensor,
    *,
    target_dim: int,
    seed: int,
    mode: str,
) -> torch.Tensor:
    """Adapt pretrained embedding rows to the native model hidden size."""
    source = rows.detach().to(dtype=torch.float32, device="cpu")
    if source.ndim != 2:
        raise ValueError(f"expected 2D pretrained rows, got shape {tuple(source.shape)}")
    target_dim = int(target_dim)
    source_dim = int(source.shape[1])
    if source_dim == target_dim:
        return source.contiguous()
    if source_dim > target_dim:
        if str(mode) == "slice":
            return source[:, :target_dim].contiguous()
        if str(mode) != "random_projection":
            raise ValueError(f"unknown pretrained projection mode: {mode}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        projection = torch.randn(
            source_dim,
            target_dim,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(float(source_dim))
        return torch.matmul(source, projection).contiguous()
    padded = torch.zeros(source.shape[0], target_dim, dtype=torch.float32)
    padded[:, :source_dim] = source
    return padded


def _resolve_hf_embedding_roots(model) -> tuple[Any, Any]:
    input_embeddings = None
    output_embeddings = None
    roots = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    for root in roots:
        if root is None:
            continue
        if input_embeddings is None and hasattr(root, "get_input_embeddings"):
            input_embeddings = root.get_input_embeddings()
        if output_embeddings is None and hasattr(root, "get_output_embeddings"):
            output_embeddings = root.get_output_embeddings()
        if input_embeddings is not None and output_embeddings is not None:
            break
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise ValueError("could not locate pretrained input embedding weight")
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        output_embeddings = input_embeddings
    return input_embeddings, output_embeddings


def _load_pretrained_lm_for_init(model_id: str, *, dtype_name: str, device: str):
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "transformers is required for --pretrained-init-model"
        ) from exc
    dtype = _dtype_from_name(dtype_name)
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if str(device):
        kwargs["device_map"] = str(device)
    errors: list[str] = []
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModel):
        try:
            return cls.from_pretrained(str(model_id), **kwargs)
        except Exception as exc:  # pragma: no cover - model dependent
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(
        f"failed to load pretrained init model {model_id!r}: " + " | ".join(errors)
    )


def initialize_native_lm_from_pretrained(
    model,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, object]:
    model_id = str(getattr(args, "pretrained_init_model", "") or "")
    if not model_id:
        return {"enabled": False}
    hf_token_ids = compact_or_full_hf_token_ids(tokenizer)
    if (
        isinstance(tokenizer, HFTokenizerAdapter)
        and not bool(getattr(args, "pretrained_init_allow_full_vocab", False))
    ):
        raise ValueError(
            "--pretrained-init-model with full HF vocab is disabled by default; "
            "use --compact-hf-vocab or pass --pretrained-init-allow-full-vocab."
        )
    pretrained = _load_pretrained_lm_for_init(
        model_id,
        dtype_name=str(getattr(args, "pretrained_init_dtype", "float16")),
        device=str(getattr(args, "pretrained_init_device_map", "cpu")),
    )
    try:
        input_embeddings, output_embeddings = _resolve_hf_embedding_roots(pretrained)
        input_weight = input_embeddings.weight.detach().to(device="cpu")
        output_weight = output_embeddings.weight.detach().to(device="cpu")
        max_id = max(hf_token_ids) if hf_token_ids else -1
        if max_id >= int(input_weight.shape[0]) or max_id >= int(output_weight.shape[0]):
            raise ValueError(
                "compact HF token id exceeds pretrained embedding rows: "
                f"max_id={max_id}, input_rows={input_weight.shape[0]}, "
                f"output_rows={output_weight.shape[0]}"
            )
        index = torch.tensor(hf_token_ids, dtype=torch.long)
        target_dim = int(model.token_embed.weight.shape[1])
        projected_input = project_pretrained_rows(
            input_weight.index_select(0, index),
            target_dim=target_dim,
            seed=int(getattr(args, "pretrained_init_seed", 336)),
            mode=str(getattr(args, "pretrained_init_projection", "random_projection")),
        )
        projected_output = project_pretrained_rows(
            output_weight.index_select(0, index),
            target_dim=target_dim,
            seed=int(getattr(args, "pretrained_init_seed", 336)) + 1,
            mode=str(getattr(args, "pretrained_init_projection", "random_projection")),
        )
        scale = float(getattr(args, "pretrained_init_scale", 1.0))
        with torch.no_grad():
            model.token_embed.weight.copy_(
                (projected_input * scale).to(
                    dtype=model.token_embed.weight.dtype,
                    device=model.token_embed.weight.device,
                )
            )
            if not bool(getattr(model, "tie_embeddings", False)):
                model.lm_head.weight.copy_(
                    (projected_output * scale).to(
                        dtype=model.lm_head.weight.dtype,
                        device=model.lm_head.weight.device,
                    )
                )
        return {
            "enabled": True,
            "model_id": model_id,
            "tokenizer_kind": type(tokenizer).__name__,
            "rows": len(hf_token_ids),
            "source_dim": int(input_weight.shape[1]),
            "target_dim": target_dim,
            "projection": str(getattr(args, "pretrained_init_projection", "random_projection")),
            "scale": scale,
            "runtime_donor": False,
        }
    finally:
        del pretrained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


INIT_CHECKPOINT_MODEL_ARGS = (
    "seq_len",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "dropout",
    "backbone",
    "encode_backbone",
    "think_backbone",
    "decode_backbone",
    "think_structure",
    "trm_l_cycles",
    "trm_full_grad_cycles",
    "hybrid_layers",
    "attn_every",
    "delta_backend",
    "delta_head_dim",
    "delta_num_v_heads",
    "delta_expand_v",
    "delta_mode",
    "delta_no_short_conv",
    "delta_conv_size",
    "delta_norm_eps",
    "attention_backend",
    "strict_backends",
    "tie_embeddings",
)


def load_init_checkpoint(args: argparse.Namespace) -> dict[str, object] | None:
    path = str(getattr(args, "init_checkpoint", "") or "")
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    for name in INIT_CHECKPOINT_MODEL_ARGS:
        if name in checkpoint_args and hasattr(args, name):
            setattr(args, name, checkpoint_args[name])
    tokenizer_payload = checkpoint.get("tokenizer", {})
    if isinstance(tokenizer_payload, dict):
        kind = str(tokenizer_payload.get("kind", ""))
        if kind == "byte_bpe":
            args.train_byte_bpe_tokenizer = True
            args.tokenizer_name = ""
            args.byte_bpe_eos_token = str(tokenizer_payload.get("eos_token", args.byte_bpe_eos_token))
            args.byte_bpe_unk_token = str(tokenizer_payload.get("unk_token", args.byte_bpe_unk_token))
        elif kind == "hf_compact":
            args.compact_hf_vocab = True
            args.tokenizer_name = str(tokenizer_payload.get("name") or args.tokenizer_name)
        elif kind == "hf":
            args.tokenizer_name = str(tokenizer_payload.get("name") or args.tokenizer_name)
    return checkpoint


def tokenizer_report_payload(tokenizer_payload: dict[str, object]) -> dict[str, object]:
    if str(tokenizer_payload.get("kind", "")) == "hf_compact":
        return {
            "kind": "hf_compact",
            "name": str(tokenizer_payload.get("name", "")),
            "compact_vocab_size": len(tokenizer_payload.get("compact_to_hf", ())),
            "unk_compact_id": int(tokenizer_payload.get("unk_compact_id", 0)),
            "eos_compact_id": tokenizer_payload.get("eos_compact_id"),
        }
    if str(tokenizer_payload.get("kind", "")) == "byte_bpe":
        return {
            "kind": "byte_bpe",
            "name": str(tokenizer_payload.get("name", "")),
            "vocab_size": int(tokenizer_payload.get("vocab_size", 0)),
            "eos_token": str(tokenizer_payload.get("eos_token", "")),
            "unk_token": str(tokenizer_payload.get("unk_token", "")),
            "eos_id": tokenizer_payload.get("eos_id"),
        }
    if str(tokenizer_payload.get("kind", "")) == "char":
        chars = tokenizer_payload.get("chars", ())
        if isinstance(chars, (list, tuple)) and len(chars) > 128:
            return {
                "kind": "char",
                "name": str(tokenizer_payload.get("name", "")),
                "vocab_size": len(chars),
                "chars_preview": list(chars[:128]),
            }
    return tokenizer_payload


def windows_for_text(tokenizer, text: str, *, seq_len: int) -> list[tuple[list[int], list[int]]]:
    tokens = tokenizer.encode(text)
    return _text_probe.make_windows(tokens, seq_len=int(seq_len))


def split_eval_windows(
    tokenizer,
    text: str,
    *,
    seq_len: int,
    eval_fraction: float,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    tokens = tokenizer.encode(text)
    if len(tokens) <= int(seq_len) + 2:
        raise ValueError("bootstrap text is too short for seq_len")
    split = max(int(seq_len) + 2, int((1.0 - float(eval_fraction)) * len(tokens)))
    train = _text_probe.make_windows(tokens[:split], seq_len=int(seq_len))
    eval_ = _text_probe.make_windows(tokens[split - int(seq_len) :], seq_len=int(seq_len))
    if not train or not eval_:
        raise ValueError("not enough train/eval windows")
    return train, eval_


def train_stage(
    model,
    args: argparse.Namespace,
    *,
    stage_name: str,
    text: str,
    tokenizer,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    windows = windows_for_text(tokenizer, text, seq_len=int(args.seq_len))
    if not windows:
        raise ValueError(f"stage {stage_name} has no windows")
    setattr(args, "_target_eos_token_id", getattr(tokenizer, "eos_token_id", None))
    last_loss = _text_probe.train_language_model(
        model,
        windows,
        args,
        device=device,
        steps=int(steps),
        think_steps=int(args.train_think_steps),
        log_prefix=stage_name,
    )
    return {
        "stage": stage_name,
        "steps": int(steps),
        "windows": len(windows),
        "last_loss": last_loss,
        "optimizer": getattr(args, "_last_optimizer_report", {}),
    }


def train_tst_language_model(
    model,
    train_windows: list[tuple[list[int], list[int]]],
    args: argparse.Namespace,
    *,
    device: torch.device,
    steps: int,
    think_steps: int,
    bag_size: int,
    log_prefix: str = "",
) -> float:
    if not hasattr(model, "forward_embeddings"):
        raise ValueError("model does not support embedding-level TST forward")
    optimizer, optimizer_report = _text_probe.build_memory_efficient_optimizer(
        model,
        optimizer_name=str(getattr(args, "optimizer", "adamw")),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        galore_rank=int(getattr(args, "galore_rank", 128)),
        galore_update_proj_gap=int(getattr(args, "galore_update_proj_gap", 200)),
        galore_scale=float(getattr(args, "galore_scale", 0.25)),
        galore_proj_type=str(getattr(args, "galore_proj_type", "std")),
        galore_min_dim=int(getattr(args, "galore_min_dim", 128)),
        galore_include_embeddings=bool(getattr(args, "galore_include_embeddings", False)),
    )
    setattr(args, "_last_optimizer_report", optimizer_report)
    last_loss = 0.0
    prefix = f"{log_prefix} " if log_prefix else ""
    for step in range(1, int(steps) + 1):
        model.train()
        x, y = _text_probe.batch_windows(
            train_windows,
            batch_size=int(args.batch_size),
            device=device,
        )
        token_stream = torch.cat([x[:, :1], y], dim=1)
        input_bags, target_bags = next_token_bags(token_stream, bag_size=int(bag_size))
        embeddings = superpose_embeddings(model.token_embed.weight, input_bags)
        logits = model.forward_embeddings(embeddings, think_steps=int(think_steps))
        loss = multi_hot_cross_entropy(logits, target_bags)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if int(args.log_every) > 0 and (step == 1 or step % int(args.log_every) == 0):
            print(
                json.dumps(
                    {"step": step, "loss": last_loss, "model": f"{prefix}tst".strip()},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return last_loss


def train_tst_stage(
    model,
    args: argparse.Namespace,
    *,
    stage_name: str,
    text: str,
    tokenizer,
    device: torch.device,
    steps: int,
    bag_size: int,
) -> dict[str, object]:
    windows = windows_for_text(tokenizer, text, seq_len=int(args.seq_len))
    if not windows:
        raise ValueError(f"stage {stage_name} has no windows")
    last_loss = train_tst_language_model(
        model,
        windows,
        args,
        device=device,
        steps=int(steps),
        think_steps=int(args.train_think_steps),
        bag_size=int(bag_size),
        log_prefix=stage_name,
    )
    return {
        "stage": stage_name,
        "objective": "token_superposition_mce",
        "steps": int(steps),
        "bag_size": int(bag_size),
        "windows": len(windows),
        "last_loss": last_loss,
        "optimizer": getattr(args, "_last_optimizer_report", {}),
    }


@torch.no_grad()
def write_on_policy_candidates(
    model,
    tokenizer,
    args: argparse.Namespace,
    *,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, object]]:
    seeds = [
        seed.strip()
        for seed in str(args.repair_seed_texts).split("||")
        if seed.strip()
    ]
    if not seeds:
        seeds = [str(args.seed_text)]
    rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds[: int(args.repair_prompt_count)]):
        sample = _text_probe.generate_text(
            model,
            tokenizer,
            seed_text=seed,
            seq_len=int(args.seq_len),
            think_steps=int(args.eval_think_steps),
            max_new_chars=int(args.max_new_chars),
            device=device,
        )
        rows.append(
            {
                "index": index,
                "seed_text": seed,
                "sample": sample,
                "degeneracy": _text_probe.sample_degeneracy(sample),
                "teacher_instruction": (
                    "Rewrite the continuation as concise, grammatical, "
                    "non-repetitive training text for QTRM-native."
                ),
            }
        )
    if rows:
        path = out_dir / "on_policy_candidates.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return rows


def _normalized_nonempty_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if line:
            lines.append(line)
    return lines


def line_loop_metrics(text: str, *, block_size: int = 2) -> dict[str, float]:
    """Measure repeated line/block replay in generated surface text."""
    lines = _normalized_nonempty_lines(text)
    line_count = len(lines)
    if line_count == 0:
        return {
            "line_count": 0.0,
            "unique_line_fraction": 0.0,
            "max_line_repeat_fraction": 1.0,
            "max_block_repeat_fraction": 1.0,
        }
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    max_line_repeat = max(counts.values()) / max(1, line_count)

    blocks: list[tuple[str, ...]] = []
    width = max(1, int(block_size))
    if line_count >= width:
        blocks = [tuple(lines[index : index + width]) for index in range(line_count - width + 1)]
    block_fraction = 0.0
    if blocks:
        block_counts: dict[tuple[str, ...], int] = {}
        for block in blocks:
            block_counts[block] = block_counts.get(block, 0) + 1
        block_fraction = max(block_counts.values()) / max(1, len(blocks))

    return {
        "line_count": float(line_count),
        "unique_line_fraction": float(len(set(lines)) / max(1, line_count)),
        "max_line_repeat_fraction": float(max_line_repeat),
        "max_block_repeat_fraction": float(block_fraction),
    }


def on_policy_loop_reject_reasons(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> list[str]:
    if not rows:
        return []
    min_lines = int(args.min_on_policy_loop_check_lines)
    min_unique_fraction = float(args.min_on_policy_unique_line_fraction)
    max_block_fraction = float(args.max_on_policy_repeated_block_fraction)
    max_line_fraction = float(args.max_on_policy_repeated_line_fraction)
    reasons: list[str] = []
    for row in rows:
        metrics = line_loop_metrics(str(row.get("sample", "")))
        row["line_loop_metrics"] = metrics
        if int(metrics["line_count"]) < min_lines:
            continue
        if (
            min_unique_fraction > 0.0
            and metrics["unique_line_fraction"] < min_unique_fraction
        ):
            reasons.append("on_policy_unique_line_fraction_too_low")
        if (
            max_block_fraction > 0.0
            and metrics["max_block_repeat_fraction"] > max_block_fraction
        ):
            reasons.append("on_policy_repeated_block_loop")
        if (
            max_line_fraction > 0.0
            and metrics["max_line_repeat_fraction"] > max_line_fraction
        ):
            reasons.append("on_policy_repeated_line_loop")
    return sorted(set(reasons))


def answer_surface_metrics(seed_text: str, sample: str) -> dict[str, object]:
    continuation = str(sample)
    if str(sample).startswith(str(seed_text)):
        continuation = str(sample)[len(str(seed_text)) :]
    stripped = continuation.strip()
    nonspace_chars = [char for char in stripped if not char.isspace()]
    informative_chars = [char for char in nonspace_chars if char.isalnum()]
    informative_fraction = (
        float(len(informative_chars) / len(nonspace_chars)) if nonspace_chars else 0.0
    )
    words = re.findall(r"[\w가-힣]+", stripped.lower(), flags=re.UNICODE)
    word_counts = Counter(words)
    max_word_repeat_fraction = (
        float(max(word_counts.values()) / len(words)) if words else 0.0
    )
    return {
        "continuation_chars": float(len(stripped)),
        "informative_char_fraction": informative_fraction,
        "word_count": float(len(words)),
        "max_word_repeat_fraction": max_word_repeat_fraction,
        "contains_control_char": any(
            ord(char) < 32 and char not in "\n\r\t" for char in continuation
        ),
        "contains_next_user": "\nUser:" in continuation or stripped.startswith("User:"),
        "contains_extra_assistant": "\nAssistant:" in continuation,
        "contains_think": "<think" in stripped.lower() or "</think" in stripped.lower(),
    }


def parse_seed_expectations(value: str) -> dict[str, list[str]]:
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--repair-seed-expectations must be JSON object text") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--repair-seed-expectations must decode to an object")
    expectations: dict[str, list[str]] = {}
    for key, keywords in parsed.items():
        if not isinstance(key, str):
            raise ValueError("repair seed expectation keys must be strings")
        if isinstance(keywords, str):
            values = [keywords]
        elif isinstance(keywords, list) and all(isinstance(item, str) for item in keywords):
            values = list(keywords)
        else:
            raise ValueError("repair seed expectation values must be strings or string lists")
        cleaned = [keyword.strip() for keyword in values if keyword.strip()]
        if key.strip() and cleaned:
            expectations[key.strip()] = cleaned
    return expectations


def semantic_relevance_metrics(
    seed_text: str,
    sample: str,
    expectations: dict[str, list[str]],
) -> dict[str, object]:
    continuation = str(sample)
    if str(sample).startswith(str(seed_text)):
        continuation = str(sample)[len(str(seed_text)) :]
    matched_key = ""
    expected: list[str] = []
    for key, keywords in expectations.items():
        if key in str(seed_text):
            matched_key = key
            expected = keywords
            break
    answer = continuation.lower()
    matched: list[str] = []
    matched_groups: list[list[str]] = []
    for keyword in expected:
        alternatives = [part.strip() for part in keyword.split("|") if part.strip()]
        if not alternatives:
            continue
        group_matched = [
            alternative
            for alternative in alternatives
            if alternative.lower() in answer
        ]
        if group_matched:
            matched.append(keyword)
            matched_groups.append(group_matched)
    return {
        "expectation_key": matched_key,
        "expected_keywords": expected,
        "matched_keywords": matched,
        "matched_keyword_groups": matched_groups,
        "matched_count": float(len(matched)),
    }


def on_policy_answer_reject_reasons(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> list[str]:
    min_chars = int(args.min_on_policy_continuation_chars)
    min_keyword_hits = int(args.min_on_policy_keyword_hits)
    min_informative_fraction = float(args.min_on_policy_informative_char_fraction)
    max_word_repeat_fraction = float(args.max_on_policy_repeated_word_fraction)
    expectations = parse_seed_expectations(str(args.repair_seed_expectations))
    reasons: list[str] = []
    for row in rows:
        metrics = answer_surface_metrics(
            str(row.get("seed_text", "")),
            str(row.get("sample", "")),
        )
        row["answer_surface_metrics"] = metrics
        relevance = semantic_relevance_metrics(
            str(row.get("seed_text", "")),
            str(row.get("sample", "")),
            expectations,
        )
        row["semantic_relevance_metrics"] = relevance
        seed = str(row.get("seed_text", ""))
        is_answer_prompt = "Assistant:" in seed
        if is_answer_prompt and float(metrics["continuation_chars"]) < float(min_chars):
            reasons.append("on_policy_answer_too_short")
        if (
            is_answer_prompt
            and min_informative_fraction > 0.0
            and float(metrics["informative_char_fraction"]) < min_informative_fraction
        ):
            reasons.append("on_policy_informative_char_fraction_too_low")
        if (
            is_answer_prompt
            and max_word_repeat_fraction > 0.0
            and float(metrics["word_count"]) >= 6.0
            and float(metrics["max_word_repeat_fraction"]) > max_word_repeat_fraction
        ):
            reasons.append("on_policy_repeated_word_loop")
        if (
            is_answer_prompt
            and relevance["expectation_key"]
            and float(relevance["matched_count"]) < float(min_keyword_hits)
        ):
            reasons.append("on_policy_semantic_relevance_too_low")
        if bool(metrics["contains_next_user"]):
            reasons.append("on_policy_cross_record_continuation")
        if bool(metrics["contains_control_char"]):
            reasons.append("on_policy_control_char_leak")
        if bool(metrics["contains_extra_assistant"]):
            reasons.append("on_policy_extra_assistant_marker")
        if bool(metrics["contains_think"]):
            reasons.append("on_policy_visible_think_leak")
    return sorted(set(reasons))


def train_bootstrap(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    init_checkpoint = load_init_checkpoint(args)
    stage_texts = build_stage_texts(args)
    all_text = "\n".join(stage_texts.values())
    if init_checkpoint is not None:
        tokenizer = tokenizer_from_payload(
            init_checkpoint.get("tokenizer", {}),
            args,
        )
    else:
        tokenizer = build_tokenizer(args, all_text=all_text)
    train_windows, eval_windows = split_eval_windows(
        tokenizer,
        stage_texts["teacher"],
        seq_len=int(args.seq_len),
        eval_fraction=float(args.eval_fraction),
    )
    model = _text_probe.build_model(args, vocab_size=tokenizer.vocab_size).to(device)
    if init_checkpoint is not None:
        if str(getattr(args, "pretrained_init_model", "") or ""):
            raise ValueError("--init-checkpoint and --pretrained-init-model cannot be combined")
        model.load_state_dict(init_checkpoint["model_state"])
        pretrained_init_report = {"enabled": False, "skipped": "init_checkpoint_loaded"}
    else:
        pretrained_init_report = initialize_native_lm_from_pretrained(model, tokenizer, args)

    stage_reports: list[dict[str, object]] = []
    stage_specs = (
        ("tiny", int(args.stage_a_steps)),
        ("edu", int(args.stage_b_steps)),
    )
    for stage_name, steps in stage_specs:
        if steps <= 0:
            continue
        stage_reports.append(
            train_stage(
                model,
                args,
                stage_name=stage_name,
                text=stage_texts[stage_name],
                tokenizer=tokenizer,
                device=device,
                steps=steps,
            )
        )
    if int(args.tst_phase_steps) > 0:
        stage_reports.append(
            train_tst_stage(
                model,
                args,
                stage_name="teacher_tst",
                text=stage_texts["teacher"],
                tokenizer=tokenizer,
                device=device,
                steps=int(args.tst_phase_steps),
                bag_size=int(args.tst_bag_size),
            )
        )
    if int(args.stage_c_steps) > 0:
        stage_reports.append(
            train_stage(
                model,
                args,
                stage_name="teacher",
                text=stage_texts["teacher"],
                tokenizer=tokenizer,
                device=device,
                steps=int(args.stage_c_steps),
            )
        )

    full_loss = _text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=int(args.eval_think_steps),
    )
    think0_loss = _text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=0,
    )
    off_loss = _text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(args.batch_size),
        device=device,
        think_steps=int(args.eval_think_steps),
        thinking_block_off=True,
    )
    depth_sweep_losses: dict[str, float] = {}
    for depth in _text_probe.parse_depth_sweep(str(args.eval_depth_sweep)):
        if depth == int(args.eval_think_steps):
            depth_sweep_losses[str(depth)] = float(full_loss)
        elif depth == 0:
            depth_sweep_losses[str(depth)] = float(think0_loss)
        else:
            depth_sweep_losses[str(depth)] = _text_probe.eval_loss(
                model,
                eval_windows,
                batch_size=int(args.batch_size),
                device=device,
                think_steps=int(depth),
            )
    shallow_losses = [
        loss
        for depth, loss in depth_sweep_losses.items()
        if int(depth) < int(args.eval_think_steps)
    ]
    best_shallow_loss = min(shallow_losses) if shallow_losses else None
    full_vs_best_shallow = _text_probe._safe_ratio(full_loss, best_shallow_loss)
    sample = _text_probe.generate_text(
        model,
        tokenizer,
        seed_text=str(args.seed_text),
        seq_len=int(args.seq_len),
        think_steps=int(args.eval_think_steps),
        max_new_chars=int(args.max_new_chars),
        device=device,
    )
    degeneracy = _text_probe.sample_degeneracy(sample)
    seed_sample_surface = answer_surface_metrics(str(args.seed_text), sample)
    random_loss = torch.log(torch.tensor(float(max(2, tokenizer.vocab_size)))).item()
    reject_reasons = _text_probe.language_reject_reasons(
        args,
        full_loss=full_loss,
        think0_loss=think0_loss,
        off_loss=off_loss,
        baseline_loss=None,
        random_loss=random_loss,
        degeneracy=degeneracy,
    )
    if (
        float(args.max_full_vs_best_shallow_loss_ratio) > 0.0
        and full_vs_best_shallow is not None
        and full_vs_best_shallow > float(args.max_full_vs_best_shallow_loss_ratio)
    ):
        reject_reasons.append("full_loss_regressed_vs_best_shallow_depth")
    if "Assistant:" in str(args.seed_text):
        if bool(seed_sample_surface["contains_next_user"]):
            reject_reasons.append("seed_sample_cross_record_continuation")
        if bool(seed_sample_surface["contains_control_char"]):
            reject_reasons.append("seed_sample_control_char_leak")
        if bool(seed_sample_surface["contains_extra_assistant"]):
            reject_reasons.append("seed_sample_extra_assistant_marker")
        if bool(seed_sample_surface["contains_think"]):
            reject_reasons.append("seed_sample_visible_think_leak")
        if float(seed_sample_surface["continuation_chars"]) < float(args.min_on_policy_continuation_chars):
            reject_reasons.append("seed_sample_answer_too_short")
        if (
            float(args.min_on_policy_informative_char_fraction) > 0.0
            and float(seed_sample_surface["informative_char_fraction"])
            < float(args.min_on_policy_informative_char_fraction)
        ):
            reject_reasons.append("seed_sample_informative_char_fraction_too_low")
        if (
            float(args.max_on_policy_repeated_word_fraction) > 0.0
            and float(seed_sample_surface["word_count"]) >= 6.0
            and float(seed_sample_surface["max_word_repeat_fraction"])
            > float(args.max_on_policy_repeated_word_fraction)
        ):
            reject_reasons.append("seed_sample_repeated_word_loop")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    on_policy_candidates = write_on_policy_candidates(
        model,
        tokenizer,
        args,
        device=device,
        out_dir=out_dir,
    )
    reject_reasons.extend(on_policy_loop_reject_reasons(args, on_policy_candidates))
    reject_reasons.extend(on_policy_answer_reject_reasons(args, on_policy_candidates))
    if on_policy_candidates:
        (out_dir / "on_policy_candidates.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in on_policy_candidates
            ),
            encoding="utf-8",
        )
    if isinstance(tokenizer, ByteBPETokenizerAdapter):
        tokenizer_payload = {
            "kind": "byte_bpe",
            "name": tokenizer.name,
            "vocab_size": tokenizer.vocab_size,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
            "eos_id": tokenizer.eos_id,
            "tokenizer_json": tokenizer.tokenizer.to_str(),
        }
    elif isinstance(tokenizer, CompactHFTokenizerAdapter):
        tokenizer_payload: dict[str, object] = {
            "kind": "hf_compact",
            "name": str(args.tokenizer_name),
            "compact_to_hf": list(tokenizer.compact_to_hf),
            "unk_compact_id": int(tokenizer.unk_compact_id),
            "eos_compact_id": tokenizer.eos_compact_id,
        }
    else:
        tokenizer_payload = {
            "kind": "hf" if args.tokenizer_name else "char",
            "name": str(args.tokenizer_name),
        }
    if tokenizer_payload.get("kind") == "char":
        tokenizer_payload["chars"] = tokenizer.chars

    report: dict[str, object] = {
        "status": "complete",
        "target_level": str(args.target_level),
        "decision": str(args.accepted_decision) if not reject_reasons else "rejected",
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "train": vars(args),
        "init_checkpoint": str(getattr(args, "init_checkpoint", "") or ""),
        "pretrained_init": pretrained_init_report,
        "tokenizer": tokenizer_report_payload(tokenizer_payload),
        "vocab_size": tokenizer.vocab_size,
        "stage_reports": stage_reports,
        "corpus": {
            "tiny_chars": len(stage_texts["tiny"]),
            "edu_chars": len(stage_texts["edu"]),
            "teacher_chars": len(stage_texts["teacher"]),
            "train_windows": len(train_windows),
            "eval_windows": len(eval_windows),
        },
        "backend_summary": _text_probe.summarize_backend(model),
        "random_loss": random_loss,
        "eval_metrics": {
            "think_eval_loss": full_loss,
            "think0_loss": think0_loss,
            "thinking_block_off_loss": off_loss,
            "loss_ratios": {
                "full_vs_think0": _text_probe._safe_ratio(full_loss, think0_loss),
                "full_vs_thinking_block_off": _text_probe._safe_ratio(full_loss, off_loss),
                "full_vs_best_shallow_depth": full_vs_best_shallow,
            },
            "depth_sweep_loss": depth_sweep_losses,
            "best_shallow_depth_loss": best_shallow_loss,
            "sample_degeneracy": degeneracy,
            "seed_sample_answer_surface_metrics": seed_sample_surface,
            "sample": sample,
        },
        "on_policy_candidates": {
            "count": len(on_policy_candidates),
            "path": str(out_dir / "on_policy_candidates.jsonl") if on_policy_candidates else "",
            "line_loop_metrics": [
                row.get("line_loop_metrics", {}) for row in on_policy_candidates
            ],
            "answer_surface_metrics": [
                row.get("answer_surface_metrics", {}) for row in on_policy_candidates
            ],
            "semantic_relevance_metrics": [
                row.get("semantic_relevance_metrics", {}) for row in on_policy_candidates
            ],
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "report": report,
            "tokenizer": tokenizer_payload,
        },
        out_dir / "last.pt",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _text_probe.build_arg_parser()
    parser.description = "Train QTRM-native with a language-first bootstrap curriculum."
    parser.set_defaults(
        out_dir="local_eval/qtrm_native_language_bootstrap",
        target_level="Language-first QTRM-native bootstrap",
        accepted_decision="accepted_qtrm_native_language_bootstrap",
        max_text_chars=120000,
        seq_len=96,
        steps=0,
        d_model=96,
        n_heads=4,
        n_kv_heads=2,
        d_ff=192,
        batch_size=64,
        lr=3e-4,
        train_think_steps=4,
        eval_think_steps=4,
        eval_depth_sweep="0,1,2,4",
        seed_text="QTRM native language: ",
        max_new_chars=180,
        max_random_loss_fraction=0.85,
        min_unique_chars=12.0,
        max_run_fraction=0.30,
        max_full_vs_think0_loss_ratio=1.35,
        max_full_vs_off_loss_ratio=1.35,
        max_full_vs_best_shallow_loss_ratio=0.0,
        log_every=100,
    )
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--compact-hf-vocab", action="store_true")
    parser.add_argument("--compact-vocab-max-size", type=int, default=0)
    parser.add_argument("--compact-vocab-min-count", type=int, default=1)
    parser.add_argument("--train-byte-bpe-tokenizer", action="store_true")
    parser.add_argument("--byte-bpe-vocab-size", type=int, default=8192)
    parser.add_argument("--byte-bpe-min-frequency", type=int, default=2)
    parser.add_argument("--byte-bpe-eos-token", default="<|qtrm_eos|>")
    parser.add_argument("--byte-bpe-unk-token", default="<|qtrm_unk|>")
    parser.add_argument("--tiny-repeats", type=int, default=256)
    parser.add_argument("--textbook-repeats", type=int, default=192)
    parser.add_argument("--surface-answer-repeats", type=int, default=96)
    parser.add_argument("--diverse-surface-answer-count", type=int, default=42)
    parser.add_argument("--record-separator", default="auto")
    parser.add_argument("--shuffle-corpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-text-file", action="append", default=[])
    parser.add_argument("--teacher-jsonl", action="append", default=[])
    parser.add_argument("--repair-jsonl", action="append", default=[])
    parser.add_argument("--repair-jsonl-repeats", type=int, default=1)
    parser.add_argument("--gate-anchor-repeats", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Load model weights/tokenizer from a prior language bootstrap checkpoint before training.",
    )
    parser.add_argument(
        "--pretrained-init-model",
        default="",
        help="Initialize native token embeddings/LM head from a pretrained HF LM, then run donorless.",
    )
    parser.add_argument(
        "--pretrained-init-projection",
        choices=("random_projection", "slice"),
        default="random_projection",
    )
    parser.add_argument("--pretrained-init-seed", type=int, default=336)
    parser.add_argument("--pretrained-init-scale", type=float, default=1.0)
    parser.add_argument("--pretrained-init-dtype", default="float16")
    parser.add_argument("--pretrained-init-device-map", default="cpu")
    parser.add_argument("--pretrained-init-allow-full-vocab", action="store_true")
    parser.add_argument("--stage-a-steps", type=int, default=400)
    parser.add_argument("--stage-b-steps", type=int, default=800)
    parser.add_argument("--stage-c-steps", type=int, default=0)
    parser.add_argument("--tst-phase-steps", type=int, default=0)
    parser.add_argument("--tst-bag-size", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--repair-prompt-count", type=int, default=4)
    parser.add_argument("--min-on-policy-loop-check-lines", type=int, default=4)
    parser.add_argument("--min-on-policy-continuation-chars", type=int, default=16)
    parser.add_argument("--min-on-policy-keyword-hits", type=int, default=2)
    parser.add_argument("--min-on-policy-informative-char-fraction", type=float, default=0.25)
    parser.add_argument("--min-on-policy-unique-line-fraction", type=float, default=0.55)
    parser.add_argument("--max-on-policy-repeated-block-fraction", type=float, default=0.24)
    parser.add_argument("--max-on-policy-repeated-line-fraction", type=float, default=0.30)
    parser.add_argument("--max-on-policy-repeated-word-fraction", type=float, default=0.35)
    parser.add_argument(
        "--repair-seed-texts",
        default=DEFAULT_REPAIR_SEED_TEXTS,
    )
    parser.add_argument(
        "--repair-seed-expectations",
        default=json.dumps(DEFAULT_REPAIR_SEED_EXPECTATIONS, ensure_ascii=False),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_bootstrap(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
