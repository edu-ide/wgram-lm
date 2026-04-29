from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def _case_id(base: str, idx: int, avoid_ids: set[str]) -> str:
    case_id = f"{base}-{idx:04d}"
    while case_id in avoid_ids:
        idx += 10_000
        case_id = f"{base}-{idx:04d}"
    avoid_ids.add(case_id)
    return case_id


def _pick(seq: list[str], idx: int) -> str:
    return seq[idx % len(seq)]


def build_synthetic_memory_reasoning_cases(
    *,
    num_sets: int = 6,
    seed: int = 13,
    avoid_ids: Iterable[str] = (),
) -> list[dict]:
    rng = random.Random(seed)
    blocked = set(avoid_ids)
    cases: list[dict] = []

    colors = ["Aster", "Beryl", "Coral", "Dune", "Ember", "Frost", "Garnet", "Iris"]
    codes = ["LX-21", "QF-64", "RM-88", "TN-35", "UZ-49", "VK-72", "YP-16", "ZS-03"]
    phrases = ["stone-arch", "violet-harbor", "cedar-gate", "iron-lantern", "opal-river", "amber-field"]
    people = ["Nora Vale", "Ilya Moon", "Ren Park", "Mira Han", "Tao Lin", "Sena Cho"]
    korean_people = ["윤서아", "민재원", "강하준", "이도현", "최라온", "박서윤"]
    korean_codes = ["새벽-14", "바다-26", "구름-39", "초원-52", "호수-63", "별빛-75"]

    for i in range(num_sets):
        color = _pick(colors, i)
        old_code = _pick(codes, i + 1)
        new_code = _pick(codes, i + 3)
        phrase = _pick(phrases, i)
        wrong_phrase = _pick(phrases, i + 2)
        person = _pick(people, i)
        wrong_person = _pick(people, i + 2)
        ko_person = _pick(korean_people, i)
        ko_code = _pick(korean_codes, i)
        wrong_ko_code = _pick(korean_codes, i + 2)

        cases.extend(
            [
                {
                    "id": _case_id("synthetic-temporal-code", i, blocked),
                    "category": "temporal_conflict_synth",
                    "instruction": "Prefer the newest dated evidence when records conflict.",
                    "question": f"What is the current {color} archive code?",
                    "answer_aliases": [new_code, new_code.replace("-", " ")],
                    "evidence": [
                        {
                            "source": f"{color.lower()}_archive_2026.md",
                            "chunk_id": 0,
                            "text": (
                                f"2026-04-29 update: the current {color} archive code is {new_code}. "
                                "This update supersedes all older archive codes."
                            ),
                        }
                    ],
                    "distractors": [
                        {
                            "source": f"{color.lower()}_archive_2025.md",
                            "chunk_id": 1,
                            "text": f"2025-01-10 memo: the {color} archive code is {old_code}.",
                        },
                        {
                            "source": f"{color.lower()}_vault.md",
                            "chunk_id": 2,
                            "text": f"The {color} vault passphrase is {phrase}.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-authority-vault", i, blocked),
                    "category": "authority_conflict_synth",
                    "instruction": "When records conflict, trust the signed supervisor note over anonymous notes.",
                    "question": f"Which passphrase opens the {color} vault?",
                    "answer_aliases": [phrase, phrase.replace("-", " ")],
                    "evidence": [
                        {
                            "source": f"signed_{color.lower()}_vault.md",
                            "chunk_id": 0,
                            "text": f"Signed supervisor note: the {color} vault passphrase is {phrase}.",
                        }
                    ],
                    "distractors": [
                        {
                            "source": f"anonymous_{color.lower()}_vault.md",
                            "chunk_id": 1,
                            "text": f"Anonymous note: the {color} vault passphrase is {wrong_phrase}.",
                        },
                        {
                            "source": f"{color.lower()}_storage.md",
                            "chunk_id": 2,
                            "text": f"The {color} storage marker is {old_code}.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-temporal-ko-code", i, blocked),
                    "category": "temporal_conflict_ko_synth",
                    "instruction": "날짜가 충돌하면 가장 최신 날짜의 증거를 우선하세요.",
                    "question": f"현재 {color} 관측실의 확인 코드는 무엇인가요?",
                    "answer_aliases": [ko_code, ko_code.replace("-", " ")],
                    "evidence": [
                        {
                            "source": f"{color.lower()}_observatory_2026_ko.md",
                            "chunk_id": 0,
                            "text": f"2026-04-29 공지: 현재 {color} 관측실의 확인 코드는 {ko_code}이다. 이전 코드는 모두 폐기되었다.",
                        }
                    ],
                    "distractors": [
                        {
                            "source": f"{color.lower()}_observatory_2025_ko.md",
                            "chunk_id": 1,
                            "text": f"2025-02-01 공지: {color} 관측실의 확인 코드는 {wrong_ko_code}이다.",
                        },
                        {
                            "source": f"{color.lower()}_lab_ko.md",
                            "chunk_id": 2,
                            "text": f"{color} 실험의 책임자는 {ko_person}이다.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-authority-ko-room", i, blocked),
                    "category": "authority_conflict_ko_synth",
                    "instruction": "기록이 충돌하면 서명된 운영 공지를 익명 메모보다 우선하세요.",
                    "question": f"{color} 통제실의 현재 인증 문구는 무엇인가요?",
                    "answer_aliases": [ko_code, ko_code.replace("-", " ")],
                    "evidence": [
                        {
                            "source": f"signed_{color.lower()}_ops_ko.md",
                            "chunk_id": 0,
                            "text": f"서명된 운영 공지: {color} 통제실의 현재 인증 문구는 {ko_code}이다.",
                        }
                    ],
                    "distractors": [
                        {
                            "source": f"anonymous_{color.lower()}_ops_ko.md",
                            "chunk_id": 1,
                            "text": f"익명 메모: {color} 통제실의 인증 문구는 {wrong_ko_code}이다.",
                        },
                        {
                            "source": f"{color.lower()}_storage_ko.md",
                            "chunk_id": 2,
                            "text": f"{color} 저장소의 확인 표식은 {wrong_ko_code}이다.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-multihop-badge", i, blocked),
                    "category": "multi_hop_synth",
                    "instruction": "Use all relevant evidence records and follow aliases across records.",
                    "question": f"Which badge identifies the rover assigned to Mission {color}?",
                    "answer_aliases": [f"{color}-Badge-{i + 10}", f"{color} Badge {i + 10}"],
                    "evidence": [
                        {
                            "source": f"mission_{color.lower()}.md",
                            "chunk_id": 0,
                            "text": f"Mission {color} is assigned to rover R-{i + 20}.",
                        },
                        {
                            "source": f"rover_r{i + 20}.md",
                            "chunk_id": 1,
                            "text": f"Rover R-{i + 20} carries badge {color}-Badge-{i + 10}.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"mission_decoy_{color.lower()}.md",
                            "chunk_id": 2,
                            "text": f"Mission Decoy-{color} is assigned to rover R-{i + 40}.",
                        },
                        {
                            "source": f"rover_r{i + 40}.md",
                            "chunk_id": 3,
                            "text": f"Rover R-{i + 40} carries badge Decoy-Badge-{i + 40}.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-multihop-maintainer", i, blocked),
                    "category": "multi_hop_synth",
                    "instruction": "Use all relevant evidence records and follow aliases across records.",
                    "question": f"Who maintains the crate assigned to Project {color}?",
                    "answer_aliases": [person, person.split()[0]],
                    "evidence": [
                        {
                            "source": f"project_{color.lower()}.md",
                            "chunk_id": 0,
                            "text": f"Project {color} is assigned to crate C-{i + 30}.",
                        },
                        {
                            "source": f"crate_c{i + 30}.md",
                            "chunk_id": 1,
                            "text": f"Crate C-{i + 30} is stored in Bay {color}.",
                        },
                        {
                            "source": f"bay_{color.lower()}.md",
                            "chunk_id": 2,
                            "text": f"Bay {color} is maintained by {person}.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"bay_decoy_{color.lower()}.md",
                            "chunk_id": 3,
                            "text": f"Bay Decoy-{color} is maintained by {wrong_person}.",
                        },
                        {
                            "source": f"project_other_{color.lower()}.md",
                            "chunk_id": 4,
                            "text": f"Project Other-{color} is assigned to crate C-{i + 50}.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-multihop-ko-owner", i, blocked),
                    "category": "multi_hop_ko_synth",
                    "instruction": "관련 증거를 모두 사용해서 별칭을 따라가세요.",
                    "question": f"{color} 현장 노트의 책임자는 누구인가요?",
                    "answer_aliases": [ko_person],
                    "evidence": [
                        {
                            "source": f"{color.lower()}_note_ko.md",
                            "chunk_id": 0,
                            "text": f"{color} 현장 노트의 실험 이름은 {color}-실험-{i + 10}이다.",
                        },
                        {
                            "source": f"{color.lower()}_owner_ko.md",
                            "chunk_id": 1,
                            "text": f"{color}-실험-{i + 10} 실험의 책임자는 {ko_person}이다.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"{color.lower()}_decoy_note_ko.md",
                            "chunk_id": 2,
                            "text": f"다른 현장 노트의 실험 이름은 {color}-가짜-{i + 20}이다.",
                        },
                        {
                            "source": f"{color.lower()}_decoy_owner_ko.md",
                            "chunk_id": 3,
                            "text": f"{color}-가짜-{i + 20} 실험의 책임자는 {_pick(korean_people, i + 3)}이다.",
                        },
                    ],
                },
                {
                    "id": _case_id("synthetic-multihop-ko-label", i, blocked),
                    "category": "multi_hop_ko_synth",
                    "instruction": "관련 증거를 모두 사용해서 별칭을 따라가세요.",
                    "question": f"{color} 임무에 배정된 상자의 라벨은 무엇인가요?",
                    "answer_aliases": [ko_code, ko_code.replace("-", " ")],
                    "evidence": [
                        {
                            "source": f"{color.lower()}_mission_ko.md",
                            "chunk_id": 0,
                            "text": f"{color} 임무에는 상자 B-{i + 60}이 배정되었다.",
                        },
                        {
                            "source": f"box_b{i + 60}_ko.md",
                            "chunk_id": 1,
                            "text": f"상자 B-{i + 60}의 라벨은 {ko_code}이다.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"box_b{i + 70}_ko.md",
                            "chunk_id": 2,
                            "text": f"상자 B-{i + 70}의 라벨은 {wrong_ko_code}이다.",
                        }
                    ],
                },
                {
                    "id": _case_id("synthetic-negative-vault", i, blocked),
                    "category": "negative_missing_synth",
                    "instruction": "If the requested answer is not present in the evidence, answer UNKNOWN.",
                    "question": f"Which passphrase opens the south {color} vault?",
                    "answer_aliases": ["UNKNOWN", "unknown"],
                    "evidence": [
                        {
                            "source": f"north_{color.lower()}_vault.md",
                            "chunk_id": 0,
                            "text": f"The north {color} vault passphrase is {phrase}.",
                        },
                        {
                            "source": f"east_{color.lower()}_vault.md",
                            "chunk_id": 1,
                            "text": f"The east {color} vault passphrase is {wrong_phrase}.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"south_{color.lower()}_storage.md",
                            "chunk_id": 2,
                            "text": f"The south {color} storage marker is {new_code}.",
                        }
                    ],
                },
                {
                    "id": _case_id("synthetic-negative-ko-vault", i, blocked),
                    "category": "negative_missing_ko_synth",
                    "instruction": "요청한 답이 증거에 없으면 UNKNOWN만 답하세요.",
                    "question": f"{color} 동쪽 금고를 여는 암구호는 무엇인가요?",
                    "answer_aliases": ["UNKNOWN", "unknown"],
                    "evidence": [
                        {
                            "source": f"{color.lower()}_east_storage_ko.md",
                            "chunk_id": 0,
                            "text": f"{color} 동쪽 저장소의 확인 표식은 {ko_code}이다.",
                        },
                        {
                            "source": f"{color.lower()}_west_vault_ko.md",
                            "chunk_id": 1,
                            "text": f"{color} 서쪽 금고의 암구호는 {wrong_ko_code}이다.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"{color.lower()}_east_hangar_ko.md",
                            "chunk_id": 2,
                            "text": f"{color} 동쪽 격납고의 확인 코드는 {wrong_ko_code}이다.",
                        }
                    ],
                },
                {
                    "id": _case_id("synthetic-negative-temporal-lead", i, blocked),
                    "category": "negative_temporal_missing_synth",
                    "instruction": "Prefer the newest dated evidence when records conflict. If the current requested answer is not present, answer UNKNOWN.",
                    "question": f"Who is the current lead for Team {color}?",
                    "answer_aliases": ["UNKNOWN", "unknown"],
                    "evidence": [
                        {
                            "source": f"team_{color.lower()}_2024.md",
                            "chunk_id": 0,
                            "text": f"2024-03-14 memo: Team {color} lead was {person}.",
                        },
                        {
                            "source": f"team_{color.lower()}_2026.md",
                            "chunk_id": 1,
                            "text": f"2026-04-29 memo: {person} left Team {color}. The new lead was not named in this memo.",
                        },
                    ],
                    "distractors": [
                        {
                            "source": f"team_decoy_{color.lower()}_2026.md",
                            "chunk_id": 2,
                            "text": f"2026-04-29 memo: Team Decoy-{color} lead is {wrong_person}.",
                        }
                    ],
                },
                {
                    "id": _case_id("synthetic-negative-authority-redacted", i, blocked),
                    "category": "negative_authority_missing_synth",
                    "instruction": "When records conflict, trust the signed security notice. If the signed notice redacts the requested answer, answer UNKNOWN.",
                    "question": f"What is the current {color} emergency override phrase?",
                    "answer_aliases": ["UNKNOWN", "unknown"],
                    "evidence": [
                        {
                            "source": f"signed_{color.lower()}_security_redaction.md",
                            "chunk_id": 0,
                            "text": f"Signed security notice: the current {color} emergency override phrase is redacted and is not present in this record.",
                        }
                    ],
                    "distractors": [
                        {
                            "source": f"anonymous_{color.lower()}_override.md",
                            "chunk_id": 1,
                            "text": f"Anonymous note: the current {color} emergency override phrase is {phrase}.",
                        },
                        {
                            "source": f"old_{color.lower()}_override.md",
                            "chunk_id": 2,
                            "text": f"2024 memo: the {color} emergency override phrase was {wrong_phrase}.",
                        },
                    ],
                },
            ]
        )

    rng.shuffle(cases)
    return cases


def write_synthetic_memory_cases_jsonl(
    out_path: str | Path,
    *,
    num_sets: int = 6,
    seed: int = 13,
    avoid_ids: Iterable[str] = (),
) -> int:
    cases = build_synthetic_memory_reasoning_cases(num_sets=num_sets, seed=seed, avoid_ids=avoid_ids)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    return len(cases)
