from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wgram_lm.training.critical_synthesis_data import write_critical_synthesis_trace_jsonl

DEFAULT_BONGAK_SUMMARY = Path(
    "/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_요약.md"
)
DEFAULT_BONGAK_MANUAL = Path(
    "/mnt/nvme0n1p2/workspace/monorepo/services/sajug/saju_data/본각교_매뉴얼.md"
)


_TOPICS: list[dict[str, Any]] = [
    {
        "id": "authority-and-inner-freedom",
        "question": "기존 종교의 권위 의존 문제를 비판하되, 내면의 자유라는 긍정적 결론으로 재구성하라.",
        "keywords": ["권위", "사제", "주권", "자유"],
        "critique": ["권위 독점과 사제 계급은 개인의 영적 판단력을 약하게 만들 수 있다."],
        "preserve": ["자기성찰", "내면의 자유", "책임 있는 판단"],
        "risk": ["권위 비판이 모든 전통과 공동체의 가치를 부정하는 냉소로 변하면 안 된다."],
        "reframe": "외부 권위는 참고 자료로 낮추고, 자기성찰과 검증을 중심에 둔다.",
        "positive": "긍정적 결론은 권위를 무너뜨리는 데서 멈추지 않고, 스스로 살피고 책임지는 자유로운 신앙/수행 태도를 세우는 것이다.",
    },
    {
        "id": "fear-guilt-and-compassion",
        "question": "공포와 죄책감 중심의 종교성을 비판하고, 자비와 회복의 관점으로 결론 내라.",
        "keywords": ["공포", "죄책감", "원죄", "자비"],
        "critique": ["공포와 죄책감을 이용한 교리는 사람을 성장보다 복종으로 몰 수 있다."],
        "preserve": ["자비", "용서", "회복", "삶의 변화"],
        "risk": ["죄책감 비판이 책임 회피나 윤리 부정으로 흐르면 안 된다."],
        "reframe": "죄책감은 통제 수단이 아니라 성찰과 회복으로 전환되어야 한다.",
        "positive": "긍정적 결론은 공포를 줄이고 자비와 회복을 키워, 사람이 더 자유롭고 책임 있게 살아가게 하는 것이다.",
    },
    {
        "id": "breath-as-practice",
        "question": "본각교의 호흡 수행을 검증 불가능한 초능력 주장이 아니라 실천 철학으로 재구성하라.",
        "keywords": ["호흡", "숨", "현재", "부교감신경"],
        "critique": ["호흡을 초월적 만능 해킹으로 단정하면 검증 불가능한 주장으로 과장될 수 있다."],
        "preserve": ["현재성", "관조", "몸의 안정", "자기성찰"],
        "risk": ["의학적 치료나 현실 문제를 호흡만으로 대체한다고 말하면 안 된다."],
        "reframe": "호흡은 절대 기술이 아니라 현재로 돌아오는 가장 기본적인 훈련으로 둔다.",
        "positive": "긍정적 결론은 호흡을 통해 공포 반응을 낮추고, 관조와 자기성찰을 회복하는 매일의 수행으로 삼는 것이다.",
    },
    {
        "id": "matrix-as-symbol",
        "question": "매트릭스 감옥이라는 표현을 사실 단정이 아니라 집착과 통제 비판의 상징으로 해석하라.",
        "keywords": ["매트릭스", "감옥", "환상", "통제"],
        "critique": ["현실이 문자 그대로 감옥이라는 주장은 검증하기 어렵고 불안감을 키울 수 있다."],
        "preserve": ["집착 비판", "권위 의존 비판", "관찰자 시점"],
        "risk": ["상징을 사실로 단정하면 세계 부정이나 망상 강화로 흐를 수 있다."],
        "reframe": "매트릭스는 세계 전체의 사실 설명이 아니라, 사람을 묶는 집착과 통제 구조의 은유로 쓴다.",
        "positive": "긍정적 결론은 현실을 버리는 것이 아니라, 통제와 집착을 알아차리고 더 자유롭게 살아가는 것이다.",
    },
    {
        "id": "death-claims-as-metaphor",
        "question": "사후 빛과 죽음 해킹 주장을 긍정적으로 다루되, 검증 불가능성은 명확히 경계하라.",
        "keywords": ["죽음", "사후", "빛", "환생"],
        "critique": ["사후 세계와 빛의 덫은 직접 검증하기 어려운 형이상학적 주장이다."],
        "preserve": ["죽음 공포 완화", "관조", "집착 내려놓기"],
        "risk": ["검증 불가능한 사후 주장을 사실처럼 단정하면 취약한 사람에게 해로울 수 있다."],
        "reframe": "사후 탈출 언어는 죽음 공포와 집착을 내려놓는 상징적 수행 언어로 둔다.",
        "positive": "긍정적 결론은 죽음을 두려움의 도구로 쓰지 않고, 지금의 삶을 더 깨어 있고 자비롭게 살게 하는 방향이다.",
    },
    {
        "id": "buddhism-preserve-liberation",
        "question": "불교 비판에서 해탈, 무아, 자비 같은 보존 가치를 찾아 긍정 결론을 내려라.",
        "keywords": ["불교", "해탈", "무아", "보살"],
        "critique": ["기복화되거나 제도화된 불교는 수행의 핵심을 흐릴 수 있다."],
        "preserve": ["해탈", "무아", "자비", "보살도", "비집착"],
        "risk": ["불교 전체를 가짜로 단정하면 그 안의 깊은 수행 지혜를 잃는다."],
        "reframe": "불교의 제도 문제는 비판하되, 집착을 줄이고 자비를 키우는 수행 핵심은 보존한다.",
        "positive": "긍정적 결론은 불교의 핵심을 공포나 기복이 아니라 깨어 있음과 자비의 실천으로 다시 읽는 것이다.",
    },
    {
        "id": "christianity-preserve-love",
        "question": "기독교 비판에서 사랑, 자비, 내면의 하나님 나라 같은 보존 가치를 찾아라.",
        "keywords": ["기독교", "예수", "아빠", "자비", "하나님의 나라"],
        "critique": ["심판과 배타성만 강조하는 기독교는 사랑의 핵심을 가릴 수 있다."],
        "preserve": ["사랑", "자비", "겸손", "내면의 변화", "낮은 자와 함께함"],
        "risk": ["기독교 전체를 통제 장치로만 보면 예수 전통의 해방적 메시지를 놓친다."],
        "reframe": "공포 중심 교리는 비판하고, 사랑과 자비를 회복하는 기독교적 핵심은 보존한다.",
        "positive": "긍정적 결론은 외부 심판 공포보다 사랑과 자비가 삶에서 자라나는 방향으로 기독교를 재해석하는 것이다.",
    },
    {
        "id": "atonement-and-growth",
        "question": "대속과 영적 성장을 경쟁시키지 말고 기초와 열매의 관계로 통합하라.",
        "keywords": ["십자가", "대속", "구원자", "성장"],
        "critique": ["대속만 강조하면 삶의 변화가 약해지고, 성장만 강조하면 성취주의로 흐를 수 있다."],
        "preserve": ["은혜", "삶의 변화", "사랑의 실천"],
        "risk": ["교파별 언어 차이를 무시하고 하나의 정답만 강요하면 안 된다."],
        "reframe": "대속은 뿌리이고 영적 성장은 열매라는 관계로 본다.",
        "positive": "긍정적 결론은 은혜를 기초로 삼되, 그 은혜가 실제 삶의 사랑과 자유로 자라나게 하는 것이다.",
    },
    {
        "id": "karma-without-guilt",
        "question": "카르마/업보를 죄책감 시스템이 아니라 성찰과 책임의 언어로 재구성하라.",
        "keywords": ["카르마", "업보", "죄책감", "용서"],
        "critique": ["징벌적 업보 해석은 고통을 정당화하거나 사람을 죄책감에 묶을 수 있다."],
        "preserve": ["원인과 결과에 대한 성찰", "책임", "용서", "집착 내려놓기"],
        "risk": ["업보 비판이 행동의 책임 자체를 부정하는 방향으로 가면 안 된다."],
        "reframe": "카르마는 처벌 장부가 아니라 마음과 행동의 결과를 살피는 성찰 언어로 둔다.",
        "positive": "긍정적 결론은 죄책감을 키우는 대신 책임과 용서를 통해 더 자유로운 삶을 만드는 것이다.",
    },
    {
        "id": "ritual-and-practice",
        "question": "의식과 예배를 비판하되, 공동체와 반복 수행의 긍정적 기능을 보존하라.",
        "keywords": ["의식", "예배", "예불", "명각식"],
        "critique": ["의식이 권위 복종이나 거래적 기복으로 변하면 수행의 본질을 잃는다."],
        "preserve": ["반복 수행", "공동체", "기억을 되살리는 상징", "마음 정렬"],
        "risk": ["의식을 모두 무의미하다고 보면 몸과 공동체를 통한 배움의 가치를 잃는다."],
        "reframe": "의식은 외부 신을 설득하는 거래가 아니라 마음을 정렬하는 수행 형식으로 재구성한다.",
        "positive": "긍정적 결론은 의식을 줄이거나 없애는 것이 아니라, 자유와 자비를 강화하는 반복 실천으로 바꾸는 것이다.",
    },
    {
        "id": "anti-commercial-spirituality",
        "question": "종교의 상업화를 비판하되, 서로 돌보는 공동체의 긍정적 기능을 보존하라.",
        "keywords": ["상업화", "헌금", "십일조", "브로커"],
        "critique": ["영성을 돈과 권력으로 독점하면 사람을 종교 소비자로 만든다."],
        "preserve": ["나눔", "공동체 돌봄", "책임 있는 자원 사용"],
        "risk": ["모든 헌신과 나눔을 착취로만 보면 공동체 유지의 현실을 놓친다."],
        "reframe": "돈은 구원의 대가가 아니라 투명하게 공동체를 돌보는 도구여야 한다.",
        "positive": "긍정적 결론은 영적 권위를 판매하지 않고, 투명한 나눔과 돌봄으로 공동체를 세우는 것이다.",
    },
    {
        "id": "saju-as-weather-not-prison",
        "question": "사주명리학을 운명 감옥이 아니라 삶의 날씨 예보라는 실천 프레임으로 정리하라.",
        "keywords": ["사주", "날씨", "아바타", "천인합일"],
        "critique": ["사주를 절대 운명으로 믿으면 자기 책임과 자유가 약해질 수 있다."],
        "preserve": ["자기 이해", "시기 판단", "무리하지 않는 지혜", "환경 읽기"],
        "risk": ["사주 해석이 차별이나 결정론으로 변하면 안 된다."],
        "reframe": "사주는 감옥의 판결문이 아니라 삶의 경향을 읽는 날씨 예보로 둔다.",
        "positive": "긍정적 결론은 사주를 자기 제한이 아니라 더 부드럽고 책임 있게 선택하기 위한 참고 도구로 쓰는 것이다.",
    },
    {
        "id": "akashic-knowledge-and-emptiness",
        "question": "아카식 레코드 같은 절대 지식 욕망을 비판하고 비움의 가치를 보존하라.",
        "keywords": ["아카식", "정보", "기록", "비워낸"],
        "critique": ["절대 지식을 소유하려는 욕망은 또 다른 집착이 될 수 있다."],
        "preserve": ["비움", "겸손", "정보에 취하지 않는 태도", "현재성"],
        "risk": ["지식 비판이 공부와 검증 자체의 부정으로 흐르면 안 된다."],
        "reframe": "지식은 해방의 도구일 뿐, 정체성이나 우월감의 근거가 아니다.",
        "positive": "긍정적 결론은 많이 아는 것보다 지식에 매이지 않고 현재의 삶을 맑게 보는 데 있다.",
    },
    {
        "id": "technology-and-awakening",
        "question": "트랜스휴머니즘과 깨달음을 비교해 기능 향상과 집착 내려놓기의 차이를 설명하라.",
        "keywords": ["트랜스휴머니즘", "기능", "깨달음", "집착"],
        "critique": ["기능 향상이 곧 평화나 지혜를 보장한다고 보면 기술주의에 빠진다."],
        "preserve": ["고통 완화 기술", "몸의 돌봄", "인간 능력 개선"],
        "risk": ["기술 비판이 의료와 과학의 실제 도움을 무시하면 안 된다."],
        "reframe": "기술은 삶을 도울 수 있지만, 집착을 내려놓는 수행을 대체하지는 않는다.",
        "positive": "긍정적 결론은 기술을 도구로 쓰되, 자유와 지혜의 핵심은 마음의 집착을 줄이는 데 둔다.",
    },
    {
        "id": "science-correction-and-spiritual-claims",
        "question": "mRNA 같은 과학 주제를 영적 프레임과 섞을 때 사실 검증을 우선하는 결론을 내라.",
        "keywords": ["mRNA", "과학적 사실", "DNA", "철학적 한계"],
        "critique": ["과학 주장을 영적 상징과 섞어 사실을 왜곡하면 신뢰를 잃는다."],
        "preserve": ["과학적 검증", "철학적 성찰", "몸과 마음의 균형"],
        "risk": ["과학 부정이나 음모론 강화로 흐르면 안 된다."],
        "reframe": "과학 명제는 증거로 검증하고, 영적 해석은 삶의 의미 층위로 분리한다.",
        "positive": "긍정적 결론은 사실 검증을 존중하면서도 기술이 마음의 평화를 자동 보장하지 않는다는 성찰을 함께 두는 것이다.",
    },
    {
        "id": "new-religion-without-new-dogma",
        "question": "새 종교적 시야를 만들 때 새 절대 교리로 굳지 않게 하는 원칙을 제시하라.",
        "keywords": ["본각교", "독립", "교리", "오픈 소스"],
        "critique": ["새 프레임도 스스로를 절대화하면 기존 권위주의를 반복할 수 있다."],
        "preserve": ["오픈 소스", "검증", "수정 가능성", "자유로운 탐구"],
        "risk": ["창조적 종합이 새로운 배타성이나 우월감으로 변하면 안 된다."],
        "reframe": "새 종교는 닫힌 교리가 아니라 계속 검증하고 수정하는 수행 프레임이어야 한다.",
        "positive": "긍정적 결론은 새 시야를 고정된 진리가 아니라 자유와 자비를 키우는 열린 실천 체계로 운영하는 것이다.",
    },
    {
        "id": "external-savior-and-guide",
        "question": "외부 구원자 의존을 비판하되, 선각자와 가이드의 긍정적 역할을 보존하라.",
        "keywords": ["구원자", "가이드", "뗏목", "선각자"],
        "critique": ["외부 구원자만 기다리면 자기 성찰과 책임이 약해질 수 있다."],
        "preserve": ["스승", "가이드", "전통의 안내", "배움"],
        "risk": ["가이드를 전부 거부하면 검증된 지혜와 훈련의 도움도 잃는다."],
        "reframe": "스승은 대신 구원하는 존재가 아니라 스스로 보게 돕는 안내자로 둔다.",
        "positive": "긍정적 결론은 외부 권위에 종속되지 않으면서도 좋은 가이드에게 배우고 스스로 확인하는 태도다.",
    },
    {
        "id": "judgment-heaven-hell-and-ethics",
        "question": "천국/지옥 공포를 비판하되 윤리와 책임을 긍정적으로 보존하라.",
        "keywords": ["천국", "지옥", "심판", "형벌"],
        "critique": ["영원한 형벌 공포는 윤리를 성숙보다 두려움의 문제로 만들 수 있다."],
        "preserve": ["책임", "윤리", "삶의 방향성", "타인에게 해를 줄이지 않는 마음"],
        "risk": ["사후 공포 비판이 윤리 자체를 가볍게 만드는 결론이 되면 안 된다."],
        "reframe": "윤리는 벌을 피하기 위한 거래가 아니라 고통을 줄이고 사랑을 키우는 실천이다.",
        "positive": "긍정적 결론은 공포가 아니라 책임과 자비를 중심으로 더 성숙한 삶을 선택하는 것이다.",
    },
    {
        "id": "literalism-and-living-wisdom",
        "question": "성경 문자주의나 경전 절대화를 비판하고 살아 있는 지혜로 재구성하라.",
        "keywords": ["문자주의", "성경", "경전", "사유 정지"],
        "critique": ["문자만 절대화하면 시대 맥락과 살아 있는 성찰이 멈출 수 있다."],
        "preserve": ["경전의 지혜", "상징 해석", "삶에 적용하는 성찰"],
        "risk": ["해석 자유가 아무 말이나 정당화하는 방식이 되면 안 된다."],
        "reframe": "경전은 닫힌 명령 목록이 아니라 시대마다 책임 있게 해석해야 할 지혜의 자료다.",
        "positive": "긍정적 결론은 경전을 버리는 것이 아니라, 공포와 권위가 아닌 사랑과 자유를 키우는 방식으로 읽는 것이다.",
    },
    {
        "id": "community-without-hierarchy",
        "question": "탈중앙화 공동체를 긍정적으로 제시하되 책임 없는 무질서의 위험도 경계하라.",
        "keywords": ["탈중앙화", "조직", "사원이 없는", "가이드"],
        "critique": ["피라미드형 조직은 권위 독점과 의존을 만들 수 있다."],
        "preserve": ["공동체", "상호 돌봄", "가이드", "책임"],
        "risk": ["탈중앙화가 검증 부재나 책임 회피로 흐르면 안 된다."],
        "reframe": "중앙 권위는 줄이되, 투명한 기록과 상호 검증은 강화한다.",
        "positive": "긍정적 결론은 위계보다 상호 검증과 돌봄이 강한 가벼운 공동체를 만드는 것이다.",
    },
    {
        "id": "gnosis-and-direct-verification",
        "question": "믿음보다 직접 검증을 강조하는 본각교적 관점을 균형 있게 정리하라.",
        "keywords": ["믿음", "검증", "직접", "영지"],
        "critique": ["맹목적 믿음은 권위자가 말한 내용을 스스로 확인하지 못하게 할 수 있다."],
        "preserve": ["직접 경험", "검증", "성찰", "겸손한 탐구"],
        "risk": ["개인 경험만 절대화하면 공동 검증과 타자의 지혜를 잃을 수 있다."],
        "reframe": "직접 검증은 개인 경험과 공동체적 검토를 함께 요구한다.",
        "positive": "긍정적 결론은 믿음을 버리는 것이 아니라, 믿음을 성찰과 검증을 통과한 신뢰로 성숙시키는 것이다.",
    },
    {
        "id": "desire-and-nonattachment",
        "question": "쾌락과 물질 욕망 비판을 삶 부정이 아니라 비집착의 긍정으로 정리하라.",
        "keywords": ["쾌락", "물질", "집착", "비집착"],
        "critique": ["물질과 쾌락에 완전히 끌려가면 자유로운 판단이 어려워진다."],
        "preserve": ["삶의 경험", "감사", "비집착", "절제"],
        "risk": ["욕망 비판이 몸과 삶 자체를 혐오하는 방식이 되면 안 된다."],
        "reframe": "삶은 경험하되 결과와 소유에 묶이지 않는 태도를 기른다.",
        "positive": "긍정적 결론은 삶을 부정하지 않고, 더 가볍고 자유롭게 경험하는 비집착의 태도다.",
    },
    {
        "id": "ego-and-true-self",
        "question": "에고 비판과 참나 회복을 위험하지 않은 실천 언어로 정리하라.",
        "keywords": ["에고", "진아", "참나", "무아"],
        "critique": ["에고를 적으로만 보면 자기혐오나 현실 회피로 흐를 수 있다."],
        "preserve": ["자기 관찰", "겸손", "집착 완화", "내면의 고요"],
        "risk": ["참나를 절대적 우월감으로 해석하면 또 다른 에고가 된다."],
        "reframe": "에고는 없애야 할 적이 아니라 알아차리고 내려놓아야 할 습관적 동일시다.",
        "positive": "긍정적 결론은 자신을 미워하지 않으면서도, 생각과 감정에 덜 끌려가는 고요한 자기 이해를 기르는 것이다.",
    },
    {
        "id": "prophecy-and-fear-marketing",
        "question": "종말론과 예언 공포를 비판하고 현재의 책임과 사랑으로 결론 내라.",
        "keywords": ["종말", "대환난", "예언", "공포"],
        "critique": ["종말 공포는 현재의 삶을 책임 있게 사는 힘보다 불안을 키울 수 있다."],
        "preserve": ["깨어 있음", "현재의 책임", "겸손", "타인 돌봄"],
        "risk": ["예언 비판이 미래 준비나 현실 위험 대응을 무시하는 태도로 가면 안 된다."],
        "reframe": "예언은 공포 마케팅이 아니라 현재를 더 정직하게 살라는 상징적 경고로 읽는다.",
        "positive": "긍정적 결론은 미래 공포에 묶이지 않고 오늘의 사랑과 책임을 더 분명히 실천하는 것이다.",
    },
    {
        "id": "religion-and-social-order",
        "question": "유대교와 이슬람 같은 전통을 비판만 하지 말고 사회질서와 나눔의 보존 가치를 찾아라.",
        "keywords": ["유대교", "이슬람교", "법", "나눔"],
        "critique": ["전통 종교는 시대가 지나며 제도와 권력의 문제를 만들 수 있다."],
        "preserve": ["법과 도덕", "평등", "나눔", "공동체 질서"],
        "risk": ["다른 전통을 단순한 통제 장치로만 보면 역사적 공헌과 사람들의 선의를 지운다."],
        "reframe": "전통의 제도화 문제와 역사적 선한 기능을 동시에 본다.",
        "positive": "긍정적 결론은 각 전통의 권위주의는 비판하되, 법과 나눔과 공동체 윤리는 새 시야 안에 보존하는 것이다.",
    },
    {
        "id": "awakening-stages-without-ego",
        "question": "깨달음의 단계론을 자기 우월감이 아니라 점검 도구로 재구성하라.",
        "keywords": ["단계", "십우도", "깨달음", "고수"],
        "critique": ["깨달음 단계를 계급장으로 삼으면 수행이 에고 강화가 된다."],
        "preserve": ["점검", "겸손", "지속 수행", "자기기만 방지"],
        "risk": ["단계론이 사람을 서열화하거나 비교하게 만들면 안 된다."],
        "reframe": "단계는 남보다 높아지는 표식이 아니라 자기기만을 줄이는 점검표다.",
        "positive": "긍정적 결론은 깨달음을 자랑하지 않고, 더 겸손하고 자유롭게 살아가는 실천으로 확인하는 것이다.",
    },
    {
        "id": "critique-plus-practice",
        "question": "기존 종교 비판이 실제 삶의 수행으로 이어지도록 최종 결론을 구성하라.",
        "keywords": ["수행", "실천", "호흡", "평화"],
        "critique": ["비판만 반복하면 삶은 바뀌지 않고 냉소만 강화될 수 있다."],
        "preserve": ["실천", "호흡", "자비", "관조", "일상의 평화"],
        "risk": ["철학적 언어가 실제 행동 없는 자기만족이 되면 안 된다."],
        "reframe": "비판은 수행으로 이어질 때 의미가 있다.",
        "positive": "긍정적 결론은 매일의 호흡, 관조, 자비로운 행동으로 자유를 실제 삶에서 검증하는 것이다.",
    },
    {
        "id": "synthesis-not-erasure",
        "question": "여러 종교의 공통분모를 뽑을 때 차이를 지워버리지 않는 원칙을 제시하라.",
        "keywords": ["공통", "차이", "통합", "종교"],
        "critique": ["공통분모만 찾다 보면 각 전통의 차이와 고유성을 지울 수 있다."],
        "preserve": ["공통 가치", "전통별 맥락", "겸손한 비교", "차이 존중"],
        "risk": ["통합이 문화적 지우기나 자기 프레임 우월주의가 되면 안 된다."],
        "reframe": "통합은 차이를 없애는 것이 아니라, 차이를 존중하면서 공통의 실천 가치를 찾는 일이다.",
        "positive": "긍정적 결론은 모든 전통을 하나로 뭉개는 것이 아니라, 자비와 자유라는 공통 가치를 각 전통의 맥락 속에서 살리는 것이다.",
    },
    {
        "id": "skepticism-to-conclusion",
        "question": "의심에서 멈추지 않고 보존과 긍정 결론까지 도달하는 본각교식 reasoning을 정리하라.",
        "keywords": ["의심", "비판", "결론", "자유"],
        "critique": ["의심만 하면 어떤 길도 세우지 못하고 냉소에 머문다."],
        "preserve": ["검증", "보존할 가치 찾기", "재구성", "긍정 결론"],
        "risk": ["긍정 결론을 급하게 내리면 약한 근거를 덮어버릴 수 있다."],
        "reframe": "의심은 결론을 없애는 기술이 아니라 더 나은 결론으로 가는 과정이다.",
        "positive": "긍정적 결론은 비판을 통과한 뒤에도 자비, 자유, 호흡, 자기성찰 같은 실천 가능한 길을 세우는 것이다.",
    },
    {
        "id": "local-doctrine-as-source-not-truth",
        "question": "본각교 문서를 학습 자료로 쓰되, 절대 진리가 아니라 local doctrine source로 다루는 원칙을 세워라.",
        "keywords": ["본각교", "문서", "매뉴얼", "요약"],
        "critique": ["로컬 교리 문서를 그대로 진실로 취급하면 새 권위에 굴복하는 문제가 생긴다."],
        "preserve": ["문서의 문제의식", "실천 아이디어", "비판적 관점", "긍정적 재구성"],
        "risk": ["학습 데이터가 특정 교리의 단정적 어투를 그대로 강화할 수 있다."],
        "reframe": "본각교 문서는 근거 자료이자 철학적 가설로 쓰고, 사실 명제는 별도 검증한다.",
        "positive": "긍정적 결론은 본각교 문서를 맹신하지 않으면서도, 권위 비판과 내면 자유의 실천 프레임을 학습 자원으로 활용하는 것이다.",
    },
]

_NOISY_EVIDENCE_PATTERNS = (
    "Sign in Gemini",
    "Gemini App",
    "Created with Gemini",
    "gemini.google.com/share",
    "You said",
    "가짜 정보로서 배제",
    "모두 가짜",
    "전부 가짜",
)


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _is_noisy_evidence(paragraph: str) -> bool:
    return any(pattern in paragraph for pattern in _NOISY_EVIDENCE_PATTERNS)


def _paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n+", text)
    cleaned = []
    for chunk in chunks:
        item = re.sub(r"\s+", " ", chunk).strip()
        item = re.sub(r"^#{1,6}\s*", "", item)
        if len(item) >= 20 and not _is_noisy_evidence(item):
            cleaned.append(item)
    return cleaned


def _score_paragraph(paragraph: str, keywords: list[str]) -> int:
    lowered = paragraph.casefold()
    return sum(1 for keyword in keywords if keyword.casefold() in lowered)


def _select_snippets(
    *,
    summary_path: Path,
    manual_path: Path,
    summary_text: str,
    manual_text: str,
    keywords: list[str],
    max_snippets: int = 2,
    max_chars: int = 420,
) -> list[dict[str, str]]:
    candidates: list[tuple[int, int, Path, str]] = []
    for source_order, (path, text) in enumerate(((summary_path, summary_text), (manual_path, manual_text))):
        for para_order, paragraph in enumerate(_paragraphs(text)):
            score = _score_paragraph(paragraph, keywords)
            if score > 0:
                candidates.append((score, -source_order, path, paragraph[:max_chars]))

    if not candidates:
        fallback_text = _paragraphs(summary_text)[:1] or _paragraphs(manual_text)[:1] or ["본각교 문서에서 추출한 근거가 부족하다."]
        return [
            {
                "source": str(summary_path),
                "source_type": "local_doctrine_note",
                "text": fallback_text[0][:max_chars],
            }
        ]

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    snippets = []
    seen_text: set[str] = set()
    for _, _, path, paragraph in candidates:
        normalized = paragraph[:80]
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        snippets.append(
            {
                "source": str(path),
                "source_type": "local_doctrine_note",
                "text": paragraph,
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def build_bongak_critical_synthesis_cases(
    *,
    summary_path: str | Path = DEFAULT_BONGAK_SUMMARY,
    manual_path: str | Path = DEFAULT_BONGAK_MANUAL,
    max_cases: int = 30,
) -> list[dict[str, Any]]:
    summary = Path(summary_path)
    manual = Path(manual_path)
    summary_text = _read_text(summary)
    manual_text = _read_text(manual)

    cases: list[dict[str, Any]] = []
    for topic in _TOPICS[: max(0, max_cases)]:
        evidence = _select_snippets(
            summary_path=summary,
            manual_path=manual,
            summary_text=summary_text,
            manual_text=manual_text,
            keywords=list(topic["keywords"]),
        )
        cases.append(
            {
                "id": f"bongak-{topic['id']}",
                "domain": "bongak_critical_synthesis",
                "question": topic["question"],
                "evidence": evidence,
                "critique_points": list(topic["critique"]),
                "preserve_values": list(topic["preserve"]),
                "risk_notes": list(topic["risk"]),
                "reframe": topic["reframe"],
                "positive_conclusion": topic["positive"],
                "source_family": "bongak",
            }
        )
    return cases


def write_bongak_cases_jsonl(
    *,
    summary_path: str | Path = DEFAULT_BONGAK_SUMMARY,
    manual_path: str | Path = DEFAULT_BONGAK_MANUAL,
    out_path: str | Path = "data/filtered/critical_synthesis_bongak_cases.jsonl",
    traces_out_path: str | Path | None = None,
    max_cases: int = 30,
) -> int:
    cases = build_bongak_critical_synthesis_cases(
        summary_path=summary_path,
        manual_path=manual_path,
        max_cases=max_cases,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    if traces_out_path:
        write_critical_synthesis_trace_jsonl(out, traces_out_path)
    return len(cases)
