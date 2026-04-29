import unittest


class CriticalSynthesisEvalTests(unittest.TestCase):
    def test_build_prompt_requires_positive_conclusion_after_critique(self):
        from qtrm_mm.eval.critical_synthesis import build_critical_synthesis_prompt

        case = {
            "id": "bon-gakgyo-frame",
            "question": "기존 종교의 문제점을 파악하고 새로운 종교적 시야를 제시하세요.",
            "domain": "religion",
            "evidence": [
                {
                    "source": "본각교_요약.md",
                    "text": "외부 권위에 기대지 말고 호흡과 자기성찰로 내면의 자유를 회복한다.",
                }
            ],
        }

        prompt = build_critical_synthesis_prompt(case)

        self.assertIn("Do not stop at suspicion", prompt)
        self.assertIn("Critique:", prompt)
        self.assertIn("Preserve:", prompt)
        self.assertIn("Risks:", prompt)
        self.assertIn("Positive conclusion:", prompt)
        self.assertIn("본각교_요약.md", prompt)
        self.assertIn("호흡", prompt)

    def test_build_target_preserves_values_and_positive_conclusion(self):
        from qtrm_mm.eval.critical_synthesis import build_critical_synthesis_target

        case = {
            "id": "atonement-growth",
            "question": "영적 성장과 대속 중 무엇이 중요한가?",
            "critique_points": ["단일 교파의 언어를 절대화하면 질문의 맥락을 놓친다."],
            "preserve_values": ["대속은 구원의 객관적 기초", "영적 성장은 삶에서 드러나는 열매"],
            "risk_notes": ["교파별 해석 차이를 지워서는 안 된다."],
            "positive_conclusion": "대속과 영적 성장은 경쟁 관계가 아니라 기초와 열매의 관계로 통합할 수 있다.",
        }

        target = build_critical_synthesis_target(case)

        self.assertIn("Critique:", target)
        self.assertIn("Preserve:", target)
        self.assertIn("Risks:", target)
        self.assertIn("Positive conclusion:", target)
        self.assertIn("기초와 열매", target)

    def test_probe_cases_require_positive_non_cynical_synthesis(self):
        from qtrm_mm.eval.critical_synthesis import load_critical_synthesis_cases

        cases = load_critical_synthesis_cases("data/eval/critical_synthesis_probe.jsonl")

        self.assertGreaterEqual(len(cases), 2)
        for case in cases:
            self.assertTrue(case.get("question"))
            self.assertTrue(case.get("critique_points"))
            self.assertTrue(case.get("preserve_values"))
            self.assertTrue(case.get("risk_notes"))
            self.assertTrue(case.get("positive_conclusion"))
            self.assertNotIn("모두 가짜", case["positive_conclusion"])


if __name__ == "__main__":
    unittest.main()
