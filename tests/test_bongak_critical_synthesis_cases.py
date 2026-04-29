import json
import tempfile
from pathlib import Path
import unittest


class BongakCriticalSynthesisCaseTests(unittest.TestCase):
    def test_build_bongak_cases_from_local_docs_preserves_positive_synthesis(self):
        from qtrm_mm.training.bongak_critical_synthesis_cases import (
            build_bongak_critical_synthesis_cases,
        )

        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "본각교_요약.md"
            manual = Path(td) / "본각교_매뉴얼.md"
            summary.write_text(
                "# 요약\n호흡과 자기성찰은 외부 권위에 기대지 않고 내면의 자유를 회복하는 실천이다.\n",
                encoding="utf-8",
            )
            manual.write_text(
                "\n\n".join(
                    [
                        "기존 종교의 공포와 죄책감은 사람을 작게 만들 수 있다.",
                        "사제 계급과 권위 독점은 영적 주권을 약하게 만든다.",
                        "매트릭스 감옥이라는 표현은 집착과 통제를 비판하는 상징으로 볼 수 있다.",
                        "자비와 관조는 삶을 덜 고통스럽게 만드는 긍정적 수행 가치다.",
                    ]
                ),
                encoding="utf-8",
            )

            cases = build_bongak_critical_synthesis_cases(
                summary_path=summary,
                manual_path=manual,
                max_cases=4,
            )

        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case["domain"], "bongak_critical_synthesis")
            self.assertTrue(case["question"])
            self.assertTrue(case["evidence"])
            self.assertTrue(case["critique_points"])
            self.assertTrue(case["preserve_values"])
            self.assertTrue(case["risk_notes"])
            self.assertTrue(case["positive_conclusion"])
            self.assertNotIn("모두 가짜", case["positive_conclusion"])

    def test_write_bongak_cases_jsonl_and_optional_traces(self):
        from qtrm_mm.training.bongak_critical_synthesis_cases import write_bongak_cases_jsonl

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "summary.md"
            manual = root / "manual.md"
            cases_out = root / "cases.jsonl"
            traces_out = root / "traces.jsonl"
            summary.write_text("호흡은 현재성, 관조, 자기성찰을 돕는다.", encoding="utf-8")
            manual.write_text(
                "기복신앙과 공포 마케팅은 비판하되 자비와 비집착은 보존한다.",
                encoding="utf-8",
            )

            count = write_bongak_cases_jsonl(
                summary_path=summary,
                manual_path=manual,
                out_path=cases_out,
                traces_out_path=traces_out,
                max_cases=3,
            )
            cases = [json.loads(line) for line in cases_out.read_text(encoding="utf-8").splitlines()]
            traces = [json.loads(line) for line in traces_out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(len(cases), 3)
        self.assertEqual(len(traces), 3)
        self.assertTrue(all(row["type"] == "critical_synthesis_trace" for row in traces))
        self.assertTrue(all("Positive conclusion:" in row["answer"] for row in traces))

    def test_build_cases_filters_scraped_chat_noise_from_evidence(self):
        from qtrm_mm.training.bongak_critical_synthesis_cases import (
            build_bongak_critical_synthesis_cases,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "summary.md"
            manual = root / "manual.md"
            summary.write_text(
                "자비와 자유를 함께 살리는 수행 관점이 필요하다.",
                encoding="utf-8",
            )
            manual.write_text(
                "\n\n".join(
                    [
                        "Sign in Gemini About Gemini 종교 통합 공통 차이 Created with Gemini You said",
                        "모든 종교에서 공통 분모를 추출하고 차이를 가짜 정보로서 배제한다.",
                        "차이를 지워버리는 일이 아니라 자비와 자유라는 가치를 찾는 일이다.",
                    ]
                ),
                encoding="utf-8",
            )

            cases = build_bongak_critical_synthesis_cases(
                summary_path=summary,
                manual_path=manual,
                max_cases=29,
            )

        case = next(row for row in cases if row["id"] == "bongak-synthesis-not-erasure")
        evidence_text = "\n".join(item["text"] for item in case["evidence"])
        self.assertNotIn("Sign in Gemini", evidence_text)
        self.assertNotIn("가짜 정보로서 배제", evidence_text)
        self.assertIn("차이를 지워버리는 일이 아니라", evidence_text)


if __name__ == "__main__":
    unittest.main()
