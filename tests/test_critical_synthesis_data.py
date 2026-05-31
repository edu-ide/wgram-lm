import json
import tempfile
from pathlib import Path
import unittest


class CriticalSynthesisDataTests(unittest.TestCase):
    def test_build_trace_rows_preserves_positive_synthesis_shape(self):
        from wgram_lm.training.critical_synthesis_data import build_critical_synthesis_trace_rows

        case = {
            "id": "bon-gakgyo-frame",
            "domain": "religion_new_frame",
            "question": "기존 종교를 비판하고 긍정적 새 시야를 제시하세요.",
            "evidence": [
                {
                    "source": "본각교_요약.md",
                    "source_type": "local_doctrine_note",
                    "text": "호흡과 자기성찰로 내면의 자유를 회복한다.",
                }
            ],
            "critique_points": ["공포와 죄책감은 사람을 작게 만든다."],
            "preserve_values": ["호흡", "자기성찰", "자비"],
            "risk_notes": ["새 프레임도 절대화되면 안 된다."],
            "reframe": "통제 구조는 걷어내고 수행 가치는 보존한다.",
            "positive_conclusion": "자유와 자비를 키우는 실천 철학으로 정리한다.",
        }

        rows = build_critical_synthesis_trace_rows([case])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["type"], "critical_synthesis_trace")
        self.assertEqual(row["case_id"], "bon-gakgyo-frame")
        self.assertEqual(row["domain"], "religion_new_frame")
        self.assertIn("Do not stop at suspicion", row["prompt"])
        self.assertIn("본각교_요약.md", row["prompt"])
        self.assertIn("Critique:", row["answer"])
        self.assertIn("Preserve:", row["answer"])
        self.assertIn("Risks:", row["answer"])
        self.assertIn("Reframe:", row["answer"])
        self.assertIn("Positive conclusion:", row["answer"])
        self.assertIn("자유와 자비", row["answer"])

    def test_write_trace_jsonl_from_probe_cases(self):
        from wgram_lm.training.critical_synthesis_data import write_critical_synthesis_trace_jsonl

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "critical_synthesis_traces.jsonl"
            count = write_critical_synthesis_trace_jsonl(
                "data/eval/critical_synthesis_probe.jsonl",
                out,
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, len(rows))
        self.assertGreaterEqual(count, 2)
        self.assertTrue(all(row["type"] == "critical_synthesis_trace" for row in rows))
        self.assertTrue(all("Positive conclusion:" in row["answer"] for row in rows))


if __name__ == "__main__":
    unittest.main()
