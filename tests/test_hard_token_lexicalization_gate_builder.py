import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = Path("scripts/325_build_hard_token_lexicalization_gate.py")
    spec = importlib.util.spec_from_file_location("hard_token_gate_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class HardTokenLexicalizationGateBuilderTests(unittest.TestCase):
    def test_selects_cases_where_donor_content_token_rank_is_hard(self):
        module = _load_module()
        rank_rows = [
            {
                "id": "easy",
                "mode": "donor_only_no_evidence",
                "answer": "52",
                "content_first_rank": 1,
                "max_rank": 1,
            },
            {
                "id": "hard",
                "mode": "donor_only_no_evidence",
                "answer": "80,120",
                "content_first_rank": 6,
                "max_rank": 6,
            },
        ]

        selected, reasons = module.hard_case_ids(
            rank_rows,
            mode="donor_only_no_evidence",
            min_content_first_rank=2,
            min_max_rank=0,
        )

        self.assertEqual(selected, {"hard"})
        self.assertEqual(reasons["hard"]["content_first_rank"], 6)

    def test_build_gate_preserves_original_case_order_and_writes_summary(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_probe = root / "ranks.jsonl"
            cases = root / "cases.jsonl"
            out = root / "hard.jsonl"
            summary_out = root / "summary.json"
            _write_jsonl(
                rank_probe,
                [
                    {
                        "id": "b",
                        "mode": "donor_only_no_evidence",
                        "content_first_rank": 4,
                        "max_rank": 4,
                    },
                    {
                        "id": "a",
                        "mode": "donor_only_no_evidence",
                        "content_first_rank": 5,
                        "max_rank": 5,
                    },
                ],
            )
            _write_jsonl(
                cases,
                [
                    {"id": "a", "prompt": "A"},
                    {"id": "b", "prompt": "B"},
                    {"id": "c", "prompt": "C"},
                ],
            )

            args = module.build_arg_parser().parse_args(
                [
                    "--rank-probe",
                    str(rank_probe),
                    "--cases",
                    str(cases),
                    "--out",
                    str(out),
                    "--summary-out",
                    str(summary_out),
                ]
            )
            summary = module.build_gate(args)

            selected = module.load_jsonl(out)
            self.assertEqual([row["id"] for row in selected], ["a", "b"])
            self.assertEqual(summary["selected_count"], 2)
            self.assertEqual(json.loads(summary_out.read_text())["selected_ids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
