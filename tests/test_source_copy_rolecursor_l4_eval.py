from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


def load_module():
    path = Path("scripts/329_run_source_copy_rolecursor_l4_eval.py")
    spec = importlib.util.spec_from_file_location("source_copy_rolecursor_l4_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SourceCopyRoleCursorL4EvalTests(unittest.TestCase):
    def test_chunk_rows_preserves_order_and_size(self):
        module = load_module()
        rows = [{"id": str(i)} for i in range(5)]

        chunks = list(module.chunk_rows(rows, chunk_size=2))

        self.assertEqual(
            [[row["id"] for row in chunk] for chunk in chunks],
            [["0", "1"], ["2", "3"], ["4"]],
        )

    def test_eval_command_uses_source_copy_rolecursor_contract(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "configs/source_copy.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--cases",
                "cases.jsonl",
                "--out-dir",
                "out",
                "--max-length",
                "192",
                "--max-new-tokens",
                "12",
            ]
        )

        command = module.eval_command(
            args,
            mode="qtrm_core_steps_8_no_evidence",
            cases_path=Path("chunk.jsonl"),
            out_path=Path("chunk.out.jsonl"),
        )

        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--core-source-position-binder", command)
        self.assertIn("--core-source-position-binder-state-st", command)
        self.assertEqual(
            command[command.index("--mode") + 1],
            "qtrm_core_steps_8_no_evidence",
        )

    def test_build_report_accepts_full_with_ablation_drops(self):
        module = load_module()
        rows = []
        for mode, hits in {
            module.FULL_MODE: [True, True],
            module.DONOR_MODE: [True, False],
            module.CORE_OFF_MODE: [True, False],
            module.PRIMITIVE_OFF_MODE: [False, False],
            module.SOURCE_SLOT_OFF_MODE: [False, False],
            module.SOURCE_BINDER_OFF_MODE: [False, False],
            module.VOCAB_RENDERER_OFF_MODE: [False, False],
        }.items():
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "id": f"{mode}-{index}",
                        "mode": mode,
                        "hit": hit,
                        "exact_match": hit,
                    }
                )

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.75,
            min_donor_margin=0.25,
            min_core_off_margin=0.25,
            min_primitive_drop=0.25,
            min_source_slot_drop=0.25,
            min_source_binder_drop=0.25,
            min_vocab_renderer_drop=0.25,
        )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["decision"], "accepted_l4_candidate")
        self.assertEqual(report["decisive_metrics"]["full_generation_accuracy"], 1.0)

    def test_output_is_complete_requires_expected_row_count(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"id": str(index), "hit": True})
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(module.output_is_complete(path, expected_rows=2))
            self.assertFalse(module.output_is_complete(path, expected_rows=3))
            self.assertFalse(
                module.output_is_complete(Path(tmp) / "missing.jsonl", expected_rows=1)
            )

    def test_compact_stdout_report_omits_large_command_lists(self):
        module = load_module()

        compact = module.compact_stdout_report(
            {
                "decision": "accepted_l4_candidate",
                "accepted": True,
                "decisive_metrics": {"full_generation_accuracy": 1.0},
                "generation_jsonl": "eval.jsonl",
                "report_path": "report.json",
                "commands": [{"command": ["very", "long"]}],
                "exit_codes": [{"exit_code": 0}],
            }
        )

        self.assertEqual(compact["decision"], "accepted_l4_candidate")
        self.assertEqual(compact["decisive_metrics"]["full_generation_accuracy"], 1.0)
        self.assertNotIn("commands", compact)
        self.assertNotIn("exit_codes", compact)


if __name__ == "__main__":
    unittest.main()
