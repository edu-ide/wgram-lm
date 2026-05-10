import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "240_select_qtrm_checkpoint_by_gate.py"
    spec = importlib.util.spec_from_file_location("select_qtrm_checkpoint_by_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SelectQTRMCheckpointByGateTests(unittest.TestCase):
    def test_parse_candidate_spec_requires_name_checkpoint_and_eval(self):
        module = load_script()

        candidate = module.parse_candidate_spec(
            "name=s080,checkpoint=ckpt.pt,eval=eval.jsonl,action=action.json"
        )

        self.assertEqual(candidate["name"], "s080")
        self.assertEqual(candidate["checkpoint"], "ckpt.pt")
        self.assertEqual(candidate["eval"], "eval.jsonl")
        self.assertEqual(candidate["action"], "action.json")

    def test_selects_highest_lm_hit_candidate_that_preserves_action_code(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s80_eval = root / "s80.jsonl"
            s240_eval = root / "s240.jsonl"
            s80_action = root / "s80_action.json"
            s240_action = root / "s240_action.json"

            s80_rows = [
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": False},
            ]
            s240_rows = [
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": False},
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": False},
            ]
            s80_eval.write_text("\n".join(json.dumps(row) for row in s80_rows) + "\n")
            s240_eval.write_text("\n".join(json.dumps(row) for row in s240_rows) + "\n")
            s80_action.write_text(
                json.dumps({"summary": {"exact_rows": 32, "rows": 32}}),
                encoding="utf-8",
            )
            s240_action.write_text(
                json.dumps({"summary": {"exact_rows": 32, "rows": 32}}),
                encoding="utf-8",
            )

            report = module.select_checkpoint(
                [
                    {
                        "name": "s080",
                        "checkpoint": "s080.pt",
                        "eval": str(s80_eval),
                        "action": str(s80_action),
                    },
                    {
                        "name": "s240",
                        "checkpoint": "s240.pt",
                        "eval": str(s240_eval),
                        "action": str(s240_action),
                    },
                ],
                mode="qtrm_core_steps_8_no_evidence",
                min_hits=1,
                min_action_exact=32,
            )

        self.assertEqual(report["selected"]["name"], "s080")
        self.assertTrue(report["selected"]["accepted"])
        self.assertEqual(report["selected"]["lm_hits"], 1)
        self.assertFalse(report["candidates"][1]["accepted"])

    def test_rejects_candidate_with_lm_hits_if_action_code_regresses(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval.jsonl"
            action_path = root / "action.json"
            eval_path.write_text(
                json.dumps({"mode": "qtrm_core_steps_8_no_evidence", "hit": True}) + "\n",
                encoding="utf-8",
            )
            action_path.write_text(
                json.dumps({"summary": {"exact_rows": 31, "rows": 32}}),
                encoding="utf-8",
            )

            report = module.select_checkpoint(
                [
                    {
                        "name": "bad_action",
                        "checkpoint": "bad.pt",
                        "eval": str(eval_path),
                        "action": str(action_path),
                    }
                ],
                mode="qtrm_core_steps_8_no_evidence",
                min_hits=1,
                min_action_exact=32,
            )

        self.assertIsNone(report["selected"])
        self.assertFalse(report["candidates"][0]["accepted"])
        self.assertIn("action_exact_below_min", report["candidates"][0]["reject_reasons"])

    def test_rejects_candidate_without_required_ablation_drop(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval.jsonl"
            action_path = root / "action.json"
            rows = [
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
                {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
                {
                    "mode": "qtrm_core_steps_8_answer_selective_context_off_no_evidence",
                    "hit": True,
                },
                {
                    "mode": "qtrm_core_steps_8_answer_selective_context_off_no_evidence",
                    "hit": True,
                },
            ]
            eval_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            action_path.write_text(
                json.dumps({"summary": {"exact_rows": 32, "rows": 32}}),
                encoding="utf-8",
            )

            report = module.select_checkpoint(
                [
                    {
                        "name": "router",
                        "checkpoint": "router.pt",
                        "eval": str(eval_path),
                        "action": str(action_path),
                        "ablation_mode": (
                            "qtrm_core_steps_8_answer_selective_context_off_no_evidence"
                        ),
                    }
                ],
                mode="qtrm_core_steps_8_no_evidence",
                min_hits=1,
                min_action_exact=32,
                min_ablation_drop=1,
            )

        self.assertIsNone(report["selected"])
        self.assertFalse(report["candidates"][0]["accepted"])
        self.assertEqual(report["candidates"][0]["ablation_hits"], 2)
        self.assertEqual(report["candidates"][0]["ablation_drop"], 0)
        self.assertIn("ablation_drop_below_min", report["candidates"][0]["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
