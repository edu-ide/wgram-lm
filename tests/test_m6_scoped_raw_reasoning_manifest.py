from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/376_build_m6_scoped_raw_reasoning_manifest.py")
    spec = importlib.util.spec_from_file_location("m6_scoped_raw_reasoning_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _qtrm_report() -> dict[str, object]:
    return {
        "accepted": True,
        "decision": "accepted_l5_multifamily",
        "target_level": "L4 QTRM-native mixed text reasoning scaffold",
        "task_families": ["modchain", "revchain", "checksum"],
        "train": {
            "eval_think_steps": 4,
            "program_len": 4,
            "modulus": 32,
            "eval_cases": 768,
            "eval_seed": 9337,
        },
        "eval_metrics": {
            "think0": {"generation_exact": 0.02},
            "think4": {
                "generation_exact": 0.61,
                "cases": 768,
                "by_family": {
                    "modchain": {"generation_exact": 0.47},
                    "revchain": {"generation_exact": 0.42},
                    "checksum": {"generation_exact": 0.94},
                },
            },
            "state_reset": {"generation_exact": 0.03},
            "op_zero": {"generation_exact": 0.04},
            "thinking_block_off": {"generation_exact": 0.02},
        },
    }


class M6ScopedRawReasoningManifestTests(unittest.TestCase):
    def test_rejects_without_qwen36_baseline_even_if_qtrm_is_strong(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            qtrm = Path(tmp) / "qtrm.json"
            qtrm.write_text(json.dumps(_qtrm_report()), encoding="utf-8")
            args = module.build_arg_parser().parse_args(["--qtrm-report", str(qtrm)])

            manifest = module.build_manifest(args)

        self.assertFalse(manifest["accepted"])
        self.assertIn("qwen36_baseline_present", manifest["reject_reasons"])
        self.assertEqual(
            manifest["best_qtrm_native"]["full_generation_exact"],
            0.61,
        )

    def test_accepts_when_qtrm_beats_matched_qwen36_baseline(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            qtrm = Path(tmp) / "qtrm.json"
            qwen = Path(tmp) / "qwen.json"
            qtrm.write_text(json.dumps(_qtrm_report()), encoding="utf-8")
            qwen.write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3.6-27B",
                        "suite_id": module.DEFAULT_SUITE_ID,
                        "prompt_protocol": module.DEFAULT_PROMPT_PROTOCOL,
                        "score": 0.52,
                        "cases": 768,
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                ["--qtrm-report", str(qtrm), "--qwen36-baseline-report", str(qwen)]
            )

            manifest = module.build_manifest(args)

        self.assertTrue(manifest["accepted"])
        self.assertTrue(manifest["acceptance_checks"]["qtrm_beats_qwen36_baseline"])

    def test_rejects_suite_mismatch(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            qtrm = Path(tmp) / "qtrm.json"
            qwen = Path(tmp) / "qwen.json"
            qtrm.write_text(json.dumps(_qtrm_report()), encoding="utf-8")
            qwen.write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3.6-27B",
                        "suite_id": "different-suite",
                        "prompt_protocol": module.DEFAULT_PROMPT_PROTOCOL,
                        "score": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                ["--qtrm-report", str(qtrm), "--qwen36-baseline-report", str(qwen)]
            )

            manifest = module.build_manifest(args)

        self.assertFalse(manifest["accepted"])
        self.assertIn("qwen36_suite_id_matches", manifest["reject_reasons"])

    def test_zero_score_baseline_is_still_a_measured_score(self):
        module = _load_script()
        baseline = module.qwen36_baseline_summary(
            {
                "model": "Qwen/Qwen3.6-27B",
                "suite_id": module.DEFAULT_SUITE_ID,
                "prompt_protocol": module.DEFAULT_PROMPT_PROTOCOL,
                "score": 0.0,
            },
            path="baseline.json",
        )

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline["score"], 0.0)

    def test_markdown_includes_reject_reasons(self):
        module = _load_script()
        manifest = {
            "decision": "rejected",
            "accepted": False,
            "suite_id": module.DEFAULT_SUITE_ID,
            "best_qtrm_native": {"full_generation_exact": 0.61, "cases": 768},
            "qwen36_baseline": {},
            "acceptance_checks": {"qwen36_baseline_present": False},
            "reject_reasons": ["qwen36_baseline_present"],
            "limitations": ["M6 cannot be accepted from a QTRM report alone."],
        }

        markdown = module.render_markdown(manifest)

        self.assertIn("M6 Scoped Raw-Reasoning Manifest", markdown)
        self.assertIn("qwen36_baseline_present", markdown)


if __name__ == "__main__":
    unittest.main()
