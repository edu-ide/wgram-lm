from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/375_build_qwen36_public_target_manifest.py")
    spec = importlib.util.spec_from_file_location("qwen36_public_target_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Qwen36PublicTargetManifestTests(unittest.TestCase):
    def test_manifest_accepts_public_targets_and_native_artifact(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            native = Path(tmp) / "native.json"
            native.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "accepted": True,
                        "decision": "accepted_qtrm_native_language_bootstrap",
                    }
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--native-report",
                    str(native),
                    "--qtrm-checkpoint",
                    "last.pt",
                ]
            )

            manifest = module.build_manifest(args)

        self.assertTrue(manifest["accepted"])
        self.assertFalse(manifest["direct_qwen36_rerun_required"])
        self.assertIn("gpqa_diamond", manifest["benchmark_map"])
        self.assertEqual(manifest["benchmark_map"]["aime_2026"]["qwen36_27b_target"], 94.1)
        ladder_ids = {row["id"] for row in manifest["official_agent_benchmark_ladder"]}
        self.assertIn("swe_bench_verified", ladder_ids)
        self.assertIn("terminal_bench_2_0", ladder_ids)
        self.assertIn("bfcl_v4", ladder_ids)
        self.assertIn("tau_bench", ladder_ids)
        self.assertIn("gaia", ladder_ids)
        self.assertFalse(manifest["agent_recognition_claim"]["accepted"])
        self.assertFalse(manifest["acceptance_checks"]["official_agent_claim_ready"])
        self.assertIn("tool_calling", manifest["agent_recognition_claim"]["missing_categories"])

    def test_manifest_rejects_without_native_artifact(self):
        module = _load_script()
        args = module.build_arg_parser().parse_args([])

        manifest = module.build_manifest(args)

        self.assertFalse(manifest["accepted"])
        self.assertFalse(manifest["acceptance_checks"]["qtrm_artifacts_present"])

    def test_agent_recognition_claim_requires_three_official_categories(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            native = Path(tmp) / "native.json"
            native.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "accepted": True,
                        "decision": "accepted_qtrm_native_language_bootstrap",
                    }
                ),
                encoding="utf-8",
            )
            reports = []
            for name, benchmark_id, qtrm_score, target in [
                ("swe.json", "swe_bench_verified", 78.0, 77.2),
                ("bfcl.json", "bfcl_v4", 71.0, 70.0),
                ("tau.json", "tau_bench", 62.0, 60.0),
            ]:
                path = Path(tmp) / name
                path.write_text(
                    json.dumps(
                        {
                            "accepted": True,
                            "benchmark_id": benchmark_id,
                            "benchmark_name": benchmark_id,
                            "official_harness": True,
                            "qtrm_score": qtrm_score,
                            "qwen36_target": target,
                        }
                    ),
                    encoding="utf-8",
                )
                reports.append(path)
            args = module.build_arg_parser().parse_args(
                [
                    "--native-report",
                    str(native),
                    "--agent-output",
                    str(reports[0]),
                    "--agent-output",
                    str(reports[1]),
                    "--agent-output",
                    str(reports[2]),
                ]
            )

            manifest = module.build_manifest(args)

        self.assertTrue(manifest["accepted"])
        self.assertTrue(manifest["agent_recognition_claim"]["accepted"])
        self.assertTrue(manifest["acceptance_checks"]["official_agent_claim_ready"])
        self.assertEqual(manifest["agent_recognition_claim"]["missing_categories"], [])
        self.assertEqual(
            set(manifest["agent_recognition_claim"]["qwen36_beating_benchmarks"]),
            {"bfcl_v4", "swe_bench_verified", "tau_bench"},
        )

    def test_agent_recognition_claim_rejects_unofficial_single_family_report(self):
        module = _load_script()
        artifact = {
            "accepted": True,
            "benchmark_id": "swe_bench_verified",
            "official_harness": False,
            "qtrm_score": 90.0,
            "qwen36_target": 77.2,
        }

        claim = module.agent_recognition_claim([artifact])

        self.assertFalse(claim["accepted"])
        self.assertIn("coding_or_terminal", claim["missing_categories"])
        self.assertIn("tool_calling", claim["missing_categories"])
        self.assertIn("long_horizon_workflow", claim["missing_categories"])

    def test_markdown_mentions_optional_direct_rerun(self):
        module = _load_script()
        manifest = {
            "decision": "accepted_qwen36_public_target_manifest",
            "accepted": True,
            "direct_qwen36_rerun_required": False,
            "qwen36": {
                "model": "Qwen/Qwen3.6-27B",
                "source_url": "https://huggingface.co/Qwen/Qwen3.6-27B",
            },
            "benchmark_map": {
                "gpqa_diamond": {
                    "display_name": "GPQA Diamond",
                    "qwen36_27b_target": 87.8,
                    "scorer": "official GPQA Diamond exact-match scorer",
                }
            },
            "official_agent_benchmark_ladder": [
                {
                    "id": "bfcl_v4",
                    "display_name": "Berkeley Function Calling Leaderboard V4",
                    "recognition_role": "tool-call correctness",
                    "status": "official_agent_benchmark_no_qwen36_target_in_manifest",
                }
            ],
            "agent_recognition_claim": {
                "status": "not_ready",
                "accepted": False,
                "missing_categories": ["tool_calling"],
            },
            "agent_benchmark_artifacts": [
                {
                    "benchmark_id": "bfcl_v4",
                    "official_harness": False,
                    "accepted": False,
                    "qtrm_score": None,
                    "qwen36_target": None,
                    "score_delta": None,
                }
            ],
            "qtrm_native": {
                "artifacts": [
                    {
                        "kind": "native_language_report",
                        "accepted": True,
                        "path": "report.json",
                    }
                ]
            },
            "limitations": ["Direct Qwen3.6 execution is optional for public target mode."],
        }

        markdown = module.render_markdown(manifest)

        self.assertIn("Direct rerun required", markdown)
        self.assertIn("GPQA Diamond", markdown)
        self.assertIn("Official Agent Benchmark Ladder", markdown)
        self.assertIn("Berkeley Function Calling Leaderboard V4", markdown)
        self.assertIn("Agent Recognition Claim", markdown)


if __name__ == "__main__":
    unittest.main()
