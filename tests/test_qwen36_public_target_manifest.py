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

    def test_manifest_rejects_without_native_artifact(self):
        module = _load_script()
        args = module.build_arg_parser().parse_args([])

        manifest = module.build_manifest(args)

        self.assertFalse(manifest["accepted"])
        self.assertFalse(manifest["acceptance_checks"]["qtrm_artifacts_present"])

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


if __name__ == "__main__":
    unittest.main()
