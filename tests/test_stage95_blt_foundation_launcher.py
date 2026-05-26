from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "558_prepare_stage95_blt_foundation_byte_sample_dgx.sh"


class Stage95BLTFoundationLauncherTests(unittest.TestCase):
    def test_plan_documents_broad_non_agentic_blt_curriculum(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "CLEANED_DATA_PATH": "/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515",
                "WORK_DIR": "/tmp/test_stage95_blt_foundation",
            }
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Stage95 BLT foundation byte sample", result.stdout)
        self.assertIn("general language", result.stdout)
        self.assertIn("reasoning", result.stdout)
        self.assertIn("math", result.stdout)
        self.assertIn("multilingual", result.stdout)
        self.assertIn("memory/context", result.stdout)
        self.assertIn("agentic/tool traces are later", result.stdout)
        self.assertIn("SOURCE_BUCKET_QUOTAS", result.stdout)
        self.assertIn("synthetic math is kept as a small spice", result.stdout)
        self.assertIn("--source-globs", result.stdout)
        self.assertIn("data_clustered/openmathinstruct2/*.parquet", result.stdout)
        self.assertIn("data_clustered/flan/*.parquet", result.stdout)
        self.assertIn("scripts/555_prepare_byte_prefixlm_sample.py", result.stdout)
        self.assertIn("SELECTION_MODE", result.stdout)
        self.assertIn("UTILITY_SCORE_JSONL", result.stdout)
        self.assertIn("OPUS projected-utility", result.stdout)
        self.assertIn("Generalization Dynamics", result.stdout)
        self.assertIn("generalization_dynamics_lite_probe.jsonl", result.stdout)
        self.assertIn("OPUS_PARAM_NAME_REGEX", result.stdout)
        self.assertIn("language-body selection", result.stdout)
        self.assertIn("score-opus", result.stdout)
        self.assertIn("scripts/614_score_opus_projected_utility.py", SCRIPT.read_text(encoding="utf-8"))

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_opus_proxy_validation_accepts_multiple_proxy_files(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("validate_opus_proxy_jsonl", text)
        self.assertIn('for proxy in ${OPUS_PROXY_JSONL//,/ }', text)
        self.assertIn("missing OPUS_PROXY_JSONL entry", text)

    def test_opus_defaults_are_operational_for_overnight_automation(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('OPUS_CANDIDATE_MAX_ROWS="${OPUS_CANDIDATE_MAX_ROWS:-256}"', text)
        self.assertIn('OPUS_PROJECTION_DIM="${OPUS_PROJECTION_DIM:-2048}"', text)
        self.assertIn("clean_decoder|hnet_byte_speaker", text)
        self.assertIn("OPUS_PARAM_NAME_REGEX", text)


if __name__ == "__main__":
    unittest.main()
