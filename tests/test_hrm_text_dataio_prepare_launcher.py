from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "533_prepare_hrm_text_dataio_sample.sh"
STAGE93_SCRIPT = ROOT / "scripts" / "535_prepare_stage93_hrm_text_large_dataio.sh"
STAGE93_LAUNCHER = ROOT / "scripts" / "536_launch_stage93_dgx_continue_prefixlm.sh"


class HrmTextDataioPrepareLauncherTests(unittest.TestCase):
    def test_plan_documents_persistent_dataio_contract(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "CLEANED_DATA_PATH": "/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515",
                "TOKENIZED_OUT": "/tmp/test_hrm_text_tokenized",
                "SAMPLED_OUT": "/tmp/test_hrm_text_sampled",
                "SOURCE_FILES": "data/gsm8k_train.jsonl data/math_train.jsonl",
            }
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("HRM-Text Data-IO sample preparation", result.stdout)
        self.assertIn(
            "CLEANED_DATA_PATH=/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515",
            result.stdout,
        )
        self.assertIn("DATA_IO_DIR=", result.stdout)
        self.assertIn("TOKENIZER_PATH=", result.stdout)
        self.assertIn("SOURCE_FILES=data/gsm8k_train.jsonl data/math_train.jsonl", result.stdout)
        self.assertIn("actions: plan status tokenize sample all", result.stdout)

    def test_stage93_plan_documents_large_data_contract(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "CLEANED_DATA_PATH": "/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515",
                "WORK_DIR": "/tmp/test_stage93_hrm_text_dataio",
                "PYTHON": str(ROOT / ".venv" / "bin" / "python"),
            }
        )
        result = subprocess.run(
            ["bash", str(STAGE93_SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Stage93 HRM-Text large Data-IO preparation", result.stdout)
        self.assertIn(
            "CLEANED_DATA_PATH=/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515",
            result.stdout,
        )
        self.assertIn("actions: plan status link tokenize sample all launch", result.stdout)
        self.assertIn("Use hardlinks, not copies", result.stdout)
        self.assertIn("INCLUDE_FLAN=0", result.stdout)

    def test_stage93_full_curriculum_uses_capped_flan(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "PROFILE": "full_curriculum",
                "CLEANED_DATA_PATH": "/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515",
                "WORK_DIR": "/tmp/test_stage93_hrm_text_full_curriculum_dataio",
            }
        )
        result = subprocess.run(
            ["bash", str(STAGE93_SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROFILE=full_curriculum", result.stdout)
        self.assertIn("INCLUDE_FLAN=1", result.stdout)
        self.assertIn("FLAN_MAX_FILES=64", result.stdout)
        self.assertIn("Represent every major shelf", result.stdout)

    def test_stage93_multilingual_curriculum_targets_translation_flan(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "ROOT": str(ROOT),
                "PROFILE": "multilingual_curriculum",
                "CLEANED_DATA_PATH": "/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515",
                "WORK_DIR": "/tmp/test_stage93_hrm_text_multilingual_dataio",
            }
        )
        result = subprocess.run(
            ["bash", str(STAGE93_SCRIPT), "plan"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROFILE=multilingual_curriculum", result.stdout)
        self.assertIn("INCLUDE_FLAN=1", result.stdout)
        self.assertIn("FLAN_MAX_FILES=all", result.stdout)
        self.assertIn("FLAN_INCLUDE_REGEX=translate|translation|wmt", result.stdout)
        self.assertIn("make multilingual measurable", result.stdout)

    def test_stage93_continue_launcher_uses_resume_checkpoint(self) -> None:
        text = STAGE93_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("--resume", text)
        self.assertIn("last_model.pt", text)
        self.assertIn("sampled data is not ready", text)
        self.assertIn("another PrefixLM training process is already running", text)


if __name__ == "__main__":
    unittest.main()
