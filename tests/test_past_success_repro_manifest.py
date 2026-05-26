from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "563_build_past_success_repro_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("past_success_repro_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(
    path: Path,
    *,
    checkpoint: Path,
    extractor_checkpoint: Path,
    include_code_commit: bool = False,
    include_eval_rows: bool = False,
) -> None:
    metadata = {
        "args": {
            "checkpoint": str(checkpoint),
            "extractor_checkpoint": str(extractor_checkpoint),
            "seed": 103,
            "eval_seed": 10042,
            "eval_count": 128,
            "samples": 64,
            "candidate_topk_per_sample": 3,
            "eval_depths": [4, 6, 8, 10, 12, 14],
            "stochastic_high_level_eval": True,
            "stochastic_transition_mode": "true_gram",
            "stochastic_posterior_guidance": True,
        }
    }
    if include_code_commit:
        metadata["code_commit"] = "abc123"
    if include_eval_rows:
        (path.parent / "materialized_eval_rows.jsonl").write_text('{"case_id": 1}\n', encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "history": [
                    {
                        "eval": {
                            "mean_selected_accuracy_oracle_depth": 0.93359375,
                            "mean_oracle_accuracy": 0.9401041666666666,
                            "mean_packed_register_answer_accuracy_oracle_depth": 0.9934895833333334,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class PastSuccessReproManifestTests(unittest.TestCase):
    def test_stage_with_present_artifacts_is_replayable_but_not_sealed(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "generator.pt"
            extractor = root / "extractor.pt"
            summary = root / "summary.json"
            checkpoint.write_bytes(b"generator")
            extractor.write_bytes(b"extractor")
            write_summary(summary, checkpoint=checkpoint, extractor_checkpoint=extractor)

            row = module.build_stage_repro_row(summary, label="Stage58B")

        self.assertEqual(row["label"], "Stage58B")
        self.assertEqual(row["reproducibility_status"], "replayable_not_sealed")
        self.assertTrue(row["can_replay"])
        self.assertFalse(row["fully_sealed"])
        self.assertIn("code commit/hash", row["blocking_gaps"][0])
        self.assertIn("materialized eval JSONL rows", " ".join(row["blocking_gaps"]))
        self.assertEqual(row["metrics"]["selected_accuracy"], 0.93359375)
        self.assertEqual(row["settings"]["eval_seed"], 10042)
        self.assertEqual(row["settings"]["candidate_topk_per_sample"], 3)
        artifact_names = [artifact["name"] for artifact in row["artifacts"]]
        self.assertEqual(artifact_names, ["summary", "generator_checkpoint", "extractor_checkpoint"])
        for artifact in row["artifacts"]:
            self.assertTrue(artifact["exists"])
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")

    def test_missing_checkpoint_makes_row_not_replayable(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "missing.pt"
            extractor = root / "extractor.pt"
            summary = root / "summary.json"
            extractor.write_bytes(b"extractor")
            write_summary(summary, checkpoint=checkpoint, extractor_checkpoint=extractor)

            row = module.build_stage_repro_row(summary, label="Stage56")

        self.assertEqual(row["reproducibility_status"], "not_replayable")
        self.assertFalse(row["can_replay"])
        self.assertIn("missing required artifact", " ".join(row["blocking_gaps"]))

    def test_manifest_and_markdown_summarize_plain_korean_verdict(self) -> None:
        module = load_module()
        row = {
            "label": "Stage58B",
            "reproducibility_status": "replayable_not_sealed",
            "can_replay": True,
            "fully_sealed": False,
            "metrics": {"selected_accuracy": 0.93359375},
            "settings": {"eval_seed": 10042},
            "artifacts": [],
            "blocking_gaps": ["no immutable code commit/hash recorded"],
        }

        manifest = module.build_reproducibility_manifest([row])
        markdown = module.render_markdown(manifest)

        self.assertEqual(manifest["overall_status"], "replayable_not_sealed")
        self.assertTrue(manifest["can_replay_any"])
        self.assertFalse(manifest["all_fully_sealed"])
        self.assertIn("다시 돌려볼 수는 있지만", markdown)
        self.assertIn("Stage58B", markdown)


if __name__ == "__main__":
    unittest.main()
