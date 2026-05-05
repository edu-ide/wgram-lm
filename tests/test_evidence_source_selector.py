from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _load_script():
    path = Path("scripts/165_train_evidence_source_selector.py")
    spec = importlib.util.spec_from_file_location("evidence_source_selector", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(record_text: str = "Signed note: the vault passphrase is amber-field.") -> dict:
    return {
        "id": "vault",
        "category": "authority_conflict_synth",
        "question": "Which passphrase opens the vault?",
        "answer_aliases": ["amber-field", "amber field"],
        "evidence": [{"source": "signed_vault.md", "text": record_text}],
        "distractors": [{"source": "anonymous_vault.md", "text": "Anonymous note: the vault passphrase is cedar-gate."}],
    }


class EvidenceSourceSelectorTests(unittest.TestCase):
    def test_build_examples_labels_only_answer_bearing_sources(self) -> None:
        module = _load_script()

        examples = module.build_examples([_case()])

        labels = {
            example.record["source"]: example.label_answer_source
            for example in examples
        }
        self.assertEqual(labels["signed_vault.md"], 1)
        self.assertEqual(labels["anonymous_vault.md"], 0)

    def test_build_examples_labels_negative_cases_as_no_source(self) -> None:
        module = _load_script()
        case = {
            "id": "missing",
            "category": "negative_missing_synth",
            "question": "Which passphrase opens the east vault?",
            "answer_aliases": ["UNKNOWN"],
            "evidence": [{"source": "west_vault.md", "text": "The west vault passphrase is amber-field."}],
            "distractors": [{"source": "storage.md", "text": "The east storage marker is VX-913."}],
        }

        examples = module.build_examples([case])

        self.assertEqual([example.label_answer_source for example in examples], [0, 0])

    def test_selector_learns_synthetic_authority_signal(self) -> None:
        module = _load_script()
        cases = [
            _case("Signed supervisor note: the vault passphrase is amber-field.")
            for _ in range(16)
        ]
        examples = module.build_examples(cases)

        model = module.train_selector(
            examples,
            epochs=120,
            lr=3e-3,
            hidden_dim=16,
            dropout=0.0,
            seed=3,
        )
        probs = module.source_probabilities(model, examples)
        threshold, metrics = module.select_threshold(examples, probs)

        self.assertGreaterEqual(threshold, 0.05)
        self.assertLessEqual(threshold, 0.95)
        self.assertEqual(metrics["case_success_rate"], 1.0)
        self.assertEqual(metrics["fn"], 0)

    def test_main_writes_checkpoint_and_report(self) -> None:
        import tempfile

        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = [_case() for _ in range(8)]
            cases_path = tmp_path / "cases.jsonl"
            with cases_path.open("w", encoding="utf-8") as f:
                for idx, case in enumerate(cases):
                    row = dict(case)
                    row["id"] = f"vault-{idx}"
                    f.write(json.dumps(row) + "\n")

            out_pt = tmp_path / "selector.pt"
            out_json = tmp_path / "selector.json"
            out_md = tmp_path / "selector.md"

            module.main(
                [
                    "--train-cases-jsonl",
                    str(cases_path),
                    "--out-pt",
                    str(out_pt),
                    "--out-json",
                    str(out_json),
                    "--markdown-out",
                    str(out_md),
                    "--epochs",
                    "80",
                    "--hidden-dim",
                    "16",
                ]
            )

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(out_pt.exists())
            self.assertTrue(out_md.exists())
            self.assertGreater(report["train_count"], 0)
            self.assertIn("eval_learned", report)


if __name__ == "__main__":
    unittest.main()
