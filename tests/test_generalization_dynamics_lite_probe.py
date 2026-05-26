from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "566_build_generalization_dynamics_probe.py"
EVAL = ROOT / "scripts" / "567_eval_blt_generalization_dynamics_probe.py"
DEPTH_PROBE = ROOT / "scripts" / "560_eval_blt_depth_residual_probe.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeneralizationDynamicsLiteProbeTests(unittest.TestCase):
    def test_builder_writes_core_anti_parrot_axes(self) -> None:
        module = load_module(BUILDER, "gd_lite_builder")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.jsonl"
            report = module.write_probe(out, module.default_cases())
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["probe_type"], "generalization_dynamics_lite")
        self.assertGreaterEqual(len(rows), 6)
        tasks = {row["task"] for row in rows}
        self.assertIn("successive_answer_icl", tasks)
        self.assertIn("intuitive_answer_zero_shot", tasks)
        self.assertIn("persona_multihop_icl", tasks)
        for row in rows:
            self.assertIn("intelligence_answer", row)
            self.assertIn("parrot_answer", row)

    def test_byte_choice_tensors_match_prefixlm_shift_contract(self) -> None:
        module = load_module(EVAL, "gd_lite_eval")

        input_ids, labels, attention_mask = module.build_choice_tensors(
            prompt="Q:",
            answer=" A",
            seq_len=8,
            byte_offset=2,
        )

        self.assertEqual(input_ids.shape, labels.shape)
        self.assertEqual(attention_mask.tolist()[0][:3], [1, 1, 1])
        self.assertEqual(labels.tolist()[0][0], module.IGNORE_LABEL_ID)
        self.assertEqual(labels.tolist()[0][1], 2 + ord(" "))
        self.assertEqual(labels.tolist()[0][2], 2 + ord("A"))

    def test_summary_accepts_only_positive_generalization_margins(self) -> None:
        module = load_module(EVAL, "gd_lite_eval_summary")

        accepted = module.summarize_rows(
            [
                {"task": "a", "normalized_margin": 0.1, "correct": True, "skipped_reason": None},
                {"task": "b", "normalized_margin": 0.2, "correct": True, "skipped_reason": None},
            ]
        )
        rejected = module.summarize_rows(
            [
                {"task": "a", "normalized_margin": 0.1, "correct": True, "skipped_reason": None},
                {"task": "b", "normalized_margin": -0.01, "correct": False, "skipped_reason": None},
            ]
        )

        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertLess(rejected["accuracy"], 1.0)

    def test_depth_probe_allows_diagnostic_only_missing_keys_for_old_checkpoints(self) -> None:
        module = load_module(DEPTH_PROBE, "gd_lite_depth_probe_loader")

        self.assertTrue(module.is_allowed_missing_checkpoint_key("answer_readback_gate_logit"))
        self.assertTrue(module.is_allowed_missing_checkpoint_key("answer_anchor_head.0.weight"))
        self.assertTrue(module.is_allowed_missing_checkpoint_key("answer_workspace_selector.1.bias"))
        self.assertTrue(module.is_allowed_missing_checkpoint_key("hierarchical_chunk_proj.weight"))
        self.assertFalse(module.is_allowed_missing_checkpoint_key("global_core.layers.0.weight"))


if __name__ == "__main__":
    unittest.main()
