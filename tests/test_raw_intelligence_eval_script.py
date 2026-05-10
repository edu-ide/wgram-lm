from pathlib import Path
import importlib.util
import unittest


def _load_eval_module():
    path = Path("scripts/192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("raw_intelligence_eval_192", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RawIntelligenceEvalScriptTest(unittest.TestCase):
    def test_uses_recursive_checkpoint_loader(self) -> None:
        script = Path("scripts/192_eval_raw_intelligence.py").read_text(encoding="utf-8")

        self.assertIn("from qtrm_mm.training.train import load_initial_checkpoint", script)
        self.assertIn("load_initial_checkpoint(model, args.checkpoint", script)
        self.assertNotIn("model.load_state_dict(state.get(\"model\", state), strict=False)", script)

    def test_primitive_role_value_ablation_mode_is_forwarded(self) -> None:
        script = Path("scripts/192_eval_raw_intelligence.py").read_text(encoding="utf-8")

        self.assertIn(
            "qtrm_core_steps_(\\d+)_primitive_role_value_off_no_evidence",
            script,
        )
        self.assertIn('"disable_core_primitive_role_value_executor": True', script)
        self.assertGreaterEqual(
            script.count("disable_core_primitive_role_value_executor=bool("),
            4,
        )

    def test_role_value_vocab_renderer_ablation_mode_is_forwarded(self) -> None:
        script = Path("scripts/192_eval_raw_intelligence.py").read_text(encoding="utf-8")

        self.assertIn(
            "qtrm_core_steps_(\\d+)_core_role_value_vocab_renderer_off_no_evidence",
            script,
        )
        self.assertIn('"disable_core_role_value_vocab_renderer": True', script)
        self.assertGreaterEqual(
            script.count("disable_core_role_value_vocab_renderer=bool("),
            4,
        )

    def test_typed_value_answer_bridge_ablation_mode_is_forwarded(self) -> None:
        script = Path("scripts/192_eval_raw_intelligence.py").read_text(encoding="utf-8")

        self.assertIn(
            "qtrm_core_steps_(\\d+)_typed_value_answer_bridge_off_no_evidence",
            script,
        )
        self.assertIn(
            '"disable_typed_algorithmic_value_state_answer_bridge": True',
            script,
        )
        self.assertGreaterEqual(
            script.count(
                "disable_typed_algorithmic_value_state_answer_bridge=bool("
            ),
            4,
        )

    def test_source_pointer_token_numeric_features_are_forwarded(self) -> None:
        script = Path("scripts/192_eval_raw_intelligence.py").read_text(encoding="utf-8")

        self.assertIn("--token-numeric-value-features", script)
        self.assertIn("--core-source-position-binder", script)
        self.assertIn("cfg.model.token_numeric_value_embedding_enabled = True", script)
        self.assertIn("cfg.model.core_source_position_binder_enabled = True", script)
        self.assertGreaterEqual(script.count("token_numeric_value_ids="), 4)

    def test_source_copy_generation_scoring_rejects_loose_contains_match(self) -> None:
        module = _load_eval_module()

        record = module.score_case_record(
            {
                "id": "case",
                "category": "source_copy_lexicalization",
                "task_family": "source_copy_lexicalization",
                "answer_aliases": ["40,32,44"],
            },
            mode="qtrm_core_steps_8_no_evidence",
            completion="55, 40, 32, 44",
            runtime={"memoryos_used": False, "retrieval_used": False},
            generated_tokens=8,
        )

        self.assertFalse(record["hit"])
        self.assertTrue(record["normalized_contains"])
        self.assertFalse(record["normalized_exact"])
        self.assertIn("strict_exact_miss", record["audit_reasons"])

    def test_general_generation_scoring_still_allows_contains_match(self) -> None:
        module = _load_eval_module()

        record = module.score_case_record(
            {
                "id": "case",
                "category": "general_qa",
                "answer_aliases": ["Paris"],
            },
            mode="donor_only_no_evidence",
            completion="The answer is Paris.",
            runtime={"memoryos_used": False, "retrieval_used": False},
            generated_tokens=5,
        )

        self.assertTrue(record["hit"])
        self.assertEqual(record["match_type"], "normalized_contains")


if __name__ == "__main__":
    unittest.main()
