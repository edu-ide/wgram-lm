from __future__ import annotations

import unittest


class ArchitectureComponentRegistryTests(unittest.TestCase):
    def test_promoted_components_are_distinct_from_diagnostic_paths(self) -> None:
        from wgram_lm.architecture.component_registry import (
            ComponentStatus,
            assert_promoted_component,
            get_component_record,
        )

        one_body = get_component_record("one_body_contract")
        bridge = get_component_record("stage99_bridge_readback_selector")

        self.assertEqual(one_body.status, ComponentStatus.PROMOTED)
        self.assertEqual(bridge.status, ComponentStatus.DIAGNOSTIC)
        self.assertEqual(assert_promoted_component("one_body_contract"), one_body)
        with self.assertRaisesRegex(ValueError, "not promoted"):
            assert_promoted_component("stage99_bridge_readback_selector")

    def test_blt_full_model_is_src_scaffold_not_best_module(self) -> None:
        from wgram_lm.architecture.component_registry import (
            ComponentStatus,
            assert_promoted_component,
            get_component_record,
        )

        record = get_component_record("bltd_byte_latent_prefixlm")

        self.assertEqual(record.status, ComponentStatus.SCAFFOLD)
        self.assertIn("src/wgram_lm/models/blt_prefixlm.py", record.locations)
        self.assertIn("remains scaffold", record.note)
        with self.assertRaisesRegex(ValueError, "not promoted"):
            assert_promoted_component("bltd_byte_latent_prefixlm")

    def test_stage102_promotes_only_full_answer_path(self) -> None:
        from wgram_lm.architecture.component_registry import (
            ComponentStatus,
            assert_promoted_final_answer_path,
            assert_promoted_component,
            get_component_record,
        )

        final_path = get_component_record("stage102z_final_freeform_answer_path")
        prompt_frontend = get_component_record("stage102f_prompt_provenance_frontend")
        freeform_frontend = get_component_record("stage102g_freeform_provenance_frontend")

        self.assertEqual(final_path.status, ComponentStatus.PROMOTED)
        self.assertEqual(prompt_frontend.status, ComponentStatus.DIAGNOSTIC)
        self.assertEqual(freeform_frontend.status, ComponentStatus.DIAGNOSTIC)
        self.assertTrue(final_path.full_answer_path)
        self.assertFalse(prompt_frontend.full_answer_path)
        self.assertFalse(freeform_frontend.full_answer_path)
        self.assertIn("full causal answer path", final_path.note)
        self.assertEqual(assert_promoted_component("stage102z_final_freeform_answer_path"), final_path)
        self.assertEqual(assert_promoted_final_answer_path("stage102z_final_freeform_answer_path"), final_path)
        with self.assertRaisesRegex(ValueError, "not promoted"):
            assert_promoted_component("stage102f_prompt_provenance_frontend")
        with self.assertRaisesRegex(ValueError, "not promoted"):
            assert_promoted_component("stage102g_freeform_provenance_frontend")
        with self.assertRaisesRegex(ValueError, "not a full answer path"):
            assert_promoted_final_answer_path("one_body_contract")


if __name__ == "__main__":
    unittest.main()
