from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GDN2RuntimeContractDocsTests(unittest.TestCase):
    def test_qtrm_mixer_docs_lock_official_gdn2_to_fail_fast(self) -> None:
        text = (ROOT / "docs/wiki/components/qtrm-mixer.md").read_text(encoding="utf-8")

        self.assertIn("official_gated_delta2 is fail-fast", text)
        self.assertIn("no Torch fallback", text)
        self.assertIn("no runtime fallback", text)
        self.assertIn("preflight automatically before partial training", text)
        self.assertIn("REQUIRED_TRITON_PTXAS_PATH", text)
        self.assertIn("TRITON_PTXAS_PATH", text)
        self.assertNotIn("fallback remains experimental", text)
        self.assertNotIn("can import, then fall back", text)
        self.assertNotIn("strict_backends: false` falls back", text)

    def test_training_diagnostics_docs_require_explicit_ptxas_contract(self) -> None:
        text = (ROOT / "docs/wiki/concepts/training-diagnostics.md").read_text(encoding="utf-8")

        self.assertIn("REQUIRED_TRITON_PTXAS_PATH", text)
        self.assertIn("TRITON_PTXAS_PATH", text)
        self.assertIn("missing required ptxas contract", text)
        self.assertIn("fail fast", text)
        self.assertNotIn("otherwise Triton may use", text)

    def test_one_body_ssot_declares_runtime_fallback_as_evidence_pollution(self) -> None:
        text = (ROOT / "docs/wiki/architecture/one-body-architecture-ssot.md").read_text(encoding="utf-8")

        self.assertIn("Runtime/Kernel Contract", text)
        self.assertIn("evidence pollution", text)
        self.assertIn("official GDN2", text)
        self.assertIn("ptxas", text)

    def test_active_decision_index_marks_official_gdn2_runtime_contract_active(self) -> None:
        text = (ROOT / "docs/wiki/decisions/0001-active-decision-index.md").read_text(encoding="utf-8")

        self.assertIn("Official GDN2 Runtime Contract", text)
        self.assertIn("2026-05-25-official-gdn2-runtime-contract.md", text)
        self.assertIn("Stage95B", text)
        self.assertIn("Stage95G/H", text)
        self.assertIn("preflight", text)


if __name__ == "__main__":
    unittest.main()
