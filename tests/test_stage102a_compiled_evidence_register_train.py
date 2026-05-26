from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "604_train_stage102a_compiled_evidence_register.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage102a_compiled_evidence_register_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage102ACompiledEvidenceRegisterTrainTests(unittest.TestCase):
    def test_source_id_to_index_is_strict(self) -> None:
        module = load_module()

        self.assertEqual(0, module.source_id_to_index("S1"))
        self.assertEqual(1, module.source_id_to_index("S2"))
        with self.assertRaises(ValueError):
            module.source_id_to_index("S3")

    def test_compiler_outputs_one_register_per_source(self) -> None:
        module = load_module()
        compiler = module.CompiledEvidenceRegister(d_model=6)

        register = compiler(["S1", "S2"], device=torch.device("cpu"))

        self.assertEqual((2, 6), tuple(register.shape))
        self.assertFalse(torch.allclose(register[0], register[1]))

    def test_row_side_source_ids_reads_original_and_counterfactual(self) -> None:
        module = load_module()
        row = {"original_source": "S1", "counterfactual_source": "S2"}

        self.assertEqual(["S1", "S2"], module.row_side_source_ids(row))


if __name__ == "__main__":
    unittest.main()
