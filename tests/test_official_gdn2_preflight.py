from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "613_preflight_official_gdn2_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("official_gdn2_preflight_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OfficialGDN2PreflightTests(unittest.TestCase):
    def test_preflight_requires_explicit_matching_ptxas(self) -> None:
        module = load_module()

        result = module.run_preflight(
            required_ptxas="",
            triton_ptxas="",
            checkpoint="",
            official_smoke="none",
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("missing required ptxas contract", "\n".join(result["blockers"]))
        self.assertIn("성적표가 오염", result["plain_language"])

    def test_preflight_rejects_legacy_fallback_checkpoint(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ptxas = tmp_path / "ptxas"
            ptxas.write_text("#!/bin/sh\n", encoding="utf-8")
            ptxas.chmod(0o755)
            checkpoint = tmp_path / "legacy.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "global_core.encode.layers.0.mixer.impl.in_proj.weight": torch.zeros(1),
                    },
                    "args": {"decoder_latent_mode": "one_body"},
                    "model": {
                        "global_core": {
                            "delta_runtime": {
                                "actual_delta_runtime": "official_runtime",
                                "delta_runtime_fallback_active_count": 0,
                                "delta_runtime_torch_direct_count": 0,
                            }
                        }
                    },
                },
                checkpoint,
            )

            result = module.run_preflight(
                required_ptxas=str(ptxas),
                triton_ptxas=str(ptxas),
                checkpoint=str(checkpoint),
                expect_decoder_latent_mode="one_body",
                official_smoke="none",
            )

        self.assertEqual(result["status"], "fail")
        self.assertIn("legacy fallback", "\n".join(result["blockers"]))

    def test_preflight_accepts_clean_official_checkpoint(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ptxas = tmp_path / "ptxas"
            ptxas.write_text("#!/bin/sh\n", encoding="utf-8")
            ptxas.chmod(0o755)
            checkpoint = tmp_path / "clean.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "global_core.encode.layers.0.mixer.impl.q_proj.weight": torch.zeros(1),
                    },
                    "args": {"decoder_latent_mode": "one_body"},
                    "model": {
                        "global_core": {
                            "delta_runtime": {
                                "actual_delta_runtime": "official_runtime",
                                "delta_runtime_fallback_active_count": 0,
                                "delta_runtime_torch_direct_count": 0,
                            }
                        }
                    },
                },
                checkpoint,
            )

            result = module.run_preflight(
                required_ptxas=str(ptxas),
                triton_ptxas=str(ptxas),
                checkpoint=str(checkpoint),
                expect_decoder_latent_mode="one_body",
                official_smoke="none",
            )

        self.assertEqual(result["status"], "pass")
        self.assertIn("오염되지 않았다", result["plain_language"])
        self.assertEqual(result["blockers"], [])

    def test_preflight_blocks_failed_official_smoke(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ptxas = tmp_path / "ptxas"
            ptxas.write_text("#!/bin/sh\n", encoding="utf-8")
            ptxas.chmod(0o755)

            original = module._run_official_gdn2_smoke
            module._run_official_gdn2_smoke = lambda mode: {
                "mode": mode,
                "status": "fail",
                "error": "kernel compile failed",
            }
            try:
                result = module.run_preflight(
                    required_ptxas=str(ptxas),
                    triton_ptxas=str(ptxas),
                    checkpoint="",
                    official_smoke="forward_auto",
                )
            finally:
                module._run_official_gdn2_smoke = original

        self.assertEqual(result["status"], "fail")
        self.assertIn("official GDN2 smoke failed", "\n".join(result["blockers"]))
        self.assertEqual(result["evidence"]["official_gdn2_smoke"]["mode"], "forward_auto")


if __name__ == "__main__":
    unittest.main()
