from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "557_train_blt_d_prefixlm_dataio.py"


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("train_blt_d_prefixlm_dataio_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeFallbackWrapper(nn.Module):
    def __init__(self, *, official: bool, fallback_active: bool) -> None:
        super().__init__()
        self.is_official_backend = bool(official)
        self.runtime_fallback = nn.Linear(2, 2)
        object.__setattr__(self, "_runtime_fallback_active", bool(fallback_active))


class TorchGatedDeltaMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class BLTDeltaRuntimeSummaryTests(unittest.TestCase):
    def test_reports_official_wrapper_runtime_fallback(self) -> None:
        trainer = load_trainer_module()
        model = nn.Sequential(RuntimeFallbackWrapper(official=False, fallback_active=True))

        summary = trainer.collect_delta_runtime_summary(model)

        self.assertEqual(summary["actual_delta_runtime"], "official_wrapper_runtime_fallback")
        self.assertEqual(summary["delta_runtime_wrapper_count"], 1)
        self.assertEqual(summary["delta_runtime_fallback_active_count"], 1)
        self.assertTrue(summary["delta_runtime_has_fallback"])

    def test_reports_direct_torch_delta(self) -> None:
        trainer = load_trainer_module()
        model = nn.Sequential(TorchGatedDeltaMixer())

        summary = trainer.collect_delta_runtime_summary(model)

        self.assertEqual(summary["actual_delta_runtime"], "torch_gated_delta")
        self.assertEqual(summary["delta_runtime_torch_direct_count"], 1)
        self.assertTrue(summary["delta_runtime_has_fallback"])

    def test_refreshes_model_summary(self) -> None:
        trainer = load_trainer_module()
        model_summary = {"global_core": {"delta_backend": "official_gated_delta2"}}

        trainer.refresh_model_runtime_summary(
            model_summary,
            nn.Sequential(RuntimeFallbackWrapper(official=True, fallback_active=False)),
        )

        self.assertEqual(
            model_summary["global_core"]["delta_runtime"]["actual_delta_runtime"],
            "official_runtime",
        )

    def test_resume_rejects_legacy_fallback_impl_weights(self) -> None:
        trainer = load_trainer_module()
        weight = object()
        source = {
            "global_core.encode.layers.0.mixer.impl.in_proj.weight": weight,
            "byte_embed.weight": object(),
        }
        target = {
            "global_core.encode.layers.0.mixer.impl.q_proj.weight": object(),
            "byte_embed.weight": object(),
        }

        with self.assertRaisesRegex(ValueError, "legacy fallback"):
            trainer.adapt_resume_state_dict_for_current_model(source, target)

    def test_resume_keeps_official_impl_weights(self) -> None:
        trainer = load_trainer_module()
        weight = object()
        source = {
            "global_core.encode.layers.0.mixer.impl.q_proj.weight": weight,
            "byte_embed.weight": object(),
        }
        target = {
            "global_core.encode.layers.0.mixer.impl.q_proj.weight": object(),
            "byte_embed.weight": object(),
        }

        adapted, summary = trainer.adapt_resume_state_dict_for_current_model(source, target)

        self.assertIs(adapted["global_core.encode.layers.0.mixer.impl.q_proj.weight"], weight)
        self.assertEqual(summary["legacy_delta_fallback_key_count"], 0)

    def test_triton_ptxas_requires_explicit_existing_path(self) -> None:
        trainer = load_trainer_module()
        with mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TRITON_PTXAS_PATH"):
                trainer.configure_triton_ptxas_path()

        with mock.patch.dict(
            os.environ,
            {"PATH": os.environ.get("PATH", ""), "TRITON_PTXAS_PATH": "/missing/ptxas"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                trainer.configure_triton_ptxas_path()

    def test_triton_ptxas_uses_only_explicit_path(self) -> None:
        trainer = load_trainer_module()
        with tempfile.TemporaryDirectory() as tmp:
            ptxas = Path(tmp) / "ptxas"
            ptxas.write_text("#!/bin/sh\n", encoding="utf-8")
            ptxas.chmod(0o755)
            with mock.patch.dict(
                os.environ,
                {"PATH": "/usr/bin", "TRITON_PTXAS_PATH": str(ptxas)},
                clear=True,
            ):
                summary = trainer.configure_triton_ptxas_path()

                self.assertEqual(summary["triton_ptxas_path"], str(ptxas))
                self.assertEqual(summary["triton_ptxas_source"], "explicit")
                self.assertTrue(summary["triton_ptxas_exists"])
                self.assertTrue(os.environ["PATH"].startswith(str(ptxas.parent)))


if __name__ == "__main__":
    unittest.main()
