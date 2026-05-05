from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = Path("scripts/157_eval_asi_controller_causal_loop.py")
    spec = importlib.util.spec_from_file_location("asi_controller_causal_loop_eval", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_asi_gate_metrics_keeps_scripted_baseline_conservative() -> None:
    module = _load_script()
    mode_summaries = {
        "qtrm_harness": {"accuracy": 1.0},
        "qtrm_latent_core_off": {"accuracy": 1.0},
    }

    metrics = module.build_asi_gate_metrics(mode_summaries)

    assert metrics["scripted_harness"] == 1.0
    assert metrics["donor_harness"] == 1.0
    assert metrics["qtrm_harness"] == 1.0
    assert metrics["qtrm_world_model_off"] == 1.0
    assert metrics["qtrm_verifier_off"] == 1.0


def test_controller_action_summary_reports_confusion() -> None:
    module = _load_script()

    summary = module.summarize_action_predictions(
        preds=[1, 2, 3, 3],
        targets=[1, 2, 2, 3],
    )

    json.dumps(summary)
    assert summary["samples"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["per_target"]["RETRIEVE_MEMORY"]["accuracy"] == 1.0
    assert summary["per_target"]["VERIFY_EVIDENCE"]["accuracy"] == 0.5
    assert summary["confusion"]["VERIFY_EVIDENCE"]["ANSWER"] == 1


def test_model_kwargs_for_mode_masks_controller_signal_dimensions() -> None:
    import torch

    module = _load_script()
    signal = torch.tensor([[1.0, 1.0], [0.5, 0.25]])
    base = {"controller_signal": signal}

    world_off = module._model_kwargs_for_mode(base, mode_name="qtrm_world_model_off")
    verifier_off = module._model_kwargs_for_mode(base, mode_name="qtrm_verifier_off")

    assert torch.equal(world_off["controller_signal"][:, 0], torch.zeros(2))
    assert torch.equal(world_off["controller_signal"][:, 1], signal[:, 1])
    assert torch.equal(verifier_off["controller_signal"][:, 0], signal[:, 0])
    assert torch.equal(verifier_off["controller_signal"][:, 1], torch.zeros(2))
    assert torch.equal(base["controller_signal"], signal)


def test_model_kwargs_for_mode_sets_learned_signal_mask_when_no_external_signal() -> None:
    import torch

    module = _load_script()
    base = {"attention_mask": torch.ones(1, 3, dtype=torch.long)}

    world_off = module._model_kwargs_for_mode(base, mode_name="qtrm_world_model_off")
    verifier_off = module._model_kwargs_for_mode(base, mode_name="qtrm_verifier_off")

    assert torch.equal(world_off["controller_signal_mask"], torch.tensor([0.0, 1.0]))
    assert torch.equal(verifier_off["controller_signal_mask"], torch.tensor([1.0, 0.0]))
    assert "controller_signal_mask" not in base


def test_render_markdown_includes_rejected_gate_reasons() -> None:
    module = _load_script()
    summary = {
        "controller_modes": {
            "qtrm_harness": {"accuracy": 1.0, "samples": 3},
            "qtrm_latent_core_off": {"accuracy": 1.0, "samples": 3},
        },
        "asi_gate_metrics": {
            "scripted_harness": 1.0,
            "donor_harness": 1.0,
            "qtrm_harness": 1.0,
            "qtrm_latent_core_off": 1.0,
            "qtrm_world_model_off": 1.0,
            "qtrm_verifier_off": 1.0,
        },
        "asi_gate": {
            "status": "rejected",
            "failed_checks": ("qtrm_does_not_beat_scripted_harness",),
        },
    }

    text = module.render_markdown(summary)

    assert "# ASI Controller Causal Loop Eval" in text
    assert "Status: `rejected`" in text
    assert "`qtrm_does_not_beat_scripted_harness`" in text
