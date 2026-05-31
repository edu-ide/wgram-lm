from __future__ import annotations

from typing import Any


ACTIVE_BLT_RUNTIME_CONTRACT_VERSION = "2026-05-31.blt-prefixlm-runtime-v2"


def build_active_blt_runtime_contract(args: Any) -> dict[str, Any]:
    """Describe the executable architecture used by the BLT PrefixLM trainer.

    This is intentionally a runtime contract, not a research wish list.  The
    repo contains several QTRM-era experimental paths; this record names the
    path that scripts/557_train_blt_d_prefixlm_dataio.py actually instantiates.
    """

    answer_attractor_weight = (
        float(getattr(args, "answer_attractor_ce_weight", 0.0))
        + float(getattr(args, "answer_attractor_monotonic_weight", 0.0))
        + float(getattr(args, "answer_attractor_residual_wrong_weight", 0.0))
    )
    imta_trajectories = int(getattr(args, "imta_trajectories", 1))
    imta_diversity_weight = float(getattr(args, "imta_diversity_weight", 0.0))
    own_latent_prediction_weight = float(getattr(args, "own_latent_prediction_weight", 0.0))
    own_latent_prediction_enabled = bool(getattr(args, "own_latent_prediction_enabled", True))
    return {
        "contract_version": ACTIVE_BLT_RUNTIME_CONTRACT_VERSION,
        "entry_script": "scripts/557_train_blt_d_prefixlm_dataio.py",
        "active_model_class": "wgram_lm.models.blt_prefixlm.BLTDByteLatentPrefixLM",
        "global_core_builder": "scripts/534_train_native_prefixlm_dataio.py::build_model",
        "global_core_class": "scripts/335_train_qtrm_native_etd_probe.py::NativeQTRMETDLM",
        "patch_boundary_mode": str(getattr(args, "patch_boundary_mode", "")),
        "decoder_latent_mode": str(getattr(args, "decoder_latent_mode", "")),
        "boundary_state_source": (
            "causal learned chunk summaries from previous boundary+1 through "
            "the current boundary; non-boundary bytes participate in the "
            "recurrent latent input"
            if str(getattr(args, "patch_boundary_mode", "")) in {"hnet_dechunk", "hnetpp_flow_dechunk"}
            else "packed fixed/dynamic patch embeddings"
        ),
        "active_answer_path": (
            "hnet_causal_speaker over gated byte residual plus dechunked recurrent latent"
            if str(getattr(args, "patch_boundary_mode", "")) in {"hnet_dechunk", "hnetpp_flow_dechunk"}
            else "clean_decoder local byte decoder"
        ),
        "decoder_one_body_meaning": (
            "In the non-hnet clean_decoder branch, BLT decoder one_body removes "
            "the direct byte decoder shortcut. In hnet_dechunk/hnetpp_flow_dechunk "
            "runs, the active answer path is a full causal byte-level speaker over "
            "a small gated byte residual plus dechunked recurrent latent state. "
            "It is not wgram_lm.blocks.OneBodyParallelHybridBlock."
        ),
        "hnet_one_body_byte_gate_init": float(getattr(args, "hnet_one_body_byte_gate_init", -2.0)),
        "hnet_one_body_latent_gate_init": float(getattr(args, "hnet_one_body_latent_gate_init", 2.0)),
        "imta_trajectories": int(imta_trajectories),
        "imta_noise_std": float(getattr(args, "imta_noise_std", 0.0)),
        "imta_selector_temperature": float(getattr(args, "imta_selector_temperature", 1.0)),
        "imta_adapter_gate_init": float(getattr(args, "imta_adapter_gate_init", -1.0)),
        "imta_diversity_weight": float(imta_diversity_weight),
        "imta_answer_path": (
            "same-body K latent trajectories with per-trajectory adapters and "
            "speaker-space selection/aggregation before hnet_causal_speaker"
            if imta_trajectories > 1
            else "disabled; single latent trajectory"
        ),
        "imta_diversity_status": (
            "active trajectory diversity auxiliary"
            if imta_trajectories > 1 and imta_diversity_weight > 0.0
            else "disabled or telemetry-only"
        ),
        "own_latent_prediction_enabled": bool(own_latent_prediction_enabled),
        "own_latent_prediction_weight": float(own_latent_prediction_weight),
        "own_latent_prediction_status": (
            "active auxiliary over recurrent BLT causal chunk/core states; does not replace the same LM-head answer path"
            if own_latent_prediction_enabled and own_latent_prediction_weight > 0.0
            else "disabled or telemetry-only"
        ),
        "backbone": str(getattr(args, "backbone", "")),
        "think_structure": str(getattr(args, "think_structure", "")),
        "delta_backend": str(getattr(args, "delta_backend", "")),
        "train_think_steps": int(getattr(args, "train_think_steps", 0) or 0),
        "uses_wgram_model_core_world_model": False,
        "lewm_world_model_status": (
            "not in the active BLT PrefixLM runtime; LeWM lives in wgram_model/"
            "training.train experimental paths and is canonical-off unless a "
            "separate semantic or answer-causal gate promotes it"
        ),
        "uses_one_body_parallel_hybrid_block": False,
        "one_body_parallel_hybrid_status": (
            "separate experimental block; blocks.py states it is not wired into "
            "the default QTRMBlockStack path used by this BLT runtime"
        ),
        "answer_attractor_status": (
            "training auxiliary over multiple think depths"
            if answer_attractor_weight > 0.0
            else "disabled"
        ),
        "gram_ptrm_status": (
            "active as same-body IMTA K-trajectory latent breadth before the "
            "normal hnet_causal_speaker/LM-head answer path"
            if imta_trajectories > 1
            else (
                "disabled for this run; RI/IMTA promotion requires K>1 same-body "
                "trajectory breadth, not detached candidate reranking"
            )
        ),
    }


def validate_active_blt_runtime_contract(args: Any) -> None:
    """Fail on settings that would make the BLT runtime contract ambiguous."""

    if bool(getattr(args, "core_world_model_enabled", False)):
        raise ValueError(
            "BLT PrefixLM active runtime does not instantiate QTRM core_world_model; "
            "do not claim LeWM/world-model-in-answer-path for this run."
        )
    if float(getattr(args, "loss_core_world_model_weight", 0.0)) != 0.0:
        raise ValueError(
            "BLT PrefixLM active runtime keeps LeWM/core-world-model loss at 0.0; "
            "world-model training belongs to the separate QTRMModel path."
        )
    if bool(getattr(args, "use_parallel_hybrid_block", False)):
        raise ValueError(
            "BLT PrefixLM active runtime does not wire OneBodyParallelHybridBlock. "
            "Use a dedicated hybrid-block experiment or add explicit integration "
            "before claiming that path."
        )
    if int(getattr(args, "imta_trajectories", 1)) < 1:
        raise ValueError("--imta-trajectories must be >= 1")
    if float(getattr(args, "imta_noise_std", 0.0)) < 0.0:
        raise ValueError("--imta-noise-std must be >= 0")
    if float(getattr(args, "imta_selector_temperature", 1.0)) <= 0.0:
        raise ValueError("--imta-selector-temperature must be > 0")
    if float(getattr(args, "imta_diversity_weight", 0.0)) < 0.0:
        raise ValueError("--imta-diversity-weight must be >= 0")
    if float(getattr(args, "own_latent_prediction_weight", 0.0)) < 0.0:
        raise ValueError("--own-latent-prediction-weight must be >= 0")
