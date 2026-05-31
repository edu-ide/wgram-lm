from __future__ import annotations

from .config import WGRAMV2Config


def build_v2_contract(config: WGRAMV2Config) -> dict[str, object]:
    promotion_ready = (
        str(config.runtime_profile) == "promotion"
        and str(config.delta_backend) == "official_gated_delta2"
        and str(config.core_implementation) == "official_gated_delta2"
        and not bool(config.allow_torch_smoke_core)
    )
    return {
        "version": "2026-05-31.wgram-reasoning-lm-v2",
        "model_class": "wgram_lm.v2.model.WGRAMReasoningLMV2",
        "legacy_model_class": "wgram_lm.v2.model.QTRMReasoningLMV2",
        "runtime_profile": str(config.runtime_profile),
        "core_implementation": str(config.core_implementation),
        "promotion_ready": bool(promotion_ready),
        "input_path": "byte_input_to_dynamic_blt_causal_chunk_summary",
        "core_path": "gated_delta_net2_3to1_attention_recurrent_core",
        "imta_path": "same_body_latent_trajectories_with_route_anti_collapse_before_speaker",
        "own_latent_prediction": "same_body_auxiliary_not_answer_bypass",
        "answer_transition_path": "prompt_context_answer_memory_prefix_plan_commitment_then_causal_token_maturation_before_same_lm_head",
        "answer_path": "hnet_causal_speaker_same_lm_head",
        "evaluation_policy": str(config.evaluation_policy),
        "boundary_state_source": str(config.boundary_state_source),
        "forbidden_paths": [
            "forced-choice promotion",
            "candidate rerank promotion",
            "external GRAM/PTRM answer selection",
            "LeWM answer path",
            "boundary-byte-only BLT core input",
            "multiple answer heads",
        ],
        "ri_requirements": {
            "RI-1": "free-generation depth scaling plus recurrence-off regression",
            "RI-2": "long-horizon stability without repeated-token collapse",
            "RI-3": "mechanism ablations hurt decoded answers",
            "RI-4": "sparse memory must be inside recurrent latent loop before promotion",
            "RI-5": "3:1 recurrent/attention hybrid synergy beats ablations",
            "RI-6": "low training waste proven by active-mechanism telemetry and ablations",
            "RI-7": "matched-data efficiency beats weaker substrates",
        },
    }


def validate_v2_contract(config: WGRAMV2Config, *, require_promotion_ready: bool = False) -> bool:
    if str(config.evaluation_policy) != "free_generation_only":
        raise ValueError("W-GRAM V2 promotion evaluation must be free-generation-only")
    if bool(config.forced_choice_promotion_enabled):
        raise ValueError("forced-choice promotion is forbidden in W-GRAM V2")
    if bool(config.candidate_rerank_promotion_enabled):
        raise ValueError("candidate rerank promotion is forbidden in W-GRAM V2")
    if bool(config.external_gram_ptrm_answer_selection):
        raise ValueError("external GRAM/PTRM answer selection is forbidden in W-GRAM V2")
    if bool(config.lewm_answer_path_enabled):
        raise ValueError("LeWM answer path is forbidden in W-GRAM V2")
    if str(config.boundary_state_source) != "causal_chunk_summary":
        raise ValueError("W-GRAM V2 requires causal chunk summary boundary states")
    if int(config.answer_head_count) != 1:
        raise ValueError("W-GRAM V2 requires a single same LM head")
    if int(config.imta_trajectories) < 1:
        raise ValueError("W-GRAM V2 requires at least one IMTA trajectory")
    if int(config.token_maturation_steps) < 1:
        raise ValueError("W-GRAM V2 requires causal token maturation before the same LM head")
    if bool(config.answer_memory_enabled) and int(config.answer_memory_steps) < 1:
        raise ValueError("W-GRAM V2 answer memory requires at least one memory step")
    contract = build_v2_contract(config)
    if require_promotion_ready and not bool(contract["promotion_ready"]):
        raise ValueError("V2 config is not promotion-ready")
    return True
