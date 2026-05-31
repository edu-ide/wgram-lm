#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from wgram_lm.agentic.cognitive_loop import Action
from wgram_lm.agentic.transition_controller import (
    TransitionStateController,
    TransitionStatePredictor,
)
from wgram_lm.config import load_config
from wgram_lm.data.jsonl_dataset import (
    _render_trace_replay_action_input,
    build_text_tokenizer,
)
from wgram_lm.eval.memory_retrieval import (
    build_case_prompt_and_workspace_memory,
    build_workspace_memory_text,
    canonical_answer_text,
    case_task_family,
    expected_unknown_case,
    load_cases,
    score_answer,
    select_evidence_results,
    summarize_records,
    target_retrieval_stats,
)
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter


RUNTIME_STATE_SUMMARY = (
    "Runtime controller state. Choose the next action from the task context "
    "and previous_observation."
)


def _load_memory_eval_script():
    path = Path("scripts/95_eval_memory_retrieval.py")
    spec = importlib.util.spec_from_file_location("memory_eval_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_transition_components(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> tuple[TransitionStateController, TransitionStatePredictor, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    controller_state = state["controller"]
    predictor_state = state["state_predictor"]
    if predictor_state is None:
        raise ValueError("transition checkpoint does not contain a state_predictor")
    args = dict(state.get("args") or {})
    d_model = int(predictor_state["net.1.weight"].shape[1])
    predictor_hidden_dim = int(predictor_state["net.1.weight"].shape[0])
    state_dim = int(predictor_state["net.4.weight"].shape[0])
    controller_hidden_dim = int(controller_state["action_head.weight"].shape[1])
    num_actions = int(controller_state["action_head.weight"].shape[0])
    signal_dim = int(controller_state["signal_head.weight"].shape[0])
    use_prev_action = "prev_action_embed.weight" in controller_state
    controller = TransitionStateController(
        d_model=d_model,
        num_actions=num_actions,
        hidden_dim=controller_hidden_dim,
        signal_dim=signal_dim,
        transition_state_dim=state_dim,
        use_prev_action=use_prev_action,
    )
    predictor = TransitionStatePredictor(
        d_model=d_model,
        state_dim=state_dim,
        hidden_dim=predictor_hidden_dim,
    )
    controller.load_state_dict(controller_state)
    predictor.load_state_dict(predictor_state)
    controller.to(device).eval()
    predictor.to(device).eval()
    return controller, predictor, args


def runtime_trace_row(
    *,
    case: dict[str, Any],
    prompt: str,
    workspace_context: str,
    step: int,
    previous_observation: str,
    strict_runtime_state: bool = True,
) -> dict[str, Any]:
    return {
        "type": "trace_replay",
        "task_id": str(case.get("id", "")),
        "chat_prompt": prompt,
        "workspace_context": workspace_context,
        "step": int(step),
        "state_summary": (
            RUNTIME_STATE_SUMMARY
            if strict_runtime_state
            else str(case.get("question", "") or RUNTIME_STATE_SUMMARY)
        ),
        "hide_trace_step_from_input": bool(strict_runtime_state),
        "previous_observation": previous_observation,
    }


@torch.no_grad()
def extract_runtime_feature(
    model: QTRMMultimodalModel,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    seq_len: int,
    device: str,
    feature_key: str,
    use_amp: bool,
) -> torch.Tensor:
    text = _render_trace_replay_action_input(row)
    input_ids = tokenizer.encode(text, seq_len).view(1, -1).to(device)
    pad_id = int(getattr(tokenizer, "pad_id", 0))
    attention_mask = (input_ids != pad_id).long()
    with torch.amp.autocast(
        "cuda",
        enabled=(device == "cuda" and bool(use_amp)),
        dtype=torch.bfloat16,
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_features_only=True,
        )
    if feature_key not in outputs:
        raise KeyError(f"feature key not returned by model: {feature_key}")
    return outputs[feature_key].detach().float().view(1, 1, -1)


@torch.no_grad()
def predict_runtime_action(
    *,
    model: QTRMMultimodalModel,
    feature_tokenizer: Any,
    controller: TransitionStateController,
    state_predictor: TransitionStatePredictor,
    row: dict[str, Any],
    seq_len: int,
    device: str,
    feature_key: str,
    use_amp: bool,
    controller_feature_scale: float,
    reset_hidden: bool,
    zero_transition_state: bool = False,
) -> dict[str, Any]:
    feature = extract_runtime_feature(
        model,
        feature_tokenizer,
        row,
        seq_len=seq_len,
        device=device,
        feature_key=feature_key,
        use_amp=use_amp,
    )
    state_outputs = state_predictor(feature)
    transition_state = state_outputs["transition_state_features"]
    if zero_transition_state:
        transition_state = torch.zeros_like(transition_state)
    controller_feature = feature * float(controller_feature_scale)
    outputs = controller(
        controller_feature,
        transition_state_features=transition_state,
        reset_each_step=reset_hidden,
    )
    action_id = int(outputs["action_logits"][0, 0].argmax(dim=-1).detach().cpu().item())
    try:
        action = Action.from_id(action_id)
        action_name = action.value
    except Exception:
        action_name = f"ACTION_{action_id}"
    return {
        "action_id": action_id,
        "action": action_name,
        "state_prediction": transition_state[0, 0].detach().float().cpu().tolist(),
        "state_logits": state_outputs["transition_state_logits"][0, 0]
        .detach()
        .float()
        .cpu()
        .tolist(),
    }


def _record_from_completion(
    *,
    case: dict[str, Any],
    mode: str,
    completion: str,
    raw_completion: str,
    action_trace: list[dict[str, Any]],
    action_success: bool,
    include_evidence: bool,
    evidence_mode: str,
    evidence_injection: str,
    evidence_results: list[tuple[float, dict[str, Any]]],
    answer_channel: str,
    answer_channel_meta: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    answer_score = score_answer(
        completion,
        case["answer_aliases"],
        expected_unknown=expected_unknown_case(case),
    )
    retrieval_stats = (
        target_retrieval_stats(case, evidence_results)
        if include_evidence
        else {
            "target_count": 0,
            "retrieved_target_count": 0,
            "retrieved_target": False,
            "all_targets_retrieved": False,
            "target_recall": 0.0,
        }
    )
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "question": case.get("question", ""),
        "task_family": case_task_family(case),
        "expected_unknown": expected_unknown_case(case),
        "mode": mode,
        "hit": bool(answer_score["hit"]),
        "exact_match": answer_score["exact_match"],
        "normalized_exact": answer_score["normalized_exact"],
        "normalized_contains": answer_score["normalized_contains"],
        "unknown_correct": answer_score["unknown_correct"],
        "match_type": answer_score["match_type"],
        "matched_aliases": answer_score["matched_aliases"],
        "canonical_answer": answer_score["canonical_answer"],
        "needs_human_audit": answer_score["needs_human_audit"],
        "audit_reasons": answer_score["audit_reasons"],
        "judge_status": answer_score["judge_status"],
        "answer_aliases": case["answer_aliases"],
        "completion": completion,
        "raw_completion": raw_completion,
        "answer_channel": answer_channel,
        "answer_channel_meta": answer_channel_meta or {},
        "action_trace": action_trace,
        "action_sequence": [item["action"] for item in action_trace],
        "action_success": bool(action_success),
        "failure_reason": failure_reason,
        "include_evidence": include_evidence,
        "evidence_mode": evidence_mode if include_evidence else "none",
        "evidence_injection": evidence_injection if include_evidence else "none",
        "retrieved_target": retrieval_stats["retrieved_target"],
        "target_count": retrieval_stats["target_count"],
        "retrieved_target_count": retrieval_stats["retrieved_target_count"],
        "all_targets_retrieved": retrieval_stats["all_targets_retrieved"],
        "target_recall": retrieval_stats["target_recall"],
        "retrieved_roles": [rec.get("evidence_role", "unknown") for _, rec in evidence_results],
        "retrieved_sources": [rec.get("source", "?") for _, rec in evidence_results],
    }


def evaluate_scripted_answer_channel(
    *,
    memory_eval: Any,
    case: dict[str, Any],
    mode: str,
    output_mode: str,
    eval_kwargs: dict[str, Any],
) -> dict[str, Any]:
    record = memory_eval.evaluate_case(case=case, mode=mode, **eval_kwargs)
    record["mode"] = output_mode
    record["action_trace"] = [
        {"action": "RETRIEVE_MEMORY"},
        {"action": "VERIFY_EVIDENCE"},
        {"action": "ANSWER"},
    ]
    record["action_sequence"] = ["RETRIEVE_MEMORY", "VERIFY_EVIDENCE", "ANSWER"]
    record["action_success"] = True
    record["failure_reason"] = ""
    return record


def evaluate_learned_state_loop_case(
    *,
    case: dict[str, Any],
    mode: str,
    answer_mode: str,
    memory_eval: Any,
    model: QTRMMultimodalModel,
    donor: QwenDonorAdapter,
    answer_tokenizer: Any,
    feature_tokenizer: Any,
    controller: TransitionStateController,
    state_predictor: TransitionStatePredictor,
    eval_kwargs: dict[str, Any],
    seq_len: int,
    device: str,
    feature_key: str,
    use_amp: bool,
    controller_feature_scale: float,
    reset_hidden: bool,
    zero_transition_state: bool,
    strict_runtime_state: bool,
) -> dict[str, Any]:
    include_evidence = True
    evidence_mode = str(eval_kwargs["evidence_mode"])
    evidence_injection = str(eval_kwargs["evidence_injection"])
    retrieval_top_k = int(eval_kwargs["retrieval_top_k"])
    memory_max_chars = int(eval_kwargs["memory_max_chars"])
    evidence_results = select_evidence_results(
        case,
        evidence_mode=evidence_mode,
        top_k=retrieval_top_k,
    )
    prompt, _ = build_case_prompt_and_workspace_memory(
        case,
        include_evidence=False,
        evidence_results=evidence_results,
        max_evidence_chars=memory_max_chars,
        evidence_injection="prompt",
    )
    workspace_context = build_workspace_memory_text(
        evidence_results,
        max_evidence_chars=memory_max_chars,
    )
    previous_observation = ""
    action_trace: list[dict[str, Any]] = []

    expected_actions = [Action.RETRIEVE_MEMORY, Action.VERIFY_EVIDENCE, Action.ANSWER]
    for step, expected in enumerate(expected_actions):
        row = runtime_trace_row(
            case=case,
            prompt=prompt,
            workspace_context=workspace_context,
            step=step,
            previous_observation=previous_observation,
            strict_runtime_state=strict_runtime_state,
        )
        prediction = predict_runtime_action(
            model=model,
            feature_tokenizer=feature_tokenizer,
            controller=controller,
            state_predictor=state_predictor,
            row=row,
            seq_len=seq_len,
            device=device,
            feature_key=feature_key,
            use_amp=use_amp,
            controller_feature_scale=controller_feature_scale,
            reset_hidden=reset_hidden,
            zero_transition_state=zero_transition_state,
        )
        prediction["expected_action"] = expected.value
        prediction["step"] = step
        action_trace.append(prediction)
        if prediction["action"] != expected.value:
            return _record_from_completion(
                case=case,
                mode=mode,
                completion="Answer: UNKNOWN",
                raw_completion="Answer: UNKNOWN",
                action_trace=action_trace,
                action_success=False,
                include_evidence=include_evidence,
                evidence_mode=evidence_mode,
                evidence_injection=evidence_injection,
                evidence_results=evidence_results,
                answer_channel=str(eval_kwargs["answer_channel"]),
                failure_reason=f"expected_{expected.value}_got_{prediction['action']}",
            )
        if expected is Action.RETRIEVE_MEMORY:
            previous_observation = workspace_context
        elif expected is Action.VERIFY_EVIDENCE:
            candidate_record = memory_eval.evaluate_case(
                case=case,
                mode=answer_mode,
                model=model,
                donor=donor,
                tokenizer=answer_tokenizer,
                device=device,
                max_length=eval_kwargs["max_length"],
                max_new_tokens=eval_kwargs["max_new_tokens"],
                memory_max_chars=memory_max_chars,
                evidence_mode=evidence_mode,
                retrieval_top_k=retrieval_top_k,
                memory_link_expansion=eval_kwargs["memory_link_expansion"],
                memory_index=eval_kwargs["memory_index"],
                memory_model_id=eval_kwargs["memory_model_id"],
                memory_backend=eval_kwargs["memory_backend"],
                hnsw_ef_search=eval_kwargs["hnsw_ef_search"],
                retrieve_top_n=eval_kwargs["retrieve_top_n"],
                rerank_backend=eval_kwargs["rerank_backend"],
                reranker_model_id=eval_kwargs["reranker_model_id"],
                reranker_device=eval_kwargs["reranker_device"],
                memoryos_case_filter=eval_kwargs["memoryos_case_filter"],
                evidence_injection=evidence_injection,
                base_qtrm_scale=eval_kwargs["base_qtrm_scale"],
                donor_logits_scale=eval_kwargs["donor_logits_scale"],
                measure_logit_shift=False,
                core_halt_mode=eval_kwargs["core_halt_mode"],
                suppressed_token_ids=eval_kwargs["suppressed_token_ids"],
                no_repeat_ngram_size=eval_kwargs["no_repeat_ngram_size"],
                short_answer_governor=eval_kwargs["short_answer_governor"],
                answer_channel=eval_kwargs["answer_channel"],
                evidence_span_max_tokens=eval_kwargs["evidence_span_max_tokens"],
                evidence_span_no_answer_threshold=eval_kwargs[
                    "evidence_span_no_answer_threshold"
                ],
                evidence_span_min_score=eval_kwargs["evidence_span_min_score"],
                truth_gate=eval_kwargs["truth_gate"],
                truth_support_threshold=eval_kwargs["truth_support_threshold"],
                truth_causal_threshold=eval_kwargs["truth_causal_threshold"],
                truth_refute_threshold=eval_kwargs["truth_refute_threshold"],
                truth_missing_threshold=eval_kwargs["truth_missing_threshold"],
            )
            candidate = canonical_answer_text(str(candidate_record["completion"]))
            previous_observation = f"verified_candidate_answer={candidate}"
        else:
            return _record_from_completion(
                case=case,
                mode=mode,
                completion=str(candidate_record["completion"]),
                raw_completion=str(candidate_record.get("raw_completion", candidate_record["completion"])),
                action_trace=action_trace,
                action_success=True,
                include_evidence=include_evidence,
                evidence_mode=evidence_mode,
                evidence_injection=evidence_injection,
                evidence_results=evidence_results,
                answer_channel=str(eval_kwargs["answer_channel"]),
                answer_channel_meta=dict(candidate_record.get("answer_channel_meta") or {}),
            )

    raise AssertionError("unreachable learned-state loop fallthrough")


def _mode_accuracy(summary: dict[str, Any], mode: str) -> float:
    return float(summary.get("by_mode", {}).get(mode, {}).get("accuracy", 0.0))


def action_success_rate(records: Iterable[dict[str, Any]], mode: str) -> float:
    selected = [row for row in records if row.get("mode") == mode]
    if not selected:
        return 0.0
    return sum(int(bool(row.get("action_success"))) for row in selected) / len(selected)


def build_answer_loop_gate(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    min_gain: float,
    min_drop: float,
    min_action_success: float,
) -> dict[str, Any]:
    learned = _mode_accuracy(summary, "learned_state_qtrm")
    state_off = _mode_accuracy(summary, "learned_state_qtrm_state_off")
    scripted_qtrm = _mode_accuracy(summary, "scripted_qtrm_answer_channel")
    scripted_donor = _mode_accuracy(summary, "scripted_donor_answer_channel")
    action_success = action_success_rate(records, "learned_state_qtrm")
    state_drop = learned - state_off
    gain_over_scripted = learned - scripted_qtrm
    gain_over_donor = learned - scripted_donor
    failed: list[str] = []
    if gain_over_scripted < float(min_gain):
        failed.append("learned_state_does_not_beat_scripted_qtrm")
    if gain_over_donor < float(min_gain):
        failed.append("learned_state_does_not_beat_scripted_donor")
    if state_drop < float(min_drop):
        failed.append("transition_state_not_causal_for_answer_reward")
    if action_success < float(min_action_success):
        failed.append("learned_action_loop_not_stable")
    return {
        "status": "accepted" if not failed else "rejected",
        "learned_state_qtrm_accuracy": learned,
        "scripted_qtrm_accuracy": scripted_qtrm,
        "scripted_donor_accuracy": scripted_donor,
        "state_off_accuracy": state_off,
        "gain_over_scripted_qtrm": gain_over_scripted,
        "gain_over_scripted_donor": gain_over_donor,
        "transition_state_drop": state_drop,
        "action_success_rate": action_success,
        "min_gain": float(min_gain),
        "min_drop": float(min_drop),
        "min_action_success": float(min_action_success),
        "failed_checks": failed,
    }


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    learned_records = [
        row for row in report.get("records", []) if row.get("mode") == "learned_state_qtrm"
    ]
    failed_action_records = [
        row for row in learned_records if not bool(row.get("action_success"))
    ]
    lines = [
        "# Learned-State Answer Loop Gate",
        "",
        "## Verdict",
        "",
        f"Status: `{gate['status']}`",
        "",
        "This is a task-level answer reward gate for the learned transition-state controller.",
        "The runtime rows hide trace-step and phase-specific state-summary text.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| learned_state_qtrm_accuracy | {gate['learned_state_qtrm_accuracy']:.4f} |",
        f"| scripted_qtrm_accuracy | {gate['scripted_qtrm_accuracy']:.4f} |",
        f"| scripted_donor_accuracy | {gate['scripted_donor_accuracy']:.4f} |",
        f"| state_off_accuracy | {gate['state_off_accuracy']:.4f} |",
        f"| gain_over_scripted_qtrm | {gate['gain_over_scripted_qtrm']:.4f} |",
        f"| gain_over_scripted_donor | {gate['gain_over_scripted_donor']:.4f} |",
        f"| transition_state_drop | {gate['transition_state_drop']:.4f} |",
        f"| action_success_rate | {gate['action_success_rate']:.4f} |",
        "",
        "## Mode Summary",
        "",
        "| Mode | Hits | Count | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, row in sorted(report["summary"].get("by_mode", {}).items()):
        lines.append(
            f"| {mode} | {int(row.get('hits', 0))} | {int(row.get('count', 0))} | "
            f"{float(row.get('accuracy', 0.0)):.4f} |"
        )
    lines.extend(["", "## Failed Checks", ""])
    failed = list(gate.get("failed_checks") or [])
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- none")
    lines.extend(["", "## Interpretation", ""])
    if gate["status"] == "accepted":
        lines.append(
            "The learned transition-state loop passed this narrow task-level gate."
        )
    elif (
        gate["gain_over_scripted_qtrm"] > 0
        and gate["gain_over_scripted_donor"] > 0
        and gate["transition_state_drop"] > 0
    ):
        lines.append(
            "This is a near-miss, not an acceptance: answer accuracy improved over "
            "both scripted baselines and the state-off ablation dropped, but at "
            "least one strict gate still failed."
        )
    else:
        lines.append(
            "This is a rejection: the learned transition-state loop did not prove "
            "task-level value over the simpler baselines."
        )
    if failed_action_records:
        lines.append("")
        lines.append("Action-loop failures:")
        for row in failed_action_records:
            sequence = " -> ".join(str(item) for item in row.get("action_sequence", []))
            lines.append(
                f"- `{row.get('id', '?')}`: {row.get('failure_reason', '')}; "
                f"sequence=`{sequence}`"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "An accepted action loop is not enough. This gate only accepts if the "
            "learned-state loop improves answer reward over both scripted QTRM and "
            "scripted donor answer-channel baselines while dropping under "
            "transition-state ablation.",
            "",
            "The controller input uses strict runtime rows: no trace-step oracle and "
            "no phase-specific state summary. Evidence is hidden from the controller "
            "prompt until the predicted `RETRIEVE_MEMORY` action places it into the "
            "previous-observation path.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate learned transition-state controller on task-level answer reward."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transition-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--feature-key", default="generation_verifier_pooled")
    parser.add_argument("--controller-feature-scale", type=float, default=None)
    parser.add_argument("--strict-runtime-state", action="store_true", default=True)
    parser.add_argument("--no-strict-runtime-state", dest="strict_runtime_state", action="store_false")
    parser.add_argument("--answer-channel", default="greedy", choices=["greedy", "evidence_span_copy"])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--short-answer-governor", action="store_true")
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    parser.add_argument("--evidence-mode", default="all", choices=["target", "all", "lexical"])
    parser.add_argument("--evidence-injection", default="ssot", choices=["ssot", "prompt", "workspace", "dual"])
    parser.add_argument("--retrieval-top-k", type=int, default=4)
    parser.add_argument("--memory-max-chars", type=int, default=2000)
    parser.add_argument("--memory-link-expansion", type=int, default=0)
    parser.add_argument("--qtrm-logits-scale", type=float, default=None)
    parser.add_argument("--donor-logits-scale", type=float, default=1.0)
    parser.add_argument("--core-halt-mode", default="config", choices=["config", "enabled", "disabled"])
    parser.add_argument("--evidence-span-max-tokens", type=int, default=16)
    parser.add_argument("--evidence-span-no-answer-threshold", type=float, default=0.5)
    parser.add_argument("--evidence-span-min-score", type=float, default=None)
    parser.add_argument("--truth-gate", action="store_true")
    parser.add_argument("--truth-support-threshold", type=float, default=0.5)
    parser.add_argument("--truth-causal-threshold", type=float, default=0.5)
    parser.add_argument("--truth-refute-threshold", type=float, default=0.5)
    parser.add_argument("--truth-missing-threshold", type=float, default=0.5)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--min-drop", type=float, default=0.03)
    parser.add_argument("--min-action-success", type=float, default=0.90)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    memory_eval = _load_memory_eval_script()
    model = memory_eval.load_qtrm(args.config, args.checkpoint, device)
    donor = QwenDonorAdapter(cfg.donor)
    from transformers import AutoTokenizer

    answer_tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if answer_tokenizer.pad_token_id is None:
        answer_tokenizer.pad_token = answer_tokenizer.eos_token
    feature_tokenizer = build_text_tokenizer(
        cfg.model.vocab_size,
        tokenizer_model_id=cfg.donor.model_id,
    )
    controller, state_predictor, transition_args = load_transition_components(
        args.transition_checkpoint,
        device=device,
    )
    controller_feature_scale = (
        float(transition_args.get("controller_feature_scale", 0.0))
        if args.controller_feature_scale is None
        else float(args.controller_feature_scale)
    )
    reset_hidden = bool(transition_args.get("reset_hidden", True))

    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    base_qtrm_scale = memory_eval.resolve_qtrm_scale(
        cfg.model.qtrm_logits_scale,
        args.qtrm_logits_scale,
    )
    suppressed_token_ids = memory_eval.visible_reasoning_token_ids(
        answer_tokenizer,
        enabled=args.suppress_visible_reasoning_tokens,
    )
    eval_kwargs = {
        "model": model,
        "donor": donor,
        "tokenizer": answer_tokenizer,
        "device": device,
        "max_length": cfg.train.seq_len,
        "max_new_tokens": int(args.max_new_tokens),
        "memory_max_chars": int(args.memory_max_chars),
        "evidence_mode": args.evidence_mode,
        "retrieval_top_k": int(args.retrieval_top_k),
        "memory_link_expansion": int(args.memory_link_expansion),
        "memory_index": None,
        "memory_model_id": None,
        "memory_backend": None,
        "hnsw_ef_search": None,
        "retrieve_top_n": None,
        "rerank_backend": "none",
        "reranker_model_id": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_device": None,
        "memoryos_case_filter": True,
        "evidence_injection": args.evidence_injection,
        "base_qtrm_scale": base_qtrm_scale,
        "donor_logits_scale": float(args.donor_logits_scale),
        "measure_logit_shift": False,
        "core_halt_mode": args.core_halt_mode,
        "suppressed_token_ids": suppressed_token_ids,
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "short_answer_governor": bool(args.short_answer_governor),
        "answer_channel": args.answer_channel,
        "evidence_span_max_tokens": int(args.evidence_span_max_tokens),
        "evidence_span_no_answer_threshold": float(args.evidence_span_no_answer_threshold),
        "evidence_span_min_score": args.evidence_span_min_score,
        "truth_gate": bool(args.truth_gate),
        "truth_support_threshold": float(args.truth_support_threshold),
        "truth_causal_threshold": float(args.truth_causal_threshold),
        "truth_refute_threshold": float(args.truth_refute_threshold),
        "truth_missing_threshold": float(args.truth_missing_threshold),
    }

    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(
            evaluate_scripted_answer_channel(
                memory_eval=memory_eval,
                case=case,
                mode="donor_only_with_evidence",
                output_mode="scripted_donor_answer_channel",
                eval_kwargs=eval_kwargs,
            )
        )
        records.append(
            evaluate_scripted_answer_channel(
                memory_eval=memory_eval,
                case=case,
                mode="qtrm_residual_with_evidence",
                output_mode="scripted_qtrm_answer_channel",
                eval_kwargs=eval_kwargs,
            )
        )
        records.append(
            evaluate_learned_state_loop_case(
                case=case,
                mode="learned_state_qtrm",
                answer_mode="qtrm_residual_with_evidence",
                memory_eval=memory_eval,
                model=model,
                donor=donor,
                answer_tokenizer=answer_tokenizer,
                feature_tokenizer=feature_tokenizer,
                controller=controller,
                state_predictor=state_predictor,
                eval_kwargs=eval_kwargs,
                seq_len=cfg.train.seq_len,
                device=device,
                feature_key=args.feature_key,
                use_amp=bool(cfg.train.use_amp),
                controller_feature_scale=controller_feature_scale,
                reset_hidden=reset_hidden,
                zero_transition_state=False,
                strict_runtime_state=bool(args.strict_runtime_state),
            )
        )
        records.append(
            evaluate_learned_state_loop_case(
                case=case,
                mode="learned_state_qtrm_state_off",
                answer_mode="qtrm_residual_with_evidence",
                memory_eval=memory_eval,
                model=model,
                donor=donor,
                answer_tokenizer=answer_tokenizer,
                feature_tokenizer=feature_tokenizer,
                controller=controller,
                state_predictor=state_predictor,
                eval_kwargs=eval_kwargs,
                seq_len=cfg.train.seq_len,
                device=device,
                feature_key=args.feature_key,
                use_amp=bool(cfg.train.use_amp),
                controller_feature_scale=controller_feature_scale,
                reset_hidden=reset_hidden,
                zero_transition_state=True,
                strict_runtime_state=bool(args.strict_runtime_state),
            )
        )

    summary = summarize_records(records)
    gate = build_answer_loop_gate(
        records,
        summary,
        min_gain=float(args.min_gain),
        min_drop=float(args.min_drop),
        min_action_success=float(args.min_action_success),
    )
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "transition_checkpoint": args.transition_checkpoint,
        "cases": args.cases,
        "case_count": len(cases),
        "strict_runtime_state": bool(args.strict_runtime_state),
        "feature_key": args.feature_key,
        "controller_feature_scale": controller_feature_scale,
        "reset_hidden": reset_hidden,
        "answer_channel": args.answer_channel,
        "evidence_mode": args.evidence_mode,
        "evidence_injection": args.evidence_injection,
        "summary": summary,
        "gate": gate,
        "records": records,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_out:
        md = Path(args.markdown_out)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
