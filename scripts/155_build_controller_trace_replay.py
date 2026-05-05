#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.agentic.cognitive_loop import Action


def split_memory_prompt_for_workspace(prompt: str) -> tuple[str, str]:
    text = str(prompt or "")
    marker = "\n\nUser prompt:\n"
    if not text.startswith("MemoryOS evidence") or marker not in text:
        return text, ""
    before_user_prompt, visible_prompt = text.split(marker, 1)
    evidence_text = before_user_prompt.split("\n\nUse the evidence above", 1)[0].strip()
    if not evidence_text:
        return text, ""
    return visible_prompt.strip(), evidence_text


def _clean_answer(text: str) -> str:
    answer = str(text or "").strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def _render_memory_case_prompt_workspace(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or "").strip()
    evidence = list(row.get("evidence") or [])
    distractors = list(row.get("distractors") or [])
    if not question or not evidence:
        return "", ""

    instruction = str(row.get("instruction") or "").strip()
    prompt_lines = [
        "Answer using only the evidence. Return only the short answer.",
        "If the evidence does not explicitly contain the requested answer, return UNKNOWN. Do not answer with related but different entities.",
    ]
    if instruction:
        prompt_lines.append(instruction)
    prompt_lines.append(f"Question: {question}")

    workspace_lines = ["MemoryOS evidence"]
    for fallback_idx, item in enumerate(evidence + distractors):
        source = str(item.get("source", "memory.md"))
        chunk_id = item.get("chunk_id", fallback_idx)
        score = float(item.get("score", 1.0))
        workspace_lines.append(f"SOURCE={source} CHUNK={chunk_id} SCORE={score:.4f}")
        workspace_lines.append(str(item.get("text", "")).strip())

    return "\n".join(prompt_lines).strip(), "\n".join(workspace_lines).strip()


def build_controller_trace_rows(
    row: dict[str, Any],
    *,
    signal_conditioned: bool = False,
) -> list[dict[str, Any]]:
    prompt = str(row.get("prompt") or row.get("chat_prompt") or "")
    visible_prompt, workspace_context = split_memory_prompt_for_workspace(prompt)
    if not workspace_context:
        workspace_context = str(row.get("workspace_context") or row.get("workspace_text") or "")
        visible_prompt = visible_prompt or prompt
    if not visible_prompt or not workspace_context:
        visible_prompt, workspace_context = _render_memory_case_prompt_workspace(row)
    if not visible_prompt or not workspace_context:
        return []

    answer = _clean_answer(str(row.get("answer") or row.get("chosen") or ""))
    if not answer:
        aliases = row.get("answer_aliases") or []
        if aliases:
            answer = _clean_answer(str(aliases[0]))
    task_id = str(row.get("case_id") or row.get("task_id") or row.get("id") or "")
    if not task_id:
        digest = hashlib.blake2b(visible_prompt.encode("utf-8"), digest_size=8).hexdigest()
        task_id = f"trace-{digest}"

    common = {
        "type": "trace_replay",
        "task_id": task_id,
        "chat_prompt": visible_prompt,
        "workspace_context": workspace_context,
        "policy_role": "scripted_trace_sft",
    }
    if signal_conditioned:
        common.update(
            {
                "state_summary": (
                    "Decide the next cognitive-loop action from controller_signal."
                ),
                "hide_trace_step_from_input": True,
                "hide_previous_observation_from_input": True,
            }
        )
    return [
        {
            **common,
            "step": 0,
            "state_summary": common.get(
                "state_summary",
                "Need to retrieve relevant MemoryOS evidence before verification.",
            ),
            "controller_signal": [0.0, 0.0],
            "controller_world_model_signal": 0.0,
            "controller_verifier_signal": 0.0,
            "action_target": Action.RETRIEVE_MEMORY.value,
            "observation": workspace_context,
            "reward": 0.0,
            "action_sample_weight": 1.0,
        },
        {
            **common,
            "step": 1,
            "state_summary": common.get(
                "state_summary",
                "Evidence has been retrieved; verify the candidate answer against evidence.",
            ),
            "previous_observation": "" if signal_conditioned else workspace_context,
            "controller_signal": [1.0, 0.0],
            "controller_world_model_signal": 1.0,
            "controller_verifier_signal": 0.0,
            "action_target": Action.VERIFY_EVIDENCE.value,
            "observation": f"candidate_answer={answer}",
            "reward": 1.0,
            "action_sample_weight": 1.0,
        },
        {
            **common,
            "step": 2,
            "state_summary": common.get(
                "state_summary",
                "Candidate answer has been verified; emit the final answer.",
            ),
            "previous_observation": "" if signal_conditioned else f"verified_candidate_answer={answer}",
            "controller_signal": [1.0, 1.0],
            "controller_world_model_signal": 1.0,
            "controller_verifier_signal": 1.0,
            "action_target": Action.ANSWER.value,
            "observation": answer,
            "reward": 1.0,
            "action_sample_weight": 1.0,
        },
    ]


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_controller_trace_replay(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    max_source_rows: int = 0,
    signal_conditioned: bool = False,
) -> int:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        if max_source_rows > 0 and index >= max_source_rows:
            break
        rows.extend(
            build_controller_trace_rows(
                row,
                signal_conditioned=signal_conditioned,
            )
        )

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build controller action trace-replay rows from MemoryOS QA rows."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument(
        "--signal-conditioned",
        action="store_true",
        help="Hide step/observation text and require controller_signal to choose the action.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    count = write_controller_trace_replay(
        args.input_jsonl,
        args.output_jsonl,
        max_source_rows=args.max_source_rows,
        signal_conditioned=args.signal_conditioned,
    )
    print(f"wrote {count} trace replay rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
