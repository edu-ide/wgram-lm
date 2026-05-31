from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from wgram_lm.eval.memory_retrieval import score_answer

from .cognitive_loop import Action, TraceTransition, VerifierResult
from .context_tape import ContextItem, ContextItemType, TypedContextTape


@dataclass(frozen=True)
class CognitiveLoopRun:
    task_id: str
    final_status: str
    final_answer: str
    transitions: tuple[TraceTransition, ...]
    context_tape: TypedContextTape

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_status": self.final_status,
            "final_answer": self.final_answer,
            "context_tape": self.context_tape.to_training_record(),
            "transitions": [step.to_json_dict() for step in self.transitions],
        }


class ScriptedCognitiveHarness:
    """Stage-0 external baseline for the ASI cognitive loop.

    The harness is intentionally simple: it retrieves provided evidence,
    verifies a candidate answer against expected answers, and records every
    transition. QTRM must beat this traceable baseline before stronger
    cognitive-core claims are accepted.
    """

    mode = "scripted_cognitive_harness_v0"

    def run_memory_qa(
        self,
        *,
        task_id: str,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        candidate_answer: str,
        expected_answers: Sequence[str],
        checkpoint: str = "scripted",
    ) -> CognitiveLoopRun:
        tape = self._build_memory_qa_tape(task_id=task_id, question=question, evidence=evidence)
        visible_prompt_hash = tape.context_hash
        evidence_ids = tuple(self._evidence_id(item, idx) for idx, item in enumerate(evidence))
        retrieved_text = self._join_evidence(evidence)

        retrieve = TraceTransition(
            task_id=task_id,
            step=0,
            state_summary=question,
            visible_prompt_hash=visible_prompt_hash,
            workspace_evidence_ids=evidence_ids,
            action=Action.RETRIEVE_MEMORY,
            action_args={"num_records": len(evidence)},
            observation=retrieved_text,
            verifier=VerifierResult(
                status="DONE",
                reward=0.0,
                reason="Retrieved provided evidence records.",
            ),
            checkpoint=checkpoint,
            mode=self.mode,
            policy_role="scripted_harness",
        )

        score = score_answer(candidate_answer, expected_answers)
        supported = bool(score["hit"])
        final_status = "SUPPORTED" if supported else "REFUTED"
        reward = 1.0 if supported else 0.0
        verify = TraceTransition(
            task_id=task_id,
            step=1,
            state_summary=question,
            visible_prompt_hash=visible_prompt_hash,
            workspace_evidence_ids=evidence_ids,
            action=Action.VERIFY_EVIDENCE,
            action_args={
                "candidate_answer": candidate_answer,
                "expected_answers": list(expected_answers),
            },
            observation=f"score={score['match_type']}",
            verifier=VerifierResult(
                status=final_status,
                reward=reward,
                reason="Candidate answer matched expected answer."
                if supported
                else "Candidate answer did not match expected answer.",
            ),
            checkpoint=checkpoint,
            mode=self.mode,
            policy_role="scripted_harness",
        )

        answer = TraceTransition(
            task_id=task_id,
            step=2,
            state_summary=question,
            visible_prompt_hash=visible_prompt_hash,
            workspace_evidence_ids=evidence_ids,
            action=Action.ANSWER,
            action_args={},
            observation=candidate_answer,
            verifier=VerifierResult(
                status=final_status,
                reward=reward,
                reason="Final answer emitted after verification.",
            ),
            checkpoint=checkpoint,
            mode=self.mode,
            policy_role="scripted_harness",
        )

        return CognitiveLoopRun(
            task_id=task_id,
            final_status=final_status,
            final_answer=candidate_answer,
            transitions=(retrieve, verify, answer),
            context_tape=tape,
        )

    def _build_memory_qa_tape(
        self,
        *,
        task_id: str,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
    ) -> TypedContextTape:
        items: list[ContextItem] = [
            ContextItem(
                ContextItemType.SYSTEM,
                "Use validated evidence, tools, and verifier status before answering.",
            ),
            ContextItem(ContextItemType.USER, question),
        ]
        for idx, item in enumerate(evidence):
            items.append(
                ContextItem(
                    ContextItemType.EVIDENCE,
                    str(item.get("text", "")),
                    item_id=self._evidence_id(item, idx),
                    metadata={
                        "source": str(item.get("source", "")),
                        "chunk_id": item.get("chunk_id", idx),
                    },
                )
            )
        return TypedContextTape(task_id=task_id, items=tuple(items))

    @staticmethod
    def _evidence_id(item: Mapping[str, Any], idx: int) -> str:
        source = str(item.get("source", "evidence"))
        chunk_id = item.get("chunk_id", idx)
        return f"{source}#{chunk_id}"

    @staticmethod
    def _join_evidence(evidence: Iterable[Mapping[str, Any]]) -> str:
        return "\n".join(str(item.get("text", "")) for item in evidence)
