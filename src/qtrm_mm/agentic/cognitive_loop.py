from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Action(str, Enum):
    OBSERVE = "OBSERVE"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    SEARCH_WEB = "SEARCH_WEB"
    VERIFY_EVIDENCE = "VERIFY_EVIDENCE"
    SIMULATE = "SIMULATE"
    WRITE_TRACE = "WRITE_TRACE"
    WRITE_MEMORY = "WRITE_MEMORY"
    WRITE_SKILL = "WRITE_SKILL"
    ANSWER = "ANSWER"
    STOP = "STOP"

    @property
    def id(self) -> int:
        return ACTION_ID_BY_NAME[self.name]

    @classmethod
    def from_id(cls, action_id: int) -> "Action":
        return ACTION_BY_ID[int(action_id)]


ACTION_ORDER: tuple[Action, ...] = (
    Action.OBSERVE,
    Action.RETRIEVE_MEMORY,
    Action.VERIFY_EVIDENCE,
    Action.ANSWER,
    Action.SEARCH_WEB,
    Action.SIMULATE,
    Action.WRITE_TRACE,
    Action.WRITE_MEMORY,
    Action.WRITE_SKILL,
    Action.STOP,
)
ACTION_ID_BY_NAME: dict[str, int] = {
    action.name: index for index, action in enumerate(ACTION_ORDER)
}
ACTION_BY_ID: dict[int, Action] = {
    index: action for index, action in enumerate(ACTION_ORDER)
}


@dataclass(frozen=True)
class VerifierResult:
    status: str
    reward: float
    reason: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reward": float(self.reward),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateMemoryWrite:
    """Validated write-back candidate for MemoryOS.

    A generated fact is never committed just because the model said it. It must
    pass the full validation chain before becoming durable memory or training
    data.
    """

    content: str
    source_ids: tuple[str, ...] = ()
    grounding_passed: bool = False
    contradiction_passed: bool = False
    novelty_passed: bool = False
    usefulness_passed: bool = False
    regression_passed: bool = False
    validation_notes: tuple[str, ...] = ()

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.grounding_passed:
            reasons.append("grounding_failed")
        if not self.contradiction_passed:
            reasons.append("contradiction_failed")
        if not self.novelty_passed:
            reasons.append("novelty_failed")
        if not self.usefulness_passed:
            reasons.append("usefulness_failed")
        if not self.regression_passed:
            reasons.append("regression_failed")
        return tuple(reasons)

    @property
    def can_commit(self) -> bool:
        return not self.failure_reasons

    @property
    def commit_status(self) -> str:
        return "commit" if self.can_commit else "quarantine"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source_ids": list(self.source_ids),
            "grounding_passed": self.grounding_passed,
            "contradiction_passed": self.contradiction_passed,
            "novelty_passed": self.novelty_passed,
            "usefulness_passed": self.usefulness_passed,
            "regression_passed": self.regression_passed,
            "validation_notes": list(self.validation_notes),
            "failure_reasons": list(self.failure_reasons),
            "commit_status": self.commit_status,
        }


@dataclass(frozen=True)
class TraceTransition:
    """One observe-action-observation-reward transition.

    This trace is the durable training source for the ASI-oriented loop. The
    policy role is explicit because QTRM should be measured as a residual
    controller over donor/model/harness baselines, not assumed to carry all
    competence by itself.
    """

    task_id: str
    step: int
    state_summary: str
    visible_prompt_hash: str
    workspace_evidence_ids: tuple[str, ...]
    action: Action
    action_args: Mapping[str, Any] = field(default_factory=dict)
    observation: str = ""
    verifier: VerifierResult = field(
        default_factory=lambda: VerifierResult(status="MISSING", reward=0.0)
    )
    memory_writes: tuple[CandidateMemoryWrite, ...] = ()
    skill_writes: tuple[str, ...] = ()
    checkpoint: str = ""
    mode: str = "asi_cognitive_loop_v0"
    policy_role: str = "residual_controller"

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": int(self.step),
            "state_summary": self.state_summary,
            "visible_prompt_hash": self.visible_prompt_hash,
            "workspace_evidence_ids": list(self.workspace_evidence_ids),
            "action": self.action.value,
            "action_args": dict(self.action_args),
            "observation": self.observation,
            "verifier": self.verifier.to_json_dict(),
            "memory_writes": [write.to_json_dict() for write in self.memory_writes],
            "skill_writes": list(self.skill_writes),
            "checkpoint": self.checkpoint,
            "mode": self.mode,
            "policy_role": self.policy_role,
        }


@dataclass(frozen=True)
class CognitiveLoopContract:
    required_causal_gates: tuple[str, ...]

    def validate_gate_report(self, gate_report: Mapping[str, bool]) -> dict[str, Any]:
        failing = tuple(
            gate
            for gate in self.required_causal_gates
            if not bool(gate_report.get(gate, False))
        )
        return {
            "status": "accepted" if not failing else "rejected",
            "failing_gates": failing,
            "passed_gates": tuple(
                gate
                for gate in self.required_causal_gates
                if bool(gate_report.get(gate, False))
            ),
        }


def asi_cognitive_loop_contract() -> CognitiveLoopContract:
    return CognitiveLoopContract(
        required_causal_gates=(
            "evidence_path",
            "latent_core",
            "world_model",
            "verifier",
            "self_improvement",
            "agent_memory",
        )
    )
