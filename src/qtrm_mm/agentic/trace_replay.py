from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .harness import CognitiveLoopRun


@dataclass(frozen=True)
class TraceReplayDataset:
    records: tuple[dict[str, Any], ...]

    @classmethod
    def from_runs(cls, runs: Iterable[CognitiveLoopRun]) -> "TraceReplayDataset":
        records: list[dict[str, Any]] = []
        for run in runs:
            context_record = run.context_tape.to_training_record()
            for transition in run.transitions:
                records.append(
                    {
                        "task_id": run.task_id,
                        "step": transition.step,
                        "context_hash": run.context_tape.context_hash,
                        "chat_prompt": context_record["chat_prompt"],
                        "workspace_context": context_record["workspace_context"],
                        "state_summary": transition.state_summary,
                        "action_target": transition.action.value,
                        "action_args": dict(transition.action_args),
                        "observation": transition.observation,
                        "verifier_status": transition.verifier.status,
                        "reward": float(transition.verifier.reward),
                        "policy_role": transition.policy_role,
                        "mode": transition.mode,
                    }
                )
        return cls(records=tuple(records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def to_jsonl_rows(self) -> Sequence[dict[str, Any]]:
        return self.records
