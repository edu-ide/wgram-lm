from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class ContextItemType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    EVIDENCE = "evidence"
    TOOL_OBSERVATION = "tool_observation"
    ACTION = "action"
    VERIFIER_RESULT = "verifier_result"
    MEMORY_CANDIDATE = "memory_candidate"
    WORLD_MODEL_PREDICTION = "world_model_prediction"
    ANSWER = "answer"


@dataclass(frozen=True)
class ContextItem:
    item_type: ContextItemType
    content: str
    item_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def stable_id(self, index: int) -> str:
        if self.item_id:
            return self.item_id
        return f"{self.item_type.value}:{index}"

    def to_json_dict(self, index: int) -> dict[str, Any]:
        return {
            "type": self.item_type.value,
            "id": self.stable_id(index),
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TypedContextTape:
    """Single source of truth for prompt, workspace, verifier, and training views."""

    task_id: str
    items: tuple[ContextItem, ...] = ()
    schema_version: str = "typed_context_tape_v0"

    @property
    def context_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "items": [
                item.to_json_dict(index)
                for index, item in enumerate(self.items)
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def render_chat_prompt(self) -> str:
        sections: list[str] = []
        for index, item in enumerate(self.items):
            item_id = item.stable_id(index)
            if item.item_type == ContextItemType.SYSTEM:
                sections.append(f"<system>\n{item.content}\n</system>")
            elif item.item_type == ContextItemType.USER:
                sections.append(f"<user>\n{item.content}\n</user>")
            elif item.item_type == ContextItemType.EVIDENCE:
                sections.append(f'<evidence id="{item_id}">\n{item.content}\n</evidence>')
            elif item.item_type == ContextItemType.TOOL_OBSERVATION:
                sections.append(
                    f'<tool_observation id="{item_id}">\n{item.content}\n</tool_observation>'
                )
            elif item.item_type == ContextItemType.ACTION:
                sections.append(f'<action id="{item_id}">\n{item.content}\n</action>')
            elif item.item_type == ContextItemType.VERIFIER_RESULT:
                sections.append(
                    f'<verifier_result id="{item_id}">\n{item.content}\n</verifier_result>'
                )
            elif item.item_type == ContextItemType.MEMORY_CANDIDATE:
                sections.append(
                    f'<memory_candidate id="{item_id}">\n{item.content}\n</memory_candidate>'
                )
            elif item.item_type == ContextItemType.WORLD_MODEL_PREDICTION:
                sections.append(
                    f'<world_model_prediction id="{item_id}">\n{item.content}\n</world_model_prediction>'
                )
            elif item.item_type == ContextItemType.ANSWER:
                sections.append(f"<assistant>\n{item.content}\n</assistant>")
        return "\n\n".join(sections)

    def render_workspace_context(self) -> str:
        evidence = [
            item.content
            for item in self.items
            if item.item_type == ContextItemType.EVIDENCE
        ]
        return "\n\n".join(evidence)

    def render_verifier_input(self, *, candidate_answer: str) -> dict[str, Any]:
        evidence_items = [
            (index, item)
            for index, item in enumerate(self.items)
            if item.item_type == ContextItemType.EVIDENCE
        ]
        return {
            "task_id": self.task_id,
            "context_hash": self.context_hash,
            "claim": candidate_answer,
            "evidence": [item.content for _, item in evidence_items],
            "evidence_ids": [item.stable_id(index) for index, item in evidence_items],
        }

    def to_training_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "context_hash": self.context_hash,
            "items": [
                item.to_json_dict(index)
                for index, item in enumerate(self.items)
            ],
            "chat_prompt": self.render_chat_prompt(),
            "workspace_context": self.render_workspace_context(),
        }
