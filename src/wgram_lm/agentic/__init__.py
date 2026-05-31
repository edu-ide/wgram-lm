__all__ = [
    "Action",
    "CandidateMemoryWrite",
    "ContextItem",
    "ContextItemType",
    "CognitiveLoopContract",
    "CognitiveLoopRun",
    "ScriptedCognitiveHarness",
    "TypedContextTape",
    "TraceTransition",
    "TraceReplayDataset",
    "TransitionStateController",
    "VerifierResult",
    "asi_cognitive_loop_contract",
    "evaluate_causal_loop_gate",
    "transition_action_loss",
]


def __getattr__(name):
    if name in {
        "Action",
        "CandidateMemoryWrite",
        "CognitiveLoopContract",
        "TraceTransition",
        "VerifierResult",
        "asi_cognitive_loop_contract",
    }:
        from . import cognitive_loop

        return getattr(cognitive_loop, name)
    if name in {"ContextItem", "ContextItemType", "TypedContextTape"}:
        from . import context_tape

        return getattr(context_tape, name)
    if name == "evaluate_causal_loop_gate":
        from .causal_gate import evaluate_causal_loop_gate

        return evaluate_causal_loop_gate
    if name in {"CognitiveLoopRun", "ScriptedCognitiveHarness"}:
        from . import harness

        return getattr(harness, name)
    if name == "TraceReplayDataset":
        from .trace_replay import TraceReplayDataset

        return TraceReplayDataset
    if name in {"TransitionStateController", "transition_action_loss"}:
        from .transition_controller import (
            TransitionStateController,
            transition_action_loss,
        )

        return {
            "TransitionStateController": TransitionStateController,
            "transition_action_loss": transition_action_loss,
        }[name]
    raise AttributeError(name)
