# Agentic Closed-Loop Planner

Primary source map:
[Agentic Closed-Loop Planning References](../sources/agentic-closed-loop-planning.md).

## Definition

For QTRM, an agentic closed-loop planner means:

```text
observe state
-> choose an action
-> execute the action in MemoryOS/tools/environment
-> observe the result
-> verify success/failure
-> update trace/memory/skill state
-> choose the next action or stop
```

This is different from a normal LLM completion. A completion predicts text once.
A closed-loop planner repeatedly changes the outside world or outside memory,
checks what happened, and adapts.

## Current QTRM Status

Current QTRM is not yet an end-to-end closed-loop agent.

What exists:

- donor-backed residual generation;
- latent workspace and recursive core;
- workspace-only evidence probe for causality testing;
- logical/causal evidence bottleneck;
- MemoryOS retrieval/eval scripts outside the model;
- core-world-model auxiliary prediction over fixed action traces.

What does not exist yet:

- learned action policy;
- tool execution loop owned by QTRM;
- trace replay buffer;
- turn-level verifier rewards;
- RL credit assignment;
- learned skill promotion;
- model-based planning over candidate actions.

Therefore the honest label is:

```text
QTRM = donor-backed cognitive/memory adapter
       + scaffold for future agentic planner
```

not:

```text
QTRM = already autonomous closed-loop agent
```

## Proposed Runtime Contract

The first working closed-loop layer should be external and trace-first:

```text
Task
  |
  v
AgentHarness
  |
  +-- QTRM/Donor policy proposes one action
  |
  +-- MemoryOS/tool runtime executes action
  |
  +-- Verifier produces reward/status
  |
  +-- TraceStore records transition
  |
  +-- Context/skill memory is updated only through gates
  |
  v
Stop or next step
```

Minimal action set:

| Action | Meaning |
| --- | --- |
| `OBSERVE` | Read current task state or visible prompt. |
| `RETRIEVE_MEMORY` | Query MemoryOS evidence. |
| `SEARCH_WEB` | Search when local memory is insufficient and policy allows it. |
| `VERIFY_EVIDENCE` | Check whether evidence supports/refutes/misses the candidate answer. |
| `WRITE_TRACE` | Persist action, observation, and verifier result. |
| `WRITE_MEMORY` | Store validated factual evidence. |
| `WRITE_SKILL` | Promote a repeatedly successful procedure. |
| `SIMULATE` | Use explicit simulator or latent world model to predict a candidate action. |
| `ANSWER` | Emit final user-facing answer. |
| `STOP` | End after verifier/budget gate. |

## Trace Schema

The trace is the training data. A minimal transition should include:

```json
{
  "task_id": "...",
  "step": 0,
  "state_summary": "...",
  "visible_prompt_hash": "...",
  "workspace_evidence_ids": [],
  "action": "RETRIEVE_MEMORY",
  "action_args": {},
  "observation": "...",
  "verifier": {
    "status": "SUPPORTED|REFUTED|MISSING|CONFLICT|ERROR|DONE",
    "reward": 0.0,
    "reason": "..."
  },
  "memory_writes": [],
  "skill_writes": [],
  "checkpoint": "...",
  "mode": "agentic_closed_loop_v0"
}
```

Do not train a controller before these traces are replayable.

## Where QTRM Components Fit

| QTRM Component | Closed-Loop Role |
| --- | --- |
| Donor logits | Base language/action proposal prior. |
| QTRM residual logits | Evidence-aware correction to donor policy. |
| LatentWorkspace | Working state over prompt/evidence/action history for one step. |
| Recursive core | Repeated latent update before action selection. |
| Core-world-model head | Future action-conditioned latent transition predictor. |
| Evidence bottleneck | Verifier/value signal for whether answer/action is grounded. |
| ControllerHeads | Future learned action policy. Currently not canonical-active. |
| MemoryOS | External factual evidence and trace/skill storage environment. |

## Training Path

Stage 0: scripted closed-loop baseline.

```text
RETRIEVE_MEMORY -> VERIFY_EVIDENCE -> ANSWER
```

This is the control group.

Stage 1: trace SFT.

Train QTRM/controller heads to imitate successful action traces and avoid
failed traces.

Stage 2: verifier preference.

For the same state, prefer actions that lead to supported evidence, lower
budget cost, and correct answers.

Stage 3: turn-level RL.

Use Agent-Lightning/RAGEN-style transition rewards and credit assignment. Do
not use only episode-final reward for long traces.

Stage 4: latent model-based planning.

Use the LeWM-style core head to predict `z_{t+1}` for candidate actions. Use
real verifier feedback to correct the world model.

## Collapse Diagnostics

Closed-loop training can fail while looking superficially healthy. Required
diagnostics:

| Failure | Symptom | Guard |
| --- | --- | --- |
| Echo trap | Agent repeats the same action or phrase. | action n-gram repeat rate and failed-loop detector. |
| Template collapse | Reasoning/action text looks diverse but ignores the input. | RAGEN-2-style input-dependence or MI proxy. |
| Interaction collapse | Model stops using tools/evidence and over-relies on internal text. | tool/evidence interaction density. |
| Reward hacking | Agent optimizes verifier wording without solving task. | hidden replay tests and executable checks. |
| Memory contamination | Reflection or speculation enters factual memory. | typed write gates and source metadata. |

## Claim Boundary

This architecture can become end-to-end trainable over action traces, but the
environment itself is not differentiable. "End-to-end" should therefore mean:

```text
the policy is trained from full observe-action-observation-reward traces
```

not:

```text
every tool, web page, database, and memory write is inside one neural network
```

The practical target is a hybrid system:

- neural policy and latent planner;
- explicit MemoryOS/tool execution;
- verifier rewards;
- durable trace/skill memory;
- conservative safety/budget gates.
