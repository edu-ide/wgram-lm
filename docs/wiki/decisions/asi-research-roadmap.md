# ASI Research Roadmap

Status: research translation, 2026-05-01.

## Claim Boundary

ASI is the long-term ambition, not the current claim. The current QTRM system is
a Qwen donor-backed latent reasoning, memory, and evidence adapter. A credible
ASI-oriented path must convert papers into falsifiable gates:

```text
detect -> retrieve/search -> reason/plan -> verify -> act -> learn -> re-test
```

No paper makes the present system ASI by itself. Each prior must map to a
measurable architecture requirement.

Current sufficiency judgment, 2026-05-02:
QTRM is not sufficient for a general ASI claim. The model/runtime boundary is
now cleaner, but the system must still pass model-only language competence,
causal latent-answer, verified self-improvement, and long-horizon task-reward
gates. See
[ASI Sufficiency Gate 2026-05-02](asi-sufficiency-gate-2026-05-02.md).

Raw-intelligence correction, 2026-05-02:
the first ASI-oriented promotion signal is not retrieval, answer formatting, or
SSOT cleanliness. It is whether the model's own recursive core and trainable
memory path improve held-out tasks under causal ablation. See
[Raw Intelligence Gates](raw-intelligence-gates.md).

Raw-depth status, 2026-05-02:
S160 raw-core SFT and S080 prompt-only depth supervision were both rejected.
The core path can change answers, but `core_steps=1/2/4/8` still produce
identical held-out choices. This means the next ASI-relevant architecture work
must make recursive depth itself causal through state-transition or
intermediate-state targets, not by adding more retrieval, answer formatting, or
generic SFT.

Raw-depth update, 2026-05-02:
full answer-token CE, Cartesian row/depth scheduling, KISS routing, and
`core_only` training were added. The best non-core KISS path can overfit train8
to 7/8, but `core_off` also reaches 7/8. The stricter core-only path reaches
only `core8=3/8` versus `core_off=2/8` and donor 5/8 on train8, with no
held-out depth scaling. This rejects the current recursive core as a sufficient
raw reasoning engine. See
[Pure Recursive Full-Sequence KISS Failure Ledger](pure-recursive-depth-fullseq-kiss-failure-ledger.md).

First model-only sweep, 2026-05-02:
the residual language-stability sweep passed a narrow repetition/format sanity
check at residual scales 0.00, 0.05, and 0.10 over four plain prompts, but it
still failed answer correctness on the arithmetic probe. This means the next
accepted progress signal must be answer-level and causal, not only fluent text.

## Required Capability Axes

| Axis | Research Signal | QTRM Requirement | Gate |
| --- | --- | --- | --- |
| Grounded knowledge | RAG, Self-RAG, CRAG, RankRAG | Evidence SSoT, retrieval quality gate, answer abstention/search | Retrieved target present but wrong answer must decrease. |
| Latent reasoning | recurrent depth, looped transformers, latent thoughts | Core depth must be causal, not decorative | No-retrieval depth sweep must show `core_steps=4/8` beats donor and `core_off`. |
| Trainable memory | MSA, LM2, neural long-term memory | Memory-on must beat memory-off without MemoryOS retrieval shortcuts | Length/distractor sweep must drop under memory-off. |
| World modeling | V-JEPA 2, LeWM, language world models | Predict next latent state / consequence before answer or action | Prediction loss must improve planning/eval, not just auxiliary loss. |
| Self-improvement | SEAL, RAGEN, Agent Lightning, Voyager, AlphaEvolve | Store traces, score them, update data/model/policy only through verified gates | New self-generated data must improve held-out, not only in-sample. |
| Verification | PRM/PAV, weak-to-strong, debate/oversight | Separate proposer, verifier, critic, and update approval roles | Verifier must catch false evidence/reasoning better than donor-only. |
| Long-horizon agency | optimizable agent graphs, agent RL, skill libraries | Externalized state, task memory, max-iteration guards, tool traces | Multi-turn tasks improve with memory while respecting safety gates. |

## Architecture Translation

The ASI-oriented QTRM architecture should be staged as:

```text
MemoryOS retrieval/rerank
-> shared evidence context (SSoT)
-> visible RAG view + latent workspace view
-> QTRM recurrent latent core
-> world-model prediction head
-> answer/action candidate proposals
-> verifier/critic gates
-> answer/tool/action channel
-> history + failure ledger + curated training buffer
-> offline/online distillation or RL update
```

The important distinction is that self-improvement is not "the model believes
itself." It is a closed loop where outputs are verified by tasks, tests,
execution, citations, or stronger judges before they become training signal.

## 2026 Literature Review Delta

The latest relevant work does not support a shortcut where a small latent core
becomes ASI just by adding more papers. It points to a stricter architecture
contract:

```text
proposal -> external/latent prediction -> evidence check -> process check
-> executable/task reward -> memory write -> training-buffer write
-> held-out regression gate
```

The immediate changes for QTRM are:

| Prior | Relevant mechanism | QTRM translation |
| --- | --- | --- |
| SEAL | model-generated self-edits become persistent updates only through downstream reward | keep a verified self-improvement buffer; no self-generated trace enters training without held-out or executable reward |
| RAGEN | multi-turn agent RL needs stabilized trajectory filtering and fine-grained reasoning-aware reward | store MemoryOS/tool trajectories as MDP-style transitions; reject shallow or hallucinated thoughts |
| Agent Lightning | decouple agent execution from RL training and assign credit over trajectories | separate runtime traces from trainer records; add transition-level credit fields to history JSONL |
| AlphaEvolve / CodeEvolve | evolutionary proposal loop succeeds when evaluators are strong and executable | use candidate architecture/code mutations only when tests/evals can score them automatically |
| V-JEPA 2 / LeWorldModel | predictive latent world models enable planning when predictions are causally useful | add world-model prediction gates; require prediction quality to correlate with downstream correctness |
| Bidirectional RAG | self-improving memory needs validated write-back to avoid hallucination pollution | MemoryOS writes must pass grounding, novelty, conflict, and regression checks |
| Agent harness residual-role measurement | fixed-model performance can shift substantially depending on the stateful harness | measure QTRM as a residual controller over scripted harness and donor-only harness baselines |
| Autonomous memory agents | memory should actively acquire, validate, and curate knowledge under cost and uncertainty | add search/escalation and write-path validation before memory commits |
| Agent memory survey | modern agent memory is a write-manage-read loop with privacy, contradiction, and latency constraints | treat MemoryOS as a governed subsystem, not a passive vector store |

This makes ASI an engineering objective, not a label:

```text
No unverified self-belief becomes memory.
No unverified memory becomes training data.
No training update is accepted without held-out regression.
No latent-reasoning claim is accepted without causal ablation.
```

Current implementation hook:

```text
TypedContextTape -> ScriptedCognitiveHarness -> TraceReplayDataset
-> action-policy controller SFT -> ASI causal-loop gate
```

This is the first causal-loop contract: the same context hash feeds prompt,
workspace, verifier, and training views; QTRM claims are rejected unless QTRM
beats donor/scripted harness baselines and component-off ablations drop.

The first trainable slice is intentionally conservative: `trace_replay` rows
supervise only `ControllerHeads.action` with `loss_action_policy_weight=1.0`
and `trainable_param_policy=controller_only`. Donor language, residual logits,
workspace, core, and coda remain frozen, so this step tests whether the loop can
learn action selection before attempting donor replacement or free-form
reasoning edits.

Controller trace-replay status, 2026-05-01:

```text
checkpoint: runs/qwen35_2b_4090_controller_trace_s300/last.pt
held-in samples: 1296
held-in accuracy: 1.0000
held-in per-action: RETRIEVE_MEMORY=432/432, VERIFY_EVIDENCE=432/432, ANSWER=432/432
held-in summary: docs/wiki/decisions/controller-trace-s300-eval-summary.json
held-out source: data/eval/memory_reasoning_heldout_expanded_72.jsonl
held-out samples: 216
held-out accuracy: 1.0000
held-out per-action: RETRIEVE_MEMORY=72/72, VERIFY_EVIDENCE=72/72, ANSWER=72/72
held-out summary: docs/wiki/decisions/controller-trace-s300-heldout-eval-summary.json
```

Interpretation:
this proves the local controller head can imitate the staged
retrieve-verify-answer loop when the state is explicit, including on unseen
MemoryOS eval cases converted to the same trace contract. It does not yet prove
answer correctness, latent reasoning, or ASI progress. The next evidence must
come from harness-level task scores and causal ablations.

Stage-1 ASI causal-loop gate, 2026-05-01:

```text
report: docs/wiki/decisions/asi-controller-causal-loop-s300.md
standard gate: docs/wiki/decisions/asi-causal-loop-gate.md
qtrm_harness: 1.0000
latent_core_off: 0.7037
workspace_off: 0.6667
world_model_off: 1.0000
verifier_off: 1.0000
status: rejected
```

Interpretation:
the latent core and workspace are causally involved in this controller policy,
because disabling them collapses many final `ANSWER` actions back to
`VERIFY_EVIDENCE`. However, the gate correctly rejects ASI progress claims
because the policy only matches the scripted retrieve-verify-answer baseline
and because world-model/verifier paths are not yet used to make the action
decision.

Stage-1.5 controller-signal causal-loop scaffold, 2026-05-01:

```text
report: docs/wiki/decisions/asi-controller-signal-causal-loop-s300.md
checkpoint: runs/qwen35_2b_4090_controller_signal_s300/last.pt
held-out action accuracy: 0.9444
world_model_off: 0.3333
verifier_off: 0.6111
controller_signal_off: 0.3333
latent_core_off: 1.0000
status: rejected
```

Interpretation:
this proves the controller can be made sensitive to explicit world-model and
verifier signals. It does not prove those signals are learned, nor that the
latent core is doing the reasoning. The next experiment must replace the oracle
signal with learned heads and route the decision through a core-dependent path.

Stage-1.6 learned controller-signal replacement, 2026-05-01:

```text
full learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_s300/last.pt
  held-out action accuracy: 0.3333
  collapse: VERIFY_EVIDENCE for all rows

head-only learned-core signal:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_head_s300/last.pt
  held-out action accuracy: 0.3333
  collapse: ANSWER for all rows

learned-readout diagnostic:
  checkpoint: runs/qwen35_2b_4090_controller_learned_signal_readout_s300/last.pt
  held-out action accuracy: 0.3704
  collapse: mostly ANSWER
  latent_core_off: 0.5926
  workspace_off: 0.6296

status: rejected
failure ledger: docs/wiki/decisions/asi-controller-learned-signal-failure-ledger.md
```

Interpretation:
this falsifies the simple replacement hypothesis. A learned two-bit readout
from the current latent core is not enough to recover the oracle scaffold. A
prompt/coda readout diagnostic also fails, so this is not only bad latent
pooling. The next research step should be a transition-state controller trained
over full trace dynamics, with explicit world-model and verifier state
influencing later actions.

Stage-1.7 explicit transition-state controller smoke, 2026-05-01:

```text
report: docs/wiki/decisions/transition-state-controller-markov-smoke.md
checkpoint: runs/qwen35_2b_4090_transition_state_controller_markov_smoke/last.pt
feature_scale: 0.0
use_prev_action: true
reset_hidden: true
held-out action accuracy: 1.0000
reset_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
status: accepted narrow smoke
```

Interpretation:
this proves the trace-sequence training/eval path and an explicit transition
state are causal. It does not prove latent reasoning. The important negative
control is that `feature_scale=1.0` and hidden-only recurrent variants failed
on held-out. The next research step is therefore not more hidden recurrence; it
is explicit observation/verifier/world-model state in the controller input,
followed by a controlled latent-feature scale-up.

Stage-1.8 explicit observation/verifier transition-state smoke, 2026-05-01:

```text
report: docs/wiki/decisions/transition-state-controller-explicit-state-smoke.md
checkpoint: runs/qwen35_2b_4090_transition_state_controller_explicit_state_smoke/last.pt
feature_scale: 0.0
use_prev_action: false
use_transition_state: true
transition_state_dim: 9
held-out action accuracy: 1.0000
zero_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
status: accepted narrow smoke
```

Interpretation:
previous observation/verifier state is now on the controller's causal path.
This is stronger than the previous-action-only smoke because it still passes
when `previous_action` is removed. It remains a scripted-state baseline. The
next research step is learned state prediction and task-level reward, not
stronger claims about latent reasoning.

Stage-1.9 learned transition-state controller smoke, 2026-05-01:

```text
report: docs/wiki/decisions/transition-state-controller-learned-state-smoke.md
checkpoint: runs/qwen35_2b_4090_transition_state_controller_learned_state_smoke/last.pt
feature_scale: 1.0
controller_feature_scale: 0.0
learn_transition_state: true
use_prev_action: false
held-out action accuracy: 1.0000
state_prediction_binary_accuracy: 0.9974
zero_transition_state accuracy: 0.3333
transition_state_drop: 0.6667
status: accepted narrow smoke
```

Interpretation:
this replaces the hand-built state path with a learned state predictor for the
trace loop. The controller still does not receive direct QTRM features or
previous actions. The next research step is to test whether this learned-state
loop improves generated answer reward and conflict handling over scripted and
donor harnesses.

Stage-1.10 strict runtime learned-state answer loop, 2026-05-01:

```text
runtime-state report: docs/wiki/decisions/transition-state-controller-runtime-state-s120.md
answer-loop report: docs/wiki/decisions/learned-state-answer-loop-runtime-state-gate.md
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_state_s120/last.pt
strict runtime-state held-out action accuracy: 0.9630
strict runtime-state transition_state_drop: 0.6296
learned_state_qtrm answer accuracy: 0.6250
scripted_qtrm answer accuracy: 0.5000
scripted_donor answer accuracy: 0.5000
state_off answer accuracy: 0.2500
action_success_rate: 0.8750
status: rejected
```

Interpretation:
the first task-level answer reward signal is positive but not accepted. The
current bottleneck is not evidence retrieval, since all 8 targets were
retrieved. It is action-loop stability and answer formation under conflict or
redaction. The next research step is action-first strict runtime controller
training, then a larger 72-case answer gate.

Stage-1.11 action-first runtime controller and answer bottleneck, 2026-05-01:

```text
action report: docs/wiki/decisions/transition-state-controller-runtime-actionfirst-s200.md
answer report: docs/wiki/decisions/answer-formation-bottleneck-after-action-loop.md
checkpoint: runs/qwen35_2b_4090_transition_state_controller_runtime_actionfirst_s200/last.pt
held-out action accuracy: 1.0000
state-off action accuracy: 0.3333

learned_state_qtrm answer accuracy: 0.5000
scripted_qtrm answer accuracy: 0.5000
scripted_donor answer accuracy: 0.5000
```

Interpretation:
the strict action loop is fixed, but answer reward does not improve because the
learned loop delegates final answer formation to the same renderer as the
scripted baseline. A separate span/truth answer channel reaches 49/72 on the
held-out set while span-reader-off and donor-only are 24/72, so the next
research step is verifier-controlled answer formation, not more fixed action
ordering.

Static answer-decision threshold probe, 2026-05-01:

```text
report: docs/wiki/decisions/answer-decision-gate-truthcal-72.md
calibration baseline -> gated: 0.6111 -> 0.8056
heldout baseline -> gated: 0.7500 -> 0.4167
heldout false positives: 2 -> 0
heldout blocked positives: 16
status: rejected
```

Interpretation:
truth probabilities contain useful signal, but threshold tuning alone is not
robust. The ASI roadmap should treat verifier-controlled answer formation as a
learned decision problem with counterfactual training data, not as a scalar
post-processing rule.

Learned answer-decision head, 2026-05-02:

```text
report: docs/wiki/decisions/answer-decision-head-truthcal-train144-eval72.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_train144_eval72/last.pt
train baseline -> learned: 0.7569 -> 0.9444
eval baseline -> learned: 0.6806 -> 0.8611
eval false positives: 13 -> 0
eval block harmed: 0
status: accepted
```

Interpretation:
this validates the next ASI-loop mechanism at post-hoc level. The system should
add a learned `ANSWER_DECISION` stage after proposal and verification. It is not
yet an ASI claim because the head is not wired into the forward path or trained
end-to-end, but it proves the relevant signal exists.

Runtime answer-decision integration, 2026-05-02:

```text
report: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision.md
records: docs/wiki/decisions/evidence-span-truthcal-72-answer-decision-records.jsonl
baseline span/truth: 49 / 72 = 0.6806
runtime decision: 62 / 72 = 0.8611
blocked candidates: 14
expected-unknown false positives: 0
by family: abstention 24/24, conflict 20/24, multi_hop 18/24
status: accepted runtime integration
```

Interpretation:
the learned answer-decision signal now works inside the actual eval path, not
only in an offline replay. This still remains a sidecar decision module. The
next ASI-relevant step is to make the answer decision an ablatable in-model
head and add `REVISE` or `SEARCH_MORE` actions for positive wrong answers.

In-model answer-decision head, 2026-05-02:

```text
report: docs/wiki/decisions/inmodel-answer-decision-head-truthcal-s200.md
checkpoint: runs/qwen35_2b_4090_answer_decision_head_truthcal_s200/last.pt
records: docs/wiki/decisions/evidence-span-truthcal-72-inmodel-answer-decision-records.jsonl
full in-model: 62 / 72 = 0.8611
feature-off: 49 / 72 = 0.6806
decision-head-off: 49 / 72 = 0.6806
block improved: 13
block harmed: 0
status: accepted in-model causal gate
```

Interpretation:
the answer-decision stage is now inside the QTRM checkpoint and passes a causal
feature-off/head-off ablation. It is still not an ASI claim: the accepted signal
is verifier telemetry, not autonomous hidden-state truth intuition. The next
ASI-relevant proof must add a learned `REVISE` or `SEARCH_MORE` branch for
positive wrong answers.

## ASI-Directed System Contract

QTRM should be treated as a candidate cognitive core only if all critical paths
are causally measurable:

| Path | Required proof |
| --- | --- |
| Evidence path | `dual` must beat prompt-only or reduce false positives on evidence-conflict tasks. |
| Latent core path | depth/core-off ablations must show the recurrent core is necessary on hard cases. |
| World-model path | predicted latent consequences must improve planning or answer accuracy. |
| Verifier path | verifier must reject false evidence and bad reasoning better than donor-only. |
| Self-improvement path | accepted self-generated data must improve held-out tasks and not collapse language quality. |
| Agent path | long-horizon tool traces must improve with memory while preserving rollback and safety gates. |

If any path cannot pass its causal proof, that component is research scaffolding,
not evidence of ASI progress.

## Candidate Experiments

### 1. Dual-Path Evidence Gate

Prior: RAG, Self-RAG, CRAG, RankRAG.

Change: keep the `dual` evidence path as canonical. Add prompt-only,
workspace-only, and dual comparisons.

Accept if:

- `dual >= prompt-only` on answer accuracy;
- `dual` reduces false positives on negative/contradictory cases;
- `workspace_memory_off` causes a drop on cases where latent memory is needed.

### 2. Core-Causal Latent Reasoning

Prior: recurrent depth and looped transformer latent reasoning papers.

Change: train/evaluate variable-depth recurrent core with halt/depth telemetry.

Accept if:

- more latent steps improve hard reasoning cases;
- early exit preserves easy-case accuracy;
- `core_off` and shallow-depth modes fail on the cases full depth solves.

### 3. World-Model Before Answer

Prior: V-JEPA 2 and LeWM.

Change: require the core to predict a future latent/evidence state or action
consequence before answer/action emission.

Accept if:

- world-model predictions correlate with answer correctness;
- disabling the prediction-trained path hurts planning or multi-hop cases;
- prediction training does not degrade donor language stability.

### 4. Verified Self-Improvement Buffer

Prior: SEAL, Agent Lightning, RAGEN, Voyager, AlphaEvolve.

Change: generation history becomes a curated improvement buffer:

```text
candidate trace -> verifier/test/execution score -> keep/reject
-> training mix -> held-out regression gate
```

Accept if:

- model-generated data improves held-out tasks;
- rejected trace diversity prevents collapse;
- false evidence and hallucinated citations are filtered before training.

### 5. Scalable Oversight Layer

Prior: PRM/PAV, Qwen PRM lessons, weak-to-strong generalization, debate.

Change: add verifier roles that are not the same path as the proposer:

```text
proposer -> critic -> evidence checker -> process verifier -> final gate
```

Accept if:

- verifier rejects plausible but false evidence;
- process score predicts downstream correctness;
- stronger teacher/verifier can supervise smaller QTRM without overfitting to
  judge artifacts.

### 6. Verified Memory Write-Back

Prior: Bidirectional RAG, Self-RAG, CRAG, RankRAG, SAFE/FActScore-style
grounding checks.

Change: MemoryOS does not only retrieve. It may propose new memory writes, but
only through a validation layer:

```text
candidate memory -> attribution check -> contradiction check -> novelty check
-> usefulness score -> regression eval -> commit or quarantine
```

Accept if:

- committed memory improves future retrieval or answer quality;
- false or weakly sourced memory is quarantined;
- repeated runs do not accumulate hallucinated facts.

### 7. Evolutionary Architecture Search Loop

Prior: AlphaEvolve, CodeEvolve, DeepEvolve-style systems.

Change: use QTRM history plus tests/evals as an architecture proposal loop:

```text
failure ledger -> candidate patch/architecture proposal -> tests/evals
-> score -> keep/reject -> wiki update
```

Accept if:

- candidate changes are selected by measured gates, not plausibility alone;
- failed candidates remain documented with kill reasons;
- the loop finds improvements over manually selected baselines.

## Kill Criteria

Stop claiming ASI progress from a component if:

- donor-only or prompt-only matches the full system;
- gains disappear on held-out tasks;
- generated self-training data improves in-sample but worsens held-out;
- extra gates only hide failures without changing the causal path;
- verifier rewards are hacked or correlate poorly with correctness.

## References

Accessed 2026-05-01.

Local PDFs for the ASI-roadmap additions are stored under
`references/papers/asi_roadmap/`; the broader baseline is tracked in
`docs/REFERENCE_BASELINE.md`.

- RAG: https://arxiv.org/abs/2005.11401
- Self-RAG: https://arxiv.org/abs/2310.11511
- CRAG: https://arxiv.org/abs/2401.15884
- RankRAG: https://arxiv.org/abs/2407.02485
- RETRO: https://arxiv.org/abs/2112.04426
- Atlas: https://arxiv.org/abs/2208.03299
- Scaling test-time compute with latent reasoning: https://arxiv.org/abs/2502.05171
- Looped transformers / latent thoughts: https://arxiv.org/abs/2502.17416
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeWorldModel: https://arxiv.org/abs/2603.19312
- SEAL / Self-Adapting LLMs: https://arxiv.org/abs/2506.10943
- RAGEN: https://arxiv.org/abs/2504.20073
- Agent Lightning: https://arxiv.org/abs/2508.03680
- Voyager: https://arxiv.org/abs/2305.16291
- Language Agents as Optimizable Graphs: https://arxiv.org/abs/2402.16823
- CodeEvolve: https://arxiv.org/abs/2510.14150
- AlphaEvolve white paper/blog: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- AlphaEvolve arXiv: https://arxiv.org/abs/2506.13131
- Bidirectional RAG: https://arxiv.org/abs/2512.22199
- Agent harness residual-role measurement: https://arxiv.org/abs/2604.07236
- Memory for autonomous LLM agents: https://arxiv.org/abs/2603.07670
- Autonomous memory agents: https://arxiv.org/abs/2602.22406
- Rewarding Progress / PAV: https://arxiv.org/abs/2410.08146
- Qwen PRM lessons: https://arxiv.org/abs/2501.07301
- Weak-to-strong generalization: https://openai.com/index/weak-to-strong-generalization/
