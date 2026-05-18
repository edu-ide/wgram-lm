# Adaptive Depth And Test-Time Compute Sources

Date: 2026-05-07

Use this source cluster for QTRM recursive-depth routing, early exit, and
verifier-guided test-time compute.

## X/Public Social Search Policy

X/Twitter is allowed as a fast discovery radar for new papers, but it is not a
source of truth for QTRM architecture decisions.

Workflow:

```text
public X search finds a paper/repo signal
-> verify arXiv/OpenReview/publisher primary page
-> verify official GitHub or model page if code or checkpoints matter
-> record the primary source here
-> map the mechanism to a falsifiable QTRM gate
```

Useful query patterns:

```text
site:x.com arxiv looped language model latent reasoning
site:x.com arxiv latent thinking language model
site:x.com arxiv fixed point looped transformer language model
site:x.com arxiv adaptive computation early exit language model
site:x.com github recursive language model reasoning
```

2026-05-18 public X search signals verified against primary sources:

- A Mechanistic Analysis of Looped Reasoning Language Models
  - Social signal: X posts linking arXiv 2604.11791
  - Primary source: https://arxiv.org/abs/2604.11791
  - QTRM relevance: looped recurrent states often converge to cyclic fixed
    points; recurrent block size, input injection, and normalization affect
    stability. This directly supports measuring QTRM state trajectories instead
    of only final exact accuracy.

- The Depth Ceiling: On the Limits of Large Language Models in Discovering
  Latent Planning
  - Social signal: X posts linking arXiv 2604.06427
  - Primary source: https://arxiv.org/abs/2604.06427
  - QTRM relevance: final-answer-only supervision may not discover deep latent
    planning reliably. QTRM should explicitly teach or externalize multi-step
    latent transitions before claiming deep latent reasoning.

- The Illusion of Superposition? A Principled Analysis of Latent Thinking in
  Language Models
  - Social signal: X posts linking arXiv 2604.06374
  - Primary source: https://arxiv.org/abs/2604.06374
  - QTRM relevance: latent thinking can collapse or shortcut when adapted from
    pretrained language models. This supports QTRM-native training gates and
    probing whether latent states actually carry multiple candidate solutions.

- The Y-Combinator for LLMs: Solving Long-Context Rot with lambda-Calculus
  - Social signal: X posts linking paper and code
  - Primary source: https://arxiv.org/abs/2603.20105
  - Official code: https://github.com/lambda-calculus-LLM/lambda-RLM
  - QTRM relevance: external recursive control can give termination and cost
    guarantees, but it is a harness/RLM path, not proof of native latent-core
    intelligence. Use as a contrast class, not as the canonical QTRM answer
    channel.

- LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via
  Shortcut Modulation
  - Primary source: https://loopformer.github.io/
  - Official code: https://github.com/armenjeddi/loopformer
  - QTRM relevance: variable trajectory conditioning and shortcut consistency
    directly target fixed-depth loop collapse. This is a high-priority prior
    for QTRM len8/len20 recurrence stability.

- Solve the Loop: Attractor Models for Language and Reasoning
  - Primary source: https://arxiv.org/html/2605.12466v1
  - Official code: https://github.com/jacobfa/Attractor
  - Local clone: `references/official/attractor`
  - QTRM relevance: fixed-point refinement, informative output-space
    initialization, persistent proposal injection, and convergence-based
    stopping directly address unstable finite unrolling. This is the strongest
    next-prior candidate if QTRM fixed-depth loops keep failing family-floor or
    ablation-drop gates.

## 2026 Dynamic Depth / Early Exit

- ADEPT: Adaptive Dynamic Early-Exit Process for Transformers
  - arXiv: https://arxiv.org/abs/2601.03700
  - Relevance: adaptive early-exit for transformer inference. Useful as a
    baseline for halt-head style routing, but QTRM needs correctness-aware
    routing rather than pure compute saving.

- A transformer architecture alteration to incentivise externalised reasoning
  - arXiv: https://arxiv.org/abs/2603.21376
  - Relevance: early exits at intermediate depths. Supports the idea that
    shallow exits should be trained with explicit targets, not inferred only
    from final-depth stability.

- Do Transformers Use their Depth Adaptively? Evidence from a Relational
  Reasoning Task
  - arXiv: https://arxiv.org/abs/2604.12426
  - Relevance: controlled evidence that depth usage is clearer after
    task-specific finetuning. This matches QTRM's result: useful depths appear
    in fixed-depth probes, but the model does not automatically learn the right
    depth selector.

- Loop the Middle: Adaptive Depth Transformers via Selective Middle-Layer
  Recurrence
  - OpenReview: https://openreview.net/forum?id=3gPSHQIJbj
  - Relevance: recurrent middle block with adaptive halting, timestep encoding,
    and cross-iteration residuals. Closest conceptual match to QTRM's looped
    core direction.

- LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via
  Shortcut Modulation
  - arXiv: https://arxiv.org/abs/2602.11451
  - Project: https://loopformer.github.io/
  - Official implementation: https://github.com/armenjeddi/loopformer
  - Local clone: `references/official/loopformer`
  - Relevance: trains looped Transformers on variable-length trajectories and
    uses shortcut consistency so shorter loop schedules stay informative while
    longer schedules refine the state. This directly addresses QTRM's
    2026-05-07 router failure: post-hoc heads cannot reliably select depth
    from frozen core states.

- A Mechanistic Analysis of Looped Reasoning Language Models
  - arXiv: https://arxiv.org/abs/2604.11791
  - Relevance: analyzes cyclic recurrence and fixed-point behavior in looped
    reasoning models. Use it as a design warning: recurrent states need stable
    trajectory dynamics, not merely more loop count or a classifier over the
    final state.

- The Detection--Extraction Gap: Models Know the Answer Before They Can Say It
  - Hugging Face paper page: https://huggingface.co/papers/2604.06613
  - Relevance: supports separating "answer is recoverable" from "answer is
    rendered." For QTRM, depth router labels should be tied to answer
    recoverability/verifier correctness, not just hidden-state stability.

## 2025-2026 Verifier-Guided Test-Time Compute

- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
  - arXiv: https://arxiv.org/abs/2502.05171
  - Relevance: latent recurrent depth as test-time compute. QTRM's fixed-depth
    probes are in this family.

- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical
  Structure
  - arXiv: https://arxiv.org/abs/2602.03975
  - Relevance: route compute with learned heuristics. The QTRM depth router
    should learn from empirical fixed-depth outcomes.

- RoBoN: Routed Online Best-of-n for Test-Time Scaling with Multiple LLMs
  - arXiv: https://arxiv.org/abs/2512.05542
  - Relevance: routing among candidate compute paths. QTRM's depth candidates
    are internal paths rather than separate models.

- Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers
  - arXiv: https://arxiv.org/abs/2502.20379
  - Relevance: verifier choice matters. QTRM should eventually compare depth
    candidates with a verifier, but the first step is supervised route labels.

## QTRM Decision

The 2026-05-07 halt probe rejected final-depth stability as the depth target:

```text
core_halt_steps_8 matched fixed core_steps_8:
  10/24, core_steps_mean=8

best fixed depths still differed:
  core_steps_1: 14/24
  core_steps_4: 13/24
  core_steps_8: 10/24
```

Therefore the next QTRM route is:

```text
fixed-depth outcome labels -> supervised depth router -> verifier-gated routing
```

not:

```text
final-depth stability target -> halt head
```

## 2026-05-07 Update After Router-Head Rejections

The first supervised-router direction was falsified:

```text
fixed-depth oracle: 16/24
best final-state route head: 12/24
best trajectory route head: 12/24
```

Revised conclusion:

```text
Outcome labels are still useful diagnostics, but the next canonical change
should train the recursive trajectory itself.
```

Preferred next loss family:

```text
LoopFormer-style shortcut consistency:
  long trajectory final state is the stable teacher
  short trajectory state must approximate the long trajectory endpoint
  short and long paths both keep normal LM/reasoning losses
  core-off/delta-off ablations must remove any gain
```

## 2026-05-07 QTRM Shortcut-Consistency V1 Result

QTRM tested a minimal shortcut-consistency approximation:

```text
core_depth_states[1..N-1] -> cosine alignment to detached final depth state
normal donor-preserving CE/preference losses remain active
```

24-case causal forced-choice result:

```text
donor_only:              9/24
core_off:                9/24
core_steps_1:           14/24
core_steps_2:           10/24
core_steps_4:           12/24
core_steps_8:            9/24
delta_off step8:         9/24
residual_gate_off step8: 9/24
```

Decision:

```text
Reject this approximation as a solved adaptive-depth method.
It preserved the step-1 signal but failed to beat the accepted baseline and
reduced longer-depth performance.
```

Research implication:

```text
Do not treat same-run early-to-final hidden-state cosine as equivalent to
LoopFormer's variable trajectory training. The next experiment must include
separate short/long loop schedules and correctness/verifier gating so the
long path is a useful teacher rather than merely the final state of the same
unstable trajectory.
```

## 2026-05-07 QTRM Variable-Trajectory V1 Result

QTRM then tested a real two-pass short/long version:

```text
outer_steps=4 long forward
outer_steps=1 short forward on the same prompt
short state aligns to detached long state
short path keeps LM loss
long path receives a small long-over-short preference margin
```

The training proxy improved:

```text
state cosine: about 0.22 -> 0.43
long-short logp margin: about -0.03 -> +0.27
```

But the raw-intelligence gate regressed:

```text
donor_only:    9/24
core_off:      9/24
core_steps_1: 13/24
core_steps_4: 11/24
core_steps_8:  8/24
```

Conclusion:

```text
For this QTRM donor-preserving core, LoopFormer-inspired trajectory alignment
is not sufficient. The next research branch should stop treating depth as a
routing/alignment problem and instead train each recurrent step as an explicit,
verifiable state transition.
```

## 2026-05-07 QTRM Depth-Text Process CE Result

QTRM tested a direct process-credit variant:

```text
return core_depth_text_logits for all depths
apply CE to each depth slice
strip the huge auxiliary depth logits from rejected/ablation forwards
```

Result:

```text
training core_depth_text_ce:
  about 11.44 -> 11.13

heldout:
  donor_only:    9/24
  core_off:      9/24
  core_steps_1: 14/24
  core_steps_4: 12/24
  core_steps_8:  9/24
```

Conclusion:

```text
Per-depth text CE is also insufficient for this donor-preserving branch.
The depth/adaptive-compute cluster has now served its purpose as a falsifier:
the blocker is not selecting depth or making each depth text-decodable, but
the lack of a verifiable recurrent state transition.
```
