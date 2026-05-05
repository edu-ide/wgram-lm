# Human-Like Loss Design Sources

Status: reference map for QTRM anti-repetition, preference, and correction-trace
training. This is not evidence that the model has human cognition. It is a
practical loss-design stack for making generated behavior more robust.

## Repetition And Degeneration

- Unlikelihood Training For Neural Text Generation
  - URL: https://arxiv.org/abs/1908.04319
  - Use in QTRM: penalize high probability on known-bad repeated tokens while
    keeping ordinary next-token CE as the main objective.
- The Curious Case of Neural Text Degeneration
  - URL: https://openreview.net/forum?id=rygGQyrFvH
  - Use in QTRM: decoding alone matters, but it is not enough when the training
    policy itself learns a repeated-token attractor.
- Repetition In Repetition Out
  - URL: https://arxiv.org/abs/2310.10226
  - Use in QTRM: repeated patterns in the corpus can be learned faithfully, so
    data filtering and loss shaping should be evaluated together.
- DITTO: Demonstration Imitation by Trajectory Transformation
  - URL: https://arxiv.org/abs/2206.02369
  - Use in QTRM: keep repetition handling data-aware; do not assume every repeat
    is invalid.

## Preference Losses

- SimPO: Simple Preference Optimization
  - URL: https://arxiv.org/abs/2405.14734
  - Use in QTRM: sequence average log-prob margin loss can train chosen answers
    over rejected generations without a separate reward model.
- ORPO: Monolithic Preference Optimization without Reference Model
  - URL: https://arxiv.org/abs/2403.07691
  - Use in QTRM: preference tuning can be merged with supervised likelihood
    rather than treated only as a second-stage RL pipeline.
- KTO: Model Alignment as Prospect Theoretic Optimization
  - URL: https://arxiv.org/abs/2402.01306
  - Use in QTRM: binary desirable/undesirable traces are useful when paired
    chosen/rejected examples are sparse.
- A Principled Finite-Margin Alignment Loss for Preference Optimization
  - URL: https://arxiv.org/abs/2508.07137
  - Use in QTRM: finite margins are a useful guard against pushing preferences
    indefinitely after the behavior is already separated.
- RE-PO: Robust Preference Optimization
  - URL: https://arxiv.org/abs/2509.24159
  - Use in QTRM: preference data can be noisy; robustness is part of the loss
    design, not just dataset cleaning.

## Process And Human-Like Behavior

- TRPA: Trust Region Preference Approximation
  - URL: https://arxiv.org/abs/2504.04524
  - Use in QTRM: preference-style optimization can be made more stable for
    reasoning than plain outcome-only reward training.
- AlphaPO
  - URL: https://openreview.net/forum?id=LmdZ0pSWtG
  - Use in QTRM: self-generated preference data needs explicit reliability
    control; otherwise self-correction traces can amplify model bias.
- Dual-Objective Language Models for Hallucination Mitigation
  - URL: https://arxiv.org/abs/2512.14549
  - Use in QTRM: hallucination mitigation should separate answer fluency from
    evidence-grounded correctness.
- ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented
  Generation
  - URL: https://arxiv.org/abs/2601.21912
  - Use in QTRM: the next MemoryOS stage should reward retrieval/search steps
    and not only final answers.

## QTRM Takeaway

Recommended staged objective:

```text
L_total =
    CE_answer
  + lambda_student * CE_qtrm_student
  + lambda_kd      * KL(qtrm || donor)
  + lambda_ul      * repeat_unlikelihood
  + lambda_pref    * preference_margin(chosen, rejected)
  + lambda_process * verified_process_loss
  + lambda_gate    * memory/residual gate regularization
```

Current implemented slice:

- `simpo_margin_loss`: preference-loss primitive over average sequence log-probs.
- `sequence_average_logprob`: chosen/rejected sequence scoring.
- JSONL `prompt`/`chosen`/`rejected` preference pair batching.
- `loss_preference_weight`, `preference_beta`, `preference_margin`: training
  config hooks.
- `preference_weight`/`confidence`: row-level robust weighting hook.
- `repetition_unlikelihood_loss`: optional adjacent repeated-token guard.
- `loss_repeat_unlikelihood_weight`: training config hook.

Not implemented yet:

- process-supervised correction trace loss;
- evidence-grounded hallucination verifier loss;
- EM-style noisy-preference reliability estimation.
