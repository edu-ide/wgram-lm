# Ouro / Looped Language Models

Date: 2026-05-05

Primary source:

- arXiv: <https://arxiv.org/abs/2510.25741>
- Project: <https://ouro-llm.github.io/>

Related latent-loop sources:

- Looped Transformers: <https://arxiv.org/abs/2502.17416>
- COCONUT / Chain of Continuous Thought: <https://openreview.net/pdf?id=Itxz7S4Ip3>
- LoopRPT: <https://papers.cool/arxiv/2603.19714>
- LoopFormer: <https://huggingface.co/papers/2602.11451>
- Mechanistic LoopLM analysis: <https://arxiv.org/abs/2604.11791>

## Citation

```text
Scaling Latent Reasoning via Looped Language Models
Rui-Jie Zhu et al.
Submitted 2025-10-29, revised 2025-11-17.
```

## Core Claims

Ouro is a family of pre-trained Looped Language Models. Unlike Samsung TRM,
which is closer to a fixed-output puzzle solver, Ouro is directly relevant to
general LLMs:

```text
decoder-only transformer
-> parameter-shared recurrent block
-> latent-space iterative computation
-> LM logits
```

Reported ideas:

- reasoning is moved into pre-training, not only post-training CoT;
- looped latent computation uses shared transformer blocks across recurrent
  steps;
- the published Ouro models use R4 recurrent steps;
- an entropy-regularized objective learns adaptive depth allocation;
- the authors report 1.4B/2.6B models matching larger standard LLMs on broad
  benchmarks, attributing gains to knowledge manipulation rather than knowledge
  storage;
- the training pipeline scales to 7.7T tokens and includes CoT annealing,
  long-context CoT, mid-training, and reasoning SFT;
- project page says code is coming soon, with model links available.

## QTRM Relevance

This is one of the strongest current references for making QTRM more like a
general looped LLM rather than a task-specific recursive solver.

Directly useful design constraints:

```text
1. The recurrent core must be in the causal LM path.
2. The recurrent depth should be trained, not only added at inference.
3. Depth allocation should be learned or regularized, not just fixed.
4. Depth sweeps must show that additional loops improve held-out answers.
5. Latent reasoning should be evaluated as knowledge manipulation, not only
   as memorization or answer formatting.
```

## Mapping To QTRM

Current QTRM alignment:

```text
prompt tokens
-> donor hidden context
-> latent workspace
-> mandatory recursive core
-> answer_state_loop
-> LM logits
```

Missing relative to Ouro:

```text
LoopLM-style recurrent block is not yet the main pretraining path.
Depth allocation is fixed/swept, not entropy-regularized.
QTRM is adapter-scaled, not trained from scratch over trillions of tokens.
Role/value heads are still probes unless they causally improve LM logits.
```

## Architecture Implication

Prefer this direction over copying Samsung TRM literally:

```text
QTRM should become a donor-initialized / donor-assisted LoopLM-style decoder,
where recurrent latent computation updates the hidden answer state before LM
logits.
```

Do not promote a side head, executor, or hidden answer channel. Ouro supports
the same universal LLM causal-path doctrine:

```text
all useful recursive computation must flow back into normal LM logits.
```

## 2026-05-06 Update After Binder Rejection

The final-answer binder probes failed even when the binder read the selected
`core_depth_state`. This changes the local interpretation of the prior:

```text
Do not expect a small post-hoc answer adapter to create LoopLM behavior.
The recurrent latent trajectory itself must be trained to become the reasoning
state that the LM head can decode.
```

The relevant research pattern is:

```text
Looped Transformers:
  gains track effective recurrent depth and latent-thought simulation.

Ouro:
  looped reasoning is trained during pretraining with depth allocation.

COCONUT:
  hidden states are fed back as continuous thoughts, with staged
  language-to-latent curriculum.

LoopFormer / LoopRPT:
  variable-depth consistency and latent-step credit assignment are more
  plausible next losses than answer-side bridge patches.
```

QTRM next implication:

```text
Add trajectory-level process credit:
  depth-shortcut consistency,
  latent-step reward/advantage,
  depth sweep non-regression,
  and strict core/bridge/binder ablations.
```

## Next QTRM Experiment Ideas

1. Add entropy/depth regularization to the existing core halt/depth head.
2. Train with randomized depth and require monotonic or non-regressive
   held-out log-prob across depths.
3. Add shortcut-consistency / latent-step process credit before adding more
   answer binders.
4. Replace side role-value probes with recurrent answer-state updates only when
   the latent trajectory objective is stable.
5. Use core-off, depth-1, depth-4, depth-8, and bridge-off ablations as the
   promotion gate.
