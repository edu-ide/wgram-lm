# Research Major-Gate Discipline

Date: 2026-05-08

Status: active research operating rule.

## Why This Exists

The project has many useful accepted smoke tests, scaffolds, and local causal
signals. That does not mean a major raw-intelligence bottleneck is solved.

The current failure mode is accept inflation:

```text
small causal probe accepted
-> many experiments accumulate
-> major bottleneck still not crossed
-> architecture progress is overestimated
```

This document is the wiki SSoT for separating small evidence from real
architecture promotion.

## Accept Levels

Every experiment must declare one target level before training or evaluation:

```text
L0 diagnostic:
  tooling, smoke, wiring, metric, data, or artifact check.

L1 scaffold:
  a component shows a causal signal on a tiny or narrow gate.

L2 local gate:
  a held-out task improves and the relevant ablation drops, but the scope is
  still narrow.

L3 major bottleneck:
  one roadmap bottleneck is solved under held-out, perturbation, and causal
  ablation gates.

L4 canonical promotion:
  the result improves the universal LLM path and becomes the default
  architecture without damaging donor language behavior.
```

Only `L3` and `L4` count as major architecture progress. `L0-L2` results are
useful evidence, not proof of general raw intelligence.

## Required Gate Header

Before launching a run, write this header in the ledger or wiki:

```text
Target level:
Major bottleneck:
Baseline to beat:
Required score:
Required ablation drop:
Perturbation/held-out split:
Promotion decision if pass:
Kill decision if fail:
```

If the result is below the required score, call it `rejected` or `partial`, not
`accepted`. If the model ties a simpler baseline, the architecture claim is
rejected even when the absolute score looks high.

## Current Count Rule

Project status must be reported in two separate counts:

```text
small accepted probes/scaffolds/local gates:
  count separately as L0-L2 evidence.

major bottlenecks accepted:
  count only L3/L4 roadmap crossings.
```

As of 2026-05-08:

```text
major bottlenecks accepted: 0 / 10
active major blocker: bottleneck 1, prompt-conditioned latent operation order
best clean partial: 99 / 128
latest candidates: rejected below the 99 / 128 partial baseline
```

## Failure Budget

Do not keep adding heads, losses, adapters, thresholds, or prompt variants
after repeated failure. Apply this budget per major bottleneck:

```text
2 failed local fixes:
  run the big-structure doubt gate before any new local patch.

3 failed local fixes:
  stop local iteration; propose a replacement architecture and a minimal A/B
  falsification experiment.

5 failed local fixes or one week of no L3 progress:
  reset to the smallest official/prior-backed reproduction before integrating
  with the larger QTRM system.
```

## Recursive-Core Reset Rule

For QTRM recursive-core work, the reset order is mandatory:

```text
1. Reproduce the simplest TRM/recursive-core depth gain without donor,
   MemoryOS, retrieval, verifier sidecars, or hidden answer channels.
2. Port the minimum recurrence into QTRM on token/donor states.
3. Prove donor-only < QTRM and core_off < QTRM on held-out cases.
4. Only then add memory, metacognition, renderer, or donor-preservation logic.
```

If isolated recurrence cannot show depth gain, do not keep tuning integrated
donor-QTRM as if the core already works.

## Promotion Rule

Promote only when all are true:

```text
universal LLM path is preserved:
  prompt -> tokenizer -> token/donor states -> QTRM core/memory -> logits -> text

the claimed path is causally necessary:
  core/register/memory/off or shuffle ablations drop

the result beats simpler baselines:
  donor-only, core-off, prompt-only, retrieval-only, or rule-solver baselines

the result survives perturbation:
  held-out prompts, lengths, value ranges, seeds, distractors, or midpoint
  checkpoint comparison

language behavior is preserved:
  donor-correct cases and greedy generation do not regress for L4 promotion
```

## Reject Rule

Reject the architecture claim when any are true:

```text
the answer is computed by a sidecar while the LLM only copies it
the component-off path ties or beats full QTRM
training loss improves but held-out generation or ablation proof does not
the result only works through thresholds, prompt formatting, or hidden channels
the system needs more guard heads to hide the same root failure
```

## Practical Consequence

The next valid move is not "more experiments" in the broad sense. It is:

```text
define L3 gate -> test smallest architecture -> reject quickly or promote
```

For the current project, that means either crossing bottleneck 1 with exact
held-out reverse composition and ablation drops, or resetting to an isolated
TRM-style depth-gain reproduction before continuing integrated QTRM work.

## One-Click Runner

The gate discipline is now executable through:

```bash
PYTHONPATH=src .venv/bin/python scripts/300_research_gate_runner.py \
  --gate donorless_recurrent_depth \
  --profile standard \
  --write-wiki
```

The runner does the token-expensive bookkeeping in files:

```text
train/eval command
-> report.json
-> gate_summary.json
-> accepted/rejected decision
-> next_action branch
-> optional wiki result append
```

Default branch policy:

```text
donorless_recurrent_depth accepted:
  open qtrm_minimal_depth gate

donorless_recurrent_depth rejected:
  stop integrated donor-QTRM tuning and redesign the donorless recurrence/task
```

Use `--dry-run` to inspect the branch without training, and `--skip-existing`
to parse an existing `report.json` without rerunning the experiment.
