# Governance

W-GRAM-LM is maintained as an open research implementation. The
project prioritizes reproducibility, inspectability, and source-backed
architecture decisions over rapid feature expansion.

## Maintainer responsibilities

Maintainers are responsible for:

- reviewing pull requests for correctness, scope, license compatibility, and
  reproducibility,
- triaging issues that affect tests, training workflows, data handling,
  architecture contracts, or security,
- keeping README, docs, configs, and tests aligned with implemented behavior,
- recording major architecture decisions in `docs/wiki/decisions/`,
- rejecting changes that vendor private data, opaque checkpoints, or
  incompatible third-party code.

## Decision process

Small fixes can be accepted through normal pull request review. Larger changes
should include a short design note, issue, or decision record that explains the
problem, trade-offs, verification plan, and rollback path.

Architecture changes should preserve the repository's core invariants:

- recurrent W-GRAM state runs over bounded latent workspace tokens,
- GRAM/PTRM-style stochastic breadth stays ablatable and on the same LM answer
  path,
- world-model prediction stays probe-gated until it improves answer-causal
  behavior,
- dense retrieval is a candidate-generation step, not a truth source,
- strict backend mode is required for production-scale runs,
- donor weights are adapted through explicit projectors or adapters rather than
  silently merged into the recurrent core.

## Release posture

The repository is currently alpha-stage research software. Releases should be
cut only when the smoke workflow, selected architecture tests, and relevant
documentation agree on the promoted behavior.

## Community expectations

Discussion should be technical, specific, and respectful. Maintainers may close
or redirect issues that are out of scope, cannot be reproduced, request private
data or weights, or conflate speculative results with verified behavior.
