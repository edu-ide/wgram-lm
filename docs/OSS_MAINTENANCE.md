# Open Source Maintenance Plan

This document describes the recurring maintenance work behind W-GRAM-LM and the
automation areas where coding agents or API credits can reduce review load
without replacing maintainer judgment.

## Current maintainer workload

- Review implementation changes across recurrent latent workspaces,
  GRAM/PTRM-style stochastic breadth, BLT-style components, MemoryOS retrieval,
  data loading, training scripts, and evaluation gates.
- Keep architecture documents, decision records, configs, tests, and scripts in
  sync as experiments evolve.
- Triage reproduction failures from local smoke tests, long-horizon runs,
  bucket-depth sweeps, and generated evaluation reports.
- Audit security-sensitive paths around artifact loading, filesystem access,
  shell scripts, downloaded corpora, and optional acceleration backends.
- Summarize long experiment logs into source-backed reports that preserve
  enough context for later maintainers.

## Maintainer automation candidates

Coding agents and API credits are useful for:

- pull request review checklists that compare code changes against architecture
  invariants,
- test failure triage and minimal reproduction generation,
- documentation drift checks between README, configs, scripts, and decision
  records,
- security review of untrusted input paths, checkpoint loading, shell scripts,
  and dependency changes,
- release note drafts and promotion-gate summaries grounded in committed
  reports and test output.

## Guardrails

Automation should assist maintainers, not make release or security decisions on
its own. Maintainers remain responsible for verifying generated summaries,
reviewing source citations, checking licenses, and deciding whether a change is
safe to merge.

## Public value

The project contributes a public, inspectable implementation surface for
predictive multi-trajectory language modeling. It is intentionally structured so
other researchers can study the code, run small tests, compare architecture
variants, and inspect how memory, world prediction, recurrent breadth,
retrieval, attractor pressure, and generation gates interact.
