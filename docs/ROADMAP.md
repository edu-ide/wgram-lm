# W-GRAM-LM Roadmap

This roadmap tracks public maintenance work for W-GRAM-LM. It is intentionally
focused on reproducibility, reviewability, and small research gates that an
outside contributor can inspect.

## Near Term

- Keep the donor-free smoke path fast and stable on GitHub Actions.
- Add a minimal public example that trains a tiny W-GRAM model on synthetic
  text without downloading donor weights.
- Separate generated experiment outputs from source-controlled evidence
  summaries.
- Expand issue labels and contribution paths for documentation, tests,
  reproducibility, and security review.

## Research Gates

- Verify that bounded latent workspaces remain causal and do not leak future
  response tokens during teacher-forced training.
- Track when GRAM/PTRM-style internal trajectory breadth is active rather than
  collapsed to a single route.
- Report world-model auxiliary behavior through small tests before promoting
  longer training runs.
- Keep final answers routed through the same causal language-model head instead
  of external answer selection.

## Maintainer Automation

- Use coding agents for pull request review checklists against architecture
  invariants.
- Generate reproduction scripts for failing smoke tests and issue reports.
- Summarize long experiment logs into source-backed reports before they are
  referenced from decision records.
- Audit dependency updates and security-sensitive data/checkpoint loading
  paths before merge.

## Release Direction

The `v0.1.x` line is for research scaffolding, smoke tests, and reproducible
architecture probes. Later releases should add clearer examples, smaller
fixtures, and stricter promotion gates before claiming model quality.
