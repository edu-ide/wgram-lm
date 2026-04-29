# Local References

This directory is intentionally ignored by git except for this manifest.

The project keeps source metadata in:

- `docs/REFERENCE_BASELINE.md`
- `docs/wiki/sources/*`

Large local artifacts such as cloned official repositories, downloaded papers,
model configs, checkpoints, FAISS indexes, and benchmark outputs should stay on
the workstation or artifact storage, not in the GitHub repository.

Current local reference examples include:

- `references/official/qwen35`
- `references/official/peft`
- `references/official/dexperts`
- `references/official/proxy-tuning`
- `references/papers/logit_sidecar/*`

Recreate or refresh them from the URLs and commit hashes recorded in
`docs/REFERENCE_BASELINE.md`.
