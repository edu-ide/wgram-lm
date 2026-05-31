# W-GRAM-LM

[![License: AGPL v3 or later](https://img.shields.io/badge/License-AGPL_v3%2B-blue.svg)](LICENSE)

W-GRAM-LM is an AGPL-licensed research codebase for world-guided recursive
language modeling: latent world prediction, GRAM/PTRM-style multi-trajectory
reasoning, answer-attractor convergence, and same-head LM generation.

W-GRAM stands for **World-Guided Generative Recursive Attractor Model**. The
repository is intended to make experimental memory, prediction, and reasoning
architectures reproducible without depending on closed maintenance
infrastructure or detached answer-selection sidecars.

The current implementation centers on:

```text
Qwen3.5 donor / teacher route
+ W-GRAM recursive latent workspace
+ GRAM/PTRM-style stochastic recurrent breadth
+ answer-attractor and same-head LM routing
+ Parcae-style stable recurrence
+ LeWorldModel / JEPA-style latent-state prediction probes
+ MemoryOS with LLM Wiki, Harrier text embeddings, and visual embeddings
```

## Why this project exists

Modern agent-memory and multimodal reasoning stacks are often difficult to
audit because the implementation, training workflow, retrieval layer, and
evaluation harness live in separate private systems. This repository keeps
those pieces together so researchers and maintainers can inspect how recurrent
state, source-backed memory, donor adapters, and evaluation gates interact.

The project is especially useful for:

- reproducible experiments with predictive, multi-trajectory language models,
- ablations of stochastic recurrent breadth, answer attractors, latent world
  prediction, and same-head generation,
- small-scale smoke tests that do not require downloading donor weights,
- auditable multimodal retrieval pipelines where dense embeddings are candidate
  generators rather than truth sources,
- documenting architecture decisions, failure analyses, and promotion gates in
  the same public codebase as the implementation.

## Supported modes

The package supports two execution modes:

1. **Standalone smoke mode**: train and test a small randomly initialized W-GRAM model without downloading Qwen weights.
2. **Donor adapter mode**: load a Qwen3.5-style multimodal donor through Hugging Face Transformers, freeze most donor weights, and train W-GRAM adapters/core/head modules.

The code intentionally separates:

- core architecture,
- backend kernels,
- Qwen donor adapters,
- multimodal workspace projection,
- MemoryOS indexing,
- training and inference workflows.

## Important status

This is a runnable research scaffold, not a production-ready model. It includes trainable PyTorch reference backends so smoke tests and small training runs work without official KDA/FlashAttention kernels. For production-scale runs, use the strict backend mode and install official backends.

The repository is actively maintained as a research system. Maintainer work is
tracked through architecture notes, decision records, tests, and experiment
reports rather than through a polished release cadence. External contributions
are welcome when they improve reproducibility, tests, documentation, security,
or implementation clarity.

## Open source maintenance

- License: [GNU Affero General Public License v3.0 or later](LICENSE)
  (`AGPL-3.0-or-later`).
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md).
- Security policy: [SECURITY.md](SECURITY.md).
- Governance and maintainer process: [GOVERNANCE.md](GOVERNANCE.md).
- Maintainer automation plan: [docs/OSS_MAINTENANCE.md](docs/OSS_MAINTENANCE.md).

Routine maintainer work includes reviewing research PRs, keeping architecture
documents synchronized with code, triaging reproduction failures, checking
security-sensitive dependency and data-handling changes, and summarizing long
experiment logs into source-backed reports.

## Quick start

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
export PYTHONPATH=$PWD/src

bash scripts/run_all_smoke_multimodal.sh
```

## Main scripts

```text
scripts/00_setup_env.sh              Create venv and install dependencies
scripts/01_smoke_forward.sh          Random tensor forward/backward test
scripts/02_train_smoke_text.sh       Tiny text-only training
scripts/03_train_smoke_multimodal.sh Tiny image+text training with random image features
scripts/04_build_text_memory.sh      Build Harrier/FAISS text memory index
scripts/05_build_visual_memory.sh    Build visual memory index from image paths
scripts/06_infer_multimodal.sh       Run multimodal inference scaffold
scripts/run_all_smoke_multimodal.sh  End-to-end smoke workflow
```

## Recommended real route

```text
1. Run standalone smoke tests.
2. Train 38M / 160M W-GRAM standalone versions.
3. Build MemoryOS indexes with Harrier text embedding.
4. Load Qwen3.5-2B-Base donor through wgram_lm.qwen_donor.
5. Freeze donor; train W-GRAM projector/core/controller adapters.
6. Use Qwen3.6-27B or Qwen3.5-4B as teacher to generate multimodal RAG/tool/verifier traces.
7. Healing tune with general text/code + reasoning + multimodal RAG traces.
```

## Repository layout

```text
configs/
  smoke_multimodal.yaml
  qwen35_2b_adapter.yaml
  qwen35_4b_adapter.yaml

docs/
  MULTIMODAL_ARCHITECTURE.md
  TRAINING_WORKFLOW.md
  BACKENDS.md

src/wgram_lm/
  config.py
  norm.py
  rotary.py
  attention.py
  ffn.py
  mixers.py
  stability.py
  workspace.py
  core.py
  heads.py
  multimodal_projector.py
  wgram_model.py
  qwen_donor.py
  losses.py
  infer.py

src/wgram_lm/backends/
  registry.py
  delta_backend.py
  attention_backend.py

src/wgram_lm/memoryos/
  chunk.py
  text_index.py
  visual_index.py
  retrieve.py
  wiki_compile.py

src/wgram_lm/training/
  synthetic_data.py
  train.py
```

## Design invariants

```text
1. The recurrent W-GRAM core only runs over bounded latent workspace tokens.
2. The causal decoder produces final text; the latent workspace is not a replacement for causal decoding.
3. Qwen3.5 donor weights are not blindly merged into W-GRAM; they are frozen or adapted through projectors/LoRA-style adapters.
4. Harrier is used for text semantic memory, not visual embeddings.
5. Visual memory uses donor vision embeddings, SigLIP-style embeddings, or reference CLIP-like embeddings.
6. Dense embeddings are candidate generators; source-backed verifier remains required for truth.
7. Strict backend mode must be used for production runs so fallback kernels are not accidentally used.
```

## Naming

Older files, classes, and decision records may still use `QTRM` as a legacy
architecture term. The public project, Python package, and repository identity
are now W-GRAM-LM / `wgram_lm`. See [docs/BRANDING.md](docs/BRANDING.md) for
the migration boundary.

## License

W-GRAM-LM is licensed under the GNU Affero General Public
License v3.0 or later. Third-party models, datasets, checkpoints, papers, and
generated experiment outputs may have separate licenses; this repository does
not relicense those external materials.
