# QTRM LLM Wiki Schema

This wiki follows the LLM Wiki pattern: raw sources are immutable, synthesized
knowledge is maintained as markdown, and every new source should update the
pages it changes.

## Layers

- `references/official`: cloned source repositories. Treat as read-only.
- `references/papers`: downloaded papers. Treat as read-only.
- `docs/wiki`: maintained synthesis. Update this when a source changes the
  architecture understanding.

## Page Types

- `sources/*`: one page per external source with commit, paper, and extraction
  notes.
- `concepts/*`: reusable ideas such as LeWM, TRM, Gated DeltaNet, MemoryOS.
- `components/*`: QTRM implementation areas mapped against concepts and sources.
- `decisions/*`: architecture decisions and tradeoffs.
- `index.md`: content index.
- `log.md`: append-only timeline of ingests, reviews, and decisions.

## Ingest Workflow

1. Add or update the raw source under `references`.
2. Create or update a `sources/*` page.
3. Update affected `concepts/*` pages.
4. Update affected `components/*` pages.
5. Add an entry to `log.md`.
6. If the source changes architecture gates, update `docs/REFERENCE_BASELINE.md`.

## Query Workflow

1. Read `docs/wiki/index.md`.
2. Read the relevant source/concept/component pages.
3. Answer with links to the wiki and raw sources.
4. If the answer contains a reusable synthesis, file it back into the wiki.

## Lint Workflow

Periodically check for:

- concept pages without source links
- components marked official without a source mapping
- contradictions between `docs/wiki` and `docs/REFERENCE_BASELINE.md`
- stale claims after a newer source is added
- unresolved architecture gates before long training
