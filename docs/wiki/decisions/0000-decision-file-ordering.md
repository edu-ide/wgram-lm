# 0000 Decision File Ordering

## Rule

New decision files should sort in the same order that the project learned the
ideas.

Use this filename shape:

```text
YYYY-MM-DD-stageNNN[-substage]-short-slug.md
```

Examples:

```text
2026-05-25-stage101a-solution-attractor-smoke.md
2026-05-25-stage101-solution-aligned-answer-attractor.md
2026-05-25-stage101n-nested-source-anchor-curriculum.md
```

For non-stage decisions, use a small ordering prefix:

```text
0000-decision-file-ordering.md
YYYY-MM-DD-001-short-slug.md
```

## Plain-Language Reason

The decisions directory is the project's memory. If filenames do not preserve
order, old rejected ideas look as current as new accepted ones, and the agent
can accidentally walk back into a dead branch.

Ordered names make the story readable:

```text
what we tried
what broke
what replaced it
what must not be repeated
```

## Migration Policy

Full archive ordering is the desired end state, because unordered historical
files can make rejected old ideas look current.

Do not do a blind manual mass-rename. The decisions archive currently has
hundreds of files and many cross-links, so the migration must be scripted and
audited.

Use this migration order:

```text
1. Create an active decision index.
   Agents must read the active index first, not a random old decision file.

2. Add status to every touched decision:
   active
   superseded
   rejected
   archived

3. Generate a rename map for the full archive.
   Prefer explicit dates/stage IDs from filenames or document titles.
   If a date cannot be proven, use archive-unknown rather than inventing one.

4. Apply renames with link rewriting in one scripted pass.

5. Run a broken-link audit before and after the migration.
```

Until the full migration is complete:

```text
All new decision files must use ordered names.
All active-chain decision files should be renamed when touched.
Agents must treat unordered historical filenames as archived context unless
the active decision index explicitly promotes them.
```

## Current Audit

Dry-run tool:

```text
scripts/579_audit_decision_archive_ordering.py
```

Latest report:

```text
local_eval/20260525_DECISION_ARCHIVE_ORDERING_AUDIT/report.json
```

Current result:

```text
decision files: 508
ordered files: 4
unordered files: 504
rename collisions: 0
pre-existing broken decision links: 76
```

Plain-language read:

```text
Full archive rename is directionally right, but should not be applied until the
broken links are either repaired or explicitly archived. The migration is safe
only when the rename map has no collisions and the link audit is clean or
accepted as historical breakage.
```
