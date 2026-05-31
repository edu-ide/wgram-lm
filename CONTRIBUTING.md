# Contributing

Thanks for helping improve W-GRAM-LM. This is a research-first
codebase, so the best contributions make experiments easier to reproduce,
review, and maintain.

## Good first contribution areas

- Reproduction fixes for smoke tests, training scripts, or evaluation scripts.
- Tests that lock down architecture contracts, data loading behavior, or
  promotion gates.
- Documentation that ties an implementation detail to a decision record,
  experiment report, or source-backed rationale.
- Security and reliability improvements for dependency handling, filesystem
  paths, data loading, serialization, and model checkpoint handling.
- Smaller reference implementations that make a research idea inspectable
  without requiring private infrastructure.

## Before opening a pull request

1. Search existing issues and docs to avoid duplicating active work.
2. Keep the change scoped to one behavior, script, or documentation thread.
3. Include tests or a smoke command when the change touches executable code.
4. Update documentation when a public interface, config, architecture invariant,
   or experiment workflow changes.
5. Do not commit private datasets, downloaded corpora, model checkpoints,
   secrets, local logs, or credentials.

## Local development

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
export PYTHONPATH=$PWD/src
pytest
```

For narrower changes, run the smallest relevant test first, then run the
broader test group before requesting review.

## Pull request expectations

- Explain the motivation and the user-visible or maintainer-visible effect.
- Link the relevant issue, decision record, experiment report, or source note
  when one exists.
- Call out any new dependencies, generated files, data requirements, or
  compatibility risks.
- Include exact commands used for verification and summarize the result.

## License of contributions

By contributing to this repository, you agree that your contributions are
licensed under the GNU Affero General Public License v3.0 or later, unless a
separate written agreement says otherwise.
