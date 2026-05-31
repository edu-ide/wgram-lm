# Security Policy

## Supported scope

Security review currently covers code, scripts, configs, tests, and
documentation in the default public branch. Generated experiment outputs,
downloaded corpora, model checkpoints, and third-party services are outside the
repository's direct support scope and may have their own security and license
constraints.

## Reporting a vulnerability

Please use GitHub private vulnerability reporting if it is enabled for this
repository. If private reporting is unavailable, open a minimal public issue
asking for a security contact and do not include exploit details, secrets,
private data, or full reproduction payloads in the public issue.

Useful reports include:

- affected file, script, config, or workflow,
- expected impact,
- safe reproduction steps,
- whether the issue involves untrusted input, serialization, path handling,
  dependency resolution, model artifacts, or dataset loading.

## Maintainer response

Maintainers aim to acknowledge actionable reports, reproduce the issue, decide
on severity, and coordinate a fix before public disclosure when appropriate.
Because this is a research codebase, response time may vary, but reports that
affect code execution, data exposure, or unsafe artifact loading are prioritized.

## Security-sensitive areas

- Loading checkpoints, pickles, tensors, JSONL, archives, and external datasets.
- Shell scripts that download, transform, prune, or upload files.
- Paths that cross workspace, cache, data, and checkpoint boundaries.
- Optional acceleration backends and dependency-specific execution paths.
- Any workflow that may run against untrusted research artifacts.
