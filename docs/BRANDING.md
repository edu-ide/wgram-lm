# W-GRAM-LM Branding

W-GRAM-LM is the public project name for this repository.

## Full Name

```text
W-GRAM-LM
World-Guided Generative Recursive Attractor Model for Language Modeling
```

## Short Meaning

- **World-Guided**: latent world-model probes predict future internal states and
  candidate consequences.
- **Generative Recursive**: recurrent GRAM/PTRM-style trajectories explore
  several internal lines of thought.
- **Attractor Model**: trajectories should converge toward stable answer basins
  that influence the normal LM head.
- **Language Modeling**: final answers must pass through the same causal
  language-model path, not a detached selector or sidecar speaker.

## Package Names

- Distribution package: `wgram-lm`
- Python import package: `wgram_lm`
- Recommended GitHub repository name: `wgram-lm`

## Legacy Terms

`QTRM` remains in older class names, config fields, historical decision
records, and experiment reports. Those names are legacy implementation terms,
not the public project brand.

When editing active code or new documentation, prefer:

```text
W-GRAM-LM
W-GRAM core
W-GRAM adapter
W-GRAM recursive trajectory
wgram_lm
```

Use `QTRM` only when referencing an existing class, config key, historical
experiment, or decision record that has not been migrated yet.
