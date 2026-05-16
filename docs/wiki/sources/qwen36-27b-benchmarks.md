# Qwen3.6-27B Benchmark Targets

Date: 2026-05-15.

Status: target baseline for QTRM-Native-2B/3B milestone gates.

Primary source:

- `https://huggingface.co/Qwen/Qwen3.6-27B`

Secondary summary sources used only for cross-checking:

- `https://benchlm.ai/models/qwen3-6-27b`
- `https://awesomeagents.ai/models/qwen-3-6-27b/`

## Model Card Targets

The current target scores used by the local milestone gate are:

| Benchmark | Qwen3.6-27B target |
|---|---:|
| SWE-bench Verified | 77.2 |
| SWE-bench Pro | 53.5 |
| SWE-bench Multilingual | 71.3 |
| Terminal-Bench 2.0 | 59.3 |
| SkillsBench Avg5 | 48.2 |
| QwenWebBench | 1487 |
| NL2Repo | 36.2 |
| Claw-Eval Avg | 72.4 |
| MMLU-Pro | 86.2 |
| GPQA Diamond | 87.8 |
| AIME 2026 | 94.1 |
| HMMT Feb 2026 | 84.3 |
| HLE | 24 |

## QTRM Interpretation

The project target is not to claim broad superiority from a single synthetic
probe. A 2B/3B QTRM-Native model beats Qwen3.6-27B only if the same native
token-to-logit path exceeds a Qwen3.6-27B score under matched prompts,
decoding, and scoring.

Bridge or donor-backed wins are diagnostic. They may show that the QTRM core
can produce a useful residual, but they do not count as a QTRM-Native win.

## Required Comparison Discipline

```text
same benchmark cases
same answer normalization
same decoding budget
same verifier/scorer
Qwen3.6-27B baseline saved
QTRM-Native score saved
core/depth/memory ablations saved
```

Public benchmark wins must be reported separately from scoped raw-reasoning
wins. A scoped raw-reasoning win is useful only as an intermediate milestone.
