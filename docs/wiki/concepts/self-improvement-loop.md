# Self-Improvement Loop

Primary source map:
[Self-Improvement And Hallucination References](../sources/self-improvement-and-hallucination.md).

Related verification source map:
[Fact Verification And Fake Info References](../sources/fact-verification-and-fake-info.md).

## Core Decision

QTRM/MemoryOS should not learn from raw self-generated correction traces as if
they were clean supervised data. The safer loop is:

```text
question
-> retrieve/search evidence
-> generate candidate answer
-> verify candidate against evidence
-> if unsupported: NEEDS_SEARCH
-> agentic search / retrieval expansion
-> verify again
-> store trace, failure tags, and chosen/rejected pair
-> train only from verified or explicitly scoped preference data
-> rerun held-out gates
```

## UNKNOWN Versus NEEDS_SEARCH

`UNKNOWN` remains useful for closed-evidence benchmarks. It means "the answer is
not present in the provided evidence."

For an open-world MemoryOS agent, `UNKNOWN` should become an internal routing
state, not the final answer. The preferred internal label is:

```text
Action: NEEDS_SEARCH
```

The final user-facing answer should be one of:

- a verified answer with cited evidence;
- a bounded statement that the system searched but could not verify the answer;
- a clarification request if the question is underspecified.

For fact-checking and fake-info tasks, preserve the reason instead of collapsing
everything into `UNKNOWN`:

```text
SUPPORTED
REFUTED
NOT_ENOUGH_INFO
CONFLICT
STALE_OR_TIME_DEPENDENT
NEEDS_SEARCH
```

This distinction is important for self-improvement. A contradicted answer, a
source-conflict answer, a stale answer, and an insufficient-evidence answer
should produce different preference rows and different failure tags.

## Training Data Types

| Data Type | Use | Risk |
| --- | --- | --- |
| Verified trace SFT | Teach answer format and simple evidence use | Can overfit if traces are narrow. |
| Preference pairs | Prefer grounded/correct answers over hallucinated distractors | Requires reliable chosen/rejected construction. |
| Reflection memory | Store why a run failed | Must not contaminate factual memory. |
| On-policy correction/RL | Train self-correction under the model's own distribution | More complex and requires stronger reward/verifier design. |
| In-Place TTT | Adapt donor fast weights at inference time | Powerful but risky; reset/budget/eval gates required. |

## QTRM Current Implementation

Added an analysis-only preference-data path:

- Module: `src/wgram_lm/training/self_improvement_data.py`
- CLI: `scripts/101_build_self_improvement_preferences.py`
- Output:
  `data/filtered/memory_self_improvement_preferences_analysis.jsonl`

Current generated analysis rows from the held-out MemoryOS eval:

- Rows: 11
- Scope: `analysis_only`
- Resolution states: `needs_search` 6, `answer` 5
- Tags include `wrong_answer`, `abstention`, `needs_search`, and
  `unknown_repetition`

These rows are not automatically training data. Held-out-derived rows are for
failure analysis and synthetic generator improvement unless explicitly copied
into a separate train-candidate dataset.

## Go/No-Go

Before using preference rows for training:

- keep held-out-derived rows out of the train split;
- generate parallel synthetic train cases for the same failure families;
- require chosen answers to be supported by evidence or marked `NEEDS_SEARCH`;
- reject rows where the "chosen" answer is an unverifiable model guess;
- rerun donor-only and QTRM residual on the same held-out gate.

## Fake-Info Training Gate

Before training on fake-info or fact-verification traces:

- each row must contain evidence text plus source metadata;
- the verdict must be one of the structured verification labels above;
- contradictory evidence must be represented explicitly, not hidden in a prose
  explanation;
- `NEEDS_SEARCH` rows should teach search routing, not final refusal;
- verifier performance must be measured separately from answer fluency.
