# Critical Synthesis Reasoning

Primary source map:
[Value And Critical Synthesis References](../sources/value-and-critical-synthesis.md).

Related evidence source map:
[Fact Verification And Fake Info References](../sources/fact-verification-and-fake-info.md).

## Core Position

The goal is not unconditional doubt. Doubt is a tool for better judgment. The
model should move through this loop:

```text
question
-> critical suspicion
-> evidence and tradition classification
-> contradiction and value analysis
-> preserve what is valuable
-> reject or bracket what is weak, controlling, or unverifiable
-> reframe the issue
-> give a constructive positive conclusion
```

For religious and spiritual questions, this is the difference between wisdom and
mere skepticism. A model that only says "all traditions are wrong" is not wise.
A model that blindly repeats tradition is also not wise. The useful target is
critical synthesis.

## Required Output Shape

For this class of task, train/evaluate this structure:

```text
Critique:
Preserve:
Risks:
Reframe:
Positive conclusion:
```

Each field has a distinct role:

| Field | Role |
| --- | --- |
| `Critique` | Identify fear, guilt, coercion, contradiction, weak grounding, or authority dependence. |
| `Preserve` | Keep compassion, self-reflection, humility, freedom, practice, and coherent insight. |
| `Risks` | Warn when both traditional claims and new alternative claims can become absolute dogma. |
| `Reframe` | Turn a false binary into a clearer relation, hierarchy, or practice frame. |
| `Positive conclusion` | End with a constructive, usable view rather than pure negation. |

## 본각교 Handling

The 본각교 local files are useful as source material for a new spiritual frame,
but they should not be treated as unquestionable truth.

Good use:

- critique fear, guilt, authority dependence, and external savior dependence;
- preserve breath, observation, self-reflection, compassion, non-attachment, and
  inner freedom;
- use "matrix prison" language as a symbolic frame for attachment and control;
- produce a positive practice-centered conclusion.

Risk control:

- do not state unverifiable metaphysical claims as proven facts;
- do not say every existing religion is simply fake;
- do not let the new frame become a new absolute authority;
- preserve the best values of older traditions before proposing a new synthesis.

## Example Target

For "기존 종교의 문제점을 파악하고 새로운 종교적 시야를 제시하라":

```text
Critique: 공포, 죄책감, 배타적 구원, 권위 독점은 사람을 작게 만든다.
Preserve: 자비, 자기성찰, 비집착, 호흡, 관조, 내면의 자유는 보존한다.
Risks: 사후 세계나 매트릭스 주장은 검증 불가능하므로 사실로 단정하지 않는다.
Reframe: 기존 종교의 통제 구조는 걷어내되 핵심 수행 가치는 남긴다.
Positive conclusion: 새 시야는 모든 전통을 부정하는 것이 아니라, 맹종을 줄이고 자유와 자비를 키우는 실천 철학이다.
```

## Current Implementation

Added a lightweight evaluation/data contract:

- Module: `src/qtrm_mm/eval/critical_synthesis.py`
- Probe data: `data/eval/critical_synthesis_probe.jsonl`
- Trace builder: `src/qtrm_mm/training/critical_synthesis_data.py`
- Trace CLI: `scripts/103_build_critical_synthesis_traces.py`
- Generated trace data: `data/filtered/critical_synthesis_traces.jsonl`
- Tests: `tests/test_critical_synthesis_eval.py`

Added a Bongakgyo-specific expansion pass:

- Case generator: `src/qtrm_mm/training/bongak_critical_synthesis_cases.py`
- Case/trace CLI: `scripts/104_build_bongak_critical_synthesis_cases.py`
- Generated cases: `data/filtered/critical_synthesis_bongak_cases.jsonl`
- Generated traces: `data/filtered/critical_synthesis_bongak_traces.jsonl`
- Tests: `tests/test_bongak_critical_synthesis_cases.py`,
  `tests/test_bongak_cases_script.py`

The Bongakgyo rows are seed training candidates, not truth labels. The local
manual and summary are treated as doctrine-style sources. Their strong claims
must be converted into critical-synthesis targets that preserve useful practice
values while bracketing unverifiable metaphysics and warning against new dogma.

This is now a supervised trace shape, but not yet a completed model training
run. The generated rows are the first seed traces for future QTRM donor-residual
fine-tuning and held-out critical-synthesis evaluation.
