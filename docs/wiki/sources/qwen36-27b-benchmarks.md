# Qwen3.6-27B Benchmark Targets

Date: 2026-05-15.

Status: target baseline for QTRM-Native-2B/3B milestone gates.

Primary source:

- `https://huggingface.co/Qwen/Qwen3.6-27B`

Secondary summary sources used only for cross-checking:

- `https://benchlm.ai/models/qwen3-6-27b`
- `https://awesomeagents.ai/models/qwen-3-6-27b/`

Agent benchmark sources:

- `https://www.swebench.com/`
- `https://www.tbench.ai/`
- `https://gorilla.cs.berkeley.edu/leaderboard`
- `https://www.tau-bench.com/`
- `https://huggingface.co/gaia-benchmark`
- `https://qwenlm.github.io/Qwen-Agent/en/benchmarks/`

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

## Official Recognition Ladder

문과적으로 말하면, synthetic OOD는 "우리 반 시험"이고 public benchmark는
"전국 시험"이다. 전국 시험에 올리려면 세 종류의 성적표가 필요하다.

1. 지식/수학/추론 성적표:
   MMLU-Pro, GPQA Diamond, AIME, HMMT처럼 모델 혼자 답을 내는 시험.

2. 코딩 에이전트 성적표:
   SWE-bench Verified/Pro/Multilingual, Terminal-Bench 2.0, NL2Repo처럼
   모델이 실제 작업장을 움직여 결과를 만들어야 하는 시험.

3. 도구/장기 에이전트 성적표:
   BFCL V4, tau-bench, GAIA, Qwen-Agent DeepPlanning처럼 모델이 함수 호출,
   여러 턴의 사용자-도구 상호작용, 검색/계획/제약 만족을 해내는 시험.

The local project rule is:

```text
No public recognition claim from synthetic OOD alone.
No Qwen3.6-27B comparison claim from a private prompt suite alone.
No agent claim without tool-call / terminal / multi-turn benchmark evidence.
```

The executable manifest rule is stricter:

```text
official_agent_claim_ready = true only when accepted official-harness reports
exist for all three categories:

1. coding_or_terminal:
   SWE-bench Verified/Pro/Multilingual, Terminal-Bench 2.0, or NL2Repo

2. tool_calling:
   BFCL V4

3. long_horizon_workflow:
   tau-bench, GAIA, or Qwen-Agent DeepPlanning
```

문과적으로는, "코딩 시험 하나 잘 봤다"는 아직 에이전트가 아니다. 진짜
에이전트 주장을 하려면 작업장에서 고치고, 도구를 정확히 부르고, 여러 턴
동안 계획을 유지하는 세 과목을 모두 통과해야 한다.

## Agent Benchmark Ladder

| Benchmark | Recognition role | Qwen3.6 public target in this manifest? | Required action |
|---|---|---:|---|
| SWE-bench Verified | coding-agent repair credibility | yes, 77.2 | run same official/scaffolded harness before claiming coding-agent parity |
| Terminal-Bench 2.0 | real terminal task completion | yes, 59.3 | run official Harbor/Terminal-Bench harness or clearly mark as not comparable |
| BFCL V4 | function/tool-call correctness | no | add OpenAI-compatible tool-call serving and run official BFCL scorer |
| tau-bench | multi-turn tool-agent-user workflow | no | run with fixed user simulator and environment; report success rate and tool errors |
| GAIA | general assistant agent problem solving | no | run official split/leaderboard protocol; report retrieval/tool budget |
| Qwen-Agent DeepPlanning | planning and constraint satisfaction | no | use as Qwen-agent-aligned planning diagnostic, not as a Qwen3.6 target score |

The local manifest builder stores this in `agent_recognition_claim` and keeps
`accepted=false` until the required official artifacts are supplied.

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
