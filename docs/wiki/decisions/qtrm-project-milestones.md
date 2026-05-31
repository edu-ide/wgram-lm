# QTRM/PV-GRAM Project Milestones

Date: 2026-05-24.

Status: active project map.

## What A Milestone Means

A milestone is not a hope, a run name, or a paper-inspired acronym. It is a
capability gate.

Plain-language definition:

```text
A milestone is a door the model has actually walked through.
To count, the model must show the behavior through the normal answer path, and
the relevant ablation/eval must prove that the claimed organ caused the gain.
```

Operating rule:

```text
Do not claim the next milestone from train loss alone.
Do not claim it from a side probe, hard-coded router, external solver, or
teacher-forced-only metric.
```

## Current North Star

Build a native QTRM/PV-GRAM model that grows from:

```text
read text -> think recurrently -> speak through one LM head
-> reason across languages
-> see images/documents/charts through the same answer path
-> use tools and memory across turns
-> handle Codex-like long-horizon coding/agent tasks
-> compete with much larger Qwen-class baselines on public reasoning/agentic gates
```

Plain-language version:

```text
First grow a person who can read, think, and speak.
Then give that same person more languages.
Then attach eyes.
Then teach office/tool procedure.
Then teach long research/coding work.
Only after that compare against large general agents.
```

## Milestone Ladder

| ID | Capability | Plain-Language Meaning | Status | Promotion Gate |
|---|---|---|---|---|
| M0 | One-body text path | One student reads, thinks, and speaks through one body. | accepted as architecture direction | Native PrefixLM path exists; answer flows through token embeddings, recurrent thought/backbone, and LM logits. |
| M1 | Small-handout learning proof | The student can learn from a thin worksheet without collapsing. | accepted / Stage92 finished | Stage92 reached step `24000` with final `eval_loss=0.4216`, finite eval, checkpoints saved. |
| M2 | Large HRM-Text reasoning continuation | Same student receives a larger textbook, not a new body. | in progress | Stage93 Data-IO tokenization/sample complete; Stage93 continues from Stage92/last_model; heldout generation and eval loss remain healthy. |
| M3 | Multilingual text reasoning | Same reasoning spine answers in more languages. | not yet claimed | `PROFILE=multilingual_curriculum` run plus Korean/translation/XQuAD probe through normal LM path. |
| M4 | Tokenizer/multilingual efficiency audit | The mouth can pronounce languages without shredding every sentence into dust. | warning open | Korean fragmentation is currently warned; pass requires post-M3 Korean generation plus acceptable token fertility or a tokenizer redesign plan. |
| M5 | Public language/reasoning benchmark readiness | The model can sit a public exam, not only its own worksheets. | pending | OpenAI/Qwen-compatible eval suites or local public-target gates show valid non-echo answers and stable scoring. |
| M6 | Native multimodal graft | Attach eyes to the same text-thinking person. | planned | Stage94 visual reader/projector/resampler uses Stage93 text spine; visual ablation removes visual-task gain while text regression stays controlled. |
| M7 | Memory/retrieval/MSA reasoning | The model can use a case file, not just its immediate prompt. | planned | Retrieval/evidence affects the normal answer path; evidence-off, memory-off, router-off, or chunk-shuffle ablation removes the gain. |
| M8 | Multi-turn tool-use agent | The student can choose a tool, fill arguments, read the result, and continue. | planned | Stateful tool simulations pass single-call JSON, argument carryover, observation-grounded final answer, and safety/refusal gates. |
| M9 | Codex-like long-horizon coding agent | The student can work in a sandbox with files, tests, logs, and progress memory. | planned | Small repo tasks pass inspect/edit/run/observe/compact cycles; then SWE-Bench-style gates. Single JSON function call is insufficient. |
| M10 | Qwen3.6-27B target comparison | Smaller native model competes with a much larger baseline on matched public tasks. | long target | Requires M2-M9 evidence plus matched scorer discipline from the 27B benchmark milestone page. |

## Where MSA Fits

MSA belongs primarily to **M7**.

Plain-language placement:

```text
M2/M3:
  teach the student to read, think, speak, and answer across languages.

M6:
  attach eyes to the same student.

M7:
  give the student a huge indexed case file and teach them to pull the right
  pages into working memory.

M8/M9:
  teach the student to use tools and work for a long time; MSA helps as the
  case-file memory, but it is not itself tool use.
```

Technical placement:

```text
MSA = sparse latent memory routing over large document/memory pools.
It is not the base text spine, not the multilingual curriculum, and not the
visual reader. It is the scalable memory organ.
```

Therefore:

```text
canonical milestone:
  M7 Memory/retrieval/MSA reasoning

supporting role:
  M8 multi-turn tool-use agent memory
  M9 Codex-like long-horizon coding workspace memory

not allowed:
  using MSA to skip M2/M3 language stability,
  using MSA to skip M6 visual graft,
  claiming 100M-token reasoning from storage capacity alone.
```

Suggested M7 sub-gates:

```text
M7A external MemoryOS retrieval:
  retrieval/rerank recall is good enough and evidence enters the canonical
  prompt or workspace in an auditable way.

M7B memory causality:
  memory_on beats memory_off on held-out memory/evidence questions, and the
  same answers flow through the normal LM path.

M7C MSA sparse routing:
  MSA/router_on beats router_off and chunk_shuffle/doc-id corruption on
  scattered multi-hop memory tasks.

M7D MSA + QTRM integration:
  MSA+QTRM beats QTRM-without-MSA and MSA-only on the same memory tasks.
```

Current MSA status:

```text
reference:
  docs/wiki/sources/memory-sparse-attention.md

scaffold/fork track:
  docs/wiki/decisions/qwen35-full-msa-fork.md

status:
  reference and scaffold exist, but MSA is not yet a promoted capability.
```

## Stage95 BLT Foundation Placement

Stage95 is a possible new M2/M3 foundation route, not an agentic milestone.

Plain-language placement:

```text
Stage92/93 BPE:
  the current student learned through a BPE mouth.

Stage95 BLT:
  raise a new byte-latent student who reads raw UTF-8 bytes, folds them into
  latent notes, thinks over the shorter note stream, and speaks bytes back out.

M8/M9 agentic:
  comes later, after the student can already speak, reason, and remember.
```

Non-negotiable Stage95 curriculum:

```text
general language first:
  ordinary dialogue, instruction, QA, summarization, narrative/world text

plus reasoning:
  natural reasoning, science/common sense, symbolic tasks

plus math:
  GSM/math/Numina/OpenMath/SYNTH style rows

plus multilingual:
  Korean/English priority, translation, XQuAD/MLQA/TyDi/XNLI-style reading

plus memory/context:
  long QA, multi-document, summarization/evidence rows
```

Reject a Stage95 launch if it is math-only, reasoning-only, or tool-trace-heavy
before ordinary language gates pass. In plain language: do not train another
student who can fill a worksheet but cannot answer a normal sentence.

Preparation entrypoint:

```text
scripts/558_prepare_stage95_blt_foundation_byte_sample_dgx.sh
```

It builds tokenizer-free UTF-8 byte PrefixLM samples from both JSONL and parquet
sources via `scripts/555_prepare_byte_prefixlm_sample.py --source-globs`.
The plan output is part of the gate: if it does not show all broad shelves, do
not launch the long from-scratch run.

## Current Evidence Snapshot

As of 2026-05-24:

```text
M0:
  active native PrefixLM trainer:
    scripts/534_train_native_prefixlm_dataio.py

M1:
  DGX Stage92:
    /mnt/data4tb/wgram-lm/local_eval/20260523_STAGE92_DGX913M_BS8_CONTINUE_TO24K
  final seen:
    step=24000
    eval_loss=0.421623428875682
    eval_nonfinite_batches=0
    eval_unresolved_nonfinite_batches=0
  checkpoint:
    last.pt exists
    last_model.pt exists

M2:
  Stage93 large data build:
    /mnt/data4tb/wgram-lm/local_eval/stage93_hrm_text_reasoning_nonflan_dataio
  latest checked:
    2026-05-24 01:46 KST
    tokenized=1010 / 1010 files
    full sampled data has started appearing but is not supervisor-ready yet
    micro hardlink sampled data is continuing from one already tokenized task
    Stage93 overnight supervisor wrapper is active on DGX, pid=967461
    Stage93 inner supervisor is active on DGX, current seen pid=967485
    Stage93 supervisor micro-hardlink fallback target is now 40000 steps
    to avoid GPU idle time if full sampling is still pending at step 30000.
    Stage93A00 micro hardlink trainer is active on DGX, current seen pid=920757
    Stage93A00 trainer has no controlling TTY and is parented by systemd,
    so it is not tied to the current SSH session.
    Stage93A00 latest seen:
      previous segment reached step=24500
      eval_loss_at_step_24500=0.476658
      continuation resumed from last_model.pt at start_step=24500
      continuation latest seen step=25700
      continuation train_loss_at_step_25700=0.1495388150215149
      eval_loss_at_step_25700=0.16637434939451246
      last_model.pt exists
  status:
    micro hardlink run keeps GPU studying while the full book prints;
    full continuation launches automatically when sampled/tokens.npy and
    metadata/index files are stable, continuing from the best checkpoint.
    post-target language/raw-intelligence gates are now attached to the
    supervisor wrapper and write:
      language_heldout_loss.json
      multilingual_generation_probe.json
      raw_intelligence_suite.json

M3/M4:
  multilingual probe/evaluator added:
    data/eval/prefixlm_multilingual_probe.jsonl
    data/eval/prefixlm_language_heldout.jsonl
    scripts/542_eval_prefixlm_multilingual_probe.py
    scripts/543_audit_prefixlm_multilingual_tokenizer.py
    scripts/544_eval_prefixlm_language_heldout_loss.py
    scripts/545_run_prefixlm_language_gates_dgx.sh
    data/eval/prefixlm_raw_intelligence_probe.jsonl
    scripts/546_eval_prefixlm_raw_intelligence_suite.py
  interpretation:
    micro eval_loss around 0.3 means the thin chapter is almost memorized;
    it does not prove free language ability.
    free-language evidence requires full/mixed heldout loss plus generation
    probe outputs through the normal PrefixLM answer path.
  raw-intelligence axes now measured:
    language
    reasoning_arithmetic
    symbolic_manipulation
    context_memory
    instruction_following
    multilingual
    tool_use_format
    multiturn_state
  first tokenizer audit:
    status=warn
    language_over_threshold=ko
    ko max tokens_per_nonspace_char=2.6667
```

## Non-Negotiable Ordering

```text
Do not start M6 multimodal from scratch before preserving the strongest M2/M3
text spine.

Do not claim M8/M9 agent ability from ordinary chat or one-shot function-call
formatting.

Do not claim M10 public-baseline victory until the answer path, benchmark
prompting, scorer, and ablations match the public-target contract.
```

## Current Next Actions

1. Wait for Stage93 Data-IO tokenization and sampling to finish.
2. Launch Stage93 continuation only after:
   `sampled/tokens.npy` exists and no active PrefixLM training process holds the
   DGX GPU.
3. Run Stage93 heldout generation/eval gates.
4. Run multilingual curriculum continuation or build the multilingual sampled
   profile if M3 is the active next target.
5. Run:

   ```text
   scripts/542_eval_prefixlm_multilingual_probe.py
   scripts/543_audit_prefixlm_multilingual_tokenizer.py
   ```

6. Promote to Stage94 multimodal graft only after text regression is stable.

## Related Milestone Pages

- [QTRM-Native 2B/3B vs Qwen3.6-27B Milestones](qtrm-native-27b-benchmark-milestones.md)
- [Training Diagnostics](../concepts/training-diagnostics.md)
- [Multimodal Architecture](../../MULTIMODAL_ARCHITECTURE.md)
- [Training Workflow](../../TRAINING_WORKFLOW.md)
