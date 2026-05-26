# 2026 Recursive, Memory, Context, And Search Papers

## Scope

This page records the May 2026 paper bundle that directly changes QTRM/MemoryOS
evaluation priorities:

- MeMo: Memory as a Model
- Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning
- Vector Policy Optimization: Training for Diversity Improves Test-Time Search
- LT2: Linear-Time Looped Transformers
- PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents
- Generalization Dynamics of LM Pre-training
- OPUS: Optimizer-induced Projected Utility Selection

Plain-language read:

```text
EqR and LT2 ask whether the model really gets better when it thinks longer.
MeMo and PEEK ask whether the system really uses memory as reusable context.
VPO asks whether the model can produce diverse candidates that search/verifiers
can exploit.
Generalization Dynamics asks whether a falling pre-training loss is hiding a
mode hop back into shallow pattern matching.
OPUS asks whether pretraining can spend tokens on examples whose optimizer-
shaped update actually points toward the target distribution.
```

## Papers

### MeMo: Memory as a Model

- arXiv: <https://arxiv.org/abs/2605.15156>
- Authors: Ryan Wei Heng Quek, Sanghyuk Lee, Alfred Wei Lun Leong, Arun Verma,
  Alok Prakash, Nancy F. Chen, Bryan Kian Hsiang Low, Daniela Rus, Armando
  Solar-Lezama.
- Date: 2026-05-14.

Claim:

```text
Encode new knowledge into a dedicated memory model while keeping the base LLM
parameters fixed.
```

QTRM/MemoryOS mapping:

- Strongly relevant to MemoryOS, LLM Wiki, and source-backed verifier design.
- Treats memory as more than a vector index: the memory subsystem should capture
  cross-document relationships and tolerate retrieval noise.
- Supports keeping Stage93 text ability separate from later MemoryOS updates.

Tests implied:

- MemoryOS answer quality with retrieval-only vs learned/compiled memory.
- Cross-document QA where the answer requires relationship synthesis.
- Noisy retrieval robustness and `UNKNOWN` handling.
- Source-backed verifier accuracy independent of answer fluency.

### Equilibrium Reasoners

- arXiv: <https://arxiv.org/abs/2605.21488>
- HTML: <https://arxiv.org/html/2605.21488v1>
- Authors: Benhao Huang, Zhengyang Geng, Zico Kolter.
- Date: 2026-05-20.

Claim:

```text
Generalizable iterative reasoning can be understood as convergence toward
task-conditioned attractors. Test-time compute scales along depth and breadth.
```

QTRM/MemoryOS mapping:

- Highest relevance to the QTRM recurrent core.
- Depth maps to more `think_steps` / core iterations.
- Breadth maps to multiple stochastic restarts or initial states.
- Fixed-point residual is a required telemetry signal, not decoration.

Tests implied:

- Depth ladder: `1, 2, 4, 8, 16` recurrent steps.
- Breadth ladder: multiple restarts per depth.
- Residual-selected top-1 vs average trajectory vs majority vote.
- Halt/ACT: easy cases should converge quickly; hard cases should consume more
  steps.
- Destructive ablations: core-off, state-frozen, residual-head-off, and
  transition-state-off.

### Vector Policy Optimization

- arXiv: <https://arxiv.org/abs/2605.22817>
- Authors: Ryan Bahlous-Boldi, Isha Puri, Idan Shenfeld, Akarsh Kumar, Mehul
  Damani, Sebastian Risi, Omar Khattab, Zhang-Wei Hong, Pulkit Agrawal.
- Date: 2026-05-21.

Claim:

```text
Training against vector-valued rewards improves diversity for test-time search,
pass@k, best@k, and evolutionary search.
```

QTRM/MemoryOS mapping:

- Relevant to generation verifier, reranking, candidate pools, and future
  post-training.
- The immediate test is not RL training; it is whether diverse QTRM candidates
  give the verifier/search loop anything useful to select.

Tests implied:

- Candidate diversity under fixed prompt and fixed checkpoint.
- `pass@k`, `best@k`, `verifier_selected@k`, and unique answer rate.
- Per-axis verifier scores: support, refute, missing, answer quality, repeat,
  stop, and task-specific reward.
- Diversity must improve selected quality, not just lexical variety.

### LT2: Linear-Time Looped Transformers

- arXiv: <https://arxiv.org/abs/2605.20670>
- Authors: Chunyuan Deng, Yizhe Zhang, Rui-Jie Zhu, Yuanyuan Xu, Jiarui Liu,
  T. S. Eugene Ng, Hanjie Chen.
- Date: 2026-05-20.

Claim:

```text
Looped transformers can replace quadratic full attention with linear/sparse
attention while preserving or improving looped-model quality.
```

QTRM/MemoryOS mapping:

- Highest relevance to efficient QTRM-native recurrent depth.
- GDN, sparse/window attention, and hybrid schedules are now direct architecture
  candidates rather than backend-only optimizations.
- The paper's GDN+DSA and Full+GDN hybrid results map cleanly to QTRM mixer
  ablations.

Tests implied:

- Mixer matrix: full attention, GDN/linear, sparse/window, GDN+sparse, Full+GDN.
- Long-context state tracking and controlled recall at multiple loop counts.
- Accuracy, tokens/sec, VRAM, and loop-count latency.
- Convert-or-continue tests for Stage93-like checkpoints before any multimodal
  graft.

### PEEK: Context Map as an Orientation Cache

- Hugging Face: <https://huggingface.co/papers/2605.19932>
- arXiv: <https://arxiv.org/abs/2605.19932>
- Blog: <https://zhuohangu.github.io/blog-post-peek/>
- Authors: Zhuohan Gu, Qizheng Zhang, Omar Khattab, Samuel Madden.
- Date: 2026-05-19.

Claim:

```text
Long-context agents benefit from a small persistent context map that records
reusable orientation knowledge about recurring external contexts.
```

QTRM/MemoryOS mapping:

- Strongly relevant to MemoryOS runtime, agent memory, and repo/corpus
  orientation.
- A context map is not a KV cache and not a raw summary. It is a small
  maintained map of what the external world contains and where useful evidence
  tends to live.
- This pairs naturally with MemoryOS: raw docs are the library; context map is
  the library floor plan.

Tests implied:

- Repeated-question corpus test with context-map on/off.
- Map budget ladder: 512, 1024, 2048 tokens.
- Stale-map and wrong-map adversarial tests.
- Metrics: source-backed hit, tool calls, retrieval calls, tokens, latency, map
  stale-error corrections.

### Generalization Dynamics of LM Pre-training

- Blog: <https://jiaxin-wen.github.io/blog/generalization-dynamics>
- Code/data: <https://github.com/Jiaxin-Wen/GDsuite>
- Authors: Jiaxin Wen, Zhengxuan Wu, Dawn Song, Lijie Chen.
- Date: 2026-05.

Claim:

```text
Pre-training can suddenly hop between shallow parrot-like algorithms and
generalizing algorithms. Ordinary loss and ordinary benchmarks can look smooth
while the generalization behavior flips.
```

QTRM/MemoryOS mapping:

- Directly relevant to Stage96D: full-data loss telemetry alone is not enough.
- Add cheap anti-parrot probes to checkpoint selection and OPUS-style data
  selection.
- Do not promote a checkpoint only because held-out PrefixLM loss improved.
  It must also prefer generalizing answers over tempting shortcut answers.

Tests implied:

- Flipped-label ICL: use the in-context rule rather than memorized sentiment.
- Repetitive/successive-answer ICL: solve the task rather than copy the answer
  pattern.
- Truthy facts: answer what is true rather than what sounds true.
- Intuitive-answer traps: prefer System-2 calculation over System-1 answer.
- Persona multi-hop: connect scattered facts into one coherent identity.

Local implementation:

```bash
PYTHONPATH=src python scripts/566_build_generalization_dynamics_probe.py \
  --out data/eval/generalization_dynamics_lite_probe.jsonl

PYTHONPATH=src python scripts/567_eval_blt_generalization_dynamics_probe.py \
  --checkpoint local_eval/.../last_model.pt \
  --sampled-data local_eval/.../sampled \
  --probe-jsonl data/eval/generalization_dynamics_lite_probe.jsonl \
  --out local_eval/.../generalization_dynamics_lite_report.json
```

Paper-fidelity audit:

```text
Current status:
  Two levels exist.

  1. GD-lite adaptation:
     data/eval/generalization_dynamics_lite_probe.jsonl
     6 hand-written smoke rows.

  2. Official GDsuite choice adapter:
     references/official/GDsuite
     data/eval/official_gdsuite_choice_probe.jsonl
     66,164 rows from the official HuggingFace data for the five logprob
     families.

Faithful to the source:
  - Uses the same plain-language question families:
      flipped answer,
      repetitive answer,
      successive answer,
      truthy answer,
      intuitive trap,
      persona multihop.
  - Scores intelligence answer vs parrot/shortcut answer by normalized
    log-probability margin.
  - Uses the gate for checkpoint/data-window acceptance rather than ordinary
    heldout loss alone.

Faithful upgrade added on 2026-05-26:
  - `scripts/623_build_official_gdsuite_choice_probe.py` imports the official
    Jiaxin-Wen/GDsuite prompt assembly rules and HuggingFace data.
  - It converts flipped, repetitive, successive, truthy, and intuitive-answer
    families into the BLT choice-probe schema.
  - `scripts/567_eval_blt_generalization_dynamics_probe.py` now labels these
    as `generalization_dynamics_official_choice_probe` and can write task
    margins/accuracy to TensorBoard.

Not faithful yet:
  - Official family 6, multi-hop persona QA, is generative + regex-based and
    still needs a custom BLT generation evaluator.
  - Does not perform long checkpoint sweeps to trace mode-hopping over a
    pretraining trajectory.

Allowed claim:
  "GD-lite anti-parrot smoke adapted from Generalization Dynamics" for the
  6-row file.

  "Official GDsuite choice-family adapter" for
  `official_gdsuite_choice_probe.jsonl`.

Forbidden claim:
  "The checkpoint passed full GDsuite" until the persona generative family and
  trajectory sweep are also handled.
```

### OPUS: Optimizer-Induced Projected Utility Selection

- arXiv: <https://arxiv.org/abs/2602.05400>
- Authors: Shaobo Wang, Xuan Ouyang, Tianyi Xu, Yuzheng Hu, Jialin Liu,
  Guo Chen, Tianyu Zhang, Junhao Zheng, Kexin Yang, Xingzhang Ren, Dayiheng
  Liu, Linfeng Zhang.
- Date: 2026-02-05, revised 2026-02-07.

Claim:

```text
Pretraining should move from more tokens to better tokens. Score candidate
documents by their optimizer-shaped projected utility against a stable target
direction, then sample useful/diverse data windows.
```

QTRM/MemoryOS mapping:

- Directly relevant to DGX pretraining efficiency.
- Use OPUS as a data-window selector, not as proof that a checkpoint
  generalizes.
- Pair it with Generalization Dynamics: a high-utility data window is accepted
  only if it improves held-out loss without reducing anti-parrot margin.
- Current implementation:
  `scripts/614_score_opus_projected_utility.py` creates the missing utility
  score file by projecting candidate optimizer-shaped updates against a stable
  proxy gradient direction. `scripts/555_prepare_byte_prefixlm_sample.py`
  consumes the score file through `--selection-mode utility`.
- Stage95 workflow:
  partial selection defaults to static `first`; when the full window is
  `utility`, partial training saves `last.pt` so the OPUS scorer can read AdamW
  optimizer state. The full sample is then selected from the partial model's
  optimizer-shaped update geometry.
- Boundary:
  this is OPUS-style projected data-window selection. It is not yet the full
  every-iteration online OPUS buffer inside the trainer.

Paper-fidelity audit:

```text
Current status:
  Two levels exist.

  1. OPUS-inspired offline data-window selection:
     `scripts/614_score_opus_projected_utility.py` scores rows before sample
     materialization.

  2. Online OPUS-style trainer selection:
     `scripts/557_train_blt_d_prefixlm_dataio.py --online-opus-enabled`
     draws candidate batches inside training, projects their updates against a
     proxy batch under the current optimizer state, and trains only the
     selected batch.

Faithful to the source:
  - Uses optimizer-induced update geometry when optimizer state exists.
  - Requires optimizer-bearing `last.pt` for AdamW-state preconditioning.
  - Computes a proxy direction and candidate directions.
  - Uses CountSketch-style projection for scalable alignment scores.
  - Materializes selected rows for pretraining and records the scorer report.
  - Online mode now scores a candidate buffer and selects the batch inside the
    same trainer step.

Not faithful yet:
  - Does not implement the paper's Ghost technique.
  - Does not implement Muon/hybrid optimizer geometry.
  - Online mode currently scores clean next-byte CE update geometry; configured
    auxiliary losses are applied only on the final selected update.
  - Online mode uses the trainer's eval/train proxy loader, not the full paper
    infrastructure for large rolling proxy buffers.
  - Offline mode remains a pre-materialized sample selector, not full OPUS.
  - Stage106D's `source_file_bucket + minimax_mean` is a local GD-protection
    modification, not a claim from the OPUS paper.

Allowed claim:
  "OPUS-style projected utility data-window selection" for the offline scorer.

  "Online OPUS-style candidate batch selection" for
  `--online-opus-enabled`.

Forbidden claim:
  "Paper-faithful full OPUS" until Ghost/Muon/hybrid optimizer geometry and the
  paper's complete rolling-buffer setup are implemented and verified.
```

Tests implied:

- Static balanced sample vs OPUS-like dynamic utility sample.
- Held-out PrefixLM loss, language samples, and GD-lite margin at the same
  token budget.
- Reject if the selected data improves CE while making the model more
  shortcut-prone.

## Unified Test Priority

1. EqR depth/breadth convergence on Stage93 or the nearest recurrent checkpoint.
2. PEEK/MeMo MemoryOS context-map vs retrieval-only recurring-corpus eval.
3. VPO-style candidate diversity plus verifier-selected `best@k`.
4. Generalization Dynamics anti-parrot checkpoint/data-selection gate.
5. OPUS-style data-window selection only when GD margin is logged.
6. LT2 mixer/loop efficiency once depth scaling is worth optimizing.

Do not claim broad model intelligence from any single paper-aligned smoke. Each
paper contributes one falsifiable axis.
