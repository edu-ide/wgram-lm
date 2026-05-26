# 2026 Paper-Driven Test Plan

## Decision

The May 2026 paper bundle updates the immediate QTRM/MemoryOS test matrix. The
next evaluations must separate four claims:

```text
1. The recurrent core improves with more thinking.
2. MemoryOS improves repeated work over the same external context.
3. Candidate diversity gives verifiers/search useful options.
4. Loop/mixer changes make recurrent depth cheaper without hiding regressions.
5. Falling loss is not hiding a mode-hop back into parrot-like shortcuts.
```

Source bundle:

- [2026 Recursive, Memory, Context, And Search Papers](../sources/2026-recursive-memory-context-papers.md)
- [2026 Local Depth / DGX Pretrain Generalization Split](2026-local-depth-dgx-pretrain-generalization-split.md)

## Resource Split

```text
Local 4090:
  Depth-scaling architecture tests. Use it to prove or reject that thinking
  longer helps.

DGX:
  Pretraining efficiency and generalization tests. Use it to compare data
  windows, OPUS-style selection, and Generalization Dynamics margins.
```

Plain-language rule:

```text
Local asks: "Does more thinking time help this student?"
DGX asks: "Does more reading make this student understand, or just parrot?"
```

Do not promote a DGX run if the corresponding local depth path is still
architecture-rejected. Do not promote a low-loss DGX checkpoint if GD-lite
margin regresses.

## Test 1: EqR Depth/Breadth Gate

Purpose:

```text
Prove or reject that QTRM gets better because iterative latent dynamics
converge, not because one lucky generation happened.
```

Input rows must include:

```text
case_id or id
depth or mode=qtrm_core_steps_N...
restart_id
completion
hit or generation_hit
fixed_point_residual or convergence_residual
```

Command:

```bash
PYTHONPATH=src python scripts/548_build_depth_breadth_probe_report.py \
  --rows local_eval/.../depth_restart_rows.jsonl \
  --out local_eval/.../depth_breadth_report.json
```

Accept signals:

- Deep depth improves over shallow depth.
- Residual-selected top-1 beats average trajectory.
- Hits have lower residual than misses.
- Core-off / state-frozen / residual-head-off ablations drop.

Reject signals:

- All depths produce the same completions.
- Residual is not correlated with correctness.
- Core-off matches full QTRM.

## Test 2: MemoryOS Context Map Gate

Purpose:

```text
Test the PEEK/MeMo claim that reusable orientation memory improves repeated
questions over the same corpus.
```

Corpus options:

- repo wiki/docs corpus;
- MemoryOS evidence corpus;
- held-out synthetic cross-document corpus.

Modes:

```text
retrieval_only
chat_summary_only
memoryos_context_map
memoryos_context_map_stale
memoryos_context_map_wrong
```

Metrics:

- answer hit;
- source-backed hit;
- retrieval/tool calls;
- input tokens;
- latency;
- stale-map correction rate;
- `UNKNOWN` correctness when sources are insufficient.

Accept signals:

- Context map improves source-backed hit at equal or lower retrieval/tool cost.
- Wrong or stale map is corrected by source evidence.
- Map budget remains bounded.

Reject signals:

- Map text becomes a hallucination amplifier.
- Map improves fluency but not source-backed correctness.
- Map grows without eviction discipline.

## Test 3: VPO Candidate-Diversity Gate

Purpose:

```text
Test whether QTRM generation produces diverse candidates that a verifier can
use, before committing to vector-reward RL.
```

Metrics:

- unique answer rate;
- pass@k;
- oracle@k;
- verifier-selected@k;
- support/refute/missing score spread;
- repeat/stop/quality verifier score spread.

Accept signals:

- Increasing `k` improves verifier-selected accuracy.
- Diversity appears in solution strategy or evidence path, not just wording.
- Verifier-selected candidates beat raw greedy.

Reject signals:

- Candidate pool collapses to near-duplicates.
- Diversity rises while selected quality does not.
- Verifier prefers fluent unsupported answers.

## Test 4: LT2 Mixer/Loop Gate

Purpose:

```text
Measure whether efficient looped mixers preserve depth-scaling gains while
reducing compute.
```

Mixer matrix:

```text
full_attention_loop
gdn_linear_loop
sparse_window_loop
gdn_sparse_hybrid
full_gdn_hybrid
```

Metrics:

- held-out loss / accuracy;
- controlled recall;
- state-tracking hit;
- tokens/sec;
- VRAM;
- latency per loop count.

Accept signals:

- Hybrid or linear/sparse mixer keeps depth/breadth gains.
- Runtime or memory improves enough to matter.
- Long-context state tracking does not regress.

Reject signals:

- Efficient mixer only wins speed while destroying looped reasoning.
- Full-attention baseline remains better at the same wall-clock budget.

## Test 5: Generalization Dynamics Anti-Parrot Gate

Purpose:

```text
Catch mode-hopping: the model may lower held-out loss while switching back to a
shallow shortcut algorithm.
```

Immediate local gate:

```bash
PYTHONPATH=src python scripts/566_build_generalization_dynamics_probe.py \
  --out data/eval/generalization_dynamics_lite_probe.jsonl

PYTHONPATH=src python scripts/567_eval_blt_generalization_dynamics_probe.py \
  --checkpoint local_eval/.../last_model.pt \
  --sampled-data local_eval/.../sampled \
  --probe-jsonl data/eval/generalization_dynamics_lite_probe.jsonl \
  --out local_eval/.../generalization_dynamics_lite_report.json
```

Metrics:

- intelligence-vs-parrot normalized logprob margin;
- per-task margin;
- accepted only when every valid row has positive margin;
- run alongside held-out loss, generation gate, and depth gate.

Accept signals:

- The checkpoint prefers intelligence answers over parrot answers on all rows.
- Margins improve or stay stable when training continues.
- OPUS/data-selection windows that improve held-out loss do not reduce the
  anti-parrot margin.

Reject signals:

- Held-out loss improves while anti-parrot margin flips negative.
- The model prefers repeated/successive pattern answers over solving the task.
- The model prefers intuitive answers over System-2 answers.

## Test 6: OPUS-Style Data-Window Efficiency Gate

Purpose:

```text
Test whether a data window is useful for the optimizer and still preserves
generalization.
```

Immediate gate:

```text
static/balanced data window
vs
OPUS-like utility-selected data window
```

Metrics:

- held-out PrefixLM loss at the same token budget;
- language generation samples;
- GD-lite intelligence-vs-parrot margin;
- family/facet regressions;
- tokens/sec and additional selection overhead.

Accept signals:

- Utility-selected data improves held-out loss at equal or lower token budget.
- GD-lite margin improves or stays positive.
- Generation samples improve rather than becoming repetitive.

Reject signals:

- Utility selection overfits to easy token loss.
- GD-lite margin worsens while CE improves.
- Selection overhead consumes the saved training time.

## Immediate Order

1. Run the EqR depth/breadth report on the smallest available Stage93 or
   recurrent-checkpoint eval rows.
2. Build a tiny repo-wiki context-map eval for PEEK/MeMo.
3. Run candidate-diversity sampling on the same prompts.
4. Run Generalization Dynamics anti-parrot gate on every promoted checkpoint.
5. Run OPUS-style data-window selection only with GD-lite logged.
6. Only then spend GPU time on LT2-style mixer substitutions.

Plain-language rule:

```text
First test whether thinking helps.
Then test whether memory helps.
Then test whether search helps.
Then test whether lower loss still means generalization.
Then test whether better data windows improve both efficiency and
generalization.
Only optimize the loop engine after those signals are real.
```
