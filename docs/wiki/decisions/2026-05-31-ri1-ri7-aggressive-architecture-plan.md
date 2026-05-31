# 2026-05-31 RI-1~RI-7 Aggressive Architecture Plan

## Goal

Build a model that is both:

```text
good at language
good at actual latent reasoning
```

under the project rule:

```text
promotion = free generation only
forced-choice / candidate-rerank / oracle pass@K = historical diagnostics only
```

The target is not a prettier loss curve. The target is decoded answers that
improve when the claimed reasoning mechanisms are active and regress when they
are ablated.

## Plain-Language Diagnosis

The current BLT/IMTA line has the right ambition but the wrong bottleneck.

It wants to work like this:

```text
read a compressed paragraph
think over the compressed meaning
try several internal solution paths
write the answer with one normal mouth
```

But the active hnet path is still too close to this:

```text
read only the boundary word of each paragraph
think over those boundary words
ask a tiny byte residual to recover the missing details
write an answer
```

That is why training loss can fall while free generation remains weak. The
model is not just undertrained; its thinker is under-informed.

## Paper Axes Used

Primary methodology sources already tracked in the wiki:

- Byte Latent Transformer / tokenizer-free dynamic patching:
  <https://arxiv.org/abs/2412.09871>
- Fast Byte Latent Transformer:
  <https://arxiv.org/abs/2605.08044>
- Gated DeltaNet family:
  <https://arxiv.org/abs/2412.06464>
- Gated DeltaNet-2:
  <https://arxiv.org/abs/2605.22791>
- Memory Sparse Attention:
  <https://arxiv.org/abs/2603.23516>
- Titans long-term neural memory:
  <https://arxiv.org/abs/2501.00663>
- Equilibrium Reasoners / attractor-style reasoning:
  <https://arxiv.org/abs/2605.21488>
- LT2 / looped transformer stability:
  <https://arxiv.org/abs/2605.20670>
- Learn from your own latents:
  <https://arxiv.org/abs/2605.27734>

Local synthesis source:

- [Latest Recurrent-Memory Substrate Papers](../sources/2026-latest-recurrent-memory-substrate-papers.md)
- [Raw Intelligence Necessary Conditions](raw-intelligence-necessary-conditions-2026-06.md)
- [BLT/IMTA Architecture Rationality Review](2026-05-31-blt-imta-architecture-rationality-review.md)

## Non-Negotiable Architecture Principles

1. **One mouth**
   Final answer must go through `hnet_causal_speaker` and the same LM head.

2. **Informed thinker**
   The recurrent core must receive causal chunk summaries, not only boundary
   byte embeddings.

3. **Internal breadth, not answer roulette**
   IMTA/GRAM/PTRM means multiple latent trajectories inside the body, not
   external answer candidates selected after the fact.

4. **Latent grammar auxiliary, not answer bypass**
   Own-latent prediction teaches hidden state structure. It does not replace
   token CE or free-generation evaluation.

5. **Memory inside the loop**
   RI-4 memory must be sparse/selective and causally injected into the recurrent
   computation, not an external retrieval note.

6. **Ablations decide truth**
   Every RI claim must have a matching off-switch that hurts decoded free
   generation or heldout reasoning.

## P0: Fix The BLT Information Bottleneck

### Current bug-shaped design flaw

`_hnet_boundary_states` currently does this:

```text
selected_embeddings = byte_embeddings at boundary positions
```

That is too weak for BLT. BLT patching needs the latent state to represent the
chunk, not only the marker where the chunk begins.

### Required replacement

Implement causal chunk summaries:

```text
for each selected boundary b:
  start = previous_boundary[b] + 1, or 0 for the first boundary
  end = boundary[b] + 1
  chunk_byte_states = byte_embeddings[start:end]
  chunk_summary = learned projection(
      boundary byte,
      causal recency-weighted mean of bytes already present
  )
  selected_embeddings[b] = chunk_summary
```

Constraints:

- No future chunk may influence an earlier boundary state.
- During autoregressive generation, a token position may only see boundary
  summaries whose bytes are already present in the input prefix.
- Non-boundary bytes inside a chunk must affect logits through the latent route.

### Tests

Add tests that prove:

- changing a non-boundary byte changes `selected_embeddings`,
- changing a non-boundary byte changes final hnet logits when byte residual is
  near zero,
- chunk summary path still respects UTF-8 and dynamic patch constraints,
- K=1 and K=3 both use the same chunk summary path.

## P1: Make IMTA A Real Internal Search Loop

Original IMTA was useful but too soft:

```text
learned offsets + optional noise + soft selector average
```

Implemented upgrade:

1. **Diversity-preserving trajectories**
   - per-trajectory latent adapters are active,
   - trajectory noise remains latent-only,
   - diversity telemetry and optional diversity loss are active.

2. **Answer-causal selector**
   - selector observes trajectory state plus shallow speaker-space state,
   - selector aggregates hidden states before `hnet_causal_speaker`,
   - no decoded candidate strings are produced before final generation.

Gate:

```text
K=3 free generation > K=1 free generation
with similar parameter budget and same data
```

## P2: Own-Latent Prediction Over Better States

Keep current own-latent auxiliary, but move its strongest target after P0:

```text
chunk_summary/core_state[t] -> chunk_summary/core_state[t+1]
```

Then add depth-aware prediction:

```text
core_state at think depth d -> core_state at depth d+1
```

This directly maps arXiv:2605.27734 into the active BLT body.

Gate:

```text
own-latent on improves:
  latent cosine / prediction loss
  same-head free generation
  depth-sweep stability

own-latent off loses that lift
```

## P3: RI-1 Depth Scaling Without Collapse

Current answer-state attractor is too expensive and caused CUDA OOM in the
long run. Replace heavy full-vocab per-depth CE with a cheaper schedule:

```text
depths = 1, 2, 4, 8
sample limited target positions
compute CE only on sampled labels
track fixed-point residual / halt / state distance
```

Required evidence:

```text
free generation at depth 4/8 > depth 1
recurrence-off loses the gain
loop residual decreases or stabilizes
```

## P4: RI-4 Sparse Memory Inside The Body

Do not bolt retrieval onto the prompt.

Add a small sparse latent memory interface to the BLT/IMTA body:

```text
chunk/core states
-> sparse top-k slot read
-> recurrent core / IMTA trajectory conditioning
-> gated write after answer-facing state update
```

This maps MSA/Titans to the local architecture:

- MSA supplies sparse selection/routing.
- Titans supplies surprise/write policy intuition.
- Project 5.56 supplies rehearsal/protection/decay ablations.

Required ablations:

```text
memory on
router off
persistence off
chunk shuffle
distractor insertion
```

## Optional Historical Validation: 5.56 Causality Matrix

The 5.56 matrix is not a required runtime component of the BLT/IMTA model.
It is a historical-validation ablation for checking whether the old
Adaptive-Rehearsal recipe still contributes on the new substrate:

```text
full
stochastic breadth off
gold/structural injection off
attractor protection off
decay disabled
```

If run, it must be measured on free-generation and no-retrieval heldout tasks.
Do not block the core architecture work on this matrix.

## P6: Language Quality Track

Reasoning machinery is not enough. The model also needs a stable language
mouth.

Language plan:

1. Keep `hnet_causal_speaker` as the only final mouth.
2. Use byte/BPE compatibility only as data/tokenizer infrastructure, not as a
   second answer path.
3. Train on mixed:
   - verified math binding,
   - short instruction following,
   - code/math text,
   - denoise/recovery examples,
   - compact natural-language explanations.
4. Evaluate only by free generation:
   - exact answer,
   - normalized answer,
   - EOS rate,
   - repetition loop rate,
   - decoded sample review.

## P7: RI-1~RI-7 Promotion Matrix

| RI | Required proof |
|---|---|
| RI-1 depth scaling | depth 1/4/8 free-generation ladder improves; recurrence-off drops |
| RI-2 long-horizon stability | perturb/carry/horizon tests do not collapse over 80-200 steps |
| RI-3 mechanism causality | turning off claimed mechanisms hurts free generation; 5.56 matrix is optional historical validation |
| RI-4 sparse memory | router/memory on beats off; chunk-shuffle/distractor tests pass |
| RI-5 hybrid synergy | recurrence + attention sync beats recurrence-only and attention-only |
| RI-6 low training waste | ablations hurt during training, not only at final checkpoint |
| RI-7 data efficiency | matched-data runs beat weaker substrate under same budget |

## Immediate Execution Order

Do this next, before any more long run:

```text
1. Implement causal chunk summary in hnet_dechunk.
2. Add non-boundary-byte causality tests.
3. Run 2-step smoke.
4. Run free-generation-only smoke on 8-32 rows.
5. Compare:
   K=1 own-latent off
   K=3 own-latent off
   K=3 own-latent on
6. Only if free-generation movement appears, launch longer training.
```

Current implementation status:

```text
1. causal chunk summary: implemented
2. non-boundary-byte causality tests: implemented
3. same-body IMTA adapters/diversity/speaker-space selector: implemented
4. free-generation smoke: required before a new long run is trusted
5. K=1/K=3/own-latent ablation: still required
```

## Kill Criteria

Stop and redesign if:

- loss improves but free generation remains flat,
- K=3 does not beat K=1 after chunk-summary fix,
- own-latent improves latent loss but not same-mouth generation,
- recurrence depth improves teacher-forced metrics only,
- memory-on gains disappear under chunk-shuffle or distractor tests.

This plan is intentionally aggressive, but it is not magic-label aggressive.
Every mechanism must earn its place through decoded free generation and clean
causal ablations.
