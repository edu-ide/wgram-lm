# 2026-05-31 BLT/IMTA Architecture Rationality Review

## Verdict

Current architecture is a reasonable research scaffold, but it is not yet a
reasonable final reasoning-language architecture.

In plain language:

```text
The system is trying to be an editor that reads a compressed draft, thinks over
it, then writes the answer.

But the active hnet_dechunk implementation often gives the thinker only the
boundary token of each chunk, not a strong summary of the whole chunk.

So the editor is asked to write from headings and first words, while the actual
paragraph content is mostly left to a small byte residual.
```

That explains the observed behavior:

```text
loss can improve
boundaries can look stable
loop rate can be reduced by decoding controls
but free generation still does not produce reliable answers
```

## Code-Level Findings

### What Is Reasonable

The PrefixLM data shift is structurally correct:

```text
input = instruction + response[:-1]
labels = ignored instruction labels + response
```

The active answer path is one-mouth:

```text
token state -> hnet_causal_speaker -> tied LM head
```

There is no promoted external candidate answer table in the active path.

The recurrent global core is causal for attention blocks:

- `scripts/335_train_qtrm_native_etd_probe.py` builds a causal mask in
  `_forward_embedded_impl`.
- attention mixers use that mask.

IMTA/GRAM-PTRM breadth is now at least executable:

```text
selected boundary states
-> K learned-offset/noisy latent trajectories
-> internal selector/aggregation
-> hnet_causal_speaker
```

Own-latent prediction is also executable as an auxiliary:

```text
selected_hidden[t] + own_latent_predictor(...)
  predicts detached selected_hidden[t+1]
```

This is compatible with arXiv:2605.27734 as an auxiliary, not as an answer
bypass.

### Main Architectural Problem

In `BLTDByteLatentPrefixLM._hnet_boundary_states`, the hnet path selects the
embedding at boundary positions:

```text
selected_embeddings[row_idx, :len(row_positions)] =
    byte_embeddings[row_idx, index_tensor]
```

This means the global recurrent core sees boundary byte embeddings, not a robust
summary of the full bytes inside each chunk.

But the one-body speaker then uses:

```text
token_hidden =
  sigmoid(byte_gate) * byte_embeddings
  + sigmoid(latent_gate) * bridged_latent
```

With the current long-run setting:

```text
byte_gate_init = -4.0
latent_gate_init = 4.0
```

the direct byte route is intentionally tiny and the latent route dominates.

So the dominant route is also the route with incomplete chunk information.

This is the most important architecture gap.

## Rationality Matrix

| Criterion | Status | Read |
|---|---|---|
| PrefixLM causal data contract | Good | Shifted response prediction is correct. |
| One-mouth answer path | Mostly good | hnet causal speaker is the only normal answer mouth. |
| BLT compression | Weak | Current hnet path does not summarize full chunks strongly enough. |
| Dynamic boundary | Partly good | UTF-8/min/max chunk constraints exist, but hard Python-loop boundary selection is not a clean differentiable compression policy. |
| Gated DeltaNet2 + attention 3:1 | Mostly good | Backbones are wired; attention is causal, delta route relies on causal mixer semantics. |
| IMTA / GRAM-PTRM | Partial | K trajectories exist, but selector is still a soft average; it is not yet a strong internal trial/check/search loop. |
| Own-latent prediction | Partial | Implemented and compatible, but it predicts over boundary-state sequence; it should become chunk-summary/depth-state aware after the BLT bottleneck fix. |
| Answer attractor | Partial and expensive | Current long run hit CUDA OOM during answer-state attractor CE. |
| Free-generation evaluation | Good | Active policy now bans forced-choice/candidate promotion. |
| Promotion readiness | No | Free generation exact remains 0/8 on smoke checkpoints. |

## Required Fix Before More Long Runs

Fix the hnet_dechunk compression input.

The selected latent input should be a causal chunk summary, not only the
boundary byte:

```text
for each selected boundary/chunk:
  chunk bytes = bytes from this boundary up to before next boundary
  chunk_summary = learned function(boundary byte, mean/max/last of chunk bytes)
  selected_embeddings = chunk_summary
```

The summary must use only bytes available up to that chunk position for causal
generation. No future response bytes may leak into earlier generation
positions.

After this change, the flow should be:

```text
input bytes
-> causal dynamic chunks
-> chunk-summary embeddings
-> NativeQTRMETDLM recurrent core
-> IMTA K trajectories
-> own-latent prediction over chunk/core states
-> dechunk to byte positions
-> hnet_causal_speaker
-> same LM head
```

## Run Status Note

The current IMTA K=3 + own-latent long run reached step 175 and then stopped
with CUDA OOM inside `answer_state_attractor_regularization_loss`.

Useful signals before the crash:

```text
official_delta_runtime = active
imta_active = 1
own_latent_prediction_weight = 0.05
own_latent_prediction_loss decreased
eval clean_loss improved to about 2.36 at step 100
```

But this should not be promoted because:

```text
free generation was not yet good
the BLT chunk-summary bottleneck remains
the run crashed under the answer-state attractor
```

## Next Architecture Move

Do not launch another long run before this smallest restoration:

1. Replace hnet boundary-token-only input with causal chunk-summary input.
2. Add tests proving a non-boundary byte inside a chunk changes downstream
   hnet logits.
3. Disable or reduce answer-state attractor memory use for the first restored
   smoke.
4. Run free-generation-only gates for:
   - K=1, own-latent off,
   - K=3, own-latent off,
   - K=3, own-latent on.

Promotion requires decoded free-generation improvement, not loss alone.
