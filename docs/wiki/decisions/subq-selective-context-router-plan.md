# SubQ Selective Context Router Plan

Date: 2026-05-06

Status: minimal probe implemented; first S020 gate rejected.

## Decision

SubQ is useful as a design prior for QTRM, but it should not replace the
current Ouro answer recurrent falsifier yet.

Reason:

```text
SubQ currently has a public technical announcement, but no public paper/code
or exact SSA recipe. Copying it would be guesswork.
```

Related public prior work narrows the safe implementation:

```text
SSA 2511.20102: train sparse and full attention together with output alignment
NSA 2502.11089: coarse compression + fine selection, natively train sparse attention
MoBA 2502.13189: MoE-style block routing for long-context attention
SeerAttention 2410.13276: learn sparse block gates by self-distillation
Quest 2406.10774: query-aware top-k KV page selection
MoSA 2505.00315: expert-choice content-based sparse attention
Routing Transformer 2003.05997: older content-based sparse routing baseline
```

The implementable QTRM interpretation is a small selective context router:

```text
canonical prompt hidden states
+ workspace/core trajectory states
-> learned content-dependent top-k selector
-> selected context states
-> answer recurrent loop
-> LM logits
```

This is compatible with the universal LLM causal path because selected states
feed the normal answer hidden state and LM head. It is not MemoryOS, RAG, or a
hidden evidence channel.

## Why This Matters

The repeated QTRM failure is:

```text
the core can solve action/finality telemetry,
but final answer logits still choose intermediate strings or ties.
```

A learned sparse router attacks a different bottleneck than role-value answer
tokens:

```text
role-value bridge: supplies a small side-state token set
selective router: learns which prompt/core states the answer hidden state
                  should read at each recurrent step
```

## Candidate Architecture

Minimal candidate after Ouro answer recurrent:

```text
answer hidden y_t
-> query projection
-> score all prompt/core states
-> top-k select or sparse mask
-> cross-attend selected states
-> recurrent answer update
-> LM logits
```

Small 4090-friendly settings:

```text
top_k: 16 or 32
router candidates: prompt hidden states + core depth states
training: dense/full cross-attention teacher vs sparse selected-context student
loss: final answer CE + optional dense/sparse hidden alignment
no long-context scale-up until smoke passes
router-off ablation required
```

## Acceptance Gate

Promote only if:

```text
no MemoryOS / no hidden evidence
held-out LM causal forced-choice improves over donor_only and core_off
router_off causes a drop
depth sweep shows non-regressive or improving answer accuracy
longer/distractor cases improve after the short gate passes
```

Reject if:

```text
router only improves speed
router only improves in-sample CE
router does not change held-out answer choices
router becomes a span-copy/answer side channel
```

## Priority

Order:

```text
1. Finish Ouro answer recurrent S80 falsifier.
2. Done: implemented minimal answer-state selective context router.
3. S020 result: full 2/8, router-off 2/8, action-code 32/32. Reject as causal
   architecture gain because router-off does not hurt.
4. Dense-alignment S020 result: full 2/8, router-off 2/8, action-code 32/32.
   Reject; KL to dense state+prompt teacher is wired but not causally useful
   at this scale.
5. Only after a short causal router-off gate passes, test long-context
   SubQ-like scaling.
```
