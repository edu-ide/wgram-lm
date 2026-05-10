# SubQ / Subquadratic Sparse Attention

Date: 2026-05-06

Primary source:

- SubQ technical blog: <https://subq.ai/introducing-subq>
- SubQ SSA explainer: <https://subq.ai/how-ssa-makes-long-context-practical>

Status:

```text
public technical blog / model announcement
no public paper, training recipe, or official implementation found yet
```

Local paper cache:

```text
references/papers/ssa_sparse_attention/
```

## Related SSA / Sparse-Attention Papers

Exact name collision:

| Paper | Link | Why It Matters |
| --- | --- | --- |
| `SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space` | <https://arxiv.org/abs/2511.20102> | Not SubQ's SSA, but directly relevant. It trains sparse and full attention together with bidirectional output alignment so sparse inference does not lose full-attention capability. |

Closest sparse-routing priors:

| Paper | Link | QTRM Takeaway |
| --- | --- | --- |
| `Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention` | <https://arxiv.org/abs/2502.11089> | Natively trained sparse attention with coarse token compression plus fine token selection. Strong prior for training sparse attention from scratch rather than bolting it on. |
| `MoBA: Mixture of Block Attention for Long-Context LLMs` | <https://arxiv.org/abs/2502.13189> | MoE-style block selection for long-context attention. Closest public analogue to content-dependent block routing. |
| `SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs` | <https://arxiv.org/abs/2410.13276> | Learns block-level sparsity gates from Q/K, with self-distillation for pretrained LLMs. Useful if QTRM needs a retrofit rather than full pretraining. |
| `Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference` | <https://arxiv.org/abs/2406.10774> | Query-aware KV-page selection. More inference-system than model architecture, but its query-dependent top-k page scoring maps well to a QTRM router probe. |
| `Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing` | <https://arxiv.org/abs/2505.00315> | Token/head routing with expert-choice sparse attention. Strong evidence that dynamic learned sparsity can beat dense baselines at fixed compute. |
| `Efficient Content-Based Sparse Attention with Routing Transformers` | <https://arxiv.org/abs/2003.05997> | Older but foundational content-based sparse routing via online k-means, reducing attention below quadratic. |

Older structural sparse-attention baselines to keep in the background:

```text
Sparse Transformer, Longformer, BigBird, Reformer, StreamingLLM/H2O
```

These are useful baselines, but they are less directly aligned with SubQ's
claim because many rely on fixed position patterns, inference-time cache
policies, or heuristic retention rather than trainable content-dependent sparse
routing.

## Core Claim

SubQ presents `Subquadratic Sparse Attention` (SSA) as an alternative to dense
attention for long-context LLMs.

Reported properties from the public writeup:

```text
content-dependent sparse attention
no fixed window-only pattern
linear compute with respect to context length
constant memory with respect to context length
1024-token sparsity per layer
ability to attend to any token across the full context
trained from scratch rather than post-hoc retrofitted
```

The model announcement reports:

```text
11B parameter base model
1.5T-token pretraining run
released checkpoints at 4K, 8K, 128K, 1M, and 1.5M context
```

## Why It Matters For QTRM

SubQ is relevant to the `MSA/LM2/MemoryOS` discussion because it argues for
long-context capability through trainable sparse attention inside the model,
not only external retrieval.

QTRM takeaway:

```text
MemoryOS/RAG is runtime context supply.
SubQ-style SSA is model-internal context routing.
They solve different parts of the problem and should not be conflated.
```

For QTRM, the actionable idea is not to copy SubQ directly. The missing recipe
means a faithful implementation is not currently possible. The useful
architecture constraint is:

```text
The recursive core and answer path should learn content-dependent sparse reads
over the canonical token stream, rather than depending on fixed workspace
tokens or hidden evidence side channels.
```

## Mapping To Current QTRM Failures

Current repeated failure:

```text
action/telemetry heads solve the trace,
but LM answer logits do not consume the right intermediate information.
```

SubQ-style implication:

```text
The answer path needs a learned sparse selector over prompt/core states.
It should choose a small set of relevant tokens/states per step and feed the
normal LM logits path.
```

This suggests a future `selective context router`:

```text
prompt hidden states + workspace/core states
-> learned content-dependent scorer
-> top-k selected context states
-> answer recurrent loop / LM logits
```

## Acceptance Gate For QTRM

Do not accept a sparse router by speed alone. It must improve raw intelligence:

```text
no MemoryOS / no hidden evidence
donor_only vs core_off vs router_off vs full
depth sweep 1/2/4/8
held-out length/value ranges
router-off must drop when full improves
```

Long-context acceptance should include:

```text
needle/distractor tasks
multi-hop facts spread across distance
length sweep: 4K, 16K, 64K, 128K+
same answer path, no external span-copy answer channel
```

## Risk

SubQ is not enough evidence for immediate architectural replacement:

```text
no public code
no public training recipe
no peer-reviewed paper yet
unclear exact sparse routing algorithm
unclear compatibility with donor-hidden adapter training
```

Use it as a design prior, not as a source of implementation truth.
