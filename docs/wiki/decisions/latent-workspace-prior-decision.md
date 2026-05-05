# Latent Workspace Prior Decision

Status: decided on 2026-04-30.

Question:

```text
Should QTRM discard the Perceiver / Q-Former / OpenFlamingo PerceiverResampler
latent-workspace lineage because it is not the newest memory architecture?
```

Decision:

```text
Do not discard it, but demote it.
```

Perceiver/Q-Former/PerceiverResampler-style learned latent slots remain useful
as a compact connector and in-context bottleneck. They should not be treated as
the main source of modern long-memory or reasoning claims.

## Classification

| Axis | Current role in QTRM | Priority |
| --- | --- | --- |
| Perceiver / Q-Former / PerceiverResampler | Learned query slots and cross-attention connector | Keep as baseline building block |
| Qwen2.5/Qwen3-style dynamic-resolution merger | Modern multimodal connector reference; often simpler than query-former stacks | Use as donor-aligned connector reference |
| LM2 | Auxiliary gated memory lane while preserving the base transformer path | High priority for internal memory design |
| G-MemLLM | Frozen backbone plus trainable latent memory bank with GRU-style update/preserve/overwrite | High priority for gated workspace design |
| MSA | End-to-end sparse memory routing toward 100M-token memory | High priority for MemoryOS scale path |
| LightMem / MemCoT | External agent memory filtering, consolidation, iterative search | High priority for runtime MemoryOS layer |

## Why Not Delete Perceiver-Style Slots

The learned-slot idea still solves a real engineering problem:

- compress variable-length context into a fixed latent width;
- provide a small interface between donor states, retrieved evidence, and QTRM;
- make workspace size independent from prompt length;
- keep the adapter cheap enough to train on a single GPU.

That is still useful even if the original Perceiver/Q-Former papers are older.
The mistake would be to claim that this alone is modern long-term memory or
latent reasoning.

## Why It Is Not Enough

Recent memory papers move beyond "latent slots attend to context":

- LM2 adds a complementary memory pathway to a decoder-only transformer and
  updates memory through gates while preserving the normal transformer path.
- G-MemLLM makes the memory bank explicitly trainable beside a frozen LLM
  backbone and uses GRU-style update, preserve, and overwrite logic.
- MSA treats very large memory as sparse document-level memory routing rather
  than placing all tokens in active attention.
- Qwen2.5-VL shows that modern multimodal connectors can also be simple and
  efficient: dynamic-resolution ViT plus an MLP-based merger, not necessarily a
  heavy Perceiver/Q-Former adapter.

## QTRM Policy

Use this wording:

```text
QTRM uses Perceiver/Q-Former-style learned slots as a connector baseline, then
adds LM2/G-MemLLM-inspired gated updates and MemoryOS/MSA-style external
memory routing as the modern memory path.
```

Avoid this wording:

```text
QTRM's latent workspace is modern long-term memory because it resembles
Perceiver or Q-Former.
```

## Implemented Architecture Step

The next change is now implemented as an eval/model path, not as another deeper
Perceiver stack. It is a causal-memory gate that can make the task impossible
without workspace state:

1. Put evidence into workspace-side memory tokens rather than the normal donor
   prompt path.
2. Keep donor logits as the base language policy.
3. Let QTRM residual logits read only the workspace-mediated evidence.
4. Require `workspace_off` and `workspace_gate_off` to drop below full residual.
5. Track permissive hit rate, normalized exact rate, and human-audit rate.

Implementation entry point:

```bash
bash scripts/117_run_workspace_evidence_path_probe.sh
```

Training entry point:

```bash
bash scripts/118_run_workspace_evidence_path_train.sh
```

Eval flag:

```text
--evidence-injection workspace
```

If this fails, Perceiver-style slots remain a connector but not a reasoning
core. If it succeeds, the claim becomes "workspace-mediated memory contributes"
rather than "Perceiver slots reason."

## Current Evidence

Latest gated-workspace ablation:

| Mode | Permissive hits | Exact / normalized exact | Causal interpretation |
| --- | ---: | ---: | --- |
| `qtrm_residual_with_evidence` | 49/72 | 24/72 | Residual adapter improves over donor |
| `qtrm_residual_head_off_with_evidence` | 26/72 | 9/72 | Residual head is causal |
| `qtrm_coda_off_with_evidence` | 39/72 | 31/72 | Coda contributes |
| `qtrm_workspace_gate_off_with_evidence` | 49/72 | 24/72 | Gate is not yet causal |
| `qtrm_workspace_off_with_evidence` | 49/72 | 24/72 | Workspace is not yet causal |

Conclusion:

```text
The current architecture should keep learned slots, but the research path must
shift from Perceiver-style resampling depth to gated memory causality and
sparse external memory routing.
```

## References

- Perceiver / Perceiver IO: `references/papers/memory_workspace/`
- BLIP-2 Q-Former: `references/papers/memory_workspace/blip2_qformer_2301.12597.pdf`
- Flamingo / OpenFlamingo: `references/papers/memory_workspace/`
- LM2: `references/papers/memory_workspace/lm2_2502.06049.pdf`
- G-MemLLM: `references/papers/memory_workspace/g_memllm_2602.00015.pdf`
- MSA: `references/official/msa/paper/MSA__Memory_Sparse_Attention_for_Efficient_End_to_End_Memory_Model_Scaling_to_100M_Tokens.pdf`
- Qwen2.5-VL: `https://arxiv.org/abs/2502.13923`
