# Output Embedding And Renderer Alignment Sources

Status: reference synthesis, 2026-05-06.

Purpose: track sources relevant to the QTRM failure where causal
forced-choice succeeds but autoregressive rendering fails.

## Sources

| Area | Source | QTRM relevance |
| --- | --- | --- |
| Output embedding geometry | Press and Wolf, "Using the Output Embedding to Improve Language Models" <https://arxiv.org/abs/1608.05859> | The output embedding is not a disposable classifier matrix; its geometry materially affects language modeling. |
| Input/output tying | Inan, Khosravi, Socher, "Tying Word Vectors and Word Classifiers" <https://arxiv.org/abs/1611.01462> | Supports tying or otherwise aligning token embeddings and output classifiers rather than treating vocab logits as arbitrary labels. |
| Latent thought before token prediction | PonderLM-2 <https://arxiv.org/abs/2509.23184> | Relevant because it trains latent thoughts so they immediately improve next-token prediction, not just post-hoc answer scoring. |
| Official PonderLM-2 code | <https://github.com/LUMIA-Group/PonderLM-2> | Local clone: `references/official/PonderLM-2` at `fa784bece621b989fb008c59b0fd8d282fa9c73c`. |
| Continuous latent reasoning | Coconut <https://arxiv.org/abs/2412.06769> and <https://github.com/facebookresearch/coconut> | Local clone: `references/official/coconut` at `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`. Shows staged latent reasoning must still decode back into the normal LM path. |
| Exposure-bias family | Scheduled Sampling <https://arxiv.org/abs/1506.03099>, Professor Forcing <https://arxiv.org/abs/1610.09038>, sequence/minimum-risk training <https://arxiv.org/abs/1511.06732>, <https://arxiv.org/abs/1512.02433> | Useful for on-policy rollout mismatch, but QTRM self-rollout already failed, so this is not the only active bottleneck. |

## QTRM Takeaway

The current evidence says:

```text
forced-choice success does not imply autoregressive LM-head readiness
```

Rejected local fixes:

```text
low-rank answer LM adapter
greedy-token margin
causal-prefix self-rollout
donor-output embedding head surgery
```

Next viable direction:

```text
Train an LM-compatible decoder/projection from answer_state_loop_hidden before
expecting answer-state vectors to produce stable tokenizer logits.
```
