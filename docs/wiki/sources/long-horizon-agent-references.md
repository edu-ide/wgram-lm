# Long-Horizon Agent References

Purpose:

This page records the external papers and repositories currently relevant to
QTRM/MemoryOS long-horizon agent architecture. The sources here are about
runtime organization around LLMs: memory, skills, protocols, recursive
inference, verification, and agent-computer interfaces.

## Core Systems View

| Source | Link | What To Extract For QTRM |
| --- | --- | --- |
| Externalization in LLM Agents | https://arxiv.org/abs/2604.08224 | Treat memory, skills, protocols, and harness engineering as first-class architecture, not as prompt decoration. |
| Memory for Autonomous LLM Agents | https://arxiv.org/abs/2603.07670 | Use a write-manage-read memory loop; evaluate memory through agentic tasks, not static recall only. |
| MemGPT: Towards LLMs as Operating Systems | https://arxiv.org/abs/2310.08560 | Use hierarchical memory and virtual context management instead of assuming a huge direct prompt. |
| Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory | https://arxiv.org/abs/2603.04257 | Keep compact working summaries plus stable indices to full-fidelity evidence; avoid summary-only memory loss. |

## Recursive And Tool-Using Inference

| Source | Link | What To Extract For QTRM |
| --- | --- | --- |
| Recursive Language Models | https://arxiv.org/abs/2512.24601 | Add a mode where long context lives outside the prompt and the controller inspects/decomposes it programmatically. |
| RLM official code | https://github.com/alexzhang13/rlm | Reference for REPL/context-variable style execution; use sandboxing and strict budgets before any production-like use. |
| ReAct | https://arxiv.org/abs/2210.03629 | Interleave reasoning, tool actions, and observations; store traces for debugging and training data. |
| Reflexion | https://arxiv.org/abs/2303.11366 | Store post-failure verbal feedback in a separate reflective memory, not as unverified factual memory. |
| Voyager | https://arxiv.org/abs/2305.16291 | Promote repeated successful procedures into a skill library after execution feedback and self-verification. |
| SWE-agent | https://arxiv.org/abs/2405.15793 | Design a constrained agent-computer interface; performance depends on the interface, test hooks, and editing tools. |

## Retrieval And Corrective Generation

| Source | Link | What To Extract For QTRM |
| --- | --- | --- |
| Self-RAG | https://arxiv.org/abs/2310.11511 | Add adaptive retrieval and self-critique signals for whether evidence is needed, useful, and sufficient. |
| Corrective RAG reproduction | https://arxiv.org/abs/2603.16169 | Add retrieval-quality judging and corrective fallback paths when first-stage retrieval is weak. |
| RAPTOR | https://arxiv.org/abs/2401.18059 | Use hierarchical summaries/tree retrieval for long documents and multi-hop memory over large corpora. |

## Immediate QTRM Reading

These references should not be used as permission to combine every idea inside
the trainable QTRM module. The stable reading is:

- QTRM remains a donor-backed residual latent-workspace adapter.
- MemoryOS provides external evidence, trace storage, and evaluation gates.
- Long-running agent capability belongs in an orchestration/harness layer.
- Recursive/RLM execution is an inference mode, not the default training target.
- Reflections and skills must be separately typed from factual evidence.

## Do Not Import Yet

- Do not implement unconstrained local REPL recursion.
- Do not write reflections into factual memory without validation.
- Do not train QTRM to be the primary language model while the donor already
  provides the base language policy.
- Do not claim 100M-token reasoning from storage capacity alone.
- Do not treat training loss on synthetic traces as success without held-out
  memory reasoning accuracy.
