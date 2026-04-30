# QTRM LLM Wiki Index

## Sources

- [Karpathy LLM Wiki](sources/karpathy-llm-wiki.md): persistent markdown wiki pattern for compounding LLM-maintained knowledge.
- [Karpathy Cognitive Core And Data Quality](sources/karpathy-cognitive-core.md): corrected source notes for cognitive core, data quality, and the 1B-vs-1.8T overclaim.
- [Qwen3.5 2B Configs](sources/qwen35-2b-configs.md): donor model card and nested HF config reference.
- [Qwen3.5 Official Repo](sources/qwen35-official-repo.md): release notes, usage, serving, and training guidance.
- [Qwen3-VL](sources/qwen3-vl.md): prior vision-language architecture report.
- [Qwen3-Omni](sources/qwen3-omni.md): prior omni-modal architecture report.
- [Qwen3.5 Omni](sources/qwen35-omni.md): latest omni-modal Qwen3.5 technical report.
- [Transfer, Merge, Healing](sources/transfer-merge-healing.md): model merging, frankenmerge, continual pretraining, and post-merge tuning references.
- [Training Diagnostics](sources/training-diagnostics.md): scaling laws, learning-curve extrapolation, gradient noise scale, and early stopping references.
- [Architecture Search](sources/architecture-search.md): design-space, NAS, transformer modification transfer, and composition-risk references.
- [CoT To Latent Reasoning](sources/cot-to-latent-reasoning.md): Coconut, CODI, HybridCoT, looped latent thoughts, latent-token reliability, and TRM/ACT halting references.
- [Donor-Logit Sidecar Prior Art](sources/donor-logit-sidecar-prior-art.md): DExperts, FUDGE, GeDi, Proxy-Tuning, Side-Tuning, Ladder Side-Tuning, and AdapterFusion mapping for QTRM donor-logit residual design.
- [Donor Annealing And Distillation](sources/donor-annealing-distillation.md): Annealing-KD, Pro-KD, MiniLLM, GKD, Distilling step-by-step, EasyDistill, MiniPLM, and cross-tokenizer KD references.
- [Test-Time Training](sources/test-time-training.md): In-Place TTT and donor-side fast-weight adaptation references.
- [Self-Improvement And Hallucination](sources/self-improvement-and-hallucination.md): latest self-improvement, preference learning, self-correction, and hallucination-control references.
- [Fact Verification And Fake Info](sources/fact-verification-and-fake-info.md): FEVER, FActScore, SAFE, RAGTruth, conflict-aware RAG, and temporal fake-info verification references.
- [Value And Critical Synthesis](sources/value-and-critical-synthesis.md): value salience, tradition-aware reasoning, religious critique, and constructive positive synthesis references.
- [Titans](sources/titans.md): unofficial Titans implementation and paper notes for neural long-term memory.
- [OpenMythos And Recurrent Depth](sources/openmythos-recurrent-depth.md): speculative OpenMythos plus paper-backed Parcae/looped-transformer references.
- [Gated DeltaNet](sources/gated-deltanet.md): official gated delta rule mixer reference.
- [LeWorldModel](sources/leworldmodel.md): newest end-to-end JEPA world-model reference.
- [Tiny Recursive Models](sources/tiny-recursive-models.md): TRM recursive z_H/z_L and ACT reference.
- [Workspace And Memory References](sources/workspace-memory.md): Perceiver, Q-Former, Flamingo, RMT, and ARMT references for QTRM workspace/memory.
- [Harrier Memory Retrieval](sources/harrier-memory-retrieval.md): Harrier 270M embedding default and FAISS retrieval decision.
- [Rerankers](sources/rerankers.md): Qwen3-Reranker, BGE, NVIDIA Nemotron, ContextualAI, and Jina reranking candidates.
- [Memory Sparse Attention](sources/memory-sparse-attention.md): MSA 100M-token latent-memory reference and QTRM mapping.
- [Long-Horizon Agent References](sources/long-horizon-agent-references.md): Externalization, MemGPT, Memex, RLM, ReAct, Reflexion, Voyager, SWE-agent, Self-RAG, CRAG, and RAPTOR reference map.

## Concepts

- [Gated DeltaNet](concepts/gated-deltanet.md): official target for delta/recurrent mixer design.
- [LeWorldModel](concepts/leworldmodel.md): current preferred JEPA objective.
- [Tiny Recursive Models](concepts/tiny-recursive-models.md): recursive reasoning core contract.
- [LLM Wiki](concepts/llm-wiki.md): documentation and knowledge maintenance pattern.
- [Cognitive Core And Data Quality](concepts/cognitive-core-data-quality.md): QTRM interpretation of small reasoning core, external memory, trace quality, and anti-collapse gates.
- [Qwen3.5 Architecture](concepts/qwen35-architecture.md): donor architecture and multimodal config constraints.
- [Transfer, Merge, Healing](concepts/transfer-merge-healing.md): practical strategy for QTRM donor adaptation and recovery.
- [Training Diagnostics](concepts/training-diagnostics.md): quick probes for deciding whether QTRM training is learning or structurally failing.
- [Compositional Architecture](concepts/compositional-architecture.md): how to combine Qwen, GatedDeltaNet, JEPA/world-model, and recursion without hiding interference.
- [Recurrent-Depth Transformers](concepts/recurrent-depth-transformers.md): looped transformer design, stable injection, depth sweeps, and telemetry.
- [CoT To Latent Transfer](concepts/cot-to-latent-transfer.md): QTRM plan for using explicit traces as supervision while running latent workspace loops with halt telemetry.
- [Workspace Memory Architecture](concepts/workspace-memory-architecture.md): separates QTRM working memory, donor-logit residuals, and future persistent memory.
- [Test-Time Adaptation](concepts/test-time-adaptation.md): separates donor base policy, QTRM residuals, and In-Place TTT donor adaptation.
- [Donor Annealing Roadmap](concepts/donor-annealing-roadmap.md): staged path from Qwen donor-logit sidecar to low-donor-scale QTRM student behavior.
- [Self-Improvement Loop](concepts/self-improvement-loop.md): verified trace, preference, NEEDS_SEARCH, and agentic search loop for hallucination control.
- [Fact Verification Reasoning](concepts/fact-verification-reasoning.md): separates predictive intuition from evidence grounding, verification, conflict arbitration, and temporal/source judgment.
- [Critical Synthesis Reasoning](concepts/critical-synthesis-reasoning.md): critique, preserve, risk-check, reframe, and positive conclusion loop for religion/value questions.
- [Neural Long-Term Memory](concepts/neural-long-term-memory.md): Titans-style memory axis and precise latent-space inference wording.
- [Long-Horizon Agent Architecture](concepts/long-horizon-agent-architecture.md): QTRM/MemoryOS runtime modes for long-running agent work, RLM, trace memory, skills, and verification gates.

## Architecture

- [QTRM Forward Pass](architecture/qtrm-forward-pass.md): Mermaid diagrams and tensor-shape ledger for the current donor-backed residual forward path.
- [Paper Diagram Prompts](architecture/paper-diagram-prompts.md): prompt bank for paper-style architecture and ablation figures.

## Components

- [QTRM Mixer](components/qtrm-mixer.md): status of `src/qtrm_mm/mixers.py` against Gated DeltaNet.
- [QTRM World Model](components/qtrm-world-model.md): status of `src/qtrm_mm/world_model.py` against LeWM.
- [QTRM Recursive Core](components/qtrm-recursive-core.md): status of `src/qtrm_mm/core.py` against TRM.
- [Qwen Donor Integration](components/qwen-donor-integration.md): status of donor loading/config/tokenizer alignment.
- [QTRM Transfer And Healing](components/qtrm-transfer-healing.md): staged training/merge recovery plan.

## Decisions

- [Reference Architecture Axes](decisions/reference-architecture-axes.md): separates generator, mixer, recursion, world model, and wiki sources.
- [QTRM Goal And Scope](decisions/qtrm-goal-and-scope.md): defines QTRM as a Qwen-backed cognitive/memory adapter and sets the next priority order.
- [QTRM Limitations And Mitigation Roadmap](decisions/limitations-mitigation-roadmap.md): maps current architecture limits to prior research, telemetry, ablations, and gated residual next steps.
- [Residual Ablation Plan](decisions/residual-ablation-plan.md): current donor-logit residual experiment matrix and go/no-go criteria.
- [Residual Adapter Proof](decisions/residual-adapter-proof.md): fixed proof package showing donor-only versus QTRM residual gains on current MemoryOS probes.
- [Expanded Workspace/Core Ablation](decisions/expanded-workspace-core-ablation.md): expanded 72-case causality ablation covering workspace/core, coda, residual-head, donor-hidden, and workspace-only paths.
- [MemoryOS 100M Scale Plan](decisions/memoryos-100m-scale-plan.md): treats 100M+ tokens as external memory, not direct prompt context.
