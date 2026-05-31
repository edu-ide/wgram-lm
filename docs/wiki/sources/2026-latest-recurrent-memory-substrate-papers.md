# Latest Recurrent-Memory Substrate Papers For RI Closure

Date: 2026-05-29

Purpose: keep the latest paper evidence that currently governs QTRM raw
intelligence substrate work in one compact source page. This page is not a
promotion claim. It is the literature-backed reason for making hybrid
recurrence depth, sparse persistent memory, stochastic breadth, and strict
backend/eval gates first-class requirements.

## Primary Sources

- Huginn / recurrent latent reasoning: <https://arxiv.org/abs/2502.05171>
- Memory Sparse Attention (MSA): <https://arxiv.org/abs/2603.23516>
- Titans long-term neural memory: <https://arxiv.org/abs/2501.00663>
- Gated DeltaNet source family: <https://arxiv.org/abs/2412.06464>
- Gated DeltaNet-2 update track: <https://arxiv.org/abs/2605.22791>
- HRM / recursive text-LM reference: <https://arxiv.org/abs/2506.21734>
- Equilibrium Reasoners / attractor-style reasoning: <https://arxiv.org/abs/2605.21488>
- Linear-Time Looped Transformers (LT2): <https://arxiv.org/abs/2605.20670>
- Byte Latent Transformer / tokenizer-free dynamic patching: <https://arxiv.org/abs/2412.09871>
- Fast Byte Latent Transformer / BLT-D diffusion and verification: <https://arxiv.org/abs/2605.08044>
- H-Net dynamic chunking: <https://arxiv.org/abs/2507.07955>
- H-Net++ hierarchical dynamic chunking: <https://arxiv.org/abs/2508.05628>
- FLEXITOKENS adaptive byte tokenization: <https://arxiv.org/abs/2507.12720>
- Learn From Your Own Latents: <https://arxiv.org/abs/2605.27734>

## Official Code References Checked

These repositories were inspected as implementation references, not copied as
drop-in code:

| Source | Local reference commit | QTRM design use |
|---|---:|---|
| BLT: <https://github.com/facebookresearch/blt> | `9774ed4` | Dynamic byte patching represents chunks with patch-level state, so QTRM hnet must feed the core causal chunk summaries rather than only boundary-byte embeddings. |
| GatedDeltaNet-2: <https://github.com/NVlabs/GatedDeltaNet-2> | `da7974d` | Confirms the recurrence substrate should be auditable erase/write-style state update, with strict backend identity instead of silent fallback. |
| MSA: <https://github.com/EverMind-AI/MSA> | `30405b2` | Supports future RI-4 memory as sparse top-k latent/chunk routing inside the recurrent loop, not prompt stuffing. |
| H-Net: <https://github.com/goombalab/hnet> | `3673fe1` | Supports learned dynamic chunking over bytes/characters with chunk-level latent states. |
| Coconut: <https://github.com/facebookresearch/coconut> | `27273cb` | Supports same-body continuous latent reasoning before the final language head, not detached answer-candidate roulette. |
| Attractor: <https://github.com/jacobfa/Attractor> | `53573db` | Supports attractor/fixed-point telemetry as a stability diagnostic, not a separate answer module. |

Related local source pages:

- [Memory Sparse Attention](memory-sparse-attention.md)
- [Gated DeltaNet](gated-deltanet.md)
- [Titans](titans.md)
- [HRM-Text](hrm-text.md)
- [2026 Recursive, Memory, Context, And Search Papers](2026-recursive-memory-context-papers.md)

## Read Across The Papers

The current research direction is converging on a small set of testable
constraints:

1. **Latent recurrence must be causally necessary.** Huginn and looped/recurrent
   text models make depth a real compute axis, not a decorative config. QTRM
   must therefore expose no-evidence depth sweeps and recurrence-off ablations
   on the same normal answer path.
2. **Sparse memory must sit inside the recurrent thinker.** MSA shifts long
   context from dense prompt stuffing to sparse latent/document memory routing.
   QTRM's RI-4 requirement is therefore top-k/selective slot routing, router-off
   and chunk-shuffle ablations, and persistence of untouched slots.
3. **Writes need a policy, not only storage.** Titans-style surprise memory and
   QTRM 5.56 rehearsal both imply that the model must decide what to update,
   preserve, or rehearse. Sparse slots without write/rehearsal causality are
   insufficient.
4. **The recurrence primitive must be strict and auditable.** Gated DeltaNet and
   GDN2-style linear recurrence are valid substrate candidates only if the code
   fails fast when the requested official backend is unavailable. Silent fallback
   invalidates substrate claims.
5. **Attractor and loop stability are first-class telemetry.** EqR and LT2 imply
   that more loop depth must improve or converge, not merely run longer. Future
   QTRM gates should add fixed-point residual, halt, and stability telemetry on
   top of the depth/slot gates added in this closure.
6. **The local 82M language substrate should stay tokenizer-free.** BLT,
   Fast BLT, H-Net/H-Net++, and FLEXITOKENS all point away from a fixed BPE
   main path for this research thread. The practical local consequence is
   dynamic byte patching plus a byte reconstruction/generation auxiliary, with
   BPE kept only as a baseline.
7. **Latent self-prediction is an auxiliary, not a second mouth.** Learn From
   Your Own Latents supports predicting answer-causal hidden/chunk states to
   improve internal structure. In QTRM this maps to same-body own-latent
   prediction over causal chunk/core states, while decoded answers still come
   from the same LM head.

## Mapping To QTRM RI Conditions

| Paper axis | QTRM condition | Immediate executable implication |
|------------|----------------|----------------------------------|
| Huginn recurrent depth | RI-1 | Add `hybrid_recurrence_depth_N_no_evidence` and `hybrid_recurrence_off_no_evidence` eval modes. |
| EqR / LT2 loop stability | RI-1, RI-2, RI-5 | Require monotonic depth gates now; add residual/halt telemetry next. |
| MSA sparse memory | RI-2, RI-4 | Keep sparse slot router active inside OneBodyParallelHybridBlock and evaluate router/slot ablations. |
| Titans surprise memory | RI-3, RI-4 | Treat 5.56 stochastic breadth and rehearsal/write policy as causal components, not optional regularizers. |
| Gated DeltaNet family | RI-5, RI-6 | Strict backend contracts and explicit v2 aliases are required so evals know which substrate actually ran. |
| BLT / Fast BLT | RI-6, RI-7, tokenizer-free local 82M track | Keep dynamic BLT as main path; require masked byte/block reconstruction targets to be nonzero on hnet-style dynamic chunking. |
| H-Net / H-Net++ / FLEXITOKENS | RI-4, RI-7 | Prefer learned/dynamic byte chunking over fixed BPE; evaluate chunking causality with boundary priors, fixed-boundary baselines, and heldout byte loss. |
| Learn From Your Own Latents | RI-1, RI-6 | Train same-body own-latent prediction over causal chunk/core states; require same-head free-generation lift, not only latent cosine improvement. |

## Current QTRM Closure Consequence

As of the 2026-05-31 update, the literature-backed requirement is no longer
only a wiki aspiration:

- The hybrid recurrence depth ladder is addressable by raw-intelligence eval
  modes and a dedicated gate builder.
- Sparse slot routing has a stable public mask contract and can be ablated
  without tensor-shape explosions.
- Stochastic breadth is represented as an active hybrid replacement in the
  component registry and strict SSOT gate.
- Official backend requests are fail-fast under strict mode instead of quietly
  falling back to a weaker substrate.
- The active BLT hnet path now uses causal learned chunk summaries so
  non-boundary bytes can influence recurrent latent input.
- The active BLT IMTA path has same-body per-trajectory adapters,
  speaker-space selection, and trajectory diversity telemetry/loss hooks.

Remaining promotion work is still substantial: long heldout runs, 150-200+
horizon stability tests, full 5.56 matrix runs, and matched data-efficiency
curves must pass before claiming paper-grade raw intelligence.
