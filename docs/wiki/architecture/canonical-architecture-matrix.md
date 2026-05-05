# Canonical Architecture Matrix

Status: implementation ledger, 2026-05-03.

Canonical active probe:
`configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml`

This page separates what is implemented from what is actually active in the
current representative architecture. The model architecture starts at the
canonical token stream. MemoryOS retrieval, reranking, and context compilation
belong to the runtime system, not to the model. See
[QTRM Model vs Runtime Boundary](model-vs-runtime-boundary.md).

Current canonical decision: LeWorldModel/core-world-model remains implemented
as an experimental probe, but it is not part of the canonical answer path. The
canonical raw-intelligence path is a single-trace mandatory TRM recursion:

```text
canonical token stream
-> frozen donor hidden-state context
-> latent workspace
-> mandatory recursive core
-> answer-state loop
-> LM logits
```

## Status Legend

| Status | Meaning |
| --- | --- |
| Active | In the current canonical probe's forward path and training objective. |
| Forward-only | Computed or returned, but not currently optimized by the canonical probe. |
| Probe-only | Has a dedicated experiment, but is not part of the latest canonical probe. |
| Runtime/eval | Used outside the model during retrieval, evaluation, or ablation. |
| Scaffold | Code exists, but current training/eval does not prove the capability. |

## Active Canonical Path

| Component | Status | Current Config/Script | Notes |
| --- | --- | --- | --- |
| Canonical token stream | Active | `scripts/193_run_pure_recursive_reasoning_depth_gate.sh` | No retrieval, hidden evidence, or MemoryOS shortcut. |
| Frozen Qwen donor hidden states | Active scaffold | `scripts/196_train_pure_recursive_depth_supervised.py` | Donor states are projected into QTRM width; donor logits are not the answer policy in this raw gate. |
| Frozen Qwen donor logits | Disabled | `donor_logits_scale: 0.0` | Donor-only is an evaluation baseline, not the canonical answer path. |
| Latent workspace | Active | `workspace_tokens: 64`, `workspace_layers: 3` | Per-forward working memory slots over the canonical prompt stream. |
| Gated core context injection | Active | `core_context_enabled: true`, `core_context_gate_init_bias: 0.5` | Recursive core reads prompt context through an ablatable path. |
| Core step conditioning | Active | `core_step_conditioning_enabled: true` | Gives depth steps distinguishable roles without external traces. |
| Mandatory recursive core | Active | `core_enabled: true`, `outer_steps` swept by eval | The raw-intelligence claim depends on depth improving held-out answers. |
| Answer-state loop | Active | `answer_state_loop_enabled: true`, `answer_state_loop_requires_core: true` | Final answer logits come from a state updated by the recursive core. |
| QTRM direct answer logits | Active | `qtrm_logits_scale: 1.0`, `qtrm_residual_gate_enabled: false` | Donor-backed residual fusion is not the raw-intelligence canonical path. |
| Causal-prefix first-token CE | Active training objective | `--causal-prefix-supervision`, `--final-logit-ce-weight 1.0` | Prevents the core from reading future answer tokens during depth supervision. |
| Progress margin | Active training objective | `--progress-margin-weight 0.25` | Encourages deeper scheduled states to improve target log-probability. |
| LeWM core trajectory prediction | Probe-only | `core_world_model_enabled: true` only in LeWM configs | Demoted from canonical because it predicts self-latent transitions without improving symbolic intermediate states or answer accuracy. |

## Runtime/System Layer

These components may affect the model input, but they are not internal QTRM
model architecture.

| Component | Status | Where It Exists | Boundary |
| --- | --- | --- | --- |
| SSOT context compilation | Runtime/eval | `scripts/95_eval_memory_retrieval.py --evidence-injection ssot` | Retrieved evidence is compiled into one canonical donor-visible token stream before model forward. |
| Canonical autoregressive answer gate | Active eval | `scripts/166_run_canonical_ssot_answer_gate.sh`, `scripts/95_eval_memory_retrieval.py --require-canonical-ssot` | Uses `--evidence-injection ssot` and `--answer-channel greedy`; rejects workspace/dual hidden-evidence and span-copy answer paths. |
| MemoryOS retrieval | Runtime/eval | `qtrm_mm/memoryos/retrieve.py`, `scripts/95_eval_memory_retrieval.py` | External search over memory records. Optional; the QTRM model must work without it. |
| MemoryOS reranker | Runtime/eval | `qtrm_mm/memoryos/rerank.py` | External candidate ordering before context compilation. |
| Learned evidence source selector | Runtime/eval accepted | `scripts/165_train_evidence_source_selector.py` | Runtime selector that creates token-aligned masks over the compiled context; not a model layer. |
| Dual evidence injection | Probe-only | `--evidence-injection dual`, `workspace_evidence_injection_mode: dual` | Deterministic visible and workspace views of one evidence context for ablations. |
| Workspace-only evidence injection | Probe-only | `--evidence-injection workspace`, `workspace_evidence_injection: true` | Hidden-evidence causality probe, not a user-facing architecture. |

## Implemented But Not Active In Canonical Probe

| Component | Status | Where It Exists | Why It Is Not Canonical Yet |
| --- | --- | --- | --- |
| Token-level LeWM/JEPA loss | Forward-only | `jepa_pred`, `jepa_target`, `loss_jepa_weight` | Computed, but canonical probe sets `loss_jepa_weight: 0.0` to avoid multi-loss interference. |
| Controller aux heads | Forward-only | `ControllerHeads`, `loss_aux_weight` | Outputs exist, but `loss_aux_weight: 0.0`; not a learned action policy yet. |
| Core halting / early exit | Probe-only | `core_halt_enabled`, `scripts/107_run_core_halt_probe.sh` | Separate halt probes exist; current canonical depth gates use fixed depth sweeps for causality. |
| Donor annealing | Probe-only | `donor_logits_scale_start/end` | Scheduling exists, but canonical raw-intelligence scoring disables donor logits with `donor_logits_scale: 0.0`. |
| Generation verifier heads | Probe-only | `generation_verifier_enabled`, `scripts/141_build_generation_verifier_dataset.py`, `scripts/142_run_generation_verifier_s020.sh`, `scripts/143_eval_generation_verifier.py` | Repeat/stop/quality heads read the coda/post-norm last valid text hidden state. First smoke shows in-sample repeat/quality signal, but thresholds and holdout generalization are not accepted yet. |
| Prompt-conditioned evidence span reader | Probe-only | `evidence_span_reader_enabled`, `scripts/build_evidence_span_reader_dataset.py`, `scripts/150_run_evidence_span_reader_train.sh` | 2026 update path. It can now score canonical prompt tokens with `evidence_span_reader_context="input"` for SSOT, or hidden workspace evidence tokens for ablation probes. It is not the canonical answer generator. |
| In-model answer-decision head | Probe-only accepted | `answer_decision_head_enabled`, `scripts/164_bootstrap_answer_decision_feature_head.py`, `scripts/95_eval_memory_retrieval.py --model-answer-decision` | Accepted on truthcal heldout: full 62/72, feature/head-off 49/72. It uses raw answer-channel telemetry, not hidden state alone. |
| Evidence span-boundary REVISE | Probe-only accepted | `scripts/95_eval_memory_retrieval.py --answer-revision evidence_span_boundary` | Accepted as a narrow renderer fix: full answer-decision path improves to 67/72 while feature/head-off ablations reach 55/72. It only repairs token-boundary truncation and does not solve source selection. |
| Learned evidence source span-mask | Probe-only accepted | `scripts/95_eval_memory_retrieval.py --evidence-source-selector-checkpoint ... --evidence-source-selector-mode span_mask` | Accepted on truthcal heldout in the workspace probe path: selector case success 72/72 and answer accuracy 71/72. In SSOT it masks final span-copy logits over canonical prompt tokens. |
| Reliability source governor | Rejected probe | `scripts/95_eval_memory_retrieval.py --evidence-source-governor reliability` | Rejected on truthcal heldout: full mode drops to 48/72 because pruning evidence shifts the answer-decision path out of calibration. Replace with learned selector/verifier before considering canonical use. |
| Agentic closed-loop planner | Scaffold | `docs/wiki/concepts/agentic-closed-loop-planner.md` | Prior research and design contract exist; no `AgentHarness`, trace replay buffer, or learned action policy is implemented yet. |
| Multimodal image path | Scaffold | `visual_features`, `image_to_features`, `MultimodalProjector` | Most current scripts run `MULTIMODAL=0`; no real Qwen vision path is proven. |
| Symbolic logic verifier | Scaffold | LINC/NL2LOGIC references only | No in-model or sidecar prover is wired into training yet. |
| Persistent long-term neural memory | Scaffold | MemoryOS docs and retrieval stack | Latent workspace is per-forward working memory, not persistent neural memory. |

## Donor-Fusion Boundary

Residual gates govern the donor-fusion residual path:

```text
text_logits = donor_logits * donor_logits_scale
            + residual_gate * bounded_qtrm_residual
```

For evidence-only proof probes, `evidence_bottleneck_applies_to_residual=true`
changes the residual term to:

```text
evidence_gate * residual_gate * bounded_qtrm_residual
```

This mode is intentionally stricter and should not be treated as the general
QTRM loop-LM path. General prompts still need QTRM latent reasoning even when
there is no external workspace evidence.

When `donor_logits` are absent or `donor_logits_scale=0.0`, raw-intelligence
gates score the QTRM answer path directly. This is now the canonical
single-trace TRM path. Donor-backed residual fusion remains a separate
language-preservation and MemoryOS-support experiment.

## Acceptance Gates

The canonical architecture is accepted only if the following are measured:

- no retrieval, hidden evidence, MemoryOS, or workspace-only answer shortcut;
- donor-only, core-off, workspace-off, and shallow-depth baselines are reported;
- deeper single-trace recursion improves held-out reasoning accuracy;
- answer outputs change under `core_off` on rows where latent recurrence is
  expected to matter;
- all LeWM/core-world-model settings are disabled for canonical runs;
- repeated-token diagnostics do not regress.

## Next Consolidation Step

Keep the current accepted small gate as the canonical baseline, then scale it
only with the same single-trace TRM contract:

```text
CONFIG=configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
CORE_WORLD_MODEL_WEIGHT=0.0
CAUSAL_PREFIX_SUPERVISION=1
```
