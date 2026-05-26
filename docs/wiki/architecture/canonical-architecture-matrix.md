# Canonical Architecture Matrix

Status: implementation ledger, 2026-05-07.

Canonical active probe:
`configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml`

This page separates what is implemented from what is actually active in the
current representative architecture. The model architecture starts at the
canonical token stream. MemoryOS retrieval, reranking, and context compilation
belong to the runtime system, not to the model. See
[QTRM Model vs Runtime Boundary](model-vs-runtime-boundary.md).

All canonical candidates must also satisfy the
[Universal LLM Causal Path Contract](universal-llm-causal-path-contract.md):
inputs enter through prompt/chat-template tokens, internal structured modules
must be learned and ablatable, and final answers must come from LM logits and
autoregressive text rather than an external solver.

Typed-register, subregister, and primitive-operation paths are therefore
scaffolds unless they satisfy that same contract. They may expose a raw
reasoning bottleneck, but they are not canonical universal-LLM architecture
when they act as a separate state-machine, calculator, or hidden answer
channel. Promotion requires a held-out LM-path gain over donor-only/core-off
and a causal drop when the register/primitive path is disabled.

Current canonical decision: LeWorldModel/core-world-model remains implemented
as an experimental probe, but it is not part of the canonical answer path.
OpenMythos/Mythos-style answer-side recurrence is also demoted to stability
prior/probe-only. The canonical raw-intelligence path is a single-trace
mandatory TRM/QTRM recursion:

```text
canonical token stream
-> frozen donor hidden-state context
-> latent workspace
-> mandatory recursive core
-> answer-state readout loop
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

## SSOT Cleanup Rule

Reusable modules are not automatically promoted architecture. A Python class may
exist for one of three reasons:

```text
promoted path:
  selected by the current SSOT and protected by ablations/gates

diagnostic/probe path:
  kept to reproduce, compare, or falsify a failed idea

deprecated/rejected path:
  documented as rejected and not allowed in promoted runs
```

Do not assume that every class in `src/qtrm_mm` is BEST. BEST requires an
explicit SSOT page, a current promotion gate, and an ablation signal. Rejected
paths should either be removed or guarded so they cannot become the default
training path by accident.

Code SSOT:

- `src/qtrm_mm/architecture/component_registry.py`

The registry is the executable version of this rule. It currently marks:

```text
promoted:
  one_body_contract
  blt_components
  qtrm_recursive_core
  state_transition_core

diagnostic:
  stage99_bridge_readback_selector

deprecated:
  typed_register_executor_family

scaffold:
  bltd_byte_latent_prefixlm
```

Use `assert_promoted_component(...)` before treating a reusable class or
script path as BEST. In plain Korean: 공구함에 있다고 전부 주력 장비가 아니다.
주력 장비는 SSOT가 promoted라고 부르고, ablation/gate가 지켜주는 것만이다.

Past-success rule:

- Before using an old high score to justify a new architecture run, separate
  the exact old metric from the new claim. Stage56/58's 0.768-0.934
  selected/oracle PTRM arithmetic success is a search/verifier clue, not a
  general language-generation claim. See
  [Past-Success Doubt Loop](../decisions/past-success-doubt-loop-stage56-stage58.md).

## Active Canonical Path

| Component | Status | Current Config/Script | Notes |
| --- | --- | --- | --- |
| Canonical token stream | Active | `scripts/193_run_pure_recursive_reasoning_depth_gate.sh` | No retrieval, hidden evidence, or MemoryOS shortcut. |
| Frozen Qwen donor hidden states | Active scaffold | `scripts/196_train_pure_recursive_depth_supervised.py` | Donor states are projected into QTRM width; donor logits are not the answer policy in this raw gate. |
| Frozen Qwen donor logits | Disabled | `donor_logits_scale: 0.0` | Donor-only is an evaluation baseline, not the canonical answer path. |
| Latent workspace | Active | `workspace_tokens: 64`, `workspace_layers: 3` | Per-forward working memory slots over the canonical prompt stream. |
| Gated core context injection | Active | `core_context_enabled: true`, `core_context_gate_init_bias: 0.5` | Recursive core reads prompt context through an ablatable path. |
| Core step conditioning | Active | `core_step_conditioning_enabled: true` | Gives depth steps distinguishable roles without external traces. |
| Mandatory TRM/QTRM recursive core | Active | `core_enabled: true`, `outer_steps` swept by eval | This is the only canonical reasoning core. The raw-intelligence claim depends on depth improving held-out answers. TRM-style no-grad inner cycles, per-sequence halt freeze, explicit detached `QTRMCoreCarry`, carry eval mode, training-only halt exploration, and q-value stop/continue halt loss are now implemented for the next core-ACT run. |
| Answer-state readout loop | Active | `answer_state_loop_enabled: true`, `answer_state_loop_requires_core: true` | Final answer logits come from a state updated by the recursive core. This is a renderer/control path, not a second reasoning core. |
| QTRM direct answer logits | Active | `qtrm_logits_scale: 1.0`, `qtrm_residual_gate_enabled: false` | Donor-backed residual fusion is not the raw-intelligence canonical path. |
| Causal-prefix first-token CE | Active training objective | `--causal-prefix-supervision`, `--final-logit-ce-weight 1.0` | Prevents the core from reading future answer tokens during depth supervision. |
| Progress margin | Active training objective | `--progress-margin-weight 0.25` | Encourages deeper scheduled states to improve target log-probability. |
| LeWM core trajectory prediction | Probe-only | `core_world_model_enabled: true` only in LeWM configs | Demoted from canonical because it predicts self-latent transitions without improving symbolic intermediate states or answer accuracy. |
| Core role-value state carry | Rejected probe | `core_state_carry_enabled`, `core_state_carry_only` | Full fine-tuning regressed value accuracy to 80/624; carry-only preserved action-code 32/32 but stayed below the 184/624 value baseline at 158/624. |
| Core role-value delta adapter | Rejected probe | `core_role_value_delta_enabled`, `core_role_value_delta_only` | Untrained config preserves 184/624, but LR 1e-4 regresses to 112/624 and LR 1e-5 only ties baseline. |
| Discrete value-delta executor | Rejected probe | `core_value_delta_code_enabled`, `core_value_delta_code_only` | Full/code-off both tie 184/624, direct code-logit readout drops to 63/624, and depth 8 does not beat depth 1. Keep as a probe scaffold only. |
| Typed register executor | Rejected probe | `core_typed_register_executor_enabled`, `core_typed_register_executor_only` | First scaffold preserves the universal LLM path and is causal, but full reaches only 106/624 while executor-off restores the 184/624 baseline. Keep as scaffold; next candidate needs operation/process supervision. |
| Typed register executor v2 | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_executor_v2_process_s120.yaml` | Process-code CE preserves action-code 32/32, but full value accuracy reaches only 102/624 while typed-register-off restores 184/624. Reject as canonical. |
| Typed register prompt binder | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_prompt_binder_s120.yaml` | Opening prompt-token cross-attention into the typed-register path still regresses: full 104/624, register-off 168/624, action-code 32/32. Prompt access alone is not the bottleneck. |
| Typed recurrent transition core | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_consistency_s120.yaml` | Auxiliary transition CE still regresses: full 104/624, register-off 184/624, action-code 32/32. The transition head was not the evaluated value path. |
| Strict transition readout | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_strict_transition_readout_s120.yaml` | Strictly causal transition readout regresses to 64/624 while typed-register-off restores 184/624; action-code remains 32/32. Pause local typed-register head variants and move to a root latent-state candidate. |
| Typed subregister scalar codec | Rejected scaffold | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_subregister_scalar_offset_s160.yaml` | Scalar-only gate learns field type/shape but not numeric transition: best held-out step_000120 has trace 0/128, step exact 9/1024, scalar_offset 0/1024, final_residual 40/896. This confirms typed-register + primitive is a diagnostic scaffold, not the final universal LLM method. |
| Final answer bridge | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_final_answer_bridge_s120.yaml` | Preserves action-code 32/32 but LM causal forced-choice remains 0/8; final-answer CE alone does not solve neural value computation. |
| Role-value answer bridge | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_role_value_answer_bridge_s120.yaml` | S80 preserves action-code 32/32 but LM causal forced-choice stays 0/8 and bridge-off matches core8. The next candidate must update answer hidden state recurrently, not only add role-value tokens. |
| Ouro answer recurrent block | Accepted smoke probe, unstable scale-up | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_s080.yaml` | S80 preserves action-code 32/32 and improves LM causal forced-choice from donor/core-off/recurrent-off 0/8 to full 2/8. S240 and choice-margin continuations preserve action-code but regress LM smoke to 0/8, so lower train CE is not accepted without held-out answer-path gain. |
| Answer selective context router | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_selective_context_s080.yaml` | Minimal SubQ/SSA-style top-k router and dense-alignment v2 both preserve S80 score but have no causal effect: full 2/8 and router-off 2/8, action-code 32/32. Keep as scaffold only. |
| Answer finality selector | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_finality_selector_hardfirst_zeroshot.yaml` | Soft and hard-first selectors preserve action-code 32/32 but do not improve LM causal forced-choice: full 2/8 and selector-off 2/8. The transition trace must feed answer-state computation at each step, not only select a final depth after the answer loop. |
| Transition joint answer bridge | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_s020.yaml` | Injects transition joint logits into answer-state recurrence and preserves action-code 32/32, but held-out full and bridge-off both score 2/8. Seeing the transition trace is insufficient without an objective that makes the answer recurrence depend on it. |
| Transition joint answer bridge contrast | Accepted smoke probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_joint_answer_bridge_s020.yaml` | Adds a full-vs-bridge-off gold logprob contrast. S020 and S080-from-S020 do not improve full accuracy beyond S80, but preserve a causal bridge dependency: full 2/8, bridge-off 0/8, action-code 32/32. Next objective must break the 2/8 ceiling, not merely continue CE. |
| Ouro trajectory monotonic process credit | Rejected probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_trajectory_monotonic_s020.yaml` | Adjacent-depth target-logp monotonic margin preserves full 4/8 and action-code 32/32, but bridge-off also scores 4/8. The training loss was already zero on logged samples, so this does not attack the active final subtract-tail failure. |
| Ouro subtract-tail counterfactual loss | Rejected probe | `scripts/243_run_qtrm_ouro_subtract_tail_counterfactual_s020.sh` | Adds preterminal-sum and final +/-1 negatives. Bridge-off drops to 2/8 and action-code remains 32/32, but full core8 regresses to 3/8 while core_steps4 is 4/8. Treat as evidence for depth overshoot, not a canonical improvement. |
| Ouro terminal-depth CE | Rejected probe | `scripts/244_run_qtrm_ouro_terminal_depth_ce_s020.sh` | Applies CE only at finality-marked depths. Full core8 stays 4/8 and action-code 32/32, but bridge-off also reaches 4/8. Terminal CE is active but does not make the transition bridge causally useful. |
| Ouro answer halt head | Active accepted probe | `configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml` | PonderNet/ACT-style answer-state halt head trained with gate disabled, then evaluated with hard-first in-loop gate. Smoke8 full 8/8 and smoke16 full 10/16 while halt-gate-off is 0/16. Bridge-off ties full, so transition_joint_answer_bridge is demoted for this checkpoint. |
| Ouro autoregressive renderer | Rejected bottleneck | `lm_generation_smoke4_eval_gate.jsonl` | The accepted halt-head checkpoint is not generation-ready: greedy generation smoke4 scores 0/8 and emits invalid/noisy text. Low-rank answer-state LM adapter and causal-prefix self-rollout both preserve forced-choice 8/8 but still generate 0/8; greedy-margin adapter generates wrong digit streams and regresses forced-choice to 4/8. A beam-width-16 diagnostic also scores 0/4, so the failure is not greedy-only. Donor-unembedding head surgery scores generation 0/8 and can regress forced-choice to 6/8. LM-head-only decoder alignment scores generation 0/8 and regresses forced-choice full to 0/8. Hidden-bridge tuning scores generation 0/4 and regresses forced-choice full to 0/4 while bridge-off restores 4/4. Forced-choice reasoning and answer rendering must remain separate claims; the next candidate should preserve the donor decoder path rather than add more private QTRM renderer patches. |
| Mythos-style answer-loop update | Rejected probe | `scripts/256_run_qtrm_ouro_answer_loop_joint_decoder_s040.sh` | Stable input-injected answer recurrence plus loop-index, LoRA, ACT, and joint decoder training still scores generation 0/8; full forced-choice is 2/4 while decoder-off is 4/4. Keep OpenMythos/Parcae ideas as TRM-core stability references, not a second answer-side reasoning core. |

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
| Answer-state LM adapter | Rejected renderer probe | `answer_state_loop_lm_adapter_enabled`, `answer_state_loop_lm_adapter_only` | Zero-init low-rank output adapter is implemented and no-op safe, but S120 keeps generation at 0/8; greedy-margin S160 changes UNKNOWN loops into wrong digits and regresses causal forced-choice. Keep as tooling only. |
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

Keep the answer-halt S080 checkpoint as the current Ouro baseline and scale it
only with the same single-trace TRM contract:

```text
CONFIG=configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_eval_gate.yaml
CHECKPOINT=local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_halt_head_s080_from_tail_s020/last.pt
CORE_WORLD_MODEL_WEIGHT=0.0
MEMORYOS=off
DONOR_LOGITS_SCALE=0.0
```

Next acceptance gate:

```text
full > halt_gate_off
full >= core_steps4 on mixed-depth tasks
core_steps must adapt when terminal depth is not always 4
greedy generation must improve without donor-logit shortcuts; self-rollout
prefix CE, donor-output head surgery, LM-head-only tuning, and hidden-bridge
tuning are now rejected as insufficient
```
