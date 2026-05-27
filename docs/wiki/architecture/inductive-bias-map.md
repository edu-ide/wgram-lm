# Inductive Bias Map (Living Artifact)

**Purpose**: Tracks the key training dynamics and inductive biases that historically produced strong causal signals in this project. This complements `component_registry.py` (which tracks *components*) by focusing on *causal ingredients in training dynamics*.

**Maintenance Rule** (per research-driven-architecture-debugging skill):
- Updated on every I→G→A promotion.
- **Mandatory update** as part of any Historical Signal Reconstruction Gate during architectural pivots, safety stashes, or core replacements.
- Every entry must link to concrete evidence (wiki decision records, ablation tables, code paths).

---

## Entry: Stochastic Recurrent Breadth (GRAM/PTRM-style High-Level Guidance)

**Bias Name**: Stochastic high-level guidance during recurrence (prior/posterior Gaussian sampling + noise injection into z_h)

**Alternative Names**: true_gram transition mode, stochastic_high_level_guidance, GRAM/PTRM stochastic breadth

**Historical Strong Signals It Contributed To**:
- Stage56/58 PTRM family (high selected/oracle accuracy via K-candidate stochastic search).
- Adaptive Rehearsal 5.53~5.56 gold recipe (combined with scheduled binding decay + attractor protection on small compatible checkpoints; produced state_ablation_median ~5.53-5.56).
- Various early GRAM/PTRM-style recurrent cores using `StateTransitionCore` with `stochastic_transition_mode="true_gram"` or `stochastic_high_level_guidance=True`.

**Key Mechanism** (exact code paths in legacy implementation):
- `src/qtrm_mm/state_transition_core.py`:
  - `_apply_true_gram_transition` (lines ~781-809): Replaces z_h with sample from learned prior (or posterior) Gaussian: `z_next = mu + std * eps`.
  - `_apply_stochastic_high_level_guidance` (~743-768): Adds small stochastic delta after shared_core update.
  - Prior/posterior networks (`true_gram_prior_*`, `true_gram_posterior_*`, stochastic_guidance_*).
  - KL term (`_diagonal_gaussian_kl`).
  - Activated in `StateTransitionCore.forward` when `stochastic_transition_mode == "true_gram"` or `"delta"`.
- Used via `qwen_backbone_state_transition.py` (QwenBackboneWithStateTransition) and old trainers (510~523 series).

**Why It Mattered (Inductive Bias Effect)**:
- During *training*, forces the recurrent high-level state (z_h) to explore multiple noisy trajectories instead of collapsing to a single deterministic path.
- Creates explicit K-trajectory diversity inside the recurrence (not post-hoc).
- Posterior guidance allows answer-conditioned refinement of the distribution.
- This is distinct from (and stronger than) post-hoc multi-trajectory scoring over memory buffers.

**Current Status in Primary Path (updated 2026-06)**:
- Legacy `state_transition_core` remains library-only (`active_in_primary_onebody_path=False`).
- **2026-06 Restoration (I-stage)**: Self-contained learned prior (delta + true_gram modes) + generation logic added directly inside `OneBodyParallelHybridBlock` — the active RI-4 recurrent engine delegated from answer_state_loop. Trainer updated to exercise the internal prior. Full ablation_zero contract. Pivot Safety process + Reverse I→G→A record followed.
- See master roadmap: `docs/wiki/decisions/2026-06-missing-inductive-biases-restoration-roadmap.md` (M1 focus).

**Ablation Flag That Would Control It** (currently does not exist in primary core):
- None. The SSOT (`internal-multitrajectory-answer-attractor-ssot.md`) requires "GRAM/PTRM stochastic breadth off (K=1 vs K>1)" as a mandatory promotion gate ablation, but no executable control exists in the code being trained.

**Reverse I→G→A Status**:
- **Not yet initiated** (as of 2026-05).
- Historical Signal Reconstruction Gate triggered in 2026-05-30 decision record.
- See linked decision record for full audit and proposed port contract.

**Evidence Links**:
- Root diagnosis: conversation history + 2026-05-28-ablation-study-plan-literature-extensions.md (stochastic breadth off entry in Phase 4 plan).
- Legacy implementation: `src/qtrm_mm/state_transition_core.py:639-809, 1421-1459` (the "opt-in near-identity" comment at 640 is the pivot decision point).
- SSOT requirement: `docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md` (Promotion Gate and Required Ablations sections).
- Component registry: `src/qtrm_mm/architecture/component_registry.py:46-51`.
- Post-pivot core forward: `src/qtrm_mm/core.py` (no prior/posterior sampling, no stochastic delta on z_h inside recurrence).

**Notes**:
- This bias was deliberately kept isolated during the 2026-05-26 new-thought-structure pivot ("so legacy checkpoints remain comparable").
- The decision to isolate it was never followed by a Reverse I→G→A decision to either port it cleanly or document deliberate discard.
- This is the canonical example of the failure mode addressed by the "Pivot Safety, Historical Inductive Bias Preservation, and Reverse I→G→A" section of the research-driven-architecture-debugging skill.

---

## Entry: 642 Gold Structural Bias + Full 5.56 Adaptive Rehearsal Curriculum Dynamics

**Bias Name**: Composite training curriculum that produced the strongest historical state_ablation_median / hard-family answer quality signal (5.53~5.56)

**Alternative Names**: Adaptive Rehearsal 5.56 gold recipe, 642 gold + scheduled decay + attractor protection + stochastic breadth

**Historical Strong Signals It Contributed To**:
- All 5.53~5.56 Adaptive Rehearsal runs (the highest non-ablated numbers in the entire project history).
- 642 adaptive_fine_tuned / rehearsal_5p51_* series checkpoints (the actual gold artifacts that later experiments tried to "inject" from).
- Combined effect of: (a) 642 bos_latent / attractor-baked starting state, (b) external scheduled binding decay 0.40→0.04, (c) ALRMC importance + permanent high rehearsal weight on gold states, (d) explicit attractor protection during the rehearsal step itself (0.7), (e) stochastic recurrent breadth throughout the curriculum (K>1 noisy trajectories preventing collapse while the curriculum focuses).

**Key Mechanism** (the thing that was repeatedly only partially ported):
- Not a single module, but a **tightly coupled long-horizon training dynamic** inside the rehearsal loop:
  - `AdaptiveRehearsal.full_curriculum_rehearsal_step` (and its 5.56 wiring in the trainer)
  - `get_current_binding_weight()` scheduled decay
  - `inject_gold_state` modulated by current decay + `step_rehearsal` with `protect_attractor`
  - Core stochastic breadth applied on every step of the curriculum (the piece that was missing post-pivot until 2026-05-30 Reverse I→G→A)
- The curriculum repeatedly pulls the recurrent state toward the high-quality 642 gold basin while slowly lowering external pressure and allowing controlled stochastic exploration.

**Why It Mattered (Inductive Bias Effect)**:
- Gold states become "trusted" because they are consistently rehearsed with the right schedule and protection.
- The attractor learns exactly which states to defend.
- Stochastic breadth keeps the latent manifold from collapsing to a single deterministic path during the long focusing process.
- Removing any one piece (especially the stochastic breadth or the during-rehearsal protection) caused large drops in the final state_ablation_median.

**Current Status in Primary Path (as of feat/architecture-integration-2026-05)**:
- **Partial but now instrumented and executable**.
- Gold structural injection + scheduled decay + attractor protection during rehearsal: present in `adaptive_rehearsal.py` + wired in the 5.56 trainer.
- Real 642 gold loading: now possible via `--gold_path` + exhaustive historical key search in `load_gold_proxy` (2026-05-30 refinement).
- Stochastic breadth: newly ported to `QTRMRecursiveCore` + `core_stochastic_breadth_*` flags + ablation_zero identity gate (the I-stage of Reverse I→G→A).
- Full end-to-end curriculum loop + rich per-step diagnostics (bind_weight, gold_alpha_effective, stochastic_diversity, gold_dist, state_stability_proxy): now live in `train_556_full_curriculum_minimal.py` + launcher.
- `scripts/launch_556_local_smoke.sh` + resume support make long-horizon 5.56 reconstruction runs practical.

**Ablation / Control Flags** (all now exist and are ablatable in the primary trainer):
- `--enable_stochastic_breadth` / `--stochastic_ablation_zero`
- RehearsalConfig: `scheduled_binding_decay_start/end`, `protect_attractor`, `attractor_protection_during_rehearsal`, `gold_state_injection_alpha`
- `--gold_path` (real vs synthetic is the most important ablation for this bias)

**Reverse I→G→A Status**:
- **In active reconstruction (2026-05-30, 1-hour autonomous session progress)**.
  - Controlled on/zero ablation proof executed: diversity 4.04 vs exactly 0.0 (strong contract validation).
  - First real 642 gold + full 5.56 curriculum + stochastic run launched (50 steps). Defensive hardening applied for gold proxy fallback.
  - Full pipeline (trainer → launcher → matrix runner → analyzer) now exercised with real data.
- Historical Signal Reconstruction Gate applied: full 5.56 recipe inventoried from wiki + stashed 642 runs + old scripts.
- Stochastic breadth (the critical post-pivot missing piece) ported first as narrow I-stage (ablation_zero contract enforced and proven in execution).
- Gold loading + curriculum instrumentation + launcher + resume + ablation harness: this session.
- 50-step real 642 gold baseline completed (strong decay + stoch_div ~6.5).
- **180-step real 642 gold long run completed** (decay 0.400→0.042 range 0.358, stoch_div max 6.13 stable across full horizon, no degradation). Real gold path attempts consistently yield higher diversity (~6.0–6.5) than synthetic (~4.0) — pattern now in two runs.
- Still pending: Full ablation matrix on real 642 gold + downstream hard-family/state_ablation_median evidence before Promotion Gate.
- Historical Signal Reconstruction Gate applied: full 5.56 recipe inventoried from wiki + stashed 642 runs + old scripts.
- Stochastic breadth (the post-pivot missing piece) ported first as narrow I-stage (ablation_zero contract enforced).
- Gold loading + curriculum instrumentation + launcher + resume: this step.
- Living artifact: this entry + the 2026-05-30-deep-dive... decision record + executable trainer/launcher.
- Still pending: actual long runs (200-600 steps) on real 642 gold ckpt with full ablations, followed by Promotion Gate decision (promote the composite or document why it does not reproduce on current One-Body recurrence).

**Evidence Links**:
- Primary decision record: `docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md`
- Stochastic breadth audit + pivot gap: `docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md`
- Trainer + launcher: `scripts/train_556_full_curriculum_minimal.py`, `scripts/launch_556_local_smoke.sh`
- Rehearsal implementation: `src/qtrm_mm/rehearsal/adaptive_rehearsal.py`
- Inductive bias preservation rule: research-driven-architecture-debugging/SKILL.md (Pivot Safety + Reverse I→G→A section)

**Notes**:
- This is the single highest-priority remaining historical signal per all prior deep dives.
- The 2026-05-30 work (stochastic + gold loading + metrics + launcher) is the first time the *full composite* has been made executable + ablatable on the post-pivot One-Body architecture.
- Future runs using the launcher with real `--gold_path` + `--enable_stochastic_breadth` are the direct test of whether the original 5.5x basin is reproducible.

---

## Template for Future Entries

(When adding a new bias, copy this structure.)

**Bias Name**:
**Alternative Names**:
**Historical Strong Signals**:
**Key Mechanism** (with file:line):
**Why It Mattered**:
**Current Status in Primary Path**:
**Ablation Flag**:
**Reverse I→G→A Status**:
**Evidence Links**:
**Notes**:

---

**Last Updated**: 2026-05-30 (initial population) + 2026-06 updates for hybrid engine restoration work.

**Master Restoration Plan**: See `docs/wiki/decisions/2026-06-missing-inductive-biases-restoration-roadmap.md` for the full prioritized inventory, M0–M4 milestones, and step-by-step Reverse I→G→A playbook across all weak/missing historical biases.
**Cross Links**:
- Full D reconstruction across multiple tracks: docs/wiki/decisions/2026-05-30-historical-reconstruction-other-tracks.md
- Deep Dive on the single highest-priority remaining composite (Full 5.56 Curriculum): docs/wiki/decisions/2026-05-30-deep-dive-full-556-rehearsal-curriculum.md (executed 2026-05-30 as part of A~D sequence)
- Actual 5.56 Rehearsal Curriculum Smoke script + run (includes stochastic breadth): `scripts/diag_556_rehearsal_curriculum_smoke.py` (executed 2026-05-30)
- Production wiring example: `scripts/example_556_full_curriculum_wiring.py`
- Real trainer-style smoke: `scripts/train_556_rehearsal_smoke_real.py`
- Production-ready minimal trainer: `scripts/train_556_full_curriculum_minimal.py` (real QTRMRecursiveCore + full_curriculum_rehearsal_step + stochastic breadth support + checkpointing + basic best-metric logging + 642 gold proxy pattern, 2026-05-30)
- Practical launcher: `scripts/launch_556_local_smoke.sh` (easy local/DGX execution of the 5.56 curriculum trainer)
**Maintained By**: research-driven-architecture-debugging skill process.

---

## Entry: 570-style Depth-wise Monotonic Answer Attractor

**Bias Name**: Depth-wise monotonic pressure using actual recurrent memory buffer (current state must be better than recent K states under the LM head)

**Historical Strong Signals**:
- 570/601-style runs showing clear depth-wise intelligence margin gains.
- Strong interaction with Adaptive Rehearsal 5.5x (attractor protection was part of the gold recipe).

**Key Mechanism (Legacy)**:
- Row contrastive + softplus monotonic push away from worst recent states in memory buffer.
- Later evolved into counterfactual/meta-gate flavor (Stage101).

**Current Status in Primary Path**:
- Partially ported in current core (Mega attractor pressure + memory_buffer usage + depth bonus).
- Good progress, but full "true monotonic over actual recent K states + counterfactual variant" still being refined.

**Ablation**: `core_answer_attractor_ablation_zero`

**Reverse I→G→A Status**: I-stage largely complete; G-stage composition testing ongoing.

**Evidence**: 2026-05-28 ablation wiki, core.py attractor section, internal-multitrajectory-answer-attractor-ssot.md

---

## Entry: Elastic / Variable Recurrence Depth

**Bias Name**: Training with random/uniform depth + inference variable unroll (elastic depth)

**Historical Strong Signals**:
- Significant gains in generalization when models were trained with variable depth instead of fixed.

**Key Mechanism**:
- `core_elastic_depth_enabled` + `core_elastic_depth_train_random`.

**Current Status**:
- Flags and basic support exist in current core (effective_outer_steps logic).
- Real learning of depth policy (not just random) is still weak.

**Ablation**: `core_elastic_depth_ablation_zero`

**Reverse I→G→A Status**: Scaffolding present; full policy learning version pending stronger I-stage.

---

## Entry: Learned Slow-Tier Hierarchical Memory Policy

**Bias Name**: Learned 4-way decision head (load / evict / compress / ignore) over memory tiers with gold structural bias

**Historical Strong Signals**:
- Part of the "큰 점프" vision and Phase 0-3 mega integration. Strong structural gold bias effect observed in memory importance + slow decisions.

**Current Status**:
- `learned_slow_tier` module + gold bias injection implemented.
- Still early; policy is not yet strongly causal in most ablations.

**Ablation**: `core_learned_slow_tier_ablation_zero`

**Reverse I→G→A Status**: I-stage scaffolding done; needs dedicated narrow gate for the decision head itself.

---

## Entry: Gated Thought Workspaces + ALRMC-aligned Importance Broadcast

**Bias Name**: Multi-domain gated workspaces with importance-based selector (ALRMC-aligned) + broadcast back into recurrent state ("뇌량")

**Historical Strong Signals**:
- Phase1 4-way ablation: +0.06185 lift from importance selector vs naive sum.
- One of the strongest causal ownership results in the integration branch.

**Current Status**:
- Fully wired in current core (workspace_projs/gates + importance selector mode).
- One of the best examples of successful I→G→A so far.

**Ablation**: `core_thought_workspace_ablation_zero` + selector variants

**Reverse I→G→A Status**: Closest to full promotion among recent tracks.

**Evidence**: Phase1 diagnostics, core.py workspace section.