# QTRM Recursive Core

Current code:

- `src/qtrm_mm/core.py`
- `src/qtrm_mm/heads.py`

Reference source:

- `docs/wiki/sources/tiny-recursive-models.md`
- `docs/wiki/sources/openmythos-recurrent-depth.md`
- `docs/wiki/decisions/orthodox-trm-general-llm-direction.md`

Status:

- Canonical target for QTRM raw reasoning, but the current implementation is
  still QTRM-specific and not fully TRM-faithful.

Current overlap:

- uses `z_l` and `z_h`
- has H/L cycle loops
- has optional truncation/detach
- has an optional halt head that emits `core_q_halt_logits` and
  `core_q_continue_logits`
- initializes the halt head with zero weights and conservative negative bias
  via `core_halt_init_bias`
- has a generic `core_halt_loss` hook when `core_halt_targets` are supplied
- can infer conservative halt targets from exact token correctness, optional
  verifier pass/fail, and optional donor-KL stability
- can infer teacher-depth halt targets from per-depth residual logits, using
  centered-logit similarity to the full-depth output
- can train `core_q_halt_logits/core_q_continue_logits` with
  `core_halt_loss_mode=q_value`, so the head learns stop/continue value targets
  instead of only independent BCE flags
- exposes `core_depth_states` and optional `core_depth_last_logits` telemetry
- uses `StableInject` with spectral normalization, gating, and learned loop
  embedding, which overlaps with recurrent-depth stable input-injection ideas
- can run TRM-style no-grad inner H cycles via
  `core_trm_no_grad_inner_cycles_enabled`
- can freeze already halted batch rows while unfinished rows continue via
  `core_halt_freeze_halted_state_enabled`
- exposes `QTRMCoreCarry(z_l, z_h, halted, steps)` for detached continuation
  across calls, and `QTRMMultimodalModel(..., core_carry=...)`
- resets rows with `carry.halted=true` to the fresh workspace state on the next
  call while continuing unfinished rows from their previous latent state
- has a raw-intelligence eval harness mode
  `qtrm_core_halt_carry_steps_N_no_evidence` that reuses `core_carry` across
  token/prefix forwards in causal forced-choice and generation
- supports training-only halt exploration via `core_halt_exploration_prob` and
  `core_halt_exploration_min_steps`
- mixed-depth S160 training fixed the reasoning-dataset loader/finality/margin
  path and produced a depth8 core-joint CE smoke improvement, but the
  answer-halt gate is still not the causal source of the gain

Missing versus TRM:

- teacher-depth targets are still proxy stability labels, not semantic proof of
  which latent depth was actually necessary
- q-value halt loss is now available, but it still needs an accepted
  raw-reasoning ACT gate before being promoted as a proven compute policy
- answer-state-loop halt CE can train a halt signal, but held-out smoke shows
  gate-off still matches gate-on after S160, so ACT is unproven
- the task-level carry harness preserves the fixed depth-4 S080 smoke, but the
  S080 checkpoint is rejected as a mixed-depth ACT controller because carry
  modes do not beat best fixed-depth modes on smoke8

Missing versus Parcae/OpenMythos recurrent-depth references:

- no explicit prelude/recurrent/coda ablation around the recursive core
- coda now supports a separate `coda_attn_every` schedule so the recursive
  prefix can reach text logits through explicit attention even when the core
  keeps a sparse attention schedule
- no Parcae-style diagonal contraction update
- no recurrent depth sweep in validation
- no telemetry for contraction factor, recurrent state norm, recurrent residual,
  or loop-by-loop logit entropy
- TRM-style no-grad inner cycles exist, but no long-horizon recurrence
  validation has accepted them yet

Gate before long training:

- Treat the TRM/QTRM recursive core as the only primary reasoning core.
- Refactor toward true TRM carry/ACT behavior before claiming TRM-faithful
  reasoning: accepted mixed-depth ACT gates.
- Keep answer-side recurrence and Mythos-style loops as readout/stability
  probes unless they pass core-off, loop-off, depth, and generation gates.
- Require `core_state_zero` or equivalent trajectory corruption to reduce the
  claimed gain before calling a result core-causal.
- Add recurrent-depth diagnostics before using longer training to interpret
  repeated-token collapse.
