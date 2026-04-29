# QTRM Recursive Core

Current code:

- `src/qtrm_mm/core.py`
- `src/qtrm_mm/heads.py`

Reference source:

- `docs/wiki/sources/tiny-recursive-models.md`
- `docs/wiki/sources/openmythos-recurrent-depth.md`

Status:

- Experimental, not yet TRM-faithful.

Current overlap:

- uses `z_l` and `z_h`
- has H/L cycle loops
- has optional truncation/detach
- has an optional halt head that emits `core_q_halt_logits` and
  `core_q_continue_logits`
- uses `StableInject` with spectral normalization, gating, and learned loop
  embedding, which overlaps with recurrent-depth stable input-injection ideas

Missing versus TRM:

- no persistent carry object
- no reset-on-halt semantics
- no no-grad H_cycles-1 schedule
- no detached carry returned for continuation across calls
- no halt exploration schedule
- no halt-head training loss against answer correctness
- current halt is batch-level, not per-sequence ACT

Missing versus Parcae/OpenMythos recurrent-depth references:

- no explicit prelude/recurrent/coda ablation around the recursive core
- no Parcae-style diagonal contraction update
- no recurrent depth sweep in validation
- no telemetry for contraction factor, recurrent state norm, recurrent residual,
  or loop-by-loop logit entropy
- no no-grad recurrence followed by a shorter backprop-through-depth window

Gate before long training:

- Decide whether the current batch-level halt hook is enough for the next
  experiment or whether QTRM needs true TRM carry/ACT behavior.
- If yes, refactor core around a carry object and add halting loss.
- If no, document the recursion as QTRM-specific, not TRM-faithful.
- Add recurrent-depth diagnostics before using longer training to interpret
  repeated-token collapse.
