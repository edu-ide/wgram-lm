# Stage119 Failure Ledger (from Stage118 regression)

## Observed Failure (Stage118)
- GD preference gate: accuracy 1.0, min_margin +0.022 (win)
- Direct generation exact: 0/12 on algebra traps (regression from prior anchors)
- Plain language: "preference signal learned, internal calculation routine absent in recurrent state"

## Evidence
- Repeated scalar preference / GD curriculum wins on narrow gate but consistent generation quality regression.
- State not forced to bind final equation operands/operation/result_var.
- LM head readback from recurrent state not enforced.

## Known Limitation Class
recurrent-state equation binding failure under misleading demonstration curriculum

## Root Architecture Hypothesis
The one-body path (token -> recurrent core -> same LM head) is intact, but the recurrent state on algebra trap trajectories is not required to explicitly represent the final equation. Additional final-answer contrast only teaches the speaker to prefer the token, not the thinker to compute/bind.

## Local Fix Budget Spent
Stage113-114 BPE-GD preference restoration + Stage117/118 generated algebra traps (multiple rounds of stronger negative / fixed-parrot data).

## Prior Work Checked
HRM-Text (reader -> recurrent thought -> speaker), QTRM one-body SSOT, typed register / belief state patterns from prior synthetic work.

## Universal LLM Path Preserved?
Yes (still token -> recurrent -> LM logits in the probe and intended 625 integration).

## Architecture Candidates Considered (this thread)
1. equation-state binding aux loss (chosen for minimal probe)
2. typed equation registers inside core (larger)
3. full readback-forced LM consistency term (follow-up)

## Recommended (Executed)
Minimal equation-state binding auxiliary (logit margin on components + readback enforcement) on algebra trap data only.

## Smallest Experiment & Result
scripts/627_run_stage119_equation_probe.py (synthetic, 15-20 steps, weight 0.3)
Result: exact 0.000 -> 0.125, verdict "probe", language 0.75 held.
Ablation in proxy not yet causal (micro model collapse).

## Decision
Record as L1 probe signal. Requires real anchor continuation for falsification. No promotion. No nearby scalar tweak on preference. If next real run shows same-family exact lift + state ablation drop on the same LM metric, promote to L2.

## Artifacts
- loss: src/qtrm_mm/losses/equation_state_binding.py (full compute + config)
- trainer hook: scripts/625 (guarded)
- runnable probe: scripts/627 (self-contained)
- this ledger + probe_results_2026-05-26.md

Timestamp: direct execution 2026-05-26
