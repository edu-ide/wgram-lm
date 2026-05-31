# 2026-05-29 Latest Research Aggressive Substrate Closure

Status: active implementation decision and audit ledger

Date: 2026-05-29

Trigger: the user explicitly requested that we keep digging through the latest
papers, check the wiki requirements, and aggressively change the existing
architecture or substrate if needed so the necessary raw-intelligence
evaluations become satisfiable.

## Big-Jump Framing

Old local minimum:

```text
Hybrid recurrence, MSA-style memory, and 5.56 stochastic breadth existed in
names, but several runtime/eval contracts made them either unreachable,
silently replaceable, or unverifiable.
```

Successful behavior preserved:

```text
The One-Body covenant is preserved: final answers must still come from the
normal recurrent thought state through the same LM-head route, not a retrieval
shortcut, sidecar oracle, or donor-only path.
```

Causal route changed:

```text
The active raw-intelligence route is now the OneBodyParallelHybridBlock with
strict GDN/GDN2 substrate handling, sparse persistent slots inside the recurrent
loop, explicit hybrid recurrence-depth eval modes, and a strict registry/gate
for the hybrid stochastic breadth replacement.
```

Fast falsification gate:

```text
Run the strict stochastic-breadth gate, the RI-4 A-mode hybrid smoke, and a
hybrid depth raw-intelligence gate. If any of those fail, the substrate is not
eligible for long promotion runs.
```

## Paper Evidence Behind The Closure

Primary source page:
[Latest Recurrent-Memory Substrate Papers](../sources/2026-latest-recurrent-memory-substrate-papers.md).

Key reads:

- Huginn/recurrent latent reasoning makes depth scaling a mandatory RI-1 axis.
- MSA makes sparse top-k/selective memory and untouched-slot persistence a
  mandatory RI-4 axis.
- Titans-style neural memory supports surprise/write-policy thinking, matching
  the 5.56 rehearsal requirement.
- Gated DeltaNet/GDN2 make strict substrate identity important. Silent fallback
  is an eval-corrupting bug, not a convenience.
- EqR/LT2 push the next cycle toward attractor residual, halt, and efficient
  loop telemetry.

## Implementation Closure

Code contracts closed in this pass:

- `src/wgram_lm/backends/__init__.py`: restored valid backend registry structure
  and exposed `torch_gated_delta2_v2` / `gated_delta2_v2` / `gdn2_v2` aliases.
- `src/wgram_lm/blocks.py`: strict official backend requests now fail fast
  instead of silently falling back; posterior guidance reads `self.cfg`; sparse
  slot masks are normalized before rich-memory masking.
- `src/wgram_lm/memory/sparse_slot_router.py`: public `slot_mask` contract is
  now 2D `(B, slots)`, with defensive handling for legacy 3D masks.
- `src/wgram_lm/config.py`: added the missing
  `core_elastic_depth_learn_policy` config field used by hybrid depth tooling.
- `src/wgram_lm/wgram_model.py`: answer-state loop now accepts the current
  three-item hybrid block result contract.
- `src/wgram_lm/architecture/component_registry.py`: registered the active
  `hybrid_stochastic_breadth_engine` so the strict SSOT gate does not reject the
  architecture merely because the old legacy component is inactive.
- `scripts/gates/check_ssot_stochastic_breadth.py`: strict mode now accepts the
  active hybrid replacement only when it is registered as a primary-path
  component.
- `src/wgram_lm/eval/raw_intelligence_gate.py`: added hybrid recurrence depth
  gate construction and RI-4 sparse persistent memory routing. The hybrid depth
  gate now rejects non-monotonic ladders, not only flat ladders. Added
  `hybrid_556_causal_matrix` for the full 5.56 recipe versus stochastic-zero,
  gold-off, protection-off, and decay-disabled ablations.
- `scripts/191_build_raw_intelligence_gate.py`: added gate types for
  `hybrid_recurrence_depth_scaling`, `ri4_sparse_persistent_memory`, and
  `hybrid_556_causal_matrix`.
- `scripts/192_eval_raw_intelligence.py`: added no-evidence hybrid depth,
  recurrence-off, stochastic-breadth-off, full 5.56 matrix, sparse-slot, and
  router/persistent memory ablation runtime modes.
- `scripts/smoke_ri4_a_mode_hybrid_recurrent_engine.py`: updated the smoke to
  verify direct recurrent, pure delegation, and 192-style real-tensor forced
  paths.
- `scripts/train_hybrid_ri4_real_continuation_minimal.py`: added safe unpacking
  for the current hybrid block tuple contract.
- `tests/test_ri_latest_architecture_closure.py`: new regression tests covering
  the above contracts.

## Requirement Status After This Closure

| RI condition | Current status | Promotion blocker |
|--------------|----------------|-------------------|
| RI-1 depth/test-time compute | Executable via hybrid depth modes and gate builder. | Needs full heldout matrix and monotonic gains from trained checkpoints. |
| RI-2 long-horizon stability | Partially executable through sparse slots and smoke stability. | Needs 80-200+ horizon real-gold robustness and attractor telemetry. |
| RI-3 5.56 causality | Full 5.56 matrix gate is executable: full, stoch_zero, gold_off, protection_off, decay_disabled. | Needs trained heldout runs showing clean drops for each ablation. |
| RI-4 sparse memory causality | Sparse slot/router/persistent-memory ablations are executable. | Needs trained router-on vs router-off and distractor/chunk-shuffle heldouts. |
| RI-5 hybrid synergy | Backend identity and hybrid runtime contracts are testable. | Needs hybrid vs recurrence-only vs attention-only trained ablation runs. |
| RI-6 low waste | Eval harness can now expose shortcut use. | Needs checkpoint curves showing mechanisms are used during training. |
| RI-7 data efficiency | Not closed by this pass. | Needs matched data-budget comparisons against weaker substrates. |

## Hard Line

This closure makes the necessary evaluations runnable and makes several hidden
substrate bugs fail loudly. It does not yet prove final raw intelligence. A
paper-grade claim still requires the long heldout runs listed above. If the
full matrix rejects, the next permitted moves are large substrate changes, not
small metric tweaks:

- fixed-point/attractor residual core;
- memory-as-primary recurrent thinker;
- deeper MSA slot hierarchy with chunk-shuffle hard negatives;
- non-recurrent parallel latent search if recurrence fails the causal gate.

## Fresh Verification

Verification run in this closure loop:

```text
PYTHONPATH=src:. .venv/bin/python -m compileall -q \
  src/wgram_lm \
  scripts/191_build_raw_intelligence_gate.py \
  scripts/192_eval_raw_intelligence.py \
  scripts/gates/check_ssot_stochastic_breadth.py \
  scripts/train_hybrid_ri4_real_continuation_minimal.py \
  scripts/smoke_ri4_a_mode_hybrid_recurrent_engine.py \
  tests/test_ri_latest_architecture_closure.py
Result: exit 0

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_ri_latest_architecture_closure \
  tests.test_raw_intelligence_gate_script \
  tests.test_raw_intelligence_eval_script \
  tests.test_raw_intelligence_gate \
  tests.test_gated_delta_adapter \
  tests.test_one_body_architecture_contract \
  tests.test_architecture_component_registry \
  tests.test_root_architecture_gate \
  tests.test_architecture_ablation_proof
Result: Ran 71 tests, OK

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python \
  -m scripts.gates.check_ssot_stochastic_breadth --strict
Result: PASS, active replacement OneBodyParallelHybridBlock,
registry record hybrid_stochastic_breadth_engine

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python \
  scripts/smoke_ri4_a_mode_hybrid_recurrent_engine.py
Result: all 4 RI-4 variants passed; pure delegation calls=5, slot_carries=9;
192-style forced path calls=3, slot_carries=5

scripts/191_build_raw_intelligence_gate.py \
  --gate-type hybrid_recurrence_depth_scaling \
  on synthetic monotonic depth records
Result: status=accepted, passed depth_scaling_gain_present,
depth_scaling_monotonic, deepest_hybrid_beats_recurrence_off,
stochastic_breadth_beats_zero_ablation, no_memoryos_shortcut

scripts/191_build_raw_intelligence_gate.py \
  --gate-type hybrid_556_causal_matrix \
  on synthetic full-vs-ablation matrix records
Result: status=accepted, passed full_beats_hybrid_556_stoch_zero_no_evidence,
full_beats_hybrid_556_gold_off_no_evidence,
full_beats_hybrid_556_protection_off_no_evidence,
full_beats_hybrid_556_decay_disabled_no_evidence, no_memoryos_shortcut
```

Operational note:

```text
The filesystem was 99% full during the first verification attempt and
`compileall` initially failed with OSError 28 while trying to write `.pyc`
files. No checkpoints or user data were deleted. After filesystem space
recovered, the normal compileall command above passed.
```
