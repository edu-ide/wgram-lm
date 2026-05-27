# Raw Intelligence PoC Execution Plan (2026-06)

**Purpose**: Make the [Raw Intelligence / Actual Reasoning Necessary Conditions SSOT](../wiki/decisions/raw-intelligence-necessary-conditions-2026-06.md) immediately executable on the current OneBodyParallelHybridBlock + real 642 gold + clean probe infrastructure.

This plan also directly advances several of the 7 S2 necessary conditions for 1B >> 27B (especially #2, #3, #4, #5). See the S2 PoC Verification Plan for the parent context.

This is the direct "how to complete the necessary conditions" companion to the S2 PoC Verification Plan.

---

## Priority 1 (Highest Value — Do This Week)

### P1.1: Extend Raw Intelligence Eval Harness to Hybrid Substrate
- **Target files**:
  - `src/qtrm_mm/eval/raw_intelligence_gate.py` — add new modes:
    - `hybrid_recurrence_depth_1/4/8/12`
    - `hybrid_sparse_memory_router_on` / `off` (once prototype exists)
    - `hybrid_556_full`, `hybrid_556_stoch_zero`, `hybrid_556_gold_off`, `hybrid_556_protection_off`
  - Add hybrid vs pure-recurrence comparison modes.
- **Data**: Extend `data/eval/pure_recursive_reasoning_heldout_72.jsonl` (and train splits) with:
  - Longer-horizon compositional cases (multi-"hop" latent state tracking).
  - Hard families that stress attractor stability and sparse selection.
- **Deliverable**: Updated eval script + at least one new 72-case (or larger) heldout JSONL that runs cleanly on the hybrid checkpoint.
- **Success signal**: The harness can produce "hybrid depth 8 full 5.56" vs all relevant ablations on the same cases used for S2 clean probes.

### P1.2: Run 5.56 Ablation Matrix on Hybrid with Both Cheap Proxy + Direct Raw Reasoning
- Use the existing `scripts/train_556_on_parallel_hybrid_minimal.py` + real 642 gold loading.
- At 80 / 120 step horizons:
  - Full recipe
  - stoch_zero (ablation_zero)
  - gold_off
  - protection_off
- For each arm, run:
  - The S2 clean `compute_pure_stochastic_contribution` + robustness probe.
  - The (extended) raw intelligence gate eval on the no-retrieval heldout.
- **Expected**: Large clean drops on both the proxy and the direct raw reasoning metrics when ablating the 5.56 components (especially stochastic breadth). This simultaneously satisfies S2 PoC #3 and RI-3.

### P1.3: 150–200 Step Horizon Scaling + Robustness on Real Gold (Hybrid)
- Extend the current 120/150 step runs to 200 steps where GPU memory allows.
- Apply the state robustness probe at multiple horizons (80/120/150/200).
- Measure whether hybrid + full 5.56 maintains robustness while historical baseline degrades (directly attacks RI-1 + RI-2).

---

## Priority 2 (Architecture — MSA Sparse Memory Inside Raw Reasoning)

### P2.1: Minimal Raven-style Sparse Slot Router Prototype inside OneBodyParallelHybridBlock
**Status (2026-06, Phase 3 landed)**: 

1. Persistent slot state across steps (done)
2. Stronger early read & deep integration into recurrence (done)
3. **RI-4 clean ablation modes + dedicated gate** (just completed - highest value measurement infrastructure)

We can now easily generate comparable records for:
- hybrid_sparse_slots_on_no_evidence
- hybrid_sparse_slots_off_no_evidence
- hybrid_persistent_memory_ablation_no_evidence

Plus `build_ri4_sparse_memory_gate()` in the eval harness for clean causal claims.

**Immediate high-value experiment command (now case-specific real evaluation):**
```bash
python scripts/train_556_on_parallel_hybrid_minimal.py --eval_ri4_heldout --steps 50 --enable_stochastic_breadth
```
Now runs real hybrid forwards on the actual heldout cases under the RI-4 configuration and produces proper records for the gate (using the case's own depth_targets for hit determination).
```bash
# Run full RI-4 (slots + persistence) on real heldout and get gate
python scripts/train_556_on_parallel_hybrid_minimal.py --eval_ri4_heldout --steps 50 --enable_stochastic_breadth

# Ablation: slots off
python ... --eval_ri4_heldout --ri4_slots_off

# Ablation: no selective persistence
python ... --eval_ri4_heldout --ri4_persistence_off
```
The script will now run the hybrid on the actual `pure_recursive_reasoning_heldout_72` and print a real `build_ri4_sparse_memory_gate` result.

Now the persistent slots:
- Are carried across recurrence steps
- Are queried early using current hidden as content-based query against carried slots
- The resulting rich memory context is injected **before** the recurrence heads run, so the iterative delta/GDN update actually builds its thinking on top of the memory
- Complementary fusion still happens post-recurrence

This is the key step that makes RI-4 memory causally participate in raw reasoning computation (previously it was mostly post-hoc additive).

**Core achievement**:
- `SparseSlotRouter` now has `apply_rehearsal_update(...)` — the 5.56 gold + attractor + decay signals are applied **selectively only to the router-chosen top-k slots**, while non-selected slots receive strong persistence (~0.92).
- This is exactly the "near-perfect persistence on untouched memory" that makes Raven/MSA powerful for long-horizon raw intelligence.
- The call site in `train_556_on_parallel_hybrid_minimal.py` now passes the router + mask during rehearsal when RI-4 is enabled.
- Reader side (already wired in previous step) + this selective write = first closed-loop sparse memory + 5.56 rehearsal system inside the One-Body Hybrid.

**Remaining micro-steps** (lower priority now):
- Maintain persistent `current_slots` across training steps inside the hybrid block.
- Full end-to-end smoke with real gold + clean probes comparing `router_on` vs `ablation_zero`.

**First usable smoke test command**:
```bash
python scripts/train_556_on_parallel_hybrid_minimal.py \
  --steps 40 \
  --use_sparse_slots true \
  --num_memory_slots 12 \
  --slot_top_k 4 \
  --real_gold_path data/gold/642_bos_latent.pt \
  --probe_every 10
```
Then compare `router_enabled` vs `ablation_zero` runs using the extended raw intelligence gate + pure stochastic effect.

This is the first concrete code artifact directly attacking the previously most insufficient condition (RI-4).

### P2.2: First Causal Measurement of Sparse Memory for Raw Intelligence
- Once P2.1 is in the hybrid block, train or fine-tune a short checkpoint with real-gold 5.56 curriculum.
- Run the extended raw intelligence eval with `router_on` vs `router_off` (dense) vs `topk=0`.
- Also measure long-horizon robustness (RI-2) with and without the sparse persistence.
- Success: Clear advantage for the sparse router version on compositional / long-horizon raw reasoning cases, with clean collapse when router is ablated.

---

## Priority 3 (Supporting Infrastructure & Documentation)

- Update `raw_intelligence_gate.py` summary output to include task-family labels and reasoning_family tags (as recommended in the old gates doc).
- Add a cheap "raw reasoning diversity under depth" metric (extension of the existing `_depth_output_diversity` logic) that can be computed on the S2 real-gold runs without full heldout eval.
- Create a small decision record or appendix in the RI SSOT documenting the first results from P1.2 + P2.1.
- Update the MSA memory architecture page and S2 PoC plan with links to this execution plan once the first data lands.

---

## Execution Order Recommendation (Strict "순서대로")

1. P1.1 (harness extension) — 1–2 days, unblocks everything.
2. P1.2 (5.56 ablation matrix on hybrid with raw reasoning metric) — highest scientific value right now.
3. P1.3 (long horizon robustness).
4. P2.1 (sparse slot prototype) — parallel track if bandwidth allows, because this is the architectural leap for RI-4.
5. P2.2 (causal measurement).

All of the above reuse the existing real 642 gold loading, clean probe code, and hybrid training scripts. No massive new infrastructure is required for the first wave.

---

**This plan turns the necessary conditions for raw intelligence into concrete, runnable experiments on the exact same assets that produced the ~5.5× S2 gap.** Once these PoCs pass with clean causal evidence, the project will have a much stronger foundation for claiming progress toward 1B-scale raw reasoning that can challenge 27B-class models.