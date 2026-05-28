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

### P1.4: RI-1 Dedicated — Test-Time Depth Scaling (Inspired by 2025-2026 Recurrent-Depth Literature)
**Background (latest papers)**:
- Geiping et al. (arXiv:2502.05171, Feb 2025) — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Huginn): Train a recurrent core with *sampled variable recurrence count* (log-normal Poisson). At test time, increase iterations on hard problems → strong monotonic gains on reasoning, especially math/coding. Key techniques: input injection every step for stability, truncated BPTT, adaptive early-exit via state change (KL/norm delta), KV-cache sharing across loops.
- LoopFormer (ICLR 2026): Elastic-depth via trajectory conditioning (condition on progress `t` and step size `Δt`) + shortcut-consistency loss. Allows smooth performance scaling across different budgets at inference without retraining.
- Other 2025-2026 directions: spectral-norm constraints for high-depth stability, training-free looping on frozen models, parallel loop execution.

**Relevance to our architecture**:
Now that Workspaces + Attractor + Provenance are properly ported and default-on in the main hybrid trainer (as of the recent changes), we have the *stability and state management primitives* that the latest recurrent-depth papers identify as critical for scaling depth without collapse.

**Lit validation (2026-06 session, research-driven-architecture-debugging skill)**:
- Huginn (arXiv:2502.05171 Geiping et al.): Confirmed log-normal Poisson recurrence sampling (heavy tail, locked-step per micro-batch), input injection every step for stability, truncated BPTT (k~8), sandwich norms. Exactly matches our partial core_elastic_depth randint scaffolding.
- LoopFormer (ICLR 2026): Confirmed trajectory conditioning (t + Δt) + shortcut-consistency loss for elastic budget scaling. Our Attractor (depth-wise monotonic on memory_buffer) + 3-track composition is the natural substrate for the consistency pressure.
- Local gap (humanistic preflight): Training always uses fixed short think_steps=4 (hardcoded in continuation trainer helpers); crude post-hoc --effective-depth proxy on 50-step synthetic base gave non-monotonic 37.5%/27.8%/31.9% (d=1/4/8). The "deeper is better" inductive bias was never trained; scaffolding in core/blocks existed but was not Reverse-I→G→A promoted into the active trainer + 3-track default path. This is now the #1 most-insufficient + highest-value item for RI-1.

**Concrete next experiments (post RI-4 closure)**:
1. **Variable Depth Training Schedule** (Huginn-style):
   - Modify `train_556_on_parallel_hybrid_minimal.py` (or the main RI-4 continuation trainer) to sample `outer_steps` / effective recurrence depth per batch or per sequence from a suitable distribution during training (start with small variance, gradually increase).
   - Leverage the already-ported Attractor for stability across variable unrolls.

2. **Adaptive Depth / Early-Exit at Inference**:
   - Add a simple per-token or per-sequence early-exit condition in the hybrid forward (e.g., stop when ||Δz_h|| or KL between consecutive iterations falls below threshold).
   - Measure on the extended raw intelligence gate: compute vs accuracy trade-off curves.

3. **Dedicated RI-1 Depth Sweep on Heldout** (once we have a model trained with the above):
   - Run clean depth sweeps (effective recurrence 1 / 4 / 8 / 12 / 16) on `pure_recursive_reasoning_heldout_72` (and harder extensions) with the full 5.56 + RI-4 recipe.
   - Compare against strong ablations: recurrence-off (pure attention), Attractor off, Workspaces off, stochastic off.
   - Primary metrics: exact accuracy + family breakdown + monotonicity of gains.

4. **Shortcut Consistency / Elastic Depth** (LoopFormer inspiration):
   - Add a lightweight consistency loss between short-trajectory and long-trajectory rollouts during training (align logits or final latent state of a "shortcut" path to a longer one with stop-gradient).
   - Goal: Make depth truly elastic so the same checkpoint gives smooth scaling across budgets.

**Success signal for this subsection**:
- First evidence that increasing test-time recurrence depth on the properly ported hybrid + three tracks produces larger, cleaner, more monotonic gains on pure reasoning heldout than previous non-ported or partially ported versions.
- Clear "depth scaling law" plot (accuracy vs effective depth) with ablations.

**2026-06 Update (M1 progress)**: 
- M1 stub + enhanced progress-biased sampling (deeper bias late in training) implemented and verified in trainer.
- 8-step clean M1 smoke (d=64) produced first monotonic strict-B accuracy on pure_72: 29.17% (d=1) → 31.94% (d=4) → 34.72% (d=8) vs pre-M1 degradation.
- Pipeline now fully measurable (clean ckpt save + robust loader).
- 20-step matched run launched for stronger signal. Target: sustained monotonic gain + higher absolute accuracy at d=8.

This P1.4 should be unblocked now that the three tracks are first-class citizens.

### RI-1 Milestones (참고: 2025-2026 Recurrent Depth Papers)

**목표**: Huginn(2502.05171)과 LoopFormer(ICLR 2026) 스타일로, 우리 하이브리드 + 3개 트랙 아키텍처에서 **test-time recurrence depth scaling**을 입증하는 것.

#### M1: Variable Depth Training 기본 적용 (1~2주)
- Trainer에서 recurrence depth(outer_steps 등)를 매 배치/시퀀스마다 sampling (작은 분포부터 시작).
- Attractor가 이미 들어와 있어서 stability가 어느 정도 확보된 상태에서 시작.
- 성공 기준: 학습이 안정적으로 돌아가고, 고정 depth 대비 최소한 동등한 성능.

#### M2: Depth Extrapolation + Stability 확보 (2~4주)
- 학습 시 max depth보다 inference 시 더 높은 depth에서 테스트.
- Huginn 스타일 input injection, LayerScale, spectral constraint 등 간단한 stability 기법 실험.
- 성공 기준: depth를 2배 이상 늘려도 상태가 폭주하거나 collapse하지 않음.

#### M3: Elastic Depth (LoopFormer 스타일) (4~6주)
- Trajectory conditioning + shortcut consistency loss 추가.
- 같은 체크포인트로 "적은 loop"와 "많은 loop" 모두에서 부드럽게 성능이 오르게 함.
- 성공 기준: inference budget을 조절했을 때 accuracy가 smooth하게 scaling.

#### M4: 첫 번째 Clear RI-1 결과 (6~10주)
- 제대로 훈련된 모델로 pure_recursive_reasoning_heldout_72 (또는 확장셋)에서 depth sweep 실험.
- Depth 1 → 4 → 8 → 12+ 에서 monotonic gain + ablation drop 확인.
- 성공 기준: "depth를 늘릴수록 정확도가 의미 있게, 일관되게 오른다"는 그래프 + clean ablation evidence.

#### M5: Adaptive Compute + 실전 적용
- Inference 시 KL divergence나 state change로 자동 early exit.
- Hard case는 더 깊게, easy case는 빠르게.
- 성공 기준: 평균 compute는 줄이면서 hard family 정확도는 유지하거나 향상.

이 마일스톤은 RI-4가 어느 정도 정리된 후 RI-1을 본격적으로 열기 위한 순서다.

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