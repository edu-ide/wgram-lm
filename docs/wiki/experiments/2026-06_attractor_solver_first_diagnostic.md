# First Explicit Attractor Solver Substrate Diagnostic (June 2026)

**Status**: Ready to run (prototype + wiring script complete)

**Current Compressed Daily Milestone (하루 마일스톤 - 2026-06, "너가 선택해" 이후)**

이 문서 상단이 현재 **실행 중인 단기 마일스톤**의 single source of truth다. (사용자 "장기 마일스톤 하루로 줄여" 지시 후 압축된 버전)

### 지금 단계 (Phase 2 - Real Internalization + Denoising Signal)
**오전**: 이전 결과 판단
**오후**: RealHybridProposal v2 + item #4 prep (--internal_fast_recurrent simulation) + wiring demo + 30-step 실행 (완료)
**저녁**: 결과 기록 + commit (완료) — 다음: item #4 prep 더 구체화 또는 trainer integration light prep

**Priority 1 (최우선, Risk #1 직접 타격 - 추천)**  
**Real proposal engine로 internalization 숫자 뽑기**
- `scripts/diag_explicit_attractor_solver_20step.py` 의 toy `proposal_proj` 를 실제 `OneBodyParallelHybridBlock + BrainMimeticTripleMemory` 호출로 교체 (trainer의 `_hybrid_forward_only` 패턴 재사용)
- SOT trainer 호출 시 `proposal_engine` 인자 제대로 넘겨서 `internalization_loss` 가 0이 아닌 값으로 나오게 만들기
- Success criteria: 20~30 step run에서 `int` loss가 의미있게 >0 이면서 점차 ↓ 하는 곡선 관찰
- 이게 안 되면 "rich proposal이 이미 equilibrium에 너무 가까워서 solver가 의미 없어지는" Risk #1이 현실이 됨

**Priority 2 (내가 "너가 선택해" 때 judgment로 제시한 modest next push)**  
**Light denoising auxiliary + timestep conditioning stub 구현**
- `AttractorSolverModule.step_with_noise_schedule` 스텁을 최소한으로 실제화
- Diagnostic script에 `--denoise_loss_weight` 추가 (noisy proposal → clean equilibrium을 맞추는 aux loss)
- Constant noise 0.2~0.3 환경에서 denoising loss 유무에 따른 recovery / densing_sig 비교
- Success: denoising term이 recovery를 더 안정적으로 만들거나, densing_sig를 개선하는 신호

**Priority 3**  
SOT / internalization weight ablation matrix 중 critical 2~3개 (A5, A7 등) diagnostic script로 빠르게 돌리기

**Phase 3 진입 조건** (위 1~2가 어느 정도 되면)
- Full trainer integration (equilibrium를 main loss path에 연결) — minimal demo started inside diagnostic via --demo_equilibrium_wiring
- Native strict-B 72 heldout RI-1 측정 시작

**주의**: 모든 작업은 "RI-1~RI-7 깨끗하게 검증 가능한 상태"를 절대 해치지 않으면서 진행. toy harness에서 real wiring으로 넘어가기 전까지는 72 측정 스크립트 손대지 말 것.

**Strategic Context (Densing Law Framing)**: This diagnostic is the minimal instrument for measuring *Inference Densing* gains from the Section 7 substrate (see main wiki Section 7 "Densing Law Framing of the Overhaul" and 7.1 Risks). Success is ultimately defined not only by raw accuracy but by improved performance per inference FLOP (solver steps / effective recurrence depth vs. answer quality). See also the IKP nuance (arXiv:2604.24827) on procedural vs. factual capacity.

**Goal**: Smallest falsifiable 20–50 step diagnostic for the Section 7 substrate (Proposal Engine + Dedicated AttractorSolverModule + Parcae stability + SOT + Equilibrium Internalization) before promoting to full native 72 RI-1 measurement.

## Recommended First Command (copy-paste)

```bash
python scripts/diag_explicit_attractor_solver_20step.py \
  --steps 20 \
  --sot_segment_length 5 \
  --attractor_solver_weight 0.15 \
  --internalization_weight 0.12 \
  --out_dir checkpoints/diag_attractor_solver_20step \
  --log_every 2
```

Expected healthy signals (first 20 steps):
- `solver_res` monotonically or mostly decreasing (Parcae + persistent injection working)
- `int` (internalization) beginning to show non-zero and decreasing trend once the proposal projection is replaced by the real hybrid citizen
- No NaN / loss explosion

## Minimal Ablation Matrix (Priority Order for First Wave)

Run each for 20–40 steps. Log solver_res, internalization_loss, primary_on_equilibrium, total.

| ID | Command delta | What it tests | Expected diagnostic value | Priority |
|----|---------------|---------------|---------------------------|----------|
| A0 | (baseline above) | Full stack (Parcae + SOT h=5 + int_w=0.12) | Reference curve | 1 |
| A1 | `--sot_segment_length 3` | Shorter SOT segments (more frequent opt steps) | Does more frequent landscape shaping help or hurt early internalization? | 2 |
| A2 | `--sot_segment_length 7` | Longer segments | Closer to classic unroll behavior | 2 |
| A3 | `--attractor_solver_weight 0.05` | Very light solver pressure | Does the solver still refine when the auxiliary is weak? | 3 |
| A4 | `--attractor_solver_weight 0.25` | Strong solver pressure | Early sign of loss interference (risk #3) | 3 |
| A5 | `--internalization_weight 0.05` | Weak internalization curriculum | How much does the "drop solver at inference" promise depend on strong L_int? | 1 |
| A6 | `--internalization_weight 0.20` | Strong internalization | Risk of proposal collapse or gradient conflict | 4 |
| A7 | (add `--parcae_negative_diag_enabled false` after wiring) | No Parcae stability | Does residual explode or training become unstable? (directly tests Parcae contribution) | 1 (critical) |
| A8 | `--ri_ni_scale 0.0` | No RI/NI (EqR basin shaping off) | Spurious attractor formation speed (risk #2) | 2 |

## Integration Roadmap (after clean 20-step)

1. Confirm residual ↓ + internalization starting to move in the minimal harness.
2. Replace the toy `proposal_proj` with a real call to the existing `OneBodyParallelHybridBlock + BrainMimeticTripleMemory` (reuse the `_hybrid_forward_only` pattern from the main trainer).
3. Wire the equilibrium output into the final LM head path (or add it as a parallel strong loss term) inside the main training loop.
4. Run the same 20-step command with `--brain_triple_memory --internal_fast_recurrent --use_explicit_attractor_solver` using the real proposal engine.
5. If A7 (no Parcae) shows clear instability → Parcae negative diagonal is validated as necessary.
6. Only after the above: promote to full native batched 72 heldout RI-1 measurement under the same strict-B + Principle Gate contract.

## Current Known Early Observation (from first 20-step runs)

In the ultra-minimal harness (toy linear proposal), `internalization_loss` stayed 0.0 because proposal and equilibrium are too close by construction. This is actually **useful data** — it previews risk #1 ("Equilibrium Internalization Fails to Materialize") when the proposal engine is already strong.

**2026-06 Priority 1 Result (Real/Rich Proposal Test)**:
- Added `--rich_proposal` mode using `RichProposalStub` (real `BrainMimeticTripleMemory` + fast recurrence stub).
- **30-step run (v2, improved stub)**:
  - `int` (SOT internalization): 0.00345 (step 1) → 0.00020 (step 30) — clear monotonic decrease.
  - `int_mse` (solver.compute_internalization_progress): 0.03452 → 0.00201 — proposal is measurably getting closer to equilibrium.
- This is the **first credible internalization learning curve** we have in the diagnostic harness.
- Implication: When the proposal carries actual slow memory signal, the internalization objective does real work. The curriculum is shaping the proposal engine toward the attractor.
- solver residual stayed healthy throughout. No stability issues from the richer proposal.
- Strong positive evidence against 7.1 Risk #1. We now have a repeatable minimal instrument to track internalization progress before full trainer wiring.

When the real rich hybrid citizen (FastGated + TripleMemory + ChunkedSlow) becomes the proposal engine, we expect `int_loss` to become meaningful and then (hopefully) decrease over training. This must be the first thing measured in the next wiring iteration.

**2026-06 Real Hybrid Proposal Attempt (Roadmap #2)**:
- v1: Basic hybrid stack + defensive fallback (int decreasing but limited fidelity).
- v2 (current): RealHybridProposal now follows the trainer's `_hybrid_forward_only` pattern more closely:
  - Carries `InferenceState` (fast_recurrent_h + step_count)
  - Proper tuple unpacking from hybrid layers
  - Passes `fast_recurrent_state` to layers
- 20-step run result:
  - `int` loss: 0.00395 (step 1) → 0.00058 (step 20) — strong, consistent decrease.
  - No crashes, stable residual.
  - This is the cleanest internalization signal we have obtained while using the actual OneBodyParallelHybridBlock stack inside the diagnostic.
- Status: Meaningful progress on "replace toy with real call to OneBodyParallelHybridBlock". The pattern reuse is working.
- Added minimal `--demo_equilibrium_wiring` (Roadmap item #3 prep): treats the equilibrium y* as the explicit "final output representation" that would go to the LM head. This is the first small wiring signal inside the diagnostic harness.
- Strengthened wiring demo run (30 steps): 
  - `int` loss: ~0.00368 → 0.00034 (very healthy decrease)
  - Equilibrium is now explicitly treated as the primary final output representation in the loss.
- This constitutes a working minimal demonstration of Roadmap item #3 inside the diagnostic harness.
- v5 (latest): Added `--internal_fast_recurrent` simulation (item #4 prep) — RealHybridProposal does 4 micro-steps when enabled. 30-step run with real hybrid + wiring + internal fast recurrent: int 0.00381 → 0.00031.
- This is the first time item #2/3/4 concepts were exercised together inside the diagnostic harness.

All numbers and failure modes will be appended here after the first real rich-proposal runs.

**Long-term Direction Note (Diffusion-Style Attractor Iteration)**

See main wiki Section 7 for the full description of the long-term "Proposal as noisy latent + Solver as learned denoiser" direction.

First small experiments to consider:
- Add controlled Gaussian (and later structured) noise to the proposal before solver steps.
- Implement simple linear/cosine noise schedule wrapper.
- Measure whether explicit noise + timestep conditioning improves:
  - Recovery from bad/spurious initial proposals
  - Final equilibrium quality
  - Inference Densing curve (quality vs. total solver steps)
- Compare against current plain attractor solver baseline.

This direction is deliberately exploratory and should only be pursued after the near-term and medium-term improvements (selective looping, curriculum internalization, densing metrics) show stable signals.
