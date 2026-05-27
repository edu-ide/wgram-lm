# 5.56 Adaptive Rehearsal Curriculum — Promotion Gate Evidence Package

**Date**: 2026-05-30  
**Branch**: feat/architecture-integration-2026-05  
**Status**: In Progress (Long 180-step real-gold run currently executing)

---

## Executive Summary (한글)

### 현재까지 증명된 것 (Proven)

1. **Stochastic Recurrent Breadth (Reverse I→G→A 핵심)** 
   - `core_stochastic_breadth_enabled + ablation_zero` contract가 실제 실행에서 완벽히 동작.
   - stochastic_ON: diversity max **4.0395** (synthetic validation) ~ **6.516** (real-gold attempt run)
   - stochastic_ablation_zero: diversity **정확히 0.0** (perfect identity)
   - 이는 post-pivot 아키텍처에서 GRAM/PTRM-style inductive bias를 성공적으로 복원한 첫 증거.

2. **Full 5.56 Curriculum Dynamics 실행 가능**
   - Scheduled binding decay (0.40 → 0.04) 정상 동작
   - Attractor protection during rehearsal (0.7) 100% active
   - Gold structural injection path (real 642 시도 시 hardening 적용)
   - 모든 요소가 `QTRMRecursiveCore` One-Body 안에서 ablation-first로 동작

3. **실제 642 Gold 연동 파이프라인 완성**
   - `--gold_path` 지원 + load_gold_proxy 다중 키 탐색
   - 1D proxy fallback 시 안전하게 injection 비활성화 (다른 dynamics 유지)
   - 50-step real-642 run 성공 완료 (decay + stochastic 모두 강한 신호)

### 아직 부족한 것 (Pending for Promotion)

- **Long-horizon real gold run 결과** (현재 180-step run 진행 중)
- Full ablation matrix on real 642 gold (decay off / protection off / stochastic off 조합)
- Downstream hard-family / state_ablation_median 스타일 메트릭 회복 여부
- 5.5x 수준의 숫자 재현 여부 (아직 proxy 수준)

---

## Executive Summary (English)

### Proven

- Stochastic breadth ablation contract holds cleanly in real execution.
- Full 5.56 curriculum (decay + protection + stochastic during rehearsal) is executable and measurable on the current One-Body architecture.
- Real 642 gold loading path + defensive hardening is working.
- First meaningful real-gold + stochastic run (50 steps) completed successfully with strong signals.

### Pending

- Results from the current long 180-step real-gold run.
- Complete ablation matrix on actual 642 gold checkpoints.
- Evidence that the original 5.53~5.56 basin can be recovered (or honest documentation if it cannot).

---

## Package Contents

- `01_ablation_proof/` — Controlled stochastic ON vs ZERO validation (strongest contract evidence)
- `02_real_gold_runs/` — All runs that used actual 642 checkpoints (50-step completed + 180-step in progress)
- `03_code_artifacts/` — Key code changes and new tools (trainer, launcher, matrix runner, analyzer)
- `04_registry_and_wiki_updates/` — component_registry and wiki decision records
- `05_ongoing_180step_run/` — Placeholder + instructions for the long run currently executing
- `scripts/` — Helper scripts for final package assembly

---

## Quick Start

**Read this first**: `CURRENT_EVIDENCE_SNAPSHOT.md`

## How to Use This Package (When the Long Run Finishes)

1. Wait for `local_556_real642_long_180step_20260527_1201` to complete.
2. Run:
   ```bash
   python scripts/analyze_556_curriculum_metrics.py \
       local_556_real642_long_180step_20260527_1201/metrics.json \
       --output 05_ongoing_180step_run/180step_real642_analysis.md
   ```
3. Copy the report into this package.
4. Update this README with final numbers.
5. Decide Promotion Gate outcome.

---

**Maintained under**: research-driven-architecture-debugging skill  
**I→G→A Phase**: Reverse I→G→A (Historical Signal Reconstruction of 5.56 gold recipe)

## 2026-05-30 Update: 180-step Real Gold Run Completed

- Decay: 0.400 → 0.042 (range 0.358)
- Stochastic diversity: max 6.13 (mean ~5.99)
- Full artifacts + analyzer report integrated into 
- Package is now up-to-date with the longest real-gold curriculum run to date.
