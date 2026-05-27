# S2: Controlled Baseline Comparison - Detailed Experimental Design (초안)

**Date**: 2026-05-30

## 실험 목적
동일 조건에서 "과거 5.56 gold recipe" vs "현재 hybrid backbone + 5.56 recipe"를 정량적으로 비교하여, 새 구조가 실제로 state robustness를 향상시키는지 검증.

## 비교 그룹

**Group A (Baseline)**: 과거 5.56 gold recipe로 학습된 최고 수준 checkpoint (또는 최대한 비슷하게 재현한 버전)

**Group B (Candidate)**: OneBodyParallelHybridBlock (Gating v2 + official preference) + 동일 5.56 recipe

## 제어 변수 (최대한 동일하게)
- Curriculum 길이
- Scheduled binding decay schedule (0.40 → 0.04)
- Gold source 및 injection alpha
- Attractor protection strength (0.7)
- Rehearsal ratio 및 importance scoring
- Stochastic breadth 설정 (K, mode, scale)
- Seed set

## 측정 지표

1. **주요**: State Robustness under Ablation (proxy metric)
2. Stochastic breadth on vs off 효과 크기
3. Gold injection 효과 크기
4. Attractor protection 효과 크기
5. Curriculum 동안 norm trajectory 및 diversity

## Ablation Matrix (두 그룹 모두 동일하게 실행)

- Full 5.56
- Stochastic breadth zero
- Gold injection off
- Attractor protection during rehearsal off
- 기타 core/recurrence 관련 ablation

## 최소 실행 규모 (Tier 1)

- 최소 3~5 seed
- Curriculum 150 step 이상
- Proxy evaluation으로 충분한 signal 확인

## 성공 기준 (S0 Gate와 연동)

- Group B가 주요 robustness metric에서 Group A 대비 명확한 우위를 보일 때
- 동시에 모든 필수 ablation에서 causal contribution이 Group B에서도 확인될 때

---

**다음 micro-step**: S1이 어느 정도 진척된 후, 이 설계를 바탕으로 실제 실험 스크립트/러너를 준비.