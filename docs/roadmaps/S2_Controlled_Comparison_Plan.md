# S2: Controlled Baseline Comparison Plan (초안)

**Date**: 2026-05-30
**Phase**: PHASE S

## 목적
동일한 5.56 curriculum 조건에서:
- 과거 5.56 gold recipe (이전 backbone 기준 또는 재현)
- 현재 OneBodyParallelHybridBlock + 5.56 recipe

를 직접 비교하여, 새 구조가 실제로 더 나은 state robustness를 제공하는지 정량적으로 확인한다.

## 비교 대상

1. **Baseline A**: 과거 5.56 gold recipe로 학습된 최고 checkpoint (또는 그에 준하는 재현)
2. **Candidate B**: 현재 hybrid backbone (Gating v2 + official preference) + 동일 5.56 recipe로 학습

## 주요 비교 메트릭 (S0 Gate와 연동)

- State Robustness under Ablation (proxy for state_ablation_median)
- Stochastic breadth on vs off 효과 크기
- Gold injection 효과 크기
- Attractor protection 효과 크기
- Curriculum 내 norm stability / diversity

## 실험 설계 원칙

- Curriculum 길이, decay schedule, gold source, rehearsal 설정을 최대한 동일하게 맞춤
- 같은 seed set 사용
- Full ablation matrix 실행
- 다중 seed로 통계적 신뢰도 확보

## 성공 기준 (초안)

- Candidate B가 주요 robustness metric에서 Baseline A 대비 통계적으로 의미 있는 우위를 보일 때
- 동시에 모든 필수 ablation에서 causal contribution이 확인될 때

## 필요 리소스

- 5.56 curriculum을 hybrid block에서 돌릴 수 있는 최소 trainer (S1에서 준비)
- Proxy evaluation script (state robustness under ablation)
- 여러 seed 실행 환경

---

**다음 micro-step**: S2를 더 구체적인 실험 설계 문서로 발전시키거나, S1이 충분히 진척된 후 본격 실행.