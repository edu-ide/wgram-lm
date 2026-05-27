# S5: Scale-up & Family-Balanced Validation Plan (초안)

**Date**: 2026-05-30

## 목적
충분한 규모와 family breadth에서 5.6을 명확히 넘는 증거를 확보하고, 최종적으로 "5.6을 뛰어넘었다"는 주장을 뒷받침할 수 있는 수준까지 도달한다.

## 주요 요구사항

- 여러 hard family에서 일관된 우위
- 더 긴 horizon (historical 5.56 수준 이상)
- OOD / 더 어려운 케이스에서도 robustness 유지
- 통계적 신뢰도 (다중 seed, confidence interval 등)
- 모든 필수 ablation에서 causal contribution 재확인

## 실행 조건

S4에서 S2 gate를 안정적으로 통과한 후에만 본격 진입.

## 최종 주장 조건 (S0 Gate와 연동)

- S0에서 정의한 Surpassing Gate를 명확히 통과
- Historical 5.56 gold 대비 우위가 재현 가능하고, 설명 가능하며, ablation으로 causal하게 뒷받침될 때

---

**다음 micro-step**: S4 결과가 나오기 전까지는 scale-up을 위한 최소 family set, horizon 길이, seed 수 등을 미리 정리하는 정도로 준비.