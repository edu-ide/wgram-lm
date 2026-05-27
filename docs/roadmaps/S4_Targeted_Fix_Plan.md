# S4: Targeted Improvement + New Gate Plan (초안)

**Date**: 2026-05-30

## 목적
S3에서 찾은 실제 bottleneck을 해결하기 위한 최소한의 개선을 하고, 개선 후 다시 S2 수준의 gate를 통과하는지 확인한다.

## 기본 원칙
- 큰 구조 변경보다는 pinpointed fix 위주.
- 개선 후 반드시 S2-style controlled comparison으로 효과를 검증.
- One-Body와 5.56 ablation contract를 절대 깨지 않게 주의.

## 예시 개선 방향 (S3 결과에 따라 달라짐)

- Fusion gate의 recurrence bias를 더 강하게 조정
- Stochastic breadth injection 위치/강도 최적화
- Rehearsal update를 fusion 후에 적용하는 방식으로 변경
- Official component 사용 시 발생하는 호환성 문제 해결 (예: MLA interface, GDN2와 rehearsal 상호작용)

## 실행 순서 (일반적)
1. S3 진단 결과로부터 1~2개의 가장 강력한 bottleneck 후보 선정
2. 최소 코드 변경으로 가설 검증 (작은 prototype)
3. 개선된 구조로 S2 수준 실험 재수행
4. S2 gate 통과 여부 확인
5. 통과하면 S5로, 실패하면 다시 S3로 피드백

---

**다음 micro-step**: S3 결과가 나오기 전까지는 일반적인 개선 프레임워크를 더 다듬는 정도로 준비.