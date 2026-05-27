# S4: Targeted Improvement + New Gate - Execution Framework (초안)

**Date**: 2026-05-30

## 기본 원칙
- S3에서 찾은 bottleneck 중 가장 강력한 1~2개를 골라 최소한의 변경으로 해결.
- 변경 후 반드시 S2 스타일 controlled comparison으로 효과 검증.
- One-Body와 5.56 ablation contract 절대 깨지 않게.

## 일반적인 개선 사이클

1. S3 진단 결과로부터 bottleneck 후보 선정 (우선순위 매김)
2. 최소 코드/하이퍼파라미터 변경으로 가설 검증 (작은 A/B)
3. 개선 버전으로 S2 수준 실험 재수행
4. S2 gate 통과 여부 확인
5. 통과하면 S5로, 실패하면 다시 S3로 피드백

## 예시 개선 방향 (S3 결과에 따라)

- Fusion gate의 recurrence bias 강화 (초기 bias 값 조정, temperature 조정)
- Stochastic breadth injection 위치 변경 (pre-fusion vs post-fusion)
- Rehearsal update를 fusion 후에 적용하는 방식으로 변경
- Official component 사용 시 발생하는 호환성 문제 해결 (예: MLA interface, GDN2와 rehearsal 상호작용)

## 산출물

- 각 bottleneck에 대한 "Fix Proposal + Expected Effect"
- 개선 후 S2 재실험 결과 리포트

---

**다음 micro-step**: S3 결과가 나오기 전까지는 개선 프레임워크를 더 다듬고, 자주 나올 법한 bottleneck 후보들에 대한 사전 준비.