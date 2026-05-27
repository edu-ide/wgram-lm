# S3: Bottleneck Diagnosis - Detailed Framework (초안)

**Date**: 2026-05-30

## 목적
S2에서 5.6을 넘지 못할 경우, 정확한 이유를 skill 방법론으로 진단.

## 진단 프레임워크 (4축)

1. **Architecture Bottleneck**
   - Fusion이 recurrence signal을 제대로 전달하지 못하는가?
   - Official component가 5.56 dynamics와 호환되지 않는가?

2. **Curriculum / Recipe Bottleneck**
   - 5.56 recipe가 새 구조에 맞지 않게 설계되어 있는가?
   - Stochastic injection 위치/강도가 suboptimal한가?

3. **Inductive Bias Mismatch**
   - 새 backbone의 inductive bias가 5.56의 stochastic/gold bias와 잘 맞지 않는가?

4. **Evaluation / Scale Bottleneck**
   - 현재 proxy metric이 너무 약하거나 noisy해서 진짜 차이를 못 보는가?
   - Scale이 너무 작아서 신호가 안 보이는가?

## 실행 방법

- S2 ablation matrix 전체를 다시 뜯어보기
- 각 ablation이 실패하는 지점을 Humanistic 관점 + quantitative로 분석
- 작은 diagnostic experiment를 설계해서 bottleneck을 좁힌다.

## 산출물

- Bottleneck 후보 1~3개 선정 + 각각에 대한 증거
- 이를 해결하기 위한 최소 가설 (S4로 연결)

---

**다음 micro-step**: S2 결과가 나오기 전까지는 이 프레임워크를 더 구체화.