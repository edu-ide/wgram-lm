# S3: Bottleneck Diagnosis Plan (초안)

**Date**: 2026-05-30

## 목적
S2에서 여전히 5.6을 넘지 못할 경우, **"왜 못 넘는가?"** 를 research-driven-architecture-debugging skill의 방법론으로 정확히 진단한다.

## 진단 원칙
- 단순히 "scale이 작아서", "학습이 부족해서" 같은 표면적 이유가 아니라, 실제 causal bottleneck을 찾는다.
- Humanistic preflight + ablation matrix + training dynamics vs downstream metric 괴리 분석을 종합.

## 주요 진단 축 (예시)

1. **Architecture 축**
   - Fusion gate가 recurrence branch의 신호를 제대로 살리지 못하는가?
   - Official component (GDN2, MLA)가 5.56 dynamics와 호환되지 않는가?

2. **Curriculum 축**
   - 5.56 recipe가 새 구조에 맞지 않게 설계되어 있는가?
   - Stochastic breadth의 injection 위치/강도가 suboptimal한가?

3. **Inductive Bias 축**
   - 5.56의 핵심 bias (stochastic breadth, gold structural bias 등)가 새 backbone의 inductive bias와 충돌하는가?

## 실행 방법
- S2 ablation matrix 전체를 다시 뜯어본다.
- 각 ablation이 실패하는 지점을 Humanistic 관점에서 분석.
- 작은 diagnostic experiment를 설계해서 bottleneck을 좁힌다.

---

**다음 micro-step**: S2 결과가 나오기 전까지는 상세 진단 checklist을 더 다듬는 정도로 준비.