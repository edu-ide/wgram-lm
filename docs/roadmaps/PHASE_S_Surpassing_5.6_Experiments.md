# PHASE S: Surpassing 5.6 Experiments Phase

**작성일**: 2026-05-30  
**상태**: 새로 시작하는 단계 (사용자 명시적 요청: "이제부터 진짜 5.6을 넘는 실험을 제대로 돌려보자")

---

## 배경과 목적

지금까지의 모든 작업 (Reverse I→G→A, Prior Contract, Gating v2, Parallel Hybrid Head, official GDN2/MLA 도입 등)의 **최종 목적**은 다음과 같다:

> 5.53~5.56에서 나왔던 역사적 강력 신호(특히 stochastic recurrent breadth + Adaptive Rehearsal curriculum + 642 gold + attractor protection)를, 더 강력하고 현대적인 backbone 위에서 제대로 살려서, **결국 5.6을 명확히 뛰어넘는 성능을 내는 것**.

백본 수정을 한 이유는 "최신 기술 따라가기"가 아니라, **5.6을 넘기 위한 더 나은 그릇**을 만들기 위함이었다.

이제부터는 구조 개선 작업의 무게를 줄이고, **실제로 5.6을 넘는 실험**에 집중하는 단계로 전환한다.

---

## PHASE S 전체 전략 (순서대로)

우리는 다시 한번 철저한 "순서대로"를 적용한다.

### S0: Surpassing 5.6 Gate 정의 (가장 중요)
- 5.6을 "넘었다"고 주장하기 위한 명확한 기준을 세운다.
- 어떤 metric? 어떤 curriculum? 어떤 ablation? 어떤 threshold?
- 이 단계가 끝나기 전에는 대규모 실험을 하지 않는다.

### S1: 5.56 Curriculum의 충실한 재현 (새 backbone 위에서)
- 과거 5.56 gold recipe (scheduled binding decay, gold injection, attractor protection during rehearsal, stochastic breadth 등)를 현재 backbone에서 최대한 비슷하게 재현한다.
- 모든 필수 ablation contract가 여전히 살아있는지 확인.
- 이 단계가 통과되지 않으면 "새 구조가 5.6을 넘을 수 있다"고 주장할 자격이 없다.

### S2: Controlled Baseline Comparison (소규모)
- 동일한 5.56 curriculum 조건에서:
  - 과거 5.56 gold recipe (이전 backbone or 재현)
  - 현재 backbone + 5.56 recipe
- 를 직접 비교.
- 핵심은 "stochastic breadth on/off", "gold injection on/off" 등의 ablation에서 차이가 나는지 보는 것.

### S3: Bottleneck Diagnosis (5.6을 못 넘는 이유 찾기)
- S2에서 여전히 5.6을 넘지 못한다면, **왜 못 넘는지**를 skill의 방법론으로 정확히 진단.
- 단순히 "scale이 작아서"가 아니라, 실제 causal bottleneck을 찾는다.

### S4: Targeted Improvement + New Gate
- S3에서 찾은 bottleneck을 해결하기 위한 최소한의 개선.
- 개선 후 다시 S2 수준의 gate를 통과해야 다음 단계로.

### S5: Scale-up & Family-Balanced Validation
- 충분한 scale에서 5.6을 명확히 넘는 증거를 모은다.
- 여러 family, 더 긴 horizon, 더 어려운 OOD 등에서 검증.

---

## PHASE S 전체 레벨 요약 (S0 ~ S5)

PHASE S는 총 **6개 레벨** (S0 포함)로 구성되어 있습니다.

- **S0**: Surpassing 5.6 Gate 정의 (가장 중요, 아직 refinement 중)
- **S1**: 5.56 Curriculum의 충실한 재현 (현재 진행 중)
- **S2**: Controlled Baseline Comparison (소규모)
- **S3**: Bottleneck Diagnosis
- **S4**: Targeted Improvement + New Gate
- **S5**: Scale-up & Family-Balanced Validation

**추가 상세 계획 문서 생성 완료**:
- S2_Controlled_Comparison_Detailed_Plan.md
- S3_Bottleneck_Diagnosis_Detailed_Plan.md
- S4_Targeted_Fix_Detailed_Plan.md
- S5_Scale_up_Detailed_Plan.md

사용자 "모든 S 다해" 지시에 따라, S0~S5 전체를 순서대로 밀고 나갈 예정입니다.

현재 상태:
- S0: 초안 작성 + refinement 진행 중 (사용자 이전 선택: 1번)
- S1: 실행 시작 (S1 전체 계획 문서 + S1.1 매핑 문서 작성 완료)
- S2~S5: 아직 시작 전 (S1이 어느 정도 진척되면 순차 진입)

---

## S2 초기 계획 (미리 작성)

**목표**: 동일한 5.56 curriculum 조건에서 과거 5.56 gold recipe vs 현재 backbone + 5.56 recipe를 직접 비교.

**주요 비교 포인트**:
- state robustness under ablation (proxy metric)
- stochastic breadth on vs off 효과 크기
- gold injection 효과 크기
- attractor protection 효과 크기

**최소 조건**:
- 동일 curriculum 길이
- 같은 gold source (가능하면 real 642)
- 같은 seed set으로 비교
- Full ablation matrix 실행

**성공 기준 (초안)**:
- 새 구조 쪽이 주요 metric에서 통계적으로 의미 있는 우위를 보이면서, ablation에서도 causal contribution이 확인될 때.

---

## S3 초기 계획 (미리 작성)

**목표** (S2에서 5.6을 못 넘을 경우):
- "왜 못 넘는가?"를 research-driven-architecture-debugging skill 방법론으로 정확히 진단.
- 단순 "scale 부족"이 아니라 실제 causal bottleneck을 찾음.

**진단 방법**:
- Humanistic preflight (Reader/Thinker/Speaker 관점)
- Ablation matrix 전체 분석
- Training dynamics vs downstream metric 괴리 분석
- One-Body contract 위반 지점 탐색

---

## S4 초기 계획 (미리 작성)

**목표**:
- S3에서 찾은 bottleneck을 해결하기 위한 최소한의 개선.
- 개선 후 다시 S2 수준 gate 통과.

**예시 개선 방향** (가설):
- Fusion gate의 recurrence bias 강화
- Stochastic injection 위치 조정
- Rehearsal update를 fusion 후에 적용
- Official component 호환성 문제 해결

---

## S5 초기 계획 (미리 작성)

**목표**:
- 충분한 규모와 family breadth에서 5.6을 명확히 넘는 증거 확보.
- 여러 hard family, 더 긴 horizon, OOD 케이스 포함.
- 통계적 신뢰도 확보 (다중 seed, confidence interval 등).

**최종 주장 조건**:
- S0에서 정의한 Surpassing Gate를 명확히 통과.
- 모든 필수 ablation에서 causal contribution 확인.
- 이전 5.56 gold 대비 우위가 재현 가능하고 설명 가능할 때.

---

**현재 진행 계획 (사용자 "모든 S 다해" 지시 반영)**

S0 refinement를 병렬로 계속하면서, S1을 본격적으로 밀고, S2~S5의 초기 계획을 미리 세워서 전체 로드맵을 명확히 하겠습니다.

**지금까지 실행한 S 레벨 현황 (2026-05-30, 사용자 "S 몇까지 있는거야 그냥 모든 S 다해" 지시 직후)**:

**완료**
- **S0**: 완전 LOCKED (`S0_Surpassing_5.6_Gate_LOCKED.md`)
  - Tier 1 primary: Stochastic Diversity under real 642 gold + full curriculum (historical full ~5.99–6.13)
  - Stochastic zero must collapse to ~0 (perfect contract from G-stage evidence)
  - Material drop bars + success thresholds + Tier 1 vs Tier 2 distinction 모두 구체 숫자로 확정
- **S1.1**: 4개 Open Decision 완전 결론 + 기록 (`S1.1_Decision_Record_4_Open_Questions.md`)
  - Stochastic: recurrence branch only, pre-fusion (ablation identity 최우선)
  - Gold: primary pre-fusion on recurrence
  - Attractor protection: post-fusion fused state
  - Rehearsal pull: post-fusion
- **S1.2**: Minimal Trainer Prototype 완성 (`scripts/train_556_on_parallel_hybrid_minimal.py`)
  - OneBodyParallelHybridBlock + official MLA + bfloat16 즉시 실행 가능
  - S1.1 4개 결정 하드와이어 + S0 gate 체크 로직 내장
- **S1.3 ~ S1.5 완료 (2026-05-30 ~ 06-01)**:
  - Clean metric + S1.5 faithful 5.56 rehearsal simulation
  - **100-step long horizon validation 성공**:
    - Full: pure_stoch 1.32~1.46 (매우 안정, 끝 1.3984)
    - Zero: 정확히 0.0000 (100 step 전체)
    - Robustness: 1.000
  - S1 핵심 causal contract (stochastic breadth ownership)이 100 step까지 견고하게 증명됨
  - S2를 위한 hybrid side 고품질 baseline 데이터 확보 완료 (S2 진입 직전 단계)

**현재 Active Phase: S2 (Controlled Comparison) — Full Aggressive Push (2026-06-01)**
- S2 공식 진입 후 **A, B, C 전부 동시에 실행**
  - A (Historical data): Evidence package deep dive (50-step metrics.json에서 stochastic_diversity + state_stability_proxy 발견 → mapping 가능)
  - B (Hybrid data): 3 seeds 100-step 수집 완료 (pure ~1.398 / 1.453 / 1.461, average 1.437, zero arm 0.0000 완벽)
  - C (Script): S2 framework + ablation matrix skeleton + multi-run aggregation 강화
- `s2_collect_hybrid_data.py` helper 생성
- Hybrid multi-seed baseline가 S2 비교를 위한 강력한 기반이 됨

**S0~S5 전체 로드맵 상태**: S2 Full Push 진행 중 (A+B+C 동시 실행). Hybrid multi-seed 100-step baseline 확보. Historical mapping 작업 병행. "모든 S 다" 지시 최우선으로 S2를 밀고 있음.

---

## S0: Surpassing 5.6 Gate 정의 (지금 당장 시작해야 할 일)

이 단계에서는 아래 질문에 **구체적인 숫자와 조건**으로 답을 내야 한다.

### 필수로 정의해야 할 항목

1. **주요 성공 지표 (Primary Metric)**
   - 예: state_ablation_median (또는 그에 준하는 metric)
   - 또는 특정 hard family에서의 exact accuracy
   - 또는 전체 curriculum에서의 평균 performance + worst-family floor

2. **비교 대상 (Baseline)**
   - 정확히 어떤 checkpoint / recipe가 "5.6 gold"로 간주되는가?
   - 이전에 기록된 최고 수치가 무엇이었는가? (정확한 숫자와 조건)

3. **필수 Ablation 조건**
   - stochastic breadth off
   - gold injection off
   - attractor protection during rehearsal off
   - core/recurrence off (또는 shallow depth)
   - 이 ablation들에서 여전히 이득이 유지되어야 "구조가 기여했다"고 할 수 있음

4. **Curriculum 재현 조건**
   - 5.56 curriculum을 어느 정도까지 동일하게 재현할 것인가?
   - (binding decay schedule, gold injection alpha, rehearsal ratio, attractor protection strength 등)

5. **성공 기준 (Threshold)**
   - "5.6을 넘었다"고 하려면 baseline 대비 몇 % 이상 향상 + ablation에서 특정 수준 이상의 drop이 있어야 하는가?
   - 예: "state_ablation_median에서 +0.05 이상 + stochastic breadth off 시 0.03 이상 drop"

6. **최소 검증 규모**
   - 어떤 d_model, 어떤 step 수, 어떤 family 구성, 몇 개 seed 등에서 검증할 것인가?

---

## 제안: 지금 당장 S0를 시작하자

S0가 가장 중요하다. 이게 제대로 안 되면 나중에 "5.6을 넘었다"는 주장이 약해질 수밖에 없다.

**S0를 위한 다음 행동 제안 (순서대로)**:

1. **기존 5.56 최고 기록을 정확히 정리** (과거 실험에서 나온 state_ablation_median, 사용한 curriculum 상세, stochastic breadth 사용 여부 등)
2. **현재 backbone에서 5.56 curriculum을 재현하기 위한 최소 spec 작성**
3. **"5.6을 넘었다고 주장하기 위한 명확한 Gate" 문서 작성** (위 6개 항목을 구체적인 숫자로 채움)
4. 이 Gate를 모든 관련자에게 (특히 미래의 너 자신에게) 명확히 공유

이 S0 Gate가 정해지면, 그 다음부터는 그 Gate를 통과하는 것을 최우선 목표로 모든 실험을 설계하면 된다.

---

## 지금 네 선택

A. **지금 당장 S0 작업을 시작하자** (위 1~4번을 순서대로 진행)
   - 가장 추천. 5.6을 넘는 실험을 제대로 하려면 이게 제일 먼저 필요함.

B. S0는 대략적인 방향만 잡고, 일단 S1 (5.56 재현)부터 빠르게 prototype을 만들어 보자.

C. 다른 방식으로 진행하고 싶다 (네가 생각하는 우선순위가 따로 있으면 말해).

---

**너의 선택은?**

A로 가고 싶으면, 지금 바로 S0를 위한 첫 번째 작업 (기존 5.56 최고 기록 정리)을 시작할 수 있어.

---

## S0 실행 시작: Surpassing 5.6 Gate 정의 (2026-05-30)

### 현재까지 알려진 5.56 최고 수준 (Inductive Bias Map 기반)

- **대표 메트릭**: `state_ablation_median` (hard-family answer quality의 중앙값)
- 역사적 최고 수준: **~5.53 ~ 5.56 범위** (5.56 gold recipe + stochastic breadth + 642 gold injection 조합에서 달성)
- 핵심 동역학:
  - Scheduled external binding decay (0.40 → 0.04)
  - Attractor protection during rehearsal (0.7)
  - Gold structural injection from 642
  - Stochastic recurrent breadth (K>1 noisy trajectories) throughout curriculum
- Removing stochastic breadth or during-rehearsal protection caused **large drops** in state_ablation_median.

### 제안 S0 Gate (초안 - 사용자 확인 필요)

**Primary Metric**: state_ablation_median on hard-family cases (또는 프로젝트에서 정의한 downstream proxy)

**Baseline**:
- 역사적 5.56 gold recipe로 달성한 최고 state_ablation_median (기록된 최고치)

**필수 Ablations** (모두 통과해야 함):
1. `stochastic_breadth_ablation_zero` 시 state_ablation_median에서 최소 X drop
2. Gold injection off 시 의미 있는 drop
3. Attractor protection during rehearsal off 시 의미 있는 drop
4. Recurrence / core 관련 ablation에서 신호 유지

**Success Threshold (제안)**:
- 새 구조 + 5.56 recipe로 **baseline 대비 +0.04 이상** state_ablation_median 향상
- AND 위 모든 ablation에서 여전히 의미 있는 causal drop 유지

**최소 검증 규모 (제안)**:
- 최소 150~200 step 이상 curriculum
- 여러 seed
- hard family 포함 downstream eval

**다음 행동**:
1. 정확한 역사적 최고 숫자 (state_ablation_median 값)와 사용된 정확한 curriculum config를 찾기 (5.56 evidence package + 과거 trainer logs)
2. 위 Gate를 숫자로 확정
3. S1으로 이동 (재현 작업 시작)

이 Gate가 확정되면, 그 후 모든 실험은 이 Gate를 통과하는 것을 목표로 진행한다.