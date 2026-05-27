# Branch B - Option 2: Path B 상세 설명 (Port to Stronger Backbone)

**작성일**: 2026-05-30

## Path B가 정확히 무엇인가?

Path B는 현재 우리가 가지고 있는 **작은 QTRMRecursiveCore** 위에서 5.56 curriculum을 계속 돌리는 것이 아니라,

**"5.56의 핵심 아이디어들(훈련 동역학 + inductive bias)"을 훨씬 더 강력한 기존 backbone에 이식**해서, 그 강력한 backbone이 가진 표현력과 데이터 처리 능력 위에서 5.56의 효과를 제대로 발휘하게 만드는 전략입니다.

즉, "모델을 키우는 것"이 아니라 **"이미 강력한 모델에 5.56의 특수한 훈련 방식을 붙이는 것"**에 가깝습니다.

## 5.56 Curriculum Logic — 정확히 무엇을 포팅하는가?

5.56에서 실제로 가치 있었던 것은 특정 아키텍처가 아니라, 다음과 같은 **훈련 동역학**입니다:

1. **Scheduled Binding Decay** (0.40 → 0.04)
   - 외부에서 강제로 binding pressure를 점진적으로 낮추는 스케줄.

2. **Gold Structural Injection from 642**
   - 642 같은 강한 gold state를 단순히 loss로 넣는 게 아니라, memory / slow-tier / attractor 등 구조적으로 주입.

3. **Attractor Protection During Rehearsal**
   - Rehearsal을 하면서도 이미 형성된 좋은 attractor basin을 보호하는 메커니즘.

4. **Stochastic Recurrent Breadth (Reverse I→G→A 핵심)**
   - 훈련 중에 recurrent state(z_h)에 의도적으로 K>1 trajectory diversity를 주는 것.
   - 이게 현재 우리가 가장 강하게 증명한 부분.

Path B에서는 위 4가지를 **더 큰 모델**에 이식하는 것을 목표로 합니다.

## Path B에서 고려할 수 있는 "Stronger Backbone" 예시

### 1. Larger QTRM Family (가장 자연스러운 연장선)
- 현재 우리가 쓰는 QTRMRecursiveCore를 d_model 512~2048, layer 수 증가, outer recurrence depth 증가 등으로 대형화한 버전.
- 장점: 아키텍처 철학이 같아서 포팅 난이도가 상대적으로 낮음.
- 단점: 여전히 "우리만의" 아키텍처라서, 외부 검증이나 커뮤니티 자산을 활용하기 어려움.

### 2. Standard Large Transformer + Recurrence/Memory Augmentation
- Llama, Qwen, Mistral 등 이미 검증된 대형 transformer backbone.
- 여기에 recurrence (예: latent state carry-over, slow tier, gated memory 등)를 추가.
- 5.56의 curriculum logic을 이 위에 올림.
- 장점: backbone 자체의 표현력과 사전학습 지식이 매우 강력. 데이터도 대규모로 활용 가능.
- 단점: recurrence/memory를 어떻게 자연스럽게 붙일지가 핵심 엔지니어링 문제.

### 3. 기존에 State / Memory를 잘 다루는 아키텍처
- RWKV, Mamba, RetNet, Griffin, Griffin-like hybrid 등.
- 이미 long-range dependency를 잘 다루는 모델 위에 5.56의 "curriculum + stochastic breadth"를 올리는 방식.
- 장점: backbone이 이미 state를 잘 유지하는 특성을 가지고 있어서, 5.56의 효과가 더 잘 드러날 가능성.
- 단점: 이런 아키텍처에 우리 curriculum logic을 이식하는 경험이 거의 없음.

## Path B의 핵심 장점 (왜 Branch B에서 Option 2로 가면 Path B를 진지하게 고려해야 하는가)

- **표현력의 차이**: 작은 모델에서는 "state를 잘 유지하는 것" 자체가 이미 어려운 과제였다. 큰 backbone에서는 이 부분이 어느 정도 해결되어 있기 때문에, 5.56 curriculum이 진짜로 "추가적인 robustness"를 주는지를 더 깨끗하게 볼 수 있다.
- **데이터의 질**: 대형 모델은 대규모 고품질 long-horizon reasoning 데이터로 학습할 수 있다. 현재 우리가 쓰는 synthetic random workspace와는 차원이 다르다.
- **평가의 현실성**: hard-family, long-horizon reasoning, state ablation after curriculum 같은 평가를 실제로 의미 있게 할 수 있는 규모가 나온다.
- **원래 5.5x가 나온 방식에 더 가깝다**: 역사적으로 5.53~5.56이 나온 환경도 "작은 toy model + 완전 synthetic"이 아니었다.

## Path B의 현실적인 어려움

- **Engineering Cost가 매우 높음**
  - 기존에 잘 돌아가는 대형 backbone에 recurrence / memory / slow-tier / stochastic breadth 같은 메커니즘을 자연스럽게 이식하는 것은 상당한 아키텍처 작업이다.
  - 단순히 "크게 만들자"가 아니라, "5.56의 inductive bias를 어떻게 backbone의 inductive bias와 잘 결합시킬 것인가"가 핵심.

- **데이터 파이프라인 구축 비용**
  - 진짜 long-horizon reasoning 데이터셋을 만들거나, 기존 대형 모델 학습에 쓰이는 고품질 데이터를 curriculum 방식으로 재가공해야 한다.

- **평가 인프라**
  - state ablation robustness를 제대로 측정할 수 있는 evaluation suite를 처음부터 만들어야 한다. 현재 우리가 쓰는 작은 synthetic probe는 이 규모에서는 의미가 없다.

- **Compute Cost**
  - 작은 모델 수십 번 돌리는 것과, 수십억~수백억 파라미터 모델에 curriculum을 적용하는 것은 비용 차이가 극심하다.

## Path B를 선택했을 때 현실적인 실행 방향 (추천 순서)

1. **먼저 "무엇을 포팅할 것인가"를 명확히 정의**
   - 5.56의 4가지 핵심 요소를, 특정 backbone에 어떻게 녹여넣을지 구체적인 설계 문서 작성.

2. **Backbone 후보를 1~2개로 압축**
   - 현재 프로젝트의 철학과 가장 잘 맞는 후보를 선택 (예: 더 큰 QTRM variant vs 특정 오픈소스 대형 모델).

3. **Minimal Viable Port**
   - 전체 모델을 처음부터 학습하지 말고, 작은 규모의 prototype으로 먼저 "5.56 curriculum logic이 이 backbone 위에서 잘 동작하는지"만 빠르게 검증.

4. **평가 메트릭을 동시에 설계**
   - Path B를 하는 이유가 "진짜 robustness를 보기 위해서"라면, 그 robustness를 어떻게 측정할지가 가장 먼저 정의되어야 한다.

## Path A vs Path B 간단 비교 (Option 2 관점)

| 항목                    | Path A (현재 아키텍처 대형화)       | Path B (강력한 backbone에 포팅)          |
|-------------------------|------------------------------------|-----------------------------------------|
| Engineering 난이도      | 중간                               | 높음                                    |
| Real signal 나올 확률   | 중간                               | 높음                                    |
| 비용                    | 높음                               | 매우 높음                               |
| 기존 5.5x와의 유사성    | 낮음~중간                          | 높음                                    |
| 실패했을 때의 학습 가치 | 중간                               | 높음 (다양한 backbone 경험 축적)         |

## 요약

Path B는 "우리 아키텍처를 더 키우자"가 아니라,  
**"5.56이 실제로 강력한 inductive bias를 줄 수 있는지, 더 강력한 모델 위에서 검증해보자"**는 전략입니다.

이 선택을 하려면, 단순히 "모델을 키우자"가 아니라  
**"5.56의 핵심을 어떤 강력한 backbone에 어떻게 녹여넣을 것인가"**에 대한 진지한 아키텍처 + 데이터 + 평가 설계가 필요합니다.

---

이 문서는 `OPTION2_DIRECT_EXECUTION_PLAN.md`의 Path B 부분을 훨씬 더 자세히 풀어쓴 버전입니다.
