# UltraData L0–L4 Tiered Data Management (OpenBMB) — RI-1 적용 가이드

**출처**: OpenBMB Position Paper “Data Science and Technology Towards AGI Part I: Tiered Data Management” (arXiv:2602.09003, 2026-02)

**실제 적용 사례**: MiniCPM5-1B (2026-05 release) — 이 프레임워크를 full-stack으로 사용해 1B급 모델에서 reasoning/math/agentic 성능이 크게 상승.

---

## 핵심 철학 (우리에게 중요한 메시지)

> “Scaling Up 시대는 끝났다. 이제는 **적절한 훈련 시점에 적절한 순도와 형태의 데이터를 주는 것**이 핵심이다.”

- Lower tier (L1/L2): Early 단계에서 대량 스케일 확보
- **Higher tier (L3)**: Late / decay / reasoning 강화 단계에서 집중 투입
- L3이 가장 강력한 marginal gain을 주는 구간 (MiniCPM5-1B의 주요 성공 요인)

---

## L0 ~ L4 요약 및 우리 프로젝트 매핑

| Tier | 이름          | 주요 작업                          | 우리 현재 상황                          | RI-1 관점 적용 가치 |
|------|---------------|------------------------------------|-----------------------------------------|---------------------|
| L0   | Raw           | 크롤링 + 기본 파싱                 | 거의 없음 (필요시 Common Crawl 등)     | 낮음 |
| L1   | Filtered      | 휴리스틱 필터 + dedup              | 일부 heuristic + gold proxy             | 기반 데이터 풀 |
| L2   | Selected      | 모델 기반 품질 스코어링            | 거의 없음                               | Mid-training용 |
| **L3** | **Refined** | **LLM 합성 + 다중 스타일 재작성** (Q&A, textbook, dialogue, CoT, competition-style 등) | **gold injection + AdaptiveRehearsal + depth consistency** (초보적 L3) | **최고** |
| L4   | Organized     | 사실 검증 + 구조화 (KG 등)         | SparseSlotRouter + Decoupled Memory + Provenance | RAG / Memory 고도화 |

### L3이 우리에게 특별히 중요한 이유

OpenBMB는 L3 데이터를 다음과 같이 정의하고 MiniCPM5-1B decay phase + SFT에서 대량 투입했다:

- 기존 고품질 문서(L2)를 LLM으로 재작성
- 여러 **교육적/추론 스타일**로 변환 (textbook 설명, teacher-student 대화, rigorous competition logic, intuitive popular science 등)
- 명시적인 reasoning signal 강화 (CoT, multi-step, verification 포함)

이것이 바로 우리가 **RI-1**에서 가장 부족한 부분이다:

> “deeper is better” inductive bias를 학습시키려면, 단순히 긴 rollout을 주는 것만으로는 부족하다.
> **“더 길게 생각하면 실제로 더 나은 latent state / answer alignment에 도달한다”**는 명시적이고 dense한 신호가 필요하다.

현재 우리가 쓰는 `trajectory_monotonic_weight`, `depth_consistency_weight`, gold injection은 **L3 정신의 초보적 구현**이다. OpenBMB 수준의 체계적인 L3 파이프라인을 도입하면 RI-1 monotonic scaling이 훨씬 안정적으로 나올 가능성이 높다.

---

## 구체적인 적용 제안 (RI-1 Phase)

### 1. L3 Synthetic Reasoning Data 생성 파이프라인 (최우선)

**목표**: 우리의 pure_reasoning bucket + attractor training에 최적화된 L3 데이터를 만드는 것.

가능한 생성 방식:
- L2 수준의 고품질 reasoning trace를 seed로 사용
- 여러 스타일로 재작성:
  - Short vs Long trajectory 비교 쌍 (depth consistency loss에 직접 사용)
  - “더 깊이 생각한 버전” vs “얕게 생각한 버전” (monotonic improvement 명시)
  - Teacher-student / Verifier 스타일 (K-candidate selection 강화)
  - Multi-step attractor basin 설명형

이 데이터는 현재 `train_hybrid_ri4_real_continuation_minimal.py`의 `_compute_heldout_answer_pressure_loss`와 `depth consistency` term에 직접 투입할 수 있다.

### 2. Training Stage별 Tier 매칭 (추천 스케줄)

- Early continuation: L1/L2 위주 (scale)
- Mid ~ Late (현재 우리가 하는 M1 단계): **L3 대량 투입** (variable depth + consistency가 가장 효과를 보는 구간)
- Final attractor shaping: L3 + 소량 L4 (verifiable reasoning)

### 3. Data-Model Co-Evolution 루프 구축 (장기)

1. 현재 M1 curriculum v2로 모델을 어느 정도 강하게 만든다.
2. 그 모델을 사용해 L3 데이터를 더 고품질로 재생성한다.
3. 재생성된 L3으로 다시 M1-style training을 한다.

이 루프가 OpenBMB가 강조하는 핵심 동력이다.

---

## 현재 우리 상태와 Gap

**강점**:
- Variable depth training + curriculum ramp (Huginn-style) 이미 도입 (2026-05-29)
- Depth-wise consistency loss, monotonic pressure, gold injection 존재
- OneBodyParallelHybridBlock + 3-track attractor substrate

**Gap**:
- L3 데이터가 아직 ad-hoc gold + synthetic proxy 수준
- 체계적인 “multi-style synthetic reasoning trace” 생성기가 없음
- Training stage별 tiered data 스케줄링이 명시적이지 않음

UltraData 프레임워크를 도입하면 이 Gap을 가장 효율적으로 메울 수 있다.

---

## 다음 실행 가능한 액션 (2026-05-29 기준)

1. **즉시** — 디스크 정리 완료 후, step430.pt 기준으로 M1 Curriculum v2를 safer 설정으로 재개 (save_every=20)
2. **단기 (1~2일)** — L3 Synthetic Reasoning Data 생성 스크립트 프로토타입 작성 (pure_reasoning bucket 대상)
3. **중기** — L3 데이터를 현재 trainer의 pressure / consistency loss에 연결하는 실험
4. **wiki 보강** — 이 문서 + `l3_synthetic_reasoning_for_attractor.md` 추가 작성

---

**참고 자료**
- Paper: arXiv:2602.09003
- Platform: https://ultradata.openbmb.cn/
- 주요 데이터셋: `openbmb/Ultra-FineWeb-L3`, `openbmb/UltraData-Math`, `openbmb/UltraData-SFT-2605`
- MiniCPM5-1B repo: https://github.com/openbmb/minicpm

---

*이 문서는 2026-05-29에 RI-1 Architectural Improvement Phase 진행 중 작성됨. UltraData 정신을 우리 recurrent hybrid substrate에 맞춰 실용적으로 적용하기 위한 가이드로 유지할 예정.*