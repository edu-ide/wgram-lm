# Gating v2 적용 예시 (Phase 0)

**작성일**: 2026-05-30

## 가장 간단한 사용법

### 방법 1: 학습 스크립트 인자에서 직접 지정 (가장 흔함)

```bash
python scripts/335_train_qtrm_native_etd_probe.py \
    ... \
    --delta_backend torch_gated_delta2_v2
```

또는 alias 사용:
```bash
--delta_backend gdn2_v2
```

### 방법 2: Python 코드에서 직접 지정

```python
from src.qtrm_mm.qwen_backbone_qtrm import build_small_qwen_qtrm_core

core_cfg = build_small_qwen_qtrm_core(
    qwen_config=your_qwen_config,
    max_seq_len=4096,
    n_core_layers=4,
    h_cycles=1,
    l_cycles=2,
    outer_steps=6,
    delta_backend="torch_gated_delta2_v2",   # ← 여기만 변경
    strict_backends=False,                   # 개발 중에는 False 추천
)
```

### 방법 3: QTRMConfig 직접 생성 시

```python
from src.qtrm_mm.config import QTRMConfig

cfg = QTRMConfig(
    d_model=1024,
    ...
    delta_backend="torch_gated_delta2_v2",   # 또는 "gated_delta2_v2"
    strict_backends=False,
)
```

## 추천 조합 (현재 시점)

| 상황                    | 추천 delta_backend          | 비고 |
|-------------------------|-----------------------------|------|
| 개발 / 빠른 실험        | "torch_gated_delta2_v2"     | 가장 안전 |
| FLA 최적화 원할 때      | "fla_gated_delta" (기존)    | 아직 v2 FLA 버전은 없음 |
| 가장 최신 Gating 원할 때| "gdn2_v2" 또는 "torch_gated_delta2_v2" | 추천 |

## 주의사항

- `strict_backends=True` 로 설정하면 아직 FLA 쪽 v2가 없어서 fallback 될 수 있음 → 개발 중에는 `False` 추천.
- v2는 현재 Torch reference 구현이므로, 대규모 학습에서는 속도가 느릴 수 있음 (나중에 FLA 포팅 필요).

