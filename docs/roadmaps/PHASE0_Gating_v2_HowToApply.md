# Gating v2 실제 적용 가이드 (Phase 0)

**작성일**: 2026-05-30

## 가장 추천하는 적용 방법 (현재 시점)

### 방법 1: 학습 스크립트 실행 시 인자로 넘기기 (가장 간단하고 안전)

대부분의 학습 스크립트(예: `scripts/335_train_qtrm_native_etd_probe.py`)에서 다음과 같이 실행하면 됩니다:

```bash
# 기존
python scripts/335_train_qtrm_native_etd_probe.py \
    --delta_backend torch_gated_delta \
    ...기타 인자...

# Gating v2 적용
python scripts/335_train_qtrm_native_etd_probe.py \
    --delta_backend torch_gated_delta2_v2 \
    ...기타 인자...
```

또는 짧은 alias 사용:
```bash
--delta_backend gdn2_v2
```

### 방법 2: 스크립트 내부에서 하드코딩으로 바꾸기 (개발/실험용)

`scripts/335_train_qtrm_native_etd_probe.py` 파일에서 다음 부분을 찾아 변경:

**변경 전 (대략 890줄 근처):**
```python
delta_backend: str = "torch_gated_delta",
```

**변경 후:**
```python
delta_backend: str = "torch_gated_delta2_v2",   # Gating v2 (2026-05-30)
```

그리고 backbone을 생성하는 부분(대략 594줄, 666줄 근처)에서 `delta_backend` 인자를 그대로 넘기고 있으면 자동으로 적용됩니다.

### 방법 3: Python 코드로 직접 생성할 때

```python
from src.qtrm_mm.qwen_backbone_qtrm import build_small_qwen_qtrm_core

core_cfg = build_small_qwen_qtrm_core(
    qwen_config=qwen_config,
    max_seq_len=4096,
    n_core_layers=4,
    h_cycles=1,
    l_cycles=2,
    outer_steps=6,
    delta_backend="torch_gated_delta2_v2",  # ← Gating v2
    strict_backends=False,
)
```

## 주의사항 (반드시 지킬 것)

1. **개발/실험 단계에서는 `strict_backends=False`** 로 두는 것을 강력 추천.
   - 아직 FLA 쪽에 v2 구현이 없기 때문에 strict 모드에서는 fallback될 수 있음.

2. v2는 현재 **Torch reference 구현**이므로, 대규모 학습에서는 속도가 느릴 수 있음.
   - 검증 목적으로만 사용하는 것을 권장.
   - 나중에 FLA 백엔드 포팅이 완료되면 속도가 크게 개선될 예정.

3. config에 제대로 반영되었는지 확인:
   ```python
   print(core_cfg.delta_backend)   # "torch_gated_delta2_v2" 가 나와야 함
   ```

## 빠른 검증 명령어

```bash
# 아주 작은 규모로 Gating v2가 정상 동작하는지 빠르게 확인
python scripts/335_train_qtrm_native_etd_probe.py \
    --delta_backend torch_gated_delta2_v2 \
    --steps 100 \
    --d_model 256 \
    --batch_size 2 \
    --strict_backends false \
    ...기타 필요한 최소 인자...
```

