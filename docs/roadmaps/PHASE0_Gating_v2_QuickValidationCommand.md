# Gating v2 빠른 검증용 실행 명령어 (Phase 0)

**작성일**: 2026-05-30

## 목적
Gating v2 (`torch_gated_delta2_v2`)가 실제 작은 규모 학습에서 정상 동작하는지 빠르게 확인하는 최소 명령어.

## 추천 최소 검증 명령어 (335 스크립트 기준)

```bash
python scripts/335_train_qtrm_native_etd_probe.py \
    --delta_backend torch_gated_delta2_v2 \
    --strict_backends false \
    --steps 200 \
    --batch_size 4 \
    --d_model 256 \
    --n_core_layers 2 \
    --h_cycles 1 \
    --l_cycles 1 \
    --outer_steps 4 \
    --log_every 50 \
    --save_dir ./gating_v2_test_$(date +%Y%m%d_%H%M) \
    --wandb_project qtrm_gating_v2_test \
    --wandb_run_name "gating_v2_smoke_$(date +%H%M)" \
    2>&1 | tee gating_v2_smoke.log
```

### 주요 포인트

- `--delta_backend torch_gated_delta2_v2` : Gating v2 사용
- `--strict_backends false` : 아직 FLA v2가 없으므로 필수
- `--steps 200` 정도면 초기 동작 + 안정성 확인 가능
- `--d_model 256` 정도로 작게 해서 빠르게 검증
- 로그와 checkpoint가 저장되도록 `--save_dir` 지정

## 기대 결과 (정상일 경우)

- 에러 없이 200 step까지 진행
- `stoch_div` 값이 정상적으로 기록됨 (0이 아닌 값)
- decay 스케줄이 정상적으로 적용
- 중간에 NaN이나 training collapse가 발생하지 않음

## 문제 발생 시 체크 포인트

1. `ModuleNotFoundError: No module named 'src'` → `PYTHONPATH=.` prefix 추가
2. Unknown delta backend 에러 → backends/__init__.py 등록 확인
3. Training이 너무 느리면 → d_model을 128로 더 낮추기

## 다음 단계 (이 검증이 성공했을 때)

- 동일 조건으로 기존 `torch_gated_delta` 또는 `fla_gated_delta`로도 한 번 돌려서 비교
- decay curve, stochastic diversity, loss 안정성 비교

