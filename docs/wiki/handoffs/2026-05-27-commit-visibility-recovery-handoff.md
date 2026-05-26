# 2026-05-27 Commit Visibility Recovery Handoff

## 문제를 문과적으로 설명하면
사용자가 느낀 “커밋이 날아감”은 실제로는 다음 중 하나였습니다.
- `main` 브랜치만 보고 있었다.
- `--all`/브랜치 집합을 안 봤다.
- 최신 실험은 실험 브랜치에 있는데, 보존용 브랜치 표시 기준을 안 맞췄다.

즉, 날아간 게 아니라 **보는 창이 다르다**가 맞는 진단입니다.

## 바로 증명되는 상태
- `main`(5/20~5/27): `7dd5e0c`만 존재
- 실험 브랜치 `ablation-opt2-isolate-memory`(5/20~5/27):
  - `d123cdc`
  - `dc431b1`
  - `cbbb960`
  - `e9ae818`
  - `0144f1e`
  - ...
- 보존 태그:
  - `ablation-step-binding-probe-30step`
  - `2026-05-20to27-preserve`

## 앞으로 복구/확인 표준 (반드시 이렇게 수행)
```bash
# 1) 현재 위치 확인
cd /home/tripleyoung/qtrm-workspace/qtrm_multimodal_memoryos

git branch --show-current

git log --oneline --max-count=5

# 2) 실험 브랜치로 전환

git checkout ablation-opt2-isolate-memory

git log --oneline --max-count=5

# 3) 5/20~5/27 전체 스냅샷 확인

git log --oneline --decorate --since='2026-05-20' --until='2026-05-27' --all

# 4) 보존 체크포인트 실행

bash scripts/999_workspace_commit_guard.sh

# 5) main에서 바로 분기 판단/이동 가이드(원하면 한 줄 정리)

bash scripts/998_checkout_stage119_if_missing.sh
```

## 운영 규칙
- 작업 기록은 브랜치 이동 전제에서 “사라짐” 판단하지 않는다.
- `main`이 과거 포인트면 실험은 무조건 안 보일 수 있다.
- 실험 상태 점검 시 기본적으로 다음 3개를 같이 본다.
  - `--all` 로그
  - 핵심 브랜치 목록
  - `scripts/999_workspace_commit_guard.sh`

## 다음 액션
- 사용자가 “커밋 없다”라고 느끼면 즉시 위 표준을 실행해서
  1) 브랜치 위치
  2) 전체 날짜 로그
  3) 태그 존재
  4) 핵심 산출물 존재
  를 한 번에 확인한다.
