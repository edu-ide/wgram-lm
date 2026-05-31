
---

## 7. Torch Reference Gating v2 상세 구현 제안 (순서대로 다음 단계)

### 7.1 현재 TorchGatedDeltaMixer 분석 (기준)

```python
# 현재 (간소화)
self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)   # u, v, decay
self.gate_proj = nn.Linear(d_model, d_model, bias=True)
...
state = decay[:, i] * state + (1.0 - decay[:, i]) * u[:, i] * m
y = gate[:, i] * v[:, i] + (1.0 - gate[:, i]) * state
```

문제점:
- decay와 gate가 대부분 scalar 또는 head-wise 수준
- delta rule이 단순함
- in-context learning rate 개념 없음

### 7.2 Gating v2 제안 구조 (Torch Reference)

```python
class TorchGatedDeltaNet2MixerV2(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 확장된 projection (u, v, decay, in_context_lr)
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)

        # Vector-valued gate (per head or per channel)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # (선택) 추가적인 forget / update 분리 게이트
        self.forget_proj = nn.Linear(d_model, d_model, bias=True)   # optional

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # LayerNorm for stability (ReGLA 스타일)
        self.norm = nn.LayerNorm(d_model)
```

**Forward 핵심 로직 (v2, sequential reference)**

```python
def forward(self, x, attention_mask=None):
    b, t, d = x.shape

    # u, v, decay, in_context_lr
    proj = self.in_proj(x)
    u, v, decay, in_context_lr = proj.chunk(4, dim=-1)

    u = torch.tanh(u)
    v = torch.tanh(v)
    decay = torch.sigmoid(decay)
    in_context_lr = torch.sigmoid(in_context_lr)

    # Vector gate (per dimension)
    gate = torch.sigmoid(self.gate_proj(x))

    # Optional forget gate (RWKV-7 스타일 강화)
    forget = torch.sigmoid(self.forget_proj(x)) if hasattr(self, 'forget_proj') else None

    state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
    outs = []

    for i in range(t):
        m = attention_mask[:, i:i+1] if attention_mask is not None else 1.0

        # Refined delta update with in-context learning rate
        update = in_context_lr[:, i] * u[:, i] * m
        if forget is not None:
            state = forget[:, i] * state + (1 - forget[:, i]) * update
        else:
            state = decay[:, i] * state + (1 - decay[:, i]) * update

        y = gate[:, i] * v[:, i] + (1 - gate[:, i]) * state
        outs.append(y)

    y = torch.stack(outs, dim=1)
    y = self.norm(y)                    # ReGLA 스타일 추가 정규화
    return self.out_proj(self.dropout(y))
```

### 7.3 주요 변경 포인트 (실제 코드 위치)

1. `TorchGatedDeltaMixer.__init__`:
   - `in_proj` 차원: `3 * d_model` → `4 * d_model` (in_context_lr 추가)
   - `gate_proj`는 유지하되, vector gate를 명확히 의도
   - (선택) `forget_proj` 추가

2. `forward` 내부:
   - decay + in_context_lr를 함께 사용하여 state update
   - vector gate 적용
   - (선택) forget 게이트 추가
   - forward 끝부분에 `self.norm(y)` 추가 (안정성)

3. **하위 호환성**:
   - 기존 `TorchGatedDeltaMixer`는 그대로 두고, `TorchGatedDeltaNet2MixerV2`라는 새 클래스로 만들 것을 강력 추천.
   - config에서 `core_recurrence_mixer: "gated_delta2_v2"` 같은 플래그로 전환 가능하게 설계.

### 7.4 One-Body 호환성 확인

- z_l / z_h dual recurrence 구조 그대로 사용 가능
- stochastic breadth 주입 지점 영향 없음
- 5.56 curriculum (gold injection, attractor protection) 위치 그대로 유지 가능
- ablation (특히 stochastic breadth ablation) 완전 호환

### 7.5 다음 단계 제안 (순서대로)

이 제안이 마음에 들면, 다음 미세 단계는:

**A-1.** 위 pseudocode를 바탕으로 실제 `TorchGatedDeltaNet2MixerV2` 클래스를 `mixers.py`에 작성 (reference 버전)

**A-2.** 간단한 unit test (synthetic sequence)로 기존 vs v2 비교

**A-3.** config에 새로운 mixer 타입 추가

원하시면 A-1부터 바로 진행할 수 있습니다.

---
**Phase 0 현재 상태 (업데이트)**

- 장기 로드맵 수립: 완료
- Backbone Audit + Paper Synthesis: 진행 중
- Gating v2 상세 기술 제안: 완료 (이 문서)
- 다음: 사용자 승인 → Torch reference v2 구현 (A-1)

순서대로 계속하겠습니다.

---

## 8. 순서대로 진행 현황 업데이트 (2026-05-30)

**완료된 미세 단계 (Phase 0 내 Gating v2 라인)**:
1. Gating v2 상세 기술 제안 (ReGLA + RWKV-7 + Gated DeltaNet 기반) — 완료
2. Torch reference 구현 (`TorchGatedDeltaNet2MixerV2`) — 완료
3. backends/__init__.py에 신규 타입 등록 (`torch_gated_delta2_v2`, `gated_delta2_v2`, `gdn2_v2`) — 완료
4. qwen_backbone_wgram.py에서 신규 이름 수용 및 전달 지원 — 완료
5. config.py 및 backends 문서에 옵션 노출 — 완료
6. 기본 smoke test (import + instantiation + forward) — **완료** (모두 정상 동작 확인)

**현재 Phase 0 Gating v2 상태**:
모든 기반 인프라(코드 + 등록 + config 노출 + 기본 동작 검증)가 순서대로 완료되었습니다.

**다음 미세 단계 (사용자 확인 필요)**:
- 실제 작은 규모 모델 config에 `delta_backend="torch_gated_delta2_v2"` (또는 alias) 를 적용해서 학습 테스트를 해볼지 여부.
- 또는 FLA 백엔드 쪽으로도 v2 포팅을 먼저 검토할지.
- 또는 Gating v2는 여기까지 하고, Phase 0의 다음 우선순위 항목(Parallel Hybrid Head 초기 스케치)으로 넘어가도 되는지.

사용자가 "순서대로해"라고 했으므로, 위 선택을 기다린 후에만 다음으로 넘어갑니다.
