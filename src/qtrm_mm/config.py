from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class QTRMConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768
    max_seq_len: int = 512
    n_prelude_layers: int = 2
    n_core_layers: int = 2
    n_coda_layers: int = 2
    attn_every: int = 4
    coda_attn_every: Optional[int] = None
    workspace_tokens: int = 32
    workspace_layers: int = 1
    workspace_ff_mult: int = 0
    workspace_include_latents_in_kv: bool = False
    h_cycles: int = 2
    l_cycles: int = 2
    outer_steps: int = 2
    dropout: float = 0.0
    rope_theta: float = 10000.0
    delta_backend: str = "torch_gated_delta"
    attention_backend: str = "sdpa"
    strict_backends: bool = False
    visual_dim: int = 512
    max_visual_tokens: int = 64
    num_actions: int = 10
    tie_embeddings: bool = True
    use_stable_inject: bool = True
    truncated_recurrence: bool = False
    core_halt_enabled: bool = False
    core_halt_min_steps: int = 1
    core_halt_use_continue: bool = False
    jepa_encoder_layers: int = 1
    jepa_predictor_layers: int = 2
    jepa_predictor_dim: Optional[int] = None
    jepa_horizon: int = 1
    jepa_sigreg_weight: float = 0.09
    jepa_sigreg_knots: int = 17
    jepa_sigreg_num_proj: int = 1024
    qtrm_logits_scale: float = 1.0
    donor_logits_scale: float = 0.0


@dataclass
class DonorConfig:
    model_id: Optional[str] = None
    load_in_4bit: bool = False
    freeze_donor: bool = True
    train_lora: bool = False
    trust_remote_code: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 128
    steps: int = 50
    lr: float = 3e-4
    loss_jepa_weight: float = 0.1
    loss_aux_weight: float = 1.0
    loss_core_halt_weight: float = 0.0
    loss_donor_kl_weight: float = 0.0
    donor_kl_beta: float = 0.0
    donor_kl_temperature: float = 1.0
    core_halt_auto_targets: bool = False
    core_halt_target_mode: str = "exact"
    core_halt_donor_kl_threshold: Optional[float] = None
    core_halt_teacher_depth_threshold: float = 0.995
    core_halt_teacher_depth_logit_kl_threshold: float = 0.05
    core_halt_teacher_depth_min_step: int = 1
    trainable_param_policy: str = "all"
    donor_logits_scale_start: Optional[float] = None
    donor_logits_scale_end: Optional[float] = None
    device: str = "auto"
    use_amp: bool = True
    log_every: int = 10
    out_dir: str = "runs/smoke_multimodal"


@dataclass
class FullConfig:
    model: QTRMConfig = field(default_factory=QTRMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    donor: DonorConfig = field(default_factory=DonorConfig)


def _filter_dataclass(cls, data: Dict[str, Any]):
    names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    return FullConfig(
        model=_filter_dataclass(QTRMConfig, raw.get("model", {})),
        train=_filter_dataclass(TrainConfig, raw.get("train", {})),
        donor=_filter_dataclass(DonorConfig, raw.get("donor", {})),
    )
