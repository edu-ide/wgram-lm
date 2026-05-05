from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransitionControllerMetrics:
    samples: int
    accuracy: float
    loss: float

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "samples": int(self.samples),
            "accuracy": float(self.accuracy),
            "loss": float(self.loss),
        }


@dataclass(frozen=True)
class TransitionStatePredictionMetrics:
    samples: int
    loss: float
    mae: float
    binary_accuracy: float

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "samples": int(self.samples),
            "loss": float(self.loss),
            "mae": float(self.mae),
            "binary_accuracy": float(self.binary_accuracy),
        }


class TransitionStatePredictor(nn.Module):
    """Predict explicit loop-state features from per-step QTRM features."""

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim or d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, steps, d_model]")
        if features.shape[-1] != self.d_model:
            raise ValueError(f"features last dim must be {self.d_model}")
        logits = self.net(features)
        return {
            "transition_state_logits": logits,
            "transition_state_features": torch.sigmoid(logits),
        }


class TransitionStateController(nn.Module):
    """Small recurrent policy over per-step QTRM states."""

    def __init__(
        self,
        d_model: int,
        num_actions: int,
        *,
        hidden_dim: int | None = None,
        signal_dim: int = 2,
        transition_state_dim: int = 0,
        dropout: float = 0.0,
        use_prev_action: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim or d_model)
        self.signal_dim = int(signal_dim)
        self.transition_state_dim = int(transition_state_dim)
        self.use_prev_action = bool(use_prev_action)
        self.start_action_id = self.num_actions
        prev_action_dim = self.d_model if self.use_prev_action else 0
        state_dim = max(0, self.transition_state_dim)
        self.feature_norm = nn.LayerNorm(self.d_model)
        self.transition_state_norm = (
            nn.LayerNorm(state_dim) if state_dim > 0 else None
        )
        self.prev_action_embed = (
            nn.Embedding(self.num_actions + 1, self.d_model)
            if self.use_prev_action
            else None
        )
        self.input_proj = nn.Linear(
            self.d_model + prev_action_dim + state_dim,
            self.hidden_dim,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.hidden_dim, self.num_actions)
        self.signal_head = nn.Linear(self.hidden_dim, self.signal_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.signal_head.weight)
        nn.init.zeros_(self.signal_head.bias)

    def initial_prev_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (int(batch_size),),
            int(self.start_action_id),
            dtype=torch.long,
            device=device,
        )

    def teacher_forced_prev_actions(self, action_targets: torch.Tensor) -> torch.Tensor:
        if action_targets.ndim != 2:
            raise ValueError("action_targets must have shape [batch, steps]")
        b, t = action_targets.shape
        prev = torch.full(
            (b, t),
            int(self.start_action_id),
            dtype=torch.long,
            device=action_targets.device,
        )
        if t > 1:
            prev[:, 1:] = action_targets[:, :-1].clamp(min=0, max=self.num_actions - 1)
        return prev

    def _compose_input(
        self,
        features: torch.Tensor,
        prev_actions: torch.Tensor | None,
        transition_state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, steps, d_model]")
        if features.shape[-1] != self.d_model:
            raise ValueError(f"features last dim must be {self.d_model}")
        features = self.feature_norm(features)
        if self.prev_action_embed is None:
            x = features
        else:
            if prev_actions is None:
                raise ValueError("prev_actions are required when use_prev_action=True")
            prev = prev_actions.clamp(min=0, max=self.start_action_id)
            x = torch.cat([features, self.prev_action_embed(prev)], dim=-1)
        if self.transition_state_norm is not None:
            if transition_state_features is None:
                raise ValueError(
                    "transition_state_features are required when transition_state_dim > 0"
                )
            if transition_state_features.shape[:2] != features.shape[:2]:
                raise ValueError("transition_state_features must match batch and steps")
            if transition_state_features.shape[-1] != self.transition_state_dim:
                raise ValueError(
                    "transition_state_features last dim must be "
                    f"{self.transition_state_dim}"
                )
            state = self.transition_state_norm(transition_state_features)
            x = torch.cat([x, state], dim=-1)
        return self.dropout(F.gelu(self.input_proj(x)))

    def forward(
        self,
        features: torch.Tensor,
        *,
        prev_actions: torch.Tensor | None = None,
        transition_state_features: torch.Tensor | None = None,
        reset_each_step: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self._compose_input(features, prev_actions, transition_state_features)
        if not reset_each_step:
            states, _ = self.rnn(x)
        else:
            states = x
        return {
            "hidden_states": states,
            "action_logits": self.action_head(states),
            "controller_signal_logits": self.signal_head(states),
        }

    @torch.no_grad()
    def predict_autoregressive(
        self,
        features: torch.Tensor,
        *,
        transition_state_features: torch.Tensor | None = None,
        reset_each_step: bool = False,
        force_start_prev_action: bool = False,
        zero_transition_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, steps, d_model]")
        b, t, _ = features.shape
        hidden = None
        prev_action = self.initial_prev_actions(b, features.device)
        logits_by_step: list[torch.Tensor] = []
        signal_logits_by_step: list[torch.Tensor] = []
        for step in range(t):
            step_features = features[:, step : step + 1, :]
            step_prev = None if self.prev_action_embed is None else prev_action.view(b, 1)
            step_state = None
            if transition_state_features is not None:
                step_state = transition_state_features[:, step : step + 1, :]
                if zero_transition_state:
                    step_state = torch.zeros_like(step_state)
            x = self._compose_input(step_features, step_prev, step_state)
            if reset_each_step:
                state = x
            else:
                state, hidden = self.rnn(x, hidden)
            logits = self.action_head(state[:, 0])
            signal_logits = self.signal_head(state[:, 0])
            logits_by_step.append(logits)
            signal_logits_by_step.append(signal_logits)
            if not force_start_prev_action:
                prev_action = logits.argmax(dim=-1)
            if reset_each_step:
                hidden = None
        return {
            "action_logits": torch.stack(logits_by_step, dim=1),
            "controller_signal_logits": torch.stack(signal_logits_by_step, dim=1),
        }


def transition_action_loss(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    sequence_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TransitionControllerMetrics]:
    if action_logits.ndim != 3:
        raise ValueError("action_logits must have shape [batch, steps, num_actions]")
    if action_targets.shape != action_logits.shape[:2]:
        raise ValueError("action_targets must have shape [batch, steps]")
    flat_logits = action_logits.reshape(-1, action_logits.shape[-1])
    flat_targets = action_targets.reshape(-1)
    if sequence_mask is None:
        flat_mask = flat_targets >= 0
    else:
        if sequence_mask.shape != action_targets.shape:
            raise ValueError("sequence_mask must match action_targets shape")
        flat_mask = sequence_mask.reshape(-1).to(torch.bool) & (flat_targets >= 0)
    if not bool(flat_mask.any().detach().cpu().item()):
        zero = flat_logits.sum() * 0.0
        return zero, TransitionControllerMetrics(samples=0, accuracy=0.0, loss=0.0)
    loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
    preds = flat_logits.argmax(dim=-1)
    correct = (preds[flat_mask] == flat_targets[flat_mask]).float().sum()
    total = int(flat_mask.sum().detach().cpu().item())
    accuracy = float((correct / max(1, total)).detach().cpu().item())
    return loss, TransitionControllerMetrics(
        samples=total,
        accuracy=accuracy,
        loss=float(loss.detach().cpu().item()),
    )


def transition_state_prediction_loss(
    state_logits: torch.Tensor,
    state_targets: torch.Tensor,
    sequence_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TransitionStatePredictionMetrics]:
    if state_logits.shape != state_targets.shape:
        raise ValueError("state_logits and state_targets must have the same shape")
    if state_logits.ndim != 3:
        raise ValueError("state tensors must have shape [batch, steps, state_dim]")
    if sequence_mask is None:
        mask = torch.ones(state_logits.shape[:2], dtype=torch.bool, device=state_logits.device)
    else:
        if sequence_mask.shape != state_logits.shape[:2]:
            raise ValueError("sequence_mask must match state batch and steps")
        mask = sequence_mask.to(torch.bool)
    if not bool(mask.any().detach().cpu().item()):
        zero = state_logits.sum() * 0.0
        return zero, TransitionStatePredictionMetrics(
            samples=0,
            loss=0.0,
            mae=0.0,
            binary_accuracy=0.0,
        )
    valid_logits = state_logits[mask]
    valid_targets = state_targets[mask].float().clamp(min=0.0, max=1.0)
    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
    probs = torch.sigmoid(valid_logits)
    mae = torch.abs(probs - valid_targets).mean()
    binary_accuracy = ((probs >= 0.5) == (valid_targets >= 0.5)).float().mean()
    samples = int(mask.sum().detach().cpu().item())
    return loss, TransitionStatePredictionMetrics(
        samples=samples,
        loss=float(loss.detach().cpu().item()),
        mae=float(mae.detach().cpu().item()),
        binary_accuracy=float(binary_accuracy.detach().cpu().item()),
    )
