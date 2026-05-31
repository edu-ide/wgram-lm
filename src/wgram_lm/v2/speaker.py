from __future__ import annotations

import torch
from torch import nn

from wgram_lm.models.blt_components import BLTDLocalDecoder

from .config import WGRAMV2Config


class CausalTokenMaturationRefiner(nn.Module):
    """Refine token hidden states before discrete commitment through the LM head."""

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        self.config = config
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(config.local_heads),
                    dim_feedforward=d_model * 4,
                    dropout=float(config.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(config.token_maturation_layers))
            ]
        )
        self.update_norm = nn.LayerNorm(d_model)
        self.update_proj = nn.Linear(d_model, d_model)
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.confidence_head = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.update_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.update_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(self.config.token_maturation_gate_init))
        nn.init.normal_(self.confidence_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.confidence_head.bias)

    @staticmethod
    def _causal_mask(device: torch.device, *, length: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((int(length), int(length)), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        steps: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float | int | str]]:
        max_steps = int(steps)
        if max_steps <= 0:
            return hidden, [], {
                "token_maturation_mode": "disabled",
                "token_maturation_steps": 0,
                "token_maturation_gate_mean": 0.0,
                "token_maturation_delta_norm": 0.0,
                "token_maturation_confidence_mean": 0.0,
            }

        mask_values = self._causal_mask(
            hidden.device,
            length=int(hidden.shape[1]),
            dtype=hidden.dtype,
        )
        key_padding_mask_bool = ~attention_mask.bool()
        key_padding_mask = hidden.new_zeros(tuple(attention_mask.shape))
        key_padding_mask = key_padding_mask.masked_fill(key_padding_mask_bool, float("-inf"))
        valid_scale = attention_mask.to(hidden.dtype).unsqueeze(-1)
        aux_hiddens: list[torch.Tensor] = []
        gate_means: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        confidence_means: list[torch.Tensor] = []
        completed_steps = 0
        threshold = float(self.config.token_maturation_confidence_threshold)

        for _ in range(max_steps):
            proposal = hidden
            for layer in self.layers:
                proposal = layer(
                    proposal,
                    src_mask=mask_values,
                    src_key_padding_mask=key_padding_mask,
                )
            update = self.update_proj(self.update_norm(proposal))
            gate = torch.sigmoid(self.gate_proj(self.gate_norm(hidden + proposal)))
            delta = gate * update * valid_scale
            hidden = (hidden + delta) * valid_scale
            confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
            valid_confidence = confidence[attention_mask.bool()]
            gate_means.append(gate.detach().float().mean())
            delta_norms.append(delta.detach().float().norm(dim=-1)[attention_mask.bool()].mean())
            if bool(valid_confidence.numel()):
                confidence_means.append(valid_confidence.detach().float().mean())
            else:
                confidence_means.append(confidence.detach().float().mean())
            aux_hiddens.append(hidden)
            completed_steps += 1
            if threshold > 0.0 and confidence_means[-1].item() >= threshold:
                break

        gate_mean = torch.stack(gate_means).mean() if gate_means else hidden.new_zeros(())
        delta_norm = torch.stack(delta_norms).mean() if delta_norms else hidden.new_zeros(())
        confidence_mean = (
            torch.stack(confidence_means).mean() if confidence_means else hidden.new_zeros(())
        )
        return hidden, aux_hiddens, {
            "token_maturation_mode": "causal_latent_refinement_same_lm_head",
            "token_maturation_steps": int(completed_steps),
            "token_maturation_layers": int(self.config.token_maturation_layers),
            "token_maturation_gate_mean": float(gate_mean.detach().cpu().item()),
            "token_maturation_delta_norm": float(delta_norm.detach().cpu().item()),
            "token_maturation_confidence_mean": float(confidence_mean.detach().cpu().item()),
            "token_maturation_confidence_threshold": float(threshold),
        }


class AnswerMemoryAttractor(nn.Module):
    """Prompt-grounded latent answer memory injected before the shared LM head."""

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        self.config = config
        self.update_norm = nn.LayerNorm(d_model)
        self.update_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.update_gate = nn.Linear(d_model, d_model)
        self.memory_output_norm = nn.LayerNorm(d_model)
        self.plan_norm = nn.LayerNorm(d_model)
        self.plan_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.plan_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(config.local_heads),
                    dim_feedforward=d_model * 4,
                    dropout=float(config.dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(config.answer_memory_plan_layers))
            ]
        )
        self.plan_position_embed = nn.Embedding(int(config.max_response_position_embeddings), d_model)
        self.prompt_context_query_norm = nn.LayerNorm(d_model)
        self.prompt_context_key_norm = nn.LayerNorm(d_model)
        self.prompt_context_attn = nn.MultiheadAttention(
            d_model,
            int(config.local_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.prompt_context_gate_norm = nn.LayerNorm(d_model * 2)
        self.prompt_context_gate = nn.Linear(d_model * 2, d_model)
        self.prompt_context_proj = nn.Linear(d_model, d_model)
        self.plan_output_norm = nn.LayerNorm(d_model)
        self.inject_norm = nn.LayerNorm(d_model * 2)
        self.inject_gate = nn.Linear(d_model * 2, d_model)
        self.inject_proj = nn.Linear(d_model, d_model)
        self.commit_norm = nn.LayerNorm(d_model * 2)
        self.commit_gate = nn.Linear(d_model * 2, d_model)
        self.commit_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, float(self.config.answer_memory_update_gate_init))
        nn.init.zeros_(self.inject_gate.weight)
        nn.init.constant_(self.inject_gate.bias, float(self.config.answer_memory_injection_gate_init))
        nn.init.normal_(self.inject_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.inject_proj.bias)
        nn.init.zeros_(self.commit_gate.weight)
        nn.init.constant_(self.commit_gate.bias, float(self.config.answer_memory_commitment_gate_init))
        nn.init.eye_(self.commit_proj.weight)
        nn.init.zeros_(self.commit_proj.bias)
        nn.init.normal_(self.plan_position_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prompt_context_gate.weight)
        nn.init.constant_(
            self.prompt_context_gate.bias,
            float(self.config.answer_memory_prompt_context_gate_init),
        )
        nn.init.normal_(self.prompt_context_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prompt_context_proj.bias)
        last = self.plan_proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _response_start_mask(response_mask: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(response_mask)
        previous[:, 1:] = response_mask[:, :-1]
        return response_mask & ~previous

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        response_prediction_mask: torch.Tensor | None,
        *,
        injection_scale: float,
        commitment_scale: float,
        prompt_context_scale: float,
        speaker_norm: nn.Module | None = None,
        lm_head: nn.Module | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, float | int | str],
    ]:
        empty_metrics = {
            "answer_memory_mode": "disabled",
            "answer_memory_rows": 0,
            "answer_memory_steps": 0,
            "answer_memory_plan_tokens": 0,
            "answer_memory_plan_layers": 0,
            "answer_memory_prompt_context_mode": "disabled",
            "answer_memory_prompt_context_tokens_mean": 0.0,
            "answer_memory_prompt_context_gate_mean": 0.0,
            "answer_memory_prompt_context_scale": 0.0,
            "answer_memory_prompt_context_delta_norm": 0.0,
            "answer_memory_plan_confidence_mean": 0.0,
            "answer_memory_plan_top1_confidence_mean": 0.0,
            "answer_memory_plan_topk_mass_mean": 0.0,
            "answer_memory_plan_entropy_complement_mean": 0.0,
            "answer_memory_injection_confidence_scale_mean": 0.0,
            "answer_memory_confidence_mode": str(self.config.answer_memory_confidence_mode),
            "answer_memory_confidence_topk": int(self.config.answer_memory_confidence_topk),
            "answer_memory_confidence_floor": float(self.config.answer_memory_confidence_floor),
            "answer_memory_confidence_gate": "disabled",
            "answer_memory_update_gate_mean": 0.0,
            "answer_memory_injection_gate_mean": 0.0,
            "answer_memory_injection_scale": 0.0,
            "answer_memory_injection_positions": 0,
            "answer_memory_injection_context": "disabled",
            "answer_memory_commitment_mode": "disabled",
            "answer_memory_commitment_positions": 0,
            "answer_memory_commitment_gate_mean": 0.0,
            "answer_memory_commitment_scale": 0.0,
            "answer_memory_commitment_confidence_gate": "disabled",
            "answer_memory_commitment_confidence_scale_mean": 0.0,
            "answer_memory_commitment_delta_norm": 0.0,
            "answer_memory_delta_norm": 0.0,
            "answer_memory_plan_delta_norm": 0.0,
        }
        if (
            not bool(self.config.answer_memory_enabled)
            or int(self.config.answer_memory_steps) <= 0
            or response_prediction_mask is None
        ):
            return hidden, None, None, None, empty_metrics
        response_mask = response_prediction_mask.bool() & attention_mask.bool()
        start_mask = self._response_start_mask(response_mask)
        if not bool(start_mask.any()):
            return hidden, None, None, None, empty_metrics

        batch, _, d_model = hidden.shape
        start_positions = start_mask.float().argmax(dim=1).to(torch.long)
        row_has_start = start_mask.any(dim=1)
        memory = hidden[torch.arange(batch, device=hidden.device), start_positions]
        memory = memory * row_has_start.to(hidden.dtype).unsqueeze(-1)
        update_gate_means: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        for _ in range(int(self.config.answer_memory_steps)):
            update = self.update_proj(self.update_norm(memory))
            gate = torch.sigmoid(self.update_gate(memory))
            delta = gate * update
            memory = memory + delta
            valid_delta = delta[row_has_start]
            update_gate_means.append(gate[row_has_start].detach().float().mean())
            if bool(valid_delta.numel()):
                delta_norms.append(valid_delta.detach().float().norm(dim=-1).mean())
            else:
                delta_norms.append(delta.detach().float().mean())

        memory = self.memory_output_norm(memory)
        plan_tokens = min(
            int(self.config.answer_memory_plan_tokens),
            int(self.config.max_response_position_embeddings),
        )
        plan_positions = torch.arange(plan_tokens, dtype=torch.long, device=hidden.device)
        plan_seed = memory.unsqueeze(1) + self.plan_position_embed(plan_positions).unsqueeze(0).to(hidden.dtype)
        plan_delta = self.plan_proj(self.plan_norm(plan_seed))
        plan_states = plan_seed + plan_delta
        prompt_context_mode = "disabled"
        prompt_context_tokens_mean = hidden.new_zeros(())
        prompt_context_gate_mean = hidden.new_zeros(())
        active_prompt_context_scale = max(0.0, float(prompt_context_scale))
        prompt_context_delta_norm = hidden.new_zeros(())
        if (
            bool(self.config.answer_memory_prompt_context_enabled)
            and active_prompt_context_scale > 0.0
        ):
            seq_positions = torch.arange(hidden.shape[1], dtype=torch.long, device=hidden.device).unsqueeze(0)
            prompt_context_mask = attention_mask.bool() & (
                seq_positions <= start_positions.unsqueeze(1)
            )
            fallback_context_mask = attention_mask.bool() & (seq_positions == 0)
            prompt_context_mask = torch.where(
                row_has_start.unsqueeze(1),
                prompt_context_mask,
                fallback_context_mask,
            )
            if bool(prompt_context_mask.any()):
                context_key_padding_mask = ~prompt_context_mask
                context_hidden = self.prompt_context_key_norm(hidden)
                attn_out, _ = self.prompt_context_attn(
                    self.prompt_context_query_norm(plan_states),
                    context_hidden,
                    context_hidden,
                    key_padding_mask=context_key_padding_mask,
                    need_weights=False,
                )
                gate_input = torch.cat([plan_states, attn_out], dim=-1)
                prompt_gate = torch.sigmoid(
                    self.prompt_context_gate(self.prompt_context_gate_norm(gate_input))
                ).to(hidden.dtype)
                prompt_delta = (
                    active_prompt_context_scale * prompt_gate * self.prompt_context_proj(attn_out)
                )
                plan_states = plan_states + prompt_delta
                prompt_context_mode = "same_body_causal_prompt_context_read"
                valid_prompt_rows = row_has_start
                prompt_context_tokens_mean = prompt_context_mask.detach().float().sum(dim=1)[
                    valid_prompt_rows
                ].mean()
                prompt_context_gate_mean = prompt_gate.detach().float().mean()
                prompt_context_delta_norm = prompt_delta.detach().float().norm(dim=-1).mean()
        for layer in self.plan_layers:
            plan_states = layer(plan_states)
        plan_states = self.plan_output_norm(plan_states)
        plan_logits = None
        plan_confidence = hidden.new_ones((batch,))
        plan_top1_confidence = hidden.new_zeros((batch,))
        plan_topk_mass = hidden.new_zeros((batch,))
        plan_entropy_complement = hidden.new_zeros((batch,))
        confidence_scale = hidden.new_ones((batch,))
        confidence_gate_mode = "disabled"
        if speaker_norm is not None and lm_head is not None:
            plan_logits = lm_head(speaker_norm(plan_states))
            with torch.no_grad():
                plan_probs = plan_logits.float().softmax(dim=-1)
                vocab_size = int(plan_probs.shape[-1])
                topk = min(max(1, int(self.config.answer_memory_confidence_topk)), vocab_size)
                plan_top1_confidence = plan_probs.max(dim=-1).values.mean(dim=1).to(hidden.dtype)
                plan_topk_mass = torch.topk(plan_probs, k=topk, dim=-1).values.sum(dim=-1).mean(dim=1).to(hidden.dtype)
                entropy = -(plan_probs * plan_probs.clamp_min(1.0e-12).log()).sum(dim=-1)
                entropy_norm = torch.log(plan_probs.new_tensor(float(max(2, vocab_size))))
                plan_entropy_complement = (1.0 - entropy / entropy_norm).clamp(min=0.0, max=1.0)
                plan_entropy_complement = plan_entropy_complement.mean(dim=1).to(hidden.dtype)
                confidence_mode = str(self.config.answer_memory_confidence_mode)
                if confidence_mode == "top1_probability":
                    plan_confidence = plan_top1_confidence
                elif confidence_mode == "topk_mass":
                    plan_confidence = plan_topk_mass
                elif confidence_mode == "entropy_complement":
                    plan_confidence = plan_entropy_complement
                elif confidence_mode == "hybrid_topk_entropy":
                    plan_confidence = torch.sqrt(
                        (plan_topk_mass * plan_entropy_complement).clamp(min=0.0)
                    )
                else:
                    plan_confidence = plan_topk_mass
            if bool(self.config.answer_memory_confidence_gate_enabled):
                floor = max(0.0, float(self.config.answer_memory_confidence_floor))
                denom = max(1.0e-6, 1.0 - floor)
                confidence_scale = ((plan_confidence - floor) / denom).clamp(min=0.0, max=1.0)
                confidence_gate_mode = f"same_lm_head_plan_confidence_{self.config.answer_memory_confidence_mode}"
        raw_response_positions = response_mask.to(torch.long).cumsum(dim=1) - 1
        commit_mask = response_mask & (raw_response_positions >= 0) & (raw_response_positions < plan_tokens)
        response_positions = raw_response_positions.clamp(min=0, max=plan_tokens - 1)
        plan_context = torch.gather(
            plan_states,
            dim=1,
            index=response_positions.unsqueeze(-1).expand(batch, hidden.shape[1], d_model),
        )
        inject_input = torch.cat([hidden, plan_context], dim=-1)
        inject_gate = torch.sigmoid(self.inject_gate(self.inject_norm(inject_input))).to(hidden.dtype)
        memory_update = self.inject_proj(plan_context)
        plan_prefix_scale = commit_mask.to(hidden.dtype).unsqueeze(-1)
        active_scale = max(0.0, float(injection_scale))
        effective_scale = active_scale * confidence_scale.view(batch, 1, 1)
        hidden = hidden + effective_scale * inject_gate * memory_update * plan_prefix_scale
        active_commitment_scale = max(0.0, float(commitment_scale))
        commitment_mode = "disabled"
        commitment_positions = int(commit_mask.detach().sum().cpu().item())
        injection_positions = int(commit_mask.detach().sum().cpu().item())
        commitment_gate_mean = hidden.new_zeros(())
        commitment_confidence_scale_mean = hidden.new_zeros(())
        commitment_delta_norm = hidden.new_zeros(())
        commitment_confidence_gate_mode = "disabled"
        if (
            bool(self.config.answer_memory_commitment_enabled)
            and active_commitment_scale > 0.0
            and bool(commit_mask.any())
        ):
            commit_input = torch.cat([hidden, plan_context], dim=-1)
            commit_gate = torch.sigmoid(self.commit_gate(self.commit_norm(commit_input))).to(hidden.dtype)
            commit_target = self.commit_proj(plan_context)
            commit_delta = commit_gate * (commit_target - hidden)
            if bool(self.config.answer_memory_commitment_confidence_gate_enabled):
                commit_confidence_scale = confidence_scale
                commitment_confidence_gate_mode = (
                    f"same_lm_head_plan_confidence_{self.config.answer_memory_confidence_mode}"
                )
            else:
                commit_confidence_scale = hidden.new_ones((batch,))
            commit_scale = active_commitment_scale * commit_confidence_scale.view(batch, 1, 1)
            hidden = hidden + commit_scale * commit_delta * commit_mask.to(hidden.dtype).unsqueeze(-1)
            valid_commit_confidence = commit_confidence_scale[row_has_start]
            commitment_confidence_scale_mean = (
                valid_commit_confidence.detach().float().mean()
                if bool(valid_commit_confidence.numel())
                else hidden.new_zeros(())
            )
            commitment_gate_mean = commit_gate[commit_mask].detach().float().mean()
            commitment_delta_norm = commit_delta[commit_mask].detach().float().norm(dim=-1).mean()
            commitment_mode = "same_lm_head_answer_prefix_state_commitment"
        if bool(commit_mask.any()):
            injection_gate_mean = inject_gate[commit_mask].detach().float().mean()
        else:
            injection_gate_mean = hidden.new_zeros(())
        update_gate_mean = (
            torch.stack(update_gate_means).mean() if update_gate_means else hidden.new_zeros(())
        )
        delta_norm = torch.stack(delta_norms).mean() if delta_norms else hidden.new_zeros(())
        valid_plan_delta = plan_delta[row_has_start]
        if bool(valid_plan_delta.numel()):
            plan_delta_norm = valid_plan_delta.detach().float().norm(dim=-1).mean()
        else:
            plan_delta_norm = plan_delta.detach().float().mean()
        valid_confidence = plan_confidence[row_has_start]
        valid_top1_confidence = plan_top1_confidence[row_has_start]
        valid_topk_mass = plan_topk_mass[row_has_start]
        valid_entropy_complement = plan_entropy_complement[row_has_start]
        valid_confidence_scale = confidence_scale[row_has_start]
        confidence_mean = (
            valid_confidence.detach().float().mean() if bool(valid_confidence.numel()) else hidden.new_zeros(())
        )
        top1_confidence_mean = (
            valid_top1_confidence.detach().float().mean()
            if bool(valid_top1_confidence.numel())
            else hidden.new_zeros(())
        )
        topk_mass_mean = (
            valid_topk_mass.detach().float().mean()
            if bool(valid_topk_mass.numel())
            else hidden.new_zeros(())
        )
        entropy_complement_mean = (
            valid_entropy_complement.detach().float().mean()
            if bool(valid_entropy_complement.numel())
            else hidden.new_zeros(())
        )
        confidence_scale_mean = (
            valid_confidence_scale.detach().float().mean()
            if bool(valid_confidence_scale.numel())
            else hidden.new_zeros(())
        )
        return hidden, memory, plan_states, plan_logits, {
            "answer_memory_mode": "prompt_grounded_prefix_plan_same_lm_head",
            "answer_memory_rows": int(row_has_start.detach().sum().cpu().item()),
            "answer_memory_steps": int(self.config.answer_memory_steps),
            "answer_memory_plan_tokens": int(plan_tokens),
            "answer_memory_plan_layers": int(self.config.answer_memory_plan_layers),
            "answer_memory_prompt_context_mode": str(prompt_context_mode),
            "answer_memory_prompt_context_tokens_mean": float(
                prompt_context_tokens_mean.detach().cpu().item()
            ),
            "answer_memory_prompt_context_gate_mean": float(
                prompt_context_gate_mean.detach().cpu().item()
            ),
            "answer_memory_prompt_context_scale": float(active_prompt_context_scale),
            "answer_memory_prompt_context_delta_norm": float(
                prompt_context_delta_norm.detach().cpu().item()
            ),
            "answer_memory_plan_confidence_mean": float(confidence_mean.detach().cpu().item()),
            "answer_memory_plan_top1_confidence_mean": float(top1_confidence_mean.detach().cpu().item()),
            "answer_memory_plan_topk_mass_mean": float(topk_mass_mean.detach().cpu().item()),
            "answer_memory_plan_entropy_complement_mean": float(entropy_complement_mean.detach().cpu().item()),
            "answer_memory_injection_confidence_scale_mean": float(confidence_scale_mean.detach().cpu().item()),
            "answer_memory_confidence_mode": str(self.config.answer_memory_confidence_mode),
            "answer_memory_confidence_topk": int(self.config.answer_memory_confidence_topk),
            "answer_memory_confidence_floor": float(self.config.answer_memory_confidence_floor),
            "answer_memory_confidence_gate": str(confidence_gate_mode),
            "answer_memory_update_gate_mean": float(update_gate_mean.detach().cpu().item()),
            "answer_memory_injection_gate_mean": float(injection_gate_mean.detach().cpu().item()),
            "answer_memory_injection_scale": float(active_scale),
            "answer_memory_injection_positions": int(injection_positions),
            "answer_memory_injection_context": "answer_prefix_only_no_tail_clamp",
            "answer_memory_commitment_mode": str(commitment_mode),
            "answer_memory_commitment_positions": int(commitment_positions),
            "answer_memory_commitment_gate_mean": float(commitment_gate_mean.detach().cpu().item()),
            "answer_memory_commitment_scale": float(active_commitment_scale),
            "answer_memory_commitment_confidence_gate": str(commitment_confidence_gate_mode),
            "answer_memory_commitment_confidence_scale_mean": float(
                commitment_confidence_scale_mean.detach().cpu().item()
            ),
            "answer_memory_commitment_delta_norm": float(commitment_delta_norm.detach().cpu().item()),
            "answer_memory_delta_norm": float(delta_norm.detach().cpu().item()),
            "answer_memory_plan_delta_norm": float(plan_delta_norm.detach().cpu().item()),
        }


class CausalByteSpeaker(nn.Module):
    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        self.config = config
        d_model = int(config.d_model)
        self.position_embed = nn.Embedding(int(config.max_position_embeddings), d_model)
        self.response_phase_embed = nn.Embedding(2, d_model)
        self.response_position_embed = nn.Embedding(int(config.max_response_position_embeddings), d_model)
        self.latent_bridge = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.adaptive_latent_bridge_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )
        self.byte_gate_logit = nn.Parameter(torch.tensor(float(config.byte_residual_gate_init)))
        self.latent_gate_logit = nn.Parameter(torch.tensor(float(config.latent_residual_gate_init)))
        self.decoder = BLTDLocalDecoder(
            d_model,
            int(config.vocab_size),
            patch_size=int(config.patch_size),
            n_heads=int(config.local_heads),
            n_layers=int(config.local_layers),
            dropout=float(config.dropout),
            causal=True,
            cross_attention=False,
        )
        self.head = self.decoder.head
        self.answer_memory = AnswerMemoryAttractor(config)
        self.last_answer_memory_plan_logits: torch.Tensor | None = None
        self.maturation = CausalTokenMaturationRefiner(config)
        self.last_maturation_logits: list[torch.Tensor] = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.response_phase_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.response_position_embed.weight, mean=0.0, std=0.02)
        gate_linear = self.adaptive_latent_bridge_gate[1]
        assert isinstance(gate_linear, nn.Linear)
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, float(self.config.adaptive_latent_bridge_gate_init))

    def tie_output_weight(self, weight: nn.Parameter) -> None:
        self.decoder.head.weight = weight
        self.head = self.decoder.head

    @property
    def norm(self) -> nn.Module:
        return self.decoder.norm

    def _stabilize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        clip_value = float(self.config.stability_activation_clip_value)
        if clip_value <= 0.0:
            return torch.nan_to_num(hidden, nan=0.0)
        return torch.nan_to_num(
            hidden,
            nan=0.0,
            posinf=clip_value,
            neginf=-clip_value,
        ).clamp(min=-clip_value, max=clip_value)

    def forward(
        self,
        byte_embeddings: torch.Tensor,
        dechunked_latent: torch.Tensor,
        attention_mask: torch.Tensor,
        response_prediction_mask: torch.Tensor | None = None,
        answer_memory_injection_scale: float | None = None,
        answer_memory_commitment_scale: float | None = None,
        answer_memory_prompt_context_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
        bridged = dechunked_latent + self.latent_bridge(dechunked_latent)
        byte_gate = torch.sigmoid(self.byte_gate_logit).to(byte_embeddings.dtype)
        latent_gate = torch.sigmoid(self.latent_gate_logit).to(byte_embeddings.dtype)
        seq_len = int(byte_embeddings.shape[1])
        position_ids = torch.arange(seq_len, dtype=torch.long, device=byte_embeddings.device)
        position_ids = position_ids.clamp(max=int(self.config.max_position_embeddings) - 1)
        position_embeddings = self.position_embed(position_ids).unsqueeze(0).to(byte_embeddings.dtype)
        adaptive_gate_mean = 1.0
        if bool(self.config.adaptive_latent_bridge_enabled):
            gate_input = torch.cat([byte_embeddings, bridged], dim=-1)
            adaptive_gate = torch.sigmoid(self.adaptive_latent_bridge_gate(gate_input)).to(byte_embeddings.dtype)
            adaptive_gate_mean = float(adaptive_gate.detach().float().mean().cpu().item())
            latent_contribution = latent_gate * adaptive_gate * bridged
        else:
            latent_contribution = latent_gate * bridged
        token_hidden = byte_gate * byte_embeddings + latent_contribution + position_embeddings
        response_phase_mean = 0.0
        if bool(self.config.use_response_phase_embeddings) and response_prediction_mask is not None:
            response_mask = response_prediction_mask.bool() & attention_mask.bool()
            phase_ids = response_mask.to(torch.long)
            phase_embeddings = self.response_phase_embed(phase_ids).to(byte_embeddings.dtype)
            response_positions = response_mask.to(torch.long).cumsum(dim=1) - 1
            response_positions = response_positions.clamp(min=0, max=int(self.config.max_response_position_embeddings) - 1)
            response_pos_embeddings = self.response_position_embed(response_positions).to(byte_embeddings.dtype)
            token_hidden = token_hidden + phase_embeddings + response_pos_embeddings * response_mask.unsqueeze(-1).to(byte_embeddings.dtype)
            response_phase_mean = float(response_mask.detach().float().mean().cpu().item())
        token_hidden = self._stabilize_hidden(token_hidden * attention_mask.to(token_hidden.dtype).unsqueeze(-1))
        hidden = self._stabilize_hidden(self.decoder.forward_hidden(token_hidden))
        (
            hidden,
            answer_memory,
            answer_memory_plan,
            answer_memory_plan_logits,
            answer_memory_metrics,
        ) = self.answer_memory(
            hidden,
            attention_mask,
            response_prediction_mask,
            injection_scale=(
                float(self.config.answer_memory_default_injection_scale)
                if answer_memory_injection_scale is None
                else float(answer_memory_injection_scale)
            ),
            commitment_scale=(
                float(self.config.answer_memory_commitment_scale)
                if answer_memory_commitment_scale is None
                else float(answer_memory_commitment_scale)
            ),
            prompt_context_scale=(
                float(self.config.answer_memory_prompt_context_default_scale)
                if answer_memory_prompt_context_scale is None
                else float(answer_memory_prompt_context_scale)
            ),
            speaker_norm=self.norm,
            lm_head=self.head,
        )
        hidden = self._stabilize_hidden(hidden)
        if (
            answer_memory is not None
            and answer_memory_plan is not None
            and answer_memory_plan_logits is not None
            and float(self.config.answer_memory_aux_loss_weight) > 0.0
        ):
            self.last_answer_memory_plan_logits = answer_memory_plan_logits
        else:
            self.last_answer_memory_plan_logits = None
        hidden, maturation_hiddens, maturation_metrics = self.maturation(
            hidden,
            attention_mask,
            steps=int(self.config.token_maturation_steps),
        )
        if float(self.config.token_maturation_aux_loss_weight) > 0.0:
            self.last_maturation_logits = [
                self.head(self.norm(self._stabilize_hidden(aux_hidden)))
                for aux_hidden in maturation_hiddens
            ]
        else:
            self.last_maturation_logits = []
        hidden = self._stabilize_hidden(hidden)
        logits = self.head(self.norm(hidden))
        return logits, hidden, {
            "answer_path": "hnet_causal_speaker_same_lm_head",
            "answer_transition_path": "prompt_context_answer_memory_prefix_plan_commitment_then_causal_token_maturation_before_same_lm_head",
            **answer_memory_metrics,
            **maturation_metrics,
            "speaker_position_encoding": "learned_absolute",
            "speaker_response_phase_encoding": (
                "learned_segment_relative" if bool(self.config.use_response_phase_embeddings) else "disabled"
            ),
            "speaker_response_phase_fraction": float(response_phase_mean),
            "speaker_input_output_embeddings_tied": bool(self.config.tie_input_output_embeddings),
            "speaker_max_position_embeddings": int(self.config.max_position_embeddings),
            "speaker_adaptive_latent_bridge": bool(self.config.adaptive_latent_bridge_enabled),
            "speaker_adaptive_latent_bridge_gate_mean": float(adaptive_gate_mean),
            "byte_residual_gate": float(byte_gate.detach().float().cpu().item()),
            "latent_residual_gate": float(latent_gate.detach().float().cpu().item()),
        }
